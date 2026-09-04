"""The SQLite app state (plan §1.3 — not CouchDB; podcast-digest's CouchDB
state model in §6 of the design maps to tables directly, with the same field
names so the design doc stays a legible reference).

One file, WAL mode, matching vault-ask/db.py — a join beats a second
container on a NAS with ~900 MB free.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

#: Stages are checkpointed independently (design §5): a failure at S4 must
#: never re-download media, so each has its own status/timestamp/error rather
#: than one coarse job status.
STAGES: tuple[str, ...] = ("resolve", "metadata", "transcript", "normalize", "summarize", "write")

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
  -- "<adapter>:<video_id>" (design §6's `video:youtube:{video_id}`, without
  -- the redundant leading "video:" — the table name already says that).
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  -- The source *adapter* (design §1.1 — youtube first, others later without
  -- renaming the service). Not to be confused with `origin` below.
  adapter TEXT NOT NULL DEFAULT 'youtube',
  -- Which ingress path created this row: 'manual' | 'watchlist' | 'inbox'.
  -- The `source` field the design's §4 hooks call for, distinguishing manual
  -- from automatic so the watchlist poller is a later addition, not a refactor.
  origin TEXT NOT NULL DEFAULT 'manual',

  url TEXT NOT NULL,
  canonical_url TEXT NOT NULL,

  metadata TEXT,               -- JSON, S1 output (title, channel, duration, ...)

  stage_resolve TEXT NOT NULL DEFAULT 'pending',
  stage_metadata TEXT NOT NULL DEFAULT 'pending',
  stage_transcript TEXT NOT NULL DEFAULT 'pending',
  stage_normalize TEXT NOT NULL DEFAULT 'pending',
  stage_summarize TEXT NOT NULL DEFAULT 'pending',
  stage_write TEXT NOT NULL DEFAULT 'pending',
  stage_errors TEXT,           -- JSON {stage: last error message}

  transcript_tier TEXT,                          -- T0 | T1 | T2
  transcript_tier_degraded INTEGER NOT NULL DEFAULT 0,
  transcript_json TEXT,          -- JSON Transcript, staged for S4/S5 after S2/S3
  asr_model TEXT,                 -- set when transcript_tier == T2
  transcript_path TEXT,        -- vault path under transcripts_dir
  note_path TEXT,               -- vault path under notes_dir

  -- When stage_write FIRST reached 'done'. Set once and never moved: it is
  -- the ordering key and cursor `GET /videos` exports by, so a consumer that
  -- tracks read/unread state against it keeps that state across a re-export.
  -- updated_at cannot serve — every stage transition and every
  -- /videos/{id}/rewrite bumps it, which would reorder the export and make
  -- already-seen items look new.
  written_at TEXT,

  digest TEXT,                  -- JSON, last VideoDigest — rewrite without re-summarising

  attempts TEXT NOT NULL DEFAULT '{}',   -- JSON {stage: count}
  last_error TEXT,

  -- Set when parked at transcript=pending because the ASR worker (the
  -- speaches container in ~/projects/asr-server) was unreachable. Cleared on
  -- success. Age past transcription.asr_stale_hours is an alert, not a retry
  -- (design §5 S2 — "the Mac has been shut for two days, not a broken
  -- pipeline").
  asr_queued_at TEXT,

  force_asr INTEGER NOT NULL DEFAULT 0,

  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_video_id ON videos(video_id, adapter);
CREATE INDEX IF NOT EXISTS idx_videos_asr_queued
  ON videos(asr_queued_at) WHERE asr_queued_at IS NOT NULL;
-- `GET /videos` orders and pages by exactly this pair (id breaks a tie, since
-- written_at is not unique — see its comment above). IF NOT EXISTS, and this
-- whole script runs on every connect, so an existing database gains it
-- without a migration step.
CREATE INDEX IF NOT EXISTS idx_videos_written
  ON videos(written_at, id) WHERE written_at IS NOT NULL;

-- Filename pinning (obsidian-vault-writer skill §2 "the filename is part of
-- the contract"): derived once, stored, never recomputed. A title correction
-- must not move the file and strand every existing wikilink.
CREATE TABLE IF NOT EXISTS note_names (
  video_id TEXT PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
  filename TEXT NOT NULL UNIQUE      -- relative to vault.notes_dir, no extension
);

-- Same pinning rule, for topic pages this app creates in 99 topics/. Keyed by
-- canonical() (video_digest/sanitize.py, ported verbatim from podcast-digest),
-- the same key podcast-digest and clippings-topics already use, so the three
-- writers agree on identity without talking to each other.
CREATE TABLE IF NOT EXISTS topic_note_names (
  topic_key TEXT PRIMARY KEY,
  filename TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id),
  force INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_video ON jobs(video_id);

-- Counts by stage, tier distribution, ASR minutes, token spend (design §8
-- /metrics — "if T2 is firing on 80% of videos, either the heuristic is
-- wrong or the ASR bill needs a policy change").
CREATE TABLE IF NOT EXISTS metric_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,           -- 'transcript_tier' | 'asr_minutes' | 'llm_tokens' | ...
  video_id TEXT,
  value REAL NOT NULL DEFAULT 1,
  meta TEXT,                    -- JSON
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metric_events_kind ON metric_events(kind, created_at);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
-- keys: schema_version, installed_yt_dlp_version, last_cleanup_run
"""

SCHEMA_VERSION = 2


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    # PRAGMA table_info is the only way to ask: SQLite has no
    # `ADD COLUMN IF NOT EXISTS`, and a duplicate ADD is a hard error.
    return any(row["name"] == column for row in conn.execute(f"PRAGMA table_info({table})"))


def _migrate_2_written_at(conn: sqlite3.Connection) -> None:
    """`videos.written_at` — when the note was FIRST written (see SCHEMA).

    Backfilled from `created_at`, not `updated_at`: `created_at` is provably
    never mutated (pipeline/resolve.py's upserts set `updated_at` and leave it
    alone), so this is deterministic and re-running it cannot move a value a
    client has already paged past. Enqueue-to-note is minutes for a normal
    video, so it is a faithful proxy; a video that parked for ASR is ordered a
    little early, which is invisible to a first full import.

    Ties are expected here — a backfill stamps a whole batch from timestamps
    that may share a second — which is why `GET /videos` pages on the
    `(written_at, id)` pair rather than the timestamp alone.
    """
    if not _has_column(conn, "videos", "written_at"):
        conn.execute("ALTER TABLE videos ADD COLUMN written_at TEXT")
    conn.execute(
        "UPDATE videos SET written_at = created_at "
        "WHERE written_at IS NULL AND stage_write = 'done'"
    )


#: Forward-only migrations, keyed by the schema version each one PRODUCES.
#: `SCHEMA` above always creates a fresh database at `SCHEMA_VERSION`, so a
#: new deployment runs none of these; an existing database runs every step
#: whose key is greater than its stored `meta.schema_version`. Each step must
#: be idempotent so a half-applied upgrade is safe to re-run — a crash between
#: the step and the version stamp below would otherwise leave a database that
#: can never be opened again. A plain SQL string is enough when every
#: statement is (`CREATE ... IF NOT EXISTS`); anything that is not — `ALTER
#: TABLE ADD COLUMN`, which SQLite rejects on a second run — is a callable
#: that checks first.
_MIGRATIONS: dict[int, str | Callable[[sqlite3.Connection], None]] = {
    2: _migrate_2_written_at,
}


def _apply_migrations(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    current = int(row["value"]) if row is not None else 0
    for version in range(current + 1, SCHEMA_VERSION + 1):
        step = _MIGRATIONS.get(version)
        if callable(step):
            step(conn)
        elif step:
            conn.executescript(step)
        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(version),))
    conn.commit()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: uvicorn runs one event loop on one thread, so
    # this connection is never touched concurrently in production, but ASGI
    # test transports (FastAPI's TestClient) run the app in a worker thread
    # different from the one that called connect() — sqlite3's default
    # thread-affinity check rejects that on the very first query. sqlite3's
    # own global lock still serialises access, so nothing about WAL's
    # reader/writer concurrency below is weakened by this.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # The API server and the scheduler's background jobs both hit this file
    # from the same process but different asyncio tasks/threads; WAL lets a
    # reader proceed without waiting on an in-progress writer (vault_ask/db.py
    # carries the same reasoning against the same class of NAS deployment).
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    _apply_migrations(conn)
    return conn
