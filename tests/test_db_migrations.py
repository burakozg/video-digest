"""The forward-only schema-migration hook in db.py — the mechanism (with a
stand-in migration) and migration 2 (`videos.written_at`) for real.

Migration 2 matters more than most: it backfills `written_at`, the ordering
key `GET /videos` pages on. A value that moves after a client has paged past
it puts the row behind that client's cursor, where it is never seen again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from video_digest import db as db_module
from video_digest.db import SCHEMA_VERSION, connect


def test_fresh_db_is_stamped_at_the_current_version(tmp_path: Path) -> None:
    conn = connect(tmp_path / "s.sqlite")
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    assert row["value"] == str(SCHEMA_VERSION)


def test_apply_migrations_runs_steps_newer_than_the_stored_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = connect(tmp_path / "s.sqlite")
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    conn.commit()

    monkeypatch.setattr(
        db_module,
        "_MIGRATIONS",
        {1: "CREATE TABLE IF NOT EXISTS _probe (x INTEGER);"},
    )
    db_module._apply_migrations(conn)

    assert (
        conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()["value"]
        == str(SCHEMA_VERSION)
    )
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "_probe" in tables


def test_apply_migrations_is_a_noop_when_already_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = connect(tmp_path / "s.sqlite")
    ran: list[int] = []
    monkeypatch.setattr(db_module, "_MIGRATIONS", {1: "CREATE TABLE _never (x);"})
    # Already stamped at SCHEMA_VERSION by connect(), so the range is empty.
    db_module._apply_migrations(conn)
    assert ran == []
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "_never" not in tables


#: The `videos` table as it stood at schema version 1 — before `written_at`.
#: A frozen copy on purpose: a migration test that builds its "old" database
#: from the *current* SCHEMA tests nothing, because the column it is meant to
#: add is already there. Only the columns migration 2 touches, plus the
#: NOT NULLs that make an INSERT possible.
_V1_VIDEOS = """
CREATE TABLE videos (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  url TEXT NOT NULL,
  canonical_url TEXT NOT NULL,
  stage_write TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def _v1_db(tmp_path: Path):
    """A database at version 1: current schema, `videos` rolled back to v1."""
    conn = connect(tmp_path / "s.sqlite")
    conn.execute("DROP TABLE videos")
    conn.executescript(_V1_VIDEOS)
    conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    conn.executemany(
        "INSERT INTO videos (id, video_id, url, canonical_url, stage_write, "
        "created_at, updated_at) VALUES (?, ?, 'u', 'u', ?, ?, ?)",
        [
            ("youtube:done", "done", "done", "2026-01-01T00:00:00+00:00", "2026-06-06T00:00:00+00:00"),
            ("youtube:pending", "pending", "pending", "2026-02-02T00:00:00+00:00", "2026-06-06T00:00:00+00:00"),
        ],
    )
    conn.commit()
    return conn


def test_migration_2_adds_written_at_and_backfills_from_created_at(tmp_path: Path) -> None:
    conn = _v1_db(tmp_path)
    assert not db_module._has_column(conn, "videos", "written_at")

    db_module._apply_migrations(conn)

    assert db_module._has_column(conn, "videos", "written_at")
    rows = {
        r["id"]: r["written_at"]
        for r in conn.execute("SELECT id, written_at FROM videos")
    }
    # Backfilled from created_at (never mutated), NOT updated_at.
    assert rows["youtube:done"] == "2026-01-01T00:00:00+00:00"
    # Never written -> no ordering key, and the export filters it out.
    assert rows["youtube:pending"] is None


def test_migration_2_is_idempotent(tmp_path: Path) -> None:
    """A crash between the ALTER and the version stamp must leave a database
    that can still be opened. Re-running must not raise, and must not move a
    timestamp a reader may already have seen."""
    conn = _v1_db(tmp_path)
    db_module._apply_migrations(conn)

    # Simulate the stamp never landing, and a value having since moved on.
    conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    conn.execute("UPDATE videos SET updated_at = '2099-01-01T00:00:00+00:00'")
    conn.commit()

    db_module._apply_migrations(conn)  # must not raise "duplicate column name"

    written = conn.execute(
        "SELECT written_at FROM videos WHERE id = 'youtube:done'"
    ).fetchone()["written_at"]
    assert written == "2026-01-01T00:00:00+00:00"


def test_fresh_db_has_written_at_without_running_the_migration(tmp_path: Path) -> None:
    """SCHEMA already carries the column, so a new deployment runs no steps."""
    conn = connect(tmp_path / "fresh.sqlite")
    assert db_module._has_column(conn, "videos", "written_at")
