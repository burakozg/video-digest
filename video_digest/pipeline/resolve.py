"""S0 (resolve & deduplicate) and S1 (metadata) — design §5.

`enqueue()` is the seam every ingress path must call (design §4): HTTP now,
the vault inbox watcher and the iOS Shortcut in M6, and a watchlist poller
whenever that lands — never construct a job inline in a handler. It runs S0
and S1 synchronously (no background queue yet; later stages that need one —
ASR's `pending_asr` drain job — get it in M5) and is itself pure of I/O
except through the two injected callables, so the dedupe/playlist/rejection
logic is testable without touching the network.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ..config import AcquisitionConfig
from ..logging_setup import get_logger
from ..sources.youtube import (
    VideoMetadata,
    VideoRejected,
    canonical_url,
    expand_playlist,
    fetch_metadata,
    resolve_url,
)
from ..utils import iso_now, new_job_id

log = get_logger(__name__)

Adapter = Literal["youtube"]
Origin = Literal["manual", "watchlist", "inbox"]

FetchMetadataFn = Callable[[str], VideoMetadata]
ExpandPlaylistFn = Callable[[str, int], list[str]]


@dataclass(slots=True)
class EnqueueResult:
    video_id: str
    job_id: str | None
    #: ``created`` — a job is queued and the runner can start now.
    #: ``queued_for_transcription`` — same, but S1 metadata shows no captions
    #: of any kind, so the note only appears once the ASR worker is next
    #: awake (design §5 S2). The iOS Shortcut shows a different message for
    #: this. ``existing`` / ``rejected`` — no job created.
    status: Literal["created", "queued_for_transcription", "existing", "rejected"]
    #: Set only when status == "existing" and the row already has a note.
    existing_note_path: str | None = None
    rejected_reason: str | None = None


def _default_fetch(cfg: AcquisitionConfig) -> FetchMetadataFn:
    def fetch(url: str) -> VideoMetadata:
        return fetch_metadata(
            url,
            max_duration_minutes=cfg.max_duration_minutes,
            cookies_file=str(cfg.cookies_file) if cfg.cookies_file else None,
        )

    return fetch


def _default_expand(cfg: AcquisitionConfig) -> ExpandPlaylistFn:
    def expand(playlist_id: str, cap: int) -> list[str]:
        return expand_playlist(playlist_id, cap=cap)

    return expand


def enqueue(
    db: sqlite3.Connection,
    cfg: AcquisitionConfig,
    url: str,
    *,
    origin: Origin = "manual",
    force: bool = False,
    force_asr: bool = False,
    adapter: Adapter = "youtube",
    _fetch: FetchMetadataFn | None = None,
    _expand: ExpandPlaylistFn | None = None,
) -> list[EnqueueResult]:
    """Resolve `url`, deduplicate against `videos`, fetch metadata, and store
    a row. Returns one result per video — more than one only for a bare
    playlist URL (design §5 S0).
    """
    fetch = _fetch or _default_fetch(cfg)
    expand = _expand or _default_expand(cfg)

    resolved = resolve_url(url)
    if resolved is None:
        # Not one of the shapes S0 recognises directly. Fall through to
        # yt-dlp itself for anything it can still place — unlisted domains,
        # youtube-nocookie.com — by treating the raw URL as a single-video
        # job and letting fetch() raise VideoRejected if it truly cannot.
        return [_resolve_one(db, url, url, origin, force, force_asr, adapter, fetch)]

    if resolved.is_playlist:
        assert resolved.playlist_id is not None
        urls = expand(resolved.playlist_id, cfg.playlist_expansion_cap)
        return [
            _resolve_one(db, u, u, origin, force, force_asr, adapter, fetch) for u in urls
        ]

    assert resolved.video_id is not None
    video_id = resolved.video_id
    canonical = canonical_url(video_id)
    return [_resolve_one(db, url, canonical, origin, force, force_asr, adapter, fetch, video_id)]


def _resolve_one(
    db: sqlite3.Connection,
    original_url: str,
    canonical: str,
    origin: Origin,
    force: bool,
    force_asr: bool,
    adapter: Adapter,
    fetch: FetchMetadataFn,
    video_id_hint: str | None = None,
) -> EnqueueResult:
    # A repeat request returns the existing row unless force=true (design §5
    # S0). The hint lets us dedupe before the network call when S0 already
    # extracted the id; when it didn't, dedupe only happens after fetch()
    # resolves one.
    if video_id_hint is not None and not force:
        existing = _find_existing(db, adapter, video_id_hint)
        if existing is not None:
            return EnqueueResult(
                video_id=video_id_hint,
                job_id=None,
                status="existing",
                existing_note_path=existing["note_path"],
            )

    try:
        meta = fetch(canonical)
    except VideoRejected as exc:
        # A real id (from S0's URL parse, or from yt-dlp itself before it
        # rejected) is what dedup keys on — only persist when one exists, so
        # unrelated dead URLs that never resolved to any id cannot collide
        # with each other under a shared placeholder key.
        video_id = video_id_hint or exc.video_id
        if video_id is not None:
            _store_rejection(db, adapter, origin, original_url, canonical, video_id, exc)
        return EnqueueResult(
            video_id=video_id or "unresolved",
            job_id=None,
            status="rejected",
            rejected_reason=exc.reason,
        )

    if not force:
        existing = _find_existing(db, adapter, meta.video_id)
        if existing is not None:
            return EnqueueResult(
                video_id=meta.video_id,
                job_id=None,
                status="existing",
                existing_note_path=existing["note_path"],
            )

    row_id = f"{adapter}:{meta.video_id}"
    now = iso_now()
    job_id = new_job_id()

    db.execute(
        """
        INSERT INTO videos (
            id, video_id, adapter, origin, url, canonical_url, metadata,
            stage_resolve, stage_metadata, force_asr, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'done', 'done', ?, ?, ?)
        ON CONFLICT(video_id, adapter) DO UPDATE SET
            metadata = excluded.metadata,
            stage_resolve = 'done',
            stage_metadata = 'done',
            force_asr = excluded.force_asr,
            updated_at = excluded.updated_at
        """,
        (
            row_id,
            meta.video_id,
            adapter,
            origin,
            original_url,
            canonical,
            json.dumps(_metadata_dict(meta)),
            int(force_asr),
            now,
            now,
        ),
    )
    db.execute(
        "INSERT INTO jobs (id, video_id, force, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 'queued', ?, ?)",
        (job_id, row_id, int(force), now, now),
    )
    db.commit()
    # No captions of any kind → S2 will fall straight through to remote ASR
    # (design §5 S2). Say so, so a caller waiting on the note knows it may be
    # a while. Borderline auto-captions that the T1 heuristic later rejects
    # cannot be predicted here — this only flags the clear-cut case.
    has_captions = bool(
        meta.has_manual_subs or meta.manual_sub_langs or meta.auto_caption_langs
    )
    status: Literal["created", "queued_for_transcription"] = (
        "created" if has_captions else "queued_for_transcription"
    )
    log.info(
        "video.enqueued", video_id=meta.video_id, origin=origin, job_id=job_id, status=status
    )
    return EnqueueResult(video_id=meta.video_id, job_id=job_id, status=status)


def _find_existing(
    db: sqlite3.Connection, adapter: str, video_id: str
) -> sqlite3.Row | None:
    row: sqlite3.Row | None = db.execute(
        "SELECT * FROM videos WHERE adapter = ? AND video_id = ?", (adapter, video_id)
    ).fetchone()
    return row


def _store_rejection(
    db: sqlite3.Connection,
    adapter: str,
    origin: Origin,
    original_url: str,
    canonical: str,
    video_id: str,
    exc: VideoRejected,
) -> None:
    """Rejections are recorded too (design §5 S1 — "reject early and
    cleanly"), so a repeat POST of a known-dead URL answers instantly instead
    of paying for another yt-dlp call, and /metrics can count them."""
    row_id = f"{adapter}:{video_id}"
    now = iso_now()
    db.execute(
        """
        INSERT INTO videos (
            id, video_id, adapter, origin, url, canonical_url,
            stage_resolve, stage_metadata, stage_errors, last_error,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'done', 'failed', ?, ?, ?, ?)
        ON CONFLICT(video_id, adapter) DO UPDATE SET
            stage_metadata = 'failed',
            stage_errors = excluded.stage_errors,
            last_error = excluded.last_error,
            updated_at = excluded.updated_at
        """,
        (
            row_id,
            video_id,
            adapter,
            origin,
            original_url,
            canonical,
            json.dumps({"metadata": exc.reason}),
            str(exc),
            now,
            now,
        ),
    )
    db.commit()
    log.warning("video.rejected", video_id=video_id, reason=exc.reason, detail=str(exc))


def _metadata_dict(meta: VideoMetadata) -> dict[str, object]:
    return {
        "title": meta.title,
        "channel": meta.channel,
        "channel_id": meta.channel_id,
        "duration_s": meta.duration_s,
        "upload_date": meta.upload_date,
        "description": meta.description,
        "chapters": meta.chapters,
        "language": meta.language,
        "has_manual_subs": meta.has_manual_subs,
        "manual_sub_langs": meta.manual_sub_langs,
        "auto_caption_langs": meta.auto_caption_langs,
        "thumbnail_url": meta.thumbnail_url,
    }


def metadata_from_row(row: sqlite3.Row) -> VideoMetadata:
    """Rebuild the S1 `VideoMetadata` from the JSON `_metadata_dict` stored on
    the `videos` row. The inverse of `_metadata_dict` — used by every stage
    that runs long after S1 (the ASR drain job, the job runner) and cannot
    assume the original in-memory object is still around.
    """
    stored = json.loads(row["metadata"] or "{}")
    return VideoMetadata(
        video_id=row["video_id"],
        title=stored.get("title", ""),
        channel=stored.get("channel", ""),
        channel_id=stored.get("channel_id", ""),
        duration_s=stored.get("duration_s", 0),
        upload_date=stored.get("upload_date"),
        description=stored.get("description", ""),
        chapters=stored.get("chapters") or [],
        language=stored.get("language"),
        has_manual_subs=bool(stored.get("has_manual_subs", False)),
        manual_sub_langs=stored.get("manual_sub_langs") or [],
        auto_caption_langs=stored.get("auto_caption_langs") or [],
        thumbnail_url=stored.get("thumbnail_url"),
    )
