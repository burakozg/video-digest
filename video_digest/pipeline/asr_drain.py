"""The `pending_asr` drain job (design §5 S2).

Scheduled every `transcription.asr_poll_interval_minutes` (default 5).
Probes the worker once; if it answers, works the queue oldest-first until a
retry fails, at which point it stops rather than hammering a worker that has
just gone back to sleep mid-drain. `max_concurrent_asr` is enforced by
running strictly sequentially — the worker itself is the sole contended
resource (design §5 S2), not this process.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..config import Settings
from ..logging_setup import get_logger
from .resolve import metadata_from_row as _metadata_from_row
from .transcript import (
    ParkedForASR,
    acquire_transcript,
    clear_pending_asr,
    pending_asr_rows,
    probe_asr_worker,
    stale_pending_asr,
)

log = get_logger(__name__)


async def drain_pending_asr(
    db: sqlite3.Connection,
    settings: Settings,
    *,
    _probe: Any = None,
    _acquire_transcript: Any = None,
) -> int:
    """Returns the number of videos successfully transcribed this run.

    `_probe`/`_acquire_transcript` are test seams (this codebase's
    established pattern), not part of the intended call surface.
    """
    probe = _probe or probe_asr_worker
    acquire = _acquire_transcript or acquire_transcript

    transcription_cfg = settings.transcription
    if not await probe(transcription_cfg):
        return 0

    drained = 0
    for row in pending_asr_rows(db):
        meta = _metadata_from_row(row)
        try:
            acquired = await acquire(
                meta,
                settings.acquisition,
                transcription_cfg,
                settings.output,
                force_asr=True,
            )
        except ParkedForASR:
            # The worker answered the health probe but this attempt still
            # failed — likely gone back to sleep mid-drain. Stop rather than
            # spending a timeout on every remaining row; the next scheduled
            # fire tries again.
            log.info("asr_drain.stopped_early", video_id=meta.video_id)
            break

        db.execute(
            "UPDATE videos SET stage_transcript = 'done', transcript_tier = ?, "
            "transcript_tier_degraded = ?, transcript_json = ?, asr_model = ?, "
            "asr_queued_at = NULL, updated_at = datetime('now') WHERE id = ?",
            (
                acquired.tier,
                int(acquired.degraded),
                json.dumps(
                    [{"start_s": p.start_s, "text": p.text} for p in acquired.transcript.paragraphs]
                ),
                acquired.asr_model,
                row["id"],
            ),
        )
        db.commit()
        clear_pending_asr(db, row["id"])
        drained += 1
        log.info("asr_drain.transcribed", video_id=meta.video_id, tier=acquired.tier)

    if drained:
        log.info("asr_drain.done", drained=drained)
    return drained


def alert_on_stale_pending_asr(
    db: sqlite3.Connection, settings: Settings
) -> list[dict[str, object]]:
    """Design §5 S2: "alert on age, not on failure" — a video parked longer
    than `asr_stale_hours` signals a Mac that has been shut for two days,
    not a broken pipeline. Returns what to alert on; sending the alert is
    notify.py's job (M6), not this module's."""
    return stale_pending_asr(db, stale_hours=settings.transcription.asr_stale_hours)
