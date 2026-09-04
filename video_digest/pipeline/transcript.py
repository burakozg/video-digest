"""S2 orchestration (design §5): T0/T1 ladder, falling through to T2 (remote
ASR) with `pending_asr` semantics when neither is usable.

`pending_asr` is a first-class state, not an error (design §5 S2): when the
worker is unreachable, this parks the job at `stage_transcript = 'pending'`
with `asr_queued_at` set and returns without touching the retry-attempt
counter — a sleeping laptop must cost a delay, never a failed video. The
drain job (`pipeline/asr_drain.py`, scheduled) works the queue once the
worker answers again.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..config import AcquisitionConfig, OutputConfig, TranscriptionConfig
from ..logging_setup import get_logger
from ..sources.youtube import VideoMetadata, download_audio
from ..transcripts.asr import ASRUnavailable, RemoteASRBackend
from ..transcripts.ladder import NeedsASR
from ..transcripts.ladder import acquire as acquire_captions
from ..transcripts.normalize import Transcript
from ..utils import iso_now

log = get_logger(__name__)

Tier = str  # "T0" | "T1" | "T2"


@dataclass(slots=True)
class TranscriptAcquired:
    transcript: Transcript
    tier: Tier
    degraded: bool = False
    asr_model: str | None = None


class ParkedForASR(Exception):
    """No T0/T1 source was usable and the ASR worker did not answer — the
    caller should record `pending_asr` (design §5 S2) rather than fail the
    video."""


async def probe_asr_worker(cfg: TranscriptionConfig, *, timeout_s: float = 5.0) -> bool:
    """A cheap reachability check before committing to an audio download —
    `GET /v1/models`, the one endpoint every OpenAI-API-compatible server
    (speaches included) implements for exactly this. Any failure at all
    reads as "not available now"; the worker being asleep is the expected
    case, not an error to surface."""
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(f"{cfg.remote_url.rstrip('/')}/v1/models")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _download_audio_sync(
    video_id: str, output_cfg: OutputConfig, acquisition_cfg: AcquisitionConfig
) -> Path:
    work_dir = Path(output_cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cookies = str(acquisition_cfg.cookies_file) if acquisition_cfg.cookies_file else None
    return Path(download_audio(video_id, str(work_dir), cookies_file=cookies))


async def acquire_transcript(
    meta: VideoMetadata,
    acquisition_cfg: AcquisitionConfig,
    transcription_cfg: TranscriptionConfig,
    output_cfg: OutputConfig,
    *,
    force_asr: bool = False,
    asr_backend: RemoteASRBackend | None = None,
    _acquire_captions: Any = None,
    _probe: Any = None,
    _download: Any = None,
) -> TranscriptAcquired:
    """Raises `ParkedForASR` when the video needs T2 and the worker is not
    answering right now — never raises for "the worker is asleep" as if it
    were a video-specific failure.

    The three trailing `_`-prefixed parameters are test seams (this
    codebase's established pattern — see `pipeline/resolve.py`'s `_fetch`),
    not part of the intended call surface.
    """
    ladder = _acquire_captions or acquire_captions
    probe = _probe or probe_asr_worker
    download = _download or _download_audio_sync

    if not force_asr:
        try:
            acquired = ladder(meta, acquisition_cfg)
            return TranscriptAcquired(transcript=acquired.transcript, tier=acquired.tier)
        except NeedsASR:
            pass

    worker_up = await probe(transcription_cfg)

    if not worker_up and transcription_cfg.degrade_to_captions_when_offline and not force_asr:
        # "A decent auto-caption transcript now beats a perfect one on
        # Tuesday" (design §5 S2) — loosened T1 is tried before parking.
        try:
            acquired = ladder(meta, acquisition_cfg, loosen=True)
            return TranscriptAcquired(
                transcript=acquired.transcript, tier=acquired.tier, degraded=True
            )
        except NeedsASR:
            pass

    if not worker_up:
        raise ParkedForASR()

    backend = asr_backend or RemoteASRBackend(transcription_cfg)
    # download_audio is a synchronous, network-bound yt-dlp call — run off
    # the event loop rather than blocking every other in-flight request for
    # however long the download takes.
    audio_path = await asyncio.to_thread(download, meta.video_id, output_cfg, acquisition_cfg)
    try:
        result = await backend.transcribe(
            audio_path, language=meta.language, chapters=meta.chapters
        )
    except ASRUnavailable as exc:
        log.warning("transcript.asr_unavailable", video_id=meta.video_id, error=str(exc))
        raise ParkedForASR() from exc
    finally:
        if not output_cfg.keep_audio:
            await asyncio.to_thread(audio_path.unlink, True)

    return TranscriptAcquired(
        transcript=result.transcript, tier="T2", asr_model=transcription_cfg.model
    )


def park_pending_asr(db: sqlite3.Connection, row_id: str) -> None:
    """Record the park (design §5 S2's `pending_asr`) without touching the
    attempt counter — this is not a failure, so it must not spend the
    video's retry budget.

    `asr_queued_at` is set only if unset (`COALESCE`): the drain job can
    call this again after a retry that failed a second time, and the
    staleness clock (`stale_pending_asr`, `asr_stale_hours`) has to measure
    from the *original* park, or a worker that flickers up just long enough
    to fail one retry would never trip the alert.
    """
    db.execute(
        "UPDATE videos SET stage_transcript = 'pending', "
        "asr_queued_at = COALESCE(asr_queued_at, ?), updated_at = ? WHERE id = ?",
        (iso_now(), iso_now(), row_id),
    )
    db.commit()
    log.info("transcript.parked_pending_asr", row_id=row_id)


def pending_asr_rows(db: sqlite3.Connection) -> list[sqlite3.Row]:
    rows: list[sqlite3.Row] = db.execute(
        "SELECT * FROM videos WHERE asr_queued_at IS NOT NULL ORDER BY asr_queued_at ASC"
    ).fetchall()
    return rows


def clear_pending_asr(db: sqlite3.Connection, row_id: str) -> None:
    db.execute(
        "UPDATE videos SET asr_queued_at = NULL, updated_at = ? WHERE id = ?",
        (iso_now(), row_id),
    )
    db.commit()


def stale_pending_asr(db: sqlite3.Connection, *, stale_hours: int) -> list[dict[str, Any]]:
    """Rows parked longer than `asr_stale_hours` (design §5 S2 — "alert on
    age, not on failure"). Returned as plain dicts rather than sqlite3.Row so
    a caller (notify.py, M6) does not need this module's connection alive."""
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(hours=stale_hours)).isoformat()
    rows = db.execute(
        "SELECT id, video_id, asr_queued_at FROM videos "
        "WHERE asr_queued_at IS NOT NULL AND asr_queued_at < ? ORDER BY asr_queued_at ASC",
        (cutoff,),
    ).fetchall()
    return [dict(r) for r in rows]
