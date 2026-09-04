"""S6: cleanup (design §5).

Per-video audio deletion after a successful transcription already happens
inline in `pipeline/transcript.py::acquire_transcript` (its `finally`
block) — that is the common case and needs no sweep. This module is the
second line of defence design §5 S6 asks for: "a scheduled job sweeps
orphaned media older than `orphan_media_hours`" — audio left behind by a
crash between download and the `finally` running, or by a `keep_audio: true`
deployment that still wants a bound on disk use. Media accumulation is how
this class of app quietly fills a NAS.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import OutputConfig
from ..logging_setup import get_logger

log = get_logger(__name__)

#: Audio files the pipeline itself downloads (sources/youtube.py's
#: download_audio uses "<video_id>.<ext>" under work_dir). Only files
#: matching a known audio extension are ever removed — the sweep must never
#: touch a stray file a human or another process left in the same directory.
_AUDIO_EXTENSIONS = {".m4a", ".webm", ".opus", ".mp3", ".wav", ".ogg", ".aac"}


def sweep_orphaned_media(output_cfg: OutputConfig, *, now: float | None = None) -> list[Path]:
    """Delete audio files in `work_dir` older than `orphan_media_hours`.
    Returns the paths removed, for logging/metrics."""
    work_dir = Path(output_cfg.work_dir)
    if not work_dir.exists():
        return []

    cutoff = (now if now is not None else time.time()) - output_cfg.orphan_media_hours * 3600
    removed: list[Path] = []
    for path in work_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed.append(path)
        except OSError as exc:
            log.warning("cleanup.sweep_failed", path=str(path), error=str(exc))

    if removed:
        log.info("cleanup.swept_orphaned_media", count=len(removed))
    return removed
