"""S2 T2: remote ASR (design §5). Always remote — plan §1.4 — there is no
local backend, unlike `podcast_agent/transcripts/asr.py` this is ported
from: the NAS this deploys to manages a realtime factor of 0.11 (ten hours
of CPU for a 68-minute episode), so local ASR is not a fallback worth having,
only a footgun worth removing.

`POST {remote_url}/v1/audio/transcriptions` — the OpenAI audio API shape,
which the speaches container in `~/projects/asr-server` speaks, and which
`podcast_agent`'s production backend already proved reliable against a
sleeping-laptop worker. One real divergence from that port, not cosmetic:
podcast episodes need no in-episode timestamps, so the original discards
everything but the flat `text` field. Video notes are built around `?t=`
deep links, so this reads `verbose_json`'s per-segment `start` times and
produces a `Transcript` through the same `merge_paragraphs` T0/T1 uses, for
one consistent paragraph granularity regardless of tier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..config import TranscriptionConfig
from ..logging_setup import get_logger
from .normalize import Transcript, merge_paragraphs

log = get_logger(__name__)

#: How often progress would be reported on a long decode — matched to
#: podcast_agent/transcripts/asr.py's reasoning, though this backend has no
#: streaming progress channel of its own to report from; kept as a constant
#: for parity should the remote API grow one.
PROGRESS_EVERY_SECONDS = 30.0


class ASRUnavailable(Exception):
    """The backend cannot run at all (unreachable endpoint, bad response).

    Every failure here is reported as this, deliberately: the remote worker
    is operator-configured infrastructure (a Mac that sleeps), and the
    pipeline's designed answer to that is to leave the work queued
    (`pending_asr`, design §5 S2) rather than blame the video.
    """


@dataclass(slots=True)
class ASRResult:
    transcript: Transcript
    language: str | None = None
    duration_s: int | None = None
    elapsed_s: float | None = None


class RemoteASRBackend:
    def __init__(self, cfg: TranscriptionConfig) -> None:
        self._cfg = cfg
        self._base = cfg.remote_url.rstrip("/")

    @property
    def name(self) -> str:
        return f"remote:{self._base}"

    async def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        chapters: list[dict[str, Any]] | None = None,
    ) -> ASRResult:
        if not self._base:
            raise ASRUnavailable("transcription.remote_url is not set")

        url = f"{self._base}/v1/audio/transcriptions"
        data: dict[str, str] = {
            "model": self._cfg.model,
            # verbose_json carries per-segment start times, which is the
            # whole reason this tier is worth running over T0/T1 skipping —
            # a flat {"text": ...} would lose every deep link.
            "response_format": "verbose_json",
        }
        if language:
            data["language"] = language

        started = time.monotonic()
        try:
            # Passed as a handle, not bytes: a long video's audio runs to
            # hundreds of megabytes and httpx streams from disk this way.
            with audio_path.open("rb") as handle:
                async with httpx.AsyncClient(timeout=self._cfg.remote_timeout_s) as client:
                    response = await client.post(
                        url,
                        data=data,
                        files={"file": (audio_path.name, handle, "application/octet-stream")},
                    )
        except httpx.HTTPError as exc:
            raise ASRUnavailable(f"{self.name} unreachable: {type(exc).__name__}: {exc}") from exc
        except OSError as exc:
            raise ASRUnavailable(f"could not read {audio_path}: {exc}") from exc

        elapsed = time.monotonic() - started

        if response.status_code >= 400:
            body = response.text[:300]
            raise ASRUnavailable(f"{self.name} returned HTTP {response.status_code}: {body}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ASRUnavailable(f"{self.name} returned non-JSON: {response.text[:200]}") from exc

        fragments = _fragments_from_payload(payload)
        if fragments is None:
            raise ASRUnavailable(f"{self.name} returned no text field: {str(payload)[:200]}")

        duration = payload.get("duration")
        log.info(
            "asr.complete",
            backend=self.name,
            model=self._cfg.model,
            segments=len(fragments),
            elapsed_s=round(elapsed, 1),
        )
        return ASRResult(
            transcript=Transcript(paragraphs=merge_paragraphs(fragments, chapters=chapters)),
            language=payload.get("language") or None,
            duration_s=int(duration) if isinstance(duration, int | float) else None,
            elapsed_s=elapsed,
        )

    async def close(self) -> None:
        return None


def _fragments_from_payload(payload: Any) -> list[tuple[float, str]] | None:
    """`(start_s, text)` pairs from a `verbose_json` response.

    Prefers `segments` (per-utterance timing); falls back to the flat `text`
    field at t=0 for a server that only implements the plain response shape
    — every deep link then points at the video's start rather than failing
    the whole transcript over a missing feature.
    """
    if not isinstance(payload, dict):
        return None
    segments = payload.get("segments")
    if isinstance(segments, list) and segments:
        fragments = [
            (float(s["start"]), str(s["text"]).strip())
            for s in segments
            if isinstance(s, dict) and str(s.get("text") or "").strip()
        ]
        return fragments or None
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return [(0.0, text.strip())]
    return None
