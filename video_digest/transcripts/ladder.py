"""S2 T0/T1: publisher and auto captions (design §5). T2 (remote ASR) is
M5's job — `NeedsASR` is what this hands off to it. Cheapest acceptable
source wins, and which tier produced it is recorded because summary quality
is downstream of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ..config import AcquisitionConfig
from ..logging_setup import get_logger
from ..sources.youtube import VideoMetadata, fetch_subtitle_vtt
from .normalize import Transcript, build_transcript

log = get_logger(__name__)

Tier = Literal["T0", "T1"]

FetchVttFn = Callable[[str, str, bool], str]

#: Tokens per minute below which a caption track reads as music/silence-heavy
#: or badly recognised (design §5 S2's "implausibly low token density" proxy).
#: Ordinary spoken English runs 110-150 wpm; this is set well below any
#: legitimate slow speaker so it only fires on tracks that are mostly gaps.
MIN_TOKENS_PER_MINUTE = 40.0

#: Halved threshold applied only when the ASR worker is offline
#: (`degrade_to_captions_when_offline`, wired in M5) — a decent transcript
#: now beats a perfect one on Tuesday.
_LOOSEN_FACTOR = 0.5


@dataclass(slots=True)
class AcquiredTranscript:
    transcript: Transcript
    tier: Tier
    #: True only when accepted under the loosened heuristic (M5).
    degraded: bool = False


class NeedsASR(Exception):
    """No T0/T1 source was usable — S2 T2 (remote ASR) must run (M5)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _primary_subtag(lang: str) -> str:
    """BCP-47's primary subtag: "en-US" -> "en", "zh-Hans" -> "zh". Language
    identity for the machine-translation check is about the language, not
    the region or script variant."""
    return lang.split("-", 1)[0].lower()


def _pick_language(available: list[str], preferred: list[str]) -> str | None:
    for lang in preferred:
        if lang in available:
            return lang
    return None


def t1_is_trustworthy(
    *,
    requested_lang: str,
    declared_language: str | None,
    cue_text: str,
    duration_s: int,
    loosen: bool = False,
) -> tuple[bool, str | None]:
    """The T1 quality heuristic (design §5 S2), as a pure function so both
    rules are testable without a network call.

    Reject when the caption language does not match the video's own declared
    language — the tell for a machine-*translated* track rather than a
    machine-*transcribed* one: YouTube auto-translates a caption into dozens
    of languages nobody asked for, and every one of those reads fluently
    while being translated from an ASR transcript, not derived from the
    audio. Also reject on implausibly low token density for the duration —
    a proxy for a music/silence-heavy or badly recognised track.
    """
    # Never loosened: "accept auto-captions that are borderline but not
    # machine-translated" (design §5 S2) — a translation is a different kind
    # of wrong from a thin transcript, and offline-degrading the density
    # floor must not also start accepting those.
    #
    # Compared on the primary subtag only ("en-US" -> "en"), not the exact
    # string. yt-dlp reports a video's declared language with a region code
    # far more often than a caption track carries one — found live on the
    # pilot run: a video declared "en-US" with real (non-translated) "en"
    # auto-captions was rejected as machine_translated on an exact-string
    # compare, which would have routed the large majority of real YouTube
    # videos to ASR needlessly.
    if declared_language and _primary_subtag(declared_language) != _primary_subtag(requested_lang):
        return False, "machine_translated"

    minutes = duration_s / 60 if duration_s else 0.0
    if minutes > 0:
        density = len(cue_text.split()) / minutes
        threshold = MIN_TOKENS_PER_MINUTE * (_LOOSEN_FACTOR if loosen else 1.0)
        if density < threshold:
            return False, "low_token_density"

    return True, None


def acquire(
    meta: VideoMetadata,
    cfg: AcquisitionConfig,
    *,
    loosen: bool = False,
    _fetch: FetchVttFn | None = None,
) -> AcquiredTranscript:
    """T0 (publisher subs) then T1 (auto-captions), in a configured language
    preference order. Raises `NeedsASR` when neither is usable.

    `loosen=True` is M5's offline-worker path (design §5 S2 —
    "prefer T0/T1 more aggressively... when the worker is asleep"): halves
    the density floor, never the language-match check. Set only when the
    caller has already confirmed the ASR worker is unreachable.
    """
    fetch = _fetch or (
        lambda video_id, lang, auto: fetch_subtitle_vtt(
            video_id,
            lang,
            auto=auto,
            cookies_file=str(cfg.cookies_file) if cfg.cookies_file else None,
        )
    )

    manual_lang = _pick_language(meta.manual_sub_langs, cfg.subtitle_languages)
    if manual_lang:
        vtt = fetch(meta.video_id, manual_lang, False)
        transcript = build_transcript(vtt, chapters=meta.chapters)
        log.info("transcript.acquired", video_id=meta.video_id, tier="T0", lang=manual_lang)
        return AcquiredTranscript(transcript=transcript, tier="T0")

    auto_lang = _pick_language(meta.auto_caption_langs, cfg.subtitle_languages)
    if auto_lang:
        vtt = fetch(meta.video_id, auto_lang, True)
        transcript = build_transcript(vtt, chapters=meta.chapters)
        trustworthy, reason = t1_is_trustworthy(
            requested_lang=auto_lang,
            declared_language=meta.language,
            cue_text=transcript.text,
            duration_s=meta.duration_s,
            loosen=loosen,
        )
        if trustworthy:
            log.info(
                "transcript.acquired",
                video_id=meta.video_id,
                tier="T1",
                lang=auto_lang,
                degraded=loosen,
            )
            return AcquiredTranscript(transcript=transcript, tier="T1", degraded=loosen)
        log.info(
            "transcript.t1_rejected", video_id=meta.video_id, lang=auto_lang, reason=reason
        )
        raise NeedsASR(reason or "t1_rejected")

    raise NeedsASR("no_captions_available")
