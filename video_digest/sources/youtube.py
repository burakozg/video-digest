"""The one YouTube-aware module (design §1.1 — other adapters enter at S1
without renaming the service).

Two halves: URL canonicalisation (S0, pure — no network) and metadata
extraction (S1, via yt-dlp's library API — never a subprocess shell-out,
design §3). `_parse_info` is kept pure and separate from `fetch_metadata` so
the rejection rules are testable against canned yt-dlp output, without a
network call in the test suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

#: YouTube's own video id shape. Not a strict guarantee (ids are opaque, and
#: yt-dlp is the real authority) — just tight enough to keep the regexes below
#: from swallowing an unrelated path segment.
_VIDEO_ID = r"[A-Za-z0-9_-]{11}"

_ID_RE = re.compile(rf"^(?P<id>{_VIDEO_ID})$")

#: Path-shape patterns carrying the id directly in the path (unlike /watch,
#: which carries it in a query param and is handled separately). Tried in
#: order; each must capture a group named `id`.
_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"^/shorts/(?P<id>{_VIDEO_ID})"),
    re.compile(rf"^/live/(?P<id>{_VIDEO_ID})"),
    re.compile(rf"^/embed/(?P<id>{_VIDEO_ID})"),
    re.compile(rf"^/v/(?P<id>{_VIDEO_ID})"),
)

#: youtu.be short links carry the id as the whole path.
_YOUTU_BE_RE = re.compile(rf"^/(?P<id>{_VIDEO_ID})")

#: Compared after stripping a leading "www." from the parsed host.
_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com"}


@dataclass(slots=True)
class ResolvedURL:
    """What S0 (design §5) settles before any network call.

    Exactly one of `video_id` / `playlist_id` is set for a video or playlist
    URL respectively — never both: a `watch?v=...&list=...` link is a single
    video that happens to sit in a playlist, not a playlist job (design §5
    S0 — "playlist URLs expand to N jobs", meaning a *bare* playlist link).
    """

    video_id: str | None = None
    playlist_id: str | None = None

    @property
    def is_playlist(self) -> bool:
        return self.playlist_id is not None and self.video_id is None


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def resolve_url(url: str, *, _depth: int = 0) -> ResolvedURL | None:
    """Canonicalise a YouTube URL to a bare video or playlist id (design §5 S0).

    Handles `youtu.be/ID`, `/watch?v=ID`, `/shorts/ID`, `/live/ID`, `&t=`,
    `&list=`, and consent/redirect wrappers (`consent.youtube.com`, which
    google.com/EU traffic is bounced through — the real target sits urlencoded
    in a `continue` query parameter). Returns None for a URL this cannot place
    at all, which callers should still hand to yt-dlp directly: it resolves
    plenty of shapes (unlisted domains, `youtube-nocookie.com`) that a
    hand-written regex will never keep up with, and it is the pipeline's own
    fallback when this is the only patterns known to be common enough to skip
    a network round trip.
    """
    if _depth > 3:  # a consent wrapper pointing at a consent wrapper is a loop, not a link
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = parsed.hostname or ""
    host = host.removeprefix("www.")

    if host.startswith("consent."):
        query = parse_qs(parsed.query)
        target = query.get("continue", [None])[0]
        if not target:
            return None
        return resolve_url(unquote(target), _depth=_depth + 1)

    if host == "youtu.be":
        match = _YOUTU_BE_RE.match(parsed.path)
        return ResolvedURL(video_id=match.group("id")) if match else None

    if host not in _YOUTUBE_HOSTS:
        return None

    if parsed.path.rstrip("/") in ("", "/watch"):
        query = parse_qs(parsed.query)
        video_id = query.get("v", [None])[0]
        if video_id and _ID_RE.match(video_id):
            return ResolvedURL(video_id=video_id)
        playlist_id = query.get("list", [None])[0]
        if playlist_id:
            return ResolvedURL(playlist_id=playlist_id)
        return None

    for pattern in _PATH_PATTERNS:
        match = pattern.match(parsed.path)
        if match:
            return ResolvedURL(video_id=match.group("id"))

    if parsed.path.rstrip("/") == "/playlist":
        query = parse_qs(parsed.query)
        playlist_id = query.get("list", [None])[0]
        if playlist_id:
            return ResolvedURL(playlist_id=playlist_id)

    return None


# ── S1: metadata (design §5) ──────────────────────────────────────────────


Availability = Literal[
    "public", "unlisted", "private", "premium_only", "subscriber_only", "needs_auth"
]


@dataclass(slots=True)
class VideoMetadata:
    video_id: str
    title: str
    channel: str
    channel_id: str
    duration_s: int
    upload_date: str | None  # ISO yyyy-mm-dd
    description: str
    chapters: list[dict[str, Any]] = field(default_factory=list)
    language: str | None = None
    has_manual_subs: bool = False
    manual_sub_langs: list[str] = field(default_factory=list)
    auto_caption_langs: list[str] = field(default_factory=list)
    thumbnail_url: str | None = None


class VideoRejected(Exception):
    """Clean, early rejection (design §5 S1) — not a fetch failure. The
    `reason` is a short machine-stable slug, safe to store and to log.
    `video_id` is set whenever yt-dlp got far enough to resolve one, which is
    everything except a URL it could not place at all — the caller only
    persists a rejection row when it has a real id to key it by."""

    def __init__(self, reason: str, detail: str, video_id: str | None = None) -> None:
        super().__init__(detail)
        self.reason = reason
        self.video_id = video_id


def _upload_date_iso(raw: str | None) -> str | None:
    if not raw or len(raw) != 8:
        return None
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def parse_info(info: dict[str, Any], *, max_duration_minutes: int) -> VideoMetadata:
    """Pure: turns yt-dlp's `extract_info` dict into `VideoMetadata`, or
    raises `VideoRejected`. No network call — kept separate from
    `fetch_metadata` so the rejection rules are unit-testable against canned
    fixtures (design §5 S1's "reject early and cleanly" list).
    """
    vid = info.get("id")

    live_status = info.get("live_status")
    if live_status in ("is_live", "is_upcoming"):
        raise VideoRejected("live_in_progress", f"live_status={live_status!r}", vid)

    availability = info.get("availability")
    if availability in ("premium_only", "subscriber_only"):
        raise VideoRejected("members_only", f"availability={availability!r}", vid)
    if availability == "needs_auth":
        raise VideoRejected(
            "age_gated", "requires authentication (age-gated, no cookies_file)", vid
        )
    if availability == "private":
        raise VideoRejected("private", "video is private", vid)

    # yt-dlp raises GeoRestrictedError itself for a hard region block before
    # info is even returned; this is the softer case where it still resolves
    # info but flags no playable format for our region.
    if not info.get("formats") and not info.get("url") and live_status != "was_live":
        raise VideoRejected("region_blocked", "no playable format returned for this region", vid)

    duration_s = int(info.get("duration") or 0)
    if duration_s > max_duration_minutes * 60:
        raise VideoRejected(
            "too_long",
            f"duration {duration_s}s exceeds max_duration_minutes={max_duration_minutes}",
            vid,
        )

    subtitles = info.get("subtitles") or {}
    auto_captions = info.get("automatic_captions") or {}

    return VideoMetadata(
        video_id=info["id"],
        title=info.get("title") or "",
        channel=info.get("channel") or info.get("uploader") or "",
        channel_id=info.get("channel_id") or "",
        duration_s=duration_s,
        upload_date=_upload_date_iso(info.get("upload_date")),
        description=info.get("description") or "",
        chapters=list(info.get("chapters") or []),
        language=info.get("language"),
        has_manual_subs=bool(subtitles),
        manual_sub_langs=sorted(subtitles),
        auto_caption_langs=sorted(auto_captions),
        thumbnail_url=info.get("thumbnail"),
    )


def fetch_metadata(
    url: str, *, max_duration_minutes: int, cookies_file: str | None = None
) -> VideoMetadata:
    """`yt-dlp` extract-info, `download=False` (design §5 S1). Thin and
    deliberately untested directly — see `parse_info` for the tested logic.
    """
    import yt_dlp

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info is None:
        raise VideoRejected("unresolvable", f"yt-dlp returned no info for {url!r}")
    return parse_info(info, max_duration_minutes=max_duration_minutes)


def expand_playlist(playlist_id: str, *, cap: int) -> list[str]:
    """Bare playlist URLs expand to N jobs, capped (design §5 S0 — "refuse
    silently-huge expansions"). `extract_flat` avoids resolving every video's
    full metadata just to enumerate ids.
    """
    import yt_dlp

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": cap + 1,  # +1 so "more than cap" is detectable, not silently truncated
    }
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    entries = list((info or {}).get("entries") or [])
    if len(entries) > cap:
        raise VideoRejected(
            "playlist_too_large",
            f"playlist has more than {cap} videos (playlist_expansion_cap)",
        )
    return [canonical_url(e["id"]) for e in entries if e.get("id")]


def fetch_subtitle_vtt(
    video_id: str,
    lang: str,
    *,
    auto: bool,
    cookies_file: str | None = None,
    timeout_s: float = 30.0,
) -> str:
    """Publisher subs (`auto=False`, T0) or auto-captions (`auto=True`, T1) —
    design §5 S2's `--write-subs` / `--write-auto-subs`, but read straight
    into memory rather than written to disk: extract_info already returns a
    signed CDN URL per language/format, so there is nothing to download twice.

    A second `extract_info` call rather than reusing S1's — deliberately: the
    signed URL is short-lived, and S2 can run long after S1 (a video queued
    for days behind other work), so a stored URL would be stale by the time
    it is used.
    """
    import httpx
    import yt_dlp

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "writesubtitles": not auto,
        "writeautomaticsub": auto,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(canonical_url(video_id), download=False)

    key = "automatic_captions" if auto else "subtitles"
    track = (info or {}).get(key, {}).get(lang)
    if not track:
        kind = "auto-caption" if auto else "manual subtitle"
        raise VideoRejected(
            "caption_track_missing", f"no {kind} track for lang={lang!r}", video_id
        )
    entry = next((f for f in track if f.get("ext") == "vtt"), track[0])
    url = entry.get("url")
    if not url:
        raise VideoRejected(
            "caption_track_missing", f"caption track for lang={lang!r} has no URL", video_id
        )

    resp = httpx.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.text


def download_audio(video_id: str, dest_dir: str, *, cookies_file: str | None = None) -> str:
    """Best-quality audio-only stream, for S2 T2 (remote ASR, design §5).

    No forced re-encode to 16 kHz mono: the ASR worker (speaches, an
    OpenAI-API-compatible whisper server) decodes whatever container ffmpeg
    understands, and re-encoding here would cost CPU on this side of the
    upload for no benefit — the file gets smaller from *transcoding*, not
    from picking a lower sample rate, and bestaudio is already compressed.
    Returns the local path yt-dlp wrote to.
    """
    import yt_dlp

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": f"{dest_dir}/{video_id}.%(ext)s",
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(canonical_url(video_id), download=True)
        if info is None:
            detail = f"yt-dlp returned no info for {video_id!r}"
            raise VideoRejected("unresolvable", detail, video_id)
        path: str = ydl.prepare_filename(info)
    return path
