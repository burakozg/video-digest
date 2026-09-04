from __future__ import annotations

import pytest

from video_digest.sources.youtube import ResolvedURL, canonical_url, resolve_url

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}",
        f"https://m.youtube.com/watch?v={VID}",
        f"https://www.youtube.com/watch?v={VID}&t=43s",
        f"https://www.youtube.com/watch?v={VID}&list=PLxxxx&index=3",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?t=43",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/live/{VID}",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube.com/v/{VID}",
    ],
)
def test_resolves_to_bare_video_id(url: str) -> None:
    resolved = resolve_url(url)
    assert resolved == ResolvedURL(video_id=VID)
    assert not resolved.is_playlist


def test_consent_wrapper_unwraps_to_the_real_target() -> None:
    from urllib.parse import quote

    target = f"https://www.youtube.com/watch?v={VID}"
    wrapper = f"https://consent.youtube.com/m?continue={quote(target, safe='')}&gl=SE"
    assert resolve_url(wrapper) == ResolvedURL(video_id=VID)


def test_bare_playlist_url_resolves_to_a_playlist() -> None:
    resolved = resolve_url("https://www.youtube.com/playlist?list=PLxxxxxxxxxxxx")
    assert resolved is not None
    assert resolved.is_playlist
    assert resolved.playlist_id == "PLxxxxxxxxxxxx"


def test_watch_url_with_list_is_a_single_video_not_a_playlist_job() -> None:
    """A `watch?v=...&list=...` link is one video that happens to sit in a
    playlist — the design's "playlist URLs expand to N jobs" means a bare
    playlist link only."""
    resolved = resolve_url(f"https://www.youtube.com/watch?v={VID}&list=PLxxxx")
    assert resolved is not None
    assert not resolved.is_playlist
    assert resolved.video_id == VID


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/",
        "https://www.youtube.com/results?search_query=x",
        "not a url at all",
        "",
    ],
)
def test_unrecognised_shapes_return_none(url: str) -> None:
    assert resolve_url(url) is None


def test_canonical_url_roundtrips() -> None:
    assert resolve_url(canonical_url(VID)) == ResolvedURL(video_id=VID)
