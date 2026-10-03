"""`_get_with_retry` (sources/youtube.py): the timedtext CDN 429s under
ordinary use, with no account or key to fall back on — a single failed GET
must not read as "no captions for this video" (the bug that let two videos
sit permanently `stage_transcript = 'failed'` after one rate-limited attempt,
2026-09-23). Retries are bounded and only for transient statuses; anything
else fails immediately since retrying the same signed URL cannot fix it.
"""

from __future__ import annotations

import time

import httpx
import pytest
import respx

from video_digest.sources import youtube as yt

URL = "https://r1---xyz.googlevideo.com/timedtext?sig=abc"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Retries must not actually block the test suite — capture the delays
    instead of sleeping them. `youtube.py` does `import time; time.sleep(...)`,
    so patching the shared stdlib module object (not an attribute reached
    through the `yt` re-export) is what actually intercepts the call."""
    delays: list[float] = []
    monkeypatch.setattr(time, "sleep", delays.append)
    return delays


class TestTransientStatusesRetry:
    @respx.mock
    def test_a_429_eventually_succeeds(self, _no_real_sleep: list[float]) -> None:
        route = respx.get(URL).mock(
            side_effect=[
                httpx.Response(429, text="rate limited"),
                httpx.Response(429, text="rate limited"),
                httpx.Response(200, text="WEBVTT\n\ncaptions"),
            ]
        )
        text = yt._get_with_retry(URL, timeout_s=5.0)
        assert text == "WEBVTT\n\ncaptions"
        assert route.call_count == 3
        assert len(_no_real_sleep) == 2  # one sleep between each retry

    @respx.mock
    def test_backoff_grows_between_attempts(self, _no_real_sleep: list[float]) -> None:
        respx.get(URL).mock(
            side_effect=[httpx.Response(503), httpx.Response(503), httpx.Response(200, text="ok")]
        )
        yt._get_with_retry(URL, timeout_s=5.0)
        assert _no_real_sleep[1] > _no_real_sleep[0]

    @respx.mock
    def test_retry_after_header_is_honoured(self, _no_real_sleep: list[float]) -> None:
        respx.get(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"retry-after": "30"}),
                httpx.Response(200, text="ok"),
            ]
        )
        yt._get_with_retry(URL, timeout_s=5.0)
        assert _no_real_sleep == [30.0]

    @respx.mock
    def test_exhausting_every_attempt_raises(self, _no_real_sleep: list[float]) -> None:
        respx.get(URL).mock(return_value=httpx.Response(429, text="still limited"))
        with pytest.raises(httpx.HTTPStatusError):
            yt._get_with_retry(URL, timeout_s=5.0)
        assert len(_no_real_sleep) == yt._RETRY_ATTEMPTS - 1


class TestPermanentStatusesFailFast:
    @respx.mock
    def test_a_404_does_not_retry(self, _no_real_sleep: list[float]) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(404))
        with pytest.raises(httpx.HTTPStatusError):
            yt._get_with_retry(URL, timeout_s=5.0)
        assert route.call_count == 1
        assert _no_real_sleep == []

    @respx.mock
    def test_a_403_on_an_expired_signed_url_does_not_retry(
        self, _no_real_sleep: list[float]
    ) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(403))
        with pytest.raises(httpx.HTTPStatusError):
            yt._get_with_retry(URL, timeout_s=5.0)
        assert route.call_count == 1
        assert _no_real_sleep == []
