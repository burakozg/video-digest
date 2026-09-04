"""The ntfy notifier (design §9). Best-effort: a delivery that fails logs and
returns False, it never raises into the pipeline that called it.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from video_digest.config import NotificationConfig
from video_digest.notify import Notifier

_URL = "https://ntfy.example"
_TOPIC = "vid-alerts"
_ENDPOINT = f"{_URL}/{_TOPIC}"


def _enabled() -> NotificationConfig:
    return NotificationConfig(enabled=True, ntfy_url=_URL, topic=_TOPIC)


@pytest.mark.asyncio
async def test_disabled_notifier_sends_nothing() -> None:
    with respx.mock:
        route = respx.post(_ENDPOINT).mock(return_value=httpx.Response(200))
        sent = await Notifier(NotificationConfig(enabled=False)).send("t", "m")
    assert sent is False
    assert not route.called


@pytest.mark.asyncio
async def test_send_posts_body_and_headers_to_the_topic() -> None:
    with respx.mock:
        route = respx.post(_ENDPOINT).mock(return_value=httpx.Response(200))
        sent = await Notifier(_enabled()).send(
            "A title", "the message", priority="high", tags=["movie_camera", "x"]
        )
    assert sent is True
    request = route.calls.last.request
    assert request.content == b"the message"
    assert request.headers["Title"] == "A title"
    assert request.headers["Priority"] == "high"
    assert request.headers["Tags"] == "movie_camera,x"
    assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_send_adds_bearer_token_when_configured() -> None:
    with respx.mock:
        route = respx.post(_ENDPOINT).mock(return_value=httpx.Response(200))
        await Notifier(_enabled(), token="tk_secret").send("t", "m")
    assert route.calls.last.request.headers["Authorization"] == "Bearer tk_secret"


@pytest.mark.asyncio
async def test_send_falls_back_to_config_priority_and_tags() -> None:
    cfg = NotificationConfig(
        enabled=True, ntfy_url=_URL, topic=_TOPIC, priority="urgent", tags=["bell"]
    )
    with respx.mock:
        route = respx.post(_ENDPOINT).mock(return_value=httpx.Response(200))
        await Notifier(cfg).send("t", "m")
    request = route.calls.last.request
    assert request.headers["Priority"] == "urgent"
    assert request.headers["Tags"] == "bell"


@pytest.mark.asyncio
async def test_transport_error_is_swallowed() -> None:
    with respx.mock:
        respx.post(_ENDPOINT).mock(side_effect=httpx.ConnectError("no route"))
        sent = await Notifier(_enabled()).send("t", "m")
    assert sent is False


@pytest.mark.asyncio
async def test_4xx_counts_as_a_failed_send() -> None:
    with respx.mock:
        respx.post(_ENDPOINT).mock(return_value=httpx.Response(403))
        sent = await Notifier(_enabled()).send("t", "m")
    assert sent is False


@pytest.mark.asyncio
async def test_job_failed_sends_a_high_priority_alert() -> None:
    with respx.mock:
        route = respx.post(_ENDPOINT).mock(return_value=httpx.Response(200))
        await Notifier(_enabled()).job_failed("dQw4w9WgXcQ", "summarize", "RuntimeError: boom")
    request = route.calls.last.request
    assert request.headers["Priority"] == "high"
    assert b"boom" in request.content
    assert "dQw4w9WgXcQ" in request.headers["Title"]


@pytest.mark.asyncio
async def test_asr_stale_is_a_noop_for_an_empty_list() -> None:
    with respx.mock:
        route = respx.post(_ENDPOINT).mock(return_value=httpx.Response(200))
        await Notifier(_enabled()).asr_stale([])
    assert not route.called


@pytest.mark.asyncio
async def test_asr_stale_names_the_parked_videos() -> None:
    with respx.mock:
        route = respx.post(_ENDPOINT).mock(return_value=httpx.Response(200))
        await Notifier(_enabled()).asr_stale(
            [{"video_id": "aaa"}, {"video_id": "bbb"}]
        )
    body = route.calls.last.request.content.decode()
    assert "aaa" in body and "bbb" in body
