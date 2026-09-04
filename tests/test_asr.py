"""RemoteASRBackend (plan §1.4/§M5): transcription on another machine.

Ported from podcast-digest's tests/test_asr_remote.py. What these protect is
not the happy path so much as the failure path: a laptop that sleeps must
cost a delay, never a video marked failed — every failure here must surface
as `ASRUnavailable`, never a raw httpx error escaping the backend.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from video_digest.config import TranscriptionConfig
from video_digest.transcripts.asr import ASRUnavailable, RemoteASRBackend

REMOTE = "http://mac.lan:8000"
ENDPOINT = f"{REMOTE}/v1/audio/transcriptions"


def _cfg(**overrides: object) -> TranscriptionConfig:
    fields: dict[str, object] = {"remote_url": REMOTE, "model": "distil-large-v3"}
    fields.update(overrides)
    return TranscriptionConfig(**fields)  # type: ignore[arg-type]


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "video.audio"
    path.write_bytes(b"not really audio, but it is bytes on disk")
    return path


class TestBackendIdentity:
    def test_the_destination_is_just_a_url(self) -> None:
        """Moving from a laptop to a server must be one config value."""
        backend = RemoteASRBackend(_cfg(remote_url="https://asr.example.internal"))
        assert backend.name == "remote:https://asr.example.internal"


class TestTranscribe:
    @respx.mock
    @pytest.mark.asyncio
    async def test_posts_the_audio_and_builds_a_timestamped_transcript(self, audio: Path) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={
                    "text": "hello world",
                    "language": "en",
                    "duration": 42.5,
                    "segments": [
                        {"start": 0.0, "end": 2.0, "text": "hello"},
                        {"start": 2.0, "end": 4.0, "text": "world"},
                    ],
                },
            )
        )
        result = await RemoteASRBackend(_cfg()).transcribe(audio, language="en")

        assert route.called
        request = route.calls.last.request
        assert b"distil-large-v3" in request.content, "the configured model must be sent"
        assert b"video.audio" in request.content, "the audio must be uploaded as a file part"
        assert result.language == "en"
        assert result.duration_s == 42
        assert result.elapsed_s is not None
        assert result.transcript.paragraphs
        assert "hello" in result.transcript.text
        assert "world" in result.transcript.text
        assert result.transcript.paragraphs[0].start_s == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_accepts_a_server_that_only_returns_flat_text(self, audio: Path) -> None:
        """Not every server implements segmented verbose_json; a flat
        {"text": ...} still produces a transcript, just with every deep
        link pointing at the video's start."""
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"text": "plain text"}))
        result = await RemoteASRBackend(_cfg()).transcribe(audio)
        assert result.transcript.text == "plain text"
        assert result.transcript.paragraphs[0].start_s == 0
        assert result.duration_s is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_chapters_are_respected_in_paragraph_merging(self, audio: Path) -> None:
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={
                    "segments": [
                        {"start": 0.0, "text": "intro material"},
                        {"start": 30.0, "text": "chapter two material"},
                    ]
                },
            )
        )
        result = await RemoteASRBackend(_cfg()).transcribe(
            audio, chapters=[{"start_time": 0}, {"start_time": 30}]
        )
        assert [p.start_s for p in result.transcript.paragraphs] == [0, 30]


class TestFailuresAreAlwaysASRUnavailable:
    """`ASRUnavailable` is what lets S2 park a job as `pending_asr` rather
    than spend its retry budget — an httpx error escaping this backend
    would make a sleeping laptop mark videos failed instead."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_unreachable_host(self, audio: Path) -> None:
        respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(ASRUnavailable, match="unreachable"):
            await RemoteASRBackend(_cfg()).transcribe(audio)

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout(self, audio: Path) -> None:
        respx.post(ENDPOINT).mock(side_effect=httpx.ReadTimeout("too slow"))
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg()).transcribe(audio)

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_error(self, audio: Path) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(503, text="model loading"))
        with pytest.raises(ASRUnavailable, match="503"):
            await RemoteASRBackend(_cfg()).transcribe(audio)

    @respx.mock
    @pytest.mark.asyncio
    async def test_garbage_response(self, audio: Path) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text="<html>nope</html>"))
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg()).transcribe(audio)

    @respx.mock
    @pytest.mark.asyncio
    async def test_json_without_text_or_segments(self, audio: Path) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"error": "busy"}))
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg()).transcribe(audio)

    @pytest.mark.asyncio
    async def test_missing_audio_file(self, tmp_path: Path) -> None:
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg()).transcribe(tmp_path / "gone.audio")
