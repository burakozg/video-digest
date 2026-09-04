from __future__ import annotations

from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from video_digest.llm.models import CallMeta, ChunkSummary, VideoDigest
from video_digest.pipeline.summarize import chunk_transcript, summarize_video
from video_digest.sources.youtube import VideoMetadata
from video_digest.transcripts.normalize import Paragraph, Transcript

T = TypeVar("T", bound=BaseModel)


def _meta(**overrides: object) -> VideoMetadata:
    fields: dict[str, object] = {
        "video_id": "v1",
        "title": "A Video",
        "channel": "A Channel",
        "channel_id": "UCxxxx",
        "duration_s": 600,
        "upload_date": "2026-08-20",
        "description": "A description.",
        "chapters": [],
    }
    fields.update(overrides)
    return VideoMetadata(**fields)  # type: ignore[arg-type]


def _paragraphs(n: int, *, seconds_apart: int = 70, words_each: int = 20) -> list[Paragraph]:
    return [
        Paragraph(start_s=i * seconds_apart, text=" ".join(f"word{i}_{w}" for w in range(words_each)))
        for i in range(n)
    ]


class TestChunkingWithoutChapters:
    def test_short_transcript_is_one_chunk(self) -> None:
        transcript = Transcript(paragraphs=_paragraphs(3))
        chunks = chunk_transcript(transcript)
        assert len(chunks) == 1
        assert chunks[0].start_s == 0

    def test_long_transcript_splits_on_token_budget(self) -> None:
        # Each paragraph ~20 words * ~6 chars = ~120 chars. Budget is 8000
        # tokens * 4 chars = 32000 chars, so >270 paragraphs forces a split.
        transcript = Transcript(paragraphs=_paragraphs(400, words_each=20))
        chunks = chunk_transcript(transcript)
        assert len(chunks) > 1

    def test_split_chunks_overlap_by_one_paragraph(self) -> None:
        transcript = Transcript(paragraphs=_paragraphs(400, words_each=20))
        chunks = chunk_transcript(transcript)
        # Chunk N's last paragraph (20 words) reappears whole as chunk N+1's
        # first paragraph.
        last_paragraph_of_first = " ".join(chunks[0].text.split()[-20:])
        assert chunks[1].text.startswith(last_paragraph_of_first)

    def test_empty_transcript_yields_no_chunks(self) -> None:
        assert chunk_transcript(Transcript(paragraphs=[])) == []


class TestChunkingWithChapters:
    def test_one_chunk_per_chapter(self) -> None:
        paragraphs = [
            Paragraph(start_s=0, text="intro"),
            Paragraph(start_s=60, text="chapter one content"),
            Paragraph(start_s=120, text="chapter two content"),
        ]
        transcript = Transcript(paragraphs=paragraphs)
        chapters = [{"start_time": 0, "title": "Intro"}, {"start_time": 60, "title": "Ch1"}, {"start_time": 120, "title": "Ch2"}]
        chunks = chunk_transcript(transcript, chapters=chapters)
        assert [c.start_s for c in chunks] == [0, 60, 120]

    def test_no_overlap_across_a_chapter_boundary(self) -> None:
        paragraphs = [
            Paragraph(start_s=0, text="alpha beta gamma"),
            Paragraph(start_s=60, text="delta epsilon zeta"),
        ]
        transcript = Transcript(paragraphs=paragraphs)
        chapters = [{"start_time": 0}, {"start_time": 60}]
        chunks = chunk_transcript(transcript, chapters=chapters)
        assert chunks[0].text == "alpha beta gamma"
        assert chunks[1].text == "delta epsilon zeta"  # no repeated words


class FakeLLM:
    """A minimal StructuredLLM stand-in — scripted per tier."""

    def __init__(self, map_results: list[ChunkSummary], reduce_result: VideoDigest) -> None:
        self.map_results = list(map_results)
        self.reduce_result = reduce_result
        self.calls: list[dict[str, Any]] = []

    async def complete_structured(
        self,
        tier: str,
        system: str,
        user: str,
        response_model: type[T],
        *,
        video_id: str | None = None,
        prompt_version: str = "",
    ) -> tuple[T, CallMeta]:
        self.calls.append({"tier": tier, "system": system, "user": user, "video_id": video_id})
        meta = CallMeta(tier=tier, provider="fake", model="fake", latency_ms=1)
        if tier == "map":
            return self.map_results.pop(0), meta  # type: ignore[return-value]
        return self.reduce_result, meta  # type: ignore[return-value]


def _chunk_summary(**overrides: object) -> ChunkSummary:
    fields: dict[str, object] = {"covered": "Something happened.", "claims": [], "entities": []}
    fields.update(overrides)
    return ChunkSummary(**fields)  # type: ignore[arg-type]


def _digest(**overrides: object) -> VideoDigest:
    fields: dict[str, object] = {
        "tldr": "tldr",
        "summary_md": "summary",
        "key_points": ["a point"],
        "relevance": "medium",
    }
    fields.update(overrides)
    return VideoDigest(**fields)  # type: ignore[arg-type]


class TestSummarizeVideo:
    @pytest.mark.asyncio
    async def test_one_map_call_per_chunk_then_one_reduce_call(self) -> None:
        transcript = Transcript(paragraphs=_paragraphs(3))  # one chunk
        llm = FakeLLM([_chunk_summary()], _digest())

        digest = await summarize_video(llm, _meta(), transcript)

        assert digest.tldr == "tldr"
        tiers = [c["tier"] for c in llm.calls]
        assert tiers == ["map", "reduce"]

    @pytest.mark.asyncio
    async def test_multiple_chunks_produce_multiple_map_calls(self) -> None:
        transcript = Transcript(paragraphs=_paragraphs(400, words_each=20))
        chunks = chunk_transcript(transcript)
        llm = FakeLLM([_chunk_summary() for _ in chunks], _digest())

        await summarize_video(llm, _meta(), transcript)

        map_calls = [c for c in llm.calls if c["tier"] == "map"]
        assert len(map_calls) == len(chunks)
        reduce_calls = [c for c in llm.calls if c["tier"] == "reduce"]
        assert len(reduce_calls) == 1

    @pytest.mark.asyncio
    async def test_video_id_is_threaded_through_every_call(self) -> None:
        transcript = Transcript(paragraphs=_paragraphs(3))
        llm = FakeLLM([_chunk_summary()], _digest())
        await summarize_video(llm, _meta(video_id="v42"), transcript)
        assert all(c["video_id"] == "v42" for c in llm.calls)

    @pytest.mark.asyncio
    async def test_reduce_prompt_carries_the_chunk_material(self) -> None:
        transcript = Transcript(paragraphs=_paragraphs(3))
        llm = FakeLLM(
            [_chunk_summary(covered="A specific fact was mentioned.", entities=["Ollama"])],
            _digest(),
        )
        await summarize_video(llm, _meta(), transcript)
        reduce_call = next(c for c in llm.calls if c["tier"] == "reduce")
        assert "A specific fact was mentioned." in reduce_call["user"]
        assert "Ollama" in reduce_call["user"]

    @pytest.mark.asyncio
    async def test_known_topics_reach_the_reduce_prompt(self) -> None:
        """The vault's existing topic vocabulary, threaded through so the
        model reuses "Anthropic" rather than inventing "Anthropic PBC" —
        `clippings_topics/extract.py`'s pattern, ported here."""
        transcript = Transcript(paragraphs=_paragraphs(3))
        llm = FakeLLM([_chunk_summary()], _digest())
        await summarize_video(llm, _meta(), transcript, known_topics=["Anthropic", "OAuth2"])
        reduce_call = next(c for c in llm.calls if c["tier"] == "reduce")
        assert "- Anthropic" in reduce_call["user"]
        assert "- OAuth2" in reduce_call["user"]

    @pytest.mark.asyncio
    async def test_no_known_topics_renders_fine(self) -> None:
        """None (the default) must render like an empty list, not raise —
        StrictUndefined means a template variable simply missing is an error,
        not a blank."""
        transcript = Transcript(paragraphs=_paragraphs(3))
        llm = FakeLLM([_chunk_summary()], _digest())
        await summarize_video(llm, _meta(), transcript)  # known_topics omitted

    @pytest.mark.asyncio
    async def test_empty_transcript_is_rejected_before_any_llm_call(self) -> None:
        llm = FakeLLM([], _digest())
        with pytest.raises(ValueError, match="empty transcript"):
            await summarize_video(llm, _meta(), Transcript(paragraphs=[]))
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_description_is_truncated_before_reaching_the_prompt(self) -> None:
        transcript = Transcript(paragraphs=_paragraphs(3))
        llm = FakeLLM([_chunk_summary()], _digest())
        long_description = "x" * 5000
        await summarize_video(llm, _meta(description=long_description), transcript)
        reduce_call = next(c for c in llm.calls if c["tier"] == "reduce")
        assert len(reduce_call["user"]) < 5000 + 1000  # nowhere near the raw length
