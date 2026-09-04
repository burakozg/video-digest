"""S4: two-pass summarisation (design §5 S4).

Pass A (map) chunks the transcript by chapter, or by ~8k tokens with a
paragraph-level overlap when chapters are absent, and reduces each chunk to
raw material (`ChunkSummary`). Pass B (reduce) merges every chunk's material
into one `VideoDigest`. Tier routing is by name (`map`/`reduce`) — Pass A is
the volume job suited to a local model, Pass B is quality-sensitive — and
both go through the same `StructuredLLM` boundary, so this module never
imports litellm or instructor directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..llm.base import StructuredLLM
from ..llm.models import ChunkSummary, VideoDigest
from ..llm.prompts import load_prompt
from ..logging_setup import get_logger
from ..sources.youtube import VideoMetadata
from ..transcripts.normalize import Paragraph, Transcript

log = get_logger(__name__)

#: Matches design §5 S4's "~8k tokens with 200-token overlap when chapters
#: are absent". A rough chars-per-token ratio, adequate here (deciding chunk
#: boundaries, not billing) and avoiding a tokenizer dependency that would
#: differ per model anyway.
CHARS_PER_TOKEN = 4
CHUNK_TARGET_TOKENS = 8000
CHUNK_OVERLAP_TOKENS = 200

#: Third-party text (design §5 S4's prompt-injection note): capped before it
#: reaches a prompt, the same guard podcast-digest applies to descriptions.
MAX_DESCRIPTION_CHARS = 2000


@dataclass(slots=True)
class Chunk:
    start_s: int
    text: str


def chunk_transcript(
    transcript: Transcript, *, chapters: list[dict[str, Any]] | None = None
) -> list[Chunk]:
    """Group paragraphs into chunks.

    With chapters: one chunk per chapter (paragraph boundaries already sit on
    chapter starts — S3's own chapter alignment, `normalize.py`'s
    `merge_paragraphs` — so this only has to group what is already broken
    there, never re-detect the boundary).

    Without chapters: token-budget grouping, breaking once a chunk would
    exceed `CHUNK_TARGET_TOKENS`, carrying the last paragraph of the
    finished chunk into the next one as an overlap — paragraph granularity
    rather than an exact token count, since paragraphs (60-90s each) are
    already close to the overlap target.
    """
    paragraphs = transcript.paragraphs
    if not paragraphs:
        return []

    chapter_starts = sorted(
        {int(c["start_time"]) for c in (chapters or []) if "start_time" in c}
    )
    target_chars = CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN

    chunks: list[Chunk] = []
    current: list[Paragraph] = []
    next_boundary_idx = 0
    while (
        next_boundary_idx < len(chapter_starts)
        and chapter_starts[next_boundary_idx] <= paragraphs[0].start_s
    ):
        next_boundary_idx += 1

    def flush() -> None:
        if current:
            chunks.append(
                Chunk(start_s=current[0].start_s, text=" ".join(p.text for p in current))
            )

    for paragraph in paragraphs:
        at_chapter_boundary = (
            next_boundary_idx < len(chapter_starts)
            and paragraph.start_s >= chapter_starts[next_boundary_idx]
        )
        current_chars = sum(len(p.text) for p in current)
        over_budget = bool(current) and current_chars + len(paragraph.text) > target_chars

        if current and (at_chapter_boundary or over_budget):
            flush()
            # Overlap only on a token-budget break — a chapter break is a
            # real structural boundary, not a mid-thought split, so nothing
            # needs to carry across it. This applies even inside a video
            # that has chapters: a single chapter long enough to need
            # sub-splitting on its own still benefits from the overlap.
            current = current[-1:] if over_budget and not at_chapter_boundary else []
            if at_chapter_boundary:
                next_boundary_idx += 1

        current.append(paragraph)
    flush()
    return chunks


def _render_chunk_material(chunks: list[Chunk], summaries: list[ChunkSummary]) -> str:
    lines: list[str] = []
    for chunk, summary in zip(chunks, summaries, strict=True):
        if not summary.covered:
            continue
        lines.append(f"[{chunk.start_s}s] {summary.covered}")
        for claim in summary.claims:
            lines.append(f"  claim: {claim}")
        if summary.entities:
            lines.append(f"  entities: {', '.join(summary.entities)}")
        if summary.quotable_t_seconds is not None:
            lines.append(f"  notable moment: {summary.quotable_t_seconds}s")
    return "\n".join(lines)


async def summarize_video(
    llm: StructuredLLM,
    meta: VideoMetadata,
    transcript: Transcript,
    *,
    known_topics: list[str] | None = None,
    map_prompt_version: str = "v1",
    reduce_prompt_version: str = "v1",
) -> VideoDigest:
    chunks = chunk_transcript(transcript, chapters=meta.chapters)
    if not chunks:
        raise ValueError(f"video {meta.video_id}: empty transcript, nothing to summarise")

    map_prompt = load_prompt("map", map_prompt_version)
    summaries: list[ChunkSummary] = []
    for index, chunk in enumerate(chunks):
        system, user = map_prompt.render(
            index=index + 1,
            total=len(chunks),
            title=meta.title,
            start_seconds=chunk.start_s,
            content=chunk.text,
        )
        result, _ = await llm.complete_structured(
            "map",
            system,
            user,
            ChunkSummary,
            video_id=meta.video_id,
            prompt_version=map_prompt.versioned_name,
        )
        summaries.append(result)
        log.info(
            "summarize.chunk_done", video_id=meta.video_id, chunk=index + 1, total=len(chunks)
        )

    reduce_prompt = load_prompt("reduce", reduce_prompt_version)
    system, user = reduce_prompt.render(
        title=meta.title,
        channel=meta.channel,
        duration_seconds=meta.duration_s,
        description=(meta.description or "")[:MAX_DESCRIPTION_CHARS],
        chunk_summaries=_render_chunk_material(chunks, summaries),
        known_topics=known_topics or [],
    )
    digest, _ = await llm.complete_structured(
        "reduce",
        system,
        user,
        VideoDigest,
        video_id=meta.video_id,
        prompt_version=reduce_prompt.versioned_name,
    )
    log.info("summarize.done", video_id=meta.video_id, chunks=len(chunks))
    return digest
