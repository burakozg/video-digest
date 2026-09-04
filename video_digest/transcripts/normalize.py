"""S3: one internal `Transcript` shape, whatever tier produced it (design §5).

The VTT-syntax regexes below (timestamp lines, NOTE/STYLE/REGION blocks, cue
tags, cue-id lines) are the same format facts `podcast_agent/transcripts/
normalize.py` encodes, but that module discards timing entirely — podcast
episodes need none. Video notes are built around `?t=seconds` deep links
(design §5 S5), so this parser keeps a start time per cue and per paragraph
throughout; nothing here is a port.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_VTT_TIMESTAMP = re.compile(
    r"^\s*(?P<start>(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
    r"(?P<end>(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3})"
)
_VTT_NOTE = re.compile(r"^\s*(NOTE|STYLE|REGION)\b")
#: Inline cue markup: <v Speaker>, <c.colour>, <00:00:01.000>, <i>.
_CUE_TAGS = re.compile(r"</?[a-zA-Z0-9.:_ -]*>")
#: Caption artefacts in square/round brackets: [Music], [Applause], (laughs).
#: Matched as a standalone token anywhere in the cue, not just a whole-cue
#: match — a rolling cue can carry one mid-stream, e.g. "...practice. [Music]".
_BRACKETED_TOKEN = re.compile(r"(?:^|(?<=\s))[\[(][^\])]{0,40}[\])](?=\s|$)")
#: A leading speaker sigil: "SPEAKER:", ">> Speaker:", "- Speaker:".
_SPEAKER_SIGIL = re.compile(r"^(?:>>\s*|-\s*)?[A-Z][A-Za-z0-9 .'-]{0,40}:\s*")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")

#: Paragraphs target 60-90s of speech (design §5 S3).
PARAGRAPH_TARGET_S = 60.0
PARAGRAPH_HARD_CAP_S = 90.0


def _parse_timestamp(raw: str) -> float:
    parts = raw.replace(",", ".").split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


@dataclass(slots=True)
class Cue:
    start_s: float
    end_s: float
    text: str


def parse_vtt(text: str) -> list[Cue]:
    """WebVTT only — the shape yt-dlp writes for both T0 (`--write-subs`) and
    T1 (`--write-auto-subs`). SRT/JSON are out of scope here: unlike
    podcast-digest's transcript ladder, this app never receives an arbitrary
    publisher format — yt-dlp normalises everything to VTT on the way out.
    """
    lines = text.splitlines()
    cues: list[Cue] = []
    i = 0
    skip_block = False
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            skip_block = False
            continue
        if line.startswith("WEBVTT"):
            continue
        if _VTT_NOTE.match(line):
            skip_block = True
            continue
        if skip_block:
            continue
        # A bare cue identifier line, immediately followed by a timestamp.
        if i < len(lines) and _VTT_TIMESTAMP.match(lines[i]) and " " not in line:
            continue
        match = _VTT_TIMESTAMP.match(line)
        if not match:
            continue
        start_s = _parse_timestamp(match.group("start"))
        end_s = _parse_timestamp(match.group("end"))
        body_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            body_lines.append(lines[i].strip())
            i += 1
        cleaned = _clean_cue_text(" ".join(body_lines))
        if cleaned:
            cues.append(Cue(start_s=start_s, end_s=end_s, text=cleaned))
    return cues


def _clean_cue_text(raw: str) -> str:
    text = _CUE_TAGS.sub("", raw).strip()
    text = _BRACKETED_TOKEN.sub(" ", text).strip()
    text = _SPEAKER_SIGIL.sub("", text).strip()
    return re.sub(r"\s+", " ", text)


def _dedupe_rolling(cues: list[Cue]) -> list[tuple[float, str]]:
    """Collapse YouTube auto-caption rolling duplication (design §5 S3).

    Auto-captions scroll: each cue commonly repeats the tail of the previous
    cue's words with a few new ones appended — not always byte-identical, so
    dropping only exact repeats (podcast-digest's `_normalize_cues`) leaves
    most of the duplication in place. This finds the longest run of the
    previous cue's trailing words that matches the new cue's leading words,
    word-by-word, and keeps only what the new cue adds.

    Returns `(start_s, added_text)` pairs — one per cue that added anything.
    """
    fragments: list[tuple[float, str]] = []
    prev_words: list[str] = []
    for cue in cues:
        words = cue.text.split()
        if not words:
            continue
        max_overlap = min(len(prev_words), len(words))
        overlap = 0
        for candidate in range(max_overlap, 0, -1):
            if prev_words[-candidate:] == words[:candidate]:
                overlap = candidate
                break
        new_words = words[overlap:]
        if new_words:
            fragments.append((cue.start_s, " ".join(new_words)))
        prev_words = words
    return fragments


@dataclass(slots=True)
class Paragraph:
    start_s: int
    text: str


@dataclass(slots=True)
class Transcript:
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)


def _chapter_starts(chapters: list[dict[str, Any]] | None) -> list[float]:
    if not chapters:
        return []
    return sorted({float(c["start_time"]) for c in chapters if "start_time" in c})


def build_transcript(
    vtt_text: str, *, chapters: list[dict[str, Any]] | None = None
) -> Transcript:
    """VTT text -> `Transcript`: dedupe, merge into sentences, merge into
    60-90s paragraphs (or chapter-aligned segments when chapters exist), each
    keeping a start timestamp for the note's deep links.
    """
    cues = parse_vtt(vtt_text)
    fragments = _dedupe_rolling(cues)
    return Transcript(paragraphs=merge_paragraphs(fragments, chapters=chapters))


def merge_paragraphs(
    fragments: list[tuple[float, str]], *, chapters: list[dict[str, Any]] | None = None
) -> list[Paragraph]:
    """Merge deduped `(start_s, text)` fragments into 60-90s paragraphs,
    breaking early at a sentence boundary once the target is reached, at the
    hard cap regardless, or at a chapter start when chapters exist (design
    §5 S3 — "chapters are the best available structural signal"). Separated
    from `build_transcript` so this — the part with actual judgment calls —
    is testable without parsing VTT at all.
    """
    if not fragments:
        return []

    boundaries = _chapter_starts(chapters)

    paragraphs: list[Paragraph] = []
    para_start = fragments[0][0]
    para_words: list[str] = []
    next_boundary_idx = 0
    # Skip any chapter boundary at or before the very first fragment — it
    # marks the start we are already at, not a break to make later.
    while next_boundary_idx < len(boundaries) and boundaries[next_boundary_idx] <= para_start:
        next_boundary_idx += 1

    def _flush() -> None:
        if para_words:
            paragraphs.append(Paragraph(start_s=int(para_start), text=" ".join(para_words)))

    for start_s, text in fragments:
        at_chapter_boundary = (
            next_boundary_idx < len(boundaries) and start_s >= boundaries[next_boundary_idx]
        )
        elapsed = start_s - para_start
        at_sentence_end = bool(para_words) and _SENTENCE_END.search(para_words[-1])

        should_break = para_words and (
            at_chapter_boundary
            or elapsed >= PARAGRAPH_HARD_CAP_S
            or (elapsed >= PARAGRAPH_TARGET_S and at_sentence_end)
        )
        if should_break:
            _flush()
            para_words = []
            para_start = start_s
            if at_chapter_boundary:
                next_boundary_idx += 1

        para_words.append(text)

    _flush()
    return paragraphs
