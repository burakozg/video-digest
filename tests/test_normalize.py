from __future__ import annotations

from pathlib import Path

from video_digest.transcripts.normalize import (
    Paragraph,
    _dedupe_rolling,
    build_transcript,
    merge_paragraphs,
    parse_vtt,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestRollingDuplicateRemoval:
    """The single highest-value test in this milestone (plan §M2): a real
    auto-caption VTT, deduplicated on word-level overlap rather than exact
    line equality."""

    def test_real_rolling_caption_fixture_has_no_duplicated_words(self) -> None:
        vtt = (FIXTURES / "rolling_captions.vtt").read_text()
        transcript = build_transcript(vtt)
        text = transcript.text
        assert (
            text
            == "today we're going to talk about machine learning and how it "
            "works in practice. Let's start with the basics."
        )
        # The naive (exact-line-dedup) reading would repeat "today we're
        # going to talk about" and "machine learning and how it" — assert
        # they appear exactly once.
        assert text.count("today we're going to talk about") == 1
        assert text.count("machine learning and how it") == 1

    def test_dedupe_drops_a_partial_word_overlap_not_just_exact_lines(self) -> None:
        cues = parse_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "the quick brown fox\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "brown fox jumps over\n"
        )
        fragments = _dedupe_rolling(cues)
        assert fragments == [
            (0.0, "the quick brown fox"),
            (2.0, "jumps over"),
        ]

    def test_no_overlap_keeps_both_cues_whole(self) -> None:
        cues = parse_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "hello there\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "completely different words\n"
        )
        fragments = _dedupe_rolling(cues)
        assert fragments == [(0.0, "hello there"), (2.0, "completely different words")]


class TestVTTParsing:
    def test_strips_cue_tags_and_inline_timestamps(self) -> None:
        cues = parse_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "hello<00:00:00.500><c> there</c><c> friend</c>\n"
        )
        assert cues[0].text == "hello there friend"

    def test_strips_note_style_region_blocks(self) -> None:
        cues = parse_vtt(
            "WEBVTT\n\n"
            "STYLE\n"
            "::cue { color: yellow; }\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "spoken text\n"
        )
        assert [c.text for c in cues] == ["spoken text"]

    def test_skips_a_bare_cue_identifier_line(self) -> None:
        cues = parse_vtt(
            "WEBVTT\n\n"
            "cue-1\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "spoken text\n"
        )
        assert [c.text for c in cues] == ["spoken text"]

    def test_strips_bracketed_artefacts_standalone_and_embedded(self) -> None:
        cues = parse_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "[Music]\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "hello there (laughs) friend\n"
        )
        # The pure-artefact cue produces no cue at all (empty after cleaning).
        assert [c.text for c in cues] == ["hello there friend"]

    def test_strips_a_leading_speaker_sigil(self) -> None:
        cues = parse_vtt(
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n>> Host: welcome back\n"
        )
        assert cues[0].text == "welcome back"

    def test_timestamps_with_hours_parse_correctly(self) -> None:
        cues = parse_vtt(
            "WEBVTT\n\n01:02:03.500 --> 01:02:05.000\nlate in the video\n"
        )
        assert cues[0].start_s == 3723.5


class TestParagraphMerging:
    def test_single_short_transcript_is_one_paragraph(self) -> None:
        paragraphs = merge_paragraphs([(0.0, "hello"), (1.0, "world")])
        assert paragraphs == [Paragraph(start_s=0, text="hello world")]

    def test_breaks_at_sentence_end_once_target_reached(self) -> None:
        fragments = [
            (0.0, "First sentence."),
            (65.0, "Second sentence."),
        ]
        paragraphs = merge_paragraphs(fragments)
        assert len(paragraphs) == 2
        assert paragraphs[0] == Paragraph(start_s=0, text="First sentence.")
        assert paragraphs[1] == Paragraph(start_s=65, text="Second sentence.")

    def test_does_not_break_mid_sentence_before_the_hard_cap(self) -> None:
        fragments = [
            (0.0, "this sentence"),
            (65.0, "keeps going without a period"),
        ]
        paragraphs = merge_paragraphs(fragments)
        assert len(paragraphs) == 1
        assert "keeps going" in paragraphs[0].text

    def test_hard_cap_breaks_even_mid_sentence(self) -> None:
        fragments = [
            (0.0, "this sentence"),
            (95.0, "keeps going without a period"),
        ]
        paragraphs = merge_paragraphs(fragments)
        assert len(paragraphs) == 2
        assert paragraphs[1].start_s == 95

    def test_chapter_boundary_forces_a_break_even_early(self) -> None:
        fragments = [(0.0, "Intro."), (10.0, "Still intro.")]
        paragraphs = merge_paragraphs(
            fragments, chapters=[{"start_time": 10.0, "title": "Chapter 2"}]
        )
        assert len(paragraphs) == 2
        assert paragraphs[0] == Paragraph(start_s=0, text="Intro.")
        assert paragraphs[1] == Paragraph(start_s=10, text="Still intro.")

    def test_chapter_at_or_before_the_first_fragment_is_not_a_spurious_break(self) -> None:
        fragments = [(5.0, "Intro."), (10.0, "More.")]
        paragraphs = merge_paragraphs(
            fragments, chapters=[{"start_time": 0.0, "title": "Chapter 1"}]
        )
        assert len(paragraphs) == 1

    def test_empty_fragments_yield_no_paragraphs(self) -> None:
        assert merge_paragraphs([]) == []
