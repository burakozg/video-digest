from __future__ import annotations

from video_digest.llm.models import Highlight, VideoDigest
from video_digest.sources.youtube import VideoMetadata
from video_digest.transcripts.normalize import Paragraph, Transcript
from video_digest.vault.render import render_digest_note, render_transcript_note

VID = "dQw4w9WgXcQ"


def _meta(**overrides: object) -> VideoMetadata:
    fields: dict[str, object] = {
        "video_id": VID,
        "title": "A Video About Ollama",
        "channel": "A Channel",
        "channel_id": "UCxxxx",
        "duration_s": 47 * 60,
        "upload_date": "2026-08-20",
        "description": "d",
    }
    fields.update(overrides)
    return VideoMetadata(**fields)  # type: ignore[arg-type]


def _digest(**overrides: object) -> VideoDigest:
    fields: dict[str, object] = {
        "tldr": "A short summary.",
        "summary_md": "Longer summary text.",
        "key_points": ["Point one", "Point two"],
        "highlights": [Highlight(t_seconds=724, label="A key moment")],
        "entities": ["Ollama", "Anthropic"],
        "topics": ["ollama", "local-llms"],
        "claims_to_verify": ["Some claim"],
        "action_items": ["Do a thing"],
        "relevance": "high",
    }
    fields.update(overrides)
    return VideoDigest(**fields)  # type: ignore[arg-type]


class TestDigestNoteFrontmatter:
    def test_shape(self) -> None:
        note = render_digest_note(
            _meta(),
            _digest(),
            tier="T1",
            transcript_tier_degraded=False,
            asr_model=None,
            summary_model="openrouter/anthropic/claude-haiku-4.5",
            vault_path="13 video-summaries/2026-08-20-a-video-about-ollama.md",
            transcript_vault_path="14 video-transcripts/2026-08-20-a-video-about-ollama.md",
            topic_links={},
            entity_links={},
        )
        assert note.markdown.startswith("---\ntype: video-digest\n")
        assert f"video_id: {VID}" in note.markdown
        assert 'title: "A Video About Ollama"' in note.markdown
        assert "transcript_tier: T1" in note.markdown
        assert "transcript_tier_degraded: false" in note.markdown
        assert "asr_model: null" in note.markdown
        assert "relevance: high" in note.markdown
        assert "published: 2026-08-20" in note.markdown
        assert "duration_min: 47" in note.markdown

    def test_quotes_and_escapes_untrusted_title_text(self) -> None:
        note = render_digest_note(
            _meta(title='A "Quoted" Title'),
            _digest(),
            tier="T0",
            transcript_tier_degraded=False,
            asr_model=None,
            summary_model="m",
            vault_path="x.md",
            transcript_vault_path="y.md",
            topic_links={},
            entity_links={},
        )
        assert 'title: "A \\"Quoted\\" Title"' in note.markdown

    def test_asr_model_set_only_for_t2(self) -> None:
        note = render_digest_note(
            _meta(),
            _digest(),
            tier="T2",
            transcript_tier_degraded=False,
            asr_model="large-v3",
            summary_model="m",
            vault_path="x.md",
            transcript_vault_path="y.md",
            topic_links={},
            entity_links={},
        )
        assert "asr_model: large-v3" in note.markdown


class TestDeepLinks:
    def test_highlight_becomes_a_timestamped_youtube_link(self) -> None:
        note = render_digest_note(
            _meta(),
            _digest(),
            tier="T0",
            transcript_tier_degraded=False,
            asr_model=None,
            summary_model="m",
            vault_path="x.md",
            transcript_vault_path="y.md",
            topic_links={},
            entity_links={},
        )
        assert f"https://youtu.be/{VID}?t=724" in note.markdown
        assert "[12:04](" in note.markdown  # 724s = 12:04

    def test_hour_long_timestamps_include_the_hour(self) -> None:
        note = render_digest_note(
            _meta(),
            _digest(highlights=[Highlight(t_seconds=3723, label="late")]),
            tier="T0",
            transcript_tier_degraded=False,
            asr_model=None,
            summary_model="m",
            vault_path="x.md",
            transcript_vault_path="y.md",
            topic_links={},
            entity_links={},
        )
        assert "[1:02:03](" in note.markdown


class TestWikilinksOnlyToKnownTargets:
    def test_entity_with_a_target_becomes_a_qualified_wikilink(self) -> None:
        note = render_digest_note(
            _meta(),
            _digest(entities=["Ollama"]),
            tier="T0",
            transcript_tier_degraded=False,
            asr_model=None,
            summary_model="m",
            vault_path="x.md",
            transcript_vault_path="y.md",
            topic_links={},
            entity_links={"Ollama": "99 topics/ollama"},
        )
        assert "[[99 topics/ollama|Ollama]]" in note.markdown

    def test_entity_with_no_target_is_plain_text(self) -> None:
        note = render_digest_note(
            _meta(),
            _digest(entities=["Some Obscure Thing"]),
            tier="T0",
            transcript_tier_degraded=False,
            asr_model=None,
            summary_model="m",
            vault_path="x.md",
            transcript_vault_path="y.md",
            topic_links={},
            entity_links={},
        )
        assert "Some Obscure Thing" in note.markdown
        assert "[[Some Obscure Thing" not in note.markdown

    def test_topic_links_are_qualified_too(self) -> None:
        note = render_digest_note(
            _meta(),
            _digest(topics=["ollama"]),
            tier="T0",
            transcript_tier_degraded=False,
            asr_model=None,
            summary_model="m",
            vault_path="x.md",
            transcript_vault_path="y.md",
            topic_links={"ollama": "99 topics/ollama"},
            entity_links={},
        )
        assert "[[99 topics/ollama|ollama]]" in note.markdown


class TestTranscriptNote:
    def test_shape_has_no_wikilinks_and_the_right_frontmatter(self) -> None:
        transcript = Transcript(paragraphs=[Paragraph(start_s=0, text="Hello world.")])
        note = render_transcript_note(
            _meta(), transcript, tier="T1", vault_path="14 video-transcripts/x.md"
        )
        assert "type: transcript" in note.markdown
        assert "tags: [transcript]" in note.markdown
        assert "tier: T1" in note.markdown
        assert "Hello world." in note.markdown
        assert "[[" not in note.markdown
