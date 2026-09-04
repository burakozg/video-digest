from __future__ import annotations

import pytest

from video_digest.llm.prompts import PromptError, load_prompt


class TestShippedPrompts:
    def test_map_v1_loads_and_renders(self) -> None:
        prompt = load_prompt("map", "v1")
        system, user = prompt.render(
            index=1, total=3, title="A Video", start_seconds=0, content="Some text."
        )
        assert "extracting raw material" in system.lower()
        assert "A Video" in user
        assert "Some text." in user
        assert "<video_slice" in user

    def test_reduce_v1_loads_and_renders(self) -> None:
        prompt = load_prompt("reduce", "v1")
        system, user = prompt.render(
            title="A Video",
            channel="A Channel",
            duration_seconds=120,
            description="",
            chunk_summaries="[0s] Some material.",
            known_topics=[],
        )
        assert "tldr" in system
        assert "A Video" in user
        assert "A Channel" in user
        # Nothing supplied -> no dangling "Topics and entities..." header.
        assert "already in use" not in user

    def test_reduce_v1_lists_known_topics_when_given_any(self) -> None:
        _, user = load_prompt("reduce", "v1").render(
            title="A Video",
            channel="A Channel",
            duration_seconds=120,
            description="",
            chunk_summaries="[0s] Some material.",
            known_topics=["Anthropic", "OAuth2"],
        )
        assert "- Anthropic" in user
        assert "- OAuth2" in user

    def test_untrusted_content_is_explicitly_flagged_in_the_system_prompt(self) -> None:
        """The design's prompt-injection defence (§5 S4) must survive a
        careless prompt edit — pinned here rather than only reviewed by eye."""
        for name in ("map", "reduce"):
            system, _ = load_prompt(name, "v1").render(
                index=1,
                total=1,
                title="t",
                start_seconds=0,
                content="c",
                channel="c",
                duration_seconds=1,
                description="",
                chunk_summaries="",
                known_topics=[],
            )
            assert "untrusted" in system.lower()
            assert "do not comply" in system.lower()


class TestMissingPrompt:
    def test_raises_a_clear_error(self) -> None:
        with pytest.raises(PromptError, match="no prompt file"):
            load_prompt("does-not-exist", "v1")


class TestMissingTemplateVariable:
    def test_raises_rather_than_rendering_blank(self) -> None:
        prompt = load_prompt("map", "v1")
        with pytest.raises(PromptError):
            prompt.render(index=1, total=1, title="t")  # missing start_seconds, content
