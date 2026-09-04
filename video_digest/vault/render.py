"""S5 note templates (design §5), rendered to markdown ready for
`LiveSyncVault.project`.

Two whole-file templates (plan §1.2 — no markers, each owned outright) plus
one section for `99 topics/`, which does use markers via `vault/notes.py`.

Wikilinks are qualified and existence-checked at the boundary here (skill
§7, §9 of the plan) — callers pass already-resolved `{name: vault_path}`
maps; a name with no entry renders as plain text rather than a dangling
link.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.models import VideoDigest
from ..sources.youtube import VideoMetadata
from ..transcripts.normalize import Transcript
from .notes import yaml_flow_list as _tag_list
from .notes import yaml_str as _yaml_str

Tier = str  # "T0" | "T1" | "T2"


def _wikilink(name: str, links: dict[str, str]) -> str:
    path = links.get(name)
    if path is None:
        return name
    return f"[[{path}|{name}]]"


def _format_timestamp(t_seconds: int) -> str:
    hours, rem = divmod(max(0, t_seconds), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _yt_link(video_id: str, t_seconds: int) -> str:
    return f"https://youtu.be/{video_id}?t={t_seconds}"


def _duration_label(duration_s: int) -> str:
    minutes = duration_s // 60
    if minutes >= 60:
        hours, rem = divmod(minutes, 60)
        return f"{hours}h {rem}m" if rem else f"{hours}h"
    return f"{minutes} min"


@dataclass(slots=True)
class RenderedNote:
    path: str
    markdown: str


def render_transcript_note(
    meta: VideoMetadata, transcript: Transcript, *, tier: Tier, vault_path: str
) -> RenderedNote:
    """`14 video-transcripts/<name>.md` — plain, no wikilinks in the body,
    excluded from graph view by an Obsidian setting on the folder (design §5
    S5). Immutable after S3: a correction is a new file, not an edit."""
    frontmatter = "\n".join(
        [
            "---",
            "type: transcript",
            "tags: [transcript]",
            f"video_id: {meta.video_id}",
            f"tier: {tier}",
            "---",
        ]
    )
    body = f"# {meta.title}\n\n{transcript.text}\n"
    return RenderedNote(path=vault_path, markdown=f"{frontmatter}\n\n{body}")


def render_digest_note(
    meta: VideoMetadata,
    digest: VideoDigest,
    *,
    tier: Tier,
    transcript_tier_degraded: bool,
    asr_model: str | None,
    summary_model: str,
    vault_path: str,
    transcript_vault_path: str,
    topic_links: dict[str, str],
    entity_links: dict[str, str],
) -> RenderedNote:
    channel_url = (
        f"https://www.youtube.com/channel/{meta.channel_id}" if meta.channel_id else None
    )
    channel_label = f"[{meta.channel}]({channel_url})" if channel_url else meta.channel
    watch_url = f"https://www.youtube.com/watch?v={meta.video_id}"
    duration_min = max(1, round(meta.duration_s / 60))

    frontmatter_lines = [
        "---",
        "type: video-digest",
        "source: youtube",
        f"video_id: {meta.video_id}",
        f"url: {watch_url}",
        f"title: {_yaml_str(meta.title)}",
        f"channel: {_yaml_str(meta.channel)}",
        f"published: {meta.upload_date or 'null'}",
        f"duration_min: {duration_min}",
        f"transcript_tier: {tier}",
        f"transcript_tier_degraded: {'true' if transcript_tier_degraded else 'false'}",
        f"asr_model: {asr_model or 'null'}",
        f"summary_model: {_yaml_str(summary_model)}",
        f"relevance: {digest.relevance}",
        f"tags: {_tag_list(['video', f'channel/{meta.channel}'])}",
        "---",
    ]

    key_points = "\n".join(f"- {p}" for p in digest.key_points) or "- (none)"
    highlights = (
        "\n".join(
            f"- [{_format_timestamp(h.t_seconds)}]"
            f"({_yt_link(meta.video_id, h.t_seconds)}) — {h.label}"
            for h in digest.highlights
        )
        or "- (none)"
    )
    entities = ", ".join(_wikilink(e, entity_links) for e in digest.entities) or "(none)"
    claims = "\n".join(f"- {c}" for c in digest.claims_to_verify) or "- (none)"
    actions = "\n".join(f"- [ ] {a}" for a in digest.action_items) or "- [ ] (none)"
    topics_line = (
        ", ".join(_wikilink(t, topic_links) for t in digest.topics) if digest.topics else ""
    )

    body_parts = [
        f"# {meta.title}",
        "",
        f"**{channel_label}** · {_duration_label(meta.duration_s)} · "
        f"{meta.upload_date or 'undated'} · [Watch]({watch_url})",
        "",
        "> [!abstract] TL;DR",
        f"> {digest.tldr}",
        "",
        "## Summary",
        digest.summary_md,
        "",
        "## Key points",
        key_points,
        "",
        "## Highlights",
        highlights,
        "",
        "## Entities",
        entities,
        "",
        "## Claims to verify",
        claims,
        "",
        "## Action items",
        actions,
        "",
        "---",
        f"Transcript: [[{transcript_vault_path}|Transcript]] · tier {tier}",
    ]
    if topics_line:
        body_parts.insert(body_parts.index("---"), f"Topics: {topics_line}\n")

    markdown = "\n".join(frontmatter_lines) + "\n\n" + "\n".join(body_parts) + "\n"
    return RenderedNote(path=vault_path, markdown=markdown)
