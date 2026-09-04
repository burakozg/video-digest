"""S5: vault write (design §5). Orchestrates what M3 built: pin filenames,
resolve topic/entity links against the vault, render, project.

`write_video_note` takes a fully-formed `VideoDigest` and `Transcript` as
arguments rather than reading them from the DB itself: `pipeline/runner.py`
rebuilds both from the stored row and passes them in, and tests exercise S5
with a fixture digest and no LLM call (plan §M3's "done when" — a note
appears and re-runs write nothing).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass

import yaml

from .. import sanitize
from ..config import VaultConfig
from ..llm.models import VideoDigest
from ..logging_setup import get_logger
from ..sources.youtube import VideoMetadata
from ..transcripts.normalize import Transcript
from ..utils import epoch_ms, iso_now
from ..vault.livesync import Entry, LiveSyncVault
from ..vault.names import (
    adopt_topic_filename,
    get_topic_filename,
    resolve_note_filename,
    resolve_topic_filename,
)
from ..vault.notes import (
    KEY_PREFIX,
    OWNER,
    split_frontmatter,
    wrap,
    yaml_flow_list,
    yaml_str,
)
from ..vault.render import Tier, render_digest_note, render_transcript_note

log = get_logger(__name__)

#: `99 topics/` page reads during one index sweep are fanned out this many at
#: a time — matches clippings-topics' fan-out bound (its LLM batch size).
_TOPIC_READ_CONCURRENCY = 8


def count_topic_mentions(db: sqlite3.Connection, topic_key: str) -> int:
    """How many stored digests mention this canonical topic — the floor for
    creating a page (design §5 S5 — "topic_creation_threshold").

    Scans every stored digest rather than maintaining a counter table: this
    is a personal vault (tens to low hundreds of videos), and a scan avoids
    a second source of truth that a missed update could drift from.
    """
    count = 0
    for row in db.execute("SELECT digest FROM videos WHERE digest IS NOT NULL"):
        try:
            digest = json.loads(row["digest"])
        except (json.JSONDecodeError, TypeError):
            continue
        names = set(digest.get("topics") or []) | set(digest.get("entities") or [])
        if any(sanitize.canonical(n) == topic_key for n in names):
            count += 1
    return count


def _frontmatter_value(lines: list[str], key: str) -> str:
    """A frontmatter key's raw value, or "" if absent — same line-by-line
    approach as split_frontmatter itself: a YAML parser would reformat
    another writer's or a human's frontmatter as the price of reading one
    key."""
    for line in lines:
        line_key, _, value = line.partition(":")
        if line_key.strip() == key:
            return value.strip()
    return ""


def _scalar_value(raw: str) -> str:
    """Unquote a frontmatter scalar we serialised with `notes.yaml_str` — the
    read side of the same round-trip. A YAML error (a human's odd quoting)
    falls back to the old naive strip. Comparison-only; never rewrites
    another writer's frontmatter."""
    raw = raw.strip()
    if not raw:
        return ""
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw.strip('"').strip("'")
    return str(parsed) if parsed is not None else ""


def _parse_list_value(value: str) -> list[str]:
    """`[a, b, c]` -> `["a", "b", "c"]`, comma/quote/bracket-aware via a YAML
    flow-scalar parse. The flow-style list convention this vault's frontmatter
    already uses for `tags:` (see ~/.claude/skills/obsidian-vault-writer).
    Empty/malformed input -> []."""
    value = value.strip()
    if not value:
        return []
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


@dataclass(frozen=True, slots=True)
class TopicPage:
    """One `99 topics/` page as the vocabulary hint and the alias matcher
    need it."""

    slug: str  #: lowercased filename stem, no ".md"
    title: str  #: frontmatter `title:`, or the stem when there is none
    #: sanitize.canonical() of every `aliases:` + `video_aliases:` entry.
    alias_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class TopicIndex:
    """The `99 topics/` folder, swept once (`load_topic_index`).

    `slugs` comes from the raw listing — every live entry, a torn page
    included — so slug-matching is byte-for-byte what it was before this
    existed. `pages` (the title and alias source) carries only pages that
    reassembled whole, matching the old per-read code that skipped a `None`.
    """

    slugs: frozenset[str]
    pages: tuple[TopicPage, ...]

    def alias_slug(self, key: str) -> str | None:
        if not key:
            return None
        return next((p.slug for p in self.pages if key in p.alias_keys), None)

    def titles(self) -> list[str]:
        return sorted({p.title for p in self.pages}, key=str.lower)


def _topic_aliases(front: list[str]) -> frozenset[str]:
    """Canonical keys for every alias on a topic page. Two sources:
    `aliases` is the generic unprefixed key any human or sibling writer
    could someday populate (nothing does yet — honouring it costs nothing
    and needs no code change the day something does); `video_aliases` is
    this writer's own accumulated record (see `_update_topic_page`)."""
    raw = _parse_list_value(_frontmatter_value(front, "aliases"))
    raw += _parse_list_value(_frontmatter_value(front, f"{KEY_PREFIX}aliases"))
    return frozenset(k for k in (sanitize.canonical(a) for a in raw) if k)


async def load_topic_index(vault: LiveSyncVault, cfg: VaultConfig) -> TopicIndex:
    """Sweep `99 topics/` once. Per-page reads fan out
    `_TOPIC_READ_CONCURRENCY` at a time rather than one-by-one — this runs on
    the S4 critical path (the reduce prompt's vocabulary hint) and replaces
    the read amplification `_alias_match` used to cause in S5 (a re-read of
    every page per unmatched name).

    A `VaultUnavailable` mid-sweep propagates: the S5 caller turns it into a
    deferred job, the S4 caller into an empty hint — both pre-existing
    outcomes.
    """
    prefix = f"{cfg.topics_dir}/"
    entries = await vault.list_prefix(prefix)
    slugs = frozenset(e.path[len(prefix) :].removesuffix(".md").lower() for e in entries)

    gate = asyncio.Semaphore(_TOPIC_READ_CONCURRENCY)

    async def _load(entry: Entry) -> TopicPage | None:
        async with gate:
            markdown = await vault.read(entry)
        if markdown is None:
            return None  # torn note: skip rather than guess, as before
        stem = entry.path[len(prefix) :].removesuffix(".md")
        front, _ = split_frontmatter(markdown)
        return TopicPage(
            slug=stem.lower(),
            title=_scalar_value(_frontmatter_value(front, "title")) or stem,
            alias_keys=_topic_aliases(front),
        )

    loaded: list[TopicPage | None] = await asyncio.gather(*(_load(e) for e in entries))
    return TopicIndex(slugs=slugs, pages=tuple(p for p in loaded if p is not None))


async def fetch_known_topics(vault: LiveSyncVault, cfg: VaultConfig) -> list[str]:
    """Every `99 topics/` page's title, for the reduce prompt's vocabulary
    hint (`clippings_topics`' pattern — "the vault's existing topic names are
    given to the model as the preferred vocabulary"). Without it the model
    invents a new near-spelling every time ("Anthropic PBC", "anthropic.com")
    that neither `sanitize.canonical` (deliberately conservative — it does not
    know "PBC" the way it knows "Inc") nor filename-slug matching will ever
    reconcile with an existing page.

    Raises rather than swallows a vault error: the S4 caller
    (`pipeline/runner.py`) is where "an unreachable vault degrades the hint,
    it does not fail the video" lives.
    """
    return (await load_topic_index(vault, cfg)).titles()


def resolve_topic_links(
    db: sqlite3.Connection,
    cfg: VaultConfig,
    names: list[str],
    index: TopicIndex,
) -> dict[str, str]:
    """`{display name: vault path}` for names that earn a link. Does no I/O —
    slug, alias and title all come from `index` (swept once by
    `load_topic_index`, shared by both the `topics` and `entities` passes).

    Matching against "existing filenames ... first" (design §5 S5): a
    filename-slug match, then a match against a page's known aliases —
    `video_aliases:` (this writer's own, see `_update_topic_page`) or the
    generic `aliases:`.

    A name earns a link when: an existing `99 topics/` page matches its slug
    (adopted, never re-slugified independently) or one of its known aliases,
    or it has `topic_creation_threshold` prior mentions and a new page is
    created. Anything else is left as plain text — under-linking is
    invisible, a page of dangling links is not (skill §7).
    """
    links: dict[str, str] = {}
    seen_keys: set[str] = set()
    for name in names:
        key = sanitize.canonical(name)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)

        filename = get_topic_filename(db, key)
        if filename is None:
            slug = sanitize.slugify(name)
            if slug in index.slugs:
                adopt_topic_filename(db, key, slug)
                filename = slug
            elif alias_slug := index.alias_slug(key):
                adopt_topic_filename(db, key, alias_slug)
                filename = alias_slug
            else:
                mentions = count_topic_mentions(db, key)
                if mentions < cfg.topic_creation_threshold:
                    continue
                filename = resolve_topic_filename(db, key, title=name)
        links[name] = f"{cfg.topics_dir}/{filename}"
    return links


async def write_video_note(
    db: sqlite3.Connection,
    vault: LiveSyncVault,
    cfg: VaultConfig,
    meta: VideoMetadata,
    digest: VideoDigest,
    transcript: Transcript,
    *,
    tier: Tier,
    transcript_tier_degraded: bool = False,
    asr_model: str | None = None,
    summary_model: str = "",
    adapter: str = "youtube",
) -> tuple[str, str]:
    """Write the digest note and the transcript note, update topic pages,
    and return `(note_path, transcript_path)`.

    Both notes are whole-file (plan §1.2): filenames are pinned once via
    `vault/names.py` and never recomputed, so re-runs — a re-summarise, a
    template change — always land on the same file.
    """
    # note_names.video_id carries a foreign key to videos.id (the
    # "<adapter>:<video_id>" composite pipeline/resolve.py uses as its
    # primary key), not the bare video id — a row must already exist there.
    row_id = f"{adapter}:{meta.video_id}"
    note_name = resolve_note_filename(
        db, row_id, published=meta.upload_date or "", title=meta.title
    )
    note_path = f"{cfg.notes_dir}/{note_name}.md"
    transcript_path = f"{cfg.transcripts_dir}/{note_name}.md"

    # Swept once, shared by both passes. (S4 does its own sweep for the
    # vocabulary hint moments earlier; threading that object in here to save
    # this one is a possible optimisation, not worth the None-on-resume code
    # path today.)
    topic_index = await load_topic_index(vault, cfg)
    topic_links = resolve_topic_links(db, cfg, digest.topics, topic_index)
    entity_links = resolve_topic_links(db, cfg, digest.entities, topic_index)

    now_ms = epoch_ms()

    transcript_note = render_transcript_note(
        meta, transcript, tier=tier, vault_path=transcript_path
    )
    digest_note = render_digest_note(
        meta,
        digest,
        tier=tier,
        transcript_tier_degraded=transcript_tier_degraded,
        asr_model=asr_model,
        summary_model=summary_model,
        vault_path=note_path,
        transcript_vault_path=transcript_path,
        topic_links=topic_links,
        entity_links=entity_links,
    )

    # Transcript is immutable after S3 (design §5 S5): written once, never
    # rewritten. `merge=False` still no-ops on identical content, but this
    # note's content never changes for a fixed (video_id, tier), so the
    # write only ever really happens the first time.
    await vault.project(
        transcript_note.path, transcript_note.markdown, mtime_ms=now_ms, merge=False
    )
    await vault.project(digest_note.path, digest_note.markdown, mtime_ms=now_ms, merge=False)

    for name, link_path in {**topic_links, **entity_links}.items():
        await _update_topic_page(vault, name, link_path, meta, note_path, now_ms)

    now = iso_now()
    db.execute(
        # `written_at` is COALESCEd, not read-then-written: the write-once rule
        # is enforced in SQL, so a rewrite (`POST /videos/{id}/rewrite`) or a
        # forced re-run physically cannot move it, and there is no read/write
        # race to reason about. It is the ordering key `GET /videos` exports
        # by — see db.py's SCHEMA. This statement is on both the pipeline path
        # and the rewrite path, so this one place covers both.
        "UPDATE videos SET note_path = ?, transcript_path = ?, "
        "written_at = COALESCE(written_at, ?), updated_at = ? WHERE id = ?",
        (note_path, transcript_path, now, now, row_id),
    )
    db.commit()
    log.info("vault.note_written", video_id=meta.video_id, note_path=note_path)
    return note_path, transcript_path


async def _update_topic_page(
    vault: LiveSyncVault,
    name: str,
    link_path: str,
    meta: VideoMetadata,
    note_path: str,
    now_ms: int,
) -> None:
    """Our region on a shared `99 topics/` page — merged, never whole-file
    (plan §1.2). Rebuilt whole from this video alone; a fuller rebuild across
    every video mentioning this topic is a natural follow-up once this is
    the bottleneck, not before.

    Also accumulates `video_aliases:` — this writer's own record of every
    spelling seen for this topic that differs from its established `title:`.
    `title:` itself is unprefixed and create-only (`merge_frontmatter`), so
    only the very first video to mention a topic gets to name its page; every
    later video phrasing it differently ("Anthropic PBC" once the page is
    already titled "Anthropic") is exactly the signal `TopicIndex.alias_slug`
    needs, and would otherwise be lost. `video_aliases:` is prefixed (ours),
    so it is replaced whole every run — the accumulation happens here, by
    reading the page first and unioning in, not via the generic create-only
    merge.

    This reads the one page it is about to write (not the whole folder), so
    it stays live rather than working off `write_video_note`'s pre-loop
    index snapshot — a topic named twice in one video, once via `topics` and
    once via `entities`, must see the first write when it does the second.
    """
    aliases: list[str] = []
    existing = await vault.read_note(f"{link_path}.md")
    if existing is not None:
        front, _ = split_frontmatter(existing)
        title = _scalar_value(_frontmatter_value(front, "title"))
        aliases = _parse_list_value(_frontmatter_value(front, f"{KEY_PREFIX}aliases"))
        if (
            title
            and sanitize.canonical(name) != sanitize.canonical(title)
            and not any(sanitize.canonical(a) == sanitize.canonical(name) for a in aliases)
        ):
            aliases = [*aliases, name]

    published = meta.upload_date or "undated"
    section = wrap(f"## From videos\n- {published} · [[{note_path}|{meta.title}]]", OWNER)
    alias_line = f"\n{KEY_PREFIX}aliases: {yaml_flow_list(aliases)}" if aliases else ""
    frontmatter = (
        f"---\ntype: topic\ntitle: {yaml_str(name)}\n"
        f"{KEY_PREFIX}mentions: 1{alias_line}\n---"
    )
    body = f"{frontmatter}\n\n# {name}\n\n{section}\n"
    await vault.project(f"{link_path}.md", body, mtime_ms=now_ms, merge=True)
