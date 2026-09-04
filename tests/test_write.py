"""S5 integration: a real note appears, a re-run writes nothing, and another
writer's `99 topics/` section survives byte-for-byte (plan §M3's "done when").
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from video_digest import sanitize
from video_digest.config import VaultConfig
from video_digest.db import connect
from video_digest.llm.models import Highlight, VideoDigest
from video_digest.pipeline.write import (
    TopicIndex,
    TopicPage,
    _parse_list_value,
    fetch_known_topics,
    load_topic_index,
    resolve_topic_links,
    write_video_note,
)
from video_digest.sources.youtube import VideoMetadata
from video_digest.transcripts.normalize import Paragraph, Transcript
from video_digest.vault.livesync import LiveSyncVault, _chunk_id
from video_digest.vault.names import adopt_topic_filename
from video_digest.vault.notes import yaml_flow_list
from video_digest.vault.render import render_digest_note, render_transcript_note

COUCH = "http://vault-couch.lan:5984"
DB = "the_brain"
VID = "dQw4w9WgXcQ"
ROW_ID = f"youtube:{VID}"


@pytest.fixture
def db(tmp_path: Path):
    conn = connect(tmp_path / "state.sqlite")
    # note_names.video_id carries a foreign key to videos.id — a row must
    # exist first, same as real usage after pipeline/resolve.py's enqueue().
    conn.execute(
        "INSERT INTO videos (id, video_id, url, canonical_url, created_at, updated_at) "
        "VALUES (?, ?, 'u', 'u', 'now', 'now')",
        (ROW_ID, VID),
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def cfg() -> VaultConfig:
    return VaultConfig(couchdb_url=COUCH, db=DB, topic_creation_threshold=2)


@pytest.fixture
def vault(cfg: VaultConfig) -> LiveSyncVault:
    return LiveSyncVault(cfg, "secret")


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
        "key_points": ["Point one"],
        "highlights": [Highlight(t_seconds=724, label="A key moment")],
        "entities": [],
        "topics": [],
        "claims_to_verify": [],
        "action_items": [],
        "relevance": "high",
    }
    fields.update(overrides)
    return VideoDigest(**fields)  # type: ignore[arg-type]


def _transcript() -> Transcript:
    return Transcript(paragraphs=[Paragraph(start_s=0, text="Hello world.")])


def _mock_empty_topics_listing() -> None:
    respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
        return_value=httpx.Response(200, json={"rows": []})
    )


def _mock_all_writes_succeed() -> None:
    respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
        return_value=httpx.Response(201, json={"ok": True})
    )


def _url(path: str) -> str:
    return f"{COUCH}/{DB}/" + path.lower().replace("/", "%2F").replace(" ", "%20")


def _mock_topic_pages(pages: dict[str, str]) -> None:
    """`{"99 topics/anthropic.md": "<full markdown, frontmatter and all>", ...}`.
    Mocks the `_all_docs` listing plus each entry's own GET and its one chunk's
    GET — everything `fetch_known_topics` and the alias-matching path in
    `resolve_topic_links` need to read a candidate page's content."""
    rows = []
    for path, markdown in pages.items():
        chunk_id = f"h:t{sanitize.slugify(path)}"
        rows.append(
            {
                "id": path,
                "value": {"rev": "1-a"},
                "doc": {"path": path, "children": [chunk_id], "mtime": 1},
            }
        )
        respx.get(_url(path)).mock(
            return_value=httpx.Response(
                200,
                json={
                    "_id": path.lower(),
                    "path": path,
                    "children": [chunk_id],
                    "ctime": 1,
                    "mtime": 1,
                    "_rev": "1-a",
                    "type": "plain",
                },
            )
        )
        respx.get(f"{COUCH}/{DB}/{chunk_id.replace(':', '%3A')}").mock(
            return_value=httpx.Response(200, json={"data": markdown, "type": "leaf"})
        )
    respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
        return_value=httpx.Response(200, json={"rows": rows})
    )


class TestANoteAppears:
    @respx.mock
    @pytest.mark.asyncio
    async def test_writes_a_pinned_digest_and_transcript_note(
        self, db, cfg, vault
    ) -> None:
        _mock_empty_topics_listing()
        _mock_all_writes_succeed()

        note_path, transcript_path = await write_video_note(
            db, vault, cfg, _meta(), _digest(), _transcript(), tier="T1"
        )

        assert note_path == "13 video-summaries/2026-08-20-a-video-about-ollama.md"
        assert transcript_path == "14 video-transcripts/2026-08-20-a-video-about-ollama.md"

        row = db.execute("SELECT * FROM videos WHERE id = ?", (ROW_ID,)).fetchone()
        assert row["note_path"] == note_path
        assert row["transcript_path"] == transcript_path

    @respx.mock
    @pytest.mark.asyncio
    async def test_filename_is_pinned_and_a_title_correction_does_not_move_it(
        self, db, cfg, vault
    ) -> None:
        _mock_empty_topics_listing()
        _mock_all_writes_succeed()

        first_path, _ = await write_video_note(
            db, vault, cfg, _meta(), _digest(), _transcript(), tier="T1"
        )
        second_path, _ = await write_video_note(
            db,
            vault,
            cfg,
            _meta(title="A Corrected Title Entirely"),
            _digest(),
            _transcript(),
            tier="T1",
        )
        assert first_path == second_path


class TestRerunWritesNothing:
    @respx.mock
    @pytest.mark.asyncio
    async def test_identical_rerun_makes_no_new_documents(
        self, db, cfg, vault
    ) -> None:
        _mock_empty_topics_listing()

        note_path = "13 video-summaries/2026-08-20-a-video-about-ollama.md"
        transcript_path = "14 video-transcripts/2026-08-20-a-video-about-ollama.md"
        transcript_note = render_transcript_note(
            _meta(), _transcript(), tier="T1", vault_path=transcript_path
        )
        digest_note = render_digest_note(
            _meta(),
            _digest(),
            tier="T1",
            transcript_tier_degraded=False,
            asr_model=None,
            summary_model="",
            vault_path=note_path,
            transcript_vault_path=transcript_path,
            topic_links={},
            entity_links={},
        )
        digest_chunk_id = _chunk_id(digest_note.markdown)
        transcript_chunk_id = _chunk_id(transcript_note.markdown)

        # Both notes already exist, content-identical to what this run would
        # produce: every chunk PUT and every entry PUT conflicts (409), and
        # every conflict resolves to "already says this" -> no write.
        respx.put(url__startswith=f"{COUCH}/{DB}/h%3At").mock(
            return_value=httpx.Response(409, json={})
        )
        respx.get(f"{COUCH}/{DB}/{digest_chunk_id}").mock(
            return_value=httpx.Response(
                200, json={"_id": digest_chunk_id, "data": digest_note.markdown, "type": "leaf"}
            )
        )
        respx.get(f"{COUCH}/{DB}/{transcript_chunk_id}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "_id": transcript_chunk_id,
                    "data": transcript_note.markdown,
                    "type": "leaf",
                },
            )
        )
        respx.put(f"{COUCH}/{DB}/{note_path.lower().replace('/', '%2F').replace(' ', '%20')}").mock(
            return_value=httpx.Response(409, json={})
        )
        respx.get(f"{COUCH}/{DB}/{note_path.lower().replace('/', '%2F').replace(' ', '%20')}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "_id": note_path.lower(),
                    "path": note_path,
                    "children": [digest_chunk_id],
                    "ctime": 1,
                    "mtime": 1,
                    "type": "plain",
                },
            )
        )
        transcript_url = f"{COUCH}/{DB}/{transcript_path.lower().replace('/', '%2F').replace(' ', '%20')}"
        respx.put(transcript_url).mock(return_value=httpx.Response(409, json={}))
        respx.get(transcript_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "_id": transcript_path.lower(),
                    "path": transcript_path,
                    "children": [transcript_chunk_id],
                    "ctime": 1,
                    "mtime": 1,
                    "type": "plain",
                },
            )
        )

        # No exception, no assertion needed beyond "it ran": the entry PUT's
        # 409-conflict branch compares `children` and returns False whenever
        # they already match, which is exactly what these mocks simulate.
        note_path_out, transcript_path_out = await write_video_note(
            db, vault, cfg, _meta(), _digest(), _transcript(), tier="T1"
        )
        assert note_path_out == note_path
        assert transcript_path_out == transcript_path


class TestAnotherWritersSectionSurvives:
    @respx.mock
    @pytest.mark.asyncio
    async def test_existing_topic_section_is_preserved(
        self, db, cfg, vault
    ) -> None:
        respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "rows": [
                        {
                            "id": "99 topics/ollama.md",
                            "value": {"rev": "1-a"},
                            "doc": {
                                "path": "99 topics/ollama.md",
                                "children": ["h:texisting"],
                                "mtime": 1,
                            },
                        }
                    ]
                },
            )
        )
        existing_page = (
            "---\n"
            "type: topic\n"
            'title: "Ollama"\n'
            "podcasts_mentions: 12\n"
            "---\n\n"
            "# Ollama\n\n"
            "My own notes on Ollama.\n\n"
            "<!-- begin:podcast-digest -->\n"
            "## From podcasts\n"
            "- 2026-07-01 · Some episode\n"
            "<!-- end:podcast-digest -->\n"
        )
        respx.get(f"{COUCH}/{DB}/99%20topics%2follama.md").mock(
            return_value=httpx.Response(
                200,
                json={
                    "_id": "99 topics/ollama.md",
                    "path": "99 topics/ollama.md",
                    "children": ["h:texisting"],
                    "ctime": 1,
                    "mtime": 1,
                    "_rev": "1-a",
                    "type": "plain",
                },
            )
        )
        respx.get(f"{COUCH}/{DB}/h%3Atexisting").mock(
            return_value=httpx.Response(200, json={"data": existing_page, "type": "leaf"})
        )
        _mock_all_writes_succeed()

        await write_video_note(
            db, vault, cfg, _meta(), _digest(topics=["Ollama"]), _transcript(), tier="T1"
        )

        chunk_puts = [
            c
            for c in respx.calls
            if c.request.method == "PUT"
            and "h%3At" in str(c.request.url)
            and "texisting" not in str(c.request.url)
        ]
        assert chunk_puts, "expected the merged topic page to be written as a new chunk"
        merged_markdown = json.loads(chunk_puts[-1].request.read())["data"]

        assert "My own notes on Ollama." in merged_markdown
        assert "<!-- begin:podcast-digest -->" in merged_markdown
        assert "Some episode" in merged_markdown
        assert "podcasts_mentions: 12" in merged_markdown
        assert "<!-- begin:video-digest -->" in merged_markdown


class TestKnownTopicsVocabulary:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_known_topics_reads_every_page_title(self, cfg, vault) -> None:
        _mock_topic_pages(
            {
                "99 topics/anthropic.md": '---\ntype: topic\ntitle: "Anthropic"\n---\n\n# Anthropic\n',
                # No title: key at all — falls back to the filename, same as
                # clippings-topics' _title_of.
                "99 topics/ollama.md": "---\ntype: topic\n---\n\n# Ollama\n",
            }
        )

        known = await fetch_known_topics(vault, cfg)

        assert known == ["Anthropic", "ollama"]


class TestAliasMatching:
    @respx.mock
    @pytest.mark.asyncio
    async def test_a_known_alias_resolves_to_the_existing_page(self, db, cfg, vault) -> None:
        """"Anthropic PBC" doesn't slug-match "anthropic.md", but it's listed
        as a video_aliases entry there — resolve_topic_links must reuse that
        page rather than falling through to the creation-threshold path and
        minting a second one."""
        _mock_topic_pages(
            {
                "99 topics/anthropic.md": (
                    '---\ntype: topic\ntitle: "Anthropic"\n'
                    "video_aliases: [Anthropic PBC]\n---\n\n# Anthropic\n"
                )
            }
        )

        index = await load_topic_index(vault, cfg)
        links = resolve_topic_links(db, cfg, ["Anthropic PBC"], index)

        assert links == {"Anthropic PBC": "99 topics/anthropic"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_generic_unprefixed_alias_also_matches(self, db, cfg, vault) -> None:
        """The generic `aliases:` key — nothing populates it yet (the
        originally-documented gap), but a human or a future sibling writer
        could, and this must not need a code change the day one does."""
        _mock_topic_pages(
            {
                "99 topics/anthropic.md": (
                    '---\ntype: topic\ntitle: "Anthropic"\n'
                    "aliases: [Anthropic PBC]\n---\n\n# Anthropic\n"
                )
            }
        )

        index = await load_topic_index(vault, cfg)
        links = resolve_topic_links(db, cfg, ["Anthropic PBC"], index)

        assert links == {"Anthropic PBC": "99 topics/anthropic"}


class TestAliasAccumulation:
    @respx.mock
    @pytest.mark.asyncio
    async def test_a_name_that_differs_from_the_title_is_recorded_as_an_alias(
        self, db, cfg, vault
    ) -> None:
        """Once a mention is already routed to an existing page under a
        different phrasing than its title, _update_topic_page must capture
        that phrasing as a video_aliases entry — the only way this writer's
        own alias knowledge grows over time."""
        # Simulates a name already resolved to this page by some other means
        # (a slug match on an earlier video, a human-curated `aliases:`) —
        # resolve_topic_links's own job is covered by TestAliasMatching above.
        adopt_topic_filename(db, sanitize.canonical("Anthropic PBC"), "anthropic")
        _mock_topic_pages(
            {"99 topics/anthropic.md": '---\ntype: topic\ntitle: "Anthropic"\n---\n\n# Anthropic\n'}
        )
        _mock_all_writes_succeed()

        await write_video_note(
            db, vault, cfg, _meta(), _digest(entities=["Anthropic PBC"]), _transcript(), tier="T1"
        )

        chunk_puts = [
            c
            for c in respx.calls
            if c.request.method == "PUT" and "h%3At" in str(c.request.url)
        ]
        assert chunk_puts, "expected the topic page to be written as a new chunk"
        merged_markdown = json.loads(chunk_puts[-1].request.read())["data"]
        assert 'title: "Anthropic"' in merged_markdown  # unchanged — create-only
        assert 'video_aliases: ["Anthropic PBC"]' in merged_markdown  # quoted flow list

    @respx.mock
    @pytest.mark.asyncio
    async def test_an_alias_already_present_is_not_duplicated(self, db, cfg, vault) -> None:
        adopt_topic_filename(db, sanitize.canonical("Anthropic PBC"), "anthropic")
        _mock_topic_pages(
            {
                "99 topics/anthropic.md": (
                    '---\ntype: topic\ntitle: "Anthropic"\n'
                    "video_aliases: [Anthropic PBC]\n---\n\n# Anthropic\n"
                )
            }
        )
        _mock_all_writes_succeed()

        await write_video_note(
            db, vault, cfg, _meta(), _digest(entities=["Anthropic PBC"]), _transcript(), tier="T1"
        )

        chunk_puts = [
            c
            for c in respx.calls
            if c.request.method == "PUT" and "h%3At" in str(c.request.url)
        ]
        merged_markdown = json.loads(chunk_puts[-1].request.read())["data"]
        assert merged_markdown.count("Anthropic PBC") == 1  # not duplicated in the list


class TestListValueRoundTrip:
    """`yaml_flow_list` write ↔ `_parse_list_value` read must survive commas,
    quotes and brackets in a name — the frontmatter fragility fix."""

    def test_round_trips_awkward_items(self) -> None:
        items = ["a, b", 'c "q"', "[d]", "plain"]
        assert _parse_list_value(yaml_flow_list(items)) == items

    def test_reads_a_legacy_unquoted_list(self) -> None:
        assert _parse_list_value("[plain, list]") == ["plain", "list"]

    def test_empty_and_malformed_are_empty(self) -> None:
        assert _parse_list_value("") == []
        assert _parse_list_value("not a list") == []
        assert _parse_list_value("[unterminated") == []  # yaml.YAMLError branch

    def test_non_string_items_are_coerced(self) -> None:
        assert _parse_list_value("[1, 2]") == ["1", "2"]


class TestLoadTopicIndex:
    @respx.mock
    @pytest.mark.asyncio
    async def test_sweeps_every_page_once(self, cfg, vault) -> None:
        _mock_topic_pages(
            {
                "99 topics/anthropic.md": (
                    '---\ntype: topic\ntitle: "Anthropic"\n'
                    "video_aliases: [Anthropic PBC]\n---\n\n# Anthropic\n"
                ),
                "99 topics/oauth2.md": '---\ntype: topic\ntitle: "OAuth2"\n---\n\n# OAuth2\n',
                # no title: — stem fallback, like clippings-topics' _title_of
                "99 topics/ollama.md": "---\ntype: topic\n---\n\n# Ollama\n",
            }
        )

        index = await load_topic_index(vault, cfg)

        assert index.slugs == {"anthropic", "oauth2", "ollama"}
        # sorted case-insensitively, stem fallback for the title-less page
        assert index.titles() == sorted(["Anthropic", "OAuth2", "ollama"], key=str.lower)
        assert index.alias_slug(sanitize.canonical("Anthropic PBC")) == "anthropic"
        assert index.alias_slug(sanitize.canonical("nope")) is None
        # every page's single chunk was fetched
        chunk_gets = [
            c
            for c in respx.calls
            if c.request.method == "GET" and "h%3At" in str(c.request.url)
        ]
        assert len(chunk_gets) == 3
        # (wall-clock parallelism is not timing-asserted — the semaphore bound
        # is covered by review, not a flaky sleep race)


class TestResolveTopicLinksIsIOFree:
    """No `@respx.mock` here: `resolve_topic_links` must touch nothing over
    the network — it reads only the in-memory `TopicIndex`."""

    def test_alias_and_exact_title_both_resolve(self, db, cfg) -> None:
        index = TopicIndex(
            slugs=frozenset({"anthropic"}),
            pages=(
                TopicPage(
                    slug="anthropic",
                    title="Anthropic",
                    alias_keys=frozenset({sanitize.canonical("Anthropic PBC")}),
                ),
            ),
        )
        assert resolve_topic_links(db, cfg, ["Anthropic PBC"], index) == {
            "Anthropic PBC": "99 topics/anthropic"
        }
        assert resolve_topic_links(db, cfg, ["Anthropic"], index) == {
            "Anthropic": "99 topics/anthropic"
        }

    def test_a_novel_under_threshold_name_earns_no_link(self, db, cfg) -> None:
        index = TopicIndex(slugs=frozenset(), pages=())
        assert resolve_topic_links(db, cfg, ["Totally New"], index) == {}


class TestFetchKnownTopicsPropagates:
    @pytest.mark.asyncio
    async def test_does_not_swallow_a_vault_error(self, cfg) -> None:
        """Swallowing an unreachable vault is the S4 caller's job
        (`pipeline/runner.py`), not this function's."""
        with pytest.raises(AttributeError):
            await fetch_known_topics(object(), cfg)  # type: ignore[arg-type]


class TestWrittenAtIsWriteOnce:
    """`written_at` is the export's ordering key. A rewrite that moves it
    reorders the row relative to a client's cursor — forward, and the row is
    skipped for good — so the write-once rule is enforced in SQL (COALESCE)
    rather than trusted to callers."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_first_write_sets_it_and_a_rewrite_does_not_move_it(
        self, db, cfg, vault, monkeypatch
    ) -> None:
        _mock_empty_topics_listing()
        _mock_all_writes_succeed()

        monkeypatch.setattr("video_digest.pipeline.write.iso_now", lambda: "2026-01-01T00:00:00+00:00")
        await write_video_note(db, vault, cfg, _meta(), _digest(), _transcript(), tier="T1")
        first = db.execute("SELECT written_at, updated_at FROM videos WHERE id = ?", (ROW_ID,)).fetchone()

        monkeypatch.setattr("video_digest.pipeline.write.iso_now", lambda: "2026-09-09T00:00:00+00:00")
        await write_video_note(db, vault, cfg, _meta(), _digest(), _transcript(), tier="T1")
        second = db.execute("SELECT written_at, updated_at FROM videos WHERE id = ?", (ROW_ID,)).fetchone()

        assert first["written_at"] == "2026-01-01T00:00:00+00:00"
        assert second["written_at"] == first["written_at"], "written_at must never move"
        # Guard against a vacuous pass: updated_at *must* have moved, or the
        # second write did nothing and this proves nothing.
        assert second["updated_at"] == "2026-09-09T00:00:00+00:00"
        assert second["updated_at"] != first["updated_at"]
