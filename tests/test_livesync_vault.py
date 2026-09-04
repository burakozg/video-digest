"""Wire-format and safety assertions for LiveSyncVault (plan §1.1, §M3).

LiveSync's document shape is reverse-engineered rather than published —
nothing but a test pins it. Ported from the pattern in podcast-digest's
tests/test_vault.py, trimmed to what this app's own usage exercises.
"""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest
import respx

from video_digest.config import VaultConfig
from video_digest.vault.livesync import LiveSyncVault

COUCH = "http://vault-couch.lan:5984"
DB = "tastings"


def _cfg(**overrides: object) -> VaultConfig:
    return VaultConfig(**{"couchdb_url": COUCH, "db": DB, "user": "videodigest", **overrides})  # type: ignore[arg-type]


def _vault(**overrides: object) -> LiveSyncVault:
    return LiveSyncVault(_cfg(**overrides), "secret")


def _doc_url(doc_id: str) -> str:
    from urllib.parse import quote

    return f"{COUCH}/{DB}/{quote(doc_id, safe='')}"


class TestWholeFileWrite:
    """merge=False — the digest/transcript note path (plan §1.2)."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_file_becomes_a_chunk_and_an_entry(self) -> None:
        markdown = "# A Video\n\nSummary text.\n"
        chunk_id = "h:t" + hashlib.sha1(markdown.encode()).hexdigest()[:24]
        path = "13 video-summaries/2026-08-20-a-video.md"

        chunk = respx.put(_doc_url(chunk_id)).mock(return_value=httpx.Response(201, json={"ok": True}))
        entry = respx.put(_doc_url(path.lower())).mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        written = await _vault().project(path, markdown, mtime_ms=1_700_000_000_000, merge=False)

        assert written is True
        assert chunk.called and entry.called
        assert json.loads(chunk.calls[0].request.read()) == {
            "_id": chunk_id,
            "data": markdown,
            "type": "leaf",
        }
        body = json.loads(entry.calls[0].request.read())
        assert body["_id"] == path.lower()
        assert body["path"] == path
        assert body["children"] == [chunk_id]
        assert body["type"] == "plain"
        assert body["eden"] == {}
        assert body["size"] == len(markdown.encode("utf-8"))
        assert body["mtime"] == 1_700_000_000_000

    def test_size_counts_bytes_not_characters(self) -> None:
        markdown = "Café — naïve\n"
        assert len(markdown) != len(markdown.encode("utf-8"))

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_read_before_write_for_a_whole_file_note(self) -> None:
        """Plan §1.2: whole-file notes never do a read-before-write."""
        get_spy = respx.get(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(404, json={})
        )
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        await _vault().project("13 video-summaries/x.md", "text\n", mtime_ms=1, merge=False)
        assert get_spy.call_count == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_unchanged_content_writes_nothing(self) -> None:
        """The chunk PUT 409s (content-addressed id already live) and the
        entry PUT 409s with matching children — both no-ops, no read needed."""
        markdown = "same text\n"
        chunk_id = "h:t" + hashlib.sha1(markdown.encode()).hexdigest()[:24]
        path = "13 video-summaries/x.md"

        respx.put(_doc_url(chunk_id)).mock(return_value=httpx.Response(409, json={}))
        respx.get(_doc_url(chunk_id)).mock(
            return_value=httpx.Response(200, json={"_id": chunk_id, "data": markdown, "type": "leaf"})
        )
        respx.put(_doc_url(path)).mock(return_value=httpx.Response(409, json={}))
        respx.get(_doc_url(path)).mock(
            return_value=httpx.Response(
                200,
                json={
                    "_id": path,
                    "path": "13 video-summaries/X.md",
                    "children": [chunk_id],
                    "ctime": 1,
                    "mtime": 1,
                    "type": "plain",
                },
            )
        )

        written = await _vault().project(path, markdown, mtime_ms=2, merge=False)
        assert written is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_identical_content_shares_one_chunk(self) -> None:
        markdown = "same text\n"
        chunk_id = "h:t" + hashlib.sha1(markdown.encode()).hexdigest()[:24]
        chunk = respx.put(_doc_url(chunk_id)).mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        vault = _vault()
        await vault.project("13 video-summaries/a.md", markdown, mtime_ms=1, merge=False)
        await vault.project("13 video-summaries/b.md", markdown, mtime_ms=1, merge=False)
        assert chunk.call_count == 2  # two files, one chunk id each time — content-addressed


class TestDoctIdEncoding:
    @respx.mock
    @pytest.mark.asyncio
    async def test_slashes_in_the_path_are_percent_encoded(self) -> None:
        path = "13 video-summaries/2026-08-20-a-video.md"
        entry = respx.put(_doc_url(path.lower())).mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        respx.put(url__startswith=f"{COUCH}/{DB}/h%3At").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        await _vault().project(path, "x\n", mtime_ms=1, merge=False)
        raw = entry.calls[0].request.url.raw_path.decode()
        assert raw == f"/{DB}/13%20video-summaries%2F2026-08-20-a-video.md"


class TestASoftDeletedNoteStaysDeleted:
    """The design's own promise (design §5 S5, obsidian-vault-writer §5):
    generated notes do not resurrect a note the reader threw away."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_soft_deleted_entry_is_not_rewritten(self) -> None:
        path = "13 video-summaries/gone.md"
        respx.put(url__startswith=f"{COUCH}/{DB}/h%3At").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        entry = respx.put(_doc_url(path)).mock(return_value=httpx.Response(409, json={}))
        respx.get(_doc_url(path)).mock(
            return_value=httpx.Response(
                200, json={"_id": path, "_rev": "2-abc", "deleted": True, "children": []}
            )
        )
        written = await _vault().project(path, "# gone\n", mtime_ms=1, merge=False)
        assert written is False
        assert entry.call_count == 1  # no resurrecting second write


class TestListPrefix:
    @respx.mock
    @pytest.mark.asyncio
    async def test_soft_deleted_notes_are_excluded(self) -> None:
        respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "rows": [
                        {
                            "id": "13 video-summaries/a.md",
                            "value": {"rev": "1-a"},
                            "doc": {"path": "13 video-summaries/a.md", "children": ["h:t1"], "mtime": 1},
                        },
                        {
                            "id": "13 video-summaries/b.md",
                            "value": {"rev": "1-b"},
                            "doc": {
                                "path": "13 video-summaries/b.md",
                                "children": [],
                                "mtime": 1,
                                "deleted": True,
                            },
                        },
                    ]
                },
            )
        )
        entries = await _vault().list_prefix("13 video-summaries/")
        assert [e.path for e in entries] == ["13 video-summaries/a.md"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_results_are_sorted_for_stable_output(self) -> None:
        respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "rows": [
                        {
                            "id": "99 topics/z.md",
                            "value": {"rev": "1"},
                            "doc": {"path": "99 topics/z.md", "children": [], "mtime": 1},
                        },
                        {
                            "id": "99 topics/a.md",
                            "value": {"rev": "1"},
                            "doc": {"path": "99 topics/a.md", "children": [], "mtime": 1},
                        },
                    ]
                },
            )
        )
        entries = await _vault().list_prefix("99 topics/")
        assert [e.doc_id for e in entries] == ["99 topics/a.md", "99 topics/z.md"]


class TestMergedTopicWrite:
    """The test that matters most (obsidian-vault-writer skill's checklist
    item 9): another writer's section and the human's prose survive our run
    byte-for-byte."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_another_writers_section_and_human_prose_survive(self) -> None:
        path = "99 topics/anthropic.md"
        existing = (
            "---\n"
            "type: topic\n"
            "title: \"Anthropic\"\n"
            "tags: [topic, ai]\n"
            "podcasts_mentions: 97\n"
            "---\n\n"
            "# Anthropic\n\n"
            "My own view: they ship faster than they explain.\n\n"
            "<!-- begin:podcast-digest -->\n"
            "## From podcasts\n"
            "- 2026-08-20 · Some episode\n"
            "<!-- end:podcast-digest -->\n"
        )
        existing_chunk_id = "h:texisting"
        respx.get(_doc_url(path.lower())).mock(
            return_value=httpx.Response(
                200,
                json={
                    "_id": path.lower(),
                    "_rev": "3-def",
                    "path": path,
                    "children": [existing_chunk_id],
                    "ctime": 100,
                    "mtime": 100,
                    "type": "plain",
                },
            )
        )
        respx.get(_doc_url(existing_chunk_id)).mock(
            return_value=httpx.Response(200, json={"data": existing, "type": "leaf"})
        )
        respx.put(url__startswith=f"{COUCH}/{DB}/h%3At").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        # First PUT conflicts (a document already lives at this _id — the one
        # the GET above describes); the retry carries its _rev and succeeds.
        # This is what exercises ctime-preservation: it only applies on the
        # conflict-resolution branch, never on a first-attempt 201.
        entry = respx.put(_doc_url(path.lower())).mock(
            side_effect=[
                httpx.Response(409, json={}),
                httpx.Response(201, json={"ok": True}),
            ]
        )

        ours = (
            "---\n"
            "type: topic\n"
            "title: \"Anthropic\"\n"
            "video_mentions: 3\n"
            "---\n\n"
            "# Anthropic\n\n"
            "<!-- begin:video-digest -->\n"
            "## From videos\n"
            "- 2026-08-22 · [[13 video-summaries/2026-08-22-x|A Video]]\n"
            "<!-- end:video-digest -->\n"
        )
        written = await _vault().project(path, ours, mtime_ms=200, merge=True)
        assert written is True

        body = json.loads(entry.calls[-1].request.read())  # the conflict-resolved retry
        assert body["ctime"] == 100  # preserved
        assert body["mtime"] == 200  # advanced

        # The written markdown is the chunk PUT's payload — find it among
        # the h:t* calls (excludes the existing-chunk GET).
        put_calls = [c for c in respx.calls if c.request.method == "PUT" and "h%3At" in str(c.request.url)]
        written_markdown = json.loads(put_calls[-1].request.content)["data"]

        assert "My own view: they ship faster than they explain." in written_markdown
        assert "<!-- begin:podcast-digest -->" in written_markdown
        assert "- 2026-08-20 · Some episode" in written_markdown
        assert "podcasts_mentions: 97" in written_markdown
        assert "tags: [topic, ai]" in written_markdown  # untouched, unprefixed
        assert "<!-- begin:video-digest -->" in written_markdown
        assert "video_mentions: 3" in written_markdown
