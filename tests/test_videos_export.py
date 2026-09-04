"""`GET /videos` — the export podcast-digest imports from.

The assertion that matters most is the cursor one: `written_at` is not
unique (migration 2 backfills it from `created_at`, and a playlist expansion
stamps several rows in the same second), so a cursor on the timestamp alone
silently drops rows that share a page boundary. That is a data-loss bug the
importer could not detect from the outside.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from video_digest.config import Settings
from video_digest.main import create_app

_MIN_LLM = {
    "tiers": {
        "map": {"primary": {"provider": "ollama", "model": "x"}},
        "reduce": {"primary": {"provider": "ollama", "model": "x"}},
    }
}
KEY = {"X-API-Key": "test-admin-key"}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm=_MIN_LLM,
        transcription={"remote_url": "http://127.0.0.1:9"},
        output={"db_path": tmp_path / "state.sqlite", "work_dir": tmp_path / "work"},
        admin_api_key="test-admin-key",
    )


def _digest(**overrides: object) -> str:
    body: dict[str, object] = {
        "tldr": "The short version.",
        "summary_md": "Some **prose**.",
        "key_points": ["A point"],
        "highlights": [{"t_seconds": 754, "label": "A moment"}],
        "entities": ["WorkOS"],
        "topics": ["CIAM"],
        "claims_to_verify": [],
        "action_items": [],
        "relevance": "high",
    }
    body.update(overrides)
    return json.dumps(body)


def _metadata(**overrides: object) -> str:
    body: dict[str, object] = {
        "title": "A Video",
        "channel": "A Channel",
        "channel_id": "UC1",
        "duration_s": 1212,
        "upload_date": "2026-07-21",
        "description": "d",
        "chapters": [],
        "language": "en",
        "has_manual_subs": False,
        "manual_sub_langs": [],
        "auto_caption_langs": ["en"],
        "thumbnail_url": "https://img.example/x.webp",
    }
    body.update(overrides)
    return json.dumps(body)


_DEFAULT = object()


def _seed(
    app,
    video_id: str,
    *,
    stage_write: str = "done",
    written_at: str | None = "2026-08-01T10:00:00+00:00",
    digest: str | object | None = _DEFAULT,
    metadata: str | object | None = _DEFAULT,
    transcript_json: str | None = None,
) -> str:
    row_id = f"youtube:{video_id}"
    app.state.db.execute(
        "INSERT INTO videos (id, video_id, url, canonical_url, metadata, digest, "
        "stage_write, written_at, transcript_json, note_path, created_at, updated_at) "
        "VALUES (?, ?, 'u', 'u', ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row_id,
            video_id,
            _metadata() if metadata is _DEFAULT else metadata,
            _digest() if digest is _DEFAULT else digest,
            stage_write,
            written_at,
            transcript_json,
            f"13 video-summaries/{video_id}.md",
            "2026-08-01T09:00:00+00:00",
            "2026-08-01T10:00:00+00:00",
        ),
    )
    app.state.db.commit()
    return row_id


class TestAuth:
    def test_requires_the_admin_key(self, settings: Settings) -> None:
        assert TestClient(create_app(settings)).get("/videos").status_code == 401

    def test_key_is_accepted(self, settings: Settings) -> None:
        assert TestClient(create_app(settings)).get("/videos", headers=KEY).status_code == 200


class TestPayload:
    def test_only_finished_summaries(self, settings: Settings) -> None:
        app = create_app(settings)
        _seed(app, "ok")
        _seed(app, "unwritten", stage_write="pending")
        _seed(app, "nodigest", digest=None)
        _seed(app, "nostamp", written_at=None)

        body = TestClient(app).get("/videos", headers=KEY).json()

        assert body["count"] == 1
        assert body["videos"][0]["video_id"] == "ok"

    def test_metadata_and_digest_are_parsed_not_raw_json(self, settings: Settings) -> None:
        app = create_app(settings)
        _seed(app, "vid1")
        item = TestClient(app).get("/videos", headers=KEY).json()["videos"][0]

        assert item["metadata"]["title"] == "A Video"  # dict, not a string
        assert item["digest"]["tldr"] == "The short version."
        assert item["digest"]["highlights"][0]["t_seconds"] == 754
        assert item["id"] == "youtube:vid1"
        assert item["note_path"] == "13 video-summaries/vid1.md"

    def test_transcript_is_never_shipped(self, settings: Settings) -> None:
        """The largest column by far, and the importer wants the summary."""
        app = create_app(settings)
        _seed(app, "vid1", transcript_json=json.dumps([{"start_s": 0, "text": "x" * 5000}]))

        resp = TestClient(app).get("/videos", headers=KEY)

        assert "transcript_json" not in resp.json()["videos"][0]
        assert "x" * 5000 not in resp.text

    def test_a_malformed_row_is_skipped_not_fatal(self, settings: Settings) -> None:
        app = create_app(settings)
        _seed(app, "good")
        _seed(app, "broken", digest="{not json")

        body = TestClient(app).get("/videos", headers=KEY).json()

        assert [v["video_id"] for v in body["videos"]] == ["good"]

    def test_newest_first(self, settings: Settings) -> None:
        app = create_app(settings)
        _seed(app, "old", written_at="2026-01-01T00:00:00+00:00")
        _seed(app, "new", written_at="2026-12-01T00:00:00+00:00")

        body = TestClient(app).get("/videos", headers=KEY).json()

        assert [v["video_id"] for v in body["videos"]] == ["new", "old"]


class TestCursor:
    def test_limit_is_bounded(self, settings: Settings) -> None:
        client = TestClient(create_app(settings))
        assert client.get("/videos?limit=0", headers=KEY).status_code == 422
        assert client.get("/videos?limit=9999", headers=KEY).status_code == 422

    def test_next_is_null_on_a_short_page(self, settings: Settings) -> None:
        app = create_app(settings)
        _seed(app, "vid1")
        assert TestClient(app).get("/videos", headers=KEY).json()["next"] is None

    def test_paging_yields_every_row_exactly_once(self, settings: Settings) -> None:
        app = create_app(settings)
        for i in range(5):
            _seed(app, f"v{i}", written_at=f"2026-08-0{i + 1}T10:00:00+00:00")
        client = TestClient(app)

        seen: list[str] = []
        url = "/videos?limit=2"
        while url:
            body = client.get(url, headers=KEY).json()
            seen += [v["video_id"] for v in body["videos"]]
            nxt = body["next"]
            url = f"/videos?limit=2&since={nxt['since']}&since_id={nxt['since_id']}" if nxt else ""

        assert seen == ["v4", "v3", "v2", "v1", "v0"]
        assert len(seen) == len(set(seen))

    def test_rows_sharing_a_timestamp_across_a_page_boundary_are_not_skipped(
        self, settings: Settings
    ) -> None:
        """The hazard the (written_at, id) pair exists for.

        Three rows share one timestamp — what migration 2's backfill and a
        playlist expansion both produce. With a `written_at`-only cursor the
        second page would start *after* the shared timestamp and lose the
        rows still owed from it.
        """
        app = create_app(settings)
        same = "2026-05-05T00:00:00+00:00"
        for vid in ("aaa", "bbb", "ccc"):
            _seed(app, vid, written_at=same)
        client = TestClient(app)

        first = client.get("/videos", params={"limit": 2}, headers=KEY).json()
        nxt = first["next"]
        assert nxt is not None
        second = client.get(
            "/videos",
            params={"limit": 2, "since": nxt["since"], "since_id": nxt["since_id"]},
            headers=KEY,
        ).json()

        got = [v["video_id"] for v in first["videos"]] + [v["video_id"] for v in second["videos"]]
        assert sorted(got) == ["aaa", "bbb", "ccc"], "a row sharing the boundary timestamp was lost"

    def test_an_unencoded_plus_in_the_cursor_still_pages(self, settings: Settings) -> None:
        """`+00:00` in a query string decodes to a space. A client that builds
        the URL by hand would otherwise get an empty page and stop early,
        having silently lost every row after the cursor."""
        app = create_app(settings)
        _seed(app, "old", written_at="2026-01-01T00:00:00+00:00")
        _seed(app, "new", written_at="2026-12-01T00:00:00+00:00")

        # Raw, unencoded — exactly what string interpolation produces.
        resp = TestClient(app).get(
            "/videos?since=2026-12-01T00:00:00+00:00&since_id=youtube:new", headers=KEY
        )

        assert [v["video_id"] for v in resp.json()["videos"]] == ["old"]

    def test_a_nonsense_cursor_is_a_400_not_an_empty_page(self, settings: Settings) -> None:
        """"No results" and "your cursor was garbage" must not look alike."""
        app = create_app(settings)
        _seed(app, "vid1")
        resp = TestClient(app).get("/videos", params={"since": "banana"}, headers=KEY)
        assert resp.status_code == 400
