from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from video_digest.config import Settings
from video_digest.llm.models import VideoDigest
from video_digest.main import create_app

_MIN_LLM = {
    "tiers": {
        "map": {"primary": {"provider": "ollama", "model": "x"}},
        "reduce": {"primary": {"provider": "ollama", "model": "x"}},
    }
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm=_MIN_LLM,
        # 127.0.0.1:9 (discard) refuses instantly, so /healthz's ASR probe
        # returns fast instead of waiting out an mDNS lookup of a fake host.
        transcription={"remote_url": "http://127.0.0.1:9"},
        output={"db_path": tmp_path / "state.sqlite", "work_dir": tmp_path / "work"},
        admin_api_key="test-admin-key",
    )


def test_healthz_is_unauthenticated_and_ok(settings: Settings) -> None:
    client = TestClient(create_app(settings))
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["db"] == "ok"
    assert body["checks"]["work_dir"] == "ok"


def test_admin_route_requires_key(settings: Settings) -> None:
    client = TestClient(create_app(settings))
    resp = client.get("/videos/does-not-exist")
    assert resp.status_code == 401


def test_admin_route_with_key_reaches_the_handler(settings: Settings) -> None:
    client = TestClient(create_app(settings))
    resp = client.get("/videos/does-not-exist", headers={"X-API-Key": "test-admin-key"})
    assert resp.status_code == 404  # authenticated, then a legitimate 404


class TestCreateJob:
    def test_posts_a_job_and_stores_metadata(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from video_digest.sources.youtube import VideoMetadata

        def fake_fetch(url: str, *, max_duration_minutes: int, cookies_file: str | None) -> VideoMetadata:
            return VideoMetadata(
                video_id="dQw4w9WgXcQ",
                title="A Video",
                channel="A Channel",
                channel_id="UCxxxx",
                duration_s=120,
                upload_date="2026-08-20",
                description="d",
                auto_caption_langs=["en"],
            )

        monkeypatch.setattr("video_digest.pipeline.resolve.fetch_metadata", fake_fetch)

        client = TestClient(create_app(settings))
        resp = client.post(
            "/jobs",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            headers={"X-API-Key": "test-admin-key"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["video_id"] == "dQw4w9WgXcQ"
        assert body["status"] == "created"
        assert body["job_id"]

        get_resp = client.get(
            "/videos/dQw4w9WgXcQ", headers={"X-API-Key": "test-admin-key"}
        )
        assert get_resp.status_code == 200
        assert "A Video" in get_resp.json()["metadata"]

    def test_repeat_returns_existing(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from video_digest.sources.youtube import VideoMetadata

        calls: list[str] = []

        def fake_fetch(url: str, *, max_duration_minutes: int, cookies_file: str | None) -> VideoMetadata:
            calls.append(url)
            return VideoMetadata(
                video_id="dQw4w9WgXcQ",
                title="A Video",
                channel="A Channel",
                channel_id="UCxxxx",
                duration_s=120,
                upload_date="2026-08-20",
                description="d",
            )

        monkeypatch.setattr("video_digest.pipeline.resolve.fetch_metadata", fake_fetch)
        client = TestClient(create_app(settings))
        headers = {"X-API-Key": "test-admin-key"}
        payload = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}

        client.post("/jobs", json=payload, headers=headers)
        resp = client.post("/jobs", json=payload, headers=headers)

        assert resp.status_code == 202
        assert resp.json()["status"] == "existing"
        assert len(calls) == 1  # second POST never re-fetched

    def test_requires_the_admin_key(self, settings: Settings) -> None:
        client = TestClient(create_app(settings))
        resp = client.post("/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
        assert resp.status_code == 401


# ── the M6 read/rewrite/metrics endpoints (design §8) ─────────────────────

_HEADERS = {"X-API-Key": "test-admin-key"}


class _FakeVault:
    def __init__(self, *, ping: bool = True) -> None:
        self._ping = ping
        self.projected: list[str] = []

    async def list_prefix(self, prefix: str) -> list:
        return []

    async def project(self, path: str, markdown: str, *, mtime_ms: int, merge: bool = True) -> bool:
        self.projected.append(path)
        return True

    async def ping(self) -> bool:
        return self._ping


def _app_with_vault(settings: Settings, vault: object):
    app = create_app(settings)
    app.state.vault = vault
    return app


def _seed_summarised_video(db, video_id: str = "vid1") -> None:
    row_id = f"youtube:{video_id}"
    digest = VideoDigest(
        tldr="t", summary_md="s", key_points=["p"], relevance="medium"
    ).model_dump_json()
    db.execute(
        "INSERT INTO videos (id, video_id, url, canonical_url, metadata, "
        "stage_resolve, stage_metadata, stage_transcript, stage_normalize, stage_summarize, "
        "stage_write, transcript_tier, transcript_json, digest, created_at, updated_at) "
        "VALUES (?, ?, 'u', 'u', ?, 'done', 'done', 'done', 'done', 'done', 'pending', 'T1', ?, ?, "
        "'now', 'now')",
        (
            row_id,
            video_id,
            json.dumps({"title": "A Video", "duration_s": 600, "channel": "C", "channel_id": "U"}),
            json.dumps([{"start_s": 0, "text": "hello"}]),
            digest,
        ),
    )
    db.execute(
        "INSERT INTO jobs (id, video_id, force, status, created_at, updated_at) "
        "VALUES ('job1', ?, 0, 'queued', 'now', 'now')",
        (row_id,),
    )
    db.commit()


class TestJobStatus:
    def test_unknown_job_is_404(self, settings: Settings) -> None:
        client = TestClient(create_app(settings))
        assert client.get("/jobs/nope", headers=_HEADERS).status_code == 404

    def test_reports_each_stage_and_the_note_path(self, settings: Settings) -> None:
        app = create_app(settings)
        _seed_summarised_video(app.state.db)
        app.state.db.execute("UPDATE videos SET note_path = 'n.md' WHERE video_id = 'vid1'")
        app.state.db.commit()
        client = TestClient(app)

        body = client.get("/jobs/job1", headers=_HEADERS).json()
        assert body["video_id"] == "vid1"
        assert body["status"] == "queued"
        assert body["stages"]["summarize"] == "done"
        assert body["stages"]["write"] == "pending"
        assert body["note_path"] == "n.md"


class TestRewrite:
    def test_rewrites_from_the_stored_digest(self, settings: Settings) -> None:
        vault = _FakeVault()
        app = _app_with_vault(settings, vault)
        _seed_summarised_video(app.state.db)
        client = TestClient(app)

        resp = client.post("/videos/vid1/rewrite", headers=_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["note_path"].endswith(".md")
        assert any("13 video-summaries" in p for p in vault.projected)
        row = app.state.db.execute(
            "SELECT stage_write FROM videos WHERE video_id = 'vid1'"
        ).fetchone()
        assert row["stage_write"] == "done"

    def test_unknown_video_is_404(self, settings: Settings) -> None:
        client = TestClient(_app_with_vault(settings, _FakeVault()))
        assert client.post("/videos/nope/rewrite", headers=_HEADERS).status_code == 404

    def test_video_without_a_digest_is_409(self, settings: Settings) -> None:
        app = _app_with_vault(settings, _FakeVault())
        app.state.db.execute(
            "INSERT INTO videos (id, video_id, url, canonical_url, created_at, updated_at) "
            "VALUES ('youtube:bare', 'bare', 'u', 'u', 'now', 'now')"
        )
        app.state.db.commit()
        resp = TestClient(app).post("/videos/bare/rewrite", headers=_HEADERS)
        assert resp.status_code == 409


class TestMetrics:
    def test_aggregates_stages_tiers_and_llm_spend(self, settings: Settings) -> None:
        app = create_app(settings)
        db = app.state.db
        _seed_summarised_video(db, "vid1")
        db.execute(
            "INSERT INTO videos (id, video_id, url, canonical_url, metadata, transcript_tier, "
            "stage_write, created_at, updated_at) VALUES ('youtube:v2', 'v2', 'u', 'u', ?, 'T2', "
            "'done', 'now', 'now')",
            (json.dumps({"duration_s": 1800}),),
        )
        db.execute(
            "INSERT INTO metric_events (kind, video_id, value, meta, created_at) VALUES "
            "('llm_call', 'vid1', 0.002, ?, 'now')",
            (json.dumps({"tier": "reduce", "input_tokens": 100, "output_tokens": 50}),),
        )
        db.commit()

        body = TestClient(app).get("/metrics", headers=_HEADERS).json()
        assert body["videos_total"] == 2
        assert body["transcript_tiers"]["T2"] == 1
        assert body["asr_audio_minutes"] == 30.0
        assert body["llm"]["calls"] == 1
        assert body["llm"]["cost_usd"] == 0.002
        assert body["llm"]["calls_by_tier"] == {"reduce": 1}


class TestHealthzDependencies:
    def test_reports_vault_and_asr_worker(self, settings: Settings) -> None:
        client = TestClient(_app_with_vault(settings, _FakeVault(ping=True)))
        body = client.get("/healthz").json()
        # vault.couchdb_url is unset in the test settings → "not configured",
        # and status stays ok.
        assert body["checks"]["vault"] == "not configured"
        assert body["asr_worker"] == "asleep"  # 127.0.0.1:9 refuses
        assert body["status"] == "ok"

    def test_configured_but_unreachable_vault_is_degraded(self, tmp_path: Path) -> None:
        s = Settings(
            llm=_MIN_LLM,
            transcription={"remote_url": "http://127.0.0.1:9"},
            vault={"couchdb_url": "http://127.0.0.1:9"},
            output={"db_path": tmp_path / "s.sqlite", "work_dir": tmp_path / "w"},
            admin_api_key="test-admin-key",
        )
        client = TestClient(_app_with_vault(s, _FakeVault(ping=False)))
        body = client.get("/healthz").json()
        assert body["checks"]["vault"] == "unreachable"
        assert body["status"] == "degraded"
