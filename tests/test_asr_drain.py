"""The pending_asr drain job (plan §M5): works the queue oldest-first while
the worker answers, stops on a retry failure, never hammers a sleeping Mac.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_digest.config import Settings
from video_digest.db import connect
from video_digest.pipeline.asr_drain import alert_on_stale_pending_asr, drain_pending_asr
from video_digest.pipeline.transcript import ParkedForASR, TranscriptAcquired, park_pending_asr
from video_digest.transcripts.normalize import Paragraph, Transcript

_MIN_LLM = {
    "tiers": {
        "map": {"primary": {"provider": "ollama", "model": "x"}},
        "reduce": {"primary": {"provider": "ollama", "model": "x"}},
    }
}


def _settings(**overrides: object) -> Settings:
    fields: dict[str, object] = {
        "llm": _MIN_LLM,
        "transcription": {"remote_url": "http://mac.local:8000", "asr_stale_hours": 48},
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def _seed_pending(db, video_id: str, *, metadata: dict | None = None) -> str:
    row_id = f"youtube:{video_id}"
    db.execute(
        "INSERT INTO videos (id, video_id, url, canonical_url, metadata, created_at, updated_at) "
        "VALUES (?, ?, 'u', 'u', ?, 'now', 'now')",
        (row_id, video_id, json.dumps(metadata or {"title": "A Video"})),
    )
    db.commit()
    park_pending_asr(db, row_id)
    return row_id


class TestWorkerOffline:
    @pytest.mark.asyncio
    async def test_no_probe_success_drains_nothing(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        _seed_pending(db, "v1")

        async def probe(cfg):
            return False

        count = await drain_pending_asr(db, _settings(), _probe=probe)
        assert count == 0
        row = db.execute("SELECT stage_transcript FROM videos WHERE video_id = 'v1'").fetchone()
        assert row["stage_transcript"] == "pending"  # untouched


class TestWorkerOnline:
    @pytest.mark.asyncio
    async def test_drains_pending_rows_oldest_first(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        _seed_pending(db, "v1")
        _seed_pending(db, "v2")

        async def probe(cfg):
            return True

        calls: list[str] = []

        async def acquire(meta, *a, **kw):
            calls.append(meta.video_id)
            return TranscriptAcquired(
                transcript=Transcript(paragraphs=[Paragraph(start_s=0, text="ok")]), tier="T2"
            )

        count = await drain_pending_asr(db, _settings(), _probe=probe, _acquire_transcript=acquire)
        assert count == 2
        assert calls == ["v1", "v2"]  # insertion order == queue order (oldest first)

        for vid in ("v1", "v2"):
            row = db.execute(
                "SELECT * FROM videos WHERE video_id = ?", (vid,)
            ).fetchone()
            assert row["stage_transcript"] == "done"
            assert row["transcript_tier"] == "T2"
            assert row["asr_queued_at"] is None
            assert json.loads(row["transcript_json"]) == [{"start_s": 0, "text": "ok"}]

    @pytest.mark.asyncio
    async def test_stops_on_the_first_retry_failure(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        _seed_pending(db, "v1")
        _seed_pending(db, "v2")

        async def probe(cfg):
            return True

        calls: list[str] = []

        async def acquire(meta, *a, **kw):
            calls.append(meta.video_id)
            raise ParkedForASR()

        count = await drain_pending_asr(db, _settings(), _probe=probe, _acquire_transcript=acquire)
        assert count == 0
        assert calls == ["v1"]  # never even tried v2

        row = db.execute("SELECT * FROM videos WHERE video_id = 'v1'").fetchone()
        assert row["stage_transcript"] == "pending"  # still parked
        assert row["asr_queued_at"] is not None

    @pytest.mark.asyncio
    async def test_metadata_is_reconstructed_from_the_stored_row(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        _seed_pending(
            db,
            "v1",
            metadata={
                "title": "Stored Title",
                "channel": "Stored Channel",
                "channel_id": "UCabc",
                "duration_s": 300,
                "language": "en",
            },
        )

        async def probe(cfg):
            return True

        seen = {}

        async def acquire(meta, *a, **kw):
            seen["title"] = meta.title
            seen["duration_s"] = meta.duration_s
            return TranscriptAcquired(transcript=Transcript(paragraphs=[]), tier="T2")

        await drain_pending_asr(db, _settings(), _probe=probe, _acquire_transcript=acquire)
        assert seen["title"] == "Stored Title"
        assert seen["duration_s"] == 300


class TestStaleAlert:
    def test_stale_rows_are_reported(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        row_id = _seed_pending(db, "v1")
        db.execute(
            "UPDATE videos SET asr_queued_at = '2020-01-01T00:00:00+00:00' WHERE id = ?",
            (row_id,),
        )
        db.commit()
        stale = alert_on_stale_pending_asr(db, _settings())
        assert [r["video_id"] for r in stale] == ["v1"]

    def test_fresh_rows_are_not_reported(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        _seed_pending(db, "v1")
        assert alert_on_stale_pending_asr(db, _settings()) == []
