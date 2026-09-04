"""The queued-job runner (design §6).

`run_job` walks one video S2 → S3 → S4 → S5, resuming from the first
unfinished stage. `run_due_jobs` picks queued jobs off the `jobs` table and
maps each outcome back onto `jobs.status`. Parked-for-ASR, LLM-down and
vault-down are non-terminal: the job stays `queued` for a later pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from video_digest.config import Settings
from video_digest.db import connect
from video_digest.llm.base import LLMUnavailable
from video_digest.llm.models import VideoDigest
from video_digest.pipeline.runner import run_due_jobs, run_job
from video_digest.pipeline.transcript import ParkedForASR, TranscriptAcquired
from video_digest.transcripts.normalize import Paragraph, Transcript
from video_digest.vault.livesync import VaultUnavailable

_MIN_LLM = {
    "tiers": {
        "map": {"primary": {"provider": "ollama", "model": "m"}},
        "reduce": {"primary": {"provider": "ollama", "model": "reduce-model"}},
    }
}


def _settings(**overrides: object) -> Settings:
    fields: dict[str, object] = {
        "llm": _MIN_LLM,
        "transcription": {"remote_url": "http://mac.local:8000"},
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def _seed(
    db: Any,
    video_id: str,
    *,
    metadata: dict | None = None,
    transcript_json: str | None = None,
    digest: str | None = None,
    stage_transcript: str = "pending",
    stage_normalize: str = "pending",
    stage_summarize: str = "pending",
    stage_write: str = "pending",
    transcript_tier: str | None = None,
    force: bool = False,
    force_asr: bool = False,
    asr_queued_at: str | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> tuple[str, str]:
    row_id = f"youtube:{video_id}"
    job_id = f"job-{video_id}"
    db.execute(
        "INSERT INTO videos (id, video_id, url, canonical_url, metadata, "
        "stage_resolve, stage_metadata, stage_transcript, stage_normalize, "
        "stage_summarize, stage_write, transcript_json, transcript_tier, digest, "
        "force_asr, asr_queued_at, created_at, updated_at) "
        "VALUES (?, ?, 'u', 'u', ?, 'done', 'done', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'now', 'now')",
        (
            row_id,
            video_id,
            json.dumps(metadata or {"title": "A Video", "duration_s": 600}),
            stage_transcript,
            stage_normalize,
            stage_summarize,
            stage_write,
            transcript_json,
            transcript_tier,
            digest,
            int(force_asr),
            asr_queued_at,
        ),
    )
    db.execute(
        "INSERT INTO jobs (id, video_id, force, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 'queued', ?, 'now')",
        (job_id, row_id, int(force), created_at),
    )
    db.commit()
    return row_id, job_id


def _acquired(
    *, tier: str = "T1", degraded: bool = False, asr_model: str | None = None
) -> TranscriptAcquired:
    return TranscriptAcquired(
        transcript=Transcript(paragraphs=[Paragraph(start_s=0, text="hello world")]),
        tier=tier,
        degraded=degraded,
        asr_model=asr_model,
    )


def _digest(**overrides: object) -> VideoDigest:
    fields: dict[str, object] = {
        "tldr": "t",
        "summary_md": "s",
        "key_points": ["p"],
        "relevance": "medium",
    }
    fields.update(overrides)
    return VideoDigest(**fields)  # type: ignore[arg-type]


class _RecordingNotifier:
    def __init__(self) -> None:
        self.failures: list[tuple[str, str, str]] = []

    async def job_failed(self, video_id: str, stage: str, error: str) -> None:
        self.failures.append((video_id, stage, error))

    async def asr_stale(self, rows: list[dict[str, object]]) -> None:  # pragma: no cover
        pass


async def _ok_write(*_a: Any, **_kw: Any) -> tuple[str, str]:
    return "13 video-summaries/note.md", "14 video-transcripts/note.md"


async def _ok_summarize(*_a: Any, **_kw: Any) -> VideoDigest:
    return _digest()


class TestRunJob:
    @pytest.mark.asyncio
    async def test_a_broken_vault_degrades_known_topics_rather_than_failing_s4(
        self, tmp_path: Path
    ) -> None:
        """S4 is documented as not depending on vault reachability (only S5's
        write is) — fetch_known_topics failing against `vault=object()` (no
        list_prefix at all, let alone a reachable one) must still let
        summarise proceed, just with an empty vocabulary hint."""
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(db, "v1")

        async def acquire(*_a: Any, **_kw: Any) -> TranscriptAcquired:
            return _acquired(tier="T0")

        seen: dict[str, Any] = {}

        async def summarize(*_a: Any, **kw: Any) -> VideoDigest:
            seen["known_topics"] = kw.get("known_topics")
            return _digest()

        outcome = await run_job(
            db,
            _settings(),
            db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone(),
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            _acquire=acquire,
            _summarize=summarize,
            _write=_ok_write,
        )

        assert outcome == "done"
        assert seen["known_topics"] == []

    @pytest.mark.asyncio
    async def test_captioned_video_runs_end_to_end(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(db, "v1")

        async def acquire(*_a: Any, **_kw: Any) -> TranscriptAcquired:
            return _acquired(tier="T0")

        async def summarize(*_a: Any, **_kw: Any) -> VideoDigest:
            return _digest(tldr="the point")

        outcome = await run_job(
            db,
            _settings(),
            db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone(),
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            _acquire=acquire,
            _summarize=summarize,
            _write=_ok_write,
        )

        assert outcome == "done"
        row = db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone()
        assert row["stage_transcript"] == "done"
        assert row["stage_normalize"] == "done"
        assert row["stage_summarize"] == "done"
        assert row["stage_write"] == "done"
        assert row["transcript_tier"] == "T0"
        assert json.loads(row["transcript_json"]) == [{"start_s": 0, "text": "hello world"}]
        assert json.loads(row["digest"])["tldr"] == "the point"

    @pytest.mark.asyncio
    async def test_resumes_from_summarize_without_reacquiring(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(
            db,
            "v1",
            stage_transcript="done",
            stage_normalize="done",
            transcript_tier="T1",
            transcript_json=json.dumps([{"start_s": 0, "text": "already here"}]),
        )

        async def acquire(*_a: Any, **_kw: Any) -> TranscriptAcquired:
            raise AssertionError("transcript already done — must not re-acquire")

        seen: dict[str, Transcript] = {}

        async def summarize(_llm: Any, _meta: Any, transcript: Transcript, **_kw: Any) -> VideoDigest:
            seen["transcript"] = transcript
            return _digest()

        outcome = await run_job(
            db,
            _settings(),
            db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone(),
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            _acquire=acquire,
            _summarize=summarize,
            _write=_ok_write,
        )

        assert outcome == "done"
        assert seen["transcript"].paragraphs[0].text == "already here"

    @pytest.mark.asyncio
    async def test_drain_handoff_completes_normalize_then_finishes(self, tmp_path: Path) -> None:
        # The ASR drain job moves only stage_transcript; the runner picks up
        # from stage_normalize on the next pass.
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(
            db,
            "v1",
            stage_transcript="done",
            stage_normalize="pending",
            transcript_tier="T2",
            transcript_json=json.dumps([{"start_s": 0, "text": "asr text"}]),
        )

        async def acquire(*_a: Any, **_kw: Any) -> TranscriptAcquired:
            raise AssertionError("must not re-acquire after a drain handoff")

        outcome = await run_job(
            db,
            _settings(),
            db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone(),
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            _acquire=acquire,
            _summarize=_ok_summarize,
            _write=_ok_write,
        )

        assert outcome == "done"
        row = db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone()
        assert row["stage_normalize"] == "done"
        assert row["stage_write"] == "done"

    @pytest.mark.asyncio
    async def test_parked_for_asr_is_not_a_failure(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(db, "v1")

        async def acquire(*_a: Any, **_kw: Any) -> TranscriptAcquired:
            raise ParkedForASR()

        outcome = await run_job(
            db,
            _settings(),
            db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone(),
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            _acquire=acquire,
        )

        assert outcome == "parked"
        row = db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone()
        assert row["stage_transcript"] == "pending"
        assert row["asr_queued_at"] is not None

    @pytest.mark.asyncio
    async def test_llm_unavailable_defers_without_failing(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(
            db,
            "v1",
            stage_transcript="done",
            stage_normalize="done",
            transcript_json=json.dumps([{"start_s": 0, "text": "x"}]),
        )

        async def summarize(*_a: Any, **_kw: Any) -> VideoDigest:
            raise LLMUnavailable("every endpoint down")

        outcome = await run_job(
            db,
            _settings(),
            db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone(),
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            _summarize=summarize,
        )

        assert outcome == "deferred"
        row = db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone()
        assert row["stage_summarize"] == "pending"
        assert row["digest"] is None

    @pytest.mark.asyncio
    async def test_vault_unavailable_defers(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(
            db,
            "v1",
            stage_transcript="done",
            stage_normalize="done",
            stage_summarize="done",
            transcript_json=json.dumps([{"start_s": 0, "text": "x"}]),
            digest=_digest().model_dump_json(),
        )

        async def write(*_a: Any, **_kw: Any) -> tuple[str, str]:
            raise VaultUnavailable("couch down")

        outcome = await run_job(
            db,
            _settings(),
            db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone(),
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            _write=write,
        )

        assert outcome == "deferred"
        row = db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone()
        assert row["stage_write"] == "pending"

    @pytest.mark.asyncio
    async def test_unexpected_error_fails_and_notifies(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(
            db,
            "v1",
            stage_transcript="done",
            stage_normalize="done",
            transcript_json=json.dumps([{"start_s": 0, "text": "x"}]),
        )
        notifier = _RecordingNotifier()

        async def summarize(*_a: Any, **_kw: Any) -> VideoDigest:
            raise RuntimeError("boom")

        outcome = await run_job(
            db,
            _settings(),
            db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone(),
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            notifier=notifier,  # type: ignore[arg-type]
            _summarize=summarize,
        )

        assert outcome == "failed"
        row = db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone()
        assert row["stage_summarize"] == "failed"
        assert "boom" in json.loads(row["stage_errors"])["summarize"]
        assert json.loads(row["attempts"])["summarize"] == 1
        assert notifier.failures == [("v1", "summarize", "RuntimeError: boom")]

    @pytest.mark.asyncio
    async def test_force_reacquires_from_transcript(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(
            db,
            "v1",
            stage_transcript="done",
            stage_normalize="done",
            stage_summarize="done",
            stage_write="done",
            transcript_tier="T1",
            transcript_json=json.dumps([{"start_s": 0, "text": "stale"}]),
            digest=_digest(tldr="stale digest").model_dump_json(),
        )

        reacquired: list[bool] = []

        async def acquire(*_a: Any, **_kw: Any) -> TranscriptAcquired:
            reacquired.append(True)
            return _acquired(tier="T2", asr_model="large-v3")

        async def summarize(*_a: Any, **_kw: Any) -> VideoDigest:
            return _digest(tldr="fresh digest")

        outcome = await run_job(
            db,
            _settings(),
            db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone(),
            force=True,
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            _acquire=acquire,
            _summarize=summarize,
            _write=_ok_write,
        )

        assert outcome == "done"
        assert reacquired == [True]
        row = db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone()
        assert row["transcript_tier"] == "T2"
        assert json.loads(row["digest"])["tldr"] == "fresh digest"


class TestRunDueJobs:
    @pytest.mark.asyncio
    async def test_done_marks_the_job_done(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        _, job_id = _seed(db, "v1")

        async def fake(*_a: Any, **_kw: Any) -> str:
            return "done"

        completed = await run_due_jobs(
            db, _settings(), llm=object(), vault=object(), _run_job=fake  # type: ignore[arg-type]
        )
        assert completed == 1
        assert db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()[0] == "done"

    @pytest.mark.asyncio
    async def test_deferred_leaves_the_job_queued(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        _, job_id = _seed(db, "v1")

        async def fake(*_a: Any, **_kw: Any) -> str:
            return "deferred"

        completed = await run_due_jobs(
            db, _settings(), llm=object(), vault=object(), _run_job=fake  # type: ignore[arg-type]
        )
        assert completed == 0
        assert (
            db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()[0] == "queued"
        )

    @pytest.mark.asyncio
    async def test_failed_marks_the_job_failed(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        _, job_id = _seed(db, "v1")

        async def fake(*_a: Any, **_kw: Any) -> str:
            return "failed"

        await run_due_jobs(
            db, _settings(), llm=object(), vault=object(), _run_job=fake  # type: ignore[arg-type]
        )
        assert (
            db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()[0] == "failed"
        )

    @pytest.mark.asyncio
    async def test_videos_parked_for_asr_are_skipped(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        _seed(db, "v1", asr_queued_at="2026-01-01T00:00:00+00:00")

        async def fake(*_a: Any, **_kw: Any) -> str:
            raise AssertionError("a parked video is the drain job's, not the runner's")

        completed = await run_due_jobs(
            db, _settings(), llm=object(), vault=object(), _run_job=fake  # type: ignore[arg-type]
        )
        assert completed == 0

    @pytest.mark.asyncio
    async def test_one_video_with_several_queued_jobs_runs_once(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        row_id, _ = _seed(db, "v1", created_at="2026-01-01T00:00:00+00:00")
        db.execute(
            "INSERT INTO jobs (id, video_id, force, status, created_at, updated_at) "
            "VALUES ('job-v1-b', ?, 1, 'queued', '2026-01-02T00:00:00+00:00', 'now')",
            (row_id,),
        )
        db.commit()

        calls: list[str] = []

        async def fake(_db: Any, _s: Any, row: Any, **_kw: Any) -> str:
            calls.append(row["video_id"])
            return "done"

        await run_due_jobs(
            db, _settings(), llm=object(), vault=object(), _run_job=fake  # type: ignore[arg-type]
        )
        assert calls == ["v1"]

    @pytest.mark.asyncio
    async def test_a_job_orphaned_at_running_is_picked_back_up(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        _, job_id = _seed(db, "v1")
        db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (job_id,))
        db.commit()

        async def fake(*_a: Any, **_kw: Any) -> str:
            return "done"

        completed = await run_due_jobs(
            db, _settings(), llm=object(), vault=object(), _run_job=fake  # type: ignore[arg-type]
        )
        assert completed == 1
        assert db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()[0] == "done"

    @pytest.mark.asyncio
    async def test_limit_caps_a_pass(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "s.sqlite")
        for i in range(3):
            _seed(db, f"v{i}", created_at=f"2026-01-0{i + 1}T00:00:00+00:00")

        calls: list[str] = []

        async def fake(_db: Any, _s: Any, row: Any, **_kw: Any) -> str:
            calls.append(row["video_id"])
            return "done"

        await run_due_jobs(
            db,
            _settings(),
            llm=object(),  # type: ignore[arg-type]
            vault=object(),  # type: ignore[arg-type]
            limit=2,
            _run_job=fake,
        )
        assert calls == ["v0", "v1"]
