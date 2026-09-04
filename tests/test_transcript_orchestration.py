"""S2 orchestration (plan §M5): the T0/T1 -> T2 handoff and pending_asr
semantics (design §5 S2). No real yt-dlp/httpx calls — every I/O boundary is
injected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from video_digest.config import AcquisitionConfig, OutputConfig, TranscriptionConfig
from video_digest.db import connect
from video_digest.pipeline.transcript import (
    ParkedForASR,
    acquire_transcript,
    clear_pending_asr,
    park_pending_asr,
    pending_asr_rows,
    stale_pending_asr,
)
from video_digest.sources.youtube import VideoMetadata
from video_digest.transcripts.asr import ASRResult, ASRUnavailable
from video_digest.transcripts.ladder import AcquiredTranscript, NeedsASR
from video_digest.transcripts.normalize import Paragraph, Transcript

VID = "dQw4w9WgXcQ"
ROW_ID = f"youtube:{VID}"


def _meta(**overrides: object) -> VideoMetadata:
    fields: dict[str, object] = {
        "video_id": VID,
        "title": "A Video",
        "channel": "A Channel",
        "channel_id": "UCxxxx",
        "duration_s": 600,
        "upload_date": "2026-08-20",
        "description": "d",
    }
    fields.update(overrides)
    return VideoMetadata(**fields)  # type: ignore[arg-type]


def _acquisition_cfg() -> AcquisitionConfig:
    return AcquisitionConfig()


def _transcription_cfg(**overrides: object) -> TranscriptionConfig:
    fields: dict[str, object] = {"remote_url": "http://mac.local:8000"}
    fields.update(overrides)
    return TranscriptionConfig(**fields)  # type: ignore[arg-type]


def _output_cfg(tmp_path: Path) -> OutputConfig:
    return OutputConfig(work_dir=tmp_path / "work")


def _transcript() -> Transcript:
    return Transcript(paragraphs=[Paragraph(start_s=0, text="Hello.")])


class FakeASRBackend:
    def __init__(self, result: ASRResult | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[Path] = []

    async def transcribe(self, audio_path: Path, *, language=None, chapters=None):
        self.calls.append(audio_path)
        if self.error:
            raise self.error
        assert self.result is not None
        return self.result


class TestT0T1Preferred:
    @pytest.mark.asyncio
    async def test_captions_available_never_probes_the_worker(self, tmp_path: Path) -> None:
        probe_calls = []

        async def probe(cfg):
            probe_calls.append(cfg)
            return True

        def ladder(meta, cfg, *, loosen=False):
            return AcquiredTranscript(transcript=_transcript(), tier="T0")

        result = await acquire_transcript(
            _meta(),
            _acquisition_cfg(),
            _transcription_cfg(),
            _output_cfg(tmp_path),
            _acquire_captions=ladder,
            _probe=probe,
        )
        assert result.tier == "T0"
        assert probe_calls == []  # never needed to ask about the ASR worker


class TestOfflineWorkerDegrades:
    @pytest.mark.asyncio
    async def test_loosened_captions_used_when_worker_is_offline(self, tmp_path: Path) -> None:
        call_log: list[bool] = []

        def ladder(meta, cfg, *, loosen=False):
            call_log.append(loosen)
            if not loosen:
                raise NeedsASR("low_token_density")
            return AcquiredTranscript(transcript=_transcript(), tier="T1", degraded=True)

        async def probe(cfg):
            return False

        result = await acquire_transcript(
            _meta(),
            _acquisition_cfg(),
            _transcription_cfg(degrade_to_captions_when_offline=True),
            _output_cfg(tmp_path),
            _acquire_captions=ladder,
            _probe=probe,
        )
        assert result.tier == "T1"
        assert result.degraded is True
        assert call_log == [False, True]  # strict first, then loosened

    @pytest.mark.asyncio
    async def test_parked_when_degrade_disabled_and_worker_offline(self, tmp_path: Path) -> None:
        def ladder(meta, cfg, *, loosen=False):
            raise NeedsASR("no_captions_available")

        async def probe(cfg):
            return False

        with pytest.raises(ParkedForASR):
            await acquire_transcript(
                _meta(),
                _acquisition_cfg(),
                _transcription_cfg(degrade_to_captions_when_offline=False),
                _output_cfg(tmp_path),
                _acquire_captions=ladder,
                _probe=probe,
            )

    @pytest.mark.asyncio
    async def test_parked_when_loosened_captions_still_fail(self, tmp_path: Path) -> None:
        def ladder(meta, cfg, *, loosen=False):
            raise NeedsASR("no_captions_available")  # fails even loosened

        async def probe(cfg):
            return False

        with pytest.raises(ParkedForASR):
            await acquire_transcript(
                _meta(),
                _acquisition_cfg(),
                _transcription_cfg(),
                _output_cfg(tmp_path),
                _acquire_captions=ladder,
                _probe=probe,
            )


class TestASRRuns:
    @pytest.mark.asyncio
    async def test_worker_available_runs_asr_and_tags_t2(self, tmp_path: Path) -> None:
        def ladder(meta, cfg, *, loosen=False):
            raise NeedsASR("no_captions_available")

        async def probe(cfg):
            return True

        def download(video_id, output_cfg, acquisition_cfg):
            path = Path(output_cfg.work_dir) / f"{video_id}.m4a"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake audio")
            return path

        backend = FakeASRBackend(result=ASRResult(transcript=_transcript()))
        result = await acquire_transcript(
            _meta(),
            _acquisition_cfg(),
            _transcription_cfg(),
            _output_cfg(tmp_path),
            asr_backend=backend,
            _acquire_captions=ladder,
            _probe=probe,
            _download=download,
        )
        assert result.tier == "T2"
        assert result.asr_model == "distil-large-v3"
        assert len(backend.calls) == 1

    @pytest.mark.asyncio
    async def test_audio_is_deleted_after_a_successful_transcription(self, tmp_path: Path) -> None:
        def ladder(meta, cfg, *, loosen=False):
            raise NeedsASR("x")

        async def probe(cfg):
            return True

        written_path = tmp_path / "work" / f"{VID}.m4a"

        def download(video_id, output_cfg, acquisition_cfg):
            written_path.parent.mkdir(parents=True, exist_ok=True)
            written_path.write_bytes(b"fake audio")
            return written_path

        backend = FakeASRBackend(result=ASRResult(transcript=_transcript()))
        await acquire_transcript(
            _meta(),
            _acquisition_cfg(),
            _transcription_cfg(),
            _output_cfg(tmp_path),
            asr_backend=backend,
            _acquire_captions=ladder,
            _probe=probe,
            _download=download,
        )
        assert not written_path.exists()

    @pytest.mark.asyncio
    async def test_keep_audio_true_leaves_the_file(self, tmp_path: Path) -> None:
        def ladder(meta, cfg, *, loosen=False):
            raise NeedsASR("x")

        async def probe(cfg):
            return True

        written_path = tmp_path / "work" / f"{VID}.m4a"

        def download(video_id, output_cfg, acquisition_cfg):
            written_path.parent.mkdir(parents=True, exist_ok=True)
            written_path.write_bytes(b"fake audio")
            return written_path

        output_cfg = OutputConfig(work_dir=tmp_path / "work", keep_audio=True)
        backend = FakeASRBackend(result=ASRResult(transcript=_transcript()))
        await acquire_transcript(
            _meta(),
            _acquisition_cfg(),
            _transcription_cfg(),
            output_cfg,
            asr_backend=backend,
            _acquire_captions=ladder,
            _probe=probe,
            _download=download,
        )
        assert written_path.exists()

    @pytest.mark.asyncio
    async def test_asr_failure_parks_rather_than_raising_a_video_error(
        self, tmp_path: Path
    ) -> None:
        def ladder(meta, cfg, *, loosen=False):
            raise NeedsASR("x")

        async def probe(cfg):
            return True

        def download(video_id, output_cfg, acquisition_cfg):
            path = Path(output_cfg.work_dir) / f"{video_id}.m4a"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake audio")
            return path

        backend = FakeASRBackend(error=ASRUnavailable("connection refused mid-transcription"))
        with pytest.raises(ParkedForASR):
            await acquire_transcript(
                _meta(),
                _acquisition_cfg(),
                _transcription_cfg(),
                _output_cfg(tmp_path),
                asr_backend=backend,
                _acquire_captions=ladder,
                _probe=probe,
                _download=download,
            )


class TestForceASR:
    @pytest.mark.asyncio
    async def test_force_asr_skips_the_caption_ladder_entirely(self, tmp_path: Path) -> None:
        ladder_calls = []

        def ladder(meta, cfg, *, loosen=False):
            ladder_calls.append(loosen)
            return AcquiredTranscript(transcript=_transcript(), tier="T0")

        async def probe(cfg):
            return True

        def download(video_id, output_cfg, acquisition_cfg):
            path = Path(output_cfg.work_dir) / f"{video_id}.m4a"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake audio")
            return path

        backend = FakeASRBackend(result=ASRResult(transcript=_transcript()))
        result = await acquire_transcript(
            _meta(),
            _acquisition_cfg(),
            _transcription_cfg(),
            _output_cfg(tmp_path),
            force_asr=True,
            asr_backend=backend,
            _acquire_captions=ladder,
            _probe=probe,
            _download=download,
        )
        assert result.tier == "T2"
        assert ladder_calls == []  # T0/T1 never attempted


class TestPendingAsrBookkeeping:
    def test_park_sets_state_without_touching_attempts(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        db.execute(
            "INSERT INTO videos (id, video_id, url, canonical_url, created_at, updated_at) "
            "VALUES (?, ?, 'u', 'u', 'now', 'now')",
            (ROW_ID, VID),
        )
        db.commit()

        park_pending_asr(db, ROW_ID)

        row = db.execute("SELECT * FROM videos WHERE id = ?", (ROW_ID,)).fetchone()
        assert row["stage_transcript"] == "pending"
        assert row["asr_queued_at"] is not None
        assert row["attempts"] == "{}"  # untouched

    def test_pending_rows_lists_only_parked_videos(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        for vid in ("a", "b"):
            db.execute(
                "INSERT INTO videos (id, video_id, url, canonical_url, created_at, updated_at) "
                "VALUES (?, ?, 'u', 'u', 'now', 'now')",
                (f"youtube:{vid}", vid),
            )
        db.commit()
        park_pending_asr(db, "youtube:a")

        rows = pending_asr_rows(db)
        assert [r["video_id"] for r in rows] == ["a"]

    def test_clear_removes_the_park(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        db.execute(
            "INSERT INTO videos (id, video_id, url, canonical_url, created_at, updated_at) "
            "VALUES (?, ?, 'u', 'u', 'now', 'now')",
            (ROW_ID, VID),
        )
        db.commit()
        park_pending_asr(db, ROW_ID)
        clear_pending_asr(db, ROW_ID)
        assert pending_asr_rows(db) == []

    def test_stale_pending_alerts_on_age(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        db.execute(
            "INSERT INTO videos (id, video_id, url, canonical_url, "
            "asr_queued_at, created_at, updated_at) VALUES (?, ?, 'u', 'u', ?, 'now', 'now')",
            (ROW_ID, VID, "2020-01-01T00:00:00+00:00"),
        )
        db.commit()
        stale = stale_pending_asr(db, stale_hours=48)
        assert [r["video_id"] for r in stale] == [VID]

    def test_recently_parked_is_not_stale(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        db.execute(
            "INSERT INTO videos (id, video_id, url, canonical_url, created_at, updated_at) "
            "VALUES (?, ?, 'u', 'u', 'now', 'now')",
            (ROW_ID, VID),
        )
        db.commit()
        park_pending_asr(db, ROW_ID)
        assert stale_pending_asr(db, stale_hours=48) == []
