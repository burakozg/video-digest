from __future__ import annotations

from pathlib import Path

import pytest

from video_digest.config import AcquisitionConfig
from video_digest.db import connect
from video_digest.pipeline.resolve import enqueue
from video_digest.sources.youtube import VideoMetadata, VideoRejected

VID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VID}"


def _meta(video_id: str = VID, **overrides: object) -> VideoMetadata:
    fields: dict[str, object] = {
        "video_id": video_id,
        "title": "A Video",
        "channel": "Some Channel",
        "channel_id": "UCxxxx",
        "duration_s": 300,
        "upload_date": "2026-08-20",
        "description": "desc",
        # A captioned video by default, so enqueue() reports "created" rather
        # than "queued_for_transcription" (design §5 S2). Override to [] for
        # the no-captions path.
        "auto_caption_langs": ["en"],
    }
    fields.update(overrides)
    return VideoMetadata(**fields)  # type: ignore[arg-type]


@pytest.fixture
def db(tmp_path: Path):
    conn = connect(tmp_path / "state.sqlite")
    yield conn
    conn.close()


@pytest.fixture
def cfg() -> AcquisitionConfig:
    return AcquisitionConfig()


class TestHappyPath:
    def test_creates_a_row_and_a_job(self, db, cfg) -> None:
        calls: list[str] = []

        def fetch(url: str) -> VideoMetadata:
            calls.append(url)
            return _meta()

        results = enqueue(db, cfg, URL, _fetch=fetch)
        assert len(results) == 1
        assert results[0].status == "created"
        assert results[0].video_id == VID
        assert results[0].job_id is not None
        assert calls == [URL]

    def test_no_captions_reports_queued_for_transcription(self, db, cfg) -> None:
        results = enqueue(
            db, cfg, URL, _fetch=lambda u: _meta(auto_caption_langs=[], manual_sub_langs=[])
        )
        assert results[0].status == "queued_for_transcription"
        assert results[0].job_id is not None  # still a real, runnable job

        row = db.execute("SELECT * FROM videos WHERE video_id = ?", (VID,)).fetchone()
        assert row["stage_metadata"] == "done"
        assert row["origin"] == "manual"

    def test_watch_url_dedupes_before_the_network_call(self, db, cfg) -> None:
        calls: list[str] = []

        def fetch(url: str) -> VideoMetadata:
            calls.append(url)
            return _meta()

        enqueue(db, cfg, URL, _fetch=fetch)
        results = enqueue(db, cfg, URL, _fetch=fetch)

        assert results[0].status == "existing"
        assert calls == [URL]  # second enqueue never called fetch at all

    def test_force_true_refetches_and_updates(self, db, cfg) -> None:
        enqueue(db, cfg, URL, _fetch=lambda u: _meta(title="Old Title"))
        results = enqueue(db, cfg, URL, force=True, _fetch=lambda u: _meta(title="New Title"))

        assert results[0].status == "created"
        row = db.execute("SELECT * FROM videos WHERE video_id = ?", (VID,)).fetchone()
        assert "New Title" in row["metadata"]

    def test_origin_is_recorded(self, db, cfg) -> None:
        enqueue(db, cfg, URL, origin="inbox", _fetch=lambda u: _meta())
        row = db.execute("SELECT * FROM videos WHERE video_id = ?", (VID,)).fetchone()
        assert row["origin"] == "inbox"


class TestRejection:
    def test_rejection_is_stored_and_reported(self, db, cfg) -> None:
        def fetch(url: str) -> VideoMetadata:
            raise VideoRejected("too_long", "duration exceeds cap", VID)

        results = enqueue(db, cfg, URL, _fetch=fetch)
        assert results[0].status == "rejected"
        assert results[0].rejected_reason == "too_long"

        row = db.execute("SELECT * FROM videos WHERE video_id = ?", (VID,)).fetchone()
        assert row["stage_metadata"] == "failed"
        assert "too_long" in row["stage_errors"]

    def test_repeat_of_a_rejected_url_does_not_refetch(self, db, cfg) -> None:
        calls: list[str] = []

        def fetch(url: str) -> VideoMetadata:
            calls.append(url)
            raise VideoRejected("too_long", "duration exceeds cap", VID)

        enqueue(db, cfg, URL, _fetch=fetch)
        # Second call still dedupes on the pre-fetch id hint from S0, so
        # fetch() is never called again even though the stored row is a
        # rejection rather than a completed video.
        results = enqueue(db, cfg, URL, _fetch=fetch)
        assert results[0].status == "existing"
        assert calls == [URL]

    def test_unresolvable_url_with_no_id_is_not_persisted(self, db, cfg) -> None:
        def fetch(url: str) -> VideoMetadata:
            raise VideoRejected("unresolvable", "yt-dlp could not place this URL")

        results = enqueue(db, cfg, "https://example.com/not-youtube", _fetch=fetch)
        assert results[0].status == "rejected"
        assert db.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0


class TestPlaylist:
    def test_bare_playlist_expands_to_n_jobs(self, db, cfg) -> None:
        urls = [f"https://www.youtube.com/watch?v=vid{i:08d}" for i in range(3)]

        def expand(playlist_id: str, cap: int) -> list[str]:
            assert playlist_id == "PLxxxx"
            return urls

        def fetch(url: str) -> VideoMetadata:
            video_id = url.rsplit("=", 1)[1]
            return _meta(video_id=video_id)

        results = enqueue(
            db,
            cfg,
            "https://www.youtube.com/playlist?list=PLxxxx",
            _fetch=fetch,
            _expand=expand,
        )
        assert len(results) == 3
        assert {r.status for r in results} == {"created"}
        assert db.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 3


class TestFallbackForUnrecognisedShapes:
    def test_non_standard_url_still_reaches_fetch(self, db, cfg) -> None:
        calls: list[str] = []

        def fetch(url: str) -> VideoMetadata:
            calls.append(url)
            return _meta()

        weird_url = "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ"
        results = enqueue(db, cfg, weird_url, _fetch=fetch)
        assert calls == [weird_url]
        assert results[0].status == "created"
