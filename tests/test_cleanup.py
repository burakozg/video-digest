from __future__ import annotations

import time
from pathlib import Path

from video_digest.config import OutputConfig
from video_digest.pipeline.cleanup import sweep_orphaned_media


def _cfg(tmp_path: Path, **overrides: object) -> OutputConfig:
    fields: dict[str, object] = {"work_dir": tmp_path / "work", "orphan_media_hours": 24}
    fields.update(overrides)
    return OutputConfig(**fields)  # type: ignore[arg-type]


def _age(path: Path, hours_ago: float) -> None:
    ts = time.time() - hours_ago * 3600
    import os

    os.utime(path, (ts, ts))


class TestSweep:
    def test_old_audio_files_are_removed(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        cfg.work_dir.mkdir(parents=True)
        stale = cfg.work_dir / "old.m4a"
        stale.write_bytes(b"x")
        _age(stale, 25)

        removed = sweep_orphaned_media(cfg)
        assert removed == [stale]
        assert not stale.exists()

    def test_recent_audio_files_are_kept(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        cfg.work_dir.mkdir(parents=True)
        fresh = cfg.work_dir / "fresh.m4a"
        fresh.write_bytes(b"x")
        _age(fresh, 1)

        removed = sweep_orphaned_media(cfg)
        assert removed == []
        assert fresh.exists()

    def test_non_audio_files_are_never_touched(self, tmp_path: Path) -> None:
        """A sweep must never guess at a file it did not create."""
        cfg = _cfg(tmp_path)
        cfg.work_dir.mkdir(parents=True)
        other = cfg.work_dir / "state.sqlite"
        other.write_bytes(b"x")
        _age(other, 100)

        removed = sweep_orphaned_media(cfg)
        assert removed == []
        assert other.exists()

    def test_missing_work_dir_is_a_no_op(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        assert sweep_orphaned_media(cfg) == []

    def test_subdirectories_are_not_descended_into_or_deleted(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        (cfg.work_dir / "sub").mkdir(parents=True)
        nested = cfg.work_dir / "sub" / "old.m4a"
        nested.write_bytes(b"x")
        _age(nested, 100)

        removed = sweep_orphaned_media(cfg)
        assert removed == []
        assert nested.exists()
