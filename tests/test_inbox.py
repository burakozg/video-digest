"""The vault inbox watcher (design §4.3): a bare URL line is enqueued and,
once its note exists, rewritten in place as a wikilink; everything else in
the note is left byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from video_digest.config import AcquisitionConfig, VaultConfig
from video_digest.db import connect
from video_digest.pipeline.inbox import poll_inbox
from video_digest.pipeline.resolve import EnqueueResult
from video_digest.vault.livesync import VaultUnavailable

VID = "dQw4w9WgXcQ"
NOTE = "13 video-summaries/2026-08-20-a-video.md"
INBOX = "13 video-summaries/_video-queue.md"


class FakeVault:
    def __init__(self, note: str | None) -> None:
        self._note = note
        self.writes: list[tuple[str, str]] = []
        self.fail = False

    async def read_note(self, path: str) -> str | None:
        if self.fail:
            raise VaultUnavailable("couch down")
        return self._note

    async def project(self, path: str, markdown: str, *, mtime_ms: int, merge: bool) -> bool:
        self.writes.append((path, markdown))
        self._note = markdown
        return True


def _cfg() -> tuple[AcquisitionConfig, VaultConfig]:
    return AcquisitionConfig(), VaultConfig(inbox_note=INBOX)


def _seed_video(db: Any, *, note_path: str | None, title: str = "A Video") -> None:
    db.execute(
        "INSERT INTO videos (id, video_id, url, canonical_url, metadata, note_path, "
        "stage_resolve, stage_metadata, created_at, updated_at) "
        "VALUES (?, ?, 'u', 'u', ?, ?, 'done', 'done', 'now', 'now')",
        (f"youtube:{VID}", VID, json.dumps({"title": title}), note_path),
    )
    db.commit()


@pytest.mark.asyncio
async def test_url_with_a_finished_note_becomes_a_wikilink(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    _seed_video(db, note_path=NOTE, title="Great Talk")
    acq, vcfg = _cfg()
    vault = FakeVault(f"# Queue\n\n- https://youtu.be/{VID}\n- watch later\n")

    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 1
    assert vault.writes[0][0] == INBOX
    new = vault.writes[0][1]
    assert f"- [[{NOTE}|Great Talk]]" in new
    assert "- watch later" in new  # untouched
    assert new.startswith("# Queue\n")
    assert new.endswith("\n")  # trailing newline preserved


@pytest.mark.asyncio
async def test_url_without_a_note_yet_is_left_alone(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    _seed_video(db, note_path=None)  # enqueued earlier, not yet written
    acq, vcfg = _cfg()
    vault = FakeVault(f"- https://youtu.be/{VID}\n")

    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 0
    assert vault.writes == []


@pytest.mark.asyncio
async def test_prose_and_headings_are_untouched(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    acq, vcfg = _cfg()
    vault = FakeVault("# My queue\n\nSome notes to self.\n\n## Later\n")

    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 0
    assert vault.writes == []


@pytest.mark.asyncio
async def test_already_linked_line_is_not_reprocessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = connect(tmp_path / "s.sqlite")
    acq, vcfg = _cfg()
    vault = FakeVault(f"- [[{NOTE}|Great Talk]]\n")

    def boom(*_a: Any, **_kw: Any) -> list[EnqueueResult]:
        raise AssertionError("a line that is already a wikilink must not be enqueued")

    monkeypatch.setattr("video_digest.pipeline.inbox.enqueue", boom)
    assert await poll_inbox(db, vault, acq, vcfg) == 0  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_missing_inbox_note_is_a_noop(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    acq, vcfg = _cfg()
    assert await poll_inbox(db, FakeVault(None), acq, vcfg) == 0  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_unreachable_vault_is_a_noop(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    acq, vcfg = _cfg()
    vault = FakeVault("- x")
    vault.fail = True
    assert await poll_inbox(db, vault, acq, vcfg) == 0  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_playlist_line_expands_to_one_bullet_per_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = connect(tmp_path / "s.sqlite")
    for i, vid in enumerate(("aaa", "bbb")):
        db.execute(
            "INSERT INTO videos (id, video_id, url, canonical_url, metadata, note_path, "
            "stage_resolve, stage_metadata, created_at, updated_at) "
            "VALUES (?, ?, 'u', 'u', ?, ?, 'done', 'done', 'now', 'now')",
            (f"youtube:{vid}", vid, json.dumps({"title": f"Vid {i}"}), f"13 video-summaries/{vid}.md"),
        )
    db.commit()
    acq, vcfg = _cfg()
    vault = FakeVault("- https://www.youtube.com/playlist?list=PLxxx\n")

    def fake_enqueue(*_a: Any, **_kw: Any) -> list[EnqueueResult]:
        return [
            EnqueueResult(video_id="aaa", job_id="j1", status="created"),
            EnqueueResult(video_id="bbb", job_id="j2", status="created"),
        ]

    monkeypatch.setattr("video_digest.pipeline.inbox.enqueue", fake_enqueue)
    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 1
    new = vault.writes[0][1]
    assert "- [[13 video-summaries/aaa.md|Vid 0]]" in new
    assert "- [[13 video-summaries/bbb.md|Vid 1]]" in new


@pytest.mark.asyncio
async def test_indentation_and_bullet_style_are_preserved(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    _seed_video(db, note_path=NOTE, title="T")
    acq, vcfg = _cfg()
    vault = FakeVault(f"  * https://youtu.be/{VID}\n")

    await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert vault.writes[0][1].startswith("  * [[")
