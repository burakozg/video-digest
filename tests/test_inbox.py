"""The vault inbox watcher (design §4.3): a bare URL line is enqueued and,
once its note exists, moved out of the queue into the completed note instead
of being rewritten in place — the queue holds only what still needs
attention. Everything else in the queue note is left byte-for-byte.
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
COMPLETED = "13 video-summaries/_video-completed.md"


class FakeVault:
    def __init__(self, notes: dict[str, str | None]) -> None:
        self._notes = dict(notes)
        self.writes: list[tuple[str, str]] = []
        self.fail = False

    async def read_note(self, path: str) -> str | None:
        if self.fail:
            raise VaultUnavailable("couch down")
        return self._notes.get(path)

    async def project(self, path: str, markdown: str, *, mtime_ms: int, merge: bool) -> bool:
        self.writes.append((path, markdown))
        self._notes[path] = markdown
        return True

    def written(self, path: str) -> str | None:
        return self._notes.get(path)


def _cfg() -> tuple[AcquisitionConfig, VaultConfig]:
    return AcquisitionConfig(), VaultConfig(inbox_note=INBOX, completed_note=COMPLETED)


def _seed_video(db: Any, *, note_path: str | None, title: str = "A Video") -> None:
    db.execute(
        "INSERT INTO videos (id, video_id, url, canonical_url, metadata, note_path, "
        "stage_resolve, stage_metadata, created_at, updated_at) "
        "VALUES (?, ?, 'u', 'u', ?, ?, 'done', 'done', 'now', 'now')",
        (f"youtube:{VID}", VID, json.dumps({"title": title}), note_path),
    )
    db.commit()


@pytest.mark.asyncio
async def test_url_with_a_finished_note_moves_to_completed(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    _seed_video(db, note_path=NOTE, title="Great Talk")
    acq, vcfg = _cfg()
    vault = FakeVault({INBOX: f"# Queue\n\n- https://youtu.be/{VID}\n- watch later\n"})

    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 1
    queue = vault.written(INBOX)
    assert f"[[{NOTE}|Great Talk]]" not in queue
    assert "- watch later" in queue  # untouched
    assert queue.startswith("# Queue\n")
    assert queue.endswith("\n")  # trailing newline preserved

    completed = vault.written(COMPLETED)
    assert f"- [[{NOTE}|Great Talk]]" in completed


@pytest.mark.asyncio
async def test_the_queue_can_end_up_fully_empty(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    _seed_video(db, note_path=NOTE, title="Great Talk")
    acq, vcfg = _cfg()
    vault = FakeVault({INBOX: f"- https://youtu.be/{VID}\n"})

    await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert vault.written(INBOX) == "\n"


@pytest.mark.asyncio
async def test_url_without_a_note_yet_is_left_alone(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    _seed_video(db, note_path=None)  # enqueued earlier, not yet written
    acq, vcfg = _cfg()
    vault = FakeVault({INBOX: f"- https://youtu.be/{VID}\n"})

    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 0
    assert vault.writes == []


@pytest.mark.asyncio
async def test_prose_and_headings_are_untouched(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    acq, vcfg = _cfg()
    vault = FakeVault({INBOX: "# My queue\n\nSome notes to self.\n\n## Later\n"})

    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 0
    assert vault.writes == []


@pytest.mark.asyncio
async def test_an_already_linked_line_migrates_out_without_enqueueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A note written before the queue/completed split existed, or a human's
    own link left in the queue by hand — either way it moves, and it must
    never be treated as a URL to (re-)enqueue."""
    db = connect(tmp_path / "s.sqlite")
    acq, vcfg = _cfg()
    vault = FakeVault({INBOX: f"- [[{NOTE}|Great Talk]]\n"})

    def boom(*_a: Any, **_kw: Any) -> list[EnqueueResult]:
        raise AssertionError("a line that is already a wikilink must not be enqueued")

    monkeypatch.setattr("video_digest.pipeline.inbox.enqueue", boom)
    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 1
    assert vault.written(INBOX) == "\n"
    assert f"[[{NOTE}|Great Talk]]" in vault.written(COMPLETED)


@pytest.mark.asyncio
async def test_a_bracket_in_the_title_does_not_hide_the_wikilink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real alias in this vault: "... | [un]prompted 2026]]" — a naive
    `\\[\\[[^\\]]+\\]\\]` stops at the inner `]` and never matches the line at
    all, so the video it names would sit in the queue forever."""
    db = connect(tmp_path / "s.sqlite")
    acq, vcfg = _cfg()
    line = "[[13 video-summaries/x.md|A Talk | [un]prompted 2026]]"
    vault = FakeVault({INBOX: line + "\n"})

    def boom(*_a: Any, **_kw: Any) -> list[EnqueueResult]:
        raise AssertionError("a line that is already a wikilink must not be enqueued")

    monkeypatch.setattr("video_digest.pipeline.inbox.enqueue", boom)
    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 1
    assert vault.written(INBOX) == "\n"
    assert line in vault.written(COMPLETED)


@pytest.mark.asyncio
async def test_completed_note_is_appended_to_not_overwritten(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    _seed_video(db, note_path=NOTE, title="Second Video")
    acq, vcfg = _cfg()
    vault = FakeVault(
        {
            INBOX: f"- https://youtu.be/{VID}\n",
            COMPLETED: "- [[13 video-summaries/older.md|Older Video]]\n",
        }
    )

    await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    completed = vault.written(COMPLETED)
    assert "[[13 video-summaries/older.md|Older Video]]" in completed
    assert f"[[{NOTE}|Second Video]]" in completed
    assert completed.index("Older") < completed.index("Second")


@pytest.mark.asyncio
async def test_missing_inbox_note_is_a_noop(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    acq, vcfg = _cfg()
    assert await poll_inbox(db, FakeVault({}), acq, vcfg) == 0  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_unreachable_vault_is_a_noop(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    acq, vcfg = _cfg()
    vault = FakeVault({INBOX: "- x"})
    vault.fail = True
    assert await poll_inbox(db, vault, acq, vcfg) == 0  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_playlist_line_expands_to_one_bullet_per_note_in_completed(
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
    vault = FakeVault({INBOX: "- https://www.youtube.com/playlist?list=PLxxx\n"})

    def fake_enqueue(*_a: Any, **_kw: Any) -> list[EnqueueResult]:
        return [
            EnqueueResult(video_id="aaa", job_id="j1", status="created"),
            EnqueueResult(video_id="bbb", job_id="j2", status="created"),
        ]

    monkeypatch.setattr("video_digest.pipeline.inbox.enqueue", fake_enqueue)
    changed = await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert changed == 1
    assert vault.written(INBOX) == "\n"
    completed = vault.written(COMPLETED)
    assert "- [[13 video-summaries/aaa.md|Vid 0]]" in completed
    assert "- [[13 video-summaries/bbb.md|Vid 1]]" in completed


@pytest.mark.asyncio
async def test_indentation_and_bullet_style_are_preserved(tmp_path: Path) -> None:
    db = connect(tmp_path / "s.sqlite")
    _seed_video(db, note_path=NOTE, title="T")
    acq, vcfg = _cfg()
    vault = FakeVault({INBOX: f"  * https://youtu.be/{VID}\n"})

    await poll_inbox(db, vault, acq, vcfg)  # type: ignore[arg-type]

    assert vault.written(COMPLETED).startswith("  * [[")
