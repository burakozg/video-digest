"""Vault inbox watcher (design §4.3).

Polls one note in the notes folder — `vault.inbox_note`, by default
`13 video-summaries/_video-queue.md` (the `_` prefix pins it to the top of
the folder in Obsidian's default name sort, above every
`<YYYY-MM-DD>-<slug>` digest note). Any line carrying a bare video URL that
is not yet a wikilink is enqueued (`origin='inbox'`), and once the runner has
written the note the line is rewritten in place as a wikilink to it.
Everything the human typed that is not a URL line is left untouched.

Enqueue is idempotent: a URL that resolves to a known video dedupes in
`resolve.py` with no network call, so re-processing an already-queued line
every poll costs nothing until its note appears.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3

from ..config import AcquisitionConfig, VaultConfig
from ..logging_setup import get_logger
from ..utils import epoch_ms
from ..vault.livesync import LiveSyncVault, VaultUnavailable
from .resolve import enqueue

log = get_logger(__name__)

_URL = re.compile(r"https?://[^\s<>()\[\]]+")
_WIKILINK = re.compile(r"\[\[[^\]]+\]\]")
#: Leading whitespace plus an optional markdown bullet — preserved when a
#: line is rewritten so a checklist stays a checklist.
_PREFIX = re.compile(r"^(\s*(?:[-*]\s+)?)")


async def poll_inbox(
    db: sqlite3.Connection,
    vault: LiveSyncVault,
    acquisition_cfg: AcquisitionConfig,
    vault_cfg: VaultConfig,
) -> int:
    """Enqueue new URLs in the inbox note and upgrade finished ones to
    wikilinks. Returns the number of lines changed. A missing note, or a
    vault that will not answer, is a no-op (returns 0), not an error.
    """
    path = vault_cfg.inbox_note
    try:
        current = await vault.read_note(path)
    except VaultUnavailable as exc:
        log.warning("inbox.vault_unavailable", error=str(exc))
        return 0
    if current is None:
        return 0

    out: list[str] = []
    changed = 0
    for line in current.splitlines():
        rewritten = _rewrite_line(db, line, acquisition_cfg, vault_cfg)
        if rewritten != line:
            changed += 1
        out.append(rewritten)

    if not changed:
        return 0

    rebuilt = "\n".join(out)
    if current.endswith("\n"):
        rebuilt += "\n"
    await vault.project(path, rebuilt, mtime_ms=epoch_ms(), merge=False)
    log.info("inbox.updated", path=path, lines_changed=changed)
    return changed


def _rewrite_line(
    db: sqlite3.Connection,
    line: str,
    acquisition_cfg: AcquisitionConfig,
    vault_cfg: VaultConfig,
) -> str:
    if _WIKILINK.search(line):
        return line
    match = _URL.search(line)
    if match is None:
        return line
    url = match.group(0)

    try:
        results = enqueue(db, acquisition_cfg, url, origin="inbox")
    except Exception as exc:  # a bad URL or a yt-dlp break must not kill the poll
        log.warning("inbox.enqueue_failed", url=url, error=f"{type(exc).__name__}: {exc}")
        return line

    if not results:
        return line

    links = [_note_link(db, r.video_id) for r in results]
    if any(link is None for link in links):
        # At least one video in this line has no note yet — leave the line as
        # it is and try again next poll.
        return line

    prefix = _PREFIX.match(line)
    head = prefix.group(1) if prefix else ""
    if len(links) == 1:
        return f"{head}{links[0]}"
    # A playlist URL expanded to several videos: one bullet per note.
    bullet = head if head.strip() else "- "
    return "\n".join(f"{bullet}{link}" for link in links)


def _note_link(db: sqlite3.Connection, video_id: str) -> str | None:
    """`[[<note path>|<title>]]` for a video whose note exists, else None.
    Matches the wikilink form the S5 writer uses on topic pages
    (`vault/write.py::_update_topic_page`) — full path, `.md` included."""
    row = db.execute(
        "SELECT note_path, metadata FROM videos WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None or not row["note_path"]:
        return None
    title = ""
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        title = str(json.loads(row["metadata"] or "{}").get("title") or "")
    return f"[[{row['note_path']}|{title}]]" if title else f"[[{row['note_path']}]]"
