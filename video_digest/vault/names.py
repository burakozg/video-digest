"""Filename pinning (obsidian-vault-writer skill §2): "derive a shared note's
filename once, store it, and never recompute it." Backed by `db.py`'s
`note_names` / `topic_note_names` tables — fill gaps only, never reassign.
"""

from __future__ import annotations

import sqlite3

from ..sanitize import slugify


def get_note_filename(db: sqlite3.Connection, video_id: str) -> str | None:
    row = db.execute(
        "SELECT filename FROM note_names WHERE video_id = ?", (video_id,)
    ).fetchone()
    return str(row["filename"]) if row else None


def resolve_note_filename(
    db: sqlite3.Connection, video_id: str, *, published: str, title: str
) -> str:
    """`<video_id> -> filename`, pinned on first call and returned unchanged
    on every later one — a corrected title never moves the file."""
    existing = get_note_filename(db, video_id)
    if existing is not None:
        return existing

    date = published[:10] if published else "undated"
    base = f"{date}-{slugify(title)}"
    candidate = base
    suffix = 2
    while db.execute(
        "SELECT 1 FROM note_names WHERE filename = ?", (candidate,)
    ).fetchone():
        candidate = f"{base}-{suffix}"
        suffix += 1

    db.execute(
        "INSERT INTO note_names (video_id, filename) VALUES (?, ?)", (video_id, candidate)
    )
    db.commit()
    return candidate


def get_topic_filename(db: sqlite3.Connection, topic_key: str) -> str | None:
    row = db.execute(
        "SELECT filename FROM topic_note_names WHERE topic_key = ?", (topic_key,)
    ).fetchone()
    return str(row["filename"]) if row else None


def adopt_topic_filename(db: sqlite3.Connection, topic_key: str, filename: str) -> None:
    """Pin a filename this app did not choose — an existing `99 topics/` page
    another writer already created. Adopting rather than reslugifying is what
    lets this app link to `crowdstrike.md` instead of minting a second
    `crowd-strike.md` beside it."""
    db.execute(
        "INSERT OR IGNORE INTO topic_note_names (topic_key, filename) VALUES (?, ?)",
        (topic_key, filename),
    )
    db.commit()


def resolve_topic_filename(db: sqlite3.Connection, topic_key: str, *, title: str) -> str:
    existing = get_topic_filename(db, topic_key)
    if existing is not None:
        return existing

    base = slugify(title)
    candidate = base
    suffix = 2
    while True:
        row = db.execute(
            "SELECT topic_key FROM topic_note_names WHERE filename = ?", (candidate,)
        ).fetchone()
        if row is None or row["topic_key"] == topic_key:
            break
        candidate = f"{base}-{suffix}"
        suffix += 1

    db.execute(
        "INSERT INTO topic_note_names (topic_key, filename) VALUES (?, ?)",
        (topic_key, candidate),
    )
    db.commit()
    return candidate
