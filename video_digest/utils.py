"""Small shared helpers: time and IDs. Mirrors podcast_agent/utils.py's shape."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    """Serialise to UTC ISO-8601. All storage is UTC."""
    return dt.astimezone(UTC).isoformat()


def iso_now() -> str:
    return iso(utcnow())


def epoch_ms(dt: datetime | None = None) -> int:
    """LiveSync's ctime/mtime unit — epoch milliseconds (obsidian-vault-writer §4)."""
    return int((dt or utcnow()).timestamp() * 1000)


def new_job_id() -> str:
    return uuid.uuid4().hex
