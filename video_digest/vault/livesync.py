"""The vault's CouchDB, in Self-hosted LiveSync's document format.

**Ported from `clippings_topics/vault.py`** (`LiveSyncVault`), itself ported
from `podcast_agent/vault.py` — the canonical implementation, with
`taster/backend/app/couchdb_client.py` as the first. The format is not
documented by the plugin: it was reverse-engineered from what a live v0.25
client writes, and is now in production in four projects. Fix bugs in all of
them or none.

Two documents per file: a **chunk**, content-addressed so identical text is
stored once, and an **entry**, keyed by the *lowercased* vault path, listing
its chunks. `list_prefix` — clippings-topics' addition over the original —
is what lets M6's inbox watcher and later a duplicate-note check discover
what is already in a folder rather than being told.

Two plugin settings must stay off, and both break this silently: `encrypt`
(E2EE) and `usePathObfuscation`. This writes plaintext chunks keyed by path.

Uses `video_digest.config.VaultConfig` directly rather than redeclaring a
narrower dataclass — the four attributes this reads (`couchdb_url`, `db`,
`user`, `timeout_s`) already exist there, alongside the vault-folder settings
nothing else here needs.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from ..config import VaultConfig
from ..logging_setup import get_logger
from .notes import merge_owned_section

log = get_logger(__name__)

#: Ids LiveSync treats as chunks live in the ``h:`` namespace; ``h:t`` is the
#: sub-namespace taster claimed for content it writes itself, and sharing it
#: is correct rather than a collision — both sides address chunks by
#: content, so identical text legitimately resolves to one document.
_CHUNK_PREFIX = "h:t"


class VaultUnavailable(Exception):
    """The vault database cannot be reached, or refused the write."""


def _chunk_id(content: str) -> str:
    return _CHUNK_PREFIX + hashlib.sha1(content.encode("utf-8")).hexdigest()[:24]  # noqa: S324


def _q(doc_id: str) -> str:
    # Entry ids are vault paths and contain "/" — left unencoded, CouchDB
    # parses them as db/doc/attachment segments and the write lands elsewhere.
    return quote(doc_id, safe="")


@dataclass(frozen=True)
class Entry:
    """One file in the vault, as the listing sees it."""

    doc_id: str  # lowercased vault path — the CouchDB _id
    path: str  # real case
    rev: str
    deleted: bool
    children: tuple[str, ...]
    mtime: int


class LiveSyncVault:
    def __init__(self, cfg: VaultConfig, password: str | None) -> None:
        self._cfg = cfg
        base = (cfg.couchdb_url or "").rstrip("/")
        self._client = (
            httpx.AsyncClient(
                base_url=base,
                auth=(cfg.user, password or ""),
                timeout=cfg.timeout_s,
            )
            if base
            else None
        )

    @property
    def name(self) -> str:
        return f"vault:{(self._cfg.couchdb_url or '').rstrip('/')}/{self._cfg.db}"

    # ── reading ──────────────────────────────────────────────────────────

    async def list_prefix(self, prefix: str) -> list[Entry]:
        """Every live file under ``prefix``.

        Entries are keyed by lowercased path, so a folder is a key range:
        ``startkey="13 video-summaries/"`` to ``endkey="13 video-summaries0"``
        — ``0`` being the byte after ``/``.

        **``include_docs=true`` is not an optimisation, it is the
        correctness requirement.** LiveSync does not tombstone a deleted
        note: it keeps a live CouchDB document and sets ``deleted: true`` in
        the *body*. So ``_all_docs`` lists deleted notes exactly like present
        ones, and the flag is only visible in the document.
        """
        if self._client is None:
            raise VaultUnavailable("vault.couchdb_url is not set")
        end = prefix[:-1] + chr(ord(prefix[-1]) + 1) if prefix else "￰"
        try:
            response = await self._client.get(
                f"/{self._cfg.db}/_all_docs",
                params={
                    "startkey": f'"{prefix}"',
                    "endkey": f'"{end}"',
                    "include_docs": "true",
                },
            )
        except httpx.HTTPError as exc:
            raise VaultUnavailable(
                f"{self.name} unreachable: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code != 200:
            raise VaultUnavailable(
                f"{self.name} refused a listing: HTTP {response.status_code} {response.text[:200]}"
            )

        entries: list[Entry] = []
        skipped = 0
        for row in response.json().get("rows") or []:
            doc = row.get("doc") or {}
            if doc.get("deleted"):
                skipped += 1
                continue
            entries.append(
                Entry(
                    doc_id=str(row["id"]),
                    path=str(doc.get("path") or row["id"]),
                    rev=str((row.get("value") or {}).get("rev") or ""),
                    deleted=False,
                    children=tuple(str(c) for c in (doc.get("children") or [])),
                    mtime=int(doc.get("mtime") or 0),
                )
            )
        log.info("vault.listed", prefix=prefix, live=len(entries), soft_deleted=skipped)
        # Sorted so anything downstream that iterates is stable run to run —
        # unchanged input must produce byte-identical output.
        return sorted(entries, key=lambda e: e.doc_id)

    async def read(self, entry: Entry) -> str | None:
        """A note's markdown, reassembled from its chunks."""
        return await self._markdown_from(entry.children)

    async def read_note(self, path: str) -> str | None:
        """The note at ``path`` as the vault currently holds it, or None when
        there is no live note there (a missing note, or a soft-deleted one).
        Used by the inbox watcher, which reads one note by a known path
        rather than listing a folder."""
        if self._client is None:
            raise VaultUnavailable("vault.couchdb_url is not set")
        return await self._existing_markdown(path.lower())

    async def ping(self) -> bool:
        """Cheap liveness check for `/healthz` — a GET of the database root.
        Any failure at all reads as "not reachable now"; never raises."""
        if self._client is None:
            return False
        try:
            response = await self._client.get(f"/{self._cfg.db}", timeout=3.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def _existing_markdown(self, entry_id: str) -> str | None:
        """The note as the vault currently holds it, reassembled from its
        chunks. None when there is no live note — including a soft-deleted
        one, so a topic the reader threw away is not quietly rebuilt."""
        entry = await self._get(entry_id)
        if entry is None or entry.get("deleted"):
            return None
        return await self._markdown_from([str(c) for c in (entry.get("children") or [])])

    async def _markdown_from(self, children: list[str] | tuple[str, ...]) -> str | None:
        parts = []
        for chunk_id in children:
            chunk = await self._get(str(chunk_id))
            if chunk is None:
                return None  # torn note; safer to rewrite than to merge into half
            parts.append(str(chunk.get("data") or ""))
        return "".join(parts)

    # ── writing ──────────────────────────────────────────────────────────

    async def project(
        self, path: str, markdown: str, *, mtime_ms: int, merge: bool = True
    ) -> bool:
        """Write one note. False when nothing needed writing.

        ``merge=False`` is the whole-file path (plan §1.2 — digest and
        transcript notes, each owned outright): no read-before-write, the
        given markdown becomes the file. ``merge=True`` is the shared-file
        path (`99 topics/`): reads the vault's current copy and merges only
        this app's marked region into it via `merge_owned_section`.

        Skipped means the note already says exactly this, or a human deleted
        it and that deletion is being respected.
        """
        if self._client is None:
            raise VaultUnavailable("vault.couchdb_url is not set")

        if merge:
            # Against the vault, not against anything of ours: nothing syncs
            # back, so no local copy can know what a person — or another
            # application — wrote into this note.
            current = await self._existing_markdown(path.lower())
            markdown = merge_owned_section(current, markdown)
            if current is not None and markdown == current:
                return False  # our section already says exactly this
        # merge=False (plan §1.2's whole-file notes): no read here. The chunk
        # id is content-addressed, so unchanged markdown reproduces the same
        # id; `_put_chunk` then finds it already live and no-ops, and
        # `_put_entry`'s own 409-conflict path compares `children` and
        # returns False when they already match. "Unchanged input writes
        # nothing" falls out of that without a read-before-write here.

        chunk_id = _chunk_id(markdown)
        await self._put_chunk(chunk_id, markdown)

        entry: dict[str, Any] = {
            "_id": path.lower(),  # LiveSync keys entries by lowercased path
            "path": path,
            "children": [chunk_id],
            "ctime": mtime_ms,
            "mtime": mtime_ms,
            "size": len(markdown.encode("utf-8")),  # BYTES; a mismatch starts a conflict
            "type": "plain",
            "eden": {},
        }
        written = await self._put_entry(entry)
        if written:
            log.info("vault.projected", path=path, bytes=entry["size"])
        return written

    async def _put_chunk(self, chunk_id: str, markdown: str) -> None:
        """Ensure the chunk exists.

        Content-addressed, so a live chunk with this id already holds this
        exact text and is correct as it stands. A soft-deleted one is
        revived unconditionally — unlike an entry, a chunk carries no
        intent: an entry whose children cannot be fetched is a file that
        renders empty.
        """
        body = {"_id": chunk_id, "data": markdown, "type": "leaf"}
        response = await self._put(chunk_id, body)
        if response.status_code != 409:
            return

        existing = await self._get(chunk_id)
        if existing is not None and not existing.get("deleted"):
            return
        rev = existing.get("_rev") if existing else await self._tombstone_rev(chunk_id)
        if rev is None:
            raise VaultUnavailable(f"conflict on chunk {chunk_id} with no revision to take over")
        await self._put_or_raise(chunk_id, {**body, "_rev": rev})

    async def _put_entry(self, entry: dict[str, Any]) -> bool:
        entry_id = str(entry["_id"])
        response = await self._put(entry_id, entry)
        if response.status_code != 409:
            return True

        existing = await self._get(entry_id)
        if existing is None:
            log.info("vault.skipped_deleted", path=entry["path"], deletion="raced")
            return False
        if existing.get("deleted"):
            # What deleting a note in Obsidian produces: LiveSync keeps the
            # document and flags it. Respected.
            log.info("vault.skipped_deleted", path=entry["path"], deletion="soft")
            return False
        if list(existing.get("children") or []) == entry["children"]:
            return False  # already present, byte for byte

        await self._put_or_raise(
            entry_id,
            # ctime preserved, mtime advanced.
            {**entry, "_rev": existing["_rev"], "ctime": existing.get("ctime") or entry["ctime"]},
        )
        return True

    async def _put(self, doc_id: str, body: dict[str, Any]) -> httpx.Response:
        assert self._client is not None
        try:
            response = await self._client.put(f"/{self._cfg.db}/{_q(doc_id)}", json=body)
        except httpx.HTTPError as exc:
            raise VaultUnavailable(
                f"{self.name} unreachable: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code in (201, 202, 409):
            return response
        # Credentials never reach the message — only what was attempted.
        raise VaultUnavailable(
            f"{self.name} refused a write: HTTP {response.status_code} {response.text[:200]}"
        )

    async def _put_or_raise(self, doc_id: str, body: dict[str, Any]) -> None:
        response = await self._put(doc_id, body)
        if response.status_code == 409:
            raise VaultUnavailable(f"{self.name}: repeated conflict writing {doc_id}")

    async def _get(self, doc_id: str) -> dict[str, Any] | None:
        assert self._client is not None
        try:
            response = await self._client.get(f"/{self._cfg.db}/{_q(doc_id)}")
        except httpx.HTTPError as exc:
            raise VaultUnavailable(
                f"{self.name} unreachable: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise VaultUnavailable(
                f"{self.name} refused a read: HTTP {response.status_code} {response.text[:200]}"
            )
        doc: dict[str, Any] = response.json()
        return doc

    async def _tombstone_rev(self, doc_id: str) -> str | None:
        assert self._client is not None
        try:
            response = await self._client.get(
                f"/{self._cfg.db}/{_q(doc_id)}",
                params={"open_revs": "all"},
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise VaultUnavailable(
                f"{self.name} unreachable: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code != 200:
            return None
        for row in response.json():
            ok = row.get("ok") if isinstance(row, dict) else None
            if isinstance(ok, dict) and ok.get("_rev"):
                rev: str = ok["_rev"]
                return rev
        return None

    async def soft_delete(self, doc_id: str) -> bool:
        """Delete a note the way Obsidian does. False if there was nothing
        to delete.

        LiveSync does not remove the document and does not use a CouchDB
        tombstone: it keeps the entry, keeps its ``children``, and sets
        ``deleted: true`` in the *body*. Anything else is invisible to the
        clients — a real ``DELETE`` leaves them holding a file the server
        can no longer describe, and they put it straight back.
        """
        assert self._client is not None
        doc = await self._get(doc_id)
        if doc is None or doc.get("deleted"):
            return False
        await self._put_or_raise(
            doc_id,
            {**doc, "deleted": True, "mtime": int(time.time() * 1000)},
        )
        log.info("vault.soft_deleted", path=doc.get("path") or doc_id)
        return True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
