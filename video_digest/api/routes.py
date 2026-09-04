"""Admin and health API. LAN-only (plan §1.10 — no port-forward, no Tailscale
for v1). All routes except /healthz require the admin key (auth.require_api_key).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sqlite3
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings
from ..logging_setup import get_logger
from ..pipeline.resolve import EnqueueResult, enqueue
from ..pipeline.runner import RewriteUnavailable, rewrite_from_stored_digest
from ..pipeline.transcript import probe_asr_worker
from ..vault.livesync import LiveSyncVault, VaultUnavailable
from .auth import require_api_key

log = get_logger(__name__)

health_router = APIRouter()
api_router = APIRouter(dependencies=[Depends(require_api_key)])


@health_router.get("/healthz")
async def healthz(request: Request) -> dict[str, object]:
    """Dependency checks (design §8): SQLite reachable, work dir writable, the
    vault's CouchDB reachable, the remote ASR worker reachable, and the
    installed yt-dlp version (pinned in the image — see main.py on why there
    is no runtime self-update).

    Never 500s on a dependency being down — that is what the body reports.
    The ASR worker being asleep is the *expected* state (design §5 S2), so it
    is reported but does not move the top-level status; the vault does.
    """
    settings: Settings = request.app.state.settings
    db: sqlite3.Connection = request.app.state.db
    vault: LiveSyncVault = request.app.state.vault

    checks: dict[str, str] = {}

    try:
        db.execute("SELECT 1").fetchone()
        checks["db"] = "ok"
    except sqlite3.Error as exc:
        checks["db"] = f"error: {exc}"

    work_dir = settings.output.work_dir
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        probe = work_dir / ".healthz-probe"
        probe.write_text("ok")
        probe.unlink()
        checks["work_dir"] = "ok"
    except OSError as exc:
        checks["work_dir"] = f"error: {exc}"

    # The two network probes run concurrently and each caps its own timeout
    # (3s vault, 3s ASR) so /healthz answers well inside the container
    # healthcheck's 8s budget even with both down.
    vault_ok, asr_ok = await asyncio.gather(
        vault.ping(),
        probe_asr_worker(settings.transcription, timeout_s=3.0),
    )
    if settings.vault.couchdb_url is None:
        checks["vault"] = "not configured"
    else:
        checks["vault"] = "ok" if vault_ok else "unreachable"

    healthy = all(v in ("ok", "not configured") for v in checks.values())
    return {
        "status": "ok" if healthy else "degraded",
        "version": __version__,
        "yt_dlp_version": request.app.state.yt_dlp_version,
        "checks": checks,
        "asr_worker": "ok" if asr_ok else "asleep",
    }


# ── /jobs (design §8) ──────────────────────────────────────────────────────


class JobRequest(BaseModel):
    url: str = Field(min_length=1)
    force: bool = False
    force_asr: bool = False


class JobResult(BaseModel):
    video_id: str
    job_id: str | None
    status: str
    existing_note: str | None = None
    rejected_reason: str | None = None

    @classmethod
    def from_enqueue(cls, r: EnqueueResult) -> JobResult:
        return cls(
            video_id=r.video_id,
            job_id=r.job_id,
            status=r.status,
            existing_note=r.existing_note_path,
            rejected_reason=r.rejected_reason,
        )


class JobResponse(BaseModel):
    """Always carries `jobs` (one entry per video). The single-result fields
    are flattened to the top level too — design §8's documented shape for the
    common case of one URL in, one job out — and left null for a playlist
    expansion (design §5 S0), where `jobs` is what a caller should read.
    """

    jobs: list[JobResult]
    video_id: str | None = None
    job_id: str | None = None
    status: str | None = None
    existing_note: str | None = None


@api_router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(body: JobRequest, request: Request) -> JobResponse:
    settings: Settings = request.app.state.settings
    db: sqlite3.Connection = request.app.state.db

    results = enqueue(
        db,
        settings.acquisition,
        body.url,
        origin="manual",
        force=body.force,
        force_asr=body.force_asr,
    )
    jobs = [JobResult.from_enqueue(r) for r in results]
    if len(jobs) == 1:
        only = jobs[0]
        return JobResponse(
            jobs=jobs,
            video_id=only.video_id,
            job_id=only.job_id,
            status=only.status,
            existing_note=only.existing_note,
        )
    return JobResponse(jobs=jobs)


@api_router.get("/videos/{video_id}")
async def get_video(video_id: str, request: Request) -> dict[str, object]:
    row = request.app.state.db.execute(
        "SELECT * FROM videos WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown video_id")
    return dict(row)


# ── /videos — the export a sibling service imports from (design §8) ────────
#
# podcast-digest pulls this to show summaries in its own reader. Deliberately
# a plain authenticated JSON route rather than a feed: this is service to
# service, so the existing X-API-Key header works and nothing needs a
# credential in a URL or an unauthenticated surface.


#: Rows per page. Bounded because the payload carries whole summaries.
_VIDEOS_PAGE_MAX = 200


def _cursor_timestamp(since: str) -> str:
    """Validate the cursor, repairing the one corruption it reliably suffers.

    `written_at` is `2026-05-05T00:00:00+00:00`. In a query string `+` means
    space, so a client that interpolates the value into a URL without
    encoding it sends `...T00:00:00 00:00` — which matches no row, returns an
    empty page, and makes the importer stop early having silently lost
    everything after the cursor. A space at the offset separator can only
    have come from an unencoded `+`, so it is unambiguous to restore.

    Anything that still will not parse is a 400 rather than an empty page:
    for a paging client, "no results" and "your cursor was nonsense" must not
    look the same.
    """
    repaired = re.sub(r"(?<=\d) (?=\d{2}:\d{2}$)", "+", since)
    try:
        datetime.fromisoformat(repaired)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"since must be an ISO-8601 timestamp, got {since!r}",
        ) from None
    return repaired


@api_router.get("/videos")
async def list_videos(
    request: Request,
    since: str | None = None,
    since_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=_VIDEOS_PAGE_MAX),
) -> dict[str, object]:
    """Finished summaries, newest first, for an importer to mirror.

    Excludes `transcript_json` — by far the largest column, and the importer
    wants the summary, not the source. `transcript_path` points at the vault
    note for anything that does want it.

    The cursor is the **(written_at, id) pair**, not `written_at` alone.
    Migration 2 backfilled `written_at` from `created_at`, and a playlist
    expansion stamps several rows in the same second, so a strict `<` on the
    timestamp alone would silently skip every row sharing a boundary
    timestamp with the last page. Pass back `next.since` and `next.since_id`
    from the previous response.
    """
    db: sqlite3.Connection = request.app.state.db
    if since is not None:
        since = _cursor_timestamp(since)

    sql = (
        "SELECT id, video_id, adapter, note_path, transcript_path, written_at, updated_at, "
        "metadata, digest FROM videos "
        "WHERE stage_write = 'done' AND written_at IS NOT NULL AND digest IS NOT NULL"
    )
    params: list[object] = []
    if since is not None:
        # Descending page: "older than the cursor", with id breaking a tie.
        sql += " AND (written_at < ? OR (written_at = ? AND id < ?))"
        params += [since, since, since_id or ""]
    sql += " ORDER BY written_at DESC, id DESC LIMIT ?"
    params.append(limit)

    videos: list[dict[str, object]] = []
    for row in db.execute(sql, params):
        try:
            metadata = json.loads(row["metadata"] or "{}")
            digest = json.loads(row["digest"])
        except (json.JSONDecodeError, TypeError):
            # One malformed row must not fail the whole page for the importer.
            log.warning("videos.row_skipped", video_id=row["video_id"])
            continue
        videos.append(
            {
                "id": row["id"],
                "video_id": row["video_id"],
                "adapter": row["adapter"],
                "note_path": row["note_path"],
                "transcript_path": row["transcript_path"],
                "written_at": row["written_at"],
                "updated_at": row["updated_at"],
                "metadata": metadata,
                "digest": digest,
            }
        )

    last = videos[-1] if videos else None
    return {
        "videos": videos,
        "count": len(videos),
        # Absent when the page was not full — nothing more to fetch.
        "next": (
            {"since": last["written_at"], "since_id": last["id"]}
            if last is not None and len(videos) == limit
            else None
        ),
    }


# ── /jobs/{id} — stage status (design §8) ─────────────────────────────────

_STAGES = ("resolve", "metadata", "transcript", "normalize", "summarize", "write")


@api_router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict[str, object]:
    """Where a job is: its `jobs.status`, each pipeline stage's state, any
    per-stage error, and the note path once S5 has run. What the iOS
    Shortcut polls after a 202."""
    db: sqlite3.Connection = request.app.state.db
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown job_id")
    video = db.execute("SELECT * FROM videos WHERE id = ?", (job["video_id"],)).fetchone()
    if video is None:  # a job always has its video row
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job has no video row")

    stage_errors = json.loads(video["stage_errors"] or "{}")
    return {
        "job_id": job_id,
        "video_id": video["video_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "stages": {name: video[f"stage_{name}"] for name in _STAGES},
        "stage_errors": stage_errors,
        "last_error": video["last_error"],
        "transcript_tier": video["transcript_tier"],
        "pending_asr_since": video["asr_queued_at"],
        "note_path": video["note_path"],
    }


# ── /videos/{id}/rewrite — re-render from the stored digest (design §8) ────


@api_router.post("/videos/{video_id}/rewrite", status_code=status.HTTP_200_OK)
async def rewrite_video(video_id: str, request: Request) -> dict[str, object]:
    """S5 only: re-render the note from the digest already on the row, no LLM
    call (design §6 — "template iteration is free"). 409 if the video has
    not been summarised yet, 503 if the vault will not accept the write."""
    settings: Settings = request.app.state.settings
    db: sqlite3.Connection = request.app.state.db
    vault: LiveSyncVault = request.app.state.vault
    try:
        note_path = await rewrite_from_stored_digest(db, settings, vault, video_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="unknown video_id"
        ) from exc
    except RewriteUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="video has no stored digest yet — run the pipeline first",
        ) from exc
    except VaultUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"vault write failed: {exc}"
        ) from exc
    return {"video_id": video_id, "note_path": note_path}


# ── /metrics (design §8) ─────────────────────────────────────────────────


@api_router.get("/metrics")
async def metrics(request: Request) -> dict[str, object]:
    """Counts by stage, transcript-tier distribution, minutes of audio sent
    to ASR, and LLM call/token/cost totals. "If T2 is firing on 80% of
    videos, either the heuristic is wrong or the ASR bill needs a policy
    change" (design §8)."""
    db: sqlite3.Connection = request.app.state.db

    def _counts(sql: str) -> dict[str, int]:
        return {str(k): int(n) for k, n in db.execute(sql).fetchall()}

    write_stage = _counts("SELECT stage_write, COUNT(*) FROM videos GROUP BY stage_write")
    tiers = _counts(
        "SELECT COALESCE(transcript_tier, 'none'), COUNT(*) FROM videos GROUP BY transcript_tier"
    )
    jobs = _counts("SELECT status, COUNT(*) FROM jobs GROUP BY status")

    asr_seconds = 0
    for (md,) in db.execute("SELECT metadata FROM videos WHERE transcript_tier = 'T2'"):
        with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
            asr_seconds += int(json.loads(md or "{}").get("duration_s") or 0)

    llm_rows = db.execute(
        "SELECT value, meta FROM metric_events WHERE kind = 'llm_call'"
    ).fetchall()
    llm_calls = len(llm_rows)
    cost_usd = 0.0
    input_tokens = 0
    output_tokens = 0
    by_tier: dict[str, int] = {}
    for value, meta_json in llm_rows:
        cost_usd += float(value or 0)
        meta = json.loads(meta_json or "{}")
        input_tokens += int(meta.get("input_tokens") or 0)
        output_tokens += int(meta.get("output_tokens") or 0)
        tier = str(meta.get("tier") or "unknown")
        by_tier[tier] = by_tier.get(tier, 0) + 1

    return {
        "videos_total": sum(write_stage.values()),
        "write_stage": write_stage,
        "transcript_tiers": tiers,
        "pending_asr": db.execute(
            "SELECT COUNT(*) FROM videos WHERE asr_queued_at IS NOT NULL"
        ).fetchone()[0],
        "asr_audio_minutes": round(asr_seconds / 60, 1),
        "jobs": jobs,
        "llm": {
            "calls": llm_calls,
            "cost_usd": round(cost_usd, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "calls_by_tier": by_tier,
        },
    }
