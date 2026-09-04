"""The queued-job runner (design §6).

`pipeline/resolve.py::enqueue` runs S0/S1 synchronously and leaves a `jobs`
row at `status='queued'`. This module is what consumes it: `run_due_jobs`
picks up queued jobs and walks each video from wherever its per-stage
checkpoints left off — S2 transcript, S3 normalise, S4 summarise, S5 write —
to a note in the vault.

Resuming from the first non-`done` stage is design §6's contract, so a
failure at S4 never re-downloads media and a re-run costs only the stages
that had not finished. Three non-terminal outcomes leave the job `queued`
for a later pass rather than failing it:

- **parked** — S2 needed the ASR worker and it was asleep. `asr_queued_at`
  is set; `pipeline/asr_drain.py` transcribes it when the Mac is next awake,
  and the next `run_due_jobs` pass resumes from S4.
- **deferred (S4)** — every LLM endpoint in the tier was unreachable
  (`LLMUnavailable`). This is the documented way work waits when
  `allow_cloud_fallback: false` and the local model is down.
- **deferred (S5)** — the vault's CouchDB was unreachable (`VaultUnavailable`).

Scheduled every `scheduler.job_poll_seconds` from `scheduler.py`. Single
runner, strictly sequential — `max_instances=1` on the scheduler job — so no
two videos are ever in-flight at once.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..config import Settings
from ..llm.base import LLMUnavailable, StructuredLLM
from ..llm.models import VideoDigest
from ..logging_setup import get_logger
from ..notify import Notifier
from ..transcripts.normalize import Paragraph, Transcript
from ..utils import iso_now
from ..vault.livesync import LiveSyncVault, VaultUnavailable
from .resolve import metadata_from_row
from .summarize import summarize_video
from .transcript import ParkedForASR, acquire_transcript, park_pending_asr
from .write import fetch_known_topics, write_video_note

log = get_logger(__name__)

#: Static per-stage checkpoint writes — a fixed statement per column rather
#: than an interpolated column name, so nothing dynamic reaches the SQL.
_STAGE_UPDATE: dict[str, str] = {
    "transcript": "UPDATE videos SET stage_transcript = ?, updated_at = ? WHERE id = ?",
    "normalize": "UPDATE videos SET stage_normalize = ?, updated_at = ? WHERE id = ?",
    "summarize": "UPDATE videos SET stage_summarize = ?, updated_at = ? WHERE id = ?",
    "write": "UPDATE videos SET stage_write = ?, updated_at = ? WHERE id = ?",
}

#: How many queued jobs one `run_due_jobs` pass will advance. Sequential and
#: some stages are slow (an ASR download, a reduce call), so a pass is
#: deliberately short; the next scheduled fire picks up the rest.
_DEFAULT_BATCH = 5


def _msg(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _reload(db: sqlite3.Connection, row_id: str) -> sqlite3.Row:
    row: sqlite3.Row = db.execute("SELECT * FROM videos WHERE id = ?", (row_id,)).fetchone()
    return row


def _transcript_from_json(raw: str | None) -> Transcript:
    """Rebuild the S3 `Transcript` from the `[{start_s, text}]` JSON that S2
    (here or the ASR drain job) stored on the row."""
    data = json.loads(raw or "[]")
    return Transcript(
        paragraphs=[Paragraph(start_s=int(p["start_s"]), text=str(p["text"])) for p in data]
    )


def _reduce_model(settings: Settings) -> str:
    """Best-effort name of the model that produced the digest, for the note's
    `summary_model` frontmatter. `summarize_video` returns only the digest,
    so this reads the configured Pass-B primary rather than the actual
    endpoint a fallback may have used."""
    tier = settings.llm.tiers.get("reduce")
    return tier.primary.model if tier else ""


def _set_stage(
    db: sqlite3.Connection,
    row_id: str,
    stage: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    db.execute(_STAGE_UPDATE[stage], (status, iso_now(), row_id))
    if error is not None:
        prev: sqlite3.Row = db.execute(
            "SELECT stage_errors, attempts FROM videos WHERE id = ?", (row_id,)
        ).fetchone()
        errors = json.loads(prev["stage_errors"] or "{}")
        errors[stage] = error
        attempts = json.loads(prev["attempts"] or "{}")
        attempts[stage] = int(attempts.get(stage, 0)) + 1
        db.execute(
            "UPDATE videos SET stage_errors = ?, attempts = ?, last_error = ?, updated_at = ? "
            "WHERE id = ?",
            (json.dumps(errors), json.dumps(attempts), error, iso_now(), row_id),
        )
    db.commit()


def _reset_for_force(db: sqlite3.Connection, row_id: str) -> None:
    """`force=true` resets from S2 (design §6): re-acquire the transcript and
    re-summarise, reusing nothing. Metadata (S0/S1) is kept.

    **`written_at` is deliberately not cleared here**, and must never be: it is
    the key `GET /videos` exports by, and moving it makes an already-imported
    item look new to a consumer tracking read/unread state against it. The
    benign consequence is that a forced re-run drops the item out of the export
    until S5 finishes (it filters on `stage_write = 'done'`), then returns it
    with the same id and the same date, so that state survives.
    """
    db.execute(
        "UPDATE videos SET stage_transcript = 'pending', stage_normalize = 'pending', "
        "stage_summarize = 'pending', stage_write = 'pending', transcript_json = NULL, "
        "transcript_tier = NULL, transcript_tier_degraded = 0, asr_model = NULL, "
        "digest = NULL, asr_queued_at = NULL, updated_at = ? WHERE id = ?",
        (iso_now(), row_id),
    )
    db.commit()


async def run_job(
    db: sqlite3.Connection,
    settings: Settings,
    row: sqlite3.Row,
    *,
    llm: StructuredLLM,
    vault: LiveSyncVault,
    force: bool = False,
    notifier: Notifier | None = None,
    _acquire: Any = None,
    _summarize: Any = None,
    _write: Any = None,
) -> str:
    """Advance one video as far as it will go this call. Returns one of
    ``"done"`` | ``"parked"`` | ``"deferred"`` | ``"failed"``.

    The three trailing `_`-prefixed parameters are test seams (this
    codebase's established pattern — see `pipeline/transcript.py`), not part
    of the intended call surface.
    """
    acquire = _acquire or acquire_transcript
    summarize = _summarize or summarize_video
    write = _write or write_video_note

    row_id = str(row["id"])
    video_id = str(row["video_id"])
    meta = metadata_from_row(row)

    if force:
        _reset_for_force(db, row_id)
        row = _reload(db, row_id)

    # ── S2 transcript (+ S3 normalise: acquire_transcript returns a
    #    normalised Transcript, so the two checkpoints move together here) ──
    if row["stage_transcript"] != "done":
        _set_stage(db, row_id, "transcript", "running")
        try:
            acquired = await acquire(
                meta,
                settings.acquisition,
                settings.transcription,
                settings.output,
                force_asr=bool(row["force_asr"]),
            )
        except ParkedForASR:
            park_pending_asr(db, row_id)  # sets stage_transcript back to 'pending'
            log.info("runner.parked_for_asr", video_id=video_id)
            return "parked"
        except Exception as exc:  # recorded on the row, reported to the notifier, not swallowed
            _set_stage(db, row_id, "transcript", "failed", error=_msg(exc))
            log.exception("runner.transcript_failed", video_id=video_id)
            if notifier:
                await notifier.job_failed(video_id, "transcript", _msg(exc))
            return "failed"

        db.execute(
            "UPDATE videos SET stage_transcript = 'done', stage_normalize = 'done', "
            "transcript_tier = ?, transcript_tier_degraded = ?, transcript_json = ?, "
            "asr_model = ?, asr_queued_at = NULL, updated_at = ? WHERE id = ?",
            (
                acquired.tier,
                int(acquired.degraded),
                json.dumps(
                    [{"start_s": p.start_s, "text": p.text} for p in acquired.transcript.paragraphs]
                ),
                acquired.asr_model,
                iso_now(),
                row_id,
            ),
        )
        db.commit()
        row = _reload(db, row_id)
    elif row["stage_normalize"] != "done":
        # Transcript arrived via the ASR drain job, which only moves
        # stage_transcript. transcript_json is already populated.
        _set_stage(db, row_id, "normalize", "done")

    # ── S4 summarise ─────────────────────────────────────────────────────
    if row["stage_summarize"] != "done":
        _set_stage(db, row_id, "summarize", "running")
        transcript = _transcript_from_json(row["transcript_json"])
        # Best-effort: S4 is documented as not depending on vault reachability
        # (only S5's write does — see the module docstring's "deferred (S5)").
        # A vault outage — or any other failure fetching the vocabulary hint —
        # degrades the reduce prompt, it does not defer the whole video.
        try:
            known_topics = await fetch_known_topics(vault, settings.vault)
        except Exception as exc:  # deliberately broad, see above
            log.warning("runner.known_topics_unavailable", video_id=video_id, error=_msg(exc))
            known_topics = []
        try:
            digest: VideoDigest = await summarize(llm, meta, transcript, known_topics=known_topics)
        except LLMUnavailable as exc:
            # Not a failure: the tier's whole chain is down, work waits.
            _set_stage(db, row_id, "summarize", "pending", error=_msg(exc))
            log.warning("runner.summarize_deferred", video_id=video_id, error=_msg(exc))
            return "deferred"
        except Exception as exc:  # recorded on the row, reported to the notifier, not swallowed
            _set_stage(db, row_id, "summarize", "failed", error=_msg(exc))
            log.exception("runner.summarize_failed", video_id=video_id)
            if notifier:
                await notifier.job_failed(video_id, "summarize", _msg(exc))
            return "failed"

        db.execute(
            "UPDATE videos SET stage_summarize = 'done', digest = ?, updated_at = ? WHERE id = ?",
            (digest.model_dump_json(), iso_now(), row_id),
        )
        db.commit()
        row = _reload(db, row_id)

    # ── S5 vault write ───────────────────────────────────────────────────
    if row["stage_write"] != "done":
        _set_stage(db, row_id, "write", "running")
        transcript = _transcript_from_json(row["transcript_json"])
        digest = VideoDigest.model_validate_json(row["digest"])
        try:
            note_path, _ = await write(
                db,
                vault,
                settings.vault,
                meta,
                digest,
                transcript,
                tier=row["transcript_tier"] or "T1",
                transcript_tier_degraded=bool(row["transcript_tier_degraded"]),
                asr_model=row["asr_model"],
                summary_model=_reduce_model(settings),
            )
        except VaultUnavailable as exc:
            _set_stage(db, row_id, "write", "pending", error=_msg(exc))
            log.warning("runner.write_deferred", video_id=video_id, error=_msg(exc))
            return "deferred"
        except Exception as exc:  # recorded on the row, reported to the notifier, not swallowed
            _set_stage(db, row_id, "write", "failed", error=_msg(exc))
            log.exception("runner.write_failed", video_id=video_id)
            if notifier:
                await notifier.job_failed(video_id, "write", _msg(exc))
            return "failed"

        _set_stage(db, row_id, "write", "done")
        log.info("runner.note_written", video_id=video_id, note_path=note_path)

    return "done"


class RewriteUnavailable(Exception):
    """A rewrite was asked for but the row has no stored digest to render
    from — the video has not been summarised yet."""


async def rewrite_from_stored_digest(
    db: sqlite3.Connection,
    settings: Settings,
    vault: LiveSyncVault,
    video_id: str,
) -> str:
    """Re-render the note from the digest already on the row — S5 only, no
    LLM call (design §6 / §8's `POST /videos/{id}/rewrite`). This is what
    makes template iteration free. Returns the note path.

    Raises `KeyError` for an unknown video, `RewriteUnavailable` when it has
    no digest yet, and `VaultUnavailable` when the write cannot land.
    """
    row = db.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    if row is None:
        raise KeyError(video_id)
    if not row["digest"] or not row["transcript_json"]:
        raise RewriteUnavailable(video_id)

    meta = metadata_from_row(row)
    digest = VideoDigest.model_validate_json(row["digest"])
    transcript = _transcript_from_json(row["transcript_json"])
    note_path, _ = await write_video_note(
        db,
        vault,
        settings.vault,
        meta,
        digest,
        transcript,
        tier=row["transcript_tier"] or "T1",
        transcript_tier_degraded=bool(row["transcript_tier_degraded"]),
        asr_model=row["asr_model"],
        summary_model=_reduce_model(settings),
    )
    db.execute(
        # Deliberately does not touch `written_at` — write_video_note above
        # already COALESCEd it, which keeps the original publish date. Adding
        # it here would move the date on every rewrite and make a consumer
        # treat an item it has already seen as new.
        "UPDATE videos SET stage_write = 'done', updated_at = ? WHERE id = ?",
        (iso_now(), row["id"]),
    )
    db.commit()
    log.info("runner.rewrote_note", video_id=video_id, note_path=note_path)
    return note_path


def _set_job_status(db: sqlite3.Connection, job_id: str, status: str) -> None:
    db.execute(
        "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?", (status, iso_now(), job_id)
    )
    db.commit()


async def run_due_jobs(
    db: sqlite3.Connection,
    settings: Settings,
    *,
    llm: StructuredLLM | None = None,
    vault: LiveSyncVault | None = None,
    notifier: Notifier | None = None,
    limit: int = _DEFAULT_BATCH,
    _run_job: Any = None,
) -> int:
    """Advance up to `limit` queued jobs, oldest first. Returns how many
    reached a finished note this pass.

    Videos parked for ASR (`asr_queued_at` set) are skipped — they are
    `pipeline/asr_drain.py`'s to clear, and reappear here on the pass after
    it does.
    """
    runner = _run_job or run_job

    # 'running' is included so a job orphaned by a mid-pass crash is picked
    # back up: only one runner exists (max_instances=1 on the scheduler job),
    # so anything still 'running' at query time is a leftover, never a live
    # concurrent run. The videos row's stage checkpoints carry the real
    # resume point.
    pending: list[sqlite3.Row] = db.execute(
        "SELECT j.id AS job_id, j.video_id AS video_id, j.force AS force "
        "FROM jobs j JOIN videos v ON v.id = j.video_id "
        "WHERE j.status IN ('queued', 'running') AND v.asr_queued_at IS NULL "
        "ORDER BY j.created_at ASC"
    ).fetchall()
    if not pending:
        return 0

    # One video may have several queued jobs (a repeated force=true). Keep
    # the oldest, drop the rest from this pass.
    seen: set[str] = set()
    queue: list[sqlite3.Row] = []
    for job in pending:
        if job["video_id"] in seen:
            continue
        seen.add(job["video_id"])
        queue.append(job)
        if len(queue) >= limit:
            break

    if llm is None:
        from ..llm.client import LLMClient

        llm = LLMClient(settings, db)
    if vault is None:
        vault = LiveSyncVault(settings.vault, settings.vault_password())

    completed = 0
    for job in queue:
        job_id = str(job["job_id"])
        video_row: sqlite3.Row | None = db.execute(
            "SELECT * FROM videos WHERE id = ?", (str(job["video_id"]),)
        ).fetchone()
        if video_row is None:
            _set_job_status(db, job_id, "failed")
            continue

        _set_job_status(db, job_id, "running")
        try:
            outcome = await runner(
                db,
                settings,
                video_row,
                force=bool(job["force"]),
                llm=llm,
                vault=vault,
                notifier=notifier,
            )
        except Exception:
            log.exception("runner.job_crashed", video_id=job["video_id"])
            _set_job_status(db, job_id, "failed")
            if notifier:
                await notifier.job_failed(str(job["video_id"]), "runner", "unhandled exception")
            continue

        if outcome == "done":
            _set_job_status(db, job_id, "done")
            completed += 1
        elif outcome == "failed":
            _set_job_status(db, job_id, "failed")
        else:  # parked | deferred — try again on a later pass
            _set_job_status(db, job_id, "queued")

    if completed:
        log.info("runner.pass_done", completed=completed, considered=len(queue))
    return completed
