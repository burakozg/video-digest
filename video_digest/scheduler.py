"""APScheduler wiring.

Every job is `max_instances=1` + `coalesce=True` (matching
`podcast_agent/scheduler.py`'s reasoning) so a slow run can never overlap the
next firing; missed fires collapse into one catch-up run rather than a burst
of them once the process is busy again.
"""

from __future__ import annotations

import sqlite3

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .config import Settings
from .logging_setup import get_logger
from .notify import Notifier
from .pipeline.asr_drain import alert_on_stale_pending_asr, drain_pending_asr
from .pipeline.cleanup import sweep_orphaned_media
from .pipeline.inbox import poll_inbox
from .pipeline.runner import run_due_jobs

log = get_logger(__name__)

#: Late fires (the loop was busy) still run, up to this much after they were
#: due — beyond it, APScheduler treats the fire as missed and waits for the
#: next one instead of running something wildly stale.
MISFIRE_GRACE_S = 600


def build_scheduler(settings: Settings, db: sqlite3.Connection) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=settings.scheduler.timezone)
    notifier = Notifier(settings.notifications, settings.notification_token())

    # Built once, not per fire: the LLM client keeps a per-endpoint cooldown
    # (llm/client.py) that a rebuild every tick would discard, and the Router
    # is not free to construct.
    from .llm.client import LLMClient
    from .vault.livesync import LiveSyncVault

    llm = LLMClient(settings, db)
    vault = LiveSyncVault(settings.vault, settings.vault_password())

    async def _run_jobs() -> None:
        try:
            await run_due_jobs(db, settings, llm=llm, vault=vault, notifier=notifier)
        except Exception:
            log.exception("scheduler.run_due_jobs_failed")

    async def _drain() -> None:
        try:
            await drain_pending_asr(db, settings)
        except Exception:
            log.exception("scheduler.drain_pending_asr_failed")

    async def _cleanup() -> None:
        try:
            sweep_orphaned_media(settings.output)
        except Exception:
            log.exception("scheduler.cleanup_failed")

    async def _inbox() -> None:
        try:
            await poll_inbox(db, vault, settings.acquisition, settings.vault)
        except Exception:
            log.exception("scheduler.inbox_failed")

    async def _stale_check() -> None:
        try:
            stale = alert_on_stale_pending_asr(db, settings)
            if stale:
                # design §5 S2: "alert on age, not on failure". Logged at
                # warning so it surfaces in `docker logs` even with
                # notifications disabled; the notifier no-ops when it is.
                video_ids = [r["video_id"] for r in stale]
                log.warning("asr.stale_pending", count=len(stale), video_ids=video_ids)
                await notifier.asr_stale(stale)
        except Exception:
            log.exception("scheduler.stale_check_failed")

    scheduler.add_job(
        _run_jobs,
        trigger=IntervalTrigger(seconds=settings.scheduler.job_poll_seconds),
        id="job_runner",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=MISFIRE_GRACE_S,
    )
    scheduler.add_job(
        _drain,
        trigger=IntervalTrigger(minutes=settings.transcription.asr_poll_interval_minutes),
        id="asr_drain",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=MISFIRE_GRACE_S,
    )
    scheduler.add_job(
        _cleanup,
        trigger=CronTrigger.from_crontab(
            settings.scheduler.cleanup_cron, timezone=settings.scheduler.timezone
        ),
        id="cleanup",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=MISFIRE_GRACE_S,
    )
    scheduler.add_job(
        _inbox,
        trigger=CronTrigger.from_crontab(
            settings.scheduler.inbox_poll_cron, timezone=settings.scheduler.timezone
        ),
        id="inbox_poll",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=MISFIRE_GRACE_S,
    )
    # Same cadence as the drain job — no point checking staleness more often
    # than the queue itself is worked.
    scheduler.add_job(
        _stale_check,
        trigger=IntervalTrigger(minutes=settings.transcription.asr_poll_interval_minutes),
        id="asr_stale_check",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=MISFIRE_GRACE_S,
    )
    return scheduler
