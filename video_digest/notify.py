"""Failure and staleness alerts (design §9 — "silent failure in a vault-writing
app is worse than a loud one").

The same ntfy delivery `podcast-digest` / `security-digest` use: a `POST` to
`{ntfy_url}/{topic}` with the message as the body and `Title` / `Priority` /
`Tags` headers. Best-effort throughout — a notifier that cannot reach ntfy
logs and returns, it never propagates into the pipeline it is reporting on.

Disabled by default (`notifications.enabled: false`); `Settings` refuses to
boot with `enabled: true` and no `ntfy_url`/`topic`, so an enabled notifier
here always has a target.
"""

from __future__ import annotations

import httpx

from .config import NotificationConfig
from .logging_setup import get_logger

log = get_logger(__name__)

#: ntfy hangs the request open for its own reasons sometimes; a failed alert
#: must never hold up the job that raised it.
_TIMEOUT_S = 5.0


class Notifier:
    def __init__(self, cfg: NotificationConfig, token: str | None = None) -> None:
        self._cfg = cfg
        self._token = token

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    async def send(
        self,
        title: str,
        message: str,
        *,
        priority: str | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """Deliver one notification. Returns False when disabled or the POST
        did not succeed — never raises."""
        if not self._cfg.enabled or not self._cfg.ntfy_url or not self._cfg.topic:
            return False

        url = f"{self._cfg.ntfy_url.rstrip('/')}/{self._cfg.topic}"
        headers = {
            "Title": title,
            "Priority": priority or self._cfg.priority,
            "Tags": ",".join(tags or self._cfg.tags),
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                response = await client.post(url, content=message.encode("utf-8"), headers=headers)
        except httpx.HTTPError as exc:
            log.warning("notify.send_failed", title=title, error=f"{type(exc).__name__}: {exc}")
            return False
        if response.status_code >= 400:
            log.warning("notify.send_rejected", title=title, status=response.status_code)
            return False
        log.info("notify.sent", title=title)
        return True

    # ── typed call sites (design §9's alert classes) ──────────────────────

    async def job_failed(self, video_id: str, stage: str, error: str) -> None:
        """A pipeline stage raised — the video produced no note (design §9)."""
        await self.send(
            f"video-digest: {video_id} failed at {stage}",
            error,
            priority="high",
            tags=["movie_camera", "x"],
        )

    async def asr_stale(self, rows: list[dict[str, object]]) -> None:
        """Videos parked for ASR past `asr_stale_hours` — "the Mac has been
        shut for two days, not a broken pipeline" (design §5 S2)."""
        if not rows:
            return
        ids = ", ".join(str(r["video_id"]) for r in rows)
        await self.send(
            f"video-digest: {len(rows)} video(s) waiting on ASR",
            f"Parked longer than the staleness window: {ids}",
            priority="default",
            tags=["movie_camera", "hourglass"],
        )
