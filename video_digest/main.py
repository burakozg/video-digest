"""Application assembly and entrypoint.

Wires configuration, SQLite state and the API together. Invalid configuration
crashes here with a readable message rather than starting half-configured
(config.py's extra="forbid" + validators do the crashing; this module is where
it happens at process start).

No runtime self-update for yt-dlp (design §9 proposes a weekly check): the
container's rootfs is read-only (docker-compose.nas.yml), which rules it out
by construction. The version is pinned in the image and rebuilt instead — see
Dockerfile — and surfaced at /healthz so staleness is at least visible.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from . import __version__
from .api.routes import api_router, health_router
from .config import Settings, load_settings
from .db import connect
from .logging_setup import configure_logging, get_logger
from .scheduler import build_scheduler
from .vault.livesync import LiveSyncVault

log = get_logger(__name__)


def _yt_dlp_version() -> str:
    try:
        return importlib.metadata.version("yt-dlp")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _default_gateway() -> str | None:
    try:
        with open("/proc/net/route") as fh:
            for line in fh.readlines()[1:]:
                fields = line.split()
                if len(fields) > 2 and fields[1] == "00000000":
                    raw = int(fields[2], 16)
                    return ".".join(str((raw >> (8 * i)) & 0xFF) for i in range(4))
    except OSError:
        return None
    return None


async def _wake_arp() -> None:
    """Send one outbound packet so the LAN can reach us right after a start.

    Ported from vault-ask's api/app.py::_wake_arp. Even with the MAC pinned
    (docker-compose.nas.yml), a container joining this NAS's macvlan can come
    back reachable-from-the-outside only after something inside it sends a
    packet — the healthcheck alone (against localhost) never does. Best
    effort: the connection is expected to fail, since nothing need listen on
    the gateway — sending the packet is the whole point.
    """
    gateway = _default_gateway()
    if gateway is None:
        return
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(gateway, 80), timeout=3.0)
        writer.close()
    except (TimeoutError, OSError):
        pass
    log.info("arp_wake.sent", gateway=gateway)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.logging)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("startup", version=__version__)
        await _wake_arp()
        scheduler = build_scheduler(settings, app.state.db)
        scheduler.start()
        yield
        scheduler.shutdown(wait=False)
        log.info("shutdown")

    app = FastAPI(title="video-digest", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.db = connect(settings.output.db_path)
    app.state.admin_api_key = (
        settings.admin_api_key.get_secret_value() if settings.admin_api_key else None
    )
    app.state.yt_dlp_version = _yt_dlp_version()
    # Constructing this opens no connection (LiveSyncVault is lazy); it is
    # here so /healthz can ping the vault and /videos/{id}/rewrite can write.
    app.state.vault = LiveSyncVault(settings.vault, settings.vault_password())

    app.include_router(health_router)
    app.include_router(api_router)
    return app


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.api.host,
        port=settings.api.port,
        log_config=None,  # our own structlog pipeline owns stdout
    )


if __name__ == "__main__":
    main()
