FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# uv itself, pinned to a known release rather than whatever pip resolves.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

# --- dependency layer ---------------------------------------------------
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# --- application layer ---------------------------------------------------
COPY video_digest ./video_digest
COPY config.yaml ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

# Non-root; rootfs is read-only at runtime (docker-compose.nas.yml) — the only
# writable path is /data, bind-mounted. yt-dlp's own scratch space and audio
# downloads live under /data/work (config.py OutputConfig.work_dir), not /tmp,
# for the same reason /tmp on a read-only rootfs needs its own tmpfs entry —
# see the compose file.
RUN groupadd -g 1000 appuser && useradd -g appuser -u 1000 appuser \
    && mkdir -p /data && chown -R appuser:appuser /data
USER appuser

EXPOSE 8090

# No runtime self-update for yt-dlp (main.py's header explains why): pinned in
# the image, version surfaced at /healthz. Bump pyproject.toml's yt-dlp pin and
# rebuild when extraction breaks (design §9).
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=8).status==200 else 1)"

CMD ["python", "-m", "video_digest"]
