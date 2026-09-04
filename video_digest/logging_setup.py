"""structlog configuration: JSON to stdout, captured by `docker logs`.

structlog, not stdlib `logging` with an `extra=` dict, because several modules
are ported **verbatim** from podcast-digest (asr.py, llm/client.py — plan
§1.5) and those call ``log.info("event", key=value, ...)`` in structlog's
kwargs style throughout. Adapting every call site to a different logging
idiom would defeat the point of a verbatim port. This is a trimmed copy of
podcast_agent/logging_setup.py: the redaction processor is kept (secrets are
real here too — vault and LLM credentials), but the CouchDB-backed
logbuffer/logstore sinks are dropped — there is no admin console reading them
back, and adding one is not in scope.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from .config import LoggingConfig

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "authorization",
    "credential",
)

_REDACTED = "***redacted***"

#: Keys that match a sensitive substring but are ordinary telemetry.
_NEVER_REDACT = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "estimated_tokens", "tokens"}
)

#: Keys carrying bulk untrusted content — transcripts, prompts, descriptions.
#: Never logged in full.
_BULK_KEYS = ("transcript", "description_raw", "prompt", "response", "summary_md")
_BULK_PREVIEW_CHARS = 200


def _redact_secrets(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    for key in list(event_dict):
        lowered = key.lower()
        if lowered in _NEVER_REDACT:
            continue
        if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
            event_dict[key] = _REDACTED
    return event_dict


def _truncate_bulk(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    for key in list(event_dict):
        if not any(part in key.lower() for part in _BULK_KEYS):
            continue
        value = event_dict[key]
        if isinstance(value, str) and len(value) > _BULK_PREVIEW_CHARS:
            event_dict[key] = f"{value[:_BULK_PREVIEW_CHARS]}…[{len(value)} chars total]"
    return event_dict


#: litellm warns on every deployment construction that a model is absent from
#: its built-in cost map. Irrelevant here: local models cost nothing and no
#: prompt caching is used. Ported reasoning from podcast_agent/logging_setup.py.
_KNOWN_NOISE = (
    "not in built-in cost map",
    "cache cost fields will default to 0",
)


def _drop_known_noise(record: logging.LogRecord) -> bool:
    message = str(record.getMessage())
    return not any(phrase in message for phrase in _KNOWN_NOISE)


def tame_litellm_logging() -> None:
    """Strip litellm's own handler and filter its known-noise warning.

    Called twice: once from configure_logging, and again after litellm is
    imported (it attaches its own handler at import time, after this module
    has already run).
    """
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        if _drop_known_noise not in logger.filters:
            logger.addFilter(_drop_known_noise)


def configure_logging(cfg: LoggingConfig) -> None:
    """Install the structlog pipeline. Idempotent — safe to call more than once."""
    level = logging.getLevelNamesMapping()[cfg.level]

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _redact_secrets,
        _truncate_bulk,
    ]

    renderer: Processor
    if cfg.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    quiet_level = max(logging.WARNING, level)
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "apscheduler.executors.default"):
        logging.getLogger(noisy).setLevel(quiet_level)

    tame_litellm_logging()
    for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(uvicorn_logger).handlers = []
        logging.getLogger(uvicorn_logger).propagate = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_run(run_id: str, **extra: Any) -> None:
    structlog.contextvars.bind_contextvars(run_id=run_id, **extra)


def clear_run_context() -> None:
    structlog.contextvars.clear_contextvars()
