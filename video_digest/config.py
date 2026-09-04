"""Layered configuration: config.yaml (non-secret) + environment (secrets).

Modelled on podcast-digest's config.py (podcast_agent/config.py) and
vault-ask's (vault_ask/config.py) — same shape, ported rather than reinvented.
Invalid configuration is a startup crash, never a half-configured run: every
model uses ``extra="forbid"`` so a typo'd key fails loudly instead of being
silently ignored.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_CONFIG_FILE = "config.yaml"


class Provider(StrEnum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"


#: Providers that send content off the LAN. Gated by ``allow_cloud_fallback``.
CLOUD_PROVIDERS: frozenset[Provider] = frozenset({Provider.OPENROUTER, Provider.ANTHROPIC})

#: litellm route prefix per provider — ``ollama_chat`` (not ``ollama``) because
#: the pipeline always sends chat messages and relies on JSON-mode structured
#: output, which needs /api/chat. Ported verbatim from podcast_agent/config.py.
LITELLM_PREFIX: dict[Provider, str] = {
    Provider.OLLAMA: "ollama_chat",
    Provider.OPENROUTER: "openrouter",
    Provider.ANTHROPIC: "anthropic",
}

DEFAULT_OLLAMA_BASE = "http://localhost:11434"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"must be an http(s) URL, got {value!r}")
    if not parsed.netloc:
        raise ValueError(f"missing host in URL {value!r}")
    return value


# ── vault (§5, §1.1 of the plan) ──────────────────────────────────────────────


class VaultConfig(StrictModel):
    """The vault's CouchDB — the same one Self-hosted LiveSync replicates
    against. Deployment topology, so not console-overridable (there is no
    console): the address is set via VIDEODIGEST_VAULT__COUCHDB_URL, never
    committed.
    """

    couchdb_url: str | None = None
    db: str = Field(default="the_brain", pattern=r"^[a-z][a-z0-9_$()+/-]*$")
    user: str = "videodigest"
    notes_dir: str = "13 video-summaries"
    transcripts_dir: str = "14 video-transcripts"
    topics_dir: str = "99 topics"
    #: The inbox watcher's queue note (design §4.3). Kept inside the notes
    #: folder rather than a separate `00 inbox/`; the `_` prefix pins it to
    #: the top of the folder in Obsidian's default name sort, above every
    #: dated digest note.
    inbox_note: str = "13 video-summaries/_video-queue.md"
    #: Videos a topic must appear in before a new topic page is created.
    #: Reuses clippings-topics' matcher and threshold logic (canonical()).
    topic_creation_threshold: int = Field(default=2, ge=1, le=100)
    timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)

    @field_validator("couchdb_url")
    @classmethod
    def _check_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return _require_http_url(value)

    @field_validator("notes_dir", "transcripts_dir", "topics_dir")
    @classmethod
    def _clean_folder(cls, value: str) -> str:
        cleaned = value.strip("/")
        if not cleaned:
            raise ValueError("a vault folder cannot be empty or '/'")
        # Nesting is allowed, but an empty segment would produce `a//b`, a
        # different vault path from `a/b` that files the note somewhere
        # nobody is looking. Ported check from podcast_agent/config.py.
        if any(not segment.strip() for segment in cleaned.split("/")):
            raise ValueError(f"a vault folder cannot contain an empty segment: {value!r}")
        return cleaned


# ── acquisition (§4, §9) ──────────────────────────────────────────────────────


class AcquisitionConfig(StrictModel):
    max_duration_minutes: int = Field(default=240, ge=1, le=1440)
    #: Preference order for publisher/auto captions (T0/T1).
    subtitle_languages: list[str] = Field(default_factory=lambda: ["en"])
    #: Escape hatch for bot-detection challenges. Not a requirement — see §9.
    cookies_file: Path | None = None
    rate_limit_per_hour: int = Field(default=20, ge=1, le=1000)
    playlist_expansion_cap: int = Field(default=25, ge=1, le=500)


# ── transcription / ASR (§2 S2, plan §1.4) ────────────────────────────────────


class TranscriptionConfig(StrictModel):
    """Remote-only, always — the NAS's realtime factor (0.11) makes local ASR
    a non-option (see podcast_agent/transcripts/asr.py, ported verbatim as
    RemoteASRBackend). Points at the existing speaches container in
    ~/projects/asr-server, not a new worker.
    """

    remote_url: str
    model: str = "distil-large-v3"
    accuracy_model: str = "large-v3"
    vad_filter: bool = True
    #: How often the drain job probes the worker and works pending_asr.
    asr_poll_interval_minutes: int = Field(default=5, ge=1, le=120)
    #: Age past which a still-parked job is worth an alert, not a retry.
    asr_stale_hours: int = Field(default=48, ge=1, le=24 * 30)
    #: Loosen the T1 caption-quality heuristic while the worker is offline —
    #: a decent auto-caption transcript now beats a perfect one on Tuesday.
    degrade_to_captions_when_offline: bool = True
    max_concurrent_asr: int = Field(default=1, ge=1, le=4)
    #: Generous: one HTTP call covers the whole decode of a long video.
    remote_timeout_s: int = Field(default=2700, ge=30, le=21600)

    _check_url = field_validator("remote_url")(_require_http_url)


# ── LLM (ported verbatim from podcast_agent/config.py — plan §1.6) ───────────


class LLMEndpoint(StrictModel):
    provider: Provider
    model: str = Field(min_length=1)
    api_base: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_base")
    @classmethod
    def _base_url(cls, v: str | None) -> str | None:
        return None if v is None else _require_http_url(v)

    @property
    def is_cloud(self) -> bool:
        return self.provider in CLOUD_PROVIDERS

    def litellm_model(self) -> str:
        return f"{LITELLM_PREFIX[self.provider]}/{self.model}"

    def resolved_api_base(self) -> str | None:
        if self.api_base:
            return self.api_base
        return DEFAULT_OLLAMA_BASE if self.provider is Provider.OLLAMA else None


class LLMTierConfig(StrictModel):
    primary: LLMEndpoint
    fallbacks: list[LLMEndpoint] = Field(default_factory=list)
    timeout_s: int = Field(default=120, ge=5, le=3600)
    validation_retries: int = Field(default=2, ge=0, le=5)
    #: Data-sovereignty switch. When false, cloud endpoints are dropped from
    #: the chain entirely — work queues instead of leaving the LAN.
    allow_cloud_fallback: bool = True

    def active_chain(self) -> list[LLMEndpoint]:
        chain = [self.primary, *self.fallbacks]
        if not self.allow_cloud_fallback:
            chain = [e for e in chain if not e.is_cloud]
        return chain

    @model_validator(mode="after")
    def _chain_not_empty_when_local_only(self) -> LLMTierConfig:
        if not self.active_chain():
            raise ValueError(
                "allow_cloud_fallback is false but every configured endpoint is a "
                "cloud provider — this tier could never run"
            )
        return self


class LLMConfig(StrictModel):
    #: Two named tiers: "map" (Pass A, chunk summaries — a volume job suited to
    #: a local model) and "reduce" (Pass B, the quality-sensitive VideoDigest).
    tiers: dict[str, LLMTierConfig]
    log_llm_io: bool = False

    REQUIRED_TIERS: ClassVar[tuple[str, ...]] = ("map", "reduce")

    @model_validator(mode="after")
    def _has_required_tiers(self) -> LLMConfig:
        missing = [t for t in self.REQUIRED_TIERS if t not in self.tiers]
        if missing:
            raise ValueError(f"llm.tiers missing required tier(s): {', '.join(missing)}")
        return self


# ── watchlist (§4 — parsed and validated, deliberately unused in v1) ─────────


class WatchlistChannel(StrictModel):
    channel_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    relevance_filter: bool = False
    min_duration_minutes: int = Field(default=0, ge=0, le=1440)
    force_asr: bool = False


# ── scheduler, API, output, notifications ─────────────────────────────────────


class SchedulerConfig(StrictModel):
    timezone: str = "Europe/Stockholm"
    #: Sweeps orphaned downloaded media older than 24h (§6 S6).
    cleanup_cron: str = "0 * * * *"
    #: Watches the inbox note for bare URL lines (§4, deferred to M6).
    inbox_poll_cron: str = "*/5 * * * *"
    #: How often the queued-job runner (pipeline/runner.py) advances work
    #: (design §6). Seconds, not minutes: a manually-POSTed URL should become
    #: a note promptly, not on the next 5-minute tick.
    job_poll_seconds: int = Field(default=20, ge=2, le=3600)

    @model_validator(mode="after")
    def _crons_parse(self) -> SchedulerConfig:
        from apscheduler.triggers.cron import CronTrigger

        for field in ("cleanup_cron", "inbox_poll_cron"):
            expr = getattr(self, field)
            try:
                CronTrigger.from_crontab(expr, timezone=self.timezone)
            except Exception as exc:
                raise ValueError(f"{field}: invalid cron expression {expr!r} ({exc})") from exc
        return self


class APIConfig(StrictModel):
    host: str = "0.0.0.0"  # noqa: S104 — bound to the container's LAN IP by compose
    port: int = Field(default=8090, ge=1, le=65535)



class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"


class OutputConfig(StrictModel):
    #: SQLite app state (plan §1.3 — not CouchDB).
    db_path: Path = Path("/data/video_digest.sqlite")
    work_dir: Path = Path("/data/work")
    keep_audio: bool = False
    #: Sweep threshold for orphaned media (§6 S6).
    orphan_media_hours: int = Field(default=24, ge=1, le=24 * 30)


class NotificationConfig(StrictModel):
    """Failed-job delivery — same pattern as podcast-digest/security-digest's
    ntfy notifier (§9: "a failed job writes nothing to the vault; send
    failures to the same delivery channel").
    """

    enabled: bool = False
    ntfy_url: str | None = None
    topic: str | None = None
    priority: Literal["min", "low", "default", "high", "urgent"] = "default"
    tags: list[str] = Field(default_factory=lambda: ["movie_camera"])

    @field_validator("ntfy_url")
    @classmethod
    def _url(cls, v: str | None) -> str | None:
        return None if v is None else _require_http_url(v)

    @model_validator(mode="after")
    def _enabled_needs_a_target(self) -> NotificationConfig:
        if self.enabled and not (self.ntfy_url and self.topic):
            raise ValueError(
                "notifications.enabled is true but ntfy_url/topic are not set — "
                "notifications would silently never fire"
            )
        return self


# ── yaml source ────────────────────────────────────────────────────────────


def _yaml_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level of the config file must be a mapping")
    return loaded


class _YamlSource(PydanticBaseSettingsSource):
    """Feeds config.yaml into the settings chain below environment variables."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        return _yaml_settings(self._path)


#: Set by load_settings() before the model is constructed. Module state is the
#: only way to parameterise a pydantic-settings source at class level.
_active_yaml_path: Path = Path(DEFAULT_CONFIG_FILE)


# ── top-level settings ────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Fully-resolved application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VIDEODIGEST_",
        env_nested_delimiter="__",
        extra="forbid",
        env_file=".env",
        env_file_encoding="utf-8",
        #: One .env is shared with docker-compose, which needs its own
        #: non-prefixed vars. Without this, extra="forbid" would refuse to
        #: boot on a correctly-filled .env (podcast_agent/config.py's fix).
        dotenv_filtering="match_prefix",
    )

    vault: VaultConfig = Field(default_factory=VaultConfig)
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    transcription: TranscriptionConfig
    llm: LLMConfig
    watchlist: list[WatchlistChannel] = Field(default_factory=list)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)

    # --- Secrets: environment only, never YAML, never logged ----------------
    admin_api_key: SecretStr | None = None
    ntfy_token: SecretStr | None = None
    vault_couchdb_password: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: init args > env > .env > config.yaml > defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls, _active_yaml_path),
            file_secret_settings,
        )

    @field_validator("watchlist")
    @classmethod
    def _unique_channel_ids(cls, v: list[WatchlistChannel]) -> list[WatchlistChannel]:
        from collections import Counter

        dupes = sorted({c for c, n in Counter(ch.channel_id for ch in v).items() if n > 1})
        if dupes:
            raise ValueError(f"duplicate watchlist channel_id(s): {', '.join(dupes)}")
        return v

    @model_validator(mode="after")
    def _cloud_creds_present(self) -> Settings:
        """A configured cloud endpoint without its API key is a silent-failure
        trap — ported verbatim from podcast_agent/config.py."""
        needed: set[Provider] = set()
        for tier in self.llm.tiers.values():
            for endpoint in tier.active_chain():
                if endpoint.is_cloud:
                    needed.add(endpoint.provider)
        if Provider.OPENROUTER in needed and not self.openrouter_api_key:
            raise ValueError(
                "an openrouter endpoint is active but VIDEODIGEST_OPENROUTER_API_KEY is "
                "unset (set the key, or set allow_cloud_fallback: false for that tier)"
            )
        if Provider.ANTHROPIC in needed and not self.anthropic_api_key:
            raise ValueError(
                "an anthropic endpoint is active but VIDEODIGEST_ANTHROPIC_API_KEY is "
                "unset (set the key, or set allow_cloud_fallback: false for that tier)"
            )
        return self

    def vault_password(self) -> str | None:
        """The vault CouchDB password (env-only secret), unwrapped — or None
        when unset, which `LiveSyncVault` treats as "no vault configured"."""
        return (
            self.vault_couchdb_password.get_secret_value()
            if self.vault_couchdb_password
            else None
        )

    def notification_token(self) -> str | None:
        """The ntfy bearer token (env-only secret), unwrapped — or None."""
        return self.ntfy_token.get_secret_value() if self.ntfy_token else None

    def api_key_for(self, provider: Provider) -> str | None:
        match provider:
            case Provider.OPENROUTER:
                key = self.openrouter_api_key
                return key.get_secret_value() if key else None
            case Provider.ANTHROPIC:
                key = self.anthropic_api_key
                return key.get_secret_value() if key else None
            case _:
                return None


def load_settings(config_file: str | Path | None = None) -> Settings:
    """Build Settings from ``config_file`` (default: $VIDEODIGEST_CONFIG_FILE
    or ./config.yaml)."""
    global _active_yaml_path
    path = Path(config_file or os.environ.get("VIDEODIGEST_CONFIG_FILE", DEFAULT_CONFIG_FILE))
    _active_yaml_path = path
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (used by FastAPI dependencies)."""
    return load_settings()
