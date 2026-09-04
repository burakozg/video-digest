"""litellm + instructor implementation of `StructuredLLM`.

**Ported from `podcast_agent/llm/client.py`** (podcast-digest@fac8a1f), with
two adaptations: telemetry writes to this app's own SQLite `metric_events`
table instead of a CouchDB `llm_call` document (plan §1.3 — no CouchDB app
state here), and `episode_id` is renamed `video_id` throughout. The fallback
mechanics, the cooldown, and `INSTRUCTOR_MODE` are unchanged — those are the
two hard-won details the plan calls out to carry over verbatim (§1.6).

Fallback semantics: fallback triggers on timeout, connection error, 5xx, or
2x validation failure. litellm's Router can only fail over on exceptions, so
it cannot express the validation-count trigger. The endpoint chain is
therefore walked here, with the Router used as the transport and provider
abstraction. One deployment group is registered per endpoint so a call can
target an exact endpoint:

    map        -> primary
    map__fb0   -> first fallback
    map__fb1   -> second fallback

Cloud endpoints are dropped from the chain entirely when a tier sets
``allow_cloud_fallback: false``, so no content can leave the LAN by accident.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Final, TypeVar, cast

import instructor
import litellm
from instructor import AsyncInstructor

# Imported explicitly: `instructor.exceptions` is not exposed as an attribute
# of the top-level package, so `instructor.exceptions.X` in an except clause
# raises AttributeError *while handling* the real error and masks it.
from instructor.core import InstructorRetryException
from litellm import exceptions as litellm_exceptions
from litellm.router import Router
from pydantic import BaseModel, ValidationError

from ..config import LLMEndpoint, Provider, Settings
from ..logging_setup import get_logger, tame_litellm_logging
from ..utils import iso_now
from .base import LLMUnavailable
from .models import CallMeta

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# litellm chatters on import and phones home for version checks by default.
litellm.telemetry = False
litellm.suppress_debug_info = True
# It also attaches its own stderr handler at import, after the app has
# already configured logging — so this has to run here, once litellm exists,
# or every line it emits is printed twice in two different formats.
tame_litellm_logging()

#: How instructor is asked to read a reply.
#:
#: MD_JSON rather than JSON, because plenty of models answer a "return JSON"
#: instruction with JSON wrapped in a markdown fence:
#:
#:     ```json
#:     {"relevance": "high", ...}
#:     ```
#:
#: ``Mode.JSON`` calls ``json.loads`` on that verbatim and fails at line 1
#: column 1 with the content perfectly intact. Because a parse failure is
#: counted as a *validation* failure, it then burns every validation retry
#: and every endpoint in the tier's chain before deferring the stage — a
#: whole tier reported as unavailable when both providers answered
#: correctly. Anthropic's models fence consistently, so this surfaced the
#: moment one entered a chain.
#:
#: MD_JSON strips the fence when there is one and is otherwise identical; it
#: parses bare JSON exactly as before. A superset, not a trade-off.
INSTRUCTOR_MODE: Final = instructor.Mode.MD_JSON

#: How long an endpoint that failed at the transport level is stepped over.
#:
#: The chain is re-walked from the primary on every call, so a model that is
#: simply down costs a full timeout per video — which is what makes a
#: deferred stage take so long to drain once the backend returns. Skipping it
#: for a minute costs at most one stale minute of avoidance and saves a
#: timeout per call.
ENDPOINT_COOLDOWN_S: Final = 60.0

#: Exceptions meaning "this endpoint is not working" — move to the next one.
_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    litellm_exceptions.Timeout,
    litellm_exceptions.APIConnectionError,
    litellm_exceptions.ServiceUnavailableError,
    litellm_exceptions.InternalServerError,
    litellm_exceptions.RateLimitError,
    litellm_exceptions.AuthenticationError,
    litellm_exceptions.BadRequestError,
    litellm_exceptions.NotFoundError,
    litellm_exceptions.ContextWindowExceededError,
)


def _root_cause(exc: BaseException) -> BaseException:
    """Unwrap instructor's retry wrapper to the error that actually occurred.

    ``InstructorRetryException`` is raised for transport failures as well as
    schema-validation failures, so the wrapper alone cannot tell them apart.
    """
    seen: set[int] = set()
    current: BaseException = exc
    while isinstance(current, InstructorRetryException):
        nxt = current.__cause__ or getattr(current, "last_exception", None)
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        current = nxt
    return current


class LLMClient:
    """Provider-agnostic structured completion with per-tier fallback chains."""

    def __init__(self, settings: Settings, db: sqlite3.Connection | None = None) -> None:
        self._settings = settings
        self._db = db
        self._log_io = settings.llm.log_llm_io
        self._chains: dict[str, list[tuple[str, LLMEndpoint]]] = {}
        #: alias -> when it last failed at the transport level. Process-local
        #: and deliberately not persisted: it is a hint about right now, and
        #: a restart is exactly when it should be forgotten.
        self._cooling: dict[str, float] = {}

        model_list: list[dict[str, Any]] = []
        fallback_map: list[dict[str, list[str]]] = []

        for tier, tier_cfg in settings.llm.tiers.items():
            chain = tier_cfg.active_chain()
            aliases: list[tuple[str, LLMEndpoint]] = []
            for index, endpoint in enumerate(chain):
                alias = tier if index == 0 else f"{tier}__fb{index - 1}"
                aliases.append((alias, endpoint))
                model_list.append(
                    {
                        "model_name": alias,
                        "litellm_params": self._litellm_params(endpoint, tier_cfg.timeout_s),
                    }
                )
            self._chains[tier] = aliases
            if len(aliases) > 1:
                # Registered for completeness; chain walking here is what
                # actually drives failover (see module docstring).
                fallback_map.append({tier: [a for a, _ in aliases[1:]]})

            dropped = len([*[tier_cfg.primary], *tier_cfg.fallbacks]) - len(chain)
            if dropped:
                log.info(
                    "llm.cloud_fallback_disabled",
                    tier=tier,
                    dropped_endpoints=dropped,
                    remaining=[e.litellm_model() for e in chain],
                )

        self._router = Router(
            model_list=model_list,
            fallbacks=cast(list[Any], fallback_map),
            # Retries and failover are decided here, not inside the Router.
            num_retries=0,
            set_verbose=False,
        )
        # async_client=True is explicit rather than inferred: Router.acompletion
        # is a coroutine function, and a sync client here would fail only at
        # runtime.
        self._instructor: AsyncInstructor = instructor.from_litellm(
            self._router.acompletion,
            mode=INSTRUCTOR_MODE,
            async_client=True,
        )

    def _litellm_params(self, endpoint: LLMEndpoint, timeout_s: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": endpoint.litellm_model(),
            "temperature": endpoint.temperature,
            "timeout": timeout_s,
        }
        if base := endpoint.resolved_api_base():
            params["api_base"] = base
        if endpoint.max_tokens is not None:
            params["max_tokens"] = endpoint.max_tokens
        if key := self._settings.api_key_for(endpoint.provider):
            params["api_key"] = key
        params.update(endpoint.extra_params)
        return params

    def _walk(self, chain: list[tuple[str, LLMEndpoint]]) -> list[tuple[int, str, LLMEndpoint]]:
        """The chain to try now, skipping endpoints in cooldown.

        Indices are positions in the *full* chain, so telemetry still
        reports whether a fallback was used. When every endpoint is cooling
        the whole chain is walked anyway: a cooldown exists to avoid a
        wasted timeout, and must never turn "slow" into "unavailable".
        """
        now = time.monotonic()
        walk = [
            (index, alias, endpoint)
            for index, (alias, endpoint) in enumerate(chain)
            if now - self._cooling.get(alias, -ENDPOINT_COOLDOWN_S) >= ENDPOINT_COOLDOWN_S
        ]
        if walk:
            return walk
        log.debug("llm.all_endpoints_cooling", endpoints=len(chain))
        return [(index, alias, endpoint) for index, (alias, endpoint) in enumerate(chain)]

    # --- public API -----------------------------------------------------

    async def complete_structured(
        self,
        tier: str,
        system: str,
        user: str,
        response_model: type[T],
        *,
        video_id: str | None = None,
        prompt_version: str = "",
    ) -> tuple[T, CallMeta]:
        chain = self._chains.get(tier)
        if not chain:
            raise LLMUnavailable(f"no endpoints configured for tier {tier!r}")

        tier_cfg = self._settings.llm.tiers[tier]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self._log_io:
            log.debug("llm.request", tier=tier, video_id=video_id, prompt=f"{system}\n\n{user}")

        attempted: list[str] = []
        total_validation_retries = 0
        last_error: Exception | None = None

        walk = self._walk(chain)
        if len(walk) < len(chain):
            log.debug(
                "llm.endpoints_skipped",
                tier=tier,
                skipped=[a for a, _ in chain if a not in {w[1] for w in walk}],
            )

        for position, (index, alias, endpoint) in enumerate(walk):
            attempted.append(endpoint.litellm_model())
            validation_failures = 0

            while True:
                started = time.perf_counter()
                try:
                    obj, raw = await self._instructor.chat.completions.create_with_completion(
                        model=alias,
                        # litellm accepts plain role/content dicts; the
                        # OpenAI TypedDict union adds nothing here.
                        messages=cast(Any, messages),
                        response_model=response_model,
                        # instructor's own retry is disabled: retries are
                        # counted here so telemetry reflects reality and so
                        # the chain can move on after the configured number
                        # of failures.
                        max_retries=1,
                    )
                except Exception as exc:
                    # instructor wraps EVERY failure in
                    # InstructorRetryException, including transport ones, so
                    # classify on the underlying cause. Treating a timeout as
                    # a validation failure would retry a dead endpoint
                    # instead of failing over, and would corrupt the
                    # validation_retries telemetry.
                    cause = _root_cause(exc)
                    is_validation = isinstance(cause, ValidationError)

                    if is_validation:
                        validation_failures += 1
                        total_validation_retries += 1
                        last_error = exc
                        log.warning(
                            "llm.validation_failed",
                            tier=tier,
                            video_id=video_id,
                            model=endpoint.litellm_model(),
                            attempt=validation_failures,
                            error=str(cause)[:300],
                        )
                        if validation_failures > tier_cfg.validation_retries:
                            break  # next endpoint (2x validation failure fails over)
                        continue

                    last_error = exc
                    log.warning(
                        "llm.endpoint_failed",
                        tier=tier,
                        video_id=video_id,
                        model=endpoint.litellm_model(),
                        error_type=type(cause).__name__,
                        error=str(cause)[:300],
                        transport_error=isinstance(cause, _TRANSPORT_ERRORS),
                        next_endpoint=(
                            walk[position + 1][2].litellm_model()
                            if position + 1 < len(walk)
                            else None
                        ),
                    )
                    if isinstance(cause, _TRANSPORT_ERRORS):
                        # The endpoint itself is unwell, so stop paying its
                        # timeout on every following call for a while. A
                        # validation failure is not this: a model emitting
                        # bad JSON is answering, and answering is what
                        # matters here.
                        if alias not in self._cooling:
                            log.info(
                                "llm.endpoint_cooling",
                                tier=tier,
                                model=endpoint.litellm_model(),
                                seconds=int(ENDPOINT_COOLDOWN_S),
                            )
                        self._cooling[alias] = time.monotonic()
                    break  # next endpoint

                latency_ms = int((time.perf_counter() - started) * 1000)
                meta = self._build_meta(
                    tier=tier,
                    endpoint=endpoint,
                    raw=raw,
                    latency_ms=latency_ms,
                    fallback_used=index > 0,
                    validation_retries=total_validation_retries,
                    prompt_version=prompt_version,
                    video_id=video_id,
                    attempted=attempted,
                )
                if self._log_io:
                    log.debug(
                        "llm.response", tier=tier, video_id=video_id, response=obj.model_dump_json()
                    )
                # It answered, so whatever was wrong with it is over.
                self._cooling.pop(alias, None)
                self._record(meta)
                log.info(
                    "llm.call_ok",
                    tier=tier,
                    video_id=video_id,
                    model=meta.model,
                    provider=meta.provider,
                    latency_ms=meta.latency_ms,
                    input_tokens=meta.input_tokens,
                    output_tokens=meta.output_tokens,
                    cost_usd=round(meta.cost_usd, 6),
                    fallback_used=meta.fallback_used,
                    validation_retries=meta.validation_retries,
                )
                return obj, meta

        raise LLMUnavailable(
            f"tier {tier!r}: all {len(chain)} endpoint(s) failed "
            f"({', '.join(attempted)}); last error: {last_error}"
        ) from last_error

    # --- telemetry --------------------------------------------------------

    def _build_meta(
        self,
        *,
        tier: str,
        endpoint: LLMEndpoint,
        raw: Any,
        latency_ms: int,
        fallback_used: bool,
        validation_retries: int,
        prompt_version: str,
        video_id: str | None,
        attempted: list[str],
    ) -> CallMeta:
        usage = getattr(raw, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        cost = 0.0
        if endpoint.provider is not Provider.OLLAMA:
            # Local models are free but still get tokens/latency recorded,
            # which is the whole point of the telemetry (local-vs-cloud
            # economics, design §8's tier distribution metric).
            try:
                cost = float(litellm.completion_cost(completion_response=raw) or 0.0)
            except Exception as exc:  # pricing gaps must never fail a good call
                log.debug("llm.cost_unavailable", model=endpoint.litellm_model(), error=str(exc))

        return CallMeta(
            tier=tier,
            provider=endpoint.provider.value,
            model=endpoint.model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            fallback_used=fallback_used,
            validation_retries=validation_retries,
            prompt_version=prompt_version,
            video_id=video_id,
            attempted_models=list(attempted),
        )

    def _record(self, meta: CallMeta) -> None:
        """Telemetry for design §8's `/metrics` (tier distribution, ASR
        minutes, token spend). SQLite, not a CouchDB `llm_call` document
        (plan §1.3) — one row in `metric_events`, kind='llm_call'."""
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT INTO metric_events (kind, video_id, value, meta, created_at) "
                "VALUES ('llm_call', ?, ?, ?, ?)",
                (
                    meta.video_id,
                    meta.cost_usd,
                    json.dumps(
                        {
                            "tier": meta.tier,
                            "provider": meta.provider,
                            "model": meta.model,
                            "latency_ms": meta.latency_ms,
                            "input_tokens": meta.input_tokens,
                            "output_tokens": meta.output_tokens,
                            "fallback_used": meta.fallback_used,
                            "validation_retries": meta.validation_retries,
                            "prompt_version": meta.prompt_version,
                            "attempted_models": meta.attempted_models,
                        }
                    ),
                    iso_now(),
                ),
            )
            self._db.commit()
        except sqlite3.Error as exc:  # telemetry is not worth failing a pipeline run
            log.warning("llm.telemetry_write_failed", error=str(exc))

    async def close(self) -> None:
        # Router owns http clients; closing is best-effort across litellm versions.
        closer = getattr(self._router, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception as exc:
                log.debug("llm.router_close_failed", error=str(exc))
