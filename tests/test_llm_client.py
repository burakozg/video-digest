"""Tests for the real LLMClient fallback/retry machinery.

Ported from podcast-digest's tests/test_llm_client.py against
`podcast_agent/llm/client.py` (the module this file's implementation is
itself ported from — plan §1.5/§1.6). Two real bugs escaped there without
tests at this exact boundary: `instructor.exceptions` is not a package
attribute (an except clause raised AttributeError while handling the real
error and masked it), and instructor wraps transport failures in
`InstructorRetryException` too, so timeouts were once miscounted as
validation failures and retried against a dead endpoint instead of failing
over. Nothing here touches the network: the instructor client is replaced
with a stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import litellm
import pytest
from instructor.core import InstructorRetryException
from pydantic import BaseModel, Field, ValidationError

from video_digest.config import Settings
from video_digest.db import connect
from video_digest.llm.base import LLMUnavailable
from video_digest.llm.client import ENDPOINT_COOLDOWN_S, LLMClient, _root_cause


class _Result(BaseModel):
    score: int = Field(ge=0, le=10)


class _Usage:
    prompt_tokens = 120
    completion_tokens = 34


class _RawResponse:
    model = "test-model"
    usage = _Usage()


class StubCompletions:
    """Stands in for instructor's chat.completions, scripted per call."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create_with_completion(self, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else self.outcomes
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, _RawResponse()


def install_stub(client: LLMClient, outcomes: list[Any]) -> StubCompletions:
    stub = StubCompletions(outcomes)
    client._instructor = type(  # type: ignore[assignment]
        "StubInstructor", (), {"chat": type("Chat", (), {"completions": stub})()}
    )()
    return stub


def two_endpoint_settings(**tier0_overrides: Any) -> Settings:
    """"map" with a local primary and a cloud fallback; "reduce" local-only."""
    llm = {
        "tiers": {
            "map": {
                "primary": {"provider": "ollama", "model": "local-small"},
                "fallbacks": [{"provider": "openrouter", "model": "vendor/remote"}],
                "validation_retries": 2,
                **tier0_overrides,
            },
            "reduce": {"primary": {"provider": "ollama", "model": "local-big"}},
        }
    }
    return Settings(
        llm=llm,
        transcription={"remote_url": "http://mac.local:8000"},
        openrouter_api_key="sk-test",
    )


def valid_result() -> _Result:
    return _Result(score=8)


def a_timeout() -> Exception:
    return litellm.Timeout(message="timed out", model="m", llm_provider="ollama")


def a_validation_error() -> ValidationError:
    try:
        _Result(score=99)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def wrapped(cause: BaseException) -> InstructorRetryException:
    """How instructor actually surfaces a failure: wrapped, with __cause__ set."""
    exc = InstructorRetryException(str(cause), n_attempts=1, total_usage=0)
    exc.__cause__ = cause
    return exc


def _llm_call_rows(db) -> list[dict[str, Any]]:
    rows = db.execute(
        "SELECT video_id, value, meta FROM metric_events WHERE kind = 'llm_call'"
    ).fetchall()
    return [{"video_id": r["video_id"], "cost_usd": r["value"], **json.loads(r["meta"])} for r in rows]


class TestRootCause:
    def test_unwraps_instructor_wrapper(self) -> None:
        cause = a_timeout()
        assert _root_cause(wrapped(cause)) is cause

    def test_returns_plain_exception_unchanged(self) -> None:
        exc = ValueError("plain")
        assert _root_cause(exc) is exc

    def test_survives_a_self_referential_cause(self) -> None:
        exc = InstructorRetryException("looping", n_attempts=1, total_usage=0)
        exc.__cause__ = exc
        assert _root_cause(exc) is exc  # terminates instead of looping forever


class TestSuccessPath:
    @pytest.mark.asyncio
    async def test_returns_result_and_telemetry(self) -> None:
        client = LLMClient(two_endpoint_settings())
        install_stub(client, [valid_result()])

        result, meta = await client.complete_structured(
            "map", "sys", "user", _Result, video_id="v1", prompt_version="map_v1"
        )

        assert result.score == 8
        assert meta.provider == "ollama"
        assert meta.fallback_used is False
        assert meta.validation_retries == 0
        assert meta.input_tokens == 120
        assert meta.output_tokens == 34
        assert meta.cost_usd == 0.0  # local model is free
        assert meta.prompt_version == "map_v1"

    @pytest.mark.asyncio
    async def test_writes_a_telemetry_row(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        client = LLMClient(two_endpoint_settings(), db)
        install_stub(client, [valid_result()])
        await client.complete_structured("map", "s", "u", _Result, video_id="v1")

        rows = _llm_call_rows(db)
        assert len(rows) == 1
        assert rows[0]["tier"] == "map"
        assert rows[0]["video_id"] == "v1"
        assert rows[0]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_targets_the_primary_deployment_first(self) -> None:
        client = LLMClient(two_endpoint_settings())
        stub = install_stub(client, [valid_result()])
        await client.complete_structured("map", "s", "u", _Result)
        assert stub.calls[0]["model"] == "map"

    @pytest.mark.asyncio
    async def test_sends_system_and_user_messages(self) -> None:
        client = LLMClient(two_endpoint_settings())
        stub = install_stub(client, [valid_result()])
        await client.complete_structured("map", "SYSTEM TEXT", "USER TEXT", _Result)
        messages = stub.calls[0]["messages"]
        assert messages[0] == {"role": "system", "content": "SYSTEM TEXT"}
        assert messages[1] == {"role": "user", "content": "USER TEXT"}


class TestTransportFailover:
    @pytest.mark.asyncio
    async def test_timeout_fails_over_immediately(self) -> None:
        """A dead endpoint must not consume the validation retry budget."""
        client = LLMClient(two_endpoint_settings())
        stub = install_stub(client, [a_timeout(), valid_result()])

        _, meta = await client.complete_structured("map", "s", "u", _Result)

        assert len(stub.calls) == 2  # one attempt each, no retry on the dead one
        assert stub.calls[1]["model"] == "map__fb0"
        assert meta.fallback_used is True
        assert meta.provider == "openrouter"
        assert meta.validation_retries == 0

    @pytest.mark.asyncio
    async def test_instructor_wrapped_timeout_is_still_transport(self) -> None:
        """The regression: wrapped timeouts were miscounted as validation failures."""
        client = LLMClient(two_endpoint_settings())
        stub = install_stub(client, [wrapped(a_timeout()), valid_result()])

        _, meta = await client.complete_structured("map", "s", "u", _Result)

        assert len(stub.calls) == 2  # failed over rather than retrying
        assert meta.fallback_used is True
        assert meta.validation_retries == 0

    @pytest.mark.asyncio
    async def test_connection_error_fails_over(self) -> None:
        client = LLMClient(two_endpoint_settings())
        install_stub(
            client,
            [
                litellm.APIConnectionError(message="refused", model="m", llm_provider="ollama"),
                valid_result(),
            ],
        )
        _, meta = await client.complete_structured("map", "s", "u", _Result)
        assert meta.fallback_used is True

    @pytest.mark.asyncio
    async def test_unknown_error_also_fails_over(self) -> None:
        """An unrecognised provider error must not strand the tier."""
        client = LLMClient(two_endpoint_settings())
        install_stub(client, [RuntimeError("something odd"), valid_result()])
        _, meta = await client.complete_structured("map", "s", "u", _Result)
        assert meta.fallback_used is True

    @pytest.mark.asyncio
    async def test_exception_handling_never_raises_a_new_error(self) -> None:
        """The AttributeError regression: the except clause itself blew up, so
        the real cause was replaced by a nonsense error at the call site."""
        client = LLMClient(two_endpoint_settings())
        install_stub(client, [a_timeout(), a_timeout()])

        with pytest.raises(LLMUnavailable) as excinfo:
            await client.complete_structured("map", "s", "u", _Result)
        assert "AttributeError" not in str(excinfo.value)
        assert "timed out" in str(excinfo.value).lower()


class TestValidationRetries:
    @pytest.mark.asyncio
    async def test_retries_on_the_same_endpoint_then_fails_over(self) -> None:
        """2x validation failure moves to the next endpoint."""
        client = LLMClient(two_endpoint_settings())
        stub = install_stub(
            client,
            [
                wrapped(a_validation_error()),
                wrapped(a_validation_error()),
                wrapped(a_validation_error()),
                valid_result(),
            ],
        )
        _, meta = await client.complete_structured("map", "s", "u", _Result)

        assert [c["model"] for c in stub.calls] == ["map", "map", "map", "map__fb0"]
        assert meta.fallback_used is True
        assert meta.validation_retries == 3

    @pytest.mark.asyncio
    async def test_recovers_without_failing_over(self) -> None:
        client = LLMClient(two_endpoint_settings())
        stub = install_stub(client, [wrapped(a_validation_error()), valid_result()])
        await client.complete_structured("map", "s", "u", _Result)
        assert [c["model"] for c in stub.calls] == ["map", "map"]


class TestChainExhaustion:
    @pytest.mark.asyncio
    async def test_raises_llm_unavailable_listing_attempts(self) -> None:
        client = LLMClient(two_endpoint_settings())
        install_stub(client, [a_timeout(), a_timeout()])
        with pytest.raises(LLMUnavailable) as excinfo:
            await client.complete_structured("map", "s", "u", _Result)
        message = str(excinfo.value)
        assert "ollama_chat/local-small" in message
        assert "openrouter/vendor/remote" in message

    @pytest.mark.asyncio
    async def test_unknown_tier_is_rejected(self) -> None:
        client = LLMClient(two_endpoint_settings())
        with pytest.raises(LLMUnavailable, match="no endpoints configured"):
            await client.complete_structured("nope", "s", "u", _Result)

    @pytest.mark.asyncio
    async def test_no_telemetry_written_when_every_endpoint_fails(self, tmp_path: Path) -> None:
        db = connect(tmp_path / "state.sqlite")
        client = LLMClient(two_endpoint_settings(), db)
        install_stub(client, [a_timeout(), a_timeout()])
        with pytest.raises(LLMUnavailable):
            await client.complete_structured("map", "s", "u", _Result)
        assert _llm_call_rows(db) == []


class TestCloudFallbackSwitch:
    @pytest.mark.asyncio
    async def test_local_only_tier_has_a_single_endpoint(self) -> None:
        """With the switch off, the cloud endpoint is not merely skipped — it
        is absent from the chain, so nothing can route to it."""
        settings = two_endpoint_settings(allow_cloud_fallback=False)
        client = LLMClient(settings)
        assert [alias for alias, _ in client._chains["map"]] == ["map"]

        install_stub(client, [a_timeout()])
        with pytest.raises(LLMUnavailable) as excinfo:
            await client.complete_structured("map", "s", "u", _Result)
        assert "openrouter" not in str(excinfo.value)


class TestEndpointParams:
    def test_local_endpoint_gets_base_url_and_no_key(self) -> None:
        settings = two_endpoint_settings()
        client = LLMClient(settings)
        params = client._litellm_params(settings.llm.tiers["map"].primary, 60)
        assert params["model"] == "ollama_chat/local-small"
        assert params["api_base"] == "http://localhost:11434"
        assert "api_key" not in params
        assert params["timeout"] == 60

    def test_cloud_endpoint_gets_its_api_key(self) -> None:
        settings = two_endpoint_settings()
        client = LLMClient(settings)
        params = client._litellm_params(settings.llm.tiers["map"].fallbacks[0], 60)
        assert params["api_key"] == "sk-test"

    def test_extra_params_pass_through(self) -> None:
        """Needed for provider-specific flags such as Ollama's `think: false`."""
        settings = Settings(
            llm={
                "tiers": {
                    "map": {
                        "primary": {
                            "provider": "ollama",
                            "model": "m",
                            "extra_params": {"think": False, "num_ctx": 8192},
                        }
                    },
                    "reduce": {"primary": {"provider": "ollama", "model": "big"}},
                }
            },
            transcription={"remote_url": "http://mac.local:8000"},
        )
        client = LLMClient(settings)
        params = client._litellm_params(settings.llm.tiers["map"].primary, 60)
        assert params["think"] is False
        assert params["num_ctx"] == 8192


class TestEndpointCooldown:
    """A down endpoint costs a full timeout on every call that walks past it.
    Remembering the failure for a minute removes the repeat."""

    @pytest.mark.asyncio
    async def test_a_second_call_skips_the_endpoint_that_just_timed_out(self) -> None:
        client = LLMClient(two_endpoint_settings())
        stub = install_stub(client, [a_timeout(), valid_result(), valid_result()])

        await client.complete_structured("map", "s", "u", _Result)
        await client.complete_structured("map", "s", "u", _Result)

        # Three calls, not four: the second request went straight to the fallback.
        assert [c["model"] for c in stub.calls] == ["map", "map__fb0", "map__fb0"]

    @pytest.mark.asyncio
    async def test_a_validation_failure_does_not_cool_an_endpoint(self) -> None:
        """A model emitting bad JSON is answering, and answering is the point."""
        client = LLMClient(two_endpoint_settings())
        stub = install_stub(
            client,
            [
                a_validation_error(),
                a_validation_error(),
                a_validation_error(),
                valid_result(),
                valid_result(),
            ],
        )
        await client.complete_structured("map", "s", "u", _Result)
        await client.complete_structured("map", "s", "u", _Result)
        assert stub.calls[-1]["model"] == "map"

    @pytest.mark.asyncio
    async def test_success_clears_the_cooldown(self) -> None:
        client = LLMClient(two_endpoint_settings())
        install_stub(client, [a_timeout(), valid_result()])
        await client.complete_structured("map", "s", "u", _Result)
        assert "map" in client._cooling

        client._cooling.clear()
        stub = install_stub(client, [valid_result(), valid_result()])
        await client.complete_structured("map", "s", "u", _Result)
        assert "map" not in client._cooling
        await client.complete_structured("map", "s", "u", _Result)
        assert [c["model"] for c in stub.calls] == ["map", "map"]

    @pytest.mark.asyncio
    async def test_the_cooldown_expires(self) -> None:
        client = LLMClient(two_endpoint_settings())
        stub = install_stub(client, [a_timeout(), valid_result(), valid_result()])
        await client.complete_structured("map", "s", "u", _Result)

        client._cooling["map"] -= ENDPOINT_COOLDOWN_S + 1
        await client.complete_structured("map", "s", "u", _Result)
        assert stub.calls[-1]["model"] == "map"

    @pytest.mark.asyncio
    async def test_every_endpoint_cooling_still_walks_the_whole_chain(self) -> None:
        """A cooldown avoids a wasted timeout. It must never be the reason a
        tier reports itself unavailable while an endpoint might answer."""
        client = LLMClient(two_endpoint_settings())
        install_stub(client, [a_timeout(), a_timeout()])
        with pytest.raises(LLMUnavailable):
            await client.complete_structured("map", "s", "u", _Result)
        assert set(client._cooling) == {"map", "map__fb0"}

        stub = install_stub(client, [valid_result()])
        _, meta = await client.complete_structured("map", "s", "u", _Result)
        assert stub.calls[0]["model"] == "map"
        assert meta.fallback_used is False

    @pytest.mark.asyncio
    async def test_one_tier_cooling_does_not_affect_another(self) -> None:
        client = LLMClient(two_endpoint_settings())
        install_stub(client, [a_timeout(), valid_result()])
        await client.complete_structured("map", "s", "u", _Result)

        stub = install_stub(client, [valid_result()])
        await client.complete_structured("reduce", "s", "u", _Result)
        assert stub.calls[0]["model"] == "reduce"
