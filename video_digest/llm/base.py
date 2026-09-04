"""The LLM boundary the rest of the application sees.

Ported from `podcast_agent/llm/base.py`. Only `video_digest/llm/` may import
litellm or instructor — pipeline stages depend on `StructuredLLM`, which
makes them trivially testable against a fake and keeps provider churn
contained to one module.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from .models import CallMeta

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    """Every endpoint in a tier's chain failed.

    The caller should leave the video queued for a later run rather than
    mark it failed — this is usually a transient local-model outage, and
    with `allow_cloud_fallback: false` it is the expected way work waits.
    """


class StructuredLLM(Protocol):
    """Sole LLM entry point."""

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
        """Return a validated `response_model` instance plus call telemetry.

        Raises `LLMUnavailable` when the tier's whole chain is exhausted.
        """
        ...
