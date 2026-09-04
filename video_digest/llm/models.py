"""The Pass B contract (design §5 S4) — defined ahead of the LLM client
that produces it (M4), because the note renderer (M3) needs a typed input
regardless of what fills it: a test fixture now, `instructor` later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Highlight(BaseModel):
    t_seconds: int = Field(ge=0)
    label: str = Field(max_length=90)


class ChunkSummary(BaseModel):
    """Pass A (map) output — one per chunk (design §5 S4)."""

    covered: str
    claims: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    quotable_t_seconds: int | None = None


class VideoDigest(BaseModel):
    """Pass B (reduce) output — design §5 S4, verbatim field-for-field."""

    tldr: str
    summary_md: str
    key_points: list[str] = Field(min_length=1, max_length=10)
    highlights: list[Highlight] = Field(default_factory=list, max_length=8)
    entities: list[str] = Field(default_factory=list)
    #: Short canonical subject tags, not descriptive phrases — a length cap
    #: is what makes this structural rather than trusting the prompt alone.
    #: Found live on the pilot run: unconstrained, the model produced
    #: "Customer Identity and Access Management (CIAM)" and similar, which
    #: `sanitize.canonical()` treats as a different topic every time it's
    #: phrased differently — defeating the whole point of accumulating
    #: mentions toward `topic_creation_threshold` (design §5 S5). A
    #: validation failure here is also cheap: instructor retries with the
    #: Pydantic error fed back to the model, which usually self-corrects.
    topics: list[Annotated[str, Field(max_length=40)]] = Field(
        default_factory=list, max_length=6
    )
    claims_to_verify: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    relevance: Literal["critical", "high", "medium", "low"]


@dataclass(slots=True)
class CallMeta:
    """Telemetry for one structured LLM invocation. Ported from
    `podcast_agent/models.py::CallMeta`."""

    tier: str
    provider: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    fallback_used: bool = False
    validation_retries: int = 0
    prompt_version: str = ""
    video_id: str | None = None
    #: Endpoints attempted, in order — useful when diagnosing fallback churn.
    attempted_models: list[str] = field(default_factory=list)
