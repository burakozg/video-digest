"""LLM abstraction. Nothing outside this package imports litellm or instructor."""

from .base import LLMUnavailable, StructuredLLM
from .models import CallMeta, ChunkSummary, Highlight, VideoDigest

__all__ = [
    "CallMeta",
    "ChunkSummary",
    "Highlight",
    "LLMUnavailable",
    "StructuredLLM",
    "VideoDigest",
    "build_llm_client",
]


def build_llm_client(settings: object, db: object = None) -> StructuredLLM:
    """Construct the real client. Imported lazily so tests never load litellm."""
    from .client import LLMClient

    return LLMClient(settings, db)  # type: ignore[arg-type]
