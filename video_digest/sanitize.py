"""Entity identity, shared with the other vault writers.

`canonical()` and `slugify()` are **ported verbatim** from
`podcast_agent/entities.py` (`canonical`, at podcast-digest@fac8a1f) and
`podcast_agent/sanitize.py` (`slugify`), by way of the same verbatim port in
`clippings_topics/topics.py` (clippings-topics@fd64996). Three applications
now write `99 topics/` and agree only by applying the same two functions to a
name — they do not talk to each other. Changing either here alone produces
``crowd-strike.md`` beside ``crowdstrike.md`` and a graph that splits one
subject in two. Do not "improve" this copy independently of the others.
"""

from __future__ import annotations

import re
import unicodedata

_CVE = re.compile(r"^cve[\s\-_]*(\d{4})[\s\-_]*(\d{4,7})$", re.IGNORECASE)
_LEADING = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_TRAILING = re.compile(r"[\s,]+(inc|inc\.|llc|ltd|ltd\.|corp|corp\.|gmbh|plc)$", re.IGNORECASE)
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def canonical(name: str) -> str:
    """The key two spellings of the same thing must share.

    Deliberately conservative. Over-merging is the worse error: it silently
    fuses two unrelated entities into one timeline that reads as evidence, and
    nothing downstream can tell. Under-merging leaves two rows a reader can see
    and interpret for themselves.
    """
    text = " ".join(str(name).split()).strip(" .,;:—-")
    if not text:
        return ""
    if match := _CVE.match(text):
        return f"cve-{match.group(1)}-{int(match.group(2)):04d}"
    text = _LEADING.sub("", text)
    text = _TRAILING.sub("", text)
    return text.casefold().strip(" .,;:—-")


def slugify(text: str, *, max_len: int = 60, fallback: str = "untitled") -> str:
    """ASCII-only, lowercase, hyphenated slug — used for note filenames."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_only).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or fallback
