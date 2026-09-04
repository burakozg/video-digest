"""canonical()/slugify() parity against podcast-digest's originals.

Three applications (podcast-digest, clippings-topics, video-digest) now write
`99 topics/` and agree on identity only by applying the same two functions to
a name — they do not talk to each other (see video_digest/sanitize.py's
header). This file pins the exact outputs of `podcast_agent.entities.canonical`
and `podcast_agent.sanitize.slugify` at the commit this was ported from
(fac8a1f), captured by actually running them, not by re-deriving the rule —
so a future edit to *this* copy that quietly diverges fails here, without this
repo needing to import podcast-digest as a dependency.
"""

from __future__ import annotations

import pytest

from video_digest.sanitize import canonical, slugify

#: (input, expected canonical(), expected slugify())
CASES: list[tuple[str, str, str]] = [
    ("CrowdStrike", "crowdstrike", "crowdstrike"),
    ("Crowd Strike", "crowd strike", "crowd-strike"),
    ("  CrowdStrike Inc.  ", "crowdstrike", "crowdstrike-inc"),
    ("The Anthropic Company", "anthropic company", "the-anthropic-company"),
    ("OLLAMA", "ollama", "ollama"),
    ("n8n", "n8n", "n8n"),
    ("CVE-2024-3094", "cve-2024-3094", "cve-2024-3094"),
    ("cve 2024 3094", "cve-2024-3094", "cve-2024-3094"),
    ("Volt Typhoon", "volt typhoon", "volt-typhoon"),
    ("A Company, LLC", "company", "a-company-llc"),
    ("Ünïcödé Näme", "ünïcödé näme", "unicode-name"),
    ("", "", "untitled"),
    ("   ", "", "untitled"),
]


@pytest.mark.parametrize("name,expected_canonical,expected_slug", CASES)
def test_matches_podcast_digest(name: str, expected_canonical: str, expected_slug: str) -> None:
    assert canonical(name) == expected_canonical
    assert slugify(name) == expected_slug


def test_two_spellings_of_one_entity_share_a_key() -> None:
    assert canonical("CrowdStrike") == canonical("  CrowdStrike Inc.  ")


def test_cve_normalises_padding() -> None:
    assert canonical("CVE-2024-3094") == canonical("cve 2024 3094")
