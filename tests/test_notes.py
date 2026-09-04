"""The tests that matter: another writer's content survives our run.

Ported from clippings-topics' tests/test_notes.py — same merge logic, this
app's OWNER/KEY_PREFIX. A merge bug here does not raise — it silently eats a
section of somebody's vault, and the damage looks like Obsidian losing notes
rather than a bug in this program.

Fixture shaped like a real note (``99 topics/anthropic.md``): human prose at
the top, then podcast-digest's owned region, with frontmatter carrying both
unprefixed keys and another writer's prefixed ones.
"""

from __future__ import annotations

from video_digest.vault import notes

EXISTING = """\
---
tags: [podcast-entity, ai]
type: topic
title: "Anthropic"
podcasts_mentions: 97
podcasts_shows: 3
---

# Anthropic

My own view: they ship faster than they explain.

<!-- begin:podcast-digest -->
## From podcasts

- 2026-08-20 · **Caveat** — When companies can hack back
<!-- end:podcast-digest -->
"""

OURS = """\
---
type: topic
title: "Anthropic"
tags: [topic]
video_mentions: 2
---

# Anthropic

<!-- begin:video-digest -->
## From videos

- 2026-08-22 · [[13 video-summaries/2026-08-22-x|A Video]]
<!-- end:video-digest -->
"""


def merged() -> str:
    return notes.merge_owned_section(EXISTING, OURS)


def test_other_writers_region_is_byte_identical() -> None:
    theirs = EXISTING[
        EXISTING.index("<!-- begin:podcast-digest -->") : EXISTING.index(
            "<!-- end:podcast-digest -->"
        )
        + len("<!-- end:podcast-digest -->")
    ]
    assert theirs in merged()


def test_human_prose_survives() -> None:
    assert "My own view: they ship faster than they explain." in merged()


def test_their_frontmatter_keys_survive() -> None:
    """The bug the skill records: a second ``tags:`` silently drops the
    reader's. PyYAML would return only the last one, with no error — so
    ``tags`` must keep the reader's value and appear exactly once."""
    out = merged()
    assert "tags: [podcast-entity, ai]" in out
    assert "tags: [topic]" not in out
    assert out.count("tags:") == 1
    assert "podcasts_mentions: 97" in out
    assert "podcasts_shows: 3" in out


def test_our_keys_are_written() -> None:
    assert "video_mentions: 2" in merged()


def test_our_region_is_added_once() -> None:
    out = merged()
    assert out.count(notes.begin_marker()) == 1
    assert out.count(notes.end_marker()) == 1
    assert "[[13 video-summaries/2026-08-22-x|A Video]]" in out


def test_rerun_is_idempotent() -> None:
    """A vault replicating over iCloud *and* LiveSync re-syncs every note we
    touch, so 'unchanged input produces byte-identical output' is a
    correctness property, not a nicety."""
    once = merged()
    assert notes.merge_owned_section(once, OURS) == once


def test_our_region_is_replaced_not_appended() -> None:
    """A re-run can remove a video, so the region is rebuilt whole."""
    once = merged()
    fewer = OURS.replace(
        "- 2026-08-22 · [[13 video-summaries/2026-08-22-x|A Video]]\n", ""
    ).replace("video_mentions: 2", "video_mentions: 0")
    twice = notes.merge_owned_section(once, fewer)
    assert "[[13 video-summaries/2026-08-22-x|A Video]]" not in twice
    assert twice.count(notes.begin_marker()) == 1
    assert "When companies can hack back" in twice


def test_creates_the_note_when_absent() -> None:
    assert notes.merge_owned_section(None, OURS) == OURS
    assert notes.merge_owned_section("", OURS) == OURS


def test_appends_when_note_exists_without_our_marker() -> None:
    """A topic note podcast-digest made before we existed: adopt, don't
    clobber."""
    out = notes.merge_owned_section(EXISTING, OURS)
    assert out.index("<!-- begin:podcast-digest -->") < out.index(notes.begin_marker())
    assert "# Anthropic" in out


DUPLICATED = """\
---
tags: [podcast-entity, ai]
type: topic
title: "Anthropic"
security_mentions: 19
security_digests: 2
podcasts_mentions: 97
security_mentions: 19
security_digests: 2
---

# Anthropic

<!-- begin:podcast-digest -->
- x
<!-- end:podcast-digest -->
"""


def test_exact_duplicate_keys_are_healed() -> None:
    """Duplicate keys are invalid YAML — Obsidian shows raw text, not
    properties. 21 notes in the real vault acquired a second copy of another
    writer's block; a merge must repair that rather than preserve it
    faithfully."""
    out = notes.merge_owned_section(DUPLICATED, OURS)
    front, _ = notes.split_frontmatter(out)
    keys = [line.split(":", 1)[0].strip() for line in front]
    assert [k for k in keys if keys.count(k) > 1] == []
    assert "security_mentions: 19" in front
    assert "podcasts_mentions: 97" in front


def test_conflicting_duplicates_are_left_alone() -> None:
    """Same key, DIFFERENT values is a real disagreement — not ours to
    decide."""
    conflicting = DUPLICATED.replace(
        "security_mentions: 19\nsecurity_digests: 2\n---",
        "security_mentions: 42\nsecurity_digests: 2\n---",
    )
    out = notes.merge_owned_section(conflicting, OURS)
    front, _ = notes.split_frontmatter(out)
    assert "security_mentions: 19" in front
    assert "security_mentions: 42" in front  # both kept, for a human to resolve


# ── frontmatter value serialization (yaml_str / yaml_flow_list) ──────────────


def test_yaml_str_quotes_a_plain_scalar() -> None:
    assert notes.yaml_str("plain") == '"plain"'


def test_yaml_str_escapes_quotes_and_backslashes() -> None:
    assert notes.yaml_str('A "Quoted" Title') == '"A \\"Quoted\\" Title"'
    assert notes.yaml_str("back\\slash") == '"back\\\\slash"'


def test_yaml_flow_list_quotes_every_item() -> None:
    assert notes.yaml_flow_list(["a, b", 'c"d']) == '["a, b", "c\\"d"]'
    assert notes.yaml_flow_list([]) == "[]"


def test_flow_list_round_trips_through_yaml() -> None:
    import yaml

    items = ["a, b", 'has "quote"', "[bracket]", "plain"]
    assert yaml.safe_load(notes.yaml_flow_list(items)) == items
