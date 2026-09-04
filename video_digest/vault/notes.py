"""One topic note, many writers.

**Ported from `podcast_agent/notes.py`** by way of `clippings_topics/notes.py`
(the third copy, and this the fourth) — only `OWNER` and `KEY_PREFIX` differ,
plus the absence of podcast-digest's legacy-adoption machinery: like
clippings-topics, this app has never written a topic note in any other
format, so there is nothing here to adopt. Fix bugs in every copy or none —
a divergence here corrupts notes the other three applications also write to.

Whole-file notes (`13 video-summaries/`, `14 video-transcripts/`) do not use
this module at all — plan §1.2 keeps section ownership to `99 topics/`, the
one folder genuinely shared with other writers. See vault/livesync.py's
`project(..., merge=False)` for the whole-file path.

A region between ``<!-- begin:<owner> -->`` and ``<!-- end:<owner> -->`` is
owned wholly by that writer, replaced on every run. Frontmatter keys are
namespaced per owner (prefixed keys are ours, replaced every run; unprefixed
keys like `title`/`tags` describe the note as a whole and are create-only —
supplied only when absent, never overwritten).
"""

from __future__ import annotations

import re

#: This application's owner tag (homelab/README.md's owner-tag table).
OWNER = "video-digest"

#: Frontmatter keys this writer owns are prefixed.
KEY_PREFIX = "video_"


def begin_marker(owner: str = OWNER) -> str:
    return f"<!-- begin:{owner} -->"


def end_marker(owner: str = OWNER) -> str:
    return f"<!-- end:{owner} -->"


def _region(owner: str) -> re.Pattern[str]:
    return re.compile(
        rf"[ \t]*{re.escape(begin_marker(owner))}.*?{re.escape(end_marker(owner))}[ \t]*",
        re.DOTALL,
    )


def wrap(body: str, owner: str = OWNER) -> str:
    """Mark a block as owned, so a later run can replace exactly this much."""
    return f"{begin_marker(owner)}\n{body.strip()}\n{end_marker(owner)}"


def yaml_str(value: str) -> str:
    """A YAML-safe double-quoted scalar. These frontmatter values are titles
    and channel names — third-party text (design §5 S4's prompt-injection
    note applies here too) — so quote and escape rather than trust them to
    be YAML-safe as written."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def yaml_flow_list(items: list[str]) -> str:
    """A YAML flow-style list with every item quoted/escaped via `yaml_str`,
    so a value containing `,` `"` or `]` round-trips instead of corrupting
    the line."""
    return "[" + ", ".join(yaml_str(i) for i in items) + "]"


_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def split_frontmatter(text: str) -> tuple[list[str], str]:
    """``(frontmatter lines, body)``. No YAML parse: these notes are line-per-key
    by construction, and a real parser would reformat a human's frontmatter as
    the price of reading it."""
    match = _FRONTMATTER.match(text)
    if not match:
        return [], text
    return match.group(1).split("\n"), text[match.end() :]


def _key(line: str) -> str:
    return line.split(":", 1)[0].strip()


def merge_frontmatter(
    existing: list[str], ours: list[str], *, prefix: str = KEY_PREFIX
) -> list[str]:
    """Replace our prefixed keys; leave every other line exactly as it was.

    Two classes of key, and the difference matters:

    * **Prefixed** (``video_topic_videos``) are ours. Replaced every run, and
      dropped when this run no longer produces them.
    * **Unprefixed** (``type``, ``title``, ``tags``) describe the note as a
      whole, so they belong to whoever created it. We supply them when the key
      is absent and never otherwise — overwriting ``tags: [topic, ai]`` with
      our own ``tags: [topic]`` would silently drop a tag the reader added,
      and YAML would not even complain, because the last duplicate key wins.
    """
    kept: list[str] = []
    for line in existing:
        if not line.strip() or line.startswith(prefix):
            continue
        # Drop an EXACT duplicate of a line already kept — a real, observed
        # corruption on this vault (21 topic notes acquired a second copy of
        # another writer's block; see the other three copies of this module
        # for the full story). Only exact duplicates: two lines with the same
        # key and different values are a real disagreement between writers,
        # left for a human to see rather than silently resolved.
        if line in kept:
            continue
        kept.append(line)
    seen = {_key(line) for line in kept}
    owned = [line for line in ours if line.startswith(prefix)]
    seeds = [line for line in ours if not line.startswith(prefix) and _key(line) not in seen]
    return kept + seeds + owned


def merge_owned_section(existing: str | None, ours: str, *, owner: str = OWNER) -> str:
    """Our section written into ``existing``, leaving everything else alone.

    ``existing`` is the note as the vault currently holds it, or None when
    there is no note yet. ``ours`` is a complete note as we would write it
    fresh — frontmatter, a title, and one marked region.

    Three cases, in the order they are checked:

    1. **No existing note** — ours becomes the file.
    2. **A marked region of ours is present** — replaced in place, so the
       human's prose above it and any other writer's section below it keep
       their position on the page.
    3. **No marked region** — our section is appended, below whatever is
       already there.
    """
    if not existing or not existing.strip():
        return ours

    our_front, our_body = split_frontmatter(ours)
    our_region = _region(owner).search(our_body)
    our_section = our_region.group(0).strip() if our_region else wrap(our_body.strip(), owner)

    front, body = split_frontmatter(existing)
    merged_front = merge_frontmatter(front, our_front)

    if _region(owner).search(body):
        # A plain replacement string would treat backslashes in our section as
        # regex escapes.
        body = _region(owner).sub(lambda _: our_section, body, count=1)
    else:
        body = body.rstrip() + "\n\n" + our_section + "\n"

    head = "---\n" + "\n".join(merged_front) + "\n---\n" if merged_front else ""
    return head + body
