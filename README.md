# video-digest

Takes a video URL (YouTube first), gets the best available transcript,
produces a structured summary, and writes an Obsidian note with deep links
back into the source — so the content can be read instead of watched. Sibling
to `podcast-digest`, `security-digest`, `taster`, `clippings-topics` and
`vault-ask` — see `~/projects/homelab/README.md` for the shared deploy
contract and vault-writing rules this follows.

**Status: pipeline + ingress wired (M0–M6).** `POST /jobs` resolves and
stores a video, and a background runner (`pipeline/runner.py`, scheduled)
walks it resolve → metadata → transcript → normalise → summarise → write
until a note lands in the vault, resuming from the first unfinished stage on
a retry. Caption-free videos park until the ASR worker is next awake; a
failed job writes nothing and (when `notifications.enabled`) alerts. A
scheduled watcher turns bare URLs dropped into `13 video-summaries/_video-queue.md`
into jobs and rewrites each line as a wikilink once its note exists.
`GET /jobs/{id}` reports stage progress, `POST /videos/{id}/rewrite`
re-renders from the stored digest, `GET /videos` exports finished summaries
(see below), and `GET /metrics` gives tier / spend totals. Still to do: the iOS Shortcut itself (client-side) and the watchlist
poller (deliberately deferred — see the design doc).

It is the **sixth** application writing to this vault, after `taster`,
`podcast-digest`, `security-digest`, `clippings-topics` and (read-only)
`vault-ask`. The rules for doing that without eating another writer's work are
in `~/.claude/skills/obsidian-vault-writer`, and this follows them: whole-file
ownership with a pinned filename for `13 video-summaries/` and
`14 video-transcripts/`, section ownership (`<!-- begin:video-digest -->`,
`video_` frontmatter prefix) only in the shared `99 topics/`.

```sh
./deploy                 # build, ship, bring it up on the NAS
./deploy check            # what's running, and reachable over the LAN
uv run python -m video_digest   # run locally, against ./config.yaml + .env
uv run pytest
```

## What it does not do (yet)

No iOS Shortcut — the API is Shortcut-shaped (202 + job id,
`status: "queued_for_transcription"` for caption-free videos, `GET /jobs/{id}`
to poll) but the Shortcut itself is not built, and the service is LAN-only
(no port-forward, no tunnel). No watchlist polling — the `watchlist` config
block is parsed and validated but deliberately unused (design §4). yt-dlp is
pinned in the image with no runtime self-update (the rootfs is read-only);
its version shows at `/healthz`.

## Reading the summaries elsewhere

`GET /videos` lists finished summaries as JSON, newest first, so another
service can mirror them. It is what `podcast-digest` imports from: the vault is
a fine place to keep notes and a poor place to work through them, since nothing
there tracks what has been read, and podcast-digest's console already does.

```
GET /videos?limit=50&since=<written_at>&since_id=<id>     X-API-Key required
```

Only rows that finished writing a note are returned (`stage_write = done`, with
both a `written_at` and a `digest`). Each carries `id`, `video_id`, `adapter`,
`note_path`, `written_at`, `updated_at` and the parsed `metadata` and `digest`.
**Never `transcript_json`** — by far the largest column, and the point of the
endpoint is the summary.

Two details a client depends on:

- **The cursor is the `(written_at, id)` pair, not the timestamp alone.**
  `written_at` is not unique: migration 2 backfilled it from `created_at`, and
  expanding a playlist stamps several rows in the same second. A timestamp-only
  cursor silently drops whichever rows share a page boundary — data loss the
  caller cannot detect, because the response looks complete.
- **A `+` in the offset must be encoded.** `+00:00` in a query string decodes
  to a space, so a hand-built URL yields an empty page and a client that stops
  early. The endpoint repairs that specific corruption rather than returning
  nothing; a cursor that still will not parse is a 400, so "no results" and
  "your cursor was garbage" never look alike.

`written_at` is write-once (enforced in SQL with `COALESCE`) precisely so this
ordering is stable: a value that moved would reorder a row relative to a
client's cursor, and moving forward would skip it for good.

## Topic linking

A video's `topics`/`entities` link into the shared `99 topics/` page for the
same subject when one already exists, rather than each video minting its own
near-duplicate. Two layers, in the order they actually help:

1. **Vocabulary hinting** (`pipeline/write.py::fetch_known_topics`, fed into
   the reduce prompt as `known_topics`). The vault's existing topic titles
   are shown to the model with an explicit instruction to reuse one exactly
   rather than invent a near-spelling — `clippings-topics`' pattern, ported
   here. This is the primary defence: it stops "Anthropic PBC" from being
   generated in the first place, for the common case where the model
   recognises the two as the same thing once shown the option.
2. **Alias matching** (`pipeline/write.py::resolve_topic_links` /
   `_alias_match`), for what hinting can't catch — an LLM that doesn't
   follow the instruction, or a spelling nobody's shown it yet. A name that
   doesn't slug-match any existing page is also checked against every
   candidate's `aliases:` (generic, unprefixed — nothing populates it yet,
   but a human or a future sibling writer could, and this needs no code
   change the day one does) and this writer's own `video_aliases:`
   (prefixed, accumulated in `_update_topic_page`: whenever a mention
   resolves to a page under a name that differs from its established
   `title:`, that phrasing is recorded as an alias, so the *next* time it
   shows up it resolves without minting a duplicate page).

Honest limit: alias matching is reactive, not proactive. The very first
time a genuinely novel spelling appears — nothing already links it by slug
or by a recorded alias — it still falls through to the ordinary
`topic_creation_threshold` path, same as before this existed. Vocabulary
hinting is what's meant to catch that case; aliases are the backstop for
once something's been seen at least once.

## Configuration

Non-secret settings in `config.yaml` (placeholders only — see the header
comment there for the two YAML traps this has to avoid). Secrets and
deployment topology (the vault's CouchDB address, the ASR worker's address)
are environment-only: copy `.env.example` to `.env` for local runs, and
`deploy.env.example` to `.deploy.env` for `./deploy`. Neither is tracked.

## The ASR worker

Transcription-by-audio (design's T2 tier) always runs remotely, on whichever
machine runs `~/projects/asr-server` (a `speaches` container speaking the
OpenAI audio API shape). There is no local fallback — the NAS's realtime
factor makes local ASR a non-option, the same finding that shaped
`podcast-digest`'s own remote ASR backend, which this vendors.
