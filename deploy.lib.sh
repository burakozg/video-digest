#!/usr/bin/env bash
# deploy.lib.sh — shared QNAP NAS deployment plumbing for the homelab projects.
#
# CANONICAL COPY: homelab/deploy.lib.sh. This file is VENDORED — an exact byte
# copy — into family_calendar/, podcast-digest/, security-digest/ and taster/,
# so each repo stays self-contained and independently publishable. Fix bugs in
# homelab/ and re-vendor; homelab/vendor-check.sh reports any drift.
#
# Sourced by each repo's ./deploy, which is expected to `set -euo pipefail`.
#
# Everything here exists because all four projects had already grown their own
# version of it, and the versions disagreed in ways that cost real data. Each
# function below is the best of the four, not a fresh rewrite — see the notes.

# ── configuration ─────────────────────────────────────────────────────────────

die() { echo "✗ $*" >&2; exit 1; }

# nas_load_env <repo-root> <default-app-dir>
#
# Real per-deployment values live in a git-ignored .deploy.env next to a tracked
# deploy.env.example, so they exist in exactly one place per project and never
# reach git history. Auto-sourced here so nothing needs exporting by hand; every
# var still accepts a plain shell export, which wins only if .deploy.env is
# silent about it.
nas_load_env() {
  local root="$1" default_app_dir="$2"

  # shellcheck disable=SC1091
  [ -f "${root}/.deploy.env" ] && . "${root}/.deploy.env"

  # NAS_HOST is RETIRED, and this check is not cosmetic. It used to mean the ssh
  # destination in two repos and the container's macvlan IP in a third. Silently
  # accepting it here would either break the first two or point ssh at a
  # container, so it is a hard error with the two replacements named.
  if [ -n "${NAS_HOST:-}" ]; then
    die "NAS_HOST is ambiguous and no longer used (it meant the ssh destination in
     some repos and a container's LAN IP in others). Set instead, in .deploy.env:
       NAS_SSH=user@host   the NAS host account to ssh into
       APP_LAN_IP=10.0.0.x the container's own address on the qnet macvlan"
  fi

  NAS_SSH="${NAS_SSH:-deploy@nas.local}"
  NAS_SSH_PORT="${NAS_SSH_PORT:-22}"
  NAS_APP_DIR="${NAS_APP_DIR:-${default_app_dir}}"
  # Container Station's docker is not on PATH for non-interactive SSH commands —
  # QNAP only wires it in for interactive logins — so it is addressed by full path.
  NAS_DOCKER_BIN="${NAS_DOCKER_BIN:-/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker}"
  # This Mac is arm64 and the NAS is x86_64. Building for the wrong architecture
  # produces an image that dies with "exec format error" on first start — not a
  # partial failure, a silent non-start. Pinned rather than inferred; confirm a
  # new NAS with `ssh -p "$NAS_SSH_PORT" "$NAS_SSH" uname -m`.
  NAS_PLATFORM="${NAS_PLATFORM:-linux/amd64}"
  # `docker` on this NAS is a shell wrapper that sources ld-wrapper.sh, which
  # does an UNCONDITIONAL `export HOME=$QPKG_DIR/homes/$(id -un)`. That directory
  # is drwxr-x--- root:administrators, so any account without one already there
  # cannot create it — and the CLI dies with
  #   mkdir .../container-station/homes/<user>: permission denied
  # before doing any work. Setting HOME yourself does nothing; the wrapper
  # overwrites it. DOCKER_CONFIG is honoured in preference to $HOME/.docker and
  # the wrapper leaves it alone, so that is the lever that actually works.
  #
  # Subcommands needing no config dir (`ps`, `config`, plain `up -d`) survive
  # without it, which is why this only surfaced on `--build`.
  #
  # Only CLI/buildx scratch state lives there and costs nothing to recreate; no
  # registry credentials are involved, every image here is local.
  NAS_DOCKER_CONFIG="${NAS_DOCKER_CONFIG:-/tmp/.docker-\$(id -un)}"
  NAS_DOCKER_ENV="DOCKER_CONFIG=${NAS_DOCKER_CONFIG}"

  export NAS_SSH NAS_SSH_PORT NAS_APP_DIR NAS_DOCKER_BIN NAS_PLATFORM
  export NAS_DOCKER_CONFIG NAS_DOCKER_ENV
}

# Consume the flags every project understands. Returns non-zero for anything
# else, so a caller can fall through to its own cases:
#
#   while [ $# -gt 0 ]; do
#     nas_global_flag "$1" || case "$1" in ... esac
#     shift
#   done
NO_APPLY=false
nas_global_flag() {
  case "$1" in
    --no-apply) NO_APPLY=true ;;
    # The old spellings, one per repo, are what this whole alignment exists to
    # kill. Recognised rather than left to fall through to "unknown argument",
    # so old muscle memory gets an answer instead of a shrug.
    --skip-run|--no-up)
      die "$1 is gone. Use --no-apply — same meaning, same spelling in every repo." ;;
    *) return 1 ;;
  esac
  return 0
}

# ── reaching the NAS ──────────────────────────────────────────────────────────

nas_ssh() { ssh -p "$NAS_SSH_PORT" "$NAS_SSH" "$@"; }

# ssh works over the VPN or on the home LAN; a cloud file-sync relay being up
# does not imply either. Fail fast with a useful message rather than hanging on
# a dead socket.
nas_require_reachable() {
  nc -z -G 5 "${NAS_SSH##*@}" "$NAS_SSH_PORT" 2>/dev/null \
    || die "Can't reach ${NAS_SSH##*@}:${NAS_SSH_PORT}. Connect the VPN (or the home LAN) first."
  # Probe key auth explicitly. Without a key every ssh below falls back to a
  # password prompt — fine by hand, but it silently blocks forever when nothing
  # can type (a background run, CI, an agent), which reads as "the deploy froze".
  if nas_ssh -o BatchMode=yes -o ConnectTimeout=5 true 2>/dev/null; then
    echo "  (ssh: ${NAS_SSH}:${NAS_SSH_PORT} — key auth OK)"
  else
    echo "  ! No ssh key for ${NAS_SSH}: every step below will prompt for a password,"
    echo "    and will hang instead of prompting if this isn't an interactive terminal."
    echo "    Fix it once:  ssh-copy-id -p ${NAS_SSH_PORT} ${NAS_SSH}"
    echo "    Keep the user as-is — it must be the account that OWNS ${NAS_APP_DIR}."
  fi
}

# ── moving files ──────────────────────────────────────────────────────────────
#
# scp and sftp DO NOT WORK against this NAS: its sshd exposes no SFTP subsystem,
# so both fail with "subsystem request failed on channel 0" and exit 255. Worse,
# written as `scp ... 2>/dev/null || true` that failure is invisible — a deploy
# that "succeeded" while transferring nothing. Everything below is plain ssh
# command execution with stdin piped through, which every SSH server supports,
# and every transfer is verified rather than assumed.

# nas_put <local> <remote> [mode]
#
# Temp file + mv: the existing file is often owned by another NAS account so it
# cannot be truncated in place, but it can be replaced — and mv is atomic, so
# there is never a half-written file for the container to read.
nas_put() {
  local src="$1" dst="$2" mode="${3:-}" want got chmod_step=""
  [ -f "$src" ] || die "nas_put: no such local file: ${src}"
  [ -n "$mode" ] && chmod_step="chmod ${mode} '${dst}.new' && "
  nas_ssh "mkdir -p '$(dirname "$dst")' && cat > '${dst}.new' && ${chmod_step}mv -f '${dst}.new' '${dst}'" < "$src" \
    || die "Couldn't write ${dst} on ${NAS_SSH} (permissions? wrong NAS_SSH?)."
  want="$(wc -c < "$src" | tr -d '[:space:]')"
  got="$(nas_ssh "wc -c < '${dst}'" | tr -d '[:space:]')"
  [ "$want" = "$got" ] \
    || die "${dst} didn't copy cleanly (${want} bytes here, ${got} there)."
}

# nas_get <remote> <local> — non-zero if the remote file is absent OR empty.
#
# The empty check is the point: without it a failed pull leaves a zero-byte file
# that reads as "the remote copy is empty", and any caller that then deletes the
# remote original destroys it with nothing kept locally. That has happened.
nas_get() {
  nas_ssh "[ -f '$1' ] && cat '$1'" > "$2" 2>/dev/null && [ -s "$2" ]
}

# nas_exists <remote> — true if the path exists at all (file, dir, anything).
nas_exists() { nas_ssh "[ -e '$1' ]" 2>/dev/null; }

# nas_sqlite_get <container> <db-path-inside-container> <local>
#
# A consistent copy of a LIVE SQLite database. Use this, never nas_get, for
# anything a running container is writing to.
#
# `nas_get` (and the plain `ssh cat` this whole library exists to standardise)
# is wrong for SQLite in WAL mode, which is the default for every database
# here: committed transactions live in a `-wal` sidecar until a checkpoint
# folds them into the `.db`. Copy the `.db` on its own and you get a file that
# opens cleanly, passes `PRAGMA integrity_check`, and is **silently missing
# recent writes**. There is no error, no short read, and no empty file for
# nas_get's non-empty check to catch — which is precisely the class of failure
# that check was added for.
#
# The work happens inside the container because it cannot happen anywhere
# else: this NAS has neither `sqlite3` nor any `python` on the host (checked
# 2026-08-28), so nothing on the host can checkpoint or snapshot a database.
# The application images have python, and SQLite's online backup API takes a
# consistent snapshot *while the application keeps writing* — no stop, no
# downtime, no lock held for the duration of the copy.
#
# Verified against vault-ask's 62 MB index with the service live: integrity ok,
# full row counts, no interruption.
nas_sqlite_get() {
  local container="$1" db="$2" out="$3" tmp want got
  tmp="/tmp/.nas_sqlite_get.$$.db"

  # Script over stdin rather than `python3 -c '...'`: the one-liner needs
  # quotes of its own, and nesting those through ssh -> docker exec -> sh is
  # how quoting bugs get written. `-i` is what lets stdin reach python.
  nas_ssh "${NAS_DOCKER_ENV} '${NAS_DOCKER_BIN}' exec -i '${container}' python3 - '${db}' '${tmp}'" <<'PY' \
    || die "nas_sqlite_get: snapshot failed inside '${container}'. No python3 in that image, or ${db} is not readable there."
import sqlite3, sys
# mode=ro: a reader must never create or migrate the database it is copying.
src = sqlite3.connect("file:" + sys.argv[1] + "?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close()
src.close()
PY

  # The redirect MUST be inside sh -c. `docker exec <c> wc -c < /path` resolves
  # the redirect in the *host* shell, so it reports "No such file" for a file
  # that is present and correct inside the container.
  want="$(nas_ssh "${NAS_DOCKER_ENV} '${NAS_DOCKER_BIN}' exec '${container}' sh -c 'wc -c < \"${tmp}\"'" | tr -d '[:space:]')"
  nas_ssh "${NAS_DOCKER_ENV} '${NAS_DOCKER_BIN}' exec '${container}' cat '${tmp}'" > "$out" \
    || { nas_ssh "${NAS_DOCKER_ENV} '${NAS_DOCKER_BIN}' exec '${container}' rm -f '${tmp}'" >/dev/null 2>&1
         die "nas_sqlite_get: could not read the snapshot back out of '${container}'."; }
  nas_ssh "${NAS_DOCKER_ENV} '${NAS_DOCKER_BIN}' exec '${container}' rm -f '${tmp}'" >/dev/null 2>&1 || true

  # Same byte-count contract as nas_put, for the same reason: a truncated
  # transfer that exits 0 is the failure worth catching.
  got="$(wc -c < "$out" | tr -d '[:space:]')"
  [ -s "$out" ] || die "nas_sqlite_get: ${out} came back empty — refusing to treat that as a backup."
  [ "$want" = "$got" ] \
    || die "nas_sqlite_get: ${out} is truncated (${want} bytes there, ${got} here)."
}

# nas_put_tree <remote-dir> <dir>...
#
# COPYFILE_DISABLE / --no-mac-metadata / --no-xattrs: macOS otherwise larks
# ._AppleDouble entries into the tar, which land as junk files on the NAS.
# __pycache__ is excluded because a stale .pyc is pure liability — and /app is
# read-only in the containers, so Python cannot rewrite them anyway.
nas_put_tree() {
  local dst="$1"; shift
  COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf - \
      --exclude '__pycache__' --exclude '.DS_Store' --exclude '._*' "$@" 2>/dev/null \
    | nas_ssh "mkdir -p '${dst}' && cd '${dst}' && tar -xzf - --overwrite --no-same-owner" \
    || die "Couldn't ship $* to ${dst}."
}

# nas_mirror_tree <remote-dir> <ext,ext,...> <dir>...
#
# nas_put_tree OVERLAYS — tar never deletes. A module removed here would live on
# forever there, and a stale source file is worse than a missing one: it still
# imports. So send a manifest of what should exist and delete everything else,
# which makes the push a true mirror. The directories themselves are left alone:
# they are bind-mounted into the running container, so replacing the inode would
# strand it.
#
# Extensions are passed as a bare comma list ("py,html") and the find expression
# is built from them here. Passing the expression itself would mean expanding it
# unquoted on both sides — where the shell strips nothing but DOES glob, so
# `-name '*.py'` reaches find with the quotes still attached and matches nothing.
nas_mirror_tree() {
  local dst="$1" exts="$2"; shift 2
  local local_expr=() remote_expr="" ext first=1
  local IFS_SAVE="$IFS"; IFS=,
  for ext in $exts; do
    IFS="$IFS_SAVE"
    if [ "$first" = 1 ]; then first=0; else
      local_expr+=(-o); remote_expr="${remote_expr} -o"
    fi
    local_expr+=(-name "*.${ext}")
    remote_expr="${remote_expr} -name '*.${ext}'"
    IFS=,
  done
  IFS="$IFS_SAVE"

  nas_put_tree "$dst" "$@"

  echo "→ Removing files that no longer exist here …"
  find "$@" -type f \( "${local_expr[@]}" \) -not -path '*__pycache__*' | sort \
    | nas_ssh "
        cd '${dst}' || exit 1
        cat > /tmp/nas_manifest
        find $* -type f \\( ${remote_expr} \\) -not -path '*__pycache__*' | sort > /tmp/nas_have
        # grep -vxF, not \`comm\`: this NAS's busybox has no comm, and the
        # failure was invisible — it printed \"comm: command not found\", the
        # prune quietly did nothing, and the deploy carried on reporting success
        # while stale files stayed on the target. -x whole-line, -F literal.
        stale=\$(grep -vxF -f /tmp/nas_manifest /tmp/nas_have || true)
        if [ -n \"\$stale\" ]; then echo \"\$stale\" | while read -r f; do echo \"    rm \$f\"; rm -f \"\$f\"; done; fi
        find $* -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
        rm -f /tmp/nas_manifest /tmp/nas_have" \
    || die "Couldn't prune stale files under ${dst}."
}

# ── images ────────────────────────────────────────────────────────────────────

# nas_ship_image <tag> <build-context>
#
# Streamed straight into `docker load` over a single SSH pipe — no intermediate
# tar on either end. The verification afterwards is not paranoia: `docker load`
# failing mid-stream still exits 0 often enough that trusting it is how you end
# up debugging the wrong thing for an hour.
nas_ship_image() {
  local tag="$1" ctx="$2"
  echo "== building ${tag} for ${NAS_PLATFORM} =="
  docker info >/dev/null 2>&1 || die "Docker isn't running. Start Docker Desktop."
  docker buildx build --platform "$NAS_PLATFORM" -t "$tag" --load "$ctx"

  echo "== streaming into 'docker load' on ${NAS_SSH} =="
  docker save "$tag" | gzip | nas_ssh "gunzip -c | '${NAS_DOCKER_BIN}' load"

  echo "== verifying the image landed =="
  nas_ssh "'${NAS_DOCKER_BIN}' image inspect '${tag}' --format 'loaded: {{.Id}} ({{.Architecture}})'" \
    || die "${tag} is not on the NAS after 'docker load' — the stream failed silently."
}

# nas_build_image_remote <tag> <remote-build-dir>
#
# The alternative to cross-building: source is already on the NAS, so build it
# there with whatever architecture its Docker daemon actually is. No --platform,
# no QEMU emulation — faster and more reliable for a pure-interpreter image.
nas_build_image_remote() {
  local tag="$1" dir="$2"
  echo "== building ${tag} natively on ${NAS_SSH} =="
  nas_ssh "mkdir -p ${NAS_DOCKER_CONFIG} && cd '${dir}' && ${NAS_DOCKER_ENV} '${NAS_DOCKER_BIN}' build -t '${tag}' ." \
    || die "Native build of ${tag} failed on the NAS."
  nas_ssh "'${NAS_DOCKER_BIN}' image inspect '${tag}' --format 'built: {{.Id}} ({{.Architecture}})'"
}

# ── applying ──────────────────────────────────────────────────────────────────

# nas_compose_up [-f file]... — LIFECYCLE=compose projects.
#
# Plain `up -d`: compose recreates only what actually changed and merely starts
# what is stopped. Deliberately NOT --force-recreate — see the MAC note in the
# projects' compose files. A needless recreate assigns a new MAC, and on this
# network that is exactly what makes a container briefly unreachable.
nas_compose_up() {
  local files="" f build=""
  # Leading --build: for projects whose compose file has `build:` contexts and
  # builds ON the NAS rather than shipping an image. Without it a source change
  # redeploys the previously built image and looks like it did nothing.
  if [ "${1:-}" = "--build" ]; then build="--build"; shift; fi
  for f in "$@"; do files="${files} -f ${f}"; done
  echo "== bringing the stack up in ${NAS_APP_DIR} =="
  # shellcheck disable=SC2086  # $files is a pre-quoted flag list
  nas_ssh "mkdir -p ${NAS_DOCKER_CONFIG} && cd '${NAS_APP_DIR}' && ${NAS_DOCKER_ENV} '${NAS_DOCKER_BIN}' compose ${files} up -d ${build}" \
    || die "compose up failed on the NAS."
  # shellcheck disable=SC2086
  nas_ssh "cd '${NAS_APP_DIR}' && ${NAS_DOCKER_ENV} '${NAS_DOCKER_BIN}' compose ${files} ps --format 'table {{.Name}}\t{{.Status}}'"
}

# nas_wait_reachable <name> <url> [path] [ok-pattern] [tries]
#
# "Up (healthy)" is NOT the same as "reachable". A container's healthcheck runs
# inside it against localhost, so it passes while nothing on the LAN can load the
# service — which is how a deploy reports success over a dead front door. Probe
# from HERE, over the network a user would actually come in on.
#
# Why a wait rather than a single probe: these containers sit on a macvlan in
# a /25 and anything on another subnet reaches them through a router.
# A container that has just been created is unreachable for a few seconds while
# it boots and the router learns it. With MAC addresses pinned in the compose
# files that settles in ~15s.
#
# WITHOUT pinned MACs it does not settle: docker assigns a random MAC on every
# `create`, the router keeps answering for the OLD one, and the container stays
# unreachable for MINUTES — healthy, correctly routed, able to reach the
# internet, and unreachable inbound. Measured here: still dark after 2 minutes,
# then instantly fine after a single outbound packet from inside the container
# (`docker exec <c> python -c "socket.create_connection(('<gateway>',80))"`),
# which is the manual remedy if it ever recurs. Pin the MAC and it does not.
nas_wait_reachable() {
  local name="$1" url="$2" path="${3:-/healthz}" ok="${4:-2..}" tries="${5:-12}" i=1
  while [ "$i" -le "$tries" ]; do
    if nas_probe "$name" "$url" "$path" "$ok" >/dev/null 2>&1; then
      nas_probe "$name" "$url" "$path" "$ok"
      return 0
    fi
    [ "$i" = 1 ] && echo "  … ${name} not up at ${url}${path} yet; waiting"
    sleep 5
    i=$((i + 1))
  done
  nas_probe "$name" "$url" "$path" "$ok" || {
    echo "    Still unreachable after $((tries * 5))s. If 'docker ps' says healthy,"
    echo "    suspect a stale router neighbour entry — see the note above."
    return 1
  }
}

# nas_mac_for_ip <ipv4> — a stable MAC for a container at that address.
#
# Docker assigns a RANDOM MAC on every `create`. On this network that is a real
# problem, not a cosmetic one: containers sit on a macvlan in a /25 and anything
# on another subnet reaches them through a router, which goes on answering for
# the old MAC after a recreate. The container comes back healthy, correctly
# routed, able to reach the internet — and unreachable inbound, with its
# healthcheck none the wiser because that probes localhost from inside. Measured
# here: dark for over two minutes, then instantly fine after one outbound packet
# from within the container. Pin the MAC and the recreate is invisible instead.
#
# Docker's own bridge-network convention: 02:42 followed by the four address
# octets. Locally-administered, unicast, and unique as long as the IPs are.
#
# Derived at deploy time rather than written into a tracked compose file, because
# the result encodes the real LAN address (the last four octets ARE the address)
# and so belongs in .deploy.env, not in git.
nas_mac_for_ip() {
  local ip="$1"
  case "$ip" in
    *[!0-9.]*|"") die "nas_mac_for_ip: not an IPv4 address: '${ip}'" ;;
  esac
  echo "$ip" | awk -F. 'NF != 4 { exit 1 }
    { for (i = 1; i <= 4; i++) if ($i < 0 || $i > 255) exit 1
      printf "02:42:%02x:%02x:%02x:%02x\n", $1, $2, $3, $4 }' \
    || die "nas_mac_for_ip: not an IPv4 address: '${ip}'"
}

# nas_render <template> <out> [sed-expr]...
#
# Some files cannot use ${VAR} substitution the way Compose does — either the
# tool has no such feature, or the file is pasted by hand into a UI with no CLI
# step in between. Those ship with placeholder tokens; this writes a git-ignored
# copy with the real values filled in, and never touches the tracked template.
nas_render() {
  local tpl="$1" out="$2"; shift 2
  local args=() e
  for e in "$@"; do args+=(-E -e "$e"); done
  mkdir -p "$(dirname "$out")"
  # ${args[@]+...}: an empty array under `set -u` is an error in bash 3.2, which
  # is still what /bin/bash is on macOS.
  sed ${args[@]+"${args[@]}"} "$tpl" > "$out"
  grep -q 'PLACEHOLDER' "$out" \
    && die "${out} still contains a PLACEHOLDER token — a substitution didn't match.
     Shipping it would deploy a literal placeholder as a real value."
  echo "✓ Wrote ${out}"
}

# nas_render_to_paste <template> <out> [sed-expr]...
#
# As above, plus the clipboard: for a file whose only consumer is a human
# pasting it into Container Station's UI.
nas_render_to_paste() {
  local out="$2"
  nas_render "$@"
  if command -v pbcopy >/dev/null 2>&1; then
    pbcopy < "$out"
    echo "  (copied to your clipboard)"
  fi
  echo "  Paste THAT file into Container Station — not ${1}, which holds placeholders."
}

# What "apply" means for a Container Station-managed project. Deliberately NOT
# `docker run`: Container Station owns the lifecycle, and a script that stops and
# recreates the container itself reverts it to whatever convention the script
# hard-codes (port mappings instead of static IPs, most memorably) with nothing
# in the output to say so. That footgun is exactly what the old --skip-run flag
# existed to avoid, and removing it is what let the flag go.
nas_station_recreate_notice() {
  local what="${1:-the application}"
  cat <<EOF

  Now, in Container Station:
    Applications → ${what} → Recreate → select all, paste, Recreate.

  ⚠ Container Station stores its OWN copy of the YAML. Recreate re-reads the
    stored copy, not your file — so a changed mount has zero effect until the
    rendered YAML above is pasted in again.
EOF
}

# ── checking ──────────────────────────────────────────────────────────────────

# nas_probe <name> <base-url> [path] [ok-code-pattern]
#
# Reports the HTTP status rather than just success/failure. `curl -f` alone
# cannot tell "nothing is listening" from "it answered and said no", and the
# difference matters: a CouchDB with require_valid_user=true answers 401 on
# every endpoint including /_up, so an -f probe reads a perfectly healthy
# database as down. Pass an extended-regex of acceptable codes when a non-2xx
# response is the proof of life you actually want (e.g. "2..|401").
nas_probe() {
  local name="$1" url="$2" path="${3:-/healthz}" ok="${4:-2..}" code
  # Capture curl's own status word, then normalise separately. `$(curl ...) ||
  # echo 000` looks equivalent and is not: on a timeout curl BOTH prints 000 and
  # exits non-zero, so the fallback appends a second one and you get "000000",
  # which matches no branch and reports a healthy service as an unknown code.
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "${url}${path}" 2>/dev/null) || true
  code=${code:-000}
  if [ "$code" = 000 ]; then
    echo "✗ ${name}: no answer at ${url}${path}"
    return 1
  elif echo "$code" | grep -qE "^(${ok})$"; then
    echo "✓ ${name}: up → ${url} (HTTP ${code})"
  else
    echo "✗ ${name}: ${url}${path} answered HTTP ${code}"
    return 1
  fi
}
