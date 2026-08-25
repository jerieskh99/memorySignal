#!/usr/bin/env bash
# migrate_agent.sh -- laptop-side migration loop driven by the Capture Console.
# RUN THIS ON YOUR LAPTOP (console.sh auto-starts it alongside the tunnel). It is
# the piece that actually moves data: the server can't reach a NAT'd laptop and
# the browser can't rsync, so the copy must be initiated here.
#
# It takes its orders from a control file the UI writes on the server
# (<parent-of-REMOTE_DIR>/.migration/control.json): the loop interval, an auto
# on/off, and an on-demand request list. Each cycle it:
#   * pulls any REQUESTED chains promptly (checked every POLL_SEC),
#   * if auto is on, sweeps all "stable" chains every <interval> minutes,
# moving each with rsync --remove-source-files (freeing the server), then appends
# a line to the server ledger (.migration/ledger.jsonl) so the UI can show the
# chain as "migrated". A migratable unit is one retention chain leaf:
#   <REMOTE_DIR>/<family>/<workload>/<param_sig>/repNNN__<runid>/*.zst
#
# Standalone use (without the console) is fine too -- just run it; with no
# control file it falls back to auto sweep at the default interval.
#
# Env: REMOTE_HOST, REMOTE_DIR, LOCAL_DIR, POLL_SEC (on-demand poll cadence),
#      SSH_OPTS. Interval / auto / stable-threshold come from the control file.
set -uo pipefail

REMOTE_HOST="${REMOTE_HOST:-jeries@cybersecurity.ac.upc.edu}"
REMOTE_DIR="${REMOTE_DIR:-/project/homes/jeries/memory_traces/zstd_local}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/thesis_traces/zstd_local}"
POLL_SEC="${POLL_SEC:-30}"
MIG_DIR="$(dirname "$REMOTE_DIR")/.migration"
CONTROL="$MIG_DIR/control.json"
LEDGER="$MIG_DIR/ledger.jsonl"
SSH_OPTS="${SSH_OPTS:--o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectTimeout=15}"

DEF_INTERVAL=15
DEF_STABLE=10

log(){ echo "== $(date '+%Y-%m-%d %H:%M:%S') migrate-agent: $* =="; }

# Pull one chain (a rep* leaf, given as its path relative to REMOTE_DIR) to the
# laptop and record it in the server ledger. mode "move" adds
# --remove-source-files (deletes each file on the server only after transfer is
# confirmed, freeing space); mode "copy" leaves the server copy in place.
# --partial resumes a half-sent file; a dropped link never loses data.
pull_chain(){
  local rel="$1" mode="${2:-move}"
  case "$rel" in *..*) return 0;; esac          # never traverse up
  local remote="$REMOTE_DIR/$rel"
  local rmflag=""; [ "$mode" = "move" ] && rmflag="--remove-source-files"
  echo "  ${mode}-ing $rel"
  mkdir -p "$LOCAL_DIR/$rel"
  local tries=0
  until rsync -a --partial $rmflag -e "ssh $SSH_OPTS" \
        "$REMOTE_HOST:$remote/" "$LOCAL_DIR/$rel/"; do
    tries=$((tries + 1))
    if [ "$tries" -ge 3 ]; then
      echo "  WARN: $rel not fully pulled after 3 tries; left on server for next pass"
      return 1
    fi
    echo "  retry $tries/3 for $rel (connection dropped)"
    sleep 5
  done
  local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"chain":"%s","ts":"%s","host":"%s","action":"%s"}\n' \
    "$rel" "$ts" "$(hostname -s 2>/dev/null || hostname)" "$mode" \
    | ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p '$MIG_DIR' && cat >> '$LEDGER'" || true
  return 0
}

cleanup_empty(){
  ssh $SSH_OPTS "$REMOTE_HOST" "find '$REMOTE_DIR' -mindepth 1 -type d -empty -delete 2>/dev/null" || true
}

LAST_SWEEP=0
log "started (poll ${POLL_SEC}s) $REMOTE_HOST:$REMOTE_DIR -> $LOCAL_DIR"

while :; do
  now=$(date +%s)
  ctrl=$(ssh $SSH_OPTS "$REMOTE_HOST" "cat '$CONTROL' 2>/dev/null" || true)

  # Parse control (missing/empty -> defaults). Line 1: interval auto stable.
  scalars=$(printf '%s' "$ctrl" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    if not isinstance(d, dict): d = {}
except Exception:
    d = {}
print(int(d.get("interval_min", '"$DEF_INTERVAL"')),
      1 if d.get("auto", True) else 0,
      int(d.get("stable_min", '"$DEF_STABLE"')))
' 2>/dev/null || true)
  [ -n "$scalars" ] || scalars="$DEF_INTERVAL 1 $DEF_STABLE"
  read -r INTERVAL AUTO STABLE <<<"$scalars"

  requested=$(printf '%s' "$ctrl" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    req = d.get("requested") or []
except Exception:
    req = []
for c in req:
    if isinstance(c, str):
        print(c + "\tmove")
    elif isinstance(c, dict) and c.get("chain"):
        print(c["chain"] + "\t" + ("copy" if c.get("mode") == "copy" else "move"))
' 2>/dev/null || true)

  pulled=0

  # --- on-demand: pull requested chains (move or copy) that still exist ---
  if [ -n "$requested" ]; then
    while IFS=$'\t' read -r rel mode; do
      [ -n "$rel" ] || continue
      [ -n "$mode" ] || mode="move"
      if ssh $SSH_OPTS "$REMOTE_HOST" "test -d '$REMOTE_DIR/$rel'" 2>/dev/null; then
        pull_chain "$rel" "$mode" && pulled=$((pulled + 1))
      fi
    done <<EOF
$requested
EOF
  fi

  # --- auto sweep of all stable chains, at most every INTERVAL minutes ---
  if [ "$AUTO" = "1" ] && [ $((now - LAST_SWEEP)) -ge $((INTERVAL * 60)) ]; then
    log "auto sweep (stable >= ${STABLE}m); next in ${INTERVAL}m"
    stable=$(ssh $SSH_OPTS "$REMOTE_HOST" \
      "find '$REMOTE_DIR' -mindepth 1 -type d -name 'rep*' -mmin +$STABLE -print 2>/dev/null" || true)
    if [ -n "$stable" ]; then
      while IFS= read -r remote_chain; do
        [ -n "$remote_chain" ] || continue
        rel="${remote_chain#"$REMOTE_DIR"/}"
        pull_chain "$rel" "move" && pulled=$((pulled + 1))
      done <<EOF
$stable
EOF
    fi
    LAST_SWEEP=$now
  fi

  [ "$pulled" -gt 0 ] && cleanup_empty
  sleep "$POLL_SEC"
done
