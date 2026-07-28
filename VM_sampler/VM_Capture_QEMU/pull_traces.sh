#!/usr/bin/env bash
# Pull completed zstd chains from the capture server to this machine, freeing
# the server as it goes. RUN THIS ON THE LAPTOP, not on the capture server --
# the transfer must be initiated by the machine that can reach the other one
# (the server cannot open a connection back to a NAT'd, sleeping laptop).
#
# Only "stable" chain folders are transferred: a chain directory whose mtime is
# older than STABLE_MIN minutes has had no new snapshot added for that long, so
# the capture has moved on and the folder is complete. This is what keeps the
# in-progress chain -- the one the running campaign is still appending to --
# from being pulled and deleted mid-write.
#
# Files are removed from the server only after rsync confirms them received.
#
# Usage:
#   ./pull_traces.sh                 # one pass, then exit
#   LOOP_MIN=15 ./pull_traces.sh     # repeat every 15 minutes until Ctrl+C
#
# Env overrides: REMOTE_HOST, REMOTE_DIR, LOCAL_DIR, STABLE_MIN, LOOP_MIN

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-jeries@cybersecurity.pc.ac.upc.edu}"
REMOTE_DIR="${REMOTE_DIR:-/project/homes/jeries/memory_traces/zstd_local}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/thesis_traces/zstd_local}"
STABLE_MIN="${STABLE_MIN:-10}"
LOOP_MIN="${LOOP_MIN:-0}"

pull_once() {
  echo "== $(date '+%Y-%m-%d %H:%M:%S') scanning $REMOTE_HOST:$REMOTE_DIR =="

  # Chain leaf dirs (family/workload/signature/repNNN__runid) untouched for
  # STABLE_MIN minutes. A new snapshot creates a new NNNNNN.zst file, which
  # bumps the parent dir's mtime -- so an old mtime means "no longer growing".
  local stable
  stable=$(ssh "$REMOTE_HOST" \
    "find '$REMOTE_DIR' -mindepth 1 -type d -name 'rep*' -mmin +$STABLE_MIN -print 2>/dev/null" \
    || true)

  if [[ -z "$stable" ]]; then
    echo "  nothing stable to pull (in-progress chains are skipped by design)"
    return 0
  fi

  local n=0
  while IFS= read -r remote_chain; do
    [[ -n "$remote_chain" ]] || continue
    local rel="${remote_chain#"$REMOTE_DIR"/}"
    echo "  pulling $rel"
    mkdir -p "$LOCAL_DIR/$rel"
    # --remove-source-files deletes each file on the server only after it is
    # confirmed transferred; a dropped connection leaves the server copy intact.
    rsync -a --remove-source-files -e ssh \
      "$REMOTE_HOST:$remote_chain/" "$LOCAL_DIR/$rel/"
    n=$((n + 1))
  done <<< "$stable"

  # rsync empties the folders but leaves the (now empty) directory tree behind.
  ssh "$REMOTE_HOST" "find '$REMOTE_DIR' -mindepth 1 -type d -empty -delete 2>/dev/null" || true

  echo "  pulled $n chain(s) -> $LOCAL_DIR"
}

if [[ "$LOOP_MIN" -gt 0 ]]; then
  echo "looping every ${LOOP_MIN}m (Ctrl+C to stop); stable threshold ${STABLE_MIN}m"
  while :; do
    pull_once || echo "  WARNING: pass failed; will retry next cycle"
    sleep $((LOOP_MIN * 60))
  done
else
  pull_once
fi
