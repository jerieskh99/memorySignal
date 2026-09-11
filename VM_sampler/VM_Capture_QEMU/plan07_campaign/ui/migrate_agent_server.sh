#!/usr/bin/env bash
# migrate_agent_server.sh -- server-side migration loop. RUN THIS ON THE CAPTURE
# SERVER (e.g. inside `screen -dmS mem_migrate_server`). It replaces the laptop
# agent (migrate_agent.sh) for the steady state: retention chains are moved from
# ZSTD_DIR (local /project disk) straight onto the NFS archive mounted on the
# same host. No laptop, no campus-link round trip -- a local copy onto the mount.
#
# It honours the SAME control file the Capture Console writes
# (<parent-of-SRC_DIR>/.migration/control.json: interval_min / auto / stable_min /
# requested) and appends to the SAME ledger (.migration/ledger.jsonl), so the
# console keeps showing chains as "migrated" exactly as before. Ledger lines carry
# an extra "dest":"nfs" so the two eras are distinguishable.
#
# Safety for the live capture: only chains whose directory has been idle >=
# stable_min are touched (the consumer writes a snapshot every few seconds, so an
# active chain is never idle). A source is removed ONLY after the copy verifies on
# snapshot count AND total bytes. Copies run under nice/ionice so the consumer's
# writes to /project are never starved.
#
# After each sweep that moved something, idle substrate CSVs in QUEUE_DIR are
# compressed and placed next to their chain on NFS via place_csv.py (one call per
# run label seen), so the archive stays self-contained without any manual step.
#
# Env: SRC_DIR, DST_DIR, QUEUE_DIR, RUNS_DIR, POLL_SEC.
set -uo pipefail

SRC_DIR="${SRC_DIR:-/project/homes/jeries/memory_traces/zstd_local}"
DST_DIR="${DST_DIR:-/mnt/nfs/jeries/memory_traces/zstd_local}"
QUEUE_DIR="${QUEUE_DIR:-/project/homes/jeries/memory_traces/queue_dir}"
RUNS_DIR="${RUNS_DIR:-$HOME/memorySignal/VM_sampler/VM_Capture_QEMU/plan07_campaign/runs}"
PLACE_CSV="${PLACE_CSV:-$(dirname "$0")/place_csv.py}"
# The archive keeps its own inventory (plan10_analysis/archive_manifest.py): a chain is
# registered there the moment its move has verified, so readers never walk the archive.
REGISTER="${REGISTER:-$(dirname "$0")/../../plan10_analysis/archive_manifest.py}"
MIG_DIR="$(dirname "$SRC_DIR")/.migration"
CONTROL="$MIG_DIR/control.json"
LEDGER="$MIG_DIR/ledger.jsonl"
POLL_SEC="${POLL_SEC:-30}"
DEF_INTERVAL=15
DEF_STABLE=10
SNAP='[0-9][0-9][0-9][0-9][0-9][0-9].zst'

log(){ echo "== $(date '+%Y-%m-%d %H:%M:%S') migrate-agent-server: $* =="; }

snap_count(){ ls "$1"/$SNAP 2>/dev/null | wc -l | tr -d ' '; }
snap_bytes(){ find "$1" -maxdepth 1 -name "$SNAP" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'; }

# Move (or copy) one chain leaf, given as its path relative to SRC_DIR.
move_chain(){
  local rel="$1" mode="${2:-move}"
  case "$rel" in *..*) return 0;; esac
  local src="$SRC_DIR/$rel" dst="$DST_DIR/$rel"
  [ -d "$src" ] || return 0
  local n_src; n_src=$(snap_count "$src")
  [ "$n_src" -gt 0 ] || return 0                   # empty husk; cleanup_empty handles it
  echo "  ${mode}-ing $rel ($n_src snaps)"
  mkdir -p "$dst"
  local tries=0
  until nice -n 19 ionice -c3 rsync -a --partial "$src/" "$dst/"; do
    tries=$((tries + 1))
    if [ "$tries" -ge 3 ]; then echo "  WARN: $rel not fully copied after 3 tries; left in place"; return 1; fi
    sleep 5
  done
  local n_dst b_src b_dst
  n_dst=$(snap_count "$dst"); b_src=$(snap_bytes "$src"); b_dst=$(snap_bytes "$dst")
  if [ "$n_src" != "$n_dst" ] || [ "$b_src" != "$b_dst" ]; then
    echo "  WARN: verify MISMATCH for $rel (src $n_src/${b_src}B  dst $n_dst/${b_dst}B); source kept"
    return 1
  fi
  [ "$mode" = "move" ] && rm -rf "$src"
  local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  mkdir -p "$MIG_DIR"
  printf '{"chain":"%s","ts":"%s","host":"%s","action":"%s","snapshots":%s,"bytes":%s,"dest":"nfs"}\n' \
    "$rel" "$ts" "$(hostname -s 2>/dev/null || hostname)" "$mode" "$n_dst" "$b_dst" >> "$LEDGER"
  echo "  ok: $rel -> nfs ($n_dst snaps, verified)"
  register_nfs "$rel"
  return 0
}

# Tell the archive's manifest about a verified recording. A failure here is loud but never
# undoes the move: the chain is safe on NFS either way, and `rebuild` reconciles later.
register_nfs(){
  [ -f "$REGISTER" ] || { echo "  WARN: no archive_manifest.py at $REGISTER; $1 not registered"; return 0; }
  python3 "$REGISTER" register "$DST_DIR" "$1" 2>&1 | sed 's/^/  manifest: /' || echo "  WARN: manifest register failed for $1"
}

cleanup_empty(){ find "$SRC_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true; }

# Compress idle substrate CSVs and place them beside their chains, per run label.
place_csvs(){
  [ -f "$PLACE_CSV" ] || return 0
  ( cd "$QUEUE_DIR" 2>/dev/null || exit 0
    find . -maxdepth 1 -name '*.substrate_trajectory.csv' ! -newermt '-15 minutes' -print0 \
      | nice -n 19 xargs -0 -r -P 1 -I{} zstd -T2 --rm -q {} 2>/dev/null )
  local label
  for label in "$@"; do
    [ -f "$RUNS_DIR/${label}_steps.txt" ] || continue
    python3 "$PLACE_CSV" "$RUNS_DIR/${label}_steps.txt" "$DST_DIR" "$QUEUE_DIR" \
      "rep001__${label}" any --skip-missing --move 2>&1 | grep -E '^(moved|PROBLEM)|mapped=' | sed 's/^/  csv: /'
  done
}

LAST_SWEEP=0
log "started (poll ${POLL_SEC}s) $SRC_DIR -> $DST_DIR"

while :; do
  now=$(date +%s)
  ctrl=$(cat "$CONTROL" 2>/dev/null || true)

  read -r INTERVAL AUTO STABLE <<<"$(printf '%s' "$ctrl" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    if not isinstance(d, dict): d = {}
except Exception:
    d = {}
print(int(d.get("interval_min", '"$DEF_INTERVAL"')), 1 if d.get("auto", True) else 0, int(d.get("stable_min", '"$DEF_STABLE"')))
' 2>/dev/null || echo "$DEF_INTERVAL 1 $DEF_STABLE")"

  requested=$(printf '%s' "$ctrl" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin); req = d.get("requested") or []
except Exception:
    req = []
for c in req:
    if isinstance(c, str): print(c + "\tmove")
    elif isinstance(c, dict) and c.get("chain"): print(c["chain"] + "\t" + ("copy" if c.get("mode") == "copy" else "move"))
' 2>/dev/null || true)

  moved=0; labels=""
  if [ -n "$requested" ]; then
    while IFS=$'\t' read -r rel mode; do
      [ -n "$rel" ] || continue
      if [ -d "$SRC_DIR/$rel" ] && move_chain "$rel" "${mode:-move}"; then
        moved=$((moved + 1)); labels="$labels ${rel##*__}"
      fi
    done <<<"$requested"
  fi

  if [ "$AUTO" = "1" ] && [ $((now - LAST_SWEEP)) -ge $((INTERVAL * 60)) ]; then
    log "auto sweep (stable >= ${STABLE}m); next in ${INTERVAL}m"
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      rel="${d#"$SRC_DIR"/}"
      if move_chain "$rel" move; then moved=$((moved + 1)); labels="$labels ${rel##*__}"; fi
    done < <(find "$SRC_DIR" -mindepth 1 -type d -name 'rep*' -mmin +"$STABLE" -print 2>/dev/null | sort)
    LAST_SWEEP=$now
  fi

  if [ "$moved" -gt 0 ]; then
    cleanup_empty
    place_csvs $(printf '%s\n' $labels | sort -u)
  fi
  sleep "$POLL_SEC"
done
