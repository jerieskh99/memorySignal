#!/usr/bin/env bash
# backfill_migrated_stats.sh -- ONE-SHOT: fill snapshot count + size into ledger
# entries for chains that were migrated by the OLD agent (which recorded only
# {chain, ts, action}, no stats). RUN THIS ON YOUR LAPTOP -- it needs the local
# chain copies AND ssh to the server.
#
# Safety: append-only. It never rewrites or deletes existing ledger content. The
# console bridge keeps the LATEST entry per chain, so each appended line (which
# carries the ORIGINAL ts + action plus the new snapshots/bytes) simply
# supersedes the old stats-less one. Chains still present on the server are left
# untouched (the bridge scans those live). Idempotent: re-running skips entries
# that already have stats.
#
# Env (same names/defaults as migrate_agent.sh): REMOTE_HOST, REMOTE_DIR (the
# server ZSTD_DIR), LOCAL_DIR (the laptop copy root), SSH_OPTS. Run it once per
# ZSTD_DIR you migrated into (e.g. zstd_local, then zstd_smoke_test).
set -uo pipefail

REMOTE_HOST="${REMOTE_HOST:-jeries@cybersecurity.ac.upc.edu}"
REMOTE_DIR="${REMOTE_DIR:-/project/homes/jeries/memory_traces/zstd_local}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/thesis_traces/zstd_local}"
SSH_OPTS="${SSH_OPTS:--o ConnectTimeout=15}"
MIG_DIR="$(dirname "$REMOTE_DIR")/.migration"
LEDGER="$MIG_DIR/ledger.jsonl"

echo "server ledger : $REMOTE_HOST:$LEDGER"
echo "local copies  : $LOCAL_DIR"

# 1. Pull the current ledger from the server.
ledger=$(ssh $SSH_OPTS "$REMOTE_HOST" "cat '$LEDGER' 2>/dev/null" || true)
if [ -z "$ledger" ]; then echo "no ledger found (nothing migrated via the console yet?)"; exit 0; fi

# 2. Latest entry per chain that is migrated (not deleted) and has no stats.
#    Emit: rel <TAB> original_ts <TAB> original_action
todo=$(printf '%s' "$ledger" | python3 -c '
import sys, json
latest = {}
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: e = json.loads(line)
    except Exception: continue
    c = e.get("chain")
    if c: latest[c] = e
for c, e in latest.items():
    if e.get("action") != "delete" and e.get("snapshots") is None:
        print("\t".join([c, e.get("ts") or "", e.get("action") or "migrate"]))
')
if [ -z "$todo" ]; then echo "nothing to backfill -- every migrated chain already has stats."; exit 0; fi

# 3. For each, measure the LOCAL copy and stage an updated ledger line.
host="$(hostname -s 2>/dev/null || hostname)"
tmp="$(mktemp)"; appended=0; missing=0
while IFS=$'\t' read -r rel ts action; do
  [ -n "$rel" ] || continue
  d="$LOCAL_DIR/$rel"
  if [ ! -d "$d" ]; then echo "  skip (not on this laptop): $rel"; missing=$((missing+1)); continue; fi
  snaps=$(find "$d" -maxdepth 1 -name '*.zst' 2>/dev/null | wc -l | tr -d ' ')
  kb=$(du -sk "$d" 2>/dev/null | awk '{print $1+0}')
  bytes=$(( ${kb:-0} * 1024 ))
  [ -n "$ts" ] || ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  [ -n "$action" ] || action="migrate"
  printf '{"chain":"%s","ts":"%s","host":"%s","action":"%s","snapshots":%s,"bytes":%s}\n' \
    "$rel" "$ts" "$host" "$action" "$snaps" "$bytes" >> "$tmp"
  echo "  backfill: $rel -> $snaps snaps, $((bytes/1024/1024)) MB"
  appended=$((appended+1))
done <<EOF
$todo
EOF

# 4. Append the staged lines to the server ledger in one shot.
if [ "$appended" -gt 0 ]; then
  cat "$tmp" | ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p '$MIG_DIR' && cat >> '$LEDGER'"
  echo "done: appended $appended updated entrie(s); $missing chain(s) not on this laptop (left as-is)."
else
  echo "done: no local copies matched; nothing appended."
fi
rm -f "$tmp"
