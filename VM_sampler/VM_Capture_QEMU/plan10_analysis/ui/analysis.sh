#!/usr/bin/env bash
# analysis.sh -- one-command launcher for the Analysis Console. Runs on the machine that
# holds the recordings (local mode) or on your laptop with an SSH source. Starts the local
# bridge on 127.0.0.1, rebuilds the served console against the scanned corpus, and opens
# the browser at the tokenised URL. Ctrl-C stops the bridge; a launched run keeps going
# (the executor is its own process) and is picked up again on the next start.
#
#   ./analysis.sh                                   local corpus at console.sh's default root
#   ./analysis.sh --root /path/to/zstd_local        local corpus elsewhere
#   ./analysis.sh --ssh user@host --key ~/.ssh/id --remote-root /project/.../zstd_local
#
# Options: --port N (default 8766), --out-dir D (runs; default ~/.cache/plan10/runs),
#          --store D (L1 store; default ~/.cache/plan10/l1), --no-open
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QEMU_DIR="$(cd "$HERE/../.." && pwd)"
PORT=8766; ROOT=""; SSH=""; KEY=""; RROOT=""; SPORT=22; OUT=""; STORE=""; OPEN="--open"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --ssh) SSH="$2"; shift 2;;
    --key) KEY="$2"; shift 2;;
    --remote-root) RROOT="$2"; shift 2;;
    --ssh-port) SPORT="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --out-dir) OUT="$2"; shift 2;;
    --store) STORE="$2"; shift 2;;
    --no-open) OPEN=""; shift;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

ARGS=(--port "$PORT")
if [[ -n "$SSH" ]]; then
  [[ -n "$RROOT" ]] || { echo "--ssh needs --remote-root" >&2; exit 2; }
  USER_="${SSH%%@*}"; HOST="${SSH#*@}"
  [[ "$SSH" == *@* ]] || { USER_=""; HOST="$SSH"; }
  SRC=$(python3 - "$HOST" "$USER_" "$KEY" "$RROOT" "$SPORT" <<'EOF'
import json, sys
h, u, k, r, p = sys.argv[1:6]
print(json.dumps({"kind": "ssh", "host": h, "user": u or None, "key": k or None, "remote_root": r, "port": int(p)}))
EOF
)
  ARGS+=(--source-json "$SRC")
elif [[ -n "$ROOT" ]]; then
  ARGS+=(--root "$ROOT")
fi
[[ -n "$OUT" ]] && ARGS+=(--out-dir "$OUT")
[[ -n "$STORE" ]] && ARGS+=(--store "$STORE")
[[ -n "$OPEN" ]] && ARGS+=("$OPEN")

cd "$QEMU_DIR"
exec python3 plan10_analysis/ui/analysis_bridge.py "${ARGS[@]}"
