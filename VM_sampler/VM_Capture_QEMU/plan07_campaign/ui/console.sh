#!/usr/bin/env bash
# console.sh -- one-command launcher for the Capture Console. RUN THIS ON YOUR
# LAPTOP. It SSHes into the capture server, builds + starts the bridge there,
# forwards the port, and opens the console in your browser. Ctrl-C stops the
# bridge and closes the tunnel.
#
# The bridge (and any capture's `screen`) run ON THE SERVER, so a launched
# capture keeps running even if your laptop sleeps or disconnects. Reconnect by
# running this again -- you rejoin the same running captures.
#
#   ./console.sh user@server
#   SERVER=user@server ./console.sh
#
# Optional env:  REMOTE_DIR (server path to VM_Capture_QEMU),
#                LPORT (port on YOUR LAPTOP, default 8765 -- 8000 is often taken),
#                RPORT (port on the SERVER, default 8000),
#                MIGRATE (1=auto-start the migration agent, default; 0=skip),
#                ZSTD_REMOTE_DIR (server chain store, default
#                  /project/homes/jeries/memory_traces/zstd_local),
#                TRACES_LOCAL_DIR (laptop dest, default $HOME/thesis_traces/zstd_local).
# The tunnel maps laptop:LPORT -> server:RPORT, so they can differ freely; your
# browser opens localhost:LPORT.
#
# NOTE: `user@server` here is your RESEARCH SERVER login -- NOT the guest VM.
# The guest (kali@...) goes in the console's Host identity fields, where
# run_files_controlled.py uses it to reach the VM.
set -euo pipefail

SERVER="${1:-${SERVER:-}}"
REMOTE_DIR="${REMOTE_DIR:-\$HOME/memorySignal/VM_sampler/VM_Capture_QEMU}"
LPORT="${LPORT:-8765}"          # laptop side (8000 is commonly in use locally)
RPORT="${RPORT:-8000}"          # server side

# Laptop-side migration agent (pulls completed chains to this machine; the server
# can't reach a NAT'd laptop, so the copy must start here). Auto-started below and
# controlled from the console's Data migration panel. Set MIGRATE=0 to skip it.
MIGRATE="${MIGRATE:-1}"
ZSTD_REMOTE_DIR="${ZSTD_REMOTE_DIR:-/project/homes/jeries/memory_traces/zstd_local}"
TRACES_LOCAL_DIR="${TRACES_LOCAL_DIR:-$HOME/thesis_traces/zstd_local}"

if [[ -z "$SERVER" ]]; then
  echo "usage: $0 user@server   (or set SERVER=user@server)" >&2
  echo "  env: REMOTE_DIR (default \$HOME/memorySignal/VM_sampler/VM_Capture_QEMU)," >&2
  echo "       LPORT (laptop port, default 8765), RPORT (server port, default 8000)" >&2
  exit 2
fi

open_url() { open "$1" 2>/dev/null || xdg-open "$1" 2>/dev/null || echo "  open this in your browser: $1"; }

echo "Connecting to $SERVER and starting the console bridge (Ctrl-C to stop)..."

# The token is generated ON THE SERVER by the bridge and only travels back over
# the encrypted SSH stream (never in any command line, so `ps` on a shared host
# can't see it). We read it from the banner and open the browser at the LOCAL
# forwarded port.
# Free RPORT of any bridge orphaned by a previous unclean disconnect. Kill by
# PORT, not by process name: a name match (console_bridge.py) also matches THIS
# remote command's own shell -- which contains that string -- and kills itself.
# fuser/lsof target only whoever actually listens on RPORT, never this shell. A
# running capture lives in its own `screen`, so freeing the bridge port is safe.
remote="(fuser -k ${RPORT}/tcp 2>/dev/null; lsof -ti tcp:${RPORT} -sTCP:LISTEN 2>/dev/null | xargs -r kill 2>/dev/null; true); sleep 0.4; \
  cd $REMOTE_DIR \
  && python3 plan07_campaign/ui/build_console.py --served >/dev/null \
  && exec python3 plan07_campaign/ui/console_bridge.py --port $RPORT"

# Start the laptop-side migration agent in the background. It reads its orders
# (interval, auto on/off, on-demand requests) from the control file the console
# writes on the server, and moves completed chains here. Ctrl-C stops it with the
# tunnel (trap). A running capture is unaffected -- only stable/complete chains
# are pulled. REMOTE_DIR is set inline for the agent only, so it never clobbers
# this script's own REMOTE_DIR (the campaign path).
AGENT="$(dirname "$0")/migrate_agent.sh"
AGENT_PID=""
if [[ "$MIGRATE" != "0" && -f "$AGENT" ]]; then
  AGENT_LOG="${TRACES_LOCAL_DIR%/}/migrate_agent.log"
  mkdir -p "$(dirname "$AGENT_LOG")"
  REMOTE_HOST="$SERVER" REMOTE_DIR="$ZSTD_REMOTE_DIR" LOCAL_DIR="$TRACES_LOCAL_DIR" \
    bash "$AGENT" >>"$AGENT_LOG" 2>&1 &
  AGENT_PID=$!
  echo "Migration agent running (pid $AGENT_PID) -> $TRACES_LOCAL_DIR   (log: $AGENT_LOG)"
fi
trap '[[ -n "$AGENT_PID" ]] && kill "$AGENT_PID" 2>/dev/null' EXIT INT TERM

opened=0
# -t: remote tty so the bridge line-buffers and Ctrl-C reaches it.
# 127.0.0.1 (not 'localhost') forces IPv4 to match the bridge's IPv4 bind, avoiding
# an IPv6 (::1) forward mismatch. ExitOnForwardFailure fails fast if the local port
# is taken; keepalives detect a dead link instead of hanging.
ssh -t -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L "${LPORT}:127.0.0.1:${RPORT}" "$SERVER" "$remote" 2>&1 | while IFS= read -r line; do
  printf '%s\n' "$line"
  if [[ $opened -eq 0 && "$line" =~ token=([A-Za-z0-9_-]+) ]]; then
    open_url "http://localhost:${LPORT}/?token=${BASH_REMATCH[1]}"
    opened=1
  fi
done
