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
#                RPORT (port on the SERVER, default 8000).
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
# Kill any bridge left over from a previous dropped session -- an unclean SSH
# disconnect can orphan it, and it keeps holding RPORT. A running capture lives in
# its own `screen`, independent of the bridge, so this is safe. Then build + start.
remote="pkill -f 'plan07_campaign/ui/console_bridge.py' 2>/dev/null; sleep 0.4; \
  cd $REMOTE_DIR \
  && python3 plan07_campaign/ui/build_console.py --served >/dev/null \
  && exec python3 plan07_campaign/ui/console_bridge.py --port $RPORT"

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
