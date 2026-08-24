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
#                LPORT (local port, default 8000), RPORT (server port, default 8000).
#
# NOTE: `user@server` here is your RESEARCH SERVER login -- NOT the guest VM.
# The guest (kali@...) goes in the console's Host identity fields, where
# run_files_controlled.py uses it to reach the VM.
set -euo pipefail

SERVER="${1:-${SERVER:-}}"
REMOTE_DIR="${REMOTE_DIR:-\$HOME/memorySignal/VM_sampler/VM_Capture_QEMU}"
LPORT="${LPORT:-8000}"
RPORT="${RPORT:-8000}"

if [[ -z "$SERVER" ]]; then
  echo "usage: $0 user@server   (or set SERVER=user@server)" >&2
  echo "  env: REMOTE_DIR (default \$HOME/memorySignal/VM_sampler/VM_Capture_QEMU)," >&2
  echo "       LPORT (default 8000), RPORT (default 8000)" >&2
  exit 2
fi

open_url() { open "$1" 2>/dev/null || xdg-open "$1" 2>/dev/null || echo "  open this in your browser: $1"; }

echo "Connecting to $SERVER and starting the console bridge (Ctrl-C to stop)..."

# The token is generated ON THE SERVER by the bridge and only travels back over
# the encrypted SSH stream (never in any command line, so `ps` on a shared host
# can't see it). We read it from the banner and open the browser at the LOCAL
# forwarded port.
remote="cd $REMOTE_DIR \
  && python3 plan07_campaign/ui/build_console.py --served >/dev/null \
  && exec python3 plan07_campaign/ui/console_bridge.py --port $RPORT"

opened=0
# -t: give the bridge a remote tty so it line-buffers and Ctrl-C reaches it.
# Stdout is piped so we can catch the token line; ssh's stdin stays the terminal,
# so key/password auth still prompts normally.
ssh -t -L "${LPORT}:localhost:${RPORT}" "$SERVER" "$remote" 2>&1 | while IFS= read -r line; do
  printf '%s\n' "$line"
  if [[ $opened -eq 0 && "$line" =~ token=([A-Za-z0-9_-]+) ]]; then
    open_url "http://localhost:${LPORT}/?token=${BASH_REMATCH[1]}"
    opened=1
  fi
done
