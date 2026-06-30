#!/usr/bin/env bash
# idle_post_workload_recovery.sh  --  benign IDLE-family control workload.
#
# An idle window scheduled immediately AFTER an active workload, to capture the
# decay of residual writeback / cache cleanup as the system returns to rest. The
# script itself only sleeps; the "post-workload" character comes from when the
# capture campaign schedules it (right after a writer cell). The optional
# --label is recorded in the log so the prior workload can be noted.
# No network, no persistence, no file writes.
#
# Usage:
#   ./idle_post_workload_recovery.sh --time 120 [--label after_cache_cold_scan]

set -euo pipefail

IDLE_TIME=120
LABEL="unspecified_prior_workload"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --time)
      if [[ $# -lt 2 ]]; then echo "ERROR: --time requires a value (seconds)."; exit 1; fi
      IDLE_TIME="$2"; shift 2 ;;
    --label)
      if [[ $# -lt 2 ]]; then echo "ERROR: --label requires a value."; exit 1; fi
      LABEL="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --time <seconds> [--label <prior-workload>]"; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1"
      echo "Usage: $0 --time <seconds> [--label <prior-workload>]"; exit 1 ;;
  esac
done

if ! [[ "$IDLE_TIME" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --time must be a non-negative integer."; exit 1
fi

echo "[IDLE] post-workload recovery after '${LABEL}': sleeping for ${IDLE_TIME}s..."
sleep "$IDLE_TIME"
echo "[IDLE] Done."
