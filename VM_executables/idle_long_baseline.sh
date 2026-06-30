#!/usr/bin/env bash
# idle_long_baseline.sh  --  benign IDLE-family control workload.
#
# A long, uncontaminated idle window: optionally drop the page cache first (so
# residual writeback from a prior workload does not leak in), then sleep. Used
# as the stationary lower-bound reference against which active workloads are
# interpreted. No network, no persistence, no file writes (besides the optional
# kernel drop_caches knob, which only releases clean cache).
#
# Usage:
#   ./idle_long_baseline.sh --time 600 [--drop-caches]

set -euo pipefail

IDLE_TIME=600
DROP_CACHES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --time)
      if [[ $# -lt 2 ]]; then echo "ERROR: --time requires a value (seconds)."; exit 1; fi
      IDLE_TIME="$2"; shift 2 ;;
    --drop-caches)
      DROP_CACHES=1; shift ;;
    -h|--help)
      echo "Usage: $0 --time <seconds> [--drop-caches]"; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1"
      echo "Usage: $0 --time <seconds> [--drop-caches]"; exit 1 ;;
  esac
done

if ! [[ "$IDLE_TIME" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --time must be a non-negative integer."; exit 1
fi

if [[ "$DROP_CACHES" -eq 1 ]]; then
  if [[ "$(id -u)" -eq 0 ]]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && echo "[IDLE] dropped page cache" \
      || echo "[IDLE] drop_caches not available; continuing"
  else
    echo "[IDLE] --drop-caches requested but not root; skipping (sync only)"
    sync
  fi
fi

echo "[IDLE] long baseline: sleeping for ${IDLE_TIME}s..."
sleep "$IDLE_TIME"
echo "[IDLE] Done."
