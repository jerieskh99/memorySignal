#!/usr/bin/env bash
# Simple idle workload: sleep for N seconds.
# Usage:
#   ./run_idle.sh --time 30

set -euo pipefail

IDLE_TIME=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --time)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --time requires a value (seconds)."
        exit 1
      fi
      IDLE_TIME="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --time <seconds>"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      echo "Usage: $0 --time <seconds>"
      exit 1
      ;;
  esac
done

if ! [[ "$IDLE_TIME" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --time must be a non-negative integer."
  exit 1
fi

echo "[IDLE] Sleeping for ${IDLE_TIME}s..."
sleep "$IDLE_TIME"
echo "[IDLE] Done."

