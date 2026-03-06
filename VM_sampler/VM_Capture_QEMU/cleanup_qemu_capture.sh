#!/usr/bin/env bash
# Helper: stop QEMU capture producer/consumer and clean all transient artefacts.
# Intended for DEVELOPMENT / TESTING only.
#
# What it does:
# 1) Finds any running capture producer / consumer processes and kills them.
# 2) Clears all job queues under ~/memory_traces/queue_dir/{pending,processing,done,failed}.
# 3) Deletes RAW memory dumps matching memory_dump* under /var/lib/libvirt/qemu/dump (sudo).

set -euo pipefail

QUEUE_ROOT="${QUEUE_ROOT:-$HOME/memory_traces/queue_dir}"
DUMP_DIR="${DUMP_DIR:-/var/lib/libvirt/qemu/dump}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/memory_traces/output_dir}"

echo "[CLEANUP] Queue root : $QUEUE_ROOT"
echo "[CLEANUP] Dump dir   : $DUMP_DIR"
echo "[CLEANUP] Output root: $OUTPUT_ROOT"

echo "[CLEANUP] Killing capture producer/consumer processes (if any)..."
# Kill any process whose command line contains our capture scripts.
pkill -f capture_producer_qemu_pmemsave.sh 2>/dev/null || true
pkill -f capture_producer_qemu_user_raw.sh 2>/dev/null || true
pkill -f capture_producer_qemu.sh 2>/dev/null || true
pkill -f capture_consumer_qemu.sh 2>/dev/null || true

echo "[CLEANUP] Cleaning queue directories..."
for sub in failed processing done pending; do
  dir="$QUEUE_ROOT/$sub"
  if [[ -d "$dir" ]]; then
    rm -f "$dir"/* 2>/dev/null || true
    echo "[CLEANUP] Cleared $dir"
  fi
done

echo "[CLEANUP] Deleting RAW dumps in $DUMP_DIR (sudo rm memory_dump*)..."
if [[ -d "$DUMP_DIR" ]]; then
  sudo rm -f "$DUMP_DIR"/memory_dump* 2>/dev/null || true
else
  echo "[CLEANUP] Dump dir does not exist: $DUMP_DIR (nothing to delete)"
fi

echo "[CLEANUP] Deleting DELTA files in $OUTPUT_ROOT (cosine/hamming)..."
if [[ -d "$OUTPUT_ROOT" ]]; then
  sudo rm -f "$OUTPUT_ROOT"/cosine/* 2>/dev/null || true
  sudo rm -f "$OUTPUT_ROOT"/hamming/* 2>/dev/null || true
else
  echo "[CLEANUP] Output dir does not exist: $OUTPUT_ROOT (nothing to delete)"
fi

echo "[CLEANUP] Done."

