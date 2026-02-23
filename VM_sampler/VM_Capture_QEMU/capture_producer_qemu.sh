#!/usr/bin/env bash
# QEMU/libvirt producer: suspend VM, dump guest memory (virsh dump), resume, enqueue job.
# Snapshots are kept on disk; consumer runs delta + streaming and deletes them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$ROOT/config_qemu.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[PRODUCER] config not found: $CONFIG (set CONFIG= or copy config_qemu.json.example to config_qemu.json)"
  exit 1
fi

domain=$(jq -r '.domain' "$CONFIG")
imageDir=$(jq -r '.imageDir' "$CONFIG")
outputDir=$(jq -r '.outputDir' "$CONFIG")
intervalMsec=$(jq -r '.intervalMsec' "$CONFIG")
qPath=$(jq -r '.queueDir' "$CONFIG")
maxPending=$(jq -r '.backpressure.maxPendingJobs // 20' "$CONFIG")
sleepOnBackpressure=$(jq -r '.backpressure.sleepOnBackpressureSeconds // 1' "$CONFIG")
timeoutSeconds=$(jq -r '.vmStatePolling.timeoutSeconds // 30' "$CONFIG")
pollIntervalMs=$(jq -r '.vmStatePolling.pollIntervalMs // 200' "$CONFIG")

qPending="$qPath/pending"
qProcessing="$qPath/processing"
mkdir -p "$qPending" "$qProcessing" "$imageDir" "$outputDir"

imageFilePrefix="memory_dump"
prevImage=""

wait_state() {
  local want="$1"
  local deadline=$((SECONDS + timeoutSeconds))
  while ((SECONDS < deadline)); do
    local state
    state=$(virsh domstate "$domain" 2>/dev/null || true)
    if [[ "$state" == "$want" ]]; then
      return 0
    fi
    sleep 0.2
  done
  echo "[PRODUCER] Timeout waiting for domain $domain state $want (current: $state)"
  return 1
}

echo "[PRODUCER] Starting (domain=$domain, interval=${intervalMsec}ms)"

while true; do
  # Backpressure
  pendingCount=$(find "$qPending" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  processingCount=$(find "$qProcessing" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  total=$((pendingCount + processingCount))
  if ((total >= maxPending)); then
    echo "[PRODUCER] Backpressure: queue size $total >= $maxPending, sleeping ${sleepOnBackpressure}s"
    sleep "$sleepOnBackpressure"
    continue
  fi

  timestamp=$(date +%Y%m%d%H%M%S%3N)
  newImage="$imageDir/${imageFilePrefix}-${timestamp}.elf"

  if ! virsh suspend "$domain" 2>/dev/null; then
    echo "[PRODUCER] WARNING: virsh suspend failed, retrying in 500ms"
    sleep 0.5
    continue
  fi

  if ! wait_state "paused"; then
    virsh resume "$domain" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  echo "[PRODUCER] Dumping memory to $newImage ..."
  if ! virsh dump "$domain" "$newImage" --memory-only 2>/dev/null; then
    echo "[PRODUCER] virsh dump failed, resuming VM"
    virsh resume "$domain" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  if [[ ! -f "$newImage" ]]; then
    echo "[PRODUCER] Dump file not created: $newImage"
    virsh resume "$domain" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  echo "[PRODUCER] Memory dump OK: $newImage"

  if [[ -n "$prevImage" && -f "$prevImage" ]]; then
    jobId="$timestamp"
    jobTmp="$qPending/${jobId}.json.tmp"
    jobFile="$qPending/${jobId}.json"
    jq -n \
      --arg prev "$prevImage" \
      --arg curr "$newImage" \
      --arg output "$outputDir" \
      '{ prev: $prev, curr: $curr, output: $output }' > "$jobTmp"
    mv "$jobTmp" "$jobFile"
    echo "[PRODUCER] Enqueued job $jobId"
  fi

  prevImage="$newImage"

  echo "[PRODUCER] Resuming VM ..."
  virsh resume "$domain" 2>/dev/null || true
  if ! wait_state "running"; then
    echo "[PRODUCER] Resume may have failed; continuing anyway"
  fi

  # Fractional sleep: e.g. 500ms -> 0.5s
  if command -v bc &>/dev/null; then
    sleep "$(echo "scale=3; $intervalMsec/1000" | bc)"
  else
    sleep "$(( intervalMsec / 1000 ))"
  fi
done
