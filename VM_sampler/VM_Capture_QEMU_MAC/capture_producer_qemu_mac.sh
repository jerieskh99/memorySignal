#!/usr/bin/env bash
# QEMU/macOS producer: pause VM via QEMU monitor, dump guest memory, resume, enqueue job.
# Uses QEMU monitor socket (not virsh/libvirt). Snapshots are kept on disk; consumer runs delta + streaming and deletes them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$ROOT/config_qemu_mac.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[PRODUCER] config not found: $CONFIG (set CONFIG= or copy config_qemu_mac.json.example to config_qemu_mac.json)"
  exit 1
fi

qemuMonitorSocket=$(jq -r '.qemuMonitorSocket' "$CONFIG")
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

# Send command to QEMU monitor socket and get response
qemu_monitor_cmd() {
  local cmd="$1"
  if command -v socat &>/dev/null; then
    echo "$cmd" | socat - UNIX-CONNECT:"$qemuMonitorSocket" 2>/dev/null || return 1
  elif command -v nc &>/dev/null; then
    echo "$cmd" | nc -U "$qemuMonitorSocket" 2>/dev/null || return 1
  else
    echo "[PRODUCER] ERROR: Neither socat nor nc found. Install one: brew install socat or use nc"
    return 1
  fi
}

# Get VM status from QEMU monitor
get_vm_status() {
  local status
  status=$(qemu_monitor_cmd "info status" 2>/dev/null | grep -i "VM status:" | awk '{print $3}' | tr -d '\r\n' || echo "")
  echo "$status"
}

wait_state() {
  local want="$1"
  local deadline=$((SECONDS + timeoutSeconds))
  while ((SECONDS < deadline)); do
    local state
    state=$(get_vm_status)
    # Normalize: "paused" or "paused (suspended)" -> "paused", "running" -> "running"
    if [[ "$state" == *"paused"* ]]; then
      state="paused"
    elif [[ "$state" == *"running"* ]]; then
      state="running"
    fi
    if [[ "$state" == "$want" ]]; then
      return 0
    fi
    sleep "$(echo "scale=3; $pollIntervalMs/1000" | bc 2>/dev/null || echo "0.2")"
  done
  echo "[PRODUCER] Timeout waiting for VM state $want (current: $state)"
  return 1
}

if [[ ! -S "$qemuMonitorSocket" ]]; then
  echo "[PRODUCER] ERROR: QEMU monitor socket not found: $qemuMonitorSocket"
  echo "[PRODUCER] Make sure QEMU is running with: -monitor unix:$qemuMonitorSocket,server,nowait"
  exit 1
fi

echo "[PRODUCER] Starting (monitor=$qemuMonitorSocket, interval=${intervalMsec}ms)"

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

  # Pause VM
  echo "[PRODUCER] Pausing VM..."
  if ! qemu_monitor_cmd "stop" 2>/dev/null; then
    echo "[PRODUCER] WARNING: QEMU monitor stop failed, retrying in 500ms"
    sleep 0.5
    continue
  fi

  if ! wait_state "paused"; then
    qemu_monitor_cmd "cont" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  # Dump memory
  echo "[PRODUCER] Dumping memory to $newImage ..."
  # QEMU monitor dump-guest-memory format: elf, kdump-zlib, kdump-lzo, kdump-snappy
  # We use "elf" for compatibility with live_delta_calc
  if ! qemu_monitor_cmd "dump-guest-memory elf $newImage" 2>/dev/null; then
    echo "[PRODUCER] dump-guest-memory failed, resuming VM"
    qemu_monitor_cmd "cont" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  # Wait a moment for dump to complete
  sleep 0.5

  if [[ ! -f "$newImage" ]]; then
    echo "[PRODUCER] Dump file not created: $newImage"
    qemu_monitor_cmd "cont" 2>/dev/null || true
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

  # Resume VM
  echo "[PRODUCER] Resuming VM ..."
  qemu_monitor_cmd "cont" 2>/dev/null || true
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
