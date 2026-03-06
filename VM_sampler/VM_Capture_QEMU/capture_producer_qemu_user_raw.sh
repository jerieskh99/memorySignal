#!/usr/bin/env bash
# User-space raw memory producer: pause VM via QEMU monitor, pmemsave (flat raw), resume, enqueue job.
# Requires QEMU run by the same user (e.g. -monitor unix:/tmp/qemu-monitor.sock).
# No virsh, no libvirt system config, no sudo. Output: flat raw physical memory (.raw).
# Uses same queue/job format as capture_producer_qemu.sh; same consumer (capture_consumer_qemu.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$ROOT/config_qemu.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[PRODUCER-RAW] config not found: $CONFIG (set CONFIG= or copy config_qemu.json.example to config_qemu.json)"
  exit 1
fi

qemuMonitorSocket=$(jq -r '.qemuMonitorSocket // ""' "$CONFIG")
if [[ -z "$qemuMonitorSocket" || "$qemuMonitorSocket" == "null" ]]; then
  echo "[PRODUCER-RAW] ERROR: qemuMonitorSocket required in config (path to QEMU monitor Unix socket)"
  exit 1
fi

ramSizeMb=$(jq -r '.ramSizeMb // 0' "$CONFIG")
if [[ -z "$ramSizeMb" || "$ramSizeMb" == "null" || "$ramSizeMb" -le 0 ]]; then
  echo "[PRODUCER-RAW] ERROR: ramSizeMb required in config (guest RAM size in MiB, e.g. 4096)"
  exit 1
fi

imageDir=$(jq -r '.imageDir' "$CONFIG")
outputDir=$(jq -r '.outputDir' "$CONFIG")
intervalMsec=$(jq -r '.intervalMsec' "$CONFIG")
qPath=$(jq -r '.queueDir' "$CONFIG")
maxPending=$(jq -r '.backpressure.maxPendingJobs // 20' "$CONFIG")
sleepOnBackpressure=$(jq -r '.backpressure.sleepOnBackpressureSeconds // 1' "$CONFIG")
timeoutSeconds=$(jq -r '.vmStatePolling.timeoutSeconds // 30' "$CONFIG")
pollIntervalMs=$(jq -r '.vmStatePolling.pollIntervalMs // 200' "$CONFIG")

# Guest RAM size in bytes for pmemsave
ramSizeBytes=$(( ramSizeMb * 1024 * 1024 ))

qPending="$qPath/pending"
qProcessing="$qPath/processing"
mkdir -p "$qPending" "$qProcessing" "$imageDir" "$outputDir"

imageFilePrefix="memory_dump"
prevImage=""

# Send command to QEMU monitor and get response
qemu_monitor_cmd() {
  local cmd="$1"
  if command -v socat &>/dev/null; then
    echo "$cmd" | socat - UNIX-CONNECT:"$qemuMonitorSocket" 2>/dev/null || return 1
  elif command -v nc &>/dev/null; then
    echo "$cmd" | nc -U "$qemuMonitorSocket" 2>/dev/null || return 1
  else
    echo "[PRODUCER-RAW] ERROR: Neither socat nor nc found. Install socat or use nc."
    return 1
  fi
}

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
  echo "[PRODUCER-RAW] Timeout waiting for VM state $want (current: $state)"
  return 1
}

if [[ ! -S "$qemuMonitorSocket" ]]; then
  echo "[PRODUCER-RAW] ERROR: QEMU monitor socket not found: $qemuMonitorSocket"
  echo "[PRODUCER-RAW] Start QEMU with: -monitor unix:$qemuMonitorSocket,server,nowait"
  exit 1
fi

echo "[PRODUCER-RAW] Starting (monitor=$qemuMonitorSocket, ramSizeMb=$ramSizeMb, interval=${intervalMsec}ms, output=.raw)"

while true; do
  pendingCount=$(find "$qPending" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  processingCount=$(find "$qProcessing" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  total=$((pendingCount + processingCount))
  if ((total >= maxPending)); then
    echo "[PRODUCER-RAW] Backpressure: queue size $total >= $maxPending, sleeping ${sleepOnBackpressure}s"
    sleep "$sleepOnBackpressure"
    continue
  fi

  timestamp=$(date +%Y%m%d%H%M%S%3N)
  newImage="$imageDir/${imageFilePrefix}-${timestamp}.raw"

  echo "[PRODUCER-RAW] Pausing VM..."
  if ! qemu_monitor_cmd "stop" 2>/dev/null; then
    echo "[PRODUCER-RAW] WARNING: stop failed, retrying in 500ms"
    sleep 0.5
    continue
  fi

  if ! wait_state "paused"; then
    qemu_monitor_cmd "cont" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  echo "[PRODUCER-RAW] Dumping raw memory to $newImage (pmemsave 0 $ramSizeBytes)..."
  # pmemsave <phys_addr> <size> <filename> -> flat raw guest physical memory
  if ! qemu_monitor_cmd "pmemsave 0 $ramSizeBytes $newImage" 2>/dev/null; then
    echo "[PRODUCER-RAW] pmemsave failed, resuming VM"
    qemu_monitor_cmd "cont" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  sleep 0.5
  if [[ ! -f "$newImage" ]]; then
    echo "[PRODUCER-RAW] Dump file not created: $newImage"
    qemu_monitor_cmd "cont" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  actualSize=$(stat -c%s "$newImage" 2>/dev/null || stat -f%z "$newImage" 2>/dev/null || echo 0)
  if [[ "$actualSize" -ne "$ramSizeBytes" ]]; then
    echo "[PRODUCER-RAW] WARNING: dump size $actualSize != expected $ramSizeBytes"
  fi

  echo "[PRODUCER-RAW] Memory dump OK: $newImage"

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
    echo "[PRODUCER-RAW] Enqueued job $jobId"
  fi

  prevImage="$newImage"

  echo "[PRODUCER-RAW] Resuming VM ..."
  qemu_monitor_cmd "cont" 2>/dev/null || true
  if ! wait_state "running"; then
    echo "[PRODUCER-RAW] Resume may have failed; continuing anyway"
  fi

  if command -v bc &>/dev/null; then
    sleep "$(echo "scale=3; $intervalMsec/1000" | bc)"
  else
    sleep "$(( intervalMsec / 1000 ))"
  fi
done
