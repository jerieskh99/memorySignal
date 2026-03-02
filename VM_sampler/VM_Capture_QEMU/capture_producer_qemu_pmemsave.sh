#!/usr/bin/env bash
# QEMU/libvirt producer (qemu:///system) using qemu-monitor-command pmemsave:
# 1) virsh suspend (pause VM)
# 2) virsh qemu-monitor-command pmemsave -> /var/lib/libvirt/qemu/dump/*.raw (or imageDir)
# 3) optional sudo chown to make dump readable by user
# 4) enqueue { prev, curr, output } job for the existing consumer.
#
# This is the \"system\" variant of the raw pmemsave producer, intended for
# VMs managed under qemu:///system where QEMU runs as libvirt-qemu and is
# allowed to write into /var/lib/libvirt/qemu/dump.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$ROOT/config_qemu.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[PRODUCER-PMEM] config not found: $CONFIG (set CONFIG= or copy config_qemu.json.example to config_qemu.json)"
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

ramSizeMb=$(jq -r '.ramSizeMb // 0' "$CONFIG")
if [[ -z "$ramSizeMb" || "$ramSizeMb" == "null" || "$ramSizeMb" -le 0 ]]; then
  echo "[PRODUCER-PMEM] ERROR: ramSizeMb required in config (guest RAM size in MiB, e.g. 2048)"
  exit 1
fi
ramSizeBytes=$(( ramSizeMb * 1024 * 1024 ))

# Optional chown of the freshly created dump so the consumer (running as user)
# can read it without changing libvirt policies. Leave empty to disable.
chownUser=$(jq -r '.chownUser // ""' "$CONFIG" 2>/dev/null || echo "")
chownGroup=$(jq -r '.chownGroup // ""' "$CONFIG" 2>/dev/null || echo "")

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
    state=$(virsh -c qemu:///system domstate "$domain" 2>/dev/null || true)
    if [[ "$state" == "$want" ]]; then
      return 0
    fi
    sleep "$(echo "scale=3; $pollIntervalMs/1000" | bc 2>/dev/null || echo "0.2")"
  done
  echo "[PRODUCER-PMEM] Timeout waiting for domain $domain state $want (current: $state)"
  return 1
}

echo "[PRODUCER-PMEM] Starting (domain=$domain, ramSizeMb=$ramSizeMb, interval=${intervalMsec}ms, imageDir=$imageDir)"

while true; do
  # Backpressure
  pendingCount=$(find "$qPending" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  processingCount=$(find "$qProcessing" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  total=$((pendingCount + processingCount))
  if ((total >= maxPending)); then
    echo "[PRODUCER-PMEM] Backpressure: queue size $total >= $maxPending, sleeping ${sleepOnBackpressure}s"
    sleep "$sleepOnBackpressure"
    continue
  fi

  timestamp=$(date +%Y%m%d%H%M%S%3N)
  newImage="$imageDir/${imageFilePrefix}-${timestamp}.raw"

  echo "[PRODUCER-PMEM] Suspending VM via virsh ..."
  if ! virsh -c qemu:///system suspend "$domain" 2>/dev/null; then
    echo "[PRODUCER-PMEM] WARNING: virsh suspend failed, retrying in 500ms"
    sleep 0.5
    continue
  fi

  if ! wait_state "paused"; then
    virsh -c qemu:///system resume "$domain" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  echo "[PRODUCER-PMEM] Dumping RAW memory to $newImage using pmemsave (size=$ramSizeBytes bytes) ..."
  pmem_cmd=$(printf '{"execute":"pmemsave","arguments":{"val":0,"size":%d,"filename":"%s"}}' "$ramSizeBytes" "$newImage")
  if ! virsh -c qemu:///system qemu-monitor-command "$domain" --cmd "$pmem_cmd" 2>/dev/null; then
    echo "[PRODUCER-PMEM] pmemsave failed, resuming VM"
    virsh -c qemu:///system resume "$domain" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  # Give QEMU a brief moment to flush the dump file
  sleep 0.5

  if [[ ! -f "$newImage" ]]; then
    echo "[PRODUCER-PMEM] Dump file not created: $newImage"
    virsh -c qemu:///system resume "$domain" 2>/dev/null || true
    sleep 0.5
    continue
  fi

  # Optional chown so consumer can read the dump
  if [[ -n "$chownUser" && "$chownUser" != "null" && -n "$chownGroup" && "$chownGroup" != "null" ]]; then
    echo "[PRODUCER-PMEM] Running sudo chown $chownUser:$chownGroup $newImage"
    if ! sudo chown "$chownUser:$chownGroup" "$newImage"; then
      echo "[PRODUCER-PMEM] WARNING: sudo chown failed; consumer may not be able to read $newImage"
    fi
  fi

  actualSize=$(stat -c%s "$newImage" 2>/dev/null || stat -f%z "$newImage" 2>/dev/null || echo 0)
  if [[ "$actualSize" -ne "$ramSizeBytes" ]]; then
    echo "[PRODUCER-PMEM] WARNING: dump size $actualSize != expected $ramSizeBytes"
  fi

  echo "[PRODUCER-PMEM] RAW memory dump OK: $newImage"

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
    echo "[PRODUCER-PMEM] Enqueued job $jobId"
  fi

  prevImage="$newImage"

  echo "[PRODUCER-PMEM] Resuming VM via virsh ..."
  virsh -c qemu:///system resume "$domain" 2>/dev/null || true
  if ! wait_state "running"; then
    echo "[PRODUCER-PMEM] Resume may have failed; continuing anyway"
  fi

  if command -v bc &>/dev/null; then
    sleep "$(echo "scale=3; $intervalMsec/1000" | bc)"
  else
    sleep "$(( intervalMsec / 1000 ))"
  fi
done

