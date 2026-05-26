#!/usr/bin/env bash
# QEMU/libvirt producer (qemu:///system) using qemu-monitor-command pmemsave.
# High-level goal: generate a *sequence* of flat RAW physical-memory images
# for a running libvirt VM, queuing prev/curr pairs for the existing consumer
# while keeping the VM paused only around the pmemsave window.
#
# Capture loop for each iteration:
# - Enforce backpressure on the queue (do not create new dumps if consumer is behind).
# - Pause the VM via `virsh -c qemu:///system suspend` and wait until domstate == paused.
# - Ask QEMU (via `virsh qemu-monitor-command`) to run pmemsave(0, ramSizeBytes, newImage)
#   into a libvirt-owned directory (typically /var/lib/libvirt/qemu/dump).
# - Optionally run `sudo chown <user>:<group> newImage` so the user-owned consumer
#   and analysis tools can read the dump without changing libvirt/SELinux/AppArmor.
# - Enqueue a `{ prev, curr, output }` JSON job pointing to the old/new dump paths.
# - Resume the VM via `virsh ... resume` and wait until domstate == running.
# - Sleep for intervalMsec, then repeat.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$ROOT/config_qemu.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[PRODUCER-PMEM] config not found: $CONFIG (set CONFIG= or copy config_qemu.json.example to config_qemu.json)"
  exit 1
fi

domain=$(jq -r '.domain' "$CONFIG")                 # libvirt domain name (e.g. "Kali Jeries")
imageDir=$(jq -r '.imageDir' "$CONFIG")             # where pmemsave writes RAW dumps
outputDir=$(jq -r '.outputDir' "$CONFIG")           # directory passed to Rust delta
intervalMsec=$(jq -r '.intervalMsec' "$CONFIG")     # capture interval in milliseconds
qPath=$(jq -r '.queueDir' "$CONFIG")                # root of {pending,processing,done,failed}
maxPending=$(jq -r '.backpressure.maxPendingJobs // 20' "$CONFIG")
sleepOnBackpressure=$(jq -r '.backpressure.sleepOnBackpressureSeconds // 1' "$CONFIG")
timeoutSeconds=$(jq -r '.vmStatePolling.timeoutSeconds // 30' "$CONFIG")
pollIntervalMs=$(jq -r '.vmStatePolling.pollIntervalMs // 200' "$CONFIG")

ramSizeMb=$(jq -r '.ramSizeMb // 0' "$CONFIG")      # guest RAM size in MiB (e.g. 2048)
if [[ -z "$ramSizeMb" || "$ramSizeMb" == "null" || "$ramSizeMb" -le 0 ]]; then
  echo "[PRODUCER-PMEM] ERROR: ramSizeMb required in config (guest RAM size in MiB, e.g. 2048)"
  exit 1
fi
ramSizeBytes=$(( ramSizeMb * 1024 * 1024 ))         # pmemsave size (must match RAM exactly)

# Optional chown of the freshly created dump so the consumer (running as user)
# can read it without changing libvirt policies. Leave empty to disable.
chownUser=$(jq -r '.chownUser // ""' "$CONFIG" 2>/dev/null || echo "")   # e.g. "jeries"
chownGroup=$(jq -r '.chownGroup // ""' "$CONFIG" 2>/dev/null || echo "") # e.g. "jeries"

qPending="$qPath/pending"
qProcessing="$qPath/processing"
mkdir -p "$qPending" "$qProcessing" "$imageDir" "$outputDir"
VM_STATE_FILE="$qPath/vm_state.txt"
echo "running" > "$VM_STATE_FILE"

imageFilePrefix="memory_dump"
prevImage=""

# Resolve our own script directory so we can launch the B+3.1 APF helper
# regardless of how the producer was invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# B+3.1 streaming APF helper: per-pair sequence counter and env-var-gated.
# When TIMING_APF_STREAM is set, after each pmemsave (with a valid prev),
# launch plan02_apf_helper.py in the background. The helper computes
# active-page-fraction between prev+curr, appends to the shared JSONL,
# writes an ack file, and deletes prev. The orchestrator's cell-end
# barrier waits for all ack files before computing F1 / CV.
APF_PAIR_SEQ=0

# Optional timing instrumentation. When TIMING_JSONL_PATH is set, the producer
# emits one JSON line per snapshot with t0..t5 host-side timestamps. See
# VM_sampler/VM_Capture_QEMU/docs/tuning_plans/01_instrumentation_logging_plan.md.
TIMING_JSONL_PATH="${TIMING_JSONL_PATH:-}"
ts_ns() { date +%s.%N; }
SNAP_SEQ=0
if [[ -n "$TIMING_JSONL_PATH" ]]; then
  mkdir -p "$(dirname "$TIMING_JSONL_PATH")"
  : > "$TIMING_JSONL_PATH"
  echo "[PRODUCER-PMEM] timing JSONL: $TIMING_JSONL_PATH"
fi
emit_timing() {
  # args: t0 t1 t2 t3 t4 t5 backpressure_flag backpressure_wait_ms pending_count image_path bytes
  [[ -z "$TIMING_JSONL_PATH" ]] && return 0
  printf '{"seq":%d,"t0_before_suspend":%s,"t1_after_suspend":%s,"t2_pmemsave_start":%s,"t3_pmemsave_end":%s,"t4_before_resume":%s,"t5_after_resume":%s,"backpressure_event":%s,"backpressure_wait_ms":%s,"queue_depth":%s,"image_path":"%s","dump_bytes":%s,"interval_msec":%s,"ram_mb":%s}\n' \
    "$SNAP_SEQ" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "$intervalMsec" "$ramSizeMb" \
    >> "$TIMING_JSONL_PATH"
  SNAP_SEQ=$((SNAP_SEQ + 1))
}

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
    if [[ -n "$TIMING_JSONL_PATH" ]]; then
      __bp_t=$(ts_ns)
      printf '{"seq":-1,"backpressure_event":true,"backpressure_wait_ms":%s,"queue_depth":%s,"t_host":%s}\n' \
        "$((sleepOnBackpressure * 1000))" "$total" "$__bp_t" >> "$TIMING_JSONL_PATH"
    fi
    sleep "$sleepOnBackpressure"
    continue
  fi

  timestamp=$(date +%Y%m%d%H%M%S%3N)
  newImage="$imageDir/${imageFilePrefix}-${timestamp}.raw"

  echo "paused" > "$VM_STATE_FILE"
  echo "[PRODUCER-PMEM] Suspending VM via virsh ..."
  __t0=$(ts_ns)
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
  __t1=$(ts_ns)

  echo "[PRODUCER-PMEM] Dumping RAW memory to $newImage using pmemsave (size=$ramSizeBytes bytes) ..."
  pmem_cmd=$(printf '{"execute":"pmemsave","arguments":{"val":0,"size":%d,"filename":"%s"}}' "$ramSizeBytes" "$newImage")
  __t2=$(ts_ns)
  if ! virsh -c qemu:///system qemu-monitor-command "$domain" --cmd "$pmem_cmd" 2>/dev/null; then
    echo "[PRODUCER-PMEM] pmemsave failed, resuming VM"
    virsh -c qemu:///system resume "$domain" 2>/dev/null || true
    sleep 0.5
    continue
  fi
  __t3=$(ts_ns)

  # Give QEMU a brief moment to flush the dump file. Set TIMING_NO_FLUSH=1
  # to skip this sleep (see exp2c flush-sensitivity test).
  if [[ -z "${TIMING_NO_FLUSH:-}" ]]; then
    sleep 0.5
  fi

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
    if [[ -n "${TIMING_APF_STREAM:-}" ]]; then
      # B+3.1 (Δ-1): spawn async APF helper. Helper writes one line to
      # ${TIMING_APF_JSONL}, an ack file to ${TIMING_APF_ACK_DIR}/seq_NNN.apf_done,
      # and deletes $prevImage. Producer continues immediately.
      #
      # Day-14 fix · lower I/O + CPU priority so the helper does not
      # compete with the next pmemsave for disk bandwidth. ionice -c 3
      # = idle class (only runs when nothing else needs the disk).
      # nice -n 19 = lowest CPU priority. ionice may be absent on some
      # hosts · fall back to nice alone.
      if command -v ionice >/dev/null 2>&1; then
        APF_PRIO="ionice -c 3 nice -n 19"
      else
        APF_PRIO="nice -n 19"
      fi
      $APF_PRIO python3 "${SCRIPT_DIR}/plan02_apf_helper.py" \
        --prev "$prevImage" \
        --curr "$newImage" \
        --apf-jsonl "${TIMING_APF_JSONL}" \
        --ack-dir "${TIMING_APF_ACK_DIR}" \
        --seq "$APF_PAIR_SEQ" \
        >> "${TIMING_APF_HELPER_LOG:-/dev/null}" 2>&1 &
      APF_PAIR_SEQ=$((APF_PAIR_SEQ + 1))
    elif [[ -n "${TIMING_SELF_CLEAN:-}" ]]; then
      # Producer-only timing mode: no consumer is running to drain the queue
      # and unlink prev dump. Delete prev ourselves to prevent disk pressure
      # from accumulating across the pass (mechanism vi).
      sudo rm -f "$prevImage" 2>/dev/null || rm -f "$prevImage" 2>/dev/null || true
    else
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
  fi

  prevImage="$newImage"

  echo "[PRODUCER-PMEM] Resuming VM via virsh ..."
  __t4=$(ts_ns)
  virsh -c qemu:///system resume "$domain" 2>/dev/null || true
  echo "running" > "$VM_STATE_FILE"
  if ! wait_state "running"; then
    echo "[PRODUCER-PMEM] Resume may have failed; continuing anyway"
  fi
  __t5=$(ts_ns)

  # Emit one JSONL timing record for this snapshot (no-op if TIMING_JSONL_PATH unset).
  emit_timing "$__t0" "$__t1" "$__t2" "$__t3" "$__t4" "$__t5" "false" "0" "$total" "$newImage" "$ramSizeBytes"

  if command -v bc &>/dev/null; then
    sleep "$(echo "scale=3; $intervalMsec/1000" | bc)"
  else
    sleep "$(( intervalMsec / 1000 ))"
  fi
done

