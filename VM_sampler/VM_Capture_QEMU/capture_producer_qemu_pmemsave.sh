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

# AppArmor workaround (2026-08-27): this VM's libvirt-generated profile
# (/etc/apparmor.d/libvirt/libvirt-<uuid>.files) does not list the configured
# imageDir, so QEMU's own open() of a dump there is denied and pmemsave fails
# with "Could not open '...': Permission denied". The SAME profile grants rwk on
# the per-domain state dir, so target that instead when useDomainDir is set.
# libvirt recreates that dir with a NEW id on every VM start, so resolve it per
# run, never from config. Flag: config .useDomainDir, or USE_DOMAIN_DIR=1 env.
# Remove this block once the dump dir is granted in
# /etc/apparmor.d/local/abstractions/libvirt-qemu. The rule must name the
# RESOLVED path -- imageDir (/var/lib/libvirt/qemu/dump) is a symlink to
# /project/dump and AppArmor matches the real path -- so the line is:
#   "/project/dump/{,**}" rw,
# then reload the domain profile with apparmor_parser -r.
useDomainDir=$(jq -r '.useDomainDir // false' "$CONFIG" 2>/dev/null || echo "false")
if [[ "${USE_DOMAIN_DIR:-}" == "1" || "$useDomainDir" == "true" ]]; then
  domId=$(virsh -c qemu:///system domid "$domain" 2>/dev/null | tr -d '[:space:]')
  if [[ ! "$domId" =~ ^[0-9]+$ ]]; then
    echo "[PRODUCER-PMEM] ERROR: useDomainDir set but domid for '$domain' is '$domId' (VM not running?)"
    exit 1
  fi
  imageDir="/var/lib/libvirt/qemu/domain-${domId}-${domain}"
  if [[ ! -d "$imageDir" ]]; then
    echo "[PRODUCER-PMEM] ERROR: resolved domain state dir does not exist: $imageDir"
    exit 1
  fi
  echo "[PRODUCER-PMEM] useDomainDir -> imageDir=$imageDir (AppArmor workaround)"
fi

qPending="$qPath/pending"
qProcessing="$qPath/processing"
mkdir -p "$qPending" "$qProcessing" "$imageDir" "$outputDir"
VM_STATE_FILE="$qPath/vm_state.txt"
echo "running" > "$VM_STATE_FILE"

# Live capture status the console reads. Atomic tmp+mv write of
# {captured, state, workload, updated} to a fixed file in the queue dir.
# workload comes from the retention env the orchestrator passes to producer and
# consumer alike. captured counts confirmed pmemsave dumps this trace.
STATUS_FILE="$qPath/capture_status.json"
CONTROL_FILE="$qPath/capture_control.json"
workload="${ZSTD_WORKLOAD:-${BORG_WORKLOAD:-}}"
captured=0
write_status() {
  local state="$1"
  local tmp="$STATUS_FILE.tmp.$$"
  printf '{"captured":%d,"state":"%s","workload":%s,"updated":"%s"}\n' \
    "$captured" "$state" "$(printf '%s' "$workload" | jq -R .)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmp" 2>/dev/null \
    && mv -f "$tmp" "$STATUS_FILE" 2>/dev/null || true
}
read_control() {
  # Current operator command (run|pause|skip); default run. Quiet if the file is
  # absent or mid-rewrite. Read only between snapshots, so control never cuts a
  # dump in flight.
  jq -r '.command // "run"' "$CONTROL_FILE" 2>/dev/null || echo "run"
}
write_status "running"

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

# Plan 06 (additive, flag-gated): host-side disk-I/O channel. When TIMING_DISKIO is
# set, after each pmemsave (VM paused -> cadence-safe) we read the guest block
# device's cumulative rd/wr byte counters via domblkstat and append one line per
# snapshot to TIMING_DISKIO_JSONL, keyed by the same SNAP_SEQ. Default UNSET ->
# this is a no-op and behaviour is byte-identical to the delta/apf/apf_queue paths.
TIMING_DISKIO="${TIMING_DISKIO:-}"
TIMING_DISKIO_JSONL="${TIMING_DISKIO_JSONL:-}"
DISKIO_DEV="${TIMING_DISKIO_DEV:-vda}"
DISKIO_STRIDE="${TIMING_DISKIO_STRIDE:-1}"   # poll every Nth snapshot (1 = every).
DISKIO_SEQ=0    # own counter: SNAP_SEQ only advances when TIMING_JSONL_PATH is set.
DISKIO_CALLS=0
if [[ -n "$TIMING_DISKIO" && -n "$TIMING_DISKIO_JSONL" ]]; then
  mkdir -p "$(dirname "$TIMING_DISKIO_JSONL")"
  : > "$TIMING_DISKIO_JSONL"
  echo "[PRODUCER-PMEM] diskio JSONL: $TIMING_DISKIO_JSONL (dev=$DISKIO_DEV, stride=$DISKIO_STRIDE)"
fi
diskio_emit() {
  [[ -z "$TIMING_DISKIO" || -z "$TIMING_DISKIO_JSONL" ]] && return 0
  # Stride: domblkstat is the costly part; sample every Nth snapshot. Cumulative
  # counters mean the per-cell rate is unchanged, only fewer (coarser) points.
  if (( DISKIO_CALLS % DISKIO_STRIDE == 0 )); then
    local stat rd wr
    stat=$(virsh -c qemu:///system domblkstat "$domain" "$DISKIO_DEV" 2>/dev/null || true)
    rd=$(awk '$2=="rd_bytes"{print $3}' <<<"$stat")
    wr=$(awk '$2=="wr_bytes"{print $3}' <<<"$stat")
    [[ -z "$rd" ]] && rd=-1
    [[ -z "$wr" ]] && wr=-1
    printf '{"seq":%d,"t_emit_epoch":%s,"rd_bytes":%s,"wr_bytes":%s}\n' \
      "$DISKIO_SEQ" "$(ts_ns)" "$rd" "$wr" >> "$TIMING_DISKIO_JSONL"
    DISKIO_SEQ=$((DISKIO_SEQ + 1))
  fi
  DISKIO_CALLS=$((DISKIO_CALLS + 1))
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

# Suspend the domain and wait until it reports paused. Returns 0 on success, 1
# on failure (caller resumes + retries). Must never be called on an
# already-paused domain -- virsh rejects a double suspend -- which the loop's
# bp_paused guard guarantees.
suspend_vm() {
  echo "paused" > "$VM_STATE_FILE"
  if ! virsh -c qemu:///system suspend "$domain" 2>/dev/null; then
    echo "[PRODUCER-PMEM] WARNING: virsh suspend failed"
    return 1
  fi
  wait_state "paused"
}

echo "[PRODUCER-PMEM] Starting (domain=$domain, ramSizeMb=$ramSizeMb, interval=${intervalMsec}ms, imageDir=$imageDir)"

while true; do
  # Backpressure gate. Hold the VM SUSPENDED for the entire wait, not running.
  # The guest clock (hence the workload's --duration budget) is frozen while the
  # domain is paused, so waiting on a slow consumer no longer burns the
  # workload's time without capturing. This keeps snapshot COUNT and sampling
  # INTERVAL uniform across workloads regardless of memory churn -- heavy
  # workloads used to starve to ~75 ragged-interval snapshots, which also
  # violated the spectral metrics' uniform-sampling assumption. bp_paused == 1
  # means WE already suspended for the wait, so the snapshot block below must not
  # suspend again (virsh rejects a double suspend).
  bp_paused=0
  while true; do
    # Operator pause: hold the VM suspended (guest frozen, --duration budget
    # preserved) until the command is no longer "pause". Same suspend as
    # backpressure, so bp_paused carries into the snapshot block's no-double-
    # suspend guard and the next snapshot resumes normally.
    ctl=$(read_control)
    if [[ "$ctl" == "pause" ]]; then
      if (( bp_paused == 0 )); then
        echo "[PRODUCER-PMEM] Pause requested; suspending VM until resumed"
        if suspend_vm; then
          bp_paused=1
        else
          virsh -c qemu:///system resume "$domain" 2>/dev/null || true
          sleep 0.5
          continue
        fi
      fi
      write_status "paused"
      sleep 0.5
      continue
    fi
    pendingCount=$(find "$qPending" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    processingCount=$(find "$qProcessing" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    total=$((pendingCount + processingCount))
    ((total < maxPending)) && break
    if (( bp_paused == 0 )); then
      echo "[PRODUCER-PMEM] Backpressure: queue $total >= $maxPending; suspending VM until it drains"
      if suspend_vm; then
        bp_paused=1
        write_status "backpressure"
      else
        virsh -c qemu:///system resume "$domain" 2>/dev/null || true
        sleep 0.5
        continue
      fi
    fi
    if [[ -n "$TIMING_JSONL_PATH" ]]; then
      __bp_t=$(ts_ns)
      printf '{"seq":-1,"backpressure_event":true,"backpressure_wait_ms":%s,"queue_depth":%s,"t_host":%s}\n' \
        "$((sleepOnBackpressure * 1000))" "$total" "$__bp_t" >> "$TIMING_JSONL_PATH"
    fi
    sleep "$sleepOnBackpressure"
  done

  timestamp=$(date +%Y%m%d%H%M%S%3N)
  newImage="$imageDir/${imageFilePrefix}-${timestamp}.raw"

  __t0=$(ts_ns)
  if (( bp_paused == 0 )); then
    echo "[PRODUCER-PMEM] Suspending VM via virsh ..."
    if ! suspend_vm; then
      echo "[PRODUCER-PMEM] WARNING: virsh suspend failed, retrying in 500ms"
      virsh -c qemu:///system resume "$domain" 2>/dev/null || true
      sleep 0.5
      continue
    fi
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

  # Plan 06 disk-I/O channel (flag-gated; VM still paused here). No-op when unset.
  diskio_emit

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
  captured=$((captured + 1))
  write_status "running"

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

