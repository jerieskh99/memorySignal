#!/usr/bin/env bash
# =============================================================================
# Gate 3 of the dwarf pilot -- the pilot capture (NARROWED after Gate 2).
#
# Gate 2 found guest-physical pages scatter, so address-shape features are out.
# The two surviving, placement-invariant features -- per-page MAGNITUDE
# (l1/l2/hamming) and FOOTPRINT SIZE (count of changed pages) -- are computed at
# EVERY differ speed, so this runs at the fast --speed 2 (live 500ms cadence),
# not the ~39s/pair --speed 0 Gate 2 needed. See docs/dwarf_pilot_design/GATE3_PLAN.md.
#
# ROBUSTNESS (learned from the first run, where the guest network dropped mid-run
# and the script silently produced 14 empty cells over ~20 min):
#   1. FAIL-FAST: probe guest reachability before the run and before every cell;
#      abort with a clear message the instant the guest is unreachable.
#   2. NEVER leave the VM paused: the producer suspends the VM around each
#      pmemsave and has NO exit-trap, so killing it mid-suspend would strand the
#      VM paused. We stop it gracefully AND always `virsh resume` afterwards, and
#      an EXIT/INT/TERM trap resumes the VM even on Ctrl-C or crash.
#   3. Ensure the VM is running (resume if paused) at the start of every cell.
#
# REQUIRED:  SSH_TARGET=user@guest-host   (optional SSH_KEY, SSH_OPTS)
# OPTIONAL:  GUEST_REPO, REPS, CAPTURE_SECS, WORKLOAD_DUR, SSH_CONNECT_TIMEOUT
#
# WHERE TO RUN: on the SERVER (virsh host), not in the guest:
#   cd /project/homes/jeries/memorySignal && git pull
#   SSH_TARGET=kali@<ip> bash VM_sampler/VM_Capture_QEMU/plan09_dwarf_pilot/gate3_pilot_capture.sh
#
# OUTPUT: one CSV per cell/rep under $SCRATCH/cells/<label>_rep<r>/substrate_trajectory.csv
# =============================================================================
set -euo pipefail

REPO="${REPO:-/project/homes/jeries/memorySignal}"
GUEST_REPO="${GUEST_REPO:-/project/homes/jeries/memorySignal}"
QEMU_DIR="$REPO/VM_sampler/VM_Capture_QEMU"
GUEST_BINDIR="$GUEST_REPO/VM_executables_phase2/bin"
BASE_CONFIG="${BASE_CONFIG:-$QEMU_DIR/config_qemu_upc.json}"

: "${SSH_TARGET:?SSH_TARGET (user@guest-host) is required -- not guessed, set it}"
SSH_KEY_OPT=(); [ -n "${SSH_KEY:-}" ] && SSH_KEY_OPT=(-i "$SSH_KEY")
SSH_OPTS="${SSH_OPTS:-}"
SSH_CT="${SSH_CONNECT_TIMEOUT:-8}"

REPS="${REPS:-3}"
CAPTURE_SECS="${CAPTURE_SECS:-70}"   # producer capture window (wall clock)
WORKLOAD_DUR="${WORKLOAD_DUR:-120}"  # guest --duration; > CAPTURE_SECS so it outlives capture
DRAIN_SECS="${DRAIN_SECS:-30}"       # at speed 2 the consumer ~keeps up; small tail drain
BOOT_WAIT="${BOOT_WAIT:-180}"        # max seconds to wait for a just-started guest's SSH to come up
SETTLE_AFTER_BOOT="${SETTLE_AFTER_BOOT:-20}"  # let boot churn subside before capturing
SCRATCH="${SCRATCH:-/var/tmp/gate3_dwarf_pilot}"
GUEST_OUTBASE="${GUEST_OUTBASE:-/tmp/gate3_cells}"

IMAGE_DIR="$SCRATCH/dump"; OUTPUT_DIR="$SCRATCH/output"; QUEUE_DIR="$SCRATCH/queue"
CELLS_DIR="$SCRATCH/cells"; GATE3_CONFIG="$SCRATCH/config_gate3.json"
mkdir -p "$IMAGE_DIR" "$OUTPUT_DIR" "$QUEUE_DIR" "$CELLS_DIR"

[ -f "$BASE_CONFIG" ] || { echo "[gate3] ERROR: base config $BASE_CONFIG missing"; exit 1; }

# Isolated config: speed 2 (magnitude + size survive here), scratch dirs, streaming
# and raw-retention off (not needed for a feature-capture pilot).
jq --arg i "$IMAGE_DIR" --arg o "$OUTPUT_DIR" --arg q "$QUEUE_DIR" \
  '.substrateSpeed = 2 | .imageDir=$i | .outputDir=$o | .queueDir=$q
   | .streaming.enabled=false | .rawRetention.enabled=false' \
  "$BASE_CONFIG" > "$GATE3_CONFIG"
echo "[gate3] scratch config -> $GATE3_CONFIG (speed 2, isolated; production config untouched)"

SUBSTRATE_PROGRAM="$(jq -r '.substrateProgram' "$GATE3_CONFIG")"
[ -x "$SUBSTRATE_PROGRAM" ] || { echo "[gate3] ERROR: differ not built: $SUBSTRATE_PROGRAM (cargo build --release)"; exit 1; }

DOMAIN="$(jq -r '.domain' "$GATE3_CONFIG")"
VIRSH="virsh -c qemu:///system"
PRODUCER_PID=""; CONSUMER_PID=""

# --- safety helpers ---------------------------------------------------------
resume_vm() { $VIRSH resume "$DOMAIN" >/dev/null 2>&1 || true; }   # idempotent

vm_state() { $VIRSH domstate "$DOMAIN" 2>/dev/null || echo unknown; }

# Ensure the domain is RUNNING before a cell. The guest is usually left shut off,
# so this STARTS it when off (not just resume-if-paused), waits for 'running', and
# settles after a fresh boot. Called at the top of every cell.
ensure_vm_up() {
  local st booted=0; st="$(vm_state)"
  case "$st" in
    running) : ;;
    paused)  echo "[gate3] VM paused -- resuming"; resume_vm ;;
    *)       echo "[gate3] VM state='$st' -- starting it ($VIRSH start \"$DOMAIN\")"
             $VIRSH start "$DOMAIN" >/dev/null 2>&1 || { echo "[gate3] ERROR: start failed"; return 1; }
             booted=1 ;;
  esac
  local i=0
  while [ "$(vm_state)" != running ] && (( i < 60 )); do sleep 1; i=$((i+1)); done
  [ "$(vm_state)" = running ] || { echo "[gate3] VM never reached 'running'"; return 1; }
  if [ "$booted" = 1 ]; then echo "[gate3] VM booted; settling ${SETTLE_AFTER_BOOT}s"; sleep "$SETTLE_AFTER_BOOT"; fi
  return 0
}

ssh_guest() { ssh -o ConnectTimeout="$SSH_CT" -o BatchMode=yes "${SSH_KEY_OPT[@]}" $SSH_OPTS "$SSH_TARGET" "$@"; }
guest_reachable() { ssh_guest true >/dev/null 2>&1; }

# Poll until the guest answers SSH, or give up after BOOT_WAIT (covers a fresh boot
# where sshd is not up yet, and distinguishes it from a truly-unreachable guest).
wait_for_reachable() {
  local i=0
  until guest_reachable; do
    (( i >= BOOT_WAIT )) && return 1
    sleep 5; i=$((i+5))
  done
  return 0
}

stop_bg() { # <pid> <name> [wait_s] : graceful TERM, then KILL if still alive
  local pid="$1" name="${2:-proc}" waitn="${3:-8}" i=0
  [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && (( i < waitn )); do sleep 1; i=$((i+1)); done
  kill -9 "$pid" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  echo "[gate3] cleanup: stopping capture procs, guaranteeing VM is not left paused"
  [ -n "$PRODUCER_PID" ] && stop_bg "$PRODUCER_PID" producer 4
  [ -n "$CONSUMER_PID" ] && stop_bg "$CONSUMER_PID" consumer 4
  resume_vm
}
trap cleanup EXIT INT TERM

# cell := "label|binary|args-without-duration-seed-outdir"  (idle has empty binary)
CELLS=(
  "qr|kernel_qr_v2|--dim 1600"
  "lu|kernel_lu_v2|--dim 2400"
  "gemm|kernel_gemm_v2|--dim 1024 --block 64"
  "gibbs|kernel_gibbs_v2|--width 1024 --height 1024"
  "stencil|kernel_stencil_jacobi_v2|--grid-n 1024"
  "idle||"
)

# run_cell returns: 0 ok, 1 non-fatal (this cell empty), 2 FATAL (abort run)
run_cell() { # label binary args rep seed
  local label="$1" bin="$2" args="$3" rep="$4" seed="$5"
  local celldir="$CELLS_DIR/${label}_rep${rep}"; mkdir -p "$celldir"
  echo "[gate3] === cell=$label rep=$rep seed=$seed ==="

  ensure_vm_up || return 2

  if [ -n "$bin" ]; then
    if ! wait_for_reachable; then
      echo "[gate3] FATAL: guest $SSH_TARGET not reachable within ${BOOT_WAIT}s before cell $label rep$rep"
      echo "[gate3]        (guest may have a new IP -- check: $VIRSH domifaddr \"$DOMAIN\")"
      return 2
    fi
    local gout="$GUEST_OUTBASE/${label}_rep${rep}"
    if ! ssh_guest "mkdir -p '$gout' && nohup '$GUEST_BINDIR/$bin' $args --duration $WORKLOAD_DUR --seed $seed --output-dir '$gout' --phase-markers >/tmp/gate3_${label}_${rep}.log 2>&1 &"; then
      echo "[gate3] FATAL: SSH-launch of $bin failed (guest reachable but launch errored; check GUEST_BINDIR=$GUEST_BINDIR)"
      return 2
    fi
    sleep 2   # let the workload enter its measure phase before capture starts
  else
    echo "[gate3] idle cell: no workload, capturing baseline guest"
  fi

  CONFIG="$GATE3_CONFIG" nohup bash "$QEMU_DIR/capture_producer_qemu_pmemsave.sh" > "$celldir/producer.log" 2>&1 &
  PRODUCER_PID=$!
  CONFIG="$GATE3_CONFIG" CAPTURE_METRIC=substrate nohup bash "$QEMU_DIR/capture_consumer_qemu.sh" > "$celldir/consumer.log" 2>&1 &
  CONSUMER_PID=$!

  echo "[gate3] producer=$PRODUCER_PID consumer=$CONSUMER_PID capturing ${CAPTURE_SECS}s ..."
  sleep "$CAPTURE_SECS"

  # Stop the producer, then IMMEDIATELY guarantee the VM is running -- the producer
  # has no exit-trap, so if we stopped it mid-suspend the VM would stay paused.
  stop_bg "$PRODUCER_PID" producer 8; PRODUCER_PID=""
  resume_vm

  echo "[gate3] producer stopped; draining consumer up to ${DRAIN_SECS}s"
  local t0=$SECONDS pend
  while (( SECONDS - t0 < DRAIN_SECS )); do
    pend=$(find "$QUEUE_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
    [ "${pend:-0}" -eq 0 ] && break
    sleep 3
  done
  stop_bg "$CONSUMER_PID" consumer 6; CONSUMER_PID=""
  resume_vm   # belt-and-braces

  # stop the guest workload so it cannot bleed into the next cell (best-effort)
  if [ -n "$bin" ] && guest_reachable; then
    ssh_guest "pkill -f '$bin' 2>/dev/null || true" || true
  fi

  local traj="$OUTPUT_DIR/substrate_trajectory.csv"
  if [ -s "$traj" ]; then
    mv "$traj" "$celldir/substrate_trajectory.csv"
    local nseq nrows
    nseq=$(awk -F, 'NR>1{print $1}' "$celldir/substrate_trajectory.csv" | sort -n -u | wc -l | tr -d ' ')
    nrows=$(( $(wc -l < "$celldir/substrate_trajectory.csv") - 1 ))
    echo "[gate3] saved $celldir/substrate_trajectory.csv ($nseq snapshots, $nrows changed-page rows)"
  else
    echo "[gate3] WARNING: no trajectory for $label rep$rep -- see $celldir/{producer,consumer}.log"
  fi
  # reset accumulators for the next cell
  rm -f "$OUTPUT_DIR/substrate_trajectory.csv" 2>/dev/null || true
  find "$QUEUE_DIR" -type f -delete 2>/dev/null || true
  find "$IMAGE_DIR" -type f -delete 2>/dev/null || true
  return 0
}

# --- preflight --------------------------------------------------------------
# Start/resume the guest if needed, then wait for SSH (it may be a cold boot).
ensure_vm_up || { echo "[gate3] aborting: VM not runnable"; exit 1; }
if ! wait_for_reachable; then
  echo "[gate3] guest $SSH_TARGET not reachable within ${BOOT_WAIT}s of start."
  echo "[gate3] The IP may have changed on boot -- check:  $VIRSH domifaddr \"$DOMAIN\"  and rerun with that SSH_TARGET."
  exit 1
fi

echo "[gate3] pilot: ${#CELLS[@]} cells x $REPS reps, ${CAPTURE_SECS}s each, speed 2 -> $CELLS_DIR"
ABORT=0
for rep in $(seq 1 "$REPS"); do
  seed=$((41 + rep))   # 42, 43, 44
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r label bin args <<< "$cell"
    run_cell "$label" "$bin" "$args" "$rep" "$seed"; rc=$?
    if [ "$rc" -eq 2 ]; then ABORT=1; break; fi
  done
  [ "$ABORT" -eq 1 ] && break
done

if [ "$ABORT" -eq 1 ]; then
  echo
  echo "[gate3] ABORTED mid-run (guest unreachable or un-runnable). Completed cells are under $CELLS_DIR."
  echo "[gate3] Recover the guest ($VIRSH domstate \"$DOMAIN\"; $VIRSH domifaddr \"$DOMAIN\"), then rerun with the correct SSH_TARGET."
  exit 2
fi

echo
echo "[gate3] DONE. Per-cell trajectories under $CELLS_DIR/<label>_rep<r>/substrate_trajectory.csv"
echo "[gate3] next: pull these back and run the Gate 3 analysis (footprint-size trajectory + magnitude distribution)."
