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
# WHAT IT DOES
#   For each pilot cell x rep: SSH-launch the kernel in the guest, capture
#   substrate metrics (speed 2, sparse) via the existing producer/consumer for a
#   fixed window, then save that cell's substrate_trajectory.csv. Isolated
#   scratch config; production config/output/queue untouched.
#
# REQUIRED:  SSH_TARGET=user@guest-host   (optional SSH_KEY, SSH_OPTS)
# OPTIONAL:  GUEST_REPO (guest-side repo path), REPS, CAPTURE_SECS, WORKLOAD_DUR
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

REPS="${REPS:-3}"
CAPTURE_SECS="${CAPTURE_SECS:-70}"   # producer capture window (wall clock)
WORKLOAD_DUR="${WORKLOAD_DUR:-120}"  # guest --duration; > CAPTURE_SECS so it outlives capture
DRAIN_SECS="${DRAIN_SECS:-30}"       # at speed 2 the consumer ~keeps up; small tail drain
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

# cell := "label|binary|args-without-duration-seed-outdir"  (idle has empty binary)
CELLS=(
  "qr|kernel_qr_v2|--dim 1600"
  "lu|kernel_lu_v2|--dim 2400"
  "gemm|kernel_gemm_v2|--dim 1024 --block 64"
  "gibbs|kernel_gibbs_v2|--width 1024 --height 1024"
  "stencil|kernel_stencil_jacobi_v2|--grid-n 1024"
  "idle||"
)

run_cell() { # label binary args rep seed
  local label="$1" bin="$2" args="$3" rep="$4" seed="$5"
  local celldir="$CELLS_DIR/${label}_rep${rep}"; mkdir -p "$celldir"
  echo "[gate3] === cell=$label rep=$rep seed=$seed ==="

  if [ -n "$bin" ]; then
    local gout="$GUEST_OUTBASE/${label}_rep${rep}"
    ssh "${SSH_KEY_OPT[@]}" $SSH_OPTS "$SSH_TARGET" \
      "mkdir -p '$gout' && nohup '$GUEST_BINDIR/$bin' $args --duration $WORKLOAD_DUR --seed $seed --output-dir '$gout' --phase-markers >/tmp/gate3_${label}_${rep}.log 2>&1 &" \
      || { echo "[gate3] ERROR: SSH-launch of $bin failed (check GUEST_BINDIR=$GUEST_BINDIR)"; return 1; }
    sleep 2   # let the workload enter its measure phase before capture starts
  else
    echo "[gate3] idle cell: no workload, capturing baseline guest"
  fi

  CONFIG="$GATE3_CONFIG" nohup bash "$QEMU_DIR/capture_producer_qemu_pmemsave.sh" > "$celldir/producer.log" 2>&1 &
  local ppid=$!
  CONFIG="$GATE3_CONFIG" CAPTURE_METRIC=substrate nohup bash "$QEMU_DIR/capture_consumer_qemu.sh" > "$celldir/consumer.log" 2>&1 &
  local cpid=$!

  echo "[gate3] producer=$ppid consumer=$cpid capturing ${CAPTURE_SECS}s ..."
  sleep "$CAPTURE_SECS"
  kill "$ppid" 2>/dev/null || true; sleep 1; kill -9 "$ppid" 2>/dev/null || true
  echo "[gate3] producer stopped; draining consumer up to ${DRAIN_SECS}s"
  local t0=$SECONDS
  while (( SECONDS - t0 < DRAIN_SECS )); do
    [ -s "$QUEUE_DIR" ] 2>/dev/null || true
    # stop early if queue dir has drained (no pending prev/curr pairs)
    local pend; pend=$(find "$QUEUE_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
    [ "${pend:-0}" -eq 0 ] && break
    sleep 3
  done
  kill "$cpid" 2>/dev/null || true; sleep 1; kill -9 "$cpid" 2>/dev/null || true

  # stop the guest workload so it cannot bleed into the next cell
  if [ -n "$bin" ]; then
    ssh "${SSH_KEY_OPT[@]}" $SSH_OPTS "$SSH_TARGET" "pkill -f '$bin' 2>/dev/null || true" || true
  fi

  local traj="$OUTPUT_DIR/substrate_trajectory.csv"
  if [ -s "$traj" ]; then
    mv "$traj" "$celldir/substrate_trajectory.csv"
    local nseq; nseq=$(awk -F, 'NR>1{print $1}' "$celldir/substrate_trajectory.csv" | sort -n -u | wc -l | tr -d ' ')
    local nrows; nrows=$(( $(wc -l < "$celldir/substrate_trajectory.csv") - 1 ))
    echo "[gate3] saved $celldir/substrate_trajectory.csv ($nseq snapshots, $nrows changed-page rows)"
  else
    echo "[gate3] WARNING: no trajectory for $label rep$rep -- see $celldir/producer.log, $celldir/consumer.log"
  fi
  # reset accumulators for the next cell
  rm -f "$OUTPUT_DIR/substrate_trajectory.csv" 2>/dev/null || true
  find "$QUEUE_DIR" -type f -delete 2>/dev/null || true
  find "$IMAGE_DIR" -type f -delete 2>/dev/null || true
}

echo "[gate3] pilot: ${#CELLS[@]} cells x $REPS reps, ${CAPTURE_SECS}s each, speed 2 -> $CELLS_DIR"
for rep in $(seq 1 "$REPS"); do
  seed=$((41 + rep))   # 42, 43, 44
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r label bin args <<< "$cell"
    run_cell "$label" "$bin" "$args" "$rep" "$seed" || echo "[gate3] cell $label rep$rep errored -- continuing"
  done
done

echo
echo "[gate3] DONE. Per-cell trajectories under $CELLS_DIR/<label>_rep<r>/substrate_trajectory.csv"
echo "[gate3] next: pull these back and run the Gate 3 analysis (footprint-size trajectory + magnitude distribution)."
