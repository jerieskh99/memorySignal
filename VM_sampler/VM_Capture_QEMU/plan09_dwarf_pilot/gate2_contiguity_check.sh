#!/usr/bin/env bash
# =============================================================================
# Gate 2 of the dwarf pilot -- guest-physical CONTIGUITY check (real capture).
#
# WHAT IT ANSWERS
#   kernel_gemm_v2's C matrix is one array from one mmap() (guest-VIRTUAL
#   contiguous), but pmemsave dumps guest-PHYSICAL RAM, and every phase-2
#   kernel opts OUT of transparent huge pages (MADV_NOHUGEPAGE), which removes
#   the one mechanism that would force physical contiguity. Every "footprint
#   shape" feature in the pilot (barnes_hut vs nbody, the LU/QR front) assumes
#   changed pages of one array show up as a contiguous (or near-contiguous)
#   run of page_index values -- not scattered across the 262144-page dump.
#   This is the single most important go/no-go in the whole pilot: everything
#   address-shape depends on it, and it is free to test.
#
# WHAT IT DOES
#   1. Launches kernel_gemm_v2 in the guest (SSH) for ~20s.
#   2. Captures real pmemsave snapshots at speed 0 --sparse (the richest
#      differ mode; page_index is only emitted in sparse output) via the
#      EXISTING producer/consumer pair -- CAPTURE_METRIC=substrate, its own
#      documented mechanism (capture_consumer_qemu.sh: substrateSpeed from
#      config, --sparse hardcoded whenever CAPTURE_METRIC=substrate).
#   3. Stops both after a fixed wall-clock window (the producer/consumer loops
#      are `while true` -- there is no snapshot-count flag to bound them, so
#      this script owns the timing and kills them itself).
#   4. Reads the accumulated substrate_trajectory.csv, picks two consecutive
#      snapshots well past the warmup, and checks whether the changed pages'
#      page_index values form one contiguous run or scatter.
#
# WHAT IT DOES NOT DO
#   Does not touch production config_qemu_upc.json, its imageDir, outputDir,
#   or queueDir -- everything here runs under its own scratch paths so a bad
#   run cannot corrupt or mix into the real campaign's data.
#
# REQUIRED (fill these in -- not guessed):
#   SSH_TARGET   user@host for the guest, e.g. "jeries@192.168.122.50"
#   (optional)   SSH_KEY, SSH_OPTS       -- passed straight to ssh
#
# WHERE TO RUN
#   On the SERVER (where virsh/libvirt and the producer/consumer scripts run),
#   not inside the guest:
#     cd /project/homes/jeries/memorySignal
#     SSH_TARGET=you@guest-ip bash VM_sampler/VM_Capture_QEMU/plan09_dwarf_pilot/gate2_contiguity_check.sh
#
# READ THE OUTPUT
#   "largest contiguous run" as a fraction of changed pages, for each of the
#   two snapshots checked:
#     > ~0.8   contiguous -- address-shape features are trustworthy, proceed
#              to Gate 3 with them included.
#     mixed    partially scattered -- treat address-shape as a secondary,
#              lower-confidence lead; keep magnitude/footprint-size as primary.
#     < ~0.2   scattered -- drop barnes_hut/nbody and the LU/QR-front address
#              claims from Gate 3; keep magnitude + footprint-size only.
# =============================================================================
set -euo pipefail

REPO="${REPO:-/project/homes/jeries/memorySignal}"
QEMU_DIR="$REPO/VM_sampler/VM_Capture_QEMU"
BINDIR="$REPO/VM_executables_phase2/bin"
BASE_CONFIG="${BASE_CONFIG:-$QEMU_DIR/config_qemu_upc.json}"

: "${SSH_TARGET:?SSH_TARGET (user@guest-host) is required -- not guessed, set it explicitly}"
SSH_KEY_OPT=()
[ -n "${SSH_KEY:-}" ] && SSH_KEY_OPT=(-i "$SSH_KEY")
SSH_OPTS="${SSH_OPTS:-}"

WORKLOAD_DUR="${WORKLOAD_DUR:-20}"   # guest-side kernel_gemm_v2 --duration
CAPTURE_SECS="${CAPTURE_SECS:-25}"   # how long producer+consumer run (wall clock)
SCRATCH="${SCRATCH:-/var/tmp/gate2_dwarf_pilot}"
GUEST_OUTDIR="${GUEST_OUTDIR:-/tmp/gate2_gemm}"

mkdir -p "$SCRATCH"
IMAGE_DIR="$SCRATCH/dump"
OUTPUT_DIR="$SCRATCH/output"
QUEUE_DIR="$SCRATCH/queue"
GATE2_CONFIG="$SCRATCH/config_gate2.json"
mkdir -p "$IMAGE_DIR" "$OUTPUT_DIR" "$QUEUE_DIR"

if [ ! -f "$BASE_CONFIG" ]; then
  echo "[gate2] ERROR: base config not found at $BASE_CONFIG (set BASE_CONFIG=)"
  exit 1
fi

# --- build an isolated gate2 config from the real one, overriding only the
#     scratch paths + speed 0 (richest metrics) + streaming/retention off
#     (not needed for a 2-snapshot check, and avoids their side effects) ------
jq \
  --arg imageDir "$IMAGE_DIR" \
  --arg outputDir "$OUTPUT_DIR" \
  --arg queueDir "$QUEUE_DIR" \
  '.substrateSpeed = 0
   | .imageDir = $imageDir
   | .outputDir = $outputDir
   | .queueDir = $queueDir
   | .streaming.enabled = false
   | .rawRetention.enabled = false' \
  "$BASE_CONFIG" > "$GATE2_CONFIG"

echo "[gate2] scratch config -> $GATE2_CONFIG (substrateSpeed=0, isolated dirs, production config untouched)"

if [ ! -x "$BINDIR/kernel_gemm_v2" ]; then
  echo "[gate2] ERROR: $BINDIR/kernel_gemm_v2 not found in the GUEST-visible repo path."
  echo "[gate2] This binary must exist on the guest (built via 'make' in VM_executables_phase2), not just here."
  echo "[gate2] Continuing anyway -- the SSH launch below is what actually matters."
fi

SUBSTRATE_PROGRAM="$(jq -r '.substrateProgram' "$GATE2_CONFIG")"
if [ ! -x "$SUBSTRATE_PROGRAM" ]; then
  echo "[gate2] ERROR: substrateProgram not built: $SUBSTRATE_PROGRAM"
  echo "[gate2] Build it:  ( cd $REPO/VM_sampler/VM_Capture/live_delta_calc_modular && cargo build --release )"
  exit 1
fi

# --- 1. launch the workload in the guest, backgrounded, self-terminating ----
# NOTE: this is the repo path INSIDE THE GUEST, which may differ from $REPO
# (the server-side path). Gate 1 showed the two are not always the same --
# override with GUEST_REPO= if the guest's checkout lives somewhere else.
GUEST_REPO="${GUEST_REPO:-/project/homes/jeries/memorySignal}"
GUEST_BIN="$GUEST_REPO/VM_executables_phase2/bin/kernel_gemm_v2"
echo "[gate2] launching kernel_gemm_v2 on $SSH_TARGET for ${WORKLOAD_DUR}s ..."
ssh "${SSH_KEY_OPT[@]}" $SSH_OPTS "$SSH_TARGET" \
  "mkdir -p '$GUEST_OUTDIR' && nohup '$GUEST_BIN' --duration $WORKLOAD_DUR --seed 42 --output-dir '$GUEST_OUTDIR' --phase-markers >/tmp/gate2_gemm.log 2>&1 &" \
  || { echo "[gate2] ERROR: could not SSH-launch the workload. Is GUEST_BIN correct for this guest? Edit the script's GUEST_BIN if the repo lives elsewhere on the guest."; exit 1; }

# --- 2. start producer + consumer, backgrounded, timed ----------------------
echo "[gate2] starting producer + consumer for ${CAPTURE_SECS}s (CAPTURE_METRIC=substrate) ..."
CONFIG="$GATE2_CONFIG" nohup bash "$QEMU_DIR/capture_producer_qemu_pmemsave.sh" \
  > "$SCRATCH/producer.log" 2>&1 &
PRODUCER_PID=$!
CONFIG="$GATE2_CONFIG" CAPTURE_METRIC=substrate nohup bash "$QEMU_DIR/capture_consumer_qemu.sh" \
  > "$SCRATCH/consumer.log" 2>&1 &
CONSUMER_PID=$!

echo "[gate2] producer pid=$PRODUCER_PID consumer pid=$CONSUMER_PID -- running ${CAPTURE_SECS}s ..."
sleep "$CAPTURE_SECS"

echo "[gate2] stopping producer + consumer (both run 'while true' -- no snapshot-count flag exists, so this script owns the stop)"
kill "$PRODUCER_PID" "$CONSUMER_PID" 2>/dev/null || true
sleep 1
kill -9 "$PRODUCER_PID" "$CONSUMER_PID" 2>/dev/null || true

TRAJ="$OUTPUT_DIR/substrate_trajectory.csv"
if [ ! -s "$TRAJ" ]; then
  echo "[gate2] ERROR: no $TRAJ produced."
  echo "[gate2] Check $SCRATCH/producer.log and $SCRATCH/consumer.log -- likely causes: VM domain name/state,"
  echo "        virsh permissions, or the workload finishing before any pair was captured."
  exit 1
fi

NROWS=$(($(wc -l < "$TRAJ") - 1))
NSEQ=$(awk -F, 'NR>1{print $1}' "$TRAJ" | sort -n -u | wc -l | tr -d ' ')
echo "[gate2] $TRAJ: $NROWS changed-page rows across $NSEQ snapshots"

if [ "$NSEQ" -lt 3 ]; then
  echo "[gate2] WARNING: only $NSEQ snapshot(s) captured -- raise CAPTURE_SECS and rerun."
fi

# --- 3. contiguity check on two consecutive snapshots, skipping the first
#        (warmup / re-seed) and the last (may be mid-write at kill time) -----
awk -F, '
NR==1 { next }
{ seq=$1+0; pidx=$2+0; rows[seq]=rows[seq]" "pidx; cnt[seq]++; if(!(seq in seen)){order[++m]=seq; seen[seq]=1} }
END {
  if (m < 3) { print "[gate2] not enough distinct snapshots to pick an interior pair"; exit 0 }
  # pick two consecutive snapshots from the middle of the run
  mid = int(m/2); if (mid < 2) mid = 2; if (mid > m-1) mid = m-1;
  s1 = order[mid]; s2 = order[mid+1];
  printf "[gate2] checking snapshots seq=%d (n=%d pages) and seq=%d (n=%d pages)\n", s1, cnt[s1], s2, cnt[s2];
  for (which = 1; which <= 2; which++) {
    s = (which==1) ? s1 : s2;
    n = split(rows[s], arr, " ");
    delete vals; k=0;
    for (i=1;i<=n;i++) if (arr[i]!="") vals[++k]=arr[i]+0;
    # sort
    for (i=1;i<=k;i++) for (j=i+1;j<=k;j++) if (vals[j]<vals[i]) { t=vals[i]; vals[i]=vals[j]; vals[j]=t; }
    runs=0; longest=1; cur=1;
    for (i=2;i<=k;i++) {
      if (vals[i]==vals[i-1]+1) { cur++; } else { runs++; if(cur>longest) longest=cur; cur=1; }
    }
    if (k>0) { runs++; if(cur>longest) longest=cur; }
    frac = (k>0) ? longest/k : 0;
    printf "  seq=%-4d changed_pages=%-6d distinct_runs=%-5d longest_contig_run=%-6d frac_in_longest_run=%.3f\n", \
           s, k, runs, longest, frac;
  }
}
' "$TRAJ"

echo
echo "[gate2] verdict guide:"
echo "  frac_in_longest_run > ~0.8  -> contiguous: trust address-shape features (barnes_hut/nbody, LU/QR front) in Gate 3"
echo "  frac_in_longest_run < ~0.2  -> scattered:  drop those features, keep magnitude + footprint-size only"
echo "  in between                  -> partial: treat address-shape as secondary/lower-confidence in Gate 3"
echo "[gate2] full trajectory -> $TRAJ  (kept for inspection; scratch dirs under $SCRATCH)"
