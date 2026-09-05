#!/usr/bin/env bash
# =============================================================================
# Gate 1 of the dwarf pilot -- resize-timing measurement (NO capture).
#
# WHAT IT ANSWERS
#   At what matrix size does ONE QR / LU factorisation take long enough to span
#   several 500 ms snapshots? Below that size the 2 Hz snapshot integrates many
#   factorisations into one "footprint fully lit" frame (aliased) and the
#   growing/shrinking front is invisible. Gate 1 finds the size where the front
#   survives -- measured on the GUEST cpu, because cache effects (memory expert's
#   caveat) make the true size differ from the pure-N^3 extrapolation.
#
# WHY STANDALONE
#   This is timing only. It runs the kernel binaries directly in the guest and
#   reads their metadata JSON counter (qr: "orthogonalisations", lu:
#   "factorisations", each = one full factorisation). No QEMU, no pmemsave, no
#   differ. Because there is no capture, the guest is never suspended, so the
#   factorisation runs at full guest speed -- exactly its speed between snapshots
#   during a real capture.
#
# WHERE TO RUN
#   Inside the guest (kali), where the kernels are built:
#     cd /home/kali/memorySignal && git pull
#     bash VM_sampler/VM_Capture_QEMU/plan09_dwarf_pilot/gate1_dwarf_timing.sh
#
# READ THE OUTPUT
#   snaps_per_fact = how many 500 ms snapshots one factorisation spans.
#     >= 8  ideal  (one factorisation ~ one 8-sample classifier window)
#     >= 3  minimum (front resolvable at all)
#     <  1  aliased (still a blur -- size too small)
#   The tail prints an interpolated pilot dim for the 8- and 3-snapshot targets.
# =============================================================================
set -euo pipefail

REPO="${REPO:-/home/kali/memorySignal}"
BINDIR="$REPO/VM_executables_phase2/bin"
SRCDIR="$REPO/VM_executables_phase2"
OUT="${OUT:-/tmp/gate1_dwarf}"
DUR="${DUR:-24}"            # measure seconds per run (long enough to count a few slow factorisations)
WARM="${WARM:-2}"           # warmup seconds
MAXMB="${MAXMB:-512}"       # per-kernel byte cap; guest is ~1 GiB so keep well under
INTERVAL_MS=500            # capture snapshot cadence (must match config_qemu_upc.json)

# Size ladders, ascending. Centred on the predicted targets (qr ~2000, lu ~3800)
# with rungs either side so the crossing point is bracketed and interpolated.
QR_DIMS="${QR_DIMS:-1024 1600 2000 2400}"
LU_DIMS="${LU_DIMS:-2800 3400 3800 4400}"

mkdir -p "$OUT"
CSV="$OUT/gate1_results.csv"
echo "kernel,dim,counter,dur_s,sec_per_fact,snaps_per_fact,footprint_pages,pct_guest_1gib" > "$CSV"

# --- binaries must already be built in the guest (the campaign built them) -----
for k in kernel_qr_v2 kernel_lu_v2; do
  if [ ! -x "$BINDIR/$k" ]; then
    echo "[gate1] ERROR: $BINDIR/$k not found."
    echo "[gate1] Build first:  ( cd $SRCDIR && make )"
    exit 1
  fi
done

extract_counter() { # <metadata.json> <key>
  grep -o "\"$2\":[0-9]*" "$1" 2>/dev/null | grep -o '[0-9]*' | head -1
}

run_one() { # <binname> <counter-key> <dim>
  local bin="$1" key="$2" dim="$3"
  local od="$OUT/${bin}_dim${dim}"
  mkdir -p "$od"
  if ! "$BINDIR/$bin" --dim "$dim" --duration "$DUR" --warmup "$WARM" \
        --seed 42 --max-mb "$MAXMB" --output-dir "$od" >/dev/null 2>&1; then
    echo "[gate1] $bin dim=$dim FAILED (over --max-mb=$MAXMB, or build issue) -- skipping"
    return 0
  fi
  local meta="$od/${bin}_metadata.json"
  local c; c="$(extract_counter "$meta" "$key")"
  if [ -z "${c:-}" ] || [ "$c" -eq 0 ] 2>/dev/null; then
    echo "[gate1] $bin dim=$dim: counter '$key' missing or zero (one factorisation > ${DUR}s? raise DUR) -- skipping"
    return 0
  fi
  awk -v k="$bin" -v d="$dim" -v c="$c" -v dur="$DUR" -v iv="$INTERVAL_MS" 'BEGIN{
    spf   = dur/c;                 # seconds per factorisation
    snaps = spf*1000.0/iv;         # snapshots (500ms) per factorisation
    pages = (d*d*8)/4096;          # one N*N f64 matrix, in 4 KiB pages
    pct   = 100.0*(d*d*8)/(1024*1024*1024);
    printf "%s,%d,%d,%d,%.3f,%.2f,%.0f,%.1f\n", k,d,c,dur,spf,snaps,pages,pct;
  }' | tee -a "$CSV"
}

printf '\n[gate1] measure=%ss warmup=%ss max-mb=%s  interval=%sms  out=%s\n' \
  "$DUR" "$WARM" "$MAXMB" "$INTERVAL_MS" "$OUT"

echo; echo "== QR (growing front) =="
printf '%-14s %6s %8s %12s %14s %10s %6s\n' kernel dim counter sec/fact snaps/fact pages %RAM
for d in $QR_DIMS; do run_one kernel_qr_v2 orthogonalisations "$d"; done

echo; echo "== LU (shrinking front) =="
printf '%-14s %6s %8s %12s %14s %10s %6s\n' kernel dim counter sec/fact snaps/fact pages %RAM
for d in $LU_DIMS; do run_one kernel_lu_v2 factorisations "$d"; done

# --- interpolate the pilot dim from the MEASURED points (not pure N^3) ---------
echo
echo "== recommended pilot dim (interpolated from measured guest timing) =="
echo "   target: 8 snapshots/factorisation (ideal), 3 (minimum resolvable)"
awk -F, '
NR>1 && ($6+0)>0 { k=$1; n[k]++; dim[k SUBSEP n[k]]=$2+0; sn[k SUBSEP n[k]]=$6+0; }
END{
  split("8 3", tg, " ");
  for (k in n) {
    for (ti=1; ti<=2; ti++) {
      t=tg[ti]+0; done=0;
      for (i=1; i<n[k]; i++) {
        d1=dim[k SUBSEP i];   d2=dim[k SUBSEP (i+1)];
        s1=sn[k SUBSEP i];    s2=sn[k SUBSEP (i+1)];
        lo=(s1<s2)?s1:s2; hi=(s1<s2)?s2:s1;
        if (t>=lo && t<=hi && s1>0 && s2>0 && d2!=d1) {
          p=log(s2/s1)/log(d2/d1);            # local snaps ~ dim^p
          dstar=d1*exp(log(t/s1)/p);
          printf "   %-16s %d snaps -> dim ~%d\n", k, t, int(dstar+0.5);
          done=1; break;
        }
      }
      if (!done) {
        smin=sn[k SUBSEP 1]; smax=sn[k SUBSEP n[k]];
        if (t < smin)
          printf "   %-16s %d snaps -> already exceeded at smallest tested dim=%d (%.1f snaps): resolvable, shrink ladder if you want a tighter fit\n", \
                 k, t, dim[k SUBSEP 1], smin;
        else
          printf "   %-16s %d snaps -> above ladder: extend to larger dims (max tested=%d @ %.1f snaps)\n", \
                 k, t, dim[k SUBSEP n[k]], smax;
      }
    }
  }
}' "$CSV"

echo
echo "[gate1] full table -> $CSV"
echo "[gate1] GO if QR and LU each reach >=3 snaps/fact within the guest RAM budget;"
echo "        pick the ~8-snap dim for the pilot (Gate 3). NO-GO if neither crosses 3."
