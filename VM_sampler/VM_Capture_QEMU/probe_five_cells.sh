#!/usr/bin/env bash
# Paper 2 probe: five cells, twelve columns, the floor gate.
# Written 2026-09-28 from the council run of epochs 0-4 plus two lenses.
# RUN THIS ON THE SERVER. The retention root is not mounted on the laptop.
#
# It reads the retention root and writes ONLY under $OUT. It does not touch the
# paper run's output, the repository, or any tracked file.
#
# What it settles, in one pass:
#   1. whether the extract computes the amount ratios correctly at all   (gibbs must read exactly 1.00)
#   2. the footprint levels, and whether stencil_jacobi joins a level-matched set
#   3. whether lexer sits at the floor
#   4. the floor's size, its own J, and its amount signature              (needs the idle cell)
#   5. the true pairs per run, hence the 644/145 ms figures and "four million rows"
#   6. whether frac_below_null is a constant zero, i.e. whether a pre-registered gate is dead
set -euo pipefail

ROOT="${ROOT:-/mnt/nfs/jeries/memory_traces/zstd_local}"
OUT="${OUT:-/tmp/p11probe}"
# this script lives in the directory that contains plan11_encoding_ladder/, so TOOLKIT is its own dir
TOOLKIT="${TOOLKIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

echo "== 0. free: order all twelve kernels by changed-byte volume =="
echo "   (an at-floor kernel shows up here immediately; lexer is predicted smallest)"
du -sh "$ROOT"/kernel/*/ 2>/dev/null | sort -h || echo "   (adjust ROOT if this listed nothing)"
echo

cd "$TOOLKIT"

echo "== 1. index the retention root =="
# --idle-marker defaults to (sleep, idle); the idle cell MUST be classified role=idle
# or floor_K is null, every pair reads floor-unmeasured, and step 4 is invisible.
python3 -m plan11_encoding_ladder.extract index --root "$ROOT" --out "$OUT"
echo
echo "   cells found, by role:"
awk -F, 'NR>1{print $NF}' "$OUT/cells.csv" | sort | uniq -c || true
echo
echo "   >>> CHECK NOW, before going on: the cell ids you want must be in $OUT/cells.csv,"
echo "   >>> and at least one must have role=idle. The regex below is a guess at the id format."
grep -Ei 'floyd|histogram|nbody|stencil|gibbs|sleep|idle' "$OUT/cells.csv" | head -20 || true
echo

echo "== 2. extract five cells =="
# floyd + nbody   : the double pair r_2 provably cannot separate (ECG seat)
# histogram       : the counter kernel, reset every pass
# gibbs           : THE CALIBRATION CELL. uint8 spins in {0,1} => r_l1l0 and r_haml0
#                   are exactly 1.00 with no free parameter. If they are not, the
#                   extract is wrong and every amount number in the paper is void.
# stencil_jacobi  : 2,048 written per sweep / 4,096 resident; settles the set collision
# one idle cell   : required for the floor gate to compute anything at all
python3 -m plan11_encoding_ladder.extract all \
  --cells-csv "$OUT/cells.csv" --out "$OUT" --jobs 3 --persist-side t \
  --only '(floyd|histogram|nbody|gibbs|stencil_jacobi).*rep0*1|sleep|idle'
echo

echo "== 3. the floor gate =="
python3 -m plan11_encoding_ladder.gates_readings gj --out "$OUT"
echo

echo "== 4. read it =="
python3 - "$OUT" <<'PY'
import csv, glob, json, os, statistics as st, sys
out = sys.argv[1]
cols = ["K","n_persist","n_union","J","J_null","J_null_inter",
        "r_l0_q05_per","r_l0_q25_per","r_l0_q50_per","r_l0_q75_per","r_l0_q95_per",
        "r_l1l0_q05_per","r_l1l0_q50_per","r_l1l0_q95_per","r_haml0_q50_per"]
paths = sorted(glob.glob(os.path.join(out, "extract", "*.csv")))
if not paths:
    print("  no extract CSVs found under", os.path.join(out, "extract")); sys.exit(0)
def q(v, p):
    if not v: return float("nan")
    s = sorted(v); i = min(len(s)-1, max(0, int(p*len(s))-1)); return s[i]
for p in paths:
    rows = list(csv.DictReader(open(p)))
    print(f"\n{os.path.basename(p)}   pairs={len(rows)}")
    for c in cols:
        v = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
        if not v:
            print(f"   {c:18s} (empty)"); continue
        print(f"   {c:18s} n={len(v):5d}  q05={q(v,0.05):<12.6g} med={st.median(v):<12.6g} q95={q(v,0.95):<12.6g}")
print("\n--- gates/gj.json ---")
gj = os.path.join(out, "gates", "gj.json")
print(json.dumps(json.load(open(gj)), indent=2)[:3000] if os.path.exists(gj) else "  not written")
PY

cat <<'NOTES'

== What to look at, in priority order ==

1. GIBBS, r_l1l0 and r_haml0. Every quantile should be EXACTLY 1.00 on the pages that
   are gibbs's own. If the LOW quantiles are 1.00 and the high ones are not, the break
   point estimates the floor's share of the persistent set. If NOTHING is 1.00, stop:
   the extract is wrong.

2. frac_below_null in gates/gj.json. Predicted 0.0 in every cell. If it is, the rule
   "J <= J_null" can never fire and the pre-registered threshold has to be re-declared
   against the floor null f/(2K-f), recorded as a change to a pre-registered gate.

3. The IDLE cell's median J. Predicted at or above 0.5. If it comes out LOW, the floor is
   not a nearly fixed set and the volatile-memory seat's whole floor model is wrong.

4. floor_median_n_persist in gj.json. This is f, the number the corrected J sentence
   should quote instead of an illustration.

5. histogram's median r_l1l0 against nbody's. Predicted 8-43 against ~75. If histogram
   is near 75, the amount axis is saturated and the dsp seat's objection was right.

6. median K per cell, for floyd / histogram / nbody / stencil_jacobi. Settles the
   footprint levels the Introduction states as "exactly", and stencil_jacobi's membership.

7. pairs= per cell. The paper says "about 900 pairs" from two trajectories and Table 1
   quotes a DUMP range as a pair range.

== What this probe cannot answer ==

The 145 ms residual. No extract column carries a host timestamp; the differ's rows are
seq, page_index, hamming, l0, l1. That needs either the done/ job filenames against the
orchestrator's timestamps.log, or a separate producer run with TIMING_JSONL_PATH set,
which suspends the guest and must not run during a capture.
NOTES
