#!/usr/bin/env python3
"""Per-cell PASS/FAIL status for the Plan 07 replicate-fill capture.

A repfill-aware copy of plan05_campaign/validate_campaign.py: same hard gates,
joined to repfill_manifest.csv instead of the v3 full_manifest.csv. Run it any
time during or after the capture to see, per cell, whether the trajectory is
present and intact, whether the workload actually ran, and whether there is
enough data to analyse.

  INTEGRITY -- trajectory present, valid JSON, required keys, seq contiguous 0..N-1.
  ACTIVITY  -- apf_max >= 0.02: the capture observed real memory change (workload ran).
  COMPLETE  -- >= 1 analysis window (pairs >= W).
  DOF       -- windows@(8,4); <= 3 flagged 'thin' (informational).

Exit 0 iff every present cell passes the three hard gates.

    python3 plan07_campaign/validate_repfill.py [data_dir]

data_dir defaults to the server queue dir (where run_matrix_*apf_trajectory.jsonl land,
matching plan05_campaign/validate_campaign.py); pass a local dir when validating an
extracted backup tar, or the output_dir if your trajectories land there instead.
"""
import csv
import json
import sys
from pathlib import Path

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/project/homes/jeries/memory_traces/queue_dir"
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "repfill_manifest.csv"
W, H = 8, 4
ACTIVITY_MIN = 0.02   # apf_max below this -> capture saw ~no change (suspect idle)


def windows(n: int) -> int:
    return (n - W) // H + 1 if n >= W else 0


def load(path: Path):
    seqs, apfs, badjson, misskey = [], [], 0, 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                badjson += 1
                continue
            if any(k not in o for k in ("seq", "t_emit_epoch", "prev", "curr", "apf")):
                misskey += 1
            if isinstance(o.get("seq"), int):
                seqs.append(o["seq"])
            if isinstance(o.get("apf"), (int, float)):
                apfs.append(float(o["apf"]))
    return seqs, apfs, badjson, misskey


def main() -> int:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST} -- run generate_repfill_steps.py first")
        return 2
    rows = list(csv.DictReader(MANIFEST.open()))

    hdr = (f"{'#':>3} {'workload':26s} {'dur':>4s} {'rep':>3s} {'pairs':>6s} "
           f"{'apf_max':>7s} {'win':>4s}  INTEG ACTIV COMPL  DOF")
    print(hdr)
    print("-" * len(hdr))
    n_fail = n_missing = n_starved = 0
    for r in rows:
        traj = Path(DATA) / r["traj_file"]
        wl, dur, rep = r["workload"], r["duration_s"], r["rep"]
        if not traj.exists():
            n_missing += 1
            print(f"{r['step_index']:>3} {wl:26s} {dur:>4s} {rep:>3s} "
                  f"{'--':>6s} {'--':>7s} {'--':>4s}  not yet / MISSING")
            continue
        seqs, apfs, badjson, misskey = load(traj)
        n = len(apfs)
        integ = (badjson == 0 and misskey == 0 and bool(seqs)
                 and seqs == list(range(len(seqs))))
        apf_max = max(apfs) if apfs else 0.0
        activ = apf_max >= ACTIVITY_MIN
        compl = windows(n) >= 1
        w = windows(n)
        starved = w <= 3
        if starved:
            n_starved += 1
        ok = integ and activ and compl
        if not ok:
            n_fail += 1

        def mark(b):
            return " ok  " if b else "FAIL "
        print(f"{r['step_index']:>3} {wl:26s} {dur:>4s} {rep:>3s} {n:>6d} "
              f"{apf_max:>7.3f} {w:>4d}  {mark(integ)} {mark(activ)} {mark(compl)}"
              f"  {'thin' if starved else 'ok'}")

    print("-" * len(hdr))
    total = len(rows)
    present = total - n_missing
    passed = present - n_fail
    print(f"cells: {total}  |  captured: {present}  |  passed hard gates: {passed}  |  "
          f"failed: {n_fail}  |  not-yet/missing: {n_missing}")
    print(f"DOF: {n_starved}/{present} captured cells with <= 3 windows (informational)")
    # During a live run, 'missing' just means not-yet-captured, so the verdict is
    # over the cells captured so far.
    verdict = "ALL CAPTURED CELLS PASS" if n_fail == 0 else "SOME CELLS FAILED"
    print(f"\nVERDICT (captured so far): {verdict}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
