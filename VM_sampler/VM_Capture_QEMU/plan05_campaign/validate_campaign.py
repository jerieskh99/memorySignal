#!/usr/bin/env python3
"""Validate the full apf_queue campaign against the C1-C8 *intent*.

plan02_validate_session.py checks eight claims (C1-C8), but it expects the
plan02 SESSION schema: per-cell workload_stderr.log (C1), run_record.json
(C2/C4), producer.log (C5), and plan03/plan04 outputs (C7/C8). The Plan-05
campaign uses run_files_controlled.py (the production orchestrator), which emits
a different layout -- only the per-step apf_trajectory.jsonl. So the literal
validator does not apply; fabricating the missing artifacts would make C1/C2/C4
meaningless.

This validator checks what the apf_queue output CAN prove, mapped to the C1-C8
intent, per cell (joined to its workload/duration/rep via full_manifest.csv):

  INTEGRITY  (C6 intent) -- trajectory present, valid JSON, all required keys,
             seq contiguous 0..N-1 (no gaps/dupes/reorder). HARD gate.
  ACTIVITY   (C1 intent) -- apf_max >= ACTIVITY_MIN: the capture observed real
             memory change, i.e. the workload actually ran. HARD gate.
  COMPLETE   (C2 intent) -- >= 1 analysis window of data (pairs >= W). HARD gate.
  DOF        (C3)        -- windows@(8,4); flag <= 3 (v3's pain point). Report.

Exit 0 iff every cell passes the three hard gates.

    python3 plan05_campaign/validate_campaign.py [data_dir]

data_dir defaults to the server queue dir; pass a local dir when validating an
extracted backup tar.
"""
import csv
import json
import sys
from pathlib import Path

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/project/homes/jeries/memory_traces/queue_dir"
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "full_manifest.csv"
W, H = 8, 4
ACTIVITY_MIN = 0.02   # apf_max below this -> capture saw ~no change (suspect idle)


def windows(n: int) -> int:
    return (n - W) // H + 1 if n >= W else 0


def load(path: Path):
    """Return (seqs, apfs, n_badjson, n_missing_keys)."""
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
        print(f"no manifest at {MANIFEST}")
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
                  f"{'--':>6s} {'--':>7s} {'--':>4s}  MISSING")
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
    passed = total - n_fail - n_missing
    print(f"cells: {total}  |  passed hard gates: {passed}  |  "
          f"failed: {n_fail}  |  missing: {n_missing}")
    print(f"DOF: {n_starved}/{total} cells with <= 3 windows "
          f"(informational; v3 baseline 53/132)")
    verdict = "PASS" if (n_fail == 0 and n_missing == 0) else "FAIL"
    print(f"\nVERDICT: {verdict}")

    print("\n=== C1-C8 mapping for the apf_queue campaign ===")
    rep_line = "PASS" if verdict == "PASS" else "see above"
    table = [
        ("C1 workload ran", "ACTIVITY (apf_max>=0.02)", rep_line),
        ("C2 snap completion", "COMPLETE (>=1 window)", rep_line),
        ("C3 n_windows", "DOF column", f"{total - n_starved}/{total} > 3 windows"),
        ("C4 lock_retries==0", "N/A", "no lock-settle in apf_queue path"),
        ("C5 producer no errors", "log scan (separate)", "1 recovered SSH-255, else clean"),
        ("C6 trajectory complete", "INTEGRITY", rep_line),
        ("C7 plan03 winner", "N/A", "plan03 not run on this campaign"),
        ("C8 plan04 segmenter", "N/A", "plan04 not run on this campaign"),
    ]
    print(f"  {'claim':24s} {'our check':26s} status")
    for claim, check, status in table:
        print(f"  {claim:24s} {check:26s} {status}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
