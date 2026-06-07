#!/usr/bin/env python3
"""Health + results view for the full 66-cell APF campaign.

Joins full_manifest.csv (step -> workload/duration/rep) with the per-step
apf_trajectory.jsonl files so every row is labelled by its actual cell, not just
the binary name. Reports pairs, APF mean, and analysis windows at the frozen
(W,H)=(8,4), plus a summary (done/pending, IDLE-flagged, DOF-starved, per-
duration window stats). Cells not yet run show as PENDING.

    python3 plan05_campaign/analyze_campaign.py [queue_dir]

Pass the queue dir if it differs from the default. Read-only; safe to run while
the campaign is still going.
"""
import csv
import glob
import json
import sys
from pathlib import Path

QUEUE = sys.argv[1] if len(sys.argv) > 1 else \
    "/project/homes/jeries/memory_traces/queue_dir"
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "full_manifest.csv"
W, H = 8, 4
IDLE_APF = 0.01     # below this, flag as possibly-idle (heuristic, see note)
STARVED_WIN = 3     # v3's pain threshold: cells with <= this many windows


def windows(n: int) -> int:
    return (n - W) // H + 1 if n >= W else 0


def read_apf(path: Path) -> list[float]:
    vals: list[float] = []
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = o.get("apf")
                if isinstance(v, (int, float)):
                    vals.append(float(v))
    except FileNotFoundError:
        pass
    return vals


def main() -> int:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST}; run generate_full_steps.py first")
        return 1
    rows = list(csv.DictReader(MANIFEST.open()))

    # A run may rename trajectories to dodge cross-run collisions; index by the
    # whole queue dir so we still find any present file by its base name.
    present = {Path(p).name: Path(p)
               for p in glob.glob(f"{QUEUE}/run_matrix_*.apf_trajectory.jsonl")}

    hdr = (f"{'#':>3} {'workload':28s} {'dur':>4s} {'rep':>3s} "
           f"{'pairs':>6s} {'APF':>8s} {'win':>5s}  flag")
    print(hdr)
    print("-" * len(hdr))

    done = idle = starved = 0
    win_by_dur: dict[str, list[int]] = {}
    for r in rows:
        traj = present.get(r["traj_file"])
        wl = r["workload"]
        dur = r["duration_s"]
        rep = r["rep"]
        if traj is None:
            print(f"{r['step_index']:>3} {wl:28s} {dur:>4s} {rep:>3s} "
                  f"{'':>6s} {'':>8s} {'':>5s}  PENDING")
            continue
        vals = read_apf(traj)
        m = sum(vals) / len(vals) if vals else 0.0
        w = windows(len(vals))
        done += 1
        win_by_dur.setdefault(dur, []).append(w)
        flags = []
        if m < IDLE_APF:
            idle += 1
            flags.append("IDLE?")
        if w <= STARVED_WIN:
            starved += 1
            flags.append("STARVED")
        print(f"{r['step_index']:>3} {wl:28s} {dur:>4s} {rep:>3s} "
              f"{len(vals):>6d} {m:>8.4f} {w:>5d}  {' '.join(flags)}")

    print("-" * len(hdr))
    print(f"done {done}/{len(rows)}  |  IDLE-flagged {idle}  |  "
          f"DOF-starved (<= {STARVED_WIN} win) {starved}  "
          f"(v3 baseline: 53/132)")
    for dur in sorted(win_by_dur, key=lambda d: int(d)):
        ws = win_by_dur[dur]
        print(f"  dur {dur:>3s}s: {len(ws):>2d} cells, "
              f"windows min/median/max = {min(ws)}/"
              f"{sorted(ws)[len(ws)//2]}/{max(ws)}")
    print("note: IDLE? is APF < 0.01 -- a heuristic. File-I/O workloads "
          "(ransom_slowburn/batched) read genuinely low; cross-check vs v3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
