#!/usr/bin/env python3
"""Summarize the subset campaign's per-step APF trajectories.

After the subset run each step wrote a per-step apf_trajectory.jsonl (under the
queueDir, named run_matrix_test{i}_*.npy.apf_trajectory.jsonl). This reports, per
step: pair count, APF mean, and the analysis-window count at the frozen
(W,H)=(8,4). Non-trivial APF + high window counts = the apf_queue + sustain
pipeline works and the DOF starvation (v3: 53/132 cells <=3 windows) is relieved.

    python3 plan05_campaign/analyze_subset.py [queue_dir]
"""
import glob
import json
import sys
from pathlib import Path

QUEUE = sys.argv[1] if len(sys.argv) > 1 else \
    "/project/homes/jeries/memory_traces/queue_dir"
W, H = 8, 4


def windows(n: int) -> int:
    return (n - W) // H + 1 if n >= W else 0


def main() -> int:
    files = sorted(glob.glob(f"{QUEUE}/run_matrix_test*.apf_trajectory.jsonl"))
    if not files:
        print(f"no apf trajectories under {QUEUE}")
        return 1
    print(f"{'step':42s} {'pairs':>6s} {'APF_mean':>9s} {'win@(8,4)':>10s}")
    starved = 0
    for f in files:
        vals = []
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(o.get("apf"), (int, float)):
                vals.append(float(o["apf"]))
        name = (Path(f).name
                .replace("run_matrix_", "")
                .replace(".npy.apf_trajectory.jsonl", ""))
        m = sum(vals) / len(vals) if vals else 0.0
        w = windows(len(vals))
        if w <= 3:
            starved += 1
        flag = "  <- IDLE?" if m < 0.01 else ""
        print(f"{name:42s} {len(vals):6d} {m:9.4f} {w:10d}{flag}")
    print(f"\n{len(files)} steps; {starved} with <=3 windows "
          f"(v3 baseline had 53/132). Non-trivial APF + high windows = pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
