#!/usr/bin/env python3
"""Windowed peak/variance shape features computed per cell from the APF
trajectory -- the candidate features to recover bursty stealth threats that the
mean misses (doc conclusion 3). Computed from the cells-dir trajectories, keyed
by cell_id, so they can be appended to the plan03 AGNOSTIC feature matrix.

  apf_max     -- peak APF in the trajectory
  apf_p95     -- 95th percentile APF
  apf_cov     -- coefficient of variation (std/mean) = burstiness
  peak2med    -- max / median (rare-spike-on-floor signature; slowburn ~337)
  duty_gt05   -- fraction of pairs with APF > 0.05 (duty cycle)

These are leakage-free in the same sense as AGNOSTIC: pure trajectory shape, not
family-conditional analyzer outputs.
"""
import json
import statistics as st
from pathlib import Path

EXTRA = ["apf_max", "apf_p95", "apf_cov", "peak2med", "duty_gt05"]


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def cell_extra(traj_path: Path) -> dict:
    v = []
    with open(traj_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("final"):
                continue
            a = o.get("apf")
            if isinstance(a, (int, float)):
                v.append(float(a))
    if not v:
        return {k: 0.0 for k in EXTRA}
    m = st.mean(v)
    sd = st.pstdev(v) if len(v) > 1 else 0.0
    med = st.median(v)
    med_g = med if med > 0 else 1e-6
    return {
        "apf_max": max(v),
        "apf_p95": _pct(v, 0.95),
        "apf_cov": (sd / m if m else 0.0),
        "peak2med": max(v) / med_g,
        "duty_gt05": sum(1 for a in v if a > 0.05) / len(v),
    }


def load_extra(cells_dir) -> dict:
    """cell_id -> {EXTRA...} for every work/<cell_id>/apf_trajectory.jsonl."""
    out = {}
    work = Path(cells_dir) / "work"
    for d in sorted(work.iterdir()):
        p = d / "apf_trajectory.jsonl"
        if p.exists():
            out[d.name] = cell_extra(p)
    return out
