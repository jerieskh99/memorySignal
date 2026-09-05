#!/usr/bin/env python3
"""b1_features.py -- turn B1 trajectories into per-cell, per-arm shape features.

Reads each cell's b1_trajectory.jsonl (+ provenance.json) under a root dir,
derives the three arm series from the stored sufficient statistics, and reduces
each to 8 scale-aware shape features. Pure stdlib -- runs on the server where the
trajectories live.

Arms (all views on n_changed, ham_sum; N = n_pages, BITS_P = 8*page_size):
  apf        = n_changed / N                       (E0, the floor)
  wapf       = ham_sum   / (N * BITS_P)            (E1, breadth x depth)
  intensity  = ham_sum   / (n_changed * BITS_P)    (E2's 2nd channel; 0 if none)
E0 = apf features; E1 = wapf features; E2 = apf (+) intensity; E2' = intensity.

8 features per series, all scale-equivariant (gate B1-G5): mean, std, cov=std/mean,
median, max, p95, peak2med=max/median, duty=fraction of samples > 0.1*max.

Output: features_long.csv -- one row per (cell, arm), columns
  cell_id,workload,family,rep,n_pairs,arm,mean,std,cov,median,max,p95,peak2med,duty

Usage: b1_features.py <b1_out_dir> [--min-pairs 50] [--out features_long.csv]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from pathlib import Path

BITS_PER_PAGE = 4096 * 8
ARMS = ("apf", "wapf", "intensity")
FEAT = ("mean", "std", "cov", "median", "max", "p95", "peak2med", "duty")

RUN_RE = re.compile(r"test(\d+)_")


def family_of(wl: str) -> str:
    """Family = the workload name's first token. The naming convention is
    <family>_<detail>..., so a prefix is exact. A SUBSTRING match is not:
    io_read_cache_hit_v2 contains 'cache_' and a substring rule misfiles it as
    cache instead of io (this bit us once -- keep it a prefix)."""
    return wl.split("_", 1)[0].lower() if wl else "other"


def _pct(xs_sorted, q):
    if not xs_sorted:
        return 0.0
    return xs_sorted[min(len(xs_sorted) - 1, int(q * len(xs_sorted)))]


def features(xs) -> dict:
    if not xs:
        return {k: 0.0 for k in FEAT}
    s = sorted(xs)
    mean = st.fmean(xs)
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    med = st.median(xs)
    mx = s[-1]
    thr = 0.1 * mx
    return {
        "mean": mean, "std": sd, "cov": (sd / mean if mean else 0.0),
        "median": med, "max": mx, "p95": _pct(s, 0.95),
        "peak2med": (mx / med if med else 0.0),
        "duty": sum(1 for x in xs if x > thr) / len(xs),
    }


def clean_workload(prov: dict) -> str:
    """Canonical workload name from the source path, independent of what the
    extractor happened to write into provenance. Strips the queue's
    `run_matrix_test<N>_` prefix and the `.npy.substrate_trajectory` suffix, so
    reps of one workload (test12/28/29/30 ...) group together."""
    src = prov.get("source", "") or ""
    base = os.path.basename(src)
    for suf in (".zst", ".gz", ".csv"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    base = re.sub(r"(\.npy)?\.substrate_trajectory$", "", base)
    base = re.sub(r"^run_matrix_test\d+_", "", base)
    return base or prov.get("workload", "?")


def load_cell(cell_dir: Path, min_pairs: int):
    prov = json.loads((cell_dir / "provenance.json").read_text())
    N = prov["n_pages"]
    wl = clean_workload(prov)
    m = RUN_RE.search(prov.get("source", ""))
    run_index = int(m.group(1)) if m else -1
    apf, wapf, inten = [], [], []
    for line in (cell_dir / "b1_trajectory.jsonl").read_text().splitlines():
        if not line:
            continue
        o = json.loads(line)
        if o.get("final"):
            continue
        k = o["n_changed"]; hs = o["ham_sum"]
        apf.append(k / N)
        wapf.append(hs / (N * BITS_PER_PAGE))
        inten.append(hs / (k * BITS_PER_PAGE) if k else 0.0)
    if len(apf) < min_pairs:
        return None
    return {
        "cell_id": cell_dir.name, "workload": wl, "family": family_of(wl),
        "run_index": run_index, "n_pairs": len(apf),
        "series": {"apf": apf, "wapf": wapf, "intensity": inten},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="b1_out dir (holds <cell>/b1_trajectory.jsonl)")
    ap.add_argument("--min-pairs", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (a.root / "features_long.csv")

    cells = []
    dropped = []
    for cd in sorted(a.root.iterdir()):
        if not (cd / "b1_trajectory.jsonl").is_file():
            continue
        c = load_cell(cd, a.min_pairs)
        if c is None:
            dropped.append(cd.name)
        else:
            cells.append(c)

    # rep index: per workload, rank by run_index (so test28/29/30 -> rep 1/2/3)
    by_wl: dict[str, list] = {}
    for c in cells:
        by_wl.setdefault(c["workload"], []).append(c)
    for wl, group in by_wl.items():
        for rep, c in enumerate(sorted(group, key=lambda x: x["run_index"]), start=1):
            c["rep"] = rep

    cols = ["cell_id", "workload", "family", "rep", "n_pairs", "arm", *FEAT]
    with out.open("w") as f:
        f.write(",".join(cols) + "\n")
        for c in sorted(cells, key=lambda x: (x["family"], x["workload"], x["rep"])):
            for arm in ARMS:
                ft = features(c["series"][arm])
                row = [c["cell_id"], c["workload"], c["family"], c["rep"],
                       c["n_pairs"], arm, *[f"{ft[k]:.9g}" for k in FEAT]]
                f.write(",".join(str(x) for x in row) + "\n")

    # readout: per workload, mean-of-series for apf vs wapf (do the arms disagree?)
    print(f"[b1_features] {len(cells)} cells, {len(by_wl)} workloads -> {out}")
    if dropped:
        print(f"[b1_features] dropped {len(dropped)} below --min-pairs {a.min_pairs}: {dropped}")
    print(f"\n{'workload':32s} {'reps':>4s} {'mean_APF':>10s} {'mean_wAPF':>11s} {'ratio w/A':>9s}")
    for wl in sorted(by_wl, key=lambda w: (family_of(w), w)):
        grp = by_wl[wl]
        a_mean = st.fmean(st.fmean(c["series"]["apf"]) for c in grp)
        w_mean = st.fmean(st.fmean(c["series"]["wapf"]) for c in grp)
        ratio = (w_mean / a_mean) if a_mean else 0.0
        print(f"{wl:32s} {len(grp):>4d} {a_mean:>10.6f} {w_mean:>11.8f} {ratio:>9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
