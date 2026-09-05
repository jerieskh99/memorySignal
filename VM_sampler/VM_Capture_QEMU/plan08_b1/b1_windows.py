#!/usr/bin/env python3
"""b1_windows.py -- slice the APF/wAPF trajectories into 8/4 windows.

Reads each cell's b1_trajectory.jsonl, rebuilds the two arm series
(apf = n_changed/N, wapf = ham_sum/(N*BITS_P)), and slides a window of W with
hop H (default 8/4, the inherited Plan-03 setting) over each. Emits two
row-aligned matrices -- X_apf and X_wapf, both [n_windows x W] -- plus a label
per window (cell/workload/family/rep) needed for the grouped CV splits.

Values are RAW. Standardization is per-fold in the model step, never here: doing
it now would fit the scaler on test windows and leak across the split.

Output: <root>/b1_windows.npz with arrays
  X_apf, X_wapf         float64 [n_windows x W]   (row-aligned; same windows)
  cell_id, workload, family   str  [n_windows]
  rep, win_start              int  [n_windows]
  window, hop                 scalars
Also writes b1_windows_meta.csv for inspection.

Usage: b1_windows.py <b1_out_dir> [--window 8] [--hop 4] [--min-pairs 50]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b1_features import clean_workload, family_of, RUN_RE   # noqa: E402  (shared label logic)

BITS_PER_PAGE = 4096 * 8


def cell_series(cd: Path):
    prov = json.loads((cd / "provenance.json").read_text())
    N = prov["n_pages"]
    m = RUN_RE.search(prov.get("source", ""))
    run_index = int(m.group(1)) if m else -1
    wl = clean_workload(prov)
    apf, wapf = [], []
    for line in (cd / "b1_trajectory.jsonl").read_text().splitlines():
        if not line:
            continue
        o = json.loads(line)
        if o.get("final"):
            continue
        k = o["n_changed"]; hs = o["ham_sum"]
        apf.append(k / N)
        wapf.append(hs / (N * BITS_PER_PAGE))
    return {"cell_id": cd.name, "workload": wl, "family": family_of(wl),
            "run_index": run_index, "apf": apf, "wapf": wapf}


def slide(series, W, H):
    return [series[i:i + W] for i in range(0, len(series) - W + 1, H)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--hop", type=int, default=4)
    ap.add_argument("--min-pairs", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (a.root / "b1_windows.npz")
    W, H = a.window, a.hop

    cells = []
    for cd in sorted(a.root.iterdir()):
        if not (cd / "b1_trajectory.jsonl").is_file():
            continue
        c = cell_series(cd)
        if len(c["apf"]) >= a.min_pairs:
            cells.append(c)

    # rep index: per workload, rank by run_index (test12/28/29/30 -> rep 1/2/3/4)
    by_wl = collections.defaultdict(list)
    for c in cells:
        by_wl[c["workload"]].append(c)
    for g in by_wl.values():
        for rep, c in enumerate(sorted(g, key=lambda x: x["run_index"]), 1):
            c["rep"] = rep

    Xa, Xw = [], []
    cid, wl, fam, rep, wstart = [], [], [], [], []
    for c in cells:
        wa = slide(c["apf"], W, H)
        ww = slide(c["wapf"], W, H)   # identical count -> row-aligned
        for j, (va, vw) in enumerate(zip(wa, ww)):
            Xa.append(va); Xw.append(vw)
            cid.append(c["cell_id"]); wl.append(c["workload"]); fam.append(c["family"])
            rep.append(c["rep"]); wstart.append(j * H)

    Xa = np.asarray(Xa, dtype=np.float64)
    Xw = np.asarray(Xw, dtype=np.float64)
    np.savez(out, X_apf=Xa, X_wapf=Xw,
             cell_id=np.asarray(cid), workload=np.asarray(wl),
             family=np.asarray(fam), rep=np.asarray(rep, dtype=np.int32),
             win_start=np.asarray(wstart, dtype=np.int32),
             window=W, hop=H)

    meta_csv = out.with_name("b1_windows_meta.csv")
    with meta_csv.open("w") as f:
        f.write("row,cell_id,workload,family,rep,win_start\n")
        for i in range(len(cid)):
            f.write(f"{i},{cid[i]},{wl[i]},{fam[i]},{rep[i]},{wstart[i]}\n")

    n = Xa.shape[0]
    print(f"[b1_windows] {len(cells)} cells -> {n} windows x {W}  (X_apf and X_wapf, W={W} H={H})")
    print(f"  -> {out}")
    print(f"  -> {meta_csv}")
    fc = collections.Counter(fam)
    wc = collections.Counter(wl)
    print(f"  {len(wc)} workloads, {len(fc)} families; windows per family:")
    for k in sorted(fc):
        print(f"    {k:10s} {fc[k]:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
