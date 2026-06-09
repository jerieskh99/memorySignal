#!/usr/bin/env python3
"""Plan 06 -- does the disk-I/O channel close the APF blind spot?

Runs after the subset capture (apf + diskio trajectories adapted into a cells-dir
and run through plan03_sweep). Compares three feature sets under the Plan 05
honest CV harness:

  AGNOSTIC               -- leakage-free APF features (Plan 05)
  +peakvar               -- + APF shape features (Plan 05)
  +peakvar+diskio        -- + disk-I/O rate features (Plan 06)

Reports, for binary threat/benign: leave-one-workload-out (LOWO) and
leave-one-FAMILY-out (LOFO, the novel-family test) per feature set, plus the
direct masquerade-pair check -- the write rate of slowburn (threat) vs pagefault
(benign), the pair APF alone cannot separate.

    python3 plan06_campaign/diskio_lift.py [sweep.csv] [cells_dir] > diskio_lift.json
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "plan05_campaign"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(P5))
import peakvar_lift as pv                       # noqa: E402  (classify, classify_lofo)
from extra_features import load_extra, EXTRA    # noqa: E402
from behavior_families import AGNOSTIC, family_of, _f  # noqa: E402
from diskio_features import load_diskio, DISKIO_FEATS   # noqa: E402

THREAT_KEYS = ("ransom", "scanner")


def is_threat(w): return any(k in w.lower() for k in THREAT_KEYS)


def main() -> int:
    sweep = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "downstream" / "sweep.csv")
    cells = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "downstream" / "cells")
    rows = [r for r in csv.DictReader(open(sweep))
            if r["window"] == "8" and r["hop"] == "4" and r["status"].startswith("ok")]
    extra = load_extra(cells)
    diskio = load_diskio(cells)
    if not diskio:
        print(json.dumps({"error": "no diskio_trajectory found in cells-dir; "
                          "was the capture run with CAPTURE_DISKIO=1 and adapted "
                          "via build_cells_dir.py?"}, indent=1))
        return 1

    wl = np.array([r["workload"] for r in rows])
    rep = np.array([r["replicate"] for r in rows])
    fam = np.array([family_of(w) for w in wl])
    ybin = np.array([1 if is_threat(w) else 0 for w in wl])

    def matrix(use_pv, use_diskio):
        M = []
        for r in rows:
            row = [_f(r, c) for c in AGNOSTIC]
            if use_pv:
                e = extra.get(r["cell_id"], {})
                row += [e.get(k, 0.0) for k in EXTRA]
            if use_diskio:
                d = diskio.get(r["cell_id"], {})
                row += [d.get(k, 0.0) for k in DISKIO_FEATS]
            M.append(row)
        return np.array(M)

    out = {"schema": "plan06.diskio_lift.v1", "n_cells": len(rows),
           "diskio_features": DISKIO_FEATS, "results": {}}
    for name, (use_pv, use_dk) in (("AGNOSTIC", (False, False)),
                                   ("AGNOSTIC+peakvar", (True, False)),
                                   ("AGNOSTIC+peakvar+diskio", (True, True))):
        X = matrix(use_pv, use_dk)
        out["results"][name] = {
            "binary_lowo": pv.classify(X, ybin, rep, wl, [0, 1]),
            "binary_lofo": pv.classify_lofo(X, ybin, fam, wl, [0, 1]),
        }

    # Direct masquerade-pair check: write rate (MB/s) per workload.
    def wr_mean(name):
        v = [diskio[r["cell_id"]]["wr_rate_mean_mbs"]
             for r in rows if r["workload"] == name and r["cell_id"] in diskio]
        return round(float(np.mean(v)), 4) if v else None

    out["masquerade_pair_wr_rate_mbs"] = {
        w: wr_mean(w) for w in sorted(set(wl))
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
