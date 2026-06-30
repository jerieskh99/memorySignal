#!/usr/bin/env python3
"""Plan 07 -- workload-level separability matrix, blocked by family (DAG node: sep-matrix).

The master artifact of the recogniser pilot. For each cell we build a feature vector,
standardize, take the per-workload centroid, and compute the pairwise centroid distance.
Ordered by family, the matrix shows:
  - diagonal blocks  = within-family cohesion  (small distance => members resemble each other)
  - off-diagonal     = between-family separation
  - low CROSS-family cells = masquerades (a threat that aliases a benign, e.g. slowburn/pagefault)

Run three feature views so we can see WHICH feature type resolves which pair:
  magnitude        = apf_mean, apf_std, apf_max, apf_p95      (level / loudness)
  scale_invariant  = apf_cov, peak2med, duty_gt05            (shape, unitless)
  combined         = AGNOSTIC (7) + peakvar (5)

This is offline and reuses the existing feature extractors; it needs NO new capture.

    python3 plan07_campaign/separability_matrix.py [sweep.csv] [cells_dir] > sep_matrix.json

Defaults to the full 66-cell two-channel run.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "plan05_campaign"
sys.path.insert(0, str(P5))
from behavior_families import family_of, AGNOSTIC, _f          # noqa: E402
from extra_features import load_extra, EXTRA                    # noqa: E402

MAGNITUDE = ["apf_mean", "apf_std", "apf_max", "apf_p95"]
SCALE_INVARIANT = ["apf_cov", "peak2med", "duty_gt05"]
FAM_ORDER = ["ransomware", "scanner", "mem_sweep", "mem_fault", "app"]
THREAT = ("ransom", "scanner")


def is_threat(w):
    return any(k in w.lower() for k in THREAT)


def _standardize(X):
    """median-impute then z-score each column; constant columns -> 0."""
    X = X.astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col) if np.any(~np.isnan(col)) else 0.0
        col[np.isnan(col)] = med
        X[:, j] = col
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _view(cells, feat_names):
    """centroid distance matrix for one feature view. cells: list of dicts."""
    wls = sorted({c["workload"] for c in cells}, key=lambda w: (FAM_ORDER.index(family_of(w)), w))
    X = _standardize(np.array([[c["feat"][f] for f in feat_names] for c in cells]))
    cent = {}
    for w in wls:
        idx = [i for i, c in enumerate(cells) if c["workload"] == w]
        cent[w] = X[idx].mean(axis=0)
    n = len(wls)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = float(np.linalg.norm(cent[wls[i]] - cent[wls[j]]))
    # within / between family stats + per-family cohesion (between/within ratio)
    fam = [family_of(w) for w in wls]
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if fam[i] == fam[j] else between).append(M[i, j])
    per_family = {}
    for f in sorted(set(fam)):
        members = [i for i in range(n) if fam[i] == f]
        if len(members) >= 2:
            w_d = np.mean([M[a, b] for a in members for b in members if a < b])
        else:
            w_d = float("nan")  # family-of-one: within-distance undefined
        others = [i for i in range(n) if fam[i] != f]
        b_d = np.mean([M[a, b] for a in members for b in others]) if others else float("nan")
        per_family[f] = {"n_workloads": len(members),
                         "within_mean": None if np.isnan(w_d) else round(float(w_d), 4),
                         "between_mean": None if np.isnan(b_d) else round(float(b_d), 4),
                         "cohesion_ratio": None if (np.isnan(w_d) or w_d == 0) else round(float(b_d / w_d), 3)}
    # masquerades = smallest CROSS-family distances
    cross = sorted(((M[i, j], wls[i], wls[j]) for i in range(n) for j in range(i + 1, n)
                    if fam[i] != fam[j]), key=lambda t: t[0])[:6]
    return {
        "order": wls, "families": fam,
        "matrix": [[round(float(v), 4) for v in row] for row in M],
        "within_family_mean": round(float(np.mean(within)), 4) if within else None,
        "between_family_mean": round(float(np.mean(between)), 4) if between else None,
        "per_family": per_family,
        "top_masquerades": [{"a": a, "b": b, "distance": round(float(d), 4)} for d, a, b in cross],
    }


def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else str(P5 / "downstream" / "sweep.csv")
    cells = sys.argv[2] if len(sys.argv) > 2 else str(P5 / "downstream" / "cells")
    rows = [r for r in csv.DictReader(open(sweep))
            if r["window"] == "8" and r["hop"] == "4" and r["status"].startswith("ok")]
    extra = load_extra(cells)

    cells_data = []
    for r in rows:
        e = extra.get(r["cell_id"], {})
        feat = {f: _f(r, f) for f in AGNOSTIC}                  # AGNOSTIC from the sweep
        for f in EXTRA:                                        # peakvar from extra_features
            feat[f] = e.get(f, float("nan"))
        feat["apf_mean"] = _f(r, "apf_mean")
        feat["apf_std"] = _f(r, "apf_std")
        cells_data.append({"workload": r["workload"], "cell_id": r["cell_id"], "feat": feat})

    views = {
        "magnitude": _view(cells_data, MAGNITUDE),
        "scale_invariant": _view(cells_data, SCALE_INVARIANT),
        "combined": _view(cells_data, list(AGNOSTIC) + list(EXTRA)),
    }
    out = {"schema": "plan07.separability_matrix.v1", "n_cells": len(rows),
           "magnitude_features": MAGNITUDE, "scale_invariant_features": SCALE_INVARIANT,
           "views": views}
    print(json.dumps(out, indent=1))

    # readable summary to stderr
    for name, v in views.items():
        sys.stderr.write(f"\n=== {name} ===  within-family {v['within_family_mean']} | "
                         f"between-family {v['between_family_mean']}\n")
        for f, s in v["per_family"].items():
            sys.stderr.write(f"  {f:11} n={s['n_workloads']} cohesion_ratio={s['cohesion_ratio']}\n")
        sys.stderr.write("  top masquerade (lowest cross-family distance): "
                         + ", ".join(f"{m['a']}~{m['b']}={m['distance']}" for m in v["top_masquerades"][:3]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
