#!/usr/bin/env python3
"""Normal-behavior baseline / profiler from the existing benign cells (offline,
no new captures).

Treats normal behavior as a first-class deliverable rather than just the backdrop
for anomaly detection. From the 36 benign (steady) cells it builds:

  1. Per-behavior fingerprint -- for each benign workload, the median and p5-p95
     band of each leakage-free feature (the "what this normally looks like" card).
  2. Normal envelope -- per-feature benign p5-p95 band = the normal operating
     region, plus a simple INTERPRETABLE deviation detector (count of features a
     cell falls outside the band) whose threat/benign ROC-AUC complements the ML
     one-class detector.
  3. Masquerade map -- for each threat cell, the nearest benign behavior in
     standardized feature space (what a threat hides as).

Scope (honest): "normal" = the 6 benign microbenchmarks, APF-only. This is a
baseline over our benign set, not a real production-traffic profile.

    python3 plan05_campaign/normal_profile.py [sweep.csv] [cells_dir] > normal_profile.json
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extra_features import load_extra   # noqa: E402

THREAT_KEYS = ("ransom", "scanner")
# Leakage-free profile features: 5 from the sweep + 5 trajectory-shape features.
SWEEP_FEATS = ["apf_mean", "apf_std", "ceps_peak_snr_db", "stat_pass_frac", "n_windows"]
EXTRA_FEATS = ["apf_cov", "apf_max", "apf_p95", "peak2med", "duty_gt05"]
FEATS = ["apf_mean", "apf_std", "apf_cov", "apf_max", "apf_p95", "peak2med",
         "duty_gt05", "ceps_peak_snr_db", "stat_pass_frac", "n_windows"]


def is_threat(w): return any(k in w.lower() for k in THREAT_KEYS)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def pct(xs, q):
    xs = sorted(x for x in xs if x == x)  # drop NaN
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def median(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main() -> int:
    sweep = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "downstream" / "sweep.csv")
    cells = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "downstream" / "cells")
    rows = [r for r in csv.DictReader(open(sweep))
            if r["window"] == "8" and r["hop"] == "4" and r["status"].startswith("ok")]
    extra = load_extra(cells)

    cells_feat = []
    for r in rows:
        d = {f: _f(r.get(f)) for f in SWEEP_FEATS}
        e = extra.get(r["cell_id"], {})
        d.update({f: _f(e.get(f)) for f in EXTRA_FEATS})
        d["workload"] = r["workload"]
        d["threat"] = is_threat(r["workload"])
        cells_feat.append(d)
    benign = [c for c in cells_feat if not c["threat"]]
    threat = [c for c in cells_feat if c["threat"]]

    # 1. per-behavior fingerprint
    fingerprint = {}
    for w in sorted(set(c["workload"] for c in benign)):
        wc = [c for c in benign if c["workload"] == w]
        fingerprint[w] = {f: {"median": round(median([c[f] for c in wc]), 4),
                              "p5": round(pct([c[f] for c in wc], 0.05), 4),
                              "p95": round(pct([c[f] for c in wc], 0.95), 4)}
                          for f in FEATS}

    # 2. normal envelope (p5-p95 across all benign) + interpretable deviation count
    env = {f: {"lo": pct([c[f] for c in benign], 0.05),
               "hi": pct([c[f] for c in benign], 0.95)} for f in FEATS}

    def deviation(c):
        return sum(1 for f in FEATS
                   if c[f] == c[f] and not (env[f]["lo"] <= c[f] <= env[f]["hi"]))

    b_dev = [deviation(c) for c in benign]
    t_dev = [deviation(c) for c in threat]
    y = [0] * len(benign) + [1] * len(threat)
    scores = b_dev + t_dev
    env_auc = float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None

    # 3. masquerade map: standardize on benign, nearest benign-workload centroid
    mu = {f: median([c[f] for c in benign]) for f in FEATS}
    sd = {}
    for f in FEATS:
        vals = [c[f] for c in benign if c[f] == c[f]]
        m = sum(vals) / len(vals) if vals else 0.0
        var = sum((v - m) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 1.0
        sd[f] = (var ** 0.5) or 1.0

    def vec(c):
        return np.array([(c[f] - mu[f]) / sd[f] if c[f] == c[f] else 0.0 for f in FEATS])

    centroids = {}
    for w in sorted(set(c["workload"] for c in benign)):
        wc = [vec(c) for c in benign if c["workload"] == w]
        centroids[w] = np.mean(wc, axis=0)

    masquerade = {}
    for w in sorted(set(c["workload"] for c in threat)):
        nearest = []
        for c in [c for c in threat if c["workload"] == w]:
            v = vec(c)
            nearest.append(min(centroids, key=lambda bw: float(np.linalg.norm(v - centroids[bw]))))
        # most common nearest-benign for this threat
        top = max(set(nearest), key=nearest.count)
        masquerade[w] = {"nearest_benign": top,
                         "count": f"{nearest.count(top)}/{len(nearest)}"}

    out = {
        "schema": "plan05.normal_profile.v1",
        "scope": "benign microbenchmarks only, APF-only; baseline over our benign "
                 "set, not real production traffic",
        "n_benign_cells": len(benign), "n_threat_cells": len(threat),
        "features": FEATS,
        "per_behavior_fingerprint": fingerprint,
        "normal_envelope_p5_p95": {f: {"lo": round(env[f]["lo"], 4),
                                       "hi": round(env[f]["hi"], 4)} for f in FEATS},
        "envelope_deviation_detector": {
            "benign_dev_median": median(b_dev), "benign_dev_max": max(b_dev),
            "threat_dev_median": median(t_dev), "threat_dev_min": min(t_dev),
            "roc_auc": round(env_auc, 4) if env_auc is not None else None,
            "note": "deviation = count of the 10 features outside the benign "
                    "p5-p95 band; a simple interpretable complement to the ML "
                    "one-class detector (ROC-AUC ~0.93 with peak/variance).",
        },
        "masquerade_map": masquerade,
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
