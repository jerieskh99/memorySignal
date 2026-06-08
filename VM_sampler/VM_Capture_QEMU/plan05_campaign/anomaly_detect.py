#!/usr/bin/env python3
"""Anomaly / novelty detection track on the Plan-05 sweep (separate from the
supervised behavioral-ID ladder).

Framing: a deployed monitor sees only BENIGN behavior at train time and must flag
a never-before-seen workload as anomalous. We train one-class novelty detectors
on the benign (steady) cells using leakage-free AGNOSTIC features, then ask how
well they flag the threat (phasic) cells as anomalies. Reported threshold-free as
ROC-AUC, plus per-workload mean anomaly score to expose WHERE the overlap is.

Two protocols:
  in_sample  -- fit on all benign, score everything (benign scores are in-sample
                -> optimistic upper bound).
  honest     -- leave-one-benign-workload-out: each benign workload's cells are
                scored by a model trained on the OTHER benign workloads; threats
                are scored by the all-benign model. Benign scores are now
                out-of-sample, so the AUC is not inflated by benign self-fit.

Leakage control: AGNOSTIC features only. coverage_ratio (a per-family constant)
is excluded -- otherwise a one-class model trivially flags every phasic cell.

    python3 plan05_campaign/anomaly_detect.py <sweep.csv> > anomaly_detect.json
"""
import csv
import json
import sys
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sklearn

DEFAULT_SWEEP = "plan05_campaign/downstream/sweep.csv"
AGNOSTIC = ["apf_mean", "apf_std", "cepstral_peak_idx", "ceps_peak_snr_db",
            "stat_pass_frac", "n_pairs", "n_windows"]
THREAT_KEYS = ("ransom", "scanner")   # phasic == threat for this workload set


def is_threat(w: str) -> bool:
    return any(k in w.lower() for k in THREAT_KEYS)


def _f(r, c):
    try:
        return float(r.get(c, ""))
    except ValueError:
        return np.nan


def make(name):
    if name == "IsolationForest":
        return IsolationForest(n_estimators=300, random_state=0)
    if name == "OneClassSVM":
        return OneClassSVM(nu=0.1, gamma="scale")
    if name == "LOF(novelty)":
        return LocalOutlierFactor(n_neighbors=10, novelty=True)
    raise ValueError(name)


def anomaly(model, Xtrain, Xscore):
    """Fit on Xtrain (benign), return anomaly score for Xscore (higher = more
    anomalous). sklearn score_samples: higher = more normal, so negate."""
    model.fit(Xtrain)
    return -model.score_samples(Xscore)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SWEEP
    rows = [r for r in csv.DictReader(open(path))
            if r["window"] == "8" and r["hop"] == "4"
            and r["status"].startswith("ok")]
    y = np.array([1 if is_threat(r["workload"]) else 0 for r in rows])  # 1=threat
    wl = np.array([r["workload"] for r in rows])
    X = np.array([[_f(r, c) for c in AGNOSTIC] for r in rows])
    benign = (y == 0)

    # Scale on benign stats (impute median, standardize). Scaling is not the
    # discriminator, so fitting on all benign is acceptable.
    imp = SimpleImputer(strategy="median").fit(X[benign])
    sc = StandardScaler().fit(imp.transform(X[benign]))
    Xs = sc.transform(imp.transform(X))

    out = {
        "schema": "plan05.anomaly_detect.v1",
        "sklearn_version": sklearn.__version__,
        "n_cells": len(rows), "n_benign": int(benign.sum()),
        "n_threat": int((y == 1).sum()),
        "features": AGNOSTIC,
        "protocol": "one-class novelty trained on benign(steady) AGNOSTIC features; "
                    "ROC-AUC for flagging threats(phasic). 1=threat/anomaly.",
        "methods": {},
    }
    benign_wls = sorted(set(wl[benign]))
    for name in ("IsolationForest", "OneClassSVM", "LOF(novelty)"):
        # in-sample
        s_in = anomaly(make(name), Xs[benign], Xs)
        auc_in = float(roc_auc_score(y, s_in))
        # honest: leave-one-benign-workload-out for benign; all-benign for threat
        s_h = np.empty(len(rows), dtype=float)
        s_h[y == 1] = anomaly(make(name), Xs[benign], Xs[y == 1])
        for w in benign_wls:
            tr = benign & (wl != w)
            te = (wl == w)
            s_h[te] = anomaly(make(name), Xs[tr], Xs[te])
        auc_h = float(roc_auc_score(y, s_h))
        # per-workload mean in-sample anomaly score (structure)
        per_wl = {w: round(float(s_in[wl == w].mean()), 3) for w in sorted(set(wl))}
        out["methods"][name] = {
            "roc_auc_in_sample": round(auc_in, 4),
            "roc_auc_honest_lowo_benign": round(auc_h, 4),
            "mean_anomaly_score_by_workload": per_wl,
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
