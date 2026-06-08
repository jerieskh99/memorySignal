#!/usr/bin/env python3
"""Measure the lift from adding windowed peak/variance features (extra_features)
to the leakage-free AGNOSTIC set, across the three tasks, with honest CV.

Compares AGNOSTIC vs AGNOSTIC+peakvar on:
  * binary phasic/steady (= threat/benign)   -- RandomForest, LORO + LOWO
  * 5-class behavioral family                -- RandomForest, LORO + LOWO
  * anomaly novelty (one-class on benign)    -- ROC-AUC, honest LOWO-benign
Per-workload LOWO accuracy is reported so the stealth threats (slowburn, batched)
and the benign quiet (pagefault) are visible.

    python3 plan05_campaign/peakvar_lift.py [sweep.csv] [cells_dir] > peakvar_lift.json
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import sklearn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extra_features import load_extra, EXTRA          # noqa: E402
from behavior_families import family_of, AGNOSTIC, _f  # noqa: E402

THREAT_KEYS = ("ransom", "scanner")
FAMS = ["ransomware", "scanner", "mem_sweep", "mem_fault", "app"]


def is_threat(w): return any(k in w.lower() for k in THREAT_KEYS)


def _rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=300,
                                                   random_state=0))])


def classify(X, y, rep, wl, labels):
    pred_lo = cross_val_predict(_rf(), X, y, groups=rep,
                                cv=GroupKFold(n_splits=len(set(rep))))
    pred_wo = cross_val_predict(_rf(), X, y, groups=wl, cv=LeaveOneGroupOut())
    perwl = {w: round(float((pred_wo[wl == w] == y[wl == w]).mean()), 3)
             for w in sorted(set(wl))}
    return {
        "loro_acc": round(float((pred_lo == y).mean()), 4),
        "lowo_acc": round(float((pred_wo == y).mean()), 4),
        "lowo_macro_f1": round(float(f1_score(y, pred_wo, labels=labels,
                                              average="macro", zero_division=0)), 4),
        "lowo_per_workload": perwl,
    }


def anomaly_auc(Xs, y, wl, benign):
    benign_wls = sorted(set(wl[benign]))
    makers = {
        "IsolationForest": lambda: IsolationForest(n_estimators=300, random_state=0),
        "LOF": lambda: LocalOutlierFactor(n_neighbors=10, novelty=True),
        "OCSVM": lambda: OneClassSVM(nu=0.1, gamma="scale"),
    }
    res = {}
    for name, mk in makers.items():
        s = np.empty(len(y), dtype=float)
        m = mk(); m.fit(Xs[benign]); s[y == 1] = -m.score_samples(Xs[y == 1])
        for w in benign_wls:
            tr = benign & (wl != w); te = (wl == w)
            mm = mk(); mm.fit(Xs[tr]); s[te] = -mm.score_samples(Xs[te])
        res[name] = round(float(roc_auc_score(y, s)), 4)
    return res


def main() -> int:
    sweep = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "downstream" / "sweep.csv")
    cells = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "downstream" / "cells")
    rows = [r for r in csv.DictReader(open(sweep))
            if r["window"] == "8" and r["hop"] == "4" and r["status"].startswith("ok")]
    extra = load_extra(cells)
    wl = np.array([r["workload"] for r in rows])
    rep = np.array([r["replicate"] for r in rows])
    ybin = np.array([1 if is_threat(r["workload"]) else 0 for r in rows])
    yfam = np.array([family_of(r["workload"]) for r in rows])

    def matrix(with_pv):
        base = [[_f(r, c) for c in AGNOSTIC] for r in rows]
        if with_pv:
            for i, r in enumerate(rows):
                e = extra.get(r["cell_id"], {})
                base[i] = base[i] + [e.get(k, 0.0) for k in EXTRA]
        return np.array(base)

    out = {"schema": "plan05.peakvar_lift.v1", "sklearn": sklearn.__version__,
           "extra_features": EXTRA, "n_cells": len(rows),
           "agnostic_features": AGNOSTIC, "results": {}}
    for setname, with_pv in (("AGNOSTIC", False), ("AGNOSTIC+peakvar", True)):
        X = matrix(with_pv)
        benign = (ybin == 0)
        imp = SimpleImputer(strategy="median").fit(X[benign])
        sc = StandardScaler().fit(imp.transform(X[benign]))
        Xs = sc.transform(imp.transform(X))
        out["results"][setname] = {
            "binary": classify(X, ybin, rep, wl, [0, 1]),
            "family5": classify(X, yfam, rep, wl, FAMS),
            "anomaly_roc_auc_honest": anomaly_auc(Xs, ybin, wl, benign),
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
