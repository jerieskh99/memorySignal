#!/usr/bin/env python3
"""Behavioral-family classification on the Plan-05 sweep (5-class), leakage-aware.

Mid-granularity rung of the identification ladder, between the binary
phasic-vs-steady (paperC leakage_ablation.py) and the 11-way instance ID
(plan04_classify.py). Groups the 11 workloads into 5 behavioral families and asks
how well APF features recover the family -- audited for the same feature leakage
paperC found (coverage_ratio is a per-family constant; cv_workingset/f1_phase are
family-conditional), under both cross-validation protocols:

  LORO  leave-one-replicate-out (GroupKFold by replicate) -- optimistic; a
        workload appears in train and test, so the model can memorize it.
  LOWO  leave-one-workload-out (LeaveOneGroupOut by workload) -- the honest
        generalization test. Singleton families (scanner, app: one workload
        each) are unpredictable here by construction; reported, not hidden.

Run on FULL (10 feat, reproduces leakage) and AGNOSTIC (7 feat, leakage-free).

    python3 plan05_campaign/behavior_families.py <sweep.csv> > behavior_families.json

Reuses the exact (8,4)+ok filter and feature lists from paperC's
leakage_ablation.py so the numbers are directly comparable.
"""
import csv
import json
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import sklearn

DEFAULT_SWEEP = "plan05_campaign/downstream/sweep.csv"
FULL = ["apf_mean", "apf_std", "cv_workingset", "f1_phase", "cepstral_peak_idx",
        "ceps_peak_snr_db", "stat_pass_frac", "n_pairs", "n_windows",
        "coverage_ratio"]
FAMILY_CONDITIONAL = ["cv_workingset", "f1_phase", "coverage_ratio"]
AGNOSTIC = [c for c in FULL if c not in FAMILY_CONDITIONAL]

# 5-class behavioral-family taxonomy (substring match on workload name).
FAMILY_MAP = [
    ("ransomware", ("ransom",)),
    ("scanner",    ("scanner",)),
    ("mem_sweep",  ("workingset", "rmw_intensity", "writemag")),
    ("mem_fault",  ("pagefault", "mmap")),
    ("app",        ("hashtable", "app_")),
]


def family_of(workload: str) -> str:
    w = workload.lower()
    for fam, keys in FAMILY_MAP:
        if any(k in w for k in keys):
            return fam
    return "other"


def _f(r, c):
    try:
        return float(r.get(c, ""))
    except ValueError:
        return np.nan


def _pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=300,
                                                   random_state=0))])


def _scores(y, pred, labels):
    acc = float((pred == y).mean())
    macro_f1 = float(f1_score(y, pred, labels=labels, average="macro",
                              zero_division=0))
    cm = confusion_matrix(y, pred, labels=labels).tolist()
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "labels": labels, "confusion": cm}


def run_loro(X, y, rep, labels):
    gkf = GroupKFold(n_splits=len(set(rep)))
    pred = cross_val_predict(_pipe(), X, y, groups=rep, cv=gkf)
    return _scores(y, pred, labels)


def run_lowo(X, y, wl, labels):
    logo = LeaveOneGroupOut()
    pred = cross_val_predict(_pipe(), X, y, groups=wl, cv=logo)
    out = _scores(y, pred, labels)
    # per-workload accuracy (how each held-out workload's cells fared)
    per_wl = {}
    for w in sorted(set(wl)):
        m = wl == w
        per_wl[w] = {"family": y[m][0], "n": int(m.sum()),
                     "acc": round(float((pred[m] == y[m]).mean()), 4)}
    out["per_workload"] = per_wl
    return out


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SWEEP
    rows = [r for r in csv.DictReader(open(path))
            if r["window"] == "8" and r["hop"] == "4"
            and r["status"].startswith("ok")]
    y = np.array([family_of(r["workload"]) for r in rows])
    rep = np.array([r["replicate"] for r in rows])
    wl = np.array([r["workload"] for r in rows])
    labels = [f for f, _ in FAMILY_MAP]
    # majority baseline
    vals, cnts = np.unique(y, return_counts=True)
    maj = float(cnts.max() / cnts.sum())

    out = {
        "schema": "plan05.behavior_families.v1",
        "sklearn_version": sklearn.__version__,
        "n_cells": len(rows),
        "n_classes": len(labels),
        "labels": labels,
        "class_counts": {v: int(c) for v, c in zip(vals, cnts)},
        "majority_baseline": round(maj, 4),
        "singleton_families_unpredictable_under_lowo":
            [f for f in labels
             if len({r["workload"] for r in rows if family_of(r["workload"]) == f}) == 1],
        "runs": {},
    }
    for name, cols in (("FULL", FULL), ("AGNOSTIC", AGNOSTIC)):
        X = np.array([[_f(r, c) for c in cols] for r in rows])
        out["runs"][name] = {
            "features": cols,
            "loro": run_loro(X, y, rep, labels),
            "lowo": run_lowo(X, y, wl, labels),
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
