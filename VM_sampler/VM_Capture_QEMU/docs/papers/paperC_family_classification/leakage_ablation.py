#!/usr/bin/env python3
"""Leakage ablation for the phasic-vs-steady binary classifier (Paper C).

Motivation: peer review flagged that the headline 100% binary accuracy may be
driven by features that are computed *per workload family* and therefore encode
the label rather than the memory behavior. This script quantifies that.

It re-runs the binary (phasic vs steady) classification on the Plan-03 sweep at
(window, hop) = (8, 4), status ok, under the same leave-one-replicate-out
GroupKFold protocol as plan04_classify.py, for several feature subsets:

  FULL      -- all 10 features (reproduces the reported 1.000).
  AGNOSTIC  -- drop the three family-conditional features
               (cv_workingset, f1_phase, coverage_ratio).
  single    -- coverage_ratio alone; ceps_peak_snr_db alone.
  apf_only  -- apf_mean + apf_std only.

Why those three are "family-conditional":
  * cv_workingset  is populated only for steady cells (NaN for all phasic).
  * f1_phase       is populated only for phasic cells (NaN for all steady).
  * coverage_ratio is assigned 0.133 to every steady cell and 0.2 to every
    phasic cell by the analyzer, i.e. it is a constant per family.

Usage:
  python3 leakage_ablation.py [path/to/sweep.csv] > leakage_ablation.json
Default sweep path is the v3 combined dataset relative to the repo root.
"""
import csv, json, sys, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn

DEFAULT_SWEEP = "../../../../../_v3_combined/plan03/sweep.csv"
PHASIC = {"sandbox_ransom_batched", "sandbox_ransom_selective",
          "sandbox_ransom_seq", "sandbox_ransom_slowburn",
          "sandbox_scanner_metadata"}
FULL = ["apf_mean", "apf_std", "cv_workingset", "f1_phase", "cepstral_peak_idx",
        "ceps_peak_snr_db", "stat_pass_frac", "n_pairs", "n_windows",
        "coverage_ratio"]
FAMILY_CONDITIONAL = ["cv_workingset", "f1_phase", "coverage_ratio"]
AGNOSTIC = [c for c in FULL if c not in FAMILY_CONDITIONAL]


def _f(r, c):
    try:
        return float(r.get(c, ""))
    except ValueError:
        return np.nan


def run(rows, cols, y, grp):
    X = np.array([[_f(r, c) for c in cols] for r in rows])
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=300,
                                                   random_state=0))])
    gkf = GroupKFold(n_splits=len(set(grp)))
    pred = cross_val_predict(pipe, X, y, groups=grp, cv=gkf)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"features": cols, "n_features": len(cols),
            "accuracy": round(float((pred == y).mean()), 4),
            "confusion": {"tn_steady": tn, "fp": fp, "fn": fn, "tp_phasic": tp}}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SWEEP
    rows = [r for r in csv.DictReader(open(path))
            if r["window"] == "8" and r["hop"] == "4"
            and r["status"].startswith("ok")]
    y = np.array([1 if r["workload"] in PHASIC else 0 for r in rows])
    grp = np.array([int(r["replicate"]) for r in rows])
    out = {
        "schema": "plan04.leakage_ablation.v1",
        "sklearn_version": sklearn.__version__,
        "protocol": "GroupKFold leave-one-replicate-out, RandomForest(n=300), median-impute+standardize",
        "n_cells": len(rows), "n_phasic": int(y.sum()), "n_steady": int((y == 0).sum()),
        "majority_baseline": round(float(max((y == 0).mean(), (y == 1).mean())), 4),
        "clopper_pearson_95_lower_for_full": round(0.025 ** (1.0 / len(rows)), 4),
        "runs": {
            "FULL_10": run(rows, FULL, y, grp),
            "AGNOSTIC_no_family_conditional": run(rows, AGNOSTIC, y, grp),
            "coverage_ratio_alone": run(rows, ["coverage_ratio"], y, grp),
            "ceps_snr_alone": run(rows, ["ceps_peak_snr_db"], y, grp),
            "apf_mean_std_only": run(rows, ["apf_mean", "apf_std"], y, grp),
        },
        "family_conditional_features": FAMILY_CONDITIONAL,
        "interpretation": (
            "The reported 1.000 binary accuracy is fully reproduced by "
            "coverage_ratio alone (0.133 steady vs 0.2 phasic, assigned per "
            "family). Removing the three family-conditional features drops "
            "accuracy to the AGNOSTIC value, near the majority baseline. The "
            "clean separation is therefore largely feature-construction "
            "leakage, not behavioral discrimination."),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
