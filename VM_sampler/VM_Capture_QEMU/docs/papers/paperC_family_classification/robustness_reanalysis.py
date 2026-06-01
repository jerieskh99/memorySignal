#!/usr/bin/env python3
"""Robustness + significance reanalysis for Paper C / Paper B.

Closes three open items flagged in reviewer_memo_round2:

  R-5  Significance of the leakage-free (AGNOSTIC) binary accuracy vs. the
       majority baseline. The point estimate (0.644 vs 0.542) hides the fact
       that the 4 replicates of a (workload, duration) are NOT independent, so
       a naive n=118 z-test overstates significance. We report:
         (a) an exact McNemar test, AGNOSTIC vs. always-predict-majority, and
         (b) a cluster bootstrap 95% CI for the AGNOSTIC accuracy, resampling
             (workload, duration) clusters -- the genuinely independent unit.

  R-7  Recompute the phasic G3 cepstral-SNR defining metric restricted to cells
       with enough spectral degrees of freedom (n_windows >= 8). The reviewer
       flagged that the median SNR was reported over cells of which 53/132 had
       <= 3 windows at (W,H)=(8,4), so the SNR was computed on too few DOF.

  LOWO Leave-one-workload-out CV (group = workload), for FULL and AGNOSTIC, to
       separate "family signal that generalizes to an unseen workload" from
       "workload memorization". Leave-one-replicate-out (the headline protocol)
       cannot see this because every workload appears in every fold.

Reuses the exact loader, (8,4)+ok filter, feature lists, pipeline, and PHASIC
set from leakage_ablation.py so the AGNOSTIC number reproduces (sanity check).

Usage:
  python3 robustness_reanalysis.py [path/to/sweep.csv] > robustness_reanalysis.json
"""
import csv, json, sys
from math import comb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
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
MIN_WINDOWS_G3 = 8
SNR_THRESHOLD = 4.5


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


def _predict(rows, cols, y, grp, cv):
    X = np.array([[_f(r, c) for c in cols] for r in rows])
    return cross_val_predict(_pipe(), X, y, groups=grp, cv=cv)


def _acc_conf(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"accuracy": round(float((pred == y).mean()), 4),
            "confusion": {"tn_steady": tn, "fp": fp, "fn": fn, "tp_phasic": tp}}


def mcnemar_exact(correct_a, correct_b):
    """Two-sided exact McNemar (binomial, p0=0.5) on discordant pairs.
    b = a-correct & b-wrong ; c = a-wrong & b-correct."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b, c, round(min(1.0, 2.0 * tail), 4)


def cluster_bootstrap_ci(correct, clusters, n_boot=20000, seed=0):
    """95% CI for mean(correct) resampling whole clusters with replacement."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(clusters))
    idx = {u: np.where(clusters == u)[0] for u in uniq}
    accs = np.empty(n_boot)
    for bi in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([idx[uniq[j]] for j in pick])
        accs[bi] = correct[sel].mean()
    return {"mean": round(float(accs.mean()), 4),
            "ci95_lower": round(float(np.percentile(accs, 2.5)), 4),
            "ci95_upper": round(float(np.percentile(accs, 97.5)), 4),
            "n_clusters": len(uniq), "n_boot": n_boot}


def g3_recompute(rows, y, workload):
    """Phasic G3 cepstral-SNR, full vs. n_windows>=8 subset, with per-workload
    medians (to reconcile the per-cell distribution against the per-workload
    '4.78 dB floor' quoted in Paper B)."""
    snr_all = np.array([_f(r, "ceps_peak_snr_db") for r in rows])
    nwin = np.array([_f(r, "n_windows") for r in rows])

    def summary(mask):
        snr = snr_all[mask]
        nw = nwin[mask]
        snr = snr[~np.isnan(snr)]
        if snr.size == 0:
            return {"n_cells": 0}
        return {"n_cells": int(snr.size),
                "snr_min_floor": round(float(snr.min()), 4),
                "snr_median": round(float(np.median(snr)), 4),
                "snr_max": round(float(snr.max()), 4),
                "median_gt_4p5": bool(np.median(snr) > SNR_THRESHOLD),
                "min_gt_4p5": bool(snr.min() > SNR_THRESHOLD),
                "n_windows_min": int(np.nanmin(nw)) if nw.size else None,
                "n_windows_median": float(np.nanmedian(nw)) if nw.size else None}

    def per_workload_medians(mask):
        out = {}
        for w in sorted(set(workload[mask])):
            wm = mask & (workload == w)
            s = snr_all[wm]
            s = s[~np.isnan(s)]
            if s.size:
                out[w] = {"n": int(s.size), "snr_median": round(float(np.median(s)), 4)}
        meds = [v["snr_median"] for v in out.values()]
        return {"by_workload": out,
                "min_of_workload_medians": round(min(meds), 4) if meds else None,
                "max_of_workload_medians": round(max(meds), 4) if meds else None}

    phasic = (y == 1)
    enough = phasic & (nwin >= MIN_WINDOWS_G3)
    n_phasic = int(phasic.sum())
    n_enough = int(enough.sum())
    return {"min_windows_required": MIN_WINDOWS_G3,
            "snr_threshold_db": SNR_THRESHOLD,
            "n_phasic_total": n_phasic,
            "n_phasic_with_ge8_windows": n_enough,
            "frac_phasic_dropped": round(1 - n_enough / n_phasic, 4) if n_phasic else None,
            "all_phasic": summary(phasic),
            "phasic_ge8_windows": summary(enough),
            "per_workload_medians_all_phasic": per_workload_medians(phasic),
            "per_workload_medians_ge8_windows": per_workload_medians(enough)}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SWEEP
    rows = [r for r in csv.DictReader(open(path))
            if r["window"] == "8" and r["hop"] == "4"
            and r["status"].startswith("ok")]
    y = np.array([1 if r["workload"] in PHASIC else 0 for r in rows])
    rep = np.array([int(r["replicate"]) for r in rows])
    workload = np.array([r["workload"] for r in rows])
    wl_dur = np.array([f'{r["workload"]}|{r["duration_s"]}' for r in rows])
    majority = 0 if (y == 0).sum() >= (y == 1).sum() else 1
    base_pred = np.full_like(y, majority)

    # Headline protocol: leave-one-replicate-out (reproduces leakage_ablation).
    gkf = GroupKFold(n_splits=len(set(rep)))
    pred_ag = _predict(rows, AGNOSTIC, y, rep, gkf)
    pred_full = _predict(rows, FULL, y, rep, gkf)

    correct_ag = (pred_ag == y)
    correct_base = (base_pred == y)
    b, c, mcp = mcnemar_exact(correct_ag, correct_base)

    # Leave-one-workload-out.
    logo = LeaveOneGroupOut()
    pred_ag_lowo = _predict(rows, AGNOSTIC, y, workload, logo)
    pred_full_lowo = _predict(rows, FULL, y, workload, logo)
    per_wl = {}
    for w in sorted(set(workload)):
        m = workload == w
        per_wl[w] = {"family": "phasic" if y[m][0] == 1 else "steady",
                     "n": int(m.sum()),
                     "agnostic_acc": round(float((pred_ag_lowo[m] == y[m]).mean()), 4),
                     "full_acc": round(float((pred_full_lowo[m] == y[m]).mean()), 4)}

    out = {
        "schema": "paperC.robustness_reanalysis.v1",
        "sklearn_version": sklearn.__version__,
        "sweep_path": path,
        "n_cells": len(rows), "n_phasic": int(y.sum()), "n_steady": int((y == 0).sum()),
        "majority_baseline": round(float(max((y == 0).mean(), (y == 1).mean())), 4),
        "n_replicate_groups": len(set(rep)),
        "n_workload_duration_clusters": len(set(wl_dur)),
        "R5_significance": {
            "leave_one_replicate_out": {
                "agnostic": _acc_conf(pred_ag, y),
                "full_leaky": _acc_conf(pred_full, y),
            },
            "mcnemar_agnostic_vs_majority": {
                "b_phasic_caught_by_agnostic": b,
                "c_steady_lost_by_agnostic": c,
                "exact_two_sided_p": mcp,
                "note": ("b = cells AGNOSTIC gets right that the majority rule "
                         "gets wrong (phasic); c = cells AGNOSTIC gets wrong "
                         "that the majority rule gets right (steady). p>0.05 "
                         "means the net gain over always-predict-steady is not "
                         "significant.")},
            "cluster_bootstrap_accuracy_ci": cluster_bootstrap_ci(correct_ag, wl_dur),
            "interpretation": (
                "If the cluster-bootstrap CI lower bound sits at or below the "
                "majority baseline, the leakage-free family signal is not "
                "significant once replicate non-independence is respected."),
        },
        "R7_g3_cepstral_snr_recompute": g3_recompute(rows, y, workload),
        "LOWO_leave_one_workload_out": {
            "protocol": "LeaveOneGroupOut by workload (11 folds)",
            "agnostic_overall_acc": round(float((pred_ag_lowo == y).mean()), 4),
            "full_overall_acc": round(float((pred_full_lowo == y).mean()), 4),
            "per_workload": per_wl,
            "interpretation": (
                "AGNOSTIC accuracy under leave-one-workload-out tests "
                "generalization to an UNSEEN workload. A large drop vs. the "
                "leave-one-replicate-out number means the modest family signal "
                "is partly workload memorization. FULL staying ~1.0 here would "
                "confirm the leakage feature (coverage_ratio) is a per-family "
                "constant invariant to which workload is held out."),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
