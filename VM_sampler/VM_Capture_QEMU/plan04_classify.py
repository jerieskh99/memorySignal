"""classify_workloads.py -- Classification phase on the v3 132-cell
dataset.

NOTE: this is the classification phase (the thesis taxonomy claim),
not Plan 04 (which is the segmenter, owned by plan04_segmenter_*).
The script is named plan04_classify.py for historical reasons; the
canonical name in the runbook is classify_workloads.

Reads the Plan 03 sweep CSV, filters to the recommended (W=8, H=4)
combo, builds a per-cell feature vector, then runs three classifiers
under leave-one-replicate-out cross-validation. Reports both the
11-way workload identification task AND the binary phasic-vs-steady
task -- the latter is the central thesis taxonomy claim.

Input:
  --sweep-csv  path to plan03_sweep.csv (combined v3_full + v3_ext)
  --output     output JSON path (default plan04_classification.json)

Output:
  JSON with per-model metrics + confusion matrices, plus an ASCII
  summary printed to stdout.

Usage:
  python3 plan04_classify.py \\
      --sweep-csv _v3_combined/plan03/sweep.csv \\
      --window 8 --hop 4 \\
      --output _v3_combined/plan04_classification.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  f1_score)
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    print(f"ERROR: scikit-learn required: {exc}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "apf_mean",
    "apf_std",
    "cv_workingset",
    "f1_phase",
    "cepstral_peak_idx",
    "ceps_peak_snr_db",
    "stat_pass_frac",
    "n_pairs",
    "n_windows",
    "coverage_ratio",
]


def _to_float(s: Any) -> float | None:
    if s is None or s == "" or s == "None":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_per_cell_features(sweep_csv: Path, window: int, hop: int
                            ) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray, list[str], list[str]]:
    """Load + filter sweep rows; build X, y, groups, cell_ids, classes."""
    rows = list(csv.DictReader(sweep_csv.open()))
    keep = [r for r in rows
            if int(r["window"]) == window
            and int(r["hop"]) == hop
            and r["status"] == "ok"]
    if not keep:
        raise SystemExit(
            f"no W={window} H={hop} status=ok rows in {sweep_csv}")
    X_list: list[list[float | None]] = []
    y_list: list[str] = []
    g_list: list[int] = []
    cid_list: list[str] = []
    for r in keep:
        feats = [_to_float(r[c]) for c in FEATURE_COLS]
        X_list.append(feats)
        y_list.append(r["workload"])
        g_list.append(int(r["replicate"]))
        cid_list.append(r["cell_id"])
    X = np.array(X_list, dtype=float)  # NaN where None
    y = np.array(y_list)
    g = np.array(g_list)
    classes = sorted(set(y_list))
    return X, y, g, cid_list, classes


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_models(random_state: int = 0) -> dict[str, Pipeline]:
    """Three classifiers wrapped in a NaN-aware Pipeline (impute + scale + fit).
    Median imputer fills NaN per feature; useful because cv_workingset is
    only populated for steady cells and f1_phase only for phasic cells.
    """
    common = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    return {
        "kNN(k=5)": Pipeline(common + [
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]),
        "RandomForest(n=300)": Pipeline(common + [
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None,
                random_state=random_state, n_jobs=-1)),
        ]),
        "MLP(64,32)": Pipeline(common + [
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=2000,
                random_state=random_state, early_stopping=False)),
        ]),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

PHASIC_KEYS = ("ransom", "scanner_metadata", "phase_boundary", "phasic")
STEADY_KEYS = ("workingset", "mmap_traversal", "pagefault_density",
               "rmw_intensity", "writemag_sweep", "hashtable_intensive",
               "compress_streaming", "compress_gzip", "decompress_gzip",
               "json_parse", "sqlite_oltp", "sqlite_analytical", "steady")


def _family_of(workload: str) -> str:
    """Mirror plan02_run._classify_workload."""
    w = workload.lower()
    if any(k in w for k in PHASIC_KEYS):
        return "phasic"
    if any(k in w for k in STEADY_KEYS):
        return "steady"
    return "unknown"


def _binary_metrics(y_true_workload: list[str],
                    y_pred_workload: list[str]) -> dict[str, Any]:
    """Compute phasic-vs-steady binary metrics from workload-level
    true + predicted labels."""
    y_true_b = [_family_of(w) for w in y_true_workload]
    y_pred_b = [_family_of(w) for w in y_pred_workload]
    families = ["phasic", "steady"]
    cm_b = confusion_matrix(y_true_b, y_pred_b, labels=families)
    report = classification_report(
        y_true_b, y_pred_b, labels=families,
        output_dict=True, zero_division=0)
    macro_f1 = float(f1_score(y_true_b, y_pred_b,
                               labels=families,
                               average="macro", zero_division=0))
    accuracy = float(np.trace(cm_b) / cm_b.sum())
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm_b.tolist(),
        "classes": families,
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in families
        },
    }


def run_cv(X: np.ndarray, y: np.ndarray, g: np.ndarray,
           classes: list[str], random_state: int = 0
           ) -> dict[str, dict[str, Any]]:
    """Leave-one-replicate-out CV (GroupKFold). Aggregate confusion across
    folds. Returns per-model metrics dict."""
    n_splits = len(set(g))
    gkf = GroupKFold(n_splits=n_splits)
    models = make_models(random_state=random_state)
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        cm_agg = np.zeros((len(classes), len(classes)), dtype=int)
        y_true_all: list[str] = []
        y_pred_all: list[str] = []
        for tr_idx, te_idx in gkf.split(X, y, g):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            cm = confusion_matrix(y_te, y_pred, labels=classes)
            cm_agg += cm
            y_true_all.extend(y_te.tolist())
            y_pred_all.extend(y_pred.tolist())
        # per-class precision/recall/F1
        report = classification_report(
            y_true_all, y_pred_all, labels=classes,
            output_dict=True, zero_division=0)
        macro_f1 = float(f1_score(y_true_all, y_pred_all,
                                   labels=classes,
                                   average="macro", zero_division=0))
        # accuracy as overall correct/total
        accuracy = float(np.trace(cm_agg) / cm_agg.sum())
        results[name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "confusion_matrix": cm_agg.tolist(),
            "classes": classes,
            "per_class": {
                cls: {
                    "precision": float(report[cls]["precision"]),
                    "recall": float(report[cls]["recall"]),
                    "f1": float(report[cls]["f1-score"]),
                    "support": int(report[cls]["support"]),
                }
                for cls in classes
            },
            "binary_phasic_vs_steady": _binary_metrics(y_true_all, y_pred_all),
        }
    return results


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _short(name: str) -> str:
    """Abbreviated workload name for the confusion matrix display."""
    if "ransom_batched" in name: return "ran_bat"
    if "ransom_seq" in name: return "ran_seq"
    if "ransom_slowburn" in name: return "ran_slw"
    if "ransom_selective" in name: return "ran_sel"
    if "scanner_metadata" in name: return "scanner"
    if "workingset_sweep" in name: return "wkset"
    if "mmap_traversal" in name: return "mmap"
    if "pagefault_density" in name: return "pgflt"
    if "rmw_intensity" in name: return "rmw"
    if "writemag_sweep" in name: return "wrmag"
    if "hashtable_intensive" in name: return "hash"
    return name[:7]


def print_summary(results: dict[str, dict[str, Any]],
                  classes: list[str], n_cells: int) -> None:
    print(f"\n{'='*72}")
    print(f"Plan 04 classification · 132-cell v3 dataset · {n_cells} usable")
    print(f"Classes: {len(classes)} workloads · CV: leave-one-replicate-out")
    print(f"{'='*72}")

    # headline ranked
    ranked = sorted(results.items(),
                    key=lambda kv: kv[1]["macro_f1"], reverse=True)
    print("\nModel ranking (by macro F1):")
    for name, r in ranked:
        print(f"  {name:<22} acc={r['accuracy']:.3f}  "
              f"macro_F1={r['macro_f1']:.3f}")

    # confusion matrix for the best model
    best_name, best = ranked[0]
    print(f"\nConfusion matrix · {best_name} (rows=true, cols=predicted):")
    short = [_short(c) for c in classes]
    header = "        " + "  ".join(f"{s:>7}" for s in short)
    print(header)
    cm = np.array(best["confusion_matrix"])
    for i, cls in enumerate(short):
        row = "  ".join(f"{cm[i, j]:>7d}" for j in range(len(short)))
        print(f"  {cls:<5} {row}")

    # Binary phasic vs steady -- the headline thesis claim
    print(f"\n{'='*72}")
    print(f"Phasic vs Steady (binary) -- the thesis taxonomy claim")
    print(f"{'='*72}")
    for name, r in ranked:
        b = r["binary_phasic_vs_steady"]
        cm_b = b["confusion_matrix"]
        print(f"\n  {name}: acc={b['accuracy']:.3f}  macro_F1={b['macro_f1']:.3f}")
        print(f"    confusion (rows=true phasic/steady, cols=predicted):")
        print(f"      phasic -> phasic={cm_b[0][0]:>3}  steady={cm_b[0][1]:>3}")
        print(f"      steady -> phasic={cm_b[1][0]:>3}  steady={cm_b[1][1]:>3}")
        for cls in ["phasic", "steady"]:
            m = b["per_class"][cls]
            print(f"    {cls:<8} precision={m['precision']:.3f} "
                  f"recall={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")

    print("\nPer-class metrics (best model, 11-way task):")
    print(f"  {'workload':<32} {'precision':>9} {'recall':>9} {'F1':>9} {'n':>5}")
    print("  " + "-" * 70)
    for cls in classes:
        m = best["per_class"][cls]
        print(f"  {cls:<32} {m['precision']:>9.3f} {m['recall']:>9.3f} "
              f"{m['f1']:>9.3f} {m['support']:>5d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sweep-csv", required=True, type=Path)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--hop", type=int, default=4)
    p.add_argument("--output", type=Path,
                    default=Path("plan04_classification.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    X, y, g, cell_ids, classes = load_per_cell_features(
        args.sweep_csv, args.window, args.hop)
    n_cells = len(cell_ids)

    print(f"Loaded {n_cells} cells across {len(classes)} workloads "
          f"at W={args.window} H={args.hop}", file=sys.stderr)
    print(f"Replicate groups: {sorted(set(g.tolist()))}", file=sys.stderr)
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}",
          file=sys.stderr)
    # NaN inventory per feature
    nan_pct = np.isnan(X).mean(axis=0)
    print("\nNaN rates per feature:", file=sys.stderr)
    for col, p_nan in zip(FEATURE_COLS, nan_pct):
        print(f"  {col:<22} {p_nan*100:>5.1f}%", file=sys.stderr)

    results = run_cv(X, y, g, classes, random_state=args.seed)

    out_payload = {
        "schema": "plan04.classification.v1",
        "input": {
            "sweep_csv": str(args.sweep_csv),
            "window": args.window,
            "hop": args.hop,
            "n_cells": n_cells,
            "n_classes": len(classes),
            "classes": classes,
            "replicate_groups": sorted(set(g.tolist())),
            "feature_cols": FEATURE_COLS,
        },
        "cv": {
            "scheme": "GroupKFold by replicate (leave-one-replicate-out)",
            "n_splits": len(set(g.tolist())),
        },
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote {args.output}", file=sys.stderr)

    print_summary(results, classes, n_cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
