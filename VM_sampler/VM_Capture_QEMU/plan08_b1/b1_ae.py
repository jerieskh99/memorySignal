#!/usr/bin/env python3
"""b1_ae.py -- per-family autoencoder bank over the B1 windows (Phase 1).

One autoencoder per family (sklearn MLPRegressor, 8 -> 3 -> 8, trained to
reconstruct its own family's windows). A test window is scored by every family
AE's reconstruction MSE; predicted family = argmin. Run per arm (APF, wAPF) over
the three splits (within-trace ceiling / LORO / LOWO). The number that matters
is the difference between the arms, same folds both times.

Discipline baked in:
  * per-fold StandardScaler fit on TRAIN windows only (never test) -- the reason
    b1_windows emits raw values.
  * grouping via b1_splits (no cell/workload straddles a split).
  * a family with too few train windows on a fold (e.g. sandbox held out under
    LOWO -- its only workload gone) is NOT modeled; its test windows fall to the
    other families -> recall ~0, by construction, matching the documented novelty.
  * fixed hyperparameters, one recorded seed. Bottleneck 3 is the one knob and is
    not swept (effective n = workloads).

Output: b1_ae_results.json + a printed arm x split accuracy table and per-family
recall for the LOWO headline.

Usage: b1_ae.py <b1_windows.npz> [--seed 0] [--bottleneck 3] [--min-train 8]
                [--max-iter 300] [--out b1_ae_results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b1_splits as S   # noqa: E402


def build_ae(bottleneck: int, seed: int, max_iter: int) -> MLPRegressor:
    return MLPRegressor(hidden_layer_sizes=(bottleneck,), activation="tanh",
                        solver="adam", max_iter=max_iter, random_state=seed,
                        tol=1e-5)


def recon_mse(ae: MLPRegressor, X: np.ndarray) -> np.ndarray:
    """Per-row reconstruction MSE."""
    Xhat = ae.predict(X)
    if Xhat.ndim == 1:                       # MLPRegressor returns 1D for 1 output
        Xhat = Xhat.reshape(X.shape)
    return np.mean((X - Xhat) ** 2, axis=1)


def classify_fold(Xtr, ytr, Xte, bottleneck, seed, min_train, max_iter):
    """Train a per-family AE bank on (Xtr, ytr), return argmin-error family per
    test row. Families with < min_train windows are skipped (not modeled)."""
    scaler = StandardScaler().fit(Xtr)       # fit on TRAIN only
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    errs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for fam in sorted(set(ytr)):
            Zf = Ztr[ytr == fam]
            if len(Zf) < min_train:
                continue                     # too thin to model -> not a candidate
            ae = build_ae(bottleneck, seed, max_iter).fit(Zf, Zf)
            errs[fam] = recon_mse(ae, Zte)
        if not errs:
            return np.array(["<none>"] * len(Zte))
        fams = list(errs)
        M = np.vstack([errs[f] for f in fams])          # [n_families x n_test]
        return np.array([fams[i] for i in np.argmin(M, axis=0)])


def _score(y_true, y_pred):
    n = len(y_true)
    acc = float(np.mean(y_true == y_pred)) if n else 0.0
    maj = max(Counter(y_true).values()) / n if n else 0.0
    recall = {}
    conf = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        conf[str(t)][str(p)] += 1
    for fam in sorted(set(y_true)):
        m = y_true == fam
        recall[str(fam)] = float(np.mean(y_pred[m] == fam))
    return {"accuracy": acc, "majority": float(maj), "n_test": n,
            "per_family_recall": recall,
            "confusion": {k: dict(v) for k, v in conf.items()}}


def run_split(X, y_fam, y_cell, folds, bottleneck, seed, min_train, max_iter):
    """Aggregate predictions across a split's folds -> one score over all test rows."""
    yt, yp = [], []
    for f in folds:
        tr, te = f["train"], f["test"]
        pred = classify_fold(X[tr], y_fam[tr], X[te], bottleneck, seed, min_train, max_iter)
        yt.append(y_fam[te]); yp.append(pred)
    return _score(np.concatenate(yt), np.concatenate(yp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bottleneck", type=int, default=3)
    ap.add_argument("--min-train", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.npz.with_name("b1_ae_results.json")

    lab = S.load_labels(a.npz)
    d = np.load(a.npz, allow_pickle=False)
    arms = {"apf": d["X_apf"], "wapf": d["X_wapf"]}
    y_fam = lab["family"]; y_cell = lab["cell_id"]

    splits = {
        "within_trace": S.fold_within_trace(lab, a.test_frac),
        "loro": S.fold_loro(lab),
        "lowo": S.fold_lowo(lab),
    }

    results = {"seed": a.seed, "bottleneck": a.bottleneck, "min_train": a.min_train,
               "max_iter": a.max_iter, "arms": {}}
    for arm, X in arms.items():
        results["arms"][arm] = {}
        for sname, folds in splits.items():
            results["arms"][arm][sname] = run_split(
                X, y_fam, y_cell, folds, a.bottleneck, a.seed, a.min_train, a.max_iter)

    out.write_text(json.dumps(results, indent=2) + "\n")

    # headline table
    print(f"\n{'':14} {'APF acc':>9} {'wAPF acc':>9} {'majority':>9}")
    for sname in ("within_trace", "loro", "lowo"):
        ra = results["arms"]["apf"][sname]; rw = results["arms"]["wapf"][sname]
        print(f"{sname:14} {ra['accuracy']:>9.3f} {rw['accuracy']:>9.3f} {ra['majority']:>9.3f}")
    print("\nLOWO per-family recall (the honest headline):")
    print(f"  {'family':10} {'APF':>7} {'wAPF':>7}")
    ra = results["arms"]["apf"]["lowo"]["per_family_recall"]
    rw = results["arms"]["wapf"]["lowo"]["per_family_recall"]
    for fam in sorted(set(ra) | set(rw)):
        print(f"  {fam:10} {ra.get(fam, 0):>7.3f} {rw.get(fam, 0):>7.3f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
