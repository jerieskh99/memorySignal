#!/usr/bin/env python3
"""b1_splits.py -- grouped train/test folds over the B1 window labels.

Three split types, all grouping-safe. The folds are index arrays over the rows
of b1_windows.npz, so the SAME fold applies to X_apf and X_wapf (row-aligned) --
that is what keeps the APF-vs-wAPF comparison fair. Model-agnostic: this returns
indices; the model decides how to use the train set (e.g. per-family AE bank).

  within_trace : per cell, the last `test_frac` of its windows -> test, the rest
                 -> train. ONE fold. Deliberate memorization CEILING (the same
                 trace is on both sides), never the headline.
  loro         : leave-one-cell-out (a cell == one rep). Each fold holds out one
                 cell; train = every other cell. "Unseen run." No cell spans
                 train and test.
  lowo         : leave-one-workload-out. Each fold holds out all cells of one
                 workload; train = every other workload. "Unseen workload." A
                 family with a single workload (sandbox) gets no same-family
                 training on its own fold -> structurally a novelty case; flagged
                 in `novelty`, not hidden.

CLI prints a participation summary. Import the fold_* functions from the harness.

Usage: b1_splits.py <b1_windows.npz> [--test-frac 0.2]
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np


def load_labels(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    n = d["X_apf"].shape[0]
    return {
        "n": n,
        "cell_id": d["cell_id"].astype(str),
        "workload": d["workload"].astype(str),
        "family": d["family"].astype(str),
        "rep": d["rep"].astype(int),
        "win_start": d["win_start"].astype(int),
    }


def _groups(keys):
    """Ordered {key: sorted index array} preserving first-seen order."""
    out = OrderedDict()
    for i, k in enumerate(keys):
        out.setdefault(k, []).append(i)
    return OrderedDict((k, np.asarray(v, dtype=np.int64)) for k, v in out.items())


def fold_within_trace(lab: dict, test_frac: float = 0.2) -> list[dict]:
    """One fold: each cell's tail (last test_frac windows) is test, head is train."""
    tr, te = [], []
    for cell, idx in _groups(lab["cell_id"]).items():
        order = idx[np.argsort(lab["win_start"][idx])]   # chronological within cell
        n = len(order)
        n_test = max(1, int(round(test_frac * n)))
        n_test = min(n_test, n - 1) if n > 1 else 0      # keep >=1 train window
        te.extend(order[n - n_test:].tolist())
        tr.extend(order[:n - n_test].tolist())
    return [{"name": "within_trace", "train": np.asarray(sorted(tr), dtype=np.int64),
             "test": np.asarray(sorted(te), dtype=np.int64)}]


def fold_loro(lab: dict) -> list[dict]:
    """Leave-one-cell-out (cell == rep). One fold per cell."""
    folds = []
    for cell, idx in _groups(lab["cell_id"]).items():
        test = idx
        train = np.setdiff1d(np.arange(lab["n"]), test, assume_unique=False)
        wl = lab["workload"][idx[0]]
        folds.append({"name": f"loro/{cell}", "held_out": cell, "workload": wl,
                      "family": lab["family"][idx[0]],
                      "train": train, "test": test})
    return folds


def fold_lowo(lab: dict) -> list[dict]:
    """Leave-one-workload-out. One fold per workload; flags novelty (the held-out
    workload's family has no other workload to train on)."""
    fam_workloads = {}
    for wl, idx in _groups(lab["workload"]).items():
        fam_workloads.setdefault(lab["family"][idx[0]], set()).add(wl)
    folds = []
    for wl, idx in _groups(lab["workload"]).items():
        fam = lab["family"][idx[0]]
        test = idx
        train = np.setdiff1d(np.arange(lab["n"]), test, assume_unique=False)
        novelty = len(fam_workloads[fam]) < 2      # nothing same-family left to train
        folds.append({"name": f"lowo/{wl}", "held_out": wl, "family": fam,
                      "novelty": novelty, "train": train, "test": test})
    return folds


def _assert_grouped(folds: list[dict], lab: dict, group_key: str):
    """No group straddles train/test."""
    for f in folds:
        gtest = set(lab[group_key][f["test"]].tolist())
        gtrain = set(lab[group_key][f["train"]].tolist())
        overlap = gtest & gtrain
        assert not overlap, f"{f['name']}: {group_key} leaks across split: {overlap}"
        assert len(set(f["train"].tolist()) & set(f["test"].tolist())) == 0, \
            f"{f['name']}: train/test row overlap"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--test-frac", type=float, default=0.2)
    a = ap.parse_args()
    lab = load_labels(a.npz)

    wt = fold_within_trace(lab, a.test_frac)
    lo = fold_loro(lab)
    lw = fold_lowo(lab)
    _assert_grouped(lo, lab, "cell_id")
    _assert_grouped(lw, lab, "workload")

    print(f"[b1_splits] {lab['n']} windows, "
          f"{len(set(lab['cell_id']))} cells, {len(set(lab['workload']))} workloads, "
          f"{len(set(lab['family']))} families")
    print(f"  within_trace: 1 fold  (train {len(wt[0]['train'])}, test {len(wt[0]['test'])}) -- CEILING")
    print(f"  loro (leave-one-cell-out):     {len(lo)} folds")
    print(f"  lowo (leave-one-workload-out): {len(lw)} folds")
    nov = [str(f["held_out"]) for f in lw if f["novelty"]]
    if nov:
        print(f"    novelty (no same-family train, expect recall ~0): {nov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
