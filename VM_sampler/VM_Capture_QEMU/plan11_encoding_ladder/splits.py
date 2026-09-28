#!/usr/bin/env python3
"""splits.py -- within-trace, LORO and LOKO folds at the unit of the cell (SPEC section 4.1).

``fold_within_trace``, ``fold_loro``, ``fold_loko`` are ``plan08_b1/b1_splits.py``'s
``fold_within_trace``, ``fold_loro``, ``fold_lowo`` copied (2026-09-16) with ``workload -> kernel``
and ``family -> archetype`` renamed, plus ``_assert_grouped`` run on every fold list before use.
Whole cells are held out; no cell's windows straddle train and test.

Citation: P2_STRUCTURE.md section V 'The splits' (unit = cell; within-trace a ceiling only; LORO an
instrument's reading; LOKO the headline); SPEC 4.1 table; section 8 item 21.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLITS = ("within_trace", "loro", "loko")
TEST_FRAC = 0.2
LORO_MODE = "cell"           # section 8 item 21; alternative "rep_index"


def make_labels(feat: dict, mask: np.ndarray | None = None) -> dict:
    """The label dict the fold functions read, from a loaded feature file (series.load_features),
    restricted to ``mask`` rows. Keys: n, cell_id, kernel, archetype, rep, win_start, campaign."""
    idx = np.arange(len(feat["cell_id"])) if mask is None else np.flatnonzero(mask)
    return {
        "n": len(idx), "_rows": idx,
        "cell_id": feat["cell_id"][idx], "kernel": feat["kernel"][idx],
        "archetype": feat["archetype"][idx], "rep": feat["rep"][idx].astype(int),
        "win_start": feat["win_start"][idx].astype(int), "campaign": feat["campaign"][idx],
    }


def _groups(keys):
    # copied from plan08_b1/b1_splits.py:_groups, 2026-09-16
    """Ordered {key: sorted index array} preserving first-seen order."""
    out = OrderedDict()
    for i, k in enumerate(keys):
        out.setdefault(k, []).append(i)
    return OrderedDict((k, np.asarray(v, dtype=np.int64)) for k, v in out.items())


def fold_within_trace(lab: dict, test_frac: float = TEST_FRAC) -> list[dict]:
    # copied from plan08_b1/b1_splits.py:fold_within_trace, 2026-09-16
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
    # copied from plan08_b1/b1_splits.py:fold_loro, 2026-09-16 (workload -> kernel, family -> archetype)
    """Leave-one-cell-out (cell == rep). One fold per cell."""
    folds = []
    for cell, idx in _groups(lab["cell_id"]).items():
        test = idx
        train = np.setdiff1d(np.arange(lab["n"]), test, assume_unique=False)
        wl = lab["kernel"][idx[0]]
        folds.append({"name": f"loro/{cell}", "held_out": cell, "kernel": wl,
                      "archetype": lab["archetype"][idx[0]],
                      "held_out_campaign": lab["campaign"][idx[0]],
                      "train": train, "test": test})
    return folds


def fold_loro_rep_index(lab: dict) -> list[dict]:
    """``loro_mode = "rep_index"``: hold out every cell of one rep index (8 folds); section 8 item 21."""
    folds = []
    for rep, idx in _groups(lab["rep"]).items():
        test = idx
        train = np.setdiff1d(np.arange(lab["n"]), test, assume_unique=False)
        folds.append({"name": f"loro/rep{int(rep):02d}", "held_out": f"rep{int(rep):02d}",
                      "train": train, "test": test})
    return folds


def fold_loko(lab: dict) -> list[dict]:
    # copied from plan08_b1/b1_splits.py:fold_lowo, 2026-09-16 (workload -> kernel, family -> archetype)
    """Leave-one-kernel-out. One fold per kernel; flags novelty (the held-out kernel's archetype has
    no other kernel to train on)."""
    fam_workloads = {}
    for wl, idx in _groups(lab["kernel"]).items():
        fam_workloads.setdefault(lab["archetype"][idx[0]], set()).add(wl)
    folds = []
    for wl, idx in _groups(lab["kernel"]).items():
        fam = lab["archetype"][idx[0]]
        test = idx
        train = np.setdiff1d(np.arange(lab["n"]), test, assume_unique=False)
        novelty = len(fam_workloads[fam]) < 2      # nothing same-archetype left to train
        folds.append({"name": f"loko/{wl}", "held_out": wl, "archetype": fam,
                      "novelty": novelty, "train": train, "test": test})
    return folds


def _assert_grouped(folds: list[dict], lab: dict, group_key: str):
    # copied from plan08_b1/b1_splits.py:_assert_grouped, 2026-09-16
    """No group straddles train/test."""
    for f in folds:
        gtest = set(lab[group_key][f["test"]].tolist())
        gtrain = set(lab[group_key][f["train"]].tolist())
        overlap = gtest & gtrain
        assert not overlap, f"{f['name']}: {group_key} leaks across split: {overlap}"
        assert len(set(f["train"].tolist()) & set(f["test"].tolist())) == 0, \
            f"{f['name']}: train/test row overlap"


def folds_for(split: str, lab: dict, *, test_frac: float = TEST_FRAC, loro_mode: str = LORO_MODE) -> list[dict]:
    """The fold list of one split with ``_assert_grouped`` applied (SPEC 4.1)."""
    if split == "within_trace":
        folds = fold_within_trace(lab, test_frac)
        for f in folds:                     # windows of one cell on both sides by design; rows disjoint
            assert len(set(f["train"].tolist()) & set(f["test"].tolist())) == 0
        return folds
    if split == "loro":
        folds = fold_loro(lab) if loro_mode == "cell" else fold_loro_rep_index(lab)
        _assert_grouped(folds, lab, "cell_id")
        return folds
    if split == "loko":
        folds = fold_loko(lab)
        _assert_grouped(folds, lab, "kernel")
        _assert_grouped(folds, lab, "cell_id")
        return folds
    raise ValueError(split)
