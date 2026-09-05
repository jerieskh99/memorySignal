#!/usr/bin/env python3
"""Smoke + correctness test for plan08_b1/b1_splits.py.

Synthetic labels with a known layout: family A has 2 workloads (w1: 2 cells,
w2: 1 cell), family B has 1 workload (w3: 1 cell = the singleton/novelty case).
Checks that LORO leaves exactly one cell out, LOWO leaves one workload out,
within_trace keeps each cell on both sides, no group leaks, and the single-
workload family is flagged novelty.

Run:  python3 tests/test_plan08_b1_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR / "plan08_b1"))
import b1_splits as S   # noqa: E402


def make_labels():
    # cells: (cell_id, workload, family, n_windows)
    spec = [
        ("cA1", "w1", "A", 10),
        ("cA2", "w1", "A", 8),    # w1 has 2 reps
        ("cB1", "w2", "A", 6),    # w2 has 1 rep, but family A still has w1
        ("cC1", "w3", "B", 5),    # w3 sole workload of family B -> novelty under LOWO
    ]
    cid, wl, fam, ws, rep = [], [], [], [], []
    rep_of = {"cA1": 1, "cA2": 2, "cB1": 1, "cC1": 1}
    for c, w, f, n in spec:
        for j in range(n):
            cid.append(c); wl.append(w); fam.append(f); ws.append(j * 4); rep.append(rep_of[c])
    n = len(cid)
    return {"n": n, "cell_id": np.array(cid), "workload": np.array(wl),
            "family": np.array(fam), "rep": np.array(rep), "win_start": np.array(ws)}, spec


def test_splits():
    lab, spec = make_labels()
    total = lab["n"]

    # within_trace: one fold, every cell on both sides, disjoint, covers all
    wt = S.fold_within_trace(lab, test_frac=0.2)
    assert len(wt) == 1
    tr, te = set(wt[0]["train"].tolist()), set(wt[0]["test"].tolist())
    assert tr.isdisjoint(te)
    assert len(tr) + len(te) == total
    for c, w, f, n in spec:                       # each cell contributes to both
        rows = set(np.where(lab["cell_id"] == c)[0].tolist())
        assert rows & tr and rows & te, f"cell {c} not on both sides"

    # loro: one fold per cell, test = exactly that cell, train = the rest
    lo = S.fold_loro(lab)
    assert len(lo) == 4, len(lo)
    for f in lo:
        te = set(f["test"].tolist())
        assert set(lab["cell_id"][f["test"]].tolist()) == {f["held_out"]}
        assert set(f["train"].tolist()).isdisjoint(te)
        assert len(f["train"]) + len(te) == total
    S._assert_grouped(lo, lab, "cell_id")

    # lowo: one fold per workload; w1 (2 cells) held out together; w3 flagged novelty
    lw = S.fold_lowo(lab)
    assert len(lw) == 3, len(lw)                   # w1, w2, w3
    by_wl = {f["held_out"]: f for f in lw}
    assert set(lab["cell_id"][by_wl["w1"]["test"]].tolist()) == {"cA1", "cA2"}, "w1 reps not held out together"
    assert by_wl["w3"]["novelty"] is True, "single-workload family not flagged novelty"
    assert by_wl["w1"]["novelty"] is False and by_wl["w2"]["novelty"] is False
    S._assert_grouped(lw, lab, "workload")

    print("PASS: within_trace ceiling, loro=leave-one-cell, lowo=leave-one-workload, "
          "novelty flag, no group leakage")


if __name__ == "__main__":
    test_splits()
