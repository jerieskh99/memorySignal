#!/usr/bin/env python3
"""Smoke + correctness test for plan08_b1/b1_ae.py.

Builds two clearly-separable families (A ~ high plateau, B ~ low plateau), so a
correct AE bank must classify them near-perfectly. Checks: classify_fold picks
the right family on held-out separable data; the full run produces the arm x
split structure; within-trace (ceiling) accuracy is high; and a single-workload
"novelty" family (held out under LOWO with nothing to train) is NOT predicted as
itself (recall ~0), matching the documented behavior.

Run:  python3 tests/test_plan08_b1_ae.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR / "plan08_b1"))
import b1_ae as AE   # noqa: E402


def test_classify_fold_separable():
    rng = np.random.default_rng(0)
    # family A ~ 0.8, family B ~ 0.1, 8-dim windows
    A = rng.normal(0.8, 0.02, size=(80, 8))
    B = rng.normal(0.1, 0.02, size=(80, 8))
    Xtr = np.vstack([A[:60], B[:60]])
    ytr = np.array(["A"] * 60 + ["B"] * 60)
    Xte = np.vstack([A[60:], B[60:]])
    yte = np.array(["A"] * 20 + ["B"] * 20)
    pred = AE.classify_fold(Xtr, ytr, Xte, bottleneck=3, seed=0, min_train=8, max_iter=400)
    acc = float(np.mean(pred == yte))
    assert acc >= 0.9, f"separable families should classify easily, got {acc}"
    print(f"  classify_fold separable acc = {acc:.3f}")


def _make_npz(path: Path):
    """3 families: A (2 workloads x 2 cells), B (2 workloads x 2 cells),
    NOV (1 workload x 1 cell) -> NOV is the LOWO novelty case."""
    rng = np.random.default_rng(1)
    rows = []  # (Xrow, family, workload, cell, win_start)
    def add(fam, wl, cell, center, nwin):
        for j in range(nwin):
            x = rng.normal(center, 0.02, size=8)
            rows.append((x, fam, wl, cell, j * 4))
    add("A", "a1", "cA1", 0.80, 40); add("A", "a1", "cA2", 0.80, 40)
    add("A", "a2", "cA3", 0.78, 40); add("A", "a2", "cA4", 0.78, 40)
    add("B", "b1", "cB1", 0.20, 40); add("B", "b1", "cB2", 0.20, 40)
    add("B", "b2", "cB3", 0.22, 40); add("B", "b2", "cB4", 0.22, 40)
    add("NOV", "n1", "cN1", 0.50, 40)                     # sole workload of its family
    X = np.array([r[0] for r in rows])
    np.savez(path, X_apf=X, X_wapf=X.copy(),
             cell_id=np.array([r[3] for r in rows]),
             workload=np.array([r[2] for r in rows]),
             family=np.array([r[1] for r in rows]),
             rep=np.ones(len(rows), dtype=np.int32),
             win_start=np.array([r[4] for r in rows], dtype=np.int32),
             window=8, hop=4)


def test_run_structure_and_novelty(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    npz = tmp / "b1_windows.npz"
    _make_npz(npz)

    import b1_splits as S
    lab = S.load_labels(npz)
    d = np.load(npz, allow_pickle=False)
    y_fam = lab["family"]

    # within-trace ceiling should classify A vs B vs NOV near-perfectly (all seen)
    wt = S.fold_within_trace(lab, 0.2)
    r = AE.run_split(d["X_apf"], y_fam, lab["cell_id"], wt, 3, 0, 8, 400)
    assert r["accuracy"] >= 0.9, f"within-trace ceiling too low: {r['accuracy']}"

    # LOWO: hold out NOV's only workload -> no NOV AE -> NOV recall ~0 (novelty)
    lw = S.fold_lowo(lab)
    r = AE.run_split(d["X_apf"], y_fam, lab["cell_id"], lw, 3, 0, 8, 400)
    nov_recall = r["per_family_recall"].get("NOV", 0.0)
    assert nov_recall == 0.0, f"single-workload family must be unrecoverable under LOWO, got {nov_recall}"
    # but the real families (2 workloads each) should still be recognized under LOWO
    assert r["per_family_recall"].get("A", 0) >= 0.8, r["per_family_recall"]
    assert r["per_family_recall"].get("B", 0) >= 0.8, r["per_family_recall"]

    print(f"  within-trace acc = {AE.run_split(d['X_apf'], y_fam, lab['cell_id'], wt, 3,0,8,400)['accuracy']:.3f}")
    print(f"  LOWO recall: A={r['per_family_recall'].get('A'):.2f} "
          f"B={r['per_family_recall'].get('B'):.2f} NOV={nov_recall:.2f} (novelty)")


if __name__ == "__main__":
    test_classify_fold_separable()
    test_run_structure_and_novelty()
    print("PASS: AE bank classifies separable families, ceiling high, LOWO novelty handled")
