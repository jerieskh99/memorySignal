#!/usr/bin/env python3
"""nulls.py -- phase-randomized surrogates, unit-level label shuffles, order shuffles and the null
summary (SPEC section 3.2).

Citation: CR 2.1 item 3 (G1's surrogate null), CR 2.1 item 9 (the unit-level label shuffle),
CR 2.2 item 27 (G-ORD's order shuffle); P2_STRUCTURE.md section V 5.1 (Plan 03 and Plan 08 as
amended).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED_SURROGATE = 20260916
SEED_LABEL_NULL = 20260917
SEED_ORDER = 20260918
SEED_FOREST = 20260919


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One phase-randomized surrogate: rfft of (x - mean), phases replaced by uniform [0, 2pi)
    with the DC and Nyquist bins kept real, irfft to len(x), mean added back. The amplitude
    spectrum and therefore the autocorrelation function (lag-1 included) are preserved exactly.
    This is the 'phase-randomized' alternative of CR 2.1 item 3 (G1); block bootstrap is not
    implemented (SPEC section 8 item 10)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3:
        return x.copy()
    m = x.mean()
    F = np.fft.rfft(x - m)
    amp = np.abs(F)
    ph = rng.uniform(0.0, 2.0 * np.pi, size=len(F))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    G = amp * np.exp(1j * ph)
    return np.fft.irfft(G, n=n) + m


def surrogates(x: np.ndarray, n: int = 200, seed: int = SEED_SURROGATE) -> np.ndarray:
    """``[n, len(x)]`` phase-randomized surrogates from one seeded generator (SPEC 3.2)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    return np.stack([phase_randomize(x, rng) for _ in range(n)], axis=0)


def shuffle_labels_units(cells: np.ndarray, kernels: np.ndarray, archetypes: np.ndarray,
                         split: str, labelspace: str, rng: np.random.Generator) -> np.ndarray:
    """Unit-level label shuffle (CR 2.1 item 9). LOKO/archetype: permute the archetype labels
    across the 12 kernels (one draw per kernel; every cell and window inherits). LORO or
    within-trace, kernel space: permute kernel labels across cells with the eight-per-kernel
    structure kept (a random permutation of the per-cell label vector). LORO or within-trace,
    archetype space: the kernel-level permutation, cells inherit. Campaign labels (G-X):
    permuted across cells. Inputs are per-cell vectors (one entry per cell); returns the
    permuted per-cell label vector. A window-level shuffle is never used."""
    cells = np.asarray(cells).astype(str)
    kernels = np.asarray(kernels).astype(str)
    labels = np.asarray(archetypes).astype(str)
    n = len(cells)
    if labelspace == "archetype":
        uk = list(dict.fromkeys(kernels.tolist()))            # first-seen order
        k_label = {k: labels[kernels == k][0] for k in uk}
        vals = [k_label[k] for k in uk]
        perm = rng.permutation(len(uk))
        newmap = {uk[i]: vals[perm[i]] for i in range(len(uk))}
        return np.array([newmap[k] for k in kernels])
    if labelspace == "kernel":
        return kernels[rng.permutation(n)]
    if labelspace == "campaign":
        return labels[rng.permutation(n)]
    raise ValueError(labelspace)


def order_shuffle(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the row order of one cell's series (G-ORD, CR 2.2 item 27)."""
    series = np.asarray(series)
    return series[rng.permutation(series.shape[0])]


def null_summary(observed: float, null: np.ndarray) -> dict:
    """{'observed', 'n', 'p95', 'p05', 'spread' (= p95 - p05), 'rank' (= number of null values
    strictly below observed), 'exceeds' (observed > p95, strict; ties fail), 'mean', 'std'}
    (SPEC 3.2; CR 2.1 item 9: strict exceedance of the 95th percentile, achieved rank reported)."""
    null = np.asarray(null, dtype=np.float64)
    n = int(len(null))
    if n == 0 or (observed is None) or np.isnan(observed):
        return {"observed": observed, "n": n, "p95": None, "p05": None, "spread": None,
                "rank": None, "exceeds": False, "mean": None, "std": None}
    p95 = float(np.quantile(null, 0.95))
    p05 = float(np.quantile(null, 0.05))
    return {
        "observed": float(observed), "n": n, "p95": p95, "p05": p05, "spread": p95 - p05,
        "rank": int(np.sum(null < observed)), "exceeds": bool(observed > p95),
        "mean": float(null.mean()), "std": float(null.std()),
    }
