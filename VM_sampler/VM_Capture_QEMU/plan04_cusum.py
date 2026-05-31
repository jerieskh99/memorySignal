"""plan04_cusum.py -- Plan 04 windowed CUSUM change-point detector.

Pure-function library that replaces the window-blind boundary stub
``plan02_metrics_per_cell.detect_boundaries_diff`` with a windowed
two-sided CUSUM detector operating on the rolling mean of a 1-D APF
trajectory. Specified in
``docs/plan04_segmenter_proposal.html`` (Section 05.1).

Algorithm
---------
1. Compute a rolling mean of the APF series at the Plan 03 grid
   ``(W, H) = (8, 4)`` (see :func:`rolling_mean_apf`).
2. Estimate a robust scale via ``MAD(mu) * 1.4826`` (proposal §05.1).
3. Run a two-sided CUSUM with reference value ``k`` (default 2.0,
   the standard 2-sigma rule). Emit a boundary every time either
   accumulator exceeds the decision threshold ``h`` (default 4.0)
   and reset both to zero (the standard reset rule).
4. Merge boundaries within ``min_separation`` window indices.
5. Map window-index boundaries back to snap-index space via
   ``snap_idx = window_idx * H + W // 2`` (window-centre
   convention, proposal §04 R3).

Scope (this module)
-------------------
* No I/O, no CLI. NumPy only.
* All public functions are independently callable (testable in
  isolation).
* Deterministic: identical input produces identical output.

Edge cases
----------
* Empty / short trajectory -> empty boundary list.
* Constant trajectory (MAD = 0 and std = 0) -> empty boundary list.
* MAD = 0 on a non-constant trajectory -> fall back to ``std + eps``
  (``eps = 1e-9``) so the CUSUM can still run.

This module is consumed by:
* ``plan04_sweep.py``      -- per-cell driver (separate file)
* ``plan04_dispatch.py``   -- family-aware scorer (separate file)
* ``plan04_aggregate.py``  -- per-workload aggregator (separate file)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rolling_mean_apf",
    "windowed_cusum",
    "detect_boundaries_cusum",
    "boundaries_to_snap_indices",
    "detect_boundaries_full",
    "stationarity_score",
]

# Numerical floor used when MAD = 0 on a non-constant trajectory.
# Avoids divide-by-zero without materially perturbing the CUSUM
# statistic. Chosen to be far below any realistic APF noise floor.
_EPS_SIGMA: float = 1e-9


# ---------------------------------------------------------------------------
# 1 · windowed rolling mean of the APF series
# ---------------------------------------------------------------------------

def rolling_mean_apf(
    traj_1d: list[float],
    window: int,
    hop: int,
) -> np.ndarray:
    """Windowed rolling mean of a 1-D APF trajectory.

    Mirrors the Plan 03 windowing convention used in
    ``plan03_metric_kernel.stationarity_per_window`` so the
    detector consumes the same series that Plan 03 reports on.

    Parameters
    ----------
    traj_1d
        Active page-fraction trajectory in snap-pair index space.
    window
        Sliding-window length ``W`` (in snap-pair units).
    hop
        Stride between successive window starts ``H``.

    Returns
    -------
    np.ndarray
        Length ``floor((T - W) / H) + 1`` array of window means
        (``float64``). Empty when ``T < W`` or ``window <= 0`` or
        ``hop <= 0``.
    """
    if window <= 0 or hop <= 0:
        return np.empty(0, dtype=np.float64)
    n = len(traj_1d)
    if n < window:
        return np.empty(0, dtype=np.float64)
    n_windows = (n - window) // hop + 1
    if n_windows < 1:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(traj_1d, dtype=np.float64)
    out = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        start = i * hop
        out[i] = arr[start:start + window].mean()
    return out


# ---------------------------------------------------------------------------
# 2 · two-sided CUSUM on the rolling-mean series
# ---------------------------------------------------------------------------

def _robust_scale(rolling_mean: np.ndarray) -> tuple[float, float]:
    """Return ``(mu, sigma)`` for the CUSUM standardisation.

    ``mu`` is the median of the rolling-mean series. ``sigma`` is
    ``MAD * 1.4826`` (proposal §05.1). When MAD = 0 we fall back to
    the (population) standard deviation plus a small epsilon so the
    CUSUM can still run on a near-degenerate input.
    """
    mu = float(np.median(rolling_mean))
    abs_dev = np.abs(rolling_mean - mu)
    mad = float(np.median(abs_dev))
    sigma = mad * 1.4826
    if sigma <= 0.0:
        sigma = float(np.std(rolling_mean)) + _EPS_SIGMA
    return mu, sigma


def windowed_cusum(
    rolling_mean: np.ndarray,
    k: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided CUSUM accumulators on the rolling-mean series.

    Standard recursion (no reset; reset is performed in
    :func:`detect_boundaries_cusum`):

        S_pos[t+1] = max(0, S_pos[t] + z[t] - k)
        S_neg[t+1] = min(0, S_neg[t] + z[t] + k)

    where ``z[t] = (mu[t] - mu_med) / sigma`` is the robustly
    standardised window mean.

    Parameters
    ----------
    rolling_mean
        Output of :func:`rolling_mean_apf` (length ``n_windows``).
    k
        Reference value (slack), in units of ``sigma``. Default 2.0
        (the standard 2-sigma CUSUM rule from proposal §05.1).

    Returns
    -------
    (S_pos, S_neg)
        Both length ``len(rolling_mean)``. ``S_pos >= 0`` and
        ``S_neg <= 0``. Empty arrays when the input is empty.
    """
    n = rolling_mean.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    mu, sigma = _robust_scale(rolling_mean)
    z = (rolling_mean.astype(np.float64) - mu) / sigma

    s_pos = np.zeros(n, dtype=np.float64)
    s_neg = np.zeros(n, dtype=np.float64)
    # D-84 fix: use z[t], matching the docstring formula and the inline
    # math in detect_boundaries_cusum. Previous z[t-1] form was off-by-one
    # and made the exported helper disagree with the detector itself.
    for t in range(1, n):
        s_pos[t] = max(0.0, s_pos[t - 1] + z[t] - k)
        s_neg[t] = min(0.0, s_neg[t - 1] + z[t] + k)
    return s_pos, s_neg


# ---------------------------------------------------------------------------
# 3 · boundary detection in window-index space
# ---------------------------------------------------------------------------

def detect_boundaries_cusum(
    rolling_mean: np.ndarray,
    *,
    k: float = 2.0,
    h: float = 4.0,
    min_separation: int = 2,
) -> list[int]:
    """Run two-sided CUSUM with the standard reset rule.

    A boundary is emitted at window index ``t`` the first time either
    ``S_pos[t] > h`` or ``S_neg[t] < -h``; both accumulators are then
    reset to zero and the recursion continues from ``t + 1``. Adjacent
    boundaries within ``min_separation`` window indices are merged
    (the earliest survives).

    Parameters
    ----------
    rolling_mean
        Output of :func:`rolling_mean_apf`.
    k
        CUSUM reference value (slack), default 2.0.
    h
        Decision threshold, default 4.0 (proposal §05.1).
    min_separation
        Adjacent boundaries within this many window indices are
        merged. Default 2.

    Returns
    -------
    list[int]
        Sorted, unique change-point indices in window-space.
        Empty when no change point is detected, when the input is
        empty, or when the rolling-mean series is constant.
    """
    n = rolling_mean.shape[0]
    if n == 0:
        return []
    # Constant input -> no change points (MAD = 0 and std = 0). The
    # robust-scale fallback would otherwise feed a divide-by-eps
    # statistic into the CUSUM, producing spurious peaks.
    if float(np.std(rolling_mean)) == 0.0:
        return []

    mu, sigma = _robust_scale(rolling_mean)
    z = (rolling_mean.astype(np.float64) - mu) / sigma

    boundaries: list[int] = []
    s_pos = 0.0
    s_neg = 0.0
    for t in range(n):
        s_pos = max(0.0, s_pos + z[t] - k)
        s_neg = min(0.0, s_neg + z[t] + k)
        if s_pos > h or s_neg < -h:
            boundaries.append(t)
            s_pos = 0.0
            s_neg = 0.0

    if not boundaries:
        return []

    # Merge adjacent boundaries within ``min_separation`` window
    # indices. Earliest survivor wins (the reset means the later
    # crossings are tail echoes of the first one).
    merged: list[int] = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] >= min_separation:
            merged.append(b)
    return sorted(set(merged))


# ---------------------------------------------------------------------------
# 4 · window-index -> snap-index mapping
# ---------------------------------------------------------------------------

def boundaries_to_snap_indices(
    window_boundaries: list[int],
    window: int,
    hop: int,
) -> list[int]:
    """Map window-space boundary indices to snap-space.

    Uses the window-centre convention from proposal §04 R3:

        snap_idx = window_idx * hop + window // 2

    so the per-workload boundary tolerance baked into the dispatcher
    absorbs the residual ``W // 2`` snap bias.

    Parameters
    ----------
    window_boundaries
        Output of :func:`detect_boundaries_cusum`.
    window, hop
        Same ``(W, H)`` pair used to build the rolling-mean series.

    Returns
    -------
    list[int]
        Sorted, unique snap-space boundary indices. Empty when the
        input is empty.
    """
    if not window_boundaries:
        return []
    half = window // 2
    snap_indices = [int(b) * int(hop) + half for b in window_boundaries]
    return sorted(set(snap_indices))


# ---------------------------------------------------------------------------
# 5 · high-level convenience wrapper
# ---------------------------------------------------------------------------

def detect_boundaries_full(
    traj_1d: list[float],
    window: int,
    hop: int,
    *,
    k: float = 2.0,
    h: float = 4.0,
    min_separation: int = 2,
) -> list[int]:
    """End-to-end change-point detection on a raw APF trajectory.

    Composes :func:`rolling_mean_apf` ->
    :func:`detect_boundaries_cusum` ->
    :func:`boundaries_to_snap_indices`. This is the entry point the
    Plan 04 sweep driver should call per cell.

    Parameters
    ----------
    traj_1d
        1-D APF trajectory in snap-pair index space (no sentinel).
    window, hop
        Plan 03 ``(W, H)``. For v3 this is universally ``(8, 4)``.
    k, h, min_separation
        Forwarded to :func:`detect_boundaries_cusum`.

    Returns
    -------
    list[int]
        Sorted, unique boundary indices in snap-space. Empty when
        the trajectory is shorter than ``window`` or yields no
        change point.
    """
    if not traj_1d:
        return []
    rolling = rolling_mean_apf(traj_1d, window, hop)
    if rolling.shape[0] == 0:
        return []
    window_boundaries = detect_boundaries_cusum(
        rolling, k=k, h=h, min_separation=min_separation,
    )
    return boundaries_to_snap_indices(window_boundaries, window, hop)


# ---------------------------------------------------------------------------
# 6 · marker-less plausibility (steady gate input)
# ---------------------------------------------------------------------------

def stationarity_score(traj_1d: list[float]) -> float:
    """Marker-less plausibility score for the steady gate.

    Runs the detector at the canonical Plan 04 grid
    ``(W, H) = (8, 4)`` and returns

        1.0 - min(1.0, n_boundaries / 3.0)

    where ``n_boundaries`` is the count returned by
    :func:`detect_boundaries_cusum` (in window-space). A
    perfectly steady cell yields 1.0; a cell with three or more
    spurious boundaries yields 0.0. Used by Plan 04's family-aware
    dispatcher as one input to the steady gate G2.

    Parameters
    ----------
    traj_1d
        1-D APF trajectory.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``. Returns ``1.0`` for the empty or
        too-short trajectory (no spurious boundaries can be
        produced).
    """
    if not traj_1d:
        return 1.0
    rolling = rolling_mean_apf(traj_1d, 8, 4)
    if rolling.shape[0] == 0:
        return 1.0
    n_boundaries = len(detect_boundaries_cusum(rolling))
    return 1.0 - min(1.0, n_boundaries / 3.0)
