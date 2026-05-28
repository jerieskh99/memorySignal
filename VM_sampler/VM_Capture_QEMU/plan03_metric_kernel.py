#!/usr/bin/env python3
"""plan03_metric_kernel.py -- Plan 03 per-(cell, window, hop) kernel.

Owner: DE (Senior Data Engineer) + EN (Engineering Skills).
Implements proposal section 05.1 of
``docs/plan03_window_hop_proposal.html``.

Pure-function kernel. Given a 1-D active-page-fraction trajectory plus an
analyzer (window, hop) combo and optional ground-truth phase indices,
return a dict of metrics ready for the sweep CSV writer.

Design deltas applied (proposal section 01):
  * Delta-3 · 1-D APF path via [T, 1] reshape (cepstrum module reused).
  * R2 · ``min_quef_idx`` override to ``max(1, T // 8)`` so small-T cells
    do not have their entire spectrum discarded by the module default.
  * Dispute 5 · stationarity test implemented inline (1-D rolling-mean
    drift vs global std). NOT the 2-D ``StabilityValidator`` path.

This module has no I/O. Easy to unit-test on synthetic ramps.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

# Plan 02 helpers reused as-is (proposal section 5.1 + table 5).
from plan02_metrics_per_cell import (
    compute_n_windows,
    cv_workingset,
    detect_boundaries_diff,
    f1_score,
)

# Make the sibling ``coherence_temp_spec_stability`` package importable
# whether the kernel is invoked from VM_Capture_QEMU/ or repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coherence_temp_spec_stability.cepstrum_stability import CepstrumStability  # noqa: E402


METRIC_KEYS = (
    "n_pairs", "n_windows", "apf_mean", "apf_std",
    "cv_workingset", "f1_phase",
    "cepstral_peak_idx", "ceps_peak_snr_db",
    "stat_pass_frac", "status",
)


def stationarity_per_window(traj_1d: list[float], window: int,
                            hop: int) -> float | None:
    """Per-window stationarity pass fraction (1-D APF path).

    A window passes if ``abs(window_mean - global_mean) / global_std < 1.0``.
    If the global std is zero (constant trajectory) every window passes
    trivially. If fewer than one window exists the function returns None
    so the caller can distinguish "no test was run" from "all failed".

    Proposal dispute 5 fixes the canonical naming confusion: this is a
    1-D stationarity test, NOT the 2-D ``StabilityValidator`` PLV/MSC
    path. ~15 LOC, lives inline per the day-3 decision.
    """
    n = len(traj_1d)
    if n < window:
        return None
    n_windows = max(0, (n - window) // hop + 1)
    if n_windows < 1:
        return None
    g_mean = statistics.fmean(traj_1d)
    g_std = statistics.pstdev(traj_1d) if n >= 2 else 0.0
    if g_std == 0.0:
        return 1.0
    passes = 0
    for i in range(n_windows):
        start = i * hop
        chunk = traj_1d[start:start + window]
        if not chunk:
            continue
        w_mean = statistics.fmean(chunk)
        if abs(w_mean - g_mean) / g_std < 1.0:
            passes += 1
    return passes / n_windows


def _skip_short_result(n_pairs: int) -> dict[str, Any]:
    return {
        "n_pairs": n_pairs, "n_windows": 0,
        "apf_mean": None, "apf_std": None,
        "cv_workingset": None, "f1_phase": None,
        "cepstral_peak_idx": None, "ceps_peak_snr_db": None,
        "stat_pass_frac": None, "status": "skip:short",
    }


def score(traj: list[float], window: int, hop: int, *,
          phase_marker_indices: list[int] | None = None,
          workload_type: str = "unknown") -> dict[str, Any]:
    """Compute Plan 03 metrics for one (cell, window, hop) combo.

    Parameters
    ----------
    traj
        1-D APF trajectory (no sentinel line, no NaN).
    window, hop
        Analyzer sliding-window size and hop, in snap-pair units.
    phase_marker_indices
        Ground-truth boundary indices in snap-pair space; only consumed
        when ``workload_type == "phasic"``. May be None.
    workload_type
        ``"phasic"`` -> F1 vs markers; ``"steady"`` -> CV; otherwise
        neither defining metric is filled.

    Returns
    -------
    dict
        Keys per ``METRIC_KEYS`` plus the ``status`` literal:
        ``"ok"`` | ``"skip:short"`` | ``"skip:nan"`` | ``"error:<msg>"``.
        Numeric fields are ``None`` when the kernel could not compute
        them (e.g. on a short trajectory or NaN spectrum).
    """
    n_pairs = len(traj)
    if n_pairs < window or window <= 0:
        return _skip_short_result(n_pairs)
    try:
        if not _HAS_NUMPY:
            raise RuntimeError("numpy is required for plan03 cepstrum path")
        x = np.asarray(traj, dtype=float).reshape(-1, 1)  # [T, 1]
        cs = CepstrumStability()
        ceps = cs.compute_cepstrum(x)  # [Q, 1]
        if not np.all(np.isfinite(ceps)):
            result = _skip_short_result(n_pairs)
            result["status"] = "skip:nan"
            return result

        min_quef = max(1, n_pairs // 8)
        # Defensive: if min_quef is past the cepstrum length we cannot
        # compute a peak; treat as skip:nan so the combo is dropped.
        Q = ceps.shape[0]
        if min_quef >= Q:
            result = _skip_short_result(n_pairs)
            result["status"] = "skip:nan"
            return result
        peaks = cs.compute_cepstral_peak(ceps, min_quef_idx=min_quef)
        peak_idx = int(peaks[0])

        ceps_mag = np.abs(ceps[:, 0])
        tail = ceps_mag[min_quef:]
        med = float(np.median(tail)) if tail.size else 0.0
        peak_val = float(ceps_mag[peak_idx])
        if med <= 0.0 or peak_val <= 0.0:
            snr_db: float | None = None
        else:
            snr_db = 10.0 * math.log10(peak_val / med)

        apf_mean = statistics.fmean(traj) if traj else None
        apf_std = statistics.pstdev(traj) if n_pairs >= 2 else None

        stat_pass = stationarity_per_window(traj, window, hop)
        n_windows = compute_n_windows(n_pairs, window, hop)

        f1_value: float | None = None
        if workload_type == "phasic" and phase_marker_indices:
            predicted = detect_boundaries_diff(traj)
            breakdown = f1_score(predicted, phase_marker_indices, tolerance=1)
            f1_value = float(breakdown["f1"])

        cv_value: float | None = None
        if workload_type == "steady":
            cv_value = cv_workingset(traj)

        return {
            "n_pairs": n_pairs,
            "n_windows": n_windows,
            "apf_mean": apf_mean,
            "apf_std": apf_std,
            "cv_workingset": cv_value,
            "f1_phase": f1_value,
            "cepstral_peak_idx": peak_idx,
            "ceps_peak_snr_db": snr_db,
            "stat_pass_frac": stat_pass,
            "status": "ok",
        }
    except Exception as exc:  # noqa: BLE001 -- we want full error surface
        result = _skip_short_result(n_pairs)
        result["status"] = f"error:{type(exc).__name__}:{exc}"[:200]
        return result
