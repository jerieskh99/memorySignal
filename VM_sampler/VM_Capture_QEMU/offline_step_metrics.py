#!/usr/bin/env python3
"""
Per-test offline metrics runner for the step-gated capture pipeline.

Called by run_files_controlled.py after each workload step's capture queue
has fully drained.  It:

  1. Loads the step-scoped run_matrix.npy  (shape on disk: [pages, frames]).
  2. Transposes to [T=frames, N=pages] — the convention expected by all
     coherence_temp_spec_stability modules.
  3. For the *baseline* step (idle, step 1 by default):
       - Computes the per-page PLV from the clean idle data.
       - Persists it to  <baseline_dir>/baseline_plv.npy  for reuse by all
         subsequent steps.
  4. For every step (including the baseline step itself):
       - Runs MSC / Cepstrum / online-PLV via run_all_features_streaming().
       - Runs baseline-aware PLV evaluation (per-page drop, anomaly counts,
         sliding windows) using the stored baseline.
       - Writes all outputs under  <output_root>/offline/<step_name>/.

Outputs
-------
  <output_root>/offline/<step_name>/
      meta.json                  — frame count, num_pages, config
      streaming.<ext>            — MSC / Cepstrum / online-PLV results
      plv_baseline_aware.json    — full-run + sliding-window baseline-PLV results
  <baseline_dir>/
      baseline_plv.npy           — shared baseline (written only for --is-baseline)

Usage
-----
  python3 offline_step_metrics.py \\
      --matrix   /path/to/run_matrix_test1_run_idle.npy \\
      --step-name test1_run_idle \\
      --output-root /project/homes/jeries/memory_traces/output_dir \\
      --project-root /project/homes/jeries/memorySignal \\
      --baseline-dir /project/homes/jeries/memory_traces/output_dir/offline/baseline \\
      [--is-baseline] \\
      [--window-size 128] \\
      [--step-size 64]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_matrix(matrix_path: str) -> np.ndarray:
    """Load [pages, frames] .npy and return [T, N] = [frames, pages].

    Preserve dtype (real or complex). Complex deltas are required for
    phase-aware metrics (PLV uses np.angle internally).
    """
    mat = np.load(matrix_path, mmap_mode="r")
    # Consumer stores (pages, frames); all metric modules expect (time, pages).
    return np.asarray(mat.T)


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------


def compute_and_save_baseline(data: np.ndarray, baseline_path: str) -> np.ndarray:
    """Compute per-page PLV baseline from clean idle data [T, N] and persist it.

    The PLV is computed as |mean(exp(j·angle(data)))| across the time axis,
    which is the same formula used by PLVStability._computePLV.

    Returns the baseline array of shape [N].
    """
    from plv_calcolator import PLVStability  # type: ignore[import]

    ps = PLVStability()
    baseline = ps.fit_baseline(data)  # shape [N]
    Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(baseline_path, baseline)
    print(f"[OFFLINE] Baseline PLV saved  -> {baseline_path}  (shape={baseline.shape})")
    return baseline


def load_baseline(baseline_path: str, expected_n: int) -> np.ndarray:
    """Load and validate the shared baseline PLV array."""
    baseline: np.ndarray = np.load(baseline_path)
    print(f"[OFFLINE] Loaded baseline     -> {baseline_path}  (shape={baseline.shape})")
    if baseline.shape[0] != expected_n:
        raise ValueError(
            f"Baseline shape {baseline.shape} does not match matrix N={expected_n}."
            " Was the baseline computed from a different VM memory size?"
        )
    return baseline


# ---------------------------------------------------------------------------
# Metric runner
# ---------------------------------------------------------------------------


def run_metrics_for_step(
    data: np.ndarray,
    baseline: np.ndarray,
    step_name: str,
    output_root: str,
    window_size: int,
    step_size: int,
) -> None:
    """Run all offline metrics for one step and write outputs.

    Parameters
    ----------
    data:        [T, N] float64 array (frames x pages).
    baseline:    [N] per-page PLV baseline from the clean idle step.
    step_name:   Label used for the output subfolder.
    output_root: Root directory; outputs go to  <output_root>/offline/<step_name>/.
    window_size: Window length for MSC / Cepstrum / sliding PLV.
    step_size:   Hop size for sliding windows.
    """
    from stability_validator import run_all_features_streaming, StabilityValidator  # type: ignore[import]

    T, N = data.shape
    out_dir = Path(output_root) / "offline" / step_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[OFFLINE] step={step_name}  T={T} frames  N={N} pages"
        f"  window={window_size}  step={step_size}  output={out_dir}"
    )

    if T < 2:
        print(f"[OFFLINE] WARNING: only {T} frame(s) — skipping metrics (need >= 2).")
        _write_meta(out_dir, step_name, T, N, window_size, step_size, baseline_used=True)
        return

    # ------------------------------------------------------------------
    # 1. Streaming metrics: MSC, Cepstrum, online PLV (no external baseline)
    # ------------------------------------------------------------------
    streaming_prefix = str(out_dir / "streaming")
    print(f"[OFFLINE]   streaming metrics -> {streaming_prefix}.[npz/json]")
    try:
        run_all_features_streaming(
            data,
            window_size=window_size,
            step_size=step_size,
            output_prefix=streaming_prefix,
        )
    except Exception as exc:
        print(f"[OFFLINE]   WARNING: streaming metrics failed: {exc}")

    # ------------------------------------------------------------------
    # 2. Baseline-aware PLV (uses the shared idle baseline)
    # ------------------------------------------------------------------
    print(f"[OFFLINE]   baseline-aware PLV ...")
    sv = StabilityValidator(numPages=N, window_size=window_size, window_step=step_size)
    # Inject the pre-computed baseline so StabilityValidator does not need to
    # fit a new baseline from the current (potentially non-idle) data.
    sv.plv_helper.baseline_plv = baseline

    plv_full: dict = {}
    try:
        plv_full = sv.compute_plv_features(data)
        # numpy arrays are not JSON-serialisable; convert to plain Python types.
        plv_full = _make_json_safe(plv_full)
    except Exception as exc:
        print(f"[OFFLINE]   WARNING: PLV full-run failed: {exc}")

    # Sliding-window PLV for finer temporal resolution.
    plv_sliding: dict[str, dict] = {}
    for start in range(0, T - window_size + 1, step_size):
        end = min(start + window_size, T)
        key = f"wind_{start}_{end}"
        try:
            win_feat = sv.compute_plv_features(data[start:end, :])
            plv_sliding[key] = _make_json_safe(win_feat)
        except Exception as exc:
            print(f"[OFFLINE]   WARNING: PLV window {start}:{end} failed: {exc}")

    plv_out = {"full_run": plv_full, "sliding_windows": plv_sliding}
    plv_path = out_dir / "plv_baseline_aware.json"
    with open(plv_path, "w", encoding="utf-8") as fh:
        json.dump(plv_out, fh, indent=2)
    print(f"[OFFLINE]   PLV saved -> {plv_path}")

    # ------------------------------------------------------------------
    # 3. Step metadata
    # ------------------------------------------------------------------
    _write_meta(out_dir, step_name, T, N, window_size, step_size, baseline_used=True)
    print(f"[OFFLINE] Step complete -> {out_dir}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_json_safe(obj: object) -> object:
    """Recursively convert numpy types to plain Python scalars/lists."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _write_meta(
    out_dir: Path,
    step_name: str,
    num_frames: int,
    num_pages: int,
    window_size: int,
    step_size: int,
    baseline_used: bool,
) -> None:
    meta = {
        "step_name": step_name,
        "num_frames": num_frames,
        "num_pages": num_pages,
        "window_size": window_size,
        "step_size": step_size,
        "baseline_used": baseline_used,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-test offline metrics runner for step-gated capture pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--matrix", required=True,
        help="Path to the step-scoped run_matrix.npy file.",
    )
    parser.add_argument(
        "--step-name", required=True,
        help="Human-readable step label (e.g. test1_run_idle).  Used as the output subfolder.",
    )
    parser.add_argument(
        "--output-root", required=True,
        help="Root directory for all offline outputs.",
    )
    parser.add_argument(
        "--project-root", required=True,
        help=(
            "Absolute path to the memorySignal repo root."
            " Added to sys.path so coherence_temp_spec_stability modules import correctly."
        ),
    )
    parser.add_argument(
        "--baseline-dir", required=True,
        help="Directory where baseline_plv.npy is saved (--is-baseline) or loaded from.",
    )
    parser.add_argument(
        "--is-baseline", action="store_true",
        help=(
            "Compute and persist the PLV baseline from this step's data."
            " Should be set only for the first (clean idle) step."
        ),
    )
    parser.add_argument("--window-size", type=int, default=128, help="Window length (default 128).")
    parser.add_argument("--step-size", type=int, default=64, help="Hop size (default 64).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Add coherence_temp_spec_stability modules to the import path.
    # ------------------------------------------------------------------
    coh_dir = os.path.join(args.project_root, "coherence_temp_spec_stability")
    for p in (coh_dir, args.project_root):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.isfile(args.matrix):
        print(f"[OFFLINE] ERROR: matrix file not found: {args.matrix}")
        return 1

    data = load_matrix(args.matrix)
    T, N = data.shape
    if T == 0 or N == 0:
        print(f"[OFFLINE] ERROR: matrix is empty (shape={data.shape}): {args.matrix}")
        return 1
    print(f"[OFFLINE] Loaded matrix: T={T} frames, N={N} pages  ({args.matrix})")

    # ------------------------------------------------------------------
    # Baseline: compute (idle step) or load (all other steps)
    # ------------------------------------------------------------------
    baseline_path = os.path.join(args.baseline_dir, "baseline_plv.npy")

    if args.is_baseline:
        print(f"[OFFLINE] Computing PLV baseline from {T} frames (idle step) ...")
        baseline = compute_and_save_baseline(data, baseline_path)
    else:
        if not os.path.isfile(baseline_path):
            print(f"[OFFLINE] ERROR: baseline file missing: {baseline_path}")
            print(
                "[OFFLINE] The baseline must be created first by running the idle step"
                " with --is-baseline."
            )
            return 1
        try:
            baseline = load_baseline(baseline_path, expected_n=N)
        except ValueError as exc:
            print(f"[OFFLINE] ERROR: {exc}")
            return 1

    # ------------------------------------------------------------------
    # Run all metrics for this step
    # ------------------------------------------------------------------
    run_metrics_for_step(
        data=data,
        baseline=baseline,
        step_name=args.step_name,
        output_root=args.output_root,
        window_size=args.window_size,
        step_size=args.step_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
