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
# Segment-level helpers
# ---------------------------------------------------------------------------

_MIN_WINDOWS_PER_SEGMENT_DEFAULT = 50


def split_trace_into_segments(
    data: np.ndarray, num_segments: int
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for num_segments contiguous, non-overlapping slices.

    The last segment absorbs any remainder from non-integer division so the
    union covers [0, T) exactly.  Does not slice the array.
    """
    T = data.shape[0]
    base_len = T // num_segments
    result: list[tuple[int, int]] = []
    for i in range(num_segments):
        start = i * base_len
        end = (i + 1) * base_len if i < num_segments - 1 else T
        result.append((start, end))
    return result


def _compute_segment_trajectory_metrics(segment_records: list[dict]) -> dict:
    """Compute within-run trajectory statistics from per-segment summary scalars."""
    metric_keys = ["msc_peak_snr_db_median", "plv_median", "cepstral_peak_idx_median"]
    out: dict = {}
    for key in metric_keys:
        vals = [
            r["summary"][key]
            for r in segment_records
            if r["summary"][key] is not None and r["status"] == "ok"
        ]
        if len(vals) < 2:
            out[key] = {"n_ok": len(vals), "note": "insufficient ok segments for trajectory metrics"}
            continue
        arr = np.array(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        out[key] = {
            "n_ok": len(vals),
            "mean": mean,
            "std": std,
            "cv": float(std / mean) if mean != 0.0 else None,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "first_minus_last": float(arr[0] - arr[-1]),
        }
    return out


def _load_run_level_summary(output_root: str, step_name: str) -> dict:
    """Load the run-level streaming.json summary for segment-to-run delta computation."""
    json_path = Path(output_root) / "offline" / step_name / "streaming.json"
    if json_path.is_file():
        try:
            with open(json_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _compute_segment_to_run_deltas(
    segment_records: list[dict], run_level_summary: dict
) -> dict:
    """Per-segment delta from the run-level value for each key present in both."""
    if not run_level_summary:
        return {}
    metric_map = {
        "msc_peak_snr_db_median": run_level_summary.get("msc_peak_snr_db_median"),
        "plv_median": run_level_summary.get("plv_median"),
    }
    out: dict = {}
    for key, run_val in metric_map.items():
        if run_val is None:
            continue
        deltas = []
        for r in segment_records:
            seg_val = r["summary"].get(key)
            delta = (seg_val - run_val) if (seg_val is not None and r["status"] == "ok") else None
            deltas.append({"segment_index": r["segment_index"], "delta": delta})
        out[key] = {"run_level_value": run_val, "per_segment_delta": deltas}
    return out


def analyze_trace_segment(
    segment_data: np.ndarray,
    baseline: np.ndarray,
    out_dir: Path,
    window_size: int,
    step_size: int,
) -> dict:
    """Run the existing metric pipeline on one temporal segment of a trace.

    Mirrors run_metrics_for_step for a single [L, N] slice.  Writes streaming.npz,
    streaming.json, and plv_baseline_aware.json under out_dir.

    Returns a summary dict for inclusion in segment_meta.json.  Tolerates partial
    failure: exceptions are caught per-block, recorded in the returned dict, and
    logged.  This ensures a single failing segment does not drop others.
    """
    from stability_validator import run_all_features_streaming, StabilityValidator  # type: ignore[import]

    L, N = segment_data.shape
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "msc_peak_snr_db_median": None,
        "plv_median": None,
        "cepstral_peak_idx_median": None,
        "status": "ok",
        "error": None,
    }

    # -- streaming metrics: MSC, Cepstrum, online PLV --
    streaming_prefix = str(out_dir / "streaming")
    try:
        run_all_features_streaming(
            segment_data,
            window_size=window_size,
            step_size=step_size,
            output_prefix=streaming_prefix,
        )
        streaming_json_path = Path(f"{streaming_prefix}.json")
        if streaming_json_path.is_file():
            with open(streaming_json_path, encoding="utf-8") as fh:
                s = json.load(fh)
            summary["msc_peak_snr_db_median"] = s.get("msc_peak_snr_db_median")
            summary["plv_median"] = s.get("plv_median")
            cep = s.get("cepstral_peak_idx_median")
            # save a scalar even if the JSON stored a list
            if isinstance(cep, list):
                summary["cepstral_peak_idx_median"] = cep[0] if cep else None
            else:
                summary["cepstral_peak_idx_median"] = cep
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        print(f"[OFFLINE]   WARNING: segment streaming metrics failed: {exc}")

    # -- baseline-aware PLV (full segment + sliding windows) --
    try:
        sv = StabilityValidator(numPages=N, window_size=window_size, window_step=step_size)
        sv.plv_helper.baseline_plv = baseline

        plv_full: dict = {}
        try:
            plv_full = _make_json_safe(sv.compute_plv_features(segment_data))
        except Exception as exc:
            print(f"[OFFLINE]   WARNING: segment PLV full-run failed: {exc}")

        plv_sliding: dict[str, dict] = {}
        for start in range(0, L - window_size + 1, step_size):
            end = min(start + window_size, L)
            key = f"wind_{start}_{end}"
            try:
                plv_sliding[key] = _make_json_safe(
                    sv.compute_plv_features(segment_data[start:end, :])
                )
            except Exception as exc:
                print(f"[OFFLINE]   WARNING: segment PLV window {start}:{end} failed: {exc}")

        plv_out = {"full_run": plv_full, "sliding_windows": plv_sliding}
        with open(out_dir / "plv_baseline_aware.json", "w", encoding="utf-8") as fh:
            json.dump(plv_out, fh, indent=2)
    except Exception as exc:
        if summary["status"] == "ok":
            summary["status"] = "partial"
        err_msg = str(exc)
        summary["error"] = (f"{summary['error']}; PLV: {err_msg}" if summary["error"] else f"PLV: {err_msg}")
        print(f"[OFFLINE]   WARNING: segment PLV pass failed: {exc}")

    return summary


def run_segment_metrics_for_step(
    data: np.ndarray,
    baseline: np.ndarray,
    step_name: str,
    output_root: str,
    window_size: int,
    step_size: int,
    num_segments: int,
    min_windows_per_segment: int = _MIN_WINDOWS_PER_SEGMENT_DEFAULT,
) -> None:
    """Split the trace into num_segments contiguous temporal segments and run metrics on each.

    Writes all outputs under <output_root>/offline/<step_name>/segments/.
    The run-level outputs written by run_metrics_for_step are NOT modified.
    This function is additive.

    Segments are contiguous temporal sub-observations of one recording, NOT
    independent samples.  The mandatory limits_of_inference block in
    segment_meta.json records this constraint for downstream consumers.
    """
    T, N = data.shape
    seg_root = Path(output_root) / "offline" / step_name / "segments"
    seg_root.mkdir(parents=True, exist_ok=True)

    base_len = T // num_segments
    tail_frames = T - base_len * num_segments  # absorbed by last segment

    # Minimum frame count for min_windows_per_segment windows
    min_frames = (min_windows_per_segment - 1) * step_size + window_size

    segment_indices = split_trace_into_segments(data, num_segments)

    any_too_short = False
    for i, (start, end) in enumerate(segment_indices):
        seg_len = end - start
        if seg_len < min_frames:
            print(
                f"[OFFLINE]   WARNING: segment {i} has {seg_len} frames,"
                f" below minimum {min_frames} for {min_windows_per_segment} windows"
                f" (window={window_size}, hop={step_size})."
                " Metrics will be attempted but may be unreliable."
            )
            any_too_short = True

    segment_records: list[dict] = []
    any_failed = False
    min_windows_observed: int | None = None

    for i, (start, end) in enumerate(segment_indices):
        seg_label = f"seg_{i:02d}"
        seg_out_dir = seg_root / seg_label
        seg_len = end - start
        n_windows_expected = max(0, (seg_len - window_size) // step_size + 1)

        print(
            f"[OFFLINE]   segment {i}/{num_segments - 1}:"
            f" frames [{start}, {end})  len={seg_len}  expected_windows={n_windows_expected}"
        )

        seg_summary = analyze_trace_segment(
            data[start:end, :], baseline, seg_out_dir, window_size, step_size
        )

        if seg_summary["status"] != "ok":
            any_failed = True

        if min_windows_observed is None or n_windows_expected < min_windows_observed:
            min_windows_observed = n_windows_expected

        segment_records.append({
            "segment_index": i,
            "frame_start": start,
            "frame_end": end,
            "n_frames": seg_len,
            "n_windows_expected": n_windows_expected,
            "window_size": window_size,
            "hop": step_size,
            "files": {
                "streaming_npz": f"{seg_label}/streaming.npz",
                "streaming_json": f"{seg_label}/streaming.json",
                "plv_baseline_aware_json": f"{seg_label}/plv_baseline_aware.json",
            },
            "summary": {
                "msc_peak_snr_db_median": seg_summary["msc_peak_snr_db_median"],
                "plv_median": seg_summary["plv_median"],
                "cepstral_peak_idx_median": seg_summary["cepstral_peak_idx_median"],
            },
            "status": seg_summary["status"],
            "error": seg_summary["error"],
        })

    trajectory_metrics = _compute_segment_trajectory_metrics(segment_records)
    run_level_summary = _load_run_level_summary(output_root, step_name)
    segment_to_run_deltas = _compute_segment_to_run_deltas(segment_records, run_level_summary)

    segment_meta: dict = {
        "schema_version": "0.1",
        "step_name": step_name,
        "config": {
            "num_segments": num_segments,
            "window_size": window_size,
            "step_size": step_size,
            "min_windows_per_segment": min_windows_per_segment,
            "metric_set_id": "current_per_run_v1",
            "trace_kind": "complex",
            "warmup_segments_excluded": 0,
        },
        "trace": {
            "n_frames_total": T,
            "n_pages": N,
            "segment_length_frames": base_len,
            "tail_frames_in_last_segment": tail_frames,
        },
        "segments": segment_records,
        "trajectory_metrics": trajectory_metrics,
        "segment_to_run_deltas": segment_to_run_deltas,
        "diagnostics": {
            "any_segment_too_short": any_too_short,
            "any_segment_failed": any_failed,
            "min_windows_observed": min_windows_observed,
        },
        "limits_of_inference": {
            "segments_are_independent": False,
            "use_for_class_significance_tests": False,
            "valid_inferential_units": ["run", "subtype"],
            "cv_grouping": "by_run",
            "note": (
                "Segments are contiguous temporal sub-observations of one recording."
                " They are NOT independent samples. Do not pool segments across runs"
                " to inflate n. See segment_level_analysis_critique_and_plan.md."
            ),
        },
    }

    meta_path = seg_root / "segment_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(_make_json_safe(segment_meta), fh, indent=2)
    print(f"[OFFLINE] Segment pass complete -> {seg_root}  (num_segments={num_segments})")


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
    parser.add_argument(
        "--segments",
        type=int,
        default=1,
        help=(
            "Number of contiguous temporal segments to split the trace into before running"
            " the metric pipeline.  Default 1 (no segmentation; current behavior preserved)."
            " When > 1, segment outputs are written under segments/seg_NN/ next to the"
            " per-run outputs.  Segments are NOT independent samples;"
            " see segment_level_analysis_critique_and_plan.md before downstream use."
        ),
    )
    parser.add_argument(
        "--min-windows-per-segment",
        type=int,
        default=_MIN_WINDOWS_PER_SEGMENT_DEFAULT,
        help=(
            f"Minimum number of windows required per segment for reliable metric estimates"
            f" (default {_MIN_WINDOWS_PER_SEGMENT_DEFAULT})."
            " Segments below this threshold emit a warning but are still attempted."
        ),
    )
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
    if args.segments < 1:
        print(f"[OFFLINE] ERROR: --segments must be >= 1, got {args.segments}")
        return 1

    if not os.path.isfile(args.matrix):
        print(f"[OFFLINE] ERROR: matrix file not found: {args.matrix}")
        return 1

    data = load_matrix(args.matrix)
    T, N = data.shape
    if T == 0 or N == 0:
        print(f"[OFFLINE] ERROR: matrix is empty (shape={data.shape}): {args.matrix}")
        return 1
    print(f"[OFFLINE] Loaded matrix: T={T} frames, N={N} pages  ({args.matrix})")

    if args.segments > T:
        print(
            f"[OFFLINE] ERROR: --segments {args.segments} exceeds frame count T={T}."
            " Reduce --segments or check the matrix."
        )
        return 1

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
    # Run all metrics for this step (full-run; always executed)
    # ------------------------------------------------------------------
    run_metrics_for_step(
        data=data,
        baseline=baseline,
        step_name=args.step_name,
        output_root=args.output_root,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    # ------------------------------------------------------------------
    # Optional segment-level pass (additive; does not modify run outputs)
    # ------------------------------------------------------------------
    if args.segments > 1:
        print(f"[OFFLINE] Starting segment pass: num_segments={args.segments}")
        run_segment_metrics_for_step(
            data=data,
            baseline=baseline,
            step_name=args.step_name,
            output_root=args.output_root,
            window_size=args.window_size,
            step_size=args.step_size,
            num_segments=args.segments,
            min_windows_per_segment=args.min_windows_per_segment,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
