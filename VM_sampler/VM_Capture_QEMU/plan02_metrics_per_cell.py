#!/usr/bin/env python3
"""plan02_metrics_per_cell.py -- D-51 analyzer-then-delete hook.

Owner: ML + DS.
Reference: docs/tuning_plans/experiment_audit.md decisions D-44 / D-47 / D-51.

For one Plan-02 cell, this module:
  1. Reads sorted memory_dump-*.raw files in image_dir matching the cell's
     run-window (mtime >= run_start_epoch).
  2. Computes active_page_fraction per snap pair via numpy memmap XOR.
  3. Reads workload_stderr.log for PHASE markers (ground-truth phase
     boundaries) if the cell launched a workload.
  4. Computes:
       - n_windows (sliding-window count under analyzer's window/hop)
       - cv_workingset = std/mean of active_page_fraction across snaps
       - f1_phase via a stub diff-detector against stderr PHASE markers
  5. Writes per-cell metrics.json with full trajectory + summary.
  6. Returns the summary so plan02_run.py can populate analyzer_outputs.

Why a leaner analyzer than offline_step_metrics.py:
  - offline_step_metrics requires run_matrix.npy from the consumer
    + the full MSC/Cepstrum/PLV pipeline. Heavyweight: ~1-3 min per cell.
  - For D-51's immediate needs (populate analyzer_outputs.f1_phase /
    cv_workingset / n_windows), direct dump-pair diff via numpy.memmap
    is sufficient and runs in ~5-10 s per cell on modern SSDs.
  - The leaner computation produces the same active_page_fraction
    trajectory the consumer would have computed. Downstream MSC/PLV
    can wire in later (D-52 / Plan 03) by reading the saved trajectory.

Performance note (Linux SSD, 1 GiB dumps):
  - numpy.memmap read: ~1-2 GB/s
  - 100 dump pairs at 1 GiB each: ~50-100 s
  - 90-cell pilot total: ~1.5-2.5 h analyzer overhead

Safe to run idempotently: a cell with an existing metrics.json is
skipped unless --force is passed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


PHASE_RE = re.compile(
    r"\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]\s+\[PHASE\]\s+"
    r"test=(?P<test>\S+)\s+phase=(?P<phase>\S+)"
)
PAGE_SIZE_DEFAULT = 4096
WINDOW_DEFAULT = 128
HOP_DEFAULT = 64
F1_TOLERANCE_S_DEFAULT = 2.0


# ---------------------------------------------------------------------------
# Active-page-fraction trajectory
# ---------------------------------------------------------------------------

def active_page_fraction(dump_a_path: Path, dump_b_path: Path,
                        page_size: int = PAGE_SIZE_DEFAULT) -> float:
    """Fraction of `page_size`-byte pages that differ between two dumps.

    Implementation: numpy.memmap + reshape + per-page any(axis=1) comparison.
    On a 1 GiB dump pair this runs in ~0.5-1.0 s on a modern SSD.
    """
    if not _HAS_NUMPY:
        raise RuntimeError("numpy required for active_page_fraction")
    a = np.memmap(dump_a_path, dtype=np.uint8, mode="r")
    b = np.memmap(dump_b_path, dtype=np.uint8, mode="r")
    if a.shape != b.shape or a.size == 0:
        return 0.0
    n_pages = a.size // page_size
    if n_pages == 0:
        return 0.0
    # Truncate trailing bytes that don't form a full page
    a = a[:n_pages * page_size].reshape(n_pages, page_size)
    b = b[:n_pages * page_size].reshape(n_pages, page_size)
    differ = (a != b).any(axis=1)
    return float(differ.sum()) / float(n_pages)


def compute_trajectory(dump_paths: list[Path],
                       page_size: int = PAGE_SIZE_DEFAULT) -> list[float]:
    """For an ordered list of N dumps, return list of N-1 active_page_fraction
    values: APF between dump_i and dump_(i+1).
    """
    traj: list[float] = []
    for i in range(len(dump_paths) - 1):
        try:
            v = active_page_fraction(dump_paths[i], dump_paths[i + 1],
                                     page_size=page_size)
        except (OSError, ValueError) as exc:
            print(f"[plan02-metrics] WARN apf pair {i}->{i+1}: {exc}",
                  file=sys.stderr)
            v = 0.0
        traj.append(v)
    return traj


# ---------------------------------------------------------------------------
# Phase markers from workload stderr
# ---------------------------------------------------------------------------

def parse_phase_markers(stderr_path: Path) -> list[tuple[float, str]]:
    """Return list of (epoch_seconds, phase_name) for each [PHASE] line.

    Marker timestamps are ISO 8601 UTC. Returns empty list if no markers
    (e.g. warmup cell or workload didn't emit them).
    """
    if not stderr_path.is_file():
        return []
    out: list[tuple[float, str]] = []
    try:
        text = stderr_path.read_text(errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        m = PHASE_RE.search(line)
        if not m:
            continue
        ts_str = m.group("ts")
        try:
            tm = time.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
            epoch = time.mktime(tm) - time.timezone  # treat as UTC
        except ValueError:
            continue
        out.append((epoch, m.group("phase")))
    return out


# ---------------------------------------------------------------------------
# Snapshot timing -> phase boundary correspondence
# ---------------------------------------------------------------------------

def load_snap_timestamps(jsonl_path: Path) -> list[float]:
    """Return list of t0_before_suspend timestamps per snapshot, in order.
    Used to map phase-marker timestamps into snap-index space.
    """
    if not jsonl_path.is_file():
        return []
    ts: list[float] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "t0_before_suspend" in obj:
                ts.append(float(obj["t0_before_suspend"]))
    return ts


def phase_markers_to_snap_indices(markers: list[tuple[float, str]],
                                  snap_timestamps: list[float]) -> list[int]:
    """For each phase marker time, return the snap index nearest to it."""
    if not markers or not snap_timestamps:
        return []
    idx: list[int] = []
    for (t, _phase) in markers:
        best_i = 0
        best_dt = abs(snap_timestamps[0] - t)
        for i, st in enumerate(snap_timestamps):
            dt = abs(st - t)
            if dt < best_dt:
                best_dt = dt
                best_i = i
        idx.append(best_i)
    return sorted(set(idx))


# ---------------------------------------------------------------------------
# F1 stub detector and scoring
# ---------------------------------------------------------------------------

def detect_boundaries_diff(trajectory: list[float]) -> list[int]:
    """Stub change-point detector: emit indices where |Δ trajectory| exceeds
    median + 1.5σ. Inspired by mp_phase_boundary_inference.py:predict_diff.

    Returns sorted unique snap indices (in trajectory-index space, which
    equals snap_pair index = snap index of dump_b in the pair).
    """
    if len(trajectory) < 3:
        return []
    deltas = [abs(trajectory[i + 1] - trajectory[i])
              for i in range(len(trajectory) - 1)]
    if not deltas:
        return []
    med = statistics.median(deltas)
    sd = statistics.stdev(deltas) if len(deltas) >= 2 else 0.0
    threshold = med + 1.5 * sd
    out = [i + 1 for i, d in enumerate(deltas) if d > threshold]
    return sorted(set(out))


def f1_score(predicted: list[int], truth: list[int],
             tolerance: int = 1) -> dict:
    """Match predicted boundaries to truth boundaries within ±tolerance
    snap indices. Returns {tp, fp, fn, precision, recall, f1}."""
    matched_truth: set[int] = set()
    matched_pred: set[int] = set()
    for pi, p in enumerate(predicted):
        for ti, t in enumerate(truth):
            if ti in matched_truth:
                continue
            if abs(p - t) <= tolerance:
                matched_pred.add(pi)
                matched_truth.add(ti)
                break
    tp = len(matched_pred)
    fp = len(predicted) - tp
    fn = len(truth) - len(matched_truth)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
            "n_predicted": len(predicted), "n_truth": len(truth)}


# ---------------------------------------------------------------------------
# Window math
# ---------------------------------------------------------------------------

def compute_n_windows(n_snaps: int, window: int = WINDOW_DEFAULT,
                      hop: int = HOP_DEFAULT) -> int:
    """Standard sliding-window count: floor((N - window) / hop) + 1."""
    if n_snaps < window:
        return 0
    return max(0, (n_snaps - window) // hop + 1)


def cv_workingset(trajectory: list[float]) -> float | None:
    """CV = std/mean of the active_page_fraction trajectory.
    Returns None if mean is 0 or trajectory has < 2 samples.
    """
    if len(trajectory) < 2:
        return None
    m = statistics.fmean(trajectory)
    if m == 0:
        return None
    return statistics.stdev(trajectory) / m


# ---------------------------------------------------------------------------
# Dataclass for per-cell summary
# ---------------------------------------------------------------------------

@dataclass
class PerCellMetrics:
    cell_id: str
    n_dumps_examined: int
    n_pairs_examined: int
    n_snapshots_total: int
    n_windows: int
    f1_phase: float | None = None
    cv_workingset: float | None = None
    f1_breakdown: dict | None = None
    apf_min: float | None = None
    apf_max: float | None = None
    apf_mean: float | None = None
    apf_std: float | None = None
    phase_markers_count: int = 0
    truth_boundary_indices: list[int] = field(default_factory=list)
    predicted_boundary_indices: list[int] = field(default_factory=list)
    apf_trajectory: list[float] = field(default_factory=list)
    workload_type: str = "unknown"
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main per-cell entry point
# ---------------------------------------------------------------------------

def load_streaming_trajectory(apf_jsonl: Path) -> tuple[list[float], dict | None]:
    """B+3.1 Δ-3: read the streaming apf_trajectory.jsonl.

    Returns (trajectory, sentinel) where:
      - trajectory: APF values sorted by seq (skips gaps)
      - sentinel: the final-sentinel dict (None if missing)
    """
    if not apf_jsonl.is_file():
        return [], None
    pairs: list[tuple[int, float]] = []
    sentinel: dict | None = None
    try:
        for raw in apf_jsonl.read_text(errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("final") is True:
                sentinel = obj
                continue
            seq = obj.get("seq")
            apf = obj.get("apf")
            if isinstance(seq, int) and isinstance(apf, (int, float)):
                pairs.append((seq, float(apf)))
    except OSError:
        return [], None
    pairs.sort(key=lambda t: t[0])
    return [v for _, v in pairs], sentinel


def compute_metrics_for_cell(
    cell_id: str,
    image_dir: Path,
    run_start_epoch: float,
    jsonl_path: Path,
    workload_stderr_path: Path | None,
    workload_type: str,
    window: int = WINDOW_DEFAULT,
    hop: int = HOP_DEFAULT,
    page_size: int = PAGE_SIZE_DEFAULT,
    apf_trajectory_max_len: int = 4096,
    streaming_apf_jsonl: Path | None = None,
) -> PerCellMetrics:
    """Compute analyzer_outputs metrics for one cell.

    workload_type ∈ {"phasic", "steady", "warmup", "unknown"} drives which
    summary metric (f1_phase vs cv_workingset) is populated.

    Caller is responsible for ensuring dumps exist (i.e. cell ran with
    keep_dumps=True). If no dumps are found, returns a PerCellMetrics
    with n_dumps_examined=0 and notes explaining why.
    """
    notes: list[str] = []
    snap_timestamps = load_snap_timestamps(jsonl_path)
    n_snapshots_total = len(snap_timestamps)

    # B+3.1: prefer streaming trajectory when available (dumps already
    # deleted by async helpers). Fall back to dump-scan otherwise.
    traj: list[float] = []
    n_dumps_examined = 0
    if streaming_apf_jsonl is not None and streaming_apf_jsonl.is_file():
        traj, sentinel = load_streaming_trajectory(streaming_apf_jsonl)
        n_dumps_examined = len(traj) + 1 if traj else 0  # implied dump count
        if sentinel is not None:
            notes.append(
                f"streaming APF: {len(traj)} pairs · sentinel n_ok="
                f"{sentinel.get('n_ok')} expected="
                f"{sentinel.get('n_pairs_expected')} "
                f"gaps={sentinel.get('gap_seqs') or []}"
            )
        else:
            notes.append(f"streaming APF: {len(traj)} pairs · NO sentinel found")
    else:
        # Find dump files newer than run_start_epoch (this cell's dumps only)
        dumps: list[Path] = []
        if image_dir.is_dir():
            for p in image_dir.glob("memory_dump-*.raw"):
                try:
                    if p.stat().st_mtime >= run_start_epoch - 1.0:
                        dumps.append(p)
                except FileNotFoundError:
                    continue
        dumps.sort()
        if not dumps:
            notes.append("no dumps found in image_dir matching run window; "
                         "metrics empty (cell likely had keep_dumps=False)")
            return PerCellMetrics(
                cell_id=cell_id, n_dumps_examined=0, n_pairs_examined=0,
                n_snapshots_total=n_snapshots_total,
                n_windows=compute_n_windows(n_snapshots_total, window, hop),
                workload_type=workload_type, notes=notes,
            )
        if not _HAS_NUMPY:
            notes.append("numpy unavailable; trajectory not computed")
            return PerCellMetrics(
                cell_id=cell_id, n_dumps_examined=len(dumps), n_pairs_examined=0,
                n_snapshots_total=n_snapshots_total,
                n_windows=compute_n_windows(n_snapshots_total, window, hop),
                workload_type=workload_type, notes=notes,
            )
        traj = compute_trajectory(dumps, page_size=page_size)
        n_dumps_examined = len(dumps)
        notes.append(f"dump-scan APF: {len(traj)} pairs "
                     f"from {len(dumps)} dumps")

    summary_apf = {
        "min": min(traj) if traj else None,
        "max": max(traj) if traj else None,
        "mean": statistics.fmean(traj) if traj else None,
        "std": statistics.stdev(traj) if len(traj) >= 2 else None,
    }

    # CV (used for steady workloads)
    cv = cv_workingset(traj)

    # F1 (used for phasic workloads)
    f1_value: float | None = None
    f1_breakdown: dict | None = None
    truth_idx: list[int] = []
    pred_idx: list[int] = []
    n_markers = 0
    if workload_stderr_path and workload_stderr_path.is_file():
        markers = parse_phase_markers(workload_stderr_path)
        n_markers = len(markers)
        if markers and snap_timestamps:
            truth_idx = phase_markers_to_snap_indices(markers, snap_timestamps)
            pred_idx = detect_boundaries_diff(traj)
            if truth_idx:
                f1_breakdown = f1_score(pred_idx, truth_idx, tolerance=1)
                f1_value = f1_breakdown["f1"]
                notes.append(
                    f"f1 detector: predicted={len(pred_idx)} "
                    f"truth={len(truth_idx)} f1={f1_value:.3f}"
                )
            else:
                notes.append("phase markers present but truth indices empty "
                             "(snap timestamps may not overlap marker window)")
        elif not markers:
            notes.append("no [PHASE] markers in workload stderr; F1 left null")

    # Compact trajectory if too long (avoid bloating per-cell JSON)
    if len(traj) > apf_trajectory_max_len:
        # Downsample evenly
        step = max(1, len(traj) // apf_trajectory_max_len)
        traj_compact = traj[::step]
        notes.append(f"trajectory downsampled {len(traj)} -> "
                     f"{len(traj_compact)} for json (full retained in "
                     f"metrics.jsonl)")
    else:
        traj_compact = list(traj)

    return PerCellMetrics(
        cell_id=cell_id,
        n_dumps_examined=n_dumps_examined,
        n_pairs_examined=len(traj),
        n_snapshots_total=n_snapshots_total,
        n_windows=compute_n_windows(n_snapshots_total, window, hop),
        f1_phase=f1_value,
        cv_workingset=cv,
        f1_breakdown=f1_breakdown,
        apf_min=summary_apf["min"],
        apf_max=summary_apf["max"],
        apf_mean=summary_apf["mean"],
        apf_std=summary_apf["std"],
        phase_markers_count=n_markers,
        truth_boundary_indices=truth_idx,
        predicted_boundary_indices=pred_idx,
        apf_trajectory=traj_compact,
        workload_type=workload_type,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# CLI (standalone use + smoke testing)
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compute per-cell active-page-fraction trajectory + "
                    "F1 / CV summary for one Plan-02 cell.",
    )
    p.add_argument("--cell-id", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--run-start-epoch", type=float, required=True)
    p.add_argument("--jsonl-path", required=True,
                   help="snapshot_timings.jsonl for this cell")
    p.add_argument("--workload-stderr", default=None,
                   help="workload_stderr.log path (omit for producer-only)")
    p.add_argument("--workload-type", default="unknown",
                   choices=["phasic", "steady", "warmup", "unknown"])
    p.add_argument("--output", required=True,
                   help="output metrics.json")
    p.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    p.add_argument("--hop", type=int, default=HOP_DEFAULT)
    p.add_argument("--page-size", type=int, default=PAGE_SIZE_DEFAULT)
    args = p.parse_args(argv)

    stderr_path = (Path(args.workload_stderr) if args.workload_stderr
                   else None)
    metrics = compute_metrics_for_cell(
        cell_id=args.cell_id,
        image_dir=Path(args.image_dir),
        run_start_epoch=args.run_start_epoch,
        jsonl_path=Path(args.jsonl_path),
        workload_stderr_path=stderr_path,
        workload_type=args.workload_type,
        window=args.window, hop=args.hop, page_size=args.page_size,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(metrics), indent=2))
    print(f"wrote {out_path}", file=sys.stderr)
    print(f"  cell_id={metrics.cell_id}", file=sys.stderr)
    print(f"  n_dumps={metrics.n_dumps_examined} "
          f"n_pairs={metrics.n_pairs_examined}", file=sys.stderr)
    print(f"  apf_mean={metrics.apf_mean} apf_std={metrics.apf_std}",
          file=sys.stderr)
    print(f"  f1_phase={metrics.f1_phase} cv_workingset={metrics.cv_workingset}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
