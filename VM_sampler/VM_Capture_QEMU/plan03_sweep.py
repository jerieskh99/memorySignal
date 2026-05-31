#!/usr/bin/env python3
"""plan03_sweep.py -- Plan 03 sweep CLI driver.

Owner: DE (Senior Data Engineer).
Implements proposal sections 04 (top-down process model) and 05.2 of
``docs/plan03_window_hop_proposal.html``.

Walks the v2 manifest, loads each cell's streaming APF trajectory, and
evaluates the (window, hop) combo grid via ``plan03_metric_kernel.score``.
Streams the per-row metrics to a CSV with a deterministic column order
and a per-cell ``traj_sha256`` reproducibility key. Optionally calls
``plan03_aggregate.aggregate`` on the output.

Zero new VM captures; this is an offline analyzer-side sweep.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import plan02_manifest as mf
from plan02_metrics_per_cell import (
    load_streaming_trajectory,
    parse_phase_markers,
    phase_markers_to_snap_indices,
)

import plan03_metric_kernel as kernel


CSV_FIELDS = [
    "cell_id", "workload", "iv_ms", "duration_s", "replicate",
    "n_pairs", "window", "hop", "hop_ratio", "n_windows",
    "apf_mean", "apf_std", "cv_workingset", "f1_phase",
    "cepstral_peak_idx", "ceps_peak_snr_db", "stat_pass_frac",
    "coverage_ratio", "traj_sha256", "status",
]

# Proposal section 08 + G2 footnote: ransom phase mean ~20 s; workingset
# cycle duration ~30 s. Coverage ratio = (W * iv) / workload_rhythm.
RHYTHM_S = {
    "phasic": 20.0,   # ransom mean phase duration
    "steady": 30.0,   # workingset cycle duration
}


def _resolve_workload_type(workload: str) -> str:
    """v3 D-81: classify all Phase-2 workloads (mirrors
    plan02_run._classify_workload). Determines which metric the
    kernel computes per cell: F1 for phasic, CV for steady."""
    w = (workload or "").lower()
    phasic_keys = ("ransom", "scanner_metadata", "phase_boundary", "phasic")
    steady_keys = ("workingset", "mmap_traversal", "pagefault_density",
                   "rmw_intensity", "writemag_sweep", "hashtable_intensive",
                   "compress_streaming", "compress_gzip", "decompress_gzip",
                   "json_parse", "sqlite_oltp", "sqlite_analytical", "steady")
    if any(k in w for k in phasic_keys):
        return "phasic"
    if any(k in w for k in steady_keys):
        return "steady"
    return "unknown"


def _iso_to_epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        # Accept "...Z" and timezone-suffixed ISO 8601.
        s = iso.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def _load_phase_markers(cell_json: Path, stderr_path: Path,
                       jsonl_path: Path) -> list[int] | None:
    """Best-effort marker resolution for phasic cells.

    Returns the snap-index list, or None on any failure (the kernel will
    then skip F1 instead of crashing).
    """
    if not stderr_path.is_file():
        return None
    try:
        markers = parse_phase_markers(stderr_path)
    except Exception:  # noqa: BLE001
        return None
    if not markers:
        return None
    # Load snap timestamps from the cell's snapshot_timings.jsonl when
    # present; fall back to per-pair apf_trajectory ``t_emit_epoch``.
    snap_ts: list[float] = []
    if jsonl_path.is_file():
        try:
            for raw in jsonl_path.read_text(errors="replace").splitlines():
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "t0_before_suspend" in obj:
                    snap_ts.append(float(obj["t0_before_suspend"]))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
    if not snap_ts:
        return None
    try:
        return phase_markers_to_snap_indices(markers, snap_ts)
    except Exception:  # noqa: BLE001
        return None


def _traj_sha256(traj: list[float]) -> str:
    """sha256 over the packed little-endian float64 trajectory. Reused
    across the 12 (W, H) combos of one cell so cells appear identical
    regardless of their write ordering in the manifest."""
    if not traj:
        return ""
    blob = struct.pack(f"<{len(traj)}d", *traj)
    return hashlib.sha256(blob).hexdigest()[:16]


def _coverage_ratio(window: int, iv_ms: int, workload_type: str) -> float:
    rhythm = RHYTHM_S.get(workload_type)
    if rhythm is None or rhythm <= 0:
        return float("nan")
    return (window * iv_ms / 1000.0) / rhythm


def _short_row(row: dict[str, Any]) -> dict[str, Any]:
    """Fill missing numeric fields with empty strings so DictWriter
    emits canonical NA columns instead of literal ``None``."""
    out: dict[str, Any] = {}
    for k in CSV_FIELDS:
        v = row.get(k)
        out[k] = "" if v is None else v
    return out


def sweep(
    cells_dir: Path,
    manifest_path: Path,
    output_csv: Path,
    windows: list[int],
    hop_ratios: list[float],
    *,
    workload_filter: str | None = None,
    status_filter: set[str] | None = None,
    min_pairs: int = 4,
    max_cells: int | None = None,
) -> int:
    """Run the sweep; write CSV; return the number of cells processed."""
    rows = mf.load(manifest_path)
    filt = re.compile(workload_filter) if workload_filter else None
    allowed_status = status_filter or {"ok"}

    selected: list[mf.ManifestRow] = []
    for r in rows:
        if r.is_warmup:
            continue
        if r.status not in allowed_status:
            continue
        if filt is not None and not filt.search(r.workload):
            continue
        selected.append(r)
    if max_cells is not None:
        selected = selected[:max_cells]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_csv.with_suffix(output_csv.suffix + ".tmp")

    csv_rows: list[dict[str, Any]] = []
    total = len(selected)
    for idx, r in enumerate(selected, start=1):
        if idx % 10 == 0 or idx == 1 or idx == total:
            print(f"[{idx}/{total}] {r.cell_id} {r.workload} iv={r.interval_ms}",
                  file=sys.stderr)

        cell_json = cells_dir / f"cell_{r.cell_id}.json"
        cell_workdir = cells_dir / "work" / r.cell_id
        apf_path = cell_workdir / "apf_trajectory.jsonl"
        traj, _sentinel = load_streaming_trajectory(apf_path)

        wt = _resolve_workload_type(r.workload)
        markers: list[int] | None = None
        if wt == "phasic":
            stderr_path = cell_workdir / "workload_stderr.log"
            jsonl_path = cell_workdir / "snapshot_timings.jsonl"
            markers = _load_phase_markers(cell_json, stderr_path, jsonl_path)

        sha = _traj_sha256(traj)

        if not traj or len(traj) < min_pairs:
            row = {
                "cell_id": r.cell_id, "workload": r.workload,
                "iv_ms": r.interval_ms, "duration_s": r.duration_s,
                "replicate": r.replicate,
                "n_pairs": len(traj), "window": "", "hop": "",
                "hop_ratio": "", "n_windows": 0,
                "apf_mean": None, "apf_std": None,
                "cv_workingset": None, "f1_phase": None,
                "cepstral_peak_idx": None, "ceps_peak_snr_db": None,
                "stat_pass_frac": None,
                "coverage_ratio": None,
                "traj_sha256": sha, "status": "skip:short",
            }
            csv_rows.append(_short_row(row))
            continue

        for w in windows:
            for hr in hop_ratios:
                hop = max(1, round(w * hr))
                metrics = kernel.score(
                    traj, w, hop,
                    phase_marker_indices=markers,
                    workload_type=wt,
                )
                cov = _coverage_ratio(w, r.interval_ms, wt)
                row = {
                    "cell_id": r.cell_id, "workload": r.workload,
                    "iv_ms": r.interval_ms, "duration_s": r.duration_s,
                    "replicate": r.replicate,
                    "n_pairs": metrics.get("n_pairs", len(traj)),
                    "window": w, "hop": hop, "hop_ratio": hr,
                    "n_windows": metrics.get("n_windows", 0),
                    "apf_mean": metrics.get("apf_mean"),
                    "apf_std": metrics.get("apf_std"),
                    "cv_workingset": metrics.get("cv_workingset"),
                    "f1_phase": metrics.get("f1_phase"),
                    "cepstral_peak_idx": metrics.get("cepstral_peak_idx"),
                    "ceps_peak_snr_db": metrics.get("ceps_peak_snr_db"),
                    "stat_pass_frac": metrics.get("stat_pass_frac"),
                    "coverage_ratio": (None if cov != cov else cov),  # NaN -> None
                    "traj_sha256": sha,
                    "status": metrics.get("status", "ok"),
                }
                csv_rows.append(_short_row(row))

    # Deterministic ordering for hash-stable output.
    csv_rows.sort(key=lambda d: (
        str(d.get("cell_id", "")),
        int(d["window"]) if str(d.get("window", "")).strip() else -1,
        int(d["hop"]) if str(d.get("hop", "")).strip() else -1,
    ))

    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    tmp.replace(output_csv)
    return len(selected)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="plan03_sweep.py",
        description="Plan 03 offline window/hop sweep over v2 APF trajectories.",
    )
    p.add_argument("--cells-dir", required=True,
                   help="path to v2_full/cells (containing cell_<id>.json and work/)")
    p.add_argument("--manifest", required=True, help="manifest.csv path")
    p.add_argument("--output-csv", required=True, help="sweep.csv output")
    p.add_argument("--output-json", default=None,
                   help="summary.json output (also triggers aggregate)")
    p.add_argument("--recommendation-json", default=None,
                   help="optional recommendation.json output")
    p.add_argument("--windows", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument("--hop-ratios", nargs="+", type=float,
                   default=[0.25, 0.50, 1.00])
    p.add_argument("--workload-filter", default=None,
                   help="regex applied to manifest workload column")
    p.add_argument("--status-filter", default="ok",
                   help="comma-separated manifest status whitelist")
    p.add_argument("--min-pairs", type=int, default=4)
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="reserved for downstream bootstrap; unused today")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing --output-csv without prompting")
    args = p.parse_args(argv)

    cells_dir = Path(args.cells_dir).expanduser().resolve()
    if not cells_dir.is_dir():
        print(f"ERROR: cells-dir not found: {cells_dir}", file=sys.stderr)
        return 2
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    out_csv = Path(args.output_csv).expanduser().resolve()
    if out_csv.exists() and not args.force:
        print(f"ERROR: {out_csv} exists; pass --force to overwrite",
              file=sys.stderr)
        return 2

    status_set = {s.strip() for s in args.status_filter.split(",") if s.strip()}
    n = sweep(
        cells_dir=cells_dir,
        manifest_path=manifest_path,
        output_csv=out_csv,
        windows=list(args.windows),
        hop_ratios=list(args.hop_ratios),
        workload_filter=args.workload_filter,
        status_filter=status_set,
        min_pairs=args.min_pairs,
        max_cells=args.max_cells,
    )
    print(f"wrote {out_csv} (cells processed: {n})", file=sys.stderr)

    if args.output_json:
        import plan03_aggregate as agg
        rec_path = (Path(args.recommendation_json).expanduser().resolve()
                    if args.recommendation_json else None)
        agg.aggregate(out_csv, Path(args.output_json).expanduser().resolve(),
                      recommendation_json=rec_path,
                      cells_dir=cells_dir)
        print(f"wrote {args.output_json}", file=sys.stderr)
        if rec_path:
            print(f"wrote {rec_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
