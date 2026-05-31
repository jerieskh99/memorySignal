#!/usr/bin/env python3
"""plan04_run.py -- Plan 04 segmenter driver CLI.

Owner: E2 (Pipeline Lead). Implements proposal sections 04, 05.3, and 08
of ``docs/plan04_segmenter_proposal.html``.

Walks the v3 manifest, looks up per-workload (W, H) from
``plan03_recommendation.json``, runs CUSUM (E1's ``plan04_cusum``) or
the legacy diff stub, scores per family (phasic F1 vs phase markers,
marker-less plausibility fallback, steady stationarity =
1 - min(1, n_spurious/3)), and writes a per-cell CSV plus a per-workload
summary JSON tagged ``plan04.segmenter_results.v1``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import plan02_manifest as mf
from plan02_metrics_per_cell import (
    detect_boundaries_diff,
    f1_score,
    load_streaming_trajectory,
    parse_phase_markers,
    phase_markers_to_snap_indices,
)
from plan02_run import _classify_workload

try:
    from plan04_cusum import detect_boundaries_full as _cusum_detect
except Exception as _cusum_exc:  # noqa: BLE001
    _cusum_detect = None
    _CUSUM_IMPORT_ERROR = _cusum_exc
else:
    _CUSUM_IMPORT_ERROR = None


RESULTS_SCHEMA = "plan04.segmenter_results.v1"

# Per-workload boundary tolerance (Delta-6 from proposal). Snap-index
# units; phasic-only -- steady falls through to the spurious-count
# branch and the tolerance column is recorded but not consulted.
PER_WORKLOAD_TOLERANCE = {
    "sandbox_ransom_batched": 1,
    "sandbox_ransom_seq": 2,
    "sandbox_ransom_selective": 1,
    "app_hashtable_intensive_v2": 2,
    "mem_pagefault_density_v2": 2,
}
DEFAULT_PHASIC_TOLERANCE = 1
DEFAULT_STEADY_TOLERANCE = 0

# G3 plausibility band (marker-less phasic): 1 <= n_pred <= k_expected+2.
# k_expected is unknown without a per-workload table, so the doc allows
# a 1..20 default band as a permissive "any-detection-at-all" check.
PLAUSIBILITY_MIN = 1
PLAUSIBILITY_MAX = 20

CSV_FIELDS = [
    "cell_id", "workload", "family", "iv_ms", "duration_s", "replicate",
    "n_pairs", "window", "hop", "detector",
    "n_boundaries_pred", "boundary_indices",
    "n_markers_truth", "marker_indices",
    "f1_phase_cusum", "stationarity_score", "tolerance",
    "gate_pass_phasic_f1", "gate_pass_steady_stationarity",
    "status",
]


def _load_recommendation(path: Path) -> dict[str, tuple[int, int]]:
    """Return {workload: (window, hop)} from plan03_recommendation.json."""
    if not path.is_file():
        raise FileNotFoundError(f"recommendation not found: {path}")
    payload = json.loads(path.read_text())
    out: dict[str, tuple[int, int]] = {}
    for entry in payload.get("recommendations", []):
        wl = entry.get("workload")
        w = entry.get("recommended_window")
        h = entry.get("recommended_hop")
        if not wl or w is None or h is None:
            continue
        try:
            out[wl] = (int(w), int(h))
        except (TypeError, ValueError):
            continue
    if not out:
        raise ValueError(f"no (workload, W, H) tuples extractable from {path}")
    return out


def _tolerance_for(workload: str, family: str) -> int:
    if family == "steady":
        return DEFAULT_STEADY_TOLERANCE
    return PER_WORKLOAD_TOLERANCE.get(workload, DEFAULT_PHASIC_TOLERANCE)


def _load_snap_timestamps(jsonl_path: Path) -> list[float]:
    if not jsonl_path.is_file():
        return []
    ts: list[float] = []
    try:
        for raw in jsonl_path.read_text(errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "t0_before_suspend" in obj:
                ts.append(float(obj["t0_before_suspend"]))
    except OSError:
        return []
    return ts


def _resolve_truth_markers(cell_workdir: Path,
                            marker_mode: str = "absolute") -> list[int]:
    """Phase-marker snap indices; empty list if absent or unresolvable.

    marker_mode:
      "absolute" -- compare marker epoch_seconds to snap epoch_seconds
                    (legacy behavior; fails if clocks are skewed).
      "relative" -- D-85 workaround for v3's wall-clock skew (~63 h
                    between workload_stderr.log and
                    snapshot_timings.jsonl, with all markers collapsing
                    to snap_idx=0). Shift markers so the FIRST marker
                    aligns to the FIRST snap, preserving inter-marker
                    spacing. Works when (a) the workload launches at or
                    near the first snap and (b) markers fire in their
                    natural cadence afterward.
    """
    stderr_path = cell_workdir / "workload_stderr.log"
    if not stderr_path.is_file():
        return []
    try:
        markers = parse_phase_markers(stderr_path)
    except Exception:  # noqa: BLE001
        return []
    if not markers:
        return []
    snap_ts = _load_snap_timestamps(cell_workdir / "snapshot_timings.jsonl")
    if not snap_ts:
        return []
    if marker_mode == "relative":
        # Shift so markers[0] -> snap_ts[0]; preserves relative spacing.
        delta = snap_ts[0] - markers[0][0]
        markers = [(t + delta, p) for (t, p) in markers]
    try:
        return phase_markers_to_snap_indices(markers, snap_ts)
    except Exception:  # noqa: BLE001
        return []


def _run_detector(detector: str, traj: list[float], window: int, hop: int,
                  *, k: float, h: float, min_separation: int) -> list[int]:
    if detector == "legacy":
        return detect_boundaries_diff(traj)
    if _cusum_detect is None:
        raise RuntimeError(
            f"detector=cusum requested but plan04_cusum import failed: "
            f"{_CUSUM_IMPORT_ERROR!r}"
        )
    return list(_cusum_detect(
        traj, window, hop, k=k, h=h, min_separation=min_separation,
    ))


def _stationarity_from_spurious(n_spurious: int) -> float:
    """Spec steady score: 1 - min(1, n_spurious/3). Bounded [0, 1].

    Note: E1's plan04_cusum.stationarity_score(traj_1d) is the
    trajectory-side analogue with a different signature; the per-cell
    score in the spec body is detector-output-based and is what we use.
    """
    return 1.0 - min(1.0, n_spurious / 3.0)


def _score_cell(family: str, workload: str, traj: list[float],
                boundaries: list[int],
                truth: list[int]) -> dict[str, Any]:
    """Family-aware scoring per proposal section 05.2 (Delta-6 tolerance)."""
    tol = _tolerance_for(workload, family)
    n_pred = len(boundaries)
    n_truth = len(truth)
    f1_value: float | None = None
    stat_score: float | None = None
    gate_phasic = ""  # "" denotes N/A in CSV cells
    gate_steady = ""

    if family == "steady":
        stat_score = _stationarity_from_spurious(n_pred)
        gate_steady = "1" if n_pred <= 2 else "0"
    elif family == "phasic" and truth:
        # Marker-rich phasic: windowed F1 vs ground-truth markers.
        breakdown = f1_score(boundaries, truth, tolerance=tol)
        f1_value = float(breakdown["f1"])
        gate_phasic = "1" if f1_value >= 0.5 else "0"
    elif family == "phasic":
        # Marker-less phasic: G3 plausibility (binary score in [0, 1]).
        if PLAUSIBILITY_MIN <= n_pred <= PLAUSIBILITY_MAX:
            f1_value = 1.0
            gate_phasic = "1"
        else:
            f1_value = 0.0
            gate_phasic = "0"

    return {
        "tolerance": tol,
        "f1_phase_cusum": f1_value,
        "stationarity_score": stat_score,
        "gate_pass_phasic_f1": gate_phasic,
        "gate_pass_steady_stationarity": gate_steady,
        "n_markers_truth": n_truth,
    }


def _na(v: Any) -> Any:
    return "" if v is None else v


def _build_csv_row(r: mf.ManifestRow, family: str, window: int | None,
                    hop: int | None, detector: str, traj_len: int,
                    boundaries: list[int], truth: list[int],
                    scored: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "cell_id": r.cell_id, "workload": r.workload, "family": family,
        "iv_ms": r.interval_ms, "duration_s": r.duration_s,
        "replicate": r.replicate, "n_pairs": traj_len,
        "window": _na(window), "hop": _na(hop), "detector": detector,
        "n_boundaries_pred": len(boundaries),
        "boundary_indices": ";".join(str(b) for b in boundaries),
        "n_markers_truth": scored.get("n_markers_truth", 0),
        "marker_indices": ";".join(str(t) for t in truth),
        "f1_phase_cusum": _na(scored.get("f1_phase_cusum")),
        "stationarity_score": _na(scored.get("stationarity_score")),
        "tolerance": scored.get("tolerance", ""),
        "gate_pass_phasic_f1": scored.get("gate_pass_phasic_f1", ""),
        "gate_pass_steady_stationarity":
            scored.get("gate_pass_steady_stationarity", ""),
        "status": status,
    }


def sweep(
    cells_dir: Path,
    manifest_path: Path,
    recommendation_path: Path,
    output_csv: Path,
    *,
    detector: str = "cusum",
    cusum_k: float = 2.0,
    cusum_h: float = 4.0,
    min_separation: int = 2,
    marker_tolerance: str = "auto",
    marker_mode: str = "absolute",
    status_filter: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]]]:
    """Per-cell sweep; writes CSV via temp+rename. Returns rows + recs."""
    recommendations = _load_recommendation(recommendation_path)
    rows = mf.load(manifest_path)
    allowed_status = status_filter or {"ok"}
    selected: list[mf.ManifestRow] = [
        r for r in rows
        if (not r.is_warmup) and r.status in allowed_status
    ]

    csv_rows: list[dict[str, Any]] = []
    total = len(selected)
    skipped_no_rec: set[str] = set()
    for idx, r in enumerate(selected, start=1):
        if idx % 10 == 0 or idx == 1 or idx == total:
            print(f"[plan04] [{idx}/{total}] {r.cell_id} {r.workload} "
                  f"iv={r.interval_ms}", file=sys.stderr)

        if r.workload not in recommendations:
            if r.workload not in skipped_no_rec:
                print(f"[plan04] WARN no (W, H) for {r.workload!r} in "
                      f"recommendation; skipping all cells",
                      file=sys.stderr)
                skipped_no_rec.add(r.workload)
            continue
        window, hop = recommendations[r.workload]
        family = _classify_workload(r.workload)

        cell_workdir = cells_dir / "work" / r.cell_id
        apf_path = cell_workdir / "apf_trajectory.jsonl"
        traj, _sentinel = load_streaming_trajectory(apf_path)
        if not traj:
            csv_rows.append(_build_csv_row(
                r, family, window, hop, detector, 0, [], [],
                {"tolerance": _tolerance_for(r.workload, family),
                 "n_markers_truth": 0},
                status="skip:no_traj",
            ))
            continue

        truth: list[int] = []
        if family == "phasic":
            truth = _resolve_truth_markers(cell_workdir, marker_mode=marker_mode)

        try:
            boundaries = _run_detector(
                detector, traj, window, hop,
                k=cusum_k, h=cusum_h, min_separation=min_separation,
            )
        except Exception as exc:  # noqa: BLE001
            csv_rows.append(_build_csv_row(
                r, family, window, hop, detector, len(traj), [], truth,
                {"tolerance": _tolerance_for(r.workload, family),
                 "n_markers_truth": len(truth)},
                status=f"error:detector:{type(exc).__name__}",
            ))
            continue

        scored = _score_cell(family, r.workload, traj, boundaries, truth)
        status = "ok:plausibility" if (family == "phasic" and not truth) else "ok"
        csv_rows.append(_build_csv_row(
            r, family, window, hop, detector, len(traj),
            boundaries, truth, scored, status=status,
        ))

    csv_rows.sort(key=lambda d: (str(d.get("workload", "")),
                                  str(d.get("cell_id", ""))))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    tmp.replace(output_csv)
    return csv_rows, recommendations


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _aggregate(csv_rows: list[dict[str, Any]],
                recommendations: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """Per-workload medians + G1-G4 gate verdicts."""
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for row in csv_rows:
        by_workload.setdefault(row["workload"], []).append(row)

    per_workload: list[dict[str, Any]] = []
    g1_pass = g1_total = 0
    g2_pass = g2_total = 0
    g3_pass = g3_total = 0
    for workload, group in sorted(by_workload.items()):
        family = group[0]["family"]
        usable = [g for g in group if g["status"] in ("ok", "ok:plausibility")]
        # Treat None and "" alike (in-memory dicts have None; CSV path "")
        def _has(v: Any) -> bool:
            return v is not None and v != ""
        marker_rich = [g for g in usable
                        if family == "phasic"
                        and _has(g["f1_phase_cusum"])
                        and g["n_markers_truth"]]
        marker_less = [g for g in usable
                        if family == "phasic"
                        and not g["n_markers_truth"]]
        steady = [g for g in usable if family == "steady"]

        f1_values = [float(g["f1_phase_cusum"]) for g in marker_rich
                      if _has(g["f1_phase_cusum"])]
        stat_values = [float(g["stationarity_score"]) for g in steady
                        if _has(g["stationarity_score"])]
        median_f1 = _median(f1_values)
        median_stat = _median(stat_values)
        # P(F1 >= 0.50) for the phasic spec conjunct
        p_f1_above = (sum(1 for v in f1_values if v >= 0.50) / len(f1_values)
                      if f1_values else None)
        # n_spurious distribution for the steady spec conjunct
        n_spur = [int(g["n_boundaries_pred"]) for g in steady
                  if _has(g["n_boundaries_pred"])]
        median_n_spur = _median(n_spur) if n_spur else None
        p_n_spur_zero = (sum(1 for v in n_spur if v == 0) / len(n_spur)
                         if n_spur else None)
        if marker_less:
            plausible_frac = (sum(1 for g in marker_less
                                   if g["gate_pass_phasic_f1"] in ("1", 1, True))
                              / len(marker_less))
        else:
            plausible_frac = None

        # D-84: gate thresholds aligned to proposal §08 spec
        g1_verdict: bool | None = None
        g2_verdict: bool | None = None
        g3_verdict: bool | None = None
        if marker_rich:
            g1_total += 1
            # spec: median F1 >= 0.67 AND P(F1 >= 0.50) >= 0.75
            g1_verdict = bool(median_f1 is not None
                              and p_f1_above is not None
                              and median_f1 >= 0.67
                              and p_f1_above >= 0.75)
            if g1_verdict:
                g1_pass += 1
        if steady:
            g2_total += 1
            # spec: median n_spurious <= 1 AND P(n_spurious == 0) >= 0.5
            g2_verdict = bool(median_n_spur is not None
                              and p_n_spur_zero is not None
                              and median_n_spur <= 1.0
                              and p_n_spur_zero >= 0.5)
            if g2_verdict:
                g2_pass += 1
        if marker_less:
            g3_total += 1
            g3_verdict = (plausible_frac is not None
                           and plausible_frac >= 0.8)
            if g3_verdict:
                g3_pass += 1
        g4_verdict: bool = True  # legacy regression placeholder; E3 wires

        passes_acceptance = True
        for v in (g1_verdict, g2_verdict, g3_verdict):
            if v is False:
                passes_acceptance = False

        # D-84: validator C8 reads pw_entry["gates"][gate_pass_*]; nest the
        # gate booleans under a "gates" sub-dict and rename to the
        # gate_pass_<family> convention expected by plan02_validate_session.
        gates_sub = {
            "gate_pass_phasic_f1": g1_verdict,
            "gate_pass_steady_stationarity": g2_verdict,
            "gate_pass_marker_less_plausibility": g3_verdict,
            "gate_pass_legacy_regression": g4_verdict,
        }
        w_hop = recommendations.get(workload)
        per_workload.append({
            "workload": workload,
            "family": family,
            "window": w_hop[0] if w_hop else None,
            "hop": w_hop[1] if w_hop else None,
            "n_cells": len(group),
            "n_cells_scored": len(usable),
            "n_marker_rich": len(marker_rich),
            "n_marker_less": len(marker_less),
            "n_steady": len(steady),
            "median_f1_phase_cusum": median_f1,
            "p_f1_above_0_50": p_f1_above,
            "median_stationarity": median_stat,
            "median_n_spurious": median_n_spur,
            "p_n_spurious_eq_zero": p_n_spur_zero,
            "plausibility_pass_fraction": plausible_frac,
            "gates": gates_sub,
            "passes_acceptance": passes_acceptance,
        })

    def _gate(num: int, denom: int, **extra: Any) -> dict[str, Any]:
        return {"applicable_workloads": denom, "passing_workloads": num,
                "pass_rate": (num / denom) if denom else None, **extra}

    n = len(per_workload)
    gates = {
        "G1_phasic_f1": _gate(g1_pass, g1_total),
        "G2_steady_stationarity": _gate(g2_pass, g2_total),
        "G3_marker_less_plausibility": _gate(g3_pass, g3_total),
        "G4_legacy_regression": _gate(n, n, note="placeholder; E3 to wire"),
    }
    return {"per_workload": per_workload, "gates": gates}


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="plan04_run.py",
        description=("Plan 04 segmenter driver: per-cell CUSUM (or legacy "
                     "diff) over v3 APF trajectories, family-aware scoring "
                     "+ per-workload G1-G4 gates."),
    )
    p.add_argument("--cells-dir", required=True,
                   help="path to v3_full/cells or v3_combined/cells")
    p.add_argument("--manifest", required=True, help="manifest.csv path")
    p.add_argument("--recommendation", required=True,
                   help="plan03_recommendation.json (per-workload W, H)")
    p.add_argument("--output-csv", required=True, help="per-cell results CSV")
    p.add_argument("--output-json", required=True,
                   help="per-workload summary + gates JSON")
    p.add_argument("--detector", default="cusum",
                   choices=["cusum", "legacy"])
    p.add_argument("--k", type=float, default=2.0,
                   help="CUSUM control limit (E1 detector)")
    p.add_argument("--h", type=float, default=4.0,
                   help="CUSUM decision threshold (E1 detector)")
    p.add_argument("--min-separation", type=int, default=2)
    p.add_argument("--marker-mode", choices=("absolute", "relative"),
                    default="absolute",
                    help="D-85: 'relative' shifts markers so the first "
                         "marker aligns with the first snap, compensating "
                         "for wall-clock skew between workload_stderr.log "
                         "and snapshot_timings.jsonl. Use when markers "
                         "appear to collapse to snap_idx=0.")
    p.add_argument("--marker-tolerance", default="auto",
                   help="int snap count or 'auto' (per-workload Delta-6 table)")
    p.add_argument("--status-filter", default="ok",
                   help="comma-separated manifest status whitelist")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cells_dir = Path(args.cells_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    rec_path = Path(args.recommendation).expanduser().resolve()
    out_csv = Path(args.output_csv).expanduser().resolve()
    out_json = Path(args.output_json).expanduser().resolve()
    for label, path, kind in (
        ("cells-dir", cells_dir, "dir"),
        ("manifest", manifest_path, "file"),
        ("recommendation", rec_path, "file"),
    ):
        ok = path.is_dir() if kind == "dir" else path.is_file()
        if not ok:
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2
    for p_existing in (out_csv, out_json):
        if p_existing.exists() and not args.force:
            print(f"ERROR: {p_existing} exists; pass --force to overwrite",
                  file=sys.stderr)
            return 2
    if args.marker_tolerance != "auto":
        try:
            int(args.marker_tolerance)
        except ValueError:
            print("ERROR: --marker-tolerance must be int or 'auto'",
                  file=sys.stderr)
            return 2
    if args.detector == "cusum" and _cusum_detect is None:
        print(f"ERROR: --detector cusum unavailable: {_CUSUM_IMPORT_ERROR!r}",
              file=sys.stderr)
        return 2

    status_set = {s.strip() for s in args.status_filter.split(",") if s.strip()}
    try:
        csv_rows, recs = sweep(
            cells_dir=cells_dir,
            manifest_path=manifest_path,
            recommendation_path=rec_path,
            output_csv=out_csv,
            detector=args.detector,
            cusum_k=args.k,
            cusum_h=args.h,
            min_separation=args.min_separation,
            marker_tolerance=args.marker_tolerance,
            marker_mode=args.marker_mode,
            status_filter=status_set,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"[plan04] wrote {out_csv} ({len(csv_rows)} rows)", file=sys.stderr)

    aggregate = _aggregate(csv_rows, recs)
    payload = {
        "schema": RESULTS_SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "cells_dir": str(cells_dir),
            "manifest": str(manifest_path),
            "recommendation": str(rec_path),
        },
        "detector": {
            "name": args.detector,
            "k": args.k,
            "h": args.h,
            "min_separation": args.min_separation,
            "marker_tolerance": args.marker_tolerance,
            "marker_mode": args.marker_mode,
        },
        "per_workload": aggregate["per_workload"],
        "gates": aggregate["gates"],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)
    tmp.replace(out_json)
    print(f"[plan04] wrote {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
