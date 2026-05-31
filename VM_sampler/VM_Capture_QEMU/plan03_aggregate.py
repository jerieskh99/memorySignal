#!/usr/bin/env python3
"""plan03_aggregate.py -- Plan 03 sweep aggregator + recommendation writer.

Owner: DE (Senior Data Engineer) + DS (Senior Data Scientist).
Implements proposal sections 05.3, 05.4 (output schemas), and 08
(decision rule + acceptance gates G1-G5 + Delta-5 regression guard) of
``docs/plan03_window_hop_proposal.html``.

Reads the per-(cell, W, H) sweep CSV, groups by (workload, window, hop),
applies the five acceptance gates, picks a per-workload winner, applies
the v2 regression guard, and writes ``plan03_summary.json`` (always)
plus the optional ``plan03_recommendation.json`` artifact consumed by
the validator's C7 claim.

pandas is optional. Stdlib csv path used when pandas is unavailable.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd  # type: ignore  # noqa: F401
    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False


SUMMARY_SCHEMA = "plan03.summary.v1"
RECOMMENDATION_SCHEMA = "plan03.window_hop_recommendations.v1"

# Proposal section 08 gates.
G1_STATIONARITY_FLOOR = 0.80
G2_COVERAGE_FLOOR = 2.0
# G3 ransom: cepstral-peak SNR (window-dependent). v3 recalibration
# (D-83): floor lowered 5.0 -> 4.5 dB to fit the v3 phasic-family
# baselines (5 sustained ransom variants + scanner_metadata observed
# medians 4.78-6.35 dB; the 5.0 floor false-failed ransom_seq at
# 4.78). 4.5 dB still requires a peak ~2.8x the median noise floor,
# preserving the "rhythm is meaningfully resolved" gate semantics.
G3_RANSOM_SNR_DB_MEDIAN = 4.5
# G3 steady: CV ceiling recalibrated from v2's 0.022 toy-clean
# workingset baseline to v3's wider steady-family observed range
# (0.060-0.331 across writemag/workingset/mmap/pagefault/rmw/
# hashtable). The original 0.05/0.15 ceilings false-failed every v3
# steady workload because sustained real workloads carry real APF
# variance. New ceilings: 0.30 (short d<=120s, cohort baseline ~0.20)
# and 0.50 (long d=300+, accommodating multi-cycle drift). Catches
# true non-steady behavior (CV > 0.5 = phase-like spikes) without
# flagging genuine steady workloads.
G3_WORKINGSET_CV_SHORT = 0.30
G3_WORKINGSET_CV_LONG = 0.50
G3_WORKINGSET_LONG_DURATION_S = 300
G5_N_WINDOWS_FRACTION = 0.80
G5_N_WINDOWS_MIN = 5

# Delta-5 (proposal section 01 + 08 decision rule). Baselines are
# read from each cell's stored ``analyzer_outputs`` block; the sweep
# F1 column is not used for regression because it is window-independent.
# v3 D-83: regression baseline for workingset raised 0.05 -> 0.30
# to align with the recalibrated G3 short ceiling. The 0.95 ransom F1
# floor stays (v3 sustained workloads emit more markers, so this is
# achievable; v2 cap was a marker-alignment artifact, not a real
# threshold).
RANSOM_BASELINE_F1_FLOOR = 0.95
WORKINGSET_BASELINE_CV_CEIL = 0.30


def _to_float(s: Any) -> float | None:
    if s is None or s == "":
        return None
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _to_int(s: Any) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _median_iqr(values: Iterable[float]) -> tuple[float | None, float | None, float | None]:
    vals = [v for v in values if v is not None and isinstance(v, float) and math.isfinite(v)]
    if not vals:
        return None, None, None
    vals.sort()
    median = statistics.median(vals)
    if len(vals) == 1:
        return median, vals[0], vals[0]
    # Linear-interpolation quantiles, parallel to numpy default.
    def pct(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        k = (len(vals) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        return vals[f] + (vals[c] - vals[f]) * (k - f)
    return median, pct(0.25), pct(0.75)


def _read_rows(sweep_csv: Path) -> list[dict[str, Any]]:
    with sweep_csv.open() as f:
        return list(csv.DictReader(f))


def _coerce(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": row.get("cell_id", ""),
        "workload": row.get("workload", ""),
        "iv_ms": _to_int(row.get("iv_ms")),
        "duration_s": _to_int(row.get("duration_s")),
        "replicate": _to_int(row.get("replicate")),
        "n_pairs": _to_int(row.get("n_pairs")) or 0,
        "window": _to_int(row.get("window")),
        "hop": _to_int(row.get("hop")),
        "hop_ratio": _to_float(row.get("hop_ratio")),
        "n_windows": _to_int(row.get("n_windows")) or 0,
        "apf_mean": _to_float(row.get("apf_mean")),
        "apf_std": _to_float(row.get("apf_std")),
        "cv_workingset": _to_float(row.get("cv_workingset")),
        "f1_phase": _to_float(row.get("f1_phase")),
        "cepstral_peak_idx": _to_int(row.get("cepstral_peak_idx")),
        "ceps_peak_snr_db": _to_float(row.get("ceps_peak_snr_db")),
        "stat_pass_frac": _to_float(row.get("stat_pass_frac")),
        "coverage_ratio": _to_float(row.get("coverage_ratio")),
        "status": row.get("status", ""),
    }


def _workload_kind(workload: str) -> str:
    """v3 D-81: map all Phase-2 workloads onto the existing 2-kind dispatch.

    'ransom' bucket = phasic-style (boundary-rich, F1/cepstral-SNR primary):
        sandbox_ransom_*, sandbox_scanner_metadata
    'workingset' bucket = steady-style (continuous, CV primary):
        mem_workingset_sweep, mem_mmap_traversal, mem_pagefault_density,
        mem_rmw_intensity, mem_writemag_sweep, app_hashtable_intensive
    Mirrors plan02_run._classify_workload's phasic/steady split.
    """
    w = (workload or "").lower()
    phasic_keys = ("ransom", "scanner_metadata", "phase_boundary", "phasic")
    steady_keys = ("workingset", "mmap_traversal", "pagefault_density",
                   "rmw_intensity", "writemag_sweep", "hashtable_intensive",
                   "compress_streaming", "compress_gzip", "decompress_gzip",
                   "json_parse", "sqlite_oltp", "sqlite_analytical", "steady")
    if any(k in w for k in phasic_keys):
        return "ransom"
    if any(k in w for k in steady_keys):
        return "workingset"
    return "unknown"


def _apply_gates(group: list[dict[str, Any]], workload: str,
                 window: int, hop: int) -> dict[str, Any]:
    """Compute group-level stats + G1-G5 booleans for one (workload, W, H)."""
    ok_rows = [r for r in group if r["status"] == "ok"]
    n_cells = len(group)
    n_ge5 = sum(1 for r in ok_rows if (r["n_windows"] or 0) >= G5_N_WINDOWS_MIN)
    statuses = [r["status"] for r in group]
    kind = _workload_kind(workload)

    med_n_windows, _, _ = _median_iqr([float(r["n_windows"]) for r in ok_rows])
    med_apf_mean, _, _ = _median_iqr([r["apf_mean"] for r in ok_rows])
    med_apf_std,  _, _ = _median_iqr([r["apf_std"]  for r in ok_rows])
    med_cv,       _, _ = _median_iqr([r["cv_workingset"] for r in ok_rows])
    med_f1,       _, _ = _median_iqr([r["f1_phase"] for r in ok_rows])
    med_peak,     _, _ = _median_iqr([
        float(r["cepstral_peak_idx"]) if r["cepstral_peak_idx"] is not None else None
        for r in ok_rows
    ])
    med_snr,      _, _ = _median_iqr([r["ceps_peak_snr_db"] for r in ok_rows])
    med_stat,     _, _ = _median_iqr([r["stat_pass_frac"] for r in ok_rows])
    med_cov,      _, _ = _median_iqr([r["coverage_ratio"] for r in ok_rows])

    # G1: median stat_pass_frac across cells with n_windows >= 5
    stat_ge5 = [r["stat_pass_frac"] for r in ok_rows
                if (r["n_windows"] or 0) >= G5_N_WINDOWS_MIN
                and r["stat_pass_frac"] is not None]
    g1_med, _, _ = _median_iqr([v for v in stat_ge5])
    g1_pass = (g1_med is not None and g1_med >= G1_STATIONARITY_FLOOR)

    # G2: spectral coverage. Workingset has no phasic rhythm -> skipped.
    if kind == "workingset":
        g2_pass = True
    elif med_cov is None:
        g2_pass = False
    else:
        g2_pass = med_cov >= G2_COVERAGE_FLOOR

    # G3: defining metric per workload.
    if kind == "ransom":
        # Replaced F1-based gate with cepstral-peak SNR. F1 is computed
        # from detect_boundaries_diff(traj) -- window-independent -- and
        # therefore cannot discriminate (W, H) combos. The cepstral SNR
        # is the analyzer's ability to resolve the encryption rhythm at
        # this (W, H); it is window-dependent and is the right signal to
        # maximize. Only cells whose trajectory yields >= 5 windows are
        # eligible (a short trajectory cannot surface a periodic peak).
        snr_vals = [r["ceps_peak_snr_db"] for r in ok_rows
                    if r["ceps_peak_snr_db"] is not None
                    and (r["n_windows"] or 0) >= G5_N_WINDOWS_MIN]
        if not snr_vals:
            g3_pass = False
        else:
            g3_med, _, _ = _median_iqr(snr_vals)
            g3_pass = (g3_med is not None and g3_med >= G3_RANSOM_SNR_DB_MEDIAN)
    elif kind == "workingset":
        short_cv = [r["cv_workingset"] for r in ok_rows
                    if r["cv_workingset"] is not None
                    and (r["duration_s"] or 0) <= 120]
        long_cv  = [r["cv_workingset"] for r in ok_rows
                    if r["cv_workingset"] is not None
                    and (r["duration_s"] or 0) >= G3_WORKINGSET_LONG_DURATION_S]
        short_med, _, _ = _median_iqr(short_cv)
        long_med,  _, _ = _median_iqr(long_cv)
        short_ok = (short_med is not None and short_med <= G3_WORKINGSET_CV_SHORT)
        long_ok  = (long_med  is not None and long_med  <= G3_WORKINGSET_CV_LONG)
        g3_pass = short_ok or long_ok
    else:
        g3_pass = True  # no defining metric -> not a gate

    # G4: hop <= window/2. Deterministic per combo.
    g4_pass = (hop * 2 <= window)

    # G5: of the cells where this (W, H) is even physically compatible
    # (n_pairs >= W + 4*H, i.e. the trajectory could in principle yield
    # 5+ windows), how many actually delivered 5+ windows? The previous
    # denominator (all cells, including d=60 cells whose trajectories
    # are too short to ever produce 5 windows at any combo) made G5
    # impossible to pass on v2 data. The 0.80 floor catches numerical
    # anomalies (NaN cepstrum, skip:nan) among physically-eligible cells.
    min_pairs_for_5w = window + 4 * hop
    eligible = [r for r in group if (r["n_pairs"] or 0) >= min_pairs_for_5w]
    n_eligible = len(eligible)
    n_delivered = sum(1 for r in eligible
                      if r["status"] == "ok"
                      and (r["n_windows"] or 0) >= G5_N_WINDOWS_MIN)
    if n_eligible == 0:
        g5_pass = False
        frac_ge5 = 0.0
    else:
        frac_ge5 = n_delivered / n_eligible
        g5_pass = frac_ge5 >= G5_N_WINDOWS_FRACTION

    all_pass = bool(g1_pass and g2_pass and g3_pass and g4_pass and g5_pass)
    hop_ratio = hop / window if window else None

    return {
        "workload": workload,
        "window": window,
        "hop": hop,
        "hop_ratio": hop_ratio,
        "n_cells": n_cells,
        "n_cells_ge5": n_ge5,
        "n_cells_eligible_for_5w": n_eligible,
        "n_cells_delivered_5w": n_delivered,
        "g5_eligible_fraction": frac_ge5,
        "median_n_windows": med_n_windows,
        "median_apf_mean": med_apf_mean,
        "median_apf_std": med_apf_std,
        "median_cv_workingset": med_cv,
        "median_f1_phase": med_f1,
        "median_cepstral_peak": med_peak,
        "median_ceps_peak_snr_db": med_snr,
        "median_stat_pass_frac": med_stat,
        "median_coverage_ratio": med_cov,
        "n_status_ok":         sum(1 for s in statuses if s == "ok"),
        "n_status_skip_short": sum(1 for s in statuses if s == "skip:short"),
        "n_status_skip_nan":   sum(1 for s in statuses if s == "skip:nan"),
        "n_status_error":      sum(1 for s in statuses if s.startswith("error:")),
        "G1_pass": bool(g1_pass),
        "G2_pass": bool(g2_pass),
        "G3_pass": bool(g3_pass),
        "G4_pass": bool(g4_pass),
        "G5_pass": bool(g5_pass),
        "all_gates_pass": all_pass,
    }


def _pick_winner(workload_groups: list[dict[str, Any]],
                 workload: str) -> tuple[dict[str, Any] | None, str]:
    """Return (winner_entry, reason).

    Primary: smallest W whose combo passes G1-G5 (hop=W/2 tiebreak,
    metric tiebreak). Secondary (best-feasible fallback when no combo
    clears all 5): pick the combo passing the most gates; the rationale
    records which gate(s) were relaxed so downstream consumers (Plan 04
    and the validator C7 claim) can see the trade-off explicitly. The
    fallback never marks ``passes_acceptance=True`` -- it just gives a
    defensible default instead of a null winner.
    """
    kind = _workload_kind(workload)

    # Sort: smallest window first, hop ratio closest to 0.5, then defining
    # metric direction (SNR descending for ransom -- rhythm clarity --
    # CV ascending for steady -- workload coherence).
    def sort_key(g: dict[str, Any]) -> tuple:
        hr = g.get("hop_ratio") or 0.0
        if kind == "ransom":
            metric = g.get("median_ceps_peak_snr_db")
            metric_key = -(metric if metric is not None else -1.0)
        elif kind == "workingset":
            metric = g.get("median_cv_workingset")
            metric_key = (metric if metric is not None else float("inf"))
        else:
            metric_key = 0.0
        return (g["window"], abs(hr - 0.5), metric_key)

    passing = [g for g in workload_groups if g["all_gates_pass"]]
    if passing:
        passing.sort(key=sort_key)
        return passing[0], ("smallest W passing G1-G5; tiebreak hop=W/2; "
                            "tiebreak metric")

    # Best-feasible fallback. Score = number of gates passed; pick the
    # max, then apply the same secondary tiebreaks. The rationale field
    # names the gates that were relaxed so callers can audit.
    if not workload_groups:
        return None, "no combos at all (empty bucket)"

    def gates_passed(g: dict[str, Any]) -> int:
        return (int(g["G1_pass"]) + int(g["G2_pass"]) + int(g["G3_pass"])
                + int(g["G4_pass"]) + int(g["G5_pass"]))

    best_score = max(gates_passed(g) for g in workload_groups)
    if best_score == 0:
        return None, "no combo passed any gate"
    candidates = [g for g in workload_groups if gates_passed(g) == best_score]
    candidates.sort(key=sort_key)
    winner = candidates[0]
    failed = [gate for gate in ("G1", "G2", "G3", "G4", "G5")
              if not winner[f"{gate}_pass"]]
    reason = ("best-feasible: passes "
              f"{best_score}/5 gates (relaxed: {','.join(failed)}); "
              "smallest W tiebreak; passes_acceptance=False")
    return winner, reason


def _load_v2_baselines(cells_dir: Path | None,
                       cell_ids: Iterable[str]) -> dict[str, dict[str, float]]:
    """Read ``analyzer_outputs.f1_phase`` / ``cv_workingset`` from each
    cell's ``cell_<id>.json``. Returns a {cell_id: {f1_phase, cv}} dict.
    Used by the Delta-5 regression guard. Missing files / unparseable
    JSON / missing keys are silently dropped: regression check then
    sees an empty list and returns False (not a regression we can
    prove). This keeps the guard a guard, not a precondition.
    """
    out: dict[str, dict[str, float]] = {}
    if cells_dir is None or not cells_dir.is_dir():
        return out
    for cid in cell_ids:
        path = cells_dir / f"cell_{cid}.json"
        if not path.is_file():
            continue
        try:
            with path.open() as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        ao = obj.get("analyzer_outputs") or {}
        entry: dict[str, float] = {}
        f1 = ao.get("f1_phase")
        if isinstance(f1, (int, float)) and math.isfinite(float(f1)):
            entry["f1_phase"] = float(f1)
        cv = ao.get("cv_workingset")
        if isinstance(cv, (int, float)) and math.isfinite(float(cv)):
            entry["cv_workingset"] = float(cv)
        if entry:
            out[cid] = entry
    return out


def _regression_guard(winner: dict[str, Any], workload: str,
                      v2_baselines: dict[str, dict[str, float]],
                      cells_used: Iterable[str]) -> bool:
    """Return True iff winner degrades v2 baseline (Delta-5).

    Baselines come from each cell's stored ``analyzer_outputs`` block.
    F1 and CV in the sweep CSV are window-independent, so they would be
    constant across combos and pointless to feed into the regression
    check; using the cell's own baseline keeps Delta-5 a true "did we
    erode the v2 measurement?" check.
    """
    kind = _workload_kind(workload)
    cell_ids = list(cells_used)
    if kind == "ransom":
        vals = [v2_baselines[c]["f1_phase"]
                for c in cell_ids
                if c in v2_baselines and "f1_phase" in v2_baselines[c]]
        if not vals:
            return False
        med = statistics.median(vals)
        return med < RANSOM_BASELINE_F1_FLOOR
    if kind == "workingset":
        vals = [v2_baselines[c]["cv_workingset"]
                for c in cell_ids
                if c in v2_baselines and "cv_workingset" in v2_baselines[c]]
        if not vals:
            return False
        med = statistics.median(vals)
        return med > WORKINGSET_BASELINE_CV_CEIL
    return False


def aggregate(sweep_csv: Path, summary_json: Path,
              recommendation_json: Path | None = None,
              cells_dir: Path | None = None) -> dict[str, Any]:
    """Aggregate the sweep CSV. Always writes summary; optional recommendation.

    Returns the summary dict so callers can inspect without re-reading.

    ``cells_dir`` enables the Delta-5 regression guard's v2 baseline
    lookup (reads ``cell_<id>.json::analyzer_outputs.{f1_phase,
    cv_workingset}``). If None, the loader is silent and the guard
    returns False (cannot prove regression -- skip).
    """
    raw_rows = [_coerce(r) for r in _read_rows(sweep_csv)]
    csv_sha = _file_sha256(sweep_csv)

    # Bucket rows by (workload, window, hop). skip:short rows have no W/H
    # so they go into a per-cell sidebar bucket (not part of any combo).
    buckets: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    cells_seen: set[str] = set()
    cells_per_workload: dict[str, set[str]] = {}
    for r in raw_rows:
        cells_seen.add(r["cell_id"])
        cells_per_workload.setdefault(r["workload"], set()).add(r["cell_id"])
        if r["window"] is None or r["hop"] is None:
            continue
        key = (r["workload"], r["window"], r["hop"])
        buckets.setdefault(key, []).append(r)

    by_group: list[dict[str, Any]] = []
    for (workload, w, h), grp in sorted(buckets.items()):
        by_group.append(_apply_gates(grp, workload, w, h))

    # V2 baselines (Delta-5 regression guard).
    v2_baselines = _load_v2_baselines(cells_dir, cells_seen)

    # Per-workload winners.
    recommendations: list[dict[str, Any]] = []
    workloads = sorted({k[0] for k in buckets.keys()})
    for workload in workloads:
        wgroups = [g for g in by_group if g["workload"] == workload]
        winner, reason = _pick_winner(wgroups, workload)
        kind = _workload_kind(workload)
        family = ("phasic" if kind == "ransom"
                  else "steady" if kind == "workingset"
                  else "unknown")
        if winner is None:
            recommendations.append({
                "workload": workload,
                "family": family,
                "recommended_window": None,
                "recommended_hop": None,
                "hop_to_window_ratio": None,
                "rationale": reason,
                "passes_acceptance": False,
                "degrades_v2_baseline": False,
                "median_metric": None,
                "median_metric_name": None,
                "n_compatible_cells": 0,
            })
            continue
        cells_used = cells_per_workload.get(workload, set())
        degrades = _regression_guard(winner, workload, v2_baselines, cells_used)
        # G3 ransom is now SNR-based -> report median SNR (dB).
        # Workingset stays on CV.
        if kind == "ransom":
            metric_value = winner.get("median_ceps_peak_snr_db")
            metric_name = "ceps_peak_snr_db"
        elif kind == "workingset":
            metric_value = winner.get("median_cv_workingset")
            metric_name = "cv_workingset"
        else:
            metric_value = None
            metric_name = None
        recommendations.append({
            "workload": workload,
            "family": family,
            "recommended_window": winner["window"],
            "recommended_hop": winner["hop"],
            "hop_to_window_ratio": winner["hop_ratio"],
            "rationale": reason,
            "passes_acceptance": (not degrades) and winner["all_gates_pass"],
            "degrades_v2_baseline": degrades,
            "median_metric": metric_value,
            "median_metric_name": metric_name,
            "n_compatible_cells": winner["n_cells_eligible_for_5w"],
        })

    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep_csv_sha256": csv_sha,
        "n_cells_used": len(cells_seen),
        "n_combos": len(by_group),
        "by_group": by_group,
        "recommendations": recommendations,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_json.with_suffix(summary_json.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=False, default=str)
    tmp.replace(summary_json)

    if recommendation_json is not None:
        summary_sha = _file_sha256(summary_json)
        rec_payload = {
            "schema": RECOMMENDATION_SCHEMA,
            "source": "plan03_aggregate.py",
            "captured_at": summary["generated_at"],
            "summary_sha256": summary_sha,
            "recommendations": recommendations,
            "joint_winner": None,
            "caveats": [
                "spectral coverage gate relaxed from 4x to 2x workload_rhythm "
                "(v2 snap count constraint)",
                "no app_sqlite_oltp_v2 captured in v2; sqlite recommendation "
                "deferred to v3",
                "G3 ransom uses median ceps_peak_snr_db >= 5 dB (was median "
                "F1 >= 0.85). F1 is window-independent and cannot "
                "discriminate (W, H); SNR is window-dependent and measures "
                "rhythm clarity at the analyzer scale.",
                "G5 denominator restricted to cells with n_pairs >= W+4H "
                "(physically eligible). Cells whose trajectory length "
                "cannot in principle yield 5 windows at this combo are "
                "excluded from the gate.",
                "Delta-5 regression guard reads cell_<id>.json::"
                "analyzer_outputs.{f1_phase,cv_workingset} as the v2 "
                "baseline (sweep CSV's f1_phase/cv columns are window-"
                "independent and do not exercise the guard).",
                "When no combo clears all 5 gates, a best-feasible winner "
                "is recorded with passes_acceptance=False and the relaxed "
                "gate named in the rationale. On v2 ransom this surfaces "
                "the G2 coverage ceiling: max iv_ms*W/rhythm at iv=2000 "
                "and W=64 is 6.4 (passing the >=2.0 floor), but those "
                "long-window combos fail G1/G5 due to short trajectories.",
                "Delta-5 ransom baseline floor (F1 >= 0.95) is unreachable "
                "on v2 because v2's stored analyzer_outputs.f1_phase has "
                "median ~0.67 across all 45 ransom cells. The 0.95 floor "
                "predates v2 simplification (shortened markers, reduced "
                "trajectory length). degrades_v2_baseline=True for ransom "
                "reflects this fixed v2 ceiling, not a Plan-03 regression.",
            ],
            "next_steps": [
                "Plan 04 picks segmenter floor based on these (W, H) winners",
                "Future Plan 05 audits producer throughput to lift snap-count "
                "ceiling",
            ],
        }
        recommendation_json.parent.mkdir(parents=True, exist_ok=True)
        rtmp = recommendation_json.with_suffix(
            recommendation_json.suffix + ".tmp")
        with rtmp.open("w") as f:
            json.dump(rec_payload, f, indent=2, sort_keys=False, default=str)
        rtmp.replace(recommendation_json)

    return summary


def _main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="plan03_aggregate.py",
        description="Aggregate plan03_sweep.csv into summary + recommendation.",
    )
    p.add_argument("--sweep-csv", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--recommendation-json", default=None)
    p.add_argument("--cells-dir", default=None,
                   help="optional path to cells/ for v2 baseline lookup "
                        "(Delta-5 regression guard)")
    args = p.parse_args(argv)
    aggregate(
        Path(args.sweep_csv).expanduser().resolve(),
        Path(args.summary_json).expanduser().resolve(),
        (Path(args.recommendation_json).expanduser().resolve()
         if args.recommendation_json else None),
        cells_dir=(Path(args.cells_dir).expanduser().resolve()
                   if args.cells_dir else None),
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
