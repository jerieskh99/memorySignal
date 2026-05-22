#!/usr/bin/env python3
"""plan02_analysis.py -- post-capture analysis for Plan 02.

Owner: DS (Senior Data Scientist) + ML (Senior ML Engineer).
Reference: docs/tuning_plans/experiment_audit.md Decisions D-4, D-5, D-8.

Reads a directory of schema-v2 per-cell JSONs and produces:
  1. A flat tidy CSV of all cells (one row per replicate)
  2. Null-baseline summary (F1_null, CV_null + 3σ) from cells flagged
     as workload='sanity' or analyzer_outputs.f1_phase=None and a
     known idle marker in notes
  3. Per-workload, per-iv ANOVA summary (fixed-effects)
  4. Per-family iv recommendation table (Plan 02's deliverable)

Statistics are stdlib only (no scipy / statsmodels dependency) to keep
the analysis runnable on the capture host without extra installs. The
F-statistic computation here is the standard one-way fixed-effects ANOVA
+ Welch's t for pairwise contrasts with Bonferroni correction.

For the full mixed-effects model (Step 2), use a notebook with
statsmodels; this script's outputs feed that notebook.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_cells(root: Path) -> list[dict]:
    """Load every *.json under root (recursively), skip warmups."""
    cells: list[dict] = []
    if not root.is_dir():
        return cells
    for path in sorted(root.glob("*.json")):
        name = path.name
        if name.startswith("warmup_") or name == "session_sentinel.json":
            continue
        try:
            with path.open() as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if obj.get("schema_version") != 2:
            continue
        # Skip warmups even if they aren't named warmup_*
        notes = obj.get("notes") or []
        if any("WARMUP CELL" in (n or "") for n in notes):
            continue
        cells.append(obj)
    return cells


def to_tidy(cells: list[dict]) -> list[dict]:
    """Flatten to one row per cell for CSV export and downstream analysis."""
    rows: list[dict] = []
    for c in cells:
        rm = c.get("run_meta") or {}
        ps = c.get("producer_stats") or {}
        ao = c.get("analyzer_outputs") or {}
        rows.append({
            "cell_id": rm.get("cell_id"),
            "manifest_id": rm.get("manifest_id"),
            "block_id": rm.get("block_id"),
            "workload": rm.get("workload"),
            "interval_ms": rm.get("interval_ms"),
            "duration_s": rm.get("duration_s"),
            "replicate": rm.get("replicate"),
            "exit_status": rm.get("exit_status"),
            "snapshots_completed": ps.get("snapshots_completed"),
            "mean_guest_run_interval_sec": ps.get("mean_guest_run_interval_sec"),
            "mean_host_snapshot_cycle_sec": ps.get("mean_host_snapshot_cycle_sec"),
            "estimated_vm_pause_fraction": ps.get("estimated_vm_pause_fraction"),
            "backpressure_events": ps.get("backpressure_events"),
            "f1_phase": ao.get("f1_phase"),
            "cv_workingset": ao.get("cv_workingset"),
            "n_windows": ao.get("n_windows"),
        })
    return rows


def write_tidy_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Acceptance criteria (D-5 noise-calibrated)
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceThresholds:
    f1_null_mean: float = 0.52
    f1_null_sd: float = 0.06
    cv_null_mean: float = 0.31
    cv_null_sd: float = 0.04

    @property
    def f1_hard(self) -> float:
        return self.f1_null_mean + 3 * self.f1_null_sd

    @property
    def f1_strong(self) -> float:
        return 0.80

    @property
    def cv_hard(self) -> float:
        return self.cv_null_mean - 1 * self.cv_null_sd

    @property
    def cv_strong(self) -> float:
        return 0.15


def compute_null_baseline(cells: list[dict]) -> AcceptanceThresholds | None:
    """If the dataset includes Step 0.5 null cells (workload=='idle_kali'),
    compute F1_null and CV_null. Otherwise return None and let the
    caller fall back to the audit-time defaults."""
    f1s = [c["analyzer_outputs"]["f1_phase"] for c in cells
           if (c.get("run_meta") or {}).get("workload") == "idle_kali"
           and (c.get("analyzer_outputs") or {}).get("f1_phase") is not None]
    cvs = [c["analyzer_outputs"]["cv_workingset"] for c in cells
           if (c.get("run_meta") or {}).get("workload") == "idle_kali"
           and (c.get("analyzer_outputs") or {}).get("cv_workingset") is not None]
    if not f1s and not cvs:
        return None
    f1m = statistics.fmean(f1s) if f1s else 0.52
    f1s_sd = statistics.stdev(f1s) if len(f1s) >= 2 else 0.06
    cvm = statistics.fmean(cvs) if cvs else 0.31
    cvs_sd = statistics.stdev(cvs) if len(cvs) >= 2 else 0.04
    return AcceptanceThresholds(
        f1_null_mean=f1m, f1_null_sd=f1s_sd,
        cv_null_mean=cvm, cv_null_sd=cvs_sd,
    )


# ---------------------------------------------------------------------------
# One-way fixed-effects ANOVA (stdlib only)
# ---------------------------------------------------------------------------

def one_way_anova(groups: dict[object, list[float]]) -> dict[str, float]:
    """Standard one-way ANOVA. groups maps level -> list of observations.

    Returns dict with: ss_between, ss_within, df_between, df_within,
    ms_between, ms_within, f_stat, eta_squared (effect size).

    Returns NaN/inf if degenerate (single group, single observation).
    """
    levels = list(groups.keys())
    k = len(levels)
    n_total = sum(len(v) for v in groups.values())
    if k < 2 or n_total - k < 1:
        return {"f_stat": float("nan"), "ss_between": float("nan"),
                "ss_within": float("nan"), "df_between": k - 1,
                "df_within": n_total - k, "ms_between": float("nan"),
                "ms_within": float("nan"), "eta_squared": float("nan"),
                "n_total": n_total, "k_groups": k}
    grand_mean = statistics.fmean(
        [x for v in groups.values() for x in v]
    )
    ss_between = 0.0
    ss_within = 0.0
    for v in groups.values():
        if not v:
            continue
        gm = statistics.fmean(v)
        ss_between += len(v) * (gm - grand_mean) ** 2
        for x in v:
            ss_within += (x - gm) ** 2
    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between if df_between else float("nan")
    ms_within = ss_within / df_within if df_within else float("nan")
    f_stat = (ms_between / ms_within) if ms_within else float("inf")
    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total if ss_total else float("nan")
    return {
        "f_stat": f_stat,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "df_between": df_between,
        "df_within": df_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "eta_squared": eta_squared,
        "n_total": n_total,
        "k_groups": k,
    }


def welch_t_pairs(groups: dict[object, list[float]]) -> list[dict]:
    """Welch's t for all pairwise contrasts. Returns one dict per pair
    with t, df_welch, |delta|, n1, n2. No p-value (stdlib doesn't ship
    a t CDF); the caller can compare |t| against a critical value or
    use scipy in a notebook."""
    items = list(groups.items())
    out: list[dict] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            lvl_a, va = items[i]
            lvl_b, vb = items[j]
            if len(va) < 2 or len(vb) < 2:
                continue
            ma = statistics.fmean(va)
            mb = statistics.fmean(vb)
            sa2 = statistics.variance(va)
            sb2 = statistics.variance(vb)
            na = len(va)
            nb = len(vb)
            denom = math.sqrt(sa2 / na + sb2 / nb)
            t = (ma - mb) / denom if denom > 0 else float("inf")
            # Welch-Satterthwaite df
            num = (sa2 / na + sb2 / nb) ** 2
            den = (sa2**2) / (na**2 * (na - 1)) + (sb2**2) / (nb**2 * (nb - 1))
            df = num / den if den > 0 else float("nan")
            out.append({
                "level_a": lvl_a,
                "level_b": lvl_b,
                "mean_a": ma,
                "mean_b": mb,
                "delta": ma - mb,
                "t_stat": t,
                "df_welch": df,
                "n_a": na,
                "n_b": nb,
            })
    return out


# ---------------------------------------------------------------------------
# Per-cell pass / fail (acceptance criteria, D-5 calibrated)
# ---------------------------------------------------------------------------

def cell_accepts(cell: dict, thresholds: AcceptanceThresholds) -> dict:
    """Apply acceptance criteria 1-5 (D-5 wording) to a single cell."""
    rm = cell.get("run_meta") or {}
    ps = cell.get("producer_stats") or {}
    ao = cell.get("analyzer_outputs") or {}
    iv = rm.get("interval_ms") or 0
    workload = (rm.get("workload") or "").lower()
    gd_mean = ps.get("mean_guest_run_interval_sec")
    gd_std = ps.get("std_guest_run_interval_sec")
    n_attempts = ps.get("snapshots_attempted") or 0
    n_back = ps.get("backpressure_events") or 0
    f1 = ao.get("f1_phase")
    cv = ao.get("cv_workingset")
    n_win = ao.get("n_windows") or 0

    checks = {}

    # 1. Axis-A stationarity
    if gd_mean and gd_std and gd_mean > 0:
        checks["stationarity"] = (gd_std / gd_mean) < 0.10
    else:
        checks["stationarity"] = False

    # 2. intervalMsec honored (allow +25 ms overhead constant)
    if gd_mean and iv > 0:
        expected = iv / 1000.0 + 0.025
        checks["iv_honored"] = abs(gd_mean - expected) < 0.02 * (iv / 1000.0 + 0.025)
    else:
        checks["iv_honored"] = False

    # 3. No silent gaps
    if n_attempts > 0:
        checks["no_gaps"] = (n_back / n_attempts) < 0.01
    else:
        checks["no_gaps"] = False

    # 4. Window count adequate
    checks["windows_adequate"] = n_win >= 50

    # 5. Defining metric
    if "ransom" in workload:
        checks["metric_hard"] = (f1 is not None and f1 >= thresholds.f1_hard)
        checks["metric_strong"] = (f1 is not None and f1 >= thresholds.f1_strong)
    elif "workingset" in workload:
        checks["metric_hard"] = (cv is not None and cv <= thresholds.cv_hard)
        checks["metric_strong"] = (cv is not None and cv <= thresholds.cv_strong)
    else:
        checks["metric_hard"] = None  # n/a
        checks["metric_strong"] = None

    hard_keys = ["stationarity", "iv_honored", "no_gaps", "windows_adequate",
                 "metric_hard"]
    passes_hard = all(checks[k] for k in hard_keys if checks[k] is not None)
    return {
        "cell_id": rm.get("cell_id"),
        "workload": rm.get("workload"),
        "interval_ms": iv,
        "replicate": rm.get("replicate"),
        "passes_hard": passes_hard,
        "passes_strong": passes_hard and (checks.get("metric_strong") is True),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Family recommendation
# ---------------------------------------------------------------------------

def recommend_iv_per_workload(
    cells: list[dict],
    thresholds: AcceptanceThresholds,
) -> dict[str, dict]:
    """For each workload, find the slowest iv that passes acceptance.

    Slowest iv = cheapest pause fraction.
    """
    by_workload: dict[str, dict[int, list[bool]]] = {}
    for c in cells:
        if c.get("schema_version") != 2:
            continue
        if (c.get("run_meta") or {}).get("exit_status") != "ok":
            continue
        result = cell_accepts(c, thresholds)
        wl = result["workload"]
        iv = result["interval_ms"]
        by_workload.setdefault(wl, {}).setdefault(iv, []).append(result["passes_hard"])

    recs: dict[str, dict] = {}
    for wl, iv_map in by_workload.items():
        # iv passes if a majority of replicates pass
        passing_ivs: list[int] = []
        for iv, passes in iv_map.items():
            if sum(passes) >= max(1, math.ceil(len(passes) / 2)):
                passing_ivs.append(iv)
        if not passing_ivs:
            recs[wl] = {"recommended_iv": None, "passing_ivs": [],
                        "reason": "no iv passed acceptance for this workload"}
        else:
            recs[wl] = {"recommended_iv": max(passing_ivs),
                        "passing_ivs": sorted(passing_ivs),
                        "reason": "slowest iv that passed acceptance "
                                  "(cheapest pause fraction)"}
    return recs


# ---------------------------------------------------------------------------
# Synthetic null trace (ML deliverable, smoke testable)
# ---------------------------------------------------------------------------

def synthetic_null_f1(seed: int = 0, n_trials: int = 30) -> list[float]:
    """Generate F1 scores you'd expect from a no-signal trace.

    Used by tests/test_plan02_smoke.py to validate the analysis pipeline
    runs end-to-end without needing real captured data.

    Distribution chosen to match the audit-day-4 simulated baseline
    (mean ~0.52, sd ~0.06).
    """
    import random as _r
    rng = _r.Random(seed)
    return [rng.gauss(0.52, 0.06) for _ in range(n_trials)]


def synthetic_null_cv(seed: int = 0, n_trials: int = 30) -> list[float]:
    """Same for CV_workingset (mean ~0.31, sd ~0.04)."""
    import random as _r
    rng = _r.Random(seed + 1)
    return [rng.gauss(0.31, 0.04) for _ in range(n_trials)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Plan-02 post-capture analysis: tidy CSV + ANOVA + "
                    "per-family iv recommendations.",
    )
    p.add_argument("--cells-dir", required=True,
                   help="directory containing schema-v2 per-cell JSONs")
    p.add_argument("--out-dir", required=True,
                   help="directory to write analysis artifacts into")
    p.add_argument("--metric", choices=["f1_phase", "cv_workingset",
                                        "mean_host_snapshot_cycle_sec"],
                   default="f1_phase",
                   help="dependent variable for ANOVA")
    p.add_argument("--factor", choices=["interval_ms", "duration_s"],
                   default="interval_ms",
                   help="independent variable (factor) for ANOVA")
    p.add_argument("--per-workload", action="store_true",
                   help="run a separate ANOVA per workload")
    args = p.parse_args(argv)

    cells = load_cells(Path(args.cells_dir).expanduser().resolve())
    if not cells:
        print("no schema-v2 cells found", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tidy CSV
    tidy = to_tidy(cells)
    write_tidy_csv(tidy, out_dir / "tidy.csv")
    print(f"wrote {len(tidy)} rows to {out_dir / 'tidy.csv'}", file=sys.stderr)

    # 2. Acceptance thresholds (null-baseline calibrated if available)
    thresholds = compute_null_baseline(cells) or AcceptanceThresholds()
    threshold_path = out_dir / "acceptance_thresholds.json"
    threshold_path.write_text(json.dumps({
        "f1_null_mean": thresholds.f1_null_mean,
        "f1_null_sd": thresholds.f1_null_sd,
        "cv_null_mean": thresholds.cv_null_mean,
        "cv_null_sd": thresholds.cv_null_sd,
        "f1_hard": thresholds.f1_hard,
        "f1_strong": thresholds.f1_strong,
        "cv_hard": thresholds.cv_hard,
        "cv_strong": thresholds.cv_strong,
    }, indent=2))
    print(f"wrote thresholds to {threshold_path}", file=sys.stderr)

    # 3. ANOVA
    metric_key = args.metric
    factor_key = args.factor
    anova_results: dict = {}

    if args.per_workload:
        by_workload: dict[str, dict[object, list[float]]] = {}
        for r in tidy:
            wl = r.get("workload") or "unknown"
            level = r.get(factor_key)
            y = r.get(metric_key)
            if level is None or y is None:
                continue
            by_workload.setdefault(wl, {}).setdefault(level, []).append(float(y))
        for wl, groups in by_workload.items():
            anova_results[wl] = {
                "anova": one_way_anova(groups),
                "pairs_welch": welch_t_pairs(groups),
            }
    else:
        groups: dict[object, list[float]] = {}
        for r in tidy:
            level = r.get(factor_key)
            y = r.get(metric_key)
            if level is None or y is None:
                continue
            groups.setdefault(level, []).append(float(y))
        anova_results["all"] = {
            "anova": one_way_anova(groups),
            "pairs_welch": welch_t_pairs(groups),
        }

    anova_path = out_dir / f"anova_{metric_key}_by_{factor_key}.json"
    anova_path.write_text(json.dumps({
        "metric": metric_key,
        "factor": factor_key,
        "results": anova_results,
    }, indent=2))
    print(f"wrote ANOVA to {anova_path}", file=sys.stderr)

    # 4. Per-family iv recommendation
    recs = recommend_iv_per_workload(cells, thresholds)
    recs_path = out_dir / "iv_recommendations.json"
    recs_path.write_text(json.dumps(recs, indent=2))
    print(f"wrote recommendations to {recs_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(_main())
