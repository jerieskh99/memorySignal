#!/usr/bin/env python3
"""plan05_aggregate.py -- Plan 05 throughput-pilot gates and recommendation.

Consumes the per-run records emitted by plan05_run.py and decides, for each
(workload, duration) group, whether a throughput lever arm beats the baseline
arm on all three Plan-05 gates:

  G-T1 throughput : median lever pmemsave_sec <= 0.5 x median baseline
                    (>= THROUGHPUT_MIN_SPEEDUP x reduction).
  G-T2 fidelity   : plan05_fidelity.fidelity_gate on the pooled APF trajectories
                    (TOST mean + bootstrap std + KS not rejected).
  G-T3 disk cap   : observed peak live-dump bytes <= DISK_CAP_BYTES.

An arm passes iff all three applicable gates pass. The recommendation, per
group, is the passing arm with the largest measured pmemsave speedup.

Input contract (one dict per capture run, produced by plan05_run.py):
    {
      "workload": str,            # e.g. "app_hashtable_intensive_v2"
      "duration_s": int,          # 120 | 600
      "arm": str,                 # e.g. "ssd_keep" | "tmpfs_keep" | "tmpfs_selfclean"
      "replicate": int,
      "mean_pmemsave_sec": float, # from e1.summarize()
      "snapshots_completed": int, # from e1.summarize()
      "peak_dump_bytes": int|None,# from e1 --instrument-peak-disk (None if off)
      "apf_jsonl": str|None,      # path to this run's apf_trajectory.jsonl
      "apf": [float]|None         # inline APF (tests); used if apf_jsonl absent
    }

Records are read from a runner JSON (`--results-json`, key "runs") or by globbing
`run_record.json` files under `--runs-dir`. No VM required.

    python3 plan05_aggregate.py --results-json plan05_results.json \\
        --summary-json plan05_summary.json --recommendation-json plan05_reco.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan05_fidelity as fidelity  # noqa: E402
from plan02_metrics_per_cell import load_streaming_trajectory  # noqa: E402

SUMMARY_SCHEMA = "plan05_summary.v1"
RECOMMENDATION_SCHEMA = "plan05_recommendation.v1"

# ---------------------------------------------------------------------------
# Calibratable gate constants (Wave 3 may revise).
# ---------------------------------------------------------------------------
THROUGHPUT_MIN_SPEEDUP = 2.0          # G-T1: lever must be >= 2x cheaper
DISK_CAP_BYTES = 3 * 1024 ** 3        # G-T3: 3 GiB peak live-dump ceiling
DEFAULT_BASELINE_ARM = "ssd_keep"     # current production setting = reference


@dataclass
class ArmComparison:
    workload: str
    duration_s: int
    arm: str
    baseline_arm: str
    n_baseline_runs: int
    n_lever_runs: int
    g_t1_throughput: dict
    g_t2_fidelity: dict
    g_t3_disk: dict
    passed: bool
    note: str = ""


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------
def load_records(results_json: Path | None, runs_dir: Path | None) -> list[dict]:
    records: list[dict] = []
    if results_json is not None:
        obj = json.loads(Path(results_json).read_text())
        runs = obj.get("runs", obj) if isinstance(obj, dict) else obj
        if not isinstance(runs, list):
            raise ValueError(f"{results_json}: expected a list under 'runs'")
        records.extend(runs)
    if runs_dir is not None:
        for p in sorted(Path(runs_dir).glob("**/run_record.json")):
            try:
                records.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
    return records


def _arm_apf(records: list[dict]) -> list[float]:
    """Pool APF across an arm's replicate runs (inline 'apf' or apf_jsonl)."""
    pooled: list[float] = []
    for r in records:
        if isinstance(r.get("apf"), list):
            pooled.extend(float(x) for x in r["apf"])
        elif r.get("apf_jsonl"):
            traj, _ = load_streaming_trajectory(Path(r["apf_jsonl"]))
            pooled.extend(traj)
    return pooled


def _median_or_none(vals: list[float]) -> float | None:
    clean = [float(v) for v in vals if isinstance(v, (int, float))]
    return float(statistics.median(clean)) if clean else None


# ---------------------------------------------------------------------------
# Per-arm gates
# ---------------------------------------------------------------------------
def _gate_throughput(base_runs: list[dict], lever_runs: list[dict],
                     min_speedup: float) -> dict:
    base_pmem = _median_or_none([r.get("mean_pmemsave_sec") for r in base_runs])
    lever_pmem = _median_or_none([r.get("mean_pmemsave_sec") for r in lever_runs])
    base_snaps = _median_or_none([r.get("snapshots_completed") for r in base_runs])
    lever_snaps = _median_or_none([r.get("snapshots_completed") for r in lever_runs])
    if base_pmem is None or lever_pmem is None or lever_pmem <= 0:
        return {"passed": False, "applicable": False,
                "reason": "missing/zero pmemsave median",
                "median_baseline_pmemsave_sec": base_pmem,
                "median_lever_pmemsave_sec": lever_pmem}
    speedup = base_pmem / lever_pmem
    return {"passed": bool(speedup >= min_speedup), "applicable": True,
            "median_baseline_pmemsave_sec": base_pmem,
            "median_lever_pmemsave_sec": lever_pmem,
            "speedup": speedup, "min_speedup": min_speedup,
            "median_baseline_snapshots": base_snaps,
            "median_lever_snapshots": lever_snaps}


def _gate_fidelity(base_runs: list[dict], lever_runs: list[dict]) -> dict:
    base_apf = _arm_apf(base_runs)
    lever_apf = _arm_apf(lever_runs)
    res = fidelity.fidelity_gate(base_apf, lever_apf)
    d = res.to_dict()
    d["applicable"] = res.applicable
    return d


def _gate_disk(lever_runs: list[dict], cap_bytes: int) -> dict:
    peaks = [r.get("peak_dump_bytes") for r in lever_runs
             if isinstance(r.get("peak_dump_bytes"), (int, float))]
    if not peaks:
        return {"passed": False, "applicable": False,
                "reason": "no peak_dump_bytes recorded (run with --instrument-peak-disk)",
                "cap_bytes": cap_bytes}
    peak = int(max(peaks))
    return {"passed": bool(peak <= cap_bytes), "applicable": True,
            "peak_dump_bytes": peak, "cap_bytes": cap_bytes,
            "peak_gib": round(peak / 1024 ** 3, 3),
            "cap_gib": round(cap_bytes / 1024 ** 3, 3)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(records: list[dict], *,
              baseline_arm: str = DEFAULT_BASELINE_ARM,
              min_speedup: float = THROUGHPUT_MIN_SPEEDUP,
              disk_cap_bytes: int = DISK_CAP_BYTES) -> dict:
    # group by (workload, duration)
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        key = (str(r.get("workload")), int(r.get("duration_s", 0)))
        groups.setdefault(key, []).append(r)

    comparisons: list[ArmComparison] = []
    notes: list[str] = []
    for (workload, duration), grp in sorted(groups.items()):
        base_runs = [r for r in grp if r.get("arm") == baseline_arm]
        if not base_runs:
            notes.append(f"{workload}@{duration}s: no baseline arm "
                         f"{baseline_arm!r}; group skipped")
            continue
        arms = sorted({r.get("arm") for r in grp if r.get("arm") != baseline_arm})
        for arm in arms:
            lever_runs = [r for r in grp if r.get("arm") == arm]
            g1 = _gate_throughput(base_runs, lever_runs, min_speedup)
            g2 = _gate_fidelity(base_runs, lever_runs)
            g3 = _gate_disk(lever_runs, disk_cap_bytes)
            applicable = g1["applicable"] and g2["applicable"] and g3["applicable"]
            passed = applicable and g1["passed"] and g2["passed"] and g3["passed"]
            if passed:
                note = "passes G-T1/G-T2/G-T3"
            else:
                note = "; ".join(
                    f"{nm}:{'pass' if g['passed'] else ('NA' if not g['applicable'] else 'fail')}"
                    for nm, g in (("G-T1", g1), ("G-T2", g2), ("G-T3", g3)))
            comparisons.append(ArmComparison(
                workload, duration, str(arm), baseline_arm,
                len(base_runs), len(lever_runs), g1, g2, g3, passed, note))

    # recommendation: best passing arm per (workload, duration) by speedup
    reco: list[dict] = []
    by_group: dict[tuple[str, int], list[ArmComparison]] = {}
    for c in comparisons:
        by_group.setdefault((c.workload, c.duration_s), []).append(c)
    for (workload, duration), comps in sorted(by_group.items()):
        passing = [c for c in comps if c.passed]
        if passing:
            best = max(passing, key=lambda c: c.g_t1_throughput.get("speedup", 0.0))
            reco.append({"workload": workload, "duration_s": duration,
                         "recommended_arm": best.arm,
                         "speedup": best.g_t1_throughput.get("speedup"),
                         "passes_all_gates": True})
        else:
            reco.append({"workload": workload, "duration_s": duration,
                         "recommended_arm": None,
                         "speedup": None, "passes_all_gates": False,
                         "note": "no arm cleared all three gates"})

    return {
        "schema": SUMMARY_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {"baseline_arm": baseline_arm, "min_speedup": min_speedup,
                   "disk_cap_bytes": disk_cap_bytes,
                   "disk_cap_gib": round(disk_cap_bytes / 1024 ** 3, 3),
                   "fidelity": {"mean_margin": fidelity.APF_MEAN_MARGIN,
                                "std_margin": fidelity.APF_STD_MARGIN,
                                "alpha_tost": fidelity.TOST_ALPHA,
                                "alpha_ks": fidelity.KS_ALPHA,
                                "ks_stat_margin": fidelity.KS_STAT_MARGIN,
                                "min_n": fidelity.FIDELITY_MIN_N}},
        "n_records": len(records),
        "n_groups": len(groups),
        "comparisons": [asdict(c) for c in comparisons],
        "recommendations": reco,
        "notes": notes,
    }


def _atomic_json(obj: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)
    tmp.replace(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="plan05_aggregate.py",
        description="Plan 05 throughput-pilot gates (G-T1/G-T2/G-T3) + recommendation.")
    src = p.add_argument_group("input (provide at least one)")
    src.add_argument("--results-json", default=None,
                     help="runner JSON with a 'runs' list of per-run records")
    src.add_argument("--runs-dir", default=None,
                     help="directory tree of run_record.json files")
    p.add_argument("--summary-json", default="plan05_summary.json")
    p.add_argument("--recommendation-json", default=None)
    p.add_argument("--baseline-arm", default=DEFAULT_BASELINE_ARM)
    p.add_argument("--min-speedup", type=float, default=THROUGHPUT_MIN_SPEEDUP)
    p.add_argument("--disk-cap-gib", type=float,
                   default=DISK_CAP_BYTES / 1024 ** 3)
    args = p.parse_args(argv)

    if not args.results_json and not args.runs_dir:
        p.error("provide --results-json and/or --runs-dir")
    records = load_records(
        Path(args.results_json) if args.results_json else None,
        Path(args.runs_dir) if args.runs_dir else None)
    if not records:
        print("no run records found", file=sys.stderr)
        return 2

    summary = aggregate(records, baseline_arm=args.baseline_arm,
                        min_speedup=args.min_speedup,
                        disk_cap_bytes=int(args.disk_cap_gib * 1024 ** 3))
    _atomic_json(summary, Path(args.summary_json))

    if args.recommendation_json:
        _atomic_json({"schema": RECOMMENDATION_SCHEMA,
                      "source": "plan05_aggregate.py",
                      "generated_at": summary["generated_at"],
                      "params": summary["params"],
                      "recommendations": summary["recommendations"]},
                     Path(args.recommendation_json))

    n_pass = sum(1 for c in summary["comparisons"] if c["passed"])
    print(f"plan05_aggregate: {len(summary['comparisons'])} arm-comparisons, "
          f"{n_pass} passing; wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
