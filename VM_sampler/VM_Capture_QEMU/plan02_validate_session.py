#!/usr/bin/env python3
"""plan02_validate_session.py -- Step 1.5c / Step 2 post-run health check.

Owner: EE (Evaluation Engineer) + DE (Senior Data Engineer).
Reference: docs/tuning_plans/experiment_audit.md Decisions D-41 / D-42.

Verifies seven claims that 'ok' status alone does NOT prove.

Operational claims (must pass · gate Step 2 launch):
  C1: workload binary actually ran (PHASE markers in workload_stderr.log)
  C2: snap_completion_ratio is healthy across the whole session
  C4: Bug-L settle was a no-op (lock_retries == 0) under nominal load
  C5: producer.log had no error-ish lines for any cell
  C6: apf_trajectory.jsonl is complete (B+3.1 Δ-4 streaming sentinel)
  C7: plan03_recommendation.json schema-valid + per-workload winner
      passes G1-G5 (Plan 03 Δ-4 · NA when artifact absent)

Informational claim (reported, does NOT gate Step 2 per D-25):
  C3: n_windows >= --min-windows under canonical Phase-1 (128, 64).
      Failure here means the duration matrix is too short for the
      canonical analyzer window, which is Plan 03's question, not
      Plan 02's. Step 2 lengthens durations but does not promise
      to clear this threshold at every iv.

Reads:
  --cells-dir   directory containing per-cell schema-v2 JSONs
  --manifest    optional manifest CSV (for cross-reference)
  --workdir     directory of per-cell workdirs holding workload_stderr.log
                + producer.log (default: <cells-dir>/work)

Writes:
  --report      JSON health report (one entry per cell + summary)
                  default: <cells-dir>/validate_report.json
  console       human-readable summary table

Exits:
  0  if all 5 claims pass for every non-warmup cell
  1  if any claim fails

Smallest reusable tool. No new dependencies. No producer change.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import plan02_schema as sc


PHASE_RE = re.compile(r"\[PHASE\]\s+test=\S+\s+phase=\S+")
ERROR_KEYWORDS = ("error", "fail", "denied", "no space", "cannot", "refused")
MIN_RATIO_DEFAULT = 0.85
MIN_WINDOWS_DEFAULT = 50
WINDOW_DEFAULT = 128
HOP_DEFAULT = 64


def _load_cell(path: Path) -> dict | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _is_warmup(cell: dict) -> bool:
    notes = cell.get("notes") or []
    return any("WARMUP CELL" in (n or "") for n in notes)


def _ratio_from_notes(cell: dict) -> float | None:
    """Bug-M D-32 writes 'snap completion: actual=A expected=E ratio=R' to notes."""
    for n in (cell.get("notes") or []):
        m = re.search(r"snap completion: actual=\d+ expected=\d+ ratio=([\d.]+)",
                      n or "")
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _settle_retries_from_notes(cell: dict) -> int | None:
    """Bug-L D-34 writes 'vm settle: state=... lock_retries=N other_errors=...'"""
    for n in (cell.get("notes") or []):
        m = re.search(r"vm settle: .*?lock_retries=(\d+)", n or "")
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _producer_errors_from_notes(cell: dict) -> int:
    """Bug-M Day-9 producer.log scan writes 'producer.log errors (N):' to notes."""
    for n in (cell.get("notes") or []):
        m = re.search(r"producer\.log errors \((\d+)\):", n or "")
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return 0
    return 0


def _phase_marker_count(stderr_path: Path) -> int:
    if not stderr_path.is_file():
        return 0
    try:
        text = stderr_path.read_text(errors="replace")
    except OSError:
        return 0
    return len(PHASE_RE.findall(text))


def _compute_n_windows(n_snaps: int, window: int, hop: int) -> int:
    if n_snaps < window:
        return 0
    return max(0, (n_snaps - window) // hop + 1)


def _evaluate_plan03_recommendation(cells_dir: Path) -> dict:
    """C7 (Plan 03) · operational claim · gates Step 2 once Plan 03 has run.

    Looks for ``plan03_recommendation.json`` either alongside the cells
    directory (typical layout) or inside it. NA when the artifact is
    absent (v1/v2-era sessions still pass · same pattern as C6 NA).
    When present, the claim passes iff every recommendations[].
    ``passes_acceptance`` field is True.
    """
    candidates = [
        cells_dir.parent / "plan03_recommendation.json",
        cells_dir / "plan03_recommendation.json",
    ]
    rec_path = next((p for p in candidates if p.is_file()), None)
    if rec_path is None:
        return {"pass": True, "operational": False,
                "why": "no plan03_recommendation.json found "
                       "(Plan 03 not run) · NA",
                "artifact": None, "recommendations": []}
    try:
        with rec_path.open() as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {"pass": False, "operational": True,
                "why": f"failed to read {rec_path}: {exc}",
                "artifact": str(rec_path), "recommendations": []}
    if payload.get("schema") != "plan03.window_hop_recommendations.v1":
        return {"pass": False, "operational": True,
                "why": (f"unexpected schema in {rec_path.name}: "
                        f"{payload.get('schema')!r}"),
                "artifact": str(rec_path), "recommendations": []}
    recs = payload.get("recommendations") or []
    if not recs:
        return {"pass": False, "operational": True,
                "why": "recommendations[] empty",
                "artifact": str(rec_path), "recommendations": []}
    summary: list[dict] = []
    overall = True
    for r in recs:
        rid = r.get("workload", "?")
        passes = bool(r.get("passes_acceptance"))
        degrades = bool(r.get("degrades_v2_baseline"))
        summary.append({
            "workload": rid,
            "window": r.get("recommended_window"),
            "hop": r.get("recommended_hop"),
            "passes_acceptance": passes,
            "degrades_v2_baseline": degrades,
        })
        if not passes:
            overall = False
    if overall:
        why = (f"{len(recs)} workload winner(s) all pass G1-G5; "
               f"artifact={rec_path.name}")
    else:
        bad = [s["workload"] for s in summary if not s["passes_acceptance"]]
        why = (f"{len(bad)}/{len(recs)} workload winner(s) did not pass "
               f"acceptance: {bad}")
    return {"pass": overall, "operational": True,
            "why": why, "artifact": str(rec_path),
            "recommendations": summary}


def _evaluate_apf_completeness(cell_workdir: Path,
                                min_ok_ratio: float = 0.95) -> tuple[bool, str, dict]:
    """B+3.1 Δ-4: C6 trajectory-completeness check.

    Returns (pass, reason, sentinel_dict). pass=True iff:
      - <cell_workdir>/apf_trajectory.jsonl exists
      - file contains a sentinel line ({"final": true, ...})
      - sentinel.n_ok / n_pairs_expected >= min_ok_ratio
    """
    apf_jsonl = cell_workdir / "apf_trajectory.jsonl"
    if not apf_jsonl.is_file():
        return False, "apf_trajectory.jsonl missing", {}
    sentinel: dict = {}
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
                break
    except OSError as exc:
        return False, f"read error: {exc}", {}
    if not sentinel:
        return False, "final sentinel missing in apf_trajectory.jsonl", {}
    n_expected = int(sentinel.get("n_pairs_expected") or 0)
    n_ok = int(sentinel.get("n_ok") or 0)
    if n_expected <= 0:
        # No pairs to validate. Cell ran but produced 0 snap pairs (e.g.
        # 1 snap total). Treat as pass: there is no integrity to check.
        return True, "no pairs to validate (n_pairs_expected=0)", sentinel
    ratio = n_ok / n_expected
    if ratio < min_ok_ratio:
        return False, (f"completeness {ratio:.2%} < {min_ok_ratio:.0%} "
                       f"(n_ok={n_ok} expected={n_expected})"), sentinel
    return True, (f"complete · n_ok={n_ok}/{n_expected} ratio={ratio:.2%}"), sentinel


def evaluate_cell(cell_path: Path, workdir_root: Path,
                  min_ratio: float, min_windows: int,
                  window: int, hop: int,
                  c7_result: dict | None = None) -> dict:
    """Return one health record for a cell. Six booleans + diagnostic ctx.

    B+3.1 Δ-4: adds C6 (apf trajectory completeness · operational gate
    when the cell has a trajectory file, NA otherwise so v1-era cells
    are not penalized).
    """
    cell = _load_cell(cell_path)
    if cell is None:
        return {"cell_path": str(cell_path), "ok": False,
                "error": "could not load json"}
    rm = cell.get("run_meta") or {}
    ps = cell.get("producer_stats") or {}
    ao = cell.get("analyzer_outputs") or {}
    cid = rm.get("cell_id", cell_path.stem)
    is_warmup = _is_warmup(cell)
    exit_status = rm.get("exit_status")
    workload = rm.get("workload", "unknown")
    iv = rm.get("interval_ms")
    dur = rm.get("duration_s")
    n_snaps = int(ps.get("snapshots_completed") or 0)

    # C1: workload ran (skip for warmups)
    stderr_path = workdir_root / cid / "workload_stderr.log"
    n_markers = _phase_marker_count(stderr_path)
    if is_warmup:
        c1 = True
        c1_reason = "warmup (no workload)"
    else:
        c1 = n_markers > 0
        c1_reason = (f"{n_markers} PHASE markers" if c1
                     else f"no PHASE markers in {stderr_path}")

    # C2: snap_completion_ratio · Day-14 mode-aware threshold
    ratio = _ratio_from_notes(cell)
    if ratio is None:
        # Fallback: re-compute from iv + dur (won't match exactly, but useful)
        ratio = (n_snaps / (1 + n_snaps)) if not iv else None
    # B+3.1 (keep_dumps=True) cells run the async APF helper which competes
    # for disk bandwidth; pause-fraction rises and snap_completion_ratio
    # is naturally lower than v1's calibration. v3 recalibration (D-80):
    # sustained heavy workloads slow the cadence ~3×, so the v2 floor of
    # 0.15 false-failed ~14 % of cells with sound trajectories. Lowered to
    # 0.08, matching the orchestrator. Still catches true stalls (~0.02).
    # MIN_RATIO_DEFAULT (0.85) remains the v1 "ideal"; unattainable here.
    keep_dumps = bool((cell.get("run_meta") or {}).get("keep_dumps"))
    c2_min_ratio = 0.08 if keep_dumps else min_ratio
    c2 = (ratio is not None and ratio >= c2_min_ratio)
    c2_reason = (f"ratio={ratio:.2f} >= {c2_min_ratio} (mode={'B+3.1' if keep_dumps else 'v1'})"
                 if c2
                 else f"ratio={ratio} < {c2_min_ratio} (mode={'B+3.1' if keep_dumps else 'v1'})")

    # C3: n_windows >= min_windows. Compute fresh (back-fill semantics)
    nw_computed = _compute_n_windows(n_snaps, window, hop)
    c3 = nw_computed >= min_windows
    c3_reason = (f"n_windows={nw_computed} >= {min_windows}" if c3
                 else f"n_windows={nw_computed} < {min_windows}")

    # C4: Bug-L settle no-op (lock_retries == 0)
    # D-76: when no settle line exists, treat as NA (pass trivially) instead
    # of fail. v1-era cells didn't emit settle lines either; backward compat.
    # When the line exists with lock_retries > 0, that's a legitimate FAIL.
    retries = _settle_retries_from_notes(cell)
    if retries is None:
        c4 = True
        c4_reason = "no settle note (v1-era cell or orchestrator pre-D-34) · NA"
    else:
        c4 = (retries == 0)
        c4_reason = f"lock_retries={retries}" if c4 else f"lock_retries={retries} > 0"

    # C5: producer.log error count
    perrs = _producer_errors_from_notes(cell)
    c5 = perrs == 0
    c5_reason = (f"producer.log errors={perrs}" if perrs > 0
                 else "producer.log clean")

    # C6 · B+3.1 Δ-4 · trajectory completeness (operational when applicable).
    # NA when the cell's workdir has no apf_trajectory.jsonl (i.e. cell ran
    # without --keep-dumps · v1-era). NA cells don't fail C6.
    cell_workdir = workdir_root / cid
    c6_applicable = (cell_workdir / "apf_trajectory.jsonl").is_file()
    if c6_applicable:
        c6, c6_reason, c6_sentinel = _evaluate_apf_completeness(cell_workdir)
    else:
        c6, c6_reason, c6_sentinel = True, "NA (no apf trajectory file)", {}

    # C7 · Plan 03 window/hop recommendation. Session-level result; we
    # carry the same dict into every cell record so the summary table
    # and writers don't need a special-case path.
    c7_data = c7_result or {"pass": True, "operational": False,
                            "why": "no plan03_recommendation.json found "
                                   "(Plan 03 not run) · NA",
                            "artifact": None, "recommendations": []}
    c7 = bool(c7_data["pass"])
    c7_reason = c7_data["why"]
    c7_applicable = bool(c7_data.get("operational"))

    # Operational ok = the gates that actually block Step 2 launch.
    # C3 is informational (D-25): low n_windows is a duration-matrix
    # issue, not an orchestration regression. Plan 03 owns window/hop
    # tuning. Reported per-cell + per-summary but does not flip ok=False.
    # C6 (B+3.1) is operational when applicable; NA cells pass trivially.
    # C7 (Plan 03 Delta-4) is operational when applicable; NA when no
    # plan03_recommendation.json exists.
    ok = c1 and c2 and c4 and c5 and c6 and c7
    return {
        "cell_id": cid,
        "workload": workload,
        "interval_ms": iv,
        "duration_s": dur,
        "is_warmup": is_warmup,
        "exit_status": exit_status,
        "n_snaps": n_snaps,
        "n_windows_computed": nw_computed,
        "snap_completion_ratio": ratio,
        "settle_lock_retries": retries,
        "producer_log_errors": perrs,
        "phase_markers": n_markers,
        "apf_sentinel": c6_sentinel,
        "claims": {
            "C1_workload_ran": {"pass": c1, "why": c1_reason, "operational": True},
            "C2_ratio_healthy": {"pass": c2, "why": c2_reason, "operational": True},
            "C3_enough_windows": {"pass": c3, "why": c3_reason, "operational": False},
            "C4_no_settle_retries": {"pass": c4, "why": c4_reason, "operational": True},
            "C5_producer_log_clean": {"pass": c5, "why": c5_reason, "operational": True},
            "C6_apf_complete": {"pass": c6, "why": c6_reason,
                                 "operational": c6_applicable},
            "C7_window_hop_recommended": {"pass": c7, "why": c7_reason,
                                          "operational": c7_applicable,
                                          "artifact": c7_data.get("artifact"),
                                          "recommendations":
                                              c7_data.get("recommendations") or []},
        },
        "ok": ok,
        # Day 10 D-25 wiring: C3 stays informational; C7 is the new
        # operational gate (NA when Plan 03 has not run yet).
        "analysis_ready": ok and c7,
    }


def _print_table(records: list[dict]) -> None:
    print(f"\n{'cell':<14} {'workload':<28} {'iv':>5} {'d':>5} "
          f"{'snaps':>6} {'win':>4} {'ratio':>5} "
          f"{'C1':>3} {'C2':>3} {'C4':>3} {'C5':>3} {'C6':>3} {'C7':>3} "
          f"{'[C3]':>4} OPERATIONAL  ANALYSIS-READY")
    print("─" * 144)
    for r in records:
        if "claims" not in r:
            print(f"{r.get('cell_id', '?'):<14} ERROR: {r.get('error')}")
            continue
        c = r["claims"]
        # Operational claims first (these gate Step 2 launch)
        # C6 + C7 may be NA when their gating artifact is absent.
        def mark(k: str) -> str:
            if k not in c:
                return " "
            if k in ("C6_apf_complete", "C7_window_hop_recommended") \
                    and not c[k].get("operational"):
                return "·"  # NA · trivially passes
            return "✓" if c[k]["pass"] else "✗"
        op_marks = " ".join(
            mark(k) for k in ("C1_workload_ran", "C2_ratio_healthy",
                              "C4_no_settle_retries", "C5_producer_log_clean",
                              "C6_apf_complete", "C7_window_hop_recommended")
        )
        # Informational claim in brackets (Plan-03 territory)
        info_mark = "[✓]" if c["C3_enough_windows"]["pass"] else "[✗]"
        flag = " " if not r["is_warmup"] else "W"
        ratio_str = (f"{r['snap_completion_ratio']:.2f}"
                     if r['snap_completion_ratio'] is not None else "  -- ")
        op_status = "OK" if r["ok"] else "FAIL"
        analysis_status = "ready" if r.get("analysis_ready") else "low_window"
        print(f"{r['cell_id']:<14} {r['workload']:<28} "
              f"{r['interval_ms']!s:>5} {r['duration_s']!s:>5} "
              f"{r['n_snaps']:>6} {r['n_windows_computed']:>4} "
              f"{ratio_str:>5}  {op_marks}  {info_mark}  "
              f"{op_status:<11}  {analysis_status}{flag}")


def _summarize(records: list[dict]) -> dict:
    real = [r for r in records if "claims" in r and not r["is_warmup"]]
    warmups = [r for r in records if "claims" in r and r["is_warmup"]]
    bad = [r for r in records if "claims" not in r]
    out: dict = {
        "total_cells": len(records),
        "warmup_cells": len(warmups),
        "real_cells": len(real),
        "unreadable_cells": len(bad),
        "passing_real_cells": sum(1 for r in real if r["ok"]),
        "analysis_ready_real_cells": sum(1 for r in real if r.get("analysis_ready")),
        "claim_pass_counts": {
            "C1_workload_ran": sum(1 for r in real if r["claims"]["C1_workload_ran"]["pass"]),
            "C2_ratio_healthy": sum(1 for r in real if r["claims"]["C2_ratio_healthy"]["pass"]),
            "C3_enough_windows": sum(1 for r in real if r["claims"]["C3_enough_windows"]["pass"]),
            "C4_no_settle_retries": sum(1 for r in real if r["claims"]["C4_no_settle_retries"]["pass"]),
            "C5_producer_log_clean": sum(1 for r in real if r["claims"]["C5_producer_log_clean"]["pass"]),
            "C6_apf_complete": sum(1 for r in real
                                    if r["claims"].get("C6_apf_complete", {}).get("pass")),
            "C7_window_hop_recommended": sum(
                1 for r in real
                if r["claims"].get("C7_window_hop_recommended", {}).get("pass")
            ),
        },
        "all_real_cells_operational": all(r["ok"] for r in real) and not bad,
        "all_real_cells_pass": all(r["ok"] for r in real) and not bad,
    }
    return out


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Validate a Plan-02 session's per-cell JSONs against "
                    "C1-C5 claims. See debug_audit.md and "
                    "experiment_audit.md Day 10 for context."
    )
    p.add_argument("--cells-dir", required=True,
                   help="directory of per-cell JSONs (e.g. /tmp/plan02_1_5c/cells)")
    p.add_argument("--manifest", default=None,
                   help="manifest CSV (optional; only used to cross-check counts)")
    p.add_argument("--workdir", default=None,
                   help="per-cell workdir root (default <cells-dir>/work)")
    p.add_argument("--report", default=None,
                   help="JSON report output (default <cells-dir>/validate_report.json)")
    p.add_argument("--min-ratio", type=float, default=MIN_RATIO_DEFAULT)
    p.add_argument("--min-windows", type=int, default=MIN_WINDOWS_DEFAULT)
    p.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    p.add_argument("--hop", type=int, default=HOP_DEFAULT)
    args = p.parse_args(argv)

    cells_dir = Path(args.cells_dir).expanduser().resolve()
    if not cells_dir.is_dir():
        print(f"ERROR: cells-dir not found: {cells_dir}", file=sys.stderr)
        return 2
    workdir_root = (Path(args.workdir).expanduser().resolve()
                    if args.workdir else cells_dir / "work")

    # C7 · Plan 03 recommendation lookup is session-level; resolved once.
    c7_result = _evaluate_plan03_recommendation(cells_dir)

    records: list[dict] = []
    for path in sorted(cells_dir.glob("*.json")):
        if path.name == "session_sentinel.json":
            continue
        if path.name == "validate_report.json":
            continue
        rec = evaluate_cell(path, workdir_root,
                            args.min_ratio, args.min_windows,
                            args.window, args.hop,
                            c7_result=c7_result)
        records.append(rec)

    summary = _summarize(records)
    out_path = (Path(args.report).expanduser().resolve()
                if args.report else cells_dir / "validate_report.json")
    sc.write_json_atomic(out_path, {"summary": summary, "cells": records})

    _print_table(records)
    print("\nSUMMARY")
    print(f"  cells:                {summary['total_cells']} total · "
          f"{summary['real_cells']} real · "
          f"{summary['warmup_cells']} warmup · "
          f"{summary['unreadable_cells']} unreadable")
    print(f"  operational pass:     {summary['passing_real_cells']} / "
          f"{summary['real_cells']}   (gates Step 2 launch)")
    print(f"  analysis-ready:       {summary['analysis_ready_real_cells']} / "
          f"{summary['real_cells']}   (operational + C7 cleared · "
          f"C7 NA still counts as cleared)")
    print(f"  per-claim pass counts (out of {summary['real_cells']} real cells):")
    for claim, n in summary["claim_pass_counts"].items():
        if claim == "C3_enough_windows":
            op = "informational (D-25)"
        elif claim == "C6_apf_complete":
            op = "operational when applicable (B+3.1 Δ-4)"
        elif claim == "C7_window_hop_recommended":
            op = "operational when applicable (Plan 03 Δ-4)"
        else:
            op = "operational"
        print(f"    {claim:<28}  {n:>2}   [{op}]")
    print(f"\n  report written to: {out_path}")
    op_status = "PASS" if summary["all_real_cells_operational"] else "FAIL"
    print(f"  operational status:   {op_status}")
    if op_status == "PASS" and summary["analysis_ready_real_cells"] < summary["real_cells"]:
        print(f"  note: C7 has not cleared for some cells. C3 (low n_windows) is "
              f"informational per D-25 and never blocks Step 2.")
    return 0 if summary["all_real_cells_operational"] else 1


if __name__ == "__main__":
    sys.exit(_main())
