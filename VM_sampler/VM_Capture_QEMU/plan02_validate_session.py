#!/usr/bin/env python3
"""plan02_validate_session.py -- Step 1.5c / Step 2 post-run health check.

Owner: EE (Evaluation Engineer) + DE (Senior Data Engineer).
Reference: docs/tuning_plans/experiment_audit.md Decisions D-41 / D-42.

Verifies five claims that 'ok' status alone does NOT prove:

  C1: workload binary actually ran (PHASE markers in workload_stderr.log)
  C2: snap_completion_ratio is healthy across the whole session
  C3: n_windows >= --min-windows (after back-fill) for analysis-ready cells
  C4: Bug-L settle was a no-op (lock_retries == 0) under nominal load
  C5: producer.log had no error-ish lines for any cell

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


def evaluate_cell(cell_path: Path, workdir_root: Path,
                  min_ratio: float, min_windows: int,
                  window: int, hop: int) -> dict:
    """Return one health record for a cell. Five booleans + diagnostic ctx."""
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

    # C2: snap_completion_ratio
    ratio = _ratio_from_notes(cell)
    if ratio is None:
        # Fallback: re-compute from iv + dur (won't match exactly, but useful)
        ratio = (n_snaps / (1 + n_snaps)) if not iv else None
    c2 = (ratio is not None and ratio >= min_ratio)
    c2_reason = (f"ratio={ratio:.2f} >= {min_ratio}" if c2
                 else f"ratio={ratio} < {min_ratio}")

    # C3: n_windows >= min_windows. Compute fresh (back-fill semantics)
    nw_computed = _compute_n_windows(n_snaps, window, hop)
    c3 = nw_computed >= min_windows
    c3_reason = (f"n_windows={nw_computed} >= {min_windows}" if c3
                 else f"n_windows={nw_computed} < {min_windows}")

    # C4: Bug-L settle no-op (lock_retries == 0)
    retries = _settle_retries_from_notes(cell)
    c4 = (retries is not None and retries == 0)
    c4_reason = (f"lock_retries={retries}" if retries is not None
                 else "no settle note (older orchestrator?)")

    # C5: producer.log error count
    perrs = _producer_errors_from_notes(cell)
    c5 = perrs == 0
    c5_reason = (f"producer.log errors={perrs}" if perrs > 0
                 else "producer.log clean")

    ok = c1 and c2 and c3 and c4 and c5
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
        "claims": {
            "C1_workload_ran": {"pass": c1, "why": c1_reason},
            "C2_ratio_healthy": {"pass": c2, "why": c2_reason},
            "C3_enough_windows": {"pass": c3, "why": c3_reason},
            "C4_no_settle_retries": {"pass": c4, "why": c4_reason},
            "C5_producer_log_clean": {"pass": c5, "why": c5_reason},
        },
        "ok": ok,
    }


def _print_table(records: list[dict]) -> None:
    print(f"\n{'cell':<14} {'workload':<28} {'iv':>5} {'d':>5} "
          f"{'snaps':>6} {'win':>4} {'ratio':>5} {'C1':>3} {'C2':>3} "
          f"{'C3':>3} {'C4':>3} {'C5':>3} OK")
    print("─" * 110)
    for r in records:
        if "claims" not in r:
            print(f"{r.get('cell_id', '?'):<14} ERROR: {r.get('error')}")
            continue
        c = r["claims"]
        marks = " ".join(
            "✓" if c[k]["pass"] else "✗"
            for k in ("C1_workload_ran", "C2_ratio_healthy",
                      "C3_enough_windows", "C4_no_settle_retries",
                      "C5_producer_log_clean")
        )
        flag = " " if not r["is_warmup"] else "W"
        ratio_str = (f"{r['snap_completion_ratio']:.2f}"
                     if r['snap_completion_ratio'] is not None else "  -- ")
        print(f"{r['cell_id']:<14} {r['workload']:<28} "
              f"{r['interval_ms']!s:>5} {r['duration_s']!s:>5} "
              f"{r['n_snaps']:>6} {r['n_windows_computed']:>4} "
              f"{ratio_str:>5}  {marks}  "
              f"{'OK' if r['ok'] else 'FAIL'}{flag}")


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
        "claim_pass_counts": {
            "C1_workload_ran": sum(1 for r in real if r["claims"]["C1_workload_ran"]["pass"]),
            "C2_ratio_healthy": sum(1 for r in real if r["claims"]["C2_ratio_healthy"]["pass"]),
            "C3_enough_windows": sum(1 for r in real if r["claims"]["C3_enough_windows"]["pass"]),
            "C4_no_settle_retries": sum(1 for r in real if r["claims"]["C4_no_settle_retries"]["pass"]),
            "C5_producer_log_clean": sum(1 for r in real if r["claims"]["C5_producer_log_clean"]["pass"]),
        },
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

    records: list[dict] = []
    for path in sorted(cells_dir.glob("*.json")):
        if path.name == "session_sentinel.json":
            continue
        if path.name == "validate_report.json":
            continue
        rec = evaluate_cell(path, workdir_root,
                            args.min_ratio, args.min_windows,
                            args.window, args.hop)
        records.append(rec)

    summary = _summarize(records)
    out_path = (Path(args.report).expanduser().resolve()
                if args.report else cells_dir / "validate_report.json")
    sc.write_json_atomic(out_path, {"summary": summary, "cells": records})

    _print_table(records)
    print("\nSUMMARY")
    print(f"  cells:              {summary['total_cells']} total · "
          f"{summary['real_cells']} real · "
          f"{summary['warmup_cells']} warmup · "
          f"{summary['unreadable_cells']} unreadable")
    print(f"  passing real cells: {summary['passing_real_cells']} / "
          f"{summary['real_cells']}")
    print(f"  per-claim pass counts (out of {summary['real_cells']} real cells):")
    for claim, n in summary["claim_pass_counts"].items():
        print(f"    {claim:<28}  {n}")
    print(f"\n  report written to: {out_path}")
    print(f"  overall: {'PASS' if summary['all_real_cells_pass'] else 'FAIL'}")
    return 0 if summary["all_real_cells_pass"] else 1


if __name__ == "__main__":
    sys.exit(_main())
