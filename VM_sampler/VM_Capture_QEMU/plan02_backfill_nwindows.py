#!/usr/bin/env python3
"""plan02_backfill_nwindows.py -- Step 1.5a back-fill.

Owner: ML (Senior ML Engineer) + DS (Senior Data Scientist).
Reference: docs/tuning_plans/experiment_audit.md Decision D-21.

For every schema-v2 per-cell JSON in a directory, compute n_windows from
the snapshot count using the Phase-1 canonical window=128 hop=64. Write
the result into analyzer_outputs.n_windows. If n_windows is below the
acceptance floor (default 50), also flip the manifest row status from
'ok' to 'skipped' (D-14: per-cell skip, not per-cell fail).

DOES NOT compute f1_phase or cv_workingset. The Step 1 pilot did not run
any workload binary and did not preserve dump content, so those metrics
are not computable from existing data. They remain None with an
explanatory note appended to the per-cell JSON.

To populate f1_phase / cv_workingset properly, run the next pilot with
workload-launching enabled (manifest --workload-command + --ssh-target
+ --keep-dumps) and then wire offline_step_metrics.py via a follow-up
script (Step 1.5b/c).

CLI:

  python3 plan02_backfill_nwindows.py \\
      --cells-dir /path/to/cells \\
      --manifest /path/to/manifest.csv \\
      --window 128 --hop 64 --min-windows 50

Safe to re-run: an already-populated cell is detected and updated
in-place. Manifest mutations are atomic (temp + rename via plan02_manifest).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import plan02_schema as sc
import plan02_manifest as mf


WINDOW_DEFAULT = 128
HOP_DEFAULT = 64
MIN_WINDOWS_DEFAULT = 50

BACKFILL_NOTE = (
    "step-1.5a: n_windows back-filled from snapshot count "
    "(window={w}, hop={h}). f1_phase and cv_workingset left None: "
    "Step 1 pilot was a capture-pipeline characterization "
    "(no workload launched, no dumps preserved); page-content metrics "
    "require Step 1.5b workload-launching."
)


def compute_n_windows(n_snaps: int, window: int = WINDOW_DEFAULT,
                       hop: int = HOP_DEFAULT) -> int:
    """Standard sliding-window count: floor((N - window) / hop) + 1.

    Returns 0 if N < window (no complete window possible).
    """
    if n_snaps < window:
        return 0
    return max(0, (n_snaps - window) // hop + 1)


def backfill_one(cell_path: Path, window: int, hop: int,
                 min_windows: int, dry_run: bool = False) -> dict:
    """Update one cell JSON. Returns dict with cell_id, n_snaps, n_windows,
    skipped (bool), updated (bool).
    """
    try:
        with cell_path.open() as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"cell_path": str(cell_path), "error": str(e)}

    if payload.get("schema_version") != sc.SCHEMA_VERSION:
        return {"cell_path": str(cell_path), "skipped": True,
                "reason": "not schema v2"}

    rm = payload.get("run_meta") or {}
    ps = payload.get("producer_stats") or {}
    ao = payload.setdefault("analyzer_outputs", {})
    notes = payload.setdefault("notes", [])

    n_snaps = int(ps.get("snapshots_completed") or 0)
    n_windows = compute_n_windows(n_snaps, window=window, hop=hop)

    ao["n_windows"] = n_windows
    ao["n_snapshots"] = n_snaps

    note_str = BACKFILL_NOTE.format(w=window, h=hop)
    if not any("step-1.5a" in (n or "") for n in notes):
        notes.append(note_str)

    skipped = n_windows < min_windows

    result = {
        "cell_id": rm.get("cell_id"),
        "workload": rm.get("workload"),
        "interval_ms": rm.get("interval_ms"),
        "duration_s": rm.get("duration_s"),
        "n_snaps": n_snaps,
        "n_windows": n_windows,
        "skipped": skipped,
        "updated": True,
    }

    if not dry_run:
        sc.write_json_atomic(cell_path, payload)

    return result


def apply_skip_to_manifest(manifest_path: Path, skip_cell_ids: set[str],
                           dry_run: bool = False) -> int:
    """Flip manifest rows: status='ok' && cell_id in skip set -> 'skipped'.

    D-14: cells with too few windows are skipped, not failed. Only flip
    cells whose current status is 'ok' so we don't clobber 'failed' or
    user-set 'skipped' rows.
    """
    rows = mf.load(manifest_path)
    flipped = 0
    for r in rows:
        if r.cell_id in skip_cell_ids and r.status == "ok":
            r.status = "skipped"
            r.notes = (r.notes + " | step-1.5a: n_windows below floor; "
                       "skipped per D-14").strip(" |")
            flipped += 1
    if flipped and not dry_run:
        mf.save(manifest_path, rows)
    return flipped


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Step 1.5a back-fill: populate n_windows on every "
                    "schema-v2 per-cell JSON and flip the manifest "
                    "status to 'skipped' for cells below the window floor."
    )
    p.add_argument("--cells-dir", required=True,
                   help="directory holding schema-v2 per-cell JSONs")
    p.add_argument("--manifest", default=None,
                   help="manifest CSV. If provided, cells below "
                        "--min-windows are flipped 'ok' -> 'skipped'.")
    p.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    p.add_argument("--hop", type=int, default=HOP_DEFAULT)
    p.add_argument("--min-windows", type=int, default=MIN_WINDOWS_DEFAULT)
    p.add_argument("--dry-run", action="store_true",
                   help="report what would happen; do not write")
    args = p.parse_args(argv)

    cells_dir = Path(args.cells_dir).expanduser().resolve()
    if not cells_dir.is_dir():
        print(f"ERROR: cells-dir not a directory: {cells_dir}", file=sys.stderr)
        return 2

    skip_ids: set[str] = set()
    results: list[dict] = []
    n_updated = 0
    n_below = 0
    for p_json in sorted(cells_dir.glob("*.json")):
        # Skip warmup and sentinel
        if p_json.name == "session_sentinel.json":
            continue
        if p_json.name.startswith("warmup_"):
            continue
        r = backfill_one(p_json, args.window, args.hop, args.min_windows,
                         dry_run=args.dry_run)
        results.append(r)
        if r.get("updated"):
            n_updated += 1
        if r.get("skipped"):
            n_below += 1
            if r.get("cell_id"):
                skip_ids.add(r["cell_id"])

    print(f"\nback-filled {n_updated} cells "
          f"(window={args.window}, hop={args.hop}, min={args.min_windows})",
          file=sys.stderr)
    print(f"  {n_below} cells fall below n_windows floor "
          f"(will be marked 'skipped' in manifest)",
          file=sys.stderr)

    # Detailed breakdown by (workload, iv, duration)
    from collections import defaultdict
    bucket: dict[tuple, list[int]] = defaultdict(list)
    for r in results:
        if "n_windows" not in r:
            continue
        bucket[(r["workload"], r["interval_ms"], r["duration_s"])].append(
            r["n_windows"])
    print("\nn_windows by (workload, iv, duration):", file=sys.stderr)
    for key in sorted(bucket):
        nws = bucket[key]
        flag = "  " if min(nws) >= args.min_windows else " *"
        print(f"{flag} {key[0]:>32}  iv={key[1]:>5}ms  d={key[2]:>4}s  "
              f"n_windows={nws}", file=sys.stderr)

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.is_file():
            print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
            return 2
        flipped = apply_skip_to_manifest(manifest_path, skip_ids,
                                          dry_run=args.dry_run)
        print(f"\nmanifest: flipped {flipped} ok -> skipped rows "
              f"(dry-run={args.dry_run})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(_main())
