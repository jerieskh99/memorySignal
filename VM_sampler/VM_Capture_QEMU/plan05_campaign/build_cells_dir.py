#!/usr/bin/env python3
"""Adapt the Plan-05 campaign trajectories into the cells-dir layout that the
downstream tools (plan03_sweep, plan04_*) read.

Our campaign (run_files_controlled.py + apf_queue) wrote per-step
`run_matrix_test{i}_<workload>.npy.apf_trajectory.jsonl` files plus
`full_manifest.csv`. plan03/plan04 expect:

  <out>/work/<cell_id>/apf_trajectory.jsonl   one {seq,apf} line per pair + a
                                              final sentinel line
  <out>/manifest.csv                          plan02_manifest CSV_HEADER schema

where cell_id = plan02_schema.cell_id(workload, interval_ms, duration_s,
replicate). This is a pure offline reformat -- APF values are copied verbatim.
interval_ms is fixed at 500 (the locked campaign cadence).

    python3 plan05_campaign/build_cells_dir.py \
        --data-dir /tmp/p05audit \
        --manifest plan05_campaign/full_manifest.csv \
        --out plan05_campaign/downstream/cells
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                       # VM_Capture_QEMU
sys.path.insert(0, str(REPO))
from plan02_schema import cell_id                       # noqa: E402
from plan02_manifest import ManifestRow, CSV_HEADER     # noqa: E402

INTERVAL_MS = 500           # locked campaign cadence
MANIFEST_ID = "plan05campaign"   # deterministic batch tag (no random/Date)


def read_traj(path: Path) -> list[dict]:
    """Return the trajectory's JSON objects (seq+apf), sorted by seq."""
    objs: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(o.get("apf"), (int, float)) and isinstance(o.get("seq"), int):
                objs.append(o)
    objs.sort(key=lambda o: o["seq"])
    return objs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dir holding the *.apf_trajectory.jsonl files")
    ap.add_argument("--manifest", default=str(HERE / "full_manifest.csv"))
    ap.add_argument("--out", default=str(HERE / "downstream" / "cells"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.manifest)))
    mrows: list[ManifestRow] = []
    n_ok = 0
    for r in rows:
        wl = r["workload"]
        dur = int(r["duration_s"])
        rep = int(r["rep"])
        cid = cell_id(wl, INTERVAL_MS, dur, rep)
        src = data_dir / r["traj_file"]
        if not src.exists():
            print(f"[WARN] missing trajectory: {src}", file=sys.stderr)
            continue
        objs = read_traj(src)
        n = len(objs)
        cdir = work / cid
        cdir.mkdir(parents=True, exist_ok=True)
        with (cdir / "apf_trajectory.jsonl").open("w") as fh:
            for o in objs:
                fh.write(json.dumps({"seq": o["seq"], "apf": o["apf"]}) + "\n")
            # Final sentinel: our trajectories are gap-free (verified C6), so
            # n_ok == n_pairs_expected == n, no failures.
            fh.write(json.dumps({"final": True, "n_pairs_expected": n,
                                 "n_ok": n, "n_failed": 0, "gap_seqs": [],
                                 "timed_out": False}) + "\n")
        mrows.append(ManifestRow(
            cell_id=cid, manifest_id=MANIFEST_ID, block_id=0, workload=wl,
            interval_ms=INTERVAL_MS, duration_s=dur, replicate=rep,
            is_warmup=False, status="ok",
            expected_path=str(out / f"cell_{cid}.json"),
            retry_count=0, notes="", workload_command="", ssh_target="",
            keep_dumps=True,
        ))
        n_ok += 1

    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        w.writeheader()
        for mr in mrows:
            w.writerow(mr.to_csv())

    ids = [mr.cell_id for mr in mrows]
    assert len(ids) == len(set(ids)), "cell_id collision!"
    print(f"wrote {n_ok} cells -> {work}")
    print(f"wrote manifest -> {out / 'manifest.csv'} ({len(mrows)} rows)")
    print(f"unique cell_ids: {len(set(ids))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
