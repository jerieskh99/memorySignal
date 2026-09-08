#!/usr/bin/env python3
"""differ.py -- run live_delta_calc_modular on one snapshot pair and read what it wrote.

The differ is the pipeline's definition of every channel; this module never computes a
metric itself. It runs the binary with `--speed N --sparse`, which writes one CSV row per
CHANGED page (hamming != 0) prefixed with page_index, and parses only the requested
columns.

Binary resolution, in order: $PLAN10_DIFFER, then
<repo>/VM_sampler/VM_Capture/live_delta_calc_modular/target/release/live_delta_calc_modular.
Neither present: refuse with the build line.

A constraint the differ does not document, found by probing it: it splits each file into
16 segments and reads 256 KB chunks from each segment's start, so unless the file size is
a multiple of 16 x 256 KB = 4 MiB the segments overlap and the running page counter
mis-indexes pages (a 256 KB pair reported one change three times). A 1 GiB guest dump is
fine; this module refuses any other size that is not a multiple of 4 MiB.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent.parent
DEFAULT_BINARY = REPO / "VM_sampler" / "VM_Capture" / "live_delta_calc_modular" / "target" / "release" / "live_delta_calc_modular"
DIFFER_SEGMENT_MULTIPLE = 16 * 256 * 1024   # THREAD_COUNT x CHUNK_SIZE in main.rs
PAGE = 4096


class DifferError(RuntimeError):
    pass


def find_differ() -> Path:
    env = os.environ.get("PLAN10_DIFFER")
    for cand in ([Path(env)] if env else []) + [DEFAULT_BINARY]:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    raise DifferError(
        "differ binary not found. Set PLAN10_DIFFER or build it:\n"
        f"  cd {DEFAULT_BINARY.parent.parent.parent} && cargo build --release")


def differ_version(binary: Path | None = None) -> dict:
    b = binary or find_differ()
    st = b.stat()
    return {"path": str(b), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def check_dump_size(path: Path) -> int:
    size = Path(path).stat().st_size
    if size % PAGE:
        raise DifferError(f"{path}: size {size} is not a whole number of pages")
    if size % DIFFER_SEGMENT_MULTIPLE:
        raise DifferError(
            f"{path}: size {size} is not a multiple of {DIFFER_SEGMENT_MULTIPLE} (16 threads x 256 KB chunks); "
            "the differ's segments would overlap and mis-index pages")
    return size // PAGE


def run_pair(prev: Path, curr: Path, out_dir: Path, speed: int, binary: Path | None = None) -> Path:
    """Run the differ; return the CSV it wrote."""
    b = binary or find_differ()
    check_dump_size(prev)
    check_dump_size(curr)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(b), "--speed", str(int(speed)), "--sparse", str(prev), str(curr), str(out_dir)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise DifferError(f"differ exit {r.returncode}: {r.stderr.strip()[:400]}")
    csvs = sorted((out_dir / "metrics").glob("page_metrics-*.csv")) if (out_dir / "metrics").is_dir() else []
    if not csvs:
        raise DifferError(f"differ wrote no metrics CSV under {out_dir}")
    return csvs[-1]


def parse_sparse_csv(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """{'page_index': int32[], <col>: float32[]} for the requested columns only."""
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        idx = {name: i for i, name in enumerate(header)}
        if "page_index" not in idx:
            raise DifferError(f"{path}: not a sparse CSV (no page_index column)")
        missing = [c for c in columns if c not in idx]
        if missing:
            raise DifferError(f"{path}: columns not in the differ's schema: {missing}")
        want = [idx[c] for c in columns]
        pages: list[int] = []
        vals: list[list[float]] = [[] for _ in columns]
        for row in rd:
            if not row:
                continue
            pages.append(int(row[0]))
            for j, i in enumerate(want):
                vals[j].append(float(row[i]))
    out = {"page_index": np.asarray(pages, dtype=np.int32)}
    for c, v in zip(columns, vals):
        out[c] = np.asarray(v, dtype=np.float32)
    return out


def diff_pair(prev: Path, curr: Path, speed: int, columns: list[str], work_dir: Path,
              binary: Path | None = None) -> dict[str, np.ndarray]:
    """Run + parse + clean up. Rows are the changed pages of this pair."""
    out_dir = Path(work_dir) / "differ_out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    csv_path = run_pair(prev, curr, out_dir, speed, binary)
    try:
        return parse_sparse_csv(csv_path, columns)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
