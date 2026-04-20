#!/usr/bin/env python3
"""
Concatenate step run_matrix .npy files along the time axis (more columns).

Each file must have the same first dimension (num_pages). On disk layout is
(pages, frames) as produced by the QEMU consumer.

Use this to merge several short idle segments so streaming_metrics has enough
frames for MSC (needs T >= win_len + hop_len with default sliding logic).

Example:

  python3 combine_run_matrices.py \\
    -o memory_traces/queue_dir/run_matrix_idle_merged.npy \\
    memory_traces/queue_dir/run_matrix_test1_run_idle.npy \\
    memory_traces/queue_dir/run_matrix_test5_run_idle.npy \\
    memory_traces/queue_dir/run_matrix_test7_run_idle.npy
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output .npy path (pages x total_frames).",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Input run_matrix .npy files, in chronological order.",
    )
    args = p.parse_args()

    mats: list[np.ndarray] = []
    n_pages: int | None = None
    for path in args.inputs:
        m = np.load(path, mmap_mode="r")
        m = np.asarray(m, dtype=np.float64)
        if m.ndim != 2:
            print(f"ERROR: expected 2D array in {path}, got shape {m.shape}", file=sys.stderr)
            return 1
        if n_pages is None:
            n_pages = m.shape[0]
        elif m.shape[0] != n_pages:
            print(
                f"ERROR: page count mismatch {path}: rows {m.shape[0]} != {n_pages}",
                file=sys.stderr,
            )
            return 1
        mats.append(m)

    merged = np.hstack(mats)
    np.save(args.output, merged)
    print(
        f"Wrote {args.output} shape={merged.shape} "
        f"(pages={merged.shape[0]}, frames={merged.shape[1]}) from {len(mats)} file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
