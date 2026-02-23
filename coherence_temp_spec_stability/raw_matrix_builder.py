"""
Build a time-series matrix from raw memory dump (ELF) files.

Reads the newest N dump files from a directory, sorts by mtime ascending (oldest -> newest),
and for each dump computes one scalar per page according to a chosen mode. Output is
a matrix of shape (num_pages, num_frames) in float32, suitable for downstream
coherence/temporal/spectral stability metrics (e.g. stability_validator / streaming_metrics).

Modes:
  mean_byte   : mean of uint8 values in each page
  var_byte    : variance of uint8 values in each page
  entropy     : Shannon entropy of the byte histogram per page (0..8 bits)
  popcount    : sum of bit counts per byte in page, normalized by (page_size*8) -> [0,1]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _page_scalar_mean_byte(page: np.ndarray) -> float:
    return float(np.mean(page.astype(np.float64)))


def _page_scalar_var_byte(page: np.ndarray) -> float:
    return float(np.var(page.astype(np.float64)))


def _page_scalar_entropy(page: np.ndarray) -> float:
    hist, _ = np.histogram(page.ravel(), bins=256, range=(0, 256))
    p = hist.astype(np.float64) / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _page_scalar_popcount(page: np.ndarray) -> float:
    # popcount per byte, sum over page, normalize by (page_size * 8) so result in [0, 1]
    bits = np.unpackbits(page.ravel())
    return float(np.sum(bits) / (page.size * 8))


MODE_FUNCS = {
    "mean_byte": _page_scalar_mean_byte,
    "var_byte": _page_scalar_var_byte,
    "entropy": _page_scalar_entropy,
    "popcount": _page_scalar_popcount,
}


def _dumps_in_dir(input_dir: str, keep: int) -> list[Path]:
    """Return paths to the newest `keep` dump files, sorted by mtime ascending (oldest first)."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        return []
    # Prefer .elf; allow any file as dump
    files = [
        p
        for p in input_path.iterdir()
        if p.is_file()
    ]
    # Sort by mtime descending (newest first), take first `keep`, then reverse so oldest->newest
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = files[:keep]
    chosen.sort(key=lambda p: p.stat().st_mtime)
    return chosen


def build_matrix(
    input_dir: str,
    keep: int,
    mode: str,
    page_size: int,
    max_bytes: int,
    output_path: str,
) -> None:
    """
    Build (num_pages, num_frames) float32 matrix from raw dumps; write .npy atomically.
    """
    dumps = _dumps_in_dir(input_dir, keep)
    if not dumps:
        raise ValueError(f"No dump files in {input_dir} or keep={keep} yielded none")

    scalar_fn = MODE_FUNCS.get(mode)
    if scalar_fn is None:
        raise ValueError(f"Unknown mode: {mode}. Choose from: {list(MODE_FUNCS)}")

    frames = []
    num_pages_prev = None

    for path in dumps:
        with open(path, "rb") as f:
            data = f.read()
        if max_bytes > 0:
            data = data[:max_bytes]
        arr = np.frombuffer(data, dtype=np.uint8)
        # Trim to full pages
        n = (len(arr) // page_size) * page_size
        arr = arr[:n].reshape(-1, page_size)

        page_scalars = np.array([scalar_fn(arr[i, :]) for i in range(arr.shape[0])], dtype=np.float32)
        if num_pages_prev is not None and num_pages_prev != page_scalars.shape[0]:
            raise ValueError(
                f"Page count mismatch: {path.name} has {page_scalars.shape[0]} pages, previous had {num_pages_prev}"
            )
        num_pages_prev = page_scalars.shape[0]
        frames.append(page_scalars)

    # Stack: each frame is (num_pages,), we want matrix (num_pages, num_frames)
    matrix = np.stack(frames, axis=1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp.npy")
    np.save(tmp, matrix)
    tmp.rename(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build raw time-series matrix from memory dump files (for PLV/MSC/Cepstrum)."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing dump (.elf) files")
    parser.add_argument("--keep", type=int, required=True, help="Use newest N dumps")
    parser.add_argument("--mode", type=str, default="mean_byte", choices=list(MODE_FUNCS),
                        help="Per-page scalar: mean_byte | var_byte | entropy | popcount")
    parser.add_argument("--page-size", type=int, default=4096, help="Page size in bytes")
    parser.add_argument("--output", type=str, required=True, help="Output .npy path (atomic write)")
    parser.add_argument("--max-bytes", type=int, default=0,
                        help="Cap bytes read per file (0 = whole file; for testing)")
    args = parser.parse_args()

    build_matrix(
        input_dir=args.input_dir,
        keep=args.keep,
        mode=args.mode,
        page_size=args.page_size,
        max_bytes=args.max_bytes,
        output_path=args.output,
    )
    print(f"Saved matrix shape (num_pages, num_frames) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
