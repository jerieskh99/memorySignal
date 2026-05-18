#!/usr/bin/env python3
"""app_decompress_gzip_v2

Gzip decompression workload. Generates a compressed input file (by compressing
deterministic random bytes with a controllable level), then decompresses it.
IO direction is inverted relative to compress: small read, large write, mixed
with CPU work.

Phase structure:
  1. setup_input   (generate then compress source file; excluded)
  2. decompress    (read compressed → decompress → write output)
  3. cleanup
"""
from __future__ import annotations

import os
import sys
import gzip
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.phase2_common import (
    base_argparser, phase, pin_cpu, Metadata, validate_sandbox, deterministic_rng,
)

TEST = "app_decompress_gzip_v2"


def build_argparser() -> argparse.ArgumentParser:
    p = base_argparser(TEST, "Gzip decompression workload")
    p.add_argument("--output-size-mb", type=int, default=1024,
                   help="Target decompressed size in MiB (default 1024)")
    p.add_argument("--level", type=int, default=6, choices=range(1, 10))
    p.add_argument("--chunk-bytes", type=int, default=1 << 20)
    p.add_argument("--reuse-input", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.duration is not None and args.duration < 0:
        args.duration = 0
    pin_cpu(args.cpu_affinity)

    sandbox = args.sandbox_dir or args.output_dir or "/tmp"
    os.makedirs(sandbox, exist_ok=True)
    raw_path  = os.path.join(sandbox, f"decompress_input_{os.getpid()}.bin")
    gz_path   = raw_path + ".gz"
    out_path  = os.path.join(sandbox, f"decompress_output_{os.getpid()}.bin")
    for p in (raw_path, gz_path, out_path):
        validate_sandbox(p, extra_safe_root=sandbox)

    total = args.output_size_mb * 1024 * 1024
    if total > 4 * 1024 * 1024 * 1024:
        raise SystemExit("output-size-mb cap: 4096")

    meta = Metadata(TEST, "Python", args.output_dir)
    meta.set_param(output_size_mb=args.output_size_mb, level=args.level,
                   chunk_bytes=args.chunk_bytes, seed=args.seed,
                   gz_path=gz_path, out_path=out_path)
    if args.dry_run:
        meta.set("status", "dry_run"); meta.write(); return 0

    rng = deterministic_rng(args.seed)

    phase(TEST, "setup_input")
    t0 = time.monotonic()
    if not (args.reuse_input and os.path.exists(gz_path)):
        # Generate raw deterministic bytes, then compress them to gz_path.
        with open(raw_path, "wb") as f:
            remaining = total
            while remaining > 0:
                n = min(args.chunk_bytes, remaining)
                f.write(rng.randbytes(n))
                remaining -= n
        with open(raw_path, "rb") as fin, gzip.open(gz_path, "wb",
                                                    compresslevel=args.level) as fout:
            while True:
                chunk = fin.read(args.chunk_bytes)
                if not chunk: break
                fout.write(chunk)
        os.unlink(raw_path)
    t_setup_end = time.monotonic()

    phase(TEST, "decompress")
    t_d_start = t_setup_end
    bytes_in = bytes_out = 0
    with gzip.open(gz_path, "rb") as fin, open(out_path, "wb") as fout:
        while True:
            chunk = fin.read(args.chunk_bytes)
            if not chunk:
                break
            fout.write(chunk)
            bytes_out += len(chunk)
            if args.duration and (time.monotonic() - t_d_start) > args.duration:
                meta.add_limitation("decompress phase truncated by --duration cap")
                break
    bytes_in = os.path.getsize(gz_path) if os.path.exists(gz_path) else 0
    t_d_end = time.monotonic()

    phase(TEST, "cleanup")
    if args.cleanup:
        for p in (gz_path, out_path):
            if os.path.exists(p):
                os.unlink(p)
    t_clean_end = time.monotonic()

    meta.add_phase("setup_input", t0, t_setup_end)
    meta.add_phase("decompress", t_d_start, t_d_end)
    meta.add_phase("cleanup", t_d_end, t_clean_end)
    meta.set("status", "ok")
    meta.set("bytes_in", bytes_in)
    meta.set("bytes_out", bytes_out)
    meta.add_limitation("Random source → high entropy; decompressor CPU vs IO ratio depends on level")
    meta.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
