#!/usr/bin/env python3
"""app_compress_gzip_v2

Gzip compression workload. Generates a deterministic input file from a seeded
PRNG (random bytes — incompressible by design so the CPU+IO mix is stable),
then runs Python's `gzip` module to compress it. Sustained CPU + IO with no
strong periodicity.

Phase structure:
  1. setup_input  (write input file; excluded from measurement)
  2. compress     (read input → compress → write output)
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

TEST = "app_compress_gzip_v2"


def build_argparser() -> argparse.ArgumentParser:
    p = base_argparser(TEST, "Gzip compression workload")
    p.add_argument("--input-size-mb", type=int, default=1024)
    p.add_argument("--level", type=int, default=6, choices=range(1, 10))
    p.add_argument("--chunk-bytes", type=int, default=1 << 20,
                   help="Read/write chunk size (default 1 MiB)")
    p.add_argument("--reuse-input", action="store_true",
                   help="Reuse existing input file if present (skip regeneration)")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.duration is not None and args.duration < 0:
        args.duration = 0
    pin_cpu(args.cpu_affinity)

    sandbox = args.sandbox_dir or args.output_dir or "/tmp"
    os.makedirs(sandbox, exist_ok=True)
    input_path  = os.path.join(sandbox, f"compress_input_{os.getpid()}.bin")
    output_path = input_path + ".gz"
    validate_sandbox(input_path, extra_safe_root=sandbox)
    validate_sandbox(output_path, extra_safe_root=sandbox)

    total_bytes = args.input_size_mb * 1024 * 1024
    if total_bytes > 4 * 1024 * 1024 * 1024:
        raise SystemExit("input-size-mb cap: 4096 (4 GiB)")

    meta = Metadata(TEST, "Python", args.output_dir)
    meta.set_param(input_size_mb=args.input_size_mb, level=args.level,
                   chunk_bytes=args.chunk_bytes, seed=args.seed,
                   input_path=input_path, output_path=output_path)
    if args.dry_run:
        meta.set("status", "dry_run"); meta.write(); return 0

    rng = deterministic_rng(args.seed)

    phase(TEST, "setup_input")
    t0 = time.monotonic()
    if not (args.reuse_input and os.path.exists(input_path)
            and os.path.getsize(input_path) == total_bytes):
        with open(input_path, "wb") as f:
            remaining = total_bytes
            while remaining > 0:
                n = min(args.chunk_bytes, remaining)
                f.write(rng.randbytes(n))
                remaining -= n
    t_setup_end = time.monotonic()

    phase(TEST, "compress")
    t_comp_start = t_setup_end
    bytes_in = bytes_out = 0
    with open(input_path, "rb") as fin, gzip.open(output_path, "wb",
                                                  compresslevel=args.level) as fout:
        while True:
            chunk = fin.read(args.chunk_bytes)
            if not chunk:
                break
            fout.write(chunk)
            bytes_in += len(chunk)
            if args.duration and (time.monotonic() - t_comp_start) > args.duration:
                meta.add_limitation("compress phase truncated by --duration cap")
                break
    bytes_out = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    t_comp_end = time.monotonic()

    phase(TEST, "cleanup")
    if args.cleanup:
        for p in (input_path, output_path):
            if os.path.exists(p):
                os.unlink(p)
    t_clean_end = time.monotonic()

    meta.add_phase("setup_input", t0, t_setup_end)
    meta.add_phase("compress", t_comp_start, t_comp_end)
    meta.add_phase("cleanup", t_comp_end, t_clean_end)
    meta.set("status", "ok")
    meta.set("bytes_in", bytes_in)
    meta.set("bytes_out", bytes_out)
    meta.set("ratio", bytes_out / bytes_in if bytes_in else None)
    meta.add_limitation("Random input is near-incompressible by design; ratio ~ 1.0")
    meta.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
