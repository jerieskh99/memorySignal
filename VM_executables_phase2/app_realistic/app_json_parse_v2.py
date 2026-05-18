#!/usr/bin/env python3
"""app_json_parse_v2

JSON parsing workload: generates a deterministic 500 MB JSONL file (one record
per line) from a seeded PRNG, then streams it through json.loads() while
maintaining running aggregates so the parser allocations are observable.

Phase structure:
  1. setup_input  (write input file; excluded)
  2. parse        (read + json.loads + aggregate)
  3. cleanup
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.phase2_common import (
    base_argparser, phase, pin_cpu, Metadata, validate_sandbox, deterministic_rng,
)

TEST = "app_json_parse_v2"


def build_argparser() -> argparse.ArgumentParser:
    p = base_argparser(TEST, "Streaming JSON parse workload")
    p.add_argument("--input-size-mb", type=int, default=500)
    p.add_argument("--reuse-input", action="store_true")
    return p


def _gen_record(rng, idx):
    return {
        "id": idx,
        "k": f"key_{idx:08d}",
        "tags": [f"t{rng.randrange(0, 32)}" for _ in range(4)],
        "score": rng.randrange(0, 1 << 31),
        "payload": rng.randbytes(64).hex(),
    }


def main() -> int:
    args = build_argparser().parse_args()
    pin_cpu(args.cpu_affinity)

    sandbox = args.sandbox_dir or args.output_dir or "/tmp"
    os.makedirs(sandbox, exist_ok=True)
    input_path = os.path.join(sandbox, f"json_input_{os.getpid()}.jsonl")
    validate_sandbox(input_path, extra_safe_root=sandbox)

    target_bytes = args.input_size_mb * 1024 * 1024
    if target_bytes > 4 * 1024 * 1024 * 1024:
        raise SystemExit("input-size-mb cap: 4096")

    meta = Metadata(TEST, "Python", args.output_dir)
    meta.set_param(input_size_mb=args.input_size_mb, seed=args.seed,
                   input_path=input_path)
    if args.dry_run:
        meta.set("status", "dry_run"); meta.write(); return 0

    rng = deterministic_rng(args.seed)

    phase(TEST, "setup_input")
    t0 = time.monotonic()
    if not (args.reuse_input and os.path.exists(input_path)
            and os.path.getsize(input_path) >= target_bytes):
        records = 0
        with open(input_path, "wb") as f:
            written = 0
            while written < target_bytes:
                line = (json.dumps(_gen_record(rng, records)) + "\n").encode()
                f.write(line)
                written += len(line)
                records += 1
        meta.set("records_written", records)
    t_setup_end = time.monotonic()

    phase(TEST, "parse")
    t_p_start = t_setup_end
    parsed = 0
    score_sum = 0
    tag_count = 0
    with open(input_path, "rb") as f:
        for line in f:
            obj = json.loads(line)
            score_sum += obj["score"]
            tag_count += len(obj["tags"])
            parsed += 1
            if args.duration and (time.monotonic() - t_p_start) > args.duration:
                meta.add_limitation("parse phase truncated by --duration cap")
                break
    t_p_end = time.monotonic()

    phase(TEST, "cleanup")
    if args.cleanup and os.path.exists(input_path):
        os.unlink(input_path)
    t_clean_end = time.monotonic()

    meta.add_phase("setup_input", t0, t_setup_end)
    meta.add_phase("parse", t_p_start, t_p_end)
    meta.add_phase("cleanup", t_p_end, t_clean_end)
    meta.set("status", "ok")
    meta.set("records_parsed", parsed)
    meta.set("score_sum", score_sum)
    meta.set("tag_count", tag_count)
    meta.add_limitation("json module is C-backed but allocation cadence is CPython-specific")
    meta.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
