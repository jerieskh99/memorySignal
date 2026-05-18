#!/usr/bin/env python3
"""app_sqlite_oltp_v2

OLTP-style sqlite workload: a fixed-size schema with mixed INSERT, UPDATE-by-PK
and point-SELECT transactions. WAL mode is enabled so the workload exposes the
WAL-append + checkpoint rhythm distinct from synthetic mem/io primitives.

Phase structure:
  1. setup     (schema + initial population, excluded from measurement)
  2. measure   (mixed transactions for --duration seconds)
  3. checkpoint (final WAL checkpoint)
  4. cleanup   (drop db file unless --keep-db)

Standard library only (uses sqlite3 from CPython).
"""
from __future__ import annotations

import os
import sys
import sqlite3
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.phase2_common import (
    base_argparser, log_info, log_warn, phase, pin_cpu, Metadata,
    validate_sandbox, deterministic_rng,
)

TEST = "app_sqlite_oltp_v2"


def build_argparser() -> argparse.ArgumentParser:
    p = base_argparser(TEST, "OLTP-style sqlite workload (WAL mode)")
    p.add_argument("--db-path", type=str, default=None,
                   help="Database file path; defaults to <output-dir or /tmp>/app_sqlite_oltp.db")
    p.add_argument("--rows", type=int, default=200000,
                   help="Initial table rows (default 200000 ≈ 100 MB)")
    p.add_argument("--tx-per-batch", type=int, default=50,
                   help="Statements per BEGIN/COMMIT batch (default 50)")
    p.add_argument("--mix", type=str, default="40-30-30",
                   help="INSERT-UPDATE-SELECT mix in percent (default 40-30-30)")
    p.add_argument("--keep-db", action="store_true",
                   help="Keep DB file after run (default: removed unless --cleanup explicit)")
    return p


def parse_mix(spec: str) -> tuple[int, int, int]:
    parts = spec.split("-")
    if len(parts) != 3:
        raise SystemExit(f"--mix must be I-U-S (got {spec})")
    i, u, s = (int(x) for x in parts)
    if i + u + s != 100:
        raise SystemExit("--mix must sum to 100")
    return i, u, s


def main() -> int:
    args = build_argparser().parse_args()
    if args.duration is None:
        args.duration = 60.0
    if args.duration > 1200:
        raise SystemExit("duration cap 1200s for safety")
    pin_cpu(args.cpu_affinity)

    db_dir = args.output_dir or args.sandbox_dir or "/tmp"
    os.makedirs(db_dir, exist_ok=True)
    db_path = args.db_path or os.path.join(db_dir, f"app_sqlite_oltp_{os.getpid()}.db")
    validate_sandbox(db_path, extra_safe_root=db_dir)

    meta = Metadata(TEST, "Python", args.output_dir)
    meta.set_param(
        rows=args.rows, tx_per_batch=args.tx_per_batch, mix=args.mix,
        duration_s=args.duration, seed=args.seed, db_path=db_path,
        sqlite_version=sqlite3.sqlite_version,
    )

    if args.dry_run:
        meta.set("status", "dry_run")
        meta.write()
        return 0

    rng = deterministic_rng(args.seed)
    mix_i, mix_u, mix_s = parse_mix(args.mix)

    if os.path.exists(db_path):
        os.unlink(db_path)

    phase(TEST, "setup")
    t0 = time.monotonic()
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA page_size=4096")
    conn.execute("PRAGMA mmap_size=0")  # disable mmap mode for clean io semantics
    conn.execute(
        "CREATE TABLE IF NOT EXISTS items "
        "(id INTEGER PRIMARY KEY, k TEXT, v BLOB, score INTEGER)"
    )
    conn.execute("BEGIN")
    for i in range(args.rows):
        conn.execute(
            "INSERT INTO items (id, k, v, score) VALUES (?, ?, ?, ?)",
            (i, f"k_{i:08d}", rng.randbytes(256), rng.randrange(0, 1 << 31)),
        )
    conn.execute("COMMIT")
    conn.execute("PRAGMA wal_checkpoint(FULL)")
    t_setup_end = time.monotonic()

    phase(TEST, "measure")
    t_meas_start = t_setup_end
    inserts = updates = selects = 0
    next_id = args.rows
    while (time.monotonic() - t_meas_start) < args.duration:
        conn.execute("BEGIN")
        for _ in range(args.tx_per_batch):
            roll = rng.randrange(0, 100)
            if roll < mix_i:
                conn.execute(
                    "INSERT INTO items (id, k, v, score) VALUES (?, ?, ?, ?)",
                    (next_id, f"k_{next_id:08d}", rng.randbytes(256), rng.randrange(0, 1 << 31)),
                )
                next_id += 1
                inserts += 1
            elif roll < mix_i + mix_u:
                target = rng.randrange(0, next_id)
                conn.execute(
                    "UPDATE items SET score=? WHERE id=?",
                    (rng.randrange(0, 1 << 31), target),
                )
                updates += 1
            else:
                target = rng.randrange(0, next_id)
                cur = conn.execute("SELECT score FROM items WHERE id=?", (target,))
                cur.fetchone()
                selects += 1
        conn.execute("COMMIT")
    t_meas_end = time.monotonic()

    phase(TEST, "checkpoint")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    t_ckpt_end = time.monotonic()

    phase(TEST, "cleanup")
    bytes_used = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    if args.cleanup or not args.keep_db:
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.unlink(p)
    t_clean_end = time.monotonic()

    meta.add_phase("setup", t0, t_setup_end)
    meta.add_phase("measure", t_meas_start, t_meas_end)
    meta.add_phase("checkpoint", t_meas_end, t_ckpt_end)
    meta.add_phase("cleanup", t_ckpt_end, t_clean_end)
    meta.set("status", "ok")
    meta.set("inserts", inserts)
    meta.set("updates", updates)
    meta.set("selects", selects)
    meta.set("db_bytes", bytes_used)
    meta.add_limitation("WAL checkpoint frequency depends on sqlite defaults; use PRAGMA to tune")
    meta.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
