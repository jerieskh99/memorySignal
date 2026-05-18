#!/usr/bin/env python3
"""app_sqlite_analytical_v2

Analytical sqlite workload: large SELECT scans, GROUP BY aggregations, and
windowed range scans on a pre-populated database. Cache is pre-warmed so the
workload exercises read-heavy access patterns with intermittent small writes
from temp-table materialization.

Phase structure:
  1. setup        (build/populate DB; may take time, excluded from measurement)
  2. prewarm      (large scan to fill page cache)
  3. measure      (repeated analytical queries for --duration seconds)
  4. cleanup      (drop db unless --keep-db)
"""
from __future__ import annotations

import os
import sys
import sqlite3
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.phase2_common import (
    base_argparser, phase, pin_cpu, Metadata, validate_sandbox, deterministic_rng,
)

TEST = "app_sqlite_analytical_v2"


def build_argparser() -> argparse.ArgumentParser:
    p = base_argparser(TEST, "Analytical sqlite workload (read-heavy)")
    p.add_argument("--db-path", type=str, default=None)
    p.add_argument("--rows", type=int, default=300000)
    p.add_argument("--query-set", type=str, default="aggregate,topn,range",
                   help="Comma-separated query types (default aggregate,topn,range)")
    p.add_argument("--keep-db", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.duration is None:
        args.duration = 60.0
    if args.duration > 1200:
        raise SystemExit("duration cap 1200s")
    pin_cpu(args.cpu_affinity)

    db_dir = args.output_dir or args.sandbox_dir or "/tmp"
    os.makedirs(db_dir, exist_ok=True)
    db_path = args.db_path or os.path.join(db_dir, f"app_sqlite_analytical_{os.getpid()}.db")
    validate_sandbox(db_path, extra_safe_root=db_dir)

    queries = [q.strip() for q in args.query_set.split(",") if q.strip()]
    valid = {"aggregate", "topn", "range"}
    if any(q not in valid for q in queries):
        raise SystemExit(f"unknown query type; allowed: {valid}")

    meta = Metadata(TEST, "Python", args.output_dir)
    meta.set_param(rows=args.rows, query_set=args.query_set,
                   duration_s=args.duration, seed=args.seed, db_path=db_path,
                   sqlite_version=sqlite3.sqlite_version)
    if args.dry_run:
        meta.set("status", "dry_run"); meta.write(); return 0

    rng = deterministic_rng(args.seed)
    if os.path.exists(db_path):
        os.unlink(db_path)

    phase(TEST, "setup")
    t0 = time.monotonic()
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA page_size=4096")
    conn.execute(
        "CREATE TABLE items "
        "(id INTEGER PRIMARY KEY, bucket INTEGER, score INTEGER, payload BLOB)"
    )
    conn.execute("CREATE INDEX idx_bucket ON items(bucket)")
    conn.execute("BEGIN")
    for i in range(args.rows):
        conn.execute(
            "INSERT INTO items VALUES (?, ?, ?, ?)",
            (i, rng.randrange(0, 100), rng.randrange(0, 1 << 31), rng.randbytes(128)),
        )
    conn.execute("COMMIT")
    conn.execute("PRAGMA wal_checkpoint(FULL)")
    t_setup_end = time.monotonic()

    phase(TEST, "prewarm")
    cur = conn.execute("SELECT COUNT(*), SUM(score) FROM items")
    cur.fetchone()
    t_prewarm_end = time.monotonic()

    phase(TEST, "measure")
    t_meas_start = t_prewarm_end
    aggregates = topns = ranges = 0
    while (time.monotonic() - t_meas_start) < args.duration:
        q = queries[rng.randrange(0, len(queries))]
        if q == "aggregate":
            cur = conn.execute(
                "SELECT bucket, COUNT(*), AVG(score) FROM items GROUP BY bucket"
            )
            cur.fetchall()
            aggregates += 1
        elif q == "topn":
            cur = conn.execute(
                "SELECT id, score FROM items ORDER BY score DESC LIMIT 100"
            )
            cur.fetchall()
            topns += 1
        else:  # range
            span = max(1, min(1000, args.rows // 4))
            hi = max(1, args.rows - span)
            lo = rng.randrange(0, hi)
            cur = conn.execute(
                "SELECT id, score FROM items WHERE id BETWEEN ? AND ?",
                (lo, lo + span),
            )
            cur.fetchall()
            ranges += 1
    t_meas_end = time.monotonic()
    conn.close()

    phase(TEST, "cleanup")
    db_bytes = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    if args.cleanup or not args.keep_db:
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.unlink(p)
    t_clean_end = time.monotonic()

    meta.add_phase("setup", t0, t_setup_end)
    meta.add_phase("prewarm", t_setup_end, t_prewarm_end)
    meta.add_phase("measure", t_meas_start, t_meas_end)
    meta.add_phase("cleanup", t_meas_end, t_clean_end)
    meta.set("status", "ok")
    meta.set("aggregates", aggregates)
    meta.set("topns", topns)
    meta.set("ranges", ranges)
    meta.set("db_bytes", db_bytes)
    meta.add_limitation("Aggregations materialize temp tables; small writes still occur")
    meta.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
