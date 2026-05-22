#!/usr/bin/env python3
"""migrate_schema_v1_to_v2.py -- one-shot v1 -> v2 migrator for Plan-02.

Owner: DE (Senior Data Engineer).
Reference: docs/tuning_plans/experiment_audit.md Decision D-2.

Converts R1-R5 era per-cell JSONs (no schema_version, ad-hoc field layout)
into Plan-02 schema v2. Preserves the original file at <name>.v1.bak
unless --no-backup is set. Writes the upgraded file at the same path.

Field mappings:
  v1 root                    -> v2 path
  experiment (string)        -> notes (one entry)
  timestamp                  -> run_meta.run_ended_at
  config.interval_ms         -> run_meta.interval_ms
  config.duration_sec        -> run_meta.duration_s
  config.ram_size_mb         -> notes (informational)
  config.vm_domain           -> notes
  config.self_clean          -> notes (Plan 1c flag)
  config.queue_files_drained -> notes (Plan 1c metric)
  summary.* (or comparison.* in 2c)
                             -> producer_stats.* (best-effort field match)
  notes                      -> notes (extended)

Unknown / non-mappable v1 fields are dropped from the v2 payload but
preserved in the .v1.bak. The audit trail is still complete.

Cell identity in v2 is built from (workload, interval_ms, duration_s,
replicate). v1 had no notion of replicate; the migrator assigns
replicate=0 unless --replicate is set. For multi-replicate v1 archives,
run the migrator with explicit --replicate values per file.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import plan02_schema as sc


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_v1_workload(payload: dict) -> str:
    """Best-effort workload inference from v1 payload."""
    exp = (payload.get("experiment") or "").lower()
    if "ransom" in exp:
        return "sandbox_ransom_batched"
    if "workingset" in exp:
        return "mem_workingset_sweep_v2"
    if "timing_instrumentation" in exp or "sanity" in exp:
        return "sanity"
    if "2a" in exp or "consumer" in exp:
        return "exp2a_consumer_isolation"
    if "2b" in exp or "interval_sweep" in exp:
        return "exp2b_interval_sweep"
    if "2c" in exp or "flush" in exp:
        return "exp2c_flush_sensitivity"
    return "unknown_v1"


def migrate_one(
    v1_payload: dict,
    workload_override: str | None,
    replicate: int,
    manifest_id: str = "v1_archive",
) -> dict:
    cfg = v1_payload.get("config") or {}
    summary = v1_payload.get("summary") or v1_payload.get("passes") or {}

    workload = workload_override or detect_v1_workload(v1_payload)
    interval_ms = int(cfg.get("interval_ms") or cfg.get("intervalMsec") or 0)
    duration_s = int(cfg.get("duration_sec") or cfg.get("per_pass_duration_sec")
                     or cfg.get("per_cell_duration_sec") or 0)

    cid = sc.cell_id(workload, interval_ms, duration_s, replicate)

    notes: list[str] = [f"migrated from v1 ({v1_payload.get('experiment','?')})"]
    for k in ("ram_size_mb", "vm_domain", "self_clean",
              "drain_queue_on_start", "queue_files_drained"):
        if k in cfg:
            notes.append(f"v1.config.{k}={cfg[k]}")
    if isinstance(v1_payload.get("notes"), list):
        for n in v1_payload["notes"]:
            notes.append(f"v1.note: {n}")

    # Producer stats: pull from summary if flat; else aggregate from passes
    ps = sc.ProducerStats()
    if "snapshots_completed" in summary:
        # flat summary (sanity / 2a / 2b cell-level / 2c summary fallback)
        flat = summary
        ps.snapshots_attempted = int(flat.get("snapshots_attempted") or 0)
        ps.snapshots_completed = int(flat.get("snapshots_completed") or 0)
        ps.mean_guest_run_interval_sec = flat.get("mean_guest_run_interval_sec")
        ps.std_guest_run_interval_sec = flat.get("std_guest_run_interval_sec")
        ps.mean_host_snapshot_cycle_sec = flat.get("mean_host_snapshot_cycle_sec")
        ps.std_host_snapshot_cycle_sec = flat.get("std_host_snapshot_cycle_sec")
        ps.mean_suspend_sec = flat.get("mean_suspend_sec")
        ps.mean_pmemsave_sec = flat.get("mean_pmemsave_sec")
        ps.mean_resume_sec = flat.get("mean_resume_sec")
        ps.backpressure_events = int(flat.get("backpressure_events") or 0)
        ps.queue_max_depth = int(flat.get("queue_max_depth") or 0)
        ps.estimated_vm_pause_fraction = flat.get("estimated_vm_pause_fraction")
    elif "flush_on" in summary or "flush_off" in summary:
        # 2c-style: collapse to first pass and note both pass keys
        first_key = next(iter(summary))
        flat = summary[first_key]
        notes.append(f"v1.collapsed_2c_to_pass={first_key}; "
                     f"other_passes={[k for k in summary if k != first_key]}")
        ps.snapshots_attempted = int(flat.get("snapshots_attempted") or 0)
        ps.snapshots_completed = int(flat.get("snapshots_completed") or 0)
        ps.mean_pmemsave_sec = flat.get("mean_pmemsave_sec")
        ps.mean_host_snapshot_cycle_sec = flat.get("mean_host_snapshot_cycle_sec")
        ps.backpressure_events = int(flat.get("backpressure_events") or 0)
        ps.queue_max_depth = int(flat.get("queue_max_depth") or 0)
        ps.estimated_vm_pause_fraction = flat.get("estimated_vm_pause_fraction")

    rm = sc.RunMeta(
        cell_id=cid,
        manifest_id=manifest_id,
        block_id=-1,
        workload=workload,
        interval_ms=interval_ms,
        duration_s=duration_s,
        replicate=replicate,
        git_sha="v1_unknown",
        host_uname="v1_unknown",
        host_kernel="v1_unknown",
        qemu_version="v1_unknown",
        vm_image_sha256="v1_unknown",
        run_started_at=cfg.get("timestamp") or v1_payload.get("timestamp") or _now_iso(),
        run_ended_at=v1_payload.get("timestamp") or _now_iso(),
        exit_status="ok",  # v1 archive assumed valid by reaching here
        retry_count=0,
    )

    record = sc.PerCellRecord(
        schema_version=sc.SCHEMA_VERSION,
        run_meta=rm,
        producer_stats=ps,
        analyzer_outputs=sc.AnalyzerOutputs(
            f1_phase=None,
            cv_workingset=None,
            n_windows=0,
            n_snapshots=ps.snapshots_completed,
        ),
        notes=notes,
    )
    return record.to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Migrate one or more v1 Plan-02 archive JSONs to v2.",
    )
    p.add_argument("jsons", nargs="+", help="paths to v1 JSONs to upgrade")
    p.add_argument("--workload", default=None,
                   help="override the workload field for all input files")
    p.add_argument("--replicate", type=int, default=0,
                   help="replicate index to assign (default 0)")
    p.add_argument("--manifest-id", default="v1_archive",
                   help="manifest_id to stamp on migrated records")
    p.add_argument("--no-backup", action="store_true",
                   help="do not write <file>.v1.bak before overwriting")
    p.add_argument("--out-dir", default=None,
                   help="if set, write upgraded files to this dir instead "
                        "of overwriting (also disables --no-backup logic)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate inputs and report what would be written")
    args = p.parse_args(argv)

    failures = 0
    for src in args.jsons:
        src_path = Path(src).expanduser().resolve()
        if not src_path.is_file():
            print(f"FAIL {src_path}: file not found", file=sys.stderr)
            failures += 1
            continue
        try:
            with src_path.open() as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"FAIL {src_path}: cannot read ({e})", file=sys.stderr)
            failures += 1
            continue

        # Already v2?
        if payload.get("schema_version") == sc.SCHEMA_VERSION:
            print(f"SKIP {src_path}: already v2", file=sys.stderr)
            continue

        upgraded = migrate_one(payload, args.workload, args.replicate,
                               manifest_id=args.manifest_id)
        ok, errors = sc.validate_v2(upgraded)
        if not ok:
            print(f"FAIL {src_path}: post-migration validation: {errors}",
                  file=sys.stderr)
            failures += 1
            continue

        if args.dry_run:
            print(f"DRY  {src_path}: would write v2 "
                  f"(cell_id={upgraded['run_meta']['cell_id']})")
            continue

        if args.out_dir:
            out = Path(args.out_dir).expanduser().resolve() / src_path.name
            sc.write_json_atomic(out, upgraded)
            print(f"OK   {src_path} -> {out}")
        else:
            if not args.no_backup:
                bak = src_path.with_suffix(src_path.suffix + ".v1.bak")
                if not bak.exists():
                    shutil.copy2(src_path, bak)
            sc.write_json_atomic(src_path, upgraded)
            print(f"OK   {src_path} (in-place; backup at .v1.bak)")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
