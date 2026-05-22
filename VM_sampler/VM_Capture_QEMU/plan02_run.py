#!/usr/bin/env python3
"""plan02_run.py -- Plan-02 cell orchestrator.

Owner: SA (Senior Architect) + EN (Engineering Skills).
Reference: docs/tuning_plans/experiment_audit.md Section 11 + Decisions
D-1, D-9, D-11, D-12.

Reads a manifest CSV (built by plan02_manifest.py), executes one cell at
a time, writes a schema-v2 JSON per cell, updates manifest status
atomically, and writes a 30 s observability heartbeat to a status file
during each cell.

Crash-recovery semantics: on restart, this script reads the manifest,
treats any 'running' row as crashed (EN policy), marks it 'failed', and
continues from the next pending row.

Per-cell flow:
  1. Read manifest row, mark status='running' on disk.
  2. Open producer log + start heartbeat thread.
  3. Reuse `run_timing_instrumentation_experiment.py`'s helpers
     (producer launch + JSONL parse + summarize) to actually run the
     producer for `duration_s` seconds.
  4. (Optional) compute analyzer outputs (F1, CV) -- placeholder hooks
     for now; integration with `offline_step_metrics.py` lands in a
     follow-up patch since the analyzer expects post-processed inputs.
  5. Write per-cell JSON v2 atomically.
  6. Mark manifest row status='ok' or 'failed' atomically.
  7. Warmup cells skip step (4)-(5); their JSON is written but tagged
     and ignored by downstream analysis.

This is the launch-ready orchestrator the audit's Section 11 calls for.
The analyzer integration is intentionally left as a clear seam so the
ML/DS owners can plug their post-processing without touching capture
logic.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Local imports
import plan02_manifest as mf
import plan02_schema as sc
import run_timing_instrumentation_experiment as e1


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_qemu_upc.json"
DEFAULT_PRODUCER = SCRIPT_DIR / "capture_producer_qemu_pmemsave.sh"
HEARTBEAT_INTERVAL_SEC = 30


def log(msg: str) -> None:
    print(f"[plan02] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Heartbeat writer (EN observability)
# ---------------------------------------------------------------------------

class HeartbeatThread(threading.Thread):
    """Writes a small status JSON every HEARTBEAT_INTERVAL_SEC seconds.

    Lets the PM / operator see 'we're at snap 18 of 30 of cell 47/90'
    without ssh-ing into the producer. Overhead < 0.5 % wall-clock.
    """

    def __init__(self, status_path: Path, cell: mf.ManifestRow,
                 jsonl_path: Path, total_cells: int, cell_index: int):
        super().__init__(daemon=True)
        self.status_path = status_path
        self.cell = cell
        self.jsonl_path = jsonl_path
        self.total_cells = total_cells
        self.cell_index = cell_index
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _snap_count(self) -> int:
        if not self.jsonl_path.exists():
            return 0
        try:
            with self.jsonl_path.open() as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def _disk_free_gib(self) -> float:
        try:
            st = os.statvfs(self.jsonl_path.parent)
            return st.f_bavail * st.f_frsize / (1 << 30)
        except OSError:
            return -1.0

    def _meminfo(self) -> dict[str, int]:
        out = {"MemAvailable_kB": -1, "Dirty_kB": -1}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) != 2:
                        continue
                    key = parts[0].strip()
                    if key in ("MemAvailable", "Dirty"):
                        val = parts[1].strip().split()[0]
                        out[f"{key}_kB"] = int(val)
        except (OSError, ValueError):
            pass
        return out

    def _write_once(self) -> None:
        snap_count = self._snap_count()
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "cell_id": self.cell.cell_id,
            "cell_index": self.cell_index,
            "total_cells": self.total_cells,
            "workload": self.cell.workload,
            "interval_ms": self.cell.interval_ms,
            "duration_s": self.cell.duration_s,
            "replicate": self.cell.replicate,
            "is_warmup": self.cell.is_warmup,
            "snapshots_written": snap_count,
            "disk_free_gib": round(self._disk_free_gib(), 2),
            "meminfo": self._meminfo(),
        }
        try:
            sc.write_json_atomic(self.status_path, payload)
        except OSError:
            pass  # heartbeat best-effort

    def run(self) -> None:
        # Write one immediately so the file exists even on short cells.
        self._write_once()
        while not self._stop.wait(HEARTBEAT_INTERVAL_SEC):
            self._write_once()
        # final write captures last state
        self._write_once()


# ---------------------------------------------------------------------------
# Per-cell execution
# ---------------------------------------------------------------------------

def execute_cell(
    cell: mf.ManifestRow,
    workdir: Path,
    config_path: Path,
    producer_script: Path,
    host_meta: dict,
    cell_index: int,
    total_cells: int,
    virsh_uri: str = "qemu:///system",
    grace_stop_seconds: int = 10,
    no_vm_start: bool = False,
) -> tuple[str, sc.PerCellRecord, list[str]]:
    """Run one cell. Returns (status, record, notes).

    status in {"ok", "failed"}.
    """
    notes: list[str] = []
    cell_workdir = workdir / cell.cell_id
    cell_workdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = cell_workdir / "snapshot_timings.jsonl"
    producer_log = cell_workdir / "producer.log"
    status_path = cell_workdir / "heartbeat.json"

    # Per-cell config: override intervalMsec in a private copy.
    base_cfg = e1.load_config(config_path)
    cfg_path, eff_cfg = e1.write_overridden_config(
        base_cfg, cell_workdir,
        interval_ms=cell.interval_ms,
        ram_mb=None,  # use config's default
    )
    vm_domain = eff_cfg.get("domain", "")

    # Pre-flight: drain stale queue + enable producer self-clean
    # (Plans 1c + 1b proven necessary in R3/R4).
    os.environ["TIMING_SELF_CLEAN"] = "1"
    queue_dir = Path(eff_cfg.get("queueDir", "/tmp/queue_dir"))
    try:
        from run_exp2a_consumer_isolation import drain_queue as _drain_queue
        drained = _drain_queue(queue_dir)
        notes.append(f"drained {drained} stale queue file(s)")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"queue drain warning: {exc}")

    # VM up
    if not no_vm_start:
        state = e1.virsh_start_if_needed(virsh_uri, vm_domain, dry_run=False)
        if state != "running":
            notes.append(f"VM not running after start attempt: {state!r}")

    run_started_at = datetime.now(timezone.utc).isoformat()
    run_start_epoch = time.time()

    # Start heartbeat
    heartbeat = HeartbeatThread(status_path, cell, jsonl_path,
                                total_cells, cell_index)
    heartbeat.start()

    # Producer
    producer_proc = None
    try:
        producer_proc = e1.start_producer(
            producer_script, cfg_path, jsonl_path, producer_log,
        )
        try:
            time.sleep(cell.duration_s)
        except KeyboardInterrupt:
            notes.append("interrupted by user during measurement window")
            raise
    finally:
        if producer_proc is not None:
            e1.stop_producer(producer_proc, grace_stop_seconds)
        e1.resume_vm_if_paused(virsh_uri, vm_domain)
        heartbeat.stop()
        heartbeat.join(timeout=5)

    run_ended_at = datetime.now(timezone.utc).isoformat()

    # Parse JSONL into producer_stats
    snaps, bps = e1.parse_jsonl(jsonl_path)
    records = []
    for i, this in enumerate(snaps):
        nxt = snaps[i + 1] if i + 1 < len(snaps) else None
        records.append(e1.build_snapshot_record(i, this, nxt))
    summary = e1.summarize(records, bps)

    producer_stats = sc.ProducerStats(
        snapshots_attempted=int(summary.get("snapshots_attempted") or 0),
        snapshots_completed=int(summary.get("snapshots_completed") or 0),
        mean_guest_run_interval_sec=summary.get("mean_guest_run_interval_sec"),
        std_guest_run_interval_sec=summary.get("std_guest_run_interval_sec"),
        mean_host_snapshot_cycle_sec=summary.get("mean_host_snapshot_cycle_sec"),
        std_host_snapshot_cycle_sec=summary.get("std_host_snapshot_cycle_sec"),
        mean_suspend_sec=summary.get("mean_suspend_sec"),
        mean_pmemsave_sec=summary.get("mean_pmemsave_sec"),
        mean_resume_sec=summary.get("mean_resume_sec"),
        backpressure_events=int(summary.get("backpressure_events") or 0),
        queue_max_depth=int(summary.get("queue_max_depth") or 0),
        estimated_vm_pause_fraction=summary.get("estimated_vm_pause_fraction"),
    )

    # Analyzer outputs: placeholder; ML/DS owners plug in
    # offline_step_metrics integration in follow-up.
    analyzer = sc.AnalyzerOutputs(
        f1_phase=None,
        cv_workingset=None,
        n_windows=0,
        n_snapshots=producer_stats.snapshots_completed,
    )

    # Clean up dump files this cell produced (TIMING_SELF_CLEAN already
    # removes them in-process, but tail-cleanup catches the final one).
    try:
        image_dir = Path(eff_cfg.get("imageDir", "/var/lib/libvirt/qemu/dump"))
        removed = e1.cleanup_run_dumps(image_dir, run_start_epoch, False)
        notes.append(f"cleanup: removed {removed} dump files")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"cleanup warning: {exc}")

    # Status decision
    status = "ok"
    if producer_stats.snapshots_completed == 0:
        status = "failed"
        notes.append("FAIL: 0 snapshots completed; producer likely crashed")
    elif producer_stats.backpressure_events > max(
        1, int(0.01 * producer_stats.snapshots_attempted)
    ):
        notes.append(
            f"WARN: high backpressure ({producer_stats.backpressure_events} "
            f"of {producer_stats.snapshots_attempted} attempts)"
        )

    run_meta = sc.RunMeta(
        cell_id=cell.cell_id,
        manifest_id=cell.manifest_id,
        block_id=cell.block_id,
        workload=cell.workload,
        interval_ms=cell.interval_ms,
        duration_s=cell.duration_s,
        replicate=cell.replicate,
        git_sha=host_meta.get("git_sha", "unknown"),
        host_uname=host_meta.get("host_uname", "unknown"),
        host_kernel=host_meta.get("host_kernel", "unknown"),
        qemu_version=host_meta.get("qemu_version", "unknown"),
        vm_image_sha256=host_meta.get("vm_image_sha256", "unknown"),
        run_started_at=run_started_at,
        run_ended_at=run_ended_at,
        exit_status=status,
        retry_count=cell.retry_count,
    )

    record = sc.PerCellRecord(
        schema_version=sc.SCHEMA_VERSION,
        run_meta=run_meta,
        producer_stats=producer_stats,
        analyzer_outputs=analyzer,
        notes=notes,
    )
    return status, record, notes


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------

def run_session(
    manifest_path: Path,
    config_path: Path,
    producer_script: Path,
    output_dir: Path,
    workdir: Path,
    virsh_uri: str = "qemu:///system",
    no_vm_start: bool = False,
    grace_stop_seconds: int = 10,
    max_cells: int | None = None,
    dry_run: bool = False,
) -> int:
    """Execute pending cells from the manifest until done or max_cells hit.

    Exits cleanly on Ctrl-C: marks current cell 'failed' + saves manifest.
    """
    rows = mf.load(manifest_path)
    if not rows:
        log(f"manifest empty or missing: {manifest_path}")
        return 2

    # Crash-recovery: any 'running' row is mid-flight crash from a prior session.
    fixed = mf.crashed_running_to_failed(rows)
    if fixed:
        log(f"WARNING: {fixed} 'running' rows auto-marked 'failed' on restart")
        mf.save(manifest_path, rows)

    # Session sentinel: write host state once per session
    host_meta = sc.collect_host_meta(repo_root=SCRIPT_DIR.parent)
    sentinel_path = output_dir / "session_sentinel.json"
    sc.write_json_atomic(sentinel_path, {
        "session_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "manifest_path": str(manifest_path),
        "host_meta": host_meta,
        "summary_at_start": mf.summarize(rows),
    })
    log(f"session sentinel: {sentinel_path}")

    summary = mf.summarize(rows)
    log(f"manifest summary at start: {summary}")

    if dry_run:
        log("[dry-run] would execute pending cells; exiting")
        return 0

    executed = 0
    aborted = False
    while True:
        rows = mf.load(manifest_path)
        cell = mf.next_pending(rows)
        if cell is None:
            log("no more pending cells; session complete")
            break
        if max_cells is not None and executed >= max_cells:
            log(f"hit --max-cells={max_cells}; stopping early")
            break

        # Determine cell_index for heartbeat
        cell_index = 1 + sum(
            1 for r in rows if r.status in {"ok", "failed", "skipped"}
        )
        total_cells = len(rows)

        log(f"[{cell_index}/{total_cells}] running cell {cell.cell_id} "
            f"(workload={cell.workload}, iv={cell.interval_ms}, "
            f"d={cell.duration_s}s, rep={cell.replicate}, "
            f"warm={cell.is_warmup})")

        mf.set_status_on_disk(manifest_path, cell.cell_id, "running")

        try:
            status, record, notes = execute_cell(
                cell, workdir, config_path, producer_script, host_meta,
                cell_index=cell_index, total_cells=total_cells,
                virsh_uri=virsh_uri,
                grace_stop_seconds=grace_stop_seconds,
                no_vm_start=no_vm_start,
            )
        except KeyboardInterrupt:
            log("KeyboardInterrupt; marking current cell failed and exiting")
            mf.set_status_on_disk(manifest_path, cell.cell_id, "failed",
                                  notes_append="interrupted by operator")
            aborted = True
            break
        except Exception as exc:  # noqa: BLE001
            log(f"cell crashed: {exc}")
            mf.set_status_on_disk(manifest_path, cell.cell_id, "failed",
                                  notes_append=f"orchestrator exception: {exc}",
                                  bump_retry=True)
            executed += 1
            continue

        # Persist per-cell JSON (skip for warmups -- write but tag, never index)
        out_path = Path(cell.expected_path) if cell.expected_path else (
            output_dir / f"cell_{cell.cell_id}.json"
        )
        if cell.is_warmup:
            record.notes.insert(0, "WARMUP CELL -- discarded by analysis")
        sc.write_json_atomic(out_path, record)

        # Validate v2 before declaring success
        ok, errors = sc.validate_v2(record.to_dict())
        if not ok:
            log(f"schema validation FAILED for {cell.cell_id}: {errors}")
            status = "failed"
            notes_append = f"schema v2 errors: {'; '.join(errors[:2])}"
        else:
            notes_append = " | ".join(notes) if notes else None

        mf.set_status_on_disk(manifest_path, cell.cell_id, status,
                              notes_append=notes_append)
        executed += 1
        log(f"[{cell_index}/{total_cells}] {cell.cell_id} -> {status} "
            f"(snapshots={record.producer_stats.snapshots_completed})")

    final_summary = mf.summarize(mf.load(manifest_path))
    log(f"session done: {final_summary}")
    return 1 if aborted else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plan02_run.py",
        description=(
            "Plan-02 cell orchestrator. Reads a manifest CSV and executes "
            "pending cells. Crash-safe: re-running resumes from where it "
            "stopped. See docs/tuning_plans/experiment_audit.md Section 11."
        ),
    )
    p.add_argument("--manifest", required=True, help="path to manifest CSV")
    p.add_argument("--output-dir", required=True,
                   help="directory for per-cell JSONs and session sentinel")
    p.add_argument("--workdir", default=None,
                   help="scratch directory for per-cell producer state "
                        "(default: <output-dir>/work)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--producer-script", default=str(DEFAULT_PRODUCER))
    p.add_argument("--virsh-uri", default="qemu:///system")
    p.add_argument("--no-vm-start", action="store_true")
    p.add_argument("--grace-stop-seconds", type=int, default=10)
    p.add_argument("--max-cells", type=int, default=None,
                   help="stop after this many cells (default: run all pending)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate manifest + write session sentinel; no capture")
    return p


def _main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = (
        Path(args.workdir).expanduser().resolve()
        if args.workdir else (output_dir / "work")
    )
    workdir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).expanduser().resolve()
    producer_script = Path(args.producer_script).expanduser().resolve()
    if not args.dry_run:
        if not config_path.is_file():
            log(f"ERROR: config not found: {config_path}")
            return 2
        if not producer_script.is_file():
            log(f"ERROR: producer not found: {producer_script}")
            return 2

    return run_session(
        manifest_path=manifest_path,
        config_path=config_path,
        producer_script=producer_script,
        output_dir=output_dir,
        workdir=workdir,
        virsh_uri=args.virsh_uri,
        no_vm_start=args.no_vm_start,
        grace_stop_seconds=args.grace_stop_seconds,
        max_cells=args.max_cells,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(_main())
