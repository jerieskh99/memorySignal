#!/usr/bin/env python3
"""plan02_schema.py -- schema v2 for Plan 02 per-cell JSON artifacts.

Owner: DE (Senior Data Engineer).
Reference: docs/tuning_plans/experiment_audit.md Decisions D-2.

This module defines the canonical schema v2 used by every Plan-02-era
capture cell. Every per-cell JSON written by the orchestrator MUST
validate against this schema. Old R1-R5 artifacts are v1; use
migrate_schema_v1_to_v2.py to upgrade them.

Schema v2 design goals:
  - Reproducibility: capture host kernel, qemu version, git sha,
    VM image hash. Re-running a year later must be possible.
  - Auditability: cell_id is a deterministic hash of (workload, iv,
    duration, replicate). Re-running an already-completed cell is
    a no-op.
  - Versioning: explicit schema_version field. v3 will add fields,
    never break v2 readers.
  - Decoupling: producer_stats + analyzer_outputs are independent
    subtrees. Analyzer can re-run without re-capturing.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Cell identity
# ---------------------------------------------------------------------------

def cell_id(workload: str, interval_ms: int, duration_s: int,
            replicate: int) -> str:
    """Deterministic SHA1 prefix identifying a cell.

    Re-running the same cell yields the same cell_id, so the manifest
    can detect and skip duplicates.
    """
    payload = f"{workload}|{interval_ms}|{duration_s}|{replicate}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------

def _git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True, text=True, check=False, timeout=5,
        )
        sha = (out.stdout or "").strip()
        return sha if sha else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _qemu_version() -> str:
    try:
        out = subprocess.run(
            ["qemu-system-x86_64", "--version"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        first = (out.stdout or "").splitlines()[:1]
        return first[0].strip() if first else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _vm_image_sha256(image_path: Path | None) -> str:
    if image_path is None or not image_path.is_file():
        return "unknown"
    try:
        h = hashlib.sha256()
        with image_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "unknown"


def collect_host_meta(repo_root: Path | None = None,
                      vm_image_path: Path | None = None) -> dict[str, Any]:
    """One-shot host snapshot for the session sentinel + per-cell JSONs.

    Cheap (~50 ms) except for vm_image_sha256 (~1-2 s for a 1 GiB image).
    Cache the result and reuse across cells in the same session.
    """
    uname = platform.uname()
    return {
        "hostname": socket.gethostname(),
        "host_uname": f"{uname.system} {uname.release} {uname.machine}",
        "host_kernel": uname.release,
        "python_version": sys.version.split()[0],
        "qemu_version": _qemu_version(),
        "git_sha": _git_sha(repo_root),
        "vm_image_sha256": _vm_image_sha256(vm_image_path),
        "session_started_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    cell_id: str
    manifest_id: str
    block_id: int
    workload: str
    interval_ms: int
    duration_s: int
    replicate: int
    git_sha: str
    host_uname: str
    host_kernel: str
    qemu_version: str
    vm_image_sha256: str
    run_started_at: str
    run_ended_at: str
    exit_status: str  # ok | failed | retried | skipped
    retry_count: int = 0
    # Day-7 additions (D-22): workload-launching artifacts
    workload_command: str | None = None
    ssh_target: str | None = None
    workload_stderr_path: str | None = None
    workload_exit_status: int | None = None
    keep_dumps: bool = False


@dataclass
class ProducerStats:
    snapshots_attempted: int = 0
    snapshots_completed: int = 0
    mean_guest_run_interval_sec: float | None = None
    std_guest_run_interval_sec: float | None = None
    mean_host_snapshot_cycle_sec: float | None = None
    std_host_snapshot_cycle_sec: float | None = None
    mean_suspend_sec: float | None = None
    mean_pmemsave_sec: float | None = None
    mean_resume_sec: float | None = None
    backpressure_events: int = 0
    queue_max_depth: int = 0
    estimated_vm_pause_fraction: float | None = None


@dataclass
class AnalyzerOutputs:
    f1_phase: float | None = None       # ransom_batched + similar phasic workloads
    cv_workingset: float | None = None  # workingset_sweep + similar steady workloads
    n_windows: int = 0
    n_snapshots: int = 0


@dataclass
class PerCellRecord:
    schema_version: int = SCHEMA_VERSION
    run_meta: RunMeta = None  # type: ignore[assignment]
    producer_stats: ProducerStats = field(default_factory=ProducerStats)
    analyzer_outputs: AnalyzerOutputs = field(default_factory=AnalyzerOutputs)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Atomic JSON write (temp + rename)
# ---------------------------------------------------------------------------

def write_json_atomic(path: Path, payload: dict[str, Any] | PerCellRecord) -> None:
    """Write JSON via temp+rename so concurrent readers never see half a file.

    On POSIX, rename(2) on the same filesystem is atomic. Caller is
    responsible for parent dir existing.
    """
    if isinstance(payload, PerCellRecord):
        payload = payload.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.flush()
        try:
            import os
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_RUN_META = {
    "cell_id", "manifest_id", "block_id", "workload", "interval_ms",
    "duration_s", "replicate", "git_sha", "run_started_at",
    "run_ended_at", "exit_status",
}


def validate_v2(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (ok, errors). Used by the orchestrator before persisting and
    by the migrator after upgrading.
    """
    errors: list[str] = []
    sv = payload.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"schema_version expected {SCHEMA_VERSION}, got {sv}")
    rm = payload.get("run_meta") or {}
    for key in REQUIRED_RUN_META:
        if key not in rm:
            errors.append(f"run_meta missing required key: {key}")
    if "producer_stats" not in payload:
        errors.append("missing producer_stats subtree")
    if "analyzer_outputs" not in payload:
        errors.append("missing analyzer_outputs subtree")
    return (not errors), errors


# ---------------------------------------------------------------------------
# CLI for ad-hoc validation
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Validate a Plan-02 v2 JSON. Exits 0 if valid, 1 if not."
    )
    p.add_argument("jsons", nargs="+", help="paths to JSONs to validate")
    args = p.parse_args(argv)
    bad = 0
    for path in args.jsons:
        try:
            with open(path) as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"FAIL {path}: cannot read ({e})", file=sys.stderr)
            bad += 1
            continue
        ok, errors = validate_v2(obj)
        if ok:
            print(f"OK   {path}")
        else:
            print(f"FAIL {path}:")
            for err in errors:
                print(f"     - {err}")
            bad += 1
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
