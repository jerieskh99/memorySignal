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
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Local imports
import plan02_manifest as mf
import plan02_schema as sc
import run_timing_instrumentation_experiment as e1
try:
    import plan02_metrics_per_cell as pmc
    _HAS_PMC = True
except ImportError:  # pragma: no cover
    _HAS_PMC = False


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_qemu_upc.json"
DEFAULT_PRODUCER = SCRIPT_DIR / "capture_producer_qemu_pmemsave.sh"
HEARTBEAT_INTERVAL_SEC = 30

# Pause-fraction estimates from the Step-1 pilot. Used by Bug-M check
# (D-32) to compute the expected snapshot count for a cell, which becomes
# the denominator of snapshot_completion_ratio. Conservative defaults
# (interp at intermediate iv values).
PAUSE_FRACTION_BY_IV = {
    100:  0.924,
    250:  0.847,
    500:  0.741,
    1000: 0.593,
    2000: 0.424,
}


def estimated_pause_fraction(iv_ms: int) -> float:
    """Linear interp between known iv values; clamp at the endpoints."""
    known = sorted(PAUSE_FRACTION_BY_IV)
    if iv_ms <= known[0]:
        return PAUSE_FRACTION_BY_IV[known[0]]
    if iv_ms >= known[-1]:
        return PAUSE_FRACTION_BY_IV[known[-1]]
    for lo, hi in zip(known, known[1:]):
        if lo <= iv_ms <= hi:
            t = (iv_ms - lo) / (hi - lo)
            return PAUSE_FRACTION_BY_IV[lo] * (1 - t) + PAUSE_FRACTION_BY_IV[hi] * t
    return 0.7  # fallback


def _classify_workload(workload: str) -> str:
    """Map workload name to phasic / steady / unknown for D-51 metric
    selection. Phasic workloads expect phase-boundary detection (F1);
    steady workloads expect active-page-fraction CV across windows.
    """
    w = (workload or "").lower()
    phasic_keys = ("ransom_batched", "ransom_seq", "ransom_selective",
                   "ransom_slowburn", "scanner_metadata",
                   "phase_boundary", "phasic")
    steady_keys = ("workingset_sweep", "writemag_sweep",
                   "rmw_intensity", "pagefault_density",
                   "mmap_traversal", "hashtable_intensive",
                   "compress_streaming", "compress_gzip",
                   "decompress_gzip", "json_parse", "sqlite_oltp",
                   "sqlite_analytical", "steady")
    if any(k in w for k in phasic_keys):
        return "phasic"
    if any(k in w for k in steady_keys):
        return "steady"
    return "unknown"


def asdict_safe(obj: object) -> dict:
    """Like dataclasses.asdict but degrades to repr() for non-serializable
    fields. Used to persist plan02_metrics_per_cell.PerCellMetrics as JSON.
    """
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj):
        return asdict(obj)
    return {"value": repr(obj)}


def expected_snapshots(duration_s: int, iv_ms: int) -> int:
    """How many snaps a cell *should* produce, given duration and iv,
    using the Step-1 pause-fraction sweep as a prior. Used only for
    quality flagging, never for acceptance gating per se.
    """
    if iv_ms <= 0:
        return 0
    pf = estimated_pause_fraction(iv_ms)
    guest_time_s = duration_s * (1 - pf)
    iv_s = iv_ms / 1000.0 + 0.025  # +25 ms producer overhead from Step 1
    return max(1, int(guest_time_s / iv_s))


# ---------------------------------------------------------------------------
# Disk-space guardrails (Day-9 debug-team fixes)
# ---------------------------------------------------------------------------

def _disk_free_gib(path: Path) -> float:
    """Return free GiB on the filesystem holding ``path`` (or its parent
    if path doesn't exist). Returns -1.0 on failure."""
    target = path if path.exists() else path.parent
    try:
        st = os.statvfs(target)
        return st.f_bavail * st.f_frsize / (1 << 30)
    except OSError:
        return -1.0


def _count_stale_dumps(image_dir: Path) -> int:
    if not image_dir.is_dir():
        return 0
    try:
        return sum(1 for _ in image_dir.glob("memory_dump-*.raw"))
    except OSError:
        return 0


def session_preflight_disk(image_dir: Path, ram_mb: int,
                            min_headroom_dumps: int = 5,
                            purge_stale: bool = False) -> dict:
    """Run-session-scope dump-dir health check.

    Returns dict with disk_free_gib, stale_dumps, purged_count, ok (bool),
    reason (str | None). Designed to be a no-op when everything is fine.
    """
    info: dict = {
        "image_dir": str(image_dir),
        "disk_free_gib_before": _disk_free_gib(image_dir),
        "stale_dumps_before": _count_stale_dumps(image_dir),
        "purged_count": 0,
        "disk_free_gib_after": None,
        "stale_dumps_after": None,
        "ok": True,
        "reason": None,
    }
    if purge_stale and info["stale_dumps_before"] > 0:
        try:
            info["purged_count"] = e1.purge_all_dumps(image_dir, use_sudo=True)
        except Exception as exc:  # noqa: BLE001
            info["reason"] = f"purge_stale failed: {exc}"
            info["ok"] = False
            return info
    info["disk_free_gib_after"] = _disk_free_gib(image_dir)
    info["stale_dumps_after"] = _count_stale_dumps(image_dir)
    needed_gib = (min_headroom_dumps * ram_mb) / 1024.0
    if info["disk_free_gib_after"] >= 0 and info["disk_free_gib_after"] < needed_gib:
        info["ok"] = False
        info["reason"] = (
            f"insufficient disk: {info['disk_free_gib_after']:.1f} GiB free "
            f"< {needed_gib:.1f} GiB required for {min_headroom_dumps} dumps "
            f"at ram_mb={ram_mb}"
        )
    return info


def pre_cell_disk_check(image_dir: Path, ram_mb: int,
                        min_headroom_dumps: int = 5) -> tuple[bool, dict]:
    """Per-cell preflight: require at least min_headroom_dumps × ram_mb GiB
    free OR fail the cell loud (avoid generating 1-snap garbage cells)."""
    free = _disk_free_gib(image_dir)
    stale = _count_stale_dumps(image_dir)
    needed_gib = (min_headroom_dumps * ram_mb) / 1024.0
    info = {
        "disk_free_gib": free,
        "stale_dumps": stale,
        "needed_gib_for_headroom": needed_gib,
    }
    if free < 0:
        return True, info  # statvfs unavailable; don't block
    return (free >= needed_gib), info


def wait_for_apf_helpers(ack_dir: Path, n_pairs_expected: int,
                          timeout_s: float = 30.0,
                          poll_interval_s: float = 0.2) -> dict:
    """B+3.1 Δ-1 cell-end barrier.

    Poll `ack_dir` until n_pairs_expected ack files exist, OR timeout.
    Returns dict with: n_ok, n_failed, n_observed, gap_seqs, timed_out.
    Caller writes the final sentinel based on this result.
    """
    deadline = time.monotonic() + timeout_s
    info = {
        "n_pairs_expected": int(max(0, n_pairs_expected)),
        "n_ok": 0,
        "n_failed": 0,
        "n_observed": 0,
        "gap_seqs": [],
        "timed_out": False,
    }
    if n_pairs_expected <= 0:
        return info
    last_observed = -1
    while time.monotonic() < deadline:
        try:
            ack_files = list(ack_dir.glob("seq_*.apf_done"))
        except OSError:
            ack_files = []
        if len(ack_files) != last_observed:
            last_observed = len(ack_files)
        if len(ack_files) >= n_pairs_expected:
            break
        time.sleep(poll_interval_s)
    # Tally
    seen: set[int] = set()
    n_ok = 0
    n_failed = 0
    try:
        for path in ack_dir.glob("seq_*.apf_done"):
            try:
                with path.open() as f:
                    rec = json.load(f)
                seq = int(rec.get("seq", -1))
                ec = int(rec.get("exit_code", -1))
                if seq >= 0:
                    seen.add(seq)
                if ec == 0:
                    n_ok += 1
                else:
                    n_failed += 1
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    except OSError:
        pass
    expected_seqs = set(range(n_pairs_expected))
    info["n_observed"] = len(seen)
    info["n_ok"] = n_ok
    info["n_failed"] = n_failed
    info["gap_seqs"] = sorted(expected_seqs - seen)
    info["timed_out"] = (len(seen) < n_pairs_expected)
    return info


def write_apf_final_sentinel(apf_jsonl: Path, barrier_info: dict) -> None:
    """B+3.1 Δ-3: append the final sentinel line so a downstream reader
    can prove the trajectory file is complete.
    """
    sentinel = {
        "final": True,
        "n_pairs_expected": int(barrier_info["n_pairs_expected"]),
        "n_ok": int(barrier_info["n_ok"]),
        "n_failed": int(barrier_info["n_failed"]),
        "gap_seqs": list(barrier_info["gap_seqs"]),
        "timed_out": bool(barrier_info.get("timed_out", False)),
    }
    line = json.dumps(sentinel, separators=(",", ":")) + "\n"
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    fd = os.open(str(apf_jsonl), flags, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def scan_producer_log(producer_log: Path, max_errors: int = 5) -> list[str]:
    """Read producer.log tail and return up to `max_errors` distinct
    error-ish lines. Used post-cell to surface pmemsave / virsh failures
    that the orchestrator otherwise misses.
    """
    if not producer_log.is_file():
        return []
    keywords = ("error", "fail", "denied", "no space", "cannot", "refused")
    seen: list[str] = []
    try:
        text = producer_log.read_text(errors="replace")
    except OSError:
        return []
    for line in text.splitlines()[-500:]:  # tail only
        lower = line.lower()
        if any(k in lower for k in keywords):
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.append(stripped)
                if len(seen) >= max_errors:
                    break
    return seen


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

    # Day-9 debug-team per-cell disk preflight: refuse to launch the cell
    # when free disk cannot hold a small headroom of dumps. This is the
    # cheap guard against the "1-snap-then-fail" pattern caused by disk
    # filling mid-run.
    image_dir_for_cell = Path(eff_cfg.get("imageDir", "/var/lib/libvirt/qemu/dump"))
    ram_mb_for_cell = int(eff_cfg.get("ramSizeMb", 1024))
    ok_disk, disk_info = pre_cell_disk_check(image_dir_for_cell,
                                              ram_mb_for_cell, 5)
    notes.append(f"pre-cell disk: free={disk_info['disk_free_gib']:.1f} GiB "
                 f"stale_dumps={disk_info['stale_dumps']}")
    if not ok_disk:
        notes.append(
            f"FAIL preflight: disk_free={disk_info['disk_free_gib']:.1f} GiB "
            f"< headroom={disk_info['needed_gib_for_headroom']:.1f} GiB. "
            f"Cell skipped before any VM operation. Free disk and retry."
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        run_meta_early = sc.RunMeta(
            cell_id=cell.cell_id, manifest_id=cell.manifest_id,
            block_id=cell.block_id, workload=cell.workload,
            interval_ms=cell.interval_ms, duration_s=cell.duration_s,
            replicate=cell.replicate,
            git_sha=host_meta.get("git_sha", "unknown"),
            host_uname=host_meta.get("host_uname", "unknown"),
            host_kernel=host_meta.get("host_kernel", "unknown"),
            qemu_version=host_meta.get("qemu_version", "unknown"),
            vm_image_sha256=host_meta.get("vm_image_sha256", "unknown"),
            run_started_at=now_iso,
            run_ended_at=now_iso,
            exit_status="failed",
            retry_count=cell.retry_count,
            workload_command=cell.workload_command or None,
            ssh_target=cell.ssh_target or None,
            workload_stderr_path=None,
            workload_exit_status=None,
            keep_dumps=cell.keep_dumps,
        )
        record_early = sc.PerCellRecord(
            schema_version=sc.SCHEMA_VERSION,
            run_meta=run_meta_early,
            producer_stats=sc.ProducerStats(),
            analyzer_outputs=sc.AnalyzerOutputs(),
            notes=notes,
        )
        return "failed", record_early, notes

    # Pre-flight: drain stale queue + enable producer self-clean
    # (Plans 1c + 1b proven necessary in R3/R4).
    # D-20: if the cell requests keep_dumps (because an offline analyzer
    # needs dump content), force TIMING_SELF_CLEAN OFF so the producer
    # leaves the dumps on disk. The tail cleanup below also honors keep_dumps.
    # B+3.1 (Δ-1): when keep_dumps is set, enable the streaming APF helper
    # so the producer launches plan02_apf_helper.py per pair. Disk peak
    # stays at ~2-3 GiB regardless of cell length.
    apf_jsonl: Path | None = None
    apf_ack_dir: Path | None = None
    apf_helper_log: Path | None = None
    if cell.keep_dumps:
        os.environ.pop("TIMING_SELF_CLEAN", None)
        apf_jsonl = cell_workdir / "apf_trajectory.jsonl"
        apf_ack_dir = cell_workdir / "apf_acks"
        apf_helper_log = cell_workdir / "apf_helper.log"
        apf_jsonl.parent.mkdir(parents=True, exist_ok=True)
        apf_ack_dir.mkdir(parents=True, exist_ok=True)
        # Truncate any prior trajectory / acks for idempotent re-runs.
        try:
            apf_jsonl.unlink()
        except FileNotFoundError:
            pass
        for stale_ack in apf_ack_dir.glob("seq_*.apf_done"):
            try:
                stale_ack.unlink()
            except FileNotFoundError:
                pass
        os.environ["TIMING_APF_STREAM"] = "1"
        os.environ["TIMING_APF_JSONL"] = str(apf_jsonl)
        os.environ["TIMING_APF_ACK_DIR"] = str(apf_ack_dir)
        os.environ["TIMING_APF_HELPER_LOG"] = str(apf_helper_log)
        notes.append("B+3.1: TIMING_APF_STREAM enabled · streaming APF helper active")
    else:
        os.environ["TIMING_SELF_CLEAN"] = "1"
        os.environ.pop("TIMING_APF_STREAM", None)
        os.environ.pop("TIMING_APF_JSONL", None)
        os.environ.pop("TIMING_APF_ACK_DIR", None)
        os.environ.pop("TIMING_APF_HELPER_LOG", None)
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

    # Workload (D-20): if cell carries a workload_command, SSH-launch it
    # in parallel with the producer. Stderr captured to per-cell file.
    # Stopped before producer (D-20 sequencing).
    workload_proc = None
    workload_stderr_path = cell_workdir / "workload_stderr.log"
    workload_log = cell_workdir / "workload.log"
    workload_exit: int | None = None
    workload_skipped_reason: str | None = None
    if cell.workload_command and cell.ssh_target:
        ssh_target = cell.ssh_target
        ssh_key = os.environ.get("SSH_KEY", "")
        ssh_opts = os.environ.get("SSH_OPTS", "")
        # Append --phase-markers if not already present (so mp_phase_boundary_
        # inference style detectors can pick up ground-truth events).
        wcmd = cell.workload_command
        if "--phase-markers" not in wcmd:
            wcmd = wcmd + " --phase-markers"
        # SSH timeout configured to outlive the cell duration
        ok = e1.wait_for_ssh(ssh_target, ssh_key, ssh_opts,
                             timeout_s=min(30, cell.duration_s))
        if not ok:
            notes.append(f"SSH not reachable at {ssh_target}; "
                         "workload not launched; producer-only this cell")
            workload_skipped_reason = "ssh_unreachable"
        else:
            # Bug-J runtime probe (D-29): verify the binary exists on the VM
            # before launching, so placeholder paths or typos fail fast at
            # cell granularity rather than producing fake-ok cells.
            binary_path = wcmd.split()[0]
            probe_cmd = e1.build_ssh_cmd(ssh_target, ssh_key, ssh_opts,
                                         f"test -x {binary_path}")
            try:
                probe = subprocess.run(
                    probe_cmd, capture_output=True, text=True,
                    timeout=15, check=False,
                )
                binary_ok = (probe.returncode == 0)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                binary_ok = False
            if not binary_ok:
                notes.append(
                    f"workload binary not executable on VM: {binary_path!r}. "
                    f"Verify path + permissions on remote. "
                    f"Cell marked failed; producer skipped."
                )
                workload_skipped_reason = "binary_not_executable"
                # Skip the cell entirely: stop heartbeat + return failed.
                heartbeat.stop()
                heartbeat.join(timeout=5)
                run_ended_at_early = datetime.now(timezone.utc).isoformat()
                run_meta_early = sc.RunMeta(
                    cell_id=cell.cell_id, manifest_id=cell.manifest_id,
                    block_id=cell.block_id, workload=cell.workload,
                    interval_ms=cell.interval_ms, duration_s=cell.duration_s,
                    replicate=cell.replicate,
                    git_sha=host_meta.get("git_sha", "unknown"),
                    host_uname=host_meta.get("host_uname", "unknown"),
                    host_kernel=host_meta.get("host_kernel", "unknown"),
                    qemu_version=host_meta.get("qemu_version", "unknown"),
                    vm_image_sha256=host_meta.get("vm_image_sha256", "unknown"),
                    run_started_at=run_started_at,
                    run_ended_at=run_ended_at_early,
                    exit_status="failed",
                    retry_count=cell.retry_count,
                    workload_command=cell.workload_command or None,
                    ssh_target=cell.ssh_target or None,
                    workload_stderr_path=None,
                    workload_exit_status=None,
                    keep_dumps=cell.keep_dumps,
                )
                record_early = sc.PerCellRecord(
                    schema_version=sc.SCHEMA_VERSION,
                    run_meta=run_meta_early,
                    producer_stats=sc.ProducerStats(),
                    analyzer_outputs=sc.AnalyzerOutputs(),
                    notes=notes,
                )
                return "failed", record_early, notes
            try:
                workload_proc = e1.start_workload(
                    ssh_target, ssh_key, ssh_opts, wcmd, workload_log,
                )
                notes.append(f"launched workload via SSH: {wcmd!r}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"workload launch failed: {exc}")
                workload_proc = None
                workload_skipped_reason = "start_workload_exception"
    elif cell.workload_command and not cell.ssh_target:
        notes.append("workload_command set but ssh_target empty; "
                     "treating as producer-only (label-only workload)")
        workload_skipped_reason = "no_ssh_target"

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
        # Stop workload first (so its stderr is flushed before we kill
        # the producer; matters for phase-marker capture).
        if workload_proc is not None:
            e1.stop_workload(workload_proc)
            try:
                workload_exit = workload_proc.returncode
            except Exception:  # noqa: BLE001
                workload_exit = None
        if producer_proc is not None:
            e1.stop_producer(producer_proc, grace_stop_seconds)
        # Bug-L settle (D-31, revised on Day 8 addendum):
        # the prior implementation polled `virsh domstate` (read-only,
        # doesn't acquire the state-change lock) which reported "stable"
        # while libvirt's remote dispatcher was still processing an
        # in-flight Domain.suspend RPC from the producer's last cycle.
        # The NEXT cell's `virsh resume` then hit "cannot acquire state
        # change lock (held by monitor=remoteDispatchDomainSuspend)".
        #
        # Real settle = retry `virsh resume` with backoff until it
        # actually succeeds (rc==0) or libvirt says "already active /
        # not paused". That is the only signal the state-change lock
        # has been released.
        try:
            settle_deadline = time.monotonic() + 15.0
            backoff = 0.3
            lock_attempts = 0
            other_errors: list[str] = []
            settled_state = None
            while time.monotonic() < settle_deadline:
                r = subprocess.run(
                    ["virsh", "-c", virsh_uri, "resume", vm_domain],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                combined = ((r.stderr or "") + (r.stdout or "")).strip()
                lower = combined.lower()
                if r.returncode == 0:
                    settled_state = "running"
                    break
                if "cannot acquire state change lock" in lower:
                    lock_attempts += 1
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 1.5)
                    continue
                if ("domain is already active" in lower
                        or "is not paused" in lower):
                    settled_state = "running"
                    break
                other_errors.append(combined[:200])
                time.sleep(backoff)
            notes.append(
                f"vm settle: state={settled_state!r} "
                f"lock_retries={lock_attempts} "
                f"other_errors={len(other_errors)}"
            )
            if other_errors:
                notes.append(f"vm settle stderr sample: {other_errors[0]!r}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"vm settle warning: {exc}")
        heartbeat.stop()
        heartbeat.join(timeout=5)

    # Move workload.log to workload_stderr.log so the per-cell artifact
    # is unambiguously named (start_workload writes a single combined log;
    # for now we treat it as stderr-equivalent).
    if workload_proc is not None and workload_log.exists():
        try:
            workload_log.replace(workload_stderr_path)
        except OSError:
            workload_stderr_path = workload_log

    run_ended_at = datetime.now(timezone.utc).isoformat()

    # Day-9 debug-team: scan producer.log for error-ish lines so the
    # post-cell record carries human-readable failure signals (pmemsave
    # I/O errors, virsh permission denials, "No space left on device",
    # etc.). Keeps per-cell notes self-describing instead of forcing
    # the reader to grep producer.log themselves.
    producer_errors = scan_producer_log(producer_log)
    if producer_errors:
        notes.append(f"producer.log errors ({len(producer_errors)}): "
                     + " | ".join(producer_errors[:3]))

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

    # B+3.1 (Δ-1 cell-end barrier): wait for async APF helpers to drain
    # before D-51 hook reads the trajectory. Then write the final sentinel
    # so the validator can prove completeness (Δ-3).
    apf_barrier_info: dict | None = None
    if cell.keep_dumps and apf_jsonl is not None and apf_ack_dir is not None:
        n_pairs_expected = max(0, int(producer_stats.snapshots_completed) - 1)
        apf_barrier_info = wait_for_apf_helpers(
            apf_ack_dir, n_pairs_expected, timeout_s=30.0,
        )
        try:
            write_apf_final_sentinel(apf_jsonl, apf_barrier_info)
        except OSError as exc:
            notes.append(f"B+3.1 sentinel write failed: {exc}")
        notes.append(
            f"B+3.1 barrier: expected={apf_barrier_info['n_pairs_expected']} "
            f"ok={apf_barrier_info['n_ok']} "
            f"failed={apf_barrier_info['n_failed']} "
            f"timed_out={apf_barrier_info['timed_out']}"
        )

    # D-51 · analyzer-then-delete hook
    # ────────────────────────────────────────────────────────────────────
    # If the cell was run with keep_dumps=True (manifest --keep-dumps),
    # the dumps are still on disk. Run the per-cell metrics computation
    # NOW (before tail-cleanup), populate analyzer_outputs, write a
    # metrics.json side-artifact, then let the cleanup step delete the
    # dumps to free disk for the next cell.
    image_dir = Path(eff_cfg.get("imageDir", "/var/lib/libvirt/qemu/dump"))
    metrics_path = cell_workdir / "metrics.json"
    pcm = None
    if cell.keep_dumps and _HAS_PMC and not cell.is_warmup:
        workload_type = _classify_workload(cell.workload)
        # B+3.1: if streaming APF trajectory exists from helpers, prefer it
        # (dumps already deleted by helpers). Otherwise fall back to dump-scan.
        use_streaming = apf_jsonl is not None and apf_jsonl.is_file()
        notes.append(
            f"D-51: running per-cell metrics "
            f"(workload_type={workload_type}, streaming={use_streaming})"
        )
        try:
            pcm = pmc.compute_metrics_for_cell(
                cell_id=cell.cell_id,
                image_dir=image_dir,
                run_start_epoch=run_start_epoch,
                jsonl_path=jsonl_path,
                workload_stderr_path=workload_stderr_path
                    if (workload_proc is not None and workload_stderr_path.exists())
                    else None,
                workload_type=workload_type,
                streaming_apf_jsonl=apf_jsonl if use_streaming else None,
            )
            # Persist full metrics.json (trajectory + summary + F1 breakdown)
            sc.write_json_atomic(metrics_path, asdict_safe(pcm))
            notes.append(
                f"D-51 metrics: n_dumps={pcm.n_dumps_examined} "
                f"n_pairs={pcm.n_pairs_examined} "
                f"apf_mean={pcm.apf_mean!r} "
                f"f1={pcm.f1_phase!r} cv={pcm.cv_workingset!r}"
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"D-51 metrics failed: {exc}")
            pcm = None
    elif cell.keep_dumps and not _HAS_PMC:
        notes.append("D-51 metrics skipped: plan02_metrics_per_cell not importable")
    elif cell.keep_dumps and cell.is_warmup:
        notes.append("D-51 metrics skipped: warmup cell")

    # Build analyzer_outputs from D-51 result (or zeros if no metrics ran)
    if pcm is not None:
        analyzer = sc.AnalyzerOutputs(
            f1_phase=pcm.f1_phase,
            cv_workingset=pcm.cv_workingset,
            n_windows=pcm.n_windows,
            n_snapshots=producer_stats.snapshots_completed,
        )
    else:
        analyzer = sc.AnalyzerOutputs(
            f1_phase=None,
            cv_workingset=None,
            n_windows=pmc.compute_n_windows(producer_stats.snapshots_completed)
                if _HAS_PMC else 0,
            n_snapshots=producer_stats.snapshots_completed,
        )

    # Clean up dump files this cell produced (now safe — metrics already
    # computed above if D-51 ran). For keep_dumps=False cells, the
    # producer's TIMING_SELF_CLEAN already removed dumps in-process; this
    # tail-cleanup catches the last one. For keep_dumps=True cells with
    # successful D-51 metrics, sweep the dumps now so the next cell has
    # disk headroom.
    try:
        if cell.keep_dumps and pcm is not None and pcm.n_dumps_examined > 0:
            removed = e1.cleanup_run_dumps(image_dir, run_start_epoch, False)
            notes.append(f"D-51 post-metrics cleanup: removed {removed} dumps")
        elif cell.keep_dumps:
            notes.append("keep_dumps=1 but D-51 did not run; dumps preserved")
        else:
            removed = e1.cleanup_run_dumps(image_dir, run_start_epoch, False)
            notes.append(f"cleanup: removed {removed} dump files")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"cleanup warning: {exc}")

    # Status decision (Bug-M fix, D-32: quantitative quality check)
    #
    # ok criterion now requires BOTH:
    #   - snapshots_completed > 0
    #   - backpressure ratio < 1%
    #   - snapshot_completion_ratio >= ratio_threshold
    #
    # Day-14 fix · mode-aware threshold (D-71):
    #   v1 (TIMING_SELF_CLEAN, keep_dumps=False): 0.30
    #   B+3.1 (keep_dumps=True, async APF helper): 0.15
    # The B+3.1 mode runs the helper concurrently with pmemsave; helper
    # I/O competes for disk bandwidth even with ionice -c 3, so the
    # effective pause-fraction rises 1.5-2× over the v1-calibrated
    # expected_snapshots() prior. A 0.15 threshold still catches the
    # true-failure modes (VM lock contention, workload absence, disk
    # full) without flagging cells whose data is intrinsically sound.
    expected = expected_snapshots(cell.duration_s, cell.interval_ms)
    actual = int(producer_stats.snapshots_completed)
    ratio = (actual / expected) if expected > 0 else 0.0
    ratio_threshold = 0.15 if cell.keep_dumps else 0.30
    # Persist ratio in producer_stats notes for the analysis layer
    notes.append(
        f"snap completion: actual={actual} expected={expected} "
        f"ratio={ratio:.2f} threshold={ratio_threshold:.2f} "
        f"(mode={'B+3.1' if cell.keep_dumps else 'v1'})"
    )

    status = "ok"
    if actual == 0:
        status = "failed"
        notes.append("FAIL: 0 snapshots completed; producer likely crashed")
    elif producer_stats.backpressure_events > max(
        1, int(0.01 * producer_stats.snapshots_attempted)
    ):
        # not auto-failing but flagging
        notes.append(
            f"WARN: high backpressure ({producer_stats.backpressure_events} "
            f"of {producer_stats.snapshots_attempted} attempts)"
        )
    if status == "ok" and ratio < ratio_threshold:
        status = "failed"
        notes.append(
            f"FAIL: snapshot_completion_ratio={ratio:.2f} < {ratio_threshold:.2f} "
            f"({actual} of expected {expected}). "
            f"Most likely cause: VM lock contention or workload absence."
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
        workload_command=cell.workload_command or None,
        ssh_target=cell.ssh_target or None,
        workload_stderr_path=(
            str(workload_stderr_path)
            if (workload_proc is not None and workload_stderr_path.exists())
            else None
        ),
        workload_exit_status=workload_exit,
        keep_dumps=cell.keep_dumps,
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
    purge_stale_dumps: bool = False,
    min_dumps_headroom: int = 5,
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

    # Bug-K preflight (D-30): if ANY pending row has a workload_command set,
    # require SSH_KEY or SSH_PASS env-var. Otherwise the operator will get
    # interactive password prompts mid-run, which is unsafe for unattended
    # pilots and tends to silently fail.
    needs_ssh = any(
        r.status == "pending" and r.workload_command for r in rows
    )
    if needs_ssh:
        ssh_key = os.environ.get("SSH_KEY", "")
        ssh_pass = os.environ.get("SSH_PASS", "")
        allow_interactive = os.environ.get(
            "PLAN02_ALLOW_INTERACTIVE_SSH", ""
        ).strip()
        if not ssh_key and not ssh_pass and not allow_interactive:
            log("ERROR: pending cells have workload_command set, but neither "
                "SSH_KEY nor SSH_PASS is exported.")
            log("  Set one of:")
            log("    export SSH_KEY=/path/to/private_key")
            log("    export SSH_PASS=<vm-password>   # requires sshpass installed")
            log("  Or to ignore (debugging only):")
            log("    export PLAN02_ALLOW_INTERACTIVE_SSH=1")
            return 3

    # Session sentinel: write host state once per session
    host_meta = sc.collect_host_meta(repo_root=SCRIPT_DIR.parent)

    # Day-9 debug-team preflight: dump-dir health + disk-free assertion.
    # Pull ram_mb from config so we can size disk requirements accurately.
    try:
        base_cfg = e1.load_config(config_path)
        ram_mb = int(base_cfg.get("ramSizeMb", 1024))
        image_dir = Path(base_cfg.get("imageDir", "/var/lib/libvirt/qemu/dump"))
    except (OSError, ValueError):
        ram_mb = 1024
        image_dir = Path("/var/lib/libvirt/qemu/dump")
    preflight_info = session_preflight_disk(
        image_dir, ram_mb,
        min_headroom_dumps=min_dumps_headroom,
        purge_stale=purge_stale_dumps,
    )
    log(f"preflight: stale_dumps_before={preflight_info['stale_dumps_before']} "
        f"disk_free_gib={preflight_info['disk_free_gib_before']:.1f} "
        f"purged={preflight_info['purged_count']} "
        f"stale_dumps_after={preflight_info['stale_dumps_after']} "
        f"disk_free_gib_after={preflight_info['disk_free_gib_after']:.1f}")
    if not preflight_info["ok"]:
        log(f"ERROR preflight: {preflight_info['reason']}")
        log("  Fix: free disk space OR run with --purge-stale-dumps "
            "OR lower --min-dumps-headroom.")
        return 4

    sentinel_path = output_dir / "session_sentinel.json"
    sc.write_json_atomic(sentinel_path, {
        "session_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "manifest_path": str(manifest_path),
        "host_meta": host_meta,
        "summary_at_start": mf.summarize(rows),
        "preflight": preflight_info,
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
    p.add_argument("--purge-stale-dumps", action="store_true",
                   help="Before the session starts, sudo-rm any "
                        "memory_dump-*.raw files in imageDir. Use this when "
                        "previous failed runs left dumps behind that the "
                        "current user cannot delete directly.")
    p.add_argument("--min-dumps-headroom", type=int, default=5,
                   help="Per-cell preflight refuses to launch unless free "
                        "disk >= this many full-size dumps (RAM-sized). "
                        "Default 5. Lower to be more permissive; raise for "
                        "safety. Day-9 debug team default.")
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
        purge_stale_dumps=args.purge_stale_dumps,
        min_dumps_headroom=args.min_dumps_headroom,
    )


if __name__ == "__main__":
    sys.exit(_main())
