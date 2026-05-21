#!/usr/bin/env python3
"""run_timing_instrumentation_experiment.py — Experiment 1 driver.

Implements only the *first* tuning-plan experiment: capture-timing
instrumentation. See ``docs/tuning_plans/01_instrumentation_logging_plan.md``
and ``docs/RUN_TIMING_INSTRUMENTATION.md`` for the rationale.

Flow
----
1. Read the chosen capture ``config_qemu_upc.json`` (or another path via
   ``--config``).
2. Optionally override ``intervalMsec`` and ``ramSizeMb`` via CLI flags.
   The override is written to a temporary copy of the config — the
   original file is never modified.
3. Make sure the libvirt domain is running. Optionally wait for SSH
   reachability if ``--ssh-target`` is given.
4. Start the instrumented producer
   (``capture_producer_qemu_pmemsave.sh``) with ``TIMING_JSONL_PATH``
   pointing at a per-run JSONL file under ``--workdir``. No consumer
   is launched — Experiment 1 only measures the producer side.
5. Optionally start a single workload command on the guest over SSH
   (``--test-command``); the orchestrator does not wait for it to
   finish, only for the duration window.
6. Sleep for ``--duration`` seconds (host wall-clock).
7. Stop the producer (SIGTERM, then SIGKILL after a grace period).
8. Optionally cancel the SSH workload (best-effort SIGINT to the SSH
   process; the guest-side workload may or may not honor it).
9. Parse the producer's JSONL timing log, compute summary statistics,
   write a single JSON output file at ``--output-json``.
10. Best-effort cleanup of the snapshot images written by this run
    (only files written *after* the run started; never older).

The script is intentionally self-contained: no consumer, no streaming
pipeline, no delta calculation. The point of Experiment 1 is to
*measure*, not to *produce signal*.

Outputs one JSON file with the schema documented in
``docs/RUN_TIMING_INSTRUMENTATION.md``.

Safety
------
- The original capture config and dump files older than the run start
  are never modified or removed.
- Snapshot images are removed only if they (a) live under the
  ``imageDir`` from the config and (b) have mtime ≥ run start.
- ``--dry-run`` validates inputs and emits the resolved plan without
  touching the VM, the producer, or the queue.
- The script exits non-zero if the JSONL file is missing or empty at
  end-of-run, so a silent capture failure is loud.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_qemu_upc.json"
DEFAULT_PRODUCER = Path(__file__).resolve().parent / "capture_producer_qemu_pmemsave.sh"


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[timing-exp] {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], check: bool = True, capture: bool = False, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a command, optionally capturing stdout/stderr."""
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_timing_instrumentation_experiment.py",
        description=(
            "Run Experiment 1 (capture-timing instrumentation) and write a "
            "single JSON file with per-snapshot timing measurements. "
            "Does not run interval / window / hop / k-segmentation tuning."
        ),
    )
    p.add_argument("--output-json", required=False,
                   default=None,
                   help="Path to the output JSON. Default: ./timing_experiment_<UTC>.json")
    p.add_argument("--duration", type=int, default=60,
                   help="Host wall-clock duration in seconds (default 60). The "
                        "producer is allowed to run for this long; snapshot count "
                        "depends on pmemsave throughput.")
    p.add_argument("--interval-ms", type=int, default=None,
                   help="Override intervalMsec in the config snapshot. Default: "
                        "use the value already in --config.")
    p.add_argument("--ram-mb", type=int, default=None,
                   help="Override ramSizeMb in the config snapshot. Default: use "
                        "the value already in --config. Must match the live VM RAM.")
    p.add_argument("--test-command", type=str, default=None,
                   help="Optional command to run on the guest over SSH (e.g. a "
                        "Phase 2 binary). Started just before the duration window "
                        "opens. Not required for the timing experiment itself.")
    p.add_argument("--ssh-target", type=str, default=os.environ.get("SSH_TARGET", ""),
                   help="user@host for the guest. Required only when --test-command is set.")
    p.add_argument("--ssh-key", type=str, default=os.environ.get("SSH_KEY", ""),
                   help="Optional SSH private key path.")
    p.add_argument("--ssh-opts", type=str, default=os.environ.get("SSH_OPTS", ""),
                   help="Optional extra SSH options (single string).")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                   help="Path to the capture config JSON. Default: ./config_qemu_upc.json")
    p.add_argument("--producer-script", type=str, default=str(DEFAULT_PRODUCER),
                   help="Path to the instrumented producer. Default: ./capture_producer_qemu_pmemsave.sh")
    p.add_argument("--workdir", type=str, default=None,
                   help="Directory for per-run scratch files (JSONL, config copy, log). "
                        "Default: a fresh ./timing_runs/<UTC>_<uuid> directory.")
    p.add_argument("--keep-dumps", action="store_true",
                   help="Do not remove snapshot images written by this run. By default "
                        "the orchestrator cleans up dumps it created.")
    p.add_argument("--no-vm-start", action="store_true",
                   help="Do not run 'virsh start'; assume the VM is already running.")
    p.add_argument("--virsh-uri", type=str, default="qemu:///system",
                   help="libvirt URI for virsh (default qemu:///system).")
    p.add_argument("--grace-stop-seconds", type=int, default=10,
                   help="Seconds to wait between SIGTERM and SIGKILL on the producer "
                        "during shutdown (default 10).")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and emit the resolved plan as JSON without "
                        "touching the VM, the producer, or any dump file.")
    p.add_argument("--verbose", action="store_true",
                   help="Print extra status messages.")
    return p


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_overridden_config(orig: dict, workdir: Path, interval_ms: int | None,
                             ram_mb: int | None) -> tuple[Path, dict]:
    cfg = json.loads(json.dumps(orig))  # deep copy
    if interval_ms is not None:
        cfg["intervalMsec"] = int(interval_ms)
    if ram_mb is not None:
        cfg["ramSizeMb"] = int(ram_mb)
    # Disable streaming + raw retention so Experiment 1 is producer-only.
    if "streaming" in cfg and isinstance(cfg["streaming"], dict):
        cfg["streaming"]["enabled"] = False
    if "rawRetention" in cfg and isinstance(cfg["rawRetention"], dict):
        cfg["rawRetention"]["enabled"] = False
    out_path = workdir / "config_timing_experiment.json"
    with out_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    return out_path, cfg


# ---------------------------------------------------------------------------
# VM control
# ---------------------------------------------------------------------------

def virsh_domstate(uri: str, domain: str) -> str:
    try:
        out = subprocess.run(
            ["virsh", "-c", uri, "domstate", domain],
            capture_output=True, text=True, check=False,
        )
        return (out.stdout or "").strip()
    except FileNotFoundError:
        return ""


def virsh_start_if_needed(uri: str, domain: str, dry_run: bool) -> str:
    state = virsh_domstate(uri, domain)
    if state == "running":
        return state
    if dry_run:
        log(f"[dry-run] would 'virsh start \"{domain}\"' (current state: {state!r})")
        return "running-simulated"
    if state == "shut off" or state == "":
        log(f"virsh start \"{domain}\"")
        subprocess.run(["virsh", "-c", uri, "start", domain], check=False)
    elif state == "paused":
        log(f"virsh resume \"{domain}\"")
        subprocess.run(["virsh", "-c", uri, "resume", domain], check=False)
    # Re-poll briefly.
    for _ in range(60):
        state = virsh_domstate(uri, domain)
        if state == "running":
            return state
        time.sleep(0.5)
    return state


# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------

def build_ssh_cmd(target: str, key: str, opts: str, command: str) -> list[str]:
    cmd = ["ssh"]
    if key:
        cmd += ["-i", key]
    if opts:
        cmd += shlex.split(opts)
    cmd += [target, command]
    return cmd


def wait_for_ssh(target: str, key: str, opts: str, timeout_s: int = 120) -> bool:
    if not target:
        return True
    probe = build_ssh_cmd(target, key, "-o ConnectTimeout=5 -o StrictHostKeyChecking=no " + opts, "true")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            r = subprocess.run(probe, capture_output=True, timeout=10)
            if r.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Producer + workload management
# ---------------------------------------------------------------------------

def start_producer(producer_script: Path, config_path: Path, jsonl_path: Path,
                   log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CONFIG"] = str(config_path)
    env["TIMING_JSONL_PATH"] = str(jsonl_path)
    log(f"starting producer: {producer_script} (CONFIG={config_path}, TIMING_JSONL_PATH={jsonl_path})")
    log_f = log_path.open("ab")
    proc = subprocess.Popen(
        ["bash", str(producer_script)],
        env=env,
        stdout=log_f,
        stderr=log_f,
        start_new_session=True,
    )
    return proc


def stop_producer(proc: subprocess.Popen, grace: int) -> None:
    if proc.poll() is not None:
        return
    log(f"stopping producer (pid={proc.pid}, grace={grace}s)")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    log("producer did not exit on SIGTERM; sending SIGKILL")
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5)


def start_workload(ssh_target: str, ssh_key: str, ssh_opts: str, command: str,
                   log_path: Path) -> subprocess.Popen | None:
    if not command:
        return None
    if not ssh_target:
        log("WARNING: --test-command supplied but --ssh-target empty; skipping workload")
        return None
    cmd = build_ssh_cmd(ssh_target, ssh_key, ssh_opts, command)
    log(f"starting workload over SSH: {' '.join(shlex.quote(c) for c in cmd)}")
    log_f = log_path.open("ab")
    return subprocess.Popen(cmd, stdout=log_f, stderr=log_f, start_new_session=True)


def stop_workload(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    log(f"stopping workload SSH (pid={proc.pid})")
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# JSONL parsing and summary
# ---------------------------------------------------------------------------

def parse_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (snapshot_records, backpressure_records)."""
    snaps: list[dict] = []
    bps: list[dict] = []
    if not path.exists():
        return snaps, bps
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("seq") == -1 and obj.get("backpressure_event"):
            bps.append(obj)
        else:
            snaps.append(obj)
    snaps.sort(key=lambda d: d.get("seq", 0))
    return snaps, bps


def safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_snapshot_record(idx: int, this: dict, nxt: dict | None) -> dict:
    t0 = safe_float(this.get("t0_before_suspend"))
    t1 = safe_float(this.get("t1_after_suspend"))
    t2 = safe_float(this.get("t2_pmemsave_start"))
    t3 = safe_float(this.get("t3_pmemsave_end"))
    t4 = safe_float(this.get("t4_before_resume"))
    t5 = safe_float(this.get("t5_after_resume"))
    next_t0 = safe_float(nxt.get("t0_before_suspend")) if nxt else None

    suspend_sec = (t1 - t0) if (t1 is not None and t0 is not None) else None
    pmemsave_sec = (t3 - t2) if (t3 is not None and t2 is not None) else None
    resume_sec = (t5 - t4) if (t5 is not None and t4 is not None) else None
    host_cycle_sec = (next_t0 - t0) if (next_t0 is not None and t0 is not None) else None
    guest_run_sec = (next_t0 - t5) if (next_t0 is not None and t5 is not None) else None

    return {
        "index": idx,
        "resume_start_ts": t4,
        "resume_done_ts": t5,
        "guest_run_start_ts": t5,
        "suspend_start_ts": t0,
        "suspend_done_ts": t1,
        "pmemsave_start_ts": t2,
        "pmemsave_done_ts": t3,
        "next_resume_ts": next_t0,
        "guest_run_interval_sec": guest_run_sec,
        "suspend_sec": suspend_sec,
        "pmemsave_sec": pmemsave_sec,
        "resume_sec": resume_sec,
        "host_snapshot_cycle_sec": host_cycle_sec,
        "queue_depth": this.get("queue_depth"),
        "backpressure": bool(this.get("backpressure_event", False)),
        "image_path": this.get("image_path"),
    }


def summarize(records: list[dict], bps: list[dict]) -> dict:
    def mean_std(key: str) -> tuple[float | None, float | None]:
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        if not vals:
            return None, None
        if len(vals) == 1:
            return float(vals[0]), 0.0
        return float(statistics.fmean(vals)), float(statistics.pstdev(vals))

    g_mean, g_std = mean_std("guest_run_interval_sec")
    h_mean, h_std = mean_std("host_snapshot_cycle_sec")
    sus_mean, _ = mean_std("suspend_sec")
    pmem_mean, _ = mean_std("pmemsave_sec")
    res_mean, _ = mean_std("resume_sec")

    queue_depths = [r["queue_depth"] for r in records if isinstance(r.get("queue_depth"), int)]
    queue_max = max(queue_depths) if queue_depths else None

    # vm pause fraction = sum(t5 - t0) / sum(next_t0 - t0)
    pause_num = 0.0
    pause_den = 0.0
    for r in records:
        t0 = r.get("suspend_start_ts")
        t5 = r.get("resume_done_ts")
        nxt = r.get("next_resume_ts")
        if isinstance(t0, (int, float)) and isinstance(t5, (int, float)) and isinstance(nxt, (int, float)):
            pause_num += (t5 - t0)
            pause_den += (nxt - t0)
    pause_fraction = (pause_num / pause_den) if pause_den > 0 else None

    return {
        "snapshots_attempted": len(records) + len(bps),
        "snapshots_completed": len(records),
        "mean_guest_run_interval_sec": g_mean,
        "std_guest_run_interval_sec": g_std,
        "mean_host_snapshot_cycle_sec": h_mean,
        "std_host_snapshot_cycle_sec": h_std,
        "mean_suspend_sec": sus_mean,
        "mean_pmemsave_sec": pmem_mean,
        "mean_resume_sec": res_mean,
        "backpressure_events": len(bps),
        "queue_max_depth": queue_max,
        "estimated_vm_pause_fraction": pause_fraction,
    }


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def _try_sudo_rm(paths: list[Path]) -> int:
    """Best-effort `sudo rm` for paths the user cannot unlink directly
    (e.g. dumps under a root-owned imageDir). Returns count attempted.
    Silent if sudo is not configured for passwordless rm."""
    if not paths:
        return 0
    try:
        out = subprocess.run(
            ["sudo", "-n", "rm", "-f", *[str(p) for p in paths]],
            capture_output=True, text=True, timeout=30,
        )
        return len(paths) if out.returncode == 0 else 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0


def cleanup_run_dumps(image_dir: Path, run_start_epoch: float, keep: bool) -> int:
    """Remove memory_dump-*.raw files created after run_start. Returns count.

    Tries direct `unlink` first. Falls back to `sudo -n rm -f` for files that
    refuse to unlink (typical when imageDir is root-owned even though the
    files themselves are jeries-owned). The sudo step is silent if sudo is
    not configured passwordless — in that case the file count reported is
    just what unlink could remove.
    """
    if keep:
        return 0
    if not image_dir.is_dir():
        return 0
    removed = 0
    needs_sudo: list[Path] = []
    for p in image_dir.glob("memory_dump-*.raw"):
        try:
            if p.stat().st_mtime >= run_start_epoch - 1.0:
                try:
                    p.unlink()
                    removed += 1
                except PermissionError:
                    needs_sudo.append(p)
        except FileNotFoundError:
            continue
        except OSError as e:
            log(f"WARNING: failed to stat {p}: {e}")
    if needs_sudo:
        sudo_removed = _try_sudo_rm(needs_sudo)
        removed += sudo_removed
        if sudo_removed == 0:
            log(f"WARNING: {len(needs_sudo)} dumps could not be removed "
                f"(parent dir is read-only for current user). Consider running "
                f"`sudo rm {image_dir}/memory_dump-*.raw` between experiments.")
    return removed


def purge_all_dumps(image_dir: Path, use_sudo: bool = True) -> int:
    """Aggressively remove ALL memory_dump-*.raw files in imageDir, ignoring
    mtime. Used by --purge-stale-dumps before a fresh experiment."""
    if not image_dir.is_dir():
        return 0
    all_paths = list(image_dir.glob("memory_dump-*.raw"))
    if not all_paths:
        return 0
    direct_removed = 0
    stragglers: list[Path] = []
    for p in all_paths:
        try:
            p.unlink(); direct_removed += 1
        except (FileNotFoundError, PermissionError):
            stragglers.append(p)
        except OSError:
            stragglers.append(p)
    if use_sudo and stragglers:
        return direct_removed + _try_sudo_rm(stragglers)
    return direct_removed


def disk_free_check(image_dir: Path, snapshots_expected: int, ram_mb: int,
                    margin: float = 1.5,
                    peak_concurrent_dumps: int | None = None) -> tuple[bool, dict]:
    """Refuse to start if free space < peak_concurrent_dumps × ram × margin.

    Plan 4: when the producer's TIMING_SELF_CLEAN is on (or the consumer is
    running normally), the steady-state dump count on disk is ~2 (prev +
    curr), not `snapshots_expected`. Callers should pass the realistic
    `peak_concurrent_dumps` matching their cleanup policy. When the
    parameter is None (back-compat) the original snapshots_expected-based
    check applies.
    """
    try:
        st = os.statvfs(image_dir if image_dir.is_dir() else image_dir.parent)
    except (OSError, AttributeError):
        return True, {"free_bytes": None, "note": "statvfs unavailable"}
    free = st.f_bavail * st.f_frsize
    peak = peak_concurrent_dumps if peak_concurrent_dumps is not None else snapshots_expected
    need = int(peak * ram_mb * 1024 * 1024 * margin)
    return (free >= need), {"free_bytes": free, "needed_bytes": need,
                             "margin": margin,
                             "snapshots_expected": snapshots_expected,
                             "peak_concurrent_dumps": peak}


def resume_vm_if_paused(virsh_uri: str, domain: str) -> bool:
    """Best-effort resume in case a SIGTERM'd producer left the VM paused.
    Returns True if the resume was attempted (regardless of outcome).
    Safe to call on a running VM — virsh resume on running domain errors
    harmlessly."""
    try:
        subprocess.run(
            ["virsh", "-c", virsh_uri, "resume", domain],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    producer_script = Path(args.producer_script).expanduser().resolve()
    if not config_path.is_file():
        log(f"ERROR: config not found: {config_path}")
        return 2
    if not producer_script.is_file():
        log(f"ERROR: producer not found: {producer_script}")
        return 2

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else (
        Path.cwd() / "timing_runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    workdir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output_json).expanduser().resolve() if args.output_json else (
        workdir / f"timing_experiment_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    orig_cfg = load_config(config_path)
    cfg_path, eff_cfg = write_overridden_config(orig_cfg, workdir, args.interval_ms, args.ram_mb)
    image_dir = Path(eff_cfg.get("imageDir", "/var/lib/libvirt/qemu/dump")).expanduser()
    vm_domain = eff_cfg.get("domain", "")
    interval_ms = int(eff_cfg.get("intervalMsec", 100))
    ram_mb = int(eff_cfg.get("ramSizeMb", 0))

    jsonl_path = workdir / "snapshot_timings.jsonl"
    producer_log = workdir / "producer.log"
    workload_log = workdir / "workload.log"

    notes: list[str] = []

    plan = {
        "experiment": "timing_instrumentation",
        "timestamp": iso_now(),
        "config": {
            "interval_ms": interval_ms,
            "duration_sec": args.duration,
            "ram_size_mb": ram_mb,
            "vm_domain": vm_domain,
            "capture_method": "pmemsave",
            "test_command": args.test_command or "",
            "config_path_input": str(config_path),
            "config_path_effective": str(cfg_path),
            "workdir": str(workdir),
            "output_json": str(out_path),
            "image_dir": str(image_dir),
        },
        "summary": None,
        "snapshots": [],
        "notes": notes,
    }

    if args.dry_run:
        plan["summary"] = {
            "snapshots_attempted": 0,
            "snapshots_completed": 0,
            "mean_guest_run_interval_sec": None,
            "std_guest_run_interval_sec": None,
            "mean_host_snapshot_cycle_sec": None,
            "std_host_snapshot_cycle_sec": None,
            "mean_suspend_sec": None,
            "mean_pmemsave_sec": None,
            "mean_resume_sec": None,
            "backpressure_events": None,
            "queue_max_depth": None,
            "estimated_vm_pause_fraction": None,
        }
        notes.append("dry-run: no VM operation, no producer started, no JSONL parsed")
        notes.append(f"would launch producer with TIMING_JSONL_PATH={jsonl_path}")
        notes.append(f"would run for --duration={args.duration}s at intervalMsec={interval_ms}")
        if args.test_command:
            notes.append(f"would run workload over SSH ({args.ssh_target}): {args.test_command}")
        else:
            notes.append("no --test-command supplied: idle VM during the measurement window")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(plan, f, indent=2)
        log(f"dry-run plan written to {out_path}")
        return 0

    # Real run.
    if not args.no_vm_start:
        state = virsh_start_if_needed(args.virsh_uri, vm_domain, dry_run=False)
        if state != "running":
            notes.append(f"VM not running after start attempt: state={state!r}; "
                         f"continuing anyway since producer will report errors")
    else:
        notes.append("--no-vm-start: trusting caller to ensure VM is running")

    if args.test_command and args.ssh_target:
        ok = wait_for_ssh(args.ssh_target, args.ssh_key, args.ssh_opts, timeout_s=120)
        if not ok:
            notes.append("SSH not reachable within 120 s; workload will not run, "
                         "timing measurement proceeds")
            args.test_command = ""  # downgrade

    run_start_epoch = time.time()
    plan["config"]["run_start_epoch"] = run_start_epoch

    producer_proc = start_producer(producer_script, cfg_path, jsonl_path, producer_log)
    workload_proc = start_workload(args.ssh_target, args.ssh_key, args.ssh_opts,
                                   args.test_command or "", workload_log)

    try:
        log(f"sleeping {args.duration}s host wall-clock")
        time.sleep(args.duration)
    except KeyboardInterrupt:
        notes.append("interrupted by user during measurement window")
    finally:
        stop_workload(workload_proc)
        stop_producer(producer_proc, args.grace_stop_seconds)

    snaps, bps = parse_jsonl(jsonl_path)
    if not snaps and not bps:
        notes.append("WARNING: JSONL empty or missing; check producer.log for errors")

    records = []
    for i, this in enumerate(snaps):
        nxt = snaps[i + 1] if i + 1 < len(snaps) else None
        records.append(build_snapshot_record(i, this, nxt))

    plan["snapshots"] = records
    plan["summary"] = summarize(records, bps)
    plan["summary"]["jsonl_path"] = str(jsonl_path)
    plan["summary"]["producer_log"] = str(producer_log)
    if args.test_command:
        plan["summary"]["workload_log"] = str(workload_log)

    removed = cleanup_run_dumps(image_dir, run_start_epoch, args.keep_dumps)
    notes.append(f"cleanup: removed {removed} dump files newer than run start "
                 f"(--keep-dumps={args.keep_dumps})")
    notes.append("intervalMsec set the *guest-running* interval, not host wall-clock; "
                 "see docs/SNAPSHOT_INTERVAL_QA.md.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(plan, f, indent=2)
    log(f"wrote {out_path}")

    if not records and not bps:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
