#!/usr/bin/env python3
"""run_exp2a_consumer_isolation.py — Experiment 2a.

Tests whether killing the consumer collapses the non-stationary host Δt
observed in Experiment 1 Run 2 (mechanism iii). Runs two passes:

  Pass 1 — consumer_on:  any pre-existing consumer left running.
  Pass 2 — consumer_off: pkill the consumer + drain queue before starting.

Both passes use the instrumented producer (with bc fix in place) and write
their own per-snapshot timing JSONL. Aggregate stats are compared in the
final JSON.

Produces one JSON file with the schema documented at the bottom.

Safety:
  - Sandbox-validates --workdir.
  - Only kills processes matching the producer/consumer script basenames.
  - Cleans only dump files written after each pass began.
  - --dry-run validates and emits the planned JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Import shared helpers from Experiment 1's orchestrator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_timing_instrumentation_experiment as e1


DEFAULT_CONSUMER_SCRIPT = Path(__file__).resolve().parent / "capture_consumer_qemu.sh"


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[exp2a] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Process control
# ---------------------------------------------------------------------------

def consumer_pids() -> list[int]:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "capture_consumer_qemu.sh"],
            capture_output=True, text=True, check=False,
        )
        return [int(p) for p in out.stdout.split() if p.isdigit()]
    except FileNotFoundError:
        return []


def kill_consumer(grace: int = 5) -> int:
    pids = consumer_pids()
    if not pids:
        return 0
    log(f"killing consumer pid(s): {pids}")
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace:
        if not consumer_pids():
            return len(pids)
        time.sleep(0.5)
    # escalate
    for p in consumer_pids():
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(pids)


def drain_queue(queue_dir: Path) -> int:
    """Remove all pending+processing job files. Returns count removed."""
    if not queue_dir.is_dir():
        return 0
    removed = 0
    for sub in ("pending", "processing"):
        d = queue_dir / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            try:
                p.unlink(); removed += 1
            except FileNotFoundError:
                pass
    return removed


# ---------------------------------------------------------------------------
# One measurement pass
# ---------------------------------------------------------------------------

def run_pass(name: str, workdir: Path, cfg_path: Path, producer_script: Path,
             duration: int, grace: int) -> dict:
    pass_dir = workdir / name
    pass_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path  = pass_dir / "snapshot_timings.jsonl"
    log_path    = pass_dir / "producer.log"

    log(f"[{name}] starting producer ({duration}s)")
    proc = e1.start_producer(producer_script, cfg_path, jsonl_path, log_path)
    t0 = time.time()
    try:
        time.sleep(duration)
    finally:
        e1.stop_producer(proc, grace)
        # ensure VM is not left paused by mid-cycle SIGTERM
        with open(cfg_path) as _f:
            _cfg = json.load(_f)
        e1.resume_vm_if_paused("qemu:///system", _cfg.get("domain", ""))
    t1 = time.time()

    snaps, bps = e1.parse_jsonl(jsonl_path)
    records = [e1.build_snapshot_record(i, s, snaps[i+1] if i+1 < len(snaps) else None)
               for i, s in enumerate(snaps)]
    summary = e1.summarize(records, bps)
    summary["wall_clock_seconds"]   = round(t1 - t0, 3)
    summary["jsonl_path"]           = str(jsonl_path)
    summary["producer_log"]         = str(log_path)
    return {"records": records, "summary": summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_exp2a_consumer_isolation.py",
        description=(
            "Experiment 2a: isolate consumer contention. Runs two passes "
            "(consumer-on, consumer-off) with the instrumented producer and "
            "compares host_dt / suspend / pmemsave distributions."
        ),
    )
    p.add_argument("--output-json", default=None,
                   help="Output JSON path. Default: ./exp2a_<UTC>.json")
    p.add_argument("--duration", type=int, default=60,
                   help="Duration of each pass in seconds (default 60).")
    p.add_argument("--interval-ms", type=int, default=None,
                   help="Override intervalMsec in the config snapshot.")
    p.add_argument("--ram-mb", type=int, default=None,
                   help="Override ramSizeMb in the config snapshot.")
    p.add_argument("--config", default=str(e1.DEFAULT_CONFIG))
    p.add_argument("--producer-script", default=str(e1.DEFAULT_PRODUCER))
    p.add_argument("--consumer-script", default=str(DEFAULT_CONSUMER_SCRIPT))
    p.add_argument("--workdir", default=None)
    p.add_argument("--skip-pass1", action="store_true",
                   help="Skip the consumer-on pass (e.g. if consumer isn't running anyway).")
    p.add_argument("--keep-dumps", action="store_true")
    p.add_argument("--purge-stale-dumps", action="store_true",
                   help="Before pass 1, aggressively `sudo rm` ALL "
                        "memory_dump-*.raw files in imageDir. Recovers from "
                        "stale dumps left by previous runs.")
    p.add_argument("--drain-before-pass1", action="store_true",
                   help="Drain the queue dir before pass 1 too (default: only before pass 2).")
    p.add_argument("--virsh-uri", default="qemu:///system")
    p.add_argument("--no-vm-start", action="store_true")
    p.add_argument("--grace-stop-seconds", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    producer    = Path(args.producer_script).expanduser().resolve()
    consumer    = Path(args.consumer_script).expanduser().resolve()
    if not config_path.is_file(): log(f"ERROR: config not found: {config_path}"); return 2
    if not producer.is_file():    log(f"ERROR: producer not found: {producer}"); return 2

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else (
        Path.cwd() / "timing_runs" /
        f"exp2a_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_json).expanduser().resolve() if args.output_json else (
        workdir / f"exp2a_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    orig_cfg = e1.load_config(config_path)
    cfg_path, eff_cfg = e1.write_overridden_config(orig_cfg, workdir,
                                                   args.interval_ms, args.ram_mb)
    queue_dir = Path(eff_cfg.get("queueDir", "/tmp/queue_dir"))
    image_dir = Path(eff_cfg.get("imageDir", "/var/lib/libvirt/qemu/dump"))

    result: dict = {
        "experiment": "2a_consumer_isolation",
        "timestamp": iso_now(),
        "config": {
            "interval_ms": int(eff_cfg["intervalMsec"]),
            "duration_sec": args.duration,
            "ram_size_mb": int(eff_cfg["ramSizeMb"]),
            "vm_domain": eff_cfg.get("domain", ""),
            "capture_method": "pmemsave",
            "config_path_input": str(config_path),
            "config_path_effective": str(cfg_path),
            "workdir": str(workdir),
            "image_dir": str(image_dir),
            "queue_dir": str(queue_dir),
            "skip_pass1": args.skip_pass1,
        },
        "passes": {},
        "comparison": {},
        "notes": [],
    }

    if args.dry_run:
        result["notes"] += [
            "dry-run: no VM operation, no producer started",
            "would run pass_consumer_on (if --skip-pass1 not set) then pass_consumer_off",
            f"would kill any process matching 'capture_consumer_qemu.sh' before pass 2",
            f"would drain queue dir {queue_dir}/pending and /processing",
        ]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f: json.dump(result, f, indent=2)
        log(f"dry-run plan written to {out_path}"); return 0

    if not args.no_vm_start:
        state = e1.virsh_start_if_needed(args.virsh_uri, eff_cfg.get("domain",""), False)
        if state != "running":
            result["notes"].append(f"VM not running after start attempt: {state!r}")

    # Optional purge of stale dumps left by prior experiments. Uses sudo.
    if args.purge_stale_dumps:
        n = e1.purge_all_dumps(image_dir, use_sudo=True)
        log(f"purged {n} stale dump file(s) from {image_dir}")
        result["pre_run"] = {"stale_dumps_purged": n}

    # Optional drain before pass 1 if user wants a clean slate
    if args.drain_before_pass1:
        d0 = drain_queue(queue_dir)
        log(f"pre-pass1: drained {d0} queue file(s)")

    run_start_epoch = time.time()

    # --- pass 1: consumer-on ---
    if not args.skip_pass1:
        pids = consumer_pids()
        result["passes"]["consumer_on"] = {"consumer_pids_at_start": pids}
        if not pids:
            result["notes"].append("pass 1 expected consumer running but none found; "
                                   "pass results reflect 'no consumer' state")
        p1 = run_pass("consumer_on", workdir, cfg_path, producer,
                      args.duration, args.grace_stop_seconds)
        result["passes"]["consumer_on"].update(p1["summary"])
        result["passes"]["consumer_on"]["records"] = p1["records"]

    # --- pass 2: consumer-off ---
    killed = kill_consumer(grace=5)
    drained = drain_queue(queue_dir)
    log(f"pass 2 prep: killed {killed} consumer process(es); drained {drained} queue file(s)")
    result["passes"]["consumer_off"] = {
        "consumer_processes_killed": killed,
        "queue_files_drained":       drained,
    }
    p2 = run_pass("consumer_off", workdir, cfg_path, producer,
                  args.duration, args.grace_stop_seconds)
    result["passes"]["consumer_off"].update(p2["summary"])
    result["passes"]["consumer_off"]["records"] = p2["records"]

    # --- comparison ---
    if "consumer_on" in result["passes"]:
        on = result["passes"]["consumer_on"]
        off = result["passes"]["consumer_off"]
        def diff(key):
            a, b = on.get(key), off.get(key)
            if not (isinstance(a, (int,float)) and isinstance(b, (int,float))): return None
            return {"on": round(a,4), "off": round(b,4),
                    "delta": round(b-a,4),
                    "ratio_off_over_on": round(b/a,3) if a else None}
        result["comparison"] = {
            "mean_host_snapshot_cycle_sec":  diff("mean_host_snapshot_cycle_sec"),
            "mean_suspend_sec":              diff("mean_suspend_sec"),
            "mean_pmemsave_sec":             diff("mean_pmemsave_sec"),
            "mean_guest_run_interval_sec":   diff("mean_guest_run_interval_sec"),
            "backpressure_events":           diff("backpressure_events"),
            "queue_max_depth":               diff("queue_max_depth"),
        }

    # --- cleanup dumps written during this run ---
    removed = e1.cleanup_run_dumps(image_dir, run_start_epoch, args.keep_dumps)
    result["notes"].append(f"cleanup: removed {removed} dump files newer than run start "
                           f"(--keep-dumps={args.keep_dumps})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f: json.dump(result, f, indent=2)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
