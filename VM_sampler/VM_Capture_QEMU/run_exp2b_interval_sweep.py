#!/usr/bin/env python3
"""run_exp2b_interval_sweep.py — Experiment 2b.

Sensitivity sweep on intervalMsec to characterise how VM pause fraction
changes with the configured guest-running interval. Optional pmemsave-size
sweep simulates "smaller VM" without rebooting (see caveat in notes).

Each cell runs producer-only (consumer killed up front) for --duration
seconds and emits one summary row. Cross-cell comparison is computed and
written to the output JSON.

Targets mechanism iv (pause-fraction noise floor). Step 2b in
docs/timing_experiment_1_lab.html.

Produces one JSON file with per-cell summaries + cross-cell comparison.

Caveats:
  - When --ram-sweep is on, the producer's pmemsave argument is reduced
    while the VM itself keeps its full RAM. This measures "pmemsave of
    N MiB" but not "real N-MiB VM" — guest behavior is unaffected. Useful
    as a planning estimate; reboot the VM with the target RAM for a
    definitive measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_timing_instrumentation_experiment as e1
import run_exp2a_consumer_isolation as e2a


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[exp2b] {msg}", file=sys.stderr, flush=True)


def parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Per-cell measurement
# ---------------------------------------------------------------------------

def run_cell(idx: int, total: int, workdir: Path, base_cfg: dict,
             interval_ms: int, ram_mb: int, duration: int,
             producer_script: Path, grace: int, image_dir: Path,
             keep_dumps: bool) -> dict:
    cell_id = f"iv{interval_ms}_ram{ram_mb}"
    cell_dir = workdir / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg_path, _eff = e1.write_overridden_config(base_cfg, cell_dir, interval_ms, ram_mb)
    jsonl_path = cell_dir / "snapshot_timings.jsonl"
    log_path   = cell_dir / "producer.log"

    log(f"[{idx}/{total}] cell {cell_id}: producer ({duration}s)")
    pass_start = time.time()
    proc = e1.start_producer(producer_script, cfg_path, jsonl_path, log_path)
    try:
        time.sleep(duration)
    finally:
        e1.stop_producer(proc, grace)
        e1.resume_vm_if_paused("qemu:///system", base_cfg.get("domain", ""))
    wall = round(time.time() - pass_start, 3)

    snaps, bps = e1.parse_jsonl(jsonl_path)
    records = [e1.build_snapshot_record(i, s, snaps[i+1] if i+1 < len(snaps) else None)
               for i, s in enumerate(snaps)]
    summary = e1.summarize(records, bps)

    # cleanup dumps for this cell
    removed = e1.cleanup_run_dumps(image_dir, pass_start, keep_dumps)

    return {
        "cell_id": cell_id,
        "interval_ms": interval_ms,
        "ram_mb": ram_mb,
        "wall_clock_seconds": wall,
        "dumps_removed": removed,
        "config_path": str(cfg_path),
        "jsonl_path": str(jsonl_path),
        "producer_log": str(log_path),
        **summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_exp2b_interval_sweep.py",
        description=("Experiment 2b: sweep intervalMsec (and optionally pmemsave size) "
                     "to characterise VM pause fraction. Producer-only; consumer killed."),
    )
    p.add_argument("--output-json", default=None)
    p.add_argument("--duration", type=int, default=60,
                   help="Per-cell duration in seconds (default 60).")
    p.add_argument("--intervals", default="100,250,500,1000",
                   help="Comma-separated intervalMsec values (default 100,250,500,1000)")
    p.add_argument("--ram-sweep", default=None,
                   help="Comma-separated pmemsave size in MiB (default: just use the "
                        "config's ramSizeMb). Smaller values simulate smaller VM.")
    p.add_argument("--config", default=str(e1.DEFAULT_CONFIG))
    p.add_argument("--producer-script", default=str(e1.DEFAULT_PRODUCER))
    p.add_argument("--workdir", default=None)
    p.add_argument("--keep-dumps", action="store_true")
    p.add_argument("--purge-stale-dumps", action="store_true",
                   help="Before the sweep, aggressively `sudo rm` ALL "
                        "memory_dump-*.raw files in imageDir.")
    p.add_argument("--per-cell-purge", action="store_true",
                   help="Purge all dumps between every cell (NOT just the one "
                        "created by the previous cell). Use if disk pressure "
                        "is contaminating cells.")
    p.add_argument("--no-self-clean", action="store_true",
                   help="Disable the producer's TIMING_SELF_CLEAN behavior. "
                        "Default is self-clean ON for producer-only timing runs.")
    p.add_argument("--virsh-uri", default="qemu:///system")
    p.add_argument("--no-vm-start", action="store_true")
    p.add_argument("--grace-stop-seconds", type=int, default=10)
    p.add_argument("--inter-cell-cooldown", type=int, default=5,
                   help="Idle seconds between cells (default 5).")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    producer    = Path(args.producer_script).expanduser().resolve()
    if not config_path.is_file(): log(f"ERROR: config not found: {config_path}"); return 2
    if not producer.is_file():    log(f"ERROR: producer not found: {producer}"); return 2

    # Plan 1b: enable producer-side rolling-delete by default.
    if not args.no_self_clean:
        os.environ["TIMING_SELF_CLEAN"] = "1"

    intervals = parse_csv_ints(args.intervals)
    if not intervals:
        log("ERROR: --intervals empty"); return 2
    if args.ram_sweep:
        rams = parse_csv_ints(args.ram_sweep)
    else:
        rams = [None]  # use config's ramSizeMb

    base_cfg = e1.load_config(config_path)
    ram_default = int(base_cfg.get("ramSizeMb", 1024))
    rams_effective = [r if r is not None else ram_default for r in rams]

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else (
        Path.cwd() / "timing_runs" /
        f"exp2b_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_json).expanduser().resolve() if args.output_json else (
        workdir / f"exp2b_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    image_dir = Path(base_cfg.get("imageDir", "/var/lib/libvirt/qemu/dump"))

    cells_plan = [(iv, ram) for iv in intervals for ram in rams_effective]

    result: dict = {
        "experiment": "2b_interval_sweep",
        "timestamp": iso_now(),
        "config": {
            "intervals_ms": intervals,
            "rams_mb": rams_effective,
            "per_cell_duration_sec": args.duration,
            "n_cells": len(cells_plan),
            "vm_domain": base_cfg.get("domain", ""),
            "config_path_input": str(config_path),
            "workdir": str(workdir),
        },
        "cells": [],
        "comparison": {},
        "notes": [],
    }
    if args.ram_sweep:
        result["notes"].append(
            "ram-sweep alters pmemsave argument only; guest VM RAM unchanged. "
            "Pause-fraction estimate is dominated by pmemsave latency which IS "
            "affected. For a real smaller-VM measurement, reboot the guest with "
            "the target RAM and re-run with --ram-sweep matching."
        )

    est_minutes = (args.duration + args.inter_cell_cooldown) * len(cells_plan) / 60.0
    log(f"plan: {len(cells_plan)} cells × {args.duration}s ≈ {est_minutes:.1f} min wall-clock")

    if args.dry_run:
        result["notes"].append("dry-run: no VM operation, no producer started")
        for iv, ram in cells_plan:
            result["cells"].append({
                "cell_id": f"iv{iv}_ram{ram}",
                "interval_ms": iv, "ram_mb": ram,
                "status": "planned",
            })
        with out_path.open("w") as f: json.dump(result, f, indent=2)
        log(f"dry-run plan written to {out_path}")
        return 0

    if not args.no_vm_start:
        state = e1.virsh_start_if_needed(args.virsh_uri, base_cfg.get("domain",""), False)
        if state != "running":
            result["notes"].append(f"VM not running after start attempt: {state!r}")

    queue_dir = Path(base_cfg.get("queueDir","/tmp/queue_dir"))
    killed = e2a.kill_consumer(grace=5)
    drained = e2a.drain_queue(queue_dir)
    purged = 0
    if args.purge_stale_dumps:
        purged = e1.purge_all_dumps(image_dir, use_sudo=True)
        log(f"pre-sweep: purged {purged} stale dump file(s)")
    log(f"pre-sweep: killed {killed} consumer; drained {drained} queue file(s); purged {purged} dump(s)")
    result["pre_sweep"] = {"consumer_processes_killed": killed,
                           "queue_files_drained": drained,
                           "stale_dumps_purged": purged}

    # Disk-free pre-check based on heaviest cell
    heaviest_snaps = max(args.duration // (iv / 1000) for iv in intervals) + 10
    heaviest_ram = max(rams_effective)
    # Plan 4: peak concurrent dumps depends on cleanup policy. With self-clean
    # ON (Plan 1b default) or per-cell purge, only ~2 dumps coexist at any time.
    # With both disabled, the heaviest cell would peak at snapshots_expected.
    self_clean_on = not args.no_self_clean
    peak_dumps = 2 if (self_clean_on or args.per_cell_purge) else int(heaviest_snaps)
    ok_disk, disk_info = e1.disk_free_check(
        image_dir, int(heaviest_snaps), heaviest_ram,
        peak_concurrent_dumps=peak_dumps,
    )
    result["pre_sweep"]["disk_free_check"] = disk_info
    if not ok_disk:
        result["notes"].append(f"WARNING: disk free space {disk_info} may be insufficient")

    for i, (iv, ram) in enumerate(cells_plan, 1):
        # Per-cell prep: drain queue, ensure VM is running, optional dump purge
        d_cell = e2a.drain_queue(queue_dir)
        e1.resume_vm_if_paused(args.virsh_uri, base_cfg.get("domain", ""))
        if args.per_cell_purge:
            n_pre = e1.purge_all_dumps(image_dir, use_sudo=True)
            log(f"[{i}/{len(cells_plan)}] per-cell purge: removed {n_pre} dump(s); drained {d_cell} queue")
        cell = run_cell(i, len(cells_plan), workdir, base_cfg, iv, ram,
                        args.duration, producer, args.grace_stop_seconds,
                        image_dir, args.keep_dumps)
        result["cells"].append(cell)
        if i < len(cells_plan) and args.inter_cell_cooldown > 0:
            time.sleep(args.inter_cell_cooldown)

    # --- comparison: extract key series ordered by interval, then ram ---
    rows = []
    for c in result["cells"]:
        rows.append({
            "cell_id": c["cell_id"],
            "interval_ms": c["interval_ms"],
            "ram_mb": c["ram_mb"],
            "guest_dt_mean":  c.get("mean_guest_run_interval_sec"),
            "host_dt_mean":   c.get("mean_host_snapshot_cycle_sec"),
            "pmemsave_mean":  c.get("mean_pmemsave_sec"),
            "suspend_mean":   c.get("mean_suspend_sec"),
            "pause_fraction": c.get("estimated_vm_pause_fraction"),
            "snapshots":      c.get("snapshots_completed"),
        })
    result["comparison"]["rows"] = rows

    # ranking by pause_fraction ascending (low is better)
    ranked = sorted(
        [r for r in rows if isinstance(r.get("pause_fraction"), (int,float))],
        key=lambda r: r["pause_fraction"],
    )
    result["comparison"]["pause_fraction_ranking"] = ranked[:5]
    result["comparison"]["worst_pause"] = ranked[-1] if ranked else None
    result["comparison"]["best_pause"]  = ranked[0]  if ranked else None

    with out_path.open("w") as f: json.dump(result, f, indent=2)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
