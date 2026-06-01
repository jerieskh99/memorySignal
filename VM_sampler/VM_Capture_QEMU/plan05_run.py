#!/usr/bin/env python3
"""plan05_run.py -- Plan 05 capture-side throughput pilot, lean matrix runner.

Loops the single-run driver ``run_timing_instrumentation_experiment.py`` (e1)
across the pilot matrix and collects one record per cell for the aggregator:

    workloads  x  durations  x  arms  x  replicates

Each cell is a *separate subprocess* invocation of e1 -- the cleanest isolation
for a matrix (fresh os.environ, fresh threads, its own output JSON). This is
exactly "loop the existing driver"; the runner adds no capture logic of its own.

Arms (dump-target lever x keep_dumps lever); all run with e1 --stream-apf so an
``apf_trajectory.jsonl`` exists on every arm (the Plan 05 decoupling):

    ssd_keep        SSD target,   keep dumps      (BASELINE for the gates)
    ssd_selfclean   SSD target,   self-clean
    tmpfs_keep      tmpfs target, keep dumps
    tmpfs_selfclean tmpfs target, self-clean

With --stream-apf on, the producer's APF helper deletes the previous dump after
differencing, so peak on-disk stays ~2-3 dumps on every arm; the keep/self-clean
distinction therefore governs post-run retention, while the dominant throughput
lever is SSD -> tmpfs. (Honest framing: do not read a producer-level throughput
gap into keep vs self-clean while APF streaming is on.)

Actors: this runner is LIVE (starts the VM, runs guest workloads, writes dumps).
It is a SERVER-side tool (Wave 2). The laptop side only unit-tests its pure
helpers (build_cells / cell_to_e1_argv / parse_workload_commands).

Guest workload commands are NOT hardcoded (they are server-specific binary
paths). Supply them like plan02_manifest:  --workload-command name=command
(placeholder literals '<'/'>' are rejected).

Run (server):
    python3 plan05_run.py \
        --workload-command app_hashtable_intensive_v2='/usr/local/bin/app_hashtable_intensive --rounds 8' \
        --workload-command sandbox_scanner_metadata='/usr/local/bin/sandbox_scanner_metadata' \
        --workload-command mem_workingset_sweep_v2='/usr/local/bin/mem_workingset_sweep' \
        --ssh-target kali@192.168.122.42 --tmpfs-dir /dev/shm/plan05_dump \
        --aggregate

Dry-run (laptop, prints the exact per-cell e1 argv, no VM):
    python3 plan05_run.py --dry-run --workload-command app_hashtable_intensive_v2=/bin/true ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_timing_instrumentation_experiment as e1

E1_PATH = Path(__file__).resolve().parent / "run_timing_instrumentation_experiment.py"

# arm name -> (dump target, keep_dumps). 'ssd' = config default imageDir;
# 'tmpfs' = override imageDir to --tmpfs-dir.
ARM_SPECS: dict[str, dict] = {
    "ssd_keep":        {"target": "ssd",   "keep": True},
    "ssd_selfclean":   {"target": "ssd",   "keep": False},
    "tmpfs_keep":      {"target": "tmpfs", "keep": True},
    "tmpfs_selfclean": {"target": "tmpfs", "keep": False},
}

DEFAULT_WORKLOADS = [
    "app_hashtable_intensive_v2",   # steady, high-APF, bimodal
    "sandbox_scanner_metadata",     # phasic, highest SNR
    "mem_workingset_sweep_v2",      # ~1 GiB allocator (RAM-floor stressor)
]
DEFAULT_DURATIONS = [120, 600]
DEFAULT_ARMS = ["ssd_keep", "ssd_selfclean", "tmpfs_keep", "tmpfs_selfclean"]
DEFAULT_TMPFS_DIR = "/dev/shm/plan05_dump"
DEFAULT_REPLICATES = 3
BASELINE_ARM = "ssd_keep"


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[plan05-run] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline -- no VM, no subprocess)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Cell:
    workload: str
    duration_s: int
    arm: str
    replicate: int

    @property
    def cell_id(self) -> str:
        return f"{self.workload}__{self.arm}__d{self.duration_s}__r{self.replicate}"


def parse_workload_commands(specs: list[str]) -> dict[str, str]:
    """Parse ``name=command`` specs into a dict, mirroring plan02_manifest.

    Rejects specs without '=' and placeholder literals '<' / '>' (an operator
    copying a runbook with <RANSOM_PATH>/<VM_IP> should fail loud, not produce
    an unrunnable matrix). Raises ValueError on bad input.
    """
    out: dict[str, str] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--workload-command requires name=command (got: {spec!r})")
        key, _, val = spec.partition("=")
        key, val = key.strip(), val.strip()
        if not key or not val:
            raise ValueError(f"--workload-command has empty name or command: {spec!r}")
        if "<" in val or ">" in val:
            raise ValueError(
                f"--workload-command for {key!r} contains placeholder '<'/'>' : {val!r}. "
                f"Replace with the real binary path.")
        out[key] = val
    return out


def inject_duration(cmd: str, duration_s: int) -> str:
    """Idempotently append --phase-markers and --duration (mirror of
    plan02_run._augment_workload_command, inlined to keep this runner standalone
    and free of the heavy orchestrator import)."""
    if "--phase-markers" not in cmd:
        cmd = cmd + " --phase-markers"
    if "--duration" not in cmd:
        cmd = cmd + f" --duration {duration_s}"
    return cmd


def build_cells(workloads: list[str], durations: list[int], arms: list[str],
                replicates: int) -> list[Cell]:
    """Expand the matrix. Order: workload -> duration -> arm -> replicate."""
    cells: list[Cell] = []
    for w in workloads:
        for d in durations:
            for a in arms:
                for r in range(replicates):
                    cells.append(Cell(w, d, a, r))
    return cells


def cell_to_e1_argv(cell: Cell, *, config: str, tmpfs_dir: str,
                    cell_dir: Path, out_json: Path,
                    command: str | None, ssh_target: str, ssh_key: str,
                    ssh_opts: str, ram_mb: int | None, virsh_uri: str,
                    producer_script: str | None,
                    no_vm_start: bool, skip_ram_check: bool,
                    grace_stop_seconds: int) -> list[str]:
    """Build the exact e1 CLI argv for one cell. Pure (no I/O)."""
    spec = ARM_SPECS[cell.arm]
    argv = [
        "--config", config,
        "--duration", str(cell.duration_s),
        "--workdir", str(cell_dir),
        "--output-json", str(out_json),
        "--virsh-uri", virsh_uri,
        "--grace-stop-seconds", str(grace_stop_seconds),
        "--stream-apf",            # Plan 05: APF trajectory on every arm
        "--instrument-peak-disk",  # Plan 05: G-T3 peak live-dump bytes
    ]
    if spec["target"] == "tmpfs":
        argv += ["--image-dir", tmpfs_dir]
    if spec["keep"]:
        argv += ["--keep-dumps"]
    if producer_script:
        argv += ["--producer-script", producer_script]
    if command:
        argv += ["--test-command", inject_duration(command, cell.duration_s)]
        if ssh_target:
            argv += ["--ssh-target", ssh_target]
        if ssh_key:
            argv += ["--ssh-key", ssh_key]
        if ssh_opts:
            argv += ["--ssh-opts", ssh_opts]
    if ram_mb is not None:
        argv += ["--ram-mb", str(ram_mb)]
    if no_vm_start:
        argv += ["--no-vm-start"]
    if skip_ram_check:
        argv += ["--skip-ram-check"]
    return argv


def record_from_run_json(cell: Cell, run_json: dict) -> dict:
    """Project an e1 output JSON into one aggregator input record."""
    summary = run_json.get("summary") or {}
    return {
        "workload": cell.workload,
        "duration_s": cell.duration_s,
        "arm": cell.arm,
        "replicate": cell.replicate,
        "mean_pmemsave_sec": summary.get("mean_pmemsave_sec"),
        "snapshots_completed": summary.get("snapshots_completed"),
        "peak_dump_bytes": summary.get("peak_dump_bytes"),
        "apf_jsonl": summary.get("apf_trajectory"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plan05_run.py",
        description="Plan 05 capture-side throughput pilot: loop e1 over "
                    "workloads x durations x arms x replicates.",
    )
    p.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS),
                   help="Comma-separated workload labels (default: the 3 pilot workloads).")
    p.add_argument("--workload-command", action="append", default=[],
                   metavar="NAME=COMMAND",
                   help="Guest command for a workload label (repeatable). "
                        "Placeholder literals '<'/'>' are rejected.")
    p.add_argument("--durations", default=",".join(str(d) for d in DEFAULT_DURATIONS),
                   help="Comma-separated per-cell durations in seconds (default 120,600).")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                   help=f"Comma-separated arms from {sorted(ARM_SPECS)} "
                        f"(default: all four).")
    p.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES,
                   help="Replicates per (workload,duration,arm) cell (default 3).")
    p.add_argument("--tmpfs-dir", default=DEFAULT_TMPFS_DIR,
                   help=f"imageDir for tmpfs arms (default {DEFAULT_TMPFS_DIR}). "
                        "Must be a real tmpfs mount on the server.")
    p.add_argument("--ssh-target", default="", help="user@host for the guest workload.")
    p.add_argument("--ssh-key", default="", help="SSH private key path.")
    p.add_argument("--ssh-opts", default="", help="Extra SSH options (single string).")
    p.add_argument("--config", default=str(e1.DEFAULT_CONFIG))
    p.add_argument("--producer-script", default=None)
    p.add_argument("--ram-mb", type=int, default=None,
                   help="Optional ramSizeMb override passed to every cell. Default: "
                        "use the config value (and let e1 verify it vs the live domain).")
    p.add_argument("--virsh-uri", default="qemu:///system")
    p.add_argument("--workdir", default=None)
    p.add_argument("--output-json", default=None,
                   help="Where to write the matrix record file. Default: under --workdir.")
    p.add_argument("--grace-stop-seconds", type=int, default=10)
    p.add_argument("--inter-cell-cooldown", type=int, default=5,
                   help="Idle seconds between cells (default 5).")
    p.add_argument("--skip-ram-check", action="store_true",
                   help="Skip the one-shot RAM preflight (passed through to cells too).")
    p.add_argument("--no-vm-start", action="store_true",
                   help="Assume the VM is already running; do not start it.")
    p.add_argument("--aggregate", action="store_true",
                   help="After the matrix, run plan05_aggregate over the collected "
                        "records and write plan05_summary.json next to the output.")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter used to launch each e1 subprocess.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned cells and the exact per-cell e1 argv; "
                        "do not touch the VM or run any producer.")
    return p


def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        log(f"ERROR: config not found: {config_path}")
        return 2

    workloads = _parse_list(args.workloads)
    durations = [int(x) for x in _parse_list(args.durations)]
    arms = _parse_list(args.arms)
    bad_arms = [a for a in arms if a not in ARM_SPECS]
    if bad_arms:
        log(f"ERROR: unknown arm(s) {bad_arms}; choose from {sorted(ARM_SPECS)}")
        return 2
    try:
        wcmds = parse_workload_commands(args.workload_command)
    except ValueError as e:
        log(f"ERROR: {e}")
        return 2

    cells = build_cells(workloads, durations, arms, args.replicates)

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else (
        Path.cwd() / "plan05_runs" /
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_json).expanduser().resolve() if args.output_json else (
        workdir / "plan05_run_records.json"
    )

    base_cfg = e1.load_config(config_path)
    vm_domain = base_cfg.get("domain", "")
    ram_effective = int(args.ram_mb if args.ram_mb is not None
                        else base_cfg.get("ramSizeMb", 0))

    result: dict = {
        "experiment": "plan05_throughput_pilot",
        "timestamp": iso_now(),
        "config": {
            "workloads": workloads,
            "durations_s": durations,
            "arms": arms,
            "replicates": args.replicates,
            "n_cells": len(cells),
            "baseline_arm": BASELINE_ARM,
            "tmpfs_dir": args.tmpfs_dir,
            "ram_mb_effective": ram_effective,
            "vm_domain": vm_domain,
            "config_path": str(config_path),
            "workdir": str(workdir),
        },
        "cells": [],
        "records": [],
        "notes": [],
    }
    if BASELINE_ARM not in arms:
        result["notes"].append(
            f"baseline arm {BASELINE_ARM!r} not in --arms; the aggregator will "
            f"have no baseline and skip all comparisons.")

    missing_cmd = [w for w in workloads if w not in wcmds]
    if missing_cmd and not args.dry_run:
        result["notes"].append(
            f"no --workload-command for {missing_cmd}; those cells run with an "
            f"IDLE guest (timing still measured, but not the intended workload).")

    est_min = sum(d + args.inter_cell_cooldown for c in cells
                  for d in [c.duration_s]) / 60.0
    log(f"plan: {len(cells)} cells across {len(workloads)} workloads x "
        f"{len(durations)} durations x {len(arms)} arms x {args.replicates} reps "
        f"~ {est_min:.1f} min live wall-clock")

    # ---- dry-run: emit the planned argv per cell, no VM ----
    if args.dry_run:
        for c in cells:
            cell_dir = workdir / c.cell_id
            run_json = cell_dir / "run_record.json"
            cmd = wcmds.get(c.workload)
            cargv = cell_to_e1_argv(
                c, config=str(config_path), tmpfs_dir=args.tmpfs_dir,
                cell_dir=cell_dir, out_json=run_json, command=cmd,
                ssh_target=args.ssh_target, ssh_key=args.ssh_key,
                ssh_opts=args.ssh_opts, ram_mb=args.ram_mb,
                virsh_uri=args.virsh_uri, producer_script=args.producer_script,
                no_vm_start=True, skip_ram_check=True,
                grace_stop_seconds=args.grace_stop_seconds)
            result["cells"].append({
                "cell_id": c.cell_id, "workload": c.workload,
                "duration_s": c.duration_s, "arm": c.arm, "replicate": c.replicate,
                "status": "planned",
                "e1_argv": [args.python, str(E1_PATH), *cargv],
            })
        result["notes"].append("dry-run: no VM operation, no producer started")
        _write_json(out_path, result)
        log(f"dry-run plan ({len(cells)} cells) written to {out_path}")
        return 0

    # ---- live: start VM once, RAM preflight once, then loop cells ----
    if not args.no_vm_start:
        state = e1.virsh_start_if_needed(args.virsh_uri, vm_domain, dry_run=False)
        if state != "running":
            result["notes"].append(f"VM not running after start attempt: {state!r}")
        else:
            log(f"VM {vm_domain!r} running")

    if not args.skip_ram_check and ram_effective > 0:
        ok, detail = e1.ram_size_matches_domain(args.virsh_uri, vm_domain, ram_effective)
        result["config"]["ram_check"] = detail
        if ok is False:
            dom_mb = detail.get("domain_max_mb") or 0.0
            log(f"ERROR: RAM mismatch -- config ramSizeMb={ram_effective} vs live "
                f"domain max {dom_mb:.0f} MiB. Resize the domain (setmaxmem + cold "
                f"reboot) or pass --skip-ram-check. Aborting matrix.")
            result["notes"].append("aborted: RAM preflight mismatch")
            _write_json(out_path, result)
            return 4
        if ok is None:
            result["notes"].append(detail.get("note", "RAM check could not verify"))

    if any(ARM_SPECS[a]["target"] == "tmpfs" for a in arms):
        Path(args.tmpfs_dir).expanduser().mkdir(parents=True, exist_ok=True)
        result["notes"].append(
            f"tmpfs arms write to {args.tmpfs_dir}; verify it is a real tmpfs "
            f"mount (df -T) before trusting the throughput delta.")

    records: list[dict] = []
    for i, c in enumerate(cells, 1):
        cell_dir = workdir / c.cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        run_json = cell_dir / "run_record.json"
        cmd = wcmds.get(c.workload)
        cargv = cell_to_e1_argv(
            c, config=str(config_path), tmpfs_dir=args.tmpfs_dir,
            cell_dir=cell_dir, out_json=run_json, command=cmd,
            ssh_target=args.ssh_target, ssh_key=args.ssh_key,
            ssh_opts=args.ssh_opts, ram_mb=args.ram_mb,
            virsh_uri=args.virsh_uri, producer_script=args.producer_script,
            no_vm_start=True,        # VM already started by the runner
            skip_ram_check=True,     # RAM already checked once above
            grace_stop_seconds=args.grace_stop_seconds)
        full = [args.python, str(E1_PATH), *cargv]
        log(f"[{i}/{len(cells)}] {c.cell_id} ({c.duration_s}s)")
        proc = subprocess.run(full)
        cell_rec = {"cell_id": c.cell_id, "workload": c.workload,
                    "duration_s": c.duration_s, "arm": c.arm,
                    "replicate": c.replicate, "returncode": proc.returncode,
                    "run_json": str(run_json)}
        if proc.returncode == 0 and run_json.is_file():
            try:
                rj = json.loads(run_json.read_text())
                rec = record_from_run_json(c, rj)
                records.append(rec)
                cell_rec["mean_pmemsave_sec"] = rec["mean_pmemsave_sec"]
                cell_rec["snapshots_completed"] = rec["snapshots_completed"]
                cell_rec["peak_dump_bytes"] = rec["peak_dump_bytes"]
            except (OSError, json.JSONDecodeError) as e:
                cell_rec["error"] = f"failed to read run json: {e}"
        else:
            cell_rec["error"] = f"e1 exit {proc.returncode} or missing {run_json.name}"
        result["cells"].append(cell_rec)
        if i < len(cells) and args.inter_cell_cooldown > 0:
            time.sleep(args.inter_cell_cooldown)

    result["records"] = records
    _write_json(out_path, result)
    log(f"wrote {out_path} ({len(records)}/{len(cells)} cells produced a record)")

    if args.aggregate:
        import plan05_aggregate as agg
        summary = agg.aggregate(records, baseline_arm=BASELINE_ARM)
        summary_path = out_path.parent / "plan05_summary.json"
        agg._atomic_json(summary, summary_path)
        log(f"wrote {summary_path} (aggregated {len(records)} records)")

    return 0


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


if __name__ == "__main__":
    sys.exit(main())
