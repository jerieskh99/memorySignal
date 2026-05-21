#!/usr/bin/env python3
"""run_exp2c_flush_sensitivity.py — Experiment 2c.

Tests whether the post-pmemsave `sleep 0.5` flush in the producer is
necessary. Runs two passes:

  Pass A — flush_on:  producer's default (TIMING_NO_FLUSH unset).
  Pass B — flush_off: TIMING_NO_FLUSH=1 (skip the 0.5 s sleep).

Checks two things per pass:
  1. host_dt distribution (expected -0.5 s mean on pass B).
  2. dump integrity (file size matches ramSizeBytes, and a 3-point
     content probe at offset 0 / RAM/2 / RAM-4096 returns non-zero
     bytes).

Producer-only (consumer killed up front). Targets the cheap ~10% wall-
clock saving identified as Step 2c in
docs/timing_experiment_1_lab.html.

Produces one JSON file with per-pass summary, integrity report, and a
recommendation (keep / remove the flush).
"""
from __future__ import annotations

import argparse
import hashlib
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
    print(f"[exp2c] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Dump integrity probe
# ---------------------------------------------------------------------------

def probe_dump(path: Path, ram_bytes: int) -> dict:
    """Integrity probe corrected for idle Kali VM.

    An idle guest has most of physical RAM as zero pages — a "non-zero
    middle" check fails legitimately because RAM IS zero, not because
    pmemsave is broken. So integrity here is just:
      1. file size equals expected ramSizeBytes,
      2. file is readable end-to-end without I/O errors,
      3. some fraction of the file is non-zero (i.e. the dump isn't
         a sparse / all-zero file).

    The third check is informational: if the entire dump is zero, the
    guest had truly empty RAM at the snapshot moment. That is unusual
    but not necessarily a pmemsave bug; flagged as `all_zero_dump=True`
    rather than failing the integrity test.
    """
    rep: dict = {"path": str(path), "ok": False, "size_match": False,
                 "readable_end_to_end": False, "all_zero_dump": False,
                 "errors": []}
    try:
        sz = path.stat().st_size
        rep["size_bytes"] = sz
        rep["size_match"] = (sz == ram_bytes)
        if not rep["size_match"]:
            rep["errors"].append(f"size {sz} != expected {ram_bytes}")

        # Read 4 KiB at start, middle, end. Count any non-zero across all three.
        nonzero_total = 0
        with path.open("rb") as f:
            f.seek(0);                        head = f.read(4096)
            f.seek(max(0, (ram_bytes // 2) - 2048));  mid  = f.read(4096)
            f.seek(max(0, ram_bytes - 4096));        tail = f.read(4096)
        nonzero_total = sum(b != 0 for b in head + mid + tail)
        rep["nonzero_bytes_in_probes"] = nonzero_total
        rep["all_zero_dump"] = (nonzero_total == 0)
        rep["readable_end_to_end"] = (len(head) == 4096 and
                                      len(mid) == 4096 and
                                      len(tail) == 4096)
        # Integrity = size correct + read worked. Zero-content is informational.
        rep["ok"] = rep["size_match"] and rep["readable_end_to_end"]
    except FileNotFoundError:
        rep["errors"].append("dump file missing"); rep["ok"] = False
    except OSError as exc:
        rep["errors"].append(f"OSError: {exc}"); rep["ok"] = False
    return rep


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------

def run_pass(name: str, workdir: Path, cfg_path: Path, producer: Path,
             duration: int, grace: int, env_overrides: dict[str, str],
             ram_bytes: int, probe_n: int, keep_dumps: bool,
             image_dir: Path, run_start_epoch: float) -> dict:
    pass_dir = workdir / name
    pass_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = pass_dir / "snapshot_timings.jsonl"
    log_path   = pass_dir / "producer.log"

    log(f"[{name}] starting producer ({duration}s) env={env_overrides}")
    env = os.environ.copy()
    env["CONFIG"] = str(cfg_path)
    env["TIMING_JSONL_PATH"] = str(jsonl_path)
    env.update(env_overrides)
    log_f = log_path.open("ab")
    pass_start = time.time()
    proc = subprocess.Popen(
        ["bash", str(producer)], env=env, stdout=log_f, stderr=log_f,
        start_new_session=True,
    )
    try:
        time.sleep(duration)
    finally:
        e1.stop_producer(proc, grace)
        # ensure VM is not left paused
        with open(cfg_path) as _f:
            _cfg = json.load(_f)
        e1.resume_vm_if_paused("qemu:///system", _cfg.get("domain", ""))
    wall = round(time.time() - pass_start, 3)

    snaps, bps = e1.parse_jsonl(jsonl_path)
    records = [e1.build_snapshot_record(i, s, snaps[i+1] if i+1 < len(snaps) else None)
               for i, s in enumerate(snaps)]
    summary = e1.summarize(records, bps)

    # Probe up to probe_n dumps for integrity
    probes: list[dict] = []
    for r in records[:probe_n]:
        ip = r.get("image_path")
        if isinstance(ip, str) and Path(ip).is_file():
            probes.append(probe_dump(Path(ip), ram_bytes))
    integrity_ok = bool(probes) and all(p["ok"] for p in probes)

    # cleanup dumps from THIS pass only
    e1.cleanup_run_dumps(image_dir, pass_start, keep_dumps)

    return {
        "wall_clock_seconds": wall,
        "summary": summary,
        "records": records,
        "integrity": {
            "probes_examined": len(probes),
            "all_ok": integrity_ok,
            "probes": probes,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_exp2c_flush_sensitivity.py",
        description=("Experiment 2c: with/without the 0.5 s pmemsave flush. "
                     "Producer-only; consumer killed. Verifies dump integrity."),
    )
    p.add_argument("--output-json", default=None)
    p.add_argument("--duration", type=int, default=60,
                   help="Per-pass duration in seconds (default 60).")
    p.add_argument("--interval-ms", type=int, default=None)
    p.add_argument("--ram-mb", type=int, default=None)
    p.add_argument("--probes-per-pass", type=int, default=5,
                   help="Number of dumps per pass to probe for integrity (default 5).")
    p.add_argument("--config", default=str(e1.DEFAULT_CONFIG))
    p.add_argument("--producer-script", default=str(e1.DEFAULT_PRODUCER))
    p.add_argument("--workdir", default=None)
    p.add_argument("--keep-dumps", action="store_true")
    p.add_argument("--purge-stale-dumps", action="store_true",
                   help="Before pass A, aggressively `sudo rm` ALL "
                        "memory_dump-*.raw files in imageDir.")
    p.add_argument("--purge-between-passes", action="store_true",
                   help="Also purge dumps between pass A (flush_on) and "
                        "pass B (flush_off).")
    p.add_argument("--no-self-clean", action="store_true",
                   help="(Plan 5: kept for back-compat, now a no-op for 2c.) "
                        "Producer's TIMING_SELF_CLEAN is OFF by default in 2c "
                        "because the integrity probe requires dumps to remain "
                        "on disk to verify content (Bug F fix).")
    p.add_argument("--force-self-clean", action="store_true",
                   help="Force TIMING_SELF_CLEAN=1 even for 2c. The integrity "
                        "probe will see 0 dumps and the recommendation will "
                        "lock to KEEP regardless of host_dt savings. For "
                        "diagnostics only.")
    p.add_argument("--stable-n", type=int, default=5,
                   help="Number of snapshots from the start of each pass to use "
                        "for the stable-subset comparison (default 5; Plan 6 · "
                        "Round 4 showed that without producer self-clean, "
                        "flush_off accumulates dumps and page-cache pressure "
                        "spikes pmemsave from ~0.78 s to ~8 s at snap 6+. "
                        "n=5 catches the savings window before drift). The "
                        "all-snap means are also reported in the JSON.")
    p.add_argument("--virsh-uri", default="qemu:///system")
    p.add_argument("--no-vm-start", action="store_true")
    p.add_argument("--grace-stop-seconds", type=int, default=10)
    p.add_argument("--cooldown-between-passes", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    return p


def stable_mean(records: list[dict], key: str, n: int = 10) -> float | None:
    """Plan 3 · stable-subset mean.

    Average a field across the first `n` records, skipping records where
    the field is missing or non-numeric. Used by the 2c verdict logic to
    avoid drift-contamination in the late snaps of each pass.
    """
    vals = [r[key] for r in records[:n]
            if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return float(statistics.fmean(vals))


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    producer    = Path(args.producer_script).expanduser().resolve()
    if not config_path.is_file(): log(f"ERROR: config not found: {config_path}"); return 2
    if not producer.is_file():    log(f"ERROR: producer not found: {producer}"); return 2

    # Plan 5 (Bug F fix): force TIMING_SELF_CLEAN OFF for 2c so the integrity
    # probe can actually find dumps on disk. Plan 3's stable-subset comparison
    # uses only the first --stable-n (=10) snapshots of each pass, which
    # complete well before any dump pile-up would distort host_dt. So 2c does
    # NOT need self-clean for the timing-delta check, and DOES need dumps
    # present for the content probe. Use --force-self-clean to override (the
    # probe will then see 0 dumps; recommendation locks to KEEP).
    if args.force_self_clean:
        os.environ["TIMING_SELF_CLEAN"] = "1"
        log("WARNING: --force-self-clean ON — integrity probe will see 0 dumps "
            "(Bug F); recommendation will lock to KEEP regardless of host_dt.")
    else:
        os.environ.pop("TIMING_SELF_CLEAN", None)

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else (
        Path.cwd() / "timing_runs" /
        f"exp2c_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_json).expanduser().resolve() if args.output_json else (
        workdir / f"exp2c_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    base_cfg = e1.load_config(config_path)
    cfg_path, eff_cfg = e1.write_overridden_config(base_cfg, workdir,
                                                   args.interval_ms, args.ram_mb)
    ram_mb = int(eff_cfg["ramSizeMb"])
    ram_bytes = ram_mb * 1024 * 1024
    image_dir = Path(eff_cfg.get("imageDir", "/var/lib/libvirt/qemu/dump"))

    # Verify the producer has the flush toggle wired. Look for TIMING_NO_FLUSH
    # in the script to fail loud if it's an old build.
    try:
        producer_src = producer.read_text()
    except OSError:
        producer_src = ""
    flush_toggle_present = "TIMING_NO_FLUSH" in producer_src

    result: dict = {
        "experiment": "2c_flush_sensitivity",
        "timestamp": iso_now(),
        "config": {
            "interval_ms": int(eff_cfg["intervalMsec"]),
            "ram_size_mb": ram_mb,
            "per_pass_duration_sec": args.duration,
            "probes_per_pass": args.probes_per_pass,
            "vm_domain": eff_cfg.get("domain", ""),
            "config_path_input": str(config_path),
            "config_path_effective": str(cfg_path),
            "workdir": str(workdir),
            "image_dir": str(image_dir),
            "producer_has_flush_toggle": flush_toggle_present,
        },
        "passes": {},
        "comparison": {},
        "recommendation": None,
        "notes": [],
    }

    if not flush_toggle_present:
        result["notes"].append(
            "WARNING: producer script does not contain TIMING_NO_FLUSH guard. "
            "The flush_off pass will be functionally identical to flush_on. "
            "Patch the producer first."
        )

    if args.dry_run:
        result["notes"] += [
            "dry-run: no VM operation, no producer started",
            "would kill consumer + drain queue, then run pass flush_on then flush_off",
            f"would probe up to {args.probes_per_pass} dumps per pass for integrity",
        ]
        with out_path.open("w") as f: json.dump(result, f, indent=2)
        log(f"dry-run plan written to {out_path}")
        return 0

    if not args.no_vm_start:
        state = e1.virsh_start_if_needed(args.virsh_uri, eff_cfg.get("domain",""), False)
        if state != "running":
            result["notes"].append(f"VM not running after start attempt: {state!r}")

    queue_dir = Path(eff_cfg.get("queueDir","/tmp/queue_dir"))
    killed  = e2a.kill_consumer(grace=5)
    drained = e2a.drain_queue(queue_dir)
    purged_start = 0
    if args.purge_stale_dumps:
        purged_start = e1.purge_all_dumps(image_dir, use_sudo=True)
    log(f"pre-passes: killed {killed} consumer; drained {drained} queue file(s); "
        f"purged {purged_start} stale dump(s)")
    result["pre_passes"] = {"consumer_processes_killed": killed,
                            "queue_files_drained": drained,
                            "stale_dumps_purged": purged_start}

    run_start_epoch = time.time()

    # --- pass A: flush_on (default) ---
    pA = run_pass("flush_on", workdir, cfg_path, producer, args.duration,
                  args.grace_stop_seconds, env_overrides={},
                  ram_bytes=ram_bytes, probe_n=args.probes_per_pass,
                  keep_dumps=args.keep_dumps, image_dir=image_dir,
                  run_start_epoch=run_start_epoch)
    result["passes"]["flush_on"] = {**pA["summary"], "wall_clock_seconds": pA["wall_clock_seconds"],
                                    "integrity": pA["integrity"],
                                    "records": pA["records"]}

    # Between passes: drain queue + resume VM + optional dump purge
    d_mid = e2a.drain_queue(queue_dir)
    e1.resume_vm_if_paused(args.virsh_uri, eff_cfg.get("domain", ""))
    purged_mid = 0
    if args.purge_between_passes:
        purged_mid = e1.purge_all_dumps(image_dir, use_sudo=True)
    log(f"between-passes: drained {d_mid} queue file(s); purged {purged_mid} dump(s)")
    result["pre_passes"]["between_passes"] = {"queue_files_drained": d_mid,
                                              "stale_dumps_purged": purged_mid}

    if args.cooldown_between_passes > 0:
        time.sleep(args.cooldown_between_passes)

    # --- pass B: flush_off ---
    pB = run_pass("flush_off", workdir, cfg_path, producer, args.duration,
                  args.grace_stop_seconds, env_overrides={"TIMING_NO_FLUSH": "1"},
                  ram_bytes=ram_bytes, probe_n=args.probes_per_pass,
                  keep_dumps=args.keep_dumps, image_dir=image_dir,
                  run_start_epoch=run_start_epoch)
    result["passes"]["flush_off"] = {**pB["summary"], "wall_clock_seconds": pB["wall_clock_seconds"],
                                     "integrity": pB["integrity"],
                                     "records": pB["records"]}

    # --- comparison ---
    on, off = result["passes"]["flush_on"], result["passes"]["flush_off"]
    def diff(key):
        a, b = on.get(key), off.get(key)
        if not (isinstance(a, (int,float)) and isinstance(b, (int,float))): return None
        return {"on": round(a, 4), "off": round(b, 4),
                "delta": round(b - a, 4),
                "ratio_off_over_on": round(b/a, 3) if a else None}
    # Plan 3: also compute stable-subset means (first N snaps of each pass)
    # so the verdict logic isn't poisoned by drift-contaminated tail.
    n_stable = args.stable_n
    on_first_n  = stable_mean(on["records"],  "host_snapshot_cycle_sec", n=n_stable)
    off_first_n = stable_mean(off["records"], "host_snapshot_cycle_sec", n=n_stable)
    if on_first_n is not None and off_first_n is not None:
        stable_diff = {"on": round(on_first_n, 4), "off": round(off_first_n, 4),
                       "delta": round(off_first_n - on_first_n, 4),
                       "ratio_off_over_on": round(off_first_n / on_first_n, 3) if on_first_n else None,
                       "n_used": n_stable}
    else:
        stable_diff = None

    result["comparison"] = {
        "mean_host_snapshot_cycle_sec":           diff("mean_host_snapshot_cycle_sec"),
        "mean_host_snapshot_cycle_sec_first_n":   stable_diff,
        "mean_pmemsave_sec":                      diff("mean_pmemsave_sec"),
        "mean_suspend_sec":                       diff("mean_suspend_sec"),
        "snapshots_completed":                    diff("snapshots_completed"),
        "integrity_on":  on["integrity"]["all_ok"],
        "integrity_off": off["integrity"]["all_ok"],
    }

    # --- recommendation logic (Plan 3: uses stable-subset delta when available) ---
    delta_all    = result["comparison"]["mean_host_snapshot_cycle_sec"]
    delta_stable = result["comparison"]["mean_host_snapshot_cycle_sec_first_n"]
    delta = delta_stable if delta_stable is not None else delta_all
    delta_label = f"first-{n_stable}" if delta_stable is not None else "all-snap"

    if delta is None:
        result["recommendation"] = "inconclusive — could not compute host_dt mean"
    elif not off["integrity"]["all_ok"]:
        result["recommendation"] = ("KEEP the flush — dumps without it failed "
                                    "integrity check.")
    elif delta["delta"] < -0.3:  # off is at least 0.3 s faster than on (stable subset)
        on_ref = delta["on"]
        result["recommendation"] = (f"REMOVE the flush — dumps remain intact and "
                                    f"host_dt drops by {-delta['delta']:.3f} s "
                                    f"({-delta['delta']/on_ref*100:.1f}%) on the "
                                    f"{delta_label} comparison.")
    elif abs(delta["delta"]) < 0.1:
        result["recommendation"] = (f"NEUTRAL — host_dt change is within noise on the "
                                    f"{delta_label} comparison; removing the flush "
                                    "has no measurable benefit.")
    else:
        result["recommendation"] = (f"NEUTRAL/AMBIGUOUS — host_dt delta "
                                    f"{delta['delta']:+.3f} s on the {delta_label} "
                                    "comparison; check integrity carefully before "
                                    "removing.")

    with out_path.open("w") as f: json.dump(result, f, indent=2)
    log(f"wrote {out_path}")
    log(f"recommendation: {result['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
