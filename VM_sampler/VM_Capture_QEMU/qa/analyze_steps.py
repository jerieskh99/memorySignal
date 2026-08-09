#!/usr/bin/env python3
"""Static validation of a campaign steps file against measured guest capacity.

Every failure this project hit in the 2026-08-09 campaign was visible in the
steps file itself, given the guest's RAM and scratch-filesystem facts:

  cache_cold_scan_v2 --working-set-mb 2048   ->  mmap fails on a 964 MB guest
  mem_mmap_traversal --file-size-mb 1024     ->  exceeds a 483 MB tmpfs
  scratch under /tmp (tmpfs)                 ->  file writes land in RAM, so an
                                                 "IO" workload measures memory

So this runs before a campaign, not after. It parses each step, classifies its
resource demands, and compares them to facts probed from the live guest.

Usage:
  analyze_steps.py STEPS_FILE --facts FACTS_FILE [--scratch-facts K=V ...] [--json]

Exit: 0 all clear, 1 findings at FAIL severity, 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

# Flags whose value is an anonymous-memory allocation: bounded by guest RAM.
RAM_FLAGS = ("--working-set-mb", "--mem-cap-mb")
# Flags whose value becomes file bytes in the scratch filesystem.
FILE_MB_FLAGS = ("--file-size-mb", "--input-size-mb", "--output-size-mb")
# Flags naming a scratch directory.
DIR_FLAGS = ("--sandbox-dir", "--backing-dir", "--output-dir", "--inputs-dir")
# Directory flags that hold a real payload (not just a few KB of metadata).
PAYLOAD_DIR_FLAGS = ("--sandbox-dir", "--backing-dir", "--inputs-dir")

# RAM the guest OS keeps for itself. The budget is derived from TOTAL guest RAM
# minus this, not from MemAvailable at probe time: the orchestrator reboots the
# guest before every step, so a workload runs against a freshly booted system
# with far more free memory than a probe taken mid-session would suggest.
# Calibrated against observed runs on a 964 MB guest: 384 MB ran fine (94
# snapshots), 512 MB ran fine, 1024 MB swapped badly, 2048 MB failed outright.
GUEST_OS_RESERVE_MB = 300
RAM_WARN_FRACTION = 0.60       # above this share of the budget, expect swapping
SCRATCH_HEADROOM_FRACTION = 0.80


def parse_step(line: str) -> dict:
    try:
        toks = shlex.split(line)
    except ValueError:
        toks = line.split()

    binaries = [t for t in toks if "/" in t and ("/bin/" in t or t.endswith((".py", ".sh")))]
    name = Path(binaries[0]).name if binaries else (toks[0] if toks else "?")

    ram_mb = 0
    file_mb = 0
    dirs: dict[str, str] = {}
    files_count = 0
    file_bytes = 0

    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if nxt is None:
            continue
        if t in RAM_FLAGS:
            ram_mb = max(ram_mb, int(nxt) if nxt.isdigit() else 0)
        elif t in FILE_MB_FLAGS:
            file_mb += int(nxt) if nxt.isdigit() else 0
        elif t in DIR_FLAGS:
            dirs[t] = nxt
        elif t == "--files":
            files_count = int(nxt) if nxt.isdigit() else 0
        elif t == "--file-size-bytes":
            file_bytes = int(nxt) if nxt.isdigit() else 0

    # Workloads that materialise N files of M bytes (the ransomware sims) declare
    # their footprint that way rather than with a --file-size-mb flag.
    if files_count and file_bytes:
        file_mb += (files_count * file_bytes) // (1024 * 1024)

    return {
        "name": name,
        "binaries": binaries,
        "ram_mb": ram_mb,
        "file_mb": file_mb,
        "dirs": dirs,
        "payload_dirs": {k: v for k, v in dirs.items() if k in PAYLOAD_DIR_FLAGS},
    }


def load_facts(path: str) -> dict:
    facts = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if ln and "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            facts[k.strip()] = v.strip()
    return facts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("--facts", required=True, help="KEY=VALUE file from probe_guest_facts")
    ap.add_argument("--scratch", action="append", default=[],
                    help="DIR:type=T,avail_mb=N  (repeatable), from probe_guest_dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    facts = load_facts(a.facts)
    ram_avail = int(facts.get("GUEST_AVAIL_MB", 0) or 0)
    ram_total = int(facts.get("GUEST_RAM_MB", 0) or 0)

    scratch: dict[str, dict] = {}
    for spec in a.scratch:
        d, _, kvs = spec.partition(":")
        entry = {}
        for kv in kvs.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                entry[k.strip()] = v.strip()
        scratch[d] = entry

    def scratch_for(path: str) -> tuple[str, dict] | tuple[None, None]:
        best = None
        for d in scratch:
            if path == d or path.startswith(d.rstrip("/") + "/"):
                if best is None or len(d) > len(best):
                    best = d
        return (best, scratch[best]) if best else (None, None)

    lines = [l.strip() for l in Path(a.steps).read_text().splitlines()
             if l.strip() and not l.strip().startswith("#")]

    findings = []
    ram_budget = max(ram_total - GUEST_OS_RESERVE_MB, 0) if ram_total else 0

    for idx, line in enumerate(lines, start=1):
        s = parse_step(line)

        # 1. Anonymous memory vs what a freshly booted guest can give out.
        if ram_budget and s["ram_mb"] > ram_budget:
            findings.append({
                "severity": "FAIL", "step": idx, "workload": s["name"], "kind": "ram",
                "detail": f"needs {s['ram_mb']} MB anonymous memory; a freshly booted guest "
                          f"offers about {ram_budget} MB ({ram_total} MB total less "
                          f"{GUEST_OS_RESERVE_MB} MB for the OS). mmap will fail (exit 1)."})
        elif ram_budget and s["ram_mb"] > ram_budget * RAM_WARN_FRACTION:
            findings.append({
                "severity": "WARN", "step": idx, "workload": s["name"], "kind": "ram",
                "detail": f"{s['ram_mb']} MB is over {int(RAM_WARN_FRACTION*100)}% of the "
                          f"~{ram_budget} MB budget; expect some swapping, and mlock() will "
                          f"fail (best-effort in these workloads, so not fatal)."})

        # 2. Scratch location: type and capacity.
        for flag, d in s["dirs"].items():
            root, sf = scratch_for(d)
            if sf is None:
                findings.append({
                    "severity": "WARN", "step": idx, "workload": s["name"], "kind": "scratch",
                    "detail": f"{flag} {d} was not probed; capacity unverified."})
                continue

            is_payload = flag in PAYLOAD_DIR_FLAGS or s["file_mb"] > 0
            if sf.get("type") == "tmpfs" and is_payload:
                findings.append({
                    "severity": "FAIL", "step": idx, "workload": s["name"], "kind": "tmpfs",
                    "detail": f"{flag} {d} is on tmpfs (RAM-backed). File writes consume "
                              f"guest RAM and register as memory-signal page changes, so an "
                              f"IO workload measures memory, not IO. Use a real filesystem."})

            avail = int(sf.get("avail_mb", 0) or 0)
            if s["file_mb"] and avail:
                budget = int(avail * SCRATCH_HEADROOM_FRACTION)
                if s["file_mb"] > budget:
                    findings.append({
                        "severity": "FAIL", "step": idx, "workload": s["name"], "kind": "space",
                        "detail": f"writes ~{s['file_mb']} MB into {d} which has {avail} MB "
                                  f"free; will fail with ENOSPC."})

            if sf.get("writable") == "no":
                findings.append({
                    "severity": "FAIL", "step": idx, "workload": s["name"], "kind": "perm",
                    "detail": f"{flag} {d} is not writable by the guest user."})

    if a.json:
        print(json.dumps({"steps": len(lines), "findings": findings}, indent=2))
    else:
        fails = [f for f in findings if f["severity"] == "FAIL"]
        warns = [f for f in findings if f["severity"] == "WARN"]
        print(f"  analysed {len(lines)} steps against guest facts "
              f"(RAM {ram_total} MB, available {ram_avail} MB)")
        for f in fails + warns:
            tag = "[FAIL]" if f["severity"] == "FAIL" else "[warn]"
            print(f"  {tag} step {f['step']:>3} {f['workload']}: {f['detail']}")
        if not findings:
            print("  no resource or scratch findings")

    return 1 if any(f["severity"] == "FAIL" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
