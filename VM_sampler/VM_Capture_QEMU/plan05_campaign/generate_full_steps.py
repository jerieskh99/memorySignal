#!/usr/bin/env python3
"""Generate the full v3 APF campaign steps file + step->cell manifest.

The v3 matrix is 11 workloads x {120,300,600}s x 2 reps = 66 cells. The
orchestrator names each step `test{i}_<binary>` (1-based file order), so the
per-step apf_trajectory.jsonl filename carries ONLY the binary name -- not the
duration or replicate. This script emits both:

  * full_steps.txt    -- the 66 command lines (one per cell), ordered
                         rep-outer -> duration -> workload, so that:
                           (a) the fast 120s cells run first each rep (early
                               failure surfaces before sinking 600s time),
                           (b) a full single-replicate matrix (33 cells) is
                               complete after the first 33 steps -- maximally
                               useful if the run dies partway, and
                           (c) the two replicates of a cell are temporally
                               separated (independent boot/layout = better
                               run-to-run variance estimate).
  * full_manifest.csv -- step_index,test_name,workload,duration_s,rep,traj_file
                         so analysis can map each testN_* trajectory back to its
                         (workload,duration,rep) cell.

Workload commands are the LOCKED v3 set (PLAN02_V3_RUNBOOK.md lines 79-89);
flags are copied verbatim. --duration <d> and --phase-markers are appended per
cell (SUSTAIN_LOOP re-runs each until --duration elapses). Do NOT edit flags
here without updating the runbook -- the campaign must match v3.

    python3 plan05_campaign/generate_full_steps.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

BIN = "/home/kali/memorySignal/VM_executables_phase2/bin"

# Real-disk scratch for file-writing workloads. The guest /tmp is a ~483 MiB
# tmpfs (RAM-backed); the file-writing workloads create multi-GB sandboxes and
# DEFAULT to /tmp -- which (a) fills instantly -> "No space left on device", and
# (b) puts the workload's file bytes in the measured RAM, contaminating APF.
# /var/tmp is on the 51 GiB vda1 disk. The orchestrator wipes this dir before
# each cell (see run_files_controlled.wipe_guest_scratch).
SCRATCH = "/var/tmp/wl_campaign"

# (binary, locked flags, scratch-dir flag | None). Flags verbatim from
# PLAN02_V3_RUNBOOK.md lines 79-89. The scratch flag is per the binary's own
# --help on the guest: the 5 sandbox_* take --sandbox-dir, mem_mmap_traversal
# takes --backing-dir, the rest are pure-memory (no scratch). Verified
# 2026-06-07; do not assume -- re-check --help if binaries change.
WORKLOADS = [
    ("sandbox_ransom_seq",        "--files 4000 --file-size-bytes 1048576",  "--sandbox-dir"),
    ("sandbox_ransom_slowburn",   "--files 200 --interval-s 3",              "--sandbox-dir"),
    ("sandbox_ransom_selective",  "--files 1500 --file-size-bytes 1048576",  "--sandbox-dir"),
    ("sandbox_ransom_batched",    "--files 12000 --file-size-bytes 1048576", "--sandbox-dir"),
    ("sandbox_scanner_metadata",  "--files 5000 --subdirs 50 --passes 40",   "--sandbox-dir"),
    ("mem_workingset_sweep_v2",   "--working-set-mb 256 --stride 4096",       None),
    ("mem_mmap_traversal_v2",     "--variant rmw --file-size-mb 256",         "--backing-dir"),
    ("mem_pagefault_density_v2",  "--variant mixed --working-set-mb 256",     None),
    ("mem_rmw_intensity_v2",      "--mode rmw --working-set-mb 256 --stride 4096", None),
    ("mem_writemag_sweep_v2",     "--working-set-mb 256 --bytes-per-page 64", None),
    ("app_hashtable_intensive_v2","--capacity-pow2 24 --inserts 6000000 --lookups 10000000", None),
]
DURATIONS = [120, 300, 600]
REPS = [1, 2]

HERE = Path(__file__).resolve().parent


def safe_label(binary: str) -> str:
    # Mirror run_files_controlled.step_name_from_command for a bare binary path:
    # no .py/.sh candidate -> Path(tokens[0]).name -> sanitize.
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", binary).strip("._-") or "step"


def main() -> int:
    steps: list[str] = []
    rows: list[dict] = []
    i = 0
    for rep in REPS:                      # outermost: full single-rep matrix first
        for dur in DURATIONS:             # mid: 120s cells lead each rep
            for binary, flags, scratch_flag in WORKLOADS:
                i += 1
                scratch = f" {scratch_flag} {SCRATCH}" if scratch_flag else ""
                cmd = (f"{BIN}/{binary} {flags}{scratch} "
                       f"--duration {dur} --phase-markers")
                steps.append(cmd)
                label = safe_label(binary)
                test_name = f"test{i}_{label}"
                rows.append({
                    "step_index": i,
                    "test_name": test_name,
                    "workload": binary,
                    "duration_s": dur,
                    "rep": rep,
                    "traj_file": f"run_matrix_{test_name}.npy.apf_trajectory.jsonl",
                })

    steps_path = HERE / "full_steps.txt"
    header = (
        "# Plan 05 -- FULL v3 APF campaign (apf_queue pipeline).\n"
        "# 11 workloads x {120,300,600}s x 2 reps = 66 cells. GENERATED by\n"
        "# generate_full_steps.py -- do not hand-edit; regenerate instead.\n"
        "# Order: rep-outer -> duration -> workload (steps 1-33 = rep1 full\n"
        "# matrix; 34-66 = rep2). Each line carries --duration; SUSTAIN_LOOP\n"
        "# re-runs it until that elapses. Launch (server, inside screen):\n"
        "#   CAPTURE_MODE=1 CAPTURE_METRIC=apf_queue SUSTAIN_LOOP=1 \\\n"
        "#     SSH_TARGET=kali@192.168.222.63 \\\n"
        "#     SSH_KEY=/project/homes/jeries/.ssh/id_ed25519 \\\n"
        "#     CONFIG=config_qemu_upc.json STEPS_FILE=plan05_campaign/full_steps.txt \\\n"
        "#     python3 -u run_files_controlled.py 2>&1 | tee plan05_campaign/full.log\n"
    )
    steps_path.write_text(header + "\n".join(steps) + "\n")

    manifest_path = HERE / "full_manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {steps_path} ({len(steps)} steps)")
    print(f"wrote {manifest_path} ({len(rows)} rows)")
    # Sanity: 11*3*2 = 66.
    assert len(steps) == 66, f"expected 66 cells, got {len(steps)}"
    print("cells: 66 (11 workloads x 3 durations x 2 reps) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
