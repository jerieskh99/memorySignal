# Plan 05 -- APF campaign runbook (Wave 4, Step 8)

Runs the real data campaign through the production `apf_queue` pipeline (producer
-> queue -> Rust `apf_calc` consumer), with `SUSTAIN_LOOP` so workloads churn the
whole window. **Subset first** to validate the pipeline, then the full 66-cell v3
matrix.

## Pre-flight (server)

```bash
cd ~/memorySignal/VM_sampler/VM_Capture_QEMU && git pull
virsh -c qemu:///system list --all          # "Kali Jeries" present
df -h /project | tail -1                     # plenty free (delete-as-you-go keeps it bounded)
screen -S plan05campaign                     # run inside screen (survives logout, not reboot)
```

## Subset (~6 cells, ~1 h)

```bash
CAPTURE_MODE=1 CAPTURE_METRIC=apf_queue SUSTAIN_LOOP=1 \
  SSH_TARGET=kali@192.168.222.63 \
  SSH_KEY=/project/homes/jeries/.ssh/id_ed25519 \
  CONFIG=config_qemu_upc.json \
  STEPS_FILE=plan05_campaign/subset_steps.txt \
  python3 run_files_controlled.py 2>&1 | tee plan05_campaign/subset.log
```

Each step: start an apf_queue capture, run the (sustained) workload over SSH,
drain the queue (apf_calc per pair), stop. The producer enqueues; the consumer
runs `apf_calc` and appends one line per pair to a per-step
`apf_trajectory.jsonl`; prev is deleted as it goes (disk stays bounded).

## Read the subset result

```bash
python3 plan05_campaign/analyze_subset.py
```
Prints, per step: pair count, APF mean, and analysis-window count at the frozen
`(W,H)=(8,4)`. **Pass** = non-trivial APF (clearly > 0, not ~0 idle) and high
window counts (>> the v3 baseline of 53/132 cells with <=3 windows) -- i.e. the
sustain-loop kept workloads churning and the faster cadence relieved the DOF
starvation.

## Full campaign (66 cells) -- after the subset is clean

The v3 matrix is **11 workloads x {120,300,600}s x 2 reps = 66 cells**. The
orchestrator names outputs per step (`test{i}_...`), so a single steps file with
all 66 lines works -- but a long single session is fragile. Recommended: one
session per (duration, replicate) batch = 6 sessions x 11 workloads, each with its
own `subset.log`-style tee, so a failure only costs one batch.

Locked params (PLAN02_V3_RUNBOOK.md): interval **500 ms**, RAM **1024 MiB**,
durations 120/300/600 s, 2 reps. Note the committed `config_qemu_upc.json` has
`intervalMsec` = 100; set it to 500 for v3-comparable cadence before the full run
(the subset is fine at either -- pmemsave dominates the cadence floor anyway).

Workload family: `sandbox_ransom_{seq,slowburn,selective,batched}`,
`sandbox_scanner_metadata`, `mem_{workingset_sweep_v2,mmap_traversal_v2,
pagefault_density_v2,rmw_intensity_v2,writemag_sweep_v2}`,
`app_hashtable_intensive_v2` (commands in PLAN02_V3_RUNBOOK.md).

## C1-C8 validation (follow-up)

`plan02_validate_session.py` expects a `--cells-dir` + `--manifest`. The
`apf_queue` orchestrator writes per-step `apf_trajectory.jsonl`, not that cells-dir
layout, so a small adapter (map per-step trajectories -> the cells-dir schema) is
needed before C1-C8 runs against the campaign. Design after the subset confirms
the pipeline.
