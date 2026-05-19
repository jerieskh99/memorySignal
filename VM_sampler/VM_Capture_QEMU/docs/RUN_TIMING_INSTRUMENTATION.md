# RUN_TIMING_INSTRUMENTATION — Experiment 1

How to run the capture-timing instrumentation experiment (the first of the
four tuning experiments under
[`tuning_plans/`](./tuning_plans/README.md)). The script produces **one
JSON file** with every measurement needed to answer the validation
questions in
[`SNAPSHOT_INTERVAL_QA.md`](./SNAPSHOT_INTERVAL_QA.md) on real data.

> **Scope.** Experiment 1 only. No interval / window-hop / k-segmentation
> tuning. Producer-side timing only. No consumer, no delta calculation,
> no streaming metrics.

## What the experiment measures

Per snapshot, the patched producer
([`capture_producer_qemu_pmemsave.sh`](../capture_producer_qemu_pmemsave.sh))
captures six host wall-clock timestamps (`CLOCK_REALTIME`, nanosecond
resolution via `date +%s.%N`):

| Timestamp                  | What it marks                                          |
| -------------------------- | ------------------------------------------------------ |
| `t0_before_suspend`        | Just before `virsh suspend`                            |
| `t1_after_suspend`         | After `domstate == paused`                             |
| `t2_pmemsave_start`        | Just before the pmemsave monitor command               |
| `t3_pmemsave_end`          | Just after the monitor command returns                 |
| `t4_before_resume`         | Just before `virsh resume`                             |
| `t5_after_resume`          | After `domstate == running`                            |

Plus, per snapshot:

- `queue_depth` — pending + processing job count when the snapshot
  started.
- `backpressure_event` — `true` if the iteration sat in the backpressure
  sleep instead of dumping (recorded as a separate JSONL entry with
  `seq:-1`).
- `image_path`, `dump_bytes`, `interval_msec`, `ram_mb`.

From the six timestamps plus the next snapshot's `t0`, the orchestrator
derives the **three time axes** named in
[`SNAPSHOT_INTERVAL_QA.md`](./SNAPSHOT_INTERVAL_QA.md):

| Derived                  | Formula                  | Axis      |
| ------------------------ | ------------------------ | --------- |
| `suspend_sec`            | `t1 − t0`                | sub-Axis B |
| `pmemsave_sec`           | `t3 − t2`                | sub-Axis B |
| `resume_sec`             | `t5 − t4`                | sub-Axis B |
| `host_snapshot_cycle_sec`| `next_t0 − t0`           | Axis B    |
| `guest_run_interval_sec` | `next_t0 − t5`           | Axis A    |
| `estimated_vm_pause_fraction` | `Σ(t5 − t0) / Σ(next_t0 − t0)` | derived |

## Why it matters

Until this experiment runs, we cannot tell whether
`intervalMsec` is actually being honored by the producer, how big each
component of the per-snapshot wall-clock cost really is, or whether
backpressure ever fires during a normal run. Every later tuning step
(interval / window-hop / k-segmentation) depends on this experiment's
output.

## What changed in the pipeline

A minimal additive instrumentation patch to the producer:

- Adds a `ts_ns()` helper and an `emit_timing` function.
- Captures `__t0`..`__t5` around the existing suspend / pmemsave /
  resume calls. No semantic change to the capture path.
- When the env var `TIMING_JSONL_PATH` is set, writes one JSON line per
  snapshot (and one with `seq:-1` per backpressure event) to that
  path. When the env var is **unset**, behavior is unchanged —
  ordinary captures see no difference.

No new dependencies. Bash 4 + GNU `date` suffice (the same prereqs the
producer already has).

## How to run it

Prereqs:

- Linux host running libvirt + QEMU; `virsh` accessible.
- A capture config at `config_qemu_upc.json` (or supplied via
  `--config`) whose `domain`, `ramSizeMb`, and `imageDir` are correct
  for the live VM.
- Python 3.9+ on the host.
- **`bc` must be installed on the capture host** (`command -v bc`
  must return a path). The producer's post-resume sleep falls back
  to integer `sleep "$(( intervalMsec/1000 ))"` when `bc` is missing,
  which evaluates to `sleep 0` for any `intervalMsec < 1000` —
  producing a guest-running interval of only a few ms instead of the
  configured value. See
  [`TIMING_EXPERIMENT_1_CONCLUSIONS.md`](./TIMING_EXPERIMENT_1_CONCLUSIONS.md)
  for the 2026-05-19 run that caught this.
- Free disk space under `imageDir` for ~`duration / interval × ramSizeMb`
  snapshot bytes (the orchestrator cleans them up by default).

Minimal command (no guest workload, idle VM):

```sh
cd VM_sampler/VM_Capture_QEMU
python3 run_timing_instrumentation_experiment.py \
    --duration 60 \
    --output-json ./timing_run.json
```

With a guest workload (recommended — backpressure only fires under
load):

```sh
python3 run_timing_instrumentation_experiment.py \
    --duration 60 \
    --interval-ms 100 \
    --ram-mb 1024 \
    --test-command "bash ~/VM_executables_phase2/scripts/run_phase2_min.sh /tmp/phase2_min_out" \
    --ssh-target jeries@10.0.2.15 \
    --output-json ./timing_run.json
```

Useful flags:

| Flag                    | Meaning                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `--duration SEC`        | Host wall-clock duration the producer is allowed to run.                |
| `--interval-ms INT`     | Override `intervalMsec` for this run only (config file untouched).      |
| `--ram-mb INT`          | Override `ramSizeMb` for this run only. Must match the live VM RAM.     |
| `--test-command STR`    | Optional workload to run on the guest over SSH during the window.       |
| `--ssh-target USER@HOST`| Required if `--test-command` is set.                                    |
| `--config PATH`         | Use a non-default capture config (default: `./config_qemu_upc.json`).   |
| `--producer-script PATH`| Use a non-default producer script.                                      |
| `--workdir PATH`        | Scratch directory for the per-run JSONL + config copy + producer log.   |
| `--no-vm-start`         | Skip `virsh start`; assume VM is already running.                       |
| `--keep-dumps`          | Do not remove snapshot images written by this run (default: remove).    |
| `--grace-stop-seconds N`| SIGTERM → SIGKILL grace period for the producer (default 10).           |
| `--dry-run`             | Validate inputs and emit the planned JSON without touching the VM.      |

Run `python3 run_timing_instrumentation_experiment.py --help` for the
full list.

## Dry-run first

Before the first real run on a new host, always do a dry-run to confirm
paths and overrides resolve correctly:

```sh
python3 run_timing_instrumentation_experiment.py \
    --dry-run \
    --duration 60 \
    --interval-ms 100 \
    --test-command "echo ok" \
    --ssh-target jeries@10.0.2.15 \
    --output-json /tmp/dryrun.json
cat /tmp/dryrun.json
```

The dry-run output is the same shape as the real output, but with all
`summary` numeric fields set to `null` and a `notes` entry explaining
what would have happened.

## Where the JSON is written

By default, the JSON goes to
`./timing_runs/<UTC>_<uuid>/timing_experiment_<UTC>.json` so concurrent
runs cannot clobber each other. Override with `--output-json`.

The per-run scratch directory (`--workdir`, or the auto-created one)
also contains:

- `config_timing_experiment.json` — a copy of the active config with
  `--interval-ms` / `--ram-mb` overrides applied. The original config
  is **never** modified.
- `snapshot_timings.jsonl` — the raw per-snapshot JSONL. Useful for
  debugging.
- `producer.log` — stdout/stderr of the producer process.
- `workload.log` — stdout/stderr of the SSH workload (if any).

## JSON output schema

```jsonc
{
  "experiment": "timing_instrumentation",
  "timestamp": "2026-05-18T12:34:56Z",
  "config": {
    "interval_ms": 100,
    "duration_sec": 60,
    "ram_size_mb": 1024,
    "vm_domain": "Kali Jeries",
    "capture_method": "pmemsave",
    "test_command": "bash ~/VM_executables_phase2/scripts/run_phase2_min.sh /tmp/p2",
    "config_path_input": "/.../config_qemu_upc.json",
    "config_path_effective": "/.../config_timing_experiment.json",
    "workdir": "/.../timing_runs/...",
    "output_json": "/.../timing_run.json",
    "image_dir": "/var/lib/libvirt/qemu/dump",
    "run_start_epoch": 1748557396.123
  },
  "summary": {
    "snapshots_attempted": 24,
    "snapshots_completed": 24,
    "mean_guest_run_interval_sec": 0.103,
    "std_guest_run_interval_sec": 0.004,
    "mean_host_snapshot_cycle_sec": 2.51,
    "std_host_snapshot_cycle_sec": 0.18,
    "mean_suspend_sec": 0.42,
    "mean_pmemsave_sec": 1.18,
    "mean_resume_sec": 0.21,
    "backpressure_events": 0,
    "queue_max_depth": 0,
    "estimated_vm_pause_fraction": 0.96,
    "jsonl_path": "/.../snapshot_timings.jsonl",
    "producer_log": "/.../producer.log"
  },
  "snapshots": [
    {
      "index": 0,
      "resume_start_ts": 1748557397.000,
      "resume_done_ts":  1748557397.210,
      "guest_run_start_ts": 1748557397.210,
      "suspend_start_ts": 1748557399.815,
      "suspend_done_ts":  1748557400.240,
      "pmemsave_start_ts": 1748557400.240,
      "pmemsave_done_ts":  1748557401.420,
      "next_resume_ts": 1748557402.330,
      "guest_run_interval_sec": 0.102,
      "suspend_sec": 0.425,
      "pmemsave_sec": 1.180,
      "resume_sec": 0.210,
      "host_snapshot_cycle_sec": 2.515,
      "queue_depth": 0,
      "backpressure": false,
      "image_path": "/var/lib/libvirt/qemu/dump/memory_dump-20260518123456789.raw"
    }
  ],
  "notes": [
    "cleanup: removed 24 dump files newer than run start (--keep-dumps=False)",
    "intervalMsec set the guest-running interval, not host wall-clock; see docs/SNAPSHOT_INTERVAL_QA.md."
  ]
}
```

### Field meanings

- `summary.mean_guest_run_interval_sec` — Axis A. With the corrected
  timing model, this should match `--interval-ms / 1000` ± a few ms
  when no backpressure fires.
- `summary.mean_host_snapshot_cycle_sec` — Axis B. The wall-clock cost
  the host pays per snapshot. Dominated by `mean_pmemsave_sec` for
  1 GiB RAM.
- `summary.estimated_vm_pause_fraction` — fraction of host wall-clock
  during which the VM was paused.
- `summary.backpressure_events` — count of iterations the producer
  spent in the backpressure sleep (queue too deep). Each such event
  inflates Axis A on the next live snapshot, so a non-zero value here
  is a quality signal, not a benign one.
- `snapshots[].guest_run_interval_sec` — per-pair Axis A. Plotting
  this over snapshot index shows whether the producer is honoring
  `intervalMsec` consistently.

Any field that cannot be measured is recorded as `null` and called out
in the `notes` array. The script never silently substitutes a guess.

## What to send back to Claude for analysis

**One file: the JSON written to `--output-json`.** That's enough to
answer all of Experiment 1's questions in
[`tuning_plans/01_instrumentation_logging_plan.md`](./tuning_plans/01_instrumentation_logging_plan.md):

- Does `intervalMsec` honor the configured value? (Compare
  `summary.mean_guest_run_interval_sec` to `config.interval_ms / 1000`.)
- What is the realistic per-snapshot host cost?
  (`summary.mean_host_snapshot_cycle_sec`.)
- Where does the cost go? (`mean_suspend_sec`, `mean_pmemsave_sec`,
  `mean_resume_sec`.)
- Are there gaps from backpressure? (`summary.backpressure_events`,
  `summary.queue_max_depth`.)

If the producer log or raw JSONL is also useful, both paths are listed
inside the JSON (`summary.jsonl_path`, `summary.producer_log`). They
are not required for the headline analysis.

## Smoke-testing on a machine without libvirt

`--dry-run` exercises everything that does not require QEMU. Use it to
verify path resolution, override behavior, and JSON shape on the
laptop before pushing the script to the capture host:

```sh
python3 run_timing_instrumentation_experiment.py \
    --dry-run \
    --interval-ms 250 --ram-mb 1024 --duration 30 \
    --output-json /tmp/dryrun.json
```

A full real run needs the capture host with `virsh`, the running VM,
and (optionally) SSH access for `--test-command`.

## Safety

- The original `config_qemu_upc.json` is **never** modified. CLI
  overrides go into a per-run copy under `--workdir`.
- Dump files older than the run start time are **never** removed.
- The orchestrator only kills the producer process group it itself
  started — it does not signal arbitrary PIDs.
- The SSH workload (if any) is started in its own process group; on
  shutdown the orchestrator sends `SIGINT` then `SIGKILL` to that
  group. The guest-side workload may or may not honor the signal.
- The `--keep-dumps` flag preserves all snapshot images for forensic
  inspection.
- Dashboard / website / NPZ / existing-results files are out of
  scope and untouched.
