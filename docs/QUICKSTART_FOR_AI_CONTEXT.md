# Quickstart For AI Context (Active Controlled QEMU)

## Exact current active scope (conservative)
This is the controlled QEMU capture pipeline centered on `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` (host-side controller).

In scope (actively used by the controller’s default flow, unless explicitly overridden by env/config):
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` (main entrypoint)
- `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh` (launcher started when `CAPTURE_MODE=1`)
- `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh` (active default producer when launched)
- `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh` (active default consumer when launched)
- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json` (default `CAPTURE_CONFIG`)
- `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py` (triggered when `OFFLINE_METRICS_MODE=1`)
- Guest workload scripts referenced by the controller’s built-in default step list in `VM_executables/`:
  - `VM_executables/run_idle.sh`
  - `VM_executables/mem_stream.py`
  - `VM_executables/mem_pointer_chase.py`
  - `VM_executables/mem_alloc_touch_pages.py`
  - `VM_executables/io_seq_fsync.py`
  - `VM_executables/io_rand_rw.py`
  - `VM_executables/io_many_files.py`

Also in scope as *analysis dependencies* (not control logic):
- `coherence_temp_spec_stability/` modules used by `offline_step_metrics.py` (see [`OFFLINE_METRICS_AND_OUTPUTS.md`](OFFLINE_METRICS_AND_OUTPUTS.md) for which are executed vs only imported).

Out of scope unless an active file explicitly references them:
- Alternate producers and cleanup helpers in `VM_sampler/VM_Capture_QEMU/`
- Narrative docs and optional `steps_*.txt` unless `STEPS_FILE` is set to them
- `VM_executables/run_files.sh` (wrapper exists but is not used by the controller’s default `load_steps()` list)

## Most important files (read first)
0. [`docs/ACTIVE_PIPELINE_FILE_MAP.md`](ACTIVE_PIPELINE_FILE_MAP.md): conservative used-vs-excluded file/folder map traced from `run_files_controlled.py` only.
1. `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`
2. `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh`
3. `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh`
4. `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
5. `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`
6. `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py`
7. Guest workload scripts listed above in `VM_executables/`

## Main entrypoint (host)
`VM_sampler/VM_Capture_QEMU/run_files_controlled.py`

It enforces the step lifecycle: VM start/resume → SSH readiness → run one guest workload command → (optional) start capture → stop producer → drain queue → stop consumer → (optional) offline metrics → rotate outputs → VM shutdown.

## Key capture scripts (only when `CAPTURE_MODE=1`)
- `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh`
  - spawns producer and consumer (background `nohup` when used by the controller)
  - writes logs: `producer.log`, `consumer.log`
  - writes PID file: `capture_pids.txt`
- `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh`
  - suspends VM, runs QEMU `pmemsave` dump, enqueues `{prev,curr,output}` JSON jobs
  - writes RAW dumps to `imageDir` and job JSON files to `queueDir/pending/`
- `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
  - consumes queued jobs, runs Rust delta program on each pair
  - appends delta frames into `RUN_MATRIX` (default `queueDir/run_matrix.npy`, step-isolated when controller sets `RUN_MATRIX`)
  - optionally triggers live streaming unless `OFFLINE_MODE=1`

## Offline metrics entrypoint (only when `OFFLINE_METRICS_MODE=1`)
`VM_sampler/VM_Capture_QEMU/offline_step_metrics.py`

It produces:
- per-step outputs under `<output_root>/offline/<step_name>/`:
  - `meta.json`
  - `streaming.*`
  - `plv_baseline_aware.json`
- a shared baseline file (for the baseline step) under:
  - `<baseline_dir>/baseline_plv.npy`

## Most important directories (as they appear in active config)
From `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`:
- `queueDir` → `/project/homes/jeries/memory_traces/queue_dir`
  - uses subfolders: `pending/`, `processing/`, `done/`, `failed/`
- `imageDir` → `/var/lib/libvirt/qemu/dump` (producer RAW artifacts)
- `outputDir` → `/project/homes/jeries/memory_traces/output_dir`
  - delta text outputs are in `outputDir/cosine/` and `outputDir/hamming/`
  - controller rotates into `outputDir/rotated/<test_name>/...`
- `streaming.streamingOutputDir` → `/project/homes/jeries/memory_traces/streaming_results`

## What to ignore
- Alternate producers: `capture_producer_qemu.sh`, `capture_producer_qemu_user_raw.sh`
- Cleanup/helpers: `cleanup_qemu_capture.sh`, `fix_line_endings_server.sh`
- Optional `steps_phase_blocks.txt`, `steps_cycle_repetition.txt`, `steps_transition_stress.txt` unless `STEPS_FILE` points to them
- Any `VM_sampler/` branches not in this dependency chain

## What is directly verified (by code inspection)
- `run_files_controlled.py` is the host orchestrator and the active entrypoint
- It enforces sequential steps and calls `ssh` for exactly one remote command per step
- Capture is started/stopped around each step only when `CAPTURE_MODE=1`
- After each capture-enabled step it waits for `queueDir/pending` and `queueDir/processing` to be empty, then stops the consumer
- Offline metrics is invoked only when `OFFLINE_METRICS_MODE=1`, after the queue drain (and after the consumer is stopped)
- The controller rotates delta `*.txt` files from `outputDir/{hamming,cosine}/` into `outputDir/rotated/<test_name>/...`
- The offline baseline step is controlled by `BASELINE_STEP_NUMBER` (default step 1) via `--is-baseline`

## What still needs validation (uncertainties)
Mark these as assumptions until you confirm in your environment:
- Guest path correctness: controller’s default SSH commands assume the guest has the workload scripts under `~/memorySignal/VM_executables/...` (host controller does not validate).
- Runtime availability of binaries/imports:
  - Rust delta program path from `config_qemu_upc.json`
  - Python importability of `coherence_temp_spec_stability` modules from `--project-root`
- Queue directory semantics are validated at the level of “pending+processing emptiness”; whether that fully captures “all needed frames written” is handled by downstream scripts and should be validated empirically.

## Safe re-grounding rule
Prefer evidence from:
- the active controller (`run_files_controlled.py`)
- the launcher/producer/consumer scripts
- `config_qemu_upc.json`
Only then interpret or generalize to metrics modules. Avoid treating repository modules as active unless this path explicitly imports/calls them.
