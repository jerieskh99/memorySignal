# Active Pipeline Overview

## What This Documents
This document reconstructs only the currently active architecture used in this project phase. The entrypoint is `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`, and the active dependency chain extends only through the QEMU capture files it invokes, the guest workloads it executes, and the offline metrics step it conditionally triggers.

## Where It Sits In The Pipeline
This is the top-level overview document for the active controlled QEMU path. It summarizes how the host orchestration, capture pipeline, guest workloads, and offline metrics relate to each other.

## Relevant Files
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`
- `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh`
- `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh`
- `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`
- `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py`
- `VM_executables/run_idle.sh`
- `VM_executables/mem_stream.py`
- `VM_executables/mem_pointer_chase.py`
- `VM_executables/mem_alloc_touch_pages.py`
- `VM_executables/io_seq_fsync.py`
- `VM_executables/io_rand_rw.py`
- `VM_executables/io_many_files.py`

## End-To-End Flow
1. The host controller ensures the libvirt VM is running and SSH-reachable.
2. It selects one guest workload command from the default step list or an external `STEPS_FILE`.
3. If `CAPTURE_MODE=1`, it starts the QEMU launcher in background mode before running the guest step.
4. The active producer captures RAW memory states and emits `prev`/`curr` queue jobs.
5. The consumer computes delta outputs, appends frame vectors into a run matrix, and may trigger live streaming metrics.
6. After the guest command finishes, the controller stops the producer, waits for queue drain, stops the consumer, optionally runs offline metrics, rotates step outputs, and stops the VM.

## Active Components And Roles
### Host Orchestrator
`run_files_controlled.py` is the host-side coordinator. It owns VM lifecycle, SSH execution, capture start/stop boundaries, queue-drain waiting, offline-step invocation, and per-step output rotation.

### Capture Launcher
`run_qemu_capture.sh` starts the producer and consumer together. In the active controller path it is used in background mode with `CONFIG`, `PRODUCER_SCRIPT`, and sometimes `RUN_MATRIX` and `OFFLINE_MODE`.

### Producer
`capture_producer_qemu_pmemsave.sh` pauses the VM, performs `pmemsave`, optionally fixes ownership, and emits queue jobs pairing adjacent memory dumps.

### Consumer
`capture_consumer_qemu.sh` processes queued dump pairs, runs the Rust delta binary, appends frames to a matrix, and conditionally runs live streaming metrics.

### Guest Workloads
The controller executes one workload per step inside the guest from `VM_executables/`. The default set covers idle, memory-intensive, and I/O-intensive behaviors.

### Offline Metrics
`offline_step_metrics.py` runs only when offline mode is enabled and only after queue drain for a completed step. It loads `coherence_temp_spec_stability` modules from `--project-root` (see [`OFFLINE_METRICS_AND_OUTPUTS.md`](OFFLINE_METRICS_AND_OUTPUTS.md) for which files are executed vs only imported).

## Inputs And Outputs
### Main Inputs
- libvirt VM state and domain name
- SSH target and optional SSH credentials/options
- optional `STEPS_FILE`
- capture configuration from `config_qemu_upc.json`
- guest workload scripts under `~/memorySignal/VM_executables/...`

### Main Outputs
- RAW dumps under `imageDir`
- queue jobs under `queueDir`
- delta text files under `outputDir/cosine/` and `outputDir/hamming/`
- `run_matrix.npy` or step-specific `run_matrix_<test_name>.npy`
- optional live streaming outputs
- optional offline outputs under `outputDir/offline/<step_name>/`

## Direct Evidence
- The host step loop and capture boundaries are directly implemented in `run_files_controlled.py`
- The launcher startup behavior is directly implemented in `run_qemu_capture.sh`
- The pmemsave capture logic is directly implemented in `capture_producer_qemu_pmemsave.sh`
- The queue-processing and run-matrix logic are directly implemented in `capture_consumer_qemu.sh`
- The current directory and output conventions are directly visible in `config_qemu_upc.json`

## Inference
- The controller assumes the guest contains the workload scripts at `~/memorySignal/VM_executables/...`; it does not verify that
- The controller does not pass `CONSUMER_SCRIPT`, so consumer selection is inferred from the defaults inside `run_qemu_capture.sh`

## Uncertainty And Scope Limits
- `rawRetention` exists in the consumer but is disabled in the checked `config_qemu_upc.json`, so it is not part of the active default path
- alternate producers exist in the same directory but are not selected by the active controller defaults
- external metric package internals are not reconstructed here because the active scripts only show how those modules are invoked, not how they work internally

## File-and-folder map (authoritative)

[`ACTIVE_PIPELINE_FILE_MAP.md`](ACTIVE_PIPELINE_FILE_MAP.md) is the conservative audit traced from `run_files_controlled.py` only. It includes:

- **Dependency map** — what the controller invokes directly vs what `run_qemu_capture.sh` starts
- **Flow map** — sequence for one capture-enabled step
- **Used files list** — scripts and config paths clearly on the active path
- **Excluded files list** — repository files not opened or executed by the controller

Use that document when you need precision; this overview stays narrative.
