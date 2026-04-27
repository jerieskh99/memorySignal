# Inputs Outputs And Directories

## Purpose
This section consolidates the path conventions, file types, and artifact movements used by the active controlled QEMU pipeline.

## Relevant Files
- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`
- `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh`
- `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
- `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py`

## Support Level
- Directly supported by the active config and scripts

## Out Of Scope
- Stale output examples from earlier experiments
- Directories unrelated to the active controlled QEMU flow

## Host-Side Inputs
### Controller Inputs
- `SSH_TARGET` and optional SSH credentials/options
- `VM_DOMAIN` and `VIRSH_URI`
- optional `STEPS_FILE`
- optional `CAPTURE_MODE` and `OFFLINE_METRICS_MODE`
- optional overrides for config path, producer path, and offline paths

### Capture Config Inputs
The checked active config file is `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`. It contributes these host-side path conventions:

- `imageDir = /var/lib/libvirt/qemu/dump`
- `queueDir = /project/homes/jeries/memory_traces/queue_dir`
- `outputDir = /project/homes/jeries/memory_traces/output_dir`
- `streaming.streamingOutputDir = /project/homes/jeries/memory_traces/streaming_results`
- `streaming.projectRoot = /project/homes/jeries/memorySignal`

## Guest-Side Inputs
The controller expects each default workload command to exist inside the guest under:

- `~/memorySignal/VM_executables/run_idle.sh`
- `~/memorySignal/VM_executables/mem_stream.py`
- `~/memorySignal/VM_executables/mem_pointer_chase.py`
- `~/memorySignal/VM_executables/mem_alloc_touch_pages.py`
- `~/memorySignal/VM_executables/io_seq_fsync.py`
- `~/memorySignal/VM_executables/io_rand_rw.py`
- `~/memorySignal/VM_executables/io_many_files.py`

These are guest-side paths referenced by SSH command strings. Their existence on the guest is assumed, not validated by the host controller.

## Queue Directory Convention
`queueDir` is the coordination root between producer, consumer, and controller.

Subdirectories used by the active flow:
- `pending/`: jobs produced but not yet claimed
- `processing/`: jobs claimed by the consumer and in flight
- `done/`: completed jobs
- `failed/`: failed jobs

Additional matrix artifact:
- `run_matrix.npy` by default, or `run_matrix_<test_name>.npy` when the controller provides a per-step override

## Dump And Delta Artifacts
### RAW Dumps
Producer outputs:
- timestamped `.raw` files under `imageDir`

Meaning:
- each file is one captured memory state
- adjacent states become `prev` and `curr` in a queue job

### Delta Outputs
Consumer outputs:
- text files under `outputDir/cosine/`
- text files under `outputDir/hamming/`

Meaning:
- each file is a per-frame page-wise delta vector produced by the Rust binary
- the active consumer appends one selected frame at a time into the run matrix

## Run Matrix Convention
On disk, the active consumer stores the matrix with shape:

- `[num_pages, num_frames]`

Active variants:
- default shared path: `queueDir/run_matrix.npy`
- step-scoped path from the controller: `queueDir/run_matrix_<test_name>.npy`

Meaning:
- rows correspond to memory pages
- columns correspond to temporal frames generated from adjacent snapshot pairs

For offline metrics, this matrix is loaded and transposed to `[frames, pages]`.

## Streaming And Offline Outputs
### Live Streaming Outputs
When live streaming is active and not suppressed by `OFFLINE_MODE=1`, the consumer writes outputs under:

- `streaming.streamingOutputDir`

These outputs are generated asynchronously after the run matrix reaches the configured frame threshold.

### Offline Outputs
When `OFFLINE_METRICS_MODE=1`, the offline script writes under:

- `<outputDir>/offline/<step_name>/`

Key files:
- `meta.json`
- `streaming.*`
- `plv_baseline_aware.json`

Shared baseline location:
- `<outputDir>/offline/baseline/baseline_plv.npy`, unless overridden

## Rotated Delta Artifacts
After each captured step completes, the host controller rotates text files from:

- `outputDir/hamming/*.txt`
- `outputDir/cosine/*.txt`

into:

- `outputDir/rotated/<test_name>/hamming/`
- `outputDir/rotated/<test_name>/cosine/`

This preserves per-step delta outputs while leaving the producer-consumer directory layout intact for subsequent capture.

## Artifact Lifecycle Summary
1. Producer writes a RAW dump into `imageDir`.
2. Producer pairs `prev` and `curr` dumps in a JSON job under `queueDir/pending/`.
3. Consumer moves that job through `processing` into `done` or `failed`.
4. Consumer writes delta text outputs under `outputDir`.
5. Consumer appends one delta frame into the active run matrix.
6. Controller waits for queue drain, stops the consumer, and stops the VM.
7. Controller optionally runs offline metrics on the completed step matrix.
8. Controller rotates delta text files into a per-step archival subdirectory.

## Directly Implemented Versus Inferred
### Directly Implemented
- the host-side queue, output, and matrix paths come from `config_qemu_upc.json` and the active scripts
- per-step run matrix isolation is implemented by the controller
- rotated per-step delta output directories are created by `rotate_delta_files()`

### Inferred Or Ambiguous
- guest-side workload paths are inferred from SSH command strings
- retention or archival semantics beyond the checked config are conditional and should not be described as always active
