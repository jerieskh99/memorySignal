# Guest Workloads

## What This Part Does
This document describes the guest-side workload scripts that the active controller actually executes. It focuses only on workloads that are directly referenced by `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` or by optional step files that the active controller can consume through `STEPS_FILE`.

## Where It Sits In The Pipeline
These workloads are the guest behaviors driven by the host controller. They sit between VM startup/SSH reachability and the host-side post-step capture drain and output handling.

## Relevant Files
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`
- `VM_executables/run_idle.sh`
- `VM_executables/mem_stream.py`
- `VM_executables/mem_pointer_chase.py`
- `VM_executables/mem_alloc_touch_pages.py`
- `VM_executables/io_seq_fsync.py`
- `VM_executables/io_rand_rw.py`
- `VM_executables/io_many_files.py`
- optional only: `VM_sampler/VM_Capture_QEMU/steps_phase_blocks.txt`
- optional only: `VM_sampler/VM_Capture_QEMU/steps_cycle_repetition.txt`
- optional only: `VM_sampler/VM_Capture_QEMU/steps_transition_stress.txt`

## Default Active Workload Sequence
When `STEPS_FILE` is not set, `run_files_controlled.py` executes the following default sequence inside the guest:

1. `run_idle.sh`
2. `mem_stream.py`
3. `mem_pointer_chase.py`
4. `mem_alloc_touch_pages.py`
5. `io_seq_fsync.py`
6. `io_rand_rw.py`
7. `io_many_files.py`

Each command is executed as a standalone remote step over SSH.

## What Each Workload Does
### `run_idle.sh`
Purpose:
- creates a simple idle baseline interval

Implemented behavior:
- sleeps for a configurable number of seconds

Inputs:
- `--time`

Outputs:
- no structured output beyond stdout

### `mem_stream.py`
Purpose:
- creates dense sequential page-touch memory activity

Implemented behavior:
- allocates a large byte buffer
- repeatedly touches one byte per page with a fixed stride

Inputs:
- `--mb`
- `--seconds`

Outputs:
- no structured output beyond heartbeat messages

### `mem_pointer_chase.py`
Purpose:
- creates pseudo-random page-granular memory traversal

Implemented behavior:
- allocates random bytes
- accesses page slots according to a deterministic LCG-style progression

Inputs:
- `--mb`
- `--seconds`
- `--seed`
- `--stride`

Outputs:
- no structured output beyond heartbeat messages

### `mem_alloc_touch_pages.py`
Purpose:
- creates allocation churn and repeated page touching

Implemented behavior:
- repeatedly allocates many medium-sized bytearrays
- touches one byte per page
- clears allocations and may sleep between batches

Inputs:
- `--seconds`
- `--objects`
- `--object-kb`
- `--sleep-ms`
- `--gc`
- `--page-size`

Outputs:
- no structured output beyond batch counters

### `io_seq_fsync.py`
Purpose:
- creates sequential write and persistence activity

Implemented behavior:
- writes fixed-size chunks to a file
- calls `fsync` at a configured cadence

Inputs:
- `--seconds`
- `--kb`
- `--fsync-wait`
- `--path`

Outputs:
- creates or overwrites the named guest file

### `io_rand_rw.py`
Purpose:
- creates random mixed read/write file activity

Implemented behavior:
- preallocates a file
- performs random block-aligned reads or writes using a configured write ratio

Inputs:
- `--seconds`
- `--file-mb`
- `--block-kb`
- `--write-ratio`
- `--path`
- `--seed`

Outputs:
- creates a guest-side file used for random I/O

### `io_many_files.py`
Purpose:
- creates metadata-heavy file churn

Implemented behavior:
- repeatedly creates many small files in a temporary directory
- usually deletes them again unless `--keep` is set

Inputs:
- `--seconds`
- `--files-per-batch`
- `--payload-bytes`
- `--keep`
- `--seed`

Outputs:
- a temporary directory and many transient small files inside the guest

## Inputs And Outputs In The Pipeline Context
### Inputs
- host-chosen command strings from `run_files_controlled.py`
- optional external step file lines through `STEPS_FILE`

### Outputs
- guest-side behavioral activity that influences captured memory evolution
- occasional guest files such as `io_seq.bin` or `io_rand.bin`
- stdout visible to the SSH command path, but no structured host-side artifact files from the workloads themselves

## Direct Evidence
- the default command sequence is directly defined in `run_files_controlled.py`
- each workload's implemented behavior is directly visible in the corresponding script under `VM_executables/`

## Inference
- the guest is assumed to contain these scripts under `~/memorySignal/VM_executables/...`
- the broader experimental interpretation of these workloads comes from the implementation plus nearby documentation, not from the controller itself

## Uncertainty
- `VM_executables/run_files.sh` exists but is not part of the default controlled loop, so it should not be treated as active architecture here
- optional `steps_*.txt` files are only active when `STEPS_FILE` points to them
