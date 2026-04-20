# Guest Workloads

## Purpose
This section documents the guest-side workloads actually executed by the active controller, and separates the default workload sequence from optional alternative step lists selected through `STEPS_FILE`.

## Relevant Files
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`
- `VM_executables/run_idle.sh`
- `VM_executables/mem_stream.py`
- `VM_executables/mem_pointer_chase.py`
- `VM_executables/mem_alloc_touch_pages.py`
- `VM_executables/io_seq_fsync.py`
- `VM_executables/io_rand_rw.py`
- `VM_executables/io_many_files.py`
- Optional only: `VM_sampler/VM_Capture_QEMU/steps_phase_blocks.txt`
- Optional only: `VM_sampler/VM_Capture_QEMU/steps_cycle_repetition.txt`
- Optional only: `VM_sampler/VM_Capture_QEMU/steps_transition_stress.txt`

## Support Level
- Directly supported for the default active command sequence and workload implementations
- Inferred for the assumption that the guest filesystem mirrors the same workload scripts at `~/memorySignal/VM_executables/...`

## Out Of Scope
- `VM_executables/run_files.sh`, which is present but not used by the default controller loop
- Broader experimental interpretation beyond the implemented workload behavior

## How Workloads Are Selected
The active controller uses one of two sources:

- `STEPS_FILE`, if set and present on the host
- otherwise the built-in default list in `load_steps()`

This means the default architecture is the built-in list, while the `steps_*.txt` files are optional scenario definitions rather than mandatory components of every run.

## Default Active Workload Sequence
The active default sequence is:

1. `bash ~/memorySignal/VM_executables/run_idle.sh --time 30`
2. `python3 ~/memorySignal/VM_executables/mem_stream.py --mb 128 --seconds TEST_EXEC_SECONDS`
3. `python3 ~/memorySignal/VM_executables/mem_pointer_chase.py --mb 1024 --seconds TEST_EXEC_SECONDS --seed 123`
4. `python3 ~/memorySignal/VM_executables/mem_alloc_touch_pages.py --objects 2000 --object-kb 256 --sleep-ms 20 --seconds TEST_EXEC_SECONDS`
5. `python3 ~/memorySignal/VM_executables/io_seq_fsync.py --seconds TEST_EXEC_SECONDS --kb 4096 --fsync-wait 1 --path io_seq.bin`
6. `python3 ~/memorySignal/VM_executables/io_rand_rw.py --seconds TEST_EXEC_SECONDS --file-mb 2048 --block-kb 64 --write-ratio 0.5 --path io_rand.bin --seed 123`
7. `python3 ~/memorySignal/VM_executables/io_many_files.py --seconds TEST_EXEC_SECONDS --files-per-batch 500 --payload-bytes 1024 --seed 123`

The controller does not batch these inside the guest. It executes exactly one command per step over SSH and waits for that command to finish before moving on.

## Implemented Workload Roles
### `run_idle.sh`
Role:
- idle baseline step

Implemented behavior:
- sleeps for a configurable number of seconds

Support:
- directly supported by `VM_executables/run_idle.sh`

### `mem_stream.py`
Role:
- dense sequential page-touch memory activity

Implemented behavior:
- allocates a `uint8` NumPy buffer sized by `--mb`
- repeatedly touches one byte per page with a fixed page stride

Support:
- directly supported by `VM_executables/mem_stream.py`

### `mem_pointer_chase.py`
Role:
- pseudo-random page-granular memory traversal

Implemented behavior:
- fills a large buffer with random bytes
- uses a deterministic LCG-style progression to access page-sized slots

Support:
- directly supported by `VM_executables/mem_pointer_chase.py`

### `mem_alloc_touch_pages.py`
Role:
- allocator churn and repeated page touching

Implemented behavior:
- repeatedly allocates many bytearrays
- touches one byte per page in each object
- optionally sleeps between batches

Support:
- directly supported by `VM_executables/mem_alloc_touch_pages.py`

### `io_seq_fsync.py`
Role:
- sequential file writing with forced persistence cadence

Implemented behavior:
- writes fixed-size blocks to a file
- calls `fsync` every `K` chunks

Support:
- directly supported by `VM_executables/io_seq_fsync.py`

### `io_rand_rw.py`
Role:
- random mixed read-write file I/O

Implemented behavior:
- preallocates a large file
- repeatedly seeks to random block-aligned offsets
- performs reads or writes according to `--write-ratio`

Support:
- directly supported by `VM_executables/io_rand_rw.py`

### `io_many_files.py`
Role:
- metadata-heavy small-file churn

Implemented behavior:
- creates many small files in a temporary directory
- writes payloads and usually deletes them in batches

Support:
- directly supported by `VM_executables/io_many_files.py`

## Optional `STEPS_FILE` Variants
The repository also contains:

- `VM_sampler/VM_Capture_QEMU/steps_phase_blocks.txt`
- `VM_sampler/VM_Capture_QEMU/steps_cycle_repetition.txt`
- `VM_sampler/VM_Capture_QEMU/steps_transition_stress.txt`

These files still point to the same guest workload scripts under `~/memorySignal/VM_executables/...`, but they are not part of the default active flow unless `STEPS_FILE` is explicitly set to one of them. For the booklet, they should be documented as optional scenario definitions, not as always-active architecture.

## Directly Implemented Versus Inferred
### Directly Implemented
- The controller's default sequence uses exactly seven guest-side commands
- Each command points to a specific script under `~/memorySignal/VM_executables/`
- Each script implements a simple, steady-state workload pattern

### Inferred Or Ambiguous
- The guest is assumed to host those scripts under `~/memorySignal/VM_executables/`, but the controller does not verify their presence
- Nearby docs mention `~/VM_executables/...` in some examples, which conflicts with the actual controller command strings

## Why `run_files.sh` Is Excluded
`VM_executables/run_files.sh` is a wrapper script present in the repository, but it is not used by the default `run_files_controlled.py` step list. The active architecture described here is the per-step SSH execution model implemented by the controller itself.
