# VM_executables_phase2

Phase 2 workload executables for the memory-signal thesis.

This folder contains controlled, reproducible workload generators used to drive
the QEMU guest while the host samples guest memory. Phase 2 extends the Phase 1
catalog with low-level MEM probes, app-realistic workloads, and **safe
behavioral simulations** of file-processing patterns (scanners, sequential and
batched per-file pipelines). Nothing in this folder is malware; nothing in this
folder performs real encryption, persistence, network communication, or any
operation outside its declared sandbox directory.

## Test inventory

| Family            | Test slug                          | Lang | Binary / script                                      |
| ----------------- | ---------------------------------- | ---- | ---------------------------------------------------- |
| MEM               | `mem_workingset_sweep_v2`          | C    | `bin/mem_workingset_sweep_v2`                        |
| MEM               | `mem_writemag_sweep_v2`            | C    | `bin/mem_writemag_sweep_v2`                          |
| MEM               | `mem_rmw_intensity_v2`             | C    | `bin/mem_rmw_intensity_v2`                           |
| MEM               | `mem_pagefault_density_v2`         | C    | `bin/mem_pagefault_density_v2`                       |
| MEM               | `mem_mmap_traversal_v2`            | C    | `bin/mem_mmap_traversal_v2`                          |
| APP-REALISTIC     | `app_sqlite_oltp_v2`               | Py   | `app_realistic/app_sqlite_oltp_v2.py`                |
| APP-REALISTIC     | `app_sqlite_analytical_v2`         | Py   | `app_realistic/app_sqlite_analytical_v2.py`          |
| APP-REALISTIC     | `app_compress_gzip_v2`             | Py   | `app_realistic/app_compress_gzip_v2.py`              |
| APP-REALISTIC     | `app_decompress_gzip_v2`           | Py   | `app_realistic/app_decompress_gzip_v2.py`            |
| APP-REALISTIC     | `app_json_parse_v2`                | Py   | `app_realistic/app_json_parse_v2.py`                 |
| APP-REALISTIC     | `app_hashtable_intensive_v2`       | C    | `bin/app_hashtable_intensive_v2`                     |
| SECURITY-LIKE     | `sandbox_ransom_seq`               | C    | `bin/sandbox_ransom_seq`                             |
| SECURITY-LIKE     | `sandbox_ransom_batched`           | C    | `bin/sandbox_ransom_batched`                         |
| SECURITY-LIKE     | `sandbox_ransom_slowburn`          | C    | `bin/sandbox_ransom_slowburn`                        |
| SECURITY-LIKE     | `sandbox_ransom_selective`         | C    | `bin/sandbox_ransom_selective`                       |
| SECURITY-LIKE     | `sandbox_scanner_metadata`         | C    | `bin/sandbox_scanner_metadata`                       |
| METHODOLOGY       | `mp_phase_boundary_inference`      | Py   | `methodology/mp_phase_boundary_inference.py`         |
| METHODOLOGY       | `mp_workingset_metric_linearity`   | Py   | `methodology/mp_workingset_metric_linearity.py`      |

Eighteen tests total. The list mirrors the Phase 2 plan
(`VM_executables/tests_phase_2_experiment_plan.md`) and the v2 cards in the
workload test site (`VM_executables/workload_test_site/data.js`).

## What Phase 2 is

Phase 1 established that the per-snapshot memory delta carries discriminative
signal across coarse workload categories. Phase 2 narrows in on **which
mechanism** the signal responds to. Each Phase 2 test isolates one of:

- spatial footprint (working-set size),
- write magnitude per page,
- read/write mixture (RMW vs pure-read vs pure-write),
- page-fault density vs steady-state touch,
- mmap-IO writeback rhythm vs anonymous writes,
- transactional workloads vs analytical workloads,
- CPU+IO mix from compression/decompression,
- streaming-allocation cadence from JSON parsing,
- two-phase trajectory of in-memory hash tables,
- behavioral patterns of file-processing pipelines, in safe synthetic form,
- methodology calibration (phase-boundary detection, metric linearity fits).

The deliverables for each test are: a metadata JSON file (parameters, timing,
counts), phase markers on stderr, and the test artifacts (sandbox files,
databases, gz files) which the test removes itself unless `--no-cleanup` is
passed.

## Safety model

The SECURITY-LIKE family is **not malware**. It is a behavioral-pattern
simulator. The full rules live in `docs/SAFETY_MODEL.md`. Highlights:

- All file operations are confined to a sandbox directory under `/tmp` or
  `/var/tmp`. A real-path validator runs before any file is touched.
- Files inside the sandbox are generated from a seeded PRNG; no user files,
  home files, system files, or parent directories are ever read or modified.
- The "transform" step is a **reversible** XOR with a fixed 32-byte key
  derived from the run seed. There is no real encryption.
- Hard caps are enforced before any work begins: **5000 files**, **5 GiB
  total**, **600 s runtime** per SECURITY-LIKE test. Exceeding any cap aborts
  before the first file is touched.
- Tests refuse to start if the supplied sandbox path is not absolute, contains
  `..`, resolves outside an approved root, or points to a symlink.
- `--dry-run` validates parameters without touching files.
- `--cleanup` (default on for SECURITY-LIKE) recursively removes the sandbox
  tree on exit, refusing to follow symlinks.

No network communication. No persistence. No privilege escalation. No
evasion. No real malware logic. No destructive behavior. The XOR transform is
included so the test can be a self-consistent reversible workload, *not* so
it can mimic encryption — the key is deterministic from `--seed` and emitted
to the metadata for verifiability.

## Build

```sh
make                  # build C executables into ./bin/
make CC=clang         # use clang
make CFLAGS="-O0 -g"  # debug build
make smoke            # build + run the smoke battery (small dry/real runs)
make clean            # remove ./bin/
```

Python tests have no build step; CPython 3.10+ standard library suffices. No
third-party Python packages are required.

## Run a single test

Every test supports `--help`. Common flags:

```text
--duration SEC        Measurement duration (where applicable)
--seed N              PRNG seed (deterministic if same seed + params)
--output-dir PATH     Where to write the metadata JSON
--sandbox-dir PATH    Sandbox parent (security_like_safe + io tests)
--safe-root PATH      Additional approved root for sandbox validation
--phase-markers       Emit per-phase markers to stderr
--dry-run             Validate args and exit without doing work
--cleanup             Remove generated artifacts on exit
--cpu-affinity N      Pin to CPU N (best effort)
```

Examples:

```sh
bin/mem_workingset_sweep_v2 \
    --working-set-mb 1024 --duration 60 --warmup 5 --seed 42 \
    --output-dir /tmp/phase2_out

bin/sandbox_ransom_seq \
    --files 100 --file-size-bytes 65536 --duration 60 \
    --sandbox-dir /tmp --seed 42 --phase-markers \
    --output-dir /tmp/phase2_out

python3 app_realistic/app_sqlite_oltp_v2.py \
    --duration 60 --rows 20000 --output-dir /tmp/phase2_out \
    --sandbox-dir /tmp --seed 42 --cleanup
```

## Run a family

For an ad-hoc batch by family, loop over the binaries directly:

```sh
for b in bin/mem_*; do "$b" --duration 30 --output-dir /tmp/phase2_out; done
for b in bin/sandbox_*; do "$b" --duration 60 --output-dir /tmp/phase2_out; done
for f in app_realistic/*.py; do python3 "$f" --duration 30 --output-dir /tmp/phase2_out --sandbox-dir /tmp --cleanup; done
```

## Run the Phase 2 minimum viable batch

```sh
scripts/run_phase2_min.sh /tmp/phase2_out
```

This runs the seven-test minimum batch from the plan (sandbox_ransom_seq,
sandbox_ransom_batched, sandbox_scanner_metadata, mem_workingset_sweep_v2,
app_sqlite_oltp_v2, app_compress_gzip_v2, mp_phase_boundary_inference) with
small parameters suitable for a local smoke run. For a real capture cycle,
drive these executables via the QEMU sampler harness.

## Clean up

```sh
make clean              # remove ./bin/
scripts/cleanup_sandboxes.sh   # remove any leftover /tmp/phase2_sandbox_* dirs
```

`cleanup_sandboxes.sh` only removes paths whose canonical form starts with
`/tmp/phase2_sandbox_` or `/var/tmp/phase2_sandbox_`. Anything else is refused.

## Connection to the memory-signal experiment

These executables run **inside the guest VM** while the host sampler captures
guest memory snapshots via `pmemsave` (see `VM_sampler/`). The metadata JSON
files written by each test let the analyzer align segments with the workload
phase structure (`docs/IMPLEMENTATION_NOTES.md`).

Phase 2 outputs feed directly into:

- the confusion-matrix analysis in `VMsig_featureExctraction/`,
- the segment-level analysis in `coherence_temp_spec_stability/`,
- the website / dashboard test cards under `VM_executables/workload_test_site/`.

This folder does not modify any of those existing folders or files.

## Layout

```
VM_executables_phase2/
  README.md
  Makefile
  common/
    phase2_common.h        # C metadata + sandbox + PRNG helpers
    phase2_common.py       # Python equivalents
    phase2_portable.h      # macOS/Linux build shims
    phase2_sandbox.h       # SECURITY-LIKE sandbox helper (C)
  mem/
    mem_workingset_sweep_v2.c
    mem_writemag_sweep_v2.c
    mem_rmw_intensity_v2.c
    mem_pagefault_density_v2.c
    mem_mmap_traversal_v2.c
  app_realistic/
    app_sqlite_oltp_v2.py
    app_sqlite_analytical_v2.py
    app_compress_gzip_v2.py
    app_decompress_gzip_v2.py
    app_json_parse_v2.py
    app_hashtable_intensive_v2.c
  security_like_safe/
    sandbox_ransom_seq.c
    sandbox_ransom_batched.c
    sandbox_ransom_slowburn.c
    sandbox_ransom_selective.c
    sandbox_scanner_metadata.c
  methodology/
    mp_phase_boundary_inference.py
    mp_workingset_metric_linearity.py
  scripts/
    smoke_all.sh
    cleanup_sandboxes.sh
    run_phase2_min.sh
  docs/
    SAFETY_MODEL.md
    TEST_CATALOG.md
    IMPLEMENTATION_NOTES.md
    SMOKE_TEST_RESULTS.md
  bin/                     # populated by `make`
```
