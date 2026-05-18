# Phase 2 test catalog

One section per test. Each section follows:

- **Purpose** — what mechanism the test isolates.
- **Behavior** — phases the test executes.
- **Parameters** — CLI flags + defaults.
- **Example** — one ready-to-run invocation.
- **Artifacts** — what the test writes to disk.
- **Safety notes** — anything tied to the safety model.
- **Memory-signal interpretation** — what the analyst should expect to see.

## MEM family

### `mem_workingset_sweep_v2`

- **Purpose.** Working-set scaling: vary the buffer footprint at fixed stride
  and fixed per-page write magnitude. Isolates the active-page-fraction
  response to footprint size.
- **Behavior.** Phases: `warmup` (memset to commit pages) → `measure`
  (repeated sequential 4 KiB-stride writes for `--duration` seconds) →
  `cooldown` (brief settle).
- **Parameters.** `--working-set-mb` (default 1024), `--duration` (60),
  `--warmup` (5), `--stride` (4096), `--seed` (42), `--max-mb` (16384),
  `--no-mlock`, `--cpu-affinity N`, `--output-dir`, `--dry-run`, `--help`.
- **Example.**
  ```sh
  bin/mem_workingset_sweep_v2 --working-set-mb 1024 --duration 60 \
      --seed 42 --output-dir /tmp/p2
  ```
- **Artifacts.** `/tmp/p2/mem_workingset_sweep_v2_metadata.json`.
- **Safety notes.** Refuses `--working-set-mb` above `--max-mb`. `mlock` is
  best-effort; warns and continues if `RLIMIT_MEMLOCK` denies it.
- **Memory-signal interpretation.** Active page fraction grows monotonically
  with footprint until the host’s page-cache / DRAM budget is exhausted; SNR
  per page falls as footprint grows because per-page revisit rate drops.

### `mem_writemag_sweep_v2`

- **Purpose.** Per-page write magnitude sweep over a fixed buffer. Isolates
  the Hamming-delta saturation curve and cosine-delta floor.
- **Behavior.** `warmup` → `measure` (write `--bytes-per-page` to every page
  each pass) → `cooldown`.
- **Parameters.** `--working-set-mb` (1024), `--bytes-per-page`
  (1 / 64 / 1024 / 4096), `--duration` (60), `--warmup` (5), `--seed`,
  `--no-mlock`, `--cpu-affinity`, `--output-dir`, `--dry-run`.
- **Example.**
  ```sh
  for b in 1 64 1024 4096; do
      bin/mem_writemag_sweep_v2 --working-set-mb 1024 --bytes-per-page $b \
          --duration 60 --output-dir /tmp/p2 --seed 42
  done
  ```
- **Safety notes.** Source bytes are PRNG-generated; the buffer holds non-zero
  content so writes register as deltas.
- **Memory-signal interpretation.** Per-page Hamming delta grows with bytes
  written until ~2 KiB / page and then saturates; cosine delta moves away
  from zero only above the ~64 B threshold.

### `mem_rmw_intensity_v2`

- **Purpose.** Separate the read-modify-write signature from pure-read and
  pure-write at matched stride and footprint.
- **Behavior.** `warmup` → `measure` with `--mode {write|read|rmw}` → `cooldown`.
- **Parameters.** `--mode` (default `rmw`), `--working-set-mb` (1024),
  `--stride` (4096), `--duration` (60), `--warmup` (5), `--seed`,
  `--no-mlock`, `--cpu-affinity`, `--output-dir`.
- **Memory-signal interpretation.** Pure-read should show near-zero per-page
  delta (load only). Pure-write shows full delta. RMW is the pre-registered
  null result: per-page delta distribution may or may not separate from
  pure-write at the metric level.

### `mem_pagefault_density_v2`

- **Purpose.** Isolate the page-fault density signature from steady-state
  touch.
- **Behavior.** Variants:
  - `fault_only`: no pre-warm, no `MAP_POPULATE`; each measurement step
    touches a new page exactly once.
  - `touch_only`: pre-fault via `MAP_POPULATE` and a warm-up `memset`;
    measurement does steady-state revisits with zero new faults.
  - `mixed`: 50 % pre-warmed; remaining pages fault in gradually during
    measurement.
- **Parameters.** `--variant`, `--working-set-mb` (1024), `--duration` (60),
  `--seed`, `--no-mlock`, `--cpu-affinity`, `--output-dir`.
- **Memory-signal interpretation.** `fault_only` produces a sharp segment-1
  peak (zero-fill + page-table updates) that decays as pages run out.
  `touch_only` is steady. `mixed` shows a slow decay.

### `mem_mmap_traversal_v2`

- **Purpose.** Compare mmap-IO semantics against anonymous mappings.
- **Behavior.** Generates a deterministic-content backing file in
  `--backing-dir` (default `/tmp`), mmaps it `MAP_SHARED`, then iterates
  with `--variant {read|write|rmw}`. Optional `--msync-interval-ms` triggers
  periodic `MS_ASYNC` calls so writeback rhythm is configurable.
- **Parameters.** `--variant`, `--file-size-mb` (1024),
  `--msync-interval-ms` (0), `--backing-dir`, `--keep-backing`, `--seed`,
  `--cpu-affinity`, `--output-dir`.
- **Safety notes.** Backing path is validated by the sandbox helper. Unless
  `--keep-backing` is set, the file is removed at exit.
- **Memory-signal interpretation.** Read variant is near-invisible; write
  shows broad delta plus writeback bursts; RMW combines both. Writeback
  cadence depends on `dirty_ratio` and `dirty_writeback_centisecs` in the
  guest.

## APP-REALISTIC family

### `app_sqlite_oltp_v2`

- **Purpose.** OLTP rhythm (WAL append + checkpoint) distinct from synthetic
  primitives.
- **Behavior.** `setup` (build schema + load `--rows` rows) → `measure`
  (mixed INSERT/UPDATE/SELECT transactions) → `checkpoint` (`PRAGMA
  wal_checkpoint(TRUNCATE)`) → `cleanup`.
- **Parameters.** `--db-path`, `--rows` (200000), `--tx-per-batch` (50),
  `--mix "I-U-S"` percent (40-30-30 default), `--duration`,
  `--sandbox-dir`, `--output-dir`, `--seed`, `--cleanup`, `--keep-db`.
- **Memory-signal interpretation.** Bimodal cepstral structure: WAL append
  cadence + periodic checkpoint bursts.

### `app_sqlite_analytical_v2`

- **Purpose.** Read-heavy analytical queries vs OLTP.
- **Behavior.** `setup` → `prewarm` (large COUNT/SUM to fill cache) →
  `measure` (mix of aggregate / top-N / range queries) → `cleanup`.
- **Memory-signal interpretation.** Near-IDLE during pure reads; small write
  bursts when an aggregation materializes a temp table.

### `app_compress_gzip_v2`

- **Purpose.** Continuous CPU + IO mix from gzip compression.
- **Behavior.** `setup_input` (generate `--input-size-mb` MiB random bytes) →
  `compress` (`gzip.open` write loop) → `cleanup`.
- **Safety notes.** Cap on input size: 4 GiB. Random input is
  near-incompressible so the CPU+IO ratio is stable.

### `app_decompress_gzip_v2`

- **Purpose.** Inverted IO direction (small read, large write).
- **Behavior.** `setup_input` (build then compress) → `decompress` → `cleanup`.

### `app_json_parse_v2`

- **Purpose.** Streaming JSON parsing as an allocation-heavy workload.
- **Behavior.** `setup_input` (write a deterministic 500 MiB JSONL file) →
  `parse` (line-by-line `json.loads` with running aggregates) → `cleanup`.

### `app_hashtable_intensive_v2`

- **Purpose.** Two-phase trajectory: write-dominant build, read-dominant probe.
- **Behavior.** `build` (open-addressing linear probing inserts) →
  `probe` (lookups; 1/8 of which intentionally miss).
- **Parameters.** `--capacity-pow2` (24), `--inserts`, `--lookups`,
  `--duration`, `--seed`, `--cpu-affinity`, `--no-mlock`.

## SECURITY-LIKE family (safe simulations)

All five tests use `common/phase2_sandbox.h` for sandbox creation, file
generation, reversible XOR transform, and validated cleanup. See
`docs/SAFETY_MODEL.md` for the full guarantee list.

### `sandbox_ransom_seq`

- **Purpose.** Per-file 5-phase sequential pattern (stat → read → XOR →
  write → rename) as a behavioral simulation.
- **Behavior.** `generate` (create `--files` sandbox files) → loop: per
  file run the five phases.
- **Parameters.** `--files` (1000), `--file-size-bytes` (1 MiB),
  `--duration` (600), `--sandbox-dir` (default `/tmp`), `--no-cleanup`.

### `sandbox_ransom_batched`

- **Purpose.** Same five phases as `_seq`, but batched: stat-all → read-all
  → transform-all → write-all → rename-all. Designed for mechanism-aligned
  segment boundaries.
- **Parameters.** As `_seq`, plus `--mem-cap-mb` (1024) that limits
  `--files * --file-size-bytes`. Refuses to start if the cap would be
  exceeded.

### `sandbox_ransom_slowburn`

- **Purpose.** Slow cadence: one file every `--interval-s` seconds.
- **Parameters.** `--files` (100), `--file-size-bytes` (1 MiB),
  `--interval-s` (5), `--duration` (600).

### `sandbox_ransom_selective`

- **Purpose.** Discovery-phase filter cost. The sandbox holds 2N files
  (`.dat` + `.bin`); only `.dat` are processed.
- **Parameters.** `--files` (500 per extension), `--file-size-bytes`
  (512 KiB), `--duration`.

### `sandbox_scanner_metadata`

- **Purpose.** Pure scan (`lstat` everything) without read / transform / write.
- **Parameters.** `--files` (5000), `--subdirs` (50), `--file-size-bytes`
  (4096), `--passes` (5), `--duration`.

## METHODOLOGY family

### `mp_phase_boundary_inference`

- **Purpose.** End-to-end harness for phase-boundary detection. Drives a
  child sandbox binary with `--phase-markers` and compares predicted vs
  ground-truth boundaries.
- **Parameters.** `--child-binary` (path), `--child-args` (extra args
  string), `--detector {fixed,diff}`, `--tolerance-s` (0.5).
- **Output.** F1, precision, recall, lists of true / predicted boundary
  timestamps.

### `mp_workingset_metric_linearity`

- **Purpose.** Fit metric response curves vs working-set or write magnitude.
- **Parameters.** `--inputs-dir` (a directory of sweep metadata),
  `--family {workingset,writemag}`, `--metric-keys` (comma-separated).
- **Output.** Linear and power-law fit parameters per metric.

## Connection to Phase 2 plan

The Phase 2 plan and the v2 site cards list these eighteen tests as the
authoritative inventory; this catalog matches that list one-to-one and does
not add or skip any test. Where the plan reserves variants (e.g.
`mem_writemag_sweep_v2` variants A–D), the same executable is invoked with
different `--bytes-per-page` values rather than separate binaries. The
metadata JSON identifies the variant via its parameters block.
