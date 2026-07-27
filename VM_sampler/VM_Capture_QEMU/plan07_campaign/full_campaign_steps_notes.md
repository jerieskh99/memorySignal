# Full campaign steps -- notes and judgment calls

Companion to `full_campaign_steps.txt` (101 lines, one line per workload,
in cpu/cache/thread/io/mem/mixed/app_realistic/security_like_safe/kernel
D1..D13/methodology order). Every flag in the steps file was verified
against a captured `--help` output (macOS smoke build) and/or the
argument-parsing source, then executed with `--dry-run` end to end with
zero failures. No workload required guessing an unverified flag.

## How verification was done

1. `make clean && make -j4 all` built all 94 C binaries cleanly (exit 0).
2. Captured `--help` for all 94 C binaries + 7 Python scripts (real,
   non-truncated usage text for every one -- no binary silently hung on
   `--help`).
3. Cross-checked every flag used in the steps file against the captured
   `--help` text with a small parser script; 0 unverified flags remain.
4. Ran every line (with `--dry-run` appended) end to end on macOS;
   confirmed 101/101 succeed and each writes a well-formed
   `<test>_metadata.json` with `"status": "dry_run"` and the intended
   parameter values.

## Judgment calls / things worth a second look

| Workload | What I did | Why |
| --- | --- | --- |
| `mem_writemag_sweep_v2` | Used the single default `--bytes-per-page 64`, not the documented 4-value sweep (1/64/1024/4096) from `docs/TEST_CATALOG.md`. | The task asked for one real invocation per workload, not a parameter sweep. If the campaign actually wants all 4 write-magnitude variants, that needs 4 separate lines/campaign passes -- flag for human decision if so. |
| `app_hashtable_intensive_v2` | Kept `--duration 60`, but its own `--help` says duration is "Hard cap on **each** phase" (build, then probe). | Real wall time for this one workload can be up to ~120s (2 x 60s), not 60s like every other workload. Not a bug, just worth knowing when sizing the total campaign duration. |
| `mem_rmw_intensity_v2`, `mem_pagefault_density_v2` | Did NOT add `--phase-markers`, unlike every other MEM binary. | Their own `--help` text does not list this flag (confirmed by reading the `.c` source: `p2_phase()` is called unconditionally in every MEM binary regardless of whether `--phase-markers` is passed -- the flag is a documented-but-inert no-op in the ones that DO list it). Since these two don't even document it, I left it out rather than pass an unverified flag. Passing it would very likely be harmless (unrecognized tokens are silently ignored by the shared arg scanner), but "verified" was the bar. |
| `mp_workingset_metric_linearity` `--metric-keys` | Used `writes,passes` instead of an "interesting" metric name like `active_page_fraction`. | Read `mem_workingset_sweep_v2.c`'s metadata emitter directly: it only ever writes `working_set_mb, duration_s, warmup_s, stride, seed, cpu_pin, no_mlock, passes, writes, status`, timestamps. The downstream analyzer metrics (active page fraction, Hamming delta) are computed later in the pipeline from the raw memory-snapshot stream, not embedded in this JSON yet. `writes`/`passes` are real, present, numeric fields that will produce a non-trivial linear fit against `working_set_mb`; a metric name that doesn't exist in the JSON fails silently (empty fit, no crash) rather than erroring, so this was a "pick a real field" judgment call, not a guess at an unverified flag. |
| `mp_phase_boundary_inference` | The line is `mkdir -p <child sandbox dir> && python3 ...` (one compound shell command), not two separate STEPS_FILE lines. | `run_files_controlled.py` (the actual orchestrator) auto-creates/wipes any `--sandbox-dir` or `--backing-dir` value it finds via `shlex.split()` on each STEPS_FILE line, via `wipe_guest_scratch()`, before running it -- so no other workload in this file needs a manual `mkdir`. But this one workload's sandbox path is nested inside the quoted `--child-args` string, so it is invisible to that top-level token scan. Confirmed via `shlex.split()` simulation that the orchestrator's scan genuinely misses it, and confirmed the compound `mkdir -p ... && ...` line is still named/parsed correctly by the orchestrator's `step_name_from_command()` and is safe under `SUSTAIN_LOOP` wrapping (idempotent `mkdir -p` re-run per loop iteration). |
| All other `--sandbox-dir` / `--backing-dir` workloads (13 total: `io_read_cache_hit_v2`, `io_direct_write_like_v2`, `mem_mmap_traversal_v2`, `mixed_mem_io_v2`, `mixed_cpu_io_v2`, and all 8 `security_like_safe/*`) | Deliberately do NOT include a `mkdir -p` in the steps file. | Verified by reading `run_files_controlled.py`: every STEPS_FILE line becomes its own atomic, capture-instrumented SSH step; adding a bare `mkdir -p ...` line would show up as a spurious "mkdir"-named capture cell in the campaign. The orchestrator's own `wipe_guest_scratch()` already creates (and wipes, then recreates) any `--sandbox-dir`/`--backing-dir` directory immediately before each relevant step runs -- confirmed by reading its implementation (`rm -rf ...; mkdir -p ...; true`) and by locally simulating that exact pre-step against all 101 lines (0 failures once simulated). Trust the existing mechanism rather than duplicate it. |
| `--seed 42` everywhere | Reused the shared default seed for every workload rather than varying it per line. | Matches every binary's own documented default and the `docs/TEST_CATALOG.md` examples; the task did not ask for seed diversity, and reusing the default keeps behavior fully predictable/reproducible against known smoke-test baselines. |
| Kernel default problem sizes | Left every kernel's problem-size flag (`--dim`, `--particles`, `--n`, etc.) at its documented default. | The task said to read `--help`/source rather than assume, but did not ask for a specific non-default problem size, and every kernel's binary-declared default is already the vetted/intended size (confirmed no default exceeds its own `--max-mb` cap for any of the 64 kernels). |

## Workloads flagged for human review

None. All 101 workloads (94 C binaries + 7 Python scripts) got a fully
confident, `--help`-verified, dry-run-tested invocation. Nothing was
guessed.

## One thing to double check before the real campaign (not a flag issue)

`docs/SMOKE_TEST_RESULTS.md` is stale -- it only documents an earlier
18-test subset (MEM + SECURITY-LIKE + APP-REALISTIC + METHODOLOGY) and
predates cpu/cache/thread/io/mixed/kernel entirely. It is not wrong about
what it covers, just incomplete as a map of the current ~101-workload
tree. Not something this steps file needed to fix, but worth knowing the
doc lags the code.
