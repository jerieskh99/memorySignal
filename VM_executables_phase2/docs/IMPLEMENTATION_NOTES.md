# Implementation notes — VM_executables_phase2

This document captures the design decisions behind the Phase 2 executables.
It is meant as a working note for whoever extends or audits the
implementation.

## Language choice per test

| Family            | Language | Reason |
| ----------------- | -------- | ------ |
| MEM               | C        | Need explicit control of `mmap`, `madvise`, `mlock`, `clock_gettime`. Python GC and allocator artifacts would mask the mechanism under test. |
| SECURITY-LIKE     | C        | Same reasons + tight loop over many files; also keeps the sandbox helper and validator in one place. |
| `app_hashtable_intensive_v2` | C | Open-addressing table walks must be cache-controlled. |
| `app_sqlite_*`    | Python   | The CPython `sqlite3` module is a thin SQLite binding; control surface is the SQL + PRAGMA layer, not the C bridge. |
| `app_compress_gzip_v2`, `app_decompress_gzip_v2`, `app_json_parse_v2` | Python | Single-purpose orchestration over well-known stdlib modules. |
| METHODOLOGY       | Python   | Pure analysis; no need for C. |

No third-party packages are required. The C build needs nothing beyond a
POSIX `cc` and `libc`. The Python tests use only the standard library.

## Build portability

The target environment is the Linux guest. Macros that are Linux-only
(`MAP_POPULATE`, `MADV_NOHUGEPAGE`) are wrapped in `common/phase2_portable.h`
so the source still compiles on macOS for smoke testing. On non-Linux hosts
those macros become no-ops, so the resulting binary won't be *semantically*
equivalent — only buildable. Real Phase 2 captures must happen on the Linux
guest.

The Makefile uses `-D_GNU_SOURCE`. Compiler choice and flags can be
overridden from the command line (`make CC=clang CFLAGS=...`).

## Common helpers

- `common/phase2_common.h` — C-side metadata JSON emitter, CLI parsing
  helpers, PRNG (xoshiro256\*\*), sandbox validator, phase markers.
- `common/phase2_common.py` — Python equivalents.
- `common/phase2_sandbox.h` — `p2_sandbox_t` plus phase helpers used by all
  five SECURITY-LIKE binaries.

CLI flag conventions (used by every test):

```
--duration SEC
--output-dir PATH
--sandbox-dir PATH
--safe-root PATH
--seed N
--phase-markers
--dry-run
--cleanup
--cpu-affinity N
--help
```

Plus test-specific flags as listed in `docs/TEST_CATALOG.md`.

## Metadata schema

Every test writes (or prints) a single JSON object with at least the
following keys:

```json
{
  "test_name": "<slug>",
  "language": "C" | "Python",
  "phase2_version": "phase2-0.1",
  "parameters": { ... },
  "phases": [ {"name": "...", "t_start_s": N, "t_end_s": N}, ... ],
  "start_time": "ISO8601",
  "end_time":   "ISO8601",
  "status": "ok" | "dry_run" | "<failure>",
  "known_limitations": "..."
}
```

C-side emitter is hand-rolled and flat-only (one level of key-value pairs).
The Python emitter is nested. The downstream analyzer normalizes both.

## Artifact and confound controls

The recommendations in `VM_executables/low_level_rewrite_recommendations.md`
and `VM_executables/artifact_confusion_matrix_connection.md` are applied as
follows:

- **Warm-up separated from measurement.** All MEM tests run a `warmup` phase
  (typically a `memset`) that commits pages before the measurement window
  starts. The phase boundary is recorded in metadata so the analyzer can
  exclude it.
- **MADV_NOHUGEPAGE applied to all MEM mmaps.** Keeps page granularity at
  4 KiB so deltas align with the snapshot pipeline.
- **mlock applied best-effort.** Avoids host-side reclaim. `--no-mlock`
  flag is available when `RLIMIT_MEMLOCK` denies the call.
- **CPU pinning.** All MEM and APP-REALISTIC tests accept `--cpu-affinity`
  to remove scheduler noise.
- **Deterministic PRNG.** xoshiro256\*\* (C) and `random.Random` (Python),
  both seeded by `--seed`. The byte stream is reproducible across runs with
  the same seed.
- **File preallocation.** The mmap-traversal test uses `ftruncate` to commit
  the backing file size before measurement; the gzip tests write the entire
  input file as a setup phase outside the measurement window.
- **Write content is not all-zero.** Source bytes for the write-magnitude
  test come from the PRNG; the buffer is initialized to a non-zero pattern
  before pure-read tests so loads return useful bytes (otherwise the
  snapshot pipeline could collapse zero-content deltas).

## Sandbox validator

`p2_sandbox_validate` (C) and `validate_sandbox` (Python) share the same
rule set:

1. path must be absolute,
2. path must not contain `..` (textual rule),
3. `realpath()` must resolve under one of `/tmp/`, `/var/tmp/`, or an
   explicit `--safe-root`,
4. final component must not be a symlink (lstat refusal).

Every file-touching test calls the validator before generating any file. The
validator is used **again** by the cleanup routine: an unvalidated path is
refused.

## Why not a single multi-mode binary?

The task brief required one executable per test. Multi-mode binaries
(`mode=seq|batched|...`) would shrink the source tree, but they conflate
the “which test is running” line in metadata with the “which mode” line and
make the operator UX harder. Keeping one binary per test slug also matches
the per-test recording metadata convention.

For genuinely identical algorithms with only a parameter difference (e.g.
the four write-magnitudes in `mem_writemag_sweep_v2`), we use a single
binary with a `--bytes-per-page` flag and rely on the metadata block to
disambiguate variants. The test slug is unchanged because the algorithm is.

## How the analyzer consumes Phase 2 output

The analyzer pipeline (in `VMsig_featureExctraction/` and
`coherence_temp_spec_stability/`) reads:

1. The host-side memory snapshot stream (captured by `VM_sampler/` via
   `pmemsave`).
2. The Phase 2 metadata JSON files written by each test.
3. Per-test phase markers from stderr (`[ISO8601] [PHASE] test=... phase=...`).

Phase boundaries from (3) are aligned with snapshot windows from (1) via the
shared monotonic clock. The methodology test
`mp_phase_boundary_inference` is a stand-in for that alignment until the
real change-point detector is plugged in.

## Known limitations

- The C metadata emitter is flat. Anything that should be nested (e.g.
  per-phase timings) is emitted as `phase_<name>_start_s` / `_end_s`
  scalar keys. The Python emitter does nested arrays of phase objects.
  Downstream readers normalize both shapes.
- `mlock` is best-effort. On most non-root setups `RLIMIT_MEMLOCK` limits
  it to 64 KiB; the tests warn and continue. To remove that confound,
  increase the limit (`ulimit -l unlimited`) before invoking.
- `MAP_POPULATE` is Linux-only. On other hosts (macOS smoke build) it is a
  no-op, so the test still runs but does not pre-fault pages.
- `MADV_NOHUGEPAGE` is Linux-only. Pre-existing transparent-huge-page
  promotions are not undone; ideally THP is disabled at the guest level
  (`/sys/kernel/mm/transparent_hugepage/enabled = never`) before captures.
- `mp_phase_boundary_inference` uses a stub change-point detector. It must
  be replaced with a real per-segment metric-based detector once that
  pipeline exists.
- `app_sqlite_*` use CPython's `sqlite3` module; the bundled SQLite version
  varies per CPython build. We record `sqlite3.sqlite_version` in metadata
  for reproducibility.
