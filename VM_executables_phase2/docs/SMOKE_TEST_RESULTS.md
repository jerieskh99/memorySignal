# Smoke test results — VM_executables_phase2

Last run: 2026-05-18, host: macOS (Darwin 25.2), CPython 3.9.6, cc (Apple clang).
Target environment is Linux; this smoke run validates that everything
**builds** and that all flag-handling and safety paths fire correctly. Real
captures must run inside the Linux guest.

## Summary

- Built executables: **11/11** C binaries clean (no warnings).
- Python tests: **7/7** loadable, `--help` and `--dry-run` both succeed.
- Smoke battery (`scripts/smoke_all.sh`): **52 / 52 passes, 0 failures**.
- Leftover sandboxes after full battery: **0**.
- Metadata JSONs produced: **18** (one per test).

## Battery contents

For each test the battery runs three invocations:

1. `--help` — must exit 0 with non-empty output.
2. `--dry-run` — must validate parameters without touching files.
3. A small *real* invocation with the smallest legal parameters so that the
   non-dry-run code paths are exercised at least once.

Plus end-of-run inspection of `/tmp/phase2_sandbox_*` to confirm no sandbox
directory survives cleanup, and a final invocation of
`mp_phase_boundary_inference` that spawns `bin/sandbox_ransom_seq` so the
methodology-level test exercises the full child + parser path.

## Per-test results

| Family            | Test                              | --help | --dry-run | real |
| ----------------- | --------------------------------- | :----: | :-------: | :--: |
| MEM               | `mem_workingset_sweep_v2`         |   ✓    |    ✓      |  ✓   |
| MEM               | `mem_writemag_sweep_v2`           |   ✓    |    ✓      |  ✓   |
| MEM               | `mem_rmw_intensity_v2`            |   ✓    |    ✓      |  ✓   |
| MEM               | `mem_pagefault_density_v2`        |   ✓    |    ✓      |  ✓   |
| MEM               | `mem_mmap_traversal_v2`           |   ✓    |    ✓      |  ✓   |
| SECURITY-LIKE     | `sandbox_ransom_seq`              |   ✓    |    ✓      |  ✓   |
| SECURITY-LIKE     | `sandbox_ransom_batched`          |   ✓    |    ✓      |  ✓   |
| SECURITY-LIKE     | `sandbox_ransom_slowburn`         |   ✓    |    ✓      |  ✓   |
| SECURITY-LIKE     | `sandbox_ransom_selective`        |   ✓    |    ✓      |  ✓   |
| SECURITY-LIKE     | `sandbox_scanner_metadata`        |   ✓    |    ✓      |  ✓   |
| APP-REALISTIC     | `app_hashtable_intensive_v2`      |   ✓    |    ✓      |  ✓   |
| APP-REALISTIC     | `app_sqlite_oltp_v2`              |   ✓    |    ✓      |  ✓   |
| APP-REALISTIC     | `app_sqlite_analytical_v2`        |   ✓    |    ✓      |  ✓   |
| APP-REALISTIC     | `app_compress_gzip_v2`            |   ✓    |    ✓      |  ✓   |
| APP-REALISTIC     | `app_decompress_gzip_v2`          |   ✓    |    ✓      |  ✓   |
| APP-REALISTIC     | `app_json_parse_v2`               |   ✓    |    ✓      |  ✓   |
| METHODOLOGY       | `mp_workingset_metric_linearity`  |   ✓    |    ✓      |  ✓   |
| METHODOLOGY       | `mp_phase_boundary_inference`     |   ✓    |    ✓      |  ✓ (via real child)   |

## Targeted safety regression checks

The following were verified by hand outside the smoke battery (after the
final code-path was wired up):

1. `--sandbox-dir relative_path` → refused with
   `ERROR sandbox-dir must be absolute: relative_path`. Dry-run included.
2. `--sandbox-dir /tmp/../etc` → refused with
   `ERROR sandbox-dir contains '..'`. Dry-run included.
3. `--sandbox-dir /Users/jeries` → refused with
   `ERROR sandbox-dir /Users/jeries not under approved root (/tmp, /var/tmp, or --safe-root)`.
4. `--sandbox-dir /Users/jeries --safe-root /Users/jeries` → accepted
   (operator explicitly approved the root).
5. `--files 6000 --file-size-bytes 1024 --dry-run` → refused with
   `ERROR num_files=6000 outside [1..5000]`. The hard cap fires in dry-run.
6. `--duration 700` → refused (cap is 600 s).
7. `/tmp/phase2_sandbox_*` listing after the full battery: empty (cleanup
   runs on the default code path).

## Safety / source-scan grep results

```text
network syscalls   : NONE  (sqlite3.connect matches are DB calls, not sockets)
python net imports : NONE
exec / system / popen : NONE in workload code
                       (methodology/mp_phase_boundary_inference.py uses
                        subprocess.run on the operator-supplied child binary —
                        documented in IMPLEMENTATION_NOTES.md)
```

## Known surface differences vs the Linux target

These warnings appear at run-time on macOS but do not appear on the Linux
guest, where they are expected to succeed:

- `cpu pinning not supported on this OS` from `p2_pin_cpu` (the macOS shim
  prints once and skips affinity setting).
- `madvise` flags `MADV_NOHUGEPAGE`, `MAP_POPULATE` are no-ops on macOS.
- `mlock` may print `[WARN] mlock failed: Cannot allocate memory` on hosts
  where `RLIMIT_MEMLOCK` is restrictive. The tests continue.

These should be verified anew once the binaries run inside the Linux guest.
Treat this document as the **buildability and safety-path** smoke report; the
Linux-side capture run is a separate verification step.

## How to reproduce

```sh
cd VM_executables_phase2
make
bash scripts/smoke_all.sh    # 52 invocations, ~30 s
ls smoke_out/*.json | wc -l  # expect 18
ls -d /tmp/phase2_sandbox_*  # expect: no matches
```
