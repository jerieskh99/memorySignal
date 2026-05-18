# 04 — Segment-Count `k` and `min_windows_per_segment` Tuning Plan

**Status:** plan, not yet implemented.
**Depends on:** [`01_instrumentation_logging_plan.md`](./01_instrumentation_logging_plan.md), [`02_interval_tuning_experiment.md`](./02_interval_tuning_experiment.md), [`03_window_hop_tuning_experiment.md`](./03_window_hop_tuning_experiment.md).
**Estimated wall-clock cost:** zero new captures. Re-uses plan 02's
frames at plan 03's `(window_size, step_size)`. ~10 host CPU-hours of
offline re-analysis.

## Question this plan answers

> Given a per-family `(intervalMsec, window_size, step_size)` triple
> chosen in plans 02 and 03, what is the right segment count `k` for
> each Phase 2 workload — and is the offline analyzer's
> `_MIN_WINDOWS_PER_SEGMENT_DEFAULT = 50` floor still the right one?

`k` is workload-specific: the batched-ransom test has five
mechanism-aligned phases by construction (`k = 5`); steady-state
tests are single-segment (`k = 1`); the slowburn test has many
small per-file events but no run-level multi-phase structure.

## Why this is the last plan

`k` is downstream of `(intervalMsec, window, hop)` because:

- A smaller window or larger hop reduces the number of windows
  available for segmentation; if `n_windows < k × min_windows`, the
  segmenter cannot run.
- Phase 1 set `min_windows_per_segment = 50` based on its own
  workload structure; Phase 2's longer phases and shorter per-file
  events may demand different values.
- The segmenter's F1 against ground-truth phase markers (already
  emitted by the Phase 2 workloads' `--phase-markers` flag) is
  jointly sensitive to all three.

## 1. Inputs assumed available

- Plan 01's instrumented producer is in production.
- Plan 02's per-family `intervalMsec` recommendations have been
  adopted.
- Plan 03's per-family `(window_size, step_size)` recommendations
  have been adopted.
- The pilot's runs (or a fresh set of equivalent runs) are
  available with `--phase-markers` enabled so ground-truth
  boundaries exist.

## 2. Independent variables

| Variable                   | Levels                                   |
| -------------------------- | ---------------------------------------- |
| `k` (segment-count target) | per workload from the table below        |
| `min_windows_per_segment`  | {10, 25, 50, 100}                        |
| Workload                   | one representative per Phase 2 family    |

Ground-truth `k` per workload:

| Workload                             | Expected `k` |
| ------------------------------------ | -----------: |
| `sandbox_ransom_batched`             | 5            |
| `sandbox_ransom_seq`                 | 5 × `n_files` (per-file × phases), but the *run-level* `k` is 1 (one long sequential block); the segmenter is not expected to recover per-file phases |
| `sandbox_ransom_slowburn`            | 1 (with very high `cepstral_peak_idx`) |
| `sandbox_ransom_selective`           | 2 (discovery + processing)             |
| `sandbox_scanner_metadata`           | 1                                       |
| `mem_workingset_sweep_v2`            | 1                                       |
| `mem_writemag_sweep_v2`              | 1                                       |
| `mem_rmw_intensity_v2`               | 1                                       |
| `mem_pagefault_density_v2` variant fault_only | 2 (fault-only decay → idle)    |
| `mem_pagefault_density_v2` variant touch_only | 1                              |
| `mem_pagefault_density_v2` variant mixed | 1 with monotonic drift             |
| `mem_mmap_traversal_v2`              | 1                                       |
| `app_sqlite_oltp_v2`                 | depends on checkpoint cadence; expected 1 (the rhythm shows up in cepstra, not as a segment break) |
| `app_sqlite_analytical_v2`           | 1                                       |
| `app_compress_gzip_v2`               | 1                                       |
| `app_decompress_gzip_v2`             | 1                                       |
| `app_json_parse_v2`                  | 1                                       |
| `app_hashtable_intensive_v2`         | 2 (build → probe)                       |
| `mp_phase_boundary_inference`        | inherits child workload                 |
| `mp_workingset_metric_linearity`     | n/a (analysis only)                     |

So **most Phase 2 workloads are k = 1**; the segmenter's job is the
exception, not the rule.

## 3. Dependent variables

- **F1 of detected segment boundaries vs ground-truth**
  (from `mp_phase_boundary_inference`).
- **Per-segment metric stationarity** — windows within a segment
  should have within-segment variance < across-segment variance by
  some margin (target: ratio ≤ 0.5).
- **Segmenter rejection rate** — fraction of runs where the
  segmenter cannot produce `k` segments because
  `n_windows < k × min_windows`.

## 4. Acceptance criteria

For each (workload, k) pair, accept the smallest
`min_windows_per_segment` that satisfies **all**:

1. F1 ≥ 0.8 on ground-truth-rich workloads
   (`sandbox_ransom_batched`, `sandbox_ransom_selective`,
   `app_hashtable_intensive_v2`, `mem_pagefault_density_v2:fault_only`).
2. Segmenter rejection rate < 5 % across the pilot runs at the
   plan-02 / plan-03 `(intervalMsec, window, hop)` for that family.
3. Per-segment stationarity ratio ≤ 0.5 (i.e. within-segment
   variance is at most half the across-segment variance).
4. For workloads where `k = 1` is the ground truth, the segmenter
   must not split spuriously — accept only if forced-k=1 yields
   stationarity-ratio ≤ 0.5 *and* a forced-k=2 split increases that
   ratio (i.e. the data does not want to be split).

## 5. Output

Two tables.

**Per-family `(min_windows_per_segment, k_default)`**:

| Family                | `min_windows_per_segment` | Default `k` | Notes |
| --------------------- | ------------------------: | ----------: | ----- |
| MEM steady-state      |                       TBD |           1 |       |
| MEM transient         |                       TBD |    1 or 2   | per-variant |
| APP-REALISTIC         |                       TBD |    1 or 2   | per-test |
| SECURITY-LIKE batched |                       TBD |           5 |       |
| SECURITY-LIKE seq     |                       TBD |           1 | per-file phases sub-segment, not run-level |
| SECURITY-LIKE selective |                     TBD |           2 |       |
| METHODOLOGY           |                  inherits |    inherits |       |

**Global default override decision:** keep
`_MIN_WINDOWS_PER_SEGMENT_DEFAULT = 50` (yes/no/with-caveats).

## 6. Implementation surface

- **`offline_step_metrics.py:246`** — make
  `_MIN_WINDOWS_PER_SEGMENT_DEFAULT` configurable from a per-run
  metadata field rather than a module-level constant. ~20 lines.
- **Per-test metadata schema (in `VM_executables_phase2/common/`)** —
  add `expected_k_run_level` (default 1) and
  `expected_per_segment_min_windows` (default `null` → inherit
  global). ~10 lines.
- **A small `segmenter_audit.py` driver** — runs the segmenter at
  the table's `(k, min_windows)` per workload and writes the
  acceptance metrics to a CSV. ~150 lines.

## 7. Risks

- **Ground-truth scarcity.** Only a subset of workloads emit
  `--phase-markers`. For the rest, F1 cannot be computed and the
  decision falls back on stationarity-ratio + per-test eye review.
- **k=1 workloads dominate.** Most Phase 2 tests do not need a
  multi-segment analyzer; the segmenter's value is concentrated in
  ~5 workloads. Right-size the plan's effort to that.
- **Floor change affects Phase 1 reanalysis.** If we lower
  `_MIN_WINDOWS_PER_SEGMENT_DEFAULT`, Phase 1 segment-level
  numbers may shift. Either keep the change Phase 2-specific
  (preferred) or re-run Phase 1 segment metrics at the new floor
  (expensive).

## 8. Open follow-ups (out of scope for this plan)

- Adaptive `k` selection from the cepstral / MSC trajectory itself
  (so we do not have to specify `k` per workload at all). This is
  a research direction, not a tuning question.
- Online segmentation in the streaming pipeline. Currently the
  streaming metrics emit per-window, not per-segment; a future plan
  could add streaming segmentation once the offline `(k,
  min_windows)` is settled.
