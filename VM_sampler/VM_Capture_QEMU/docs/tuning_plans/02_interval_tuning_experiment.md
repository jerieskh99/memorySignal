# 02 — `intervalMsec` Tuning Experiment

**Status:** plan, not yet implemented.
**Depends on:** [`01_instrumentation_logging_plan.md`](./01_instrumentation_logging_plan.md).
**Feeds:** [`03_window_hop_tuning_experiment.md`](./03_window_hop_tuning_experiment.md), [`04_k_segmentation_tuning_plan.md`](./04_k_segmentation_tuning_plan.md).
**Estimated wall-clock cost:** ≈ 50 host hours total (factorial pilot, see Section 4).

## Question this plan answers

> Given the corrected timing model — `intervalMsec` is Axis A, not
> Axis B — what value (or per-family profile) of `intervalMsec`
> gives the best **guest-time temporal resolution** within a
> tolerable **host wall-clock budget**, for each Phase 2 workload
> family?

This is the question that the original (and incorrect) study tried
to answer. With instrumentation from plan 01 in place, the answer
can be measured rather than estimated.

## Inputs assumed available

- Plan 01 has landed; the producer emits per-snapshot timestamps and
  the consumer emits `timing_summary.json` for every run.
- The Phase 2 catalog in
  `VM_executables_phase2/docs/TEST_CATALOG.md` is the authoritative
  list of workloads and their per-family rhythms of interest.
- The current Phase 1 canonical `window_size=128`, `step_size=64`
  are still in effect *during this plan*; tuning them is plan 03's
  job.

## 1. Independent variables (the tuning knobs)

| Variable                             | Levels for the pilot                  | Why these levels                                                |
| ------------------------------------ | ------------------------------------- | --------------------------------------------------------------- |
| `intervalMsec`                       | {100, 250, 500, 1000, 2000}           | Spans Phase 1's 100 ms and a 20× slowdown.                       |
| Guest duration `T_guest`             | {1, 2, 5, 10} minutes (guest time)    | Enough range to cover both single-window and multi-window runs. |
| VM RAM (Axis B lever)                | {256, 512, 1024} MiB                  | Validates the throughput claim from Section 10 of the study.    |
| Flush sleep (`sleep 0.5` line 114)   | {keep, remove}                        | Validates the "redundant flush" hypothesis from Q3 in the QA.   |
| Workload                             | `sandbox_ransom_batched`, `mem_workingset_sweep_v2` | Phasic + steady probes per Phase 1's playbook.   |

5 × 4 × 3 × 2 × 2 = 240 cells — too many. The pilot subsamples:

- **Core pilot (40 runs):** 5 intervals × 4 durations × 2 workloads,
  fixed VM RAM = 1024 MiB, fixed flush = keep.
- **VM-RAM sweep (12 runs):** 3 RAMs × 2 intervals (100 ms, 1 s) ×
  2 workloads, fixed duration = 5 min.
- **Flush sensitivity (8 runs):** 2 flush settings × 2 intervals
  (100 ms, 1 s) × 2 workloads, fixed duration = 5 min, fixed
  RAM = 1024 MiB.

Total = 60 runs, all using the same instrumentation. Two replicates
of each = 120 runs ≈ 50 host hours at default 1 GiB RAM.

## 2. Dependent variables (what we record)

From plan 01's per-run summary:

- `guest_dt_mean_s`, `guest_dt_std_s`, `guest_dt_p99_s` (Axis A)
- `host_dt_mean_s` (Axis B)
- `pmemsave_ns_mean`
- `vm_pause_fraction`
- `backpressure_events`
- `snapshot_count` and `n_windows`

Plus per-workload defining metrics:

- `sandbox_ransom_batched`:
  - F1 score of detected vs ground-truth phase boundaries (already
    emitted by `mp_phase_boundary_inference`).
  - Per-segment metric trajectory (active_page_fraction,
    snr_high_frac, cep_periodicity_score) at `window=128`,
    `hop=64`.
- `mem_workingset_sweep_v2`:
  - `CV` (coefficient of variation) of `active_page_fraction` across
    windows — low CV is the steady-state goal.
  - Mean active_page_fraction itself.

## 3. Acceptance criteria per cell

A cell passes if **all** of these hold:

1. **Axis-A stationarity.** `guest_dt_std_s / guest_dt_mean_s < 0.10`
   *and* `guest_dt_p99_s < 1.25 × guest_dt_mean_s`.
2. **`intervalMsec` honored.** `|guest_dt_mean_s − intervalMsec/1000|
   < 0.02 × intervalMsec/1000`. If this fails the producer is
   gating in some non-backpressure way (e.g. queue I/O latency).
3. **No silent gaps.** `backpressure_events / snapshot_count < 0.01`.
4. **Window count adequate.** `n_windows ≥ 50` for the
   `_MIN_WINDOWS_PER_SEGMENT_DEFAULT` floor.
5. **Defining metric reaches its target.**
   - `sandbox_ransom_batched`: F1 ≥ 0.8 at k = 5.
   - `mem_workingset_sweep_v2`: CV ≤ 0.15.

A failure in (1) or (3) rejects the cell. A failure in (4) or (5)
rejects only that workload at that `(intervalMsec, T_guest)` pair.

## 4. Statistical model

For each defining metric `y`, fit a mixed-effects model:

```
y ~ log10(interval_msec) + log10(T_guest_s)
    + log10(interval_msec) × log10(T_guest_s)
    + vm_ram_mb
    + flush_kept
    + (1 | workload) + (1 | replicate)
```

Bonferroni-corrected pairwise interval contrasts find the slowest
`intervalMsec` that still meets all acceptance criteria, per
workload. The slowest meeting all criteria is the
preferred per-family value (subject to manual sanity check).

For the throughput trade-off, fit:

```
host_dt_total_s ~ snapshot_count + vm_ram_mb + flush_kept + (1 | host_load)
```

The coefficient on `vm_ram_mb` quantifies how much smaller VMs help;
the coefficient on `flush_kept` quantifies the flush-sleep cost.

## 5. Family-level recommendations (target output)

The deliverable of this plan is a table like:

| Family                                           | Recommended `intervalMsec` | Recommended guest duration | Notes |
| ------------------------------------------------ | -------------------------: | -------------------------: | ----- |
| MEM steady-state                                 |                  TBD       |                       TBD  | Low rhythm needs; large `intervalMsec`. |
| MEM transient (pagefault, mmap)                  |                  TBD       |                       TBD  | Constrained by `--msync-interval-ms`. |
| APP-REALISTIC OLTP                               |                  TBD       |                       TBD  | Constrained by checkpoint cadence. |
| APP-REALISTIC steady (compress, analytical, json, hashtable) | TBD                |                       TBD  | Two-phase trajectory. |
| SECURITY-LIKE batched / scanner / slowburn       |                  TBD       |                       TBD  | Coarse phase structure. |
| SECURITY-LIKE seq / selective                    |                  TBD       |                       TBD  | May need workload parameter retuning rather than fast `intervalMsec`. |
| METHODOLOGY                                      |                  inherits  |                  inherits  | Inherits from the child workload it drives. |

The table is filled in by the pilot. The current Phase 1 default
(100 ms) becomes a *legacy* setting retained for backward-compat
runs only.

## 6. Cross-cutting validation

After per-family values are chosen:

1. **Verify Phase 1 ↔ Phase 2 cepstral consistency.** Re-run one or
   two Phase 1 workloads (`mem_stream`, `io_seq_fsync`) at the new
   `intervalMsec` and confirm cepstral peak indices shift by the
   expected factor (`new_intervalMsec / old_intervalMsec`).
2. **Confirm Δt scaling on the segmenter.** Pick the same
   `sandbox_ransom_batched` run at 100 ms and at 1 s, confirm that
   detected phase boundaries align in *guest time* (modulo coarser
   resolution at 1 s).

## 7. Open follow-up: the `sleep 0.5` flush

The flush-sensitivity arm tests whether removing line 114 changes
any of:

- `pmemsave_ns_mean` (it should not — pmemsave returns
  synchronously).
- Dump integrity (consumer error rate; should be zero either way).
- Axis B (should drop by ≈ 0.5 s per snapshot).

If the flush proves redundant on the test host, recommend removing
it in a tiny follow-up patch. If even one dump fails without it,
keep it and document why.

## 8. Implementation surface

- **`run_files_controlled.py`** — accept a per-step `interval_msec`
  parameter and emit a config snapshot for the producer before
  launching capture. ~30 lines of orchestration plumbing.
- **`offline_step_metrics.py`** — use `guest_dt_mean_s` from the
  per-run summary as Δt_frame instead of assuming the global
  `intervalMsec`. ~10 lines.
- **Analyzer side-script (new):** the statistical model above as a
  small pandas + statsmodels notebook in a `tuning_plans/results/`
  folder (created on first run, not part of this plan).
- **Phase 2 metadata schema:** add `interval_msec`, `vm_ram_mb`,
  `flush_kept` to the per-test metadata JSON for cross-link.

## 9. Risks

- **Host noise.** A noisy host can corrupt Axis B even with the
  correct `intervalMsec`. Pin the host's CPU governor to
  performance and disable any background indexers before each
  pilot row.
- **VM-RAM sweep affects working-set tests.**
  `mem_workingset_sweep_v2` defaults to `--working-set-mb 1024`,
  which will not fit in a 256 MiB VM. For the RAM-sweep arm, drop
  the workload's working-set to `vm_ram_mb / 2` so the test is
  still inside the guest's physical RAM.
- **Backpressure can mask the answer.** If `backpressure_events`
  fires often, the cell rejects via criterion (3), but the *cause*
  may be slow disk on the host, not anything about
  `intervalMsec`. Plan 01's instrumentation lets us tell which.
- **Concurrent host load.** The pilot expects a quiet host. Any
  competing process (a background build, an Antivirus scan) will
  inflate Axis B. Run on a dedicated host or schedule for off-hours.
