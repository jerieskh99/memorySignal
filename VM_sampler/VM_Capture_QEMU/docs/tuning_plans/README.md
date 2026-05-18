# Tuning plans — index

Companion to [`../SNAPSHOT_INTERVAL_TUNING_STUDY.md`](../SNAPSHOT_INTERVAL_TUNING_STUDY.md),
[`../SNAPSHOT_INTERVAL_QA.md`](../SNAPSHOT_INTERVAL_QA.md), and
[`../snapshot_interval_qa.html`](../snapshot_interval_qa.html). The
documents in this folder are **plans-to-be-implemented-later**, not
executed work. They define experiments that must run in sequence
because each depends on the instrumentation / metric definitions
established by the previous one.

> **Status.** Every file in this folder is planning text. No script,
> config, or data file has been modified. The plans below are written
> against the corrected timing model from the snapshot-interval QA;
> any future revision of that model must propagate here.

## Execution order

The order is non-negotiable: 01 must land before 02, 02 before 03, 03
before 04. Each downstream plan relies on the metrics emitted by the
previous step.

| # | Plan                                                                                       | Why it sits here                                                                                          |
| - | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| 01 | [`01_instrumentation_logging_plan.md`](./01_instrumentation_logging_plan.md)               | Adds six per-snapshot timestamps + derived aggregates. Without this, the rest cannot tell Axis A from B.  |
| 02 | [`02_interval_tuning_experiment.md`](./02_interval_tuning_experiment.md)                   | Tunes `intervalMsec` jointly with VM RAM / pmemsave knobs. Picks per-family `intervalMsec` profiles.       |
| 03 | [`03_window_hop_tuning_experiment.md`](./03_window_hop_tuning_experiment.md)               | With Δt_frame fixed by 02, finds the best `window_size` / `step_size` per analysis goal.                  |
| 04 | [`04_k_segmentation_tuning_plan.md`](./04_k_segmentation_tuning_plan.md)                   | With (Δt_frame, window, hop) fixed, tunes the segment count `k` and the `_MIN_WINDOWS_PER_SEGMENT_DEFAULT` floor. |

A single revisit pass back to 02 is allowed after 04 if the segmenter
needs a finer Δt_frame than 02 initially picked. This loop is bounded
to one iteration; otherwise we churn forever.

## Cross-cutting principles

1. **Axis discipline.** Every plan refers to the three time axes
   defined in `SNAPSHOT_INTERVAL_QA.md`. Spectral and segmenter
   parameters live on Axis A (guest-time); throughput targets live on
   Axis B / D.
2. **Defining metrics are agreed up front.** Each plan names the
   accept/reject metric *before* any data is collected. No metric is
   ever picked post-hoc to make the result look good.
3. **One independent variable per pilot row.** Holding everything else
   fixed prevents the "we changed three things at once" failure mode
   that contaminated the original 100 ms decision.
4. **All decisions are saved in metadata.** Every Phase 2 run records
   the active `(intervalMsec, window, hop, k_target, vm_ram_mb)` in
   its per-run metadata JSON, so re-analysis a year from now can be
   commensurable.
5. **No script edits during plan review.** This folder describes
   experiments; running them requires code changes (instrumentation
   in the producer, config-snapshot plumbing in the orchestrator,
   metadata-schema additions in the analyzer). Those edits are out
   of scope for the planning phase and listed in each plan's
   "Implementation surface" section.

## Open questions deferred until each plan runs

| Question                                                                                       | Resolved by | Notes                                                                                              |
| ---------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------- |
| Is the `sleep 0.5` flush after pmemsave (line 114) redundant?                                  | 02          | Cheapest pilot row: remove it on one branch, measure Axis B and dump completeness.                 |
| What is the right `intervalMsec` for each Phase 2 family?                                      | 02          | Output of the per-family acceptance criteria.                                                      |
| Is `window=128, hop=64` the right pair for Phase 2 rhythms?                                    | 03          | Phase 1 canonical defaults; Phase 2 rhythms may demand different.                                  |
| Is `_MIN_WINDOWS_PER_SEGMENT_DEFAULT=50` the right floor?                                      | 04          | A hard floor in `offline_step_metrics.py:246`. Could be reduced for short runs with care.          |
| For each Phase 2 family, what's the right `k`?                                                 | 04          | Driven by the workload's phase structure + segmenter performance vs ground truth.                  |

## Where the plans connect to the rest of the codebase

- Producer: `capture_producer_qemu_pmemsave.sh` — adds the six
  per-snapshot timestamps in plan 01.
- Orchestrator: `run_files_controlled.py` — writes per-step config
  snapshots with the chosen `intervalMsec` in plan 02.
- Offline analyzer: `offline_step_metrics.py` — reads
  `(window_size, step_size, k)` per run from metadata, exposes
  acceptance metrics in plans 03 / 04.
- Phase 2 catalog: `VM_executables_phase2/docs/TEST_CATALOG.md` —
  per-family rhythm requirements (read-only input for plan 02).

## How to use this folder

1. Read the plans in order. Each is self-contained and includes its
   own acceptance criteria.
2. Decide whether the plan as written matches the team's goals; mark
   it `approved` (add a header line) before any code change.
3. Implement the producer / orchestrator / analyzer changes named in
   each plan's "Implementation surface" section.
4. Run the experiment.
5. Append the results to a sibling `results/` folder (to be created
   when the first plan executes; not part of this initial planning
   pass).
