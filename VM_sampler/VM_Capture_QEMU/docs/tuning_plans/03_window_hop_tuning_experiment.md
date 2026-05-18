# 03 — Window / Hop Tuning Experiment

**Status:** plan, not yet implemented.
**Depends on:** [`01_instrumentation_logging_plan.md`](./01_instrumentation_logging_plan.md), [`02_interval_tuning_experiment.md`](./02_interval_tuning_experiment.md).
**Feeds:** [`04_k_segmentation_tuning_plan.md`](./04_k_segmentation_tuning_plan.md).
**Estimated wall-clock cost:** *zero new captures.* This plan re-uses
the runs collected in plan 02 and just re-runs `offline_step_metrics.py`
over those existing frames at different `(window_size, step_size)` pairs.
Total compute: ~10 host CPU-hours.

## Question this plan answers

> Given a per-family `intervalMsec` chosen in plan 02 (and therefore a
> known Δt_frame in guest time), what `(window_size, step_size)` pair
> gives the best signal-to-noise on the defining metric for each
> family — keeping in mind that
> `window × intervalMsec` is the lowest-frequency rhythm a single
> window can resolve, and `hop × intervalMsec` is the temporal
> resolution of the sliding statistics?

The Phase 1 canonical `(128, 64)` was chosen for Phase 1 workloads at
100 ms `intervalMsec`. Phase 2's rhythms (sandbox phases, sqlite
checkpoint, hashtable build→probe transition) and its potentially
different `intervalMsec` profiles may demand different choices.

## Why this is decoupled from `intervalMsec`

Both quantities multiply Δt_frame to produce a guest-time scale, so
naively they look entangled. They are not, because:

- `intervalMsec` controls the **Nyquist ceiling** (`1 / (2 ×
  intervalMsec)` Hz in guest time).
- `(window_size × intervalMsec)` controls the **lowest resolvable
  frequency** in a single window.
- `(step_size × intervalMsec)` controls the **temporal resolution of
  the sliding-window statistics**.

For a fixed `intervalMsec`, `window_size` and `step_size` set the
spectral leakage and the resolution-vs-noise trade-off. The cleanest
way to study them is to *hold `intervalMsec` fixed* and sweep
window/hop, which is exactly what this plan does.

## 1. Independent variables

| Variable     | Levels                                            | Why                                                                                  |
| ------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `window_size`| {32, 64, 128, 256, 512}                           | Spans 4× below and 4× above the Phase 1 canonical.                                   |
| `step_size`  | {window/4, window/2, window} (i.e. 75 / 50 / 0 % overlap) | Standard short-time analysis ratios; the canonical is 50 % (hop = window/2). |
| `intervalMsec` | the per-family value chosen in plan 02          | One value per family; the analyzer re-uses plan 02's runs at that interval.          |
| Workload     | the same two probes used in plan 02 + one
representative APP-REALISTIC (`app_sqlite_oltp_v2`)                  | Phasic + steady + bimodal.                                                          |

5 × 3 × 3 workloads = 45 analyzer-side cells per family. Since the
analyzer is offline, we run every cell on the same captured frames —
no new captures are needed.

## 2. Dependent variables

For each cell:

- **Spectral resolution:** width of the dominant cepstral peak in
  bins, then converted to guest-time period.
- **Spectral leakage:** ratio of peak height to neighbor-bin height.
  Larger windows reduce leakage; smaller windows widen the main
  lobe.
- **Per-window stationarity:** within-window variance of the
  per-snapshot delta — a window so long that the workload's behavior
  changed inside it is a bad window. Measured by Phase 1's
  `StabilityValidator`.
- **Defining metrics (re-used from plan 02):**
  - F1 of segmenter boundaries for `sandbox_ransom_batched`.
  - CV of `active_page_fraction` for `mem_workingset_sweep_v2`.
  - Cepstral periodicity score at the checkpoint cadence for
    `app_sqlite_oltp_v2`.

## 3. Acceptance criteria

A `(window_size, step_size)` pair is accepted for a workload family
if **all** hold:

1. **Stationarity.** `StabilityValidator` accepts ≥ 90 % of windows.
2. **Spectral coverage.** `window × intervalMsec` ≥ 4 × (workload's
   slowest rhythm of interest, in guest time). For
   `sandbox_ransom_batched` with five ~10-s phases, the slowest
   rhythm is 50 s; a 50-s window at `intervalMsec=500ms` is
   `window=100` — so the chosen `window` must be ≥ 100.
3. **Defining metric meets the plan-02 target.** Same thresholds as
   plan 02 (F1 ≥ 0.8, CV ≤ 0.15, etc.).
4. **Hop is ≤ window/2.** Higher hops alias the sliding statistic.

Among pairs that pass, pick the **smallest window** (because smaller
windows give finer time-localization), with `hop = window / 2` unless
spectral leakage on that workload demands `hop = window / 4`.

## 4. Default fallbacks

If no `(window_size, step_size)` pair from the sweep passes the
acceptance criteria for a given family, the choices are, in order:

1. Re-run plan 02 with a faster `intervalMsec` for that family
   (loop back, max one iteration per the README's policy).
2. Lengthen the test's guest duration so `n_windows` grows.
3. Tighten the workload's per-test parameters (e.g. fewer / larger
   files in `sandbox_ransom_seq`) so phases are longer in guest
   time.

Only after all three fail do we relax the acceptance thresholds
themselves — and only with explicit team sign-off.

## 5. Output

A table per family, similar in shape to plan 02's deliverable:

| Family                            | `intervalMsec` | `window_size` | `step_size` | Rationale                                |
| --------------------------------- | -------------: | ------------: | ----------: | ---------------------------------------- |
| MEM steady-state                  |          TBD   |          TBD  |        TBD  | Short windows OK; steady-state stats.    |
| APP-REALISTIC OLTP                |          TBD   |          TBD  |        TBD  | Window must contain ≥ 2 checkpoint periods. |
| SECURITY-LIKE batched             |          TBD   |          TBD  |        TBD  | Window covers ≥ 1 full five-phase cycle. |
| SECURITY-LIKE slowburn            |          TBD   |          TBD  |        TBD  | Hop tuned to per-file interval.          |
| ... (one row per family)          |                |               |             |                                          |

The values feed into the Phase 2 metadata schema so the analyzer
applies them per-test rather than as a global.

## 6. Implementation surface

- **`offline_step_metrics.py`** — already accepts
  `--window-size` and `--step-size`. The plan needs no producer
  changes; only a small sweep driver script that iterates over
  `(window_size, step_size)` and writes a CSV of metric outputs.
  ~80 lines of Python.
- **`_MIN_WINDOWS_PER_SEGMENT_DEFAULT = 50`** in
  `offline_step_metrics.py:246` — kept for now; the segmenter floor
  is tuned in plan 04.
- **Phase 2 metadata schema** — add `window_size`, `step_size`
  alongside `interval_msec` so the analyzer applies them per-test.

## 7. Open question for plan 04

Whether the segmenter's minimum-windows-per-segment floor should
change with `window_size`. A larger window naturally produces fewer
windows for the same trace; lowering the floor proportionally is
tempting but may degrade segmenter stability. Plan 04 owns this.

## 8. Risks

- **Stale captures.** If plan 02's runs were collected at
  parameters that get tweaked later, this plan must be re-run.
  Mitigated by storing `(intervalMsec, T_guest, vm_ram_mb,
  flush_kept)` alongside every captured frame and only mixing
  matched runs.
- **Cherry-picking.** The temptation to pick the
  `(window_size, step_size)` that maximizes F1 on
  `sandbox_ransom_batched` will be high. Resist: pick the one that
  passes acceptance on **all** representative workloads, not the
  one that wins on the easy case.
- **Spectral artifacts at very small windows.** Below 32 frames,
  cepstral peak estimation becomes noisy. The pilot's lower bound
  is set at 32 for this reason; do not go lower without a separate
  validation.
