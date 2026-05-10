# Segment-Level Code Integration Plan

This document specifies how to add optional segment-level analysis to the existing per-test offline metrics pipeline with minimal disruption. It is implementation-ready but does not contain code. It is bound by `segment_level_analysis_critique_and_plan.md` (methodology) and `segment_level_analysis_context_and_connections.md` (purpose). All claims about file structure cite the existing repo.

---

## 1. Executive Summary

The current offline pipeline computes one feature vector per workload run from a single per-test `run_matrix_<test>.npy` file. The integration adds an optional flag `--segments S` to `offline_step_metrics.py`. When `S > 1`, the loaded `[T, N]` trace is split into `S` contiguous, non-overlapping temporal segments. The existing metric functions are reapplied inside each segment. Segment outputs are written under a new `segments/` subfolder of the existing per-test output directory. Run-level outputs are unchanged. With `--segments 1` (default) the pipeline is byte-identical to the current behavior.

The integration is intentionally additive. It does not modify metric definitions, the `coherence_temp_spec_stability` package, or downstream JSON consumers. It does not produce class-level or cross-test artifacts. Cross-run aggregation (subtype trajectories, confusion localization, centroid distances) is left to a separate post-processing layer that operates on the per-test segment outputs after all tests have run.

---

## 2. Authoritative Methodology Constraints

The following rules from `segment_level_analysis_critique_and_plan.md` bind this design and are referenced here so the implementation does not silently violate them:

1. Segments are quasi-independent at best. The pipeline must not produce any artifact that pretends segments are iid samples. Concretely: no leave-one-segment-out CV, no inflated `n`, no per-class statistics computed from pooled segments inside `offline_step_metrics.py`.
2. `k` is a research-question selector, not a hyperparameter. The CLI accepts a single `S`, but each invocation records `S` explicitly in metadata so downstream sensitivity grids (`S=2,4,8`) can be assembled by running the pipeline multiple times rather than by hidden defaults.
3. Each segment must contain enough windows. The provisional floor is roughly 50 windows per segment, which translates to about 3,328 frames at `window=128, hop=64` (`50 * 64 + 128`). Segments below that threshold must trigger a clear warning and be tagged in the output.
4. A mandatory `limits_of_inference` block lives in segment-level JSON, structurally enforcing the rule that segments are not class-level evidence.
5. The current per-run aggregation must remain the canonical run-level result. Segments are diagnostic, not replacements.

---

## 3. Current Pipeline Anchors (Evidence)

The integration plan rests on the following invariants observed in the repo:

| Anchor | File and location |
|---|---|
| Sole offline entry point | `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py` |
| Trace already loaded as `[T, N]` before metrics | `load_matrix()` and `data = load_matrix(args.matrix)` in `main()` |
| Per-step metric driver | `run_metrics_for_step(data, baseline, step_name, output_root, window_size, step_size)` |
| Streaming MSC/Cepstrum/PLV call | `run_all_features_streaming(data, window_size, step_size, output_prefix)` from `coherence_temp_spec_stability/stability_validator.py` (line 328) |
| Baseline-aware PLV (full + sliding) | `StabilityValidator.compute_plv_features(data)` plus a per-window loop over `data[start:end, :]` |
| Output directory convention | `<output_root>/offline/<step_name>/` containing `meta.json`, `streaming.npz`, `streaming.json`, `plv_baseline_aware.json` |
| Existing CLI shape | `--matrix`, `--step-name`, `--output-root`, `--project-root`, `--baseline-dir`, `--is-baseline`, `--window-size`, `--step-size` |
| Upstream invokers (must be left untouched) | `run_files_controlled.py:run_offline_step_metrics()` and `rebuild_matrices_and_rerun_offline.py:run_offline()` |

The data stays in time-major form `[T, N]` from the moment `load_matrix()` returns until it is consumed by `run_all_features_streaming` and `compute_plv_features`. This is the only place segmentation can be inserted without touching the metric package.

---

## 4. Proposed Minimal Architecture

### 4.1 Single integration site

All new behavior lives inside `offline_step_metrics.py`. The metric package (`coherence_temp_spec_stability/`) is not modified. The host controller `run_files_controlled.py` and the rebuild script `rebuild_matrices_and_rerun_offline.py` gain one optional argument each that they forward verbatim to the offline script.

### 4.2 Control flow with `--segments S`

```
load_matrix(args.matrix)               # unchanged
baseline = compute_or_load(...)        # unchanged

run_metrics_for_step(data, baseline, ...)   # full-run path; UNCHANGED outputs

if S > 1:
    run_segment_metrics_for_step(
        data, baseline, step_name, output_root,
        window_size, step_size, num_segments=S,
    )
```

The full-run call always runs first and always produces the existing artifacts. The segment call runs after, only if `S > 1`, and writes only into a new `segments/` subdirectory. There is no shared state. If the segment pass fails for any reason, the run-level artifacts are already on disk.

### 4.3 Segment output layout (proposed)

```
<output_root>/offline/<step_name>/
    meta.json                       # unchanged
    streaming.npz                   # unchanged
    streaming.json                  # unchanged
    plv_baseline_aware.json         # unchanged
    segments/                       # NEW, only created when S > 1
        segment_meta.json           # NEW, top-level segment-pass metadata
        seg_00/
            streaming.npz
            streaming.json
            plv_baseline_aware.json
        seg_01/
            ...
        seg_{S-1}/
            ...
```

This layout has three properties that matter for backward compatibility:

1. The per-test output directory is otherwise unchanged.
2. Existing tools that read `<step_name>/streaming.npz` and `<step_name>/plv_baseline_aware.json` continue to work without knowing segments exist.
3. A downstream segment-aggregation script can locate segment data deterministically by globbing `<step_name>/segments/seg_*/streaming.npz`.

### 4.4 Where confusion localization and centroid distances live

Per-segment confusion localization, segment-to-class-centroid distances, segment-to-subtype-centroid distances, within-subtype trajectory summaries, and the `per_subtype_summary` block in the JSON specification of `segment_level_analysis_critique_and_plan.md` Section 10 require cross-test context. They cannot be produced by `offline_step_metrics.py`, which sees one test at a time.

Those derivations belong in a separate post-processing script, executed after all tests have completed. That script is out of scope for this integration plan and should be tracked as a follow-up. This integration deliberately limits its surface to per-test segment metric extraction.

---

## 5. Function-Level Design

All new functions live in `offline_step_metrics.py`. Existing functions are not modified.

### 5.1 New helpers

```text
split_trace_into_segments(data: np.ndarray, num_segments: int) -> list[tuple[int, int]]
```
- Inputs: `[T, N]` array, integer `num_segments`.
- Returns: ordered list of `(start, end)` index pairs, contiguous, non-overlapping, covering `[0, T)`.
- Boundary policy: floor division. The last segment absorbs the remainder so `end == T`. This preserves total frame count and avoids dropping tail samples.
- Does NOT slice the array. Returning index pairs lets the caller apply slicing where memmap-friendly.

```text
analyze_segment(
    segment_data: np.ndarray,       # [L, N] view of data[start:end, :]
    baseline: np.ndarray,
    out_dir: pathlib.Path,
    window_size: int,
    step_size: int,
) -> dict
```
- Behavior: replicates the body of `run_metrics_for_step` for a single segment, writing `streaming.npz`, `streaming.json`, and `plv_baseline_aware.json` under `out_dir`.
- Reuses `run_all_features_streaming` and the `StabilityValidator` PLV loop. No new metric code.
- Returns a small dictionary with segment-level summary fields needed by `segment_meta.json` (windows produced, MSC peak SNR median, PLV median, cepstral peak idx median, success flag, error message if any).
- Internally tolerates partial failure: if a segment is too short for MSC, the function logs a warning, writes a stub JSON with a `skipped: true` flag, and returns failure status without raising.

```text
aggregate_segment_results(
    step_name: str,
    segment_records: list[dict],
    config: dict,
    diagnostics: dict,
) -> dict
```
- Builds the JSON payload for `segment_meta.json`. Pure data assembly; no I/O of its own beyond the caller passing it to `json.dump`.
- The `config` block records `num_segments`, `window_size`, `step_size`, `min_windows_per_segment`, `trace_kind`, `metric_set_id`.
- The `diagnostics` block carries flags such as `any_segment_too_short`, `total_frames`, `segment_length_used`, `tail_frames_in_last_segment`.
- Embeds the mandatory `limits_of_inference` block from `segment_level_analysis_critique_and_plan.md` Section 10.

```text
run_segment_metrics_for_step(
    data: np.ndarray,
    baseline: np.ndarray,
    step_name: str,
    output_root: str,
    window_size: int,
    step_size: int,
    num_segments: int,
) -> None
```
- Entry point invoked from `main()` only when `num_segments > 1`.
- Builds the output directory `<output_root>/offline/<step_name>/segments/` and `seg_NN/` subfolders.
- Calls `split_trace_into_segments` once.
- Calls `analyze_segment` per segment, collects `segment_records`.
- Calls `aggregate_segment_results` and writes `segment_meta.json`.
- Validates pre-conditions before looping (see Section 9).

### 5.2 Helpers reused without modification

- `load_matrix`
- `compute_and_save_baseline`
- `load_baseline`
- `_make_json_safe`
- `_write_meta`
- `run_metrics_for_step` (continues to write run-level artifacts; the segment pass is additive)

### 5.3 What must NOT be modified

- `coherence_temp_spec_stability/streaming_metrics.py` (no change to `run_streaming_on_time_series` or `save_streaming_results`)
- `coherence_temp_spec_stability/stability_validator.py` (no change to `run_all_features_streaming` or `StabilityValidator.compute_plv_features`)
- `coherence_temp_spec_stability/plv_calcolator.py`, `magnitude_squared_coherence.py`, `cepstrum_stability.py`
- `run_files_controlled.py` orchestration logic except for one passthrough argument (Section 6.3)
- `rebuild_matrices_and_rerun_offline.py` orchestration logic except for one passthrough argument (Section 6.3)

---

## 6. CLI Design

### 6.1 New flag in `offline_step_metrics.py`

```text
--segments INT  (default 1)
```

Validation rules, applied in `main()` before any work:

- `S` must be a positive integer.
- `S = 1` is allowed and is the default (no segment pass).
- `S < 1` is rejected with a non-zero exit code and a clear error message.
- `S > T` is rejected before splitting (`T` known after `load_matrix`).
- If `S > 1` and any segment would have fewer than `min_windows_per_segment * step_size + window_size` frames, the pipeline emits a single bold warning, records `any_segment_too_short = true` in `segment_meta.json`, and continues. The metric functions themselves will raise on truly empty windows; that exception is caught by `analyze_segment` and recorded per segment.
- The minimum-windows threshold is exposed as `--min-windows-per-segment INT` (default 50) but is OPTIONAL. The default reflects the methodology floor; only advanced sensitivity studies should change it.

### 6.2 Help-text wording (specification)

The `--segments` help text must explicitly state:

> Number of contiguous temporal segments to split the trace into before running the metric pipeline. Default 1 (no segmentation; current behavior). When S > 1, segment outputs are written under `segments/seg_NN/` next to the per-run outputs. Segments are NOT independent samples; see `segment_level_analysis_critique_and_plan.md` before downstream use.

This wording is part of the contract. Removing or weakening it counts as a methodology violation, not a doc nit.

### 6.3 Passthrough in upstream invokers

Two callers forward arguments to the offline script. Both gain one optional argument and one optional environment variable.

`run_files_controlled.py`:
- New env var `OFFLINE_SEGMENTS` (default `1`).
- When non-default, the controller appends `--segments <value>` to the offline command. No other change.

`rebuild_matrices_and_rerun_offline.py`:
- New CLI flag `--segments INT` (default 1).
- Forwarded to `run_offline()` and appended to the subprocess command.

These two changes are intentionally minimal. The orchestration logic, queue handling, and baseline gating are untouched.

---

## 7. JSON Output Design

The new file is `<step_name>/segments/segment_meta.json`. Per-segment artifacts under `seg_NN/` mirror the existing run-level files exactly, so any tool that already reads `streaming.npz` understands them without changes.

### 7.1 `segment_meta.json` shape (specification)

```json
{
  "schema_version": "0.1",
  "step_name": "test6_mem_alloc_touch_pages",
  "config": {
    "num_segments": 4,
    "window_size": 128,
    "step_size": 64,
    "min_windows_per_segment": 50,
    "metric_set_id": "current_per_run_v1",
    "trace_kind": "complex",
    "warmup_segments_excluded": 0
  },
  "trace": {
    "n_frames_total": 18432,
    "n_pages": 65536,
    "segment_length_frames": 4608,
    "tail_frames_in_last_segment": 0
  },
  "segments": [
    {
      "segment_index": 0,
      "frame_start": 0,
      "frame_end": 4608,
      "n_frames": 4608,
      "n_windows_expected": 71,
      "files": {
        "streaming_npz": "seg_00/streaming.npz",
        "streaming_json": "seg_00/streaming.json",
        "plv_baseline_aware_json": "seg_00/plv_baseline_aware.json"
      },
      "summary": {
        "msc_peak_snr_db_median": 1.83,
        "plv_median": 0.412,
        "cepstral_peak_idx_median": 6.0
      },
      "status": "ok",
      "error": null
    }
  ],
  "diagnostics": {
    "any_segment_too_short": false,
    "any_segment_failed": false,
    "min_windows_observed": 71
  },
  "limits_of_inference": {
    "segments_are_independent": false,
    "use_for_class_significance_tests": false,
    "valid_inferential_units": ["run", "subtype"],
    "cv_grouping": "by_run"
  }
}
```

### 7.2 What `segment_meta.json` does NOT contain

- Cross-segment trajectory metrics (within-run CV, lag-1 autocorrelation, trend slope). These can be computed from the per-segment summaries during downstream aggregation; placing them here would couple the per-test pass to a fixed analysis policy. Deferred by design.
- Class-level or subtype-level statistics. The per-test pass has no access to other tests.
- Confusion-localization predictions. Centroids are global. Computing them inside this script would require a state-sharing pattern this integration explicitly avoids.

### 7.3 Existing JSON files remain untouched

`streaming.json`, `plv_baseline_aware.json`, and `meta.json` at the per-step root are not edited, renamed, or augmented. Adding fields there would silently break downstream consumers and contradict the integration's "additive only" property.

### 7.4 Per-segment artifacts are byte-format-identical

Each `seg_NN/streaming.npz` is produced by the same `save_streaming_results()` function as the run-level file. Each `seg_NN/plv_baseline_aware.json` follows the same `{"full_run": ..., "sliding_windows": ...}` shape as the run-level file. This is intentional: it means a tool that reads run-level outputs can be reused unchanged on segment outputs.

---

## 8. Backward Compatibility

The strongest guarantee this plan offers is that `--segments 1` (the default) leaves the offline output bit-equal to the current pipeline.

Concrete preservation rules:

1. The `<step_name>/` root directory contains exactly the same set of files it does today.
2. `meta.json` keys are unchanged. The plan does NOT add a `segments` key to the run-level meta; the segment metadata lives only inside `segments/segment_meta.json`. This avoids tripping any downstream parser.
3. The `streaming.npz` and `plv_baseline_aware.json` writers are called with the same arguments as before.
4. The CLI argparse signature accepts the new flags as optional. Existing invocations without `--segments` continue to work and to produce no `segments/` directory at all.
5. `run_files_controlled.py` and `rebuild_matrices_and_rerun_offline.py` continue to invoke the offline script identically when the new env var or flag is not set.
6. Validation: a run-level diff after `S=1` invocation against a baseline run produces zero changed bytes for `meta.json`, `streaming.json`, `streaming.npz`, and `plv_baseline_aware.json`.

---

## 9. Edge Cases

The following must each have a defined behavior. None of them should cause a crash that loses run-level outputs already written.

| Case | Detection | Behavior |
|---|---|---|
| `T < window_size + step_size` (already handled at run level) | Existing check in `run_metrics_for_step` | Run-level path emits `meta.json` only and skips. Segment pass is also skipped with a logged reason. |
| `S = 1` | CLI parse | Skip the entire segment pass. No `segments/` directory created. |
| `S < 1` | CLI parse | Reject with non-zero exit, clear error. |
| `S > T` | After `load_matrix` | Reject before any segment work. Run-level pass already complete. |
| Non-integer division `T / S` | `split_trace_into_segments` | Last segment absorbs the remainder. Record `tail_frames_in_last_segment` in `segment_meta.json`. |
| One or more segments below the minimum-windows floor | Pre-loop check inside `run_segment_metrics_for_step` | Emit warning; set `diagnostics.any_segment_too_short = true`; still attempt segment metrics. |
| One segment fails inside `run_streaming_on_time_series` (e.g. MSC needs >= 2 windows) | Try/except inside `analyze_segment` | Per-segment record gets `status="failed"` and an error message. Other segments continue. `diagnostics.any_segment_failed = true`. |
| Disk write failure for one segment | Try/except around `save_streaming_results` | Log error; mark that segment failed; continue. |
| Baseline `N` mismatch | Already handled at run level | Hard error stays at the run level; segment pass never starts. |
| Output directory already exists from a previous run | Existing `mkdir(parents=True, exist_ok=True)` semantics | Segment pass overwrites prior segment files for the same `S`. If the user requests a different `S`, prior segment subfolders for the previous `S` are NOT cleaned up; this is intentional so a sensitivity grid `S=2,4,8` can be produced by three separate invocations into adjacent subfolder names if desired (see 9.1). |

### 9.1 Handling multiple `S` values in the same step output

Because `k` is a research-question selector, users will want sensitivity grids. Two policies are defensible; the plan picks one and documents it:

- **Chosen policy**: each invocation overwrites `segments/`. Different `S` values are run as separate invocations into different `output_root` directories, which keeps reproducibility cleanly tied to the directory tree.
- **Rejected**: an alternative would write `segments_S2/`, `segments_S4/`, `segments_S8/` side by side. This is more convenient for a single dashboard but couples per-test outputs to a sensitivity policy that should live downstream. The chosen policy preserves the principle that per-test outputs encode one analysis decision.

If experience later argues for the side-by-side layout, the change is local: rename the folder, update `segment_meta.json` to store its `num_segments` in its own filename suffix, and adjust the downstream aggregator. No metric code changes.

---

## 10. Validation Checks

These are the acceptance criteria for the integration. Each one is testable without writing new metric code.

### 10.1 Equivalence under `S = 1`

- Run the offline script on a representative `run_matrix_*.npy` with no `--segments` flag.
- Run the offline script on the same matrix with `--segments 1`.
- Both must produce byte-identical `meta.json`, `streaming.npz`, `streaming.json`, `plv_baseline_aware.json`, and neither must produce a `segments/` directory.

### 10.2 No-side-effect on existing callers

- Run `run_files_controlled.py` end-to-end with `OFFLINE_SEGMENTS` unset. Output tree should match a current production run for the same trace.
- Run `rebuild_matrices_and_rerun_offline.py` without `--segments`. Output tree should match a current rebuild.

### 10.3 Functional checks for `S > 1`

- For one representative trace, run with `S = 2`, `S = 4`, `S = 8`.
- Verify that `segments/` exists with `S` subfolders named `seg_00..seg_{S-1}`.
- Verify each subfolder contains `streaming.npz`, `streaming.json`, `plv_baseline_aware.json`.
- Verify that the union of segment frame intervals reconstructs `[0, T)` with no overlap and no gap.
- Verify that `segment_meta.json` exists, validates against the schema in Section 7.1, and contains the mandatory `limits_of_inference` block.

### 10.4 Minimum-windows enforcement

- Construct or pick a short trace where `S = 8` would put each segment below the floor.
- Run with `S = 8`. The pipeline should complete, mark `any_segment_too_short = true`, and emit a visible warning. Run-level outputs must still be present and unaffected.

### 10.5 Failure isolation

- Inject a forced failure inside one segment (e.g. a unit test override of `analyze_segment`).
- Verify that other segments still produce outputs, that `segment_meta.json` records the failure, and that the run-level outputs are untouched.

### 10.6 Traceability to parent recording

- Every per-segment artifact must be locatable from the parent `step_name`. Acceptance test: given an arbitrary `seg_NN/streaming.npz`, the path's grandparent equals `<step_name>` and `segment_meta.json` records the same `step_name`.

### 10.7 Non-regression of the metric package

- After integration, the `coherence_temp_spec_stability/` files must have an unchanged hash. Acceptance test: `git diff --stat coherence_temp_spec_stability/` is empty.

---

## 11. Implementation Sequence

The integration is safe to land in a single feature branch. Within the branch, the order below minimizes the time that the working tree is in a half-converted state.

1. **Add helpers without wiring**: introduce `split_trace_into_segments`, `analyze_segment`, `aggregate_segment_results`, and `run_segment_metrics_for_step` as private functions in `offline_step_metrics.py`. They are not called yet. Run the existing test invocation to confirm zero behavior change.
2. **Wire CLI flags**: add `--segments` (and `--min-windows-per-segment`) to `argparse`. Default `1`. Validate. Still no behavior change at default.
3. **Wire the segment pass**: add the conditional call to `run_segment_metrics_for_step` after the existing `run_metrics_for_step` call. Confirm that `--segments 1` continues to produce no `segments/` directory.
4. **Add upstream passthroughs**: introduce `OFFLINE_SEGMENTS` env in `run_files_controlled.py` and `--segments` flag in `rebuild_matrices_and_rerun_offline.py`. Default-off. Confirm equivalence runs.
5. **Run validation Section 10.1 and 10.2** against a known-good representative test directory before any sensitivity work.
6. **Run validation Section 10.3 through 10.6** with `S = 2, 4, 8` on a representative trace.
7. **Document the new flags** in the offline script docstring and in `docs/OFFLINE_METRICS_AND_OUTPUTS.md`. Cite this plan and the methodology files. The documentation must include the warning text from Section 6.2.

A separate downstream pass is required before any per-cycle, per-subtype, or confusion-localization claims can be made from segment output. That pass is out of scope for this integration; it consumes `<step_name>/segments/segment_meta.json` files from all tests, joins them with the existing run-level features, and produces the trajectory metrics, phase-purity tests, and confusion-localization plots described in `segment_level_analysis_critique_and_plan.md` Sections 8 and 9.

---

## 12. Thesis-Safe Interpretation Notes

These constraints are not implementation details but they are part of the contract. Any UI, dashboard, or paper that uses segment outputs must respect them, and the code is designed so that bypassing them takes deliberate effort.

1. The per-segment `streaming.npz` and `plv_baseline_aware.json` describe within-run dynamics only. Reporting any per-class statistic computed from pooled segments without group-aware cross-validation is forbidden by `segment_level_analysis_critique_and_plan.md` Section 6.1. The `limits_of_inference` block in `segment_meta.json` makes this rule machine-readable for downstream consumers.
2. `S` chosen for any one analysis must be reported with the result. `segment_meta.json` records `num_segments` precisely so that no downstream report can lose it.
3. Segment-level confusion localization is a hypothesis-generation tool and inherits the four-cause model from `confusion_matrix_diagnostic_methodology.md` Section 5. It does not produce verdicts.
4. Within-run trajectory shape stability is not the same as cross-run reproducibility. A subtype with internally consistent segments and noisy run-to-run features is still flagged as unstable at the run level. The integration cannot detect this conflict on its own; it merely makes the segment-level evidence visible so the downstream pass can flag it.
5. IDLE residual decay analysis requires per-segment ordering aware of which workload preceded the idle window. That information lives in the cycle structure documented in `VM_sampler/VM_Capture_QEMU/steps_cycle_repetition.txt` and is reconstructible from `step_name`. The per-test integration neither knows nor needs that ordering; the downstream pass handles it.
6. Cross-cycle drift questions need cycle indexing, which is not encoded by the per-test integration. The downstream pass reconstructs cycles from `step_name` patterns. The integration only guarantees that segment outputs are addressable per test.

The integration intentionally produces the smallest, most regular set of artifacts that supports every methodology-approved use described in the segment-level critique without enabling any forbidden one inside the per-test step itself.
