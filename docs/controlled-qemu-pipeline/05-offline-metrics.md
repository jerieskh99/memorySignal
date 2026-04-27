# Offline Metrics

> **Full module import graph and used-vs-loaded files:** [`../OFFLINE_METRICS_AND_OUTPUTS.md`](../OFFLINE_METRICS_AND_OUTPUTS.md)

## Purpose
This section documents the conditional offline metrics stage that the host controller can trigger after each captured workload step.

## Relevant Files (active pipeline)
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` — invokes offline script after queue drain
- `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py` — only offline script in the active chain
- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json` — supplies `streaming.projectRoot` when `OFFLINE_PROJECT_ROOT` is unset

Analysis code is under **`coherence_temp_spec_stability/`** at `--project-root`. Only a **subset** of that directory is imported by `offline_step_metrics.py`; see the linked doc for the exact list (active computation vs import-only vs unused).

## Support Level
- Directly supported for invocation timing, CLI arguments, and output paths in `offline_step_metrics.py`
- Import graph verified against `offline_step_metrics.py`, `stability_validator.py`, and `streaming_metrics.py`

## Out Of Scope
- Plotting and ad-hoc tools in `coherence_temp_spec_stability/` that are not imported by the offline path (`plot_results.py`, `visualize_single_run.py`, `deep_behavior_plots.py`, …)
- `raw_matrix_builder.py` (consumer raw-retention path, not `offline_step_metrics.py`)

## When The Offline Stage Runs
The offline stage runs only when all of the following are true:

1. `CAPTURE_MODE=1`
2. `OFFLINE_METRICS_MODE=1`
3. a step-specific run matrix path was created
4. the producer has already been stopped
5. the queue has fully drained
6. the consumer has been stopped
7. the VM has been stopped

This ordering is enforced by `run_files_controlled.py`, which invokes offline metrics only after queue drain, consumer shutdown, and VM shutdown for a completed step. The offline analysis therefore runs as host-side post-processing while the guest is no longer executing.

## Why Live Streaming Is Disabled In This Mode
When offline metrics mode is active, the controller exports `OFFLINE_MODE=1` before starting the launcher. The consumer interprets that flag as a command to skip live streaming so metrics can be computed in a step-gated manner afterward.

## What Runs Inside `offline_step_metrics.py` (summary)
1. Load step matrix `.npy`, transpose to `[T, N]`.
2. Baseline step: `PLVStability.fit_baseline` → `baseline_plv.npy`.
3. Every step: **`run_all_features_streaming`** (MSC / cepstrum / online PLV via `streaming_metrics.py`) + **`StabilityValidator.compute_plv_features`** (baseline-aware PLV JSON).
4. Write `meta.json`, `streaming.*`, `plv_baseline_aware.json` under `<output_root>/offline/<step_name>/`.

## Inputs
The host controller passes the arguments documented in `offline_step_metrics.py` (`--matrix`, `--step-name`, `--output-root`, `--project-root`, `--baseline-dir`, window/step sizes, optional `--is-baseline`).

## Baseline Step Logic
The active offline design assumes one step is the clean baseline, controlled by `BASELINE_STEP_NUMBER` in the host controller. By default, that is step 1.

## Outputs
Per step: `<output_root>/offline/<step_name>/` with `meta.json`, `streaming` prefix outputs, `plv_baseline_aware.json`. Shared baseline: `<baseline_dir>/baseline_plv.npy`.

## Directly Implemented Versus Inferred
### Directly Implemented
- offline metrics run after queue drain, consumer shutdown, and VM shutdown
- matrix transpose convention `[pages, frames]` → `[frames, pages]`
- one baseline step can persist `baseline_plv.npy`
- per-step output subdirectory named by `step_name`

### Inferred
- full key structure inside `.npz` / JSON unless read from `streaming_metrics.save_streaming_results` and `PLVStability.evaluate_run`
