# Offline Metrics And Outputs

## What This Part Does
This document describes the **offline metrics stage** triggered when `run_files_controlled.py` runs with **`OFFLINE_METRICS_MODE=1`** (and capture enabled). It focuses on what **`offline_step_metrics.py`** actually imports and executes, and which repository files are **active analysis dependencies** versus **import-only** or **unused** for this path.

## Where It Sits In The Pipeline
Post-capture, post-queue-drain, and post-VM-shutdown: one step’s `run_matrix_*.npy` is analyzed and results are written under `<output_root>/offline/...`.

---

## Entry point

| File | Role |
|------|------|
| `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py` | Sole offline script invoked by the host controller (`python3 …` with CLI args). |

Host prerequisites (from `run_files_controlled.py`): `CAPTURE_MODE=1`, queue drained, consumer stopped, VM stopped, `OFFLINE_PROJECT_ROOT` or `streaming.projectRoot` in `CAPTURE_CONFIG`, matrix file present.

---

## Path setup (`--project-root`)

Before imports, the script prepends to `sys.path`:

1. `<project_root>/coherence_temp_spec_stability`
2. `<project_root>`

So analysis code is loaded from the **`coherence_temp_spec_stability/`** package directory at the repo root given by `--project-root`.

---

## Direct imports and calls in `offline_step_metrics.py`

**Top-level imports:** standard library, `numpy` only.

**Lazy imports inside functions:**

| Import | Used in | Purpose |
|--------|---------|---------|
| `from plv_calcolator import PLVStability` | `compute_and_save_baseline` | `fit_baseline(data)` → save `baseline_plv.npy` |
| `from stability_validator import run_all_features_streaming, StabilityValidator` | `run_metrics_for_step` | Streaming-style MSC/cepstrum/online PLV block + baseline-aware PLV JSON |

No other modules are imported by `offline_step_metrics.py` itself.

---

## Transitive imports (loading `stability_validator.py`)

Importing `stability_validator` executes its **module-level** imports (repository evidence):

```1:5:coherence_temp_spec_stability/stability_validator.py
import numpy as np
from plv_calcolator import PLVStability
from magnitude_squared_coherence import MagnitudeSquaredCoherence
from cepstrum_stability import CepstrumStability
from streaming_metrics import run_streaming_on_time_series, save_streaming_results
```

So **before** any offline function runs, Python loads:

- `coherence_temp_spec_stability/plv_calcolator.py`
- `coherence_temp_spec_stability/magnitude_squared_coherence.py`
- `coherence_temp_spec_stability/cepstrum_stability.py`
- `coherence_temp_spec_stability/streaming_metrics.py`

**Important:** `offline_step_metrics.py` calls:

- **`run_all_features_streaming`** → delegates to **`streaming_metrics.run_streaming_on_time_series`** and **`save_streaming_results`** (MSC / cepstrum / online PLV in the streaming implementation live **inside** `streaming_metrics.py`).
- **`StabilityValidator.compute_plv_features`** → uses only **`self.plv_helper.evaluate_run(...)`**, i.e. **`plv_calcolator.PLVStability`** (after the script injects `sv.plv_helper.baseline_plv = baseline`).

`StabilityValidator.__init__` still constructs **`MagnitudeSquaredCoherence`** and **`CepstrumStability`** (classes from `magnitude_squared_coherence.py` and `cepstrum_stability.py`), but **`offline_step_metrics.py` does not call** `compute_msc_features` or other MSC/cepstrum methods on the validator.

| Component | Used by offline execution paths? | Notes |
|-----------|----------------------------------|--------|
| `plv_calcolator.py` | **Yes** | Baseline fit; `evaluate_run` for `plv_baseline_aware.json` |
| `streaming_metrics.py` | **Yes** | Full `run_all_features_streaming` / save path |
| `stability_validator.py` | **Yes** | `run_all_features_streaming` definition; `StabilityValidator` + `compute_plv_features` only |
| `magnitude_squared_coherence.py` | **Import dependency only** | Loaded because `stability_validator` imports it; **not** invoked by methods used offline |
| `cepstrum_stability.py` | **Import dependency only** | Same |

**Uncertainty:** If `stability_validator.py` were refactored to lazy-import MSC/cepstrum helpers, offline might no longer need those two files at import time. **As the repository stands**, they must be importable for `from stability_validator import …` to succeed.

---

## `coherence_temp_spec_stability/` — active vs not used by this offline path

| File | Relationship to `offline_step_metrics.py` |
|------|-------------------------------------------|
| `plv_calcolator.py` | **Active** — direct and via PLV features |
| `streaming_metrics.py` | **Active** — streaming block outputs |
| `stability_validator.py` | **Active** — orchestrates the two blocks above |
| `magnitude_squared_coherence.py` | **Loaded** — not used by offline call chain |
| `cepstrum_stability.py` | **Loaded** — not used by offline call chain |
| `raw_matrix_builder.py` | **Not imported** — used elsewhere (e.g. consumer raw-retention path), not offline |
| `deep_behavior_plots.py`, `plot_results.py`, `visualize_single_run.py` | **Not imported** — not part of offline analysis |

---

## Generated outputs (explicit in `offline_step_metrics.py`)

Under `<output_root>/offline/<step_name>/` (default `output_root` from controller = capture `outputDir` unless overridden):

| Output | Description |
|--------|-------------|
| `meta.json` | Step metadata: frames, pages, window sizes |
| `streaming.<prefix>.npz` and `streaming.<prefix>.json` | Prefix is `…/streaming`; `save_streaming_results` in `streaming_metrics.py` writes compressed arrays + JSON summary |
| `plv_baseline_aware.json` | Full-run + sliding-window baseline-aware PLV (from `StabilityValidator.compute_plv_features`) |

Under `<baseline_dir>/`:

| Output | When |
|--------|------|
| `baseline_plv.npy` | Baseline step with `--is-baseline` |

If `T < 2`, the script writes **`meta.json`** only and skips metric computation (see `run_metrics_for_step`).

---

## Controller and config touchpoints

- **Invocation:** `run_files_controlled.py` → `run_offline_step_metrics()` → `python3 offline_step_metrics.py …`
- **Project root:** `OFFLINE_PROJECT_ROOT` env or `streaming.projectRoot` in `CAPTURE_CONFIG`
- **Output root:** `OFFLINE_OUTPUT_ROOT` or capture `outputDir` from JSON

---

## Evidence vs inference

| Claim | Evidence |
|-------|----------|
| Which modules `offline_step_metrics` imports | `offline_step_metrics.py` source |
| `stability_validator` top-level imports | `stability_validator.py` lines 1–5 |
| `compute_plv_features` only uses `plv_helper` | `stability_validator.py` `compute_plv_features` body |
| `run_all_features_streaming` uses `streaming_metrics` | `stability_validator.py` `run_all_features_streaming` |
| Which `coherence_temp_spec_stability` files exist | Directory listing |

| Inference |
|-----------|
| Exact numeric contents of `.npz` / JSON keys — follow `streaming_metrics.save_streaming_results` and `PLVStability.evaluate_run` implementations |

---

## See also

- [`RUN_FILES_CONTROLLED_FLOW.md`](RUN_FILES_CONTROLLED_FLOW.md) — when offline runs relative to capture
- [`ACTIVE_PIPELINE_FILE_MAP.md`](ACTIVE_PIPELINE_FILE_MAP.md) — host files read/moved
