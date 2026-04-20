# Issue 01 - Real Run Matrix vs Combined Complex Path

## What We Found

In the active controlled-QEMU pipeline, the step run matrix (`run_matrix_<test>.npy`) is built from one metric stream at a time (configured by `streaming.deltaMetric`, currently `cosine`).

This means the active offline/streaming metrics in `VM_sampler/VM_Capture_QEMU` are currently operating on a real-valued per-page time series (cosine-only in current config), not on the later combined complex representation.

## The Two Data Paths

1. Active controlled-QEMU metrics path:
   - Producer/consumer generate delta text frames in `outputDir/cosine/` and `outputDir/hamming/`.
   - Consumer appends only one selected stream (currently cosine) into `run_matrix_<test>.npy`.
   - `offline_step_metrics.py` and streaming outputs (`streaming_results/*.npz|*.json`) use that step matrix.

2. Downstream combined-analysis path:
   - Separate feature scripts read both hamming and cosine frame files.
   - They combine them into a complex matrix:
     - `combined_data = hamming * exp(j * 2*pi*cosine)`
   - Wavelet/spectral analyses can run on this combined matrix.

## When This Affects Results

This issue affects interpretation whenever conclusions are drawn from:
- `run_matrix_<test>.npy`
- `offline_step_metrics.py` outputs
- `streaming_results/*.npz` or `streaming_results/*.json`

In those cases (with current config), results reflect cosine-derived dynamics only.

It does not affect analyses that explicitly use the combined matrix produced by the downstream combine step.

## Why This Is Still Good News

Even with cosine-only input in the active metrics path, we observed meaningful clustering/classification behavior.

This is positive evidence that:
- the captured memory-evolution signal is informative,
- the cosine channel alone already contains discriminative structure.

## Conclusions

- The repository currently contains both:
  - a real-valued active metrics path (single selected metric), and
  - a complex-valued combined downstream path.
- Reported results must explicitly state which path was used.
- Current observed separability from cosine-only analysis is a strong baseline result, not a failure.
