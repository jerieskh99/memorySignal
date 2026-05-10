# Running the Offline Analysis

This document covers every flag for the three scripts in the offline analysis pipeline, with examples. For what each output file contains, see `OFFLINE_METRICS_AND_OUTPUTS.md`. For the full capture pipeline flow, see `ACTIVE_PIPELINE_OVERVIEW.md`.

---

## Scripts covered

| Script | When to use |
|---|---|
| `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py` | Run metrics on one already-captured test matrix. Main analysis entry point. |
| `VM_sampler/VM_Capture_QEMU/batch_offline_step_metrics_zst.py` | Batch-process a folder of `.npy.zst` matrices: decompresses one file at a time, runs `offline_step_metrics.py`, deletes the temporary `.npy`, then moves to the next. The first file (sorted by test number) is used as the PLV baseline. Use this when matrices are stored compressed and you want to process a full batch without pre-decompressing everything. |
| `VM_sampler/VM_Capture_QEMU/rebuild_matrices_and_rerun_offline.py` | Reconstruct all matrices from rotated delta frames and rerun metrics on them. Use after a full capture session. |
| `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` | Live pipeline controller. Invokes offline analysis automatically via env vars. |

---

## 1. `offline_step_metrics.py`

Runs MSC, Cepstrum, and PLV metrics on a single test matrix. Optionally splits the trace into contiguous temporal segments and runs the same pipeline inside each segment.

### Full flag reference

```
python3 offline_step_metrics.py \
  --matrix            PATH   (required) Path to run_matrix_<test>.npy
  --step-name         NAME   (required) Label for the output folder, e.g. test6_mem_alloc_touch_pages
  --output-root       PATH   (required) Root directory; outputs go to <output-root>/offline/<step-name>/
  --project-root      PATH   (required) Repo root; coherence_temp_spec_stability/ is loaded from here
  --baseline-dir      PATH   (required) Directory containing baseline_plv.npy (or where it will be written)
  --is-baseline              (flag)     Mark this step as the idle baseline; computes and saves baseline_plv.npy
  --window-size       INT    (default 128) Window length in frames for MSC/Cepstrum/PLV
  --step-size         INT    (default 64)  Hop size in frames between windows
  --segments          INT    (default 1)   Number of contiguous temporal segments; 1 means no segmentation
  --min-windows-per-segment INT (default 50) Minimum windows per segment before emitting a warning
```

**Flag details**

`--matrix`
Path to a `.npy` file with shape `(pages, frames)` as written by the capture consumer. The script transposes it internally to `(frames, pages)` before any computation.

`--step-name`
String label. Used as the output subfolder name under `<output-root>/offline/`. Should match the test folder name convention, e.g. `test1_run_idle`, `test6_mem_alloc_touch_pages`.

`--output-root`
Parent directory for all offline outputs. The script creates `<output-root>/offline/<step-name>/` automatically.

`--project-root`
Absolute path to the `memorySignal` repo root. The script prepends `<project-root>/coherence_temp_spec_stability` and `<project-root>` to `sys.path` so metric modules import correctly.

`--baseline-dir`
Directory for the shared `baseline_plv.npy` file. All non-baseline steps load this file; the baseline step writes it. Must be the same directory for all steps in one dataset.

`--is-baseline`
Pass this flag only for the first clean idle step. Computes the per-page PLV baseline from that step's data and saves it to `<baseline-dir>/baseline_plv.npy`. Omit for all other steps.

`--window-size`
Length of each sliding analysis window in frames. Default 128. Shared by MSC, Cepstrum, and PLV. Changing this value changes what the metrics measure; do not mix values within one dataset.

`--step-size`
Hop between consecutive windows in frames. Default 64 (50% overlap). Lower values increase window density and correlation; higher values reduce correlation but also reduce the number of windows per segment.

`--segments`
Number of contiguous temporal segments to split the trace into before metric extraction. Default 1 (no segmentation; current full-run behavior is preserved exactly). When S > 1, the trace `[0, T)` is partitioned into S equal-length slices; the last slice absorbs any remainder. The existing metric pipeline runs inside each slice independently. Outputs are written under `<step-name>/segments/seg_NN/` and a `segment_meta.json` summarizes all segments. **Segments are temporal sub-observations of one run, not independent samples. Do not pool them to inflate n.**

`--min-windows-per-segment`
Minimum number of analysis windows required per segment for metric estimates to be reliable. Default 50, which corresponds to roughly 3 300 frames at window=128/hop=64. Segments below this floor still run but emit a warning and are flagged in `segment_meta.json`. Adjust only for sensitivity studies; the default reflects the methodology floor in `segment_level_analysis_critique_and_plan.md` Section 5.2.

### Output structure

With `--segments 1` (default):
```
<output-root>/offline/<step-name>/
    meta.json
    streaming.npz
    streaming.json
    plv_baseline_aware.json
```

With `--segments S` where S > 1 (additive; run-level files unchanged):
```
<output-root>/offline/<step-name>/
    meta.json
    streaming.npz
    streaming.json
    plv_baseline_aware.json
    segments/
        segment_meta.json
        seg_00/
            streaming.npz
            streaming.json
            plv_baseline_aware.json
        seg_01/ ... seg_{S-1}/
```

`segment_meta.json` contains: segment frame boundaries, expected window counts, per-segment metric summaries, within-run trajectory metrics (CV, std, range, first-vs-last), per-segment deltas from the run-level value, and a mandatory `limits_of_inference` block.

### Examples

**Baseline step (idle):**
```bash
python3 VM_sampler/VM_Capture_QEMU/offline_step_metrics.py \
  --matrix /data/output/run_matrix_test1_run_idle.npy \
  --step-name test1_run_idle \
  --output-root /data/output \
  --project-root /home/user/memorySignal \
  --baseline-dir /data/output/offline/baseline \
  --is-baseline
```

**Non-baseline step (full-run only):**
```bash
python3 VM_sampler/VM_Capture_QEMU/offline_step_metrics.py \
  --matrix /data/output/run_matrix_test6_mem_alloc_touch_pages.npy \
  --step-name test6_mem_alloc_touch_pages \
  --output-root /data/output \
  --project-root /home/user/memorySignal \
  --baseline-dir /data/output/offline/baseline
```

**Same step with 4 temporal segments (stability diagnostic):**
```bash
python3 VM_sampler/VM_Capture_QEMU/offline_step_metrics.py \
  --matrix /data/output/run_matrix_test6_mem_alloc_touch_pages.npy \
  --step-name test6_mem_alloc_touch_pages \
  --output-root /data/output \
  --project-root /home/user/memorySignal \
  --baseline-dir /data/output/offline/baseline \
  --segments 4
```

**Segment sensitivity grid (run three times, each into its own output root):**
```bash
for S in 2 4 8; do
  python3 VM_sampler/VM_Capture_QEMU/offline_step_metrics.py \
    --matrix /data/output/run_matrix_test6_mem_alloc_touch_pages.npy \
    --step-name test6_mem_alloc_touch_pages \
    --output-root /data/output_seg${S} \
    --project-root /home/user/memorySignal \
    --baseline-dir /data/output/offline/baseline \
    --segments ${S}
done
```

---

## 2. `batch_offline_step_metrics_zst.py`

A batch wrapper that processes a folder of `.npy.zst` compressed matrices without requiring pre-decompression of all files. Each file is decompressed to a temporary `.npy`, analyzed by `offline_step_metrics.py`, and the temporary file is deleted before the next file starts. Disk usage at any moment is limited to one uncompressed matrix. The original `.npy.zst` files are never modified.

This script is a convenience automation layer. It does not change any analysis logic; it only automates the decompress-run-delete cycle and baseline assignment across a batch.

### Full flag reference

```
python3 batch_offline_step_metrics_zst.py \
  --matrix-folder           PATH   (required) Folder containing .npy.zst matrix files
  --folder-name             NAME   (required) Label for this batch run (printed in progress output)
  --output-root             PATH   (required) Root for offline outputs; passed to offline_step_metrics.py
  --project-root            PATH   (required) Repo root; coherence_temp_spec_stability/ is loaded from here
  --baseline-dir            PATH   (required) Shared directory for baseline_plv.npy across all files in the batch
  --window-size             INT    (default 128) Window length in frames; forwarded to offline_step_metrics.py
  --step-size               INT    (default 64)  Hop size in frames; forwarded to offline_step_metrics.py
  --segments                INT    (default 1)   Number of temporal segments; forwarded to offline_step_metrics.py
  --min-windows-per-segment INT    (default 50)  Min windows per segment warning threshold; forwarded
  --offline-script          PATH   (default: same dir as this script) Path to offline_step_metrics.py
```

**Flag details**

`--matrix-folder`
Directory containing the `.npy.zst` compressed matrix files to process. All files matching `*.npy.zst` in this directory are included.

`--folder-name`
Human-readable label for this batch run. Printed in progress output and the final summary. Does not affect output paths.

`--output-root`, `--project-root`, `--baseline-dir`
Forwarded unchanged to `offline_step_metrics.py`. See Section 1 for their exact semantics. The same `--baseline-dir` is used for every file in the batch; the first file writes `baseline_plv.npy` there and all subsequent files read it.

`--window-size`, `--step-size`
Forwarded unchanged to `offline_step_metrics.py`. Default 128 and 64. Do not mix values within one dataset.

`--segments`
Forwarded to `offline_step_metrics.py --segments`. Default 1 (no segmentation). Set to 2, 4, or 8 to enable the temporal segment pass for every file in the batch. Segments are temporal sub-observations, not independent samples. See Section 5 for interpretation rules.

`--min-windows-per-segment`
Forwarded to `offline_step_metrics.py`. Only relevant when `--segments > 1`.

`--offline-script`
Path to `offline_step_metrics.py`. Defaults to the file in the same directory as this script.

### Sorting and baseline behavior

Files are sorted **numerically** by the test number extracted from the filename (`test<N>`). Files without a `test<N>` pattern sort after all numbered files, lexicographically.

The **first file in sorted order** is passed to `offline_step_metrics.py` with `--is-baseline`. It computes and writes `baseline_plv.npy` to `--baseline-dir`. All subsequent files load that baseline.

Example sort order for:
```
run_matrix_test4_mem_pointer_chase.npy.zst
run_matrix_test1_run_idle.npy.zst
run_matrix_test18_mem_alloc_touch_pages.npy.zst
```
Sorted result: `test1` (baseline), `test4`, `test18`.

### Step-name derivation

The `--step-name` passed to `offline_step_metrics.py` is the filename with `.npy.zst` stripped:

```
run_matrix_test18_mem_alloc_touch_pages.npy.zst
  -> run_matrix_test18_mem_alloc_touch_pages
```

Output goes to `<output-root>/offline/run_matrix_test18_mem_alloc_touch_pages/`.

### Decompression behavior

- Uses the system `zstd` binary (`zstd -d`). Must be on `PATH`. Fails immediately at startup with a clear message if not found.
- Decompresses to a temporary `.npy` in the same directory as the `.npy.zst` file.
- The temporary `.npy` is always deleted after processing, even if `offline_step_metrics.py` fails.
- If decompression fails for a file, that file is skipped and the batch continues.
- If `offline_step_metrics.py` fails for a file, the temporary `.npy` is still deleted, the failure is recorded, and the batch continues.

### Output behavior

Progress is printed per file. The final summary reports:

```
Total files    : N
Succeeded      : N
Failed decomp  : N
Failed analysis: N
```

The script returns exit code 0 only if every file succeeded. Returns 1 if any file failed decompression or analysis.

### Examples

**Example 1: Batch full-run analysis (default `--segments 1`)**
```bash
python3 VM_sampler/VM_Capture_QEMU/batch_offline_step_metrics_zst.py \
  --matrix-folder /data/matrices_zst/cycle1 \
  --folder-name cycle1 \
  --output-root /data/output \
  --project-root /home/user/memorySignal \
  --baseline-dir /data/output/offline/baseline
```

**Example 2: Batch with 4 temporal segments per test**
```bash
python3 VM_sampler/VM_Capture_QEMU/batch_offline_step_metrics_zst.py \
  --matrix-folder /data/matrices_zst/cycle1 \
  --folder-name cycle1_seg4 \
  --output-root /data/output_seg4 \
  --project-root /home/user/memorySignal \
  --baseline-dir /data/output/offline/baseline \
  --segments 4
```

Segment outputs appear under each test's `segments/` subfolder alongside the unchanged run-level outputs. Segments are temporal sub-observations of one recording and must not be pooled across tests to inflate n.

---

## 3. `rebuild_matrices_and_rerun_offline.py`
<!-- formerly Section 2 -->

Reconstructs all per-test matrices from rotated delta `.txt` files and reruns the offline metrics on each one. Use this to reproduce or rerun analysis after a full capture session without re-running the capture.

### Full flag reference

```
python3 rebuild_matrices_and_rerun_offline.py \
  --rotated-root      PATH   (required) Path to <output-dir>/rotated (contains test*/hamming and test*/cosine folders)
  --matrix-dir        PATH   (required) Directory where rebuilt run_matrix_<test>.npy files are written
  --output-root       PATH   (required) Root for offline outputs; passed to offline_step_metrics.py
  --project-root      PATH   (required) Repo root for coherence imports
  --representation    STR    (default complex) Matrix format: complex | cosine | hamming
  --phase-scale       FLOAT  (default 2.0) Scale in exp(j * phase_scale * pi * cosine) for complex mode
  --baseline-step-number INT (default 1) Step number to treat as idle baseline
  --baseline-test-name STR   (optional) Exact test folder name for baseline; overrides --baseline-step-number
  --tests             LIST   (optional) Explicit test folder names to process; default: all discovered
  --sort              STR    (default name) Frame ordering within each test: name | mtime
  --offline-script    PATH   (default: same dir as this script) Path to offline_step_metrics.py
  --python-bin        STR    (default python3) Python executable to use
  --window-size       INT    (default 128) Forwarded to offline_step_metrics.py
  --step-size         INT    (default 64)  Forwarded to offline_step_metrics.py
  --segments          INT    (default 1)   Forwarded to offline_step_metrics.py; enables segment pass when > 1
  --continue-on-error        (flag) Continue processing remaining tests if one fails
  --delete-matrix-after-success (flag) Delete rebuilt .npy after metrics succeed to save disk space
```

**Key flag details**

`--rotated-root`
Directory containing per-test subfolders in the format `test<N>_<subtype>/hamming/*.txt` and `test<N>_<subtype>/cosine/*.txt`. This is the `rotated/` subfolder inside the capture `outputDir`.

`--representation`
`complex` (default) uses both hamming and cosine files to form complex delta values: `D = hamming * exp(j * phase_scale * pi * cosine)`. `hamming` or `cosine` uses only one metric. Use `complex` for all analyses that need PLV (which requires phase information).

`--baseline-step-number`
Step number (1-based, matching the `testN` prefix) that is marked as the idle baseline. Ignored if `--baseline-test-name` is set explicitly.

`--segments`
Forwarded unchanged to `offline_step_metrics.py --segments`. Same semantics as described above. Default 1 disables segmentation.

`--continue-on-error`
Useful for partial re-runs. If one test's matrix rebuild or metric run fails, the script logs a warning and continues rather than aborting.

### Examples

**Rerun all tests (full-run only):**
```bash
python3 VM_sampler/VM_Capture_QEMU/rebuild_matrices_and_rerun_offline.py \
  --rotated-root /data/output/rotated \
  --matrix-dir /data/matrices \
  --output-root /data/output \
  --project-root /home/user/memorySignal
```

**Rerun with 4-segment analysis:**
```bash
python3 VM_sampler/VM_Capture_QEMU/rebuild_matrices_and_rerun_offline.py \
  --rotated-root /data/output/rotated \
  --matrix-dir /data/matrices \
  --output-root /data/output \
  --project-root /home/user/memorySignal \
  --segments 4
```

**Rerun only specific tests and delete matrices after to save space:**
```bash
python3 VM_sampler/VM_Capture_QEMU/rebuild_matrices_and_rerun_offline.py \
  --rotated-root /data/output/rotated \
  --matrix-dir /data/matrices \
  --output-root /data/output \
  --project-root /home/user/memorySignal \
  --tests test2_mem_stream test4_mem_pointer_chase \
  --baseline-test-name test1_run_idle \
  --segments 4 \
  --delete-matrix-after-success \
  --continue-on-error
```

---

## 4. `run_files_controlled.py` (live pipeline)

The live controller invokes `offline_step_metrics.py` automatically after each step's capture queue drains. Configure via environment variables before launching.

### Relevant env vars for offline analysis

```
OFFLINE_METRICS_MODE=1          Enable offline metrics (default: disabled)
OFFLINE_METRICS_SCRIPT=PATH     Path to offline_step_metrics.py (default: same dir as controller)
OFFLINE_PROJECT_ROOT=PATH       Repo root for coherence imports (required when OFFLINE_METRICS_MODE=1)
OFFLINE_BASELINE_DIR=PATH       Baseline dir (default: <outputDir>/offline/baseline)
OFFLINE_OUTPUT_ROOT=PATH        Output root override (default: capture outputDir)
OFFLINE_WINDOW_SIZE=128         Window length forwarded to offline_step_metrics.py
OFFLINE_STEP_SIZE=64            Hop size forwarded to offline_step_metrics.py
OFFLINE_SEGMENTS=1              Number of segments forwarded to offline_step_metrics.py (default: 1)
BASELINE_STEP_NUMBER=1          Which step number is the idle baseline
```

**`OFFLINE_SEGMENTS`**
Set to 2, 4, or 8 to enable the segment pass for every test in the live run. The segment outputs appear alongside the run-level outputs automatically. When unset or set to 1, the live run behavior is identical to the pre-segment implementation.

### Example

```bash
OFFLINE_METRICS_MODE=1 \
OFFLINE_PROJECT_ROOT=/home/user/memorySignal \
OFFLINE_SEGMENTS=4 \
python3 VM_sampler/VM_Capture_QEMU/run_files_controlled.py
```

---

## Choosing `--segments`

`k` is a research-question selector, not a hyperparameter. Different values address different timescales:

| S | Scientific question |
|---|---|
| 1 | Is the run distinctive overall (current analysis, default) |
| 2 | Is the run internally consistent (split-half reliability) |
| 4 | Quartile structure: warmup / early steady / late steady / cooldown |
| 8 | Fine phase structure where run duration permits |

Run the script three times into separate `--output-root` directories if you want a full sensitivity grid. Do not average or pool results across S values.

**Minimum trace length for each S** (at window=128, hop=64, min 50 windows per segment):

| S | Min frames needed | At 1 frame/snapshot |
|---|---|---|
| 2 | 6 656 | 6 656 snapshots |
| 4 | 13 312 | 13 312 snapshots |
| 8 | 26 624 | 26 624 snapshots |

If a segment falls below the 50-window floor, the script emits a warning and tags it in `segment_meta.json` under `diagnostics.any_segment_too_short`. The segment is still attempted.

---

## Important notes on segment outputs

- Segments are **contiguous temporal sub-observations** of one recording. They are not independent runs.
- Never pool segments from different runs to inflate n for Kruskal-Wallis, Mann-Whitney, silhouette, or LDA.
- `segment_meta.json` contains a mandatory `limits_of_inference` block that encodes these constraints for downstream consumers.
- Use segment outputs for: temporal stability diagnostics, phase purity checks, confusion localization, and within-run drift detection.
- The cross-run and cross-subtype analyses that require class centroids belong in a separate downstream aggregation step, not in this per-test script.
- See `segment_level_analysis_critique_and_plan.md` for full methodology rules before interpreting any segment-level result.
