#!/usr/bin/env bash
# QEMU consumer for memory snapshots.
# High-level responsibilities:
# - Treat each producer snapshot pair as an opaque byte blob (ELF or RAW, does not matter).
# - For every job { prev, curr, output }:
#     - Run the Rust delta binary to compute per-page cosine / hamming distances.
#     - Append the resulting 1D frame vector as a new *column* in a long-lived run_matrix.npy
#       with shape [num_pages, num_frames].
#     - Optionally run streaming metrics (e.g. PLV/MSC/Cepstrum) on the accumulated matrix.
#     - Optionally maintain a rolling window of RAW dumps and run stability metrics on a
#       derived "raw feature" matrix (e.g. mean_byte/var_byte per page).
#     - Clean up prev/curr snapshot files according to configured retention policy.
# - Drive the queue state machine: pending -> processing -> done/failed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$ROOT/config_qemu.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[CONSUMER] config not found: $CONFIG"
  exit 1
fi

rustDeltaCalculationProgram=$(jq -r '.rustDeltaCalculationProgram' "$CONFIG")
qPath=$(jq -r '.queueDir' "$CONFIG")
streamingEnabled=$(jq -r '.streaming.enabled // false' "$CONFIG")
streamingPython=$(jq -r '.streaming.python // "python3"' "$CONFIG")
streamingModule=$(jq -r '.streaming.streamingMetricsModule // "coherence_temp_spec_stability.streaming_metrics"' "$CONFIG")
streamingOutputDir=$(jq -r '.streaming.streamingOutputDir // ""' "$CONFIG")
minFramesForStreaming=$(jq -r '.streaming.minFramesForStreaming // 128' "$CONFIG")
deltaMetric=$(jq -r '.streaming.deltaMetric // "cosine"' "$CONFIG")
projectRoot=$(jq -r '.streaming.projectRoot // ""' "$CONFIG")

# Raw retention / raw-matrix path:
# - Optionally keep a rolling window of the newest N *curr* dumps (RAW or ELF) on disk.
# - Periodically build a 2D raw feature matrix from those dumps (pages x frames) using
#   a separate Python builder (coherence_temp_spec_stability.raw_matrix_builder).
# - Optionally run the same stability validator on this raw matrix to compare RAW vs DELTA.
rawRetentionEnabled=$(jq -r '.rawRetention.enabled // false' "$CONFIG" 2>/dev/null)
[[ "${RAW_RETENTION:-}" == "1" || "${RAW_RETENTION:-}" == "true" ]] && rawRetentionEnabled="true"
keepDumps=$(jq -r '.rawRetention.keepDumps // 50' "$CONFIG" 2>/dev/null)
[[ -n "${RAW_KEEP_DUMPS:-}" ]] && keepDumps="$RAW_KEEP_DUMPS"
rawDir=$(jq -r '.rawRetention.rawDir // ""' "$CONFIG" 2>/dev/null)
rawMatrixNpy=$(jq -r '.rawRetention.rawMatrixNpy // ""' "$CONFIG" 2>/dev/null)
rawBuildEnabled=$(jq -r '.rawRetention.rawBuild.enabled // true' "$CONFIG" 2>/dev/null)
builderProgram=$(jq -r '.rawRetention.rawBuild.builderProgram // "python3 -m coherence_temp_spec_stability.raw_matrix_builder"' "$CONFIG" 2>/dev/null)
rawBuildMode=$(jq -r '.rawRetention.rawBuild.mode // "mean_byte"' "$CONFIG" 2>/dev/null)
rawBuildPageSize=$(jq -r '.rawRetention.rawBuild.pageSize // 4096' "$CONFIG" 2>/dev/null)
rawBuildMaxBytes=$(jq -r '.rawRetention.rawBuild.maxBytes // 0' "$CONFIG" 2>/dev/null)
rawMetricsEnabled=$(jq -r '.rawRetention.rawMetrics.enabled // true' "$CONFIG" 2>/dev/null)
rawMetricsProjectRoot=$(jq -r '.rawRetention.rawMetrics.projectRoot // ""' "$CONFIG" 2>/dev/null)
validatorModule=$(jq -r '.rawRetention.rawMetrics.validatorModule // "coherence_temp_spec_stability.stability_validator"' "$CONFIG" 2>/dev/null)
rawMetricsWindowSize=$(jq -r '.rawRetention.rawMetrics.windowSize // 128' "$CONFIG" 2>/dev/null)
rawMetricsStepSize=$(jq -r '.rawRetention.rawMetrics.stepSize // 64' "$CONFIG" 2>/dev/null)
rawMetricsOutputDir=$(jq -r '.rawRetention.rawMetrics.outputDir // ""' "$CONFIG" 2>/dev/null)
RAW_MATRIX_LOCK="${rawMatrixNpy}.lock"

qPending="$qPath/pending"
qProcessing="$qPath/processing"
qDone="$qPath/done"
qFailed="$qPath/failed"
mkdir -p "$qPending" "$qProcessing" "$qDone" "$qFailed"

# In-memory accumulation is represented on disk as a single .npy file:
#   RUN_MATRIX: shape [num_pages, num_frames], i.e. "pages x time".
# - Each processed job adds exactly one new column (one time step) to this matrix.
# - Streaming metrics treat the transposed view (time x pages) internally.
RUN_MATRIX="${RUN_MATRIX:-$qPath/run_matrix.npy}"
RUN_MATRIX_LOCK="${RUN_MATRIX}.lock"

echo "[CONSUMER] Consumer started (streaming=${streamingEnabled}, minFrames=${minFramesForStreaming}, rawRetention=${rawRetentionEnabled})"
echo "[CONSUMER] Queue dir: $qPath"
echo "[CONSUMER] Rust program: $rustDeltaCalculationProgram"

# Delete helper that prefers sudo (needed for /var/lib/libvirt/qemu/dump) and logs truthfully.
delete_file() {
  local path="$1"
  local context="${2:-snapshot}"
  if [[ ! -e "$path" ]]; then
    return 0
  fi
  sudo rm -f "$path" 2>/dev/null || rm -f "$path" 2>/dev/null || true
  if [[ ! -e "$path" ]]; then
    echo "[CONSUMER] Deleted $context: $path"
  else
    echo "[CONSUMER] WARNING: could not delete $context (permission?): $path"
  fi
}

# Append one frame (column vector num_pages) to the run matrix. Matrix on disk is (num_pages, num_frames).
append_frame() {
  local frameFile="$1"
  (
    flock 200
    if [[ -f "$RUN_MATRIX" ]]; then
      python3 -c "
import sys
import numpy as np
mat_path = sys.argv[1]
frame_path = sys.argv[2]
out_path = sys.argv[3]
mat = np.load(mat_path)
frame = np.loadtxt(frame_path, dtype=np.float64)
if frame.ndim == 1:
  frame = frame.reshape(-1, 1)
else:
  frame = frame.reshape(-1, 1)
if mat.shape[0] != frame.shape[0]:
  raise ValueError('Frame length %s != matrix rows %s' % (frame.shape[0], mat.shape[0]))
new_mat = np.hstack([mat, frame])
np.save(out_path, new_mat)
" "$RUN_MATRIX" "$frameFile" "$RUN_MATRIX"
    else
      python3 -c "
import sys
import numpy as np
frame_path = sys.argv[1]
out_path = sys.argv[2]
frame = np.loadtxt(frame_path, dtype=np.float64)
if frame.ndim == 1:
  frame = frame.reshape(-1, 1)
else:
  frame = frame.reshape(-1, 1)
np.save(out_path, frame)
" "$frameFile" "$RUN_MATRIX"
    fi
  ) 200>"$RUN_MATRIX_LOCK"
}

# Run streaming metrics on current run matrix and write results.
run_streaming_metrics() {
  local runNpy="$1"
  local outPrefix="$2"
  # Use absolute path so it works when we cd to projectRoot
  if [[ "$runNpy" != /* ]]; then
    runNpy="$ROOT/$runNpy"
  fi
  # streaming_metrics loads .npy and does .T -> expects (pages, time) on disk -> (time, pages) in memory
  # Our matrix is (num_pages, num_frames) so it's already (pages, time). Good.
  if [[ -z "$projectRoot" ]]; then
    "$streamingPython" -m "$streamingModule" --input "$runNpy" --output "$outPrefix" 2>/dev/null || true
  else
    (cd "$projectRoot" && "$streamingPython" -m "$streamingModule" --input "$runNpy" --output "$outPrefix") 2>/dev/null || true
  fi
}

process_job() {
  local jobPath="$1"
  local jobName
  jobName=$(basename "$jobPath")
  local prev curr output
  prev=$(jq -r '.prev' "$jobPath")
  curr=$(jq -r '.curr' "$jobPath")
  output=$(jq -r '.output' "$jobPath")

  echo "[CONSUMER] Running delta: prev=$(basename "$prev") curr=$(basename "$curr")"

  if ! "$rustDeltaCalculationProgram" "$prev" "$curr" "$output"; then
    echo "[CONSUMER] ERROR: Rust delta failed (rc=$?)"
    mv "$jobPath" "$qFailed/"
    return 1
  fi

  # Find the delta output we just wrote (cosine or hamming, one value per line = one frame)
  local subdir="cosine"
  [[ "$deltaMetric" == "hamming" ]] && subdir="hamming"
  local latestFrame
  if [[ -d "$output/$subdir" ]]; then
    latestFrame=$(ls -t "$output/$subdir"/*.txt 2>/dev/null | head -1)
  fi
  if [[ -z "$latestFrame" || ! -f "$latestFrame" ]]; then
    echo "[CONSUMER] WARNING: No delta output file found under $output/$subdir"
  else
    append_frame "$latestFrame"
    echo "[CONSUMER] Appended frame from $latestFrame"
  fi

  # Run streaming when we have enough frames
  if [[ "$streamingEnabled" == "true" && -f "$RUN_MATRIX" && -n "$streamingOutputDir" ]]; then
    local numFrames
    numFrames=$(python3 -c "import numpy as np; m=np.load('$RUN_MATRIX'); print(m.shape[1])" 2>/dev/null || echo "0")
    if [[ -n "$numFrames" && "$numFrames" -ge "$minFramesForStreaming" ]]; then
      mkdir -p "$streamingOutputDir"
      local outPrefix="$streamingOutputDir/streaming_$(date +%Y%m%d%H%M%S)"
      echo "[CONSUMER] Running streaming metrics (frames=$numFrames) -> $outPrefix"
      run_streaming_metrics "$RUN_MATRIX" "$outPrefix" || echo "[CONSUMER] Streaming metrics failed (non-fatal)"
    fi
  fi

  # Delete only prev. curr becomes the next job's prev and is deleted when that job runs.
  if [[ -f "$prev" ]]; then
    delete_file "$prev" "snapshot"
  fi

  # Raw retention: optionally move curr into rawDir (newest N); otherwise leave curr on disk for next job.
  if [[ "$rawRetentionEnabled" == "true" && -n "$rawDir" ]]; then
    mkdir -p "$rawDir"
    dest="$rawDir/$(basename "$curr")"
    if mv "$curr" "$dest" 2>/dev/null; then
      echo "[CONSUMER] Retained snapshot -> $dest"
    else
      if cp "$curr" "$dest" 2>/dev/null; then
        delete_file "$curr" "snapshot"
        echo "[CONSUMER] Copied snapshot -> $dest (then removed original)"
      else
        delete_file "$curr" "snapshot"
        echo "[CONSUMER] Failed to move/copy curr; attempted cleanup of original."
      fi
    fi
    # Prune rawDir to keep only newest keepDumps (by mtime; portable: ls -t)
    count=$(find "$rawDir" -maxdepth 1 -type f 2>/dev/null | wc -l)
    if [[ "$count" -gt "$keepDumps" ]]; then
      idx=0
      for f in $(ls -t "$rawDir" 2>/dev/null); do
        [[ -f "$rawDir/$f" ]] || continue
        idx=$((idx + 1))
        if [[ $idx -gt $keepDumps ]]; then
          delete_file "$rawDir/$f" "old raw dump"
        fi
      done
    fi
  fi
  # When rawRetention is false: do not delete curr; it is the next job's prev and will be deleted then.

  # Raw build + raw metrics (when raw retention enabled and enough dumps)
  if [[ "$rawRetentionEnabled" == "true" && -n "$rawDir" && -n "$rawMatrixNpy" ]]; then
    rawCount=$(find "$rawDir" -maxdepth 1 -type f 2>/dev/null | wc -l)
    if [[ "$rawCount" -ge "$keepDumps" && "$rawBuildEnabled" == "true" ]]; then
      (
        exec 200>"$RAW_MATRIX_LOCK"
        flock 200 || exit 0
        echo "[CONSUMER] Building raw matrix from $rawDir (keep=$keepDumps) -> $rawMatrixNpy"
        if eval "$builderProgram --input-dir \"$rawDir\" --keep \"$keepDumps\" --mode \"$rawBuildMode\" --page-size \"$rawBuildPageSize\" --output \"$rawMatrixNpy\" --max-bytes \"$rawBuildMaxBytes\""; then
          if [[ "$rawMetricsEnabled" == "true" && -n "$rawMetricsProjectRoot" && -n "$rawMetricsOutputDir" ]]; then
            mkdir -p "$rawMetricsOutputDir"
            absMatrix="$rawMatrixNpy"
            [[ "$absMatrix" != /* ]] && absMatrix="$ROOT/$absMatrix"
            echo "[CONSUMER] Running raw stability validator -> $rawMetricsOutputDir"
            if (cd "$rawMetricsProjectRoot" && python3 -m "$validatorModule" "$absMatrix" --window-size "$rawMetricsWindowSize" --step-size "$rawMetricsStepSize" --output-dir "$rawMetricsOutputDir" --prefix raw) 2>/dev/null; then
              echo "[CONSUMER] Raw metrics done."
            else
              echo "[CONSUMER] WARNING: Raw validator failed (module missing or error); job still done."
            fi
          fi
        else
          echo "[CONSUMER] WARNING: Raw matrix builder failed; retained dumps unchanged."
        fi
      ) || true
    fi
  fi

  mv "$jobPath" "$qDone/"
  echo "[CONSUMER] Job done -> done: $jobName"
  return 0
}

while true; do
  jobFile=$(find "$qPending" -maxdepth 1 -name '*.json' -print 2>/dev/null | sort | head -1)
  if [[ -z "$jobFile" ]]; then
    sleep 0.1
    continue
  fi

  jobName=$(basename "$jobFile")
  jobProcessing="$qProcessing/$jobName"
  mv "$jobFile" "$jobProcessing"
  echo "[CONSUMER] Picked job: $jobName"

  if process_job "$jobProcessing"; then
    : # moved to done
  else
    : # moved to failed
  fi
done
