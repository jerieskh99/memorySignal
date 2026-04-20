#!/usr/bin/env bash
# QEMU/macOS consumer: process snapshot pairs (run delta), accumulate frames, run streaming
# metrics when enough frames, then delete snapshot files. Runs alongside the producer.
# Same logic as Linux version; no virsh dependencies.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$ROOT/config_qemu_mac.json}"

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

# Raw retention: optional keep N raw dumps, build raw matrix, run stability metrics (env overrides)
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

# In-memory accumulation is in a temp file (matrix shape: num_pages x num_frames, stored as .npy)
# We append one column (one frame) per job; streaming runs on (time, pages) = (num_frames, num_pages)
RUN_MATRIX="${RUN_MATRIX:-$qPath/run_matrix.npy}"
RUN_MATRIX_LOCK="${RUN_MATRIX}.lock"

echo "[CONSUMER] Consumer started (streaming=${streamingEnabled}, minFrames=${minFramesForStreaming}, rawRetention=${rawRetentionEnabled})"
echo "[CONSUMER] Queue dir: $qPath"
echo "[CONSUMER] Rust program: $rustDeltaCalculationProgram"

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
      # date +%Y%m%d%H%M%S is only second-resolution; include frame count + PID + RANDOM
      # so concurrent or back-to-back runs never clobber the same path.
      local outPrefix="$streamingOutputDir/streaming_f${numFrames}_$(date +%Y%m%d%H%M%S)_$$_${RANDOM}"
      echo "[CONSUMER] Running streaming metrics (frames=$numFrames) -> $outPrefix"
      run_streaming_metrics "$RUN_MATRIX" "$outPrefix" || echo "[CONSUMER] Streaming metrics failed (non-fatal)"
    fi
  fi

  # Always delete prev (keeps rolling prev/curr chain correct)
  if [[ -f "$prev" ]]; then
    rm -f "$prev"
    echo "[CONSUMER] Deleted snapshot: $prev"
  fi

  # Raw retention: keep only curr in rawDir (newest N); else delete curr
  if [[ "$rawRetentionEnabled" == "true" && -n "$rawDir" ]]; then
    mkdir -p "$rawDir"
    dest="$rawDir/$(basename "$curr")"
    if mv "$curr" "$dest" 2>/dev/null; then
      echo "[CONSUMER] Retained snapshot -> $dest"
    else
      if cp "$curr" "$dest" 2>/dev/null; then
        rm -f "$curr"
        echo "[CONSUMER] Copied snapshot -> $dest (then removed original)"
      else
        rm -f "$curr"
        echo "[CONSUMER] Failed to move/copy curr, deleted: $curr"
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
          rm -f "$rawDir/$f"
          echo "[CONSUMER] Pruned old raw dump: $rawDir/$f"
        fi
      done
    fi
  else
    if [[ -f "$curr" ]]; then
      rm -f "$curr"
      echo "[CONSUMER] Deleted snapshot: $curr"
    fi
  fi

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
