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
apfCalculationProgram=$(jq -r '.apfCalculationProgram // ""' "$CONFIG")
# Substrate metric: the modular per-page feature differ (live_delta_calc_modular), run
# with --speed N --sparse. Used when CAPTURE_METRIC=substrate (see process_job below).
substrateProgram=$(jq -r '.substrateProgram // ""' "$CONFIG")
substrateSpeed=$(jq -r '.substrateSpeed // 2' "$CONFIG")
# Capture metric, inherited from the orchestrator env: "delta" (default;
# Cosine/Hamming via rustDeltaCalculationProgram) or "apf_queue" (run apf_calc and
# append one line per pair to apf_trajectory.jsonl, skipping run_matrix/streaming).
CAPTURE_METRIC="${CAPTURE_METRIC:-delta}"
APF_SEQ=0
qPath=$(jq -r '.queueDir' "$CONFIG")
streamingEnabled=$(jq -r '.streaming.enabled // false' "$CONFIG")
streamingPython=$(jq -r '.streaming.python // "python3"' "$CONFIG")
streamingModule=$(jq -r '.streaming.streamingMetricsModule // "coherence_temp_spec_stability.streaming_metrics"' "$CONFIG")
streamingOutputDir=$(jq -r '.streaming.streamingOutputDir // ""' "$CONFIG")
minFramesForStreaming=$(jq -r '.streaming.minFramesForStreaming // 128' "$CONFIG")
deltaMetric=$(jq -r '.streaming.deltaMetric // "cosine"' "$CONFIG")
projectRoot=$(jq -r '.streaming.projectRoot // ""' "$CONFIG")
BORG="${BORG:-0}"
BORG_REPO="${BORG_REPO:-}"
BORG_PASSPHRASE="${BORG_PASSPHRASE:-}"
# Per-capture identity for structured borg archive names ({workload}__{run}__{dump_ts}).
# The orchestrator sets these per step; defaults keep manual runs working.
BORG_WORKLOAD="${BORG_WORKLOAD:-unknown}"
BORG_RUN_ID="${BORG_RUN_ID:-$(date +%Y%m%d%H%M%S)}"
# --- zstd delta-chain retention (alternative to BORG; store each snapshot as a
# zstd --patch-from diff against the previous one). See retain_zstd_delta below. ---
ZSTD="${ZSTD:-0}"
ZSTD_DIR="${ZSTD_DIR:-}"
ZSTD_WORKLOAD="${ZSTD_WORKLOAD:-unknown}"
ZSTD_RUN_ID="${ZSTD_RUN_ID:-$(date +%Y%m%d%H%M%S)}"
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"
ZSTD_NEXT=0   # next chain sequence this session (0 -> first call also writes the full base)
# When OFFLINE_MODE=1, skip the live streaming trigger entirely.
# Offline metrics are computed per-step by offline_step_metrics.py instead.
OFFLINE_MODE="${OFFLINE_MODE:-0}"

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
RUN_MATRIX_STREAM="${RUN_MATRIX}.frames.bin"
RUN_MATRIX_STREAM_META="${RUN_MATRIX}.frames.meta"
# PID file for the async streaming metrics background process.
# Used to skip triggering a second streaming run while one is already in flight.
STREAMING_PID_FILE="${RUN_MATRIX}.streaming.pid"

echo "[CONSUMER] Consumer started (streaming=${streamingEnabled}, minFrames=${minFramesForStreaming}, rawRetention=${rawRetentionEnabled}, offlineMode=${OFFLINE_MODE})"
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

sanitize_name() {
  local s="$1"
  # borg archive names are restrictive; keep alnum, dot, underscore, dash.
  s="${s//[^A-Za-z0-9._-]/_}"
  # avoid empty archive tokens
  [[ -z "$s" ]] && s="item"
  echo "$s"
}

archive_with_borg() {
  # Synchronously archive $path into the borg repo as {workload}__{run_id}__{dump_ts}.
  # Returns 0 ONLY if borg create succeeds -- the caller deletes the source only then,
  # so a failed archive (NFS down, etc.) never loses a snapshot.
  local path="$1"
  [[ -e "$path" ]] || return 1
  if [[ "$BORG" != "1" && "$BORG" != "true" && "$BORG" != "yes" ]]; then
    return 1
  fi
  if [[ -z "$BORG_REPO" ]]; then
    echo "[CONSUMER] WARNING: BORG enabled but BORG_REPO unset; not archiving $path"
    return 1
  fi
  if ! command -v borg >/dev/null 2>&1; then
    echo "[CONSUMER] WARNING: BORG enabled but 'borg' not found; not archiving $path"
    return 1
  fi

  # Dump timestamp parsed from memory_dump-<ts>.raw; synthesize if name is unexpected.
  local image ts archive
  image=$(basename "$path")
  ts="${image#memory_dump-}"; ts="${ts%.raw}"
  [[ "$ts" == "$image" ]] && ts=$(date +%Y%m%d%H%M%S%3N)
  archive="$(sanitize_name "$BORG_WORKLOAD")__$(sanitize_name "$BORG_RUN_ID")__$(sanitize_name "$ts")"

  echo "[CONSUMER] BORG archiving $image -> ::$archive"
  local rc
  (
    export BORG_REPO
    export BORG_UNKNOWN_UNENCRYPTED_REPO_ACCESS_IS_OK=yes
    [[ -n "$BORG_PASSPHRASE" ]] && export BORG_PASSPHRASE
    borg create "::${archive}" "$path" </dev/null >/dev/null 2>&1
  )
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[CONSUMER] BORG archived ::$archive"
  else
    echo "[CONSUMER] WARNING: borg create failed (rc=$rc) for $path; keeping local copy"
  fi
  return $rc
}

retain_zstd_delta() {
  # Store curr as a zstd delta against prev in a per-workload chain folder. The first call
  # also writes the full base (from prev) as 000000.zst -- the chain root. File numbers
  # track the snapshot sequence (a failed write leaves a numbered gap, never a mislabel).
  # Returns 0 only if every write succeeded; the caller deletes the raw prev only then.
  local prev="$1" curr="$2"
  [[ -e "$prev" && -e "$curr" ]] || return 1
  if [[ "$ZSTD" != "1" && "$ZSTD" != "true" && "$ZSTD" != "yes" ]]; then
    return 1
  fi
  if [[ -z "$ZSTD_DIR" ]]; then
    echo "[CONSUMER] WARNING: ZSTD enabled but ZSTD_DIR unset; not archiving $curr"
    return 1
  fi
  if ! command -v zstd >/dev/null 2>&1; then
    echo "[CONSUMER] WARNING: ZSTD enabled but 'zstd' not found; not archiving $curr"
    return 1
  fi

  local dir
  dir="$ZSTD_DIR/$(sanitize_name "$ZSTD_WORKLOAD")__$(sanitize_name "$ZSTD_RUN_ID")"
  mkdir -p "$dir" 2>/dev/null

  local seq=$ZSTD_NEXT
  ZSTD_NEXT=$((ZSTD_NEXT + 1))
  local rc=0

  # First snapshot of this workload: write the full base from prev (chain root).
  if [[ "$seq" -eq 0 ]]; then
    if zstd -q -"$ZSTD_LEVEL" "$prev" -o "$dir/000000.zst" 2>/dev/null; then
      echo "[CONSUMER] ZSTD base  -> $dir/000000.zst"
    else
      echo "[CONSUMER] WARNING: zstd base write failed for $prev"
      rc=1
    fi
  fi

  # Delta: curr patched against prev (--long=31 window covers <=2 GB reference dumps).
  local n
  printf -v n "%06d" $((seq + 1))
  if zstd -q -"$ZSTD_LEVEL" --long=31 --patch-from="$prev" "$curr" -o "$dir/${n}.zst" 2>/dev/null; then
    echo "[CONSUMER] ZSTD delta -> $dir/${n}.zst"
  else
    echo "[CONSUMER] WARNING: zstd delta write failed for $curr"
    rc=1
  fi
  return $rc
}

# Append one frame (column vector num_pages) to the run matrix. Matrix on disk is (num_pages, num_frames).
append_frame() {
  local frameFile="$1"
  if [[ "$OFFLINE_MODE" == "1" || "$OFFLINE_MODE" == "true" ]]; then
    (
      flock 200
      python3 -c "
import os
import sys
import numpy as np

frame_path = sys.argv[1]
stream_path = sys.argv[2]
meta_path = sys.argv[3]

frame = np.loadtxt(frame_path, dtype=np.float64).reshape(-1)
num_pages = int(frame.shape[0])

if os.path.isfile(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as fh:
        expected = int((fh.read() or '0').strip())
    if expected != num_pages:
        raise ValueError(f'Frame length {num_pages} != expected pages {expected}')
else:
    with open(meta_path, 'w', encoding='utf-8') as fh:
        fh.write(str(num_pages))

with open(stream_path, 'ab') as out:
    frame.tofile(out)
" "$frameFile" "$RUN_MATRIX_STREAM" "$RUN_MATRIX_STREAM_META"
    ) 200>"$RUN_MATRIX_LOCK"
    return
  fi
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

  if [[ "$CAPTURE_METRIC" == "apf_queue" ]]; then
    # APF metric: compute APF with apf_calc, append one trajectory line, then skip
    # the Cosine/Hamming delta + run_matrix + streaming. The shared prev-delete /
    # raw-retention / done-move below run for both metrics.
    echo "[CONSUMER] Running apf_calc: prev=$(basename "$prev") curr=$(basename "$curr")"
    if ! "$apfCalculationProgram" "$prev" "$curr" "$output"; then
      echo "[CONSUMER] ERROR: apf_calc failed (rc=$?)"
      mv "$jobPath" "$qFailed/"
      return 1
    fi
    local apfFile apfVal apfTraj
    apfFile=$(ls -t "$output/apf"/*.txt 2>/dev/null | head -1)
    apfVal=$(cat "$apfFile" 2>/dev/null)
    [[ -z "$apfVal" ]] && apfVal=0
    rm -f "$apfFile" 2>/dev/null || true
    apfTraj="${TIMING_APF_JSONL:-$output/apf_trajectory.jsonl}"
    printf '{"seq":%d,"t_emit_epoch":%s,"prev":"%s","curr":"%s","apf":%s}\n' \
      "$APF_SEQ" "$(date +%s)" "$prev" "$curr" "$apfVal" >> "$apfTraj"
    echo "[CONSUMER] APF seq=$APF_SEQ apf=$apfVal -> $apfTraj"
    APF_SEQ=$((APF_SEQ + 1))
  elif [[ "$CAPTURE_METRIC" == "substrate" ]]; then
    # Substrate metric: compute the sparse per-page feature vectors with the modular
    # differ (--speed N --sparse), then append this snapshot's changed-page rows -- each
    # stamped with the snapshot seq -- to a per-cell trajectory CSV. Skips the
    # Cosine/Hamming delta + run_matrix + streaming; the shared prev-delete runs below.
    echo "[CONSUMER] Running substrate (--speed $substrateSpeed --sparse): prev=$(basename "$prev") curr=$(basename "$curr")"
    if [[ -z "$substrateProgram" ]]; then
      echo "[CONSUMER] ERROR: CAPTURE_METRIC=substrate but .substrateProgram is unset in $CONFIG"
      mv "$jobPath" "$qFailed/"
      return 1
    fi
    if ! "$substrateProgram" --speed "$substrateSpeed" --sparse "$prev" "$curr" "$output"; then
      echo "[CONSUMER] ERROR: substrate differ failed (rc=$?)"
      mv "$jobPath" "$qFailed/"
      return 1
    fi
    local metFile substrateTraj metRows
    metFile=$(ls -t "$output/metrics"/page_metrics-*.csv 2>/dev/null | head -1)
    substrateTraj="${SUBSTRATE_CSV:-$output/substrate_trajectory.csv}"
    if [[ -z "$metFile" || ! -f "$metFile" ]]; then
      echo "[CONSUMER] WARNING: no substrate CSV found under $output/metrics"
    else
      # Write the seq-prefixed header once (the differ header is "page_index,<metrics>").
      if [[ ! -s "$substrateTraj" ]]; then
        printf 'seq,%s\n' "$(head -1 "$metFile")" > "$substrateTraj"
      fi
      # Append this snapshot's changed-page rows, each stamped with the snapshot seq.
      metRows=$(( $(wc -l < "$metFile") - 1 ))
      awk -v seq="$APF_SEQ" 'NR>1{print seq","$0}' "$metFile" >> "$substrateTraj"
      rm -f "$metFile" 2>/dev/null || true
      echo "[CONSUMER] substrate seq=$APF_SEQ rows=$metRows -> $substrateTraj"
    fi
    APF_SEQ=$((APF_SEQ + 1))
  else
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

  # Run streaming when we have enough frames.
  # IMPORTANT: launched in background so the consumer loop is never blocked.
  # A PID file prevents launching a second streaming run while one is in flight.
  # Skipped when OFFLINE_MODE=1 (step-gated mode) because offline_step_metrics.py
  # computes all metrics after the queue drains for each step.
  if [[ "$OFFLINE_MODE" != "1" && "$OFFLINE_MODE" != "true" ]] && \
     [[ "$streamingEnabled" == "true" && -f "$RUN_MATRIX" && -n "$streamingOutputDir" ]]; then
    local numFrames
    numFrames=$(python3 -c "import numpy as np; m=np.load('$RUN_MATRIX'); print(m.shape[1])" 2>/dev/null || echo "0")
    if [[ -n "$numFrames" && "$numFrames" -ge "$minFramesForStreaming" ]]; then
      local skip_streaming=false
      if [[ -f "$STREAMING_PID_FILE" ]]; then
        local spid
        spid=$(cat "$STREAMING_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$spid" ]] && kill -0 "$spid" 2>/dev/null; then
          echo "[CONSUMER] Streaming already in flight (pid=$spid, frames=$numFrames), skipping."
          skip_streaming=true
        fi
      fi
      if [[ "$skip_streaming" == "false" ]]; then
        mkdir -p "$streamingOutputDir"
        # Unique prefix: date has only 1s resolution — multiple runs in the same
        # second used to overwrite the same file, producing "identical" outputs.
        local outPrefix="$streamingOutputDir/streaming_f${numFrames}_$(date +%Y%m%d%H%M%S)_$$_${RANDOM}"
        echo "[CONSUMER] Launching streaming metrics in background (frames=$numFrames) -> $outPrefix"
        (
          run_streaming_metrics "$RUN_MATRIX" "$outPrefix" || echo "[CONSUMER] Streaming metrics failed (non-fatal)"
          rm -f "$STREAMING_PID_FILE"
        ) &
        local streaming_bg_pid=$!
        echo "$streaming_bg_pid" > "$STREAMING_PID_FILE"
        disown "$streaming_bg_pid" 2>/dev/null || true
      fi
    fi
  fi
  fi

  # Delete only prev. curr becomes the next job's prev and is deleted when that job runs.
  # With a retention backend (ZSTD or BORG): archive prev/curr first, then delete prev only
  # on success (a failed archive keeps the local copy so no snapshot is lost). ZSTD and BORG
  # are mutually exclusive; ZSTD takes precedence if somehow both are set.
  if [[ -f "$prev" ]]; then
    if [[ "$ZSTD" == "1" || "$ZSTD" == "true" || "$ZSTD" == "yes" ]]; then
      if retain_zstd_delta "$prev" "$curr"; then
        delete_file "$prev" "snapshot (zstd delta stored)"
      else
        echo "[CONSUMER] WARNING: keeping $prev on disk (zstd delta did not succeed)"
      fi
    elif [[ "$BORG" == "1" || "$BORG" == "true" || "$BORG" == "yes" ]]; then
      if archive_with_borg "$prev"; then
        delete_file "$prev" "snapshot (archived to borg)"
      else
        echo "[CONSUMER] WARNING: keeping $prev on disk (borg archive did not succeed)"
      fi
    else
      delete_file "$prev" "snapshot"
    fi
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
