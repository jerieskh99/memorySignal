# Consumer And Run Matrix

> **Full subsystem (launcher + producer + consumer + artifacts):** [`../QEMU_CAPTURE_PIPELINE.md`](../QEMU_CAPTURE_PIPELINE.md)

## Purpose
This section documents the active consumer stage: how queue jobs are processed, how delta outputs are generated, how the per-step or default run matrix is accumulated, and when live streaming metrics are triggered.

## Relevant Files
- `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`

## Support Level
- Directly supported for queue handling, delta processing, matrix accumulation, live streaming triggers, and current config state
- Inferred only when translating config flags into operational scope statements

## Out Of Scope
- Internal implementation of the Rust delta binary
- Mathematical internals of `coherence_temp_spec_stability`
- Raw-retention metrics as an active branch, because the checked config disables them

## Consumer Role
The consumer operates as the downstream stage of the producer-consumer pipeline. Its top-level comments summarize the active responsibilities:

```2:13:VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh
# QEMU consumer for memory snapshots.
# High-level responsibilities:
# - Treat each producer snapshot pair as an opaque byte blob (ELF or RAW, does not matter).
# - For every job { prev, curr, output }:
#     - Run the Rust delta binary to compute per-page cosine / hamming distances.
#     - Append the resulting 1D frame vector as a new *column* in a long-lived run_matrix.npy
#       with shape [num_pages, num_frames].
#     - Optionally run streaming metrics (e.g. PLV/MSC/Cepstrum) on the accumulated matrix.
#     - Optionally maintain a rolling window of RAW dumps and run stability metrics on a
#       derived "raw feature" matrix.
# - Drive the queue state machine: pending -> processing -> done/failed.
```

Within the active project phase, the important point is that the consumer treats adjacent memory dumps as opaque inputs for delta computation. It does not decode guest semantics.

## Queue State Machine
The consumer creates and uses the queue subdirectories:

- `pending`
- `processing`
- `done`
- `failed`

Its main loop repeatedly picks the oldest pending JSON job, moves it to `processing`, processes it, and finally moves it to `done` or `failed`:

```339:356:VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh
while true; do
  jobFile=$(find "$qPending" -maxdepth 1 -name '*.json' -print 2>/dev/null | sort | head -1)
  if [[ -z "$jobFile" ]]; then
    sleep 0.1
    continue
  fi

  jobName=$(basename "$jobFile")
  jobProcessing="$qProcessing/$jobName"
  mv "$jobFile" "$jobProcessing"

  if process_job "$jobProcessing"; then
    : # moved to done
  else
    : # moved to failed
  fi
done
```

This queue layout is what allows the host controller to stop the producer first, wait for `pending` and `processing` to reach zero, and only then stop the consumer.

## Delta Processing
For each job, the consumer reads `prev`, `curr`, and `output` from the JSON file and invokes the Rust delta binary declared in the config:

```203:218:VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh
process_job() {
  local jobPath="$1"
  ...
  prev=$(jq -r '.prev' "$jobPath")
  curr=$(jq -r '.curr' "$jobPath")
  output=$(jq -r '.output' "$jobPath")

  echo "[CONSUMER] Running delta: prev=$(basename "$prev") curr=$(basename "$curr")"

  if ! "$rustDeltaCalculationProgram" "$prev" "$curr" "$output"; then
    echo "[CONSUMER] ERROR: Rust delta failed (rc=$?)"
    mv "$jobPath" "$qFailed/"
    return 1
  fi
```

In the checked config, `rustDeltaCalculationProgram` points to:

- `/project/homes/jeries/memorySignal/VM_sampler/VM_Capture/live_delta_calc/target/release/live_delta_calc`

That binary is external to the active `VM_Capture_QEMU` folder, so this booklet treats it as a dependency rather than reconstructing its internals.

## Run Matrix Accumulation
After delta generation, the consumer locates the newest delta frame under `output/<metric>/` and appends it as a new column in `RUN_MATRIX`:

```220:231:VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh
  local subdir="cosine"
  [[ "$deltaMetric" == "hamming" ]] && subdir="hamming"
  local latestFrame
  if [[ -d "$output/$subdir" ]]; then
    latestFrame=$(ls -t "$output/$subdir"/*.txt 2>/dev/null | head -1)
  fi
  ...
  append_frame "$latestFrame"
  echo "[CONSUMER] Appended frame from $latestFrame"
```

The matrix convention is explicit:

- on disk: shape `[num_pages, num_frames]`
- each processed job contributes one new time column
- the controller can override the default `queueDir/run_matrix.npy` with a step-specific `RUN_MATRIX`

This is how the active controller isolates each workload step into its own `run_matrix_<test_name>.npy`.

## Live Streaming Metrics
Live streaming is controlled by both config and the controller's offline mode choice.

In `config_qemu_upc.json`, the checked active state is:
- `streaming.enabled = true`
- `streaming.minFramesForStreaming = 128`
- `streaming.deltaMetric = "cosine"`
- `streaming.streamingOutputDir = "/project/homes/jeries/memory_traces/streaming_results"`

The consumer triggers streaming only when:

1. `OFFLINE_MODE` is not enabled
2. streaming is enabled in config
3. the run matrix exists
4. the number of frames is at least `minFramesForStreaming`
5. no earlier streaming process is still in flight

```237:265:VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh
  if [[ "$OFFLINE_MODE" != "1" && "$OFFLINE_MODE" != "true" ]] && \
     [[ "$streamingEnabled" == "true" && -f "$RUN_MATRIX" && -n "$streamingOutputDir" ]]; then
    local numFrames
    numFrames=$(python3 -c "import numpy as np; m=np.load('$RUN_MATRIX'); print(m.shape[1])" 2>/dev/null || echo "0")
    if [[ -n "$numFrames" && "$numFrames" -ge "$minFramesForStreaming" ]]; then
      ...
      (
        run_streaming_metrics "$RUN_MATRIX" "$outPrefix" || echo "[CONSUMER] Streaming metrics failed (non-fatal)"
        rm -f "$STREAMING_PID_FILE"
      ) &
```

Because `run_files_controlled.py` exports `OFFLINE_MODE=1` when `OFFLINE_METRICS_MODE=1`, live streaming is deliberately suppressed in the step-gated offline path.

## Cleanup And Retention
The active cleanup behavior is asymmetric:

- `prev` is deleted after it has been consumed, unless a Borg archival handoff is enabled
- `curr` is normally kept because it becomes the next job's `prev`
- if raw retention were enabled, `curr` could instead be moved into a raw dump retention directory

This rolling-chain behavior is important because it preserves adjacency across dump pairs without re-dumping the earlier state.

## Configured But Disabled Branch: Raw Retention
`capture_consumer_qemu.sh` contains a substantial raw-retention and raw-metrics branch. However, in the checked active config:

```19:20:VM_sampler/VM_Capture_QEMU/config_qemu_upc.json
  "rawRetention": {
    "enabled": false,
```

For the current project phase, that means:

- the raw-retention branch exists in code
- it should be documented only as conditional or inactive
- it should not be presented as part of the default active pipeline

## Relationship To The Host Controller
The host controller depends on the consumer in three practical ways:

- the controller waits for consumer queue drain before ending a step
- the controller relies on the consumer to populate the step-specific run matrix
- the controller rotates delta text files only after the consumer has finished writing them

The consumer therefore defines the active boundary between capture and feature computation.
