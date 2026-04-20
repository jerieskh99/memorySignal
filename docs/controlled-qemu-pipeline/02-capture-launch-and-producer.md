# Capture Launch And Producer

> **Full subsystem (launcher + producer + consumer + artifacts):** [`../QEMU_CAPTURE_PIPELINE.md`](../QEMU_CAPTURE_PIPELINE.md)

## Purpose
This section explains how the host controller enters the active QEMU capture pipeline and how the active producer acquires RAW memory dumps and transforms them into queue jobs for downstream processing.

## Relevant Files
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`
- `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh`
- `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh`
- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`

## Support Level
- Directly supported for launcher invocation, background process startup, producer behavior, and active config fields
- Inferred only where the Python controller relies on shell-script defaults rather than selecting a value itself

## Out Of Scope
- Alternate producer implementations not selected by the active default flow
- The SSH utility mode provided by `run_qemu_capture.sh`
- Detailed internals of the downstream consumer

## Launcher Handoff From The Host Controller
When `CAPTURE_MODE=1`, `run_files_controlled.py` starts capture by exporting a small set of environment variables and invoking `run_qemu_capture.sh` in background mode:

```171:174:VM_sampler/VM_Capture_QEMU/run_files_controlled.py
    cmd = (
        f"cd {root_q} && "
        f"{env_prefix}CONFIG={cfg_q} PRODUCER_SCRIPT={producer_q} BACKGROUND=1 ./run_qemu_capture.sh"
    )
```

The handoff establishes the active capture contract:

- `CONFIG` points to the active capture config, by default `config_qemu_upc.json`
- `PRODUCER_SCRIPT` points to the active producer, by default `capture_producer_qemu_pmemsave.sh`
- `BACKGROUND=1` causes both producer and consumer to be launched with `nohup`
- `RUN_MATRIX` is set per step so the consumer appends into an isolated matrix for that step
- `OFFLINE_MODE=1` is added only when `OFFLINE_METRICS_MODE=1`, disabling live streaming during capture so post-step offline processing can own the metric stage

## Why The Consumer Is Still Part Of This Path
The Python controller passes only `PRODUCER_SCRIPT`, not `CONSUMER_SCRIPT`. However, the launcher itself assigns a default consumer script:

```9:12:VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh
PRODUCER_SCRIPT="${PRODUCER_SCRIPT:-$ROOT/capture_producer_qemu_pmemsave.sh}"
CONSUMER_SCRIPT="${CONSUMER_SCRIPT:-$ROOT/capture_consumer_qemu.sh}"
CONFIG="${CONFIG:-$ROOT/config_qemu.json}"
BACKGROUND="${BACKGROUND:-0}"
```

In the active controller path, this means consumer startup is indirectly selected through launcher defaults. That is why this booklet treats consumer participation as part of the active flow, but labels the selection mechanism as inferred from the launcher rather than explicit in Python.

## Background Startup Behavior
With `BACKGROUND=1`, the launcher starts both capture processes in the same shell session using `nohup`, writes their process IDs into `capture_pids.txt`, and logs to `producer.log` and `consumer.log`:

```70:79:VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh
if [[ "${BACKGROUND}" == "1" || "${BACKGROUND}" == "true" ]]; then
  echo "Starting producer and consumer in background (same terminal, nohup)."
  nohup bash "$PRODUCER_SCRIPT" >> "$ROOT/producer.log" 2>&1 &
  PROD_PID=$!
  nohup bash "$CONSUMER_SCRIPT" >> "$ROOT/consumer.log" 2>&1 &
  CONS_PID=$!
  echo "$PROD_PID" > "$ROOT/capture_pids.txt"
  echo "$CONS_PID" >> "$ROOT/capture_pids.txt"
```

The controller later uses process-name-based `pkill` calls to stop producer and consumer separately.

## Active Producer Role
The active producer is `capture_producer_qemu_pmemsave.sh`. Its comments describe the intended behavior clearly:

```2:16:VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh
# QEMU/libvirt producer (qemu:///system) using qemu-monitor-command pmemsave.
# High-level goal: generate a *sequence* of flat RAW physical-memory images
# for a running libvirt VM, queuing prev/curr pairs for the existing consumer
# while keeping the VM paused only around the pmemsave window.
#
# Capture loop for each iteration:
# - Enforce backpressure on the queue (do not create new dumps if consumer is behind).
# - Pause the VM via `virsh -c qemu:///system suspend` and wait until domstate == paused.
# - Ask QEMU (via `virsh qemu-monitor-command`) to run pmemsave(0, ramSizeBytes, newImage)
#   into a libvirt-owned directory (typically /var/lib/libvirt/qemu/dump).
```

## Producer Inputs From Config
The producer reads the active configuration from `config_qemu_upc.json`. In the checked file, the most relevant fields are:

- `domain`: `"Kali Jeries"`
- `ramSizeMb`: `2048`
- `imageDir`: `"/var/lib/libvirt/qemu/dump"`
- `outputDir`: `"/project/homes/jeries/memory_traces/output_dir"`
- `queueDir`: `"/project/homes/jeries/memory_traces/queue_dir"`
- `intervalMsec`: `100`
- `backpressure.maxPendingJobs`: `20`
- `vmStatePolling.timeoutSeconds`: `30`
- `chownUser` and `chownGroup`: both `"jeries"`

These values are directly consumed by the producer script and determine where dumps are written and when jobs are throttled.

## RAW Dump Acquisition Loop
Each producer iteration does the following:

1. Count pending and processing jobs.
2. Sleep when queue backpressure exceeds the configured maximum.
3. Compute a timestamped RAW dump path under `imageDir`.
4. Suspend the VM with `virsh suspend`.
5. Wait until `domstate == paused`.
6. Use `virsh qemu-monitor-command ... pmemsave` to dump physical memory.
7. Optionally `sudo chown` the dump so user-space tools can read it.
8. If there was a previous dump, write a queue job pairing `prev` and `curr`.
9. Store `curr` as the next iteration's `prev`.
10. Resume the VM and sleep for the configured interval.

The queue job emission is explicit:

```135:145:VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh
  if [[ -n "$prevImage" && -f "$prevImage" ]]; then
    jobId="$timestamp"
    jobTmp="$qPending/${jobId}.json.tmp"
    jobFile="$qPending/${jobId}.json"
    jq -n \
      --arg prev "$prevImage" \
      --arg curr "$newImage" \
      --arg output "$outputDir" \
      '{ prev: $prev, curr: $curr, output: $output }' > "$jobTmp"
    mv "$jobTmp" "$jobFile"
```

## Architectural Meaning Of The Producer
Within the active pipeline, the producer does not perform semantic inspection of guest memory. Its job is narrower:

- acquire time-ordered RAW memory states
- preserve adjacency through `prev` and `curr` pairing
- hand off each adjacent pair to the consumer using queue files

This matches the project framing in which the active pipeline treats memory as a sequence of states whose evolution is later summarized by delta representations and downstream metrics.

## Directly Implemented Versus Inferred
### Directly Implemented
- The active producer uses `pmemsave`, not `virsh dump`, in the default controller path
- Queue jobs are JSON files written under `queueDir/pending`
- The producer pauses the VM only around the dump window
- Backpressure is enforced before new dumps are created

### Inferred Or Conditional
- The exact consumer process started alongside the producer is inferred from launcher defaults
- The producer assumes the configured `imageDir` is writable under libvirt/QEMU constraints and that `sudo chown` is available when ownership changes are needed

## Exclusions For This Section
This section intentionally excludes the alternate ELF producer and the user-run RAW producer. Both are present in the directory, but they are not part of the active default flow rooted at `run_files_controlled.py`.
