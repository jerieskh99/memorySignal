# QEMU VM memory capture for macOS (producer + consumer + streaming)

This directory provides a **QEMU**-based capture pipeline for **macOS hosts**. Unlike the Linux version (`VM_Capture_QEMU`), this uses **QEMU monitor commands** directly (not virsh/libvirt), since macOS doesn't have libvirt.

## Differences from VM_Capture_QEMU (Linux)

| Aspect | VM_Capture_QEMU (Linux) | VM_Capture_QEMU_MAC |
|--------|-------------------------|---------------------|
| Hypervisor | QEMU/KVM via `virsh` (libvirt) | QEMU directly via monitor socket |
| Platform | Linux | macOS |
| VM Control | `virsh suspend/resume` | QEMU monitor `stop`/`cont` |
| Memory Dump | `virsh dump --memory-only` | QEMU monitor `dump-guest-memory elf` |
| Status Check | `virsh domstate` | QEMU monitor `info status` |
| Monitor Access | N/A (virsh handles it) | Unix socket (socat/nc) |

## Flow

1. **Producer** (`capture_producer_qemu_mac.sh`)
   - Pauses the VM: sends `stop` to QEMU monitor socket
   - Dumps guest memory: sends `dump-guest-memory elf <file>` to QEMU monitor
   - Resumes: sends `cont` to QEMU monitor
   - Enqueues a job `{ prev, curr, output }` into `queueDir/pending/` (does **not** delete any snapshot)
   - Repeats every `intervalMsec`

2. **Consumer** (`capture_consumer_qemu_mac.sh`) — runs **alongside** the producer
   - Same as Linux version: picks jobs, runs Rust delta, accumulates frames, runs streaming metrics, handles raw retention, deletes snapshots

## Requirements

- **macOS** host with:
  - **QEMU** installed (e.g. `brew install qemu`)
  - **socat** or **nc** (netcat) for QEMU monitor communication (`brew install socat`)
  - `jq` (`brew install jq`)
  - `bash`
  - Python 3 + numpy (for consumer append/streaming)
- **QEMU VM running** with a monitor socket enabled:
  ```bash
  qemu-system-x86_64 \
    -monitor unix:/tmp/qemu-monitor.sock,server,nowait \
    ...other QEMU args...
  ```
- Built **live_delta_calc** (same Rust binary as in VM_Capture).
- For streaming: project root with `coherence_temp_spec_stability.streaming_metrics` (set `streaming.projectRoot` in config).

## Config

Copy `config_qemu_mac.json.example` to `config_qemu_mac.json` and set:

- **qemuMonitorSocket**: path to QEMU monitor Unix socket (e.g. `/tmp/qemu-monitor.sock`). **Must match** the `-monitor` argument used when starting QEMU.
- **imageDir**: directory where memory dumps (ELF) are written.
- **outputDir**: directory passed to the Rust delta program (it writes `cosine/` and `hamming/` here).
- **rustDeltaCalculationProgram**: path to `live_delta_calc` binary.
- **queueDir**: base path for queues (`pending`, `processing`, `done`, `failed`).
- **streaming** (optional): same as Linux version.
- **rawRetention** (optional): same as Linux version.

### Raw dump retention + raw-matrix + metrics (optional)

When enabled, the consumer keeps a **fixed number N of raw memory dumps** (ELF files), builds a **raw time-series matrix** from them, and runs the **same stability metrics** (PLV / MSC / Cepstrum) on that raw matrix so you can compare **RAW-metrics** vs **DELTA-metrics**.

- **rawRetention.enabled** (default: `false`): set to `true` to enable. Env override: `RAW_RETENTION=1`.
- **rawRetention.keepDumps**: how many raw dumps to keep (e.g. `50`). Env override: `RAW_KEEP_DUMPS=50`.
- **rawRetention.rawDir**: directory where the newest N dumps are stored (only **curr** is retained per job; **prev** is always deleted to keep the rolling chain correct).
- **rawRetention.rawMatrixNpy**: path of the built matrix (shape `[num_pages, num_frames]`).
- **rawRetention.rawBuild**: builder that turns raw dumps into a matrix:
  - **enabled**, **builderProgram** (e.g. `python3 -m coherence_temp_spec_stability.raw_matrix_builder`), **mode** (`mean_byte` | `var_byte` | `entropy` | `popcount`), **pageSize**, **maxBytes** (0 = whole file).
- **rawRetention.rawMetrics**: run stability validator on the raw matrix:
  - **enabled**, **projectRoot**, **validatorModule** (e.g. `coherence_temp_spec_stability.stability_validator`), **windowSize**, **stepSize**, **outputDir**. Outputs are written under **outputDir** with prefix **raw** (e.g. `raw.npz`, `raw.json`) so they do not overwrite delta outputs.

Example: set `rawRetention.enabled` to `true`, set `keepDumps` to `5` for testing. After processing ≥5 jobs, `rawDir` will contain exactly 5 dumps, `rawMatrixNpy` will exist with shape `[num_pages, 5]`, and the validator will write to `rawMetrics.outputDir`.

## Run

```bash
cd VM_sampler/VM_Capture_QEMU_MAC
cp config_qemu_mac.json.example config_qemu_mac.json
# edit config_qemu_mac.json (set qemuMonitorSocket to match your QEMU -monitor arg)

# Foreground (run producer and consumer in two terminals)
./run_qemu_capture_mac.sh

# Background (same shell, logs to producer.log / consumer.log)
BACKGROUND=1 ./run_qemu_capture_mac.sh
```

Override paths via environment:

- `CONFIG` — config file path
- `PRODUCER_SCRIPT` / `CONSUMER_SCRIPT` — script paths
- `RUN_MATRIX` — consumer's accumulated matrix file (default: `queueDir/run_matrix.npy`)
- `RAW_RETENTION=1` — enable raw retention (overrides `rawRetention.enabled`)
- `RAW_KEEP_DUMPS=N` — keep N raw dumps (overrides `rawRetention.keepDumps`)

## QEMU Setup

**Important**: Your QEMU VM must be started with a monitor socket. Example:

```bash
qemu-system-x86_64 \
  -monitor unix:/tmp/qemu-monitor.sock,server,nowait \
  -m 4G \
  -hda your-disk.qcow2 \
  ...other options...
```

The socket path (`/tmp/qemu-monitor.sock`) must match `qemuMonitorSocket` in your config.

**Monitor commands** (for testing):
```bash
# Check VM status
echo "info status" | socat - UNIX-CONNECT:/tmp/qemu-monitor.sock

# Pause VM
echo "stop" | socat - UNIX-CONNECT:/tmp/qemu-monitor.sock

# Resume VM
echo "cont" | socat - UNIX-CONNECT:/tmp/qemu-monitor.sock

# Dump memory (manual test)
echo "dump-guest-memory elf /tmp/test.elf" | socat - UNIX-CONNECT:/tmp/qemu-monitor.sock
```

## Notes

- **QEMU monitor** commands are sent via Unix socket using `socat` (preferred) or `nc` (fallback).
- The producer checks that the monitor socket exists before starting.
- Backpressure: if `pending + processing` job count exceeds `backpressure.maxPendingJobs`, the producer sleeps before taking the next capture.
- The consumer uses a lock file around appending to the run matrix so only one process should run the consumer (or use a single consumer process).
- **macOS-specific**: If `socat` is not installed, the script will try `nc` (netcat), but `socat` is more reliable. Install with `brew install socat`.
