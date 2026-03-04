# QEMU VM memory capture (producer + consumer + streaming)

This directory provides a **QEMU/libvirt**-based alternative to the VBoxManage (VirtualBox) capture pipeline. It is intended to run on a **Linux** host where the VM is managed by libvirt (e.g. KVM/QEMU).

## Differences from VM_Capture (VirtualBox)

| Aspect | VM_Capture (VBox) | VM_Capture_QEMU |
|--------|-------------------|-----------------|
| Hypervisor | VirtualBox (`VBoxManage`) | QEMU/KVM (`virsh`) |
| Platform | Windows (PowerShell) | Linux (bash) |
| Snapshots | Consumer deletes previous image after delta | Snapshots kept until consumer has run **delta + streaming metrics**, then both snapshots deleted |
| Streaming | Not integrated | Optional: accumulate delta frames, run `streaming_metrics` when enough frames, then delete snapshot files |

## Flow

1. **Producer** (choose one)
   - **ELF via virsh** — `capture_producer_qemu.sh`
     - Suspends the VM: `virsh suspend <domain>`
     - Dumps guest memory: `virsh dump <domain> <file> --memory-only` (ELF dump)
     - Resumes: `virsh resume <domain>`
     - Enqueues a job `{ prev, curr, output }` into `queueDir/pending/` (does **not** delete any snapshot)
   - **RAW via pmemsave (qemu:///system)** — `capture_producer_qemu_pmemsave.sh`
     - Suspends the VM: `virsh -c qemu:///system suspend <domain>`
     - Uses `virsh qemu-monitor-command ... pmemsave` to dump **flat raw physical memory** into a libvirt-owned directory (e.g. `/var/lib/libvirt/qemu/dump`)
     - Optionally runs `sudo chown` on the new dump so the consumer (running as user) can read it
     - Resumes: `virsh -c qemu:///system resume <domain>`
     - Enqueues `{ prev, curr, output }` in the same way as the ELF producer
   - **RAW via user-run QEMU (no virsh)** — `capture_producer_qemu_user_raw.sh`
     - QEMU is started by the user with `-monitor unix:/tmp/qemu-monitor.sock,server,nowait`
     - Uses the monitor `pmemsave` command to write flat raw memory into a user-owned directory
     - No virsh / libvirt / sudo required; see **Raw physical memory (user-space only, no virsh)** below.

2. **Consumer** (`capture_consumer_qemu.sh`) — runs **alongside** the producer
   - Picks a job from `pending`, moves to `processing`
   - Runs the Rust delta program: `live_delta_calc prev curr outputDir`
   - Appends the new delta frame (cosine or hamming, one value per page) to an accumulated run matrix
   - When `streaming.enabled` is true and the matrix has at least `minFramesForStreaming` frames, runs **streaming metrics** on it (Python `streaming_metrics` module), then continues
   - **Deletes** both `prev` and `curr` snapshot files after processing
   - Moves the job to `done` or `failed`

So snapshots are kept only until the consumer has used them for delta + (optionally) streaming; then they are removed to free disk space while capture continues.

## Requirements

- **Linux** host with:
  - `virsh` (libvirt-client)
  - `jq`
  - `bash`
  - Python 3 + numpy (for consumer append/streaming)
- VM registered as a **libvirt domain** (e.g. `virsh list --all`).
- Built **live_delta_calc** (same Rust binary as in VM_Capture).
- For streaming: project root with `coherence_temp_spec_stability.streaming_metrics` (set `streaming.projectRoot` in config).

## Config

Copy `config_qemu.json.example` to `config_qemu.json` and set:

- **domain**: libvirt domain name (e.g. `my-vm`). Not used by the user-raw producer.
- **qemuMonitorSocket** (optional): for [user-space raw capture](RAW_CAPTURE_ALTERNATIVE.md), path to QEMU monitor Unix socket (e.g. `/tmp/qemu-monitor.sock`). Required when using `capture_producer_qemu_user_raw.sh`.
- **ramSizeMb** (optional): for raw capture, guest RAM size in MiB (e.g. `4096`). Required when using `capture_producer_qemu_user_raw.sh` and `capture_producer_qemu_pmemsave.sh` (used to compute the `pmemsave` size).
- **chownUser / chownGroup** (optional): when set (e.g. `jeries` / `jeries`), `capture_producer_qemu_pmemsave.sh` will run `sudo chown chownUser:chownGroup <dump>` after each `pmemsave` so the consumer can read the dump written under `/var/lib/libvirt/qemu/dump`.
- **imageDir**: directory where memory dumps (ELF or .raw) are written. For the pmemsave producer, this should typically be `/var/lib/libvirt/qemu/dump` or another libvirt-allowed directory.
- **outputDir**: directory passed to the Rust delta program (it writes `cosine/` and `hamming/` here).
- **rustDeltaCalculationProgram**: path to `live_delta_calc` binary.
- **queueDir**: base path for queues (`pending`, `processing`, `done`, `failed`).
- **streaming** (optional):
  - **enabled**: `true` to run streaming metrics when enough frames are accumulated.
  - **streamingMetricsModule**: Python module to run (default: `coherence_temp_spec_stability.streaming_metrics`).
  - **streamingOutputDir**: where to write streaming outputs (e.g. `.npz` / `.json`).
  - **minFramesForStreaming**: run streaming every time the run matrix has at least this many frames (default: 128).
  - **deltaMetric**: `cosine` or `hamming` (which delta output to use as the time series).
  - **projectRoot**: working directory when invoking the streaming module (root of `mem_sig` repo).

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
cd VM_sampler/VM_Capture_QEMU
cp config_qemu.json.example config_qemu.json
# edit config_qemu.json

# Foreground (ELF via virsh dump --memory-only)
./run_qemu_capture.sh

# Foreground (RAW via pmemsave under qemu:///system; dumps into /var/lib/libvirt/qemu/dump)
PRODUCER_SCRIPT=$ROOT/capture_producer_qemu_pmemsave.sh ./run_qemu_capture.sh

# Foreground (RAW via user-run QEMU + monitor pmemsave; no libvirt/system)
PRODUCER_SCRIPT=$ROOT/capture_producer_qemu_user_raw.sh ./run_qemu_capture.sh

# Background (same shell, logs to producer.log / consumer.log)
BACKGROUND=1 ./run_qemu_capture.sh
```

Override paths via environment:

- `CONFIG` — config file path
- `PRODUCER_SCRIPT` / `CONSUMER_SCRIPT` — script paths
- `RUN_MATRIX` — consumer’s accumulated matrix file (default: `queueDir/run_matrix.npy`)
- `RAW_RETENTION=1` — enable raw retention (overrides `rawRetention.enabled`)
- `RAW_KEEP_DUMPS=N` — keep N raw dumps (overrides `rawRetention.keepDumps`)

## Raw physical memory (user-space only, no virsh)

If the VM is under **qemu:///system**, `virsh dump` produces ELF dumps that may be owned by the service account (preventing user-space processing), and **pmemsave** via QMP cannot write into user directories due to libvirt confinement. To obtain a **flat raw physical memory image** (suitable for mmap and deterministic delta computation) **without sudo or any libvirt/system policy changes**, use the **user-run QEMU** alternative:

1. **Start the VM yourself** (not via system libvirt), with a monitor socket in user space, e.g.  
   `-monitor unix:/tmp/qemu-monitor.sock,server,nowait`
2. In config, set **qemuMonitorSocket** to that path and **ramSizeMb** to the guest RAM size in MiB.
3. Run the **raw producer** instead of the virsh producer:
   ```bash
   PRODUCER_SCRIPT=$ROOT/capture_producer_qemu_user_raw.sh ./run_qemu_capture.sh
   ```
   Output is `.raw` (flat raw); the **same consumer** and **live_delta_calc** are used.

See **[RAW_CAPTURE_ALTERNATIVE.md](RAW_CAPTURE_ALTERNATIVE.md)** for the full problem statement, constraints, and step-by-step methodology.

## Notes

- **virsh dump** pauses the VM during the dump by default; no need to suspend separately if you rely on that, but the script explicitly suspends/resumes so state is consistent.
- Backpressure: if `pending + processing` job count exceeds `backpressure.maxPendingJobs`, the producer sleeps before taking the next capture.
- The consumer uses a lock file around appending to the run matrix so only one process should run the consumer (or use a single consumer process).
