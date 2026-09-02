# Capture Slowdown Ranking and Timestamp Analysis

## Scope and evidence

This note rates likely slowdown sources for the active controlled QEMU pipeline, using:

- code-path evidence in:
  - `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh`
  - `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
  - `VM_sampler/VM_Capture/live_delta_calc/src/main.rs`
  - `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`
- runtime evidence provided by user snippets:
  - `done/*.json` filename timestamps
  - `consumer.log` lines (`Running delta`, `Appended frame`, `Job done`)

This is a ranked engineering diagnosis, not a full measured benchmark report.

## Executive finding

The dominant cause of wall-clock expansion is **capture cadence mismatch**:

- configured target interval: `intervalMsec = 100` (10 samples/sec target)
- actual per-job service time visible in logs: roughly **~57s to ~68s** between consecutive `memory_dump_cosine_results_par-<timestamp>.txt` writes in the shown excerpt

This means the system is operating in strong backlog mode: producer intent is far faster than end-to-end service capacity.

## Timestamp observations from provided snippets

### A) Done queue filenames (producer/job cadence seen in queue artifacts)

Examples from provided list:

- `20260421144135177.json`
- `20260421144310941.json`  (about +95.8s)
- `20260421144455920.json`  (about +104.9s)
- `20260421144632308.json`  (about +96.4s)

The spacing is in the order of about 1-2 minutes and not stable.

### B) Consumer log timestamps (consumer service cadence)

From provided excerpt:

- `...results_par-20260401043138.txt`
- `...results_par-20260401043246.txt`  (+68s)
- `...results_par-20260401043343.txt`  (+57s)
- `...results_par-20260401043446.txt`  (+63s)
- `...results_par-20260401043550.txt`  (+64s)
- `...results_par-20260401043646.txt`  (+56s)
- `...results_par-20260401043751.txt`  (+65s)

Observed service time is roughly about a minute per processed pair in this segment.

## Ranked slowdown contributors (by likely impact)

## 1) Full-memory `pmemsave` under VM pause (highest impact)

### Why high

Producer loop explicitly does:

1. `virsh suspend`
2. `qemu-monitor-command pmemsave 0 <ramSizeBytes> <file>`
3. optional `chown`
4. `virsh resume`

With `ramSizeMb: 2048`, each sample is a 2GB raw dump. This is heavy disk I/O plus pause/resume orchestration each cycle.

### Evidence

- `capture_producer_qemu_pmemsave.sh` loop and comments
- `config_qemu_upc.json`: `ramSizeMb = 2048`, `imageDir = /var/lib/libvirt/qemu/dump`

## 2) Target interval far below achievable service rate (very high impact)

### Why high

`intervalMsec=100` requests a new capture intent every 0.1s, while observed completed consumer work is on the order of ~60s/job in the provided segment. This guarantees queue pressure and backpressure waits.

### Evidence

- `config_qemu_upc.json`: `intervalMsec = 100`
- producer has explicit backpressure gate on pending+processing
- observed done/log timestamps are minute-scale

## 3) Consumer matrix append cost grows with frame count (high impact over long runs)

### Why high

`append_frame()` currently loads full `RUN_MATRIX`, does `np.hstack`, and writes full matrix back on each frame. This is O(total_matrix_size) every append, so later frames cost more than early frames.

### Evidence

- `capture_consumer_qemu.sh` `append_frame()` python blocks:
  - `mat = np.load(mat_path)`
  - `new_mat = np.hstack([mat, frame])`
  - `np.save(out_path, new_mat)`

## 4) Rust delta compute + I/O per pair (medium-high impact)

### Why medium-high

Even optimized, Rust still reads both full snapshots and computes per-page metrics across all pages. With 2GB snapshots, this is substantial CPU+I/O per job.

### Evidence

- `live_delta_calc/src/main.rs` reads full files in chunked parallel segments
- computes hamming + cosine for every page

## 5) Suspend/resume state polling and control overhead (medium impact)

### Why medium

Producer waits for VM states (`paused`, `running`) each cycle, adding control latency and jitter.

### Evidence

- `wait_state()` calls in producer
- polling with `vmStatePolling` config

## 6) Offline metrics stage (low impact for RAW accumulation symptom)

### Why low for this symptom

Offline metrics run after step queue drain in controller logic. They can extend total step wall time, but they are not the primary cause of dump accumulation during active capture.

### Evidence

- `run_files_controlled.py` calls offline only after queue drain and consumer stop path
- consumer accumulation/deletion behavior occurs during capture loop before step finalization

## Why timestamp gaps are unstable

The observed jitter is expected from variable per-cycle cost:

- `pmemsave` duration variance
- storage throughput variance
- queue/backpressure waits
- consumer service-time variance
- suspend/resume latency variance

Page similarity can influence Rust compute time (identical-page fast path), but this is secondary to capture/disk/backpressure effects in the presented evidence.

## Interpretation in one line

Current behavior is consistent with a queueing regime where requested sampling rate is orders of magnitude above service capacity, so timestamps reflect system throughput limits, not nominal `intervalMsec`.

## Practical immediate implication

If the goal is wall-clock realism, you need either:

- much cheaper per-sample capture path, and/or
- much lower sampling request rate, and/or
- reduced captured memory size / different capture strategy.

