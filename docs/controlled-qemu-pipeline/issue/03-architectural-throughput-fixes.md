# Architectural Throughput Fixes

Companion to `02-capture-slowdown-ranking-and-timestamp-analysis.md`.
Each fix below targets one of the ranked slowdown sources and is ordered from
highest expected throughput impact to lowest. Items marked as safe keep the
existing scientific contract (per-page hamming/cosine per frame, pages x time
run matrix). Items marked as semantic impact change the data produced and
require validation before adoption.

## 1. Remove the per-cycle full-memory pause+dump from the hot path

Targets slowdown #1 and #5.

Current: each cycle does `virsh suspend` + full `pmemsave` of 2GB + `virsh resume`.
This is the largest single wall-time contributor, and it freezes guest time.

### Option A (safe, biggest win)

Reduce the working set captured per cycle.

- Lower `ramSizeMb` for experiments that do not need the full guest image.
- Or capture only the guest physical address range actually exercised by the workload (if known), instead of 0..ramSizeBytes.

Impact: cycle cost roughly linear in captured bytes. Halving capture size roughly halves cycle time, and halves pressure on every downstream stage.

### Option B (semantic impact, largest architectural win)

Move from full-memory dumps to incremental dirty-page capture.

- Use QEMU dirty-bitmap / live migration dirty tracking to emit per-cycle only the pages that changed since the previous sample.
- Producer writes a compact "changed pages + offsets" artifact.
- Consumer still produces per-page hamming/cosine, but only over changed pages; unchanged pages are filled from a constant already known to your pipeline (identical-page calibrated cosine, 0 hamming).

Impact: cycle wall time becomes proportional to change magnitude, not full RAM. For idle/stable workloads this is orders of magnitude cheaper. Requires a redesign of producer and of the Rust delta input format.

Risk: different capture semantics; must be validated against current method before adoption.

## 2. Stop operating above service capacity (queueing control)

Targets slowdown #2.

Current symptom: `intervalMsec=100` produces far faster than the consumer can service; queue always under backpressure; wall time diverges from guest time.

### Fix (safe)

Make sampling rate adaptive to measured service time, not a fixed target.

- Producer records the rolling mean end-to-end service time `T_svc` (timestamp of `done/*.json` - timestamp of enqueue).
- Producer sets its effective sleep to `max(intervalMsec, k * T_svc)` with `k` around 1.0-1.2.
- `maxPendingJobs` is kept small (e.g., 2-4) so backlog cannot mask drift.

Impact: eliminates the "tries to sample 10x faster than the system can handle" regime, stabilizes inter-sample timing, and removes hidden variability from backpressure waits.

## 3. Fix the O(N) append to the run matrix

Targets slowdown #3.

Current: each frame rewrites the entire `run_matrix_<step>.npy` via `np.hstack` + `np.save`. Late frames in a step cost proportionally more than early frames.

### Fix (safe)

Replace the append model with an O(1) append layout.

Choose one:

### Option A: append-only binary stream of fixed-width frames

- Consumer writes each delta frame to `run_matrix_<step>.bin` as a raw float32/float64 block of length `num_pages`.
- At step-drain time (controller already waits for drain), a finalizer converts `.bin` to `.npy` once (single O(N) cost, not per frame).

### Option B: chunked .npz / memmap

- Store columns as a memory-mapped array with known `(num_pages, max_frames)` capacity and a frame counter file.
- Append = one memmap write, no load, no re-save.

Impact: removes per-frame re-read/re-write of the entire accumulating matrix. Consumer service time per job becomes roughly constant instead of growing.

## 4. Make the Rust delta compute read-once and allocation-free per pair

Targets slowdown #4.

Current Rust binary already has the identical-page fast path and reused buffers (post-optimization), but still reads both full snapshots from disk per pair.

### Fix A (safe, cheap)

Keep the most recent RAW in memory across jobs.

- Consumer already processes pairs as a rolling chain (prev,curr -> then curr,next).
- Pass the previously-loaded `curr` bytes directly into the next delta call, so only the new `curr` has to be read from disk.

Impact: roughly halves per-pair I/O.

### Fix B (safe, higher gain)

Do the delta compute in-process (consumer) without spawning the Rust binary per pair.

- Keep the Rust compute, but expose it as a long-lived worker that accepts pairs on a pipe/socket.
- Removes per-pair process spawn, per-pair file open/close, per-pair thread pool setup.

Impact: reduces per-job fixed overhead independent of snapshot content.

## 5. Replace queue-as-filesystem with a lighter-weight signalling layer

Targets slowdown #5 and queue-scan cost (referenced by reference-scan reaper).

Current: jobs are JSON files; directories are the state machine. `dump_is_referenced` walks pending+processing JSON files to decide deletions.

### Fix (safe)

Add an in-memory index maintained by the consumer:

- Consumer maintains a set of "live referenced dumps" updated on queue transitions.
- Filesystem queue remains the durable log, but deletion decisions consult the in-memory set first.

Impact: removes O(queue_size) filesystem scans per deletion decision, which becomes noticeable under backlog conditions.

## 6. Separate capture from metrics entirely

Targets slowdown #6 and overall pipeline coupling.

Current: offline metrics run inside the step lifecycle. This is fine for correctness (queue drains first), but any metric regression inflates step wall time.

### Fix (safe, optional)

Make offline metrics a post-run batch stage:

- Controller completes all capture steps first, writing `run_matrix_<step>.npy` per step.
- A second pass iterates steps offline and runs metrics.
- Optionally done in parallel across steps when inputs are independent.

Impact: reduces per-step wall time; metrics computation no longer blocks the next step's start.

## 7. Storage layer alignment (supporting fix for #1, #3, #4)

Current: RAW dumps land in `/var/lib/libvirt/qemu/dump`, run matrix in project home. If these live on the same spindle or on a slow shared filesystem, every slowdown above is amplified.

### Fix (safe, infra-level)

- Put `imageDir` on the fastest local device available (NVMe, tmpfs, or a dedicated SSD partition).
- Put `queueDir` and `run_matrix_*.npy` on the same fast device to avoid cross-device copies.
- Avoid NFS/home mounts for hot-path artifacts during capture.

Impact: removes a class of latency spikes in `pmemsave` and in consumer appends; improves the variance component of inter-sample timing.

## Suggested order of adoption

Ordered by ratio of expected gain to implementation risk:

1. Storage alignment (#7) - infra-only, no code risk.
2. Adaptive sampling rate (#2) - eliminates the root cause of backlog.
3. O(1) run-matrix append (#3) - removes the growing per-job cost.
4. In-memory queue index (#5) - reduces filesystem work.
5. Long-lived Rust worker / read-once (#4) - shaves fixed overhead per pair.
6. Reduce captured memory size (#1-A) - direct linear speedup.
7. Dirty-page capture redesign (#1-B) - largest long-term win but requires validation.

## What not to do

- Do not raise `maxPendingJobs` to hide backlog. It does not increase throughput; it only delays the visible backpressure signal while accumulating more RAW files.
- Do not shorten `intervalMsec` further without changing the capture mechanism; it guarantees overload.
- Do not parallelize consumers until the matrix append is O(1) and dump deletion uses the in-memory index; otherwise you add contention without speedup.

## Validation checklist before adopting any change

- Confirm per-page hamming/cosine output per pair is numerically unchanged on a known fixture.
- Confirm `run_matrix_<step>.npy` final shape and ordering match the pre-change artifact.
- Confirm offline metrics outputs (`plv_baseline_aware.json`, `streaming.*`) are bit-equivalent on a short controlled run.
