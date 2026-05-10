# Low-Level Rewrite Recommendations for VM Workload Executables

This is Deliverable 3 in the three-phase audit sequence. It translates the findings from `python_runtime_os_artifact_audit.md` and `artifact_confusion_matrix_connection.md` into specific, implementation-targeted recommendations for each workload.

The goal is experimental control over confounds, not performance. Every recommendation is anchored to a specific artifact or confusion identified in Phase 1.

---

## 1. Executive Verdict

### What requires redesign (not only rewrite)

`mem_pointer_chase` requires **redesign before rewrite**. The current workload is read-only and invisible to the delta pipeline. Rewriting in C without adding writes makes the invisible workload faster but not observable. The fundamental issue is workload semantics, not language.

### What requires both rewrite and parameter change

`mem_stream` requires a C rewrite AND an increase in working-set size. At 128 MB, the active fraction of guest memory is too small relative to the total page count. Most of the current `mem_stream → run_idle` confusion is a metric inadequacy caused by the small active fraction, not Python overhead.

### What benefits most from C

`mem_alloc_touch_pages` benefits most from a pure C rewrite because its intended signal is the mmap/munmap cadence, and the current Python path routes through glibc's allocator with nondeterministic threshold switching. A C version using `mmap`/`munmap` directly makes the cadence deterministic.

### IO tests: C is justified but for different reasons than MEM

IO tests are kernel-mediated, so Python overhead is small relative to disk and journal latency. However, C enables use of specific open flags, syscall variants, and preallocation strategies that expose kernel behavior more precisely. For IO tests, C is justified for **experimental clarity**, not performance.

### Is assembly justified?

Assembly is justified for exactly two things:
1. Cache-bypassing stores in `mem_stream` (MOVNTI/MOVNTQ) to force DRAM traffic independent of cache state.
2. A true data-dependent load chain in `mem_pointer_chase` write-variant to prevent speculative prefetch from hiding poor locality.

For all other tests, C is sufficient.

### What remains after rewriting

All of the following persist regardless of language:
- OS scheduler effects (kworker, rcu, softirq) inside the guest.
- pmemsave pause variance: workload-dependent, capture-pipeline issue.
- THP promotion (unless explicitly disabled with madvise).
- Page fault timing (still OS-mediated event).
- Writeback timing and dirty-page flush cadence.
- KVM/host scheduling effects.
- Hardware prefetcher (partially controllable with non-temporal stores but not fully suppressible).
- Virtual disk I/O latency variability.
- NUMA effects if the guest is on a NUMA host.

---

## 2. Per-Test Rewrite Plans

---

### 2.1 `run_idle.sh`

#### A. Current problem

The script executes `sleep N`. The signal during the IDLE window is entirely OS background. Phase 1 identified that IDLE scatters across 24 recordings because each idle follows a different prior workload, leaving different residual page-cache and writeback state. CV = 0.973. test1 and test3 have negative silhouette scores.

This is not a code problem. No rewrite can remove kernel residue from a passive observation window.

#### B. Rewrite objective

The objective is **experimental protocol change**, not code change. The goal is to make IDLE a more reproducible baseline.

#### C. What to control

- **Prior workload residue**: Drop caches before each IDLE window (`echo 3 > /proc/sys/vm/drop_caches` from an SSH privileged command). This is the highest-impact change.
- **Minimum idle settle time**: Increase idle window from 60 s to 120-180 s. Writeback from IO-heavy workloads can persist for 60-90 s.
- **Idle position documentation**: Record which workload preceded each IDLE run. Annotate the metadata (`meta.json` or a sidecar file) with the prior workload name and its position in the cycle.
- **Pre-idle metric snapshot**: Capture `/proc/meminfo` `Dirty` and `Writeback` at the start of each idle window. If these are nonzero, the idle is contaminated and can be flagged.

#### D. What should remain uncontrolled

Background OS daemons, kernel timers, and scheduler activity are the authentic IDLE signal. Do not suppress them. The goal is to suppress only the carry-over from the preceding workload.

#### E. Recommended implementation level

Keep bash `sleep`. The script is not the problem. Change the protocol:
1. Issue a cache-drop command before each idle (via SSH or a privileged companion script).
2. Increase idle duration to 120 s minimum.
3. Add a metadata sidecar capturing `Dirty`, `Writeback`, and the prior workload name.

#### F. Suggested variants

- **Clean idle**: cache-drop + 120 s sleep. The controlled baseline.
- **Contaminated idle (current)**: 60 s sleep after prior workload. Useful for studying residue decay if intentional.
- **Stratified analysis**: group the 24 existing IDLE recordings by prior-workload type (post-IO vs post-MEM vs post-idle-start). Analyze each stratum separately.

#### G. Verification instrumentation

- Per-second `/proc/meminfo` `Dirty`, `Writeback`, `Cached` during each idle window.
- `vmstat 1` for `bo` (blocks out): confirm writeback completion timing.
- `/proc/diskstats` delta to detect residual I/O.
- After the protocol change, compare per-stratum CV before and after cache-drop.

#### H. Expected effect on metrics

- CV across IDLE recordings should drop substantially after cache-drop + longer duration.
- `dc_coherence` mean and variance across IDLE runs should converge.
- test1 and test3 silhouette scores should improve toward the cluster mean.
- The IDLE class centroid should be more stable, directly improving the classifier boundary with `mem_stream`.

#### I. Experimental comparison

Run two matched cycle sets: one with the current 60 s idle, one with cache-drop + 120 s idle. Compare IDLE-class CV, silhouette, and the confusion rate for MEM workloads that currently confuse with IDLE.

---

### 2.2 `mem_stream.py`

#### A. Current problem

Phase 1 identified three separate problems:

1. **Small active fraction**: 128 MB out of 1-4 GB guest RAM. Most pages are static. Per-recording metrics are dominated by the static background. This is the leading cause of `mem_stream → run_idle`.

2. **CoW lazy allocation**: `np.zeros` returns zero-page-backed virtual memory. The first sweep causes 32K page faults mixed with steady-state writes. Warm-up is not separated from measurement.

3. **THP promotion**: `khugepaged` may promote the buffer from 4 KB pages to 2 MB pages mid-run. This changes the per-4-KB-page delta computation in unpredictable ways.

4. **Hardware prefetcher**: Sequential access pattern is the canonical prefetcher target. This compresses apparent DRAM traffic.

5. **Python loop rate**: Interpreter dispatch limits sweep throughput. The actual sweep period is interpreter-bound, not hardware-bound.

#### B. Rewrite objective

Produce a write-streaming workload that:
- Controls the active fraction of guest pages precisely.
- Separates warm-up (first-touch, THP settlement) from the measurement phase.
- Fixes page granularity to 4 KB for the entire measurement phase.
- Can operate with or without cache bypassing to produce two distinct signal profiles.
- Runs at hardware bandwidth, not interpreter bandwidth.

#### C. What to control

**Memory allocation**:
- `mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0)` with explicit size.
- `madvise(MADV_NOHUGEPAGE)` immediately after mmap to disable THP for the buffer. Fix page granularity for the entire run.
- `mlock()` after pre-faulting to prevent the OS from swapping or reclaiming the buffer.

**Lazy-memory behavior**:
- Warm-up phase: `memset(buf, 0, size)` to commit all pages before measurement begins. This triggers all page faults in a controlled window outside the measurement period.
- Measurement phase: only the sweep loop. All pages are already committed and resident.

**Cache behavior**:
- Variant 1 (cached writes): `buf[i] = v` with normal stores. Hardware prefetcher may predict the pattern.
- Variant 2 (cache-bypassing writes): `_mm_stream_si32((int*)(buf + i), v)` or inline assembly `MOVNTI` to write directly to DRAM. This disables cache and forces DRAM traffic regardless of prefetcher.

**Working set size**:
- Increase from 128 MB to 512 MB or 1 GB. At 1 GB, the active fraction is 25-50% of a 2-4 GB guest. Per-recording metrics are no longer diluted by the static background.

**Timing**:
- Replace `time.time()` with `clock_gettime(CLOCK_MONOTONIC)` for tighter timing.
- Or use RDTSC-based sweep counting to pace iterations deterministically.

**CPU affinity**:
- `sched_setaffinity(0, ...)` to pin to one CPU. Remove scheduler migration noise.

**Syscalls**:
- Zero syscalls during the measurement loop. Only `clock_gettime` for termination check.

#### D. What should remain uncontrolled

The write itself: one store per page is the intended signal. Do not remove it. Hardware prefetcher activity is acceptable to leave unless the variant explicitly targets DRAM traffic (in which case use MOVNTI).

#### E. Recommended implementation level

**C with optional inline assembly**.

C is required for: `mmap`, `madvise`, `mlock`, `sched_setaffinity`, separated warm-up phase.

Assembly (inline `__asm__` or compiler intrinsics `_mm_stream_si32`) is justified for the cache-bypassing variant. This is a scientifically interesting variant because pmemsave captures DRAM state. Cache-bypassing writes land in DRAM immediately, making them more reliably visible to the snapshot pipeline than cached writes that may be in L1/L2 at snapshot time.

#### F. Suggested variants

| Variant | Working set | Page granularity | Write method | Warm-up separation |
|---|---|---|---|---|
| `stream_small_cached` (baseline) | 128 MB (current) | 4 KB (NOHUGEPAGE) | cached store | yes |
| `stream_large_cached` | 1 GB | 4 KB | cached store | yes |
| `stream_large_nt` | 1 GB | 4 KB | MOVNTI (NT store) | yes |
| `stream_firsttouch` | 1 GB | 4 KB | cached store | no (include faults) |
| `stream_readonly` | 1 GB | 4 KB | read-only | yes (pure reads, for MI baseline) |

The `stream_readonly` variant deliberately produces near-zero delta signal. This confirms the metric pipeline's read/write asymmetry and establishes a noise floor.

#### G. Verification instrumentation

- `/proc/$pid/smaps` entry for the buffer: `AnonHugePages` field confirms whether THP promotion occurred. Check at start and end of run.
- `perf stat -e LLC-store-misses,dTLB-store-misses,minor-faults` to confirm hardware behavior differs between variants.
- `perf stat -e cache-misses` should be much higher for the MOVNTI variant (cache-bypassing).
- Snapshot delta frame comparison: compare delta magnitude per page between cached and MOVNTI variants. If MOVNTI increases per-page SNR, cache-latency was hiding writes from the snapshot pipeline.

#### H. Expected effect on metrics

- `snr_mean`, `dc_coherence`: increase significantly with 1 GB working set (larger active fraction).
- `snr_high_frac`: increase as active pages now represent 25%+ of total.
- `cepstral_peak_idx`: shift toward higher quefrency (more structured temporal signal).
- CV across runs: expected to decrease with warm-up separation and MADV_NOHUGEPAGE.
- `mem_stream → run_idle` confusion: expected to disappear with 1 GB working set.
- `mem_stream → io_rand_rw` confusion: depends on whether EO contamination was the cause. Unchanged by rewrite alone; requires the IDLE protocol change in 2.1.

#### I. Experimental comparison

Record four conditions in matched order:
1. Python original (128 MB, current parameters).
2. C, 128 MB, warm-up separated, MADV_NOHUGEPAGE, mlock.
3. C, 1 GB, warm-up separated, MADV_NOHUGEPAGE, mlock.
4. C, 1 GB, MOVNTI stores.

Compare: per-recording feature distributions, CV across cycles, silhouette score, confusion matrix. Each condition runs through the same 4-cycle protocol with the same capture pipeline.

---

### 2.3 `mem_pointer_chase.py`

#### A. Current problem

Phase 1 identified this as a **workload design flaw**: the chase loop only reads pages. A page-content-delta pipeline sees no signal from read operations. Three of four runs (test16, test28, test40) collapse to near-IDLE statistics (`snr_mean ≈ 0.05`). The single outlier (test4, cycle 1) is most plausibly the 1 GB initialization write phase, not the chase.

This is not a Python artifact. A C rewrite of the current design produces the same invisible signal faster.

#### B. Rewrite objective

Redesign the workload to produce a **pseudo-random write pattern** with poor cache locality, making the delta signal reflect genuine random-access memory pressure. Two sub-objectives:

1. Every page visit must write, not only read, so the delta pipeline observes the access pattern.
2. The access pattern must be genuinely difficult to predict by the hardware prefetcher.

#### C. What to control

**Workload redesign (mandatory)**:

The chase must include a write. The write must be data-dependent on the chase state so it cannot be hoisted or speculated away.

Recommended pattern: at each LCG step, write the current LCG state to byte 0 of the visited page:
```
buf[idx * page_stride] = (uint8_t)(x & 0xFF);   /* read-modify-write */
```
This causes:
- One write per page visit.
- Written value changes each step (LCG state progresses).
- Hamming delta between snapshots fires for every visited page.

**Memory allocation**:
- `mmap(NULL, 1GB, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0)`.
- `madvise(MADV_NOHUGEPAGE)` to maintain 4 KB page granularity.
- `mlock()` after pre-faulting to keep the working set resident.
- Pre-fault: `memset(buf, init_val, size)` before measurement starts. This initializes all pages to a known state, separating the 1 GB initialization write from the chase measurement.

**Access pattern control**:
- LCG with full-period configuration (same as current: a=1664525, c=1013904223, m=2^18 for 1 GB / 4 KB pages). Period = 262144, full working-set coverage.
- A data-dependent variant: instead of `idx = LCG(idx)`, use `idx = (buf[idx * stride] * A + C) % m`. This makes the next access address depend on the current read value, which prevents speculative prefetch from predicting the sequence.

**TLB pressure**:
- 1 GB / 4 KB = 262144 unique pages. Exceeds TLB capacity. TLB-miss-driven DRAM accesses are the intended mechanism. MADV_NOHUGEPAGE ensures no THP collapse.

**Syscalls**:
- Zero during measurement. Only `clock_gettime` at loop termination check.

**CPU affinity**: `sched_setaffinity` to pin to one CPU.

#### D. What should remain uncontrolled

TLB miss rate: this is the intended signal. Do not cache-hint or prefetch the chase targets. The difficulty of the hardware prefetcher is by design.

The random distribution of writes across 262K pages is the intended signal. Do not coerce this toward a pattern.

#### E. Recommended implementation level

**C for the base rewrite. Inline assembly for the data-dependent variant.**

C is sufficient for the LCG-chase-and-write design. The LCG state fits in a register; no boxing, no GIL, pure arithmetic + store.

Assembly is justified for the data-dependent addressing variant:
- In C, a compiler may reorder loads and precompute addresses speculatively.
- Inline assembly with `LFENCE` between loads ensures truly sequential dependent addressing.
- `MOVNTI` is NOT appropriate here: the intent is poor-locality DRAM pressure, not cache bypass. Use regular stores.

#### F. Suggested variants

| Variant | Access pattern | Page writes | Write value |
|---|---|---|---|
| `chase_read_only` (current Python) | LCG deterministic | none | n/a |
| `chase_write_lcg` (recommended) | LCG deterministic | yes, LCG state | `(uint8_t)x` |
| `chase_write_dependent` | data-dependent addressing | yes, address-dependent | `(uint8_t)(x & 0xFF)` |
| `chase_write_sequential` | sequential (control) | yes | `(uint8_t)i` |
| `chase_small` | LCG, 128 MB working set | yes | `(uint8_t)x` |

The `chase_read_only` variant should be recorded as a noise-floor calibration. It quantifies the background delta rate when the workload is invisible. Comparing this to `chase_write_lcg` isolates the pure write-access signal.

The `chase_write_sequential` is a control to distinguish "poor locality" from "regular locality" in the write pattern. If it looks like `mem_stream`, the spatial pattern of writes matters more than locality.

#### G. Verification instrumentation

- `/proc/$pid/io` `wchar` and `write_bytes`: must be nonzero for the write variants and zero for `chase_read_only`. Hash the buffer content before and after the read-only run to confirm no writes.
- `perf stat -e dTLB-load-misses,dTLB-store-misses,LLC-load-misses,LLC-store-misses`: confirm high TLB miss rate (expected: near 100% for 1 GB working set).
- Compare delta frame counts between `chase_read_only` and `chase_write_lcg`. The write variant should show orders-of-magnitude higher event rate.
- `/proc/$pid/smaps` `AnonHugePages`: must remain 0 with MADV_NOHUGEPAGE.

#### H. Expected effect on metrics

After redesign to `chase_write_lcg`:
- `snr_mean`, `dc_coherence`: dramatically higher than current (should now exceed IDLE by large margin).
- `event_rate`: high (262K pages accessed and written per LCG period).
- `cepstral` content: broadband (no fixed temporal rhythm), higher entropy than `mem_alloc_touch_pages`.
- CV: expected to drop from 0.617 to near IO-subtype levels once all 4 runs have visible signal.
- The `mem_pointer_chase → mem_stream` confusion: expected to disappear once the signal is visible.
- New potential confusion: `mem_pointer_chase → io_rand_rw` (both produce scattered random writes). This would be a true behavioral overlap, and is acceptable because it reflects genuine mechanism similarity.

#### I. Experimental comparison

Record three conditions:
1. Python original (read-only, current).
2. C, `chase_read_only` (same semantics, C implementation, confirms noise floor).
3. C, `chase_write_lcg` (redesigned, writes LCG state per page).
4. C, `chase_write_dependent` (truly data-dependent addressing).

Compare `snr_mean`, `cepstral_peak_idx_var`, CV, silhouette, confusion matrix. The transition from condition 1/2 to 3/4 should produce a step change in all write-sensitive metrics.

---

### 2.4 `mem_alloc_touch_pages.py`

#### A. Current problem

Phase 1 identified one critical issue: glibc's dynamic `M_MMAP_THRESHOLD` may switch between direct-mmap and arena-reuse modes mid-run. When the allocator switches to arena reuse, `mmap`/`munmap` calls stop, and the kernel-level syscall pattern changes. This introduces run-to-run nondeterminism in the syscall cadence, even though the median CV is already excellent (0.054).

Secondary issue: Python's `bytearray` object carries 57 bytes of header overhead, refcount operations, and the Python GC machinery, even with `gc.disable()`. The bytearray allocation routes through `PyMem_Malloc` then `malloc`, adding indirection.

#### B. Rewrite objective

Produce a deterministic mmap/munmap rhythm that does not route through glibc's allocator. Every allocation in the measurement phase calls `mmap` and every release calls `munmap`. The cadence is fully predictable and not subject to threshold-switching heuristics.

#### C. What to control

**Memory allocation**:
- Direct `mmap(NULL, obj_size, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0)` for each object.
- Direct `munmap(ptr, obj_size)` for each release.
- No routing through `malloc`/`free`. No `brk`. No arena.

**Lazy-memory behavior**:
- Pages remain lazy (CoW from zero page) after `mmap`. This is the intended signal: the first-touch write commits the page.
- Touch loop: write one byte (or one int, or a full cache line) to byte 0 of each page. This is the page-fault trigger and the observable write.

**Page alignment**:
- `mmap` returns page-aligned addresses by definition. No alignment management needed.

**Page granularity**:
- `madvise(MADV_NOHUGEPAGE)` on each mmap'd region immediately after allocation to prevent THP promotion within the object.

**Sleep cadence**:
- `nanosleep(&(struct timespec){0, 20000000}, NULL)` for the 20 ms inter-batch sleep. Avoids the Python `time.sleep` overhead and the Python object creation it implies.

**Batch structure**:
- Allocate all objects in batch, touch all, then munmap all, then sleep. This preserves the intended burst structure.
- Log the exact start time of each batch with `clock_gettime(CLOCK_MONOTONIC)` to a small ring buffer. Use this to confirm cadence consistency.

**Write content**:
- Current: one byte per page (minimal per-page change).
- Proposed improvement: write a 4-byte sequence number to byte 0 of each page. This increases per-page Hamming/cosine delta magnitude and makes each allocation distinct.

**Syscalls**: the only syscalls in the measurement loop are `mmap`, `munmap`, `nanosleep`, and `clock_gettime`. No Python overhead, no GIL, no boxed integers.

#### D. What should remain uncontrolled

The page fault itself: the lazy-allocation-then-first-touch pattern is the core signal. Do not use `MAP_POPULATE` or `memset` before the touch loop. The zero-page-CoW-break event is part of the intended mechanism.

The kernel's buddy allocator and VMA structure management: these are the OS-level correlates of the workload and part of the observable signal.

#### E. Recommended implementation level

**C, using direct `mmap`/`munmap` syscalls. No assembly justified.**

The dominant cost is kernel-mediated page faults and syscall transitions, not user-space computation. Assembly cannot speed these up.

#### F. Suggested variants

| Variant | Allocation path | Touch per page | Release path | Sleep |
|---|---|---|---|---|
| `alloc_mmap_byte` (recommended) | mmap direct | 1 byte | munmap | 20 ms |
| `alloc_mmap_line` | mmap direct | full 64-byte cache line | munmap | 20 ms |
| `alloc_mmap_dontneed` | mmap direct, arena-like | 1 byte | madvise MADV_DONTNEED | 20 ms |
| `alloc_mmap_populate` | mmap + MAP_POPULATE | skip (all pre-faulted) | munmap | 20 ms |
| `alloc_python` (current) | bytearray → malloc | 1 byte | bytearray destructor | 20 ms |

The `alloc_mmap_populate` variant separates the page-fault phase from the touch phase. By pre-faulting all pages with MAP_POPULATE before the touch loop, the touch loop runs without page faults. This is a control for whether the page-fault events or the touch writes drive the observable signal.

The `alloc_mmap_dontneed` variant emulates what glibc does on the arena path. Comparing this to `alloc_mmap_byte` isolates the munmap-vs-MADV_DONTNEED difference.

#### G. Verification instrumentation

- `strace -c -e trace=mmap,munmap,madvise,brk,nanosleep` on the C version. Confirm zero `brk` calls and mmap/munmap count equals 2000 per batch.
- Compare with Python version under `strace -c`. Check whether Python shows `brk` or fewer `mmap` calls (arena path).
- `/proc/$pid/maps` snapshot at mid-run: VMA count should fluctuate by exactly 2000 entries per batch.
- `perf stat -e minor-faults`: confirm exactly 64 faults per object (256 KB / 4 KB pages) or fewer if MADV_HUGEPAGE applies.
- Per-batch timing from the `clock_gettime` log: confirm cadence jitter is under 1 ms across the 300 s run.

#### H. Expected effect on metrics

- CV: expected to drop from 0.054 toward ~0.020. Already the best MEM subtype; deterministic mmap path tightens it further.
- `cep_periodicity_score`: stronger cadence signal when all batches have the same syscall count.
- `snr_fano`: lower (more regular event spacing).
- The `io_rand_rw → mem_alloc_touch_pages` confusion: no direct improvement from this rewrite. That confusion is driven by metric inadequacy in distinguishing page-cache writes from anonymous writes. The rewrite cleans cadence noise, not signal shape.

#### I. Experimental comparison

Record Python original alongside C direct-mmap version with matched parameters (2000 objects, 256 KB each, 20 ms sleep). Compare batch-level cadence timing from side-channel logging, per-recording CV, and segment-level trajectory stability from `segment_level_analysis_critique_and_plan.md`.

---

### 2.5 `io_rand_rw.py`

#### A. Current problem

Phase 1 identified three issues:

1. **Sparse-file extent allocation**: `f.truncate(2GB)` creates a sparse file. Each write to a previously unwritten hole triggers filesystem extent allocation, which involves journal activity beyond the pure data write. This only occurs on first-pass writes; subsequent overwrites skip allocation.

2. **Indeterminate page-cache warmth**: Over 300 s, the 2 GB file may be partially cached. Whether a given random access is a page-cache hit or a disk miss depends on prior access history, which varies across cycles and runs.

3. **Separate lseek + read/write**: `f.seek()` followed by `f.write()` or `f.read()` are two separate syscalls. Between them the file descriptor state can be preempted. This is a negligible Python artifact but a clean-up target.

The `io_rand_rw → mem_alloc_touch_pages` confusion was classified as probable behavioral overlap plus metric inadequacy. The rewrite addresses this by offering an `O_DIRECT` variant that isolates page-cache dynamics from raw I/O.

#### B. Rewrite objective

Produce a random-access I/O workload with:
- Known preallocation state (no sparse-file extent allocation during measurement).
- Explicit page-cache warmth control (cold-cache vs warm-cache variants).
- `pread`/`pwrite` for atomic offset + I/O (no separate lseek).
- An `O_DIRECT`-like variant (via `posix_fadvise POSIX_FADV_DONTNEED` or kernel-level bypass) that removes page-cache participation from the signal.

#### C. What to control

**File preallocation**:
- `fallocate(fd, 0, 0, file_size)` instead of `truncate`. This allocates all disk blocks before measurement begins. The filesystem journal entry for extent allocation happens once, not during the measurement phase.

**Page-cache warmth**:
- Cold-cache variant: call `posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED)` before the measurement loop. This drops the file from page cache. Each access then triggers a disk read (for reads) or a clean page write (for writes to new locations).
- Warm-cache variant: pre-read the entire file sequentially before the measurement loop. All data is in cache. The random-access loop then stresses cache management rather than disk I/O.

**Syscall sequence**:
- `pread(fd, buf, block_size, offset)` and `pwrite(fd, buf, block_size, offset)` instead of lseek + read/write.
- `posix_fadvise(fd, offset, block_size, POSIX_FADV_RANDOM)` at open time.

**Page-cache bypass variant**:
- Open with `O_DIRECT | O_RDWR`. Requires buffer alignment to sector size (512 B or 4096 B). Use `posix_memalign(4096, block_size)`.
- `O_DIRECT` bypasses the page cache entirely. Raw block I/O signal only.
- This produces a fundamentally different signal from the cached version: no page-cache dirty pages, no writeback, pure block I/O timing.

**Write content**:
- Keep the write buffer as a fixed pseudo-random pattern (use an LCG to fill at startup). Same content every run for reproducibility.

**Timing**:
- Remove Python random number generator overhead by using a pure LCG in C for offset selection.
- This is fast enough to not be a bottleneck, but removes any Python-specific timing artifacts.

#### D. What should remain uncontrolled

Page-cache dynamics for the cached variant: this IS the intended signal for the buffered random I/O test. Do not suppress cache hits and misses in the standard variant. Only the `O_DIRECT` variant bypasses them.

Disk latency variability: this is an authentic property of the virtual disk path. Do not try to remove it.

#### E. Recommended implementation level

**C with specific flags: `fallocate`, `pread`/`pwrite`, `posix_fadvise`, and an `O_DIRECT` variant.**

No assembly justified. The workload is kernel-bound (disk or page-cache). User-space loop timing is irrelevant.

#### F. Suggested variants

| Variant | Preallocation | Cache state | Syscall path | Buffer alignment | I/O direction |
|---|---|---|---|---|---|
| `rand_buffered_cold` (recommended default) | fallocate | cold (FADV_DONTNEED) | pread/pwrite | natural | 50/50 |
| `rand_buffered_warm` | fallocate | warm (pre-read) | pread/pwrite | natural | 50/50 |
| `rand_direct` | fallocate | bypassed (O_DIRECT) | pread/pwrite | 4096-aligned | write-only |
| `rand_writeonly` | fallocate | cold | pwrite only | natural | write only |
| `rand_readonly` | fallocate | cold | pread only | natural | read only |
| `io_rand_rw_python` (current) | truncate (sparse) | uncontrolled | seek+read/write | unaligned | 50/50 |

The `rand_writeonly` variant removes the read-half. The `rand_readonly` variant removes the write-half. Comparing these two isolates page-cache-fill (reads) from page-cache-dirty (writes) in the observable delta signal. Since the delta pipeline only observes writes, `rand_readonly` should produce a much weaker signal.

The `rand_direct` variant removes page-cache entirely. Its delta signal comes from whatever the block layer and virtual disk driver write into guest memory (journal, device buffer state). This separates "raw I/O memory footprint" from "page-cache memory footprint."

#### G. Verification instrumentation

- `iostat -x 1`: `r_await`, `w_await`, `%util`. Compare between cold and warm variants.
- `/proc/meminfo` `Cached` before and after pre-read to confirm warm variant actually populates cache.
- `strace -c`: confirm zero `lseek` calls (pread/pwrite do not use lseek).
- `/proc/$pid/io` `read_bytes`, `write_bytes`: compare O_DIRECT vs buffered.
- For O_DIRECT: confirm no `Dirty` pages in `/proc/meminfo` attributable to this process (since writes bypass cache and go directly to disk).

#### H. Expected effect on metrics

- `rand_buffered_cold` vs current Python: similar signal, but more reproducible (fallocate removes per-run extent allocation variability).
- `rand_writeonly` vs current: stronger signal (writes only; all 50% of current iterations that were reads were contributing no delta).
- `rand_direct`: fundamentally different signal. Near-zero page-cache activity. May reveal block-layer device memory signals.
- CV: expected improvement with fallocate + cold cache standardization.
- `io_rand_rw → mem_alloc_touch_pages` confusion: `rand_writeonly` and `rand_direct` variants isolate the signal components and may reveal which component is responsible for the overlap.

#### I. Experimental comparison

Record Python original, C `rand_buffered_cold`, C `rand_writeonly`, and C `rand_direct`. Use identical block size, file size, and seed. Compare per-recording feature distributions, silhouette relative to `mem_alloc_touch_pages`, and the `io_rand_rw → mem_alloc_touch_pages` confusion rate.

---

### 2.6 `io_seq_fsync.py`

#### A. Current problem

Phase 1 identified two structural issues:

1. **Unbounded file growth**: The file opens in `wb` mode (truncate on open), then appends indefinitely. Over 300 s the file may grow tens of GB. As the file grows, new filesystem extents are allocated per write, and the journal records extent allocations as well as data. During early writes, extent allocation occurs per chunk. After the first pass through the disk, subsequent overwrites do not reallocate but may not occur since the file only grows.

2. **fsync scope**: `os.fsync(fd)` calls `fsync(2)`, which flushes data AND metadata. `fdatasync(2)` flushes only data, avoiding metadata journal overhead. The choice between these changes the observable signal.

These are not Python artifacts. They are workload design ambiguities. C fixes them by making the choices explicit.

#### B. Rewrite objective

Produce a sequential write+sync workload with:
- A preallocated fixed-size file. Writes roll over within the file (no unbounded growth).
- Explicit choice between `fsync` and `fdatasync`.
- Predictable journal cadence.

#### C. What to control

**File preallocation**:
- `fallocate(fd, 0, 0, file_size)` with a fixed file size (e.g., 10 GB, large enough to sustain 300 s of writes at max throughput).
- Writes use `pwrite(fd, buf, chunk_size, offset)` with a rolling offset: `offset = (offset + chunk_size) % file_size`. This avoids extent allocation after the initial fill.

**Sync variant**:
- `fsync(fd)`: flush data and metadata. Current behavior.
- `fdatasync(fd)`: flush data only. Removes filesystem metadata journal entry from the observable signal.
- `O_SYNC` flag on open: kernel enforces sync-on-write without a separate syscall. Removes the explicit fsync from the loop.

**Chunk size**:
- Current: 4 MB. Keep this as the default.
- `--fsync-wait 1` means every chunk is synced. This is the maximum sync frequency. Keep as default.

**Write content**:
- Use `os.urandom` equivalent in C via getrandom(2) or `/dev/urandom` once at startup. Fixed random buffer.

**Timing**:
- The loop is sync-latency-dominated. No additional timing control needed.

**Syscalls per iteration** (C version):
- `pwrite(fd, buf, 4MB, offset)` + `fdatasync(fd)` (or `fsync(fd)`).
- Two syscalls per chunk. Deterministic.

#### D. What should remain uncontrolled

Disk latency: this drives the fsync period, which is the core signal. Do not attempt to control it.

The write-then-sync rhythm itself: this is the intended mechanism. Do not pre-flush or asynchronize.

#### E. Recommended implementation level

**C for precision. No assembly justified.**

The signal is kernel-dominated. Assembly cannot help.

#### F. Suggested variants

| Variant | File management | Sync method | Chunk size |
|---|---|---|---|
| `fsync_rolling_fsync` (recommended) | fallocate, rolling write | fsync | 4 MB |
| `fsync_rolling_fdatasync` | fallocate, rolling write | fdatasync | 4 MB |
| `fsync_rolling_osync` | fallocate, rolling write | O_SYNC flag | 4 MB |
| `fsync_growing_fsync` | growing file (current-like) | fsync | 4 MB |
| `fsync_small_chunk` | fallocate, rolling | fdatasync | 64 KB |

The `fsync_small_chunk` variant increases fsync frequency. If the cadence is faster than the snapshot interval, the signal becomes a sequence of closely spaced flushes. Compare cepstral structure with the 4 MB chunk version.

The `fsync_growing_fsync` variant is the Python-equivalent behavior in C. Use this to confirm that the file-growth artifact explains any difference between Python and C results.

#### G. Verification instrumentation

- `iostat -x 1`: write throughput (`wMB/s`). For `rolling_fsync`, steady-state throughput stabilizes earlier than for the growing variant.
- `/proc/diskstats` delta for flush count: confirms fsync reaches block layer.
- `blktrace` on the virtual disk: distinguishes data writes from journal writes per chunk.
- Compare `Dirty` in `/proc/meminfo` during the write phase vs after fsync. Should near-zero after each fsync.

#### H. Expected effect on metrics

- CV: expected marginal improvement with rolling write (eliminating extent allocation variability).
- `cep_periodicity_score`: unchanged (the fsync cadence is already very stable).
- `fdatasync` variant may show a slightly different cepstral shape than `fsync` (no metadata journal entries). This is scientifically interesting.
- No confusion-matrix improvement expected: `io_seq_fsync` is already 4/4 correct.

#### I. Experimental comparison

Record Python original, C rolling-fsync, C rolling-fdatasync. Compare cepstral profiles and journal-commit frequency from `blktrace`. The goal is to understand whether the current signal is primarily data-flush or metadata-flush dominated.

---

### 2.7 `io_many_files.py`

#### A. Current problem

Phase 1 identified one critical undocumented ambiguity: `tempfile.mkdtemp` places the working directory under `/tmp`. Whether `/tmp` is tmpfs or disk-backed depends on the guest configuration and is not recorded anywhere in the current metadata. If `/tmp` is tmpfs, the entire workload is RAM-backed VFS operations with zero disk I/O. If `/tmp` is ext4 on disk, there is genuine disk I/O per file.

The signal interpretation differs fundamentally between these two cases. This is a scientific validity concern, not a performance concern.

Secondary issue: `random.getrandbits(30)` as a filename suffix introduces Python's Mersenne Twister into the critical path, adding Python object overhead per file creation.

#### B. Rewrite objective

Produce a metadata-heavy file workload with:
- Explicit, documented filesystem target (not `/tmp`).
- Two distinct modes: tmpfs-backed and disk-backed, run separately.
- Deterministic filename generation without Python random overhead.
- Known file open/write/close/unlink counts per batch.

#### C. What to control

**Target directory**:
- Remove `tempfile.mkdtemp`. Accept an explicit `--dir` argument.
- Document in the metadata sidecar whether the target is tmpfs or disk.
- Recommended: run one version on a known tmpfs mount (`mount -t tmpfs tmpfs /mnt/tmpfs`) and one on a known disk path.

**Filename generation**:
- Replace `random.getrandbits(30)` with a sequential counter: `f_{batch}_{i}.bin`. Deterministic and faster.
- If filename randomness is intentional (to stress dentry hash distribution), use a LCG in C.

**Syscall sequence per file** (explicit in C):
- `openat(dirfd, filename, O_WRONLY|O_CREAT|O_TRUNC, 0600)`: create with O_TRUNC, relative to a directory fd.
- `write(fd, payload, 1024)`: write payload.
- `close(fd)`: close.
- Later: `unlinkat(dirfd, filename, 0)`: unlink by name.

**Directory fd**:
- Use `openat` with a directory file descriptor rather than constructing full paths. Avoids string concatenation overhead. Not a signal concern, but cleaner.

**Batch granularity**:
- Parameterize batch size and inter-batch behavior (sleep or immediate). Current Python has no sleep between batches.

**Optional fsync per file or per batch**:
- For the disk-backed variant, adding `fdatasync(fd)` per file forces journal commits during measurement. This is a separate variant.
- Without fsync, file data may sit in page cache and writes may be invisible to the delta pipeline until writeback.

**Payload**:
- `os.urandom(1024)` equivalent: `getrandom(payload, 1024, 0)` or a fixed pseudo-random buffer generated at startup. Same payload every file within a run.

#### D. What should remain uncontrolled

Filesystem journal rate: this is the intended mechanism for the disk-backed variant. Do not suppress or pace it.

Dentry and inode cache churn: this is the memory-resident signal for the tmpfs variant. Do not suppress it.

The create-write-close-unlink sequence: this is the workload. Do not merge steps.

#### E. Recommended implementation level

**C for filesystem clarity and target control. No assembly justified.**

#### F. Suggested variants

| Variant | Target FS | fsync per file | Filename scheme | Batch size |
|---|---|---|---|---|
| `manyfiles_tmpfs_nofync` (recommended) | explicit tmpfs | none | sequential | 500 |
| `manyfiles_disk_nofync` | explicit ext4 | none | sequential | 500 |
| `manyfiles_disk_fsync` | explicit ext4 | fdatasync per file | sequential | 500 |
| `manyfiles_tmpfs_large` | explicit tmpfs | none | sequential | 5000 |
| `manyfiles_python` (current) | /tmp (unknown) | none | random suffix | 500 |

The comparison between `manyfiles_tmpfs_nofync` and `manyfiles_disk_nofync` isolates the slab/VFS memory signal (tmpfs) from the disk+journal signal (ext4). This has scientific value because `io_many_files` is currently 4/4 correct. Understanding WHICH mechanism drives the clean signal (VFS slab churn vs disk journal pressure) tells us whether the workload would remain clean on a different filesystem.

The `manyfiles_tmpfs_large` variant (5000 files per batch) stresses the slab allocator harder. If the signal is dentry-cache-dominated, more files per batch should increase the amplitude.

#### G. Verification instrumentation

- `mount | grep tmp` before any run to confirm filesystem type. **This is mandatory and must be added to run metadata.**
- `/proc/slabinfo` `dentry` and `inode_cache` columns: delta per batch to confirm slab churn.
- `/proc/diskstats` during run: confirm zero I/O for tmpfs variant, positive I/O for disk variant.
- `strace -c`: confirm openat/write/close/unlinkat counts match expected per-batch count.
- For disk variant with fsync: `blktrace` to confirm journal commit per file.

#### H. Expected effect on metrics

- `manyfiles_tmpfs` vs `manyfiles_disk`: expected different cepstral structure. tmpfs signal comes from slab (VFS objects in RAM). Disk signal comes from journal pressure plus data writes.
- CV: expected to be similar or better once target FS is explicit and deterministic.
- Sequential filenames vs random-suffix: minimal effect on signal, since dentry hash distribution is one factor among many.
- The current 4/4 clean result should be reproduced by both tmpfs and disk variants, but the feature profiles may differ enough to distinguish them as two separate subtypes if needed.

#### I. Experimental comparison

Record Python original (unknown /tmp), C tmpfs variant, C disk variant. Compare per-recording feature profiles. Determine which filesystem the Python version was actually using. Reinterpret the current confusion matrix entry for `io_many_files` with this knowledge.

---

## 3. Rewrite Priority Ranking

| Priority | Test | Action | Reason | Expected impact |
|---|---|---|---|---|
| 1 | `mem_pointer_chase` | Redesign + rewrite in C | Read-only workload invisible to pipeline; three of four runs near-IDLE | Eliminates the largest source of MEM class scatter |
| 2 | `mem_stream` | Rewrite in C + increase working set to 1 GB | Small active fraction causes MI-driven IDLE confusion; Python loop rate is interpreter-bound | Expected to eliminate `mem_stream → run_idle` confusion |
| 3 | `run_idle` | Protocol change (cache-drop, longer window) | IDLE CV = 0.973; prior-workload contamination dominates the signal | Expected to reduce IDLE scatter and improve MEM/IDLE boundary |
| 4 | `mem_alloc_touch_pages` | Rewrite in C (direct mmap/munmap) | glibc threshold switching introduces nondeterminism in the cadence | Tightens already-good CV from 0.054 toward ~0.020 |
| 5 | `io_many_files` | Rewrite in C + explicit FS target | Undocumented /tmp backing may invalidate the current signal interpretation | Scientifically mandatory for validity claims |
| 6 | `io_rand_rw` | Rewrite in C + add O_DIRECT variant + fallocate | Only IO confusion; page-cache warmth variability contributes to scatter | Expected to separate `io_rand_rw` signal from `mem_alloc_touch_pages` |
| 7 | `io_seq_fsync` | Rewrite in C + rolling file + fdatasync variant | Already cleanest IO subtype (CV 0.092); rewrite adds variant scientific value | Marginal CV improvement; fsync vs fdatasync comparison has thesis value |

---

## 4. C vs Assembly Decision

| Test | C sufficient? | Assembly justified? | Assembly purpose |
|---|---|---|---|
| `run_idle` | not applicable | no | signal is OS background; neither helps |
| `mem_stream` | partially | yes, for NT-store variant | MOVNTI/MOVNTQ bypasses cache to force DRAM writes visible to snapshot pipeline |
| `mem_pointer_chase` | yes for write variant | yes, for data-dependent chain | `LFENCE` + dependent loads prevent compiler/CPU from eliminating the poor-locality pressure |
| `mem_alloc_touch_pages` | yes | no | kernel-bound; assembly cannot speed syscalls |
| `io_rand_rw` | yes | no | kernel-bound |
| `io_seq_fsync` | yes | no | kernel-bound |
| `io_many_files` | yes | no | kernel-bound |

Assembly is justified for exactly two variants. In both cases, the justification is not performance but experimental control:

- `mem_stream` MOVNTI: C has no portable way to issue non-temporal stores without compiler intrinsics. `__builtin_ia32_movnti` is available in GCC but assembly is cleaner and more portable across toolchains.

- `mem_pointer_chase` dependent-load chain: C compilers may legally reorder or hoist loads. An `LFENCE`-serialized dependent-load chain in inline assembly guarantees that each load completes before the next address is computed, preserving the intended poor-locality behavior.

---

## 5. What Remains After Rewriting

These artifacts persist regardless of language choice or implementation quality:

| Artifact | Can C control it? | Can it be measured? | Mitigation |
|---|---|---|---|
| pmemsave pause variance | no | yes (pause histogram) | Capture pipeline change, not workload change |
| Host scheduling (KVM) | no | yes (perf kvm) | CPU pinning on host if possible |
| Hardware prefetcher | partially (MOVNTI for streams) | yes (perf cache-miss events) | MOVNTI variant or `PREFETCH NTA` in assembly |
| THP promotion | yes (MADV_NOHUGEPAGE) | yes (/proc/smaps) | Explicitly disable per-buffer |
| OS kernel timer wakeups | no | yes (perf irq, /proc/interrupts) | Accept as background noise floor |
| Writeback from prior workload | no | yes (/proc/vminfo Dirty) | Protocol change: cache-drop before each workload |
| Virtual disk I/O latency | no | yes (iostat, blktrace) | Accept as part of IO workload signal; document |
| KVM exit/entry overhead | no | yes (perf kvm events) | Accept; affects capture pause timing |
| NUMA remote access | no | yes (numastat) | CPU+memory pinning if NUMA |

The only fully removable artifacts are Python-specific:

- GIL futex traffic: removed by C.
- Python object boxing and refcount churn: removed by C.
- glibc dynamic allocator threshold switching: removed by calling mmap directly.
- Numpy dispatch overhead: removed by C.
- Python random (Mersenne Twister) state from `random.py`: removed by using a fixed LCG in C.

Everything else is OS, hardware, or capture-pipeline level.

---

## 6. Thesis-Safe Wording

**On why rewrites are being considered:**

> The current workload executables are written in Python. Python introduces interpreter overhead, dynamic object allocation, and glibc allocator heuristics that are not part of the intended workload behavior. These contribute runtime artifacts to the observable memory signal. The proposed lower-level implementations are intended to isolate the intended workload mechanism from these uncontrolled runtime effects. They are not motivated by performance or by dissatisfaction with the current results; they are a methodological refinement to improve the signal-to-artifact ratio and to produce variant workloads that can distinguish competing explanations for the observed confusion-matrix patterns.

**On the read-only issue in `mem_pointer_chase`:**

> The current `mem_pointer_chase` implementation traverses memory pages via reading without writing them. The page-content-delta observation pipeline records byte changes between snapshots; read-only operations produce no observable change. The proposed redesign adds a write to each visited page so that the pseudo-random access pattern becomes observable. This is a workload design correction, not a performance change. It does not invalidate the current results; it indicates that the current workload does not produce the intended stimulus for the observation method in use.

**On the C rewrites not controlling the kernel:**

> Rewriting a workload in C does not give the experimenter control over kernel behavior. The kernel's page allocator, writeback daemon, filesystem journal, and hardware prefetcher operate independently of user-space language choice. What C provides is control over how user-space invokes kernel behavior: which syscalls are issued, with which flags, in which sequence, and at which rate. This finer control over the invocation path makes the kernel's response more reproducible and better-characterized, without modifying the kernel itself.

**On the segment-level connection:**

> The proposed C rewrites are designed to produce workloads with separated warm-up and measurement phases, deterministic cadences, and explicit variant structure. These properties make them compatible with the segment-level analysis framework in `segment_level_analysis_critique_and_plan.md`. In particular, a C-rewritten `mem_stream` with explicit warm-up separation allows segment-level analysis to compare early (post-warmup) versus late (steady-state) segments without the warmup confound present in the current Python version.

**On what the rewrites will not resolve:**

> The proposed rewrites address Python-specific artifacts and undocumented workload ambiguities. They do not address pmemsave capture-side artifacts, host scheduling effects, or virtual disk I/O variability. The `io_rand_rw → mem_alloc_touch_pages` confusion, which was classified as probable behavioral overlap at the metric level, is not expected to be resolved by rewrites alone. It requires either a metric that distinguishes page-cache writes from anonymous-mapping writes, or a workload that eliminates the shared signal mechanism (such as the `O_DIRECT` variant for `io_rand_rw`, which removes page-cache participation entirely).
