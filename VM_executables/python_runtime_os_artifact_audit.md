# Python / Runtime / OS / VM Artifact Audit of Workload Executables

This document audits the seven workload executables for hidden runtime, OS, and VM-level effects that may contaminate the intended workload signal. Analysis is grounded in the actual source under `VM_executables/`, the invocation parameters from `VM_sampler/VM_Capture_QEMU/steps_cycle_repetition.txt`, and the current results in `/Users/jeries/Desktop/fulltests/`.

Each per-test section follows the same ten-point template before any cross-test conclusions.

---

## Common Background

### Pipeline observability constraint

Memory snapshots are flat RAW dumps via QEMU `pmemsave`. The downstream Hamming and cosine deltas are computed between consecutive page-aligned dumps. **A page contributes to the delta only when its byte content changes.** Reading pages does not produce a delta. Only writes do. This single fact dominates the artifact analysis.

Source: `docs/QEMU_CAPTURE_PIPELINE.md` (producer = `capture_producer_qemu_pmemsave.sh`).

### Per-snapshot pause artifact

The producer issues `virsh ... suspend`, takes the dump, then resumes. Pause duration is workload-dependent (host memory pressure, SSD speed). Any workload with high syscall rate will be paused mid-syscall on some snapshots. This contributes a temporally-correlated capture noise floor, especially for short-syscall I/O workloads.

### Invocation parameters from `steps_cycle_repetition.txt`

| Workload | Parameters |
|---|---|
| `mem_stream` | `--mb 128 --seconds 300` |
| `mem_pointer_chase` | `--mb 1024 --seconds 300 --seed 123` |
| `mem_alloc_touch_pages` | `--objects 2000 --object-kb 256 --sleep-ms 20 --seconds 300` |
| `io_seq_fsync` | `--seconds 300 --kb 4096 --fsync-wait 1` |
| `io_rand_rw` | `--seconds 300 --file-mb 2048 --block-kb 64 --write-ratio 0.5 --seed 123` |
| `io_many_files` | `--seconds 300 --files-per-batch 500 --payload-bytes 1024 --seed 123` |
| `run_idle` | `--time 60` between workloads, `--time 120` at end |

---

## 1. `run_idle.sh`

### 1.1 Intended clean workload mechanism

A `bash sleep N` to leave the VM passively idle. No synthetic workload.

### 1.2 What Python / runtime does behind the scenes

Bash runs a single `sleep` builtin. Bash internally calls `nanosleep` or similar. No interpreter loop, no allocations beyond shell startup.

### 1.3 What syscalls / kernel mechanisms are likely involved

- `nanosleep(N)` once
- shell startup: `execve`, `read` of bash itself, brk, mmap of libc
- shell teardown: `exit_group`

Almost nothing during the idle interval itself.

### 1.4 Lazy allocation / page faults / cache / scheduler / VM effects

- During the 60s sleep, bash is descheduled. Other processes (`systemd-journald`, `kworker`, `NetworkManager`, `rngd`, `dbus-daemon`, `crond`, kernel timers, RCU callbacks) run.
- Page-cache writeback continues if dirty pages remain from the previous workload. Kernel `pdflush`/`kworker` threads issue writeback over tens of seconds depending on dirty ratio.
- Slab caches (dentry, inode) get reclaimed under pressure.
- After `mem_alloc_touch_pages`, freed mmaps may sit on glibc's tcache or be returned to the kernel asynchronously.
- After `io_seq_fsync` or `io_rand_rw`, the page cache may hold gigabytes of data that the kernel reclaims gradually.
- After `io_many_files`, the dentry cache may still contain entries for unlinked paths.
- VM scheduler (KVM): host may schedule other tenants on the physical CPU.
- pmemsave pause artifacts continue to occur in IDLE snapshots.

### 1.5 Memory-signal artifacts these mechanisms could produce

- Decaying writeback signal: high page activity in early IDLE seconds, fading toward the end.
- Periodic kernel timer activity: kthread wakeups every few hundred ms.
- Slab reclaim bursts: occasional spikes.
- VM-host context-switch artifacts.

The IDLE signal is therefore not a "zero" reference. It is a residual plus background process plus capture artifact mixture. This is consistent with `run_idle` median CV = 0.973 reported in `stochastic_characterization_summary.txt`.

### 1.6 Which metrics could be affected

- `event_rate`, `snr_high_frac`: elevated by writeback bursts.
- `cepstral` periodicity: contaminated by kernel timer cadence.
- `dc_coherence`: pulled up by sustained low-level activity.
- `snr_zero_frac`: lower than expected if page-cache eviction is active.

### 1.7 Which other workload subtype could it accidentally resemble

- `mem_stream` (sparse low-amplitude writes), if writeback rate is similar.
- A weak version of `io_*`, if writeback from a prior IO test dominates.

### 1.8 Would rewriting this in C reduce ambiguity

No. The signal is from the OS, not from the executable. C cannot remove writeback, kernel timers, or VM scheduling.

### 1.9 Would assembly add meaningful control

No. Same reason.

### 1.10 Verification instrumentation

- `cat /proc/meminfo` snapshots every second during IDLE: track `Dirty`, `Writeback`, `MemFree`.
- `vmstat 1`: track `bi`, `bo`, `cs`, `in`.
- `/proc/diskstats` deltas during IDLE: identify ongoing writeback.
- `perf stat -a -e cs,page-faults,minor-faults,major-faults sleep 60`: per-IDLE counters.
- `pmap` of long-lived processes pre/post each cycle.
- `slabtop -o` snapshot at start and end of each IDLE.

---

## 2. `mem_stream.py`

### 2.1 Intended clean workload mechanism

Sequential page-strided writes: one byte per 4096-byte page across a 128 MB buffer, repeated. Touches every page of the buffer once per sweep.

### 2.2 What Python / runtime does behind the scenes

```python
buf = np.zeros(nMB, dtype=np.uint8)
for i in range(0, n, page_stride):
    buf[i] = v
```

- `np.zeros` calls `calloc`. On Linux this returns `MAP_ANONYMOUS | MAP_PRIVATE` virtual pages backed by the kernel zero page (CoW). No physical commitment yet.
- The first sweep performs ~32,768 page faults to commit the buffer.
- The inner loop is interpreted CPython. `range(0, n, page_stride)` produces a `range` object reused per sweep.
- `buf[i] = v` dispatches through numpy `__setitem__`. Each assignment crosses the C-API boundary.
- `int(v)` and `np.uint8(...)` create temporary boxed integer objects per outer iteration.
- `time.time()` checked once per outer sweep, so timing overhead is small relative to inner loop dispatch.
- GIL is held by the single thread.
- pymalloc handles small Python object allocations (boxed ints, intermediate refs); larger allocations (the buffer) bypass pymalloc.

### 2.3 What syscalls / kernel mechanisms are likely involved

Steady state (after first sweep): essentially none. No I/O syscalls. Rare `clock_gettime` from `time.time()`. Possible `futex` from GIL servicing.

### 2.4 Lazy allocation / page faults / cache / scheduler / VM effects

- First sweep: ~32K minor page faults to commit the buffer (`__GFP_ZERO` zero-page CoW break).
- Hardware prefetcher detects sequential pattern and prefetches ahead. Effective DRAM traffic is far below naive expectation.
- L1 cache: each store goes to a new cache line; line is dirtied then evicted. No reuse.
- TLB: 32K pages × 4 KB = 128 MB working set. With 2 MB Transparent Huge Pages, this collapses to 64 entries. Without THP, large TLB miss rate.
- THP promotion: kernel `khugepaged` may promote stretches of the buffer to 2 MB pages over time. This changes cache and TLB pressure mid-run.
- Scheduler: VM is single-tasked at the workload level, but kernel threads still run. RCU, kworker, ksoftirqd may preempt briefly.
- Host: KVM may schedule other tenants. CPU frequency scaling may step down due to memory-bound workload (low IPC).

### 2.5 Memory-signal artifacts

- The 128 MB region produces dirty pages between snapshots; everything else stays static.
- 128 MB out of typical 1-4 GB guest RAM is a small fraction of total pages. Most pages contribute zero delta.
- Per-page write is one byte. Hamming distance per dirty page is small. Cosine distance per dirty page is small.
- Sweep period vs snapshot period: if sweeps are faster than snapshot interval, the entire 128 MB region looks uniformly dirty; if slower, only a subset of pages per snapshot.
- Hardware prefetcher does not affect the **delta signal** because writes still occur. But it does compress the time the workload takes per page, which changes the effective cadence.
- THP promotion mid-run can shift page granularity in the dump. The delta computation operates on 4 KB pages; a 2 MB hugepage's dirty-byte signature may change differently than expected.

### 2.6 Which metrics could be affected

- `snr_mean`, `snr_high_frac`: pulled down by the large fraction of static pages.
- `cepstral_peak_idx`: dominated by the static-page background, peaking at low quefrency.
- `dc_coherence`: low if active fraction is small.
- `event_rate`: depends on whether the metric counts dirty pages within the buffer or normalizes over total pages.
- Spectral slope: shallow because a small contiguous region carries all the signal.

This matches the observed `mem_stream` results: `dc_coh` 0.13-0.48 across runs (`stochastic_characterization_summary.txt`), with the LOWEST values in cycles 3 and 4 (test26, test38).

### 2.7 Which other subtype could `mem_stream` accidentally resemble

- `run_idle`: when the active-page fraction is small enough that aggregated metrics are dominated by static-page background. Confirmed in confusion matrix: `mem_stream → run_idle`.
- `io_rand_rw`: if residual page-cache writeback from the previous cycle's `io_many_files` overlaps with the current `mem_stream` window. Confirmed: `mem_stream → io_rand_rw`.

### 2.8 Would rewriting this in C reduce ambiguity

Yes, partially.

C improvements available:
- Explicit `mlock` to keep buffer resident, removing variable page-fault timing.
- Explicit `madvise(MADV_NOHUGEPAGE)` or `MADV_HUGEPAGE` to fix page granularity for the entire run.
- `MAP_POPULATE` on the initial mmap to pre-fault deterministically.
- `sched_setaffinity` to pin to a specific CPU and avoid migration noise.
- Tighter loop with no GIL, no boxed ints, no numpy dispatch. The actual store rate becomes hardware-limited, not interpreter-limited.

C does NOT remove:
- Hardware prefetcher behavior.
- Capture pause artifact.
- Background OS activity in the guest.
- Cache eviction effects.

### 2.9 Would assembly add meaningful control beyond C

Yes, for specific control:
- Non-temporal stores (`MOVNTI`, `MOVNTQ`) to bypass cache and force DRAM traffic.
- `PREFETCHT0/T1/T2/NTA` to control prefetching explicitly.
- Tight unrolled SIMD loops with deterministic instruction count per page.
- `RDTSC`-based pacing for deterministic per-page timing independent of clock APIs.
- `CLFLUSH` / `CLFLUSHOPT` to evict cache lines and force the next access through memory.

These give "pure memory pressure" stimuli with controlled cache behavior. Worth doing if the goal is to characterize the **observation pipeline's** sensitivity to memory traffic rather than the workload's behavior.

### 2.10 Verification instrumentation

- `perf stat -e cache-misses,cache-references,LLC-loads,LLC-stores,page-faults,minor-faults,dTLB-load-misses` during the workload.
- `/proc/$pid/smaps` snapshots: track `AnonHugePages` to detect THP promotion.
- `/proc/$pid/status` `VmRSS`, `VmSwap`: confirm no swap activity.
- `/sys/kernel/mm/transparent_hugepage/enabled` setting check.
- `numastat` if NUMA: track local vs remote allocations.
- `strace -c` on a short replay run to confirm no surprise syscalls in steady state.

---

## 3. `mem_pointer_chase.py`

### 3.1 Intended clean workload mechanism

Pseudo-random page-strided memory access over a 1 GB buffer using an LCG to compute the next page index.

### 3.2 What Python / runtime does behind the scenes

```python
data = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
i = (a * x + c) % m
acc ^= int(arr[idx])
```

- `rng.integers(...)` allocates 1 GB of pseudo-random uint8. This is an active **write** during initialization, taking measurable seconds.
- The chase loop is CPython interpreted. Each iteration: integer multiply + add + mod, two attribute lookups, numpy item access, `int()` boxing, XOR, assignment.
- `int(arr[idx])` creates a fresh Python `int` object per iteration. The result is in `[0, 255]`, inside the small-int cache, so allocation cost is reduced but not zero.
- The XOR accumulator `acc` stays in the small-int range mostly.
- LCG params: `a=1664525`, `c=1013904223`, `m = n // stride = 262144`. With `m = 2^18`, the LCG has full period 262144 (a-1 divisible by 4, c odd). Same `--seed 123` every cycle, so the access sequence is identical every run.

### 3.3 What syscalls / kernel mechanisms are likely involved

After init: essentially none. The chase is pure user-space reads. Possibly `clock_gettime` checks. GIL futex.

### 3.4 Lazy allocation / page faults / cache / scheduler / VM effects

- 1 GB allocation: `rng.integers(...)` writes random data, so all 262K pages are committed immediately. No lazy faults during the chase.
- Working set 1 GB exceeds typical L1/L2/L3 caches and exceeds typical TLB capacity. TLB miss rate near 100% per access. Page table walks on most accesses.
- Pseudo-random LCG sequence is **deterministic**. The hardware prefetcher cannot learn the pattern.
- Cache: each access reads one byte from a different page. Cache line for that page may or may not be present depending on prior fills. Effective cache is not reused.
- DRAM controller: random-page row activations.
- Scheduler: same as `mem_stream`.

### 3.5 Memory-signal artifacts (CRITICAL)

**The chase loop only READS pages. It does not write.** The memory signal pipeline observes page CONTENT CHANGES via Hamming/cosine on consecutive snapshots. Reading does not change content.

What CAN change between snapshots while this workload runs:
- The XOR accumulator if it spills from a register to the stack. With `acc` being a Python `int` object, there is reference counting churn, but the actual stack frame and Python object header are small.
- Python interpreter state: eval stack, frame objects, refcounts. Bounded set of pages.
- `time.time()` system call buffers.
- Background OS activity unchanged from IDLE.

Implications:
- The "signal" during `mem_pointer_chase` is largely interpreter and kernel background, not the chase itself.
- This is a fundamentally different observable than `mem_stream` (which writes).
- This is the leading hypothesis for why three of four `mem_pointer_chase` runs (test16, test28, test40) collapse to near-IDLE statistics: `snr_mean ≈ 0.05` versus test4 = 2.583. The chase signal is invisible to a delta-based pipeline.

The single test4 outlier may reflect run-specific Python state churn (e.g., extra `time.time()` calls, GC trigger if not disabled, allocator state from prior workload) rather than any chase property.

### 3.6 Which metrics could be affected

- All write-sensitive metrics collapse: `snr_mean`, `dc_coherence`, `event_rate` go toward IDLE values.
- `cepstral_peak_idx` looks idle-like.
- `spectral_slope` flat.

### 3.7 Which other subtype could `mem_pointer_chase` accidentally resemble

- `run_idle`: directly, because the workload is largely invisible to the delta pipeline.
- `mem_stream`: when accumulator-related writes happen to align with the stream signature in the small per-run signal that does exist. Confirmed in confusion matrix: `mem_pointer_chase → mem_stream`.

### 3.8 Would rewriting this in C reduce ambiguity

Only modestly, and **rewriting alone does not fix the read-only issue**.

C improvements:
- Removes interpreter overhead so the loop runs orders of magnitude faster, increasing access count per second.
- Removes integer boxing churn so the only writes left are the accumulator (in a register).
- Eliminates `time.time()` cost.

What C still does NOT change:
- Reads do not produce page deltas.

To make `mem_pointer_chase` actually visible to the delta pipeline, the workload must be **redesigned**, not just rewritten:
- Add a per-access write to a different victim page (still pseudo-random).
- Or use compare-and-swap on the chased page.
- Or write the visited index to a log page.
- Or alternate read-chase and write-chase phases.

Rewriting in C without redesign just makes the invisible workload run faster.

### 3.9 Would assembly add meaningful control beyond C

Yes, for full control:
- Force every access to bypass cache (`MOVNTDQA` for reads, `MOVNTI` for writes).
- Use `LFENCE`/`MFENCE` to serialize loads.
- Construct an explicit dependent-load chain (each next address depends on the current load) to defeat speculation.
- Pace the loop with `RDTSC`.

But again: assembly cannot make a pure-read workload visible to a content-delta pipeline. This is a workload-design issue, not an implementation-language issue.

### 3.10 Verification instrumentation

- `perf stat -e dTLB-load-misses,iTLB-load-misses,LLC-load-misses,instructions,cycles`: confirm DRAM activity.
- `/proc/$pid/io`: verify zero `wchar` and `write_bytes`.
- `vmstat 1`: confirm zero `bi`/`bo`.
- Run with content-changing variant (write the chased index to a sink page) and re-record. Compare delta signal magnitude.
- `time strace -c python3 mem_pointer_chase.py ...` for a 30s run to confirm syscall count.
- Hash the page content of the 1 GB buffer at start and end of run. They should match (read-only). If they differ, the chase is unintentionally writing.

---

## 4. `mem_alloc_touch_pages.py`

### 4.1 Intended clean workload mechanism

Repeated batches of 2000 × 256 KB allocations, with one byte touched per page in each, then released. Sleep 20 ms between batches.

### 4.2 What Python / runtime does behind the scenes

```python
gc.disable()
buffers = []
for _ in range(n_obj):
    curr_obj = bytearray(obj_bytes)
    touch_page(curr_obj, page_size, v)
    buffers.append(curr_obj)
buffers.clear()
del buffers
```

- `bytearray(256*1024)` requests 256 KB. CPython routes large requests through `PyObject_Malloc` which delegates to `malloc` for sizes above the pymalloc threshold (512 bytes). glibc `malloc` for 256 KB typically returns `mmap`'d memory directly because it exceeds `M_MMAP_THRESHOLD` (default 128 KB, can grow dynamically).
- `touch_page` is a Python loop. For a 256 KB object: 64 pages, 64 byte writes.
- `buffers.append` keeps refs alive.
- `buffers.clear()` and `del buffers` drop refs. Refcount reaches zero, `bytearray.__dealloc__` calls `free`, which `munmap`s the region.
- `gc.disable()` is set so generational GC does not run. Reference counting still runs.
- 2000 allocations × 256 KB = 500 MB per batch.
- glibc `M_MMAP_THRESHOLD` may dynamically grow above 256 KB if the program frees back fast enough, switching to arena reuse. This **changes the syscall pattern mid-run**. Behavior is non-deterministic across runs.

### 4.3 What syscalls / kernel mechanisms are likely involved

Per batch (assuming mmap path):
- 2000 × `mmap(NULL, 262144+overhead, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)`
- 2000 × `munmap(...)`
- One `nanosleep(20ms)` per batch.

If glibc switches to arena reuse:
- Far fewer mmap/munmap calls. Allocations come from `brk` or arena memory. Returned memory may be `madvise(MADV_DONTNEED)` rather than unmapped.

### 4.4 Lazy allocation / page faults / cache / scheduler / VM effects

- Each `mmap` is lazy. Pages commit only on first touch.
- `touch_page` writes to byte 0 of each 4 KB page → 64 page faults per object → 128,000 minor page faults per batch.
- `munmap` immediately reclaims pages. Page tables shrink.
- `MADV_DONTNEED` (if arena path) keeps virtual mapping but releases physical pages.
- Kernel slab activity: VMA structures created/destroyed.
- Scheduler: 20 ms sleep deschedules the workload, allowing kworker/journal threads to run.

### 4.5 Memory-signal artifacts

- Each batch produces a strong page-table churn signature: 128K page faults followed by 128K teardowns.
- Page content changes: each touched page has its first byte modified. Between snapshots, many freshly-zeroed pages appear (post-allocation, pre-touch state) and many recently-touched pages disappear.
- **The delta signal is large because both allocation-then-touch (zero-to-data transition) and release (data-to-zero, depending on kernel) leave content changes.**
- The 20 ms sleep gives a clean cadence. Spectral signature has periodic component near 1/(batch_period) Hz.
- Allocator nondeterminism: glibc threshold switching can change syscall frequency mid-run, but overall page-fault rhythm remains.
- `mem_alloc_touch_pages` median CV is 0.054, the lowest of all subtypes. The strong rhythmic structure dominates noise.

### 4.6 Which metrics could be affected

- `cepstral_peak_idx_var` strongly elevated (irregular but bursty).
- `cep_periodicity_score` strong if 20 ms cadence aligns with snapshot timing.
- `dc_coherence` elevated.
- `snr_mean` elevated.
- `spectral_slope` more negative (low-frequency dominant).

### 4.7 Which other subtype could `mem_alloc_touch_pages` accidentally resemble

- `io_rand_rw`: in the reverse direction. Confirmed: `io_rand_rw → mem_alloc_touch_pages`. Both produce page-cache and allocator churn observable as scattered dirty pages.
- Less likely to confuse with `mem_stream` (different rhythm) or `run_idle` (much higher activity).

### 4.8 Would rewriting this in C reduce ambiguity

Yes, in a useful way.

C version could:
- Remove the dynamic glibc `M_MMAP_THRESHOLD` adjustment by calling `mmap` and `munmap` directly. This makes the syscall pattern fully deterministic.
- Allow choosing huge-page-backed mmaps explicitly.
- Allow choosing `MAP_POPULATE` to fault in eagerly without the touch loop.
- Allow `madvise(MADV_DONTNEED)` deliberately to compare reclaim behavior.

This is the workload that benefits most from C, because the rhythmic signal IS the signal, and Python's arena heuristics introduce noise into the cadence.

### 4.9 Would assembly add meaningful control beyond C

Marginal. The work is dominated by syscalls and page faults, which are kernel-resident. Assembly cannot speed them up.

### 4.10 Verification instrumentation

- `perf stat -e minor-faults,major-faults,page-faults` per batch.
- `strace -c -p <pid>` to count `mmap`, `munmap`, `madvise`, `brk`, `nanosleep`.
- `/proc/$pid/maps` snapshots before, mid, and post run to observe VMA churn.
- `/proc/buddyinfo`, `/proc/pagetypeinfo` snapshots to track buddy allocator.
- Glibc env: `MALLOC_MMAP_THRESHOLD_=$((256*1024)) MALLOC_MMAP_MAX_=4096` to lock allocator behavior, then re-record.
- Compare runs with `gc.enable()` to observe whether the absent GC is or is not relevant.

---

## 5. `io_rand_rw.py`

### 5.1 Intended clean workload mechanism

Random offset 64 KB reads and writes against a preallocated 2 GB file, 50% write ratio.

### 5.2 What Python / runtime does behind the scenes

```python
data = os.urandom(64*1024)
with open(path, "r+b", buffering=0) as f:
    while ...:
        off = random.randrange(0, max_offset+1, block_size)
        f.seek(off)
        if random.random() < write_ratio:
            f.write(data)
        else:
            _ = f.read(block_size)
```

- `os.urandom(64K)` reads from `/dev/urandom` once at startup. A non-trivial syscall but only once.
- `random.randrange` and `random.random` use Python's Mersenne Twister, pure user-space, no syscalls.
- `buffering=0` disables Python-side buffering. `f.write(data)` and `f.read(block_size)` go directly to `write` and `read` syscalls.
- Read returns bytes objects each time (`_ = f.read(...)`); the underscore name binding releases the previous object's last ref. Refcounting churn at one per iteration.
- `f.seek` is a `lseek` syscall.

### 5.3 What syscalls / kernel mechanisms are likely involved

Per iteration:
- `lseek(fd, off, SEEK_SET)`
- `write(fd, buf, 64K)` OR `read(fd, buf, 64K)`

Plus very rare `clock_gettime`.

### 5.4 Lazy allocation / page faults / cache / scheduler / VM effects

- Reads: page cache miss likely on cold pages → blocking disk read. Page is loaded to page cache; subsequent read of same page hits cache.
- Writes: dirty the page in page cache. Async writeback by kernel `pdflush`/`kworker` threads.
- File preallocated via `f.truncate(2GB)` produces a sparse file. Writes turn sparse holes into allocated extents (filesystem allocator is involved).
- 2 GB file vs typical guest RAM (1-4 GB): page cache may fit a substantial fraction. Cache hit rate depends on access spread.
- Random access pattern has poor cache reuse but bounded working set means some hits.
- Filesystem journal (ext4 ordered): metadata-only journaling. Data blocks bypass journal but extent-allocation events go through it.
- VM/host: writes propagate to host page cache, then to virtual disk image.

### 5.5 Memory-signal artifacts

- Page cache churn produces dirty pages randomly across the cache region. Snapshot-to-snapshot deltas show many random pages with content change.
- Writeback bursts produce additional page changes when pdflush flushes to the virtual disk.
- The pattern is irregular but high-volume: "random scattered dirty pages" signature.
- This visually resembles random writes from `mem_alloc_touch_pages` because both produce scattered random dirty pages: hence the observed `io_rand_rw → mem_alloc_touch_pages` confusion.

### 5.6 Which metrics could be affected

- `event_rate` high.
- `snr_skewness` near zero (broad distribution).
- `cepstral` content broadband, less peaked than fsync.
- Higher `spectral entropy` than sequential I/O.
- `Fano factor` high (bursty writeback).

### 5.7 Which other subtype could `io_rand_rw` accidentally resemble

- `mem_alloc_touch_pages` (confirmed).
- Possibly `run_idle` if the workload mostly hits page cache (no real I/O). Unlikely at 2 GB file, 64 KB blocks, 300 s.

### 5.8 Would rewriting this in C reduce ambiguity

Modestly. Python overhead per syscall is small relative to the syscall plus disk latency. Where C helps:
- `O_DIRECT` to bypass the page cache entirely. This produces a much cleaner "raw I/O" signature without page-cache churn.
- `pread`/`pwrite` to avoid the separate `lseek` syscall.
- `posix_fadvise(POSIX_FADV_RANDOM)` to hint random access pattern.

C does NOT remove:
- Filesystem allocator behavior on first write to a hole.
- Writeback timing.
- Disk latency variability.

### 5.9 Would assembly add meaningful control beyond C

No. The bottleneck is in the kernel and on disk, not in user code.

### 5.10 Verification instrumentation

- `iostat -x 1` during the workload: track `r_await`, `w_await`, `%util`.
- `/proc/diskstats` deltas.
- `/proc/$pid/io` per-process read/write byte counts.
- `vmstat 1`: track `bi`/`bo`.
- Compare runs with `O_DIRECT` (custom variant) versus current.
- Drop caches before run (`echo 3 > /proc/sys/vm/drop_caches`) and compare.

---

## 6. `io_seq_fsync.py`

### 6.1 Intended clean workload mechanism

Sequential 4 MB chunked writes followed by `fsync` after every chunk (`fsync-wait=1`).

### 6.2 What Python / runtime does behind the scenes

```python
data = os.urandom(4*1024*1024)
with open(p, "wb", buffering=0) as f:
    while ...:
        f.write(data)
        if chunks % K == 0:
            os.fsync(f.fileno())
```

- `os.urandom(4MB)` once at startup.
- `f.write(D)` directly maps to `write` syscall.
- `os.fsync(fd)` directly maps to `fsync` syscall, which blocks until data is durable.
- `wb` mode TRUNCATES the file on open. After that, sequential writes APPEND.
- Python overhead per iteration is negligible compared to fsync latency.

### 6.3 What syscalls / kernel mechanisms are likely involved

Per chunk:
- `write(fd, 4MB)` → 4 MB to page cache, returns quickly.
- `fsync(fd)` → blocks. Kernel issues writeback I/O, waits for completion. Filesystem journal commits.

Filesystem journal (ext4 ordered mode default):
- Data blocks first, then journal commit, then metadata.
- Each `fsync` typically forces a journal transaction commit.

### 6.4 Lazy allocation / page faults / cache / scheduler / VM effects

- File grows by 4 MB per chunk. Filesystem allocator allocates extents.
- Page cache fills with the new dirty data, then flushed by fsync. Page state oscillates: clean → dirty → clean per chunk.
- Disk write pattern: very regular sequential writes punctuated by journal commits.
- Block-layer flushes (`REQ_OP_FLUSH`) force device cache to disk. On virtual disks this passes to the host.
- Host-side: virtual disk I/O. If host has write barriers configured, the host disk also flushes per fsync.

### 6.5 Memory-signal artifacts

- The page cache region used for the workload oscillates regularly. Snapshot deltas show large blocks of pages flipping between "data present" and "data flushed".
- The cadence is dominated by fsync latency (depends on disk), not by Python loop speed.
- Strong periodic pattern: write → fsync → write → fsync. This produces a clear cepstral peak at the fsync period.
- `io_seq_fsync` median CV is 0.092. Very stable.

### 6.6 Which metrics could be affected

- `cepstral` periodicity score elevated.
- `spectral_slope` strongly negative (low-frequency dominated).
- `dc_coherence` elevated.
- `snr_fano` low (regular spacing).

### 6.7 Which other subtype could `io_seq_fsync` accidentally resemble

- `mem_alloc_touch_pages` (allocate-touch-release rhythm has similar regular structure). Not observed in the confusion matrix at current sample size, but plausible at higher n.
- Less likely to confuse with random workloads.

### 6.8 Would rewriting this in C reduce ambiguity

Marginally. The signal IS the kernel cadence. Python overhead is negligible compared to fsync latency. C could:
- Use `fdatasync` instead of `fsync` to avoid metadata-only sync overhead and isolate data-flush behavior.
- Use `O_SYNC` open flag and skip explicit fsync calls.
- Use `pwrite` for clarity.

These are stylistic. The workload is already kernel-dominated.

### 6.9 Would assembly add meaningful control beyond C

No. Kernel-bound.

### 6.10 Verification instrumentation

- `iostat -x 1` to confirm sustained sequential write rate.
- `/proc/$pid/io` byte counters.
- `blktrace` to capture block-layer events including journal commits.
- `dmesg` for any I/O timeouts or errors.
- Compare `fsync-wait=1` versus `fsync-wait=8` to confirm the cadence dependency.

---

## 7. `io_many_files.py`

### 7.1 Intended clean workload mechanism

Per batch: create 500 files, write 1 KB to each, unlink them. Repeat for 300 s.

### 7.2 What Python / runtime does behind the scenes

```python
payload = os.urandom(1024)
base = tempfile.mkdtemp(prefix="io_many_")
while ...:
    for i in range(500):
        suffix = random.getrandbits(30)
        f_path = ...
        with open(f_path, "wb") as f:
            f.write(payload)
        batch_files.append(f_path)
    if not keep:
        for fp in batch_files:
            os.unlink(fp)
```

- `tempfile.mkdtemp` once. Creates a directory under `/tmp`.
- `os.urandom(1024)` once.
- Per file: `open` (O_WRONLY|O_CREAT|O_TRUNC), `write` (1 KB), `close`, `unlink` later.
- Python overhead per file is small relative to the syscall and journal cost. The workload is kernel-bound.

### 7.3 What syscalls / kernel mechanisms are likely involved

Per file:
- `openat(..., O_WRONLY|O_CREAT|O_TRUNC, 0666)` → inode allocation, dentry creation, journal entry.
- `write(fd, payload, 1024)` → 1 KB to page cache.
- `close(fd)` → flush metadata to journal in some FS modes.
- `unlink(path)` → dentry removal, inode freed, journal entry.

### 7.4 Lazy allocation / page faults / cache / scheduler / VM effects

- Filesystem journal pressure: hundreds of journal entries per second.
- Inode and dentry caches churn heavily. Slab allocator pressure on `inode_cache`, `dentry`, `ext4_inode_cache`.
- Page cache: 1 KB writes round up to one page each. 500 pages dirtied per batch.
- Writeback by kernel of dirty inode metadata.
- `unlink` of a file with refcount 0 actually frees the inode and the data extents.
- `/tmp` may be tmpfs. If so, all file activity is RAM-backed and reads/writes are pure memory operations. This dramatically changes the signal.
- Scheduler: tight syscall loop, frequent kernel transitions.

`/tmp` filesystem must be checked. If `/tmp` is tmpfs in the guest, the entire workload is in-RAM filesystem operations. If `/tmp` is on the virtual disk, real I/O occurs.

### 7.5 Memory-signal artifacts

- Per batch: heavy slab cache churn, page cache churn, journal block writes.
- Many small transient page-content changes from inode and dentry allocations.
- Strong rhythmic structure at batch period.
- `io_many_files` median CV is 0.048. Very stable signal.

### 7.6 Which metrics could be affected

- `cepstral_peak_idx_var` elevated (high event variance).
- `event_rate` very high.
- `snr_active_frac` and `active_page_fraction` elevated.
- `cep_entropy` is the top discriminator for IO subtypes (`stochastic_characterization_summary.txt`, sep=702.066).

### 7.7 Which other subtype could `io_many_files` accidentally resemble

- Not currently confused with anything (4/4 in preliminary sample).
- Possible distant resemblance to `mem_alloc_touch_pages` because both produce many small allocations and frees. Distinguishing factor: fs journal overhead in `io_many_files` produces a different cepstral structure.

### 7.8 Would rewriting this in C reduce ambiguity

Modestly. The workload is kernel-bound. Useful C variants:
- `O_TMPFILE` to create unlinked files without dentries (changes the signal).
- Direct `creat/write/close/unlink` without Python's `with` context manager overhead. Negligible.

### 7.9 Would assembly add meaningful control beyond C

No.

### 7.10 Verification instrumentation

- `mount | grep tmp` to confirm whether `/tmp` is tmpfs or backed by disk. This is **critical** and directly affects what the workload measures.
- `/proc/slabinfo` deltas for `dentry`, `inode_cache`, `ext4_inode_cache`.
- `iostat`, `/proc/diskstats` to confirm whether disk I/O occurs.
- `strace -c` per batch.
- `perf stat -e syscalls:sys_enter_openat,syscalls:sys_enter_unlinkat` if available.

---

## 8. Cross-Test Comparison and Artifact Risk Ranking

### 8.1 Read-vs-write asymmetry summary

| Test | Writes pages | Reads pages | Allocates pages | Visible to delta pipeline |
|---|---|---|---|---|
| `run_idle` | only via background OS | indirect | no | weak, residual |
| `mem_stream` | yes (sparse, 1 byte/page) | no | once at start | yes but small per-page change |
| `mem_pointer_chase` | no (read-only chase) | yes | once at start | minimal |
| `mem_alloc_touch_pages` | yes (1 byte/page per object) | minimal | continuously | yes, strong rhythm |
| `io_rand_rw` | yes (write-half) | yes (read-half) | filesystem-mediated | yes, broadband |
| `io_seq_fsync` | yes (sequential) | no | filesystem-mediated | yes, strong periodic |
| `io_many_files` | yes (small payloads) | no | filesystem-mediated | yes, strong rhythm |

### 8.2 Artifact risk ranking

Ordered from highest hidden-artifact contamination to lowest.

| Rank | Test | Dominant artifact source | Why it ranks here |
|---|---|---|---|
| 1 | `run_idle` | OS background and prior-test residue | Workload IS background; signal is everything other than the script |
| 2 | `mem_pointer_chase` | Python interpreter and kernel background | Read-only workload nearly invisible; signal is interpreter and noise |
| 3 | `mem_stream` | Hardware prefetcher, THP promotion, allocator zero-page CoW | Small footprint, prefetcher-friendly, page granularity may shift mid-run |
| 4 | `mem_alloc_touch_pages` | glibc `M_MMAP_THRESHOLD` dynamic adjustment, kworker timing | Strong intended signal, but allocator heuristics introduce noise into cadence |
| 5 | `io_rand_rw` | Page cache hit rate variability, writeback timing | Strong intended signal; cache hit rate varies across runs |
| 6 | `io_many_files` | tmpfs-vs-disk dependency, dentry/inode cache state | Strong rhythmic signal; signal definition shifts with `/tmp` backing |
| 7 | `io_seq_fsync` | Disk latency variability, host fsync barrier behavior | Strongly kernel-dominated, but kernel makes the cadence highly regular |

The IO tests rank lowest because their signals are **kernel-mediated rhythms**. The kernel imposes its own structure that is largely insensitive to Python overhead. The MEM tests rank higher because their intended signals are **user-space memory access patterns** that interact with hardware features (prefetcher, TLB, cache, THP, allocator) in nondeterministic ways. The IDLE test ranks first because the script does nothing on purpose; everything observed is artifact by definition.

### 8.3 What C/assembly would and would not help

C would meaningfully improve: `mem_stream` (pinning, mlock, MAP_POPULATE, THP control), `mem_alloc_touch_pages` (deterministic mmap path), and partially `mem_pointer_chase` (only after redesigning to include writes).

C would marginally help: `io_rand_rw` (O_DIRECT for cleaner signal), `io_seq_fsync` (style only), `io_many_files` (style only).

C cannot help: `run_idle` (no script-level signal to clean up).

Assembly would meaningfully help only if the goal is to characterize the **observation pipeline's** sensitivity to specific memory-traffic patterns under controlled cache and prefetch conditions. For the workload-classification thesis, assembly is not needed; what is needed is a workload **redesign** for the read-only case and tighter **C-level controls** for the small-footprint write case.

### 8.4 What none of these address: the capture-side artifact

QEMU `pmemsave` pause variance, host page-cache behavior on the dump file, and snapshot interval jitter are NOT controlled by any change to the workloads. They affect every test equally in form but possibly unequally in magnitude (a high-syscall workload like `io_seq_fsync` is more often paused mid-syscall than `mem_pointer_chase`). This is the capture-artifact axis of the four-cause model in `confusion_matrix_diagnostic_methodology.md` Section 5 and must be addressed in the capture pipeline, not in the workloads.

---

## 9. Verification Test Plan Summary

A practical sequence of measurements that would test this audit's hypotheses without requiring code changes:

1. **Drop caches and instrument I/O** before each workload to standardize starting page-cache state. Compare runs.
2. **Disable THP guest-side** (`echo never > /sys/kernel/mm/transparent_hugepage/enabled`) and re-record `mem_stream`. Compare to current results.
3. **Lock glibc allocator threshold** with `MALLOC_MMAP_THRESHOLD_=131072 MALLOC_MMAP_MAX_=4096` and re-record `mem_alloc_touch_pages`. Compare CV.
4. **Confirm `/tmp` filesystem type**. Re-record `io_many_files` on a known disk-backed path versus tmpfs.
5. **Add a write-chase variant** of `mem_pointer_chase` that writes the visited index to a sink page. Re-record. Compare to original.
6. **Pin all workloads to a single CPU** with `taskset -c N`. Re-record full cycle. Compare CV per subtype.
7. **strace -c** each workload for 30 s. Confirm syscall histograms match the expectations in this audit.
8. **Per-second `/proc/meminfo` and `/proc/vmstat`** capture during each workload. Build per-workload syscall and page-fault profiles.
9. **`perf stat`** with cache, TLB, and faults counters per workload, recorded once per cycle.
10. **Snapshot pause histogram** from QEMU. Cross-correlate pause spikes with delta-frame outliers.

Each of these directly tests one or more hypotheses in this document. None require modifying the workload sources.
