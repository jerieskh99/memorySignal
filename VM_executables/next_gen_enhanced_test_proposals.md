# Next-Generation Enhanced Test Proposals

This document proposes specific enhanced workload tests and methodology probes that resolve the gaps identified in `next_gen_workload_test_design_review.md`. The proposals are not additional members of the existing taxonomy. They are targeted experimental designs answering specific scientific questions left open by the filled spec.

Each proposal isolates one mechanism. Each proposal has a falsifiable expected outcome. None are toy workloads designed for fake separability.

The design review reduced the 25-test inventory to a realistic 10 to 13 implementable tests. This document proposes the **enhanced and replacement designs** for those tests, plus a small set of **methodology probes** that the design review identified as missing.

---

## 1. Document Organization

The proposals are grouped into four categories:

1. **Methodology probes**: targeted experiments that validate the assumptions the filled spec adopts without empirical anchoring. Implementation-priority HIGH.
2. **Enhanced replacement tests**: redesigns of existing or proposed tests that the design review flagged as broken or duplicative.
3. **New experimental tests**: tests that fill genuine gaps not covered by the filled spec.
4. **Family viability probes**: minimal tests that determine whether THREAD and NETWORK families are worth implementing at all.

For each test, the proposal includes the 17 required fields. Brevity is preferred; details are anchored to specific Phase 1 evidence.

---

## 2. Methodology Probes (HIGH PRIORITY)

These six probes resolve assumptions the filled spec relies on. They should be executed before committing to any speculative test family.

---

### 2.1 `mp_capture_pause_calibration`

**Behavior family:** Methodology

**Experimental question:** What is the per-snapshot noise floor introduced by pmemsave pauses, and how does it vary with snapshot interval and host load?

**Behavioral mechanism isolated:** The capture pipeline itself, with no synthetic workload. The only signal source is the OS background and the snapshot pause artifact.

**Why current tests do not isolate this mechanism:** All current tests run a workload during capture, mixing workload signal with pause artifact. No control isolates the pause artifact alone.

**Confusion/artifact problem it resolves:** The "VM/capture artifact" leg of the four-cause model in `confusion_matrix_diagnostic_methodology.md` Section 5. Currently asserted but never measured.

**Low-level control technique:** Run a no-op `sleep 600` workload while pmemsave snapshots are taken at varying intervals (1 s, 2 s, 5 s, 10 s).

**Required implementation level:** Python acceptable (bash sleep + capture-pipeline configuration only).

**OS/kernel controls required:** Cache drop before run. No background workload allowed during run. Document whether host has other tenants.

**Cache/TLB/hardware-aware controls required:** None.

**Filesystem/page-cache controls required:** Cache drop at start.

**Runtime controls required:** None.

**Expected memory-signal effect:**
- Per-snapshot delta should be near-zero plus pause-correlated noise.
- Hamming/cosine deltas should correlate with snapshot interval (longer intervals collect more background activity).
- Event rate near zero.
- Noise floor characterization for SNR metrics.
- Pause histogram from QEMU monitor logs is the primary output.

**Expected confusions:** None expected. The result is a noise-floor measurement, not a class.

**Realism justification:** The capture pipeline runs in every recording. Characterizing its noise floor is a precondition for interpreting any signal.

**Variants:**
- Snapshot intervals 1, 2, 5, 10 s.
- Variants with the host under load and idle.

**Implementation priority:** **VERY HIGH**. Must execute before any new test recording.

---

### 2.2 `mp_cache_dram_observability`

**Behavior family:** Methodology

**Experimental question:** Does cache-resident dirty data appear in pmemsave output, or does it require cache flush to be observable?

**Behavioral mechanism isolated:** The relationship between CPU cache state and physical memory state at snapshot time.

**Why current tests do not isolate this mechanism:** No existing test compares CLFLUSH versus no-CLFLUSH on identical workload. The filled spec acknowledges the question for `cache_hot_loop` but does not propose the experiment.

**Confusion/artifact problem it resolves:** Whether the cache-bypass MOVNTI variant of `mem_stream` is necessary, and whether the small-footprint cache-hot tests are observable at all.

**Low-level control technique:** Tight C loop dirties pages in a 32 KB region (L1-resident). One variant uses CLFLUSH after each write. The other does not.

**Required implementation level:** **C strongly recommended. Assembly justified for CLFLUSH if compiler intrinsic is not available.**

**OS/kernel controls required:** CPU pinning. Disable C-state transitions if possible.

**Cache/TLB/hardware-aware controls required:** Buffer 32 KB to fit in L1. CLFLUSH variant uses `__builtin_ia32_clflush` after each write. No-CLFLUSH variant relies on natural eviction.

**Filesystem/page-cache controls required:** None (anonymous mmap).

**Runtime controls required:** No allocations during measurement.

**Expected memory-signal effect:**
- No-CLFLUSH variant: per-snapshot delta may be much smaller than expected because dirty data sits in L1.
- CLFLUSH variant: per-snapshot delta reflects actual write activity.
- Difference between the two quantifies the cache-hidden signal.
- If the difference is large, all cache-resident workloads need explicit cache management.
- If the difference is small, the snapshot pause itself flushes the cache (or the eviction policy keeps cache and DRAM consistent enough at typical snapshot intervals).

**Expected confusions:** None.

**Realism justification:** Directly tests an observability assumption used implicitly in every workload design.

**Variants:**
- L1-sized buffer (32 KB) and L2-sized buffer (512 KB).
- Read-modify-write versus write-only access.

**Implementation priority:** **VERY HIGH**. Determines whether MOVNTI variants and CACHE family are worthwhile.

---

### 2.3 `mp_thp_promotion_timing`

**Behavior family:** Methodology

**Experimental question:** Does Transparent Huge Page promotion occur during a 300 s workload window, and does it change observable per-4-KB-page deltas?

**Behavioral mechanism isolated:** Kernel `khugepaged` activity and its effect on the delta-pipeline page granularity.

**Why current tests do not isolate this mechanism:** The filled spec recommends `MADV_NOHUGEPAGE` for `mem_stream` and `mem_pointer_chase` redesign, but no experiment shows whether this matters.

**Confusion/artifact problem it resolves:** Whether THP-related artifact accounts for any portion of `mem_stream` cycle drift.

**Low-level control technique:** Run `mem_stream` (1 GB) with three explicit settings: `MADV_HUGEPAGE`, `MADV_NOHUGEPAGE`, default. Snapshot `/proc/$pid/smaps` `AnonHugePages` at fixed intervals to track promotion.

**Required implementation level:** **C strongly recommended.**

**OS/kernel controls required:** `madvise` calls per variant. CPU pinning.

**Cache/TLB/hardware-aware controls required:** TLB flush observable via `perf stat -e dTLB-load-misses` per variant.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** No allocations during measurement.

**Expected memory-signal effect:**
- `MADV_HUGEPAGE` variant: faster TLB, possibly different per-4-KB-page delta because writes within a 2 MB region may not align to 4 KB granularity.
- `MADV_NOHUGEPAGE` variant: deterministic 4 KB granularity throughout.
- Default variant: may show mid-run promotion as `AnonHugePages` increases.

**Expected confusions:** Default variant may show mid-run drift toward `MADV_HUGEPAGE` profile.

**Realism justification:** Directly tests the THP control assumption used in MEM and CACHE rewrites.

**Variants:** As listed.

**Implementation priority:** **HIGH**. Validates one of three competing hypotheses for `mem_stream` cycle drift.

---

### 2.4 `mp_glibc_allocator_control`

**Behavior family:** Methodology

**Experimental question:** Does glibc's dynamic `M_MMAP_THRESHOLD` switching contribute to `mem_alloc_touch_pages` cadence variability?

**Behavioral mechanism isolated:** glibc allocator threshold heuristics.

**Why current tests do not isolate this mechanism:** Phase 1 audit identified the threshold-switching hypothesis. The filled spec proposes bypassing glibc with direct mmap but does not measure the effect.

**Confusion/artifact problem it resolves:** Whether the C rewrite's expected CV improvement for `mem_alloc_touch_pages` is genuinely needed.

**Low-level control technique:** Run `mem_alloc_touch_pages` (current Python form) under three glibc environment configurations: default, locked at 128 KB threshold, locked above 256 KB threshold.

**Required implementation level:** Python acceptable (Python source unmodified, glibc env var control).

**OS/kernel controls required:** Set `MALLOC_MMAP_THRESHOLD_` and `MALLOC_MMAP_MAX_` env vars.

**Cache/TLB/hardware-aware controls required:** None.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** Document allocator state via `strace -c` per variant.

**Expected memory-signal effect:**
- "Force mmap" variant: deterministic mmap/munmap rate per batch. Cleanest signal.
- "Force arena" variant: fewer mmap calls, more `madvise(MADV_DONTNEED)` or arena recycling.
- Default variant: some mix, possibly shifting mid-run.

**Expected confusions:** None.

**Realism justification:** Quantifies the cost of NOT controlling glibc behavior. If the difference is small, the C rewrite's value is marginal.

**Variants:** As listed.

**Implementation priority:** **HIGH**. Cheap to run (env-var configuration only).

---

### 2.5 `mp_cycle_drift_probe`

**Behavior family:** Methodology

**Experimental question:** Does running the same workload eight times consecutively produce signal drift across cycles?

**Behavioral mechanism isolated:** Cycle-position effects independent of execution order.

**Why current tests do not isolate this mechanism:** Phase 1 evidence shows `mem_stream` `dc_coh` decays from cycle 1 (0.483) to cycle 4 (0.133), but the cycle structure interleaves all workload types. The drift could be (a) accumulated state from many prior workloads, (b) thermal/host effect, or (c) cycle-position-dependent of the workload itself.

**Confusion/artifact problem it resolves:** Separates execution-order contamination from intrinsic drift.

**Low-level control technique:** Run `mem_stream` eight times in a row with one 60 s idle between each, no other workloads. Repeat with `mem_alloc_touch_pages` and `io_seq_fsync` separately.

**Required implementation level:** Python or C (whichever the workload is implemented in). Protocol change only.

**OS/kernel controls required:** Standard.

**Cache/TLB/hardware-aware controls required:** Standard.

**Filesystem/page-cache controls required:** Cache drop before each cycle.

**Runtime controls required:** Standard.

**Expected memory-signal effect:**
- If signal is stable across all 8 runs, there is no intrinsic cycle drift.
- If signal drifts (e.g. `dc_coh` declines), the drift is intrinsic to the workload and not caused by other workloads.

**Expected confusions:** None.

**Realism justification:** Direct test of the cycle-drift hypothesis seen in the current `mem_stream` data.

**Variants:** Per workload (`mem_stream`, `mem_alloc_touch_pages`, `io_seq_fsync`).

**Implementation priority:** **HIGH**. Cheap to add to the next batch.

---

### 2.6 `mp_reverse_cycle_probe`

**Behavior family:** Methodology

**Experimental question:** Does the cycle position correlate with prior-workload identity, or with absolute cycle index?

**Behavioral mechanism isolated:** Execution-order contamination as a function of position in the cycle.

**Why current tests do not isolate this mechanism:** All current data uses one cycle order. The relationship between IDLE state and prior workload is confounded with cycle position.

**Confusion/artifact problem it resolves:** Distinguishes EO contamination (which is order-dependent) from cycle-drift (which is position-dependent).

**Low-level control technique:** Run the existing `steps_cycle_repetition.txt` workload sequence, then immediately run it again with the steps in reverse order.

**Required implementation level:** Python acceptable (steps file change only).

**OS/kernel controls required:** Standard.

**Cache/TLB/hardware-aware controls required:** Standard.

**Filesystem/page-cache controls required:** Standard.

**Runtime controls required:** Standard.

**Expected memory-signal effect:**
- If `mem_stream → io_rand_rw` confusion appears in both forward and reverse cycles, the confusion is not order-dependent and EO is not the dominant cause.
- If the confusion appears only in forward order (where `mem_stream` follows IO-heavy workloads), EO is the dominant cause.

**Expected confusions:** Direct test of `mem_stream → io_rand_rw` causation hypothesis.

**Realism justification:** Cheapest test of an EO hypothesis.

**Variants:** Forward and reverse cycle order.

**Implementation priority:** **HIGH**.

---

## 3. Enhanced Replacement Tests

These are redesigns of existing or proposed tests that the design review flagged.

---

### 3.1 `idle_long_baseline_v2`

**Behavior family:** IDLE

**Experimental question:** What is the cleanest reproducible IDLE signature achievable on this VM and host?

**Behavioral mechanism isolated:** Background OS, kernel timers, capture-pipeline noise floor, after all residue from prior workloads has decayed.

**Why current tests do not isolate this mechanism:** `run_idle` is bracketed by active workloads with only 60 s idle. Residue dominates.

**Confusion/artifact problem it resolves:** Provides the controlled baseline that the Validation Strategy in the filled spec depends on.

**Low-level control technique:** 600 s idle window after a 90 s pre-window settle. Cache drop before pre-window. `/proc/meminfo` checks confirm `Dirty == 0` and `Writeback == 0` before measurement starts.

**Required implementation level:** Python acceptable (bash sleep + protocol).

**OS/kernel controls required:** No active workload within 5 minutes prior. No background scheduled tasks (verify `systemd-timers` quiet during the window).

**Cache/TLB/hardware-aware controls required:** None.

**Filesystem/page-cache controls required:** Cache drop. Verify drained.

**Runtime controls required:** None.

**Expected memory-signal effect:**
- Per-snapshot delta should approach the noise floor measured by `mp_capture_pause_calibration`.
- Event rate very low.
- CV across runs much lower than current `run_idle`.
- Segment-level analysis should show stationarity throughout the window.

**Expected confusions:**
- `idle_long_baseline_v2 → run_idle` (legacy)
  - Type: behavioral proximity
  - Reason: same mechanism (idle), differ only in residue.
- `idle_long_baseline_v2 → cpu_invisible_canonical`
  - Type: metric inadequacy
  - Reason: both produce minimal observable delta.

**Realism justification:** This is the experimentally controllable best case for IDLE. The system cannot be quieter without modifying the guest OS.

**Variants:**
- After fresh boot.
- After 5+ minutes settle following last active workload.

**Implementation priority:** **VERY HIGH**. First test to implement.

---

### 3.2 `idle_residue_decay_per_workload`

**Behavior family:** IDLE

**Experimental question:** How does residue from each prior workload decay over a 600 s idle?

**Behavioral mechanism isolated:** Per-workload writeback, slab reclaim, freed-mmap reclaim, dentry-cache reclaim curves.

**Why current tests do not isolate this mechanism:** Filled spec proposes `idle_post_workload_recovery` but does not specify per-prior-workload variants. The signal IS the prior-workload identity.

**Confusion/artifact problem it resolves:** Quantifies the EO contamination time constant. Determines minimum idle duration to use between active workloads.

**Low-level control technique:** Run one fixed prior workload for 300 s. Then 600 s idle while capturing. Capture per-segment metrics for decay analysis. Repeat once for each prior workload type.

**Required implementation level:** Python acceptable.

**OS/kernel controls required:** Same as `idle_long_baseline_v2` for the post-residue portion.

**Cache/TLB/hardware-aware controls required:** None.

**Filesystem/page-cache controls required:** No cache drop (residue IS the signal).

**Runtime controls required:** None.

**Expected memory-signal effect:**
- Decreasing event rate from segment 1 to segment k.
- Decay shape varies by prior workload: writeback-dominated workloads decay slowly, allocator-dominated decay quickly, idle-only is flat.
- Segment-level slope is the diagnostic feature.

**Expected confusions:**
- `idle_residue_decay_per_workload(post-X) → X`
  - Type: execution-order contamination
  - Reason: by design.

**Realism justification:** Directly measures residue decay timescales. Informs idle-duration policy.

**Variants:**
- One per prior workload: `mem_stream`, `mem_pointer_chase`, `mem_alloc_touch_pages`, `io_rand_rw`, `io_seq_fsync`, `io_many_files`.

**Implementation priority:** **MEDIUM**. After `idle_long_baseline_v2` is established.

---

### 3.3 `mem_stream_v2`

**Behavior family:** MEM

**Experimental question:** Does a write-stream workload at 1 GB working set with controlled THP and warm-up phases produce a cleaner signal than the current `mem_stream`?

**Behavioral mechanism isolated:** Sequential page-strided writes at hardware bandwidth with deterministic page granularity.

**Why current tests do not isolate this mechanism:** Current `mem_stream` at 128 MB has small active fraction; metrics are diluted. THP and prefetcher behavior are uncontrolled.

**Confusion/artifact problem it resolves:** Should eliminate `mem_stream → run_idle` confusion (MI cause).

**Low-level control technique:** mmap + MADV_NOHUGEPAGE + mlock + memset warm-up. Tight C inner loop with one 4-byte sequence write per page. CPU pinning.

**Required implementation level:** **C strongly recommended**.

**OS/kernel controls required:** CPU pinning via `sched_setaffinity`. Document THP setting.

**Cache/TLB/hardware-aware controls required:** Default cache behavior. Optional MOVNTI variant (only after `mp_cache_dram_observability` confirms it is needed).

**Filesystem/page-cache controls required:** None (anonymous).

**Runtime controls required:** No allocations during measurement. Warm-up phase clearly separated.

**Expected memory-signal effect:**
- Higher dc_coherence and snr_mean than current `mem_stream` due to larger active fraction.
- Lower CV than current.
- Segment-level stability higher because warm-up is excluded.
- Hamming/cosine delta concentrated in the 1 GB region.

**Expected confusions:**
- `mem_stream_v2 → cache_stride_unified (page stride)`
  - Type: behavioral proximity
  - Reason: same mechanism at large working set.

**Realism justification:** This is the canonical "structured sequential writes" probe. Real workloads produce sequential writes (memcpy, large arrays).

**Variants:**
- Working sets: 256 MB, 1 GB, 4 GB. Each tests a different cache-vs-DRAM regime.
- Per-page write magnitudes: 1, 4, 64 bytes. Quantifies how delta magnitude depends on bytes-changed-per-page.

**Implementation priority:** **HIGH**. Among the first 5.

---

### 3.4 `mem_chase_unified`

**Behavior family:** MEM

**Experimental question:** Does a write-visible random-page-access workload produce a clean signature distinct from sequential writes?

**Behavioral mechanism isolated:** Pseudo-random page accesses with each access producing a write (so the access is observable to the delta pipeline).

**Why current tests do not isolate this mechanism:** Current `mem_pointer_chase` is read-only and invisible. Both `mem_pointer_chase` redesign and `mem_random_write_pages` in the filled spec do the same thing with different PRNGs. Merging is appropriate.

**Confusion/artifact problem it resolves:** Resolves `mem_pointer_chase` invisibility. Eliminates the `mem_random_write_pages` redundancy.

**Low-level control technique:** Single workload with a parameterized PRNG choice (LCG variant, xoshiro variant). Each access writes a byte derived from PRNG state to the visited page.

**Required implementation level:** **C strongly recommended**. Assembly only if `mp_cache_dram_observability` shows compiler reordering hides the dependency.

**OS/kernel controls required:** mlock + MAP_POPULATE + MADV_NOHUGEPAGE. CPU pinning.

**Cache/TLB/hardware-aware controls required:** TLB pressure expected near 100% miss rate at 1 GB working set.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** No allocations during measurement.

**Expected memory-signal effect:**
- High event rate.
- Broadband cepstral content (no rhythm).
- High spectral entropy.
- Should clearly separate from `mem_stream_v2` in feature space.

**Expected confusions:**
- `mem_chase_unified → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both produce scattered random writes.
- `mem_chase_unified → io_rand_rw`
  - Type: behavioral proximity
  - Reason: same surface signal, different page type.
- `mem_chase_unified (LCG) → mem_chase_unified (xoshiro)`
  - Type: behavioral proximity (very high)
  - Reason: same mechanism with different PRNG; expected to overlap.

**Realism justification:** Random page access patterns are common in pointer-heavy programs (graph traversal, hash tables, garbage collection scans).

**Variants:**
- LCG-driven (deterministic across runs at fixed seed).
- xoshiro-driven (different PRNG quality).
- Working sets 1 GB and 4 GB.
- Read-only calibration variant (replaces legacy `mem_pointer_chase`; demonstrates invisibility).

**Implementation priority:** **VERY HIGH**. Resolves the largest source of MEM scatter.

---

### 3.5 `mem_alloc_v2_cadence_sweep`

**Behavior family:** MEM

**Experimental question:** How does the per-batch sleep cadence affect the cleanliness of allocation-driven signal?

**Behavioral mechanism isolated:** mmap-touch-munmap rhythm at parameterized cadences.

**Why current tests do not isolate this mechanism:** Current `mem_alloc_touch_pages` uses fixed 20 ms sleep. The cadence-vs-cleanliness relationship is unknown.

**Confusion/artifact problem it resolves:** Quantifies the relationship between batch period and observable cepstral periodicity. Tests whether the strong CV (0.054) of the current variant is due specifically to the 20 ms cadence or to mmap rhythm in general.

**Low-level control technique:** C with direct mmap/munmap. nanosleep cadence parameterized: 5, 20, 50, 100, 200 ms.

**Required implementation level:** **C strongly recommended**.

**OS/kernel controls required:** Document `nr_dirty_writeback_centisecs` and other writeback parameters.

**Cache/TLB/hardware-aware controls required:** MADV_NOHUGEPAGE per allocation.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** No glibc allocator. Direct mmap.

**Expected memory-signal effect:**
- Cepstral peak shifts with cadence.
- CV stable across cadences if rhythm is the source of cleanliness.
- CV varies with cadence if cadence-snapshot alignment is the source.

**Expected confusions:**
- Across cadences, expected behavioral proximity.
- `mem_alloc_v2 (long sleep) → mem_alloc_v2 (short sleep)`
  - Type: behavioral proximity
  - Reason: same mechanism, different cadence.

**Realism justification:** Real allocators run at varying cadences. Periodic allocation patterns are realistic.

**Variants:**
- 5 cadences as listed.
- Object size variants: 64 KB, 256 KB, 1 MB. Different mmap thresholds.

**Implementation priority:** **MEDIUM**. After `mem_chase_unified` and `mem_stream_v2` are validated.

---

### 3.6 `io_rand_rw_v2`

**Behavior family:** IO

**Experimental question:** Does separating cache-state and read/write components reveal which component drives the `io_rand_rw → mem_alloc_touch_pages` confusion?

**Behavioral mechanism isolated:** Random-offset block I/O with explicit cache-state and read/write controls.

**Why current tests do not isolate this mechanism:** Current `io_rand_rw` mixes read and write halves and has uncontrolled cache warmth.

**Confusion/artifact problem it resolves:** Tests the BO-vs-MI hypothesis for the only IO confusion in current data.

**Low-level control technique:** C with `pread`/`pwrite`, `posix_fadvise(POSIX_FADV_RANDOM)`, `fallocate`, parameterized cache state via `posix_fadvise(POSIX_FADV_DONTNEED)` or pre-warm.

**Required implementation level:** **C recommended**.

**OS/kernel controls required:** Document virtual disk type. Drop caches before each variant.

**Cache/TLB/hardware-aware controls required:** None.

**Filesystem/page-cache controls required:** Explicit cold versus warm. Optional `O_DIRECT` variant after `mp_capture_pause_calibration` and pre-verification confirm O_DIRECT propagates through virtio-blk.

**Runtime controls required:** Standard.

**Expected memory-signal effect:**
- Write-only cold-cache variant: should produce strongest signal (every write dirties a new page).
- Write-only warm-cache variant: pages are already in cache; writes dirty existing pages.
- Read-only cold-cache variant: kernel populates cache; per-page deltas come from cache fills.
- Read-only warm-cache variant: minimal observable signal.
- O_DIRECT write variant: substantially lower active_page_fraction.

**Expected confusions:**
- `io_rand_rw_v2 (write cold) → mem_chase_unified`
  - Type: behavioral proximity
  - Reason: scattered random writes.
- `io_rand_rw_v2 (read warm) → idle_long_baseline_v2`
  - Type: metric inadequacy
  - Reason: read-only invisibility.

**Realism justification:** Real random I/O workloads have varying cache warmth and read/write ratios. Separating them is a basic methodological control.

**Variants:**
- 2x2 grid: read versus write, cold versus warm.
- O_DIRECT variant if pre-verified.

**Implementation priority:** **HIGH**. Resolves a Phase 1 confusion.

---

### 3.7 `io_seq_fsync_v2`

**Behavior family:** IO

**Experimental question:** Does the choice of fsync, fdatasync, or O_SYNC change the observable cepstral structure?

**Behavioral mechanism isolated:** Three different kernel synchronization paths for sequential writes.

**Why current tests do not isolate this mechanism:** Current `io_seq_fsync` only uses fsync. The choice between fsync (data + metadata), fdatasync (data only), and O_SYNC (per-write enforcement) is methodologically meaningful.

**Confusion/artifact problem it resolves:** Distinguishes data-flush rhythm from metadata-flush rhythm.

**Low-level control technique:** C with `pwrite` and one of three sync mechanisms per variant. Rolling write within a fallocate'd file.

**Required implementation level:** **C recommended** (specifically for fdatasync; Python acceptable for fsync-only).

**OS/kernel controls required:** Document filesystem type and journal mode.

**Cache/TLB/hardware-aware controls required:** None.

**Filesystem/page-cache controls required:** fallocate before measurement.

**Runtime controls required:** Standard.

**Expected memory-signal effect:**
- fsync variant: data + metadata journal blocks.
- fdatasync variant: data blocks only; metadata path quiet.
- O_SYNC variant: per-write sync, may produce higher event rate.
- Cepstral structure may differ enough to distinguish them.

**Expected confusions:**
- All three variants likely cluster as "io_seq_fsync_v2" sub-cluster, distinct from other workloads but similar to each other.

**Realism justification:** Real applications use different sync calls based on durability requirements. The distinction is operationally meaningful.

**Variants:**
- fsync, fdatasync, O_SYNC.
- Rolling-write versus growing-file.

**Implementation priority:** **MEDIUM**. After main MEM and IO rewrites are done.

---

### 3.8 `io_many_files_v2_split`

**Behavior family:** IO

**Experimental question:** Does the same workload produce different signatures on tmpfs versus disk-backed filesystem?

**Behavioral mechanism isolated:** VFS slab churn (tmpfs) versus VFS slab churn plus journal plus block I/O (disk).

**Why current tests do not isolate this mechanism:** Current `io_many_files` uses `tempfile.mkdtemp` which puts the directory under `/tmp` of unknown filesystem type.

**Confusion/artifact problem it resolves:** Resolves the scientific validity ambiguity in the current data.

**Low-level control technique:** C with explicit `--dir` argument. Two variants: tmpfs-mounted directory and disk-backed directory.

**Required implementation level:** **C recommended**.

**OS/kernel controls required:** Document filesystem type for each variant. Verify with `mount`.

**Cache/TLB/hardware-aware controls required:** None.

**Filesystem/page-cache controls required:** Drop caches before each variant.

**Runtime controls required:** Sequential filenames (no `random.getrandbits` overhead).

**Expected memory-signal effect:**
- tmpfs variant: heavy slab churn (`dentry`, `inode_cache`). Zero disk I/O. May or may not look the same as current `io_many_files`.
- Disk variant: slab churn plus journal pressure plus actual block writes. Should look the same as current `io_many_files` if `/tmp` was disk-backed.
- The two variants should have measurably different cepstral structures.

**Expected confusions:**
- `io_many_files_v2 (tmpfs) → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both produce many small slab allocations.
- `io_many_files_v2 (disk) → io_seq_fsync`
  - Type: behavioral proximity
  - Reason: both produce journal-mediated rhythm.

**Realism justification:** Real workloads target different filesystem types. The distinction is operationally meaningful.

**Variants:**
- tmpfs and ext4 (or whichever disk filesystem the guest has).
- Optional fdatasync-per-file variant for disk.

**Implementation priority:** **HIGH**. Validity blocker for current results.

---

### 3.9 `cache_stride_unified`

**Behavior family:** CACHE

**Experimental question:** How does observable signal depend on stride and working set size relative to the cache hierarchy?

**Behavioral mechanism isolated:** Stride-cache-line and stride-page-size relationships.

**Why current tests do not isolate this mechanism:** Filled spec has both `cache_stride_sweep` and `mem_stride_sweep_large` doing similar things at different scales. Merging gives one sweep with two axes.

**Confusion/artifact problem it resolves:** Removes the redundancy. Provides a single parameter sweep covering cache-bounded and DRAM-bounded regimes.

**Low-level control technique:** C with parameterized buffer size and stride. Inner loop dirties one byte per access.

**Required implementation level:** **C strongly recommended**.

**OS/kernel controls required:** MADV_NOHUGEPAGE. CPU pinning.

**Cache/TLB/hardware-aware controls required:** Buffer sizes covering L1, L2, L3, and exceeds-LLC. Strides covering 8, 64, 256, 4096, 65536 bytes.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** Pre-fault before measurement.

**Expected memory-signal effect:**
- Per-(buffer, stride) cell: distinct cepstral peak frequency.
- Buffer size at L1: minimal observable delta (cache hides).
- Buffer size exceeding LLC: every access goes to DRAM.
- Stride at cache-line size: maximum cache misses per access.

**Expected confusions:**
- `cache_stride_unified (large buffer, page stride) → mem_stream_v2`
  - Type: behavioral proximity (very high)
  - Reason: same mechanism at large working set.

**Realism justification:** Stride access patterns appear in real numerical code (sparse matrices, multi-dimensional arrays).

**Variants:**
- 4 buffer sizes × 5 strides = 20 cells. Pick 6 to 8 representative cells.

**Implementation priority:** **MEDIUM-HIGH**. After methodology probes confirm cache observability.

---

### 3.10 `cpu_canonical`

**Behavior family:** CPU

**Experimental question:** Is register-resident compute distinguishable from idle in the delta pipeline?

**Behavioral mechanism isolated:** Pure CPU work with no intentional memory traffic.

**Why current tests do not isolate this mechanism:** Filled spec has `cpu_hash_loop` and `cpu_branch_random` as separate tests. Both produce the same negative finding.

**Confusion/artifact problem it resolves:** Demonstrates the read versus write asymmetry of the observation method.

**Low-level control technique:** Tight C loop computing a hash over an internal counter. No allocations. CPU pinned.

**Required implementation level:** **C strongly recommended**. Assembly not needed.

**OS/kernel controls required:** CPU pinning.

**Cache/TLB/hardware-aware controls required:** Inner loop fits in L1. State in registers.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** No allocations.

**Expected memory-signal effect:**
- Near-idle on all delta-pipeline metrics.
- Distinguishable from idle only in CPU usage (not in delta pipeline).

**Expected confusions:**
- `cpu_canonical → idle_long_baseline_v2`
  - Type: metric inadequacy
  - Reason: register-resident compute is invisible.

**Realism justification:** Real CPU-bound workloads (compression, crypto, parsing) often run with small working sets. Demonstrating their invisibility validates the read versus write thesis.

**Variants:**
- Hash variant.
- Branch variant.
- These should produce statistically indistinguishable signals; if they do not, the difference is itself informative.

**Implementation priority:** **LOW**. One canonical probe suffices. Useful as a calibration measurement.

---

## 4. New Experimental Tests (filling gaps)

These are tests that the filled spec did not include but that resolve specific Phase 1 questions.

---

### 4.1 `mem_stream_writevalue_sweep`

**Behavior family:** MEM

**Experimental question:** Does the magnitude of bytes-changed-per-page affect observable signal at fixed page count?

**Behavioral mechanism isolated:** Per-page Hamming/cosine delta sensitivity to write magnitude.

**Why current tests do not isolate this mechanism:** Current `mem_stream` writes 1 byte per page. The pipeline's sensitivity to per-page change magnitude is uncharacterized.

**Confusion/artifact problem it resolves:** Quantifies metric sensitivity. Determines whether the small-active-fraction problem is the dominant cause of `mem_stream → run_idle` or whether per-page magnitude matters too.

**Low-level control technique:** C `mem_stream` variant with parameterized per-page write magnitude: 1, 4, 64, 4096 bytes (full page).

**Required implementation level:** **C strongly recommended**.

**OS/kernel controls required:** Standard `mem_stream_v2`.

**Cache/TLB/hardware-aware controls required:** Standard.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** Standard.

**Expected memory-signal effect:**
- 1 byte per page: minimal Hamming delta per page.
- 4096 bytes per page (full page write): maximum Hamming delta per page.
- The progression isolates the metric's per-page magnitude sensitivity.

**Expected confusions:**
- All variants are still `mem_stream`-type behaviorally; differ in observable amplitude.

**Realism justification:** Real workloads write different amounts per page (cache lines, buffers, full pages). The distinction is real.

**Variants:**
- 4 magnitudes as listed.

**Implementation priority:** **MEDIUM-HIGH**. Cheap variant of `mem_stream_v2`.

---

### 4.2 `io_writeback_dynamics`

**Behavior family:** IO

**Experimental question:** What is the writeback decay shape after a burst of dirty-page creation?

**Behavioral mechanism isolated:** Kernel writeback throttling and `pdflush`/`kworker` cadence.

**Why current tests do not isolate this mechanism:** No current test characterizes writeback behavior independent of workload activity.

**Confusion/artifact problem it resolves:** Quantifies writeback time constant. Informs idle-duration choice for residue elimination.

**Low-level control technique:** Workload writes 500 MB of dirty pages quickly via buffered writes (no fsync). Workload then exits. Capture continues for 600 s of "idle." The signal during the post-workload window IS the writeback decay.

**Required implementation level:** **C recommended**.

**OS/kernel controls required:** Document `vm.dirty_ratio`, `vm.dirty_background_ratio`, `vm.dirty_writeback_centisecs`.

**Cache/TLB/hardware-aware controls required:** None.

**Filesystem/page-cache controls required:** Buffered writes only. No fsync.

**Runtime controls required:** Fixed write volume.

**Expected memory-signal effect:**
- Sharp peak during the 500 MB write burst.
- Decay over tens of seconds as the kernel writes back to disk.
- Decay shape is approximately exponential with time constant determined by kernel parameters.

**Expected confusions:**
- During decay phase, `io_writeback_dynamics → idle_residue_decay_per_workload(post-io_seq_fsync)`
  - Type: behavioral proximity
  - Reason: same mechanism (kernel writeback).

**Realism justification:** Writeback decay is a real phenomenon affecting every IO-heavy workload's transition to idle.

**Variants:**
- 500 MB burst, 1 GB burst, 2 GB burst.

**Implementation priority:** **MEDIUM**. Useful diagnostic; not blocking.

---

### 4.3 `mem_first_touch_isolated`

**Behavior family:** MEM

**Experimental question:** Is the first-touch page-fault phase of a workload separable from the steady-state phase in the delta signal?

**Behavioral mechanism isolated:** Lazy allocation and CoW page-fault dynamics.

**Why current tests do not isolate this mechanism:** Current `mem_stream` mixes first-touch with steady-state. The contributions are not separable from full-run aggregation.

**Confusion/artifact problem it resolves:** Quantifies the contribution of page-fault events to the per-recording feature vector.

**Low-level control technique:** Workload allocates 1 GB anonymous mapping, does NOT pre-fault, then begins steady-state writes. Capture starts before allocation. Compare per-segment metrics: segment 1 (allocation + first-touch) versus segments 2-k (steady-state).

**Required implementation level:** **C strongly recommended**.

**OS/kernel controls required:** No `MAP_POPULATE`. No prior `memset`.

**Cache/TLB/hardware-aware controls required:** MADV_NOHUGEPAGE.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** Standard.

**Expected memory-signal effect:**
- Segment 1: high event rate from page faults plus stream writes.
- Later segments: lower event rate (faults complete after first sweep).
- Segment-level analysis exposes the boundary.

**Expected confusions:**
- `mem_first_touch_isolated (segment 1) → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: page-fault burst resembles allocation burst.

**Realism justification:** Real workloads have warm-up phases. Separating them is a basic methodological control.

**Variants:** Compare to `mem_stream_v2` (which has warm-up isolated by design).

**Implementation priority:** **MEDIUM**. Useful for segment-level methodology.

---

## 5. Family Viability Probes

These are minimum-effort experiments that determine whether the THREAD and NETWORK families produce observable signal at all.

---

### 5.1 `thread_viability_probe`

**Behavior family:** THREAD (viability probe)

**Experimental question:** Does adding multi-threading to `mem_alloc_touch_pages` produce signal distinguishable from the single-threaded version?

**Behavioral mechanism isolated:** Multi-threaded version of a known-clean MEM workload.

**Why current tests do not isolate this mechanism:** None of the THREAD tests in the filled spec have been validated for observability.

**Confusion/artifact problem it resolves:** Determines whether the THREAD family is worth implementing.

**Low-level control technique:** N threads (N = guest CPU count) each run a stripped-down `mem_alloc_touch_pages` loop with direct mmap.

**Required implementation level:** **C strongly recommended** (Python disqualified due to GIL).

**OS/kernel controls required:** Threads pinned to distinct cores. Document kernel `mmap_lock` contention via `perf` if available.

**Cache/TLB/hardware-aware controls required:** Each thread operates on its own buffer to avoid intentional sharing.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** Standard.

**Expected memory-signal effect:**
- If observable: higher event rate, higher Fano factor than single-threaded.
- If not observable: signal is the sum of N independent single-threaded contributions, no qualitative difference.

**Expected confusions:**
- `thread_viability_probe → mem_alloc_touch_pages`
  - Type: behavioral proximity (high)
  - Reason: same fundamental mechanism.

**Realism justification:** Real workloads are concurrent. Knowing whether the pipeline can detect concurrency matters for generalization.

**Variants:** Thread counts 2, 4, 8.

**Implementation priority:** **HIGH**. Gates the THREAD family.

---

### 5.2 `net_viability_probe`

**Behavior family:** NETWORK (viability probe)

**Experimental question:** Does TCP loopback traffic produce observable memory-signal activity beyond what the involved processes' user-space activity would produce alone?

**Behavioral mechanism isolated:** Kernel network stack (sk_buff allocation, socket buffers).

**Why current tests do not isolate this mechanism:** None of the NETWORK tests have been validated for observability.

**Confusion/artifact problem it resolves:** Determines whether the NETWORK family is worth implementing.

**Low-level control technique:** Two C processes on the guest. Producer sends 1 MB at high rate over TCP loopback. Consumer receives.

**Required implementation level:** **C recommended**.

**OS/kernel controls required:** Pin processes to distinct cores. Standard TCP settings.

**Cache/TLB/hardware-aware controls required:** None.

**Filesystem/page-cache controls required:** None.

**Runtime controls required:** Document `/proc/slabinfo` `skbuff_head_cache` deltas during the run.

**Expected memory-signal effect:**
- If observable: kernel slab churn produces measurable per-page deltas.
- If not observable: signal is dominated by user-space buffer activity in producer and consumer.

**Expected confusions:**
- `net_viability_probe → io_seq_fsync`
  - Type: behavioral proximity
  - Reason: both are streaming kernel-mediated activity.

**Realism justification:** Network-mediated workloads are common. Knowing if they leave a fingerprint matters.

**Variants:** TCP only initially. UDP and small-message variants only after this probe is positive.

**Implementation priority:** **HIGH**. Gates the NETWORK family.

---

## 6. MIXED Family Redesign

The filled spec's three MIXED tests are tautological as point tests. Redesign as parameter sweeps.

---

### 6.1 `mixed_mem_io_ratio_sweep`

**Behavior family:** MIXED

**Experimental question:** As the mem-vs-io throughput ratio changes from 100% MEM to 100% IO, does the feature space trajectory pass through known parent clusters or take a different path?

**Behavioral mechanism isolated:** Continuous mixing of two parent mechanisms with controlled rates.

**Why current tests do not isolate this mechanism:** Filled spec proposes one fixed ratio.

**Confusion/artifact problem it resolves:** Tests whether the feature space behaves continuously between MEM and IO clusters.

**Low-level control technique:** Two threads with parameterized rate-ratio. Five operating points: 100/0, 75/25, 50/50, 25/75, 0/100. The 100/0 and 0/100 points are calibration anchors (single-component baseline).

**Required implementation level:** **C strongly recommended**.

**OS/kernel controls required:** Threads pinned. Document throughput rates.

**Cache/TLB/hardware-aware controls required:** Standard.

**Filesystem/page-cache controls required:** Standard.

**Runtime controls required:** Document achieved rate per thread.

**Expected memory-signal effect:**
- Trajectory through feature space.
- 100/0 anchor matches `mem_chase_unified` or `mem_random_write_pages`.
- 0/100 anchor matches `io_rand_rw_v2`.
- Intermediate points sit on the trajectory.

**Expected confusions:**
- Each ratio variant is expected to closely resemble its dominant component.

**Realism justification:** Real workloads have varying ratios. The trajectory is the experiment.

**Variants:** 5 ratios as listed.

**Implementation priority:** **LOW**. After parent tests are well-characterized.

---

## 7. Top Recommended Enhanced Tests Overall

Ranked by combined scientific value, cleanliness, feasibility, and relevance to current confusion matrix.

| Rank | Test | Why it ranks here |
|---|---|---|
| 1 | `idle_long_baseline_v2` | Foundational baseline; every other comparison depends on it |
| 2 | `mem_chase_unified` | Resolves the largest source of MEM scatter |
| 3 | `mem_stream_v2` (1 GB + warm-up isolated) | Resolves the leading MI cause of `mem_stream → run_idle` |
| 4 | `io_many_files_v2_split` (tmpfs + disk) | Validity blocker; resolves the tmpfs ambiguity |
| 5 | `mp_capture_pause_calibration` | Quantifies the VM/capture artifact noise floor |
| 6 | `mp_cache_dram_observability` | Resolves whether MOVNTI variants are necessary |
| 7 | `io_rand_rw_v2` (cold/warm × read/write) | Tests the BO-vs-MI hypothesis for the only IO confusion |
| 8 | `mp_cycle_drift_probe` | Tests the cycle-drift hypothesis seen in `mem_stream` |
| 9 | `cache_stride_unified` | Single parameterized sweep replacing 2 separate tests |
| 10 | `mem_alloc_v2_cadence_sweep` | Quantifies the cadence-cleanliness relationship |

---

## 8. Minimal Next Experiment Batch

To validate the next design step before committing to a larger collection, run this batch first:

| Order | Test | Estimated capture time | Purpose |
|---|---|---|---|
| 1 | `mp_capture_pause_calibration` | 600 s × 4 intervals = 40 min | Noise floor characterization |
| 2 | `mp_cache_dram_observability` | 120 s × 4 variants = 8 min | Resolves cache visibility question |
| 3 | `idle_long_baseline_v2` | 600 s × 5 runs = 50 min | Establishes controlled baseline |
| 4 | `mem_stream_v2` (1 GB) | 300 s × 8 runs = 40 min | Tests MEM scatter hypothesis |
| 5 | `mem_chase_unified` (LCG, write-visible) | 300 s × 8 runs = 40 min | Resolves the read-only invisibility |
| 6 | `io_many_files_v2_split` (both filesystems) | 300 s × 16 runs = 80 min | Resolves tmpfs ambiguity |
| 7 | `mp_cycle_drift_probe` (mem_stream_v2) | 300 s × 8 + 60 s idle × 8 = 48 min | Tests intrinsic drift |
| Total | | ~5 hours | First validation pass |

**Why this batch is enough**: It validates the four leading hypotheses for current confusion patterns (small active fraction, read-only invisibility, tmpfs ambiguity, cycle drift) and characterizes the pipeline's noise floor. If all four hypotheses resolve, the priority subset of tests can be implemented with confidence. If any fail, the test design must be revisited before committing to broader implementation.

---

## 9. Tests to Avoid

The following test designs should not be implemented as currently specified.

| Test | Reason to avoid |
|---|---|
| Multiple invisibility-demonstration probes | Adding three CPU tests, plus `io_read_cache_hit`, plus read-only `cache_*` variants all repeat one negative finding. One canonical probe suffices. |
| Fixed-ratio MIXED tests | Each combines two mechanisms at one fixed ratio. The dominant component dominates by construction. The result is tautological. |
| Network tests before viability probe | Three NETWORK tests assume kernel network stack memory is observable. Unverified. |
| Two of three thread tests before viability probe | Same logic. |
| `mem_stride_sweep_large` as separate test | Subsumable into `cache_stride_unified` with parameterized buffer size. |
| `cache_cold_scan` as separate test | Subsumable into `mem_stream_v2` variants. |
| Tests designed to maximize separation rather than isolate mechanism | These tend to be artificially clean and not generalize. The thesis values mechanistic interpretation, not separability scores. |
| Adding repeat-runs of the existing 7 tests as the only change | Without addressing the read-only invisibility and small-footprint problems, more repeats do not resolve the confusion patterns. They only tighten provisional cleanliness designations. |

---

## 10. Thesis-Safe Wording

**On the enhanced test plan as a whole**:

> The enhanced test plan refines a smaller set of high-priority workloads identified in the design review of `next_gen_workload_test_mini_specs_filled.md`. The plan adds methodology probes that validate observation-pipeline assumptions, redesigns existing tests where Phase 1 audits identified specific artifact contributions, and replaces redundant probes with single canonical variants. The enhanced plan does not expand the test inventory; it focuses it. Cleanliness expectations remain provisional until empirical validation per `confusion_matrix_diagnostic_methodology.md` Section 6.

**On the methodology probes**:

> Six methodology probes are proposed to validate the observation-pipeline assumptions that the workload spec relies on: capture-pause noise floor, cache-versus-DRAM observability, Transparent Huge Page promotion timing, glibc allocator threshold effects, cycle-position drift, and execution-order contamination via reverse-cycle. These probes are not workloads in the classification sense; they are experiments that measure what the pipeline can and cannot see. Their outputs constrain the interpretation of all downstream classification claims.

**On the enhanced replacements**:

> Each enhanced test replaces a specific identified gap: a controlled baseline replacing the contaminated `run_idle`, a write-visible random-access workload replacing the read-only `mem_pointer_chase`, a 1 GB working set with isolated warm-up replacing the small-footprint `mem_stream`, and an explicit-filesystem `io_many_files` resolving the `/tmp` ambiguity. These are not optimizations of the original tests; they are corrections of identified workload-design or measurement-protocol issues.

**On the family viability probes**:

> Two family viability probes (`thread_viability_probe`, `net_viability_probe`) are proposed before committing to broader implementation of the THREAD and NETWORK families. The probes ask whether kernel-mediated activity in those subsystems leaves measurable traces in the delta-pipeline observation. If the answer is negative, the families should not be implemented as classification subtypes; they may still be useful as calibration measurements demonstrating observation-method limits.

**On what remains uncertain after this enhanced plan**:

> The enhanced plan does not address artifacts that originate outside the workload code: pmemsave pause variance characterization (probe characterizes the noise floor but does not eliminate it), KVM/host scheduling effects, and Transparent Huge Page promotion (probe characterizes timing but does not control it from inside the guest). These remain uncontrolled confounds. The plan separates what the workload designer can control from what the workload designer can only measure.

---

## 11. Summary

This document proposes 16 enhanced or new tests across four categories (methodology, enhanced replacements, gap-filling experiments, viability probes), grouped under the eight families. Compared to the 25-test filled spec, this plan focuses scope on:

- 6 methodology probes that validate pipeline assumptions
- 8 enhanced replacements for tests with identified design issues
- 2 family viability probes
- 1 redesigned MIXED sweep replacing 3 tautological MIXED tests

The minimal next experiment batch fits in approximately 5 hours of capture time. It directly tests the four leading hypotheses for current confusion patterns. After this batch validates (or refutes) those hypotheses, the broader test inventory can be implemented with empirical anchoring.

The plan does not increase dataset size for its own sake. Each test answers a specific scientific question. Each test isolates one mechanism. None are toy workloads designed for fake separability.
