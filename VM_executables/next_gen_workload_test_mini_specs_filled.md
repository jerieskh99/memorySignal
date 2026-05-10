# Next-Generation Workload Test Mini-Specs (Filled)

This document is the completed and updated version of `next_gen_workload_test_mini_specs.md`. Every test in the eight families is specified using the extended template below. The skeleton structure is preserved.

Existing tests are updated using the findings from:

- `confusion_matrix_diagnostic_methodology.md` (four-cause model, provisional language, typed expected confusions)
- `python_runtime_os_artifact_audit.md` (Phase 1, per-test artifact analysis)
- `artifact_confusion_matrix_connection.md` (Phase 1, mapping artifacts to confusions)
- `low_level_rewrite_recommendations.md` (Phase 1, C/assembly rewrite plans)

Proposed tests are specified concretely enough to guide later implementation in C or Python.

---

## Extended Mini-Spec Template

Every test below uses the following structure:

```text
Test name:
Status: Already used / Proposed
Behavior family:
Subtype:
Mechanism stressed:
Expected memory pattern:
Expected difference from existing tests:
Metrics likely affected:
Potential confounds:
Expected confusions:
Low-level control requirements:
Python/runtime artifact risk:
OS/kernel artifact risk:
VM/capture artifact risk:
Cleanliness expectation:
Suggested implementation level:
```

For "Expected confusions" the typed format from `confusion_matrix_diagnostic_methodology.md` Section 13 is used:

```text
- `[true subtype] → [predicted subtype]`
  - Type: behavioral proximity | capture artifact | execution-order contamination | metric inadequacy
  - Reason: [short mechanistic explanation]
```

Implementation levels:

- **Python acceptable**: language overhead does not contaminate the intended signal materially.
- **C recommended**: C reduces measurable artifacts but Python could still produce valid results.
- **C strongly recommended**: Python introduces artifacts that the audit identified as material.
- **Assembly only if justified**: a specific control (cache-bypass stores, dependent-load chain) is required that C cannot express portably.

---

## Common Observability Premise (apply to every test below)

The Hamming and cosine deltas operate on per-page byte content between consecutive QEMU snapshots. **A page contributes signal only when its bytes change.** Reads do not produce delta signal. Writes do.

Three classes of pages contribute:

1. Pages the workload itself writes.
2. Pages the kernel writes on the workload's behalf (page cache, journal, slab churn, page tables).
3. Pages anyone else writes during the snapshot interval (background OS, capture-side noise, residual writeback).

The relative weight of these classes determines what each test actually measures.

---

# 1. IDLE Family

## Family Explanation

Baseline / control behavior. Idle windows are not zero-signal; they include OS background, scheduler noise, cache cleanup, residual writeback from prior workloads, and capture-side pause artifacts.

## Why IDLE Matters

IDLE is the reference against which active workloads are interpreted. Phase 1 evidence shows IDLE in the current dataset has CV = 0.973 and that test1 and test3 silhouette scores are strongly negative, indicating non-stationarity driven by cycle position and prior-workload residue.

---

## 1.1 `run_idle`

**Test name:** `run_idle`

**Status:** Already used (24 recordings in the current 4-cycle dataset)

**Behavior family:** IDLE

**Subtype:** Idle baseline / control window between active workloads

**Mechanism stressed:**
No intentional synthetic workload. The bash script issues `sleep N`. The signal during the window is OS background plus residual state from the prior workload plus capture-pipeline artifacts.

**Expected memory pattern:**
- Decaying writeback signal that fades within the first 30-60 seconds when following IO-heavy workloads.
- Periodic kernel timer and kthread wakeups.
- Slab reclaim bursts.
- Variable signal level depending on prior workload.

**Expected difference from existing tests:**
- Lower mean event rate than any active workload.
- Higher CV than active workloads because background composition varies by cycle position.
- Cannot be assumed stationary across recordings.

**Metrics likely affected:**
- `event_rate` low.
- `snr_zero_frac` high.
- `dc_coherence` variable depending on residual writeback.
- `cep_periodicity_score` shaped by kernel timer cadences.
- High `CV` across runs.

**Potential confounds:**
- Residual writeback from prior workload (the dominant confound at current 60 s idle duration).
- Background OS daemons (`systemd-journald`, `dbus-daemon`, `crond`, `kworker`).
- Capture-pipeline pause variance.
- VM scheduler noise when host runs other tenants.
- First-idle-after-boot differs from later idle windows.

**Expected confusions:**
- `run_idle → run_idle` (subtype split, position-dependent)
  - Type: execution-order contamination
  - Reason: idle following IO-heavy workloads carries writeback residue; idle following idle accumulates only OS background. These can form distinct sub-clusters.
- `mem_stream → run_idle`
  - Type: metric inadequacy
  - Reason: small active fraction of `mem_stream` may be averaged toward background level by per-recording aggregation.
- `mem_pointer_chase → run_idle`
  - Type: metric inadequacy
  - Reason: read-only chase produces no observable writes; recorded signal is interpreter and OS background, very close to true idle.

**Low-level control requirements:**
- Drop caches before each idle window via privileged shell command.
- Capture `/proc/meminfo` `Dirty` and `Writeback` at start and end of window.
- Record prior workload identity in metadata sidecar.
- Optionally extend idle duration to 120-180 seconds to allow full writeback decay.

**Python/runtime artifact risk:** Very low. Bash sleep has no interpreter loop.

**OS/kernel artifact risk:** Very high. The OS IS the signal source.

**VM/capture artifact risk:** Medium. pmemsave pauses still occur in idle windows.

**Cleanliness expectation:** Provisionally low until protocol changes are applied. The current dataset shows median CV 0.973, the highest of all classes.

**Suggested implementation level:** Python acceptable (the script is one bash sleep). The fix is protocol-level, not code-level.

---

## 1.2 `idle_long_baseline`

**Test name:** `idle_long_baseline`

**Status:** Proposed

**Behavior family:** IDLE

**Subtype:** Long uncontaminated idle baseline

**Mechanism stressed:**
A controlled idle window of 600 seconds or more, taken either after a fresh VM boot or after an extended pre-warm-up idle period and a cache-drop. The objective is to establish a stationary lower-bound reference signal against which contaminated idle windows can be compared.

**Expected memory pattern:**
- Stable low event rate after an initial 60-90 s settle period.
- Background OS daemon activity at known cadences.
- No detectable writeback after the settle period.
- Lower CV than `run_idle` because no prior workload residue is present.

**Expected difference from existing tests:**
- Compared with `run_idle`, the signal should be stationary across the measurement portion and lower in mean amplitude.
- Compared with all active workloads, the signal should be substantially weaker on every metric sensitive to writes.

**Metrics likely affected:**
- `event_rate` very low.
- `snr_zero_frac` very high.
- `cep_periodicity_score` weak; if any peak appears it should align with kernel timer interrupts (e.g. 250 Hz or 1000 Hz `HZ`).
- `CV` across runs much lower than `run_idle`.

**Potential confounds:**
- Background daemons may still issue periodic activity. These should be characterized but not eliminated.
- Capture pause artifacts persist.
- Long duration may overlap with cron jobs (e.g. `systemd-timers`) that fire occasionally.

**Expected confusions:**
- `idle_long_baseline → run_idle`
  - Type: behavioral proximity
  - Reason: same fundamental stimulus (none); difference is residue, not mechanism. The two should overlap if `run_idle` happens to follow another idle window.
- `cpu_hash_loop → idle_long_baseline`
  - Type: metric inadequacy
  - Reason: register-resident compute may produce no page-content delta and look idle-like.

**Low-level control requirements:**
- VM should be at thermal and scheduler steady state before the window starts.
- Cache drop and 60-90 s pre-window settle.
- All filesystem operations from previous activity must have drained (verified via `/proc/meminfo` `Dirty == 0` and `Writeback == 0`).
- No active workload may have run within the previous 5 minutes.

**Python/runtime artifact risk:** Very low. Bash sleep.

**OS/kernel artifact risk:** Medium. Background daemons still active. Scope is to characterize, not suppress.

**VM/capture artifact risk:** Medium. Capture pause variance unchanged.

**Cleanliness expectation:** Provisionally high. Should be the cleanest baseline possible without modifying the guest OS.

**Suggested implementation level:** Python acceptable. The mechanism is bash sleep with a tightly controlled protocol around it.

---

## 1.3 `idle_post_workload_recovery`

**Test name:** `idle_post_workload_recovery`

**Status:** Proposed

**Behavior family:** IDLE

**Subtype:** Post-workload recovery / residue characterization

**Mechanism stressed:**
A deliberately contaminated idle window placed immediately after a known active workload, recorded for the explicit purpose of measuring how the prior workload's residue decays in time. The active workload is fixed (one at a time), and the idle is long enough to capture the full decay curve.

**Expected memory pattern:**
- High initial signal in the first 30-60 seconds, decaying toward `idle_long_baseline` levels.
- Decay shape depends on the prior workload identity: page-cache writeback, slab reclaim, freed-mmap unmap pressure, dentry cache reclaim each have distinct decay profiles.
- Segment-level analysis (per `segment_level_analysis_critique_and_plan.md`) recommended to localize the decay.

**Expected difference from existing tests:**
- Compared with `idle_long_baseline`, the early portion is contaminated and the mean is higher.
- Compared with `run_idle`, this version is structured: the contamination is deliberate and the prior workload identity is recorded.

**Metrics likely affected:**
- Early `event_rate` elevated; late `event_rate` near `idle_long_baseline`.
- Slope of `event_rate` versus segment index becomes the discriminative feature.
- `cep_periodicity_score` may show prior-workload rhythm bleeding into early segments (e.g. fsync residue).

**Potential confounds:**
- Variable initial conditions across runs of the same prior workload.
- Capture-pipeline jitter blurs the early decay.
- Filesystem journal commits may occur asynchronously and produce late spikes.

**Expected confusions:**
- `idle_post_workload_recovery_after_io_seq_fsync → io_seq_fsync`
  - Type: execution-order contamination
  - Reason: ongoing writeback from the prior workload creates a signal that closely resembles the active fsync rhythm in the early seconds.
- `idle_post_workload_recovery_after_mem_alloc_touch_pages → mem_alloc_touch_pages`
  - Type: execution-order contamination
  - Reason: residual unmap pressure from the prior batch persists for some seconds.

**Low-level control requirements:**
- Each variant tests recovery from exactly one prior workload identity.
- Workload-to-idle transition timing must be precise (start the recovery measurement at workload exit, not at a fixed wall-clock interval).
- Capture per-segment metrics for decay analysis.

**Python/runtime artifact risk:** Very low.

**OS/kernel artifact risk:** Very high (intentional). The OS post-workload behavior is the signal.

**VM/capture artifact risk:** Medium.

**Cleanliness expectation:** Provisionally medium. By design the workload sits between idle and the prior active workload in feature space. Not "clean" in the same sense as a probe; it is a calibration measurement.

**Suggested implementation level:** Python acceptable. Bash sleep wrapped in a controlled cycle template.

---

# 2. MEM Family

## Family Explanation

Direct user-space stress on volatile memory. Subtypes differ in access order, locality, allocation behavior, and write granularity.

## Why MEM Matters

Phase 1 identified MEM as the highest-scatter class (silhouette = -0.258) in the current dataset. The principal cause is `mem_pointer_chase` being read-only and invisible to the delta pipeline. The proposed updates and additions explicitly address this.

---

## 2.1 `mem_stream`

**Test name:** `mem_stream`

**Status:** Already used (4 recordings)

**Behavior family:** MEM

**Subtype:** Sequential page-strided writes

**Mechanism stressed:**
Sequential traversal of a large buffer, writing one byte to byte 0 of every 4 KB page. The intended signal is regular, locality-preserving, prefetcher-friendly memory pressure with deterministic dirty-page progression.

**Expected memory pattern:**
- Many small per-page Hamming deltas concentrated in a contiguous virtual region.
- The per-snapshot delta covers either the full active buffer (if sweep period < snapshot interval) or a moving subset (otherwise).
- Strong DC and low-frequency MSC components if active fraction is large.

**Expected difference from existing tests:**
- Compared with `mem_pointer_chase` (post-redesign), the spatial access order is regular, not pseudo-random.
- Compared with `mem_alloc_touch_pages`, no allocation events occur during measurement.
- Compared with `cache_cold_scan`, this version writes; `cache_cold_scan` reads.
- Compared with IO tests, no syscalls occur during measurement.

**Metrics likely affected:**
- `dc_coherence` elevated when active fraction is large.
- `cepstral_peak_idx` shifted depending on sweep periodicity.
- `snr_mean` and `snr_high_frac` increase with working set size.
- `spectral_slope` more negative with regular pattern.

**Potential confounds:**
- Hardware prefetcher detects the pattern and compresses DRAM traffic.
- Transparent Huge Pages promotion may change page granularity mid-run.
- CoW lazy allocation: the first sweep is mixed with steady-state writes unless warm-up is separated.
- Small working set (current 128 MB) dilutes per-recording metrics by averaging over static pages.
- Python interpreter overhead bounds sweep rate below hardware capacity.

**Expected confusions:**
- `mem_stream → run_idle`
  - Type: metric inadequacy
  - Reason: at 128 MB working set, the active fraction is small relative to total guest pages; per-recording aggregation pulls the mean toward background.
- `mem_stream → io_rand_rw`
  - Type: execution-order contamination
  - Reason: prior IO-heavy workloads leave page-cache writeback that bleeds into the `mem_stream` window.
- `mem_stream → cache_cold_scan` (post-rewrite, write-stream variant)
  - Type: behavioral proximity
  - Reason: both produce contiguous regular page-content changes; differ only in working-set size relative to LLC.

**Low-level control requirements:**
- Increase working set to 1 GB for the canonical variant.
- `mmap` + `madvise(MADV_NOHUGEPAGE)` + `mlock()` + warm-up phase separated by `memset`.
- Loop in C; no interpreter dispatch in the steady-state loop.
- CPU pinning via `sched_setaffinity`.
- Optional MOVNTI variant for cache-bypassing writes.

**Python/runtime artifact risk:** Medium-high in current Python form (interpreter dispatch sets sweep rate). Eliminated by C rewrite.

**OS/kernel artifact risk:** Medium. THP and prefetcher remain even after C rewrite unless explicitly suppressed.

**VM/capture artifact risk:** Medium. Pause variance lands during user-space stores; effect is bounded.

**Cleanliness expectation:** Provisionally weak in the current form. Provisionally medium-high after C rewrite at 1 GB working set with MADV_NOHUGEPAGE. Assembly variant (MOVNTI) expected to be the cleanest because it forces DRAM traffic independent of cache state.

**Suggested implementation level:** C strongly recommended. Assembly only if justified for the cache-bypass variant.

---

## 2.2 `mem_pointer_chase`

**Test name:** `mem_pointer_chase`

**Status:** Already used (4 recordings; **flagged for redesign**)

**Behavior family:** MEM

**Subtype:** Pseudo-random page traversal with poor locality

**Mechanism stressed (current and intended):**
Pseudo-random page-granular traversal driven by an LCG. The current implementation is read-only and invisible to a content-delta observation pipeline. The redesigned version writes the LCG state byte to the visited page on each step, making the access sequence observable.

**Expected memory pattern (after redesign):**
- One per-page write at LCG-determined random offsets across a 1 GB working set.
- High TLB miss rate and DRAM access scattering.
- Weaker cepstral periodicity than `mem_stream` because the LCG period is long relative to snapshot intervals.

**Expected difference from existing tests:**
- Compared with `mem_stream`, the spatial pattern is pseudo-random, not sequential.
- Compared with `mem_random_write_pages`, this version uses a deterministic LCG with a fixed seed; `mem_random_write_pages` may use a different RNG or seed strategy.
- Compared with `io_rand_rw`, this version writes anonymous pages; no filesystem involvement.

**Metrics likely affected (after redesign):**
- `event_rate` high.
- `cepstral` content broadband.
- `snr_skewness` near zero (broad distribution).
- `MSC` mid-frequency content elevated.
- `spectral_entropy` higher than streaming.

**Potential confounds:**
- LCG with fixed seed produces identical access order each run; while not noise, this pins the access pattern to one realization. A randomized-seed variant should be added for diversity.
- TLB pressure varies with THP setting.
- Memory pressure at 1 GB may push other VM processes into reclaim.
- Compiler may reorder or speculate around the dependent-load chain in C; assembly with `LFENCE` may be needed for the data-dependent variant.

**Expected confusions (after redesign):**
- `mem_pointer_chase → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both produce scattered random page-content changes.
- `mem_pointer_chase → io_rand_rw`
  - Type: behavioral proximity
  - Reason: both produce random scattered writes; differ in page type (anonymous vs file-cache).
- `mem_pointer_chase (read-only legacy) → run_idle`
  - Type: metric inadequacy
  - Reason: read-only operations are invisible to the delta pipeline. This confusion only applies to the legacy (current) implementation.

**Low-level control requirements:**
- Redesign: add a write per access (e.g. `buf[idx*stride] = (uint8_t)(x & 0xFF)`).
- `mmap` + `MAP_POPULATE` for explicit pre-fault, separated from measurement.
- `madvise(MADV_NOHUGEPAGE)` to keep 4 KB granularity.
- `mlock()` to keep the working set resident.
- Optional assembly for a true data-dependent load chain with `LFENCE`.

**Python/runtime artifact risk:** High in current form (interpreter limits chase rate). Lower after C rewrite, but the read-only invisibility issue is workload design, not language.

**OS/kernel artifact risk:** Medium. TLB and DRAM behavior remain.

**VM/capture artifact risk:** Medium.

**Cleanliness expectation:** Currently provisionally weak (3 of 4 runs collapse to near-IDLE). After redesign with writes, expected provisionally medium-high.

**Suggested implementation level:** C strongly recommended for redesign. Assembly only if justified for guaranteed dependent-load chain.

---

## 2.3 `mem_alloc_touch_pages`

**Test name:** `mem_alloc_touch_pages`

**Status:** Already used (4 recordings; cleanest current MEM subtype)

**Behavior family:** MEM

**Subtype:** Allocation churn with page touching

**Mechanism stressed:**
Repeated batches of object allocations, page touches, and releases. Each batch creates a strong rhythm of mmap-touch-munmap or arena-equivalent activity, separated by a fixed sleep. The signal IS the rhythm.

**Expected memory pattern:**
- Strong cepstral peak at 1 / batch_period frequency.
- Bursty page-table activity: 128K minor faults per batch (current parameters).
- Low CV across runs (currently 0.054, the best of all subtypes).

**Expected difference from existing tests:**
- Compared with `mem_stream`, this is allocation-driven; pages appear and disappear, not just get dirtied.
- Compared with `mem_pointer_chase` (post-redesign), the access pattern is regular within each object but the allocation/release boundaries are the dominant signal.
- Compared with `io_many_files`, this is anonymous mmap activity, not filesystem objects.

**Metrics likely affected:**
- `cep_periodicity_score` strongly elevated.
- `cepstral_peak_idx_var` reflects burst structure.
- `snr_fano` low (regular event spacing).
- `dc_coherence` elevated.

**Potential confounds:**
- glibc's `M_MMAP_THRESHOLD` may dynamically grow above 256 KB and switch to arena reuse, changing the syscall pattern mid-run.
- Python `bytearray` adds object-header and refcount overhead.
- Kernel slab activity for VMA structures adds noise.
- Sleep cadence drift if `time.sleep` is preempted by kworker.

**Expected confusions:**
- `io_rand_rw → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both produce scattered random page-content changes; current metrics do not separate anonymous from page-cache pages.
- `mem_alloc_touch_pages → io_seq_fsync`
  - Type: behavioral proximity
  - Reason: both have a strong batch rhythm; if cycle periods coincide, cepstral signatures may overlap.
- `mem_alloc_touch_pages → thread_parallel_alloc`
  - Type: behavioral proximity
  - Reason: extension to multi-threading should produce a similar but more bursty version of the same signal.

**Low-level control requirements:**
- Bypass glibc allocator: direct `mmap`/`munmap` syscalls per object.
- Fixed sleep via `nanosleep(20 ms, NULL)`.
- 4-byte sequence number written per page instead of 1 byte for stronger per-page delta.
- `madvise(MADV_NOHUGEPAGE)` per allocation.
- Optional `MAP_POPULATE` variant to separate page-fault phase from touch phase.

**Python/runtime artifact risk:** Medium (allocator threshold switching). Eliminated by C rewrite using direct mmap.

**OS/kernel artifact risk:** Medium-high. Page-fault and slab dynamics remain by design; they are part of the intended signal.

**VM/capture artifact risk:** Medium.

**Cleanliness expectation:** Provisionally high (currently 4/4 correct, CV 0.054). Expected to tighten further after C rewrite.

**Suggested implementation level:** C strongly recommended (eliminates allocator nondeterminism). Assembly not justified.

---

## 2.4 `mem_random_write_pages`

**Test name:** `mem_random_write_pages`

**Status:** Proposed

**Behavior family:** MEM

**Subtype:** Random write pressure on persistent buffer

**Mechanism stressed:**
A 1 GB anonymous buffer is allocated and pre-faulted once. The measurement loop selects pseudo-random page indices via a fast PRNG (xoshiro or similar) and writes a 4-byte sequence number to byte 0 of each. Unlike `mem_pointer_chase`, the access pattern is independent of page content; unlike `mem_alloc_touch_pages`, no allocation occurs during measurement.

**Expected memory pattern:**
- High event rate of scattered random page changes.
- No batch rhythm; signal is broadband.
- Strong contrast with `mem_stream` because of spatial randomness.
- Strong contrast with `mem_alloc_touch_pages` because of absent batch boundaries.

**Expected difference from existing tests:**
- Compared with `mem_pointer_chase` post-redesign, this uses a faster PRNG (not LCG) and writes a multi-byte value per page (stronger per-page delta).
- Compared with `mem_stream`, the spatial pattern is uniformly random; no prefetcher benefit.
- Compared with `cache_cold_scan`, this writes; the cold scan reads.

**Metrics likely affected:**
- `spectral_entropy` very high.
- `cep_periodicity_score` low (no rhythm).
- `event_rate` high.
- `snr_mean` and `snr_high_frac` elevated.
- `MSC` mid-to-high frequency components broadband.

**Potential confounds:**
- PRNG state may pin the access pattern across runs unless seeded differently.
- TLB pressure and DRAM scheduling vary with host load.
- Per-page write value choice may bias delta magnitude (4 bytes versus 1 byte).

**Expected confusions:**
- `mem_random_write_pages → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both produce scattered random page-content changes when viewed at the per-recording level.
- `mem_random_write_pages → io_rand_rw`
  - Type: behavioral proximity
  - Reason: both produce random scattered writes at page granularity.
- `mem_random_write_pages → mem_pointer_chase`
  - Type: behavioral proximity
  - Reason: both are random-access write workloads; the PRNG and addressing schemes differ.

**Low-level control requirements:**
- Pre-fault the full 1 GB working set before measurement starts.
- Use a fast user-space PRNG (xoshiro256++ or similar) running in registers.
- Write 4 bytes per page touch, not 1.
- `madvise(MADV_NOHUGEPAGE)` and `mlock` on the buffer.
- Random seed parameter: fixed for reproducibility, plus a randomized-seed variant.

**Python/runtime artifact risk:** Medium in Python (per-iteration dispatch). Low in C.

**OS/kernel artifact risk:** Medium. TLB and DRAM remain.

**VM/capture artifact risk:** Medium.

**Cleanliness expectation:** Provisionally high in C. Expected to be a clean canonical "random-write" probe that disambiguates the read-only collapse problem of legacy `mem_pointer_chase`.

**Suggested implementation level:** C strongly recommended.

---

## 2.5 `mem_stride_sweep_large`

**Test name:** `mem_stride_sweep_large`

**Status:** Proposed

**Behavior family:** MEM

**Subtype:** Stride-controlled large-region traversal

**Mechanism stressed:**
A large buffer (4-8 GB if memory permits, otherwise 2 GB) is traversed with a parameterized stride. The same buffer is swept many times during the measurement window, with different strides forming variants. The objective is to isolate stride period from access locality and from memory volume.

**Expected memory pattern:**
- For small strides (page-aligned, 4 KB): nearly equivalent to `mem_stream` but on a larger buffer.
- For large strides (multiple of TLB stride, e.g. 1 MB): few touched pages per sweep but many sweeps; signal is rhythmic and sparse.
- For irrational/coprime strides (e.g. 4099 bytes): pattern eventually visits every page but in non-sequential order; differs from random-access by being deterministic and nearly-uniform.

**Expected difference from existing tests:**
- Compared with `mem_stream`, the stride is parameterized, not fixed at one page.
- Compared with `mem_pointer_chase`, the order is deterministic and stride-derived; no PRNG.
- Compared with `cache_stride_sweep`, this version operates on a buffer larger than LLC; cache effects are different.

**Metrics likely affected:**
- `cep_periodicity_score` strongly elevated and stride-dependent.
- `MSC` shows peaks at the stride frequency.
- `spectral_slope` shape varies with stride.
- `snr_fano` low (very regular).

**Potential confounds:**
- Hardware prefetcher behavior depends strongly on stride; some strides defeat it.
- THP promotion makes 2 MB strides effectively page-step rather than byte-step.
- L1/L2 cache behavior depends on stride relative to set associativity.

**Expected confusions:**
- `mem_stride_sweep_large (page stride) → mem_stream`
  - Type: behavioral proximity
  - Reason: at stride = 4 KB the workload IS sequential streaming on a larger buffer.
- `mem_stride_sweep_large (large stride) → mem_pointer_chase` (post-redesign)
  - Type: metric inadequacy
  - Reason: per-recording metrics may not distinguish "deterministic large stride" from "random page selection."

**Low-level control requirements:**
- Buffer size 2-8 GB depending on guest RAM.
- Stride parameterized: 4 KB, 64 KB, 1 MB, 4099 bytes (coprime variant).
- `mmap` + `madvise(MADV_NOHUGEPAGE)` to fix granularity.
- Pre-fault all pages before measurement.
- Optional: assembly variant with explicit `PREFETCHT0` or `PREFETCHNTA` hints to study prefetch effects.

**Python/runtime artifact risk:** High in Python (loop rate becomes interpreter-bound). Low in C.

**OS/kernel artifact risk:** Medium-high. Prefetcher and THP remain unless explicitly disabled.

**VM/capture artifact risk:** Medium.

**Cleanliness expectation:** Provisionally medium per stride. Variants should be analyzed independently because each stride asks a different scientific question.

**Suggested implementation level:** C strongly recommended.

---

# 3. IO Family

## Family Explanation

Filesystem and storage path stress. IO workloads affect volatile memory through page cache, dirty pages, metadata, kernel buffers, and journal commits.

## Why IO Matters

Phase 1 evidence shows IO subtypes cluster more cleanly than MEM (median CV 0.05-0.19) because the kernel imposes a deterministic cadence. The IO redesign focus is therefore on **explicit control of which kernel mechanism is invoked**, not on cleaning user-space code.

---

## 3.1 `io_rand_rw`

**Test name:** `io_rand_rw`

**Status:** Already used (4 recordings)

**Behavior family:** IO

**Subtype:** Random block read/write file I/O

**Mechanism stressed:**
Random-offset block-level reads and writes against a 2 GB file with mixed read/write ratio. The intended signal is page-cache churn from random-access.

**Expected memory pattern:**
- Scattered random dirty pages in the page cache.
- Variable page-cache hit rate.
- Async writeback bursts.
- Higher entropy and Fano factor than sequential I/O.

**Expected difference from existing tests:**
- Compared with `io_seq_fsync`, this lacks fsync rhythm and writeback is async.
- Compared with `io_many_files`, this stresses block data, not metadata.
- Compared with `mem_alloc_touch_pages`, the pages are file-cache-backed, not anonymous.
- Compared with `io_direct_write_like`, this version uses the page cache; the direct variant bypasses it.

**Metrics likely affected:**
- `event_rate` high.
- `spectral_entropy` high.
- `Fano factor` high.
- `cepstral` content broadband.

**Potential confounds:**
- Sparse-file extent allocation on first write to a hole (truncate creates sparse file).
- Page-cache warmth depends on cycle position and file size relative to RAM.
- Kernel writeback timing variability.
- `lseek + read/write` is two separate syscalls; preemption between them adds jitter (small).

**Expected confusions:**
- `io_rand_rw → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both produce scattered random page-content changes; current metrics do not separate page-cache from anonymous mappings.
- `io_rand_rw → io_seq_fsync`
  - Type: behavioral proximity
  - Reason: writeback bursts in random I/O may align with fsync rhythm at certain dirty-ratio thresholds.
- `io_rand_rw → mem_random_write_pages` (post-`mem_random_write_pages` introduction)
  - Type: behavioral proximity
  - Reason: similar surface signal; differ only by mapping type.

**Low-level control requirements:**
- Replace `truncate` with `fallocate` to commit extents before measurement.
- Use `pread`/`pwrite` instead of `lseek + read/write`.
- `posix_fadvise(POSIX_FADV_RANDOM)` at open.
- Cache-state variants: cold-cache (`POSIX_FADV_DONTNEED` before measurement) versus warm-cache (pre-read entire file).
- Separate read-only and write-only variants for diagnostic clarity.

**Python/runtime artifact risk:** Low (Python overhead is small relative to syscall + disk).

**OS/kernel artifact risk:** High (page cache, writeback, journal). Intended.

**VM/capture artifact risk:** Medium-high. Long-syscall workloads land snapshot pauses inside the kernel.

**Cleanliness expectation:** Provisionally medium (CV 0.189, the highest among IO). Expected to tighten with `fallocate` and explicit cache-state control.

**Suggested implementation level:** C recommended for the canonical variant (uses `pread`/`pwrite`/`fallocate`/`posix_fadvise`). Python acceptable as a fallback.

---

## 3.2 `io_seq_fsync`

**Test name:** `io_seq_fsync`

**Status:** Already used (4 recordings)

**Behavior family:** IO

**Subtype:** Sequential write with forced synchronization

**Mechanism stressed:**
Sequential 4 MB chunked writes followed by `fsync` after each chunk. The signal is the kernel-mediated write-flush rhythm: page cache fills, then drains via fsync, repeatedly.

**Expected memory pattern:**
- Strongly periodic at the fsync cadence.
- Page-cache region oscillates between dirty and clean.
- Journal commit pressure.
- Disk-latency-dominated period.

**Expected difference from existing tests:**
- Compared with `io_rand_rw`, this is rhythmic and sequential.
- Compared with `io_many_files`, this is bulk data, not metadata.
- Compared with `net_tcp_loopback_stream`, the synchronization is filesystem-driven, not network-buffer-driven.

**Metrics likely affected:**
- `cep_periodicity_score` high.
- `cepstral_peak_idx` near 1 (strong low-quefrency periodicity).
- `dc_coherence` elevated.
- `spectral_slope` strongly negative.

**Potential confounds:**
- Unbounded file growth: `wb` mode + repeated appends grows the file each chunk. Filesystem extent allocation per chunk adds journal noise to early writes.
- `fsync` flushes both data and metadata; the choice of `fdatasync` would isolate data.
- Disk latency variability.
- Host fsync barrier policy varies by virtual disk driver.

**Expected confusions:**
- `io_seq_fsync → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both have strong periodic batch structure. Currently not observed but plausible at higher n.
- `io_seq_fsync → net_tcp_loopback_stream`
  - Type: behavioral proximity
  - Reason: both produce rhythmic streaming-style buffer activity.

**Low-level control requirements:**
- `fallocate` a fixed-size file (e.g. 10 GB) and use rolling `pwrite` with offset modulo file size.
- Variants: `fsync` versus `fdatasync` versus `O_SYNC`.
- Variant: small-chunk (64 KB) versus default chunk (4 MB).
- Document the underlying disk type and fsync barrier configuration.

**Python/runtime artifact risk:** Very low (sync latency dominates).

**OS/kernel artifact risk:** Very high (intended). The kernel rhythm is the signal.

**VM/capture artifact risk:** Medium-high. fsync blocks for tens of ms; capture pauses can land in the middle.

**Cleanliness expectation:** Provisionally high (currently 4/4, CV 0.092). Expected to tighten with rolling-write variant.

**Suggested implementation level:** C recommended. Python acceptable but loses the `fdatasync` versus `fsync` distinction.

---

## 3.3 `io_many_files`

**Test name:** `io_many_files`

**Status:** Already used (4 recordings)

**Behavior family:** IO

**Subtype:** Metadata-heavy small-file churn

**Mechanism stressed:**
Repeated batches of file create + write small payload + close + unlink. Stresses filesystem metadata, dentry and inode caches, journal entries.

**Expected memory pattern:**
- Heavy slab cache churn for `dentry` and `inode_cache`.
- Frequent journal entries (one per create, one per unlink).
- Small per-file page-cache writes (1 KB rounds up to one page each).
- Strong batch rhythm.

**Expected difference from existing tests:**
- Compared with `io_seq_fsync`, this is metadata-dominated, not data-dominated.
- Compared with `io_rand_rw`, this stresses VFS object lifecycle, not block access.
- Compared with `mem_alloc_touch_pages`, this is filesystem-mediated, not anonymous.

**Metrics likely affected:**
- `event_rate` very high (many small events per batch).
- `cep_entropy` is the top discriminator for IO subtypes (Phase 1 evidence: sep = 702 in `stochastic_characterization_summary.txt`).
- `snr_active_frac` and `active_page_fraction` elevated.
- `cepstral_peak_idx_var` high.

**Potential confounds:**
- **`/tmp` filesystem type unknown** in the current Python implementation. If `/tmp` is tmpfs the workload is RAM-backed VFS only; if disk-backed there is real I/O. This is the largest unresolved confound for the current results.
- Dentry and inode cache state at start depends on prior cycle activity.
- Filesystem journal mode (ordered, writeback, journal) changes the signal.

**Expected confusions:**
- `io_many_files → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both produce many small object create/destroy events. Distinguished by journal pressure on disk-backed fs and dentry pressure on tmpfs.
- `io_many_files → io_seq_fsync`
  - Type: behavioral proximity
  - Reason: both have strong batch rhythm; differ in payload size and metadata density.

**Low-level control requirements:**
- Explicit target directory parameter (no `tempfile.mkdtemp`); document filesystem type in metadata.
- Two variants: tmpfs and disk-backed, run separately and labeled.
- Optional: with `fdatasync` per file vs without, to isolate journal commit cadence.
- Sequential filenames (no `random.getrandbits` overhead).

**Python/runtime artifact risk:** Low (kernel-bound).

**OS/kernel artifact risk:** Very high (intended). Signal is VFS/journal/slab.

**VM/capture artifact risk:** Medium-high. Many short syscalls per batch produce many opportunities for capture pauses to land mid-syscall.

**Cleanliness expectation:** Provisionally high (currently 4/4, CV 0.048). Provisional designation should split tmpfs and disk variants when implemented.

**Suggested implementation level:** C recommended (forces explicit FS choice and removes tmpfs ambiguity). Python acceptable only with explicit directory.

---

## 3.4 `io_read_cache_hit`

**Test name:** `io_read_cache_hit`

**Status:** Proposed

**Behavior family:** IO

**Subtype:** Cache-hot read-heavy I/O

**Mechanism stressed:**
A file (e.g. 256 MB, smaller than page cache) is pre-read sequentially to fully populate the page cache. The measurement loop performs random reads against this cache-resident region. No writes. The objective is to characterize what the delta pipeline sees during a workload that does heavy I/O syscall traffic but produces no new dirty pages.

**Expected memory pattern:**
- Very low delta signal (reads do not dirty pages).
- High syscall rate but minimal observable Hamming/cosine activity.
- Possible signal from kernel read-ahead structures and small buffer churn.
- Should approximate `idle_long_baseline` in delta-pipeline metrics despite high CPU and syscall activity.

**Expected difference from existing tests:**
- Compared with `io_rand_rw`, this is read-only and cache-hot.
- Compared with `idle_long_baseline`, this has high CPU and syscall activity but similar delta signal.
- Compared with `cpu_hash_loop`, this exercises kernel-userspace transitions even though both produce minimal dirty pages.

**Metrics likely affected:**
- `event_rate` low.
- `snr_zero_frac` high.
- May still show small periodic activity from read-buffer reuse.

**Potential confounds:**
- Page-cache eviction if the file is larger than free RAM.
- Read-ahead populates additional pages; visible as a one-time event.
- Per-process buffer reads do touch some user-space pages (the read buffer itself). One page is repeatedly written to.

**Expected confusions:**
- `io_read_cache_hit → run_idle`
  - Type: metric inadequacy
  - Reason: read-only workloads produce minimal page-content delta; the delta pipeline cannot distinguish them from idle in standard form.
- `io_read_cache_hit → idle_long_baseline`
  - Type: metric inadequacy
  - Reason: same reason.
- `io_read_cache_hit → cpu_hash_loop`
  - Type: behavioral proximity
  - Reason: both are user-space-bound workloads with minimal observable memory deltas.

**Low-level control requirements:**
- File size much smaller than guest RAM.
- Pre-warm cache by sequential read of full file before measurement.
- Use `pread` for atomic read at random offset.
- Read into a single shared buffer to bound user-space writes.
- Optional `posix_fadvise(POSIX_FADV_WILLNEED)` to keep the file resident.

**Python/runtime artifact risk:** Low.

**OS/kernel artifact risk:** Medium. Read-ahead and cache-eviction dynamics.

**VM/capture artifact risk:** Medium-high. High syscall rate increases capture pause-mid-syscall events.

**Cleanliness expectation:** Provisionally medium for the purpose of being a "deliberately quiet IO" calibration probe. Cleanliness here means consistency, not strong signal.

**Suggested implementation level:** C recommended (precise control of buffer size and cache pre-warm). Python acceptable.

---

## 3.5 `io_direct_write_like`

**Test name:** `io_direct_write_like`

**Status:** Proposed

**Behavior family:** IO

**Subtype:** Page-cache-bypassing direct-style writes

**Mechanism stressed:**
Sequential or random writes opened with `O_DIRECT`, bypassing the page cache. The kernel issues raw block I/O to the disk without intermediate dirty-page accumulation. The signal is whatever the block layer and virtual disk path write into guest memory (DMA descriptors, journal blocks, device buffers).

**Expected memory pattern:**
- Almost no page-cache dirty pages (by design).
- Kernel block-layer activity: request structures, DMA setup.
- Filesystem journal entries (extent allocation if writing to holes; metadata if not preallocated).
- Strong contrast with `io_seq_fsync` because no page-cache stage.

**Expected difference from existing tests:**
- Compared with `io_seq_fsync`, no fsync needed; data goes straight to disk.
- Compared with `io_rand_rw`, this version eliminates the page-cache from the signal entirely.
- Compared with `mem_alloc_touch_pages`, this is kernel block-layer activity, not user anonymous mapping.

**Metrics likely affected:**
- `event_rate` lower than buffered IO at same throughput.
- Cepstral content shaped by disk-block-layer rhythm rather than fsync cadence.
- `dc_coherence` lower than buffered IO.
- Strong contrast with all other IO subtypes on `active_page_fraction`.

**Potential confounds:**
- `O_DIRECT` requires sector-aligned buffers; alignment errors cause silent fallback.
- Some virtual disk drivers (virtio-blk) may not preserve `O_DIRECT` semantics through to the host.
- Filesystem support for `O_DIRECT` varies; tmpfs does not support it.
- If `fallocate` is not used, extent allocation noise persists in the journal.

**Expected confusions:**
- `io_direct_write_like → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both produce kernel-side memory activity at comparable rate; differ in mechanism (block layer vs anonymous mapping).
- `io_direct_write_like → idle_long_baseline`
  - Type: metric inadequacy
  - Reason: if the block layer's memory footprint per write is small, the delta pipeline may observe near-zero signal.

**Low-level control requirements:**
- Open with `O_DIRECT | O_RDWR`.
- `posix_memalign(buffer, 4096, block_size)` for sector-aligned buffer.
- `fallocate` the file before measurement.
- Use `pwrite` at sector-aligned offsets.
- Document the host disk type and virtio-blk configuration.

**Python/runtime artifact risk:** Low.

**OS/kernel artifact risk:** Medium-high. Block layer and journal remain.

**VM/capture artifact risk:** Medium-high. Long block-write latency increases pause-mid-syscall odds.

**Cleanliness expectation:** Provisionally medium-high. Should be a structurally distinct probe from buffered IO if `O_DIRECT` is honored end-to-end.

**Suggested implementation level:** C strongly recommended (`O_DIRECT` requires alignment and direct syscall control).

---

# 4. CPU Family

## Family Explanation

CPU-bound computation. Tests user-space arithmetic, bit operations, and branch logic. Most CPU work is register-resident and produces minimal page-content delta.

## Why CPU Matters

Phase 1 evidence shows the delta pipeline is content-write-sensitive. CPU workloads test what the metrics see when the workload genuinely does no memory writes at page granularity. This separates "memory-bound" from "CPU-bound" classes and provides a control for read-only-style invisibility.

---

## 4.1 `cpu_hash_loop`

**Test name:** `cpu_hash_loop`

**Status:** Proposed

**Behavior family:** CPU

**Subtype:** Tight register-resident hash computation

**Mechanism stressed:**
A tight loop computing a hash (e.g. SipHash, FNV-1a, or xxHash) over a small register-resident state. The input is internally generated (e.g. an incrementing counter) so no memory reads are required. No writes to memory beyond stack frame state.

**Expected memory pattern:**
- Minimal page-content delta. Stack frame and one or two user-space pages may show small changes.
- High CPU instruction throughput, low cache pressure beyond instruction stream.
- No syscalls during measurement.
- Should approximate `idle_long_baseline` in delta metrics.

**Expected difference from existing tests:**
- Compared with `idle_long_baseline`, this consumes CPU cycles but produces similar memory signal.
- Compared with `cpu_matrix_mult`, this has minimal data working set.
- Compared with `mem_pointer_chase` (legacy read-only), both are nearly invisible to the delta pipeline.

**Metrics likely affected:**
- `event_rate` very low.
- `snr_zero_frac` high.
- `dc_coherence` low.
- Should be near-indistinguishable from idle on all signal metrics.

**Potential confounds:**
- Compiler may unroll the loop, changing instruction-stream cache behavior. The instruction stream itself is read-only and invisible to the pipeline.
- Stack frame writes are bounded but nonzero. The same small set of pages is repeatedly written.
- If implemented in Python, interpreter eval-stack churn could dominate the signal.

**Expected confusions:**
- `cpu_hash_loop → run_idle`
  - Type: metric inadequacy
  - Reason: register-resident compute is invisible to a content-delta observation pipeline.
- `cpu_hash_loop → idle_long_baseline`
  - Type: metric inadequacy
  - Reason: same.
- `cpu_hash_loop → io_read_cache_hit`
  - Type: behavioral proximity
  - Reason: both produce minimal observable memory delta despite high active CPU usage.

**Low-level control requirements:**
- Tight C inner loop. No allocation during measurement.
- Hash function with no global state (avoid runtime state writes).
- CPU pinning to one core.
- Optional assembly variant to ensure no compiler-introduced memory traffic.

**Python/runtime artifact risk:** Very high in Python (interpreter eval-stack writes dominate). Eliminated in C.

**OS/kernel artifact risk:** Very low (no syscalls).

**VM/capture artifact risk:** Low-medium.

**Cleanliness expectation:** Provisionally clean as a "CPU-bound near-idle" calibration probe. If it cannot be distinguished from idle, that is a useful negative finding for the thesis.

**Suggested implementation level:** C strongly recommended. Assembly only if justified for true zero memory writes.

---

## 4.2 `cpu_matrix_mult`

**Test name:** `cpu_matrix_mult`

**Status:** Proposed

**Behavior family:** CPU

**Subtype:** Dense numeric computation with structured memory reuse

**Mechanism stressed:**
Repeated multiplication of two square float matrices. Matrix size is parameterized to fit in L2, L3, or exceed LLC. The output matrix is written each iteration. Stresses combined compute and memory-locality, with the balance tunable by matrix size.

**Expected memory pattern:**
- Output matrix pages are repeatedly written.
- Input matrix pages are repeatedly read.
- Strong locality if matrix fits in L3; cache-cold scan-like behavior if exceeds.
- Cepstral periodicity at iteration boundary.

**Expected difference from existing tests:**
- Compared with `cpu_hash_loop`, this has explicit memory traffic.
- Compared with `mem_stream`, the access pattern reuses pages (locality), and only the output is written.
- Compared with `cache_hot_loop`, the working set is structured; the loop has dependent access patterns.

**Metrics likely affected:**
- `event_rate` moderate; depends on output matrix size.
- `cep_periodicity_score` elevated at iteration cadence.
- `dc_coherence` elevated if output matrix is small enough to be repeatedly dirtied.

**Potential confounds:**
- Compiler vectorization and loop tiling change memory access patterns.
- If implemented in Python with NumPy, calls into BLAS (OpenBLAS, MKL); the BLAS implementation determines memory traffic structure entirely.
- Cache size and associativity effects.

**Expected confusions:**
- `cpu_matrix_mult → mem_stream`
  - Type: behavioral proximity
  - Reason: large-matrix variant scans output sequentially, similar to streaming.
- `cpu_matrix_mult → cache_hot_loop`
  - Type: behavioral proximity
  - Reason: small-matrix variant reuses cache repeatedly.
- `cpu_matrix_mult → mem_random_write_pages`
  - Type: behavioral proximity
  - Reason: large-matrix output matrix touched in sequence-but-spaced pattern at row-stride; aliasing with random-page can occur.

**Low-level control requirements:**
- Matrix size variants: fits in L1, L2, L3, exceeds LLC.
- C with explicit loop ordering (i-j-k vs i-k-j) to control access pattern.
- No BLAS unless documented.
- Aligned matrices via `posix_memalign`.

**Python/runtime artifact risk:** Very high if using NumPy/BLAS (the BLAS implementation IS the workload). Lower in plain C.

**OS/kernel artifact risk:** Low (no syscalls).

**VM/capture artifact risk:** Low.

**Cleanliness expectation:** Provisionally medium. Heavily depends on matrix size variant; each size answers a different question.

**Suggested implementation level:** C recommended. Python acceptable only if BLAS choice is documented.

---

## 4.3 `cpu_branch_random`

**Test name:** `cpu_branch_random`

**Status:** Proposed

**Behavior family:** CPU

**Subtype:** Branch-heavy unpredictable computation

**Mechanism stressed:**
A loop containing many data-dependent branches whose outcomes are pseudo-random. Stresses branch prediction without producing significant memory traffic. The state is register-resident.

**Expected memory pattern:**
- Minimal page-content delta.
- High instruction throughput, branch-misprediction-bound.
- No syscalls.
- Should approximate `cpu_hash_loop` in observable delta signal.

**Expected difference from existing tests:**
- Compared with `cpu_hash_loop`, this has irregular control flow.
- Compared with `cpu_matrix_mult`, this has minimal data working set.
- Compared with all MEM and IO tests, this should produce far less observable signal.

**Metrics likely affected:**
- `event_rate` very low.
- Should be near-indistinguishable from idle.

**Potential confounds:**
- Compiler-generated branch prediction hints.
- If implemented in Python, the eval loop branches ARE the workload's dominant signal.

**Expected confusions:**
- `cpu_branch_random → cpu_hash_loop`
  - Type: behavioral proximity
  - Reason: both are register-resident compute with minimal observable signal.
- `cpu_branch_random → run_idle`
  - Type: metric inadequacy
  - Reason: invisible to delta pipeline.

**Low-level control requirements:**
- C tight loop with conditional branches whose outcome depends on a fast PRNG state.
- No memory allocation during measurement.
- CPU pinning.

**Python/runtime artifact risk:** Very high in Python.

**OS/kernel artifact risk:** Very low.

**VM/capture artifact risk:** Low.

**Cleanliness expectation:** Provisionally clean as a near-idle CPU-bound calibration probe.

**Suggested implementation level:** C strongly recommended.

---

# 5. CACHE Family

## Family Explanation

Locality and cache behavior. Tests memory access patterns whose distinguishing feature is their relationship to L1, L2, or LLC capacity, not the working-set size in bytes.

## Why CACHE Matters

CACHE tests separate "memory volume" from "memory locality" as discriminative features. They directly probe whether the metric pipeline distinguishes access STRUCTURE versus access AMOUNT. This is core to the semantic-free thesis.

---

## 5.1 `cache_hot_loop`

**Test name:** `cache_hot_loop`

**Status:** Proposed

**Behavior family:** CACHE

**Subtype:** Hot-cache small-working-set repeated access

**Mechanism stressed:**
A small buffer (e.g. 32 KB, fits in L1) is repeatedly traversed and dirtied with a tight loop. Cache-resident throughout. Effectively no DRAM traffic after the first pass.

**Expected memory pattern:**
- Same small set of pages repeatedly dirtied. Per-snapshot delta covers a tiny region.
- Cache lines dirty but rarely write back to DRAM.
- Snapshot-pause may capture cache state inconsistency: the snapshot reads DRAM, but the dirty data lives in cache. **This is a subtle but important point: cache-resident dirty data may not appear in pmemsave output.**
- Therefore, the observable delta may underrepresent the actual memory activity.

**Expected difference from existing tests:**
- Compared with `mem_stream`, the working set is much smaller and resident in L1.
- Compared with `cache_cold_scan`, the working set fits in cache.
- Compared with `cpu_hash_loop`, this has explicit memory writes (small region).

**Metrics likely affected:**
- `event_rate` low to medium (only the small region appears in deltas).
- `dc_coherence` may be low if write-back to DRAM is infrequent.
- Working set is concentrated; `active_page_fraction` very low.

**Potential confounds:**
- Cache write-back timing depends on cache associativity and line eviction policy.
- pmemsave reads physical RAM at snapshot time. If dirty data is in cache and not yet written back, the snapshot may miss it.
- Snapshot pause itself may flush the cache to DRAM (unclear without verification).

**Expected confusions:**
- `cache_hot_loop → run_idle`
  - Type: metric inadequacy
  - Reason: cache-resident writes may not propagate to DRAM frequently enough for the pipeline to observe.
- `cache_hot_loop → cpu_hash_loop`
  - Type: behavioral proximity
  - Reason: both are CPU-cache-resident with minimal observable DRAM traffic.

**Low-level control requirements:**
- Buffer size 16-64 KB (L1 range) and 256-512 KB (L2 range) variants.
- Tight C loop with read-modify-write per cache line.
- Optional assembly variant with `CLFLUSH` to force write-back, contrasted with default to study cache-residency effects.
- CPU pinning.

**Python/runtime artifact risk:** Medium (interpreter overhead may dominate).

**OS/kernel artifact risk:** Low (no syscalls).

**VM/capture artifact risk:** Medium-high. The cache-vs-DRAM observability question is unique to this test.

**Cleanliness expectation:** Provisionally low. This test's primary scientific value is exposing the cache-vs-DRAM observability gap, not producing a clean signal.

**Suggested implementation level:** C strongly recommended. Assembly only if justified for `CLFLUSH` controls.

---

## 5.2 `cache_cold_scan`

**Test name:** `cache_cold_scan`

**Status:** Proposed

**Behavior family:** CACHE

**Subtype:** Larger-than-cache linear scan with cache-bypass

**Mechanism stressed:**
A buffer larger than LLC (e.g. 2 GB on a system with 16 MB LLC) is scanned linearly with reads, writes, or both. Each access misses cache and goes to DRAM. The objective is to produce maximum DRAM traffic with sequential pattern.

**Expected memory pattern:**
- Sequential page-content changes if the scan writes.
- Strong DRAM bandwidth pressure.
- Hardware prefetcher may detect and prefetch ahead.
- Streaming-like signal; close to `mem_stream` at large working set.

**Expected difference from existing tests:**
- Compared with `cache_hot_loop`, the working set exceeds cache; every access touches DRAM.
- Compared with `mem_stream`, the variant set distinguishes read-scan from write-scan; `mem_stream` is currently write-only.
- Compared with `mem_stride_sweep_large`, this version uses unit stride.

**Metrics likely affected:**
- `event_rate` high if writing.
- `cep_periodicity_score` reflects sweep period.
- `dc_coherence` strong with regular pattern.
- Read-only variant: minimal delta signal.

**Potential confounds:**
- Hardware prefetcher behavior.
- THP promotion may collapse the buffer into 2 MB pages.
- Read-only variant suffers same invisibility as legacy `mem_pointer_chase`.

**Expected confusions:**
- `cache_cold_scan (write) → mem_stream`
  - Type: behavioral proximity
  - Reason: identical mechanism at large working set; differ only in conceptual labeling.
- `cache_cold_scan (read) → cpu_hash_loop`
  - Type: metric inadequacy
  - Reason: read-only scans do not produce delta signal.

**Low-level control requirements:**
- Buffer size much larger than LLC (typically 2-8 GB).
- Variants: read-scan, write-scan, read-modify-write-scan.
- `madvise(MADV_NOHUGEPAGE)`.
- Pre-fault before measurement.
- Optional NT-store variant for write to force DRAM-visible writes.

**Python/runtime artifact risk:** High in Python. Eliminated in C.

**OS/kernel artifact risk:** Medium (THP, prefetcher).

**VM/capture artifact risk:** Medium.

**Cleanliness expectation:** Provisionally medium-high for the write variant. The read variant is provisionally a "deliberately invisible" probe like `cpu_hash_loop`.

**Suggested implementation level:** C strongly recommended.

---

## 5.3 `cache_stride_sweep`

**Test name:** `cache_stride_sweep`

**Status:** Proposed

**Behavior family:** CACHE

**Subtype:** Stride-controlled cache locality probe

**Mechanism stressed:**
Access a buffer of cache-comparable size (e.g. 8-16 MB, matching LLC) with parameterized stride. Strides at and around the cache line size, page size, and TLB entry stride probe different locality regimes. Differs from `mem_stride_sweep_large` by being sized to fit in or just exceed LLC.

**Expected memory pattern:**
- Small-stride: cache-line reuse, low DRAM traffic.
- Cache-line stride (64 bytes): every access causes a fill.
- Page stride (4 KB): every access also causes TLB miss.
- Larger strides: revisits begin and reuse appears at the access-period boundary.

**Expected difference from existing tests:**
- Compared with `cache_hot_loop`, the access pattern is parameterized.
- Compared with `cache_cold_scan`, the buffer size targets cache boundary, not DRAM.
- Compared with `mem_stride_sweep_large`, this version probes the cache hierarchy specifically.

**Metrics likely affected:**
- Per-stride variation in `dc_coherence` and `cep_periodicity_score`.
- Dramatic shift between strides that fit in L1 versus L2 versus exceed LLC.

**Potential confounds:**
- Cache associativity effects: certain strides cause associativity conflict misses.
- THP changes effective stride at TLB level.

**Expected confusions:**
- `cache_stride_sweep (page stride) → mem_stream`
  - Type: behavioral proximity
  - Reason: identical mechanism at different working-set scale.
- `cache_stride_sweep (cache-line stride) → cache_cold_scan`
  - Type: behavioral proximity.

**Low-level control requirements:**
- Buffer size variants: 4 MB (L2), 16 MB (LLC), 64 MB (exceeds LLC).
- Stride parameter set: 8 B, 64 B, 256 B, 4 KB, 64 KB, 1 MB.
- C inner loop with stride as a constant per variant for compiler-friendly code.
- `madvise(MADV_NOHUGEPAGE)` to keep stride-vs-page-size relationship clean.

**Python/runtime artifact risk:** High in Python.

**OS/kernel artifact risk:** Medium.

**VM/capture artifact risk:** Medium.

**Cleanliness expectation:** Per-stride provisionally medium. Each stride is a separate scientific question.

**Suggested implementation level:** C strongly recommended.

---

# 6. THREAD Family

## Family Explanation

Concurrency and synchronization. Multi-threaded workloads with shared state, locks, or producer-consumer patterns. Phase 1 did not analyze thread workloads directly; the audit findings about Python's GIL, allocator, and scheduling extend here.

## Why THREAD Matters

Real workloads are often concurrent. Threaded workloads expose synchronization-driven bursts and may produce less stationary signatures than single-threaded tests. They test whether the metric pipeline handles non-stationary, irregular signal shapes.

---

## 6.1 `thread_lock_contention`

**Test name:** `thread_lock_contention`

**Status:** Proposed

**Behavior family:** THREAD

**Subtype:** Lock contention pressure

**Mechanism stressed:**
N threads (N = guest CPU count) each repeatedly acquire and release a single shared mutex, performing a small bounded operation under the lock. The operation includes one write to a shared cache line. Stresses kernel futex traffic, scheduler wake-ups, and inter-CPU cache-line ping-pong.

**Expected memory pattern:**
- Inter-CPU cache-line ping-pong on the shared variable.
- Frequent kernel futex syscalls when lock is contended.
- Short syscall storms with irregular spacing.
- Per-snapshot delta dominated by the shared cache line and per-thread stack frames.

**Expected difference from existing tests:**
- Compared with `mem_alloc_touch_pages`, the syscall pattern is futex-driven, not mmap-driven.
- Compared with `cache_hot_loop`, this involves multiple CPUs and cross-core traffic.
- Compared with single-threaded MEM/CPU tests, signal includes scheduler wake-up bursts.

**Metrics likely affected:**
- `event_rate` moderate-high.
- `Fano factor` high (bursty due to contention).
- `cepstral` content broadband.
- High CV across runs (scheduler nondeterminism).

**Potential confounds:**
- Thread placement on physical cores affects ping-pong cost.
- Kernel futex implementation details.
- Python's GIL serializes Python-level execution; for a Python implementation, true contention is impossible at the bytecode level. **Python is fundamentally inappropriate for this test.**

**Expected confusions:**
- `thread_lock_contention → io_seq_fsync`
  - Type: behavioral proximity
  - Reason: both produce kernel-mediated rhythmic activity at irregular cadence.
- `thread_lock_contention → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both produce many short syscalls with batch-like structure.

**Low-level control requirements:**
- C with `pthread_mutex_t` or futex syscalls directly.
- N threads = guest CPU count; pinned to distinct cores.
- Shared variable on one cache line, isolated from other data.
- Bounded inner-critical-section duration (e.g. 100 ns).

**Python/runtime artifact risk:** Disqualifying. GIL prevents true contention; Python implementation does not measure the intended mechanism.

**OS/kernel artifact risk:** High (intended). Kernel futex and scheduler are part of the signal.

**VM/capture artifact risk:** High. Irregular short syscalls land snapshot pauses unpredictably.

**Cleanliness expectation:** Provisionally low (CV expected to be high due to scheduler nondeterminism).

**Suggested implementation level:** C strongly recommended. Python disqualified.

---

## 6.2 `thread_producer_consumer`

**Test name:** `thread_producer_consumer`

**Status:** Proposed

**Behavior family:** THREAD

**Subtype:** Producer-consumer queue coordination

**Mechanism stressed:**
One producer thread and one consumer thread connected by a bounded ring buffer protected by a condition variable or semaphore. The producer fills bytes into the ring; the consumer reads them out. Stresses inter-thread coordination and queue cache-line traffic.

**Expected memory pattern:**
- Ring buffer pages repeatedly dirtied by producer.
- Condition variable / queue head and tail counters cause cache-line ping-pong.
- More structured than `thread_lock_contention`; coordinated rhythm.

**Expected difference from existing tests:**
- Compared with `thread_lock_contention`, the threads cooperate rather than fight.
- Compared with `mem_stream`, multi-threaded with cross-CPU traffic.
- Compared with `net_tcp_loopback_stream`, similar producer-consumer pattern but no kernel network buffers.

**Metrics likely affected:**
- `cep_periodicity_score` may be elevated at queue-fill cadence.
- `event_rate` moderate.
- More periodic than `thread_lock_contention`.

**Potential confounds:**
- Buffer size relative to L1/L2 cache.
- Condition variable wakeup cadence depends on scheduler and sleep granularity.
- Python `queue.Queue` adds significant overhead; Python disqualified for the same GIL reasons.

**Expected confusions:**
- `thread_producer_consumer → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: both have batch-rhythmic structure.
- `thread_producer_consumer → net_tcp_loopback_stream`
  - Type: behavioral proximity
  - Reason: similar producer-consumer rhythm; differ in kernel mediation.

**Low-level control requirements:**
- C with `pthread` + `pthread_cond_t` or futex.
- Ring buffer size parameter (small to fit in cache, large to exceed).
- Producer/consumer pinned to distinct cores.

**Python/runtime artifact risk:** Disqualifying.

**OS/kernel artifact risk:** High.

**VM/capture artifact risk:** Medium-high.

**Cleanliness expectation:** Provisionally medium.

**Suggested implementation level:** C strongly recommended.

---

## 6.3 `thread_parallel_alloc`

**Test name:** `thread_parallel_alloc`

**Status:** Proposed

**Behavior family:** THREAD

**Subtype:** Concurrent allocation churn

**Mechanism stressed:**
Multi-threaded extension of `mem_alloc_touch_pages`. N threads each run their own batched allocate-touch-release loop. Stresses concurrent allocator behavior, kernel mmap_sem (now `mmap_lock`), and parallel page-fault handling.

**Expected memory pattern:**
- Parallel page-fault storms.
- Allocator contention (glibc arena per-thread, kernel `mmap_lock` for direct mmap).
- More bursty than single-threaded version; less regular cadence.

**Expected difference from existing tests:**
- Compared with `mem_alloc_touch_pages`, parallel; allocator scaling visible.
- Compared with `thread_lock_contention`, contention is on kernel mmap_lock, not user mutex.

**Metrics likely affected:**
- `event_rate` higher than single-threaded version.
- `Fano factor` higher (bursty parallel storms).
- `cep_periodicity_score` weaker than single-threaded (parallel cycles overlap).

**Potential confounds:**
- glibc arena-per-thread changes the per-thread allocator path.
- Kernel `mmap_lock` contention varies with NUMA and thread count.

**Expected confusions:**
- `thread_parallel_alloc → mem_alloc_touch_pages`
  - Type: behavioral proximity
  - Reason: same fundamental mechanism with concurrent execution.

**Low-level control requirements:**
- C with `pthread`. Each thread issues direct `mmap`/`munmap`.
- Thread count parameter.
- Optional: per-thread arena via `MAP_PRIVATE` only (no shared mappings).

**Python/runtime artifact risk:** Disqualifying (GIL).

**OS/kernel artifact risk:** Very high (intended).

**VM/capture artifact risk:** High.

**Cleanliness expectation:** Provisionally medium.

**Suggested implementation level:** C strongly recommended.

---

# 7. NETWORK Family

## Family Explanation

Network I/O over loopback. The host network stack mediates all traffic. Tests packet-like timing and kernel network buffer activity that differ from filesystem I/O patterns.

## Why NETWORK Matters

Adds a non-filesystem I/O family to test whether observed IO-class signatures generalize beyond disk. Distinguishing fsync rhythm from network-buffer rhythm is a key generalization test for the semantic-free thesis.

---

## 7.1 `net_tcp_loopback_stream`

**Test name:** `net_tcp_loopback_stream`

**Status:** Proposed

**Behavior family:** NETWORK

**Subtype:** TCP loopback bulk stream

**Mechanism stressed:**
Producer process opens a TCP socket on `127.0.0.1`, connects to a consumer process, and streams large buffers (e.g. 1 MB) sequentially. The kernel TCP stack mediates with congestion windows, ACK cadence, socket buffers.

**Expected memory pattern:**
- Kernel socket buffers (`sk_buff`) churn on each transfer.
- Producer process page cache (sendmsg) and consumer page cache (recvmsg) both contribute.
- Strong throughput rhythm if congestion window is steady.
- Different cadence from `io_seq_fsync` because no disk barrier.

**Expected difference from existing tests:**
- Compared with `io_seq_fsync`, no disk; cadence is TCP-driven.
- Compared with `mem_stream`, kernel-mediated rather than user buffer.
- Compared with `thread_producer_consumer`, mediated by kernel network stack.

**Metrics likely affected:**
- `cep_periodicity_score` moderate.
- `dc_coherence` elevated.
- `event_rate` high.

**Potential confounds:**
- Loopback network performance is host-CPU-bound, not real network.
- TCP congestion control varies with kernel version.
- Socket buffer sizes (`SO_SNDBUF`, `SO_RCVBUF`) affect cadence.

**Expected confusions:**
- `net_tcp_loopback_stream → io_seq_fsync`
  - Type: behavioral proximity
  - Reason: both produce streaming kernel-mediated activity with rhythmic structure.
- `net_tcp_loopback_stream → thread_producer_consumer`
  - Type: behavioral proximity
  - Reason: both have producer-consumer cadence; differ by kernel mediation.

**Low-level control requirements:**
- C, two processes (producer + consumer) on the same VM.
- Fixed socket buffer sizes via `setsockopt`.
- TCP_NODELAY enabled or disabled (variant choice).
- Document loopback driver and TCP congestion algorithm.

**Python/runtime artifact risk:** Low (kernel-bound).

**OS/kernel artifact risk:** Very high.

**VM/capture artifact risk:** Medium.

**Cleanliness expectation:** Provisionally medium-high.

**Suggested implementation level:** C recommended.

---

## 7.2 `net_udp_burst`

**Test name:** `net_udp_burst`

**Status:** Proposed

**Behavior family:** NETWORK

**Subtype:** UDP burst traffic

**Mechanism stressed:**
Producer sends UDP datagrams at burst rate (e.g. 10000 per second of 1 KB each) to a consumer on the same host. No congestion control, no acknowledgment. Stresses kernel UDP socket queue and may produce drops.

**Expected memory pattern:**
- Bursty kernel socket buffer activity.
- High packet rate; potentially drops if consumer cannot keep up.
- Less periodic than TCP because no congestion window control loop.

**Expected difference from existing tests:**
- Compared with `net_tcp_loopback_stream`, less rhythmic.
- Compared with `io_rand_rw`, network buffers instead of page cache.
- Compared with `io_many_files`, no filesystem.

**Metrics likely affected:**
- `Fano factor` high.
- `event_rate` high.
- `spectral_entropy` higher than TCP variant.

**Potential confounds:**
- UDP drops at consumer side reduce signal magnitude irregularly.
- Loopback driver behavior.

**Expected confusions:**
- `net_udp_burst → io_rand_rw`
  - Type: behavioral proximity
  - Reason: both produce irregular high-rate kernel-mediated activity.

**Low-level control requirements:**
- C with `sendmsg`/`recvmsg`.
- Fixed packet rate via `nanosleep`.
- Pre-allocated send and receive buffers.

**Python/runtime artifact risk:** Low.

**OS/kernel artifact risk:** Very high.

**VM/capture artifact risk:** Medium-high.

**Cleanliness expectation:** Provisionally medium.

**Suggested implementation level:** C recommended.

---

## 7.3 `net_many_small_messages`

**Test name:** `net_many_small_messages`

**Status:** Proposed

**Behavior family:** NETWORK

**Subtype:** Many small TCP messages

**Mechanism stressed:**
Producer opens a TCP socket and sends many small messages (e.g. 64 bytes each, 50000 per second). Consumer receives and discards. Stresses per-message syscall overhead and small-buffer kernel activity, analogous to `io_many_files` but for the network stack.

**Expected memory pattern:**
- High syscall rate (`sendmsg`, `recvmsg`).
- Many small kernel `sk_buff` allocations.
- Strong batch rhythm if messages are sent in micro-batches.

**Expected difference from existing tests:**
- Compared with `net_tcp_loopback_stream`, message granularity is small; per-message overhead dominates.
- Compared with `io_many_files`, network-buffer rather than VFS object lifecycle.

**Metrics likely affected:**
- `event_rate` very high.
- `cep_periodicity_score` moderate-strong.
- `spectral_entropy` lower than UDP burst (TCP imposes order).

**Potential confounds:**
- Nagle's algorithm coalesces small writes; `TCP_NODELAY` may be needed for true small-message behavior.
- Kernel slab allocator pressure on `sk_buff` cache.

**Expected confusions:**
- `net_many_small_messages → io_many_files`
  - Type: behavioral proximity
  - Reason: both stress kernel small-object allocation at high rate; differ by subsystem.
- `net_many_small_messages → net_tcp_loopback_stream`
  - Type: behavioral proximity
  - Reason: same protocol, differ in granularity.

**Low-level control requirements:**
- C, `setsockopt(TCP_NODELAY)` enabled.
- Fixed message size and rate.
- Documented kernel slab cache state at start.

**Python/runtime artifact risk:** Low-medium.

**OS/kernel artifact risk:** Very high.

**VM/capture artifact risk:** High (many short syscalls).

**Cleanliness expectation:** Provisionally medium.

**Suggested implementation level:** C recommended.

---

# 8. MIXED Family

## Family Explanation

Intentionally combined workloads. Each MIXED test specifies the exact combination and the expected position in feature space relative to the parent classes.

## Why MIXED Matters

Real workloads do not isolate a single mechanism. MIXED tests probe whether the feature space behaves continuously: do mixed workloads sit between the parent clusters, or do they produce a third cluster, or do they collapse to one parent's cluster?

The MIXED tests are diagnostic: they test the assumption that the feature space is continuous and that classifier decisions reflect the underlying mechanism mixture, not arbitrary thresholds.

---

## 8.1 `mixed_mem_io`

**Test name:** `mixed_mem_io`

**Status:** Proposed

**Behavior family:** MIXED

**Subtype:** Concurrent memory pressure and file I/O

**Mechanism stressed:**
Two threads (or alternating phases): one runs `mem_random_write_pages` (or `mem_stream` write variant) on anonymous pages. The other runs `io_rand_rw` on a page-cache-backed file. The combination tests whether the metric pipeline can resolve mixed signals.

**Expected memory pattern:**
- Both anonymous and file-cache page changes per snapshot.
- Roughly the union of `mem_*` and `io_*` signal characteristics.
- May or may not sit between MEM and IO clusters in feature space.

**Expected difference from existing tests:**
- Compared with `mem_random_write_pages`, this includes file-cache contribution.
- Compared with `io_rand_rw`, this includes anonymous-mapping contribution.

**Metrics likely affected:**
- `event_rate` high (sum of components).
- `active_page_fraction` high.
- Position in feature space depends on which component dominates.

**Potential confounds:**
- One component may dominate the other due to relative throughput.
- Memory pressure from MEM component may evict page-cache pages of the IO component.

**Expected confusions:**
- `mixed_mem_io → mem_random_write_pages`
  - Type: behavioral proximity
  - Reason: if the MEM component's page-write rate exceeds the IO component's, the feature vector resembles MEM.
- `mixed_mem_io → io_rand_rw`
  - Type: behavioral proximity
  - Reason: same logic, IO-dominant case.

**Low-level control requirements:**
- C, two threads pinned to distinct cores.
- Configurable rate ratio between MEM and IO components.
- Document the relative throughput at measurement time.

**Python/runtime artifact risk:** Disqualifying due to GIL.

**OS/kernel artifact risk:** High.

**VM/capture artifact risk:** Medium-high.

**Cleanliness expectation:** Provisionally medium. Used as a feature-space-continuity probe, not a clean subtype.

**Suggested implementation level:** C strongly recommended.

---

## 8.2 `mixed_cpu_mem`

**Test name:** `mixed_cpu_mem`

**Status:** Proposed

**Behavior family:** MIXED

**Subtype:** Concurrent compute and memory pressure

**Mechanism stressed:**
One thread runs `cpu_hash_loop` (CPU-bound, near-invisible). Another runs `mem_stream` write variant. Tests whether the CPU-bound component contributes any observable signal alongside an active memory workload.

**Expected memory pattern:**
- Dominated by the MEM component because the CPU component is near-invisible.
- May resemble `mem_stream` alone, slightly suppressed by CPU contention for memory bandwidth.

**Expected difference from existing tests:**
- Compared with `mem_stream`, the CPU thread reduces effective memory throughput.
- Compared with `cpu_hash_loop`, the MEM signal dominates.

**Metrics likely affected:**
- Most metrics resemble `mem_stream` alone.
- `dc_coherence` may be reduced if CPU contention slows the stream.

**Potential confounds:**
- Memory bandwidth contention.
- Cache competition.

**Expected confusions:**
- `mixed_cpu_mem → mem_stream`
  - Type: behavioral proximity (very high)
  - Reason: CPU component is invisible; signal IS the MEM component.

**Low-level control requirements:**
- C, two threads pinned to distinct cores.
- Document throughput rates.

**Python/runtime artifact risk:** Disqualifying.

**OS/kernel artifact risk:** Low-medium.

**VM/capture artifact risk:** Medium.

**Cleanliness expectation:** Provisionally medium-high (resembles `mem_stream`).

**Suggested implementation level:** C strongly recommended.

---

## 8.3 `mixed_cpu_io`

**Test name:** `mixed_cpu_io`

**Status:** Proposed

**Behavior family:** MIXED

**Subtype:** Concurrent compute and file I/O

**Mechanism stressed:**
One thread runs `cpu_hash_loop`. Another runs `io_seq_fsync`. Tests whether the CPU-bound component contributes any observable signal alongside an active IO workload.

**Expected memory pattern:**
- Dominated by the IO component.
- IO cadence may be slightly perturbed by CPU contention if both threads share a core.

**Expected difference from existing tests:**
- Compared with `io_seq_fsync` alone, possibly slightly noisier rhythm.
- Compared with `cpu_hash_loop`, IO dominates.

**Metrics likely affected:**
- Most metrics resemble `io_seq_fsync` alone.
- `cep_periodicity_score` may be slightly weakened by CPU competition.

**Potential confounds:**
- CPU contention with kernel writeback threads.

**Expected confusions:**
- `mixed_cpu_io → io_seq_fsync`
  - Type: behavioral proximity (very high)
  - Reason: CPU component is invisible; signal IS the IO component.

**Low-level control requirements:**
- C, two threads pinned to distinct cores.
- Document throughput rates.

**Python/runtime artifact risk:** Disqualifying.

**OS/kernel artifact risk:** Very high.

**VM/capture artifact risk:** Medium-high.

**Cleanliness expectation:** Provisionally medium-high.

**Suggested implementation level:** C strongly recommended.

---

# Implementation and Validation Notes

## Recommended Collection Strategy

For every subtype:
- Collect at least 8 to 10 repeated runs (per `confusion_matrix_diagnostic_methodology.md` Section 6 provisional language rule).
- Randomize execution order across cycles.
- Apply cache-drop and longer idle windows around active workloads when possible.
- Record exact parameters used.
- Record prior-workload identity in metadata sidecar for every recording.

## Recommended Minimum Repeats

Provisional cleanliness designations require at least 8 runs per subtype. 4 runs per subtype is exploratory only.

## Validation Strategy

For each family:
1. Compute metrics relative to `idle_long_baseline`, not `run_idle`.
2. Check within-subtype CV across runs and across segments.
3. Check between-subtype distances.
4. Apply the four-cause model to every off-diagonal confusion.
5. Use group-aware (run-level) cross-validation, not segment-level.
6. Run leave-one-recording-out LDA validation.
7. Bootstrap confidence intervals for any reported effect size.
8. Document direction with `true → predicted` notation.

## Promotion Rule

A proposed test becomes a current test only after:
- Implementation exists at the suggested level (Python, C, or assembly).
- At least 8 to 10 repeated runs collected.
- Provisional cleanliness assessment completed using the four-cause model.
- Expected confusions confirmed or updated based on actual data.
- Confounds documented with `/proc`-level evidence per `python_runtime_os_artifact_audit.md` instrumentation guidance.

---

# Final Principle

Every workload is a controlled stimulus, not a script. The objective is to build a behavioral map of volatile-memory signal space using probes whose mechanisms are known and whose artifacts are characterized.

When two probes overlap in feature space, the four-cause model decides whether the overlap reflects genuine mechanism similarity, a metric inadequacy, an OS artifact, a Python artifact, a capture artifact, or execution-order contamination. Each cause has a different remediation path. The mini-spec fields above are designed to make that diagnosis tractable from the design phase forward.
