# Next-Generation Workload Test Mini-Specs

This document defines both:

1. **Existing workload tests already used**
2. **Planned next-generation workload tests to be implemented later**

The goal is to build a systematic taxonomy of controlled workload generators for semantic-free volatile-memory behavior classification.

Each workload should be treated as a **black-box signal generator**. The purpose is not to identify the executable by name, but to test whether the memory signal contains repeatable and separable behavioral structure.

---

## Mini-Spec Template

Every workload test should eventually be described using this structure:

```text
Test name:
Behavior family:
Subtype:
Mechanism stressed:
Expected memory pattern:
Expected difference from existing tests:
Metrics likely affected:
Potential confounds:
```

---

## Experimental Logic

A useful workload test should help answer one or more of these questions:

- Does this behavior produce a repeatable memory signature?
- Does it separate from other behaviors in feature space?
- Does it expose a specific system mechanism?
- Does it help explain whether classification depends on intensity, locality, periodicity, randomness, synchronization, or mixed behavior?

A strong behavioral fingerprint requires both:

1. **Repeatability**  
   Repeated runs of the same subtype should remain close.

2. **Separability**  
   Different subtypes should move apart in descriptor space.

The long-term goal is to build a behavioral map of volatile-memory signal space.

---

# 1. IDLE Family

## Family Explanation

The IDLE family represents baseline/control behavior.

Idle tests are used to understand background VM behavior when no synthetic workload is intentionally stressing the system.

However, IDLE should not be treated as a perfect zero-signal condition. Idle windows may still include:

- OS background activity
- scheduler noise
- cache cleanup
- delayed writeback
- effects from previous workloads
- VM/host-level interference

## Why IDLE Matters

IDLE is the baseline against which active workloads are interpreted. It helps answer:

> What does background volatile-memory behavior look like, and how stable is it?

Future IDLE variants may test whether baseline behavior changes depending on duration, placement in the test cycle, or previous workload history.

---

## Existing / Already Used Tests

### Test name: `run_idle`

**Status:** Already used

**Behavior family:**  
IDLE

**Subtype:**  
Idle baseline / control window

**Mechanism stressed:**  
No intentional synthetic workload. The VM is allowed to remain mostly inactive while natural background system activity continues.

**Expected memory pattern:**  
- Low event rate
- Low SNR activity
- Weak or inconsistent periodic structure
- Background-level spectral and cepstral behavior
- Possible small bursts from OS daemons, scheduler activity, cache cleanup, or previous workloads

**Expected difference from existing tests:**  
Should differ strongly from active MEM and IO workloads because it should have lower signal intensity and fewer structured memory events.

However, it may not be perfectly stable.

**Metrics likely affected:**  
- event_rate
- snr_median / snr_mean
- snr_zero_frac
- cepstral energy / cepstral variance
- MSC low-frequency coherence
- spectral entropy
- coefficient of variation across idle windows

**Potential confounds:**  
- Residual effects from previous workload
- Page cache cleanup
- Background OS activity
- VM scheduling noise
- Host system interference
- First idle baseline may differ from later idle segments

---

## Planned / New IDLE Tests

### Test name: `idle_long_baseline`

**Status:** Proposed

**Behavior family:**  
IDLE

**Subtype:**  
Long idle baseline

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
TODO

**Expected difference from existing tests:**  
TODO

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `idle_post_workload_recovery`

**Status:** Proposed

**Behavior family:**  
IDLE

**Subtype:**  
Post-workload recovery idle

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
TODO

**Expected difference from existing tests:**  
TODO

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

# 2. MEM Family

## Family Explanation

The MEM family directly stresses volatile memory behavior.

MEM tests are designed to differ in:

- access order
- locality
- page-touching pattern
- allocation behavior
- burst structure
- randomness vs regularity

## Why MEM Matters

MEM tests help answer:

> Can memory-derived metrics distinguish different memory access structures, not just detect “more memory activity”?

This family is central to the thesis because it tests whether volatile-memory signatures encode access behavior such as streaming, random traversal, or allocation churn.

---

## Existing / Already Used Tests

### Test name: `mem_stream`

**Status:** Already used

**Behavior family:**  
MEM

**Subtype:**  
Sequential memory streaming

**Mechanism stressed:**  
Large-buffer sequential or page-strided memory traversal. The workload repeatedly touches memory in a regular order.

**Expected memory pattern:**  
- Regular memory-access structure
- Predictable page-touch behavior
- Potentially stronger periodic or quasi-periodic components
- Moderate-to-high event activity
- Lower randomness than pseudo-random traversal

**Expected difference from existing tests:**  
Compared with `mem_pointer_chase`, it should be more regular and locality-preserving.

Compared with `mem_alloc_touch_pages`, it should stress traversal of existing memory more than allocation/deallocation churn.

Compared with IO tests, it should produce memory pressure without filesystem metadata or synchronization behavior.

**Metrics likely affected:**  
- event_rate
- cepstral peak strength
- cepstral mean / variance
- autocorrelation decay
- spectral entropy
- MSC frequency profile
- SNR median / tail behavior
- permutation entropy

**Potential confounds:**  
- CPU cache effects
- Hardware prefetching
- Page size and stride alignment
- Python interpreter overhead
- VM memory ballooning or host paging
- Buffer size relative to cache and available RAM

---

### Test name: `mem_pointer_chase`

**Status:** Already used

**Behavior family:**  
MEM

**Subtype:**  
Pseudo-random memory traversal / poor-locality memory access

**Mechanism stressed:**  
Memory-intensive pointer-chasing-like access pattern with reduced locality and less predictable page traversal.

**Expected memory pattern:**  
- Irregular memory-access ordering
- Weaker periodic structure than `mem_stream`
- Higher apparent randomness
- Faster autocorrelation decay
- Potentially higher entropy
- Less coherent mid-frequency behavior

**Expected difference from existing tests:**  
Compared with `mem_stream`, this test should reduce locality and regularity.

Compared with `mem_alloc_touch_pages`, it should stress irregular access to allocated memory rather than repeated allocation/release phases.

Compared with IO tests, it should remain memory-bound rather than filesystem-bound.

**Metrics likely affected:**  
- permutation entropy
- sample entropy
- spectral entropy
- autocorrelation decay / correlation length
- MSC mid-frequency coherence
- cepstral peak stability
- SNR kurtosis / skewness
- burstiness index

**Potential confounds:**  
- Randomization method may still produce repeatable pseudo-random structure
- CPU cache and TLB behavior
- Python loop overhead
- Memory allocator layout
- Page faults if memory pressure exceeds available RAM
- Host/VM scheduling effects

---

### Test name: `mem_alloc_touch_pages`

**Status:** Already used

**Behavior family:**  
MEM

**Subtype:**  
Allocation churn + page touching

**Mechanism stressed:**  
Repeated allocation, page touching, and release/clearing of memory objects.

**Expected memory pattern:**  
- Burst-like allocation phases
- Repeated allocate-touch-release structure
- Stronger batch pattern than simple streaming
- Possible page fault / page commitment behavior
- Potentially repeatable temporal phases if loop structure is stable

**Expected difference from existing tests:**  
Compared with `mem_stream`, this test stresses allocator behavior and memory lifecycle rather than simple traversal.

Compared with `mem_pointer_chase`, it should be less about irregular pointer-like access and more about repeated allocation phases.

Compared with IO tests, it should produce memory churn without direct filesystem pressure.

**Metrics likely affected:**  
- burstiness / Fano factor
- event_rate
- SNR peak / SNR tail metrics
- cepstral variance
- cepstral peak strength
- wavelet or multiscale energy distribution
- autocorrelation at batch periods
- coefficient of variation across repeated runs

**Potential confounds:**  
- Python garbage collection behavior
- Memory allocator behavior
- OS page allocator state
- Transparent huge pages
- Swap or memory pressure on host
- Residual effects from previous test cycle
- Object size and batch size choices

---

## Planned / New MEM Tests

### Test name: `mem_random_write_pages`

**Status:** Proposed

**Behavior family:**  
MEM

**Subtype:**  
Random page write memory pressure

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
TODO

**Expected difference from existing tests:**  
TODO

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `mem_stride_sweep_large`

**Status:** Proposed

**Behavior family:**  
MEM

**Subtype:**  
Stride-controlled memory traversal

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
TODO

**Expected difference from existing tests:**  
TODO

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

# 3. IO Family

## Family Explanation

The IO family stresses the filesystem and storage path.

IO workloads affect volatile memory indirectly through:

- page cache activity
- dirty page management
- metadata structures
- file object churn
- synchronization behavior
- kernel buffering

## Why IO Matters

IO tests help answer:

> Can volatile-memory metrics distinguish different filesystem and storage behaviors?

They also test whether observed IO structure is specific to filesystem behavior or generalizable to other forms of system activity.

---

## Existing / Already Used Tests

### Test name: `io_rand_rw`

**Status:** Already used

**Behavior family:**  
IO

**Subtype:**  
Random read/write file I/O

**Mechanism stressed:**  
Random block-oriented reads and writes against a file, typically with mixed read/write operations at random offsets.

**Expected memory pattern:**  
- Irregular I/O bursts
- Page-cache activity
- Mixed dirty/clean cache behavior
- Less predictable timing than sequential I/O
- Potentially higher variability across runs than structured IO subtypes

**Expected difference from existing tests:**  
Compared with `io_seq_fsync`, this test should be less rhythmic and less sequential.

Compared with `io_many_files`, it stresses data block access more than metadata/file-object churn.

Compared with MEM tests, it should reflect filesystem/cache behavior rather than direct memory traversal.

**Metrics likely affected:**  
- burstiness / Fano factor
- spectral entropy
- SNR variance / SNR kurtosis
- MSC frequency profile
- event clustering
- autocorrelation decay
- cepstral peak instability
- coefficient of variation across repeated runs

**Potential confounds:**  
- Page cache may absorb reads/writes
- Storage backend behavior
- File preallocation state
- Random seed / access sequence
- Writeback timing
- Host filesystem caching
- VM disk configuration
- Prior run cache contamination

---

### Test name: `io_seq_fsync`

**Status:** Already used

**Behavior family:**  
IO

**Subtype:**  
Sequential write with forced synchronization

**Mechanism stressed:**  
Sequential writes combined with periodic or frequent `fsync` calls, forcing synchronization of dirty data.

**Expected memory pattern:**  
- More regular I/O rhythm than random I/O
- Repeated write/sync phases
- Possible periodic flush behavior
- Stronger temporal structure than `io_rand_rw`
- Stable subtype signature if synchronization cadence is consistent

**Expected difference from existing tests:**  
Compared with `io_rand_rw`, this test should be more sequential and rhythmically structured.

Compared with `io_many_files`, it stresses bulk sequential data synchronization rather than filesystem metadata churn.

Compared with MEM tests, it should reflect kernel buffering and disk synchronization rather than direct user-space memory pressure.

**Metrics likely affected:**  
- cepstral peak strength
- autocorrelation at sync cadence
- spectral peaks / lower spectral entropy
- MSC frequency profile
- SNR event statistics
- event_rate
- wavelet scale energy near sync bursts
- coefficient of variation across repeated runs

**Potential confounds:**  
- `fsync` behavior depends on storage backend
- Page cache and dirty-page writeback policy
- Disk/SSD latency variability
- Host filesystem behavior
- VM storage driver
- Chunk size and fsync interval
- Background writeback activity

---

### Test name: `io_many_files`

**Status:** Already used

**Behavior family:**  
IO

**Subtype:**  
Many-small-files / metadata-heavy I/O

**Mechanism stressed:**  
Creation, writing, and deletion of many small files, stressing filesystem metadata and object management.

**Expected memory pattern:**  
- Metadata-heavy kernel activity
- Directory entry / inode / dentry cache effects
- Small repeated bursts
- Stronger filesystem-object churn than bulk data movement
- Potentially stable subtype signature if batch structure is consistent

**Expected difference from existing tests:**  
Compared with `io_seq_fsync`, this test stresses many small metadata operations rather than sequential synchronized writes.

Compared with `io_rand_rw`, it stresses file creation/deletion and metadata churn rather than random block access.

Compared with MEM tests, it should produce filesystem-mediated memory behavior rather than direct memory traversal.

**Metrics likely affected:**  
- event_rate
- burstiness / Fano factor
- SNR event statistics
- MSC frequency profile
- cepstral variance
- wavelet/multiscale energy distribution
- entropy of event timing
- coefficient of variation across repeated runs

**Potential confounds:**  
- Filesystem type
- Directory cache and inode cache state
- Whether files are deleted or kept
- Temporary directory location
- Page cache state
- Host filesystem behavior
- Background journaling/writeback
- File count and file size choices

---

## Planned / New IO Tests

### Test name: `io_read_cache_hit`

**Status:** Proposed

**Behavior family:**  
IO

**Subtype:**  
Cached read-heavy I/O

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
TODO

**Expected difference from existing tests:**  
TODO

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `io_direct_write_like`

**Status:** Proposed

**Behavior family:**  
IO

**Subtype:**  
Reduced-cache / direct-write-style I/O

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
TODO

**Expected difference from existing tests:**  
TODO

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

# 4. CPU Family

## Family Explanation

The CPU family stresses computation rather than direct memory pressure or filesystem I/O.

These workloads are useful because active computation may still alter volatile-memory behavior through instruction execution, cache usage, stack activity, temporary objects, branch behavior, and interpreter/runtime activity.

## Expected Family Behavior

- High compute activity
- Lower I/O
- Potentially lower memory-event rate than MEM
- Different spectral/SNR pattern from direct memory-pressure workloads

## Why CPU Matters

This family helps answer:

> Can memory behavior distinguish CPU-bound work from memory-bound work?

It also tests whether active workloads that do not intentionally stress memory still leave distinct volatile-memory signatures.

---

## Planned CPU Tests

### Test name: `cpu_hash_loop`

**Status:** Proposed

**Behavior family:**  
CPU

**Subtype:**  
Hash/computation loop

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
High compute activity with limited explicit memory pressure; possible stable loop rhythm and moderate cache/register-local behavior.

**Expected difference from existing tests:**  
Should differ from MEM tests by producing less direct page-level memory pressure and from IO tests by producing little filesystem activity.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `cpu_matrix_mult`

**Status:** Proposed

**Behavior family:**  
CPU

**Subtype:**  
Dense numeric computation

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
High compute activity with structured memory reuse; may show different cache/memory behavior depending on matrix size.

**Expected difference from existing tests:**  
Should sit between pure CPU and memory-locality behavior if matrix size stresses cache or RAM.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `cpu_branch_random`

**Status:** Proposed

**Behavior family:**  
CPU

**Subtype:**  
Branch-heavy irregular computation

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
High compute activity with more irregular control flow; may show less periodicity than hash or matrix loops.

**Expected difference from existing tests:**  
Should differ from regular CPU loops by adding branch unpredictability without becoming a memory or I/O workload.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

# 5. CACHE Family

## Family Explanation

The CACHE family stresses memory locality and cache behavior.

These tests are close to the core thesis because they help determine whether the extracted memory metrics detect access structure, rather than merely detecting “more memory activity.”

## Expected Family Behavior

- `cache_hot_loop`: repeatedly touches small memory region
- `cache_cold_scan`: scans larger-than-cache region
- `cache_stride_sweep`: accesses memory with controlled stride patterns

## Why CACHE Matters

This family helps explain whether MEM separability is driven by:

- memory volume
- access locality
- stride periodicity
- randomness

It directly tests whether memory metrics capture access structure.

---

## Planned CACHE Tests

### Test name: `cache_hot_loop`

**Status:** Proposed

**Behavior family:**  
CACHE

**Subtype:**  
Hot-cache small working set

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Repeated touches to a small memory region; likely stable and low-entropy compared with cold scans.

**Expected difference from existing tests:**  
Should differ from `mem_stream` by keeping activity in a small working set instead of sweeping a large region.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `cache_cold_scan`

**Status:** Proposed

**Behavior family:**  
CACHE

**Subtype:**  
Cold scan / larger-than-cache traversal

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Scanning of a region larger than cache; should create more cache misses and broader memory-access activity than `cache_hot_loop`.

**Expected difference from existing tests:**  
Should resemble streaming behavior but be explicitly designed around cache capacity effects.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `cache_stride_sweep`

**Status:** Proposed

**Behavior family:**  
CACHE

**Subtype:**  
Controlled stride locality test

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Accesses memory using controlled stride patterns; may create periodic or quasi-periodic signatures depending on stride.

**Expected difference from existing tests:**  
Should help distinguish stride periodicity from random pointer-chasing and simple sequential streaming.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

# 6. THREAD Family

## Family Explanation

The THREAD family stresses concurrency, synchronization, and shared-state behavior.

Real workloads are often concurrent. Threaded workloads may create messier signatures than simple single-threaded tests.

## Expected Family Behavior

- More temporal irregularity
- Synchronization bursts
- Possibly higher variance/CV
- Stronger burstiness metrics

## Why THREAD Matters

This family tests whether the method can handle more realistic behavior involving concurrency.

Concurrency likely creates less clean and more variable signatures, so it should be added after simpler families are understood.

---

## Planned THREAD Tests

### Test name: `thread_lock_contention`

**Status:** Proposed

**Behavior family:**  
THREAD

**Subtype:**  
Lock contention / synchronization pressure

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Temporal irregularity caused by contention; synchronization bursts; possible high variance in event timing.

**Expected difference from existing tests:**  
Should differ from single-threaded MEM/CPU tests by adding synchronization-driven burst structure.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `thread_producer_consumer`

**Status:** Proposed

**Behavior family:**  
THREAD

**Subtype:**  
Producer-consumer coordination

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Coordinated bursts between producer and consumer phases; potential periodic queue-like behavior.

**Expected difference from existing tests:**  
Should differ from lock contention by having more structured inter-thread exchange rather than pure contention.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `thread_parallel_alloc`

**Status:** Proposed

**Behavior family:**  
THREAD

**Subtype:**  
Parallel allocation churn

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Concurrent allocation/deallocation pressure; potentially high burstiness and allocator contention.

**Expected difference from existing tests:**  
Should extend `mem_alloc_touch_pages` into a concurrent setting.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

# 7. NETWORK Family

## Family Explanation

The NETWORK family introduces I/O behavior that is not dominated by the filesystem.

Network-like workloads may create packet-like timing, buffering, and burst structures that differ from disk/file I/O.

## Expected Family Behavior

- Packet-like burst structure
- Different timing from file I/O
- Likely different spectral/burstiness signature

## Why NETWORK Matters

This family gives another I/O family and helps avoid the claim that current IO results only apply to filesystem behavior.

It tests whether volatile-memory signatures can distinguish network-style I/O from file-system-style I/O.

---

## Planned NETWORK Tests

### Test name: `net_tcp_loopback_stream`

**Status:** Proposed

**Behavior family:**  
NETWORK

**Subtype:**  
TCP loopback stream

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Streaming packet/buffer activity over loopback; potentially regular throughput-oriented network behavior.

**Expected difference from existing tests:**  
Should differ from `io_seq_fsync` because synchronization is network-buffer driven rather than filesystem flush driven.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `net_udp_burst`

**Status:** Proposed

**Behavior family:**  
NETWORK

**Subtype:**  
UDP burst traffic

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Packet-like bursts with less connection-level structure than TCP; possibly high burstiness.

**Expected difference from existing tests:**  
Should differ from random file I/O by creating network-buffer activity rather than page-cache block activity.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `net_many_small_messages`

**Status:** Proposed

**Behavior family:**  
NETWORK

**Subtype:**  
Many small network messages

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Frequent small packet/message activity; potentially analogous to `io_many_files` but in network buffers rather than filesystem metadata.

**Expected difference from existing tests:**  
Should test whether many-small-object behavior differs between network and filesystem subsystems.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

# 8. MIXED Family

## Family Explanation

The MIXED family combines two or more workload mechanisms.

These tests are the next level after clean synthetic workloads. They test whether the feature space behaves continuously when behaviors overlap.

## Expected Family Behavior

- Overlap between known families
- May sit between clusters
- Useful for testing whether feature space behaves continuously

## Why MIXED Matters

This family helps answer:

> If behavior is mixed, does the memory signal move between known fingerprints?

Mixed workloads are important for real-world applicability because actual workloads rarely isolate one mechanism perfectly.

---

## Planned MIXED Tests

### Test name: `mixed_mem_io`

**Status:** Proposed

**Behavior family:**  
MIXED

**Subtype:**  
Combined memory pressure and file I/O

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Combination of memory-access structure and filesystem/page-cache behavior; may sit between MEM and IO clusters.

**Expected difference from existing tests:**  
Should test whether feature space interpolates between MEM and IO signatures.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `mixed_cpu_mem`

**Status:** Proposed

**Behavior family:**  
MIXED

**Subtype:**  
Combined compute and memory pressure

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
High compute with direct memory pressure; may overlap CPU and MEM descriptors.

**Expected difference from existing tests:**  
Should test whether CPU-bound and memory-bound signatures combine or one dominates.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

### Test name: `mixed_cpu_io`

**Status:** Proposed

**Behavior family:**  
MIXED

**Subtype:**  
Combined compute and I/O pressure

**Mechanism stressed:**  
TODO

**Expected memory pattern:**  
Compute activity combined with filesystem/page-cache behavior; may show both periodic compute structure and I/O bursts.

**Expected difference from existing tests:**  
Should test whether CPU and IO signatures remain separable when combined.

**Metrics likely affected:**  
TODO

**Potential confounds:**  
TODO

---

# Implementation and Validation Notes

## Recommended Collection Strategy

For every subtype:

- collect repeated runs
- randomize or rotate execution order
- include idle windows before and after active tests
- avoid always running tests in the same sequence
- keep runtime and resource limits consistent
- record exact parameters used

## Recommended Minimum Repeats

Minimum:

```text
5 runs per subtype
```

Better:

```text
10 runs per subtype
```

## Validation Strategy

For each family:

1. Compute metrics relative to baseline
2. Check within-subtype CV
3. Check between-subtype distances
4. Plot PCA/MDS/LDA
5. Run leave-one-out LDA validation
6. Run centroid-margin analysis
7. Run hidden-label clustering
8. Confirm whether repeated runs cluster together

## Promotion Rule

A proposed test becomes a current/validated test only after:

- implementation exists
- repeated runs are collected
- metrics are extracted
- basic repeatability is measured
- separability is evaluated
- confounds are documented

---

# Final Principle

Do not add random tests just to increase dataset size.

Every new workload must test a specific behavioral hypothesis.

The goal is to build a structured, explainable map of volatile-memory behavior space.
