# Workload Test Mini-Specs

This document defines the current workload tests as controlled signal-generating probes for semantic-free volatile-memory behavior analysis.

The goal is not to identify programs by name. The goal is to determine whether different workload behaviors produce repeatable and separable memory-signal signatures.

Each test is described using the following template:

- Test name
- Behavior family
- Subtype
- Mechanism stressed
- Expected memory pattern
- Expected difference from existing tests
- Metrics likely affected
- Potential confounds

---

## 1. IDLE Family

### Test name: `run_idle`

**Behavior family:**  
IDLE

**Subtype:**  
Idle baseline / control window

**Mechanism stressed:**  
No intentional synthetic workload. The VM is allowed to remain mostly inactive while background operating-system and VM activity continue naturally.

**Expected memory pattern:**  
- Low event rate
- Low SNR activity
- Weak or inconsistent periodic structure
- Background-level spectral and cepstral behavior
- Possible small bursts from OS daemons, scheduler activity, cache cleanup, or delayed effects from previous workloads

**Expected difference from existing tests:**  
`run_idle` should differ most strongly from active MEM and IO workloads because it should have lower signal intensity and fewer structured memory events.

However, IDLE should not be assumed to be perfectly flat. It is a baseline regime, not a mathematically pure zero-signal state.

**Metrics likely affected:**  
- event_rate
- snr_median / snr_mean
- snr_zero_frac
- cepstral energy / cepstral variance
- MSC low-frequency coherence
- spectral entropy
- coefficient of variation across idle windows

**Potential confounds:**  
- Residual effects from previous workload in cyclic execution
- Page cache cleanup
- Background OS activity
- VM scheduling noise
- Host system interference
- Idle windows occurring after different workload types
- First idle baseline may differ from later idle segments

---

## 2. MEM Family

The MEM family is designed to stress volatile memory directly. These tests should share broad memory-intensive behavior, but their subtypes intentionally differ in locality, access regularity, allocation behavior, and temporal structure.

---

### Test name: `mem_stream`

**Behavior family:**  
MEM

**Subtype:**  
Sequential memory streaming

**Mechanism stressed:**  
Large-buffer sequential/page-strided memory traversal. The workload repeatedly touches memory in a regular order.

**Expected memory pattern:**  
- Relatively regular memory-access structure
- More predictable page-touch behavior than pointer chasing
- Potentially stronger periodic or quasi-periodic components
- Moderate-to-high event activity
- Lower randomness than pseudo-random traversal

**Expected difference from existing tests:**  
Compared with `mem_pointer_chase`, `mem_stream` should appear more regular and locality-preserving.

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
- Potentially more repeatable temporal phases if loop structure is stable

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

## 3. IO Family

The IO family is designed to stress the filesystem and storage path. These tests may affect volatile memory through page cache activity, metadata structures, dirty pages, synchronization behavior, and kernel buffering.

---

### Test name: `io_rand_rw`

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
Compared with `io_rand_rw`, this test should be more sequential and more rhythmically structured.

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

## Cross-Test Interpretation Logic

The current tests are not random scripts. They are controlled probes designed to isolate different system behaviors.

The analysis should ask:

1. Do broad families separate?
   - IDLE vs MEM vs IO

2. Do subtypes separate inside a family?
   - MEM subtypes
   - IO subtypes

3. Are repeated runs stable?
   - low within-subtype variation

4. Are different subtypes distinct?
   - high between-subtype separation

5. Do unsupervised and supervised views agree?
   - PCA/MDS geometry
   - LDA separability
   - LOOCV validation

---

## What Would Count as Strong Evidence?

A workload subtype is a strong behavioral fingerprint candidate if:

- repeated runs cluster together
- coefficient of variation is low
- distance to own subtype centroid is lower than distance to other subtype centroids
- PCA/MDS show natural proximity
- LDA separates it but does not rely only on in-sample fitting
- leave-one-out validation predicts it correctly
- the pattern persists across additional runs

---

## Current Principle for Future Test Design

Every future workload should be added only if it has a clear behavioral hypothesis.

Template for new tests:

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

The purpose is to build a systematic map of volatile-memory behavior space, not merely to collect more scripts.
