# Workload-Based Memory Dynamics Validation

## Overview

This test suite implements a **controlled, reproducible workload-driven experimental framework** designed to validate a signal-based model of **volatile memory evolution**. The core objective is to demonstrate that *distinct system behaviors* (idle, memory-intensive, and I/O-intensive execution) induce **distinct, measurable signatures** in the *memory-delta signal domain*, **without relying on OS semantics, system calls, or application-level artifacts**.

All workloads are intentionally **simple, deterministic, and semantics-agnostic**, ensuring that observed effects originate from *memory dynamics themselves* rather than high-level program meaning. This aligns with the thesis’ broader goal: **modeling memory as a discrete-time signal**, not as a structured OS object.

---

## Experimental Rationale

### Why workload-driven validation?

The proposed memory model represents system behavior through **page-wise deltas between consecutive memory snapshots**. To validate that this abstraction is meaningful, we must show that:

1. Different *classes of activity* produce **statistically separable memory-delta signals**.
2. These differences are observable using **signal-processing metrics** (PLV, MSC, cepstrum, entropy, sparsity, burstiness).
3. No semantic knowledge (process names, syscalls, file paths, kernel hooks) is required.

Synthetic workloads provide **ground truth behavioral regimes**:

* Idle → baseline noise floor
* Memory-intensive → spatially dense, RAM-dominated dynamics
* I/O-intensive → cache-, metadata-, and writeback-driven dynamics

---

## High-Level Experiment Structure

Each experiment run follows a **within-subject temporal structure**:

1. Initial idle period (baseline stabilization)
2. Memory-intensive workloads (A1–A3)
3. Idle recovery
4. I/O-intensive workloads (B1–B3)
5. Final idle recovery

All workloads:

* Run for a fixed duration
* Are terminated externally by a controller script (`timeout`)
* Produce continuous, steady-state memory activity

This design enables:

* Change-point detection
* Segment-level classification
* Distributional comparison across workload types

---

## Memory-Intensive Workloads (A-series)

### A1 — Sequential Memory Sweep (`mem_stream.py`)

**Description**
Allocates a contiguous memory buffer and repeatedly writes to it in a deterministic, page-stride pattern.

**Behavioral Role**
This workload represents **maximally structured RAM activity**:

* High spatial density (many pages touched)
* Strong temporal regularity
* Minimal allocator or kernel involvement

**Expected Signal Signature**

* High delta magnitude across contiguous page regions
* Strong periodic components (cepstrum peaks)
* High magnitude-squared coherence (MSC)
* Strong phase-locking value (PLV)

**Why it matters**
A1 serves as the **canonical reference** for a clean, periodic memory signal. It establishes the upper bound for coherence and periodicity in the proposed model.

---

### A2 — Random Pointer Chasing (`mem_pointer_chase.py`)

**Description**
Performs a pseudo-random walk across memory pages using a deterministic linear congruential generator (LCG). Accesses occur at page granularity without allocating a full permutation array.

**Behavioral Role**
Represents **unstructured, dispersed memory access**:

* Randomized spatial access
* No global sweep or ordering
* Minimal temporal regularity

**Expected Signal Signature**

* Sparse, scattered page deltas
* Weak or absent cepstral peaks
* Reduced MSC and PLV relative to A1
* Increased entropy and burstiness

**Why it matters**
A2 demonstrates that the model distinguishes *structured* vs. *unstructured* memory activity even when total memory volume is similar.

---

### A3 — Allocation Churn with Page Touching (`mem_alloc_touch_pages.py`)

**Description**
Repeatedly allocates many medium-sized objects, touches one byte per page, and frees them. Optional pauses introduce burst cycles.

**Behavioral Role**
Captures **allocator-driven dynamics**:

* Page faults
* Heap growth and release
* TLB and allocator metadata churn

**Expected Signal Signature**

* Bursty delta magnitude aligned with allocation cycles
* Alternation between dense and sparse regimes
* Cepstral peaks corresponding to churn period
* High entropy variance

**Why it matters**
A3 bridges pure memory access and OS-managed memory behavior, showing that allocator-induced dynamics are visible *without semantic awareness*.

---

## I/O-Intensive Workloads (B-series)

### B1 — Sequential File Writes with fsync (`io_seq_fsync.py`)

**Description**
Writes fixed-size chunks to disk sequentially, forcing persistence via `fsync` at a controlled cadence.

**Behavioral Role**
Induces **writeback- and cache-driven memory activity**:

* Dirty page accumulation
* Periodic flushes
* Filesystem and block-layer interaction

**Expected Signal Signature**

* Moderate delta magnitude
* Strong periodic components aligned with fsync cadence
* Coherent activity across kernel-managed page regions

**Why it matters**
B1 shows that *I/O-induced memory dynamics* produce signatures distinct from pure RAM writes, validating sensitivity to system state.

---

### B2 — Random Read/Write on Preallocated File (`io_rand_rw.py`)

**Description**
Performs random reads and writes on a large preallocated file with a configurable write ratio.

**Behavioral Role**
Models **cache churn and mixed I/O behavior**:

* Non-sequential access
* Competing read/write patterns
* Reduced temporal regularity

**Expected Signal Signature**

* Scattered deltas across memory
* Lower coherence than B1
* Increased entropy and burstiness

**Why it matters**
B2 demonstrates that the framework differentiates *types* of I/O behavior, not just the presence of I/O.

---

### B3 — Metadata-Heavy File Creation/Deletion (`io_many_files.py`)

**Description**
Repeatedly creates and deletes large numbers of small files in a single directory.

**Behavioral Role**
Stresses **filesystem metadata structures**:

* Inodes, directories, journaling
* Minimal data payload
* High operation rate

**Expected Signal Signature**

* Many small, scattered deltas
* High entropy and sparsity
* Weak cepstral structure unless explicitly periodic

**Why it matters**
B3 isolates *metadata-dominated* behavior, showing that even non-data-centric activity is visible in memory-delta space.

---

## Runner Script (`run_all.sh`)

The runner orchestrates all workloads with:

* Fixed execution durations
* Controlled idle intervals
* Timestamped logs
* External termination via `timeout`

### Why external termination?

Workloads are intentionally written as **steady-state loops**. External termination:

* Guarantees identical runtime across tests
* Avoids workload-specific stop conditions
* Simplifies experimental control

The resulting `KeyboardInterrupt` or exit code `124` is **expected and intentional**.

---

## Experimental Guarantees

This test suite ensures:

* Deterministic execution (fixed seeds)
* Reproducible timing and sizes
* No dependence on OS semantics or tracing
* Clear linkage between *behavioral regime* and *signal morphology*

It provides a **defensible experimental backbone** for evaluating memory as a complex-valued discrete-time signal.

---

## Intended Use in the Thesis

These workloads support:

* Feature validation (PLV, MSC, cepstrum, entropy)
* Ablation studies (magnitude vs. orientation)
* Change-point detection experiments
* Behavioral separability analysis

They are not benchmarks, stress tests, or application emulations — they are **controlled signal generators** for volatile memory dynamics.
