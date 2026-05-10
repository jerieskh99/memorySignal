# Purpose and Experimental Mentality Behind the VM Workload Executables

This document explains the purpose of the workload executables used in the VM experiments.

It is written for an AI system or analysis agent that needs to understand the *experimental logic* behind the dataset before interpreting results, generating analysis code, or drawing conclusions.

---

## Core Idea

The workload executables are not the classifier.

They are not the analysis pipeline.

They are **controlled behavioral stimuli**.

Their purpose is to generate known kinds of system behavior inside the VM so that the memory-capture and feature-extraction pipeline can observe how volatile memory changes under different workload conditions.

The central research question is:

> If we run workloads with known behavioral differences, can the resulting memory signal recover those differences without using program semantics?

In other words, the analysis should not depend on:
- process names
- source code semantics
- API calls
- file names as classification input
- program intent

Instead, the analysis should depend only on:
- volatile memory behavior
- signal-derived metrics
- statistical structure
- repeatability and separability

---

## Mental Model

Think of each workload executable as a **black-box signal generator**.

The executable creates a known behavioral condition inside the VM.  
The memory analysis pipeline then observes the resulting memory behavior as a signal.

The labels such as `mem_stream`, `io_rand_rw`, or `io_many_files` are used only as **ground truth for evaluation**.

They should not be treated as semantic input to the classifier or clustering process.

The goal is not:

> “Can we identify the Python script?”

The goal is:

> “Does this kind of workload leave a stable and distinguishable stochastic footprint in volatile memory?”

---

## Experimental Structure

The workload set is organized into three broad behavioral classes:

1. **IDLE**
2. **MEM**
3. **IO**

Each broad class may contain subtypes.

### MEM Subtypes

- `mem_stream`
- `mem_pointer_chase`
- `mem_alloc_touch_pages`

### IO Subtypes

- `io_rand_rw`
- `io_seq_fsync`
- `io_many_files`

### IDLE

- baseline/control idle windows

The system may run these workloads in cycles, with idle periods between active tests.

This matters because idle windows are not necessarily perfect zero-signal states. They may contain:
- background OS activity
- scheduler noise
- cache effects
- residual behavior from previous tests
- VM-level background activity

Therefore, IDLE should be treated as a baseline/control regime, not as a mathematically pure absence of activity.

---

## Purpose of Each Workload

### `run_idle.sh`

Purpose: baseline/control behavior.

This script intentionally does not create strong memory or I/O stress. It allows the system to observe the background volatile-memory signal of the VM when no synthetic workload is being actively generated.

Expected role:

> Establish what background memory behavior looks like.

Important interpretation:

IDLE is a control condition, but it may still contain system noise and residual effects. It should not automatically be assumed to be perfectly stable.

---

### `mem_stream.py`

Purpose: sequential memory pressure.

This workload creates a large memory buffer and repeatedly touches memory in a regular, page-strided pattern.

Expected behavior:
- sequential memory access
- regular spatial traversal
- repeated page touching
- relatively structured access pattern

Expected signal interpretation:

> If the memory signal captures behavioral structure, `mem_stream` should appear as a relatively regular memory-intensive process.

---

### `mem_pointer_chase.py`

Purpose: pseudo-random memory traversal.

This workload stresses memory using less predictable page access patterns. It is still memory-intensive, but unlike streaming, it reduces locality and makes access order less regular.

Expected behavior:
- memory pressure
- poor locality
- pseudo-random traversal
- weaker sequential structure

Expected signal interpretation:

> `mem_pointer_chase` should not look identical to `mem_stream`, even though both belong to the MEM class.

This is important because it tests whether the metrics can distinguish **subtypes inside the same broad class**.

---

### `mem_alloc_touch_pages.py`

Purpose: allocation churn and page-touch behavior.

This workload repeatedly allocates memory, touches pages, and releases memory.

Expected behavior:
- allocation/deallocation cycles
- page touching
- allocator activity
- burst-like memory pressure
- repeated batch structure

Expected signal interpretation:

> This workload may produce a more repeatable batch-like signature because its behavior has clear allocate-touch-release phases.

This subtype is important because it is memory-intensive, but it stresses memory differently from both streaming and pointer chasing.

---

### `io_rand_rw.py`

Purpose: random read/write I/O behavior.

This workload performs random reads and writes against a file.

Expected behavior:
- random file offsets
- mixed reads/writes
- page-cache interaction
- block-level I/O pressure
- irregular access structure

Expected signal interpretation:

> This should produce an I/O-heavy signal with randomness and bursts, distinct from sequential or metadata-heavy I/O.

---

### `io_seq_fsync.py`

Purpose: sequential write behavior with forced synchronization.

This workload writes sequentially and periodically forces data to be synchronized.

Expected behavior:
- sequential writes
- repeated synchronization barriers
- flush/fsync behavior
- more rhythmic I/O pressure

Expected signal interpretation:

> This may produce a stable I/O subtype signature because the write-sync rhythm is more structured than random I/O.

---

### `io_many_files.py`

Purpose: metadata-heavy small-file I/O.

This workload creates, writes, and deletes many small files.

Expected behavior:
- many file creations
- many file deletions
- metadata updates
- directory/cache activity
- filesystem object churn

Expected signal interpretation:

> This is different from bulk read/write I/O because it stresses filesystem metadata and object management rather than only data movement.

This helps test whether volatile memory behavior captures differences between different kinds of I/O activity.

---

## What the Analysis Is Trying to Prove

The analysis should test whether memory-derived metrics can recover structure at two levels.

---

### Level 1: Broad Behavioral Class

Can memory behavior distinguish:

- IDLE
- MEM
- IO

This asks whether volatile memory signal patterns can recover coarse workload families.

A strong result at this level means:

> Memory behavior contains enough information to distinguish broad system states.

---

### Level 2: Behavioral Subtype

Can memory behavior distinguish subtypes inside a broad class?

For example:

- `mem_stream` vs `mem_pointer_chase` vs `mem_alloc_touch_pages`
- `io_rand_rw` vs `io_seq_fsync` vs `io_many_files`

This asks whether volatile memory signal patterns contain finer-grained behavioral fingerprints.

A strong result at this level means:

> Memory behavior is not only class-level structured; it may encode repeatable process-level stochastic signatures.

---

## Repeatability and Separability

The key logic is:

A useful behavioral fingerprint must be both:

1. **Repeatable**
2. **Separable**

---

### Repeatability

Repeated runs of the same subtype should produce similar metric values or similar positions in feature space.

Example:

If several runs of `io_seq_fsync` are captured, they should appear close to each other.

This means the workload leaves a stable memory signature.

---

### Separability

Different subtypes should appear measurably different.

Example:

`io_seq_fsync` should be distinguishable from `io_rand_rw` and `io_many_files`.

This means the memory signal carries information about the type of behavior being executed.

---

### Why Both Matter

Separability alone is not enough.

A method can separate samples in a plot but still be unreliable if repeated runs of the same workload are unstable.

Repeatability alone is also not enough.

A workload can be stable but indistinguishable from another workload.

The strongest evidence is:

> Low variation within the same subtype and high separation between different subtypes.

This is the central fingerprinting idea.

---

## Why CV Matters

CV means **coefficient of variation**.

It measures relative variability:

> CV = standard deviation / mean

In this experiment, CV is useful because it measures whether repeated runs of the same subtype are stable.

Low CV means:

> The same workload produces a similar memory signature across repeated runs.

High CV means:

> The same workload changes too much across runs, so its fingerprint is less reliable.

Therefore, CV is not the classifier.  
CV is a reliability check.

A subtype with low CV is a better candidate for fingerprinting than a subtype with high CV.

---

## Why LDA Must Be Treated Carefully

LDA is useful, but it is supervised.

It uses the known labels to find directions that best separate the labeled groups.

Therefore, an LDA plot can show strong separation even when the true generalization ability is uncertain.

LDA should be interpreted as:

> “There exists a label-aware linear basis where these samples separate.”

It should not be interpreted as:

> “The system can classify future runs.”

To trust LDA more, it should be validated using:
- leave-one-out cross-validation
- held-out run testing
- centroid margin analysis
- bootstrap stability
- future data collection

The correct use of LDA here is as a diagnostic tool, not as final proof.

---

## Why PCA and MDS Matter

PCA and MDS are useful because they are not label-optimized in the same way as LDA.

### PCA

PCA shows the main variance structure in the data.

If subtypes separate in PCA, that suggests the separation is present in the natural feature geometry.

### MDS

MDS tries to preserve pairwise distances.

If repeated runs of the same subtype are close in MDS, that supports the claim that the descriptor space encodes behavioral similarity.

### Combined Interpretation

If PCA, MDS, and LDA all show compatible structure, confidence increases.

The strongest interpretation is not based on one plot. It comes from agreement across:
- repeatability statistics
- feature rankings
- PCA geometry
- MDS distances
- LDA separability
- cross-validation

---

## How to Think About the Executables

The executables are deliberately simple and controlled.

That is a strength.

They are not meant to represent all possible real-world behavior.  
They are meant to create clear experimental probes:

- regular memory pressure
- pseudo-random memory pressure
- allocation churn
- random I/O
- sequential synchronized I/O
- metadata-heavy I/O
- idle baseline

The purpose is to test whether the memory-signal pipeline can detect differences between these known stimuli.

If it cannot distinguish these controlled workloads, it is unlikely to work on more complex behavior.

If it can distinguish them, that supports the idea that volatile memory contains semantic-free behavioral structure.

---

## Correct Thesis-Level Interpretation

A safe interpretation is:

> The workload executables generate controlled behavioral regimes inside the VM. By observing only volatile-memory signal patterns, the analysis tests whether these regimes produce repeatable and separable stochastic signatures.

Another safe interpretation:

> The labels are not used as semantic input; they are used only to validate whether the signal-derived metrics recover known experimental structure.

A stronger but still careful interpretation:

> Some workload types appear to leave stable memory fingerprints, especially when repeated runs cluster together and remain separated from other workload subtypes.

---

## Claims That Are Safe

The following claims are reasonable if supported by the data:

- The executables serve as controlled workload stimuli.
- Memory behavior can be treated as a signal.
- Some workload regimes produce repeatable memory signatures.
- Some subtype-level behaviors are separable in feature space.
- Classification claims require validation beyond visual separation.
- LDA is useful but must be cross-validated.
- Low within-subtype variation plus high between-subtype separation supports behavioral fingerprinting.

---

## Claims to Avoid Without More Validation

Avoid claiming:

- “The system can classify all workloads generally.”
- “LDA proves classification.”
- “The model will work on unseen future runs.”
- “The workload filename is the identity recovered by the system.”
- “IDLE is perfectly clean and stable.”
- “The system understands program semantics.”

The system does not understand semantics.

It measures signal structure.

---

## Final Summary for an AI Agent

These workload executables are the controlled causes in the experiment.

They generate known memory and I/O behavior inside the VM.

The analysis pipeline observes only memory-derived signal metrics.

The scientific goal is to determine whether those metrics reveal stable and distinguishable stochastic signatures corresponding to workload behavior.

The correct mentality is:

> Treat the executable labels as ground truth for evaluation, not as semantic input.

The central test is:

> Do repeated runs of the same workload subtype stay close, and do different workload subtypes move apart?

If yes, the experiment supports semantic-free behavioral fingerprinting of volatile memory activity.
