# Confusion Matrix as a Diagnostic Tool for Workload Test Cleanliness

This document organizes the methodology for interpreting subtype confusion results in semantic-free volatile-memory behavior classification.

It is written for an AI system or analysis agent that needs to understand the reasoning philosophy behind the confusion matrix before judging whether the results are scientifically meaningful.

The central idea is:

> A confusion matrix is not only an accuracy table. It is also a diagnostic map of behavioral overlap, hidden side effects, test impurity, and metric limitations.

---

## 1. Core Philosophy

The goal of this experiment is not simply to ask:

> Did the classifier get the label correct?

The deeper question is:

> Why does one workload sometimes look like another workload in memory-signal space?

A misclassification should not immediately be treated as failure. It may reveal something important about the workload, the system, or the measurement process.

For example, if `mem_stream` is predicted as `io_rand_rw`, the question should be:

> What hidden mechanisms made a sequential memory workload produce a signal similar to random I/O?

This reasoning helps identify:

- which tests are clean probes
- which tests are mixed or impure probes
- which metrics are reliable
- which mechanisms overlap in volatile-memory signal space
- how future tests should be redesigned

---

## 2. Observed Signal Is Not the Same as Intended Workload

Every executable has an intended behavior.

Example:

```text
mem_stream → sequential memory pressure
```

But the measured memory signal may contain more than the intended behavior.

The observed signal can include:

```text
observed memory signal
=
intended workload
+
runtime effects
+
kernel effects
+
VM noise
+
residual state
+
measurement artifacts
```

This means the memory signal may contain contributions from:

- intended workload behavior
- Python runtime behavior
- memory allocation
- garbage collection
- page faults
- page cache activity
- dirty-page writeback
- scheduler behavior
- kernel buffering
- VM/host interference
- residual effects from previous tests

Therefore, a workload label should be treated as the **intended stimulus**, not as a guarantee that the observed memory signal is pure.

---

## 3. What the Confusion Matrix Tells Us

The confusion matrix shows how held-out recordings were classified using the learned descriptor space.

Rows are true labels.  
Columns are predicted labels.  
Diagonal entries are correct predictions.  
Off-diagonal entries are misclassifications.

However, the off-diagonal entries are not just errors.

They are clues.

Each confusion should be interpreted as:

> The true workload produced a signal that was closer to the predicted subtype than to its own subtype.

That can happen because of:

- true behavioral overlap
- weak or unstable signal from the true workload
- strong attraction toward a cleaner subtype centroid
- hidden subsystem effects
- execution-order effects
- insufficient metric specificity
- baseline-relative suppression
- runtime or VM noise

---

## 4. General Interpretation Workflow

For every misclassification, use this reasoning template:

```text
True test:
Predicted as:
What behavior do they share?
What hidden subsystem could connect them?
Was the true test weak or unstable?
Was the predicted class/subtype strong or attractive?
Could execution order explain it?
Could baseline-relative metrics explain it?
What should be changed in the test design?
```

This shifts the analysis from simple scoring to mechanism discovery.

---

## 5. Four-Cause Diagnostic Model

Every misclassification must be evaluated against four distinct and separable explanations before any interpretation is made. These causes are not equivalent; each has a different experimental remedy.

| Cause | What it means | Experimental remedy |
|---|---|---|
| True behavioral overlap | The two workloads share a mechanism that produces similar memory-signal structure | Redesign the workload to isolate the intended mechanism |
| Capture artifact | The measurement process itself (pmemsave pause duration, sampling cadence, page granularity) introduces apparent similarity | Fix or characterize the capture pipeline |
| Execution-order contamination | A prior workload leaves residual state (page cache, dirty pages, kernel buffers) that bleeds into the current measurement window | Randomize run order, extend idle recovery, or record prior workload context |
| Metric inadequacy | The descriptor fails to distinguish two genuinely different signals | Add, replace, or refine features |

When interpreting any off-diagonal confusion, rate each cause:

```text
Cause 1 — true behavioral overlap:          likely / possible / unlikely / unknown
Cause 2 — capture artifact:                 likely / possible / unlikely / unknown
Cause 3 — execution-order contamination:    likely / possible / unlikely / unknown
Cause 4 — metric inadequacy:                likely / possible / unlikely / unknown
```

Do not collapse these four into a single narrative. A collapsed narrative obscures which fix to apply.

> **Critical constraint**: A claim that a confusion reveals "behavioral overlap" is only defensible after capture artifacts, execution-order contamination, and metric inadequacy have been rated as unlikely or unknown. Until isolation experiments exist, all four causes must remain open hypotheses.

---

## 6. Provisional Language Rule

At fewer than 8 to 10 independent runs per subtype, no cleanliness designation is final.

**Rule**: Do not assign labels such as "clean probe," "high cleanliness," or "distinctive fingerprint" as settled facts at small sample sizes. Assign only provisional labels.

**Why**: A 95% confidence interval on 1 error in 4 trials spans roughly 1% to 81%. A result of 4/4 correct is consistent with a true error rate anywhere from near 0% to roughly 60%. That is not a stable empirical property; it is a preliminary positive observation.

**Required language in all cleanliness assessments at n < 8**:

Use:
- "provisionally high cleanliness"
- "consistent with a clean probe, pending replication"
- "4/4 correct in this preliminary sample"
- "lowest observed accuracy in this preliminary sample"

Do not use:
- "clean probe" without qualification
- "distinctive fingerprint" as a settled claim
- "most important test to investigate" based on a single small sample

This rule applies to all sections in this document and all future cleanliness tables.

---

## 7. Confusion Direction Rule

Every confusion must state its direction explicitly.

**Format**:

```text
true subtype → predicted subtype
```

**Examples**:

```text
mem_stream → run_idle     (a mem_stream run was predicted as run_idle)
mem_stream → io_rand_rw   (a mem_stream run was predicted as io_rand_rw)
```

**Why direction matters**: The direction of a confusion has a different mechanistic interpretation from its reverse.

- `A → B` means A's signal was pulled toward B's centroid. This may implicate weakness in A's signal, unusual strength in B's centroid, or execution-order residue from a prior B run.
- `B → A` means B's signal was pulled toward A's centroid. This implicates a different mechanism and a different remedy.

In the cleanliness table and in the Expected Confusions field, always use `true subtype → predicted subtype` notation. Never use symmetric labels like "A confuses with B."

---

## 8. Clean vs Impure Workload Tests

A clean workload test is one where:

- the intended mechanism dominates the observed signal
- repeated runs cluster together
- within-subtype variation is low
- between-subtype separation is high
- misclassification is rare
- the signal is interpretable

An impure workload test is one where:

- hidden mechanisms dominate or mix with the intended mechanism
- repeated runs scatter
- it gets confused with other families or subtypes
- its descriptors are unstable
- its behavior depends strongly on execution order, cache state, or VM background effects

The goal is not to force every test to be perfect.

The goal is to understand which tests are clean probes and which tests are composite probes.

---

## 9. Current Confusion Matrix: High-Level Reading

From the current subtype confusion matrix (n=4 per subtype, preliminary sample), the highest observed accuracy subtypes are:

```text
mem_alloc_touch_pages: 4/4 correct  (provisional; see Section 6)
io_many_files:         4/4 correct  (provisional; see Section 6)
io_seq_fsync:          4/4 correct  (provisional; see Section 6)
```

These are provisionally consistent with being clean subtype probes. They are not confirmed as such at this sample size.

Moderately clean subtypes in this sample:

```text
mem_pointer_chase: 3/4 correct  — one run: mem_pointer_chase → mem_stream
io_rand_rw:        3/4 correct  — one run: io_rand_rw → mem_alloc_touch_pages
```

Lowest observed accuracy subtype in this sample:

```text
mem_stream: 2/4 correct  — one run: mem_stream → run_idle
                         — one run: mem_stream → io_rand_rw
```

This does not mean the entire analysis is bad.

It means `mem_stream` is the highest-priority subtype for follow-up investigation, given its preliminary result. The result must be replicated before drawing conclusions.

---

## 10. Cleanest Current Tests

### 7.1 `mem_alloc_touch_pages`

Current result:

```text
4/4 correct
```

Likely reason:

This workload has a strong repeated structure:

```text
allocate → touch pages → release → repeat
```

That repeated structure may create a stable memory-management signature.

Interpretation:

> `mem_alloc_touch_pages` appears to be a clean MEM subtype because the allocation/touch/release phases create a repeatable and separable memory fingerprint.

Possible reason it works well:

- strong allocation rhythm
- repeated page-touching
- clear burst structure
- distinctive memory-management behavior
- less ambiguity with filesystem-driven I/O than expected

---

### 7.2 `io_many_files`

Current result:

```text
4/4 correct
```

Likely reason:

This workload stresses filesystem metadata and object churn:

```text
create many files → write small payloads → delete files
```

This creates a distinctive kernel/filesystem pattern.

Interpretation:

> `io_many_files` appears to be a clean IO subtype because metadata-heavy small-file activity produces a repeatable memory signature.

Possible reason it works well:

- many file objects
- directory/inode/dentry activity
- repeated metadata churn
- small bursty operations
- distinct from bulk sequential or random block I/O

---

### 7.3 `io_seq_fsync`

Current result:

```text
4/4 correct
```

Likely reason:

This workload has a repeated write/sync rhythm:

```text
write → fsync → write → fsync
```

That synchronization pattern likely creates stable temporal structure.

Interpretation:

> `io_seq_fsync` appears to be a clean IO subtype because forced synchronization creates a repeatable rhythm in the memory signal.

Possible reason it works well:

- repeated synchronization barriers
- dirty-page flush behavior
- rhythmic I/O phases
- strong temporal structure
- separable from random I/O and metadata-heavy I/O

---

## 11. Moderately Clean Tests

### 8.1 `mem_pointer_chase`

Current result:

```text
3/4 correct
1 run predicted as mem_stream
```

This confusion is understandable.

Both `mem_pointer_chase` and `mem_stream` are memory traversal workloads.

The intended distinction is:

```text
mem_stream         → regular/sequential traversal
mem_pointer_chase  → irregular/pseudo-random traversal
```

But a pointer-chase implementation may still produce some structured behavior depending on:

- pseudo-random generator behavior
- stride pattern
- page alignment
- TLB/cache behavior
- Python runtime overhead
- working-set size
- whether the access order is truly data-dependent

Interpretation:

> `mem_pointer_chase` is mostly separable, but it overlaps with `mem_stream` because both are fundamentally memory page traversal workloads.

Potential design improvements:

- make pointer chasing more truly data-dependent
- reduce predictable stride behavior
- increase working set beyond cache/TLB
- avoid patterns that hardware prefetchers can learn
- compare fixed-seed vs variable-seed runs
- record whether confusion is stable or run-specific

---

### 8.2 `io_rand_rw`

Current result:

```text
3/4 correct
1 run predicted as mem_alloc_touch_pages
```

This is also understandable.

Random I/O can activate memory-management mechanisms through:

- page-cache allocation
- dirty page creation
- kernel buffers
- metadata updates
- writeback
- cache replacement
- file-backed memory pressure

So although the intended behavior is I/O, the memory system may experience part of it as allocation/touch/dirty-page churn.

Interpretation:

> `io_rand_rw` is mostly separable, but it can resemble `mem_alloc_touch_pages` because random file I/O can trigger page-cache and memory-management behavior.

Potential design improvements:

- separate read-only random I/O from write-heavy random I/O
- test cached vs cold-cache random I/O
- control file preallocation
- control random seed
- isolate from prior I/O workload effects
- record whether the misclassified run followed a memory-heavy or I/O-heavy previous test
- consider a direct-I/O-like variant if available

---

## 12. Deep Dive: `mem_stream` as the Weakest Current Test in This Preliminary Sample

Current result:

```text
mem_stream: 2/4 correct
misclassified as:
- run_idle
- io_rand_rw
```

In this preliminary sample, `mem_stream` has the lowest observed classification accuracy. At n=4, this is not a stable finding; it is a priority signal for targeted follow-up investigation.

The result does not necessarily mean `mem_stream` failed as a workload. It means the observed memory-signal signature of some `mem_stream` runs was not consistently unique in this sample.

This is valuable because it reveals that sequential memory streaming may be harder to fingerprint than expected.

---

### 9.1 Why Could `mem_stream` Be Predicted as IDLE?

Possible explanations:

#### A. Sequential streaming may be too smooth

A regular memory stream may not produce many sharp memory events if the system handles it efficiently.

For example:

- hardware prefetching works well
- pages are already mapped
- memory access becomes smooth
- few abrupt changes occur
- the signal becomes low-event and quiet

Memory-signal interpretation:

```text
regular + optimized + smooth
→ lower event structure
→ more idle-like
```

Important implication:

> A workload can be active but still not produce strong detectable memory events.

---

#### B. Hardware prefetching may reduce visible irregularity

Sequential access patterns are exactly what CPUs and memory subsystems optimize.

If the hardware prefetcher detects the pattern, it may reduce stalls and smooth the memory behavior.

This can make `mem_stream` less distinctive than more irregular memory tests.

---

#### C. The run may have had weak workload intensity

The misclassified run may not have stressed memory strongly enough.

Possible reasons:

- buffer too small
- duration too short
- access pattern too efficient
- measurement window missed the active phase
- workload stabilized after warm-up
- sampling resolution was too coarse
- CPU/runtime bottleneck dominated instead of memory behavior

---

#### D. Baseline-relative metrics may suppress the signal

If metrics are computed relative to the first idle baseline, then a smooth stream may not deviate strongly.

In simplified terms:

```text
mem_stream - idle_baseline ≈ small difference
```

If the difference is small, the classifier may see the run as IDLE-like.

---

#### E. IDLE itself may not be perfectly stable

If some IDLE runs contain background/residual activity, the boundary between quiet active workloads and noisy idle windows can blur.

This means the confusion may not only be caused by `mem_stream`; it may also reflect instability in the IDLE baseline.

---

### 9.2 Why Could `mem_stream` Be Predicted as `io_rand_rw`?

Possible explanations:

#### A. Page faults and demand paging

If `mem_stream` touches pages that were not physically committed before measurement, the OS may allocate/map pages during the run.

This can create:

- page faults
- page table changes
- memory-management bursts
- kernel activity

Random I/O also activates page cache and kernel memory-management behavior.

So the overlap may happen through the OS memory-management layer.

---

#### B. Page-cache or writeback residue from previous tests

If the experiment runs in cycles, the previous workload can affect the current measurement window.

Example:

```text
previous IO test → idle → mem_stream
```

The `mem_stream` window may still contain:

- delayed writeback
- cache recovery
- dirty-page cleanup
- filesystem background activity

That can make the memory signal appear more I/O-like.

---

#### C. Python runtime effects

Even if the conceptual workload is simple, Python may introduce:

- interpreter overhead
- loop dispatch effects
- object checks
- allocation artifacts
- garbage collection
- runtime periodicity

These can add irregular components that make a smooth stream less pure.

---

#### D. Large memory streams can trigger system-level data movement

If the memory footprint is large enough, sequential streaming may interact with:

- page allocation
- reclaim behavior
- VM memory pressure
- swap risk
- host memory pressure
- NUMA or VM memory placement effects

This can make the behavior look more like system-level data movement than pure user-space memory traversal.

---

### 9.3 What `mem_stream` Teaches Us

`mem_stream` is not necessarily a bad test.

It may be a **sensitive diagnostic test** because it reveals whether the memory-signal pipeline detects smooth sequential access or only stronger disruptive behavior.

Possible interpretation:

> `mem_stream` may be too optimized, too smooth, or too sensitive to environmental effects to produce a stable fingerprint under the current setup.

This is useful.

It tells us that future MEM tests should explicitly separate:

- read streaming
- write streaming
- pre-faulted streaming
- cold-page streaming
- cache-hot streaming
- cache-cold streaming
- stride-based streaming

---

### 9.4 How to Improve `mem_stream`

Potential design improvements:

- preallocate memory before measurement
- pre-fault all pages before measurement
- separate warm-up phase from measurement phase
- use fixed measurement window after warm-up
- increase working set beyond cache
- compare read stream vs write stream
- compare sequential stride vs larger stride
- use native implementation if Python overhead dominates
- randomize test order to reduce cycle bias
- add longer idle recovery before `mem_stream`
- record previous workload type before each run
- test whether misclassification correlates with previous I/O workloads

---

## 13. Expected Confusions as Part of Test Design

Future mini-specs must include an `Expected confusions` field.

Each entry must specify three things:

1. Direction using `true subtype → predicted subtype` notation (Section 7)
2. Cause type from the four-cause model (Section 5): one of `behavioral proximity`, `capture artifact`, `execution-order contamination`, or `metric inadequacy`
3. A brief mechanistic reason

**Format template**:

```text
Expected confusions:
- `[true subtype] → [predicted subtype]`
  - Type: [cause type]
  - Reason: [mechanistic explanation]
```

**Example for `mem_stream`**:

```text
Expected confusions:
- `mem_stream → run_idle`
  - Type: metric inadequacy / weak event structure
  - Reason: smooth sequential access may produce low delta-event activity indistinguishable from background noise
- `mem_stream → io_rand_rw`
  - Type: execution-order contamination or capture artifact
  - Reason: page-cache/writeback residue from a prior I/O run, or demand-paging during first-touch, may produce I/O-like memory effects
```

**What this replaces**: Do not use:

```text
Expected confusions:
- mem_stream may confuse with idle
```

That format omits direction, omits cause type, and cannot guide experimental correction.

**Why typed expected confusions matter**: Pre-registering expected confusions before analysis reduces post-hoc rationalization. Typing them by cause constrains the remediation path. Behavioral proximity calls for workload redesign. Capture artifact calls for pipeline fixes. Execution-order contamination calls for run-order controls or longer recovery windows. Metric inadequacy calls for feature revision.

This is not an excuse for poor classification. It is a way to build a more honest and falsifiable experimental taxonomy.

---

## 14. How This Should Influence Future Test Design

For every future workload, define:

### Intended mechanism

What is this test supposed to stress?

Examples:

- memory traversal
- allocation churn
- filesystem metadata
- sequential synchronization
- CPU computation
- cache locality
- thread contention
- network buffering
- mixed behavior

### Unwanted mechanisms

What should this test avoid?

Example for `mem_stream`:

- filesystem I/O
- allocation during measurement
- garbage collection during measurement
- page faults during measurement
- previous-test residue
- host/VM interference

### Control strategy

How do we make it cleaner?

Examples:

- warm-up phase
- preallocation
- prefaulting
- fixed duration
- randomized order
- repeated runs
- longer recovery idle
- separating cached vs cold-cache variants
- explicit logging of previous test context

---

## 15. Test Cleanliness Table

> **Provisional designations only.** All cleanliness labels below are based on n=4 runs per subtype. At this sample size no label is final (Section 6). All four-cause ratings are working hypotheses, not conclusions from isolation experiments.

> **Direction notation**: confusions are written as `true subtype → predicted subtype` (Section 7).

| True subtype | Confusions (true → predicted) | Four-cause assessment | Provisional cleanliness | Next change |
|---|---|---|---|---|
| `mem_stream` | `mem_stream → run_idle`, `mem_stream → io_rand_rw` | behavioral overlap: possible; capture artifact: possible; execution-order contamination: possible; metric inadequacy: likely | Provisionally weak | prefault, warm-up, stronger write stream, isolate from prior I/O |
| `mem_pointer_chase` | `mem_pointer_chase → mem_stream` | behavioral overlap: likely; capture artifact: unlikely; execution-order contamination: possible; metric inadequacy: possible | Provisionally medium | make access more truly random/data-dependent |
| `mem_alloc_touch_pages` | none (4/4 in preliminary sample) | — | Provisionally high (pending replication at n >= 8) | keep as clean MEM subtype |
| `io_rand_rw` | `io_rand_rw → mem_alloc_touch_pages` | behavioral overlap: possible; capture artifact: possible; execution-order contamination: possible; metric inadequacy: possible | Provisionally medium | split cached vs uncached/read-only/write-heavy variants |
| `io_many_files` | none (4/4 in preliminary sample) | — | Provisionally high (pending replication at n >= 8) | keep as clean IO subtype |
| `io_seq_fsync` | none (4/4 in preliminary sample) | — | Provisionally high (pending replication at n >= 8) | keep as clean IO subtype |
| `run_idle` | `run_idle → mem_stream`, `run_idle → mem_pointer_chase` | execution-order contamination: possible; behavioral overlap: possible; capture artifact: possible; metric inadequacy: possible | Provisionally medium | separate first-idle, long-idle, post-test idle; record prior workload context |

**Note on direction ambiguity resolved**: The original table listed `run_idle` as confused with `mem_stream` and `mem_pointer_chase` without specifying direction. This table clarifies that the `run_idle` row represents runs where a true `run_idle` recording was predicted as an active workload (`run_idle → mem_stream`), which is distinct from the confusion in the `mem_stream` row where a true `mem_stream` run was predicted as `run_idle` (`mem_stream → run_idle`). Both directions exist and have different mechanistic implications.

---

## 16. Final Methodological Statement

The confusion matrix should be used as a scientific diagnostic tool.

A misclassification means:

> The observed memory signal of one workload was closer to another workload's signature.

That can reveal real behavioral overlap, hidden subsystem activity, test impurity, or metric weakness.

The goal is not only to maximize accuracy.

The goal is to understand which workload generators produce clean, repeatable, and separable volatile-memory signatures — and why.

This methodology turns classification errors into experimental feedback for designing better workload probes.
