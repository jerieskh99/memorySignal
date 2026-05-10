# Window-Level Trace Analysis Methodology

This document describes a proposed improvement to the current workload-behavior validation methodology.

The core idea is to stop treating each full recording as only one summarized sample, and instead analyze each recording as a sequence of local trace windows.

This may help explain whether misclassifications are caused by the entire workload being similar to another workload, or only by specific temporal phases such as warm-up, cleanup, page faults, residual writeback, or measurement artifacts.

---

## 1. Current Method

The current pipeline roughly behaves as follows:

```text
NPZ recording
→ many delta snapshots
→ windowed traces of size 128 with hop 64
→ metrics computed per trace/window
→ metrics aggregated over the full recording
→ one feature vector per recording
→ classifier / clustering / LDA validation
```

In this setup, each entire test run becomes a single summarized point in feature space.

This is simple and useful, but it can hide temporal structure.

---

## 2. Limitation of Full-Recording Aggregation

A full recording may contain multiple phases:

```text
startup / warm-up
→ active workload
→ stabilized behavior
→ cleanup / residual effects
```

If the pipeline averages all of these phases into one feature vector, the resulting point may become a mixture of different behaviors.

This can cause problems such as:

- active workloads looking IDLE-like
- memory workloads looking I/O-like
- transient artifacts dominating the whole recording summary
- stable active phases being diluted by quiet phases
- phase-specific behavior being lost

Therefore, a single full-run feature vector may not accurately represent the internal temporal structure of the workload.

---

## 3. Proposed New Method

Instead of summarizing the whole recording immediately, treat each trace window as a local sample.

```text
recording
→ trace_1
→ trace_2
→ trace_3
→ ...
```

Each trace/window receives its own feature vector.

Then classification can be performed at two levels:

1. **Window-level classification**
2. **Recording-level aggregation**

---

## 4. Window-Level Classification

For every local trace window, predict the most likely workload label.

Example:

```text
test14_mem_stream:
window_001 → mem_stream
window_002 → mem_stream
window_003 → run_idle
window_004 → io_rand_rw
...
```

This reveals whether the recording is temporally stable or internally mixed.

The goal is not only to improve accuracy.

The deeper goal is to understand *where* and *when* confusion happens inside a recording.

---

## 5. Recording-Level Aggregation

After predicting every window in a held-out recording, aggregate the window predictions into one recording-level prediction.

Possible aggregation methods:

- majority vote
- mean predicted probability
- median confidence
- temporal smoothing
- active-window-only voting

Example:

```text
test14_mem_stream:
65% windows predicted as mem_stream
20% windows predicted as run_idle
15% windows predicted as io_rand_rw

final recording prediction = mem_stream
```

This gives both:

- a final recording-level label
- a temporal purity profile

---

## 6. Critical Validation Rule: Avoid Leakage

This is the most important rule.

Do **not** randomly split windows.

Wrong approach:

```text
train: some windows from test14_mem_stream
test: other windows from test14_mem_stream
```

This leaks recording-specific information and produces fake high accuracy.

Correct approach:

```text
train: all windows from other recordings
test: all windows from one held-out recording
```

The validation split must be done by recording, not by window.

This is called:

> Leave-one-recording-out validation

Each fold holds out one complete recording and tests all windows from that recording.

---

## 7. Why This May Improve Accuracy

Window-level analysis may improve recording-level accuracy because it avoids letting short artifact phases dominate the full recording summary.

For example, a recording that was previously classified incorrectly may actually contain mostly correct windows:

```text
full-run prediction:
test14_mem_stream → io_rand_rw

window-level profile:
70% mem_stream-like
20% io_rand_rw-like
10% idle-like
```

In that case, majority vote may recover the correct label.

However, even if accuracy does not improve, the method is still valuable because it explains the temporal structure of the confusion.

---

## 8. Why This Helps Diagnose Misclassification

### Case: `mem_stream → run_idle`

Question:

> Is the whole `mem_stream` recording idle-like, or only some windows?

Possible interpretations:

- If only early/late windows are idle-like:
  - the issue may be warm-up, cleanup, or phase contamination.

- If the entire recording is idle-like:
  - `mem_stream` may be too smooth, optimized, or weak to produce strong memory-event structure.

---

### Case: `mem_stream → io_rand_rw`

Question:

> Are I/O-like windows clustered in a specific part of the run?

Possible interpretations:

- I/O-like windows at the beginning:
  - demand paging, first-touch page faults, initialization effects.

- I/O-like windows at the end:
  - cleanup, delayed writeback, residual system activity.

- I/O-like windows scattered throughout:
  - metric instability, background noise, or true behavioral overlap.

- Entire run I/O-like:
  - stronger evidence that the workload activates memory-management behavior similar to random I/O.

---

### Case: `io_rand_rw → mem_alloc_touch_pages`

Question:

> Does random I/O look like allocation churn throughout the full run, or only during bursts?

Possible interpretations:

- Burst-localized confusion:
  - page-cache allocation, dirty-page creation, or kernel buffer churn.

- Whole-run confusion:
  - the selected metrics may not distinguish random I/O from memory allocation/page touching well enough.

---

## 9. New Diagnostic Metrics

Window-level analysis enables several new metrics.

### 9.1 Window Purity

Fraction of windows predicted as the true recording label.

```text
window purity = correctly predicted windows / total windows
```

High purity means:

> The workload is temporally stable and consistently recognizable.

Low purity means:

> The recording contains mixed phases, artifacts, or unstable behavior.

### 9.2 Dominant Predicted Subtype

The most common predicted subtype across windows.

This tells us what the recording mostly looks like.

### 9.3 Confusion Entropy

Measures whether window predictions are concentrated or scattered.

Low confusion entropy:

```text
most windows predicted as one label
```

High confusion entropy:

```text
windows scattered across many predicted labels
```

High entropy suggests unstable or mixed behavior.

### 9.4 Early / Middle / Late Confusion Breakdown

Divide each recording into temporal thirds:

```text
early
middle
late
```

Then report prediction distributions for each segment.

This helps identify whether confusion is phase-localized.

### 9.5 Longest Stable Segment

The longest consecutive sequence of windows predicted as the same label.

A clean workload should have long stable segments.

### 9.6 Transition Count

Number of times the predicted label changes across the recording.

Clean workload:

```text
low transition count
high purity
long stable segments
```

Mixed or artifact-heavy workload:

```text
high transition count
low purity
many short segments
```

---

## 10. Scientific Value

This method is valuable because it changes the question from:

> Was the whole recording classified correctly?

to:

> What parts of the recording look like which behavior?

This can distinguish between:

- global similarity
- local artifacts
- startup effects
- cleanup effects
- execution-order contamination
- unstable workload signatures
- insufficient metrics

It also helps determine whether full-run averaging is hiding useful structure.

---

## 11. Relationship to Existing Confusion-Matrix Methodology

The confusion matrix tells us which full recordings were misclassified.

Window-level analysis helps explain why.

Together:

```text
full-run confusion matrix
→ identifies suspicious recordings

window-level timeline
→ explains when and how confusion occurs
```

This makes the methodology more diagnostic and less dependent on a single aggregate prediction.

---

## 12. Proposed Claude Implementation Prompt

Use the following prompt to ask an analysis agent to implement this method.

```text
Use the data-scientist agent.

I want to test a new validation idea using the existing NPZ recordings.

Current method:
Each recording is summarized into one feature vector by aggregating metrics over all trace windows.

New method:
Treat each recording as a sequence of local trace windows.
Compute metrics per window/trace.
Train classifiers on windows, but split by recording to avoid leakage.

Implement:
1. Window-level feature extraction using the same trace size and hop already used:
   - trace size = 128
   - hop = 64
2. Leave-one-recording-out validation:
   - hold out one full recording
   - train on windows from all other recordings
   - predict every window in the held-out recording
3. Aggregate window predictions to recording prediction by:
   - majority vote
   - mean class probability if available
4. Output:
   - recording-level accuracy
   - window-level purity per recording
   - confusion entropy per recording
   - early/mid/late confusion breakdown
   - longest stable predicted segment
   - transition count
   - per-window prediction timeline
5. Produce JSON results.
6. Add a dashboard tab showing:
   - per-recording temporal prediction timeline
   - window purity bar chart
   - confusion entropy
   - before/after comparison against full-run LDA LOOCV

Important:
Do not randomly split windows. Splits must be by recording to avoid leakage.
```

---

## 13. Final Methodological Statement

Dividing recordings into local trace windows is scientifically meaningful because it allows the analysis to distinguish full-run similarity from phase-localized confusion.

The method should not be used to artificially inflate sample size by randomly splitting windows.

Instead, it should be used with leave-one-recording-out validation to preserve a fair test.

The most important output is not only improved accuracy.

The most important output is a temporal explanation of workload behavior:

> Which parts of a recording look like the intended workload, and which parts look like something else?
