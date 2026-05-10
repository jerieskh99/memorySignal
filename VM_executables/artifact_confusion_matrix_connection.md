# Artifact-to-Confusion-Matrix Connection

This document maps the hidden runtime, kernel, and capture artifacts identified in `python_runtime_os_artifact_audit.md` onto the observed confusion patterns in `behavioral_classification_summary.txt` and the four-cause model in `confusion_matrix_diagnostic_methodology.md` Section 5.

Every claim is labeled as **hypothesis** or **evidence-supported**. The four causes are abbreviated:

- BO = true behavioral overlap
- PY = Python or runtime artifact
- OS = OS or kernel artifact
- VM = VM or capture artifact
- MI = metric inadequacy
- EO = execution-order contamination

---

## 1. Common Observability Premise

The Hamming and cosine deltas operate on per-page byte content between consecutive QEMU snapshots. **A page contributes signal only when its bytes change.** This is evidence-supported by the producer code path (`capture_producer_qemu_pmemsave.sh` plus the Rust delta program). Every confusion analysis below depends on this premise.

Three classes of pages contribute to the delta:

1. Pages the workload itself writes.
2. Pages the kernel writes on the workload's behalf (page cache fills, journal blocks, slab cache churn, page-table updates).
3. Pages anyone else writes during the snapshot interval (background OS, capture-side noise, residual writeback).

The relative weight of (1), (2), and (3) varies dramatically by workload and is the central explanatory axis for the observations.

---

## 2. Per-Confusion Analysis

### 2.1 `mem_stream → run_idle`

Observation: in 1 of 4 `mem_stream` runs, the predicted class is `run_idle` (`confusion_matrix_diagnostic_methodology.md` Section 9).

Competing hypotheses, ranked by plausibility:

| Cause | Plausibility | Reasoning |
|---|---|---|
| MI | high | 128 MB working set with 1-byte writes per page produces small per-page change. Most of guest RAM is static. Per-recording aggregation across 262K pages dilutes the active fraction toward background. |
| OS | medium | Hardware prefetcher and THP promotion reduce visible DRAM traffic. Per-page write delta is unchanged but the OS makes the workload less "memory-heavy" in any other view. |
| EO | medium | This run may have followed an IO-heavy cycle (`io_many_files`, `io_seq_fsync`) where writeback was still draining. The background was elevated, while the small active fraction stayed similar to background. |
| VM | low | pmemsave pause variance affects everything but does not specifically pull `mem_stream` toward IDLE. |
| PY | low | Python overhead is not dominant for this workload at the per-page level. |
| BO | very low | The workload genuinely writes pages; the underlying memory behavior is not the same as IDLE. |

The leading explanation is metric inadequacy plus OS-level prefetcher and THP behavior, both of which suppress the apparent magnitude of a workload that does, in fact, write pages.

Proposed isolation tests:

1. Re-record with `--mb 1024` instead of `--mb 128`. If a larger active fraction eliminates the IDLE confusion, MI is confirmed.
2. Disable THP guest-side and re-record. Compare per-page change distributions.
3. Re-record with the cycle order reversed so `mem_stream` does not follow IO-heavy workloads. Compare to current.
4. Add a counter in the workload that writes a non-trivial pattern (4-byte value or full cache line) per page touch. If the IDLE confusion disappears, MI dominated.

Thesis-safe wording:

> A `mem_stream` run was classified as `run_idle`. The workload writes one byte per page across a 128 MB region. At the dataset level this active region is a small fraction of total guest pages, and the per-page change magnitude is small. These factors are consistent with metric inadequacy when per-recording aggregation averages over many static pages. Capture-pipeline artifacts and execution-order residue cannot be excluded by the current data.

---

### 2.2 `mem_stream → io_rand_rw`

Observation: another `mem_stream` run was classified as `io_rand_rw`.

Competing hypotheses:

| Cause | Plausibility | Reasoning |
|---|---|---|
| EO | high | `mem_stream` runs first in each cycle. From cycle 2 onward it follows the previous cycle's `io_many_files` plus a 60 s idle. Page cache and dentry/inode reclaim from the prior IO-heavy cycle continues into the `mem_stream` window. The background looks I/O-like. |
| OS | medium | Kernel reclaim and writeback from prior workloads produce scattered dirty-page deltas during `mem_stream`. These dominate the small active fraction of the `mem_stream` workload itself. |
| MI | medium | The metrics' broadband behavior is shared between writeback churn and small-footprint active workloads. |
| VM | low | Capture pause variance does not selectively make `mem_stream` look like random I/O. |
| PY | low | Python overhead is dwarfed by kernel-mediated writeback. |
| BO | very low | `mem_stream` is not actually performing file I/O. |

Strong execution-order signature is consistent with the cycle position: cycle 2/3/4 `mem_stream` runs (test14, test26, test38) show declining `dc_coh` (0.345, 0.134, 0.133) while cycle 1 (test2) is highest (0.483). This is **evidence-supported** for EO contamination, though not proof.

Proposed isolation tests:

1. Insert a longer idle (5 minutes) before `mem_stream` and re-record. If the IO confusion drops, EO confirmed.
2. Run `mem_stream` standalone with a fresh VM boot. Compare to in-cycle runs.
3. Drop caches before `mem_stream` (`echo 3 > /proc/sys/vm/drop_caches`). Compare delta signatures.
4. Re-record with cycle order randomized. Check whether the confusion correlates with prior-workload identity.

Thesis-safe wording:

> A `mem_stream` run was classified as `io_rand_rw`. The cycle structure places `mem_stream` after a sequence of IO-heavy workloads, separated only by a 60 s idle. Page-cache reclaim and writeback from prior workloads can extend into the next workload window. This is consistent with execution-order contamination. The current data do not isolate this from genuine behavioral overlap or from metric inadequacy.

---

### 2.3 `mem_pointer_chase → mem_stream`

Observation: 1 of 4 `mem_pointer_chase` runs was classified as `mem_stream`.

Background fact: `mem_pointer_chase` is read-only. The workload reads bytes via `acc ^= int(arr[idx])` and never writes the buffer. **The intended workload is largely invisible to a content-delta pipeline.** This is evidence-supported by source inspection.

Competing hypotheses:

| Cause | Plausibility | Reasoning |
|---|---|---|
| MI | very high | The metric pipeline cannot see the chase activity directly. Whatever signal exists comes from interpreter and background sources, which can drift toward any nearby class boundary. |
| PY | high | The 1 GB random-data buffer is allocated and written ONCE at startup. This produces a brief but large write burst. Cycle-1 captured at-or-near this burst would look stronger and more `mem_stream`-like than cycle-3/4 runs that captured only steady-state reads. |
| EO | medium | Prior workload residue contributes to whatever signal exists. |
| OS | medium | Background writeback and kernel timers may produce sparse contiguous-looking writes that pattern-match `mem_stream`. |
| VM | low | Capture variance does not specifically push toward `mem_stream`. |
| BO | very low | Pointer-chase is not behaviorally similar to streaming at the user level. |

The dataset evidence supports the read-only invisibility hypothesis: across 4 runs, `snr_mean` is 2.583, 0.086, 0.052, 0.078. Three of four runs collapse to near-IDLE. The single elevated run (test4, cycle 1) is most plausibly capturing the 1 GB initialization write rather than the chase itself.

Proposed isolation tests:

1. Run `mem_pointer_chase` with a much larger working set and shorter init phase. Check whether all runs collapse equally.
2. Add a write-chase variant that writes the visited index to a sink page. Re-record. If signals stabilize, MI was the dominant cause.
3. Hash buffer content before and after the run. Confirm no writes during steady state.
4. Trim the first 30 s of each recording and recompute features. If the initialization-write hypothesis is right, the first-cycle outlier should reduce.

Thesis-safe wording:

> A `mem_pointer_chase` run was classified as `mem_stream`. The current `mem_pointer_chase` implementation reads pages without writing them. A page-content-delta pipeline does not directly observe read-only activity. The signal that does appear during the run is dominated by interpreter and kernel background, which can drift across class boundaries. This is consistent with metric inadequacy. The current confusion is not evidence of true behavioral overlap.

---

### 2.4 `io_rand_rw → mem_alloc_touch_pages`

Observation: 1 of 4 `io_rand_rw` runs was classified as `mem_alloc_touch_pages`.

Both workloads produce **scattered random dirty pages** as their dominant delta-pipeline signal. `io_rand_rw` dirties random page-cache pages via the kernel; `mem_alloc_touch_pages` dirties random anonymous pages via mmap-touch-unmap.

Competing hypotheses:

| Cause | Plausibility | Reasoning |
|---|---|---|
| BO | high | The two workloads share the gross signal shape: many small scattered page-content changes per snapshot interval. The kernel mediates one and the user mediates the other, but the per-page delta looks similar. |
| MI | high | The current feature set does not separate page-cache writes from anonymous page writes. |
| EO | medium | Prior workload writeback can elevate cache-related pages; the run with the confusion may have had heavier residual state. |
| OS | medium | Kernel allocator and page-cache state at the time of the run may have shifted closer to the allocator-churn signature. |
| PY | low | Python overhead is small relative to either kernel I/O or mmap. |
| VM | low | Not selectively biased. |

This confusion is the most plausible candidate for **genuine behavioral overlap** in the dataset. Both workloads stress the kernel memory subsystem with similar surface phenomena. The metric pipeline does not currently distinguish anonymous-mapping page changes from page-cache page changes.

Proposed isolation tests:

1. Stratify pages by VMA type (anonymous vs file-backed) before computing features. Re-classify. If the confusion disappears, MI is dominant; if not, BO.
2. Compare cepstral periodicity within the run: `mem_alloc_touch_pages` has a 20 ms allocator rhythm, `io_rand_rw` has no fixed cadence. A periodicity-aware feature should distinguish them.
3. Re-record `io_rand_rw` with `O_DIRECT` (bypassing page cache). If the confusion disappears, the page-cache mechanism caused the apparent overlap.
4. Cross-correlate per-page change positions across snapshots. Random-anonymous churn and random-page-cache churn may have different spatial autocorrelation patterns.

Thesis-safe wording:

> An `io_rand_rw` run was classified as `mem_alloc_touch_pages`. Both workloads produce scattered random page-content changes between snapshots, one through the page cache and one through anonymous mmap-touch-unmap cycles. The current per-page summary metrics do not encode the distinction between page-cache and anonymous mapping deltas. This confusion is consistent with either true behavioral overlap at the metric level or metric inadequacy at the feature level. These two are not distinguished by the present data.

---

### 2.5 Other patterns visible in the dashboard reports

`stochastic_characterization_summary.txt` and `behavioral_classification_summary.txt` document the following non-confusion patterns that connect directly to artifacts:

**`mem_pointer_chase` collapse across cycles 2, 3, 4** (test16/28/40 with `snr_mean ≈ 0.05`):

Hypotheses, ranked:

| Cause | Plausibility | Reasoning |
|---|---|---|
| MI | very high | Read-only workload invisible to delta pipeline; only init-phase write burst in cycle 1 produced a strong signal. |
| EO | low | Same prior workloads in every cycle; this should affect all four runs similarly, not selectively. |
| OS / VM | low | Same reason. |
| PY | low | Python is consistent across cycles. |
| BO | not applicable | |

This is essentially evidence-supported as MI dominated, given the source-code inspection.

**`mem_stream` cycle-1 dominance** (test2 `dc_coh = 0.483`, test38 `dc_coh = 0.133`):

Hypotheses, ranked:

| Cause | Plausibility | Reasoning |
|---|---|---|
| EO | high | Cycle 1 starts from a fresh VM with empty page cache. Cycle 4 starts after 4 cycles of accumulated state. The IO-heavy workloads in earlier cycles fill the page cache and shift the kernel allocator state. The relative signal of `mem_stream` against this elevated background drops. |
| OS | high | Kernel state drift across cycles produces a higher background. |
| VM | medium | Host memory pressure may rise across cycles. |
| PY | low | |
| MI | medium | |
| BO | low | |

**IDLE outliers test1 and test3** (silhouette -0.481 and -0.469 in `behavioral_classification_summary.txt`):

Hypotheses:

| Cause | Plausibility | Reasoning |
|---|---|---|
| EO | high | test1 is the very first IDLE in cycle 1 (initial 60 s before any active workload). test3 is the IDLE immediately after the first active workload (`mem_stream`). Both occur at cycle positions with strong residual signal: test1 may capture VM-boot residue, test3 captures `mem_stream` cooldown. |
| VM | medium | First-snapshot effects in QEMU. |
| OS | medium | Initial filesystem mount completion; `systemd` settling. |
| PY | low | No Python in IDLE. |
| MI | low | The metrics are valid; the input is just non-stationary. |
| BO | not applicable | |

---

## 3. Class-Level Scattering and Repeatability Analysis

The four required class-level questions, answered as competing hypotheses with proposed tests.

### 3.1 Why does IDLE not look consistently idle? (median CV = 0.973)

Observation: `run_idle` median CV across runs is 0.973 (`stochastic_characterization_summary.txt`). The single highest CV among all classes. test1 and test3 sit inside MEM/IO geometry (negative silhouette). `dc_coh` ranges 135-fold across IDLE runs (`stochastic_results_scientific_evaluation.md` W4).

Hypotheses, in decreasing plausibility:

**H1 (EO, high)**: 24 IDLE recordings come from different positions in the cycle. Each prior workload leaves different residual state at the moment the next idle starts:

| Prior workload | Likely residue during the following idle |
|---|---|
| `io_seq_fsync` | sustained writeback, journal flushes for tens of seconds |
| `io_rand_rw` | scattered page-cache reclaim, dirty-page flushing |
| `io_many_files` | dentry and inode cache reclaim, journal commits |
| `mem_alloc_touch_pages` | freed mmaps draining, slab churn |
| `mem_stream` | small region of warm pages cooling |
| `mem_pointer_chase` | 1 GB allocated random-data buffer still mapped (Python process exited so OS reclaims) |
| (cycle boundary) | accumulated residue from a full cycle |

Each idle window therefore measures a different "residual signature." This is **evidence-supported** by source code (each prior workload has different post-condition) but not yet quantified per-position.

**H2 (OS, medium)**: kernel timers, kthreads, and userspace daemons (`systemd-journald`, `crond`, `dbus`, `NetworkManager`) inject non-zero activity at unpredictable intervals. This is true for every system but produces variable signatures across captures.

**H3 (VM, medium)**: pmemsave pause variance and host scheduling create a background noise floor that is not constant across recordings.

**H4 (capture-window-position artifact, medium)**: the IDLE recording is bracketed by the previous workload's stop and the next workload's start. Capture timing relative to these boundaries is not perfectly aligned across runs.

**H5 (BO, not applicable)**.

Why this matters for confusion patterns:

- An IDLE recording with strong residue can have signal levels overlapping with weak active workloads. This produces `mem_stream → run_idle` confusions: the boundary is ambiguous, and runs near the boundary go either way.
- Conversely, an active workload with weak signal (e.g. `mem_pointer_chase`) ends up close to the high-residue IDLE cluster.

Proposed isolation tests:

1. Stratify the 24 IDLE recordings by their immediately preceding workload. Compute per-stratum CV.
2. Insert a long warm-up idle (5+ minutes) at the start of each cycle. Compare to current.
3. Run a control cycle of all-IDLE recordings with no active workloads in between. Establish a "true IDLE" baseline.
4. Snapshot `/proc/meminfo` `Dirty`, `Writeback`, `MemAvailable` at the start and end of each idle. Correlate with feature drift.

Thesis-safe wording:

> IDLE recordings show high run-to-run variability (median CV = 0.973). The IDLE windows are not isolated control conditions; they are recorded between active workloads, separated by 60 s gaps. Each prior workload leaves a different residual state in the page cache, allocator, and filesystem caches that the kernel reclaims gradually. The IDLE signal therefore depends on the cycle position of the recording, not only on the absence of an active workload. Background OS daemons and capture-pipeline pause variance contribute additional non-stationarity. These three factors are not separated in the current dataset.

---

### 3.2 Why does MEM scatter more than IO? (MEM silhouette = -0.258, MEM-IO centroid distance = 0.301)

Observation: MEM silhouette is negative and MEM-IO centroid distance is far smaller than within-class spread (`behavioral_classification_summary.txt`). MEM subtypes have median CVs from 0.05 to 0.62; IO subtypes have median CVs from 0.05 to 0.19.

Hypotheses, decreasing plausibility:

**H1 (MI, very high)**: One of the three MEM subtypes (`mem_pointer_chase`) is read-only and largely invisible to the delta pipeline. The class label "MEM" is therefore being applied to a mix of well-observed and barely-observed runs. The class centroid is pulled in unpredictable directions by the invisible-workload runs. This is **evidence-supported** by source and by the test4-vs-test16/28/40 outlier pattern.

**H2 (PY+OS, high)**: The MEM workloads operate in user-space against memory regions whose visibility to the kernel is strongly mediated by:
- Lazy page commitment and CoW from the zero page (initial sweep timing varies).
- Hardware prefetcher behavior on `mem_stream` (smooths and reduces apparent activity).
- Transparent Huge Pages promotion (changes per-4 KB page granularity mid-run, alters delta computation).
- glibc `M_MMAP_THRESHOLD` dynamic adjustment in `mem_alloc_touch_pages` (changes syscall frequency mid-run).
- TLB miss rates and cache eviction patterns vary with overall guest-host contention.

These are NOT controlled by the workload; they emerge from the runtime stack.

**H3 (memory-pressure variability, medium)**: 1 GB allocation in `mem_pointer_chase` may push other things into reclaim. 500 MB per batch in `mem_alloc_touch_pages` produces variable host memory pressure. IO workloads do not allocate large anonymous regions; their kernel-mediated buffers are smaller and more deterministic in size.

**H4 (capture-side jitter, medium)**: pmemsave pause hits user-space loops at varying instruction positions. For tight Python loops, the interruption may land mid-numpy-op or mid-allocation, perturbing the next snapshot's delta. IO workloads spend more time in the kernel where pause behavior is structurally similar across runs.

**H5 (cycle-position drift, medium)**: cycle-to-cycle drift visible in `mem_stream` `dc_coh` (test2 = 0.483 down to test38 = 0.133) suggests systematic state buildup. IO workloads also cycle but their kernel-mediated rhythms are robust to this.

**H6 (BO, low)**: MEM and IO classes do share generic "system activity" features. But the centroid collapse is too extreme to attribute to BO alone.

Why this connects to the observed scatter:

- The high-MI source (`mem_pointer_chase`) injects two different point-clouds (active test4 versus collapsed test16/28/40) into the MEM class.
- The PY+OS sources (`mem_stream`) inject cycle-correlated drift.
- Only `mem_alloc_touch_pages` produces a clean, repeatable signal (median CV = 0.054), and this is the one MEM subtype that most closely mimics a kernel-mediated rhythm via its mmap/munmap cadence.

Proposed isolation tests:

1. Add the write-chase variant of `mem_pointer_chase`. If MEM scatter drops, MI was dominant.
2. Disable THP and re-record `mem_stream`. Compare per-run feature stability.
3. Lock glibc allocator (`MALLOC_MMAP_THRESHOLD_=131072 MALLOC_MMAP_MAX_=4096`) and re-record `mem_alloc_touch_pages`. If CV drops further, allocator nondeterminism contributed.
4. Re-record only the three MEM subtypes in a tight cycle without IO interleaving. Compare MEM scatter.
5. Force a specific page-size policy (`madvise(MADV_NOHUGEPAGE)` via a wrapper) and re-record.

Thesis-safe wording:

> MEM subtypes show greater run-to-run scatter than IO subtypes. The MEM class includes one read-only workload (`mem_pointer_chase`) that the page-content-delta pipeline cannot directly observe; its run-level features are dominated by background activity rather than the intended stimulus. The remaining two MEM workloads operate in user space against memory regions whose visibility is shaped by lazy page commitment, hardware prefetching, Transparent Huge Pages promotion, and glibc allocator heuristics. These factors are not controlled by the workload code and introduce per-run variability. Genuine behavioral overlap between MEM and IO at the metric level cannot be excluded but is not the leading hypothesis.

---

### 3.3 Why do IO subtypes cluster more cleanly?

Observation: `io_many_files` median CV = 0.048, `io_seq_fsync` = 0.092, `io_rand_rw` = 0.189. All three IO subtypes have several stable metrics with CV < 0.15 (`stochastic_characterization_summary.txt`).

Hypotheses, decreasing plausibility:

**H1 (kernel-mediated rhythm, very high)**: All IO workloads drive the kernel through deterministic syscall sequences:

- `io_seq_fsync`: write 4 MB → fsync. Repeats. The fsync forces journal commit and writeback. The cadence is dominated by disk latency, which is more stable than user-space loop timing.
- `io_many_files`: open → write → close → unlink × 500 per batch. Filesystem journal rate is the bottleneck.
- `io_rand_rw`: 64 KB read or write at random offsets. Page-cache hit/miss rate settles into a steady distribution.

The kernel imposes a regular and repeatable cadence. User-space variability is a small perturbation on a kernel-dominated rhythm. This is the cleanest explanation for the observed CV differences and is **strongly supported by source-code inspection**.

**H2 (high-amplitude content writes, high)**: IO writes large amounts of data to pages (4 MB chunks for `io_seq_fsync`, 64 KB blocks for `io_rand_rw`, 1 KB payloads for `io_many_files` plus journal blocks of typically 4 KB). The per-page Hamming and cosine deltas are large, producing high SNR signal that is robust to small perturbations.

By contrast, MEM workloads change one byte per page (`mem_stream`, `mem_alloc_touch_pages` touch loop) or zero bytes per page (`mem_pointer_chase` chase loop). Per-page deltas are tiny, dragged toward background by averaging.

**H3 (Python overhead small relative to kernel), high**: For IO workloads, Python's per-iteration overhead is dominated by syscall and disk latency. Switching to C would change runtime by single-digit percent. The signal is therefore largely independent of language choice.

**H4 (filesystem journal as a synchronization clock, medium)**: ext4 (or whichever journal-based FS is mounted) commits the journal at regular intervals or per fsync. This effectively imposes a hardware-clock-like cadence on the workload. The MEM workloads have no such external clock; they depend entirely on the loop period set by Python.

**H5 (BO with kernel internals, low-but-present)**: `io_rand_rw` resembles `mem_alloc_touch_pages` because both produce scattered random page changes. The IO confusion observed (`io_rand_rw → mem_alloc_touch_pages`) is the one place where IO clean-clustering breaks down. This is consistent with H4 being weakest for `io_rand_rw` (no rhythmic external clock; access is random).

Implication for the IO-subtype median-CV ordering:

- `io_many_files` (CV = 0.048) and `io_seq_fsync` (CV = 0.092) have explicit batch and fsync rhythms.
- `io_rand_rw` (CV = 0.189) has no explicit rhythm; cache hit rate variability across runs reduces stability.

This is **evidence-supported**: the within-IO ordering matches the rhythm-strength ordering.

Proposed isolation tests:

1. Vary `fsync-wait` parameter in `io_seq_fsync` (1, 4, 16). Stronger rhythm should produce lower CV.
2. Vary `files-per-batch` in `io_many_files`. Larger batches change rhythm; check CV response.
3. Cross-correlate per-recording snapshot timestamps with workload syscall traces (added via timestamped writes to a log page). Confirm rhythmic alignment.
4. Run `io_seq_fsync` with `O_SYNC` instead of explicit fsync. Compare CV.
5. Run an in-RAM `io_seq_fsync` variant on tmpfs (no actual disk). If CV stays similar, kernel rhythm dominated; if it changes, disk latency contributed.

Thesis-safe wording:

> IO subtypes show lower run-to-run scatter than MEM subtypes. Each IO workload drives the kernel through repeated syscall sequences that the kernel mediates with deterministic structure: filesystem journal commits, page-cache writeback, disk-flush barriers. The kernel imposes a cadence that user-space variability does not easily perturb. IO workloads also write large amounts of data to pages, producing per-page deltas that are large relative to background noise. These two factors plausibly explain the higher within-subtype repeatability and the cleaner inter-subtype separation observed for IO compared to MEM.

---

### 3.4 Connecting CV, Scatter, Clustering, and Confusion Direction

The above three subsections converge on a single picture. The observed patterns are not independent.

| Pattern | Likely dominant cause |
|---|---|
| IDLE high CV (0.973) | EO across cycle positions plus OS background variability |
| MEM scatter (silhouette = -0.258) | MI from read-only `mem_pointer_chase` plus PY+OS variability for `mem_stream` |
| IO clean clustering (CV 0.05 to 0.19) | Kernel-mediated rhythm imposing an external clock |
| `mem_stream → run_idle` | MI from small active fraction plus EO from prior IO residue |
| `mem_stream → io_rand_rw` | EO from prior IO writeback dominating the small `mem_stream` signal |
| `mem_pointer_chase → mem_stream` | MI from read-only invisibility |
| `io_rand_rw → mem_alloc_touch_pages` | BO at the metric level (both produce scattered random page changes) plus MI |

Direction asymmetries are informative:

- All observed confusions involve the lower-signal class being predicted as the higher-signal class, OR a workload with a similar surface signal stealing the prediction.
- `mem_stream` is "pulled toward" IDLE and `io_rand_rw`. `mem_pointer_chase` is "pulled toward" `mem_stream`. `io_rand_rw` is "pulled toward" `mem_alloc_touch_pages`.
- This pattern is consistent with class boundaries running along **active-fraction gradients** rather than along behavioral gradients. The current metrics see "how much page activity" more cleanly than "what kind of page activity."

This is also **evidence-supported** by the LDA findings in `behavioral_classification_summary.txt` (LD1 weights dominated by `cepstral_low_frac`, `cepstral_var_zero_frac`, `snr_skewness`: all activity-magnitude indicators) and by the silhouette result that IDLE separates well while MEM and IO collapse together.

The implication for next-generation tests:

1. The MEM class needs to inject more visible signal into the delta pipeline. The single low-MI fix is to ensure every MEM workload writes pages, not only reads them. `mem_pointer_chase` should be redesigned to chase-and-write.
2. The MEM class needs more deterministic cadence to compete with IO's kernel-mediated rhythms. Adding explicit timing structure (sleep barriers, batch boundaries) at known frequencies would give MEM workloads a similar cepstral fingerprint shape.
3. The IDLE class needs to be controlled, not measured. Either (a) record a dedicated long IDLE baseline before any active workload, (b) record IDLE only after a fixed-length warm-up, or (c) stratify IDLE by prior-workload context and analyze each stratum separately.

---

## 4. What This Audit Does Not Resolve

These connections remain hypotheses, not conclusions, until the proposed isolation tests are run.

- The relative weight of MI versus PY versus OS for `mem_stream` cycle-1 dominance.
- The relative weight of BO versus MI for `io_rand_rw → mem_alloc_touch_pages`.
- Whether THP promotion mid-run actually changes per-4-KB-page deltas in measurable ways for `mem_stream`.
- Whether glibc allocator heuristics shift mid-run in `mem_alloc_touch_pages`.
- Whether `/tmp` is tmpfs in the guest (changes the entire interpretation of `io_many_files`).
- The size of pmemsave pause variance and its correlation with delta-frame outliers.

The proposed tests in Sections 2 and 3 are designed to reduce these remaining uncertainties without modifying any current dataset.

---

## 5. Thesis-Safe Summary Wording

> Observed confusions and class-scatter patterns were analyzed against four candidate causes from `confusion_matrix_diagnostic_methodology.md` Section 5. The leading explanations differ by confusion. `mem_pointer_chase → mem_stream` is most plausibly a metric-inadequacy artifact, because the current `mem_pointer_chase` implementation does not write pages and is therefore largely invisible to a content-delta observation pipeline. `mem_stream` confusions with IDLE and `io_rand_rw` are consistent with a combination of metric inadequacy and execution-order contamination, given the small active fraction of the workload and its position in the cycle. `io_rand_rw → mem_alloc_touch_pages` is the most plausible candidate for genuine behavioral overlap at the metric level, because both workloads produce scattered random page-content changes that the current per-recording aggregation does not separate by mapping type. The class-level pattern (clean IO, scattered MEM, non-stationary IDLE) is consistent with kernel-mediated I/O rhythms producing more stable signatures than user-space MEM loops, with read-only and small-footprint MEM workloads contributing the largest within-class scatter. None of these claims are isolated by the current data; the proposed control experiments would distinguish among the candidate causes.
