# Next-Generation Workload Test Design Review

This document critiques `next_gen_workload_test_mini_specs_filled.md`. It is a separate critical evaluation, not a rewrite of the spec.

The review uses the findings established in `python_runtime_os_artifact_audit.md`, `artifact_confusion_matrix_connection.md`, `low_level_rewrite_recommendations.md`, and `confusion_matrix_diagnostic_methodology.md`. Where the filled spec overstates or under-justifies a claim, the review says so explicitly.

---

## 1. Executive Critique

The filled spec is internally consistent and well-structured. Its weaknesses are:

1. **Test count inflation**. 25 tests is too many. Several pairs are substantively the same probe at different parameters.

2. **Multiple "near-invisible" tests duplicate one finding**. `cpu_hash_loop`, `cpu_branch_random`, `io_read_cache_hit`, and the read variants of `cache_*` and `mem_pointer_chase` legacy all converge to "indistinguishable from idle". One canonical invisibility-demonstration probe is sufficient. The others repeat one negative result.

3. **Cache-vs-DRAM observability is under-treated**. The pmemsave snapshot reads physical RAM. Dirty data resident in CPU caches may or may not appear in the dump depending on whether the pause flushes the cache. This affects every test that depends on cache-resident writes. The filled spec mentions this for `cache_hot_loop` but does not propagate it to `mem_stream` MOVNTI, `mem_pointer_chase` redesign, or any cache test broadly.

4. **THREAD and NETWORK families assume observable signal without justification**. Kernel `futex` traffic, `sk_buff` allocations, and per-CPU kernel structures may or may not produce delta-pipeline-visible signal. The filled spec specs these tests as if they will produce signal. This is a hypothesis, not a finding.

5. **MIXED tests are pre-determined to resemble their dominant parent**. The filled spec acknowledges this for `mixed_cpu_mem` and `mixed_cpu_io`. As written, these tests confirm a tautology rather than test feature-space continuity.

6. **The Validation Strategy contradicts prior critique**. It mandates leave-one-recording-out LDA validation. `stochastic_results_scientific_evaluation.md` flagged LDA as in-sample overfitting and suggested LORO with reported accuracy as the verification. The filled spec adopts LORO but does not reconcile with the LDA-overfitting concern.

7. **Cyclic dependency in baseline reference**. The Validation Strategy says metrics should be computed relative to `idle_long_baseline`. But `idle_long_baseline` is itself a proposed test. Until it is implemented and characterized, the baseline reference must be the current `run_idle` recordings, with all their known contamination.

8. **Cleanliness expectations are over-confident for unimplemented tests**. "Provisionally medium-high" appears without empirical support for proposed tests. This contradicts the n < 8 provisional language rule from `confusion_matrix_diagnostic_methodology.md` Section 6.

9. **Implementation-level designations are inconsistently applied**. Tests with similar artifact profiles are labeled with different urgency levels without a stated criterion.

10. **Assembly justifications are weak**. The two assembly cases (`mem_stream` MOVNTI, `mem_pointer_chase` dependent-load chain) are introduced as solutions to problems that have not yet been demonstrated. Each adds implementation complexity for a hypothesis that should first be tested in pure C.

---

## 2. Global Critique of the Filled Spec

### 2.1 Strongest parts

- The extended template fields are well-designed. Adding Expected confusions, artifact risks, and Cleanliness expectation makes each spec falsifiable.
- The typed expected confusions follow the four-cause model from `confusion_matrix_diagnostic_methodology.md` Section 13. Direction notation is correct throughout.
- The Common Observability Premise at the top correctly anchors all reasoning. Read versus write asymmetry is the load-bearing constraint.
- Existing tests are updated against Phase 1 evidence rather than copied. `mem_pointer_chase` is correctly marked **flagged for redesign**.
- The `idle_long_baseline` and `idle_post_workload_recovery` proposals are well-motivated by the IDLE CV finding.
- Implementation-level distinction between "Python disqualified" (THREAD), "C strongly recommended" (most MEM), and "Python acceptable" (idle, simple compute) is mostly defensible.

### 2.2 Weak or overcomplicated parts

- The total test count is excessive. Realistic implementation budget given a thesis timeline is 8 to 12 tests, not 25.
- Three CPU tests duplicate the same negative finding. CPU class is already conceptually weak because it is mostly invisible to the pipeline.
- Three CACHE tests overlap with three MEM tests. The cache-vs-MEM family boundary is not distinct in observable signal.
- Three MIXED tests are predicted to resemble their parents. As classification tests, they offer little. As feature-space probes, they need different design.
- The NETWORK family is speculative. No evidence that loopback network stack memory is observable. Three tests are committed before that question is answered.
- THREAD family is similar. Multi-threading produces signal IF kernel data structures are visible, which is unverified.

### 2.3 Missing controls

- No test characterizes the **capture pipeline pause artifact** itself. The filled spec lists this as risk but offers no probe. A `capture_calibration` workload that does nothing for 300 s and is captured at varying snapshot rates would directly measure pause-induced signal.
- No test characterizes **cache-vs-DRAM observability**. A controlled experiment with cache-flush vs no-flush variants would resolve whether cache-resident dirty data is observable.
- No test characterizes **execution-order contamination as a function of idle duration**. The filled spec adds `idle_post_workload_recovery` but only measures decay shape, not the parameter-recovery curve.
- No test characterizes **THP promotion timing**. `khugepaged` runs every 10 seconds by default. Whether THP promotion actually changes observable per-page deltas mid-run is an unverified hypothesis.
- No test characterizes **glibc M_MMAP_THRESHOLD switching directly**. The spec proposes bypassing it in `mem_alloc_touch_pages` but does not propose measuring its effect.

### 2.4 Risky assumptions

- The spec assumes that "kernel-mediated rhythm produces clean signals" generalizes to all kernel mechanisms. The current data show this for filesystem journal and page cache. It does not necessarily extend to network stack, futex traffic, or scheduler activity.
- The spec assumes that adding a write makes a workload visible. This is necessary but may not be sufficient. Cache-resident writes may not propagate to DRAM by snapshot time.
- The spec assumes 8 to 10 runs per subtype is enough for provisional cleanliness. With 25 subtypes that is 200 to 250 recordings. At 300 s active workload plus 60 s idle plus capture overhead, that is ~30 hours of recording time minimum, not counting analysis. The cost of this commitment is not addressed.
- The spec assumes proposed tests will produce the predicted feature-space positions. Predictions are reasonable hypotheses but should not be reported as expected outcomes without empirical anchoring.

### 2.5 Tests that may not be worth implementing

By cost-benefit analysis given thesis timeline:

- `cpu_hash_loop` AND `cpu_branch_random`: redundant invisibility demonstrations. Keep one canonical.
- `mem_stride_sweep_large`: subsumable into `cache_stride_sweep` with parameterized buffer size.
- `cache_cold_scan`: subsumable into `mem_stream` variants at large working set.
- `io_read_cache_hit`: another invisibility demonstration. Marginal additional value.
- All three `mixed_*` tests as currently designed: pre-determined to resemble parents.
- All three `net_*` tests until pre-verification confirms loopback network stack signal is observable.

---

## 3. Per-Family Review

---

### 3.1 IDLE Family

**Purpose**: Establish a baseline and characterize residual or background memory behavior in the absence of an active workload.

**Why it matters**: All active workload features are interpreted relative to the IDLE baseline. If IDLE is contaminated or non-stationary, every active claim inherits that uncertainty.

**What it isolates**: OS background activity, kernel timer cadences, residual writeback decay, capture-pipeline pause artifacts.

**What it accidentally mixes in**: Prior workload residue (the dominant confound), non-stationary scheduler activity, host noise.

**Cleanliness expectation**: Provisionally low until protocol changes are validated. The current dataset shows median CV 0.973.

**Thesis question this family helps answer**: Is the IDLE class stationary enough to serve as a reference, or does it require stratification by prior context?

**First to implement**: `idle_long_baseline`. Without a controlled baseline, every other comparison is contaminated.

**Delay or merge**: `idle_post_workload_recovery` is diagnostic. It can be implemented after `idle_long_baseline` to characterize residue decay.

---

#### `run_idle` (existing)

**Scientific value**: Medium. It IS what the dataset has. Its scattering across cycle positions is itself a useful finding for IDLE non-stationarity. The protocol around it is the variable, not the script.

**Cleanliness critique**: Provisionally low. CV 0.973 is the highest of all classes. test1 and test3 sit in MEM/IO geometry geometrically. The filled spec acknowledges this correctly.

**Likely confusions**:
- `mem_stream → run_idle`, `mem_pointer_chase → run_idle` (legacy): driven by metric inadequacy.
- IDLE-IDLE intra-class split by prior workload: execution-order contamination.

**Artifact risks**:
- Python/runtime: very low (bash sleep).
- OS/kernel: very high (the OS IS the signal).
- VM/capture: medium (pause variance still present).
- Execution-order contamination: very high.
- Metric inadequacy: medium (the metrics are valid; the input is non-stationary).
- True behavioral overlap: not applicable.

**Should split into variants**: Yes. The 24 existing recordings should be analytically stratified by prior workload identity. This is an analysis-side change, not a code change.

**Low-level controls**: Cache drop before each idle. Idle duration of at least 120 s. Metadata sidecar with prior-workload identity. Snapshot of `/proc/meminfo` `Dirty`/`Writeback` at start and end.

**C vs assembly**: Neither helps. Bash is sufficient.

**Expected metric effect after protocol fix**: CV drop from ~0.97 toward IO-class levels. Mean signal level decrease. test1, test3 silhouette improvement.

**Final recommendation**: **Keep but modify the protocol**. No code change. Stratify existing recordings. Apply protocol changes for next collection.

---

#### `idle_long_baseline` (proposed)

**Scientific value**: High. This is the missing controlled baseline. Many subsequent comparisons depend on it.

**Cleanliness critique**: Should achieve the cleanest possible IDLE signature. If it does not, then no IDLE recording can be clean and the IDLE class is fundamentally a residue measurement. Either result is informative.

**Likely confusions**:
- `idle_long_baseline → run_idle`: behavioral proximity (same mechanism, different residue level).
- `idle_long_baseline → cpu_hash_loop`: metric inadequacy (both produce minimal observable delta).

**Artifact risks**:
- Python/runtime: very low.
- OS/kernel: medium (still includes daemons, timers).
- VM/capture: medium (pause variance unchanged).
- Execution-order contamination: low by design.
- Metric inadequacy: medium (metrics may have a noise floor).
- True behavioral overlap: low.

**Should split into variants**: Optional. A "fresh boot" variant and a "post long warm-up" variant could be compared.

**Low-level controls**: Pre-window settle of 60 to 90 s. Cache drop. Verify `Dirty == 0` and `Writeback == 0` before measurement. No active workload within 5 minutes prior.

**C vs assembly**: Neither needed.

**Expected metric effect**: Lower mean event rate than `run_idle`, lower CV across runs.

**Final recommendation**: **Implement first**. This is the foundation for every subsequent comparison.

---

#### `idle_post_workload_recovery` (proposed)

**Scientific value**: Medium-high as a diagnostic. Low as a classification class member.

**Cleanliness critique**: Not a clean probe. By design it captures residue decay. Useful for measuring decay shape per prior workload, not for adding a new class.

**Likely confusions**:
- `idle_post_workload_recovery_after_X → X`: execution-order contamination. By design.

**Artifact risks**:
- Python/runtime: very low.
- OS/kernel: very high (intended).
- VM/capture: medium.
- Execution-order contamination: very high (this IS the test).
- Metric inadequacy: low.
- True behavioral overlap: medium.

**Should split into variants**: Yes. One variant per prior workload identity. The decay-curve-versus-prior-workload comparison is the scientific output.

**Low-level controls**: Precise transition timing. Per-segment metric capture for decay curve analysis (refer to `segment_level_analysis_critique_and_plan.md`).

**C vs assembly**: Neither needed.

**Expected metric effect**: Decreasing event rate from segment 1 to segment k. Slope is the diagnostic feature.

**Final recommendation**: **Keep but implement after idle_long_baseline**. Treat as a diagnostic experiment, not a classification class.

---

### 3.2 MEM Family

**Purpose**: Stress volatile memory directly with controlled access patterns.

**Why it matters**: Central to the thesis. Tests whether the memory signal distinguishes access STRUCTURE versus access VOLUME.

**What it isolates**: User-space memory access patterns. With the redesigns, page-write structure becomes the signal.

**What it accidentally mixes in**: Hardware prefetcher behavior, THP promotion, allocator nondeterminism, capture-side artifacts.

**Cleanliness expectation**: Provisionally medium after C rewrites. `mem_alloc_touch_pages` provisionally high.

**Thesis question**: Can volatile memory metrics distinguish sequential, random, and allocation-driven access patterns?

**First to implement**: `mem_pointer_chase` redesign (the read-only invisibility is the leading cause of MEM scatter).

**Delay or merge**: `mem_stride_sweep_large` is subsumable into `cache_stride_sweep` with a parameterized buffer size.

---

#### `mem_stream` (existing)

**Scientific value**: High. The canonical sequential write probe.

**Cleanliness critique**: Currently provisionally weak (CV 0.528, three of four runs collapsing in cycles 3-4). The 1 GB rewrite is expected to fix this. **Expected** is a hypothesis, not a finding.

**Likely confusions**:
- `mem_stream → run_idle`: metric inadequacy at small footprint. Should disappear at 1 GB.
- `mem_stream → io_rand_rw`: execution-order contamination from prior IO. Will only resolve with IDLE protocol fix.
- `mem_stream → cache_cold_scan`: behavioral proximity if both implemented; same mechanism.

**Artifact risks**:
- Python/runtime: medium-high (currently). Low after C.
- OS/kernel: medium (THP, prefetcher remain).
- VM/capture: medium.
- Execution-order contamination: high.
- Metric inadequacy: high in current form.
- True behavioral overlap: medium with cache_cold_scan write variant.

**Should split into variants**: Yes. The filled spec lists 5 variants. Two are sufficient: `stream_large_cached` and `stream_large_nt`. The other three are redundant or marginal.

**Low-level controls**: 1 GB working set, MAP_POPULATE warm-up, MADV_NOHUGEPAGE, mlock, CPU pinning.

**C vs assembly**: C strongly recommended. Assembly only if cache-vs-DRAM observability turns out to be a measurable concern. Test pure-C version first; only escalate to MOVNTI if cache hides writes.

**Expected metric effect**: Higher dc_coherence and snr_mean at 1 GB. Lower CV.

**Final recommendation**: **Implement first** (in priority block). Two variants only.

---

#### `mem_pointer_chase` (existing)

**Scientific value**: Currently low (read-only invisibility). High after redesign.

**Cleanliness critique**: Currently provisionally weak. Three of four runs collapse to near-IDLE. The redesign with writes is mandatory.

**Likely confusions** (post-redesign):
- `mem_pointer_chase → mem_alloc_touch_pages`: behavioral proximity (both scatter random writes).
- `mem_pointer_chase → io_rand_rw`: behavioral proximity (similar surface signal).
- `mem_pointer_chase → mem_random_write_pages`: behavioral proximity (subsumable).

**Artifact risks**:
- Python/runtime: high in current form, low in C.
- OS/kernel: medium.
- VM/capture: medium.
- Metric inadequacy: very high in legacy. Medium in redesign.
- True behavioral overlap: high with `mem_random_write_pages`.

**Should split into variants**: The redesign overlaps significantly with `mem_random_write_pages`. Consider merging: keep `mem_pointer_chase` as the LCG-deterministic-write variant and `mem_random_write_pages` as the PRNG-randomized-write variant. They differ only in PRNG choice.

**Low-level controls**: mmap + MAP_POPULATE, MADV_NOHUGEPAGE, mlock. Add per-page write of LCG state byte.

**C vs assembly**: C strongly recommended. Assembly justification (dependent-load chain) is weak unless empirical evidence shows compiler reordering hides the chase.

**Expected metric effect after redesign**: dc_coherence elevated, event rate high, broadband cepstral content.

**Final recommendation**: **Redesign and implement first**. Reframe as the LCG-write variant of a unified random-write subtype.

---

#### `mem_alloc_touch_pages` (existing)

**Scientific value**: High. Cleanest current MEM probe.

**Cleanliness critique**: Provisionally high (CV 0.054, 4/4 correct). The C rewrite tightens this further but the marginal value is small. The current Python version is already empirically clean.

**Likely confusions**:
- `io_rand_rw → mem_alloc_touch_pages`: behavioral proximity; the only IO confusion.
- `mem_alloc_touch_pages → io_seq_fsync`: behavioral proximity; not currently observed.

**Artifact risks**:
- Python/runtime: medium (allocator threshold switching).
- OS/kernel: medium-high (intended).
- VM/capture: medium.
- Metric inadequacy: medium.
- True behavioral overlap: high with `io_rand_rw`.

**Should split into variants**: The filled spec proposes 4 variants. Two are sufficient: `alloc_mmap_byte` (canonical) and `alloc_mmap_dontneed` (arena-emulation). The MAP_POPULATE variant separates fault-vs-touch and has scientific value but lower priority.

**Low-level controls**: Direct mmap/munmap, no glibc allocator. nanosleep cadence.

**C vs assembly**: C recommended. Assembly not justified.

**Expected metric effect**: Marginal CV improvement.

**Final recommendation**: **Keep as-is for now, rewrite later in the priority block**. Already clean. Lower urgency than fixing broken probes.

---

#### `mem_random_write_pages` (proposed)

**Scientific value**: Medium. Heavily overlaps with `mem_pointer_chase` post-redesign.

**Cleanliness critique**: Predicted "provisionally high" without evidence. This is overconfidence.

**Likely confusions**:
- Heavy overlap with `mem_pointer_chase` post-redesign.
- `mem_random_write_pages → mem_alloc_touch_pages`: behavioral proximity.
- `mem_random_write_pages → io_rand_rw`: behavioral proximity.

**Artifact risks**: Similar to `mem_pointer_chase` redesign.

**Should split into variants**: Or **merge** with `mem_pointer_chase`. The two as separate subtypes do not address distinct scientific questions if both write per-access. The PRNG choice (LCG vs xoshiro) is unlikely to produce a distinguishable signal.

**Low-level controls**: Same as `mem_pointer_chase` redesign.

**C vs assembly**: C strongly recommended.

**Expected metric effect**: Same as `mem_pointer_chase` post-redesign.

**Final recommendation**: **Merge with mem_pointer_chase**. Treat as one subtype with two variants (LCG vs PRNG). Single-subtype implementation reduces cost without scientific loss.

---

#### `mem_stride_sweep_large` (proposed)

**Scientific value**: Medium. Heavily overlaps with `cache_stride_sweep`.

**Cleanliness critique**: Per-stride provisional medium. Each stride is a separate scientific question.

**Likely confusions**:
- `mem_stride_sweep_large (4 KB stride) → mem_stream`: behavioral proximity.
- `mem_stride_sweep_large (large stride) → mem_random_write_pages`: behavioral proximity.

**Artifact risks**: Same as MEM family overall plus prefetcher dependence on stride.

**Should split into variants**: Already structured as a stride sweep. The sweep is the experiment.

**Low-level controls**: Buffer 2 to 8 GB, parameterized stride, MADV_NOHUGEPAGE.

**C vs assembly**: C strongly recommended.

**Expected metric effect**: Per-stride cepstral peak shift.

**Final recommendation**: **Merge with cache_stride_sweep**. The two differ only in buffer size relative to LLC. One parameterized test with buffer-size-as-axis covers both questions.

---

### 3.3 IO Family

**Purpose**: Stress the kernel filesystem and storage path.

**Why it matters**: The IO class clusters cleanly in current results because the kernel imposes deterministic cadence. Testing whether this generalizes beyond filesystem (to network) is a key thesis question.

**What it isolates**: Page cache, dirty pages, journal commits, dentry/inode caches, block-layer activity.

**What it accidentally mixes in**: Page-cache hit-rate variability across cycles; tmpfs-vs-disk ambiguity for `io_many_files`.

**Cleanliness expectation**: Provisionally high for `io_many_files` and `io_seq_fsync`. Provisionally medium for `io_rand_rw`.

**Thesis question**: Are kernel-mediated rhythms reproducible enough to serve as a behavioral fingerprint regardless of language?

**First to implement**: `io_many_files` with explicit FS choice. The tmpfs-vs-disk ambiguity in current data is a scientific validity blocker.

**Delay or merge**: `io_read_cache_hit` adds little beyond invisibility-demonstration. Subsumable into a generic invisibility probe.

---

#### `io_rand_rw` (existing)

**Scientific value**: Medium-high.

**Cleanliness critique**: Provisionally medium (CV 0.189). The fallocate + cache-warmth control is expected to tighten this.

**Likely confusions**:
- `io_rand_rw → mem_alloc_touch_pages`: behavioral proximity. The only IO confusion.

**Artifact risks**:
- Python/runtime: low.
- OS/kernel: high (intended).
- VM/capture: medium-high.
- Execution-order contamination: high (cache warmth from prior runs).
- Metric inadequacy: high (does not separate page-cache from anonymous).
- True behavioral overlap: high with `mem_alloc_touch_pages`.

**Should split into variants**: Yes. The filled spec proposes 6 variants. Three are essential: `rand_buffered_cold`, `rand_writeonly`, `rand_direct`. The others can be follow-up.

**Low-level controls**: fallocate, pread/pwrite, posix_fadvise, optional O_DIRECT.

**C vs assembly**: C recommended (specifically for O_DIRECT and posix_fadvise).

**Expected metric effect**: CV improvement with explicit cache-state control. O_DIRECT variant produces a structurally different signal.

**Final recommendation**: **Keep, implement with three variants**. The O_DIRECT variant directly tests the BO-vs-MI hypothesis for the `io_rand_rw → mem_alloc_touch_pages` confusion.

---

#### `io_seq_fsync` (existing)

**Scientific value**: Medium. Already clean.

**Cleanliness critique**: Provisionally high (CV 0.092, 4/4). The C rewrite is marginal.

**Likely confusions**: Currently none observed. Predicted overlap with `mem_alloc_touch_pages` is plausible at higher n.

**Artifact risks**:
- Python/runtime: very low.
- OS/kernel: very high (intended).
- VM/capture: medium-high.
- Execution-order contamination: medium.
- Metric inadequacy: low.
- True behavioral overlap: medium with `mem_alloc_touch_pages`.

**Should split into variants**: The fdatasync vs fsync vs O_SYNC variant has scientific value (separates data-flush from metadata-flush). The rolling-write-vs-growing-file variant has marginal value.

**Low-level controls**: fallocate, pwrite, fdatasync.

**C vs assembly**: C recommended (specifically for fdatasync). Python acceptable as fallback.

**Expected metric effect**: Marginal. fsync-vs-fdatasync may produce a slightly different cepstral shape.

**Final recommendation**: **Keep, low priority for rewrite**. Already empirically clean. Variants are scientifically interesting but not urgent.

---

#### `io_many_files` (existing)

**Scientific value**: High. Cleanest IO probe.

**Cleanliness critique**: Provisionally high (CV 0.048). **However, the filesystem backing is unknown**. This is a scientific validity issue.

**Likely confusions**:
- `io_many_files → mem_alloc_touch_pages`: behavioral proximity (both have many small allocations).

**Artifact risks**:
- Python/runtime: low.
- OS/kernel: very high (intended).
- VM/capture: high (many short syscalls).
- Execution-order contamination: medium.
- Metric inadequacy: low.
- True behavioral overlap: medium with `mem_alloc_touch_pages`.

**Should split into variants**: **Mandatory**. tmpfs and disk-backed are different mechanisms. The current results may apply to one or both; without knowing which, the interpretation is ambiguous.

**Low-level controls**: Explicit `--dir` argument; document filesystem type in metadata.

**C vs assembly**: C recommended (forces explicit FS choice, removes Python tempfile.mkdtemp ambiguity). Python acceptable with explicit directory.

**Expected metric effect**: tmpfs and disk-backed should produce different cepstral structures.

**Final recommendation**: **Implement first**. Resolving the tmpfs/disk ambiguity is a validity blocker for the existing results.

---

#### `io_read_cache_hit` (proposed)

**Scientific value**: Low-medium. It is another invisibility-demonstration test.

**Cleanliness critique**: By design produces minimal signal. "Cleanliness" here means consistency, not strong signal.

**Likely confusions**:
- `io_read_cache_hit → run_idle`: metric inadequacy (read-only invisibility).
- `io_read_cache_hit → cpu_hash_loop`: behavioral proximity (both near-invisible).

**Artifact risks**: Low across the board because the workload barely registers.

**Should split into variants**: No.

**Low-level controls**: pread, pre-warm cache, file size much smaller than RAM.

**C vs assembly**: Neither essential.

**Expected metric effect**: Near-zero delta signal.

**Final recommendation**: **Merge with a single canonical "near-invisible probe"**. Adding this as a separate test only repeats the invisibility finding.

---

#### `io_direct_write_like` (proposed)

**Scientific value**: High IF O_DIRECT propagates through virtio-blk to host. Unknown.

**Cleanliness critique**: Provisional medium-high. Depends on whether the underlying virtual disk honors O_DIRECT semantics.

**Likely confusions**:
- `io_direct_write_like → mem_alloc_touch_pages`: behavioral proximity (kernel-side allocations).
- `io_direct_write_like → idle_long_baseline`: metric inadequacy (block-layer footprint may be small).

**Artifact risks**:
- Python/runtime: low.
- OS/kernel: high (block layer dynamics).
- VM/capture: high.
- Metric inadequacy: medium-high.

**Should split into variants**: Sequential write and random write variants both have value.

**Low-level controls**: O_DIRECT, sector-aligned buffers, fallocate, pwrite.

**C vs assembly**: C strongly recommended (alignment requirement).

**Expected metric effect**: Substantially lower active_page_fraction than buffered IO. May or may not produce strong signal.

**Final recommendation**: **Pre-verify before committing**. Run a 60 s probe to confirm O_DIRECT actually bypasses the cache end-to-end (verify with `/proc/meminfo` `Dirty == 0`). If yes, implement. If no, drop.

---

### 3.4 CPU Family

**Purpose**: Stress computation without intentional memory traffic.

**Why it matters**: Tests whether CPU-bound workloads produce a distinctive signature, OR confirms that they are invisible to the delta pipeline. Either result is informative.

**What it isolates**: Register-resident arithmetic, branch logic, instruction stream.

**What it accidentally mixes in**: Stack frame writes, Python interpreter eval-stack churn (in Python implementations).

**Cleanliness expectation**: Provisionally low if "cleanliness" means strong distinct signal. Provisionally consistent if "cleanliness" means stable invisibility.

**Thesis question**: Does the delta pipeline distinguish CPU-bound from idle?

**First to implement**: `cpu_matrix_mult` (the only CPU test with intentional memory traffic).

**Delay or merge**: `cpu_hash_loop` and `cpu_branch_random` both demonstrate invisibility. Consolidate into one canonical probe.

---

#### `cpu_hash_loop` (proposed)

**Scientific value**: Low-medium. Demonstrates invisibility of register-resident compute.

**Cleanliness critique**: Will collapse to idle in delta metrics.

**Likely confusions**:
- `cpu_hash_loop → run_idle`: metric inadequacy.
- `cpu_hash_loop → idle_long_baseline`: metric inadequacy.

**Artifact risks**:
- Python/runtime: very high in Python (eval stack dominates).
- OS/kernel: very low.
- VM/capture: low.
- Metric inadequacy: very high.

**Should split into variants**: No.

**Low-level controls**: Tight C inner loop, no allocations.

**C vs assembly**: C strongly recommended. Assembly only if zero memory traffic verification is essential.

**Expected metric effect**: Near-idle on all metrics.

**Final recommendation**: **Consolidate with cpu_branch_random** into one canonical "register-resident compute probe". Implement that single probe.

---

#### `cpu_matrix_mult` (proposed)

**Scientific value**: Medium-high. Only CPU test with intentional memory traffic.

**Cleanliness critique**: Heavily depends on matrix size variant. Each size answers a different question.

**Likely confusions**:
- `cpu_matrix_mult (large) → mem_stream`: behavioral proximity.
- `cpu_matrix_mult (small) → cache_hot_loop`: behavioral proximity.

**Artifact risks**:
- Python/runtime: very high if NumPy/BLAS used. Lower in plain C.
- OS/kernel: low.
- VM/capture: low.
- Metric inadequacy: medium.

**Should split into variants**: Yes. L1, L2, L3, exceeds-LLC sizes.

**Low-level controls**: Plain C, no BLAS, aligned allocations.

**C vs assembly**: C strongly recommended.

**Expected metric effect**: Variable by size. Small: cache-hot pattern. Large: streaming pattern.

**Final recommendation**: **Keep as the canonical CPU-with-memory probe**. Skip the other two CPU tests.

---

#### `cpu_branch_random` (proposed)

**Scientific value**: Low. Duplicates `cpu_hash_loop` in observable signal.

**Cleanliness critique**: Same as `cpu_hash_loop`.

**Likely confusions**: Same as `cpu_hash_loop`.

**Artifact risks**: Same as `cpu_hash_loop`.

**Should split into variants**: No.

**Low-level controls**: Same.

**C vs assembly**: Same.

**Expected metric effect**: Same as `cpu_hash_loop`.

**Final recommendation**: **Remove or merge with `cpu_hash_loop`**. Both are register-resident; both will look idle. One probe suffices.

---

### 3.5 CACHE Family

**Purpose**: Probe the relationship between memory access locality and observable signal.

**Why it matters**: If the metric pipeline distinguishes locality patterns, this is direct evidence of structural sensitivity beyond mere "memory pressure" detection.

**What it isolates**: Working set size relative to L1, L2, LLC, and DRAM. Stride-vs-cache-line relationships.

**What it accidentally mixes in**: Hardware prefetcher behavior, THP, cache-vs-DRAM observability question.

**Cleanliness expectation**: Per-stride and per-size variants each have provisional medium designation. None are clean canonical probes; they are sweeps.

**Thesis question**: Does the metric pipeline detect cache-vs-DRAM access regimes?

**First to implement**: `cache_stride_sweep` (high scientific value for the stride-vs-cache-line question).

**Delay or merge**: `cache_hot_loop` and `cache_cold_scan` overlap with other tests.

---

#### `cache_hot_loop` (proposed)

**Scientific value**: High as a cache-vs-DRAM observability probe. Low as a classification test.

**Cleanliness critique**: The filled spec correctly identifies that cache-resident dirty data may not be visible to pmemsave. This is the central scientific question for this test.

**Likely confusions**:
- `cache_hot_loop → run_idle`: metric inadequacy (cache hides writes).
- `cache_hot_loop → cpu_hash_loop`: behavioral proximity (both register-or-cache-resident).

**Artifact risks**:
- Python/runtime: medium.
- OS/kernel: low.
- VM/capture: medium-high (pause may flush cache; observability question).
- Metric inadequacy: very high.

**Should split into variants**: With CLFLUSH versus without. The comparison answers the cache-vs-DRAM question.

**Low-level controls**: Buffer 16 to 64 KB (L1) and 256 to 512 KB (L2). Optional CLFLUSH variant.

**C vs assembly**: C strongly recommended. Assembly justified for CLFLUSH (compiler intrinsic available).

**Expected metric effect**: Without CLFLUSH: near-idle. With CLFLUSH: visible writes.

**Final recommendation**: **Keep as a methodological experiment, not a classification test**. The CLFLUSH-vs-not comparison directly probes pipeline observability. One careful experiment, not an ongoing class member.

---

#### `cache_cold_scan` (proposed)

**Scientific value**: Low-medium. Heavily overlaps with `mem_stream` write at large working set.

**Cleanliness critique**: Predicted provisionally medium-high. Same mechanism as `mem_stream`.

**Likely confusions**:
- `cache_cold_scan (write) → mem_stream`: behavioral proximity (very high).
- `cache_cold_scan (read) → run_idle`: metric inadequacy (read invisibility).

**Artifact risks**: Same as `mem_stream`.

**Should split into variants**: Read versus write. Read variant overlaps with legacy `mem_pointer_chase`.

**Low-level controls**: Same as `mem_stream`.

**C vs assembly**: Same.

**Expected metric effect**: Same as `mem_stream`.

**Final recommendation**: **Merge with mem_stream variants**. Add a read-only variant of `mem_stream` to cover the "cold scan, read-only" case if needed.

---

#### `cache_stride_sweep` (proposed)

**Scientific value**: High. Stride-vs-cache-line sweep is a strong scientific probe.

**Cleanliness critique**: Per-stride results vary by design. The sweep IS the experiment.

**Likely confusions**: By design overlaps with `mem_stream` at page stride.

**Artifact risks**:
- Python/runtime: high (loop rate matters). Low in C.
- OS/kernel: medium (THP, prefetcher).
- VM/capture: medium.
- Metric inadequacy: medium.

**Should split into variants**: Already structured as a sweep over (buffer size, stride).

**Low-level controls**: Buffer 4, 16, 64 MB. Stride 8, 64, 256, 4096, 65536 bytes.

**C vs assembly**: C strongly recommended.

**Expected metric effect**: Per-stride cepstral peak shift. Per-buffer cache-regime transition.

**Final recommendation**: **Implement, absorb mem_stride_sweep_large**. Parameterize buffer size to cover both cache-bounded and DRAM-bounded regimes in one sweep.

---

### 3.6 THREAD Family

**Purpose**: Probe concurrency, synchronization, and inter-CPU traffic.

**Why it matters**: Real workloads are concurrent. Whether the metric pipeline handles non-stationary multi-threaded signal is a generalization question.

**What it isolates**: Kernel futex, sk_buff, scheduler wake-ups, cache-line ping-pong.

**What it accidentally mixes in**: Scheduler nondeterminism, glibc per-thread arena, host CPU contention.

**Cleanliness expectation**: Provisionally low to medium. High CV expected.

**Thesis question**: Does the metric pipeline produce stable signatures for concurrent workloads?

**Critical caveat**: All three THREAD tests assume kernel-mediated synchronization activity is observable in pmemsave dumps. **This is unverified**. A pre-experiment is needed before committing to three thread tests.

**First to implement**: A pre-verification probe on `thread_parallel_alloc`. If it produces signal beyond the single-threaded version, the family is viable.

**Delay**: `thread_lock_contention` and `thread_producer_consumer` until viability is confirmed.

---

#### `thread_lock_contention` (proposed)

**Scientific value**: Speculative.

**Cleanliness critique**: Predicted provisionally low. High CV expected. May not produce signal beyond a single-threaded equivalent.

**Likely confusions**:
- `thread_lock_contention → io_seq_fsync`: behavioral proximity (rhythmic kernel activity).
- `thread_lock_contention → run_idle`: metric inadequacy if futex traffic is invisible.

**Artifact risks**:
- Python/runtime: disqualifying (GIL).
- OS/kernel: high (intended) IF observable.
- VM/capture: high.
- Metric inadequacy: high.
- True behavioral overlap: medium.

**Should split into variants**: No.

**Low-level controls**: pthread_mutex or futex syscalls, N threads pinned.

**C vs assembly**: C strongly recommended. Python disqualified.

**Expected metric effect**: Bursty Fano factor. Possibly indistinguishable from idle if kernel structures are not visible.

**Final recommendation**: **Delay**. Pre-verify viability. If futex traffic is invisible, the test produces no signal beyond stack-frame writes.

---

#### `thread_producer_consumer` (proposed)

**Scientific value**: Speculative. Slightly more visible than lock contention because the ring buffer is in user space.

**Cleanliness critique**: Predicted provisionally medium.

**Likely confusions**:
- `thread_producer_consumer → mem_alloc_touch_pages`: behavioral proximity.
- `thread_producer_consumer → net_tcp_loopback_stream`: behavioral proximity.

**Artifact risks**:
- Python/runtime: disqualifying.
- OS/kernel: high.
- VM/capture: medium-high.
- Metric inadequacy: medium.
- True behavioral overlap: medium.

**Should split into variants**: Buffer size variants.

**Low-level controls**: pthread + condvar, ring buffer.

**C vs assembly**: C strongly recommended.

**Expected metric effect**: Periodic at ring-fill cadence.

**Final recommendation**: **Delay**. Implement only if first thread test confirms family viability.

---

#### `thread_parallel_alloc` (proposed)

**Scientific value**: Medium. Most likely thread test to produce observable signal because each thread runs `mem_alloc_touch_pages` which is known to produce strong signal.

**Cleanliness critique**: Predicted provisionally medium. Burstier than single-threaded.

**Likely confusions**:
- `thread_parallel_alloc → mem_alloc_touch_pages`: behavioral proximity (very high; same fundamental mechanism).

**Artifact risks**:
- Python/runtime: disqualifying.
- OS/kernel: very high (kernel mmap_lock contention).
- VM/capture: high.
- Metric inadequacy: medium.
- True behavioral overlap: very high with single-threaded version.

**Should split into variants**: Thread-count sweep.

**Low-level controls**: pthread, direct mmap per thread, no shared mutex.

**C vs assembly**: C strongly recommended.

**Expected metric effect**: Higher event rate, higher Fano factor than single-threaded. Possibly weaker periodicity.

**Final recommendation**: **Implement first as the thread-family viability probe**. If it produces signal distinguishable from `mem_alloc_touch_pages`, the family is viable.

---

### 3.7 NETWORK Family

**Purpose**: Probe network I/O behavior on loopback.

**Why it matters**: Tests whether kernel-mediated rhythm generalizes beyond filesystem.

**What it isolates**: Kernel network stack, sk_buff allocations, TCP/UDP socket buffers.

**What it accidentally mixes in**: Loopback driver behavior, host CPU contention.

**Cleanliness expectation**: Provisional medium per filled spec, with no empirical anchoring.

**Thesis question**: Does kernel-mediated I/O rhythm generalize from filesystem to network?

**Critical caveat**: NETWORK family viability is unverified. Loopback network may produce minimal observable memory delta if sk_buff lives in slab caches that are reused without content change.

**First to implement**: A pre-verification probe on `net_tcp_loopback_stream` to confirm signal viability.

**Delay**: All three until viability is confirmed.

---

#### `net_tcp_loopback_stream` (proposed)

**Scientific value**: Speculative-high. If observable, generalizes IO rhythm finding to non-filesystem.

**Cleanliness critique**: Predicted provisionally medium-high. No empirical basis.

**Likely confusions**:
- `net_tcp_loopback_stream → io_seq_fsync`: behavioral proximity (both rhythmic streaming).
- `net_tcp_loopback_stream → thread_producer_consumer`: behavioral proximity.

**Artifact risks**:
- Python/runtime: low.
- OS/kernel: very high.
- VM/capture: medium.
- Metric inadequacy: high if sk_buff churn is invisible.

**Should split into variants**: TCP_NODELAY enabled or disabled.

**Low-level controls**: Two C processes, fixed socket buffer sizes, documented TCP congestion algorithm.

**C vs assembly**: C recommended.

**Expected metric effect**: If observable, periodic at TCP-throughput cadence. If not, near-idle.

**Final recommendation**: **Pre-verify**. Run a 60 s probe with traffic; check `/proc/slabinfo` `skbuff_head_cache` and other kernel slab churn. If pmemsave deltas show meaningful per-snapshot change attributable to network activity, implement. Otherwise drop.

---

#### `net_udp_burst` (proposed)

**Scientific value**: Lower than TCP variant. UDP has less internal structure.

**Cleanliness critique**: Predicted provisionally medium.

**Likely confusions**:
- `net_udp_burst → io_rand_rw`: behavioral proximity.
- `net_udp_burst → run_idle`: metric inadequacy if invisible.

**Artifact risks**: Same family as TCP.

**Should split into variants**: No.

**Low-level controls**: sendmsg/recvmsg, fixed packet rate.

**C vs assembly**: C recommended.

**Expected metric effect**: Bursty Fano factor.

**Final recommendation**: **Delay**. Implement only after TCP variant confirms family viability.

---

#### `net_many_small_messages` (proposed)

**Scientific value**: Speculative. Network analog of `io_many_files`.

**Cleanliness critique**: Predicted provisionally medium.

**Likely confusions**:
- `net_many_small_messages → io_many_files`: behavioral proximity.
- `net_many_small_messages → net_tcp_loopback_stream`: behavioral proximity.

**Artifact risks**: Same family.

**Should split into variants**: No.

**Low-level controls**: TCP_NODELAY on, fixed message size and rate.

**C vs assembly**: C recommended.

**Expected metric effect**: High syscall rate; may or may not produce visible signal.

**Final recommendation**: **Delay**. Network family viability is the gating question.

---

### 3.8 MIXED Family

**Purpose**: Test whether feature space is continuous when mechanisms combine.

**Why it matters**: Real workloads do not isolate one mechanism. Whether the metric pipeline maps mixtures to intermediate positions or collapses them to one parent is a generalization question.

**What it isolates**: Currently nothing distinct. Each MIXED test inherits its parent components.

**What it accidentally mixes in**: One component dominates and the mixture resembles it.

**Cleanliness expectation**: Provisionally medium. By design these are not clean probes.

**Thesis question**: Does the feature space behave continuously?

**Critical caveat**: As currently specified, all three MIXED tests are predicted to resemble their dominant parent. This makes them a tautology: a strong-MEM-and-weak-CPU test resembles MEM. The scientific value is in studying the **transition**, not the endpoint.

**First to implement**: None as currently specified. Redesign as rate-ratio sweeps.

**Delay**: All three until redesign.

---

#### `mixed_mem_io` (proposed)

**Scientific value**: Medium IF rate-ratio is swept. Low as a single-point test.

**Cleanliness critique**: Predicted provisionally medium. By design intermediate.

**Likely confusions**:
- `mixed_mem_io → mem_random_write_pages`: behavioral proximity.
- `mixed_mem_io → io_rand_rw`: behavioral proximity.

**Artifact risks**:
- Python/runtime: disqualifying.
- OS/kernel: high.
- VM/capture: medium-high.
- Execution-order contamination: medium.
- Metric inadequacy: high.
- True behavioral overlap: very high with parents.

**Should split into variants**: As a rate-ratio sweep with at least 5 ratios (e.g. 100/0, 75/25, 50/50, 25/75, 0/100). Each ratio probes a different point on the continuity curve.

**Low-level controls**: Two threads with parameterized rate-ratio.

**C vs assembly**: C strongly recommended.

**Expected metric effect**: Trajectory through feature space from MEM cluster to IO cluster.

**Final recommendation**: **Redesign as a rate-ratio sweep**. Implement after parent tests are validated.

---

#### `mixed_cpu_mem` (proposed)

**Scientific value**: Low as currently specified. The CPU component is invisible; the mixture resembles MEM alone.

**Cleanliness critique**: Predicted provisionally medium-high (resembles `mem_stream`). Tautology.

**Likely confusions**:
- `mixed_cpu_mem → mem_stream`: behavioral proximity (very high).

**Artifact risks**: Same as MEM family.

**Should split into variants**: Only if the CPU component is upgraded to one with intentional memory traffic (e.g. small-matrix `cpu_matrix_mult`).

**Low-level controls**: Two threads.

**C vs assembly**: C strongly recommended.

**Expected metric effect**: Same as `mem_stream`.

**Final recommendation**: **Remove or redesign**. Pairing an invisible CPU test with a visible MEM test produces no scientific information.

---

#### `mixed_cpu_io` (proposed)

**Scientific value**: Same problem as `mixed_cpu_mem`.

**Cleanliness critique**: Predicted provisionally medium-high (resembles `io_seq_fsync`). Tautology.

**Likely confusions**:
- `mixed_cpu_io → io_seq_fsync`: behavioral proximity (very high).

**Artifact risks**: Same as IO family.

**Should split into variants**: As above, only if CPU component has visible signal.

**Low-level controls**: Two threads.

**C vs assembly**: C strongly recommended.

**Expected metric effect**: Same as `io_seq_fsync`.

**Final recommendation**: **Remove or redesign**.

---

## 4. Implementation Priority Plan

### 4.1 First 5 to implement

1. **`idle_long_baseline`** (IDLE). Foundational reference. Without it, every comparison is contaminated.
2. **`mem_pointer_chase` redesign** (MEM). Largest source of MEM scatter. Read-only invisibility is the leading hypothesis for current confusion patterns.
3. **`mem_stream` C rewrite at 1 GB working set** (MEM). Largest source of `mem_stream → run_idle` and `mem_stream → io_rand_rw` confusions.
4. **`io_many_files` with explicit FS choice** (IO). Tmpfs-vs-disk ambiguity in current data is a validity blocker.
5. **`mem_alloc_touch_pages` C rewrite** (MEM). Already cleanest; rewrite tightens the cleanest probe and removes glibc nondeterminism.

### 4.2 Next 5 to implement

6. **`io_rand_rw` with fallocate + cold-cache + O_DIRECT variants**. Tests the BO-vs-MI hypothesis for the only IO confusion.
7. **`cache_stride_sweep`**. High scientific value for the access-structure question. Absorbs `mem_stride_sweep_large` and partially absorbs `cache_cold_scan`.
8. **`cpu_matrix_mult`**. Only CPU test with intentional memory traffic. Single CPU representative.
9. **`idle_post_workload_recovery`**. Diagnostic for IDLE subtype split per prior workload.
10. **`thread_parallel_alloc`**. Family viability probe for THREAD.

### 4.3 Tests to delay

- All three NETWORK tests until pre-verification confirms family viability.
- `thread_lock_contention` and `thread_producer_consumer` until `thread_parallel_alloc` confirms family viability.
- `io_direct_write_like` until pre-verification confirms O_DIRECT propagates through virtio-blk.
- `io_seq_fsync` C rewrite (already clean; marginal value).

### 4.4 Tests to remove or merge

- **Remove**: `cpu_hash_loop`, `cpu_branch_random`. Redundant invisibility demonstrations. One canonical "near-invisible compute probe" in the form of a calibration measurement is sufficient if needed at all.
- **Remove**: `io_read_cache_hit`. Another invisibility demonstration.
- **Merge**: `mem_random_write_pages` into `mem_pointer_chase` redesign as a PRNG-variant.
- **Merge**: `mem_stride_sweep_large` into `cache_stride_sweep` with a parameterized buffer size axis.
- **Merge**: `cache_cold_scan` into `mem_stream` variants (read variant of `mem_stream`).
- **Redesign or remove**: All three MIXED tests as rate-ratio sweeps, or remove if rate-ratio implementation is too costly.

### 4.5 Reduced test inventory

After this priority plan, the realistic implementation set is:

| Family | Tests |
|---|---|
| IDLE | `run_idle` (existing), `idle_long_baseline`, `idle_post_workload_recovery` |
| MEM | `mem_stream` (rewrite), `mem_pointer_chase` (redesign with LCG and PRNG variants), `mem_alloc_touch_pages` (rewrite) |
| IO | `io_rand_rw` (rewrite + variants), `io_seq_fsync` (existing or rewrite later), `io_many_files` (rewrite + tmpfs/disk variants), optional `io_direct_write_like` if pre-verified |
| CPU | `cpu_matrix_mult` (one canonical) |
| CACHE | `cache_stride_sweep` (parameterized) |
| THREAD | `thread_parallel_alloc` first (viability probe) |
| NETWORK | Pre-verification probe only initially |
| MIXED | Rate-ratio sweep of `mixed_mem_io` only if scope allows |

Total: 10 to 13 tests, each implementable and each addressing a distinct scientific question.

---

## 5. Candidate Improvement Directions

These are brief notes. Full enhanced-test proposals belong in a later document.

1. **Capture-pipeline calibration probe**. A workload that runs nothing for 300 s while pmemsave snapshots are taken at varying intervals. Directly characterizes the pause-induced noise floor. Mandatory if pause variance is to be subtracted from observed signals.

2. **Cache-vs-DRAM observability probe**. A controlled experiment comparing CLFLUSH-after-write versus no-CLFLUSH on the same workload. Resolves whether cache-resident dirty data is observable.

3. **THP-control probe**. Same workload (e.g. `mem_stream`) with MADV_HUGEPAGE versus MADV_NOHUGEPAGE versus default. Quantifies whether THP promotion affects per-page deltas.

4. **Glibc allocator probe**. Same workload (`mem_alloc_touch_pages`) with `MALLOC_MMAP_THRESHOLD_` set high (force arena) versus low (force mmap-direct) versus default. Quantifies allocator-path nondeterminism.

5. **Rate-ratio MIXED sweep**. Replace the three point-style MIXED tests with one rate-ratio-parameterized test. Five to seven points along a parent-component balance curve. The trajectory through feature space IS the result.

6. **Per-cycle drift probe**. Run the same workload (e.g. `mem_alloc_touch_pages`) eight times in a row with no other workloads between. Compares cycle-1 to cycle-8 signature directly. Tests the cycle drift hypothesis seen in `mem_stream` cycle-3/4 collapse.

7. **Reverse-cycle probe**. Run the steps_cycle in reverse order (IO first, then MEM, then idle). Tests the execution-order contamination hypothesis seen in `mem_stream → io_rand_rw`.

These directions test methodological assumptions that the filled spec adopts without empirical anchoring. They should precede implementation of speculative test families.

---

## 6. Thesis-Safe Wording

**On the next-generation taxonomy as a whole**:

> The next-generation workload taxonomy proposed in `next_gen_workload_test_mini_specs_filled.md` defines 25 candidate workloads across eight behavioral families. The full taxonomy is aspirational. The scope of an actual implementation effort should focus on a smaller priority subset addressing the highest-value scientific questions identified in the Phase 1 audits: a controlled IDLE baseline, a redesigned `mem_pointer_chase` with observable writes, a larger-footprint `mem_stream`, and an explicit-filesystem `io_many_files`. Other tests in the taxonomy are useful conceptual placeholders but should be implemented only after their viability has been pre-verified or the priority subset has been validated.

**On the "near-invisible" tests**:

> Several proposed workloads (CPU-bound register-resident probes, read-only cache probes) are predicted to be largely invisible to the page-content-delta observation pipeline. Their primary scientific value is to demonstrate the read versus write asymmetry of the observation method, not to add classifiable subtypes. A single canonical invisibility-demonstration probe is sufficient; multiple variants of the same negative finding add cost without scientific gain.

**On the THREAD and NETWORK families**:

> The THREAD and NETWORK families are speculative until their viability is empirically confirmed. The page-content-delta pipeline observes guest physical RAM. Whether kernel-mediated synchronization activity (futex, sk_buff allocations, scheduler wake-ups) leaves observable signatures is a hypothesis, not a finding. Each family should begin with one viability probe before broader implementation.

**On the MIXED family**:

> The MIXED family as currently specified is structurally limited because each test combines mechanisms whose relative throughput is fixed. The dominant component will dominate the feature vector by construction. To convert MIXED tests into useful feature-space-continuity probes, the design must include rate-ratio parameter sweeps. Without that, the MIXED tests confirm that dominant components dominate, which is a tautology rather than a scientific finding.

**On the implementation level designations**:

> Recommendations to rewrite workloads in C are motivated by experimental control over runtime artifacts, not performance. C does not give the experimenter control over kernel behavior; it gives finer control over how user-space invokes kernel behavior. Assembly is justified in only two specific variants where C cannot portably express the required control. Most tests do not benefit from assembly. Most IO tests do not benefit substantially from C either; the kernel mediation is the dominant signal regardless of language.

**On the cleanliness expectations for unimplemented tests**:

> Cleanliness expectations stated for proposed tests are hypotheses, not findings. Provisional cleanliness designations should be assigned only after at least 8 to 10 runs have been collected per `confusion_matrix_diagnostic_methodology.md` Section 6. The cleanliness predictions in the filled spec serve as testable predictions, not conclusions.

---

## 7. Summary

The filled mini-spec is a thorough first draft of an aspirational test taxonomy. Its main weakness is volume: 25 tests is more than can be reasonably implemented and characterized within a thesis timeline. Several tests duplicate one finding (invisibility of register-resident or read-only workloads). Several families (THREAD, NETWORK, MIXED) commit to multiple tests before their viability is verified.

The recommended path forward is:

1. Reduce the test inventory to 10 to 13 tests by merging redundant probes and deferring speculative families.
2. Implement the foundational tests first: `idle_long_baseline`, `mem_pointer_chase` redesign, `mem_stream` C rewrite at 1 GB, `io_many_files` with explicit FS, `mem_alloc_touch_pages` C rewrite.
3. Use the proposed methodology probes (capture-pipeline calibration, cache-vs-DRAM observability, THP control, allocator control) to validate assumptions before committing to speculative families.
4. Convert MIXED tests into rate-ratio sweeps or remove them.
5. Treat all cleanliness expectations as testable hypotheses, not predicted outcomes.

This review does not invalidate the filled spec. It scopes it to what can be implemented and validated rigorously within a realistic thesis budget.
