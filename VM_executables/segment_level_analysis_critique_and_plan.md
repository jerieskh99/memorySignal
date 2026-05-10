# Segment-Level Analysis: Critique and Plan

This document evaluates the proposed segment-level analysis direction, lists its valid uses, defines the conditions under which segments may be treated as quasi-independent observations, and specifies what should and should not be done. It assumes the context already established in `segment_level_analysis_context_and_connections.md`.

---

## 1. Verdict

Segment-level analysis is **scientifically valid as a within-run diagnostic tool**. It is **not valid as a sample-size inflation tool** for class-level claims. The distinction is load-bearing and must be enforced throughout downstream use.

Defensible immediately:
- temporal stability inside a single run
- localization of confusion within a misclassified run
- phase identification (warmup / steady / cooldown)
- subtype fingerprint validation against expected temporal patterns
- drift detection across cycles
- segment-level visualizations that expose what full-run averaging hides

Not defensible without further evidence:
- treating segment counts as effective sample size for Kruskal-Wallis, Mann-Whitney, LDA, or silhouette numbers
- pooling segments and computing classification accuracy without group-aware cross-validation
- claiming segment-level effect sizes as evidence of class separability
- converting `k` segments × 48 runs into "192 samples" for any inferential test

The methodology rules in `confusion_matrix_diagnostic_methodology.md` (Sections 5, 6, 7) apply unchanged and are extended, not replaced, by segmentation.

---

## 2. Strongest Uses

### 2.1 Within-run temporal stability

Compute per-segment feature distances or per-segment metric values within one run. Plot per-segment trajectories. Quantify with within-run CV across segments and lag-1 autocorrelation of segment features. This is the segmentation use with the lowest risk of overclaiming because the comparison is intra-run, not cross-class.

### 2.2 Phase purity check

Compare segment-1 (early) versus segment-k (late) within each run. If they differ systematically across the population of runs of one subtype, the run contains an identifiable phase structure. This validates or contradicts the temporal claims in `current_workload_test_mini_specs.md` (e.g. `mem_alloc_touch_pages` expecting allocate-touch-release rhythm).

### 2.3 Confusion localization

For each off-diagonal confusion in the confusion matrix, classify each segment of the misclassified run independently. Check whether all segments are misclassified, only early ones, only late ones, or a single outlier segment dominates. This is the most direct contribution to the four-cause model in `confusion_matrix_diagnostic_methodology.md` Section 5.

### 2.4 Subtype fingerprint validation

For each subtype, look at the segment-trajectory shape. A clean probe should have segment trajectories that overlap across runs. An impure probe will have run-specific trajectory shapes. This is a richer reproducibility check than per-run CV alone.

### 2.5 Drift detection across cycles

Compare segment trajectories of the same subtype across cycles 1, 2, 3, 4. Cycle-to-cycle drift visible at segment level but not at run level indicates either thermal drift, residual state buildup, or capture pipeline drift. The currently observed `mem_stream` cycle-1 strength versus cycle-3/4 weakness is testable this way.

### 2.6 Confidence estimation

Per-run confidence can be estimated by within-run segment dispersion in the feature space. A run whose segments all sit at the same point in feature space is a high-confidence representative of its subtype. A run whose segments scatter widely is low-confidence. This produces a per-run reliability score without expanding the labeled dataset.

### 2.7 Capture artifact characterization

If pmemsave pause spikes or other capture-side noise concentrate in specific segments, segment-level inspection can detect this. The current per-run aggregation hides it because the spike contributes to one of many windows averaged into the per-run feature.

---

## 3. Independence Analysis

### 3.1 Segments are NOT independent by default

The continuous VM execution makes segments share state across boundaries. Specifically:

- page cache, dentry cache, allocator state evolve continuously
- workload-internal phases (warmup, steady-state, cooldown) are not exchangeable
- per-page MSC and PLV depend on a temporal extent that may overlap or border segment boundaries
- capture-side noise (pmemsave pause variance, host-system jitter) has temporal autocorrelation
- garbage collection events affect multiple consecutive windows

### 3.2 Segments can approach quasi-independence under explicit conditions

Conditions, all required:

1. Workload is in steady-state. Warmup segments are excluded or modeled separately.
2. Segment length is much greater than the autocorrelation length of segment-level features. Concretely, lag-1 autocorrelation of per-segment features must be measured and reported. A reasonable provisional threshold is `r < 0.3`.
3. Segments do not span phase boundaries internal to the workload. For workloads with cyclic structure, segment length should be an integer multiple of the cycle period or much larger than it.
4. Each segment contains enough windows to make the per-segment metric estimates statistically meaningful (lower bound discussed in Section 5).

Even when these conditions hold, segments are **quasi-independent**, not independent. Inferential procedures must respect the run-level grouping.

### 3.3 Cyclic workloads as a special case

The mini-specs explicitly describe cyclic mechanisms in some workloads:

- `mem_alloc_touch_pages`: allocate-touch-release rhythm
- `io_seq_fsync`: write-fsync rhythm
- `io_many_files`: per-batch file create-write-delete

For these subtypes, IF the segment length aligns with or much exceeds the cycle period, each segment can approximate a quasi-replicated realization of the same generative process within one run. This is the strongest case for treating segments as semi-independent.

For workloads without intentional cyclic structure (`mem_stream` smooth traversal, `mem_pointer_chase` pseudo-random traversal), segments are different temporal slices of one continuous process. They are NOT replicates. Treating them as such is invalid.

### 3.4 IDLE is a separate problem

`run_idle` is not a workload but a passive observation window. The expected behavior is non-stationary by construction (residual effects, OS daemons, scheduler noise). Segmenting IDLE asks a different question: how does residual signal decay along the idle window. Segments here are useful for decay-shape analysis, not for stationarity claims.

---

## 4. Pre-Segmentation Checks

Before any cross-class or cross-subtype analysis on segment-level features, the following must be reported:

1. **Within-run autocorrelation.** Lag-1 autocorrelation of the chosen segment-level summary metric, per run, summarized per subtype.
2. **Segment-level stationarity test.** Augmented Dickey-Fuller or KPSS on the per-segment trajectory, per run. Report fraction of runs that fail stationarity.
3. **Warmup detection.** Identify the smallest leading segment count whose exclusion eliminates a systematic mean shift. Report this per subtype.
4. **Cycle alignment estimate.** For workloads with a known cycle period (allocate-touch-release for `mem_alloc_touch_pages`, fsync for `io_seq_fsync`, batch for `io_many_files`), report whether the chosen `k` produces segments that contain an integer number of cycles. If not, document the alignment error.
5. **Capture spike density.** Count of large outlier windows per segment (e.g. windows with delta magnitude above the 99th run-percentile). If certain segments concentrate spikes, report this before drawing within-run conclusions.

These checks are diagnostic. They condition any later interpretation. They do not need to all pass for the analysis to proceed, but their results must be documented alongside any segment-level claim.

---

## 5. Choosing k and Hop

### 5.1 k is not a hyperparameter; it is a research-question selector

Different `k` values ask different scientific questions:

| k | Scientific question |
|---|---|
| 1 | Is the run distinctive overall (current analysis) |
| 2 | Is the run internally consistent (split-half reliability) |
| 3-4 | How does the run evolve from start to end (warmup / steady / late) |
| 6-8 | What micro-phases does the workload contain |
| `n_windows` | Window-level dynamics (loses cross-window MSC and cepstral structure) |

Increasing `k` does not "improve" the analysis. It changes what is measured. Reports must state explicitly which question is being addressed.

### 5.2 Lower bound on segment length

Each segment must contain enough windows for the per-segment metric pipeline to produce stable estimates. For the existing pipeline:

- window size = 128
- hop = 64

So each segment of length `L_frames` produces approximately `(L_frames - 128) / 64 + 1` windows.

Practical floor: at least about 50 to 100 windows per segment for MSC and cepstral estimates with usable variance. This implies `L_frames` of roughly 3,300 to 6,500 frames per segment minimum, depending on how strict the variance bound must be.

Concrete rule for choosing k:

```
k_max = floor(N_frames_per_run / L_min)
```

where `L_min` is set by the metric stability floor, not by aesthetic preference. The data already in `meta.json` per run determines `N_frames_per_run`. The rule must be applied per run; if runs differ in length, `k` should be chosen so the slowest run still satisfies the floor.

### 5.3 Hop tuning relative to k

The 50% overlap (hop=64 vs window=128) creates correlation between adjacent windows. Two policies are defensible:

- **Conservative**: keep hop=64 to preserve continuity with the current pipeline so segment-level features are directly comparable to run-level features. Accept that segment-internal windows are correlated.
- **Reduced overlap**: increase hop toward window size to reduce within-segment window correlation, at the cost of fewer windows per segment. Useful only if `k` is small enough that each segment retains enough non-overlapping windows for the metric.

Recommended initial choice: keep hop=64 for the first segment-level study to maintain comparability with all existing per-run results. Treat hop changes as a separate sensitivity analysis.

### 5.4 Recommended initial k grid

Use a small grid, not a single value. Treat results as a sensitivity analysis:

| k | Purpose |
|---|---|
| 2 | split-half reliability |
| 4 | quartile structure (matches the 4-cycle dataset symmetry conceptually but does NOT confound cycles with segments since segmentation is within-run) |
| 8 | finer phase structure where workload duration permits |

Validate `L_min` per run before committing to any `k`. Report `k`, `L_min`, mean windows per segment, and the autocorrelation summary alongside any segment-level finding.

---

## 6. What Not To Do

### 6.1 Do not pool segments and report inflated statistics

Concretely forbidden:

- Reporting "n = 192" instead of n = 48
- Running Kruskal-Wallis or Mann-Whitney on pooled segment-level features as if they were independent samples
- Computing Cohen's d on pooled segment distributions as a class-level effect size
- Computing silhouette scores at the segment level and citing them as evidence of class separability

These produce inflated effect sizes because intra-run segments share noise. They violate the four-cause model rule that demands isolation before claims.

### 6.2 Do not run leave-one-segment-out cross-validation

Leakage is severe: same-run segments share VM state, same-run capture noise, and same-run hardware state. The minimum acceptable cross-validation unit is the run, not the segment. Use group-aware folds (Group K-fold with run as the group) if any cross-validation is performed.

### 6.3 Do not compute classification accuracy on segments and report it as generalization

Even with group-aware folds, segment-level "accuracy" measures classifier behavior on temporal slices. It does not measure the per-run classification claim that the thesis cares about. Report run-level accuracy as the headline number; segment-level results stay diagnostic.

### 6.4 Do not silently change `k` between analyses

If different sections of the eventual report use different `k`, every section must state its `k` explicitly. `k` is part of the methodology, not an implementation detail.

### 6.5 Do not assume uniform segmentation works for all subtypes

Cyclic workloads benefit from cycle-aligned segmentation. Smooth workloads do not. IDLE is non-stationary by design. Different subtypes may justify different segmentation policies. Forced uniformity is a methodological choice, not a default.

### 6.6 Do not interpret a clean per-segment trajectory as a generalization claim

A run whose 4 segments are tightly clustered tells you the run is internally consistent. It does not tell you the workload generalizes to a new run, a new machine, or a new time. The stochastic evaluation already documents this distinction (`stochastic_results_scientific_evaluation.md` Section 11).

---

## 7. Uses Beyond `mem_stream`

The user prompt warned against framing this as a `mem_stream` rescue effort. The following uses are not about `mem_stream`.

### 7.1 Test the mini-spec temporal claims directly

`mem_alloc_touch_pages` mini-spec states "burst-like allocation phases" and "repeated allocate-touch-release structure". Segment-level cepstral periodicity scores within a single run can confirm or contradict this. The current per-run summary cannot separate "stable rhythm" from "stable averaged value with no rhythm".

### 7.2 Diagnose the `mem_pointer_chase` regime collapse

Three of four `mem_pointer_chase` runs show `snr_mean` near 0.05, while test4 shows 2.583. Segmentation can ask: did test16 / test28 / test40 START active and decay, START inactive and stay inactive, or oscillate. Each pattern points to a different cause among the four in `confusion_matrix_diagnostic_methodology.md` Section 5.

### 7.3 IDLE residual decay shape

Test1 and test3 sit inside MEM/IO geometry with negative silhouette scores. Segment-level IDLE analysis can show whether early segments look like the prior workload and later segments converge to a quiet baseline. This would directly test the "Residual effects from previous workload in cyclic execution" confound listed in the IDLE mini-spec.

### 7.4 Cycle-to-cycle drift across the 4-cycle dataset

Segments are within-run, but their segment-level summaries can be aggregated by cycle. This separates "workload signature drifts across cycles" from "workload signature changes within a single run". The dataset already has 4 independent realizations per subtype; segmentation adds a within-realization view orthogonal to cycle index.

### 7.5 IO subtype repeatability internalization

`io_many_files` and `io_seq_fsync` show low CV across runs. Segmentation can ask whether their internal structure is similarly stable. If yes, this strengthens their "clean probe" provisional designation in the cleanliness table. If no, the cleanliness comes only from averaging and the within-run dynamics are noisier than implied.

### 7.6 Detect confusion that current pipeline hides

A run might have segment 1 looking like its true subtype and segments 2-4 looking like a different subtype, with the per-run feature averaging to "true class". The current confusion matrix would record this as a correct classification despite hidden instability. Segmentation surfaces this.

### 7.7 Capture-pipeline drift detection independent of workload

If certain segments consistently produce capture artifacts (pmemsave pause spikes, host-system noise bursts) regardless of subtype, segmentation surfaces this as a pipeline characterization, not a workload finding. This is the missing tool the confusion-matrix methodology Section 5 calls for under "capture artifact".

---

## 8. Suggested Segment-Level Metrics

Apply the existing per-run feature pipeline inside each segment. The output is the same feature vector schema as the current per-run vector. In addition, compute segment-trajectory metrics that describe the SHAPE of the segment sequence within a run:

| Metric | Definition | Purpose |
|---|---|---|
| within-run segment CV | std / mean of each per-segment feature across the k segments of one run | within-run stability |
| within-run segment range | max - min of each per-segment feature across segments | extreme-deviation detection |
| segment lag-1 autocorrelation | corr(seg_i, seg_{i+1}) of each per-segment scalar | sequential stationarity |
| segment trend slope | linear regression slope of each per-segment scalar against segment index | drift inside a run |
| first-vs-last segment delta | seg_k - seg_1 for each per-segment scalar | warmup vs late-state contrast |
| segment-to-run-mean distance | per-segment feature distance to the run's own segment-mean centroid | within-run outlier detection |
| segment-to-class-centroid distance | per-segment feature distance to each class centroid (run-level centroid) | confusion localization |
| per-segment confidence margin | difference between distance to true-class centroid and distance to nearest other class centroid | per-segment classification confidence |

Discriminative power of these segment-trajectory metrics can be tested by Kruskal-Wallis at the run level (one trajectory metric value per run, n=48), avoiding the inflated-n trap.

---

## 9. Suggested Visualizations

| Visualization | What it shows |
|---|---|
| Per-run segment trajectory line plots | feature value vs segment index, one line per run, colored by subtype |
| Segment-level small multiples | k subplots per run showing per-segment feature distributions |
| Segment-level PCA, color by subtype, marker by segment index | whether segments of one subtype overlap across runs and across segment positions |
| Segment-distance heatmap, runs × segments | quick scan for runs with anomalous segments |
| Confusion-localization plot | for each misclassified run, distance from each segment to each class centroid, plotted as bars |
| Cycle-and-segment grid for one subtype | rows = cycles, columns = segments, cell color = feature value, separates within-run drift from across-cycle drift |
| First-vs-last segment scatter | seg_1 feature on x-axis, seg_k feature on y-axis, one point per run, identity line drawn, colored by subtype |
| IDLE residual decay plot | per-segment feature value vs segment index for IDLE runs grouped by what workload preceded them |

All visualizations use run-level groupings and segment indices as visible structure. None pool segments to imply iid samples.

---

## 10. Suggested JSON Result Structure

This is a specification, not an implementation. Field types and nesting are illustrative and should be finalized when the analysis is built.

```json
{
  "schema_version": "0.1",
  "config": {
    "k": 4,
    "window": 128,
    "hop": 64,
    "min_windows_per_segment": 50,
    "metric_set_id": "current_per_run_v1",
    "warmup_segments_excluded": 0,
    "trace_kind": "hamming"
  },
  "per_run": {
    "test2_mem_stream": {
      "subtype": "mem_stream",
      "class": "MEM",
      "cycle_index": 1,
      "n_frames": 18432,
      "n_segments": 4,
      "segment_length_frames": 4608,
      "segments": [
        {
          "segment_index": 0,
          "frame_start": 0,
          "frame_end": 4608,
          "n_windows": 71,
          "features": {
            "snr_mean": 1.21,
            "dc_coh": 0.31,
            "cep_periodicity_score": 0.84
          }
        }
      ],
      "trajectory_metrics": {
        "snr_mean": {
          "within_cv": 0.42,
          "lag1_autocorr": 0.61,
          "trend_slope": -0.18,
          "first_minus_last": 0.55,
          "min_to_run_mean_dist": 0.08,
          "max_to_run_mean_dist": 0.31
        }
      },
      "confusion_localization": {
        "predicted_class_per_segment": ["MEM", "MEM", "IDLE", "IDLE"],
        "margin_per_segment": [0.62, 0.51, -0.18, -0.24]
      }
    }
  },
  "per_subtype_summary": {
    "mem_stream": {
      "n_runs": 4,
      "trajectory_summary": {
        "snr_mean.within_cv": {"median": 0.40, "iqr": [0.31, 0.55]}
      },
      "phase_purity_test": {
        "first_minus_last_kw_pvalue": 0.07,
        "interpretation": "marginal early-vs-late difference, not corrected for multiple comparisons"
      }
    }
  },
  "diagnostics": {
    "autocorr_check_pass_fraction": 0.62,
    "stationarity_check_pass_fraction": 0.55,
    "capture_spike_density_per_segment": [0.02, 0.04, 0.18, 0.05],
    "warmup_segments_recommended": {
      "mem_stream": 1,
      "mem_pointer_chase": 1,
      "mem_alloc_touch_pages": 0,
      "io_rand_rw": 0,
      "io_seq_fsync": 0,
      "io_many_files": 0,
      "run_idle": 0
    }
  },
  "limits_of_inference": {
    "segments_are_independent": false,
    "use_for_class_significance_tests": false,
    "valid_inferential_units": ["run", "subtype"],
    "cv_grouping": "by_run"
  }
}
```

The `limits_of_inference` block is mandatory. Any downstream consumer of the JSON should respect it. This is the structural enforcement of Section 6 above.

---

## 11. Thesis-Safe Wording

These phrasings can be used in the eventual write-up without overclaiming.

**On the purpose of segmentation:**
"Segment-level analysis was applied as a within-run diagnostic tool. Each Hamming or cosine trace was split into k contiguous temporal segments, and the existing window-based metric pipeline was applied within each segment. The resulting segment-level features were used for within-run stability analysis, phase characterization, and confusion localization. They were not used to inflate the effective sample size for class-level inferential tests."

**On independence:**
"Segments within a single run were treated as quasi-independent observations only when within-run feature autocorrelation was below a documented threshold and the segment length was much larger than the workload's intrinsic cycle period. Otherwise, segments were treated as repeated measures of the same run, with run as the inferential unit."

**On choice of k:**
"The number of segments k was treated as a research-question selector rather than a hyperparameter. Different k values address different temporal scales, and results are reported as a sensitivity grid (k = 2, 4, 8) rather than as a single optimal value."

**On confusion localization:**
"For each misclassified run, segment-level classification was used to localize the confusion in time. This is consistent with the four-cause diagnostic model in `confusion_matrix_diagnostic_methodology.md` Section 5. It is a hypothesis-generating tool. Confirmatory evidence requires isolation experiments outside the scope of this segmentation analysis."

**On limitations:**
"Segment-level findings describe within-run dynamics. They are not corroborated by the current dataset as evidence of cross-run generalization. The cross-run sample size remains n = 4 per non-idle subtype, and the conservative analysis in `confusion_matrix_diagnostic_methodology.md` Section 6 (provisional language rule) still applies to all class-level claims."

---

## 12. Explicit Connections to the Three Methodology Files

### 12.1 To `current_workload_test_mini_specs.md`

| Mini-spec field | What segmentation lets us check |
|---|---|
| `Mechanism stressed` (per workload) | Whether the temporal mechanism actually shows up inside the run, not only in the run average |
| `Expected memory pattern` for cyclic workloads | Whether the rhythm is detectable at the segment level via cepstral periodicity per segment |
| `Potential confounds` listed for each test | Whether confounds concentrate in particular segments (warmup, end-of-run cleanup, residual contamination) |
| `Cross-Test Interpretation Logic` Step 3 ("Are repeated runs stable") | Extended to a within-run stability question |
| `What Would Count as Strong Evidence` ("repeated runs cluster together") | Refined: do segments AND runs cluster |

The mini-spec template should add an optional field after `Potential confounds`:

```text
Expected segment-level pattern:
- expected within-run trajectory shape
- expected segment-level stability (CV)
- expected warmup region length
- expected cycle period if any
```

This is optional, not a full new section. It only adds a temporal-structure prediction to subtypes that have one.

### 12.2 To `confusion_matrix_diagnostic_methodology.md`

Segmentation is the most direct in-data tool for the four-cause model in Section 5:

| Cause from Section 5 | Segmentation evidence pattern |
|---|---|
| true behavioral overlap | all segments of the misclassified run consistently look like the wrong class, and same-subtype runs without confusion show similar segment patterns |
| capture artifact | a single segment dominates the misclassification, with high spike density in that segment |
| execution-order contamination | early segments look like the prior workload, later segments converge to the true subtype |
| metric inadequacy | segments are internally consistent but the metric averages them into the wrong region of feature space |

Section 6 (provisional language rule) is unchanged. Segmentation does not increase n.

Section 7 (confusion direction rule) extends to segments: a misclassified segment should be reported as `true subtype, segment i → predicted subtype`.

Section 13 (typed expected confusions) gains a fifth field option for confusions that are localized in time:

```text
- `[true subtype] → [predicted subtype]`
  - Type: [cause type]
  - Reason: [mechanistic explanation]
  - Segment localization: [early / middle / late / scattered / all]   <-- new optional field
```

### 12.3 To `vm_workload_executables_purpose_for_ai.md`

That file frames the workloads as controlled stimuli and emphasizes:

> A useful behavioral fingerprint must be both repeatable and separable.

Segmentation extends repeatability to two timescales:

- across runs (existing CV analysis)
- within a run (segment-level CV)

A subtype is a strong fingerprint candidate if both are stable. A subtype that is run-stable but segment-unstable is masked by aggregation. A subtype that is segment-stable but run-unstable has reproducible local dynamics but unreliable global identity. Both patterns are now visible.

That document also states:

> A method can separate samples in a plot but still be unreliable if repeated runs of the same workload are unstable.

Segmentation supports this skepticism. Per-segment trajectories that disagree across runs of the same subtype indicate the run-level "fingerprint" is an averaging artifact.

---

## 13. Summary

Segmentation is a useful diagnostic. It is not a sample-size multiplier. Its strongest uses are within-run stability, confusion localization, phase identification, and capture-artifact characterization. Treating segments as iid observations is the principal failure mode and must be explicitly disallowed in downstream usage. The choice of `k` is a research-question selector, not a hyperparameter, and should be reported as a sensitivity grid. All segment-level findings remain bounded by the provisional-language rule and the four-cause diagnostic model already adopted in `confusion_matrix_diagnostic_methodology.md`.
