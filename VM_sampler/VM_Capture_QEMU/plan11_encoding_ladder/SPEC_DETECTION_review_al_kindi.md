# SPEC_DETECTION.md, review by al-Kindi (2026-09-17)

Read in full: `SPEC_DETECTION.md`; `apf_paper/P3_RAID_STRUCTURE.md`; `raid_council/06_al_kindi_revised.md`
(the moves are its section 4, not section 5 as the task says; section 5 is the overrule list);
`raid_council/02_ml_engineer_protocol.md`; `raid_council/08_RAID_COUNCIL_REPORT.md` sections 1.5 and 2;
`raid_council/05_al_farabi_certification.md` section 5. Verified in code: `series.py` (the feature
builder, the head drop, the rung series), `schema.py`, `extract.py` (the ratio columns, `J_null`,
`dt_est_s`), `models.py` (`make_forest`), `splits.py`, `nulls.py`, `verdicts.py`,
`gates_precondition.py` (C1 and G-K0), `gates_readings.py` (the G-J mask), `gates_comparison.py`
(G-X), `synth.py` (presets, `SynthSpec`). No server, no data, no name of the sandbox family read.

The question I was asked: does this layer compute what my detection moves need (the fused plane per
tier with members as shapes, the miss table with the nearest benign cloud and the axis, the
time-to-hear ladder, the level-matched control), does anything smuggle the class label or the level
into a feature, and is the three-level question computed exactly as `P3 0a` decided it.

## 1. Passes as specified

1. **The feature matrix is clean.** `series.build_features` writes `X` as its own array and the
   metadata (`cell_id`, `kernel`, `archetype`, `campaign`, `role`, `rep`, `win_start`,
   `n_series_cell`) as separate arrays; nothing in `feature_names(rung)` is a count, a cadence, an
   order index or a label. The normalized channels are level-free by construction (`K / K_median`,
   `ham_sum / (K_median * 32768)`, `J - J_null`, the fifteen `r_*_per` ratios), and the
   normalization is a per-cell function computed once at build time, identical in every fold, as
   al-Farabi's condition 5 requires. The raw `apf` row is kept and labelled as the level-inclusive
   ceiling. The exclusion list `excluded_by_declaration` is recorded before B1-G3 runs. This is the
   form's line and the spec holds it.
2. **The class label enters only where it may.** The mapping file is the author's, read and never
   guessed; `y` lives in the label dict and is used for fitting, fold selection, the null, scoring
   and the denominators; the letter sequence is class-only; every `cell_id` becomes public at
   `classes apply` before `extract all`. The join carries `split_role = test_only` for stage 3 so an
   external cell is never trained on. Stages 2 and 3 enter by rows of the class file with no schema
   change.
3. **Level 1 is exactly `P3 0a` and `P3 D5`.** LOWO over the workloads of both classes (21 folds on
   stage 1), whole cells held out, sandbox folds giving the TPR and benign folds the FPR, the
   in-fold threshold at the declared five percent with the realized out-of-fold FPR beside it, the
   one percent reading labelled at the resolution limit, ROC area, the workload-level shuffle null
   with `null_not_estimable` below 20 assignments, the majority baseline and the random scorer,
   per-member recall in eighths and never a mean. LOCO, LOFO and the one-class reading stand as rows
   at every S (my overrule of the ML engineer, `06_` section 5 point 1, is honoured).
4. **Levels 2 and 3 have the right shape.** Level 2 holds one member out and learns from every
   other sandbox cell of every sub-family, reports a row per sub-family, and B reads "one member, no
   held-out test" with no fold; level 3 is leave-one-rep-out over the eight members and every row
   carries `SIGNATURE_CEILING`. Both use the multiclass forest with majority vote per cell, which
   keeps the unit the cell. Two corrections to their nulls and denominators are in section 2.
5. **The level-matched control is honest for stage 1.** G-LM bands on median K with the label
   `median K, stage 1` on every row, the band a factor of two, the operating point recomputed by
   retraining with the threshold on the band benign only, and the three outcomes I asked for
   (`GLM_NOT_DETECTED`, `GLM_EMPTY_BAND`, `GLM_LEVEL_ONLY` against `GLM_SURVIVES`) are all reachable
   and all tested through the pure verdict function. The per-iteration quantity waits for the
   `[SUSTAIN]` markers, correctly.
6. **The time-to-hear ladder is my move 17.** Prefixes of 30, 60, 120, 300, 600 s in pairs at a
   fixed spacing (47, 93, 186, 466, 932, capped at the cell), the prefix normalized by its own
   median K (what a detector at 30 s would know), the two readings with `from_boundary` refusing
   itself until stage 2, the in-fold threshold recomputed per prefix, the unit still the cell.
7. **The miss table and the false-positive table exist with the columns I asked for**: nearest
   benign workload, its family (the tier of the neighbour), the distance, an axis, the
   coordinates, `AT_FLOOR_NOT_A_MISS` for the floor, the physical reason left to the author, the
   note "assignments with counts, never a confusion matrix". Two corrections to the plane and the
   axis semantics are in section 2.
8. **The figures obey the name-only rule**: members as eight marker shapes indexed by member,
   colour by sub-family letter, legend `member m (A)`; the idle cloud in every panel.
9. **The blind order test, G-ANCHOR read as a size and not as a switch, the cross-campaign row as
   `not applicable: stage 1`, the harness clause refusing itself until stage 2, the explicit
   not-built list (rung 2', the yield, the cepstrum, the comparator, the mixture arm)**: all as
   decided. Nothing is filled silently.
10. **The synthetic corpus is a real instrument**: eight members with known answers on distinct
    axes (amount, identity, floor, a pure level copy of a kernel, an empty band), so every verdict
    of this layer has a case that must produce it. One known-answer row is wrong (section 2, item 8).

## 2. Must change before build

1. **SPEC 3.3.1 and 3.3.2, `THRESHOLD_SOURCE = "oob"`: the out-of-bag threshold is in-sample by
   sibling windows and cannot deliver the declared operating point.** The rows of `X` are windows,
   and adjacent windows at `W8_H4` share four of eight pairs. A window's out-of-bag trees were
   trained on about two thirds of its own cell's other windows, so a benign training cell's
   out-of-bag score is close to its in-sample score, the 95th percentile of those scores is far
   below the score an unseen benign workload will get, and the realized FPR will overshoot five
   percent by construction, not by chance. The ML engineer's wording (ML 1.6 item 2) assumed one row
   per cell (ML 1.1); with window rows the in-fold threshold must be set on benign scores that are
   out-of-workload. Correction: `THRESHOLD_SOURCE = "inner_lowo"` becomes the default for every
   two-class split and for the null (the same rule in the observed run and in every permutation);
   add `"inner_group_kfold"` with `INNER_K = 4` (benign training workloads grouped by
   `workload_key`, seeded from `SEED_FOREST + offset`) as the cheap variant; keep `"oob"` and label
   it in `params` as `optimistic: sibling windows in-bag`. G-OP's `setters_05` are then the benign
   training cells whose in-fold score under the chosen source is at or above the threshold. Update
   the cost table of 6.2 (LOWO headline 21 x (1 + B) or 21 x 5 fits; the null multiplied
   accordingly) and section 7 item 9. Which inner rule the author pays for is section 3, item 1.
2. **SPEC 3.3.6, 3.3.7 (CLI), 5.2.7 and 6.2: the one-class run has no null, so F9 cannot be read.**
   Table 11 promises the sentence "separable when heard, not flagged when not" when the one-class
   `tpr05` is inside its null while LOWO's is above it, but section 3 defines no one-class null and
   the `one-class` CLI has no `--null-perm`. Correction: `run_one_class(..., n_perm, n_jobs)` takes
   the same workload subsets as 3.3.4 (`workload_label_permutations`, the same seed); per subset the
   permuted-benign workloads are the training pool (per-workload folds for the inner threshold,
   the final model on all of them), the permuted positives are scored, `tpr05` and `auc` are
   recorded, `null_verdict` as 3.3.4; written to `one_class__<variant>/null.json` and
   `scores.json["null"]`. Add `--null-perm` and `--n-jobs` to the `one-class` CLI, let
   `--null-splits` accept `one_class`, and have the driver pass it in D7 and D9 to D12.
3. **SPEC 3.4 `run_level2` and 5.2.8: the level-2 null statistic is a macro average, which `P3 0a`
   forbids, and it contradicts the table's per-row null columns.** `P3 0a`: "a level-2 headline is
   reported per sub-family, never averaged". Correction: per permutation record the recall of every
   headline letter separately; `null.json` holds one array per letter; `confusion.csv` carries
   `null_p95, rank, verdict` per row from its own letter's array; `macro_recall` may stay in
   `scores.json` as a number with no verdict and is not printed. The exhaustive 280 assignments are
   unchanged.
4. **SPEC 3.4 `run_level2` and `run_level3`: at-floor cells stay in the level-2 and level-3
   denominators.** F3 and ML 3.6 put a cell at floor out of every true-positive denominator with its
   verdict; a sub-family recall or a member accuracy that counts a floor cell as a miss is the same
   error one level down. Correction: both levels compute recall over cells with `floor_verdict ==
   GK0_ABOVE_FLOOR`; `confusion.csv` and `members.csv` gain `n_at_floor`; at-floor cells are listed
   in `predictions.csv` with `in_denominator = false` exactly as in 3.3.7. Also record in level 3's
   `scores.json` and `params` the line `null_unit = "cell"` with the reason (the label is the
   workload, so a workload-level permutation is a relabel and leaves accuracy unchanged; this null
   is the weak one of ML 1.4 and the level is the signature ceiling anyway).
5. **SPEC 3.1.4 and 3.5.1, `det_c1_rule`: the C1 rule is asymmetric across classes.** A C1 fail
   excludes a `benign_kernel` cell but reports a cell of any other class. A benign kernel at a low
   level (the small-footprint kernels that E1 6.7 says the 0.02 default refuses) leaves the benign
   world while a sandbox member at the same level stays; that biases the benign side upward in level
   exactly where the members may sit, and it is an inclusion rule that depends on the class label.
   My claim says "every benign family at the floor is reported at floor, never as a detection
   failure", not excluded. Correction: `det_c1_rule` applies to every class alike: `"report"`
   (default) keeps every C1-failing cell of every class admissible with G-K0's floor verdict;
   `"exclude"` applies plan11's rule to every class. Table 4 gains `n C1 fail (reported)` per class.
   The encoding paper's C1 stays the encoding paper's.
6. **SPEC 3.1 (silent), `inputs/head_drop.csv`: the head drop is keyed by workload and the spec
   never says what the sandbox members get.** `series.build_features` calls
   `head_drop_for(head_drop, c["kernel"])`, so after `classes apply` the key is
   `sandbox_member_<m>`; the template (`write_head_drop_template`) lists kernels and `idle` only. A
   drop applied to benign workloads and not to members is a per-class preprocessing (al-Farabi's
   address (ii)). Correction: state in 3.1.1 that the head drop is 0 for every workload key of every
   class unless the author declares a value per key before the data; `classes apply` extends the
   template with one `sandbox_member_<m>` row per member (0, reason "default"); every `params` block
   records `head_drop_json` as it already does for the kernels.
7. **SPEC 3.5.16 and 5.3 `fig4`: the miss table and the fused plane use different identity
   coordinates and different masks, so the table cannot be read against the figure.** The miss
   table's identity is `J - J_null`; Figure 4 draws raw `J`. The figure applies the G-J mask to the
   kernels only (`gates_readings.gate_gj` writes masks for `kern + idle`; sandbox cells get none),
   so kernel clouds lose their floor-like pairs and member clouds keep theirs; the miss table's
   per-cell medians are unmasked. Correction: one plane. Identity is `J - J_null` on both (the
   level-free coordinate; `J_null` grows with K, and this plane spans levels from hundreds of pages
   to tens of thousands), with `--identity excess|raw` on both and `excess` the default. The mask is
   one per-pair rule for every class, `K > k_factor * floor_K` with `k_factor` and `floor_K` read
   from `gates/gj.json`, computed inside `figures_detection` and `miss_table` from the extract (no
   edit to `gates_readings.py`), recorded as `plane_mask = "gj_mask_K, every class"`; when `gj.json`
   is absent both are unmasked and say so.
8. **SPEC 3.5.16 and 4.4: the `axis` column has one rule in the docstring and the opposite reading
   in the known answers.** The rule is "the axis whose absolute standardized difference to the
   centroid is smaller" (the resemblance axis, which is what my move 18 means by "the axis of
   smallest distance"). Member 5 is `double` content with churn 0.60 and rmat_gen is `double` with
   churn 0.30, so their amount coordinates coincide and they differ on identity: under the rule the
   axis is `amount`, yet 4.4 says "nearest rmat_gen on the identity axis". Correction: keep `axis`
   as the resemblance axis, add `axis_of_largest` (where the residual difference lies, the lead
   that could have heard it), and print both standardized differences (`d_amount`, `d_identity`);
   fix the 4.4 row to `member 5: nearest rmat_gen, axis amount, axis_of_largest identity`. Member 8
   is `double` with churn 0.05, which is spmm's shape exactly (not nbody's as 4.2 says), so its
   test asserts `nearest_family = kernels` and a small distance, never the axis.
9. **SPEC 3.5.14 `alias_detection`: a regression across all cells confounds cadence with class.**
   If the sandbox cells were captured at a slightly different realized spacing than the kernels
   (a different week, a different host load), every feature that separates the classes also
   correlates with `dt_est_s` across all cells and would be declared an alias, including genuine
   behaviour; conversely nothing is learned about cadence itself. Correction: (a) the regression
   uses workload fixed effects (feature and `dt_est_s` centred per `workload_key`, pooled over all
   admissible cells; `alias_unit = "within_workload"`), which is what asks whether a feature moves
   with cadence holding the workload fixed; (b) add the ML 3.4 row literally: `cadence_as_class`,
   the one-feature threshold model on `dt_est_s` alone (or `n_pairs`) under LOWO with the
   workload-level null, verdict `cadence audible` on strict exceedance; record the per-class median
   `dt_est_s` and the between-class difference in `params`. Both regressors remain logs, never
   features.
10. **SPEC 3.5.7 `drift_regression`: on a class blocked by workload the slope measures the member
    levels' order, not drift.** The sandbox cells ran in order by member, so `K_med` against
    `order_index` inside the class is a staircase of eight member levels; any non-flat ordering of
    those levels exceeds a shuffle null and the paper would print a false drift disclosure.
    Correction: regress with workload fixed effects (residual `K_med` after the workload mean, on
    `order_index`), null = shuffle of `order_index` within each workload, `drift_unit =
    "within_workload"`; keep the plain regression as a second row labelled `confounded with workload
    order (blocked)`.
11. **SPEC 3.5.7 `order_test`: at stage 1 the within-class half label is member identity by
    construction, so the required row of RQ3 cannot be read as spec'd.** For the sandbox class the
    first half is members 1 to 4 and the second is 5 to 8, which is level 2 in disguise; the row can
    only read `ORDER_AUDIBLE` and says nothing about position. Correction: add `half_rule =
    "within_workload"` (default when a class is blocked, detected as every workload inside one
    half): the label is the first four against the last four cells of each workload by realized
    order, folds LOWO across workloads, so the test asks whether position is audible once identity
    is stripped, which is the question. Null: the half labels permuted within each workload (C(8,4)
    per workload; 500 draws). Keep `"within_class"` as the ML-literal row beside it. For idle this
    coincides with `idle_early_late`, and the two rows must agree.
12. **SPEC 5.3 `fig4`: one panel per sub-family letter loses the one thing I watch for.** Move 11
    asks whether the members fall together as a class; three panels cannot show that. Correction:
    one panel for the class with all eight members as shapes, coloured by letter (the colour already
    answers the sub-family question inside the one panel); per-letter panels only under
    `--plane-per-letter`.
13. **SPEC 2.2 items 9 and 17: two refusal strings quote the author's `path_prefix`.** That is the
    only channel by which an author-typed workload name can reach a file under `gates/detection/`
    (the manifest hashes it and a future table might print it). Correction: quote row numbers only
    (`refused: duplicate path_prefix in rows <n> and <k>`; `refused: two members of one workload
    path in rows <n> and <k>`).

## 3. For the author

1. **The inner threshold rule (item 1 above).** `inner_lowo` is exact and costs B extra fits per
   fold (about 21 x 14 per LOWO run, times 500 for the null); `inner_group_kfold` with k = 4 costs
   21 x 5. Both are honest; `oob` is not, with window rows. Decide before the data and pass one
   value to every split, the null and the ladder.
2. **The LOCO null.** Without it G-SIG cannot refuse (F2). `--loco-mode rep_index` gives 8 folds
   and a null of 4,000 fits per rung; `cell` gives 84,000. The rung inside the LOWO null is refused
   by G-L (i) anyway; what the LOCO null adds is the sentence "signature matching, not detection".
3. **LOFO under `kernel_family_rule = "tier"` has two folds at stage 1**, and the kernels' fold
   trains on idle plus the sandbox only, which will flag every kernel. That is the definition's
   reading, and it is an honest size, but it is almost content-free. Both rules are cheap (2 and 4
   folds); I would run `tier` and `archetype` as two labelled rows of Table 11.
4. **The ladder without a null** reads TPR against prefix length and nothing else; the N3 sentence
   "the shortest prefix at which the rung clears the null" needs `--ladder-null-perm 500` on at
   least the headline rung. The one-class reading per prefix (14 isolation forests per prefix) is
   cheap and is the reading "the conductor hears an off-note at 30 s"; it is not in the spec and I
   would add it as `--ladder-one-class`.
5. **G-ANCHOR (ii) at stage 1 is `not applicable: one idle campaign`**, because the eight idle cells
   are one set captured after the sandbox run. The kernel period (6 to 8 September) has no idle
   floor of its own; the three launch labels of the kernels under G-X (part (i)) and the drift rows
   are the only campaign readings available. State that limitation in Table 6's note.
6. **`NULL_VERDICT_STATISTIC = "tpr05"`** carries the verb; AUC's verdict is printed beside it. I
   agree with the choice (`P3 D5`); confirm it.
7. **`ONE_CLASS_MODEL = "isolation_forest"`** is neither generative nor a density model (ML 1.6);
   `gmm` is. Either is defensible; the primary must be named now.
8. **`GK0_VERDICT_QUANTITIES`** gates on the three K quantities and only discloses `l0` and J; CR3
   2.8 says "every quantity". The default is the safe one for cells with no pair above the band;
   confirm or include all five.
9. **The realized order** must be supplied as 168 per-cell `order_index` rows before the order
   test, the drift rows, early-against-late idle and the letter sequence can run; without it those
   rows refuse.
10. **The sandbox source part of G-K0** (`inputs/gk0_source_sandbox.csv`) is yours, one numbered
    line per member. The synthetic corpus writes `unstated`.
