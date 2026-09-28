# Build report: the detection library (builder A, build epoch 2)

Written 2026-09-17 against `SPEC_DETECTION.md` sections 1(a) to 1(e) and 4, with the corrections
marked "must change before build" in the three reviews (`SPEC_DETECTION_review_al_kindi.md`,
`SPEC_DETECTION_review_ml.md`, `SPEC_DETECTION_review_al_farabi.md`) implemented in place of the
original SPEC lines. Every test uses synthetic data generated on this machine. No server, no real
data, no workload name of the sandbox family, no run record was opened. Nothing was committed.

## 1. Files written

New modules under `VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/`:

| File | Owns |
|---|---|
| `classes.py` | the class mapping file, its validator (items 1 to 18), `apply`, the join, the letter sequence, `inherit-selection`, the head-drop extension, the CLI of SPEC_DETECTION 2.8 |
| `detection_splits.py` | the label dict at the cell unit (`ROW_UNIT = "cell"`), LOWO, LOCO, LOFO, the one-class folds, level 2, level 3, the order folds with both half rules, the idle anchor folds, the grouping assertion on every list |
| `detection_metrics.py` | the forest with in-fold thresholds (`inner_lowo`, `inner_group_kfold`, `oob`), the operating point and its pooled reading, the workload-level null, the one-feature model and the L1 row, B1-G3 two-class, the one-class run with its null, the split stage (`run_detection_split`), the binary-label splits (order, idle sets, early against late) with their nulls, the ladder, the CLI of SPEC_DETECTION 3.3.7 |
| `detection_levels.py` | level 2 with the per-letter null and level 3 with the cell-level null, the CLI of 3.4 |
| `gates_detection.py` | admissibility, G-K0 two-class, G-N, G-L (i), G-OP, G-LM, G-ANCHOR, the order test, the drift regression, G-SIG, G-FP, G-1C, the harness clause, G-CAL, G-M, G-DIM, the alias falsifier with the cadence row, the leak probes, G-V two-class, the miss table, the head-drop template, the CLI of 3.5.17 |
| `synth_detection.py` | the two-class corpus with known answers and its switches (section 4) |

Existing modules extended additively (the encoding-paper behaviour is unchanged, the existing
suite passes, section 3):

| Module | Extension |
|---|---|
| `verdicts.py` | the constants of SPEC_DETECTION 3.0 appended, plus `LEAK_AUDIBLE`, `LEAK_NOT_AUDIBLE`, `CADENCE_AUDIBLE` (ML review 2.6, al-Kindi 2.9) and `REPS_NEAR_IDENTICAL` (ML review 2.8); the named refusals of 3.0 added to `NAMED_REFUSALS`; `level2_no_heldout(n)` |
| `models.py` | `make_forest(..., *, oob_score=False, min_samples_leaf=1, class_weight=None)` with defaults that reproduce the present forest exactly; the public alias `reduce_fit = _reduce_fit` |
| `synth.py` | `SynthSpec.family = "kernel"` and `SynthSpec.test_label_fmt = "kernel_{name}_v2"`; `test_label` and `cell_dir` read them; the defaults reproduce every present path |

`run_moves.py` was not touched: its extensions of SPEC_DETECTION 1.3 serve builder B's driver.

Tests under `tests/` (named `test_detection_*.py` as the task requires; SPEC 1.1 names them
`test_classes.py` and so on): `_det_common.py`, `test_detection_classes.py`,
`test_detection_splits.py`, `test_detection_metrics.py`, `test_detection_levels.py`,
`test_detection_gates.py`, `test_detection_synth.py`.

## 2. Gate by gate and metric by metric

Every function's docstring cites the definition it implements. The table names the passing test
and the refusing test (or the named outcome) of SPEC_DETECTION 4.5; `main`, `on` and `tiny` are
the three synthetic corpora of `tests/_det_common.py` (section 4).

| Function | Must pass | Must refuse or the named outcome | Citation |
|---|---|---|---|
| `classes.validate_classes` | `test_validate_ok_and_counts` (the written file, `status ok`, counts, members, letters, the M1 warning) | `test_validate_refusals` (21 one-line variants, one per item of 2.2, each the exact string; items 9 and 17 quote row numbers only) | P3 0a; CR3 1.8; CR3 2.31 |
| `classes.apply_classes` | `test_apply_rewrites_public_ids_and_is_idempotent` (`sandbox_member_<m>__rep<rr>__stage1`, `role = sandbox`, `status = ok`, `cells.pre_classes.csv`, a second apply changes nothing, the join, S = 3, B = 3, C(6, 3) = 20) | `test_apply_refusing_file_leaves_cells_untouched` (cells.csv byte-identical, the validation file written); `test_apply_refuses_stray_extract_dirs` (M11); `test_apply_copies_classes_and_refuses_a_different_existing_file` | P3 0a; al-Farabi M1, M11 |
| `classes.build_join` | `test_join_archetype_rule_and_unassigned` (`kernel_family_rule = archetype`; an ok cell without a row is `unassigned` with empty derived columns) | the same test (a refused cell is not listed) | ML 1.2; P3 0a |
| `classes.letter_sequence` | `test_letter_sequence_and_not_run` (the token string) | the same test (`not run: order_index missing for 12 cells`) | P3 D4; ML 3.2 |
| `classes.inherit_selection` | `test_inherit_selection_default_and_from_and_refusal` (`--default W8_H4` with `grid_source`; `--from` verbatim with the sha256) | the same test (a `select`-written file without `--force` refused; the record in `inherit_selection.json`, M10) | ML 1.1; SPEC 3.5.7 |
| `classes.extend_head_drop` | `test_extend_head_drop_adds_member_rows` | (a declared constant; no refusal) | al-Kindi 2.6; al-Farabi M5 |
| `detection_splits.make_detection_labels` | `test_make_labels_cell_rows_collapse_windows` (one row per cell, the nanmean, the aliases) | `test_make_labels_window_rows_and_missing_cells` (window rows; a cell absent from the feature file listed) | ML 1.1; CR3 1.5; ML review 2.1; al-Farabi M4 |
| `detection_splits.fold_lowo` | `test_fold_lowo_counts_and_grouping` (7 folds on the hand-built dict, `lowo/final` for a test-only class); `test_folds_on_the_main_corpus` (6 + 1 + 8 folds) | `test_assert_grouped_refuses_a_leaking_fold` (`AssertionError`) | ML 1.3; CR3 1.5; K3 2.2 |
| `detection_splits.fold_loco` | `test_fold_loco_modes_and_final` (cell and rep_index; test-only never trained on) | (the grouping assertion) | ML 1.3; K3 F2 |
| `detection_splits.fold_lofo` | `test_fold_lofo_and_one_class` (kernels, idle; sandbox in every training set) | the same test (`[]` with no benign family) and `test_loco_and_lofo` (`not applicable: sandbox never held out under LOFO`) | K3 2.2; K3 F8 |
| `detection_splits.fold_one_class` | `test_fold_lofo_and_one_class` (no sandbox row in any training set) | (n/a) | ML 1.3; CR3 2.19 |
| `detection_splits.fold_level2`, `fold_level3` | `test_fold_level2_and_level3` (7 folds, B has none; 2 rep folds) | the same test (`min_members_test = 5` gives `[]`) | P3 0a |
| `detection_splits.order_half_labels`, `fold_order`, `fold_anchor_idle` | `test_order_half_labels_and_folds` (both half rules; LOWO across workloads; leave-one-cell-out for idle) | the same test (`[]` without order_index) | ML 3.2; al-Kindi 2.11 |
| `detection_metrics.in_fold_threshold` | `test_in_fold_threshold_linear_and_strict_flag` (100 scores at fpr 0.05 give 94.05) | (a table test) | ML 1.6 item 2 |
| `detection_metrics.run_operating_point`, `summarize_operating_point` | `test_run_operating_point_separable_and_chance` (tpr05 1.0, fpr05 0.0, auc 1.0, setters non-empty) | the same test (the same distribution gives auc in [0.2, 0.8]); `test_denominator_rules_control_pending_and_no_score` (control counts as at floor; a pending G-K0 gives `not run: gk0 not run`; a cell without a score is never a miss) | P3 D5; CR3 1.5; ML 1.6, 2.3, 2.4, 3.7; ML review 2.2; al-Farabi M3, M4 |
| the threshold sources | `test_threshold_sources_and_window_rows` (`inner_lowo`, `inner_group_kfold`, `oob`) | `test_split_written_refusals` (`oob` under window rows refused) | al-Kindi 2.1; ML review 2.1 |
| at-floor positives out of training | `test_at_floor_positives_leave_training` (`n_train_dropped_at_floor`) | the same test (`train_on_at_floor = True` keeps them) | ML review 2.5; al-Farabi M2 |
| the null (`count_assignments`, `workload_label_permutations`, `null_verdict`) | `test_count_assignments_and_permutations` (203,490; exhaustive 15 subsets), `test_null_verdict_rules` (PASS above p95) | the same tests (`NULL_NOT_ESTIMABLE` at 19; a tie is `NULL_INSIDE`; `not run: 20 permutations < 500`) | CR3 2.1; ML 1.4, 2.2 |
| `binary_label_permutations` | `test_exact_binary_label_permutations` (counts kept within each workload; the workload unit when every workload lies inside one half) | (a table test) | ML 3.1, 3.2; al-Kindi 2.11 |
| `l1_row`, `l1_feature_run`, `quarantine_l1_two_class` | `test_l1_row_and_quarantine_on_hand_built_data` (the L1 row at 1.0; the noise feature never quarantined); `test_lowo_content_known_answers` (the default corpus quarantines content features and keeps both readings) | the same tests | ML 1.5; ML review 2.4; CR3 2.2 |
| `run_detection_split` lowo | `test_lowo_content_known_answers` (section 4) | `test_lowo_smoke_null_verb` (`not run: 4 permutations < 500`); `test_split_written_refusals` (no selection; raw combined; `gk0_cells.csv` missing) | CITATION_OP |
| `run_detection_split` loco, lofo, subset, exclude, matched | `test_loco_and_lofo`, `test_subset_and_exclude_families`, `test_reduce_to_strongest_cli_and_matched_row` | (in the same tests) | K3 F2, F8; CR3 2.17, 2.15, 2.32 |
| `run_one_class` | `test_one_class_primary_null_and_usage_refusal` (primary, the null, no sandbox cell in any training fold) | the same test (`usage: a second one-class model needs --secondary`, exit 3, nothing written) | CR3 2.19; ML review 2.7; al-Farabi M10 |
| `prefix_rows`, `slice_extract`, `boundary_start`, `run_ladder` | `test_prefix_rows_and_slice_extract` (47, 93, 186, 466, 932), `test_ladder_prefixes_and_whole_cell_equals_headline` (47 and 120 rows; the 600 s features equal the headline's on every cell; tpr05 equals the headline) | the same test (`from pair 1 only` without a boundaries file; `not applicable: prefix shorter than one window (n = 6 < W = 8)`) | K3 move 17; CR3 2.29; al-Farabi M6 |
| `detection_levels.run_level2` | `test_level2_rows_per_letter_null_and_denominator` (A headline, B `one member, no held-out test`, C with 3 at floor, the null exhaustive at 280, one array per letter) | `test_level2_smoke_null_and_no_headline` (`not run: 5 permutations < 500`; `not applicable: no headline sub-family`; `4 members, no held-out test`) | P3 0a; CR3 2.5; al-Kindi 2.3, 2.4 |
| `detection_levels.run_level3` | `test_level3_signature_ceiling_and_one_member` (8 rows with `signature ceiling`, the cell-level null) | the same test (`not applicable: one member`) | P3 0a; al-Kindi 2.4 |
| `gates_detection.admissibility` | `test_admissibility_report_and_exclude` (member 6 and the lexer admissible under `report` with the reason string) | the same test (`exclude`); `test_admissibility_without_preconditions_and_unassigned` (M9; `not run: gates/preconditions.csv missing`) | CR3 2.23; K3 move 2; al-Kindi 2.5 |
| `gates_detection.gk0_cells` | `test_gk0_cells_floor_verdicts_and_edge` (member 6 at floor, member 1 above, the lexer at floor, the edge equals `gates/gk0.csv`) | `test_gk0_cells_refusals` (`not run: no admissible idle cell`; `refused: idle band edge differs from gates/gk0.csv (...)`, M8) | CR3 2.8; K3 F3; ML 3.6 |
| `gates_detection.gn_two_class` | `test_gn_two_class_counts` (sandbox `headline` with 7 members above floor, idle `single workload`) | the same test (`one training workload per fold` at 2, `no supervised headline` at 1, by fixture) | CR3 2.5; ML 2.6 |
| `gates_detection.gl_two_class` | `test_gl_two_class_pass_and_level_only` (`content` after a passing null) | the same test (`level only` from a `NULL_INSIDE` fixture; the null's own refusal copied; `lowo norm split not run`) | CR3 2.4 |
| `gates_detection.gop`, `gop_verdict` | `test_gop_supported_and_set_by_few` (every fold supported) | the same test (one fold set by one workload; the `realized_fps` rule) | CR3 2.13; ML 2.3; ML review 2.3 |
| `gates_detection.glm_verdict` | `test_glm_verdict_table` (`GLM_SURVIVES`) | the same table (`GLM_LEVEL_ONLY`, `GLM_NOT_DETECTED`, `GLM_EMPTY_BAND`, `set by one workload, family named (level band)`) | CR3 2.17; ML 4.3; K3 F1 |
| `gates_detection.glm` | `test_glm_on_the_corpus` (at least two of members 1 to 4 `GLM_SURVIVES`, `median K, stage 1` on every row) | the same test (member 6 `at floor (G-K0)`, member 7 `GLM_NOT_DETECTED`, member 8 `GLM_EMPTY_BAND`; the `headline_oof` model labelled post hoc) | CR3 2.17 |
| `gates_detection.anchor` | `test_anchor_on_corpus_audible` (`on`: the two idle sets differ by their floor, AUC 1.0; part (i) from `gates/gx.csv`) | `test_anchor_main_not_applicable_and_gx_missing` (`not applicable: one idle campaign (n = 4 cells)`; `gates/gx.csv missing`) | CR3 2.14; ML 3.1; K3 move 4, 21 |
| `gates_detection.order_test` | `test_order_test_not_audible_audible_void_and_missing` (`main`: `position not audible above the null` under both half rules) | the same test (`on`: `position audible` on the within-class row with the workload null unit; `--consequence void` gives `ORDER_VOID`; `tiny`: `not run: order_index missing`; the campaign scope row) | ML 3.2; CR3 2.20; K3 F5; al-Kindi 2.11; al-Farabi M12 |
| `gates_detection.drift_regression` | `test_drift_none_and_slope` (`main`: `no drift above the shuffle null` on the within-workload rows) | the same test (`on` with `--drift-level`: `drift: slope above the shuffle null`; the plain row labelled `confounded with workload order (blocked)`; `tiny`: not run) | CR3 2.20; al-Kindi 2.10 |
| `gates_detection.gsig` | `test_gsig_reported_identity_and_not_run` (`gap reported`) | the same test (`recognizes workload identity, not family behaviour`; `not run: loco null not run`) | CR3 2.16; K3 F2 |
| `gates_detection.gfp` | `test_gfp_attributed_and_inseparable` (`attributed`) | the same test (5 of 8 idle cells flagged gives `inseparable from sandbox under this rung` and the `lowo__without_idle` run) | CR3 2.15; K3 F8 |
| `gates_detection.g1c` | `test_g1c_primary_and_search` (`primary`) | the same test (two directories, none primary: `refused: a second one-class model without a declared primary reads as a search`) | CR3 2.19 |
| `gates_detection.harness` | `test_harness_stage2_absent_and_present` (`tiny`: three margins per feature and a verdict; the relaunch block) | the same test (`main`: `not run: stage 2 absent`) | CR3 2.21; K3 F4; CR3 2.12 |
| `gates_detection.gcal` | `test_gcal_agree_and_perfold` (`GCAL_AGREE`) | the same test (`GCAL_PERFOLD`; `not run: null spread unavailable`) | CR3 2.18 |
| `gates_detection.exact_sign_test`, `gm` | `test_exact_sign_test_and_gm` (0.03125, 0.03515625, 0.0195; six improving and none worsening) | the same test (0.125; `difference with margin`; `not run: seed spread unavailable`) | CR3 2.7; ML 2.5 |
| `gates_detection.gdim` | `test_exact_sign_test_and_gm` (`full vector`, d = 36) | the same test (`not run: lowo_matched split not run`) | CR3 2.2; SPEC 3.7.7 |
| `gates_detection.alias_detection` | `test_alias_stays_moves_and_cadence_row` (`does not move with the interval` with workload fixed effects; the `cadence_as_class` row) | the same test (`moves with the interval` when dt is proportional to the strongest feature; `not run: no iteration count (stage 1)`) | CR3 2.26; al-Kindi 2.9; ML 3.4 |
| `gates_detection.leak_probe` | `test_leak_probe_not_audible_and_audible` (`main`: `leak not audible above the null` on n_pairs and dt_est_s) | the same test (`tiny` with the members at 32 pairs: AUC 1.0, `leak audible` or `null_not_estimable` at 15 assignments) | ML 3.4, 3.5; ML review 2.6 |
| `gates_detection.gv_two_class` | `test_gv_two_class_report` (36 rows, the summary note rule, the per-member `L0_ratio` and `REPS_NEAR_IDENTICAL`) | (a report; no refusal by definition) | CR3 2.11; K3 move 20; ML review 2.8 |
| `gates_detection.miss_table` | `test_miss_table_and_fp_table` (every miss with `nearest_family = kernels`, `axis` the resemblance axis, `axis_of_largest` its complement, the signed differences; member 5 on `amount` with `identity` largest; the mask from `gates/gj.json` when present) | the same test (member 6 `at floor, not a miss`) | K3 move 18; N3 Sec. 1 RQ2; al-Kindi 2.7, 2.8 |
| `gates_detection.head_drop_template` | `test_head_drop_template_and_cli_exit_codes` | (exit 2 on a missing `cells.csv`; `not run: no selection`) | al-Farabi M5 |
| `synth_detection` | `test_truth_reproduces_the_extract_on_a_member_cell` (every column, every row, to 1e-9 on member 1's rep 0), `test_corpus_specs_layout_off_and_on`, `test_written_corpus_classes_and_boundaries` | `test_cli_refuses_member_9` (`ValueError`, exit 3) | SPEC_DETECTION 4 |

## 3. Test results

RESULTS_PLACEHOLDER

## 4. The synthetic corpora the tests run on

`tests/_det_common.py` builds three corpora once per pytest session (about ten minutes of wall
time on the build machine, whose load average was above 500 during this build):

- `main`: the reduced corpus of SPEC_DETECTION 4.5 (`--kernels gemm,floyd,gibbs,histogram,fft,lexer
  --reps 3 --idle 4 --members all --n-pairs 120`, `--write-classes --write-boundaries`), with member
  8 at 10,000 pages instead of 40,000 (`--member8-k0 10000`; a 40,000-page cell takes four and a
  half minutes to synthesize and 95 MB on disk; at 10,000 pages the band [5,000, 20,000] is still
  empty of benign workloads, which is the answer member 8 carries).
- `on`: `--kernels gemm,floyd,fft,histogram,lexer,rmat_gen --members 1,2,3,4 --reps 3 --idle 4
  --n-pairs 60 --order-confound on --campaign-labels confounded --idle-campaigns 2 --drift-level`:
  the audible cases (position by member block, campaign by archetype, two idle floors, a level
  drift).
- `tiny`: `--kernels gemm,gibbs --members 1,2 --reps 2 --idle 2 --n-pairs 40 --sandbox-n-pairs 32
  --stage2-fixture`, applied with the class file without `order_index`: the stage-2 classes, the
  cadence leak, the missing-order refusals.

The chain on each: `extract index`, `classes apply`, `extract all`, `preconditions
--c1-activity-min 0.001`, `inherit-selection --default W8_H4`, features for the five rungs at
`W8_H4`, `gk0`, `admissibility`, `gk0-sandbox-template`, `gk0-cells`, `gn`. The forests in the tests
use `n_estimators = 20` (the module default stays 300) and nulls of 4 to 10 permutations with
`n_perm_required` lowered so that the verdict verb is exercised; the smoke-run verb (`not run: N
permutations < 500`) is tested separately.

Known answers observed on `main` (`content`, LOWO, 20 trees, 6 permutations): members 1, 2 and 4
at 3/3, member 3 at 0/3 (its scores 0.80 to 0.90 against a threshold of 0.90 set by gibbs's cells,
the spin kernel; with three reps and six kernels the 95th percentile of the benign in-fold scores
is their maximum), members 5, 7 and 8 at 0/3, member 6 at floor and out of the denominator (3
cells), realized FPR 0.14 (three gibbs cells), AUC 0.82, the null on tpr05 PASS at rank 6 of 6.
Level 2 on `content`: A recall 1.0 (PASS at rank 266 of 280, exhaustive), B `one member, no
held-out test`, C recall 0.83 over the six above-floor cells. G-LM: members 1, 2, 4 survive level
matching, 3, 5, 7 not detected, 6 at floor, 8 empty band. The miss table: member 5 nearest gemm
on `amount` with `identity` the largest difference; members 7 and 8 nearest gemm (spmm and nbody
are not in the reduced corpus). The known answers of SPEC_DETECTION 4.4 are for the full corpus
and were not run (the eight cells of member 8 alone would take half an hour to synthesize).

## 5. Deviations from SPEC_DETECTION.md and why

Every deviation below is a review item marked "must change before build" (implemented in place
of the SPEC line) or a cost or collision found in the build.

1. `ROW_UNIT = "cell"` (ML review 2.1): `make_detection_labels` collapses the window rows to one
   row per cell (the nanmean over its windows, `models.cell_vectors`' rule); every fold, threshold,
   null, L1 run, level and ladder runs on cell rows; `cell_scores` is the identity and
   `params.score_aggregation` reads `not applicable: row unit is the cell`; `ROW_UNIT = "window"`
   stays as the declared alternative and refuses the out-of-bag threshold.
2. `THRESHOLD_SOURCE = "inner_lowo"` (al-Kindi 2.1), with `"inner_group_kfold"` (`INNER_K = 4`)
   and `"oob"` labelled `optimistic: sibling reps in bag` in `params.threshold_source_label`. The
   same source serves the observed run and every permutation.
3. `TRAIN_ON_AT_FLOOR = False` (ML review 2.5; al-Farabi M2 asked for the constant and the CLI
   flag `--train-on-at-floor`, and named the ML seat as the one to decide the default; the ML
   seat's must-change sets it to False): positive cells at floor, at harness floor or `control`
   leave every training set in the observed run and under every permutation; the fold record
   carries `n_train_dropped_at_floor`. The alias `AT_FLOOR_POSITIVES_IN_TRAINING` is the same value.
4. The denominator rule (ML review 2.2; al-Farabi M3): `control` counts as at floor in the
   observed run and under every permutation (`NULL_DENOMINATOR_RULE = "same_as_observed"`); a
   pending G-K0 makes every true-positive field `not run: gk0 not run`; a missing
   `gk0_cells.csv` is the written refusal `not run: gates/detection/gk0_cells.csv missing (run
   gk0-cells)`.
5. A cell without a finite score (al-Farabi M4): `score_status = not applicable: no finite
   window`, `in_denominator = false`, listed in `excluded_no_score` with `n_without_score`.
6. B1-G3 and the L1 row (ML review 2.4): the one-feature model at the operating point (direction
   by the sign of the median difference, threshold the `1 - fpr` quantile of the benign training
   values); `scores.json["l1"]` with the best feature per fold by training AUC and its own null
   from the same subsets; the headline is the full-feature forest (`score_source = "full"`), the
   quarantined features and the re-run stand beside it in `with_quarantine`;
   `--conjunction-members` is the empty-quarantine corpus switch (the switch is built and tested at
   the preset level; a corpus with it was not run for cost).
7. The one-class null (al-Kindi 2.2; ML review 2.7): `run_one_class(..., n_perm)` and
   `--null-perm` on the CLI; `--null-splits` defaults to `lowo,loco`.
8. Level 2's null is per letter (al-Kindi 2.3); at-floor cells leave the level-2 and level-3
   denominators and, unless `train_on_at_floor`, their training sets (al-Kindi 2.4); level 3
   records `null_unit = "cell"` with the reason.
9. `det_c1_rule` applies to every class alike (al-Kindi 2.5): under `report` the lexer of the
   synthetic corpus (a `benign_kernel` at floor) stays admissible with its G-K0 verdict, so the
   reduced corpus has 6 + 1 + 8 LOWO folds, not the 5 + 1 + 8 that SPEC 4.5 counted with the
   encoding paper's exclusion.
10. The head drop for every workload key (al-Kindi 2.6; al-Farabi M5): `classes apply` extends an
    existing `inputs/head_drop.csv` with zero rows; `gates_detection head-drop-template` writes
    one row per workload key of `cell_classes.csv`.
11. The miss table (al-Kindi 2.7, 2.8): identity is `J - J_null` (`--identity excess|raw`); one
    per-pair mask for every class from `gates/gj.json` (`plane_mask = "gj_mask_K, every class"`),
    unmasked and said so when absent; `axis` is the resemblance axis, `axis_of_largest` the
    residual one, `d_amount` and `d_identity` printed.
12. The alias falsifier (al-Kindi 2.9): workload fixed effects (`unit = "within_workload"`); the
    `cadence_as_class` row (the one-feature model on `dt_est_s` under LOWO with the workload-level
    null, verdict `cadence audible`); the per-class medians in `params`. The SPEC's pass case
    "every feature `ALIAS_STAYS` (dt is constant)" cannot be produced by a regression on a
    constant regressor (the epoch-1 form writes a not-run for it too); the row reads `not
    applicable: regressor has no spread`, and the test varies `dt_est_s` in the sidecars to
    produce `ALIAS_STAYS` and `ALIAS_MOVES`.
13. The drift regression (al-Kindi 2.10): the disclosure row uses workload fixed effects with the
    null shuffling `order_index` within each workload; the plain regression stays as a second row
    labelled `confounded with workload order (blocked)`. `DRIFT_NULL_PERCENTILE = 95.0` (M7).
14. The order test (al-Kindi 2.11; al-Farabi M12): two rows per (rung, class), `within_workload`
    (the default; the null permutes the half labels within each workload) and `within_class` (the
    ML-literal row; the workload-level null when every workload lies inside one half, else the
    cell-level one, recorded in `order_null_unit`); `ORDER_TEST_SCOPE = "within_class"` as a
    constant and `--scope` flag, with `campaign` adding one row over every classed cell.
15. G-OP counts the threshold setters per fold and rolls up as the worst fold (ML review 2.3):
    `n_folds_unsupported`, the union as a disclosure, `GOP_SETTER_RULE = "ge"` (M7); the CLI's
    `--split all` runs it on lowo, loco and lofo.
16. The leak probes (ML review 2.6): `gates_detection leak-probe` on `n_pairs`, `dt_est_s`,
    `frac_above_band`, `K_med` with the vocabulary `LEAK_AUDIBLE` / `LEAK_NOT_AUDIBLE`; the
    corpus switch `--sandbox-n-pairs`; the test that every feature name of every npz is in
    `series.feature_names` and none of `excluded_by_declaration` is a feature.
17. The rep-identity disclosure (ML review 2.8): `gv_two_class_members.csv` with `L0_member`,
    `L0_ratio` and `REPS_NEAR_IDENTICAL` below `REPS_IDENTICAL_RATIO = 0.1`.
18. The campaign token (al-Farabi M1): `stage1` for every sandbox and external id unless
    `--campaign-label` is given; the raw launch label is never copied; the validator warns
    `campaign token defaulted to stage1 for <n> rows`.
19. The ladder cap (al-Farabi M6): a prefix is capped at the extract's `_n_rows`, so the 600 s
    prefix's feature file equals the headline's on every cell (asserted array by array rather than
    by file hash, because `np.savez` output is not byte-stable across metadata);
    `LADDER_HEAD_DROP_RULE = "from_pair1_only"` in `ladder.json`.
20. The inline thresholds of M7 are constants written into params: `GCAL_SPREAD_RULE`,
    `DRIFT_NULL_PERCENTILE`, `GV_NOTE_FRACTION`, `LEVEL_WORKLOAD_AGG`, `GOP_SETTER_RULE`.
21. G-K0's edge (al-Farabi M8): recomputed by `gates/gk0.json`'s own `idle_pool` and
    `idle_percentile`; a mismatch is the written refusal on every cell and in `gk0.json`.
22. Unassigned cells (al-Farabi M9): listed in `admissibility.csv` with `admissible = false` and
    the reason `unassigned: no class row in inputs/classes.csv`.
23. Refusal addresses (al-Farabi M10): `inherit-selection` writes
    `gates/detection/inherit_selection.json`; the one-class guard is `usage: a second one-class
    model needs --secondary` with exit code 3 (outside the refusal vocabulary and SPEC 7.1's
    codes); `synth_detection` refuses a bad member index with exit 3 likewise.
24. `apply` refuses stray extract directories (al-Farabi M11).
25. The exhaustive rule honours `n_perm`: a run whose `n_perm` is below `comb(S + B, S)` (or
    below the count of letter permutations at level 2) draws `n_perm` subsets and is
    non-exhaustive, so a smoke run at 20 permutations reads `not run: 20 permutations < 500` as
    section 7 item 32 says; with the default 500 every count below 500 is enumerated as CR3 2.1
    says.
26. The stage-2 harness-idle fixture's test label is `harness_floor`, not `harness_idle` (SPEC
    4.3): `schema.IDLE_MARKERS_DEFAULT` marks any test label containing `idle` as the idle role and
    `extract index` would then give those cells the id `idle__rep<rr>__synth`, colliding with the
    idle cells' ids.
27. `--member8-k0` (default 40,000, the SPEC's value) exists so the tests can use 10,000 pages.
28. The known answers of SPEC 4.4 marked "full" were not run in the tests (cost); the tests assert
    the answers that hold on the reduced corpus and the file shapes elsewhere (section 4).
29. Test file names: `test_detection_*.py` (the task's rule) instead of the names in SPEC 1.1.
30. The one-class secondary model `gmm` uses `covariance_type = "diag"` (the ML review's
    for-the-author item 6: a full covariance is singular on about 100 rows of d = 60).

## 6. For the author

Each item is a parameter whose value is written into the result files' `params`; the default runs
unless you say otherwise before the data.

1. `THRESHOLD_SOURCE = "inner_lowo"` (B extra fits per fold; on cell rows this is seconds).
   `inner_group_kfold` with `INNER_K = 4` is the cheap variant; `oob` is labelled optimistic.
2. `TRAIN_ON_AT_FLOOR = False`: at-floor positives leave every training set. Pass
   `--train-on-at-floor true` to every split, level and null if you want them kept.
3. `ROW_UNIT = "cell"` with the cell vector `nanmean_over_windows`; the mean-plus-quantiles vector
   of the ML review's item 2 is not built.
4. `NULL_VERDICT_STATISTIC = "tpr05"`; AUC's verdict is printed beside it in every `null` block.
5. `GOP_CELLS = "threshold_setters"` counted per fold with the worst-fold roll-up (ML review 2.3);
   `--cells-rule realized_fps` is al-Farabi's recommendation. Both counts are always written.
6. `ONE_CLASS_MODEL = "isolation_forest"`; `gmm` (diagonal covariance) and `ocsvm` run only with
   `--secondary`.
7. `GK0_VERDICT_QUANTITIES = (K_med, K_q90, frac_above_band)`; `l0_med_above` and `J_consec_above`
   are disclosed, not gated. `GK0_ENVELOPE_PERCENTILE = 95`.
8. `LEVEL_QUANTITY = "median_K"` labelled `median K, stage 1`; `per_iteration_K_sum` runs from
   `inputs/iteration_boundaries.csv` when you supply it.
9. The order test's default half rule is `within_workload` (al-Kindi 2.11); the `within_class` row
   is written beside it. On stage 1 the within-class row of the sandbox class reads audible by
   construction (member identity and position coincide, P3 0a).
10. `ORDER_TEST_SCOPE = "within_class"`; `--scope campaign` adds the campaign-wide row, void by
    construction on stage 1.
11. The realized order: 168 per-cell rows with `order_index` in `inputs/classes.csv`; without them
    the order test, the drift rows, early-against-late idle and the letter sequence read
    `not run: order_index missing`.
12. `inputs/gk0_source_sandbox.csv`: one numbered line per member, yours; the template writes
    `unstated`.
13. `--campaign-label TEXT` for `classes apply`; without it every sandbox id carries `stage1`.
14. The head drop per workload key (`inputs/head_drop.csv`): 0 unless you declare a value before
    the data; run `gates_detection head-drop-template` after `classes apply`.
15. `REPS_IDENTICAL_RATIO = 0.1` (ML review 2.8) and the leak-probe vocabulary are new; confirm.
16. `HARNESS_COMPARABLE_TOL = 0.10`, `GFP_FLAG_FRACTION = 0.5`, `GLM_BAND_FACTOR = 2.0`,
    `GLM_MODEL = "retrain"`, `GLM_VANISH_RULE = "tpr_le_fpr"`, `ALIAS_TOP_K = 10`, `ALIAS_R2 = 0.5`,
    `MISS_DISTANCE = "standardized_euclidean_to_centroid"`, `MISS_IDENTITY = "excess"`, `GM_ALPHA =
    0.05`, `GM_N_SEEDS = 5`, `LEVEL3_SPLIT = "rep_index"`, `LADDER_DT_S = 0.644`, `LADDER_NORM =
    "prefix"`: the SPEC's defaults, unchanged.
17. `--null-splits lowo,loco` by default (ML review 2.7); on cell rows the LOCO null under
    `--loco-mode cell` is 168 x 500 fits of a forest on about 160 rows.
18. On the reduced synthetic corpus the honest one-class threshold (`inner_lowo`) flags no
    sandbox cell: with seven benign workloads every held-out benign workload is novel to the model
    and the 95th percentile of their scores sits above the members'. That is the reading's bounded
    strength (ML 1.3), not a bug; expect it to relax with thirteen benign workloads.
19. The full-corpus known answers of SPEC 4.4 remain to be checked by the smoke run of SPEC 6.2
    (`synth_detection corpus --root ... --write-classes` at the defaults, then the driver).
