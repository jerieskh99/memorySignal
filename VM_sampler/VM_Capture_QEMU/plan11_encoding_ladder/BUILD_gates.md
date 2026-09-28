# BUILD_gates.md: builder 2's report (the gate library, nulls, splits and models)

Written 2026-09-16. Everything below was built and tested on this machine against synthetic data
only. No server was touched, no path under `/mnt/nfs` or `/project` appears in the code, and the
sandbox family is neither read nor named anywhere in these modules or tests. Nothing was committed
to git, and no file outside `plan11_encoding_ladder/` was edited.

## 1. Files written

All paths are under
`/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/`.

| File | What it holds |
|---|---|
| `verdicts.py` | The verdict vocabulary of SPEC 3.0 as string constants, the four parameterised forms, `Refusal`, `is_refusal`. Three strings were added, each from a review item marked "must change before build": `GC_ALIASED_BY_DESIGN` (al-Kindi 2), `GORD_ORDER_BLIND_BY_CONSTRUCTION` (al-Farabi 2.11 b), `GDEC_NO_BOUNDARY` (al-Kindi 4 b). |
| `series.py` | Per-rung per-snapshot series from an extract, head drop, level normalization, the window rule, the eight shape features (the verbatim copy of `b1_features.features` plus a vectorized form asserted equal to it), the window feature vectors, the feature files at every grid point, per-cell headline readings, and the shared file helpers every builder 2 module uses (cells.csv, extract.csv, sidecar.json, result files with `schema` / `params` / `citation`, input hashes, admissibility). CLI: `head-drop-template`, `features`. |
| `nulls.py` | Phase-randomized surrogates, unit-level label shuffles, order shuffles, the null summary, the four fixed seeds. |
| `splits.py` | `fold_within_trace`, `fold_loro`, `fold_loko` copied from `plan08_b1/b1_splits.py` with `workload -> kernel` and `family -> archetype`, `_assert_grouped`, `folds_for`, the `rep_index` LORO mode. |
| `models.py` | The forest pipeline, B1's L1 tree, unit aggregation, unit scores, the majority baseline (B1-G6), the B1-G1 verdict, the B1-G3 quarantine, the per-fold G-DIM reduction, the split stage `run_split_stage`, clustering with ARI and NMI against the unit-level null. CLI: `splits`, `cluster`. |
| `gates_precondition.py` | C1 to C8 as re-mapped, the `failed/` count, G-K0 with its source template, G-F with the admissibility template and the three floors. CLI: `preconditions`, `gk0-template`, `idle-admissibility-template`, `gk0`, `gf`. |
| `gates_calibration.py` | The pass table template and loader, G-P, G-C (with al-Kindi's two corrections), the alias falsifier with al-Kindi's operational definition of "separates a level-matched pair". CLI: `pass-table`, `gc`, `gp`, `alias`. |
| `gates_temporal.py` | G1 with its surrogate null (the verbatim copy of `plan03_metric_kernel.stationarity_per_window` plus a vectorized batch form asserted equal), G2 in seconds at both bracket ends and in pair units, G3 as the per-kernel flag with the order-shuffle null, G4, G5 reported, G-ORD, the roll-up with al-Farabi's `rollup_kernel_refusals`, the selection rule re-pointed from `plan03_aggregate._pick_winner`, `grid_complete.json`, and the C7 refresh. CLI: `grid`, `g3`, `gord`, `select`. |
| `gates_readings.py` | G-J with the two masks, G-DEC with al-Kindi's boundary, control and no-boundary corrections. CLI: `gj`, `gdec`. |
| `gates_comparison.py` | G-L (both parts), G-N, G-X, G-DIM, G-M; re-exports the B1 helpers that the split stage applies. CLI: `gl`, `gn`, `gx`, `gdim`, `gm`. |
| `variance.py` | G-V. CLI: the module itself. |
| `_schema_compat.py` | A fallback re-declaring the SPEC constants that builder 2 reads, used only when builder 1's `schema.py` is absent (it was absent when this build started; it is present now and is the module actually imported). See deviation 1. |
| `tests/_synth_b2.py` | Builder 2's minimal in-test generator at the extract level (SPEC section 5's set dynamics and content models, writing `extract.csv`, `sidecar.json`, `cells.csv` directly). See deviation 2. |
| `tests/_b2_common.py` | Shared test set-up (path insertion, corpus helpers). |
| `tests/test_gates_verdicts.py`, `test_gates_series.py`, `test_gates_nulls.py`, `test_gates_splits.py`, `test_gates_models.py`, `test_gates_precondition.py`, `test_gates_calibration.py`, `test_gates_temporal.py`, `test_gates_readings.py`, `test_gates_comparison.py`, `test_gates_variance.py` | One test module per owned module; for every gate one case that must pass and one that must refuse. |
| `tests/test_gates_chain.py` | The gate chain on builder 1's own pipeline (`synth.py corpus` then `extract.py all`), run when both modules are present and skipped otherwise. |
| `BUILD_gates.md` | This report. |

Every result file written by these modules carries `schema`, `params` and `citation`. A CSV-only
result gets a sibling `<name>.params.json` with the same three keys (the SPEC lists JSON companions
only for some gates; the sibling file is how the "every threshold is written into every result
file's params block" rule is honoured for the rest). Every `params` block includes `inputs_sha256`,
a map from each input file read to its sha256 (al-Farabi 2.8).

How to run the tests: from `plan11_encoding_ladder/`, `python3 -m pytest -q tests/`. The tests are
pytest-style functions and need `pytest` (`python3 -m pip install --user pytest` when absent); they
do not run under `python3 -m unittest`. The whole builder 2 suite takes about nine minutes on this
machine, most of it in the two B1-G1 cases that run the full 500 unit-level permutations. Forests in
the tests use `n_estimators = 10`; the module default is the SPEC's 300 and the value used is
written into `params`.

## 2. Gate by gate

Each row names the function that implements the gate, the test that must pass, the test that must
refuse, and the citation carried in the function's docstring. All tests are under `tests/` and all
passed on 2026-09-16 (see section 5 for the totals).

| Gate | Function | Test, must pass | Test, must refuse | Citation in the docstring |
|---|---|---|---|---|
| C1 | `gates_precondition.gate_preconditions` | `test_c1_pass_refuse_and_idle_not_applicable`: a kernel cell with `K0 = 6000` passes; an idle cell reads `not applicable: control (C1 re-mapped)` and is never refused | the same test: `K0 = 100` fails and is listed in `excluded_cells` | P2 Sec. V 5.1 Plan 02 and 5.2 preconditions; CR 2.1 item 1; `validate_campaign.py` lines 17-28, 39 |
| C2, C3 | `gate_preconditions` | `test_c2_c3_c6_pass_and_refuse`: 120 pairs | the same test: 4 pairs fails C2 (and C3, informational) | as above; `plan02_validate_session.py` D-25 for C3 |
| C4, C5, C8 | `gate_preconditions` | the fixed strings are asserted in `test_c1_pass_refuse_and_idle_not_applicable` | (fixed strings by definition) | SPEC 3.3.1 |
| C6 | `gate_preconditions` | `test_c2_c3_c6_pass_and_refuse`: a clean sidecar passes; a two-gap cell passes with `n_seq_gaps 2` in `C6_reason` | the same test: an extractor status `refused: seq not monotone at row 77` fails C6; `test_c6_header_reference_is_modal_and_ties_refuse`: a header differing from the modal header fails, a tie refuses every cell with `refused: header mismatch, no majority` | SPEC 3.3.1; al-Farabi 2.11 (a) |
| C7 | `gate_preconditions.c7_verdict`, refreshed by `gates_temporal.select` | `test_gates_chain`: after `select --rung apf` the column reads `pass` or `fail` | `pending: filled by the temporal gates (move 6)` before any selection (asserted) | SPEC 3.3.1 |
| The `failed/` count | `gates_precondition.failed_verdict` | `test_failed_count_verdicts`: count 0 passes; count null with `--assume-failed-zero --assume-reason` passes with `failed_source = "declared zero: ..."` | the same test: count 2 gives `refused: failed count 2 > 0, seq axis uncorrected`; null without the flag gives `refused: failed count not recorded`; `inputs/failed_counts.csv` overrides the sidecar; `test_failed_count_excludes_the_cell_from_the_persist_split_only` shows the cell absent from the persist predictions and present in the APF ones | P2 Sec. V 5.2 preconditions (*min*); CR 2.1 item 2; K2 move 2; AA A5; al-Farabi 2.4, 2.9 (a) |
| G-K0 | `gates_precondition.gate_gk0` | `test_gk0_relabels_the_lexer_and_not_histogram`: histogram reads `above floor` | the same test: the lexer preset reads `IDLE, measured` with `archetype_measured = IDLE`; with no idle cell every kernel row reads `not run: no admissible idle cell` | P2 Sec. V 5.2 G-K0; CR 2.2 item 20; K2 move 4 |
| G-F (i) | `gates_precondition.gate_gf` | `test_gf_part1_inseparable_vs_void_and_part2_pass_vs_at_floor`: six idle cells from one preset read `inseparable at floor` | the same test: idle reps with a per-rep floor jitter read `void: idle reps separable under this rung` (and the reported form under `part1_consequence = "report"`); the admissibility record missing or no idle cell gives the two `not run` strings | P2 Sec. V 5.2 G-F and the tripwire clause; CR 2.2 item 21, 2.1 item 15; K2 moves 4 and 14 |
| G-F (ii) | `gate_gf` | the same test: gemm against the idle envelope passes | the same test: the lexer preset reads `at floor in this lead` | as above |
| G-C, apf / persist / wapf | `gates_calibration.gate_gc` | `test_gc_apf_persist_wapf_pass_on_the_corpus_and_stat_a_below_two`: the corpus gemm passes in every rep with the observed ratio between 1.5 and 2 | `test_gc_refuses_disconnected_lead_and_names_the_aliased_regime`: no pulse with gemm below the full-footprint band, and mixed reps, give `disconnected lead`; no pulse at full footprint with J near one gives the not-applicable string of al-Kindi item 2 | P2 Sec. V 5.2 G-C; CR 2.2 item 22; K2 move 3; al-Kindi items 1 and 2 |
| G-C, content | `gate_gc` | `test_gc_content_ordering_pass_and_break_order`: the corpus orderings hold under both pairings | the same test: `break_order` gives `disconnected lead` on persistent and on all pages | as above |
| G-C, combined | `gate_gc` | the apf test: `pass` when the four rungs pass | the disconnected test: `disconnected lead` when any rung is | as above |
| G-P | `gates_calibration.gp_cell`, `gate_gp` | `test_gp_cell_verdicts`: `gemm, 10` at 120 pairs is `resolvable`; `40` is `marginal`; `100` is `aliased by design at this size`; nbody at 6,147 is aliased with `T_pairs` about 0.15 | the same test: blank is `undeclared`; `passes = 3` is `rhythm under-sampled`; `passes = 100` is `pass aliased`; an inferred row carries ` (INFERRED)` | P2 Sec. V 5.2 G-P; CR 2.2 item 23; K2 Sec. 2 |
| The alias falsifier | `gates_calibration.alias_falsifier`, `run_alias`, `separating_features` | `test_alias_falsifier_moves_and_stays`: a feature linear in `dt` reads `moves with the interval` | the same test: a feature independent of `dt` reads `does not move with the interval`; fewer than three cells is `not run` | P2 Sec. 2 falsifier (2); CR 2.2 item 23; al-Kindi item 5 |
| G1 | `gates_temporal.g1_cell`, `g1_kernel`, `gate_grid` | `test_g1_pass_trend_present_and_fail`: a stationary cell passes against its surrogate null | the same test: `trend = 2.0` reads `trend present`; a plateau (no linear drift) reads `fail`; an equal-halves step reads `trend present` (see deviation 8) | P2 Sec. V 5.1 Plan 03 G1 as amended; CR 2.1 item 3 |
| G2 | `gates_temporal.g2_kernel` | `test_g2_cases_in_seconds_and_pairs`: `passes = 100`, `W = 32` passes at both dt and in pairs; `passes = 300`, `W = 8` passes at both | the same test: `passes = 6147` is `not applicable, rhythm above Nyquist`; `passes = 240`, `W = 8` is `undetermined by the interval calibration`; undeclared is `undeclared` | CR 2.1 item 4; K2 Sec. 2 rung 0 (b); al-Kindi item 7 |
| G3 flag | `gates_temporal.g3_cell`, `gate_g3` | `test_g3_flag_present_absent_and_the_quefrency_floor`: the gemm pulse at 120 and at 240 pairs reads `rhythm flag: present` above the order-shuffle null; `test_g3_file_and_kernel_flag`: the kernel flag from the cell count | the same test: no pulse reads `rhythm flag: absent`; a period-8 sinusoid below the floor reads absent; the phase-randomized null ties the observed SNR to machine precision and can never flag (al-Kindi item 3, asserted) | P2 Sec. V 5.1 and Sec. 6 item 7 option (a); CR 2.1 item 5; al-Kindi item 3 |
| G4 | `gates_temporal.g4_pass` | `test_g4_and_g5`: `H = W/4` and `H = W/2` pass | the same test: `H = W` fails, the whole-cell point fails by construction | CR 2.1 item 6; `plan03_aggregate.py g4_pass` |
| G5 | `gate_grid` | `test_g4_and_g5`: 13 windows at (8, 4) pass with the non-overlapping count reported | the same test: no window at W = 64 on 59 rows fails; never part of the selection rule | CR 2.1 item 7 |
| G-ORD | `gates_temporal.gate_gord` | `test_gord_resolution_vs_order_blind`: two archetypes with one marginal and different autocorrelation read `resolution` at every W, and the value is copied into every grid point sharing the W | the same test: the level-only corpus reads `order-blind`; the whole-cell point reads `order-blind (by construction)` without running | P2 Sec. V 5.1 G-ORD; CR 2.2 item 27; al-Farabi 2.11 (b) |
| Roll-up and selection | `gates_temporal.select` | `test_select_rule_and_rollup_of_kernel_refusals`: with nbody above Nyquist not applicable, the smallest W passing G1, G2, G4 is 32 and the hop ratio nearest 0.5 is taken (`W32_H16`, `passes_acceptance = true`, "3 of 3"); a trending kernel is not applicable for G1 and `n_kernels_na_G1 = 1` | the same test: under `rollup_kernel_refusals = "blocks"` no point passes, the selection is `best-feasible` with `passes_acceptance = false` and Table 5 prints `selected: best-feasible`; `test_gc_verdict_propagates_into_the_grid_refusal_column`: a disconnected rung carries `disconnected lead` in every grid row's `refusal` | P2 Sec. V (the binding condition); `plan03_aggregate._pick_winner`; SPEC 3.5.7; al-Farabi 2.1, 2.2, 2.5, 2.11 (c) |
| G-J | `gates_readings.gate_gj` | `test_gj_interpretable_floor_overlap_and_floor_unmeasured`: gemm pairs read `interpretable` with `mask_K` and `mask_persist` all true | the same test: lexer pairs read `floor overlap` with `mask_K` all false; no idle cell reads `floor unmeasured` on every cell | P2 Sec. V 5.2 G-J and Sec. 6 item 6; CR 2.2 item 31; K2 Sec. 2 rung 1 (a); al-Kindi item 9 |
| G-DEC | `gates_readings.gate_gdec` | `test_gdec_decay_and_no_decay_control`: the floyd preset with 12 declared passes reads `decay` under both boundary sources; gibbs reads `no slope (k of n reps with a slope)` | `test_gdec_refusals`: undeclared reads `decay not resolved`; K decaying inside the pass reads `no decay beyond breadth`; idle cells with a negative l0 trend read `no decay beyond floor or host`; no pulse reads `decay not resolved: no pass boundary`; no idle cell appends ` (floor unmeasured)` | P2 Sec. V 5.2 G-DEC; CR 2.2 item 34; K2 Sec. 2 rung 2 (e); al-Kindi item 4 |
| B1-G1 | `models.b1_g1_verdict` inside `run_split_stage` | `test_b1_g1_passes_on_an_archetype_consistent_corpus_and_refuses_on_one_preset`: LOKO/archetype on a corpus whose archetypes share a preset passes against 500 unit-level shuffles with the rank reported | the same test: the one-preset corpus reads `near_unfalsifiable` and the row is written to `gates/excluded_rows.csv`; fewer than 500 permutations reads `not run: N permutations < 500` (`test_b1_g6_...`, `test_b1_helpers_are_exported`) | P2 Sec. V 5.1 Plan 08; CR 2.1 item 9; al-Farabi 2.7 |
| B1-G3 | `models.quarantine_l1` inside `run_split_stage` | `test_b1_g3_quarantines_the_level_feature_on_raw_apf_of_a_level_only_corpus`: the standard corpus has an empty quarantine and `with_quarantine = null` | the same test: the level-only corpus on the raw APF features quarantines `apf.k_over_n.mean` and the re-run is kept beside it | CR 2.1 item 10 |
| B1-G6 | `models.majority_baseline` | `test_b1_g6_majority_is_half_under_loko_on_twelve_kernels`: 0.5 under LOKO on the 12-kernel corpus | (a baseline, never refuses) | CR 2.1 item 11 |
| G-L (i) | `gates_comparison.gate_gl` | `test_gl_part1_pass_and_level_only_and_part2_random_vs_shot`: a normalized LOKO score above its null p95 passes | the same test: below the p95 reads `level only`; a rung without a selection reads `not run: no selection for <rung>` | P2 Sec. V 5.2 G-L; CR 2.2 item 24 |
| G-L (ii) | `gate_gl`, `gl_part2_regression` | the same test: the `cv_case = random` corpus passes with r2 at most 0.5 | the same test: the `cv_case = shot` corpus reads `refused: shot noise explains CV` with r2 above 0.5 | as above |
| G-N | `gates_comparison.gate_gn`, `gn_status` | `test_gn_statuses`: 6 / 3 / 2 / 1 / 0 kernels read `headline`, `headline`, `one training kernel per fold`, `structural novelty`, `no kernel row`; after a G-K0 relabel the IDLE row gains its kernel | (a labelling, never refuses) | P2 Sec. V 5.2 G-N; CR 2.2 item 25 |
| G-X | `gates_comparison.gate_gx`, `gx_confound` | `test_gx_pooling_stands_vs_leak_with_total_confound`: three campaigns assigned round-robin read `pooling stands` and `confound: none` | the same test: one archetype's kernels all in `01c`, another's all in `01c1`, with a per-campaign jitter offset, read `campaign predictable`, `confound: total` and the headline mark `refused: campaign leak with total confound`; a single campaign label reads `not applicable: one campaign label` | P2 Sec. V 5.2 G-X; CR 2.2 item 26; K2 Sec. 4 item 3 |
| G-DIM | `gates_comparison.gate_gdim`; the per-fold reduction in `models.fit_predict_units` | `test_gdim_full_vs_declared_reduction_and_matched_row`: 96 cells with d = 60 read `full vector`; the `combined (matched)` row is reduced to the strongest single rung's d by `train_importance` | the same test and `test_gdim_reduction_when_dimension_exceeds_training_cells`: 12 cells with d = 60 read `declared reduction` (reduced per fold to the training cell count) | P2 Sec. V 5.2 G-DIM; CR 2.2 item 32 |
| G-M | `gates_comparison.gm_compare`, `gate_gm` | `test_gm_beats_vs_difference_with_margin`: a perfect separator against chance reads `beats` (9 improving, 0 worsening, diff above the spread) | the same test: a difference inside the seed spread, or with fewer than six improving kernels, reads `difference with margin`; the end-to-end run writes the seed re-runs under `gates/gm_runs/` | P2 Sec. V 5.2 G-M; CR 2.2 item 33 |
| G-V | `variance.gate_gv`, `variance_levels` | `test_gv_estimable_vs_not_estimable`: the archetype-consistent corpus reads `estimable`; `test_variance_levels_arithmetic` checks L0, L2, L3 on a hand-made matrix | the same test: the one-preset corpus reads `LOKO not estimable` (every informative feature with L0 above L3) | P2 Sec. V 5.2 G-V; CR 2.3 item 35 |
| Clustering | `models.run_clustering`, `cluster_cells`, `ari_nmi` | `test_clustering_exceeds_null_on_archetype_consistent_and_not_on_one_preset`: k = 4, KMeans primary, ARI above the unit-level null | the same test: the one-preset corpus does not exceed; no selection reads `not run: no selection for <rung>` | P2 Sec. V 'Models'; SPEC 4.4 |
| The splits | `splits.folds_for` and the three fold functions | `test_folds_equal_original_b1_splits` (equality with `b1_splits.py`), `test_no_cell_straddles_train_and_test` | (structural; `_assert_grouped` raises on a leak) | P2 Sec. V 'The splits'; SPEC 4.1 |
| The shape features and windows | `series.shape_features`, `shape_features_windows`, `window_features`, `n_windows` | `test_shape_features_equal_original_b1_features` (equality with `b1_features.py`), `test_n_windows_matches_plan02_rule`, `test_grid_points_are_the_declared_thirteen` | (structural) | SPEC 3.1.3, 3.1.4 |
| The feature files | `series.build_features`, `build_all_grid` | `test_build_features_stores_predicted_archetype_only`: the file's bytes do not depend on whether G-K0 ran (al-Farabi 2.3); `test_g4_and_g5`: every grid point's file exists after `grid` (al-Farabi 2.2) | (structural) | SPEC 3.1.5; al-Farabi 2.2, 2.3 |
| The chain on builder 1's pipeline | every module | `test_gates_chain.test_chain_on_builder1_extracts` | (the verdicts above on the real extract interface) | (all of the above) |

## 3. Review corrections implemented ("must change before build")

From `SPEC_review_al_kindi.md`:

1. Item 1, G-C's threshold. `jump_predicted = 2.0` is recorded and `jump_detect_ratio = 1.5` is the
   event rule (`K_t >= 1.5 * reference`); the observed per-rep maximum ratio is written as `stat_a`.
   On the synthetic gemm the ratio lands at 1.96 to 1.98 in every rep, below 2 as the review computed.
2. Item 2, G-C's absence verdict. When no rep shows the event, the regime is read blind: if in
   every rep `K_median / 4096` lies in `[1/1.5, 1.5]` and `J_q50 >= 0.75`, the verdict is
   `not applicable: pulse aliased by design (full footprint lit every snapshot)` and the rung is
   neither passed nor voided; mixed reps stay `disconnected lead`. The constants
   `pulse_full_footprint_pages = 4096`, `regime_band_ratio = 1.5`, `regime_j_min = 0.75` are
   parameters and the docstring carries the INFERRED tag on the pulse shape.
3. Item 3, G3's null. The null is the order shuffle (`g3_null = "order_shuffle"`, alternative
   `"phase_randomize"` kept only so the test can show it ties the observed SNR to machine precision);
   the peak search stops at `len(c) // 2` so the mirror index is never returned. The G3 test cell has
   240 pairs.
4. Item 4, G-DEC. `"period"` anchors at `seq_first + phase_pairs` with `phase_pairs = 0`; a cell with
   fewer than two boundaries writes `decay not resolved: no pass boundary`; the control and the idle
   clause (e) are whole-cell slopes against phase-randomized surrogates when no boundary exists; the
   synthetic floyd preset has `pulse_extra = 2048` so the `k_jump` path is exercised, and the test
   runs both boundary sources.
5. Item 5, alias falsifier (b). A feature separates a level-matched pair when the two kernels'
   per-cell window means have disjoint ranges (no threshold), at APF's selected point on the
   normalized features; `gates_calibration.py alias` is idempotent (rows of a kind are replaced) and
   can be run again at the end of move 7 as the review asks.
6. Item 7, G2 in pair units. `coverage_pairs = W * passes / n_pairs` per cell, its median per kernel,
   and `G2_pairs` by the same floor 2.0 are in `temporal_per_kernel.csv` and `table5_grid.csv`; the two
   seconds columns stay as the bracket representation.
7. Item 9, the fused plane's mask. `gates/gj_mask/<cell_id>.npy` is a structured bool array with two
   fields, `mask_K` (the definition) and `mask_persist` (`n_persist > k_factor * the idle cells' median
   n_persist`); `gj.json` names `mask_K` as the default.

Items 6 and 8 (G-P at move 2; G-X per rung in moves 9 to 12) are driver-order items for builder 3;
`gates_calibration.py gp` needs only the pass table and the sidecars, and `gates_comparison.py gx`
takes `--rung`, so both are runnable as the review asks.

From `SPEC_review_al_farabi.md`:

1. Item 2.1, the roll-up. `rollup_kernel_refusals = "not_applicable"` (alternative `"blocks"`) is a
   parameter of `select` and is written into `selection.json` `params`; a kernel whose G1 is
   `trend present` is not applicable for G1, a kernel above Nyquist at a dt is not applicable for G2
   at that dt, the roll-up is per dt column and then combined by 3.5.2's rule, the verdicts stay in the
   kernel's own row of `table5_long.csv`, `table5_grid.csv` carries `n_kernels_na_G1`,
   `n_kernels_na_G2` (and, separately, `n_kernels_undeclared_G2`), the idle row is out of the
   roll-up, `gates_passed` is "k of m" over the applicable gates, and the split stage never routes a
   cell by a G1 verdict.
2. Item 2.2, feature artifacts at every grid point. `gates_temporal.py grid` builds the raw and
   normalized feature files of all 13 points (`series.build_all_grid`) before the verdicts;
   `grid_complete.json` lists the 13 CSVs and the 13 feature pairs (norm only for `combined`) with a
   `complete` flag the driver can refuse on.
3. Item 2.3. The feature files store the predicted archetype only; `run_split_stage`,
   `run_clustering`, `gate_gord`, `gate_gn` and `gate_gv` apply G-K0's relabelling at read time and
   write `gk0_applied` and `relabelled_kernels` into `params`. A test asserts the file's bytes do not
   change when G-K0 runs.
4. Item 2.4. A cell whose `failed_verdict` is a refusal is excluded from the persist, content and
   combined series (`series.admissible_cells`), kept in apf and wapf, listed in `preconditions.json`
   `excluded_cells_pair_rungs` and in every affected `params`. The must-refuse row of 5.3 is
   `test_failed_count_excludes_the_cell_from_the_persist_split_only`.
5. Item 2.5. The rung's G-C verdict is written into every later result file's `params`
   (`gc_verdict`) and into the `refusal` column of every `table5_grid.csv` row of that rung; the
   numbers stay in the files for builder 3 to mask.
6. Item 2.7. `near_unfalsifiable` is written as the verdict in `scores.json`, the score stays in
   `scores.json`, and the row goes to `gates/excluded_rows.csv`; builder 3 prints the string.
7. Item 2.8. `inputs_sha256` in every `params` block.
8. Item 2.9 (a) `failed_verdict = pass` with the new column `failed_source`; (b)
   `gates_comparison.py`, `variance.py` and `models.py cluster` write `not run: no selection for
   <rung>` and exit 0 when a rung has no selection; (c) a cell whose windows were all dropped is
   written to `predictions.csv` with `y_pred = "not run: no windows"`, counted in
   `n_cells_no_windows`, and excluded from accuracy.
9. Item 2.11 (a) the C6 header reference is the modal `header_sha256`, a tie refuses every cell;
   (b) `GORD = "order-blind (by construction)"` at the whole-cell point without running the shuffles;
   (c) `selected` reads `selected: best-feasible` when the acceptance rule failed.
10. Section 3 item 2 (a recommendation, not a must): `part1_consequence = "void" | "report"` is
    exposed on G-F with the SPEC's `"void"` as the default.

Items 2.6 and 2.10 are builder 3's (move order and the runbook); `gates_precondition.py gf
--all-rungs` reads every rung's selected point from `selection.json`, so the re-run can be placed
before `tables.py` as item 2.6 asks.

## 4. Deviations from SPEC.md, and why

1. `_schema_compat.py` exists. Builder 1's `schema.py` was not present when this build started, and
   the SPEC forbids builder 2 from writing it. Every builder 2 module does `from
   plan11_encoding_ladder import schema` first and falls back to `_schema_compat` only on
   ImportError. `schema.py` has since landed and is the module actually imported; the fallback is
   inert and re-declares the SPEC constants verbatim. It can be deleted once the author is satisfied
   nobody runs the package without `schema.py`.
2. The tests use builder 2's own generator, `tests/_synth_b2.py`, at the extract level. Builder 1's
   `synth.py` and `extract.py` were absent when the gates were built and tested, as the task
   anticipated. The helper follows SPEC section 5.1's set dynamics and content models and writes the
   58-column extract exactly as 2.2 defines it (persist side "t"), the sidecar of 2.3 and the
   cells.csv of 2.7; for the `double` and `decay` models it draws each page's `l1` and `hamming` from
   the exact first two moments of |a - b| and popcount(a XOR b) over unequal byte pairs rather than
   byte by byte. Builder 1's modules have since landed; `tests/test_gates_chain.py` runs the gate chain
   on their output, and the verdicts agree with the in-test generator's on the same presets. The
   helper carries six additions the SPEC assigns to `synth.py` flags or that specific tests need
   (`step` with a `step_mode`, `k_decay`, an `l0_trend` on the idle model, `floor_noise`, a slow
   sinusoidal modulation `k_sine_*`, and per-cell `failed_count` / status fields).
3. `make_forest` takes `n_estimators` (default the SPEC's 300) so that the tests and a smoke run are
   affordable; every `params` block records the value used. No other model parameter moved.
4. CSV-only results get a sibling `<name>.params.json` (see section 1). `selection.json` holds one
   top-level key per rung plus `schema`, `params` (per rung, with the input hashes) and `citation`.
5. G-C's must-refuse case of SPEC 5.3 (`--break-pulse -> disconnected lead`) does not hold as stated
   once al-Kindi's item 2 is implemented: a pulse-less gemm at its full footprint with J near one is
   exactly the aliased-by-design regime and reads the not-applicable string. The tests therefore
   refuse with `disconnected lead` on a pulse-less gemm below the band (`K0 = 1024`) and on mixed
   reps, and assert the not-applicable string on the full-footprint case.
6. G-DEC's boundary detector uses the same corrected detection ratio as G-C (`jump_detect_ratio =
   1.5`, `jump_predicted = 2.0` recorded), because the SPEC names it "the G-C detector" and the
   floor makes a 2.0 threshold miss the synthetic floyd pulse (ratio 1.93) for the same reason
   al-Kindi gave for gemm. Three further executable choices had to be made and are parameters: the
   fall must hold in at least `pass_frac = 0.5` of a cell's passes at the common phase; the phase is
   the one satisfied by the most reps, ties by the highest mean fall fraction; the control kernel's
   verdict uses the same `min_reps` standard as the exhibit and reads `no slope (k of n reps with a
   slope)`; the idle clause (e) fires on any idle cell by default (`idle_slope_rule = "any"`, the
   definition's words) with `"min_reps"` as the alternative.
7. G-J's per-cell `mask_verdict` needs a rule the definition does not give (the mask is per pair):
   `interpretable` when more than half the cell's pairs are interpretable under `mask_K`, else
   `floor overlap` (`gj_cell_verdict_rule = "majority_of_pairs"`).
8. G1's must-refuse `fail` case. An equal-halves step (`synth.py --step`) is a least-squares drift of
   about 1.7 population standard deviations, so under the SPEC's own trend rule it reads `trend
   present`, not `fail`. The test asserts that, and uses a plateau (`step_mode = "middle"`, no linear
   drift, pass fraction 0.57) for the `fail` case. When both the trend rule and the pass-fraction rule
   fire, `trend present` takes precedence (`g1_trend_precedence = "trend_first"`; CR 2.1 item 3 says a
   trend refuses and hands the cell to the whole-cell reading); the alternative `"pass_first"` is not
   implemented, only named here.
9. G3 and the quefrency floor. At 240 pairs the floor `n // 8 = 29` sits above the period 24, so the
   peak found is the rahmonic 48 (al-Kindi section 3 item 5); the test accepts a multiple of 24 and
   asserts 24 at 120 pairs. SPEC 5.3's "pulse_period = 8 -> absent, which documents the floor" does
   not hold for a pulse train: its rahmonics at 32 and 40 are above the floor and the flag reads
   `present` through them. The floor documents itself on a period-8 sinusoid, which reads `absent`,
   and that is what the test asserts. A further observation: the Plan 03 cepstrum is a harmonic-comb
   detector and does not flag a pure sinusoidal modulation at any period on this data.
10. G-ORD's must-pass case. SPEC 5.3's ord-case (two archetypes differing in pulse period only, 6
    against 12) reads `order-blind` under the eight shape features, correctly: the features are
    permutation-invariant within a window and an order shuffle preserves the pulse density per
    window. The resolution case in the test is two archetypes with one marginal and different
    autocorrelation (a slow sinusoid against the same values in random order), which reads
    `resolution` at every W. The level-only corpus reads `order-blind` as the SPEC says.
11. G-F (i) runs on the rung's level-normalized features, the ones the splits use
    (`part1_features = "norm"`). SPEC 5.3's must-refuse case (idle reps at `floor_F = 150 + 40 * rep`)
    is a level step that normalization removes, so it reads `inseparable at floor` under the
    normalized rung and `void` only under `part1_features = "raw"`; the test asserts both. The
    normalized must-refuse case is a per-rep floor jitter.
12. G-X's must-refuse case. A per-campaign `K0` offset (SPEC 5.3) is invisible after level
    normalization, so the test's confounded corpus carries a per-campaign jitter offset instead.
13. G-V: a constant feature (L0 = L3 = 0, the `duty` of a flat series) is not a counterexample to
    "L0 > L3 for every feature"; it is flagged in a `constant` column and counted in
    `n_features_constant`.
14. C1's threshold is a parameter (`c1_activity_min`, CLI `--c1-activity-min`, default the SPEC's
    0.02). See "For the author" item 1 for why it matters.
15. The CV of G3 is the population std over the mean as SPEC 3.5.3 states; the original
    `plan02_metrics_per_cell.cv_workingset` uses the sample std. `params.cv_std = "population"`
    records the choice.
16. `K_median_cell` is the median of K over the cell's rows after head drop including the last seq
    row (the definition says "the cell's rows after head drop"; the series drops the last row, the
    median does not).
17. B1-G3's re-run without the quarantined features also re-runs the null, so the table row that
    uses the re-run has its own null p95 and rank. This only happens when the quarantine is non-empty.
18. Extra columns beyond the SPEC's lists, all additive: `temporal_per_kernel.csv` gains
    `coverage_pairs` and `G2_pairs`; `table5_grid.csv` gains `coverage_pairs`, `G2_pairs`,
    `n_kernels_na_G1`, `n_kernels_na_G2`, `n_kernels_undeclared_G2`, `gc_verdict`, `gf_part1`;
    `preconditions.csv` gains `failed_source`; `gp.csv` gains one roll-up row per kernel with
    `cell_id = "all"`; `gj.csv` gains `floor_median_n_persist` and `frac_interpretable_persist`;
    `gx.csv` gains `headline_mark`, `n_labels`, `grid_id`; `gdim.csv` gains `loko_score` and
    `matched_to`; `gv_summary.csv` gains `grid_id` and `n_features_constant`; `clustering.csv` gains
    `primary`, `grid_id`, `status`; `gf.csv` rows of a (rung, grid_id) are replaced on re-run and every
    other row is kept. The G-M seed re-runs live under `gates/gm_runs/seed<i>/`, G-X's split under
    `gates/gx_runs/`, and G-DIM's matched split under `gates/splits_matched/`.
19. The tests are pytest functions and need pytest; a `unittest` fallback is not provided.
20. `series.py` hosts the shared file helpers (cells, extracts, sidecars, result writers, hashes,
    admissibility) rather than a separate helper module, to stay inside the file list the SPEC
    assigns to builder 2.

## 5. Test totals

`python3 -m pytest -q tests/` from `plan11_encoding_ladder/` on 2026-09-16, with all three builders'
modules present: 154 passed, 1 skipped, in 6 minutes 12 seconds. Builder 2's twelve test files hold 61
tests and all 61 passed; the one skip is builder 1's `test_extract.py` zstandard-module case (the
optional module is not installed on this machine). The chain test on builder 1's pipeline
(`tests/test_gates_chain.py`) passed. Every module also runs as a script by absolute path and exits 2
with the missing path on stderr when `cells.csv` is absent (checked for `variance.py`,
`gates_temporal.py select` and `models.py cluster`).

## 6. For the author

Each item is a choice the definitions leave open, exposed as a parameter with the default stated, or
a finding the synthetic runs surfaced. Values used are written into every result file's `params`.

1. **C1's re-mapped threshold excludes most of the corpus.** `apf_max >= 0.02` means `K_max >= 5,243`
   pages. From P2 Table 3's footprints, floyd, histogram and nbody (2,048 pages), gibbs (256), the
   lexer, and probably spmm and rmat_gen have `K_max` below that even with pass-boundary spikes, so
   with the SPEC default they fail C1 and are excluded from every later stage. The threshold came from
   the sandbox family's apf_queue re-map, not from the kernels. `--c1-activity-min` exists (default
   0.02); decide before the data whether to keep it, lower it, or express it against the idle floor.
2. **`rollup_kernel_refusals`** (al-Farabi 2.1): `"not_applicable"` by default, `"blocks"` as the
   alternative. On this dataset the choice decides whether any (W, H) can pass acceptance, since nbody
   is above Nyquist and rmat_gen, bnb_tsp or the lexer may trend.
3. **G1's trend precedence** (`g1_trend_precedence = "trend_first"`): a series with both a drift
   above 1 sd and a pass fraction below 0.8 reads `trend present`. A step reads as a trend under the
   drift rule (deviation 8). The alternative order is not implemented.
4. **G3 is a harmonic-comb detector.** It flags pulse trains through their rahmonics even when the
   fundamental sits below the `n // 8` floor, and it does not flag a smooth sinusoidal modulation
   (deviation 9). `g3_null = "order_shuffle"`; `--min-quef-frac` lowers the floor; `g3_min_cells = 7`;
   the reported quefrency may be a rahmonic, which the alias falsifier tolerates when every cell
   reports the same multiple.
5. **G-C's constants.** `jump_detect_ratio = 1.5`, `jump_reference = "cell_median_K"` (alternatives
   `"local_median_8"` and al-Kindi's `"cell_q25_K"`), `j_dip_max = 0.75`, `dip_window_pairs = 1`,
   `min_events_per_rep = 1`, and the aliased-regime band (`pulse_full_footprint_pages = 4096`,
   `regime_band_ratio = 1.5`, `regime_j_min = 0.75`). The wAPF pulse is the APF rule on the wAPF series.
   The content statistic is on persistent pages (`content_page_set = "persistent"`), pairing by rep
   index.
6. **G-DEC's rule (c) against the K-jump boundary.** At phase 0 after a K-jump boundary, K's relative
   drop over the run is the pulse's own drop-out (about |A| / (|A| + |C| + F)), which competes with
   the l0 decay in rule (c). On the synthetic floyd at `decay_factor = 0.7` the two drops were 0.48
   and 0.51, a near miss that the test avoids with `decay_factor = 0.6`. Decide whether (c) should be
   measured from the first post-boundary snapshot when the boundary was detected by a K jump. Also:
   `pass_frac = 0.5`, `idle_slope_rule = "any"`, `min_reps = 7`, `min_run = 3`, `boundary_source =
   "k_jump"` with `phase_pairs = 0` for `"period"`.
7. **G-F.** `part1_features = "norm"` (the rung as used; `"raw"` hears a level step between idle
   reps), `part1_consequence = "void"` (`"report"` per al-Farabi section 3 item 2), `part2_rule =
   "all_cells_outside"`, `gf_default_grid = "W8_H4"` before a selection exists. The l0 floor in
   `gf_floors.json` is pooled from per-snapshot medians (a quantile of quantiles, al-Kindi section 3
   item 4); an exact page-level pass is not implemented.
8. **G-J.** `k_factor = 3.0`; the below-null rule is `J <= J_null`; the per-cell verdict is by
   majority of pairs; the default mask is `mask_K` and `mask_persist` is the alternative for the fused
   plane (al-Kindi item 9); no floor subtraction.
9. **G-X cannot see a pure level offset between campaigns**, because it runs on the level-normalized
   features by definition; a campaign that differs in level only will read `pooling stands`, and the
   raw APF ceiling of Table 6 is where such an offset would show.
10. **G-K0.** `tail_fraction = 0.80`, `idle_pool = "pooled_snapshots"` (alternative
    `"cell_medians"`), `idle_percentile = 95`. G-K0 does not require the admissibility record (the
    SPEC asks for it only in G-F); say whether it should.
11. **The unit-level nulls cost.** B1-G1 at 500 permutations is 500 times the fold count in forest
    fits per (rung, split, label space): about 6,000 under LOKO and about 48,000 under LORO with
    `loro_mode = "cell"`; `--n-jobs` parallelizes over permutations and `--null-splits` chooses which
    splits get a null. G-ORD is about 121 LOKO runs per W per rung at the defaults
    (`gord_n_order_perm = 20`, `gord_null_perm = 100`), so several thousand fits per rung.
12. **G-V's constant-feature exclusion** (deviation 13) and its population variances; L2 over
    archetypes with at least two kernels.
13. **G-L (ii)** at the kernel level (`gl2_level = "kernel"`), r2 threshold 0.5, the drop set
    `("cov", "std", "peak2med")` when refused; the CV feature is the raw APF `cov`, mean over windows.
14. **G-M**: `gm_n_seeds = 5`, spread = max - min of the APF LOKO score over seeds; the sign rule
    (6, 0) or (7, 1). **G-DIM**: `dim_match_method = "train_importance"` (`"pca"` alternative), the
    over-dimension reduction target is the fold's training cell count.
15. **Selection tie-break**: at a W where more than one hop passes, the hop ratio nearest 0.5 wins
    (so `W32_H16` over `W32_H8`); this is Plan 03's rule and the test documents it.
16. **The pass table's `inferred:` rows** must come from source and base parameters, never from the
    trajectory or any toolkit output (al-Farabi section 3 item 4); the template's notes say so.
17. **Other defaults carried from SPEC section 8**: `wapf_norm = "median_K"`, `duty_rule = 0.1 *
    max`, the temporal channel `r_l0_q50_per` for content and combined, head drop 0 for every kernel,
    `test_frac = 0.2`, `loro_mode = "cell"`, `unit_aggregation = "cell_majority"`, the forest as
    declared, `b1g3_max_disagree = 1`, the four seeds `20260916` to `20260919` with `--seed-offset`.
18. **G3's CV** uses the population std (SPEC) where the original used the sample std (deviation 15).
19. **Tests need pytest** and take about nine minutes; the two 500-permutation B1-G1 cases are most
    of it.
