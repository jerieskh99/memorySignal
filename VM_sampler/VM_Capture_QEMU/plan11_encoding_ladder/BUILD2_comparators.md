# BUILD2_comparators.md: build epoch 2, builder 1, the comparators module set

Written 2026-09-17. Builder 1 of epoch 2 (the epoch-2 brief: "the comparators module set",
sections 1(1), 2 and 3 of the addendum, the tests of its section 6). Every rule of the brief was
kept: no server, no path under `/mnt/nfs` or `/project`, the sandbox family named only as such and
not needed, no paper prose, no git commit, no file outside `plan11_encoding_ladder/` touched, no
council file, `P2_STRUCTURE.md`, `P2E_STRUCTURE.md` or `P2_AUTHOR_ANSWERS.md` edited. Every test
uses the toolkit's own generator (`synth.py`, or the extract-level generator `tests/_synth_b2.py`
where no trajectory is needed).

## 0. What happened, in plain words, before the details

1. The brief's section numbers and file layout (`comparators/savoldi.py`, `comparators/dhodapkar_smith.py`,
   `comparators/law.py`, an adapter, the CLIs) are those of the superseded three-builder draft
   (`SPEC_epoch2.three_builders_0407.md`). The file the brief points to, `SPEC_EPOCH2.md`, is on
   this case-insensitive disk the same file as `SPEC_epoch2.md`, byte-identical to
   `SPEC_epoch2.two_builders.md`, the version al-Farabi certified. I built the definitions and the
   file layout of the certified version (Part 1.1 to 1.6) with al-Farabi's corrections applied
   (section 5 below), arranged as the module set the brief names.
2. While I was building, a parallel builder (the two-builder epoch's builder A) wrote a complete
   `plan11_encoding_ladder/comparators.py` (a single module, the same definitions, the same
   internal names `cmp_savoldi`, `cmp_dhodapkar`, `cmp_law`, the same output layout and the same CLI
   subcommands), plus the move-14 driver hook, the two `tests/test_driver.py` amendments, the Table
   7 comparator tables, the sweep figure, the runbook section and the SPEC.md row, all pointing at
   that module. A package named `comparators/` shadows a module `comparators.py` on import, so my
   package was breaking that builder's tree the moment it existed. I renamed it to
   `comparators_modules/` (CLI `python3 -m plan11_encoding_ladder.comparators_modules <sub> --out O`)
   and kept it as an independent implementation with the same output layout. Nothing of the other
   builder's work was edited or duplicated: no second driver hook, no second runbook section, no
   second SPEC row, no tables, no figure.
3. The two implementations were then cross-checked. At the function level (the last test of
   `tests/test_comparators_modules.py`): Savoldi's numbers, the Dhodapkar-Smith delta, quantiles,
   boundary counts, stability and mean phase length at every grid point, and Law's whole
   dynamic-for-X and static-for-X series, `t_first` and the "ever" counts under both `x_unit` values
   and both head-drop rules agree exactly. At the file level (both CLIs run without splits on copies
   of the same 26-cell synthetic corpus, compared by hand): `savoldi.csv`, `dhodapkar.csv`, `law.csv`,
   the sweeps at every common grid point (260 Dhodapkar-Smith rows, 104 Law rows), the three
   per-kernel files and all six feature files (names, cell order, every value to 1e-9, the idle rows
   `IDLE`) are identical; the one difference found, which cell's `U_text` a per-kernel row prints
   when the kernel has an even number of cells, was removed by adopting the same order-independent
   rule (the lower median). One documented difference remains under a non-default alternative
   (section 7, item 3). Two independent codes agreeing to the last integer on brute-force-verified
   page sets is, I think, the most useful thing this collision produced. Their `tests/test_comparators.py`
   also passes in the tree with my package present (`11 passed`), so nothing of theirs is disturbed.
4. Which of the two the epoch keeps is the orchestrator's or the author's call; section 8 says what
   deleting either one costs. The driver, the tables and the figure run `comparators.py` today. My
   CLI accepts the driver's four move-14 argument lists unchanged (verified by parsing them), so
   running this module set from the driver needs two small edits in `run_moves.py` (the other
   builder's region, not made by me): the module name `"comparators"` to `"comparators_modules"` in
   the move-14 block, and `_module_path` returning `_HERE / module / "__main__.py"` when
   `_HERE / module` is a directory (today it returns `_HERE / f"{module}.py"` and would report a
   package as an absent module). The downstream side needs nothing: `tables --only
   table7_comparators,table_comparators` and `figures --only dhodapkar_sweep` run on this module
   set's output directory as they stand (checked: 13 per-kernel rows, 30 Table 7 rows, the sweep
   figure drawn on the 13-point grid with the default marked).

## 1. Files written (all new; nothing existing was edited by me)

| File | What it holds |
|---|---|
| `comparators_modules/__init__.py` | The package docstring: the three comparators, C14, the relation to `comparators.py`. |
| `comparators_modules/_common.py` | The citation strings of SPEC_epoch2 1.1 verbatim (`CIT_SAVOLDI`, `CIT_DHODAPKAR`, `CIT_LAW`); the cell context (cells.csv, `inputs/head_drop.csv`, `gates/preconditions.csv` and `.json`); the identity columns; `admissible` and `excluded_pair_rung` annotation; per-kernel grouping over admissible cells in `schema.KERNELS` order then `idle`; the CLI plumbing (split flags, exit codes 0/2/1); `MissingInput` and `DefaultOffGrid`. |
| `comparators_modules/adapter.py` | The feature-matrix adapter `write_feature_files` (the keys of `series.build_features`, SPEC 3.1.5 / SPEC_epoch2 1.6); `run_comparator_splits` (the split stage through `models.run_split_stage`, both variants, the five (split, label space) pairs, nothing else); `comparator_gates` (G-L (i), G-DIM, G-M against APF, G-X through `gates_comparison.gate_gx`, and `verdicts.csv`); the fixed not-applicable strings of SPEC_epoch2 1.5. |
| `comparators_modules/savoldi.py` | Savoldi 2010: `savoldi_cell`, `per_kernel_rows`, `run_savoldi`, a standalone CLI. |
| `comparators_modules/dhodapkar_smith.py` | Dhodapkar and Smith 2003: `phases_at`, `dhodapkar_cell`, `per_kernel_rows`, `run_dhodapkar`, the grid and default record rules, a standalone CLI. |
| `comparators_modules/law.py` | Law 2010: the streaming per-page run pass `law_stream` (with `PageList`, `LAW_STEP_HOOK`, `_LIVE_PAGE_ARRAYS`), the series files, `sweep_rows_from_series`, `check_x2_equals_K`, `feature_vectors`, `per_kernel_rows`, the resumable multiprocessing `run_law`, a standalone CLI. |
| `comparators_modules/__main__.py` | The package CLI: `savoldi`, `dhodapkar`, `law`, `gates`, `all`. |
| `tests/test_comparators_modules.py` | Eleven tests (section 4). |
| `BUILD2_comparators.md` | This report. |

Output layout (identical to SPEC_epoch2 1.2 to 1.6 and to `comparators.py`): `gates/comparators/{savoldi,dhodapkar,law}.csv`
with `.params.json`, `dhodapkar_sweep.csv`, `law_sweep.csv` (each with its `.params.json`),
`*_per_kernel.csv`, `law_cells.json`, `law_series/<cell_id>.npz`, `features/cmp_<name>/Wall_Hall_{raw,norm}.npz`,
`gates/splits/cmp_<name>/Wall_Hall/<split>__<labelspace>[__raw]/`, `gates/comparators/{gl,gdim,gm,verdicts}.csv`
with `.params.json`, and the comparator rows of `gates/gx.csv`. Every JSON carries `schema`,
`params`, `citation`; `params` carries `epoch: 2` and every constant and CLI value used.

## 2. Comparator by comparator

| Comparator | Definition implemented | Extract columns read | Feature row at the whole-cell point (raw; level-normalized) | Test that passes | Test that refuses | Citation in the docstring |
|---|---|---|---|---|---|---|
| `cmp_savoldi` (`savoldi.py`) | For each consecutive pair the number of differing pages; mu_dmp the sample mean and sigma_dmp the sample SD of that count over the run; U = mu +/- sigma. In our rows the count is the extract's `K` on every row after the head drop (`rows = all_after_head_drop`; `rung_series` drops the last seq). | `K` only (and `series.k_median_cell`, the median of the same `K`). | raw (2): `cmp_savoldi.K.mean`, `cmp_savoldi.K.sd` in pages; norm (2): `cmp_savoldi.k_over_med.mean`, `.sd` (divided by the cell's median K, P2 Sec. V G-L (i)). | `test_savoldi_matches_truth`: `K_mean` and `K_sd` (ddof 1) equal the generator's truth to 1e-9; `rung_series` drops the last row; `U_text` has C14's form; `test_feature_files_and_split_layout`: the split stage runs on the file, `feature_count == 2`, `dim_status full vector`. | The same test: one row after the head drop gives `not run: fewer than two rows`; an unknown `rows` rule raises; `test_cli_exit_codes`: a missing cells.csv exits 2 naming it. | C14 cand. 1 (the exact sentence: sample mean and SD of the per-pair count; "Only the interval differs"); P2 Sec. 0 Baseline; P2E Sec. 7 (the one-number baseline); P2 Sec. V 'The splits', 'Models', 5.1 Plan 08 through `models.run_split_stage`; AA 2026-09-17 (the pick accepted); Part 4 items 1, 2. |
| `cmp_dhodapkar` (`dhodapkar_smith.py`) | delta_{i,i-1} = (\|W_i u W_{i-1}\| - \|W_i n W_{i-1}\|) / \|W_i u W_{i-1}\|; a phase change when delta exceeds a threshold; stability; average phase length. With W_i our changed-page set, `n_persist` and `n_union` give delta = 1 - J identically; computed as `1 - ex["J"]`, never a second intersection. Every grid point computed and kept; the default marked, never selected against labels. | `J` only, rows `[head_drop:-1]`; a blank J dropped and counted in `n_pairs_blank`. | raw (3) at the default threshold: `cmp_dhodapkar.n_boundaries`, `.stability`, `.mean_phase_length`; norm (3): `boundary_rate = B / n`, `stability`, `mean_phase_frac = mpl / n` (the delta is level-free; the pair count is the only per-cell scale). | `test_dhodapkar_delta_is_one_minus_J_and_the_sweep_is_whole`: delta equals 1 - truth J to 1e-12; on a pulsed cell the boundary count at 0.3 equals twice the pulses in range (the set enters and leaves: two dips of about 1/3), stability and mean phase length follow the formulas; the sweep has every grid point once per cell with one `is_default`; `test_pair_rung_exclusion_...`: the split stage runs (`feature_count == 3`), `default_source` cites AA 2026-09-17. | The same tests: a default off the grid raises `DefaultOffGrid` and the CLI exits 2 with `missing input: delta_th_default 0.05 is not on the grid ...`; no defined delta gives `not run: no pair with a defined delta`; a cell with a refused failed count is absent from `cmp_dhodapkar`'s predictions and present in `cmp_savoldi`'s (`series.PAIR_RUNGS`). | C14 cand. 3 (the delta between consecutive working sets; "Their delta is one minus Jaccard, identically"; threshold, stability, average phase length); P2 Sec. 0 Baseline; P2E Sec. 7 (the named method on the overlap axis); AA 2026-09-17 (a declared sweep, every point kept, 0.04 the default); C14 author (declare delta_th before the labels or sweep it and show the sweep); P2 Sec. V through `models.run_split_stage`; Part 4 items 3, 4, 5. |
| `cmp_law` (`law.py`) | A page is dynamic in X consecutive dumps if X or more consecutive hashes differ, static if identical; an index of run lengths answers all X in one pass. In our rows "dynamic in X" is a membership run of length X - 1 in the changed sets, "static in X" a non-membership run of length X - 1 (`x_unit = dumps`, C14's exact reading; `pairs` is the brief's alternative, a run of length X). The pass holds four int32 arrays of length N (the current and the longest membership and non-membership run per page), never a past page set, and reads every X of the grid off them at every pair. | No extract column for the pass: the trajectory's `seq` and `page_index` through `extract.open_text`, with the row loop copied from `extract._stream` (2026-09-17, reduced to seq and page_index). The extract's `K` for `check_x2_equals_K` and `series.k_median_cell` for the normalized row. | raw (4) at the default X, `feature_source = window`: `cmp_law.dyn.mean`, `.dyn.sd`, `.sta.mean`, `.sta.sd` in pages; norm (4): `dyn_over_med.mean`, `dyn_over_med.sd`, `union_over_med.mean` (= (N - sta) / K_median), `sta_over_med.sd`. Under `ever`: `dyn.ever`, `sta.ever` and their two normalized forms. | `test_law_streaming_pass_matches_brute_force`: on a cell with a gap and kept page sets, every dyn and sta series equals the brute-force intersection and union of the L(X) sets ending at t for X in (2, 3, 4, 5, 8, 10, 16) under both units; the gap pair reads 0 and N; the "ever" counts equal the brute-force longest runs; `reset_at_head_drop` restarts the runs; `check_x2_equals_K` is `true`; the feature vectors equal the series' moments; `test_law_memory_model_and_resume`: at most one page array is live at every step, the state is exactly 4 x 4 x N bytes, a second run re-streams nothing and `--force` does; `test_chain_with_law_on_builder1_pipeline`: the whole chain on `synth.py corpus` -> `extract.py` -> `comparators_modules all --jobs 2`, `check_x2_equals_K` true on all 26 cells, 30 verdict rows. | `test_law_refuses_a_non_monotone_seq_and_a_missing_trajectory`: `refused: seq not monotone at row <n>` and no series; a missing trajectory is a per-cell `not run: trajectory file missing: <path>` with the cell out of the feature file and counted in `n_windows_dropped`; a default X off the grid raises `DefaultOffGrid` and the CLI exits 2; `x_unit = pairs` makes `check_x2_equals_K` `not applicable: ...`. | C14 cand. 2 (the sentence and C14's exact reading); P2 Sec. 0 Baseline (Law and Savoldi as Table 7 rows for the IFIP version); P2E Sec. 7 (held for IFIP); P2 Sec. V through `models.run_split_stage`; AA 2026-09-17 (built as a third module for the IFIP version); Part 4 items 6 to 9; SPEC_epoch2_review_al_farabi.md section 1 (this pass is a second extract, bound to the first by `check_x2_equals_K`) and section 5 item 2 (the `PageList` wrapper). |

The gates applied to every comparator row, through `models.run_split_stage` exactly as for a
rung (SPEC_epoch2 1.5): B1-G1 at the unit, B1-G3's quarantine, B1-G6's majority, G-N's headline
recall, G-K0's relabelling at read time, the admissibility record at read time; then, in
`adapter.comparator_gates`: G-L (i) on the norm variant's LOKO/archetype run (`pass` / `level only`
/ the `not run:` forms of `gates_comparison.gate_gl`), G-DIM per variant (`d`, `dim_status`, no
matched row), G-M against APF like with like (`gm_compare`, `spread` from `gates/gm.params.json`,
the split's own `not applicable:` string on within-trace, `not run: gates/gm.params.json missing
(move 12)`, `not run: no selection for apf`, the missing scores.json by path), and G-X through
`gates_comparison.gate_gx(out, "cmp_<name>", grid_id="Wall_Hall")`. Not applied, with the strings
of SPEC_epoch2 1.5 recorded in `verdicts.params.json` under `not_applied`: G-C, G-F (i), G-P,
G-DEC, the temporal gates and G-ORD; `resolution` reads `whole cell (per cell, by definition)`.

## 3. The declared parameters, with their defaults (every one written into `params`)

| Parameter | Default | Alternatives | Where |
|---|---|---|---|
| `rows` (Savoldi) | `all_after_head_drop` | `rung_series` | Part 4 item 1; `--rows` |
| `ddof` (Savoldi) | 1 | 0 | Part 4 item 2; `--ddof` |
| `DHODAPKAR_GRID` | `(0.02, 0.04, 0.08, 0.1, 0.16, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)`, the union of the brief's declared sweep {0.02, 0.04, 0.08, 0.16} and SPEC_epoch2 1.1's ten points | any list through `--grid`; the default must be on it | Part 4 item 3; section 6 item 1 |
| `DHODAPKAR_DEFAULT` | 0.04 (AA 2026-09-17), asserted equal to `schema.COMPARATOR_DECLARED_DEFAULTS["cmp_dhodapkar"]` | `--delta-th-default`, recorded as a departure | Part 4 item 3 |
| `boundary_rule` | `gt` | `ge` | Part 4 item 4 |
| `phase_length_rule` | `n_over_b_plus_1` | `interior` (the pairs strictly between two consecutive boundaries; NaN below two) | Part 4 item 5 |
| `LAW_X_GRID` | `(2, 3, 4, 5, 8, 10, 16)`, the union of the brief's declared set {2, 3, 5, 10} and SPEC_epoch2 1.1's (2, 4, 8, 16) | any list through `--x-grid`; the default must be on it | Part 4 item 6; section 6 item 1 |
| `LAW_X_DEFAULT` | 4 (SPEC_epoch2 Part 4 item 6; the author has declared no X), asserted equal to `schema.COMPARATOR_DECLARED_DEFAULTS["cmp_law"]` | `--x-default`, recorded as a departure | Part 4 item 6 |
| `x_unit` | `dumps` | `pairs` | Part 4 item 7 |
| `head_drop_rule` | `runs_from_seq_first` | `reset_at_head_drop` (all four state arrays restart at the head drop; recording starts when the window lies inside the kept rows) | Part 4 item 8 |
| `feature_source` | `window` | `ever` | Part 4 item 9 |
| `COMPARATOR_NORM_RULE` | `median_K` (Dhodapkar-Smith by the pair count) | none | Part 4 item 10 |
| off-grid default | refused, exit 2 (`missing input: <name> <v> is not on the grid <grid>`) | al-Farabi's other option, append and record, not built | review section 2 (b) |

## 4. Tests

`tests/test_comparators_modules.py`, eleven pytest functions, all synthetic, forests at
`n_estimators = 10` and `n_perm <= 5`:

1. `test_savoldi_matches_truth` (SPEC_epoch2 1.10 test 1).
2. `test_dhodapkar_delta_is_one_minus_J_and_the_sweep_is_whole` (test 2; the boundary count is twice the pulses, see section 7 item 4).
3. `test_law_streaming_pass_matches_brute_force` (test 3, both units, the gap, the reset rule, the "ever" counts, the run through the corpus path, the feature vectors, the `ever` source, the record strings, the off-grid refusal).
4. `test_law_refuses_a_non_monotone_seq_and_a_missing_trajectory` (the refusing half of test 3, plus the missing-trajectory refusal).
5. `test_law_memory_model_and_resume` (test 4, with the `PageList` wrapper of review item 2; the hook sees every snapshot's page array; a changed pass parameter re-runs the cell).
6. `test_feature_files_and_split_layout` (test 5, first half).
7. `test_pair_rung_exclusion_keeps_savoldi_and_drops_dhodapkar` (test 5, second half, on the extract-level corpus; Law's exclusion is in test 10).
8. `test_gates_for_comparators` (test 6: G-L (i), G-DIM, G-M with and without the measured spread and without APF's selection, G-X, 30 verdict rows).
9. `test_cli_exit_codes` (test 10: exit 2 on a missing cells.csv and on an off-grid default for both comparators; `all --no-splits` writes the six per-cell and per-kernel CSVs and the six feature files with Law's rows refused because the extract-level corpus has no trajectory).
10. `test_chain_with_law_on_builder1_pipeline` (test 11, with the Law pair-rung exclusion, the idle rows `IDLE`/`idle` against a cells.csv that says `control`/`sleep`, G-K0's relabelling at read time, a resumed Law run that re-streams nothing, and no sandbox name under `gates/comparators/`).
11. `test_cross_check_against_parallel_comparators_py` (skipped when `comparators.py` is absent): Savoldi, Dhodapkar-Smith at every grid point and Law's series, `t_first` and "ever" counts under both units and both head-drop rules agree between the two implementations.

Tests 7, 8 and 9 of SPEC_epoch2 1.10 (the tables, the figure, the driver plan) are outside my
scope and belong with the tables, figures and driver hook the other builder wrote.

Timings under the concurrent load of this session (four builders and several suites running,
load average 500 to 630): the five core tests 42 s, the four extract-level corpus tests 192 s,
the chain and cross-check 233 s; the whole file on its final state, `11 passed in 466.11s
(0:07:46)`. On an idle machine expect about a third of that.

## 5. Corrections from `SPEC_EPOCH2_review_al_farabi.md` applied (where the review says a line fails or must change)

1. Section 5 item 2 (a `numpy.ndarray` cannot be a `WeakSet` member): `law_stream` wraps each
   snapshot's page array in `PageList` and tracks the wrappers in `_LIVE_PAGE_ARRAYS`; the hook
   receives the wrapper's array; the test asserts at most one live wrapper at every step.
2. Section 5 item 4 / section 2 (a) (the record follows the value): `default_source` cites AA
   2026-09-17 only when the default equals the module value, else `CLI --delta-th-default <v>
   (departs from AA 2026-09-17's 0.04)`; `grid_source` names the module constant or `CLI --grid`;
   `x_default_source` and `x_grid_source` likewise; `default_is_module_value` and
   `grid_is_module_value` are written as booleans so the tables print "declared default" only for
   the module value. The module constants are asserted equal to `schema.COMPARATOR_DECLARED_DEFAULTS`
   at import.
3. Section 2 (b) (a default off the grid): refused before any file is written, exit 2, the message
   `missing input: delta_th_default <v> is not on the grid <grid>` (and `x_default <v> ...`).
4. Section 5 item 5 (the G-M row of a not-applicable split): the comparator's `gm.csv` row carries
   the split's own `not applicable: one window per cell` before any `not run:` about the spread or
   the missing files.
5. Section 1's mark (the Law pass is a second extract, not a reading of the first): the module
   docstring says so, `check_x2_equals_K` binds it to `extract.csv`, and a non-monotone seq or an
   out-of-range page index is refused with the extractor's own form.
6. The condition (4) failure of the review is B1 (builder B's), not mine; nothing to apply here.

## 6. Deviations from SPEC_EPOCH2.md and from the brief, each with its reason

1. The declared grids are the union of the two declarations on record. The brief says, under "the
   author's decisions", delta_th in {0.02, 0.04, 0.08, 0.16} and X in {2, 3, 5, 10}; `P2_AUTHOR_ANSWERS.md`
   "Decisions of 2026-09-17" names no grid value at all (only "a declared sweep with every point
   kept and 0.04 marked as the default"); the certified SPEC_epoch2 1.1 names (0.04, 0.1 .. 0.9) and
   (2, 4, 8, 16) with X = 4 the default, al-Farabi's review section 6 item 10 says "Law's X = 4 is the
   addendum's choice, not yours", and `schema.COMPARATOR_DECLARED_DEFAULTS` (already in the tree,
   read by the tables) declares X = 4, which is not in {2, 3, 5, 10}. Picking one set would either
   contradict the brief or leave the shared schema constant naming a default off the grid. Every point
   of both sets is computed and kept (a per-cell loop over a tuple costs nothing), the defaults are the
   ones every record agrees on or the certified spec's (0.04; 4), and both grids stay CLI flags, so
   `--grid 0.02,0.04,0.08,0.16` and `--x-grid 2,3,5,10 --x-default 3` run the brief's sets alone (the
   record then says the default departs from the module value, as it should). Section 7 item 1 puts
   the choice to the author.
2. The layout is a package (`comparators_modules/`, one module per method, an adapter, a package
   CLI) as the brief names, not the single `comparators.py` of the certified SPEC_epoch2 1.1; renamed
   from `comparators/` for the reason in section 0. Consequence: `python3 -m
   plan11_encoding_ladder.comparators_modules <sub>` instead of `comparators.py <sub>`; the module's
   internal names, files and columns are the spec's.
3. No driver hook, runbook section, SPEC.md row, `schema.py` block, `series.PAIR_RUNGS` line,
   `test_driver.py` amendment, table or figure was written by me: every one of them already existed
   in the tree when I reached it, written by the parallel builder, and all point at `comparators.py`.
   I verified them instead: `run_moves plan --moves 14` lists the seven commands of SPEC_epoch2 1.7
   in order; `parse_moves("0-14")` is `range(15)` and `parse_moves("15")` raises; the two
   `test_driver.py` tokens read 15; the runbook has "Move 14: the comparators (build epoch 2)" and
   the sentence on `--moves 0-14`; SPEC.md section 7 has the move-14 row; `series.PAIR_RUNGS` lists
   `cmp_dhodapkar` and `cmp_law` (my exclusion tests depend on it and pass). The four `comparators
   <sub>` argument lists the plan emits parse against my CLI unchanged (the `gates` subparser accepts
   the common `--null-splits` for that reason). I also ran the hook as it stands, `run_moves run
   --moves 14 --null-perm 5 --n-estimators 10 --null-splits loko --n-jobs 2`, on a copy of the
   26-cell smoke corpus: all seven move-14 commands `done`, exit 0, the Savoldi level-normalized
   LOKO row of `table7_comparators.csv` reading `feature count 2`, `accuracy 0.250`, `null p95 not
   run: 5 permutations < 500`, the G-C, G-L, G-M and G-X strings of SPEC_epoch2 1.5 and 1.8, and
   `report/manifest.json` listing `gates/comparators/verdicts.csv`.
4. `check_x2_equals_K` is a string, not only `true`/`false`: `not applicable: x_unit = pairs (...)`
   and `not applicable: X = 2 not on the grid` when the identity does not exist, in the toolkit's
   refusal vocabulary rather than a silent `false`.
5. `reset_at_head_drop` zeroes all four state arrays (the two maxima included), not only the two
   current-run arrays the pseudo-code names; "the state restarts there" read literally, recorded in
   `params.reset_scope`. Recording under that rule starts at `head_drop + L - 1` (the window inside
   the kept rows), which is what SPEC_epoch2 1.10 test 3's sentence about "sets 5..7 only" requires.
6. The Law resume also re-runs a cell whose recorded pass parameters (`x_grid`, `x_unit`,
   `head_drop`, `head_drop_rule`) differ from the current ones, not only a cell without `status ok`;
   otherwise a changed grid would silently reuse a series file computed for another grid. Recorded in
   `params.resume_rule`. The sweep and the feature files are rebuilt from the series files on every
   run, so a resumed run is complete; `--only REGEX` restricts the pass and keeps the other cells'
   entries and rows (checked by hand: `--only gemm --force` re-streams two cells and writes 26 rows).
7. A missing trajectory file and an out-of-range `page_index` are per-cell refusals (`not run:
   trajectory file missing: <path>`; `refused: page_index <p> outside 0..N-1 at row <n>`), not an
   exit 2 of the whole command: the run continues, the cell is out of the feature file and counted in
   `n_windows_dropped`. A per-cell internal error ends the command with exit 1 after every file is
   written (`law_cells.json` carries the traceback).
8. `law.csv` carries the default-X statistics beside the identity, `head_drop`, `check_x2_equals_K`
   and `status` (SPEC_epoch2 1.4 lists the latter only) so the row is readable without the sweep.
9. The hashed working-set-signature variant of Dhodapkar and Smith is not built and the params say so
   (`signature_variant`), as the three-builder draft required and the certified spec implies.

## 7. For the author

1. The grids (section 6 item 1). Say which delta_th sweep and which X set the paper declares: the
   brief's four and four, the certified spec's ten and four, or the union that runs by default. Every
   point is on disk either way; only the figure's x-axis and the sweep files' length change.
2. Law's default X = 4 is the certified spec's choice, not yours (review section 6 item 10). Under
   the brief's set {2, 3, 5, 10} the analogous choice by the spec's own rule ("the smallest grid point
   above the trivial X = 2") would be 3. Declare one; the record already says which source it came
   from.
3. The `interior` phase-length rule (Part 4 item 5, not the default) is read two ways by the two
   implementations: this module set counts the pairs strictly between two consecutive boundaries
   (`b_{i+1} - b_i - 1`), `comparators.py` counts the spacing (`b_{i+1} - b_i`, the boundary pair
   inside the phase). They differ by exactly one pair. If you ever switch to `interior`, say which.
   Under the default rule the two agree exactly.
4. A one-snapshot pulse is two phase changes under Dhodapkar-Smith: the pulse set enters at seq
   b - 1 and leaves at seq b (delta about 2/3 on the test cell, 4,096 extra pages on 2,048; about
   1/2 with the corpus's gemm preset, 4,096 on 4,096). On the real
   gemm cells the two dips per pass will double the boundary count relative to the pass count. That
   is the method as published, not an artefact; SPEC_epoch2 1.10 test 2's sentence ("equals the
   number of boundaries") undercounts it, and my test asserts the double.
5. At churn 0.02 the steady-state delta is 1 - 0.98/1.02 = 0.039, within 0.001 of the declared
   default 0.04: on the synthetic corpus the boundary count at 0.04 is a coin flip per pair for the
   low-churn kernels (gemm, nbody, histogram at about 31 of 59). On the real corpus J's level is not
   known to me; the sweep exists so that this sensitivity is visible before any label is read.
6. The brief's phrase "{2, 3, 5, 10} pairs" could be read as `x_unit = pairs`; C14's exact reading
   (X dumps span X - 1 pairs) is the default `dumps`, and the alternative is one flag. Under `dumps`
   X = 2 is the changed set itself and `check_x2_equals_K` is the built-in consistency test; under
   `pairs` that identity does not exist and the check reads `not applicable`.
7. Law's cost on the real corpus: the pass ran at about 31,000 rows per second per process here
   (8.4 s for a 263,000-row synthetic gemm cell), so a four-million-row cell is about two minutes per
   process; 96 cells at `--comparator-jobs 4` about fifty minutes, resumable per cell.
8. B1-G3 will most likely quarantine two of Dhodapkar-Smith's three normalized features
   (`boundary_rate = 1 - stability` identically; `mean_phase_frac = 1 / (B + 1)` under the default
   rule); the row's score is then a one-feature re-run through `models.effective_scores`. Expected;
   recorded in `params.feature_note` (review section 6 item 9).

## 8. Open items and the collision (for the orchestrator)

1. Two implementations of the same comparators now exist: `comparators.py` (the two-builder
   builder A; wired into the driver, the tables, the figure, the runbook) and
   `comparators_modules/` (this report). They agree exactly on every default-rule number the tests
   compare. Keeping both costs the suite the runtime of `tests/test_comparators_modules.py` (about
   eight minutes under load, three on an idle machine) and leaves two CLIs for one thing. Deleting
   `comparators_modules/` and its test loses nothing the tree needs and keeps the cross-check as a
   one-time result in this report; deleting `comparators.py` instead needs the two `run_moves.py`
   edits of section 0 item 4 (the module name in the move-14 block and a package-aware
   `_module_path`), the tables' and the figure's file reads are unchanged (same layout), and the
   other builder's `tests/test_comparators.py` would go with it.
2. `tests/test_comparators_modules.py` marks nothing slow; the chain test builds a 26-cell
   trajectory corpus (40 pairs) once per module.
3. Nothing in this module set reads a score, a verdict or a selection to compute a per-cell
   statistic; the only writes outside `gates/comparators/`, `features/cmp_*` and `gates/splits/cmp_*`
   are the `cmp_*` rows of `gates/gx.csv` through `gates_comparison.gate_gx`'s per-rung replacement.

## 9. The suite

Before (`python3 -m pytest -q tests` from `plan11_encoding_ladder/`, started 04:35 before any
file of mine existed, on a machine running three other builders' suites):

    158 passed, 1 skipped in 2715.71s (0:45:15)

After (the same command, started 05:25 with my files in place; pytest collected 205 tests, the
tree as it stood at that moment, every builder's concurrent files included; load average 500 to
630 throughout):

    1 failed, 203 passed, 1 skipped in 5776.43s (1:36:16)
    FAILED tests/test_comparators.py::test_gates_for_comparators - AssertionError...

The one failure is in the parallel builder's own test of their own module (`tests/test_comparators.py`
line 359, `gd["cmp_law"]["d"] == "4"` against `'3'`: the G-DIM feature count of `cmp_law` read
through `models.effective_scores` is the re-run's count once B1-G3 has quarantined a Law feature,
a data-dependent outcome of their run). That file imports `plan11_encoding_ladder.comparators`,
never `comparators_modules`, and the same file run alone in the same tree at 05:50 gave
`11 passed in 600.94s`; the version pytest collected at 05:25 was 05:21's, which its author was
still editing. My eleven tests are among the 203 passes. Every one of the 158 pre-existing tests
is among the passes without amendment; the only amended existing test in the epoch is the two-token
change of `tests/test_driver.py` made by the parallel builder (SPEC_epoch2 1.7), not by me.

Because files kept changing under the run, I also ran the test files one by one on the tree as it
stood between 05:50 and 07:00 (each in a separate process, same load): `test_comparators_modules.py`
`11 passed in 466.11s`; `test_comparators.py` `11 passed`; `test_driver.py` `11 passed`;
`test_gates_chain.py` `1 passed`; `test_gates_models.py` + `test_gates_series.py` +
`test_gates_splits.py` `21 passed`; `test_schema.py` + `test_gates_verdicts.py` + `test_gates_nulls.py`
+ `test_gates_variance.py` + `test_runner_guard.py` `24 passed`; `test_gates_precondition.py` +
`test_gates_readings.py` + `test_gates_calibration.py` `20 passed`; `test_gates_comparison.py`
`8 passed`. A sibling builder's full run that collected later (236 tests, `3 failed, 224 passed,
1 skipped in 5105.63s`) reported three failures; the one name visible in its tail was
`tests/test_driver.py::TestEpoch2Builder3::test_unchanged_resume_runs_zero_split_stages` (builder
B's new driver tests, in progress); I did not see the other two names and make no claim about them.
