# BUILD_epoch2_comparators.md: builder A's report on build epoch 2, Part 1 (the comparators)

Written 2026-09-17 by builder A of build epoch 2 against `SPEC_epoch2.md` (the two-builder
addendum), with the corrections of `SPEC_epoch2_review_al_farabi.md` applied where that review
says a test or a record cannot hold as written (its items 5.2, 5.4, 5.5 and 5.6; section 5 lists
each). This file is `plan11_encoding_ladder/BUILD_epoch2_comparators.md`, the name the epoch
brief gave; `SPEC_epoch2.md` Part 3.4 calls the same report `BUILD_epoch2_A.md`. No server was
touched, no path under `/mnt/nfs` or `/project` was read, the sandbox family is not named
anywhere in the code, the tests or this report, no paper file and no council file was edited,
nothing was committed. Every test uses synthetic data from `synth.py` (trajectory level,
extracted by `extract.py`) or the report fixture `tests/report_fixtures.py`.

## 1. What was built

Three exact-input comparators, each computed per cell from the extract (or, for Law, from the
trajectory in a second streaming pass) and submitted to the same split stage and the same
comparison gates as the ladder's rungs, at the whole-cell point `Wall_Hall` (one vector per cell,
by definition):

- **Savoldi 2010** (`cmp_savoldi`; C14 cand. 1): `U = mu_dmp +/- sigma_dmp`, the sample mean and
  standard deviation of the per-pair changed-page count `K` over the run. Raw features
  `(K_mean, K_sd)`; level-normalized `(K_mean / K_median, K_sd / K_median)` (the count-rung rule of
  P2 Sec. V G-L (i)); `U_text` per cell in the paper's own form, for example `0.8392% +/- 0.0104%`.
- **Dhodapkar and Smith 2003** (`cmp_dhodapkar`; C14 cand. 3): `delta = 1 - J` between
  consecutive changed sets (the derivation `(n_union - n_persist) / n_union = 1 - J` is in the
  docstring; the module never computes a second intersection), a phase boundary when
  `delta > delta_th`, stability `1 - B / n` and mean phase length `n / (B + 1)`, on the declared
  grid `(0.04, 0.1, 0.2, ..., 0.9)` with `0.04` the author's declared default (AA 2026-09-17).
  Every grid point is written to `dhodapkar_sweep.csv` (ten rows per cell), the default is
  marked with `is_default`, nothing is selected against labels.
- **Law et al. 2010** (`cmp_law`; C14 cand. 2): a second streaming pass over the trajectory
  (`law_stream`, the row loop copied from `extract._stream` and reduced to `seq` and
  `page_index`) that keeps per page the current membership and non-membership run lengths in the
  changed sets and their maxima (four int32 arrays of length N, one snapshot's page list at a time,
  never a page set beyond the current snapshot) and records at every pair position the number of
  pages dynamic for X (a membership run of length `X - 1`) and static for X for every X of the
  declared grid `(2, 4, 8, 16)`, `X = 4` the default. `check_x2_equals_K` binds the pass to the
  first extract (at X = 2 the dynamic count is K by identity) and is `true` on every synthetic cell.

Around them: the feature files `features/cmp_<name>/Wall_Hall_{raw,norm}.npz` in the exact shape
of `series.build_features` (idle rows `IDLE` / `idle`); the split stage through
`models.run_split_stage` on both variants and the five (split, label space) pairs (B1-G1, B1-G3,
B1-G6, G-N, G-DIM, G-K0 at read time, exactly as for a rung; within-trace reads `not applicable:
one window per cell`); G-L (i), G-DIM, G-M against APF (like against like: the norm variant against
APF's norm run, the raw variant against APF's raw run, with move 12's measured spread) and G-X
through `gates_comparison.gate_gx`; the `verdicts.csv` summary (30 rows); the two tables
(`table7_comparators`, the 30 comparator rows of Table 7, raw row then level-normalized row per
comparator and split; `table_comparators`, the methods' own per-kernel numbers); the sweep figure
`fig_dhodapkar_sweep`; the driver's move 14 (seven commands); the runbook section; the SPEC
move-table row; and eleven tests.

What each Table 7 row means (recorded in every `params.json` as `norm_row_meaning`): the raw row
is the method as published, level-inclusive by construction; the level-normalized row divides
Savoldi's and Law's count features by the cell's median K (so Savoldi's first normalized feature
is the mean over the median, a number near one carrying only the skew of the count distribution,
and the second the relative spread), and divides Dhodapkar-Smith's counts by the pair count (the
delta is a Jaccard, level-free, so no level division exists).

## 2. Files touched (with regions)

New:

- `plan11_encoding_ladder/comparators.py` (the module, SPEC_epoch2 Part 1.1 to 1.6). Every
  public function's docstring cites the definition it implements (the C14 candidate, P2 Sec. 0
  Baseline, P2E Sec. 7, P2 Sec. V through `models.run_split_stage`, AA 2026-09-17) and names the
  parameters the definition leaves open with their Part 4 item; the three citation strings
  `CIT_SAVOLDI`, `CIT_DHODAPKAR`, `CIT_LAW` are written verbatim into every `params.json`.
- `plan11_encoding_ladder/tests/test_comparators.py` (the eleven tests of Part 1.10; section 3).
- `plan11_encoding_ladder/BUILD_epoch2_comparators.md` (this report).

Edited, in builder A's regions only (SPEC_epoch2 Part 3.2), with targeted replacements:

- `schema.py`: the appended `COMPARATORS` block (`COMPARATORS`, `COMPARATOR_NAMES`,
  `COMPARATOR_DISPLAY`, `COMPARATOR_GRID_ID`) and, inside the same block,
  `COMPARATOR_DECLARED_DEFAULTS = {"cmp_dhodapkar": ("delta_th", 0.04), "cmp_law": ("X", 4)}`
  (section 5, item 1).
- `series.py`: the `PAIR_RUNGS` line only.
- `tables.py`: `TABLE_NAMES` (+ `table7_comparators`, `table_comparators`), `CITATIONS` (both
  names), `TABLE7_VARIANT = "both"`, a guarded import of `schema`'s comparator block (the fallback
  tuple mirrors it, as `_report_common` mirrors the rest of `schema`), `_gate_csv` and the
  indirection inside `_gl_text`, `_gdim_text` and `_gm_text` (the last with the keyword
  `rung_b="apf"`; the three behave exactly as before for the rungs, the `not run:` texts included),
  `_feature_count_text` and the CHECK_3 M5 line of `table7` on both branches (al-Farabi 5.6), the
  new functions `table7_comparators`, `table_comparators`, `table_comparators_columns`, the
  constant `TABLE_COMPARATORS_COLUMNS`, and the two dispatch lines of `run`.
- `figures.py`: `FIGURE_NAMES` (+ `dhodapkar_sweep`), `fig_dhodapkar_sweep`, its dispatch line,
  the name list of the module docstring.
- `latex_skeleton.py`: `\input{tables/table7_comparators.tex}` and
  `\input{tables/table_comparators.tex}` after the Table 7 shell in the same generated-table form,
  the `fig_dhodapkar_sweep` placeholder in the figure list. Comment lines only; no sentence.
- `run_moves.py`: `MAX_MOVE = 14` and the bound in `parse_moves`; the `--moves` default `0-14`
  and the four flags `--delta-th-default`, `--law-x-default`, `--savoldi-rows`,
  `--comparator-jobs` in `_add_run_args`; the move-14 block after the move-13 line (the seven
  commands of Part 1.7 (c), with the one change of section 5 item 4); and the default of the
  keyword `run_plan(max_move=...)` that another session had added meanwhile (section 5 item 5).
- `tests/test_driver.py`: the two one-token amendments of Part 1.7 (`parse_moves("14")` becomes
  `"15"` in `test_parse_moves`; `--moves 14` becomes `15` in `test_cli_guards`). Nothing else.
- `RUNBOOK.md`: the section "Move 14: the comparators (build epoch 2)" after move 13, and one
  sentence in the driver section on `--moves 0-14` and the four flags.
- `SPEC.md` section 7: one appended row for move 14 in the move table.

Not touched: `verdicts.py` (no new verdict constant; every new string is a `not run:` or
`not applicable:` form), `_b2_common.py`, `_synth_b2.py`, `extract.py` (the Law pass copies its
row loop with the comment `# copied from extract._stream, 2026-09-17, reduced to seq and
page_index`), `models.py`, `gates_comparison.py`, every other builder B file, every file outside
`plan11_encoding_ladder/`, and no council or paper file.

## 3. Tests added (`tests/test_comparators.py`; `n_estimators = 10`, `n_perm <= 10` wherever a forest runs)

| # | Test | What it proves |
|---|---|---|
| 1 | `test_savoldi_matches_truth` | `K_mean`, `K_sd` (ddof 1 and 0), `K_median` and `n_rows_used` against the generator's `K[3:]` to 1e-9; `rows="rung_series"` drops the last row; `U_text` matches `^\d+(\.\d+)?% \+/- \d+(\.\d+)?%$`; `params.ddof == 1`, `rows`, the citation, `norm_row_meaning`, `interval_note`; the per-kernel `U_text_median`; one row after the head drop reads `not run: fewer than two rows` |
| 2 | `test_dhodapkar_delta_is_one_minus_J_and_the_sweep_is_whole` | `delta == 1 - ex["J"]` on the kept rows (the generator's J to 1e-9, the extract's ten-digit precision; section 5 item 6); at `delta_th = 0.3` the boundary count equals `count(delta > 0.3)`, which is two per pulse boundary (section 5 item 7); `stability == 1 - B / n`; `mean_phase_length == n / (B + 1)`; the `interior` rule's mean and its NaN below two boundaries; the `ge` rule; the sweep CSV has exactly ten rows per cell, every grid value, one `is_default` equal to `params.default = 0.04`; `default_source`, `grid_source` and `default_appended_to_grid` follow the value that ran; a default off the grid is refused (`ValueError`; exit 2 on the CLI) or appended and recorded under `--off-grid-default append` |
| 3 | `test_law_streaming_pass_matches_brute_force` | on a 30-pair cell with a gap at seq 7 written with `keep_sets`, for X in (2, 4, 8) under both `x_unit` values the pass's `dyn` and `sta` series equal the brute force over the truth sets exactly, `dyn_ever` equals the brute-force count of pages with a membership run >= L, the gap pair reads `dyn = 0` and `sta = N`; `check_x2_equals_K` is `true` (and `not applicable` under `pairs`); `reset_at_head_drop` with `head_drop = 9` restarts the runs (the first recorded window is sets 9..11; the `ever` counts restart with them) and `runs_from_seq_first` does not; a `--corrupt seq_reverse` cell raises `Refusal("seq not monotone at row <n>")`, the worker records `refused: seq not monotone at row <n>` and deletes a stale series file |
| 4 | `test_law_memory_model_and_resume` | `LAW_STEP_HOOK` sees one `_PageList` wrapper per snapshot with `len(_LIVE_PAGE_ARRAYS) <= 1` at every step and the page counts of the truth; `memory_model` reports four N-length int32 arrays (`4 * N * 4` bytes); a second `run_law` skips the cell (`law_cells.json` `elapsed_s` identical, `resumed = true`, `n_run = 0`, `n_resumed = 1`), `--force` re-runs it, a changed pass parameter (`x_unit`) re-runs it; `law_sweep.csv` has one row per X with `X = 4` marked |
| 5 | `test_feature_files_and_split_layout` | on the corpus, `comparators.py savoldi --null-perm 5 --n-estimators 10 --null-splits loko`: the feature files carry every key of Part 1.6, one row per ok cell in cells.csv order, idle rows `IDLE` / `idle`, `W = H = -1`, `grid_id = Wall_Hall`, `wapf_norm = ""`; `scores.json` `params.grid_id == "Wall_Hall"`, `feature_count == 2`, `params.gc_verdict` null; within-trace `not applicable: one window per cell`; the raw run in `loko__archetype__raw/`; with a cell's `failed_verdict` set to a refusal in `preconditions.csv` and its id in `preconditions.json` `excluded_cells_pair_rungs`, the cell is absent from `cmp_dhodapkar`'s and `cmp_law`'s `predictions.csv` and present in `cmp_savoldi`'s (the `PAIR_RUNGS` line), and `excluded_pair_rung` reads `true` in the two per-cell CSVs; `check_x2_equals_K` is `true` on every cell |
| 6 | `test_gates_for_comparators` | after APF's split runs at `W8_H4` in both variants (LOKO/archetype, LORO/kernel), a hand-written `selection.json` and `gm.params.json` (`spread = 0.04`), `comparators.py all`: `gl.csv` has one part (i) row per comparator (at five permutations it reads B1-G1's `not run: 5 permutations < 500`, as `gate_gl` does for a rung, with the score and the null p95 in their columns; the pass / level-only reading is checked on the same scores with the null's verdict set; section 5 item 8), `gl.params.json` `part_ii`; `gdim.csv` rows `cmp_<name>` and `cmp_<name>__raw` with `full vector`, `d` equal to the rung's effective feature count (the re-run's when B1-G3 quarantined a feature) and `d_matched = d`; `gm.csv` rows with `rung_b` in (`apf`, `apf__raw`), a verdict in (`beats`, `difference with margin`) on LOKO and the split's own `not applicable: one window per cell` on within-trace (al-Farabi 5.5); `gx.csv` rows for the three comparators (`not applicable: one campaign label` on a one-campaign corpus); `verdicts.csv` has 30 rows with the raw rows' `gl` = `level-inclusive (as published)`; without `gm.params.json` the LOKO G-M verdicts read `not run: gates/gm.params.json missing (move 12)`; a comparator whose split directories are removed gets `not run:` rows and the params blocks stay |
| 7 | `test_table7_comparators_and_summary_table` | on the report fixture: `table7_comparators.csv` has 30 rows with the 16 columns of Table 7, every score cell `not run: gates/splits/cmp_<name>/Wall_Hall/<split>__<ls>[__raw]/scores.json missing (move 14)`, the fixed `G-C`, `G-F (i)` and `resolution` strings, the rung texts (`Savoldi 2010 (as published; U = mean +/- SD of K)`, `Dhodapkar-Smith 2003 (as published; delta_th = 0.04, declared default)`, `Law 2010 (level-normalized; X = 4, declared default)`), the raw row's `G-L` `level-inclusive (as published)` and the norm row's `not run: gates/comparators/gl.csv missing`; `table7.csv` still has 30 rows; with a hand-written LOKO norm `scores.json` carrying a quarantine the row prints the re-run's numbers through `effective_scores` (`0.600`, `0.450`, `rank 470 of 500`) and `feature count` = `feature_count_used` = 1; `_gm_text` with `rung_b = "apf__raw"` and its bare `not applicable` form; `table_comparators.csv` (`schema.KERNELS` order then `idle`) prints `not run: gates/comparators/<file> missing (move 14)` without the files and the medians with hand-written `*_per_kernel.csv` |
| 8 | `test_figure_dhodapkar_sweep` | without the sweep, `figures.run(out, ["dhodapkar_sweep"])` writes the placeholder PNG and PDF and `figures.json` `status == "ok"`; with a hand-written sweep (two kernels, two reps, the ten grid points) the return carries `n_panels == 2`, `n_cells_drawn == 4`, `default == 0.04` |
| 9 | `test_driver_move14_plan_and_run` | `parse_moves("0-14") == range(15)`, `parse_moves("15")` raises; `build_plan` on the epoch-1 Namespace lists the seven move-14 commands in order with `--delta-th-default 0.04`, `--x-default 4`, `--rows all_after_head_drop`, `--n-estimators` and `--seed-offset`; the flags reach the commands when set; `run_moves.main(["plan", ..., "--moves", "14"])` exits 0; on the corpus (preconditions and G-K0 by function call) `run --moves 14 --null-perm 5 --null-splits loko` exits 0 with every move-14 command `done`, the six LOKO rows of `table7_comparators.csv` print a number and `G-M vs APF` reads `not run: gates/gm.params.json missing (move 12)`, `manifest.json` lists `gates/comparators/verdicts.csv`, the figure exists; a second identical run skips all seven |
| 10 | `test_cli_exit_codes` | `savoldi --out <empty>` exits 2 naming `cells.csv`; `all --no-splits` exits 0 and writes the six per-cell and per-kernel CSVs, the two sweeps, `law_cells.json`, the three params files and the six feature files, and no split directory; an `--x-default` off `--x-grid` exits 2 (`missing input: x_default 4 is not on the grid ...`) and `--off-grid-default append` records the appended grid with `x_grid_source = "CLI --x-grid"`; a kernel whose cells are not admissible leaves the per-kernel summary and its rows say `admissible = false` |
| 11 | `test_chain_with_law_on_builder1_pipeline` | `synth corpus --reps 2 --idle 2 --n-pairs 60` -> `extract index` -> `extract all` -> preconditions (`c1_activity_min = 0.0`) -> `comparators all --null-perm 5 --n-estimators 10 --null-splits loko --jobs 2`, every step a subprocess: every file of Part 1.2 to 1.6 exists, 26 series files, `check_x2_equals_K` true and `status = ok` on all 26 cells, 26 x 4 and 26 x 10 sweep rows, 30 verdict rows, every split directory of both variants, G-X `not applicable: one campaign label`, only P2 Table 3's kernels and the idle cells appear under `gates/comparators/` |

The corpus of tests 5, 6, 9 and 10 is a `synth.py` corpus with trajectories (six kernels x two
reps plus two idle cells at 48 pairs, the presets' `K0` and `pulse_extra` divided by four so the
trajectories write and stream fast; the comparators do not depend on the level), indexed and
extracted by `extract.py` by function call, built once per module and copied per test (section 5
item 9).

## 4. How to run

```
cd VM_sampler/VM_Capture_QEMU
python3 -m plan11_encoding_ladder.run_moves run --out <out> --moves 14 --null-perm 500 --n-jobs 4 --comparator-jobs 4
python3 -m plan11_encoding_ladder.comparators all --out <out> --no-splits          # the statistics and the feature files only
python3 -m plan11_encoding_ladder.comparators savoldi   --out <out> [--rows all_after_head_drop|rung_series] [--ddof 1] [split flags] [--no-splits]
python3 -m plan11_encoding_ladder.comparators dhodapkar --out <out> [--grid 0.04,0.1,...] [--delta-th-default 0.04] [--boundary-rule gt|ge]
                                                       [--phase-length-rule n_over_b_plus_1|interior] [--off-grid-default refuse|append] [split flags] [--no-splits]
python3 -m plan11_encoding_ladder.comparators law       --out <out> [--x-grid 2,4,8,16] [--x-default 4] [--x-unit dumps|pairs]
                                                       [--head-drop-rule runs_from_seq_first|reset_at_head_drop] [--feature-source window|ever]
                                                       [--jobs 1] [--force] [--only REGEX] [--off-grid-default refuse|append] [split flags] [--no-splits]
python3 -m plan11_encoding_ladder.comparators gates     --out <out> [--null-perm 500] [--n-jobs 1] [--n-estimators 300] [--seed-offset 0]
python3 -m plan11_encoding_ladder.tables --out <out> --only table7_comparators,table_comparators
python3 -m plan11_encoding_ladder.figures --out <out> --only dhodapkar_sweep
cd plan11_encoding_ladder && python3 -m pytest -q tests                              # the whole suite
python3 -m pytest -q tests/test_comparators.py                                       # builder A's eleven tests
```

The split flags are `--null-perm 500 --null-splits loko,loro,within_trace --n-jobs 1
--n-estimators 300 --seed-offset 0`. Exit codes: 0 on success (a written refusal included), 2 when
`cells.csv` is missing (its path on stderr) or a declared default is not a grid point, 1 on an
internal error. `all --no-splits` skips the gates with the split stage they read.

## 5. Deviations from SPEC_epoch2.md and the review's corrections, each with the reason

1. **`schema.py` gains `COMPARATOR_DECLARED_DEFAULTS` inside the `COMPARATORS` block** (Part 1.0
   lists four names). Reason: al-Farabi 2 (a) / 5.4 requires Table 7 to print `declared default`
   only when the value that ran equals the declared one, and `tables.py` never imports
   `comparators.py` (Part 1.1), so the declared values must live in a module both read.
   `comparators.py` asserts its own `DHODAPKAR_DEFAULT` and `LAW_X_DEFAULT` equal them.
2. **The record follows the value (al-Farabi 5.4, applied).** `dhodapkar.params.json` carries
   `default_source` (`AA 2026-09-17: 0.04 marked as the default` only when the default is 0.04, else
   `CLI --delta-th-default <v> (departs from AA 2026-09-17's 0.04)`), `default_is_declared`,
   `grid_source` (the module constant or `CLI --grid`), `default_appended_to_grid`; `law.params.json`
   likewise (`x_default_source`, `x_grid_source`); Table 7 prints `declared default` or `default set
   by --delta-th-default` / `--x-default`, and the figure's legend follows the same rule.
3. **A default off the grid is refused, and the refusal is a parameter (al-Farabi 2 (b), his
   preference).** `--off-grid-default refuse|append`, module constant `OFF_GRID_DEFAULT_RULE =
   "refuse"`: `refuse` exits 2 with `missing input: delta_th_default <v> is not on the grid ...`;
   `append` adds the default to the grid and records `default_appended_to_grid = true`. Listed for
   the author (section 7 item 13).
4. **The two report steps of move 14 carry `gates/preconditions.json` beside their own input.**
   Part 1.7 (c) gives `inputs=["gates/comparators/dhodapkar_sweep.csv"]` for the figure and
   `["gates/comparators/verdicts.csv"]` for the manifest; the existing
   `tests/test_driver.py::test_move_table_carries_the_review_corrections` asserts that every
   non-internal command from move 3 on (alias and the skeleton aside) declares the admissibility
   record (CERT 7.1), and that test may not be amended, so `ADMISSIBILITY` is appended to both
   `inputs` lists. Nothing else in the block departs from the text.
5. **`run_plan(max_move=...)` defaults to `MAX_MOVE`.** A concurrent session (the detection
   paper's driver, `SPEC_DETECTION.md` 1.3) had given `parse_moves` and `run_plan` a `max_move`
   keyword with the default 13 before my edit; with the spec's `MAX_MOVE = 14` on `parse_moves`
   alone, `run --moves 14` would still have been rejected inside `run_plan`. The bound is builder
   A's region ("MAX_MOVE and the bound in parse_moves"); the one keyword default is the same
   bound, so it reads `MAX_MOVE` too. Callers that pass their own `max_move` are unaffected.
6. **Test 2's tolerance on the generator's J is 1e-9, not 1e-12.** The extract writes `J` with
   ten significant digits (SPEC 2.2, `format(x, ".10g")`), so the module's `delta = 1 - ex["J"]`
   matches the generator's J only to that precision; the identity the module implements
   (`delta == 1 - ex["J"]`, and the quantiles and mean of that delta) is asserted to 1e-12.
7. **Test 2's boundary count is twice the number of pulse boundaries.** Part 1.10 test 2 says
   `n_boundaries` at `delta_th = 0.3` "equals the number of truth boundaries"; under the
   generator's dynamics (SPEC 5.1: the pulse set A is lit for one snapshot and dropped at the next)
   the J dip toward `K0 / (K0 + A) = 1/3` appears on the pair entering the boundary and on the
   pair leaving it, so the method sees two phase changes per pulse. The test asserts
   `count(delta > 0.3) == 2 * n_boundaries == n_boundaries_at_0.3`, with the mechanism in a
   comment. The module is unchanged; only the expectation was wrong.
8. **Test 6's G-L (i) verdict at five permutations is `not run: 5 permutations < 500`.** Part 1.10
   test 6 expects `pass` or `level only`; B1-G1 at five permutations reads `not run: 5 permutations
   < 500` (SPEC 3.7.1, `models.b1_g1_verdict`) and `gates_comparison.gate_gl` inherits that
   string for a rung, so the comparators' G-L (i) inherits it too (the same reading, as Part 1.5
   demands). The test asserts the inherited string with the numbers in their columns and checks the
   pass / level-only reading on the same scores with the null's verdict set to `pass`. Likewise
   `gdim.csv`'s `d` is the effective feature count (the re-run's when B1-G3 quarantined a feature,
   as `gate_gdim` reads it), which the test reads from `models.effective_scores` rather than
   assuming 4 and 3.
9. **Tests 5, 6, 9 and 10 use a `synth.py` corpus with trajectories, not `_b2_common.corpus`.**
   The in-test generator writes extracts only; `cmp_law` needs the trajectory, so on that corpus
   the pair-rung exclusion of Law (test 5) would have been vacuous. The corpus is built by function
   call (`synth.write_cell`, `extract.build_index`, `extract.extract_all`) once per module.
10. **Test 3's refusing case runs through `law_stream` and `_law_worker`, not through
    `comparators law` on the corpus.** `extract.py` refuses the same `seq_reverse` cell first, so
    such a cell never has an extract and never reaches the comparator through the normal path; the
    worker's refusal (`refused: seq not monotone at row <n>`, the stale series file deleted) is
    exercised directly.
11. **`reset_at_head_drop` resets the four arrays, not two, and records from `head_drop + L - 1`.**
    Part 1.4's pseudo-code resets `run_m` and `run_n` and records from `t >= head_drop and t >= L -
    1`; under that literal form the `ever` counts would mix the dropped head into a per-cell
    statistic whose window statistics exclude it, and the first L - 1 recorded windows after the
    restart would be all-zero counts. Test 3's own expectation ("the first recorded dyn at X = 4
    equals the brute force over sets 5..7 only") is met by the restart of all four arrays and a
    `t_first` of `head_drop + L - 1`. The default rule `runs_from_seq_first` is exactly as written.
12. **`gates/comparators/gm.csv` carries the split's own `not applicable:` string (al-Farabi 5.5,
    applied)**, and `tables._gm_text` prints such a comparator verdict alone (there is no margin to
    print); for the rungs the helper's output is byte-identical to before.
13. **`_feature_count_text` on both branches of `table7` (al-Farabi 5.6, applied)**; before builder
    B's `feature_count_used` lands the fallback prints `feature_count` as today.
14. **`law.csv` carries the default-X statistics beside the identity columns** (Part 1.4 names
    the identity columns, `head_drop`, `check_x2_equals_K`, `status`); a superset, so the author
    reads the default row without opening the sweep.
15. **`all --no-splits` skips `gates` as well.** `gates` reads the split stage and G-X fits a
    forest; the inspection path is "the statistics and the feature files only" (Part 1.6).
16. **The Law pass refuses a page index outside `0 .. N-1`** (`refused: page_index <p> outside
    0..N-1 at seq <s>`): the run-length arrays are defined on N pages, and such a row would mean
    the cell's N is not the instrument's. The extractor does not check this; the pass records it
    as a written refusal rather than absorbing it.
17. **Part 1.10 test 4's `WeakSet` of arrays (al-Farabi 5.2, applied):** the pass wraps each
    snapshot's page array in `_PageList` (as `extract.Snapshot`) and tracks the wrappers in
    `_LIVE_PAGE_ARRAYS`; the hook receives the wrapper.

## 6. Open items in the other builder's regions, and the concurrent sessions

1. **Builder B's flags on the driver.** `--n-estimators` (B25) was not on the driver when test 9
   was written; the test passes it only when `run_moves._add_run_args` knows it (it has since
   landed) and otherwise runs the comparators at the SPEC's 300 trees. `cmp_common` reads
   `getattr(o, "n_estimators", 300)` and `getattr(o, "seed_offset", 0)` as Part 1.7 says.
2. **Two other sessions edited this tree while this build ran.** A session executing the
   superseded three-builder draft (`SPEC_epoch2.three_builders_0407.md`, which
   `SPEC_epoch2_review_al_farabi.md` says is not under review) wrote a package first named
   `comparators/` (which shadowed `comparators.py` for about twenty minutes, then was renamed
   `comparators_modules/`), plus `tables_eusipco.py`, `latex_skeleton_eusipco.py`,
   `tests/test_comparators_modules.py`, `tests/test_eusipco.py`, `tests/fixtures_eusipco/`, its
   report `BUILD2_comparators.md`, a runbook section "The EUSIPCO outputs", C1 flags named
   `--c1-rule` / `--c1-abs-fraction` on the driver and in `gates_precondition.py`, an argv-based
   staleness rule in the driver, and a test class in `tests/test_driver.py` (which Part 3.2
   reserves to builder A's two amendments). A second session (the detection paper,
   `SPEC_DETECTION.md`) added `gates_detection.py`, `detection_metrics.py`, `detection_levels.py`,
   `detection_splits.py`, `classes.py`, `run_detection.py`, `synth_detection.py`,
   `tables_detection.py`, `figures_detection.py`, `latex_skeleton_p3.py`, `RUNBOOK_DETECTION.md`,
   `tests/detection_fixtures.py`, `tests/test_report_detection.py`, `tests/test_run_detection.py`
   and the `max_move` / `ledger_name` keywords of the driver. The full-suite run of section 8
   collected every test file present at that moment, those sessions' included. Two
   implementations of the comparators therefore exist side by side: this one (`comparators.py`,
   move 14, `gates/comparators/`, the SPEC_epoch2 interfaces) and `comparators_modules/` (its own
   CLI and layout). They no longer collide at import, and the driver schedules only move 14, but the
   author should keep one; if this one is kept, `comparators_modules/`, `tables_eusipco.py`,
   `latex_skeleton_eusipco.py`, their tests and fixtures, and the runbook's EUSIPCO section are the
   other session's to remove or re-point. None of those files was read for this build beyond
   identifying the collision, and none was edited.
3. **The suite's wall time.** The machine ran three or four sessions' test suites at once (load
   average 450 to 580 on the reference machine); the baseline suite that took 3 min 46 s on
   2026-09-17 took 46 min at the start of this build, and the final run below took what section 8
   shows. The durations say nothing about the toolkit.
4. **al-Farabi's section 6 item 3 (a changed flag on resume)** was implemented by the
   three-builder session's driver work (`stale: arguments changed`, cost flags excluded), not by
   this build; move 14's commands carry every comparator flag explicitly, so the rule covers them.

## 7. For the author (every choice left as a parameter, with its default and where it is recorded)

1. `savoldi_rows = "all_after_head_drop"` (`--rows`; Part 4 item 1): every extract row after the
   head drop, the last `seq` included (K is defined on every row). Alternative `rung_series`.
2. `savoldi_ddof = 1` (`--ddof`; Part 4 item 2): the sample SD (C14's wording). Alternative 0.
3. The Dhodapkar-Smith grid `(0.04, 0.1, ..., 0.9)` with `delta_th_default = 0.04` (`--grid`,
   `--delta-th-default`; Part 4 item 3; AA 2026-09-17). Every point is in `dhodapkar_sweep.csv`;
   `params.default_source` says whether the value that ran is the declared one. One observation
   from the synthetic corpus, not from data: at steady state `delta = 1 - J` is about
   `2c / (1 + c)` for a churn fraction `c`, so `0.04` sits at the steady-state delta of a
   two-percent churn and flags about half of those pairs by noise. On the real corpus, read
   `delta_q50` per kernel in `dhodapkar.csv` against 0.04 before trusting the default's boundary
   counts; the sweep shows the alternatives.
4. `dhodapkar_boundary_rule = "gt"` (`--boundary-rule`; Part 4 item 4). Alternative `ge`.
5. `dhodapkar_phase_length_rule = "n_over_b_plus_1"` (`--phase-length-rule`; Part 4 item 5):
   `n / (B + 1)`. Alternative `interior`: a phase runs from one boundary pair to the pair before the
   next, the interior mean is `(b_B - b_1) / (B - 1)`, NaN below two boundaries.
6. `law_x_default = 4` on the grid `(2, 4, 8, 16)` (`--x-default`, `--x-grid`; Part 4 item 6).
   The definition names no X; `params.x_default_source` records `SPEC_epoch2 Part 4 item 6 (the
   author has declared no X)` until the author declares one (al-Farabi 6.10).
7. `law_x_unit = "dumps"` (`--x-unit`; Part 4 item 7): "dynamic in X" is a run of length `X - 1`.
   Alternative `pairs` (a run of length X).
8. `law_head_drop_rule = "runs_from_seq_first"` (`--head-drop-rule`; Part 4 item 8). Alternative
   `reset_at_head_drop` (section 5 item 11 says what restarts).
9. `law_feature_source = "window"` (`--feature-source`; Part 4 item 9). Alternative `ever`; both
   statistics are in `law_sweep.csv`.
10. `comparator_norm_rule = "median_K"` (Part 4 item 10): Savoldi's and Law's count features over
    the cell's median K; Dhodapkar-Smith's counts over the pair count. What each normalized row
    means is in section 1 and in every `params.json`.
11. `TABLE7_VARIANT = "both"` (`tables.py`; Part 4 item 11): the raw row then the norm row per
    comparator and split, 30 rows. Alternatives `raw`, `norm`.
12. G-M's `spread` is read from `gates/gm.params.json` (move 12), never re-measured (Part 4 item
    12); until move 12 has run the cell reads `not run: gates/gm.params.json missing (move 12)`.
13. `OFF_GRID_DEFAULT_RULE = "refuse"` (`--off-grid-default`; al-Farabi 6.5): a declared default
    that is not a grid point is refused with exit 2; `append` adds it and records the append.
14. The per-kernel summaries are over admissible cells (`all_hard_pass` true; every cell when
    `preconditions.csv` is absent), and Savoldi's `U_text_median` is the `U_text` of the cell whose
    `K_mean` is the lower median (recorded as `params.per_kernel_rule`).
15. The Table 7 `G-C` string of a comparator row is `not applicable: comparator, not a lead of
    the ladder (...)` as Part 1.5 gives it; al-Farabi 6.4 offers the `inherited: G-C of apf` /
    `of persist` reading for Savoldi and Dhodapkar-Smith. A string change in `table7_comparators`
    only (`tables.CMP_GC_TEXT`); not made.
16. The Dhodapkar-Smith normalized row has one degree of freedom under the default phase-length
    rule (`boundary_rate = 1 - stability`, `mean_phase_frac = 1 / (B + 1)`; al-Farabi 6.9;
    `params.feature_note`): expect B1-G3 to quarantine two of its three features and the row to be
    a one-feature re-run. On the synthetic corpus B1-G3 quarantined one feature of Law's four and
    of Dhodapkar-Smith's three in the norm variant (test 6 reads the effective counts).
17. Gates not computed for comparators in this epoch (Part 4 item 13): G-J, G-V, the clustering,
    G-F (ii), G-P, G-DEC, G-C, the temporal gates and G-ORD; each prints its `not applicable:`
    string in Table 7, and `verdicts.params.json` lists them under `not_applied`.
18. The Law pass's cost on the real corpus was not measured here (no real data on this machine);
    the runbook carries the extractor's rate as the estimate, one to three minutes per cell per
    process, resumable per cell.

## 8. The full suite, verbatim (`python3 -m pytest -q tests` from `plan11_encoding_ladder/`)

SUITE_OUTPUT_PLACEHOLDER
