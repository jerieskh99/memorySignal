# SPEC_epoch2.md: build epoch 2 of the paper 2 analysis toolkit (addendum to SPEC.md)

Written 2026-09-17. Two builders implement this in parallel without talking to each other:
builder A (Part 1, the comparators) and builder B (Part 2, the fix pass). Part 3 fixes the
boundary between them; Part 4 lists every choice left to the author. SPEC.md stays the source of
the interfaces; where the code as built differs from SPEC.md, section 0.3 below says so and the
code wins. Nothing in SPEC.md is repealed by this file except where a Part 2 item says "replaces".

Baseline this epoch starts from (verified on this machine 2026-09-17, `python3 -m pytest -q tests`
from the package directory): 158 passed, 1 skipped (the `zstandard` reader path), 3 min 46 s.
CHECK_3.md: no blocking findings. CERTIFY_al_farabi.md: the five conditions hold. Neither may
regress: every one of those 158 tests still passes at the end of each builder's work, every
threshold and verdict string of SPEC 3.0 stays as defined, every grid point stays computed and kept.

## 0. Binding rules and conventions for both builders

### 0.1 Rules (verbatim from the epoch brief; binding)

1. Server: forbidden. No ssh, scp, rsync; no path under `/mnt/nfs` or `/project`; no remote
   command. All tests use synthetic data from the toolkit's own generator (`synth.py`, or the
   in-test generator `tests/_synth_b2.py` where a trajectory is not needed).
2. The sandbox family: name only. Its workloads and sources are not read, grepped, opened,
   summarized or named. Nothing in this epoch needs them.
3. No paper prose. No sentence of the paper body anywhere. The LaTeX skeleton, if touched, stays
   headings, environments, placeholders and comment blocks.
4. No fabrication. Every comparator and every gate cites, in its docstring, the definition it
   implements (the council file and candidate number, or `P2_STRUCTURE.md` section V, or
   `P2_AUTHOR_ANSWERS.md` item). Where a definition leaves a choice, the choice is a parameter with
   the default given here and listed in Part 4; nothing is chosen silently.
5. Plain register in reports: full sentences.
6. No git commit. Edits only under `plan11_encoding_ladder/`. `P2_STRUCTURE.md`,
   `P2E_STRUCTURE.md`, `P2_AUTHOR_ANSWERS.md` and every council file are read, never edited.
   `apf_paper/EPOCH2_BUILD_REPORT.md` is written by the epoch's reporter after both builders finish,
   not by a builder (Part 3.4).
7. Python 3.10+, numpy, scikit-learn, scipy optional, matplotlib for figures; pytest for tests
   (`python3 -m pytest -q tests` from `plan11_encoding_ladder/`). Streaming over trajectories: a
   trajectory is never loaded whole. Each builder runs the full suite before returning; a change
   without a passing test is not done.
8. No regression of CHECK_3 or CERTIFY_al_farabi: see the baseline above.

### 0.2 Conventions of this epoch

- Citations in docstrings use these short names: `C14 cand. N` = `apf_paper/council/14_hunayn_exact_input_comparators.md`,
  section "Candidates, ranked", candidate N; `C14 author` = its "For the author" list;
  `P2 Sec. 0 Baseline` = `apf_paper/P2_STRUCTURE.md` section 0, row "Baseline";
  `P2 Sec. V` = `P2_STRUCTURE.md` section V (gates, splits, models); `P2 Sec. 4 Table 7`;
  `P2E Sec. 7` = `apf_paper/P2E_STRUCTURE.md` section 7; `AA T1` = `P2_AUTHOR_ANSWERS.md`,
  "Build epoch 1: the author's decisions", item T1; `AA 2026-09-17` = its "Decisions of 2026-09-17
  (for build epoch 2)"; `CHECK_3 Mn`; `CERT 1(a)` etc. = `CERTIFY_al_farabi.md` section 1 residual
  mark (a); `CERT 7.n` = its section 7 item n; `E1 4` / `E1 6.n` = `apf_paper/EPOCH1_BUILD_REPORT.md`
  section 4 / section 6 item n.
- Every new result file has the three top-level keys `schema` (`"plan11.<name>.v1"`), `params`,
  `citation` (SPEC section 1). CSV-only results get `<stem>.params.json` beside them through
  `series.write_params` (the existing naming: `gates/gm.csv` has `gates/gm.params.json`).
- Refusal strings come from `verdicts.py` (`not_run`, `not_applicable`, `refused`, the named
  constants). A cell that is not a number names what is missing (SPEC section 7).
- Verdict strings never change. No new verdict constant is added to `verdicts.py` in this epoch;
  the new strings are `not applicable: ...` / `not run: ...` forms.
- Shared files are edited with targeted replacements (the Edit tool), never rewritten whole, so that
  the other builder's concurrent edits to other regions of the same file survive. Part 3 names the
  regions.
- `run_moves.build_plan(o)` reads every namespace attribute added in this epoch with
  `getattr(o, "<name>", <default>)`, because `tests/test_driver.py::_ns` builds a Namespace without
  them. This applies to both builders.
- New keyword parameters on existing functions get defaults; no existing positional parameter,
  keyword name, return type or file layout changes (Part 3.3 lists the frozen signatures).
- Tests: builder A's tests go in `tests/test_comparators.py` (new); builder B's new tests go in
  `tests/test_epoch2_fixes.py` (new), `tests/test_driver_end_to_end.py` (new),
  `tests/test_runner_guard.py` (new), plus additions to the existing gate test files named per item.
  `tests/test_driver.py` is edited by builder A only, and only as Part 1.7 says.
- When the full suite fails in a test the other builder owns and the failure is not caused by your
  change, re-run the suite once after a pause; report the failure in your build report if it
  persists. Never edit the other builder's tests.

### 0.3 The code as built, where it supersedes SPEC.md (read before coding)

1. The driver is `run_moves.py`; `driver.py` is a one-line alias. Its move table differs from SPEC
   section 7: `gates_calibration gp` runs at move 2; `alias` runs at move 6 and again at move 7;
   `gx <rung>` runs at moves 7, 9, 10, 11, 12; G-C runs for five rungs at move 3 (combined last);
   move 12 runs `gl` again ("gl (all rungs)") and `gf --all-rungs` at the selected points before
   `tables`; move 13 is the internal `gf-check` on Table 7. `parse_moves` accepts 0 to 13.
2. The split stage's raw variant lives in `gates/splits/<rung>/<grid_id>/<split>__<labelspace>__raw/`
   (`models.split_dir(..., normalized=False)`); SPEC 4.5's path has no raw segment.
3. `scores.json` carries, beyond SPEC 4.5, `feature_count`, `dim_status`, `min_train_cells`,
   `null_summary`, `score_source`, `predictions_file`, `headline_classes`, `excluded_row`; a B1-G3
   re-run writes `predictions_with_quarantine.csv` beside `predictions.csv`; every consumer reads
   through `models.effective_scores` (CHECK_2 B1).
4. `gates/gx.csv` columns are `rung, grid_id, score, null_p95, rank, leak_verdict, confound_verdict,
   headline_mark, n_labels`; `gate_gx(out, rung, ..., grid_id=None)` replaces only the rows of its
   rung and merges `gates/gx.json` per rung.
5. `gates/gdim.csv` columns are `rung, grid_id, d, d_matched, method, status, loko_score, matched_to`;
   the matched run lives under `gates/splits_matched/combined/<gid>/loko__archetype/` (LOKO only, CHECK_3 M4).
6. `gates/gm.params.json` holds `params.spread` (the measured max minus min of the APF LOKO score over
   the five seeds) and `params.seed_scores`; `gates_comparison.gm_compare(scores_a, scores_b, spread)`
   is a pure function.
7. `series.build_features` writes each row's `archetype` from `cells.csv` (`control` for idle cells,
   the value `schema.parse_cell_path` assigns) and `kernel` from the label-derived name; SPEC 3.1.5
   says `IDLE` and `idle`. Part 2 item B24 makes the file follow SPEC.
8. `tables.py` writes `.csv`, `.md` and `.tex`; `TABLE7_COLUMNS` are `rung, resolution (W x H),
   feature count, split, label space, accuracy, macro recall (headline rows), null p95, rank,
   majority, G-C, G-F (i), G-L, G-DIM, G-M vs APF, G-X` (16 columns); `table7.csv` has 30 rows
   (six rungs including `combined (matched)` times three splits, 18 rows, plus six rungs times the
   two appended archetype-space rows for LORO and within-trace, 12 rows) and
   `tests/test_report.py::test_rows_columns_and_gm_text` asserts that count, which is why the
   comparator rows go to a separate file (Part 1.8).
9. `_report_common.load_selection` and `selected_grid` return entries for the five rungs only.
10. `models.py splits --rung` is restricted to the five rungs on the CLI; the function
    `run_split_stage` accepts any rung string and reads `features/<rung>/<grid_id>_{raw,norm}.npz`.
11. `gates_precondition preconditions` already has `--c1-activity-min` (a fraction of N); the driver
    does not carry it (CHECK_3 M1).
12. `figures.py` keeps `FIGURE_NAMES`, writes a placeholder PDF/PNG through `_placeholder` for a
    figure that cannot be drawn, and `figures.json` `status` is `"ok"` only when no figure raised.
    `tests/test_report.py::test_all_figures_written` asserts every name in `FIGURE_NAMES` has a PDF and
    a PNG after `figures.run(out)` on the fixture and that `status == "ok"`.
13. `latex_skeleton.py` uses `\IfFileExists{figures/fig_<stem>.pdf}{\includegraphics...}{placeholder}`
    and `\input{tables/<name>.tex}`; `tests/test_report.py::test_targets_exist_and_no_prose` asserts
    every target exists after `tables.run(out)` and `figures.run(out)`.
14. `series.admissible_cells(out, cells, rung)` applies the pair-rung exclusion (`failed_verdict`
    refused) only when `rung in series.PAIR_RUNGS`.
15. `tests/test_driver.py::_ns` builds the driver Namespace with the epoch-1 keys only (section 0.2).
16. G2's seconds coverage and G-P's `T_seconds` use the constant `schema.DURATION_S = 600`, not the
    cell's declared duration (`gates_temporal.py` `T = DURATION_S / entry.passes`;
    `gates_calibration.py` line 126). Part 2 item B12 changes that.

---

## Part 1. The comparators (builder A)

### 1.0 Scope and names

Three exact-input comparators, each per `C14`: Savoldi 2010 (`C14 cand. 1`), Dhodapkar and Smith
2003 (`C14 cand. 3`), Law 2010 (`C14 cand. 2`). `P2 Sec. 0 Baseline` names all three; `P2E Sec. 7`
assigns Savoldi and Dhodapkar-Smith to the EUSIPCO table and holds Law for the IFIP version;
`AA 2026-09-17` accepts the picks and declares the Dhodapkar-Smith sweep with 0.04 as the default.

Internal names (directory-safe, never confused with a rung): `cmp_savoldi`, `cmp_dhodapkar`,
`cmp_law`. Builder A appends to `schema.py`:

```python
# Exact-input comparators (build epoch 2, SPEC_epoch2.md Part 1; C14 candidates 1, 3, 2)
COMPARATORS: tuple[tuple[str, str], ...] = (
    ("cmp_savoldi", "Savoldi 2010"),
    ("cmp_dhodapkar", "Dhodapkar-Smith 2003"),
    ("cmp_law", "Law 2010"),
)
COMPARATOR_NAMES: tuple[str, ...] = tuple(k for k, _ in COMPARATORS)
COMPARATOR_DISPLAY: dict[str, str] = dict(COMPARATORS)
COMPARATOR_GRID_ID = "Wall_Hall"      # one row per cell: the whole-cell point, by definition
```

`series.RUNGS`, `_report_common.RUNGS` and every loop over them stay unchanged: no existing stage
picks a comparator up by accident. The one change to `series.py` that builder A makes is the
`PAIR_RUNGS` line (Part 3.2):

```python
PAIR_RUNGS = ("persist", "content", "combined", "cmp_dhodapkar", "cmp_law")   # epoch 2: the two comparators that read pair adjacency
```

so that `series.admissible_cells` keeps a cell whose `failed_verdict` is a refusal out of the two
comparators that depend on consecutive pairs (Dhodapkar-Smith reads J; Law reads runs across
consecutive changed sets) and keeps it in Savoldi (a per-snapshot count, defined on every row).

### 1.1 The module `comparators.py`

A gate-side module in builder 2's style. Imports allowed: `schema`, `verdicts`, `series`,
`nulls`, `splits`, `models` (`run_split_stage`, `split_dir`, `effective_scores`, `N_ESTIMATORS`,
`B1G1_MIN_PERM`), `gates_comparison` (`gm_compare`, `gate_gx`, `GL_COLUMNS`), and
`extract.open_text` (the reader) plus `extract.Refusal`. It never imports `tables`, `figures`,
`run_moves` or `_report_common`.

Docstring form (every public function):

```
"""<Method name and year>: <one-line definition in the method's own terms>.

Citation: <C14 cand. N, the exact sentence paraphrased: what it computes>; <P2 Sec. 0 Baseline>;
<P2E Sec. 7 or P2 Sec. 0 for the venue assignment>; <P2 Sec. V ... for every gate applied through
models.run_split_stage>; <AA 2026-09-17 where the author fixed a default>.
Parameters the definition leaves open: <name> = <default> (<alternatives>), SPEC_epoch2 Part 4 item <n>.
"""
```

Citation strings (module constants, used verbatim in `params["citation"]` and the docstrings):

```python
CIT_SAVOLDI = ("C14 cand. 1: Savoldi, Gubian, Echizen 2010, 'Uncertainty in Live Forensics', Advances in Digital "
               "Forensics VI, IFIP AICT 337, pp. 171-184, DOI 10.1007/978-3-642-15506-2_12: per consecutive pair the "
               "number of differing 4 KiB pages; U = mu_dmp +/- sigma_dmp, the sample mean and standard deviation of that "
               "count over the run (EXACT input: our per-pair K); P2 Sec. 0 Baseline; P2E Sec. 7 (the one-number baseline); "
               "P2 Sec. V 'The splits', 'Models' and 5.1 Plan 08 (B1-G1, B1-G6) through models.run_split_stage")
CIT_DHODAPKAR = ("C14 cand. 3: Dhodapkar and Smith 2003, 'Comparing Program Phase Detection Techniques', MICRO-36, pp. 217-227 "
                 "(and ISCA 2002): delta_{i,i-1} = (|W_i u W_{i-1}| - |W_i n W_{i-1}|) / |W_i u W_{i-1}| between consecutive "
                 "working sets, a phase change when delta exceeds a threshold, stability and average phase length; with W_i "
                 "our changed-page set the delta is one minus Jaccard identically (C14: 'Their delta is one minus Jaccard, "
                 "identically'); P2 Sec. 0 Baseline; P2E Sec. 7 (the named method on the overlap axis); AA 2026-09-17 "
                 "(a declared sweep, every point kept, 0.04 the default); C14 author (declare delta_th before the labels or "
                 "sweep it and show the sweep); P2 Sec. V through models.run_split_stage")
CIT_LAW = ("C14 cand. 2: Law et al. 2010, 'Identifying Volatile Data from Multiple Memory Dumps in Live Forensics', "
           "Advances in Digital Forensics VI, IFIP AICT 337, pp. 185-194, DOI 10.1007/978-3-642-15506-2_13: a page is "
           "dynamic in X consecutive dumps if X or more consecutive hashes differ, static if identical; an index of run "
           "lengths answers all X in one pass; in our rows 'dynamic in X' = a membership run of length X-1 in the changed "
           "sets, 'static in X' = a non-membership run of length X-1 (C14's exact reading); P2 Sec. 0 Baseline (Law and "
           "Savoldi as Table 7 rows for the IFIP version); P2E Sec. 7 (held for IFIP); P2 Sec. V through models.run_split_stage")
```

Constants (module level, every one recorded in `params`):

```python
SAVOLDI_ROWS = "all_after_head_drop"        # Part 4 item 1: "all_after_head_drop" | "rung_series"
SAVOLDI_DDOF = 1                            # Part 4 item 2: sample SD (C14: "sample mean ... and standard deviation")
DHODAPKAR_GRID = (0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # Part 4 item 3: the brief's nine points plus the author's declared default
DHODAPKAR_DEFAULT = 0.04                    # AA 2026-09-17; marked "declared default" everywhere it is printed
DHODAPKAR_BOUNDARY_RULE = "gt"              # Part 4 item 4: boundary when delta > delta_th ("gt") or delta >= delta_th ("ge")
DHODAPKAR_PHASE_LENGTH_RULE = "n_over_b_plus_1"   # Part 4 item 5: n_pairs / (n_boundaries + 1) | "interior"
LAW_X_GRID = (2, 4, 8, 16)                  # the declared grid, every point computed and kept
LAW_X_DEFAULT = 4                           # Part 4 item 6
LAW_X_UNIT = "dumps"                        # Part 4 item 7: "dumps" (run length X-1, C14's reading) | "pairs" (run length X)
LAW_HEAD_DROP_RULE = "runs_from_seq_first"  # Part 4 item 8: "runs_from_seq_first" | "reset_at_head_drop"
LAW_FEATURE_SOURCE = "window"               # Part 4 item 9: "window" (the per-pair series) | "ever" (distinct pages over the cell)
COMPARATOR_NORM_RULE = "median_K"           # Part 4 item 10: the count-rung rule of P2 Sec. V G-L (i) applied to the comparator features
```

(`TABLE7_VARIANT = "both"`, Part 4 item 11, lives in `tables.py`, which never imports this module.)

Every per-cell statistic is computed for every cell whose `cells.csv` status is `ok` and whose
`extract/<cell_id>/extract.csv` exists (the comparator does not re-decide admissibility; the split
stage applies it at read time through `models.prepare_split_data`, as for every rung). Each per-cell
row carries `admissible` (`true`/`false` from `gates/preconditions.csv` `all_hard_pass` when the file
exists, else `"preconditions not run"`) and, for `cmp_dhodapkar` and `cmp_law`, `excluded_pair_rung`
(`true` when the cell is in `gates/preconditions.json` `excluded_cells_pair_rungs`). Per-kernel
summaries are over admissible cells only.

Head drop: `hd = series.load_head_drop(out / "inputs" / "head_drop.csv")` and the per-cell value is
`series.head_drop_for(hd, "idle" if c["role"] == "idle" else c["kernel"])` (this form works before
and after builder B's item B9 and is the one builder A uses).

Idle cells: every feature file row of an idle cell carries `archetype = "IDLE"` and `kernel = "idle"`
(SPEC 3.1.5), whatever `cells.csv` says; kernel rows carry `archetype_predicted` from `cells.csv`
(G-K0's relabelling is applied at read time by `prepare_split_data`, never stored).

### 1.2 Savoldi 2010 (`cmp_savoldi`)

Definition implemented (`C14 cand. 1`): for each consecutive pair the number of differing pages;
`mu_dmp` the sample mean and `sigma_dmp` the sample standard deviation of that count over the run;
`U = mu_dmp +/- sigma_dmp`. In our rows the per-pair count is the extract's `K` (every `seq` row is
the differ's output for the pair ending at that snapshot; SPEC 2.1). Interval: ours is one snapshot
pair (about 0.644 s), theirs one acquisition time; `params["interval_note"]` records this sentence
of C14 ("Only the interval differs").

Per cell (`savoldi_cell(ex, head_drop, *, rows=SAVOLDI_ROWS, ddof=SAVOLDI_DDOF, n_pages=N) -> dict`):

- `rows = "all_after_head_drop"`: `K = ex["K"][head_drop:]` (every row, the last `seq` included:
  `K` is defined on every row, SPEC 2.2); `"rung_series"`: `ex["K"][head_drop:-1]` (the rows the
  rung series use, SPEC 3.1.1).
- `n_rows_used = len(K)`; `K_mean = mean(K)`; `K_sd = std(K, ddof=ddof)` (`ddof = 1`: the sample SD;
  with `n_rows_used < 2` the SD is NaN and the row's `status` reads `not run: fewer than two rows`);
  `K_median = series.k_median_cell(ex, head_drop)` (the number the rungs normalize by, SPEC 3.1.1);
  `U_mean_pct = 100 * K_mean / N`; `U_sd_pct = 100 * K_sd / N`;
  `U_text = f"{U_mean_pct:.4g}% +/- {U_sd_pct:.3g}%"` (C14's per-run form, "U = 65.5% +/- 0.15%").
- Raw feature vector (2): `(K_mean, K_sd)` in pages, names `cmp_savoldi.K.mean`, `cmp_savoldi.K.sd`.
- Level-normalized feature vector (2), the count-rung rule of P2 Sec. V G-L (i) (divide by the cell's
  own median K, SPEC 3.1.1 row `apf`): `(K_mean / K_median, K_sd / K_median)`, names
  `cmp_savoldi.k_over_med.mean`, `cmp_savoldi.k_over_med.sd`. What this row means, stated in the
  module docstring and in `savoldi.params.json` `params["norm_row_meaning"]`: the first feature is
  the mean over the median, a number near one that carries only the skew of the count distribution;
  the second is the relative spread; the normalized row therefore tests whether skew and relative
  spread alone carry the label, and the raw row is the method as published (level-inclusive by
  construction). Both rows are computed; which one Table 7 prints is Part 4 item 11.

Outputs:

- `gates/comparators/savoldi.csv`: `cell_id, kernel, role, archetype_predicted, rep, campaign,
  admissible, head_drop, rows_rule, n_rows_used, K_mean, K_sd, K_median, U_mean_pct, U_sd_pct,
  U_text, status`; `gates/comparators/savoldi.params.json` (schema `plan11.comparators.savoldi.v1`).
- `gates/comparators/savoldi_per_kernel.csv`: `kernel, n_cells, K_mean_median, K_mean_min, K_mean_max,
  K_sd_median, U_text_median` (the median cell's `U_text`), one row per kernel present plus `idle`
  when idle cells exist; over admissible cells.
- `features/cmp_savoldi/Wall_Hall_raw.npz` and `Wall_Hall_norm.npz` (layout in 1.6).
- The split stage (1.6).

### 1.3 Dhodapkar and Smith 2003 (`cmp_dhodapkar`)

Definition implemented (`C14 cand. 3`): `delta_{i,i-1} = (|W_i u W_{i-1}| - |W_i n W_{i-1}|) /
|W_i u W_{i-1}|` between consecutive working sets; a phase change when `delta` exceeds a threshold;
stability; average phase length. With `W_i` our changed-page set of pair `i`: `|W_i n W_{i-1}|` is
the extract's `n_persist`, `|W_i u W_{i-1}|` is `n_union`, so `delta = (n_union - n_persist) / n_union
= 1 - J` identically (SPEC 2.2 defines `J = n_persist / n_union`); the docstring cites C14's sentence
and shows this one-line derivation. The module computes `delta = 1 - ex["J"]`, never a second
intersection.

Per cell (`dhodapkar_cell(ex, head_drop, *, grid=DHODAPKAR_GRID, default=DHODAPKAR_DEFAULT,
boundary_rule=..., phase_length_rule=...) -> dict`):

- Rows: `J = ex["J"][head_drop:-1]` (the last `seq` has no J, SPEC 2.4); a blank J (both sets empty,
  NaN) is dropped from the pair count and counted in `n_pairs_blank`; `delta = 1 - J` over the kept
  rows; `n_pairs_used = len(delta)`.
- `delta_mean`, `delta_q05, delta_q25, delta_q50, delta_q75, delta_q95` (`numpy.quantile`, linear).
- For every `delta_th` in `grid` (all computed, all kept, none selected against labels; the default is
  the declared parameter and is recorded, not chosen): `n_boundaries = count(delta > delta_th)`
  (`"gt"`; `"ge"` counts `>=`); `stability = 1 - n_boundaries / n_pairs_used` (the fraction of pairs
  not flagged as a phase change); `mean_phase_length_pairs`: under `"n_over_b_plus_1"` the mean length
  of the `n_boundaries + 1` segments the boundaries cut the kept rows into, `n_pairs_used /
  (n_boundaries + 1)`; under `"interior"` the mean length of the segments between two boundaries
  (NaN when `n_boundaries < 2`). `is_default = (delta_th == default)`.
- Raw feature vector (3) at the default threshold: `(n_boundaries, stability, mean_phase_length_pairs)`,
  names `cmp_dhodapkar.n_boundaries`, `cmp_dhodapkar.stability`, `cmp_dhodapkar.mean_phase_length`
  (the method's three named outputs; two of them are algebraically tied and that is the method's own
  redundancy, recorded in `params["feature_note"]`).
- Normalized feature vector (3): the delta is level-free (a Jaccard), so no level division exists;
  the only per-cell scale the counts carry is the pair count, and the normalized row removes it:
  `(n_boundaries / n_pairs_used, stability, mean_phase_length_pairs / n_pairs_used)`, names
  `cmp_dhodapkar.boundary_rate`, `cmp_dhodapkar.stability`, `cmp_dhodapkar.mean_phase_frac`.
  `params["norm_row_meaning"]` says so.

Outputs:

- `gates/comparators/dhodapkar_sweep.csv`: one row per (cell, `delta_th`): `cell_id, kernel, role,
  rep, campaign, admissible, excluded_pair_rung, delta_th, is_default, n_pairs_used, n_pairs_blank,
  n_boundaries, stability, mean_phase_length_pairs`. Ten rows per cell; the sweep is written whole
  and never filtered.
- `gates/comparators/dhodapkar.csv`: the default-threshold row per cell with the delta statistics:
  `cell_id, kernel, role, archetype_predicted, rep, campaign, admissible, excluded_pair_rung,
  head_drop, n_pairs_used, n_pairs_blank, delta_mean, delta_q05, delta_q25, delta_q50, delta_q75,
  delta_q95, delta_th_default, n_boundaries, stability, mean_phase_length_pairs, status`;
  `dhodapkar.params.json` (schema `plan11.comparators.dhodapkar.v1`) with `grid`, `default`,
  `boundary_rule`, `phase_length_rule`, `default_source = "AA 2026-09-17: 0.04 marked as the default"`,
  and `grid_source = "SPEC_epoch2 Part 1.3: the brief's nine points 0.1 .. 0.9 plus the author's declared default"`.
- `gates/comparators/dhodapkar_per_kernel.csv`: per kernel at the default threshold:
  `kernel, n_cells, n_boundaries_median, stability_median, mean_phase_length_median, delta_q50_median`.
- `features/cmp_dhodapkar/Wall_Hall_{raw,norm}.npz`; the split stage (1.6).

### 1.4 Law 2010 (`cmp_law`)

Definition implemented (`C14 cand. 2`, followed where the brief's wording differs): a page is
dynamic in `X` consecutive dumps if `X` or more consecutive hashes differ, static if identical; in
our rows "dynamic in X" is a membership run of length `X - 1` in the changed sets and "static in X" a
non-membership run of length `X - 1` (C14's exact reading; `X` dumps span `X - 1` consecutive pairs).
The brief's alternative reading ("in every one of the X changed sets", a run of length `X`) is the
parameter `x_unit = "pairs"` (Part 4 item 7). C14 also says the index of run lengths answers all `X`
in one pass; that is the streaming form below, which holds two run-length arrays of length `N`
(and two maxima), never a page set and never more than one snapshot's page list. This is stricter
than "at most X page sets in memory" and is recorded in `params["memory_model"]`.

Input: the trajectory, not the extract (a second streaming pass in `extract.py`'s style, inside
`comparators.py`; `extract.py` is not edited). The trajectory path is `Path(cells.csv row["path"])
/ row["traj_file"]` (never a hard-coded root). Reader: `extract.open_text`. The row loop is copied
from `extract._stream` reduced to `seq` and `page_index` (comment `# copied from extract._stream,
2026-09-17, reduced to seq and page_index`): header located by name; a parse failure counted in
`n_rows_skipped`; a decreasing `seq` is `refused: seq not monotone at row <n>` and the cell's row
carries that `status`; a gap `seq` is an empty snapshot; duplicates of `page_index` within a `seq`
are dropped (`numpy.unique`).

The pass (`law_stream(traj_path, *, n_pages, x_grid, x_unit, head_drop, head_drop_rule) -> dict`):

```
run_m = zeros(N, int32); run_n = zeros(N, int32); max_m = zeros(N, int32); max_n = zeros(N, int32)
L(X) = X - 1 if x_unit == "dumps" else X            # the run length that makes a page dynamic/static for X
t = 0                                              # 0-based pair index from seq_first, gaps included
for each finished snapshot (sorted unique pages P; K = |P|), in seq order, gaps as empty P:
    if head_drop_rule == "reset_at_head_drop" and t == head_drop: run_m[:] = 0; run_n[:] = 0
    mask = zeros(N, bool); mask[P] = True
    run_m = where(mask, run_m + 1, 0); run_n = where(mask, 0, run_n + 1)
    max_m = maximum(max_m, run_m); max_n = maximum(max_n, run_n)
    for X in x_grid:
        if t >= head_drop and t >= L(X) - 1:      # the window of L(X) pairs ending at t exists
            dyn[X].append(count(run_m >= L(X))); sta[X].append(count(run_n >= L(X)))
    t += 1
at the end: dyn_ever[X] = count(max_m >= L(X)); sta_ever[X] = count(max_n >= L(X))
```

Under `"runs_from_seq_first"` (the default) the run-length state includes the dropped head and
recording starts at `t = head_drop`; under `"reset_at_head_drop"` the state restarts there. At
`X = 2` under `"dumps"` (`L = 1`) `dyn = K_t` and `sta = N - K_t`: the pass writes
`check_x2_equals_K` (`true` when every recorded `dyn[2]` equals the extract's `K` on the same rows,
else `false`) into the cell's record, a built-in consistency check against `extract.csv`.

Per cell and per `X`: `n_windows = len(dyn[X])`, `dyn_mean, dyn_sd (population), dyn_min, dyn_max,
dyn_frac_mean = dyn_mean / N`, the same five for `sta` plus `sta_frac_mean`, `dyn_ever, sta_ever,
dyn_ever_frac, sta_ever_frac`, `is_default = (X == x_default)`.

Feature vectors at the default `X`, `feature_source = "window"`: raw (4) `(dyn_mean, dyn_sd,
sta_mean, sta_sd)` in pages, names `cmp_law.dyn.mean`, `cmp_law.dyn.sd`, `cmp_law.sta.mean`,
`cmp_law.sta.sd`; normalized (4), the count-rung rule with `K_median = series.k_median_cell(ex,
head_drop)`: `(dyn_mean / K_median, dyn_sd / K_median, (N - sta_mean) / K_median, sta_sd / K_median)`
(`N - sta` is the union of the window's changed sets, a count), names `cmp_law.dyn_over_med.mean`,
`cmp_law.dyn_over_med.sd`, `cmp_law.union_over_med.mean`, `cmp_law.sta_over_med.sd`. Under
`feature_source = "ever"`: raw `(dyn_ever, sta_ever)`, normalized `(dyn_ever / K_median,
(N - sta_ever) / K_median)`, names with `ever` in place of the window statistic.

Outputs:

- `gates/comparators/law_sweep.csv`: one row per (cell, `X`): `cell_id, kernel, role, rep, campaign,
  admissible, excluded_pair_rung, X, run_length_L, is_default, n_windows, dyn_mean, dyn_sd, dyn_min,
  dyn_max, dyn_frac_mean, sta_mean, sta_sd, sta_min, sta_max, sta_frac_mean, dyn_ever, sta_ever,
  dyn_ever_frac, sta_ever_frac, status`.
- `gates/comparators/law.csv`: the default-`X` row per cell with the identity columns, `head_drop`,
  `check_x2_equals_K`, and `status`; `law.params.json` (schema `plan11.comparators.law.v1`) with
  `x_grid, x_default, x_unit, head_drop_rule, feature_source, memory_model`.
- `gates/comparators/law_cells.json`: per cell the pass counters (`traj_file, source_bytes,
  n_rows_in, n_rows_skipped, n_rows_dup_page, seq_first, seq_last, n_seq_gaps, elapsed_s, status`);
  a cell whose entry has `status == "ok"` is skipped on a re-run unless `--force` (the per-cell
  resume of `extract all`, SPEC 2.5).
- `gates/comparators/law_series/<cell_id>.npz`: `X` (int array), `dyn_<X>` and `sta_<X>` (int32 per
  recorded pair) for every grid point, `t_first` (the first recorded pair index per `X`).
- `gates/comparators/law_per_kernel.csv`: per kernel at the default `X`: `kernel, n_cells,
  dyn_mean_median, dyn_frac_mean_median, sta_frac_mean_median, dyn_ever_median`.
- `features/cmp_law/Wall_Hall_{raw,norm}.npz`; the split stage (1.6).

`comparators.py law --jobs N` runs the pass with `multiprocessing` one process per cell (as
`extract all --jobs`); results are identical at any job count.

### 1.5 Which gates apply to a comparator row, and which do not

Applied through `models.run_split_stage` exactly as for a rung (1.6), on the norm and the raw
feature file, for the three splits and the label spaces `models splits --all-splits` uses
(within-trace and LORO in kernel and archetype space, LOKO in archetype space):

- B1-G1 at the unit (`P2 Sec. V 5.1 Plan 08`; SPEC 3.7.1): the label-shuffle null of `--null-perm`
  permutations, `pass` / `near_unfalsifiable` / `not run: N permutations < 500`; a
  `near_unfalsifiable` row goes to `gates/excluded_rows.csv` as every rung's does.
- B1-G3 (SPEC 3.7.2): the one-feature quarantine runs as part of the split stage (`quarantine=True`);
  the rung's score is the re-run (`models.effective_scores`).
- B1-G6 (SPEC 3.7.3): `scores.json["majority"]`.
- G-N (SPEC 3.7.5): the macro recall over headline archetypes is what `run_split_stage` writes.
- G-K0's relabelling (SPEC 3.3.3): applied at read time by `prepare_split_data`.
- G-DIM (SPEC 3.7.7): `dim_status` from `scores.json` (`full vector` for d = 2, 3, 4 on this corpus;
  `declared reduction` when d exceeds a fold's training cell count); no matched row: the comparator
  is not the combined rung. Written to `gates/comparators/gdim.csv` with the columns of
  `gates/gdim.csv` (`rung, grid_id, d, d_matched, method, status, loko_score, matched_to`;
  `d_matched = d`, `method = ""`, `matched_to = ""`), one row per (comparator, variant), the raw
  variant's `rung` suffixed `__raw`.
- G-M against APF (SPEC 3.7.8): `gates_comparison.gm_compare(scores_cmp, scores_apf, spread)` on
  each split, archetype space for LOKO and kernel space for the others; the norm variant against
  APF's norm run and the raw variant against APF's raw run (`models.split_dir(out, "apf", gid_apf,
  split, ls, normalized=False)`, `gid_apf` from `series.selected_grid_id(out, "apf", None)`);
  `spread` read from `gates/gm.params.json` `params["spread"]` (the measured spread of move 12;
  reused, not re-measured; Part 4 item 12). Written to `gates/comparators/gm.csv` with
  `gates/gm.csv`'s columns (`split, rung_a, rung_b, score_a, score_b, diff, spread, improving,
  worsening, ties, verdict`), `rung_a` = `cmp_<name>` or `cmp_<name>__raw`, `rung_b` = `apf` or
  `apf__raw`. When `gates/gm.params.json` is missing the verdict is `not run:
  gates/gm.params.json missing (move 12)`; when APF's split is missing, `not run:
  gates/splits/apf/<gid>/<split>__<ls>/scores.json missing`; when APF has no selection, `not run: no
  selection for apf`.
- G-L (i) (SPEC 3.7.4), the norm variant's LOKO/archetype score against its B1-G1 null p95: `pass`
  or `level only`, with the same `not run:` forms `gates_comparison.gate_gl` uses. Written to
  `gates/comparators/gl.csv` with `gates_comparison.GL_COLUMNS`, one `part = "i"` row per
  comparator and no part (ii) row (the shot-noise route is APF's); `gl.params.json` records
  `part_ii = "not applicable: G-L (ii) is APF's shot-noise route"`, so the Table 7 cell of the norm
  row prints `(i) <verdict>` through the existing `_gl_text` format.
- G-X (SPEC 3.7.6): `gates_comparison.gate_gx(out, "cmp_<name>", n_perm=..., n_jobs=...,
  n_estimators=..., seed_offset=..., grid_id="Wall_Hall")`, which writes the comparator's row into
  `gates/gx.csv` (per-rung replacement, section 0.3 item 4) and its run under
  `gates/gx_runs/cmp_<name>/Wall_Hall/loko__campaign/`; norm features only, as for the rungs.

Not applied, with the string printed in the comparator's Table 7 row:

- G-C: `not applicable: comparator, not a lead of the ladder (G-C calibrates the ladder's leads
  against the gemm pulse)`. The comparator is a published reduction, not one of the paper's
  encodings; its connection to the pulse is not a claim the paper makes.
- G-F (i): `not applicable: one row per cell (G-F (i) is the within-trace window design)`; the
  idle-inseparability design of SPEC 3.3.4 needs windows of each idle cell and a comparator has one
  vector per cell.
- G-P: `not applicable: per-kernel pass-period reading, not a row property`. G-P reads the pass table
  per kernel and has no per-row form; it is printed in Table 5's companion and gp.csv, not here.
- G-DEC: `not applicable: a reading of the content rung on one kernel`. G-DEC is the decay exhibit of
  the content-change rung on floyd.
- The temporal gates G1 to G5, G-ORD and the grid: `not applicable: no (W, H) (one vector per cell by
  definition)`; the `resolution (W x H)` cell reads `whole cell (per cell, by definition)`.
- G-J, G-V, the clustering, G-F (ii): not computed for comparators in this epoch (Part 4 item 13
  names them for the author).

### 1.6 File layout and the split stage

Feature files, identical in keys to `series.build_features`'s output so that `series.load_features`,
`splits.make_labels`, `models.prepare_split_data` and `models.run_split_stage` read them unchanged:

`features/cmp_<name>/Wall_Hall_raw.npz` and `Wall_Hall_norm.npz` with arrays `X` (float64,
`[n_cells, d]`), `feature_names` (str), per row `cell_id, kernel, archetype, campaign, role` (str),
`rep, win_start (= 0), n_series_cell (= the cell's n_rows_used or n_pairs_used or n_windows at the
default X)` (int64), scalars `W = -1, H = -1, grid_id = "Wall_Hall", normalized, head_drop_json,
n_windows_dropped (= the count of cells with a refused status, whose rows are omitted), wapf_norm =
""`. Rows: cells in `cells.csv` order, every `status == ok` cell with an extract (and, for Law, a
completed pass); idle rows with `archetype = "IDLE"`, `kernel = "idle"`.

The split stage: for each comparator, for `normalized in (False, True)`, for `(split, labelspace)` in
`(("within_trace", "kernel"), ("within_trace", "archetype"), ("loro", "kernel"), ("loro",
"archetype"), ("loko", "archetype"))`:

```python
models.run_split_stage(out, "cmp_<name>", "Wall_Hall", split, labelspace, normalized=normalized,
                       n_perm=null_perm, n_jobs=n_jobs, n_estimators=n_estimators,
                       run_null=split in null_splits, seed_offset=seed_offset)
```

which writes `gates/splits/cmp_<name>/Wall_Hall/<split>__<labelspace>[__raw]/{predictions.csv,
scores.json, null.json, l1_quarantine.json}`. Within-trace reads `not applicable: one window per
cell` by the existing rule (one row per cell). Nothing else about the split stage is changed or
wrapped: `params` of each `scores.json` carries `grid_id = "Wall_Hall"`, `gc_verdict = null` (no G-C
row exists for a comparator), and the comparator's own `params.json` says why
(`"grid_id_reason": "one vector per cell: the whole-cell point by definition"`).

The gate files of 1.5 under `gates/comparators/`: `gl.csv`, `gdim.csv`, `gm.csv`, each with its
`.params.json`; G-X in `gates/gx.csv`. A `gates/comparators/verdicts.csv` summary, one row per
(comparator, variant, split, labelspace): `rung, variant, split, labelspace, feature_count, accuracy,
macro_recall, null_p95, rank, majority, b1_g1, gl, gdim, gm_vs_apf, gx, status`, read through
`models.effective_scores`, so a reader has the whole record in one file; `tables.py` reads the
source files, not this summary.

CLI (`comparators.py`; exit 0 on success, a written refusal included; 2 when `cells.csv` or a named
input is missing, its path on stderr; 1 on an internal error):

```
comparators.py savoldi   --out O [--rows all_after_head_drop|rung_series] [--ddof 1]
                         [--null-perm 500] [--null-splits loko,loro,within_trace] [--n-jobs 1]
                         [--n-estimators 300] [--seed-offset 0] [--no-splits]
comparators.py dhodapkar --out O [--grid 0.04,0.1,...,0.9] [--delta-th-default 0.04]
                         [--boundary-rule gt|ge] [--phase-length-rule n_over_b_plus_1|interior]
                         [the split flags above] [--no-splits]
comparators.py law       --out O [--x-grid 2,4,8,16] [--x-default 4] [--x-unit dumps|pairs]
                         [--head-drop-rule runs_from_seq_first|reset_at_head_drop]
                         [--feature-source window|ever] [--jobs 1] [--force] [--only REGEX]
                         [the split flags above] [--no-splits]
comparators.py gates     --out O [--null-perm 500] [--n-jobs 1] [--n-estimators 300] [--seed-offset 0]
                         (G-L (i), G-DIM, G-M vs APF, G-X and verdicts.csv for every comparator whose
                          split directories exist; a missing comparator gets not run: rows)
comparators.py all       --out O [every flag above]     (savoldi, dhodapkar, law, gates in that order)
```

`--no-splits` computes the statistics and the feature files only (the runbook's inspection path).
Every command writes its `params` block; `--seed-offset` shifts the four seeds as everywhere.
`cells.csv` is always `<out>/cells.csv` (the split stage reads that path and nothing else).

### 1.7 The driver hook: move 14

Builder A edits `run_moves.py` in exactly three places (builder B does not touch these three):

(a) A module constant and the bound in `parse_moves`:

```python
MAX_MOVE = 14      # epoch 2: move 14 = the comparators (SPEC_epoch2.md Part 1.7)
...
    bad = [m for m in moves if m < 0 or m > MAX_MOVE]
    if bad:
        raise ValueError(f"moves outside 0-{MAX_MOVE}: {bad}")
```

(b) `_add_run_args`: `--moves` default becomes `"0-14"`; new flags `--delta-th-default`
(float, default 0.04), `--law-x-default` (int, default 4), `--savoldi-rows` (default
`all_after_head_drop`), `--comparator-jobs` (int, default None: falls back to `--n-jobs`).

(c) In `build_plan`, immediately after the move-13 `_cmd(13, "gf check on Table 7", ...)` line and
before the comment `# seed offsets and n-jobs on the commands that take them`, this block and
nothing else:

```python
    # ---- move 14: the exact-input comparators (SPEC_epoch2.md Part 1; builder A owns this block)
    cmp_common = [*O, "--null-perm", o.null_perm, "--null-splits", o.null_splits, "--n-jobs", o.n_jobs,
                  "--n-estimators", getattr(o, "n_estimators", 300), "--seed-offset", getattr(o, "seed_offset", 0)]
    cmp_inputs = ["cells.csv", "inputs/head_drop.csv", ADMISSIBILITY, "gates/gk0.csv",
                  "json:gates/selection.json:apf", "gates/gm.params.json"]
    P.append(_cmd(14, "comparators savoldi", "comparators", "savoldi",
                  cmp_common + ["--rows", getattr(o, "savoldi_rows", "all_after_head_drop")],
                  outputs=["gates/comparators/savoldi.csv", "gates/splits/cmp_savoldi/Wall_Hall/loko__archetype/scores.json"],
                  inputs=cmp_inputs))
    P.append(_cmd(14, "comparators dhodapkar", "comparators", "dhodapkar",
                  cmp_common + ["--delta-th-default", getattr(o, "delta_th_default", 0.04)],
                  outputs=["gates/comparators/dhodapkar_sweep.csv", "gates/splits/cmp_dhodapkar/Wall_Hall/loko__archetype/scores.json"],
                  inputs=cmp_inputs))
    P.append(_cmd(14, "comparators law", "comparators", "law",
                  cmp_common + ["--x-default", getattr(o, "law_x_default", 4),
                                "--jobs", getattr(o, "comparator_jobs", None) or o.n_jobs],
                  outputs=["gates/comparators/law_sweep.csv", "gates/splits/cmp_law/Wall_Hall/loko__archetype/scores.json"],
                  inputs=cmp_inputs))
    P.append(_cmd(14, "comparators gates", "comparators", "gates", cmp_common,
                  outputs=["gates/comparators/gm.csv", "gates/comparators/verdicts.csv"],
                  inputs=cmp_inputs + ["gates/comparators/savoldi.csv", "gates/comparators/dhodapkar.csv", "gates/comparators/law.csv"]))
    P.append(_cmd(14, "tables table7_comparators,table_comparators", "tables", None,
                  [*O, "--only", "table7_comparators,table_comparators"],
                  outputs=["report/tables/table7_comparators.csv", "report/tables/table_comparators.csv"],
                  inputs=cmp_inputs + ["gates/comparators/verdicts.csv"]))
    P.append(_cmd(14, "figures dhodapkar_sweep", "figures", None, [*O, "--only", "dhodapkar_sweep"],
                  outputs=["report/figures/fig_dhodapkar_sweep.pdf|report/figures/SKIPPED.txt"],
                  inputs=["gates/comparators/dhodapkar_sweep.csv"]))
    P.append(_cmd(14, "tables manifest (after comparators)", "tables", None, [*O, "--only", "manifest"],
                  outputs=["report/manifest.json"], inputs=["gates/comparators/verdicts.csv"]))
```

The comparator commands carry `--seed-offset`, `--n-jobs` and `--n-estimators` explicitly, so
`RANDOM_COMMANDS` and `NJOBS_COMMANDS` are not edited (builder B's region). The `json:` input form
degrades to `absent` until builder B's item B2 lands and is then hashed per key; both states are
consistent across a resume.

`tests/test_driver.py`: builder A makes exactly two one-token amendments and nothing else there:
in `test_parse_moves`, `run_moves.parse_moves("14")` becomes `run_moves.parse_moves("15")`; in
`test_cli_guards`, `["run", "--out", "/x", "--moves", "14"]` becomes `[..., "--moves", "15"]`. Both
assert the bound, and the bound moves by one. This is the only amendment of an existing test in the
epoch and both builders' reports name it.

`RUNBOOK.md`: builder A appends a "Move 14: the comparators" section after move 13 (what runs, what
it writes, what the author looks at: the `U_text` per kernel, the sweep figure, the Table 7
comparator rows; the cost: the Law pass re-streams every trajectory, one to three minutes per cell
per process, `--comparator-jobs 4` on the 96-cell corpus, resumable per cell) and, in the driver
section, one sentence that `--moves` defaults to `0-14`. `SPEC.md` section 7: builder A appends one
row for move 14 to the move table (after builder B's rewrite of rows 3 and 12 or before it; the rows
are disjoint).

### 1.8 Tables (builder A owns `tables.table7`, the two new tables, `TABLE_NAMES` and `CITATIONS`)

`table7()` itself is unchanged in its rows (30 rows; `tests/test_report.py` asserts it) except for
CHECK_3 M5's table half (the only line builder A changes inside it): the `feature count` cell prints
`scores.json["feature_count_used"]` when that key is present and non-null, else `feature_count`
(builder B's item B5 writes the key; before it lands the fallback prints as today).

New table `table7_comparators` (`report/tables/table7_comparators.{csv,md,tex}`), the same 16
`TABLE7_COLUMNS`, `label = "tab:table7_comparators"`, `note_comment = "comparator rows of Table 7;
append below tables/table7.tex; one row per (comparator, variant, split) in the split's primary label
space, then the archetype-space rows of LORO and within-trace"`. Rows, in order: for each comparator
in `schema.COMPARATORS` order, for each variant selected by `TABLE7_VARIANT` (`"both"`: the raw row
first, then the norm row; `"raw"`; `"norm"`), the three primary rows (LOKO archetype, LORO kernel,
within-trace kernel), then the appended archetype-space rows (LORO, within-trace). With `"both"`:
3 x 2 x 5 = 30 rows.

Cell rules for a comparator row (`rung` = display name plus the variant and the declared default,
for example `Dhodapkar-Smith 2003 (as published; delta_th = 0.04, declared default)` and
`Dhodapkar-Smith 2003 (level-normalized; delta_th = 0.04, declared default)`; Law prints `X = 4,
declared default`; Savoldi prints `U = mean +/- SD of K`):

- `resolution (W x H)`: `whole cell (per cell, by definition)`.
- `feature count`, `accuracy`, `macro recall (headline rows)`, `null p95`, `rank`, `majority`: from
  `models.split_dir(out, "cmp_<name>", "Wall_Hall", split, ls, normalized)` (import through
  `_report_common.split_dir(..., raw=not normalized)`), read through `effective_scores`; the
  `b1g1_block` and `rank_text` / `null_text` rules of `_report_common` apply unchanged; a missing
  file prints `not run: gates/splits/cmp_<name>/Wall_Hall/<split>__<ls>[__raw]/scores.json missing
  (move 14)` in every score cell; a `not applicable:` accuracy prints that string in every score cell
  as `table7` does.
- `G-C`, `G-F (i)`: the fixed strings of 1.5, always non-empty (the move-13 `gf_check` reads
  `table7.csv` only, but the comparator table follows the same rule).
- `G-L`: from `gates/comparators/gl.csv` for the norm row (`(i) pass` / `(i) level only` / the
  `not run:` form); the raw row prints `level-inclusive (as published)`.
- `G-DIM`: from `gates/comparators/gdim.csv` by `rung` (`cmp_<name>` or `cmp_<name>__raw`), the
  `_gdim_text` format; `G-M vs APF`: from `gates/comparators/gm.csv`, the `_gm_text` format with
  `rung_b` `apf` (norm row) or `apf__raw` (raw row); `G-X`: `_gx_text(out, "cmp_<name>")` from
  `gates/gx.csv` for both rows (G-X runs on the norm features).
- The total-confound refusal of SPEC 3.7.6 applies to the LOKO row as in `table7`.

Implementation: a helper `_gate_csv(out, name, rung)` returning `gates/comparators/<name>` when
`rung` starts with `cmp_` and `gates/<name>` otherwise, used by `_gl_text`, `_gdim_text`, `_gm_text`
(the three helpers gain the indirection and behave exactly as before for the rungs); `_gm_text`
also gains a keyword `rung_b: str = "apf"` (the row is matched on `rung_a in aliases and rung_b ==
rung_b`), which the comparator row builder sets to `"apf__raw"` for the raw row; `_gm_text`'s early
return for `rung == "apf"` is unchanged.

New table `table_comparators` (`report/tables/table_comparators.{csv,md,tex}`), the methods' own
per-kernel numbers (the "per-run form" of C14): columns `kernel, n cells, Savoldi U (median cell),
Savoldi mean K (median), Savoldi SD K (median), D-S boundaries (median, delta_th = <default>), D-S
stability (median), D-S mean phase length (median), Law dynamic-for-X (median, X = <default>), Law
static-for-X fraction (median), Law dynamic ever (median)`; one row per kernel in `schema.KERNELS`
order then `idle`; from the three `*_per_kernel.csv`; a missing file prints `not run:
gates/comparators/<file> missing (move 14)` in its cells. `note_comment` names the two declared
defaults and that every grid point is on disk in the sweep files.

`TABLE_NAMES += ("table7_comparators", "table_comparators")`; `CITATIONS` gains both with
`CIT_SAVOLDI`/`CIT_DHODAPKAR`/`CIT_LAW` abbreviated to their first clause plus `P2 Sec. 4 Table 7;
P2E Sec. IV Table 2 (the external comparator row)`. `tables.run` dispatches both; `tables (all)` at
move 12 therefore writes both with `not run: ... (move 14)` cells, and move 14 rewrites them.

`latex_skeleton.py`: after the `table7` shell, `\input{tables/table7_comparators.tex}` (the same
generated-table form) and `\input{tables/table_comparators.tex}`; one figure placeholder for
`fig_dhodapkar_sweep` in section VII. Comment lines only; no sentence.

### 1.9 The figure (builder A owns `figures.py` in this epoch)

`FIGURE_NAMES += ("dhodapkar_sweep",)`. `fig_dhodapkar_sweep(out, cells, ex, *, default=None)`:
reads `gates/comparators/dhodapkar_sweep.csv` (and `dhodapkar.params.json` for the default and the
grid); when the CSV is absent, writes the placeholder `not run: gates/comparators/dhodapkar_sweep.csv
missing (move 14)` through `_placeholder` on the normal path (not the exception path, so
`figures.json` `status` stays `"ok"` and `tests/test_report.py::test_all_figures_written` holds).
When present: one panel per kernel in `schema.KERNELS` order (a 13th panel `idle` when idle rows
exist), x = `delta_th` (the grid points, linear axis), left y = `stability` (solid lines, 0 to 1),
right y = `mean_phase_length_pairs` (dashed lines, log scale), eight reps overlaid in the kernel's
one colour (the convention of `fig_apf_per_kernel`), a vertical dotted line at the declared default
labelled `declared default delta_th = <value>` in the legend, admissible cells only, the panel
titles being the kernel names. Files `report/figures/fig_dhodapkar_sweep.png/.pdf`; returns
`{"png", "pdf", "n_panels", "n_cells_drawn", "default"}`. Axis labels and the legend are the only
text.

### 1.10 Builder A's tests (`tests/test_comparators.py`, new; pytest style; uses `tests/_b2_common.py`
and `synth.py`)

Every test uses synthetic data; `n_estimators = 10` and `n_perm <= 10` where a forest runs.

1. `test_savoldi_matches_truth`: `synth.write_cell(SynthSpec(name="gemm", seed=42, n_pairs=40))`,
   `extract.extract_cell` into `<out>`, then `savoldi_cell(ex, head_drop=3)`: `K_mean` equals
   `numpy.mean(truth["K"][3:])` and `K_sd` equals `numpy.std(truth["K"][3:], ddof=1)` to `1e-9`;
   `rows="rung_series"` drops the last row; `U_text` matches `^\d+(\.\d+)?% \+/- \d+(\.\d+)?%$`;
   `params["ddof"] == 1`.
2. `test_dhodapkar_delta_is_one_minus_J_and_the_sweep_is_whole`: a cell with `pulse_period=10,
   pulse_extra=4096, K0=2048, churn=0.02, n_pairs=60`: `delta` equals `1 - truth["J"]` on the kept
   rows to `1e-12`; at `delta_th = 0.3` `n_boundaries` equals the number of `truth["boundaries"]`
   inside the kept rows (the J dip at a boundary is about 1/3, SPEC 5.1), `stability == 1 - B / n`,
   `mean_phase_length_pairs == n / (B + 1)`; the sweep CSV has exactly `len(DHODAPKAR_GRID)` rows per
   cell, every grid value present, exactly one `is_default = true` per cell and it equals
   `params["default"]`; `phase_length_rule="interior"` gives the interior mean.
3. `test_law_streaming_pass_matches_brute_force`: `SynthSpec(name="gibbs", seed=5, n_pairs=30,
   K0=64, churn=0.3, floor_F=0, gap_seqs=(7,))` written with `keep_sets=True`; brute force from
   `truth["sets"]`: for each `X` in `(2, 4, 8)` and each `t`, `dyn = |intersection of the L(X) sets
   ending at t|` and `sta = N - |union|`; the pass's series equal these exactly under both
   `x_unit` values; at the gap pair `dyn = 0` and `sta = N` for `X = 2`; `check_x2_equals_K` is
   `true`; `dyn_ever` equals the brute-force count of pages with a membership run `>= L`;
   `head_drop_rule="reset_at_head_drop"` with `head_drop=5` restarts the runs (the first recorded
   `dyn` at `X = 4` equals the brute force over sets `5..7` only); a `--corrupt seq_reverse` cell
   yields `status == "refused: seq not monotone at row <n>"` and no series file.
4. `test_law_memory_model_and_resume`: `law_stream` exposes a module-level hook `LAW_STEP_HOOK`
   (called once per finished snapshot with the snapshot's page array, as `extract.EMIT_HOOK` is)
   and tracks every live page array in a `weakref.WeakSet` `_LIVE_PAGE_ARRAYS` (as
   `extract._LIVE_SNAPSHOTS` does); the test installs a hook that asserts `len(_LIVE_PAGE_ARRAYS)
   <= 1` at every step on a 30-pair cell and that the four run-length arrays are the only
   `N`-length state (`law_stream` returns their `nbytes` in `params["memory_model"]`); a second
   `comparators law` run on the same `<out>` skips every cell with `status ok` (`law_cells.json`
   unchanged, `elapsed_s` identical) and `--force` re-runs.
5. `test_feature_files_and_split_layout`: `_b2_common.corpus(reps=2, idle=2, kernels=[...six...],
   n_pairs=60)`; `comparators.py savoldi --out ... --null-perm 5 --n-estimators 10 --null-splits loko`:
   `features/cmp_savoldi/Wall_Hall_norm.npz` loads through `series.load_features` with every key of
   1.6, one row per ok cell, idle rows `archetype == "IDLE"` and `kernel == "idle"`;
   `gates/splits/cmp_savoldi/Wall_Hall/loko__archetype/scores.json` has `params.grid_id == "Wall_Hall"`
   and `feature_count == 2`; `within_trace__kernel/scores.json` `status` starts with `not applicable:
   one window per cell`; the raw run sits in `loko__archetype__raw/`; with a cell's `failed_verdict`
   set to a refusal in `gates/preconditions.csv` and its id in `preconditions.json`
   `excluded_cells_pair_rungs`, the cell is absent from `cmp_dhodapkar`'s and `cmp_law`'s
   `predictions.csv` and present in `cmp_savoldi`'s (the `PAIR_RUNGS` line).
6. `test_gates_for_comparators`: on the corpus of test 5, after `series.build_features` for `apf`
   at `(8, 4)` in both variants, `models.run_split_stage` for `apf` at `W8_H4` (LOKO/archetype and
   LORO/kernel, norm and raw, `n_perm=5`, `n_estimators=10`), a hand-written
   `gates/selection.json` naming `W8_H4` for `apf`, and a hand-written `gates/gm.params.json` with
   `params.spread = 0.04`: `comparators.py gates` writes `gates/comparators/gl.csv` with a `part = i`
   row per comparator whose verdict is `pass` or `level only`; `gdim.csv` rows `cmp_savoldi` and
   `cmp_savoldi__raw` with `status == "full vector"` and `d == 2`; `gm.csv` rows with `rung_b` in
   (`apf`, `apf__raw`) and a verdict in (`beats`, `difference with margin`); with `gm.params.json`
   removed the G-M verdict reads `not run: gates/gm.params.json missing (move 12)`; `gates/gx.csv`
   has a row `rung == "cmp_savoldi"`; `verdicts.csv` has 30 rows.
7. `test_table7_comparators_and_summary_table`: on `report_fixtures.make_out` (no comparator files):
   `tables.run(out, ["table7_comparators", "table_comparators"])` writes 30 rows whose every score
   cell reads `not run: gates/splits/cmp_<name>/Wall_Hall/<split>__<ls>[__raw]/scores.json missing
   (move 14)`, `G-C` and `G-F (i)` the fixed strings on every row, `resolution` the fixed string;
   `table7.csv` still has 30 rows; then with a hand-written
   `gates/splits/cmp_savoldi/Wall_Hall/loko__archetype/scores.json` (`accuracy 0.75`, `null_p95
   0.5`, `b1_g1 pass`, `feature_count 2`, `feature_count_used 1`, `with_quarantine` with one
   feature) the LOKO norm row prints the re-run's numbers through `effective_scores` and `feature
   count` prints `1`; `table_comparators.csv` prints `not run: ... (move 14)` without the files and
   the medians with hand-written `*_per_kernel.csv`.
8. `test_figure_dhodapkar_sweep`: without the sweep, `figures.run(out, ["dhodapkar_sweep"])`
   writes the placeholder PNG and PDF and `figures.json` `status == "ok"`; with a hand-written sweep
   (two kernels, two reps, the ten grid points) the return carries `n_panels == 2` and `default ==
   0.04`.
9. `test_driver_move14_plan_and_run`: `run_moves.parse_moves("0-14") == list(range(15))`;
   `parse_moves("15")` raises; `build_plan` on a Namespace built like `tests/test_driver.py::_ns`
   (without the epoch-2 keys) lists the seven move-14 commands in the order of 1.7 with
   `--delta-th-default 0.04` and `--x-default 4` present; `run_moves.main(["plan", "--out", "/x",
   "--moves", "14"])` exits 0; on the corpus of test 5 (extracted by `extract.py`, preconditions and
   G-K0 run by function call, `--null-perm 5 --n-estimators 10 --null-splits loko`) `run_moves run
   --moves 14` exits 0, every move-14 command is `done`, the G-M cells read `not run:
   gates/gm.params.json missing (move 12)`, and `report/manifest.json` lists
   `gates/comparators/verdicts.csv`.
10. `test_cli_exit_codes`: `comparators.py savoldi --out <empty>` exits 2 naming `cells.csv`;
    `comparators.py all --no-splits` on the corpus exits 0 and writes the six per-cell and per-kernel
    CSVs and the six feature files.
11. `test_chain_with_law_on_builder1_pipeline` (marked like `tests/test_gates_chain.py`, skipped
    without `synth.py`): `synth corpus --reps 2 --idle 2 --n-pairs 60` -> `extract index` ->
    `extract all` -> preconditions (`c1_activity_min=0.0`) -> `comparators all --null-perm 5
    --n-estimators 10 --null-splits loko --jobs 2`: every file of 1.2 to 1.6 exists, `law.csv`
    `check_x2_equals_K` is `true` on every cell, `verdicts.csv` has 30 rows, and no file under
    `gates/comparators/` names a sandbox workload (the corpus has none).

---

## Part 2. The fix pass (builder B)

Each item: source, file(s), what changes, the test that proves it. Items marked "(choice)" expose a
parameter listed in Part 4. Function-level defaults are never changed where an existing test calls
the function directly; the CLI carries the new default where the two differ, and the difference is
stated in the docstring.

**B1. C1 re-declared (AA T1; CHECK_3 M1; CERT 6.5; E1 6.7, 6.61).** `gates_precondition.py`,
`run_moves.py` (builder B's regions), `tables.py::table4_status`, `RUNBOOK.md` move 2.
Rule, citing AA T1 in the docstring of `gate_preconditions`: a kernel cell passes C1 when its peak
changed-page count over the cell (`sidecar["K_max"]`, the undropped maximum the extractor records,
SPEC 2.3) exceeds the idle cells' 95th percentile of K, the same floor G-K0 uses, when admissible
idle cells exist; until they exist, the interim `K_max >= 200` pages. C1 stays a hard gate
(`all_hard_pass` unchanged in form). Precisely:

- New module constants `C1_ACTIVITY_MIN_PAGES = 200` (AA T1), `C1_FLOOR = True`, and the floor
  machinery extracted from `gate_gk0` into one function used by both:
  `idle_k_edge(out, idle_cells, *, idle_pool=GK0_IDLE_POOL, idle_percentile=GK0_IDLE_PERCENTILE)
  -> float | None` (the pooled-snapshot or cell-median percentile of `K` over the given idle cells'
  `extract.csv`, exactly the code now inline in `gate_gk0`; `gate_gk0` calls it, so C1 and G-K0
  read one number from one function and one file, `extract/<cell_id>/extract.csv`).
- `gate_preconditions(out, *, ..., c1_activity_min: float | None = None,
  c1_activity_min_pages: int = C1_ACTIVITY_MIN_PAGES, c1_floor: bool = C1_FLOOR,
  c1_idle_pool: str = GK0_IDLE_POOL, c1_idle_percentile: float = GK0_IDLE_PERCENTILE)`.
  Order of rules: (i) if `c1_activity_min` is given (not None): the inherited fraction rule
  `apf_max >= c1_activity_min`, `C1_rule = "inherited: apf_max >= <v> (validate_campaign.py line
  39; superseded by AA T1)"`, kept so that the existing tests that pass `c1_activity_min=0.0` and
  `0.0005` run unchanged; (ii) else if `c1_floor` and at least one idle cell has `cells.csv` status
  `ok`, a sidecar, `C2 == pass` and `C6 == pass` (their C1 is the not-applicable string, so their
  admissibility is C2 and C6; computed in a first loop over the idle cells before any kernel row):
  `K_max > edge` with `edge = idle_k_edge(...)` over those idle cells, `C1_rule = "floor: K_max >
  <edge> (idle p95 of K over <n> idle cells; AA T1; the G-K0 edge)"`; (iii) else `K_max >=
  c1_activity_min_pages`, `C1_rule = "interim: K_max >= <n> pages (AA T1; no admissible idle cell)"`
  or `"... (AA T1; --c1-floor off)"`. Idle cells keep `C1_IDLE_STRING`.
- `preconditions.csv` gains `C1_K_max`, `C1_rule`, `C1_threshold_pages` (the edge, the interim
  pages, or `ceil(frac * N)` for the inherited rule) after `C1_apf_max`; `PRE_COLUMNS` updated.
  `preconditions.json` `params` gains `C1_rule_in_force` (the rule string of the kernel rows),
  `C1_floor` (bool), `C1_floor_edge`, `C1_floor_idle_cells` (the ids pooled), `C1_activity_min_pages`,
  `C1_inherited_fraction` (None or the value), keeping `C1_ACTIVITY_MIN_spec_default`.
- CLI `gates_precondition preconditions`: `--c1-activity-min FLOAT` (default None: the inherited rule
  only when given), `--c1-activity-min-pages INT` (default 200), `--c1-floor on|off` (default on),
  `--c1-idle-pool`, `--c1-idle-percentile` (the G-K0 names and defaults).
- Driver (`_add_run_args`, builder B's region): `--c1-activity-min`, `--c1-activity-min-pages`,
  `--c1-floor`, appended to the move-2 `preconditions` args (read with `getattr`).
- `tables.table4_status`: the Plan 02 row's `APF at 500 ms` text (`p02`, today `"<n> of <m> cells
  all_hard_pass; C7: ..."`) gains the suffix `; C1 rule: <C1_rule_in_force>` from
  `gates/preconditions.json` `params`, or `; C1 rule: not run: gates/preconditions.json missing`
  (or `... has no C1_rule_in_force`, for a file written before this epoch) when it cannot be read.
- `RUNBOOK.md` move 2: the rule, the two flags, and that `preconditions.csv` says which rule was in
  force for each run (AA T1: "the runbook records which rule was in force for each run").
- Tests (`tests/test_gates_precondition.py`, appended; `tests/test_epoch2_fixes.py`): (a) a corpus
  with idle cells: a kernel cell with `K_max` above the idle p95 passes, one below fails, `C1_rule`
  starts with `floor:`, and `C1_threshold_pages` equals `gk0.csv`'s `idle_band_edge` after
  `gate_gk0` runs on the same out (one number, two files); (b) no idle cell: `C1_rule` starts with
  `interim:`, `K_max = 199` fails and `200` passes; (c) idle cells present and `c1_floor=False`:
  interim; (d) `c1_activity_min=0.0`: inherited, every kernel cell passes (the existing tests are
  this case); (e) `build_plan` carries the three flags to the `preconditions` command when set and
  omits them when unset; (f) `table4_status.csv`'s Plan 02 row contains `C1 rule: floor:` on a
  fixture whose `preconditions.json` says so.

**B2. Per-key staleness (CHECK_3 M2; CERT 1(c), 7.3; E1 6.29).** `run_moves.py` (`_stale_reason`,
`inputs_sha256` use, the `inputs=` lists in `temporal()`/`splits()`/moves 6 and 7): declare
`json:gates/selection.json:<rung>` for `splits <rung>`, `gx <rung>` and `tables table6` (apf), and
`csv:gates/g3_flags.csv:rung=apf` for the two `alias` steps; keep the whole file for `gl` (both
runs), `gdim`, `gm`, `variance`, `cluster`, `tables (all)`, `gf all rungs at the selected points`;
a new `_input_digest(out, spec)` hashes `json.dumps(j[key], sort_keys=True)` for the `json:` form
(looking under the top level, then `selection` / `rungs`, as `_output_exists` does) and
`json.dumps(matching rows, sort_keys=True)` for the `csv:` form, `"absent"` when the file or key is
missing, and the file bytes otherwise; ledger records keep the spec string as the key. Test
(`tests/test_epoch2_fixes.py`): `_input_digest` on a two-rung `selection.json` changes when the
named rung's entry changes and not when the other rung's entry is added; and
`tests/test_driver_end_to_end.py` (B28) asserts that a `--dry-run` after the full run marks zero
commands `stale`.

**B3. The permutation floor on G-F (i), G-X and the clustering (CHECK_3 M3; CERT 3, 6.7; E1 6.36)
(choice).** `gates_precondition.py::_gf_part1` and `gate_gf`, `gates_comparison.py::gate_gx`,
`models.py::run_clustering`: a keyword `perm_floor: int = 0` on the three functions (0 = judge on any
count, the present behaviour, which the existing direct-call tests depend on) and a CLI flag
`--perm-floor` on `gates_precondition gf`, `gates_comparison gx`, `models cluster` whose default is
`models.B1G1_MIN_PERM` (500): when `0 < n_perm < perm_floor` the G-F part (i) verdict, the G-X
`leak_verdict` and the clustering `exceeds_ari` / `exceeds_nmi` read `not run: <n> permutations <
<floor>` exactly as `models.b1_g1_verdict` writes it, the numbers stay in their columns, and
`perm_floor` is recorded in `params`. `_report_common.rung_override` already treats only `GF_VOID`
as an override. Docstrings state the two defaults and why. Test: CLI runs with `--n-perm 5` write
the string in the three files; `gate_gf(out, rung="apf", n_perm=30)` (the existing call form)
still writes `GF_VOID` on the separable fixture; `perm_floor=500` in a direct call writes the string.

**B4. The matched comparison's splits (CHECK_3 M4; E1 6.48) (choice).** `gates_comparison.py::
gate_gdim`: a parameter `matched_splits: str = "all"` (`"all"`: the five (split, label space) pairs
Table 7 plans, `("loko","archetype"), ("loro","kernel"), ("within_trace","kernel"),
("loro","archetype"), ("within_trace","archetype")`; `"loko"`: the present single run) and
`null_splits: str = "loko,loro,within_trace"` passed as `run_null=split in null_splits` to each
matched `run_split_stage` (so LORO's null costs what the author allows); CLI `--matched-splits`,
`--null-splits`; the driver passes `--null-splits o.null_splits` to `gdim` (builder B's move-12
line). `gdim.csv` gains one `combined (matched)` row per (split, label space) with a `split` and
`labelspace` column (the existing columns keep their names; `tables._gdim_text` matches the first
row of the rung as today, which is the LOKO row when written first). Test: after `gate_gdim` on the
fixture corpus every one of the five `gates/splits_matched/combined/<gid>/<split>__<ls>/scores.json`
exists and `table7.csv`'s `combined (matched)` LORO row prints a number, not a `not run:`.

**B5. `feature_count_used` (CHECK_3 M5; E1 6.49), the models half.** `models.py::run_split_stage`
writes `feature_count_used = min(v["d_used"] for v in preds.values())` into `scores.json` and into
`with_quarantine` (the re-run's own minimum); `effective_scores` carries the key through. The table
half is builder A's (Part 1.8). Test (`tests/test_gates_models.py`, appended): a run with
`reduce_to=3` on a 60-feature file writes `feature_count == 60` and `feature_count_used == 3`.

**B6. `--n-jobs` on `gord` and `grid` (CHECK_3 M6; E1 6.28).** `gates_temporal.py::gate_gord`: the
`n_order_perm` order shuffles and the `null_perm` label shuffles are pre-drawn in the main process
(one permutation per cell per repetition from `rng_o`, one label vector per repetition from `rng_l`,
drawn in the present order so the streams are identical at any job count) and scored through
`joblib.Parallel(n_jobs=n_jobs)(delayed(...))` as `_gf_part1` does; `gate_grid`: the per-cell G1
surrogate statistics through the same. `RUNBOOK.md` 0b heading becomes "The smoke run (synthetic
corpus; about an hour at the full corpus)" and the "accepted and not used" sentence is removed.
Test: `gate_gord(..., n_jobs=2)` writes `gord.json` numerically identical to `n_jobs=1` on the same
corpus (every value of `score_shuffled`, `null_summary`), and `gate_grid` likewise for
`temporal_per_kernel.csv`.

**B7. The roll-up with no applicable kernel for G1 (CHECK_3 M7; CERT 3, 6.6, 7.4; E1 6.22) (choice).**
`gates_temporal.py::select`: a parameter `g1_none_applicable: str = "drop"` (`"drop"`: when G1's
roll-up starts with `not applicable`, G1 leaves `applicable` as G2 does; `"refuse"`: the selection
entry gets `refusal = "not run: no applicable kernel for G1"`, `passes_acceptance = false`,
`selected_by = "no applicable kernel"`, and the grid rows keep their columns); CLI
`--g1-none-applicable`; recorded in `selection.json` `params`. Test: thirteen hand-written grid CSVs
with `G1 = TREND_PRESENT` on every kernel and G2 not applicable: under `drop` every integer-W point
with `G4 = pass` reads `gates_passed = "1 of 1"` and the selection passes acceptance at `W8_H4`
(the smallest W, hop ratio nearest 0.5); under `refuse` the entry carries the three strings above
and `passes_acceptance` is false.

**B8. Table 6 and the wAPF table over admissible cells (CHECK_3 M8; E1 6.42).** `tables.py::table6`
and `wapf_over_apf` (builder B's regions): kernels whose every cell is in `gates/preconditions.json`
`excluded_cells` print `not run: excluded by C1-C8 (all_hard_pass false)` in every score cell and are
not counted in `n`; a kernel with some cells excluded counts the admissible ones; the wAPF table
averages admissible cells only and prints the same string for a fully excluded kernel; a
`preconditions.json` without the `excluded_cells` key, or no file at all, excludes nothing (the
fixture of `tests/test_report.py` is that case and its assertions stay as they are). Test
(`tests/test_epoch2_fixes.py`): on `report_fixtures.make_out` with `preconditions.json`
`excluded_cells` set to every `floyd` cell, the floyd row of `table6.csv` reads the string in its
score cells and the `all` row's `n` drops by one kernel; `table_wapf_over_apf.csv` likewise.

**B9. The idle head-drop key (CHECK_3 M9; E1 6.10).** `series.head_drop_for(head_drop, kernel,
role: str | None = None)`: when `role == "idle"` the key is `"idle"`; the seven call sites
(`gates_precondition.py` G-F (ii) kernel loop, `gates_temporal.py` three sites, `gates_readings.py`,
`gates_comparison.py`, `series.build_features`) pass `role=c["role"]`. Test: `inputs/head_drop.csv`
with `idle,5`: `build_features` gives idle rows `n_series_cell == n_pairs - 1 - 5` and G-F (ii)'s
idle medians use the dropped series (the params record `head_drop_idle = 5`).

**B10. The G-L (ii) re-run (CHECK_3 M10; CERT 6.9; E1 6.38) (choice).** `run_moves.py`: a driver
flag `--gl2-rerun auto|manual` (default `auto`, SPEC 3.7.4's sentence as written) and an internal
step `gl2-rerun` at move 7 right after `gl`: it reads `gates/gl.csv` part (ii); when the verdict is
`GL_SHOT_NOISE` and the mode is `auto`, it runs (as a nested subprocess recorded in the ledger under
`argv_nested`) `models splits --out O --rung apf --all-splits --raw-and-norm --null-perm <n>
--null-splits <s> --feature-drop cov,std,peak2med --base-dir splits_gl2drop` and writes
`gates/gl2_rerun.json` (`status = "done"`, the base dir, the drop set); when the verdict is `pass`
the step records `not run: G-L (ii) passed`; under `manual` it records `not run: manual
(--gl2-rerun manual)`. `models.py splits` CLI gains `--base-dir` (default `splits`) passed as
`base_dir`. The tables are not changed: the refused run's numbers stay printed with the G-L refusal
beside them (the present rule) and the re-run lives under `gates/splits_gl2drop/apf/` for the author.
`RUNBOOK.md` move 7's sentence is replaced accordingly. Test: with a fixture `gl.csv` whose part (ii)
row reads `GL_SHOT_NOISE` and `subprocess.run` mocked, the nested argv contains `--feature-drop
cov,std,peak2med` and `--base-dir splits_gl2drop`; with `--gl2-rerun manual` the status string is as
above; with part (ii) `pass` nothing is run.

**B11. `--pass-frac` on `gates_readings gdec` (CHECK_3 M11; E1 6.46).** `gates_readings.py::main`:
`--pass-frac FLOAT` (default `GDEC_PASS_FRAC`) passed to `gate_gdec`. Test: the CLI accepts it and
`gdec.params.json` records the given value.

**B12. The synthetic corpus's duration (CHECK_3 M12; E1 6.62) (choice).** As built, G2's seconds
coverage and G-P's `T_seconds` use the constant `schema.DURATION_S = 600` (`gates_temporal.py`
`T = DURATION_S / entry.passes`; `gates_calibration.py` line 126), so on a 120-pair synthetic cell a
declared pass count is read against 600 s whatever the cell's duration. The fix has three parts.
(i) `synth.py`: `SynthSpec.duration_s: float | None = None`; when None the generator records
`n_pairs * 0.644` (the paper's median guest spacing, SPEC 2.3) in `truth.json` `spec.duration_s`
and `truth.duration_s`. (ii) `extract.py cell` and `all` gain `--duration-s FLOAT` (default
`schema.DURATION_S`) passed to `extract_cell(duration_s=...)` (the parameter exists; the sidecar
already records it as `duration_s_declared`); `extract_all(..., duration_s=600)`; `run_moves.py`
`--duration-s` (default 600) appended to the move-1 `extract all` args (builder B's region);
`RUNBOOK.md` 0b's smoke command adds `--duration-s 77.28` (120 x 0.644) and one sentence saying
why. (iii) `gates_calibration.gate_gp` derives each cell's `T_seconds = duration_s_declared /
passes` from that cell's sidecar (600 on the real corpus, the same number as today), and
`gates_temporal.g2_kernel` uses the median over the kernel's cells of `duration_s_declared`
(recording `duration_s_min`, `duration_s_max` in the per-kernel row's params; a sidecar without the
key falls back to `schema.DURATION_S`, recorded); the pass table's column keeps its name
`passes_per_600s` and its docstring says the count is per cell duration, which is 600 s on the real
corpus and the declared `--duration-s` on a synthetic one; `params` record `duration_source =
"sidecar duration_s_declared"`. Tests: (`tests/test_synth.py` / `tests/test_extract.py`, appended)
`truth.duration_s` is present and extracting with `duration_s=truth["duration_s"]` gives
`dt_est_s = 0.644` inside `DT_BRACKET_S`; (`tests/test_gates_temporal.py`, appended) a `synth.py`
cell set with `duration_s = 77.28`, extracted at `--duration-s 77.28`, `gemm` declared at 5 passes:
`G2_0500 = pass` and `G2_0644 = pass` at `W64` (coverage 2.07 and 2.67) and `fail` at `W32`; the
existing selection tests (sidecars at 600 through `tests/_synth_b2.py`) are unchanged.

**B13. The smoke corpus's minima (CHECK_3 M13).** `RUNBOOK.md` 0b: one sentence that a corpus with
fewer than 8 reps or 12 kernels reads G3's kernel flag, G-DEC's roll-up and G-M's sign test at their
fixed minima by construction, and that `--min-cells`, `--min-reps` are passed by hand for a smaller
corpus. No code; the test is the runbook check of B30.

**B14. The staleness trigger and hand edits (CERT 1(a), 6.1, 7.1).** `RUNBOOK.md` driver section:
gate result files under `gates/` are never edited by hand; a re-run of the preconditions with
another flag is the sanctioned path. No code (CERT 7.1's `csv-drop:` form is not required and not
built; Part 4 item 20 names it).

**B15. `gf_part1` in `table5_grid.csv` (CERT 1(b), 6.2, 7.2 first form).** `gates_temporal.py`:
remove `gf_part1` from `GRID_COLUMNS`, the `gf_rows` read of `gates/gf.csv` and the column fill in
`select`; G-F's verdict lives in `gates/gf.csv` only, where the tables read it. Test: after
`select`, `table5_grid.csv`'s header has no `gf_part1`; `tests/test_report.py` (which reads
`gf.csv` directly) is unchanged and passes.

**B16. `select` over a missing grid CSV (CERT 1(d), 7.4 second half).** `gates_temporal.py::select`:
when any of the thirteen CSVs is missing, the selection entry gets `refusal = "not run: grid
incomplete (<n> points missing)"`, `passes_acceptance = false`, and `selected_by` keeps its text
with the suffix ` (grid incomplete)`; the choice among the existing points is still recorded so the
author sees it. Test: delete one grid CSV, run `select`, assert the three fields.

**B17. The empty `refusal` on a best-feasible selection (CERT 3 first marker).** `select`: when
`passes_acceptance` is false and `refusal` would be the empty string, write `refusal =
"acceptance failed: <gate>: <verdict>, ..."` listing every gate in `applicable` whose verdict at the
chosen point is not `pass`; `GC_DISCONNECTED` keeps precedence as today; `selected_by` stays
`"best-feasible"` (`tests/test_gates_temporal.py` asserts it). Test: on the `blocks` case of the
existing selection test the entry's `refusal` starts with `acceptance failed: G2:`.

**B18. The `splits` CLI fallback and `grid_source` (CERT 3 third marker, 6.13, 7.5; E1 6.56).**
`models.py`: `run_split_stage(..., grid_source: str = "argument")` recorded in `params`; the
`splits` CLI uses `S.selected_grid_id(out, rung, None)` when `--grid-id` is absent and exits 2 with
`missing input: gates/selection.json has no entry for <rung> (run gates_temporal select first)`
when there is none, and passes `grid_source = "selection.json"` or `"argument"`. Test: the CLI
without a selection exits 2 with that message; a run's `scores.json` `params.grid_source` is
`"selection.json"`.

**B19. `wapf_norm` half exposed (CERT 5, 6.13, 7.6 first form; E1 6.31).** `gates_temporal.py::
gate_grid(..., wapf_norm: str = series.WAPF_NORM_DEFAULT)` passed to `series.build_all_grid`;
CLI `--wapf-norm median_K|median_self` on `grid`; recorded in `temporal.params.json`; the driver
gains `--wapf-norm` (default `median_K`) passed to both `series features` and `gates_temporal grid`
in `temporal()` (builder B's region). Test: `gate_grid(out, "wapf", wapf_norm="median_self")`
leaves `features/wapf/W8_H4_norm.npz` with the scalar `wapf_norm == "median_self"` and the params
record it; the default path is unchanged.

**B20. The template CLIs overwrite (CERT 6.12, 7.7; E1 6.13).** `gates_calibration pass-table`,
`gates_precondition gk0-template`, `gates_precondition idle-admissibility-template`, `series
head-drop-template`: refuse to overwrite an existing file unless `--force`, printing `kept: author
input exists: <path>` and exiting 0; the `write_*_template` functions are unchanged (the existing
tests call them directly). Test: run each CLI twice; the file's sha256 is unchanged after the second
run; with `--force` it is rewritten.

**B21. `_schema_compat.py` deleted (E1 4 "can be deleted").** Delete the file; `series.py`'s
import becomes `from plan11_encoding_ladder import schema` with no fallback. Test
(`tests/test_schema.py`, appended): `_schema_compat.py` does not exist and `series.schema is
schema`.

**B22. The runbook's pytest note (E1 4 "Documentation known to be stale").** `tests/
test_runner_guard.py` (new; a `unittest.TestCase`): the one test asserts
`os.environ.get("PYTEST_CURRENT_TEST")` is set (pytest sets it during every test) and otherwise
fails with the message `the gate tests are pytest functions that unittest does not discover; run:
python3 -m pytest -q tests`. `RUNBOOK.md` section 0 and 0a: the two sentences counting 96 of 159 are
replaced by one: `python3 -m pytest -q tests` is the only supported runner and `python3 -m unittest`
stops on `test_runner_guard` with the same instruction. Test: the guard passes under pytest (it is
in the suite); `python3 -m unittest tests.test_runner_guard` run by the test through `subprocess`
exits non-zero with that message in its output.

**B23. The stale SPEC.md section 7 move table (E1 4, 6.65).** `SPEC.md` section 7: the move-3 row
lists five G-C rungs with combined last; the move-12 row adds `gl (all rungs)` and `gf --all-rungs`
at the selected points before `tables`; the move-13 row reads the `gf-check` on Table 7; a sentence
that the driver is `run_moves.py` with `driver.py` as its alias and that `gp` runs at move 2 and
`alias` at moves 6 and 7. Builder A's move-14 row is appended separately (Part 1.7). Test: B30's
runbook check parses every command in SPEC section 7's table against the module CLIs (the check
script of CHECK_3 record (5), reimplemented as a test that `argparse` accepts each command line
with `--help`-free parsing of its flags).

**B24. The idle cells' stored archetype (CERT 6.11; E1 6.59; SPEC 3.1.5).** `series.build_features`:
rows of a cell with `role == "idle"` are written with `archetype = "IDLE"` and `kernel = "idle"`
whatever `cells.csv` carries; `cells.csv` and the sidecars are unchanged (`control`, the label-derived
name: `tests/test_extract.py` asserts them). The docstring says so. Test: on a `synth.py` corpus with
idle cells (whose `cells.csv` says `control`/`sleep`), the feature file's idle rows read `IDLE`/`idle`.

**B25. `--n-estimators` on the driver (E1 6.61).** `run_moves.py`: `--n-estimators` (default 300)
appended to `gord`, `splits`, `gf`, `gx`, `gdim`, `gm` (a `NEST_COMMANDS` set beside
`NJOBS_COMMANDS`, builder B's region); read with `getattr`. Test: `build_plan` with
`n_estimators=10` carries `--n-estimators 10` on those commands and on none other.

**B26. The G-ORD cost flags (needed by B28).** `run_moves.py`: `--gord-n-order-perm` (default 20)
and `--gord-null-perm` (default 100) passed to `gates_temporal gord` as `--n-order-perm` /
`--null-perm` when set (the CLI flags exist). Test: in `build_plan`.

**B27. `perm_floor` and the smoke run's promise (CHECK_3 M3, the runbook side).** `RUNBOOK.md` 0b:
one sentence that at `--null-perm 20` every null-judging gate (B1-G1, G-F (i), G-X, the clustering)
reads `not run: 20 permutations < 500` by design and the numbers stay in the files.

**B28. The cross-builder driver test (E1 4 "Not reached by any automated test"; CHECK_1 M10;
CHECK_2 M17).** `tests/test_driver_end_to_end.py` (new): `synth corpus --root <tmp> --reps 2
--idle 2 --n-pairs 60`; `run_moves run --out <tmp>/out --root <tmp> --moves 0-13 --null-perm 5
--n-estimators 10 --n-jobs 2 --gord-n-order-perm 2 --gord-null-perm 5 --duration-s 38.64
--assume-failed-zero --assume-reason "end-to-end test"` exits 0; every name in `tables.TABLE_NAMES`
has its CSV, every name in `figures.FIGURE_NAMES` its PDF and PNG (or `SKIPPED.txt` exists),
`report/paper2_skeleton.tex` and `report/manifest.json` exist, no command in the ledger has a status
starting with `failed`; then `run_moves run --out ... --moves 0-13 --dry-run` (same flags) records
zero commands with a `stale` key (B2). The test runs under five minutes on the reference machine
(eight cores) and says so in its docstring; it is not marked slow.

**B29. Signatures builder A relies on stay frozen** (Part 3.3). No test; a rule.

**B30. The runbook and SPEC command check as a test.** `tests/test_epoch2_fixes.py::
test_runbook_and_spec_commands_parse`: every fenced command line in `RUNBOOK.md` and every command
cell of `SPEC.md` section 7's table that starts with `python3 -m plan11_encoding_ladder.` or a
module name is split with `shlex` and its flags are checked against the module's `argparse` parser
(`parse_known_args` on a parser built by the module's `main` with `--out /x`-style placeholders
substituted); unknown flags fail the test. This keeps the runbook honest for both builders' flags
(builder A's move-14 section included, once present).

**B31. Builder B's report.** `plan11_encoding_ladder/BUILD_epoch2_B.md`: files touched, the item
numbers B1 to B30 with their tests, the full-suite result line, deviations, and Part 4 items
confirmed as defaults.

---

## Part 3. The boundary between the two builders

### 3.1 Files each builder may touch (no overlap)

Builder A only: `comparators.py` (new), `tests/test_comparators.py` (new), `figures.py`,
`latex_skeleton.py`, `schema.py` (append the `COMPARATORS` block only), `BUILD_epoch2_A.md` (new),
and the regions of shared files named in 3.2.

Builder B only: `gates_precondition.py`, `gates_calibration.py`, `gates_temporal.py`,
`gates_readings.py`, `gates_comparison.py`, `models.py`, `variance.py`, `splits.py`, `nulls.py`,
`extract.py`, `synth.py`, `series.py` except the `PAIR_RUNGS` line, `_schema_compat.py` (delete),
`tests/test_epoch2_fixes.py` (new), `tests/test_driver_end_to_end.py` (new),
`tests/test_runner_guard.py` (new), the existing gate test files (append only), `tests/test_synth.py`
and `tests/test_extract.py` (append only), `tests/report_fixtures.py` (append only, never a changed
fixture value), `BUILD_epoch2_B.md` (new), and the regions of shared files named in 3.2.

Neither builder: `verdicts.py`, `_b2_common.py`, `_synth_b2.py`, `requirements.txt`, `__init__.py`,
the check, fix, certify and build reports of epoch 1, and every file outside `plan11_encoding_ladder/`.

### 3.2 Shared files, by region (targeted edits only; never a whole-file rewrite)

| File | Builder A's regions | Builder B's regions |
|---|---|---|
| `run_moves.py` | `MAX_MOVE` and the bound in `parse_moves`; the `--moves` default and the four flags of Part 1.7(b) in `_add_run_args`; the move-14 block of Part 1.7(c) | everything else: `_add_run_args`'s other new flags (B1, B10, B12, B19, B25, B26), the move-2 and move-1 args, `temporal()`, `splits()`, the move-12 `gdim` line, `_stale_reason` and the `inputs=` lists (B2), the `gl2-rerun` internal step (B10), `NEST_COMMANDS`, the seed/n-jobs loop |
| `tables.py` | `table7` (the M5 line only), the new `table7_comparators` and `table_comparators` functions, `_gate_csv` and the indirection inside `_gl_text`, `_gdim_text`, `_gm_text`, `TABLE_NAMES`, `CITATIONS`, `run`'s two new dispatch lines | `table6`, `wapf_over_apf`, `table4_status` (B1, B8) |
| `series.py` | the `PAIR_RUNGS` line | everything else (B9, B21, B24) |
| `tests/test_driver.py` | the two one-token amendments of Part 1.7 | nothing |
| `RUNBOOK.md` | the new "Move 14" section; one sentence on `--moves 0-14` in the driver section | sections 0, 0a, 0b, the driver section's resume rule, moves 2, 6, 7, 12 |
| `SPEC.md` | one appended move-14 row in section 7's table | the rewritten rows 3, 12, 13 and the driver sentence of B23 |

A builder who finds that a needed change falls in the other builder's region does not make it;
the builder records it in the build report as an open item with the exact line, and, where the
epoch's own tests would otherwise fail, writes the test to tolerate both states (Part 1.8's
`feature_count_used` fallback is the model).

### 3.3 The one shared interface: frozen signatures and files

Builder A calls and builder B may extend but not change (new keyword parameters with defaults only;
no renamed or removed parameter, no changed return type, no changed file layout or column name):

`models.run_split_stage(out, rung, grid_id, split, labelspace, *, normalized, n_perm, seed, n_jobs,
feature_drop, include_idle, n_estimators, run_null, seed_offset, reduce_to, reduce_method, test_frac,
loro_mode, base_dir, label_override, quarantine, unit_rule)`; `models.split_dir`;
`models.effective_scores`; `models.prepare_split_data`; `models.B1G1_MIN_PERM`, `models.N_ESTIMATORS`;
`gates_comparison.gm_compare(scores_a, scores_b, spread)`; `gates_comparison.gate_gx(out, rung, *,
n_perm, n_jobs, n_estimators, seed_offset, grid_id)`; `gates_comparison.GL_COLUMNS`;
`series.load_features`, `series.features_path`, `series.admissible_cells`, `series.gk0_relabel`,
`series.load_cells`, `series.load_extract_cached`, `series.k_median_cell`, `series.head_drop_for`
(the two-argument form keeps working), `series.load_head_drop`, `series.selected_grid_id`,
`series.write_csv`, `series.write_json`, `series.write_params`, `series.read_csv`,
`series.read_json`, `series.inputs_sha256`, `series.fmt_num`, `series.WHOLE_GRID_ID`;
`splits.make_labels`; `nulls.null_summary`, the four seeds; `extract.open_text`, `extract.Refusal`;
the layouts `features/<rung>/<grid_id>_{raw,norm}.npz` (keys of 1.6),
`gates/splits/<rung>/<grid_id>/<split>__<labelspace>[__raw]/scores.json` and its keys,
`gates/gx.csv` columns and its per-rung replacement, `gates/gm.params.json` `params.spread`,
`gates/gdim.csv` and `gates/gl.csv` columns, `gates/preconditions.csv` `all_hard_pass` and
`failed_verdict`, `gates/preconditions.json` `excluded_cells` and `excluded_cells_pair_rungs`,
`gates/gk0.csv`, `gates/selection.json[rung].grid_id`.

Builder B relies on nothing of builder A's.

The driver hook (Part 1.7) is builder A's; builder B is forbidden from touching the three places
named there. The Table 7 comparator rows and the two new tables are builder A's; builder B does not
add rows to any table.

### 3.4 Reports and the final suite

Each builder, at the end: `python3 -m pytest -q tests` from `plan11_encoding_ladder/` with zero
failures (skips allowed only for the `zstandard` reader), the last line pasted into the build
report; every existing test unchanged except the two amendments of Part 1.7; every new test listed
by name and item. The two reports `BUILD_epoch2_A.md` and `BUILD_epoch2_B.md` have the sections:
what was built, files touched (with regions), tests added, the suite's last line, deviations from
this file (each with the reason), open items in the other builder's regions, and the Part 4 items
this builder implemented with their defaults. `apf_paper/EPOCH2_BUILD_REPORT.md` is assembled from
the two by the epoch's reporter afterwards; no builder writes it.

---

## Part 4. For the author: every choice left as a parameter, with its default

Each is a module constant or a CLI flag whose value is written into the result files' `params`. The
default runs unless the author says otherwise.

1. `savoldi_rows = "all_after_head_drop"` (`comparators.py`; `--rows`). C14 says "over consecutive
   snapshots"; K is defined on every extract row, so every row after the head drop is used, the last
   `seq` included. Alternative `"rung_series"` (the rows the rungs use, last `seq` dropped).
2. `savoldi_ddof = 1` (`--ddof`): the sample standard deviation, C14's wording ("sample mean and
   standard deviation"). Alternative `0` (population, the toolkit's shape features' convention).
3. The Dhodapkar-Smith threshold grid `(0.04, 0.1, 0.2, ..., 0.9)` with `delta_th_default = 0.04`
   (`--grid`, `--delta-th-default`). The brief declares the nine points 0.1 to 0.9; AA 2026-09-17
   declares 0.04 as the default, which is not on that grid; this file puts both on one grid so that
   the author's default is a computed point and every one of the brief's points is kept. AA T1's
   companion decision names no other value. Every grid point is written to `dhodapkar_sweep.csv`;
   the default is marked, never selected against labels.
4. `dhodapkar_boundary_rule = "gt"` (`--boundary-rule`): a phase boundary when `delta > delta_th`
   (C14: "a phase change when delta exceeds a threshold"). Alternative `"ge"`.
5. `dhodapkar_phase_length_rule = "n_over_b_plus_1"` (`--phase-length-rule`): the mean phase
   length is the mean length of the `B + 1` segments the `B` boundaries cut the cell into
   (`n / (B + 1)`), the first and last partial segments included. Alternative `"interior"` (only
   segments between two boundaries; NaN below two boundaries). C14 names "average phase length"
   without fixing the end segments.
6. `law_x_default = 4` (`--x-default`) on the declared grid `(2, 4, 8, 16)` (`--x-grid`). The
   definition names no X; 4 is the smallest grid point above the trivial `X = 2` (where
   dynamic-for-2 is the changed set itself and static-for-2 its complement). Every X is kept in
   `law_sweep.csv` and `law_series/`.
7. `law_x_unit = "dumps"` (`--x-unit`): "dynamic in X" is a membership run of length `X - 1`
   (C14's exact reading: X dumps span X - 1 pairs). Alternative `"pairs"` (a run of length X, the
   brief's "in every one of the X changed sets").
8. `law_head_drop_rule = "runs_from_seq_first"` (`--head-drop-rule`): the run-length state starts
   at the first `seq` and recording starts after the head drop. Alternative `"reset_at_head_drop"`.
9. `law_feature_source = "window"` (`--feature-source`): the Table 7 feature vector is the mean and
   SD over pair positions of the dynamic-for-X and static-for-X counts. Alternative `"ever"` (the
   count of distinct pages ever dynamic or static for X over the cell). Both are in `law_sweep.csv`.
10. `comparator_norm_rule = "median_K"`: the level-normalized comparator features divide the count
    features by the cell's median K (P2 Sec. V G-L (i), the count rung's rule); Dhodapkar-Smith's
    counts are divided by the pair count instead (the delta is level-free). Part 1.2 to 1.4 state
    what each normalized row means; the raw row is the method as published.
11. `table7_variant = "both"` (`tables.py` constant `TABLE7_VARIANT`; not a CLI flag in this epoch):
    Table 7's comparator block prints the raw row (as published, level-inclusive by construction)
    and the level-normalized row for each comparator and split. Alternatives `"raw"`, `"norm"`. The
    G-M cell of each row compares like with like (raw against APF raw, norm against APF norm).
12. G-M's `spread` for the comparator rows is read from `gates/gm.params.json` (the five-seed spread
    measured on APF's LOKO run at move 12), not re-measured. A re-measurement would be the same
    five runs.
13. Gates not computed for comparators in this epoch: G-J, G-V, the clustering, G-F (ii), G-P,
    G-DEC, G-C, the temporal gates and G-ORD (Part 1.5 gives the printed string for each). If the
    author wants G-F (ii)'s floor envelope on Savoldi's mean K or G-V on the comparator vectors, it
    is a later item.
14. C1's rule (B1): `c1_floor = on`, `c1_activity_min_pages = 200` (AA T1). AA 2026-09-17's second
    bullet gives the interim as 0.1 percent of memory (262 pages) while AA T1 gives 200 pages; the
    brief instructs 200 and cites T1, so 200 is the default and `--c1-activity-min-pages 262` is one
    flag away. The floor is strict (`K_max > edge`, T1's "exceeds"); the interim is `>=`, T1's
    `K_max >= 200`. The inherited fraction rule stays reachable through `--c1-activity-min` for the
    record. The floor's idle cells are those passing C2 and C6 (their own admissibility, C1 being
    not applicable to them); G-K0 does not require the admissibility record and neither does C1
    (E1 6.18 stays open).
15. `perm_floor` (B3): 500 on the CLIs of `gf`, `gx`, `cluster` (B1-G1's rule, CR 2.1 item 9);
    0 at the function level so the existing direct-call tests keep their contract. The driver runs
    the CLIs, so the paper's path refuses an under-powered null.
16. `matched_splits = "all"` (B4): the `combined (matched)` comparison for every Table 7 split (SPEC
    6.3's reading), each null governed by `--null-splits`. Alternative `"loko"` (CR 2.2 item 32's
    one-row reading).
17. `g1_none_applicable = "drop"` (B7): G1 leaves the roll-up when no kernel is applicable, as G2
    does. Alternative `"refuse"`.
18. `--gl2-rerun auto` (B10): the driver runs the G-L (ii) consequence (SPEC 3.7.4) into
    `gates/splits_gl2drop/apf/`; `manual` records the step as the author's.
19. `SynthSpec.duration_s = n_pairs * 0.644` (B12) for the synthetic corpus; the extractor's
    `--duration-s` defaults to 600 and the runbook's smoke command passes `77.28`. G2 and G-P now
    read each cell's declared duration from its sidecar instead of the constant 600; on the real
    corpus every sidecar says 600 and no number moves. The pass table's count is per cell
    duration (per 600 s on the real corpus; the column name `passes_per_600s` is kept).
20. The staleness trigger for admissibility stays `gates/preconditions.json` (CERT 1(a)); the
    runbook says gate files are not edited by hand (B14). CERT 7.1's `csv-drop:` form is not built.
21. `gf_part1` leaves `table5_grid.csv` (B15, CERT 7.2's first form); `gates/gf.csv` is the one
    place G-F's verdict lives.
22. `wapf_norm` is now exposed on `grid` and the driver (B19); the default `median_K` is unchanged.
23. B1-G3 runs on the comparator vectors (Part 1.5) because it is part of the split stage every row
    uses; with two to four features a quarantine can leave one feature. `quarantine=True` is the
    split stage's default and is not changed for comparators.
24. The comparator statistics are computed on every `ok` cell and admissibility is applied at read
    time by the split stage, as for the rungs; the per-cell CSVs carry `admissible` and
    `excluded_pair_rung` so the author can see which rows entered the splits.
25. The two one-token amendments of `tests/test_driver.py` (Part 1.7): the move bound moves from
    13 to 14 and the two assertions that 14 is out of range become assertions that 15 is. No other
    existing test changes.
26. Amendment of 2026-09-28 (AA A8). Three recordings hold an unplanned second run after the
    sustain loop relaunched the workload and the guest clock stepped back: gemm seed 7703 (first
    run pairs 1 to 940), fem_assembly seed 2714 (1 to 895; its folder name truncates the seed to
    `271`) and fft seed 7548 (1 to 926); fem_assembly from the `sandbox_deepdive_01c` launch, gemm
    and fft from `sandbox_deepdive_01c1`. The extract (move 1) takes `--keep-first-pairs <csv>`
    (columns `path, keep_first_pairs, reason`; the declared file is
    `declared/keep_first_pairs.csv`) and, for a listed cell, reads only the first N pairs of the
    file in seq order: every row with `seq <= seq_first + N - 1`, a gap counting as a pair; later
    rows are ignored and nothing under the retention root is written. The sidecar records the
    cut, the reason and the file's own pair count; `n_pairs` and `dt_est_s` are over the kept
    pairs. A row that matches no cell or several, or an N above the file's pair count, stops the
    command. A sidecar written under another cut is re-extracted. The driver passes the flag to
    move 1 and declares the file as an input, so its sha256 is in the ledger and a change makes
    move 1 stale. This changes which pairs are read, not any gate: G-P, G-D, G-ORD, G-X and G-F
    are unchanged and read the cut extract as they read any other.
27. Amendment of 2026-09-28 (AA A12). `LEVEL_MATCHED_SETS` gains a third set, C = fft,
    stencil_jacobi (4,096 pages by source), after the two declared sets, which stay exactly as
    declared and in the same order: A = floyd, histogram, nbody; B = fft, gemm. B is kept on
    purpose: the declared expectation was that the count cannot separate fft and gemm, and the
    row reports what the data say. `LEVEL_MATCHED_ADDED` records, per set index, the date it was
    added, and every output that lists sets carries a status of `declared` or `added 2026-09-28`:
    Table 3's csv, md, tex and params (`status` column, `params.set_status`), the level-matched
    figure's panel titles, and `gates_calibration.separating_features` rows (`status`) with
    `gates/alias.csv` (`set_status`). Set letters are A, B, C. In Table 3's `.tex` the added row's
    label carries a marker and a note below the table reads
    `\slot{note: C added 2026-09-28, after the declared sets; wording by the author}`; the
    wording is the author's. The per-kernel `level set` column lists every set a kernel is in
    (fft: `B, C`). This adds a row; it removes nothing and changes no gate.
28. Amendment of 2026-09-28 (verified that day by running `extract index` on an empty copy of
    the real corpus layout). The campaign names a cell folder by the retention signature of its
    command, cut at 60 characters plus an 8-hex sha1 fingerprint (`plan07_campaign/ui/place_csv.py`
    `signature`). For gibbs, histogram, rmat_gen and spmm the cut falls before the seed, so the
    name shows none; for fem_assembly it falls inside the seed (2714 reads as 271). The index
    (move 0) read the seed from the name, gave every seedless cell rep 0 and one shared `cell_id`,
    and reported them `ok`; move 1 would have collapsed the eight runs of each such kernel into
    one folder without an error. Now `extract index` takes `--seed-map <csv>` (`path, seed,
    source`; the declared file is `declared/seed_map.csv`, 96 rows made by
    `declared/make_seed_map.py`, which rebuilds each folder name from the corpus settings for
    every seed 0 to 99,999 and keeps the unique exact match). A listed kernel cell takes its seed
    from the map; a seed the name also shows must be a leading-digit prefix of the map's seed
    (truncation), else the command stops naming the cell. `cells.index.json` records which cells
    took their seed from the map. A kernel cell with no seed from either source is listed as
    `refused: seed unknown` and is never given rep 0 by default; idle cells keep
    `rep = rep_dir - 1`. Every copy of a `cell_id` that appears more than once is refused
    (`refused: duplicate cell_id`), so move 1 can never write two runs to one folder. The driver
    passes `--seed-map` to move 0 and declares the file as an input (sha256 in the ledger). This
    changes where the seed is read from, not any gate.
29. Amendment of 2026-09-29 (the author's decision during the paper run `eusipco_run_20260928`).
    On the 4-core server, LORO's 500-permutation null measured about 33 hours for one (rung,
    label space, variant), so a run with every LORO null would take about three weeks. The paper run
    therefore uses the fallback the runbook already names (SPEC section 8 item 25):
    `--null-splits loko,within_trace`. LORO's accuracy is still computed and reported; its null
    column reads `not run`. The skipped nulls become move 15, an optional move after move 14 that
    the default `--moves 0-14` never runs (`MAX_MOVE = 15`): LORO's split stage with its null for
    every rung at its selected point (same seeds, so the scores are unchanged and only the null is
    added), then G-DIM and the comparators under the full `loko,loro,within_trace`, then G-L, G-M
    and the tables. Move 15 is meant for a many-core machine. This changes when LORO's null is
    computed, not any gate or its rule; a table built before move 15 prints `not run` in LORO's
    null cells and says why. `tests/test_driver.py` changes in two tokens and
    `tests/test_comparators.py` in one: the out-of-range move in their three bound checks becomes 16.
30. Amendment of 2026-10-05 (the author's decision after the run of 2026-09-29; `apf_paper/P2_AUTHOR_ANSWERS.md`
    A20, A21, A22). A20 traced why Table 2's readings were refused: the instrument check for the content rung
    (`gates_calibration.gate_gc`, `content_rows`) orders gibbs < histogram < gemm by `l1_q50_per / 4096`, the byte
    change averaged over the whole page, while the prediction it carries is about the size of a change per changed
    byte; the page average multiplies that size by the share of bytes changed and puts gibbs and histogram on the
    same level, so the check's statistic does not test its own prediction (on the run of 2026-09-29 the predicted
    order holds in all 8 runs per changed byte and in 2 of 8 per page). G-F part (i) voided wAPF, persistence,
    content and combined besides. A21 corrects the check's statistic to the median over pairs of
    `r_l1l0_q50_per`, the change per changed byte on persistent pages, the l0 ordering unchanged, and keeps the
    original verdict beside the corrected one, dated. A22 adds one move, 16, labelled "added 2026-10-05, after the
    run of 2026-09-29", opt-in like move 15 and never part of the declared moves 0 to 15 (`MAX_MOVE = 16`): (1) the
    corrected check (`gates_calibration.gate_gc_corrected`, `gates/added/gc_corrected.csv`; `gate_gc` and
    `gates/gc.csv` unchanged); (2) the idle common-ground test (`gates_idle_common_ground.run`): leave-one-run-out
    with the 8 idle runs as a 13th class beside the 12 kernels (`models.prepare_split_data(include_idle=True)`,
    kernel labels), per rung at its selected point, normalized features, the recall of every class, the LORO null's
    run-level label shuffles scored on the accuracy and on the idle recall (`--null-perm`, default 500; below 500
    the verdict reads `not run: N permutations < 500`, numbers kept), under `gates/added/idle_common_ground/`;
    (3) a second Table 2, `report/tables/eusipco_table2_corrected.*`, the same columns and scores as the declared
    one, built by `tables_eusipco.table2_corrected` from `tables.table7_rows` under the twin refusal
    `_report_common.rung_override_corrected`: the corrected instrument check in place of the original, and in
    place of G-F part (i) the rule that a reading is void only when a held-out idle run is not recognised as idle
    above chance (idle recall not above the null's 95th percentile). The declared Table 2 and every existing gate
    file stay as they are; both tables are reported, and the `.tex` carries a `\slot` note for the author's
    wording of the disclosure. Traced: the window rule (`gates_temporal.select`) chooses by G1, G2 and G4 and
    attaches the G-C verdict only as a label, so the correction moves no selected window; Table 2 applies its
    refusals in `_report_common.rung_override` (G-C first, then G-F part (i)), which is why the twin lives there.
    `tests/test_driver.py` changes in two tokens, `tests/test_comparators.py` in one and
    `tests/test_plan10_bridge.py` in one: the out-of-range move becomes 17 and the console's board has 18 rows.

31. Amendment of 2026-10-06 (the author's decision after the SPL paper's new-block test; `apf_paper/P2_AUTHOR_ANSWERS.md`
    A25; `plan12_grounding/PROMPT_new_blocks_test.md`). One optional move, 17, labelled "added 2026-10-06", never in
    the default `--moves 0-14` (`MAX_MOVE = 17`): the new-block test on the five readings. A model sees the first
    round(0.8 n) windows of every admissible cell (the hard-excluded cells out, as in move 16; the 8 idle runs in, as
    one more answer at every level), in time order; one window is skipped, so that no new window shares a pair with a
    seen one (asserted from the pair ranges; the within-trace split has no gap: at 64 x 32 its last training and
    first test windows share 32 pairs); the rest are new blocks named alone and in pools of 2 and 3 consecutive
    blocks (majority vote, ties by the summed class probability) at three levels: archetype (the feature files'
    archetype_predicted, no G-K0 relabel; "idle" for idle), kernel (13 classes), run (each cell its own class). The
    readings are the toolkit's five from their own feature files `features/<rung>/<grid>_norm.npz`, at W64_H32 for
    every reading (the primary) and at each reading's own selected window (the secondary rows; "same as primary"
    where the own window is W64_H32); the comparators take no part (one window per run). One forest per level,
    reading and window on the seen windows with the toolkit's settings, its dimension rule (applied as written; no
    reading has more features than training cells) and its L1 quarantine at the block; the rules that do not apply
    at window level (the splits, the label-shuffle null, the cell-majority unit, G-N's headline macro recall, LOKO's
    per-fold majority baseline, the G-K0 relabel) are named in params.json and replaced by nothing. Scores per level,
    reading, window and pool size with a 95% bootstrap interval over recordings (1,000 resamples) on every accuracy
    and on the margins against APF; per-kernel recall; accuracy by block position; the run level read against idle.
    The cut is the run's own (`inputs/head_drop.csv`). `new_blocks.py` copies the split, the gap check, the pooling,
    the bootstrap and the scoring from `plan12_grounding/new_blocks.py` (commit 6c13f9e; plan11 never imports plan12).
    Outputs under `gates/added/new_blocks/<window>/`, `gates/added/new_blocks.csv`, `report/tables/new_blocks_*` and
    `report/figures/new_blocks_*.svg` with their CSV data. The driver declares every file each command reads; no
    existing command changes its arguments or declared inputs, with one deliberate exception: move 16's
    `tables_eusipco table2_corrected` now declares `report/tables/table7_comparators.csv`, which it reads, so on a run
    where move 16 ran that one command is reported stale, and nothing else. The console's waiting rule for the
    optional moves 15, 16 and 17: each waits for the moves whose files it reads, not for the move numbered before it;
    moves 0 to 14 keep the runbook's order. `tests/test_driver.py` changes in two tokens, `tests/test_comparators.py`
    in one and `tests/test_plan10_bridge.py` in one: the out-of-range move becomes 18 and the console's board has 19 rows.
