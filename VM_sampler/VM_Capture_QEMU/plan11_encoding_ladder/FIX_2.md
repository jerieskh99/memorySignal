# FIX_2.md: fixer's report on the paper 2 analysis toolkit, cycle 2 (2026-09-16)

Rules kept: no server, no path under a server mount, no remote command; the sandbox family's
sources and workload names were not read, grepped or named; no paper prose was written (the one
change to the delivered skeleton is the escaping of two Table 2 cells, shown as a diff below); no
file outside `plan11_encoding_ladder/` was edited except the regenerated `apf_paper/p2_skeleton.tex`
(inside the paper 2 folder the task names); nothing was committed; `P2_STRUCTURE.md` was not touched.
Every test and every driver run below used synthetic data generated on this machine with builder 1's
generator. Environment: Python 3.10.12, numpy 2.2.6, scikit-learn 1.7.2, scipy 1.15.3, matplotlib
3.10.7, pytest 9.1.1, the `zstd` binary on PATH, the `zstandard` module absent, no TeX compiler.

What was applied: the checker's one blocking finding (B1), al-Farabi's two fails under condition (1)
(his corrections 7.1 and 7.2), and the minor findings whose fix is one line (M2, M5, M6, M9, M11,
M12, M13, M14, M15, and one sentence each for M7 and M8). Refused, with the reason in each entry:
M1, M3, M4, M10, M16, M17, the code parts of M7 and M8, and al-Farabi's 7.3 to 7.6, which are not
fails of his certification. No gate threshold and no verdict string was changed.

---

## 1. The checker's findings

### B1. G-L, G-DIM and G-M judged the full-model score while Table 7 printed the re-run's. Fixed.

The reading applied is the one SPEC 3.7.2 states ("the split is re-run without the quarantined
features and both scores are kept ... Table rows use the re-run"), which is also what builder 3
already did; the change makes builder 2 read the same way, through one function that both builders
call.

- `models.py` 341-375: a new `effective_scores(sc)` beside the writer of `scores.json`. When
  `with_quarantine` carries a non-empty `quarantined_features`, the re-run's keys (`accuracy`,
  `macro_recall`, `recall_per_class`, `recall_per_kernel`, `majority`, `b1_g1`, `b1_g1_rank`,
  `null_p95`, `feature_count`) are merged over the full model's, `quarantined_features` is carried
  and `score_source` says `with_quarantine ...`; otherwise the document is returned as written. The
  full model's keys stay in the file under their own names for the record. When every feature was
  quarantined (no re-run exists) the score keys become None and `b1_g1` carries the `not run: every
  feature quarantined` string, so no consumer prints or judges the full model in that case either.
  The docstring cites SPEC 3.7.2 and CR 2.1 item 10.
- `gates_comparison.py` 42-51: `_scores` reads through `models.effective_scores`, so G-L (i)
  (`gate_gl`), G-DIM's `d*` and feature counts (`gate_gdim`) and G-M's scores and per-kernel recalls
  (`gate_gm`) judge the re-run. Each of `gl.params.json`, `gdim.params.json`, `gm.params.json` now
  carries `score_source` (the string `SCORE_SOURCE`, line 42). In `gate_gl` the `not run` guard on
  `b1_g1` now comes before the missing-score guard (lines 95-103), so a smoke run or an
  every-feature-quarantined split writes its own string rather than `scores.json missing`; a
  `not applicable:` split carries its own string too.
- `_report_common.py` `effective_scores`: delegates to `models.effective_scores` when builder 2's
  module is importable, and keeps an identical copy as the fallback for a process without it. The
  test below patches the import away and asserts the fallback gives the same reading on every shape
  of the document (no quarantine, a quarantine, every feature quarantined, None).
- The per-unit predictions: `models.run_split_stage` (lines 492-512) now keeps the re-run's
  predictions and writes them beside the full model's as `predictions_with_quarantine.csv` (same
  columns, same directory), removes a stale copy when a later run quarantines nothing, and records
  `predictions_file` and `score_source` in `scores.json` and `score_after_quarantine` in `params`.
  `tables._loko_assignments` (Table 8's LOKO assignments) reads `predictions_with_quarantine.csv`
  first when it exists, and `table8.json` `params` records which file it read
  (`predictions_file`). G-X's held-out campaign is unaffected: `gate_gx` runs with
  `quarantine=False`.
- The exclusion list: `excluded_rows.csv` was written from the full model's B1-G1 verdict; it now
  follows the same reading (`models.py` 514-519): the row is excluded when the rung's score (the
  re-run when a feature is quarantined) is `near_unfalsifiable`, with that score. SPEC 3.7.1 ("a
  NEAR_UNFALSIFIABLE row is written to excluded_rows.csv and never printed") read with 3.7.2 ("Table
  rows use the re-run") admits only this reading; the other would exclude a row the tables never
  print and print a row the list never excludes.

Test added (`tests/test_gates_comparison.py`
`test_gl_gdim_gm_judge_the_rerun_after_a_b1_g3_quarantine_and_the_tables_print_the_same_number`):
on a level-only corpus the LOKO/archetype `scores.json` is written with the full model at 0.8
against a null p95 of 0.6 and a `with_quarantine` block at 0.5 (two features quarantined, six
left); it asserts that `gl.csv` part (i) reads `level only` with `score_norm` 0.5, that
`gdim.csv` prints `d = 6` for APF, that both `params` files carry `score_source` naming
`with_quarantine`, and, on the same run, that Table 7's APF LOKO row prints the re-run
(`near_unfalsifiable` in the score cell, feature count 6, `level only` in the G-L cell); then that
without a quarantine both places print 0.8; then that with every feature quarantined G-L writes
`not run: every feature quarantined`. The scores document is written by the test rather than fitted,
because a fitted full model that beats the null while its re-run does not is not reproducible at
test size (it needs a 500-permutation null on a forest); the interface is what B1 is about and the
written document exercises every branch of it. The fitted case is covered by
`tests/test_gates_models.py::test_b1_g3_quarantines_the_level_feature_on_raw_apf_of_a_level_only_corpus`,
extended to assert that `predictions_with_quarantine.csv` exists with as many rows as
`predictions.csv`, that `scores.json` names it, and that `effective_scores` returns the re-run's
accuracy; and that a run without a quarantine writes no such file and names `predictions.csv`.

The in-situ check on the driver run is in section 3.

### M1. `--c1-activity-min` not reachable from the driver or the runbook. Refused.

More than one line (a driver argument, its pass-through, the ledger, the runbook sentence), and the
decision it serves is the author's ("For the author" item 1 of CHECK_2; al-Farabi 6.3). Refused for
the same reason in cycle 1. Note that after the al-Farabi 7.1 fix below, a preconditions re-run by
hand with another threshold does make every later move stale, which is the part of M1 that was a
correctness matter; the runbook's resume paragraph now says so.

### M2. The floyd decay figure was a placeholder under `decay (floor unmeasured)`. Fixed (one line).

`figures.py` 471: the comparison is on the head of the verdict, `str(verdict).split(" (")[0] !=
GDEC_DECAY`; the title already prints the full verdict with its suffix (line 498).

### M3. Table 7's `feature count` for `combined (matched)` prints the pre-reduction width. Refused.

Three lines across two builders (two in `models.run_split_stage`, one in `tables.table7`); not one
line. Listed for the author under item 3 of section 4.

### M4. The roll-up counts "no applicable kernel for G1" as a failed gate. Refused.

Two lines, and the decision is put to the author by both reports (CHECK_2 M4 offers the drop-from-count
form; al-Farabi 7.3 offers that form or an outright refusal of the selection, and his 6.4 asks the
author to say which). I do not choose between them.

### M5. Three runbook lines did not run as written. Fixed (three one-line replacements).

`RUNBOOK.md` moves 10, 11, 12: each `gates_temporal grid/g3/gord/select` shorthand is replaced by
`python3 -m plan11_encoding_ladder.run_moves plan --out <out> --moves <n>` with the comment that
the plan prints the four `gates_temporal` lines (grid, g3, gord, select) as move 9 shows them.
Verified: `run_moves plan --moves 10` prints exactly those four command lines.

### M6. The runbook's `unittest` alternative runs 96 of 158 tests and reports OK. Fixed (three one-line edits).

`RUNBOOK.md` section 0 (prerequisites) and the test command in section 0b, and
`requirements.txt`: `pytest` is required; `unittest` discovers 96 of the 159 (the 63 gate tests are
pytest-style functions, the B1 test of this cycle among them) and reports OK without them;
`python3 -m pip install --user pytest` when absent. Verified: `python3 -m unittest discover -s tests
-p "test_*.py"` reports `Ran 96 tests ... OK (skipped=1)` while pytest collects 159.

### M7. `gord` and `grid` ignore `--n-jobs`; the runbook's cost claims are not borne out. Partly.

The measured cost is now stated in `RUNBOOK.md` section 0b with the checker's cycle-2 numbers
(G-ORD 424 to 772 s per rung at 12 cells and 20 permutations, single process; `--n-jobs` accepted and
not used by `gord` and `grid`; the split stages 176 to 332 s per rung; the better part of an hour
per rung for G-ORD at the 104-cell corpus). The parallelization of `gate_gord` and the driver's
`--n-estimators` pass-through are refused as more than one line. On my run (10 admissible cells, 80
pairs, 5 permutations) G-ORD took 11 minutes for APF, which agrees with the statement.

### M8. The G-L (ii) refusal's consequence is not executed by the driver. Partly.

`RUNBOOK.md` move 7 now carries one paragraph: on `refused: shot noise explains CV` the driver does
not schedule the SPEC 3.7.4 re-run; the author runs `models splits --rung apf --all-splits
--raw-and-norm --feature-drop cov,std,peak2med --null-perm 500` by hand (it overwrites
`gates/splits/apf/`, so the refused run is copied aside first), and the rung's numbers are not
citable while the refusal stands. The conditional driver command is refused as more than one line;
al-Farabi 6.7 puts the choice between the two forms to the author.

### M9. `--seed-offset` did not reach `gdim`; `--n-jobs` did not reach `gdim` or `gm`. Fixed (two lines).

`run_moves.py` 68-72: `("gates_comparison", "gdim")` added to `RANDOM_COMMANDS`; `gdim` and `gm`
added to `NJOBS_COMMANDS`. Asserted in `tests/test_driver.py`
(`test_move_table_carries_the_review_corrections`).

### M10. The idle cells' head drop is keyed two ways. Refused.

The one-line form inside `series.head_drop_for` cannot see the cell's role, so it would have to
fall back on the kernel name not being one of the twelve, which is a heuristic and not the rule the
checker states ("idle if role == idle"); the role-based form touches `head_drop_for` and six call
sites. Harmless at the default drop of 0, as the checker says. Listed for the author (item 4 of
section 4).

### M11. G-X wrote a confound verdict on a corpus with one campaign label. Fixed (one line).

`gates_comparison.py` `gate_gx`: when fewer than two campaign labels exist, `confound_verdict`
carries `not applicable: one campaign label` beside the leak verdict's same string. The confound
rule of CR 2.2 item 26 presupposes campaigns to be confounded with; with one label it yields
`partial` for every multi-kernel archetype by construction, which is what the checker asked to
replace. Never arises on the real corpus (three labels).

### M12. Table 5's `refusal` cell duplicated a disconnected lead. Fixed (one line).

`tables.py` 183: the `G-C: disconnected lead` part is appended only when the grid row's own
`refusal` does not already carry it.

### M13. Two Table 2 cells of the skeleton printed their math escaped. Fixed (one line).

`latex_skeleton.py` `_cell`: the text is split on `$` and the even segments escaped, so a math
span, whole-cell or inline, is left as it is. `apf_paper/p2_skeleton.tex` was regenerated with
`latex_skeleton --standalone`; the diff against the previous copy is the timestamp comment and the
two cells:

```
112c112
< 0 & APF & breadth & one scalar \$K\_t / N\$ & amount, identity, position & none beyond the row count \\
---
> 0 & APF & breadth & one scalar $K_t / N$ & amount, identity, position & none beyond the row count \\
114c114
< 1 & Persistence (Jaccard) & identity over time & one scalar \$J(t)\$ & amount, breadth (normalized out) & one set intersection per pair \\
---
> 1 & Persistence (Jaccard) & identity over time & one scalar $J(t)$ & amount, breadth (normalized out) & one set intersection per pair \\
```

### M14. Three gate-decision helpers carried no citation. Fixed (three lines).

`gates_calibration.k_jump_events` (P2 Sec. V 5.2 G-C; CR 2.2 item 22; SPEC_review_al_kindi.md
item 1), `gates_readings.dec_cell` (P2 Sec. V 5.2 G-DEC rules (b) to (d); CR 2.2 item 34; SPEC
3.6.2), `models.aggregate_units` (P2 Sec. V 'The splits'; SPEC 4.2; section 8 item 22).

### M15. G-X's split run recorded a meaningless majority baseline. Fixed (one line).

`models.majority_baseline`, the LOKO branch: the most populous label among the training kernels is
counted in the label space in use (`y_unit`) rather than always the archetype. For archetype labels
`y_unit` is the archetype map, so nothing changes there (the existing test
`test_b1_g6_majority_is_half_under_loko_on_twelve_kernels` still passes); for G-X's campaign labels
the LOKO rule of CR 2.1 item 11 is applied to campaigns, which is the rule's own reading and not a
new choice. `gx.csv` does not read the value.

### M16. Table 6 prints `--` for kernels excluded at C1. Refused.

More than one line (reading `preconditions.json`, mapping cells to kernels, replacing the row's
cells). Listed for the author (item 5 of section 4).

### M17. No test runs the driver across the three builders' modules. Refused as a new file.

The cross-builder assertion M17 asks for ("gl.csv part (i)'s score_norm equals Table 7's accuracy on
the same row") is carried by the B1 test above, which runs builder 2's `gate_gl` and `gate_gdim` and
builder 3's `table7` on one output directory and compares the cells. A full driver run across the
modules was done by hand in section 3.

---

## 2. Al-Farabi's certification

Condition (1) is the only one that fails, on the resume path; conditions (2) to (5) hold. Both of his
corrections for it were applied.

### 7.1. The admissibility record as a declared input of every step from move 3 on. Applied, with one change to the file named.

`run_moves.py` `build_plan`: every command from move 3 onward that reads admissibility now declares
the record in its `inputs` (the constant `ADMISSIBILITY`, 29 sites: `gc` x5, `gk0`, `gf` (both
runs), the move-5 figures, and for each rung `features`, `grid`, `g3`, `gord`, `select`, `splits`,
`gx`; `gl` (both runs), `gn`, `tables table6`, `gj`, the fused-plane figure, the j_hist figure,
`gdec`, the ratio/decay figures, `tables wapf_over_apf`, `gdim`, `gm`, `variance`, `cluster`,
`tables (all)`, `figures (all)`, `tables manifest`). Not declared: the two `alias` steps (they read
`pass_table.csv` and `g3_flags.csv`, the latter itself now downstream of the record) and the
skeleton (it reads no admissibility).

The file declared is `gates/preconditions.json`, not `gates/preconditions.csv` as 7.1 names. The
reason: `gates_temporal._refresh_c7` rewrites the C7 column of `preconditions.csv` in place after
`select apf` (move 6), so with the CSV as the trigger every step that ran before move 6 (G-C, G-K0,
G-F, the APF features, grid, G3 and G-ORD) would read `stale: preconditions.csv changed` on the first
resume although the admissible set had not changed, and G-F and G-ORD would re-run for nothing at
their full cost. `preconditions.json` is written by `gate_preconditions` only, from the same rows,
carries `excluded_cells` and `excluded_cells_pair_rungs` (the admissibility record al-Farabi names in
section 1 (a)), and `series.write_json` adds no timestamp, so its hash changes exactly when the
preconditions are recomputed. What admissibility reads at run time is the CSV's `all_hard_pass` and
`failed_verdict` columns, which the C7 refresh never touches (al-Farabi section 2), so the JSON is a
faithful proxy for the columns that matter. Verified by the probes in section 3: a preconditions
re-run at another C1 marks every step from move 3 on stale; a re-run at the same C1 leaves every
step skipped.

### 7.2. G-ORD's inputs a superset of the grid's triggers. Applied.

`run_moves.py` `temporal()`: the `grid` step's inputs are `cells.csv`, `inputs/head_drop.csv`,
`inputs/pass_table.csv`, `gates/preconditions.json`; the `gord` step's are those plus
`gates/gk0.csv`, so any rebuild of the grid re-runs G-ORD before `select` reads its label.
Asserted in `tests/test_driver.py` `test_move_table_carries_the_review_corrections`
(`set(grid.inputs) <= set(gord.inputs)` for every rung, and `gates/preconditions.json` in the inputs
of every non-internal step from move 3 on except the skeleton and the two `alias` steps) and
verified by probe C in section 3 (a changed `pass_table.csv` re-runs grid, gord and select, and the
rebuilt `table5_grid.csv` carries G-ORD's labels, not `pending: gate_gord`).

### 7.3 to 7.6. Not applied.

Not fails of the certification (condition (3) holds; 7.3 is CHECK_2 M4, the author's; 7.4 records a
`grid_source` and refuses a defaulted `splits` CLI run; 7.5 is the half-exposed `wapf_norm`; 7.6 the
template CLIs' overwrite). All more than one line; listed for the author (section 4).

The runbook's resume paragraph (section 1) now states both 7.1 and 7.2 in one sentence each.

---

## 3. Verification

### (1) The test suite, verbatim

Command, from `plan11_encoding_ladder/`: `python3 -m pytest -q -rs tests`

```
.............................s.......................................... [ 45%]
........................................................................ [ 90%]
...............                                                          [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_extract.py:515: zstandard module not installed
158 passed, 1 skipped in 312.72s (0:05:12)
```

159 collected: the 158 of CHECK_2's record plus the B1 test. Nothing was removed. The same suite
under `python3 -m unittest discover -s tests -p "test_*.py"`: `Ran 96 tests in 65.553s`,
`OK (skipped=1)` (the M6 statement in the runbook).

### (2) The driver end to end on a synthetic corpus, with the checker's run-2 configuration

Corpus: builder 1's presets (`synth.corpus_specs(n_pairs=80, reps=2, idle=2)`) filtered to gemm,
floyd, gibbs, histogram (plus the lexer preset, whose content is the idle model) and the two idle
cells: 12 cells, 80 pairs each, compressed with the `zstd` binary, under the session scratchpad
(`.../scratchpad/fix2/root`). Moves 0 to 2 through the driver; then by hand
`gates_precondition preconditions --c1-activity-min 0.001` (10 of 12 cells `all_hard_pass`; the two
lexer cells fail C1 at `apf_max` below 0.001) and `inputs/idle_admissibility.json` filled; then
`run --moves 3-13 --null-perm 5 --n-jobs 4 --assume-failed-zero --assume-reason "fixer cycle 2 run"`.
Exit 0 after 1 h 00 min 24 s wall time; all 70 ledger entries of moves 3 to 13 `done`; `gf_check`
`pass`. Where the time went: G-ORD 674 s (apf), 638 s (persist), 630 s (content), 627 s (wapf),
624 s (combined), single process as M7 says; the split stages 107 s (apf, raw and norm) and 84 s
(combined) at 5 permutations; `gdim` 18 s.

B1 in situ. B1-G3 quarantined a feature in the normalized LOKO/archetype split of every rung, so
this run exercises exactly the case of the finding. Per rung, the full model against the re-run
(accuracy / null p95, feature count), and what the gate and the tables now print:

| rung | full model | re-run (quarantined) | `gl.csv` (i) score_norm / p95 | Table 7 LOKO accuracy | `gdim.csv` d | Table 7 feature count |
|---|---|---|---|---|---|---|
| apf (W8_H2) | 0.750 / 0.750, 8 | 0.750 / 0.750, 6 (`duty`, `median`) | 0.75 / 0.75 | 0.750 | 6 | 6 |
| wapf (W8_H4) | 0.500 / 0.500, 8 | 0.500 / 0.500, 3 (`max`, `mean`, `median`, `p95`, `std`) | 0.5 / 0.5 | 0.500 | 3 | 3 |
| persist (W16_H8) | 0.750 / 0.750, 8 | 0.750 / 0.750, 7 (`j_excess.duty`) | 0.75 / 0.75 | `void: idle reps separable under this rung` | 7 | 7 |
| content (W8_H4) | 0.500 / 0.500, 36 | 0.750 / 0.750, 28 (eight `r_l0_*`) | 0.75 / 0.75 | `void: ...` | 28 | 28 |
| combined (W8_H4) | 0.500 / 0.500, 60 | 0.500 / 0.500, 42 (18 features) | 0.5 / 0.5 | `void: ...` | 42 | 42 |

The content rung is the case CHECK_2 reproduced (the two numbers differ, 0.500 against 0.750):
`gl.csv` now carries the re-run's 0.75, the number Table 7 would print (its LOKO cell reads the
G-F (i) void override on this run because G-F part (i) at the selected point found the two idle reps
separable under persist, content and combined; `part1_consequence = "void"`, al-Farabi 6.6). Every
G-L (i) verdict is `not run: 5 permutations < 500` by design. Table 6's `all` row LOKO norm 0.750
equals `gl.csv`'s APF score_norm. Every `scores.json` carries `score_source = with_quarantine ...`
and `predictions_file = predictions_with_quarantine.csv`, and that file exists beside
`predictions.csv` in every LOKO/archetype directory; `table8.json` `params.predictions_file` names
it. `gm.csv` (LOKO): wapf vs apf 0.5 against 0.75, persist and content vs apf 0.75 against 0.75,
combined vs apf 0.5 against 0.75, every pair `difference with margin` (spread 0). `gx.csv` (M11):
`not applicable: one campaign label` in both the leak and the confound columns. `excluded_rows.csv`
absent (no `near_unfalsifiable` at 5 permutations). Table 5's refusal cells (M12): `G-F (i): ...`
only, no duplicated lead. G-DEC read `decay not resolved` for floyd (no pass table declared on
this run), so the M2 branch (`decay (floor unmeasured)`) did not arise; `fig_floyd_decay` is the
placeholder with that verdict, as before. Table 7's `combined (matched)` feature count prints 42
beside `matched 6` in the G-DIM cell: M3, refused, unchanged.

### (3) The resume probes (al-Farabi's A, C, D on this run)

Probe A, nothing changed, `run --moves 3-13 --dry-run`: 58 steps `skipped: outputs exist and inputs
unchanged`; 12 steps read stale on files that accumulate across moves. See section 5 below: this
is a pre-existing defect of the driver's staleness rule, outside this cycle's mandate, recorded and
not fixed.

Probe D, the preconditions re-run at the default C1 (4 of 12 cells admissible; the record changes),
then `--dry-run`: 51 steps `dry-run (stale: preconditions.json changed since move <n>)`: every
`gc`, `gk0`, `gf` (both), the move-5 figures, every `features`, `grid`, `g3`, `gord`, `select`,
`splits`, `gx`, `gl` (both), `gn`, `gj`, `gdec`, every tables and figures step, `gdim`, `gm`,
`variance`, `cluster`, the manifest; the 7 skipped are the five `grid complete` checks, the
`gf check` (internal steps with no inputs) and the skeleton. Then the preconditions restored at
0.001 with the same `--assume-reason`: `preconditions.json` and `preconditions.csv` byte-identical
to the copies taken before the probe, and the dry-run shows the same 58 skipped as probe A. So the
JSON is a stable trigger: it moves exactly when the admissible set is recomputed.

Probe C, one `notes` field of `inputs/pass_table.csv` edited, `run --moves 6` for real: `features
apf` skipped, `grid apf` done (stale: pass_table.csv), `g3 apf` skipped, `gord apf` done (stale:
pass_table.csv), `select apf` done (stale), `alias` done (stale); exit 0 in 11 min. The rebuilt
`gates/table5_grid.csv` carries G-ORD's labels in every APF row (`order-blind` at W8_H2, W8_H4,
W8_H8; `resolution` at W16_H4, W16_H8, ...; `order-blind (by construction)` at the whole-cell
point), no `pending: gate_gord`, and `gates/grid/apf/W8_H4/gord.json` reads `order-blind`
(ordered 0.75 against shuffled mean 0.744, spread 0.25). The pass table was restored afterwards.

### (4) The skeleton and the runbook

`apf_paper/p2_skeleton.tex` regenerated; the diff is in M13 above (the timestamp comment and the
two cells). `run_moves plan --moves 10` prints the four `gates_temporal` lines the M5 replacement
refers to. No TeX compiler on this machine, so compilation is unverified, as in both earlier cycles.

---

## 4. For the author

Each item is a choice the definitions leave open, or a finding this cycle refused as more than one
line. I decide none of them.

1. **Which score is the rung's score after a B1-G3 quarantine** (CHECK_2 B1 and "For the author"
   item 2; al-Farabi 6.5). Applied as SPEC 3.7.2 states it: the re-run, for the comparison gates,
   the tables, Table 8's assignments and the exclusion list alike, with the full model kept in
   `scores.json` under its own keys and the choice named in every `params` (`score_source`). If you
   prefer the other reading (the gates and the tables both use the full model with the quarantine
   reported beside it), the change is confined to `models.effective_scores` (one function; both
   builders read through it) and to SPEC 3.7.2's sentence.
2. **`--c1-activity-min` on the driver** (M1, refused in both cycles as more than one line): the
   threshold decides which kernels exist for the paper (CHECK_2 "For the author" item 1; al-Farabi
   6.3). If you lift the one-line rule for it, the fix is a driver argument plus its pass-through
   to `preconditions` and a runbook sentence; the staleness side is already done (7.1).
3. **Table 7's `combined (matched)` feature count** (M3, refused): the row prints the
   pre-reduction width (42 on this run) beside `matched 6` in the G-DIM cell; the fix is
   `feature_count_used = min(d_target over folds)` in `scores.json` and one line in `table7`.
4. **The idle cells' head-drop key** (M10, refused): G-F (ii) reads the `idle` row, every other
   site the idle cell's label-derived kernel name; harmless at the default 0. The clean fix is
   role-based at six call sites.
5. **Table 6's rows for kernels excluded at C1** (M16, refused): they print `--`; the fix reads
   `preconditions.json` `excluded_cells` and prints `not run: excluded by C1-C8`.
6. **The G-L (ii) re-run** (M8; al-Farabi 6.7): the runbook now states it is the author's manual
   step; add the conditional driver command if you want it automatic.
7. **G-ORD's cost** (M7): 10 to 11 minutes per rung on 10 cells at 80 pairs here, single process;
   the parallelization of its two loops over `--n-jobs` is a builder-2 change of a few lines.
8. **The roll-up with no applicable kernel for G1** (M4; al-Farabi 6.4 and 7.3): drop G1 from the
   count as G2 is, or refuse the selection outright; choose one.
9. **Al-Farabi 7.4 to 7.6**: `grid_source` in the split stage's `params` and a refusal of a
   defaulted `splits` CLI run; `wapf_norm` exposed on `grid` or removed from `series features`;
   the template CLIs refusing to overwrite an existing author input without `--force`.
10. **The one-label confound record** (M11): with one campaign label `gx.csv` now writes
    `not applicable: one campaign label` in the confound column too; on the real corpus (three
    labels) the rule of CR 2.2 item 26 runs unchanged.
11. **The every-feature-quarantined case**: when B1-G3 quarantines every feature there is no
    re-run; `effective_scores` then blanks the score keys and carries `not run: every feature
    quarantined` in `b1_g1`, so G-L writes that string and Table 7 prints it in the null and rank
    cells with `--` in the score cells. The definitions do not say what the rung's score is in that
    case; this reading keeps the full model out of every consumer, which is what B1-G3 is for.
12. **The remaining exposed defaults** stand as CHECK_2 "For the author" item 8 lists them.

---

## 5. Observed on this run and not fixed (outside the mandate; for the next check)

**The driver's staleness rule against files that accumulate across moves.** Probe A above (nothing
changed) marks 12 steps stale: `splits apf`, `gx apf`, `gl`, `tables table6` and the `splits` and
`gx` steps of persist, content and wapf on `selection.json`, and the two `alias` steps on
`g3_flags.csv`. Mechanism: `select <rung>` adds the rung's entry to `gates/selection.json` at moves
6, 9, 10, 11 and 12, and `g3 <rung>` rewrites `gates/g3_flags.csv` at each of those moves, so a step
that ran at move 7 recorded the one-rung file's hash and finds a five-rung file on resume
(`splits apf` recorded `96f34dbd...` at 19:03:57; the file is `d417c358...` after move 12). The
same holds for `gc.csv`, `gx.csv` and `gf.csv`, which grow by rung. On the real corpus a resume of
moves 7 to 13 with nothing changed would re-run four split stages at 500 permutations (hours) for
no reason. Not a change I made and not in either report (al-Farabi's probe A covered move 6 only,
before any later rung's selection existed). A fix is builder 3's: hash the rung's own entry of a
per-rung file (`json:gates/selection.json:<rung>`, `csv:gates/gc.csv:rung=<rung>`, the same
syntax the outputs already use) instead of the whole file, or record the hash after the move's
own writes for the two `alias` steps.

**`gates/gf.csv` keeps two rows per rung after move 12** (the `W8_H4` row of move 4 and the
selected-point row), as CHECK_2 records; `tables` picks the selected point's row. Unchanged.

---

Files edited (all under `plan11_encoding_ladder/` unless stated): `models.py`,
`gates_comparison.py`, `_report_common.py`, `tables.py`, `run_moves.py`, `figures.py`,
`gates_calibration.py`, `gates_readings.py`, `latex_skeleton.py`, `RUNBOOK.md`,
`requirements.txt`, `tests/test_gates_comparison.py`, `tests/test_gates_models.py`,
`tests/test_driver.py`; and the regenerated `apf_paper/p2_skeleton.tex`. This file:
`/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/FIX_2.md`.
Scratch artifacts of the runs are under
`/private/tmp/claude-501/-Users-jeries-Desktop-projects-thesis-memorySignal-mem-sig/14810c8c-2535-466d-a296-d2aae9739c16/scratchpad/fix2/`.
