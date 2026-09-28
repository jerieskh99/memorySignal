# Build report, epoch 2, builder B: the detection layer's report modules, driver, runbook, skeleton and bibliography

Written 2026-09-17 by builder B (report). Task: `SPEC_DETECTION.md` section 1.2 (tables, figures,
the driver extension of section 1(g), the runbook, the LNCS skeleton, `p3.bib`), with the
corrections of the three reviews (`SPEC_DETECTION_review_al_kindi.md`, `_ml.md`, `_al_farabi.md`)
implemented where they say "must change before build". No server was touched; every test uses
synthetic data generated on this machine; the sandbox family is named by index and letter only in
every file written; no paper prose was written; no git commit was made.

## 1. Files written

Under `/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/`:

| File | What it is |
|---|---|
| `tables_detection.py` | Tables 4, 5, 6, 7, 8, 9, 10 (misses and false positives, per rung and for `--table10-rung`), 11, the level-2 and level-3 tables (per rung and for the chosen rung), the ladder table, the G-V two-class table, the per-cell appendix table, the manifest; each as CSV, Markdown and LaTeX through `_report_common.write_table`, each with a `<name>.json` params block. |
| `figures_detection.py` | `fig2_three_floors`, `fig4_fused_plane_tiers`, `fig5_roc_lowo`, `fig6_ladder`, `fig_level_map`, `fig_apf_per_tier`, `fig_level2_confusion` (PNG and PDF; `SKIPPED.txt` when matplotlib is absent; a placeholder figure carrying the `not run:` string when an input is missing). |
| `latex_skeleton_p3.py` | Writes `report/detection/paper3_skeleton.tex` and, with `--standalone PATH`, the same file to `apf_paper/p3_skeleton.tex`. |
| `run_detection.py` | The driver for D0 to D15 in al-Kindi's order, built on `run_moves.run_plan(..., max_move=15, ledger_name="driver_detection_state.json", internal_steps=...)`; two internal steps (`features-at-selection`, `tripwire-check`); the class-file placement at D0. |
| `RUNBOOK_DETECTION.md` | The command-line runbook: prerequisites, the smoke run, the class file, the letter sequence, D0 to D15 with flags, what each writes and what to look at, the cost table, stage 2 and 3 entry, the resume discipline, the choices left to the author. |
| `run_moves.py` (additive extension only, SPEC_DETECTION 1.3) | `parse_moves(spec, max_move)`, `load_ledger(out, name)`, `save_ledger(out, ledger, name)`, `run_plan(o, plan, *, max_move, ledger_name, internal_steps)`, `print_status(out, name)`; every default keeps the epoch-1 behaviour. |
| `tests/detection_fixtures.py` | A 24-cell synthetic `<out>` tree in the record shapes of SPEC_DETECTION section 3 with the review corrections (two kernels, four idle cells, three sandbox members named `sandbox_member_<m>`, sub-families A and B), with one knob per "refusal printed as a string" rule. |
| `tests/test_report_detection.py` | 23 tests: every table and figure written and checked (columns exact, no blank cell, no number in a verdict cell), every refusal rule, the skeleton's structure. |
| `tests/test_run_detection.py` | 8 tests: the move table with the review corrections, the CLI guards, a dry run, the class-file refusal, a real run of the driver's own steps with resume, staleness, `--force` and `status`, the internal steps. |
| `BUILD_detection_report.md` | This file. |

Under `/Users/jeries/Desktop/projects/thesis/memorySignal/apf_paper/`:

| File | What it is |
|---|---|
| `p3_skeleton.tex` | The standalone LNCS skeleton (`\documentclass[runningheads]{llncs}`; headings of P3 Sec. 2 / N3 Sec. 2; Tables 1 to 3 as static shells; 28 `\IfFileExists` inputs and figures; comment blocks with the substance bullets and the box contents; `\bibliographystyle{splncs04}`, `\bibliography{p3}`). Zero prose lines by the SPEC_DETECTION 4.5 test. |
| `p3.bib` | Twenty entries copied byte for byte from `p2.bib` with Hunayn's attached comment blocks (the nineteen keys of SPEC_DETECTION 5.4 plus `vanderkouwe2019sok`, named by the task), then the `% TODO (Hunayn)` block with the `% NEEDED` lines. No entry written from memory. |

Nothing else was written or edited. `_report_common.py`, `figures.py`, `latex_skeleton.py`,
`series.py` and the rest of the epoch-1 package are imported, never changed.

## 2. How to run

From `VM_sampler/VM_Capture_QEMU/`:

```
python3 -m plan11_encoding_ladder.run_detection plan --out <out> --root <root> --grid-default W8_H4 --moves 0-15
python3 -m plan11_encoding_ladder.run_detection run  --out <out> --root <root> --classes <your classes.csv> --selection-from <sel> \
    --assume-failed-zero --assume-reason "AA A5: any failed job re-runs the whole cell" --n-jobs 8
python3 -m plan11_encoding_ladder.run_detection status --out <out>
python3 -m plan11_encoding_ladder.tables_detection  --out <out> [--only NAME,...] [--table10-rung combined]
python3 -m plan11_encoding_ladder.figures_detection --out <out> [--only NAME,...] [--identity excess|raw] [--plane-per-letter]
python3 -m plan11_encoding_ladder.latex_skeleton_p3 --out <out> [--documentclass llncs|article] [--standalone apf_paper/p3_skeleton.tex]
```

The smoke run and every move by hand are in `RUNBOOK_DETECTION.md` sections 1 and 4. The
tests: `cd plan11_encoding_ladder && python3 -m pytest -q tests/test_report_detection.py
tests/test_run_detection.py`.

## 3. Test results

RESULTS_PLACEHOLDER

## 4. The pdflatex result

`pdflatex` is not installed on this machine (`which pdflatex` finds nothing; no `/Library/TeX`,
no `/usr/local/texlive`, no `llncs.cls` anywhere on disk), so the skeleton was not compiled. In
its place the structural checks of SPEC_DETECTION 4.5 were run on `apf_paper/p3_skeleton.tex` and
pass: brace balance 0 outside comments; every `\begin` matched by its `\end` (27 `table`, 27
`tabular`, 4 `figure`, `document`, `abstract`); 28 `\input` and `\includegraphics` targets, every
one wrapped in `\IfFileExists` so the file compiles standing alone (the shells) and beside a full
run (the generated tables and figures; `tests/test_report_detection.py::TestSkeleton` checks every
target exists after `tables_detection` and `figures_detection` ran); zero prose lines (no line
outside a comment ends in a period or carries a word of more than three letters outside a LaTeX
command); every `\cite` key of the skeleton resolves to an entry of `p3.bib`. Every table and
figure environment is single column (`table`, `figure`, never the starred forms), because llncs
is single column and I could not verify on this machine that llncs.cls defines `table*`. The
author compiles once with `pdflatex p3_skeleton.tex` where llncs.cls and splncs04.bst are
installed; with `--documentclass article` the same file compiles under the standard class
(`\institute{}` is emitted only under llncs).

## 5. Deviations from SPEC_DETECTION.md, each from a review marked "must change before build"

1. al-Kindi 1: the driver passes `--threshold-source inner_lowo` to every two-class split, to the
   matched split and to the ladder; `oob` and `inner_group_kfold` are the CLI alternatives. The
   runbook's cost table counts 21 x (1 + B) fits per LOWO run.
2. al-Kindi 2 and ML 2.7: the one-class command carries `--null-perm` and `--n-jobs`; the
   driver's `--null-splits` default is `lowo,loco,one_class`; Table 11's note reads the one-class
   null verdict from the one-class `scores.json["null"]` and prints the F9 sentence when it is
   inside while LOWO's passes.
3. al-Kindi 3: the level-2 table and heat map carry `null p95, rank, verdict` per sub-family row;
   `macro_recall` is never printed.
4. al-Kindi 4: the level tables carry `n at floor`; the level-3 accuracy row prints the flat null
   block builder A writes (`null_unit = cell`).
5. al-Kindi 5: Table 4 carries `n C1 fail (reported)`; the driver passes `--c1-rule report`.
6. al-Kindi 6 and al-Farabi M5: the head-drop template at D2 is `gates_detection
   head-drop-template` (one row per workload key of every class); the runbook states the rule
   that a member's head drop is the author's number, fixed at D2, never from the APF(t) figure.
7. al-Kindi 7: Figure 4's identity coordinate is `J - J_null` (`--identity excess|raw`), the
   mask is `K > k_factor * floor_K` from `gates/gj.json` applied to every class, recorded as
   `plane_mask = "gj_mask_K, every class"`; unmasked with the title saying so when `gj.json` is
   absent. One addition of mine, listed in section 6: the idle and harness-idle clouds are drawn
   unmasked, because the K mask removes the floor's own cloud by construction.
8. al-Kindi 8: Table 10 prints `axis`, `axis_of_largest`, `d_amount`, `d_identity` (the columns
   are read from `miss_table.csv`'s header, so builder A's order is kept).
9. al-Kindi 10 and 11: Table 6 and Table 9 print one drift row per (class, unit) and one order
   row per (rung, class, half rule); the readers accept builder A's `unit` and `label` columns.
10. al-Kindi 12: one fused-plane panel for the class with all members as shapes coloured by
    letter; per-letter panels only under `--plane-per-letter`.
11. ML 2.1: Table 7's note prints `row_unit`, `threshold_source`, `train_on_at_floor`,
    `null_denominator_rule`, `score_aggregation` and `det_c1_rule` from the split's params; the
    driver passes `--row-unit` only when given (the module constant otherwise).
12. ML 2.3: the driver runs `gates_detection gop` for `lowo`, `loco` and `lofo`; Table 5 rolls
    G-OP up over all three; Table 7's G-OP cell carries the setter families beside
    `GOP_SET_BY_FEW`.
13. ML 2.4: Table 7 prints the full-feature forest as the headline (`score_source = full`), the
    quarantined feature in the `B1-G3 quarantine` column, the re-run's TPR in `TPR at 5% without
    quarantined`, and one `<rung>: l1 (best single feature)` row per rung from
    `scores.json["l1"]` (the feature from `feature`, `features_chosen` or `best_feature_by_fold`).
14. ML 2.5 and al-Farabi M2: `--train-on-at-floor true|false` is passed through by the driver when
    given; the value used is printed in Table 7's note from every split's params.
15. ML 2.6: the driver runs `gates_detection leak-probe` at D7; Table 9 carries the `cadence`
    (`n_pairs`, `dt_est_s`), `active fraction` (`frac_above_band`) and `level (beside G-L)`
    (`K_med`) rows; Table 5 rolls the probe up.
16. ML 2.8: Table 8 prints, after each LOCO row, the `loco: L0 ratio (reps identical?)` row from
    `gv_two_class_members.csv` with the note `reps near identical: LOCO reads as within-trace`
    for a member below `REPS_IDENTICAL_RATIO = 0.1`.
17. al-Farabi M4: `n without score` beside `n at floor` in Tables 7 and 8; the per-cell table
    prints `score_status` in the score cell.
18. al-Farabi M9: Table 4 carries an `unassigned` row; the per-cell table prints the class
    `unassigned`; the runbook's D0 says a non-zero `unassigned_cells` is the author's to resolve.
19. al-Farabi M10: a differing `--classes` against an existing `inputs/classes.csv` is refused by
    the driver and recorded as the D0 command's status in the ledger (exit 2); the runbook
    documents exit 3 for the one-class usage error.
20. al-Farabi M11: the per-cell table never prints `path`, `label`, `test_label`, `traj_file` or
    `order_index`; the runbook says what `apply` refuses when `extract/` holds pre-apply names.
21. al-Farabi M12: `--order-scope within_class|campaign` is passed through when given.
22. The tripwire check (SPEC_DETECTION 6.1, D15) needs a G-F (i) verdict and the two drift-clause
    verdicts on every Table 7 and Table 11 row, so both tables carry the columns `G-F (i)`,
    `G-ANCHOR (ii)` and `early-late idle` (SPEC_DETECTION 5.2.3 and 5.2.7 list G-F (i) for Table
    7 only). `ORDER_VOID` under consequence `void`, `GF_VOID` and `refused: disconnected lead`
    print in every score cell of the affected rung in Tables 7, 8, 11, the level tables and the
    ladder table, as SPEC_DETECTION section 5 says.

Other deviations, not from a review:

23. `run_moves.print_plan` was not extended (it is not in SPEC_DETECTION 1.3's list), so
    `run_detection.py` carries its own `print_plan` bounded by 15. Another epoch-2 builder raised
    `run_moves.MAX_MOVE` to 14 and made it the default of `parse_moves` while this build was in
    progress; the additive extension of section 1.3 is intact (`ledger_name`, `internal_steps`,
    `max_move` on `parse_moves`, `run_plan`, `load_ledger`, `save_ledger`, `print_status`) and the
    epoch-1 driver tests pass.
24. The stale message names the changed part (`stale: classes.csv changed since move 14 (...)`)
    through the same builder's keyed-hash refinement of `_stale_reason`; my tests assert the
    prefix only.
25. Builder A's `scores.json` writes no `external` block: the summary pools test-only cells into
    the positive denominators by member index. Table 7's external block and Table 8's `external m`
    columns therefore fall back to the rows of `predictions.csv` whose `class` is `external`
    (counts of `flag_05` over those rows, labelled `predictions.csv (scores.json has no external
    block)` in the row); when builder A writes the block, it is read as is.
26. Table 8 and Table 11 print `not applicable: the one-class run is on the normalized features
    (SPEC_DETECTION 3.3.6)` in the `apf raw` row's one-class cells instead of a missing-file
    string.
27. The ladder table's `pairs at 0.644 s` and `pairs at 0.500 s` are the declared fixed-spacing
    counts `round(prefix_s / dt)` of K3 move 17 (uncapped: 47, 93, 186, 466, 932 and 60, 120,
    240, 600, 1200); the capped count per cell is `pairs (median over cells)` from `ladder.csv`.
28. `p3.bib` carries `vanderkouwe2019sok` (named by the task, not by SPEC_DETECTION 5.4). The
    task's TODO list names Clark 2005 and Lindemann and Fischer 2018 as still needing verified
    entries; both are in `p2.bib` (tiers READ and METADATA) and are copied, and the TODO block
    says so. The NEEDED lines cover Ninan 2024, Qu 2025, Dunn and Ghosh 2024, the encoding paper's
    own entry, and the in-guest-agent row of Table 2.

## 6. Interface notes against builder A's modules

Builder A's modules (`classes.py`, `detection_splits.py`, `detection_metrics.py`,
`detection_levels.py`, `gates_detection.py`, `synth_detection.py`) landed while this build was in
progress; their tests were not yet under `tests/` when the suite below ran. My readers were
written against SPEC_DETECTION's shapes and then checked against builder A's column tuples and
`scores.json` keys; where they differ the readers accept both:

- `drift.csv`: builder A writes `unit` and `label` (I had `drift_unit`, `statistic`); both read.
- `gdim.csv`: builder A writes `row` (the display name) beside `rung`; the matched row is found
  by `row`.
- `scores.json["null"]`: a dict with `status` (`ok`, `not run: null not requested`, `not
  applicable: no workload to permute`) beside the per-statistic blocks; the null verdict cell
  prints the status when it is not `ok`.
- level 3's `scores.json["null"]` is a flat block (`statistic = accuracy`, `p95`, `rank_text`,
  `verdict`, `null_unit`); level 2's carries `per_letter`; the level-2 table reads
  `confusion.csv`'s per-row columns as SPEC_DETECTION says.
- `ganchor.csv`: the harness-idle rows carry `part = "idle_sets (harness_idle)"`, so the
  `idle_sets` and `idle_early_late` lookups are not confused by them.
- The driver's command lines were checked against builder A's argparse definitions: every flag
  the driver passes exists (`splits`: `--threshold-source`, `--loco-mode`, `--row-unit`,
  `--train-on-at-floor`, `--null-splits`, `--n-jobs`, `--seed-offset`, `--reduce-to-strongest`;
  `one-class`: `--model`, `--null-perm`, `--n-jobs`, `--seed-offset`; `ladder`: `--null-perm`,
  `--threshold-source`, `--n-jobs`, `--seed-offset`; `level2`/`level3`: `--null-perm`,
  `--train-on-at-floor`, `--n-jobs`, `--seed-offset`; `gates_detection`: `admissibility
  --c1-rule`, `head-drop-template`, `gk0-sandbox-template`, `gop --split --cells-rule`, `glm
  --level-quantity --n-jobs --seed-offset`, `anchor --part --null-perm --n-jobs --seed-offset`,
  `order --consequence --scope --null-perm --n-jobs --seed-offset`, `drift --null-perm
  --seed-offset`, `leak-probe --null-perm --seed-offset`, `gfp --n-jobs --seed-offset`, `gm
  --n-jobs --seed-offset`; `classes`: `validate --classes --relaunched-grouping`, `apply
  --kernel-family-rule --relaunched-grouping --campaign-label`, `inherit-selection --from
  --default --force`, `letter-sequence`). `--row-unit` is passed to `splits` only (the one-class
  and level CLIs do not take it); `--train-on-at-floor` to `splits` and the levels.
- End-to-end against builder A's real modules on a tiny synthetic corpus (`synth_detection corpus
  --kernels gemm,floyd --reps 2 --idle 2 --members 1,2,5 --n-pairs 60 --write-classes`): D0
  (classes copy, extract index, classes validate, apply, inherit-selection, letter-sequence), D1,
  D2 (preconditions, the five templates, gp, admissibility), D3 (five gc), D4 (gk0, gk0-cells,
  gn, the five feature builds at the inherited grid, gf, anchor idle_sets) and D5, D6 ran and were
  recorded `done`; the letter sequence read `Br0 Br0 Br1 Br1 S1r0 S2r0 S5r0 S1r1 S2r1 S5r1 Ir0
  Ir1`; `anchor idle_early_late` failed inside builder A's `detection_metrics.run_detection_split`
  (`KeyError: 'records'` at line 915, the binary-label path) and the driver stopped at it as it
  should, recording the exit code and the stderr tail in the ledger. E2E_PLACEHOLDER

## 7. For the author

1. The three review-changed defaults the driver runs with: `--threshold-source inner_lowo`,
   `--null-splits lowo,loco,one_class`, `--c1-rule report` for every class. Under `inner_lowo`
   the LOCO null at `--loco-mode cell` is 168 x 500 x 14 fits per rung and is not affordable; the
   runbook's cost table gives the counts and `--loco-mode rep_index` (8 x 500 x 14) as the cheap
   reading. Without the LOCO null G-SIG cannot refuse (F2). Decide `--null-splits`, `--null-rungs`
   and `--loco-mode` before the data. On this machine, under a load average of about 500 from the
   other builders' suites, one 300-tree fit on 160 rows by 60 features took 3.6 s and one on
   39,000 window rows took 1010 s; the ratio is the point, the absolute numbers are not.
2. `--train-on-at-floor`: the ML review recommends `false`, al-Farabi names `true` as the SPEC's
   present behaviour; the driver passes the flag only when given and Table 7's note prints the
   value builder A used.
3. `--row-unit cell` is builder A's constant (ML review 2.1); `window` is epoch 1's habit under
   which `oob` is not admissible.
4. The fused plane draws the idle and harness-idle clouds unmasked (`IDLE_CLOUD_RULE` in
   `figures_detection.py`, recorded in `figures.json`): the K mask of al-Kindi review 7, applied
   to the idle cells, removes them by construction, and move 11 asks whether the members' region
   is empty of benign cells once idle is drawn. Masking them instead is a one-line change.
5. `REPS_IDENTICAL_RATIO = 0.1` (Table 8's L0 ratio row) and the leak probe's vocabulary are the
   ML review's proposed defaults; confirm or change them.
6. Table 7's `n assignments` and the null verdict cells print `null_not_estimable` with no number
   below 20 assignments; the smoke run prints `not run: 20 permutations < 500` with the numbers
   beside it and is never admissible.
7. The external block (stage 3): builder A's summary pools test-only cells by member index into
   the same `per_member` as the sandbox members (an external member 1 collides with sandbox
   member 1); the tables fall back to `predictions.csv` for the external block and say so. Ask
   builder A for an `external` block in `scores.json` before stage 3, or accept the fallback.
8. Table 2 (prior work by observer position) holds the bib keys of the six named rows and `--`
   elsewhere; the in-guest-agent row has no verified entry yet (`% NEEDED` in `p3.bib`). Ninan
   2024, Qu 2025 and Dunn and Ghosh 2024 are `% NEEDED` comments, never entries, until Hunayn
   verifies them; the encoding paper's own entry follows its submission.
9. The skeleton's three title lines are placeholders (`open, P3 Sec. 9; the author's`); the
   substance bullets are abbreviations of P3 Sec. 2's "Carries" column and N3 Sec. 1's boxes,
   every one a comment to delete or reword; no line outside a comment is a sentence.
10. `--table10-rung combined` chooses the rung of the unsuffixed Table 10 and of the level tables
    the skeleton inputs; `--identity excess|raw`, `--plane-per-letter`, `--documentclass` and
    `--standalone-tex` reach the figures and the skeleton through the driver.
11. The manifest hashes every file under `report/detection/` and `gates/detection/` and carries
    every params block; it does not hash `inputs/classes.csv`, `cells.csv`, the sidecars or
    `extract/` (the author's names live there; al-Farabi review, for the author 7).
12. The driver stops at the first failing command of builder A (as at `anchor idle_early_late` on
    the tiny corpus above) and records the exit code and stderr tail in
    `driver_detection_state.json`; a re-run resumes at the failed command once the module is
    fixed; nothing before it re-runs unless an input changed.
