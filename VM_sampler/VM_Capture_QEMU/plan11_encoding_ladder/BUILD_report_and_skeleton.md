# Builder 3 report: tables, figures, driver, runbook and the LaTeX skeleton (2026-09-16)

This report covers the report layer of the paper 2 analysis toolkit (`plan11_encoding_ladder`):
the table generators, the figure generators, the driver that runs al-Kindi's moves in order,
the author's runbook, and the LaTeX skeleton. Nothing was run on any server; every test uses
synthetic data generated on this machine. The sandbox family is not named anywhere in this
layer. No paper prose was written: the skeleton is headings, table shells, figure placeholders
and comment blocks.

## 1. Files written

All under `mem_sig/VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/` unless stated.

| File | What it is |
|---|---|
| `tables.py` | Tables 5, 5-G3 (companion), 6, 7, 8, G-V, the Table 4 status column, the preconditions copy, the wAPF-over-APF table, and `report/manifest.json`. Each table is written as `.csv`, `.md` (Markdown) and `.tex` (booktabs `tabular` inside a `table`/`table*` environment with empty `\caption{}`, a `\label{}`, and a `% columns:` comment line), plus a `.json` with `schema`, `params` and `citation`. Verdict cells print the verdict string; an undefined number prints `--`; a missing input prints `not run: <file> missing`. |
| `figures.py` | The eight figures of P2 Sec. VII (SPEC 6.7), one function each, PNG and PDF: `apf_per_kernel`, `level_matched`, `fused_plane`, `j_hist`, `ratio_hist`, `floyd_decay`, `piano_roll`, `table5_grid`. Without matplotlib it writes `report/figures/SKIPPED.txt` naming the module and exits 0. |
| `latex_skeleton.py` | Writes `<out>/report/paper2_skeleton.tex` and, with `--standalone PATH`, a copy elsewhere. IEEEtran conference class by default, `--documentclass article` as SPEC 6.8's fallback. |
| `run_moves.py` | The driver (the build brief's name). `run`, `status` and `plan` subcommands; moves 0 to 13 with a `--moves` selector; the ledger `driver_state.json`; the skip, staleness and template rules; the grid-completeness and G-F checks. |
| `driver.py` | SPEC section 1's name for the driver: a one-line alias of `run_moves.main`. |
| `_report_common.py` | Shared helpers for the four modules above: the `schema`/`verdicts` import with SPEC-declared fallbacks (the other builders' files were not on disk when this layer was started), CSV/JSON/LaTeX/Markdown writers, sha256, the readers for builder 2's result files. Not in SPEC's file table; see deviation 1. |
| `RUNBOOK.md` | The author's runbook: prerequisites and the `pip --user` fallback, the smoke run, the driver, the thirteen moves with their exact commands, what each writes, what to look at, how to add the idle cells, where everything is. |
| `requirements.txt` | `numpy>=1.21`, `scikit-learn>=1.0`, `matplotlib>=3.4`; optional `scipy`, `zstandard`, `pytest`. |
| `tests/report_fixtures.py` | The synthetic `<out>` tree in SPEC's declared record shapes (cells, extracts, sidecars, every gate file, the split directories, one trajectory for the piano roll), with knobs that make one cell refuse at a time. |
| `tests/test_report.py` | 24 tests for tables, figures and the skeleton. |
| `tests/test_driver.py` | 8 tests for the driver. |
| `apf_paper/p2_skeleton.tex` | The standalone IEEEtran skeleton (the build brief's item (e)), 432 lines, 135 of them comments; identical to the generator's output with no `inputs/pass_table.csv` present. |
| `BUILD_report_and_skeleton.md` | This report. |

## 2. How to run

From `VM_sampler/VM_Capture_QEMU/`:

```
python3 -m plan11_encoding_ladder.tables         --out <out> [--only table5,table7,...] [--table8-rung combined]
python3 -m plan11_encoding_ladder.figures        --out <out> [--only fused_plane,...] [--piano-cell ID] [--piano-stride 16]
                                                 [--fused-plane-mask K|persist] [--decay-jump-ratio 1.5]
python3 -m plan11_encoding_ladder.latex_skeleton --out <out> [--documentclass IEEEtran|article] [--standalone PATH]
python3 -m plan11_encoding_ladder.run_moves run  --out <out> --root <root> [--moves 0-13] [--assume-failed-zero --assume-reason TEXT]
                                                 [--n-jobs N] [--null-perm 500] [--null-splits loko,loro,within_trace] [--seed-offset 0]
                                                 [--force] [--dry-run] [--skip-missing-modules] [--only-modules M,...]
python3 -m plan11_encoding_ladder.run_moves status --out <out>
python3 -m plan11_encoding_ladder.run_moves plan   --out <out> --root <root> [--moves ...]
```

The tests:

```
cd VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder
python3 -m pytest -q tests/test_report.py tests/test_driver.py     # or: python3 -m unittest tests.test_driver ; python3 -m unittest tests/test_report.py
```

The full sequence, the wall-time expectations and the smoke run are in `RUNBOOK.md`.

## 3. Test results

`python3 -m pytest -q tests/test_report.py tests/test_driver.py`: 32 passed in 96 s (the
figure tests render every figure several times). Under `python3 -m unittest`: 24 + 8 tests,
OK. Every table has a passing case and a refusing case:

- Table 5: 65 rows, one `selected` per rung, `selected: best-feasible` for a best-feasible
  selection, G-F (i) carried on the selected row, `G2 (pairs)` present; a disconnected rung
  carries `G-C: disconnected lead` on all 13 of its rows; a missing `table5_grid.csv` prints
  `not run: gates/table5_grid.csv missing` in every gate cell.
- Table 6: 12 kernel rows, the archetype rows after G-K0's relabelling (the lexer under IDLE),
  `all`; the level-set markers; `near_unfalsifiable` printed in every score cell of that split
  and in no other; a smoke run's `not run: 20 permutations < 500` in the null and rank cells
  with the scores still printed; `not run: no selection for apf`; the total-confound leak
  marked on the G-X cell.
- Table 7: 30 rows (six rungs by three splits, plus the appended archetype-space rows for LORO
  and within-trace); `G-M vs APF` as `beats (diff 0.200 > spread 0.040; 8 up, 0 down)`; the
  G-C, G-F (i), G-L, G-DIM and G-X columns; a void rung prints its void string and a
  disconnected rung prints `refused: disconnected lead` in every score cell; a rung without a
  selection prints `not run: no selection for <rung>`; a best-feasible resolution prints
  `8 x 4 (selected: best-feasible)`.
- Table 8: assignments with counts and kernel names, the lexer in the `IDLE (measured)`
  column, the clusters column as `c0:n ...`; missing predictions print `not run:` naming the
  file.
- G-V: sorted by `L0/L3` within rung, one summary row per rung carrying `estimable` or
  `LOKO not estimable`.
- Table 4 status, the preconditions copy, wAPF over APF, the manifest (sha256 of every report
  and gate file, every `params` block, the ledger).
- LaTeX and Markdown forms: `\toprule`/`\midrule`/`\bottomrule`, underscores escaped, `--` for
  undefined numbers, no `nan` anywhere.
- Figures: all eight PDFs and PNGs; the piano roll re-streams the trajectory (row count
  recorded); the fused plane applies the G-J mask (hollow points counted) under both `mask_K`
  and `mask_persist`; the J histograms draw the floor quantiles from `gj.json`; the floyd decay
  draws passes when G-DEC reads `decay` and a placeholder with the verdict otherwise; a
  missing `--piano-cell` writes a placeholder; `PLAN11_NO_MPL=1` exercises the
  `SKIPPED.txt` path.
- Skeleton: every `\input` and `\includegraphics` target exists beside the report copy after
  tables and figures ran; no line outside comments is anything but a LaTeX command, an
  environment line or a tabular row; the IEEEtran and article variants; balanced
  `begin`/`end` for every environment; Table 3 merges the pass table's declared column.
- Driver: `parse_moves`; the move table carries every review correction (see section 4);
  the CLI guards (`--assume-failed-zero` without a reason, move 0 without `--root`, a move
  outside 0 to 13); a dry run records every command; a real run through the report layer with
  the other builders' commands filtered (`--only-modules`), then a second run skipped, then a
  stale re-run after `inputs/pass_table.csv` is edited, then `--force`; a missing module stops
  with exit 2 unless `--skip-missing-modules`; the grid-completeness check refuses the split
  stage with exit 1 when the split module will run and passes on a complete grid; the move-13
  G-F check refuses when a Table 7 row lacks its verdict.

The real chain on builder 1's synthetic corpus (`synth corpus --reps 3 --idle 3 --n-pairs 60`,
39 cells) was also run with the driver as a cross-builder interface check. Moves 0 to 5
completed through the driver (index, extract, preconditions, the four templates, G-P, G-C on
four rungs, G-K0, G-F on five rungs, the first two figures), and move 6's `features`, `grid`
and `g3` completed; `gord` (builder 2's forest fits, about 5,800 fits at the smoke settings)
was still running after 25 minutes when this report was closed, so moves 7 to 13 of the real
chain were not reached here. On the partial real tree the report layer was run by hand:
`tables` read builder 2's `gc.csv` (`rep = all` rows), `gf.csv` (`part = i`), `gk0.csv`,
`gp.csv`, `preconditions.csv` (with `failed_source`) and `temporal_per_kernel.csv` (with
`coverage_pairs` and `G2_pairs`) exactly as written, every selection-dependent cell read
`not run: no selection for <rung>` and every absent file was named; the wAPF-over-APF table
came out of the real extracts; `figures` drew the J and ratio histograms from the real
extracts and the piano roll re-streamed a real `.csv.zst` trajectory through the `zstd`
binary (262,387 rows, stride 16). The interface gaps that remain are in section 5.

## 4. Deviations, each with its reason

1. **An extra module, `_report_common.py`.** SPEC section 1 lists five files for builder 3;
   the four modules share the readers for builder 2's files and the `schema`/`verdicts`
   import-with-fallback, and duplicating them four times would have been worse. It computes no
   verdict. The fallback exists because the three builders worked in parallel and neither
   `schema.py` nor `verdicts.py` was on disk when this layer started; both are now, both are
   imported first, and the fallback values are SPEC 2.6 and 3.0 verbatim.
2. **The driver is `run_moves.py`, with `driver.py` as an alias.** The build brief names
   `run_moves.py`; SPEC names `driver.py`. Both work; `driver.py` is one import.
3. **The skeleton's document class is IEEEtran (conference).** SPEC 6.8 says `article`; the
   build brief says IEEEtran conference class, and the brief is the later instruction.
   `--documentclass article` keeps SPEC's form. Every generated table is brought in as
   `\IfFileExists{tables/<name>.tex}{\input{...}}{<shell>}` so that one skeleton compiles both
   beside the generated tables (`<out>/report/`) and standing alone (`apf_paper/p2_skeleton.tex`,
   where the shells with the column headers and one empty row show).
4. **Not compiled.** No TeX installation exists on this machine (`pdflatex`, `latexmk`,
   `tectonic` all absent; no `IEEEtran.cls`), so the skeleton was checked structurally instead:
   brace balance 0 outside comments, every `\begin` matched by its `\end` (11 tabular, 8
   table*, 3 table, 5 figure*, 3 figure, abstract, document), every tabular row with the right
   number of `&`, every `\input`/`\includegraphics` target present beside the report copy after
   a full tables-and-figures run, zero prose lines. The runbook gives the compile command
   (`cd <out>/report && pdflatex paper2_skeleton.tex`). One thing I could not verify without a
   compiler: IEEEtran's `\maketitle` with an empty `\title{}` and `\author{}`; if it complains,
   the author puts the chosen working title in.
5. **Markdown added to every table.** SPEC 6 says CSV and LaTeX; the brief asks for Markdown
   and LaTeX. All three are written.
6. **Table 5 has one extra column, `G2 (pairs)`**, and Table 7 has four columns beyond SPEC
   6.3's list (`label space`, `G-C`, `G-F (i)`, `G-X`), all from the review corrections listed
   below. Table 5's G3 companion has one extra column, `cells present` (the count of cells whose
   flag is present over the cell count, so the 7-of-8 rule is visible).
7. **The driver's extra flags** `--dry-run`, `--skip-missing-modules`, `--only-modules`,
   `--plan`, `--standalone-tex`, `--piano-cell`, `--piano-stride`, `--table8-rung`,
   `--persist-side`, `--failed-counts`. The first three exist so the driver can be tested and
   inspected when another builder's module is absent or not wanted; everything they do is
   written into the ledger as `dry-run` or `not run: module ... (absent | filtered)`, never
   silently.
8. **`figures.py` has two flags SPEC 7.1 does not list**: `--fused-plane-mask K|persist`
   (al-Kindi review 9) and `--decay-jump-ratio 1.5` (al-Kindi review 1's detection ratio; the
   floyd figure reconstructs the pass boundaries for drawing only, from the K jump, and records
   the ratio in `params`; it never produces a verdict).
9. **The G-F run at move 4 is `--all-rungs` at the default grid `W8_H4`**, not APF only. SPEC
   section 7 says "apf (and each rung, after move 6 for the grid id)"; with al-Farabi 2.6
   moving the selected-point re-run into move 12, running every rung at the declared default
   point at move 4 gives the tripwire its early reading on every rung, and both rows are kept
   (SPEC section 8 item 38).

### The review corrections implemented (each marked "must change before build")

- al-Kindi 5: `gates_calibration alias` runs again at the end of move 7, after Table 6's
  features exist (the driver; the runbook).
- al-Kindi 6: `gates_calibration gp` runs at move 2, right after the pass-table template; move
  6 keeps `alias`.
- al-Kindi 7: Table 5 carries `G2 (pairs)` (from `G2_pairs` and `coverage_pairs`); when builder
  2's file lacks the column the cell says so.
- al-Kindi 8: `gates_comparison gx --rung <rung>` runs for every rung (moves 7, 9 to 12); Table
  6's G-X cell is APF's; Table 7 carries each rung's own leak verdict.
- al-Kindi 9: the fused plane applies `mask_K` by default and `mask_persist` on request; the
  mask file's two boolean columns are read (1-D, 2-D, structured or npz).
- al-Farabi 2.2: every temporal stage starts with `series features --all-grid --both` (norm
  only for `combined`); the driver refuses the split stage unless all 13 verdict CSVs and the
  13 (or 13, norm only) feature npz files exist, recording the check in
  `gates/grid_complete.json`.
- al-Farabi 2.5: Table 7's `G-C` column; Table 5's `refusal` carries the verdict on every row
  of a disconnected rung; every score cell of the rung prints `refused: disconnected lead`.
- al-Farabi 2.6: `gf --all-rungs` runs inside move 12 before `tables`; Table 7's `G-F (i)`
  column; Table 5's selected row carries it; the void string replaces the rung's score cells;
  move 13 is the driver's check that every Table 7 row carries a G-F (i) verdict
  (`gates/gf_check.json`).
- al-Farabi 2.7: `near_unfalsifiable` is printed in every score cell of that split; the score
  never is.
- al-Farabi 2.8: every driver command records `inputs_sha256`; the skip rule compares them and
  re-runs on any change with `stale: <file> changed since move <n>` in the ledger; `--force`
  stays unconditional; author inputs are never overwritten.
- al-Farabi 2.9(b): a rung with no selection prints `not run: no selection for <rung>` in the
  tables.
- al-Farabi 2.11(c): a best-feasible selection prints `selected: best-feasible` in Table 5 and
  `(selected: best-feasible)` in Table 7's resolution cell.
- al-Kindi 1, 2, 3, 4 and al-Farabi 2.1, 2.3, 2.4, 2.9(a), 2.9(c), 2.10, 2.11(a), 2.11(b) belong
  to builders 1 and 2; the report layer reads their results (`failed_source`, `G2_pairs`,
  `n_kernels_na_*`, `order-blind (by construction)`, the aliased-by-design G-C verdict, which is
  printed as it is and does not void the rung) and the runbook carries the gap-reading
  sentence of 2.10.

## 5. Cross-builder interface notes (read before the first real run)

These are things the report layer had to choose because SPEC leaves the location or the key
unnamed, or because the other builders' code as it stands today diverges from SPEC. None of
them is a threshold; each is a file path or a key name.

1. **The raw APF split variant has no path in SPEC 4.5.** The report layer looks for
   `<split>__<labelspace>__raw/`, then `<split>__<labelspace>/raw/`, then
   `gates/splits_raw/<rung>/<gid>/<split>__<labelspace>/`, then the shared directory when its
   `scores.json` says `params.normalized = false`. Builder 2's `run_split_stage` today writes
   both variants to the same directory, so under `--raw-and-norm` the raw scores are
   overwritten by the norm run and Table 6's raw columns will read `not run: ... missing`.
   Builder 2 should write the raw variant under one of the first three paths (the first is
   the report layer's preference).
2. **The `combined (matched)` scores.** SPEC 3.7.7 names no path. Builder 2's `gdim` writes
   them under `gates/splits_matched/combined/<gid>/loko__archetype/` (its docstring says
   `gates/splits/combined_matched/...`); the report layer reads both, and `gates/gdim.json`
   (`matched[<split>__<ls>]`) as a third option. Only the LOKO row is produced there; the
   `combined (matched)` LORO and within-trace rows read `not run:` naming the path.
3. **`gates/clustering.json` keys.** SPEC names none. The report layer reads per-cell labels
   under `labels` (a `{cell_id: label}` dict or a list aligned with `cell_id`), optionally
   nested under the rung and the algorithm (`combined` / `kmeans`), or a `count_matrix`; `k`
   from the same node or from `clustering.csv`.
4. **`gates/gj.json` keys.** SPEC names none for the floor quantiles; any key containing
   `quant` and `J` with five numbers (or a `q05..q95` dict) is accepted.
5. **`gates/selection.json` layout.** Rung keys at the top level beside `schema`, `params`,
   `citation`, or under a `selection`/`rungs` key: both are read.
6. **`gates/gf.csv` part label.** `i`, `1`, `(i)`, `part1` or `part (i)` are all read as
   part (i).
7. **Table rows use the B1-G3 re-run** when `scores.json["with_quarantine"]` has a non-empty
   `quarantined_features` list (SPEC 3.7.2).

## 6. For the author

Choices the definitions leave open that this layer exposes as parameters (each has a default
that runs unless changed, and the value used is written into the result's `params`):

1. `table8_rung = "combined"` (SPEC section 8 item 32): the rung whose LOKO assignment fills
   Table 8; `--table8-rung apf` etc. for the alternative.
2. `fused_plane_mask = "K"` (al-Kindi review 9): the G-J mask applied to the fused plane;
   `persist` uses `n_persist` against three times the idle cells' median `n_persist`.
3. `decay_jump_ratio = 1.5` (al-Kindi review 1): the K-jump detection ratio the floyd figure
   uses to reconstruct pass boundaries for drawing; the same value G-C detects with.
4. `piano_cell` = the first gemm cell in `cells.csv` order, `piano_stride = 16` (SPEC 6.7).
5. The LaTeX document class (IEEEtran conference or article) and whether the standalone copy
   is regenerated after the tables exist (`--standalone`).
6. Whether LORO's null column is run (`--null-splits`; SPEC section 8 item 25): about 48,000
   forest fits per rung at 500 permutations.
7. The `failed/` count: `--assume-failed-zero --assume-reason "AA A5: any failed job re-runs
   the whole cell"` is the runbook's path (SPEC section 8 item 37); the text goes into every
   persistence reading's `params` and belongs in Limitations once (al-Farabi review, for the
   author 5).
8. Table 5's `refusal` column composes three strings (the row's own refusal, `G-C: ...` on a
   disconnected rung, `G-F (i): ...` on the selected row) separated by `; `; if you prefer
   separate columns, say so.
9. Table 6's `null p95`, `majority` and `rank` are LOKO's values repeated on every row (P2
   Sec. VII: "null and majority on every row"); the archetype rows' within-trace and LORO
   cells are the mean of the member kernels' recalls (SPEC 6.2).
10. The `physical reason` column of Table 8 is empty text for you.
11. The skeleton's substance bullets are abbreviations of P2_STRUCTURE.md section 3; every
    bullet is a comment and can be deleted or reworded without touching the structure.
12. The interface notes of section 5 are for builder 2 as much as for you; item 1 (the raw
    APF split path) is the one that changes a table's content if left as it is.

Things this layer does not do, by design: it never recomputes a verdict from the extracts (the
one arithmetic it does is the wAPF-over-APF table's means from the extract columns and the
figures' plotting); it never reads or names the sandbox family; it never deletes anything
under `gates/`; it never overwrites an author input.
