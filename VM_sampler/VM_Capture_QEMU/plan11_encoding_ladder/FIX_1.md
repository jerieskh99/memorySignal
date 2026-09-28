# FIX_1.md: the fixer's report on the paper 2 analysis toolkit, cycle 1 (2026-09-16)

Rules kept: no server, no path under a server mount, no remote command; the sandbox family is not
read, grepped or named; no paper prose; no file outside `plan11_encoding_ladder/` was edited;
nothing was committed; `P2_STRUCTURE.md` was not touched. Every test and the driver run below used
synthetic data generated on this machine with builder 1's generator. Environment: Python 3.10,
numpy, scikit-learn, scipy, matplotlib present, pytest present, the `zstd` binary on PATH, the
`zstandard` module absent.

Applied: the five BLOCKING findings of `CHECK_1.md` (B1 to B5), the one failing condition of
`CERTIFY_al_farabi.md` (condition (1), the resume path; its correction 7.1), and the MINOR findings
whose fix is one line (M6, M7, M11, M12 (a) and (b)). Every other MINOR finding is listed below as
refused with the reason, which in every case is that the fix is more than one line. No threshold
and no verdict string was moved.

The line numbers below are those of the files as they stand after the edits.

---

## 1. Per finding

### B1. The raw APF split run was overwritten by the normalized run. Fixed.

`models.py:340-346`: `split_dir` takes `normalized: bool = True` and returns
`gates/<base>/<rung>/<grid_id>/<split>__<labelspace>` for the normalized run and
`<split>__<labelspace>__raw` for the raw run; the docstring cites SPEC 4.5, SPEC 6.2 (Table 6's
raw columns) and CR 2.2 item 24 (G-L (i)'s level-inclusive ceiling). `models.py:369`
(`run_split_stage`) passes `normalized=normalized`; the docstring of `run_split_stage` names the
`__raw` directory. Every other caller of `split_dir` (`gates_comparison._scores`, the tests, the
report layer's own `split_dir`) reads the normalized run and is unchanged. Builder 3's reader
(`_report_common.split_dir(raw=True)`) already tries `__raw` first, so `tables.py` needed no
change.

Test added: `tests/test_gates_models.py:73` `test_raw_and_norm_write_two_directories_and_keep_both`
runs the `splits` CLI with `--raw-and-norm` on a synthetic corpus and asserts that
`loko__archetype/scores.json` (`params.normalized = true`) and `loko__archetype__raw/scores.json`
(`params.normalized = false`) both exist with their four files, and that builder 3's reader
resolves each to the right directory.

Verified on the driver run (section 3): `gates/splits/apf/W8_H4/` holds ten directories, five
`__raw` beside five normalized, and `report/tables/table6.csv` has numeric `within-trace raw`,
`LORO raw` cells beside the norm cells on the gemm, WORKING-SET and `all` rows (the LOKO cells
read `--` on that run because only gemm passed C1 at the default threshold, the M1 condition,
which is not part of this cycle).

### B2. G-L ran once, at move 7, so four rungs never received a G-L verdict. Fixed.

`run_moves.py:210`: `gl (all rungs)` is scheduled inside move 12 immediately after `gx combined`
and before `gf all rungs at the selected points`, with `outputs=["gates/gl.csv"]` and
`inputs=["cells.csv", "gates/selection.json"]`, so the staleness rule re-runs it whenever a later
selection lands. The move-7 run stays and fills APF's row early. `RUNBOOK.md:340` carries the same
command in move 12 between `gx --rung combined` and `gf --all-rungs`, and the "Writes" paragraph of
move 12 (`RUNBOOK.md:352-353`) names the rewritten `gl.csv`.

Verified on the driver run: `gates/gl.csv` has a part (i) row with a grid id for every rung
(`apf W8_H4, wapf W8_H4, persist W8_H4, content W16_H8, combined W16_H8`) and Table 7's `G-L`
column no longer reads `not run: no selection for <rung>` on any row. On that run every part (i)
verdict is `not run: LOKO/archetype scores.json missing`, because with one kernel surviving C1
the LOKO split has no accuracy; see "For the author" item 2 for the wording of that string.

Test added: `tests/test_driver.py` `test_move_table_carries_the_review_corrections` asserts the
presence and the order of `gl (all rungs)` in move 12.

### B3. G-C never ran for the combined rung. Fixed.

`run_moves.py:133`: move 3 iterates `("apf", "persist", "content", "wapf", "combined")`, combined
last because `gate_gc` reads the other four `rep = all` rows from `gc.csv` (SPEC 3.4.3; CR 2.2
item 22). `RUNBOOK.md:173` loops over the same five and the paragraph after it says that
`combined` runs last and that re-running any single rung drops the combined row (that is how
`gate_gc` is written: `gates_calibration.py`, the `old` filter), so `--rung combined` is re-run
after any of the four. No change in builder 2's code.

Verified on the driver run: `gates/gc.csv` has the row
`combined,apf+wapf+persist+content,all,...,"not run: no admissible cell for gibbs,histogram"`, the
content rung's verdict propagated by the combined rule, and Table 7's `G-C` cell for `combined`
and `combined (matched)` carries that string instead of `not run: gates/gc.csv missing or has no
row for combined`. `_report_common.rung_override` therefore sees the combined rung's G-C verdict
and will mask its score cells when a constituent rung is `disconnected lead`.

Test added: `tests/test_driver.py` asserts move 3's five commands in that order.

### B4. The J-histogram figure never found the floor null in builder 2's `gj.json`. Fixed.

`figures.py:317-350` (`_floor_j_quantiles`): builder 2's exact shape is read first, `idle_J =
{"quantiles": [...], "J": [q05, ..., q95], ...}`; `idle_J = null` (no idle cell) and
`idle_J.J = null` (idle cells with no valid J pair) are recognised and named; the walker over
`*quant*J*` keys stays as the fallback. The function now returns `(quantiles, reason)`, and
`fig_j_hist` (`figures.py:365, 395, 399`) prints the actual reason on the x axis (`no idle cell
(gates/gj.json idle_J = null)`, `gates/gj.json missing`, `gates/gj.json carries no idle J
quantiles`) and records it as `floor_quantiles_absent_reason` beside `floor_quantiles` in
`figures.json`. `_floor_j_quantiles` has one caller; the checker's direct call now returns a pair.

Fixture and tests: `tests/report_fixtures.py` writes `gj.json` in builder 2's exact shape
(`idle_J` with `quantiles`, `J`, `n_pairs`, `mean`; `null` when the fixture has no idle cell).
`tests/test_report.py:332` `test_j_hist_reads_builder_2_floor_quantiles` covers builder 2's shape
(five quantiles found, empty reason), `idle_J = null` (none found, reason names `no idle cell`),
the missing file, and the walker fallback. The existing `test_all_figures_written` still asserts
the five quantiles through the new path.

Verified on the driver run (two idle cells): `figures.json` `written.j_hist.floor_quantiles` holds
five values and `floor_quantiles_absent_reason` is empty; the generator's idle cell has a constant
J, so the five quantiles coincide, which is the corpus and not the reader.

### B5. Table 8's clustering column never found builder 2's cluster labels. Fixed.

`tables.py:593-620` (`_cluster_counts`): when `clustering.json` has `per_algo[<algo>]`, the node is
`{**per_algo[algo], "cell_id": j["cells"], "k": j["k"]}` with `count_matrix` set to
`cluster_by_predicted_archetype` when absent, and the existing branches run (`labels` aligned
with `cell_id` first, the count matrix second); the rung-and-algorithm nesting remains the
fallback. One more line: a `clustering.json` whose top-level `status` is a `not run:` string
(builder 2's no-selection file) returns that string instead of `has no labels or count_matrix`,
so the cell names what is missing.

Fixture and tests: `tests/report_fixtures.py` writes `clustering.csv` and `clustering.json` in
builder 2's exact layout (`k`, `cells`, `kernels`, `archetype`, `per_algo` with `labels`, `ari`,
`nmi`, `cluster_by_predicted_archetype`). `tests/test_report.py:212`
`test_table8_reads_builder_2_clustering_layout` asserts the counts sum to the cell count, every
archetype row of Table 8 prints `c<n>:<count>`, the count-matrix route gives the same counts when
`labels` is removed, and the no-selection file's `not run` string is passed through.

Verified on the driver run: `report/tables/table8.csv` prints `c0:3` in the `clusters (k = 1)`
cell of the WORKING-SET row (k = 1 because only gemm survived C1 on that run).

### al-Farabi condition (1), the resume path: `select` was not re-run after its grid was rebuilt. Fixed.

`run_moves.py:158-162`: the `select <rung>` step's inputs are
`inputs/pass_table.csv, gates/gk0.csv, gates/gc.csv` plus the 13
`gates/grid/<rung>/<grid_id>/temporal_per_kernel.csv`, exactly the files `select` itself hashes
into `selection.json` `params.inputs_sha256_<rung>` (certification 7.1, first alternative). A
rebuilt grid therefore makes the selection `stale` and the driver re-runs it.

Verified in place on the finished driver run: resuming move 6 in dry-run with nothing changed
reports `select apf` as `skipped: outputs exist and inputs unchanged`; after one byte is appended
to `gates/grid/apf/W8_H4/temporal_per_kernel.csv`, the same resume reports `select apf` as
`dry-run (stale: temporal_per_kernel.csv changed since move 6 ...)`. (The ledger and the CSV were
restored afterwards.) Test added: `tests/test_driver.py` asserts the 13 grid CSVs, `gk0.csv` and
`gc.csv` in the `select apf` inputs.

### M6. `gdim` ignored the driver's `--null-perm`. Fixed (one line).

`run_moves.py:213`: `--null-perm <o.null_perm>` appended to the `gdim` args. On the driver run
`gdim` took 1.7 s.

### M7. `select` wrote G-C's not-applicable and not-run verdicts into Table 5's `refusal` column. Fixed (one line).

`gates_temporal.py:607`: `refusal` carries `gc_v` only when it equals `V.GC_DISCONNECTED` (CR 2.2
item 22, the one G-C verdict that refuses the rung); `gc_verdict` on the same line still carries
every G-C verdict. Verified on the driver run: the content rung's `table5_grid.csv` rows and
`selection.json["content"]` have `refusal = ""` and `gc_verdict = "not run: no admissible cell for
gibbs,histogram"`; the existing test `test_gc_verdict_propagates_into_the_grid_refusal_column`
(disconnected lead in every row) still passes.

### M11. Three gate-decision helpers carried no citation. Fixed (one line each).

`gates_temporal.py:148` (`g1_kernel`: P2 Sec. V 5.1; CR 2.1 item 3; `CIT_GRID`),
`gates_temporal.py:268-269` (`g4_pass`: a docstring citing CR 2.1 item 6 and `CIT_GRID`, and the
copy comment in the SPEC's form `# copied from plan03_aggregate.py:g4_pass, 2026-09-16`),
`variance.py:24` (`variance_levels`: P2 Sec. V 5.2 G-V; CR 2.3 item 35; `CIT_GV`). The docstring
lines were reflowed to the file's width; no logic touched.

### M12. Two documentation mismatches. Fixed (one line each).

(a) `gates_comparison.py:262`: the `gate_gdim` docstring names the directory the code writes,
`gates/splits_matched/combined/<grid_id>/loko__archetype/`.
(b) `SPEC.md:505`: the on-disk guarantee pointer in 3.1.3 reads 3.5.7.

### Refused (more than one line; nothing else in the brief covers them)

- M1 (`--c1-activity-min` through the driver and the runbook; `preconditions.csv` in every
  downstream `inputs` list): a new flag, its plumbing, a runbook paragraph and a change to
  every command's inputs. Not one line. It is also the threshold decision the checker put to
  the author ("For the author" item 1 of `CHECK_1.md`), which the fixer does not make.
- M2 (the runbook's `unittest` alternative runs 94 of the tests): two sentences in `RUNBOOK.md`
  and a line in `requirements.txt`. Not one line. Noted again under "For the author" because the
  author may run it on the server and read "OK".
- M3 (three shorthand runbook lines `grid/g3/gord/select`): four commands each, three places.
- M4 (`gord --n-jobs` accepted and ignored; the cost note): a parallelization or a CLI removal
  plus the runbook cost sentence.
- M5 (the G-L (ii) re-run with `feature_drop` is not driven): a conditional driver command, a
  `--base-dir` flag on the `splits` CLI and a Table 6 change (also al-Farabi 7.4).
- M8 (G1 with no applicable kernel counted as a failed gate in the roll-up): a two-line rule
  change and a new column; and it is a reading of al-Farabi 2.1 that the author should confirm.
- M9 (runbook timing claims; `--n-estimators` on the driver): a paragraph and a flag.
- M10 (`tests/test_driver_chain.py`): a new test file. Section 3 below records the equivalent
  manual run for this cycle.
- al-Farabi 7.2 (a `grid incomplete` or `no applicable kernel` refusal in the selection entry),
  7.3 second half (`grid_source` in the split stage's `params`; the CLI refusing without a
  selection), 7.5 (the template CLIs refusing to overwrite): listed by the certification as
  proposals under conditions that hold, not as fails; each is more than one line.

---

## 2. The tests, verbatim

Command, from `plan11_encoding_ladder/`: `python3 -m pytest -q -rs tests`

```
.............................s.......................................... [ 45%]
........................................................................ [ 91%]
..............                                                           [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_extract.py:515: zstandard module not installed
157 passed, 1 skipped in 312.97s (0:05:12)
```

Before this cycle the suite was 154 passed, 1 skipped; the three new tests are
`test_raw_and_norm_write_two_directories_and_keep_both` (B1),
`test_table8_reads_builder_2_clustering_layout` (B5) and
`test_j_hist_reads_builder_2_floor_quantiles` (B4); the driver plan test grew four assertions
(B2, B3, M6, al-Farabi 7.1). The skip is the `zstandard` module path, absent on this machine.

---

## 3. The driver end to end on a synthetic corpus, verbatim

Corpus: builder 1's presets (`synth.corpus_specs`) filtered to gemm, floyd, gibbs and histogram,
3 reps each, plus 2 idle cells, 120 pairs per cell, compressed with the `zstd` binary: 14 cells
(the checker's run 1 recipe). Command, from `VM_Capture_QEMU/`:

```
python3 -m plan11_encoding_ladder.run_moves run --out <S>/out --root <S>/root \
    --null-perm 20 --n-jobs 4 --assume-failed-zero --assume-reason "fixer cycle 1 smoke run"
```

Exit 0 after 4 min 51 s wall time; 78 ledger entries `done` (76 before this cycle, plus
`gc combined` at move 3 and `gl (all rungs)` at move 12); `gf_check` `pass`. The five outputs the
checker found wrong on this recipe now read:

```
gates/splits/apf/W8_H4/: loko__archetype  loko__archetype__raw  loro__archetype  loro__archetype__raw
                         loro__kernel  loro__kernel__raw  within_trace__archetype
                         within_trace__archetype__raw  within_trace__kernel  within_trace__kernel__raw
  (every __raw scores.json: params.normalized = false; every other: true)

report/tables/table6.csv (header and the all row):
row,level set,n,within-trace raw,within-trace norm,LORO raw,LORO norm,LOKO raw,LOKO norm,null p95 (LOKO norm),majority (LOKO),rank (LOKO norm),G-N status,G-X
all,,12,1,1,1,1,--,--,not run: 20 permutations < 500,0,not run: 20 permutations < 500,,not applicable: one campaign label; confound: none

gates/gl.csv:
rung,part,grid_id,score_norm,null_p95,r2,slope,n_points,verdict
apf,i,W8_H4,,,,,,not run: LOKO/archetype scores.json missing
wapf,i,W8_H4,,,,,,not run: LOKO/archetype scores.json missing
persist,i,W8_H4,,,,,,not run: LOKO/archetype scores.json missing
content,i,W16_H8,,,,,,not run: LOKO/archetype scores.json missing
combined,i,W16_H8,,,,,,not run: LOKO/archetype scores.json missing
apf,ii,W8_H4,,,,,1,not run: fewer than three points or no spread

gates/gc.csv (rep = all rows):
apf,gemm,all,12,,,,,,pass
persist,gemm,all,12,,,,,,pass
content,gibbs+histogram+gemm,all,,,,,,,"not run: no admissible cell for gibbs,histogram"
wapf,gemm,all,12,,,,,,pass
combined,apf+wapf+persist+content,all,,,,,,,"not run: no admissible cell for gibbs,histogram"

report/figures/figures.json written.j_hist:
  floor_quantiles = [0.9607843137, 0.9607843137, 0.9607843137, 0.9607843137, 0.9607843137]
  floor_quantiles_absent_reason = ""

report/tables/table8.csv:
predicted \ measured,IDLE (measured),WORKING-SET,SCATTER,SEQUENTIAL-GROW,FRONTIER-CHURN,clusters (k = 1),physical reason
IDLE (0 predicted),0,0,0,0,0,0,
WORKING-SET (3),0,0,0,0,0,c0:3,
SCATTER (1),0,0,0,0,0,0,
SEQUENTIAL-GROW (0),0,0,0,0,0,0,
FRONTIER-CHURN (0),0,0,0,0,0,0,
```

The `--` and `not run` cells above are the M1 condition (floyd, gibbs and histogram fail C1 at
the default 0.02, so one kernel and the idle cells remain; LOKO has no accuracy and clustering has
k = 1) and the smoke run's 20 permutations; neither is a finding of this cycle.

The resume check for al-Farabi condition (1), on the same run directory, verbatim:

```
--- before any change: resume move 6 dry-run ---
[move  6] features apf all grid                        skipped: outputs exist and inputs unchanged
[move  6] grid apf                                     skipped: outputs exist and inputs unchanged
[move  6] g3 apf                                       skipped: outputs exist and inputs unchanged
[move  6] gord apf                                     skipped: outputs exist and inputs unchanged
[move  6] select apf                                   skipped: outputs exist and inputs unchanged
[move  6] alias                                        dry-run  (stale: g3_flags.csv changed since move 6 (2026-09-16T16:25:22+00:00))
--- one grid CSV rewritten (a rebuilt grid) ---
[move  6] features apf all grid                        skipped: outputs exist and inputs unchanged
[move  6] grid apf                                     skipped: outputs exist and inputs unchanged
[move  6] g3 apf                                       skipped: outputs exist and inputs unchanged
[move  6] gord apf                                     skipped: outputs exist and inputs unchanged
[move  6] select apf                                   dry-run  (stale: temporal_per_kernel.csv changed since move 6 (2026-09-16T16:25:22+00:00))
[move  6] alias                                        dry-run  (stale: g3_flags.csv changed since move 6 (2026-09-16T16:25:22+00:00))
```

(`alias` at move 6 is stale on any resume because `g3_flags.csv` is shared and moves 9 to 12
append to it; `alias` is idempotent and the move-7 re-run covers it. Pre-existing, not changed.)

---

## 4. Files touched

Code: `models.py`, `run_moves.py`, `figures.py`, `tables.py`, `gates_temporal.py`,
`variance.py`, `gates_comparison.py`. Documentation: `RUNBOOK.md` (moves 3 and 12), `SPEC.md`
(one pointer, 3.1.3). Tests: `tests/test_gates_models.py`, `tests/test_report.py`,
`tests/report_fixtures.py`, `tests/test_driver.py`. This report. Nothing else.

---

## 5. For the author

Choices the definitions leave open that this cycle touched or noticed; the fixer decided none of
them.

1. **`figures.json` records `floor_quantiles_absent_reason`.** The reason strings are the
   fixer's wording (`no idle cell (gates/gj.json idle_J = null)`, `gates/gj.json missing`,
   `idle cells have no valid J pair (gates/gj.json idle_J.J = null)`, `gates/gj.json carries no
   idle J quantiles`). They are figure-axis text, not verdict vocabulary; rename them if you
   prefer.
2. **G-L (i)'s `not run` string when the LOKO split has no accuracy.** `gate_gl`
   (`gates_comparison.py`, the `sc is None or sc.get("accuracy") is None` test) writes
   `not run: LOKO/archetype scores.json missing` in both cases; on the run above the file exists
   and its `accuracy` is `null` because one kernel survived C1. A second string for that case
   (`not run: LOKO/archetype has no score`) would name what is missing more exactly. Not changed:
   it is builder 2's wording and not a finding of this cycle.
3. **The combined rung's G-C verdict when a constituent is `not run`.** SPEC 3.4.3 says a
   not-applicable rung makes combined not applicable; `gate_gc` extends this to any non-pass
   verdict (`next(v for v in verdicts.values() if v != PASS)`), so on the run above combined
   reads the content rung's `not run: no admissible cell for gibbs,histogram`. That is the
   existing rule; B3 only made the driver execute it.
4. **`SPEC.md` section 7's move table** still lists move 3 as "`gc --rung apf` and the same for
   `persist`, `content`, `wapf`" and move 12 without the second `gl`; the driver and the runbook
   now run five G-C rungs and the second G-L. Editorial; not changed because the SPEC is the
   builders' contract and the brief limits SPEC edits to one-line pointers.
5. **M2 stands.** `python3 -m unittest discover -s tests -p "test_*.py"` still runs 94 of the
   tests and reports OK; use `pytest` on the server (`python3 -m pip install --user pytest`).
6. **The checker's "For the author" items 1 to 5** (C1's threshold, B1-G1's refusal in Table 6,
   G2 in seconds versus pairs, G-DEC's idle clause, the exposed defaults) and al-Farabi's section 6
   items 1 to 7 are unchanged by this cycle and still need decisions before the data.
