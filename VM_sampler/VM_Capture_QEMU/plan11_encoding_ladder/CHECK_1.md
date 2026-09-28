BLOCKING FINDINGS: 5 (B1 to B5); MINOR: 12. The test suite passes (154 passed, 1 skipped) and the driver runs end to end with exit 0, but one table loses its raw columns, two gates never reach the artifact for some rungs, one figure loses a null, and one table column is always `not run`.

# CHECK_1.md: checker's report on the paper 2 analysis toolkit, cycle 1 (2026-09-16)

Scope and rules kept: no server, no path under a server mount, no remote command; the sandbox
family is not read, grepped or named; no paper prose was written; no file outside
`plan11_encoding_ladder/` was edited and nothing was committed. Every test and every driver run
below used synthetic data generated on this machine with builder 1's generator (`synth.py`).
Environment: Python 3.10.12, numpy 2.2.6, scikit-learn 1.7.2, scipy 1.15.3, matplotlib present,
pytest 9.1.1, the `zstd` binary on PATH (1.5.7), the `zstandard` module absent, no TeX compiler.

Findings are listed most severe first under BLOCKING and MINOR. Each carries the file and line,
what is wrong, and the exact fix. A section "For the author" lists the choices the definitions
leave open that this check surfaced as decisive, and the record of every run follows.

---

## BLOCKING

### B1. The raw APF split run is overwritten by the normalized run, so Table 6 has no raw column

`models.py:340-341` (`split_dir`) builds the split directory as
`gates/<base>/<rung>/<grid_id>/<split>__<labelspace>` with no raw/norm segment, and
`models.py:363` (`run_split_stage`) writes both variants there. `models.py:626` (`--raw-and-norm`)
runs the raw variant first and the normalized variant second into the same directory, so the raw
`scores.json`, `predictions.csv`, `null.json` and `l1_quarantine.json` are silently replaced.

What it breaks: P2 Sec. VII and Sec. 4 Table 6 ("APF alone under the three splits, raw and
level-normalized"). Builder 3's report layer looks for `<split>__<labelspace>__raw/`, then
`<split>__<labelspace>/raw/`, then `gates/splits_raw/...`, then the shared directory only when
its `params.normalized` is false; after the overwrite none of these holds. Verified on the
synthetic driver run: every raw cell of `report/tables/table6.csv` reads
`not run: gates/splits/apf/W8_H4/<split>__<labelspace>__raw/scores.json missing`, and
`gates/splits/apf/W8_H4/loko__archetype/scores.json` carries `params.normalized = true`. The raw
run's B1-G3 quarantine (SPEC 5.3: the level-only corpus must quarantine `apf.k_over_n.mean` on the
raw APF features) is lost with it. Builder 3 named this gap in `BUILD_report_and_skeleton.md`
section 5 item 1; it is confirmed and it changes a table's content.

Fix (builder 2, three lines): give `split_dir` a `normalized: bool = True` parameter and return
`base / rung / grid_id / (f"{split}__{labelspace}" + ("" if normalized else "__raw"))`; pass
`normalized` at `models.py:363`; leave every other caller (they are all normalized runs). Builder
3's first candidate path is exactly `__raw`, so `tables.py` needs no change. Add to
`tests/test_gates_models.py` one assertion that after `--raw-and-norm` both
`loko__archetype/scores.json` (normalized true) and `loko__archetype__raw/scores.json`
(normalized false) exist.

### B2. The driver runs G-L once, at move 7, so four of the five rungs never receive a G-L verdict

`run_moves.py:172` schedules `gates_comparison gl` inside move 7 only, when only APF has a
selection. `gates_comparison.py:71-96` (`gate_gl`) writes `not run: no selection for <rung>` for a
rung without a selection and is never called again after moves 9 to 12 create the selections of
`persist`, `content`, `wapf` and `combined`. Verified on the synthetic run: `gates/gl.csv` and
Table 7's `G-L` column read `(i) not run: no selection for wapf` (and persist, content, combined)
after the full run to move 13 completed.

What it deviates from: P2 Sec. V 5.2 G-L (i) and CR 2.2 item 24 are per rung ("the rung's Table 7
row is marked 'level only' and the rung cannot be cited in the recovery clause"); G-L is in
al-Kindi's minimum set (P2 Sec. V 5.4). As driven, the verdict never reaches the artifact for
four rungs, which is a silently absorbed gate.

Fix (builder 3): in `run_moves.py:build_plan`, add
`P.append(_cmd(12, "gl (all rungs)", "gates_comparison", "gl", [*O], outputs=["gates/gl.csv"], inputs=["cells.csv", "gates/selection.json"]))`
immediately after the `gx combined` command (before `gf all rungs at the selected points` at
line 202), and add the same line to `RUNBOOK.md` move 12 (between `gx --rung combined` and `gf
--all-rungs`). The move-7 run may stay (it fills APF's row early). The `gl.csv` `params` already
carry `inputs_sha256` of `selection.json`, so the staleness rule re-runs it when a later
selection lands.

### B3. The driver never runs G-C for the combined rung, so the combined rows of Table 7 carry no calibration verdict and are never voided

`run_moves.py:132` iterates `("apf", "persist", "content", "wapf")` at move 3; `combined` is
absent. `gates_calibration.py:355-366` defines the combined verdict (pass iff the four pass;
disconnected if any is disconnected; SPEC 3.4.3) but is never invoked by the driver or by the
runbook (`RUNBOOK.md:173` loops the same four). Verified on the synthetic run: Table 7's `G-C`
cell for `combined` and `combined (matched)` reads `not run: gates/gc.csv missing or has no row
for combined`, and `_report_common.py:540-551` (`rung_override`) therefore never masks the
combined rung's scores even when a constituent rung is `disconnected lead`.

What it deviates from: CR 2.2 item 22 ("refuses the whole rung ... every negative it produced is
void until the implementation is fixed"), al-Farabi review 2.5 (Table 7's G-C column on every
rung, the void string in every score cell), and P2 Sec. IV rung 3 (the combined rung is the union
of the four; a void member voids the union).

Fix (builder 3): change `run_moves.py:132` to iterate `("apf", "persist", "content", "wapf",
"combined")` (combined last: `gate_gc` reads the other four rows from `gc.csv`), and change the
runbook's move 3 loop at `RUNBOOK.md:173` to `for r in apf persist content wapf combined`. No
change in builder 2's code.

### B4. The J-histogram figure never finds the floor null in builder 2's `gj.json`, so P2's "both nulls" figure draws one null and mislabels the absence

`gates_readings.py:78` writes the idle cells' J distribution as
`{"idle_J": {"quantiles": [0.05, ...], "J": [q05, ..., q95], "n_pairs": ..., "mean": ...}}`.
`figures.py:317-343` (`_floor_j_quantiles`) walks the file for a key whose name contains both
`quant` and `j`; `idle_J` contains no `quant` and `quantiles` contains no `j`, so it returns
`None`. Verified by calling `figures._floor_j_quantiles` on run 2's `gj.json` (two idle cells,
`idle_J.J` present): `None`. `figures.py:351` and `:381` then draw `fig_j_hist` without the floor
quantile lines and print on the x axis `floor null: gates/gj.json missing, no idle quantiles
drawn`, which is false when the file exists.

What it breaks: P2 Sec. VII, "J(t) histograms per kernel with both nulls" (the independence null
and G-J's empirical floor null, CR 2.2 item 31; K2 move 9); the figure carries one null and a
wrong caption text once the idle cells are captured.

Fix (builder 3, three lines): at the top of `_floor_j_quantiles`, after `j = read_json(p)`,
`ij = j.get("idle_J"); if isinstance(ij, dict) and isinstance(ij.get("J"), list) and len(ij["J"]) == 5: return [float(x) for x in ij["J"]]`;
keep the walker as the fallback and change the axis text at `figures.py:381` to name the actual
reason (`no idle cell` when `gj.json` has `idle_J = null`). Add to `tests/test_report.py` a case
whose `gj.json` fixture has builder 2's exact shape (`report_fixtures.py` presently writes builder
3's own guess).

### B5. Table 8's clustering column never finds builder 2's cluster labels

`models.py:570-573` (`run_clustering`) writes `clustering.json` as `{"k": ..., "cells": [...],
"kernels": [...], "archetype": [...], "per_algo": {"kmeans": {"labels": [...], "ari": ..., "nmi": ...,
"cluster_by_predicted_archetype": {archetype: {"c0": n, ...}}}, ...}}`. `tables.py:593-631`
(`_cluster_counts`, ending before `table8` at line 634) descends `j[rung][algo]` (no `per_algo` level), pairs a `labels` list with a
`cell_id` key (builder 2's is `cells`), and accepts a count matrix only under `count_matrix`
(builder 2's is `cluster_by_predicted_archetype`). Verified on the synthetic run:
`report/tables/table8.csv` prints `not run: gates/clustering.json has no labels or count_matrix`
in every row's `clusters (k = 1)` cell while `gates/clustering.csv` and `.json` are present and
complete. Builder 3 named the key uncertainty in `BUILD_report_and_skeleton.md` section 5 item 3;
none of its guesses matches.

What it breaks: P2 Sec. VII Table 8 ("beside the clustering"; K2 move 13: "beside the clustering
with k fixed ... ARI and NMI against the unit-level null"). The column is always `not run`.

Fix (builder 3, four lines in `_cluster_counts` after `j = read_json(p)`):
`pa = j.get("per_algo", {}).get(algo)`; if it is a dict, set `node = {**pa, "cell_id": j.get("cells"), "k": j.get("k")}`
and, when `"count_matrix" not in node`, `node["count_matrix"] = pa.get("cluster_by_predicted_archetype")`;
then let the existing branches run. Add to `tests/test_report.py` a fixture with builder 2's
exact layout (`report_fixtures.py` presently writes a `labels` dict).

---

## MINOR

### M1. C1's default threshold (`apf_max >= 0.02`) is not reachable from the driver or the runbook, and on the synthetic corpus it excludes 9 of 12 kernel cells before any gate runs

`gates_precondition.py:28` (`C1_ACTIVITY_MIN = 0.02`) implements CR 2.1 item 1 as written (the
apf_queue re-map, only the idle cell re-mapped), and exposes `--c1-activity-min` at
`gates_precondition.py:476`. `run_moves.py` does not pass it through (no such flag in
`_add_run_args`, line 460 onward) and `RUNBOOK.md` never mentions it. On the synthetic driver run
with the default, floyd, gibbs and histogram were all `C1 = fail` (`apf_max` 0.0165, 0.0016,
0.0088) and `all_hard_pass = false`, so G-C content read `not run: no admissible cell for
gibbs,histogram`, G-DEC `not run: no admissible floyd cell`, G-ORD `not run: fewer than two
archetypes with windows`, and every split ran on gemm alone (`n_all_hard_pass = 5` of 14). The
runbook's smoke run (section 0b, "every table present") therefore cannot exercise G-C content,
G-DEC, G-ORD or any LOKO fold. Builder 2 flagged the threshold in `BUILD_gates.md` section 6 item
1; the missing pass-through is the finding here. Builder 2's own cross-builder test
(`tests/test_gates_chain.py:29`) lowers the threshold to `c1_activity_min=0.0005` before anything
else runs, so the default path is never tested end to end. Not a deviation from the definition.

Fix (builder 3): add `--c1-activity-min` (float, default 0.02) to `_add_run_args`, append it to
the `preconditions` command's args at `run_moves.py:119-124`, record it in the ledger `params`,
and name it in `RUNBOOK.md` move 2 with the sentence that the default excludes every kernel whose
`K_max` is below 5,243 pages (see "For the author" item 1). Also add `gates/preconditions.csv` to
the `inputs` list of every downstream command in `build_plan` (moves 3 to 13), so that a
re-run of move 2 with a different threshold makes the downstream outputs `stale` instead of
leaving them skipped as done (today only `--force` recovers).

### M2. The runbook's `unittest` alternative silently runs 94 of 155 tests

`RUNBOOK.md:26` and `RUNBOOK.md:55` say the tests "also run under `python3 -m unittest discover
-s tests -p "test_*.py"`". Verified: that command reports `Ran 94 tests ... OK (skipped=1)`,
while pytest collects 155 (154 passed, 1 skipped). Builder 2's twelve test modules
(`tests/test_gates_*.py`, 61 tests) are pytest-style functions with no `TestCase`, so
`unittest` discovers nothing in them and reports success; `BUILD_gates.md` section 4 item 19 says
so. An author on a server without pytest would read "OK" and believe every gate test passed.

Fix (builder 3): replace the alternative at `RUNBOOK.md:55` with "`pytest` is required for the
gate tests (61 of the 155); `python3 -m pip install --user pytest` when absent; `unittest`
discovery runs only builders 1 and 3's 94 tests", and change the parenthesis at `RUNBOOK.md:26`
accordingly. `requirements.txt` may list `pytest` as required for the tests rather than optional.

### M3. Three runbook lines are shorthand and do not run as written

`RUNBOOK.md:301`, `319`, `334`:
`python3 -m plan11_encoding_ladder.gates_temporal grid/g3/gord/select --out <out> --rung content
(four commands, as above)`. argparse rejects `grid/g3/gord/select`. Every other command in the
runbook was checked against the CLIs (`extract`, `synth`, `series`, `gates_precondition`,
`gates_calibration`, `gates_temporal`, `gates_readings`, `models`, `gates_comparison`,
`variance`, `tables`, `figures`, `latex_skeleton`, `run_moves run|status|plan`, `driver`) and
runs as written, including `run_moves plan --moves 6-7` and `tables --only manifest`.

Fix: expand the three lines to the four commands each (as move 6 and move 9 already do), or
replace them with `python3 -m plan11_encoding_ladder.run_moves plan --out <out> --moves 10` and
say the plan prints the exact lines.

### M4. `gates_temporal gord` accepts `--n-jobs` and ignores it; the driver passes it and the SPEC's cost note assumes it parallelizes

`gates_temporal.py:424-425` (`gate_gord`) takes `n_jobs` and never uses it (the only occurrence
is the signature); `run_moves.py:71-72` lists `("gates_temporal", "gord")` in `NJOBS_COMMANDS`.
On the synthetic run G-ORD took 7 min 47 s for one rung of 12 cells at 120 pairs with
`--n-jobs 4` (121 LOKO runs per W, 4 W, 300 trees), and it is the step that kept builder 3's
39-cell run from reaching move 7. The 96-cell run will take hours per rung at the defaults
regardless of `--n-jobs`.

Fix (builder 2): parallelize the `n_order_perm` and `null_perm` loops of `gate_gord` over
`n_jobs` with `joblib.Parallel` as `_gf_part1` (`gates_precondition.py:347-349`) already does,
or drop `--n-jobs` from the CLI and from `NJOBS_COMMANDS`; either way state the measured cost in
`RUNBOOK.md` section 1 ("the temporal grid minutes per rung" understates it: G-ORD is the cost).

### M5. The G-L (ii) refusal's consequence is not executed by the driver and is absent from the runbook

`SPEC.md` 3.7.4 (line 919) says that on `GL_SHOT_NOISE` "the driver then re-runs the split stage with
`feature_drop = ("cov", "std", "peak2med")` and reports both"; `gates_comparison.py:34`
declares `GL_FEATURE_DROP` and writes it into `params`, but nothing in `run_moves.py` reads
`gl.csv` or re-runs `models splits --feature-drop cov,std,peak2med`, and `RUNBOOK.md` move 7 does
not tell the author to. The refusal itself is written (`gl.csv`; Table 7's `G-L` column), so
nothing is silently absorbed, but CR 2.2 item 24 (ii) says the level-normalized comparison "is
refused until `cov` and its relatives are dropped", and Table 7 prints the refused comparison's
numbers beside the refusal text (`tables.py:423-433` only builds the text).

Fix (builder 3): after `gl` at move 7 (and at the move-12 re-run of B2), add a conditional
command that, when `gates/gl.csv` has `part = ii, verdict = refused: shot noise explains CV`,
runs `models splits --rung apf --all-splits --raw-and-norm --feature-drop cov,std,peak2med` under
a `base_dir` such as `gates/splits_dropped/`, and have `tables.py` print the dropped-set scores
in a second row (or replace the score cells with the refusal string, as al-Farabi 2.5 does for
G-C) until that re-run exists. Alternatively, state in the runbook that the re-run is the
author's manual step and that Table 7's numbers for the rung are not citable while the G-L (ii)
cell reads refused.

### M6. `gdim` ignores the driver's `--null-perm`

`run_moves.py:204` schedules `gates_comparison gdim` with no `--null-perm`, so
`gate_gdim` (`gates_comparison.py:255`) runs the `combined (matched)` LOKO split with the module
default of 500 permutations even on a smoke run at `--null-perm 20`, and with a different count
from every other Table 7 row of that run. Harmless in the real run (500 everywhere); slow and
inconsistent in the smoke run: on run 2 (12 kernel cells, `--null-perm 20`) `gdim` ran for
23 min 28 s on the 500-permutation null of the matched split, each permutation refitting the
per-fold importance reduction and the forest, while every other split of the run used 20.

Fix: append `"--null-perm", o.null_perm` to the `gdim` args at `run_moves.py:204`.

### M7. `select` writes G-C's not-applicable verdict into Table 5's `refusal` column as though it were a refusal

`gates_temporal.py:606`: `"refusal": gc_v if (gc_v and V.is_refusal(gc_v)) else ""`.
`verdicts.py:122-134` (`is_refusal`, line 132) returns true for every `not applicable:` string, so
`GC_ALIASED_BY_DESIGN` (`not applicable: pulse aliased by design ...`, al-Kindi review item 2:
"the rung is neither passed nor voided") and a `not run:` G-C (as on the synthetic run:
`selection.json["content"]["refusal"] = not run: no admissible cell for gibbs,histogram`) land in
the `refusal` column of every Table 5 row of the rung and in `selection.json`'s `refusal`. Table
7 handles it correctly (`rung_override` masks on `GC_DISCONNECTED` only).

Fix (builder 2): at `gates_temporal.py:606` write `gc_v` into `refusal` only when
`gc_v == V.GC_DISCONNECTED`; the `gc_verdict` column (same line) already carries every G-C
verdict for Table 5.

### M8. The roll-up treats "every kernel not applicable for G1" as a failed gate, unlike G2

`gates_temporal.py:595-597`: `applicable = {"G1": g1, "G4": g4}` always, and G2 is dropped from
the rule only when its roll-up is `not applicable`. `_rollup` (`gates_temporal.py:533-551`)
returns `not applicable: no applicable kernel` for G1 when every kernel is `trend present` (or
relabelled IDLE), and that string is not `pass`, so `_all` is false and every point falls to
`best-feasible`. Under al-Farabi review 2.1 a not-applicable kernel must not decide the point;
when no kernel is applicable the gate has no vote, as the code already does for G2. Rare on this
dataset (every kernel trending), but it is a reading the builder made rather than a declared
rule.

Fix: build `applicable` from `{g: v for g, v in (("G1", g1), ("G2", g2), ("G4", g4)) if not
v.startswith("not applicable")}` and record `n_gates_not_applicable` in `table5_grid.csv`.

### M9. The runbook's timing claims are not borne out on this machine

`RUNBOOK.md:71` ("finishes in minutes with every table present") and `RUNBOOK.md:102`
("the temporal grid minutes per rung"). Measured: the 14-cell corpus (12 kernel cells, 2 idle,
120 pairs) took 4 min 04 s end to end when only 5 cells passed C1 (run 1), and G-ORD alone took
7 min 47 s per rung when all 14 passed (run 2; see the record below). The 104-cell smoke corpus
of section 0b would run G-ORD on the 24 cells that survive C1 (gemm, fem_assembly, bnb_tsp; see
"For the author" item 1) for about an hour, and the `--null-perm 20` splits on top.

Fix: state the measured numbers (G-ORD dominates; about 8 min per rung per dozen cells at 300
trees, single process until M4 is done) and either lower `--n-estimators` for the smoke run
(`gates_temporal gord --n-estimators` exists; the driver does not pass it) or add
`--n-estimators` to the driver for the smoke path with the value recorded in `params`.

### M10. No test runs the driver across the three builders' modules

`tests/test_driver.py` runs the driver with `--only-modules` (builder 3's own modules; lines 128,
145, 153, 160, 182) and mocks the others; `tests/test_gates_chain.py` calls builder 2's functions
directly, never `run_moves`. The five BLOCKING findings all sit on the path no test walks
(the driver's move table feeding builder 2's CLIs and builder 3's readers). The runbook's smoke
run (section 0b) is that test, and it is manual.

Fix: add `tests/test_driver_chain.py` that writes builder 1's corpus with 2 or 3 reps and 2 idle
cells at 60 pairs, fills the admissibility record and a pass table, runs `run_moves run --moves
0-13 --null-perm 5 --n-jobs 2 --c1-activity-min 0.0005` (after M1) with `--n-estimators 10`
(after M9), and asserts: exit 0; Table 6's raw and norm columns both numeric on the `all` row
(B1); `gl.csv` has a part (i) row with a grid id for every rung (B2); `gc.csv` has a `combined`
`rep = all` row (B3); `figures.json` records five floor quantiles for `j_hist` (B4); Table 8's
clusters column is numeric (B5); `gf_check` `pass`. Mark it slow; it is the one test that would have caught
this cycle's blocking findings.

### M11. Three gate-decision helpers carry no citation in their own docstring

Rule 4 asks every gate to cite its definition in a docstring. The enclosing gate functions all
do (checked programmatically over 47 functions), but the helper that makes the decision does not
in three places: `gates_temporal.py:147` (`g1_kernel`, the G1 per-kernel rule: CR 2.1 item 3;
P2 Sec. V 5.1), `gates_temporal.py:267` (`g4_pass`, no docstring; the copy comment reads
`# plan03_aggregate.py: g4_pass = ...` rather than the SPEC's `# copied from <file>:<function>,
<date>` form: CR 2.1 item 6), and `variance.py:23` (`variance_levels`, the L0/L2/L3 arithmetic:
CR 2.3 item 35). `gates_calibration.py:440` (`run_alias`) cites nothing but only assembles rows.

Fix: one line each, the citation strings the modules already hold (`CIT_GRID`, `CIT_GV`,
`CIT_ALIAS`).

### M12. Two documentation mismatches inside the toolkit

(a) `gates_comparison.py:262` docstring says the matched split is written under
`gates/splits/combined_matched/<grid_id>/...`; the code (`gates_comparison.py:284`,
`base_dir="splits_matched"`) writes `gates/splits_matched/combined/<grid_id>/...`. Builder 3
reads both. Fix the docstring.
(b) `SPEC.md` 3.1.3 still points the on-disk guarantee at "3.5.6" (al-Farabi 2.2 asked for
"3.5.7"); the code follows 3.5.7. Editorial.

---

## Checks that passed (so the author knows what was looked at)

- The whole test suite: 154 passed, 1 skipped (the `zstandard` module path, absent here), 6 min
  08 s under pytest. Verbatim output in the record below.
- Every gate function's docstring cites its definition (P2 Sec. V item and CR 2.x item; the
  review item where a review changed it), with the three helper exceptions of M11. Checked:
  `gate_preconditions`, `failed_verdict`, `gate_gk0`, `gate_gf`, `gate_gp`, `gate_gc`,
  `alias_falsifier`, `g1_cell`, `g2_kernel`, `g3_cell`/`gate_g3`, `gate_grid` (G4, G5),
  `gate_gord`, `select`, `gate_gj`, `gate_gdec`, `gate_gl`, `gate_gn`, `gate_gx`, `gate_gdim`,
  `gm_compare`/`gate_gm`, `gate_gv`, `b1_g1_verdict`, `quarantine_l1`, `majority_baseline`,
  `run_clustering`, `make_forest`, `make_l1`, the four `nulls.py` functions, the three fold
  functions.
- Threshold values against the definitions: C1 0.02 (CR 2.1 item 1), C2 8 pairs, C3 informational;
  G-K0 tail 0.80 and the idle 95th percentile (item 20); G-F envelope rule and the
  leave-one-rep-out impossibility resolved as declared (item 21; SPEC 8.15); G-C two-to-one
  recorded with detection at 1.5 and the J dip at 0.75 within one snapshot, in every rep, the
  content orderings `mean_abs` gibbs < histogram < gemm and `l0` histogram < gibbs < gemm
  (item 22; al-Kindi 1, 2); G-P 4 dt / 2 dt, five passes, three snapshots, the four verdicts plus
  "rhythm under-sampled" and "pass aliased" (item 23); G1 z 1.0, floor 0.80, surrogate p05,
  "trend present" (item 3); G2 coverage floor 2.0 at both dt, above-Nyquist, undetermined (item
  4); G3 as a flag off the decision with the order-shuffle null (item 5; al-Kindi 3); G4 (item 6);
  G5 reported (item 7); Delta-5 stated not applied (item 8); B1-G1 500 permutations, strict
  exceedance, rank, `near_unfalsifiable` (item 9); B1-G3 one-feature model, at most one unit
  (item 10); B1-G6 at the unit (item 11); G-ORD spread p95 - p05 (item 27); G-J three times the
  floor's median K, no floor subtraction (item 31; P2 Sec. 6 item 6); G-DEC (a) to (e) with the
  three named refusals (item 34); G-L (i) strict exceedance, (ii) r2 > 0.5 (item 24); G-N 3/2/1/0
  (item 25); G-X leak plus confound (item 26; K2 4.3); G-DIM (item 32); G-M spread and the
  (6, 0) / (7, 1) sign rule (item 33); G-V (item 35).
- Verdict strings: `verdicts.py` holds every string of SPEC 3.0 verbatim plus the three review
  additions, each attributed.
- Refusals written: every gate writes its refusal string into its CSV and a `params` block with
  `inputs_sha256`; `excluded_rows.csv` for `near_unfalsifiable`; `excluded_cells_pair_rungs` for
  a refused failed count; `grid_complete.json` and the driver's refusal of the split stage.
- Interfaces: `extract.csv` has the 58 columns of SPEC 2.2 in order; `sidecar.json` has every
  key of SPEC 2.3 (plus `rep_source`, `source_sha256`, `params`, `citation`); `cells.csv` has
  the 12 columns of SPEC 2.7; `parse_cell_path` reads server-shaped paths (`kernel/
  kernel_gemm_v2/--dim_1024_..._--seed_1548_..._ab12cd34/rep001__sandbox_deepdive_01c1` gives
  gemm, seed 1548, campaign `01c1`; `dwarfs1_resume` gives `dwarfs1`; a `sleep` label gives
  role idle); builder 2 reads them by those names; builder 3 reads builder 2's `gc.csv`
  (`rep = all`), `gf.csv` (`part = i`), `gk0.csv`, `gp.csv`, `gj.json`, `selection.json`,
  `table5_grid.csv` (with `coverage_pairs`, `G2_pairs`, `n_kernels_na_*`), `gl/gn/gx/gdim/gm.csv`,
  `gv*.csv`, the split directories, as written. The interface breaks are B1 (the raw split
  directory), B4 (`gj.json`'s floor quantiles) and B5 (`clustering.json`'s labels).
- The skeleton (`report/paper2_skeleton.tex` from the run and the standalone
  `apf_paper/p2_skeleton.tex`, 432 lines, 135 comments): every non-comment line is a LaTeX
  command, an environment line, or a tabular row whose cells are P2 Sec. 4's table cells;
  `\title{}`, `\author{}`, every `\caption{}` empty; no sentence outside comments; the comment
  bullets are P2 Sec. 3's substance bullets in abbreviated form. The two files differ only in
  Table 3's nbody pass-period cell (the run merges `inputs/pass_table.csv`). No TeX compiler on
  this machine, so compilation is unverified (builder 3 said the same).

---

## For the author

1. **C1's threshold decides which kernels exist for the paper.** With `apf_max >= 0.02`,
   `K_max` must reach 5,243 pages in at least one snapshot. From P2 Table 3's footprints, floyd,
   histogram and nbody (2,048 pages, `apf` about 0.0078), gibbs (256), the lexer, and probably
   spmm and rmat_gen fail C1 unless a pass-boundary spike lifts `K_max` above 5,243; the
   level-matched triple of the claim would then be excluded from every table by a threshold that
   came from the sandbox family's apf_queue re-map, not from the kernels. It also removes the
   lexer at move 2, before G-K0 (move 4) can relabel it `IDLE, measured`, so the lexer would be
   absent from every table instead of "reported as at floor, never as an encoding failure" (P2
   Sec. 2; CR 2.2 item 20). The re-map the council asked for (CR 2.1 item 1) was for the idle
   cell only. Decide before the data: keep 0.02, lower
   it (`--c1-activity-min`), or express it against the idle floor (for example `K_max` above the
   idle band's 95th percentile), and record the choice as a change to a pre-registered gate.
2. **B1-G1's refusal hides the measured blind spot.** CR 2.1 item 9 makes a LOKO score that does
   not strictly exceed the unit-level null `near_unfalsifiable`, "excluded from every table", and
   al-Farabi 2.7 prints the string in every score cell of that split. P2 Sec. VII expects Table 6
   to show "LOKO inside the null" as the measured blind spot; under the definitions that cell will
   read `near_unfalsifiable`, and the number stays in `scores.json`. Say whether Table 6's LOKO
   norm column prints the score with the verdict beside it, or the string.
3. **The selection rule uses G2 in seconds, not in pairs.** `select` gates on `G2` (the roll-up of
   the 0.500 s and 0.644 s columns, CR 2.1 item 4) and reports `G2_pairs` (al-Kindi review 7)
   without gating on it. On the real cells the 0.644 column equals the pair-unit verdict; the
   0.500 column can differ, and a flip is `undetermined by the interval calibration`, which is not
   a pass. On the synthetic corpus (120 pairs for 600 s, `dt_est` 5 s) the two disagree at every W,
   which is why run 2 below selected `best-feasible`. Confirm this is the intended rule.
4. **G-DEC's idle clause under `idle_slope_rule = "any"` refuses by chance about one time in
   five.** Rule (e) (CR 2.2 item 34) is executed per idle cell as "whole-cell slope of `l0_q50_per`
   beyond the 95th percentile of its own phase-randomized surrogates, same sign as the decay", and
   `"any"` (the definition's words) lets one idle cell void the exhibit. A cell with no slope
   fails that test about 5 percent of the time, half of it with the decay's sign; over the eight
   idle cells to be captured the chance that at least one fires is about 1 - 0.975^8, about 18
   percent, so a real floyd decay would read `no decay beyond floor or host` about one run in
   five with no host effect present. Run 2 below shows it on two synthetic idle cells (one of
   them "slope significant" by chance; every floyd rep then reads `no decay beyond floor or
   host`). The alternative `"min_reps"` (at least seven idle cells with a same-sign significant
   slope) is exposed; choose one before the data and record it.
5. **The other choices the builders exposed** and that this check found consequential on the
   synthetic runs: `rollup_kernel_refusals = "not_applicable"` (al-Farabi 2.1);
   `part1_consequence = "void"` for G-F (i) (al-Farabi section 3 item 2); `g3_min_cells = 7` (a
   count, so a kernel with fewer than 7 cells can never flag); `jump_detect_ratio = 1.5`,
   `jump_reference = "cell_median_K"` and the aliased-regime band for G-C; `pass_frac = 0.5` and
   `idle_slope_rule = "any"` for G-DEC; `gl2_level = "kernel"`; `gm_n_seeds = 5`;
   `dim_match_method = "train_importance"`; `table8_rung = "combined"`; `fused_plane_mask = "K"`.
   The full list with defaults is SPEC section 8 and the three BUILD reports' "For the author"
   sections; nothing in the code moves a threshold away from those.

---

## Record

### (1) The test suite, verbatim

Command, from `plan11_encoding_ladder/`: `python3 -m pytest -q tests`

```
.............................s.......................................... [ 46%]
........................................................................ [ 92%]
...........                                                              [100%]
154 passed, 1 skipped in 368.60s (0:06:08)
```

The skip, from a second run with `-rs`: `SKIPPED [1] tests/test_extract.py:515: zstandard module
not installed` (154 passed, 1 skipped in 365.80s).

The runbook's alternative, `python3 -m unittest discover -s tests -p "test_*.py"`:
`Ran 94 tests in 82.452s`, `OK (skipped=1)` (see M2: 61 gate tests are not discovered).

### (2) The driver end to end on a synthetic corpus

Corpus: builder 1's presets (`synth.corpus_specs`) filtered to four kernels, gemm, floyd, gibbs
and histogram (the pulse kernel, the decay exhibit, the control, and the content-ordering
triple), 3 reps each (seeds 42, 1000 + i, 2000 + i), plus 2 idle cells (`kernel_sleep_v2`,
label `idle`), 120 pairs per cell, compressed with the `zstd` binary: 14 cells, written in 19.7 s.

Run 1, the runbook's standard command (`run_moves run --out <out> --root <root> --null-perm 20
--n-jobs 4 --assume-failed-zero --assume-reason "checker smoke run"`): exit 0, 4 min 04 s, all
76 ledger entries `done`, `gf_check` `pass`. It wrote `cells.csv` (14 rows, all `ok`; rep 0 =
seed 42), 14 extracts and sidecars (58 columns, `header_ncols 66`, `n_pairs 120`, `dt_est_s 5.0`),
`gates/preconditions.*` (C1 `fail` on floyd, gibbs, histogram; C1 not applicable on the idle
cells; `n_all_hard_pass 5` of 14; `failed_source` `declared zero: checker smoke run`), the four
input templates, `gp.csv` (every kernel `undeclared`), `gc.csv` (apf, persist, wapf `pass` on
gemm in 3 of 3 reps with `stat_a` 1.976 to 1.986 and `j_at_event` 0.48 to 0.49; content `not run:
no admissible cell for gibbs,histogram`), `gk0.csv` and `gf.csv` (`not run: no admissible idle
cell; admissibility record missing`, the template being unfilled), 13 grid points times 5 rungs
(`temporal_per_kernel.csv`, `g1_surrogates.npz`, 4 `gord.json` per rung), 26 feature files per
rung (13 for combined), `g3_flags.csv`, `alias.csv`, `table5_long/grid.csv`, `selection.json`
(apf, wapf, persist W8_H4; content, combined W16_H8; every `passes_acceptance` true; G2 `not
applicable: no kernel with a declared pass period`), `gj.*` and `gj_mask/` (5 cells, `floor
unmeasured`), `gdec.csv` (`not run: no admissible floyd cell`), the split directories for the
five rungs (`b1_g1 = not run: 20 permutations < 500` by design; LOKO accuracy `None` since one
kernel survived C1), `gl/gn/gx/gdim/gm/gv/clustering`, `gm_runs/seed0..4`, `splits_matched`,
`report/tables/` (table5, table5_g3, table6, table7, table8, tablegv, table4_status,
preconditions, table_wapf_over_apf, each as csv/md/tex/json), eight figures as PDF and PNG,
`paper2_skeleton.tex`, `manifest.json`, `driver_state.json`. Failures: none at the process level.
What is wrong in the outputs is B1 (Table 6's raw columns missing), B2 (G-L for four rungs), B3
(G-C for combined), B5 (Table 8's clusters column `not run`), M1 (9 kernel cells excluded by C1),
M7 (the not-run G-C string in the content rung's refusal column).

Run 2, the same corpus with C1 lowered by hand (`gates_precondition preconditions
--c1-activity-min 0.001`, all 14 cells `all_hard_pass`), `inputs/idle_admissibility.json` filled,
and a pass table declaring gemm 5 and floyd 12 passes per 600 s with histogram `inferred: 6147`:
moves 0 to 1 through the driver, move 2 by hand, then `run_moves run --moves 3-13 --null-perm 20
--n-jobs 4`. Results of the finished moves: G-C `pass` on all four rungs (content orderings
hold in 3 of 3 reps); G-K0 every kernel `above floor` (idle band edge 150); G-F part (i)
`inseparable at floor` on all five rungs at W8_H4, part (ii) `pass` everywhere except histogram
on the content rung, `at floor in this lead` (its counter content gives the same `r_l0_q50_per` as
the idle model, a synthetic artifact); G-P floyd and gemm `resolvable`/`admitted`, histogram
`aliased by design at this size (INFERRED)`, gibbs `undeclared`; G3 present per cell on floyd
(period 20, a rahmonic of 10) and gemm (24), absent on gibbs, every kernel flag absent because
`g3_min_cells = 7` exceeds the 3 reps; G-ORD `resolution` at W 8 to 32 and `order-blind` at 64;
APF selection `W8_H4`, `selected: best-feasible`, `2 of 3` (G2 fails in seconds at every W with
`dt_est` 5 s, see "For the author" item 3); the alias falsifier `not run: fewer than three cells
or no spread` (every cell has the same `dt_est`). Exit 0 after 1 h 29 min 58 s wall time, 70
ledger entries `done` (moves 3 to 13), `gf_check` `pass` with `inseparable at floor` on every
Table 7 row. Where the time went: G-ORD 7 min 47 s (apf), 7 min 40 s (persist), 14 min 06 s
(content), 10 min 54 s (combined); the split stages 2 to 6 min per rung at `--null-perm 20`;
`gdim` 23 min on its own 500-permutation null (M6). Every rung selected `best-feasible` (G2 in
seconds fails at every W on this synthetic `dt_est`); LOKO/archetype accuracies 0.583 (apf,
after B1-G3 quarantined `apf.k_over_med.mean` and `apf.k_over_med.peak2med`, feature count 6),
0.417 (wapf, persist), 0.500 (content, combined), majority 0.75, every B1-G1 `not run: 20
permutations < 500` by design except `combined (matched)`, whose 500-permutation null made it
`near_unfalsifiable` and printed the string in its score cells (al-Farabi 2.7 works). G-DEC:
floyd 10 passes of 10 snapshots found by the K jump, the fall present at phase 0 with
`l0_rel_drop` 0.516 against `k_rel_drop` 0.475 (rule (c) passes), but one of the two idle cells
carries a "slope significant" whole-cell `l0` slope, so every rep reads `no decay beyond floor or
host` and the kernel row `no decay` (3 reps below `min_reps = 7`; see "For the author" item 4);
gibbs `no slope (0 of 3 reps with a slope)`. G-J: gemm, floyd, histogram `interpretable`, gibbs
`floor overlap` (K 406 against 3 x 150). G-V `estimable` on every rung. Clustering k = 2, ARI
0.195, not above its null. Table 8: WORKING-SET row `2 (gemm, floyd)` under WORKING-SET and
`1 (gibbs)` under SCATTER; SCATTER row `1 (histogram)` under WORKING-SET; the clusters column
`not run` (B5). Figures: all eight, the fused plane with 357 masked points (the three gibbs cells),
`j_hist` with `floor_quantiles = None` despite the two idle cells (B4), the piano roll from
525,161 streamed rows of the first gemm cell. `manifest.json`: 461 files hashed. The outputs
confirm B1 to B5 on a corpus where every cell is admissible.

### (3) to (6)

Covered above: (3) the gate functions against their definitions (BLOCKING B2, B3 for the driver
order; no threshold or verdict-string deviation found in the gate code itself; M7, M8 for the
roll-up; M11 for citations); (4) the interfaces (B1, B4, B5; the rest agree with SPEC); (5) the
runbook commands (M2, M3, M9; everything else runs as written); (6) the skeleton (no body prose).
