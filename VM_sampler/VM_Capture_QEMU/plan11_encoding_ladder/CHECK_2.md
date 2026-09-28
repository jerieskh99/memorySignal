BLOCKING FINDINGS: 1 (B1); MINOR: 17 (M1 to M17). The test suite passes (157 passed, 1 skipped), the driver runs end to end with exit 0 on the synthetic corpus, no gate threshold or verdict string deviates from its definition, and the skeleton carries no body prose. The one blocking finding is a cross-builder disagreement about which score is the rung's score once B1-G3 has quarantined a feature: the comparison gates (G-L, G-DIM, G-M) judge the pre-quarantine score, and Table 7 prints the post-quarantine re-run beside their verdicts.

# CHECK_2.md: checker's report on the paper 2 analysis toolkit, cycle 2 (2026-09-16)

Rules kept: no server, no path under a server mount, no remote command; the sandbox family's sources
and workload names were not read, grepped or named; no paper prose was written; no file outside
`plan11_encoding_ladder/` was edited and the only file written inside it is this report; nothing was
committed; `P2_STRUCTURE.md` was not touched. Every test and every driver run below used synthetic
data generated on this machine with builder 1's generator (`synth.py`). Environment: Python 3.10.12,
numpy 2.2.6, scikit-learn 1.7.2, scipy 1.15.3, matplotlib 3.10.7, joblib 1.5.2, pytest 9.1.1, the
`zstd` binary on PATH (1.5.7), the `zstandard` module absent, no TeX compiler.

Findings are listed most severe first under BLOCKING and MINOR. Each carries the file and line,
what is wrong, and the exact fix. A section "For the author" lists the choices the definitions leave
open that this check found consequential, and the record of every run follows. The cycle-1 fixes
(B1 to B5 of `CHECK_1.md`) were re-verified on the driver runs and hold; they are listed under
"Checks that passed".

---

## BLOCKING

### B1. After a B1-G3 quarantine, G-L, G-DIM and G-M judge the full-model score while Table 7 prints the re-run's score beside their verdicts

`gates_comparison.py:42-44` (`_scores`) returns `scores.json` as written, and every comparison gate
reads its top-level keys: `gates_comparison.py:87-93` (`gate_gl`, part (i): `sc["accuracy"] >
sc["null_p95"]`), `gates_comparison.py:271-277` (`gate_gdim`: `sc.get("feature_count")`,
`sc["accuracy"]` to choose the strongest single rung and its `d*`), `gates_comparison.py:346-351`
(`gate_gm`: `sc[rung]["accuracy"]` and `recall_per_kernel`). Those keys are the full model's, fitted
on every feature. `models.py:436-445` (`run_split_stage`) runs B1-G3 afterwards and, when a feature
is quarantined, stores the re-run's `accuracy`, `null_p95`, `b1_g1`, `b1_g1_rank`,
`recall_per_kernel` and `feature_count` under `scores.json["with_quarantine"]`. Builder 3 reads
the other way: `_report_common.py:593-606` (`effective_scores`) merges `with_quarantine` over the
top-level keys, and `tables.py:535, 545-549` (Table 7) and `tables.py:344` (Table 6) print the
re-run's accuracy, null p95, rank and feature count. So the `G-L`, `G-DIM` and `G-M vs APF` cells of
a Table 7 row are verdicts on numbers that are not the numbers printed in that row.

The definitions: CR 2.1 item 10 (B1-G3 restated) says "quarantine, name, re-run without it; run
per split and per label space"; SPEC 3.7.2 fixes the interface with "the split is re-run without
the quarantined features and both scores are kept ... Table rows use the re-run". CR 2.2 item 24
(G-L (i), in al-Kindi's minimum set) says "the level-normalized LOKO archetype score must exceed
the kernel-level shuffle null's 95th percentile" and CR 2.2 item 33 (G-M) compares "the score".
Once B1-G3 has named a feature as the label and the chain has re-run without it, the rung's score
is the re-run; judging G-L on the full-model score lets the quarantined feature decide the
level-blindness verdict that B1-G3 exists to protect.

The same split also happens inside builder 2's own artifact: `models.py:449-457` writes
`predictions.csv` from `preds`, the full model's per-unit predictions, and the re-run's `preds2`
(`models.py:442`) is discarded after its scores are taken. Table 8's LOKO assignments
(`tables.py:573-590`, `_loko_assignments`) and G-X's held-out campaign therefore come from the full
model while Table 7's row for the same split prints the re-run.

Verified on run 2 (record (2) below): on APF at `W8_H4` the quarantine fired in the LOKO archetype
split (`apf.k_over_med.mean` and `apf.k_over_med.peak2med` quarantined): the full model scored 0.500
against a null p95 of 0.500 (rank 2 of 20) and the re-run 0.583 against 0.583 (rank 11 of 20, 6
features); on the raw LOKO split the full model scored 0.083 and the re-run 0.583 after five
features were quarantined. `gates/gl.csv` then carries `apf, i, W8_H4, score_norm 0.5, null_p95 0.5`
while `report/tables/table6.csv` prints LOKO norm 0.583 on the `all` row and
`report/tables/table7.csv` prints accuracy 0.583 with feature count 6 on the APF LOKO row (and, for
persist, `gl.csv` 0.25 against Table 7's 0.417; wapf 0.5 against 0.417). On the real corpus the same
feature set (`mean`, `median`, `max`, `p95` of a level-normalized count) is the one most likely to
be quarantined, and the two numbers differ whenever the re-run moves the score across the null.

Fix (builder 2, five lines plus one file): in `gates_comparison.py` give `_scores` the merge that
`_report_common.effective_scores` does:

```python
def _scores(out, rung, gid, split, labelspace, base="splits"):
    p = M.split_dir(out, rung, gid, split, labelspace, base) / "scores.json"
    if not p.is_file():
        return None
    sc = S.read_json(p)
    wq = sc.get("with_quarantine")
    if isinstance(wq, dict) and wq.get("quarantined_features"):
        sc = {**sc, **{k: v for k, v in wq.items() if k != "quarantined_features"}, "quarantined_features": wq["quarantined_features"]}
    return sc
```

and write `"score_source": "with_quarantine when a feature is quarantined (SPEC 3.7.2), else full model"`
into the `params` of `gl.csv`, `gdim.csv` and `gm.csv`. In `models.run_split_stage`, when `quar` is
non-empty, write the re-run's per-unit predictions as `predictions.csv` (the file the tables read)
with the full model's under a new column `y_pred_full`, or as a second file
`predictions_with_quarantine.csv` that `tables._loko_assignments` reads first; either way the
choice is recorded in `params`. Add to `tests/test_gates_comparison.py` one case with a level-only
corpus in which the full model exceeds the null and the re-run does not, and assert that `gl.csv`
part (i) reads `level only` and that its `score_norm` equals the re-run's accuracy. If the author
prefers the other reading (the gates judge the full model and the tables print it), the change is
in `_report_common.effective_scores` instead; either way the two builders must read the same key,
and the choice is listed under "For the author" item 2.

---

## MINOR

### M1. C1's default threshold is still not reachable from the driver or the runbook, so the runbook's smoke run cannot exercise G-C content, G-DEC, G-ORD or any LOKO fold (CHECK_1 M1, refused by the fixer as more than one line; still open)

`gates_precondition.py:476-477` exposes `--c1-activity-min` (default `C1_ACTIVITY_MIN = 0.02`,
`gates_precondition.py:28`, the apf_queue re-map of CR 2.1 item 1); `run_moves.py:469-491`
(`_add_run_args`) has no such flag and `run_moves.py:118-124` does not pass it, and `RUNBOOK.md`
never names it. Run 1 below (the runbook's standard command on 4 kernels x 3 reps + 2 idle cells):
floyd, gibbs and histogram read `C1 = fail` (`apf_max` 0.0165, 0.0016, 0.0088) and
`all_hard_pass = false`, so 5 of 14 cells entered the chain, G-C content read `not run: no
admissible cell for gibbs,histogram`, G-DEC `not run: no admissible floyd cell`, every G-ORD
`not run: fewer than two archetypes with windows`, every LOKO accuracy `None`, and Table 6's LOKO
columns `--`. Not a deviation from the definition (CR 2.1 item 1 re-maps C1 for the idle cell only);
the decision is the author's ("For the author" item 1). Fix (builder 3): add `--c1-activity-min`
(float, default 0.02) to `_add_run_args`, append it to the `preconditions` args at
`run_moves.py:118-124`, record it in the ledger `params`, name it in `RUNBOOK.md` move 2 with the
sentence that the default excludes every kernel whose `K_max` is below 5,243 pages, and add
`gates/preconditions.csv` to the `inputs` list of every command from move 3 onward so a re-run of
move 2 with another threshold makes the downstream outputs `stale` instead of leaving them skipped.

### M2. The floyd decay figure is a placeholder whenever G-DEC's verdict carries the `(floor unmeasured)` suffix

`figures.py:471` (`fig_floyd_decay`): `if verdict != GDEC_DECAY: return _placeholder(...)`.
`gates_readings.py:332-333, 354-355` append `" (floor unmeasured)"` to every exhibit verdict when no
idle cell exists (CR 2.2 item 34 (e); SPEC 3.6.2), so a `decay (floor unmeasured)` reading, which is
a decay with its idle clause labelled not run, draws the placeholder instead of the exhibit. Until
the idle cells are captured this is the only decay verdict the real data can produce. Fix (one
line): compare the head, `verdict.split(" (")[0] != GDEC_DECAY`, and print the suffix in the title
(`figures.py:498`).

### M3. Table 7's `feature count` for `combined (matched)` prints the pre-reduction width, not the matched dimension

`models.py:417, 473` (`run_split_stage`) write `scores.json["feature_count"] = d_full` (the width
of `X` before the per-fold reduction) and, after a quarantine, `with_quarantine["feature_count"] =
len(keep)`; the dimension the forest saw (`d_target`, `models.py:142-149`) is returned per unit by
`fit_predict_units` as `d_used` and never written to a file. `tables.py:544` prints that count, so
on run 2 the `combined (matched)` LOKO row reads `feature count 40` (60 features, 20 quarantined) beside a
`G-DIM` cell that says `matched 8, train_importance`. CR 2.2 item 32 asks that every Table 7 row
state its feature count and that the matched row be at the strongest single rung's dimension.
Fix (builder 2, two lines): write `feature_count_used = min(d_target over folds)` into
`scores.json` (and into `with_quarantine`); (builder 3, one line) Table 7 prints it when present.

### M4. The roll-up counts "no applicable kernel for G1" as a failed gate, so every point falls to best-feasible in that case (CHECK_1 M8, still open)

`gates_temporal.py:597-600`: `applicable = {"G1": g1, "G4": g4}` always includes G1, and G2 is
dropped only when it is `not applicable`. `_rollup` (`gates_temporal.py:542-543`) returns
`not applicable: no applicable kernel` for G1 when every kernel is `TREND_PRESENT` or relabelled
IDLE, which is not `pass`, so `_all` is false and the selection reads `best-feasible` for a reason
unrelated to the grid point. Under al-Farabi review 2.1 a not-applicable kernel must not decide the
point; when no kernel is applicable the gate has no vote, as the code already does for G2. Fix: build
`applicable` from `{g: v for g, v in (("G1", g1), ("G2", g2), ("G4", g4)) if not
v.startswith("not applicable")}` and record `n_gates_not_applicable` in `table5_grid.csv`.

### M5. Three runbook lines still do not run as written (CHECK_1 M3, still open)

`RUNBOOK.md:304`, `322`, `337`: `python3 -m plan11_encoding_ladder.gates_temporal grid/g3/gord/select
--out <out> --rung content   (four commands, as above)`. Verified: argparse rejects
`grid/g3/gord/select` (`invalid choice`). Every other runbook command was parsed against its CLI
(record (5) below) and runs as written. Fix: expand each line to the four commands (as moves 6 and 9
already do), or replace them with `python3 -m plan11_encoding_ladder.run_moves plan --out <out>
--moves 10` and say the plan prints the exact lines.

### M6. The runbook's `unittest` alternative runs 96 of the 158 tests and reports OK (CHECK_1 M2, still open)

`RUNBOOK.md:26` and `:55`, `requirements.txt:7`. Verified: `python3 -m unittest discover -s tests -p
"test_*.py"` reports `Ran 96 tests ... OK (skipped=1)` while pytest collects 158 (157 passed, 1
skipped); builder 2's twelve `tests/test_gates_*.py` modules are pytest-style functions that
`unittest` does not discover. An author without pytest on the server would read OK and believe the
gate tests passed. Fix: replace the alternative with "`pytest` is required for the gate tests (62 of
158); `python3 -m pip install --user pytest` when absent", in both places and in `requirements.txt`.

### M7. `gates_temporal gord` and `grid` accept `--n-jobs` and ignore it; the runbook's cost claims are not borne out (CHECK_1 M4 and M9, still open)

`gates_temporal.py:426-427` (`gate_gord`) and `gates_temporal.py:290-291` (`gate_grid`) take
`n_jobs` and never use it; `run_moves.py:71-72` lists both in `NJOBS_COMMANDS`. Measured on run 2
(12 kernel cells, 120 pairs, `--n-jobs 4`): G-ORD apf 424 s, persist 363 s, content 772 s, wapf 463
s, combined 718 s, 2,740 s in all, single process. `RUNBOOK.md:71` ("finishes in minutes with every
table present") and `RUNBOOK.md:102` ("the temporal grid minutes per rung") understate it: at the
runbook's 104-cell smoke corpus G-ORD would run for the better part of an hour per rung
(extrapolating from 12 cells). Fix (builder 2): parallelize
the `n_order_perm` and `null_perm` loops of `gate_gord` over `n_jobs` with `joblib.Parallel` as
`_gf_part1` (`gates_precondition.py:347-349`) does, or drop the flag; and (builder 3) state the
measured cost in `RUNBOOK.md` section 1 and pass `--n-estimators` through the driver for the smoke
path (the `gord`, `splits`, `gf`, `gx`, `gdim`, `gm` CLIs all accept it; the driver passes it to
none).

### M8. The G-L (ii) refusal's consequence is not executed by the driver and is absent from the runbook (CHECK_1 M5, still open)

`SPEC.md` 3.7.4 says that on `GL_SHOT_NOISE` "the driver then re-runs the split stage with
`feature_drop = ("cov", "std", "peak2med")` and reports both"; `gates_comparison.py:30` declares
`GL_FEATURE_DROP` and writes it into `params`, but nothing in `run_moves.py` reads `gl.csv` or
schedules `models splits --feature-drop cov,std,peak2med`, and `RUNBOOK.md` move 7 does not tell
the author to. The refusal itself is written (`gl.csv`; Table 7's `G-L` column), so nothing is
silently absorbed, but Table 7 prints the refused comparison's numbers beside the refusal text.
Fix: a conditional driver command after each `gl` run that, when `gates/gl.csv` has
`part = ii, verdict = refused: shot noise explains CV`, runs the APF split stage with
`--feature-drop cov,std,peak2med` under a `base_dir` such as `gates/splits_dropped/`, and a Table 7
row (or the refusal string in the score cells) until that re-run exists; or a runbook sentence that
the re-run is the author's manual step and the rung's numbers are not citable while refused.

### M9. The driver's `--seed-offset` does not reach `gdim`, and `--n-jobs` does not reach `gdim` or `gm`

`run_moves.py:68-70` (`RANDOM_COMMANDS`) omits `("gates_comparison", "gdim")` although `gate_gdim`
draws a `n_perm`-permutation null through `run_split_stage` (`gates_comparison.py:283-284`) and its
CLI accepts `--seed-offset` (`gates_comparison.py:372-373`); `NJOBS_COMMANDS` (`run_moves.py:71-72`)
omits `gdim` and `gm`, whose CLIs accept `--n-jobs`. SPEC 3.2 says the driver's offset "adds to all
four [seeds] and is recorded"; on a `--seed-offset N` run the matched combined row's null is drawn
at the unshifted seed. Fix: add `("gates_comparison", "gdim")` to `RANDOM_COMMANDS` and
`("gates_comparison", "gdim")`, `("gates_comparison", "gm")` to `NJOBS_COMMANDS`.

### M10. The idle cells' head drop is keyed two ways

`series.py:220-226` (`write_head_drop_template`) writes a row `idle`; `gates_precondition.py:419`
(G-F part (ii)) reads the idle cells' drop as `head_drop_for(hd, "idle")`, while `series.py:610`
(`build_features`), `gates_temporal.py:309, 383-384, 446`, `gates_readings.py:84` and
`gates_comparison.py:116` read it as `head_drop_for(hd, c["kernel"])`, where an idle cell's
`kernel` is its label-derived name (`sleep` on the synthetic corpus, whatever the idle capture's
test label yields on the real one). A non-zero `idle` row would apply in G-F (ii) only. Harmless at
the default 0. Fix (one line in `series.head_drop_for` or its callers): resolve the key as
`"idle" if role == "idle" else kernel`.

### M11. G-X writes a confound verdict on a corpus with one campaign label

`gates_comparison.py:222` computes `gx_confound` before the label count is known and
`gates_comparison.py:231` writes it beside `not applicable: one campaign label`, so run 2's
`gx.csv` and Table 6's `all` row read `not applicable: one campaign label; confound: partial`:
with one label every archetype's kernels sit in exactly one campaign by construction. Harmless on
the real corpus (three labels). Fix: when `len(labels) < 2`, write the not-applicable string into
`confound_verdict` as well.

### M12. Table 5's `refusal` cell duplicates a disconnected lead

`gates_temporal.py:608` writes `gc_v` into the grid row's `refusal` when it is `GC_DISCONNECTED`
(the CHECK_1 M7 fix), and `tables.py:183-184` appends `G-C: disconnected lead` to the same cell
again, so the row reads `disconnected lead; G-C: disconnected lead`. Cosmetic. Fix: in
`tables.py:183` skip the append when `gc` is already in `refusal_parts`.

### M13. Two Table 2 cells of the skeleton print their math escaped

`latex_skeleton.py:56, 58` write `one scalar $K_t / N$` and `one scalar $J(t)$`; `_cell`
(`latex_skeleton.py:262-267`) leaves a cell unescaped only when the whole cell is `$...$`, so
`apf_paper/p2_skeleton.tex:112` reads `one scalar \$K\_t / N\$` and `:114` `one scalar \$J(t)\$`,
which typeset as the literal characters. No prose; a rendering defect. Fix: escape around the math
span (split on `$` and escape the even segments), or write the two cells as text (`one scalar K_t / N`
as P2 Table 2 does, in backticks).

### M14. Three gate-decision helpers carry no citation

`gates_calibration.py:199` (`k_jump_events`, the G-C event detector: CR 2.2 item 22;
SPEC_review_al_kindi.md item 1), `gates_readings.py:157` (`dec_cell`, the per-cell G-DEC
arithmetic of rules (b) to (d): CR 2.2 item 34) and `models.py:102` (`aggregate_units`, the
unit-aggregation rule `cell_majority` with its tie-breaks: SPEC 4.2; section 8 item 22). The
enclosing gates cite, as in CHECK_1 M11; one line each with the strings the modules already hold
(`CIT_GC`, `CIT_GDEC`, `CITATION_SPLIT`).

### M15. G-X's split run records a meaningless majority baseline

`models.py:189-206` (`majority_baseline`) under `split = "loko"` scores the most populous
training archetype against `y_unit`; `gate_gx` (`gates_comparison.py:233-235`) runs the stage with
campaign labels, so `y_unit[c] == top` compares a campaign string with an archetype and the
`majority` written to `gates/gx_runs/.../scores.json` is always 0. `gx.csv` does not read it.
Fix: in `majority_baseline`, when `labelspace == "campaign"`, count the most populous campaign by
cell count (the LORO branch), or write `null`.

### M16. Table 6 prints `--` for kernels excluded at C1 rather than naming the exclusion

`tables.py:329, 365-374` build the kernel rows from every `status = ok` cell in `cells.csv`, so a
kernel excluded by `all_hard_pass` (run 1: floyd, gibbs, histogram) gets a row whose score cells
read `--` (no `recall_per_kernel` entry) with a `G-N status` copied from the surviving kernels'
`gn.csv`. SPEC section 7 asks that a cell which is not a number names what is missing. Fix: read
`gates/preconditions.json` `excluded_cells` and print `not run: excluded by C1-C8 (all_hard_pass
false)` in those rows.

### M17. No test runs the driver across the three builders' modules (CHECK_1 M10, still open)

`tests/test_driver.py` runs the driver with `--only-modules` (builder 3's own) and mocks the
others; `tests/test_gates_chain.py` calls builder 2's functions directly. B1 above sits on the
path no test walks (builder 2's `scores.json` read by builder 2's comparison gates and by builder
3's tables on one run). Fix: `tests/test_driver_chain.py` as CHECK_1 M10 describes, with the
assertion that `gl.csv` part (i)'s `score_norm` equals Table 7's `accuracy` on the same row.

---

## Checks that passed (so the author knows what was looked at)

- The whole test suite: 157 passed, 1 skipped (the `zstandard` module path, absent here), 4 min
  11 s under pytest. Verbatim output in record (1) below.
- The driver end to end, twice, exit 0 both times: run 1 (the runbook's standard command at the
  default C1) in 6 min 41 s with all 78 ledger entries `done` and `gf_check` `pass`; run 2 (every
  cell admissible, the admissibility record and a pass table filled) in 1 h 09 min 51 s with all 70
  ledger entries of moves 3 to 13 `done` and `gf_check` `pass`. Record (2).
- The cycle-1 fixes, re-verified on the runs: B1 (`gates/splits/apf/W8_H4/` holds five `__raw`
  directories beside five normalized ones, `params.normalized` false and true respectively, and
  Table 6 has numeric raw columns); B2 (`gates/gl.csv` has a part (i) row with a grid id for all
  five rungs after move 12); B3 (`gates/gc.csv` has the `combined, apf+wapf+persist+content, all`
  row and Table 7's `G-C` cell for `combined` carries it); B4 (`figures.json` `written.j_hist`
  records five floor quantiles from `gj.json`'s `idle_J` with an empty absent-reason); B5 (Table 8's
  `clusters (k = ...)` column prints `c<n>:<count>` from builder 2's `per_algo` layout); al-Farabi
  condition (1) (the `select` steps list the 13 grid CSVs, `gk0.csv` and `gc.csv` as inputs); CHECK_1 M6
  (`gdim` receives `--null-perm`); CHECK_1 M7 (Table 5's `refusal` carries only `GC_DISCONNECTED`); CHECK_1 M11 and
  CHECK_1 M12 (the citations and the docstring path).
- Every gate function's docstring cites the definition it implements (P2 Sec. V item and CR 2.x
  item, plus the review item where a review changed it), checked programmatically over every
  public function of builder 2's ten modules: `gate_preconditions`, `failed_verdict`,
  `c7_verdict`, `gate_gk0`, `tail_median_K`, `gate_gf`, `gp_verdict_T`, `gp_cell`, `gate_gp`,
  `gate_gc`, `alias_falsifier`, `separating_features`, `stationarity_per_window`, `trend_drift_sd`,
  `g1_cell`, `g1_kernel`, `g2_kernel`, `cepstrum`, `cepstral_peak`, `g3_cell`, `gate_g3`, `g4_pass`,
  `gate_grid`, `gate_gord`, `select`, `gate_gj`, `boundaries_of`, `whole_cell_slope_test`,
  `gate_gdec`, `gl_part2_regression`, `gate_gl`, `gn_status`, `gate_gn`, `gx_confound`, `gate_gx`,
  `gate_gdim`, `gm_compare`, `gate_gm`, `variance_levels`, `gate_gv`, `make_forest`, `make_l1`,
  `fit_predict_units`, `score_units`, `majority_baseline`,
  `b1_g1_verdict`, `headline_classes_of`, `quarantine_l1`, `run_split_stage`, `cluster_cells`,
  `ari_nmi`, `run_clustering`, the four `nulls.py` functions, the three fold functions. The only
  decision helpers without their own citation are the three of M14.
- Threshold values and verdict strings against the definitions, one by one: C1 `apf_max >= 0.02`
  for a kernel cell and the not-applicable string for an idle cell (CR 2.1 item 1); C2 8 pairs; C3
  informational; C6 on the modal header (al-Farabi 2.11 (a)); the `failed/` count `pass` at 0,
  `refused: failed count n > 0, seq axis uncorrected`, `refused: failed count not recorded` unless
  declared (CR 2.1 item 2; AA A5), with the declared cells kept out of the pair rungs (al-Farabi
  2.4); G-K0 tail 0.80, the idle 95th percentile of pooled K, `IDLE, measured` at or below the edge
  (CR 2.2 item 20); G-F part (i) strict exceedance of the p95 over 500 window-level shuffles in the
  declared executable form, part (ii) the [min, max] envelope of idle per-cell medians with every
  cell outside, `at floor in this lead` otherwise, the admissibility record gating both, the three
  floors written (CR 2.2 item 21; 2.1 item 15); G-C the two-to-one prediction recorded with detection
  at 1.5, the J dip at or below 0.75 within one snapshot of the K jump, in every rep, the content
  orderings `mean_abs` gibbs < histogram < gemm and `l0` histogram < gibbs < gemm on every rep index
  present, `disconnected lead` refusing the whole rung, the aliased-by-design absence verdict, and
  `combined` from the four (CR 2.2 item 22; al-Kindi review 1, 2); G-P `T >= 4 dt` resolvable,
  `2 dt <= T < 4 dt` marginal, `T < 2 dt` aliased by design at this size, unknown undeclared, five
  passes per cell for a rhythm feature, three snapshots per pass for a within-pass feature,
  `rhythm under-sampled` and `pass aliased` distinct, pair units from the cell's own count, the two
  dt representations, the `(INFERRED)` suffix (CR 2.2 item 23); the alias falsifier `r2 > 0.5`
  (item 23 last clause; al-Kindi review 5); G1 z 1.0, floor 0.80, the phase-randomized surrogates'
  5th percentile, `trend present` handing the cell to the whole-cell reading (CR 2.1 item 3); G2
  coverage `W dt / T` with floor 2.0 at 0.500 and 0.644 s, `not applicable, rhythm above Nyquist`
  when `T < 2 dt`, `undetermined by the interval calibration` on a flip, `undeclared` without a
  count, and the pair-unit column beside them (CR 2.1 item 4; al-Kindi review 7); G3 a per-kernel
  flag off the decision with the 4.5 dB and CV ceilings retired (CR 2.1 item 5; P2 Sec. 6 item 7);
  G4 `2H <= W` (item 6); G5 reported at 5 windows (item 7); the Delta-5 guard stated not applied
  (item 8); B1-G1 500 permutations at the unit, strict exceedance, rank reported,
  `near_unfalsifiable` excluded through `excluded_rows.csv` and printed in the split's score cells
  (CR 2.1 item 9; al-Farabi 2.7); B1-G3 the one-feature tree reproducing the full model on all but
  at most one unit, quarantine, re-run (item 10); B1-G6 by kernel count under LOKO and by cell count
  otherwise (item 11); G-ORD at `H = W // 2`, 20 order shuffles, `spread = p95 - p05` of 100 label
  shuffles, `order-blind (by construction)` at the whole-cell point (CR 2.2 item 27; al-Farabi
  2.11 (b)); the selection rule as Plan 03's `_pick_winner` re-pointed (smallest W passing G1, G2,
  G4; hop ratio nearest 0.5; best-feasible never `passes_acceptance`), every grid point kept, the
  roll-up with `rollup_kernel_refusals = "not_applicable"` (al-Farabi 2.1, 2.2, 2.11 (c)); G-J the
  independence null beside the idle cells' own J, interpretable where `K_t > 3 x` the floor's
  median K, no floor subtraction (CR 2.2 item 31; P2 Sec. 6 item 6); G-DEC (a) to (e) with
  `decay not resolved` when G-P's within-pass verdict is not admitted, `no decay beyond breadth`
  when K falls more than median `l0`, `no decay beyond floor or host` on a same-sign significant
  idle slope, the within-pass block-shuffle surrogate at the 5th percentile, seven of eight reps at
  one phase (CR 2.2 item 34); G-L (i) strict exceedance of the LOKO null and `level only`, (ii)
  `r2 > 0.5` on the per-kernel CV against `1 / sqrt(K_median)` (item 24); G-N 3, 2, 1, 0 kernels
  (item 25); G-X the blind LOKO campaign test at 500 shuffles plus ML's confound record and the
  total-confound refusal of the LOKO headline (item 26; K2 4.3); G-DIM feature count on every row,
  `d*` from the strongest single rung, the per-fold reduction when d exceeds the training cells
  (item 32); G-M the five-seed spread and the (6, 0) / (7, 1) sign rule (item 33); G-V L0, L2, L3
  as population variances, `LOKO not estimable` when L0 > L3 for every informative feature (CR 2.3
  item 35).
- Verdict strings: `verdicts.py` holds every string of SPEC 3.0 verbatim plus the three review
  additions (`GC_ALIASED_BY_DESIGN`, `GORD_ORDER_BLIND_BY_CONSTRUCTION`, `GDEC_NO_BOUNDARY`), each
  attributed; `is_refusal` enumerates the named refusals.
- Refusals written: every gate writes its refusal string into its CSV and a `params` block with
  `inputs_sha256`; 236 JSON result files on run 1 checked for the three keys `schema`, `params`,
  `citation` (none missing); `excluded_rows.csv`, `excluded_cells_pair_rungs`, `grid_complete.json`
  and the driver's refusal of the split stage on an incomplete grid.
- Interfaces (step 4): `extract.csv` has the 58 columns of SPEC 2.2 in `schema.EXTRACT_COLUMNS`
  order; `sidecar.json` has every key of SPEC 2.3 (plus `rep_source`, `source_sha256`, `params`,
  `citation`); `cells.csv` has the 12 columns of SPEC 2.7; the 13 grid CSVs, 13 `g1_surrogates.npz`,
  the 4 `gord.json` and the 13 (norm) or 26 (raw and norm) feature files exist for every rung; the
  SPEC section 1 output layout is complete on run 1 (every listed file present); builder 3 reads
  builder 2's `gc.csv` (`rep = all`), `gf.csv` (`part = i`), `gk0.csv` (`archetype_measured`),
  `gp.csv`, `gj.json` (`idle_J`), `selection.json`, `table5_grid.csv` (`coverage_pairs`, `G2_pairs`,
  `n_kernels_na_*`, `gc_verdict`, `gf_part1`), `gl.csv`, `gn.csv`, `gx.csv`, `gdim.csv`, `gm.csv`,
  `gv.csv`, `gv_summary.csv`, `clustering.json` (`per_algo`), the split directories (`__raw` first)
  and `scores.json` by the names builder 2 writes. The one interface disagreement is B1 (which
  score a `scores.json` stands for after a quarantine).
- The streaming pass (SPEC 2.4): `extract.py:252-351` holds two `Snapshot` objects and the row
  buffer, emits through `csv.writer` to a `.tmp` renamed at the end, refuses a non-monotone seq,
  synthesizes a K = 0 row for a gap, and reads `.csv.zst` through `zstd -dc -q` first, the
  `zstandard` module second, gzip and plain text otherwise; the piano roll is the only figure that
  re-streams a trajectory and it subsamples through the same reader.
- Binding rules: no path under `/mnt/nfs` or `/project` in any `.py` file (the runbook names the
  server root once as the value the author passes); no `ssh`, `scp` or `rsync`; the sandbox family
  is not named in the code (the campaign labels `sandbox_deepdive_01c`/`01c1` appear as label
  strings only, as SPEC 2.6 requires).
- The runbook (step 5): every `python3 -m plan11_encoding_ladder.<module> ...` line was parsed
  against its CLI (record (5)): all run as written except the three shorthand lines of M5; the
  flags the runbook names (`--fused-plane-mask`, `--idle-marker`, `--documentclass article`,
  `--c1-activity-min` absent from the driver as M1 says) exist where it says they do; `run_moves
  plan --moves 6-7` and `status` print as described.
- The skeleton (step 6): `apf_paper/p2_skeleton.tex` (432 lines, 135 comment lines) and the run
  copy `report/paper2_skeleton.tex`: every non-comment line is a LaTeX command, an environment line
  or a tabular row whose cells are column headers or P2 Sec. 4's own table cells (checked by a line
  scan and by `latex_skeleton.prose_lines`); `\title{}`, `\author{}` and every `\caption{}` are
  empty; the comment bullets are P2 Sec. 3's substance bullets in abbreviated form, none of them a
  paper sentence. No TeX compiler on this machine, so compilation is unverified (as in cycle 1).

---

## For the author

1. **C1's threshold decides which kernels exist for the paper** (unchanged from CHECK_1 item 1;
   run 1 shows it again: 9 of 12 kernel cells out before any gate runs). With `apf_max >= 0.02`,
   `K_max` must reach 5,243 pages in at least one snapshot; P2 Table 3's footprints put floyd,
   histogram, nbody (2,048), gibbs (256) and the lexer below it unless a pass-boundary spike lifts
   them, which removes the level-matched triple of the claim and removes the lexer before G-K0 can
   relabel it. Decide before the data: keep 0.02, lower it (`--c1-activity-min`, once M1 exposes
   it), or express it against the idle floor, and record the choice as a change to a pre-registered
   gate.
2. **Which score is the rung's score after a B1-G3 quarantine.** B1 above follows SPEC 3.7.2
   ("Table rows use the re-run") and makes the comparison gates read the same. The other reading
   (the gates and the tables both use the full model, with the quarantine reported beside it) is
   defensible if you hold that B1-G3 names a feature without changing the score. Say which; the
   fix is five lines either way, but it must be one reading for both builders.
3. **G-ORD's rule is two-sided as written.** CR 2.2 item 27 reads "if the shuffled score is within
   the null's spread of the ordered score"; `gate_gord` implements `|ordered - shuffled| <=
   spread`. On run 2 APF at W 8, 16 and 32 read `resolution` because the shuffled score was
   *higher* than the ordered one by more than the spread (0.74 against 0.50 at W 8), which is not
   evidence that W resolves a rhythm. If you want "resolution" to mean "order helps", declare the
   one-sided rule (`ordered - shuffled > spread`) and record it as `gord_rule` in `params`.
4. **G-V's "every feature" excludes constant features.** `variance.py:68-76` leaves a feature
   with `L0 = L3 = 0` out of the count before deciding `LOKO not estimable`; CR 2.3 item 35 says
   "every feature". A rung whose only informative features have `L0 > L3` reads not estimable while
   its constant features (a `duty` of 1.0 everywhere) would otherwise have saved it. The builder's
   reading is recorded in `params` and in the `constant` column; confirm or revert.
5. **G-DEC's idle clause under `idle_slope_rule = "any"`** (unchanged from CHECK_1 item 4) refuses a
   real decay by chance about one run in five once eight idle cells exist; `"min_reps"` is exposed.
   Choose before the data.
6. **The selection gates G2 in seconds** (unchanged from CHECK_1 item 3): `select` uses the
   roll-up of the 0.500 s and 0.644 s columns (CR 2.1 item 4's flip rule) and reports `G2_pairs`
   without gating on it. On the real cells the 0.644 column equals the pair-unit verdict at the
   median cell; the 0.500 column is 0.78 of it, so a point whose pair coverage lies in [2.0, 2.58)
   reads `undetermined` and the selection moves to a larger W. Run 2 shows the extreme case on
   the synthetic corpus (dt_est 5 s: every W fails in seconds, W 64 passes in pairs; every rung
   selected `best-feasible`). Confirm this is the intended rule or gate on `G2_pairs`.
7. **B1-G1's refusal in Table 6** (unchanged from CHECK_1 item 2): under the definitions the LOKO
   norm cell of the measured blind spot will read `near_unfalsifiable`, not the score. Say whether
   Table 6 prints the score with the verdict beside it or the string.
8. **The remaining exposed defaults** this check found consequential on the synthetic runs:
   `part1_consequence = "void"` for G-F (i), `g3_min_cells = 7` (a count, never met by fewer than 7
   cells), `jump_detect_ratio = 1.5` and `jump_reference = "cell_median_K"` for G-C, `pass_frac =
   0.5` for G-DEC, `gl2_level = "kernel"`, `gm_n_seeds = 5`, `dim_match_method =
   "train_importance"`, `table8_rung = "combined"`, `fused_plane_mask = "K"`, the head-drop key of
   M10. The full list with defaults is SPEC section 8 and the three BUILD reports; nothing in the
   code moves a threshold away from those.
9. **G-C content with fewer than eight admissible reps.** `gates_calibration.py:337-338` requires
   every rep index present in all three kernels and `len(reps) == max(cells per kernel)`, so one
   inadmissible gibbs, histogram or gemm cell (a C1 failure, a recorded failed count) turns the
   rung's G-C into `disconnected lead` although the orderings hold on every rep that exists. CR
   2.2 item 22 says "eight of eight"; it does not say what a missing rep is. Choose between the
   literal refusal (as coded) and `not run: n of 8 reps admissible`, and record it.

---

## Record

### (1) The test suite, verbatim

Command, from `plan11_encoding_ladder/`: `python3 -m pytest -q -rs tests`

```
.............................s.......................................... [ 45%]
........................................................................ [ 91%]
..............                                                           [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_extract.py:515: zstandard module not installed
157 passed, 1 skipped in 251.42s (0:04:11)
```

Exit code 0. The count is the one FIX_1.md reports (157 passed, 1 skipped); nothing was added or
removed between the fixer's run and this one. The runbook's alternative, `python3 -m unittest
discover -s tests -p "test_*.py"`: `Ran 96 tests in 136.912s`, `OK (skipped=1)` (M6: 62 gate tests
are not discovered; the count was 94 in cycle 1 and is 96 now because two of the fixer's three
new tests are `unittest`-style).

### (2) The driver end to end on a synthetic corpus

Corpus: builder 1's presets (`synth.corpus_specs(n_pairs=120, reps=3, idle=2)`) filtered to gemm,
floyd, gibbs and histogram (the pulse kernel, the decay exhibit, the control, and the
content-ordering triple), 3 reps each (seeds 42, 1000 + i, 2000 + i by `synth.rep_seed`), plus 2
idle cells (`kernel_sleep_v2`, label `idle`, seeds 42 and 1012), 120 pairs per cell, written by
`synth.write_cell` and compressed with the `zstd` binary: 14 cells in 63.3 s. Both runs from
`VM_Capture_QEMU/` through `python3 -m plan11_encoding_ladder.run_moves`.

**Run 1, the runbook's standard command** (`run --out <S>/run1/out --root <S>/run1/root
--null-perm 20 --n-jobs 4 --assume-failed-zero --assume-reason "checker cycle 2 smoke run"`):
exit 0 after 401 s wall time (6 min 41 s), all 78 ledger entries `done`, `gf_check` `pass`; the
five split stages took 122 s (apf, raw and norm), 64 s (combined), 58 s (content), 49 s
(persist), 47 s (wapf), the extract 9.2 s, everything else under 3 s. It wrote `cells.csv` (14
rows, all `ok`; rep 0 = seed 42; the idle cells `idle__rep00__idle`, `idle__rep01__idle` with
role `idle`), 14 extracts and sidecars (58 columns, `header_ncols 66`, `n_pairs 120`, `dt_est_s
5.0`, `status ok`), `gates/preconditions.*` (C1 `fail` on floyd, gibbs, histogram at `apf_max`
0.0165, 0.0016, 0.0088; the not-applicable string on the idle cells; `n_all_hard_pass 5` of 14;
`failed_source` `declared zero: checker cycle 2 smoke run`), the four input templates, `gp.csv`
(gemm `undeclared` on every column), `gc.csv` (apf, persist, wapf `pass` on gemm in 3 of 3 reps
with `stat_a` 1.976 to 1.988, `j_at_event` 0.480 to 0.490, 4 events per rep from seq 25;
content and combined `not run: no admissible cell for gibbs,histogram`), `gk0.csv` (gemm `above
floor`, idle band edge 150), `gf.csv` (every row `not run: no admissible idle cell;
admissibility record missing`, the template being unfilled), 13 grid points x 5 rungs
(`temporal_per_kernel.csv`, `g1_surrogates.npz`, 4 `gord.json` per rung, every G-ORD `not run:
fewer than two archetypes with windows`), 26 feature files per rung (13 for combined),
`g3_flags.csv`, `alias.csv`, `table5_long/grid.csv`, `selection.json` (apf, wapf, persist
`W8_H4`; content, combined `W16_H8`; every `passes_acceptance` true, `2 of 2` with G2 `not
applicable: no kernel with a declared pass period`), `gj.*` and `gj_mask/` (5 cells, `floor
unmeasured`), `gdec.csv` (`not run: no admissible floyd cell`), the ten split directories for apf
(five `__raw`) and five for each other rung (`b1_g1 = not run: 20 permutations < 500` by design;
LOKO accuracy `None` since one kernel survived C1), `gl/gn/gx/gdim/gm/gv/clustering`,
`gm_runs/seed0..4`, `splits_matched`, `report/tables/` (table5, table5_g3, table6, table7, table8,
tablegv, table4_status, preconditions, table_wapf_over_apf, each as csv/md/tex/json), eight figures
as PDF and PNG with `figures.json`, `paper2_skeleton.tex`, `manifest.json` (467 files hashed),
`driver_state.json`. Failures: none at the process level. What is wrong in the outputs is M1 (9
kernel cells excluded by C1, so G-C content, G-DEC, G-ORD and LOKO are not exercised) and M16
(Table 6 prints `--` for the excluded kernels); B1 cannot show on this run because no LOKO score
exists.

**Run 2, every cell admissible.** The same corpus copied to `<S>/run2/root`; moves 0 to 2 through
the driver; then by hand `gates_precondition preconditions --c1-activity-min 0.001` (all 14 cells
`all_hard_pass`), `inputs/idle_admissibility.json` filled (every key), `inputs/pass_table.csv` with
gemm 5 and floyd 12 passes per 600 s declared (the presets' pulse periods 24 and 10 at 120 pairs)
and histogram `inferred: 6147` (the aliased case), `gates_calibration gp` re-run (floyd, gemm
`resolvable / admitted / admitted`; gibbs `undeclared`; histogram `aliased by design at this size
(INFERRED)`, `admitted (INFERRED)`, `pass aliased (INFERRED)`); then `run --moves 3-13 --null-perm 20
--n-jobs 4 --assume-failed-zero --assume-reason "checker cycle 2 run 2"`. Exit 0 after 4,191 s
wall time (1 h 09 min 51 s), 70 ledger entries `done` (moves 3 to 13), `gf_check` `pass`. Where the
time went: G-ORD 424 s (apf), 363 s (persist), 772 s (content), 463 s (wapf), 718 s (combined), 2,740 s
in all with `--n-jobs 4` ignored (M7); the split stages 332 s (apf, raw and norm), 209 s (persist),
303 s (content), 176 s (wapf), 308 s (combined) at 20 permutations; `gdim` 60 s; G-F 7 s and 15 s;
everything else under 5 s.

Results of the moves: G-C `pass` on all five rungs (apf, persist, wapf on gemm 3 of 3 reps,
`stat_a` 1.976 to 1.988, `j_at_event` 0.480 to 0.490; content orderings hold in 3 of 3 reps with
`mean_abs` 0.00122 < 0.01196 < 10.68 and `l0` on the persistent pages in the declared order;
combined from the four). G-K0: every kernel `above floor` (idle band edge 150; tail medians gemm
4247, floyd 2211, gibbs 406, histogram 2213). G-F at `W8_H4` (move 4) and at every rung's selected
point (move 12; both rows kept): part (i) `inseparable at floor` on all five rungs (score at or
below the null's p95 on 2 idle cells); part (ii) `pass` everywhere except histogram on the content
rung, `at floor in this lead` (2 of 3 cell medians inside the idle envelope: the generator's counter
content gives the same `r_l0_q50_per` as its idle model, a synthetic artifact); `gf_floors.json` K
[150 x 5], `l0` per changed page [2, 2, 2.5, 3, 3], J [0.961 x 5]. G3 on APF: floyd and gemm
`rhythm flag: present` in 3 of 3 cells each (SNR 8.0 to 8.5 dB against a surrogate p95 of 7.6 to
7.7 for floyd, a rahmonic of its period 10; 11.9 to 13.0 against 8.8 to 9.0 for gemm, period 24),
gibbs and histogram absent, every kernel flag `absent` because `g3_min_cells = 7` exceeds 3 reps.
G-ORD: APF `resolution` at W 8, 16, 32 (ordered 0.50 against shuffled 0.74, 0.72, 0.55; spread
0.083, 0, 0) and `order-blind` at W 64 ("For the author" item 3); persist `order-blind` at every W
(ordered 0.25, 0.75, 0.50, 0.50 against shuffled 0.49, 0.63, 0.74, 0.63 with spreads 0.5 to 0.75);
content, wapf, combined `order-blind` at every W (ordered and shuffled both 0.50). Selection: every
rung `best-feasible`, `2 of 3` (apf, wapf, persist `W8_H4`; content, combined `W16_H8`): G2 fails in
seconds at every W with `dt_est` 5 s while `G2_pairs` passes at W 64 ("For the author" item 6);
Table 5 and Table 7 print `selected: best-feasible`. The alias falsifier `not run: fewer than three
cells or no spread` on every row (every cell has the same `dt_est`). APF splits at `W8_H4`: the ten
directories (five `__raw`); LOKO/archetype norm accuracy 0.500 (full model, null p95 0.500, rank 2
of 20) and 0.583 after B1-G3 quarantined `apf.k_over_med.mean` and `.peak2med` (null p95 0.583,
rank 11 of 20, 6 features); LOKO raw 0.083 full and 0.583 after five features were quarantined;
LORO kernel 0.917 / 0.833; within-trace kernel 0.833 / 0.750; every `b1_g1` `not run: 20
permutations < 500` by design. The same on the other rungs: persist LOKO 0.25 full and 0.417 after
`persist.j_excess.mean` was quarantined; content 0.50 both ways after eight `r_l0_*` features were
quarantined (28 left); combined 0.50 both ways (40 left). `gl.csv` part (i): apf `0.5 / 0.5`, wapf
`0.5 / 0.5`, persist `0.25 / 0.75`, content `0.5 / 0.75`, combined `0.5 / 0.5` (the full-model
numbers; B1), every verdict `not run: 20 permutations < 500`; part (ii) `pass` (r2 0.21 on 4
kernels). Table 7's LOKO rows print 0.583 (apf, 6 features), 0.417 (wapf, 3), 0.417 (persist, 7),
0.500 (content, 28), 0.500 (combined, 40), 0.500 (combined (matched), 40; M3) with `G-M vs APF`
for persist `difference with margin (diff -0.250 <= spread 0; 0 up, 1 down)` computed from the
full-model 0.25 against 0.5 (B1). `gn.csv`: WORKING-SET 3 `headline`, SCATTER 1 `structural
novelty`. `gx.csv`: every rung `not applicable: one campaign label; confound: partial` (M11). Table
6 `all` row: within-trace raw 0.917, norm 0.750; LORO raw 1, norm 0.833; LOKO raw 0.583, norm 0.583
(the re-run's numbers; B1); majority 0.750; level set A on floyd and histogram, B on gemm. G-J:
floyd, gemm, histogram `interpretable`, gibbs `floor overlap` (K 406 against 3 x 150). G-DEC: floyd
10 passes of 10 snapshots found by the K jump, the fall present at phase 0 in every rep with
`l0_rel_drop` 0.516 against `k_rel_drop` 0.474 to 0.477 (rule (c) passes), but one of the two idle
cells carries a `slope significant` whole-cell `l0` slope, so every rep reads `no decay beyond
floor or host` and the kernel row `no decay` (3 reps below `min_reps = 7`; "For the author" item
5); gibbs `no slope (0 of 3 reps with a slope)`; `fig_floyd_decay` is the placeholder with that
verdict. G-DIM: apf, wapf, persist `full vector (d = 8)`, content and combined `declared reduction`
(36 and 60 against 9 training cells), `combined (matched)` reduced to apf's 8 by train importance.
G-M: the five APF seeds all 0.5, spread 0; every pair `difference with margin`. G-V `estimable` on
every rung (apf 1 of 8 features with L0 > L3, combined 2 of 60). Clustering k = 2 on the combined
rung, ARI 0.195 equal to its null p95 (`exceeds_ari` false), the same for NMI and for the two
alternatives. Table 8: WORKING-SET row `2 (gemm, floyd)` under WORKING-SET and `1 (gibbs)` under
SCATTER; SCATTER row `1 (histogram)` under WORKING-SET; clusters `c0:6 c1:3` and `c1:3`. Figures:
all eight, the fused plane with 357 masked points (the three gibbs cells), `j_hist` with the five
floor quantiles from `gj.json` and an empty absent-reason, the piano roll from the first gemm cell.
Table 5 65 rows, Table 7 30 rows, `manifest.json` 481 files hashed. Failures: none at the process
level; what is wrong in the outputs is B1 (the two scores per split), M11 (the confound string),
M3 (the matched row's feature count), and the placeholder of M2 does not arise here because the
verdict was not `decay`.

### (3) to (6)

Covered above: (3) the gate functions against their definitions (no threshold or verdict-string
deviation found; B1 for which score the comparison gates judge after a quarantine; M2, M4, M15,
M11 for readings around the gates; M14 for two helper citations); (4) the interfaces (B1 and M3 on
`scores.json`; the rest agree with SPEC, including the five cycle-1 interface fixes); (5) the
runbook commands (M5, M6, M7; everything else runs as written); (6) the skeleton (no body prose;
M13 for two escaped math cells).

### (5) The runbook commands, parsed

Every `python3 -m plan11_encoding_ladder.<module> ...` line of `RUNBOOK.md` (64 command lines,
continuation lines joined) was run against its CLI with `<out>` and `<root>` pointing at an empty
scratch directory (`synth corpus` with `--reps 0 --idle 1 --n-pairs 8 --no-compress` appended so it
finishes; `run_moves run` with `--dry-run`): an argparse rejection (a `usage:` line on stderr)
counts as not running as written, an exit 2 with `missing input: .../cells.csv` counts as running.
Result: 61 run as written (exit 0 for `latex_skeleton`, `run_moves plan`, `run_moves status`,
`run_moves run --dry-run`, `synth corpus`; exit 2 naming `cells.csv` for every other module), 3
are rejected (`RUNBOOK.md:304, 322, 337`, the `grid/g3/gord/select` shorthand of M5). The `for r in
apf persist content wapf combined` loop of move 3 runs as a shell loop (checked by expanding it).

