# plan11_encoding_ladder, the detection layer: runbook for the author

Written 2026-09-17 by builder B (report) of build epoch 2. This is the command-line companion to
`SPEC_DETECTION.md` section 6: what to install, the exact commands in al-Kindi's move order
(K3 Sec. 4; SPEC_DETECTION 6.2), what each move writes, what to look at afterwards, the cost of
each step, the smoke run, and how the stage-2 and stage-3 cells enter. Every command is also run
for you, in this order, by the detection driver (`run_detection.py`), which keeps its own ledger
(`<out>/driver_detection_state.json`) and never touches the encoding paper's `driver_state.json`.

Conventions. `<out>` is a fresh output root for the detection run (not the encoding paper's; the
detection layer writes only under `<out>/gates/detection/`, `<out>/detection/`,
`<out>/report/detection/`, `<out>/features/` and `<out>/extract/`, plus the one rewrite of
`cells.csv` described at D0). `<root>` is the retention root you pass; the toolkit never assumes
it. `<sel>` is the encoding run's `gates/selection.json`. Every command is run from
`VM_sampler/VM_Capture_QEMU/` as `python3 -m plan11_encoding_ladder.<module> ...`. Every command
exits 0 on success (a written refusal is a success), 2 when an input file is missing (its path on
stderr), 1 on an internal error; the one-class CLI adds exit 3 for a usage error (`usage: a second
one-class model needs --secondary`, al-Farabi review M10), which is outside the refusal
vocabulary and writes nothing. Every result file carries a `params` block with every parameter
value used and `grid_source`; every table cell that reads `not run:` names the file or the move
that is missing.

Three defaults in this runbook differ from `SPEC_DETECTION.md` because a review marked them "must
change before build": the in-fold threshold is set on inner leave-one-benign-workload-out scores
(`--threshold-source inner_lowo`; al-Kindi review 1; `oob` is the labelled, optimistic
alternative); the workload-level null runs for `lowo`, `loco` and `one_class` (`--null-splits
lowo,loco,one_class`; ML review 2.7; al-Kindi review 2); and the C1 rule under `report` applies to
every class alike (al-Kindi review 5). Section 8 lists every choice the definitions leave open.

The sandbox family is named only by index and letter in every file this layer writes. The class
file (section 2) is yours; no agent fills it.

---

## 0. Prerequisites

As `RUNBOOK.md` section 0: Python 3.10 or newer; `numpy`, `scikit-learn`, `matplotlib`;
optional `scipy` and `zstandard`; `pytest` for the tests. Nothing new to install for this layer.
Check once:

```
python3 --version
python3 -c "import numpy, sklearn, matplotlib; print(numpy.__version__, sklearn.__version__, matplotlib.__version__)"
which zstd || python3 -c "import zstandard"
```

Run the tests once before the first real move (synthetic data only; nothing under `<root>` is
touched). The report tests of this layer are `tests/test_report_detection.py` and
`tests/test_run_detection.py`; builder A's are `tests/test_classes.py`, `tests/test_detection_*.py`,
`tests/test_gates_detection.py`, `tests/test_synth_detection.py`:

```
cd VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder
python3 -m pytest -q
```

## 1. The smoke run (synthetic corpus; minutes to tens of minutes)

```
cd VM_sampler/VM_Capture_QEMU
python3 -m plan11_encoding_ladder.synth_detection corpus --root /tmp/p11det --write-classes --order-confound on --campaign-labels round_robin
python3 -m plan11_encoding_ladder.run_detection run --out /tmp/p11det/out --root /tmp/p11det --classes /tmp/p11det/classes.csv \
    --grid-default W8_H4 --null-perm 20 --n-jobs 2 --assume-failed-zero --assume-reason "smoke run" --c1-activity-min 0.001
python3 -m plan11_encoding_ladder.run_detection status --out /tmp/p11det/out
```

Every table is present afterwards and every two-class null row reads `not run: 20 permutations
< 500` by design; the smoke run is never admissible for the paper (SPEC_DETECTION section 7
item 32). When builder A's modules are absent on the machine, the driver's own steps still run
with `--only-modules tables_detection,figures_detection,latex_skeleton_p3,driver
--skip-missing-modules` on a fixture tree (`tests/detection_fixtures.py`).

## 2. The class file, `<out>/inputs/classes.csv` (yours; SPEC_DETECTION 2.1)

A CSV with exactly these columns, in any order: `path_prefix, class, member_index,
subfamily_letter, rep, order_index, family, workload_key`. Any other column is refused, so no
free-text column can carry a name. `class` is one of `benign_kernel, benign_relaunched,
benign_breadth, idle, harness_idle, sandbox, external`. A `sandbox` row needs `member_index`
(1 to 8) and `subfamily_letter` (A, B or C); `rep` and `order_index` are per cell (a row that sets
them must match exactly one cell); `family` is needed for `benign_breadth`; `workload_key` is the
parent kernel of a `benign_relaunched` row (CR3 2.31).

Stage 1, the shape (the family and test-label path components are yours and never appear in
this repository):

```
path_prefix,class,member_index,subfamily_letter,rep,order_index,family,workload_key
kernel,benign_kernel,,,,,,
<idle family>/<idle test label>,idle,,,,,,
<family>/<member 1 test label>,sandbox,1,A,,,,
<family>/<member 2 test label>,sandbox,2,A,,,,
...
<family>/<member 8 test label>,sandbox,8,C,,,,
```

plus, once you have the realized order, one row per cell with `order_index` (168 rows: the
kernels first, then the sandbox cells in order by member, then the idle cells; P3 0a). Without
`order_index` the order test, the drift regression, early-against-late idle and the letter
sequence read `not run: order_index missing`. Matching is by `path` (a cell directory, a prefix
of one, or its last four components); the longest matching prefix wins; a cell that matches no
row is `unassigned`, is listed in `admissibility.csv` as inadmissible, appears as its own row of
Table 4, and is yours to resolve before D2 (al-Farabi review M9). The validator's refusals quote
row numbers, class values and column names only (al-Kindi review 13).

The campaign token of every sandbox and external `cell_id` is `--campaign-label TEXT` when you
pass it and the literal `stage1` otherwise; the raw launch label is never copied into a public id
(al-Farabi review M1).

## 3. Reading the letter sequence

`gates/detection/letter_sequence.txt` is the only representation of the realized order any table,
figure or log of this layer shows: one token per cell, sorted by `order_index`, grammar
`<L>[<m>]r<rep>` with `L` in `S` (sandbox), `B` (kernel or breadth), `I` (idle), `H`
(harness-idle), `R` (re-launched), `X` (external), `<m>` the member index for `S` and `X` only.
Stage 1 reads 96 `Br<r>` tokens (one `Br0 ... Br7` run per kernel), then `S1r0 ... S8r7`, then `Ir0 ... Ir7`. The
kernel's identity is dropped on purpose. `order_index` itself appears only in `cell_classes.csv`;
the per-cell appendix table prints the token.

## 4. The moves, in order (96 + 8 + 64 cells)

The driver runs every command below; by hand, each is one line. The driver's flags that reach
the commands are given in brackets.

### D0, identity (K3 move 1)

```
python3 -m plan11_encoding_ladder.run_detection run --out <out> --root <root> --classes <your classes.csv> --selection-from <sel> --moves 0
```

or by hand: `extract index --root <root> --out <out>`; `classes validate --out <out> --classes
<out>/inputs/classes.csv`; `classes apply --out <out> [--campaign-label TEXT]`; `classes
inherit-selection --out <out> --from <sel>` (or `--default W8_H4` when no encoding selection
exists); `classes letter-sequence --out <out>`. Writes `cells.csv` (rewritten with public ids),
`cells.pre_classes.csv`, `gates/detection/classes_validation.json`, `cell_classes.*`,
`letter_sequence.*`, `gates/selection.json` (with `params.grid_source`) and
`gates/detection/inherit_selection.json`. The driver copies `--classes` to
`<out>/inputs/classes.csv` when absent there; a differing file already present is `refused:
inputs/classes.csv exists; edit it or pass --force`, recorded in the ledger (al-Farabi review
M10). Look at: `classes_validation.json` `status ok` and `unassigned_cells = 0`; 96 kernel
rows, 8 idle rows, 64 sandbox rows with `status = ok` and public ids; `S = 8`, `B = 13`,
`n_assignments = 203490`; the letter sequence (or its `not run` line). If `extract all` ran before
`apply`, `apply` refuses (`refused: extract/ holds <n> directories not keyed by a public
cell_id; remove them and re-run`; al-Farabi review M11): remove `<out>/extract/` and re-run D0
and D1.

### D1, the extracts

`extract all --cells-csv <out>/cells.csv --out <out> --jobs 4`. Writes `extract/<cell_id>/*`
for 168 cells; one to three minutes per cell, the longest step. Look at the sidecars: `n_pairs`
about 890 to 945, `header_ncols = 66`, `status = ok`.

### D2, preconditions, your inputs, admissibility (K3 move 2)

`gates_precondition preconditions --out <out> --assume-failed-zero --assume-reason "AA A5: any
failed job re-runs the whole cell" [--c1-activity-min F]`; then the templates `gates_calibration
pass-table`, `gates_precondition gk0-template`, `gates_precondition idle-admissibility-template`,
`gates_detection head-drop-template`, `gates_detection gk0-sandbox-template` (each written once,
never overwritten); you edit `inputs/*`; then `gates_calibration gp --out <out>` and
`gates_detection admissibility --out <out> --c1-rule report`. Writes `gates/preconditions.*`,
`gates/gp.csv`, `inputs/*`, `gates/detection/admissibility.*`. Look at: `all_hard_pass` per
cell; which cells of any class fail C1 and enter through the report rule (Table 4's `n C1 fail
(reported)`); the eight numbered lines of `inputs/gk0_source_sandbox.csv` that you write.

The head drop (al-Farabi review M5; al-Kindi review 6): `inputs/head_drop.csv` holds one row per
workload key of every class (the public keys, `sandbox_member_<m>` included) with
`head_drop_pairs = 0` and the reason `default 0; declared at D2`. A member's head drop is your
number from the program's specification, fixed at D2 and never read off the APF(t) figure of D5;
a normalization constant chosen from the data is a map learned from the fold, which the
certification forbids. Every feature file and every split records the full map.

### D3, G-C (K3 move 3)

`gates_calibration gc --out <out> --rung R` for `apf, persist, content, wapf, combined`. Writes
`gates/gc.csv`. Look at: one `pass` per rung on the kernels, "lead connected, calibrated on the
benign side"; a `disconnected lead` voids every negative of that rung, and every score cell of
that rung in Tables 7, 8, 11, the level tables and the ladder prints `refused: disconnected lead`.

### D4, the floors, the level map, the features (K3 moves 4, 5)

`gates_precondition gk0 --out <out>`; `gates_detection gk0-cells --out <out>`; `gates_detection gn
--out <out>`; the driver's internal step `features-at-selection <rung>` for the five rungs (it
reads `gates/selection.json` at run time and calls `series.build_features` for raw and norm, norm
only for combined, recording `grid_source` in `gates/detection/features_at_selection.json`; by
hand: `series features --out <out> --rung R --grid-id <gid> --both`); `gates_precondition gf --out
<out> --all-rungs --n-perm 500`; `gates_detection anchor --out <out> --part idle_sets`;
`gates_detection anchor --part idle_early_late`; `gates_detection drift --out <out>`;
`figures_detection --out <out> --only fig2_three_floors,fig_level_map`. Writes `gates/gk0.csv`,
`gates/detection/gk0_cells.csv`, `gk0_members.csv`, `gk0.json`, `gn.csv`, `features/*`,
`gates/gf.csv`, `ganchor.csv` (idle rows), `drift.csv`, two figures. Look at: which members have
cells `at floor` (they leave the denominator); G-F (i) `inseparable at floor` on every rung; the
idle sets row (`not applicable: one idle campaign (n = 8 cells)` in stage 1); early-against-late
idle; the level map: does the sandbox band overlap the benign band (R8).

### D5, APF(t) per tier (K3 move 6)

`figures_detection --out <out> --only fig_apf_per_tier`. Look at: reps overlaying; the members'
spikes beside the kernels'. Never read a head drop off this figure (D2).

### D6, the harness clause (K3 moves 7, 8)

`gates_detection harness --out <out> --rung R` for the five rungs. Writes `harness.csv`. Every
row reads `not run: stage 2 absent` until stage 2.

### D7, apf, breadth alone (K3 move 10)

`detection_metrics splits --out <out> --rung apf --raw-and-norm --all-splits --null-perm 500
--null-splits lowo,loco,one_class --threshold-source inner_lowo --loco-mode cell --n-jobs N`;
`detection_metrics one-class --rung apf --null-perm 500 --n-jobs N`; `gates_comparison gx --out
<out> --rung apf --null-perm 500`; `gates_detection anchor --part kernels`; `gates_detection
order --out <out> --consequence size`; `gates_detection leak-probe --out <out>`; `gates_detection
gop --rung apf --split lowo|loco|lofo`; `gates_detection glm --rung apf`; `gates_detection gl
--rung apf`; `tables_detection --out <out> --only table9_pitfalls`. Writes
`gates/detection/splits/apf/*`, `gates/gx.csv`, `ganchor.csv`, `order.csv`, `leak_probe.csv`,
`gop.csv`, `glm.csv`, `gl.csv`, Table 9 (apf rows). Look at: the raw row against the normalized
row: how much is level; the order test's size under both half rules (`within_workload` strips
member identity; `within_class` is the definition's letter, which at stage 1 is member identity
by construction); G-ANCHOR (i)'s size; the cadence and active-fraction probes; G-LM per member
with the label `median K, stage 1`.

### D8, the fused plane (K3 move 11)

`gates_readings gj --out <out>`; `figures_detection --out <out> --only fig4_fused_plane_tiers
[--identity excess|raw] [--plane-per-letter]`. Writes `gates/gj.*` and the fused plane (one
panel per benign tier, one for the class with all members as marker shapes coloured by letter,
the idle cloud in every panel; the identity coordinate is `J - J_null`, the miss table's; the K
mask of `gj.json` applied to every class, the idle cloud drawn unmasked because it is the floor
itself). Look at: do the members fall together; is where they fall empty of benign cells once
idle is drawn.

### D9, D10, D11: persist, content, wapf (K3 moves 12, 13, 14)

For each rung the same as D7 with `--norm` (splits, one-class, gx, gop x 3, glm, gl); at D9 also
`detection_levels level2 --rung persist` and `detection_levels level3 --rung persist`. Look at:
J against its null; G-P's `undeclared` label stands beside every persistence reading of the
sandbox side; the amount axis; members 1 to 4 of the synthetic analogue.

### D12, combined, the levels, the per-rung gates, the detection tables (K3 move 16)

The same for `combined`, plus `detection_metrics splits --rung combined --norm --split lowo
--reduce-to-strongest`; `detection_levels level2` and `level3` for `apf, wapf, content,
combined`; `gates_detection gsig`, `gfp`, `g1c`, `gcal` per rung; `gates_detection gm`, `gdim`;
`tables_detection --out <out> --only
table7_detection,table8_member_recall,table11_splits,table_level2,table_level3`. Writes
`gsig.csv`, `gfp.csv`, `g1c.csv`, `gcal.csv`, `gm.csv`, `gdim.csv`, Tables 7, 8, 11, the level
tables. Look at: the headline: TPR at 5% with its realized FPR under LOWO above the null;
per-member eighths; the L0 ratio row under LOCO (a member below 0.1 reads as within-trace); the
four splits side by side; G-SIG's gap; which family fires; the level-2 rows per sub-family with
their own null verdicts (B reads `one member, no held-out test`).

### D13, the time-to-hear ladder (K3 move 17)

`detection_metrics ladder --out <out> --rung R --null-perm 0 --threshold-source inner_lowo` for the
five rungs; `tables_detection --only table_ladder`; `figures_detection --only fig6_ladder`.
Writes `ladder/*`, `ladder.csv`, `ladder.json`, the table, Figure 6. Look at: the shortest prefix
at which a rung reaches its 600 s reading (the whole-cell row equals the LOWO headline by
construction, al-Farabi review M6); `from_boundary` reads `from pair 1 only` in stage 1.

### D14, the miss table, the alias falsifier, G-V, the report (K3 moves 18 to 21)

`gates_detection miss-table --rung R`, `alias --rung R`, `gv --rung R` for the five rungs;
`tables_detection --out <out>` (all); `figures_detection --out <out>` (all);
`latex_skeleton_p3 --out <out> [--standalone apf_paper/p3_skeleton.tex]`; `tables_detection
--only manifest`. Writes `miss_table.csv`, `fp_table.csv`, `alias.csv`, `gv_two_class.*`,
`report/detection/*`. Look at: the miss table: what each missed member resembles, on which axis
(`axis` is the resemblance axis; `axis_of_largest` the residual); the cross-campaign row `not
applicable: stage 1`; `manifest.json`.

### D15, the tripwire (K3 move 23)

The driver's internal `tripwire-check`: every Table 7 and Table 11 row carries its G-F (i)
verdict and the drift-clause verdicts (G-ANCHOR (ii), early-against-late idle); a row without
them is listed and the verdict is `refused: <n> rows without a tripwire verdict`. Writes
`gates/detection/tripwire_check.json`; Table 5 prints it.

The full run, in one line:

```
python3 -m plan11_encoding_ladder.run_detection run --out <out> --root <root> --classes <your classes.csv> --selection-from <sel> \
    --assume-failed-zero --assume-reason "AA A5: any failed job re-runs the whole cell" --n-jobs 8 --null-perm 500 \
    [--null-rungs apf,content,combined] [--loco-mode rep_index] [--standalone-tex apf_paper/p3_skeleton.tex]
```

## 5. Cost

The counts below are forest fits per rung (300 trees each). The row unit is the cell (ML review
2.1), so one fit is on about 160 rows by 8 to 60 features, a small fraction of the epoch-1 fit on
about 39,000 window rows; the null's permutations are independent and run in parallel with
`--n-jobs`. No timing is asserted here: time one fit on the smoke run on your machine before you
start and multiply by the counts.

| Step | Fits per rung | Note |
|---|---|---|
| LOWO headline (`inner_lowo`) | 21 x (1 + 13) = 294 | the inner threshold costs B = 13 extra fits per fold; `inner_group_kfold` (k = 4) gives 21 x 5 = 105; `oob` 21 (optimistic) |
| LOWO null at 500 permutations | 500 x 294 = 147,000 | the expensive step; `--null-rungs` limits it to the rungs you name; `--n-jobs` splits the permutations |
| LOCO (`cell`) | 168 x 14 = 2,352 | its null 1,176,000 fits per rung: not affordable; `--loco-mode rep_index` gives 8 x 14 = 112 fits and a null of 56,000 |
| LOFO | 2 x 14 | seconds |
| one-class | 14 isolation forests; its null 500 x 14 = 7,000 | minutes |
| level 2, level 3 | 7 and 8 fits (nulls 280 x 7 exhaustive, 500 x 8) on 64 cells | minutes |
| G-LM (retrain) | about 6 x 14 per member, 8 members | minutes |
| the ladder | 5 prefixes x 2 readings x 294 (no null unless `--ladder-null-perm`) | minutes per rung |
| G-M seed spread | 4 x 294 on apf | minutes |
| G-FP recomputation | 294 per inseparable family | as needed |

The decision that sets the bill is the LOCO null: without it G-SIG cannot refuse (F2) and reads
`not run: loco null not run`; `--loco-mode rep_index` is the cheap reading of the "sibling reps
train" split; `cell` is ML 1.3's definition. Section 8 item 2.

## 6. Stage 2 and stage 3 cells

Add rows to `inputs/classes.csv` (`benign_relaunched` with `workload_key` = the parent kernel;
`benign_breadth` with `family`; `harness_idle`; `external` with `member_index`), then re-run D0
and D1 for the new cells (`extract all` skips cells whose sidecars exist), then the driver from
D2: a changed `inputs/classes.csv` makes every move from D2 stale on its own and the driver
re-runs them. No column and no code path changes: the harness clause, the harness floor of G-K0
and G-ANCHOR part (ii) switch on from the class counts in `cell_classes.csv`; external cells are
`test_only`, never trained on, scored by the `lowo/final` fold and printed as their own block of
Tables 7 and 8. The stage-2 inputs `inputs/iteration_boundaries.csv` (cell_id, boundary_seqs)
and `inputs/iteration_counts.csv` (cell_id, iteration_count) unlock the ladder's `from_boundary`
reading, the `per_iteration_K_sum` level quantity and the alias falsifier's second regressor.

## 7. Resume discipline

Gate result files are never edited by hand: the driver's staleness test hashes the parts it
reads (a file, one rung's entry of `gates/selection.json`, one rung's rows of a gate CSV) and a
by-hand edit is either invisible to it or makes every later move re-run. A re-run of `classes
apply` or a changed `inputs/classes.csv` makes every move from D2 stale; a changed
`gates/selection.json` makes every feature file and every split stale; a changed
`gates/detection/gk0_cells.csv` (the floor) makes every split, level and gate stale. `--force`
re-runs unconditionally; `--dry-run` records the plan without running; `--only-modules` and
`--skip-missing-modules` are for the smoke tests. The (W, H) per rung is inherited from the
encoding paper's selection (`inherit-selection --from`); re-gridding on the detection data is
allowed only under the grid rule (SPEC 3.5.7) and is not scheduled by this driver.

## 8. For the author: the choices this layer leaves to you (with the defaults that run)

1. `inputs/classes.csv` is yours, with the 168 `order_index` rows once the realized order is at
   hand; `--campaign-label TEXT` or the default token `stage1` for the sandbox ids.
2. `--threshold-source inner_lowo` (default; B extra fits per fold), `inner_group_kfold` (k = 4,
   the cheap variant) or `oob` (the definition's letter; optimistic through sibling reps; the
   realized FPR is the measurement). `--loco-mode cell|rep_index` and `--null-splits` decide the
   bill (section 5).
3. `--train-on-at-floor true|false`: the ML review recommends `false` (an at-floor positive teaches
   the forest that the floor is sandbox); al-Farabi's review names `true` as the SPEC's present
   behaviour. The driver passes the flag only when you give it; builder A's constant runs
   otherwise and every `params` block records the value used.
4. `--row-unit cell|window`: the cell is what D5 decided (ML review 2.1); `window` is epoch 1's
   habit, under which `oob` is not admissible.
5. `--null-rungs`: which rungs carry the 500-permutation null; the others read `not run: null not
   requested for <rung>`.
6. `NULL_VERDICT_STATISTIC = tpr05` carries the verb (P3 D5); AUC's verdict is printed beside it.
7. `--one-class-model isolation_forest` (neither generative nor a density model, ML 1.6) or `gmm`
   (`covariance_type` must be `diag` at d = 60 on about 100 benign rows); the primary is declared
   now; any other runs only as `--secondary` and is labelled not citable.
8. `--gop-cells-rule threshold_setters|realized_fps`: al-Farabi recommends `realized_fps` as the
   count the verdict reads; the roll-up is the worst fold (ML review 2.3); both counts are written.
9. `--order-consequence size` (stage 1, blocked by member) or `void` (an interleaved campaign);
   `--order-scope within_class|campaign` (al-Farabi review M12); the `within_workload` half rule
   is the row that strips identity (al-Kindi review 11).
10. `--level-quantity median_K` (stage 1, labelled on every G-LM row) or `per_iteration_K_sum`
    once `inputs/iteration_boundaries.csv` exists; `GLM_BAND_FACTOR = 2.0`.
11. `--kernel-family-rule tier|archetype`: under `tier` LOFO has two folds at stage 1 and the
    kernels' fold trains on idle plus the sandbox only; read it as a size; both reviews suggest
    the `archetype` reading as a second labelled row.
12. `--identity excess|raw` for the fused plane (`J - J_null` is the miss table's coordinate);
    `--plane-per-letter` for the per-sub-family panels; the idle cloud is drawn unmasked.
13. `REPS_IDENTICAL_RATIO = 0.1` (Table 8's L0 ratio row; a member below it reads `reps near
    identical: LOCO reads as within-trace`); the leak probe's vocabulary (`leak audible`, `leak
    not audible above the null`); both proposed by the ML review.
14. `--table10-rung combined` chooses the rung of the unsuffixed Table 10 and of the level tables
    in the skeleton; `--documentclass llncs|article` for the skeleton.
15. The sandbox source part of G-K0 (`inputs/gk0_source_sandbox.csv`), one numbered line per
    member, is yours; the synthetic corpus writes `unstated`.
