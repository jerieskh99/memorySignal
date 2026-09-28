# al-Farabi: review of SPEC_DETECTION.md before the build (2026-09-17)

Scope: the five conditions I was asked to certify, held against `SPEC_DETECTION.md` (read in full),
`P3_RAID_STRUCTURE.md` sections 0a, 1, 3, 9, 10, `raid_council/08_RAID_COUNCIL_REPORT.md` sections
1.5 and 2, `raid_council/02_ml_engineer_protocol.md` in full, `raid_council/06_al_kindi_revised.md`
sections 2.3 and 5, my own `raid_council/05_al_farabi_certification.md` section 5 and
`plan11_encoding_ladder/CERTIFY_al_farabi.md` condition (5), and the epoch-1 code the SPEC reuses
(`series.py`, `models.py`, `splits.py`, `nulls.py`, `schema.py`, `extract.py`,
`gates_precondition.py`, `gates_comparison.py`, `run_moves.py`; line numbers below are from those
files as they stand). No server, no data, no corpus specification, no steps file, no workload tree
was opened. The sandbox family is named by index and letter only.

Verdict in one line: cannot certify as written; certifiable after the corrections of section 2,
none of which changes the shape of the design.

## 1. Passes as specified

(1) No detection gate reads an analytical interpretation to fix anything upstream of the extract.
Holds. The only rewrite upstream of the extract is `classes apply` (SPEC 2.3), and it is driven by
the author's mapping file alone: it changes `role`, `kernel`, `cell_id` and one status string by a
declared rule and reads no result. Admissibility (3.1.4, 3.5.1) reads `gates/preconditions.csv`,
which is a validity check on the extract, plus the class as a declared inclusion rule. G-K0's floor
verdict (3.5.2) is an extract statistic against the idle envelope and enters only the denominator
(3.3.3), never a feature and never the extract. The three places where an interpretation sets a
later run, the B1-G3 quarantine (3.3.5), G-FP's recomputation (3.5.9) and `--reduce-to-strongest`
(3.3.7), are downstream of the extract, are defined by CR3 2.2, 2.15 and 2.32, and keep both
readings. The inherited (W, H) (2.7) is an interpretation of the encoding paper's kernels, declared
before the sandbox and idle cells and recorded as `grid_source` in every params block. The runbook's
staleness rule (6.1) makes a changed class file re-run everything from D2, so nothing downstream can
survive an upstream change silently.

(2) The class label enters only on the ground-truth side of a score, never as a feature; the class
mapping file is a step reference the author supplies. Holds. The feature matrix is
`series.build_features`'s `X`, whose names are `<prefix>.<channel>.<stat>` over
`FEAT = (mean, std, cov, median, max, p95, peak2med, duty)` and `wmean` (`series.py` 475 to 485):
no count, no metadata column. The label dict of 3.2.1 rides beside `X` and is read at fitting, fold
construction and scoring only; `splits._assert_grouped` (`splits.py` 107 to 115) runs on `cell_id`
and `workload_key`. The exclusion list `excluded_by_declaration` (3.1.1) is recorded before any
test. The mapping file (2.1) is a rule from the plan's path token to a class, validated by name rules
(`schema.parse_cell_path`, `schema.KERNEL_NAMES`) and reading no artifact, which is exactly the
"ground truth held outside the form, derived by declared rules from Stimulus's structural output"
of my certification section 5. G-ANCHOR part (i) reuses `gates_comparison gx`, which masks
`role == "kernel"` (`gates_comparison.py` 231 to 246; `models.py` 253), so no sandbox cell can enter
a campaign-label run. The order test, G-ANCHOR (ii) and the harness margins use class, campaign and
half as group labels on the ground-truth side (3.5.7, 3.5.11).

(3) Every refusal is written to the artifact. Holds for the gate vocabulary of 3.0: every verdict
string has a file under `gates/detection/`; `NULL_NOT_ESTIMABLE`, the smoke run's `not run: N
permutations < 500`, `G1C_SEARCH`, `not run: no selection for <rung>`, `not run: order_index
missing`, `not applicable: at floor (G-K0)`, `level2_no_heldout(n)` and the tripwire's `refused: <n>
rows without a tripwire verdict` are all written, and Table 5 prints the roll-up with a refusal
beating a pass. The exceptions are items M3, M4, M9, M10 and M11 of section 2.

(4) Level normalization is fixed per rung before the data and identical in every fold. Holds by
reuse: `k_median_cell` and `rung_series` are per-cell functions of the cell's own rows (`series.py`
252 to 256, 275 to 319), the feature file is built once per (rung, grid point) before any fold exists
(588 onward), every fold slices that one array, and the per-fold imputer and scaler act on columns
that no longer carry level (my CERTIFY condition (5)). The ladder's prefix normalization (3.1.3) is
also a per-cell function computed before any fold, and `ladder_norm` is declared. The exceptions
are items M5 (the head-drop key for non-kernel workloads) and M6 (the ladder's cap).

(5) Every threshold is declared before the data and recorded in params. Holds for every named
constant: `FPR_DECLARED = 0.05`, `FPR_RESOLUTION_LIMIT = 0.01`, `N_PERM = 500`,
`NULL_EXHAUSTIVE_BELOW = 500`, `NULL_MIN_ASSIGNMENTS = 20`, the 95th percentiles of G-K0,
`GLM_BAND_FACTOR = 2.0`, `GOP_MIN_CELLS = 5`, `GOP_MIN_WORKLOADS = 3`, `GFP_FLAG_FRACTION = 0.5`,
`HARNESS_COMPARABLE_TOL = 0.10`, `ALIAS_TOP_K = 10`, `ALIAS_R2 = 0.5`, `GM_ALPHA = 0.05`,
`GM_N_SEEDS = 5`, `LEVEL2_MIN_MEMBERS_HEADLINE = 3`, `LEVEL2_MIN_MEMBERS_TEST = 2`,
`B1G3_MAX_DISAGREE_WORKLOADS = 1`, `LADDER_PREFIXES_S`, and `LADDER_DT_S = 0.644`, which is
`schema.DT_BRACKET_S[1]` (`schema.py` 26; the SPEC cites the constant for 0.500 only and should cite
it for both). Each is a module constant or a CLI parameter whose value goes into params (SPEC
preamble, "Numbers never move"). The exceptions are the inline thresholds of item M7 and the scope
choice of item M12.

## 2. Must change before build

M1. SPEC 2.3 and section 7 item 2: the campaign token of every sandbox and external `cell_id`.
Wrong: `<campaign>` defaults to `schema.campaign_of(label)`, and `campaign_of` returns an unknown
label unchanged (`schema.py` 176 to 187), so the sandbox capture's raw launch label lands in every
public id, every extract directory name, every `predictions.csv` and the letter-sequence CSV unless
the author remembers `--campaign-label`. Rule 2 must hold by default, not by care. Correction: for
classes `sandbox` and `external` the campaign component is `--campaign-label TEXT` when given and
the fixed literal `stage1` otherwise; the raw label is never copied for those classes; the same
token fills the `campaign` column of `cell_classes.csv` for those rows; the validator warns
`campaign token defaulted to stage1 for <n> rows` in `classes_validation.json`.

M2. SPEC 3.2.1, 3.3.3, 3.3.4, 3.4: cells at floor train silently. Wrong: the label dict admits every
admissible classed cell, so a sandbox cell at floor (and the at-floor member of the synthetic corpus)
is a positive training row and a member of the null's workload pool while leaving the denominator.
CR3 2.8, K3 F3 and ML 3.6 say "leaves the true-positive denominator" and nothing about training;
the SPEC chose without saying. Correction: `TRAIN_ON_AT_FLOOR: bool = True` (the SPEC's present
behaviour) as a constant in `detection_metrics.py`, an argument of `run_detection_split`,
`run_level2`, `run_level3` and a CLI flag `--train-on-at-floor true|false`, recorded in every
params block and listed in section 7. State also, in 3.4, whether level 2's `denominator` and
`recall` exclude at-floor cells (the members.csv columns imply yes; say it).

M3. SPEC 3.3.3 and 3.3.4: the denominator rule under the null is not uniform. Wrong: the
denominator is "positive cells whose floor verdict is `above floor`", but idle and harness-idle
cells carry the verdict `control` and a cell without `gk0_cells.csv` carries `pending: gk0 not
run`; under a permutation that labels the idle workload sandbox, its cells fall out of the
denominator by a string that is not a floor verdict, and without G-K0 every positive is `pending`,
the denominator is zero and the TPR is silently undefined. Correction: (a) `run_detection_split`
writes `status = not run: gates/detection/gk0_cells.csv missing (run gk0-cells)` and exits 0 when
the floor file is absent; (b) declare `NULL_DENOMINATOR_RULE = "same_as_observed"` with the
sentence "a permuted-positive cell is in the denominator when its verdict is `GK0_ABOVE_FLOOR`;
`control` counts as at floor", record it in params, list it in section 7.

M4. SPEC 3.3.2 `cell_scores`: "a cell with no finite window is NaN and counted". Wrong: a NaN score
is never strictly above a threshold, so a sandbox cell without a score becomes a silent miss and a
benign one a silent true negative; that is a refusal absorbed into a rate. Correction: such a cell
has `score`, `flag_05`, `flag_01` empty, `in_denominator = false`, a new `predictions.csv` column
`score_status` holding `not applicable: no finite window`, its id listed in
`scores.json.excluded_no_score`, and a count `n_without_score` printed in Tables 7 and 8 beside
`n at floor`.

M5. SPEC 3.1.2 and D2: the head drop for non-kernel workload keys is undeclared. Wrong: the head
drop is part of the normalization (`k_median_cell(ex, head_drop)` takes the median after the drop),
`series.head_drop_for` returns 0 for a key absent from `inputs/head_drop.csv` (`series.py` 242 to
249), and `series head-drop-template` writes rows for `schema.KERNELS` and `idle` only (220 to 226),
so the sandbox members' constant exists by omission. Correction: a `gates_detection
head-drop-template` that writes one row per workload key of `cell_classes.csv` (the public keys)
with `head_drop_pairs = 0` and the reason `default 0; declared at D2`; every feature file's
`head_drop_json` and every split's params record the full map; the runbook states that a member's
head drop is the author's number from the program's specification, fixed at D2 and never read off
the APF(t) figure of D5 (a normalization constant chosen from the data is the map learned from the
fold, which condition (5) forbids).

M6. SPEC 3.1.3 and 3.3.9: the ladder's whole-cell row is not the headline's. Wrong: (a) the cap
"the cell's series length (n_pairs - 1)" slices the extract one row short of `_n_rows`, and since
`k_median_cell` uses every remaining row while `rung_series` uses one fewer (`series.py` 255, 286),
the 600 s prefix's median K differs from the whole cell's and the SPEC's own known answer ("tpr05 at
600 s equals the LOWO headline", 4.4) fails by construction; (b) how the head drop composes with the
`from_boundary` start is unstated. Correction: cap the prefix at the extract's `_n_rows` (the series
length then follows from `rung_series`), require in the test that the capped prefix feature file
hashes equal to the headline feature file, and declare `LADDER_HEAD_DROP_RULE = "from_pair1_only"`
(the boundary start replaces the head drop under `from_boundary`) recorded in `ladder.json`.

M7. SPEC 3.3.8, 3.5.6, 3.5.7, 3.5.5, 3.5.15: five thresholds live inline with no constant and no
section 7 entry. Wrong: G-CAL's spread is "p95 - p05 of the tpr05 null"; the drift regression's null
edge is "the 95th percentile of |slope|"; G-V's note fires at "at least half of the features";
`level_of_workload` aggregates cells by "the median"; G-OP's setters use `>=` while flags use `>`.
Correction: `GCAL_SPREAD_RULE = "p95_minus_p05"`, `DRIFT_NULL_PERCENTILE = 95.0`,
`GV_NOTE_FRACTION = 0.5`, `LEVEL_WORKLOAD_AGG = "median"`, `GOP_SETTER_RULE = "ge"`, each a
module constant written into the gate's params and added to section 7.

M8. SPEC 3.5.2: the idle band edge is "asserted <= 1e-9" against `gates/gk0.csv`. Wrong: an
assertion is a crash, not a written refusal, and epoch-1 `gate_gk0` has an `idle_pool` parameter
(`pooled_snapshots` or `cell_medians`, `gates_precondition.py` 236 to 265) that the SPEC's recipe
ignores. Correction: read `idle_pool` and `idle_percentile` from `gates/gk0.json`'s params and
compute the edge by the same rule; on a mismatch write
`refused: idle band edge differs from gates/gk0.csv (<a> against <b>)` into `gk0.json` and into
every cell's verdict, exit 0.

M9. SPEC 5.1 and 3.5.1: unassigned cells are invisible. Wrong: a cell that matches no class row is
`class = unassigned` in the join and "enters no detection stage", but Table 4 prints one row per
class in `CLASSES` (which has no `unassigned`) and 3.5.1 does not say whether admissibility lists
it; a silent exclusion. Correction: Table 4 carries an `unassigned` row with its count;
`admissibility.csv` lists every unassigned cell with `admissible = false` and
`reason = "unassigned: no class row in inputs/classes.csv"`; the D0 row of the runbook says a
non-zero `unassigned_cells` is the author's to resolve before D2.

M10. SPEC 2.7, 6.1, 3.3.6: three refusals have no artifact address. Wrong: `inherit-selection`'s
`refused: gates/selection.json exists and was written by gates_temporal select; pass --force to
replace` and the driver's `refused: inputs/classes.csv exists; edit it or pass --force` are stated
with no file; the one-class guard `refused: a second one-class model needs --secondary` is written
nowhere by design yet wears the refusal vocabulary and exit code 2, which SPEC 7.1 reserves for a
missing input. Correction: `inherit-selection` writes `gates/detection/inherit_selection.json`
with `status` and the refusal string; the driver records its refusal as the D0 command's status in
`driver_detection_state.json`; the one-class guard is renamed `usage: a second one-class model needs
--secondary` (a usage error outside the refusal vocabulary) with exit code 3, documented beside
SPEC 7.1's codes.

M11. SPEC 2.3, 6.2 and 5.2.12: the author's names can remain on disk and in one read field. Wrong:
if `extract all` ran before `apply`, the directories `extract/<pre-apply cell_id>/` keep the
workload names in their names and the runbook only says "re-run D0 and D1"; and every sidecar
carries `path` and `traj_file` (`extract.py` 463, 495), which 5.2.12's never-printed list omits.
Correction: `apply` scans `<out>/extract/` and refuses with `refused: extract/ holds <n> directories
not keyed by a public cell_id; remove them and re-run` (written to `classes_validation.json`); add
`traj_file` and the sidecar's `path` to 5.2.12's never-printed list and to the alias falsifier's
docstring (it reads `dt_est_s` only).

M12. SPEC 3.5.7: the order test's scope is a choice made silently. Wrong: ML 3.2 and CR3 2.20 define
the test as "first half against second half of the campaign under LOWO"; the SPEC runs it within
each class, which is the right reading for a member-blocked stage 1 (the campaign-wide halves
coincide with the classes) but is a departure from the definition's letter. Correction:
`ORDER_TEST_SCOPE = "within_class"` (alternative `"campaign"`) as a constant and CLI flag, recorded
in `order.csv`'s params with the stage-1 note, listed in section 7 with the sentence that the
campaign-wide reading is void by construction on stage 1.

## 3. For the author

1. G-OP's default count (section 7 item 18). ML 2.3's own example, "one benign workload whose eight
   cells all fire gives 8 of 104", is about the realized false positives, not the training cells that
   set the threshold; under `threshold_setters` the union over 21 folds almost always clears five
   cells from three workloads, so the gate rarely bites. Both counts are written either way. My
   recommendation is `realized_fps` as the count the verdict reads; your decision.

2. Whether cells at floor train (M2's default). An at-floor cell is "undetectable by construction"
   (ML 3.6); keeping it as a positive training row teaches the forest that the floor is sandbox and
   pushes the idle cells toward false positives. My recommendation is `TRAIN_ON_AT_FLOOR = False`
   (scored, reported, out of training and out of the null's pool); the ML engineer should weigh in
   before you fix it.

3. `det_c1_rule = "report"` reads the class: a kernel cell that fails C1 is refused, a sandbox cell
   that fails C1 is kept and labelled by G-K0. A uniform rule (C1 fail reported for every class,
   the floor verdict from G-K0 for every class) reads no class label at the inclusion step and
   leaves a benign occupant at floor other than idle in the benign world. E1 6.7 says the 0.02
   default refuses several real kernels; decide the rule and the threshold before D2 and pass them to
   both drivers.

4. The head drop for the eight members (M5): declare it at D2 from what you know of each program's
   specification, default 0; never from the APF(t) figure of D5.

5. The campaign token for the sandbox ids (M1): `stage1` by default, or a token you choose; the raw
   launch label is never copied.

6. The order test's scope (M12): within class for stage 1 as the SPEC says; the campaign-wide reading
   becomes meaningful only for an interleaved stage 2.

7. What stays on disk under `<out>` with your names on it: `inputs/classes.csv`, `cells.csv`,
   `cells.pre_classes.csv`, `cells.index.json`, and every `extract/<cell_id>/sidecar.json` (`path`,
   `traj_file`). None is printed by any table and the manifest hashes none of them, but `<out>` is
   not deposit-clean; the anonymisation map of CR3 1.8 must cover these five before any release.

8. Under `kernel_family_rule = "tier"` LOFO has two folds and the kernels' fold trains on idle plus
   sandbox only, so the kernels will fire almost entirely; that is the definition's reading of "does
   an unseen benign family fire" and section 7 item 3 says so, but read the row as a size, not a
   failure, until stage 2 supplies other benign families.
