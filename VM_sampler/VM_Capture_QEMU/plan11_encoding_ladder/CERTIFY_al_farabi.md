# CERTIFY_al_farabi.md: certification of the built toolkit (form against realization), cycle 3

Written 2026-09-16 by al-Farabi. This version supersedes the cycle-2 version written at 20:33 the
same day. Since that version the fixer applied FIX_2 (among its items the two corrections I proposed
for the resume path, 7.1 and 7.2, and the checker's B1 on which score is the rung's score after a
B1-G3 quarantine) and the checker wrote CHECK_3 with no blocking finding. The modules under
certification are the ones on disk now: `gates_temporal.py` and `series.py` (unchanged since 18:35
and 15:43), and `run_moves.py`, `models.py`, `gates_comparison.py`, `_report_common.py`, `tables.py`,
`figures.py`, `gates_calibration.py`, `gates_readings.py`, `latex_skeleton.py` as edited by FIX_2
between 20:41 and 20:44. Every line number below refers to those files as they stand.

Scope: the five conditions named in the task, checked against the code as written under
`plan11_encoding_ladder/` and against SPEC.md, my SPEC_review_al_farabi.md, BUILD_gates.md, FIX_2.md
and CHECK_3.md. No server was touched and no path under `/mnt/nfs` or `/project` was read. Every
empirical claim below comes from synthetic data written by the toolkit's own generator. I made one
fresh run of my own: the driver's moves 0 to 7 on the checker's synthetic corpus (four kernel presets
by three reps plus two idle cells, 120 pairs per cell, compressed with the `zstd` binary), with the
preconditions re-run by hand at `--c1-activity-min 0.001` so that every kernel cell is admissible,
the pass table filled for gemm and floyd from the generator's declared periods, the idle
admissibility record filled, and `--null-perm 5` so that the run fits the session; exit 0 in 9 min
38 s, all 27 ledger entries `done`, `skipped` or `kept`. On that output I ran five resume probes in
dry-run and one for real. For the artifacts of moves 9 to 13, which cost the better part of an hour
to regenerate, I audited the checker's completed run 2 (the `check3/out2` directory in the session
scratchpad), which I did not produce but can read; every claim from it is marked as such. The sandbox
family is not named, read or needed anywhere in this certification. Nothing was committed and no
file outside the toolkit directory was edited; this file is the only file I wrote.

## Summary

| Condition | Verdict | Where it lives | Residual marks (none is a fail) |
|---|---|---|---|
| (1) Every (W, H) point computed and written before any selection; the selection rule a declared function; the selected point marked, not the only one kept | Holds, in the fresh path and in the resume path. Both cycle-2 fails are fixed and verified on my run: a preconditions re-run marks every step from move 3 stale (probe D), and a rebuilt grid re-runs G-ORD before the selection reads its label (probe C, for real) | `gates_temporal.py` 290-359, 555-650; `series.py` 653-666; `run_moves.py` 73-81, 152-173, 314-320, 326-346, 401-407, 414-428 | the driver's admissibility trigger is `preconditions.json`, a proxy for the CSV columns admissibility reads (probe F); `select` reads `gf.csv` undeclared and copies a G-F label into `table5_grid.csv` that no table prints (`gates_temporal.py` 571, 609); the whole-file hash of `selection.json` re-runs four split stages on a plain resume (CHECK_3 M2, conservative) |
| (2) No gate reads an analytical interpretation to fix anything upstream of the extract | Holds | every write site enumerated again after FIX_2; the two new sites are `models.py` 510 and 512, both under `gates/splits/` | none |
| (3) Every refusal is written to the artifact | Holds. The three markers carried by a flag or a default rather than by a refusal string stand unchanged from cycles 1 and 2 | `verdicts.py`; every `gate_*` writes its rows before returning; `models.py` 341-370 (the every-feature-quarantined case now carries its string to every consumer), 432-450, 513-529; `gates_comparison.py` 91-104, 239-242 | `gates_temporal.py` 597-600 and 626; `models.py` 680; a void from a five-draw null prints as a finding (CHECK_3 M3) |
| (4) (W, H) chosen per encoding, never per kernel, in the code as written | Holds | `gates_temporal.py` 611-628, 635-638; `series.py` 568-579; every caller of `selected_grid_id` passes a rung (twelve sites, listed); `models.py` 236-271, 387-413 | none |
| (5) The level normalization of G-L is fixed per rung before the data and identical in every fold | Holds. FIX_2's B1 change does not touch the normalization: the re-run after a quarantine is a column subset of the same stored matrix, and every consumer now reads one score | `series.py` 252-256, 275-319, 588-640; `models.py` 141, 236-271, 478-490; `gates_comparison.py` 46-51, 94-104, 109, 126 | `wapf_norm` half exposed (`series.py` 687 against `gates_temporal.py` 290-291, 304), unchanged |

Nothing must be fixed before the toolkit runs on the real corpus under these five conditions. The
items in section 6 are choices the definitions leave open or hazards outside the toolkit's own paths;
I decide none of them.

---

## 1. Condition (1): the grid before the selection, the rule declared, the selected point marked

**The Form.** Orchestration of the resolution axis: every declared resolution is realized as an
artifact before any resolution is chosen; the choice is a declared function of those artifacts and
of nothing else; the chosen resolution is marked among its siblings, which remain. A realization
satisfies this invariant when a reader who arrives after the run can recompute the choice from what
is on disk and obtain the same answer, and when every artifact the choice was computed from is the
artifact that corresponds to the declared inputs at the time of reading. A realization violates it
when the choice can be made before some sibling exists, when the choice reads something that is not
on disk, when the chosen point is the only one that survives, or when the artifacts the choice was
made from have gone stale against a declared input without the run noticing.

**Where it lives.**

- The grid is declared before the data: `schema.py` `GRID_WINDOWS`, `GRID_HOP_RATIOS`, `grid_id`,
  `grid_points`; `series.py` 351-363 (`grid_points_ids`, asserting thirteen).
- Every point is computed and written before the verdicts are rolled up: `gates_temporal.py` 303-304
  builds the raw and normalized feature files of all thirteen points (`series.build_all_grid`,
  `series.py` 653-666); 317-358 loop over the thirteen points and write `temporal_per_kernel.csv`,
  `g1_surrogates.npz` and `temporal.params.json` for each.
- The selection rule is a declared function: `gates_temporal.py` 555-650 (`select`). Inputs: the
  thirteen per-kernel CSVs (573-581), the G-K0 relabel set (569), the G-C verdict (570, used only
  for the `refusal` and `gc_verdict` columns, 608) and G-F part (i) (571, a column only, 609). The
  rule is 611-620: candidates are the integer-W points (611); the smallest W whose point passes every
  applicable gate among G1, G2, G4, tie broken by the hop ratio nearest 0.5 (612-616); else the point
  passing the most gates by the same tie-break, `best-feasible` with `passes_acceptance = false`
  (617-620). G-ORD enters as a label only (596, 606). The two declared parameters `rollup` and
  `rollup_kernel_refusals` are written into `selection.json` `params` (627-628).
- The selected point is marked, not kept alone: 621-623 write `selected` or `selected:
  best-feasible` into one row; 631-634 merge all thirteen rows per rung into `table5_grid.csv` and
  every per-kernel row into `table5_long.csv`, replacing only the rung's own rows; 635-643 write
  `selection.json` with, under `params`, `inputs_sha256_<rung>` over the thirteen grid CSVs plus
  `gk0.csv` and `gc.csv`; 644-647 write `grid_complete.json`. The report layer prints every row and
  the mark (`tables.py` 121-193).
- The driver refuses the split stage unless all thirteen CSVs and all feature files exist:
  `run_moves.py` 326-346 (`grid_check`) and 424-427 (stop with exit 1).
- The resume rule: the driver records the hash of every declared input when a step starts
  (`run_moves.py` 392), and skips a step only when its outputs exist and every declared input hashes
  as recorded (401-407, `_stale_reason` 314-320; an absent file hashes as `absent`,
  `_report_common.py` 323-330).
- The two cycle-2 corrections, as applied. 7.1: the admissibility record is a declared input of every
  step from move 3 on (`run_moves.py` 79 `ADMISSIBILITY = "gates/preconditions.json"`, used at
  140-147, 149-150, 154-156, 159-160, 165-166, 170-173, 180-183, 190-195, 197-199, 208-211, 222-232,
  241-243). The file declared is the JSON and not the CSV, for the reason the comment at 73-78 gives:
  `_refresh_c7` rewrites the CSV's C7 column in place after `select apf` (`gates_temporal.py`
  648-649, 653-665), which would mark every step that ran before move 6 stale on the first resume
  although the admissible set had not changed. I verified the proxy: `gate_preconditions` writes the
  JSON's `excluded_cells` and `excluded_cells_pair_rungs` from the same `all_hard_pass` and
  `failed_verdict` values it writes into the CSV (`gates_precondition.py` 182-188, 199-201);
  `admissible_cells` reads exactly those two columns (`series.py` 533, 536); `write_json` adds no
  timestamp (`series.py` 127-136); the only two writers of `preconditions.csv` are
  `gate_preconditions` (189) and `_refresh_c7` (665), and the second touches C7 only. 7.2: the
  `gord` step's inputs are the `grid` step's inputs plus `gates/gk0.csv` (`run_moves.py` 154,
  165-166), so any trigger that rebuilds the grid also re-runs G-ORD before `select` reads the label;
  `tests/test_driver.py` 100-114 asserts both properties and passes.

**Verified on my run (fresh path).** After move 6: thirteen directories under `gates/grid/apf/`,
twenty-six feature files under `features/apf/`, `table5_grid.csv` with thirteen APF rows and
`table5_long.csv` with sixty-five rows; exactly one row marked `selected: best-feasible` (W8_H4,
`gates_passed: 2 of 3`; G2 in seconds fails at every W on this corpus because the generator declares
600 s for 120 pairs, CHECK_3 M12, a property of the corpus and not of the code); `selection.json`
holds the key `apf` only, with `params.inputs_sha256_apf` listing fifteen files;
`grid_complete.json["apf"].complete` is true and the driver's `grid_complete.json["grid_complete"]
["apf"].verdict` is `complete`; `selection.json` was written 0.3 s after the last grid CSV, which
G-ORD rewrote; the GORD column carries `resolution` at nine points, `order-blind` at three and
`order-blind (by construction)` at the whole-cell point, with `gord.json` at W8_H4 agreeing with the
CSV; the C7 column of `preconditions.csv` reads `fail` on all fourteen cells after the refresh. On the
checker's run 2 (moves 0 to 13) I recomputed the selection of every rung from `table5_grid.csv` alone
by the rule above and obtained the same grid id and the same `passes_acceptance` as `selection.json`
and the same row marked in the CSV for all five rungs; the `content` rung's rows show why W16_H8 and
not W8_H4 (G1 fails at W = 8 for that rung, so W = 16 is the smallest W among the `2 of 3` points and
H = 8 is the hop nearest 0.5).

**Verified on my run (resume path).** Every probe is a driver invocation of moves 2 to 7 (or 6 to 7)
with `--dry-run`, which records the skip or the stale reason in the ledger without running anything,
except probe C, which was run for real.

- Probe A, nothing changed: 23 steps `skipped: outputs exist and inputs unchanged`, 4 templates
  `kept: author input exists`. The C7 refresh at move 6 did not make moves 3 to 5 stale.
- Probe D, the preconditions re-run by hand at the default C1 (5 of 14 cells admissible instead of
  14): 18 steps `stale: preconditions.json changed since move <n>`, namely every `gc`, `gk0`, `gf`,
  the move-5 figures, `features`, `grid`, `g3`, `gord`, `select`, `splits`, `gx`, `gl`, `gn` and
  `tables table6`; 5 skipped (the preconditions step itself, whose declared inputs had not changed,
  and the internal and template steps). This is the cycle-2 fail (a), closed.
- Probe D', the preconditions restored at 0.001 with the same `--assume-reason`:
  `preconditions.json` and `preconditions.csv` byte-identical to the copies taken before probe D,
  and the dry-run shows 23 skipped and 4 kept, as in probe A. The JSON is a stable trigger.
- Probe C in dry-run, one `notes` field of `inputs/pass_table.csv` edited: `grid apf`, `gord apf`,
  `select apf` and both `alias` steps stale on `pass_table.csv`; `features apf` and `g3 apf` skipped
  (neither reads the pass table). Then for real, `run --moves 6`: `grid apf` done (stale), `g3`
  skipped, `gord apf` done (stale, 7 min 47 s), `select apf` done (stale), `alias` done (stale); exit
  0. The rebuilt `table5_grid.csv` carries G-ORD's labels in every row and not one `pending:
  gate_gord` cell exists across the thirteen per-kernel CSVs; the rebuilt `table5_grid.csv` and the
  `apf` entry of `selection.json` (including `inputs_sha256_apf`) are identical to the copies taken
  before the probe, as they should be when only a notes field changed. This is the cycle-2 fail (b),
  closed. The pass table was restored afterwards.

**Residual marks (none a fail).**

(a) *The trigger is a proxy for the file that is read.* Probe F: I edited `preconditions.csv` by
hand, setting one kernel cell's `all_hard_pass` to `false`, and left the JSON untouched; the dry-run
of moves 3 to 7 reported 25 steps skipped. Admissibility had changed and the driver could not see it.
No toolkit path produces this state (the two writers of the CSV are listed above), so it is a hazard
for an author who edits a gate result by hand, which the driver's contract does not cover (its
declared inputs are `cells.csv` and `inputs/*`, `run_moves.py` 80-81). A stronger trigger would hash
the CSV with its C7 column blanked; I put that in section 7 as an option, not a requirement. The CSV
was restored after the probe.

(b) *A G-F label copied into the grid's artifact without a declared channel.* `select` reads
`gates/gf.csv` (`gates_temporal.py` 571) and writes the part (i) verdict at the matching (rung,
grid_id) into the `gf_part1` column of `table5_grid.csv` (609). `gf.csv` is not among the `select`
step's declared inputs (`run_moves.py` 172-173) nor among the files `select` hashes into
`selection.json` (639-640). At move 6 the only G-F rows that exist are move 4's at the default point
W8_H4; move 12's re-run at the selected points comes after every `select`. On the checker's run 2 the
column therefore carries a value at W8_H4 for all five rungs and nothing at the selected W16_H8 rows
of `content` and `combined`. This is the same class as the cycle-2 G-ORD defect (one role's label
written into another role's channel without the driver knowing), but it has no consumer: Table 5 and
Table 7 read `gf.csv` directly through `gf_part1_verdict` with the selected grid id (`tables.py` 143,
523, 554; `_report_common.py` 521-547), and `rung_override` does the same (540-550). The choice is
unaffected because G-F is off the decision. A reader of `table5_grid.csv` could nonetheless see a
G-F cell that move 12 has since replaced. Section 7 item 2 gives the one-line forms.

(c) *Files that accumulate across moves make later steps stale for no change* (CHECK_3 M2; FIX_2
section 5). `splits <rung>`, `gx <rung>`, `gl` and `tables table6` declare `gates/selection.json`
whole (`run_moves.py` 181, 183, 190, 195), which `select` of each later rung extends; the two `alias`
steps declare `gates/g3_flags.csv` whole (187, 193). On a resume of moves 7 to 13 with nothing
changed the four split stages re-run at full cost and produce the same files at the same seeds. This
is the opposite failure from the one the invariant forbids: the driver re-runs when it need not,
never skips when it must. It is a cost, and on the real corpus a large one (hours). CHECK_3's fix
(per-key hashing with the syntax the outputs already use) is the right shape; I list it under the
author's items because it changes the driver's dependency language.

(d) Unchanged small marks from cycles 1 and 2: `select` continues over a missing grid CSV and chooses
among the points that exist (578-579), the gap being recorded in `grid_complete.json` and caught by
the driver's `grid_check` before the split stage, but `selection.json[rung]` carries no sign of it;
the `grid` CLI's `--no-features` flag (675) is an author-side foot-gun the driver never uses.

**Verdict.** Holds, in the fresh path and in the resume path.

---

## 2. Condition (2): no interpretation reaches upstream of the extract

**The Form.** Separation of Analysis from Acquisition and Representation: the extract and the
author's declared inputs are the boundary; an analytical reading (a score, a prediction, a fitted
parameter, a verdict) may organize what lies downstream and may never redefine what lies upstream. A
realization satisfies this invariant when no code path that reads a score, prediction, fitted
parameter or verdict writes to the retention root, `cells.csv`, `extract/`, or the author's
`inputs/`. A realization violates it when a gate's outcome changes a capture parameter, a cell's
identity, a head drop, a pass count, an admissibility decision, or the extract itself.

**Where it lives.** I enumerated every write and delete site again (`write_csv`, `write_json`,
`write_text`, `write_params`, `np.savez`, `open(..., "w")`, `savefig`, `unlink`, `rename`, `replace`,
`rmtree`, `shutil`, `os.remove`) across the gate modules, the report layer, the driver and the
extractor, since FIX_2 edited nine files. Every gate result lands under `gates/` or `features/`
(`gates_precondition.py` 189-199, 289-293, 448-461; `gates_calibration.py` 181-182, 373-374, 465-466;
`gates_temporal.py` 347-357, 396-398, 499-512, 631-647, 665; `gates_readings.py` 109-110, 361-362;
`gates_comparison.py` 139-143, 181-182, 255-260, 303-306, 341-342, 364-367; `variance.py` 77-79;
`models.py` 433-441, 508-512, 520-532, 541-548, 598-599, 631-632; `series.py` 632). The two sites
FIX_2 added are `models.py` 510 (the re-run's predictions beside the full model's) and 512 (the
removal of a stale `predictions_with_quarantine.csv` when a later run of the same split quarantines
nothing); both are inside the split's own directory under `gates/splits/`, and the second removes an
artifact the same stage wrote, so that a superseded reading does not outlive its cause. The report
layer writes only under `report/` (`_report_common.py` 268, 286, 446-450; `tables.py` 84, 1005;
`figures.py` 79-80, 660-662, 701; `latex_skeleton.py` 440, 447) and, when the author asks with
`--standalone`, the skeleton to the path the author names (445). The driver writes
`driver_state.json`, `gates/grid_complete.json` and `gates/gf_check.json` (`run_moves.py` 270, 345,
364). The only writes into `inputs/` are the four template writers, each writing fixed constants
from `schema.py` or P2 Table 3 column 5 and reading no trajectory, extract, score or verdict
(`series.py` 220-226; `gates_calibration.py` 61-74; `gates_precondition.py` 214-218, 298-306), and
the driver never overwrites an existing author input (`run_moves.py` 397-399). `extract.py` and
`schema.py` read nothing under `gates/` (grep); the extractor's own `unlink` calls (492, 508-514)
remove its temporary files and a failed extract.

The places where an interpretation is read, all downstream of the extract and unchanged in kind
since cycle 2: G-ORD's LOKO score becomes a label column (`gates_temporal.py` 501-512; read by
`select` as a label only, 596); G-K0's level verdict relabels the archetype at read time
(`series.py` 543-553; `models.py` 254-257) and the feature files store the predicted archetype only
(`series.py` 622); G-L, G-DIM and G-M read `scores.json` through `effective_scores`
(`gates_comparison.py` 46-51, 94, 282, 358); B1-G3 reads predictions and re-runs the split with both
results kept (`models.py` 478-490); `_refresh_c7` rewrites the C7 column of `preconditions.csv`
after `select apf` (`gates_temporal.py` 653-665), and admissibility reads the `all_hard_pass` column,
which is set at move 2 from C1, C2 and C6 (`gates_precondition.py` 182-183) and never touched by the
refresh (`series.py` 533), so no verdict changes admissibility; the alias falsifier reads the G3
peaks and emits a label; the exclusion list now follows the re-run's B1-G1 verdict (`models.py`
513-519), a record under `gates/`.

**Verdict.** Holds. The operational hazard of cycles 1 and 2 stands: the template CLIs overwrite an
existing author input when run by hand outside the driver.

---

## 3. Condition (3): every refusal written to the artifact

**The Form.** Representation's permission to organize but never to redefine: a gate's refusal is a
record, not an absence. A realization satisfies this invariant when every path that ends in a
refusal ends by writing a string from the declared vocabulary into the result file, and when the
tables print that string rather than a number or a blank. A realization violates it when a refusal
returns early without a write, is replaced by a default, or is printed as a score.

**Where it lives.** `verdicts.py` 20-36 (the four constructors), 108-113 (the named refusals),
123-134 (`is_refusal`). Every gate writes its rows before returning: the preconditions and the
`failed/` count (`gates_precondition.py` 189-199), G-K0 (289), G-F including the `not run` rows when
the admissibility record or the idle cells are missing (403, 416, 453-461), G-C (`gates_calibration.py`
373-374), G-P (181), the alias falsifier (465), the grid with every G1, G2, G4, G5 string per kernel
(`gates_temporal.py` 347), G3 (396), G-ORD (499, `not run: fewer than two archetypes with windows`
at 470-471), the roll-up (631-634, `not applicable` strings from 542-543), the selection entry with
`not run: no grid point computed` when nothing was computed (626), G-J and G-DEC
(`gates_readings.py` 109-110, 361), the comparisons with `not run: no selection for <rung>`
(`gates_comparison.py` 54-55, 91-93, 105-107, 239-240, 279-281, 340-342), G-V (`variance.py` 77-78),
the clustering (`models.py` 596-601), and the split stage (`models.py` 432-450 for the three `not
applicable` cases, 498 and 502-506 for `not run: no windows`, 513-519 for `near_unfalsifiable` into
`excluded_rows.csv`, 520-529 into `scores.json`; the `not run: every feature quarantined` case at
490). The report layer replaces score cells with the refusal string for a disconnected or void rung
and for `near_unfalsifiable` (`_report_common.py` 540-550, 622-635), and prints `not run: ...` for a
missing point.

What FIX_2 changed here, verified: `effective_scores` (`models.py` 341-370) is the one reading of
`scores.json` for the comparison gates (`gates_comparison.py` 46-51), the tables
(`_report_common.py` 593-619, delegating to the same function) and the exclusion list (`models.py`
514-519). When every feature was quarantined and no re-run exists, it blanks the score keys and
carries `not run: every feature quarantined` in `b1_g1` (360-363), so the full model is printed and
judged nowhere; `gate_gl` tests the `not run` string before the missing-score test (97-101), so a
smoke run writes its own string rather than `scores.json missing`. G-X with one campaign label writes
`not applicable: one campaign label` in both the leak and the confound columns (241-242). Table 5's
refusal cell no longer duplicates a disconnected lead (`tables.py` 183).

**Verified on my run.** `gl.csv` carries `not run: no selection for <rung>` for the four rungs
without a selection at move 7 and `not run: 5 permutations < 500` for APF part (i) with
`score_norm 0.583` beside it; `gx.csv` carries `not applicable: one campaign label` in both columns;
`gf.csv` part (i) carries `void: idle reps separable under this rung` for apf and persist and
`inseparable at floor` for the other three; every `scores.json` of the ten APF split directories
carries `b1_g1: not run: 5 permutations < 500`; Table 6 prints the G-F void string in every score,
null and rank cell of every row (the override of `_report_common.rung_override`), and its `all` row
does not print a number. On the checker's run 2 the same strings appear where the same conditions
hold, `excluded_rows.csv` is absent (no `near_unfalsifiable` at five permutations), and the `refusal`
column of `table5_grid.csv` carries only `GC_DISCONNECTED` when it carries anything
(`gates_temporal.py` 608).

**Three markers carried by a flag or a default rather than by a refusal string** (unchanged since
cycle 1; the fixer refused them as more than one line in both cycles):

- `gates_temporal.py` 626: when the acceptance rule fails and G-C passed, `selection.json[rung].refusal`
  is the empty string; the failure is carried by `passes_acceptance: false` and `selected_by:
  best-feasible`. Nothing is lost, but the field named `refusal` does not say why acceptance failed.
- `gates_temporal.py` 597-600: `applicable` always includes G1, so `not applicable: no applicable
  kernel` for G1 counts as a failed gate, while G2 in the same state is dropped from the count
  (CHECK_3 M7).
- `models.py` 680: the `splits` CLI falls back to `W8_H4` when the rung has no selection
  (`selected_grid_id` with its default), and `run_split_stage` does not record a `grid_source` in
  `params` (415-430), so a defaulted run is indistinguishable from a selected one. `run_clustering`
  does record it (594). The driver's order never triggers the fallback.

**One mark on the spirit of the condition** (CHECK_3 M3, the checker's finding, which I confirm on my
run): G-F part (i), G-X and the clustering judge on whatever permutation count they are given, while
B1-G1 refuses below 500 with `not run: N permutations < 500`. On my run a five-draw null voided APF
and persistence, and the void then replaced every score cell of Table 6. The refusal is written and
printed, so the letter of the condition holds; but a void asserted on a null that the sibling gate
refuses as under-powered is a claim, not a refusal of one, and on a smoke run it hides the numbers
the runbook promises to show. At 500 permutations nothing deviates. The author's item 2 of CHECK_3
covers it; I add nothing to the decision.

**Verdict.** Holds: no refusal path returns without writing its string.

---

## 4. Condition (4): (W, H) per encoding, never per kernel

**The Form.** The resolution is a property of the encoding, chosen once for all targets it will be
asked to separate; a resolution chosen per target would carry the target's identity into the
representation. A realization satisfies this invariant when the selection artifact is keyed by
encoding alone and every downstream stage resolves one (W, H) per encoding for every cell it
touches. A realization violates it when any stage looks up a grid point by kernel, routes cells to
different resolutions by a verdict, or lets a kernel-dependent quantity stand in for W.

**Where it lives.** `select(out, rung)` writes one entry per rung (`gates_temporal.py` 635-638);
the per-kernel rows are the input to a per-rung roll-up (535-552, 582-600), never a per-kernel
choice. `series.selected_grid_id(out, rung, default)` (568-579) is keyed by rung and every caller
passes a rung: `gates_calibration.py` 411, `gates_comparison.py` 91, 105, 222, 279, 289, 338, 355,
`gates_precondition.py` 388, `models.py` 591, 680, `variance.py` 53 (grep, no exception; the report
layer reads `load_selection(out).get(rung)`, `_report_common.py` 471-483, `tables.py` 777).
`run_split_stage(out, rung, grid_id, ...)` takes one grid id and `prepare_split_data` loads the one
feature file for it (`models.py` 243-246); admissibility, role and relabel are applied by masks
(249-257), never by re-windowing; the B1-G3 re-run is a column subset of the same matrix (485). The
split stage never routes a cell by its G1 verdict: `TREND_PRESENT` is read only in `_applicable`
(`gates_temporal.py` 526). At the whole-cell point W resolves per cell to `n_series` (`series.py`
374-377), but `n_series_cell` and `win_start` are stored beside `X` and never enter it (`series.py`
632-639), and the eight shape features are length-free. G-F's pre-selection point is the declared
per-rung constant `W8_H4` (`run_moves.py` 146). G-DIM's matched re-run takes `d*` from the
strongest single rung's feature count, a per-encoding quantity (`gates_comparison.py` 287-295).
G-M's seed re-runs use the APF rung's selected point (338, 346-347). G-X runs at the rung's selected
point in its own base directory (222, 244-246).

**Verified.** On my run `selection.json` has one entry, `apf`, and every one of the ten split
directories under `gates/splits/apf/W8_H4/` records `params.grid_id = W8_H4` and hashes the one
feature file of that point (`features/apf/W8_H4_norm.npz` for the five normalized runs,
`W8_H4_raw.npz` for the five raw). On the checker's run 2 `selection.json` has five entries keyed by
rung and nothing else.

**Verdict.** Holds.

---

## 5. Condition (5): the level normalization of G-L, fixed per rung, identical in every fold

**The Form.** Level blindness is a property of the representation, declared before the data and
applied identically to every unit; if the map that removes level were learned from a training fold,
the fold would carry level back in through the map. A realization satisfies this invariant when the
normalization is a per-cell function of declared per-cell statistics, computed once and stored
before any split, so that every fold reads the same normalized values, and when the raw
(level-inclusive) reading that G-L (i) is measured against survives beside it. A realization
violates it when the normalization is fitted on a training fold, when its constants change between
folds, when a level-bearing quantity is introduced after the normalization, or when the raw ceiling
is overwritten.

**Where it lives.** `series.py` 252-256 (`k_median_cell`: the median of the cell's own K after the
kernel-keyed head drop) and 275-319 (`rung_series`): apf `K / K_median_cell`; wapf
`ham_sum_all / (K_median_cell * 32768)` under the default `wapf_norm = "median_K"` (or the cell's
own median wapf under `median_self`); persist `J - J_null`; content the fifteen `r_*_per` columns,
level-free by construction; combined the concatenation of the four. The head drop is an author input
fixed before the data (`series.py` 229-247; `inputs/head_drop.csv`). The normalized feature matrix
is built once per (rung, grid point) and stored (`build_features`, 588-640) before any fold exists;
`prepare_split_data` (`models.py` 236-271) reads it and applies only masks, relabels and column
drops; `fit_predict_units` (127-161) slices `X[tr]` and `X[te]` from that one array (141). Every fold
of every split therefore sees byte-identical normalized values. G-L part (i) reads the normalized
LOKO score of that file (`gates_comparison.py` 94-104); part (ii) reads the raw APF feature file
(109) and the same `K_median_cell` (126). G-ORD and G-F part (i) use the same `rung_series` and
feature files (`gates_temporal.py` 446; `gates_precondition.py` through `series.features_path`).

What FIX_2 changed in this neighbourhood, and why it does not touch the normalization: the re-run
after a B1-G3 quarantine is `run_point(X[:, keep], ...)` on the same loaded matrix (`models.py`
482-485), so the re-run's score is computed on the same normalized values with columns removed; the
re-run's per-unit predictions are written beside the full model's (508-510); `effective_scores`
selects which of the two scores the consumers read (341-370) and computes nothing. G-L (i) now reads
the re-run's score when a feature was quarantined (`gates_comparison.py` 46-51, 94), the same number
Table 7 prints; the cycle-2 disagreement (CHECK_2 B1) is closed.

Model-side operations fitted per fold, on already level-normalized columns, and therefore not the
level normalization of G-L: the median imputer and the `StandardScaler` of the forest pipeline
(`models.py` 51-66, the plan04 and `b1_ae.py` rule), and the train-fold feature reduction when d
exceeds the training cell count (144-150). None can reintroduce per-cell level because per-cell
level no longer exists in their input.

**Verified on my run.** For every cell and every window of `features/apf/W8_H4_norm.npz`,
`W32_H16_norm.npz` and `Wall_Hall_norm.npz` I recomputed `K / K_median_cell` from `extract.csv` and
the head-drop file and compared the window mean with the stored `apf.k_over_med.mean`: maximum
absolute difference 0.0 over 392, 84 and 14 windows; each file records `wapf_norm = median_K` and
stores the predicted archetype only. On the checker's run 2 the same recomputation for apf, wapf
(`ham_sum_all / (K_median_cell * 32768)`) and persist (`J - J_null`) at W8_H4 gives a maximum
absolute difference of 0.0 over 70 windows each. After move 7 on my run, `gates/splits/apf/W8_H4/`
holds ten directories, five `__raw` beside five normalized, with `params.normalized` false and true
respectively, each hashing its own feature file. G-L (i) read `score_norm 0.583` from the normalized
LOKO file after a two-feature quarantine, the same number `models.effective_scores` and
`_report_common.effective_scores` return for that file; part (ii) read `r2 0.21` over four kernels
and wrote `pass`. On the checker's run 2, for all five rungs, `gl.csv` `score_norm` equals
`effective_scores(scores.json)["accuracy"]` from both readers (0.583, 0.417, 0.417, 0.5, 0.5) and
Table 7's LOKO accuracy cell prints the same number where the G-F (i) void override does not
replace it (wapf 0.417, content 0.500, combined 0.500).

**One hazard on the exposed alternative** (unchanged). `wapf_norm = "median_self"` is a parameter of
`series features` (`series.py` 687) but not of `gates_temporal grid`, which rebuilds every feature
file with the default (`gates_temporal.py` 290-291, 304). An author who chose `median_self` by hand
would have it silently reverted at the next `grid`; the driver never passes it, so the default path
is consistent. The parameter is only half exposed.

**Verdict.** Holds.

---

## 6. For the author

Each item is a choice the definitions leave open, or a hazard this certification surfaced outside the
toolkit's own paths. I decide none of them.

1. **The staleness trigger is a proxy** (section 1 (a), probe F). The driver watches
   `gates/preconditions.json`; admissibility reads `gates/preconditions.csv`. The two move together
   under every toolkit path, and a by-hand edit of the CSV's `all_hard_pass` or `failed_verdict`
   column is invisible to the driver. Either state in the runbook that gate result files are never
   edited by hand (a re-run of the preconditions with a different flag is the sanctioned path), or
   apply section 7 item 1.
2. **`gf_part1` in `table5_grid.csv`** (section 1 (b)) is a copy taken at `select` time from move 4's
   default-grid run; after move 12 it is stale or empty at the selected point while every table reads
   `gf.csv` directly. Say whether the column stays (documented as "at select time") or goes; section
   7 item 2 gives both forms.
3. **The whole-file hash of `selection.json` and `g3_flags.csv`** (section 1 (c); CHECK_3 M2) re-runs
   four split stages on a plain resume of moves 7 to 13. It never skips wrongly. Choose between the
   per-key hashing CHECK_3 describes and a runbook sentence that a resume after move 8 costs the split
   stages again.
4. **`rollup_kernel_refusals`** (`gates_temporal.py` 47; `select --rollup-kernel-refusals`):
   default `not_applicable`, alternative `blocks`, written into `selection.json` `params`. On the
   real corpus this parameter decides whether any integer-W point can pass acceptance (my review
   item 2.1). Choose it before the data.
5. **`c1_activity_min`** (`gates_precondition.py` 28; CHECK_3 M1): on this corpus the default 0.02
   admits one kernel of four; by P2 Table 3's footprints it refuses at least six of the twelve real
   kernels before any gate runs. The staleness side of a by-hand re-run is now correct (probe D); the
   threshold itself is the author's, and a driver flag for it is three lines (CHECK_3 M1).
6. **A `best-feasible` selection with no applicable kernel** is still a selection (section 3;
   CHECK_3 M7). Say whether `select` should refuse outright when `n_applicable` for G1 is zero, and
   whether G1's `not applicable` should be dropped from the count as G2's is.
7. **The permutation floor on G-F (i), G-X and the clustering** (section 3; CHECK_3 M3): apply
   B1-G1's 500 rule to every gate that cites the label-shuffle null, or leave them judging on whatever
   `--n-perm` gives and never read a smoke run's void.
8. **`part1_consequence`** for G-F (`gates_precondition.py` 45): default `void`, alternative
   `report`; my review section 3 item 2 explains why `void` from a within-trace window-level test may
   void a rung for the wrong reason. On both runs of this cycle two idle reps at chance were voided by
   a five-draw null.
9. **The G-L (ii) re-run** promised by SPEC 3.7.4 is the author's manual step (RUNBOOK.md move 7;
   CHECK_3 M10). Either add the conditional driver command or keep the runbook sentence.
10. **Which score after a B1-G3 quarantine**: FIX_2 applied SPEC 3.7.2's reading (the re-run)
    everywhere, through one function; verified on both runs. Unchanged unless you prefer the full
    model, in which case the change is confined to `models.effective_scores` and SPEC 3.7.2.
11. **The idle cells' stored archetype is `control`** (`schema.py`; visible in every feature file on
    both runs) while builder 2's docstrings say `IDLE` (`series.py` 595); the split stage masks idle
    cells by role, so nothing I certified depends on it. Unverified consequence for any consumer that
    reads the archetype of an idle row; I list it so it is not lost.
12. **The template CLIs overwrite** an existing `inputs/*` file when run by hand. The driver keeps an
    existing author input (`run_moves.py` 397-399); the module CLIs do not.
13. **`wapf_norm` half exposed** (section 5), **`grid_source` absent from the split stage's `params`**
    and **the `splits` CLI's `W8_H4` fallback** (section 3): the cycle-2 items 7.4 and 7.5, refused
    as more than one line, still open.

---

## 7. Proposed corrections (I do not execute them; the author approves and applies)

None is required for the five conditions to hold. Each closes a residual mark.

1. **`run_moves.py` 314-320 (section 1 (a)).** If the author wants the trigger to be the file that
   is read, let `_stale_reason` accept a `csv-drop:gates/preconditions.csv:C7` input form that hashes
   the CSV with the named column blanked, and declare it in place of `ADMISSIBILITY`. The JSON proxy
   is otherwise sound; this is an option, not a requirement.
2. **`gates_temporal.py` 571 and 609 (section 1 (b)).** Either remove the `gf_part1` column from
   `GRID_COLUMNS` (62-64) and the two lines that fill it, so that G-F's verdict lives in `gf.csv`
   only, where the tables already read it; or add `"gates/gf.csv"` to the `select` step's declared
   inputs (`run_moves.py` 172) and re-run `select` for every rung after `gf all rungs at the selected
   points` at move 12, so that the copy is taken after the re-run. The first form is one line fewer
   and removes an undeclared channel; the second keeps a convenience column at the cost of one
   dependency.
3. **`run_moves.py` 181, 183, 187, 190, 193, 195 (section 1 (c); CHECK_3 M2).** Declare
   `json:gates/selection.json:<rung>` for the rung's own `splits` and `gx` steps and
   `csv:gates/g3_flags.csv:rung=<rung>` for the `alias` steps, and make `_stale_reason` hash the named
   entry or the matching rows; keep the whole file for `gl`, `tables table6` and the move-12
   consumers. A test: after a full run, `--dry-run` marks zero steps stale.
4. **`gates_temporal.py` 597-600 and 626.** When `n_applicable` for G1 is zero, either drop G1 from
   `applicable` as G2 is dropped or write `refusal: not run: no applicable kernel` into the selection
   entry instead of `best-feasible`; when any of the thirteen CSVs is missing, write `refusal: not
   run: grid incomplete (<n> points missing)` into the selection entry.
5. **`models.py` 680 and 415-430.** Use `S.selected_grid_id(out, args.rung, None)` in the CLI and
   refuse with `not run: no selection for <rung>` when there is none; record `grid_source` in
   `run_split_stage` `params` as `run_clustering` does (594).
6. **`gates_temporal.py` 290-291, 304.** Give `gate_grid` a `wapf_norm` parameter, pass it to
   `build_all_grid`, expose it on the `grid` CLI, and record it in `temporal.params.json`; or remove
   the alternative from `series features` so that one place owns the choice.
7. **The template CLIs.** Refuse to overwrite an existing `inputs/*` file unless `--force` is given.

## 8. Server

Nothing in this certification needs to run on the server, and I ran nothing there. When the author
runs the toolkit on the real corpus under the present code, the resume discipline is: a re-run of the
preconditions by hand with another threshold makes every later move stale on its own (probe D); a
change to `inputs/pass_table.csv` re-runs the grid, G-ORD and the selection on its own (probe C);
a resume of moves 7 to 13 with nothing changed re-runs the four split stages for no change (section 1
(c)), so the author should either apply section 7 item 3 first or expect that cost; and gate result
files under `gates/` are not to be edited by hand (probe F).

Scratch artifacts of this cycle are under
`/private/tmp/claude-501/-Users-jeries-Desktop-projects-thesis-memorySignal-mem-sig/14810c8c-2535-466d-a296-d2aae9739c16/scratchpad/cert3/`
(`out/` my run, `run_b.log`, `probes.sh`, `probe_c_real.log`, and the copies taken before each
probe). This file:
`/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/CERTIFY_al_farabi.md`.
