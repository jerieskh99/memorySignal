# plan11_encoding_ladder, build epoch 2: the detection layer for the detection paper (RAID)

Written 2026-09-17. Two builders implement this in parallel without talking to each other:
builder A (data, splits, metrics, levels, gates, the synthetic two-class corpus) and builder B
(tables, figures, driver, runbook, LNCS skeleton, bibliography). Every interface below is fixed:
the interfaces between the two builders are files on disk with the schemas of this document and
the verdict strings of section 3.0. Builder B reads builder A's files by their documented column
names and never recomputes a verdict; builder A never writes under `report/`. Where a definition
in the sources leaves a choice, the choice is a parameter with the default stated here and listed
in section 7; a builder never picks a different default and never adds an undeclared threshold.

The epoch-1 toolkit (`SPEC.md`, 158 tests passing, al-Farabi certified) is reused as the method
and is not changed in behaviour: every existing module keeps its behaviour for the encoding paper
and the existing test suite (`tests/test_*.py` as it stands) must still pass unchanged. New modules
are added; existing modules are extended only additively (new optional arguments or new constants
whose defaults preserve the current behaviour); the exact list of additive extensions is in
section 1.3 and nothing outside it is touched.

Sources of truth, in this order (a builder reads them before coding; docstrings cite them with
the shorthand in brackets):

1. `apf_paper/P3_RAID_STRUCTURE.md`, sections 0a, 1, 3, 9, 10 [`P3 0a`, `P3 D5`, `P3 Sec. 3`].
2. `apf_paper/raid_council/08_RAID_COUNCIL_REPORT.md`, section 2 items 1 to 31 [`CR3 2.N`],
   sections 1.5 and 1.6 [`CR3 1.5`, `CR3 1.6`].
3. `apf_paper/raid_council/02_ml_engineer_protocol.md` in full [`ML x.y`].
4. `apf_paper/raid_council/06_al_kindi_revised.md`, sections 2.2, 2.3 and 5 [`K3 2.2`, `K3 F<n>`,
   `K3 move N`].
5. `apf_paper/raid_council/07_al_nadim_final.md`, sections 1 and 2 [`N3 Sec. 1`, `N3 Sec. 2`].
6. `plan11_encoding_ladder/SPEC.md` [`SPEC x.y`] and `apf_paper/EPOCH1_BUILD_REPORT.md`
   [`E1 6.N`] for what exists and how it is shaped.

Binding rules for both builders (each restates the task's rules; none is negotiable):

- SERVER: FORBIDDEN. No ssh, scp, rsync; no path under the server mounts (the NFS mount, the project mount) in code, tests,
  fixtures or docs (the runbook says "the retention root you pass"); no remote command. Every
  test uses synthetic data generated on this machine by `synth.py` and `synth_detection.py`.
- SANDBOX FAMILY: INDEX AND LETTER ONLY. The family is "the sandbox family"; its members are
  members 1 to 8; its sub-families are A (4 members), B (1), C (3). No workload name of that
  family appears in any code, test, fixture, docstring, comment, report, table, figure, file name
  or directory name written by this toolkit. Class membership comes from the author's mapping
  file (section 2); the code reads the file and never guesses a class from a name. Synthetic
  fixtures use `sandbox_member_<m>` only. Neither builder opens `docs/test_families_spec.md`,
  any steps file, the workload source tree, any run record, or the public simulator repository
  of stage 3.
- NO PAPER PROSE. `p3_skeleton.tex` is headings, table environments, figure placeholders and
  comment blocks with substance bullets; not one sentence of body text.
- NO FABRICATION. Every gate function's docstring cites the definition it implements (`P3`,
  `CR3 2.N`, `ML x.y`, `K3`, `N3`). No bibliography entry is invented (section 1.2, builder B).
- PLAIN REGISTER in every report file and docstring: full sentences, no emojis, no em-dashes.
- No git commit. No file outside `plan11_encoding_ladder/`, `apf_paper/p3_skeleton.tex` and
  `apf_paper/p3.bib` is written.
- Python 3.10+; `numpy`, `scikit-learn`, `matplotlib` as `requirements.txt` already requires;
  no new hard dependency (`scipy` stays optional; `math.comb` and `itertools.combinations` are
  the stdlib tools for the null's support). `.csv.zst` is read only through `extract.open_text`.
  Streaming: no stage reads a trajectory except `extract.py`; every later stage reads the
  per-cell extract (about 1,000 rows), which may be loaded freely.
- Numbers never move: every threshold is a module-level constant or a CLI parameter with the
  default written in this file; the value used is written into every result file's `params`.
- Refusals are strings from section 3.0's vocabulary (this file's additions plus SPEC 3.0's),
  written to the artifact, never a number, never a blank. A table cell that cannot be filled
  prints the refusal string or `--` for an undefined number, never a blank.
- Every output file of this layer lives under `<out>/gates/detection/`, `<out>/detection/` or
  `<out>/report/detection/` (section 1.4), so the encoding paper's outputs under the same
  `<out>` are never overwritten. The one exception is `cells.csv`, which `classes apply`
  rewrites (section 2.3) after copying the original to `cells.pre_classes.csv`.

What this layer does NOT build, stated so that no builder fills the gap silently:

- Rung 2', the content channel (K3 move 15; CR3 2.28): the plan11 extract carries `hamming`,
  `l0`, `l1` only; the content family (`ent_q`, `distinct_bytes`, ...) would need an extract
  extension. Every table row for rung 2' prints
  `not run: content family columns not in the plan11 extract (rung 2' needs an extract extension)`.
- The state-change yield, F11 (CR3 2.27): no `/proc/vmstat` record exists in stage 1; the row
  prints `not run: no vmstat record (stage 1)`.
- The comparator row, RQ5 (ML 3.9): decided in another session; Table 7 carries the row
  `comparator` with `not run: comparator row from another session (RQ5)`; section 3.5.13 says
  where a comparator's scores would enter if that session writes them in this layer's schema.
- The harness rhythm (cepstrum against the logged iteration period, K3 move 7) and the harness
  comparison (K3 move 8) need `[SUSTAIN]` markers and the two stage-2 controls; without them
  every row reads `not run: stage 2 absent` (section 3.5.11); the cepstrum itself is not built
  in this epoch (the author decides after stage 2, section 7).
- The mixture arm (K3 move 22): not built.
- The cross-campaign row (K3 move 21): in stage 1 the 01c kernels ARE the benign class
  (`P3 0a`), so the row prints `not applicable: stage 1 (the 01c cells are the benign class)`.

---

## 1. Module layout

### 1.1 Builder A (data, splits, metrics, levels, gates, synthetic corpus)

| File | Owns |
|---|---|
| `classes.py` | The class mapping file (`inputs/classes.csv`), its validator, `apply` (rewrites `cells.csv`), the join table `gates/detection/cell_classes.csv`, the class-only letter sequence, `inherit-selection` (section 2). |
| `detection_splits.py` | The detection label dict and the folds: LOWO over both classes, LOCO, LOFO, the one-class folds, the level-2 and level-3 folds, the order-test folds; every fold list passed through `splits._assert_grouped` (section 3.2). |
| `detection_metrics.py` | The two-class forest run with in-fold thresholds: per-cell scores, the operating point at the declared FPR with the realized out-of-fold FPR, the one-percent reading, ROC and its area, per-member recall in eighths, the workload-level label-shuffle null with `null_not_estimable`, the majority baseline and the random scorer, G-OP's support counts, G-CAL's two readings, B1-G3 restated for two classes, the one-class run, the time-to-detect ladder (section 3.3). |
| `detection_levels.py` | Level 2 (sub-family confusion with one member held out) and level 3 (exact member under leave-one-rep-out, the signature ceiling) (section 3.4). |
| `gates_detection.py` | Detection admissibility, G-K0 for every cell (three quantities, three verdicts), G-N two-class, G-L (i) two-class, G-OP, G-LM, G-ANCHOR (kernels; idle sets; early-against-late idle), the order test with the drift regression, G-SIG, G-FP, G-1C, the harness clause, G-CAL, G-M two-class, G-DIM, the alias falsifier, G-V two-class, the miss table (section 3.5). |
| `synth_detection.py` | The synthetic two-class corpus with known answers, built on `synth.py` (section 4). |
| `tests/test_classes.py`, `tests/test_detection_splits.py`, `tests/test_detection_metrics.py`, `tests/test_detection_levels.py`, `tests/test_gates_detection.py`, `tests/test_synth_detection.py`, `tests/_det_common.py` | Builder A's tests: for every gate one case that must pass and one that must refuse, built by `synth_detection.py` (section 4.5), plus the function-level tests named there. |

### 1.2 Builder B (report, driver, runbook, skeleton, bibliography)

| File | Owns |
|---|---|
| `tables_detection.py` | Tables 4 (tiers with counts), 5 (the gates in one row each), 6 (validity), 7, 8, 9, 10, 11, the level-2 and level-3 tables, the ladder table, the per-cell appendix table, the manifest; each as CSV, Markdown and LaTeX through `_report_common.write_table` (section 5). |
| `figures_detection.py` | Figures 2, 4, 5, 6, the level map, APF(t) per tier, the level-2 confusion (section 5.3). |
| `latex_skeleton_p3.py` | Writes `report/detection/paper3_skeleton.tex` and, with `--standalone PATH`, the same file to `apf_paper/p3_skeleton.tex` (section 5.4). |
| `run_detection.py` | The driver for the detection moves D0 to D15 in al-Kindi's order, reusing `run_moves.py`'s ledger, skip and staleness machinery through the additive extensions of section 1.3 (section 6). |
| `RUNBOOK_DETECTION.md` | The command-line runbook for 96 + 8 + 64 cells (section 6). |
| `apf_paper/p3_skeleton.tex` | The standalone copy written once by `latex_skeleton_p3.py --standalone`. |
| `apf_paper/p3.bib` | Entries copied verbatim from `apf_paper/p2.bib` for the keys listed in section 5.4 and nothing else; every needed entry that is not in `p2.bib` is a `% NEEDED (Hunayn):` comment line. No entry is written from memory. |
| `tests/test_report_detection.py`, `tests/test_run_detection.py`, `tests/detection_fixtures.py` | Builder B's tests. `detection_fixtures.py` writes result files under `gates/detection/` in the schemas of section 3 by hand (as `tests/report_fixtures.py` did in epoch 1), so builder B's tests do not depend on builder A's code. |

Builder B may also import `_report_common.py` (readers, writers, `fmt_num`, `is_refusal`,
`write_table`) and `run_moves.py` (through the additive extensions only). Builder B reads verdict
strings from `verdicts.py` when the attribute exists and otherwise uses the literal of section 3.0
(the `_vattr` pattern of `_report_common.py`), so the two builds do not block each other.

### 1.3 Additive extensions to existing modules (the complete list; nothing else is edited)

Every extension keeps the current default so that the encoding-paper test suite passes unchanged.

| Module | Extension | Default (current behaviour) |
|---|---|---|
| `verdicts.py` | The constants of section 3.0 appended; the new named refusals added to `NAMED_REFUSALS`. | Existing strings unchanged; `is_refusal` unchanged on every existing string. |
| `models.py` | `make_forest(seed, n_jobs, n_estimators, *, oob_score: bool = False, min_samples_leaf: int = 1, class_weight=None)`; a public alias `reduce_fit = _reduce_fit`. | `oob_score=False`, `min_samples_leaf=1`, `class_weight=None` reproduce the present forest exactly. |
| `synth.py` | `SynthSpec` gains `family: str = "kernel"` and `test_label_fmt: str = "kernel_{name}_v2"`; `test_label` returns `test_label_fmt.format(name=self.name)`; `cell_dir` uses `family`. `write_cell` unchanged otherwise. | The defaults reproduce every present path and name. |
| `run_moves.py` | `parse_moves(spec, max_move: int = 13)`; `load_ledger(out, name: str = LEDGER)`; `save_ledger(out, ledger, name: str = LEDGER)`; `run_plan(o, plan, *, max_move: int = 13, ledger_name: str = LEDGER, internal_steps: dict | None = None)` where `internal_steps` maps an internal `sub` name to a callable `(out: Path, args: list[str]) -> tuple[str, dict]` consulted before the built-in `grid-check` / `gf-check`. `print_status(out, name: str = LEDGER)`. | With the defaults every existing call is unchanged. |

No other file of the epoch-1 package is edited. `schema.py`, `series.py`, `splits.py`, `nulls.py`,
`extract.py`, the five `gates_*.py`, `variance.py`, `tables.py`, `figures.py`,
`latex_skeleton.py`, `_report_common.py` are imported, never changed.

What is reused, by function (the map both builders code against):

| Reused | Used for |
|---|---|
| `extract.py index / all`, `schema.parse_cell_path`, `schema.cell_id_of`, `schema.campaign_of` | the cell index and the per-cell extract of every cell of every class (the extractor is class-blind) |
| `series.load_cells`, `load_extract_cached`, `load_sidecar`, `k_median_cell`, `head_drop_for`, `load_head_drop`, `rung_series`, `window_features`, `feature_names`, `features_path`, `load_features`, `build_features`, `selected_grid_id`, `resolve_wh`, `n_windows`, `write_csv`, `write_json`, `read_csv`, `read_json`, `inputs_sha256`, `sha256_file`, `fmt_num`, `RUNGS`, `AXIS_OF_RUNG`, `PAIR_RUNGS` | features per rung at the inherited grid point, level normalization (identical in every fold because it is a per-cell function computed once, SPEC 3.1.1 and al-Farabi's certification condition 5), I/O |
| `splits._groups`, `splits._assert_grouped`, `splits.fold_loro`, `splits.fold_loro_rep_index` | fold construction and the grouping assertion |
| `nulls.null_summary`, `nulls.SEED_*` | the null summary (p95, rank, strict exceedance), the fixed seeds |
| `models.make_forest`, `models.make_l1`, `models.aggregate_units`, `models.reduce_fit` | the forest with hyperparameters declared before the data, the one-feature threshold model, majority votes for the multiclass levels, the dimension-matched reduction |
| `gates_precondition.py preconditions / gk0 / gf / gk0-template / idle-admissibility-template`, `gates_calibration.py gc / gp / pass-table`, `gates_readings.py gj`, `gates_comparison.py gx` | the carried-over gates (C1 to C8, the `failed/` count, G-K0 on the kernels, G-F, G-C, G-P, G-J, G-X as G-ANCHOR part (i)); run unchanged inside `<out>` |
| `_report_common.write_table`, `read_csv`, `read_json`, `fmt_num`, `is_refusal`, `result_json`, `inputs_sha256`, `now_iso` | every table and the manifest |
| `run_moves.build_plan`'s `_cmd`, `run_plan`, `load_ledger`, `save_ledger`, `_output_exists`, `_stale_reason`, `print_status`, `print_plan` | the detection driver |

### 1.4 Output layout under `<out>`

```
<out>/
  cells.csv                                  plan11's index, rewritten by `classes apply` (2.3)
  cells.pre_classes.csv                      the index before the rewrite (written once, never overwritten)
  inputs/classes.csv                         AUTHOR: the class mapping file (2.1)
  inputs/gk0_source_sandbox.csv              AUTHOR: one line per member, numbered, never named (3.5.2); template by builder A
  inputs/iteration_boundaries.csv            AUTHOR, optional (stage 2): cell_id, boundary_seqs (3.3.9)
  inputs/iteration_counts.csv                AUTHOR, optional (stage 2): cell_id, iteration_count (3.5.14)
  gates/selection.json                       inherited by `classes inherit-selection` (2.7); the plan11 CLIs read it
  gates/preconditions.*  gates/gk0.csv  gates/gf.csv  gates/gc.csv  gates/gp.csv  gates/gj.*  gates/gx.csv
                                             plan11's gates, run unchanged inside <out>
  features/<rung>/<grid_id>_{raw,norm}.npz   plan11's feature files at the inherited grid point, every ok cell of every class
  gates/detection/
    classes_validation.json                  (2.2)
    cell_classes.csv  cell_classes.json      the join (2.4)
    letter_sequence.txt  letter_sequence.csv (2.5)
    admissibility.csv  admissibility.json    (3.5.1)
    gk0_cells.csv  gk0_members.csv  gk0.json (3.5.2)
    gn.csv                                   (3.5.3)
    gl.csv                                   (3.5.4)
    splits/<rung>/<grid_id>/<split>__<variant>/          (3.3.7) split in {lowo, loco, lofo, one_class[__<model>],
                                                          lowo_matched, lowo_seed<i>, order_<class>, anchor_idle,
                                                          idle_early_late, lowo__without_<family>, glm_<member>};
                                                          variant in {raw, norm}
        predictions.csv  scores.json  null.json  roc.csv  folds.json  l1_quarantine.json
        predictions_with_quarantine.csv      only when a feature was quarantined
    splits/<rung>/<grid_id>/level2/          (3.4)  confusion.csv  members.csv  scores.json  null.json  predictions.csv
    splits/<rung>/<grid_id>/level3/          (3.4)  confusion.csv  members.csv  scores.json  null.json  predictions.csv
    ladder/<rung>/<grid_id>/<reading>/prefix<T>s/predictions.csv, scores.json   (3.3.9)
    ladder.csv  ladder.json
    gop.csv  glm.csv  ganchor.csv  order.csv  drift.csv  gsig.csv  gfp.csv  g1c.csv  harness.csv
    gcal.csv  gm.csv  gdim.csv  alias.csv  gv_two_class.csv  gv_two_class_summary.csv
    miss_table.csv  fp_table.csv
    tripwire_check.json                      the driver's move D15
  detection/features/<rung>/<grid_id>_{raw,norm}_prefix<T>s_<reading>.npz    the ladder's prefix features (3.3.9)
  report/detection/tables/<name>.csv|.md|.tex
  report/detection/figures/<name>.png|.pdf   or figures/SKIPPED.txt when matplotlib is absent
  report/detection/paper3_skeleton.tex
  report/detection/manifest.json
  driver_detection_state.json                the detection driver's ledger (its own file)
```

Every JSON result file has the three top-level keys `schema` (`"plan11.detection.<name>.v1"`),
`params` (every parameter value used, by name, plus `inputs_sha256` of every input file read and
`grid_source`), and `citation` (the definition string of the function's docstring), then its
payload. Every CSV has a header row; integers as integers, floats through `series.fmt_num`,
refusals as strings, undefined numbers as the empty string in CSV and `--` in the tables.

`grid_source` is one of `"inherited: <path>"`, `"default: W8_H4 (no inherited selection)"` or
`"selection.json"` and is written into every `params` block and printed in every table's note
line, because the (W, H) per rung is inherited from the encoding paper's Table 5, fixed per
encoding and never per class (ML 1.1; K3 2.2 "Hyperparameters and (W, H)").

---

## 2. The class file and the letter sequence (`classes.py`, builder A)

### 2.1 `inputs/classes.csv`, the author's mapping file

CSV with a header row and exactly these columns, in any order; any other column is refused
(2.2 item 1), so no free-text column can carry a name:

| Column | Type | Required when | Meaning |
|---|---|---|---|
| `path_prefix` | text | always | A cell directory, or a prefix of one, as `cells.csv` lists `path`, or its last four path components (`<family>/<test_label>/<param-sig>/rep<NNN>__<label>`), or any leading part of those four components cut at a `/`. POSIX separators; a trailing `/` is ignored; `..` is refused. |
| `class` | one of `CLASSES` | always | `benign_kernel`, `benign_relaunched`, `benign_breadth`, `idle`, `harness_idle`, `sandbox`, `external`. |
| `member_index` | integer >= 1 | `sandbox`, `external` | The member index (deposit index i equals paper member i, CR3 1.8). |
| `subfamily_letter` | one capital letter A to Z | `sandbox` (optional for `external`, written `-` when blank) | The sub-family letter (`P3 0a`). |
| `rep` | integer 0 to 7 | never | Overrides plan11's rep index for the matched cells (SPEC 2.6). Blank: plan11's rule stands. |
| `order_index` | integer >= 1, unique across the file | never | The cell's position in the realized capture order (`P3 D4`, ML 3.2). Needed by the order test, the drift regression, early-against-late idle and the letter sequence; blank makes those read `not run: order_index missing`. |
| `family` | text | `benign_breadth` | The benign family name for LOFO and G-FP (the corpus family the author chose the workload from, ML 1.2). For other classes it is derived (2.4) and a filled value overrides the derivation. |
| `workload_key` | text | `benign_relaunched` when `relaunched_grouping = "parent"` (2.4) | The workload key the cell shares for LOWO and the null. For a re-launched control this is its parent kernel's name (CR3 2.31). |

Matching: a cell matches a row when `cell.path == prefix`, or `cell.path` starts with
`prefix + "/"`, or `tail4 == prefix`, or `tail4` starts with `prefix + "/"`, where `tail4` is
the cell's last four path components joined by `/`. When several rows match one cell, the row
with the longest `path_prefix` wins; equal lengths are a duplicate (refused). A row that
matches no cell is counted in `classes_validation.json` `unmatched_rows` (a warning, not a
refusal, because the author may write the file before every cell is captured). A cell that
matches no row is `class = "unassigned"` in the join (2.4) and enters no detection stage.

The `rep` column and the `order_index` column are per cell, so rows that set them must match
one cell each (a workload-level prefix with `rep` or `order_index` filled is refused, 2.2 item
12). A workload-level row and per-cell rows for the same workload may coexist: the per-cell rows
win by length and inherit nothing, so every per-cell row states its own `class`, `member_index`
and `subfamily_letter`; the validator checks that they agree with the workload-level row when
both match a cell (2.2 item 8).

Stage 1, as the author would write it (an illustration of the shape; the family and test-label
components are the author's and never appear in this repository):

```
path_prefix,class,member_index,subfamily_letter,rep,order_index,family,workload_key
kernel,benign_kernel,,,,,,
<idle family>/<idle test label>,idle,,,,,,
<family>/<member 1 test label>,sandbox,1,A,,,,
<family>/<member 2 test label>,sandbox,2,A,,,,
...
<family>/<member 8 test label>,sandbox,8,C,,,,
```

plus, when the author has the realized order, one row per cell with `order_index` (168 rows).
For stage 1 the sandbox order is by member (`P3 0a`: member 1's eight reps, then member 2's),
the kernels precede them (captured 2026-09-06 to 09-08 under three launch labels) and the idle
cells follow them (`P3 0a`, "Answered 2026-09-17", item 5).

### 2.2 The validator (`classes validate`), its refusals

`classes.validate_classes(path: Path, cells: list[dict], *, relaunched_grouping: str = "parent")
-> dict` returns `{"status": "ok" | "refused", "refusals": [str, ...], "unmatched_rows": int,
"unassigned_cells": int, "counts": {class: n_cells}, "members": {index: letter},
"subfamilies": {letter: [indices]}}` and the CLI writes it to
`gates/detection/classes_validation.json`. Each refusal is one of these strings, exactly:

1. `refused: unknown column <name>`
2. `refused: missing column <name>`
3. `refused: unknown class <value> in row <n>` (the allowed set is `CLASSES`)
4. `refused: sandbox row <n> without member_index` / `refused: sandbox row <n> without subfamily_letter`
5. `refused: member_index not a positive integer in row <n>`
6. `refused: subfamily_letter not one capital letter in row <n>`
7. `refused: member <m> mapped to two sub-families (<x>, <y>)`
8. `refused: rows <n> and <k> disagree on <column> for one cell`
9. `refused: duplicate path_prefix <prefix>`
10. `refused: path_prefix contains ".." in row <n>`
11. `refused: rep outside 0..7 in row <n>` / `refused: order_index not a positive integer in row <n>` / `refused: duplicate order_index <i>`
12. `refused: per-cell column <rep|order_index> on a row that matches <k> cells (row <n>)`
13. `refused: benign_breadth row <n> without family`
14. `refused: benign_relaunched row <n> without workload_key (parent kernel; CR3 2.31)` (only under `relaunched_grouping = "parent"`)
15. `refused: workload_key <x> of benign_relaunched row <n> is not a kernel name` (under `parent`; the kernel names are `schema.KERNEL_NAMES`)
16. `refused: benign_kernel row <n> matches a cell whose role is <role>` (a `benign_kernel` row must match cells with `role == "kernel"` and a kernel in `schema.KERNEL_NAMES`; an `idle` row must match cells with role `idle` or `unknown`)
17. `refused: two members of one workload path (<prefix>) in rows <n> and <k>` (a sandbox `path_prefix` at workload level may carry one member only)
18. `refused: member_index <m> used by both sandbox and external rows`

The file's absence is exit 2 (its path on stderr). A refusal string quotes only row numbers,
class values, column names and, in items 9 and 17, the author's own `path_prefix` text; the
validation file is an artifact under `gates/detection/` and builder B prints its `status` in
Table 5, never its `refusals` list.

Citation for the whole module: `P3 0a` (members by index, sub-families by letter, agents see
letters only); `P3 D4` and ML 3.2 (the realized order as a class-only letter sequence); CR3 1.8
(deposit index i equals paper member i); CR3 2.31 (the re-launched control's workload key).

### 2.3 `classes apply`: the rewrite of `cells.csv`

`classes.apply_classes(out: Path, classes_csv: Path, *, campaign_label: str | None = None,
relaunched_grouping: str = "parent") -> dict`. Refuses (writes the validation file and exits 0
with `cells.csv` untouched) unless validation is `ok`. Otherwise copies `cells.csv` to
`cells.pre_classes.csv` once (never overwritten if it exists), then rewrites every matched row:

| class | `role` | `kernel` | `archetype_predicted` | `cell_id` | `status` |
|---|---|---|---|---|---|
| `benign_kernel` | `kernel` (unchanged) | unchanged | unchanged | unchanged | unchanged |
| `idle` | `idle` | `idle` | `control` | `idle__rep<rr>__<campaign>` | `refused: unknown kernel` becomes `ok`; any other status unchanged |
| `sandbox` | `sandbox` | `sandbox_member_<m>` | `sandbox` | `sandbox_member_<m>__rep<rr>__<campaign>` | as idle |
| `benign_relaunched` | `relaunched` | `relaunched_<workload_key>` | `relaunched` | `relaunched_<workload_key>__rep<rr>__<campaign>` | as idle |
| `benign_breadth` | `breadth` | unchanged (the path-derived name of a non-sandbox family's workload is public) | `breadth` | unchanged | as idle |
| `harness_idle` | `harness_idle` | `harness_idle` | `control` | `harness_idle__rep<rr>__<campaign>` | as idle |
| `external` | `external` | `external_member_<m>` | `external` | `external_member_<m>__rep<rr>__<campaign>` | as idle |

`<rr>` is the row's `rep` when the class file sets it, else plan11's rep from `cells.csv`
(SPEC 2.6: seed 42 is rep 0, then ascending seed; for a workload with no seed-42 cell the reps
are 0 upward by ascending seed). `<campaign>` is the row's campaign from `cells.csv`
(`schema.campaign_of(label)`) unless `--campaign-label TEXT` is given, in which case every
rewritten `sandbox` and `external` row takes `TEXT` as its campaign (section 7: the launch label
of the sandbox capture is the author's and may not be safe to print). The columns `path`,
`label`, `test_label`, `param_sig`, `seed`, `rep_dir`, `family` of `cells.csv` are left as they
are: `cells.csv` is the author's index, not a report, and no table of this layer prints them.
The rewrite is idempotent (matching uses `path`, which is never changed). `cells.index.json`
gains the key `classes_applied` with the class file's sha256 and the counts per class. After
the rewrite `apply` calls `build_join` (2.4) and `letter_sequence` (2.5), so one command leaves
`cells.csv`, `cell_classes.*` and `letter_sequence.*` consistent.

After `apply`, every `cell_id` in `cells.csv` is a public identifier, so every later file that
is keyed by `cell_id` (the extracts, the feature files, every split's `predictions.csv`, every
table) carries no sandbox workload name. The extract directories `extract/<cell_id>/` are
therefore anonymous when `apply` runs before `extract all` (the runbook's order, D0 before D1);
if the author ran the extractor first, the runbook tells him to re-run D0 and D1 (`extract all`
skips nothing when the cell ids changed, since the sidecars are keyed by the new ids).

### 2.4 The join: `gates/detection/cell_classes.csv`

`classes.build_join(out: Path, *, kernel_family_rule: str = "tier", relaunched_grouping: str =
"parent") -> list[dict]` writes one row per `ok` cell of `cells.csv` (and one per `unassigned`
cell, with `class = "unassigned"` and every other derived column empty), columns:

`cell_id, class, y, workload_key, family, member_index, subfamily_letter, rep, campaign,
order_index, order_token, split_role`

- `y`: `"sandbox"` for `sandbox` and `external`; `"benign"` for the five benign classes
  (`ML 1.2`: two classes; the family and the workload stay in the metadata as group keys and
  reporting keys and are never features).
- `workload_key`: `benign_kernel` -> the kernel name; `benign_relaunched` -> the row's
  `workload_key` under `relaunched_grouping = "parent"` (CR3 2.31), its own `kernel` under
  `"own"`; `benign_breadth` -> the row's `workload_key` when filled, else its `kernel`;
  `idle` -> `idle`; `harness_idle` -> `harness_idle`; `sandbox` -> `sandbox_member_<m>`;
  `external` -> `external_member_<m>`.
- `family` (the benign family for LOFO, G-FP and the per-family recall, ML 1.6 item 4): the
  row's `family` when filled; else `benign_kernel` -> `kernels` under `kernel_family_rule =
  "tier"` (ML 1.6 item 4 names "kernels, idle, and each captured family"), or the kernel's
  predicted archetype from `schema.ARCHETYPE_OF` under `"archetype"`; `benign_relaunched` ->
  `relaunched`; `idle` -> `idle`; `harness_idle` -> `harness_idle`; `sandbox` -> `sandbox`;
  `external` -> `external`.
- `split_role`: `"train_test"` for every class except `external`, which is `"test_only"`
  (`P3 0a` stage 3: trained on the author's family, tested on theirs; an external cell is never
  in a training set, 3.2.2).
- `order_token`: 2.5.

`cell_classes.json` carries `params` (`classes_csv_sha256`, `kernel_family_rule`,
`relaunched_grouping`, `campaign_label`), the counts per class, per family, per member and per
sub-family, `S` (the number of sandbox members with at least one admissible cell, filled after
3.5.1), `B` (the number of benign workload keys), and `n_assignments = math.comb(S + B, S)`.

Constants in `classes.py`:

```python
CLASSES = ("benign_kernel", "benign_relaunched", "benign_breadth", "idle", "harness_idle", "sandbox", "external")
BENIGN_CLASSES = ("benign_kernel", "benign_relaunched", "benign_breadth", "idle", "harness_idle")
POSITIVE_CLASSES = ("sandbox", "external")
TEST_ONLY_CLASSES = ("external",)
CLASS_LETTER = {"sandbox": "S", "benign_kernel": "B", "benign_breadth": "B", "idle": "I",
                "harness_idle": "H", "benign_relaunched": "R", "external": "X"}
ROLE_OF_CLASS = {"benign_kernel": "kernel", "idle": "idle", "sandbox": "sandbox", "benign_relaunched": "relaunched",
                 "benign_breadth": "breadth", "harness_idle": "harness_idle", "external": "external"}
KERNEL_FAMILY_RULE = "tier"          # section 7 item 3
RELAUNCHED_GROUPING = "parent"       # section 7 item 4
```

### 2.5 The class-only letter sequence

`classes.letter_sequence(join_rows) -> list[str]`: when every row with a class has an
`order_index`, the tokens sorted by `order_index`, one per cell, grammar
`<L>[<m>]r<rep>` with `L = CLASS_LETTER[class]`, `<m>` the member index for `S` and `X` only,
`rep` the rep index: `S3r0`, `Br5`, `Ir2`, `Hr0`, `Rr7`, `X1r0`. The kernel's identity is
dropped on purpose: the sequence is class-only (`P3 D4`; ML 3.2). Written to
`gates/detection/letter_sequence.txt` (space-separated, one line) and
`letter_sequence.csv` (`order_index, token, class`). When any classed cell lacks
`order_index`, both files hold the single line `not run: order_index missing for <n> cells`.
This sequence is the ONLY representation of the realized order that any table, figure, log line
or docstring of this layer ever shows; `order_index` itself appears in `cell_classes.csv` and in
the per-cell appendix table (5.2.12) as an integer beside the public `cell_id`, which is
class-only by 2.3.

### 2.6 How the 96 + 8 + 64 cells map, and how stages 2 and 3 enter without a schema change

Stage 1 (`P3 0a`): 12 kernels x 8 reps = 96 cells of class `benign_kernel` (workload keys = the
twelve kernel names; family `kernels`); 8 idle cells of class `idle` (one workload key `idle`,
family `idle`); 8 members x 8 reps = 64 cells of class `sandbox` (workload keys
`sandbox_member_1` to `sandbox_member_8`; sub-families A = 4 members, B = 1, C = 3, as the
author's file says; the code never assumes which indices carry which letter). So S = 8, B = 13,
`n_assignments = C(21, 8) = 203,490`, and every rate is over 168 cells before admissibility.

Stage 2 (`P3 0a`, proposed): rows of class `benign_relaunched` (workload key = the parent
kernel, family `relaunched`), `benign_breadth` (family = the corpus family the author names,
public), `harness_idle` (family `harness_idle`); B grows, every gate that switches on the
presence of a class does so from `cell_classes.csv` counts (3.5.11 the harness clause, 3.5.2
the harness floor, 3.5.6 G-ANCHOR part (ii)); no column and no code path changes.

Stage 3 (`P3 0a`): rows of class `external` with member indices (the repository is never named;
its members are indices in the author's file); `split_role = test_only`; scored by the final
LOWO model (3.2.2); the row `external` appears in Tables 7, 8 and 11 as its own block.

### 2.7 `classes inherit-selection`

`classes.inherit_selection(out: Path, *, from_path: Path | None, default_grid: str | None) ->
Path` writes `<out>/gates/selection.json` either as a copy of the encoding run's
`gates/selection.json` (the payload verbatim; `params.inherited_from = <path>`,
`params.inherited_sha256`, `params.grid_source = "inherited: <path>"`) or, with
`--default W8_H4`, as `{rung: {grid_id: "W8_H4", W: 8, H: 4, passes_acceptance: false,
selected_by: "default: no inherited selection", gates_passed: [], refusal: ""}}` for the five
rungs with `params.grid_source = "default: W8_H4 (no inherited selection)"`. Exactly one of
`--from` and `--default` is given. If `<out>/gates/selection.json` already exists and was not
written by this command (no `params.grid_source`), the command refuses:
`refused: gates/selection.json exists and was written by gates_temporal select; pass --force to replace`.
Every later stage reads the grid id per rung through `series.selected_grid_id(out, rung, None)`
and records `grid_source` from this file's `params`. The runbook (section 6) says: the
encoding paper's selection is the (W, H) per rung; re-gridding on the detection data is
allowed only under the grid rule (SPEC 3.5.7) and is the author's decision (section 7 item 5).

### 2.8 The CLI of `classes.py`

```
classes.py validate          --out O --classes CSV [--relaunched-grouping parent|own]
classes.py apply             --out O [--classes CSV] [--campaign-label TEXT] [--kernel-family-rule tier|archetype]
                             [--relaunched-grouping parent|own]
classes.py join              --out O [--kernel-family-rule tier|archetype] [--relaunched-grouping parent|own]
classes.py letter-sequence   --out O
classes.py inherit-selection --out O (--from PATH | --default GRID_ID) [--force]
```

`apply` reads `<out>/inputs/classes.csv` unless `--classes` names another file, which it
copies to `<out>/inputs/classes.csv` first (refusing when a different file already exists
there, as 6.1 says). Exit codes as SPEC 7.1.

---

## 3. Every gate and every metric as a function signature

### 3.0 Verdict vocabulary additions (`verdicts.py`, appended; builder A writes them, builder B falls back to the literals)

```python
# the two-class null (CR3 2.1; ML 1.4)
NULL_NOT_ESTIMABLE = "null_not_estimable"
NULL_INSIDE = "inside the workload-level null"
# G-OP (CR3 2.13; ML 2.3, 4.3)
GOP_SUPPORTED = "operating point supported"
GOP_SET_BY_FEW = "set by one workload, family named"
# G-LM (CR3 2.17; ML 4.3; K3 F1)
GLM_SURVIVES = "detection survives level matching"
GLM_LEVEL_ONLY = "detected by level in this lead"
GLM_NOT_DETECTED = "not applicable: not detected at the operating point"
GLM_EMPTY_BAND = "not applicable: no benign workload within the level band"
GLM_LABEL_MEDIAN_K = "median K, stage 1"
# G-ANCHOR and the order test (CR3 2.14, 2.20; ML 3.1, 3.2; K3 F5; P3 0a)
GANCHOR_AUDIBLE = "campaign audible"
GANCHOR_NOT_AUDIBLE = "campaign not audible above the null"
ORDER_AUDIBLE = "position audible"
ORDER_NOT_AUDIBLE = "position not audible above the null"
ORDER_VOID = "void: position audible inside the interleaved campaign"
DRIFT_SLOPE = "drift: slope above the shuffle null"
DRIFT_NONE = "no drift above the shuffle null"
# G-SIG (CR3 2.16; K3 F2)
GSIG_REPORTED = "gap reported"
GSIG_IDENTITY = "recognizes workload identity, not family behaviour"
# G-FP (CR3 2.15; K3 F8)
GFP_ATTRIBUTED = "attributed"
GFP_INSEPARABLE = "inseparable from sandbox under this rung"
# G-1C (CR3 2.19; K3 section 5 point 1)
G1C_PRIMARY = "primary"
G1C_SECONDARY = "secondary, not citable"
G1C_SEARCH = "refused: a second one-class model without a declared primary reads as a search"
# the harness clause (CR3 2.12, 2.21; K3 F4)
HARNESS_STAGE2_ABSENT = "not run: stage 2 absent"
HARNESS_COMPARABLE = "harness's (comparable margins)"
HARNESS_CLASS_EXCEEDS = "class margin exceeds the harness margin"
HARNESS_RELAUNCH_NOT_CLASS = "re-launch, not class"
# G-CAL (CR3 2.18; ML 1.6 item 2)
GCAL_AGREE = "per-fold and pooled agree within the null spread"
GCAL_PERFOLD = "per-fold reported; pooled curve threshold set post hoc on the held-out scores"
POST_HOC_LABEL = "threshold set post hoc on the held-out scores"
# G-K0 two-class (CR3 2.8; K3 F3)
GK0_AT_FLOOR = "at floor"
GK0_AT_HARNESS_FLOOR = "at harness floor"
GK0_MIXED = "cells in more than one verdict"
# G-N two-class (CR3 2.5)
GN_ONE_TRAIN_WORKLOAD = "one training workload per fold"
GN_NO_SUPERVISED = "no supervised headline"
GN_SINGLE_WORKLOAD = "single workload"
# levels 2 and 3 (P3 0a; G-N)
L2_ONE_TRAIN = "one training member per fold"
SIGNATURE_CEILING = "signature ceiling"
def level2_no_heldout(n: int) -> str:
    return "one member, no held-out test" if n == 1 else f"{n} members, no held-out test"
# the time-to-detect ladder (K3 move 17; CR3 2.29)
LADDER_FROM_PAIR1_ONLY = "from pair 1 only"
# the miss table (K3 move 18)
AT_FLOOR_NOT_A_MISS = "at floor, not a miss"
# stage 1 placeholders (this file, preamble)
CROSS_CAMPAIGN_STAGE1 = "not applicable: stage 1 (the 01c cells are the benign class)"
RUNG2P_NOT_BUILT = "not run: content family columns not in the plan11 extract (rung 2' needs an extract extension)"
COMPARATOR_ELSEWHERE = "not run: comparator row from another session (RQ5)"
YIELD_NOT_RECORDED = "not run: no vmstat record (stage 1)"
```

Added to `NAMED_REFUSALS`: `NULL_NOT_ESTIMABLE, NULL_INSIDE, GOP_SET_BY_FEW, GLM_LEVEL_ONLY,
ORDER_VOID, GSIG_IDENTITY, GFP_INSEPARABLE, G1C_SEARCH, HARNESS_RELAUNCH_NOT_CLASS,
GK0_AT_FLOOR, GK0_AT_HARNESS_FLOOR, GN_NO_SUPERVISED`. (`GLM_NOT_DETECTED`, `GLM_EMPTY_BAND`
and the `not run:` strings are refusals through their prefix already.) `PASS` and `FAIL` are
reused. A verdict with a suffix in parentheses (`"pass (median K, stage 1)"`) is judged on its
head, as `is_refusal` already does.

### 3.1 Features, level normalization, admissibility

3.1.1 Features. For every rung at the inherited grid point, `series.build_features` (through
the CLI `series features --out O --rung R --grid-id ID --both`; `--norm` only for `combined`)
writes `features/<rung>/<grid_id>_{raw,norm}.npz` over every `ok` cell of `cells.csv`, of every
class, with the per-row `role`, `kernel`, `archetype`, `campaign`, `rep`, `cell_id`,
`win_start` as SPEC 3.1.5 defines them. Nothing in this layer computes a feature by another
rule; the feature names are `series.feature_names(rung, normalized)` and the feature matrix is
"the declared per-cell summaries of the artifact and nothing else" (K3 2.2): never a metadata
column, never the pair count or any cadence-derived count, never the realized interval or the
iteration count. The pre-declared exclusion list is recorded in every `params` block as
`excluded_by_declaration = ("n_pairs", "n_windows", "dt_est_s", "iteration_count",
"order_index", "campaign", "label", "path")` (ML 3.4; CR3 2.2).

3.1.2 Level normalization (G-L) is plan11's, per rung, unchanged: `apf` `K / K_median_cell`;
`wapf` by `K_median_cell * 32768` (`wapf_norm = "median_K"`); `persist` `J - J_null`; `content`
the fifteen level-free ratios; `combined` the concatenation (SPEC 3.1.1). It is a per-cell
function of the cell's own statistics, computed once at feature build time and identical in
every fold (al-Farabi's condition 5). The raw variant exists for `apf` (the level-inclusive
ceiling, `P3 D5`; CR3 2.4 "the raw row stays") and `wapf`, `persist`, `content`; Table 7 prints
`apf raw` and the normalized rows.

3.1.3 The ladder's prefixes (3.3.9) are the one place features are built by this layer:
`detection_metrics.build_prefix_features(out, rung, grid_id, normalized, n_rows_by_cell:
dict[str, tuple[int, int]], reading: str) -> Path` truncates each cell's extract to rows
`[start, start + n)` (`detection_metrics.slice_extract(ex, start, n) -> dict`, which slices every
array of the extract dict and sets `_n_rows = n`) and then calls `series.rung_series` and
`series.window_features` exactly as `build_features` does, so the normalization of a prefix
uses the prefix's own median K (`ladder_norm = "prefix"`, section 7 item 20; the alternative
`"whole_cell"` divides by the whole cell's median instead). The npz has the layout of SPEC 3.1.5
plus the scalars `prefix_s`, `reading`, `n_rows_by_cell_json`.

3.1.4 Detection admissibility (`gates_detection.admissibility`, 3.5.1) replaces
`series.admissible_cells` for this layer: `all_hard_pass` is read from
`gates/preconditions.csv`, but C1's `fail` on a cell whose class is not `benign_kernel` is
recorded and does not exclude the cell under `det_c1_rule = "report"` (the default; section 7
item 7), because a sandbox or idle-like cell that fails C1 is the at-floor case that G-K0
labels and that leaves the denominator with its label, never a refused observation (K3 F3;
CR3 2.8). Under `"exclude"` the plan11 rule applies to every class. The pair rungs' `failed/`
exclusion (SPEC 3.3.2; `series.PAIR_RUNGS`) applies unchanged.

### 3.2 `detection_splits.py`

3.2.1 The label dict. `make_detection_labels(feat: dict, join: list[dict], admissible: set[str],
*, mask_classes: tuple[str, ...] | None = None) -> dict` restricts the loaded feature file to
rows whose `cell_id` is admissible and classed (and, when `mask_classes` is given, in those
classes) and returns, per row (window): `n`, `_rows`, `cell_id`, `y` (`sandbox` | `benign`),
`cls` (the class), `workload_key`, `family`, `member_index` (int, 0 for non-members),
`subfamily_letter` (`-` for non-members), `rep`, `campaign`, `order_index` (int or -1),
`split_role`, `win_start`, `floor_verdict` (from `gk0_cells.csv` when it exists, else
`"pending: gk0 not run"`), plus the aliases `kernel` (= `workload_key`) and `archetype` (= `cls`)
so that `splits.fold_loro`, `splits.fold_loro_rep_index` and `splits._assert_grouped` run on
it unchanged. Every fold function takes this dict. Unit = cell (CR3 1.5; ML 1.1):
whole cells are held out; no cell's windows straddle train and test; `splits._assert_grouped`
runs on `cell_id` for every fold list and on `workload_key` for every fold list that groups by
workload.

3.2.2 The folds.

```python
def fold_lowo(lab: dict) -> list[dict]:
    """Leave-one-workload-out over all workloads of both classes (ML 1.3; CR3 1.5; K3 2.2;
    P3 D5). One fold per workload_key among rows with split_role == "train_test"; the fold
    holds out every cell of that workload (all reps) and trains on every other train_test
    cell of both classes. Sandbox folds give the true-positive rate, benign folds the
    false-positive rate. Fold dict: name "lowo/<workload_key>", held_out, y_held (sandbox |
    benign), family, member_index, train, test. If any test_only cells exist, one extra
    fold {name: "lowo/final", held_out: "external", train: every train_test row, test: every
    test_only row} is appended (P3 0a stage 3). Asserted grouped on workload_key and cell_id."""
def fold_loco(lab: dict, mode: str = "cell") -> list[dict]:
    """Leave-one-cell-out, the signature ceiling (ML 1.3; CR3 1.5; K3 F2): mode "cell" is
    splits.fold_loro on the detection labels (one fold per cell; the model trains on the
    sibling reps); mode "rep_index" is splits.fold_loro_rep_index (one fold per rep index,
    8 folds; section 7 item 11). test_only rows are never in train and are appended as the
    "loco/final" fold as in fold_lowo. Asserted grouped on cell_id."""
def fold_lofo(lab: dict) -> list[dict]:
    """Leave-one-benign-family-out (K3 2.2, LOFO; CR3 1.5; K3 F8): one fold per benign
    family (the family key of the benign rows); the fold holds out every cell of that family
    and trains on every other train_test cell of both classes; sandbox cells are in every
    training set, so LOFO yields a false-positive rate per unseen family and no true-positive
    rate (scores.json marks tpr fields not_applicable("sandbox never held out under LOFO")).
    Asserted grouped on family and cell_id."""
def fold_one_class(lab: dict) -> list[dict]:
    """The one-class reading (ML 1.3; K3 2.2; CR3 2.19): benign-only training. One fold per
    benign workload_key (train = every other benign train_test cell; test = that workload's
    cells) for the false-positive rate, then the fold "one_class/final" (train = every benign
    train_test cell; test = every sandbox cell and every test_only cell), so every sandbox
    cell is scored by a model that never saw a sandbox cell. Asserted grouped on workload_key."""
def fold_level2(lab: dict, *, min_members_test: int = 2) -> list[dict]:
    """Level 2 (P3 0a; G-N): rows restricted to class sandbox. One fold per member whose
    sub-family has at least min_members_test members (test = that member's cells; train =
    every other sandbox cell of every sub-family). A sub-family with fewer members has no
    fold and is reported by detection_levels as level2_no_heldout(n). Asserted grouped on
    workload_key."""
def fold_level3(lab: dict, mode: str = "rep_index") -> list[dict]:
    """Level 3 (P3 0a): rows restricted to class sandbox; mode "rep_index": one fold per rep
    index, holding out that rep of every member (the leave-one-rep-out reading); mode
    "cell": one fold per cell. Labelled SIGNATURE_CEILING by the caller."""
def fold_order(lab: dict, cls: str) -> list[dict]:
    """The order test (ML 3.2; CR3 2.20; P3 0a): rows restricted to class cls with order_index
    >= 1; LOWO folds within the class (one per workload_key), the label is the half (3.5.7).
    For a class with one workload (idle) the folds are leave-one-cell-out instead and the
    fold dict carries unit = "cell"."""
def fold_anchor_idle(lab: dict) -> list[dict]:
    """G-ANCHOR part (ii) (ML 3.1; CR3 2.14): rows of class idle (and harness_idle apart),
    leave-one-cell-out, label = campaign."""
```

| split | folds on stage 1 | held out | gives |
|---|---|---|---|
| `lowo` | 21 (12 kernels, idle, 8 members) | all cells of one workload | TPR (sandbox folds) and FPR (benign folds); the headline |
| `loco` | 168 (`cell`) or 8 (`rep_index`) | one cell, or one rep index of every workload | the signature ceiling; G-SIG's minuend |
| `lofo` | 2 (kernels, idle) | all cells of one benign family | FPR of an unseen family; F8 |
| `one_class` | 13 + 1 | one benign workload; then every sandbox cell | the one-class reading; G-1C; F9 |
| `level2` | 7 (4 of A, 3 of C; B has no fold) | one member | the sub-family confusion |
| `level3` | 8 | one rep of every member | the exact-member ceiling |
| `order_<cls>` | per class: 8 (sandbox), 12 (kernels), 8 cells (idle) | one workload (or cell) | position predictability |
| `anchor_idle`, `idle_early_late` | 8 cells | one idle cell | campaign or half predictability of the floor |

### 3.3 `detection_metrics.py`

3.3.1 Constants.

```python
FPR_DECLARED = 0.05                 # P3 D5; CR3 2.13; ML 2.3 (declared before the data)
FPR_RESOLUTION_LIMIT = 0.01         # labelled "resolution limit" (ML 1.6 item 3)
THRESHOLD_SOURCE = "oob"            # ML 1.6 item 2: the 95th percentile of the benign training cells' out-of-bag scores
THRESHOLD_QUANTILE_METHOD = "linear"   # numpy.quantile's default; section 7 item 9
SCORE_AGGREGATION = "window_mean_proba"   # the cell's score = mean over its windows of the forest's P(sandbox); section 7 item 8
N_PERM = 500                        # CR3 2.1
NULL_EXHAUSTIVE_BELOW = 500         # exhaustive enumeration when comb(S + B, S) < 500
NULL_MIN_ASSIGNMENTS = 20           # below it: NULL_NOT_ESTIMABLE (ML 1.4, 2.2)
NULL_STATISTICS = ("tpr05", "auc")  # both recorded per permutation
NULL_VERDICT_STATISTIC = "tpr05"    # the headline's verdict is on TPR at the declared FPR (P3 D5); AUC's verdict beside it (ML 1.6 item 1); section 7 item 10
N_ESTIMATORS = 300; MIN_SAMPLES_LEAF = 1; CLASS_WEIGHT = None   # ML 3.7, question 12; SPEC 4.2; section 7 item 6
LADDER_PREFIXES_S = (30, 60, 120, 300, 600)   # K3 move 17
LADDER_DT_S = 0.644                 # pair units from the derived spacing; 0.500 the configured interval (schema.DT_BRACKET_S); section 7 item 20
LADDER_READINGS = ("from_pair1", "from_boundary")
LADDER_NORM = "prefix"
ONE_CLASS_MODEL = "isolation_forest"   # G-1C's primary, declared here before the data; section 7 item 12
ONE_CLASS_THRESHOLD_SOURCE = "inner_lowo"   # section 7 item 13
B1G3_MAX_DISAGREE_WORKLOADS = 1     # CR3 2.2: all but at most one held-out workload
CITATION_OP = "P3 D5; CR3 1.5; ML 1.6 (metrics), 2.3 (resolution), 2.4 (per-workload recall), 3.7 (in-fold thresholds)"
```

3.3.2 The forest and the cell score.

```python
def make_detection_forest(seed: int, n_jobs: int = 1, *, n_estimators: int = N_ESTIMATORS,
                          min_samples_leaf: int = MIN_SAMPLES_LEAF, class_weight=CLASS_WEIGHT):
    """models.make_forest(seed, n_jobs, n_estimators, oob_score=True, min_samples_leaf=...,
    class_weight=...): the same imputer, scaler and RandomForestClassifier as the encoding
    paper (SPEC 4.2), with out-of-bag scoring on so the in-fold threshold can be set on the
    benign training cells' out-of-bag scores (ML 1.6 item 2). Citation: P3 D5 'a random forest
    with hyperparameters declared before the data'; ML 3.7."""
def cell_scores(cell_ids: np.ndarray, proba_pos: np.ndarray, rule: str = SCORE_AGGREGATION) -> dict[str, float]:
    """The cell's score: 'window_mean_proba' the mean over its windows of P(sandbox);
    'window_median_proba' the median; 'vote_fraction' the fraction of windows predicted
    sandbox. NaN windows are skipped; a cell with no finite window is NaN and counted."""
def oob_cell_scores(pipeline, X_train, y_train, cell_ids_train) -> dict[str, float]:
    """The benign training cells' out-of-bag scores (ML 1.6 item 2): the fitted forest's
    oob_decision_function_ column for the class 'sandbox', aggregated per cell by
    cell_scores; a window never out of bag (NaN) is skipped. The imputer and scaler of the
    pipeline are fitted on the training fold, so the out-of-bag rows were transformed by
    them, which is recorded as params.oob_note."""
def in_fold_threshold(oob_benign: dict[str, float], fpr: float, method: str = THRESHOLD_QUANTILE_METHOD) -> float:
    """numpy.quantile(values, 1 - fpr, method=method) over the benign training cells' scores
    (the 95th percentile at fpr = 0.05; the 99th at 0.01). A cell is flagged when its score is
    strictly greater than the threshold (strict, as every exceedance in this toolkit)."""
```

3.3.3 The operating point, per fold, then pooled.

```python
def run_operating_point(X, lab, folds, *, seed, n_jobs, n_estimators, min_samples_leaf, class_weight,
                        fpr_declared=FPR_DECLARED, fpr_limit=FPR_RESOLUTION_LIMIT,
                        threshold_source=THRESHOLD_SOURCE, method=THRESHOLD_QUANTILE_METHOD,
                        score_rule=SCORE_AGGREGATION, positive="sandbox", label_key="y",
                        reduce_to=None, reduce_method="train_importance") -> dict:
    """Per fold: fit the forest on the training rows; threshold_05 and threshold_01 from the
    benign training cells' out-of-bag scores (threshold_source 'oob'; 'inner_lowo' instead
    fits, inside the training fold, one forest per benign training workload with that
    workload held out and uses those out-of-fold scores, section 7 item 9); score every test
    cell (cell_scores over its windows); flag at both thresholds. Returns {'records': [per
    test cell: cell_id, fold, y, cls, workload_key, family, member_index, subfamily_letter,
    rep, campaign, score, threshold_05, threshold_01, flag_05, flag_01, n_windows],
    'folds': [per fold: name, held_out, n_train_cells, n_benign_train_cells, threshold_05,
    threshold_01, setters_05 (the benign training cells whose out-of-bag score >= threshold_05),
    d_used, importance (mean impurity importance per feature)], 'importance_mean'}.
    `positive` and `label_key` let the same runner score the campaign, half and other binary
    labels (3.5.7): when label_key is not "y" no threshold is set (the threshold and flag
    fields are empty) and the run's statistic is the AUC alone; with `reduce_to` the features
    are reduced per fold on the training fold only through models.reduce_fit (SPEC 3.7.7).
    Citation: CITATION_OP."""
def summarize_operating_point(records, folds, floor_by_cell: dict[str, str], *, fpr_declared, fpr_limit) -> dict:
    """The pooled reading (ML 1.6; CR3 1.5): the denominator of every true-positive rate is
    the set of positive cells whose floor verdict is 'above floor' (K3 F3: at-floor and
    at-harness-floor cells leave the denominator with their verdict, listed in
    'excluded_at_floor'); tpr_05 = flagged / denominator; fpr_05_realized = flagged benign /
    all benign scored (the realized out-of-fold rate, a measurement); tpr_01, fpr_01_realized
    likewise; auc = roc_auc_score over the pooled out-of-fold scores (positives in the
    denominator only); roc = roc_curve points; tpr_at_fpr05_pooled = the TPR read from the
    pooled ROC at FPR exactly 0.05 (interpolated), labelled POST_HOC_LABEL; per_member = {m:
    {hits, denominator, at_floor, eighths: 'k/n'}} (never a mean; ML 2.4); per_family_fpr =
    {family: {n, flagged, fpr}} (benign recall per family = 1 - fpr, ML 1.6 item 4);
    per_workload_outcome = {workload_key: recall over its cells (sandbox) or 1 - flagged
    fraction (benign)} (G-M's paired unit, ML 2.5); majority_accuracy = n_benign / (n_benign +
    n_positive_in_denominator) (B1-G6, CR3 2.3); random_scorer = {'auc': 0.5, 'tpr_equals_fpr': true};
    gop = {'setter_cells', 'n_setter_cells', 'n_setter_workloads', 'setter_families',
    'realized_fp_cells', 'n_realized_fp_workloads'} (3.5.5); gcal = {'per_fold_tpr05',
    'pooled_tpr_at_fpr05'} (3.5.12)."""
```

3.3.4 The workload-level label-shuffle null.

```python
def count_assignments(S: int, B: int) -> int:                # math.comb(S + B, S)
def workload_label_permutations(workload_keys: list[str], S: int, n_perm: int, rng) -> tuple[list[frozenset], bool]:
    """The S sandbox labels reassigned to a random S-subset of the S + B workloads, every
    cell inheriting its workload's label (ML 1.4; CR3 2.1; K3 2.2). Exhaustive
    (itertools.combinations) when comb(S + B, S) < NULL_EXHAUSTIVE_BELOW, else n_perm draws
    from rng.choice(S + B, S, replace=False) (duplicates allowed and counted in
    n_distinct_drawn). Returns (subsets, exhaustive). test_only workloads are never in the
    pool (they are never trained on)."""
def null_distribution(X, lab, folds_fn, subsets, *, statistics=NULL_STATISTICS, **op_kw) -> dict[str, np.ndarray]:
    """For every subset: relabel y at the workload, rebuild nothing (LOWO folds are
    label-free), re-run run_operating_point and summarize_operating_point, record tpr05 and
    auc. The same seed for every permutation; permutations are independent and are
    distributed over n_jobs processes."""
def null_verdict(observed: float, null_values: np.ndarray, n_assignments: int) -> tuple[str, dict]:
    """nulls.null_summary plus the verdict: NULL_NOT_ESTIMABLE when n_assignments <
    NULL_MIN_ASSIGNMENTS (the row carries no verb); else PASS when observed strictly exceeds
    p95 (ties fail), else NULL_INSIDE; the rank is reported as 'rank r of n'. A run with fewer
    than 500 permutations and a non-exhaustive pool writes not_run('N permutations < 500') as
    the verdict (a smoke run; the numbers stay). Citation: CR3 2.1; ML 1.4, 2.2."""
```

3.3.5 B1-G3 restated for two classes (CR3 2.2; ML 3.6). `quarantine_l1_two_class(X, names,
lab, folds, forest_records, *, max_disagree_workloads=B1G3_MAX_DISAGREE_WORKLOADS, seed)`: for
each feature, `models.make_l1(2)` is fitted per fold on that feature column alone over the
training windows and its per-cell decision is the majority vote of its window predictions
(`models.aggregate_units`, the plan11 rule; a tree has no out-of-bag score, so no threshold is
set for it); a feature whose per-cell decisions agree with the forest's `flag_05` decisions on
the cells of all but at most one held-out workload is quarantined;
`l1_quarantine.json` lists `{feature, n_disagree_workloads, disagreeing_workloads,
n_workloads}`; the split is re-run without the quarantined features and both readings are
kept (`scores.json["with_quarantine"]`, `predictions_with_quarantine.csv`); the tables print
the re-run (SPEC 3.7.2's rule, `params.score_source`). The exclusion by declaration (3.1.1)
is recorded before the test runs.

3.3.6 The one-class run.

```python
def make_one_class(model: str = ONE_CLASS_MODEL, seed: int = SEED_FOREST):
    """'isolation_forest': Pipeline(SimpleImputer(median), StandardScaler(),
    IsolationForest(n_estimators=300, contamination='auto', random_state=seed)); score =
    -score_samples (higher = more anomalous). 'gmm': GaussianMixture(n_components=4, n_init=5,
    random_state=seed) on the standardized rows, score = -score_samples (negative
    log-likelihood). 'ocsvm': OneClassSVM(kernel='rbf', gamma='scale', nu=0.05), score =
    -decision_function. One is primary (ONE_CLASS_MODEL); any other is run only with
    --secondary and labelled G1C_SECONDARY. Citation: CR3 2.19; ML 1.6 (a single density model
    declared as primary before the data); K3 section 5 point 1."""
def run_one_class(out, rung, grid_id, *, model=ONE_CLASS_MODEL, primary=True, normalized=True,
                  threshold_source=ONE_CLASS_THRESHOLD_SOURCE, fpr_declared=..., fpr_limit=..., seed=..., n_jobs=1) -> Path:
    """fold_one_class: per benign workload fold the model is fitted on the other benign cells
    and scores the held-out workload (its FPR); the final model is fitted on every benign cell
    and scores every sandbox and test_only cell. Threshold: 'inner_lowo' = the (1 - fpr)
    quantile of the benign cells' out-of-fold scores from the per-workload folds (the honest
    analogue of the forest's out-of-bag rule; the same threshold serves the final model);
    'train_in_sample' = the quantile of the training cells' in-sample scores. Writes the split
    directory one_class (primary) or one_class__<model> (secondary) with the files of 3.3.7;
    scores.json carries model, primary, g1c_label."""
```

3.3.7 `run_detection_split`, the split stage of this layer.

```python
def run_detection_split(out: Path, rung: str, grid_id: str, split: str, *, normalized: bool = True,
                        n_perm: int = N_PERM, run_null: bool = True, seed: int = SEED_FOREST, seed_offset: int = 0,
                        n_jobs: int = 1, n_estimators: int = N_ESTIMATORS, min_samples_leaf: int = MIN_SAMPLES_LEAF,
                        class_weight=CLASS_WEIGHT, fpr_declared: float = FPR_DECLARED, fpr_limit: float = FPR_RESOLUTION_LIMIT,
                        threshold_source: str = THRESHOLD_SOURCE, threshold_quantile_method: str = THRESHOLD_QUANTILE_METHOD,
                        score_aggregation: str = SCORE_AGGREGATION, loco_mode: str = "cell",
                        reduce_to: int | None = None, reduce_method: str = "train_importance",
                        quarantine: bool = True, feature_drop: tuple = (),
                        subset_workloads: tuple[str, ...] | None = None, exclude_families: tuple[str, ...] = (),
                        label_key: str = "y", positive: str = "sandbox", dir_name: str | None = None) -> Path:
```

Writes `gates/detection/splits/<rung>/<grid_id>/<dir_name or split>__<raw|norm>/`:

- `predictions.csv`: `cell_id, class, y, workload_key, family, member_index, subfamily_letter,
  rep, campaign, fold, score, threshold_05, threshold_01, flag_05, flag_01, n_windows,
  floor_verdict, in_denominator` (`in_denominator` true for positives above floor; benign
  cells `true`; at-floor positives `false`).
- `scores.json` payload: `status` (`ok` or a `not applicable:` string), `split`, `n_folds`,
  `tpr_05`, `fpr_05_realized`, `tpr_01`, `fpr_01_realized`, `auc`, `tpr_at_fpr05_pooled`,
  `pooled_label` (= `POST_HOC_LABEL`), `n_positive_scored`, `n_positive_in_denominator`,
  `n_positive_at_floor`, `excluded_at_floor` (cell ids), `n_benign_scored`, `per_member`,
  `per_family_fpr`, `per_workload_outcome`, `majority_accuracy`, `random_scorer`, `null`
  (`{statistic: {p95, p05, spread, rank, rank_text, n, exceeds, verdict}}`, `n_assignments`,
  `exhaustive`, `n_distinct_drawn`, `S`, `B`), `gop`, `gcal`, `feature_count`,
  `feature_count_used` (after reduction), `dim_status` (`GDIM_FULL` | `GDIM_REDUCED`),
  `importance_mean` (per feature name), `quarantine` (`{quarantined_features,
  n_disagree_by_feature}`), `with_quarantine` (the same payload keys recomputed without the
  quarantined features, or `null`), `score_source`, `grid_source`, `seed`, `n_perm`.
- `null.json`: the `null_summary` per statistic and the arrays of permuted `tpr05` and `auc`.
- `roc.csv`: `fpr, tpr, threshold` of the pooled ROC.
- `folds.json`: the fold records of 3.3.3 (thresholds and setters per fold).
- `l1_quarantine.json`; `predictions_with_quarantine.csv` when a feature was quarantined.

`subset_workloads` restricts the rows to those workload keys (G-LM's level band, 3.5.6);
`exclude_families` drops benign families (G-FP's recomputation, 3.5.9); `dir_name` names the
directory for those runs (`glm_<member>`, `lowo__without_<family>`). For `lofo` the `tpr_*`
keys hold `not_applicable("sandbox never held out under LOFO")` and `per_family_fpr` is filled
from the held-out folds. Every `params` block records the class counts, `S`, `B`,
`excluded_by_declaration`, `det_c1_rule`, `floor_source = gates/detection/gk0_cells.csv` and
`inputs_sha256` of `cells.csv`, the feature file, `gates/detection/cell_classes.csv`,
`gates/detection/admissibility.csv`, `gates/detection/gk0_cells.csv`, `gates/selection.json`.

CLI (the driver calls these names; 6.2 lists every flag):

```
detection_metrics.py splits   --out O --rung R [--grid-id ID] (--split lowo|loco|lofo | --all-splits)
                              (--raw | --norm | --raw-and-norm) [--null-perm 500] [--null-splits lowo]
                              [--n-jobs 1] [--n-estimators 300] [--min-samples-leaf 1] [--class-weight none|balanced]
                              [--threshold-source oob|inner_lowo] [--score-aggregation window_mean_proba|window_median_proba|vote_fraction]
                              [--loco-mode cell|rep_index] [--reduce-to-strongest] [--feature-drop F,...] [--seed-offset 0]
detection_metrics.py one-class --out O --rung R [--grid-id ID] [--model isolation_forest|gmm|ocsvm] [--secondary]
                              [--threshold-source inner_lowo|train_in_sample] [--seed-offset 0]
detection_metrics.py ladder   --out O --rung R [--grid-id ID] [--prefixes 30,60,120,300,600] [--dt 0.644|0.500|per_cell]
                              [--norm prefix|whole_cell] [--null-perm 0] [--n-jobs 1] [--seed-offset 0]
```

`--reduce-to-strongest` runs the `combined (matched)` row: `reduce_to = d*`, the feature count
of the strongest single rung under LOWO (by `NULL_VERDICT_STATISTIC`, AUC the tie-break), read
from the four single rungs' `scores.json`; written to `splits/combined/<grid>/lowo_matched__norm/`
(CR3 2.32's G-DIM; SPEC 3.7.7). `--null-splits` default `lowo` (LOCO's null costs 168 x 500
fits per rung under `loco_mode = cell`; section 7 item 11).

3.3.8 G-CAL, inside the split (CR3 2.18): `gcal = {per_fold_tpr05, pooled_tpr_at_fpr05,
difference, null_spread (p95 - p05 of the tpr05 null, or null), verdict}` with
`GCAL_AGREE` when `|difference| <= null_spread`, `GCAL_PERFOLD` otherwise, and
`not_run("null spread unavailable")` when the null did not run. Table 7 prints the per-fold
reading as the headline in every case and the pooled reading beside it labelled
`POST_HOC_LABEL` (ML 1.6 item 2); G-CAL says whether they agree.

3.3.9 The time-to-detect ladder (K3 move 17; CR3 2.29; N3 Sec. 1 RQ1, Figure 6).

```python
def prefix_rows(n_pairs_cell: int, prefix_s: int, dt) -> int:
    """The prefix length in pairs: round(prefix_s / dt) with dt a fixed spacing (LADDER_DT_S =
    0.644 s, the derived guest spacing; 0.500 s the configured interval) so that a 30 s prefix
    is 47 pairs, 60 s 93, 120 s 186, 300 s 466, 600 s 932 (K3 move 17 names about 46, 93, 186,
    465 and 930 at 645 ms), capped at the cell's series length (n_pairs - 1), so 600 s reads
    the whole cell; with dt = "per_cell" the cell's own spacing 600 / n_pairs_cell is used
    instead (round(prefix_s * n_pairs_cell / 600)). Both bracket readings and the per-cell
    reading are recorded per cell (pairs_at_0500, pairs_at_0644, pairs_per_cell)."""
def boundary_start(cell_id: str, boundaries: dict[str, list[int]], seq_first: int) -> int:
    """The row index of the first [SUSTAIN] boundary from inputs/iteration_boundaries.csv
    (cell_id, boundary_seqs as ';'-separated ascending seqs), or 0 when the cell has no
    boundary (a kernel honouring --duration has one iteration and nothing to drop; CR3 2.29
    applies the drop rule to every cell uniformly)."""
def run_ladder(out, rung, grid_id, *, prefixes_s=LADDER_PREFIXES_S, dt=LADDER_DT_S, readings=LADDER_READINGS,
               norm=LADDER_NORM, n_perm=0, **split_kw) -> Path:
    """For each reading and each prefix: build the prefix features (3.1.3), run
    run_detection_split with split lowo (and its null only when n_perm > 0) into
    ladder/<rung>/<grid_id>/<reading>/prefix<T>s/, and write ladder.csv (one row per rung,
    reading, prefix: rung, grid_id, reading, prefix_s, dt, n_rows_median, n_windows_median,
    n_cells_with_window, tpr_05, fpr_05_realized, auc, null_verdict, note). The reading
    'from_boundary' runs only when inputs/iteration_boundaries.csv exists and names at least
    one admissible cell; otherwise its rows carry note = LADDER_FROM_PAIR1_ONLY and empty
    numbers. A prefix shorter than one window at the rung's (W, H) for every cell reads
    not_applicable('prefix shorter than one window (n = <k> < W = <W>)'); cells without a
    window at a prefix are dropped for that prefix and counted in n_cells_with_window."""
```

### 3.4 `detection_levels.py`

```python
LEVEL2_MIN_MEMBERS_HEADLINE = 3      # G-N's headline count (CR3 2.5); P3 0a (A and C scored, B not)
LEVEL2_MIN_MEMBERS_TEST = 2          # 2 members: the row carries L2_ONE_TRAIN
LEVEL3_SPLIT = "rep_index"           # section 7 item 17
def run_level2(out, rung, grid_id, *, normalized=True, n_perm=N_PERM, seed=..., seed_offset=0, n_jobs=1,
               n_estimators=N_ESTIMATORS, min_members_headline=LEVEL2_MIN_MEMBERS_HEADLINE,
               min_members_test=LEVEL2_MIN_MEMBERS_TEST) -> Path:
    """Level 2, which sub-family (P3 0a; G-N, CR3 2.5). Sandbox rows only (at-floor cells kept
    and marked). The forest of SPEC 4.2 (models.make_forest, oob off) as a multiclass model
    over the sub-family letters; folds = fold_level2; the cell's prediction is the majority
    vote over its windows (models.aggregate_units, ties by mean probability). Writes
    splits/<rung>/<grid_id>/level2/: confusion.csv (rung, true_subfamily, n_members,
    n_cells, pred_<letter> for every letter present, recall, status) where status is
    GN_HEADLINE when n_members >= min_members_headline, L2_ONE_TRAIN when n_members ==
    min_members_test, level2_no_heldout(n) when n_members < min_members_test (the row's
    counts empty); members.csv (rung, member_index, subfamily_letter, hits, denominator,
    eighths); predictions.csv; scores.json (macro_recall over the headline rows, never over
    all three (P3 0a), n_folds, null); null.json. Null: the letters permuted across members
    with the counts kept (the multiset permutations; exhaustive when their number is below
    NULL_EXHAUSTIVE_BELOW; 8!/(4!1!3!) = 280 on stage 1), the statistic = the headline macro
    recall, null_verdict as 3.3.4 (NULL_NOT_ESTIMABLE below 20 distinct assignments)."""
def run_level3(out, rung, grid_id, *, normalized=True, split=LEVEL3_SPLIT, n_perm=N_PERM, ...) -> Path:
    """Level 3, the exact member under leave-one-rep-out (P3 0a), labelled SIGNATURE_CEILING
    on every row. The same multiclass forest over member indices; folds = fold_level3;
    confusion.csv (rung, true_member, pred_1 .. pred_S, recall, label = SIGNATURE_CEILING);
    members.csv as level 2; scores.json (accuracy, per-member recall, null); null = the
    per-cell member-label vector permuted across sandbox cells (the kernel-space rule of
    nulls.shuffle_labels_units, since the label is the workload itself), 500 permutations."""
```

CLI: `detection_levels.py level2 --out O --rung R [--grid-id ID] [--null-perm 500] [--n-jobs 1]
[--seed-offset 0]`; `detection_levels.py level3 --out O --rung R [--grid-id ID] [--split rep_index|cell]
[--null-perm 500] [--n-jobs 1] [--seed-offset 0]`.

### 3.5 `gates_detection.py`

Every function here writes one CSV under `gates/detection/` plus a `.params.json` beside it
(`series.write_params`), replaces its own rows on a re-run (keyed by `rung` and the row key),
and never edits another gate's file. Every function reads the split results of 3.3 by file and
never re-fits a model except where the definition says "recomputed" (G-LM, G-FP), and then it
calls `run_detection_split` with `subset_workloads` or `exclude_families`.

3.5.1 Admissibility.

```python
DET_C1_RULE = "report"     # section 7 item 7
def admissibility(out: Path, *, det_c1_rule: str = DET_C1_RULE) -> Path:
    """gates/detection/admissibility.csv: cell_id, class, status_cells_csv, all_hard_pass, C1,
    C2, C6, failed_verdict, c1_rule_applied, admissible, admissible_pair_rungs, reason.
    admissible = status ok and C2 == pass and C6 == pass and (C1 in (pass, not applicable) or
    (C1 == fail and class != benign_kernel and det_c1_rule == 'report')); a cell admitted
    through the report rule has reason 'C1 fail on a non-kernel class: reported, floor
    verdict by G-K0 (K3 F3)'. admissible_pair_rungs additionally needs failed_verdict not a
    refusal (SPEC 3.3.2). With gates/preconditions.csv absent every row reads
    admissible = false, reason = not_run('gates/preconditions.csv missing'). Citation: CR3 2.23
    (the validity check per cell, extended; refusals written, never absorbed); K3 move 2; SPEC 3.3.1."""
```

3.5.2 G-K0 for every cell, three quantities, three verdicts (CR3 2.8; K3 F3; K3 move 5).

```python
GK0_IDLE_PERCENTILE = 95.0           # the idle band edge: plan11's rule (SPEC 3.3.3), the same number
GK0_ENVELOPE_PERCENTILE = 95.0       # the envelope's quantile per quantity; OPEN in CR3 2.8, section 7 item 14
GK0_VERDICT_QUANTITIES = ("K_med", "K_q90", "frac_above_band")    # section 7 item 14
def gk0_cells(out: Path, *, idle_percentile=GK0_IDLE_PERCENTILE, envelope_percentile=GK0_ENVELOPE_PERCENTILE,
              verdict_quantities=GK0_VERDICT_QUANTITIES) -> Path:
    """Source part: inputs/gk0_source_sandbox.csv (member_index, steady_state_changes_content,
    source), template written by `gk0-sandbox-template` with one numbered row per member and
    'unstated'; the author fills it; agents write nothing about the members (CR3 2.8, ML 4.2).
    Measured part, per admissible cell of every class, from extract.csv: K_med (median K over
    all rows), K_q90 (90th percentile of K), frac_above_band (fraction of rows with K > the idle
    band edge, the idle_percentile-th percentile of K pooled over every admissible idle cell's
    rows, identical to gates_precondition.gate_gk0's edge; when gates/gk0.csv exists its
    idle_band_edge is read and the difference asserted <= 1e-9, else computed here), l0_med_above
    (median of l0_q50_all over the rows above the band; blank when none), J_consec_above
    (median J over the pairs above the band; blank when none). Envelopes: for each quantity
    the envelope_percentile-th percentile over the idle cells' per-cell values (the idle
    envelope) and, when harness_idle cells exist, over the harness-idle cells' (the harness
    envelope). Verdict per cell, in order: GK0_AT_FLOOR when every quantity in
    verdict_quantities is <= its idle envelope edge; GK0_AT_HARNESS_FLOOR when not at floor,
    harness cells exist, and every quantity is <= its harness envelope edge; else
    GK0_ABOVE_FLOOR (verdicts.py, existing). Idle and harness-idle cells themselves carry
    verdict 'control'. Without an admissible idle cell every verdict is not_run('no admissible
    idle cell'). Writes gk0_cells.csv (cell_id, class, workload_key, member_index, K_med, K_q90,
    frac_above_band, l0_med_above, J_consec_above, idle_band_edge, env_K_med, env_K_q90,
    env_frac, harness_env_K_med, harness_env_K_q90, harness_env_frac, verdict),
    gk0_members.csv (member_index, subfamily_letter, source_statement, n_cells, n_at_floor,
    n_at_harness_floor, n_above_floor, verdict = the common verdict or GK0_MIXED), gk0.json
    (the envelopes, the edge, the params). Citation: CR3 2.8 (three quantities, three verdicts,
    the source part by the author); K3 F3; ML 3.6 ('at floor: undetectable by construction',
    excluded from the true-positive denominator)."""
```

3.5.3 G-N two-class (CR3 2.5; ML 2.6, 4.2).

```python
GN_SANDBOX_HEADLINE_MIN = 3
def gn_two_class(out: Path) -> Path:
    """gn.csv: one row for the sandbox class (row = 'sandbox', n_workloads = the members with
    at least one admissible above-floor cell, status GN_HEADLINE when >= 3, GN_ONE_TRAIN_WORKLOAD
    when 2, GN_NO_SUPERVISED when 1 or 0) and one row per benign family (row = the family,
    n_workloads, status GN_HEADLINE when >= 2 else GN_SINGLE_WORKLOAD); columns row, class,
    n_workloads, n_cells, workloads (public keys), status. Refusal carried into the tables: a
    family-level sentence about the sandbox family is not written below three members; Table 7's
    note prints the sandbox row's status. Citation: CR3 2.5; ML 2.6."""
```

3.5.4 G-L (i) two-class (CR3 2.4; ML 3.6, 4.2; K3 F1).

```python
def gl_two_class(out: Path, rung: str, grid_id: str | None = None) -> Path:
    """gl.csv row per rung: rung, grid_id, tpr05_norm, null_p95_norm, rank_norm, tpr05_raw (apf
    only; else ''), verdict: PASS when the normalized LOWO tpr05 null verdict is PASS;
    GL_LEVEL_ONLY when it is NULL_INSIDE (the rung's row is marked and cannot be cited as
    detecting behaviour; the raw row stays as the level-inclusive ceiling); the null's own
    refusal (NULL_NOT_ESTIMABLE, not run) is copied when the null did not decide. Part (ii)
    (shot noise) is paper 2's and is not re-run here; part (iii) is G-LM. Citation: CR3 2.4."""
```

3.5.5 G-OP (CR3 2.13; ML 2.3, 4.3).

```python
GOP_MIN_CELLS = 5; GOP_MIN_WORKLOADS = 3
GOP_CELLS = "threshold_setters"     # section 7 item 18; alternative "realized_fps"
def gop(out: Path, rung: str, grid_id: str | None = None, *, split: str = "lowo", variant: str = "norm",
        min_cells=GOP_MIN_CELLS, min_workloads=GOP_MIN_WORKLOADS, cells_rule=GOP_CELLS) -> Path:
    """gop.csv row per (rung, split, variant): fpr_declared, threshold_in_fold (true),
    n_setter_cells, n_setter_workloads, setter_families, n_realized_fp_cells,
    n_realized_fp_workloads, fp_families, verdict. The cells behind the declared rate are, under
    'threshold_setters', the union over folds of the benign training cells whose out-of-bag
    score is >= the fold's threshold_05 (folds.json setters_05); under 'realized_fps' the
    out-of-fold benign cells flagged at threshold_05. verdict = GOP_SUPPORTED when n_cells >=
    min_cells and n_workloads >= min_workloads, else GOP_SET_BY_FEW with the families named in
    setter_families; the true-positive rate at that operating point carries no verb (Table 7's
    G-OP column and note). Citation: CR3 2.13; ML 2.3."""
```

3.5.6 G-LM, the level-matched control, G-L part (iii) (CR3 2.17; ML 4.3; K3 F1; VME's
correction of the level quantity).

```python
LEVEL_QUANTITY = "median_K"         # stage 1; "per_iteration_K_sum" when inputs/iteration_boundaries.csv exists; section 7 item 16
GLM_BAND_FACTOR = 2.0                # ML question 13: a factor of two
GLM_MODEL = "retrain"                # section 7 item 16
GLM_VANISH_RULE = "tpr_le_fpr"       # section 7 item 16
def level_of_workload(out, join, *, quantity=LEVEL_QUANTITY, iteration_quantile=0.5) -> dict[str, float]:
    """The level per workload key: the median over its admissible cells of the cell's level.
    median_K: series.k_median_cell(extract, head_drop) (the same statistic as plan11's
    normalization), labelled GLM_LABEL_MEDIAN_K in every row. per_iteration_K_sum: from
    inputs/iteration_boundaries.csv, the sum of K over each iteration (between consecutive
    boundary seqs; the first iteration dropped, CR3 2.29) and the iteration_quantile of those
    sums per cell (0.5 = the median iteration; 'or a declared upper quantile', K3 F1); a cell
    without boundaries under this quantity reads not_run('no iteration boundary for <cell>')
    and the workload's level is blank."""
def glm_verdict(recall_unrestricted, fpr_unrestricted, recall_lm, fpr_lm, n_band_cells, n_band_workloads,
                *, rule=GLM_VANISH_RULE, min_cells=GOP_MIN_CELLS, min_workloads=GOP_MIN_WORKLOADS) -> str:
    """A pure function (tested with a table of inputs, 4.5). GLM_EMPTY_BAND when
    n_band_workloads == 0; GLM_NOT_DETECTED when recall_unrestricted <= fpr_unrestricted (there
    is no detection to vanish; the miss table names what the member resembles);
    GOP_SET_BY_FEW + ' (level band)' when the band holds fewer than min_cells cells or
    min_workloads workloads (the level-matched operating point has no support; the
    level-matched recall is written but carries no verb); else GLM_LEVEL_ONLY when detection
    vanishes (rule 'tpr_le_fpr': recall_lm <= fpr_lm, the random scorer's line, ML 1.5; rule
    'below_half_unrestricted': recall_lm <= 0.5 * recall_unrestricted), else GLM_SURVIVES."""
def glm(out, rung, grid_id=None, *, quantity=LEVEL_QUANTITY, band_factor=GLM_BAND_FACTOR, model=GLM_MODEL,
        vanish_rule=GLM_VANISH_RULE, n_jobs=1, seed_offset=0) -> Path:
    """Per (rung, member m) on the norm features (and raw for apf); a member whose every cell
    is at floor gets the row verdict not_applicable('at floor (G-K0)') and no run. L = the
    member's level; the band [L / band_factor, L * band_factor]; the level-matched benign =
    every benign workload whose level lies in the band. model 'retrain': run_detection_split(split='lowo',
    subset_workloads = every sandbox member + the band's benign workloads, dir_name =
    'glm_<m>'): the member's recall at the in-fold threshold set on the band benign's out-of-bag
    scores, and the realized FPR on the band benign folds. model 'headline_oof': no retrain;
    the headline LOWO's out-of-fold scores, the threshold recomputed as the (1 - fpr) quantile of
    the band benign's out-of-fold scores (labelled POST_HOC_LABEL). glm.csv columns: rung,
    grid_id, member_index, subfamily_letter, level_quantity, level_label, level, band_lo,
    band_hi, band_workloads, n_band_cells, n_band_workloads, recall_unrestricted,
    fpr_unrestricted, recall_lm, fpr_lm, threshold_lm_median, verdict. Every row's level_label
    is GLM_LABEL_MEDIAN_K in stage 1. Citation: CR3 2.17 (the operating point recomputed
    against only the benign workloads whose level lies within a declared band); ML 4.3; VME
    section 4 item 7 through K3 F1 (the per-iteration changed-page count)."""
```

3.5.7 G-ANCHOR and the order test (CR3 2.14, 2.20; ML 3.1, 3.2; K3 F5; P3 0a "the order test
is a required row of RQ3").

```python
ORDER_TEST_CONSEQUENCE = "size"      # stage 1: blocked by member, read as a size (P3 0a); "void" for an interleaved campaign (K3 F5); section 7 item 19
ORDER_NULL_PERM = 500
def anchor(out, *, part: str = "all", n_perm=500, n_jobs=1, seed_offset=0) -> Path:
    """ganchor.csv rows:
    part (i), 'kernels': per rung, campaign predictability from level-normalized features on
      the benign kernels under LOKO with label = campaign, read from gates/gx.csv (the driver
      runs gates_comparison gx --rung R per rung; the same forest, the same features, 500
      campaign-label shuffles); columns rung, part, label_space, n_cells, n_labels, score,
      null_p95, rank, verdict = GANCHOR_AUDIBLE when leak_verdict == GX_LEAK, GANCHOR_NOT_AUDIBLE
      when GX_POOLING_STANDS, the string itself when gx wrote not applicable; not_run('gates/gx.csv
      missing (run gates_comparison gx)') when absent. Read as a size, never as a switch between
      two headlines (K3 section 5 point 3).
    part (ii), 'idle_sets': per rung, the idle cells (class idle; harness_idle as a second block
      when present) under leave-one-cell-out with label = campaign (run_operating_point with
      label_key 'campaign', positive = the later campaign in sorted order; score = AUC), null =
      500 campaign-label shuffles across the idle cells (cell-level, as plan11's G-X);
      not_applicable('one idle campaign (n = <k> cells)') when the idle cells carry one label.
    'idle_early_late': per rung, the idle cells with label = first half against second half by
      order_index within the class, leave-one-cell-out, AUC, the cell-level shuffle null;
      not_run('order_index missing for idle') when absent. The drift clause of the tripwire
      (CR3 2.12): a rung whose idle_sets row is GANCHOR_AUDIBLE cannot pool the two idle sets;
      a rung whose idle_early_late row is ORDER_AUDIBLE writes no pooled verdict until the
      order test has run; both are printed in Table 6 and in the driver's tripwire check.
    Citation: CR3 2.14; ML 3.1; K3 move 4 and 21."""
def order_test(out, *, consequence=ORDER_TEST_CONSEQUENCE, n_perm=ORDER_NULL_PERM, n_jobs=1, seed_offset=0) -> Path:
    """order.csv: per rung and per class with >= 2 workloads and order_index on every
    admissible cell: the half label (first / second by the rank of order_index within the
    class; the first floor(n / 2) cells are 'first'); the folds fold_order (LOWO within the
    class); score = AUC (positive = second); null: when every workload lies inside one half
    the workload-level permutation of the half labels with the count of first-half workloads
    kept (the label is a per-workload constant, ML 1.4), else a permutation of the per-cell
    half vector within the class (recorded order_null_unit = 'cell' with the reason);
    verdict ORDER_AUDIBLE on strict exceedance else ORDER_NOT_AUDIBLE; consequence 'size': the
    verdict is a row of Table 9; 'void': ORDER_VOID replaces ORDER_AUDIBLE and Table 7 prints
    it in every score cell of that rung (K3 F5). Columns: rung, class, n_cells, n_workloads,
    half_rule, order_null_unit, score, null_p95, rank, n_assignments, verdict, consequence.
    Stage-1 note written into params: the sandbox cells ran in order by member, so member
    identity and position coincide by construction (P3 0a) and the row is a size."""
def drift_regression(out, *, n_perm=ORDER_NULL_PERM, seed_offset=0) -> Path:
    """drift.csv: per class with order_index, the per-cell floor statistic (K_med from
    gk0_cells.csv) regressed on order_index by ordinary least squares: slope, intercept, r2,
    n; null = n_perm shuffles of order_index within the class, the 95th percentile of
    |slope|; verdict DRIFT_SLOPE when |slope| strictly exceeds it, else DRIFT_NONE. A drift
    disclosure for Table 1 / Table 6 (CR3 2.20 'a regression of the per-cell floor statistics on
    the cell index')."""
```

3.5.8 G-SIG (CR3 2.16; ML 4.3; K3 F2).

```python
def gsig(out, rung, grid_id=None) -> Path:
    """gsig.csv row per rung: tpr05_loco, tpr05_lowo, gap_tpr05 (= loco - lowo), auc_loco,
    auc_lowo, gap_auc, loco_null_verdict, lowo_null_verdict, verdict: GSIG_IDENTITY when the
    LOCO tpr05 null verdict is PASS and the LOWO one is NULL_INSIDE (the rung cannot be cited
    for detection, only for signature matching with that sentence); GSIG_REPORTED otherwise;
    not_run('<split> null not run') when either null is absent (the gap number still written).
    Citation: CR3 2.16."""
```

3.5.9 G-FP (CR3 2.15; ML 4.3; K3 F8).

```python
GFP_FLAG_FRACTION = 0.5
def gfp(out, rung, grid_id=None, *, flag_fraction=GFP_FLAG_FRACTION, n_jobs=1, seed_offset=0) -> Path:
    """gfp.csv row per (rung, benign family) from the LOWO norm predictions: n_cells, n_flagged_05,
    fraction, verdict = GFP_INSEPARABLE when fraction >= flag_fraction else GFP_ATTRIBUTED; for
    an inseparable family the operating point recomputed without it (run_detection_split with
    exclude_families=(family,), dir_name 'lowo__without_<family>'): tpr05_without, fpr05_without,
    and the with-family numbers beside. Refuses nothing; it prevents a family from being averaged
    away. Citation: CR3 2.15."""
```

3.5.10 G-1C (CR3 2.19; K3 section 5 point 1).

```python
def g1c(out, rung, grid_id=None) -> Path:
    """g1c.csv row per (rung, one-class model directory): model, primary, tpr05, fpr05_realized,
    tpr01, auc, threshold_source, label = G1C_PRIMARY for the declared primary, G1C_SECONDARY for
    every other; verdict G1C_SEARCH on the rung when more than one directory exists and none is
    primary (cannot arise through the CLI, which requires --secondary for a non-primary; written
    if the author copies directories by hand). K3 F9 is read by builder B in Table 11: the
    one-class tpr05 inside its null while LOWO's is above it prints the sentence 'separable when
    heard, not flagged when not' in the note column (a label, not a verdict string of this
    module). Citation: CR3 2.19."""
```

3.5.11 The harness clause (CR3 2.12, 2.21; K3 F4; P3 0a: stage 2 absent in stage 1).

```python
HARNESS_COMPARABLE_TOL = 0.10        # OPEN in CR3 2.21; section 7 item 22
HARNESS_RELAUNCH_RULE = "ge_median_member_recall"
def harness(out, rung, grid_id=None, *, comparable_tol=HARNESS_COMPARABLE_TOL, relaunch_rule=HARNESS_RELAUNCH_RULE) -> Path:
    """When the classes benign_relaunched and harness_idle are both absent from
    cell_classes.csv, harness.csv holds one row per rung with verdict HARNESS_STAGE2_ABSENT and
    empty numbers. Otherwise, per (rung, feature) on the norm features: the per-cell feature
    (mean over the cell's windows) and three margins, each 2 * |AUC - 0.5| of the feature
    between two groups: margin_sb (sandbox against every benign cell), margin_hi (harness_idle
    against idle), margin_rp (benign_relaunched against its parent kernel's cells, pooled over
    the re-launched pairs); label HARNESS_COMPARABLE when |margin_sb - margin_hi| <= tol and
    |margin_sb - margin_rp| <= tol (the feature is the harness's and leaves the evidence), else
    HARNESS_CLASS_EXCEEDS. A second block, one row per rung: relaunched_flagged_fraction at the
    LOWO operating point beside median_member_recall; HARNESS_RELAUNCH_NOT_CLASS when the
    control's flagged fraction >= the median member recall ('a benign re-launched control
    flagged at the sandbox members' rate'). Columns: rung, block, feature, margin_sb, margin_hi,
    margin_rp, relaunched_flagged_fraction, median_member_recall, verdict. Citation: CR3 2.21;
    K3 F4 and move 8; CR3 2.12 (the harness clause of the tripwire)."""
```

3.5.12 G-CAL: computed inside the split (3.3.8); `gcal(out, rung)` copies the per-rung values
into `gcal.csv` (`rung, per_fold_tpr05, pooled_tpr_at_fpr05, difference, null_spread, verdict`).

3.5.13 G-M two-class (CR3 2.7; ML 2.5) and G-DIM (CR3 2.2 unchanged; SPEC 3.7.7).

```python
GM_ALPHA = 0.05; GM_N_SEEDS = 5
def exact_sign_test(improving: int, worsening: int) -> float:
    """P(Bin(improving + worsening, 1/2) <= worsening), the exact one-sided binomial over the
    non-tied units (ML 2.5: five up and none down is 0.031; seven up and one down of eight is 0.035)."""
def gm(out, *, n_seeds=GM_N_SEEDS, alpha=GM_ALPHA, split="lowo", n_jobs=1, seed_offset=0) -> Path:
    """gm.csv: every ordered pair of rows among (apf raw, apf, wapf, persist, content, combined,
    combined (matched), and 'comparator' if splits/comparator/ exists in this layer's schema):
    per_workload_outcome of A minus of B per workload (recall for a sandbox workload, one minus
    its flagged fraction for a benign workload; a tie is neither); improving, worsening, ties,
    p_exact; spread = max - min of the apf LOWO tpr05 over n_seeds forest seeds (SEED_FOREST + i,
    written under splits/apf/<grid>/lowo_seed<i>__norm/ through dir_name); verdict GM_BEATS when tpr05_A -
    tpr05_B > spread and p_exact <= alpha, else GM_DIFFERENCE. On stage 1 most benign workloads
    tie, so the non-tied units are mostly the eight members (ML 2.5). Citation: CR3 2.7."""
def gdim(out) -> Path:
    """gdim.csv: rung, d, d_used, d_matched (combined (matched) only), method, status
    (GDIM_FULL | GDIM_REDUCED) from every LOWO scores.json. Citation: CR3 2.2 (G-DIM unchanged);
    SPEC 3.7.7."""
```

3.5.14 The alias falsifier, detection form (CR3 2.26; K3 move 19).

```python
ALIAS_TOP_K = 10; ALIAS_R2 = 0.5
def alias_detection(out, rung, grid_id=None, *, top_k=ALIAS_TOP_K, r2_threshold=ALIAS_R2) -> Path:
    """alias.csv: for the top_k features by importance_mean of the LOWO norm run (every feature
    when d <= top_k): the per-cell feature (mean over windows) regressed across all admissible
    cells on dt_est_s from the sidecars (slope, r2, n, verdict ALIAS_MOVES when r2 > r2_threshold
    else ALIAS_STAYS) and, when inputs/iteration_counts.csv exists (cell_id, iteration_count), on
    the iteration count likewise; else the second block reads not_run('no iteration count
    (stage 1)'). Both regressors are logs, never features. Columns: rung, feature, regressor,
    slope, intercept, r2, n, dt_spread_s, verdict. Citation: CR3 2.26; SPEC 3.4.4 (the epoch-1 form,
    gates_calibration.alias_falsifier, whose verdict strings are reused)."""
```

3.5.15 G-V two-class (CR3 2.11; K3 move 20).

```python
def gv_two_class(out, rung, grid_id=None) -> Path:
    """On the norm features at the grid point, the per-cell vector = the mean over the cell's
    windows. Per feature: L0 = mean over sandbox members of the population variance across the
    member's cells; L2 = the population variance across member means; L3 = the population
    variance across the two class means; and for the benign side L0_b (mean over benign
    workloads of the within-workload variance), L2_b (variance across benign workload means
    within a family, averaged over families with >= 2 workloads), L3_families (the population
    variance across benign family means). gv_two_class.csv: rung, feature, L0, L2, L3, L0_b,
    L2_b, L3_families, L3_over_L3_families; gv_two_class_summary.csv: rung, n_features,
    n_features_L3_le_L3_families, note = 'a class whose L3 is inside the benign families'
    mutual spread has no more form than any two families have between them' when the count
    is at least half of the features, else ''. No refusal; a report."""
```

3.5.16 The miss table (K3 move 18; N3 Sec. 1 RQ2, Table 10).

```python
MISS_AXES = ("amount", "identity")
MISS_DISTANCE = "standardized_euclidean_to_centroid"   # section 7 item 25
def miss_table(out, rung, grid_id=None, *, distance=MISS_DISTANCE) -> Path:
    """The plane: per cell, amount = the median over its rows of r_l0_q50_per (the content
    headline, level-free) and identity = the median over its rows of J - J_null (the persistence
    channel against its independence null); both from extract.csv, both level-free and
    placement-free (K3 move 11). Each axis standardized by its population std over every
    admissible cell. Per benign workload the cloud = its cells' points and its centroid. For
    every sandbox cell missed under LOWO at the operating point (flag_05 false, in_denominator
    true): the nearest benign workload by the standardized Euclidean distance to its centroid,
    its family, the distance, the axis of smallest distance (the axis whose absolute
    standardized difference to the centroid is smaller), the cell's and the centroid's
    coordinates. A sandbox cell at floor is listed with verdict AT_FLOOR_NOT_A_MISS and no
    neighbour. miss_table.csv: cell_id, member_index, subfamily_letter, rung, score, threshold_05,
    nearest_workload, nearest_family, distance, axis, amount_cell, identity_cell,
    amount_centroid, identity_centroid, status. For every benign false positive at the
    operating point: fp_table.csv: cell_id, family, workload_key, rung, score, threshold_05,
    nearest_member_index, nearest_subfamily_letter, distance, axis, amount_cell, identity_cell.
    Read as assignments with counts, never as a confusion matrix in the statistical sense.
    The physical reason (M1 to M6) is the author's column in Table 10 and is left empty.
    Citation: K3 move 18; N3 Sec. 1 RQ2."""
```

3.5.17 The CLI of `gates_detection.py` (every subcommand takes `--out O`; the driver calls
these names):

```
gates_detection.py admissibility       [--c1-rule report|exclude]
gates_detection.py gk0-sandbox-template
gates_detection.py gk0-cells            [--idle-percentile 95] [--envelope-percentile 95]
gates_detection.py gn
gates_detection.py gl        --rung R [--grid-id ID]
gates_detection.py gop       --rung R [--grid-id ID] [--split lowo] [--variant norm|raw] [--cells-rule threshold_setters|realized_fps]
gates_detection.py glm       --rung R [--grid-id ID] [--level-quantity median_K|per_iteration_K_sum] [--band-factor 2.0]
                             [--model retrain|headline_oof] [--vanish-rule tpr_le_fpr|below_half_unrestricted] [--n-jobs 1] [--seed-offset 0]
gates_detection.py anchor    [--part all|kernels|idle_sets|idle_early_late] [--null-perm 500] [--n-jobs 1] [--seed-offset 0]
gates_detection.py order     [--consequence size|void] [--null-perm 500] [--n-jobs 1] [--seed-offset 0]
gates_detection.py drift     [--null-perm 500] [--seed-offset 0]
gates_detection.py gsig      --rung R [--grid-id ID]
gates_detection.py gfp       --rung R [--grid-id ID] [--flag-fraction 0.5] [--n-jobs 1] [--seed-offset 0]
gates_detection.py g1c       --rung R [--grid-id ID]
gates_detection.py harness   --rung R [--grid-id ID] [--comparable-tol 0.10]
gates_detection.py gcal      --rung R [--grid-id ID]
gates_detection.py gm        [--n-seeds 5] [--n-jobs 1] [--seed-offset 0]
gates_detection.py gdim
gates_detection.py alias     --rung R [--grid-id ID] [--top-k 10] [--r2-threshold 0.5]
gates_detection.py gv        --rung R [--grid-id ID]
gates_detection.py miss-table --rung R [--grid-id ID]
```

Exit codes as SPEC 7.1: 0 on success (a written refusal is a success), 2 when an input file is
missing (its path on stderr), 1 on an internal error. In every CLI of this layer `--grid-id`
defaults to the rung's point from `gates/selection.json` (`series.selected_grid_id(out, rung,
None)`); when no selection exists the command writes `not run: no selection for <rung> (run
classes inherit-selection)` into its result file and exits 0, so no detection stage ever runs
at an undeclared grid point.

---

## 4. The synthetic two-class corpus (`synth_detection.py`, builder A)

### 4.1 What it reuses

`synth.write_cell`, `synth.SynthSpec` (with the two additive fields of 1.3), `synth.PRESETS`,
`synth.IDLE_PRESET`, `synth.rep_seed`, `synth.corpus_specs` for the benign side. The benign
side of the detection corpus is the encoding corpus of SPEC 5.2 (12 kernels x `reps` at the
presets, `idle` idle cells) built by `synth.corpus_specs(cv_case="preset", ...)` so that every
kernel keeps its preset `k_noise` and the class separations below are by construction (the
`random` draw of `k_noise` is not used here). G-C's orderings therefore hold on this corpus as
on SPEC 5.2's, and `gates_calibration gc` passes on it.

### 4.2 The sandbox side

Eight members under the family directory `synthfam` with `test_label_fmt =
"synthfam_member_{name}"` and `name = str(m)`, so `schema.parse_cell_path` yields family
`synthfam`, kernel `synthfam_member_<m>`, role `unknown`, and the index lists the cells as
`refused: unknown kernel` until `classes apply` runs (the same path the real cells take).
`write_cell` is unchanged: the trajectory file inside the directory is named by
`synth.TRAJ_NAME_FMT` with `name = "<m>"` (`run_matrix_test1_kernel_<m>_v2.npy.substrate_trajectory.csv[.zst]`),
which carries no workload name, and the extractor finds it by the glob `*substrate_trajectory.csv*`.

| member | sub-family | preset | the known answer it carries |
|---|---|---|---|
| 1 | A | `K0 = 2048, content = "spin", spin_bytes = (48, 96), churn = 0.02, k_noise = 0.02` | at a benign level (floyd, histogram, nbody); separable on the amount axis: `r_l0_q50_per` about 72 / 4096 = 0.018 against gibbs's 0.001, the double kernels' 0.125 and histogram's 0.0004; `r_l1l0 = 1`, `r_haml0` about 2 |
| 2 | A | as member 1 with `K0 = 4096` | at gemm's and fft's level; the same amount signature |
| 3 | A | as member 1 with `K0 = 1024` | at spmm's level |
| 4 | A | as member 1 with `K0 = 3072` | at stencil_jacobi's level |
| 5 | B | `K0 = 2048, content = "double", churn = 0.60, k_noise = 0.02` | separable on the identity axis only: `J` about (1 - 0.6) / (1 + 0.6) = 0.25 against the kernels' 0.54 (rmat_gen) and above; no other member shares the axis, so under LOWO it is the miss whose nearest benign is rmat_gen on the identity axis, and the one-class reading flags it |
| 6 | C | `K0 = 0, floor_F = 150, floor_churn = 0.02, content = "idle"` (the idle preset) | at floor: G-K0 `at floor` on every cell; leaves the denominator; `AT_FLOOR_NOT_A_MISS` |
| 7 | C | the `spmm` preset exactly (`K0 = 1024, content = "double", churn = 0.05`, spmm's `k_noise`) | a pure level copy of a kernel: missed on every rung; the miss table names `spmm` as its nearest benign; G-LM reads `GLM_NOT_DETECTED` |
| 8 | C | `K0 = 40000, content = "double", churn = 0.05, k_noise = 0.02` | a level no benign workload reaches: G-LM's band [20000, 80000] is empty (`GLM_EMPTY_BAND`, R8's benign gap); missed on the normalized rungs (its shape is nbody's) |

Rep seeds: `synth.rep_seed(r, 12 + m)` (rep 0 = seed 42, then `1000 r + 12 + m`), so plan11's
rep rule assigns reps 0 to 7 by seed. Labels: `label = "synth"` for every cell unless
`--campaign-labels` says otherwise (4.3).

### 4.3 The switches

```
synth_detection.py corpus --root R [--n-pairs 240] [--reps 8] [--idle 8] [--kernels all|<comma list>]
                          [--seed 20260916] [--order-confound on|off] [--campaign-labels one|round_robin|confounded]
                          [--idle-campaigns 1|2] [--write-classes] [--write-boundaries] [--no-compress]
                          [--members all|<comma list of indices>]
```

- The realized order. In both settings the classes occupy consecutive ranges of
  `order_index`, the stage-1 shape: the kernel cells 1 to 96, then the member cells, then the
  idle cells. `--order-confound off` (default): within each class the halves are interleaved
  inside every workload: the kernels' cells are first laid out as position `8 j + r` for kernel
  `j` in `schema.KERNEL_NAMES` order and rep `r`, then the even positions are listed before the
  odd ones, so every kernel has four cells in each half; the members likewise from position
  `8 (m - 1) + r`; the idle cells likewise from their rep. `k_noise` is the preset's for every
  cell. The order test reads `ORDER_NOT_AUDIBLE` for `sandbox` and `benign_kernel` and its
  null unit is the cell (no workload lies inside one half). `on`: the positions stay in blocks
  (the kernels by archetype then by kernel, the members by member, the stage-1 shape) and
  every member's `k_noise` is `0.02 + 0.03 * (m - 1)` (a monotone ramp with position, visible
  after level normalization through `cov`); the kernels carry the same ramp over their block
  index (`0.02 + 0.01 * block`). The order test reads `ORDER_AUDIBLE` for both classes with
  the null unit at the workload. The drift regression on `K_med` reads `DRIFT_NONE` in both
  settings (the ramp is on jitter, not level) unless `--drift-level` is also given, which
  multiplies every cell's `K0` by `1 + 0.002 * order_rank` (the level drifts with position:
  `DRIFT_SLOPE`).
- `--campaign-labels one` (default): every kernel cell labelled `synth`. `round_robin`: the
  labels `sandbox_deepdive_01c`, `sandbox_deepdive_01c1`, `dwarfs1_synth` assigned to the
  kernels' cells round-robin by rep index so that every kernel sits in every campaign
  (G-ANCHOR part (i) `GANCHOR_NOT_AUDIBLE`, as plan11's G-X pass case). `confounded`: each
  archetype's kernels in one campaign and each campaign's cells with its own `k_noise` (0.02,
  0.05, 0.08), visible after normalization (`GANCHOR_AUDIBLE`).
- `--idle-campaigns 2`: the idle cells split 4 + 4 into labels `sandbox_deepdive_01c` and
  `dwarfs1_synth` with `floor_F = 150` and `190` (G-ANCHOR part (ii) `GANCHOR_AUDIBLE`); `1`
  (default) gives one label (`not applicable: one idle campaign`).
- `--write-classes` writes `<root>/classes.csv` in the format of 2.1 with the eight member rows
  (`synthfam/synthfam_member_<m>` as `path_prefix`, sub-families A = 1..4, B = 5, C = 6..8), the
  kernel row (`kernel`), the idle row (the idle test label's prefix, class `idle`), and one row
  per cell with `order_index` laid out by the `--order-confound` setting; every test
  copies it to `<out>/inputs/classes.csv` (a test that needs a file without `order_index`
  drops those rows). Variants for the validator's refusing cases are written by the tests
  themselves from this file (an unknown class, a sandbox row without a letter, a duplicate
  prefix, and so on; 4.5).
- `--stage2-fixture` adds four cells of class `benign_relaunched` (the gemm preset under the
  family directory `relaunch` with `test_label_fmt = "relaunched_{name}"`, `name = "gemm"`) and
  four of class `harness_idle` (the idle preset with `floor_F = 300` under the family directory
  `harnessidle`, `test_label_fmt = "harness_{name}"`, `name = "idle"`), and their class rows
  (`workload_key = gemm` for the re-launched rows) in the written `classes.csv`, so that the
  harness clause, the harness floor of G-K0 and the stage-2 code paths have a fixture.
- `--write-boundaries` writes `<root>/iteration_boundaries.csv` from `truth.json`'s
  `boundaries` for every cell that has a pulse (gemm, floyd), so the ladder's `from_boundary`
  reading and the `per_iteration_K_sum` level quantity have a code path to run on; the tests
  copy it to `<out>/inputs/`.
- `--kernels` and `--members` restrict the corpus for fast tests; `--n-pairs 240` is the default
  so that the ladder's fixed-spacing prefixes (47, 93, 186 pairs; 300 s and 600 s capped at
  239) are distinct on the corpus; under `--dt per_cell` the synthetic spacing is `600 / 240 =
  2.5 s` per pair and the 30 s prefix is 12 rows (one `W8_H4` window), while at 120 pairs it is
  6 rows and reads `not applicable: prefix shorter than one window`, which is also a tested
  outcome.

### 4.4 Known answers on the full synthetic corpus (12 kernels x 8, 8 idle, 8 members x 8, `--n-pairs 240`)

Under `classes apply`, admissibility with `det_c1_rule = "report"`, the grid `W8_H4` by
`inherit-selection --default`, and `n_estimators = 300`:

| Quantity | Expected |
|---|---|
| G-K0 cells | member 6: 8 of 8 `at floor`; every other member's cells `above floor`; `gk0_members.csv` member 6 `at floor`, the rest `above floor`; the denominator 56 |
| LOWO `content` and `combined` (norm) | members 1 to 4 recall 8/8 each; member 5 recall 0/8 on `content`; members 7 and 8 0/8; realized FPR at or below 0.10; `null` verdict on `tpr05` PASS at 500 permutations |
| LOWO `persist` (norm) | members 1 to 4 near 0/8 (their J is the kernels'); member 5 0/8 (no sibling on its axis); member 7, 8 0/8 |
| LOWO `apf` raw | members 1 to 4 near 0/8 (benign levels); member 8 near 0/8 (no sandbox training cell near its level); member 7 0/8 |
| one-class `combined`, isolation forest | member 5's cells flagged (its identity coordinate is outside every benign cell's); members 1 to 4 flagged; member 7 not flagged; the FPR under leave-one-benign-workload-out at or below 0.10 |
| G-LM (`content`) | members 1 to 4 `GLM_SURVIVES`; member 7 `GLM_NOT_DETECTED`; member 8 `GLM_EMPTY_BAND`; member 6 `not applicable: at floor (G-K0)`; member 5 `GLM_NOT_DETECTED` |
| level 2 (`content`) | row A: recall 1.0 over 32 cells (`GN_HEADLINE`); row C: recall for members 7 and 8 against 6 unspecified but the row present with `GN_HEADLINE`; row B: `one member, no held-out test`; the null over 280 exhaustive assignments |
| level 3 (`combined`) | the 8 x 8 confusion counts and the accuracy row are reported; the test asserts the file's shape, the `SIGNATURE_CEILING` label on every row, and the null's presence, not a number (members 1 to 4 share every normalized feature by construction and are confused with each other) |
| the miss table (`combined`) | member 7's cells nearest `spmm`; member 5's cells nearest `rmat_gen` on the `identity` axis; member 8's cells nearest one of the double kernels on the `amount` axis; member 6's rows `AT_FLOOR_NOT_A_MISS` |
| G-OP (`combined`) | `GOP_SUPPORTED` (the setters come from several kernels) |
| G-N | sandbox `GN_HEADLINE` (7 members above floor); `kernels` `GN_HEADLINE`; `idle` `GN_SINGLE_WORKLOAD` |
| G-FP | no family `GFP_INSEPARABLE` on `combined`; on `persist` the kernels may be, and the test asserts the file's shape only |
| G-SIG | with `--null-splits lowo` every row `not run: loco null not run`; with `--null-splits lowo,loco --loco-mode rep_index` on the small corpus the verdict is `GSIG_REPORTED` on `content` |
| G-ANCHOR (i) | `round_robin` -> `GANCHOR_NOT_AUDIBLE`; `confounded` -> `GANCHOR_AUDIBLE` |
| G-ANCHOR (ii) | `--idle-campaigns 1` -> `not applicable: one idle campaign (n = 8 cells)`; `2` -> `GANCHOR_AUDIBLE` |
| order test | `off` -> `ORDER_NOT_AUDIBLE` for `sandbox` and `benign_kernel`; `on` -> `ORDER_AUDIBLE` for both; `--consequence void` with `on` -> `ORDER_VOID` |
| drift | `DRIFT_NONE`; with `--drift-level` `DRIFT_SLOPE` |
| harness | every row `HARNESS_STAGE2_ABSENT` |
| ladder (`content`, `from_pair1`, `--dt 0.644`) | tpr05 at 600 s (239 rows, the whole cell) equals the LOWO headline; at 30 s (47 rows, 10 windows at `W8_H4`) the number exists; `from_boundary` rows read `LADDER_FROM_PAIR1_ONLY` unless `--write-boundaries` was used |
| the letter sequence | with `order_index` present: 168 tokens, the first 96 `Br<r>`, then `S1r0 ... S8r7` under `on`, then `Ir0 ... Ir7`; without: the `not run` line |

### 4.5 Tests, one pair per gate (builder A) and the report tests (builder B)

Builder A's tests run on a reduced corpus (`--kernels gemm,floyd,gibbs,histogram,fft,lexer
--reps 3 --idle 4 --members all --n-pairs 120`, `n_estimators = 30`, `--null-perm 20`) except
where the row says "full"; a smoke-run null of 20 permutations writes `not run: 20 permutations
< 500` by design and the tests assert the numbers, not the verb, in those cases. `tests/_det_common.py`
builds the corpus once per session into a temp dir and runs `extract index`, `classes apply`,
`extract all`, `preconditions --c1-activity-min 0.001` (the lexer at floor and member 6 fail C1
at 0.001, gibbs at K0 = 256 plus the floor set passes; so the report rule of 3.5.1 is exercised
and the lexer, a `benign_kernel`, is excluded as the encoding paper's rule says), `inherit-selection --default W8_H4`,
`series features` for the five rungs at `W8_H4`, `gk0` and `gk0-cells`, then exposes `out`.

| Gate or function | must pass | must refuse or the named outcome |
|---|---|---|
| `classes validate` | the written `classes.csv` -> `status ok` | each of 2.2's items 1 to 18 from a one-line variant of the file -> that exact string |
| `classes apply` | every sandbox cell's `cell_id` is `sandbox_member_<m>__rep<rr>__synth`, `role = sandbox`, `status = ok`; `cells.pre_classes.csv` exists; a second `apply` changes nothing | with a refusing file: `cells.csv` byte-identical, the validation file written |
| the letter sequence | `on` corpus -> the token string of 4.4 | no `order_index` -> the `not run` line |
| `inherit-selection` | `--default W8_H4` -> five rungs, `grid_source` recorded; `--from` a copied plan11 `selection.json` -> verbatim payload | an existing `select`-written file without `--force` -> the refusal |
| `fold_lowo` | 5 + 1 + 8 folds on the reduced corpus (five kernels after the lexer's C1 exclusion, idle, eight members); `_assert_grouped` on `workload_key` and `cell_id`; a `test_only` class present -> the `lowo/final` fold | a hand-built label dict with one cell of a workload in train and one in test -> `AssertionError` from `_assert_grouped` |
| `fold_lofo` | 2 folds (`kernels`, `idle`); sandbox rows in every training set | a label dict with no benign family -> `[]` and `run_detection_split` writes `not applicable: no benign family` |
| `fold_one_class` | 6 + 1 folds; no sandbox row in any training set | (n/a) |
| `fold_level2` | 7 folds (A: 4, C: 3); B has none | `min_members_test = 5` -> `[]` |
| `in_fold_threshold` | 100 scores 0..99 at `fpr = 0.05` -> 94.05 (linear) | (a table test) |
| `run_operating_point` | a hand-built X with two separable classes -> tpr05 = 1.0, fpr05 = 0.0, auc = 1.0; the threshold setters per fold non-empty | X with the same distribution for both classes -> auc within [0.3, 0.7] |
| the null | `count_assignments(8, 13) = 203490`; `workload_label_permutations` with `S + B = 6, S = 2` -> exhaustive 15 subsets; `null_verdict` with `n_assignments = 19` -> `NULL_NOT_ESTIMABLE`; observed above p95 -> PASS; tie -> `NULL_INSIDE` | (a table test) |
| `run_detection_split lowo` (full corpus, `content`) | the 4.4 row | `--null-perm 20` -> `not run: 20 permutations < 500` in `null.tpr05.verdict` |
| `run_one_class` | `combined`: member 5 flagged (4.4) | a second model without `--secondary` when the primary directory exists -> a usage refusal (`refused: a second one-class model needs --secondary`) on stderr, nothing written, exit 2 |
| `quarantine_l1_two_class` | a corpus where one feature alone reproduces the forest -> that feature quarantined and the re-run present | the reduced corpus `content` -> empty quarantine |
| `run_ladder` | `--n-pairs 240`, `--dt 0.644`: five prefixes present with 47, 93, 186, 239, 239 rows; `from_boundary` rows `LADDER_FROM_PAIR1_ONLY`; with `--write-boundaries` the `from_boundary` rows carry numbers | `--n-pairs 120 --dt per_cell`: the 30 s row `not applicable: prefix shorter than one window (n = 6 < W = 8)` |
| level 2 | the 4.4 rows; the B row string `one member, no held-out test`; the null exhaustive 280 (full corpus) | `--members 1,2,5` -> every row `level2_no_heldout(n)` or `L2_ONE_TRAIN` and `scores.json.macro_recall` `not applicable: no headline sub-family` |
| level 3 | `confusion.csv` 8 x 8 with `SIGNATURE_CEILING`; the null present | `--members 1` -> `not applicable: one member` |
| admissibility | member 6's cells admissible under `report` with the reason string | `--c1-rule exclude` -> member 6's cells `admissible = false` |
| `gk0_cells` | member 6 `at floor`; member 1 `above floor`; the edge equals `gates/gk0.csv`'s | no idle cells -> `not run: no admissible idle cell` |
| `gn_two_class` | sandbox `GN_HEADLINE`; idle `GN_SINGLE_WORKLOAD` | `--members 1,2` -> `GN_ONE_TRAIN_WORKLOAD`; `--members 1` -> `GN_NO_SUPERVISED` |
| `gl_two_class` | `content` after a passing LOWO null -> PASS | a `scores.json` fixture with `null.tpr05.verdict = NULL_INSIDE` -> `GL_LEVEL_ONLY` |
| `gop` | `combined` -> `GOP_SUPPORTED` | a `folds.json` fixture whose setters come from one workload -> `GOP_SET_BY_FEW` |
| `glm_verdict` (pure function) | `(0.9, 0.05, 0.85, 0.06, 40, 5)` -> `GLM_SURVIVES` | `(0.9, 0.05, 0.05, 0.06, 40, 5)` -> `GLM_LEVEL_ONLY`; `(0.0, 0.05, ...)` -> `GLM_NOT_DETECTED`; `n_band_workloads = 0` -> `GLM_EMPTY_BAND`; `(0.9, 0.05, 0.9, 0.0, 8, 1)` -> `GOP_SET_BY_FEW (level band)` |
| `glm` on the corpus (`content`) | members 1 to 4 `GLM_SURVIVES` | member 8 `GLM_EMPTY_BAND`; member 7 `GLM_NOT_DETECTED` |
| `anchor` | `round_robin` -> `GANCHOR_NOT_AUDIBLE`; `--idle-campaigns 2` -> `GANCHOR_AUDIBLE` on (ii) | `confounded` -> `GANCHOR_AUDIBLE` on (i); one idle label -> the not-applicable string; `gates/gx.csv` absent -> the not-run string |
| `order_test`, `drift_regression` | `off` -> `ORDER_NOT_AUDIBLE`, `DRIFT_NONE` | `on` -> `ORDER_AUDIBLE`; `--consequence void` -> `ORDER_VOID`; `--drift-level` -> `DRIFT_SLOPE`; no `order_index` -> the not-run string |
| `gsig` | fixtures: LOCO PASS and LOWO PASS -> `GSIG_REPORTED` | LOCO PASS and LOWO `NULL_INSIDE` -> `GSIG_IDENTITY`; LOCO null absent -> `not run: loco null not run` |
| `gfp` | `combined` -> every family `GFP_ATTRIBUTED` | a `predictions.csv` fixture with 5 of 8 idle cells flagged -> `GFP_INSEPARABLE` and the `without` run present |
| `g1c` | the primary run -> `G1C_PRIMARY` | two directories, none primary (fixture) -> `G1C_SEARCH` |
| `harness` | (stage 2 fixture: cells of classes `benign_relaunched` and `harness_idle` written by `synth_detection` with `--stage2-fixture`, which adds one re-launched copy of gemm and 4 harness-idle cells at `floor_F = 300`) -> rows with three margins and a verdict per feature | stage 1 -> `HARNESS_STAGE2_ABSENT` |
| `exact_sign_test`, `gm` | `(5, 0)` -> 0.03125; `(7, 1)` -> 0.03516; rung A perfect, rung B chance -> `GM_BEATS` | `(3, 0)` -> 0.125 -> `GM_DIFFERENCE` |
| `alias_detection` | the corpus -> every feature `ALIAS_STAYS` (dt is constant) | a sidecar fixture with `dt_est_s` proportional to a feature -> `ALIAS_MOVES` |
| `gv_two_class` | the file's shape and the summary row | (a report; no refusal) |
| `miss_table` | the 4.4 row | member 6 -> `AT_FLOOR_NOT_A_MISS` |
| `synth_detection` | every member's `truth.json` reproduces the extract to 1e-9 (as `tests/test_extract.py`, reused on one member's cell) | `--members 9` -> `ValueError` |

Builder B's tests (`tests/test_report_detection.py`, `tests/test_run_detection.py`) use
`tests/detection_fixtures.py`, which writes `cells.csv` (after apply), `cell_classes.csv`,
`admissibility.csv`, `gk0_cells.csv`, `gk0_members.csv`, `gn.csv`, one `splits/<rung>/W8_H4/`
directory per rung and split with `predictions.csv`, `scores.json`, `null.json`, `roc.csv`,
`folds.json`, the level-2 and level-3 directories, `ladder.csv`, every gate CSV of 3.5 and the
sidecars and extracts of a 24-cell corpus, all by hand in the schemas above (with refusal strings
placed in chosen cells so that every "refusal printed as a string" rule of section 5 is
exercised); then every table and figure is written and checked (columns exact, every verdict
cell holds a vocabulary string or `--`, no number in a verdict cell, no blank cell), the
skeleton is checked structurally (brace balance, every `\input` and `\includegraphics` target
present, zero prose lines: no line outside a comment that ends in a period and contains a
space-separated word of more than three letters other than a LaTeX command), and the driver is
run with `--only-modules tables_detection,figures_detection,latex_skeleton_p3,driver`
and `--skip-missing-modules` through `plan`, `run`, `status`, a resume that skips, a stale
input that re-runs, and `--force`.

---

## 5. The tables and figures (builder B)

Every table is written as `report/detection/tables/<name>.csv`, `.md` and `.tex` through
`_report_common.write_table` (a `tabular` inside a `table` environment with an empty
`\caption{}` and `\label{tab:<name>}`, a `% columns:` comment line and a `% note:` comment
line that carries `grid_source`, the class counts and every standing label named below). A
verdict cell prints the verdict string; an undefined number prints `--`; a missing input file
prints `not run: <file> missing`; a `NULL_INSIDE`, `NULL_NOT_ESTIMABLE`, `GL_LEVEL_ONLY`,
`ORDER_VOID`, `GF_VOID` or `GC_DISCONNECTED` verdict is printed in the row's verdict column
and, for `ORDER_VOID` (consequence void), `GF_VOID` and `GC_DISCONNECTED`, in every score cell
of the affected rung as well (the numbers stay in `scores.json`). A refusal is never a number
and never a blank. No prose anywhere; the note lines are labels.

Rung display names: `apf raw` (the level-inclusive ceiling), `apf`, `wapf`, `persist`,
`content`, `combined`, `combined (matched)`, `content channel 2'`, `comparator`. Rows for the
last two carry `RUNG2P_NOT_BUILT` and `COMPARATOR_ELSEWHERE` in every cell that would hold a
number (the preamble).

### 5.1 Table 4, the tiers with counts (`table4_tiers`)

From `cell_classes.csv` and `admissibility.csv`: `tier (class), n workloads, n cells, n
admissible, n at floor, n at harness floor, family key rule, letter`. One row per class
present, in `CLASSES` order, plus a `total` row. Stage-2 classes absent print a row with `0`
cells and `not run: stage 2 absent` in the admissible column.

### 5.2 The result tables

5.2.1 Table 5, the gates in one row each (`table5_gates`): `gate, what it checks, threshold,
verdict (roll-up), refusal string, file`. One row per gate of this layer (admissibility, G-K0
two-class, G-N, G-L (i), G-OP, G-LM, G-ANCHOR (i), G-ANCHOR (ii), early-against-late idle, the
order test, the drift regression, G-SIG, G-FP, G-1C, the harness clause, G-CAL, G-M, G-DIM,
B1-G1 two-class, B1-G3 two-class, the alias falsifier, G-V two-class) and one per carried-over
gate that ran inside `<out>` (C1 to C8, the `failed/` count, G-C, G-P, G-K0 kernels, G-F, G-J,
G-X), the roll-up being the worst verdict across rungs (a refusal beats a pass) with the count
of rungs in parentheses as text. `what it checks` and `threshold` are the short strings of this
document's docstrings (labels, not prose).

5.2.2 Table 6, validity (`table6_validity`): rows `G-C (calibrated core), idle floor, harness
floor, idle set against idle set (G-ANCHOR ii), early against late idle, drift regression,
state-change yield, G-K0 counts per tier, G-F (i) per rung`; columns `row, rung, quantity,
value, null p95, rank, verdict, note`. `harness floor` and `state-change yield` print
`HARNESS_STAGE2_ABSENT` and `YIELD_NOT_RECORDED` in stage 1. The G-K0 count row prints, per
tier, `n at floor / n at harness floor / n above floor` as text.

5.2.3 Table 7, detection (`table7_detection`), LOWO, one row per rung display name in the
order above: `rung, axis, resolution (W x H), grid source, feature count, G-DIM, ROC area,
AUC null p95, AUC rank, TPR at 5% (in-fold), realized FPR, TPR at 5% (pooled, post hoc),
G-CAL, TPR at 1% (resolution limit), realized FPR at 1%, TPR null p95, TPR rank, null
verdict, n assignments, random scorer, G-OP, G-L (i), G-C, G-F (i), B1-G3 quarantine, n
sandbox cells scored, n at floor`. The majority baseline prints once in the note line as
`majority (always benign): accuracy a = n_b / (n_b + n_s)` (CR3 2.3). `TPR at 5% (pooled, post
hoc)` carries the label `POST_HOC_LABEL` in the column header comment. `random scorer` prints
`AUC 0.5, TPR = FPR`. `G-N` for the sandbox class prints in the note line. `external` cells,
when present, get a second block of rows `<rung> (external)` with the `lowo/final` fold's
numbers.

5.2.4 Table 8, per-member recall in eighths (`table8_member_recall`): one row per (rung,
split) for `lowo`, `loco`, `one_class`; columns `rung, split, member 1, ..., member S, min,
median, max, n at floor`, each member cell the string `k/n` from `per_member.eighths`
(`n` = 8 minus the member's at-floor cells; `at floor (8)` when every cell is at floor). A
second header row `sub-family` with the letters is written as the `% note:` line and as the
first data row of the CSV (`rung = "sub-family"`). No mean column (ML 2.4; P3 D5 "never a mean
alone"). With `external` present: the members of the external block as `external m`.

5.2.5 Table 9, the three pitfalls as sizes (`table9_pitfalls`): columns `pitfall, instrument,
rung, class or member, quantity, size, null p95, rank, verdict, note`. Rows: `level` x
`G-L (i)` per rung (`apf raw` TPR at 5% against `apf` TPR at 5%; the `apf` verdict); `level` x
`G-LM` per (rung, member) (the level, the band, `recall_unrestricted`, `recall_lm`, the
verdict, `level_label`); `campaign` x `G-ANCHOR (i)` per rung; `campaign` x `G-ANCHOR (ii)` per
rung; `order` x `order test` per (rung, class); `order` x `drift regression` per class;
`harness` x `harness clause` per rung (stage 1: `HARNESS_STAGE2_ABSENT`); `campaign` x
`cross-campaign row` (`CROSS_CAMPAIGN_STAGE1`). Every `size` is a number as text or a refusal
string; the table never prints a pass mark without its size.

5.2.6 Table 10, the miss table (`table10_misses` and `table10_false_positives`): the columns of
`miss_table.csv` and `fp_table.csv` in that order with `physical reason (M1 to M6)` appended as
an empty text column for the author, for the rung given by `--table10-rung combined` (every
rung's file is written as `table10_misses_<rung>`; the unsuffixed one is the chosen rung). The
note line reads "assignments with counts, never a confusion matrix".

5.2.7 Table 11, the four splits side by side (`table11_splits`): one row per rung; columns
`rung, LOWO TPR 5% (FPR), LOWO AUC, LOCO TPR 5% (FPR), LOCO AUC, G-SIG gap (TPR), G-SIG gap
(AUC), G-SIG, LOFO FPR per family, one-class TPR 5% (FPR), one-class AUC, one-class model, G-1C,
G-FP (families flagged at or above half), benign recall per family (LOWO), note`. `LOFO FPR per
family` and `benign recall per family` are text lists `kernels 0.02; idle 0.00`. The note
prints `separable when heard, not flagged when not` when the one-class `tpr05` null verdict is
`NULL_INSIDE` while LOWO's is PASS (K3 F9; the one-class run carries its own null when
`--null-perm` is set for it, else the note reads `one-class null not run`), and the LOCO column
header carries `signature ceiling`.

5.2.8 Level 2 (`table_level2_<rung>`): rows `true sub-family`; columns `true sub-family, n
members, n cells, predicted A, predicted B, predicted C, ..., recall, status, null p95, rank,
verdict`; one table per rung plus `table_level2` for `--table10-rung`'s rung. The B row's
counts print `--` and its status the `no held-out test` string.

5.2.9 Level 3 (`table_level3_<rung>`): rows `true member 1..S`; columns the predicted members,
`recall (eighths)`, `label` (`SIGNATURE_CEILING`), then a summary row `accuracy` with the null.

5.2.10 The ladder (`table_ladder`): `rung, reading, prefix (s), dt rule, pairs (median over
cells), pairs at 0.644 s, pairs at 0.500 s, n windows (median), n cells with a window, TPR at
5%, realized FPR, ROC area, null verdict, note`.

5.2.11 G-V two-class (`table_gv_two_class`): the columns of `gv_two_class.csv` sorted by rung
then `L3_over_L3_families` descending, with the summary row per rung.

5.2.12 The per-cell appendix (`table_cells_detection`): `cell_id, class, member, sub-family,
rep, order token, campaign, n pairs, dt (s), K median, G-K0 verdict, admissible, LOWO score
per rung (five columns), flagged at 5% per rung (five columns)`. `order token` from
`letter_sequence.csv` (the `not run` line when absent). This table never prints `path`,
`label`, `test_label` or `order_index` itself.

5.2.13 The manifest (`report/detection/manifest.json`): every file under `report/detection/`
and `gates/detection/` with its sha256 and size, the package version, the detection ledger,
every `params` block collected, the class counts, `S`, `B`, `n_assignments`, `grid_source`.

### 5.3 Figures (`figures_detection.py`, `report/detection/figures/`)

Matplotlib only; PNG and PDF; `SKIPPED.txt` naming the missing module when matplotlib is
absent (exit 0). The eight members are drawn with the marker shapes
`("o", "s", "^", "v", "D", "P", "X", "*")` indexed by member (member 1 = `o`), one colour per
sub-family letter from the default cycle, and the legend reads `member m (A)`; no member is
named by anything else (K3 Sec. 4 preamble). Benign kernels keep plan11's per-kernel colours;
idle is grey; harness-idle, when present, is black hollow.

| file | content | source |
|---|---|---|
| `fig2_three_floors` | three panels (K, `l0` per changed page, J): the idle cells' pooled distributions as histograms with the five quantiles as vertical lines; the harness-idle distribution overlaid when present, else the panel title carries `HARNESS_STAGE2_ABSENT`; a fourth panel: the idle sets by campaign as overlaid K histograms when two exist, else the text `not applicable: one idle campaign` (N3 Sec. 1 validity block, Figure 2; K3 move 4) | extracts of the idle and harness-idle cells; `gk0.json` |
| `fig4_fused_plane_tiers` | per snapshot `(r_l0_q50_per, J)` with the G-J mask applied to kernels (`gates/gj_mask/` when present, else unmasked and labelled); one panel per benign tier present (`kernels`, and each stage-2 family), one panel per sub-family letter with members as marker shapes, and the idle cloud (grey) plus the harness-idle cloud (when present) drawn in every panel; boundary pairs hollow when `inputs/iteration_boundaries.csv` exists (K3 move 11; N3 Sec. 1 RQ2, Figure 4) | extracts, `cell_classes.csv`, `gj_mask` |
| `fig5_roc_lowo` | one ROC curve per rung display name from `roc.csv` under LOWO (norm; `apf raw` dashed), the random scorer's diagonal, the operating point marked at the in-fold reading (realized FPR, TPR at 5%) per rung (N3 Sec. 1 RQ1, Figure 5) | `roc.csv`, `scores.json` |
| `fig6_ladder` | TPR at 5% against prefix seconds (30 to 600, log x) per rung, solid for `from_pair1`, dashed for `from_boundary` (absent when `LADDER_FROM_PAIR1_ONLY`), with the realized FPR as a thin line in a second axis (N3 Sec. 1 RQ1, Figure 6; K3 move 17) | `ladder.csv` |
| `fig_level_map` | one histogram per tier of the level quantity per cell (`K_med` from `gk0_cells.csv`, log x), the idle band edge and the idle envelope edge as vertical lines, the members as marker shapes at their per-cell levels above the histogram (K3 move 5) | `gk0_cells.csv`, `gk0.json` |
| `fig_apf_per_tier` | APF(t) (`K / N`, log y) overlaid: one panel per benign tier (kernels: twelve sub-panels as plan11's `fig_apf_per_kernel`; idle), and one panel per sub-family with each member's eight reps in one colour and the member's marker at the line's end (K3 move 6) | extracts |
| `fig_level2_confusion` | the level-2 confusion counts as a heat map per rung (letters on both axes; the B row hatched with its `no held-out test` string) | `level2/confusion.csv` |

CLI: `figures_detection.py --out O [--only NAME,...] [--table10-rung combined]`.

### 5.4 The LNCS skeleton (`latex_skeleton_p3.py`) and `p3.bib`

`latex_skeleton_p3.py --out O [--standalone PATH] [--documentclass llncs|article]` writes
`report/detection/paper3_skeleton.tex`: `\documentclass[runningheads]{llncs}` (the RAID
2026 call names the LNCS template; `article` when `llncs.cls` is absent, a comment says which),
`booktabs`, `graphicx`, `amsmath`; `\title{}`, `\author{}`, `\institute{}` empty; the working
title candidates as three comment lines (`P3 Sec. 9` open item, the author's); the section
headings of `P3 Sec. 2` / `N3 Sec. 2` in order: 1 Introduction, 2 Background and prior work,
3 The channel and the instrument, 4 Dataset and capture design, 5 Evaluation protocol, 6
Validity of the campaign, 7 Results (subsections RQ1 to RQ5, RQ6 as a comment), 8 Discussion
and limitations, 9 Conclusion, Appendices (the gate chain, per-cell tables, the pilot's
numbers, the comparator mapping, hyperparameters, the anonymisation procedure). Under each
heading a comment block `% - ...` with the substance bullets of `P3 Sec. 2`'s "Carries" column
and, for sections 6 and 7, the box contents in both outcomes from `N3 Sec. 1` as `% box if it
holds:` and `% box if it fails:` bullets. Table shells: Table 1 (campaign identity) as a static
tabular whose cells are the stage-1 facts of `P3 0a` as labels (`commit fcc184e`, `1024 MiB`,
`500 ms`, `speed 2`, `retention combined`, `--duration 600`, `seed on every line`) and `--`
elsewhere; Table 2 (prior work by observer position) as a static tabular whose first column
holds the bib keys of the six rows named in `P3 Sec. 2` and `--` elsewhere; Table 3 (the
ladder) as plan11's Table 2 static tabular (`rung, axis`); then `\input{tables/table4_tiers.tex}`,
`\input{tables/table5_gates.tex}` (section 5), `\input{tables/table6_validity.tex}` (section 6),
`\input{tables/table7_detection.tex}`, `table8_member_recall`, `table_ladder` (RQ1),
`table10_misses`, `table10_false_positives`, `table_level2`, `table_level3` (RQ2),
`table9_pitfalls` (RQ3), `table11_splits` (RQ4), a comment-only shell for the comparator row
(RQ5), and the appendix inputs (`table_cells_detection`, `table_gv_two_class`, every
`table_level2_<rung>`, `table_level3_<rung>`). Figure placeholders with empty captions for
`fig2_three_floors` (section 6), `fig5_roc_lowo`, `fig6_ladder` (RQ1), `fig4_fused_plane_tiers`
(RQ2), and a comment line for Figure 7 (the harness rhythm, appendix, not built). Bibliography:
`\bibliographystyle{splncs04}` and `\bibliography{p3}`. No sentence outside comments; the
test of 4.5 checks it. Every `\input` and `\includegraphics` target must exist after a full
run; before it the file compiles only with the tables present, which the runbook says.

`apf_paper/p3.bib` (builder B, written once by hand, not by the script): a header comment
naming its rule; then, copied verbatim from `apf_paper/p2.bib` including Hunayn's attached
comment blocks, exactly these keys and no other: `law2010volatile`, `savoldi2010uncertainty`,
`clark2005livemigration`, `lindemann2018identification`, `oliveri2025inconsistencies`,
`hirano2022ransomware`, `hirano2022ransap`, `hirano2025ransmap`, `purnaye2022bishm`,
`purnaye2025dataverse`, `purnaye2026agent`, `khoury2026architecture`, `vomel2012correctness`,
`pagani2019temporal`, `nosek2018preregistration`, `hofman2023preregistration`,
`simmons2011falsepositive`, `gelman2013forking`, `asanovic2006landscape`. Then, as comment
lines only, `% NEEDED (Hunayn, not in p2.bib): <what the P3 design cites>` for: the RAID
negative-finding precedent (Ninan, `council/09_` line 44), the dataset-shift protocol (Qu,
`council/09_` line 54), the anonymous deposit precedent (Dunn and Ghosh, `council/09_` line
102), and the encoding paper's own entry (after its submission). Nothing is written from
memory; a key not in `p2.bib` is a comment, never an entry.

---

## 6. The driver and the runbook (builder B)

### 6.1 `run_detection.py`

`run_detection.py run --out O --root R --classes CSV (--selection-from PATH | --grid-default W8_H4)
[--moves 0-15] [--assume-failed-zero --assume-reason TEXT] [--c1-activity-min 0.02] [--c1-rule report|exclude]
[--campaign-label TEXT] [--n-jobs 1] [--null-perm 500] [--null-splits lowo] [--null-rungs apf,wapf,persist,content,combined]
[--ladder-null-perm 0] [--loco-mode cell|rep_index] [--one-class-model isolation_forest] [--level-quantity median_K|per_iteration_K_sum]
[--order-consequence size|void] [--table10-rung combined] [--seed-offset 0] [--force] [--dry-run]
[--skip-missing-modules] [--only-modules M,...] [--standalone-tex PATH]`;
`run_detection.py status --out O`; `run_detection.py plan --out O ... [--moves 0-15]`.

It builds a plan of `run_moves._cmd` dicts (module, sub, args, outputs, inputs, template,
internal) and calls `run_moves.run_plan(o, plan, max_move=15, ledger_name="driver_detection_state.json",
internal_steps={"features-at-selection": features_at_selection, "tripwire-check": tripwire_check})`,
so the ledger, the skip rule ("outputs exist
and every declared input still hashes as recorded"), the staleness reason, `--force`,
`--dry-run`, `--skip-missing-modules`, `--only-modules` and the author-input protection
(`kept: author input exists`) are the epoch-1 machinery unchanged. Every command is a subprocess
`python3 -m plan11_encoding_ladder.<module> <sub> ...` from the package's parent directory.
For a rung not in `--null-rungs` the split commands are given `--null-splits ""` and the
rung's null verdicts read `not run: null not requested for <rung>`. Builder B may import the
underscore-prefixed helpers of `run_moves.py` (`_cmd`, `_output_exists`, `_stale_reason`) as
well as its public functions.
`--classes` is copied to `<out>/inputs/classes.csv` at move D0 when absent there (never
overwritten once present; a differing `--classes` path with an existing file is
`refused: inputs/classes.csv exists; edit it or pass --force`). The declared inputs of every
step include `inputs/classes.csv`, `cells.csv`, `gates/detection/cell_classes.csv`,
`gates/detection/admissibility.csv`, `gates/selection.json` and `gates/detection/gk0_cells.csv`
where the step reads them, so a change to the class file, the selection or the floor makes every
later move stale on its own. The tripwire check (D15) writes `gates/detection/tripwire_check.json`:
every Table 7 and Table 11 row carries a G-F (i) verdict and the drift-clause verdicts
(G-ANCHOR (ii), early-against-late idle) of its rung; a row without them is listed and the
verdict is `refused: <n> rows without a tripwire verdict`.

### 6.2 The moves, in al-Kindi's order (K3 Sec. 4), for 96 + 8 + 64 cells

Every command is run from `VM_sampler/VM_Capture_QEMU/` as `python3 -m plan11_encoding_ladder.<module> ...`;
`<out>` is a fresh output root for the detection run (not the encoding paper's), `<root>` the
retention root the author passes, `<sel>` the encoding run's `gates/selection.json`.

| Move | K3 move | Commands | Writes | The author looks at |
|---|---|---|---|---|
| D0 | 1 (identity) | `extract index --root <root> --out <out>`; `classes validate --out <out> --classes <out>/inputs/classes.csv`; `classes apply --out <out> [--campaign-label TEXT]`; `classes inherit-selection --out <out> --from <sel>` (or `--default W8_H4`); `classes letter-sequence --out <out>` | `cells.csv` (rewritten), `cells.pre_classes.csv`, `gates/detection/classes_validation.json`, `cell_classes.*`, `letter_sequence.*`, `gates/selection.json` | `classes_validation.json` `status ok`; 96 kernel rows, 8 idle rows, 64 sandbox rows with `status = ok` and public ids; `S = 8`, `B = 13`, `n_assignments = 203490`; the letter sequence (or its `not run` line) |
| D1 | 1 | `extract all --cells-csv <out>/cells.csv --out <out> --jobs 4` | `extract/<cell_id>/*` for 168 cells | the sidecars: `n_pairs` about 890 to 945, `header_ncols = 66`, `status = ok`; the longest step (one to three minutes per cell) |
| D2 | 2 | `gates_precondition preconditions --out <out> --assume-failed-zero --assume-reason "AA A5: any failed job re-runs the whole cell" [--c1-activity-min F]`; the templates `gates_calibration pass-table`, `gates_precondition gk0-template`, `gates_precondition idle-admissibility-template`, `series head-drop-template`, `gates_detection gk0-sandbox-template`; the author edits `inputs/*`; `gates_calibration gp --out <out>`; `gates_detection admissibility --out <out> [--c1-rule report]` | `gates/preconditions.*`, `gates/gp.csv`, `inputs/*`, `gates/detection/admissibility.*` | `all_hard_pass` per cell; which sandbox cells fail C1 and enter through the report rule; the eight numbered source lines of `inputs/gk0_source_sandbox.csv` written by the author |
| D3 | 3 | `gates_calibration gc --out <out> --rung R` for `apf, persist, content, wapf, combined` | `gates/gc.csv` | one `pass` per rung on the kernels: "lead connected, calibrated on the benign side"; a `disconnected lead` voids every negative of that rung |
| D4 | 4, 5 (floors, level map) | `gates_precondition gk0 --out <out>`; `gates_detection gk0-cells --out <out>`; `gates_detection gn --out <out>`; the driver's internal step `features-at-selection <rung>` for the five rungs (it reads `gates/selection.json` at run time and calls `series.build_features` for raw and norm, norm only for combined, recording `grid_source`; by hand: `series features --out <out> --rung R --grid-id <gid> --both`); `gates_precondition gf --out <out> --all-rungs --n-perm 500`; `gates_detection anchor --out <out> --part idle_sets`; `gates_detection anchor --part idle_early_late`; `gates_detection drift --out <out>`; `figures_detection --out <out> --only fig2_three_floors,fig_level_map` | `gates/gk0.csv`, `gates/detection/gk0_cells.csv`, `gk0_members.csv`, `gn.csv`, `features/*`, `gates/gf.csv`, `ganchor.csv` (idle rows), `drift.csv`, two figures | which members have cells `at floor` (they leave the denominator); G-F (i) `inseparable at floor` on every rung; the idle sets row (`not applicable: one idle campaign` in stage 1); early-against-late idle; the level map: does the sandbox band overlap the benign band (R8) |
| D5 | 6 | `figures_detection --out <out> --only fig_apf_per_tier` | the figure | reps overlaying; the members' spikes beside the kernels' |
| D6 | 7, 8 | `gates_detection harness --out <out> --rung R` for the five rungs | `harness.csv` | every row `not run: stage 2 absent` until stage 2 |
| D7 | 10 (Table D6, breadth alone) | `detection_metrics splits --out <out> --rung apf --raw-and-norm --all-splits --null-perm 500 --null-splits lowo --n-jobs N`; `detection_metrics one-class --rung apf`; `gates_comparison gx --out <out> --rung apf --null-perm 500`; `gates_detection anchor --part kernels`; `gates_detection order --out <out>`; `gates_detection gop --rung apf`; `gates_detection glm --rung apf`; `gates_detection gl --rung apf`; `tables_detection --out <out> --only table9_pitfalls` | `gates/detection/splits/apf/*`, `gates/gx.csv`, `ganchor.csv`, `order.csv`, `gop.csv`, `glm.csv`, `gl.csv`, Table 9 (apf rows) | the raw row against the normalized row: how much is level; the order test's size; G-ANCHOR (i)'s size; G-LM per member with the label `median K, stage 1` |
| D8 | 11 | `gates_readings gj --out <out>`; `figures_detection --out <out> --only fig4_fused_plane_tiers` | `gates/gj.*`, the fused plane | do the members fall together; is where they fall empty of benign cells once idle is drawn |
| D9 | 12 | for `persist`: `detection_metrics splits --norm --all-splits ...`, `one-class`, `gates_comparison gx`, `gates_detection gop`, `glm`, `gl`; `detection_levels level2 --rung persist`; `detection_levels level3 --rung persist` | as D7 for `persist`, `splits/persist/*/level2`, `level3` | J against its null; G-P's label stands beside every persistence reading of the sandbox side (undeclared) |
| D10 | 13 | the same for `content` | | the amount axis; members 1 to 4 of the synthetic analogue |
| D11 | 14 | the same for `wapf` | | |
| D12 | 16 (the detection tables) | the same for `combined` plus `detection_metrics splits --rung combined --norm --split lowo --reduce-to-strongest`; `detection_levels level2/level3 --rung apf,wapf,content,combined`; `gates_detection gsig`, `gfp`, `g1c`, `gcal` per rung; `gates_detection gm`, `gdim`; `tables_detection --out <out> --only table7_detection,table8_member_recall,table11_splits,table_level2,table_level3` | `gsig.csv`, `gfp.csv`, `g1c.csv`, `gcal.csv`, `gm.csv`, `gdim.csv`, Tables 7, 8, 11 | the headline: TPR at 5% with its realized FPR under LOWO above the null; per-member eighths; the four splits side by side; G-SIG's gap; which family fires |
| D13 | 17 (time-to-hear) | `detection_metrics ladder --out <out> --rung R [--null-perm 0]` for the five rungs; `tables_detection --only table_ladder`; `figures_detection --only fig6_ladder` | `ladder/*`, `ladder.csv`, the table, Figure 6 | the shortest prefix at which a rung reaches its 600 s reading; `from_boundary` reads `from pair 1 only` in stage 1 |
| D14 | 18, 19, 20, 21 (miss table, alias, G-V, cross-campaign) and the report | `gates_detection miss-table --rung R` (five rungs); `gates_detection alias --rung R`; `gates_detection gv --rung R`; `tables_detection --out <out>` (all); `figures_detection --out <out>` (all); `latex_skeleton_p3 --out <out> [--standalone apf_paper/p3_skeleton.tex]`; `tables_detection --only manifest` | `miss_table.csv`, `fp_table.csv`, `alias.csv`, `gv_two_class.*`, `report/detection/*` | the miss table: what each missed member resembles and on which axis; the cross-campaign row `not applicable: stage 1`; `manifest.json` |
| D15 | 23 (the tripwire) | the driver's internal `tripwire-check` | `gates/detection/tripwire_check.json` | every table row carries its G-F (i) verdict and the drift clause |

`RUNBOOK_DETECTION.md` lists every command above with its flags, what it writes and what to
look at, in this order, plus: the prerequisites (as `RUNBOOK.md` section 0; nothing new to
install); the class file's format with the stage-1 example of 2.1 and the rule that the file
is the author's and no agent fills it; the reading of the letter sequence; the cost table
below; the smoke run; how stage 2 and stage 3 cells enter (add rows to `inputs/classes.csv`,
re-run D0 and D1 for the new cells, then the driver from D2; the staleness rule re-runs every
later move); the rule that gate result files are never edited by hand; the resume discipline
(a re-run of `classes apply` or a changed `inputs/classes.csv` makes every move from D2 stale;
a changed `gates/selection.json` makes every feature and split stale).

Cost, stated in the runbook before the author starts (forest fits at about 39,000 windows x 60
features and 300 trees take seconds each on one core; the counts are what the runbook prints):

| Step | Fits per rung | Note |
|---|---|---|
| LOWO headline | 21 (+ 21 x 4 for the seed spread of G-M on apf) | minutes |
| LOWO null at 500 permutations | 10,500 | the expensive step: hours per rung; parallel over permutations with `--n-jobs`; `--null-rungs` limits it |
| LOCO (`cell`) | 168 (its null 84,000, not run by default) | `--loco-mode rep_index` gives 8 folds and a null of 4,000 |
| LOFO | 2 | seconds |
| one-class | 14 (isolation forests) | seconds |
| level 2, level 3 | 7 and 8 (their nulls 280 x 7 exhaustive, 500 x 8) on 64 cells | minutes |
| G-LM (retrain) | about 6 per member, 48 per rung | minutes |
| the ladder | 5 prefixes x 2 readings x 21 | about an hour per rung without a null |
| G-FP recomputation | 21 per inseparable family | as needed |

The smoke run: `synth_detection corpus --root /tmp/p11det --write-classes --order-confound on
--campaign-labels round_robin` then `run_detection run --out /tmp/p11det/out --root /tmp/p11det
--classes /tmp/p11det/classes.csv --grid-default W8_H4 --null-perm 20 --n-jobs 2
--assume-failed-zero --assume-reason "smoke run" --c1-activity-min 0.001`; it finishes in
minutes to tens of minutes with every table present and every two-class null row reading
`not run: 20 permutations < 500` by design; the smoke run is not admissible for the paper.

---

## 7. For the author: every choice the definitions leave open, with the default specified

Each item is a parameter (a module constant or a CLI flag) whose value is written into the
result files' `params`; the default runs unless the author says otherwise before the data.

1. **`inputs/classes.csv` is the author's.** No agent fills it; the code never guesses a class
   from a name. The stage-1 file has the kernel row, the idle row and eight member rows (2.1),
   plus per-cell rows with `order_index` once the realized order is at hand (the sandbox cells in
   order by member, `P3 0a`). Without `order_index` the order test, the drift regression,
   early-against-late idle and the letter sequence read `not run: order_index missing`.
2. **`--campaign-label TEXT`** (`classes apply`): the sandbox capture's launch label becomes the
   campaign component of every sandbox `cell_id`; pass a neutral text if the label names
   anything (section 2.3).
3. **`kernel_family_rule = "tier"`**: the twelve kernels are one benign family `kernels` for
   LOFO, G-FP and the per-family recall (ML 1.6 item 4 names "kernels, idle, and each captured
   family"); `"archetype"` makes the four predicted archetypes the families. Under `tier`, LOFO
   on stage 1 has two folds (kernels, idle) and holding out the kernels trains on idle and the
   sandbox only, which is the definition's reading of "does an unseen benign family fire".
4. **`relaunched_grouping = "parent"`** (CR3 2.31, proposed by al-Nadim and not yet adopted by
   another seat): a stage-2 re-launched control shares its parent kernel's workload key under
   LOWO and the null; `"own"` gives it its own key. Decide before stage 2.
5. **The (W, H) per rung is inherited** from the encoding paper's `gates/selection.json`
   (`inherit-selection --from`); with none, `--default W8_H4` and `grid_source` says so in
   every table note. Re-gridding on the detection data is allowed only under the grid rule
   (SPEC 3.5.7: declare, compute every point, keep every point, mark the selected one) and is
   not scheduled by this driver.
6. **The forest**: `n_estimators = 300`, `max_features = "sqrt"`, `min_samples_leaf = 1`,
   `class_weight = None`, median imputation and per-fold standardization (SPEC 4.2), out-of-bag
   scoring on; ML question 12 asks the author to fix `n_estimators` and the minimum leaf size
   before the data; these are the values unless changed now.
7. **`det_c1_rule = "report"`**: a C1 fail on a cell of any class but `benign_kernel` is reported
   and the cell stays admissible, its floor verdict coming from G-K0 two-class (K3 F3); the
   kernel cells keep the encoding paper's C1 rule and its `--c1-activity-min`, whose 0.02
   default refuses several real kernels (E1 6.7); decide the threshold before the data as for
   the encoding paper and pass it to both drivers.
8. **`score_aggregation = "window_mean_proba"`**: the cell's score is the mean over its windows
   of the forest's P(sandbox); alternatives `window_median_proba`, `vote_fraction`. The
   definition (ML 1.6) says "each cell's out-of-fold sandbox probability" and does not say how
   windows become a cell.
9. **`threshold_source = "oob"`** (ML 1.6 item 2, literally: the 95th percentile of the benign
   training cells' out-of-bag scores). The out-of-bag score of a training cell is optimistic
   because its sibling windows are in-bag for the same trees, so the realized out-of-fold FPR
   may exceed 5 percent; that is why the realized FPR is reported beside the TPR. The alternative
   `inner_lowo` sets the threshold on inner leave-one-benign-workload-out scores inside the
   training fold at B extra fits per fold. `threshold_quantile_method = "linear"` (numpy's
   default) is the percentile's interpolation; `"higher"` is the conservative alternative.
10. **`NULL_VERDICT_STATISTIC = "tpr05"`**: the headline's null verdict is on the TPR at the
    declared FPR (`P3 D5`: "the true-positive rate ... above the workload-level shuffle null");
    ML 1.6 item 1 names AUC as the statistic the null is computed on because it is not
    quantized; both nulls are computed from the same permutations and both verdicts are
    printed; the author chooses which carries the verb.
11. **`--null-splits lowo`** by default; LOCO's null at 168 folds x 500 permutations is 84,000
    fits per rung and is the author's decision; without it G-SIG reads `not run: loco null
    not run` and cannot refuse. `--loco-mode rep_index` (8 folds) is the cheap reading of the
    "sibling reps train" split (ML 1.3 defines LOCO as one fold per cell).
12. **`ONE_CLASS_MODEL = "isolation_forest"`** with `n_estimators = 300`, `contamination =
    "auto"`, as G-1C's one primary model declared before the data (CR3 2.19; ML 1.6 says "a
    single generative or density model"; an isolation forest is neither, a GMM is; `--model gmm`
    with `n_components = 4` is the density alternative and must be made primary now if
    preferred). Any other model runs only as `--secondary` and is labelled not citable.
13. **`ONE_CLASS_THRESHOLD_SOURCE = "inner_lowo"`**: the one-class threshold is the (1 - fpr)
    quantile of the benign cells' out-of-fold scores under leave-one-benign-workload-out (the
    honest analogue of out-of-bag; the definition gives no rule); `train_in_sample` is the
    alternative.
14. **`GK0_ENVELOPE_PERCENTILE = 95`** and **`GK0_VERDICT_QUANTITIES = (K_med, K_q90,
    frac_above_band)`**: CR3 2.8 leaves the envelope's quantile OPEN (paper 2 used the 95th
    of K) and says "every quantity"; the default gates on the three K quantities and discloses
    `l0` and J, which are undefined for a cell with no pair above the band. The idle band edge
    is plan11's (the 95th percentile of K pooled over idle rows).
15. **The sandbox source part** of G-K0 (`inputs/gk0_source_sandbox.csv`) is written by the
    author, one numbered line per member; agents write nothing about the members.
16. **`LEVEL_QUANTITY = "median_K"`** in stage 1 with the label `median K, stage 1` on every
    G-LM row (the per-iteration changed-page count needs `[SUSTAIN]` markers, absent in stage 1;
    K3 F1; VME section 4 item 7); `GLM_BAND_FACTOR = 2.0` (ML question 13); `GLM_MODEL =
    "retrain"` (the operating point recomputed by re-fitting on the other members plus the
    band's benign, with the in-fold threshold on the band benign's out-of-bag scores; the
    alternative `headline_oof` sets the threshold post hoc on the headline's held-out scores);
    `GLM_VANISH_RULE = "tpr_le_fpr"` (detection vanishes when the level-matched recall is at or
    below the level-matched realized FPR, the random scorer's line). Note for the author: with
    discrete benign levels a member at a benign-empty level keeps its detection under a
    factor-two band, so G-LM refuses only when band benign overlap the member in the lead;
    that is R8's precondition (a benign occupant at every sandbox level), which stage 2 must
    supply; the synthetic corpus therefore exercises the refusal through the verdict function
    and the empty-band and not-detected outcomes through the corpus (4.5).
17. **`LEVEL3_SPLIT = "rep_index"`**: level 3 holds out one rep index of every member (8 folds);
    `cell` holds out one cell (64). `LEVEL2_MIN_MEMBERS_HEADLINE = 3`, `LEVEL2_MIN_MEMBERS_TEST
    = 2` (G-N's counts).
18. **`GOP_CELLS = "threshold_setters"`**: "the cells behind the declared rate" are the benign
    training cells at or above each fold's threshold (the ones that set it), pooled over folds;
    `realized_fps` counts the out-of-fold false positives instead; both counts are written.
19. **`ORDER_TEST_CONSEQUENCE = "size"`** for stage 1 (the sandbox cells ran in order by member,
    so position and identity coincide by construction and the row is a size, `P3 0a`); `"void"`
    for an interleaved campaign (K3 F5 voids the detection table when the leak is inside it).
    The order label is the half by `order_index` rank within the class; the null unit is the
    workload when every workload lies inside one half, else the cell (recorded).
20. **`LADDER_DT_S = 0.644`**: prefix lengths in pairs from a fixed spacing, the derived guest
    spacing of about 0.644 s (47, 93, 186, 466, 932 pairs), the same pair count for every cell;
    `--dt 0.500` uses the configured interval (60, 120, 240, 600, 1200 pairs, the last two
    capped at the cell's length) and `--dt per_cell` the cell's own `600 / n_pairs`; the three
    pair counts are recorded per cell. `LADDER_NORM = "prefix"` (a prefix is normalized by its
    own median K, what a detector at 30 s would know); `--ladder-null-perm 0` (the ladder's
    rungs carry no null unless asked; N3's "the shortest prefix at which the rung clears the
    null" needs 500 per prefix).
21. **The drop rule for iteration 1** (CR3 2.29) is applied through `inputs/iteration_boundaries.csv`
    when it exists; in stage 1 the `from_boundary` reading reads `from pair 1 only`.
22. **`HARNESS_COMPARABLE_TOL = 0.10`** on margins in [0, 1] (`2 |AUC - 0.5|`): CR3 2.21 says
    "comparable" has no numeric rule; this is the value to confirm or change before stage 2.
    `HARNESS_RELAUNCH_RULE = "ge_median_member_recall"`.
23. **`GFP_FLAG_FRACTION = 0.5`** (CR3 2.15's "at least half its cells").
24. **G-M**: the exact one-sided binomial at `GM_ALPHA = 0.05` over non-tied units plus the
    margin rule `diff > spread` with `GM_N_SEEDS = 5` (SPEC 3.7.8's spread), so that "rung A
    beats rung B" needs both.
25. **`MISS_DISTANCE = "standardized_euclidean_to_centroid"`** in the plane (median
    `r_l0_q50_per`, median `J - J_null`) per cell; the alternative `nearest_cell`. The physical
    reason (M1 to M6) is the author's column.
26. **`ALIAS_TOP_K = 10`**, `ALIAS_R2 = 0.5` (SPEC 3.4.4's threshold); the iteration-count
    regressor waits for `inputs/iteration_counts.csv`.
27. **`B1G3_MAX_DISAGREE_WORKLOADS = 1`** (CR3 2.2: "all but at most one held-out workload").
28. **`--table10-rung combined`** chooses the rung of the unsuffixed Table 10 and of the level
    tables in the skeleton.
29. **`--documentclass llncs`** for the skeleton (`article` if `llncs.cls` is absent); `p3.bib`
    holds only entries copied from `p2.bib`; the `NEEDED` comments are Hunayn's list.
30. **Rung 2', the state-change yield, the harness cepstrum, the comparator, the mixture arm
    and the cross-campaign row** are not built in this epoch (the preamble); their rows carry
    the fixed strings, and the author decides after stage 2 whether the extract is extended
    for the content family.
31. **The seeds**: plan11's four (`20260916` to `20260919`), shifted together by `--seed-offset`;
    the null's permutations use `SEED_LABEL_NULL + offset`; the level-2 and level-3 nulls the same.
32. **The smoke run is never admissible**: 20 permutations print `not run: 20 permutations <
    500`; the tables show the numbers with that verb-less verdict.
