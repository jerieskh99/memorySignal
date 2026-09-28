# plan11_encoding_ladder: specification addendum for build epoch 2

Written 2026-09-17. Three builders implement this in parallel without talking to each other. It
extends the certified epoch-1 toolkit (`EPOCH1_BUILD_REPORT.md`; `CHECK_3.md` "NO BLOCKING
FINDINGS"; `CERTIFY_al_farabi.md`, all five conditions hold). Nothing in `SPEC.md` is withdrawn;
where this document and `SPEC.md` differ, this document wins for the files it names and `SPEC.md`
stands everywhere else. Every interface below is fixed. Where a definition leaves a choice, the
choice is a parameter with the default stated here and listed in section 7; a builder never picks a
different default and never adds an undeclared threshold.

Sources of truth for this epoch, in this order, with the short names used in docstrings:

1. `apf_paper/council/14_hunayn_exact_input_comparators.md`, cited as `C14 candidate N` (the
   comparators' definitions) and `C14 "For the author"` (Hunayn's notes).
2. `apf_paper/P2_AUTHOR_ANSWERS.md`, the block "Decisions of 2026-09-17 (for build epoch 2)",
   cited as `AD 2026-09-17` (the comparator picks, the delta_th sweep, the C1 re-map, the defaults).
3. `apf_paper/EPOCH1_BUILD_REPORT.md` section 4, cited as `E1 sec. 4 Mn` (the open minors M1 to
   M13) and section 6 items by number, cited as `E1 sec. 6 item n`.
4. `apf_paper/P2E_STRUCTURE.md`, cited as `P2E sec. n` (the EUSIPCO paper: the section plan, the
   tables, the compression map).
5. `SPEC.md` (the epoch-1 specification), cited as `SPEC n.n`, for every existing interface: the
   extract schema (2.2), the verdict vocabulary (3.0), the feature files (3.1.5), the split stage
   (4.5), the table inputs (6), the driver (7).

Binding rules for every builder, unchanged from `SPEC.md` and restated:

- The server is forbidden. No `ssh`, `scp`, `rsync`, no path under `/mnt/nfs` or `/project`, no
  remote command. No real data exists on this machine; every test uses `synth.py` (or the
  extract-level generator `tests/_synth_b2.py`, which follows `SPEC.md` section 5).
- The sandbox family is named only as "the sandbox family". It is not relevant to this epoch.
- No paper prose. The EUSIPCO skeleton is headings, equation placeholders, table and figure shells
  and comment blocks with substance bullets. Not one sentence of body text.
- No fabrication. Every comparator function and every changed gate cites, in its docstring, the
  definition it implements: `C14 candidate N` for the comparators, `AD 2026-09-17` for C1,
  `E1 sec. 4 Mn` for the fixes. A bibliography entry carries only the fields its source states.
- Plain register in reports; full sentences.
- Nothing is committed to git. No file outside `plan11_encoding_ladder/` and `apf_paper/` is
  edited, and inside `apf_paper/` only `p2e_skeleton.tex` (new) and `p2.bib` (append only).
  `P2_STRUCTURE.md` and `P2E_STRUCTURE.md` are not edited.
- Every existing test keeps passing (`python3 -m pytest -q tests`, 158 passed and 1 skipped at the
  end of epoch 1). A builder who needs an existing assertion changed has read this document wrong.
  In particular `tests/test_schema.py` pins `__version__ == "0.1.0"`, so the version string is not
  bumped; the epoch is recorded as `"epoch": 2` in the new result files' `params`.
- Python 3.10+, `numpy`, `scikit-learn` as already used; `joblib` is what scikit-learn ships and
  is already imported by `gates_precondition._gf_part1`; `pytest` is the runner.
- Numbers never move: every threshold is a module-level constant or a CLI parameter with the
  default written here, and the value used is written into every result file's `params` block.
- Refusals are strings from `verdicts.py`'s vocabulary written to the artifact, never a number,
  never a blank.

---

## 0. What this epoch adds, and what it keeps

Added: three published comparators computed on the same per-cell rows and run through the same
split stage as the rungs (section 2 and 3); the EUSIPCO Table 2 and Table 3, the five-page skeleton
and four bibliography entries (section 3.6 to 3.8); the C1 re-map of `AD 2026-09-17` (section 4);
the staleness refinement and the G-ORD parallelization (section 5); the `combined (matched)` rows
of Table 7 for every split, the matched feature count in Table 7, `SPEC.md` section 7 refreshed to
the driver as it runs, and `_schema_compat.py` deleted (section 1.3 and 3.5).

Kept, and not to be touched by anyone: al-Kindi's move numbers 0 to 13 (P2 Sec. 5; K2 Sec. 5;
`SPEC.md` section 7; `run_moves.parse_moves` accepts 0 to 13 and a test pins it). The comparators
therefore run as a named stage inside move 12, positioned after every rung's split stage
(`gx combined`) and before the comparisons and the tables (`gl (all rungs)`); section 3.4 fixes
the exact position. The five conditions of `CERTIFY_al_farabi.md` are invariants of this epoch:
every grid point computed and kept before any selection; no gate writes upstream of the extract;
every refusal written to the artifact; (W, H) chosen per encoding never per kernel; the level
normalization fixed per rung before the data. The comparators are computed at the whole-cell
point and select nothing, so they touch none of the five.

---

## 1. Builder assignment and file ownership

Three builders, no shared files. A file appears in exactly one list. Interfaces between builders are
files on disk with the schemas in this document, plus the epoch-1 functions named below, which are
called but not edited. Every module runs as `python3 -m plan11_encoding_ladder.<module>` from
`VM_sampler/VM_Capture_QEMU/` and by absolute path (each begins with the `sys.path` insertion of
its package's parent; a module inside the `comparators/` package inserts its parent's parent's
parent).

### 1.1 Builder 1, comparators (owns these files and nothing else)

| File | Owns |
|---|---|
| `comparators/__init__.py` | Empty except a docstring naming the three comparators and `C14`. |
| `comparators/__main__.py` | The package CLI (section 3.1): subcommands `law-pass`, `features`, `splits`, `status`. `python3 -m plan11_encoding_ladder.comparators <sub> ...`. |
| `comparators/adapter.py` | The registry constants (section 3.2), the feature-matrix adapter that writes `features/<comparator>/Wall_Hall_norm.npz` in exactly the shape of `SPEC 3.1.5` (section 3.2), the human-readable `features.csv` twin, `write_registry`, `load_registry`, and `run_comparator_splits`, which calls `models.run_split_stage` (section 3.3). |
| `comparators/savoldi.py` | Savoldi, Gubian, Echizen 2010: U per cell and per kernel (section 2.1). Runs standalone: `python3 -m plan11_encoding_ladder.comparators.savoldi --out O [...]`. |
| `comparators/dhodapkar_smith.py` | Dhodapkar and Smith 2003: the relative working set distance, the threshold sweep, phases, stability, mean phase length (section 2.2). Standalone as above. |
| `comparators/law.py` | Law et al. 2010: pages dynamic-for-X and static-for-X from the per-page run index (section 2.3), the `features` side. Standalone as above. |
| `comparators/page_runs.py` | The second streaming pass over the raw trajectory that builds the per-page run index Law needs (section 2.4). Reads the trajectory through `extract.open_text` (called, not copied). |
| `tests/test_comparators.py`, `tests/comparator_fixtures.py` | Builder 1's tests and their fixture helper (section 6.1). |

Builder 1 calls, and never edits: `series.py` (`load_cells`, `load_extract_cached`, `load_sidecar`,
`load_head_drop`, `head_drop_for`, `admissible_cells`, `write_csv`, `read_csv`, `write_json`,
`read_json`, `inputs_sha256`, `features_path`, `load_features`, `WHOLE_GRID_ID`, `PAIR_RUNGS`),
`models.py` (`run_split_stage`, `split_dir`, `N_ESTIMATORS`, `B1G1_MIN_PERM`), `splits.py`
(`SPLITS`), `nulls.py` (the seeds), `verdicts.py`, `schema.py`, `extract.py` (`open_text`).
`models.run_split_stage`'s signature is frozen for this epoch (builder 3 adds keys to its output,
never a parameter), so builder 1 codes against the signature as it stands in `SPEC 4.5` and in
`models.py` today.

### 1.2 Builder 2, eusipco (owns these files and nothing else)

| File | Owns |
|---|---|
| `tables_eusipco.py` | The EUSIPCO Table 2 and Table 3 generators, CSV, Markdown and LaTeX, reading `report/tables/table7.csv` and the artifacts named in section 3.6 and 3.7. CLI: `tables_eusipco.py --out O [--only table2,table3] [--comparators savoldi2010uncertainty,dhodapkar2003comparing] [--include-matched]`. |
| `latex_skeleton_eusipco.py` | The five-page IEEE conference skeleton (section 3.8), built by `build_p2e_skeleton()` and written by `write_p2e_skeleton()`; it imports `_comment_block`, `_shell`, `_figure`, `_generated`, `targets`, `prose_lines` from `latex_skeleton.py` and edits nothing there. CLI: `latex_skeleton_eusipco.py --out O [--documentclass IEEEtran\|article] [--standalone PATH]`. |
| `/Users/jeries/Desktop/projects/thesis/memorySignal/apf_paper/p2e_skeleton.tex` | The standalone skeleton, written once by the generator with `--standalone` at the end of the build (and regenerated whenever the generator changes). |
| `/Users/jeries/Desktop/projects/thesis/memorySignal/apf_paper/p2.bib` | Append only: the block of section 3.9. Nothing above the appended block is edited. |
| `RUNBOOK.md` | The whole runbook is builder 2's in this epoch: the new section for the comparators stage, the EUSIPCO outputs in move 12, and the paragraphs of Appendix C (C1, staleness, G-ORD, matched rows), pasted as written there. Builder 3 does not edit the runbook. |
| `tests/test_eusipco.py`, `tests/fixtures_eusipco/` | Builder 2's tests and their fixed fixture files (section 6.2). |

Builder 2 calls, and never edits: `_report_common.py` (`read_csv`, `write_csv`, `md_table`,
`tex_table` as the model for its own cite-aware writer, `fmt_num`, `cell_text`, `latex_escape`,
`not_run`, `not_applicable`, `to_float`, `read_json`, `write_json`, `result_json`, `now_iso`,
`inputs_sha256`, `RUNG_DISPLAY`, `SPLIT_DISPLAY`, `LEVEL_MATCHED_SETS`, `KERNELS`,
`PACKAGE_VERSION`, `effective_scores`, `load_scores`, `split_dir`), `gates_calibration.py`
(`separating_features`), `series.py` (`features_path`, `load_features`), `latex_skeleton.py`
(the helpers above).

### 1.3 Builder 3, fixes (owns these files and nothing else)

| File | Owns |
|---|---|
| `gates_precondition.py` | The C1 re-map (section 4): `gate_preconditions` and its CLI. |
| `gates_temporal.py` | G-ORD parallelized over its two loops with `--n-jobs` honoured (section 5.2). |
| `run_moves.py` (and `driver.py`, unchanged alias) | The C1 flags, the keyed staleness rule (section 5.1), the comparators stage and the EUSIPCO steps in the move table (section 3.4), `_module_path` resolving a package to its `__main__.py`. |
| `models.py` | `feature_count_used` in `scores.json` (section 3.5.2). Additive only; the signature of `run_split_stage` does not change. |
| `gates_comparison.py` | G-DIM's matched run for every split and label space, `--null-splits` on `gdim`; comparator rows in `gdim.csv` and `gm.csv` from the registry (section 3.5). |
| `tables.py` | Table 7: the matched rows of every split, the matched feature count, the comparator rows (section 3.5). |
| `_report_common.py` | `load_comparator_registry(out)` (section 3.2), nothing else. |
| `series.py` | Two edits only: `PAIR_RUNGS` gains the two pair-adjacency comparators (section 3.3), and the `try/except ImportError` around `from plan11_encoding_ladder import schema` becomes the plain import. |
| `_schema_compat.py` | Deleted (`E1 sec. 4`, "Named but not implemented": "an inert fallback ... that can be deleted"). Nothing else references it once the `series.py` fallback is gone (`BUILD_*.md` are historical reports and are not edited). |
| `SPEC.md` | Section 1's file table (the new modules, the deletion), section 7's move table replaced by Appendix A, section 7.1's CLI contract extended by Appendix B, section 8 extended by the items of section 7 of this document as items 41 onward. Nothing else in `SPEC.md` is edited. |
| `tests/test_gates_precondition.py`, `tests/test_gates_temporal.py`, `tests/test_driver.py`, `tests/test_report.py`, `tests/report_fixtures.py`, `tests/test_gates_comparison.py` | Builder 3 appends tests (section 6.3) and extends the fixture; no existing assertion is changed. |

Builder 3 reads builder 1's registry and split files by the layout in section 3.2 and 3.3, and
never imports from `comparators/`. Builder 3's own tests build those files with its fixture.

---

## 2. Exact definitions, in the toolkit's own column names

Notation. A cell's extract (`SPEC 2.2`) has one row per `seq` from `seq_first` to `seq_last`,
`n_pairs = seq_last - seq_first + 1` rows. Row `t` (0-based row index `i`) carries `K` (the
changed-page count of pair `t`, never blank) and `J` (the Jaccard between `S_t` and `S_{t+1}`,
blank on the last row and when both sets are empty). Head drop `h = series.head_drop_for(head_drop,
kernel)` from `inputs/head_drop.csv` (idle cells use the key `idle`, as `gate_gf` does). Every
comparator records `head_drop` and `n_pairs_used` per cell. `N = schema.N_PAGES = 262144`.

Every comparator writes, per cell, a `status` that is `ok` or a refusal string, and never a row of
numbers under a refusal. The refusal for too few pairs is exactly
`verdicts.refused(f"too few pairs ({n} < {min_pairs}) for {comparator}")`, for example
`refused: too few pairs (1 < 2) for savoldi`.

### 2.1 Savoldi, Gubian, Echizen 2010 (`C14 candidate 1`), module `comparators/savoldi.py`

Definition implemented (docstring citation: `C14 candidate 1: "a series of S snapshots; for each
consecutive pair the number of differing 4 KiB pages; the sample mean (mu_dmp) and standard
deviation (sigma_dmp) of that count; U = mu_dmp +/- sigma_dmp"; AD 2026-09-17 "Comparators"`).

Constants: `SAVOLDI_DDOF = 1` (the paper says "sample"), `SAVOLDI_SPAN = "whole"` (alternative
`"tail80"`), `SAVOLDI_TAIL_FRACTION = 0.80` (matches `gates_precondition.GK0_TAIL_FRACTION` and its
`tail_median_K` slice: `start = floor(n * (1 - 0.80))`), `SAVOLDI_MIN_PAIRS = 2` (a sample
standard deviation needs two values), `SAVOLDI_UNITS = "pages"`.

Per cell, on `K[h:]` (all remaining rows, the last `seq` included, the same rows as
`series.k_median_cell`):

- `K_mean = mean(K[h:])`, `K_sd = std(K[h:], ddof=SAVOLDI_DDOF)`; `n_pairs_used = n_pairs - h`;
  refuse when `n_pairs_used < SAVOLDI_MIN_PAIRS`.
- `K_tail80_mean`, `K_tail80_sd`: the same two numbers on `K[h:][start:]` with
  `start = floor(n_pairs_used * (1 - SAVOLDI_TAIL_FRACTION))` (G-K0's tail rule applied after the
  head drop; blank when fewer than two rows remain).
- `K_mean_pct_N = 100 * K_mean / N`, `K_sd_pct_N = 100 * K_sd / N` (Savoldi's own units,
  "U = 65.5% +/- 0.15%"; readings only, never features).

The feature row (d = 2, `SPEC 3.1.5` naming `<rung>.<channel>.<feat>`): `savoldi.K.mean`,
`savoldi.K.sd`, from the whole span (`SAVOLDI_SPAN = "whole"`; `"tail80"` puts the tail pair in the
row instead and records it). The feature is named `sd`, not `std`, so that `feature_drop`'s
last-component match (`models.prepare_split_data`) never drops it by accident.

Per kernel (`gates/comparators/savoldi/per_kernel.csv`): the mean over the kernel's admissible cells
of each per-cell value, plus `U_text = f"{K_mean_pct_N:.3f}% +/- {K_sd_pct_N:.3f}%"` built from the
kernel means; one row per kernel in `schema.KERNELS` order, then `idle` when idle cells exist.

### 2.2 Dhodapkar and Smith 2003 (`C14 candidate 3`), module `comparators/dhodapkar_smith.py`

Definition implemented (docstring citation: `C14 candidate 3: delta_{i,i-1} = (|W_i u W_{i-1}| -
|W_i n W_{i-1}|) / |W_i u W_{i-1}| between consecutive working sets; a phase change when delta
exceeds a threshold; stability, average phase length; "EXACT with W_i = our changed-page set of
pair i. Their delta is one minus Jaccard, identically"; AD 2026-09-17 "Comparators": the sweep with
0.04 marked as the default`).

Constants: `DELTA_TH_DEFAULT = 0.04`, `DELTA_TH_SWEEP = (0.02, 0.04, 0.08, 0.16)`,
`MIN_PHASE_LEN = 1`, `DS_MIN_PAIRS = 3` (two deltas), `DS_FEATURES = "phase"` (alternative
`"phase_and_delta_mean"`), `DELTA_TH_SOURCE = "AD 2026-09-17: 0.04, the ROC knee of Dhodapkar and
Smith 2003; a declared sweep, every point kept"`.

Mapping, exactly: `W_i` is the changed-page set `S_i` of pair `i`. The extract's `J` at row `t` is
the Jaccard of `(S_t, S_{t+1})`, so `delta_{t+1,t} = 1 - J[t]`. The deltas of a cell are
`delta = 1 - J[h : n_pairs - 1]` (the last row's blank `J` is not a delta), `n_intervals = n_pairs -
1 - h` of them; refuse when `n_pairs - h < DS_MIN_PAIRS`. A blank `J` inside the range (both sets
empty; `SPEC 2.2`) is an undefined delta: counted in `n_undefined`, excluded from every count below
and treated as not stable (it ends a phase). `J = 0` (exactly one set empty) gives `delta = 1`, a
boundary. The hashed working-set-signature variant of Dhodapkar and Smith (the bit-vector signature
and its relative signature distance) is NOT built; the module docstring says so and the registry
records `"signature_variant": "not built"`.

For each `delta_th` in the sweep (the function `phases(delta, delta_th, *, min_phase_len)`):

- interval `t` (one per defined delta) is stable when `delta_t <= delta_th`, a boundary when
  `delta_t > delta_th`; `n_stable`, `n_boundaries`;
- `stability = n_stable / (n_intervals - n_undefined)` (the fraction of intervals in stable
  phases); when the denominator is 0 the cell is `refused: no defined delta`;
- a phase is a maximal run of consecutive stable intervals with length `>= min_phase_len`;
  `n_phases` is their count; `mean_phase_len` is the mean of their lengths in intervals (pairs); with
  no phase both are 0;
- `delta_mean`, `delta_sd` (`ddof = 1`) over the defined deltas, readings only unless
  `DS_FEATURES = "phase_and_delta_mean"`.

Worked example, binding for the test (section 6.1): `J = [1.0, 1.0, 0.9, 1.0, 0.5, 1.0]` gives
`delta = [0, 0, 0.1, 0, 0.5, 0]`; at `delta_th = 0.04`: stable flags `[T, T, F, T, F, T]`,
`n_stable = 4`, `n_boundaries = 2`, `stability = 4/6`, phases of lengths `[2, 1, 1]`, `n_phases =
3`, `mean_phase_len = 4/3`; at `0.16`: flags `[T, T, T, T, F, T]`, `stability = 5/6`, phases `[4,
1]`, `n_phases = 2`, `mean_phase_len = 2.5`; at `0.02` and `0.08` the same numbers as at `0.04`.
With `J[1]` blank: `delta = [0, NaN, 0.1, 0, 0.5, 0]`, `n_undefined = 1`, at `0.04`: `n_stable =
3`, `n_boundaries = 2`, `stability = 3/5`, phases `[1, 1, 1]`, `n_phases = 3`, `mean_phase_len =
1.0`.

The feature row (d = 3) at `DELTA_TH_DEFAULT`: `dhodapkar_smith.delta.stability`,
`dhodapkar_smith.delta.mean_phase_len`, `dhodapkar_smith.delta.n_phases`. The full sweep is kept in
the sidecar `gates/comparators/dhodapkar_smith/sweep.csv` (every cell, every `delta_th`) and the
per-kernel means per sweep point in `per_kernel.csv` with an `is_default` column; `--delta-th`
picks another sweep point for the feature row and the value is recorded in `params` and in the
registry (al-Farabi's grid condition: every point computed and kept, the marked one used).

### 2.3 Law et al. 2010 (`C14 candidate 2`), module `comparators/law.py`

Definition implemented (docstring citation: `C14 candidate 2: "a page is dynamic in X consecutive
dumps if X or more consecutive hashes differ; static if identical; an index of run lengths answers
all X in one pass"; "Dynamic in X = a membership run of length X-1 in our changed sets; static in
X = a non-membership run of length X-1"; AD 2026-09-17 "Comparators": built as a third module for
the IFIP version, X in {2, 3, 5, 10}, all kept`).

Constants: `X_SET = (2, 3, 5, 10)`, `LAW_MIN_PAIRS = 10` (`max(X_SET)`; a membership run of `X - 1
= 9` pairs is observable only from 9 changed sets, and one more is required so that a static run of
the same length can coexist), `LAW_DENOMINATOR = "ever_changed"` (alternative `"all_pages"`).

Mapping, exactly: a "dump" is a snapshot; consecutive dumps `(t, t+1)` form pair `t`, whose
changed set is `S_t`; a page `p` is a member of pair `t` iff `p in S_t`. Along a cell, page `p`'s
membership sequence over the pairs `t = seq_first + h .. seq_last` is a 0/1 string; its maximal runs
of 1 are membership runs, its maximal runs of 0 (leading, inner and trailing) are non-membership
runs. `dynamic-for-X(p)` iff the longest membership run of `p` is `>= X - 1`; `static-for-X(p)` iff
the longest non-membership run of `p` is `>= X - 1`. A page is ever-changed iff it is a member of
at least one pair in the span. Per cell and per `X`:

- `dyn_X = |{p ever-changed : dynamic-for-X(p)}| / n_ever_changed`,
- `stat_X = |{p ever-changed : static-for-X(p)}| / n_ever_changed`,

with `n_ever_changed` the denominator (`LAW_DENOMINATOR = "ever_changed"`; `"all_pages"` divides by
`N` and then the never-changed pages count as static for every `X`). `dyn_X2` is 1.0 for every cell
by construction (an ever-changed page has a run of length at least 1); it stays in the row because
the author declared the set, and the registry notes it as constant. Refuse when `n_pairs - h <
LAW_MIN_PAIRS`; refuse `no page ever changed` when `n_ever_changed = 0`.

The feature row (d = 8): `law.pages.dyn_X2`, `law.pages.dyn_X3`, `law.pages.dyn_X5`,
`law.pages.dyn_X10`, `law.pages.stat_X2`, `law.pages.stat_X3`, `law.pages.stat_X5`,
`law.pages.stat_X10`.

Per kernel: the means over admissible cells of the eight fractions and of `n_ever_changed`,
`frac_ever_changed = n_ever_changed / N`.

Worked example, binding for the test (section 6.1). A plain-text trajectory with the 66-column
header (`schema.TRAJ_HEADER_LINE`), `seq` 1 to 12, `hamming = l0 = l1 = 1` and every other metric
0 on each row, and these memberships: page 7 in seqs {1, 2, 3, 10, 11, 12}; page 9 in {1, 3, 5};
page 11 in {2, 3, 4, 5, 6}; page 13 in {6}; page 15 in {3, 4, ..., 12}. Every seq has at least one
row. Expected: `n_ever_changed = 5`; longest membership runs 7: 3, 9: 1, 11: 5, 13: 1, 15: 10;
longest non-membership runs 7: 6 (seqs 4 to 9), 9: 7 (seqs 6 to 12), 11: 6 (seqs 7 to 12), 13: 6
(seqs 7 to 12; its leading run is 5), 15: 2 (seqs 1 and 2); `dyn_X2 = 1.0, dyn_X3 = 0.6, dyn_X5 =
0.4, dyn_X10 = 0.2`; `stat_X2 = 1.0, stat_X3 = 1.0, stat_X5 = 0.8, stat_X10 = 0.0`;
`member_run_hist = {1: 4, 3: 2, 5: 1, 10: 1}`; `gap_run_hist = {1: 3, 2: 1, 5: 1, 6: 3, 7: 1}`
(leading and trailing runs included, zero-length runs not recorded). Variant with every row of
`seq = 8` removed (a gap, a K = 0 snapshot per `SPEC 2.1`): page 15's runs become 5 and 4, so its
longest membership run is 5, `dyn_X10 = 0.0`, `member_run_hist = {1: 4, 3: 2, 4: 1, 5: 2}`, and the
gap contributes a non-membership run of 1 to page 15.

### 2.4 The second streaming pass for Law, module `comparators/page_runs.py`

Why a second pass: the extractor holds at most two snapshots (`SPEC 2.4`) and writes no page set,
so per-page run lengths cannot be read from `extract.csv`. Law needs, per page, the longest
membership run and the longest non-membership run over the cell. `page_runs_pass` re-streams the
raw trajectory once, with the same row grouping as `extract._stream` (rows grouped by `seq`,
`seq` non-decreasing, a decrease is `refused: seq not monotone at row <n>`, duplicate `page_index`
inside a `seq` collapsed by `numpy.unique`, a missing `seq` an empty snapshot), reading only the
`seq` and `page_index` columns located by name in the header; it opens the file through
`extract.open_text` (zstd binary, `zstandard`, gzip, plain).

State, exactly (all `numpy.int32`, size `N`): `last_seq` (the last `seq` in which the page was a
member; `-1` never), `run_len` (the length of the membership run ending at `last_seq`), `max_mem`
(the longest membership run closed so far), `max_gap` (the longest non-membership run closed so
far). Two `numpy.int64` histograms of size `n_pairs_used + 1`: `member_run_hist` (every membership
run of every page) and `gap_run_hist` (every non-membership run of length `> 0` of every
ever-changed page, leading and trailing included). Scalars: `seq_first_used = seq_first + h`
(snapshots with `seq < seq_first_used` are read and discarded), `seq_last`, `n_pairs_used`.

Per finished snapshot `t >= seq_first_used` with unique sorted pages `p` (`int64`):

1. `cont = last_seq[p] == t - 1`; for `q = p[cont]`: `run_len[q] += 1`.
2. `new = ~cont`; among `q = p[new]`, for those with `last_seq[q] >= 0` (a page returning after a
   gap): close the old membership run (`max_mem[q] = max(max_mem[q], run_len[q])`,
   `member_run_hist[run_len[q]] += 1` via `numpy.add.at`) and the gap `g = t - last_seq[q] - 1`
   (`max_gap[q] = max(max_gap[q], g)`, `gap_run_hist[g] += 1`); for those with `last_seq[q] == -1`
   (first membership): the leading gap `g = t - seq_first_used` (`max_gap[q] = g`, and
   `gap_run_hist[g] += 1` only when `g > 0`). Then `run_len[p[new]] = 1`.
3. `last_seq[p] = t`.

At end of file, for every page with `last_seq >= 0`: close the last membership run as in step 2 and
the trailing gap `g = seq_last - last_seq` (recorded only when `g > 0`). Then `n_ever_changed =
count(last_seq >= 0)`, and for any `X`: `dyn_X = count(max_mem[ever] >= X - 1) / n_ever_changed`,
`stat_X = count(max_gap[ever] >= X - 1) / n_ever_changed`; the two `max_*` histograms
(`max_member_run_hist`, `max_gap_run_hist`, counts by run length over ever-changed pages) are also
written so that any `X` is answerable from the file without another pass.

Memory bound, stated in the module docstring and in the runbook: four `int32` arrays of `N` entries
(4 MiB at `N = 262144`) plus the row buffer of the snapshot being read (at most `K_t` page indices;
below 2 MiB for `K_t <= N`) plus two `int64` histograms of `n_pairs_used + 1` entries (about 15 KiB
at 931 pairs). Under 8 MiB regardless of the trajectory's length. Time: one pass in `csv.reader`,
the same order as the extractor (about 25 s per synthetic cell of 4 million rows on this machine,
one to three minutes per real cell); `law-pass --jobs N` runs cells in a `multiprocessing` pool as
`extract all --jobs` does, and skips a cell whose `page_runs/<cell_id>.json` exists with `status ==
"ok"` and the same `head_drop` unless `--force`.

Output per cell, `gates/comparators/law/page_runs/<cell_id>.json`, deterministic (no timestamp
inside; timing goes to `page_runs/timing.csv`):

```
{"schema": "plan11.law_page_runs.v1", "params": {"head_drop": h, "N": N, "x_set_declared": [2, 3, 5, 10],
  "seq_first_used": ..., "seq_last": ..., "min_pairs": 10, "epoch": 2, "traj_file": "<basename>", "traj_sha256_head": "<sha256 of the header line>"},
 "citation": "C14 candidate 2; AD 2026-09-17",
 "cell_id": "...", "status": "ok" | "refused: ...", "n_pairs_used": ..., "n_rows_in": ..., "n_rows_skipped": ..., "n_seq_gaps": ...,
 "n_ever_changed": ..., "frac_ever_changed": ...,
 "max_member_run_hist": [...], "max_gap_run_hist": [...], "member_run_hist": [...], "gap_run_hist": [...],
 "fractions": {"dyn": {"2": ..., "3": ..., "5": ..., "10": ...}, "stat": {"2": ..., ...}}}
```

and the index `gates/comparators/law/page_runs/index.csv` with columns `cell_id, status,
n_pairs_used, head_drop, sha256` (the sha256 of the cell's JSON file; deterministic content, so an
unchanged re-run leaves the index byte-identical), rewritten by every `law-pass` run over the
current `cells.csv`. `law.py`'s `features` side reads the JSONs through the index, refuses a cell
with `not run: law page-run pass missing for <cell_id>` when absent, and with
`not run: page_runs computed with head_drop <h1>, inputs/head_drop.csv says <h2>; re-run law-pass
--force` on a mismatch.

Extract columns read, per comparator, stated once: Savoldi reads `K` only. Dhodapkar-Smith reads `J`
only (the extract's `n_persist` and `n_union` are not needed: `1 - J` is their delta exactly). Law
reads no extract column; it reads the page sets from the raw trajectory in the pass above, and the
sidecar's `n_pairs`, `seq_first`, `seq_last` for the refusal check before streaming.

---

## 3. How the comparators enter the existing machinery

### 3.1 The package CLI (`comparators/__main__.py`)

```
comparators law-pass  --out O [--cells-csv CSV] [--jobs 1] [--force]
comparators features  --out O [--cells-csv CSV] [--only savoldi,dhodapkar_smith,law]
                      [--delta-th 0.04] [--delta-th-sweep 0.02,0.04,0.08,0.16] [--min-phase-len 1]
                      [--ds-features phase|phase_and_delta_mean]
                      [--x-set 2,3,5,10] [--law-denominator ever_changed|all_pages]
                      [--savoldi-span whole|tail80] [--savoldi-ddof 1] [--tail-fraction 0.8]
comparators splits    --out O (--comparator C | --all) [--null-perm 500]
                      [--null-splits loko,loro,within_trace] [--n-jobs 1] [--n-estimators 300]
                      [--seed-offset 0]
comparators status    --out O            (prints the registry; exit 2 when absent)
```

Exit codes as `SPEC 7.1`: 0 on success (a written refusal is a success), 2 when an input file is
missing (its path on stderr; `cells.csv`, an extract, the page-run index for `features` of `law`),
1 on an internal error. `features` refuses to write the `law` file when the page-run index is
absent (exit 2 with the index path) unless `--only` excludes `law`. `splits` checks
`set(PAIR_COMPARATORS) <= set(series.PAIR_RUNGS)` first and exits 1 with `internal error:
series.PAIR_RUNGS lacks the pair-adjacency comparators (SPEC_EPOCH2 3.3)` otherwise, so the
failed-count exclusion of section 3.3 can never be silently skipped on a real run. Each of
`savoldi.py`, `dhodapkar_smith.py`, `law.py` also runs standalone with `--out O` and its own
options above, writing only its own directory and its own feature file, then rewriting the registry
entry for itself.

### 3.2 The adapter and the registry (`comparators/adapter.py`)

Constants, fixed:

```python
COMPARATORS = ("savoldi", "dhodapkar_smith", "law")
PAIR_COMPARATORS = ("dhodapkar_smith", "law")          # read consecutive-set relations
LEAD_RUNG = {"savoldi": "apf", "dhodapkar_smith": "persist", "law": "persist"}   # the rung on the same axis; a label, never a verdict source
CITATION_KEY = {"savoldi": "savoldi2010uncertainty", "dhodapkar_smith": "dhodapkar2003comparing", "law": "law2010volatile"}
DISPLAY = {"savoldi": "Savoldi 2010", "dhodapkar_smith": "Dhodapkar-Smith 2003", "law": "Law 2010"}
COUNCIL_CANDIDATE = {"savoldi": 1, "dhodapkar_smith": 3, "law": 2}
IN_EUSIPCO_TABLE2 = {"savoldi": True, "dhodapkar_smith": True, "law": False}     # AD 2026-09-17
GRID_ID = series.WHOLE_GRID_ID                          # "Wall_Hall": one row per cell, the per-run statistic
REGISTRY = "gates/comparators/registry.json"
```

The feature file. `write_feature_file(out, comparator, rows, feature_names, *, head_drop) -> Path`
writes `features/<comparator>/Wall_Hall_norm.npz` with exactly the arrays and scalars of `SPEC
3.1.5` and `series.build_features`: `X` (`float64`, `[n_cells_ok, d]`), `feature_names` (`str`),
per row `cell_id`, `kernel`, `archetype` (the PREDICTED archetype from `cells.csv`; `IDLE`'s
stored value for idle rows is whatever `cells.csv` carries, as `build_features` does), `campaign`,
`role`, `rep` (`int64`), `win_start` (`int64`, all 0), `n_series_cell` (`int64`, the cell's
`n_pairs_used`), and scalars `W = -1`, `H = -1`, `grid_id = "Wall_Hall"`, `normalized = True`,
`head_drop_json` (the same JSON `build_features` writes), `n_windows_dropped = 0`, `wapf_norm =
"n/a"`. Rows: every cell with `status == ok` in `cells.csv` whose comparator status is `ok`, in
`cells.csv` order; a refused cell has no row (it becomes a `cells_no_windows` entry of the split
stage, whose `predictions.csv` then reads `y_pred = not run: no windows` for it; the comparator's
`features.csv` carries the real refusal string). Idle cells are included with their role; the split
stage masks them (`SPEC 4.1`). Admissibility is applied by the reader, never here (the artifact
does not depend on a gate's verdict; `CERTIFY_al_farabi.md` condition 2). "norm" in the file name
means "the file the split stage reads"; the comparators are written as published, with no level
normalization, and the registry records `"normalization": "none (published definition)"`; the
`_raw` variant is not written. `series.load_features` must load the file and `splits.make_labels`
must accept it; the test asserts both.

The human-readable twin, `gates/comparators/<comparator>/features.csv`, deterministic, every cell
with `status == ok` in `cells.csv` (refused cells included with their refusal), columns:

- savoldi: `cell_id, kernel, role, archetype_predicted, campaign, rep, status, head_drop,
  n_pairs_used, K_mean, K_sd, K_mean_pct_N, K_sd_pct_N, K_tail80_mean, K_tail80_sd`
- dhodapkar_smith: `cell_id, kernel, role, archetype_predicted, campaign, rep, status, head_drop,
  n_pairs_used, delta_th, n_intervals, n_undefined, n_stable, n_boundaries, stability, n_phases,
  mean_phase_len, delta_mean, delta_sd`
- law: `cell_id, kernel, role, archetype_predicted, campaign, rep, status, head_drop,
  n_pairs_used, n_ever_changed, frac_ever_changed, dyn_X2, dyn_X3, dyn_X5, dyn_X10, stat_X2,
  stat_X3, stat_X5, stat_X10`

Beside it: `per_kernel.csv` (section 2; over admissible cells: `series.admissible_cells(out,
cells, comparator)`, so that the pair-adjacency exclusion of 3.3 applies to the readings of the two
pair comparators), `params.json` (`schema = "plan11.comparator.<id>.v1"`, every constant and CLI
value, `epoch: 2`, `excluded_cells_hard`, `excluded_cells_pair_rungs`, `inputs_sha256` of
`cells.csv`, `inputs/head_drop.csv`, `gates/preconditions.csv`, and for law the page-run index),
and for dhodapkar_smith `sweep.csv` with columns `cell_id, kernel, role, delta_th, n_intervals,
n_undefined, n_stable, n_boundaries, stability, n_phases, mean_phase_len`; `per_kernel.csv` for
dhodapkar_smith has one row per `(kernel, delta_th)` with `is_default` true on the marked point.

The registry, `gates/comparators/registry.json`, written by `features` (and rewritten entry-wise
by a standalone module run):

```
{"schema": "plan11.comparators.registry.v1",
 "params": {"epoch": 2, "grid_id": "Wall_Hall", "cli": {...every features option...}, "written_at": "<iso>"},
 "citation": "C14 candidates 1, 2, 3; AD 2026-09-17 'Comparators'",
 "order": ["savoldi", "dhodapkar_smith", "law"],
 "comparators": {
   "savoldi": {"display": "Savoldi 2010", "citation_key": "savoldi2010uncertainty", "council_candidate": 1,
               "grid_id": "Wall_Hall", "feature_names": ["savoldi.K.mean", "savoldi.K.sd"], "feature_count": 2,
               "lead_rung": "apf", "pair_adjacency": false, "in_eusipco_table2": true,
               "normalization": "none (published definition)",
               "features_npz": "features/savoldi/Wall_Hall_norm.npz",
               "features_csv": "gates/comparators/savoldi/features.csv",
               "splits_dir": "gates/splits/savoldi/Wall_Hall",
               "n_cells_ok": <int>, "n_cells_refused": <int>, "refusals": {"<cell_id>": "<string>", ...},
               "params": {...the module's constants and values used...}},
   "dhodapkar_smith": {... "feature_count": 3, "lead_rung": "persist", "pair_adjacency": true, "in_eusipco_table2": true,
               "params": {"delta_th": 0.04, "delta_th_sweep": [0.02, 0.04, 0.08, 0.16], "min_phase_len": 1,
                          "signature_variant": "not built", "delta_th_source": "..."}},
   "law": {... "feature_count": 8, "lead_rung": "persist", "pair_adjacency": true, "in_eusipco_table2": false,
               "params": {"x_set": [2, 3, 5, 10], "denominator": "ever_changed", "min_pairs": 10, "constant_features": ["law.pages.dyn_X2"]}}}}
```

`_report_common.load_comparator_registry(out) -> dict` (builder 3) returns the `comparators` dict
in `order`, or `{}` when the file is absent; builder 3's tables and gates take "no comparator rows"
from `{}` and never fail on it.

### 3.3 The split stage, the null, the baseline, the gates

`run_comparator_splits(out, comparator, *, n_perm, null_splits, n_jobs, n_estimators, seed_offset)`
calls, for every `split` in `splits.SPLITS` and every admissible label space (`archetype` only for
`loko`; `kernel` and `archetype` for `loro` and `within_trace`, exactly `models.py main`'s rule
for `--all-splits --labelspace all`):

```python
models.run_split_stage(out, comparator, "Wall_Hall", split, labelspace, normalized=True,
                       n_perm=n_perm, n_jobs=n_jobs, n_estimators=n_estimators,
                       run_null=(split in null_splits), seed_offset=seed_offset)
```

and nothing else. Everything the author's decision names then comes from the epoch-1 code
unchanged: the same three splits at the unit of the cell, the same unit-level label shuffle
(`nulls.shuffle_labels_units`, 500 permutations at paper strength, `b1_g1 = not run: N
permutations < 500` below it), the same majority baseline (B1-G6 at the unit), B1-G3's quarantine
and `effective_scores`, G-N's headline classes in `macro_recall`, G-K0's relabelling at read time,
the admissibility of `preconditions.csv`, and `feature_count` and `dim_status` for G-DIM. Files land
under `gates/splits/<comparator>/Wall_Hall/<split>__<labelspace>/` with the four files of `SPEC
4.5`. Within-trace on a one-row-per-cell file is `not applicable: one window per cell` by the
split stage's own rule (`models.run_split_stage`); that is the correct reading of a per-run
statistic and is printed as such.

The failed-count exclusion. `series.admissible_cells(out, cells, rung)` drops a cell whose
`failed_verdict` is a refusal when `rung in series.PAIR_RUNGS` (`SPEC_review_al_farabi.md` 2.4:
the seq axis after a failed job is uncorrected, so every consecutive-set relation is unusable).
Dhodapkar-Smith and Law read exactly such relations; Savoldi reads `K` like `apf`. Builder 3
therefore sets `series.PAIR_RUNGS = ("persist", "content", "combined", "dhodapkar_smith", "law")`,
and the split stage applies the exclusion to the two pair comparators through
`prepare_split_data` with no other change. Builder 1's guard in 3.1 refuses to run splits until
that line exists; builder 1's test of the exclusion skips with the reason
`series.PAIR_RUNGS not yet extended (builder 3)` when it is absent and runs after integration.

G-DIM and G-M (builder 3, `gates_comparison.py`):

- `gate_gdim`: after the rung rows, one row per registered comparator with `rung = <comparator
  id>`, `grid_id = "Wall_Hall"`, `d = feature_count`, `d_matched = feature_count`, `method = ""`,
  `status = dim_status` from the comparator's `loko__archetype/scores.json` (or `not run:
  LOKO/archetype scores.json missing`), `loko_score`. Comparators never enter the choice of `d*`
  (the strongest single rung is a rung).
- `gate_gm`: the score dict per split includes every registered comparator whose
  `scores.json` at `Wall_Hall` has an accuracy, keyed by its id, ordered after the rungs in registry
  order; `itertools.permutations` then yields every ordered pair among rungs and comparators, so
  `gm.csv` carries `(comparator, apf)`, `(apf, comparator)` and `(comparator, <each rung>)` rows
  with the same margin `spread` (the APF seed spread) and the same sign rule. Within-trace
  comparator rows have no accuracy and are simply absent from that split's pairs.

G-N, G-C, G-F, G-L, G-X: G-N is a property of the label space and needs no comparator row. G-C is
per rung (an instrument check on a lead) and is not defined for a comparator; G-F part (i) needs
windows and a one-row cell has none; G-L is the level rule of the rungs; G-X is not run for the
comparators in this epoch (listed in section 7). Table 7 prints the strings of 3.5.3 in those
cells.

### 3.4 The driver: the comparators stage in move 12

Builder 3 adds to `run_moves.build_plan`, in move 12, immediately after `gx combined` and before
`gl (all rungs)`, these steps in this order (module `comparators`; the driver's `_module_path`
returns `_HERE / module / "__main__.py"` when `_HERE / module` is a directory, so the existing
absent-module logic applies unchanged):

| name | module, sub, args | outputs | inputs |
|---|---|---|---|
| `comparators: law pass` | `comparators law-pass --out O --jobs <n_jobs>` | `gates/comparators/law/page_runs/index.csv` | `cells.csv`, `inputs/head_drop.csv`, `gates/preconditions.json` |
| `comparators: features` | `comparators features --out O` | `gates/comparators/registry.json` | `cells.csv`, `inputs/head_drop.csv`, `gates/preconditions.json`, `gates/comparators/law/page_runs/index.csv` |
| `comparators: splits savoldi` | `comparators splits --out O --comparator savoldi --null-perm <null_perm> --null-splits <null_splits>` | `gates/splits/savoldi/Wall_Hall` | `cells.csv`, `inputs/head_drop.csv`, `gates/gk0.csv`, `gates/preconditions.json`, `gates/comparators/savoldi/features.csv` |
| `comparators: splits dhodapkar_smith` | the same with `dhodapkar_smith` | `gates/splits/dhodapkar_smith/Wall_Hall` | the same with its `features.csv` |
| `comparators: splits law` | the same with `law` | `gates/splits/law/Wall_Hall` | the same with its `features.csv` |

`("comparators", "splits")` joins `RANDOM_COMMANDS` and `NJOBS_COMMANDS`; `law-pass` takes
`--jobs <n_jobs>` when `n_jobs != 1` (as `extract all` does). The staleness trigger of a
comparator's split stage is its deterministic `features.csv`, so a re-run of `features` that
changes no number re-runs no split. `gates/preconditions.json` is an input of every step (the
existing test requires it of every non-internal step from move 3 on). Also in move 12, after
`tables (all)` and before `figures (all)`: `tables eusipco` (`tables_eusipco --out O`, outputs
`report/tables/eusipco_table2.csv`, inputs `cells.csv`, `report/tables/table7.csv`,
`gates/preconditions.json`) and `latex skeleton eusipco` (`latex_skeleton_eusipco --out O
[--standalone <--standalone-p2e-tex>]`, outputs `report/p2e_skeleton.tex`, inputs
`inputs/pass_table.csv`, `gates/preconditions.json`; the existing test exempts only the module
named `latex_skeleton` from the admissibility input, so this one carries it). The driver gains
`--standalone-p2e-tex PATH`. `plan --moves 12` prints the
stage; `status` lists it; the existing orderings that tests pin (`features combined all grid`
first in move 12; `gx combined` before `gl (all rungs)` before `gf all rungs at the selected
points` before `tables (all)`; move 13 is `gf check on Table 7` alone) are untouched.

### 3.5 Table 7, the matched rows and the matched dimension (builder 3, `tables.py`, `gates_comparison.py`, `models.py`)

3.5.1 The matched rows (`E1 sec. 4 M4`; `SPEC 6.3` against `CR 2.2 item 32`). Decision: the row
stays for every split; `gate_gdim` runs `models.run_split_stage(out, "combined", cg, split, ls,
reduce_to=d_star, reduce_method=method, base_dir="splits_matched", n_perm=n_perm,
run_null=(split in null_splits), ...)` for the five combinations
`MATCHED_COMBOS = (("loko", "archetype"), ("loro", "kernel"), ("loro", "archetype"),
("within_trace", "kernel"), ("within_trace", "archetype"))`, in that order, with one `d*` (chosen by
LOKO as today) for all five. `gates_comparison.py gdim` gains `--null-splits` (default
`loko,loro,within_trace`; the driver passes its own). `gdim.csv` gains the columns `split` and
`labelspace` appended after `matched_to`; the rung rows leave them blank; the `combined (matched)`
rows are five, the `loko` one first (the existing test reads the first match). `tables._matched_scores`
reads `gates/splits_matched/combined/<gid>/<split>__<ls>/scores.json` first, then the epoch-1
alternatives, then prints the existing `not run:` string. Within-trace at the whole-cell point
stays `not applicable: one window per cell`.

3.5.2 The matched dimension (`E1 sec. 4 M5`; `CHECK_2 M3`). `models.run_split_stage` adds to
`scores.json`, for the full run and for `with_quarantine`: `feature_count_used` (the `d_used` the
forest saw: an `int` when every fold used the same width, else the string `"<min>-<max>"`) and
`feature_count_used_per_fold` (`{fold name: d_used}`), taken from `fit_predict_units`' per-cell
`d_used`. `effective_scores` carries them like every other key. Table 7's `feature count` cell
prints `feature_count_used` when present, else `feature_count`; G-DIM's cell keeps the
pre-reduction width (`d = 60, matched 36, train_importance`). With the fixture's matched
`feature_count_used = 36`, the `combined (matched)` rows print `36`.

3.5.3 The comparator rows. When `load_comparator_registry(out)` is non-empty, Table 7 appends,
after the six `rung` rows of each block (the primary block of `(rung, split)` rows and the appended
archetype-space block), one row per comparator per split in registry order, so the row count is
`30 + 5 * n_comparators` (45 with three). Cells:

- `rung`: `f"comparator: {display} [{citation_key}]"`, for example
  `comparator: Savoldi 2010 [savoldi2010uncertainty]`, the same text in the CSV, the Markdown and
  the `.tex` (`write_table` escapes every cell; no `\cite` in Table 7).
- `resolution (W x H)`: `whole cell (per-run statistic)`.
- `feature count`, `accuracy`, `macro recall (headline rows)`, `null p95`, `rank`, `majority`: from
  `gates/splits/<id>/Wall_Hall/<split>__<ls>/scores.json` through `effective_scores`, the same
  code path as a rung row (`split_dir(out, id, "Wall_Hall", split, ls)`); a `scores.json` whose
  `status` starts with `not applicable` prints that string in the five score cells (this also
  serves the rungs' whole-cell rows; the existing tests accept `^not applicable`).
- `G-C`: `not applicable: comparator row (G-C is per rung)`.
- `G-F (i)`: `not applicable: one row per cell (comparator)` (non-empty, so move 13's
  `gf_check` passes on these rows).
- `G-L`: `not applicable: comparator row (G-L is the rungs' level rule)`.
- `G-DIM`: `_gdim_text(out, id)` (the comparator id is its `rung` value in `gdim.csv`).
- `G-M vs APF`: `_gm_text(out, id, split)`.
- `G-X`: `not run: G-X not run for comparators (SPEC_EPOCH2 section 7)`.

The `table7.params.json` records `comparators: [ids]` and the registry's sha256 among its inputs.

### 3.6 EUSIPCO Table 2 (builder 2, `tables_eusipco.py`)

Built from `report/tables/table7.csv` (P2E sec. 3.IV, "Table 2: reductions compared"; P2E sec. 5,
"Table 2 (from Table 7 plus the comparator)"). Rows, in this order and with these labels: `APF`
(`apf`), `wAPF` (`wapf`), `content-change` (`content`), `persistence` (`persist`), `combined`
(`combined`), then one row per comparator key in `--comparators` (default
`savoldi2010uncertainty,dhodapkar2003comparing`, `AD 2026-09-17`), labelled with the display text
of the Table 7 row (`Savoldi 2010`) and, in the `.tex`, `~\cite{<key>}`. A Table 7 comparator row is
recognised by the regex `^comparator: (?P<display>.+) \[(?P<key>[^\]]+)\]$` on the `rung` cell.
`--include-matched` (default off) appends `combined (matched)`.

Columns: `reduction, feature count, LOKO accuracy, LOKO macro recall (headline archetypes),
LOKO null p95, LOKO majority, LORO accuracy, margin vs APF (LOKO)`. Sources per row: the Table 7
row with `split == "LOKO"` and `label space == "archetype"` gives `feature count`, `accuracy`,
`macro recall (headline rows)`, `null p95`, `majority`, `G-M vs APF`; the row with `split ==
"LORO"` and `label space == "kernel"` gives `LORO accuracy`. Every cell is copied verbatim
(refusals printed as words; `--` stays `--`); a missing row prints `not run: table7.csv has no
<split>/<label space> row for <label>`; a missing `table7.csv` exits 2 with its path. Written as
`report/tables/eusipco_table2.{csv,md,tex}`: the CSV and the Markdown through
`_report_common.write_csv` and `md_table`; the `.tex` through builder 2's own writer
`tex_table_cite(columns, rows, *, label, cite_col, cite_keys)`, which produces exactly the shape of
`_report_common.tex_table` (`% columns:` line, `table*`, `\centering`, `\footnotesize`,
`\caption{}`, `\label{tab:p2e_table2}`, booktabs rules) and passes every cell through
`latex_escape` except that the `reduction` cell of a comparator row is emitted as
`f"{latex_escape(display)}~\\cite{{{key}}}"`. `eusipco_table2.params.json` beside it (`schema =
"plan11.eusipco_table2.v1"`, `params` with the comparator keys, `include_matched`, `inputs_sha256` of
`table7.csv`, `citation = "P2E sec. 3.IV Table 2; P2E sec. 5; AD 2026-09-17"`). Table 3 uses the
same writer with no cite column.

### 3.7 EUSIPCO Table 3 (builder 2, `tables_eusipco.py`)

The level-matched test (P2E sec. 3.IV, "Table 3"; P2 Sec. 2 and Sec. VI; `schema.LEVEL_MATCHED_SETS`).
Rows: set `A` (floyd, histogram, nbody) and set `B` (fft, gemm). Columns under four readings
`R in (apf, content, persist, dhodapkar_smith)`, displayed as `under APF`, `under content-change`,
`under persistence`, `under Dhodapkar-Smith 2003`:

The CSV carries, per `R`, four columns: `R: separating features` (`k`), `R: feature count` (`d`),
`R: LORO kernel recall (set mean)` (`r`), `R: within-set confusion` (`c`); the `.md` and `.tex`
print one compact cell per `R`: `f"{k}/{d} sep; LORO {r:.2f}; conf {c:.2f}"`, with `none` in place
of `k/d` when `k = 0`, and a `not run: ...` string verbatim when an input is missing. Then the
columns `alias check (APF)` and `gemm pass period (G-P)` (the latter filled on row `B` only, `--`
on row `A`), and the leading columns `set`, `kernels`.

Definitions, all readings, no verdict:

- `k`: the number of features that separate at least one pair of the set under the envelope rule,
  from `gates_calibration.separating_features(out, R, grid_id)` (al-Kindi item 5: disjoint ranges
  of the per-cell means, no threshold), counted as distinct feature names over the set's pairs;
  `grid_id` is the rung's selected point from `gates/selection.json` for a rung and `Wall_Hall`
  for the comparator; `d` is the feature file's width. For a rung with no selection: `not run: no
  selection for <R>`.
- `r`: the mean over the set's kernels of `recall_per_kernel[kernel]` from
  `gates/splits/<R>/<grid_id>/loro__kernel/scores.json` read through `effective_scores` (a
  kernel absent from the dict is skipped and the count of kernels used is written to the params).
- `c`: from the same split's predictions file (`predictions_with_quarantine.csv` when
  `scores.json["predictions_file"]` names it, else `predictions.csv`): among the set's cells with a
  `y_pred` that is not a `not run:` string, the fraction whose `y_pred` is a member of the same set
  other than `y_true`.
- `alias check (APF)`: the distinct `verdict` values of `gates/alias.csv` rows with `kind ==
  table6_feature` whose `pair_or_set` starts with the set letter, joined by `; `; `--` when none.
- `gemm pass period (G-P)`: from `gates/gp.csv`, the gemm rows' majority `verdict_pairs` and
  `within_pass_verdict`, printed `f"{verdict_pairs}; within pass: {within_pass_verdict}"`; `not
  run: gates/gp.csv missing` when absent.

Written as `report/tables/eusipco_table3.{csv,md,tex}` (`label = "tab:p2e_table3"`, `wide =
True`) with `eusipco_table3.params.json` (`schema = "plan11.eusipco_table3.v1"`, `citation =
"P2E sec. 3.IV Table 3; P2 Sec. 2 falsifier (2), Sec. VI; SPEC_review_al_kindi.md item 5"`,
`inputs_sha256` of every file read).

### 3.8 The EUSIPCO skeleton (builder 2, `latex_skeleton_eusipco.py`)

`build_p2e_skeleton(*, documentclass="IEEEtran") -> str` and `write_p2e_skeleton(out, *,
documentclass, standalone) -> dict`, writing `<out>/report/p2e_skeleton.tex` beside
`report/tables/eusipco_table2.tex`, `eusipco_table3.tex` and `report/figures/`, plus
`report/p2e_skeleton.json` (`targets`, `n_prose_lines`, `written`), and the standalone copy at
`apf_paper/p2e_skeleton.tex`. The document, in order, and every text line outside comments a
heading, a LaTeX command, a tabular row or environment syntax (so that
`latex_skeleton.prose_lines(tex) == []`):

1. `\documentclass[conference]{IEEEtran}` (or `[10pt,twocolumn]{article}`), the same package
   lines as `latex_skeleton.build_skeleton` (`inputenc`, `fontenc`, `booktabs`, `graphicx`,
   `amsmath`, `\graphicspath{{./}}`).
2. A comment header: written by whom and when; `source of truth: apf_paper/P2E_STRUCTURE.md
   sections 2, 3, 5 (2026-09-17); comparators: council/14, P2_AUTHOR_ANSWERS.md Decisions of
   2026-09-17`; the four gates of P2E sec. 0 as four comment lines; the compression map of P2E
   sec. 5 as one comment line per table row; the venue facts of P2E sec. 1 that shape the page
   (five pages, four to six sections, one to four equations, two to four result tables, 19 to 23
   references) as comment lines; `% working title: the author's (P2E sec. 7: shorter than paper
   2's; the representation in the title)`; `\title{}`, `\author{}`.
3. `\begin{abstract}` with the one claim of P2E sec. 2 as `% - ` bullets (the three axes, the
   twelve kernels at 500 ms, the level-matched separation, the named external feature set, the
   one-pass cost), `\end{abstract}`.
4. `\section{Introduction}` with P2E sec. 3.I's bullets as comments (the three-sentence framing,
   the folded related work, the three contributions, the pointer to the encoding paper).
5. `\section{The representation}` with P2E sec. 3.II's bullets, then four equation placeholders,
   each `% Eq. n: <P2E's one-line substance>` followed by
   `\begin{equation}\label{eq:<name>}\phantom{\cdot}\end{equation}` with names `set`, `breadth`,
   `content`, `persistence`; then Figure 1 through `latex_skeleton._figure("fused_plane",
   "fig:p2e_plane", "<P2E's Figure 1 line>", wide=True)`; then the resolution paragraph and the
   extraction paragraph as comment bullets.
6. `\section{Data and protocol}` with P2E sec. 3.III's bullets; Table 1 as a static shell from
   `latex_skeleton._shell(["field", "value"], "tab:p2e_table1", rows=TABLE1_ROWS, note="Table 1,
   the dataset (P2E sec. 3.III; DOI blank until the release)")` with `TABLE1_ROWS` the fragments of
   P2E sec. 3.III (`kernels: 12`; `reps per kernel: 8`; `guest time per cell: 600 s`; `configured
   interval: 500 ms`; `pairs per cell: about 930`; `guest memory: 1 GiB`; `N (pages): 262,144`;
   `launch labels: three, one commit`; `dataset DOI: ` blank; `idle cells: ` blank); the external
   comparator line and the two clocks line as comments.
7. `\section{Results}` with P2E sec. 3.IV's bullets; Table 2 through
   `latex_skeleton._generated("eusipco_table2", _shell(EUSIPCO_TABLE2_COLUMNS, "tab:p2e_table2",
   wide=True, size="scriptsize"))`; Table 3 likewise with `EUSIPCO_TABLE3_COLUMNS` (the display
   columns of 3.7); Figure 2 (optional) as `_figure("table8_assignment", "fig:p2e_assign", "optional:
   the store-predicted versus state-measured assignment, compact (from report/tables/table8.csv);
   no generator writes this file, the framed placeholder stands")`; the caveat bullets as comments.
8. `\section{Conclusion}` with P2E sec. 3.V's bullets.
9. The reference plan of P2E sec. 3 ("References: 19 to 23") as comment lines naming the bib keys
   that exist in `p2.bib` (`law2010volatile`, `savoldi2010uncertainty`,
   `oliveri2025inconsistencies`, `hirano2022ransomware`, `asanovic2006landscape`,
   `purnaye2022bishm`, `khoury2026architecture`, `vanderkouwe2019sok`, `kalibera2013rigorous`,
   `clark2005livemigration`, `dhodapkar2003comparing`, `dhodapkar2002managing`), then
   `% \bibliographystyle{IEEEtran}`, `% \bibliography{p2}`, `\end{document}`.

`EUSIPCO_TABLE2_COLUMNS` and `EUSIPCO_TABLE3_COLUMNS` are module constants of `tables_eusipco.py`
imported by the skeleton module. The `pdflatex` compile is not required (no TeX on this machine,
`E1 sec. 4`); the test checks structure (section 6.2).

### 3.9 The bibliography additions (builder 2, `apf_paper/p2.bib`, append only)

`law2010volatile`, `savoldi2010uncertainty` and `clark2005livemigration` already exist and are not
touched; the comparator citation keys of 3.2 point at the first two. Appended at the end of the
file, after the existing "For the author" block, under a header comment
`% F. Build epoch 2 additions (council/14_hunayn_exact_input_comparators.md, read 2026-09-17).
Fields limited to what council/14 states; not re-verified in this epoch; MISSING lines name what
the author fills.` Four entries, each preceded by its tier comment in the file's own convention:

```
% council/14 candidate 3; tier READ (Hunayn: jes.ece.wisc.edu/papers/micro03.ashutosh.pdf; IEEE Xplore 1253197).
% MISSING: given names, full proceedings title, publisher, DOI (council/14 gives family names, "MICRO-36, 2003, pp. 217-227").
@inproceedings{dhodapkar2003comparing,
  author    = {Dhodapkar and Smith},
  title     = {Comparing Program Phase Detection Techniques},
  booktitle = {MICRO-36},
  pages     = {217--227},
  year      = {2003}
}

% council/14 candidate 3; tier METADATA. MISSING: given names, full proceedings title, publisher, DOI.
@inproceedings{dhodapkar2002managing,
  author    = {Dhodapkar and Smith},
  title     = {Managing multi-configuration hardware via dynamic working set analysis},
  booktitle = {ISCA 2002},
  pages     = {233--244},
  year      = {2002}
}

% council/14 candidate 4; tier READ (cl.cam.ac.uk Akoush PDF). MISSING: co-authors' names, given names, publisher, DOI.
@inproceedings{akoush2010predicting,
  author    = {Akoush and others},
  title     = {Predicting the Performance of Virtual Machine Migration},
  booktitle = {MASCOTS 2010},
  pages     = {37--46},
  year      = {2010}
}

% council/14 candidate 4; tier READ (documentation; gitlab.com, qemu, qapi/migration.json). MISSING: exact URL, QEMU version or commit, access date beyond 2026-09-17.
@misc{qemu_calc_dirty_rate,
  title        = {calc-dirty-rate},
  howpublished = {QEMU source repository on gitlab.com, file qapi/migration.json (QAPI documentation)},
  note         = {Read 2026-09-17 (council/14 candidate 4)}
}
```

No field beyond these is entered. The builder may, if it opens the URL council/14 names and reads
the front matter, fill a MISSING field and change the tier line to `READ (<url>, <date>, epoch 2)`;
it never fills a field from memory.

---

## 4. The C1 re-map (builder 3, `gates_precondition.py`, `run_moves.py`)

Definition implemented (docstring citation: `AD 2026-09-17 "C1 for kernel cells": a kernel cell is
active if its maximum changed-page count is above the idle floor's 95th percentile once the idle
cells exist; until then a declared absolute of 0.1 percent of memory (262 pages), recorded in the
run's params; the inherited 0.02 stays only as a documented alternative. Idle cells keep the
epoch-1 re-map (CR 2.1 item 1: never refused). E1 sec. 4 M1 and sec. 6 item 7.`)

Constants (module level, `gates_precondition.py`):

```python
C1_RULE_DEFAULT = "auto"                 # "auto" | "idle_floor" | "absolute" | "legacy_apf_max"
C1_ABS_FRACTION = 0.001                  # AD 2026-09-17: 0.1 percent of memory
C1_ABS_PAGES = int(C1_ABS_FRACTION * schema.N_PAGES)      # 262
C1_IDLE_PERCENTILE = 95.0                # AD 2026-09-17; = GK0_IDLE_PERCENTILE
C1_IDLE_POOL = "pooled_snapshots"        # = GK0_IDLE_POOL; the same edge as G-K0's band
C1_MIN_IDLE_CELLS = 1
C1_ACTIVITY_MIN = 0.02                   # kept: the legacy apf_queue re-map (5,243 pages), alternative only
```

The rule, per kernel cell, on the sidecar's integer `K_max` (never on the float `apf_max`):

- `idle_floor`: `pass` iff `K_max > edge` (strict, "exceeds"), where `edge` is the
  `C1_IDLE_PERCENTILE`-th percentile of `K` pooled over the rows of the idle cells that enter the
  floor; an idle cell enters when its `cells.csv` status is `ok`, its sidecar exists, and its own
  `C2` and `C6` are `pass` (computed in the same run; C1 is not applicable to it). The admissibility
  record `inputs/idle_admissibility.json` is not required (as G-K0 does not require it; `E1 sec.
  6 item 18`). The edge is computed by one function shared with `gate_gk0`,
  `idle_band_edge(out, idle_cells, *, idle_pool, idle_percentile) -> float | None`, so C1's floor
  and G-K0's band are the same number when their parameters are the defaults.
- `absolute`: `pass` iff `K_max >= C1_ABS_PAGES` (262).
- `legacy_apf_max`: `pass` iff `apf_max >= c1_activity_min` (the epoch-1 rule, unchanged).
- `auto` (the default): `idle_floor` when at least `C1_MIN_IDLE_CELLS` idle cells enter the floor,
  else `absolute`.

Idle cells: `C1 = not applicable: control (C1 re-mapped)` as today. `all_hard_pass` unchanged in
form. Implementation: two passes over the cells inside `gate_preconditions`: the first computes
every column but C1 (so the idle cells' C2 and C6 are known), the second computes the edge and C1.

Flags on `gates_precondition preconditions`: `--c1-rule auto|idle_floor|absolute|legacy_apf_max`
(default `auto`), `--c1-abs-fraction 0.001`, `--c1-idle-percentile 95`, `--c1-min-idle-cells 1`,
and the existing `--c1-activity-min 0.02` (used only under `legacy_apf_max`). `--c1-rule
idle_floor` with no idle cell entering the floor is `refused: C1 idle floor requested, no idle
cell enters the floor` on every kernel row and `all_hard_pass = false` for them (a refusal, never a
silent fallback). The driver (`run_moves.py`) gains the same four flags (`--c1-rule`,
`--c1-abs-fraction`, `--c1-idle-percentile`, `--c1-activity-min`) and passes each to
`preconditions` only when the author sets it; the preconditions command records the values used
regardless.

Written to `gates/preconditions.csv`: the existing columns, then three new columns appended after
`all_hard_pass`: `C1_K_max` (int), `C1_threshold_pages` (the edge or 262 or 5243, as a number), and
`C1_rule` (`idle_floor_p95`, `absolute_0.001`, `legacy_apf_max_0.02`; blank on idle rows). Written
to `gates/preconditions.json` `params`: `C1_rule_requested`, `C1_rule_applied`,
`C1_threshold_pages`, `C1_idle_band_edge` (or `null`), `C1_idle_cells_in_floor` (the ids),
`C1_idle_percentile`, `C1_idle_pool`, `C1_abs_fraction`, `C1_abs_pages`, `C1_legacy_apf_max`
(0.02, "alternative, not applied" unless applied), `C1_change_record = "pre-registered gate C1
re-mapped for kernel cells: AD 2026-09-17; the epoch-1 rule (apf_max >= 0.02) kept as
legacy_apf_max"`. `PRE_COLUMNS` grows by the three names at its end; `_refresh_c7` rewrites with
`PRE_COLUMNS` and is unaffected.

The runbook's standard command does not change (the default is `auto`); the move-2 paragraph and
section 3 ("adding the idle cells") change as written in Appendix C: on the first run without idle
cells the params say `absolute_0.001`; once the idle cells are indexed and move 2 re-runs, C1
switches to `idle_floor_p95`, `preconditions.json` changes, and the staleness rule re-runs every
move from 3 on by the existing admissibility trigger.

---

## 5. The staleness rule and G-ORD (builder 3)

### 5.1 Keyed staleness (`run_moves.py`; `E1 sec. 4 M2`; `CERTIFY_al_farabi.md` section 7 item 3)

Today an input spec is a path and `inputs_sha256` hashes the whole file. `selection.json` is
written per rung by `select` and grows through moves 6 to 12, so a resume of moves 7 to 13 with
nothing changed re-runs every split stage (hours on the real corpus; never a wrong skip). The
refinement is content hashing of exactly the part of a file that a command reads, declared per
command:

Input spec forms (the `inputs` list of `_cmd`), hashed by a new `_input_hash(out, spec) -> str |
None` and collected by `_inputs_sha256(out, specs) -> dict` whose keys are the spec strings
themselves:

- `path`: sha256 of the bytes, as today; absent file, no entry (as today).
- `json:<path>:<key>`: `doc = read_json(path)`; the hashed object is
  `{"key": key, "entry": doc.get(key), "params": (doc.get("params") or {}).get(key),
  "params_inputs": (doc.get("params") or {}).get("inputs_sha256_" + key)}` serialised with
  `json.dumps(..., sort_keys=True, separators=(",", ":"), default=str)`; absent file, no entry.
- `csv:<path>:<col>=<value>`: the header list plus every row (as a dict) with `row[col] == value`,
  serialised the same way.

`_stale_reason` compares spec by spec; the message names the part: `stale: selection.json[apf]
changed since move 7 (...)`, `stale: g3_flags.csv[rung=apf] changed ...`, and the plain
`stale: pass_table.csv changed ...` as today.

Which commands change, and only these:

| command | epoch-1 input | epoch-2 input | what the command reads |
|---|---|---|---|
| `splits <rung>` (moves 7, 9, 10, 11, 12) | `gates/selection.json` | `json:gates/selection.json:<rung>` | `series.selected_grid_id(out, rung)` reads `doc[rung]` only |
| `gx <rung>` (the same moves) | `gates/selection.json` | `json:gates/selection.json:<rung>` | the same call |

Every other command keeps its whole-file inputs: `gl`, `gdim`, `gm`, `variance`, `cluster`,
`tables`, `gf all rungs at the selected points` read every rung's selection; `alias` and `tables
table6` read `g3_flags.csv` and `selection.json` and cost seconds; the `select <rung>` commands
already hash their thirteen per-rung grid CSVs. The admissibility proxy
(`gates/preconditions.json`, `E1 sec. 6 item 12`) is unchanged. The comparators' split stages
hash their `features.csv` (3.4).

The invariant, stated in the module docstring and proven by the table above: a command is skipped
only when (a) a `done` ledger record exists for its key, (b) every declared output exists, and (c)
for every declared input spec the keyed hash now equals the hash recorded at (a). The keyed hash
of `json:gates/selection.json:<rung>` covers `doc[<rung>]`, `doc["params"][<rung>]` and
`doc["params"]["inputs_sha256_<rung>"]`, which is everything `select <rung>` writes for that rung
and everything `splits <rung>` and `gx <rung>` read from the file; a change to any of it changes the
hash; a change to another rung's entry does not, and is not read. So no wrong skip is possible,
and the only cost of the change is one extra re-run of the split stages on the first resume of a
ledger written before this epoch, because the recorded key changes from the path to the spec
string (no real-data ledger exists yet, `E1 sec. 4`).

`_output_exists` already parses the `json:` and `csv:` forms; `_input_hash` reuses that parsing.

### 5.2 G-ORD parallelized (`gates_temporal.gate_gord`; `E1 sec. 4 M6`; `CHECK_3 M6`)

The two loops are the `n_order_perm` order shuffles and the `null_perm` label shuffles per `W`.
Plan, exactly, inside the existing `for W in ...` loop:

1. Pre-draw every random object in the main thread, in epoch 1's draw order, from the same seeds:
   `rng_o = default_rng(SEED_ORDER + seed_offset)`; `order_perms = [[rng_o.permutation(
   series_[c["cell_id"]].shape[0]) for c in cells] for _ in range(n_order_perm)]` (repetition
   outer, cell inner, which is exactly what `featurize(rng_o)` drew through `nulls.order_shuffle`);
   `rng_l = default_rng(SEED_LABEL_NULL + seed_offset)`; `label_perms =
   [nulls.shuffle_labels_units(ucells, ukern, uarc, "loko", "archetype", rng_l) for _ in
   range(null_perm)]`.
2. Two pure functions at module level: `_gord_shuffled_score(series_, cells, arche, perms, W, H,
   rung, seed, n_estimators) -> float` (re-windows each cell's series under its permutation, one
   LOKO accuracy) and `_gord_null_score(F, cid, ker, arc, labels_override, seed, n_estimators) ->
   float`.
3. `from joblib import Parallel, delayed`; `with Parallel(n_jobs=n_jobs, prefer="threads") as
   par:` `shuf = par(delayed(_gord_shuffled_score)(...) for perms in order_perms)` and `null =
   par(delayed(_gord_null_score)(...) for pl in label_perms)`. Threads, not processes: the
   forest's tree building and the numpy windowing release the GIL, nothing is pickled, the series
   dict is shared, and the result order is the input order.
4. The rest (`null_summary`, the verdict, `gord.json`, the copy into the thirteen grid CSVs) is
   unchanged. `params` records `n_jobs` and `parallel_backend = "joblib threads"`.

Because the forest seed is fixed and every random draw happens before dispatch, the output is
identical for any `n_jobs`, and identical to epoch 1's single-process output for the same inputs.
`gates_temporal grid --n-jobs` stays accepted and unused; its `params` gains `n_jobs_used = 1`.

---

## 6. Tests each builder must add

All tests use `synth.py`, `tests/_synth_b2.py` (extract level), or hand-written extracts and
trajectories; none reads anything outside the test's temporary directory. Small forests
(`n_estimators = 10`) and small permutation counts are fine; a test never asserts a verdict that
depends on a null below 500 draws except the `not run: N permutations < 500` string itself.

### 6.1 Builder 1, `tests/test_comparators.py` (helper `tests/comparator_fixtures.py`)

The helper writes, into a fresh directory: `cells.csv` (`SPEC 2.7` columns), per cell an
`extract.csv` with the 58 columns of `schema.EXTRACT_COLUMNS` holding a given `K` array and a given
`J` array (other columns blank or 0) and a `sidecar.json` with `status = ok`, `n_pairs`,
`seq_first = 1`, `seq_last = n_pairs`, `K_max`, `apf_max`, `failed_count = 0`; and, on request, a
plain-text trajectory file at the cell's `path`/`traj_file` from a `{seq: [pages]}` dict with the
66-column header and `hamming = l0 = l1 = 1`.

1. `test_savoldi_u_analytic`: `K = [10, 20, ..., 100]` gives `K_mean = 55`, `K_sd =
   30.276503540974915` (`ddof = 1`), `K_tail80_mean = 65`, `K_tail80_sd = 24.49489742783178`;
   `features.csv` and the npz row agree to `1e-12`; `per_kernel.csv` for a kernel with two such
   cells (the second `K + 10`) holds the mean of the two rows.
2. `test_dhodapkar_smith_phases_analytic`: the two worked examples of 2.2, every number at every
   sweep point, from `phases()` directly and from `features.csv` and `sweep.csv` (four rows per
   cell); the feature row equals the `0.04` sweep row; `--delta-th 0.16` moves the feature row and
   the registry records it.
3. `test_law_page_runs_analytic`: the twelve-seq trajectory of 2.3 gives every number listed
   there (`n_ever_changed`, the two max arrays via the histograms, the eight fractions, the two run
   histograms); the gap variant (rows of `seq = 8` removed) gives the listed changes; `n_seq_gaps
   = 1`.
4. `test_law_against_synth_truth`: `synth.write_cell(SynthSpec("gemm", 42, n_pairs=40, K0=64,
   churn=0.2, floor_F=0), root, keep_sets=True)`, then `extract.extract_cell`, then `law-pass`; a
   ten-line reference implementation in the test computes the longest membership and
   non-membership run per page from `truth.json["sets"]` and the fractions match exactly.
5. `test_refusal_too_few_pairs`: a cell with `n_pairs = 1` gives savoldi `refused: too few pairs
   (1 < 2) for savoldi`; `n_pairs = 2` gives dhodapkar_smith `refused: too few pairs (2 < 3) for
   dhodapkar_smith`; a six-seq trajectory gives law `refused: too few pairs (6 < 10) for law`; in
   every case the cell is absent from the npz, present in `features.csv` with the string, and
   counted in the registry's `n_cells_refused` and `refusals`.
6. `test_feature_file_shape_and_split_stage`: on `_synth_b2` corpus `corpus(reps=2, idle=1,
   n_pairs=40)` (or `synth.py corpus` plus `extract all`, at the builder's choice), `features`
   writes the three npz files; `series.load_features` loads each; `feature_names`, `W == -1`,
   `grid_id == "Wall_Hall"`, one row per ok cell, `win_start` all 0, `role` carries `idle`;
   `splits.make_labels(feat)` works; `run_comparator_splits(out, "savoldi", n_perm=5,
   null_splits={"loko"}, n_jobs=1, n_estimators=10, seed_offset=0)` writes the five split
   directories; `loko__archetype/scores.json` has `feature_count == 2`, `b1_g1 == "not run: 5
   permutations < 500"`, a numeric `majority`; `within_trace__kernel/scores.json["status"] ==
   "not applicable: one window per cell"`; `loro__kernel/predictions.csv` has one row per kernel
   cell.
7. `test_pair_comparators_exclude_refused_failed_cells`: skipped with the reason of 3.3 when
   `"dhodapkar_smith" not in series.PAIR_RUNGS`; otherwise a cell whose `preconditions.csv`
   `failed_verdict` is `refused: failed count 2 > 0, seq axis uncorrected` (written by the
   fixture) is absent from `dhodapkar_smith`'s and `law`'s `predictions.csv` and present in
   `savoldi`'s; `per_kernel.csv` of the two pair comparators excludes it and the params list it
   under `excluded_cells_pair_rungs`.
8. `test_registry_and_cli_exit_codes`: `status` exits 2 before `features` and 0 after; `features`
   without the page-run index exits 2 naming `page_runs/index.csv`; `features --only
   savoldi,dhodapkar_smith` exits 0 without it; a second `law-pass` leaves `index.csv`
   byte-identical; `splits --comparator law` with a monkeypatched `series.PAIR_RUNGS` lacking the
   comparators exits 1 with the message of 3.1.

### 6.2 Builder 2, `tests/test_eusipco.py` (fixtures under `tests/fixtures_eusipco/`)

The fixture is fixed files, not generated: `table7.csv` with the exact Table 7 columns and 45 rows
in the format of 3.5.3 (six rungs and three comparator rows per split, primary block then
appended block, with numeric cells, one `not applicable: one window per cell` within-trace
comparator row, one `near_unfalsifiable` cell, one `not run:` cell); `expected_table2.csv`,
`expected_table3.csv`; `gp.csv` (gemm rows `aliased by design at this size`, `pass aliased`);
`alias.csv` with two `table6_feature` rows for set `A`; and a small `selection.json`. The
feature files and the LORO split files are written by the test from fixed arrays (three cells per
kernel of the two sets, two features, chosen so that set `A` has one separating feature under
`content` and none under `apf`, and `dhodapkar_smith` at `Wall_Hall` has one), with
`scores.json` `recall_per_kernel` and `predictions.csv` fixed so that `r` and `c` are known.

1. `test_table2_from_fixture`: `tables_eusipco.table2(out)` writes the three files; the CSV equals
   `expected_table2.csv` cell for cell (seven rows, eight columns); `--include-matched` adds the
   eighth row; a fixture without the `dhodapkar2003comparing` rows prints the `not run: table7.csv
   has no ...` string in that row's cells; the `.tex` carries `\cite{savoldi2010uncertainty}`,
   `\caption{}`, `\label{tab:p2e_table2}` and no line outside `tabular` that
   `latex_skeleton.prose_lines` flags.
2. `test_table3_from_fixture`: the CSV equals `expected_table3.csv` (two rows; for each of the four
   readings the four decomposed columns; `alias check (APF)` from the fixture's two rows; row `B`'s
   G-P text `aliased by design at this size; within pass: pass aliased`); the `.md` compact cell
   for set `A` under `apf` reads `none; LORO 0.33; conf 0.67` with the fixture's numbers; a missing
   selection for `persist` prints `not run: no selection for persist` in the four cells.
3. `test_p2e_skeleton_structure`: `build_p2e_skeleton()` starts with
   `\documentclass[conference]{IEEEtran}`; `--documentclass article` gives the article line;
   exactly five `\section{` lines with the titles of 3.8; four `\begin{equation}` with labels
   `eq:set`, `eq:breadth`, `eq:content`, `eq:persistence`; `\IfFileExists{tables/eusipco_table2.tex}`
   and `...table3.tex`; `\IfFileExists{figures/fig_fused_plane.pdf}`; `latex_skeleton.prose_lines(tex)
   == []`; after `tables_eusipco.run(out)` on the fixture, every `latex_skeleton.targets` entry that
   names a table exists under `<out>/report/`; `write_p2e_skeleton(out, standalone=tmp/p2e.tex)`
   writes both copies identical; braces balance and every `\begin{X}` has its `\end{X}`.
4. `test_bib_block_appended`: `apf_paper/p2.bib` parses as a sequence of `@type{key,` blocks; the
   four keys of 3.9 exist exactly once each; every pre-existing key of the file still exists; the
   text before the appended header is byte-identical to a sha256 recorded in the test at the time
   of appending (the test stores the hash of the pre-epoch-2 prefix and its length); the appended
   block contains no field other than those listed in 3.9.

### 6.3 Builder 3 (appended to the existing test modules)

1. `tests/test_gates_precondition.py::test_c1_remap_absolute_default_without_idle`: a corpus of
   two kernel cells built with `_synth_b2` (`floor_F = 0`; `K0 = 300` and `K0 = 100`) and no idle
   cell: C1 `pass` and `fail`; `C1_rule == "absolute_0.001"`, `C1_threshold_pages == 262` on both
   rows; `preconditions.json` `params["C1_rule_applied"] == "absolute"`, `params["C1_idle_band_edge"]
   is None`; the same corpus with `--c1-rule legacy_apf_max` refuses both (`K_max < 5243`) and
   records `legacy_apf_max_0.02`.
2. `tests/test_gates_precondition.py::test_c1_remap_idle_floor_when_idle_cells_exist`: the same
   two kernel cells plus a kernel cell at `K0 = 0, floor_F = 100, floor_churn = 0` (its `K_max =
   100`) plus three idle cells from `SY.IDLE_PRESET`: `C1_rule == "idle_floor_p95"` on kernel rows,
   `C1_threshold_pages` equals `gate_gk0`'s `idle_band_edge` after `gate_gk0` runs on the same
   directory (to `1e-9`), the `K0 = 300` cell passes, the `floor_F = 100` cell fails (`100 <=
   edge`), idle rows keep `C1 = not applicable: control (C1 re-mapped)` and `all_hard_pass = true`;
   `--c1-rule absolute` on the same directory applies 262 and records `C1_idle_band_edge` anyway;
   `--c1-rule idle_floor` on the idle-free corpus writes the refusal string of section 4 on every
   kernel row.
3. `tests/test_gates_temporal.py::test_gord_equal_for_one_and_four_jobs`: `corpus(reps=2, idle=0,
   n_pairs=60, kernels=["gemm", "gibbs", "fft", "lexer"])`, `gate_grid(out, "apf", n_surrogates=2)`,
   `gate_gord(out, "apf", n_order_perm=2, null_perm=4, n_estimators=10, n_jobs=1)`; copy the four
   `gord.json`; run again with `n_jobs=4`; every payload key equal except `params["n_jobs"]`
   (compare `score_ordered`, `score_shuffled` list, `null_summary`, `GORD`); the
   `temporal_per_kernel.csv` GORD columns equal.
4. `tests/test_driver.py::test_unchanged_resume_runs_zero_split_stages`: on `make_out`, write
   `selection.json` holding `apf` only; seed the ledger with `done` records for `splits apf` and
   `gx apf` whose `inputs_sha256` is `run_moves._inputs_sha256(out, cmd["inputs"])` at that
   moment; restore the full five-rung `selection.json`; run `run --moves 7 --only-modules driver`
   and assert both records read `skipped: outputs exist and inputs unchanged` (the skip test
   precedes the module filter in `run_plan`, so no module runs); then change
   `selection.json["apf"]["grid_id"]` and assert `splits apf` is `stale: selection.json[apf]
   changed since move 7 (...)` while a seeded `splits persist` record stays skipped; and
   `_input_hash(out, "csv:gates/gx.csv:rung=apf")` changes when an `apf` row changes and not when a
   `persist` row is appended.
5. `tests/test_driver.py::test_move_table_epoch2`: the comparators stage's five steps exist in
   move 12 in the order of 3.4, between `gx combined` and `gl (all rungs)`; each carries
   `gates/preconditions.json`; `comparators: splits savoldi` gets `--n-jobs` and `--seed-offset`
   under `_ns(seed_offset=3, n_jobs=4)`; `comparators: law pass` gets `--jobs 4`; `tables eusipco`
   sits after `tables (all)` and before `figures (all)`; `_module_path("comparators")` ends with
   `comparators/__main__.py`; the C1 flags appear on `preconditions` only when set
   (`_ns(c1_rule="absolute")` adds `--c1-rule absolute`; the default adds nothing).
6. `tests/test_report.py` (fixture extended: `splits_matched/combined/<gid>/` for the five
   combinations with `feature_count_used = 36` in each `scores.json`, and an optional
   `comparators` flag on `make_out` writing the registry of 3.2 plus `gates/splits/<id>/Wall_Hall/`
   for the five combinations and `gdim.csv`/`gm.csv` comparator rows):
   `test_table7_matched_rows_all_splits` asserts numeric accuracy on the `combined (matched)` LORO
   and within-trace rows in both label spaces and `feature count == "36"` on all five, while the
   `combined` rows keep `"60"` and row count stays 30 without comparators;
   `test_table7_comparator_rows` asserts 45 rows with `comparators=True`, the rung strings
   `comparator: Savoldi 2010 [savoldi2010uncertainty]` (and the other two), `resolution (W x H) ==
   "whole cell (per-run statistic)"`, the five fixed cells of 3.5.3, the within-trace comparator
   rows printing `not applicable: one window per cell` in the score cells, and `gf_check` reading
   `pass` on the table.
7. `tests/test_gates_comparison.py::test_gdim_matched_all_splits_and_comparator_rows`: on the
   96-cell corpus of the existing `test_gdim_...`, after `_prep` for `apf` and `combined` in every
   split (the existing helper extended to `--all-splits`), `gate_gdim(out, n_perm=0,
   null_splits=set())` writes five `combined (matched)` rows with `split`/`labelspace` filled and
   the five `splits_matched` directories; with a fixture registry and comparator `scores.json`
   written by hand, `gdim.csv` has a `savoldi` row with `d == "2"` and `gm.csv` has
   `(savoldi, apf)` and `(apf, savoldi)` rows on `loko`.
8. `tests/test_gates_models.py` is not builder 3's; the `feature_count_used` key is asserted in
   `tests/test_gates_comparison.py` on a `run_split_stage(..., reduce_to=8)` result:
   `feature_count == 60`, `feature_count_used == 8`, `feature_count_used_per_fold` has one entry
   per fold.

---

## 7. For the author: every choice this addendum had to make

Each is a parameter or a documented decision; the default is what runs unless the author says
otherwise, and the value used is written into `params`. Builder 3 appends these to `SPEC.md`
section 8 as items 41 onward.

41. **The comparators' place in the move order.** Al-Kindi's numbers 0 to 13 are kept (a test
    pins `parse_moves`, and the runbook, the epoch-1 report and `SPEC.md` all name move 12 as the
    combined rung and the report, move 13 as the tripwire). The comparators are a named stage
    inside move 12, after `gx combined` and before `gl (all rungs)`, which is after every rung's
    split stage and before every comparison and table. Giving them their own number means
    renumbering 12 and 13 and editing three epoch-1 assertions; it was not done here.
42. **One row per cell, at the whole-cell point.** Savoldi's U, Dhodapkar-Smith's stability and
    phase length and Law's fractions are per-run statistics by their definitions, so the feature
    file is one row per cell at `Wall_Hall`; within-trace is `not applicable: one window per
    cell`, the split stage's own string, and EUSIPCO Table 2 has no within-trace column. A
    head/tail two-row variant (the comparator on the first 80 and the last 20 percent of a cell)
    would make within-trace run and is not built.
43. **No level normalization of the comparators.** They are written as published (`normalization
    = "none (published definition)"`), so Savoldi's row is level-inclusive by construction; G-L is
    not run on comparator rows and Table 7 says so. `SAVOLDI_UNITS = "pages"`; the percent-of-N
    form is a reading in `per_kernel.csv`.
44. **Savoldi**: `SAVOLDI_DDOF = 1` (the paper says "sample"; alternative 0); `SAVOLDI_SPAN =
    "whole"` with the G-K0-tail pair kept beside it (`"tail80"` swaps them into the row); the
    feature is named `sd` so that `feature_drop`'s last-component match never drops it;
    `SAVOLDI_MIN_PAIRS = 2`.
45. **Dhodapkar-Smith**: the feature row is the three phase numbers (`DS_FEATURES = "phase"`;
    `"phase_and_delta_mean"` adds the mean distance, which overlaps the persistence rung's own
    headline); `MIN_PHASE_LEN = 1` (the 2003 paper's minimum phase length, if any, is not stated in
    council/14); an undefined delta (both sets empty) is excluded from every count and ends a
    phase; the first interval has no delta and is outside the denominator; `DELTA_TH_DEFAULT =
    0.04` is the author's own decision and its "ROC knee" provenance is recorded, not verified; the
    hashed working-set-signature variant is not built; `DS_MIN_PAIRS = 3`.
46. **Law**: `X_SET = (2, 3, 5, 10)` as declared; `LAW_DENOMINATOR = "ever_changed"` (the
    fractions are over ever-changed pages; `"all_pages"` divides by `N` and makes every
    never-changed page static for every `X`); leading and trailing non-membership runs count as
    runs; `dyn_X2` is identically 1.0 and stays because the set was declared; `LAW_MIN_PAIRS =
    10`; the second streaming pass costs one to three minutes per real cell and `law-pass --jobs`
    parallelizes over cells; the per-page run index is kept in each cell's JSON so any other `X`
    needs no new pass.
47. **The failed-count exclusion** applies to Dhodapkar-Smith and Law (consecutive-set relations)
    exactly as to the pair rungs, and not to Savoldi (K only, like APF); `series.PAIR_RUNGS` lists
    them.
48. **Which gates the comparators enter**: the three splits, the unit-level null, the majority
    baseline, B1-G3's quarantine, G-K0's relabelling, G-N's headline classes, G-DIM's feature
    count and G-M against APF and every rung. Not G-C (per rung), not G-F (i) (no windows), not
    G-L (the rungs' level rule), not G-X (not run in this epoch; it is cheap and the author may ask
    for it), and comparators never enter the choice of `d*`.
49. **Table 7's comparator rows** carry the word `comparator`, the display name and the bib key in
    the `rung` cell, `whole cell (per-run statistic)` as resolution, and the fixed not-applicable
    strings of 3.5.3 in the G-C, G-F (i), G-L and G-X cells; the row count becomes 45.
50. **EUSIPCO Table 2** has the seven rows of `AD 2026-09-17` and eight columns; Law is excluded by
    default (`--comparators` adds `law2010volatile` for the IFIP version) and the matched row is
    excluded by default (`--include-matched`).
51. **EUSIPCO Table 3** prints readings and no verdict: the envelope count of separating features
    (al-Kindi item 5), the set-mean LORO kernel recall, the within-set confusion, the alias
    verdicts for APF, and gemm's G-P line; the four readings are `apf`, `content`, `persist`,
    `dhodapkar_smith`; "separable: no" is the author's sentence, written when `k = 0`.
52. **The skeleton** is IEEEtran conference by default, with four empty equation environments
    (`\phantom{\cdot}` bodies), a static Table 1 of P2E's fragments with the DOI blank, and a
    framed placeholder for the optional Figure 2 that no generator fills.
53. **The bibliography block** enters only the fields council/14 states; given names, full
    proceedings titles, publishers and DOIs are `MISSING` lines for the author (or for a builder
    who reads the named URL); `akoush2010predicting` and `qemu_calc_dirty_rate` are entered because
    council/14 names them, and the author decides whether they are cited.
54. **C1's two operators**: `>=` against the absolute (262 pages, `int(0.001 * N)`) and strict `>`
    against the idle edge ("exceeds"); the operand is the integer `K_max`, never `apf_max`.
55. **C1's floor** is the 95th percentile of `K` pooled over idle rows (`pooled_snapshots`), the
    same function and defaults as G-K0's band, from at least one idle cell whose own C2 and C6 pass;
    the admissibility record is not required (as for G-K0, `E1 sec. 6 item 18`); with idle cells
    present the maximum of about 930 draws of an idle-like kernel cell will usually exceed the
    idle 95th percentile, so this C1 refuses only a cell that never once rises above the idle band,
    which is the reading the author asked for (the lexer reaches G-K0).
56. **`legacy_apf_max`** (0.02, 5,243 pages) stays selectable and is recorded as the documented
    alternative; `auto` is the default and records which branch it took.
57. **Keyed staleness** is applied to the split stages and G-X only; `gl`, `alias`, `tables
    table6`, `gdim`, `gm`, `variance`, `cluster` and `tables` keep whole-file hashes because they
    read every rung or cost seconds; the admissibility proxy is unchanged; the first resume of a
    pre-epoch-2 ledger re-runs the split stages once because the recorded key changes.
58. **G-ORD** uses joblib threads with every random draw made before dispatch, so the result is
    identical for any `n_jobs` and to epoch 1's; `grid --n-jobs` remains accepted and unused.
59. **The matched rows** run for all five (split, label space) combinations at the one `d*`
    chosen by LOKO, under `gdim --null-splits` (the LORO matched null costs what LORO's own null
    costs); the row is never removed.
60. **`feature_count_used`** is what Table 7 prints when present; the pre-reduction width stays in
    the G-DIM cell.
61. **The version string stays `0.1.0`** because `tests/test_schema.py` pins it; epoch 2 is
    recorded as `"epoch": 2` in the new result files' `params` and in `SPEC.md` section 1.
62. **The runbook is builder 2's file** in this epoch; the C1, staleness, G-ORD and matched-row
    paragraphs are written in Appendix C and pasted, so builder 3 edits no runbook text.

---

## Appendix A. The driver's move table after epoch 2 (builder 3 writes this into `SPEC.md` section 7)

Every command is `python3 -m plan11_encoding_ladder.<module> <sub> --out <out> ...`; `O` is
`--out <out>`; the driver adds `--seed-offset` to the random commands when non-zero and `--n-jobs`
(or `--jobs`) to the commands that take it when not 1; `ADM = gates/preconditions.json` is a
declared input of every non-internal command from move 3 on except the two skeleton writers and
`alias`.

| Move | Steps, in order (name: command) | Writes |
|---|---|---|
| 0 | `extract index`: `extract index --root R O` | `cells.csv` |
| 1 | `extract all`: `extract all --cells-csv CSV O --jobs N [--persist-side] [--failed-counts]` | `extract/*` |
| 2 | `preconditions`: `gates_precondition preconditions O [--assume-failed-zero --assume-reason T] [--failed-counts] [--c1-rule ...] [--c1-abs-fraction] [--c1-idle-percentile] [--c1-activity-min]`; four templates (`pass-table`, `gk0-template`, `idle-admissibility-template`, `head-drop-template`; never overwritten); `gp`: `gates_calibration gp O` | `gates/preconditions.*`, `gates/failed_counts.csv`, `inputs/*`, `gates/gp.csv` |
| 3 | `gc apf`, `gc persist`, `gc content`, `gc wapf`, `gc combined`: `gates_calibration gc O --rung R` | `gates/gc.csv` |
| 4 | `gk0`: `gates_precondition gk0 O`; `gf all rungs at W8_H4`: `gates_precondition gf O --all-rungs --grid-id W8_H4 --n-perm P` | `gates/gk0.csv`, `gates/gf.csv`, `gates/gf_floors.json` |
| 5 | `figures apf_per_kernel,level_matched` | two figures |
| 6 | `features apf all grid`: `series features O --rung apf --all-grid --both`; `grid apf`; `g3 apf`; `gord apf`; `select apf` (all `gates_temporal ... O --rung apf`); `alias`: `gates_calibration alias O` | `features/apf/*`, `gates/grid/apf/*`, `g3_flags.csv`, `table5_*`, `selection.json`, `alias.csv` |
| 7 | `grid complete apf` (internal); `splits apf`: `models splits O --rung apf --all-splits --raw-and-norm --null-perm P --null-splits S` (input `json:gates/selection.json:apf`); `gx apf`: `gates_comparison gx O --rung apf --null-perm P` (the same keyed input); `gl`; `gn`; `alias (again, table6 features)`; `tables table6` | `gates/splits/apf/*`, `gl.csv`, `gn.csv`, `gx.*`, `report/tables/table6.*` |
| 8 | `gj`: `gates_readings gj O`; `figures fused_plane` | `gj.*`, `gj_mask/`, the fused plane |
| 9 | the five temporal steps for `persist`; `grid complete persist`; `splits persist` (`--norm`); `gx persist`; `figures j_hist` | as 6 and 7 for `persist` |
| 10 | the same for `content`; `gdec`: `gates_readings gdec O`; `figures ratio_hist,floyd_decay` | as above, `gdec.csv` |
| 11 | the same for `wapf`; `tables wapf_over_apf` | as above, `report/tables/table_wapf_over_apf.*` |
| 12 | the five temporal steps for `combined` (`--norm`); `grid complete combined`; `splits combined`; `gx combined`; **`comparators: law pass`**: `comparators law-pass O [--jobs N]`; **`comparators: features`**: `comparators features O`; **`comparators: splits savoldi`**, **`... dhodapkar_smith`**, **`... law`**: `comparators splits O --comparator C --null-perm P --null-splits S`; `gl (all rungs)`; `gf all rungs at the selected points`: `gates_precondition gf O --all-rungs --n-perm P`; `gdim`: `gates_comparison gdim O --null-perm P --null-splits S`; `gm`; `variance`; `cluster`: `models cluster O --rung combined --null-perm P`; `tables (all)`: `tables O --table8-rung combined`; **`tables eusipco`**: `tables_eusipco O`; `latex skeleton`: `latex_skeleton O [--standalone PATH]`; **`latex skeleton eusipco`**: `latex_skeleton_eusipco O [--standalone PATH]` (with `ADM`); `figures (all)`; `tables manifest` | `gates/comparators/*`, `features/<comparator>/*`, `gates/splits/<comparator>/*`, `gdim.csv`, `gm.csv`, `gv.*`, `clustering.*`, `report/tables/*` (Table 7 with the matched and comparator rows; `eusipco_table2`, `eusipco_table3`), `report/paper2_skeleton.tex`, `report/p2e_skeleton.tex`, `report/figures/*`, `report/manifest.json` |
| 13 | `gf check on Table 7` (internal) | `gates/gf_check.json` |

## Appendix B. CLI contract additions (builder 3 appends these to `SPEC.md` section 7.1)

```
comparators law-pass  --out O [--cells-csv CSV] [--jobs 1] [--force]
comparators features  --out O [--cells-csv CSV] [--only C,...] [--delta-th 0.04]
                      [--delta-th-sweep 0.02,0.04,0.08,0.16] [--min-phase-len 1]
                      [--ds-features phase|phase_and_delta_mean] [--x-set 2,3,5,10]
                      [--law-denominator ever_changed|all_pages] [--savoldi-span whole|tail80]
                      [--savoldi-ddof 1] [--tail-fraction 0.8]
comparators splits    --out O (--comparator C | --all) [--null-perm 500]
                      [--null-splits loko,loro,within_trace] [--n-jobs 1] [--n-estimators 300]
                      [--seed-offset 0]
comparators status    --out O
comparators.savoldi / comparators.dhodapkar_smith / comparators.law   --out O [their options above]
tables_eusipco        --out O [--only table2,table3]
                      [--comparators savoldi2010uncertainty,dhodapkar2003comparing] [--include-matched]
latex_skeleton_eusipco --out O [--documentclass IEEEtran|article] [--standalone PATH]
gates_precondition preconditions  ... [--c1-rule auto|idle_floor|absolute|legacy_apf_max]
                      [--c1-abs-fraction 0.001] [--c1-idle-percentile 95] [--c1-min-idle-cells 1]
                      [--c1-activity-min 0.02]
gates_comparison gdim ... [--null-splits loko,loro,within_trace]
driver.py run / run_moves.py run  ... [--c1-rule ...] [--c1-abs-fraction ...] [--c1-idle-percentile ...]
                      [--c1-activity-min ...] [--standalone-p2e-tex PATH]
```

## Appendix C. Runbook text (builder 2 pastes these, in the places named)

**Section 1, the driver, after the resume-rule bullet.** "Epoch 2 refined the resume rule for the
split stages and G-X: their trigger on `gates/selection.json` is the rung's own entry, not the whole
file, so a resume of moves 7 to 13 with nothing changed re-runs no split stage (the ledger prints
`stale: selection.json[<rung>] changed since move <n>` when the rung's own selection did change).
Every other command keeps the whole-file rule. The first resume of a ledger written before epoch 2
re-runs the split stages once, because the recorded key changed."

**Move 2, after the `Look at` paragraph.** "C1 for kernel cells is re-mapped (P2_AUTHOR_ANSWERS.md,
Decisions of 2026-09-17). Without idle cells a kernel cell is active when its `K_max` is at least
262 pages (0.1 percent of memory; `C1_rule = absolute_0.001` in `preconditions.csv` and
`preconditions.json`). Once idle cells are indexed and pass their own C2 and C6, the rule switches
by itself to `idle_floor_p95`: active when `K_max` exceeds the 95th percentile of `K` pooled over the
idle cells' rows, the same edge G-K0 uses; the edge and the idle cells used are in
`preconditions.json` `params`. The inherited 0.02 (5,243 pages) is `--c1-rule legacy_apf_max` and
is recorded as the alternative. Look at the new columns `C1_K_max`, `C1_threshold_pages`, `C1_rule`."

**Move 6, after the `Look at` paragraph.** "`gord` honours `--n-jobs` since epoch 2 (joblib
threads over its two loops; the numbers do not depend on the job count). `grid` still runs in one
process."

**Move 12, a new subsection "The comparators stage" placed after `gx combined` in the command
list.** The five commands of section 3.4 as the driver prints them, then: "Writes
`gates/comparators/registry.json`, per comparator `gates/comparators/<id>/{features.csv,
per_kernel.csv, params.json}` (`sweep.csv` for Dhodapkar-Smith; `page_runs/` for Law), the feature
files `features/<id>/Wall_Hall_norm.npz`, and the split stages `gates/splits/<id>/Wall_Hall/...`
with the same four files as a rung. Law re-streams every trajectory once (one to three minutes per
real cell; `--jobs` parallelizes over cells; a finished cell is skipped on re-run). Look at:
`per_kernel.csv` of Savoldi (U = mean +/- SD of K per kernel, in pages and as percent of N);
Dhodapkar-Smith's `sweep.csv` at the four thresholds with 0.04 the row in the feature file; Law's
`dyn_X`/`stat_X` per kernel; in Table 7 the rows labelled `comparator:` with their null, majority
and `G-M vs APF`; within-trace on a comparator reads `not applicable: one window per cell` by
design (a per-run statistic)."

**Move 12, after the `Writes` paragraph.** "Table 7's `combined (matched)` row is now filled for
every split and label space (`gates/splits_matched/combined/<grid_id>/<split>__<labelspace>/`),
and its `feature count` prints the matched dimension (`feature_count_used` in `scores.json`); the
pre-reduction width stays in the G-DIM cell. `gdim --null-splits` follows the driver's setting, so
the LORO matched null costs what LORO's own null costs. The EUSIPCO tables land as
`report/tables/eusipco_table2.*` (from Table 7's rows: APF, wAPF, content-change, persistence,
combined, Savoldi 2010, Dhodapkar-Smith 2003) and `eusipco_table3.*` (the level-matched sets under
APF, content-change, persistence and Dhodapkar-Smith), and the five-page skeleton as
`report/p2e_skeleton.tex` (standalone copy: `apf_paper/p2e_skeleton.tex`, `--standalone-p2e-tex`)."

**Section 3, adding the idle cells, step 4.** Append: "C1 switches from the absolute of 262 pages
to the idle floor's 95th percentile on this re-run; `preconditions.json` records the switch, and
because that file is the admissibility input of every later move, everything from move 3 re-runs."
