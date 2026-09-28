# SPEC_epoch2_review_al_farabi.md: certification of the build epoch 2 addendum, before the build

Written 2026-09-17 by al-Farabi, on `SPEC_epoch2.md` as it stands (byte-identical to
`SPEC_epoch2.two_builders.md`; the three-builder draft is not under review). I read the addendum in
full, my own `CERTIFY_al_farabi.md`, `council/14_hunayn_exact_input_comparators.md` sections 1 to 3
and its "For the author" list, `P2_AUTHOR_ANSWERS.md` item T1 and the decisions of 2026-09-17,
`P2_STRUCTURE.md` section V (the G-K0, G-F, G-L, G-X, G-DIM, G-M definitions and the Plan 08
restatement), the `Baseline` row of its section 0, `P2E_STRUCTURE.md` section 7, `CHECK_3.md`'s
thirteen headings and `EPOCH1_BUILD_REPORT.md` section 4. Every claim the addendum makes about the
code I checked against the modules on disk: `series.py`, `models.py`, `gates_comparison.py`,
`gates_precondition.py`, `gates_temporal.py`, `run_moves.py`, `tables.py`, `_report_common.py`,
`verdicts.py`, `extract.py`, `synth.py`, `schema.py`, and the tests. No server was touched, no path
under `/mnt/nfs` or `/project` was read, the sandbox family is not named, no council file and no
paper file was edited, nothing was committed. One numerical check was run on synthetic data from
the in-test generator in the session scratchpad (section 4). This file is the only file I wrote.

## Summary

| Condition | Verdict | SPEC_epoch2 lines | What must change before the build |
|---|---|---|---|
| (1) Each comparator is an Analysis-side reading of the extract or the trajectory; writes nothing upstream; reads no interpretation | Holds | 1.1 lines 225-239; 1.2 lines 250-279; 1.3 lines 283-330; 1.4 lines 343-410; 1.6 lines 477-507 | nothing; one mark on the Law pass (a second extract, not a reading of the first) |
| (2) Every comparator grid declared before the data, every point computed and kept, the default marked, none selected against labels | Holds | 1.1 lines 208-221; 1.3 lines 298-327; 1.4 lines 353-406; 1.8 lines 623-626; 1.9; Part 4 items 3, 6 | two precisions so that the record follows the value: the `default_source` / `grid_source` strings and the "declared default" suffix must be conditional on the value actually run; a default off the grid must be refused or appended and recorded (section 2) |
| (3) The C1 floor reads a statistic of the control cells (a quantile of K over idle cells), not a model output, a score or a verdict; the interim is recorded as a pre-registered change with its source | Holds | B1 lines 783-834; Part 4 item 14 | nothing in the rule; the interim's two values (200 in AA T1, 262 in the 2026-09-17 bullet) are the author's to reconcile (section 6 item 1) |
| (4) Nothing in the fix list moves a threshold or a verdict string away from its definition | Holds for the thresholds and the verdict strings; **fails the epoch's own regression rule at one point**: B1's function-level default (line 797) breaks an existing test, verified numerically | B1 line 797 against Part 2 preamble lines 779-781 and rule 8; B17 line 991 against 0.2 line 57-58; B4 line 870 against 3.3 line 1145 | keep `gate_preconditions`' function default at `C1_ACTIVITY_MIN` and carry the T1 rule on the CLI default, the pattern B3 already uses (section 4) |

Two further corrections to the builders' instructions, neither a condition of this certification but
each a test that cannot pass as written: builder A's test 4 (line 715) asks for a `weakref.WeakSet`
of page arrays, and a `numpy.ndarray` is unhashable, so a `WeakSet` refuses it (verified on this
machine); builder B's B12 (lines 946-953) passes `77.28` through `extract_cell`, whose sidecar
writes `int(duration_s)` at `extract.py` lines 450 and 465, so the declared duration would be
recorded as `77`. Section 5 gives both fixes.

---

## 1. Condition (1): an Analysis-side reading that writes nothing upstream and reads no interpretation

**The Form.** A comparator is a Representation of the same delta under a published rule, submitted
to the same gate chain as the ladder's rungs. It is located after the extract (or, where the
published rule needs page identity across more than two snapshots, after the trajectory) and before
the split stage. It emits a per-cell statistic and a feature file; it reads the extract's columns,
the author's declared inputs, and the admissibility record; it reads no score, prediction, fitted
parameter or gate verdict, and it writes into no artifact that any rung stage or the extractor
reads. A realization satisfies this invariant when every write of the comparator lands under its
own directory or its own feature file (or, for a shared gate file, under a key no rung loop
enumerates), and when the per-cell statistic can be recomputed from `extract.csv` (or the trajectory)
and `inputs/head_drop.csv` alone. A realization violates it when a comparator statistic reads
`scores.json`, `gk0.csv`'s measured archetype, or a selection; when it writes to `cells.csv`,
`extract/`, `inputs/`, or the retention root; or when a rung stage picks the comparator up by
iterating a list it was appended to.

**Where it lives in the addendum, and whether it holds.**

- Savoldi (1.2, lines 250-279): reads `ex["K"]` after the head drop and `series.k_median_cell`;
  writes `gates/comparators/savoldi*.csv`, `features/cmp_savoldi/Wall_Hall_{raw,norm}.npz`. Holds.
- Dhodapkar-Smith (1.3, lines 283-330): `delta = 1 - ex["J"]`, "never a second intersection"
  (line 288); the derivation from `n_union` and `n_persist` is the extract's own columns (SPEC 2.2).
  Writes under `gates/comparators/` and `features/cmp_dhodapkar/`. Holds.
- Law (1.4, lines 343-410): reads the trajectory through `extract.open_text`, with the row loop
  copied from `extract._stream`; writes `law_cells.json`, `law_series/`, `law*.csv` under
  `gates/comparators/` and `features/cmp_law/`. `extract.py` is not edited. Holds, with a mark:
  this is not a reading of the extract but a second extract (a per-page run-length representation
  the first extract does not carry). It is a legitimate Representation-side channel; the addendum
  binds it to the first extract through `check_x2_equals_K` (line 374), which is the right
  consistency test (at X = 2 under `dumps` the dynamic count is K by identity), and it refuses a
  non-monotone `seq` with the extractor's own string. The `law_series/` files are its extract and
  should be understood as such by the author.
- The `admissible` and `excluded_pair_rung` columns (lines 227-231) read `gates/preconditions.csv`
  `all_hard_pass` and `preconditions.json` `excluded_cells_pair_rungs`. These are the validity
  record of Acquisition (C1, C2, C6 and the failed count), not an analytical interpretation; they
  annotate the per-cell rows and filter the per-kernel summaries only. The per-cell statistic is
  computed for every `ok` cell with an extract (line 225), and admissibility is applied at read time
  by `models.prepare_split_data` (`models.py` 249-253), exactly as for a rung. Holds.
- The feature files store `archetype_predicted` from `cells.csv` and `IDLE` / `idle` for idle rows
  (lines 237-239, 481-483); G-K0's relabelling stays at read time (`models.py` 254-257). Holds.
- The one shared-file write the comparators make outside their directory is G-X into `gates/gx.csv`
  and `gates/gx.json` (line 450-453), through `gates_comparison.gate_gx`'s per-rung replacement
  (`gates_comparison.py` 225, 255-256), keyed `cmp_<name>`. Every consumer of `gx.csv` is a
  per-rung lookup (`tables.py` 287-309); no loop over `_report_common.RUNGS` or `series.RUNGS`
  enumerates it; `RUNGS` is unchanged (line 150-151). The split stage's `excluded_rows.csv` append
  (`models.py` 519) has no consumer outside one test. Holds.
- The only line of `series.py` builder A touches is `PAIR_RUNGS` (line 155-160). `admissible_cells`
  tests `rung in PAIR_RUNGS` (`series.py` 536), so the two pair-reading comparators inherit the
  failed-verdict exclusion and Savoldi does not; `rung=None` callers (G-K0) are unaffected. Holds.

**Bars.** Instantiable: the split stage already reads a whole-cell feature file with these keys
(`series.build_features` at `W = None`). Bite: a builder who computed Savoldi's per-kernel summary
over `gates/splits/apf/.../predictions.csv` to "align with Table 6", or who wrote the D-S default
into `inputs/`, would violate it. Neither is in the addendum.

**Verdict.** Holds.

---

## 2. Condition (2): the grids declared before the data, every point kept, the default marked, none selected against labels

**The Form.** A comparator's free parameter is a declared axis, realized in full before any label is
read; the point that enters the model is marked among its siblings, which remain on disk; the
choice of that point is a declaration, never a function of a score. A realization satisfies this
invariant when the grid is a module constant (or a CLI value recorded in `params`), the sweep file
holds every point unfiltered, exactly one point per cell carries `is_default = true` and it equals
`params.default`, the feature file is built at that point only, and no code path that reads
`scores.json` or `predictions.csv` writes a default. A realization violates it when a threshold is
chosen by the LOKO score, when the sweep is trimmed to the points that separate, when the default
is silently absent from the grid, or when the record names a source for a value that source did
not declare.

**Where it lives in the addendum.**

- `DHODAPKAR_GRID`, `DHODAPKAR_DEFAULT`, `LAW_X_GRID`, `LAW_X_DEFAULT` are module constants
  (lines 211-216), every one written into `params`; the sweep CSVs are "written whole and never
  filtered" (line 319) and `law_series/` keeps every X per pair (line 403-404); `is_default` marks
  one point (lines 304, 379); the feature vectors are taken "at the default threshold" and "at the
  default X" only (lines 305, 381); Table 7 prints the value with "declared default" (lines
  623-626); the figure shows the whole sweep per kernel with the default as a dotted line (1.9).
  Nothing in Part 1 reads a score before writing a default. C14's instruction ("declare delta_th
  before the labels or sweep it and show the sweep") is met on both branches. Holds.
- Savoldi has no grid; its two open choices (`rows`, `ddof`) are parameters with recorded defaults
  (Part 4 items 1, 2). Holds.
- `TABLE7_VARIANT = "both"` prints raw and normalized rows for every comparator (Part 4 item 11), so
  no variant is selected by its number. Holds.

**Two precisions required before the build (the invariant's record, not its mechanism).**

(a) `default_source = "AA 2026-09-17: 0.04 marked as the default"` and `grid_source = "SPEC_epoch2
Part 1.3: ..."` (lines 326-327) are fixed strings, and the Table 7 suffix "declared default"
(lines 624-626) is fixed text, while `--delta-th-default`, `--x-default`, `--grid` and `--x-grid`
(lines 516-521) and the driver's `--delta-th-default` / `--law-x-default` (line 547-549) let any
value run. A run at `--delta-th-default 0.3` would record that AA 2026-09-17 declared 0.3 and
Table 7 would print "0.3, declared default". The record must follow the value: `default_source`
reads the AA citation only when `default == DHODAPKAR_DEFAULT`, else
`"CLI --delta-th-default <v> (departs from AA 2026-09-17's 0.04)"`; `grid_source` likewise
(`"module constant DHODAPKAR_GRID"` or `"CLI --grid"`); for Law, `x_default_source` is
`"SPEC_epoch2 Part 4 item 6 (the author has declared no X)"` at 4 and the CLI form otherwise; and
the Table 7 suffix prints `declared default` only under the module value, else `default set by
--delta-th-default`. Same for `X`.

(b) A default that is not a grid point (for example `--grid 0.1,...,0.9` with the default 0.04) has
no computed sweep row and no `is_default = true`; the feature vector "at the default" is then
undefined. Part 4 item 3 says the intent ("the author's default is a computed point"); the code
must enforce it: refuse with exit 2 (`missing input: delta_th_default <v> is not on the grid`) or
append the default to the grid and record `default_appended_to_grid = true`. I prefer the refusal;
the author decides (section 6 item 5).

**One hazard outside the addendum, sharper now.** The driver's skip rule compares declared inputs
only (`run_moves.py` 314-320, 401-407); the command's arguments are not part of the key
(`_cmd_key`, 273-274). A resume with a changed `--delta-th-default` (or any of the epoch's eight new
flags: `--law-x-default`, `--savoldi-rows`, `--c1-activity-min-pages`, `--c1-floor`, `--wapf-norm`,
`--duration-s`, `--gl2-rerun`) skips the step as "outputs exist and inputs unchanged" and the
result files keep the old value, honestly recorded in their own `params`, while the ledger's run
record carries the new flag. Pre-existing (the same is true of `--null-perm` today); section 6
item 3 gives the one-line form.

**Bars.** Instantiable: the sweep is a per-cell loop over a tuple. Bite: `max(stability by
delta_th)` per kernel written back as the default, or a `--grid` trimmed after the figure was seen,
would violate it; the addendum builds neither, and after (a) the record would show either.

**Verdict.** Holds, with (a) and (b) applied to Part 1.3, 1.4 and 1.8 before builder A starts.

---

## 3. Condition (3): the C1 floor reads a control-cell statistic, and the interim is a recorded pre-registered change

**The Form.** A condition of valid observation is defined against the instrument's own floor, not
against a number inherited from another corpus, and it reads only what Acquisition and
Representation have recorded: counts, integrity fields, and the control cells' distribution of the
same count. A realization satisfies this invariant when the edge is a declared quantile of K over
control cells whose own admissibility is decided by counts and integrity alone (C2, C6), when the
kernel cell's side of the comparison is a recorded extract field (the undropped `K_max`), when the
rule in force is written per run with its source, and when no kernel-cell verdict feeds back into
the edge. A realization violates it when the edge reads a score, a G-K0 verdict, or a selection;
when the idle set for the edge is chosen by a kernel-side outcome; or when the interim is applied
without its source in the record.

**Where it lives in the addendum, and in the code.**

- The edge: `idle_k_edge(out, idle_cells, *, idle_pool, idle_percentile)` (lines 793-796), "exactly
  the code now inline in `gate_gk0`", which I confirm is `numpy.percentile` over the pooled
  undropped `K` of the idle cells' `extract.csv` (`gates_precondition.py` 257-263) at
  `GK0_IDLE_PERCENTILE = 95.0` (line 40). That is P2 Sec. V's "idle floor's 95th percentile of K"
  and AA T1's "the same floor G-K0 uses". One function, one number, two files (test (a), line 829).
  Holds.
- The idle set for the edge: `cells.csv` status `ok`, a sidecar, `C2 == pass`, `C6 == pass`
  (lines 803-806). C2 is `n_pairs >= 8`, C6 is the extractor's status, header width, modal header
  hash and zero skipped rows (`gates_precondition.py` 151-172): counts and integrity, no
  interpretation. It is the same set `gate_gk0` uses through `admissible_cells(out, cells, None)`
  once `preconditions.csv` exists (an idle cell's `all_hard_pass` is C2 and C6, line 182). Holds.
- The kernel side: `sidecar["K_max"]`, the undropped maximum (`extract.py` 531); strict `>` for
  the floor (T1's "exceeds") and `>=` for the interim (T1's `K_max >= 200`) (Part 4 item 14, lines
  1220-1222). Both undropped, so the two sides are commensurate. Holds.
- The record: `C1_rule` per row (`"floor: K_max > <edge> (idle p95 of K over <n> idle cells; AA T1;
  the G-K0 edge)"`, `"interim: K_max >= <n> pages (AA T1; no admissible idle cell)"`,
  `"inherited: ... (superseded by AA T1)"`), `C1_threshold_pages`, and in `preconditions.json`
  `params` `C1_rule_in_force`, `C1_floor_edge`, `C1_floor_idle_cells`, `C1_activity_min_pages`
  (lines 806-814); Table 4's Plan 02 row gains `C1 rule: <rule>` (lines 820-823); the runbook says
  which rule was in force (line 824-825, T1's own sentence). The interim's source is AA T1 in the
  string itself. Holds.
- Direction of dependence: control cells decide the edge; kernel cells are judged against it; the
  idle cells' C1 stays the not-applicable string. No verdict of a kernel cell enters the edge. Holds.
- The staleness side is already correct for the admissible set (my probe D: a preconditions re-run
  marks every step from move 3 stale). One small gap: `gate_preconditions` hashes the sidecars into
  `params.inputs_sha256` (`gates_precondition.py` 146, 198) but under B1 it also reads the idle
  cells' `extract.csv`; those paths should be appended to the hashed `inputs` list so the record
  names everything the edge was computed from. A record, not a trigger.

**Bars.** Instantiable: `gate_gk0` already computes this number on every run with idle cells. Bite:
an edge taken from `gk0.csv`'s `idle_band_edge` column (a file written by a later move) or an idle
set filtered by `GK0_IDLE_MEASURED` would violate it; the addendum does neither.

**One bite the author should see** (section 6 item 2): the rule switches by itself when the idle
cells arrive (`c1_floor` on by default). A kernel admitted under the interim (gibbs at 256 pages,
T1's own example) is refused under the floor if the idle p95 on the real corpus exceeds its
footprint. That is the rule as declared, recorded per run; it is also every step from move 3 going
stale on that day, at the cost of the whole chain.

**Verdict.** Holds.

---

## 4. Condition (4): nothing in the fix list moves a threshold or a verdict string away from its definition

I went through B1 to B31 item by item against SPEC 3.0's vocabulary (`verdicts.py`), SPEC 3.5.7's
selection rule, P2 Sec. V 5.2's "every threshold is declared now and does not move", and the tests.

**Thresholds.**

- B1 moves C1's threshold, from `apf_max >= 0.02` to the floor / the interim 200 pages. This is the
  author's re-declaration (AA T1: "a change to a pre-registered gate", the same class as the Plan 08
  restatements P2 Sec. V records), and the old rule stays reachable and labelled `superseded`. Not a
  move away from the definition; a move to the definition's new text. Holds.
- B3 applies B1-G1's floor of 500 permutations to G-F (i), G-X and the clustering, in the `not
  run: <n> permutations < 500` form B1-G1 writes, numbers kept in their columns (lines 855-858).
  P2 Sec. V gives G-X's null as "the unit-level null" and G-F (i) as "inside the shuffle null", the
  null whose floor Plan 08 states as "at least 500 permutations"; the fix applies the null's own
  floor to the gates that cite it. The function-level default 0 keeps the existing direct-call
  contracts. Holds.
- B7's default `drop` changes `passes_acceptance` in one state (no kernel applicable for G1): SPEC
  3.5.7 says "every applicable gate among G1, G2, G4", and `_rollup` already returns `not
  applicable: no applicable kernel` for G1 (`gates_temporal.py` 543); counting that as a failure
  (597-600) was CHECK_3 M7's finding. `drop` follows the sentence; `refuse` is exposed. Holds, with
  the bite named in section 6 item 6: on the real corpus G2 is not applicable for eleven kernels, so
  a rung with every kernel under `TREND_PRESENT` would be selected by G4 alone and pass acceptance.
- B12: on the real corpus every sidecar declares 600 and G2 and G-P read the sidecar, so no number
  moves (line 1237-1238); on the synthetic corpus G2 becomes decidable, which is the fix. Holds.
- B4, B5, B6, B8 to B11, B13 to B16, B18 to B30: no threshold. Holds.

**Verdict strings.**

- 0.2 line 57-58: "Verdict strings never change ... the new strings are `not applicable: ...` /
  `not run: ...` forms." B16's `not run: grid incomplete (<n> points missing)` and B7's `not run:
  no applicable kernel for G1` are in that form. B17's `refusal = "acceptance failed: <gate>:
  <verdict>, ..."` (line 991) is not: it is neither a `not run:`, `not applicable:` or `refused:`
  string nor a named refusal, so `verdicts.is_refusal` is false on it. No consumer keys on
  `is_refusal(entry["refusal"])` (Table 5 prints the field as text, `tables.py` 180-185;
  `resolution_text` reads `selected_by` and `passes_acceptance`, `_report_common.py` 504), so
  nothing misprints; but it is a fourth form the addendum's own rule does not admit. Section 6 item
  7 gives the two ways out; the author chooses.
- B24 makes the feature file say `IDLE` / `idle`, SPEC 3.1.5's own text. B15 removes a column SPEC
  3.5.7 never listed. B5 adds a key. Holds.

**The regression rule (rule 8; Part 2 preamble lines 779-781).** B1 line 797 changes
`gate_preconditions`' function default from `c1_activity_min: float = C1_ACTIVITY_MIN` (0.02;
`gates_precondition.py` 106) to `float | None = None`, and under `None` the floor or the interim
runs. Seven existing tests call `gate_preconditions(out)` with no argument
(`tests/test_gates_precondition.py` 12, 28, 45, 53, 66, 72, 77). Six assert C2, C3, C6 or the
failed count on cells at `K0 = 6000` and still pass. The first,
`test_c1_pass_refuse_and_idle_not_applicable` (lines 8-18), asserts that gibbs at `K0 = 100` fails
C1 and is the only excluded cell. I wrote that test's corpus with the in-test generator and read
the sidecars: gibbs `K_max = 252` (100 plus the 150-page floor set), the idle cell's pooled p95 of
K is `150.0`, and the interim is 200. Under the floor gibbs passes (252 > 150); under the interim
gibbs passes (252 >= 200); under the inherited rule it fails (0.00096 < 0.02). The existing test
therefore fails under B1 as written, and 0.2 forbids amending it (the two `test_driver.py` tokens are
the only permitted amendment). The addendum's sentence "kept so that the existing tests that pass
`c1_activity_min=0.0` and `0.0005` run unchanged" (line 801-802) covers the other three files but
not this one.

The fix is the pattern the addendum itself prescribes for B3 (function 0, CLI 500) and in its
preamble ("the CLI carries the new default where the two differ"): the function keeps
`c1_activity_min: float | None = C1_ACTIVITY_MIN`, `None` means "the T1 rule", the CLI flag
`--c1-activity-min` defaults to `None`, the driver passes nothing unless set, and the docstring
states the two defaults and why. B1's tests (a) to (d) then call with `c1_activity_min=None`
explicitly, and test (d) is the existing tests. Exact text in section 5.

**Verdict.** Holds for every threshold and every verdict string; fails rule 8 at B1 line 797 until
the function default is restored as above.

---

## 5. Corrections to the builders' instructions (exact forms; the author approves, the builders apply)

1. **B1, line 797 (builder B).** Replace `c1_activity_min: float | None = None` with
   `c1_activity_min: float | None = C1_ACTIVITY_MIN` and add, after the order of rules, the sentence:
   "The function default is the inherited 0.02 so that every existing direct call keeps its
   contract; the CLI's `--c1-activity-min` defaults to `None`, which selects the T1 rule, and the
   driver passes the flag only when set, so the paper's path runs T1. The docstring states both."
   Test (d) at line 831-832 becomes "the existing tests are this case, by the function default".
2. **1.10 test 4, line 713-720 (builder A).** A `numpy.ndarray` cannot be a `WeakSet` member
   (unhashable; `TypeError: cannot use 'weakref.ReferenceType' as a set element`, verified on this
   machine). `law_stream` should wrap each snapshot's page array in a small class (as
   `extract.Snapshot` is, `extract.py` 143-153) and track those in `_LIVE_PAGE_ARRAYS`; the hook
   receives the wrapper's array. The assertion `len(_LIVE_PAGE_ARRAYS) <= 1` is unchanged in
   meaning.
3. **B12 (ii), lines 948-950 (builder B).** `extract_cell` writes `"duration_s": int(duration_s)`
   and `"duration_s_declared": int(duration_s)` (`extract.py` 450, 465); with `77.28` the sidecar
   would say `77` and G2's per-kernel `T` would be read against 77, not 77.28. Both casts become
   `float(duration_s)`; the parameter annotation becomes `float`; on the real corpus `600` prints
   as `600.0`, and `tests/test_extract.py`'s equality with 600 still holds.
4. **1.3 lines 326-327, 1.4 line 397-398, 1.8 lines 623-626 (builder A).** The conditional
   `default_source`, `grid_source`, `x_default_source` and the Table 7 suffix of section 2 (a); the
   refusal (or recorded append) of a default off the grid of section 2 (b).
5. **1.5 line 432-437 (builder A), a string.** For the within-trace row the comparator's
   `scores.json` carries `not applicable: one window per cell` and `gm_compare` returns `not run: a
   score or the spread is missing` (`gates_comparison.py` 322-323). The comparator's `gm.csv` row
   should carry the split's own `not applicable:` string when the comparator's `accuracy` is a
   `not applicable` string, so that the Table 7 cell says why rather than that something is
   missing.
6. **1.8 line 611-613 (builder A), a count.** Table 7's `feature count` is filled on two lines,
   `tables.py` 542 (the `not applicable` branch) and 544 (the normal branch), not one; the
   `feature_count_used` fallback applies to both.
7. **B1 (builder B), a record.** Append the idle cells' `extract.csv` paths read by `idle_k_edge`
   to the `inputs` list hashed into `preconditions.json` `params.inputs_sha256` (section 3).

---

## 6. For the author

Each is a choice the definitions leave open, or a hazard this review surfaced; I decide none.

1. **The interim value: 200 or 262.** AA T1 says 200 pages (`K_max >= 200`); the 2026-09-17 bullet
   says 0.1 percent of memory, 262 pages. The addendum takes 200 and cites T1 (Part 4 item 14), and
   `--c1-activity-min-pages 262` is one flag away. The two records are yours and disagree; say which
   stands, in `P2_AUTHOR_ANSWERS.md`, before the run, so that the `C1_rule` string cites one value.
2. **The rule switches by itself when the idle cells arrive.** Under `c1_floor = on` the first run
   with admissible idle cells re-decides every kernel cell's C1 against the measured p95 and marks
   every step from move 3 stale. A kernel admitted under the interim can be refused under the floor
   (gibbs at 256 pages if the idle p95 is above it). This is T1 as written; know the day it happens.
3. **The driver ignores a changed argument on resume** (section 2). Either a runbook sentence ("a
   changed flag needs `--force` on the steps it feeds") or one addition to `_stale_reason`
   (`run_moves.py` 314-320): compare `prev.get("argv")` with the current `argv` after dropping
   `--n-jobs`, `--jobs` and their values, and return `stale: arguments changed` on a difference. The
   second is builder B's region and about eight lines; it is not in the addendum.
4. **G-C on the comparators: `not applicable` or inherited?** The addendum prints `not applicable:
   comparator, not a lead of the ladder` (line 457). Savoldi reads the same `K` that G-C calibrates
   for APF (gemm's two-to-one jump), and Dhodapkar-Smith's `delta` is `1 - J`, the quantity G-C
   calibrates for persistence (the J dip toward one half). A comparator that fails to see the pulse
   is a broken baseline, and a broken baseline that loses flatters APF. Before:
   `not applicable: comparator, not a lead of the ladder (G-C calibrates the ladder's leads against
   the gemm pulse)`. After, for Savoldi: `inherited: G-C of apf (the same K; see gates/gc.csv apf,
   rep = all)`; for Dhodapkar-Smith: `inherited: G-C of persist (delta = 1 - J; see gates/gc.csv
   persist, rep = all)`; for Law: `not applicable: no pulse prediction for the dynamic-for-X count`.
   A string change in `table7_comparators` only; no new gate runs. Your call; the addendum's
   reading ("its connection to the pulse is not a claim the paper makes") is defensible.
5. **A default off the grid: refuse or append** (section 2 (b)). I prefer the refusal.
6. **B7 `drop` on the real corpus.** With eleven kernels undeclared for G2, a rung whose every
   kernel reads `TREND_PRESENT` under G1 is selected by G4 alone and passes acceptance as `1 of 1`.
   If a resolution accepted on the window-count rule alone is not what you want printed as
   `selected`, choose `--g1-none-applicable refuse` (Part 4 item 17).
7. **B17's string form.** Either admit `acceptance failed: ...` as a fourth form in 0.2 (it is
   descriptive and no consumer misreads it), or write `refused: acceptance failed: G2: fail, ...`
   so that `is_refusal` recognizes it; the second makes a best-feasible selection read as a
   refusal in Table 5's `refusal` column while its numbers still print. I lean to the first.
8. **B4 against 3.3.** B4 adds `split` and `labelspace` columns to `gates/gdim.csv` (line 870-871);
   3.3 freezes that file's columns for builder A (line 1145). Harmless (every reader is by column
   name, and builder A's `gates/comparators/gdim.csv` keeps the eight columns), but say in 3.3 that
   appended columns are allowed so the two builders do not each report the other as deviating.
9. **The Dhodapkar-Smith normalized row has one degree of freedom.** `boundary_rate = 1 -
   stability` identically and `mean_phase_frac = 1 / (B + 1)` under the default phase-length rule;
   B1-G3 will most likely quarantine two of the three and the row's score will be a one-feature
   re-run (Part 4 item 23 anticipates this). The raw row keeps `n_pairs_used` as a second degree.
   Expect it; it is the method's own redundancy, recorded in `feature_note`.
10. **Law's `X = 4` is the addendum's choice, not yours** (Part 4 item 6: "the definition names no
    X"). Declare it or accept it before the run; after section 5 item 4 the record will say which.
11. **Law's cost.** The pass re-streams every trajectory (one to three minutes per cell per process
    by the runbook's extrapolation, never measured on the real corpus); at `--comparator-jobs 4`
    on 96 cells, roughly half an hour to an hour and a quarter, resumable per cell.

## 7. Server

Nothing in this review needs the server, and nothing was run there. When the two builders finish,
the full suite from `plan11_encoding_ladder/` is `python3 -m pytest -q tests`; the baseline to
beat is 158 passed, 1 skipped, and the one existing test that section 4 names must be among the
passes without amendment.

Scratch artifact of this review: the synthetic corpus of section 4 under
`/private/tmp/claude-501/-Users-jeries-Desktop-projects-thesis-memorySignal-mem-sig/14810c8c-2535-466d-a296-d2aae9739c16/scratchpad/rev_e2/`.
This file:
`/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/SPEC_epoch2_review_al_farabi.md`.
