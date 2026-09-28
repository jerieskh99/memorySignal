# BUILD_extract.md: builder 1's report (the streaming extractor and the synthetic generator)

Written 2026-09-16. Builder 1 of paper 2's analysis toolkit, per `SPEC.md` section 1 (the
`__init__.py`, `schema.py`, `extract.py`, `synth.py` rows and their three test files). No server
was touched, no path under a server mount appears in any file, and the sandbox family is not
named anywhere. Nothing was committed to git. No file outside `plan11_encoding_ladder/` was
edited, and no file owned by builders 2 or 3 was edited (they were writing into the same
directory in parallel; their `_schema_compat.py`, gate modules and tests are theirs).

## 1. Files written

All under `/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/`:

| File | Lines | What it is |
|---|---|---|
| `__init__.py` | 1 | `__version__ = "0.1.0"` only. |
| `schema.py` | 267 | The shared constants (`N_PAGES`, `PAGE_SIZE`, `BITS_PER_PAGE`, `DURATION_S`, `DT_BRACKET_S`, `QUANTILES`, `SIDECAR_SCHEMA`, the 66-column trajectory header as read from the differ's `metrics/mod.rs`), `EXTRACT_COLUMNS` (58, SPEC 2.2 order), `KERNELS`, `ARCHETYPES`, `LEVEL_MATCHED_SETS`, the grid (`GRID_WINDOWS`, `GRID_HOP_RATIOS`, `grid_id`, `grid_points`, `hop_of`), and cell identity (`campaign_of`, `kernel_of_test_label`, `parse_cell_path`, `cell_id_of`, `CELLS_COLUMNS`, the four `cells.csv` status strings). |
| `extract.py` | 880 | The streaming per-cell extractor (SPEC 2.1 to 2.5), the cell index (SPEC 2.6, 2.7), the batch mode, and the CLI with the three subcommands `index`, `cell`, `all`. |
| `synth.py` | 811 | The synthetic trajectory generator and corpus (SPEC section 5), with `SynthSpec`, `write_cell`, `corpus_specs`, `write_corpus`, and the CLI with `cell` and `corpus`. |
| `tests/test_schema.py` | 183 | 13 tests of `schema.py`. |
| `tests/test_synth.py` | 424 | 23 tests of `synth.py`. |
| `tests/test_extract.py` | 699 | 26 tests of `extract.py` against the generator's known answers. |
| `BUILD_extract.md` | | This report. |

Every public function's docstring cites the definition it implements (P2 section, CR item, K2
move, or SPEC section), and every result file (`sidecar.json`, `cells.index.json`,
`extract/extract_all.json`, `truth.json`, `corpus.json`) carries `schema`, `params` and
`citation`.

## 2. How to run

From `/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU/`
(the modules also run by absolute path, `python3 plan11_encoding_ladder/extract.py ...`):

```
# the synthetic corpus (SPEC 5.2), 104 cells; --jobs 4 takes about 160 s on this machine, 235 s single
python3 -m plan11_encoding_ladder.synth corpus --root /tmp/p11/root --jobs 4
# one synthetic cell with every knob (SPEC 5.1)
python3 -m plan11_encoding_ladder.synth cell --root /tmp/p11/root --name gemm --seed 42 --n-pairs 120 \
    --k0 4096 --pulse-period 24 --pulse-extra 4096 --churn 0.02 --content double --gap-seqs 5,9

# move 0: the cell index (SPEC 2.7)
python3 -m plan11_encoding_ladder.extract index --root <retention root> --out <out>
# move 1: every ok cell of cells.csv, four processes (SPEC 2.5); a finished cell is skipped unless --force
python3 -m plan11_encoding_ladder.extract all --cells-csv <out>/cells.csv --out <out> --jobs 4
# one cell by hand (a directory or the trajectory file itself)
python3 -m plan11_encoding_ladder.extract cell --cell-dir <cell dir> --out <out> [--persist-side t|t+1] \
    [--failed-count N | --failed-dir D] [--role kernel|idle] [--cell-id ID] [--rep R] [--force]

# builder 1's tests (62 collected; one skips when the zstandard module is absent)
python3 -m pytest -q plan11_encoding_ladder/tests/test_schema.py plan11_encoding_ladder/tests/test_synth.py \
    plan11_encoding_ladder/tests/test_extract.py
# without pytest
python3 -m unittest discover -s plan11_encoding_ladder/tests -p 'test_schema.py'   (and test_synth.py, test_extract.py)
```

`python3 -m pytest -q plan11_encoding_ladder/tests` runs every builder's tests together; the
numbers below are for builder 1's three files alone.

What the server needs for builder 1's part: Python 3.10 or later, `numpy` (any version from 1.21;
tested with 2.2.6), and one of the `zstd` binary or the `zstandard` module to read `.csv.zst`
(the binary is tried first, the module second, plain `.csv` and `.csv.gz` always work). The
`pip --user` fallback is `python3 -m pip install --user numpy zstandard`. `scikit-learn`, `scipy`
and `matplotlib` are not used by builder 1's modules.

Measured throughput on this machine (8 cores, Python 3.10.12, numpy 2.2.6, zstd 1.5.7):

- `extract cell` on a synthetic 931-pair cell of 4,202,859 rows (85 MB `.csv.zst`): 25 to 27 s,
  peak resident set 36 MB (the two-snapshot bound holds; nothing is loaded whole). The 931 rows
  matched the generator's truth on every column. Real rows are longer (66 populated columns, many
  floats), so the `csv.reader` time will be higher on the server: the SPEC's "one to three
  minutes per cell" stands as the runbook figure, and `--jobs 4` divides it.
- `extract all --jobs 4` on the 104-cell synthetic corpus (32,965,965 rows): 65 s.
- `synth corpus` (104 cells, 618 MB compressed): 235 s single process, 160 s with `--jobs 4`;
  cells are byte-identical across job counts (checked, 104 of 104).

## 3. Test results, verbatim

```
$ python3 -m pytest -q plan11_encoding_ladder/tests/test_schema.py plan11_encoding_ladder/tests/test_synth.py plan11_encoding_ladder/tests/test_extract.py -rs
.........................................................s....           [100%]
=========================== short test summary info ============================
SKIPPED [1] plan11_encoding_ladder/tests/test_extract.py:515: zstandard module not installed
61 passed, 1 skipped in 15.61s
```

```
$ python3 -m unittest discover -s plan11_encoding_ladder/tests -p 'test_s*.py'
OK (skipped=1)
```

What the tests cover (a gate is not done without a passing and a refusing case; builder 1 has no
gate, but every refusal path of the extractor has its case):

- `test_extract.py`: `extract.csv` equals `truth.per_seq_channels` on every one of the 58 columns
  and every row (to 1e-9 relative; the CSV carries 10 significant digits) for a re-seed pulse
  cell with three gaps and a floor set (compressed), a persistent-set cell (churn 0: J = 1,
  `n_persist = K`), a floor-like idle cell (K about 150, `apf_max` below C1's 0.02), a
  level-matched pair (the same `K_median`, `r_l1l0` about 86 against about 30), and a cell
  written with shuffled rows and `seq_first = 0`. The `t+1` persist side against
  `per_seq_channels_t1`. J, `J_null_inter`, `J_null`, `n_persist`, `n_union` and the fifteen
  ratio quantiles are in that comparison. The gap semantics (K = 0 row, zero sums, blank
  quantiles, J = 0 against an empty set, blank when both empty) and the last row's blanks are
  asserted explicitly. The sidecar's every field. Refusals: `seq not monotone at row N` (from
  `--corrupt seq_reverse`, and a second case where the decrease sits deep inside a compressed
  file so the reader is stopped mid-stream without masking the reason), `trajectory file count
  != 1` (zero and two files), `header lacks column l0`, `no data rows`, `empty trajectory file`;
  on every refusal the sidecar is written, no `extract.csv` or `.tmp` is left, and the CLI exits
  0. Counters: duplicate page rows (first kept), `hamming = 0` rows (kept in `S_t`, counted),
  `l0 = 0` rows (counted, excluded from the two division ratios, kept in `r_l0`), malformed
  rows (skipped, counted). The reader: byte-for-byte equality of `open_text` with the original
  `plan08_b1/b1_extract_hamming.py:open_text` on `.csv`, `.csv.gz` and `.csv.zst`; a `.csv.gz`
  trajectory end to end; the error when neither `zstd` nor `zstandard` exists names both; the
  `zstandard` path (skipped here, the module is not installed). The index on a small corpus
  (rep 0 = seed 42, reps by ascending seed, cell ids, roles, archetypes, `cells.index.json`),
  its refusals (`duplicate seed` with both kept and distinct ids, `unknown kernel`, `trajectory
  file count != 1`), `--role-overrides` and custom idle markers. Batch mode: `--only`,
  `--failed-counts` into the sidecar, skip on rerun, `--force`, `--jobs 2`, refused `cells.csv`
  rows skipped with the reason recorded. CLI exit codes 2 for every missing input, the module
  and the absolute-path entry points, `cell` skip and `--force`.
- The memory test (SPEC 2.4, instrumented): every `Snapshot` registers in a `weakref.WeakSet`;
  a hook called at every emitted row asserts that at most two snapshots are alive and that
  they are exactly the adjacent pair `(t, t+1)`; the run reaches 2 (so the count is real), the
  last row is emitted against `None` with one alive, one row is emitted per seq including gaps,
  and nothing survives the pass.
- `test_synth.py`: the exact 66-column header; on every row `1 <= l0 <= 4096`, `l1 >= l0`,
  `1 <= hamming <= 8 l0`, `1 <= linf <= 255`, `l2 <= l1`, `mean_abs = l1 / 4096`, `cosine = 0`
  and the other 57 columns 0; rows grouped by non-decreasing seq, pages unique and ascending
  within a seq, gap seqs absent; the pulse (K jumps by `pulse_extra`, J dips to about one half
  on the pair before each boundary, `expected.K_jump_ratio` recorded); the content models'
  ratios (spin `l1 = l0`; counter `l0` in {1, 2} and `l1/l0` about 30; double `l0` a multiple
  of 8, `l1/l0` about 86, `hamming/l0` about 4; idle `l0` in 1..4); decay falling inside each
  pass and resetting at the boundary; trend, step, k-decay and l0-trend; churn 0 giving J = 1
  and churn 0.10 giving mean J near (1 - c)/(1 + c); determinism; the truth's completeness and
  the `sets` rule; the corpus presets, seeds (AA A4) and every case switch; `write_corpus`;
  the CLI flags and both entry points.
- `test_schema.py`: the constants, the 58 columns and their order, the header positions
  (`hamming` 2, `l0` 4, `l1` 5), the twelve kernels and archetype counts (6/3/2/1), the grid
  (13 points, ids, the rejection of off-grid pairs), `campaign_of`, `parse_cell_path` on
  real-shaped paths (a `dwarfs1_resume` stencil path with a hash suffix, an idle path without a
  seed, an unknown kernel, custom markers), `cell_id_of`, `format_value`.

## 4. Deviations from SPEC.md, and why

Corrections from the two reviews that touch builder 1's files (implemented as the reviews say,
not as the SPEC line):

1. al-Kindi 2.4 (d): the synthetic floyd preset has `pulse_extra = 2048` (the SPEC's preset had
   no K jump, so G-DEC's `k_jump` boundary source could never fire on its must-pass case).
   `synth.PRESETS["floyd"]` and `corpus.json` say so.
2. al-Farabi 2.8: every result file's `params` carries `inputs_sha256`. The sidecar records the
   trajectory file's sha256 (also at top level as `source_sha256`, a chunked read of the
   compressed bytes, about a second per real cell) and, in batch mode, the sha256 of `cells.csv`
   and of `--failed-counts`; `cells.index.json` records `--role-overrides`.
3. al-Kindi 2.3 (b) and "for the author" 7: no change in the generator is needed; the G3 must-pass
   cell is `synth.py cell ... --pulse-period 24 --pulse-extra 4096 --n-pairs 240`.

Choices the SPEC left to the builder, resolved without adding a threshold:

4. `sidecar.json` holds every key of SPEC 2.3 and, in addition, `params`, `citation`,
   `source_sha256`, `rep_source` and `extractor_version`'s companions, because SPEC 7.1 says
   every command writes its `params` block and every JSON result file carries `schema`,
   `params`, `citation`. Three files are added to the fixed layout for the same reason:
   `<out>/cells.index.json` (the `index` command's params and status counts),
   `<out>/extract/extract_all.json` (the `all` command's params, per-cell statuses and skips),
   and `<root>/corpus.json` (the corpus's params and every cell's spec).
5. SPEC 2.7 says both "one row per directory that holds exactly one trajectory file" and a
   status `refused: trajectory file count != 1`. `build_index` lists every `rep*__*` directory
   and writes the count refusal for the ones with zero or several trajectories (the eight
   stencil directories of AA S3 are that case), so the author sees them rather than losing
   them. Such a row still gets a rep index and a cell id, so the id is stable once the file is
   filed.
6. `extract_cell` takes four keyword arguments beyond SPEC 2.5's signature: `rep` (the paper's
   rep index from `cells.csv`; without it the sidecar uses `rep_dir - 1` and says so in
   `rep_source`), `archetype_predicted` (so an archetype the author edited in `cells.csv`
   reaches the sidecar), `failed_count_source` (the text for `--failed-counts` rows),
   `idle_markers`, and `extra_inputs_sha256`. The CLI gains `cell --rep R`. `derive_cell_id`
   lets `cell` honour the skip-unless-`--force` rule without `--cell-id`.
7. `open_text` is "copied and extended" (SPEC 2.5 says extended), so the required equality test
   is behavioural (same lines from `.csv`, `.csv.gz`, `.csv.zst`) rather than source equality.
   One extension beyond the fallback chain: when the caller leaves the block by an exception
   (a refusal mid-file), the `zstd` process is killed and the refusal propagates; the original
   would have raised the decompressor's return code instead of the reason.
8. `schema.grid_id(W, H)` rejects a pair that is not on the declared grid (an `H` that is not a
   declared hop of its `W`, or a non-grid `W` with `H != W`) with `ValueError`, so a mis-typed
   point can never be written under a grid id; it accepts `"whole"`, `None` and
   `(n_series, n_series)` for the whole-cell point. `grid_points(None)` returns `("whole",
   "whole")` as the thirteenth pair (builder 3's `_report_common.py` calls it that way).
9. `SynthSpec` has seven fields beyond SPEC 5.1, all defaults off, each for a case of SPEC 5.3:
   `step_factor` (`--step`: 3 K0 in the second half), `k_decay_factor` (`--k-decay`, default
   0.5 from `--k-decay-factor`: K also shrinks inside a pass, the G-DEC (c) refusal),
   `l0_trend` (`--l0-trend`: the content's l0 scale drifts over the cell, for "idle cells
   generated with trend -0.5 on l0", the G-DEC (e) refusal; the SPEC's `trend` field is K's
   drift and could not produce it), `corrupt` (`--corrupt seq_reverse`: the third and fourth
   snapshot blocks are swapped in the file, the truth unchanged), `rep_dir`, `row_order`
   (`random` exercises the extractor's sort), `pulse_burst` (item 12 below). `synth.py cell`
   gains `--no-truth-sets`; `corpus` gains `--cv-case preset`, `--ord-periods`, `--ord-mode`,
   `--truth-sets` and `--jobs`.
10. Corpus cells carry the `"t"`-side truth only (`per_seq_channels_t1` is null) and no page
    lists (`sets` null unless `--truth-sets`), because the corpus is written for builders 2 and 3
    and the smoke run, which read neither, and the two together cost about a third of the
    generation time and hundreds of megabytes of JSON. `synth.py cell` keeps the SPEC rule
    (`sets` when `n_pairs * K0 <= 2e6`) and both sides.
11. `--cv-case random` (the default) draws `k_noise` for every kernel whose preset does not set
    it explicitly; the SPEC's own preset table sets `bnb_tsp` to 0.4 (its FRONTIER-CHURN
    fluctuating level) and the two rules collide, so the explicit value wins. Under
    `--level-only`, `--one-preset` and `--ord-case` no random draw is made (every kernel keeps
    `k_noise = 0.02`), because a per-kernel noise level would itself carry the label into a
    corpus whose whole point is that only the declared difference does.
12. `--ord-case`: SPEC 5.2 says "two archetypes with identical K0 and content, periods 6 and
    12". Implemented literally as `--ord-mode period` (archetypes at even index in
    `ARCHETYPES[1:]`, WORKING-SET and SEQUENTIAL-GROW, get `--ord-periods[0]` = 6; SCATTER and
    FRONTIER-CHURN get 12; `K0 = 2048`, double content, `pulse_extra = 2048`). Two periods with
    one `pulse_extra` also change the marginal distribution of K (a period-6 cell spends twice
    the fraction of snapshots at the high level), so an order-blind learner can still separate
    the classes by window mean and the verdict may read "order-blind". `--ord-mode burst` is
    provided as the clean order-only fixture: both classes have period 12 and about the same
    pulse count, and the second class fires its pulses in adjacent pairs every 24 snapshots
    (`pulse_burst`), so the marginals match and only the order differs. Section 5 item 8 has
    the simulation.
13. `--one-preset` uses `K0 = 2048`, double, churn 0.02, no pulse (the SPEC names no preset).
    `--level-only` maps `K0` 2048 / 4096 / 8192 / 16384 onto WORKING-SET / SCATTER /
    SEQUENTIAL-GROW / FRONTIER-CHURN in `ARCHETYPES` order (the SPEC gives the four numbers, not
    the mapping).
14. The sidecar's `K_median` is the median over every row from `seq_first` to `seq_last`, gap
    rows (K = 0) included and no head drop applied; builder 2's `K_median_cell` (SPEC 3.1.1,
    after head drop) is recomputed from `extract.csv` and is the one the normalization uses.
15. `r_l1l0_*` and `r_haml0_*` are blank when no persistent row has `l0 >= 1`, even when
    `n_persist > 0` (SPEC 2.2 names the guard but not this corner); `r_l0_*` keeps every
    persistent row. Not expected in sparse mode (a changed page has `l0 >= 1`).
16. Rep seeds in the corpus: rep 0 is 42, rep r is `1000 r + i` with `i` the kernel's index in
    `KERNELS` (idle cells `i = 12`); the SPEC says "a per-kernel base from the kernel name's
    index" and this is the literal reading.

## 5. For the author

Choices the definitions leave open, each a parameter with the default written into `params`,
plus the facts the author should know before move 1:

1. `persist_side = "t"` (SPEC section 8 item 1): the persistent pages' channel values are the
   ones at `seq = t`. `--persist-side t+1` is the alternative; the tests pass on both.
2. `J_null` is the ratio of expectations `J_null_inter / (K_t + K_{t+1} - J_null_inter)` (SPEC
   section 8 item 2); both columns are in the extract.
3. A missing `seq` is a K = 0 snapshot, counted in `n_seq_gaps` with the first 100 in
   `gap_seqs` (SPEC section 8 item 3). Al-Farabi 2.10: confirm every gap seq against the
   consumer log's `substrate seq=<n> rows=0` line, not against `failed/`; a `WARNING: no
   substrate CSV found` line also produces a gap and is not a failed job.
4. The paper's rep index comes from the seed (rep 0 = seed 42, then ascending), and for a cell
   without a parsed seed from `rep_dir - 1` (SPEC section 8 item 4). Two cells of one kernel
   with the same seed are both listed as `refused: duplicate seed` and both keep an id; the
   author edits `cells.csv`.
5. Idle cells are detected by the substrings `sleep`, `idle` in the test label
   (`--idle-marker`, SPEC section 8 item 5); the real idle cells' directory shape is not known
   yet (they are to be captured), so `--role-overrides CSV` (`path, role[,
   archetype_predicted]`) and hand edits of `cells.csv` are the fallbacks. If the idle cells'
   param signature carries no `seed_`, their rep is `rep_dir - 1`.
6. `failed_count` is never assumed: without `--failed-count`, `--failed-dir` or a
   `--failed-counts` CSV the sidecar says `null` and `"not recorded"`, and builder 2's
   `--assume-failed-zero --assume-reason` is the declared path (AA A5).
7. The `seq` origin is not assumed (SPEC section 8 item 40): `seq_first` and `seq_last` are
   recorded and `n_pairs = seq_last - seq_first + 1`; only interior gaps count.
8. `--ord-case` design (section 4 item 12): the literal "periods 6 and 12" fixture may read
   "order-blind" because the two periods also change K's marginal; `--ord-mode burst` differs
   in order only. A rough simulation of G-ORD on the K series (the eight B1 shape features at
   `H = W // 2`, LOKO by kernel in archetype space, a 200-tree forest, cell-majority vote,
   three order shuffles, a twelve-permutation label null) says the period fixture reads
   "order-blind" at every W and the burst fixture "resolution" at every W (section 6). The
   default is still the SPEC's period design; the author decides which one builder 2's
   must-pass test is built on.
9. `--cv-case random` keeps `bnb_tsp`'s explicit `k_noise = 0.4` (section 4 item 11). If the
   G-L (ii) must-pass case needs every kernel drawn from Uniform(0.01, 0.10), pass
   `--cv-case random` after removing the explicit value from the preset, or accept the
   present reading.
10. The three added result files (`cells.index.json`, `extract/extract_all.json`, `corpus.json`)
    and the extra sidecar keys are additions, not changes; builder 3's manifest may hash them.
11. Throughput: about 25 s per 4.2-million-row synthetic cell here with a 36 MB resident set.
    Real rows are two to three times longer, so plan on the SPEC's one to three minutes per
    cell on the server and run move 1 with `--jobs 4`; the runbook's "2 to 5 hours at
    `--jobs 4`" is consistent with that.
12. The `zstandard`-module reading path is untested on this machine (the module is not
    installed; the `zstd` binary is). If the server lacks the binary, install the module
    (`python3 -m pip install --user zstandard`) and run one cell by hand before move 1.
13. The generator's content models honour the differ's identities exactly (every changed byte
    is an actual pair), but they are fixtures, not physics: `counter` bumps wrap modulo 256
    (a few `|a - b|` values near 255), `double` draws two independent bytes, and the floor set
    changes 1 to 4 bytes per page. The G-C orderings of SPEC 5.1 hold on the corpus by
    construction; the corpus's `K_median` per kernel (from the 104 sidecars) reads gemm 4,248,
    floyd 2,209, gibbs 406, nbody 2,196, spmm 1,176, stencil_jacobi 3,228, fft 4,240,
    histogram 2,202, fem_assembly 8,410, lexer 150, rmat_gen 790, bnb_tsp 4,672, idle 150
    (the floor set of 150 pages is inside every kernel's count).

## 6. The G-ORD simulation (section 5 item 8)

Four reps per kernel, 120 pairs, the K series level-normalized by the cell's median, the eight
B1 shape features per window at `H = W // 2`, LOKO by kernel in archetype space, a 200-tree
forest with cell-majority vote; `shuffled` is the mean over three order shuffles per cell,
`spread` is p95 minus p05 of twelve archetype-label permutations across kernels. This is a
sanity check of the fixture, not builder 2's gate (which uses 20 order shuffles and 100 label
permutations); the rule is SPEC 3.5.6: order-blind when `|ordered - shuffled| <= spread`.

```
ord-case period W= 8 H= 4: ordered=0.69 shuffled=0.56 null p05-p95 spread=0.14 -> order-blind
ord-case period W=16 H= 8: ordered=0.73 shuffled=0.60 null p05-p95 spread=0.25 -> order-blind
ord-case period W=32 H=16: ordered=0.73 shuffled=0.64 null p05-p95 spread=0.23 -> order-blind
ord-case burst  W= 8 H= 4: ordered=0.69 shuffled=0.49 null p05-p95 spread=0.07 -> resolution
ord-case burst  W=16 H= 8: ordered=0.69 shuffled=0.46 null p05-p95 spread=0.13 -> resolution
ord-case burst  W=32 H=16: ordered=0.58 shuffled=0.35 null p05-p95 spread=0.12 -> resolution
```

Reading: with periods 6 and 12 the shuffled score stays well above chance (0.56 to 0.64), because
the period also sets how often a window contains a pulse, and the difference sits inside the
null's spread at every W; the SPEC 5.3 must-pass case ("resolution at W >= 16") does not hold
on that fixture. With the burst design the shuffled score falls to the chance region (0.35 to
0.49) and the difference exceeds the spread at every W, including W = 8. The ceiling near 0.7
in both modes is by construction: two archetypes share each period class, so a kernel of the
second archetype in a class can never be assigned its own label. The default stays the SPEC's
`--ord-mode period`; the author chooses whether builder 2's G-ORD must-pass test is built on
`--ord-mode burst` (recommended by these numbers) and whether the must-refuse case ("classes
differing in K0 only") is `--level-only`, as SPEC 5.3 says.
