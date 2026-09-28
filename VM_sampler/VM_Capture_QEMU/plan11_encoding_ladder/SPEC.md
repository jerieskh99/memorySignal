# plan11_encoding_ladder: specification for the paper 2 analysis toolkit

Written 2026-09-16. Three builders implement this in parallel without talking to each other.
Every interface below is fixed. Where a definition in the sources leaves a choice, the choice is
exposed as a parameter with the default stated here and listed in section 8; a builder never
picks a different default and never adds an undeclared threshold.

Sources of truth, in this order (a builder reads them before coding; the citations in docstrings
point at them):

1. `apf_paper/P2_STRUCTURE.md` (sections 2, IV, V, 4 tables, 5 blind-move order). Cited below as
   `P2 Sec. V` etc.
2. `apf_paper/council/12_P2_COUNCIL_REPORT.md` section 2, items 1 to 37. Cited as `CR 2.x item N`.
3. `apf_paper/council/10_al_kindi_revised.md` sections 2 and 5. Cited as `K2 Sec. 2` and `K2 move N`.
4. `apf_paper/P2_AUTHOR_ANSWERS.md`. Cited as `AA A1` etc.

Existing gate code that is re-pointed, not reinvented (all under
`mem_sig/VM_sampler/VM_Capture_QEMU/`): `plan03_sweep.py`, `plan03_aggregate.py`,
`plan03_metric_kernel.py`, `plan08_b1/b1_splits.py`, `plan08_b1/b1_features.py`,
`plan08_b1/b1_extract_hamming.py` (the zstd streaming reader), `plan02_validate_session.py`,
`plan05_campaign/validate_campaign.py`. The toolkit does not import them (their import paths pull in
plan02 modules and a repo-root package); it copies the named functions verbatim with a comment
`# copied from <file>:<function>, <date>` and a test that asserts equality against the original when
the original is importable.

Binding rules for every builder:

- No server access of any kind. No path under `/mnt/nfs` or `/project` appears in code except as a
  documented example default that the runbook tells the author to pass explicitly.
- The sandbox family is referred to only as "the sandbox family". Its workloads are not read,
  grepped, or named. Nothing in this toolkit needs them.
- No paper prose. The LaTeX skeleton is headings, table environments, figure placeholders and
  comment blocks with substance bullets.
- Every gate function's docstring cites the definition it implements (P2 section and CR item).
- Python 3.10+. `numpy` required. `scikit-learn` required for the forest and the clustering.
  `scipy` optional (a stdlib or numpy fallback where cheap). `matplotlib` required for figures
  only; when absent the figure step writes `figures/SKIPPED.txt` naming the missing module and
  exits 0. `zstandard` optional: `.csv.zst` is read through the `zstd` binary (`zstd -dc -q`) first,
  through `zstandard.ZstdDecompressor().stream_reader` second, and plain `.csv` always works.
- Stream. A cell trajectory has about four million rows and is never loaded whole. Every later
  stage reads the per-cell extract (about 1,000 rows), which may be loaded freely.
- Nothing is committed to git. No file outside `plan11_encoding_ladder/` is edited.
- Numbers never move: every threshold is a module-level constant or a CLI parameter with the
  default written in this file; the value used is written into every result file's `params` block.
- Refusals are strings from section 3.0's vocabulary, written to the artifact, never a number,
  never a blank.

---

## 1. Package layout

Directory: `mem_sig/VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/`, a Python package
(`__init__.py` with `__version__ = "0.1.0"`). Modules are run as `python3 -m
plan11_encoding_ladder.<module>` from `VM_Capture_QEMU/`, and every module also runs as a script
by absolute path (each begins with the two-line `sys.path` insertion of its own parent's parent).

Builder 1, extract (owns these files and nothing else):

| File | Owns |
|---|---|
| `__init__.py` | `__version__ = "0.1.0"` only. |
| `schema.py` | Every shared constant and column list: `N_PAGES = 262144`, `PAGE_SIZE = 4096`, `BITS_PER_PAGE = 32768`, `DURATION_S = 600`, `DT_BRACKET_S = (0.500, 0.644)`, `QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)`, `EXTRACT_COLUMNS` (section 2.2), `SIDECAR_SCHEMA = "plan11.extract.v1"`, `KERNELS` (section 2.6), `ARCHETYPES`, `GRID_WINDOWS = (8, 16, 32, 64, "whole")`, `GRID_HOP_RATIOS = (0.25, 0.50, 1.00)`, `grid_id(W, H)`, `grid_points(n_series)`, `campaign_of(label)`, `parse_cell_path(path)`. Builders 2 and 3 import it and never edit it. |
| `extract.py` | The streaming per-cell extractor (K2 move 1; section 2) and the cell index (`index` subcommand). |
| `synth.py` | The synthetic trajectory generator and the synthetic corpus (section 5). Every test in the package uses it. |
| `tests/test_schema.py`, `tests/test_extract.py`, `tests/test_synth.py` | Builder 1's tests. |

Builder 2, gates (owns these files and nothing else):

| File | Owns |
|---|---|
| `verdicts.py` | The verdict vocabulary as string constants (section 3.0) and `Refusal`, a `str` subclass; `is_refusal(s)`. |
| `series.py` | Per-rung per-snapshot series from an extract, head drop, level normalization, windowing, the eight shape features, feature matrices (section 3.1). |
| `nulls.py` | Phase-randomized surrogates, unit-level label shuffles, order shuffles, the null summary (p95, rank, strict exceedance) (section 3.2). |
| `splits.py` | within-trace, LORO, LOKO folds at the unit of the cell (section 4.1). |
| `models.py` | The forest, the one-feature threshold model (B1's L1), unit aggregation, scores, clustering with ARI and NMI (section 4.2 to 4.4). |
| `gates_precondition.py` | C1 to C8 as re-mapped, the `failed/` count, G-K0, G-F (section 3.3). |
| `gates_calibration.py` | The pass table, G-C, G-P, the alias falsifier (section 3.4). |
| `gates_temporal.py` | G1 to G5 as amended, G3 as a per-kernel flag, G-ORD, the grid roll-up and the selection rule (section 3.5). |
| `gates_readings.py` | G-J, G-DEC (section 3.6). |
| `gates_comparison.py` | B1-G1, B1-G3, B1-G6, G-L, G-N, G-X, G-DIM, G-M (section 3.7). |
| `variance.py` | G-V (section 3.8). |
| `tests/test_gates_*.py` (one per module above) | Builder 2's tests: for every gate one case that must pass and one that must refuse, both generated by `synth.py`. |

Builder 3, report (owns these files and nothing else):

| File | Owns |
|---|---|
| `tables.py` | Tables 5, 6, 7, 8 and G-V as CSV and as LaTeX `tabular` fragments (section 6). |
| `figures.py` | The figures of P2 Sec. VII (section 6.7). |
| `latex_skeleton.py` | Writes `paper2_skeleton.tex` (section 6.8). |
| `driver.py` | The run-everything driver in al-Kindi's move order, resumable (section 7). |
| `RUNBOOK.md` | The command-line runbook (section 7). |
| `requirements.txt` | `numpy>=1.21`, `scikit-learn>=1.0`, `matplotlib>=3.4`; optional: `scipy`, `zstandard`. |
| `tests/test_report.py`, `tests/test_driver.py` | Builder 3's tests, run on the synthetic corpus end to end. |

Interfaces between builders are files on disk with the schemas in this document; no builder calls
another builder's functions except `schema.py` and `verdicts.py` (constants only). Builder 3 reads
builder 2's result files by their documented column names and never recomputes a verdict.

Output root: every command takes `--out <dir>`; the layout under it is fixed:

```
<out>/
  cells.csv                                  builder 1, the cell index (2.7)
  extract/<cell_id>/extract.csv              builder 1 (2.2)
  extract/<cell_id>/sidecar.json             builder 1 (2.3)
  inputs/pass_table.csv                      author input, template by builder 2 (3.4.1)
  inputs/gk0_source.csv                      author input, template by builder 2 (3.3.3)
  inputs/head_drop.csv                       author input, template by builder 2 (3.1.2)
  inputs/failed_counts.csv                   author input, optional (3.3.2)
  inputs/idle_admissibility.json             author input, template by builder 2 (3.3.4)
  inputs/cell_order.csv                      author input, optional (3.7.6)
  features/<rung>/<grid_id>_{raw,norm}.npz   builder 2 (3.1.5)
  gates/preconditions.csv  gates/preconditions.json
  gates/failed_counts.csv
  gates/gk0.csv  gates/gf.csv  gates/gc.csv  gates/gp.csv  gates/alias.csv
  gates/grid/<rung>/<grid_id>/temporal_per_kernel.csv      every grid point, kept
  gates/grid/<rung>/<grid_id>/g1_surrogates.npz
  gates/grid/<rung>/<grid_id>/gord.json
  gates/table5_long.csv  gates/table5_grid.csv  gates/selection.json
  gates/g3_flags.csv  gates/gj.csv  gates/gdec.csv
  gates/splits/<rung>/<grid_id>/<split>__<labelspace>/predictions.csv
  gates/splits/<rung>/<grid_id>/<split>__<labelspace>/scores.json
  gates/splits/<rung>/<grid_id>/<split>__<labelspace>/null.json
  gates/splits/<rung>/<grid_id>/<split>__<labelspace>/l1_quarantine.json
  gates/gl.csv  gates/gn.csv  gates/gx.csv  gates/gx.json  gates/gdim.csv  gates/gm.csv
  gates/gv.csv  gates/clustering.csv  gates/clustering.json
  report/tables/table{5,6,7,8,gv}.csv and .tex
  report/figures/*.png and *.pdf
  report/paper2_skeleton.tex
  report/manifest.json                       builder 3: every file above with sha256
  driver_state.json                          builder 3: the move ledger
```

Every JSON result file has the three top-level keys `schema` (a string
`"plan11.<name>.v1"`), `params` (every parameter value used, by name) and `citation` (the
definition string from the gate's docstring), then its payload.

---

## 2. The per-cell extract (builder 1)

### 2.1 Input: the substrate trajectory

One file per cell, `*substrate_trajectory.csv.zst` (fallbacks `.csv.gz`, `.csv`), found by the
glob `*substrate_trajectory.csv*` inside the cell directory (the name is
`run_matrix_test<N>_<test_label>.npy.substrate_trajectory.csv.zst`; verified in
`plan10_analysis/corpus_manifest.py` line 50 and `capture_consumer_qemu.sh` line 374). Exactly one
match per cell; zero or more than one is `refused: trajectory file count != 1`.

Header (verified 2026-09-16 on the server and in `live_delta_calc_modular/src/metrics/mod.rs`
`csv_header`): `seq,page_index,` followed by the 64 metric names, 66 columns. The extractor
locates `seq`, `page_index`, `hamming`, `l0`, `l1` by name and ignores every other column. Units
(from `metrics/family_a/positional.rs`): `hamming` = bits flipped (popcount of p XOR q), `l0` =
number of changed bytes (1..4096 on a changed page), `l1` = sum of absolute byte differences,
`mean_abs` = `l1 / 4096`. Because `mean_abs` is `l1 / 4096` exactly, the extract does not carry it;
any reading on `mean_abs` is computed from the `l1` columns divided by 4096 (this matters for G-C's
content ordering, 3.4.3).

Row semantics: one row per changed page per snapshot (sparse mode emits only `hamming != 0`), rows
grouped by `seq`, `seq` non-decreasing through the file (`capture_consumer_qemu.sh` line 384 appends
a snapshot's rows in one `awk` call with the snapshot's counter, line 388 increments it). The extractor asserts
non-decreasing `seq`; a decrease is `refused: seq not monotone at row <n>`. A snapshot whose
differ output has no changed page appends no row at all (the same `awk` prints nothing), so a
missing `seq` is a K = 0 snapshot, not a failed job; see 2.4. Duplicate `page_index` inside one
`seq` is not expected; the first row is kept and `n_rows_dup_page` counts the rest.

`seq` was observed to start at 1 on the server; `capture_consumer_qemu.sh` line 35 initialises the
counter at 0. The extractor assumes neither: it records `seq_first` and `seq_last` and defines the
pair count as `seq_last - seq_first + 1`.

### 2.2 Output 1: `extract/<cell_id>/extract.csv`

CSV, header row, one row per `seq` from `seq_first` to `seq_last` inclusive (filled gaps
included), rows in increasing `seq`. Column order is fixed (`schema.EXTRACT_COLUMNS`), 58 columns:

```
seq
K
n_persist
n_union
J
J_null_inter
J_null
ham_sum_all  ham_q05_all  ham_q25_all  ham_q50_all  ham_q75_all  ham_q95_all
l0_sum_all   l0_q05_all   l0_q25_all   l0_q50_all   l0_q75_all   l0_q95_all
l1_sum_all   l1_q05_all   l1_q25_all   l1_q50_all   l1_q75_all   l1_q95_all
ham_sum_per  ham_q05_per  ham_q25_per  ham_q50_per  ham_q75_per  ham_q95_per
l0_sum_per   l0_q05_per   l0_q25_per   l0_q50_per   l0_q75_per   l0_q95_per
l1_sum_per   l1_q05_per   l1_q25_per   l1_q50_per   l1_q75_per   l1_q95_per
r_l0_q05_per    r_l0_q25_per    r_l0_q50_per    r_l0_q75_per    r_l0_q95_per
r_l1l0_q05_per  r_l1l0_q25_per  r_l1l0_q50_per  r_l1l0_q75_per  r_l1l0_q95_per
r_haml0_q05_per r_haml0_q25_per r_haml0_q50_per r_haml0_q75_per r_haml0_q95_per
```

Definitions, with `S_t` the set of `page_index` values at `seq = t`, `K_t = |S_t|`, and
`P_t = S_t ∩ S_{t+1}` the persistent pages of pair `(t, t+1)`:

| Column | Definition | Blank when |
|---|---|---|
| `K` | `|S_t|` (row count after de-duplication) | never (0 on a filled gap) |
| `n_persist` | `|P_t|` | `t = seq_last` |
| `n_union` | `K_t + K_{t+1} - n_persist` | `t = seq_last` |
| `J` | `n_persist / n_union`; `J = 0` if exactly one of the two sets is empty; blank if both are empty | `t = seq_last`, or both sets empty |
| `J_null_inter` | `K_t * K_{t+1} / N` (the independence null's expected intersection, P2 Sec. IV rung 1; K2 Sec. 2 rung 1) | `t = seq_last` |
| `J_null` | `J_null_inter / (K_t + K_{t+1} - J_null_inter)`, the Jaccard implied by the independence null as a ratio of expectations; 0 when the denominator is 0 | `t = seq_last` |
| `<ch>_sum_all` | sum of channel `ch` over all rows of `seq = t` | never (0 on a gap) |
| `<ch>_qNN_all` | quantile `NN/100` of channel `ch` over all rows of `seq = t` | `K_t = 0` |
| `<ch>_sum_per`, `<ch>_qNN_per` | the same over the rows of the persistent side (2.5) restricted to `P_t` | `t = seq_last` or `n_persist = 0` |
| `r_l0_qNN_per` | quantile of `l0 / 4096` over the persistent rows | as above |
| `r_l1l0_qNN_per` | quantile of `l1 / l0` over the persistent rows (`l0 >= 1` on every changed page, so no division by zero; a row with `l0 = 0` is counted in `n_rows_zero_l0` and excluded from the ratios) | as above |
| `r_haml0_qNN_per` | quantile of `hamming / l0` over the persistent rows, same guard | as above |

`ch` ranges over `ham` (the `hamming` column), `l0`, `l1`. Quantiles use `numpy.quantile(...,
method="linear")` (the default). Integers are written as integers, floats with `format(x, ".10g")`,
blanks as the empty string.

The three ratios are the fixed per-snapshot summary of the content-change rung (P2 Sec. IV rung 2;
K2 Sec. 2 rung 2 (b), (c)); the sums and quantiles over all rows serve APF, wAPF and G-K0.

### 2.3 Output 2: `extract/<cell_id>/sidecar.json`

```
{
  "schema": "plan11.extract.v1",
  "extractor_version": "<package __version__>",
  "cell_id": "...", "kernel": "...", "role": "kernel" | "idle",
  "archetype_predicted": "...", "seed": <int or null>, "rep": <int>,
  "rep_dir": <int>, "label": "...", "campaign": "...",
  "path": "<cell directory as given>", "traj_file": "<basename>", "source_bytes": <int>,
  "N": 262144, "page_size": 4096, "bits_per_page": 32768,
  "duration_s_declared": 600,
  "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
  "persist_side": "t" | "t+1",
  "header_sha256": "<sha256 of the header line, bytes, no newline>",
  "header_ncols": 66,
  "columns_used": {"seq": 0, "page_index": 1, "hamming": 2, "l0": 4, "l1": 5},
  "n_rows_in": <int>, "n_rows_skipped": <int>, "n_rows_dup_page": <int>,
  "n_rows_zero_hamming": <int>, "n_rows_zero_l0": <int>,
  "seq_first": <int>, "seq_last": <int>, "n_seq_present": <int>,
  "n_pairs": <int>, "n_seq_gaps": <int>, "gap_seqs": [<first 100>],
  "dt_est_s": <600 / n_pairs>, "dt_bracket_s": [0.5, 0.644],
  "K_median": <float>, "K_max": <int>, "apf_max": <float>,
  "failed_count": <int or null>, "failed_count_source": "<string>",
  "status": "ok" | "refused: <reason>",
  "started_at": "<iso>", "finished_at": "<iso>", "elapsed_s": <float>
}
```

`n_pairs = seq_last - seq_first + 1` is the paper's pair count (Table 1 "realized pairs per
cell"); `dt_est_s = 600 / n_pairs` is the derived guest spacing (AA A6, S5). `n_rows_in` is the
input's data row count (header excluded). `failed_count` is a recorded input, never computed: from
`--failed-count <int>` or from `--failed-dir <path>` (the count of files in that directory), else
`null` with `failed_count_source = "not recorded"`. `n_rows_zero_hamming` counts rows with
`hamming = 0` (not expected in sparse mode; they are kept in `S_t`, since the row says the page
changed under some channel, and counted).

### 2.4 The streaming pass, exactly

Memory holds at most two snapshots. State: `buf_cur` (lists of `page_index`, `hamming`, `l0`,
`l1` for the `seq` being read), `snap_prev` (the finished snapshot `t`: `page_index` sorted
ascending as `numpy.int64`, the three channel arrays permuted into the same order), `snap_cur`
(the finished snapshot `t+1`, same layout). The loop:

1. Read the header; locate the five columns; hash the header line; record `header_ncols`.
2. For each data row: parse `seq`, `page_index`, `hamming`, `l0`, `l1` as `int`; a parse failure
   increments `n_rows_skipped` and continues. If `seq < current seq`: refuse. If `seq > current
   seq`: finalize `buf_cur` into a snapshot (sort by `page_index`, drop duplicates keeping the
   first, count them), call `emit(prev=snap_prev, cur=that snapshot)` for the previous snapshot,
   then for every `g` in `current seq + 1 .. seq - 1` synthesize an empty snapshot for `g` and
   emit through the same path (so a gap is a K = 0 snapshot with all-zero sums, blank quantiles,
   and `J` computed against an empty set), then start `buf_cur` for `seq`.
3. At end of file: finalize the last snapshot and emit it with `cur = None`.

`emit(prev, cur)` writes the row for `prev.seq`: the `_all` columns from `prev`; if `cur` is
`None` (the last `seq`), every `_per`, `J`, `n_persist`, `n_union`, `J_null_inter`, `J_null` column
is blank; otherwise `P = numpy.intersect1d(prev.pages, cur.pages, assume_unique=True,
return_indices=True)` gives the persistent pages and their positions in both snapshots,
`n_persist = len(P)`, `n_union = K_prev + K_cur - n_persist`, `J` as defined, the persistent-side
channel values are taken from `prev` (`persist_side = "t"`, the default) or from `cur`
(`persist_side = "t+1"`), and the `_per` and `r_*_per` columns follow.

So at every moment the arrays in memory are two snapshots of at most `N` entries each plus the
buffer of the snapshot being read. The extract file is written row by row through a `csv.writer`
on a `.tmp` path renamed at the end (as `plan03_sweep.py` does). The pass never seeks, never
re-reads and never holds a third snapshot.

At the last `seq` there is no `S_{t+1}` and therefore no `J`: the row exists with its `_all`
columns and blanks elsewhere. Every rung series in section 3.1 drops the last `seq`, so all rungs
have `n_pairs - 1` samples aligned by `seq`.

### 2.5 Function signatures (builder 1)

```python
# extract.py
def open_text(path: str):                         # copied from plan08_b1/b1_extract_hamming.py:open_text,
    ...                                           # extended: zstd binary -> zstandard module -> gzip -> plain
def extract_cell(cell_dir: Path, out_dir: Path, *,
                 n_pages: int = 262144, page_size: int = 4096,
                 quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
                 persist_side: str = "t",          # "t" | "t+1"
                 duration_s: int = 600,
                 failed_count: int | None = None, failed_dir: Path | None = None,
                 role: str | None = None,           # None -> from parse_cell_path
                 cell_id: str | None = None) -> dict:   # returns the sidecar dict
def build_index(root: Path, out_csv: Path, *, idle_markers: tuple[str, ...] = ("sleep", "idle"),
                role_overrides: Path | None = None) -> list[dict]:
def main(argv: list[str] | None = None) -> int:
# CLI:
#   extract.py index  --root <retention root or any dir holding kernel/...> --out <out>
#   extract.py cell   --cell-dir <dir> --out <out> [--persist-side t|t+1] [--failed-count N | --failed-dir D]
#                     [--role kernel|idle] [--cell-id ID]
#   extract.py all    --cells-csv <out>/cells.csv --out <out> [--jobs 1] [--only <regex on cell_id>]
#                     [--failed-counts <csv>]   (one process per cell; --jobs > 1 uses multiprocessing)
```

`extract.py all` skips a cell whose `sidecar.json` exists with `status == "ok"` unless `--force`.
Throughput expectation: one to three minutes per cell of four million rows in pure Python
`csv.reader`; the runbook states it.

### 2.6 Cell identity: `schema.parse_cell_path`

The retention layout is `<family>/<test_label>/<param-sig>/rep<NNN>__<label>/` (verified:
`run_files_controlled.py` `retention_workload_path`, lines 1059 to 1065; `param_signature_from_command`,
line 1009, appends `_<sha1[:8]>` only when the signature exceeds 60 characters; the `__<label>` suffix is added
by the archive writer, `plan10_analysis/results_view.py` line 109 documents the same shape).
`parse_cell_path(path) -> dict` reads the last four path components and returns:

- `family`: component 1 (expected `kernel`; the idle cells may carry another family, see role).
- `test_label`: component 2, e.g. `kernel_gemm_v2`.
- `kernel`: `test_label` with a leading `kernel_` removed and a trailing `_v<digits>` removed
  (`kernel_gemm_v2 -> gemm`, `kernel_stencil_jacobi_v2 -> stencil_jacobi`).
- `param_sig`: component 3; `seed`: the first match of `seed_(\d+)` in it (the regex of
  `results_view.py` line 44), else `null`.
- `rep_dir`: the integer of `rep(\d{3})` in component 4; `label`: the text after the first `__`
  in component 4, else the empty string.
- `campaign`: `schema.campaign_of(label)`: `"dwarfs1"` if the label starts with `dwarfs1`,
  `"01c"` if it equals `sandbox_deepdive_01c`, `"01c1"` if it equals `sandbox_deepdive_01c1`,
  otherwise the label itself.
- `role`: `"idle"` if any of `idle_markers` is a substring of `test_label` (case-insensitive);
  else `"kernel"` if `kernel` is in `schema.KERNELS`; else `"unknown"`, and the cell's
  `cells.csv` status is `refused: unknown kernel` (excluded until the author edits the row). The
  author overrides per cell through `--role` or by editing `cells.csv`.
- `archetype_predicted`: from `schema.KERNELS` for a kernel; `"control"` for idle;
  `"unknown"` otherwise.

`rep` (the paper's rep index, 0 to 7; P2 Sec. VI, AA A4) is assigned by `build_index`, not by the
path: within one kernel, the cell with `seed == 42` is rep 0; the remaining cells are ordered by
seed ascending and numbered 1 upward. A cell without a parsed seed (an idle cell) takes
`rep = rep_dir - 1`. Two cells of one kernel with the same seed are both kept, flagged
`status = "refused: duplicate seed"` in `cells.csv`, and the author resolves it.

`cell_id = f"{kernel}__rep{rep:02d}__{campaign}"` for kernels and `f"idle__rep{rep:02d}__{campaign}"`
for idle cells. `cell_id` is filesystem-safe by construction.

`schema.KERNELS` (P2 Table 3; AA A2), an ordered tuple of `(kernel, archetype)`:

```
("gemm", "WORKING-SET"), ("floyd", "WORKING-SET"), ("gibbs", "WORKING-SET"),
("nbody", "WORKING-SET"), ("spmm", "WORKING-SET"), ("stencil_jacobi", "WORKING-SET"),
("fft", "SCATTER"), ("histogram", "SCATTER"), ("fem_assembly", "SCATTER"),
("lexer", "SEQUENTIAL-GROW"), ("rmat_gen", "SEQUENTIAL-GROW"),
("bnb_tsp", "FRONTIER-CHURN")
```

`schema.ARCHETYPES = ("IDLE", "WORKING-SET", "SCATTER", "SEQUENTIAL-GROW", "FRONTIER-CHURN")`.
`schema.LEVEL_MATCHED_SETS = (("floyd", "histogram", "nbody"), ("fft", "gemm"))` (P2 Sec. 2 and
Sec. VI; AA A7).

### 2.7 `cells.csv`

Columns: `cell_id, kernel, role, archetype_predicted, seed, rep, rep_dir, label, campaign, path,
traj_file, status`. One row per directory under `--root` that holds exactly one trajectory file;
`status` is `ok`, `refused: trajectory file count != 1`, `refused: duplicate seed`, or
`refused: unknown kernel`. `build_index` walks `root.rglob("rep*__*")` and keeps directories. The author
may edit `role` and `archetype_predicted`; every later stage reads this file, never the paths.

---

## 3. The gates (builder 2)

### 3.0 Verdict vocabulary (`verdicts.py`)

Every verdict column holds exactly one of these strings. A gate that cannot run writes the
`NOT_RUN` or `NOT_APPLICABLE` form with its reason; a gate that runs and refuses writes the
`REFUSED` form or one of the named refusals. Numbers go in their own columns.

```python
PASS = "pass"
FAIL = "fail"
def not_run(reason: str) -> str:        return f"not run: {reason}"
def not_applicable(reason: str) -> str: return f"not applicable: {reason}"
def refused(reason: str) -> str:        return f"refused: {reason}"
def pending(reason: str) -> str:        return f"pending: {reason}"

# G1
TREND_PRESENT = "trend present"
# G2
G2_ABOVE_NYQUIST = "not applicable, rhythm above Nyquist"
G2_UNDETERMINED = "undetermined by the interval calibration"
# G-P
GP_RESOLVABLE = "resolvable"
GP_MARGINAL = "marginal"
GP_ALIASED = "aliased by design at this size"
GP_UNDECLARED = "undeclared"
GP_RHYTHM_UNDERSAMPLED = "rhythm under-sampled"
GP_PASS_ALIASED = "pass aliased"
GP_ADMITTED = "admitted"
# G3 flag
G3_PRESENT = "rhythm flag: present"
G3_ABSENT = "rhythm flag: absent"
# G-ORD
GORD_ORDER_BLIND = "order-blind"
GORD_RESOLUTION = "resolution"
# G-K0
GK0_IDLE_MEASURED = "IDLE, measured"
GK0_ABOVE_FLOOR = "above floor"
# G-F
GF_INSEPARABLE = "inseparable at floor"
GF_VOID = "void: idle reps separable under this rung"
GF_AT_FLOOR = "at floor in this lead"
# G-C
GC_DISCONNECTED = "disconnected lead"
# G-J
GJ_INTERPRETABLE = "interpretable"
GJ_FLOOR_OVERLAP = "floor overlap"
GJ_FLOOR_UNMEASURED = "floor unmeasured"
# G-DEC
GDEC_DECAY = "decay"
GDEC_NO_DECAY = "no decay"
GDEC_NO_BEYOND_BREADTH = "no decay beyond breadth"
GDEC_NO_BEYOND_FLOOR = "no decay beyond floor or host"
GDEC_NOT_RESOLVED = "decay not resolved"
# B1-G1
NEAR_UNFALSIFIABLE = "near_unfalsifiable"
# G-L
GL_LEVEL_ONLY = "level only"
GL_SHOT_NOISE = "refused: shot noise explains CV"
# G-N
GN_HEADLINE = "headline"
GN_ONE_TRAIN = "one training kernel per fold"
GN_NOVELTY = "structural novelty"
GN_NO_ROW = "no kernel row"
# G-X
GX_POOLING_STANDS = "pooling stands"
GX_LEAK = "campaign predictable"
GX_CONFOUND_NONE = "confound: none"
GX_CONFOUND_PARTIAL = "confound: partial"
GX_CONFOUND_TOTAL = "confound: total"
# G-DIM
GDIM_FULL = "full vector"
GDIM_REDUCED = "declared reduction"
# G-M
GM_BEATS = "beats"
GM_DIFFERENCE = "difference with margin"
# G-V
GV_NOT_ESTIMABLE = "LOKO not estimable"
GV_ESTIMABLE = "estimable"
# Alias falsifier
ALIAS_MOVES = "moves with the interval"
ALIAS_STAYS = "does not move with the interval"
```

`is_refusal(s)` is true for `FAIL`, any `refused:` / `not run:` / `not applicable:` string, and
the named refusals `TREND_PRESENT, G2_ABOVE_NYQUIST, G2_UNDETERMINED, GF_VOID, GF_AT_FLOOR,
GC_DISCONNECTED, GDEC_NO_BEYOND_BREADTH, GDEC_NO_BEYOND_FLOOR, GDEC_NOT_RESOLVED,
NEAR_UNFALSIFIABLE, GL_LEVEL_ONLY, GL_SHOT_NOISE, GV_NOT_ESTIMABLE`.

### 3.1 Series, normalization, windows, features (`series.py`)

3.1.1 Rungs. `RUNGS = ("apf", "wapf", "persist", "content", "combined")`. Per cell, from
`extract.csv`, the per-snapshot channel matrix `S[rung]` has `n_series = n_pairs - 1 - head_drop`
rows (the last `seq` dropped; the first `head_drop` dropped) and these channels, in this order:

| rung | raw channels (d) | level-normalized channels (d) | rule, citation |
|---|---|---|---|
| `apf` | `K / N` (1) | `K / K_median_cell` (1) | count rung: divide by the cell's own median K (P2 Sec. V G-L (i); CR 2.2 item 24) |
| `wapf` | `ham_sum_all / (N * 32768)` (1) | `ham_sum_all / (K_median_cell * 32768)` (1) when `wapf_norm = "median_K"` (the literal count-rung rule); `wapf / median_cell(wapf)` when `wapf_norm = "median_self"` | same rule; the choice is listed in section 8 |
| `persist` | `J` (1) | `J - J_null` (1) | persistence: report J against its independence null (G-L (i)) |
| `content` | the 15 `r_*_per` columns, primary three first: `r_l0_q50_per, r_l1l0_q50_per, r_haml0_q50_per`, then the 12 other quantiles in extract order (15) | identical (the ratios are level-free; K is not a feature of this rung: P2 Sec. IV rung 2, "K is not a feature of this rung (G-L)") | CR 2.2 item 24 "the magnitude rung: drop the K column" |
| `combined` | not defined raw | concatenation of the four normalized matrices (18) | P2 Sec. IV rung 3; K2 Sec. 2 "the combined rung" |

Channel names (used in feature names): `apf` raw `k_over_n`, normalized `k_over_med`; `wapf` raw
`wapf`, normalized `wapf_norm`; `persist` raw `j`, normalized `j_excess`; `content` the extract
column names.

`K_median_cell` is the median of `K` over the cell's rows after head drop (P2 Sec. V: "level
normalization from per-cell statistics"; CR 2.1 item 19). A blank `J` (both sets empty) or a blank
`r_*` (no persistent page) is `NaN` in the matrix and is imputed per fold (4.2).

3.1.2 Head drop (`inputs/head_drop.csv`, columns `kernel, head_drop_pairs, reason`). Default 0 for
every kernel. P2 Sec. VI says the lexer's first pass is kept out of every steady-state statistic
and that the warm-up guard drops by phase marker, not by count; no marker is in the trajectory, so
the number of pairs to drop is the author's input. `series.load_head_drop(path) -> dict[str,int]`.
G-K0's "last 80 percent" is computed on the undropped series by its own definition.

3.1.3 Windows. `n_windows(n, W, H) = (n - W) // H + 1 if n >= W else 0` (Plan 02
`compute_n_windows`, used by `plan03_metric_kernel.py`). Window `i` covers rows
`[i*H, i*H + W)`. The grid: `W in (8, 16, 32, 64)` with `H = max(1, round(W * r))` for
`r in (0.25, 0.50, 1.00)` (`plan03_sweep.py` defaults, P2 Sec. V "hop ratios as Plan 03"), plus
the whole cell, `W = n_series` of that cell, `H = W`, one window (K2 Sec. 2 rung 0 (b): "W =
whole cell added to the grid"). `schema.grid_id(W, H)` is `f"W{W}_H{H}"` for the integer points
and `"Wall_Hall"` for the whole cell; `schema.grid_points(n_series)` returns the 13 `(W, H)` pairs
in that order. Every grid point is computed and kept (P2 Sec. V, al-Farabi's condition); the
on-disk guarantee is 3.5.7.

3.1.4 Shape features. `shape_features(x: np.ndarray) -> np.ndarray[8]` is
`plan08_b1/b1_features.py:features` copied verbatim (including `_pct`'s index rule
`sorted[min(n-1, int(q*n))]` and `duty = fraction of samples > 0.1 * max`), returning
`(mean, std_population, cov, median, max, p95, peak2med, duty)` with the zero guards of the
original. `FEAT = ("mean", "std", "cov", "median", "max", "p95", "peak2med", "duty")`. B1's
design note says duty was to be redefined against the cell's median; the code in the repository
uses `0.1 * max`; the code is what is copied, and the choice is listed in section 8
(`duty_rule`).

3.1.5 Window feature vectors. For each rung and grid point, per window:

- `apf`, `wapf`, `persist`: the 8 shape features of the single channel (d = 8).
- `content`: the 8 shape features of each of the three primary channels (24) plus the window mean
  of each of the 12 secondary quantile channels (12): d = 36, in that order.
- `combined`: `apf` (8) + `wapf` (8) + `persist` (8) + `content` (36): d = 60.

Feature names are `f"{rung}.{channel}.{feat}"` (`apf.k_over_med.mean`, `content.r_l0_q50_per.duty`,
`content.r_l1l0_q05_per.wmean`). A window with any `NaN` channel value keeps `NaN` in the
affected features (the imputer handles it); a window entirely `NaN` is dropped and counted in
the npz scalar `n_windows_dropped`.

`build_features(out: Path, cells_csv: Path, rung: str, W, H, normalized: bool, head_drop) ->
Path` writes `features/<rung>/<grid_id>_{raw|norm}.npz` with arrays `X` (float64,
`[n_rows, d]`), `feature_names` (str), `cell_id`, `kernel`, `archetype`, `campaign`, `role` (str
per row), `rep`, `win_start`, `n_series_cell` (int per row), and scalars `W`, `H`, `grid_id`,
`normalized`, `head_drop_json`. Row order: cells in `cells.csv` order, windows chronological.
The `raw` variant of `combined` is not written. Idle cells are included with `archetype =
"IDLE"` and `kernel = "idle"`; the split functions select roles.

Per-cell headline readings (`series.cell_headline(cell, rung) -> float`), used by G-F (ii), G-K0
and the fused plane: `apf`: median of `K / N`; `wapf`: median of `wapf`; `persist`: median of `J`;
`content`: median of `r_l0_q50_per`; `combined`: not defined (G-F (ii) runs on the four rungs).

### 3.2 Nulls (`nulls.py`)

```python
def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One phase-randomized surrogate: rfft of (x - mean), phases replaced by uniform [0, 2pi)
    with the DC and Nyquist bins kept real, irfft to len(x), mean added back. The amplitude
    spectrum and therefore the autocorrelation function (lag-1 included) are preserved exactly.
    This is the 'phase-randomized' alternative of CR 2.1 item 3 (G1); block bootstrap is not
    implemented (section 8)."""
def surrogates(x, n: int = 200, seed: int = 20260916) -> np.ndarray:   # [n, len(x)]
def shuffle_labels_units(cells: np.ndarray, kernels: np.ndarray, archetypes: np.ndarray,
                         split: str, labelspace: str, rng) -> np.ndarray:
    """Unit-level label shuffle (CR 2.1 item 9). LOKO/archetype: permute the archetype
    labels across the 12 kernels (one draw per kernel; every cell and window inherits).
    LORO or within-trace, kernel space: permute kernel labels across cells with the
    eight-per-kernel structure kept (a random permutation of the per-cell label vector).
    LORO or within-trace, archetype space: the kernel-level permutation, cells inherit.
    Campaign labels (G-X): permuted across cells. Returns the per-cell label vector."""
def order_shuffle(series: np.ndarray, rng) -> np.ndarray:
    """Permute the row order of one cell's series (G-ORD, CR 2.2 item 27)."""
def null_summary(observed: float, null: np.ndarray) -> dict:
    """{'observed', 'n', 'p95', 'p05', 'spread' (= p95 - p05), 'rank' (= number of null values
    strictly below observed), 'exceeds' (observed > p95, strict; ties fail), 'mean', 'std'}."""
```

Seeds: `SEED_SURROGATE = 20260916`, `SEED_LABEL_NULL = 20260917`, `SEED_ORDER = 20260918`,
`SEED_FOREST = 20260919`. All fixed; a `--seed-offset` on the driver adds to all four and is
recorded.

### 3.3 Preconditions (`gates_precondition.py`)

Inputs: `cells.csv`, every `sidecar.json` and `extract.csv`. Output `gates/preconditions.csv`
with one row per cell and columns `cell_id, role, C1, C1_apf_max, C2, C2_n_pairs, C3, C3_n_windows_8_4,
C4, C5, C6, C6_reason, C7, C8, failed_count, failed_verdict, all_hard_pass`, and
`gates/preconditions.json` with params and citation.

3.3.1 C1 to C8 re-mapped (P2 Sec. V 5.1 Plan 02; CR 2.1 item 1; `plan05_campaign/validate_campaign.py`
lines 17 to 28 and 39 for the apf_queue re-map; `plan02_validate_session.py` for the original
claims):

| Claim | Runs on the extract? | Rule | Verdict strings |
|---|---|---|---|
| C1 workload ran | yes, re-mapped | kernel cell: `apf_max >= C1_ACTIVITY_MIN` (0.02, `validate_campaign.py` line 39). Idle cell: `not applicable: control (C1 re-mapped)`, never refused (CR 2.1 item 1). | `pass`, `fail`, the not-applicable string |
| C2 snap completion | yes, re-mapped | `n_pairs >= C2_MIN_PAIRS` (8, one window at (8, 4)) | `pass`, `fail` |
| C3 degrees of freedom | yes, informational | `n_windows(n_pairs - 1, 8, 4) > 3`; reported, never gates (`plan02_validate_session.py` D-25) | `pass`, `fail` (informational; excluded from `all_hard_pass`) |
| C4 lock retries | no | `not applicable: no settle record in the retention layout` | fixed string |
| C5 producer log clean | no | `not run: producer.log is not in the trajectory; the author reads the campaign log` | fixed string |
| C6 trajectory complete | yes, re-mapped | `status == "ok"`, `seq` monotone (asserted by the extractor), `header_ncols == 66`, `header_sha256` identical across all cells (the first cell's is the reference), `n_rows_skipped == 0`; `n_seq_gaps` is reported in `C6_reason` and does not fail C6 (a gap is K = 0, 2.1) | `pass`, `fail` |
| C7 Plan 03 winner | later | `pending: filled by the temporal gates (move 6)` until `gates/selection.json` exists, then `pass` if the APF selection has `passes_acceptance == true` else `fail` | strings as stated |
| C8 Plan 04 segmenter | no | `not applicable: change-point view not in this paper (decided 2026-09-16)` (P2 Sec. 6 item 8) | fixed string |

`all_hard_pass = C1 in (pass, not applicable) and C2 == pass and C6 == pass`. A cell with
`all_hard_pass == false` is excluded from every later stage and listed in `preconditions.json`
`excluded_cells`.

3.3.2 The `failed/` count (P2 Sec. V 5.2 preconditions; CR 2.1 item 2; K2 move 2; AA A5).
`failed_verdict` per cell: `pass` if `failed_count == 0`; `refused: failed count <n> > 0, seq axis
uncorrected` if positive (the cell's `J` after the first failure is unusable; without the failure
positions the whole persistence series is refused for that cell); if `failed_count` is `null`:
`refused: failed count not recorded` unless the driver was given `--assume-failed-zero
--assume-reason "<text>"`, in which case the verdict is `pass (declared zero: <text>)` and the
text is stored in `params`. The runbook tells the author to pass `--assume-reason "AA A5: any
failed job re-runs the whole cell"`. `inputs/failed_counts.csv` (columns `cell_id, failed_count,
source`) overrides the sidecar when present.

3.3.3 G-K0 (P2 Sec. V 5.2; CR 2.2 item 20; K2 move 4).

```python
def gate_gk0(out: Path, cells: list[dict], *, tail_fraction: float = 0.80,
             idle_pool: str = "pooled_snapshots", idle_percentile: float = 95.0) -> Path:
    """G-K0, state-change disclosure. Source part: inputs/gk0_source.csv (kernel,
    steady_state_changes_content, source), template pre-filled from P2 Table 3 column 5
    ('yes' | 'no (after the first pass)' | 'only through pass phase' | 'unstated'), the
    author edits. Measured part: per cell, median K over the last `tail_fraction` of the
    rows (by seq, undropped series) against the idle band's upper edge: the
    `idle_percentile`-th percentile of K pooled over every idle cell's rows
    (idle_pool="pooled_snapshots") or of the idle cells' per-cell medians
    (idle_pool="cell_medians"). Per kernel: relabelled GK0_IDLE_MEASURED when the median of
    its cells' tail medians is inside the band (<= edge), else GK0_ABOVE_FLOOR. With no idle
    cell: not_run('no admissible idle cell'). Citation: P2 Sec. V 5.2 G-K0; CR 2.2 item 20."""
```

Output `gates/gk0.csv`: `kernel, source_statement, n_cells, tail_median_K_median,
tail_median_K_min, tail_median_K_max, idle_band_edge, verdict, archetype_measured`. A relabelled
kernel gets `archetype_measured = "IDLE"` and is removed from the denominator of every recovery
statement: the split stage uses `archetype_measured` as the LOKO label for that kernel and G-N
counts it under IDLE. The idle rows themselves get `verdict = "control"`.

3.3.4 G-F (P2 Sec. V 5.2; CR 2.2 item 21; K2 move 4 and move 14).

```python
def gate_gf(out: Path, cells, rung: str, grid_id: str, *, n_perm: int = 500,
            part1_design: str = "within_trace_window", test_frac: float = 0.2,
            part2_rule: str = "all_cells_outside") -> Path:
    """G-F, floor. grid_id: the rung's selected point from selection.json when it exists,
    else gf_default_grid = "W8_H4" (move 4 runs before move 6; move 13 re-runs at the
    selected point). Admissibility record first: inputs/idle_admissibility.json, written by
    the author (keys: same_ssh_path, rebooted_per_cell, interval_ms, differ_speed,
    image_state_note, duration_s, capture_loop_note); absent -> every G-F row is
    not_run('no admissible idle cell; admissibility record missing') and the tripwire row
    reads 'not run'. Part (i): the idle cells must be mutually inseparable under the rung.
    The definition says leave-one-rep-out on idle cells alone; with one cell per rep label a
    held-out rep can never be predicted, so the executable form is part1_design =
    'within_trace_window': label = rep, the last test_frac of each idle cell's windows is
    test, the forest of 4.2, score = window accuracy, null = 500 window-level label shuffles
    within the idle set; verdict GF_INSEPARABLE if the observed score does not strictly
    exceed the null's p95, else GF_VOID (the rung's result is void). Part (ii), per kernel:
    the kernel's per-cell medians of the rung's headline reading (3.1.5) against the
    envelope [min, max] of the idle cells' per-cell medians; part2_rule='all_cells_outside':
    every cell median outside -> pass, else GF_AT_FLOOR; 'median_of_cells_outside': the
    median of the cell medians. Reported as a finding, never as an encoding failure.
    Citation: P2 Sec. V 5.2 G-F and the tripwire clause; CR 2.2 item 21, 2.1 item 15."""
```

Output `gates/gf.csv`: `rung, grid_id, part, kernel, n_cells, score, null_p95, n_inside_envelope,
envelope_lo, envelope_hi, verdict`. Part (i) rows have `kernel = "idle"`. The three floors named
by the admissibility text (the idle cells' distributions of K, of `l0` per changed page, of J) are
written to `gates/gf_floors.json` as the five quantiles of each, pooled over idle cells.

### 3.4 Calibration (`gates_calibration.py`)

3.4.1 The pass table (`inputs/pass_table.csv`; P2 Sec. V G-P and Table 3; AA A7; CR 2.1 item 4).
Columns: `kernel, passes_per_600s, source, notes`. `write_pass_table_template(path)` pre-fills
`nbody, 6147, "declared: kernel_nbody_v2_metadata.json (AA A7)"` and the other eleven with a blank
count and `source = "undeclared"`. The author may fill a kernel's row with a count and
`source = "inferred: <reason>"` (the VME 1.5 operation-count estimate, P2 G-P) or
`"declared: <file>"`. `load_pass_table(path) -> dict[kernel, PassEntry(passes, source_kind in
{"declared", "inferred", "undeclared"}, note)]`. `T_seconds = 600 / passes`; per cell
`T_pairs = n_pairs / passes`. A verdict derived from an inferred row carries the suffix
`" (INFERRED)"`.

3.4.2 G-P (P2 Sec. V 5.2 G-P; CR 2.2 item 23; K2 Sec. 2 "readings").

```python
def gate_gp(out: Path, cells, pass_table, *, dt_bracket=(0.500, 0.644),
            min_passes_rhythm: int = 5, min_snaps_within_pass: int = 3) -> Path:
    """G-P, pass period, per cell (rolled up per kernel by majority; a kernel whose cells
    disagree is reported with the count). Pair units: T_pairs = n_pairs / passes.
    verdict_pairs: T_pairs >= 4 -> GP_RESOLVABLE; 2 <= T_pairs < 4 -> GP_MARGINAL;
    T_pairs < 2 -> GP_ALIASED; no count -> GP_UNDECLARED. rhythm_verdict: passes >=
    min_passes_rhythm -> GP_ADMITTED else GP_RHYTHM_UNDERSAMPLED. within_pass_verdict:
    T_pairs >= min_snaps_within_pass -> GP_ADMITTED else GP_PASS_ALIASED. Seconds
    representation, for the paper's bracket: verdict_dt_0500 and verdict_dt_0644 from
    T_seconds against 4 dt and 2 dt with the same three names. Undeclared kernels: every
    verdict GP_UNDECLARED; their readings stay in every table labelled so (al-Kindi over
    DSP, CR 2.2 item 23). Citation: P2 Sec. V 5.2 G-P; CR 2.2 item 23."""
```

Output `gates/gp.csv`: `kernel, cell_id, n_pairs, passes_per_600s, source_kind, T_seconds,
T_pairs, dt_est_s, verdict_pairs, rhythm_verdict, within_pass_verdict, verdict_dt_0500,
verdict_dt_0644`. On this dataset the expected content is nbody `T_pairs` about 0.15,
`GP_ALIASED`, and eleven kernels `GP_UNDECLARED`.

3.4.3 G-C (P2 Sec. V 5.2 G-C; CR 2.2 item 22; K2 move 3).

```python
def gate_gc(out: Path, cells, rung: str, *, pulse_kernel: str = "gemm",
            jump_factor: float = 2.0, jump_reference: str = "cell_median_K",
            j_dip_max: float = 0.75, dip_window_pairs: int = 1, min_events_per_rep: int = 1,
            content_kernels=("gibbs", "histogram", "gemm"), pairing: str = "by_rep_index",
            content_page_set: str = "persistent") -> Path:
    """G-C, calibration pulse, per rung. Refuses the whole rung (GC_DISCONNECTED) when the
    pulse is missing in any rep; every negative of that rung is void until fixed.
    apf: in every cell of pulse_kernel, at least min_events_per_rep snapshots with
      K_t >= jump_factor * reference (reference = the cell's median K, or the median K of
      the preceding 8 snapshots when jump_reference='local_median_8').
    persist: the apf event and, within +/- dip_window_pairs snapshots of it, J <= j_dip_max
      (the dip toward one half at the pass boundary), in every rep.
    wapf: the definition names no pulse for wAPF; the apf rule applied to the wapf series
      is the default (section 8).
    content: the orderings mean_abs(gibbs) < mean_abs(histogram) < mean_abs(gemm) and
      l0(histogram) < l0(gibbs) < l0(gemm), where a cell's statistic is the median over
      seqs of l1_q50_<set> / 4096 (mean_abs = l1 / 4096, positional.rs) and of l0_q50_<set>,
      set = 'per' (persistent pages) or 'all'; pairing='by_rep_index' compares rep r of the
      three kernels for r = 0..7 and needs 8 of 8; pairing='envelope' needs
      max(gibbs cells) < min(histogram cells) < ... over all cells.
    combined: pass iff the four rungs pass.
    Citation: P2 Sec. V 5.2 G-C; CR 2.2 item 22."""
```

Output `gates/gc.csv`: `rung, kernel_or_triple, rep, n_events, first_event_seq, j_at_event,
stat_a, stat_b, stat_c, verdict` with a final row per rung `rep = "all"` carrying `pass` or
`GC_DISCONNECTED`. The pass boundary is defined by the K jump itself; no marker is read.

3.4.4 The alias falsifier (P2 Sec. 2 falsifier (2); CR 2.2 item 23 last clause).

```python
def alias_falsifier(feature_per_cell: dict[str, float], dt_per_cell: dict[str, float],
                    *, r2_threshold: float = 0.5) -> dict:
    """Regress the per-cell feature on the cell's realized mean interval (dt_est_s) within a
    kernel (8 points). Returns slope, intercept, r2, n, verdict ALIAS_MOVES if r2 >
    r2_threshold else ALIAS_STAYS. The dt spread on this dataset is about 0.635 to 0.674 s
    (890 to 945 pairs); the result is reported with that spread. Citation: P2 Sec. 2
    falsifier (2); CR 2.2 item 23."""
```

Run by the driver for (a) every G3 cepstral peak frequency per kernel (`gates/alias.csv` rows
`kind = "g3_peak"`), and (b) any feature that separates a level-matched pair under Table 6
(`kind = "table6_feature"`), see 3.7.

### 3.5 Temporal gates (`gates_temporal.py`)

Per rung, per kernel, per grid point on the level-normalized single-channel series (for `content`
and `combined` the temporal gates run on the first channel, `r_l0_q50_per`, and this is recorded
in `params.temporal_channel`). Output per grid point `gates/grid/<rung>/<grid_id>/temporal_per_kernel.csv`
with one row per kernel (idle cells as kernel `idle`): `rung, grid_id, W, H, hop_ratio, kernel,
n_cells, n_windows_median, n_windows_min, n_windows_nonoverlap_median, stat_pass_frac_median,
g1_surrogate_p05, g1_trend_cells, G1, coverage_0500, coverage_0644, G2_0500, G2_0644, G2, G4, G5,
gord_score_ordered, gord_score_shuffled_mean, gord_null_spread, GORD`.

3.5.1 G1, stationarity with its surrogate null (P2 Sec. V 5.1 Plan 03; CR 2.1 item 3).
`stationarity_per_window` is `plan03_metric_kernel.py:stationarity_per_window` copied verbatim
(z-threshold 1.0, population std). Per cell: `pf_obs = stationarity_per_window(x, W, H)`; 200
surrogates of `x` (3.2), `pf_sur[i]` the same statistic; `trend`: the least-squares slope of `x`
on its index times `(n - 1)`, in units of the population std of `x`, exceeds
`g1_trend_drift_sd = 1.0` in absolute value. Per kernel: `G1 = pass` iff the median over cells of
`pf_obs` is `>= 0.80` and not below the median over cells of the surrogates' 5th percentile;
`G1 = TREND_PRESENT` when more than half the kernel's cells have `trend` (the cell is handed to
the whole-cell reading, not discarded; the count is `g1_trend_cells`); else `fail`. Surrogate
statistics are saved in `g1_surrogates.npz` (`cell_id`, `pf_obs`, `pf_sur` `[n_cells, 200]`).

3.5.2 G2, spectral coverage in pair units (P2 Sec. V 5.1; CR 2.1 item 4). Coverage
`= W * dt / T_seconds` at `dt in (0.500, 0.644)`, floor 2.0. Per kernel: no pass count ->
`G2 = GP_UNDECLARED` (the NaN of the code, named); `T_seconds < 2 * dt` -> `G2_ABOVE_NYQUIST` at
that dt; else `pass` if coverage `>= 2.0` else `fail`. `G2` (the roll-up) is the common verdict
when `G2_0500 == G2_0644`, else `G2_UNDETERMINED`. The old `RHYTHM_S` table of `plan03_sweep.py`
is not used. For the whole-cell point, `W = n_series` per cell and the median coverage is
reported.

3.5.3 G3, the per-kernel signal flag off the grid decision (P2 Sec. V 5.1 and Sec. 6 item 7,
option (a); CR 2.1 item 5). Not a grid column. `gates/g3_flags.csv`: `rung, kernel, cell_id,
ceps_peak_idx, ceps_peak_freq_cyc_per_pair, ceps_snr_db, snr_surrogate_p95, cv, cv_shot_floor,
cv_ratio, flag_cell, flag_kernel`. The cepstrum is `plan03_metric_kernel.py:score`'s path copied:
`rfft` of the series, `log(|.| + 1e-10)`, `irfft`, peak search over `|c[q]|` for `q >=
max(1, n // 8)` (the whole tail, as the original), `snr_db = 10 log10(peak / median(tail))`.
The quefrency floor `n // 8` (Plan 03's R2 override) means only rhythms with a period of at
least `n / 8` pairs (about 116 pairs, roughly 75 s, on a 931-pair cell) can be found; faster
rhythms are outside the search by construction, and this is written into `params.quefrency_floor`
and listed in section 8. `snr_surrogate_p95` from 200 phase-randomized surrogates; `flag_cell = G3_PRESENT` if `snr_db >
snr_surrogate_p95` (strict) else `G3_ABSENT`; `flag_kernel = G3_PRESENT` if at least
`g3_min_cells = 7` of the kernel's cells are present. `cv` is `plan02_metrics_per_cell.cv_workingset`
(population std over mean of the whole series, re-implemented in three lines and named as such);
phase randomization preserves mean and variance exactly, so no surrogate threshold exists for CV;
it is reported beside the shot-noise floor `1 / sqrt(K_median_cell)` as `cv_ratio = cv /
cv_shot_floor`, no verdict (section 8). The 4.5 dB and 0.30/0.50 ceilings are not used.

3.5.4 G4, hop validity: `H * 2 <= W` (`plan03_aggregate.py` `g4_pass`), unchanged. The whole-cell
point has `H = W` and `G4 = fail` by construction, kept and reported.

3.5.5 G5, window count, reported not gated (CR 2.1 item 7): `n_windows_median`, `n_windows_min`,
`n_windows_nonoverlap_median = median over cells of n_series // W`; `G5 = pass` when
`n_windows_min >= 5` (the numerical-anomaly catch), else `fail`; not part of the selection rule.
The Delta-5 regression guard is not applied (`params.delta5 = "not applied: no kernel counterpart
(CR 2.1 item 8)"`).

3.5.6 G-ORD, the time-shuffle null (P2 Sec. V 5.1; CR 2.2 item 27). Per rung and per `W` (at
`H = W // 2`, the middle hop ratio, and at the whole-cell point), split LOKO, label space
archetype, the forest of 4.2: `score_ordered`; then `gord_n_order_perm = 20` order shuffles per
cell (3.2, one permutation drawn per cell per repetition), re-windowed, the same split, the mean
`score_shuffled`; and a label-shuffle null of `gord_null_perm = 100` permutations at that `W` for
its spread only. `GORD = GORD_ORDER_BLIND` if `|score_ordered - score_shuffled| <= spread`
(`spread = p95 - p05` of the null) else `GORD_RESOLUTION`. Written to `gord.json` under the
`H = W // 2` grid directory and copied into the `GORD` column of every grid point that shares
that `W` (the same value on every kernel row; the test is per encoding and per `W`, not per
`H`). It re-labels the `W` axis of Table 5 and refuses nothing.

3.5.7 Roll-up and selection (`gates/table5_long.csv`, `gates/table5_grid.csv`,
`gates/selection.json`). `table5_long.csv` is the concatenation of every per-kernel CSV (one row
per `(rung, kernel, grid_id)`; 5 rungs x 12 kernels (13 with the idle row when idle cells
exist) x 13 grid points). `table5_grid.csv` has one
row per `(rung, grid_id)`: `rung, axis, grid_id, W, H, hop_ratio, n_windows_median,
n_windows_nonoverlap_median, G1, G2_0500, G2_0644, G2, G4, G5, GORD, gates_passed, selected,
selected_by, refusal`. Roll-up rule `grid_rollup =
"all_kernels"`: a gate passes at a grid point when it passes for every kernel that is not
`GK0_IDLE_MEASURED` and for which the gate is applicable (G2 applies only to kernels with a
declared or inferred count; `GP_UNDECLARED` kernels do not block G2 and G2 is then
`not applicable: no kernel with a declared pass period` when none has one); `"majority"`: more than
half of the applicable kernels. Selection per rung (from `plan03_aggregate.py:_pick_winner`,
re-pointed): the smallest integer `W` whose point passes every applicable gate among G1, G2, G4
(G5 reported; G-ORD a label), tie-break `hop_ratio` closest to 0.5; when none passes, the point
passing the most gates with the same tie-breaks, `selected_by = "best-feasible"` and
`passes_acceptance = false` in `selection.json`. `selection.json`: `{rung: {grid_id, W, H,
passes_acceptance, selected_by, gates_passed, refusal}}`. Every grid point stays on disk; the driver
refuses to start the split stage for a rung unless all 13 `gates/grid/<rung>/<grid_id>/` directories
exist with their CSV (`grid_complete.json` records the check). Nothing is ever deleted from
`gates/grid/`.

### 3.6 Readings (`gates_readings.py`)

3.6.1 G-J (P2 Sec. V 5.2 G-J and Sec. 6 item 6; CR 2.2 item 31; K2 Sec. 2 rung 1 (a)).

```python
def gate_gj(out: Path, cells, *, k_factor: float = 3.0) -> Path:
    """G-J, the persistence null. Two nulls per pair: the independence null J_null (extract
    column) and the idle cells' own J distribution (empirical floor null: the five quantiles
    of J pooled over idle cells, to gates/gj.json). Mask: a kernel's J is interpretable
    only at pairs where K_t > k_factor * (the floor's median K, pooled over idle cells'
    rows); pairs below carry GJ_FLOOR_OVERLAP. With no idle cell every pair is
    GJ_FLOOR_UNMEASURED and J is reported unmasked and labelled. No floor subtraction
    (decided 2026-09-16, P2 Sec. 6 item 6). Per cell summary: mean J, the five quantiles,
    fraction of pairs interpretable, fraction of interpretable pairs with J <= J_null (the
    'below null' fraction; the definition names 'a declared null-relative threshold' and this
    toolkit declares J <= J_null), G-P verdict beside it. Citation: P2 Sec. V 5.2 G-J;
    CR 2.2 item 31."""
```

Output `gates/gj.csv`: `kernel, cell_id, n_pairs_J, floor_median_K, k_threshold,
frac_interpretable, J_mean, J_q05, J_q25, J_q50, J_q75, J_q95, J_null_q50, frac_below_null,
gp_verdict_pairs, mask_verdict`. The per-pair mask is saved as `gates/gj_mask/<cell_id>.npy`
(bool per series row) for the fused plane and the persistence features (the split stage does not
apply the mask; the figures and the per-cell summaries do; section 8).

3.6.2 G-DEC (P2 Sec. V 5.2 G-DEC; CR 2.2 item 34; K2 Sec. 2 rung 2 (e)).

```python
def gate_gdec(out: Path, cells, pass_table, gp_csv, *, kernel: str = "floyd",
              control_kernel: str = "gibbs", boundary_source: str = "k_jump",
              jump_factor: float = 2.0, min_run: int = 3, min_reps: int = 7,
              n_surrogates: int = 200) -> Path:
    """G-DEC, decay validity, for one kernel (default floyd, the decay exhibit) with gibbs
    as the no-slope control. Admission: the kernel's within_pass_verdict must be
    GP_ADMITTED, else GDEC_NOT_RESOLVED for every rep. Pass boundaries: boundary_source =
    'k_jump' (snapshots where K_t >= jump_factor * cell median K, the G-C detector) or
    'period' (every round(T_pairs) rows from the first jump). (a) the series is the
    per-snapshot median l0 over persistent pages (l0_q50_per), l1_q50_per second, hamming
    as a direction check (sign of its slope reported); (b) inside a pass the series falls
    strictly across at least min_run consecutive snapshots starting at the same phase
    (offset from the boundary) and resets upward at the next boundary, in at least
    min_reps of 8 reps; (c) over the same span the relative drop of K is not larger than
    the relative drop of median l0, else GDEC_NO_BEYOND_BREADTH; (d) a time-block-shuffle
    surrogate: the snapshot order is permuted within each pass block (blocks of one pass,
    n_surrogates times), the mean within-pass slope recomputed; the observed slope must be
    below the surrogates' 5th percentile (more negative), else GDEC_NO_DECAY; (e) the idle
    cells' whole-cell slope of the same series must not have the same sign while exceeding
    its own phase-randomized surrogate 95th percentile in magnitude, else
    GDEC_NO_BEYOND_FLOOR (with no idle cell, (e) is not_run and the verdict carries the
    suffix ' (floor unmeasured)'). The control kernel is reported with the same columns and
    the expectation 'no slope'. Citation: P2 Sec. V 5.2 G-DEC; CR 2.2 item 34."""
```

Output `gates/gdec.csv`: `kernel, cell_id, role_in_test (exhibit|control|idle), n_passes,
snaps_per_pass_median, phase_offset, run_length, slope_l0, slope_l1, hamming_sign, k_rel_drop,
l0_rel_drop, surrogate_p05_slope, verdict`, plus one summary row per kernel with `cell_id = "all"`.

### 3.7 Comparisons (`gates_comparison.py`)

All comparisons run at the selected grid point per rung (`selection.json`), on the `norm`
features, and (Table 6) on the `raw` features for APF. The split stage (4.5) has already written
`scores.json` per `(rung, grid_id, split, labelspace)`; the comparison gates read those.

3.7.1 B1-G1 at the unit (P2 Sec. V 5.1 Plan 08; CR 2.1 item 9). `n_perm = 500`, unit-level
shuffle (3.2), the same folds, the same model and seed, score = unit accuracy; `null.json`:
`null_summary` plus the 500 scores. Verdict in `scores.json["b1_g1"]`: `pass` when
`exceeds` (strict) else `NEAR_UNFALSIFIABLE`; a `NEAR_UNFALSIFIABLE` row is written to
`gates/excluded_rows.csv` and never printed in a table. The rank is reported as
`"rank r of 500"`. A run with fewer than 500 permutations (a smoke run) writes
`b1_g1 = "not run: N permutations < 500"`; the row is printed with that string in its null and
rank cells and is not admissible for the paper (only `NEAR_UNFALSIFIABLE` rows are excluded).

3.7.2 B1-G3 restated (CR 2.1 item 10). Per split and label space: for each feature, a one-feature
threshold model (4.3) fitted on the training fold; the feature whose model reproduces the full
model's per-unit predictions on all but at most `b1g3_max_disagree = 1` units (over all folds) is
quarantined: `l1_quarantine.json` lists `{feature, n_disagree, n_units}`; the split is re-run
without the quarantined features and both scores are kept (`scores.json["with_quarantine"]`).
Table rows use the re-run.

3.7.3 B1-G6 at the unit (CR 2.1 item 11). `majority`: LOKO/archetype: the most populous
archetype by kernel count among the training kernels of each fold, scored on the held-out kernel
(so 6/12 = 0.5 overall on this corpus before G-K0); LORO and within-trace, kernel space: the most
populous kernel by cell count (1/12 with eight per kernel); archetype space under LORO or
within-trace: the most populous archetype by cell count. Written to `scores.json["majority"]`.

3.7.4 G-L (P2 Sec. V 5.2 G-L; CR 2.2 item 24). Part (i): per rung, the normalized LOKO/archetype
score against its B1-G1 null p95; `pass` or `GL_LEVEL_ONLY` (the rung's Table 7 row is marked
and the rung cannot be cited in the recovery clause; the raw score, where it exists, is reported as
the level-inclusive ceiling). Part (ii): at the selected APF point, per cell the mean over windows
of the within-window CV of the raw APF series; per kernel the mean of that and of
`1 / sqrt(K_median_cell)`; ordinary least squares across the 12 kernels (`gl2_level = "kernel"`;
`"cell"` uses the 96 cells); `r2 > 0.5` -> `GL_SHOT_NOISE` (the level-normalized comparison is
refused until `cov` and its relatives are dropped: the driver then re-runs the split stage with
`feature_drop = ("cov", "std", "peak2med")` and reports both); else `pass`. Output `gates/gl.csv`:
`rung, part, score_norm, null_p95, r2, slope, n_points, verdict`.

3.7.5 G-N (P2 Sec. V 5.2 G-N; CR 2.2 item 25). Per archetype row (after G-K0's relabelling):
`n_kernels >= 3` -> `GN_HEADLINE`; `2` -> `GN_ONE_TRAIN`; `1` -> `GN_NOVELTY`; `0` -> `GN_NO_ROW`.
Macro recall over the headline rows only; a per-row recall with `n` for the others; no average
over rows that are not headline. Output `gates/gn.csv`: `archetype, n_kernels, kernels, status`.

3.7.6 G-X, the blind campaign-leak test (P2 Sec. V 5.2 G-X; CR 2.2 item 26; K2 Sec. 4 item 3).
The same forest, the same normalized features at the selected point of each rung, label =
`campaign` (three labels: `dwarfs1`, `01c`, `01c1`), split LOKO, unit accuracy, null = 500
campaign-label shuffles across cells; `GX_LEAK` if the score strictly exceeds p95 else
`GX_POOLING_STANDS`. Then the confound record regardless: for every archetype with at least two
kernels, the set of campaigns its kernels sit in; `GX_CONFOUND_TOTAL` if some archetype's kernels
all sit in one campaign while another archetype's all sit in a different one, `GX_CONFOUND_PARTIAL`
if any archetype's kernels sit in exactly one campaign, else `GX_CONFOUND_NONE`; and under LORO the
held-out cell's campaign is written into `predictions.csv`. When `GX_LEAK` and
`GX_CONFOUND_TOTAL` both hold, the LOKO archetype headline is marked
`refused: campaign leak with total confound` in Table 7. Output `gates/gx.csv` (`rung, score,
null_p95, rank, leak_verdict, confound_verdict`) and `gates/gx.json` (the per-archetype campaign
sets and the cell order columns as recorded inputs `inputs/cell_order.csv`, optional:
`cell_id, launch_label, order_in_launch`).

3.7.7 G-DIM (P2 Sec. V 5.2 G-DIM; CR 2.2 item 32). Every Table 7 row carries its feature count.
The combined rung is accompanied by a feature-count-matched comparison: `d* = the feature count
of the strongest single rung by LOKO score`; the combined vector is reduced to `d*` per fold by
`dim_match_method = "train_importance"` (rank features by the forest's impurity importance fitted
on the training fold only, keep the top `d*`) or `"pca"` (PCA fitted on the training fold, `d*`
components); the reduced score is the `combined (matched)` row. A rung whose `d` exceeds the
number of training cells in a fold runs only with the same reduction to `d = n_train_cells`
(`GDIM_REDUCED`), else `GDIM_FULL`. Output `gates/gdim.csv`: `rung, d, d_matched, method,
status`.

3.7.8 G-M (P2 Sec. V 5.2 G-M; CR 2.2 item 33). Margin: the APF rung's LOKO score re-run with
`gm_n_seeds = 5` forest seeds (`SEED_FOREST + i`); `spread = max - min` of the five scores (fold
assignment is fixed by the kernels under LOKO and by the cells under LORO; the seed is the only
source of spread there, and within-trace uses the fixed last-20-percent tail; section 8). For a
pair of rungs (A, B) on one split: `diff = score_A - score_B`; the sign test on the 12 per-kernel
outcomes (per-kernel recall under A minus under B; a tie is neither): `GM_BEATS` when `diff >
spread` and (`improving >= 6 and worsening == 0` or `improving >= 7 and worsening <= 1`); else
`GM_DIFFERENCE`. Output `gates/gm.csv`: `split, rung_a, rung_b, score_a, score_b, diff, spread,
improving, worsening, ties, verdict`, for every ordered pair of rungs.

### 3.8 G-V (`variance.py`)

P2 Sec. V 5.2 G-V; CR 2.3 item 35. On the normalized features at the selected point of each rung,
the per-cell vector is the mean over the cell's windows. Per feature: `L0` = mean over kernels of
the variance across the kernel's 8 cells; `L2` = mean over archetypes with at least two kernels of
the variance across the archetype's kernel means; `L3` = variance across archetype means
(archetypes with at least one kernel, after G-K0 relabelling). Variances are population variances.
A rung with `L0 > L3` for every feature is `GV_NOT_ESTIMABLE`, else `GV_ESTIMABLE`. Output
`gates/gv.csv`: `rung, feature, L0, L2, L3, L0_over_L3` and `gates/gv_summary.csv`: `rung,
n_features, n_features_L0_gt_L3, verdict`.

---

## 4. Splits and models (builder 2)

### 4.1 Splits (`splits.py`)

Unit = cell; whole cells are held out; no cell's windows straddle train and test (P2 Sec. V
"The splits"). `fold_within_trace`, `fold_loro`, `fold_loko` are `b1_splits.py`'s
`fold_within_trace`, `fold_loro`, `fold_lowo` copied, with `workload -> kernel` and `family ->
archetype` renamed, plus `_assert_grouped` run on every fold list before use.

| split | folds | held out | label spaces run | role of the score |
|---|---|---|---|---|
| `within_trace` | 1 | the last `test_frac = 0.2` of every cell's windows (chronological) | `kernel`, `archetype` | ceiling only, never the headline |
| `loro` | one per cell (96 on this corpus; `loro_mode = "cell"`; `"rep_index"` holds out all cells of one rep index, 8 folds) | one cell | `kernel`, `archetype` | an instrument's reading |
| `loko` | one per kernel (12) | all cells of one kernel | `archetype` only (`kernel` is impossible: the held-out label is unseen, `not applicable: held-out label unseen`) | the headline |

At the whole-cell grid point every cell has one window, so `within_trace` cannot hold out a
tail: its rows read `not applicable: one window per cell`; `loro` and `loko` run on the 96
one-row cells.

Idle cells (`role == "idle"`) are excluded from every split's training and test sets unless the
split is run with `include_idle = True` (Table 8's IDLE (measured) column uses G-K0's relabelling
of kernel cells, not the idle cells themselves). The LOKO label of a kernel is
`archetype_measured` from `gk0.csv` when it exists, else `archetype_predicted`.

### 4.2 The forest (`models.py`)

```python
def make_forest(seed: int = SEED_FOREST, n_jobs: int = 1) -> Pipeline:
    """Pipeline(SimpleImputer(strategy='median'), StandardScaler(),
    RandomForestClassifier(n_estimators=300, max_depth=None, max_features='sqrt',
    min_samples_leaf=1, class_weight=None, bootstrap=True, random_state=seed,
    n_jobs=n_jobs)). n_estimators=300 and the imputer+scaler wrapping follow
    plan04_classify.py line 131 and plan05_campaign/peakvar_lift.py line 41; the scaler is
    fitted on the training fold only (b1_ae.py's rule). Citation: P2 Sec. V 'Models'."""
```

Fit on window rows of the training cells; predict every window of the test cells; the unit
prediction is the majority vote over the cell's windows (`unit_aggregation = "cell_majority"`;
ties broken by the higher mean predicted probability, then by class name order;
`"cell_mean_proba"` takes the argmax of the mean probability). Scores at the unit:
`accuracy` (fraction of test cells correctly labelled), `recall_per_class`, `macro_recall`
(over G-N headline classes only when the label space is `archetype`; over all classes in
`kernel` space), `recall_per_kernel` (fraction of the kernel's cells correct; the row unit of
Tables 6 and 8).

### 4.3 The one-feature threshold model (B1's L1)

`make_l1(n_classes) -> DecisionTreeClassifier(max_depth=None, max_leaf_nodes=n_classes,
random_state=SEED_FOREST)` on one feature column, the feature chosen on the training fold by
training accuracy (B1 doc, model L1: "single-feature threshold, feature picked on the training
fold only"). With more than two classes a single threshold cannot express the labels, so the tree
carries at most `n_classes - 1` thresholds; this is the restatement's executable form (section 8).

### 4.4 Clustering

`cluster_cells(Xcell, k, algo="kmeans", seed=SEED_FOREST) -> labels`: per-cell vectors (mean over
windows of the normalized features at the selected point, standardized over cells), `k` = the
number of predicted archetypes present among the kernel cells after G-K0 (4 on this corpus: 5 if
any kernel is relabelled IDLE), primary algorithm `KMeans(n_clusters=k, n_init=10,
random_state=seed)`; alternatives `"gmm"` (`GaussianMixture(n_components=k, n_init=5,
random_state=seed)`) and `"agglomerative"` (`AgglomerativeClustering(n_clusters=k,
linkage="ward")`) are run and written but the primary is the one in Table 8. `ari_nmi(labels_true,
labels_pred)` from `sklearn.metrics`; the unit-level null: 500 archetype-label permutations across
kernels (cells inherit), ARI and NMI recomputed against the fixed clustering; `null_summary` on
each. Output `gates/clustering.csv`: `rung, algo, k, ari, ari_null_p95, ari_rank, nmi,
nmi_null_p95, nmi_rank, exceeds_ari, exceeds_nmi`, and `gates/clustering.json` with the per-cell
cluster labels and the cluster-by-predicted-archetype count matrix.

### 4.5 The split stage (`models.run_split_stage`)

```python
def run_split_stage(out: Path, rung: str, grid_id: str, split: str, labelspace: str, *,
                    normalized: bool = True, n_perm: int = 500, seed: int = SEED_FOREST,
                    n_jobs: int = 1, feature_drop: tuple[str, ...] = (),
                    include_idle: bool = False) -> Path:
```

Writes `gates/splits/<rung>/<grid_id>/<split>__<labelspace>/`:
- `predictions.csv`: `cell_id, kernel, archetype, campaign, rep, fold, y_true, y_pred,
  n_windows, vote_fraction, held_out_campaign` (the last for LORO);
- `scores.json`: `schema, params, citation, accuracy, macro_recall, recall_per_class,
  recall_per_kernel, majority, b1_g1 (verdict), b1_g1_rank, null_p95, with_quarantine
  {accuracy, ..., quarantined_features}, feature_count, dim_status, seed, n_perm`;
- `null.json`: the `null_summary` and the array of permuted scores;
- `l1_quarantine.json`.

The stage is run for every rung at its selected point in `norm` and, for `apf`, also in `raw`; for
every split and admissible label space; by the driver (section 7). The G-ORD, G-F (i), G-X and
G-M runs call the same function with their own labels and parameters and write under
`gates/grid/...`, `gates/gf_runs/`, `gates/gx_runs/`, `gates/gm_runs/` respectively.

Cost note for the runbook: one forest fit on about 22,000 windows (W = 8, H = 4, 96 cells) and 60
features takes seconds; B1-G1 at 500 permutations costs 500 x (folds) fits per (rung, split):
about 6,000 fits under LOKO and about 48,000 under LORO (`loro_mode = "cell"`). The runbook prints
these counts before starting and the driver accepts `--null-splits` (default all three) and
`--n-jobs`.

---

## 5. The synthetic generator (builder 1, `synth.py`)

### 5.1 One cell

```python
@dataclass
class SynthSpec:
    name: str                       # kernel name used in the path, e.g. "gemm"
    seed: int
    n_pairs: int = 120
    N: int = 262144
    K0: int = 2048                  # working-set size (breadth level)
    k_noise: float = 0.02           # per-snapshot relative jitter of K (uniform)
    churn: float = 0.05             # fraction of the changed set replaced by fresh pages each snapshot
    pulse_period: int | None = None # pass length in pairs; None = no pulse
    pulse_extra: int = 0            # extra pages A lit at a boundary (K jumps by pulse_extra)
    content: str = "double"         # "counter" | "spin" | "double" | "decay" | "idle"
    counter_step_max: int = 60      # content="counter": |byte difference| uniform on 1..counter_step_max
    spin_bytes: tuple[int, int] = (2, 8)   # content="spin": l0 per page uniform on this range, |diff| = 1
    double_words: int = 64          # content="double"/"decay": mean number of re-randomized 8-byte words per page
    decay_factor: float = 0.7       # per-snapshot multiplier of the median l0 inside a pass (content="decay")
    floor_F: int = 150              # idle-like fixed set size added to every snapshot (0 = none)
    floor_churn: float = 0.02       # churn of the floor set
    trend: float = 0.0              # linear drift of K over the cell, as a fraction of K0
    seq_first: int = 1
    gap_seqs: tuple[int, ...] = ()  # snapshots emitted with zero rows (K = 0)
    label: str = "synth"
```

Content models per changed page (integers, consistent with the differ's units:
`1 <= l0 <= 4096`, `l1 >= l0`, `hamming <= 8 * l0`, `linf <= 255`, `l2 <= l1`, `mean_abs = l1 /
4096`; every other column 0):

Every changed byte is generated as an actual pair of byte values `(a, b)` with `a != b`, so
`l1` accumulates `|a - b|`, `hamming` accumulates `popcount(a XOR b)`, `linf` is the largest
`|a - b|`, `l2` is `sqrt(sum (a - b)^2)`, and the differ's identities hold exactly.

| `content` | `l0` per page | the byte pair `(a, b)` | expected per-page ratios |
|---|---|---|---|
| `counter` | 1 or 2 (a bin count bumped many times between snapshots) | `a` uniform, `b = (a + d) mod 256` with `d` uniform on `1..counter_step_max` | `l0/4096` about 4e-4, `l1/l0` about 30, `hamming/l0` about 2 to 3 |
| `spin` | uniform on `spin_bytes` (a few one-byte state flips) | `a` uniform, `b = a +/- 1` | `l1/l0` = 1, `hamming/l0` about 2 |
| `double` | `8 x Poisson(double_words)` clipped to 4096 (re-randomized doubles) | `a`, `b` independent uniform bytes conditioned on `a != b` (mean `\|a - b\|` about 85.7, mean popcount about 4.0) | `l1/l0` about 86, `hamming/l0` about 4 |
| `decay` | as `double` at the first snapshot after a boundary, then the median scaled by `decay_factor` each snapshot within the pass | as `double` | within-pass decreasing median `l0`, reset at the boundary |
| `idle` | 1 to 4 | as `counter` with `counter_step_max = 8` | floor-like |

With these models the calibration orderings of G-C hold by construction on the corpus of 5.2:
`mean_abs` (that is, `l1` per page) gibbs (`spin`, about 5) < histogram (`counter`, about 45)
< gemm (`double`, about 44,000), and `l0` histogram (about 1.5) < gibbs (about 5) < gemm
(about 512).

Set dynamics, per snapshot `t`: the changed set is `W_t ∪ F_t` where `W_t` is the working set
(size about `K0 (1 + trend * t / n_pairs)` with jitter) evolving by replacing a `churn` fraction
with pages not currently in the set (uniform over `0..N-1`), and `F_t` the floor set (size
`floor_F`, churn `floor_churn`). At a boundary (`t % pulse_period == 0`, `t > 0`) the set is
`W_t ∪ A_t ∪ F_t` with `A_t` `pulse_extra` fresh pages, so `K` jumps by `pulse_extra` and
`J(t-1)` dips to about `|W| / (|W| + |A|)` (the gemm re-seed of K2 Sec. 2 rung 1 (b)); `A_t` is
dropped at `t + 1`. The generator records everything.

Outputs of `write_cell(spec, root, *, compress=True) -> Path`: the directory
`<root>/kernel/kernel_<name>_v2/<sig>/rep001__<label>/` with `sig =
f"--seed_{seed}_--duration_{n_pairs}"` (a seed-bearing signature so `parse_cell_path` finds the
seed), the trajectory `run_matrix_test1_kernel_<name>_v2.npy.substrate_trajectory.csv.zst`
(plain `.csv` when `zstd` and `zstandard` are both unavailable or `compress=False`) with the exact
66-column header of 2.1, and `truth.json`:

```
{"schema": "plan11.synth_truth.v1", "spec": {...}, "seq": [...], "K": [...],
 "J": [...] (null at the last), "n_persist": [...], "boundaries": [seqs],
 "sets": null or [[sorted page lists]] when n_pairs * K0 <= 2e6,
 "per_seq_channels": {"l0_q50_all": [...], "l1_q50_all": [...], "ham_q50_all": [...],
                      "l0_q50_per": [...], ...},   # every extract column, computed by the generator
 "expected": {"J_steady": (1 - churn) / (1 + churn), "J_at_boundary": K0 / (K0 + pulse_extra),
              "hamming_over_l0": ..., "l1_over_l0": ...}}
```

`truth.json` holds every extract column recomputed by the generator from its own arrays, so
`tests/test_extract.py` asserts equality of `extract.csv` with `truth.per_seq_channels` to
`1e-9` on every column and every row, including gaps and the last row's blanks.

### 5.2 The corpus

`synth.py corpus --root <dir> [--n-pairs 120] [--reps 8] [--idle 8] [--seed 20260916]` writes
12 kernels x `reps` cells plus `idle` idle cells with the presets:

| kernel | preset |
|---|---|
| gemm | `K0 = 4096, pulse_period = 24, pulse_extra = 4096, churn = 0.02, content = double` (the pulse kernel; 24 is above the cepstral quefrency floor `n // 8 = 14` at 120 pairs, 3.5.3) |
| floyd | `K0 = 2048, pulse_period = 10, content = decay, decay_factor = 0.7` |
| gibbs | `K0 = 256, content = spin, churn = 0.03` (no pulse, no decay: the control) |
| nbody | `K0 = 2048, content = double, churn = 0.01` |
| spmm | `K0 = 1024, content = double, churn = 0.05` |
| stencil_jacobi | `K0 = 3072, content = double, churn = 0.01` |
| fft | `K0 = 4096, content = double, churn = 0.01` |
| histogram | `K0 = 2048, content = counter, churn = 0.01` |
| fem_assembly | `K0 = 8256, content = double, churn = 0.10` |
| lexer | `K0 = 0, floor_F = 150, content = idle` (at floor) |
| rmat_gen | `K0 = 512, content = double, churn = 0.30, trend = 0.5` |
| bnb_tsp | `K0 = 4500, content = double, churn = 0.15, k_noise = 0.4` |
| idle (x8) | `K0 = 0, floor_F = 150, floor_churn = 0.02, content = idle`, `label = "idle"`, `name = "sleep"` (so `role = "idle"` by the marker rule) |

Rep seeds follow AA A4: rep 0 seed 42, reps 1..7 `1000 * r + base` with a per-kernel base from
the kernel name's index. The corpus makes every calibration ordering true (5.1), so G-C passes on
it; `synth.py corpus --break-pulse` writes the gemm cells without the pulse so G-C must refuse;
`--break-order` swaps gibbs's and gemm's content presets so the content ordering must fail.
`--cv-case random` (default) draws `k_noise` per kernel from `Uniform(0.01, 0.10)` independent of
`K0`; `--cv-case shot` sets `k_noise = 2 / sqrt(K0)` per kernel (the G-L (ii) refusal case).
`--level-only` gives every kernel the gemm preset without the pulse and a per-archetype `K0`
(2048, 4096, 8192, 16384), so only breadth carries the label. `--one-preset` gives every kernel
the same preset and only the seed differs (the unfalsifiable corpus).

### 5.3 Cases with a known answer, one pair per gate

Builder 2's tests build each case with `synth.py` and assert the verdict strings exactly.

| Gate | must pass | must refuse |
|---|---|---|
| extract J | any cell: `extract.J == truth.J` | (n/a) |
| C1 | kernel cell with `K0 >= 0.02 N` (6,000) | kernel cell with `K0 = 100` -> `fail`; idle cell -> the not-applicable string, never `fail` |
| C2 / C6 | `n_pairs = 120` | `n_pairs = 4` -> C2 `fail`; a trajectory with a decreasing seq (written by `synth.py --corrupt seq_reverse`) -> extractor `refused: seq not monotone` -> C6 `fail` |
| failed count | `--failed-count 0` | `--failed-count 2` -> the refused string; `null` without the assume flag -> `refused: failed count not recorded` |
| G-K0 | histogram preset vs the idle cells -> `above floor` | lexer preset -> `IDLE, measured`; no idle cells -> `not run: no admissible idle cell` |
| G-F (i) | 8 idle cells from one preset with different seeds -> `inseparable at floor` | 8 idle cells with `floor_F = 150 + 40 * rep` -> `void: idle reps separable under this rung` |
| G-F (ii) | gemm vs idle -> `pass` | lexer preset vs idle -> `at floor in this lead` |
| G-C apf/persist | corpus gemm -> `pass` | `--break-pulse` -> `disconnected lead` |
| G-C content | corpus -> `pass` | `--break-order` -> `disconnected lead` |
| G-P | pass table `gemm, 10` with `n_pairs = 120` -> `resolvable`; `gemm, 40` -> `marginal`; `gemm, 100` -> `aliased by design at this size` | blank -> `undeclared`; `passes = 3` -> `rhythm under-sampled` |
| G1 | stationary cell -> `pass` | `trend = 2.0` -> `trend present`; a series with the first half at `K0` and the second at `3 K0` (`synth.py --step`) -> `fail` |
| G2 | `T_seconds = 6.0` (passes = 100), `W = 32` -> coverage 2.67 / 3.43 -> `pass` | passes = 6147 -> `not applicable, rhythm above Nyquist`; `passes = 300`, `W = 8` -> 2.0 at 0.500 and 2.58 at 0.644 -> `pass` both; `passes = 240` -> 1.6 / 2.06 -> `undetermined by the interval calibration` |
| G3 flag | `pulse_period = 24, pulse_extra = 4096, n_pairs = 120` -> `rhythm flag: present` | no pulse -> `rhythm flag: absent`; `pulse_period = 8` (below the quefrency floor) -> `rhythm flag: absent`, which documents the floor |
| G4 | `H = W/4` -> `pass` | `H = W` -> `fail` |
| G-ORD | a cell set whose class difference is the pulse period only (`synth.py --ord-case`: two archetypes with identical K0 and content, periods 6 and 12) -> `resolution` at `W >= 16` | classes differing in `K0` only -> `order-blind` |
| G-J | idle cells present -> `interpretable` on gemm pairs, `floor overlap` on lexer pairs | no idle cells -> `floor unmeasured` |
| G-DEC | floyd preset (`pulse_period = 10`, 120 pairs) with `passes = 12` declared -> `decay`; gibbs -> `no decay` | floyd undeclared -> `decay not resolved`; floyd with `K` also decaying inside the pass (`synth.py --k-decay`) -> `no decay beyond breadth`; idle cells generated with `trend = -0.5` on `l0` -> `no decay beyond floor or host` |
| B1-G1 | the corpus under LOKO (distinct presets) -> `pass` | `--one-preset` corpus -> `near_unfalsifiable` |
| B1-G3 | corpus -> empty quarantine | `--level-only` corpus on the raw APF features -> the `apf.k_over_n.mean` feature quarantined |
| B1-G6 | majority `0.5` under LOKO on the 12-kernel corpus | (reported, never refuses) |
| G-L (i) | corpus -> `pass` | `--level-only` corpus -> `level only` |
| G-L (ii) | `--cv-case random` corpus -> `pass` | `--cv-case shot` corpus -> `refused: shot noise explains CV` |
| G-N | 6/3/2/1 kernels -> `headline, headline, one training kernel per fold, structural novelty` | (a labelling, never refuses) |
| G-X | labels assigned round-robin across kernels -> `pooling stands` | one archetype's kernels all labelled `01c` and another's all `01c1` with a per-campaign `K0` offset -> `campaign predictable` and `confound: total` |
| G-DIM | `d = 60 < n_train_cells` -> `full vector` | a corpus of 20 cells -> `declared reduction` |
| G-M | rung A a perfect separator, rung B at chance -> `beats` | two rungs from the same features with different seeds -> `difference with margin` |
| G-V | corpus -> `estimable` | `--one-preset` corpus -> `LOKO not estimable` |
| clustering | corpus -> `exceeds_ari = true` | `--one-preset` corpus -> `false` |

---

## 6. The tables and figures (builder 3)

Every table is written as CSV (`report/tables/<name>.csv`) and as a LaTeX fragment
(`report/tables/<name>.tex`, a `tabular` inside a `table` environment with `\caption{}` and
`\label{}` left empty and a `% columns:` comment line). Verdict cells print the verdict string;
a number that is undefined prints `--`. No prose anywhere.

### 6.1 Table 5, gate verdicts per rung at each grid point (P2 Sec. 4 Table 5)

From `gates/table5_grid.csv` and `gates/g3_flags.csv`. Columns, in order:
`encoding, axis, grid_point (W x H), n_windows (median), G1, G2 (0.500 s), G2 (0.644 s), G4, G5
(n non-overlapping windows), G-ORD, selected, refusal`. `axis` from Table 2 (`breadth`,
`breadth x amount folded`, `identity over time`, `amount`, `all three`). One row per (encoding,
grid point), 65 rows; the selected point's row carries `selected = "selected"` and the others
`""`. A companion `table5_g3.csv/.tex`: `encoding, kernel, cepstral SNR (dB), surrogate p95,
flag, CV, CV / shot floor`.

### 6.2 Table 6, APF alone (P2 Sec. 4 Table 6; Sec. VII)

Rows: one per kernel, then one per archetype (after G-K0 relabelling), then `all`. Columns:
`row, n (cells or kernels), within-trace raw, within-trace norm, LORO raw, LORO norm, LOKO raw,
LOKO norm, null p95 (LOKO norm), majority (LOKO), rank (LOKO norm), G-N status, G-X`. Kernel
rows: per-kernel recall in kernel space (within-trace, LORO) and the fraction of the kernel's
cells assigned to its archetype under LOKO; archetype rows: per-class recall under LOKO and the
mean of member kernels' recalls under the other two; `all`: accuracy. `G-X` is the one leak verdict
repeated on the `all` row, `""` elsewhere. Level-matched sets get a marker column `level set`
(`A` for floyd/histogram/nbody, `B` for fft/gemm). A refused split (B1-G1 `near_unfalsifiable`)
prints the string in every score cell of that split.

### 6.3 Table 7, rungs compared at gated resolution (P2 Sec. 4 Table 7)

Rows: `apf, wapf, persist, content, combined, combined (matched)`. Columns: `rung, resolution
(W x H), feature count, split, accuracy, macro recall (headline rows), null p95, rank, majority,
G-L, G-DIM, G-M vs APF`. One row per (rung, split) for the three splits, archetype space for LOKO
and kernel space for the others (and archetype space rows for LORO and within-trace appended
below, marked). `G-M vs APF` is the G-M verdict of `(rung, apf)` on that split with the margin in
parentheses as text (`"beats (diff 0.17 > spread 0.04; 8 up, 0 down)"`).

### 6.4 Table 8, store-predicted against state-measured (P2 Sec. 4 Table 8)

Rows: `IDLE (0 predicted), WORKING-SET (6), SCATTER (3), SEQUENTIAL-GROW (2), FRONTIER-CHURN (1)`.
Columns: `IDLE (measured), WORKING-SET, SCATTER, SEQUENTIAL-GROW, FRONTIER-CHURN`, then
`clusters (k = ...)` as `c0:n c1:n ...`, then `physical reason` (empty text column, the
author's). A cell holds the count of kernels whose LOKO assignment (majority archetype over the
kernel's 8 cells, from the combined rung's `predictions.csv` at its selected point;
`table8_rung = "combined"`, alternative any rung) is that column, written as `n (kernel, kernel)`.
The IDLE (measured) column holds the kernels relabelled by G-K0. Read as assignments with counts,
never as a confusion matrix in the statistical sense (the caption comment says so).

### 6.5 The G-V table

From `gates/gv.csv` and `gv_summary.csv`: `rung, feature, L0 (within-kernel), L2
(within-archetype), L3 (between-archetype), L0/L3`, sorted by rung then `L0/L3` descending, with
a summary row per rung carrying the `estimable` / `LOKO not estimable` verdict.

### 6.6 Table 4 status column and the preconditions

`report/tables/table4_status.csv`: one row per plan with the verdict strings the toolkit
produced (Plan 02: the count of cells passing `all_hard_pass` and C7's status; Plan 03: the
selection per rung; Plan 05: the fixed not-applicable strings; council gates: one cell each with
the roll-up verdict). `report/tables/preconditions.csv` is a copy of `gates/preconditions.csv`.
`report/tables/table_wapf_over_apf.csv/.tex` (move 11): `kernel, n cells, mean APF, mean wAPF,
wAPF / APF, mean flipped bits per changed page (= wAPF / APF x 32768)`, per-kernel means of the
per-cell means, from the extracts (P2 Sec. IV rung 0'; K2 move 11).

### 6.7 Figures (P2 Sec. VII "Figures"), `report/figures/`

| file | content | source |
|---|---|---|
| `fig_apf_per_kernel.png/.pdf` | APF(t) per kernel, eight reps overlaid, one panel per kernel, 12 panels, shared y in log scale; idle cells as a 13th panel when present | extracts |
| `fig_level_matched.png/.pdf` | the level-matched sets side by side: panel A floyd, histogram, nbody; panel B fft, gemm; all reps, median line per kernel | extracts |
| `fig_fused_plane.png/.pdf` | per snapshot `(r_l0_q50_per, J)`, one panel per kernel, eight reps, idle cells overlaid in grey, G-J mask applied (masked pairs hollow) | extracts, `gj_mask` |
| `fig_j_hist.png/.pdf` | J(t) histograms per kernel with the independence null (per-pair `J_null` histogram) and the floor null (idle J quantiles as vertical lines) | extracts, `gj.json` |
| `fig_ratio_hist.png/.pdf` | histograms of the three ratios' per-snapshot medians per kernel | extracts |
| `fig_floyd_decay.png/.pdf` | floyd's within-pass median `l0` aligned at the boundary, reps overlaid; written only when `gdec.csv` says `decay`, else a placeholder PNG with the G-DEC verdict as text | `gdec.csv` |
| `fig_piano_roll.png/.pdf` | one cell's changed-page raster (`seq` on x, `page_index` on y, one dot per row; the cell chosen by `--piano-cell`, default the first gemm cell) beside its APF and J; this is the one figure that re-streams a trajectory (through `extract.open_text`) and it subsamples rows by `--piano-stride` (default 16) | trajectory |
| `fig_table5_grid.png/.pdf` | the Table 5 verdict grid as a categorical heat map, one panel per rung | `table5_grid.csv` |

Matplotlib only, no seaborn, colours from the default cycle, reps of one kernel in one colour.

### 6.8 The LaTeX skeleton (`latex_skeleton.py` -> `report/paper2_skeleton.tex`)

A complete compilable document (`\documentclass{article}`, `booktabs`, `graphicx`), with the
section headings of P2 Sec. 3 (Abstract; I to X; Artifact and reproducibility), each followed by
a comment block `% - ...` listing P2's substance bullets for that section in abbreviated form,
`\input{tables/<name>.tex}` where the section's table belongs (III: Table 1 as a comment-only
shell and `preconditions`; IV: Table 2 as a static tabular of P2's cells; V: `table4_status`,
`table5`, `table5_g3`; VI: Table 3 as a static tabular of P2's cells with the pass table's
declared column merged; VII: `table6`, `table7`, `table8`, `tablegv`), and
`\begin{figure}...\includegraphics{figures/<name>.pdf}...\end{figure}` placeholders with empty
captions. No sentence outside comments. `pdflatex` is not required; the test checks that every
`\input` and `\includegraphics` target exists.

`report/manifest.json`: every file under `report/` and `gates/` with its sha256, sizes, the
package version, the `driver_state.json` ledger and every `params` block collected.

---

## 7. The driver and the runbook (builder 3)

`driver.py run --out <out> --root <retention root> [--cells-csv <out>/cells.csv] [--moves 0-13] [--assume-failed-zero
--assume-reason "..."] [--n-jobs N] [--null-perm 500] [--null-splits loko,loro,within_trace]
[--seed-offset 0] [--force]`. The driver executes al-Kindi's moves in order (P2 Sec. 5; K2 Sec.
5), each as a subprocess invocation of the owning module's CLI (never an import of another
builder's functions beyond `schema`/`verdicts`), records every command line, start, end and
exit code in `driver_state.json`, skips a move whose outputs exist unless `--force`, and stops at
the first non-zero exit. `driver.py status --out <out>` prints the ledger. Every move's command is
also runnable by hand; `RUNBOOK.md` lists them in this order with what each writes and what the
author looks at afterward.

The driver as built is `run_moves.py`; `driver.py` is a one-line alias of it (both invoke the same
`main`). The table below is the move table as the driver runs it after build epoch 2 (refreshed
2026-09-17 by builder 3 from `run_moves.build_plan`; E1 sec. 4 "Documentation known to be stale"):
`gates_calibration gp` runs at move 2, not move 6; `gates_calibration alias` runs at move 6 and
again at the end of move 7 (idempotent; the `table6_feature` rows appear once Table 6's features
exist); every command from move 3 on except `alias` and the skeleton writer declares
`gates/preconditions.json` as an input, so a re-run of the preconditions makes it stale; the driver
appends `--seed-offset` to the random commands when it is not 0 and `--n-jobs` to the commands that
take it when it is not 1; `driver.py plan --out <out> --moves N` prints the exact argument lists.
The skip rule (al-Farabi 2.8, refined in epoch 2, SPEC_epoch2 section 5.1): a command is skipped
only when a `done` ledger record exists for it, every declared output exists, and every declared
input's keyed hash equals the recorded one; `splits <rung>` and `gx <rung>` declare the rung's own
selection entry `json:gates/selection.json:<rung>`, so a resume with nothing changed re-runs no
split stage, and the ledger names the part that changed (`stale: selection.json[apf] changed since
move 7`); a command whose arguments differ from its `done` record's (ignoring `--n-jobs` and
`--jobs`) is `stale: arguments changed since move <n>` (SPEC_epoch2_review_al_farabi.md 6.3).

| Move | Command | Writes | The author looks at |
|---|---|---|---|
| 0 | `python3 -m pip install --user -r plan11_encoding_ladder/requirements.txt` (fallback when the server lacks a package; the runbook lists `numpy`, `scikit-learn`, `matplotlib`, optional `scipy`, `zstandard`, and the `zstd` binary) | nothing | `python3 -c "import numpy, sklearn"` succeeds |
| 0 | `extract.py index --root <retention root> --out <out>` | `cells.csv` | 96 kernel rows (+ idle rows) with `status = ok`; roles and archetypes right |
| 1 | `extract.py all --cells-csv <out>/cells.csv --out <out> --jobs 4` (the driver adds `--duration-s` when it departs from 600: every cell's declared duration, read by G2 and G-P from the sidecar, SPEC_epoch2 B12) | `extract/*` | the sidecars: `n_pairs` in 890 to 945, `n_seq_gaps`, `header_ncols = 66`, `status = ok`; the runbook says this is the longest step (minutes per cell) |
| 2 | `gates_precondition.py preconditions --out <out> --assume-failed-zero --assume-reason "AA A5: any failed job re-runs the whole cell"` (the driver adds `--failed-counts CSV`, `--c1-rule`, `--c1-abs-fraction`, `--c1-idle-percentile`, `--c1-activity-min` only when given; the command's own C1 default is `--c1-rule auto`: the idle floor's 95th percentile of K once idle cells enter it, else the declared absolute of 0.001 N = 262 pages; SPEC_epoch2 section 4, AD 2026-09-17) | `gates/preconditions.*` (with `C1_K_max`, `C1_threshold_pages`, `C1_rule` per row and the rule in force in `params`), `gates/failed_counts.csv` | `all_hard_pass` on every cell; the C6 header check; any excluded cell; `C1_rule` (`absolute_0.001` before the idle cells, `idle_floor_p95` after) |
| 2 | `gates_calibration.py pass-table --out <out>` then edit `inputs/pass_table.csv`; `gates_precondition.py gk0-template --out <out>` then edit `inputs/gk0_source.csv`; `gates_precondition.py idle-admissibility-template --out <out>` then fill `inputs/idle_admissibility.json` once the idle cells exist; `series.py head-drop-template --out <out>` | `inputs/*` | nbody's row; the lexer's head drop if known |
| 3 | `gates_calibration.py gc --out <out> --rung apf` and the same for `persist`, `content`, `wapf`, then `combined` last (its verdict is read from the other four rows; SPEC 3.4.3; CHECK_1 B3) | `gates/gc.csv` | one `pass` per rung; a `disconnected lead` stops the reading of that rung |
| 4 | `gates_precondition.py gk0 --out <out>`; `gates_precondition.py gf --out <out> --all-rungs --grid-id W8_H4 --n-perm 500` (the default grid point, SPEC 8 item 38; the run at the selected points is inside move 12) | `gates/gk0.csv`, `gates/gf.csv`, `gates/gf_floors.json` | which kernels are `IDLE, measured`; part (i) `inseparable at floor` on every rung; with no idle cells both say `not run` |
| 5 | `figures.py --out <out> --only apf_per_kernel,level_matched` | two figures | flat lines at kernel levels, reps overlaying, floyd over histogram, fft over gemm |
| 6 | `series.py features --out <out> --rung apf --all-grid --both` (first in every temporal stage, al-Farabi 2.2), `gates_temporal.py grid --out <out> --rung apf` (all 13 points, G1 to G5), `gates_temporal.py g3 --out <out> --rung apf`, `gates_temporal.py gord --out <out> --rung apf` (honours `--n-jobs` since epoch 2, SPEC_epoch2 5.2; its inputs are a superset of the grid's so a rebuilt grid re-runs it before `select`), `gates_temporal.py select --out <out> --rung apf`; `gates_calibration.py alias --out <out>` (`gp` runs at move 2) | `features/apf/*`, `gates/grid/apf/*`, `g3_flags.csv`, `table5_*`, `selection.json`, `alias.csv` | the selected (W, H) for APF and whether `passes_acceptance`; which W are `order-blind` |
| 7 | `grid complete apf` (internal: refuses the split stage unless the 13 grid CSVs and the 13 feature files exist; `gates/grid_complete.json`); `models.py splits --out <out> --rung apf --all-splits --raw-and-norm --null-perm 500 --null-splits loko,loro,within_trace` (input `json:gates/selection.json:apf`); `gates_comparison.py gx --out <out> --rung apf --null-perm 500` (the same keyed input); `gates_comparison.py gl --out <out>`; `gl2 rerun (feature drop after G-L (ii))` (internal, SPEC_epoch2 B10: when part (ii) reads `refused: shot noise explains CV` and `--gl2-rerun auto`, the driver runs `models.py splits --out <out> --rung apf --all-splits --raw-and-norm --feature-drop cov,std,peak2med --base-dir splits_gl2drop` and writes `gates/gl2_rerun.json`; `manual` leaves it to the author); `gates_comparison.py gn --out <out>`; `gates_calibration.py alias --out <out>` (again, for the `table6_feature` rows); `tables.py --out <out> --only table6` | `gates/splits/apf/*`, `gx.*`, `gl.csv`, `gn.csv`, `alias.csv`, `report/tables/table6.*` | LOKO norm inside the null; LORO collapsing after normalization; level-matched sets at chance; G-X `pooling stands` |
| 8 | `gates_readings.py gj --out <out>`; `figures.py --out <out> --only fused_plane` | `gj.*`, `gj_mask/`, the fused plane | clouds separated along `l0` by content type; J near one except the slow-pass kernel and the lexer at the floor |
| 9 | `series.py features --rung persist --all-grid --both`, `gates_temporal.py grid/g3/gord/select --rung persist`; `grid complete persist`; `models.py splits --rung persist --all-splits --norm --null-perm 500 --null-splits ...`; `gates_comparison.py gx --rung persist`; `figures.py --only j_hist` | as move 6 and 7 for `persist` | the gemm dip in every rep; G-P beside every reading |
| 10 | the same for `content` (`splits` and `gx` included); `gates_readings.py gdec --out <out>`; `figures.py --only ratio_hist,floyd_decay` | as above, `gdec.csv` | floyd and histogram separated; fft and gemm not; G-DEC's verdict |
| 11 | the same for `wapf` (`splits` and `gx` included); `tables.py --only wapf_over_apf` (a per-kernel table `mean wAPF / mean APF`) | as above | crude separation of floyd from histogram |
| 12 | the same for `combined` (`series.py features --rung combined --all-grid --norm`, the four temporal steps, `grid complete combined`, `splits combined --norm`, `gx combined`); `gates_comparison.py gl --out <out>` again (`gl (all rungs)`: G-L (i) for the four rungs whose selections now exist; CHECK_1 B2); `gates_precondition.py gf --out <out> --all-rungs --n-perm 500` (G-F at every rung's selected point, al-Farabi 2.6, before the tables); `gates_comparison.py gdim --out <out> --null-perm 500 --null-splits loko,loro,within_trace` (the `combined (matched)` run for every Table 7 split and label space, SPEC_epoch2 3.5.1); `gates_comparison.py gm --out <out>`; `variance.py --out <out>`; `models.py cluster --out <out> --rung combined --null-perm 500`; `tables.py --out <out> --table8-rung combined` (all tables); `latex_skeleton.py --out <out> [--standalone PATH]`; `figures.py --out <out> [--piano-cell ID] --piano-stride 16` (all); `tables.py --out <out> --only manifest` | `gates/splits/combined/*`, `gl.csv`, `gf.csv`, `gdim.csv`, `gates/splits_matched/combined/<gid>/<split>__<ls>/`, `gm.csv`, `gv.*`, `clustering.*`, `report/tables/*` (Table 7's `feature count` prints the matched dimension `feature_count_used` on the `combined (matched)` rows), `report/paper2_skeleton.tex`, `report/figures/*`, `report/manifest.json` | Tables 5, 7, 8 and G-V; the first rung off the null; the five `combined (matched)` rows; `manifest.json` |
| 13 | `gf check on Table 7` (internal; al-Farabi 2.6): every row of `report/tables/table7.csv` carries a non-empty `G-F (i)` cell, else `refused: <n> Table 7 rows without a G-F (i) verdict` (the G-F run at the selected points itself is inside move 12) | `gates/gf_check.json` | `verdict = pass`; the tripwire in the corrected wording on every rung |
| 14 | `comparators.py savoldi --out <out>`, `comparators.py dhodapkar --out <out> --delta-th-default 0.04`, `comparators.py law --out <out> --x-default 4 --jobs 4`, `comparators.py gates --out <out>` (build epoch 2, `SPEC_epoch2.md` Part 1: the three exact-input comparators of `council/14_`, each a feature vector per cell at `Wall_Hall` through `models.run_split_stage`, then G-L (i), G-DIM, G-M vs APF, G-X); `tables.py --out <out> --only table7_comparators,table_comparators`; `figures.py --out <out> --only dhodapkar_sweep`; `tables.py --out <out> --only manifest` | `gates/comparators/*` (the per-cell CSVs, the two sweeps, `law_cells.json`, `law_series/`, `gl.csv`, `gdim.csv`, `gm.csv`, `verdicts.csv`), `features/cmp_*/`, `gates/splits/cmp_*/`, the comparators' rows in `gates/gx.csv`, `report/tables/table7_comparators.*`, `report/tables/table_comparators.*`, `report/figures/fig_dhodapkar_sweep.*` | Savoldi's `U_text` per kernel; the sweep figure; the Table 7 comparator rows (raw and level-normalized) against APF and combined; `check_x2_equals_K` true on every cell |

The runbook also states: the expected wall time per step on a 96-cell corpus (extract about 2 to
5 hours at `--jobs 4`; the temporal grid minutes per rung; the split nulls hours, LORO's null the
longest); that every table cell that reads `not run:` or `pending:` names what is missing; that
the smoke run on the synthetic corpus is `synth.py corpus --root <tmp> && driver.py run --out
<tmp>/out --root <tmp> --null-perm 20 --n-jobs 2` and finishes in minutes with every table
present and every B1-G1 row marked `not run: 20 permutations < 500` by design.

### 7.1 The CLI contract

Every module exposes exactly these subcommands and flags (the driver calls them by these names;
a builder adds no required flag). Defaults are the values of sections 2 to 6. Every command exits
0 on success (a written refusal is a success), 2 when an input file is missing (its path on
stderr), 1 on an internal error. Every command that draws random numbers accepts `--seed-offset
INT` (default 0). Every command writes its `params` block.

```
extract.py index  --root R --out O [--idle-marker M ...] [--role-overrides CSV]
extract.py cell   --cell-dir D --out O [--persist-side t|t+1] [--failed-count N | --failed-dir D]
                  [--role kernel|idle] [--cell-id ID] [--force] [--duration-s 600]
extract.py all    --cells-csv CSV --out O [--jobs 1] [--only REGEX] [--failed-counts CSV]
                  [--persist-side t|t+1] [--force] [--duration-s 600]      (epoch 2, SPEC_epoch2 B12:
                  the declared duration, a float in the sidecar; a synthetic corpus passes n_pairs x 0.644)
synth.py cell     --root R --name NAME --seed S [--n-pairs 120] [--k0 2048] [--k-noise 0.02]
                  [--churn 0.05] [--pulse-period P] [--pulse-extra A]
                  [--content counter|spin|double|decay|idle] [--counter-step-max 60]
                  [--spin-bytes 2,8] [--double-words 64] [--decay-factor 0.7] [--floor-f 150]
                  [--floor-churn 0.02] [--trend 0.0] [--seq-first 1] [--gap-seqs 5,9]
                  [--label synth] [--no-compress] [--corrupt seq_reverse] [--step] [--k-decay]
synth.py corpus   --root R [--n-pairs 120] [--reps 8] [--idle 8] [--seed 20260916]
                  [--break-pulse] [--break-order] [--cv-case random|shot] [--level-only]
                  [--one-preset] [--ord-case] [--no-compress] [--duration-s SECONDS]
                  (synth.py cell takes --duration-s too; default n_pairs x 0.644 recorded in truth.json, SPEC_epoch2 B12)
series.py head-drop-template --out O [--force]        (epoch 2, SPEC_epoch2 B20: an existing author input is kept)
series.py features --out O [--cells-csv CSV] --rung RUNG (--grid-id ID | --all-grid)
                  [--raw | --norm | --both] [--wapf-norm median_K|median_self]
gates_precondition.py preconditions --out O [--assume-failed-zero --assume-reason TEXT]
                  [--failed-counts CSV]
                  [--c1-rule auto|idle_floor|absolute|legacy_apf_max] [--c1-abs-fraction 0.001]
                  [--c1-idle-percentile 95] [--c1-idle-pool pooled_snapshots|cell_medians]
                  [--c1-min-idle-cells 1] [--c1-activity-min 0.02]     (epoch 2, SPEC_epoch2 section 4;
                  the CLI default is auto, the function default legacy_apf_max)
                  [--c1-activity-min-pages N]      (SPEC_epoch2 B1: AA T1's 200 pages for the absolute rule, when given)
gates_precondition.py gk0-template --out O [--force]
gates_precondition.py idle-admissibility-template --out O [--force]
gates_precondition.py gk0 --out O [--tail-fraction 0.8]
                  [--idle-pool pooled_snapshots|cell_medians] [--idle-percentile 95]
gates_precondition.py gf --out O (--rung RUNG | --all-rungs) [--grid-id ID] [--n-perm 500]
                  [--part1-design within_trace_window] [--test-frac 0.2]
                  [--part2-rule all_cells_outside|median_of_cells_outside] [--n-jobs 1]
                  [--n-estimators 300] [--perm-floor 500]      (SPEC_epoch2 B3: part (i) reads `not run: N
                  permutations < 500` when 0 < n-perm < the floor; the function default is 0)
gates_calibration.py pass-table --out O [--force]
gates_calibration.py gc --out O --rung RUNG [--pulse-kernel gemm] [--jump-factor 2.0]
                  [--jump-reference cell_median_K|local_median_8] [--j-dip-max 0.75]
                  [--dip-window-pairs 1] [--min-events-per-rep 1]
                  [--pairing by_rep_index|envelope] [--content-page-set persistent|all]
gates_calibration.py gp --out O [--min-passes-rhythm 5] [--min-snaps-within-pass 3]
gates_calibration.py alias --out O [--r2-threshold 0.5]
gates_temporal.py grid --out O --rung RUNG [--n-surrogates 200] [--trend-drift-sd 1.0] [--n-jobs 1]
                  [--wapf-norm median_K|median_self]      (SPEC_epoch2 B19; --n-jobs honoured since epoch 2, B6)
gates_temporal.py g3 --out O --rung RUNG [--min-cells 7] [--n-surrogates 200] [--min-quef-frac 0.125]
gates_temporal.py gord --out O --rung RUNG [--n-order-perm 20] [--null-perm 100] [--n-jobs 1]
gates_temporal.py select --out O --rung RUNG [--rollup all_kernels|majority]
                  [--rollup-kernel-refusals not_applicable|blocks] [--g1-none-applicable drop|refuse]   (SPEC_epoch2 B7)
gates_readings.py gj --out O [--k-factor 3.0]
gates_readings.py gdec --out O [--kernel floyd] [--control-kernel gibbs]
                  [--boundary-source k_jump|period] [--jump-factor 2.0] [--min-run 3]
                  [--min-reps 7] [--n-surrogates 200] [--pass-frac 0.5]      (SPEC_epoch2 B11)
models.py splits  --out O --rung RUNG [--grid-id ID] (--split S | --all-splits)
                  [--labelspace kernel|archetype|all] (--raw | --norm | --raw-and-norm)
                  [--null-perm 500] [--null-splits loko,loro,within_trace] [--n-jobs 1]
                  [--feature-drop F,...] [--include-idle] [--n-estimators 300] [--base-dir splits]
                  (SPEC_epoch2 B18: without --grid-id the selected point is used and exit 2 says
                  `missing input: gates/selection.json has no entry for <rung>` when there is none)
models.py cluster --out O [--rung combined] [--algo kmeans|gmm|agglomerative] [--null-perm 500]
                  [--perm-floor 500]      (SPEC_epoch2 B3)
gates_comparison.py gl   --out O [--gl2-level kernel|cell]
gates_comparison.py gn   --out O
gates_comparison.py gx   --out O --rung RUNG [--null-perm 500] [--n-jobs 1] [--n-estimators 300]
                  [--perm-floor 500]      (SPEC_epoch2 B3)
gates_comparison.py gdim --out O [--method train_importance|pca] [--null-perm 500] [--n-jobs 1]
                  [--n-estimators 300] [--matched-splits all|loko]
                  [--null-splits loko,loro,within_trace]                (epoch 2, SPEC_epoch2 3.5.1)
gates_comparison.py gm   --out O [--n-seeds 5]
variance.py       --out O
tables.py         --out O [--only NAME,...] [--table8-rung combined]
figures.py        --out O [--only NAME,...] [--piano-cell ID] [--piano-stride 16]
latex_skeleton.py --out O
driver.py run     --out O --root R [--cells-csv CSV] [--moves 0-13] [--assume-failed-zero
                  --assume-reason TEXT] [--n-jobs 1] [--null-perm 500]
                  [--null-splits loko,loro,within_trace] [--seed-offset 0] [--force]
                  [--dry-run] [--only-modules M,...] [--skip-missing-modules]
                  [--c1-rule auto|idle_floor|absolute|legacy_apf_max] [--c1-abs-fraction 0.001]
                  [--c1-idle-percentile 95] [--c1-activity-min 0.02]   (epoch 2: passed to
                  `preconditions` only when given; run_moves.py is the implementation)
                  [--c1-activity-min-pages N] [--gl2-rerun auto|manual] [--duration-s 600]
                  [--wapf-norm median_K|median_self] [--n-estimators 300]
                  [--gord-n-order-perm 20] [--gord-null-perm 100]      (SPEC_epoch2 B1, B10, B12,
                  B19, B25, B26; each passed to its command when it departs from the default)
driver.py status  --out O
driver.py plan    --out O [--moves 0-13] ...    (print the command list without running)
```

`--grid-id` defaults to the rung's selected point from `gates/selection.json` when it exists and
to `W8_H4` otherwise (the value used is recorded); since build epoch 2 `models.py splits` has no
`W8_H4` fallback: without a selection it exits 2 (SPEC_epoch2 B18), and `params.grid_source`
records `selection.json` or `argument`. `models.py splits --all-splits` runs
`within_trace` and `loro` in both label spaces and `loko` in `archetype`. `series.py features`
is called by the temporal and split stages themselves; the standalone command exists for
inspection.

---

## 8. For the author: every choice the definitions leave open, with the default specified

Each item is a parameter (module-level constant or CLI flag) whose value is written into the
result files' `params`. The default is what runs unless the author says otherwise.

1. `persist_side = "t"` (extract). The definition says "the pages present in both `S_t` and
   `S_{t+1}`" and does not say which snapshot's channel values are summarized on them. Default:
   the values at `seq = t`. Alternative `"t+1"`.
2. `J_null` as a ratio of expectations (`J_null_inter / (K_t + K_{t+1} - J_null_inter)`). The
   definition gives the expected intersection `K_t K_{t+1} / N` only; a Jaccard-scaled form is
   needed to put it beside J. Both columns are kept.
3. Gap handling: a missing `seq` is a K = 0 snapshot (verified in the consumer), counted in
   `n_seq_gaps`; it does not fail C6 and does not count as a failed job. The author confirms this
   reading against the `failed/` directories of the campaign.
4. `rep` from the seed (rep 0 = seed 42, the rest by ascending seed); idle cells from the
   directory counter. The path's `rep001` is not the paper's rep index.
5. Idle role detection by the substrings `sleep`, `idle` in the test label; `cells.csv` is
   editable.
6. `head_drop.csv` default 0 for every kernel; the lexer's first pass is the author's number.
7. `wapf_norm = "median_K"` (the literal count-rung rule). Alternative `"median_self"`.
8. `duty_rule`: `0.1 * max` (the code of `b1_features.py`), not the B1 design note's "against
   the cell's median".
9. Temporal gates for `content` and `combined` run on the channel `r_l0_q50_per`.
10. G1's trend rule: least-squares drift over the cell exceeding `g1_trend_drift_sd = 1.0`
    population standard deviations. G1's surrogate: phase-randomized (block bootstrap not
    implemented). G1 per kernel: the median over cells against the median of the surrogates'
    5th percentiles; `TREND_PRESENT` when more than half the kernel's cells have a trend.
11. G3 flag: `g3_min_cells = 7` of 8 cells above their own surrogate p95; the cepstral tail
    search mirrors `plan03_metric_kernel.py` (whole tail from `n // 8`), which excludes any rhythm
    faster than about `n / 8` pairs (about 75 s on this dataset) from the search; the author may
    lower the floor with `--min-quef-frac` (default `1/8`, recorded). No CV threshold exists
    under phase randomization; CV is reported beside `1 / sqrt(K_median)` without a verdict.
12. G-ORD: `gord_n_order_perm = 20` order shuffles, `gord_null_perm = 100` label shuffles for the
    spread, `spread = p95 - p05`, at `H = W // 2` under LOKO/archetype.
13. Grid roll-up `grid_rollup = "all_kernels"` (every applicable kernel must pass); selection =
    Plan 03's rule (smallest W passing G1, G2, G4; hop ratio nearest 0.5; best-feasible fallback).
    `GP_UNDECLARED` kernels do not block G2.
14. G-K0: `tail_fraction = 0.80`; idle band edge = the 95th percentile of K pooled over idle
    rows (`idle_pool = "pooled_snapshots"`); kernel verdict from the median of its cells' tail
    medians.
15. G-F part (i) executable form: within-trace window-level 8-class test with a window-level
    label shuffle (`part1_design = "within_trace_window"`), because leave-one-rep-out cannot
    predict an unseen rep label. Part (ii) `part2_rule = "all_cells_outside"`.
16. G-C: `jump_factor = 2.0` against the cell's median K; `j_dip_max = 0.75` within
    `dip_window_pairs = 1`; `min_events_per_rep = 1`; the pass boundary is the K jump itself;
    content ordering statistic = the cell's median over seqs of the per-seq median on persistent
    pages (`content_page_set = "persistent"`), pairing `by_rep_index`; wAPF's pulse = the APF rule
    on the wAPF series (the definition names none).
17. G-P: pair units from the cell's own pair count; `min_passes_rhythm = 5`,
    `min_snaps_within_pass = 3`; an `inferred:` pass-table row is used with the `(INFERRED)` suffix.
18. Alias falsifier: `r2_threshold = 0.5` over 8 cells per kernel; the dt spread is about
    0.04 s on this dataset and the result is reported with it.
19. G-J: `k_factor = 3.0`; the "declared null-relative threshold" for the below-null fraction is
    `J <= J_null`; the mask is applied to the fused plane and the per-cell summaries, not to the
    split features.
20. G-DEC: `boundary_source = "k_jump"`, `min_run = 3`, `min_reps = 7`, the surrogate is a
    within-pass order permutation (200), (d) compares the observed slope to the surrogates' 5th
    percentile, (e) uses the idle cells' whole-cell slope against its own phase-randomized p95.
21. Splits: `loro_mode = "cell"` (96 folds; `"rep_index"` gives 8); `test_frac = 0.2`;
    within-trace label spaces `kernel` and `archetype`; LOKO in `kernel` space not run.
22. Unit aggregation `cell_majority` (ties by mean probability, then class name).
23. The forest: `n_estimators = 300`, `max_features = "sqrt"`, median imputation, per-fold
    standardization, seed `20260919`.
24. B1's L1 for more than two classes: a one-feature tree with `max_leaf_nodes = n_classes`;
    `b1g3_max_disagree = 1`.
25. B1-G1: `n_perm = 500`; LORO's null costs about 48,000 forest fits per rung; the author decides
    whether to run it (`--null-splits`) or to leave LORO's null column `not run`.
26. G-L (ii) at the kernel level (`gl2_level = "kernel"`, 12 points); the drop set when refused is
    `("cov", "std", "peak2med")`.
27. G-X: 500 campaign-label shuffles across cells; the label normalization
    `dwarfs1* -> dwarfs1`; the cell order is an optional input (`inputs/cell_order.csv`).
28. G-DIM: `dim_match_method = "train_importance"` (alternative `"pca"`); the matched dimension is
    the strongest single rung's; the over-dimension reduction target is `n_train_cells`.
29. G-M: `gm_n_seeds = 5`, `spread = max - min` of the APF LOKO score over seeds; fold assignment
    is fixed by the split definitions so the seed is the only spread source.
30. G-V: population variances; L2 over archetypes with at least two kernels.
31. Clustering: `KMeans(n_init = 10)` primary; `k` = the number of predicted archetypes present
    after G-K0; per-cell vector = mean over windows at the selected point.
32. Table 8's assignment rung: `table8_rung = "combined"`.
33. Content feature set: primary three medians through the 8 shape features (24) plus the 12
    secondary quantiles' window means (36 total).
34. The content-change rung's temporal gates, G-C and G-F use persistent-page statistics; the
    all-rows columns are kept in the extract for the author's alternative.
35. The seed offsets: four fixed seeds (`20260916` to `20260919`); `--seed-offset` shifts all.
36. `C1_ACTIVITY_MIN = 0.02` (the apf_queue re-map), `C2_MIN_PAIRS = 8`, C3's informational
    threshold `> 3` windows at (8, 4).
37. The per-domain `failed/` count is a recorded input; the toolkit never assumes zero without
    `--assume-failed-zero --assume-reason`.
38. `gf_default_grid = "W8_H4"`: G-F part (i) at move 4 runs before any (W, H) is selected and
    is re-run at the selected point at move 13; both rows are kept.
39. The whole-cell grid point has `H = W` and fails G4 by construction; it stays in Table 5 as
    APF's whole-cell reading (the footprint-size trajectory folded in) and is never selected by
    the Plan 03 rule; the author decides whether Table 6 also reports it (`--grid-id Wall_Hall`).
40. The `seq` origin (0 in the consumer's initialisation, 1 as observed on the server) is not
    assumed; `seq_first` is recorded. If the author finds the first differ pair is skipped by the
    producer, the pair count in Table 1 is `n_pairs` as recorded here, unchanged.
