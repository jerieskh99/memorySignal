# plan11_encoding_ladder: runbook for the author

Written 2026-09-16 by builder 3 (report). This is the command-line companion to `SPEC.md`
section 7: what to install, the exact commands in al-Kindi's move order (P2_STRUCTURE.md
section 5; K2 section 5), what each move writes, what to look at afterwards, and how to add the
idle cells once they are captured. Every command is also run for you, in this order, by the
driver (`run_moves.py`; `driver.py` is the same program under SPEC's name).

Conventions. `<out>` is the output root you choose (everything the toolkit writes goes under
it; nothing is written anywhere else). `<root>` is the retention root that holds
`kernel/<test_label>/<param-sig>/rep<NNN>__<label>/` (on the server this is the path
P2_AUTHOR_ANSWERS.md S1 names, `/mnt/nfs/jeries/memory_traces/zstd_local`; the toolkit never
assumes it, you pass it). Every command is run from `VM_sampler/VM_Capture_QEMU/` as
`python3 -m plan11_encoding_ladder.<module> ...`; each module also runs by absolute path.
Every command exits 0 on success (a written refusal is a success), 2 when an input file is
missing (its path is printed on stderr), 1 on an internal error. Every result file carries a
`params` block with every parameter value used; every table cell that reads `not run:` or
`pending:` names the file or the move that is missing.

---

## 0. Prerequisites

Python 3.10 or newer. Required packages: `numpy`, `scikit-learn`, `matplotlib`. Optional:
`scipy` (a stdlib or numpy fallback exists), `zstandard` (only when the `zstd` binary is not on
`PATH`). `python3 -m pytest -q tests` (from `plan11_encoding_ladder/`) is the only supported test
runner: `python3 -m unittest` does not discover the pytest-style gate tests and stops on
`tests/test_runner_guard.py` with the same instruction. The `zstd` binary is the first choice for
reading `.csv.zst` (`zstd -dc -q`), the `zstandard` module the second, and a plain `.csv` always
works.

Check what the server has:

```
python3 --version
python3 -c "import numpy, sklearn; print(numpy.__version__, sklearn.__version__)"
python3 -c "import matplotlib; print(matplotlib.__version__)"
which zstd || python3 -c "import zstandard"
```

If a required package is missing and you have no root, install into your user site:

```
python3 -m pip install --user -r VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/requirements.txt
python3 -c "import numpy, sklearn"
```

If `pip` itself is missing: `python3 -m ensurepip --user` first. If `matplotlib` cannot be
installed, every figure step writes `report/figures/SKIPPED.txt` naming the missing module and
exits 0; the tables and the skeleton do not need it.

Run the toolkit's tests once on the server before the first real move (they use synthetic data
only and touch nothing under `<root>`):

```
cd VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder
python3 -m pytest -q tests      # the only supported runner; python3 -m pip install --user pytest if it is missing
```

## 0b. The smoke run (synthetic corpus; about an hour at the full corpus)

Before touching the data, run the whole chain on the synthetic corpus so that every table and
figure exists once with every `not run:` cell named:

```
cd VM_sampler/VM_Capture_QEMU
python3 -m plan11_encoding_ladder.synth corpus --root /tmp/p11smoke
python3 -m plan11_encoding_ladder.run_moves run --out /tmp/p11smoke/out --root /tmp/p11smoke \
    --null-perm 20 --n-jobs 2 --duration-s 77.28 --assume-failed-zero --assume-reason "smoke run"
```

`--duration-s 77.28` is the synthetic cells' duration (120 pairs at the paper's median guest
spacing of 0.644 s, the value `truth.json` records): G2's seconds coverage and G-P's `T_seconds`
read each cell's declared duration from its sidecar, so a declared pass count on a 120-pair cell
is read against 77.28 s and not against the real corpus's 600 s (SPEC_epoch2 B12). By design every
B1-G1 row of that run reads `not run: 20 permutations < 500` (SPEC 3.7.1), and so does every
other null-judging gate at `--null-perm 20`: G-F part (i), G-X's `leak_verdict` and the
clustering's `exceeds_ari` / `exceeds_nmi` read `not run: 20 permutations < 500` while their
numbers stay in the files (SPEC_epoch2 B3, B27); the smoke run is not admissible for the paper.
A corpus with fewer than 8 reps or 12 kernels reads G3's kernel flag (7 cells), G-DEC's roll-up
(7 reps) and G-M's sign test (6 of 12 kernels) at their fixed minima by construction; pass
`--min-cells` and `--min-reps` by hand on `g3` and `gdec` for a smaller corpus (SPEC_epoch2 B13). Measured cost (checker, cycle 2, 12 kernel cells at
120 pairs, `--null-perm 20`): G-ORD 424 to 772 s per rung in a single process (measured before
epoch 2; `gord` and `grid` honour `--n-jobs` since epoch 2, the numbers identical at any job
count), the split stages 176 to 332 s per rung, the rest under a minute; at the 104-cell corpus
above expect G-ORD to take the better part of an hour per rung in one process.

---

## 1. The driver

```
cd VM_sampler/VM_Capture_QEMU
python3 -m plan11_encoding_ladder.run_moves run --out <out> --root <root> \
    --assume-failed-zero --assume-reason "AA A5: any failed job re-runs the whole cell" \
    --keep-first-pairs plan11_encoding_ladder/declared/keep_first_pairs.csv \
    --n-jobs 4 --null-perm 500 --null-splits loko,loro,within_trace
python3 -m plan11_encoding_ladder.run_moves status --out <out>      # the ledger
python3 -m plan11_encoding_ladder.run_moves plan --out <out> --root <root> --moves 6-7   # print, do not run
```

- `--moves 0-13`, `--moves 6,7`, `--moves 2-4,12` select moves. Move 0 needs `--root`. Since
  build epoch 2 `--moves` defaults to `0-14`: move 14 is the comparators (its section below), with
  the driver flags `--delta-th-default 0.04`, `--law-x-default 4`, `--savoldi-rows
  all_after_head_drop` and `--comparator-jobs N` (the Law pass's processes; default `--n-jobs`).
- The driver records every command line, start, end, exit code, the tail of stdout/stderr and
  the sha256 of every author input it read in `<out>/driver_state.json`. It stops at the first
  non-zero exit; fix the cause and run the same command again.
- Resume rule: a command whose outputs exist is skipped, unless an input it read
  (`cells.csv`, `inputs/*`, and the gate files it depends on) has changed since, in which case
  the ledger says `stale: <file> changed since move <n>` and the command re-runs. `--force`
  re-runs everything selected. The author's inputs (`inputs/*`) are never overwritten once they
  exist (`kept: author input exists`). The admissibility record (`gates/preconditions.json`,
  written by move 2 only) is a declared input of every command from move 3 on, so a re-run of
  the preconditions (by hand with another `--c1-activity-min`, or after an edited
  `inputs/failed_counts.csv`) makes every later move stale; and G-ORD's inputs are a superset
  of the grid's, so a rebuilt grid re-runs G-ORD before `select` reads its label (al-Farabi
  certification, cycle 2, 7.1 and 7.2).
- Epoch 2 refined the resume rule for the split stages and G-X: their trigger on
  `gates/selection.json` is the rung's own entry, not the whole file, so a resume of moves 7 to 13
  with nothing changed re-runs no split stage (the ledger prints `stale: selection.json[<rung>]
  changed since move <n>` when the rung's own selection did change). Every other command keeps
  the whole-file rule. The first resume of a ledger written before epoch 2 re-runs the split
  stages once, because the recorded key changed. A changed flag on a step re-runs it: the
  driver compares the command's arguments with the `done` record's (ignoring `--n-jobs` and
  `--jobs`) and prints `stale: arguments changed since move <n>`, so `--moves 2 --c1-rule
  absolute` after a run at the default re-runs the preconditions instead of keeping the old
  file under the new flag; the changed `gates/preconditions.json` then makes every later move
  stale, as before.
- The gate result files under `gates/` are never edited by hand (CERT 1(a), 7.1; SPEC_epoch2
  B14): the staleness rule reads them as the moves wrote them, and an edited file would be taken
  for a result. The sanctioned way to change the admissible set is a re-run of the preconditions
  with another flag (move 2), which makes every later move stale by itself.
- `--seed-offset N` shifts the four fixed seeds (SPEC 3.2) on every command that draws random
  numbers and is recorded in every `params` block.
- The `failed/` count: the driver never assumes zero. Pass
  `--assume-failed-zero --assume-reason "AA A5: any failed job re-runs the whole cell"` (the
  text is stored in `params` and should be stated once in Limitations), or give the recorded
  counts with `--failed-counts <csv>` (columns `cell_id, failed_count, source`).
- `--keep-first-pairs <csv>` (AA A8; SPEC_epoch2 Part 4 item 26): the declared "keep only the
  first N pairs" file, passed to move 1. `plan11_encoding_ladder/declared/keep_first_pairs.csv`
  lists the three runs whose recording holds an unplanned second run (gemm 940, fem_assembly 895,
  fft 926 pairs). The path is relative to the driver's cwd (`VM_sampler/VM_Capture_QEMU`); the
  driver declares the file as an input of move 1, so its sha256 is in the ledger and a change
  makes move 1 stale.
- Cost: extract about 1 to 3 minutes per cell of four million rows (2 to 5 hours for 96 cells at
  `--jobs 4`); the temporal grid minutes per rung; the split nulls hours, LORO's null the longest
  (about 48,000 forest fits per rung at 500 permutations, `loro_mode = "cell"`). If LORO's null
  is too expensive, run `--null-splits loko,within_trace` and LORO's null column reads
  `not run` (SPEC section 8 item 25).
- Every move below is also runnable by hand with the command shown; the driver's `plan`
  subcommand prints the exact argument lists.

---

## 2. The moves, in order

### Move 0: the cell index

```
python3 -m plan11_encoding_ladder.extract index --root <root> --out <out>
```

Writes `<out>/cells.csv` (one row per cell directory: `cell_id, kernel, role,
archetype_predicted, seed, rep, rep_dir, label, campaign, path, traj_file, status`).

Look at: 96 kernel rows (plus idle rows once captured) with `status = ok`; the roles and the
predicted archetypes right (Table 3); `rep 0` is the seed-42 cell of every kernel; any
`refused: trajectory file count != 1` (the stencil_jacobi cells until their trajectories are
filed, P2_AUTHOR_ANSWERS.md S3), `refused: duplicate seed` or `refused: unknown kernel` row is
yours to resolve by editing `cells.csv`. Every later stage reads this file, never the paths.

### Move 1: the per-cell extract (the longest step)

```
python3 -m plan11_encoding_ladder.extract all --cells-csv <out>/cells.csv --out <out> --jobs 4 \
    --keep-first-pairs plan11_encoding_ladder/declared/keep_first_pairs.csv
```

Writes `<out>/extract/<cell_id>/extract.csv` (one row per `seq`, 58 columns, SPEC 2.2) and
`sidecar.json` (SPEC 2.3). A cell whose sidecar says `status = ok` under the same cut is skipped
on re-run.

`--keep-first-pairs <csv>` (AA A8; SPEC_epoch2 Part 4 item 26) has the columns `path,
keep_first_pairs, reason`; `path` is the cell directory relative to the retention root
(`family/workload/variant/rep`). For a listed cell the extract keeps the first N pairs of the
file in seq order, every row with `seq <= seq_first + N - 1` (a seq gap counts as a pair), and
ignores the later rows; unlisted cells are unchanged, and nothing under the retention root is
written. The sidecar records `keep_first_pairs`, `keep_first_reason`, the file's own extent
(`file_n_pairs`, `file_seq_last`, `n_rows_after_cut`), and `n_pairs` and `dt_est_s` are computed
on the kept pairs only. A row that matches no cell of `cells.csv` or more than one, or an N larger
than the file's pair count, stops the command and names the row. A sidecar written under another
cut (or none) is not `ok` for this cut: the cell is re-extracted, not skipped. The same flag on
`extract cell` cuts a single cell.

Look at the sidecars: `n_pairs` in 890 to 945, `header_ncols = 66`, `n_rows_skipped = 0`,
`n_seq_gaps` (a gap is a K = 0 snapshot, not a failed job; SPEC 2.1), `status = ok`,
`dt_est_s` about 0.64 s. Confirm the gap reading against the consumer log: every gap `seq` must
correspond to a `substrate seq=<n> rows=0` line, not to a `WARNING: no substrate CSV found`
line (al-Farabi review 2.10).

### Move 2: validity of observation, the author's inputs, G-P

```
python3 -m plan11_encoding_ladder.gates_precondition preconditions --out <out> \
    --assume-failed-zero --assume-reason "AA A5: any failed job re-runs the whole cell"
python3 -m plan11_encoding_ladder.gates_calibration pass-table --out <out>
python3 -m plan11_encoding_ladder.gates_precondition gk0-template --out <out>
python3 -m plan11_encoding_ladder.gates_precondition idle-admissibility-template --out <out>
python3 -m plan11_encoding_ladder.series head-drop-template --out <out>
#   ... edit inputs/pass_table.csv, inputs/gk0_source.csv, inputs/head_drop.csv (and, once the
#   idle cells exist, inputs/idle_admissibility.json), then:
python3 -m plan11_encoding_ladder.gates_calibration gp --out <out>
```

Writes `gates/preconditions.csv/.json` (C1 to C8 re-mapped, the `failed/` verdict,
`all_hard_pass`), `gates/failed_counts.csv`, the four templates under `inputs/`, and
`gates/gp.csv` (G-P runs here, not at move 6: it needs only the pass table and `n_pairs`;
al-Kindi review 6).

Look at: `all_hard_pass` on every cell; C6's header check; any excluded cell
(`preconditions.json` `excluded_cells`); nbody's pass-table row (declared, 6,147 per 600 s) and
the eleven `undeclared` rows. If you fill a kernel's row with an `inferred:` count it must come
from source and base parameters (VME 1.5), never from the trajectory, the cepstrum or any toolkit
output (al-Farabi review, for the author 4). `gates/gp.csv`: nbody `aliased by design at this
size`, the rest `undeclared`. `head_drop.csv` default 0; if you set the lexer's first pass,
record its source (file and line of the phase marker) in the `reason` column.

C1 for kernel cells is re-mapped (P2_AUTHOR_ANSWERS.md, Decisions of 2026-09-17; SPEC_epoch2
section 4). The standard command above does not change: the command's default is `--c1-rule
auto`. Without idle cells a kernel cell is active when its `K_max` is at least 262 pages (0.1
percent of memory; `C1_rule = absolute_0.001` in `preconditions.csv` and `preconditions.json`).
Once idle cells are indexed and pass their own C2 and C6, the rule switches by itself to
`idle_floor_p95`: active when `K_max` exceeds the 95th percentile of `K` pooled over the idle
cells' rows, the same edge G-K0 uses; the edge and the idle cells used are in `preconditions.json`
`params` (`C1_idle_band_edge`, `C1_idle_cells_in_floor`). The inherited 0.02 (5,243 pages) is
`--c1-rule legacy_apf_max` and is recorded as the alternative; `--c1-rule idle_floor` with no idle
cell entering the floor refuses every kernel row rather than falling back. The driver's
`--c1-rule`, `--c1-abs-fraction`, `--c1-idle-percentile`, `--c1-activity-min` and
`--c1-activity-min-pages` reach the command only when you set them. AA T1 states the interim as
`K_max >= 200` pages while the 2026-09-17 bullet says 0.1 percent of memory (262 pages): the
command's default is 262 and `--c1-activity-min-pages 200` selects T1's number
(`C1_rule = absolute_200pages`); say which stands in `P2_AUTHOR_ANSWERS.md` before the run
(SPEC_epoch2_review_al_farabi.md section 6 item 1). Look at the new columns `C1_K_max`, `C1_threshold_pages`,
`C1_rule`, and `params.C1_rule_in_force` (also printed in Table 4's Plan 02 cell): they record
which rule was in force for each run.

### Move 3: the instrument check, G-C, per rung

```
for r in apf persist content wapf combined; do
  python3 -m plan11_encoding_ladder.gates_calibration gc --out <out> --rung $r
done
```

Writes `gates/gc.csv` (per rung, per rep, and a `rep = all` row). `combined` runs last: its
verdict is read from the other four `rep = all` rows (pass iff the four pass; any `disconnected
lead` makes it disconnected), and re-running any single rung drops the combined row, so re-run
`--rung combined` after any of the four.

Look at: one `pass` per rung. A `disconnected lead` voids every negative of that rung: Table 5
carries it in `refusal` on every row of the rung and Table 7 prints `refused: disconnected
lead` in the rung's score cells until you fix the implementation and re-run (al-Farabi review
2.5). The `not applicable: pulse aliased by design (full footprint lit every snapshot)` verdict
(al-Kindi review 2) neither passes nor voids: the level pulse alone was seen; the 600 s gemm run
without capture (P2 section 6 item 5) settles the regime. `stat_a` is the observed per-rep
maximum K ratio (detection at 1.5, prediction 2.0; al-Kindi review 1).

### Move 4: the floor, G-K0 and G-F

```
python3 -m plan11_encoding_ladder.gates_precondition gk0 --out <out>
python3 -m plan11_encoding_ladder.gates_precondition gf --out <out> --all-rungs --grid-id W8_H4 --n-perm 500
```

Writes `gates/gk0.csv`, `gates/gf.csv`, `gates/gf_floors.json`.

Look at: which kernels read `IDLE, measured` (expected: the lexer); G-F part (i)
`inseparable at floor` on every rung; part (ii) `at floor in this lead` per (rung, kernel) as a
finding. With no idle cell both read `not run: no admissible idle cell` and the tripwire row of
the claim stays in Limitations. This G-F run is at the declared default grid point `W8_H4`
(SPEC section 8 item 38); move 12 re-runs it at every rung's selected point and both rows are
kept.

### Move 5: APF(t) per kernel, the level-matched sets

```
python3 -m plan11_encoding_ladder.figures --out <out> --only apf_per_kernel,level_matched
```

Writes `report/figures/fig_apf_per_kernel.pdf/.png`, `fig_level_matched.pdf/.png`.

Look at: flat lines at kernel-specific levels, reps overlaying, floyd on top of histogram (and
nbody), fft on top of gemm (or gemm alternating between about 4,096 and a few hundred if its
pass outlasts the interval, K2 move 5).

### Move 6: the temporal gate on APF, Plan 03 as amended

```
python3 -m plan11_encoding_ladder.series features --out <out> --rung apf --all-grid --both
python3 -m plan11_encoding_ladder.gates_temporal grid   --out <out> --rung apf
python3 -m plan11_encoding_ladder.gates_temporal g3     --out <out> --rung apf
python3 -m plan11_encoding_ladder.gates_temporal gord   --out <out> --rung apf
python3 -m plan11_encoding_ladder.gates_temporal select --out <out> --rung apf
python3 -m plan11_encoding_ladder.gates_calibration alias --out <out>
```

Writes `features/apf/<grid_id>_{raw,norm}.npz` at all 13 points (al-Farabi review 2.2),
`gates/grid/apf/<grid_id>/temporal_per_kernel.csv` (13 points, every one kept),
`g1_surrogates.npz`, `gord.json`, `gates/g3_flags.csv`, `gates/table5_long.csv`,
`gates/table5_grid.csv`, `gates/selection.json`, `gates/alias.csv`.

Look at: the selected (W, H) for APF and whether `passes_acceptance` is true (a
`best-feasible` selection prints as `selected: best-feasible` in Tables 5 and 7 and is never a
gated resolution; al-Farabi review 2.11(c)); which W are `order-blind`; G2 in pair units beside
the two dt columns (al-Kindi review 7); G3's per-kernel flag (the quefrency floor `n // 8`
excludes rhythms faster than about 75 s; SPEC section 8 item 11); `alias.csv` rows of kind
`g3_peak`. Nothing is ever deleted from `gates/grid/`.

`gord` honours `--n-jobs` since epoch 2 (joblib threads over its two loops, every random draw
made before dispatch; the numbers do not depend on the job count and `gord.json` `params`
records `n_jobs` and `parallel_backend`).

### Move 7: Table 6, APF under the three splits

```
python3 -m plan11_encoding_ladder.models splits --out <out> --rung apf --all-splits --raw-and-norm \
    --null-perm 500 --null-splits loko,loro,within_trace --n-jobs 4
python3 -m plan11_encoding_ladder.gates_comparison gl --out <out>
python3 -m plan11_encoding_ladder.gates_comparison gn --out <out>
python3 -m plan11_encoding_ladder.gates_comparison gx --out <out> --rung apf --null-perm 500
python3 -m plan11_encoding_ladder.gates_calibration alias --out <out>      # again: table6_feature rows (al-Kindi review 5)
python3 -m plan11_encoding_ladder.tables --out <out> --only table6
```

Before `splits`, the driver checks that all 13 `temporal_per_kernel.csv` and the 13 feature
npz files of the rung exist (`gates/grid_complete.json`) and refuses the split stage otherwise.
Writes `gates/splits/apf/<grid_id>/<split>__<labelspace>/{predictions.csv, scores.json,
null.json, l1_quarantine.json}`, `gates/gl.csv`, `gates/gn.csv`, `gates/gx.csv/.json`,
`report/tables/table6.csv/.md/.tex`.

If `gates/gl.csv` part (ii) reads `refused: shot noise explains CV`, the driver's own step
`gl2 rerun` (right after `gl`, `--gl2-rerun auto`, the default) runs the re-run SPEC 3.7.4 names,
`python3 -m plan11_encoding_ladder.models splits --out <out> --rung apf --all-splits --raw-and-norm
--feature-drop cov,std,peak2med --null-perm 500 --base-dir splits_gl2drop`, into
`gates/splits_gl2drop/apf/` (the refused run stays under `gates/splits/apf/`) and records it in
`gates/gl2_rerun.json` and in the ledger under `argv_nested`; with `--gl2-rerun manual` the step
records `not run: manual` and the command above is yours to run. The tables do not change: the
refused run's numbers stay printed with the G-L refusal beside them, and the rung's numbers are
not citable while the refusal stands (SPEC_epoch2 B10).

Look at: LOKO norm inside the null (the measured blind spot); LORO collapsing after
normalization; the level-matched sets (`level set` column A, B and C; C added 2026-09-28, AA A12) at chance; G-X `pooling
stands`; every row's rank `rank r of 500`; a `near_unfalsifiable` split prints that string in
every score cell of the split (its score is never printed). If a level-matched pair separates
on variance or duty, read `alias.csv` (`table6_feature` rows) before reading it as workload.

### Move 8: the fused plane

```
python3 -m plan11_encoding_ladder.gates_readings gj --out <out>
python3 -m plan11_encoding_ladder.figures --out <out> --only fused_plane
```

Writes `gates/gj.csv`, `gates/gj.json`, `gates/gj_mask/<cell_id>.npy` (two boolean columns,
`mask_K` applied by default and `mask_persist` under `--fused-plane-mask persist`; al-Kindi
review 9), `report/figures/fig_fused_plane.pdf/.png`.

Look at: clouds separated along `l0` by content type; J near one except the slow-pass kernel
and the lexer at the floor; masked pairs hollow. With no idle cell every pair is `floor
unmeasured` and J is reported unmasked and labelled.

### Move 9: persistence in full

```
# the same five temporal commands as move 6, with --rung persist:
python3 -m plan11_encoding_ladder.series features --out <out> --rung persist --all-grid --both
python3 -m plan11_encoding_ladder.gates_temporal grid   --out <out> --rung persist
python3 -m plan11_encoding_ladder.gates_temporal g3     --out <out> --rung persist
python3 -m plan11_encoding_ladder.gates_temporal gord   --out <out> --rung persist
python3 -m plan11_encoding_ladder.gates_temporal select --out <out> --rung persist
python3 -m plan11_encoding_ladder.models splits --out <out> --rung persist --all-splits --norm --null-perm 500 --n-jobs 4
python3 -m plan11_encoding_ladder.gates_comparison gx --out <out> --rung persist --null-perm 500
python3 -m plan11_encoding_ladder.figures --out <out> --only j_hist
```

Writes the same files as moves 6 and 7 for `persist`, plus `fig_j_hist`.

Look at: the gemm dip in every rep (G-C); G-P's verdict beside every reading (nbody aliased by
design; the rest undeclared and labelled so); the `frac_below_null` column of `gj.csv`.

### Move 10: the content-change rung in full

```
python3 -m plan11_encoding_ladder.series features --out <out> --rung content --all-grid --both
python3 -m plan11_encoding_ladder.run_moves plan --out <out> --moves 10   # prints the four gates_temporal lines (grid, g3, gord, select --rung content) exactly as move 9 shows them
python3 -m plan11_encoding_ladder.models splits --out <out> --rung content --all-splits --norm --null-perm 500 --n-jobs 4
python3 -m plan11_encoding_ladder.gates_comparison gx --out <out> --rung content --null-perm 500
python3 -m plan11_encoding_ladder.gates_readings gdec --out <out>
python3 -m plan11_encoding_ladder.figures --out <out> --only ratio_hist,floyd_decay
```

Writes the content rung's grid, selection, splits, `gates/gdec.csv`, `fig_ratio_hist`,
`fig_floyd_decay` (a placeholder carrying G-DEC's verdict unless it reads `decay`).

Look at: floyd and histogram separated; fft and gemm not; G-DEC's verdict (`decay not resolved`
is expected while floyd's pass period is undeclared: G-DEC admits the exhibit only when G-P's
within-pass verdict is `admitted`).

### Move 11: wAPF

```
python3 -m plan11_encoding_ladder.series features --out <out> --rung wapf --all-grid --both
python3 -m plan11_encoding_ladder.run_moves plan --out <out> --moves 11   # prints the four gates_temporal lines (grid, g3, gord, select --rung wapf) exactly as move 9 shows them
python3 -m plan11_encoding_ladder.models splits --out <out> --rung wapf --all-splits --norm --null-perm 500 --n-jobs 4
python3 -m plan11_encoding_ladder.gates_comparison gx --out <out> --rung wapf --null-perm 500
python3 -m plan11_encoding_ladder.tables --out <out> --only wapf_over_apf
```

Writes the wAPF rung's files and `report/tables/table_wapf_over_apf.csv/.md/.tex` (per kernel:
mean APF, mean wAPF, their ratio, the mean flipped bits per changed page).

Look at: crude separation of floyd from histogram in the ratio column.

### Move 12: the combined rung, the floor at the selected points, the comparisons, the report

```
python3 -m plan11_encoding_ladder.series features --out <out> --rung combined --all-grid --norm
python3 -m plan11_encoding_ladder.run_moves plan --out <out> --moves 12   # prints the four gates_temporal lines (grid, g3, gord, select --rung combined) exactly as move 9 shows them
python3 -m plan11_encoding_ladder.models splits --out <out> --rung combined --all-splits --norm --null-perm 500 --n-jobs 4
python3 -m plan11_encoding_ladder.gates_comparison gx --out <out> --rung combined --null-perm 500
python3 -m plan11_encoding_ladder.gates_comparison gl --out <out>                                # again: G-L (i) for every rung, now that every selection exists
python3 -m plan11_encoding_ladder.gates_precondition gf --out <out> --all-rungs --n-perm 500     # at the selected points (al-Farabi review 2.6)
python3 -m plan11_encoding_ladder.gates_comparison gdim --out <out>
python3 -m plan11_encoding_ladder.gates_comparison gm --out <out>
python3 -m plan11_encoding_ladder.variance --out <out>
python3 -m plan11_encoding_ladder.models cluster --out <out> --rung combined --null-perm 500
python3 -m plan11_encoding_ladder.tables --out <out>                        # all tables
python3 -m plan11_encoding_ladder.latex_skeleton --out <out>
python3 -m plan11_encoding_ladder.figures --out <out>                       # all figures
python3 -m plan11_encoding_ladder.tables --out <out> --only manifest
```

Writes `gates/gl.csv` (rewritten with a part (i) row per rung; the move-7 run had a selection
for APF only), `gates/gdim.csv`, `gates/gm.csv`, `gates/gv.csv`, `gates/gv_summary.csv`,
`gates/clustering.csv/.json`, the appended `gates/gf.csv` rows at the selected points,
`report/tables/*` (Tables 5, 5-G3, 6, 7, 8, G-V, the Table 4 status column, the preconditions
copy, wAPF over APF; each as `.csv`, `.md` and `.tex`), `report/figures/*` (eight figures as PDF
and PNG), `report/paper2_skeleton.tex`, `report/manifest.json` (every file under `report/` and
`gates/` with its sha256, the ledger and every `params` block).

Look at: Tables 5, 7, 8 and G-V; the first rung off the null in Table 7; Table 7's `G-C`,
`G-F (i)`, `G-L`, `G-DIM`, `G-M vs APF` and `G-X` columns; Table 8 read as assignments with
counts (never as a confusion matrix); the `physical reason` column of Table 8 is yours;
`manifest.json`.

To compile the skeleton beside its tables and figures:

```
cd <out>/report && pdflatex paper2_skeleton.tex
```

(IEEEtran conference class; `--documentclass article` on `latex_skeleton` when IEEEtran.cls is
not installed. **Do not point `--standalone-tex` at `apf_paper/p2_skeleton.tex`.** Since
2026-09-27 that file is hand-maintained and holds approved prose the scaffold does not; the writer
refuses to overwrite it. Write the scaffold elsewhere if you want to see the empty shells.)

### Move 13: the idle tripwire on every table row

The driver's own check (`gates/gf_check.json`): every row of Table 7 carries its G-F (i)
verdict (al-Farabi review 2.6). A `refused:` here means a rung's `gf.csv` part (i) row is
missing at its selected point: re-run move 12's `gf --all-rungs` and `tables`. With no idle
cell every row reads `not run: no admissible idle cell; admissibility record missing`, which is
the tripwire row of the claim in its corrected wording (idle reps must be mutually inseparable
under every rung; a kernel inseparable from idle is at floor in that lead).

### Move 14: the comparators (build epoch 2)

The three exact-input comparators of `council/14_hunayn_exact_input_comparators.md`, each
computed per cell and run through the same split stage and the same comparison gates as a rung
(`SPEC_epoch2.md` Part 1): Savoldi 2010 (U = mean +/- SD of the per-pair changed-page count K),
Dhodapkar-Smith 2003 (delta = 1 - Jaccard between consecutive changed sets, a phase boundary
when delta exceeds a threshold, stability, mean phase length; a declared sweep with every point
kept and 0.04 marked as the default, `P2_AUTHOR_ANSWERS.md` 2026-09-17), Law 2010 (pages dynamic
for X and static for X consecutive dumps from a per-page run-length index; X on the declared
grid 2, 4, 8, 16 with 4 the default). Savoldi and Dhodapkar-Smith serve the EUSIPCO table, Law is
held for the IFIP version (`P2E_STRUCTURE.md` section 7).

```
python3 -m plan11_encoding_ladder.comparators savoldi   --out <out> --null-perm 500 --null-splits loko,loro,within_trace --n-jobs 4
python3 -m plan11_encoding_ladder.comparators dhodapkar --out <out> --delta-th-default 0.04 --null-perm 500 --n-jobs 4
python3 -m plan11_encoding_ladder.comparators law       --out <out> --x-default 4 --jobs 4 --null-perm 500 --n-jobs 4
python3 -m plan11_encoding_ladder.comparators gates     --out <out> --null-perm 500 --n-jobs 4
python3 -m plan11_encoding_ladder.tables --out <out> --only table7_comparators,table_comparators
python3 -m plan11_encoding_ladder.figures --out <out> --only dhodapkar_sweep
python3 -m plan11_encoding_ladder.tables --out <out> --only manifest
python3 -m plan11_encoding_ladder.comparators all --out <out> --no-splits     # the statistics and the feature files only (no split stage, no gates), for a first look
```

Writes `gates/comparators/`: `savoldi.csv` (per cell: `K_mean`, `K_sd`, `U_text`),
`savoldi_per_kernel.csv`, `dhodapkar_sweep.csv` (one row per cell and threshold, the whole
sweep, never filtered), `dhodapkar.csv` (the default-threshold row with the delta quantiles),
`dhodapkar_per_kernel.csv`, `law_sweep.csv` (one row per cell and X), `law.csv` (the default-X
row with `check_x2_equals_K`, the built-in check that the pass agrees with `extract.csv`),
`law_cells.json` (the pass counters, resumable per cell), `law_series/<cell_id>.npz`,
`law_per_kernel.csv`, each with its `.params.json`; `features/cmp_<name>/Wall_Hall_{raw,norm}.npz`
(one vector per cell: the raw row is the method as published, the norm row the count-rung
normalization by the cell's median K, or by the pair count for Dhodapkar-Smith); the split stage
under `gates/splits/cmp_<name>/Wall_Hall/` (B1-G1, B1-G3, B1-G6, G-N, G-DIM as for a rung);
`gates/comparators/gl.csv` (G-L (i)), `gdim.csv`, `gm.csv` (G-M against APF, like against like,
with the spread of `gates/gm.params.json` from move 12), the comparators' rows in `gates/gx.csv`
(G-X), and `gates/comparators/verdicts.csv` (the whole record in one file);
`report/tables/table7_comparators.*` (the Table 7 rows: for each comparator the raw row and the
level-normalized row, `tables.TABLE7_VARIANT = "both"`), `report/tables/table_comparators.*`
(the methods' own per-kernel numbers), `report/figures/fig_dhodapkar_sweep.*`.

Look at: `U_text` per kernel in `savoldi_per_kernel.csv` and `table_comparators.csv` (the
per-run form of Savoldi's paper, "U = 65.5% +/- 0.15%"); the sweep figure (stability and mean
phase length against delta_th per kernel, the declared default as the dotted line); the Table 7
comparator rows against APF and the combined rung, with `G-M vs APF` read like against like
(raw against APF raw, norm against APF norm); `check_x2_equals_K` = `true` on every cell of
`law.csv`; `G-C`, `G-F (i)` and the temporal columns read `not applicable:` with the reason
(a comparator is a published reduction, not a lead of the ladder, and has one vector per cell).

Cost: Savoldi and Dhodapkar-Smith read the extracts (seconds). The Law pass re-streams every
trajectory (one to three minutes per cell per process, the extractor's own rate), so
`--comparator-jobs 4` on the driver (`--jobs 4` by hand) on the 96-cell corpus, resumable per
cell (`law_cells.json`; `--force` re-runs, `--only REGEX` restricts). The split stage costs what
a rung's does at the whole-cell point (one row per cell; within-trace is not applicable).
A declared default that is not a grid point is refused (exit 2); `--off-grid-default append`
adds it to the grid and records `default_appended_to_grid`. The G-M cells read `not run:
gates/gm.params.json missing (move 12)` until move 12 has measured APF's five-seed spread.

### The EUSIPCO outputs (build epoch 2): Table 2, Table 3, the five-page skeleton

Run after Table 7 exists and after the comparators' move has written its rows (the comparators
are move 14 in `SPEC_EPOCH2.md`; their rows land in `report/tables/table7_comparators.csv`, or
inside `table7.csv` as `comparator: <name> [<bib key>]` rows under the three-builder draft; both
layouts are read). The driver does not schedule these two commands in this epoch (its move table
is another builder's region; see `BUILD2_eusipco.md`), so run them by hand once move 14 is done,
and again whenever Table 7 or the comparator rows change:

```
python3 -m plan11_encoding_ladder.tables_eusipco --out <out>
```

**Do NOT run `latex_skeleton_eusipco --standalone <the live paper>`.** That line was here until
2026-09-27 and it is now wrong. `apf_paper/p2e_skeleton.tex` is hand-maintained: it forked from the
scaffold on 2026-09-17 and carries the author-approved prose, including the EUSIPCO/encoding
separation. The builder emits a scaffold with **zero prose lines**, so that command would delete
every approved sentence. `write_p2e_skeleton()` now refuses it unless `force_standalone=True`.

To see the scaffold without touching the paper, write it somewhere else:

```
python3 -m plan11_encoding_ladder.latex_skeleton_eusipco --out <out>
```

Options: `--only table2,table3`; `--comparators savoldi2010uncertainty,dhodapkar2003comparing`
(the EUSIPCO picks of `P2_AUTHOR_ANSWERS.md`, Decisions of 2026-09-17; add `law2010volatile` for
the IFIP version); `--include-matched` appends the `combined (matched)` row; `--comparator-variant
"as published"` (default; the method as published) or `level-normalized` chooses which of the two
comparator rows of `table7_comparators.csv` enters Table 2; `--ds-id` names the Dhodapkar-Smith
reading's directory when it is not `dhodapkar_smith` or `cmp_dhodapkar`. `tables_eusipco` exits 2
naming `report/tables/table7.csv` when Table 2 is requested without it; every other missing input
is a `not run:` cell naming the file.

Writes `report/tables/eusipco_table2.{csv,md,tex}` and `eusipco_table2.params.json` (rows APF, wAPF,
content-change, persistence, combined, Savoldi 2010, Dhodapkar-Smith 2003; columns feature count,
LOKO accuracy, LOKO macro recall over the headline archetypes, LOKO null p95, LOKO majority, LORO
accuracy, the G-M margin against APF; every cell copied verbatim from Table 7's LOKO/archetype and
LORO/kernel rows, refusals printed as words; the `.tex` cites each comparator through `\cite`);
`report/tables/eusipco_table3.{csv,md,tex}` and `eusipco_table3.params.json` (the level-matched
sets A = floyd, histogram, nbody and B = fft, gemm, declared, and C = fft, stencil_jacobi, added
2026-09-28 (AA A12; `status` column in every form, the `.tex` marks the added row and carries a
`\slot` note for the author), under APF, content-change, persistence and
Dhodapkar-Smith 2003: the number of features separating a pair of the set under the envelope rule
of `gates_calibration.separating_features`, the feature count, the set-mean LORO kernel recall,
the within-set confusion, then the alias verdicts of `gates/alias.csv` for APF and gemm's G-P line
from `gates/gp.csv`; the CSV carries the four numbers per reading, the `.md` and `.tex` one compact
cell `k/d sep; LORO r; conf c`, with `none` when k = 0); `report/p2e_skeleton.tex` and
`report/p2e_skeleton.json`. **No standalone copy is written to `apf_paper/`**: that file is
hand-maintained (see the warning above).

Look at: Table 2's comparator rows against APF and against the combined rung (read honestly:
better, worse, or within the G-M margin); the `params.json` line `comparator_sources`, which names
the file and the exact Table 7 row each comparator came from (for Dhodapkar-Smith that row's label
carries the threshold that ran and whether it was the declared default); Table 3's APF column,
which the claim expects to read `none` on both rows, and which axis separates each set; the
`not run: no selection for <rung>` cells until every rung's selection exists (moves 6, 9, 10);
`p2e_skeleton.json` `n_cite` (19 to 23 placeholders keyed to `apf_paper/p2.bib`) and
`n_prose_lines` (always 0). Table 3 reads the `_norm.npz` feature file of every reading (al-Kindi's
envelope rule is over the normalized features), so the Dhodapkar-Smith column of Table 3 and the
`as published` row of Table 2 are two readings of the same comparator; both are recorded in the
params files.

To compile the skeleton beside its tables and figures:

```
cd <out>/report && pdflatex p2e_skeleton.tex
```

(IEEEtran conference class; `--documentclass article` when IEEEtran.cls is not installed. The
`\cite` placeholders resolve only with `p2.bib` beside the file and the two bibliography lines
uncommented; without them pdflatex prints `[?]` and warns, which is expected for the skeleton.
No TeX installation exists on the build machine, so the skeleton was checked structurally only:
brace balance, every environment closed, five sections, four equations, zero prose lines.)

---

## 3. Adding the idle cells once they are captured

The idle capture (P2_AUTHOR_ANSWERS.md A3: `sleep 600`, 8 reps, 600 s, 500 ms, same pipeline
at `fcc184e`, guest rebooted per cell) lands in the retention layout like any cell. The role is
detected from the test label: any label containing `sleep` or `idle` (case-insensitive) is an
idle cell (SPEC 2.6; `extract index --idle-marker` adds markers). Then:

1. Re-run move 0 (`extract index`) so `cells.csv` gains the idle rows (`role = idle`,
   `archetype_predicted = control`, `cell_id = idle__rep<NN>__<campaign>`); check the eight
   rows, or edit `role` by hand for any cell the marker rule misses.
2. Re-run move 1 (`extract all`): only the new cells run; finished sidecars are skipped.
3. Fill `inputs/idle_admissibility.json` (the template of move 2: `same_ssh_path`,
   `rebooted_per_cell`, `interval_ms`, `differ_speed`, `image_state_note`, `duration_s`,
   `capture_loop_note`). Without it every G-F row stays `not run: no admissible idle cell;
   admissibility record missing`.
4. Run the driver again from move 2. The staleness rule does the rest: `cells.csv` changed, so
   the preconditions, G-K0, G-F, G-J and every split stage (whose inputs include `gk0.csv`)
   re-run, and the tables and figures follow. Expect: G-K0's idle band edge measured, the lexer
   `IDLE, measured` (its LOKO label becomes IDLE and it leaves the recovery denominator), G-F
   part (i) `inseparable at floor` on every rung, the fused plane with the idle cells in grey,
   the J histograms with the floor's quantiles as vertical lines, Table 8's `IDLE (measured)`
   column filled, and the tripwire rows of move 13 in the corrected wording. C1 switches from
   the absolute of 262 pages to the idle floor's 95th percentile on this re-run;
   `preconditions.json` records the switch (`C1_rule_in_force`), and because that file is the
   admissibility input of every later move, everything from move 3 re-runs. A kernel admitted
   under the absolute can be refused under the floor if the measured idle p95 is above its
   footprint; read `C1_K_max` against `C1_threshold_pages` for any newly excluded cell.
5. If the idle cells were captured after the kernels with a different image state, say so in
   `image_state_note`; Limitations states the mismatch (P2 Sec. IX).

Until the idle cells exist, every floor reading is `not run` or `floor unmeasured` and labelled
so in the tables; nothing is filled in by assumption.

---

## 4. Where things are

```
<out>/cells.csv                     the cell index (move 0)
<out>/extract/<cell_id>/            extract.csv, sidecar.json (move 1)
<out>/inputs/                       your inputs: pass_table.csv, gk0_source.csv, head_drop.csv,
                                    idle_admissibility.json, failed_counts.csv, cell_order.csv
<out>/features/<rung>/              feature matrices at all 13 grid points
<out>/gates/                        every gate's result files (see SPEC section 1 for the list)
<out>/gates/grid/<rung>/<grid_id>/  every grid point, kept
<out>/gates/splits/<rung>/<grid_id>/<split>__<labelspace>/   predictions, scores, null, quarantine
<out>/report/tables/                table{5,5_g3,6,7,8,gv,4_status}, preconditions, table_wapf_over_apf (.csv .md .tex .json)
<out>/report/figures/               fig_*.pdf and .png, figures.json (or SKIPPED.txt)
<out>/report/paper2_skeleton.tex    the LaTeX skeleton beside its tables and figures
<out>/report/tables/eusipco_table{2,3}.{csv,md,tex,params.json}   the EUSIPCO tables (build epoch 2; tables_eusipco)
<out>/report/p2e_skeleton.tex       the EUSIPCO five-page SCAFFOLD (not the paper; apf_paper/p2e_skeleton.tex is hand-maintained)
<out>/report/manifest.json          sha256 of every report and gate file, the ledger, every params block
<out>/driver_state.json             the move ledger
```
