# plan12_grounding: runbook for the author

Written 2026-10-01 by the builder of slice 3; revised the same day after the first check pass (fixes 1).
The command-line companion to `SPEC.md`: how to run the smoke and the real run, what each move writes,
how to resume, and what never to do. The contract is `SPEC.md`; where this file and the SPEC disagree,
the SPEC wins and this file is wrong.

Conventions. Every command is run from `VM_sampler/VM_Capture_QEMU/` as
`python3 -m plan12_grounding.<module> ...`. `<out>` is the output folder you choose (decision D4;
default `~/.cache/plan12/grounding_run_<date>`); everything the engine writes goes under it, plus the
per-page store (`--store`) and its own fetch cache (`--cache`, default `~/.cache/plan12/fetch`, never
the console's). The driver runs every move for you in order and keeps the record book
`<out>/driver_state.json`; a move is skipped when its outputs exist and neither its inputs' sha256,
its arguments, nor the code it ran with changed, so re-running the same command resumes.

## 0. Prerequisites

Python 3.10 with numpy, scipy, scikit-learn, matplotlib (the laptop: 3.10.12, numpy 2.2.6, scipy 1.15.3,
scikit-learn 1.7.2, matplotlib 3.10.7), `zstd` on the PATH, and `rsync` for the ssh source. The engine
imports `plan11_encoding_ladder` (read only), `plan10_analysis` (the console's sources and per-page
store) and `plan08_b1` (B1's shape features); nothing is installed.

```
cd VM_sampler/VM_Capture_QEMU
python3 -c "import numpy, scipy, sklearn, matplotlib; print(numpy.__version__, scipy.__version__, sklearn.__version__)"
which zstd rsync
```

## 1. The smoke run (synthetic corpus, no server; about 45 minutes)

The whole chain on a synthetic corpus written by `synth_grounding.py` (the encoding toolkit's writer
with a varying cosine column), through a `--root` on the laptop. The encoding run that move 6 needs
(decision D2) is made by the encoding toolkit's own driver on the same root: its moves 0 to 2, so that
`cells.csv` and `gates/preconditions.csv` exist (move 0 stops without them), and, to give the Table 2
check something to compare with, its combined-rung split records at W8_H4 (the author's head-drop
input copied from the declared file first, as the real run does). `--duration-s` = pairs x 0.644.

```
cd VM_sampler/VM_Capture_QEMU
S=~/.cache/plan12/smoke
python3 -m plan12_grounding.synth_grounding corpus --root $S/root --n-pairs 300 --jobs 4
python3 -m plan11_encoding_ladder.run_moves run --out $S/encoding_out --root $S/root --moves 0-2 \
    --n-jobs 4 --duration-s 193.2 --assume-failed-zero --assume-reason "smoke run"
cp plan11_encoding_ladder/declared/head_drop_values.csv $S/encoding_out/inputs/head_drop.csv
python3 -m plan11_encoding_ladder.series features --out $S/encoding_out --rung combined --grid-id W8_H4 --norm
python3 -m plan11_encoding_ladder.models splits --out $S/encoding_out --rung combined --grid-id W8_H4 --all-splits --norm \
    --null-perm 5 --null-splits loko,within_trace --n-estimators 30 --n-jobs 4
python3 -m plan12_grounding.run_moves run --out $S/out --root $S/root --store $S/l1 \
    --encoding-out $S/encoding_out --allow-unmatched-declared \
    --n-shuffles 200 --null-perm 5 --n-estimators 30 --n-jobs 4 --moves 0-10
python3 -m plan12_grounding.run_moves run --out $S/out --root $S/root --store $S/l1 \
    --encoding-out $S/encoding_out --allow-unmatched-declared \
    --n-shuffles 200 --null-perm 5 --n-estimators 30 --n-jobs 4 --room-removal --moves 9,10
python3 -m plan12_grounding.run_moves status --out $S/out --root $S/root --room-removal
```

`--allow-unmatched-declared` is for the smoke only: the declared keep-first rows name three real
recordings that a synthetic corpus does not have. The small `--null-perm`, `--n-estimators` and
`--n-shuffles` are smoke sizes; every null then reads `not run: N permutations < 500`, the encoding
toolkit's own rule. The synthetic idle runs have a constant page count, so move 7's power ratios
against idle are meaningless there (`n_idle_zero_power_bins` says so), and they share no
always-changing page, so move 9 removes nothing on the smoke.

## 2. The real run (the author starts it)

The source is the laptop console's own: `--ssh jeries@cybersecurity.ac.upc.edu --remote-root
/mnt/nfs/jeries/memory_traces/zstd_local` (`mem_sig/docs/briefs/DATA_OPS_BRIEF.md` section 5). The
engine reads the server in fetch mode only, and only in move 1: `rsync` of one trajectory at a time
into `--cache` (its own folder), extraction into `--store`, and the fetched copy is removed only after
the server copy is verified by size and sha256 (the console's rule). The E0 identity check runs in
move 1 too, on the one trajectory it has in the cache at that moment. No other move reaches the
server; nothing is ever written on it.

Decisions to fill before the first command (SPEC section 9):

- **D2**, the baseline: `--encoding-out <the encoding run's output folder>`, on the laptop. Move 0
  reads its `cells.csv` and `gates/preconditions.csv` (both must exist; admissibility comes from them,
  which is where lexer seed 6898 is refused) and copies their kernel and idle rows under
  `<out>/inputs/encoding_run/` with the originals' sha256; move 1 recomputes one recording against its
  `extract/`; move 6 reads its extracts, `gates/selection.json`, `gates/gk0.csv` and the preconditions,
  all declared with sha256 in `moves/06_classify/e0_declared.json`. If the run lives on the server,
  copy its folder to the laptop first (`rsync -a jeries@cybersecurity.ac.upc.edu:/project/homes/jeries/<run>/
  ~/encoding_run/`, read only; a few hundred MB with the extracts) and name that copy.
- **D3**, the window: taken from the encoding run's `gates/selection.json` (the combined rung's
  selected point). When that file is absent, W=8, H=4; `moves/06_classify/classify.json` records
  which (`params.D3_grid_id`, `params.D3_window.source`).
- **D4**, the output folder: `--out ~/.cache/plan12/grounding_run_<date>`.
- **D1**, the removal: off; `--room-removal --moves 9,10` later, on the same `<out>` (it adds
  `moves/09_removed/`; nothing else is re-run).

```
cd VM_sampler/VM_Capture_QEMU
OUT=~/.cache/plan12/grounding_run_$(date +%Y%m%d)
python3 -m plan12_grounding.run_moves plan --out $OUT --moves 0-10 \
    --ssh jeries@cybersecurity.ac.upc.edu --remote-root /mnt/nfs/jeries/memory_traces/zstd_local \
    --store ~/.cache/plan10/l1 --encoding-out ~/encoding_run --n-jobs 4
python3 -m plan12_grounding.run_moves run --out $OUT --moves 0-10 \
    --ssh jeries@cybersecurity.ac.upc.edu --remote-root /mnt/nfs/jeries/memory_traces/zstd_local \
    --store ~/.cache/plan10/l1 --encoding-out ~/encoding_run --n-jobs 4
```

`plan` prints every command without running one. `--dry-run` runs the driver and makes each module
print what it would do (for the ssh source: the listing, the rsync and the verify commands) and write
nothing at all; the record book records the dry run without demoting a done move. The default
`--store` is the console's (`~/.cache/plan10/l1`), so a recording the console already extracted with
the same columns is reused, not fetched again; the fetch cache is the engine's own
(`~/.cache/plan12/fetch`), so its cleanup never touches the console's files.

Move 0 stops unless it finds exactly 8 idle recordings at `sleep/sleep/sleep_600/rep00N__idle_01c`
and every kernel recording is one of the twelve kernels; any other recording under the root is
counted in `params.json` and never named.

Move 6's settings are the encoding paper's Table 2: 300 trees, 500 permutations, unit = cell, LOKO on
archetype labels, B1-G3's quarantine with the re-run (`models.effective_scores`), the encoding run's
cell order and its pair-rung exclusion. The null runs by default on LOKO and within-trace
(`--null-splits loko,within_trace`, the encoding run's own paper preset): LORO's null costs 96 folds
per permutation (about 48,000 forest fits per encoding and cut at 500 permutations), which is what
stopped the encoding run's own move 7. LORO's point estimate always runs; its null verdict then reads
`not run: null not requested (--null-splits)`. To pay for LORO's null, add
`--null-splits loko,within_trace,loro` (hours to days). `--n-jobs` sets the processes; changing it
never makes a move stale. `moves/06_classify/table2_check.json` says whether E0 at the declared cut
reproduces each Table 2 row of the encoding run exactly, naming the first difference otherwise.

Timing on the laptop (Apple M2): move 1 fetches and extracts one recording at a time (minutes each;
the first real run is the long one); moves 2 to 5 take minutes; move 6 at the default null splits
takes roughly an hour per cut with 300 trees, plus the quarantine's one-feature fits; moves 7, 8 and
10 take seconds to minutes; move 9 repeats moves 3 to 7 once more.

## 3. Resuming, forcing, status

- Re-run the same `run` command: every command whose outputs exist and whose inputs, arguments and
  code are unchanged is skipped (`skipped: outputs exist and inputs unchanged`); the rest runs. The
  code is compared by a fingerprint per command: the sha256 of the plan12 modules that command reaches
  through its imports, recorded with the `done` that made its outputs (a skip record keeps that
  fingerprint and says what the fingerprint is now). Editing one module makes exactly the moves that
  reach it stale; `status` shows each move's fingerprint and names the commands whose code changed.
- A move stopped mid-run (Ctrl-C, the console's Stop, a crash) reads `not run` in `status` until a
  later attempt records every one of its commands; that attempt runs the move as if forced (its
  commands are stale), and move 1 then rebuilds every series. Every module turns a SIGTERM into an
  orderly exit, so nothing half-made is left behind (move 0's listing mirror, built outside `<out>`
  in the system's temp folder, is removed on the way out).
- `--force` re-runs every selected move and reaches move 1: every series is rebuilt from its store
  and the identity check runs again. Without it, move 1 reuses a series only when its metadata
  matches the current values (the keep-first cut, the cut convention, the series format, the sha256
  of `extract.py`) and its store exists.
- `--moves 6` runs move 6 alone (move 0's `cells.csv` must exist).
- A second driver on the same `<out>` is refused (`.driver.lock`, the writer's pid); a stale lock
  from a dead process is taken over.
- Changing `--encoding-out`, `--null-perm`, `--n-estimators` or `--seed-offset` makes move 6 stale
  (its arguments changed) and it re-runs; so does any change to the encoding run's files move 6
  reads (they are its declared inputs); changing `--n-jobs` does not.

```
python3 -m plan12_grounding.run_moves status --out $OUT --root /any   # the record book, move by move, with each move's code fingerprint
```

## 4. What each move writes (all under `<out>`)

| move | module | writes |
|---|---|---|
| 0 | `inputs.py` | `cells.csv` (one row per corpus recording: identity, index status, keep-first, head drop, admissible, reason), `params.json` (cuts and their convention, D1 to D4, where admissibility came from, counts), `inputs/` (copies of the declared files; the encoding run's `cells.csv` and `preconditions.csv` kernel and idle rows only, with the originals' sha256 and the count of rows left out; `inputs/sha256.json`), `moves/00_index/index.json` |
| 1 | `extract.py` | the per-page store (`--store`), `series/<cell>.npz` (pair, N, H, A, C, S, A_unweighted, n_at_90, n_at_0, hist18, meta with the series format, the code hash and the conventions), `moves/01_extract/extract.json` and `recordings.csv`, `moves/01_extract/e0_identity.json` and `e0_identity/` (the recomputed extract of the identity check) |
| 2 | `sanity.py` | `moves/02_sanity/{counts.csv, violations.csv, sanity.json}`; a violation stops the run (the magnitude rules; the store's seq column in file order with no gap, repeat or duplicate (seq, page) row; the series' pairs and recount) |
| 3 | `figures.py every-run` | `moves/03_every_run/cut<H>/{average,overlay,runs}/*.svg + .csv`, `index.html`, `figures.json` |
| 4 | `figures.py portraits` | `moves/04_portraits/cut<H>/portrait_<S>.svg + .csv`, `index.html`, `figures.json` |
| 5 | `stats.py hand-check`, `similarity.py` | `moves/05_similarity/hand_check.txt`; `cut<H>/{stats_per_run.csv, icc.csv, icc_bars.svg, pca_map.csv/.svg, pca_loadings.csv, pca_dropped.csv, spectral_*.csv, spectral_similarity.svg, loso.csv, loso_summary.csv, loso.svg, summary.json}`, `similarity.json` |
| 6 | `classify.py` | `moves/06_classify/{e0_declared.json, table2_check.json, classify.json}`; `cut<H>/{scores.csv (with the null verdict in the encoding toolkit's words, the score source and the quarantined features), margins.csv, gap.csv, recall_per_kernel.csv, predictions_<split>_<E>.csv (and predictions_full_model_* when a feature was quarantined), confusion_<split>_<E>.csv, null_<split>_<E>.csv, features.npz, excluded_cells.csv, bars.svg + .csv, confusion_loko.svg, confusion_loro.svg, summary.json}` |
| 7 | `floor.py` | `moves/07_floor/cut<H>/{floor_bins.csv, floor_means.csv, spectra_<S>_<mode>.csv, floor_<S>_<mode>.svg, spikes.csv, summary.json}`, `floor.json` |
| 8 | `startup.py` | `moves/08_startup/{spike_rate.csv, spike_rate_by_group.csv, bins.csv, spikes_per_run.csv, stencil_1298.csv (when present), startup.svg, startup.json}` |
| 9 | `removal.py` (only with `--room-removal`) | `moves/09_removed/{removal.json, removal_set.csv, pages_per_recording.csv, before_after.csv/.svg, series/, extract.json, 03_every_run/, 04_portraits/, 05_similarity/, 06_classify/, 07_floor/}` |
| 10 | `summary.py` | `report/table_per_kernel.csv + .html`, `report/table_overall.csv`, `report/figures/` (the figure set with `index.csv`), `report/manifest.json` (sha256 of every output; the console's `.console/` folder and any dot-path are left out), `moves/10_summary/summary.json` |

Every JSON record carries the command, the package version, the toolkit fingerprint (sha256 of every
`plan12_grounding/*.py`) and the time; every SVG has its data as a CSV of the same name.

The cuts: every number of moves 3 to 7 and 10 is computed twice, at 16 pairs (the declared value) and
at 112 (the council's measured end of the start-up), as `cut16/` and `cut112/`. A cut of H pairs drops
the first H pairs AND the last pair of a recording's series, in pair-index order (pair = the
trajectory's seq + 1): exactly the rows the encoding toolkit's `rung_series` drops, so a series of
n pairs has n - 1 - H rows after the cut, and E0's windows and the statistics' windows cover the same
pairs. Move 8 (the start-up) reads the uncut series. The series' pair axis is every pair of the kept
range: a pair without a changed page reads N = 0, H = 0 and an undefined angle (that is move 9's case;
on the data as captured such a pair is a sanity violation).

Move 1's E0 check: the first admissible kernel recording with an ok sidecar in the encoding run (or
`--identity-cell`) is recomputed with `plan11_encoding_ladder.extract.extract_cell` from the trajectory
move 1 has in hand, with the sidecar's own parameters, and the two `extract.csv` must be byte-identical;
move 1 stops otherwise (`e0_identity.json` names the first differing row). Move 6 uses E0 only when
that record passed and still names the encoding run's current extract; it never fetches.

## 5. Never

- Never write under the encoding run's folder or under `plan11_encoding_ladder/`; moves 0, 1 and 6 read them.
- Never start two drivers on one `<out>`; never run one output folder from two machines or two code
  versions (the record book is per `<out>`; the fingerprints are recorded with every command).
- Never touch the server beyond move 1's fetch: no command here writes there, and none should be added.
- Never delete `series/`, the store or `driver_state.json` to "clean up": move `<out>` to `~/.Trash`
  whole if it must go. To rebuild the series, run move 1 with `--force`.
- Never read, list or name the sandbox family's recordings: move 0 counts non-corpus rows and never
  names them (its index mirror lives outside `<out>` and is removed, on a stop too), and the copies
  of the encoding run's files hold kernel and idle rows only.
