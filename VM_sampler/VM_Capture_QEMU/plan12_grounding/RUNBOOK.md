# plan12_grounding: runbook for the author

Written 2026-10-01 by the builder of slice 3. The command-line companion to `SPEC.md`: how to run the
smoke and the real run, what each move writes, how to resume, and what never to do. The contract is
`SPEC.md`; where this file and the SPEC disagree, the SPEC wins and this file is wrong.

Conventions. Every command is run from `VM_sampler/VM_Capture_QEMU/` as
`python3 -m plan12_grounding.<module> ...`. `<out>` is the output folder you choose (decision D4;
default `~/.cache/plan12/grounding_run_<date>`); everything the engine writes goes under it, plus the
per-page store (`--store`) and a fetch cache (`--cache`). The driver runs every move for you in order
and keeps the record book `<out>/driver_state.json`; a move is skipped when its outputs exist and
neither its inputs' sha256 nor its arguments changed, so re-running the same command resumes.

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

## 1. The smoke run (synthetic corpus, no server; about 40 minutes)

The whole chain on a synthetic corpus written by `synth_grounding.py` (the encoding toolkit's writer
with a varying cosine column), through a `--root` on the laptop. The encoding run that move 6 needs
(decision D2) is made by the encoding toolkit's own driver, moves 0 and 1, on the same root
(`--duration-s` = pairs x 0.644, the synthetic spacing).

```
cd VM_sampler/VM_Capture_QEMU
S=~/.cache/plan12/smoke
python3 -m plan12_grounding.synth_grounding corpus --root $S/root --n-pairs 300 --jobs 4
python3 -m plan11_encoding_ladder.run_moves run --out $S/encoding_out --root $S/root --moves 0-1 \
    --n-jobs 4 --duration-s 193.2 --assume-failed-zero --assume-reason "smoke run"
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
`--n-shuffles` are smoke sizes; every null then reads `not run: N permutations < 500` where the
encoding toolkit's rule applies. The synthetic idle runs have a constant page count, so move 7's
power ratios against idle are meaningless there (`n_idle_zero_power_bins` says so).

## 2. The real run (the author starts it)

The source is the laptop console's own: `--ssh jeries@cybersecurity.ac.upc.edu --remote-root
/mnt/nfs/jeries/memory_traces/zstd_local` (`mem_sig/docs/briefs/DATA_OPS_BRIEF.md` section 5). The
engine reads the server in fetch mode only: `rsync` of one trajectory at a time into `--cache`,
extraction into `--store`, and the fetched copy is removed only after the server copy is verified by
size and sha256 (the console's rule). Nothing is ever written on the server.

Decisions to fill before the first command (SPEC section 9):

- **D2**, the baseline: `--encoding-out <the encoding run's output folder>`, on the laptop. Move 6
  reads its `cells.csv`, `extract/<cell>/extract.csv` and `sidecar.json`, `gates/selection.json`,
  `gates/gk0.csv` and `gates/preconditions.csv` in place, and move 0 copies `cells.csv` under
  `<out>/inputs/encoding_run/` with sha256. If the run lives on the server, copy its folder to the
  laptop first (`rsync -a jeries@cybersecurity.ac.upc.edu:/project/homes/jeries/<run>/ ~/encoding_run/`,
  read only; it is a few hundred MB with the extracts) and name that copy.
- **D3**, the window: taken from the encoding run's `gates/selection.json` (the combined rung's
  selected point). When that file is absent, W=8, H=4; `moves/06_classify/classify.json` records
  which.
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
print what it would do (for the ssh source: the listing, the rsync and the verify commands) and touch
nothing; the record book records the dry run without demoting a done move. The default `--store`
is the console's (`~/.cache/plan10/l1`), so a recording the console already extracted with the same
columns is reused, not fetched again.

Move 6's settings are the encoding paper's Table 2: 300 trees, 500 permutations, unit = cell, LOKO on
archetype labels. The null runs by default on LOKO and within-trace (`--null-splits loko,within_trace`,
the encoding run's own paper preset): LORO's null costs 96 folds per permutation (about 48,000 forest
fits per encoding and cut at 500 permutations), which is what stopped the encoding run's own move 7.
LORO's point estimate always runs. To pay for LORO's null, add `--null-splits loko,within_trace,loro`
(hours to days). `--n-jobs` sets the processes; changing it never makes a move stale.

Timing on the laptop (Apple M2): move 1 fetches and extracts one recording at a time (minutes each;
the first real run is the long one); moves 2 to 5 take minutes; move 6 at the default null splits
takes roughly an hour per cut with 300 trees; moves 7, 8 and 10 take seconds to minutes; move 9 repeats
moves 3 to 7 once more.

## 3. Resuming, forcing, status

- Re-run the same `run` command: every command whose outputs exist and whose inputs and arguments are
  unchanged is skipped (`skipped: outputs exist and inputs unchanged`); the rest runs.
- A move stopped mid-run (Ctrl-C, a crash) reads `not run` in `status` until a later attempt records
  every one of its commands; move 1 resumes per recording (a complete series is reused).
- `--force` re-runs every selected move; `--moves 6` runs move 6 alone (move 0's `cells.csv` must exist).
- A second driver on the same `<out>` is refused (`.driver.lock`, the writer's pid); a stale lock
  from a dead process is taken over.
- Changing `--encoding-out`, `--null-perm`, `--n-estimators` or `--seed-offset` makes move 6 stale
  (its arguments changed) and it re-runs; changing `--n-jobs` does not.

```
python3 -m plan12_grounding.run_moves status --out $OUT --root /any   # the record book, move by move
```

## 4. What each move writes (all under `<out>`)

| move | module | writes |
|---|---|---|
| 0 | `inputs.py` | `cells.csv` (one row per recording: identity, index status, keep-first, head drop, admissible, reason), `params.json` (cuts, D1 to D4, counts), `inputs/` (copies of the declared files and the encoding run's `cells.csv`, `inputs/sha256.json`), `moves/00_index/index.json` |
| 1 | `extract.py` | the per-page store (`--store`), `series/<cell>.npz` (pair, N, H, A, C, S, A_unweighted, n_at_90, n_at_0, hist18, meta), `moves/01_extract/extract.json` and `recordings.csv` |
| 2 | `sanity.py` | `moves/02_sanity/{counts.csv, violations.csv, sanity.json}`; a violation stops the run |
| 3 | `figures.py every-run` | `moves/03_every_run/cut<H>/{average,overlay,runs}/*.svg + .csv`, `index.html`, `figures.json` |
| 4 | `figures.py portraits` | `moves/04_portraits/cut<H>/portrait_<S>.svg + .csv`, `index.html`, `figures.json` |
| 5 | `stats.py hand-check`, `similarity.py` | `moves/05_similarity/hand_check.txt`; `cut<H>/{stats_per_run.csv, icc.csv, icc_bars.svg, pca_map.csv/.svg, pca_loadings.csv, pca_dropped.csv, spectral_*.csv, spectral_similarity.svg, loso.csv, loso_summary.csv, loso.svg, summary.json}`, `similarity.json` |
| 6 | `classify.py` | `moves/06_classify/{e0_declared.json, e0_identity.json, e0_identity/ (the recomputed extract), classify.json}`; `cut<H>/{scores.csv, margins.csv, gap.csv, recall_per_kernel.csv, predictions_<split>_<E>.csv, confusion_<split>_<E>.csv, null_<split>_<E>.csv, features.npz, excluded_cells.csv, bars.svg + .csv, confusion_loko.svg, confusion_loro.svg, summary.json}` |
| 7 | `floor.py` | `moves/07_floor/cut<H>/{floor_bins.csv, floor_means.csv, spectra_<S>_<mode>.csv, floor_<S>_<mode>.svg, spikes.csv, summary.json}`, `floor.json` |
| 8 | `startup.py` | `moves/08_startup/{spike_rate.csv, spike_rate_by_group.csv, bins.csv, spikes_per_run.csv, stencil_1298.csv (when present), startup.svg, startup.json}` |
| 9 | `removal.py` (only with `--room-removal`) | `moves/09_removed/{removal.json, removal_set.csv, pages_per_recording.csv, before_after.csv/.svg, series/, extract.json, 03_every_run/, 04_portraits/, 05_similarity/, 06_classify/, 07_floor/}` |
| 10 | `summary.py` | `report/table_per_kernel.csv + .html`, `report/table_overall.csv`, `report/figures/` (the figure set with `index.csv`), `report/manifest.json` (sha256 of every output), `moves/10_summary/summary.json` |

Every JSON record carries the command, the package version, the toolkit fingerprint (sha256 of every
`plan12_grounding/*.py`) and the time; every SVG has its data as a CSV of the same name.

The cuts: every number of moves 3 to 7 and 10 is computed twice, at 16 pairs (the declared value) and
at 112 (the council's measured end of the start-up), as `cut16/` and `cut112/`. A cut of H pairs drops
the first H pairs of a recording's series in pair-index order (pair = the trajectory's seq + 1), the
same rows the encoding toolkit's `head_drop` drops.

Move 6's E0 check: before E0 is used, one recording (the first kernel cell, or `--identity-cell`) is
recomputed with `plan11_encoding_ladder.extract.extract_cell` from the trajectory at the source with
the sidecar's own parameters, and the two `extract.csv` must be byte-identical; the move stops
otherwise (`e0_identity.json` names the first differing row). For the ssh source this fetches that one
trajectory again (read only) and removes the copy after verification.

## 5. Never

- Never write under the encoding run's folder or under `plan11_encoding_ladder/`; move 6 reads them.
- Never start two drivers on one `<out>`; never run one output folder from two machines or two code
  versions (the record book is per `<out>`; the fingerprint is recorded with every command).
- Never touch the server beyond the fetch: no command here writes there, and none should be added.
- Never delete `series/`, the store or `driver_state.json` to "clean up": move `<out>` to `~/.Trash`
  whole if it must go.
- Never read, list or name the sandbox family's recordings: move 0 counts non-corpus rows and never
  names them (its index mirror is removed after use).
