# plan10_analysis: the Analysis Console and its runner

The analysis-side sibling of `plan07_campaign/ui/` (the capture console). Compose a scheme
as a graph of typed modules, over the recordings that actually exist (local or behind SSH),
be refused when it is invalid and warned when it is questionable, run it, and get a
feature file with a sidecar that records every choice and every acknowledged warning.

## Start it

```bash
cd VM_sampler/VM_Capture_QEMU
plan10_analysis/ui/analysis.sh                                   # local corpus, console.sh's default root
plan10_analysis/ui/analysis.sh --root /path/to/zstd_local        # local corpus elsewhere
plan10_analysis/ui/analysis.sh --ssh user@host --key ~/.ssh/id_ed25519 --remote-root /project/.../zstd_local
plan10_analysis/ui/analysis.sh --ssh user@host --key ~/.ssh/id_ed25519 --remote-root /project/.../zstd_local --remote
```

`--remote` runs the extraction **on the server**: the chain never crosses the network and only
the L1 npz comes back. It needs the repo there (where the capture console already puts it,
`--remote-repo`, default `$HOME/memorySignal/VM_sampler/VM_Capture_QEMU`) and a python with
numpy (`remote_python`, since a login shell may resolve a different interpreter than the venv
holding it). Test in the Source tab first: it probes the server and reports its python, numpy,
zstd, differ and trace root before any run starts.

The launcher scans the source, rebuilds the served console against it, starts the bridge on
`127.0.0.1:8766` and opens the browser at the tokenised URL it prints. Ctrl-C stops the
bridge; a launched run is its own process and keeps going.

Needs: `python3` (3.10+), `numpy`, `PyWavelets`, `kymatio` (see `requirements.txt`), the `zstd` CLI, the differ binary
(`cd VM_sampler/VM_Capture/live_delta_calc_modular && cargo build --release`, or set
`PLAN10_DIFFER`), and for an SSH source `ssh` and `rsync`.

## Use it

1. **Source** tab (bottom drawer): local path, or host / user / key / remote root for SSH.
   Test, then Reload: the archive's own manifest is read in one round trip and the Cells module
   selects from it. Its picker filters by family, and a chip per workload selects or clears
   every usable seed of that workload; ticking one row keeps the list where it was scrolled. Scan is the slow reconciliation (a full walk that rewrites that manifest);
   it is for an archive nobody registered into, not for every session.
2. Drop modules from the palette, pipe output ports to input ports. Ports are typed. Load an
   example from the header to start from a working graph. Drag a node anywhere on it to move
   it; the pipes follow. The tidy button beside the zoom controls lays the graph out by depth.
3. Fix what is red (hard), acknowledge what is amber (soft) in the inspector with a note.
   Save scheme writes the JSON; Launch runs it.
4. **Run** tab: differ speed (default the config's), max pairs (0 = all), progress, log,
   pause, stop. **Results** tab: every run, its rows, its sidecar. **Monitor** (header) is the
   full-screen view of the same run.
5. **Explore** (header, or the Results tab's button): plots and tables of what a run wrote,
   and runs side by side. See "Explore" below.
6. **Learn** (header): pipelines of preprocessing, models and scores over what runs wrote,
   every configuration of a sweep, per split and fold; results through the same views plus
   the ones learning needs. See "Learn" below.
7. **Encoding paper** (header): the encoding paper's toolkit, `plan11_encoding_ladder`, launched
   one move at a time in its runbook's order and its output files shown as they are. See
   "Encoding paper" below.

Runs land under `~/.cache/plan10/runs/<label>/`: `features.npz` (`X`, `feature_names`,
`tile_keys`), `features.csv`, `sidecar.json`, `status.json`, `run.log`. Extracted channels
are cached under `~/.cache/plan10/l1/` per (recording, speed, channel set) and reused.

## The pieces, each separable

| File | Does | Test |
|---|---|---|
| `channel_roster.py` | the 64 columns and the speed each dies at, parsed from the differ's Rust and reconciled with its HELP table | `test_plan10_channel_roster.py` |
| `corpus_manifest.py` | recordings from a listing (`scan_listing`); local walk and SSH `find` produce the same manifest; `scan_source` prefers the archive's manifest and walks only when there is none | `test_plan10_corpus_manifest.py` |
| `archive_manifest.py` | the archive's own inventory at `<root>/.manifest/manifest.json`, maintained by its writers (`register`, `unregister`) under an NFS-safe lock; `rebuild` is the reconciling walk | `test_plan10_archive_manifest.py` |
| `sources.py` | `LocalSource` / `SshSource`: listing, fetch to cache, test | `test_plan10_sources.py` |
| `known_issues.py` | the flag registry, every number recomputed from the artifact it cites | `test_plan10_build.py` |
| `modules.py` | module and port registry; feature lists read from the implementing code | `test_plan10_scheme.py` |
| `scheme.py` | the scheme format, validator (hard / soft / note), estimate, examples | `test_plan10_scheme.py` |
| `runner/chain.py` | walk a zstd patch chain with a two-file rolling window | `test_plan10_runner_extract.py` |
| `runner/differ.py` | run the differ on a pair, parse its sparse CSV; refuses dumps not a multiple of 4 MiB | same |
| `runner/extract.py` | the L1 store | same |
| `runner/trajectory.py` | read the substrate trajectory a capture already wrote: its header is what a recording can serve, its rows become the L1 without a re-diff | same |
| `runner/stages.py` | one pure function per module kind; reuses b1_features, CepstrumStability, PLVStability, plan04_cusum, normal_profile | `test_plan10_runner_executor.py` |
| `runner/executor.py` | order, run per recording, status, control, output, sidecar | same |
| `results_view.py` | what a run produced, summarised and aggregated in numpy for the Explore view: per-metric statistics, the five views, several runs aligned by metric name | `test_plan10_results_view.py` |
| `learn/registry.py` | the Learn palette: tiers, shapes, modules, what each needs, what is unbuilt | `test_plan10_learn.py` |
| `learn/pipeline.py` | the pipeline format, its validator (hard / soft / note), the sweep, the worked examples | same |
| `learn/data.py`, `learn/splits.py` | rows from features.npz, paths and images from tiles.npz; grouped folds (within-trace, leave-one-recording/workload/campaign-out) | same |
| `learn/preprocess.py`, `learn/models.py`, `learn/scores.py` | fitted transforms; one wrapper over sklearn, numpy and torch models; the scores, the null, the bootstrap, the explanations | same |
| `learn/executor.py` | run every configuration; status, control, outputs, sidecar | same |
| `learn/results.py` | a Learn run as frames for the five views, plus confusion, ROC, embedding, saliency, importance, null, calibration, training curves, tiles as drawn | same, and `test_plan10_bridge.py` |
| `encoding_panel.py` | the Encoding paper panel's backend: the toolkit's driver launched per move, its ledger read for the board's states, its files served as they are, the launch records | `test_plan10_encoding_panel.py`, and `test_plan10_bridge.py` |
| `ui/analysis_bridge.py` | the local HTTP backend | `test_plan10_bridge.py` |
| `ui/build_analysis_console.py` | injects everything above into the template; static build has no network code | `test_plan10_build.py` |
| `ui/analysis_console.template.html` | the console; the bridge client sits between SERVED_ONLY markers | loaded in a browser |
| `testing/synth.py` | a 4 MiB-dump synthetic corpus with known answers | used by the runner tests |

Tests are plain asserts (`python3 tests/test_plan10_*.py`) or pytest.

## Reading the capture's own trajectory

A capture run with `CAPTURE_METRIC=substrate` already ran the differ on every pair and left the
result beside the chain: `run_matrix_test<N>_<workload>.npy.substrate_trajectory.csv.zst`, one row
per changed page per pair, 64 columns. Walking the chain again reproduces it at hours per
recording. Measured on the M2 laptop, one 931-pair recording: the differ path is ~7.5 s per pair
(~2 s differ, ~5.5 s rebuilding a 1 GiB dump from the delta chain), about 2 h, after a 27 min
11 GB fetch; the trajectory path reads the same rows in 54 s after a 755 MB fetch.

So the executor tries the trajectory first. `corpus_manifest` reports it from the listing
(`has.substrate_join: in-chain`), which is what an SSH source needs since a metrics root is
local-only. `SshSource.fetch_trajectory` pulls the CSV alone. When it carries every requested
column, `extract.extract_from_trajectory` writes the same L1 store as `extract.extract` (same key,
same arrays) and the run log says so; otherwise the chain is fetched and re-diffed, and the log
names the missing columns. A trajectory whose header reads but whose body does not, a truncated
or damaged file, is re-diffed from its chain the same way, logged and recorded as
`trajectory_error`, rather than ending the run. The fetch always lets rsync check a cached
trajectory, so a transfer interrupted earlier is finished rather than read as it stands.
The Channels module shows the same fact per channel: a green ring is in
every selected trajectory (read directly), amber in some, dimmed in none; headers are read on the
server in one round trip via `/trajectory_columns`, never fetched.

Two things the file cannot tell you, and the store records as such. The differ speed it was made
at is unrecorded (config `substrateSpeed`, the value the console labels "assumed"); the L1 meta
carries `speed_assumed: true`. `n_pages` is the config default unless the data addresses a page
beyond it. Two format facts are load-bearing: the trajectory numbers pairs from 0 and `walk_chain`
from 1 (the reader adds one), and `differ.parse_sparse_csv` cannot read it (it takes `row[0]` as
the page index, which here is `seq`), hence a separate reader.

The **Monitor** button in the header is the full-screen view of a run: fetch and differ progress
as separate bars (the fetch reports nothing of its own; the bar sizes the cache against the
archive's byte count), every selected trace by name, counts, inputs, outputs, and the log.
`/cache/drop` deletes fetched chains only after re-stat'ing the archive copy over ssh and matching
snapshot count and bytes exactly; the archive is never written to, so there is nothing to send back.

The executor does the same for itself, per recording, in ssh fetch mode: once a recording's L1
store is complete (written, or reused), its fetched files, the trajectory or the chain members,
leave the cache. Before any deletion one read-only ssh round trip checks that the server still
holds an identical copy of every file about to go: present at the path it was fetched from, same
size, same sha256 (`wc -c`, `sha256sum`; nothing is written). If any file fails any check, or the
server cannot be reached, nothing of that recording is deleted, the files stay and the log says
which check failed; the analysis itself is never stopped by it. Only paths under the cache are ever
removed; the L1 stores stay. A later scheme that needs a column the store lacks fetches again.
`--keep-fetched` (the Run tab's "keep fetched files") turns it off. The sidecar records the setting
and, per recording, verified or not and which check failed, deleted or kept, and MB freed.

## The archive keeps its own manifest

The archive is append-only at the file level: a snapshot is written once and never rewritten,
so the only party that knows when a recording is complete is the tool that put it there and
verified it. That tool registers it in `<root>/.manifest/manifest.json`, and every reader,
the console first of all, reads that one file instead of stat-ing a hundred thousand snapshots
over NFS (the full walk took 239 s under capture load; the manifest reads in under a second).
A recording still arriving is not in the manifest and so cannot be selected: there is no
stability guess.

The file is a corpus manifest (`plan10.corpus_manifest.v1`, the shape `corpus_manifest.py`
produces from a walk) so the console consumes it unchanged, plus an `archive_manifest` block
(`rebuilt_at`, `updated_at`, `registered_since_rebuild`, `last_registered`) and, per recording,
`has.substrate_columns`: the trajectory's header, read in place at registration. That is what
lets the Channels module light its rings on page load without a fetch or a round trip.

Who writes it: `plan07_campaign/ui/migrate_agent_server.sh` after each verified move from
`/project` to the NFS archive, `plan07_campaign/ui/place_csv.py` after placing a trajectory
beside its chain, and the laptop's `push_to_nfs.sh` after a verified push. Each calls
`python3 plan10_analysis/archive_manifest.py register <root> <rel>`. Concurrent writers take a
lock (an atomically created directory, since flock is unreliable on NFS; a lock older than
120 s is a dead writer and is broken; release renames the directory away first, because the
NFS client can leave a silly-renamed `.nfs*` file inside that defeats a plain rmdir), rewrite
to a temp file and rename it over the old one, so a reader never sees a partial manifest.

`rebuild <root>` walks the whole archive and rewrites the manifest from what is there. It is
the safety net for files copied in by hand and the only path that ever walks the archive.
A registration made while its walk runs is kept from the live manifest, not from the walk's
stale glimpse of a directory that was still filling. In the console, **Reload** reads the
manifest; **Scan** runs `rebuild` on the source and then reads it. The bridge's startup line
and the header say which one the corpus came from.

## Explore: what a run wrote, and runs side by side

`features.npz` is tiles x metrics with six keys per row (`recording, workload, family, block,
t_index, seq_start`). Every plot the console offers is one or two metrics, one grouping key,
one statistic over that:

| view | draws | table under it |
|---|---|---|
| distribution | one horizontal box per group (p25-p75, median line, mean diamond, p5-p95 whiskers, min/max dots, n), or a histogram over edges every group shares | n, n_nan, min, p5, p25, median, p75, p95, max, mean, std per group |
| over time | one line per group of the chosen stat at each `t_index` (or `seq_start`), p25-p75 band where several rows share an x, hover reads every series at that x | rows, points, x and y ranges per series |
| matrix | rows key x columns key, one sequential ramp, value written in when the cell is wide enough, n in the tooltip | the matrix |
| scatter | two metrics, one colour per group, at most 5000 points by a fixed stride | n, drawn, Pearson r, Spearman rho per group |
| table | nothing | one stat of every metric per group |

The numbers are the bridge's, not the page's: `/results/summary?label=L` and
`/results/agg?labels=A,B&view=&y=&x=&group=&stat=&scale=&bins=&rows=&cols=` call
`results_view.py`, which computes in float64 from the float32 the run wrote, NaN-aware, and
rounds to 6 significant digits (the precision of `features.csv`). Percentiles are numpy's
linear interpolation; std is the sample standard deviation (ddof 1). The caption under each
plot says what every mark means, and the header line carries the sidecar facts (written,
speed and whether it was assumed, where extraction ran, acknowledged warnings), so a figure
is never separated from how it was made.

**Comparison.** "Add to comparison" puts the current run on the right-hand pane. Several runs
aggregate as one frame: metric columns aligned by name (the intersection, in the first run's
order), rows stacked, a `run` key added, so the same views group by run, by run and family,
or by family with the runs pooled. A run lacking the chosen metric is left out and named in
the caption. The set of runs and every control persist in the browser (localStorage), so a
bridge restart's new URL does not lose them.

"Copy table as CSV" and "Download CSV" export the table as drawn. Paper figures are not made
here: read the same `features.npz` from a script.

## Learn: pipelines over what runs wrote

Designed in `LEARN_DESIGN_BRIEF.md`; built 2026-09-17. The view composes a **pipeline** as a
chain of slots in tier order (input, preprocess, represent, model, score, output); each slot
holds one or more alternatives and the run is the product of them, over the chosen splits and
seeds. That is "execute every possibility", counted in the verdict and recorded in every result.
The graph is a chain rather than free-form piping: one board, one card per alternative.

**Inputs.** `features.npz` (rows: one vector of metrics per tile) or `tiles.npz`, which the
scheme's new `Write tiles` module writes beside the features: a collapsed series' windows
(read as a one-block path), a **path** (a blocked series: frames x blocks per window, a walk
through block-space), or an **image** (the page-by-time tile). Labels are the keys every run
writes: family, workload, recording, plus the campaign derived from the recording id.

**Folds** (`learn/splits.py`, B1's `b1_splits.py` generalised): within-trace tail (the
memorisation ceiling, never the headline), leave-one-recording-out, leave-one-workload-out (the
headline), leave-one-campaign-out. No group straddles a split and the executor asserts it. Every
fitted step (scaler, PCA, encoder, model) fits on the training rows of the fold only. A fold with
no training rows is skipped and named; a family with no same-family training on its fold is a
novelty fold, flagged, not hidden.

**Modules** (`learn/registry.py`): every module says which shapes it reads, what it emits, what
it needs. The palette shows unavailable modules with the reason (a missing package) and designed
but unbuilt ones with what they would need (TabPFN, the time-series foundation models, USAD,
TranAD, S4 / Mamba, TS2Vec, MAE; diffusion is a locked slot, deferred to its own study). Built:
scalers, log1p, differencing, per-recording z (a soft warning: it removes level), flatten, page
pooling, PCA / kernel PCA / ICA / random projection / selection; AE (MLP), VAE, conv AE as
encoders; logistic regression, linear SVM, kNN, random forest, extra trees, gradient boosting,
MLP, FT-Transformer, MiniRocket (numpy), LSTM / GRU / RNN, 1-D CNN, TCN, InceptionTime, a patch
transformer, 2-D CNN; k-means, GMM, hierarchical (k fixed in advance); isolation forest, LOF,
one-class SVM, ECOD (numpy), kNN distance, Deep SVDD; the AE / VAE / conv-AE **banks** (one
model per family, argmin error, B1 phase 1). torch models run on the CPU by default
(`PLAN10_TORCH_DEVICE` overrides) and are seeded.

**Scores** (`learn/scores.py`), per fold and aggregated per (configuration, split): accuracy,
balanced accuracy, macro F1, kappa, MCC, per-class precision / recall / F1, the confusion matrix,
top-2, Brier, log loss, calibration (ECE, reliability bins), train accuracy and the gap; AUROC,
AUPRC and recall at a fixed FPR for novelty scores, in the fold where it holds both classes and
against the training tiles' own scores (in-sample, and labelled so) where it holds only novel
tiles; ARI and NMI against family and workload, silhouette, Davies-Bouldin, Calinski-Harabasz,
purity and the contingency table for clusterers; silhouette and kNN purity of embeddings;
permutation importance per column or per block; input-gradient saliency for torch models;
training curves. The **null** permutes the test labels against the predictions: per fold where
the fold holds several classes, and **pooled** over every fold's out-of-fold predictions, which
is the null a leave-one-out design needs (a single fold often holds one class and has none of
its own). The **bootstrap** resamples recordings. Effective n is workloads: a sweep past 12 fits
per split is a soft warning, and its size is in every result.

**Outputs** (`~/.cache/plan10/learn/<label>/`): `learn_results.json`, `predictions.npz` (one row
per scored test tile), `embeddings.npz`, `sidecar.json` (the pipeline, the input runs and the
hashes of their files, the folds per split by recording id, seeds, versions, device, the sweep
size), `status.json`, `run.log`, `control.json`. The executor is a subprocess, like the
extraction runner, so a run outlives a page reload; pause and stop go through `control.json`.

**Views.** The results are two frames, scores (one row per configuration, split, fold, seed)
and tiles (one row per scored test tile), drawn through Explore's five views. On top: the
confusion matrix, ROC curves, the embedding in 2-D (PCA on the bridge; UMAP when installed),
the null next to the observed score, calibration, importance, saliency, training curves, and a
path or image tile as drawn (a path shows the most active block per frame as a line). Every
table exports as CSV. The bridge routes are `/learn/modules`, `/learn/inputs`, `/learn/validate`,
`/learn/run`, `/learn/runs`, `/learn/status`, `/learn/control`, `/learn/results`, `/learn/agg`,
`/learn/view`, `/learn/tile`, `/learn/tiles_index`; the CLI is
`python3 -m plan10_analysis.learn.executor run PIPELINE.json --out-dir OUT --runs-root RUNS`.

Verified 2026-09-17 on the laptop: the synthetic three-shape suite (`tests/test_plan10_learn.py`,
rows, paths and images through sklearn, numpy and torch models), the HTTP suite, and in the
browser two real runs (`b1_apf_floor_bnb_tsp`, one workload, 1851 rows; `b1_real_smoke`, three
families, 27 rows) plus a synthetic path run through MiniRocket and an LSTM. No real recording
has tiles yet: a scheme with `Write tiles` against the NFS corpus is the next capture-side step.

## Encoding paper: the plan11 toolkit, launched and shown, never reimplemented

`plan11_encoding_ladder/` is the encoding paper's pre-registered, command-line toolkit and the
source of truth for the paper's numbers; its `RUNBOOK.md` is the contract. The panel
(`encoding_panel.py`, the `ep` overlay, the `/encoding/*` routes) obeys one rule: it launches
the toolkit's own commands and displays the toolkit's own files. It computes no number, it
writes nothing under `plan11_encoding_ladder/`, and it touches no server.

**Launching.** Each move's Run button starts the toolkit's driver for that one move,
`python3 -m plan11_encoding_ladder.run_moves run --out <out> --root <root> --moves N <flags>`,
from `VM_Capture_QEMU/` (the driver's cwd), with the flags the author set and nothing else;
Re-run adds `--force`. The driver's own resume and staleness rules apply unchanged (it skips a
command whose outputs exist and inputs are unchanged, re-runs what a changed input or argument
makes stale, never overwrites the author's `inputs/*`). The flag form is built from the driver's
own argparse (`run_moves._add_run_args`), so every flag, default and help line is the toolkit's;
two presets set exactly the values of RUNBOOK section 1 (the paper run) and section 0b (the
smoke run). A set flag is passed on every launch, because the driver compares arguments between
launches. The order is enforced: a move's Run is enabled only when the move before it is done
(the runbook's order, and the order the driver runs `--moves 0-14` in); the board also lists which
earlier moves each move reads files from (the producers of its declared inputs, from the driver's
plan). One toolkit process at a time; a driver started from a shell on the same `<out>` is
detected by its command line, shown as running, and the console refuses to start a second writer
to `driver_state.json`. Stop terminates the driver's process group (the toolkit has no pause);
the move it was on then reads **not run** with the time it was cut, whatever finished before the
stop (the ledger's open `runs` entry says so), the next move stays disabled, and running it again
lets the driver resume: it skips the commands that finished and runs the rest.

**The board's states** come from the ledger the driver writes (`<out>/driver_state.json`): the
last record of every command of a move, its status string as written (`done`, `skipped: ...`,
`kept: author input exists`, `failed: exit N`, a `refused:` verdict of an internal step), rolled
up to not run / running / done / partial / failed / refused. The commands a move will run are the
driver's plan for the current flags, and the exact lines `run_moves plan --moves N` prints are
shown beside them. The runbook's own "Look at" paragraph per move is shown as written.

**Every launch is recorded** under `<out>/.console/launches/<id>.json` (a directory the toolkit
neither reads nor hashes): the command line, cwd, the flags and preset, the toolkit's identity
(repository HEAD, whether the toolkit's files are tracked and clean at it, a sha256 fingerprint
over its top-level .py files, its package version), and when the run ends its exit code, the
driver's `params` block and its `runs` record from the ledger. The driver's stdout is streamed
to `<id>.log` and shown in the Log tab. As of 2026-09-17 the toolkit is untracked by git, so the
record says "not committed at HEAD" and the fingerprint is what identifies the code that ran.

**Cells.** `cells.csv` (move 0) joined for display with each cell's `extract/<cell_id>/sidecar.json`
(`n_pairs`, status, gaps, `dt_est_s`, `K_max`; move 1) and its `gates/preconditions.csv` row
(`all_hard_pass`, C1 and its rule, the failed verdict; move 2) plus `preconditions.json`'s
excluded list. Kernel and idle rows are listed; any other directory under the retention root
(the toolkit indexes the whole root with `rglob("rep*__*")`) is counted with its status and never
named, since the paper's corpus is the kernels plus the idle cells. The idle cells' real layout
(`sleep/sleep/sleep_600/rep00N__idle_01c`) is indexed by the toolkit as role `idle`, test label
`sleep`, rep = rep_dir - 1; verified on synthetic copies.

**Look at.** Each view is a list of the toolkit's files: a PNG is shown inline with its PDF
linked, a CSV as a table with every cell verbatim (refusal strings in colour, empty cells empty,
a `.params.json` sidecar linked), a JSON as its object, a Markdown table as text. A missing file
is named as not yet written, with the move that writes it. The temporal-grid view highlights the
grid point `selection.json` names per rung and shows every other point. The `browse <out>`
list opens any file under the output root. Nothing outside `<out>` is served (`safe_path`).

**For the author.** The driver's params block from the ledger, the author's input files as they
are (`inputs/*`; edit them in the shell, the toolkit never overwrites them), and every params
block the toolkit wrote under `<out>` (each JSON with a top-level `params`), read-only;
`report/manifest.json` collects them once move 12 has run.

**The EUSIPCO row.** The runbook's two commands that the driver does not schedule
(`tables_eusipco`, `latex_skeleton_eusipco`) are a final row, enabled after move 14; its state is
read from the files they write. **`--standalone` must not point at a live paper.** As of 2026-09-27
`apf_paper/p2e_skeleton.tex` and `p2_skeleton.tex` are hand-maintained; the builders emit a
scaffold with zero prose lines and the writers refuse to overwrite a target that holds prose.
`standalone_tex` in the panel should be unset, or a scratch path.

Tested on the toolkit's own synthetic corpus (`synth corpus`), never on the server: moves 0 to 5
through the panel in the browser, a shell-started driver detected on another output root, and
the suites `tests/test_plan10_encoding_panel.py` and `test_encoding_panel_endpoints_over_http`.
Moves 6 to 14 and the comparators were not driven to completion here (G-ORD alone takes tens of
minutes per rung even on a tiny corpus); their views render whatever files exist and name what
does not.

## What is not implemented, and says so

Nothing. Every module runs, and an SSH source can either fetch chains here or run the
extraction on the server.

Blocks along the address axis take any width and hop: equal to tile, smaller to overlap
(each page's row is replicated once per block it falls in, so cost scales with wp/hp), larger
to sample with gaps (pages between blocks are dropped). Whole blocks only, the same rule
Window's `edge=drop` uses in time. The block count the console shows assumes the capture
config's pages-per-dump, since no recording records its own; the runner uses each recording's
actual page count.

## Head drop

The Cells module can drop the first pairs of every recording, off by default, 32 pairs when on.
It applies once, where the store is loaded, upstream of every branch, so every reading drops the
same pairs and the tiles stay aligned for Concat; the pairs after the drop renumber from 1 and the
pair count and the shortest-recording count shrink. When on the count must be at least 1 and must
leave the Window enough pairs; several Cells modules must agree. The Cells card says "head drop:
off" or "head drop: 32 pairs", and the sidecar records the toggle, the count and what was applied
on every run. Same meaning as plan11's `inputs/head_drop.csv` `head_drop_pairs`, so the canvas and
the toolkit can be given one number.

## Ratios: the content-change family

`Ratios` is a Compose module: per changed page, one amount channel over another, or over the
page size. Its options are read from the roster's amount group, so the list follows the differ;
the default is the encoding paper's three: `l0/page` (bytes changed over the page size),
`l1/l0` and `hamming/l0` (magnitude and bit count per changed byte). Only the ratios leave the
module, named `l1_over_l0` and so on, so the lenses' per-channel suffixes read as
`mean:l1_over_l0`. The upstream Channels module must carry every channel a chosen ratio uses;
the validator refuses otherwise and notes any channel that feeds no ratio. A zero denominator
gives 0, never NaN, and the count of such rows rides on the field as `zero_denominators`.

Products and per-channel gating, the rest of the combiner family, remain the palette's dashed
"designed, unbuilt" entry.

## Persistence: overlap along time

`Persistence` is a Divide module beside Collapse: it reduces the address axis by set overlap
instead of averaging. For each pair it takes the set of pages that changed and compares it
with the set `lag` pairs later: `jaccard` (shared over either), `forward` (the fraction of this
pair's pages that change again) or `overlap` (shared over the smaller set). Only which pages
changed is read, so a complex field works and the channel values never matter. The output is
one series per recording (one per block for a blocked field), named after the measure. The
last `lag` pairs have no partner: by default the final value is repeated so the series keeps
its length and its tiles share keys with the other readings, which Concat requires; `zero`
pads with 0, `drop` shortens the series by `lag`. An empty denominator gives 0, or 1 with
`empty=one`.

## Concat, and the encoding paper's five readings

A lens's feature block now records which channel it read. `Concat features` joins blocks on
identical tile keys and refuses duplicate names, since a single-channel block names its
statistics plainly (`mean`, `std`, ...): two such branches, APF and persistence say, would
collide. Its `prefix=channel` option suffixes those plain names with the block's channel
(`mean:changed_fraction`, `mean:jaccard`), the form a multi-channel block already carries
(`mean:l1_over_l0`), so every column of a combined run says what it is a statistic of. The
validator predicts the names and refuses the collision before the run; a Write fed by several
blocks directly is checked the same way and points at Concat.

The five readings of the encoding paper are one scheme: one Cells; a Channels block carrying
`hamming` feeding APF (Collapse, changed_fraction), wAPF (Collapse, mean, unchanged = zero) and
persistence (Persistence, jaccard); a second Channels block carrying `l0`, `l1`, `hamming`
feeding Ratios then Collapse (mean, unchanged = excluded: an unchanged page has no ratio); each
branch through its own Window and Simple statistics; one Concat with `prefix=channel`; one
Write. 48 columns per tile, none duplicated. `tests/test_plan10_runner_executor.py`
`_five_readings_scheme` builds it.

## To add a module

1. `modules.py`: its ports, params, flags.
2. `runner/stages.py`: a pure function over the signal dicts.
3. `runner/executor.py` `_eval_local` (per recording) or the cross-recording branch.
4. `scheme.py` `node_constraints` and the mirror in the template's `nodeConstraints`.
5. A case in `tests/test_plan10_runner_executor.py`.

## References: two kinds, not interchangeable

`Baseline` produces one of two things and the consumers check which:

- **cell** fits a PLV phase baseline on one clean recording's complex tiles. Only `PLV` reads it.
- **benign** fits the p5-p95 band per feature over a chosen set of recordings, which is
  `plan05_campaign/normal_profile.py`'s normal operating region. Only `Deviation` reads it.

`Deviation` emits `dev_n_outside` (that file's detector: how many features fall outside the
band, NaN counting as inside), `dev_frac_outside`, and the distance past the edge normalised
by the band width, falling back to |median| then 1.0 where the benign set pins a feature to a
single value.

Choosing no benign set fits the envelope over every recording reaching the node, threats
included; that is warned about, since a normal region defined partly by what it should flag is
not one. As `normal_profile.py` says of itself, "normal" here means the chosen recordings, not
production traffic.

## The page axis

Window takes either a series (after Collapse or Block) or a field that still carries the page
axis. The second gives **page-resolution tiles**: the page-by-time image the methodology
describes, materialised dense as (frames x pages) per tile.

`page_mode=active` keeps only the pages that change in that recording, which is what the
differ's sparse output is for: on the synthetic fixture that is 47 columns against 1024 for
the same 77 non-zero cells. `page_mode=all` keeps every page, and a memory budget refuses the
combination before it runs, naming the size it would need.

Downstream, every lens runs per page and reports the median across pages, named `<feature>_median`
-- the convention `StabilityValidator` already uses for `msc_peak_snr_db_median` and
`cepstral_peak_idx_median`. PLV and Baseline are not wrapped: `PLVStability` takes `[T, N]` and
aggregates the page axis itself, so on a complex page-resolution tile it fits one PLV per page,
which is what that code was written for.

A page-resolution tile carries one channel or a complex field; several real channels would make
it four-dimensional and no lens here reads that.

### Measured on the real corpus (2026-09-09)

Page-resolution tiles verified on `~/thesis_traces/zstd_local`, three workloads at 40 pairs:

| recording | active pages | of 262144 | dense at 40 pairs | at 700 pairs |
|---|---:|---:|---:|---:|
| mem_pagefault_density_v2 | 61,193 | 23.3% | 10 MB | 171 MB |
| io_direct_write_like_v2 | 35,814 | 13.7% | 6 MB | 100 MB |
| cpu_branch_random_v2 | 10,208 | 3.9% | 2 MB | 29 MB |

`page_mode=all` is 42 MB at 40 pairs and 734 MB at 700, which the 256 MB default budget
refuses, naming the figure. `active` fits at both.

Cost: each lens runs per page, so a full 700-pair recording is about 7 minutes per lens for
the busiest workload, against roughly 40 minutes to extract it. Extraction dominates a first
run; the lens loop dominates when re-running schemes against a cached L1 store.

The `mean_median` statistic separates the workloads on real data: 177 for
mem_pagefault_density against 0.07 to 2.4 for cpu_branch_random and 2.24 for io_direct_write.

PLV on a real complex page-resolution tile fits one PLV per page over 10,005 pages, which is
what `PLVStability` was written for and what no collapsed tile can give it.

### Verified against the real server (2026-09-09)

`jeries@cybersecurity.ac.upc.edu` (`pcrserral`), repo at
`/project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU`, traces at
`/project/homes/jeries/memory_traces/zstd_local`.

- **probe**: python 3.13.14, numpy 2.4.6, zstd, and the server's own differ binary.
- **listing**: the remote `find` produced the manifest in one round trip: 1 recording,
  470 pairs, 5.9 GB.
- **remote extraction**: 6 pairs of a real 1 GiB dump in 22 s. What came back was a
  **164 KB npz** holding 29,583 rows over 8,379 active pages; no `.zst` crossed the network.
- **full scheme**: validated, reused the server's L1 store on the second run, and wrote
  features (APF 0.0201 and 0.0162 per window) with a sidecar naming the host and mode.

The server's `plan10_analysis` is a copy pushed with rsync, not a git checkout; the branch has
not been pushed there. Re-sync it after changing the runner, or a remote run executes the old
code. **Fetch mode ran against the real server on 2026-09-11**: two `kernel_bnb_tsp_v2` recordings
(931 and 933 pairs) through `b1_apf_floor`, 463 rows x 8 features in 198 s, both read from their
trajectories; the second pulled its 804 MB CSV and no snapshot at all.
