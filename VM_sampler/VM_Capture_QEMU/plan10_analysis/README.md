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
   selects from it. Scan is the slow reconciliation (a full walk that rewrites that manifest);
   it is for an archive nobody registered into, not for every session.
2. Drop modules from the palette, pipe output ports to input ports. Ports are typed. Load an
   example from the header to start from a working graph.
3. Fix what is red (hard), acknowledge what is amber (soft) in the inspector with a note.
   Save scheme writes the JSON; Launch runs it.
4. **Run** tab: differ speed (default the config's), max pairs (0 = all), progress, log,
   pause, stop. **Results** tab: every run, its rows, its sidecar. **Monitor** (header) is the
   full-screen view of the same run.
5. **Explore** (header, or the Results tab's button): plots and tables of what a run wrote,
   and runs side by side. See "Explore" below.

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
names the missing columns. The Channels module shows the same fact per channel: a green ring is in
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

## What is not implemented, and says so

Nothing. Every module runs, and an SSH source can either fetch chains here or run the
extraction on the server.

Blocks along the address axis take any width and hop: equal to tile, smaller to overlap
(each page's row is replicated once per block it falls in, so cost scales with wp/hp), larger
to sample with gaps (pages between blocks are dropped). Whole blocks only, the same rule
Window's `edge=drop` uses in time. The block count the console shows assumes the capture
config's pages-per-dump, since no recording records its own; the runner uses each recording's
actual page count.

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
