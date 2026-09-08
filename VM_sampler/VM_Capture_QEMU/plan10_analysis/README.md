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
```

The launcher scans the source, rebuilds the served console against it, starts the bridge on
`127.0.0.1:8766` and opens the browser at the tokenised URL it prints. Ctrl-C stops the
bridge; a launched run is its own process and keeps going.

Needs: `python3` (3.10+), `numpy`, `PyWavelets`, `kymatio` (see `requirements.txt`), the `zstd` CLI, the differ binary
(`cd VM_sampler/VM_Capture/live_delta_calc_modular && cargo build --release`, or set
`PLAN10_DIFFER`), and for an SSH source `ssh` and `rsync`.

## Use it

1. **Source** tab (bottom drawer): local path, or host / user / key / remote root for SSH.
   Test, then Scan. The Cells module now selects from what was found.
2. Drop modules from the palette, pipe output ports to input ports. Ports are typed. Load an
   example from the header to start from a working graph.
3. Fix what is red (hard), acknowledge what is amber (soft) in the inspector with a note.
   Save scheme writes the JSON; Launch runs it.
4. **Run** tab: differ speed (default the config's), max pairs (0 = all), progress, log,
   pause, stop. **Results** tab: every run, its rows, its sidecar.

Runs land under `~/.cache/plan10/runs/<label>/`: `features.npz` (`X`, `feature_names`,
`tile_keys`), `features.csv`, `sidecar.json`, `status.json`, `run.log`. Extracted channels
are cached under `~/.cache/plan10/l1/` per (recording, speed, channel set) and reused.

## The pieces, each separable

| File | Does | Test |
|---|---|---|
| `channel_roster.py` | the 64 columns and the speed each dies at, parsed from the differ's Rust and reconciled with its HELP table | `test_plan10_channel_roster.py` |
| `corpus_manifest.py` | recordings from a listing (`scan_listing`); local walk and SSH `find` produce the same manifest | `test_plan10_corpus_manifest.py` |
| `sources.py` | `LocalSource` / `SshSource`: listing, fetch to cache, test | `test_plan10_sources.py` |
| `known_issues.py` | the flag registry, every number recomputed from the artifact it cites | `test_plan10_build.py` |
| `modules.py` | module and port registry; feature lists read from the implementing code | `test_plan10_scheme.py` |
| `scheme.py` | the scheme format, validator (hard / soft / note), estimate, examples | `test_plan10_scheme.py` |
| `runner/chain.py` | walk a zstd patch chain with a two-file rolling window | `test_plan10_runner_extract.py` |
| `runner/differ.py` | run the differ on a pair, parse its sparse CSV; refuses dumps not a multiple of 4 MiB | same |
| `runner/extract.py` | the L1 store | same |
| `runner/stages.py` | one pure function per module kind; reuses b1_features, CepstrumStability, PLVStability, plan04_cusum, normal_profile | `test_plan10_runner_executor.py` |
| `runner/executor.py` | order, run per recording, status, control, output, sidecar | same |
| `ui/analysis_bridge.py` | the local HTTP backend | `test_plan10_bridge.py` |
| `ui/build_analysis_console.py` | injects everything above into the template; static build has no network code | `test_plan10_build.py` |
| `ui/analysis_console.template.html` | the console; the bridge client sits between SERVED_ONLY markers | loaded in a browser |
| `testing/synth.py` | a 4 MiB-dump synthetic corpus with known answers | used by the runner tests |

Tests are plain asserts (`python3 tests/test_plan10_*.py`) or pytest.

## What is not implemented, and says so

Tiles at full page resolution (put Collapse or Block before Window) and remote execution
(SSH fetches to a local cache; the runner is local). Each refuses
with a message in the run log and, where the console can see it, as a hard constraint.

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
