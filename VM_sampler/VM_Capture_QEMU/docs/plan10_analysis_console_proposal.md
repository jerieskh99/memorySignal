# Plan 10 · The Analysis Console

**Replace the hand-run analysis with a composable, launchable, gated analysis scheme —
the same arc the capture side already has, applied to everything below the seam.**

Status: **proposal. Nothing built.** Drafted 2026-09-07.

> **Amended 2026-09-07 by `plan10_analysis_console_UX.md`, on three points.** (1) Execution is
> **laptop-local**, not server-side over ssh: the traces are assumed migrated to this machine, an
> assumption the console states in its own source panel. (2) The span is **AGGREGATE + feature
> extraction only** — compose, divide, lens. ORGANISE and MODEL leave this tool, as `b1_ae.py` is
> already separate from `b1_features.py`. (3) The corpus location is an assumption held open, not an
> open question blocking the design. Sections 3.1, 4.7, 4.8 and 4.9 below are superseded where they
> conflict; the disputes in §5 stand except Dispute 1's ssh assumption and Dispute 5, which concerns
> a comparison stage this tool no longer contains.

Evidence convention, inherited: `[traced]` read from the named file this session. `[computed]`
derived here. `[inferred]` assembled, source does not say so. `[proposed]` designed, not built.

Authorship: drafted by an assistant from a direction JK set ("build the analysis UI, same as the
capture UI; the analysis part starts at the aggregation and runs to the ML part"). Sections marked
`[derived]` are mechanisms the assistant worked out from that direction, not positions JK has taken.

---

## 1 · Executive summary

The capture side can be composed, launched, watched, and trusted. The analysis side is run by hand:
B1 was five scripts invoked in sequence over an ad-hoc `sources_list`, with the corpus manifest
never written and the results pulled to a laptop as an `npz` `[traced: b1_extract_all.sh,
plan08_b1/*.py, ~/b1_data/]`. That is the exact condition the capture side was in before Plan 07.

This plan proposes the analysis counterpart of `plan07_campaign/ui/`: a single-file vanilla console,
a stdlib bridge on the server, and a composer that turns a chosen **path through the Entry 13 graph**
into real commands over a real corpus — with analysis-side gates that refuse rather than report.

The unit the console composes is an **analysis scheme**: one traversal of

```
AGGREGATE  ->  COMPOSE  ->  ANALYSE  ->  ORGANISE  ->  MODEL
```

exactly as a capture scheme is one traversal of workload x duration x scale x rep. The research
question Entry 13 states — *which path through the graph separates workloads* — becomes an operation
in the UI rather than a paragraph in a diary.

Three things this plan asserts up front, because they set its scope:

1. **The console executes nothing.** It composes schemes, launches a runner, and reads status files.
   Analysis logic lives in `plan10_analysis/` modules, exactly as capture logic lives in
   `subset_run.py` -> `run_files_controlled.py` and never in `console_bridge.py` `[traced]`.
2. **The vocabulary is the whole graph; the executable set is what exists.** Unbuilt nodes appear as
   the figure draws them — visible, labelled unbuilt, unselectable. Hiding them would reproduce the
   drift the seam label already names: *"metric selection — resolved in the run config, wired to
   nothing in capture"* `[traced: part_graph.html]`.
3. **The corpus manifest is step zero and blocks everything.** B1 named it blocking and it was never
   written `[traced: EXPERIMENT_B1_ENCODING_FLOOR.md build order 0]`. The console's corpus panel is
   how it finally gets written, because a panel that cannot list the corpus cannot be shipped.

---

## 2 · Baseline · what exists, file by file

All `[traced]` this session.

### 2.1 The capture console, as the worked example

| File | Lines | What it is |
|---|---:|---|
| `plan07_campaign/ui/capture_console.template.html` | 1110 | hand-edited source; single-file vanilla HTML/CSS/JS, no framework, no build deps |
| `plan07_campaign/ui/build_console.py` | 143 | injects live pipeline data at `/*@@GENERATED_DATA@@*/`; two builds |
| `plan07_campaign/ui/console_bridge.py` | 1080 | stdlib `ThreadingHTTPServer`, 127.0.0.1 + token, polling |
| `plan07_campaign/ui/console.sh` | 96 | laptop launcher: ssh, remote build, bridge, `-L` forward, browser, migration agent |
| `plan07_campaign/ui/migrate_agent.sh` | 150 | laptop-side pull of completed chains |

Properties worth copying, each one load-bearing:

- **The UI cannot drift.** `build_console.py` imports `SCALABLE`, `CLAMP_MB`, `binary_of` from
  `generate_database_steps.py` and `FEATURES_BY_GROUP`, `SUBMODULE_COLUMNS` from `subset_run.py`,
  and derives `CAPPED` rather than hardcoding it. The template hardcodes nothing about workloads,
  scaling, or the metric taxonomy.
- **Two builds from one template.** `SERVED_ONLY` markers bracket the network block. The default
  build strips it, giving a console with zero network code (the publishable artifact); `--served`
  keeps it for the bridge.
- **The token never touches a command line.** The bridge generates it on the server and it returns
  only over the ssh stream, so `ps` on a shared host cannot see it. `console.sh` parses it from the
  banner and opens `localhost:LPORT/?token=...`.
- **State lives in files, not memory.** `capture_status.json` (producer writes), `capture_control.json`
  (UI writes via tmp + `replace`), append-only `.migration/ledger.jsonl`, saved configs under
  `configs/`. A UI or bridge restart loses nothing.
- **The run refuses to collide.** `subset_run.py` exits if `runs/<label>.json` already exists.

Routes today: POST `/plan` `/preflight` `/launch` `/stop` `/save` (+ delete-saved, clean-host,
clean-guest, health, control, migration-*); GET `/status` `/log` `/saved` `/run_status`
`/capture_status`.

### 2.2 The analysis code that already exists

**Plan 08 / B1 — the only end-to-end analysis chain in the tree** `[traced: plan08_b1/]`:

| Script | Interface, verbatim |
|---|---|
| `b1_extract_hamming.py` | `src out_dir [--n-pages 262144] [--page-size 4096] [--no-field] [--cell-id] [--workload]` |
| `b1_extract_all.sh` | `<sources_list> <out_dir> [n_pages]`, resumable on the `"final"` sentinel |
| `b1_windows.py` | `root [--window 8] [--hop 4] [--min-pairs 50] [--out]` |
| `b1_features.py` | `root [--min-pairs 50] [--out]` |
| `b1_splits.py` | `npz [--test-frac 0.2]` |
| `b1_ae.py` | `npz [--seed 0] [--bottleneck 3] [--min-train 8] [--max-iter 300] [--test-frac 0.2] [--out]` |

That chain **is** one path through the graph: AGGREGATE(W=8,H=4, page axis collapsed) ->
COMPOSE(none) -> ANALYSE(none) -> ORGANISE(raw window) -> MODEL(AE bank). It is the first scheme the
console must be able to reproduce, byte for byte, or the console is wrong.

**First-generation lens code, off the live path** `[traced]`: `VMsig_featureExctraction/`
(`block_feature_extractor.py`, `spectral_analysis_features*.py`, `wavelet_analysis_features*.py`,
`multi_batch_size_wavelet2d_analysis.py`) and `coherence_temp_spec_stability/`
(`cepstrum_stability.py`, `magnitude_squared_coherence.py`, `plv_calcolator.py`,
`streaming_metrics.py`). These populate the ANALYSE tier of the figure. They are **not** wired to the
campaign corpus and `block_feature_extractor.py` carries the phase collision at ~30 inline sites.
Adopting them is a port, not an import, and the port is out of scope for release 1 (§7, Dispute 3).

### 2.3 The constraints any design here inherits

| Constraint | Value | Source |
|---|---|---|
| Live channels | **51 of 64**; 13 emit zeros into an unchanged schema | `config_qemu_upc.json:11` `substrateSpeed: 2`; drops counted in `family_a/informational.rs:35,40,62`, `family_b/structure.rs:31`, `family_b/spatial_shift.rs:13`, `family_c.rs:88-89`, `family_d/texture.rs:19,44` `[traced, counted this session]` |
| Dead channel names | `ncd`, `lz_change`, `kendall`, `cross_corr_lag`, `phase_corr`, `byte_rotation`, `bigram_ent`, `autocorr_peak`, `glcm_{contrast,homogeneity,energy,correlation}`, `high_freq_frac` | same |
| Differ cost | ~39 s/pair @speed 0, 5.5 @1, **2.1 @2**, 1.5 @3, 1.2 @4 (1 GB pair, ~5% activity, self-labelled laptop estimate) | `live_delta_calc_modular/src/main.rs:40-48` |
| Guest physical layout | **SCATTERED**: `frac_in_longest_run` 0.087-0.162 across 3 snapshots, 2 runs | `docs/dwarf_pilot_design/GATE2_RESULT.md` |
| Window default | W=8 H=4, APF-derived, smallest-W tiebreak, 128 never swept | `plan05_campaign/downstream/recommendation.json`, `plan03_sweep.py:275` |
| Validator window | counts at W=128 H=64, `MIN_WINDOWS_DEFAULT=50` | `plan02_validate_session.py:57-62` `[traced]` |
| B1 floor | APF lowo acc 0.3423 (app 0.841, mem 0.753, cpu 0.024, io 0.018, thread 0.010); wAPF lowo 0.2536; majority 0.234 | `~/b1_data/b1_ae_results.json` `[traced, read this session]` |

**Gate 2 lands directly on this design.** Address blocking is a first-class control in the AGGREGATE
stage, and Gate 2 measured that a contiguous guest-virtual array arrives in guest-physical RAM as
~3000 fragments. So a block is not a region of the program's data; it is a region of the allocator's
placement. The control stays — the axis is real and the methodology's §8 anticipated exactly this —
but it ships with the measured warning attached and with gate A5 (translation invariance across
repetitions) able to refuse.

---

## 3 · Top-down · what a scheme is

### 3.1 The five stages, and their vocabularies

Node names are the figure's, verbatim `[traced: part_graph.html]`. Availability is this session's
verification.

**Stage 0 · CORPUS** (not a graph tier; the analysis analogue of "workload selection")
: which cells, by family / workload / duration / rep, and which retention each one has (zstd chain,
substrate CSV, APF trajectory only). Availability is a property of the server tree, discovered by
the manifest scan, never typed in.

**Stage 1 · AGGREGATE** — *the layer the figure does not draw*
: temporal: window `W`, hop `H`. Spatial: page-axis treatment — collapse (what every result to date
does), fixed blocks of `B` pages, or full resolution. Whole-page constraint: `B_bytes mod 4096 == 0`
`[traced: methodology §5]`. Carries the Gate 2 warning on any non-collapse choice.

**Stage 2 · COMPOSE**
: `none` (keep channels separate — the ablation Entry 2 records as never run) · `complex`
(magnitude x direction) with an explicit **phase convention** field: `2pi` (the collision), `pi`
(half turn), `arccos` (true angle). The convention is recorded in run metadata because it cannot be
reconstructed from the output afterwards.

**Stage 3 · ANALYSE**
: `none` · spectral: `FFT`, `Cepstrum` · time-frequency: `Wavelet`, `Scattering` · synchrony:
`PLV`, `MSC` · change-point: `CUSUM`.

**Stage 4 · ORGANISE**
: `flatten` · `StandardScaler` · `PCA(k)`.

**Stage 5 · MODEL**
: `AE bank` · `OCSVM` · `novelty` · `RF` · `LSTM`, plus the **protocol**: `within_trace`, `LORO`,
`LOWO`, `LOFO`, and the split unit.

Channel selection sits between CORPUS and AGGREGATE and is the seam the capture console stops at.
It offers the 51 live channels grouped by the figure's families (A amount 20, B direction 19,
C content 13, D texture 12) and shows the 13 dead ones **disabled, with the reason string
`zero at substrateSpeed=2`** — see Dispute 4.

### 3.2 What one scheme produces

```
scheme  ->  N runs  ->  one results row per run  ->  one comparison table
```

A scheme with a list-valued stage expands by cross product, the same way `build_cells` crosses
`workload_family x duration_s x scales x reps` `[traced: subset_run.py:358-418]`. The expansion is
shown as a checkbox list with per-run enable/disable and reordering, because the cross explodes far
faster here than on the capture side (Dispute 2).

### 3.3 Scope boundary

**In:** compose, expand, curate, preflight, launch, monitor, gate, retain, compare. Server-side
execution under `screen`, survivable across laptop disconnect, mirroring `SCREEN_PREFIX`.

**Out:** new metrics (capture side owns METRICS), new capture, the phase fix itself (a ~30-site
change in `block_feature_extractor.py`, its own plan), and any claim about *where in memory*
anything happened while the page axis is collapsed.

### 3.4 The form, separated from this realisation

What any valid analysis console must preserve, stated so a different implementation could satisfy it:

1. A scheme is a **path through the graph**, declared before execution, serialisable, and replayable.
2. Every stage's vocabulary is **read from the code that implements it**, never typed into the UI.
3. Every run carries the **provenance of every choice** that cannot be recovered from its output —
   phase convention first among them.
4. A gate can **refuse to write**, not merely report.
5. Comparison between paths is a **first-class artifact**, not a spreadsheet a human assembles.
6. The instrument's own blind spots are **visible in the composer** (dead channels, scattered
   physical layout, inherited window), so the shortcut is refused where the hand reaches.

---

## 4 · Bottom-up · file by file

New tree, mirroring `plan07_campaign/` `[proposed]`:

```
plan10_analysis/
  scheme_run.py            composer: scheme JSON -> curated run list -> launch lines
  channel_roster.py        the 51/64 truth, derived not typed
  corpus_manifest.py       walk the server retention tree -> cells + what each one has
  stages/
    aggregate.py           W/H windowing + page-axis blocking
    compose.py             none | complex(convention)
    analyse.py             none | fft | cepstrum | wavelet | scattering | plv | msc | cusum
    organise.py            flatten | standardise | pca
    model.py               ae_bank | ocsvm | novelty | rf | lstm  + protocols
  gates.py                 A1-A8, each able to refuse
  analysis_run.py          the orchestrator: one scheme, N runs, status file, control file
  ui/
    analysis_console.template.html
    build_analysis_console.py
    analysis_bridge.py
    analysis.sh
```

### 4.1 `channel_roster.py`

Emits the roster the UI shows and the runner validates against:

```json
{"schema": "plan10.channel_roster.v1",
 "substrate_speed": 2,
 "channels": [{"name": "hamming", "family": "A", "type": "positional", "live": true},
              {"name": "ncd", "family": "A", "type": "informational", "live": false,
               "reason": "zero at substrateSpeed=2"}],
 "n_total": 64, "n_live": 51,
 "derivation": "measured" }
```

Names come from `csv_header()` in `metrics/mod.rs`; liveness is **measured, not parsed** (Dispute 4).

### 4.2 `corpus_manifest.py`

Walks the retention tree (the chain-leaf pattern already exists at `console_bridge.py:649
_chain_leaves`, `_chain_stats`) and emits one row per cell:

```json
{"schema": "plan10.corpus_manifest.v1", "scanned_at": "...", "root": "...",
 "cells": [{"cell_id": "...", "workload": "...", "family": "mem", "duration_s": 600, "rep": 1,
            "has_zstd_chain": true, "has_substrate_csv": false, "has_apf_trajectory": true,
            "n_pairs": 398, "bytes": 12345678, "speed": 2}]}
```

This is B1's blocking step zero, finally written, and it is the panel's data source.

### 4.3 `scheme_run.py` — the `subset_run.py` analogue

Mirrors its contract exactly: validate the label against `^[A-Za-z0-9_-]+$`, refuse when
`runs/<label>.json` exists, seed derived reproducibly (`zlib.crc32`, never salted `hash()`),
`--curation` applied after expansion, `--print-cells` for a dry run `[traced: subset_run.py]`.

Scheme object:

```json
{"schema": "plan10.scheme.v1",
 "label": "e2_split_channels",
 "corpus": {"families": ["mem","cpu","io","thread","app"], "min_pairs": 50,
            "require": ["substrate_csv"]},
 "channels": ["hamming"],
 "aggregate": {"window": [8], "hop": [4], "page_axis": "collapse", "block_pages": null},
 "compose":  {"mode": "none", "phase_convention": null},
 "analyse":  ["none"],
 "organise": ["flatten"],
 "model":    {"kind": "ae_bank", "protocols": ["within_trace","loro","lowo"],
              "seed": 0, "params": {"bottleneck": 3, "min_train": 8, "max_iter": 300}},
 "baseline": "b1_apf_floor",
 "gates": ["A1","A2","A3","A4","A5","A6","A7","A8"]}
```

Expansion, curation, and the emitted per-run command lines are the only thing this file does. It
writes `runs/<label>.json` with the git SHA, the resolved roster, and the full expanded run list.

### 4.4 `analysis_run.py` — the orchestrator

One scheme in, N runs out, resumable on a per-run sentinel (the `"final"` pattern
`b1_extract_all.sh` already uses). Writes, atomically, after every run:

```json
{"schema": "plan10.analysis_status.v1", "label": "...", "state": "running|paused|done|failed",
 "run_index": 7, "n_runs": 24, "current": {"stage": "MODEL", "run_id": "..."},
 "gates": {"A3": {"pass": true, "why": "...", "operational": true}},
 "started_at": "...", "updated_at": "..."}
```

and reads `analysis_control.json` (`{"command": "run|pause|skip"}`), written by the bridge via
tmp + `replace`, mirroring `ep_control` `[traced: console_bridge.py:933-948]`.

### 4.5 `gates.py`

A1-A7 as proposed in `ANALYSIS_PIPELINE_METHODOLOGY.md` §11.5, plus A8 which this plan adds:

| Gate | Claim | Stage it blocks | On failure |
|---|---|---|---|
| A1 | no extracted feature is constant within a class | ORGANISE | quarantine feature, name it, re-run without it |
| A2 | no feature's NaN pattern encodes the label | ORGANISE | refuse to write |
| A3 | the split never puts two windows of one trace on both sides | MODEL | refuse to run |
| A4 | the score beats its own label-shuffled surrogate | MODEL | write flagged `near_unfalsifiable`, excluded from tables |
| A5 | per-tile features stable across the 3 repetitions | AGGREGATE (non-collapse only) | refuse; report as the Gate-2 consequence |
| A6 | the phase mapping is injective on its domain | COMPOSE | refuse to run |
| A7 | enough independent windows for the estimator, per channel | AGGREGATE | refuse to run |
| **A8** | **no selected channel is identically zero over the selected corpus** | CHANNEL SELECT | refuse to launch, naming the channels |

A8 exists because the 64-column schema lies about the data, and a gate that catches it at
*selection* time costs one pass over a sample and saves a whole run built on zeros.

### 4.6 `ui/build_analysis_console.py`

Same no-drift contract, same marker, same two builds. Injects:

```
CHANNEL_ROSTER   <- channel_roster.py            (51 live / 13 dead + reasons)
STAGE_VOCAB      <- stages/*.py registries       (which nodes exist, which are unbuilt)
GATE_ROSTER      <- gates.py                     (id, claim, blocking stage)
PROTOCOLS        <- stages/model.py
BASELINES        <- results index                (the B1 floor row)
```

If a stage module gains a node, the UI gains it on the next build. If it does not, the UI cannot
offer it.

### 4.7 `ui/analysis_bridge.py`

Stdlib only, `ThreadingHTTPServer`, 127.0.0.1 + token, polling. Routes:

| Route | Purpose |
|---|---|
| POST `/scheme` | validate + expand a scheme, return the run list (dry run, writes nothing) |
| POST `/preflight` | corpus reachable, roster matches, disk headroom, A8 sample check |
| POST `/launch` | start `analysis_run.py` under `screen` (`SCREEN_PREFIX = "mem_analysis_"`) |
| POST `/stop`, `/control` | stop; pause / continue / skip the in-flight run |
| POST `/save`, GET `/saved`, POST `/delete_saved` | schemes under `plan10_analysis/schemes/` |
| GET `/corpus` | the manifest (cached, `?rescan=1` to walk again) |
| GET `/status`, `/log`, `/run_status` | bridge health, log tail, run progress |
| GET `/analysis_status` | the status file, verbatim |
| GET `/results`, `/compare` | result rows; the comparison table across schemes |

### 4.8 `ui/analysis.sh`

`console.sh` with two constants changed (default `LPORT 8766`, `RPORT 8001`, so both consoles can
run at once) and the migration agent dropped — analysis outputs are small enough to fetch on demand.
Everything else identical, including freeing the port by port rather than by process name.

### 4.9 Panels, in order

1. **Scheme identity** — label, seed, notes.
2. **Corpus** — manifest table, filter by family/workload/duration/rep/retention, counts live.
3. **Channels** — the four families, 51 selectable, 13 disabled with reason.
4. **Aggregate** — W, H (multi-valued), page-axis mode, block size, with the Gate 2 warning inline
   on any non-collapse choice, and the live window-count readout per selected cell.
5. **Compose** — none / complex + phase convention, with the collision stated at the control.
6. **Analyse** — the four groups; unbuilt nodes shown greyed as the figure draws them.
7. **Organise** — flatten / standardise / PCA(k).
8. **Model** — learner, params, protocols, split unit.
9. **Gates** — A1-A8, each with its claim in plain language; which are blocking is fixed, not
   a checkbox.
10. **Generated runs** — the expansion, per-run checkboxes, reorder, count and estimated cost.
11. **Preflight / Launch / Live** — verdict list, launch, log tail, pause/continue/skip.
12. **Results and comparison** — every finished run against the pinned baseline, per-family recall
    shown always, never a single pooled score.

---

## 5 · Key disputes and their resolutions

### Dispute 1 · Extend `console_bridge.py`, or a second bridge?

**For extending:** one process, one token, one tunnel; the migration, health, and screen helpers are
already written and tested.
**Against:** `console_bridge.py` is 1080 lines whose every path assumes a capture (`SUBSET_RUN`,
`CAPTURE_CONFIG`, `VM_DOMAIN`, the guest scratch, the queue dir). Analysis has no VM, no guest, no
queue. Folding it in doubles the surface of the file that must keep working while a campaign runs.

**Resolution: a second bridge, sharing the pattern, not the process.** Different port, its own token,
its own screen prefix. The capture console must stay untouched and runnable during an analysis run —
that is a hard requirement, since analysis will be run against a corpus while more capture happens.
Shared helpers (`_valid_label`, `_run`, `_write_tmp`, `_screens`, the token/serve scaffolding) move
to `plan07_campaign/ui/_bridge_common.py` and are imported by both, with the capture console's
behaviour pinned by its existing tests.

### Dispute 2 · Full cross product, or curated paths?

The capture cross is bounded (workloads x durations x scales x reps). The analysis cross is not:
51 channels x window grid x compose x 8 analyse nodes x 3 organise x 5 models x 4 protocols reaches
five figures of runs before anyone has decided anything.

**Resolution: the composer expands the cross, and the run list is curated by default with nothing
selected above a configured cap.** A scheme that expands past `max_runs` (default 64) cannot be
launched until runs are deselected or a stage is pinned. The project's own rule decides this: with
19 workloads the effective n is 19, and a leaderboard over thousands of paths is a garden of forking
paths, not a comparison. The console must make the sweep *hard to do accidentally* and easy to do
deliberately, one stage at a time — "change one role at a time, hold the rest".

### Dispute 3 · Where does the expensive pass live?

Reconstruct-plus-differ dominates every cost in this system (2.1 s/pair at speed 2; hundreds of pairs
per trace). Two candidates: recompute per run, or cache a per-(cell, channels, aggregate) intermediate.

**Resolution: a two-level store, explicit in the UI.** L1 = per-cell extracted channel trajectories
(what `b1_extract_hamming.py` already writes, keyed by cell and channel set); L2 = per-run features.
A scheme that changes only stages 2-5 re-uses L1 and costs seconds; a scheme that changes channels or
aggregation invalidates L1 and says so in the cost readout **before** launch. This is the methodology
document's own architecture (one extraction pass, small feature tensors, delete the giants) with the
correction it already carries: never regenerate per epoch.

Corollary: the ANALYSE tier ships with `none`, `fft`, `cepstrum` in release 1. `wavelet`,
`scattering`, `plv`, `msc` are visible-but-unbuilt until the first-generation code is ported and its
phase convention resolved, because importing `block_feature_extractor.py` as-is would import the
collision at ~30 sites.

### Dispute 4 · How is the live-channel roster derived?

**(a) Parse the Rust.** Read `csv_header()` for names and the `if speed >= N` sites for drops.
Exact today, brittle: a refactor that moves a gate silently produces a wrong roster.
**(b) Pin a Python table.** Simple, and precisely the drift the build-injection rule exists to prevent.
**(c) Measure it.** Run the differ at the configured speed over one random-content page pair, where
every channel should be non-zero, and mark identically-zero columns dead.

**Resolution: (c), with (a) as a cross-check in a test.** Measurement cannot go stale, and it matches
the project's habit of turning the instrument on itself. The test asserts measured == parsed and
fails loudly when they disagree, which is the event worth being told about. The derivation method is
recorded in the roster artifact so a reader knows which one produced it.

### Dispute 5 · Comparison: leaderboard, or fixed baseline?

**Resolution: fixed baseline with a measured margin.** Every scheme is reported against the pinned
B1 APF floor (lowo 0.3423, per-family recall attached), and a difference is called only when it
exceeds a margin measured from the spread of the same score across seeds and folds on the baseline
arm alone. A bar not anchored to measured noise is decoration, and this project has already retired
a gate for exactly that. No pooled score across families is ever the headline; per-family recall is
always shown, because the whole B1 finding lives in the per-family split.

### Dispute 6 · Does the spatial axis ship in release 1?

Gate 2 measured guest-physical scatter, which weakens address blocking as a *program-structure*
axis. Arguments to defer: the strongest analysis result to date is APF (page axis collapsed), and
A5 will likely refuse most block schemes.

**Resolution: ship the control, gated by A5, defaulted to collapse.** The spatial axis is the new
PhD development (O8) and the reason this stage exists; deferring the control would leave the console
unable to express the objective it was built for. But it defaults to collapse, carries the Gate 2
number inline, and A5 blocks rather than warns. If A5 refuses everywhere, that is a measured result
about the substrate, produced by the console rather than argued in a document.

### Dispute 7 · Which gates block, and can the UI relax one?

**Resolution: blocking is a property of the gate, fixed in `gates.py`, not a UI checkbox.** The
capture side's own history decides this: thresholds moved after seeing data (C2 0.85 -> 0.15 -> 0.08),
one gate shipped hardcoded `True` while reported as passing, and C3 sits outside the conjunction.
Every one of those is defensible in its own file and indefensible as a toggle in a UI. A gate can be
changed by editing the module and recording the change as a fitted parameter, with a date and a
reason — which is a diff, in git, not a click.

### Dispute 8 · Label collision and resume

**Resolution: mirror `subset_run.py` exactly.** `runs/<label>.json` existing is a hard exit. Resume is
a separate, explicit action against an existing label, and it resumes only runs lacking the final
sentinel. Silent overwrite of a result set is the one failure this project cannot audit its way out
of.

---

## 6 · Acceptance gates for the console itself

The console is an artifact and gets its own claims, checked before it is called done `[proposed]`:

| Gate | Claim | Test |
|---|---|---|
| **K1 no-drift** | the UI offers exactly what the code implements | remove a node from a stage module, rebuild, assert it vanishes from the built HTML |
| **K2 reproduces B1** | the console can express the executed B1 chain and reproduce its numbers | scheme -> run -> assert per-family recall equals `b1_ae_results.json` within float tolerance |
| **K3 gates refuse** | each of A1-A8 blocks on a constructed violation | one synthetic fixture per gate; assert no result file is written |
| **K4 survives restart** | live state is reconstructed from files | kill the bridge mid-run, restart, assert the UI shows the same progress |
| **K5 no capture regression** | the capture console is unaffected | its existing tests pass unchanged after the shared-helper extraction |
| **K6 offline build** | the static build contains zero network code | assert the stripped build has no `fetch(`/`XMLHttpRequest` |

Decision rule: **release 1 ships only when K1-K6 all pass.** K2 is the one that matters most — a
console that cannot reproduce the one result the project already has is not an instrument, it is a
form.

---

## 7 · Validation strategy

Tests live beside the existing ones (`tests/test_plan08_b1_*.py` is the pattern `[traced]`):

- `test_plan10_scheme.py` — expansion, curation, label validation, collision refusal, seed
  reproducibility across processes.
- `test_plan10_roster.py` — measured roster equals parsed roster; 51 live, 13 dead, names exact.
- `test_plan10_gates.py` — one violating fixture per gate; each refuses.
- `test_plan10_manifest.py` — manifest over a synthetic tree, including a cell with only an APF
  trajectory (which must be marked unusable for anything needing per-page data).
- `test_plan10_build.py` — K1 and K6 mechanically.

Plus the parity check B1 already named as the cheapest correctness test available: the console's
aggregation of APF must agree **exactly** with `plan02_apf_helper`'s column, or the pass is wrong.

---

## 8 · Statistical design

Unchanged from the record and restated because the UI must enforce it:

- Effective n is **workloads** (19 in the B1 corpus), not windows (8971). The console displays both,
  with the window count subordinate, so no one reads 8971 as evidence count.
- Overlapping windows share half their samples; grouping is enforced by A3, not by discipline.
- LORO is reported and never headline. LOWO and LOFO are the headline.
- Per-family recall always; no pooled score as headline.
- Families with one workload (sandbox) or thin data (cache, 150 windows) are labelled structurally
  inapplicable for LOWO rather than reported as zeros `[traced: EXPERIMENT_B1 realized panel]`.

---

## 9 · Risks

| Risk | Mitigation |
|---|---|
| The corpus needed does not exist on the server | the manifest is step zero and blocks launch; the panel shows what is missing rather than failing at run time |
| The cross product invites a fishing expedition | `max_runs` cap, curated-by-default, fixed baseline + measured margin, one-stage-at-a-time framing |
| The console hardens a bad default (W=8/H=4) by making it the path of least resistance | the window control is multi-valued by default and the readout states the provenance: APF-derived, smallest-W tiebreak, 128 never swept |
| Porting the first-generation lens code imports the phase collision | ANALYSE ships `none`/`fft`/`cepstrum` only; the rest stay visibly unbuilt until the ~30-site fix is its own plan |
| Two consoles, two bridges, one server | distinct ports, tokens, screen prefixes; K5 pins the capture side |
| A5 refuses every spatial scheme | that is a measured result about guest-physical scatter, reported as one, not a reason to relax the gate |

---

## 10 · Recommendation

Build it in four steps, each independently useful and each ending in something checkable:

1. **`corpus_manifest.py` + `channel_roster.py`.** Unblocks B1's step zero and produces the two facts
   every panel needs. Useful even if no UI is ever built.
2. **`scheme_run.py` + `gates.py` + `analysis_run.py`, driven from the command line.** The analysis
   becomes composable and gated before it becomes clickable. K2 is provable at this step.
3. **The console: template, build, bridge, launcher.** Panels 1-11.
4. **Comparison (panel 12) and the results index.** The part that answers Entry 13's question.

Order matters, and it is the capture side's own order: plans, implementation, named gates, an
orchestrator that enforces the lifecycle, then a console that lets a human compose and watch. The
console was last on the capture side because it is the least valuable half-built.

---

## Sources

Read directly this session, on branch `b1-encoding-floor` unless noted:
`plan07_campaign/ui/{build_console.py, console.sh, console_bridge.py, capture_console.template.html}`,
`plan07_campaign/subset_run.py`, `plan08_b1/*.py`, `plan08_b1/b1_extract_all.sh`,
`plan02_validate_session.py`, `config_qemu_upc.json`,
`live_delta_calc_modular/src/metrics/{mod.rs, family_a/informational.rs, family_b/structure.rs,
family_b/spatial_shift.rs, family_c.rs, family_d/texture.rs}`,
`docs/{ANALYSIS_PIPELINE_METHODOLOGY.md, METHODOLOGY_AS_EXECUTED.md,
RAJA_REVIEW_ANALYSIS_METHODOLOGY.md, EXPERIMENT_B1_ENCODING_FLOOR.md}`,
`docs/b1_diagnosis/00_SYNTHESIS.md`, `docs/dwarf_pilot_design/{GATE2_RESULT.md, GATE3_PLAN.md}`,
`docs/research-diary/{part1_reduction.html, part_method.html, part_graph.html}`,
`docs/plan04_segmenter_proposal.html` (template), `~/b1_data/b1_ae_results.json`.

Channel counts, the 13 dead names, and the B1 recall figures were recomputed here from those files
rather than taken from any document.
