# Plan 10 · Analysis Console — how it looks and behaves

Status: **design, nothing built.** Drafted 2026-09-07. Companion to
`plan10_analysis_console_proposal.md`, which it **amends on three points** (§0).

Evidence: `[traced]` read from the named file. `[computed]` derived here. `[proposed]` designed,
not built. `[derived]` assistant mechanism worked out from a direction JK set, not a position he took.

---

## 0 · What changed since the proposal

| Point | Proposal said | Now |
|---|---|---|
| **Where it runs** | server-side bridge over ssh, mirroring `console.sh` | **laptop-local.** The migrated traces are assumed present on this machine. A bridge is still needed (a browser cannot read the filesystem) but it binds 127.0.0.1 with a token and is launched locally — no ssh, no `-L` tunnel, no migration agent. |
| **Span** | AGGREGATE -> COMPOSE -> ANALYSE -> ORGANISE -> MODEL | **AGGREGATE + feature extraction only.** Composition, division, lenses. It emits features. ORGANISE (scale/PCA) and MODEL (AE/RF/OCSVM/LSTM) are **out** — a separate consumer, exactly as `b1_ae.py` is separate from `b1_features.py` today `[traced]`. |
| **Assumption status** | corpus location an open question | **assumed migrated and local, until we change it.** Stated as an assumption in the console's own source panel, so the day it stops being true the UI says so rather than failing silently. |

The locality assumption is cheap to reverse: the source panel takes a root path. Pointing it at a
mounted or synced server tree changes nothing above it.

---

## 1 · What this console is

**It turns traces of metric channels into feature tensors, by a scheme you compose.**

Input: per-cell trajectories of the metric channels the differ already wrote.
Output: a feature tensor per scheme, plus a provenance sidecar recording every choice.

It is the second half of one pair. The capture console composes *how memory is recorded*; this one
composes *how the recording is read*. They share the metric taxonomy by importing the same source,
never by restating it `[traced: build_console.py imports FEATURES_BY_GROUP, SUBMODULE_COLUMNS from
subset_run.py]`.

**Not in it:** capture, new metrics, models, classification scores, accuracy tables. If a panel would
show accuracy, it belongs in the next tool, not this one.

---

## 2 · The five stages

```
  SOURCE        COMPOSE          DIVIDE            LENS              OUTPUT
  which cells   how channels     how the field     which transforms  what is written,
  which         become one       is cut in time    are applied to    in what form
  channels      signal object    and in address    each tile
```

Left rail = the stages, in order, numbered like the capture console's panels (`01 Subset identity`,
`02 Cadence`, ...) `[traced: capture_console.template.html]`. Right rail = a persistent **readout**:
live counts, estimated cost, warnings, and the scheme JSON, again mirroring the capture console's
`aside.readout`.

```
+-----------------------------------------------+---------------------+
| 01  SOURCE      root, cells, what each has     |  READOUT            |
| 02  COMPOSE     single / vector / complex      |  tiles: 5,568       |
| 03  DIVIDE      W,H in time  ·  W,H in address |  per-tile dims: 8   |
| 04  LENS        fft · cepstrum · wavelet · ... |  est. cost: 4m20s   |
| 05  OUTPUT      tensor layout, retention       |  warnings (2)       |
|                                                |  scheme.json  [copy]|
+-----------------------------------------------+---------------------+
| GENERATED RUNS    per-run checkboxes, reorder, run, live log         |
+----------------------------------------------------------------------+
```

---

## 3 · Panel 01 · Source

**Controls**

- **Trace root** — one path, defaulted to the migration agent's local destination
  `$HOME/thesis_traces/zstd_local` `[traced: console.sh TRACES_LOCAL_DIR default]`, editable, with a
  **Rescan** button.
- **Cell table** — one row per discovered cell: workload, family, duration, rep, `n_pairs`, and
  **what it has**: `zstd chain` / `substrate CSV` / `APF trajectory only`.
- **Filters** — family, workload, duration, rep, minimum `n_pairs`.

**Behaviour**

- Availability is *discovered*, never typed. A cell that has only an APF trajectory is shown greyed
  with the reason `per-page data discarded at capture` — it can never feed composition, because that
  trajectory stored `K/N` and nothing else `[traced: EXPERIMENT_B1_ENCODING_FLOOR.md]`.
- The header states the standing assumption in one line: *"traces assumed migrated to this machine;
  scanned <root> at <time>, N cells."*
- Selecting cells updates the readout's tile count immediately, because every downstream number
  depends on `n_pairs`.

---

## 4 · Panel 02 · Compose

This is where "what is the signal object" gets decided, and it is the panel with the real rules.

### 4.1 The three modes

| Mode | What you pick | What one page becomes |
|---|---|---|
| **Single** | one channel | a scalar |
| **Vectorize** (naive) | N channels | a length-N real vector — no cross-channel math, channels stay separate |
| **Complex** | exactly **one magnitude + one direction** | one complex number: magnitude x angle |

Vectorize is the honest control arm. Entry 2 records that the thesis never ran the ablation of the
fused representation against its parts, and this is that ablation, available as a click rather than
a rebuild.

### 4.2 The channel picker

Same taxonomy as the capture console, imported from the same file, rendered the same way — group
header with `all / none / custom`, then one toggle button per column, then a footer line
`[traced: capture_console.template.html renderGroups]`:

| Group | Family | Columns | Live at `substrateSpeed: 2` |
|---|---|---:|---:|
| amount | A | 20 | **18** (`ncd`, `lz_change` dead) |
| direction | B | 19 | **15** (`kendall`, `cross_corr_lag`, `phase_corr`, `byte_rotation` dead) |
| content | C | 13 | **11** (`bigram_ent`, `autocorr_peak` dead) |
| texture | D | 12 | **7** (4 GLCM, `high_freq_frac` dead) |

`[traced: FEATURES_BY_GROUP in subset_run.py; drops counted in the Rust this session]`

Dead columns render **disabled with a reason chip** — `zero at speed 2` — not hidden. Hiding them
would repeat the lie the 64-column schema already tells.

### 4.3 The rules the panel enforces

1. **Complex needs one of each.** Zero magnitude or zero direction selected: the mode is
   unselectable and says why. Two of either: it asks you to drop one. This is the whole point of the
   mode — a magnitude and an angle, not a bag of channels.
2. **A `content` channel cannot be the angle.** Family C describes the *new page*, not the change:
   it is state, while A/B/D are change. Fusing a state channel with a derivative channel produces a
   number that does not mean what it looks like `[traced: methodology §2.1]`. The picker allows it
   only behind an explicit override with the warning written at the control.
3. **The phase convention is a required field of complex mode**, never a default that hides:
   - `2pi x distance` — **the current convention, and it collides.** `cosine` is a *distance* where
     0 = identical and an all-zero previous page maps to 1, so a page rewritten in place and a page
     freshly allocated land on the same angle `[traced: family_b/structure.rs]`.
   - `pi x distance` — half turn; the extremes become antipodal.
   - `arccos(similarity)` — the true geometric angle.
   The collision is drawn at the control, not in a tooltip: a small dial showing 0 and 1 landing on
   the same tick under `2pi`. Whichever is chosen is written to the sidecar, because it cannot be
   recovered from the output afterwards.
4. **Vectorize warns above a channel count**, because the tile payload and the effective-n gap both
   scale with it.

### 4.4 Greyed, visible, not yet built

`other combiners · gates` from the figure — ratios, products, per-channel gating — render greyed
exactly as the diagram draws them. Visible so the space is legible; unselectable so the console
never claims a capability it lacks.

---

## 5 · Panel 03 · Divide

Two axes, and the panel says so before it asks anything: **time** (snapshots) and **address**
(pages). The field being cut is `channels x pages x snapshots`.

### 5.1 Controls

| Axis | Window | Hop | Notes |
|---|---|---|---|
| **Time** | `W_t` frames, multi-valued | `H_t` frames | live readout: windows per cell = `(n_pairs - W_t)//H_t + 1` |
| **Address** | `W_p` pages, multi-valued | `H_p` pages | must be a whole number of pages; `collapse` is a first-class choice, not `W_p = all` |

`collapse` (average the page axis away) is offered as its own option because every result in the
record to date is a collapse, and it must stay one click away.

### 5.2 Two different things, both called "padding"

The panel splits them, because conflating them is a real error:

- **Edge handling** — what to do with the leftover at the end of an axis: `drop` (default; only whole
  windows), `zero`, `reflect`, `replicate`. Affects *how many* tiles exist.
- **Taper (apodization)** — what each window is multiplied by before a spectral lens:
  `rectangular` (none), `hann`, `hamming`, `blackman`. Affects *spectral leakage*, not tile count.
  This is what "smoother transitions" means for FFT/cepstrum, and it is inert for a
  statistics-only lens — the panel greys it when no spectral lens is selected.

Both are recorded in the sidecar. A tapered window and an untapered one are different data.

### 5.3 The live readout, which is the point of this panel

Whenever a window is set, the readout shows the tradeoff table for the selected cells `[computed,
formula traced from plan02_validate_session.py:125-128]`:

```
W_t=8   H_t=4    ->  174 windows/cell    4 freq bins     21 s span
W_t=32  H_t=16   ->   42 windows/cell   16 freq bins     83 s span
W_t=128 H_t=64   ->    9 windows/cell   64 freq bins    333 s span
```

and beneath it, the sentence the record insists on: **effective n is workloads, not windows.**
Selected: 19 workloads · 8,971 windows. The window number is subordinate typography; the workload
number is the big one.

### 5.4 Two warnings pinned to this panel

- **W=8/H=4 is inherited, not chosen.** Tuned on the APF trajectory, won partly on a smallest-W
  tiebreak, and 128 was never in the sweep grid `[traced]`. The control shows this beside the default
  so the path of least resistance is not silently the fitted one.
- **Address blocks are allocator placement, not program structure.** Gate 2 measured
  `frac_in_longest_run` at 0.087-0.162: a contiguous array in the program's view arrives in guest
  physical RAM as ~3000 fragments `[traced: GATE2_RESULT.md]`. Any `W_p` other than `collapse`
  carries that number inline.

---

## 6 · Panel 04 · Lens

Fan-out: pick many, each produces its own feature block, each declares what it needs.

| Lens | Params | Requires | Emits |
|---|---|---|---|
| **Simple statistics** | none | any tile | the 8 scale-equivariant features already implemented: `mean, std, cov, median, max, p95, peak2med, duty` `[traced: b1_features.py:34]` — full roster and the duty-cycle conflict in §10.1 |
| **Deep statistics** | which set | any tile | `stat_pass_frac`, `n_boundaries`, `skew`, `kurtosis`, `entropy`, **`tau`** — the full inventory, traced to plans 01-05, is §10.2 |
| **FFT** | bins (derived `W_t/2`), output = spectrum \| peak \| band energies, log/linear | real or complex tile; taper recommended | per-band values |
| **Cepstrum** | quefrency range, peak + SNR | same | peak quefrency, `ceps_peak_snr_db` (the name the record already uses) |
| **Wavelet** | family from `pywt.wavelist()`, levels, DWT \| CWT | real or complex tile | per-level energies/coefficients |
| **Scattering** | `J`, `Q`, 1D \| 2D | tile | scattering coefficients |
| **MSC** | Welch params (its own internal windowing) | two or more channels | `msc_peak_snr_db` per page and median `[traced]` |
| **PLV** | drop/normal/perfect thresholds | **`compose = complex`** AND a fitted **reference** baseline | per-page PLV verdicts `[traced: plv_calcolator.py]` |

**Behaviour**

- A lens whose contract is unmet is disabled with the reason on its face: PLV greyed with
  *"needs 2+ channels; 1 selected"*. This is the same grammar as complex mode needing one of each —
  every block states its input contract, and the UI refuses to compose an invalid one rather than
  letting the runner fail later.
- **Wavelet and scattering are honest about their dependencies.** The existing implementations import
  `pywt`, `kymatio`, and `torch` `[traced: VMsig_featureExctraction/wavelet_analysis_features.py]`,
  while the pinned analysis env is `numpy` + `scikit-learn` only
  `[traced: plan08_b1/requirements.txt]`. Until that is resolved they render as *available, needs
  env* with the missing packages named — visible, not silently absent.
- Selecting a spectral lens un-greys the taper control in panel 03 and marks `rectangular` as a
  choice rather than a default.

---

## 7 · Panel 05 · Output

- **Layout** — one row per tile: `cell_id, workload, family, rep, t_index, p_index, <features...>`.
  Long format so a downstream consumer can group any way it likes; the tensor form is a derived view.
- **Format** — `npz` (matching what the model side already loads `[traced: b1_ae.py takes an npz]`)
  plus a CSV/parquet twin for inspection.
- **Sidecar, non-optional** — `scheme.json` recording: source root and scan time, cell list with
  `n_pairs`, channel list, compose mode **and phase convention**, both windows and both hops, edge
  mode, taper, lens list with params, code SHA, and the derived channel roster with its liveness
  method. A feature file without its sidecar is not a valid output.
- **Naming** — `label` under `^[A-Za-z0-9_-]+$`, and an existing label is a **hard refusal**, not an
  overwrite, mirroring `subset_run.py` `[traced]`.

---

## 8 · The canvas question: N8N, or a spine?

You asked whether this could be drag-and-drop blocks.

**What an N8N-style editor buys** is arbitrary topology — worth it when the graph shape is the thing
the user is choosing. **Here the topology is fixed:** source, then compose, then divide, then lens.
The only branching is fan-out at the lens stage. A free-form editor would spend its budget letting
you wire compose into source, and then the gates would spend theirs rejecting it.

**Recommendation `[derived]`: a fixed spine, drawn as a canvas, configured in a panel.**

- The centre shows the five stages as connected nodes, drawn in the Entry 13 idiom — lit path for
  what is selected, grey for built-but-off, amber for designed-unbuilt. It is the same figure the
  diary already uses, made live.
- Clicking a node opens its controls in the right rail. No dragging to connect, because there is
  nothing to connect wrongly.
- The **lens stage fans out visibly**: pick three lenses, three branches appear, each with its own
  chip showing its output width. That is where the N8N feel actually earns something — you can see
  the scheme's shape at a glance.
- Later, if COMPOSE grows real alternatives (gates, ratios, per-channel routing) the same canvas
  can gain edges without a rewrite, because it is already drawing a graph.

Cost check: the capture console is one hand-edited vanilla file, no framework, no build deps
`[traced]`. A static SVG spine with clickable nodes stays inside that constraint. A drag-drop graph
engine does not.

---

## 9 · What the console must refuse

Superseded in detail by §13, which types each of these as hard or soft and gives its source. Kept
here as the short list:

1. A dead channel in any selection — refuse, name the channels, say `zero at speed 2`.
2. Complex mode without exactly one magnitude and one direction.
3. Complex mode without an explicitly chosen phase convention.
4. A window longer than the shortest selected cell — refuse, or drop those cells and say which.
5. An address block that is not a whole number of pages.
6. A spectral lens on a single-sample window (`W_t < 4`), where the spectrum is meaningless.
7. An existing output label.
8. Writing features without the sidecar.

---

## 10 · The statistics this project has actually used

Decided 2026-09-07: **deep statistics ships, with `tau`. `pywt` is added to the analysis env.**

Everything below is traced to the plan code, so the lens panel offers what the record already used
rather than a generic statistics menu. Where a name exists in the code, the console uses **that
name** — a feature called `cov` here must be the same quantity as `cov` in `sweep.csv`.

### 10.1 Simple statistics — implemented, keep the names

| Feature | Definition | Source |
|---|---|---|
| `mean` | arithmetic mean of the series | `b1_features.py:34`, `plan03_metric_kernel` as `apf_mean` |
| `std` | population sd (`pstdev`) | same, as `apf_std` |
| `cov` | `std/mean`, the coefficient of variation | `b1_features.py`; identical to `cv_workingset` in `plan02_metrics_per_cell.py:247`, which uses sample `stdev` |
| `median` | median | `b1_features.py` |
| `max` | max | `b1_features.py`; `apf_max` in `extra_features.py:20` |
| `p95` | 95th percentile | `b1_features.py`; `apf_p95` |
| `peak2med` | `max / median` | `b1_features.py`; `extra_features.py` |
| `duty` | fraction of samples above a threshold | **two conventions exist**: `b1_features.py` uses `0.1 * max` (scale-equivariant); `extra_features.py:56` uses an absolute `0.05` calibrated to APF's scale |

`[traced]` **The duty-cycle split is a real conflict, and the console must resolve it, not inherit
both.** B1's pre-registration already ruled on it: an absolute threshold silently hands a comparison
to whichever arm shares its scale, so any feature that cannot be defined scale-equivariantly is
dropped rather than adapted. **Resolution: `duty` = fraction above `0.1 * max`, per series.** The
absolute variant is offered only as `duty_gt05`, named for what it is, for reproducing plan05 rows.

Note the population-vs-sample sd difference between `b1_features.py` (`pstdev`) and
`cv_workingset` (`stdev`), and that `sweep.csv` carries the population sd while `metrics.json`
carries the sample one `[traced: METHODOLOGY_AS_EXECUTED §4.4]`. The console picks one — population,
matching `sweep.csv`, the file every model actually reads — and records the choice in the sidecar.

### 10.2 Deep statistics — what the plans used, plus `tau`

| Feature | Definition | Source |
|---|---|---|
| `stat_pass_frac` | fraction of windows whose mean sits within 1 global sd of the global mean | `plan03_metric_kernel.py:60`; **1-D stationarity, not the 2-D PLV/MSC path** — the naming confusion is called out in that docstring |
| `cepstral_peak_idx` | argmax of the cepstrum above `min_quef = max(1, n_pairs//8)` | `plan03_metric_kernel.py` via `CepstrumStability` |
| `ceps_peak_snr_db` | `10*log10(peak / median(tail))` over the same quefrency tail | same |
| `n_boundaries` / `stationarity_score` | CUSUM boundary count at (8,4), scored `1 - min(1, n/3)` | `plan04_cusum.py:340` |
| `f1_phase` | boundary F1 against PHASE markers, tolerance 1 | `plan02_metrics_per_cell.py:210` — **100% NaN in the shipped sweep** and structurally unavailable per D-86; offered only with that warning |
| `coverage_ratio` | `(W * iv_ms/1000) / rhythm`, rhythm = 20 s phasic / 30 s steady | `plan03_sweep.py:130,49-52` — **a per-family constant, the June leak**; available only inside a `FULL` contaminated-control preset, never in the default set |
| **`tau`** | decorrelation time: first zero crossing of the autocorrelation, or the integral timescale to that crossing | **new** `[proposed]`, from methodology §10.2 |
| `skew`, `kurtosis` | 3rd/4th standardised moments | new; the differ has `skew_q`/`kurt_q` per page, this is the same idea per window |
| `entropy` | Shannon entropy of the window's value histogram | new |

`tau` earns its place because it is the measurement the per-channel window question rests on: does a
channel's own decorrelation time predict which window wins for it. Producing `tau` per channel per
cell **is** the falsifiable half of that proposal, and this console is where it gets produced.

### 10.3 Spectral and coherence — the existing implementations

| Lens | Class / entry point | Contract that matters |
|---|---|---|
| Cepstrum | `CepstrumStability.compute_cepstrum` / `.compute_cepstral_peak(min_quef_idx)` | takes `[T, N]`; the plan03 path feeds a real series reshaped to `[T, 1]` `[traced]` |
| MSC | `MagnitudeSquaredCoherence.compute_msc`, `.compute_peak_snr` | Welch-style windowing of its own (`generate_window_slices(T, L, S)`) — **its windowing must not be double-applied on top of panel 03's** |
| PLV | `PLVStability.fit_baseline` then `.evaluate_run(drop/normal/perfect thresholds)` | **needs a fitted baseline from a clean run, and takes the phase of a complex signal** |
| Combined | `StabilityValidator.compute_all_features`, defaults `window_size=128, step_size=64` | returns `msc_peak_snr_db_{per_page,median}`, `cepstral_peak_idx_{per_page,median}` `[traced]` |

Two hard consequences for the UI, both new:

1. **PLV requires `compose = complex`.** It reads the phase of a complex signal
   `[traced: plv_calcolator.py:9]`. Under `single` or `vectorize` the lens is not merely unhelpful,
   it has no input. It greys out with that reason.
2. **PLV requires a reference.** It is not a per-tile function: a baseline must be fitted on a clean
   run first. So the console needs a **reference selector** — which cell is the baseline — and PLV
   is unavailable until one is chosen. This is a panel that did not exist in the last draft (§11.1).

Also note `StabilityValidator`'s 128/64 default is the same 128/64 the session validator counts at,
and neither matches the live 8/4. The console shows a lens's own internal window when it has one,
so the two are never silently stacked.

### 10.4 Statistics used for *comparison*, which this console does not do

Recorded so they are not accidentally rebuilt here: one-way ANOVA and Welch t on pairs
(`plan02_analysis.py:159,209`), TOST equivalence, bootstrap sd equivalence, and two-sample KS with
`D <= 0.10 OR p > 0.05` (`plan05_fidelity.py`). These belong to gate and comparison tooling, not to
feature extraction.

---

## 11 · What else we need

Gaps the inventory exposed. Each one is a thing the console must have that no draft has named yet.

### 11.1 A reference / baseline selector

PLV needs a fitted baseline; MSC is a pairwise quantity; any "deviation from normal" framing needs a
normal. `normal_profile.py` already builds a benign envelope from p5/p95 per feature `[traced]`.
**Add a small panel: reference mode = `none` | `cell` | `benign set`, with the chosen cells listed.**
Lenses that need it stay disabled until it is set.

### 11.2 The sampling interval must be read, never assumed

`coverage_ratio` and every time-span readout are functions of `iv_ms`, and this project has already
been burned once by a time axis that was wrong by 15x. **Panel 01 shows `iv_ms` per cell, and a
scheme mixing intervals is refused or split**, because one window means different durations across
those cells.

### 11.3 Sparse-to-dense is a decision, not plumbing

The differ emits sparse rows — an absent page means unchanged `[traced: main.rs]`. So a tile of
`W_p` pages holds only the changed ones. Whether an unchanged page enters a statistic as a zero or
as absent changes `mean`, `duty`, and every moment. **Panel 03 needs an explicit `unchanged pages =
zero | excluded` control**, defaulted to `zero` (the definition APF already uses: `K/N` over all
pages), with the readout showing the resulting occupancy.

### 11.4 A status vocabulary, not silent drops

Plan 03 already has one: `ok | skip:short | skip:nan | error:<msg>` `[traced]`. The console adopts it
verbatim, per cell and per lens, and the run summary shows counts by status. A cell that produced
nothing must be visible as a status, never as an absence.

### 11.5 Input hashing

`plan03_sweep.py` writes `traj_sha256` per row `[traced]`. **The sidecar carries the same:** a hash
of each input series. It is what lets a feature file be tied to the exact input it came from, and it
costs nothing.

### 11.6 Resource controls, now that this runs on a laptop

The dense field is `262,144 pages x ~700 snapshots x channels`. That does not fit naively in memory,
and the laptop has no scheduler in front of it. **A runtime panel: worker count, memory ceiling,
chunk size along the page axis, and a per-cell sentinel so a run resumes** — the same resumability
`b1_extract_all.sh` already implements with its `"final"` sentinel `[traced]`.

### 11.7 A cost estimate before launch, not after

Panel 03 and 04 changes move cost by orders of magnitude. The readout should state, before launch:
cells x tiles x lenses, estimated wall time, estimated output size, and **whether the L1 cache is
invalidated** by the current selection.

### 11.8 One question I cannot answer from the code

`n_pages` is 262,144 for a 1 GiB guest `[traced: config]`, but whether the guest's physical range
contains holes (MMIO, reserved) that appear as permanently dead rows is listed as an open question in
the methodology and was never measured. It matters for the address axis: dead rows at fixed positions
would inflate every "unchanged" count identically across all workloads. **Cheap to measure once from
any existing chain**, and worth doing before the address axis is used in anger.

---

## 12 · Flags: the known-issue registry

Decided 2026-09-07: **nothing known-bad is removed from the UI. It stays, badged, with its history on
hover.** A feature deleted from the console is a feature someone re-derives in six months; a feature
labelled `DO NOT USE — it did this, in that plan` cannot be re-derived innocently.

This is Entry 11's fifth move implemented as a widget: *write the objection where the temptation will
occur*. The prohibition lives on the control, not in a document beside it.

### 12.1 Mechanism

One module, `plan10_analysis/known_issues.py`, injected at build time exactly as `FEATURES_BY_GROUP`
is `[traced: build_console.py]`. The UI holds no prose of its own.

```python
{"id": "f1_phase_unavailable",
 "applies_to": ["feature:f1_phase"],
 "severity": "do_not_use",              # do_not_use | caution | note
 "headline": "Empty in every row of the shipped sweep",
 "detail":   "...",                     # the history, with numbers
 "source":   "plan05_campaign/downstream/sweep.csv; Plan 04 / D-86",
 "verified": "2026-09-07"}
```

Rendering: a badge on the control (red `do not use`, amber `caution`, grey `note`); hover shows
headline, detail, source, and the verification date. Selecting a `do_not_use` item is allowed but
routes through the same acknowledgment path as a soft constraint (§13.3) and writes its id into the
sidecar. Nothing is silently possible; nothing is silently forbidden.

### 12.2 The registry, as it stands today

Every number below was recomputed from the artifacts this session, not copied from a document.

| Item | Severity | Hover text (abridged) | Verified against |
|---|---|---|---|
| `f1_phase` | **do not use** | Empty in **0 of 792** rows of the shipped sweep. Marker-aligned F1 is structurally unavailable: PHASE markers describe events inside the workload, CUSUM detects mean shifts in the memory signal — different kinds of event. Plan 04 / D-86. | `sweep.csv` `[traced, recomputed]` |
| `coverage_ratio` | **do not use** | Not measured from the trace. Computed as `(W x iv)/rhythm` with `rhythm` a per-family constant (20 s phasic, 30 s steady), so at any fixed window it takes **one value per family** — 0.2 vs 0.1333 at W=8. Alone it reproduces the headline 1.000 binary accuracy **exactly** (36/0/0/30). Kept only as the contaminated control arm that makes the leak measurable. | `plan03_sweep.py:130,49-52`; `leakage_ablation.json` `[traced]` |
| `ceps_peak_snr_db` | caution | A presence gate, not a discriminator. Alone it classifies phasic-vs-steady at **0.5152**, against a majority baseline of **0.5455** — it does not beat guessing. Fine for "is there a rhythm here", wrong for "which class is this". | `leakage_ablation.json` `ceps_snr_alone` `[traced]` |
| `cv_workingset` | caution | Only computed for `steady` workloads in the plan03 kernel, so its presence encodes family membership. Sits with `coverage_ratio` and `f1_phase` in `FAMILY_CONDITIONAL`. | `plan03_metric_kernel.py`; `behavior_families.py:40` `[traced]` |
| `duty_gt05` | caution | Absolute 0.05 threshold, calibrated to APF's scale. Not scale-equivariant: in any comparison across encodings it favours whichever arm shares that scale. Use `duty` (fraction above `0.1 x max`) instead. | `extra_features.py:56` `[traced]` |
| `detect_boundaries_diff` | caution | Window-independent by construction, so it returned the same value at every (W,H). A gate built on it could not discriminate the thing it was gating. | `plan02_metrics_per_cell.py:190`; D-83 |
| CUSUM plausibility band | caution | Null-calibrated and it failed: real 0.850, time-shuffled 0.850, IID Gaussian 0.850 — gap **0.000**. The artifact self-flags `band_near_unfalsifiable`. | Plan 04 record |
| phase convention `2pi` | **do not use** (except to reproduce) | `cosine` is a **distance**: 0 = identical, and an all-zero previous page pins to 1. Multiplying by `2pi` glues both ends to the same angle, so a page rewritten in place and a page freshly allocated are indistinguishable. Kept because reproducing B1 and the first-generation features requires it. | `family_b/structure.rs` `[traced]` |
| W=8 / H=4 | note | Inherited, not chosen: tuned on the APF trajectory alone, won partly on a "smallest W" tiebreak, and W=128 was never in the sweep grid. | `recommendation.json`; `plan03_sweep.py:275` `[traced]` |
| the 13 dead channels | **do not use** (hard-disabled) | Identically zero at `substrateSpeed: 2`, while the schema still writes 64 columns. | the Rust, counted `[traced]` |

**One discrepancy found while verifying.** `METHODOLOGY_AS_EXECUTED.md` §5.4 states cepstral SNR
alone classifies at **0.508**; the artifact says **0.5152** (`ceps_snr_alone`, GroupKFold
leave-one-replicate-out, RF n=300). The registry uses the artifact's number and cites it. The
document's figure may come from a different protocol; it is not reconciled here, and the divergence
is noted rather than smoothed.

---

## 13 · Constraints: hard and soft, declared not hardcoded

A control that requires something states the requirement at the point of choosing, and the console
enforces it by **type**, not by hoping the user read a doc.

### 13.1 The two severities

- **hard** — the option cannot be selected while the condition is unmet, and Run is disabled. The
  message names what to change. There is no override, because the lens has no input at all.
- **soft** — the option is selectable, but the readout shows a strict warning and Run requires an
  explicit acknowledgment. The acknowledgment is recorded (§13.3), so a questionable choice becomes
  a fact in the record instead of an invisible default.

Declared beside the known-issue registry, same shape, same build injection:

```python
{"id": "plv_needs_complex", "target": "lens:plv",
 "requires": "compose.mode == 'complex'",
 "severity": "hard",
 "message": "PLV reads the phase of a complex signal. Set Compose to complex.",
 "fix": {"label": "Switch Compose to complex", "sets": {"compose.mode": "complex"}},
 "source": "coherence_temp_spec_stability/plv_calcolator.py:9"}
```

### 13.2 The constraints, typed

| Constraint | Severity | Why that severity |
|---|---|---|
| **PLV requires `compose = complex`** | **hard** | it takes the phase of a complex signal; under `single`/`vectorize` there is nothing to read `[traced: plv_calcolator.py:9]` |
| **PLV requires a fitted reference baseline** | **hard** | `evaluate_run` raises without `fit_baseline` first `[traced: plv_calcolator.py; stability_validator.py:100]` |
| **MSC needs 2+ channels** | **hard** | it is a pairwise quantity |
| **MSC internal window (128/64) vs the panel's tile** | **soft, strict** | MSC re-windows internally with its own Welch slicing; at `W_t = 8` its 128-sample window cannot form a single sub-window, so the value is degenerate rather than wrong-in-principle. Offered with a fix: align panel 03 to 128/64, or set the internal window to `W_t` `[traced: magnitude_squared_coherence.py:30,37; stability_validator run defaults 128/64]` |
| **Complex needs exactly one magnitude + one direction** | **hard** | the mode is defined by it |
| **Complex needs an explicit phase convention** | **hard** | it cannot be recovered from the output afterwards |
| **`content` (family C) as the complex angle** | **soft, strict** | C describes the new page (state), A/B/D describe the change (derivative); fusing them makes a number that does not mean what it looks like `[traced: methodology §2.1]` |
| **Spectral lens with `W_t < 4`** | **hard** | fewer than 2 frequency bins; there is no spectrum |
| **Window longer than the shortest selected cell** | **hard** | yields zero windows; the cells must be dropped or the window reduced |
| **Address block not a whole number of pages** | **hard** | the signal is only defined on the page lattice `[traced: methodology §5]` |
| **A dead channel in the selection** | **hard** | identically zero at speed 2 |
| **Address blocking at all (any `W_p` other than collapse)** | **soft, strict** | Gate 2 measured `frac_in_longest_run` 0.087-0.162: blocks are allocator placement, not program structure `[traced: GATE2_RESULT.md]` |
| **Mixed `iv_ms` across selected cells** | **soft, strict**, with a `split by interval` fix | one window means different wall-clock spans across those cells |
| **`vectorize` above N channels** | **soft** | payload and the effective-n gap both scale with N |
| **Output label already exists** | **hard** | mirrors `subset_run.py`'s refusal `[traced]` |

### 13.3 Acknowledgment is part of the artifact

Every soft warning that is proceeded past, and every `do_not_use` item that is selected anyway,
writes into the sidecar:

```json
"acknowledged": [
  {"id": "msc_internal_window", "at": "2026-09-07T14:20:11Z",
   "note": "reproducing the 128/64 first-generation path"}
]
```

A free-text note is optional but prompted. This is the project's own rule about fitted parameters —
a later change is recorded with its date and reason, never described as a recalibration — applied to
UI choices rather than to constants.

An output whose sidecar carries acknowledgments displays them in the results view. A run that ignored
a warning is not indistinguishable from one that never faced it.

---

## 14 · Open decisions, updated

**Settled:** deep statistics with `tau` (§10.2). `pywt` added for wavelets (`kymatio`/`torch` for
scattering stay a separate call). `duty` defined scale-equivariantly at `0.1 * max`; the absolute
variant survives only as `duty_gt05`.

**Still open:**

1. **Multi-valued stages.** Panels 03 and 04 accept lists, so one scheme expands to several runs.
   Confirm expansion belongs here, or one scheme = one run with comparison across schemes.
2. **`n_pages` per cell** — read per cell from the capture config, or a source-panel constant?
3. **The population/sample sd choice** (§10.1). I propose population, matching `sweep.csv`.
**Settled 2026-09-07:** known-bad features stay in the UI, badged `do not use`, with their history on
hover (§12). Constraints are declared and typed: hard blocks, soft warns and requires an
acknowledgment that lands in the sidecar (§13).

---

## 11 · Build order, unchanged in shape

1. Corpus scan + channel roster — panel 01's data, and the manifest B1 called blocking.
2. The scheme object and its validation rules (§9) — headless, testable.
3. The console: spine canvas + panels 01-05 + readout.
4. The runner, driving compose -> divide -> lens over the selected cells.
5. Reproduce the B1 chain through the console and assert the features match `b1_features.py` exactly.
   Same acceptance test as before, one stage shorter: the console must reproduce what already ran.
