# Methodology As Executed: Plans 01-07

Derived 2026-09-01 from a full read of the plan documents, the plan code, the tuning-plan
pre-registrations and audits, the Desktop deepdives, the doctoral plan, and the defense decks.

Companion: `ANALYSIS_PIPELINE_METHODOLOGY.md` (the forward-looking feature-extraction layer,
which is O3/O4/O8 of the doctoral plan). This document is the backward-looking record it
sits on top of.

## What this document is

The corpus states the methodology in five places that do not agree: the doctoral plan, the
per-plan overview documents, the pre-registered tuning plans, the code, and the defense decks.
This document derives one methodology from all five, states which source won each conflict and
why, and lists separately the conflicts that no rule can settle.

**Accuracy claim.** Every parameter, threshold and result here is traced to a file, and where a
number was recomputed it was recomputed here. **Completeness claim.** Every conflict found is
listed, including the ones left open. A document that read smoothly over a contradiction would
be less accurate, not more, so the open items are named rather than smoothed.

**A note on how to read Section 4.** An earlier draft of this document presented the defect
record as a list of live bugs. That was wrong, and checking each item against current code
showed why: **the failures were found and fixed, and several of the apparent defects are
deliberate design.** `coverage_ratio` in the `FULL` feature set is the control arm of the
leakage ablation, not a leak. The `g4_verdict` placeholder is excluded from the acceptance
conjunction and so cannot cause a false pass. The simulated null thresholds gate nothing that
runs. Older modules retain older conventions while the live path has converged on one.

The distinction that matters throughout: **fixed** (Section 4), **not yet started**
(Section 4bis), **records needing update** (Sections 4bis and 8). There is no fourth category
of live breakage in the code.

### Resolution rules

Applied in this order. Where a rule decides, the resolution is marked `[rule N]`.

1. **Code over document.** What executes is what happened.
2. **Later plan over earlier.** Plans supersede in order; a supersession is noted even when the
   earlier document was never withdrawn.
3. **Campaign over pilot.** A 66-cell production run beats a 6-cell probe.
4. **Measured over simulated.** A number from a real run beats one labelled simulated.
5. **Artifact over prose.** A committed result JSON beats a sentence about it.

Where two rules conflict, or none applies, the item goes to **Section 8: open decisions**.

### Evidence tags

`[traced]` read directly from the named file. `[computed]` recomputed for this document.
`[inferred]` assembled across sources; the reasoning is given. `[derived]` an assistant
proposal, not a position taken by JK.

---

## 1. The chain, as it actually ran

```
Plan 01  calibrate the instrument      -> 3 time axes; pause-fraction curve; flush verdict
Plan 02  sampling interval per family  -> iv table; claims C1-C5; apf_trajectory.jsonl
Plan 03  analysis window and hop       -> (W=8, H=4); gates G1-G5; claim C7
Plan 04  change-point segmentation     -> CUSUM; claim C8; the marker non-observability result
Plan 05  capture throughput            -> delete-as-you-go; 66-cell campaign; claim C9 (unadjudicated)
Plan 06  disk as a second channel      -> masquerade resolved; disk is not a threat marker
Plan 07  behaviour recogniser          -> Step 1 only; retracts Plan 06's framing
```

Each plan consumes the previous plan's artifact and refuses to run without it. Plan 04 checks
`schema == plan03.window_hop_recommendations.v1` before starting `[traced:
plan02_validate_session.py:157]`.

**A gap in the chain.** Plan 03 states it consumes "the 90 trajectories captured under the
B+3.1 lossless v2 pipeline". Plan 02 documents its 90 cells as having no dumps and null
F1/CV, and documents post-D-51 work only as a 6-cell sub-pilot. **The 90-cell lossless v2
capture that Plan 03 reads is described in no document read.** `[inferred]` Plan 03 also cites
"Bug O", which appears in neither Plan 01's bug list (D-H) nor Plan 02's (J-M).

---

## 2. Canonical parameters: what is actually live

Every value below is from code, not from a plan document `[rule 1]`.

| Quantity | Live value | Source | Superseded value still in circulation |
|---|---|---|---|
| Page size | 4096 B | `plan02_apf_helper.py:60` | - |
| Guest RAM | 1024 MiB | `config_qemu_upc.json` | 512/256 arms designed, never run |
| Sampling interval | **500 ms, all workloads** | `plan05_campaign/build_cells_dir.py:37` | Plan 02's per-family table (250/500/1000) |
| Analysis window / hop | **W=8, H=4** | `plan04_classify.py:324-325`; campaign modules | `WINDOW_DEFAULT=128 / HOP_DEFAULT=64` in the validator |
| Sweep grid | W {8,16,32,64} x hop ratio {0.25,0.5,1.0} | `plan03_sweep.py:275-277` | tuning plan proposed up to 512 |
| Working trace length | 300 s recommended | Plan 07 window yield | 120/300/600 all captured |
| Rhythm constants | phasic 20.0 s, steady 30.0 s | `plan03_sweep.py:49-52` | - |
| CUSUM steady | k=2.0, h=4.0 | `plan04_cusum.py:185-186` | - |
| CUSUM phasic | **k=0.1, h=0.5** | `plan04_run.py:317-326` | **not recorded in the output JSON** |
| Compression | zstd-3, fixed, no level flag | D-64; measured 15.12% | assumed 30% prior |

### The interval supersession `[rule 2 + rule 3]`

Plan 02's deliverable was a six-row per-family interval table. Plan 05 locked every one of the
11 workloads at 500 ms so the window gain would be attributable to retention alone, and the
66-cell campaign ran that way. **Plan 02's table is superseded in practice and was never
formally withdrawn.** Anything citing per-family intervals as a live result is citing a table
no campaign used.

The Plan 03 proposal independently states workingset = 250 ms, against Plan 02's 1000 ms
`[traced]`. Both are moot under the 500 ms lock.

### The window conflict `[rule 1]`

`plan03_overview.html` says in prose that (8,4) is "the universal winner across all eleven
workloads" and in its own table gives `app_hashtable_intensive_v2` as **(32,8)**. Plan 04 read
the recommendation artifact and ran the hashtable at (8,4) `[traced]`. **Live value is (8,4)
for all eleven.** The (32,8) row rests on n=2 and Plan 03 itself calls it fragile.

Separately the validator still counts windows at 128/64 while every analysis runs at 8/4. That
is not a contradiction (different subsystems) but it is the source of the C3 problem in §3.

---

## 3. The gate ledger

### 3.1 Per-cell claims C1-C9

| Claim | Test | Threshold | Blocking |
|---|---|---|---|
| C1 workload_ran | `PHASE` markers in `workload_stderr.log` > 0 | > 0 | yes |
| C2 ratio_healthy | `snap_completion_ratio` >= floor | **0.08** if `keep_dumps` else 0.30 | yes |
| C3 enough_windows | `(n_snaps-W)//H + 1 >= min_windows` | **>= 50** at W=128, H=64 | **no** |
| C4 no_settle_retries | `lock_retries == 0`; absent note = NA-pass | == 0 | yes |
| C5 producer_log_clean | error count in `producer.log` | == 0 | yes |
| C6 apf_complete | sentinel present and `n_ok/n_expected` | **>= 0.95** | conditional |
| C7 window_hop_recommended | all `passes_acceptance` true | all-pass | conditional |
| C8 segmenter_quality | the Plan 04 gate for this family | per-family | conditional |
| **C9 throughput** | proposed by Plan 05, **never adjudicated** | >= 2x | - |

`ok = c1 and c2 and c4 and c5 and c6 and c7 and c8` `[traced:
plan02_validate_session.py:437]`. C3 is absent from the conjunction.

**C9 is an orphan.** Plan 05 defines it, G-T1 failed all 18 comparisons, and no pass/fail
verdict appears in any Plan 05 document `[inferred]`. The ledger should either adjudicate it
as failed or withdraw it.

### 3.2 C3 cannot referee the window question

C3 does not ask "is there room for one window". It asks for **at least 50 windows**, counted
at the validator's W=128/H=64. Working backwards `[computed]`:

| Window / hop | Windows at 700 frames | C3? | Snapshots needed for 50 |
|---:|---:|:---:|---:|
| 128 / 64 (validator default) | 9 | FAIL | 3,264 |
| 64 / 32 | 20 | FAIL | 1,632 |
| 32 / 16 | 42 | FAIL | 816 |
| 16 / 8 | 86 | PASS | 408 |
| 8 / 4 (live grid) | 174 | PASS | 204 |

C3's verdict is a function of the window under test, so it structurally prefers short windows
and cannot arbitrate between candidates. Under the live (8,4) grid it passes easily; under its
own declared default it fails at any realistic trace length. **Restate it as statistical
adequacy (enough independent windows for the estimator) or retire it.** `[derived]`

### 3.3 Analyzer gates G1-G5 (Plan 03)

| Gate | Test | Live value | Pre-registered value |
|---|---|---|---|
| G1 stationarity | median `stat_pass_frac` | >= 0.80 | >= 0.90 acceptance of windows |
| G2 coverage | median `coverage_ratio` | >= **2.0**, auto-pass for steady | >= 4x rhythm |
| G3 phasic | median cepstral SNR | >= **4.5 dB** | >= 5.0 dB, and before that F1 >= 0.8 |
| G3 steady | CV ceiling | **0.30** short / **0.50** long | 0.05 / 0.15 |
| G4 hop | `hop*2 <= window` | deterministic | same |
| G5 yield | fraction of eligible cells with >= 5 windows | >= 0.80 | same |

**G2 has never been cleared by any phasic workload**, 0/12 at every combo across all five
`[traced]`. Plan 03 calls the phasic recommendation "best-feasible, not a clean pass".

### 3.4 Segmenter gates (Plan 04)

| Gate | Live test | Status |
|---|---|---|
| G1 phasic F1 | median >= 0.67 AND P(F1>=0.50) >= 0.75 | **retired**; 0 applicable |
| G2 steady | median spurious <= 1 AND P(spurious=0) >= 0.5 | 5/6 = 0.833 |
| G3 plausibility | fraction in band [1,20] >= **0.70** | 5/5 = 1.000 |
| G4 legacy regression | **`g4_verdict: bool = True`** hardcoded | reported as 11/11, 1.000 |

**G4 is a placeholder reported as a passing gate** `[traced: plan04_run.py:450, 502]`.

**G3's ceiling is inert.** Max boundaries observed is 10, so the band [1,20] reduces to
"n >= 1". The code says so: "a permissive 'any-detection-at-all' check" `[traced:
plan04_run.py:59-61]`.

### 3.5 Throughput gates (Plan 05)

G-T1 speedup >= 2.0: **FAIL, all 18**. G-T2 fidelity: 9/18. G-T3 disk <= 3 GiB: PASS all.
Note G-T2's third leg is a disjunction (`D <= 0.10 OR p > 0.05`) inside a gate documented as
"all three must pass" `[traced: plan05_fidelity.py:213-215]`.

---

## 4. The defect record: what broke, and how it was fixed

**Read this section as history, not as a bug list.** Everything below was found and closed.
It is kept because the failures are instructive and because the thesis's strongest claim is
that the instrument catches its own errors. What genuinely remains is in Section 5 (unstarted)
and Section 6 (bookkeeping).

### 4.1 Capture-side failures, all closed

| Was broken | Consequence if undetected | Closed by |
|---|---|---|
| `bc` missing on the server turned a sleep into `sleep 0` | every frequency ever computed placed on the wrong part of the spectrum, **15x** off | Plan 01 |
| `cell.workload` was metadata only; no workload was launched | 90 cells of **idle Kali** passing every gate | D-19/D-20 |
| `TIMING_SELF_CLEAN=1` deleted every dump | no data to analyse | D-19/D-20 |
| libvirt state-change lock held across cells | campaign hangs; D-31's `domstate` poll watched the wrong signal | D-34 (retry `virsh resume`) |
| `status=ok` on cells with 3 of ~85 snapshots | false-positive cells entering analysis | D-32 completion ratio |
| Guest scratch on a 483 MB **tmpfs** | IO workloads silently measuring **memory**, not IO | `c15c83f` -> `/var/tmp` |
| Working sets larger than guest RAM; `--max-mb` default 8192 | `mmap` failures on 9 + 6 steps | `f107d42`, `8b6498d` |
| Guest zsh aborts a command whose glob matches nothing | wipe/reclaim silently did nothing | `2a4bd7b` (`find -delete`) |
| Workload SSH returns 255 inside a producer suspend window | 7 of 101 steps failed | `93a0c33` retries |
| Consumer archived (prev,curr) but deleted only prev | ~101 GiB leaked per campaign | `eade1bd` sweep |
| pmemsave denied by the domain's AppArmor profile | no dumps at all | `1322c41` per-domain dir |
| Producer slept during backpressure with the VM **running** | guest `--duration` budget burned; non-uniform sampling, which the spectral metrics assume away | `8f5b352` VM suspend |
| Single-pass workloads exit in seconds | 2-7 snapshots per cell | `6eae374` SUSTAIN_LOOP |
| Dump dir owned by libvirt-qemu; `unlink()` failed silently | 41 dumps / 44 GiB in a 120 s smoke | `TIMING_SUDO_DELETE` |
| Cold-boot SSH race | cells launched against a VM that was not up | `fbff629` (3 consecutive probes) |
| QA harness `ssh` ate the while-read loop's stdin | **false pass on 1 of 101** | `b072190` (`ssh -n`) |

### 4.2 Analysis-side failures, all closed

| Was broken | Consequence | Closed by |
|---|---|---|
| Gate booleans written flat, validator read a nested `gates` dict | every lookup hit the NA branch: **132/132 faked**; honest count 60/132 | D-84 |
| `windowed_cusum` used `z[t-1]`, docstring and inline detector used `z[t]` | exported helper disagreed with the detector | D-84 |
| G3-ransom gated on an F1 from a **window-independent** detector | identical value at every (W,H); the gate could not discriminate what it gated | swapped to cepstral SNR |
| G5 denominator counted cells too short to ever yield 5 windows | gate impossible to pass | restricted to `n_pairs >= W+4H` |
| ~63-hour clock skew between stderr and snapshot timings | every marker collapsed to snap-index 0 | D-85 `--marker-mode relative` |
| **Feature leakage**: `coverage_ratio` is a per-family constant | binary accuracy **1.000**, entirely spurious | AGNOSTIC split (§4.3) |
| Marker-aligned F1 assumed to be measurable | it is structurally unavailable; markers and mean-shifts are different events | D-86, disclosed as the headline result |
| Plausibility band assumed to detect phase | null calibration: real 0.850, shuffled 0.850, IID 0.850, **gap 0.000** | measured and disclosed; `band_near_unfalsifiable: true` |
| Cepstral SNR read as a discriminator | SNR alone classifies at **0.508**, chance | demoted to a presence gate |
| Per-rep seed used Python's salted `hash()` | not reproducible across processes | `subset_run.py:288` uses `zlib.crc32`, with the reason in a comment |
| **53 of 132 cells DOF-starved** | coverage gate unmeetable, segmenter hedged | Plan 05 retention -> **0 of 66** |

### 4.3 The leakage fix, stated precisely

This one is worth stating exactly, because a casual read of the code suggests the leak is
still present. It is not. `behavior_families.py` is headed "leakage-aware" and defines:

```python
FULL = [..., "cv_workingset", "f1_phase", ..., "coverage_ratio"]
FAMILY_CONDITIONAL = ["cv_workingset", "f1_phase", "coverage_ratio"]
AGNOSTIC = [c for c in FULL if c not in FAMILY_CONDITIONAL]
```

`FULL` is **retained deliberately as the contaminated control arm**, which is what makes the
leak measurable. Every live campaign module uses `AGNOSTIC`: `peakvar_lift.py` ("the
leakage-free AGNOSTIC set"), `anomaly_detect.py` ("Leakage control: AGNOSTIC features only"),
`separability_matrix.py`, `diskio_lift.py` `[traced]`.

Two consequences follow, and both are benign:

- `coverage_ratio` taking exactly two values (0.2 phasic / 0.1333 steady, `[computed]`) is the
  demonstration, not the defect.
- `f1_phase` being 100% NaN in the shipped sweep (`[computed]`: 0 of 792 rows) affects only
  `FULL`, since `f1_phase` is inside `FAMILY_CONDITIONAL`. No honest result depends on it.

### 4.4 Convergence in the live path

Older modules retain older conventions; the live path converged. Verified `[traced]`:

- **Taxonomy.** `peakvar_lift`, `diskio_lift`, `make_comprehensive_report` and
  `separability_matrix` all import one `family_of` from `behavior_families`. The other
  classification lists belong to modules doing different jobs (capture-side dispatch, gate
  selection).
- **`apf_std`.** Models read `sweep.csv`, which carries `plan03_metric_kernel`'s population sd.
  `metrics.json` (sample sd) is read by no model.
- **Simulated nulls.** `F1_null = 0.52` / `CV_null = 0.31` live only in `plan02_analysis.py`,
  which nothing imports outside a test. **They gate nothing live.**
- **`plan04_classify.py`** is superseded by `behavior_families.py` and imported by nothing.
- **NA-passes are labelled, not silent.** C6/C7/C8 carry `"operational": False` with an
  explicit reason string, e.g. `"no plan03_recommendation.json found (Plan 03 not run) · NA"`.

---

## 4bis. What is genuinely open, and why

Nothing in this list is broken. Each item is either work not yet started, or a record that
needs updating.

### Not started (correctly deferred)

| Item | Blocked on |
|---|---|
| **E3 / G4 legacy regression** | G1's F1 numbers are structurally zero, so there is nothing to regress against. Note G4 is **not in the acceptance conjunction** (`for v in (g1, g2, g3)`, `plan04_run.py:452-455`), so it cannot cause a false pass. |
| **Task C / `TIMING_MAGENT`** | spec written (`magnitude_entropy_spec.md`), capture emit not implemented. RQ5 and the stealth family wait on it. |
| **Substrate families G and H** | need a capture-side state model (pinned baseline / EWMA / lagged buffer); the pipeline deletes `prev` immediately. |
| **`SUBSTRATE_FEATURES` backend** | resolution logic complete in `subset_run.py`; no consumer reads the env var yet. |
| **The entire spatial axis** | every result to date is APF, which averages the page axis away. This is the subject of `ANALYSIS_PIPELINE_METHODOLOGY.md`. |
| **Non-invertibility proof** | not written. Five doctoral-plan sections depend on it, including the data-protection argument. |
| **Vortex GPGPU realization** | O7/M6; planning done, implementation not started. |
| **Encoding comparison** | named as a central methodological contribution (R2); never run. |

### Minor provenance items

- **Phasic detector settings in the artifact.** `plan04_run.py:627-635` records `k`, `h`,
  `min_separation`, `marker_tolerance`, `marker_mode`, `phasic_mode`. It does not record
  `k_phasic` / `h_phasic`, so the two numeric overrides are not recoverable from the artifact
  alone (though `phasic_mode` tells a reader which path ran). Disclosed in the deepdive.
- **A stale word in a changelog entry.** `plan03_aggregate.py:550` sits inside the artifact's
  `caveats` list, alongside "spectral coverage gate relaxed from 4x to 2x". It still says
  ">= 5 dB" where D-83 moved the live constant to 4.5 (`:48`). The gate is correct; the
  historical note beside it was not updated.

---

## 5. Results that stand

Stated with the protocol attached, because the protocol is what the numbers mean.

### 5.1 The capture campaign worked

**0 of 66 cells DOF-starved**, against a v3 baseline of **53 of 132** `[traced]`. 29,003
snapshot pairs, 7,159 analysis windows, all 66 `rc=0`. Median windows per cell at 600 s: 242.

Copy per snapshot 7.06 s -> 0.79 s; snapshot cycle 9.75 s -> ~1.5 s; snapshots in 600 s
53 -> 398. **The lever was delete-as-you-go retention, not RAM and not tmpfs.**

### 5.2 The 7.06 s cost was an artifact

`[traced: plan05_deepdive.md]` The pilot ran delete-as-you-go on every arm, so all four arms
already sat at the copy floor and the gate compared already-fixed arms. The 7.06 s median was
inflated by never deleting dumps, which filled the disk and slowed every write. It is the
median of a right-skewed distribution (0.76 / 7.06 / 21.57).

**Any paper quoting 7.06 s as an intrinsic snapshot cost is quoting a retention artifact.**

### 5.3 The honest classification numbers

On the 66-cell campaign, AGNOSTIC (7 features) vs +peakvar (12):

| Task and protocol | AGNOSTIC | +peakvar |
|---|---|---|
| binary, leave-one-replicate-out | 0.924 | 0.970 |
| binary, **leave-one-workload-out** | 0.606 | **0.742** |
| binary, **leave-one-family-out** | 0.273 | 0.576 |
| family 5-class, LORO | 0.909 | 0.955 |
| family 5-class, LOWO | 0.364 | 0.349 |
| one-class LOF, LOWO | 0.797 | **0.942** |
| one-class LOF, LOFO-benign | 0.805 | **0.959** |

Binary prior 0.545, family prior 0.364. The starved-v3 AGNOSTIC LOWO of 0.466 becomes 0.636
on healthy data.

**Read the LOWO and LOFO rows, not LORO.** LORO trains on one replicate of the same workload;
it measures memorization of the run.

### 5.4 The negative results, which are the strongest work

- **The 100% is one leaked feature.** `coverage_ratio` alone reproduces it exactly.
- **The CUSUM plausibility gate has no phase-discriminative power.** Real phasic 0.850;
  time-shuffled surrogates 0.850; IID Gaussian 0.850. Gaps of **0.000**. The artifact
  self-flags `band_near_unfalsifiable: true`.
- **Marker-aligned F1 is structurally unavailable.** Phase markers describe events inside the
  workload; CUSUM detects mean shifts in the memory signal. Different kinds of event.
- **Cepstral SNR is a presence gate, not a discriminator.** SNR alone classifies at 0.508,
  chance; the steady hash-table workload has among the highest SNR of all.
- **Disk is not a threat marker.** Benign `mem_mmap_traversal` writes 140.7 MB/s, more than
  any threat workload.

### 5.5 The masquerade is resolved, on a second channel

`ransom_slowburn` 0.20 MB/s vs `mem_pagefault_density` 0.002 MB/s, ~100x separation, on the
pair that are twins in memory `[traced]`. But adding disk **hurts** detection (binary LOWO
0.742 -> 0.636), which is why Plan 07 demotes it to a supporting ablation.

### 5.6 Plan 07 retracts Plan 06's framing

> "the earlier 'characterisation, not detection' reading **understated the signal** ... the
> negative result was a property of **how we asked the question** (a supervised boundary
> across too few workloads), not of the memory signal."

Different in kind from the leakage result: the instrument was sound, the question was badly
posed.

---

## 6. Conflicts resolved by rule

| # | Conflict | Resolution | Rule |
|---|---|---|---|
| 1 | iv per-family table vs 500 ms lock | 500 ms; Plan 02 table superseded | 2, 3 |
| 2 | workingset iv 1000 vs 250 ms | moot under the lock | 2 |
| 3 | (8,4) universal vs hashtable (32,8) | (8,4); Plan 04 ran it that way | 1 |
| 4 | G3 SNR 5.0 vs 4.5 dB | 4.5 live; the 5.0 in artifacts is stale text | 1 |
| 5 | F1 >= 0.8 (plans) vs >= 0.67 (code) | 0.67 | 1 |
| 6 | CV <= 0.15 (plans) vs 0.30/0.50 (code) | 0.30/0.50 | 1 |
| 7 | Plausibility 0.80 vs 0.70 | 0.70 | 1 |
| 8 | C2 floor 0.85 vs 0.30 vs 0.08 | 0.08 under `keep_dumps` | 1 |
| 9 | Flush saving 10% vs 25% | 25% is the arithmetic of 1.520 -> 1.140 s | computed |
| 10 | pmemsave 7.06 s vs 0.79 s | 0.79 s; 7.06 was a retention artifact | 2, 3 |
| 11 | Plan 06 "not a detector" vs Plan 07 | Plan 07 retraction stands | 2 |
| 12 | "66 cells" two datasets | disambiguate: v3-half vs campaign | inferred |
| 13 | v2 median ransom F1 1.0 / 0.67 / 0.95 | 0.67 across 45 cells | artifact |
| 14 | G2 arithmetic W>=40 vs W>=80 | W >= 80 follows from the stated rule | computed |

---

## 7. Designed and never run

| Item | Where designed | Status |
|---|---|---|
| Plan 01 two-row validation pilot | `01_instrumentation_logging_plan.md` | no result |
| VM-RAM sweep {256,512,1024} | `02_interval_tuning_experiment.md` | dropped, "not on the critical path" |
| Flush-sensitivity arm | `02_...md` | dropped |
| Step 0 sensitivity probe (18 cells) | `experiment_audit.md` | result "TBD" |
| Step 0.5 null baseline on the real host | `experiment_audit.md` | simulated instead |
| Step 3 classifier validation | `experiment_audit.md` | never reported |
| Plan 02 v2 metric-driven re-capture | `iv_recommendations_v1.json` | only a 6-cell probe |
| keep-dumps modes 1 and 2 | `keep_dumps_audit.md` | queued, never landed |
| MAGENT capture (`TIMING_MAGENT`) | `magnitude_entropy_spec.md` | "SPEC (not yet implemented)" |
| Substrate families G and H | `feature_substrate_spec.md` | deferred, need capture redesign |
| `SUBSTRATE_FEATURES` backend | `subset_run.py:264-275` | env var nothing reads |
| True keep-everything baseline | Plan 05 | the clean contrast never run |
| Plan 07 Steps 2a onward | `plan07_execution_log.html` | only Step 1 executed |

**The tuning plans and the code have diverged.** `tuning_plans/03_*.md` and `04_*.md` both
still read "Status: plan, not yet implemented" while the code that replaced them has been
running for months with different thresholds. The `results/` folder the README instructs work
to land in was never created. **Anyone reading that folder as the pre-registration will quote
the wrong thresholds into the thesis.**

---

## 8. Open decisions: no rule settles these

### 8.1 Which downstream numbers are canonical

Plan 05 and Plans 06/07 report **different values for eleven metrics on the identical 66-cell
dataset**. Only two rows are reconciled anywhere.

| Metric | Plan 05 | Plan 06/07 |
|---|---|---|
| binary LOWO, AGNOSTIC | 0.636 | 0.606 |
| binary LOWO, +peakvar | 0.788 | 0.742 |
| binary LOFO, AGNOSTIC | 0.303 | 0.273 |
| binary LOFO, +peakvar | 0.636 | 0.576 |
| 5-class LORO, AGNOSTIC | 0.955 | 0.909 |
| 5-class LORO, +peakvar | 1.000 | 0.955 |
| 11-way LORO, AGNOSTIC | 0.879 | 0.833 |
| 11-way LORO, +peakvar | 0.894 | 0.924 |

The Plan 07 handout then mixes both sources in one sentence. **This must be resolved by
re-running one pipeline and declaring its output canonical**, not by choosing a column.

### 8.2 C9

Adjudicate as failed (G-T1 failed 18/18) or withdraw it from the ledger.

### 8.3 C7's pass semantics

All-pass vs any-pass vs an NA carve-out for phasic. Plan 03's team recommended the carve-out;
the proposal recommended all-pass with a documented carve-out. Never resolved `[traced]`.

### 8.4 Whether to re-derive the simulated thresholds

F1 >= 0.70 and CV <= 0.27 rest on simulated nulls (§4.5). Either measure them on the real host
or relabel them as design choices.

### 8.5 The doctoral plan's unbacked claims

The doctoral-plan lane found claims with no evidence attached, ranked by dependency:

| Claim | Sections depending on it | Evidence |
|---|---|---|
| Non-invertibility proved | O1, R1, R5, DMP FAIR, DMP ethics | none found |
| Spatiotemporal localization | O8, O5, R3, M4 | none |
| Form generalizes across substrates | O6, O7, M6, R4, R6 | Vortex unbuilt |
| Encoding comparison | O2, O3, O5, R2, R3 | never run |
| Early anomaly detection | O5, O8, R3, 3 papers | none |
| False-positive rate bound | O5, R3 | none |

The non-invertibility proof is the highest-leverage: **five sections depend on it, including
the entire data-protection argument.** The decks state "so GDPR does not arise" while the same
deck lists the proof under ANTICIPATED RESULTS, and one deck says "proved" four slides after
listing it as anticipated.

### 8.6 v7's two centres

The Motivation says the novelty is the behavioural capability; R1 says the central conceptual
contribution is treating encoding as a design choice and comparing against a fixed baseline.
Incomplete migration from the v5/v6 form-first spine.

### 8.7 Which deck is authoritative

`RTD_defense_FINAL.pptx` is dated 2026-07-10 13:41; eight decks postdate it, and a `to_submit/`
folder holds the 12-minute version (2026-07-14). `[computed from file mtimes]` Every measured
result number was deleted between drafts; the only quantitative claim left on any slide is the
spurious 100% obtained by deliberately cheating. `deck_anchoring_review.md` prescribed six
fixes and **none were applied**.

---

## 9. Principles the corpus establishes

These are JK's, stated in his own notes, and they are the reusable part.

- **A perfect score is a symptom, not a triumph.** Stated in the 2025 thesis results chapter
  unprompted, and again 16 months later at the leakage finding.
- **Plausibility is not correctness.**
- **A gate that never fails anything is evidence of nothing.** "the single exception (the
  bimodal hash-table at (32,8)) shows the gates discriminate rather than rubber-stamp."
- **Presence gate vs discriminator.** A statistic that says "there is a rhythm here" is not a
  statistic that says "this class differs from that one."
- **Bracket everything; trust neither extreme.** Page size, chunk size, interval, window, k:
  each is placed between two named failure modes rather than derived.
- **Where it happened is noise; what kind is signal.**
- **A difference is a derivative**, so the whole signal-processing toolbox applies.
- **Change one role at a time**, hold the rest, re-check the same invariants.
- **Tuning needs one clean exemplar per class; classification needs the full set.**
- **A failed gate can be a result about the framing, not about the lever.**
- **The negatives are the contribution.**

---

## 10. Where the analysis layer attaches

`ANALYSIS_PIPELINE_METHODOLOGY.md` designs the feature-extraction layer. Its place here:

- It implements doctoral objectives **O3, O4 and O8**, methodology stages **M4 and M5**.
- O8's own defense states the dependency: localization "only becomes possible once the richer
  per-unit complex encoding is run at campaign scale."
- **Every result in this document is APF**, a scalar that averages the page axis away. The
  spatial axis is entirely prospective; the complex encoding was used only in the first
  generation (19 recordings, n=2 per active subtype).
- The general block is already defined in O8: "A block is one region over one window, and it
  is the unit we learn from and classify."
- M4 already commits to re-opening the temporal tuning per region, which is the sibling of the
  per-channel window question in that document's section 10.

The analysis gates proposed there (A1 no-constant-feature, A2 no-label-correlated-missingness,
A3 grouping-respected, A4 null-calibrated, A5 translation-invariant, A6 phase-injective,
A7 window-adequate) are the analysis-side equivalents of the capture-side gates. Each encodes
a failure mode from Section 4 that was caught by audit and would be better caught at
extraction time: A1 and A2 encode the leakage lesson (4.3), A4 the null-calibration lesson,
A3 the CV-protocol lesson.

The point of building them is not that the analysis is broken. It is that the capture side
already refuses to start a bad run, and the analysis side does not have that property yet
because it has not been built for the spatial axis at all.

---

## Sources

Six independent reads, cross-checked against each other and spot-verified here:

1. Plan 01-07 overview HTML, proposals, results MD, execution log, handout.
2. All `plan0*.py` modules and `plan0*_campaign/` modules.
3. `docs/tuning_plans/` pre-registrations and the three audits.
4. `~/Desktop/plan/plan0*_deepdive.md`, `methodology_defense_brief.md`,
   `defense_content_pack.md`.
5. `~/Desktop/plan/` doctoral plan v5/v6/v7, CONTEXT_SYNTHESIS, SOTA walkthroughs,
   OBJECTIVES_PLAYBOOK, SOURCE_TO_DOC_CONNECTION_MAP.
6. `~/Desktop/plan_final/*.pptx` and `~/Desktop/plan/*.pptx` plus the markdown deck sources.

Recomputed for this document: the C3 window arithmetic, the `f1_phase` NaN count (0/792), the
`coverage_ratio` value counts at (8,4), and the deck file modification times.
