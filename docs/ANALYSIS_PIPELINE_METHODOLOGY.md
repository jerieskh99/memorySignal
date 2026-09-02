# Analysis Pipeline Methodology

Working document. Status: draft, under active discussion. Last revised 2026-09-01.

Scope: the feature extraction layer that turns captured memory traces into inputs for ML.
This is the analysis-side counterpart to the controlled QEMU capture pipeline.

Companion reading: `docs/research-diary/` Entry 1 (the reduction) and Entry 2 (the thesis),
which define the original five-reduction ladder this document rebuilds on new data.

**Evidence convention**, borrowed from the research diary. Every factual claim about the
existing code or data is marked `[traced]` if it was read directly from a file or computed
here, or `[inferred]` if it was assembled from indirect evidence. Anything designed but not
yet built is marked `[proposed]`. A full source list is at the foot of the document.

**Authorship convention.** This document was drafted by an assistant from a conversation with
JK, and the two are not the same thing.

- Unmarked prose states a direction, question, or decision JK has taken.
- `[derived]` marks a mechanism, fix, or argument the assistant worked out **from** a
  direction JK set. It follows from his question but he did not state it, has not endorsed
  it, and it has not been tested or compared against alternatives. Derived material is kept
  because it is useful to have written down, not because it is settled. It is open to
  rejection or wholesale replacement.

This matters because the failure mode of this project has repeatedly been a plausible claim
hardening into an assumption through repetition. A document that blurs whose idea something
was makes that easier, not harder. The sections currently carrying derived material are
2.1, 2.2 (the risk analysis), 4 (block-size bracketing), 7.2, 8.1, 9, and most of 10.

---

## 1. Memory as a three-dimensional signal

When a VM's memory is dumped, the result is 1 GiB of bytes. It is natural to treat that as a
one-dimensional array of addresses, but it is not. The address splits into two meaningful
parts:

```
address = 4096 * (page number) + (offset inside that page)
```

So a single memory dump is already two-dimensional: **which page**, and **where inside that
page**. With 1 GiB of RAM and 4 KiB pages there are 262,144 pages, each 4096 bytes wide.
`[traced: config_qemu_upc.json, ramSizeMb=1024, pageSize=4096]`

Snapshots over time add a third axis. The raw recording is:

```
S_raw(p, o, t)
  p = which page           (262,144 of them)
  o = offset in the page   (4,096 of them)
  t = which snapshot       (~700 of them)
```

Three axes, byte-valued. Memory is a 3D signal in the literal sense: a function of three
variables, not a metaphor.

---

## 2. What the metrics do to it

The differ compares each page against its own previous version. For page `p` at time `t` it
takes two 4096-byte vectors, the page now and the page one snapshot earlier, and computes a
number from them. Hamming distance, cosine distance, entropy delta, and so on. Sixty-four
such numbers. `[traced: live_delta_calc_modular/src/metrics/mod.rs, csv_header() lists 64
columns]`

The metric operator consumes the **offset axis**: all 4096 bytes of a page in, one number
out. But it does this once per metric, and there are 64 metrics. So the offset axis does not
disappear. It is **replaced**:

```
before:  S_raw(p, o, t)     third axis = 4,096 byte offsets
after:   S(p, t, m)         third axis = 64 metric channels
```

Still three-dimensional. Same page axis, same time axis. What changed is that the third axis
stopped being "raw bytes at positions" and became "64 different measurements of what
happened." Fine spatial detail inside the page was traded for semantic richness about the
page.

This is what a convolutional network does in its first layer: a patch of raw pixels becomes
one position with many channels. The same operation has been applied to memory. Within-page
detail is gone; in its place are 64 characterizations of the change.

**Key property: the reduction preserves the dimensionality of the signal. It relocates
resolution rather than removing an axis.**

### 2.1 Wrinkle: Family C is not a difference operator

Families A, B and D are difference operators: they compare page-now against page-before.
Family C is not. It describes only the current page: entropy, fraction of zeros, whether it
looks like text.

So Family C channels measure **state** while the others measure **change**. Position and
velocity are mixed into the same tensor. This is acceptable if deliberate, but fusing a
Family C channel into a complex number alongside a Family A channel combines a quantity with
its own derivative, and the result will not mean what it appears to mean.

### 2.2 The tensor is sparse, not dense

The differ short-circuits pages that did not change at all:

> "Identical page = no change. Every channel is 0; skip all the work (unchanged pages are
> the bulk of a real dump)."
> `[traced: live_delta_calc_modular/src/metrics/mod.rs:53]`

and the sparse output mode is documented as

> "~20x smaller at 5% activity, no info lost (absent page = unchanged)."
> `[traced: live_delta_calc_modular/src/main.rs:34]`

That 20x figure implies a working assumption of roughly **5 percent of pages changing per
frame**. `[inferred: the 20x is stated as an example at 5% activity, not as a measured
corpus-wide average. Real activity per workload is unmeasured and will vary widely.]`

Two consequences, and they cut in opposite directions.

**In favour:** every storage estimate in this document falls by roughly 20x. The analysis is
far more tractable than a dense reading suggests.

**Against:** a tile is mostly empty. At 5 percent activity a 32 MB block holds about 409
changed pages out of 8192 in any given frame. Feeding a mostly-zero field into a wavelet or a
Fourier transform is a different proposition from feeding it a dense one, and the sparsity
pattern itself may carry more signal than the values do. **This is an open design question,
not a solved one.**

A measurement worth taking early: the real per-workload activity rate. If idle workloads sit
at 0.1 percent and heavy ones at 30 percent, then "fraction of pages active" is itself a
strong feature, and it is also a confound that could separate classes on its own. That is
precisely the shape of the `coverage_ratio` leak from June, so it should be checked before it
is used.

---

## 3. What it looks like, concretely

The metric tensor is `262,144 x 700 x 64`. It is a **multi-channel image**:

- **Height** = 262,144 rows, one per page. The Y axis: memory address.
- **Width** = 700 columns, one per snapshot. The X axis: time.
- **Depth** = 64 channels, one per metric.

An ordinary photograph is roughly 1080 x 1920 x 3. This is far taller than wide, a ratio of
about 374:1, with 64 channels instead of 3. And per section 2.2 it is mostly zeros.

The aspect ratio has a practical consequence. There are a great many rows and comparatively
few columns, so any tiling scheme must be asymmetric: the address axis can afford many cuts,
the time axis supports only a handful of windows at large window sizes.

---

## 4. The general block

A tile is a rectangle cut out of that image:

```
Tile(k, w) = S[ k*dp : k*dp + Bp ,  w*dt : w*dt + Bt , : ]
```

Cut along **time** into overlapping windows (batching over the X axis). Cut along **address**
into blocks. Each resulting rectangle is one sample, and features are extracted from it.

This is not a new construction. It is what the original thesis already did, per Entry 1:
"assemble a page-by-time matrix, tile it into overlapping blocks, push each block through the
four lenses." This document rebuilds that structure on much better data.

**Block size is not yet chosen.** The 32 MB figure used throughout this document is an
example for arithmetic, not a decision. It deserves the same treatment Entry 1 gives page
size and chunk size: bracket it between two named failure modes rather than deriving it from
a formula. Too small and a block holds too few changed pages to have any shape at all (see
2.2); too large and distinct behaviours average together. The bracketing cannot be completed
until the real activity rate is measured. `[proposed]`

---

## 5. Why blocks must be whole pages

After the metric reduction the signal is only **defined** at page granularity. There is no
value at "page 5.5", because every value was produced by an operator that consumed an entire
4096-byte page. The domain along the address direction is the integer lattice 0 .. 262,143,
not a continuous line.

Any window along that axis must therefore be a whole number of lattice points. In byte terms:

```
block size must satisfy  B_bytes mod 4096 == 0
```

A 32 MB block gives exactly 8192 pages. This holds for free at any whole-megabyte size, since
1 MB is exactly 256 pages. The constraint only becomes a live risk for block sizes not
expressed in clean multiples.

---

## 6. Sample counts

### 6.1 The window is not settled, and the old answer does not transfer

An earlier draft of this document put a 128-frame window beside Plan 03's 8/4 recommendation
as though they were competing results. They were never in competition.

- Plan 03 swept **windows {8, 16, 32, 64}** and hops {2, 4, 8, 16, 32, 64}.
  **128 was never tested.** `[traced: plan03_sweep.py:275 default `[8,16,32,64]`, confirmed
  against the distinct values present in plan05_campaign/downstream/sweep.csv]`
- The `windowSize: 128` in `config_qemu_upc.json` governs the streaming metrics, a different
  subsystem, and was never validated against that sweep. `[traced]`
- W=8 won partly on a tiebreak: the recorded rationale is *"smallest W tiebreak"*.
  `[traced: plan05_campaign/downstream/recommendation.json]`
- The sweep's ceiling of 64 was almost certainly imposed by trace length. On the old short
  cells a 64-frame window barely fits and a 128-frame window cannot fit at all. `[inferred]`
- Most importantly, **Plan 03 tuned on the APF trajectory**: a single scalar channel, the
  mean fraction of pages changed. `[traced: plan03_sweep.py operates on the APF series]`

So 8/4 is the best window **for one averaged scalar on short traces**. It is not a property
of memory, and there is no evidence either for or against 128.

**Working position: hold 8/4 as the inherited default, explicitly labelled as APF-derived,
and re-sweep now that 700-frame traces make the larger windows testable for the first time.**

### 6.2 What each window size buys on a 700-frame trace

| Window | Hop | Windows per trace | Frequency bins | Time span at ~2.6 s/frame |
|---:|---:|---:|---:|---:|
| 8 | 4 | 174 | 4 | 21 s |
| 16 | 8 | 86 | 8 | 42 s |
| 32 | 16 | 42 | 16 | 83 s |
| 64 | 32 | 20 | 32 | 166 s |
| 128 | 64 | 9 | 64 | 333 s |
| 256 | 128 | 4 | 128 | 666 s |

The tradeoff is stark and it is the classic one. A window of 8 gives 174 samples but only 4
frequency bins, which is almost no spectral resolution at all. A window of 128 gives 64 bins
but only 9 samples per trace. **Sample count and spectral resolution are in direct
opposition**, and the right point on that curve is a per-channel question, not a global one.
That is section 10.

### 6.3 Tile counts

Using 32 MB blocks (32 of them across 1 GiB) as the illustrative case:

| Window | Windows/trace | Tiles/trace | x3 reps | x100 workloads |
|---:|---:|---:|---:|---:|
| 8 | 174 | 5,568 | 16,704 | 1,670,400 |
| 64 | 20 | 640 | 1,920 | 192,000 |
| 128 | 9 | 288 | 864 | 86,400 |

Tile payload at ~5 percent activity, 32 MB block, complex64:

| Window | Dense cells | Active cells (~5%) | Size |
|---:|---:|---:|---:|
| 8 | 65,536 | ~3,276 | ~26 KB |
| 64 | 524,288 | ~26,214 | ~210 KB |
| 128 | 1,048,576 | ~52,428 | ~419 KB |

### 6.4 Sample count is not evidence count

No matter which row above is chosen, the training-sample count does **not** equal the number
of independent observations:

- Adjacent time windows overlap by half. They literally share half their snapshots.
- All tiles from one trace share a VM boot, an allocator state, and a run.
- Spatially adjacent blocks are correlated.

**Effective n for any claim is 100 workloads**, or 300 traces at best.

Practical rule: train on tiles freely, but split train/test **by workload**. Never let two
tiles from the same trace, let alone two overlapping windows, land on opposite sides of the
split. Doing so shows the model its test data during training. This is structurally the same
failure as the June 2026 leakage result, at finer granularity.

Note that this gets *worse* as the window shrinks. At W=8 there are 1.67 million tiles over
the same 100 workloads, which is an inflation factor of about 16,700. A model can look
spectacular on that and mean nothing.

---

## 7. Open problem 1: the angle is not injective

This is Entry 11 in the research diary (index card written, entry not yet written). The
finding is now confirmed in the differ source.

The complex construction in `VMsig_featureExctraction/block_feature_extractor.py`:

```python
def make_complex(mag, phase):
    return mag * np.exp(1j * 2 * np.pi * phase)
```

The phase input is a cosine **distance**. Confirmed in the Rust:

```rust
pub cosine: f32,      // cosine DISTANCE (0 = identical)
```
`[traced: live_delta_calc_modular/src/metrics/family_b/structure.rs:8]`

Multiplying a value in [0, 1] by 2*pi stretches it around a **full turn** of the circle. A
full turn returns to its origin: `exp(i*2*pi*0)` and `exp(i*2*pi*1)` are both 1. Same angle.

Formally: the interval [0,1] has been glued at its endpoints into a circle, the quotient
`[0,1]/(0 ~ 1) = S^1`. That operation destroys exactly one piece of information, the
difference between the minimum and the maximum of the direction measurement.

Those extremes are physically opposite events, and the differ's own comment states the
mechanism:

> "the `distances` crate maps an all-zero side to 1 = max"
> `[traced: family_b/structure.rs:16-18]`

| Event | Previous page | Cosine distance | Resulting angle |
|---|---|---|---|
| Page **written in place** (keeps most content, direction barely moves) | has content | near 0 | 0 |
| Page **freshly allocated** (previous version all zeros) | all zeros | pins at 1 | 0 |

A page being modified and a page being created are indistinguishable in the phase. Per the
Entry 11 index card, most of the data sits exactly at that collision.

Note the code author already knew the all-zero case maps to 1. What was not noticed is what
the subsequent phase stretch does to it.

Unchanged pages are not affected: they emit all-zero channels, so their magnitude is zero and
they contribute nothing regardless of angle. `[traced: metrics/mod.rs:53]`

### 7.1 Why it cannot be fixed downstream

All four lenses read the phase: Fourier, cepstrum, complex wavelets, PLV. A degenerate phase
encoding is inherited by every one of them. The fix must land before extraction.

### 7.2 Candidate fixes

`[derived]` The problem in section 7 is JK's (Entry 11). The two fixes below were
introduced by the assistant and have not been evaluated against each other on real data.

- **Half turn.** Use `pi * distance` rather than `2*pi * distance`. Range maps to [0, pi];
  the two extremes become antipodal, maximally distinguishable rather than identical.
- **Stop rescaling.** Cosine *similarity* runs [-1, 1], and `arccos` of it gives the genuine
  geometric angle between the two page-vectors, in [0, pi]. It is already an angle.

A third option not yet explored: keep magnitude and direction as two real channels and do not
fuse them at all. That is exactly the ablation Entry 2 records as never having been run, so
it may be the honest control arm rather than a fallback.

Implementation note: the differ stores the **distance**, not the similarity. Recovering the
true angle therefore requires either a change in the Rust or a documented inverse applied at
extraction time. Whichever is chosen must be recorded in the run metadata, because it cannot
be reconstructed from the output afterwards.

### 7.3 Bonus

Entry 2 records that the thesis never ran the ablation comparing the full complex
representation against magnitude alone. Once the phase carries injective information that
comparison becomes meaningful, and it is the single most consequential untested decision in
the project.

---

## 8. Open problem 2: fixed addresses versus translation invariance

Entry 1 states the principle: "Where it happened is noise. What kind is signal." Addresses
are close to arbitrary. The same program on the same input can dirty entirely different
addresses depending on allocator behaviour. This is why the original work reached for wavelet
scattering, which provides translation invariance.

Address-blocking is in tension with that principle. Cutting at fixed offsets (block 0 =
addresses 0 to 32 MB, block 1 = 32 to 64 MB, ...) means "block 5" in repetition 1 and "block
5" in repetition 2 may contain unrelated structures. A model permitted to see block identity
learns the allocator's habits, not the workload's behaviour.

### 8.1 Resolution

`[derived]` The tension above follows from Entry 1's stated principle; the
resolution below is the assistant's suggestion.

Treat the blocks as an **unordered collection**. Pool across them order-independently: mean,
max, or attention over blocks. Never feed block index into the model as a feature.

Page order **within** a tile stays, and is necessary: Family D texture, run lengths and GLCM
statistics all depend on local spatial arrangement. What must not matter is which absolute
block a tile came from.

### 8.2 A free test, now confirmed viable

The orchestrator brings each cell up with `virsh start`, a cold boot, not a snapshot restore.
`[traced: run_files_controlled.py:269 ensure_vm_running]` So the 3 repetitions of each
workload genuinely get independently randomized guest layouts, and the invariance test is
meaningful rather than trivially passing.

If per-tile features are stable across repetitions, the representation is
translation-invariant in practice. If not, it is measuring the allocator. This is a cheap
direct measurement and belongs in the pipeline as a gate, not as an afterthought.

Caveat: booting the same disk image lays out kernel memory similarly each time, so
independence is stronger in userspace regions than in kernel regions. `[inferred]`

---

## 9. On-demand materialization

Generate features on demand, feed the model, delete the intermediates. Correct, with one
correction about which stage is throwaway.

Cost profile of the three stages:

| Stage | Cost |
|---|---|
| Reconstruct zstd chain + run differ (L0 -> L1) | Expensive. Dominates. `[inferred, not measured]` |
| Tile and transform (L1 -> L2) | Moderate |
| Resulting features | Tiny |

Cached feature size for the whole corpus, at 86,400 tiles (W=128 case):

| Features kept per tile | Total corpus |
|---:|---:|
| 16 | 5.5 MB |
| 32 | 11.1 MB |
| 64 | 22.1 MB |

At W=8 the tile count is roughly 19x higher, so scale accordingly: still only a few hundred
MB in the worst case.

**Architecture: one extraction pass, write the small feature tensors, delete the giant
intermediates. Then train unlimited epochs against a file that fits comfortably on a
laptop.**

Correction: do not regenerate on demand **per training epoch**. That would re-reconstruct 300
compressed chains every epoch, hours of work repeated. "Ephemeral" applies to the intermediate
representation, not to the extracted features.

### 9.1 Consequence: the pass is one-shot

The pass is expensive and runs over data on the server. Metric selection, tile geometry and
phase mapping are all frozen the moment it runs. Changing any of them means redoing it.

This is why the angle problem (section 7) must be settled before the pass, not discovered
during analysis afterwards.

---

## 10. Per-channel window selection

This section is the main open research direction.

**Authorship note.** The research question in this section is JK's, stated as: "what if each
metric could require a different window? Isn't that what we want to look into, and research?
For now we can hold 8/4 as the window/hop." Section 10.1 expands that question.

Sections 10.2 through 10.4 are `[derived]`: mechanisms for answering that question, worked
out from it rather than stated by JK. They are one possible approach among several and are
open to being cut or replaced wholesale. Alternatives not yet evaluated include a plain
per-channel sweep with no derivation, selection by downstream task performance, or
spectral-entropy-based criteria.

Sections 10.5 and 10.6 build on JK's observation that the C1-C7 gate architecture is the
project's strength and that things can be changed and tested against it.

### 10.1 The problem with one global window

Section 6.1 established that 8/4 was tuned on the APF trajectory: one scalar, the mean
fraction of pages changed. There is no reason for that answer to transfer to 64 channels
measuring physically different things.

Different channels plausibly have different temporal character. The following three examples
are `[derived, unmeasured]` illustrations of the shape the differences might take,
offered to make the argument concrete. They are predictions, not observations, and the
measurement in 10.2 would confirm or refute them:

- **Bit churn** (hamming, L0, L1) may fluctuate frame to frame, closer to a noise process
  with short memory.
- **Compressibility and entropy** (ncd, csize_delta, struct_ent_q) may drift slowly, tracking
  the accumulated character of a page rather than an instantaneous edit.
- **Spatial-shift channels** (cross_corr_lag, phase_corr) may be event-like: mostly quiet,
  with occasional large excursions when content actually moves.

None of this has been checked against the corpus. If all 64 channels turn out to share one
timescale, the premise of this section collapses and 8/4 (or whatever a single sweep picks)
is the right answer after all. **That is a real possible outcome and it should be tested
before the apparatus below is built.**

A window that resolves the first will smear the second. A window that resolves the second
will average the first into a constant. **There is no single correct window across 64
channels**, and the sweep that produced 8/4 never had the opportunity to discover that,
because it only ever saw one channel.

### 10.2 A candidate mechanism: derive the window from autocorrelation time

`[derived]` This is one way to answer 10.1's question, introduced by the
assistant. It has not been agreed, tested, or compared against alternatives.

The proposal: the appropriate analysis window for a channel is set by that channel's own
**autocorrelation time**, the timescale over which its values stop resembling themselves.

For each metric `m`, take its trajectory and compute the autocorrelation function
`rho_m(tau)`. Extract a decorrelation time `tau_m` by either:

- the **first zero crossing** of `rho_m`, or
- the **integral timescale**, `tau_m = sum over tau of rho_m(tau)` up to the first zero.

Then set the window proportional to that timescale, `W_m ~ k * tau_m`, with the same constant
`k` for every channel so the choice is principled rather than per-channel tuning.

The intuition: a window shorter than `tau_m` sees only a fragment of one coherent event and
cannot characterize it. A window much longer than `tau_m` averages many independent events
into a mean and loses the structure. The window should be a small multiple of the memory
length of the signal it is applied to.

### 10.3 The argument for 10.2 over a plain sweep

`[derived]` The case below is the assistant's, and it is an argument, not a
result. A plain per-channel sweep remains a perfectly defensible alternative and is cheaper.

A sweep produces "we tried six values and this one scored best." A derivation produces "this
channel has a memory of 12 frames, therefore its window is 24, and here is the prediction
before we measured."

Crucially, the derivation is **falsifiable against the sweep**. The test:

> Does the measured `tau_m` predict which window wins the C7 gate for channel `m`?

If yes, window selection has been converted from a search into a measurement, and every
future channel gets its window without a sweep. If no, that is also a real result: it means
the relevant timescale is not the autocorrelation time, and the next question is what it is.

Either outcome is publishable. A sweep alone is not.

This also retrospectively explains the APF result. APF is a mean over 262,144 pages, and
averaging that aggressively destroys temporal memory. A short decorrelation time, and
therefore a short optimal window, is exactly what the theory predicts for it. `[inferred,
testable]`

### 10.4 The limit of the idea is the wavelet

`[derived]` Wavelets were not raised by JK in this discussion; the connection
below is the assistant's. It is included because the repo already contains wavelet and
scattering machinery, so the option is cheap to reach for.

If every channel wants its own window, the natural next question is why commit to a window at
all.

A wavelet transform **is** a window sweep folded into the transform: it examines every scale
simultaneously, giving coarse structure and fine detail together. The scattering transform
goes further, adding the translation invariance that section 8 argues is required because
addresses are arbitrary.

So there are two coherent designs, and they are complementary rather than rival:

| Design | What it gives | What it costs |
|---|---|---|
| Per-channel fixed window from `tau_m`, then Fourier or cepstrum | Interpretable, cheap, yields a defensible table, and makes a falsifiable prediction | Commits to one scale per channel |
| Multi-resolution (wavelet, scattering) | Dissolves the choice entirely; translation invariant | Heavier, less interpretable, more parameters |

Entry 1's stated habit already anticipates this: "never commit to one lens, instrument all of
them, let the comparison be the contribution." What is new is that the original comparison ran
on **one channel at one window**. Running it per-channel with per-channel windows is a far
richer version of the same results table, and it is the natural centerpiece of the analysis.

### 10.5 Relation to the gate architecture

The gates are explained in full in section 11. The point relevant here is that **C3's
pass/fail depends on which window you choose**, which makes it unusable as a neutral referee
for choosing one. See 11.4.

Under per-channel windows C3 would become per-channel too: a trace may be long enough for
the fast channels and too short for the slow ones. That is a more honest gate than a single
global pass or fail, but it has to be constructed carefully to avoid the circularity in
11.4. `[derived]`

### 10.6 A caution carried forward

The gate architecture is the strength of this project, but the record shows it drifting under
pressure. In the June 2026 work, thresholds moved after seeing where the data landed (D-80,
D-83, D-88), one gate was hardcoded to `True` while being reported as passing, and C3 was
demoted to informational when it failed everything.

Gates are a strength only while the discipline holds. The concrete implication for this
pipeline: **fix thresholds on a held-out split before the full pass runs**, and record any
later change as a fitted parameter rather than as a recalibration.

---

## 11. The gate architecture

### 11.1 What a gate is, and where it runs

A gate is a named, machine-checked claim about one captured cell, evaluated after the cell
has run and recorded in the cell's validation record as `{pass, why, operational}`. The `why`
string carries the numbers behind the verdict, so a failure is self-explaining rather than a
bare False.

Two properties make this architecture worth copying rather than admiring:

1. **A gate names the claim it is checking**, not the code path it exercises. "The workload
   ran" is a claim about the experiment; "the subprocess returned 0" would be a claim about
   plumbing. The former catches the 90-idle-cells failure of Plan 02; the latter does not.
2. **`operational` separates blocking from informational.** A gate can be computed and
   reported without being allowed to stop the pipeline.

Crucially, **C1 through C8 are capture-side gates.** They validate that a recording is sound.
They say nothing about whether an analysis run on that recording is sound. The analysis side
currently has no equivalent, and building one is the substance of mirroring the capture
methodology. See 11.5.

### 11.2 The roster, from source

All definitions `[traced: plan02_validate_session.py]`, line numbers in brackets.

| Gate | Actual test | Threshold | Blocking? |
|---|---|---|---|
| **C1** workload_ran | count of `PHASE` markers in `workload_stderr.log` > 0; warmup cells auto-pass | > 0 markers [316-323] | yes |
| **C2** ratio_healthy | `snap_completion_ratio` >= threshold | **0.08** when `keep_dumps`, else `min_ratio` (default 0.85) [336-342] | yes |
| **C3** enough_windows | `(n_snaps - W)//H + 1 >= min_windows` | **>= 50 windows**, computed at W=128 H=64 by default [125-128, 344-348] | **no** |
| **C4** no_settle_retries | `lock_retries == 0`; NA for pre-D-34 cells | == 0 [356-360] | yes |
| **C5** producer_log_clean | error count in `producer.log` | == 0 [364-365] | yes |
| **C6** apf_complete | `apf_trajectory.jsonl` exists, carries a `{"final": true}` sentinel, and `n_ok / n_pairs_expected >= 0.95` | **0.95** [246-283] | conditional (NA if no trajectory file) |
| **C7** window_hop_recommended | every entry in `plan03_recommendation.json` has `passes_acceptance == True` | all-must-pass [131-139] | conditional (NA if artifact absent) |
| **C8** segmenter_quality | the Plan 04 gate for this cell's family | per-family [399-424] | conditional (NA if workload/family not evaluated) |

The overall verdict is:

```python
ok = c1 and c2 and c4 and c5 and c6 and c7 and c8
```
`[traced: plan02_validate_session.py:437]`

**C3 is absent from that conjunction.** It is computed, printed in brackets in the report,
and excluded from the pass. That is what `"operational": False` means in practice.

### 11.3 The thresholds carry their own history

Two of the numbers above are not defaults but recalibrations, and the source says so:

- **C2's 0.85 is dead.** The code comment reads: *"MIN_RATIO_DEFAULT (0.85) remains the v1
  'ideal'; unattainable here."* [336] The live threshold is **0.08**, reached in two steps
  (0.15 then 0.08) because the async APF helper competes for disk bandwidth and pushes the
  pause fraction up. The comment records that 0.15 false-failed about 14 percent of cells
  with sound trajectories.
- **C6's 0.95** is a completeness floor on the APF trajectory, not a tuned value.

This is worth stating plainly in a methodology document: the gate architecture is sound, and
two of its thresholds were moved after seeing the data. Both moves are documented with a
stated cause, which is the right way to do it, but they are fitted parameters and should be
described as such rather than as design constants.

### 11.4 Correction: C3 does not start passing at 700 frames

An earlier draft of this document claimed 700-frame traces would let C3 pass for the first
time. **That is wrong at the validator's own defaults**, and the error is instructive.

C3 does not ask "is there room for one window." It asks for **at least 50 windows**
(`MIN_WINDOWS_DEFAULT = 50` [60]), counted at `WINDOW_DEFAULT = 128`, `HOP_DEFAULT = 64`
[61-62]. Working the arithmetic backwards:

| Window / hop | Windows at 700 frames | C3 at 700? | Snapshots needed for 50 windows |
|---:|---:|:---:|---:|
| 128 / 64 (validator default) | 9 | **FAIL** | 3,264 |
| 64 / 32 | 20 | **FAIL** | 1,632 |
| 32 / 16 | 42 | **FAIL** | 816 |
| 16 / 8 | 86 | PASS | 408 |
| 8 / 4 | 174 | PASS | 204 |

`[traced: formula at plan02_validate_session.py:125-128; constants at 59-62. Arithmetic
computed here.]`

So at the defaults C3 needs **3,264 snapshots** and still fails at 700. It passes only at
W <= 16, and 8/4 passes comfortably at 174 windows.

Note also that 128 is not absent from the codebase after all. It is not in the Plan 03 sweep
(section 6.1 stands), but it **is** the validator's assumed window for counting. So the
project already carries two different window conventions in two different files, and they
disagree: Plan 03 recommends 8, the validator counts at 128.

**The circularity this exposes.** C3's verdict is a function of the window under test. A
short window passes the gate by producing many windows; a long window fails it by producing
few. So C3 cannot arbitrate between candidate windows, because it structurally prefers the
short ones. Any per-channel window selection (section 10) must therefore be judged by
something other than C3, or C3 must be restated as a claim about **statistical adequacy**
(enough independent windows for the estimator being used) rather than a raw count. `[derived]`

### 11.5 What the analysis side is missing

Every gate above validates a recording. None validates an analysis. The gaps this document
has surfaced map onto gates that do not exist yet, all `[derived]`:

| Proposed gate | Claim it would check | Drawn from |
|---|---|---|
| A1 no_constant_feature | no extracted feature is constant within a class | the `coverage_ratio` leak, section 2.2 |
| A2 no_label_correlated_missingness | no feature's NaN pattern encodes the label | `cv_workingset` / `f1_phase` in the June audit |
| A3 grouping_respected | train/test split never puts two tiles from one trace on both sides | section 6.4 |
| A4 null_calibrated | detector pass-rate on shuffled and IID surrogates is materially below the real rate | the CUSUM null result, gap 0.000 |
| A5 translation_invariant | per-tile features stable across the 3 repetitions | section 8.2 |
| A6 phase_injective | the phase mapping in use is injective on its domain | section 7 |
| A7 window_adequate | enough independent windows for the estimator, per channel | the restatement of C3 in 11.4 |

The discipline that makes these worth building is the one already proven on the capture side:
a gate must be able to **stop** something. A1 that merely reports a constant feature will be
read past; A1 that refuses to write the feature to disk will not.

---

## Where this leaves us

The reduction ladder with current numbers:

```
2 GB of memory
  -> one page against its past self
  -> two numbers (magnitude, direction)
  -> one complex number per page        <- never measured; everything rests on it
  -> a block, tiled over time and address
  -> one score
```

The third rung is the one the diary records as never having been measured, and it is the rung
with the degenerate angle.

### Decisions gating the first extraction pass

1. **The phase mapping.** Half-turn, or `arccos` of the similarity. Determines whether phase
   carries real information. Must be recorded in run metadata (7.2).
2. **Which metrics to extract.** Freezes what the pass produces.
3. **Whether windows are global or per-channel.** If per-channel, `tau_m` must be measured
   first, which requires a preliminary pass over at least a sample of traces.

### Immediate measurements worth taking

- **Real per-workload activity rate** (2.2). Determines block size, tile occupancy, and
  whether "fraction of pages active" is itself a leaking feature.
- **Autocorrelation time per channel** (10.2). Determines windows and tests the derivation.
- **Cost of reconstruct-plus-differ on one chain** (9). Currently an assumption.

### Open questions not yet settled

- Where the campaign data lives and whether extraction runs on the server or locally.
  (Confirmed 2026-09-01: not on the laptop.)
- What the ML target is: workload identity, selection-family, signature archetype, or
  masquerade detection.
- Per-tile classification with voting, versus pooling tiles into one trace-level vector.
  These support different claims and change what a confusion matrix means.
- Page-axis treatment for the cache: full collapse, blocks, or full page resolution.
  Full collapse discards the axis Family D and Family E depend on.
- Whether an offline batch mode for the differ exists, or must be written. The consumer
  currently invokes it per queue job during live capture.
- Whether the guest's 1 GiB physical range contains holes (MMIO, reserved) that would appear
  as permanent dead rows at fixed positions in every tile.
- Whether physical page identity is stable enough within a single trace to treat a fixed row
  as a coherent time series. The guest OS remaps pages continuously; Entry 1's principle
  covers variance across runs, not drift within one.

---

## Sources

Files read directly for this document, all on branch `fullv5` unless noted:

- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json` (RAM size, page size, window settings)
- `VM_sampler/VM_Capture/live_delta_calc_modular/src/metrics/mod.rs` (64-column schema,
  identical-page short circuit)
- `VM_sampler/VM_Capture/live_delta_calc_modular/src/main.rs` (sparse mode, 20x figure)
- `VM_sampler/VM_Capture/live_delta_calc_modular/src/metrics/family_b/structure.rs`
  (cosine is a distance; all-zero side maps to 1)
- `VM_sampler/VM_Capture_QEMU/plan03_sweep.py` (swept window and hop values)
- `VM_sampler/VM_Capture_QEMU/plan05_campaign/downstream/recommendation.json`
  (W=8 H=4 and the "smallest W tiebreak" rationale)
- `VM_sampler/VM_Capture_QEMU/plan02_validate_session.py` (C1-C8 gate roster)
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` (ensure_vm_running, cold boot)
- `VM_sampler/VM_Capture_QEMU/docs/feature_substrate_spec.md` (families A-H)
- `VMsig_featureExctraction/block_feature_extractor.py` (make_complex)
- `docs/research-diary/` index, Entry 1, Entry 2

Arithmetic in sections 3, 6.2, 6.3 and 9 was computed for this document and is reproducible
from the stated inputs.
