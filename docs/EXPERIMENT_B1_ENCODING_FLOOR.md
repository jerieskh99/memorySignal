# Experiment B1: the encoding floor

Status: **proposal and pre-registration. Nothing has been run.** Written 2026-09-03.

**The spine.** Weighted APF is the same reduction as APF, with one substitution: where APF counts
a changed page as 1, weighted APF counts it by how much it changed. It needs no new measurement —
per-page Hamming, the "how much", is a channel the differ already computes at every speed. The one
trap is that weighted APF also has an exact reading as APF multiplied by a second quantity, and
that product form is lossy in a way the sum form is not. The experiment is built around keeping
those two readings from being confused.

Around that sits the second reason to run this first: the thesis commits to comparing every
encoding against one fixed baseline, and that comparison has never actually been performed. If
the protocol cannot be made to work on the two simplest encodings in the family, it cannot be
trusted on the complex one. This experiment produces the floor, and the machinery for measuring
anything against it.

Later papers vary one further coordinate each — orientation restored, fusion versus separation,
the analysis lens, the spatial axis, the model family. All of them need this number to open with.

Evidence tags: `[traced]` read from the named file. `[computed]` derived here. `[proposed]`
designed, not built, not run.

---

## The data this needs is not on the laptop, and may or may not be on the server

Start here, because everything below is conditional on it.

Weighted APF is a function of per-page Hamming. So it needs data that retained Hamming per page,
or the raw dump pairs from which Hamming is recomputed. It cannot be built from an APF trajectory,
because that trajectory stored only the per-snapshot scalar `K/N` and threw the per-page detail
away. `[traced: apf_trajectory.jsonl lines are {"seq":n,"apf":f} and nothing else]`

That is exactly what the published corpus is. The 66-cell campaign ran the streaming-APF helper,
which computes `K/N` and then **deletes the previous dump** — the flag is called `keep_dumps` but
the mode does not keep them. `[traced: plan02_apf_helper docstring "deletes the previous dump",
exit 0 = "prev deleted"]` So its per-page Hamming is gone, and **that corpus cannot serve this
experiment.** Opus 5's draft reached the right conclusion here; the reason is the delete in the
helper, not a `retention: metrics` setting.

What can serve it, in order of preference:

- **The Plan 07 zstd chain corpus.** The August campaign kept every raw snapshot as a zstd patch
  chain. Reconstruct a chain, run one pass per pair, both arms out of the same loop. No recapture.
  This is the first thing raw retention pays for. Its existence on the server is the open question
  below.
- **Any substrate capture.** A cell taken with `CAPTURE_METRIC=substrate` already wrote per-page
  Hamming as a CSV column, for every changed page, at every speed. `[traced: main.rs emits
  page_index + the metrics row for pages with hamming != 0; hamming computed before the speed
  branch, mod.rs:51]` For those cells both arms are a **column sum, with no differ run and no new
  code at all** — the changed-page count is the row count, the bit total is the Hamming column.
  Larger artifacts, but the cheapest path if the data exists.
- **A purpose-built subset.** A targeted raw capture at 500 ms, if neither of the above covers the
  workloads the predictions need.
- **Live emission.** Implementing `TIMING_MAGENT`. Correct eventually, unnecessary now, and it
  puts a capture change on the critical path of an offline question.

**No source above is confirmed present on this machine — none were found here.** `[traced: no
*.zst, no page_metrics CSV, only APF trajectories in the tree]` `runs/` holds a single test
registration; the August campaign is attested only in the research diary. So step zero is to walk
the retention tree on the server and emit a real manifest. Until that exists, every count in this
document is a placeholder, and two of the five predictions below cannot be scheduled because their
workloads may not be in the tree.

---

## Weighting is APF with the count replaced by the amount

Write `N` for pages in a dump, `K` for changed pages, `h_i` for bits flipped in page `i`, and
`BITS_P = 32768` bits per page (4096 bytes × 8).

Both encodings are the same sum over *all* pages. They differ only in what each page contributes:

```
APF   =  (1/N) Σ_i  [ h_i > 0 ]        # indicator: changed = 1, unchanged = 0
wAPF  =  (1/N) Σ_i  ( h_i / BITS_P )   # weight: how much changed, unchanged = 0
```
`[computed, verified against plan02_apf_helper on the apf column]`

An unchanged page has `h_i = 0` and contributes nothing to either — the differ's own
short-circuit. `[traced: mod.rs:51, identical page returns all-zero]` A changed page contributes
`1` to APF and its graded amount to wAPF. That is the entire difference: swap the count for the
amount. It makes E0 and E1 two members of one operator family under different magnitude
derivatives — the M1 claim, shown by the arithmetic rather than asserted.

**A free consequence, and a sanity check: `wAPF ≤ APF`, always.** A changed page contributes at
most `1` to wAPF, reached only when every bit flipped (`h_i = BITS_P`, a full rewrite), and less
when lightly touched. So `wAPF = APF` exactly when every changed page was fully rewritten, and the
gap `APF − wAPF` *is* the lightly-touched mass — the signal the depth axis exists to expose. A
capture that ever reports `wAPF > APF` has a bug. `[computed]`

The same quantity also factors as a product, and this reading is where the care is needed:

```
wAPF  =  Σh_i / (N · BITS_P)  =  (K/N) · (Σh_i / (K · BITS_P))  =  APF · mean-intensity-on-changed
```
`[computed]`

The product is exact but lossy. `0.5 × 0.02` and `0.02 × 0.5` are the same number and opposite
behaviours: many pages barely touched, versus a handful rewritten end to end. The second is the
stealth-encryptor shape the magnitude axis was introduced to catch. So a null result on wAPF is
ambiguous by itself — it could mean depth carries nothing, or that collapsing it into one number
threw it away. The sum form has no such ambiguity; it is the honest definition. The product is
kept only to name why a third arm is needed.

So three arms. All are members of the same operator family — a magnitude derivative per page,
orientation discarded, reduced to a scalar per sample — differing only in the magnitude derivative
and in whether the reduction is one number or a pair.

| Arm | Per-sample value | What it measures |
|---|---|---|
| **E0** APF, the floor | `(1/N) Σ [h_i > 0]` | breadth: how many pages moved |
| **E1** wAPF | `(1/N) Σ (h_i / BITS_P)` | breadth and depth, as one number |
| **E2** APF ⊗ intensity | the pair `(K/N, (Σh_i/K)/BITS_P)` | breadth and depth, kept apart |

E1 stays in despite E2, because the honest floor — and the claim a paper can make — is *one scalar
per snapshot*. E1 keeps that property and E2 does not. If E2 wins where E1 does not, the finding is
that the one-number reduction is where the loss happens, sharper than either arm gives alone.

This is the ablation the 2025 thesis never ran — full representation versus its parts — moved onto
the magnitude axis, where it costs one extra channel instead of a rebuild.

**Everything else is pinned across arms:** the same cells, the same `seq` set per cell, 500 ms
sampling, W=8 and H=4, one feature function parameterised by arm, one learner with fixed
hyperparameters, one set of folds, one recorded seed.

W=8/H=4 is inherited, not chosen. It was tuned on the APF trajectory by a sweep that never tested
128 and won partly on a smallest-window tiebreak `[traced]`. That favours no arm deliberately,
but it is not neutral either: it is fitted to one arm's statistics. The headline runs at 8/4, and
a single larger window is checked afterwards to bound how much that matters. It is not re-swept
here.

---

## The model is deliberately dull, because eleven workloads cannot feed a clever one

Budget before ambition. The effective sample size is the number of **workloads**, not cells —
overlapping windows and repeated runs of one workload are not independent observations. On this
project's own rule, anything with more than one free knob is over-parameterised.

| | Model | Free knobs | Role |
|---|---|---|---|
| **L0** | majority class | none | the floor under the floor |
| **L1** | single-feature threshold, feature picked on the training fold only | none at test | is the learning earning its place? |
| **L2** | logistic regression, L2, `C = 1.0` fixed | one, fixed in advance | **primary** |
| **L3** | `RandomForestClassifier(n_estimators=300)` | many | incumbent, comparability only |

L2 is the headline. L3 exists so the numbers line up with the record's existing figures, carrying
the standing caveat that it is over-parameterised here `[traced: peakvar_lift.py:44]`. L1 exists
because a hand-written rule already matched the model once in this project, 0.95 against 0.95 —
and if that happens again, that is the result.

**Features, eight per channel:** mean, standard deviation, coefficient of variation, median, max,
95th percentile, max-over-median, and duty cycle.

Duty cycle needs care. The existing implementation thresholds at an absolute 0.05, calibrated to
APF's scale `[traced: extra_features.py:56]`. Bit-fraction lives on a completely different scale,
so an absolute threshold silently hands the comparison to one arm. It is redefined against each
cell's own median. Any feature that cannot be defined scale-equivariantly is dropped rather than
adapted — that rule is a gate, not a preference.

Excluded on purpose: `n_pairs` and `n_windows`, which are capture-side properties identical
across arms and correlated with cell duration, a design variable — the same route that produced
the `coverage_ratio` leak; and the analyzer outputs `stat_pass_frac`, `cepstral_peak_idx`,
`ceps_peak_snr_db`, which are out of scope for a floor experiment.

E2 gets the eight features on each of its two channels, so sixteen against eight. That is a
capacity difference, not only an encoding difference. Two mitigations, both applied: identical
regularisation across arms, and a fourth arm **E2′** carrying only the intensity channel's eight,
so a like-for-like comparison exists. If E2 wins only when it has twice the features, the paper
says that.

**Tasks:** binary phasic-versus-steady (T1), five-class behavioural family (T2), workload identity
(T3, one protocol only — leaving a workload out is undefined for it).

**Protocols:** leave-one-replicate-out is reported and **never headline**, because it trains on
another run of the same workload and therefore measures memorisation. Leave-one-workload-out is
the headline for T1 and T2. Leave-one-family-out is the headline for generalisation. A one-class
novelty model trained on benign only, with the whole threat family unseen, is the headline for
detection. No pooled score across families is reported, on principle.

---

## Every gate here can refuse; none of them merely reports

Named before any result is seen. The lesson this encodes is old and expensive: a gate that
reports gets read past, and a gate that cannot fail on bad input cannot be believed on good.

| Gate | The claim | On failure |
|---|---|---|
| **B1-G1** null calibration | the score is distinguishable from chance: it must beat its own label-shuffled surrogate on the same folds, by at least the surrogate's 95th percentile over 200 shuffles | not a result; written to the artifact flagged `near_unfalsifiable` and excluded from every table |
| **B1-G2** grouping | no workload spans a leave-one-workload-out split, no family spans a family split, no two cells of one replicate span a fold | refuse to run |
| **B1-G3** no feature is the label | every feature varies within every class, and no single feature reproduces full-model accuracy to within 0.02 | quarantine it, name it, re-run without it |
| **B1-G4** arm parity | identical cells, identical `seq` set per cell, identical folds across arms; a cell missing from one arm is dropped from all | refuse to run |
| **B1-G5** scale neutrality | every feature is scale-equivariant, or defined by one rule applied per arm; no absolute constant fitted on one arm | refuse to run |
| **B1-G6** floor sanity | the primary model beats the trivial baselines on the headline task | reported as a negative result, never silently omitted |

Parity (G4) is satisfied by construction rather than by inspection: both arms are read off the
same source per snapshot. There are two sources, depending on what the manifest turns up, and both
already contain per-page Hamming — no new capture primitive is written for either.

**Substrate CSV (column sum, no differ run).** The file already lists every changed page with its
Hamming. Then, with `N = 262144` fixed by config:

```
K = row_count;  ham_sum = Σ hamming_column
emit { seq, n_pages: N, n_changed: K,
       apf: K/N, ham_sum, wapf: ham_sum/(N*BITS_P),
       ham_mean: (ham_sum/K if K else 0.0) }
```

**Zstd chain (reconstruct, one pass).** Same arithmetic, Hamming recomputed from the pair:

```
for i in 0..N:
    if pages_equal(prev_i, curr_i): continue     # unchanged -> contributes 0 to both
    K += 1;  ham_sum += popcount(prev_i XOR curr_i)
# identical emit; ham_mean = 0 when K = 0
```

All magnitude fields are 0 when `K = 0`, so `ham_mean` is guarded rather than dividing by zero.

Cost. The substrate route is a column sum — trivial. The reconstruct route runs the existing
differ over reconstructed pairs; that cost is unmeasured and must be measured before the run is
sized. For scale, the 51-channel differ is documented at about 2.1 s per pair on a 1 GB pair at
roughly 5% activity, and that figure calls itself a laptop estimate in its own source
`[traced: main.rs:40-48]`. This pass does strictly less — Hamming only, no transforms, no
compression — so it should come in well under. "Should" is not a number; time it on one real chain
first.

**The decision rule, fixed in advance.** For the headline task under leave-one-workload-out, with
the primary model:

```
if neither E0 nor E1 clears the null gate:
    "no encoding separates at this sample size"        # a result about the corpus
elif E1 − E0 ≥ margin and E1 clears the null gate:
    "weighting helps"  -> report the per-workload delta table
elif |E1 − E0| < margin:
    "weighting is not the lever at this granularity"
    if E2 − max(E0, E1) ≥ margin:
        "...and the product is lossy; the factors are not"
else:
    "weighting hurts"  -> investigate; do not discard
```

`margin` is deliberately left unset. It must be filled before the run with a measured quantity:
the spread of the same score across seeds and folds on the E0 arm alone. A bar not anchored to a
measured noise floor is decoration, and this project has retired a gate for exactly that reason.

Every constant above is fixed by this document. Any later change gets recorded as a fitted
parameter, with its date and reason — not described as a recalibration.

---

## Three of the five ways this can end are negative, and all of them are publishable

Predictions written before the measurement, so they can be wrong. Two of this project's earlier
predictions already were, usefully.

**Where weighting should win.** APF is blind to intensity by construction. A workload that
rewrites a few pages completely — the slow-burn shape — sits near the noise floor in APF and
should stand clear of it once weighted. Expect that to surface as *per-workload recall on the
quiet threats under the family-holdout*, not as a jump in headline accuracy. `[proposed]`

**Where it should not matter.** The coarse phasic-versus-steady split is driven by burstiness in
time, which both arms carry equally. Expect little difference. A large win here should be treated
as a scale artifact until proven otherwise.

**The sharpest and most falsifiable one.** A workload that writes a few bytes into each of many
pages registers near-maximum APF and low bit-fraction — the measurement counts pages touched, not
bytes changed, as this project discovered by accident. `mem_writemag_sweep` is the probe built to
vary exactly that axis. So it should be **the workload that moves most between arms, and it should
move in the opposite direction from slow-burn.** If those two move together, the weighting is not
measuring what this document claims it measures.

**On the third arm.** Either outcome pays. If keeping the factors apart beats multiplying them,
the single-scalar reduction is where the information goes, and that finding shapes every later
encoding. If it does not, the product is sufficient and the floor stays one number — a cleaner
story for a baseline paper.

**On the absolute numbers.** Expect the workload-holdout in a 0.60–0.80 band and the family
holdout materially below it, consistent with the record's 0.606 and 0.273 under leakage control.
Anything at or near 1.000 triggers the leak hunt before it is reported as anything at all.

What each ending licenses:

| Outcome | The claim it earns |
|---|---|
| Weighted wins on quiet threats, ties elsewhere | the depth axis is real and specific; APF's blindness becomes a measured limitation instead of an assumed one |
| The arms tie everywhere | breadth already carries what this granularity can carry — a stronger floor than a win, and the bar the complex encoding must clear |
| Only the two-channel arm wins | the reduction to one scalar is the lossy step, not the choice of magnitude derivative |
| Weighting loses | weighting dilutes; the mechanism is named and directly testable against the write-magnitude probe |
| Nothing clears the null gate | sample size, not encoding, is the binding constraint — a result about the corpus that sizes the next campaign |

---

## What this experiment is not allowed to claim

Both arms average the page axis away, so nothing here can speak to *where* in memory anything
happened. A region-and-time claim is structurally unavailable and must not be implied.

Orientation is discarded in all three arms, so nothing here bears on the complex encoding.

The setting is controlled workloads with known ground truth and held-out splits, so nothing here
bears on deployment, where there is neither.

And nothing here establishes that the inherited window suits the weighted arm. It is flagged,
sensitivity-checked, and left alone.

Three checks run *after* the headline is locked, never used to choose it: one larger window, to
bound how much the inherited setting flatters the arm it was tuned on; a second seed for every
fold assignment, which is where the decision rule's margin comes from; and per-workload deltas
for every arm, so no workload can hide inside an average — the guard the record already requires
for the stealth family.

Things that could go wrong, and the answer to each: the data may not cover the workloads the
predictions need, which the step-zero manifest settles before any code is written. Bit-fraction
runs small enough at low activity that float artifacts could read as signal, so the raw bit total
is reported beside it and the distribution is inspected before features are built. The
feature-count asymmetry is handled by the E2′ arm. Entropy is a separate axis and does not come
out of a Hamming-only pass; where a substrate CSV already carries a content-entropy column it is
simply **not read**, and it is **pre-registered out of scope** — if this paper uses it, it says so
in the open.

---

## Build order

Nothing is started.

0. **Manifest the retention tree** on the server. Blocking; everything else is conditional on it.
1. **Time the offline pass** on one real chain. Replaces the estimate above with a number.
2. **`b1_encode.py`** — emit both arms per snapshot as `{seq, n_pages, n_changed, apf, ham_sum,
   wapf, ham_mean}`, from whichever source the manifest found: sum the columns of a substrate CSV,
   or reconstruct a zstd chain and run the Hamming-only pass. Its `apf` column must agree *exactly*
   with the existing helper, or the pass is wrong. That equality is the cheapest correctness check
   available and belongs in a test, not a one-off.
3. **`b1_features.py`** — the eight features, one function parameterised by arm.
4. **`b1_gates.py`** — the six gates, each able to refuse.
5. **`b1_run.py`** — arms × tasks × protocols × learners, one seed, one artifact, gates wired to
   block rather than warn.
6. The three sensitivity checks, last.

**Where it sits in the plan.** Objectives O1 (first empirical comparison of two members of the
encoding family) and O2 (encoding as a configurable choice, compared on one captured trace); O3
and O4 only minimally, since the analyzer is inherited and the learner is chosen to be
uninteresting. Methodology M1 and M5. Expected results: R2's baseline-grounded encoding comparison,
executed for the first time, and R3's floor measured under two encodings instead of one.

---

**The shape of it.** The experiment is small, the finding it is built to survive is that weighting
changes nothing, and the only genuinely novel thing in it is the insistence that a product be
tested against its own factors.

Urgent: the manifest, and the margin. Neither is analysis — both are prerequisites that decide
whether the analysis means anything. Everything else can wait for them.

---

## Realized panel (as captured, 2026-09) and family caveats

Decision (JK): run the initial plan on **all 7 families**. Weak families are expected and
documented here so a low score on them is read as a corpus property, not a method failure. Don't
panic on cache/sandbox.

State after extraction + windowing (8/4, `--min-pairs 50`, 2 aborted ransom stubs dropped):
55 cells, **8971 windows**, 19 workloads, 7 families.

| family | workloads | windows | notes |
|---|---|---|---|
| mem | 4 | 2095 | solid |
| thread | 3 | 1976 | solid |
| cpu | 3 | 1785 | solid |
| app | 3 | 1676 | solid |
| io | 2 | 1253 | solid |
| cache | 3 | 150 | **THIN**: all singletons (1 rep each); two tiny (16, 17 windows) |
| sandbox | 1 | 36 | **SINGLETON**: one workload (stealth_paced); ransom aborted/dropped |

**Split applicability (structural, not a bug):**
- **within-trace** (sanity ceiling): all cells.
- **LORO** (leave-one-rep-out): only workloads with >= 2 reps. Excludes all cache workloads,
  sandbox_stealth, mem_writemag (each 1 rep) -- nothing to hold out.
- **LOWO** (leave-one-workload-out): only families with >= 2 workloads. cache (3 workloads) runs
  but on tiny data -- expect weak. **sandbox (1 workload) cannot** hold out within-family: remove
  its only workload and no sandbox AE can be trained, so its LOWO recall is structurally ~0. It
  therefore doubles as a **novelty case** (does the bank flag it as none-of-the-known?).

**What to remember:** cache and sandbox will likely score low or erratic. That is the corpus
(few workloads/reps), not APF/wAPF or the method. The APF-vs-wAPF comparison is read on the five
solid families; cache and sandbox are reported with this caveat attached, never as the headline.
`b1_splits.py` records which families/workloads actually participated in each split, so the tables
are self-documenting.
