# Companion critique: review of the two analysis-methodology documents

Reviewed: `docs/ANALYSIS_PIPELINE_METHODOLOGY.md` (771 lines) and `docs/METHODOLOGY_AS_EXECUTED.md`
(564 lines), pinned at commit `a4fd3ea` on `fullv5`.

This is a review gate, not a summary. It exists because those two documents were written quickly
and merged before anyone had checked them against the code. Read it beside them.

**Method and its limits.** Claims below were checked by opening the referenced source files, and
are marked CONFIRMED where the code was read at the cited location. `~/Desktop/plan/` exists and
contains the doctoral-plan sources, but was **not** read during this review, so every objective
mapping (O1 through O8, M4 through M6, R1 through R6) in either document is UNVERIFIED here.

---

## Verdict

Do not accept these as the settled basis for the extraction pass yet.

Two of their arguments are correct and settle real decisions: the phase-collision finding, and the
argument that C3 cannot arbitrate window choice because its verdict is a function of the window
under test. Keep both.

But the centrepiece proposal, deriving each channel's window from that channel's own
autocorrelation time, is written against 64 channels when the captured corpus contains 51, and two
of its three worked examples are channels that are identically zero in the captured data. And every
sample count, payload size, and cache budget in the forward-looking document scales off an assumed
corpus that the companion reconciliation document never establishes.

---

## Findings neither document contains

**1. The captured corpus has 51 metric channels, not 64.** CONFIRMED.
`config_qemu_upc.json` sets `substrateSpeed: 2`. `capture_consumer_qemu.sh` reads it and passes
`--speed` to the differ. `metrics/mod.rs` documents that speed 2 drops `lz_change` plus the heavy
twelve (FFT spatial, autocorrelation, high-frequency, four GLCM, Kendall, bigram, `ncd`). The
dropped thirteen still emit zeros into an unchanged 64-column schema.

This lands on the proposal's centrepiece. Of the three channel archetypes offered to motivate
per-channel windows, `ncd` is in the dropped set, and the spatial-shift channels
(`cross_corr_lag`, `phase_corr`) are explicitly zeroed at speed 2. The argument that within-tile
page order must be preserved for GLCM texture statistics is weakened for the same reason.

**2. The cost the proposal calls unmeasured is already measured, in a file it cites.** CONFIRMED.
The proposal marks reconstruct-plus-differ as inferred and lists measuring it as future work.
`main.rs` carries a measured table: about 39 s/pair at speed 0, 5.5 s at speed 1, 2.1 s at speed 2,
1.5 s at 3, 1.2 s at 4, on a 1 GB dump pair at roughly 5% activity. At a few hundred pairs per
trace that is tens of minutes per trace at speed 2 and hours per trace at speed 0. This converts
the document's architectural preference into a hard constraint.

**3. The phase fix touches about thirty sites, not one.** CONFIRMED.
The collision itself is real and correctly argued: `family_b/structure.rs` declares
`cosine` as a **distance** (0 = identical) and notes that an all-zero side maps to 1, while
`make_complex` in `VMsig_featureExctraction/block_feature_extractor.py` multiplies that value by
`2*pi`. Distance 0 and distance 1 therefore land on the same angle, making a page rewritten in
place indistinguishable from a page freshly allocated from zeros.

But the same `mag * exp(1j * 2 * pi * phase)` expression is written inline roughly thirty more
times in that same file (the max/min/range feature block, several commented-out blocks, and a bare
`2.0 * np.pi`). Patching only the function yields a feature vector mixing two phase conventions,
which is worse than either convention alone.

**4. Gate C7 passes trivially today and is one filename away from failing the whole campaign.**
CONFIRMED, latent. C7 requires every entry of `plan03_recommendation.json` to have
`passes_acceptance == true`. No file of that name exists in the tree, so C7 evaluates
not-applicable and passes. The artifact that would plausibly fill that name,
`plan05_campaign/downstream/recommendation.json`, has 7 of 11 entries at `passes_acceptance: false`
(rationale strings read "best-feasible: passes 4/5 gates (relaxed: G3)"). Copying or renaming that
file into a cells directory flips every cell to `ok = false`. Both documents list C7 as an ordinary
conditional gate; neither notices.

**5. A causal claim about C3 is unsupported.** CONFIRMED as a mischaracterisation.
The proposal cites "C3 was demoted to informational when it failed everything" as evidence that
gate discipline drifts under pressure. The code comment gives a scope reason instead: low window
count is a duration-matrix issue owned by Plan 03, not an orchestration regression. The other two
items in that same sentence (thresholds moved after seeing data, a gate hardcoded true) do check
out.

---

## Where the two documents contradict each other

The second document exists to check the first. In several places it is the less accurate of the two.

1. **C2's non-`keep_dumps` threshold.** The proposal says 0.08 when `keep_dumps`, else the
   `min_ratio` default of 0.85. The reconciliation says 0.08 else 0.30. CONFIRMED: the code reads
   `c2_min_ratio = 0.08 if keep_dumps else min_ratio`, with the default at 0.85. The proposal is
   right. The reconciliation's own conflict table lists all three candidate values and then prints
   the wrong one.

2. **The corpus.** The proposal computes everything from 100 workloads x 3 repetitions x about 700
   snapshots. The reconciliation's entire executed record is 11 workloads and 66 cells
   (`summary.json` reads `n_cells_used: 66`). Neither document reconciles the two. This is the
   clearest place where the reconciliation did not actually happen.

3. **The hardcoded G4.** The proposal reads it as evidence of drifting discipline. The
   reconciliation shows G4 sits outside the acceptance conjunction and so cannot cause a false
   pass. CONFIRMED that the mitigating detail is real; both descriptions of the fact are true and
   the interpretations are opposed.

4. **Why C3 is informational.** See finding 5. The proposal asserts a cause; the reconciliation
   offers none; the code gives a third answer.

5. **C7's semantics.** The proposal presents all-must-pass as traced fact. The reconciliation lists
   C7's pass semantics as an open question never resolved. The code implements all-pass, so both
   can be true, but neither notes the landmine in finding 4.

6. **Substrate family coverage.** The reconciliation lists only families G and H as not started.
   The differ implements only A, B, C, D, so E and F appear in neither the schema nor that
   not-started list. If families E and F are what the analysis layer is meant to build, the
   proposal's scope line should say so.

---

## Weaker points worth correcting, lower priority

- The proposal contradicts itself on RAM size: 1 GiB and 262,144 pages in one section, "2 GB of
  memory" in the closing ladder. The config says 1024 MiB. The page count, the aspect ratio, and
  the blocks-per-GiB figure all ride on the former.
- `windowSize: 128` is attributed to the streaming metrics. In the config that key sits under a
  raw-retention block whose parent is disabled; the streaming block uses a different key. The
  document's point (that 128 was never validated against the sweep) survives; the mechanism stated
  is wrong.
- The Plan 03 sweep is described as sweeping hops {2, 4, 8, 16, 32, 64}. The code sweeps hop
  *ratios* {0.25, 0.50, 1.00} of each window. The realised union is the same set, but it is not a
  free cross product.
- A per-frame timing column is roughly double what the shipped artifacts imply, which inflates
  every time-span figure in the same table.

## What the documents get right and should keep

- The phase-collision argument, in full.
- The C3 circularity argument: a gate whose verdict is a function of the window under test cannot
  arbitrate between windows. The supporting arithmetic is correct in all rows.
- The honesty about sparsity: the 20x figure is flagged as an example at 5% activity rather than a
  corpus measurement, and the warning that "fraction of pages active" could separate classes by
  itself, in the same shape as the `coverage_ratio` leak, is well aimed.
- The retraction on `coverage_ratio`: it looks like a live leak and is not, because it is retained
  deliberately as the contaminated control arm that makes the leak measurable.
- The observation that a gate must name the claim rather than the code path.
- The closing principle: a gate that reports is read past; a gate that refuses to write the file is
  not.
- The arithmetic itself. Every table that was recomputed checked out. The inputs are the problem,
  not the algebra.

## Unverified, and load-bearing

Six doctoral-plan claims rest entirely on `~/Desktop/plan/`, which was not read here. One of them,
a non-invertibility proof, is load-bearing for several sections including a data-protection
argument, and the source decks are reported to be internally inconsistent about whether it is
proved or merely anticipated. Anything thesis-facing or legal that leans on this needs its own
verification pass against those files.

## Priority

**Before anything runs:** re-derive the channel roster against the real `substrateSpeed`; replace
the assumed corpus with the real manifest; lift the differ's measured per-pair cost into the
architecture argument; plan the phase fix as a thirty-site change.

**Can wait:** the C2 threshold, the `windowSize` misattribution, the RAM figure, the hop-sweep
description, the C3-demotion sentence.

**Separately:** read `~/Desktop/plan/` against the six unverified claims.
