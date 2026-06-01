# Paper Portfolio -- VM Memory-Signal Capture Thesis

This directory holds the publication-planning deliverables produced by the
research team (project manager, lead researcher, two IEEE reviewers, and the
engineering group) from the Phase-2 v3 results. The intent is to turn the
thesis results into one or more papers. Everything here is a **skeleton**:
introductions, contributions, and results are written as bullets and are meant
to be polished into prose later.

## Contents

| File | What it is |
|---|---|
| `index.html` | Portfolio overview, executive summary, results-at-a-glance, paper roadmap. Start here. |
| `research_log.html` | The team's "month" of analysis: kickoff, data audit, paper-conception debate, reviewer critiques, revisions. |
| `reviewer_memo.html` | The two IEEE reviewers' independent critiques and the team's responses. |
| `synthesis_essay.tex` | Long-form essay-style analysis of the v3 results (the umbrella PDF). |
| `refs.bib` | Shared starter bibliography (most entries are TODO placeholders to verify). |
| `paperA_apf_capture/skeleton.tex` | Paper A -- the Active Page Fraction capture method and cost model. |
| `paperB_gated_tuning/skeleton.tex` | Paper B -- gated grid search for sampling-interval and analysis-window selection. |
| `paperC_family_classification/skeleton.tex` | Paper C -- workload-family discrimination from a single-scalar liveness trajectory. |
| `paperD_cusum_segmentation/skeleton.tex` | Paper D -- family-aware CUSUM segmentation of coarse liveness trajectories. |
| `paperE_timing_instrumentation/skeleton.tex` | Paper E -- timing instrumentation and the snapshot-interval inertness finding (the measurement/methods paper). |
| `paperF_streaming_capture/skeleton.tex` | Paper F -- the B+3.1 streaming, delete-as-you-go whole-VM snapshot-capture system (the systems paper). |
| `paperG_workload_design/skeleton.tex` | Paper G -- the reversible-XOR synthetic-workload taxonomy (11 workloads, phasic/steady, the safety model). |
| `paperH_offline_spectral/skeleton.tex` | Paper H -- the deferred high-fidelity offline 2-D per-page spectral/stability pipeline (PLV/MSC/cepstrum), traded away for APF. |

**Companion skeletons E-H** were added to reflect the full experiment record (timing
instrumentation, the capture system, the workload design, and the deferred 2-D pipeline).
Scope and venue are deliberately deferred; they capture work the four headline papers
(A-D) do not, and several may later merge into A-D rather than stand alone.

## How to build the PDFs

There is no LaTeX toolchain in this checkout. On a machine with TeX Live:

```bash
# essay (uses base 'article' class, compiles with a minimal TeX install)
pdflatex synthesis_essay.tex

# each paper skeleton targets IEEEtran (conference). If IEEEtran is missing,
# change \documentclass to 'article' (a comment in each file shows the swap).
cd paperA_apf_capture && pdflatex skeleton.tex
```

## Honesty notes baked into every skeleton

Each skeleton states its limitation alongside its claim, because the v3 results
carry real tension that the team agreed must not be hidden:

- The effective snapshot cadence is **variable, median ~10 s/snapshot (range
  2-24 s)**, dominated by `pmemsave` itself (median ~7 s, ~81% of the cycle),
  not the requested 500 ms guest interval (the guest idles ~0.52 s between
  snapshots). Trajectories run from a handful to ~100 points (median ~30); the
  old "7-28 points" and "15-21 s" figures were near the per-duration minima.
- Workload-family separation *looks* perfect at the **binary** level (phasic vs.
  steady, 100% with a RandomForest under leave-one-replicate-out), but this is
  **feature-construction leakage**: `coverage_ratio` alone -- a per-family
  constant (0.133 steady / 0.200 phasic) -- reproduces the 100%. With the three
  family-conditional features removed the genuine signal is weak: 0.644 vs. a
  0.542 majority baseline -- and that gap is **not significant** once the four
  non-independent replicates are accounted for (~33 independent groups). The
  **11-way** instance problem is only ~33% accurate;
  APF does not separate instances *within* a family. (See `reviewer_memo.html`
  and `paperC_family_classification/leakage_ablation.json`.)
- Phasic periodicity is **detectable** (cepstral SNR 4.8-6.3 dB) but
  **under-sampled** at the achievable cadence (spectral-coverage gate G2 fails),
  so Plan 03 awards only 5 of 11 workloads a clean acceptance.
- The CUSUM segmenter validates phasic cells by boundary-count **plausibility**,
  not by marker-aligned F1, because the workload phase markers are not
  observable as APF mean shifts at this cadence.

These are not failures to bury; several are the most publishable findings.
