# ROUND 1 — ML engineer, independent analysis

Reproduced the reported result on the laptop npz (AE bank, LOWO APF: app 0.841, mem 0.753, cpu 0.024,
io 0.018, thread 0.010, cache/sandbox 0.0), consistent with the committed `b1_ae_results.json`. Every
number below was run; leaky rows are labeled.

## Data geometry (the setup for everything)

Per-family raw APF window level:

| family | #wl | #cell | #win | mean APF | win std |
|---|---|---|---|---|---|
| mem | 4 | 11 | 2095 | 0.190 | 0.105 |
| app | 3 | 9 | 1676 | 0.0358 | 0.0247 |
| cache | 3 | 3 | 150 | 0.065 | 0.124 |
| cpu | 3 | 11 | 1785 | **0.00080** | 0.0012 |
| io | 2 | 8 | 1253 | **0.00083** | 0.0057 |
| thread | 3 | 12 | 1976 | **0.00068** | 0.0010 |
| sandbox | 1 | 1 | 36 | 0.0016 | 0.0021 |

cpu/io/thread share the same mean-APF neighborhood (~7-8e-4), indistinguishable by level. app/mem are
separated by level alone. That is the whole problem in one column.

## The decisive ablation: same windows, same splits, swap the model

Per-family recall, forest vs the AE bank, identical `b1_splits` folds:

| arm/split | model | cpu | io | thread |
|---|---|---|---|---|
| APF LOWO | AE bank | 0.024 | 0.018 | 0.010 |
| APF LOWO | forest | **0.457** | 0.000 | **0.568** |
| APF LORO (no cell straddle) | forest | 0.836 | 0.309 | 0.736 |
| APF within_trace (leaky ceiling) | forest | 0.848 | 0.408 | 0.826 |
| wAPF within_trace (leaky ceiling) | forest | 0.997 | 0.832 | 0.786 |
| wAPF LOWO | forest | 0.622 | 0.027 | 0.015 |

Holding representation and folds fixed, the forest lifts cpu 0.024->0.457 (19x) and thread
0.010->0.568 (59x) at LOWO. The signal is in the 8-dim APF window; the AE bank was throwing it away.

## Mechanism, confirmed directly

- **AE degeneracy:** for cpu/io/thread windows, the cpu-AE, io-AE, thread-AE reconstruction MSEs are
  near-tied at ~1e-4; argmin is decided by 4th-decimal noise. A synthetic constant window at cpu's own
  level (0.0007) is labeled **thread** by argmin — the quiet families collapse *by construction*.
  mem/app win their own AE decisively (0.03 vs 4.3), which is why only the loud families survive.
- **Standardization:** the global scaler maps cpu/io/thread to z = -0.555/-0.555/-0.557 — on top of
  each other. It doesn't *cause* the collapse (they were already co-located in level); it confirms
  level carries no quiet-trio information. The forest still separates them at the ceiling using
  nonlinear within-window shape.
- **Linear separability:** 3-class logistic on {cpu,io,thread} gets cpu 0.19 / io 0.00 / thread 0.98
  at the ceiling and ~chance at LOWO. The separating structure is **nonlinear** (burstiness/shape),
  which is why logistic and a flat-fitting AE both miss it and the forest catches it.

## 1. POSITION

**Dominant culprit = MODEL/SCORING, not representation or data.** The per-family AE + argmin-
reconstruction is degenerate on near-flat windows: every quiet family's AE reconstructs every quiet
window to ~1e-4, so argmin is noise -> mutual cpu<->io<->thread confusion. Proven by swapping to a
forest on the identical APF windows/folds (cpu 0.024->0.457, thread 0.010->0.568 at LOWO). A zero
here was the model, not the signal.

Mapping to the three physical candidates:
- **APF impoverishment** — real but **second-order, and only for io**: APF caps io's ceiling at 0.408;
  the richer wAPF arm (already on the laptop) lifts it to 0.832. cpu/thread are already ~0.85 on APF
  at the ceiling.
- **Window size (8)** and **trace length** — **not binding at the ceiling**: 8 raw samples suffice for
  the forest to reach 0.83-0.85 on cpu/thread. They may tax LOWO generalization, but the data
  (workload count) dominates that.

**Residual limit after fixing the model = DATA (effective n = workloads).** LORO->LOWO gap: cpu
0.836->0.457, thread 0.736->0.568. Structural zeros for thin families: io (2 wl) -> 0.000, sandbox
(1 wl) -> 0.000, cache (3 singletons) -> 0.16 at LOWO — no scoring fix rescues these.

**Target = legitimate.** cpu/io/thread ARE separable (forest ceiling proves it); the boundary is
subtle and nonlinear, not artificial.

Ranking for the quiet families: **model/scoring (dominant, cpu+thread) > data/effective-n (io,
sandbox, cache at LOWO) > representation/APF-impoverishment (io's ceiling) >> target.**

## 2. FALSIFYING TEST

Run a random forest on the raw 8-dim APF windows under the same LOWO folds; **if it also gets ~0 on
cpu and thread, it is not the AE — it is representation/data, and I am wrong.** I ran it: forest cpu
0.457, thread 0.568 (not ~0). Position survived its own falsifier. The sharper falsifier for "the AE
scoring is the defect": replace argmin-absolute-MSE with a per-AE self-normalized novelty score.
Expected: cpu/thread recover toward the forest; if they don't, the degeneracy story is wrong.

## 3. DEEP H3 TEST (richer representation)

- **Answerable now, on the laptop:** wAPF is a strictly-richer scalar in the npz. At the ceiling it
  lifts the APF-starved families (io 0.408->0.832, cache 0.207->0.793, forest). So a partial H3 is
  already positive for io/cache without server access.
- **The deep H3 (spatial map, spectral, phase, magnitude distribution) is NOT answerable here.** The
  per-snapshot signal was collapsed to two scalars (n_changed/N, ham_sum/(N.BITS)); which pages
  changed, the per-page hamming distribution, and spatial autocorrelation cannot be reconstructed from
  an 8-dim APF window. Needs **server re-extraction** emitting richer per-snapshot features. Critical:
  test H3 at the **ceiling (within_trace/LORO), never at LOWO** — richer features raise the ceiling but
  cannot manufacture workloads, so at LOWO H3 and the data-limit confound and a null is uninterpretable.

## 4. LIMITS (what my lens cannot settle)

- Whether the within-window shape the forest exploits is a **real memory-behavior signature vs a
  trace-incidental artifact** (scheduler/timing) — **ecg-signal-scientist / volatile-memory-expert**.
- Whether a **spectral/phase** view adds beyond within-window variance, and whether 8 samples at hop 4
  support one — **dsp-engineer**.
- Whether **"family" is the right grouping** vs collapsing cpu/thread/io into one "quiet/compute"
  superclass — a target-design call.
- Unverified: laptop trajectories match the exact server run only qualitatively; within_trace is a
  leaky ceiling (window 8 > hop 4 -> overlap), so I lean on LORO for the "info exists" claim.

## Proposals (not executed — need approval)

- **Model fix (laptop, cheap):** report the forest as the B1 family classifier alongside the AE bank,
  or replace argmin-MSE with a self-normalized novelty score.
- **Server re-extraction for the deep H3:** re-emit per-snapshot features beyond the two scalars —
  spatial changed-page histogram/entropy, per-page hamming distribution moments, region
  autocorrelation — then re-run B1 at the within_trace/LORO ceiling.
