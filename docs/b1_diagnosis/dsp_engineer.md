# ROUND 1 — DSP engineer, independent position

All numbers computed on-laptop from `~/b1_data/b1_windows.npz` + `b1_ae_results.json`. Sampling
interval fs=2 Hz (500 ms) verified in `docs/EXPERIMENT_B1_ENCODING_FLOOR.md`, not assumed.

## The sizing arithmetic (settles the window question first)

- fs = 2 Hz, dt = 0.5 s, W=8 -> window span **4.0 s**. Freq resolution df = fs/W = **0.25 Hz**.
  Nyquist = **1 Hz** (min resolvable period 1.0 s). The 8-window has exactly **4 non-DC bins**:
  periods {4, 2, 1.33, 1.0} s. It can represent essentially nothing but a level plus one slow ramp.
- Measured autocorrelation time of the reconstructed per-cell series: **tau(1/e) = 0.5 s for the
  quiet trio (= 1 sample), 1.0 s for app**; lag-1 ACF cpu 0.23 / io 0.19 / thread 0.26 / mem 0.07.
  The signal decorrelates within 1-2 samples for everyone.
- Within-window **AC power is 3-9% of total** (mean); 91-97% is the DC level. mem is 0.3% AC (a flat
  saturated plateau). Full-trace spectra are broadband with no peak (low/mid/high band split
  ~0.20/0.42/0.37 for the trio ~= white).

Conclusion from the arithmetic: **there is no temporal rhythm to resolve at any window length.** A
bigger window cannot recover a structure that the autocorrelation shows is absent. Architecturally
this is forced: memory writes happen at ns-us; sampling at 2 Hz is ~6 orders of magnitude below the
process, so any real temporal signature is aliased to noise (reasoning, not a measured bin). The only
axis that can carry family is **spatial** (which pages, how hard), and that is precisely the axis APF
integrates away.

## 1. POSITION

**Cause (3), APF impoverishment, dominates for cpu/io/thread — specifically the spatial averaging
over 262144 pages. This is an observability failure of the APF/wAPF observable, not a resolution
failure of the 8/4 window and not physiological absence.** Trace length (1) is a distant third;
window size (2) is not the bottleneck.

Evidence:

- **The trio writes; it is not at the floor.** Median changed pages/snapshot: cpu 134, io 146, thread
  142 (windowed); full-trace 169/267/184. `frac_at_0 = 0.000`, `<=1 page = 0.00%` for every family.
  So the substrate is not empty for them — this rules out "they don't write" as the explanation.
- **Their scalar level is degenerate and stays degenerate at every temporal extent.** Windowed
  p10-p90 bands overlap **66-80%** across the trio. I ran the falsifier — full-trace APF (maximal
  window): cpu 169 / thread 184 / io 267 median pages, still one band. **Longer window/trace does not
  separate them.** This kills (1) and (2) as the dominant cause. Contrast the visible families:
  app 7945, mem 65355 pages — separated from the trio by 1.5-3 orders of magnitude, which is why the
  AE bank works purely as a *level* detector.
- **The discriminative signal is provably still in the per-page field.** The wAPF/APF ratio (mean
  bits flipped per changed page, /32768) is **cpu 0.137, io 0.093, thread 0.027 — a 5x spread**
  across the trio (thread ~= 111 bytes/changed page, cpu ~= 561). That is a real magnitude
  discriminator sitting in the Hamming channel. Yet the wAPF *arm* — that same quantity spatially
  summed into a scalar — gives LOWO recall cpu 0.001 / io 0.016 / thread 0.005. **The averaging
  destroys a discriminator that demonstrably exists per-page.**
- **io's apparent success is level-memorization, not structure.** io recall runs within-trace 0.968
  -> LORO 0.027 -> LOWO 0.018. Most io cells sit at ~200-270 pages, but a subpopulation spikes to
  72k pages (cv 10.3) — workload-specific bulk sequential writes the model memorizes on the same
  trace and cannot generalize.

## 2. FALSIFYING TEST (single cheapest, would prove me wrong)

From the **already-captured** `hc_field.csv.zst` (columns `seq,page_index,hamming,cosine` — no
recapture; cosine is stored but UNREAD by B1), compute one **purely spatial/magnitude descriptor per
snapshot that APF discards** — e.g. count of distinct `page_index` in the cumulative working set,
spatial entropy of the changed-page histogram, and 2-3 quantiles of the per-page `hamming` — then run
the existing LOWO split on the quiet trio, window and trace unchanged.

- If I'm right (cause 3): the trio separates **well above the current ~0.01-0.02 recall** using a
  feature that changes nothing about time — proving representation, not resolution, was the wall.
- **Falsifier:** if the per-page spatial/magnitude descriptors **also** fail to separate cpu/io/thread,
  the signal is not in the write channel at all, cause (3) is wrong, and it moves to
  substrate-observability (writes-only) / workload physiology.

The temporal half of this falsifier I already ran (full-trace level; trio does not separate), so the
remaining decisive measurement is the per-page one, which needs the server.

## 3. DEEP H3 TEST — needs and answerability

- **Channels:** per-page `hamming` (magnitude), `page_index` (spatial location / working-set shape),
  and `cosine` (orientation) — all present in `hc_field.csv.zst` from the Aug substrate campaign. B1
  currently reads none of the spatial/orientation content.
- **Cells:** the trio (cpu 11, io 8, thread 12 cells) plus app/mem as positive controls, across
  multiple workloads per family so LOWO is meaningful.
- **Server commands:** the `hc_field` files are server-side and unreachable from the laptop. Proposed
  for approval (not executed): a small read-only script that streams each `hc_field.csv.zst`, emits
  per-snapshot {n_distinct_pages, spatial_entropy, hamming p50/p90} aligned to `seq`, windows 8/4,
  scores with the unchanged LOWO. Proposed new script, not an existing capability.
- **Answerable without full access?** Partially. Temporal/level half settled on the laptop: no. The
  "is the discriminative energy in the per-page field" half needs the server `hc_field`.

## 4. LIMITS (what my lens cannot settle)

- Whether a learner can extract the per-page representation, and whether effective-n (workloads, not
  cells) supports LOWO generalization — **ml-engineer**.
- Whether cpu/io/thread are physiologically distinguishable in *writes* at all, or whether their
  signature lives in reads/cache/registers the QEMU write-diff substrate never records —
  **volatile-memory-expert** and **ecg-signal-scientist**.
- cache (recall 0 despite non-novelty; bimodal) and sandbox (single-workload novelty) — more label/ML
  questions than DSP.

Unverified flags: the aliasing/Nyquist architectural argument is reasoning, not a measured spectral
bin; the proposed spatial-features script is a proposal, not an existing script.
