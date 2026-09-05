# ROUND 1 — Independent position (ecg-signal-scientist)

Read the two flavors on the laptop (APF, wAPF), reconstructed the full per-cell series, ran the
transforms, then read the differ substrate and the original extractor to see what APF discarded.
Numbers first; the analogy is held to account by them.

## What the data actually shows (computed, not asserted)

**Amplitude.** Quiet-family APF mean ~= 0.00077 (cpu 0.00080, io 0.00083, thread 0.00068). That is
~247x below mem, ~47x below app — but **not** at the quantization floor. At a ~GB guest (~260k pages)
this is ~150-350 changed pages per snapshot, i.e. ~200 quanta of dynamic range. There is room for
morphology.

**Not asystole (decisive).** Order-shuffle surrogate test on the full series (200 surrogates):
**100% of cpu, io, thread series are structured** (median p~0.005, the floor). All three are
autocorrelated (cpu/thread AC peak ~0.29-0.31 at lag ~15 = ~7 s; io 50% above the white band). There
is a heartbeat in every quiet family. The write channel is not flat.

**But the beat is not family-discriminative in the APF/wAPF lead (also decisive).** Leave-one-workload-
out, 3-class (cpu/io/thread), **RandomForest** (stronger than the AE bank), majority baseline 0.394:

| feature set (from APF/wAPF only) | LOWO acc | io recall |
|---|---|---|
| raw APF-8 (what B1 used) | 0.387 | 0.00 |
| 6 shape feats (std, range, ac1, cv, slope) | 0.375 | 0.005 |
| level-normalized shape (amplitude removed) | 0.346 | 0.005 |
| longer windows + full spectral (32 s) | 0.441 | 0.05 |
| wAPF-8 (magnitude) | 0.231 | 0.02 |
| APF+wAPF (2-channel, the E2 plan) | 0.333 | 0.00 |
| intensity ratio wAPF/APF | 0.225 | 0.00 |

Nothing beats majority. Longer windows help only marginally (0.39 to 0.44) — window length is a real
but second-order factor, not the fix. The magnitude lead on the laptop (wAPF) *hurts*. The mean-ACF
shapes overlap across families (cpu~thread 0.53, cpu~io 0.66): the rhythm exists but looks the same
in this lead. **This is a channel ceiling, not a model ceiling** — RF with spectral features still
cannot separate them, so Cause 1/2 (traces/window) are not the story; Cause 3 (APF is impoverished)
is.

**The within-trace io=0.968 is a memorized DC offset, not transferable shape.** io separates within a
recording (per-run baseline write rate) but collapses to ~0 recall across runs/workloads. Trace-
specific level, not family morphology.

**Within-family splits (the real reason the pool forms).** All three quiet families are near-identical
in breadth (~0.0005 median) *and* internally split between a writer and a near-non-writer:
- cpu: matrix_mult APF 0.00108, CoV 0.42 (streaming result matrix, louder/structured); hash_loop &
  branch_random at floor 0.00045.
- io: direct_write (O_DIRECT, writes, burst max 0.021) vs read_cache_hit (read-bound, median at
  floor). Medians nearly equal — indistinguishable in breadth.
- thread: producer_consumer & parallel_alloc (write a ring / zero-fill fresh pages) vs lock_contention
  (futex spin, near-silent). All three ~0.0005.

## What APF discarded, and whether it plausibly carries the morphology

From the differ (`metrics/mod.rs`, `common.rs`) the substrate is 64 per-page channels, all
change-gated (identical/unwritten pages short-circuit to all-zeros). APF = spatial-mean of the changed
indicator, then windowed. It threw away, for the ~200 changed pages/snapshot: **magnitude**
(hamming/l1/l2), **direction/phase** (cosine, family B), **where in the address space** (the per-page
field / change-location), and it never applied the FFT/wavelet/cepstrum lenses to anything but the
resulting scalar.

Two hard facts constrain the fix:
- **At speed 2 (the campaign setting) the phase lens is OFF.** `family_b/spatial_shift.rs` and
  `family_d/texture.rs` return zeros at speed>=2: cross_corr_lag, phase_corr, byte_rotation (FFT
  phase), within-page autocorr, high_freq_frac, GLCM x4 all emit 0. **Surviving at speed 2: magnitude,
  cosine (direction), change-location (where-in-page), content.** So the founding intuition's named
  phase/FFT lens was literally not computed in the run that produced B1.
- The design graph (`part_graph.html`) already names the missing leads as families **E - spatial
  (field)** and **F - temporal (FFT/cepstrum/wavelet)** — "specified, not captured." The original
  `block_feature_extractor.py` *does* realize the full path: `make_complex(mag,phase)=mag.e^{j2pi.phase}`,
  FFT, a **2D spatial wavelet** (cA/cH/cV/cD, magnitude+phase per level), and a Hilbert temporal phase.
  B1 is the maximal retreat from that to a scalar count.

## 1. POSITION — per quiet family

Definition held: **wall = the family's identity lives in reads/compute, which do not write, so no
lead recovers it.** **Re-representable = the family writes distinctively, but on an axis (space,
per-page magnitude/direction) APF averaged away.** None of the three is uniform asystole — all write
and all are temporally structured — so the verdict is per-workload inside each family.

- **cpu — RE-REPRESENTABLE (conditional), not a wall.** matrix_mult writes 2.4x more, structured
  (CoV 0.42, streaming result matrix -> contiguous spatial sweep); all cpu series pass the structure
  test. The discriminator is spatial (scatter of a hash table vs a contiguous matrix sweep), APF
  averaged out. Caveat: hash_loop/branch_random sit at the floor and are compute-in-cache; only the
  spatial field will tell whether they are structured or genuinely near-silent.
- **io — SPLIT; the most wall-like family.** io_read_cache_hit is the diary's mem_pointer_chase
  lesson: reads produce nothing -> **observability wall.** io_direct_write does write -> re-representable
  (bursty temporal + spatial write band). Indistinguishable in breadth, so the *family label* averages
  a writer with a near-non-writer. **Do not expect "io" to come back as one coherent family from the
  write channel; it is not acoustically one family.** The cleanest asystole finding: half of io does
  not beat.
- **thread — RE-REPRESENTABLE (conditional), SPLIT.** producer_consumer writes a ring -> moving
  spatial band / sweeping centroid, the clearest spatial-rhythm signature; parallel_alloc zero-fills
  fresh pages -> scattered strong spatial write signature. lock_contention is futex spin -> near-silent,
  borderline wall.

Net: the founding intuition is **vindicated but bounded**. Restoring magnitude+phase+spatial+spectral
should recover the *writers* (cpu matrix_mult, io_direct_write, thread producer_consumer/parallel_alloc).
It will **not** manufacture io_read_cache_hit or any pure-read/spin member — the real asystole.

## 2. FALSIFYING TEST — cheapest view that settles re-representable vs wall

The laptop scalar has already falsified "window/length is the fix" (0.39->0.44 ceiling; io ~0
everywhere). The remaining question is settled by **one offline pass over the already-captured
per-page hamming+cosine field** (no re-capture): for each quiet workload, per snapshot compute the
**spatial write-location entropy** and **change-centroid trajectory** across the address space (the
E-spatial family), then run LOWO on those spatial features for the quiet three.

- **Re-representable:** matrix_mult / producer_consumer / parallel_alloc / io_direct_write show low,
  stable, family-distinct spatial write signatures, and LOWO recall rises well above the 0.44 breadth
  ceiling.
- **Wall:** io_read_cache_hit and lock_contention have empty or indistinguishable write-location
  fields and stay unseparated no matter the lens.

## 3. DEEP H3 TEST — channels / region / lens / cells + server command

- **Channels:** magnitude (hamming/l1) + **direction (cosine)** + **change-location/centroid** (all
  alive at speed 2) *plus* the phase/FFT leads that need **speed 0** (cross_corr_lag, phase_corr,
  byte_rotation, within-page autocorr, high_freq, GLCM).
- **Unit/region:** the per-page **field** in address-space order, restricted to the anonymous/heap
  region where working sets live.
- **Lens:** the original `block_feature_extractor.py` path — `make_complex(hamming, cosine)`, **2D
  spatial wavelet** on the complex field per snapshot, then **temporal FFT/cepstrum** of the
  wavelet-energy series.
- **Cells (maximally-separated writer/near-wall pairs):** cpu **matrix_mult vs hash_loop**; thread
  **producer_consumer vs lock_contention**; io **direct_write vs read_cache_hit**. Two representative
  consecutive-dump pairs each.

Server command (differ `live_delta_calc_modular`, usage `[--speed N] [--sparse] <prev> <new> <out>`;
speed 0 = all leads):

```
live_delta_calc_modular --speed 0 <dump_t.raw> <dump_t+1.raw> <out>/cpu_matrix_mult
live_delta_calc_modular --speed 0 <dump_t.raw> <dump_t+1.raw> <out>/cpu_hash_loop
live_delta_calc_modular --speed 0 <dump_t.raw> <dump_t+1.raw> <out>/thread_prodcons
live_delta_calc_modular --speed 0 <dump_t.raw> <dump_t+1.raw> <out>/thread_lock
live_delta_calc_modular --speed 0 <dump_t.raw> <dump_t+1.raw> <out>/io_direct_write
live_delta_calc_modular --speed 0 <dump_t.raw> <dump_t+1.raw> <out>/io_read_cache_hit
# then block_feature_extractor.py (complex mag/phase -> FFT + 2D wavelet) over each metrics CSV field.
```
(Exact dump paths / cell->workload manifest are on the server — the user fills `<dump_*.raw>`;
propose, do not execute.)

**Answerable without full access?** Partly, cheaply: the **spatial + magnitude + direction** leads are
recoverable *offline from already-captured data* (per-page hamming/cosine field on the server; cosine
+ change-location survived at speed 2). That is the falsifying test in section 2, zero re-capture.
Only the **phase/FFT-shift** lead requires the targeted speed-0 re-capture — a handful of cells.

## 4. LIMITS

- **Cannot measure the spatial field on the laptop** (only the two scalar reductions). The
  re-representability claim for the writers is a strong prediction, not yet a measurement. Labeled
  unverified.
- Guest page count N (~260k) from the mandate, not verified from these files.
- **dsp-engineer** owns window-length/Nyquist and whether the temporal lens has enough cycles.
- **ml-engineer** owns whether, given restored features, the AE-bank can exploit them, and whether the
  within-trace-vs-LOWO gap is instead a leakage/normalization artifact.
- **volatile-memory-expert** owns the ground truth on whether each near-wall workload writes anything
  recoverable — the one call the analogy must defer to physics on.

**Bottom line:** No quiet family is a flatline (all beat). But the discriminating morphology is not in
write-*breadth* or write-*magnitude* (proven to the channel ceiling on the laptop) — it is in *where*
and *what* they write, the E-spatial and B-direction leads APF averaged away and speed-2 partly
switched off. The writers are re-representable; the pure readers/spinners (io_read_cache_hit above all)
are the genuine wall.
