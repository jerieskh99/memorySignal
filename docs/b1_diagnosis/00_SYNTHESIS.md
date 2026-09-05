# Why B1's quiet families fail — four-expert diagnosis + synthesis

Round 1 of a multi-expert investigation (2026-09-05). Four independent analysts — a DSP engineer,
an ML engineer, a volatile-memory engineer, and an ECG/biomedical-signal scientist — each analyzed,
on the laptop, the real B1 windows (`~/b1_data/b1_windows.npz`) and results, plus the code. Their
verbatim reports are the sibling files in this folder. This is the orchestrator's synthesis,
including a clinching per-workload test that overturned two of their predictions.

The question: **B1's per-family autoencoder bank recognizes app (0.84) and mem (0.75) on unseen
workloads but collapses cpu/io/thread to ~0.01. Why, and what fixes it?** Three candidate causes:
(1) traces too short, (2) window too small (8 samples = 4 s), (3) APF too impoverished a metric.

## Where all four converged

- **Causes (1) and (2) are ruled out — on the laptop, with data.** DSP: the signal decorrelates in
  1-2 samples; there is no temporal rhythm for a longer window to resolve. ECG: swept raw / shape /
  spectral / longer-window (32 s) features with a random forest — best 0.44, nothing beats the ~0.39
  majority baseline. Window length is a second-order factor at most.
- **The quiet families are not silent.** They write ~130-150 changed pages/snapshot (DSP), and 100%
  of their series are structured / autocorrelated (ECG surrogate test). Not asystole.
- **But the discriminator is not in APF's scalar.** DSP: the per-page magnitude spread (5x across the
  trio) is destroyed by spatial averaging. Memory: the quiet families sit **at the guest-OS floor** —
  `cpu_hash_loop` / `cpu_branch_random`, designed to write nothing, still produce ~118 pages/snapshot
  of pure OS bookkeeping (timekeeping, scheduler, per-CPU stats). The workload signal is smaller than
  the OS noise it rides on.

## The ML vs ECG apparent clash, reconciled

ML: a random forest on the *same* 8-dim APF windows lifts cpu 0.024 -> 0.457, thread 0.010 -> 0.568
at LOWO — a 19-59x win over the AE bank, proving the AE's argmin-reconstruction scoring is degenerate
on near-flat windows (every quiet family's AE reconstructs every quiet window to ~1e-4, so argmin is
4th-decimal noise; a synthetic cpu-level window is labeled thread). ECG: the same forest **does not
beat majority (0.39)**. Both are the same numbers: the forest crushes the broken AE but only reaches
~chance in absolute terms. The AE degeneracy is a real, fixable model bug — and fixing it is not
sufficient.

## The clinching test — per-workload recall inverts two predictions

Memory and ECG predicted the **writers** (matrix_mult, parallel_alloc) are re-representable and the
**readers/spinners** (hash, branch, lock) are walls. The per-workload forest LOWO recall says the
opposite:

| workload | forest LOWO recall | expert prediction |
|---|---|---|
| thread_lock_contention | 0.833 | "wall / borderline" |
| cpu_branch_random | 0.725 | "wall" |
| thread_producer_consumer | 0.714 | re-representable (correct) |
| cpu_hash_loop | 0.678 | "wall (read-only)" |
| thread_parallel_alloc | 0.114 | "re-representable" |
| cpu_matrix_mult | 0.011 | "the structured writer" |
| io_direct_write / io_read | 0.000 / 0.000 | mixed |

**The forest recovers the floor-riders and rejects the actual writers.** Mechanism: LOWO holds out
one workload and trains on its family-mates. cpu = {hash, branch, matrix_mult}. Hold out matrix_mult
-> train on two floor workloads -> matrix_mult (structured, high-activity) does not resemble its
floor-riding family-mates -> rejected. Hold out hash_loop -> its floor twin branch_random is in
training -> recognized.

## What it means (the real conclusion)

The forest's "recovery" is **the OS-background floor signature, not workload behavior.** Each quiet
family is mostly floor-riders sharing an OS pattern, plus a minority of real writers; the model learns
the floor (the majority) and classifies by it. So:

1. APF family-classification of the quiet families largely measures the **environment / OS floor**,
   not the workload.
2. **"Family" is not a coherent target** for them — each averages floor-riders with writers, and the
   model learns whichever dominates.
3. **app/mem (0.84/0.75) are now suspect as pure *level* recognition** and must be re-checked.

Not "add a richer metric and it's fixed," but: **the family success is largely floor/level
recognition; the writers are the ones the family model rejects; and the honest signal, if any, is
spatial and per-workload, not per-family.**

## Decisive next tests (cheap, mostly existing data)

1. **Idle baseline** (memory's falsifier, one server command): capture an idle guest; if idle is
   indistinguishable from hash/branch/lock, the "recovery" is provably OS-environment, not workload
   -> the quiet families are unclassifiable from writes at the family level. The single most important
   missing measurement.
2. **Spatial signatures from the already-captured `hc_field`** (no recapture — cosine + change-location
   survived at speed 2): do the writers (matrix_mult, parallel_alloc) have distinctive spatial
   write-location signatures? This is the real H3, testable offline on existing data.
3. **Re-check app/mem for level/floor leakage** (laptop, now): is their 0.84/0.75 behavior or just
   level recognition?

Two flags: the **phase/FFT lens** the founding intuition centered on was **switched off at speed 2**
in the B1 run (ECG: `phase_corr`, within-page autocorr, GLCM all emit 0), so testing it needs a few
speed-0 recaptures. And the target may need to shift from "which family" to "does it write, and what
is its spatial signature."

## Per-expert ownership (who settles what next)

- **DSP** — window/Nyquist arithmetic; whether the temporal lens has enough cycles.
- **ML** — model/scoring fix; whether restored spatial features are learnable; effective-n limits.
- **Volatile-memory** — ground truth on whether each near-floor workload writes anything recoverable
  (the idle baseline is theirs).
- **ECG** — whether the recovered signal is real morphology vs artifact; the "family is not one
  family" reframing.

See the four verbatim reports beside this file: `dsp_engineer.md`, `ml_engineer.md`,
`volatile_memory_expert.md`, `ecg_signal_scientist.md`.
