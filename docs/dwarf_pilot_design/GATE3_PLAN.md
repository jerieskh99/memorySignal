# Gate 3 plan — the pilot capture (narrowed after Gate 2)

Status: drafted 2026-09-06, after Gate 1 (GO, QR~1600 / LU~2400, measured by
[`gate1_dwarf_timing.sh`](../../VM_sampler/VM_Capture_QEMU/plan09_dwarf_pilot/gate1_dwarf_timing.sh))
and [Gate 2](GATE2_RESULT.md) (SCATTERED → address-shape features dropped).

## What changed, and why this is now cheaper

Gate 2 killed the address-shape feature family (guest-physical pages scatter). That
family was the **only** reason the pilot needed the slow `--speed 0` differ (~39s/pair,
offline-only) — it was the only mode emitting `page_index`. The two surviving features
are **placement-invariant** and both are computed at every differ speed:

- per-page **magnitude** (`hamming`, `l1`, `l2`, `linf`, `mean_abs`) — in
  `live_delta_calc_modular/src/metrics/family_a/positional.rs`, no speed gating.
- **footprint size** — the count of changed pages per snapshot (rows per `seq` in
  sparse output).

Only the heavy-12 texture/spectral metrics drop at speed 2 (`metrics/mod.rs:46`), and we
do not use them. **So Gate 3 runs at `--speed 2`** — the fast, live, 500ms-cadence
production setting. No drain backlog; the same path the 101-workload campaign used.

## The two questions Gate 3 decides

1. **Footprint-size trajectory** — does the *count* of changed pages over time separate
   the resized fronts? QR (growing front) should ramp up within a factorisation; LU
   (shrinking front) should ramp down; GEMM (full C rewrite every pass) should stay flat.
   This is the surviving, count-based form of the temporal arm.
2. **Magnitude distribution** — does *how much* each page changes separate a smooth
   low-amplitude writer (gibbs, single-byte cell updates that decay as it mixes) from a
   high-entropy re-seeder (GEMM/QR reseed a whole matrix with fresh randoms each pass)?

## Pilot cells (all verified: names, flags, built binaries)

| label | binary | args | role |
|---|---|---|---|
| qr | kernel_qr_v2 | `--dim 1600` | growing footprint (Gate 1 dim) |
| lu | kernel_lu_v2 | `--dim 2400` | shrinking footprint (Gate 1 dim) |
| gemm | kernel_gemm_v2 | `--dim 1024 --block 64` | flat footprint / WORKING-SET null |
| gibbs | kernel_gibbs_v2 | `--width 1024 --height 1024` | smooth low-magnitude / decay |
| stencil | kernel_stencil_jacobi_v2 | `--grid-n 1024` | ECG aliasing control (should stay flat/saturated) |
| idle | (none) | — | baseline; the tripwire — must be inseparable at floor |

3 reps each (seeds 42/43/44), `--duration 70` (≈140 snapshots at 500ms; the design's
≥70s / 145-frame target), `--speed 2`, sparse substrate mode.

## Analysis (after capture — deferred, built once data exists)

Per cell/rep, from `substrate_trajectory.csv`:
- footprint-size series: rows per `seq` → time series → shape features (within-pass slope,
  sawtooth period, mean/variance of size). Normalise by trace-fraction; drop warm-up.
- magnitude distribution: per-snapshot moments/histogram of `l1` (and `l2`) over changed
  pages; and its evolution over time (does it decay for gibbs?).
Discriminative model (trees), not an AE bank. Split unit = rep (leave-one-rep-out).

## Go / no-go

- **GO** — footprint-size trajectory separates {qr growing, lu shrinking, gemm flat}
  leave-one-rep-out, AND magnitude distribution separates gibbs from gemm, AND idle stays
  at floor (inseparable): the placement-invariant features carry dwarf signal → design a
  fuller campaign (more kernels per archetype, cross-motif leave-one-dwarf-out).
- **NO-GO** — they collapse: the 500ms memory-write signal does not separate these
  compute kernels beyond gross footprint size → deliver the dwarf→archetype confusion
  matrix (the pre-registered honest result) and stop the temporal/magnitude arm.

## How it runs

Lean per-cell capture loop, extending the Gate 2 script (isolated scratch config, speed 2,
production config untouched): [`gate3_pilot_capture.sh`](../../VM_sampler/VM_Capture_QEMU/plan09_dwarf_pilot/gate3_pilot_capture.sh).
The full campaign orchestrator (`subset_run.py`) is deliberately NOT used — it derives
cells from a scale-factor ladder that does not include the custom Gate-1 dims (1600, 2400),
and is far heavier than a 6-cell pilot needs.
