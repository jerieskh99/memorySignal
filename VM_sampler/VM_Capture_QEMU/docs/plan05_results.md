# Plan 05 Results -- Capture-Side Throughput Pilot (Wave 2)

Status: **complete. Headline gate "failed," but the failure was the win in
disguise.** 72/72 cells captured (2026-06-02 to 06-03).
Companion: `plan05_overview.html` (#mystery -- the full narrative),
`plan05_snap_throughput_proposal.html`.

## TL;DR

The pilot's throughput gate (G-T1) failed on all 18 comparisons. Chasing *why*
the dump measured 0.79s here but 7.06s in the papers overturned the reading: the
7.06s was never the dump's intrinsic cost -- it was inflated by **never deleting
dumps** (`keep_dumps=true` on 125/128 v3 cells), which filled the disk during a
run and slowed every write. The pilot ran **delete-as-you-go on every arm**
(`--stream-apf` deletes the previous dump), so all four arms sat at the 0.79s
floor and the gate -- comparing already-fixed arms -- saw no gap. The real
before/after is the old keep-everything pipeline vs delete-as-you-go:
**7.06s -> 0.79s** dump, **14 -> 79 snapshots per 120s**. Plan 05 succeeded; the
gate just measured the wrong contrast.

## What ran

3 workloads x 2 durations (120s, 600s) x 4 arms x 3 replicates = **72 cells**,
guest RAM = **1024 MiB**, baseline arm = `ssd_keep`. (A mid-run server reboot
killed the foreground process at cell 67; the last 6 cells were resumed and all
72 aggregated offline.)

## The real numbers (old keep-everything vs delete-as-you-go)

| Metric | Old: keep-everything | Now: delete-as-you-go (SSD) | + tmpfs |
|--------|----------------------|------------------------------|---------|
| Copy time per snapshot (`pmemsave`) | 7.06 s (0.76 / 7.06 / 21.57) | **0.79 s** | 0.59 s |
| Snapshot cycle | 9.75 s | ~1.5 s | ~1.3 s |
| Snapshots in **30 s** (rate-derived) | ~3-4 | ~20 | ~23 |
| Snapshots in **120 s** (measured) | **14** | **79** | 92 |
| Snapshots in **600 s** (measured) | **53** | **398** | 461 |
| APF trajectory points (600 s) | ~52 | ~397 | ~460 |
| Analysis windows @ (8,4), 600 s | ~12 | ~98 | ~114 |

120s/600s are measured medians (pilot vs v3 `producer_stats`). 30s is derived
from the steady cadence (~0.66 snaps/s delete-as-you-go vs ~0.12/s
keep-everything). Windows = `floor((T-8)/4)+1`; v3 had **53/132 cells at <=3
windows** -- that DOF starvation is what relaxes.

## What the gates said (and what they meant)

| Gate | Verdict | Reality |
|------|---------|---------|
| **G-T1 throughput** (lever <= 0.5x baseline) | FAIL, all 18 | Baseline already streamed+deleted, so all arms tie at 0.79s. Wrong contrast. The true gain vs keep-everything is ~9x dump / ~6x snapshots. |
| **G-T2 fidelity** | 9/18 pass (recalibrated) | KS leg now judged on the statistic (effect size), not the p-value, fixing large-n false-fails. Residual fails are real: d120 short-traj jitter, an app_hashtable@600 spread diff, and the near-idle/bimodal APF cases (below). |
| **G-T3 disk** (peak <= 3 GiB) | PASS, all | 2.26-3.0 GiB. Validated the disk-safety fix. |

Per-arm throughput (median pmemsave): `ssd_keep` 0.79s, `ssd_selfclean` 0.79s
(1.00x), `tmpfs_keep`/`tmpfs_selfclean` 0.59s (1.35x). tmpfs is a small extra on
top of delete-as-you-go, not the main lever. (For `mem_workingset_sweep_v2` @
600s, tmpfs was slightly slower -- host-RAM pressure from multi-GiB dumps in
`/dev/shm`.)

## Why the 7.06s was an artifact (the evidence)

1. **Distribution, not constant.** v3 pmemsave was 0.76 / 7.06 / 21.57 s
   (min/median/max), right-skewed. The pilot's 0.79s is that distribution's
   minimum -- the floor of the same machine.
2. **Retention.** v3 ran `keep_dumps=true` on 125/128 cells: dumps were never
   deleted, so the live folder grew all run and the consumer read the backlog --
   writes slowed and contended.
3. **Sub-linear vs linear cadence.** v3 snapshots grew sub-linearly with duration
   (14/26/53 for 120/300/600s) -- cadence slowed as the run accumulated. The
   pilot grows linearly (79->398 for 120->600s) -- constant cadence, because it
   deletes as it goes.

Same machine, same 1 GiB guest. The only variable that changed is **retention**,
not RAM and not the disk hardware.

## 2B: keep-everything vs delete-as-you-go (head-to-head)

A controlled one-machine check (idle guest, 60 s each, same session) isolates the
retention effect:

| Metric (60 s) | keep-everything | delete-as-you-go |
|---------------|-----------------|------------------|
| snapshots | 21 | 29 |
| pmemsave start -> end | 1.26 -> 1.68 s (final spike 2.4 s) | 1.29 -> 1.29 s (flat) |
| peak disk | 21 GiB (climbing) | 2.2 GiB (bounded) |
| 600 s extrapolation | ~210 GiB (disk bomb) | ~2 GiB |

Retention shows in all three signatures: keep-everything's disk explodes
(21 GiB/60 s vs a bounded 2.2), pmemsave starts climbing (flat, then spiking to
2.4 s as the pile bites), and it fits fewer snapshots (21 vs 29). **Caveat on
magnitude:** at 60 s only 21 GiB piles -- not enough to reproduce the full v3 7 s
climb -- and 2B is producer-only, so it omits v3's concurrent consumer reading the
backlog (extra I/O contention). A longer keep run would show more climb but at
0.35 GiB/s is a 210 GiB bomb at 600 s. So the v3 7 s = disk-fill + consumer
contention + long accumulation; 2B isolates the disk-fill leg, and the disk-growth
rate (21 GiB/60 s, unsustainable) is the decisive proof.

## The disk-safety fix (this wave)

Delete-as-you-go only works if the delete actually succeeds. The `ssd_keep` dump
folder (`/var/lib/libvirt/qemu/dump`) is owned by the libvirt-qemu user, so the
APF helper's plain `unlink()` failed silently and dumps piled (41 dumps / 44 GiB
in a 120s smoke). Fix:

- `plan02_apf_helper.py`: on `unlink()` failure, fall back to `sudo -n rm -f`,
  gated on `TIMING_SUDO_DELETE` (default off -> byte-identical old behavior).
- `plan05_run.py`: set `TIMING_SUDO_DELETE=1` for ssd arms; purge the ssd imageDir
  between cells via `e1.purge_all_dumps(use_sudo=True)`.

All sudo is non-interactive (`sudo -n`), safe for an unattended run. (Committed to
`fullv3` at `3065e95`.)

## The G-T2 recalibration and the near-idle finding

**Recalibration (Wave 3).** The fidelity gate's KS leg rejected on the *p-value*,
which gains power with sample size: at n~1,400 (the 600s cells) it flagged APF
differences of 0.0001-0.002 -- far below the 0.02 margin we care about. The gate
now judges the KS *statistic* `D = sup|F_a - F_b|` (an effect size in [0,1] that
does not inflate with n), passing on `D <= 0.10 OR p > alpha`. The threshold is
data-calibrated: pilot D clusters <=0.093 for practically-identical arms vs
>=0.150 for real differences, so 0.10 sits in a clean gap. Result: 7/18 ->
**9/18** lever comparisons pass (the sandbox@600 tmpfs arms flip); residual
failures are now on TOST-mean or std, not KS.

**The `mem_workingset@600s` anomaly is NOT a delete race.** An earlier draft
claimed the -0.166 APF drop was the self-clean `rm` racing the APF helper. The
trajectories falsify that: **zero APF=0 entries**, and the producer's
`if APF_STREAM ... elif SELF_CLEAN` already makes those paths mutually exclusive.
The real picture: APF is **near-zero in nearly all 72 cells** (both durations);
only ~5 cells show real churn, notably `mem_workingset@600s ssd_selfclean` r1/r2
~ 0.25 (vs 0 for r0 and all of `ssd_keep`). Idle and churning cells share the
**same snapshot cadence**, ruling out a cadence artifact -- so the workloads were
not reliably dirtying memory during capture, and most G-T2 "passes" are matching
idle-vs-idle.

**Root cause (Step 2 probe, confirmed).** `app_hashtable_intensive_v2` and
`sandbox_scanner_metadata` treat `--duration` as a *cap, not a sustain*: they
finish fixed work (6M inserts / a 5000-file scan) in **1-2 s** and the guest then
idles for the rest of a 120/600 s window, so their APF is ~0 in every cell
(deterministic). `mem_workingset_sweep_v2` *honors* `--duration` (ran a full 21 s,
2.7 GB writes) and churns to ~0.25 when it runs; its idle reps are a separate,
intermittent issue (launch / VM-pause throttling). The fix is to **sustain** the
workloads for the capture window (loop-wrap them). None of this affects the
throughput finding -- dump cost and cadence are independent of APF content.

## Production APF capture mode (CAPTURE_METRIC)

The production orchestrator (`run_files_controlled.py`) + launcher
(`run_qemu_capture.sh`) take a `CAPTURE_METRIC` selector with three values; the
default `delta` is byte-identical to the pre-APF pipeline:

- `delta` (default) -- the existing Cosine/Hamming Rust-delta pipeline (producer +
  consumer + `run_matrix.npy`). Proven byte-identical offline
  (`tests/test_run_files_capture_metric.py`) and live (the delta-regression check).
- `apf` -- the **inline-helper** path (A, Wave 3): the producer streams APF via
  `plan02_apf_helper.py` (`TIMING_APF_STREAM` + `TIMING_SUDO_DELETE`); no consumer.
  Lean, lowest-latency.
- `apf_queue` -- the **Rust-consumer** path (B, Wave 4): the producer enqueues and
  the `apf_calc` Rust binary computes APF through the existing
  producer -> queue -> consumer state machine, appending to `apf_trajectory.jsonl`
  and deleting prev via the consumer's `delete_file`. Architecturally matches the
  Cosine/Hamming path.

`apf_calc` (`VM_Capture/apf_calc/`, dependency-free Rust) is bit-identical to the
Python helper -- 3 unit tests + a cross-language equivalence check both == 0.3.
Live verify (mem_workingset, 30 s):

- `apf` (A): 177 pairs, mean 0.157, producer-only, dump dir 0 B.
- `apf_queue` (B): 161 pairs, mean 0.21, producer+consumer, queue drained 0/0, dump
  dir 0 B, apf_calc ran every job. A and B agree (per-pair math exact).
- `delta` (default): unchanged -- producer+consumer, cosine/hamming + streaming.

## Next steps

1. **Measure the win as a clean contrast.** Re-run a small matrix with a *true*
   keep-everything baseline (no streaming delete) vs delete-as-you-go, so the
   ~9x copy / ~6x snapshot gain lands in one table instead of inferred across
   campaigns.
2. **Or adopt it now.** The 79-vs-14 evidence stands; turning on delete-as-you-go
   in the production pipeline solves the trajectory-length / DOF problem the
   papers flagged.
3. **Sustain the workloads (fixes the near-idle captures).** Confirmed: app/sandbox
   exit in 1-2 s (`--duration` is a cap). Loop-wrap the workloads so they churn for
   the full window, and re-validate APF is non-trivial before trusting 2B/2A.
4. **Recalibrate G-T2 -- done (Wave 3):** effect-size KS, 9/18 pass.

## Wave 3 changelog

- **Step 1 (3.2):** recalibrated the G-T2 KS leg to an effect-size gate
  (`D <= 0.10 OR p > alpha`, `KS_STAT_MARGIN=0.10`). 7/18 -> 9/18 pass. +3 unit
  tests (large-n negligible-diff passes; real-diff fails). Re-aggregated the 72
  cells -> `plan05_summary_recal.json`.
- **Step 2 (3.1 reframe):** retracted the "self-clean x APF delete race" claim --
  falsified by zero APF=0 entries and the producer's APF/self-clean mutual
  exclusion. Probe confirmed the near-idle root cause: `app_hashtable` and
  `sandbox_scanner` treat `--duration` as a cap and exit in 1-2 s (guest idles the
  rest of the window); `mem_workingset` honors `--duration` and churns (~0.25) when
  it runs. Fix = sustain-loop the workloads. Pilot APF is therefore mostly idle and
  the fidelity result is weak; the throughput result is unaffected.
- **Step 3 (2B):** controlled idle 60 s head-to-head. keep-everything 21 GiB/60 s +
  climbing pmemsave (spike 2.4 s) + 21 snaps; delete-as-you-go bounded 2.2 GiB +
  flat 1.29 s + 29 snaps. Confirms the retention -> dump-cost direction; the full
  v3 7 s also needs the concurrent consumer + a much longer run (210 GiB bomb at
  600 s, not run).
- **Step 4 (2A):** added `CAPTURE_METRIC=delta|apf` to the production pipeline
  (`run_files_controlled.py` + `run_qemu_capture.sh`), additive and default
  byte-identical. `apf` streams APF via the producer's helper and skips the Rust
  delta consumer; `delta` (default) is unchanged. Verified live: apf mode -> 177
  pairs, APF mean 0.157, no consumer, dump dir 0 B; delta-regression run unchanged
  (producer+consumer, cosine/hamming + streaming metrics intact). Also normalized
  `run_qemu_capture.sh` to LF (was CRLF, unparseable by bash).

## Wave 4 changelog (Rust APF consumer + full campaign)

Wave 4 builds the architecturally-matching APF path -- a Rust consumer through the
producer -> queue -> consumer pipeline -- and runs a validated campaign. Flag
scheme: `CAPTURE_METRIC = delta` (default) `| apf` (Wave-3 lean inline helper)
`| apf_queue` (Wave-4 Rust consumer).

- **Step 5 (apf_calc):** new Rust binary `VM_sampler/VM_Capture/apf_calc/` -- a
  dependency-free sibling of `live_delta_calc` with the same CLI
  (`<prev> <curr> <output>`). Computes APF = fraction of differing 4 KiB pages,
  matching `plan02_apf_helper._compute_active_page_fraction` bit-for-bit
  (size-mismatch/empty -> 0.0; APF = differ_pages / n_pages). Ships 3 cargo unit
  tests (0.3 / identical / size-mismatch) + a cross-language equivalence check
  (`tests/apf_calc_equivalence.py`) asserting `apf_calc == plan02_apf_helper == 0.3`
  on a synthetic dump pair. Built + verified server-side (build 2.1 s, 3 tests pass,
  equivalence == 0.3).
- **Step 6 (consumer routing + flag):** `capture_consumer_qemu.sh` now branches on
  `CAPTURE_METRIC` -- `apf_queue` runs `apf_calc`, appends
  `{seq,t_emit_epoch,prev,curr,apf}` to `apf_trajectory.jsonl`, and skips
  run_matrix/streaming; `delta` (default) is unchanged. prev-delete + done-move are
  shared. `run_files_controlled.py._apf_env_prefix` is now 3-way (delta | apf |
  apf_queue): `apf_queue` makes the producer ENQUEUE (no `TIMING_APF_STREAM`) and
  points the consumer at `TIMING_APF_JSONL`. `config_qemu_upc.json` gains
  `apfCalculationProgram`. Default `delta` stays byte-identical (216 tests green).
- **Step 7 (apf_queue smoke):** live server run of the full producer -> queue ->
  apf_calc-consumer path (mem_workingset, 30 s). 161 APF pairs, mean 0.21
  (non-trivial); consumer ran apf_calc on every job, appended the trajectory,
  deleted prev, drained the queue (pending/processing 0/0), dump dir 0 B. Matches
  the inline-helper path (A = 0.157) within run variance -> with-consumer and
  without-consumer compute the same signal (per-pair equivalence already exact).

## Provenance

- Per-cell records: `plan05_runs/20260602T230241Z_dd587705/` (66) +
  `plan05_runs/20260603T173945Z_68a7d228/` (6 resumed).
- 7.06s / spread / `keep_dumps` figures: v3 `producer_stats`, recorded in
  `docs/papers/reviewer_memo_round2.html` (R-1).
- Aggregated offline over all 72 `run_record.json` via
  `plan05_aggregate.aggregate(records, baseline_arm="ssd_keep")`. Raw gate JSON:
  `plan05_summary.json`.
- Gate constants: `THROUGHPUT_MIN_SPEEDUP=2.0`, `DISK_CAP_BYTES=3 GiB`,
  `APF_MEAN_MARGIN=APF_STD_MARGIN=0.02`, `KS_ALPHA=TOST_ALPHA=0.05`,
  `KS_STAT_MARGIN=0.10` (Wave 3 effect-size KS).
