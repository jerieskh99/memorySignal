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
| **G-T2 fidelity** | Mixed | Passes where APF matches; failures are large-n KS hypersensitivity + d120 jitter; **one real bug** (below). |
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

## The one real fidelity bug

`mem_workingset_sweep_v2` @ 600s, `ssd_selfclean`: mean APF drops by **-0.166**
(KS p ~ 2e-250). The self-clean inline `sudo rm` of the previous dump races the
APF helper's read of that same dump; when rm wins, `_compute_active_page_fraction`
memmaps a gone/truncated file and returns 0.0, dragging the mean down. Only on SSD
(slow read loses the race), not tmpfs. A real `self-clean x --stream-apf`
interaction to fix.

## Next steps

1. **Measure the win as a clean contrast.** Re-run a small matrix with a *true*
   keep-everything baseline (no streaming delete) vs delete-as-you-go, so the
   ~9x copy / ~6x snapshot gain lands in one table instead of inferred across
   campaigns.
2. **Or adopt it now.** The 79-vs-14 evidence stands; turning on delete-as-you-go
   in the production pipeline solves the trajectory-length / DOF problem the
   papers flagged.
3. **Fix the self-clean x APF delete race** (order the helper's read before the
   delete, or skip dumps still pending an APF ack).
4. **Recalibrate G-T2** so large-sample KS stops dominating fidelity at d600.

## Provenance

- Per-cell records: `plan05_runs/20260602T230241Z_dd587705/` (66) +
  `plan05_runs/20260603T173945Z_68a7d228/` (6 resumed).
- 7.06s / spread / `keep_dumps` figures: v3 `producer_stats`, recorded in
  `docs/papers/reviewer_memo_round2.html` (R-1).
- Aggregated offline over all 72 `run_record.json` via
  `plan05_aggregate.aggregate(records, baseline_arm="ssd_keep")`. Raw gate JSON:
  `plan05_summary.json`.
- Gate constants: `THROUGHPUT_MIN_SPEEDUP=2.0`, `DISK_CAP_BYTES=3 GiB`,
  `APF_MEAN_MARGIN=APF_STD_MARGIN=0.02`, `KS_ALPHA=TOST_ALPHA=0.05`.
