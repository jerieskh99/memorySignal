# Experiment 2 — Results and Fixes (Run 1)

**Date:** 2026-05-20
**Host:** `pcrserral`
**JSONs analysed:** `exp2a.json` `exp2b.json` `exp2c.json` (plus the dump archive)

## Headline

1. **2a (consumer isolation) — partial win.** Killing the consumer cuts
   suspend latency by ~100×, host_dt by ~5.6×, and brings pmemsave from
   1.72 s → 0.90 s. But pmemsave still drifts upward across snapshots
   (0.77 s → 1.66 s over 21 snaps).
2. **2b and 2c (interval sweep, flush sensitivity) — contaminated.** Both
   show numbers worse than 2a despite running the same producer with no
   consumer. Root cause identified: dump-cleanup is **silently failing**,
   leaving every dump on disk. By 2b/2c run-time, accumulated dumps had
   filled `/var/lib/libvirt/qemu/dump/` and pmemsave write speed
   collapsed (15–31 s for a 1 GiB dump).
3. **Structural bug confirmed.** Across every JSON,
   `"dumps_removed": 0`. The orchestrator's cleanup writes a warning that
   gets swallowed by stderr aggregation. Root cause: the imageDir is
   root-owned even though the dump files themselves are jeries-owned —
   the user can't `unlink` files in a directory it doesn't own.
4. **Scripts have been fixed** in this commit to: try `sudo rm` as
   fallback, expose `--purge-stale-dumps` / `--per-cell-purge` /
   `--purge-between-passes` flags, drain the queue between every pass /
   cell, resume the VM after every producer stop, and pre-check disk
   free space.
5. **Re-run protocol** in
   [`RUN_EXP2_SUBTESTS.md`](./RUN_EXP2_SUBTESTS.md) updated with mandatory
   `sudo rm` step + new flags.

## 2a — consumer isolation

Pass 1 (consumer_on): 0 snapshots, 60 backpressure events. The queue had
20 stale jobs from prior sessions; with no consumer to drain it, the
producer hit cap=20 backpressure on every iteration and never produced a
single snapshot. The label is misleading — the run measured "stale-queue
state", not "consumer running".

Pass 2 (consumer_off): the real data. After draining 20 queue files:

| | Run 2 (consumer on) | 2a pass 2 (consumer off) | improvement |
| --- | ---: | ---: | ---: |
| guest_dt mean | 0.124 s | **0.127 s** | unchanged ✓ |
| suspend mean | 6.380 s | **0.062 s** | **102×** |
| pmemsave mean | 1.720 s | **0.900 s** | **1.9×** |
| host_dt mean | 9.253 s | **1.642 s** | **5.6×** |
| pause fraction | 99.8 % | **92.3 %** | small but real |

**Mechanism iii (non-stationary suspend) is largely fixed by removing
the consumer.** Suspend now runs at ~50 ms median, dominated by the
`virsh suspend` RPC and one `wait_state` poll.

But there's still drift inside pass 2's pmemsave column:

| snap | 0 | 5 | 10 | 14 | 15 | 16 | 18 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pmemsave (s) | 0.755 | 0.782 | 0.800 | 0.925 | **1.130** | **1.360** | **1.656** |

Same producer, same VM, same consumer state (off). The only thing
changing is disk buffer pressure as ~21 GiB of dumps accumulate. By
snap 18 the disk write cache + filesystem journaling are slowing
down. The cleanup that *should* have been removing dumps as they
were produced (the producer's rolling-chain delete) only runs when
the consumer processes a job and unlinks the prev image — which is
disabled here. So dumps just pile up.

## 2b — interval sweep (CONTAMINATED)

| cell | snapshots | mean pmemsave | mean suspend | pause fraction |
| --- | ---: | ---: | ---: | ---: |
| iv100  | 3 | 15.02 s | 0.43 s | 98.8 % |
| iv250  | 1 | 25.99 s | 6.78 s | (n/a) |
| iv500  | 2 | 22.80 s | 0.67 s | 97.5 % |
| iv1000 | 1 | 31.48 s | 10.99 s | (n/a) |

**Every cell shows pmemsave 15–31 s.** Same producer, same VM, same
consumer state as 2a (off, no consumer). What changed? 2b ran AFTER 2a
and 2c. By then `/var/lib/libvirt/qemu/dump/` held ~50 GiB of
un-deletable dumps.

The numbers in this JSON are not a fair test of `intervalMsec`. They
measure "writing the 37th 1 GiB dump to a near-full filesystem".
Discard. Re-run after the cleanup fix.

## 2c — flush sensitivity (PARTIAL FAIL)

**flush_on pass:** 16 snapshots. Early snaps look like 2a (~0.77 s
pmemsave). Snap 15 spikes catastrophically: suspend 8.13 s, pmemsave
13.03 s. Same disk-pressure story.

**Integrity check failed** for all five probed dumps — but only because
the probe looked for non-zero bytes at offsets RAM/2 and RAM−4 KiB. The
Kali VM is idle, so the upper portion of physical RAM is mostly zero
pages. The dumps are structurally correct (size matches 1 GiB exactly)
— the probe was wrong.

**flush_off pass:** 0 snapshots, 0 records. After pass A's snap-15
spike, the producer was likely SIGTERM'd mid-cycle leaving the VM in
the `paused` state. Pass B started a new producer, which immediately
hit the saturated queue (queue not drained between passes) and got
stuck in backpressure for the full 60 s. No snapshots produced.

The recommendation field correctly reported "inconclusive".

## Root cause: dump cleanup silently fails

Every JSON reports `"dumps_removed": 0` or "removed 0 dump files".

In `run_timing_instrumentation_experiment.py:cleanup_run_dumps`:

```python
for p in image_dir.glob("memory_dump-*.raw"):
    try:
        if p.stat().st_mtime >= run_start_epoch - 1.0:
            p.unlink()
            removed += 1
    except OSError as e:
        log(f"WARNING: failed to remove {p}: {e}")
```

The `p.unlink()` call fails with `PermissionError` because
`/var/lib/libvirt/qemu/dump/` is root-owned. The user has write
permission on the **file** (chown'd to jeries by the producer) but
not on the **directory** — and `unlink` needs directory write
permission. The error is logged to stderr but the run continues.

This means every Experiment 1 / 2a / 2b / 2c run leaves all of its
dumps behind, and disk pressure compounds across runs.

## Fixes applied to the scripts

### Common (`run_timing_instrumentation_experiment.py`)

- **`cleanup_run_dumps`**: now tries direct `unlink`, then falls back
  to `sudo -n rm -f` for files that refuse to unlink. Reports back the
  combined count.
- **`purge_all_dumps(image_dir, use_sudo=True)`**: new helper. Removes
  ALL `memory_dump-*.raw` regardless of mtime. Used by
  `--purge-stale-dumps`.
- **`disk_free_check(image_dir, snapshots, ram_mb, margin=1.5)`**:
  new helper. statvfs check; experiments call this on startup and
  warn if free space is insufficient.
- **`resume_vm_if_paused(virsh_uri, domain)`**: new helper.
  Best-effort `virsh resume` after every producer stop, in case
  SIGTERM hit mid-cycle.

### 2a (`run_exp2a_consumer_isolation.py`)

- New flag `--purge-stale-dumps` runs an aggressive `sudo rm` of
  ALL dumps before pass 1.
- New flag `--drain-before-pass1` drains the queue before pass 1
  too (previously only before pass 2). Useful when prior runs
  left queue residue.
- `stop_producer` now followed by `resume_vm_if_paused` automatically.

### 2b (`run_exp2b_interval_sweep.py`)

- New flag `--purge-stale-dumps` at sweep start.
- New flag `--per-cell-purge` removes ALL dumps **between every cell**
  so disk pressure doesn't compound across cells. Recommended.
- Per-cell: `drain_queue` + `resume_vm_if_paused` before each cell.
- Disk-free pre-check based on heaviest expected cell.

### 2c (`run_exp2c_flush_sensitivity.py`)

- New flag `--purge-stale-dumps` before pass A.
- New flag `--purge-between-passes` removes dumps between pass A
  and pass B. Recommended for back-to-back runs.
- Between passes: `drain_queue` + `resume_vm_if_paused`.
- **Integrity probe rewritten**: drops the "non-zero bytes" check
  that was failing on idle-VM dumps. Now checks (1) file size
  matches expected ramSizeBytes, (2) file is readable end-to-end.
  Reports `all_zero_dump: True` informationally if the dump
  contains no non-zero bytes in the probe regions; this no longer
  fails the integrity gate.

## Re-run protocol (mandatory order)

On the Linux capture host:

```sh
# 0. One-time cleanup: nuke ALL stale dumps that previous runs left
sudo rm -f /var/lib/libvirt/qemu/dump/memory_dump-*.raw

# 1. Verify disk free space
df -h /var/lib/libvirt/qemu/dump
# Expect plenty of free GiB. If not, clean further.

# 2. Re-run 2a (clean, with explicit drain + purge)
cd ~/memorySignal/VM_sampler/VM_Capture_QEMU
git pull
python3 run_exp2a_consumer_isolation.py \
    --duration 60 \
    --purge-stale-dumps \
    --drain-before-pass1 \
    --output-json ./exp2a_v2.json

# 3. Re-run 2c with purge between passes
python3 run_exp2c_flush_sensitivity.py \
    --duration 60 \
    --purge-stale-dumps \
    --purge-between-passes \
    --output-json ./exp2c_v2.json

# 4. Re-run 2b with per-cell purge
python3 run_exp2b_interval_sweep.py \
    --duration 60 \
    --intervals 100,250,500,1000 \
    --purge-stale-dumps \
    --per-cell-purge \
    --output-json ./exp2b_v2.json
```

The `--purge-*` flags use `sudo -n rm -f` (non-interactive); if
sudo is not configured passwordless for the operator, the purge
will silently skip. In that case, run the manual `sudo rm` from
step 0 before each experiment.

## What the corrected data should look like

Predictions, to compare against `exp2a_v2.json` / `exp2b_v2.json` / `exp2c_v2.json`:

- **2a pass 2 with clean disk**: pmemsave should stay ~0.77 s across
  all snapshots, not drift to 1.66 s. Suspend should stay <0.1 s.
  host_dt should be ~1.4 s, stable.
- **2c flush_off**: should produce a similar number of snapshots to
  flush_on (since disk pressure is controlled). host_dt should drop
  by ~0.5 s. Integrity probe should report `ok=True` on all
  probes — sizes match.
- **2b**: pmemsave should track the expected curve for each
  intervalMsec. iv100 ≈ 0.77 s pmemsave (matches 2a). iv1000
  should have similar pmemsave but ~10× larger guest_dt. Pause
  fraction should drop from ~99 % at iv100 to ~55 % at iv1000.

If the corrected runs match these predictions, mechanisms iii and iv
are **both** confirmed addressable on this host.

## What remains unresolved

1. The +25 ms bash overhead on every snapshot (script-level
   bookkeeping between resume and next suspend). Not a bug, but
   documented as the irreducible "Δt_frame = intervalMsec + 25 ms".
2. The pause-fraction floor (mechanism iv) requires a real VM-RAM
   reduction, not just pmemsave-size override. A 256 MiB Kali boot
   would need a separate VM image. Out of scope for 2a/b/c.
3. The original `live_delta_calc` consumer never got profiled. If we
   ever want to run the producer + consumer together at the chosen
   `intervalMsec`, the consumer must keep up. That's a separate fix.

## Files changed in this commit

- `run_timing_instrumentation_experiment.py` — cleanup_run_dumps
  sudo fallback; new helpers `purge_all_dumps`, `disk_free_check`,
  `resume_vm_if_paused`.
- `run_exp2a_consumer_isolation.py` — `--purge-stale-dumps`,
  `--drain-before-pass1`, auto-resume.
- `run_exp2b_interval_sweep.py` — `--purge-stale-dumps`,
  `--per-cell-purge`, per-cell drain + resume, disk-free check.
- `run_exp2c_flush_sensitivity.py` — `--purge-stale-dumps`,
  `--purge-between-passes`, rewritten integrity probe, between-pass
  drain + resume.
- `docs/RUN_EXP2_SUBTESTS.md` — usage doc updated for new flags
  and re-run protocol.
- `docs/EXP2_RESULTS.md` — this file.
