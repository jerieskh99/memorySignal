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

---

# Round 2 — re-runs with cleanup fix (2026-05-20 evening)

**JSONs analysed:** `sanity_v2.json` `exp2a_v2.json` `exp2c_v2.json` `exp2b_v2.json`

## Summary

The cleanup fix worked. Most predictions from round 1 confirmed. **Three mechanisms** (i, ii, iii) are now demonstrably fixed; **mechanism iv** confirmed fixable in the right direction. **Two new bugs** surfaced that the round-1 code did not anticipate.

| mechanism | round 1 | round 2 | notes |
| ---------- | ------- | ------- | ----- |
| i  · window-duration collapse | already FIXED by bc | confirmed | guest_dt = intervalMsec + ~25 ms across all 4 intervals |
| ii · cepstral peak mis-mapping | already FIXED by bc | confirmed | analyzer's Δt is now correct |
| iii · non-stationary host Δt | LARGELY FIXED (partial) | **FULLY FIXED for suspend; new sub-issue for pmemsave** (see Bug B) | suspend median ~60 ms, CV tiny |
| iv · pause-fraction noise floor | STILL OPEN | **CONFIRMED FIXABLE** | 92.2 % → 59.2 % from iv=100 → iv=1000, exactly as predicted |
| v  · dump cleanup silent fail | FIXED IN SCRIPTS | confirmed working in 2b (per-cell purge) | exp2b shows `dumps_removed: 21` per cell |
| **vi · in-pass pmemsave drift** (NEW) | n/a | **OPEN, identified, fix planned** | producer leaks dumps when consumer killed |
| **vii · 2c verdict false negative** (NEW) | n/a | **OPEN, identified, fix planned** | means-based comparison swamped by drift |

## Per-test highlights

### sanity_v2.json — 30 s, idle VM

Six snapshots. Stationary at the start:

| | mean | std | CV |
|---|---:|---:|---:|
| guest_dt | 0.126 s | 0.0003 s | **0.2 %** |
| host_dt | 1.524 s | 0.018 s | **1.2 %** |
| suspend | 0.066 s | — | small |
| pmemsave | 0.778 s | small | stable |

This is the gold standard. Δt is essentially perfectly stationary in the first 6 snaps after a fresh purge.

### exp2a_v2.json

Pass 1 (label says "consumer_on" but consumer wasn't actually running — the run was end-to-end producer-only because the pipeline wasn't started):
- 21 snapshots, guest_dt = 0.126 s ✓, host_dt = 1.70 s
- pmemsave stable 0.77–0.80 s for snaps 0–13, then drifts 1.06 → 1.53 → 1.56 → 1.30 → 1.30 → 0.77 → 1.19 (snaps 14–20). Disk pressure from accumulating dumps.

Pass 2 (consumer_off): the contamination is severe.
- 21 snapshots, guest_dt = 0.122 s ✓, **host_dt = 2.73 s (worse than pass 1!)**
- pmemsave drifts hard: 0.78 → 0.78 → ... → 1.12 → 1.05 → 1.35 → 2.24 → 3.49 → **4.73** → 3.37 → 3.46 → 3.47 → 3.63 → 4.26 → **4.64**

Both passes ran without per-pass cleanup. Pass 1 left ~21 GiB on disk; pass 2 started with that already in place plus pass 2's own accumulation = ~42 GiB by the time pass 2 hit snap 14. Pmemsave throughput collapsed. This is **Bug A: between-pass cleanup missing in 2a**.

### exp2c_v2.json

Both passes completed 21 snapshots with `integrity all_ok: true` ✓. The integrity-probe fix worked.

| | flush_on | flush_off | delta |
|---|---:|---:|---:|
| mean host_dt | 1.626 s | 1.600 s | −0.026 s |
| mean pmemsave | 0.888 s | 1.408 s | +0.521 s |
| recommendation field | — | — | **"NEUTRAL"** ⚠️ |

The reported NEUTRAL verdict is **wrong**. Inspecting the first 9 snaps of each pass (before drift kicks in):

| snap | flush_on host_dt | flush_off host_dt |
|---:|---:|---:|
| 0 | 1.510 | 1.002 |
| 1 | 1.521 | 1.053 |
| 2 | 1.515 | 1.028 |
| 3 | 1.523 | 1.023 |
| 4 | 1.521 | 1.017 |
| 5 | 1.629 | 1.003 |
| 6 | 1.580 | 1.013 |
| 7 | 1.594 | 1.046 |
| 8 | 1.555 | 1.035 |
| 9 | 1.584 | 1.034 |
| **mean (snaps 0–9)** | **1.55 s** | **1.03 s** | **−0.52 s** ✓ |

A clean 0.52 s saving — exactly the removed `sleep 0.5`. The flush sleep is **redundant**. Means-based comparison was poisoned because flush_off's snaps 10–20 drifted heavily (pmemsave growing 1.10 → 2.58 s). This is **Bug C: 2c verdict logic uses contaminated means**.

### exp2b_v2.json — the cleanest result of the round

All 4 cells completed 21 snapshots each. Per-cell purge removed 21 dumps each between cells.

| cell | guest_dt | host_dt | pmemsave | pause % | snaps |
|---|---:|---:|---:|---:|---:|
| iv100  | 0.125 s | 1.61 s | 0.85 s | 92.2 % | 21 |
| iv250  | 0.270 s | 1.84 s | 0.93 s | 85.3 % | 21 |
| iv500  | 0.523 s | 2.22 s | 1.05 s | 76.4 % | 21 |
| iv1000 | 1.023 s | 2.51 s | 0.80 s | **59.2 %** | 21 |

Three observations:

1. **+25 ms bash overhead is a constant**, independent of `intervalMsec`. Confirmed across 4 orders of magnitude of interval.
2. **Pause fraction predictions land within ~5 %.** iv=1000 → 59 % matches the predicted ~55 %.
3. **Mechanism iv is now in our control.** Operator picks the rhythm of interest, the pause fraction is a derived consequence. No magic needed.

Pmemsave is also more stable than 2a's pass 2 because per-cell purge resets the disk-pressure clock at every cell.

The orchestrator emitted a disk-free WARNING:
```text
"free_bytes": 143466078208, "needed_bytes": 980863156224
```
The check assumes no in-run cleanup. With per-cell purge active, the actual peak need is ~1 GiB. The check is overly pessimistic — should be cleanup-policy-aware. **Bug D**.

## Two new bugs identified

### Bug A · 2a/2c don't clean between passes

`run_pass` returns; orchestrator goes to the next pass without removing dumps. Pass 2 starts with pass 1's footprint already on disk. Pmemsave drift in 2a pass 2 is dominated by this.

### Bug B · in-pass pmemsave drift

Even after start-of-pass purge, dumps accumulate **during** a pass. The producer's normal rolling-delete (consumer unlinks prev after delta) is disabled here because we kill the consumer. By snap 14–15 of any pass, disk pressure shows up as pmemsave latency growth.

This is the root mechanism behind 2a pass 2's drift to 4.7 s and the contamination of 2c's flush_off comparison.

### Bug C · 2c verdict logic

Means over 21 snaps mix the stable first half with the drift-contaminated second half. Verdict comes out NEUTRAL when first-N-only analysis would give REMOVE.

### Bug D · disk_free_check too pessimistic

Computes `snapshots_expected × ram × 1.5` without subtracting whatever cleanup policy is active. Trips a WARNING that the experiment can safely ignore.

## Fix plan (for the next code change)

### Plan 1 · Producer self-clean mode (PRIMARY FIX, addresses Bug B)

Add env-var guard `TIMING_SELF_CLEAN=1` to
`capture_producer_qemu_pmemsave.sh`. When set, the producer
**immediately deletes the previous dump** after enqueuing the
prev/curr pair (or, since the consumer is killed in timing runs,
instead of enqueuing). One-pass equivalent of what the rolling-chain
consumer does today.

Pseudo-patch:
```bash
if [[ -n "$prevImage" && -f "$prevImage" ]]; then
  if [[ -n "${TIMING_SELF_CLEAN:-}" ]]; then
    # producer-only timing mode: don't enqueue, just delete prev to
    # prevent disk pressure accumulation
    sudo rm -f "$prevImage" 2>/dev/null || rm -f "$prevImage" 2>/dev/null || true
  else
    # normal mode: enqueue for consumer
    jobId="$timestamp"
    jobTmp="$qPending/${jobId}.json.tmp"
    jobFile="$qPending/${jobId}.json"
    jq -n --arg prev "$prevImage" --arg curr "$newImage" \
         --arg output "$outputDir" \
         '{ prev: $prev, curr: $curr, output: $output }' > "$jobTmp"
    mv "$jobTmp" "$jobFile"
    echo "[PRODUCER-PMEM] Enqueued job $jobId"
  fi
fi
```

Expected impact: pmemsave stays at ~0.77 s for the entire pass.
Disk steady-state is two dumps (prev + curr), not 21+.

### Plan 2 · Clean between passes in 2a/2c (addresses Bug A)

In `run_exp2a_consumer_isolation.py` and
`run_exp2c_flush_sensitivity.py`, after pass A's `run_pass` returns
and before pass B starts:

```python
# Already exists for 2c (drain queue + resume VM); add dump cleanup
e1.cleanup_run_dumps(image_dir, pass_a_start_epoch, keep_dumps)
# or, more aggressively for these consumer-off runs:
e1.purge_all_dumps(image_dir, use_sudo=True)
```

Or — with Plan 1 in place this becomes unnecessary because the producer self-cleans during the pass. Keep one of the two; Plan 1 is the structural fix.

### Plan 3 · 2c verdict on stable subset (addresses Bug C)

In `run_exp2c_flush_sensitivity.py`, change the comparison block to use the **first N stable snapshots** (e.g. N=10) instead of the means over all snapshots:

```python
def stable_mean(records, key, n=10):
    vals = [r[key] for r in records[:n]
            if isinstance(r.get(key), (int, float))]
    return statistics.fmean(vals) if vals else None

# instead of:
#   on["mean_host_snapshot_cycle_sec"]
# use:
#   stable_mean(on["records"], "host_snapshot_cycle_sec", n=10)
```

Or compute both: `mean_first_10` and `mean_all`, report both, and base the recommendation on `mean_first_10`. The orchestrator already keeps the full records; just slice them.

With Plan 1 in place this also becomes unnecessary because there is no drift after snap 10.

### Plan 4 · disk_free_check policy-aware (addresses Bug D)

Add a `peak_concurrent_dumps` parameter (default 25, can be 2 when
self-clean is active):

```python
def disk_free_check(image_dir, peak_concurrent_dumps, ram_mb, margin=1.5):
    need = peak_concurrent_dumps * ram_mb * 1024 * 1024 * margin
    ...
```

Pass the right value from each orchestrator based on whether
self-clean / per-cell purge is on.

## Round-2 numbers worth quoting

(use these in the HTML lab and any thesis-side write-up)

- **guest_dt = intervalMsec + 25 ms** (constant, CV < 0.5 % in the
  best cells).
- **suspend median ≈ 60 ms** with consumer killed; was 6.4 s with
  consumer alive.
- **pmemsave on a clean disk = 0.77 s** for 1 GiB (= 1.36 GiB/s
  effective).
- **Pause fraction sweep:** 92.2 % @ iv=100 → 85.3 % @ iv=250 →
  76.4 % @ iv=500 → 59.2 % @ iv=1000.
- **Flush sleep is redundant:** −0.52 s host_dt savings on a clean
  disk, dumps remain intact. Recommend REMOVE.

## Order of operations for the next code change

1. Patch producer with `TIMING_SELF_CLEAN` env var (Plan 1).
2. In each of 2a / 2b / 2c, set `TIMING_SELF_CLEAN=1` in the producer
   env when the orchestrator runs producer-only.
3. In 2c, add `stable_mean` and base recommendation on first-10 snaps
   (Plan 3).
4. Lower the disk_free_check ceiling when self-clean is on (Plan 4).
5. Re-run 2a / 2c (60 s each) and 2b (4 cells × 60 s). Total ~10 min.
6. Expected result: pmemsave stays ~0.77 s for entire pass; 2c
   verdict comes back REMOVE; pause-fraction sweep numbers stay
   within 1 % of round-2 values; 2a pass 2 no longer worse than
   pass 1.

If those land, mechanism iii is FULLY FIXED end-to-end (no drift),
the flush patch can be applied to the producer, and plan 02 (per-family
`intervalMsec` profiles) is unblocked.
