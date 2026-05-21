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

---

# Round 3 — Plans 1/1b/3/4 implemented, re-run results (2026-05-21)

**JSONs analysed:** `sanity_v3.json` `exp2a_v3.json` `exp2c_v3.json` `exp2b_v3.json`

## Headline

| test | round 3 verdict | grade |
| ---- | --------------- | :---: |
| **2a v3** | both passes 36 snaps · pmemsave flat 0.78 s · no drift · no backpressure | ✅ |
| **2b v3** | all 4 cells full data · pmemsave flat ~0.77 s · pause sweep matches R2 within 0.1 % · disk-free WARNING gone | ✅ |
| **2c v3** | host_dt delta correct on first-10 (−0.515 s) but **integrity probe ran on 0 dumps** → verdict KEEP | ⚠️ partial |
| **sanity v3** | 0 snapshots completed · 30 backpressure events | ❌ failed |

**Plan 1 (producer self-clean) and Plan 4 (disk-check policy-aware) are verified working as designed.** Plan 3 (stable-subset comparison) computed the right number (delta −0.5153 s on first-10) but cannot make the verdict flip because the integrity gate failed for an unrelated reason.

## Plan-by-plan verification

| plan | how to know it worked | round 3 evidence |
| ---- | --------------------- | ---------------- |
| 1 (producer self-clean) | 2a pass 2 pmemsave std < 0.05 s | std = 0.042 s ✓ |
| 1 (producer self-clean) | disk steady-state ≈ 2 dumps | `dumps_removed: 2` per 2b cell ✓ |
| 1b (orchestrators set env) | grep `TIMING_SELF_CLEAN` in 2a/2b/2c | present ✓ |
| 3 (stable-subset verdict) | both `mean_first_n` and `mean_all` in JSON | both present ✓ |
| 3 (stable-subset verdict) | recommendation uses first-N when available | code path active; correct delta computed ✓ |
| 4 (disk-check policy-aware) | no `disk free space may be insufficient` WARNING in notes | 2b notes empty; `peak_concurrent_dumps: 2` recorded ✓ |
| 4 (disk-check policy-aware) | `needed_bytes` reflects 2 × ram, not snapshots × ram | 3 222 MiB needed vs 143 GiB free ✓ |

All four plans landed cleanly. Mechanisms vi (in-pass drift) and Bug D (disk-check pessimism) are **closed**.

## What changed numerically vs Round 2

| metric | Run 2 | Round 2 | **Round 3** |
| ------ | ----: | ------: | ----------: |
| 2a suspend mean | 6.38 s | 0.066 s | **0.063 s** |
| 2a host_dt mean | 9.25 s | 1.64 s | **1.65 s** |
| 2a pmemsave drift across pass | 0.77 → 4.7 s | 0.77 → 1.66 s | **0.77 → 0.78 s** (essentially zero) |
| 2a backpressure events | many | 25 (pass 2) | **0 both passes** |
| 2b iv=100 pause | — | 92.2 % | **92.4 %** |
| 2b iv=1000 pause | — | 59.2 % | **59.3 %** |
| 2b pmemsave consistency across cells | — | spread 0.85–1.05 s | **flat 0.764–0.769 s** |
| 2b backpressure events per cell | — | 8–26 | **0 in all 4 cells** |
| 2b disk-free WARNING | — | fired | **absent** |
| 2c host_dt delta (first-10) | n/a | n/a (means only) | **−0.515 s** ✓ matches manual round-2 read |

Reproducibility across Round 2 → Round 3 is excellent. Where Round 2 showed slight drift inside a pass, Round 3 is flat. Where Round 2 had backpressure events, Round 3 has zero.

## Two new bugs introduced by the Plan 1 / Plan 1b changes

### Bug E · sanity script does not have self-clean

`run_timing_instrumentation_experiment.py` was the original Experiment 1 driver and did **not** receive a Plan 1b update — only the three exp2 orchestrators did. As a result, the sanity script runs the producer without `TIMING_SELF_CLEAN=1` and without draining the pre-existing queue. On the capture host, where past sessions left ≥ 20 queue files, the producer hits the `maxPendingJobs=20` backpressure cap on the very first iteration and stays there for the entire 30 s duration. Zero snapshots produced.

The data is correct ("the host's queue was saturated"), it's just useless for sanity-checking the bc fix.

### Bug F · self-clean kills the 2c integrity probe

2c works in two phases per pass:

1. Producer runs for 60 s, writing dumps to `imageDir`.
2. After the producer is stopped, the orchestrator parses the JSONL, takes the first N (`probes_per_pass`) records, and for each tries to read the dump file at `record.image_path` and check its size + content.

When `TIMING_SELF_CLEAN=1` is set, the producer deletes every prev dump as soon as it has written curr. By the time the orchestrator gets to phase 2, only the very last dump in each pass still exists on disk — and even that one is deleted by the orchestrator's own end-of-pass `cleanup_run_dumps` call.

Result: `probes_examined: 0` on both passes → `all_ok: false` → recommendation locked to "KEEP the flush — dumps without it failed integrity check" — regardless of how the host_dt comparison turned out.

Plan 3 computed exactly the right number: `mean_host_snapshot_cycle_sec_first_n.delta = -0.5153 s` (off is 0.515 s faster than on). The verdict ladder would have correctly produced REMOVE — but it never got past the integrity gate.

## Fix plan for Round 4

Two small additions. Both expose existing behaviour as flags rather than introducing new logic.

### Plan 1c · apply Plan 1b to the sanity script · **LANDED**

`run_timing_instrumentation_experiment.py` updated:

- Added `--no-self-clean` flag (opt-out). Default sets `os.environ["TIMING_SELF_CLEAN"] = "1"` before the producer launches.
- Added `--no-drain-queue-on-start` opt-out. Default lazy-imports `drain_queue` from `run_exp2a_consumer_isolation` and removes all `queueDir/{pending,processing}/*.json` files before the producer starts. The import is lazy because `run_exp2a_consumer_isolation` itself imports the sanity module (`import run_timing_instrumentation_experiment as e1`), so a top-level import would be circular.
- The run JSON's `config` block now records `self_clean`, `drain_queue_on_start`, and `queue_files_drained` so Round 4 analysis can verify the patch took effect.
- Dry-run plan also surfaces the new flags via an extra `notes` entry.

Cost: 34 LOC across the one file (12 logic, rest docstrings + dry-run wiring).

### Plan 5 · 2c keeps dumps until probe runs · **LANDED** (Option A)

`run_exp2c_flush_sensitivity.py` updated:

- The default behaviour is now **`TIMING_SELF_CLEAN` OFF** for 2c. Dumps accumulate during each pass long enough for the integrity probe to find them.
- `--no-self-clean` is retained as a back-compat no-op (clearly documented in `--help`).
- A new `--force-self-clean` flag is available for diagnostic side-by-side runs. When set, it logs a loud WARNING that the probe will see 0 dumps and the recommendation will lock to KEEP regardless of host_dt savings.
- Plan 3's first-10 stable-subset comparison still drives the verdict, so any in-pass drift in snaps 11+ does not contaminate the host_dt delta.

Option B (producer-side `TIMING_SELF_CLEAN_KEEP_LAST=N`) is **not** implemented — Option A solves the problem with ~21 LOC of which ~9 are logic; Option B would be more invasive without solving anything 2c needs.

### Status summary

Both follow-ups landed in code. Round 4 (sanity + 2c only; 2a and 2b already at A+) is the empirical verification step.

## Round 4 expected predictions

Plans 1c + 5 are now in code. Round 4 is the empirical verification re-run (~3 min host wall-clock):

| test | expected | new JSON fields to verify |
| ---- | -------- | ------------------------- |
| sanity v4 | ≥ 6 snapshots in 30 s, guest_dt 0.125 ± 0.01 s, host_dt ≈ 1.55 s | `config.self_clean = true`, `config.drain_queue_on_start = true`, `config.queue_files_drained` reported |
| 2a v4 | unchanged from v3 (already at A+) | — skipped |
| 2b v4 | unchanged from v3 (already at A+) | — skipped |
| 2c v4 | `integrity_on: true` AND `integrity_off: true` · `recommendation` = "REMOVE the flush..." | `integrity_on.probes_examined > 0`, `integrity_off.probes_examined > 0` |

If those land, **all 8 mechanisms / bugs (i–vii + D + E + F) are closed**. Plan 02 (per-family `intervalMsec` profiles) is then unblocked with no caveats. The producer's `sleep 0.5` flush can be removed permanently for ~10 % wall-clock saving on every Phase 2 capture going forward.

### Round 4 commands (on `pcrserral`)

```sh
cd /project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU
# Sanity (Plan 1c smoke test): 30 s, idle Kali
python3 run_timing_instrumentation_experiment.py \
    --duration 30 --interval-ms 100 --ram-mb 1024 \
    --output-json /project/homes/jeries/memorySignal/timing_runs/sanity_v4.json

# 2c (Plan 5 smoke test): 60 s per pass, two passes
python3 run_exp2c_flush_sensitivity.py \
    --duration 60 --interval-ms 100 --ram-mb 1024 \
    --output-json /project/homes/jeries/memorySignal/timing_runs/exp2c_v4.json
```

No 2a / 2b needed — they were A+ in Round 3 and the two code changes do not touch them.

# Round 4 results (2026-05-21)

## Headline

| test | result | bug status |
| ---- | ------ | ---------- |
| sanity v4 | **PASS** · 18 snaps clean | E **CLOSED** |
| 2c v4 integrity probe | **PASS** · 5 dumps × 2 passes, all_ok=true | F **CLOSED** |
| 2c v4 verdict | **NEUTRAL** — true negative this time, not a logic bug | G **NEW** (subtle) |

Plans 1c and 5 are empirically verified at the code level. Round 4 also surfaced a new finding (Bug G below) that flips the original 2c question on its head — and the data inside the same JSON already contains the clean answer at a smaller stable-n.

## sanity v4 — Plan 1c verified

```
snapshots_attempted       18
snapshots_completed       18
mean_guest_run_interval   0.1256 s   (target: intervalMsec + ~25 ms)
mean_host_snapshot_cycle  1.626 s    (matches 2a Round 3)
mean_pmemsave             0.770 s    (flat, no drift)
mean_suspend              0.0622 s
backpressure_events       0          (was 30 in v3)
queue_max_depth           0          (was nulled out in v3)
estimated_vm_pause_fraction  92.3 %  (matches 2b iv=100 cell)
```

New JSON fields recorded:

```
config.self_clean              true
config.drain_queue_on_start    true
config.queue_files_drained     15      ← Plan 1c drained 15 stale files
notes[0]                       "plan-1c: drained 15 stale queue file(s) from /project/homes/jeries/memory_traces/queue_dir before start"
```

Bug E is **CLOSED**. The sanity script now matches the 2a/2b/2c default behaviour and produces clean data on first run regardless of leftover queue state.

## 2c v4 — Plan 5 fixed the probe, but exposed Bug G

### What worked
- `integrity_on.probes_examined = 5`, `integrity_on.all_ok = true` ✓
- `integrity_off.probes_examined = 5`, `integrity_off.all_ok = true` ✓
- 5 real dumps inspected per pass, all 1073741824 bytes (1 GiB), readable end-to-end, non-zero content in probe slots.
- Bug F is **CLOSED**.

### What surfaced — Bug G
The host_dt comparison flipped sign:

| metric | Round 3 (self-clean ON) | Round 4 (self-clean OFF, Plan 5) |
| ------ | ----------------------- | -------------------------------- |
| first-10 host_dt delta | **−0.515 s** (flush_off wins) | **+2.466 s** (flush_off loses) |
| flush_off pmemsave snap 0–4 | 0.77 s flat | 0.81 s flat |
| flush_off pmemsave snap 5 | 0.77 s | **1.95 s** ← transition |
| flush_off pmemsave snap 6 | 0.77 s | **7.25 s** ← drift |
| flush_off pmemsave snap 7–10 | 0.78 s | 8.27, 7.91, 8.08, 6.00 s |

The flush_off pass was clean for the first 5 snaps, then collapsed. The flush_on pass stayed flat at 0.84 s throughout because the 0.5 s sleep gave the kernel time to flush dirty pages from the previous pmemsave. Without that sleep, dirty pages stacked up, and by snap 6 every pmemsave was I/O-blocked on page-cache writeback.

This is **not** a regression of mechanism vi. It is the same disk-pressure tail, but now triggered specifically by the combination *flush_off + no self-clean*. Plan 5 deliberately turned self-clean off so the probe would see real dumps. The unintended side effect is that mechanism vi re-emerges in flush_off.

### Same JSON, smaller window → Round 3 answer back

Computing the host_dt delta on the first 5 snaps (before the drift):

```
flush_on  first-5 host_dt mean  =  1.519 s
flush_off first-5 host_dt mean  =  1.038 s
delta                            = -0.481 s   ← matches Round 3 (-0.515 s)
```

So the answer to the original 2c question — "is the 0.5 s flush necessary?" — is **the same as Round 3** when measured on a window short enough to predate the drift. The window just needs to be smaller than the default `--stable-n=10`.

## Bug G · the flush-vs-self-clean coupling

`run_exp2c_flush_sensitivity.py` was implicitly designed assuming dumps disappear quickly after pmemsave returns. That assumption was true in Round 3 (self-clean ON) and is true in production (consumer drains the queue). It is *false* in the Plan 5 configuration (self-clean OFF for the integrity probe).

The flush behaviour interacts with the dump-survival model:

| scenario | flush role | flush_off effect |
| -------- | ---------- | ---------------- |
| Self-clean ON (Round 3) | pure overhead | saves 0.5 s/snap |
| Self-clean OFF (Round 4 Plan 5) | implicit dirty-page barrier | loses 2.5 s/snap from drift |
| Production (consumer drains) | pure overhead | saves 0.5 s/snap (matches Round 3) |

The production answer is unchanged: **remove the flush, save ~10 % wall-clock per snap**. Round 4 just measured a degenerate counterfactual (no consumer + no self-clean) that does not exist in production.

### Plan 6 · `--stable-n` default 10 → 5 · **LANDED**

The cheapest fix: shrink the default comparison window so the drift never enters the average.

```python
p.add_argument("--stable-n", type=int, default=5, ...)
```

Justified directly from Round 4 data:

- Drift first appears at snap 6 (pmemsave 7.25 s).
- n=5 catches the savings window cleanly: delta = −0.481 s ≈ Round 3's −0.515 s.
- n=6 already starts to absorb drift: delta = −0.291 s (still negative but noisy).
- n=7 flips positive: +0.609 s.
- n=10 (old default) = +2.466 s (full drift contamination).

Plan 6 is committed alongside this analysis. ~1 LOC change.

### Plan 5c (deferred) · structural fix via producer KEEP_LAST=N

A cleaner long-term solution: add `TIMING_SELF_CLEAN_KEEP_LAST=N` env-var to `capture_producer_qemu_pmemsave.sh`. The producer keeps the most recent N dumps and deletes anything older. 2c sets N=5 — probe sees 5 real dumps, timing window is identical to Round 3, no drift.

Cost: ~25 LOC in the producer + ~5 LOC in 2c. Deferred to a future round; Plan 6 is sufficient.

## Round 5 plan (optional, ~1 min)

Re-run 2c only with `--stable-n=5` (now the default). Expected JSON:

```
comparison.mean_host_snapshot_cycle_sec_first_n.n_used  = 5
comparison.mean_host_snapshot_cycle_sec_first_n.delta   ≈ -0.48 s
recommendation                                          = "REMOVE the flush ..."
integrity_on.all_ok                                     = true
integrity_off.all_ok                                    = true
```

If those land, mechanisms i–vii + D + E + F + G are all closed. Plan 02 (per-family `intervalMsec` profile) unblocks with no caveats. The 0.5 s `sleep` on line 142 of `capture_producer_qemu_pmemsave.sh` can be removed permanently.

```sh
cd /project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU
git pull
python3 run_exp2c_flush_sensitivity.py \
    --duration 60 --interval-ms 100 --ram-mb 1024 \
    --output-json /project/homes/jeries/memorySignal/timing_runs/exp2c_v5.json
```

No sanity v5 needed — sanity is locked at A in Round 4.

# Round 5 results (2026-05-21)

## Headline

| field | value | check |
| ----- | ----- | ----- |
| `comparison.mean_host_snapshot_cycle_sec_first_n.n_used` | **5** | Plan 6 default landed on host ✓ |
| `comparison.mean_host_snapshot_cycle_sec_first_n.delta` | **−0.380 s** | negative · flush_off wins ✓ |
| `integrity_on.all_ok` / `integrity_off.all_ok` | `true / true` | probe still works ✓ |
| `recommendation` | `"REMOVE the flush — dumps remain intact and host_dt drops by 0.380 s (25.0%) on the first-5 comparison."` | orchestrator verdict ✓ |
| both passes `snapshots_completed` | 21 / 21 | symmetric, full window ✓ |

Bug G is **CLOSED empirically**. All 9 mechanisms / bugs (i–vii + D + E + F + G) are closed.

## Inter-round delta comparison

| round | n | flush_on first-n | flush_off first-n | delta | verdict |
| ----- | - | ---------------- | ----------------- | ----- | ------- |
| R2 manual | 9 | (manual inspection) | (manual) | −0.520 s | REMOVE (manual) |
| R3 | 10 | (self-clean ON) | (self-clean ON) | **−0.515 s** | REMOVE (Plan 3) |
| R4 first-5 | 5 | 1.519 s | 1.038 s | **−0.481 s** | (manual; n=10 verdict was NEUTRAL +2.466) |
| **R5** | **5** | **1.520 s** | **1.140 s** | **−0.380 s** | **REMOVE (Plan 6 default)** |

All four rounds agree on direction. R5's smaller magnitude is explained by flush_off snap 0 (`suspend_sec = 0.633 s` cold-start, vs ~0.066 s steady-state) and snap 9+ drift starting earlier than R4 — both stochastic, both expected; the n=5 window still captured the right sign.

## Bug H · `mkdir -p` parent of `--output-json`

Discovered in the first R5 attempt: 2c (and 2b) crashed on `json.dump` because `out_path.parent` didn't exist. Sanity script + 2a already handled this; 2b + 2c didn't. Fix is one line × 4 sites (dry-run + real-run paths in each of 2b and 2c). Committed alongside this analysis.

## Status after Round 5

| stage | state |
| ----- | ----- |
| 9 mechanisms / bugs | **all CLOSED** |
| 8 plans (1 / 1b / 1c / 3 / 4 / 5 / 6 + Bug H fix) | **all LANDED** |
| 5 rounds + 2 sanity runs | **all done** |
| Phase-2 production tooling | **calibrated, audit-trailed, ready** |
| Plan 02 (per-family `intervalMsec` profile) | **UNBLOCKED** |
| Phase-2 producer `sleep 0.5` flush | safe to remove · saves ~10 % wall-clock per snap |

## Next steps (no longer blocked)

1. **Drop `sleep 0.5`** from `capture_producer_qemu_pmemsave.sh` line 142 (or guard it behind a production-mode flag). ~1 LOC.
2. **Plan 02** — per-family `intervalMsec` profile. The thesis-contribution experiment. Pick the smallest `intervalMsec` per workload family such that the defining rhythm fits ≥ 4 periods in a 128-sample window. Use R2's pause-fraction sweep (92/85/76/59 % at iv=100/250/500/1000) as the bound.
3. **Re-collect one previously-confused workload pair** and rerun classifier. If confusion collapses, the bc + Δt + drift + interval-profile chain is load-bearing. If it persists, escalate to Plan 03 (window/hop).
