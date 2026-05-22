# Plan 02 · Operator Runbook

**Purpose:** step-by-step launch + monitoring guide for the Plan-02 pilot. Use this when you sit down at `pcrserral` to run the pilot, want to check progress mid-run, recover from a crash, or hand off the host overnight.

**Audience:** operator on `pcrserral` (or any future capture host). Not a design doc — read [`FRAMEWORK_OVERVIEW.md`](../FRAMEWORK_OVERVIEW.md) and [`experiment_audit.md`](./experiment_audit.md) first if you need the *why*.

**Last verified:** 2026-05-22 against commit `ca50923`.

---

## 0 · Quick reference

| What you want | Command |
| ------------- | ------- |
| Check status | `python3 plan02_manifest.py summary <manifest.csv>` |
| See currently-running cell | `python3 plan02_manifest.py list <manifest.csv> --status running` |
| Tail heartbeat (live snap count) | `cat <output-dir>/work/<cell_id>/heartbeat.json` |
| Tail producer log | `tail -f <output-dir>/work/<cell_id>/producer.log` |
| List failed cells | `python3 plan02_manifest.py list <manifest.csv> --status failed` |
| Retry a failed cell | `python3 plan02_manifest.py force-status <manifest.csv> --cell-id <cid> --status pending` |
| Resume after crash | re-run the same `plan02_run.py` command — manifest survives |

---

## 1 · Prerequisites

Before any launch:

1. `git pull` on `pcrserral` to pick up latest `plan02_*.py`. Commit `ca50923` or later.
2. Smoke-test the tooling once on the host (no VM needed):
   ```sh
   cd /project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU
   python3 tests/test_plan02_smoke.py
   ```
   Expected: `Ran 15 tests in 0.0Ns ... OK`. Stop here if any test fails.
3. Confirm VM defined:
   ```sh
   virsh list --all | grep "Kali Jeries"
   ```
   Should show the domain. State doesn't matter; orchestrator brings it up.
4. Free disk on `/var/lib/libvirt/qemu/dump`:
   ```sh
   df -h /var/lib/libvirt/qemu/dump
   ```
   Need ≥ 5 GiB free. Steady-state dump load is ~2 GiB (self-clean rolls), but the orchestrator may briefly hold 3 dumps during transition.
5. `bc` installed (already confirmed since R2, but verify in case host was rebuilt):
   ```sh
   which bc
   ```

If any of these fail, **stop and fix before launching**. The pilot is overnight; you do not want to come back to a 4-hour failure mode.

---

## 2 · Build the manifest

The manifest is a CSV describing every cell the pilot will run. Build it once; the orchestrator consumes it. Re-runs of the same matrix produce the same `cell_id`s, so the manifest is reproducible.

### Canonical Plan-02 pilot manifest

```sh
cd /project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU
mkdir -p /project/homes/jeries/memorySignal/timing_runs/plan02_pilot

python3 plan02_manifest.py build \
    --workloads sandbox_ransom_batched mem_workingset_sweep_v2 \
    --intervals-ms 100 250 500 1000 2000 \
    --durations-s 60 120 300 \
    --replicates 3 \
    --block-size 24 \
    --output  /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv \
    --cell-output-dir /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells
```

Expected output:
```
wrote 94 rows to ...
   pending: 94
   warmup:  4
   total:   94
```

`94` = `5 iv × 3 durations × 2 workloads × 3 reps` + `4 warmup cells` (one per block).

### Variants

**Step 0.5 null-baseline only** (~1 h, runs idle Kali):
```sh
python3 plan02_manifest.py build \
    --workloads idle_kali \
    --intervals-ms 250 \
    --durations-s 300 \
    --replicates 3 \
    --no-warmup \
    --output  /project/homes/jeries/memorySignal/timing_runs/null_baseline/manifest.csv \
    --cell-output-dir /project/homes/jeries/memorySignal/timing_runs/null_baseline/cells
```

**Step 0 sensitivity probe** (~6 h, 18 cells):
```sh
python3 plan02_manifest.py build \
    --workloads sandbox_ransom_batched \
    --intervals-ms 100 500 2000 \
    --durations-s 300 \
    --replicates 2 \
    --no-warmup \
    --output  /project/homes/jeries/memorySignal/timing_runs/sensitivity/manifest.csv \
    --cell-output-dir /project/homes/jeries/memorySignal/timing_runs/sensitivity/cells
```

(Sensitivity probe varies `intervalMsec` and `window` to test coupling; window levels are set in the analyzer post-capture, not at capture time — capture only needs the iv axis.)

### Inspect the manifest before launch

```sh
python3 plan02_manifest.py summary /tmp/.../manifest.csv
python3 plan02_manifest.py list    /tmp/.../manifest.csv | head -20
```

Check that block-0 starts with a warmup row (`warm=True`) followed by ~23 real cells in randomized order.

---

## 3 · Dry-run validation

Before committing 4-5 host hours, dry-run the orchestrator. This validates the manifest and writes a session sentinel but does not touch the VM or producer.

```sh
python3 plan02_run.py \
    --manifest  /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv \
    --output-dir /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells \
    --dry-run
```

Expected output:
```
[plan02] session sentinel: .../session_sentinel.json
[plan02] manifest summary at start: {'pending': 94, ...}
[plan02] [dry-run] would execute pending cells; exiting
```

Inspect the sentinel:
```sh
cat /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells/session_sentinel.json | python3 -m json.tool
```

Should contain: `host_meta` (kernel, qemu version, git sha, vm image sha256), `summary_at_start`. If `git_sha` reads `"unknown"`, you launched from outside the git tree — check `pwd`.

---

## 4 · Launch real run

```sh
cd /project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU

python3 plan02_run.py \
    --manifest  /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv \
    --output-dir /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells \
    2>&1 | tee /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/orchestrator.log
```

The `tee` matters — `plan02_run.py` writes only to stderr; without `tee` you cannot inspect orchestrator messages after the session ends.

### What you should see per cell

```
[plan02] [3/94] running cell 85064daa6bfb (workload=sandbox_ransom_batched, iv=500, d=120s, rep=0, warm=True)
[plan02] [3/94] 85064daa6bfb -> ok (snapshots=95)
```

Each line means the manifest CSV has been updated atomically.

### Per-cell wall-clock

| cell duration | actual wall-clock |
| ------------- | ----------------- |
| 60 s  | ~80 s  (60 s + setup/teardown ~20 s) |
| 120 s | ~140 s |
| 300 s | ~320 s |

Total pilot wall-clock ≈ 4-5 h (94 cells, mixed durations).

### Useful launch options

| Flag | Purpose |
| ---- | ------- |
| `--max-cells N` | Stop after N cells. Useful for "run first block tonight, second tomorrow". |
| `--no-vm-start` | Trust caller to have started the VM (skip `virsh start`). |
| `--grace-stop-seconds N` | How long to wait between SIGTERM and SIGKILL on producer (default 10). Increase if pmemsave is slow on this host. |
| `--virsh-uri URI` | Override libvirt URI (default `qemu:///system`). |

---

## 5 · Live status checks

All these commands are **safe to run while a session is active**. They only read the manifest CSV and per-cell JSON files. They never block the orchestrator.

### A · Quick counts

```sh
python3 plan02_manifest.py summary  \
    /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv
```

Output (example mid-run):
```
running:  1
   ok:   12
 failed:  0
pending: 81
 warmup:  4
  total: 94
```

Sums to total. `running` should be exactly 1 during a session, 0 between cells, 0 after session done.

### B · Which cell is currently running

```sh
python3 plan02_manifest.py list  \
    /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv  \
    --status running
```

Output (example):
```
85064daa6bfb  blk= 0  sandbox_ransom_batched  iv= 500  d= 120s  rep=0  warm=True  st=running
```

The `cell_id` (first column) is what you'll use to drill into heartbeats and logs.

### C · Heartbeat (30 s refresh; snap count + disk + dirty pages)

```sh
CID=$(python3 plan02_manifest.py list \
    /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv \
    --status running | awk '{print $1}')

cat /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells/work/$CID/heartbeat.json | python3 -m json.tool
```

Or watch it refresh:
```sh
watch -n 5 "cat /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells/work/$CID/heartbeat.json | python3 -m json.tool"
```

Example payload:
```json
{
  "ts": "2026-05-22T22:14:33+00:00",
  "cell_id": "85064daa6bfb",
  "cell_index": 18,
  "total_cells": 94,
  "workload": "sandbox_ransom_batched",
  "interval_ms": 500,
  "duration_s": 120,
  "replicate": 0,
  "is_warmup": false,
  "snapshots_written": 41,
  "disk_free_gib": 138.7,
  "meminfo": {
    "MemAvailable_kB": 23456789,
    "Dirty_kB": 32100
  }
}
```

What to look at:
- `snapshots_written` grows monotonically — should hit ~`duration_s / (iv/1000 + 1.5)` by end of cell
- `disk_free_gib` should stay ≥ 5 GiB. Self-clean rolls dumps, so this is steady.
- `Dirty_kB` should stay under ~500 MB. If it climbs into the GBs, dirty-page pressure is back (Bug G regime); flag for investigation.

### D · Producer log

The producer's stderr stream lives in `work/<cell_id>/producer.log`. Tail it for any errors:
```sh
tail -f /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells/work/$CID/producer.log
```

Things to look for:
- `pmemsave: ...` — every snap. Should average ~0.77 s/snap.
- `backpressure` — should NOT appear (Plan 1c keeps queue drained).
- `error` or `failed` — investigate immediately.

### E · Orchestrator log

If you launched with `tee` (recommended):
```sh
tail -f /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/orchestrator.log
```

This is the cell-by-cell progression. Useful for seeing which cells already passed and which are pending.

### F · Failed cells

```sh
python3 plan02_manifest.py list \
    /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv \
    --status failed
```

Each row's `notes` column tells you why. Common reasons:
- `"crashed; auto-marked failed on restart"` — host rebooted, kernel oom-killed orchestrator, etc.
- `"FAIL: 0 snapshots completed"` — producer crashed without writing JSONL.
- `"orchestrator exception: ..."` — bug in `plan02_run.py`. File it.

### G · Completed cells (most recent first)

```sh
ls -lt /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells/cell_*.json | head -10
```

Each file is a schema-v2 per-cell JSON. Validate one:
```sh
python3 plan02_schema.py /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells/cell_<cid>.json
```
Should print `OK`.

### H · Session sentinel (host state at session start)

```sh
cat /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells/session_sentinel.json | python3 -m json.tool
```

One-shot capture of host kernel, qemu version, git sha, VM image hash, session start time. Audit forensics if you need to retrace conditions later.

---

## 6 · Crash recovery

The orchestrator is **crash-safe by design.** The manifest CSV is the source of truth. Atomic writes (temp-file + rename) guarantee it is never half-written.

### What counts as a crash

- Orchestrator process killed (SIGKILL, OOM, host reboot, ssh disconnect)
- VM crash mid-cell (rare; producer aborts, cell marked failed)
- Producer hang past `--grace-stop-seconds` (escalates to SIGKILL)

### Resume procedure

```sh
# 1) Inspect what state the manifest is in
python3 plan02_manifest.py summary  \
    /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv

# 2) Re-launch the exact same orchestrator command
python3 plan02_run.py \
    --manifest  /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv \
    --output-dir /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells \
    2>&1 | tee -a /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/orchestrator.log
```

On restart, the orchestrator:
1. Reads the manifest.
2. Finds any row with `status='running'` (left over from the crash) and marks it `'failed'` with a `"crashed; auto-marked failed on restart"` note.
3. Saves the manifest.
4. Picks the next `pending` row and continues.

You will see a startup log line like:
```
[plan02] WARNING: 1 'running' rows auto-marked 'failed' on restart
```

### Retrying a specific failed cell

If a cell failed because of a transient issue (host load spike, etc.) and you want to re-attempt it:

```sh
python3 plan02_manifest.py force-status \
    /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/manifest.csv \
    --cell-id <cell_id> \
    --status pending \
    --note "retry after host load normalized"
```

The next launch picks it up.

To re-attempt **all** failed cells in one shot:
```sh
for CID in $(python3 plan02_manifest.py list /tmp/.../manifest.csv --status failed | awk '{print $1}'); do
    python3 plan02_manifest.py force-status /tmp/.../manifest.csv --cell-id $CID --status pending --note "bulk retry"
done
```

---

## 7 · Disk budget

| What | Steady-state | Notes |
| ---- | ------------ | ----- |
| Dumps in `/var/lib/libvirt/qemu/dump/` | ~2 × 1 GiB = 2 GiB | self-clean (Plan 1) rolls them; never accumulates |
| Per-cell JSONs | 94 × ~5 KiB = 0.5 MiB | negligible |
| Per-cell JSONL (raw timestamps) | 94 × ~50 KiB = ~5 MiB | negligible |
| Per-cell producer.log | 94 × ~10 KiB = ~1 MiB | negligible |
| Total under `--output-dir/cells/` | ~10 MiB | small |

So the disk footprint is ~2 GiB dumps + ~10 MiB metadata. No 80 GiB hoarding. Self-clean does its job.

If you want to **keep dumps** for offline analysis (e.g., Plan 04), set `TIMING_SELF_CLEAN_KEEP_LAST=N` (not implemented; Plan 5c deferred). For now, dumps disappear as cells complete.

---

## 8 · After completion · analysis

Once the manifest is all `ok` or `failed`, run the analysis script:

```sh
cd /project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU

python3 plan02_analysis.py \
    --cells-dir /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells \
    --out-dir   /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/analysis \
    --metric f1_phase \
    --factor interval_ms \
    --per-workload
```

Outputs:

| File | Content |
| ---- | ------- |
| `tidy.csv` | One row per cell. Drop into pandas / Excel for visual inspection. |
| `acceptance_thresholds.json` | F1_null / CV_null derived from idle_kali cells (or audit defaults if none). |
| `anova_f1_phase_by_interval_ms.json` | F-stat + Welch t pairs for the iv main effect. |
| `iv_recommendations.json` | The Plan-02 deliverable — slowest passing iv per workload. |

Inspect:
```sh
cat /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/analysis/iv_recommendations.json | python3 -m json.tool
```

Expected shape (example):
```json
{
  "sandbox_ransom_batched": {
    "recommended_iv": 500,
    "passing_ivs": [100, 250, 500],
    "reason": "slowest iv that passed acceptance (cheapest pause fraction)"
  },
  "mem_workingset_sweep_v2": {
    "recommended_iv": 1000,
    ...
  }
}
```

If `recommended_iv` is `null` for a workload, no iv passed the acceptance criteria for it. Investigate before generalizing.

### Manual Welch t critical value lookup

The stdlib analysis script outputs Welch's t-statistic but not p-values (stdlib lacks a t-CDF). For two-sided α = 0.005 (Bonferroni-corrected over 10 contrasts at α = 0.05), with df ≈ 4 the critical |t| is roughly **4.6**. Any pair with |t| ≥ 4.6 is significant after correction.

For exact p-values, copy the JSON into a notebook and use `scipy.stats.t.sf`.

---

## 9 · Common failure modes

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| All cells `snapshots_completed=0` | producer not finding the VM (`virsh resume` failed) | Manually run `virsh start "Kali Jeries"` then re-launch with `--no-vm-start` |
| Some cells fail with `"orchestrator exception: ..."` | bug in `plan02_run.py` | Capture the traceback from `orchestrator.log`, file an issue |
| `disk_free_gib` drops below 2 in heartbeat | self-clean off (someone passed `--force-self-clean`?) | Stop session, check `TIMING_SELF_CLEAN` env, verify producer script line 142 |
| `backpressure_events > 0` in JSON | queue not drained at session start, OR consumer accidentally running | Check `queue_files_drained` in JSON's notes; verify no `consumer_qemu` process running |
| Heartbeat file empty / not updating | thread crashed silently | Restart orchestrator; this is a bug — file it. Manifest survives. |
| `git_sha` is `"unknown"` in sentinel | launched from outside git tree | Run from `VM_sampler/VM_Capture_QEMU/` directory, not from `/tmp` |
| Manifest gets out of sync (`running` row with no process) | crash during status write (very rare given atomic rename) | `force-status` it to `pending` and resume |

---

## 10 · Cleanup after a pilot completes

After the analysis JSONs are written and reviewed:

```sh
# Optional: archive the whole pilot directory
tar czf plan02_pilot_$(date +%Y%m%d).tar.gz \
    /project/homes/jeries/memorySignal/timing_runs/plan02_pilot

# Optional: prune the per-cell work directories (kept for forensics)
rm -rf /project/homes/jeries/memorySignal/timing_runs/plan02_pilot/cells/work
# The per-cell JSON files (cell_*.json) and the session sentinel stay.

# Dumps in /var/lib/libvirt/qemu/dump/ should already be gone (self-clean).
# Belt-and-braces:
sudo rm -f /var/lib/libvirt/qemu/dump/memory_dump-*.raw
```

The archive at `plan02_pilot_YYYYMMDD.tar.gz` is the full audit trail: manifest + every per-cell JSON + session sentinel + orchestrator log. Keep it for the thesis chapter.

---

## 11 · Handoff checklist

Before you walk away from a pilot session:

- [ ] Manifest summary shows `pending: 0` (or you've explicitly hit `--max-cells`)
- [ ] No `running` rows (every cell terminated cleanly)
- [ ] Failed cells documented in PM notes (with reasons from the `notes` column)
- [ ] Analysis ran and `iv_recommendations.json` written
- [ ] Archive tarball created
- [ ] `df -h /var/lib/libvirt/qemu/dump` shows dumps cleared

When that's all green, the pilot artifact is ready to feed Step 2 (Generalize). See [`experiment_audit.md`](./experiment_audit.md) Section 11 for the next phase.

---

## 12 · Where to read next

- **What this whole thing is for** — `FRAMEWORK_OVERVIEW.md`
- **Why we designed it this way** — `experiment_audit.md`
- **Original Plan 02 spec (pre-team-review)** — `02_interval_tuning_experiment.md`
- **The capture-pipeline calibration story (R1-R5)** — `../EXP2_RESULTS.md`
- **The HTML lab** — `../timing_experiment_1_lab.html`
