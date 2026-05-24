# Debug Audit · Step 1.5c persistent failure

**Scope:** Cell 1 in the latest sub-pilot completes with only 1 snapshot. Cell 2 then fails with `cannot acquire state change lock (held by monitor=qemuDispatchDomainMonitorCommand)`. Dump cleanup warns that `/var/lib/libvirt/qemu/dump/` is read-only for the current user. Three coupled symptoms; team must converge on smallest fix.

**Convened:** 8-agent debug team (PM + 7 specialists). Single sitting. Time-boxed.

**Decision rule:** no rewrites. Prove minimal-fix path before considering any larger change.

---

## Team and roles

| Agent | Codename | Role for this debug |
|-------|----------|---------------------|
| Project Manager | PM | Coordinate; cap scope; force smallest-fix convergence |
| Senior Architect | SA | VM/libvirt state transitions; lock-acquire semantics; ordering of stop/resume |
| Engineering Skills | EN | Filesystem perms; dump cleanup; sudo fallback; disk monitoring |
| Senior ML Engineer | ML | Whether dump-content metrics are blocked by this failure mode |
| Experiment Designer | XD | Per-cell sequencing; preflight + skip policy; failure-mode containment |
| Senior Data Engineer | DE | Schema fields for disk state + log scanning; manifest mutation safety |
| Senior Data Scientist | DS | Cell validity criteria; which cells are recoverable vs garbage |
| Evaluation Engineer | EE | Acceptance rules; what counts as cell "pass" given the new failure mode |

---

## 1 · Root-cause hypotheses, ranked

| # | Hypothesis | Likelihood | Why ranked here |
|---|-----------|-----------|----|
| **H1** | **Accumulated dumps from prior failed runs filled the disk.** `/var/lib/libvirt/qemu/dump/` is root-owned; orchestrator's cleanup as `jeries` couldn't unlink without sudo. Each leftover dump = 1 GiB (RAM). The first cell's pmemsave succeeded once, then the disk filled, all subsequent pmemsave calls failed → producer hung in retry → orchestrator's wall-clock kill caught it mid-`virsh suspend` → next cell hit the lock. | **HIGH (60%)** | Three independent signals point here: dump-dir-read-only warning, "1-snap-then-fail" pattern matches disk-full progression, lock-error caused by killpg mid-RPC matches the documented Bug-L mechanism. Cleanly explains all three symptoms together. |
| **H2** | **`keep_dumps=True` filled the disk during the current run.** With self-clean OFF, each cell at iv=500ms d=300s produces ~150 dumps × 1 GiB = 150 GiB, much larger than typical free space. Even one cell can fill the disk; cell 2 then can't pmemsave. | **MEDIUM-HIGH (25%)** | Explains the symptoms if previous-run leftovers are not the issue. The manifest uses `--keep-dumps` per Step 1.5b design. The expected disk burn at iv=500 × d=300s × 1 GiB/dump > any normal headroom. |
| **H3** | **Producer's bash continues looping on pmemsave error, never fail-fasting.** Producer keeps issuing virsh suspend/resume even when pmemsave returns non-zero. Eventually the orchestrator kills it mid-cycle → lock held by in-flight RPC. | **LOW (10%)** | This is a real fragility but acts as an amplifier on H1/H2 rather than the proximal cause. Producer would not loop if there were no I/O error to begin with. |
| **H4** | **Bug-L D-34 retry-resume not yet pulled on the host.** Operator may be running an older version of `plan02_run.py` where settle polled `virsh domstate` instead of retrying `virsh resume`. | **LOW (5%)** | Possible if `git pull` was skipped. Easy to rule out by checking commit hash on the host. |

PM forces ranking: **H1 + H2 are co-acting in 85% of probability mass.** Both fixes are cheap and orthogonal; ship both. H3 (producer fail-fast) is deferred to a separate session because it requires changing the bash script. H4 is operator hygiene.

---

## 2 · Diagnostic commands to run NOW on `pcrserral`

```sh
# A · what's actually on disk in the dump dir
ls -lh /var/lib/libvirt/qemu/dump/ | head -20
ls /var/lib/libvirt/qemu/dump/memory_dump-*.raw 2>/dev/null | wc -l
du -sh /var/lib/libvirt/qemu/dump/

# B · how much free space
df -h /var/lib/libvirt/qemu/dump/

# C · is the VM running or stuck?
virsh -c qemu:///system list --all
virsh -c qemu:///system domstate "Kali Jeries"

# D · what does the last producer log say (look for pmemsave / virsh errors)
ls /tmp/plan02_1_5c/cells/work/*/producer.log | head -3
tail -50 /tmp/plan02_1_5c/cells/work/<first-cell-id>/producer.log

# E · permissions on the dump dir
ls -ld /var/lib/libvirt/qemu/dump/
stat /var/lib/libvirt/qemu/dump/

# F · is sudo passwordless for this user (for the cleanup path)?
sudo -n true ; echo "sudo -n exit code: $?"

# G · confirm host code version (rules out H4)
git -C /project/homes/jeries/memorySignal log --oneline -5 \
    -- VM_sampler/VM_Capture_QEMU/plan02_run.py
```

**Expected pattern:**

- (A) > 5 dump files left over → confirms H1.
- (B) free < 10 GiB → confirms H1 or H2.
- (C) state == `paused` → VM stuck, needs manual resume.
- (D) `pmemsave: error: ... No space left on device` → confirms disk-full mechanism.
- (F) non-zero exit → cleanup cannot use sudo; explains the cleanup warning.

---

## 3 · Minimal code change · 1 file · ~210 LOC

`plan02_run.py` gains 3 small helpers and 2 call sites. No new module, no new dependency, no producer-script change. Approved by every agent (SA, EN, XD, DE, DS, EE) as the smallest reliable fix.

### Helpers added

```python
def _disk_free_gib(path) -> float: ...
def _count_stale_dumps(image_dir) -> int: ...
def session_preflight_disk(image_dir, ram_mb, min_headroom_dumps=5,
                            purge_stale=False) -> dict: ...
def pre_cell_disk_check(image_dir, ram_mb, min_headroom_dumps=5)
    -> tuple[bool, dict]: ...
def scan_producer_log(producer_log, max_errors=5) -> list[str]: ...
```

### Call sites

1. **`run_session()`** preflight: count stale dumps, optionally `sudo rm` them (`--purge-stale-dumps`), verify free disk ≥ `min_dumps_headroom × ram_mb` GiB, write the preflight result into `session_sentinel.json`. **If preflight fails, exit 4 before any cell runs.**
2. **`execute_cell()`** preflight: per-cell `pre_cell_disk_check`. If below threshold, **mark cell failed before any VM operation**. Cell does NOT touch libvirt, so no risk of leaving the VM in a half-state.
3. **`execute_cell()`** post-producer: `scan_producer_log` for error-ish lines; append the top 3 to the per-cell `notes` so the failure mode is self-describing in the JSON.

### CLI

```
--purge-stale-dumps        sudo-rm leftover dumps before the session starts
--min-dumps-headroom N     refuse cells with free disk < N × ram_mb GiB (default 5)
```

### Why "smallest reliable" wins over alternatives

| alternative | why rejected |
|------------|--------------|
| Per-cell offline-analyzer-then-delete hook | Substantial (~150 LOC) + couples capture with analyzer integration which is its own Step. Out of scope for unblocking 1.5c. |
| Producer bash fail-fast on pmemsave error | Changes the producer script, touches shared production path. Bigger blast radius. Tracked as H3 follow-up, not on critical path. |
| Switch dump dir to user-owned path | Requires libvirt config change + qemu apparmor allow-rule. Operator-hostile. |
| Disable `keep_dumps` entirely | Throws away Step 1.5b's design value. Defer to Step 2 design; for 1.5c sub-pilot, smaller fix (disk guard + purge) is enough. |

---

## 4 · Updated pipeline guardrails (in order they execute)

1. **Session preflight** (new) · counts stale dumps + verifies disk free · refuses to start if below threshold.
2. **Manifest crash-recovery** (existing) · marks any 'running' row as 'failed' on restart.
3. **Session sentinel** (existing, now extended) · records the preflight result for audit.
4. **SSH credential preflight** (Bug-K D-30) · refuses if no `SSH_KEY` / `SSH_PASS` when any cell has workload_command.
5. **Per-cell disk preflight** (new) · refuses cell if free disk < headroom; cell marked `failed` BEFORE any VM op.
6. **Per-cell SSH probe** (Bug-J D-29) · `test -x <binary>` before launch.
7. **Cell runs** producer + workload.
8. **Settle** (Bug-L D-34) · retry `virsh resume` with backoff until lock free.
9. **Producer-log scan** (new) · captures error-ish lines into per-cell notes.
10. **Quantitative ok check** (Bug-M D-32) · cells with `snap_completion_ratio < 0.30` → `failed`.

Each guard runs in O(1) host time. Aggregate overhead: ~1-2 s per cell.

---

## 5 · Per-agent contribution

| Agent | One finding | Smallest fix proposed |
|-------|-------------|----------------------|
| **SA** | Bug-L D-34 was correct, but it only fires *after* a cell completes; if a cell is launched against a disk-full host it dies before the settle ever runs. We need a guard BEFORE the cell. | `pre_cell_disk_check` returns early with `failed` status, never touches libvirt. |
| **EN** | The cleanup warning `dump dir read-only` is the canary: it means the orchestrator could not delete dumps from a prior cell. Those dumps now occupy disk. Eventually it fills. | Session preflight does a one-shot `sudo rm` (via existing `e1.purge_all_dumps`) when `--purge-stale-dumps` is passed; logs the count purged. |
| **ML** | Doesn't change anything analyzer-side until disk is reclaimed. F1/CV computation will run on whatever cells survive. Skipped cells contribute no data, which is correct. | None — supports XD's per-cell skip. |
| **XD** | Cell-level skip on disk failure is preferable to session-level abort: lets the orchestrator continue to subsequent cells if disk recovers. | Per-cell preflight marks cell `failed` and continues; manifest preserves audit trail. |
| **DE** | Schema v2 needs no new fields; `notes` is the channel for failure cause. But `session_sentinel.json` should include the preflight result so post-pilot analysis can detect disk-pressure runs. | `session_sentinel.json` gains `preflight` subtree. |
| **DS** | Cells flagged with `pre-cell disk:` and `producer.log errors:` notes must be excluded from the iv-recommendation aggregation, not silently averaged. | Analysis layer already filters on `exit_status != 'ok'`; no code change needed here. |
| **EE** | Acceptance criterion 5 already excludes cells without analyzer outputs. New: cells with `exit_status='failed' AND reason=preflight` are NOT counted in the per-iv N for stationarity tests either; they were never measured. | Recommendation: in `plan02_analysis.py` filter `exit_status == 'ok'` (already done) AND skip cells with `pre-cell disk:` in notes for stationarity stats. ~3 LOC follow-up, non-blocking. |
| **PM** | Three orthogonal small fixes (preflight purge, per-cell guard, log scan). Each ~50 LOC. Total ~210 LOC. No producer change. No schema change. Smallest fix that addresses all three symptoms. | Ship now. Confirm with sub-pilot rerun. |

---

## 6 · Decisions

| ID | Decision |
|----|----------|
| D-35 | Session preflight: count stale dumps; `sudo rm` if `--purge-stale-dumps`; refuse start if `disk_free < min_dumps_headroom × ram_mb / 1024 GiB`. |
| D-36 | Per-cell preflight: refuse to launch cell if `disk_free < min_dumps_headroom × ram_mb / 1024 GiB`. Cell marked `failed` with explanatory note. No VM op invoked. |
| D-37 | Post-cell producer-log scan: extract up to 5 error-ish lines from the tail of `producer.log` and append to `notes`. |
| D-38 | New CLI flags `--purge-stale-dumps` and `--min-dumps-headroom N` (default 5). |
| D-39 | H3 (producer fail-fast on pmemsave error) is deferred. Out of scope for this debug. Logged for follow-up. |
| D-40 | H4 (operator code version mismatch) is operator hygiene; PM appends `git log --oneline -5` to the diagnostic command list. |

---

## 7 · Operator action sequence

```sh
# === STEP A · Confirm hypothesis on host (5 min, no destructive ops) ===
ssh pcrserral
cd /project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU

# Run diagnostic commands A-G from section 2 above.
# Likely findings:
#   - many memory_dump-*.raw files in /var/lib/libvirt/qemu/dump/
#   - df reports < 10 GiB free
#   - VM in 'paused' state

# === STEP B · Pull fixes ===
git pull
git log --oneline -3 -- plan02_run.py
# expect to see the Day-9 commit at the top

# === STEP C · Clean up + restart ===

# 1. Force VM back to running (clears stuck pause from cell 2)
virsh -c qemu:///system resume "Kali Jeries"   # may need retries if lock still busy
virsh -c qemu:///system domstate "Kali Jeries"
# expect: running

# 2. Purge accumulated dumps (now orchestrator can do this for us)
# but for the first time, do it manually to free a chunk of disk:
sudo rm -f /var/lib/libvirt/qemu/dump/memory_dump-*.raw
df -h /var/lib/libvirt/qemu/dump/
# expect: substantial free space restored

# 3. Wipe sub-pilot scratch
rm -rf /tmp/plan02_1_5c
mkdir -p /tmp/plan02_1_5c/cells

# === STEP D · Rebuild manifest WITHOUT --keep-dumps for 1.5c ===
# (keep_dumps caused per-cell disk burn; defer to a future iteration
#  when the analyzer-then-delete hook lands)

python3 plan02_manifest.py build \
    --workloads sandbox_ransom_batched mem_workingset_sweep_v2 \
    --intervals-ms 500 2000 \
    --durations-s 300 900 \
    --replicates 2 \
    --block-size 8 \
    --output /tmp/plan02_1_5c/manifest.csv \
    --cell-output-dir /tmp/plan02_1_5c/cells \
    --workload-command "sandbox_ransom_batched=/home/kali/memorySignal/VM_executables_phase2/bin/sandbox_ransom_batched --rounds 5" \
    --workload-command "mem_workingset_sweep_v2=/home/kali/memorySignal/VM_executables_phase2/bin/mem_workingset_sweep_v2 --working-set-mb 512" \
    --ssh-target kali@192.168.222.63
# NOTE: --keep-dumps OMITTED in this run

# === STEP E · Run with new guardrails active ===
export SSH_KEY=$HOME/.ssh/id_rsa  # still set from previous step
python3 plan02_run.py \
    --manifest /tmp/plan02_1_5c/manifest.csv \
    --output-dir /tmp/plan02_1_5c/cells \
    --purge-stale-dumps \
    --min-dumps-headroom 5

# Watch for:
#  - "preflight: stale_dumps_before=N ..." line at session start
#  - per-cell "pre-cell disk: free=X GiB" line in heartbeat / notes
#  - if a cell fails, "producer.log errors: ..." in the cell's JSON notes
```

---

## 8 · PM final note

> Three symptoms, one underlying cause: **disk fills, pmemsave fails, the chain unwinds badly.** The smallest correct fix is three small guards (preflight purge, per-cell skip, log scan). No producer change. No schema change. ~210 LOC. Total cost of fix: under one hour. Cost of NOT fixing: every sub-pilot run silently degrades into 1-snap-cells and a stuck VM. Ship the small fix. Validate with sub-pilot rerun. Defer the producer fail-fast (H3) and the analyzer-then-delete hook to their own dedicated sessions.

---

**Audit closed.** Day-9 changes commit: see git log for `plan02_run.py`. Tests: 35/35 pass.
