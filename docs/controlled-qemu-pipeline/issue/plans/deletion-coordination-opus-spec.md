# Dump Deletion Coordination (Opus Spec)

## 1. Executive Summary
Replace the current "delete-my-prev" rule with a **reference-scan reaper**: after a job moves to `done/`, probe `pending/` + `processing/` JSON files for any reference to the job's `prev` and `curr` dump paths; delete only those no longer referenced anywhere. This is the smallest change that is correct under both single- and multi-consumer operation, is idempotent, requires no new queue state, and preserves the existing file-based queue.

## 2. Core Invariants
- A dump is referenced by at most 2 jobs: as `curr` in job N and as `prev` in job N+1.
- A dump MUST NOT be deleted while it appears as `prev` or `curr` in any job currently in `pending/` or `processing/`.
- `done/` and `failed/` references do NOT keep a dump alive.
- Jobs transition atomically between queue subdirs via `mv` within one filesystem.
- Deletion MUST be idempotent (safe re-run, safe race between workers).
- Producer never writes a job referencing a dump that has already been deleted (producer always has the newest dump as its in-memory `prev`).
- No new persistent state files (refcounts, locks, manifests) are introduced.

## 3. Proposed Change Set
- `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh` -> replace "delete prev" block with reference-scan reaper applied to both `prev` and `curr`; add helper `maybe_delete_dump` and `dump_is_referenced`.
- `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh` -> no functional change; add short comment that dump lifetime is managed by the consumer reaper.
- `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh` -> no code change (single-consumer today); document that multi-consumer is safe under the new rule.
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` -> no functional change; queue-drain wait remains the correctness barrier before step end.
- `docs/controlled-qemu-pipeline/03-consumer-and-run-matrix.md` -> add "Dump Deletion Rule" subsection describing the reaper semantics.
- `docs/controlled-qemu-pipeline/07-ambiguities-and-out-of-scope.md` -> note: raw-retention and Borg archive branches keep their existing deletion paths; reaper only governs the default `rawRetention=false` path.

## 4. Implementation Plan (Ordered)
1. **Add helpers to consumer**
   - Objective: centralize reference check + idempotent delete.
   - Logic: two new bash functions:
     - `dump_is_referenced <path>`: returns 0 if any file under `$qPending` or `$qProcessing` matches the path as a JSON value of `prev` or `curr`; returns 1 otherwise.
     - `maybe_delete_dump <path> <context>`: if file exists AND not referenced AND not in Borg-pending state AND `rawRetention` branch has not claimed it, call existing `delete_file`.
   - Affected: `capture_consumer_qemu.sh` helper region.
   - Migration: none.

2. **Rewire post-job cleanup**
   - Objective: enforce reference-scan deletion for both `prev` and `curr`.
   - Logic: after the job is moved to `done/` (cleanup must occur *after* the `mv` into `done/` so a concurrent reaper for N+1 cannot miscount), replace the current `if [[ -f "$prev" ]] ...` block with:
     - `maybe_delete_dump "$prev" "snapshot(prev)"`
     - `maybe_delete_dump "$curr" "snapshot(curr)"`
   - Preserve the existing `archive_with_borg_async` and `rawRetention` branches by short-circuiting `maybe_delete_dump` when those branches take ownership of the file.
   - Affected: `capture_consumer_qemu.sh` lines around current `# Delete only prev` comment (~270) and the `rawRetention` block.
   - Migration: pre-existing dumps on disk are cleaned lazily the next time a job referencing them completes; a one-time startup sweep (below) handles orphans.

3. **Startup orphan sweep (bounded)**
   - Objective: on consumer start, reclaim dumps left over from a prior crashed run.
   - Logic: list files in `imageDir`; for each, if not referenced by `pending/` or `processing/` and older than N seconds (configurable, default 300), delete.
   - Affected: `capture_consumer_qemu.sh` startup.
   - Migration: disabled if `imageDir` unset; respects `rawRetention` and Borg paths.

4. **Move to `done/` BEFORE deletion**
   - Objective: ensure a crashed worker leaves either (a) job in `processing/` (dump protected) or (b) job in `done/` (dump eligible), never an ambiguous in-between.
   - Logic: reorder existing flow to: compute delta → append frame → `mv jobProcessing -> qDone` → reaper.
   - Affected: `process_job` in `capture_consumer_qemu.sh`.
   - Migration: small reorder; behavior-equivalent for single consumer.

5. **Doc updates**
   - Objective: record the rule.
   - Affected: `03-consumer-and-run-matrix.md`, `07-ambiguities-and-out-of-scope.md`.

## 5. Algorithms / Pseudocode

```text
# --- reference scan ---
dump_is_referenced(path):
    # grep is safe: jq-written JSON escapes paths; exact string match on the value.
    for dir in [qPending, qProcessing]:
        if grep -lF -- "\"$path\"" "$dir"/*.json 2>/dev/null | head -1 : return 0
    return 1

# --- idempotent, race-tolerant delete ---
maybe_delete_dump(path, ctx):
    if path empty or file missing: return
    if borg_is_archiving(path): return            # existing branch retains file
    if raw_retention_claimed(path): return        # existing branch retains file
    if dump_is_referenced(path): return
    delete_file(path, ctx)                         # rm -f; idempotent by design

# --- post-job flow (new order) ---
process_job(jobPath):
    parse prev, curr, output
    run rust_delta(prev, curr, output) or -> mv jobPath qFailed; return 1
    append_frame(latest_delta_file)
    trigger_streaming_if_eligible()
    mv jobPath qDone                               # commit state BEFORE reaping
    maybe_delete_dump(prev, "snapshot(prev)")
    maybe_delete_dump(curr, "snapshot(curr)")
    return 0

# --- startup sweep ---
orphan_sweep():
    now = epoch_seconds
    for f in imageDir/*.raw:
        if age(f) < ORPHAN_GRACE_SEC: continue
        if dump_is_referenced(f): continue
        if raw_retention_dir_contains(f): continue
        delete_file(f, "orphan dump")
```

## 6. Concurrency / Failure Analysis
- **Two workers completing adjacent jobs simultaneously (N and N+1).**
  - N's dumps: (A, B). N+1's dumps: (B, C).
  - Both move their job to `done/` then reap.
  - Worker-N scan sees N+1 in `pending/` or `processing/` (atomic `mv` guarantees visibility in exactly one) → B not referenced elsewhere? Actually B IS referenced by N+1 → NOT deleted by worker-N. A has no other reference → deleted.
  - Worker-N+1 sees no further job referencing B or C (unless N+2 exists). Deletes B; defers C if N+2 exists.
  - Safe.
- **Double-delete attempts.** `rm -f` is idempotent; `delete_file` already swallows errors. Safe.
- **Crash between `mv -> done/` and reaper.** Dump survives until next reaper pass (startup sweep or next adjacent job completion). Never orphaned forever.
- **Crash between delta finish and `mv -> done/`.** Job remains in `processing/`; dump protected by reference scan. On restart, either reprocess (if retry enabled) or surface as stuck job; no premature deletion.
- **Retries / duplicate processing.** Re-running a job that already has its dumps deleted: the Rust binary will fail to read inputs → job moves to `failed/`. Deletion rule remains correct (it never created the inconsistency).
- **Malformed / stale job metadata.** `jq` failures cause `move_failed`; reaper never runs for malformed jobs. A malformed pending JSON that cannot be parsed by `grep -F` on the exact path still matches literally (grep is string-based), so it still protects dumps — conservative.
- **Producer writes a new job during a scan.** `mv pending.tmp -> pending/<id>.json` is atomic. Scan either sees the old state or the new one; both states keep the dump alive if the new job references it.
- **Idempotency.** Every reaper step is safe to re-run. No state mutation outside `rm -f`. Startup sweep is idempotent.
- **Filesystem requirement.** `imageDir`, `qPending`, `qProcessing`, `qDone` must be on the same filesystem for atomic `mv`. (Already required by existing pipeline.)

## 7. Performance Impact
- Per-job overhead: 2 `grep -lF` scans over `pending/` (≤ `maxPendingJobs = 20`) + `processing/` (≤ #workers). Each JSON ≈ 213 B. Total scanned bytes < 10 KB. Dominated by syscalls, microseconds; negligible vs. Rust delta (~2 min).
- Startup sweep: O(#dumps in `imageDir`) × O(#pending+processing). Bounded by backpressure; negligible.
- No new locks, no fsync, no additional IPC. Acceptable.

## 8. Test Plan
- **Unit (shell-level, mocked dirs)**
  - `dump_is_referenced` returns true when exact path appears as `prev` or `curr` in any pending/processing JSON; false otherwise.
  - `maybe_delete_dump` no-ops when file missing, referenced, Borg-claimed, or raw-retention-claimed.
  - Delete is idempotent (call twice → one deletion, no error).
- **Integration (single consumer)**
  - Run a 10-step synthetic producer feeding 20 jobs. Assert: every dump ends deleted exactly once; final `imageDir` empty (modulo the last `curr`, which remains until the next job or final sweep).
  - Assert no `rm` error in logs.
- **Concurrency (2 consumers, deferred but covered by design)**
  - Spawn 2 consumer processes on the same queue. Run 50 synthetic jobs with overlapping adjacency. Assert: zero premature deletions (no "Rust delta failed: file not found" lines); every non-final dump deleted exactly once; `done/` count = job count.
- **Crash / restart**
  - Kill consumer mid-job (SIGKILL) with a job in `processing/`. Restart. Assert dump files referenced by `processing/` are retained; startup sweep removes orphaned dumps older than grace.
- **Regression**
  - `rawRetention=true` path: dumps still move into `rawDir`, reaper does not fight the move.
  - Borg path: archive handoff still retains source file until archive completes.
  - `run_files_controlled.py` queue-drain timing unchanged; step completes only after `pending` and `processing` reach zero.
- **Edge cases**
  - Last dump in a run (no N+1 yet): survives in `imageDir`. Controller's step teardown or next startup sweep cleans it.
  - Duplicate job IDs (producer bug): `mv` collision surfaces as error; deletion logic untouched.
  - Path with spaces: `grep -F` on quoted JSON value handles correctly.

## 9. Acceptance Criteria
- Consumer deletes each dump exactly once in a multi-step run; no "file not found" errors from the Rust delta binary.
- Under a 2-consumer test, zero jobs move to `failed/` due to missing inputs.
- After normal shutdown, `imageDir` contains at most one dump (the final `curr`) or is empty after startup sweep.
- `process_job` commits the `mv -> done/` before invoking the reaper.
- Reaper never deletes a dump referenced in `pending/` or `processing/`.
- No new persistent queue state (no refcount files, no manifests).
- Docs updated: `03-consumer-and-run-matrix.md` has "Dump Deletion Rule" subsection.

## 10. Rollout / Rollback
- **Rollout:** single commit to `capture_consumer_qemu.sh` + doc edits. No data migration. Deploy by restarting the consumer at a step boundary. First run after deploy: existing orphaned dumps in `imageDir` will be cleaned by the startup sweep.
- **Feature flag:** `REAPER_MODE` env var in consumer: `scan` (new, default) or `legacy` (previous prev-only delete). Allows instant revert without redeploy.
- **Rollback:** set `REAPER_MODE=legacy` and restart consumer; OR `git revert` the commit.

## 11. Executor Handoff Prompt
```
Implement ONLY the changes in docs/controlled-qemu-pipeline/issue/plans/deletion-coordination-opus-spec.md.

Scope:
- Modify VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh only (plus doc notes listed below).
- Do NOT change queue format, producer logic, run_matrix logic, Rust binary, or offline metrics.

Tasks:
1. Add two bash helpers in capture_consumer_qemu.sh near delete_file:
   - dump_is_referenced <path>: returns 0 if any JSON file under $qPending or $qProcessing contains the exact path string as the value of "prev" or "curr". Use `grep -lF -- "\"$path\""` for speed and literal safety.
   - maybe_delete_dump <path> <ctx>: no-op if path empty, file missing, Borg async-archiving it, rawRetention has claimed it, or dump_is_referenced returns 0. Otherwise call existing delete_file.

2. In process_job, reorder so the job is moved to $qDone BEFORE any deletion runs. Then replace the current "Delete only prev" block with:
     maybe_delete_dump "$prev" "snapshot(prev)"
     maybe_delete_dump "$curr" "snapshot(curr)"
   Keep existing archive_with_borg_async and rawRetention branches intact; make them short-circuit maybe_delete_dump.

3. Add a startup orphan sweep function orphan_sweep(): iterate files in $imageDir older than ORPHAN_GRACE_SEC (default 300, overridable via env), delete if not referenced and not claimed by rawRetention. Call once at consumer startup before main loop.

4. Add REAPER_MODE env var: "scan" (default, new behavior) or "legacy" (prior delete-only-prev behavior preserved verbatim). Wrap the new block in a mode check so rollback is flag-only.

5. Update docs/controlled-qemu-pipeline/03-consumer-and-run-matrix.md: append "Dump Deletion Rule" subsection stating dumps are deleted only after their referencing job is in done/ AND no pending/processing job still references them.

6. Update docs/controlled-qemu-pipeline/07-ambiguities-and-out-of-scope.md: note rawRetention and Borg branches manage their own lifetime and are untouched by the reaper.

Do NOT:
- Introduce refcount files, manifests, locks, or a separate reaper process.
- Alter queue subdir names or the pending->processing->done/failed flow.
- Touch the producer beyond an optional one-line comment.
- Change any Python logic in run_files_controlled.py.

Verify:
- Single-consumer 10-step run: zero delta failures due to missing files; imageDir ends empty or with a single trailing curr.
- Two-consumer simulated run (spawn a second consumer process manually): same outcome; no premature deletions.
- REAPER_MODE=legacy restores prior behavior exactly.

Deliverables: one commit with the edits above and a short CHANGELOG-style note in the commit body.
```
