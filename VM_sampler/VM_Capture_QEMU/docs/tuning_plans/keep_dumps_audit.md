# keep-dumps Audit · should we preserve raw VM memory dumps?

**Scope:** Decide whether the Plan-02 pipeline should retain `memory_dump-*.raw` files (1 GiB each) beyond their analyzer window. User flagged: limited host storage; naive retention will fill the disk.

**Convened:** Full team (PM + SA + EN + ML + XD + DE + DS + EE). One sitting. No code changes shipped — evidence required first.

**Decision rule:** smallest reliable retention policy that supports the actual downstream needs. Prove the need before storing the bytes.

---

## Team and roles

| Agent | Codename | Role for this audit |
|-------|----------|---------------------|
| Project Manager | PM | Force decision · cap scope · own audit trail |
| Senior Architect | SA | Retention model · cleanup policy · failure modes |
| Engineering Skills | EN | Disk-space safeguards · compression test plan · idempotent retention |
| Senior ML Engineer | ML | What does the analyzer actually need? trajectory vs raw bytes |
| Experiment Designer | XD | Reproducibility cost · which cells justify retention |
| Senior Data Engineer | DE | Storage budget math · compression-ratio modeling · archive schema |
| Senior Data Scientist | DS | Statistical reproducibility · what data must survive for re-analysis |
| Evaluation Engineer | EE | Auditability claims · what artifacts justify which acceptance |

---

## 1 · Recommendation · TL;DR

**Do NOT enable `--keep-dumps` by default.** D-51 (the analyzer-then-delete hook) already extracts everything downstream needs from each cell. Raw dumps add no analytical value once the trajectory + workload stderr + producer log + heartbeat are persisted.

Three opt-in modes for special cases:

| Mode | Trigger | What gets kept | Storage cost (90-cell pilot) |
|------|---------|----------------|------------------------------|
| **default** | always | nothing (D-51 outputs only) | ~2-5 MiB total (JSONL + stderr + heartbeat per cell) |
| **debug** | `--keep-dumps-on-fail` (proposed flag) | only failed cells, zstd-3 compressed | bounded by failure count · expect < 5 GiB |
| **archive** | `--archive-compressed` (proposed flag) | every cell, zstd-3, size-capped, ring buffer | run compression test FIRST · estimated 50-200 GiB |

**Default + debug mode together cover all real needs.** Archive mode is reserved for one specific scenario (publication-grade reproducibility) and gated on compression-ratio evidence the operator collects.

---

## 2 · What raw dumps are actually needed for

`ML` opens with the requirement audit:

| Use case | Needs raw dumps? | Needs trajectory? | Notes |
|----------|------------------|-------------------|-------|
| F1 / CV per cell (D-51) | **briefly · per-cell** | yes | analyze-then-delete already implemented |
| Plan 03 window/hop tuning | **no** | yes | operates on trajectory at different (window, hop) |
| Classifier training/validation | no | derived features (run_matrix) | matrix is ~100× smaller than raw dump |
| Re-derive features with a new algorithm | yes | partial | only if the new algorithm needs page-level content |
| Debugging cells that fail | **yes for that one cell** | helpful | inspecting dumps explains pmemsave errors, mlock failures, etc. |
| Reproducibility for thesis defense | optional | yes | trajectory + workload + producer.log is enough for reviewer replication |
| Audit trail | optional | yes | metadata (`schema v2 RunMeta` + `metrics.json`) suffices |

`ML`'s verdict: "raw dumps are needed BRIEFLY (for D-51's hook) and OCCASIONALLY (for failed-cell debugging). They are NOT needed for any downstream analysis, classifier, or thesis-grade reproducibility once the trajectory is preserved."

`DS` concurs: "Trajectory is sufficient statistic for all downstream metrics we care about. Page-content is only required if we add a new metric class (e.g. per-page entropy) that we have not currently designed."

---

## 3 · Storage budget · raw vs compressed · realistic numbers

`DE` lays out the math:

### Raw dumps at typical configurations

| Cell config | dumps/cell | raw/cell | 90 cells | 16 cells (1.5c) |
|-------------|-----------|----------|----------|-----------------|
| iv=100 d=300s  | ~181 | 181 GiB | **16.3 TB** | **2.9 TB** |
| iv=500 d=300s  | ~148 | 148 GiB | 13.3 TB | 2.4 TB |
| iv=500 d=900s  | ~445 | 445 GiB | 40 TB | 7.1 TB |
| iv=2000 d=300s | ~85  | 85 GiB | 7.6 TB | 1.4 TB |
| iv=2000 d=1800s | ~512 | 512 GiB | 46 TB | 8.2 TB |

`DE`: "Even the smallest cell in our matrix produces ~85 GiB raw. The host has ~145 GiB free. **One single iv=500 d=900 cell, kept raw, fills 3× the available headroom.** Raw retention is not just inadvisable, it is literally impossible at our cell count."

### Compressed dumps · expected ratios

`DE`'s priors for VM memory:
- Idle Kali (lots of zero pages): zstd-3 → 5-15 % of raw
- Workload running (mixed entropy): zstd-3 → 30-50 % of raw
- Worst case (high-entropy pages, e.g. encryption mid-flight): zstd-3 → 50-80 % of raw

Best/worst projection for the 90-cell pilot at iv=500 d=900s:

| compression ratio | 90 cells | 16 cells (1.5c) |
|-------------------|----------|-----------------|
| 10 % (best) | 4 TB | 710 GiB |
| 30 % (median) | 12 TB | 2.1 TB |
| 50 % (worst) | 20 TB | 3.6 TB |

`DE`: "**Even at the optimistic 10 % ratio, the full pilot exceeds host storage by 25×.** Archive mode is only viable for a very small targeted sub-pilot (single-digit cells) OR with a remote/external storage backend that we do not have."

### What actually fits

`SA` shifts the question:

> "Wrong question is 'how much storage do we need to keep everything'. Right question is 'how many GiB can we devote to retention without breaking the pipeline'. If we cap retention at, say, 50 GiB (~1/3 of free disk), then compression ratio tells us how many cells we can archive:
>
> - 10 % ratio → 50 GiB ÷ 0.1 ÷ 1 GiB-per-dump = up to 500 dumps archived
> - 30 % ratio → up to 167 dumps archived
> - 50 % ratio → up to 100 dumps archived
>
> So an opt-in archive cap of 50 GiB with zstd-3 lets us keep maybe 100-200 dump-pairs depending on entropy. **That is enough for a focused 2-3 cell archive, not a 90-cell pilot.**"

---

## 4 · Compression test plan · run these BEFORE changing retention behavior

`EN` proposes a 20-minute test the operator runs on `pcrserral`. Real ratios and timings, no hand-waving:

```sh
# Pre-flight · make sure zstd is installed (most Linux distros ship it)
which zstd || apt install -y zstd

# Pick a representative dump that's already on disk
DUMP=$(ls -t /var/lib/libvirt/qemu/dump/memory_dump-*.raw 2>/dev/null | head -1)
if [ -z "$DUMP" ]; then
    echo "No dumps on disk. Capture one cell first or use any 1 GiB file."
    exit 1
fi
echo "Testing against: $DUMP"
ls -lh "$DUMP"
RAW=$(stat -c%s "$DUMP")
echo "raw size: $RAW bytes"

# Test 1 · zstd default level (3) · fast + good ratio
echo "=== zstd -3 ==="
time zstd -3 -f -o /tmp/dump.zst "$DUMP"
ZST3=$(stat -c%s /tmp/dump.zst)
python3 -c "print(f'ratio: {$ZST3/$RAW:.3%} (raw {$RAW/1e9:.2f} GB -> zst {$ZST3/1e9:.2f} GB)')"

# Test 2 · zstd -9 · slower but smaller
echo "=== zstd -9 ==="
time zstd -9 -f -o /tmp/dump9.zst "$DUMP"
ZST9=$(stat -c%s /tmp/dump9.zst)
python3 -c "print(f'ratio: {$ZST9/$RAW:.3%}')"

# Test 3 · zstd -19 · best compression but slow
echo "=== zstd -19 ==="
time zstd -19 -f -o /tmp/dump19.zst "$DUMP"
ZST19=$(stat -c%s /tmp/dump19.zst)
python3 -c "print(f'ratio: {$ZST19/$RAW:.3%}')"

# Test 4 · decompression time (matters for re-analysis later)
echo "=== zstd -d ==="
time zstd -d -f -o /tmp/dump.decompressed /tmp/dump.zst
cmp "$DUMP" /tmp/dump.decompressed && echo "decompressed matches original ✓"

# Test 5 · gzip baseline (slower + worse than zstd, but everywhere)
echo "=== gzip -3 ==="
time gzip -3 -k -f -c "$DUMP" > /tmp/dump.gz
GZ=$(stat -c%s /tmp/dump.gz)
python3 -c "print(f'ratio: {$GZ/$RAW:.3%}')"

# Test 6 · lz4 (very fast, larger output)
echo "=== lz4 ==="
which lz4 || apt install -y lz4
time lz4 -3 -f "$DUMP" /tmp/dump.lz4
LZ4=$(stat -c%s /tmp/dump.lz4)
python3 -c "print(f'ratio: {$LZ4/$RAW:.3%}')"

# Summary
echo
echo "=== summary table ==="
printf "%-12s %12s %10s\n" "method" "bytes" "ratio"
python3 -c "
import os
files = [('raw', '$DUMP'), ('zstd-3', '/tmp/dump.zst'), ('zstd-9', '/tmp/dump9.zst'),
         ('zstd-19', '/tmp/dump19.zst'), ('gzip-3', '/tmp/dump.gz'), ('lz4', '/tmp/dump.lz4')]
raw = os.path.getsize('$DUMP')
for n, p in files:
    s = os.path.getsize(p)
    print(f'{n:<12} {s:>12} {s/raw:>9.2%}')
"

# Cleanup test artifacts
rm -f /tmp/dump.zst /tmp/dump9.zst /tmp/dump19.zst /tmp/dump.gz /tmp/dump.lz4 /tmp/dump.decompressed
```

**Expected outcomes:**

| algorithm | expected ratio | compression time | decompression time | recommendation |
|-----------|----------------|------------------|--------------------|----|
| zstd-3 | 10-40 % | 3-5 s / GiB | 1-2 s / GiB | **default if archive mode adopted** |
| zstd-9 | 8-30 % | 8-15 s / GiB | 1-2 s / GiB | acceptable trade-off |
| zstd-19 | 7-25 % | 60-180 s / GiB | 1-2 s / GiB | too slow for inline use |
| gzip-3 | 15-50 % | 15-30 s / GiB | 5-10 s / GiB | not competitive |
| lz4 | 20-60 % | 1-2 s / GiB | 1 s / GiB | fastest but worst ratio |

`EN`: "**zstd-3 is the right default.** Fast enough that compressing inline (after each pmemsave) adds ~3-5 s per snap — well under the existing ~1.5 s host_dt, so it would extend cell wall-clock by ~50-100 %. Trade-off only worth taking for opt-in archive mode."

---

## 5 · Per-agent verdicts

| Agent | One finding | Vote |
|-------|-------------|------|
| **SA** | Default-off retention is the only structurally safe choice. The host cannot hold raw dumps at any realistic cell count. Compression is necessary IF retention is enabled. Cleanup-on-cell-end must be the invariant; retention must be the exception. | **default OFF** |
| **EN** | zstd-3 is the right algorithm. Compression as part of cell teardown adds ~3-5 s/snap. Disk-budget cap with ring-buffer eviction prevents fill. Inline compression should be opt-in via a separate manifest column, not the existing `keep_dumps` boolean (which currently means "skip cleanup, keep all"). | **opt-in only** |
| **ML** | D-51 already extracts the analytical sufficient statistic (active_page_fraction trajectory). Raw dumps add nothing for analyzer purposes. The only legitimate analyzer use case for retained dumps is "we discover a new metric class six months from now and want to re-derive it from old captures" — for that, archive a representative subset, not the whole pilot. | **default OFF** |
| **XD** | Reproducibility for the thesis chapter does NOT require raw dumps. The artifact chain — manifest CSV + per-cell JSON + metrics.json + workload_stderr + producer.log + heartbeat — is fully sufficient for a reviewer to re-run any cell and obtain the same numbers (given the same VM image, kernel, qemu, etc., which schema v2 records). | **default OFF** |
| **DE** | Storage math says raw retention is mathematically impossible at our matrix size. Compressed retention is only viable for small targeted sub-pilots, capped at ~50 GiB total. If we ever need full-pilot archive, we need external storage (S3, NAS) — not on the capture host. | **default OFF + size cap** |
| **DS** | Statistical reproducibility is satisfied by preserving the trajectory, not the bytes that produced it. If a future analysis needs page-level data, the cheapest path is to re-capture a small representative sub-pilot than to archive all dumps speculatively. | **default OFF** |
| **EE** | Auditability claims in the v1 deliverable are met by schema v2 reproducibility metadata + per-cell metrics.json + the audit log. No claim currently requires raw dump retention. Adding it would inflate audit-artifact size 1000× for no validation gain. | **default OFF** |
| **PM** | 7/7 unanimous · default OFF. Two opt-in modes proposed: `--keep-dumps-on-fail` (debug) and `--archive-compressed` (rare archive scenarios, gated on compression test results). | **default OFF · opt-in for special cases** |

---

## 6 · Cleanup and retention policy

Default behavior (no flag set):
```
producer writes dump → pmemsave done
   ↓
TIMING_SELF_CLEAN deletes prev dump in-process    (Plan 1 · vi)
   ↓
producer ends · tail-cleanup removes the last dump   (cleanup_run_dumps)
   ↓
disk free returns to baseline before next cell
```

If `--keep-dumps` is set AND D-51 hook ran (`keep_dumps=True` AND not warmup AND numpy available):
```
producer writes dump → pmemsave done
   ↓
no TIMING_SELF_CLEAN
   ↓
producer ends · D-51 reads dumps · computes APF trajectory + F1 + CV
   ↓
metrics.json written · analyzer_outputs populated
   ↓
tail-cleanup removes all dumps (D-57)               ← key invariant
```

**Proposed new mode 1 · `--keep-dumps-on-fail`** (debug):
```
cell completes
   ↓
if status == 'failed' AND cell.keep_dumps:
   zstd -3 the dumps to <workdir>/dumps/*.zst
   delete the raw dumps
else:
   normal cleanup (delete dumps)
```
Storage cost: bounded by failure count. At expected failure rate < 5 %, < 5 cells × ~100 GiB compressed = at most 5-10 GiB.

**Proposed new mode 2 · `--archive-compressed`** (rare archive):
```
preflight: check disk has >= 50 GiB free
cell completes successfully
   ↓
zstd -3 the dumps to <archive-dir>/<cell_id>/*.zst
delete the raw dumps
   ↓
after each cell: check archive dir size
   if > 50 GiB: evict oldest cell archive (ring buffer)
```
Bounded at 50 GiB. Allows ~50-200 dump-pairs depending on ratio.

**Disk-space safeguards** (already in place from Day-9):
- Session preflight refuses to start if `disk_free < min_dumps_headroom × ram_mb / 1024 GiB`
- Per-cell preflight skips cells that fall below headroom mid-pilot
- These also protect the proposed new modes — they don't need new safeguards

---

## 7 · Minimal pipeline change · only if archive mode is adopted

PM constrains: ship NOTHING until the compression test in Section 4 runs and produces a real ratio number from this host.

If results justify implementation, smallest change is:

| change | cost |
|--------|------|
| Add `--keep-dumps-on-fail` flag to manifest builder | ~5 LOC |
| `execute_cell` post-cleanup hook: when status=failed AND flag set, zstd dumps before delete | ~25 LOC |
| `--archive-compressed` flag + ring-buffer eviction | ~40 LOC |
| Total | ~70 LOC if both modes adopted |

**No producer change.** No schema change. Reuses existing `cell_workdir` for per-cell archive directory.

If results show poor compression (ratio > 60 %) on representative dumps, **scrap archive mode entirely.** Debug mode alone (only failed cells) remains viable since failed cells are rare.

---

## 8 · PM final recommendation

> Default behavior stays as it is today: dumps live only as long as D-51 needs them (or not at all if `keep_dumps=False`). The pipeline is safe; the analytical sufficient statistic (trajectory) is preserved; disk is bounded.
>
> Operator runs the Section-4 compression test before any code change. If zstd-3 returns < 30 % ratio on a real dump from `pcrserral`, debug mode is worth implementing (~25 LOC). If ratio > 50 %, even debug mode buys little — just keep raw dumps for failed cells without compression (rare, bounded by failure rate).
>
> Archive mode is reserved for a future thesis-review scenario where a reviewer asks for raw page-level data on a specific cell. At that point, re-capture that one cell with manual `--keep-dumps` and hand off to the reviewer. No need to archive speculatively.

---

## 9 · Decisions

| ID | Decision |
|----|----------|
| D-58 | Default retention policy stays OFF. `--keep-dumps` continues to mean "D-51 analyzer runs then cleans up". No naive retention path is added. |
| D-59 | Operator runs Section 4 compression test on `pcrserral` BEFORE any retention-feature code is written. Result drives whether mode 1 / mode 2 / neither is implemented. |
| D-60 | If mode 1 (debug · failed-only) is adopted: ~25 LOC patch · per-cell `if status=='failed' AND flag set: zstd dumps before delete`. |
| D-61 | If mode 2 (archive · compressed) is adopted: ~40 LOC patch including ring-buffer eviction. Gated on Section 4 results showing zstd-3 ratio < 30 %. |
| D-62 | Reproducibility for the thesis chapter requires preserving: schema-v2 cell JSONs, metrics.json (D-51 output), workload_stderr.log, producer.log, manifest CSV, session_sentinel.json. **None of these require raw dump retention.** |
| D-63 | If a reviewer ever requests page-level data for a specific cell, re-capture that one cell with manual `--keep-dumps` rather than archiving speculatively. |

---

**Audit closed.** No code changes shipped. Operator action: run Section 4 compression test. Drop the output back for the team to interpret + decide between mode 1 / mode 2 / neither.

---

## 10 · Empirical result · ratio measured on `pcrserral`

Operator ran `keep_dumps_compression_test.sh` against a 1.07 GB dump of the live Kali VM. Real numbers below; refines every estimate above.

```
method        bytes       ratio    comp(s)  decomp(s)  notes
raw         1073741824    100.00 %    0.000      0.000  identity
zstd-3       162311758     15.12 %    3.079      0.949  round-trip OK
zstd-9       152643446     14.22 %   13.025      0.914  round-trip OK
zstd-19      132416951     12.33 %  318.848      1.166  round-trip OK
gzip-3       178199639     16.60 %   10.881         —   not round-trip tested
```

### Interpretation

| algorithm | conclusion |
|-----------|-----------|
| **zstd-3** | **chosen.** 15.12 % ratio · 3.1 s compress · 0.95 s decompress. Sweet spot. |
| zstd-9 | 4× slower for 0.9 pp better ratio. Not worth. |
| zstd-19 | 100× slower for 2.8 pp better ratio. Wasteful. |
| gzip-3 | Strictly worse than zstd-3 on both ratio and time. Reject. |
| lz4 | Not installed on host. Skip; zstd-3 strictly dominates lz4's typical ratio anyway. |

### Disk budget at the measured 15.12 % ratio

| scenario | raw | zstd-3 | host fits? |
|----------|-----|--------|-----------|
| single failed cell (~150 dumps avg) | 150 GiB | ~23 GiB | yes · trivially |
| 90-cell pilot full archive | ~40 TB | ~6 TB | NO · still 40× over |
| 50 GiB archive cap | — | ~330 dumps | yes · 3-5 cells worth |
| 100 GiB archive cap | — | ~660 dumps | yes · 6-10 cells |

### Revised verdict · what the team would build now

`DE`: "15 % is **way better than my 30 % prior**. Mode 1 storage cost is negligible at expected failure rates."

`EN`: "3 s zstd-3 per dump · post-cell batch only · never inline. Doubling cell wall-clock for compression is unacceptable."

`ML`: "Decompression is < 1 s. Archived cell is cheap to re-open. Mode 2 (ring-buffered archive · 3-10 cells) is now a real option, not aspirational."

`XD`: "Mode 1 (failed-only) is the obvious build. Mode 2 stays operator-driven · only when an actual archive need arises."

`DS`: "Trajectory + workload stderr stay the analytical sufficient statistic. Mode 1 is debug support, not analysis-required data."

`SA`: "Mode 1 hooks into the existing failure path. Mode 2 would reuse cell_workdir + a ring-buffer pruner. ~25 + ~40 LOC respectively."

`EE`: "'Failed cells preserved compressed' is a defensible thesis footnote when describing artifact retention policy."

`PM`: "Default OFF stays. **Mode 1 is now worth implementing.** Mode 2 stays optional · operator-triggered when archive need actually arises."

### Updated decisions

| ID | Decision (updated) |
|----|-------------------|
| D-58 | UNCHANGED · default retention OFF. |
| D-59 | CLOSED · compression test ran · zstd-3 ratio measured at 15.12 % on `pcrserral`. |
| D-60 | **PROMOTED to GO-WHEN-OPERATOR-APPROVES** · mode 1 (`--keep-dumps-on-fail`) is worth building. ~25 LOC. |
| D-61 | UNCHANGED · mode 2 (`--archive-compressed`) deferred · operator-triggered. |
| D-62 | UNCHANGED · thesis reproducibility does not require raw retention. |
| D-63 | UNCHANGED · reviewer-request scenario uses one-off `--keep-dumps`. |
| D-64 (new) | If mode 1 ships, the compression algorithm is fixed to `zstd -3` based on the empirical measurement. No tunable level flag · keep the code simple. |

### Pending operator approval

Mode 1 code (~25 LOC) is queued. Will land if operator approves with "go". Otherwise this audit closes with the empirical result captured and no behavior change.
