# Plan 02 v3 · Full-Family Capture · Operator Runbook

Captures the full Phase-2 workload family (11 safe synthetic research
sandboxes) with corrected deployment. The orchestrator now auto-injects
`--duration <cell.duration_s>` and `--phase-markers` into every workload
command, so duration-respecting workloads stay active for the whole
capture window instead of finishing early.

All workloads are the existing safe synthetic probes under
`VM_executables_phase2/` (reversible XOR, no network, no persistence,
validated sandbox dir). None are real malware.

**Server:** host `pcrserral`, guest `kali@192.168.222.63`, domain
"Kali Jeries". B+3.1 streaming APF (`--keep-dumps`). iv collapsed to a
single 500 ms (cadence floor makes sub-500 moot; comparability for
classification). Durations 120/300/600 s (600 = binary hard cap).

---

## Step 0 · pull code

```sh
ssh pcrserral
cd /project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU
git pull   # picks up the --duration auto-injection + v3 tests
```

## Step 1 · build the workload binaries on the guest

None are prebuilt. Build all, run the smoke target, list `bin/`:

```sh
ssh kali@192.168.222.63 'cd ~/memorySignal/VM_executables_phase2 && make && make smoke && ls -1 bin/'
```

Confirm all 11 binaries exist in `bin/`:
`sandbox_ransom_seq`, `sandbox_ransom_slowburn`, `sandbox_ransom_selective`,
`sandbox_ransom_batched`, `sandbox_scanner_metadata`,
`mem_workingset_sweep_v2`, `mem_mmap_traversal_v2`, `mem_pagefault_density_v2`,
`mem_rmw_intensity_v2`, `mem_writemag_sweep_v2`, `app_hashtable_intensive_v2`.

## Step 2 · pre-flight phase-marker check (load-bearing)

Validator claim C1 gates each cell's operational `ok` on at least one
`[PHASE]` marker. Confirm every binary emits one under `--phase-markers`,
especially the 5 `mem_*` and the hashtable. Quick probe (5 s each):

```sh
ssh kali@192.168.222.63 'for b in mem_workingset_sweep_v2 mem_mmap_traversal_v2 mem_pagefault_density_v2 mem_rmw_intensity_v2 mem_writemag_sweep_v2 app_hashtable_intensive_v2; do echo "== $b =="; ~/memorySignal/VM_executables_phase2/bin/$b --phase-markers --duration 5 2>&1 | grep -c "\[PHASE\]"; done'
```

Each count must be `>= 1`. If any is `0`, that workload's cells will FAIL
C1 — stop and investigate before launching.

## Step 3 · build the v3 manifest

Fresh output dir (`v3_full`) so cell_ids don't collide with v2. Each
`--workload-command` carries only intensity flags — the orchestrator adds
`--duration` and `--phase-markers`.

```sh
mkdir -p /project/homes/jeries/memory_traces/v3_full/cells
B=/home/kali/memorySignal/VM_executables_phase2/bin

python3 plan02_manifest.py build \
  --workloads \
      sandbox_ransom_seq sandbox_ransom_slowburn sandbox_ransom_selective \
      sandbox_ransom_batched sandbox_scanner_metadata \
      mem_workingset_sweep_v2 mem_mmap_traversal_v2 mem_pagefault_density_v2 \
      mem_rmw_intensity_v2 mem_writemag_sweep_v2 app_hashtable_intensive_v2 \
  --intervals-ms 500 \
  --durations-s 120 300 600 \
  --replicates 2 \
  --block-size 24 \
  --output /project/homes/jeries/memory_traces/v3_full/manifest.csv \
  --cell-output-dir /project/homes/jeries/memory_traces/v3_full/cells \
  --ssh-target kali@192.168.222.63 \
  --keep-dumps \
  --workload-command "sandbox_ransom_seq=$B/sandbox_ransom_seq --files 4000 --file-size-bytes 1048576" \
  --workload-command "sandbox_ransom_slowburn=$B/sandbox_ransom_slowburn --files 200 --interval-s 3" \
  --workload-command "sandbox_ransom_selective=$B/sandbox_ransom_selective --files 1500 --file-size-bytes 1048576" \
  --workload-command "sandbox_ransom_batched=$B/sandbox_ransom_batched --files 12000 --file-size-bytes 1048576" \
  --workload-command "sandbox_scanner_metadata=$B/sandbox_scanner_metadata --files 5000 --subdirs 50 --passes 40" \
  --workload-command "mem_workingset_sweep_v2=$B/mem_workingset_sweep_v2 --working-set-mb 256 --stride 4096" \
  --workload-command "mem_mmap_traversal_v2=$B/mem_mmap_traversal_v2 --variant rmw --file-size-mb 256" \
  --workload-command "mem_pagefault_density_v2=$B/mem_pagefault_density_v2 --variant mixed --working-set-mb 256" \
  --workload-command "mem_rmw_intensity_v2=$B/mem_rmw_intensity_v2 --mode rmw --working-set-mb 256 --stride 4096" \
  --workload-command "mem_writemag_sweep_v2=$B/mem_writemag_sweep_v2 --working-set-mb 256 --bytes-per-page 64" \
  --workload-command "app_hashtable_intensive_v2=$B/app_hashtable_intensive_v2 --capacity-pow2 24 --inserts 6000000 --lookups 10000000"
```

Expect: 66 production cells (11 x 1 iv x 3 dur x 2 rep) + warmups.

Do **not** put `--duration` or `--phase-markers` in these commands — the
orchestrator injects both idempotently.

## Step 4 · dry-run validation

```sh
python3 plan02_run.py \
  --manifest /project/homes/jeries/memory_traces/v3_full/manifest.csv \
  --output-dir /project/homes/jeries/memory_traces/v3_full/cells \
  --dry-run
```

## Step 5 · launch in screen

~7-8 h wall-clock at replicates=2.

```sh
screen -S v3
export SSH_KEY=$HOME/.ssh/id_rsa
python3 plan02_run.py \
  --manifest /project/homes/jeries/memory_traces/v3_full/manifest.csv \
  --output-dir /project/homes/jeries/memory_traces/v3_full/cells \
  --purge-stale-dumps --min-dumps-headroom 5
# detach: Ctrl-A D
```

## Step 6 · mid-run health (separate shell)

```sh
python3 plan02_manifest.py summary /project/homes/jeries/memory_traces/v3_full/manifest.csv
python3 plan02_manifest.py list /project/homes/jeries/memory_traces/v3_full/manifest.csv --status failed
```

## Step 7 · validate

```sh
python3 plan02_validate_session.py \
  --cells-dir /project/homes/jeries/memory_traces/v3_full/cells \
  --manifest /project/homes/jeries/memory_traces/v3_full/manifest.csv
```

## Step 8 · bundle for review — INCLUDE workload_stderr.log

The v2 bundle omitted `workload_stderr.log`, which broke offline F1
recompute. Include it this time.

```sh
cd /project/homes/jeries/memory_traces
tar czf v3_full_review_$(date +%Y%m%d_%H%M).tar.gz \
  v3_full/manifest.csv \
  v3_full/cells/validate_report.json \
  v3_full/cells/session_sentinel.json \
  v3_full/cells/cell_*.json \
  v3_full/cells/warmup_block*.json \
  v3_full/cells/work/*/apf_trajectory.jsonl \
  v3_full/cells/work/*/snapshot_timings.jsonl \
  v3_full/cells/work/*/workload_stderr.log \
  v3_full/cells/work/*/metrics.json \
  v3_full/cells/work/*/producer.log \
  v3_full/cells/work/*/apf_helper.log
ls -lh v3_full_review_*.tar.gz
```

## Step 9 · post-run analysis (local, on the bundle)

1. Confirm phasic workloads now show sustained activity — active-snap
   fraction should be far above v2's 4 % (idle-tail bug fixed).
2. Re-run the Plan 03 window/hop sweep — longer cells admit larger
   windows, so check whether the sustained phasic workloads
   (seq, slowburn) now clear gate G2 (`passes_acceptance=true`):
   ```sh
   python3 plan03_sweep.py \
     --cells-dir <bundle>/v3_full/cells \
     --manifest  <bundle>/v3_full/manifest.csv \
     --output-csv  <bundle>/v3_full/plan03/sweep.csv \
     --output-json <bundle>/v3_full/plan03/summary.json \
     --recommendation-json <bundle>/v3_full/plan03/recommendation.json --force
   ```
3. This dataset is the input to the classification / confusion-matrix
   experiment (the next phase after tuning).

---

## Notes + caveats

- `sandbox_ransom_batched` parses `--duration` but ignores it (one pass
  ~13 s then exits). Kept as a bursty probe; its short active window is
  expected, not a failure.
- Binaries reject `--duration > 600`. v3 stays <= 600. A future longer
  sweep must clamp in `_augment_workload_command`.
- `app_hashtable_intensive_v2` is bimodal (build -> probe); labeled
  `steady`, gets CV primary + incidental F1. A dedicated bimodal metric
  is deferred.
- iv is fixed at 500 ms for all workloads (single common value for
  cross-workload comparability; pmemsave cadence floor makes finer iv
  irrelevant).
