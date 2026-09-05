# ROUND 1 — Volatile-memory / hypervisor-introspection lens

**Substrate verified from code, not summaries.** `apf_calc/src/main.rs`: APF = (4096-B pages where
*any* byte differs) / 262144, pure binary, no magnitude/count/location. `capture_producer_qemu_pmemsave.sh`:
the VM is `virsh suspend`-ed around each `pmemsave`; the guest runs ~500 ms between dumps, so APF
integrates ~500 ms of guest execution. `config_qemu_upc.json`: 1024 MiB, 262144 pages, 500 ms.
`b1_features.py`: the differ stores `n_changed` and `ham_sum` (bit-level Hamming) per snapshot;
`apf=n_changed/N`, `wapf=ham_sum/(N.32768)`, `intensity=ham_sum/(n_changed.32768)` = per-changed-page
depth. So magnitude is already a captured sufficient statistic; spatial location is not.

**The floor (measured on `~/b1_data`).** The two workloads *designed* to write nothing —
`cpu_hash_loop` (4 KiB L1-resident read-only buffer, register hash) and `cpu_branch_random` (64 KiB
read-only table, register accumulators) — produce a median of **118-119 pages/snapshot** (APF
0.00045), depth 0.0050. That is the guest-OS background ("idle is never idle"): timekeeping, scheduler
runqueue, per-CPU stats. Every quiet workload's median sits within ~10-46 pages of this floor. **The
workload signal is smaller than the OS baseline it rides on.**

## 1. POSITION per quiet family

Per-snapshot medians from `X_apf`/`X_wapf`; **depth = wAPF/APF** (per-changed-page bit-magnitude;
floor = 0.0050).

**CPU — SPLIT, and that split is the confusion.**
- `cpu_hash_loop` (+1 pg over floor, depth 0.0050) and `cpu_branch_random` (+0 pg, depth 0.0049):
  **(c) hard observability wall.** By construction they write ~0 workload pages (accumulator in
  registers, input read-only). Their APF *is* the floor. No re-representation of writes that don't
  exist can help.
- `cpu_matrix_mult` (+126 pg, p95 569, **depth 0.209 = 42x floor**): **(a) metric/representation
  limit.** memsets and fully recomputes a contiguous C matrix (dim 512 -> 2 MiB = **exactly 512
  pages**) every pass. Real, deep, structured writes binary APF flattens. Magnitude/spatial recovers.
- Why CPU is mutually confused: two of its three workloads *are* the floor; the classifier can't
  exploit the third because APF averaged its structure away.

**IO — (c) at this acquisition; the discriminative signal is real but not in guest-RAM writes.**
- `io_read_cache_hit` (+10 pg, depth 0.0104): cache-**hit** reads into one reused 4 KiB buffer. Reads
  never change bytes -> invisible to a write differ, by physics.
- `io_direct_write_like` (+46 pg, depth 0.0069): `O_DIRECT` writes **bypass the guest page cache** and
  land on the virtual block device; guest-RAM footprint is ~1 buffer page + block-layer/virtio
  structures. The data write is not in the pages `pmemsave` dumps.
- Both near-floor with floor-level depth. Distinguishing information (syscall rate, cache-hit rate,
  block-I/O bytes) exists — in channels this instrument cannot see. The repo already conceded this by
  adding the Plan 06 `domblkstat` rd/wr channel. Needs different acquisition.

**THREAD — SPLIT between (c) and (b).**
- `thread_lock_contention` (+28 pg, depth 0.0086) and `thread_producer_consumer` (+31 pg, depth
  0.0107): **(c).** Payload is one shared 8-byte counter, or a 2-3 page ring; all the action is
  futex/scheduler/cross-CPU cache-line ping-pong — kernel bookkeeping in a handful of fixed pages.
- `thread_parallel_alloc` (+10 pg, **depth 0.0062 = floor-level, despite enormous write activity**):
  **(b) sampling/temporal-aliasing limit.** Each thread `malloc`s 1-256 KiB, touches one byte per
  page, `free`s, repeats at high rate. glibc **recycles the same arena pages**, so the distinct-page
  footprint stays small, and APF counts each dirtied page *once per 500 ms* no matter how many
  thousand times it was rewritten. Activity is huge; footprint is aliased to the floor. Its depth is
  floor-level (one byte per page), so **magnitude will not save it either** — needs a per-page write
  *count* at finer cadence.

## 2. FALSIFYING TEST separating (a) from (c) — cheapest, on existing data

**The `intensity`/depth arm (`ham_sum/n_changed`) isolates the magnitude factor wAPF destroyed by
multiplying it into the count.** Already recorded per snapshot in each cell's `b1_trajectory.jsonl`
(arm E2' in `b1_features.py`); laptop proxy `wAPF/APF` computable now.

Rank each quiet workload's depth against the floor (0.0050). **This already returns the verdict:**
matrix_mult 0.209 (42x) separates cleanly -> **(a) confirmed, structure present**;
hash/branch/io_read/io_direct/lock/producer all 0.0049-0.0107 (<=2.2x) -> **(c) confirmed, no
magnitude structure**; parallel_alloc floor-level despite activity -> **(b).**

- If wrong about (c): a supposedly-quiet workload would show depth >> floor. None do except
  matrix_mult.
- Second, decisive falsifier for (c) (one cheap server command): capture a pure-idle guest and show
  hash/branch/io_read/lock are statistically indistinguishable from idle APF+depth. If workload ==
  idle, no representation of its (absent) writes can classify it. Command for the user to run/approve:
  `CONFIG=config_qemu_upc.json CAPTURE_METRIC=apf_queue ./capture_producer_qemu_pmemsave.sh` against
  an idle guest for ~120 snapshots, then compare its APF/depth distribution to the quiet workloads'.

## 3. DEEP H3 TEST — does the crux need a different ACQUISITION?

H3 = "does the discriminative info exist in the physical write behavior at all, at this
granularity/cadence?" **Per-workload, and mostly an acquisition question, not a metric one:**

- **(a) case (matrix_mult): answerable WITHOUT new acquisition.** The substrate stores
  `ham_sum`+`n_changed`; the intensity arm and a magnitude view recover it from existing trajectories.
  The *one* thing the sufficient statistics drop is **spatial contiguity** (the 512-page C block). To
  prove that needs per-page data, not a different cadence. Command (already wired,
  `CAPTURE_METRIC=substrate` -> `live_delta_calc_modular --speed 2 --sparse`): run the substrate differ
  on one retained prev/curr pair from a `cpu_matrix_mult` cell and one from `cpu_hash_loop`; check
  matrix_mult shows a contiguous ~512-page high-Hamming block vs hash_loop's scattered floor pages.
  Per-page corpus is on the server.
- **(c)/(b) cases: NOT answerable by any re-representation of these writes — need different
  acquisition:** `parallel_alloc` -> per-page write-*count* via dirty-bit sampling (KVM dirty-log) or
  ~50 ms cadence; `io_read` -> page access-bit / read tracking (Plan 06 `rd_bytes`); `io_direct` ->
  block-layer `wr_bytes` (Plan 06 `domblkstat`); `lock`/`producer` -> context-switch / futex-syscall
  counters — the signal is scheduler activity, not RAM.

So within the current pmemsave-diff acquisition, **the only quiet workload recoverable is the
structured writer (matrix_mult), and it needs magnitude/spatial, not a re-capture.** For compute-,
read-, and sync-bound work the substrate has a genuine observability wall.

## 4. LIMITS

- Whether a richer **transform** of the same apf/wapf/intensity series can lift matrix_mult or
  de-alias parallel_alloc within 4 s — **DSP engineer**.
- Whether the **classifier** wastes the structure present in matrix_mult — **ML engineer**.
- Whether the **confusion morphology** carries diagnostic meaning — **ECG scientist**.
- Adequacy of the 8-sample/4 s window — DSP/ML.

**Unverified, labeled:** `ham_sum` = bit-level Hamming taken from `b1_features.py` +
`test_plan08_b1_extract.py`, not re-derived from the Rust differ. matrix_mult page count assumes
`--dim` scales with `scale` (512 pages at scale 1.0). The idle-baseline falsifier needs one server
capture.

**One-line position:** the instrument hears writers and is deaf to thinkers because the thinkers'
signal is *below the guest-OS floor*; among the quiet families exactly one workload (matrix_mult) is a
re-representation problem (a), one (parallel_alloc) is a temporal-aliasing problem (b), and the rest
are a hard observability wall (c) whose information lives in reads, disk, and scheduler state that
`pmemsave` cannot see.
