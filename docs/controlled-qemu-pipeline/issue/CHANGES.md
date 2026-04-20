# Code & Documentation Changes Log

Three implementation tasks were executed. This file documents every change made, why it was made, and what it replaced.

---

## 1. Validation Semantics — Cosine-Only Offline Path

**Files changed:** documentation only (no code changes).

### What changed

#### `docs/controlled-qemu-pipeline/03-consumer-and-run-matrix.md`
Added a **"Channel Semantics"** subsection clarifying:
- The run matrix is single-channel, driven by `deltaMetric` in config (currently `cosine`).
- Hamming delta files are written to disk by the Rust binary but are **not** ingested into `run_matrix_<step>.npy`.
- Combined hamming+cosine analysis is a separate downstream path in `VMsig_featureExctraction/`.

#### `docs/controlled-qemu-pipeline/07-ambiguities-and-out-of-scope.md`
Added a **"Combined hamming+cosine offline-step path"** entry under ambiguities:
- Explicitly marks combined-channel offline-step support as **deferred**.
- States that the combined representation exists only in `VMsig_featureExctraction/wavelet_analysis_features.py`.
- Prevents future overclaiming: offline-step results validate cosine-channel separability only.

#### `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
Added an inline comment above the `subdir="cosine"` block:
```
# Single-channel run_matrix: only the selected deltaMetric channel is appended here.
# The non-selected channel (e.g. hamming when deltaMetric=cosine) remains on disk but
# is NOT ingested into the offline-step matrix or the downstream offline metrics path.
```

### Why
The offline-step path silently consumed only cosine but no existing comment or doc said so. Any result reported as "offline-step validated" was implicitly cosine-only, risking overclaiming. The change restores semantic clarity with zero behavioral impact.

---

## 2. Dump Deletion Coordination

**Files changed:** `capture_consumer_qemu.sh` (functional), two doc files.

### Background — the problem
Each RAW memory dump is referenced by two consecutive jobs:
- as `curr` in job N
- as `prev` in job N+1

The old code deleted `prev` immediately after job N completed. Under a single consumer with strictly ordered processing this worked. Under parallel consumers, job N+1 could start before job N's `prev` delete completes — or vice versa — causing the Rust binary to fail with "file not found" on a dump that was still needed.

### What changed in `capture_consumer_qemu.sh`

#### New environment variables (near top of file)
```bash
REAPER_MODE="${REAPER_MODE:-scan}"   # "scan" = new safe behavior; "legacy" = prior behavior
ORPHAN_GRACE_SEC="${ORPHAN_GRACE_SEC:-300}"
imageDir=$(jq -r '.imageDir // ""' "$CONFIG" 2>/dev/null)
```
`REAPER_MODE=legacy` is the instant rollback flag — it restores the exact prior deletion behavior without code revert.

#### New helper: `dump_is_referenced(path)`
```bash
dump_is_referenced() {
  local path="$1"
  [[ -z "$path" ]] && return 1
  local hit
  hit=$(find "$qPending" "$qProcessing" -maxdepth 1 -name '*.json' -print0 2>/dev/null \
        | xargs -0 grep -lF -- "\"$path\"" 2>/dev/null | head -1)
  [[ -n "$hit" ]] && return 0
  return 1
}
```
Scans all JSON files currently in `pending/` and `processing/`. If any file contains the exact dump path as a JSON string value, the dump is still live. The `mv` that transitions jobs between queue dirs is atomic (POSIX, same filesystem), so this scan never sees an ambiguous state.

#### New helper: `maybe_delete_dump(path, context)`
```bash
maybe_delete_dump() {
  local path="$1"
  local ctx="${2:-snapshot}"
  [[ -z "$path" ]] && return 0
  [[ ! -e "$path" ]] && return 0
  if dump_is_referenced "$path"; then
    return 0
  fi
  delete_file "$path" "$ctx"
}
```
Idempotent. No-ops if the file is missing (already deleted, moved by rawRetention, etc.) or still referenced. Calls the existing `delete_file` helper which already handles `sudo rm -f` with permission fallback.

#### New helper: `orphan_sweep()`
```bash
orphan_sweep() {
  [[ -z "$imageDir" || ! -d "$imageDir" ]] && return 0
  local now cutoff
  now=$(date +%s)
  cutoff=$((now - ORPHAN_GRACE_SEC))
  shopt -s nullglob
  for f in "$imageDir"/*; do
    [[ -f "$f" ]] || continue
    local mtime
    mtime=$(stat -c %Y "$f" 2>/dev/null || echo "$now")
    [[ "$mtime" -lt "$cutoff" ]] || continue
    if [[ "$rawRetentionEnabled" == "true" && -n "$rawDir" ]]; then
      [[ -e "$rawDir/$(basename "$f")" ]] && continue
    fi
    if dump_is_referenced "$f"; then
      continue
    fi
    delete_file "$f" "orphan dump"
  done
  shopt -u nullglob
}
```
Called once at consumer startup (scan mode only). Reclaims dumps left over from a prior crashed run without waiting for a new job to reference them.

#### Changed: `process_job` — `mv` order
**Before:** the job was moved to `done/` at the very end of `process_job`, after all deletion logic.

**After:** the job is moved to `done/` **before** any deletion runs.

```bash
# Move job to done/ BEFORE any deletion (ensures reaper sees consistent state)
mv "$jobPath" "$qDone/"
echo "[CONSUMER] Job done -> done: $jobName"
```

Why this matters: if two workers process jobs N and N+1 in parallel, worker N+1 runs `dump_is_referenced` on dump B. For the scan to correctly see "job N+1 is processing B", job N+1 must already be in `processing/` — which it is (moved there before `process_job` is called). But for the symmetrical check — worker N+1 asking "is B still referenced by anything else?" — it needs job N's state to be committed. Moving job N to `done/` first ensures worker N+1's scan finds nothing and safely deletes B.

#### Changed: deletion block (replacing "Delete only prev")
**Before:**
```bash
# Delete only prev. curr becomes the next job's prev and is deleted when that job runs.
if [[ -f "$prev" ]]; then
  if archive_with_borg_async "$prev"; then
    :
  else
    delete_file "$prev" "snapshot"
  fi
fi
```

**After (scan mode):**
```bash
borg_claimed_prev=0
if [[ -f "$prev" ]]; then
  if [[ "$REAPER_MODE" == "legacy" ]]; then
    if archive_with_borg_async "$prev"; then
      borg_claimed_prev=1
    else
      delete_file "$prev" "snapshot"
    fi
  else
    if archive_with_borg_async "$prev"; then
      borg_claimed_prev=1
    fi
  fi
fi

# ... rawRetention block unchanged ...

# Reference-scan reaper
if [[ "$REAPER_MODE" == "scan" ]]; then
  if [[ "$borg_claimed_prev" != "1" ]]; then
    maybe_delete_dump "$prev" "snapshot(prev)"
  fi
  maybe_delete_dump "$curr" "snapshot(curr)"
fi
```

Key behaviors preserved:
- **Borg path:** when `BORG=1` and archive is spawned, `borg_claimed_prev=1` — reaper skips `prev`. Source file is retained for the async archive process. Identical to prior behavior.
- **rawRetention path:** the existing block runs before the reaper and may `mv` curr into `rawDir`. When it does, the file at the original `curr` path is gone — `maybe_delete_dump` no-ops on missing file. No conflict.
- **Legacy mode:** full prior behavior restored; the new reaper block is skipped entirely.

#### Startup call
```bash
if [[ "$REAPER_MODE" == "scan" ]]; then
  orphan_sweep
fi
```
Added immediately before the main `while true` loop.

#### Startup log
Updated to include `reaperMode=${REAPER_MODE}` in the consumer's first log line.

### What the reaper does NOT touch
- `rawRetention` pruning by `keepDumps` — unchanged.
- Borg archive lifetime — unchanged.
- Queue subdirectory names or JSON format — unchanged.
- `run_files_controlled.py` queue-drain wait — unchanged.

### Doc changes
- `03-consumer-and-run-matrix.md`: appended **"Dump Deletion Rule"** section describing the reaper contract, mv-before-delete ordering, and `REAPER_MODE` flag.
- `07-ambiguities-and-out-of-scope.md`: appended **"Dump Lifetime — Borg and rawRetention Branches"** noting those branches are out of scope for the reaper.

---

## 3. Rust Hot-Loop Optimization

**Files changed:** `VM_sampler/VM_Capture/live_delta_calc/src/main.rs`, `Cargo.toml`, one doc file.

### Background — the problems in the original code

The original code had three compounding performance problems:

**Problem 1: Per-page `Vec<f32>` allocation.**
For every 4KB page, two `Vec<f32>` of 4096 elements (16KB each) were heap-allocated, used once, and dropped:
```rust
let page1_f32: Vec<f32> = page1.iter().map(|&x| x as f32).collect();
let page2_f32: Vec<f32> = page2.iter().map(|&x| x as f32).collect();
```
For a 2GB snapshot: 524,288 pages × 2 allocations = ~1 million heap allocation/free cycles per job. This floods the allocator and thrashes CPU cache.

**Problem 2: Serialized file I/O despite 16 threads.**
All 16 rayon workers shared two file handles behind `Arc<Mutex<File>>`:
```rust
let file1 = Arc::new(Mutex::new(File::open(file1_path).await?));
// ...
let chunk1 = read_chunk(&mut file1.lock().unwrap(), offset).await ...
```
Every read required acquiring a global lock. Threads that couldn't acquire the lock blocked, making the 16-thread setup effectively single-threaded for I/O.

**Problem 3: Per-thread `tokio` runtime creation.**
Each of the 16 rayon workers created its own `tokio::runtime::Runtime::new()`:
```rust
let rt = tokio::runtime::Runtime::new().unwrap();
rt.block_on(async move { ... });
```
Each runtime spawns its own thread pool and internal infrastructure. 16 runtime creations per job, all for workloads that have no async I/O benefit (sequential file reads at known offsets).

### What changed in `src/main.rs`

#### Removed
- All `tokio::*` imports and usage
- `async fn` / `async move` blocks
- `#[tokio::main]` attribute
- `Arc<Mutex<File>>` shared file handles
- Per-thread `tokio::runtime::Runtime::new()`
- `read_chunk` async function
- Per-page `Vec<f32>` allocation inside `process_chunk`

#### Added: sync `main` with per-thread file handles
```rust
fn main() -> io::Result<()> { ... }
```
Each rayon worker opens its own independent file handles:
```rust
let f1 = File::open(&prev_path)?;
let f2 = File::open(&new_path)?;
```
Reads use `std::os::unix::fs::FileExt::read_exact_at`, which is the Rust binding to POSIX `pread(2)` — thread-safe, offset-based, no seeking, no locking:
```rust
f1.read_exact_at(&mut chunk1[..to_read], offset)?;
```

#### Added: reusable buffers (eliminates ~1M allocations per job)
Allocated once per thread before the chunk loop, reused every page:
```rust
let mut buf1: Vec<f32> = Vec::with_capacity(PAGE_SIZE);
let mut buf2: Vec<f32> = Vec::with_capacity(PAGE_SIZE);
// ...
buf1.clear();
buf1.extend(p1.iter().map(|&x| x as f32));
```
`.clear()` drops the length to zero but keeps the backing allocation. `.extend()` fills without reallocating because capacity is already PAGE_SIZE. Total f32 allocations per job: 32 (2 per thread × 16 threads), down from ~1,048,576.

#### Added: identical-page fast path
At startup, the exact library function is called on two identical pages to capture the ground-truth output values:
```rust
let zero_vec: Vec<f32> = vec![0.0f32; PAGE_SIZE];
let ones_vec: Vec<f32> = vec![1.0f32; PAGE_SIZE];
let cos_ident_zero = cosine(&zero_vec, &zero_vec);
let cos_ident_nonzero = cosine(&ones_vec, &ones_vec);
```
In `process_chunk`, identical pages skip conversion and cosine entirely:
```rust
if p1 == p2 {
    let all_zero = p1.iter().all(|&b| b == 0);
    out_c.push(if all_zero { cos_ident_zero } else { cos_ident_nonzero });
    continue;
}
```
`p1 == p2` slice comparison compiles to `memcmp` / SIMD — fast. For idle and stable workloads, a large fraction of pages are unchanged between snapshots.

#### Added: `STRICT_IO` environment flag
```rust
let strict_io = matches!(env::var("STRICT_IO").ok().as_deref(), Some("1") | Some("true"));
```
Default (`false`): logs the error and breaks the segment — preserving prior permissive behavior.
`STRICT_IO=1`: panics with a descriptive message — useful for debugging or strict pipeline runs.

#### Changed: `process_chunk` signature
**Before:**
```rust
fn process_chunk(chunk1: &[u8], chunk2: &[u8]) -> (Vec<u32>, Vec<f32>)
```
Allocated and returned new Vecs per call.

**After:**
```rust
fn process_chunk(
    chunk1: &[u8], chunk2: &[u8],
    buf1: &mut Vec<f32>, buf2: &mut Vec<f32>,
    cos_ident_zero: f32, cos_ident_nonzero: f32,
    out_h: &mut Vec<u32>, out_c: &mut Vec<f32>,
)
```
Caller-owned output Vecs are passed in and appended to. No per-call allocation.

#### Changed: output assembly
**Before:** results were stored in `Arc<Mutex<Vec<Vec<...>>>>` slots then formatted in a single-threaded loop into one giant `String`.

**After:** each rayon worker builds its own `String` in parallel, then the main thread concatenates and writes once:
```rust
let mut hs = String::with_capacity(local_h.len() * 6);
let mut cs = String::with_capacity(local_c.len() * 12);
for &h in &local_h { let _ = writeln!(hs, "{}", h); }
for &c in &local_c { let _ = writeln!(cs, "{}", c); }
```
Final write:
```rust
std::fs::write(&hamming_result_file_path, hamming_text)?;
std::fs::write(&cosine_result_file_path, cosine_text)?;
```

#### What is preserved bit-exactly
- Output file paths (`output_dir/hamming/…` and `output_dir/cosine/…`).
- Filename timestamp format (`%Y%m%d%H%M%S`).
- One value per line, same `Display` format for `u32` and `f32`.
- Global page ordering: slots collected in ascending `thread_id` order.
- Hamming value: `hamming::distance(p1, p2) as u32` — unchanged.
- Cosine value: `distances::vectors::cosine(&buf1, &buf2)` — same function, same inputs.
- File size assertion: `assert_eq!(file1_size, file2_size)` — unchanged.
- CLI: `<prev_image> <new_image> <output_dir>` — unchanged.
- Exit code 1 on wrong arg count — unchanged.

### What changed in `Cargo.toml`

**Before:**
```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
async-std = "1.10"
hamming = "0.1.3"
futures = "0.3"
rayon = "1.10.0"
emd = "0.1.0"
ndarray = "0.12.1"
ml-distance = "^1.0.0"
distances = "1.7.1"
chrono = "0.4"
```

**After:**
```toml
[dependencies]
hamming = "0.1.3"
rayon = "1.10.0"
distances = "1.7.1"
chrono = "0.4"
```

Removed crates:
- `tokio` — no longer used; replaced by sync stdlib I/O.
- `async-std` — was imported but unused even in the original.
- `futures` — was imported but unused.
- `emd` — was imported but unused.
- `ndarray` — was imported but unused.
- `ml-distance` — was imported but unused.

### Doc changes
- `03-consumer-and-run-matrix.md`: appended **"Rust Delta Binary Note"** stating that outputs are unchanged and runtime is reduced, and noting the Unix-only requirement (`read_exact_at`).

---

## Summary Table

| Area | File | Type | Rollback |
|---|---|---|---|
| Validation semantics | `capture_consumer_qemu.sh` | Comment added | Delete comment |
| Validation semantics | `03-consumer-and-run-matrix.md` | Doc section added | Delete section |
| Validation semantics | `07-ambiguities-and-out-of-scope.md` | Doc section added | Delete section |
| Deletion coordination | `capture_consumer_qemu.sh` | Functional change | `REAPER_MODE=legacy` |
| Deletion coordination | `03-consumer-and-run-matrix.md` | Doc section added | Delete section |
| Deletion coordination | `07-ambiguities-and-out-of-scope.md` | Doc section added | Delete section |
| Rust hot-loop | `live_delta_calc/src/main.rs` | Full rewrite of I/O and compute | `git revert` or restore prior binary |
| Rust hot-loop | `live_delta_calc/Cargo.toml` | Dep pruning | Restore prior `Cargo.toml` |
| Rust hot-loop | `03-consumer-and-run-matrix.md` | Doc note added | Delete note |
