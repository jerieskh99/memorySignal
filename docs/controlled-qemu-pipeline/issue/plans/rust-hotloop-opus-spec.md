# Rust Hot-Loop Optimization (Opus Spec)

## Blocking Unknowns
- Exact semantics of `distances::vectors::cosine` (distance vs similarity) in crate version 1.7.1. Assumption: treat the library's output value as the ground-truth reference and preserve it bit-exactly by keeping the same function call on identical inputs for calibration; fast-path emits the calibrated constant.

## 1. Executive Summary
Three decisive, locally-scoped changes to `src/main.rs`: (a) hoist two per-thread reusable `Vec<f32>` buffers out of the per-page loop and `clear()`/`extend` them in place — eliminates ~1M heap allocations per job; (b) add an identical-page fast path using byte-slice equality that emits hamming=0 and a calibrated "identical-page" cosine constant; (c) replace `tokio::fs` + `Arc<Mutex<File>>` with `std::fs::File` + POSIX `read_exact_at` per rayon thread — eliminates per-thread tokio runtime spawning and global file-handle mutex contention. Output format and numerical semantics are preserved. No unsafe code.

## 2. Core Invariants
- Output files `<output_dir>/hamming/…txt` and `<output_dir>/cosine/…txt` have the same path format, line ordering, and line format as today.
- Hamming value per page is the same `hamming::distance(page1, page2) as u32`.
- Cosine value per page is bit-exact with the current `distances::vectors::cosine(&page1_f32, &page2_f32)` output for all byte pairs.
- Page order in the output files matches chunk segmentation order (thread_id ascending, then intra-segment offset ascending) — identical to current behavior.
- PAGE_SIZE and CHUNK_SIZE remain 4096 and 262144.
- File size assertion (`file1_size == file2_size`) remains.
- No change in CLI arguments or exit codes.
- No new crate dependencies beyond what stdlib provides (allow removal of `async-std`, `emd`, `ndarray`, `ml-distance`, `futures` if unused).

## 3. Proposed Change Set
- `VM_sampler/VM_Capture/live_delta_calc/src/main.rs` -> rewrite `process_chunk` for reusable buffers and identical-page fast path; replace tokio file I/O with `std::fs::File` + `read_exact_at`; drop per-thread tokio runtimes; add one-time calibration of identical-page cosine constant at startup.
- `VM_sampler/VM_Capture/live_delta_calc/Cargo.toml` -> remove `tokio` feature weight (drop to default features) or remove the dep entirely; prune unused deps (`async-std`, `emd`, `ndarray`, `ml-distance`, `futures`). Keep `hamming`, `distances`, `rayon`, `chrono`.
- `docs/controlled-qemu-pipeline/03-consumer-and-run-matrix.md` -> one-line note that the Rust binary’s output values are unchanged; runtime is reduced.

## 4. Implementation Plan (Ordered)
1. **Calibrate identical-page cosine constant at startup**
   - Objective: determine the exact `f32` value returned by `distances::vectors::cosine` for two identical all-zero pages, and for two identical non-zero pages (to detect if the library behaves differently on zero-norm inputs).
   - Logic: in `main()`, before spawning workers, call the library once with `[0u8;PAGE_SIZE]→f32` on both sides, and once with `[1u8;PAGE_SIZE]→f32`. Store results in two `const`-style locals (`COS_IDENT_ZERO`, `COS_IDENT_NONZERO`) passed by value to workers.
   - Affected: `main.rs` main function.
   - Migration: none.

2. **Rewrite `process_chunk` with reusable buffers + fast path**
   - Objective: eliminate per-page allocations and short-circuit identical pages.
   - Logic:
     - New signature: `fn process_chunk(chunk1: &[u8], chunk2: &[u8], buf1: &mut Vec<f32>, buf2: &mut Vec<f32>, cos_ident_zero: f32, cos_ident_nonzero: f32, out_h: &mut Vec<u32>, out_c: &mut Vec<f32>)`.
     - For each page: if `page1 == page2` (byte-slice eq), push `0u32` and either `cos_ident_zero` (if `page1[0]==0 && page1.iter().all(|&b| b==0)` — cheap memcmp already done by eq check on all-zero slice — actually just test with `page1.iter().all(|&b| b==0)`) or `cos_ident_nonzero`. Otherwise: `buf1.clear(); buf1.extend(page1.iter().map(|&x| x as f32)); buf2.clear(); buf2.extend(page2.iter().map(|&x| x as f32)); let c = cosine(&buf1, &buf2);` and push.
     - Hamming always computed via `distance(page1, page2) as u32`.
     - Appends go to caller-owned `out_h` / `out_c` vectors to avoid allocating per-call return Vecs.
   - Affected: `main.rs` `process_chunk`.
   - Migration: buffers pre-allocated once per rayon thread with `Vec::with_capacity(PAGE_SIZE)`.

3. **Replace tokio file I/O with sync pread**
   - Objective: eliminate (a) per-rayon-thread `tokio::runtime::Runtime::new()`, (b) `Arc<Mutex<File>>` contention serializing all thread reads.
   - Logic:
     - Open each input file once per rayon thread via `std::fs::File::open(...)`.
     - Read each chunk via `std::os::unix::fs::FileExt::read_exact_at(&mut buf, offset)` into a pre-allocated `Vec<u8>` of size `CHUNK_SIZE`.
     - Remove all `tokio::` uses from the worker path. Drop `#[tokio::main]`; make `main` sync. Keep `chrono` for timestamp.
   - Affected: `main.rs` main + worker closures; `Cargo.toml`.
   - Migration: Linux/Unix assumption is already implied by the pipeline (libvirt, virsh). Document Unix requirement in a top-of-file comment.

4. **Stream output assembly per thread**
   - Objective: remove global `String` concatenation overhead.
   - Logic: per-thread local `String` with `write!` in the same order; at end, concatenate the 16 pre-formed strings and write once with `std::fs::write` to each output file. (The current final-write pattern is fine; the change is moving formatting into the parallel region to parallelize it.)
   - Affected: `main.rs` main function tail.
   - Migration: formatting of values uses `Display` for `u32` and `f32` as today — no format-string changes.

5. **Prune unused dependencies**
   - Objective: smaller build, fewer attack surface crates.
   - Logic: remove from `Cargo.toml`: `tokio`, `async-std`, `emd`, `ndarray`, `ml-distance`, `futures` (verify none is used by the rewritten file). Keep `hamming`, `distances`, `rayon`, `chrono`.
   - Affected: `Cargo.toml`.
   - Migration: `cargo build --release` must succeed; no API change.

## 5. Algorithms / Pseudocode

```text
# --- startup calibration ---
let zero_page = [0u8; PAGE_SIZE];
let ones_page = [1u8; PAGE_SIZE];
let zero_f = zero_page.iter().map(|&x| x as f32).collect::<Vec<_>>();
let ones_f = ones_page.iter().map(|&x| x as f32).collect::<Vec<_>>();
let cos_ident_zero    = cosine(&zero_f, &zero_f);   // may be NaN if library divides by 0
let cos_ident_nonzero = cosine(&ones_f, &ones_f);

# --- per-rayon-thread setup ---
let mut buf1 = Vec::<f32>::with_capacity(PAGE_SIZE);
let mut buf2 = Vec::<f32>::with_capacity(PAGE_SIZE);
let mut chunk1 = vec![0u8; CHUNK_SIZE];
let mut chunk2 = vec![0u8; CHUNK_SIZE];
let f1 = File::open(path1)?;
let f2 = File::open(path2)?;
let mut local_h: Vec<u32> = Vec::with_capacity(segment_pages);
let mut local_c: Vec<f32> = Vec::with_capacity(segment_pages);
let mut local_out_h = String::with_capacity(segment_pages * 6);
let mut local_out_c = String::with_capacity(segment_pages * 12);

# --- chunk loop ---
let mut offset = start_offset;
while offset < end_offset {
    let to_read = min(CHUNK_SIZE, (end_offset - offset) as usize);
    f1.read_exact_at(&mut chunk1[..to_read], offset)?;
    f2.read_exact_at(&mut chunk2[..to_read], offset)?;
    process_chunk(&chunk1[..to_read], &chunk2[..to_read],
                  &mut buf1, &mut buf2,
                  cos_ident_zero, cos_ident_nonzero,
                  &mut local_h, &mut local_c);
    offset += to_read as u64;
}

# --- per-page processing ---
for i in 0..(chunk_len / PAGE_SIZE):
    let p1 = &chunk1[i*PAGE_SIZE..(i+1)*PAGE_SIZE];
    let p2 = &chunk2[i*PAGE_SIZE..(i+1)*PAGE_SIZE];
    local_h.push(distance(p1, p2) as u32);
    if p1 == p2:
        // slice eq compiles to memcmp / SIMD
        let all_zero = p1.iter().all(|&b| b == 0);
        local_c.push(if all_zero { cos_ident_zero } else { cos_ident_nonzero });
        continue
    buf1.clear(); buf1.extend(p1.iter().map(|&x| x as f32));
    buf2.clear(); buf2.extend(p2.iter().map(|&x| x as f32));
    local_c.push(cosine(&buf1, &buf2));

# --- output emission (per thread) ---
for h in &local_h: writeln!(local_out_h, "{}", h)?;
for c in &local_c: writeln!(local_out_c, "{}", c)?;
# store local_out_h, local_out_c into slot[thread_id].

# --- final write (single-threaded) ---
join 16 slots in order; std::fs::write(hamming_path, concatenated)?; same for cosine.
```

## 6. Correctness / Failure Analysis
- **Numerical equivalence (non-identical pages).** `buf.clear(); buf.extend(bytes.map(as f32))` produces the same `Vec<f32>` contents as `bytes.iter().map(...).collect()`. The subsequent `cosine(&buf1,&buf2)` call is byte-identical to current behavior → outputs bit-identical.
- **Identical-page fast path.** For `p1 == p2`, `buf1` would equal `buf2` element-wise; the library must return the same scalar as the calibrated constant by determinism of the function on identical inputs. Validated by: byte eq → f32 eq → cosine is a pure fn of its two args. Risk: library returns `NaN` for all-zero vectors (zero norms); calibrating this once and emitting the same NaN (or whatever value) preserves output. If the library uses any nondeterministic path (unlikely for cosine), calibration still captures the canonical value.
- **Precision / conversion.** `x as f32` for `u8` is exact (f32 has 24-bit mantissa; 8-bit ints lossless). No change.
- **Edge pages / partial chunks.** Final chunk may be < CHUNK_SIZE if `file_size % CHUNK_SIZE != 0`. Code must use the actual bytes read as chunk length. `process_chunk` already iterates `chunk.len() / PAGE_SIZE`, discarding any trailing < PAGE_SIZE bytes, matching current behavior. Pre-condition assertion: `chunk_len % PAGE_SIZE == 0` when not the last segment.
- **Thread safety.** Each rayon thread owns its buffers, its own `File` handles (opened locally), and its own output `String`. No shared mutable state during the parallel region. Final merge reads immutable per-thread buffers. No locks.
- **Allocator / memory layout.** `Vec::with_capacity(PAGE_SIZE)` + `.clear()` + `.extend()` reuses the backing allocation; no re-allocation occurs as long as extend count ≤ capacity (always 4096 here). Stack usage unchanged; heap traffic drops from O(pages × 2) to O(threads × 2).
- **Error handling.** `read_exact_at` returns `io::Result`; propagate. Current code swallows errors with `unwrap_or_else(|_| vec![])` and `break`s — the spec preserves that permissive behavior as a transitional compatibility flag `STRICT_IO` (default false) to avoid changing operational semantics.
- **Output ordering.** Thread slots are written in ascending `thread_id` order at the end — identical to current global ordering.
- **Crash safety.** Output files are written once at end via `std::fs::write`; no partial files unless the process dies mid-write (same as today; reaper in the consumer handles partial outputs as job failure).

## 7. Performance Impact
- **Allocation pressure**: ~1M `Vec<f32>` heap allocations per job eliminated → measurable drop in allocator wall time and cache thrash. Expected 10–25% reduction of CPU portion.
- **Mutex contention on file handles**: current code serializes all 16 threads through two `Arc<Mutex<File>>` locks for every read. Eliminating this unblocks real parallelism on the read path. Expected 3–8× speedup on the I/O side when storage is fast (your SSD case).
- **Per-thread tokio runtime creation (16×)**: each runtime allocates worker threads and internal structures. Removal is pure win.
- **Identical-page fast path**: idle and many IO workloads produce large fractions of unchanged pages (often >50%). For those pages, cost drops from byte→f32 conversion + cosine call to one memcmp. Workload-dependent; 1.5–3× overall reduction is plausible for idle-dominated captures.
- **Combined expected wall-clock for the current 2-min-per-job bottleneck**: target ≤30s per job for a 2GB pair on your SSD, depending on workload mix. Verify empirically.

## 8. Test Plan
- **Unit**
  - Golden-output test: select two RAW pages (one identical, one differing) and assert the new `process_chunk` produces identical `(hamming, cosine)` bit-pattern to a direct call of `hamming::distance` + `distances::vectors::cosine` on the same inputs.
  - Fast-path test: all-zero identical pages → emitted cosine equals `cos_ident_zero` calibration value. All-ones identical pages → emits `cos_ident_nonzero`.
  - Partial-chunk test: feed a chunk of length `3 * PAGE_SIZE + 17` bytes; assert 3 page results emitted; 17 trailing bytes ignored.
- **Integration (binary level)**
  - Regenerate two small synthetic RAW pairs (e.g., 8 MB each); run both old and new binaries; `diff` the `hamming/*.txt` and `cosine/*.txt` outputs. Must be byte-identical.
  - Run on a real captured pair from the pipeline; same diff check.
- **Regression**
  - Output directory layout and filenames unchanged.
  - `assert_eq!(file1_size, file2_size)` still enforced.
  - Exit code 1 on arg-count mismatch preserved.
- **Benchmark / profiling**
  - `hyperfine` or `time` on a 2GB pair before/after. Record: total wall time, CPU time, max RSS.
  - Allocator profile (heaptrack) to confirm <1K allocations per job (vs. ~1M prior).
  - Confirm all 16 cores are actually busy during the parallel region (prior code had them blocked on file mutex).
- **Edge cases**
  - File size not multiple of CHUNK_SIZE: last segment handled.
  - Single-thread invocation (`THREAD_COUNT=1`): works, same outputs.
  - Read error on one file: binary exits non-zero with clear message (under new `STRICT_IO=true`).

## 9. Acceptance Criteria
- `cargo build --release` succeeds with the pruned `Cargo.toml`.
- For a real 2GB captured pair from the pipeline, `hamming/*.txt` and `cosine/*.txt` outputs are byte-identical between the pre-change and post-change binaries.
- Per-job allocation count drops by ≥99% (measured via allocator profiler) vs. current binary.
- Wall-clock per job on the current host+storage is ≤50% of the pre-change baseline for a non-idle workload; ≤30% for an idle/stable workload.
- No new unsafe code.
- No new external crate dependencies.
- All 16 rayon workers show near-equal CPU utilization for the duration of a job (no prolonged mutex-induced idling).

## 10. Rollout / Rollback
- **Rollout**: rebuild the binary (`cargo build --release` in `VM_sampler/VM_Capture/live_delta_calc/`). The consumer config (`rustDeltaCalculationProgram`) path is unchanged; replacing the binary is the deployment. Restart the consumer at a step boundary.
- **Verification on deploy**: run one short captured step; diff outputs against a baseline pair saved before the swap.
- **Rollback**: keep the previous binary at `live_delta_calc.prev`. Swap back by renaming. Zero config changes needed.

## 11. Executor Handoff Prompt
```
Implement ONLY the changes in docs/controlled-qemu-pipeline/issue/plans/rust-hotloop-opus-spec.md.

Scope:
- Edit VM_sampler/VM_Capture/live_delta_calc/src/main.rs and VM_sampler/VM_Capture/live_delta_calc/Cargo.toml.
- Do NOT change CLI, output paths, output line format, page/chunk sizes, THREAD_COUNT, or the hamming/cosine libraries.

Tasks:
1. In main.rs, drop `#[tokio::main]`; make `main` a plain sync fn returning `io::Result<()>`. Remove all `tokio::fs::*` and `async-std::*` imports and use. Replace file I/O with `std::fs::File` + `std::os::unix::fs::FileExt::read_exact_at`.
2. Open input files once per rayon worker (inside the closure). Pre-allocate two `Vec<u8>` of length CHUNK_SIZE as chunk buffers, and two `Vec<f32>` of capacity PAGE_SIZE as conversion buffers. All owned by the closure; no Arc/Mutex around files or buffers.
3. At main() startup, compute two `f32` calibration constants by calling `distances::vectors::cosine` on two identical all-zero `[f32; PAGE_SIZE]` (as Vec) and on two identical all-ones `[f32; PAGE_SIZE]`. Pass both constants by value into worker closures.
4. Rewrite `process_chunk` with signature:
   fn process_chunk(chunk1: &[u8], chunk2: &[u8],
                    buf1: &mut Vec<f32>, buf2: &mut Vec<f32>,
                    cos_ident_zero: f32, cos_ident_nonzero: f32,
                    out_h: &mut Vec<u32>, out_c: &mut Vec<f32>)
   Per page: always compute hamming via `distance(p1,p2) as u32`. If p1 == p2 (slice eq), push cos_ident_zero if all bytes are zero else cos_ident_nonzero. Else clear/extend buf1 and buf2 from bytes and push `cosine(&buf1,&buf2)`. Never allocate a new Vec inside the loop.
5. In each rayon worker, after the chunk loop, format the per-thread results into two local `String`s (one per output stream) using `writeln!`. Store into Vec<Vec<String>> slots indexed by thread_id. After rayon join, concatenate slots in ascending thread order and write once via `std::fs::write` to the hamming and cosine output paths. Keep the existing timestamp + `output_dir/{hamming,cosine}/` layout and filenames exactly.
6. Last chunk may be smaller than CHUNK_SIZE; compute `to_read = min(CHUNK_SIZE, (end_offset-offset) as usize)` and pass `&chunk1[..to_read]` etc. to process_chunk.
7. Preserve current permissive-I/O behavior behind an env-gated `STRICT_IO` flag (default false = log+break like today; true = propagate errors).
8. Cargo.toml: remove `tokio`, `async-std`, `emd`, `ndarray`, `ml-distance`, `futures`. Keep `hamming`, `distances`, `rayon`, `chrono`. Confirm `cargo build --release` succeeds.

Verification (must all pass):
- `cargo build --release` clean.
- Byte-diff of outputs from a real 2GB captured pair against a baseline produced by the old binary: zero differences.
- `hyperfine` before/after on the same pair: wall-clock ≥2× faster.
- No `unsafe` blocks introduced.

Do NOT:
- Introduce a custom cosine implementation (preserve library call exactly).
- Change output text format, filenames, or directory layout.
- Add new crate dependencies.
- Parallelize differently than rayon over THREAD_COUNT segments.

Deliverables: one commit modifying main.rs and Cargo.toml only, plus a short benchmark note in the commit body (before/after wall-clock for one sample pair).
```
