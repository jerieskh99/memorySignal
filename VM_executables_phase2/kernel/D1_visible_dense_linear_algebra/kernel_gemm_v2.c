/* kernel_gemm_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  BLOCKED GEMM:  tiled dense matrix multiply  C = A * B  on N x N doubles
 * ============================================================================
 *
 *  DWARF   : Dense Linear Algebra (Berkeley computational motif D1)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of a workload that rewrites its ENTIRE
 *            output footprint every pass. GEMM is the canonical dense kernel,
 *            so its static full-matrix rewrite is the reference against which
 *            the migrating fronts of LU and QR are contrasted.
 *
 *  PICTURE (top view):
 *      C  is rewritten in full, one BS x BS tile at a time, every pass.
 *      Unlike LU/QR the written region never shrinks or grows -- the whole
 *      of C is touched on every single pass (a static, full-footprint front).
 *
 *          C  ( N x N )            =        A            *        B
 *      +----+----+----+                +------------+       +----+----+----+
 *      | T  | T  | T  |   each T is    |  row band  |       | B  | B  | B  |
 *      +----+----+----+   one BS x BS  |  of A read |       | tile columns|
 *      | T  | T  | T  |   output tile  +------------+       |  of B read  |
 *      +----+----+----+   accumulated  (reads only,         +----+----+----+
 *      | T  | T  | T  |   in place      invisible to         (reads only,
 *      +----+----+----+                 the write-signal)     invisible)
 *
 *  ALGORITHM (per pass):
 *      1. Re-seed A with fresh random entries (a full-matrix write of A).
 *      2. Zero the output C with memset (a full-matrix write of C).
 *      3. Multiply in BS x BS blocks: for each (ii, kk, jj) tile triple, take
 *         a scalar A[i][k] and fold the B row into the C row,
 *         C[i][j] += A[i][k] * B[k][j], accumulating each C tile in place.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      The full N x N output matrix C is written on every pass, plus the A
 *      re-seed -- a large, STATIC write footprint that does not migrate.
 *      Reads of A and B are invisible to a write-only signal. Honest caveat:
 *      blocking changes only the ORDER in which C's cells are written, not the
 *      set of cells written; over a snapshot interval that order averages out,
 *      so blocked and naive GEMM present the same footprint. The distinguishing
 *      feature is the large C footprint itself, not the tiling.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Dense Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (matmuls) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_gemm_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign blocked dense matmul; Dense-LA kernel)\n"
"  --dim N               Matrix side length (default 1024; uses 3 * N*N * 8 bytes)\n"
"  --block BS            Tile size (default 64)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

static inline double rng_unit(p2_rng_t *r) {
    return (double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long dim        = p2_get_i64(argc, argv, "--dim", 1024);
    long long block      = p2_get_i64(argc, argv, "--block", 64);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (dim < 32 || dim > 16384) { P2_LOG_ERR("dim %lld out of range (32..16384)", dim); return 2; }
    if (block < 8 || block > 1024) { P2_LOG_ERR("block %lld out of range (8..1024)", block); return 2; }
    size_t N = (size_t)dim, BS = (size_t)block;
    size_t cells = N * N;
    size_t bytes = 3 * cells * sizeof(double);   /* A, B, C */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Dense Linear Algebra");
    p2_meta_kv_str(&m, "scheme", "blocked (tiled) GEMM, C = A*B");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "dim", dim);
    p2_meta_kv_i64(&m, "block", block);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *arena = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (arena == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(arena, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(arena, bytes);
    double *A = arena, *B = arena + cells, *C = arena + 2 * cells;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < cells; i++) { A[i] = rng_unit(&rng); B[i] = rng_unit(&rng); }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed A and zero C so every pass writes the full A and C footprints
         * afresh. The C rewrite is the workload's signature write; re-seeding A
         * keeps the input changing so the compiler cannot hoist the product. */
        for (size_t i = 0; i < cells; i++) A[i] = rng_unit(&rng);   /* re-seed A */
        memset(C, 0, cells * sizeof(double));
        /* Blocked (tiled) triple loop. Iterating over BS x BS tiles keeps the
         * active slices of A, B and C small enough to stay cache-resident, so
         * each element is reused many times before it is evicted. The tile order
         * only reorders the writes to C; it does not change which cells of C are
         * written (still all N*N of them every pass). */
        for (size_t ii = 0; ii < N; ii += BS)
            for (size_t kk = 0; kk < N; kk += BS)
                for (size_t jj = 0; jj < N; jj += BS) {
                    /* Clamp each tile's far edge so the final partial tile does
                     * not run past the matrix when N is not a multiple of BS. */
                    size_t iE = ii + BS < N ? ii + BS : N;
                    size_t kE = kk + BS < N ? kk + BS : N;
                    size_t jE = jj + BS < N ? jj + BS : N;
                    for (size_t i = ii; i < iE; i++)
                        for (size_t k = kk; k < kE; k++) {
                            /* Hoist the scalar A[i][k] and the B / C row bases out
                             * of the innermost loop so the hot loop is a pure
                             * scale-and-accumulate over one contiguous C row. */
                            double aik = A[i * N + k];
                            const double *brow = B + k * N;
                            double *crow = C + i * N;
                            /* Rank-1 update of one C row tile: C[i][j] += A[i][k]*B[k][j]. */
                            for (size_t j = jj; j < jE; j++) crow[j] += aik * brow[j];
                        }
                }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = C[(N / 2) * N + (N / 2)];

    munmap(arena, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "center_value", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: full output-matrix C rewrite (large footprint) + A re-seed");
    p2_meta_close(&m);
    return 0;
}
