/* kernel_lu_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  LU FACTORISATION:  in-place Doolittle elimination  A = L * U  (no pivoting)
 * ============================================================================
 *
 *  DWARF   : Dense Linear Algebra (Berkeley computational motif D1)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of a workload whose active write region
 *            SHRINKS over time. LU eliminates one pivot column per step and
 *            only ever touches the trailing submatrix, so its footprint retreats
 *            toward the bottom-right corner -- the mirror image of QR's growing
 *            front, and distinct from GEMM's static full-matrix rewrite.
 *
 *  PICTURE (top view):
 *      At step k, only the trailing submatrix A[k+1..][k+1..] is rewritten.
 *      "d" marks the just-fixed rows/columns (never touched again); "#" marks
 *      the active trailing block that still gets updated; the "#" block shrinks
 *      toward the lower-right as k advances.
 *
 *          k = 0            k = 1            k = 2
 *      +--------------+  +--------------+  +--------------+
 *      | # # # # # #  |  | d d d d d d  |  | d d d d d d  |
 *      | # # # # # #  |  | d # # # # #  |  | d d d d d d  |
 *      | # # # # # #  |  | d # # # # #  |  | d d # # # #  |
 *      | # # # # # #  |  | d # # # # #  |  | d d # # # #  |
 *      | # # # # # #  |  | d # # # # #  |  | d d # # # #  |
 *      +--------------+  +--------------+  +--------------+
 *          full            shrinking          smaller  ->  ... -> single cell
 *
 *  ALGORITHM (per factorisation, for k = 0 .. N-1):
 *      1. Read the pivot A[k][k] (kept safely non-zero by diagonal dominance).
 *      2. Scale the sub-diagonal column: A[i][k] /= A[k][k]  for all i > k.
 *         These multipliers are the entries of L, stored in place.
 *      3. Rank-1 update of the trailing submatrix:
 *         A[i][j] -= A[i][k] * A[k][j]  for all i, j > k.
 *         The surviving upper triangle (including the diagonal) becomes U.
 *      Each measure pass first re-seeds A and refactorises from scratch.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      One large N x N matrix whose actively-written region migrates and
 *      shrinks across the factorisation -- a shrinking front on a fixed buffer.
 *      Honest caveat: this retreat is only visible if a single factorisation
 *      lasts long enough to span several host snapshot intervals; for small N a
 *      whole factorisation finishes inside one snapshot and the front collapses
 *      to a full-matrix touch, indistinguishable from GEMM. Large N is required.
 *
 *  Real-world use: the linear-solve workhorse -- SPICE circuit simulation,
 *  finite-element solvers, interior-point optimisation steps (LAPACK dgetrf).
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Dense Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (factorisations) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_lu_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign in-place LU factorisation; Dense-LA kernel)\n"
"  --dim N               Matrix side length (default 1024; uses N*N * 8 bytes)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on matrix bytes (default 8192)\n"
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

/* Re-seed A with fresh random entries in [0,1), then force strict diagonal
 * dominance. Because this benchmark factorises WITHOUT pivoting, a small or zero
 * pivot A[k][k] would blow up the multipliers (or divide by zero); adding N to
 * each diagonal entry guarantees every pivot dominates its row, which keeps the
 * unpivoted elimination numerically stable pass after pass. */
static void reseed(double *A, size_t N, p2_rng_t *rng) {
    for (size_t i = 0; i < N; i++) {
        double *row = A + i * N;
        for (size_t j = 0; j < N; j++) row[j] = rng_unit(rng);
        row[i] += (double)N;   /* dominant diagonal */
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long dim        = p2_get_i64(argc, argv, "--dim", 1024);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (dim < 16 || dim > 16384) { P2_LOG_ERR("dim %lld out of range (16..16384)", dim); return 2; }
    size_t N = (size_t)dim;
    size_t bytes = N * N * sizeof(double);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("matrix bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Dense Linear Algebra");
    p2_meta_kv_str(&m, "scheme", "in-place LU factorisation (Doolittle, diagonally-dominant; shrinking trailing front)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "dim", dim);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *A = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (A == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(A, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(A, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    reseed(A, N, &rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t facts = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        reseed(A, N, &rng);   /* fresh, diagonally-dominant matrix each pass */
        /* In-place Doolittle LU. Outer loop advances the pivot k; everything at
         * or above/left of row/column k is finished once this iteration ends, so
         * the region still being written is the trailing block below-and-right of
         * the pivot. That block shrinks by one row and one column each step,
         * producing the retreating write front that is this workload's tell. */
        for (size_t k = 0; k < N; k++) {
            double inv = 1.0 / A[k * N + k];   /* reciprocal pivot (safe: diag-dominant) */
            const double *ak = A + k * N;      /* pivot row k, the source of the update */
            for (size_t i = k + 1; i < N; i++) {   /* each trailing row below the pivot */
                double *ai = A + i * N;
                /* Multiplier f = A[i][k]/A[k][k]. Stored back into A[i][k] as the
                 * L factor (the strict lower triangle holds L; the diagonal of L
                 * is an implicit 1 and is never stored). */
                double f = ai[k] * inv;
                ai[k] = f;
                /* Rank-1 elimination of the trailing row: subtract f times the
                 * pivot row from this row, but only for columns j > k -- exactly
                 * the trailing submatrix. The result left in the upper triangle
                 * (columns >= k) is the U factor. */
                for (size_t j = k + 1; j < N; j++) ai[j] -= f * ak[j];
            }
        }
        facts++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = A[(N - 1) * N + (N - 1)];   /* last U pivot */

    munmap(A, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "factorisations", facts);
    p2_meta_kv_f64(&m, "last_pivot", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "shrinking-front tell needs one factorisation to span multiple 500ms snapshots (large N)");
    p2_meta_close(&m);
    return 0;
}
