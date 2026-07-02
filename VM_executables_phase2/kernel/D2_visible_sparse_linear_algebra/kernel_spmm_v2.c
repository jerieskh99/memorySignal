/* kernel_spmm_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Sparse Linear Algebra dwarf (Berkeley motif D2), the SpMM variant: a sparse
 * matrix multiplied by a DENSE matrix, C = A * B. This is the core aggregation
 * kernel of a Graph Neural Network -- A is the (sparse) graph adjacency, B is
 * the dense per-node feature matrix, and C is the aggregated features. A plain
 * SpMV (kernel_spmv_v2) multiplies the same sparse A by a single VECTOR and
 * writes back only a tiny vector; SpMM multiplies it by many columns at once
 * and writes back a whole dense matrix.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs quiet SpMV)
 * ----------------------------------------------------------------------------
 * SpMV and SpMM READ almost the same way: both stream the sparse structure and
 * gather rows/elements of the dense operand. Reads are invisible to a host
 * memory WRITE-signal, so on the read side the two look alike. The difference
 * is entirely on the WRITE side. SpMV's output is one length-M vector -- a
 * negligible write, so it reads as near-idle (that is precisely why it is used
 * as the QUIET control). SpMM's output C is a full DENSE M x N matrix that is
 * zeroed and then completely rewritten every pass. That large dense rewrite,
 * driven by the sparse structure of A, is the VISIBLE signature: the same
 * sparse-algebra motif, but WRITE-VISIBLE instead of quiet.
 *
 * PICTURE (top view):  sparse A (CSR) times dense B accumulates into dense C.
 *
 *      A (M x K, CSR)                B (K x N, dense)         C (M x N, dense)
 *      row i -> nonzeros             row k is a length-N      row i is a length-N
 *      (col_idx[t], vals[t])         dense feature vector     accumulator
 *
 *        row_ptr : [ r0 r1 .. rM ]   B[k, :] = [ b .. b ]     C[i, :] = [ c .. c ]
 *        col_idx : [ .. k .. ]  ---------------^                        ^
 *        vals    : [ .. a .. ]                  \  (scale by a)        /
 *                                                 +-- a * B[k,:] --> C[i,:]  (AXPY)
 *
 *      For row i:  C[i, :] = sum over its nonzeros (k=col_idx[t], a=vals[t])
 *                            of  a * B[k, :]      (each term a dense length-N AXPY)
 *
 *      C is the dominant buffer and the large visible write; A and B are READ.
 *
 * ============================================================================
 * ALGORITHM (per measured pass)
 * ============================================================================
 *   1. Re-seed the sparse matrix A: for each of the M rows, pick nnz-per-row
 *      random column indices in [0, K) and random coefficients (a fresh random
 *      CSR every pass, so the product is never trivially constant).
 *   2. Re-seed the dense operand B (K x N random values).
 *   3. Zero C, then for each row i walk A's nonzeros and, for each nonzero
 *      (k, a), add a * B[k, :] into C[i, :] (a dense AXPY of length N).
 *
 * The AXPY inner loop over the N columns is what turns the sparse structure
 * into a large dense write: every nonzero of A touches an entire row of C.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Sparse Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + first seed) / measure (SpMM passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_spmm_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign sparse x dense matmul; WRITE-VISIBLE, Sparse-LA)\n"
"  --rows M              Rows of sparse A / output C (default 4096)\n"
"  --inner K             Inner dim: cols of A, rows of B (default 4096)\n"
"  --cols N              Cols of dense B / output C (default 64)\n"
"  --nnz-per-row K       Nonzeros per row of A (default 16)\n"
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
    long long rows       = p2_get_i64(argc, argv, "--rows", 4096);
    long long inner      = p2_get_i64(argc, argv, "--inner", 4096);
    long long cols       = p2_get_i64(argc, argv, "--cols", 64);
    long long nnz_pr     = p2_get_i64(argc, argv, "--nnz-per-row", 16);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (rows  < 1  || rows  > 8000000LL)  { P2_LOG_ERR("rows %lld out of range (1..8e6)", rows); return 2; }
    if (inner < 1  || inner > 8000000LL)  { P2_LOG_ERR("inner %lld out of range (1..8e6)", inner); return 2; }
    if (cols  < 1  || cols  > 65536LL)    { P2_LOG_ERR("cols %lld out of range (1..65536)", cols); return 2; }
    if (nnz_pr < 1 || nnz_pr > (long long)inner) {
        P2_LOG_ERR("nnz-per-row %lld out of range (1..inner=%lld)", nnz_pr, inner); return 2;
    }
    size_t M = (size_t)rows, K = (size_t)inner, N = (size_t)cols, R = (size_t)nnz_pr;
    size_t nnz = M * R;                                 /* fixed nnz-per-row => total nnz = M*R */
    /* Buffers: sparse A in CSR (row_ptr[M+1], col_idx[nnz], vals[nnz]),
     * dense B (K x N), dense output C (M x N). C is the dominant/visible one. */
    size_t bytes_C   = M * N * sizeof(double);          /* the large visible write */
    size_t bytes_B   = K * N * sizeof(double);
    size_t bytes_row = (M + 1) * sizeof(size_t);
    size_t bytes_col = nnz * sizeof(uint32_t);
    size_t bytes_val = nnz * sizeof(double);
    size_t bytes     = bytes_C + bytes_B + bytes_row + bytes_col + bytes_val;
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("total bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Sparse Linear Algebra");
    p2_meta_kv_str(&m, "scheme", "SpMM: sparse A (CSR) x dense B -> dense C rewritten each pass (visible)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "rows_M", rows);
    p2_meta_kv_i64(&m, "inner_K", inner);
    p2_meta_kv_i64(&m, "cols_N", cols);
    p2_meta_kv_i64(&m, "nnz_per_row", nnz_pr);
    p2_meta_kv_u64(&m, "nnz", nnz);
    p2_meta_kv_u64(&m, "output_C_bytes", bytes_C);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* Dense output C is the dominant buffer -> mmap + madvise + mlock it. It is
     * zeroed and completely rewritten every pass, which is the workload's
     * signature (large dense) write, driven by the sparse structure of A. */
    double *C = (double *)mmap(NULL, bytes_C, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (C == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes_C, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(C, bytes_C, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(C, bytes_C);

    /* Sparse A (CSR) and dense B are READ each pass; they live in plain malloc'd
     * arrays. Because nnz-per-row is fixed, row_ptr is regular (row i spans
     * [i*R, i*R + R)); we still store it explicitly to keep a true CSR shape. */
    size_t   *row_ptr = (size_t   *)malloc((M + 1) * sizeof(size_t));
    uint32_t *col_idx = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    double   *vals    = (double   *)malloc(nnz * sizeof(double));
    double   *B       = (double   *)malloc(K * N * sizeof(double));
    if (!row_ptr || !col_idx || !vals || !B) {
        P2_LOG_ERR("malloc failed");
        free(row_ptr); free(col_idx); free(vals); free(B);
        munmap(C, bytes_C);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    if (!no_mlock) { p2_mlock_soft(col_idx, nnz * sizeof(uint32_t)); p2_mlock_soft(vals, nnz * sizeof(double)); }

    /* CSR row offsets are constant for a fixed nnz-per-row: set them once. */
    for (size_t i = 0; i <= M; i++) row_ptr[i] = i * R;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Seed A's nonzeros (random columns in [0,K) + random coefficients) and the
     * dense operand B once during warmup; the measured passes re-seed both. */
    for (size_t t = 0; t < nnz; t++) {
        col_idx[t] = (uint32_t)(p2_rng_next(&rng) % (uint64_t)K);
        vals[t]    = rng_unit(&rng);
    }
    for (size_t i = 0; i < K * N; i++) B[i] = rng_unit(&rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed the sparse structure A and the dense operand B each pass so the
         * product is not trivially constant. These are READS during the matmul;
         * only C (below) is the sizeable visible write. */
        for (size_t t = 0; t < nnz; t++) {
            col_idx[t] = (uint32_t)(p2_rng_next(&rng) % (uint64_t)K);
            vals[t]    = rng_unit(&rng);
        }
        for (size_t i = 0; i < K * N; i++) B[i] = rng_unit(&rng);

        /* SpMM: zero C, then for each row i accumulate a * B[k,:] into C[i,:] for
         * every nonzero (k=col_idx[t], a=vals[t]) of A's row i. The AXPY over the
         * N columns rewrites the ENTIRE dense C every pass -- the visible write. */
        memset(C, 0, bytes_C);                          /* full dense clear */
        for (size_t i = 0; i < M; i++) {
            double *crow = &C[i * N];                    /* row i of C (length N) */
            for (size_t t = row_ptr[i]; t < row_ptr[i + 1]; t++) {
                uint32_t k = col_idx[t];                 /* column of A / row of B */
                double   a = vals[t];                    /* the nonzero coefficient */
                const double *brow = &B[(size_t)k * N];  /* row k of B (length N) */
                for (size_t j = 0; j < N; j++) crow[j] += a * brow[j];   /* dense AXPY */
            }
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = C[(M / 2) * N + (N / 2)];    /* one live output element */

    free(row_ptr); free(col_idx); free(vals); free(B);
    munmap(C, bytes_C);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "sink", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "WRITE-VISIBLE: dense C rewritten each pass is the tell vs quiet SpMV (tiny vector); A/B reads invisible");
    p2_meta_close(&m);
    return 0;
}
