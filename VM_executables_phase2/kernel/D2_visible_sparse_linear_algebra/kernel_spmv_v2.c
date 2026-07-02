/* kernel_spmv_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  SPARSE MATRIX-VECTOR MULTIPLY:  y = A*x, gather-dominated (QUIET control)
 * ============================================================================
 *
 *  DWARF   : Sparse Linear Algebra (D2) (Berkeley 13 computational motif)
 *  FAMILY  : IDLE                       (first-division, memory-signature label)
 *  PURPOSE : Serve as a deliberate QUIET / null control. This workload has a
 *            large ACTIVE footprint but a tiny VISIBLE one, so it should read as
 *            near-idle. It tests the null hypothesis: is a sparse / gather access
 *            pattern invisible to the host memory WRITE-signal? (expected yes.)
 *
 *  The kernel computes y = A*x for a sparse matrix A stored in CSR-like form with
 *  a fixed K nonzeros per row. Because every row has exactly K entries, the usual
 *  CSR row_ptr is implicit -- row i simply owns the slice [i*K, i*K + K) of the
 *  col_idx and vals arrays -- so only those two parallel arrays are stored. For
 *  each output element, y[i] = sum over k of vals[k] * x[col_idx[k]]; the term
 *  x[col_idx[k]] is an INDIRECT GATHER, reading x at a data-dependent, scattered
 *  index rather than sequentially.
 *
 *  The point of the control is the asymmetry between reads and writes. The big
 *  structure (col_idx + vals, potentially hundreds of MB) and the source vector x
 *  are all READ; the only thing WRITTEN back is the small output vector y (plus
 *  the small x refill each pass). A host write-signal sees only those tiny
 *  writes, never the large read-only sweep -- hence near-idle despite real work.
 *
 *  PICTURE (top view):  CSR-like layout and the indirect gather x[col_idx[k]].
 *
 *      row i owns a length-K slice (base = i*K) of two parallel arrays:
 *
 *        col_idx : [ .. | c0 c1 c2 .. c(K-1) | .. ]   scattered column numbers
 *        vals    : [ .. | v0 v1 v2 .. v(K-1) | .. ]   the matching coefficients
 *                          |  |        \
 *                          |  |         \  (each c is an index INTO x)
 *                          v  v          v
 *        x       : [ x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 .. ]   <-- GATHER reads (random)
 *
 *        y[i] = v0*x[c0] + v1*x[c1] + ... + v(K-1)*x[c(K-1)]
 *
 *        y       : [ .. y[i] .. ]   <-- the ONLY sizeable writes (small, near-idle)
 *
 *  ALGORITHM (per measured pass):
 *      1. Re-seed the source vector x with fresh random values.
 *      2. For each row i, walk its K nonzeros, gather x at each stored column
 *         index, multiply by the stored value, accumulate, and store into y[i].
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Almost nothing. The dominant memory traffic is READS -- the streamed
 *      col_idx / vals arrays and the scattered gather into x -- all invisible to
 *      a write-signal. The only writes are the small y vector and the small x
 *      refill, so the observable footprint is tiny and the workload looks
 *      near-idle. HONEST CAVEAT: this "quiet" reading holds precisely because the
 *      sparse structure is never modified; a variant that wrote back into the
 *      matrix would no longer be a quiet control.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: IDLE (quiet control). Dwarf: Sparse Linear Algebra.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (build matrix) / measure (matvecs) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_spmv_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign sparse matrix-vector; QUIET control, Sparse-LA)\n"
"  --rows N              Matrix rows / vector length (default 1000000)\n"
"  --nnz-per-row K       Nonzeros per row (default 16)\n"
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
    long long rows       = p2_get_i64(argc, argv, "--rows", 1000000);
    long long nnz_pr     = p2_get_i64(argc, argv, "--nnz-per-row", 16);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (rows < 1000 || rows > 200000000LL) { P2_LOG_ERR("rows %lld out of range", rows); return 2; }
    if (nnz_pr < 1 || nnz_pr > 1024) { P2_LOG_ERR("nnz-per-row %lld out of range (1..1024)", nnz_pr); return 2; }
    size_t N = (size_t)rows, K = (size_t)nnz_pr;
    size_t nnz = N * K;
    size_t bytes = nnz * (sizeof(uint32_t) + sizeof(double)) + 2 * N * sizeof(double);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "IDLE");
    p2_meta_kv_str(&m, "dwarf", "Sparse Linear Algebra");
    p2_meta_kv_str(&m, "role", "quiet control: gather-dominated, read-only structure");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "rows", rows);
    p2_meta_kv_i64(&m, "nnz_per_row", nnz_pr);
    p2_meta_kv_u64(&m, "nnz", nnz);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    uint32_t *col = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    double   *val = (double *)malloc(nnz * sizeof(double));
    double   *x   = (double *)malloc(N * sizeof(double));
    double   *y   = (double *)malloc(N * sizeof(double));
    if (!col || !val || !x || !y) {
        P2_LOG_ERR("malloc failed");
        free(col); free(val); free(x); free(y);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    if (!no_mlock) { p2_mlock_soft(col, nnz * sizeof(uint32_t)); p2_mlock_soft(val, nnz * sizeof(double)); }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Build the sparse matrix once, during warmup. Every nonzero gets a random
     * column in [0, N) and a random coefficient; the random columns are what make
     * the later gather scattered and cache-unfriendly. After this the structure
     * is never written again -- that read-only-ness is exactly what keeps the
     * measured phase quiet on the write-signal. */
    for (size_t k = 0; k < nnz; k++) {
        col[k] = (uint32_t)(p2_rng_next(&rng) % (uint64_t)N);
        val[k] = rng_unit(&rng);
    }
    for (size_t i = 0; i < N; i++) x[i] = rng_unit(&rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed x each pass so the product is not trivially constant. This is a
         * small O(N) write over one vector -- one of only two things this kernel
         * ever writes -- so it stays cheap and near-idle on the write-signal. */
        for (size_t i = 0; i < N; i++) x[i] = rng_unit(&rng);   /* re-seed x */
        /* The matvec itself: for each row, accumulate its K nonzeros. Reads
         * dominate -- val/col are streamed sequentially while x is gathered at the
         * random column indices col[..] -- and the ONLY write is the single
         * y[i] per row. Large read footprint, tiny visible write footprint. */
        for (size_t i = 0; i < N; i++) {
            double s = 0.0;
            size_t base = i * K;                                /* row i owns [base, base+K) */
            for (size_t kk = 0; kk < K; kk++) s += val[base + kk] * x[col[base + kk]];  /* gather + MAC */
            y[i] = s;                                           /* the lone write per row */
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = y[N / 2];

    free(col); free(val); free(x); free(y);

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
                   "expected NEAR-IDLE: read-only sparse structure invisible; only x/y (small) change");
    p2_meta_close(&m);
    return 0;
}
