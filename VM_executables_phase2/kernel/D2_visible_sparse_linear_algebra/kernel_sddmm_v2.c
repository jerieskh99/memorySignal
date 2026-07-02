/* kernel_sddmm_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  SAMPLED DENSE-DENSE MATRIX MULTIPLY (SDDMM):  dense dot, sampled -> sparse
 * ============================================================================
 *
 *  DWARF   : Sparse Linear Algebra (Berkeley computational motif D2)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of a sparse-linear-algebra workload whose
 *            arithmetic is a FULL dense dot product but whose OUTPUT is sparse.
 *            SDDMM is the kernel behind graph-attention networks and recommender
 *            systems: it multiplies two dense matrices but keeps the product
 *            only at the non-zero positions of a sparse mask. It is the loud
 *            counterpart to the quiet SpMV control -- same dwarf, opposite tell.
 *
 *  WHY IT IS A DISTINCT MEMORY SIGNATURE (vs the quiet kernel_spmv_v2)
 *  ----------------------------------------------------------------------------
 *  SpMV (y = A*x) reads a large sparse structure and writes only a tiny output
 *  vector, so it reads as near-idle. SDDMM shares the sparse-mask read pattern
 *  but INVERTS the write balance: for every one of the nnz mask positions it
 *  computes a fresh dense dot product and STORES the sampled value. The output
 *  O has one double per mask non-zero -- a large, scattered, sparse write that a
 *  host write-signal sees plainly. The dense reads of A and B (the two feature
 *  matrices) are invisible; the sampled sparse output is the visible signature.
 *
 *  DEFINITION.  Two dense feature matrices: A is M x K (row-major) and B is
 *  N x K (row-major, so row j of B is a K-vector). A sparse mask S has a CSR
 *  position structure over an M x N grid: row_ptr[M+1] delimits each row's slice
 *  of col_idx[nnz], and s_val[nnz] carries a per-entry scalar. The output O has
 *  exactly one value per mask non-zero:
 *
 *      O[t] = s_val[t] * dot( A[i, :], B[j, :] )
 *
 *  for the t-th non-zero, which sits at row i and column j = col_idx[t]. The dot
 *  product runs over the full K-length feature vectors -- dense arithmetic -- but
 *  is evaluated ONLY at the sampled (i, j) positions -- sparse output.
 *
 *  PICTURE (top view):  dense rows dotted only where the mask fires.
 *
 *        A (M x K)                 B (N x K)               O over mask nnz
 *      +-----------+             +-----------+           row i: [ .. .. .. ]
 *      | A[i,:] -> |  dot with   | <- B[j,:] |   ==>       positions j in
 *      +-----------+  full K     +-----------+             col_idx[row_ptr[i]
 *       (dense read)              (dense read)              .. row_ptr[i+1])
 *                                                          are the ONLY writes
 *      row i owns O[ row_ptr[i] .. row_ptr[i+1] ) and, for each such slot t,
 *      O[t] = s_val[t] * ( A[i,0]*B[j,0] + A[i,1]*B[j,1] + ... + A[i,K-1]*B[j,K-1] ).
 *
 *  ALGORITHM (per measured pass):
 *      1. Re-seed A and B with fresh random features (dense writes to malloc'd
 *         input matrices -- invisible-ish churn, kept changing so the compiler
 *         cannot hoist the product).
 *      2. Re-seed the mask: for each row draw nnz-per-row random columns in
 *         [0, N) into col_idx, and fresh s_val scalars. (row_ptr is fixed: every
 *         row owns exactly nnz-per-row slots, so slot t = i*nnz_pr + local.)
 *      3. For each mask non-zero at (i, j), dot the length-K rows A[i,:] and
 *         B[j,:], scale by s_val[t], and STORE into O[t] -- the sampled sparse
 *         output living in the mmap'd buffer.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      The sampled output O -- (M * nnz-per-row) doubles in an mmap'd buffer --
 *      is written in full every pass. That is the dominant, VISIBLE write: a
 *      sparse, scattered footprint distinct from GEMM's dense full-matrix rewrite
 *      and from SpMV's near-silent tiny vector. The dense reads of A[i,:] and the
 *      gathered rows B[j,:] are invisible to a write-only signal. HONEST CAVEAT:
 *      the input re-seed of A and B (kept small relative to O when K is modest)
 *      and the mask rebuild also write; O dominates when M*nnz-per-row far
 *      exceeds M*K + N*K, i.e. when the sampled output is the largest buffer.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Sparse Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (SDDMM passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_sddmm_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign sampled dense-dense matmul; Sparse-LA kernel)\n"
"  --rows M              Rows of A / mask rows (default 8192)\n"
"  --cols N              Rows of B / mask columns (default 8192)\n"
"  --feat K              Feature length (dot-product dimension) (default 32)\n"
"  --nnz-per-row K       Sampled non-zeros per mask row (default 32)\n"
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
    long long rows       = p2_get_i64(argc, argv, "--rows", 8192);
    long long cols       = p2_get_i64(argc, argv, "--cols", 8192);
    long long feat       = p2_get_i64(argc, argv, "--feat", 32);
    long long nnz_pr     = p2_get_i64(argc, argv, "--nnz-per-row", 32);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (rows < 16 || rows > 100000000LL) { P2_LOG_ERR("rows %lld out of range (16..1e8)", rows); return 2; }
    if (cols < 16 || cols > 100000000LL) { P2_LOG_ERR("cols %lld out of range (16..1e8)", cols); return 2; }
    if (feat < 1 || feat > 4096) { P2_LOG_ERR("feat %lld out of range (1..4096)", feat); return 2; }
    if (nnz_pr < 1 || nnz_pr > (long long)cols) {
        P2_LOG_ERR("nnz-per-row %lld out of range (1..cols=%lld)", nnz_pr, cols); return 2;
    }
    size_t M = (size_t)rows, N = (size_t)cols, K = (size_t)feat, KPR = (size_t)nnz_pr;
    size_t nnz = M * KPR;                            /* one output value per sampled position */
    /* O is the mmap'd sampled sparse output (the dominant, visible write). The
     * mask position arrays (col_idx + s_val) and the two dense feature matrices
     * A (M x K) and B (N x K) are all counted toward the cap too. */
    size_t o_bytes    = nnz * sizeof(double);
    size_t mask_bytes = nnz * sizeof(uint32_t) + nnz * sizeof(double);   /* col_idx + s_val */
    size_t ab_bytes   = (M * K + N * K) * sizeof(double);                /* dense A and B    */
    size_t bytes      = o_bytes + mask_bytes + ab_bytes;
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Sparse Linear Algebra");
    p2_meta_kv_str(&m, "scheme", "SDDMM: O = s_val .* (A rows . B rows), sampled at sparse mask positions");
    p2_meta_kv_str(&m, "role", "visible writer: full dense dot product stored only at sparse mask positions");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "rows", rows);
    p2_meta_kv_i64(&m, "cols", cols);
    p2_meta_kv_i64(&m, "feat", feat);
    p2_meta_kv_i64(&m, "nnz_per_row", nnz_pr);
    p2_meta_kv_u64(&m, "nnz", nnz);
    p2_meta_kv_u64(&m, "output_bytes", o_bytes);
    p2_meta_kv_u64(&m, "mask_bytes", mask_bytes);
    p2_meta_kv_u64(&m, "ab_bytes", ab_bytes);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The sampled sparse output O is the dominant buffer and the workload's
     * signature write -> mmap + mlock it (it is rewritten in full every pass). */
    double *O = (double *)mmap(NULL, o_bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (O == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", o_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(O, o_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(O, o_bytes);

    /* Dense feature matrices A (M x K), B (N x K) and the sparse mask position
     * arrays (col_idx + s_val) are read-side state (plus a small per-pass
     * re-seed), so they live in plain malloc'd memory. row_ptr is implicit --
     * every row owns exactly KPR slots -- so slot t = i*KPR + local. */
    double   *A   = (double *)malloc(M * K * sizeof(double));
    double   *B   = (double *)malloc(N * K * sizeof(double));
    uint32_t *col = (uint32_t *)malloc(nnz * sizeof(uint32_t));   /* sampled column j per slot */
    double   *sv  = (double *)malloc(nnz * sizeof(double));       /* per-entry mask scalar s_val */
    if (!A || !B || !col || !sv) {
        P2_LOG_ERR("malloc failed");
        free(A); free(B); free(col); free(sv);
        munmap(O, o_bytes);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    if (!no_mlock) {
        p2_mlock_soft(A, M * K * sizeof(double));
        p2_mlock_soft(B, N * K * sizeof(double));
        p2_mlock_soft(col, nnz * sizeof(uint32_t));
        p2_mlock_soft(sv, nnz * sizeof(double));
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Seed the dense features once and build an initial mask. Every measured
     * pass re-seeds all of these, but doing it here warms the pages and lets the
     * first measured pass start from a populated state. */
    for (size_t i = 0; i < M * K; i++) A[i] = rng_unit(&rng);
    for (size_t i = 0; i < N * K; i++) B[i] = rng_unit(&rng);
    for (size_t t = 0; t < nnz; t++) {
        col[t] = (uint32_t)(p2_rng_next(&rng) % (uint64_t)N);       /* random sampled column */
        sv[t]  = rng_unit(&rng);                                    /* random mask scalar    */
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed the dense features and the mask each pass so the sampled output
         * is not trivially constant and cannot be hoisted. These are reads during
         * the dot product; the writes here are the input churn, kept small
         * relative to O so the visible signature stays the sampled output. */
        for (size_t i = 0; i < M * K; i++) A[i] = rng_unit(&rng);   /* re-seed A */
        for (size_t i = 0; i < N * K; i++) B[i] = rng_unit(&rng);   /* re-seed B */
        for (size_t t = 0; t < nnz; t++) {
            col[t] = (uint32_t)(p2_rng_next(&rng) % (uint64_t)N);   /* re-sample columns */
            sv[t]  = rng_unit(&rng);                                /* re-seed s_val     */
        }
        /* SDDMM proper: for each row i, dot its dense A row against the gathered
         * dense B row at every sampled column, scale by s_val, and STORE the one
         * sampled value into O. A[i,:] is streamed; B[j,:] is a gather over the
         * random sampled column j -- both READS. The ONLY write is O[t] per
         * sampled position: the scattered sparse output. */
        for (size_t i = 0; i < M; i++) {
            const double *arow = A + i * K;                         /* dense row A[i,:] */
            size_t base = i * KPR;                                  /* row i owns [base, base+KPR) */
            for (size_t local = 0; local < KPR; local++) {
                size_t t = base + local;
                const double *brow = B + (size_t)col[t] * K;        /* gathered dense row B[j,:] */
                double dot = 0.0;
                for (size_t k = 0; k < K; k++) dot += arow[k] * brow[k];   /* full dense dot */
                O[t] = sv[t] * dot;                                 /* the lone sampled write */
            }
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = O[nnz / 2];                             /* a live sampled value */

    free(A); free(B); free(col); free(sv);
    munmap(O, o_bytes);

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
                   "expected signature: sampled sparse output O (scattered writes over mask nnz); "
                   "dense A/B reads invisible; distinct from quiet SpMV and from dense GEMM full rewrite");
    p2_meta_close(&m);
    return 0;
}
