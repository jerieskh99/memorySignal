/* kernel_spgemm_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Sparse Linear Algebra dwarf (Berkeley motif D2), the SpGEMM variant: sparse
 * matrix times sparse matrix producing a NEW sparse matrix,  C = A * B, where A
 * is M x K, B is K x N, and C is M x N, all stored in CSR (compressed sparse
 * row) form. This is the computational core of algebraic-multigrid setup (the
 * Galerkin triple product A_coarse = P^T A P is two SpGEMMs) and of triangle
 * counting on graphs (A * A over the adjacency matrix), so it is a workload that
 * matters in practice, not a toy.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs kernel_spmv_v2, the QUIET control)
 * ----------------------------------------------------------------------------
 * SpMV (y = A*x) is the quiet sibling in this same dwarf: its big structures are
 * all READ and it writes back only a tiny output vector, so a host write-signal
 * sees almost nothing. SpGEMM is the opposite. It MATERIALISES A WHOLE NEW
 * SPARSE MATRIX: for every row of A it fans out through B, accumulates the
 * products, and then APPENDS the surviving entries into C's column-index and
 * value arrays. Those two output arrays -- the freshly created matrix, with its
 * fill-in (C typically has many more nonzeros than A or B) -- are the dominant,
 * VISIBLE write. Same dwarf, opposite tell: SpMV is read-bound and near-idle,
 * SpGEMM streams out a brand-new structure that a write-signal cannot miss.
 *
 * ============================================================================
 * ALGORITHM (Gustavson's row-by-row method with a dense accumulator)
 * ============================================================================
 * SpGEMM is C[i,j] = sum over k of A[i,k] * B[k,j]. Gustavson computes it one
 * row of C at a time, which is cache-friendly and needs no search structure:
 *
 *   For each row i of A:
 *     1. Start with an empty dense scratch accumulator of length N (a double
 *        array `acc` plus a `touched` marker list so we know which columns are
 *        live without scanning all N).
 *     2. For each nonzero (kc = A.col[t], a = A.val[t]) in row i of A:
 *          for each nonzero (jc = B.col[u], bv = B.val[u]) in row kc of B:
 *              if column jc is not yet live, mark it live and remember jc;
 *              acc[jc] += a * bv.                     (the scatter-accumulate)
 *     3. GATHER the live columns out of acc into C: append each (jc, acc[jc])
 *        to C.col / C.val and advance C.row_ptr[i+1]. This append into C's
 *        output arrays is the workload's signature write.
 *     4. Clear the live markers (and the acc slots we touched) so the next row
 *        starts clean -- an O(nnz_of_row) reset, not an O(N) wipe.
 *
 * The dense accumulator makes each A*B fan-out a scatter into `acc` at random
 * columns jc; the gather then streams the survivors into C sequentially.
 *
 * OUTPUT CAP AND OVERFLOW
 * ----------------------------------------------------------------------------
 * The exact size of C is not known before computing it (worst case nnz(C) can
 * reach M*N). We therefore mmap a fixed output pool sized by a cap and, if a row
 * would push C past the cap, we CLIP that row (write what fits, drop the rest)
 * and set an overflow flag recorded in the metadata. The cap is chosen from the
 * inputs -- cap = M * min(N, nnzA_per_row * nnzB_per_row), i.e. the per-row
 * fan-out bound M*(K_a*K_b) but never more than the fully-dense M*N -- and is
 * still bounded by --max-mb so a pathological request cannot exhaust memory.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Sparse Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + seed A,B) / measure (SpGEMM passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_spgemm_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign sparse x sparse -> new sparse matrix; Sparse-LA kernel)\n"
"  --rows M              Rows of A / rows of C   (default 2048)\n"
"  --inner K             Cols of A / rows of B   (default 2048)\n"
"  --cols N              Cols of B / cols of C   (default 2048)\n"
"  --nnz-per-row K       Nonzeros per row in A and B (default 8)\n"
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

/* ---------------------------------------------------------------------------
 * A fixed-nnz-per-row CSR matrix used for the two INPUTS A and B.
 *   rows, cols   : logical dimensions.
 *   k            : nonzeros per row (every row has exactly k, so the row_ptr is
 *                  implicit -- row i owns the slice [i*k, i*k + k) of col/val).
 *   col, val     : the two parallel arrays of length rows*k.
 * Inputs are cheap to (re)seed and are READ during the product; only C is the
 * visible written structure, so A and B live in ordinary malloc'd memory.
 * --------------------------------------------------------------------------- */
typedef struct {
    size_t    rows, cols, k;
    uint32_t *col;
    double   *val;
} CSRFixed;

/* Fill a fixed-nnz CSR matrix with random columns in [0, cols) and random
 * coefficients. The random column spread is what makes the later B-row fan-out
 * scatter across the dense accumulator (and gives C its irregular fill-in). */
static void csr_reseed(CSRFixed *A, p2_rng_t *rng) {
    size_t nnz = A->rows * A->k;
    for (size_t t = 0; t < nnz; t++) {
        A->col[t] = (uint32_t)(p2_rng_next(rng) % (uint64_t)A->cols);
        A->val[t] = rng_unit(rng);
    }
}

/* ---------------------------------------------------------------------------
 * Gustavson SpGEMM: C = A * B, one row of C at a time.
 *   A          : M x K input (fixed-nnz CSR).
 *   B          : K x N input (fixed-nnz CSR).
 *   acc        : dense length-N scratch accumulator (caller-owned, all zero on
 *                entry and left all zero on exit -- we clear only what we touch).
 *   live       : per-column "is this column already in the touched list" flag
 *                (length N, all zero on entry, restored to zero on exit).
 *   c_col,c_val: C's output arrays (the mmap'd, visible structure); we append.
 *   c_row_ptr  : C's row pointers, length M+1; c_row_ptr[0] must be 0.
 *   cap        : capacity of c_col / c_val in entries.
 *   overflow   : set to 1 if any row was clipped because C hit the cap.
 * Returns nnz(C) actually written.
 * --------------------------------------------------------------------------- */
static size_t spgemm(const CSRFixed *A, const CSRFixed *B,
                     double *acc, uint32_t *live,
                     uint32_t *c_col, double *c_val, size_t *c_row_ptr,
                     size_t cap, int *overflow) {
    size_t M = A->rows;
    size_t ka = A->k, kb = B->k;
    size_t nnzC = 0;
    /* `touched` collects the live column indices of the CURRENT row so we can
     * gather them and then clear them in O(row) rather than scanning all N. It
     * can hold at most ka*kb entries (the per-row fan-out), so size it to that. */
    size_t touch_cap = ka * kb;
    uint32_t *touched = (uint32_t *)malloc((touch_cap ? touch_cap : 1) * sizeof(uint32_t));
    if (!touched) { *overflow = 1; c_row_ptr[0] = 0; return 0; }

    c_row_ptr[0] = 0;
    for (size_t i = 0; i < M; i++) {
        size_t ntouch = 0;
        size_t abase = i * ka;                          /* row i of A: [abase, abase+ka) */
        /* Fan A's row i through B and scatter-accumulate into the dense acc. */
        for (size_t t = 0; t < ka; t++) {
            uint32_t kc = A->col[abase + t];            /* A[i,kc] hits B's row kc */
            double   a  = A->val[abase + t];
            size_t   bbase = (size_t)kc * kb;           /* row kc of B: [bbase, bbase+kb) */
            for (size_t u = 0; u < kb; u++) {
                uint32_t jc = B->col[bbase + u];        /* contributes to C[i,jc] */
                double   bv = B->val[bbase + u];
                if (!live[jc]) {                        /* first hit on column jc */
                    live[jc] = 1;
                    touched[ntouch++] = jc;             /* remember it for gather/clear */
                    acc[jc] = a * bv;
                } else {
                    acc[jc] += a * bv;                  /* scatter-accumulate */
                }
            }
        }
        /* GATHER the live columns of this row into C. This append into C's
         * output arrays is the dominant, visible write of the whole workload.
         * If C would exceed the cap, clip the remainder and flag overflow. */
        for (size_t s = 0; s < ntouch; s++) {
            uint32_t jc = touched[s];
            if (nnzC < cap) {
                c_col[nnzC] = jc;                       /* the newly-created matrix... */
                c_val[nnzC] = acc[jc];                  /* ...streamed out to memory */
                nnzC++;
            } else {
                *overflow = 1;                          /* out of room: drop the rest */
            }
            live[jc] = 0;                               /* clear markers for next row */
            acc[jc]  = 0.0;                             /* restore acc slot to zero */
        }
        c_row_ptr[i + 1] = nnzC;                        /* advance C's row pointer */
    }

    free(touched);
    return nnzC;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long rows       = p2_get_i64(argc, argv, "--rows", 2048);
    long long inner      = p2_get_i64(argc, argv, "--inner", 2048);
    long long cols       = p2_get_i64(argc, argv, "--cols", 2048);
    long long nnz_pr     = p2_get_i64(argc, argv, "--nnz-per-row", 8);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (rows  < 1 || rows  > 200000000LL) { P2_LOG_ERR("rows %lld out of range", rows); return 2; }
    if (inner < 1 || inner > 200000000LL) { P2_LOG_ERR("inner %lld out of range", inner); return 2; }
    if (cols  < 1 || cols  > 200000000LL) { P2_LOG_ERR("cols %lld out of range", cols); return 2; }
    if (nnz_pr < 1 || nnz_pr > 1024) { P2_LOG_ERR("nnz-per-row %lld out of range (1..1024)", nnz_pr); return 2; }

    size_t M = (size_t)rows, K = (size_t)inner, N = (size_t)cols, Kp = (size_t)nnz_pr;
    size_t nnzA = M * Kp;                                /* A is M x K, Kp per row */
    size_t nnzB = K * Kp;                                /* B is K x N, Kp per row */

    /* Cap for C: per-row fan-out is at most Kp*Kp distinct columns, but a row can
     * never have more than N columns, so bound the per-row nnz by min(N, Kp*Kp)
     * and multiply by M rows. This is generous for typical fill-in yet finite. */
    size_t per_row_cap = Kp * Kp;
    if (per_row_cap > N) per_row_cap = N;
    size_t c_cap = M * per_row_cap;
    if (c_cap == 0) c_cap = 1;

    /* Byte budget: inputs A,B (col+val) + dense accumulator acc + live markers +
     * C's output pool (col+val) + C's row_ptr. The C pool is the dominant term
     * and the one we mmap; everything is counted against --max-mb. */
    size_t bytes_inputs = (nnzA + nnzB) * (sizeof(uint32_t) + sizeof(double));
    size_t bytes_scratch = N * sizeof(double) + N * sizeof(uint32_t);
    size_t bytes_cpool  = c_cap * (sizeof(uint32_t) + sizeof(double));
    size_t bytes_crow   = (M + 1) * sizeof(size_t);
    size_t bytes = bytes_inputs + bytes_scratch + bytes_cpool + bytes_crow;
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
    p2_meta_kv_str(&m, "scheme", "SpGEMM C=A*B (Gustavson dense-accumulator); writes a whole new sparse matrix with fill-in");
    p2_meta_kv_str(&m, "role", "visible write: materialises a new sparse matrix, distinct from quiet SpMV");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "rows", rows);
    p2_meta_kv_i64(&m, "inner", inner);
    p2_meta_kv_i64(&m, "cols", cols);
    p2_meta_kv_i64(&m, "nnz_per_row", nnz_pr);
    p2_meta_kv_u64(&m, "nnz_a", nnzA);
    p2_meta_kv_u64(&m, "nnz_b", nnzB);
    p2_meta_kv_u64(&m, "c_cap", c_cap);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* Inputs A and B, plus the dense scratch (acc + live) live in malloc'd RAM:
     * they are READ during the product, not the visible output. */
    uint32_t *a_col = (uint32_t *)malloc(nnzA * sizeof(uint32_t));
    double   *a_val = (double *)malloc(nnzA * sizeof(double));
    uint32_t *b_col = (uint32_t *)malloc(nnzB * sizeof(uint32_t));
    double   *b_val = (double *)malloc(nnzB * sizeof(double));
    double   *acc   = (double *)calloc(N, sizeof(double));      /* dense accumulator */
    uint32_t *live  = (uint32_t *)calloc(N, sizeof(uint32_t));  /* touched-column flags */
    size_t   *c_row = (size_t *)malloc((M + 1) * sizeof(size_t));
    if (!a_col || !a_val || !b_col || !b_val || !acc || !live || !c_row) {
        P2_LOG_ERR("malloc failed");
        free(a_col); free(a_val); free(b_col); free(b_val);
        free(acc); free(live); free(c_row);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    /* C's output pool is the dominant, visible structure -> mmap + mlock it (it
     * is rewritten from scratch every pass, which is the workload's signature
     * write). One pool holds both column indices and values back-to-back. */
    size_t cbytes = bytes_cpool;
    void *cpool = mmap(NULL, cbytes, PROT_READ | PROT_WRITE,
                       MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (cpool == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", cbytes, strerror(errno));
        free(a_col); free(a_val); free(b_col); free(b_val);
        free(acc); free(live); free(c_row);
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(cpool, cbytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(cpool, cbytes);
    uint32_t *c_col = (uint32_t *)cpool;                        /* C column indices */
    double   *c_val = (double *)(c_col + c_cap);                /* C values follow  */

    CSRFixed A = { M, K, Kp, a_col, a_val };
    CSRFixed B = { K, N, Kp, b_col, b_val };

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Seed A and B once during warmup so the first measured pass has inputs;
     * each measured pass re-seeds them again below. */
    csr_reseed(&A, &rng);
    csr_reseed(&B, &rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    size_t   last_nnzC = 0;
    int      overflow = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed A and B each pass so C is not trivially constant. These are
         * O(nnz) writes over the (small) input arrays; the big write is C. */
        csr_reseed(&A, &rng);
        csr_reseed(&B, &rng);
        /* The product: materialise a whole new sparse matrix C into the mmap'd
         * pool. The appends into c_col / c_val are the dominant visible write. */
        last_nnzC = spgemm(&A, &B, acc, live, c_col, c_val, c_row, c_cap, &overflow);
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = (last_nnzC > 0) ? c_val[last_nnzC / 2] : 0.0;

    free(a_col); free(a_val); free(b_col); free(b_val);
    free(acc); free(live); free(c_row);
    munmap(cpool, cbytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "nnz_c", last_nnzC);
    p2_meta_kv_i64(&m, "c_overflowed", overflow);
    p2_meta_kv_f64(&m, "sink", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "the newly-created matrix C is the distinct write vs quiet SpMV; C is capped by --max-mb and clipped on overflow");
    p2_meta_close(&m);
    return 0;
}
