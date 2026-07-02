/* kernel_kalman_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  ENSEMBLE KALMAN:  E parallel linear-Gaussian filters, each updating a small
 *  dense covariance matrix P (d x d) every timestep.
 * ============================================================================
 *
 *  DWARF   : Graphical Models (D12)      (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the host write-signal of a Gaussian graphical model whose
 *            live state is NOT one big array but a POPULATION of many small dense
 *            matrices. A Kalman filter is exact inference on a linear-Gaussian
 *            chain; its belief is a Gaussian (x, P). Running E of them in lockstep
 *            means E covariance matrices, each d x d, are rewritten in full on
 *            every step -- a dense, small-block, scattered write pattern distinct
 *            from a single monotone wavefront (HMM/DP) or one large GEMM tile.
 *
 *  PICTURE (the mmap'd covariance pool, one row per filter):
 *      Each filter e owns a d x d covariance matrix P_e. The E matrices are laid
 *      out contiguously (filter-major) in one flat buffer; every timestep touches
 *      ALL of them, so the write front is not a moving band but the whole pool,
 *      refreshed block by small block.
 *
 *        filter 0 : [ P00 P01 ... P0d ]   <- d x d dense, rewritten each step
 *        filter 1 : [ P00 P01 ... P0d ]   <- d x d dense, rewritten each step
 *        filter 2 : [ P00 P01 ... P0d ]
 *          ...            ...
 *      filter E-1 : [ P00 P01 ... P0d ]
 *                     \___ every entry updated: P = (I-KH) (F P F^T + Q) ___/
 *
 *      The state vectors x_e (d each) form a second, much smaller pool. The fixed
 *      system matrices F, Q, H, R are shared read-only across the ensemble and so
 *      are invisible to a write-only host signal -- only the P (and x) pools show.
 *
 *  ALGORITHM (per pass = `steps` timesteps applied to every filter):
 *      1. Re-randomise the synthetic measurement z (fresh evidence each pass; the
 *         shared model F, Q, H, R is reused, the belief pool evolves).
 *      2. For each timestep, for each filter e, run one Kalman recursion on
 *         (x_e, P_e) with the small dense linear-algebra below:
 *           PREDICT:  x = F x
 *                     P = F P F^T + Q
 *           UPDATE:   S  = H P H^T + R           (m x m innovation covariance)
 *                     K  = P H^T S^{-1}          (d x m Kalman gain; S^{-1} by
 *                                                 Gauss-Jordan, small m)
 *                     x  = x + K (z - H x)       (correct the state)
 *                     P  = (I - K H) P           (shrink the covariance)
 *      3. Symmetrise P  ->  P = (P + P^T) / 2 after the update. Round-off makes
 *         the Joseph-free form (I-KH)P drift slightly asymmetric; forcing exact
 *         symmetry each step is the standard cheap numerical safeguard and keeps
 *         P a valid covariance (symmetric, and in exact arithmetic PSD).
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A fixed-size pool of E small dense d x d matrices, EVERY ONE fully
 *      overwritten each timestep, plus a small x-vector pool. The access is dense
 *      within each d x d block and strided across blocks -- many tiny hot regions
 *      rather than one contiguous sweep. HONEST CAVEAT: the scratch matrices used
 *      inside one filter's update (F P, P F^T, temporaries) live on the stack /
 *      in a small reused work buffer and are cache-resident, so a large fraction
 *      of the actual store traffic never reaches DRAM; what the host observes is
 *      dominated by the write-back of the P pool itself, whose size (E*d*d*8) is
 *      chosen to exceed cache. The read-only F/Q/H/R and the measurement z are not
 *      part of the write signal.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Graphical Models. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (Kalman passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_kalman_v2";

/* Hard cap on the state / measurement dimensions. The per-filter work buffers are
 * a few d x d and d x m scratch matrices allocated once and reused, so these
 * bounds keep that scratch small and the m x m inverse well-conditioned. */
#define KAL_MAX_DIM  64

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign ensemble Kalman filter; graphical-models kernel)\n"
"  --ensemble E          Number of parallel filters (default 8192; P pool is E*d*d*8 bytes)\n"
"  --dim d               State dimension per filter (default 8; each P is d*d)\n"
"  --meas m              Measurement dimension (default 4; must be 1..d)\n"
"  --steps N             Kalman timesteps per pass (default 32)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on covariance-pool bytes (default 8192)\n"
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
 * Small dense linear-algebra helpers. All matrices are row-major flat arrays;
 * dimensions are passed explicitly. These are the primitives every filter's
 * Kalman recursion is built from. Kept local and tiny (d, m <= KAL_MAX_DIM) so
 * the inner update is branch-light and cache-resident.
 * --------------------------------------------------------------------------- */

/* C[ra x cb] = A[ra x ca] * B[ca x cb]  (ca is the shared/contracted dim). */
static void mat_mul(const double *A, const double *B, double *C,
                    size_t ra, size_t ca, size_t cb) {
    for (size_t i = 0; i < ra; i++)
        for (size_t j = 0; j < cb; j++) {
            double acc = 0.0;
            for (size_t k = 0; k < ca; k++) acc += A[i * ca + k] * B[k * cb + j];
            C[i * cb + j] = acc;
        }
}

/* C[ra x cb] = A[ra x ca] * B^T, where B is stored as [cb x ca] (so B^T is
 * ca x cb). Avoids materialising the transpose: contracts A's columns against
 * B's columns. Used for P F^T, P H^T, and (H P) H^T. */
static void mat_mul_bt(const double *A, const double *B, double *C,
                       size_t ra, size_t ca, size_t cb) {
    for (size_t i = 0; i < ra; i++)
        for (size_t j = 0; j < cb; j++) {
            double acc = 0.0;
            for (size_t k = 0; k < ca; k++) acc += A[i * ca + k] * B[j * ca + k];
            C[i * cb + j] = acc;
        }
}

/* out[n x n] = M[n x n]^{-1} via Gauss-Jordan elimination with partial pivoting.
 * n is the small measurement dimension m, so the O(n^3) cost is negligible next
 * to the ensemble sweep. Returns 0 on success, -1 if M is singular (a zero pivot
 * even after pivoting); the caller then skips the correction for that step.
 * work must hold 2*n*n doubles (the [M | I] augmented tableau). */
static int mat_inv(const double *M, double *out, size_t n, double *work) {
    double *aug = work;                 /* n rows, 2n cols: [ M | I ] */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) aug[i * 2 * n + j] = M[i * n + j];
        for (size_t j = 0; j < n; j++) aug[i * 2 * n + (n + j)] = (i == j) ? 1.0 : 0.0;
    }
    for (size_t c = 0; c < n; c++) {
        /* partial pivot: largest-magnitude entry in column c at/below the diagonal */
        size_t piv = c; double best = fabs(aug[c * 2 * n + c]);
        for (size_t r = c + 1; r < n; r++) {
            double v = fabs(aug[r * 2 * n + c]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) return -1;   /* singular: no usable pivot */
        if (piv != c)                    /* swap rows piv and c */
            for (size_t j = 0; j < 2 * n; j++) {
                double t = aug[c * 2 * n + j];
                aug[c * 2 * n + j] = aug[piv * 2 * n + j];
                aug[piv * 2 * n + j] = t;
            }
        double inv_p = 1.0 / aug[c * 2 * n + c];
        for (size_t j = 0; j < 2 * n; j++) aug[c * 2 * n + j] *= inv_p;   /* normalise pivot row */
        for (size_t r = 0; r < n; r++) {                                  /* eliminate column c elsewhere */
            if (r == c) continue;
            double f = aug[r * 2 * n + c];
            if (f == 0.0) continue;
            for (size_t j = 0; j < 2 * n; j++) aug[r * 2 * n + j] -= f * aug[c * 2 * n + j];
        }
    }
    for (size_t i = 0; i < n; i++)                                        /* right half is now M^{-1} */
        for (size_t j = 0; j < n; j++) out[i * n + j] = aug[i * 2 * n + (n + j)];
    return 0;
}

/* Force exact symmetry in-place: P = (P + P^T) / 2. Undoes the tiny asymmetry
 * that round-off introduces in the (I - K H) P covariance update and keeps P a
 * legitimate covariance matrix (a Cholesky / PD check then succeeds). */
static void symmetrise(double *P, size_t d) {
    for (size_t i = 0; i < d; i++)
        for (size_t j = i + 1; j < d; j++) {
            double a = 0.5 * (P[i * d + j] + P[j * d + i]);
            P[i * d + j] = a; P[j * d + i] = a;
        }
}

/* ---------------------------------------------------------------------------
 * Per-filter scratch: work matrices reused across every filter and timestep so
 * the update allocates nothing in the hot loop. Sized to the largest operand.
 * These are cache-resident (see MEMORY SIGNATURE caveat) -- the P/x pools, not
 * this scratch, are what the host write-signal sees.
 * --------------------------------------------------------------------------- */
typedef struct {
    double *t_dd;    /* d x d temporary (F P, and (I-KH))           */
    double *t_dd2;   /* d x d temporary (P F^T result, P update)    */
    double *PHt;     /* d x m  (P H^T)                              */
    double *S;       /* m x m  (innovation covariance H P H^T + R)  */
    double *Sinv;    /* m x m  (S^{-1})                             */
    double *K;       /* d x m  (Kalman gain P H^T S^{-1})           */
    double *KH;      /* d x d  (K H)                                */
    double *inv_wk;  /* 2m x m (Gauss-Jordan augmented tableau)     */
    double *v_m;     /* m      (innovation z - H x)                 */
    double *v_d;     /* d      (F x, and x update accumulator)      */
} KalScratch;

/* One full Kalman recursion on filter (x, P) with shared model F,Q,H,R.
 *   predict:  x <- F x ;  P <- F P F^T + Q
 *   update:   S <- H P H^T + R ; K <- P H^T S^{-1} ;
 *             x <- x + K (z - H x) ; P <- (I - K H) P ; symmetrise(P)
 * If S is singular (mat_inv fails) the correction is skipped: the predicted
 * (x, P) is kept for this step, which stays numerically valid. */
static void kalman_step(double *x, double *P,
                        const double *F, const double *Q,
                        const double *H, const double *R,
                        const double *z, size_t d, size_t m, KalScratch *w) {
    /* ---- PREDICT ---- */
    /* x = F x  (into v_d, then copy back) */
    for (size_t i = 0; i < d; i++) {
        double acc = 0.0;
        for (size_t k = 0; k < d; k++) acc += F[i * d + k] * x[k];
        w->v_d[i] = acc;
    }
    for (size_t i = 0; i < d; i++) x[i] = w->v_d[i];
    /* P = F P F^T + Q :  t_dd = F P ; t_dd2 = t_dd * F^T ; P = t_dd2 + Q */
    mat_mul(F, P, w->t_dd, d, d, d);            /* F P            */
    mat_mul_bt(w->t_dd, F, w->t_dd2, d, d, d);  /* (F P) F^T      */
    for (size_t i = 0; i < d * d; i++) P[i] = w->t_dd2[i] + Q[i];

    /* ---- UPDATE ---- */
    /* PHt = P H^T   (d x m) ;  S = H PHt + R   (m x m) */
    mat_mul_bt(P, H, w->PHt, d, d, m);          /* P H^T (H is m x d -> H^T is d x m) */
    mat_mul(H, w->PHt, w->S, m, d, m);          /* H (P H^T)      */
    for (size_t i = 0; i < m * m; i++) w->S[i] += R[i];
    /* Sinv = S^{-1}; if singular, skip the correction (keep predicted x, P). */
    if (mat_inv(w->S, w->Sinv, m, w->inv_wk) != 0) { symmetrise(P, d); return; }
    /* K = PHt * Sinv   (d x m) */
    mat_mul(w->PHt, w->Sinv, w->K, d, m, m);
    /* innovation v_m = z - H x   (m) */
    for (size_t i = 0; i < m; i++) {
        double hx = 0.0;
        for (size_t k = 0; k < d; k++) hx += H[i * d + k] * x[k];
        w->v_m[i] = z[i] - hx;
    }
    /* x = x + K v_m */
    for (size_t i = 0; i < d; i++) {
        double acc = 0.0;
        for (size_t k = 0; k < m; k++) acc += w->K[i * m + k] * w->v_m[k];
        x[i] += acc;
    }
    /* P = (I - K H) P :  KH = K H ; t_dd = I - KH ; t_dd2 = t_dd * P ; P = t_dd2 */
    mat_mul(w->K, H, w->KH, d, m, d);           /* K H (d x d)    */
    for (size_t i = 0; i < d; i++)
        for (size_t j = 0; j < d; j++)
            w->t_dd[i * d + j] = (i == j ? 1.0 : 0.0) - w->KH[i * d + j];   /* I - K H */
    mat_mul(w->t_dd, P, w->t_dd2, d, d, d);     /* (I - K H) P    */
    for (size_t i = 0; i < d * d; i++) P[i] = w->t_dd2[i];
    symmetrise(P, d);                           /* enforce covariance symmetry */
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long ensemble   = p2_get_i64(argc, argv, "--ensemble", 8192);
    long long dim        = p2_get_i64(argc, argv, "--dim", 8);
    long long meas       = p2_get_i64(argc, argv, "--meas", 4);
    long long steps      = p2_get_i64(argc, argv, "--steps", 32);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (dim < 1 || dim > KAL_MAX_DIM) { P2_LOG_ERR("dim %lld out of range (1..%d)", dim, KAL_MAX_DIM); return 2; }
    if (meas < 1 || meas > dim) { P2_LOG_ERR("meas %lld out of range (1..dim=%lld)", meas, dim); return 2; }
    if (ensemble < 1 || ensemble > (1LL << 26)) { P2_LOG_ERR("ensemble %lld out of range (1..2^26)", ensemble); return 2; }
    if (steps < 1 || steps > (1LL << 20)) { P2_LOG_ERR("steps %lld out of range (1..2^20)", steps); return 2; }
    size_t E = (size_t)ensemble, d = (size_t)dim, m = (size_t)meas;
    size_t p_cells = E * d * d;                  /* covariance-pool cells */
    size_t bytes = p_cells * sizeof(double);     /* the P pool dominates the footprint */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("covariance-pool bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t meta;
    p2_meta_open(&meta, outdir, TEST);
    p2_meta_kv_str(&meta, "test_name", TEST);
    p2_meta_kv_str(&meta, "language", "C");
    p2_meta_kv_str(&meta, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&meta, "behavior_family", "KERNEL");
    p2_meta_kv_str(&meta, "dwarf", "Graphical Models");
    p2_meta_kv_str(&meta, "scheme", "ensemble Kalman filter (E parallel d-dim linear-Gaussian filters; per-step dense d x d covariance update)");
    p2_meta_kv_str(&meta, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&meta, "ensemble", ensemble);
    p2_meta_kv_i64(&meta, "dim", dim);
    p2_meta_kv_i64(&meta, "meas", meas);
    p2_meta_kv_i64(&meta, "steps", steps);
    p2_meta_kv_u64(&meta, "total_bytes", bytes);
    p2_meta_kv_i64(&meta, "duration_s", duration_s);
    p2_meta_kv_i64(&meta, "warmup_s", warmup_s);
    p2_meta_kv_u64(&meta, "seed", seed);
    p2_meta_kv_i64(&meta, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&meta, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&meta, "status", "dry_run"); p2_meta_close(&meta); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The covariance pool is the dominant buffer -> mmap + mlock it (every one of
     * its E small d x d matrices is rewritten each step, the signature write). */
    double *Ppool = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (Ppool == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&meta, "status", "mmap_failed"); p2_meta_close(&meta); return 1;
    }
    p2_madvise(Ppool, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(Ppool, bytes);

    /* State-vector pool (E filters x d), the shared model matrices, the synthetic
     * measurement, and the reused per-filter scratch. */
    double *xpool = (double *)malloc(E * d * sizeof(double));
    double *F = (double *)malloc(d * d * sizeof(double));   /* transition        */
    double *Q = (double *)malloc(d * d * sizeof(double));   /* process noise     */
    double *H = (double *)malloc(m * d * sizeof(double));   /* measurement       */
    double *R = (double *)malloc(m * m * sizeof(double));   /* measurement noise */
    double *z = (double *)malloc(m * sizeof(double));       /* synthetic measurement */
    KalScratch w;
    w.t_dd   = (double *)malloc(d * d * sizeof(double));
    w.t_dd2  = (double *)malloc(d * d * sizeof(double));
    w.PHt    = (double *)malloc(d * m * sizeof(double));
    w.S      = (double *)malloc(m * m * sizeof(double));
    w.Sinv   = (double *)malloc(m * m * sizeof(double));
    w.K      = (double *)malloc(d * m * sizeof(double));
    w.KH     = (double *)malloc(d * d * sizeof(double));
    w.inv_wk = (double *)malloc(2 * m * m * sizeof(double));
    w.v_m    = (double *)malloc(m * sizeof(double));
    w.v_d    = (double *)malloc(d * sizeof(double));
    if (!xpool || !F || !Q || !H || !R || !z ||
        !w.t_dd || !w.t_dd2 || !w.PHt || !w.S || !w.Sinv || !w.K || !w.KH ||
        !w.inv_wk || !w.v_m || !w.v_d) {
        free(xpool); free(F); free(Q); free(H); free(R); free(z);
        free(w.t_dd); free(w.t_dd2); free(w.PHt); free(w.S); free(w.Sinv);
        free(w.K); free(w.KH); free(w.inv_wk); free(w.v_m); free(w.v_d);
        munmap(Ppool, bytes); P2_LOG_ERR("malloc failed");
        p2_meta_kv_str(&meta, "status", "alloc_failed"); p2_meta_close(&meta); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);

    /* Fixed shared system matrices (read-only across the whole ensemble):
     *   F : near-identity transition with a little off-diagonal coupling, so the
     *       state genuinely mixes without blowing up.
     *   Q : small diagonal process noise (keeps P from collapsing to zero).
     *   H : identity-ish measurement, the first m states observed (m <= d).
     *   R : diagonal measurement noise (guarantees S = H P H^T + R is invertible).
     */
    for (size_t i = 0; i < d; i++)
        for (size_t j = 0; j < d; j++)
            F[i * d + j] = (i == j) ? 1.0 : ((j == i + 1) ? 0.02 : 0.0);   /* near-identity + coupling */
    for (size_t i = 0; i < d; i++)
        for (size_t j = 0; j < d; j++)
            Q[i * d + j] = (i == j) ? 1e-3 : 0.0;                          /* small diagonal */
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < d; j++)
            H[i * d + j] = (i == j) ? 1.0 : 0.0;                           /* observe first m states */
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < m; j++)
            R[i * m + j] = (i == j) ? 1e-1 : 0.0;                          /* diagonal, keeps S PD */

    /* Initialise every filter: state near zero, covariance = identity (a valid,
     * symmetric, positive-definite prior). */
    for (size_t e = 0; e < E; e++) {
        double *xe = xpool + e * d;
        double *Pe = Ppool + e * d * d;
        for (size_t i = 0; i < d; i++) xe[i] = 0.05 * (2.0 * rng_unit(&rng) - 1.0);
        for (size_t i = 0; i < d; i++)
            for (size_t j = 0; j < d; j++)
                Pe[i * d + j] = (i == j) ? 1.0 : 0.0;                      /* P0 = I */
    }
    for (size_t i = 0; i < m; i++) z[i] = 2.0 * rng_unit(&rng) - 1.0;      /* initial measurement */
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        for (size_t i = 0; i < m; i++) z[i] = 2.0 * rng_unit(&rng) - 1.0;  /* fresh evidence this pass */
        /* Apply `steps` Kalman timesteps to every filter. Each timestep fully
         * rewrites all E covariance matrices in the pool -- the distinctive,
         * scattered small-block write the host signal is meant to expose. */
        for (long long t = 0; t < steps; t++) {
            for (size_t e = 0; e < E; e++) {
                double *xe = xpool + e * d;
                double *Pe = Ppool + e * d * d;
                kalman_step(xe, Pe, F, Q, H, R, z, d, m, &w);
            }
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = Ppool[(E - 1) * d * d];   /* a live covariance entry P_{E-1}[0][0] */

    free(xpool); free(F); free(Q); free(H); free(R); free(z);
    free(w.t_dd); free(w.t_dd2); free(w.PHt); free(w.S); free(w.Sinv);
    free(w.K); free(w.KH); free(w.inv_wk); free(w.v_m); free(w.v_d);
    munmap(Ppool, bytes);

    p2_meta_kv_f64(&meta, "warmup_t0_s", t0);
    p2_meta_kv_f64(&meta, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&meta, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&meta, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&meta, "passes", passes);
    p2_meta_kv_f64(&meta, "last_cov00", (double)sink);
    p2_meta_kv_str(&meta, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&meta, "end_time", tend);
    p2_meta_kv_str(&meta, "known_limitations",
                   "distinctive write is the pool of E small dense covariance matrices, all rewritten each step; "
                   "per-filter update scratch (F P, P F^T, temporaries) is cache-resident and mostly invisible to the host signal; "
                   "shared F/Q/H/R and measurement z are read-only");
    p2_meta_close(&meta);
    return 0;
}
