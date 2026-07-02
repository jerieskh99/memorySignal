/* kernel_sparse_cholesky_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Sparse Linear Algebra dwarf (Berkeley motif D2), the DIRECT-SOLVER side:
 * a banded sparse Cholesky factorisation A = L * L^T. A sparse matvec (see
 * kernel_spmv_v2) only READS its structure, so it is a QUIET control. A
 * factorisation is the opposite: it takes a sparse matrix and WRITES a new
 * factor L that has far more nonzeros than the input -- the classic FILL-IN of
 * direct solvers. That growing factor, written in place over the sweep, is the
 * visible memory signature this kernel is built to expose.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs quiet SpMV)
 * ----------------------------------------------------------------------------
 * SpMV streams a fixed, read-only sparse structure and writes back only a tiny
 * output vector -- near-idle on a host write-signal. Cholesky instead takes the
 * banded input (whose off-diagonal band cells are mostly ZERO) and, column by
 * column, fills those cells with nonzero L values. The dominant buffer -- the
 * packed band -- is written across its whole extent, progressively, as the
 * factorisation sweeps j = 0..N-1. Reads are invisible to a write-signal, but
 * this fill-in is a genuine, sweeping WRITE. That is the distinct tell.
 *
 * ============================================================================
 * ALGORITHM (banded Cholesky, A = L * L^T, fill-in inside the band)
 * ============================================================================
 * We use a symmetric positive-definite (SPD) BANDED matrix of dimension N with
 * half-bandwidth b: nonzeros live only where |i - j| <= b. The lower band is
 * stored PACKED in one buffer band[N * (b+1)], where
 *
 *     band[i*(b+1) + (j - (i - b))]  holds  L[i][j],  for  max(0,i-b) <= j <= i.
 *
 * The local column offset is (j - i + b): the diagonal L[i][i] sits at offset b
 * (the last slot of row i's strip); columns below i occupy the lower offsets.
 * For the top rows (i < b) the lowest offsets map to columns < 0 (outside the
 * matrix) and stay unused. This packed band is the dominant, mmap'd buffer, and
 * it is what fills in during factorisation.
 *
 * Each measured pass:
 *   1. Build A SPD + banded: every in-band off-diagonal entry gets a small
 *      random value (many are left at zero -> the sparse holes that later fill),
 *      and the diagonal is made dominant (diag = 2b + random) so A is SPD and
 *      the factorisation is numerically stable. A's band is copied into the band
 *      buffer, which is then factored IN PLACE:
 *
 *        for j = 0 .. N-1:
 *          s = A[j][j] - sum_{k=max(0,j-b)}^{j-1} L[j][k]^2 ;   L[j][j] = sqrt(s);
 *          for i = j+1 .. min(j+b, N-1):
 *            L[i][j] = ( A[i][j] - sum_{k=max(0,i-b)}^{j-1} L[i][k]*L[j][k] ) / L[j][j];
 *
 *      Every (i,j) read here is already resolved (columns k < j are final) or is
 *      the in-place A value about to be overwritten, so one sweep suffices. The
 *      band cells that were zero in A but are inside the band get overwritten
 *      with nonzero L -- that overwrite IS the fill-in.
 *   2. Count the factorisation.
 *
 * The band is symmetric-positive-definite by construction, so sqrt's argument s
 * stays positive; we still guard it defensively.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Sparse Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 * HONEST CAVEAT: this is the BANDED case of sparse Cholesky (common for
 * 1D-PDE / FEM systems, where the band is the exact nonzero envelope). A GENERAL
 * sparse Cholesky needs a symbolic factorisation and an elimination tree to
 * predict and order the fill-in; neither is implemented here. The banded case is
 * the honest, self-contained subset that still exhibits the fill-in write.
 *
 * Phases: warmup (alloc + first build) / measure (build+factor passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_sparse_cholesky_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign banded sparse Cholesky A=LL^T; Sparse-LA kernel)\n"
"  --dim N               Matrix dimension N (default 8192)\n"
"  --bandwidth b         Half-bandwidth b; nonzeros where |i-j|<=b (default 64)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on band-buffer bytes (default 8192)\n"
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

/* Packed-band index of L[i][j] (or A[i][j]) within row i's (b+1)-long strip.
 * The diagonal sits at local offset b; column j<=i occupies offset (j - i + b).
 * Valid only for max(0,i-b) <= j <= i, which every access below respects. */
static inline size_t band_idx(size_t i, size_t j, size_t b) {
    return i * (b + 1) + (j - i + b);
}

/* Build a fresh SPD banded matrix directly into the packed band buffer.
 * In-band off-diagonal cells (j < i) get a small random value OR zero -- the
 * zeros are the sparse holes that fill in during factorisation. The diagonal is
 * made dominant (2b + random) so the matrix is SPD. Cells whose column is < 0
 * (the unused low offsets of the top rows) are zeroed so the sweep reads clean.
 * band[] afterwards holds A's band; the factorisation overwrites it in place. */
static void build_spd_band(double *band, size_t N, size_t b, p2_rng_t *rng) {
    for (size_t i = 0; i < N; i++) {
        size_t jlo = (i >= b) ? (i - b) : 0;
        double *strip = band + i * (b + 1);
        /* Zero the unused low offsets (columns < 0) for the top rows. */
        for (size_t off = 0; off < (b - (i - jlo)); off++) strip[off] = 0.0;
        /* Off-diagonal in-band entries: small random, or a sparse hole (zero). */
        for (size_t j = jlo; j < i; j++) {
            double u = rng_unit(rng);
            /* Leave ~40% of in-band off-diagonals as structural zeros so the
             * fill-in is real: those cells are zero in A but nonzero in L. */
            band[band_idx(i, j, b)] = (u < 0.40) ? 0.0 : (u - 0.5) * 0.1;
        }
        /* Dominant diagonal -> SPD, stable, sqrt argument stays positive. */
        band[band_idx(i, i, b)] = (double)(2 * b) + rng_unit(rng);
    }
}

/* Banded Cholesky A = L * L^T, IN PLACE on the packed band buffer.
 * On entry band[] holds A's band; on exit it holds L's band. One left-looking
 * sweep over columns j: finalise the diagonal, then each sub-diagonal entry in
 * the band below it. Zero cells inside the band get overwritten with nonzero L
 * -- the fill-in. Returns 0 on success, -1 if a non-positive pivot appears. */
static int cholesky_band_inplace(double *band, size_t N, size_t b) {
    for (size_t j = 0; j < N; j++) {
        size_t klo_j = (j >= b) ? (j - b) : 0;      /* first in-band column of row j */
        /* Diagonal: s = A[j][j] - sum_{k} L[j][k]^2 over already-final columns. */
        double s = band[band_idx(j, j, b)];
        for (size_t k = klo_j; k < j; k++) {
            double ljk = band[band_idx(j, k, b)];
            s -= ljk * ljk;
        }
        if (!(s > 0.0)) return -1;                  /* defensive: SPD => s>0 */
        double ljj = sqrt(s);
        band[band_idx(j, j, b)] = ljj;
        /* Sub-diagonal entries L[i][j] for i in (j, min(j+b, N-1)]. */
        size_t ihi = (j + b < N - 1) ? (j + b) : (N - 1);
        for (size_t i = j + 1; i <= ihi; i++) {
            size_t klo_i = (i >= b) ? (i - b) : 0;  /* first in-band column of row i */
            /* Inner sum runs from max(0,i-b) to j-1; since i>=j we have i-b>=j-b,
             * so every k here is in-band for BOTH row i and row j. */
            double sum = band[band_idx(i, j, b)];   /* the in-place A[i][j] */
            for (size_t k = klo_i; k < j; k++)
                sum -= band[band_idx(i, k, b)] * band[band_idx(j, k, b)];
            band[band_idx(i, j, b)] = sum / ljj;     /* fill-in write */
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long dim        = p2_get_i64(argc, argv, "--dim", 8192);
    long long band_b     = p2_get_i64(argc, argv, "--bandwidth", 64);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (dim < 16 || dim > 1000000LL) { P2_LOG_ERR("dim %lld out of range (16..1e6)", dim); return 2; }
    if (band_b < 1 || band_b >= dim) { P2_LOG_ERR("bandwidth %lld out of range (1..dim-1)", band_b); return 2; }
    size_t N = (size_t)dim, b = (size_t)band_b;
    size_t band_cells = N * (b + 1);                 /* packed lower band */
    size_t bytes = band_cells * sizeof(double);      /* the band buffer dominates the footprint */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("band bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Sparse Linear Algebra");
    p2_meta_kv_str(&m, "scheme", "banded sparse Cholesky A=LL^T; factor fills in within the band (progressive write)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "dim", dim);
    p2_meta_kv_i64(&m, "bandwidth", band_b);
    p2_meta_kv_u64(&m, "band_cells", band_cells);
    p2_meta_kv_u64(&m, "band_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The packed band is the dominant buffer -> mmap + mlock it. It is rebuilt
     * (from A) and then filled in by the factorisation every pass, which is the
     * workload's signature write. */
    double *band = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (band == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(band, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(band, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* One build+factor during warmup to touch every page and settle the caches. */
    build_spd_band(band, N, b, &rng);
    (void)cholesky_band_inplace(band, N, b);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t fail_passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Rebuild A's band fresh each pass (so the factor is not trivially
         * constant), then factor in place. The rebuild + fill-in is the full,
         * sweeping write over the dominant band buffer -- the visible signal. */
        build_spd_band(band, N, b, &rng);
        if (cholesky_band_inplace(band, N, b) != 0) fail_passes++;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = band[band_idx(N / 2, N / 2, b)];   /* a factored diagonal */

    munmap(band, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "nonspd_passes", fail_passes);
    p2_meta_kv_f64(&m, "sink", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "BANDED case only (1D-PDE/FEM envelope); general sparse Cholesky needs symbolic factorisation + elimination tree, not implemented; the in-band fill-in is the distinct write vs quiet SpMV");
    p2_meta_close(&m);
    return 0;
}
