/* kernel_multigrid_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  MULTIGRID V-CYCLE:  geometric multigrid solver for the 2D Poisson problem
 * ============================================================================
 *
 *  DWARF   : Structured Grids (D5)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                  (first-division, memory-signature label)
 *  PURPOSE : Probe a multi-scale, time-varying grid write pattern: the active
 *            footprint shrinks and grows as the solver descends and climbs a
 *            hierarchy of grids, so the working set migrates rather than sitting
 *            on one fixed region.
 *
 *  PICTURE (top view):  the V-cycle (footprint size on the vertical axis)
 *
 *      fine  L0  [========================]  smooth --------> smooth
 *                        \                                   /
 *       mid  L1  [==============]  smooth ----------> smooth
 *                        \                           /
 *      coarse L2  [======]  solve directly ---------
 *
 *      Descent (\): smooth, take the residual, RESTRICT it to the next coarser,
 *                   half-size grid. Ascent (/): PROLONGATE the coarse correction
 *                   back up, add it, smooth again. One pass down and up = one
 *                   V-cycle. Footprint is largest at the fine level and roughly
 *                   quarters at each step down.
 *
 *  ALGORITHM (one V-cycle, applied recursively from the finest level):
 *      1. Pre-smooth the current level with a few red-black Gauss-Seidel sweeps
 *         (the same smoother used by kernel_stencil_seidel_v2).
 *      2. Compute the residual r = f - A u on this level.
 *      3. Restrict r to the next coarser grid by full-weighting (a 9-point
 *         weighted average), zero that level's solution, and recurse into it.
 *      4. At the coarsest level, "solve" by smoothing extra hard instead of
 *         recursing further.
 *      5. On the way back up, prolongate the coarse-grid correction with
 *         bilinear interpolation, add it to this level, and post-smooth.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A sequence of dense grid rewrites whose SIZE changes over time: large at
 *      the fine level, then successively smaller as the cycle restricts down,
 *      then growing again on the way up. The write front therefore migrates
 *      across several fixed regions (one per level) within each V-cycle instead
 *      of hammering a single footprint -- multi-scale, quasi-periodic
 *      working-set migration. Honest caveat: each level individually still shows
 *      the strided-checkerboard smoother writes of the Gauss-Seidel kernel; the
 *      distinguishing feature here is the migration ACROSS levels, not the
 *      per-level pattern.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Structured Grids. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (V-cycles) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_multigrid_v2";

#define MG_MAX_LEVELS 24

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign multigrid V-cycle; structured-grid kernel)\n"
"  --grid-n N            Finest grid side (snapped to 2^k+1; default 1025)\n"
"  --nu N                Smoother sweeps per level (default 2)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the initial field (default 42)\n"
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

/* Red-black Gauss-Seidel smoother for the discrete Poisson operator, where the
 * update 4u - neighbours = f rearranges to u = 0.25*(neighbours + f). "Smoothing"
 * is the multigrid term for a few relaxation sweeps: it cheaply damps the
 * high-frequency (short-wavelength) error that the current grid resolves well,
 * leaving the smooth low-frequency error for coarser grids to handle. The
 * red-black colouring is used for the same reason as in the standalone Seidel
 * kernel -- it lets each colour update in place using fresh opposite-colour
 * values, and it is the source of this level's strided checkerboard writes. */
static void mg_smooth(double *u, const double *f, size_t n, int nu) {
    for (int s = 0; s < nu; s++)                      /* nu relaxation sweeps    */
        for (int c = 0; c < 2; c++)                   /* red pass then black pass */
            for (size_t i = 1; i < n - 1; i++) {
                double *row = u + i * n; const double *up = u + (i - 1) * n;
                const double *dn = u + (i + 1) * n; const double *fr = f + i * n;
                size_t j0 = 1 + (((i + 1) + (size_t)c) & 1);   /* first cell of colour c */
                for (size_t j = j0; j < n - 1; j += 2)
                    row[j] = 0.25 * (up[j] + dn[j] + row[j - 1] + row[j + 1] + fr[j]);
            }
}

/* Residual r = f - A u, where A u = 4u - neighbours is the discrete Laplacian.
 * The residual measures how far the current solution is from satisfying the
 * equation; it is precisely the error signal that gets carried down to the
 * coarser grid, so it is written to its own buffer rather than overwriting u. */
static void mg_residual(double *r, const double *u, const double *f, size_t n) {
    for (size_t i = 1; i < n - 1; i++) {
        const double *row = u + i * n, *up = u + (i - 1) * n, *dn = u + (i + 1) * n, *fr = f + i * n;
        double *rr = r + i * n;
        for (size_t j = 1; j < n - 1; j++)
            rr[j] = fr[j] - (4.0 * row[j] - (up[j] + dn[j] + row[j - 1] + row[j + 1]));
    }
}

/* Full-weighting restriction: transfer the fine residual rf(nf) down to the
 * coarse right-hand side fc(nc), with nc = (nf-1)/2 + 1. Each coarse cell is the
 * 9-point weighted average of the fine cell it sits on (weight 4/16) and that
 * cell's neighbours (edges 2/16, corners 1/16). Weighted averaging (rather than
 * plain sampling) is what keeps the coarse problem a faithful low-frequency
 * picture of the fine residual. This is the step that HALVES the footprint. */
static void mg_restrict(const double *rf, size_t nf, double *fc, size_t nc) {
    memset(fc, 0, nc * nc * sizeof(double));          /* clear coarse RHS first */
    for (size_t ic = 1; ic < nc - 1; ic++)
        for (size_t jc = 1; jc < nc - 1; jc++) {
            size_t i = 2 * ic, j = 2 * jc;            /* co-located fine cell */
            fc[ic * nc + jc] = 0.0625 * (4.0 * rf[i * nf + j]
                + 2.0 * (rf[(i - 1) * nf + j] + rf[(i + 1) * nf + j] + rf[i * nf + j - 1] + rf[i * nf + j + 1])
                + (rf[(i - 1) * nf + j - 1] + rf[(i - 1) * nf + j + 1] + rf[(i + 1) * nf + j - 1] + rf[(i + 1) * nf + j + 1]));
        }
}

/* Guarded accumulate into the fine grid: skip anything outside the interior so
 * the fixed boundary is never modified by a prolongation write. */
static inline void addf(double *uf, size_t nf, size_t i, size_t j, double v) {
    if (i >= 1 && i <= nf - 2 && j >= 1 && j <= nf - 2) uf[i * nf + j] += v;
}

/* Bilinear prolongation: interpolate the coarse-grid correction uc(nc) up to the
 * fine grid uf(nf) and ADD it (a correction, not a replacement). Every coarse
 * cell maps directly onto a fine cell; the fine cells that fall between coarse
 * points get the linear blends of the surrounding two or four coarse values.
 * This is the step that roughly QUADRUPLES the footprint back to fine size, and
 * because it writes the full fine interior it dominates the ascent's write traffic. */
static void mg_prolong_add(const double *uc, size_t nc, double *uf, size_t nf) {
    for (size_t ic = 0; ic < nc - 1; ic++)
        for (size_t jc = 0; jc < nc - 1; jc++) {
            /* the four coarse corners of one coarse cell */
            double a = uc[ic * nc + jc], b = uc[ic * nc + jc + 1];
            double c = uc[(ic + 1) * nc + jc], d = uc[(ic + 1) * nc + jc + 1];
            size_t i = 2 * ic, j = 2 * jc;            /* matching fine origin */
            addf(uf, nf, i, j, a);                    /* co-located: copy corner   */
            addf(uf, nf, i, j + 1, 0.5 * (a + b));    /* horizontal edge midpoint  */
            addf(uf, nf, i + 1, j, 0.5 * (a + c));    /* vertical edge midpoint     */
            addf(uf, nf, i + 1, j + 1, 0.25 * (a + b + c + d));  /* cell centre     */
        }
}

/* the grid hierarchy (set up in main) */
static double *U[MG_MAX_LEVELS], *F[MG_MAX_LEVELS], *R[MG_MAX_LEVELS];
static size_t NL[MG_MAX_LEVELS];
static int NLEV, NU;

/* One recursive V-cycle step at grid level l (0 = finest). The recursion depth
 * traces the "V": descend to the coarsest level, then unwind back to the finest.
 * Each level owns fixed U/F/R buffers, so the sequence of calls here is exactly
 * what makes the active write footprint shrink on the way down and grow on the
 * way up -- the working-set migration this benchmark is built to expose. */
static void vcycle(int l) {
    mg_smooth(U[l], F[l], NL[l], NU);                 /* pre-smooth: damp fine error */
    if (l < NLEV - 1) {
        /* Descend: measure what the smoother left behind and hand it downward. */
        mg_residual(R[l], U[l], F[l], NL[l]);         /* r = f - A u on this level   */
        mg_restrict(R[l], NL[l], F[l + 1], NL[l + 1]);/* r becomes the coarse RHS    */
        memset(U[l + 1], 0, NL[l + 1] * NL[l + 1] * sizeof(double)); /* start coarse guess at 0 */
        vcycle(l + 1);                                /* recurse to coarser (footprint shrinks) */
        /* Ascend: lift the coarse correction back and clean up the interpolation. */
        mg_prolong_add(U[l + 1], NL[l + 1], U[l], NL[l]);  /* u += interpolated correction */
        mg_smooth(U[l], F[l], NL[l], NU);             /* post-smooth: remove interp error */
    } else {
        /* Coarsest level: the grid is tiny, so smoothing extra hard stands in
         * for an exact direct solve -- cheap and good enough to seed the ascent. */
        mg_smooth(U[l], F[l], NL[l], NU * 4);         /* coarse "solve" */
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long grid_n     = p2_get_i64(argc, argv, "--grid-n", 1025);
    long long nu_in      = p2_get_i64(argc, argv, "--nu", 2);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (grid_n < 17 || grid_n > 131073) { P2_LOG_ERR("grid-n %lld out of range", grid_n); return 2; }
    if (nu_in < 1 || nu_in > 100) { P2_LOG_ERR("nu %lld out of range (1..100)", nu_in); return 2; }
    NU = (int)nu_in;

    /* snap the finest side to 2^k + 1 so coarsening (n-1)/2+1 stays integral */
    size_t mm = (size_t)grid_n - 1;
    mm--; mm |= mm >> 1; mm |= mm >> 2; mm |= mm >> 4; mm |= mm >> 8;
    mm |= mm >> 16; mm |= mm >> 32; mm++;
    size_t N = mm + 1;

    /* build the level sizes: N, (N-1)/2+1, ... down to <= 5 */
    NLEV = 0; size_t n = N;
    while (NLEV < MG_MAX_LEVELS) { NL[NLEV++] = n; if (n <= 5) break; n = (n - 1) / 2 + 1; }

    size_t doubles = 0;
    for (int l = 0; l < NLEV; l++) doubles += 3 * NL[l] * NL[l];   /* U, F, R per level */
    size_t bytes = doubles * sizeof(double);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("total bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Structured Grids");
    p2_meta_kv_str(&m, "scheme", "geometric multigrid V-cycle (red-black GS smoother)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "grid_n", (long long)N);
    p2_meta_kv_i64(&m, "levels", NLEV);
    p2_meta_kv_i64(&m, "nu", NU);
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

    /* Carve the U (solution), F (right-hand side) and R (residual) grids for
     * every level out of the single arena, packed back to back. One contiguous
     * mmap keeps every level's footprint at a fixed, distinct address range, so
     * the write front visibly hops between those ranges as the V-cycle descends
     * and ascends -- which is the migration signal we are trying to produce. */
    size_t off = 0;
    for (int l = 0; l < NLEV; l++) {
        size_t c = NL[l] * NL[l];                     /* cells at this level */
        U[l] = arena + off; off += c;
        F[l] = arena + off; off += c;
        R[l] = arena + off; off += c;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    const double BOUND = 1.0;
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* arena is zero-filled by mmap, so F[*]=0 and R[*]=0 already. Fine grid u:
     * boundary fixed, interior random. */
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            int edge = (i == 0 || j == 0 || i == N - 1 || j == N - 1);
            U[0][i * N + j] = edge ? BOUND : rng_unit(&rng);
        }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t cycles = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed the fine interior each cycle so the solver always has error to
         * relax (otherwise it converges and the writes stop changing bytes). */
        for (size_t i = 1; i < N - 1; i++)
            for (size_t j = 1; j < N - 1; j++)
                U[0][i * N + j] = rng_unit(&rng);
        vcycle(0);
        cycles++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = U[0][(N / 2) * N + (N / 2)];

    munmap(arena, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "vcycles", cycles);
    p2_meta_kv_f64(&m, "center_value", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: multi-scale, time-varying footprint (working-set migration across levels)");
    p2_meta_close(&m);
    return 0;
}
