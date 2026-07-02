/* kernel_pic_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  PARTICLE-IN-CELL:  charged particles coupled to a field on a grid (2D)
 * ============================================================================
 *
 *  DWARF   : N-Body Methods (Berkeley computational motif D4)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of the one N-body variant that couples the
 *            particles to a GRID. Unlike gravity/MD (particles only), PIC each
 *            step SCATTERS particle charge onto a grid, solves the field on that
 *            grid, then GATHERS the field back. The particle->grid scatter-deposit
 *            (an irregular, accumulating write into a large grid) plus the grid
 *            solve is the distinctive write pattern -- a particle/grid hybrid that
 *            neither the plain nbody nor Barnes-Hut nor cell-list MD produces.
 *
 *  This is the standard electrostatic PIC loop: charge is spread to grid nodes
 *  with Cloud-in-Cell (bilinear) weights, Poisson's equation is relaxed on the
 *  grid to get the potential, the electric field is its gradient, and that field
 *  is interpolated back to push the particles.
 *
 *  PICTURE (top view):
 *      Particles live on a continuous plane; the field lives on a grid. Each step
 *      couples them twice -- SCATTER charge out to the 4 surrounding grid nodes,
 *      and later GATHER the field back with the same bilinear weights.
 *
 *          particle p (px,py)              4 surrounding grid nodes:
 *                *                            n00 ------- n10
 *               /|\   -- scatter q -->         |    p    |     w00..w11
 *                |    <-- gather E --          n01 ------- n11    sum to 1
 *
 *      per step:  DEPOSIT charge (scatter) -> subtract mean (neutral background)
 *                 -> Poisson solve (Jacobi sweeps) -> E = -grad phi
 *                 -> GATHER E to particles -> push particles (+ periodic wrap)
 *
 *  ALGORITHM (per step):
 *      1. Deposit: zero the grid charge rho, then for each particle add its charge
 *         to its 4 nearest grid nodes with bilinear weights (the weights sum to 1,
 *         so charge is conserved exactly). THIS SCATTER IS THE SIGNATURE WRITE.
 *      2. Subtract the mean charge (a uniform neutralising background) so the net
 *         charge is zero -- required for a solvable periodic Poisson problem.
 *      3. Poisson solve: relax phi with K Jacobi sweeps of
 *         phi[i][j] = 0.25*(phi_up+phi_down+phi_left+phi_right + rho[i][j]),
 *         warm-started from the previous step's phi (periodic boundaries).
 *      4. Field: E = -grad phi by central differences on the grid.
 *      5. Gather: interpolate E at each particle with the same bilinear weights.
 *      6. Push: v += q*E*dt ; x += v*dt ; wrap into the periodic box.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A large grid (charge / potential / field arrays) rewritten every step, fed
 *      by an irregular particle->grid scatter, plus the compact particle arrays.
 *      Distinct from the particle-only N-body methods (which touch no grid) and
 *      from the pure stencils (which have no scatter). Honest caveat: the Jacobi
 *      solve is run for a fixed, not-necessarily-converged, number of sweeps --
 *      the goal is the memory-access pattern, not a physically exact field.
 *
 *  Real-world use: plasma physics, particle accelerators, and semiconductor
 *  device simulation.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: N-Body Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init + first deposit/solve) / measure (steps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>
#include <string.h>

static const char *TEST = "kernel_pic_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign particle-in-cell; N-Body kernel)\n"
"  --grid NG             Grid nodes per dimension (default 512; grid is NG x NG)\n"
"  --particles N         Number of particles (default 200000)\n"
"  --solve-iters K       Jacobi sweeps of the Poisson solve per step (default 30)\n"
"  --dt-milli DT         Timestep x1000 (default 50 = 0.05)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on grid bytes (default 8192)\n"
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

/* One Jacobi sweep of the periodic Poisson equation (h = 1 grid spacing):
 * dst[i][j] = 0.25 * (up + down + left + right + rho[i][j]).  Reads src, writes
 * dst; the caller ping-pongs the two buffers. */
static void poisson_jacobi(double *dst, const double *src, const double *rho, int NG) {
    for (int i = 0; i < NG; i++) {
        int ip = (i + 1) % NG, im = (i - 1 + NG) % NG;
        for (int j = 0; j < NG; j++) {
            int jp = (j + 1) % NG, jm = (j - 1 + NG) % NG;
            dst[i * NG + j] = 0.25 * (src[ip * NG + j] + src[im * NG + j] +
                                      src[i * NG + jp] + src[i * NG + jm] +
                                      rho[i * NG + j]);
        }
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long ngll       = p2_get_i64(argc, argv, "--grid", 512);
    long long nparts     = p2_get_i64(argc, argv, "--particles", 200000);
    long long iters      = p2_get_i64(argc, argv, "--solve-iters", 30);
    long long dt_milli   = p2_get_i64(argc, argv, "--dt-milli", 50);   /* float as integer-milli */
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (ngll < 8 || ngll > 8192) { P2_LOG_ERR("grid %lld out of range (8..8192)", ngll); return 2; }
    if (nparts < 64 || nparts > (1LL << 26)) { P2_LOG_ERR("particles %lld out of range (64..2^26)", nparts); return 2; }
    if (iters < 1 || iters > 4096) { P2_LOG_ERR("solve-iters %lld out of range (1..4096)", iters); return 2; }
    int NG = (int)ngll; size_t N = (size_t)nparts; int K = (int)iters;
    double dt = (double)dt_milli / 1000.0;
    double Lg = (double)NG;                          /* box side in grid units (h = 1) */
    size_t G = (size_t)NG * (size_t)NG;              /* grid nodes */
    size_t gbytes = 5 * G * sizeof(double);          /* rho, phi, phi_new, Ex, Ey */
    if (gbytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("grid bytes %zu exceed --max-mb %lld", gbytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "N-Body Methods");
    p2_meta_kv_str(&m, "scheme", "electrostatic particle-in-cell (CIC scatter/gather + Jacobi Poisson solve on a grid)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "grid", ngll);
    p2_meta_kv_i64(&m, "particles", nparts);
    p2_meta_kv_i64(&m, "solve_iters", iters);
    p2_meta_kv_i64(&m, "dt_milli", dt_milli);
    p2_meta_kv_u64(&m, "grid_bytes", gbytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* grid fields: one mmap'd block sliced into five NG*NG arrays */
    double *buf = (double *)mmap(NULL, gbytes, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", gbytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(buf, gbytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(buf, gbytes);
    double *rho = buf, *phi = buf + G, *phi_new = buf + 2 * G, *Ex = buf + 3 * G, *Ey = buf + 4 * G;

    /* particle arrays */
    double *px = (double *)malloc(N * sizeof(double));
    double *py = (double *)malloc(N * sizeof(double));
    double *pvx = (double *)malloc(N * sizeof(double));
    double *pvy = (double *)malloc(N * sizeof(double));
    if (!px || !py || !pvx || !pvy) { free(px); free(py); free(pvx); free(pvy); munmap(buf, gbytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    /* CIC deposit: spread every particle's unit charge to its 4 nearest nodes.
     * This inner routine is the workload's defining scatter write. */
    #define DEPOSIT() do {                                                        \
        for (size_t g_ = 0; g_ < G; g_++) rho[g_] = 0.0;                          \
        for (size_t p_ = 0; p_ < N; p_++) {                                       \
            int gx = (int)px[p_], gy = (int)py[p_];                               \
            double fx = px[p_] - gx, fy = py[p_] - gy;                            \
            int gx1 = (gx + 1) % NG, gy1 = (gy + 1) % NG;                         \
            rho[(size_t)gy  * NG + gx ] += (1.0 - fx) * (1.0 - fy);               \
            rho[(size_t)gy  * NG + gx1] += fx * (1.0 - fy);                       \
            rho[(size_t)gy1 * NG + gx ] += (1.0 - fx) * fy;                       \
            rho[(size_t)gy1 * NG + gx1] += fx * fy;                               \
        }                                                                         \
        double mean_ = 0.0;                                                       \
        for (size_t g_ = 0; g_ < G; g_++) mean_ += rho[g_];                       \
        mean_ /= (double)G;                                                       \
        for (size_t g_ = 0; g_ < G; g_++) rho[g_] -= mean_;   /* neutral bg */    \
    } while (0)

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) {                 /* random particles, small velocities */
        px[i] = rng_unit(&rng) * Lg;
        py[i] = rng_unit(&rng) * Lg;
        pvx[i] = 0.1 * (2.0 * rng_unit(&rng) - 1.0);
        pvy[i] = 0.1 * (2.0 * rng_unit(&rng) - 1.0);
    }
    for (size_t g_ = 0; g_ < G; g_++) phi[g_] = 0.0;
    DEPOSIT();
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t steps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        DEPOSIT();                                    /* (1)+(2) scatter + neutralise */

        double *a = phi, *b = phi_new;                /* (3) Poisson: K Jacobi sweeps */
        for (int k = 0; k < K; k++) { poisson_jacobi(b, a, rho, NG); double *t = a; a = b; b = t; }
        if (a != phi) memcpy(phi, a, G * sizeof(double));   /* leave the result in phi */

        for (int i = 0; i < NG; i++) {                /* (4) E = -grad phi (central diff) */
            int ip = (i + 1) % NG, im = (i - 1 + NG) % NG;
            for (int j = 0; j < NG; j++) {
                int jp = (j + 1) % NG, jm = (j - 1 + NG) % NG;
                Ex[i * NG + j] = -0.5 * (phi[i * NG + jp] - phi[i * NG + jm]);
                Ey[i * NG + j] = -0.5 * (phi[ip * NG + j] - phi[im * NG + j]);
            }
        }

        for (size_t p = 0; p < N; p++) {              /* (5) gather E + (6) push */
            int gx = (int)px[p], gy = (int)py[p];
            double fx = px[p] - gx, fy = py[p] - gy;
            int gx1 = (gx + 1) % NG, gy1 = (gy + 1) % NG;
            double w00 = (1.0 - fx) * (1.0 - fy), w10 = fx * (1.0 - fy);
            double w01 = (1.0 - fx) * fy,         w11 = fx * fy;
            double exi = w00 * Ex[(size_t)gy * NG + gx] + w10 * Ex[(size_t)gy * NG + gx1]
                       + w01 * Ex[(size_t)gy1 * NG + gx] + w11 * Ex[(size_t)gy1 * NG + gx1];
            double eyi = w00 * Ey[(size_t)gy * NG + gx] + w10 * Ey[(size_t)gy * NG + gx1]
                       + w01 * Ey[(size_t)gy1 * NG + gx] + w11 * Ey[(size_t)gy1 * NG + gx1];
            pvx[p] += exi * dt; pvy[p] += eyi * dt;   /* charge = mass = 1 */
            px[p] += pvx[p] * dt; py[p] += pvy[p] * dt;
            px[p] -= Lg * floor(px[p] / Lg);          /* periodic wrap into [0, NG) */
            py[p] -= Lg * floor(py[p] / Lg);
        }
        steps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = px[0];                     /* a live particle coordinate */

    free(px); free(py); free(pvx); free(pvy);
    munmap(buf, gbytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "steps", steps);
    p2_meta_kv_f64(&m, "particle0_x", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "the particle->grid scatter + grid solve is the distinct write; Jacobi solve is fixed-iteration, not fully converged");
    p2_meta_close(&m);
    return 0;
}
