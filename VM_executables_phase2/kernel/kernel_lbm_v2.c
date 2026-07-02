/* kernel_lbm_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  LATTICE-BOLTZMANN (D2Q9):  fluid flow by streaming + colliding distributions
 * ============================================================================
 *
 *  DWARF   : Structured Grids (Berkeley computational motif D5)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of a structured-grid method that carries
 *            MANY field arrays per cell instead of one. A single-field stencil
 *            (jacobi/seidel/multigrid) rewrites one grid; Lattice-Boltzmann keeps
 *            NINE particle-distribution values per cell (the D2Q9 model) and, each
 *            step, STREAMS them to neighbour cells and then COLLIDES them locally.
 *            The wide, nine-array, streaming write is the distinctive pattern.
 *
 *  D2Q9 = 2D lattice, 9 discrete velocities per cell (rest + 4 axial + 4 diagonal):
 *
 *          6   2   5           index : velocity (ex,ey)      weight
 *            \ | /             0 : ( 0, 0)  rest             4/9
 *          3 - 0 - 1           1..4 : (+1,0)(0,+1)(-1,0)(0,-1) 1/9
 *            / | \             5..8 : (+1,+1)(-1,+1)(-1,-1)(+1,-1) 1/36
 *          7   4   8
 *
 *  PICTURE (top view):
 *      Each cell holds 9 distribution values f_0..f_8. STREAMING moves f_i from a
 *      cell to its neighbour in direction i (here done as a "pull": each cell
 *      gathers the incoming f_i from the upstream neighbour x - e_i). COLLISION
 *      then relaxes the 9 values toward their local equilibrium.
 *
 *          cell (x,y):  [f0 f1 f2 f3 f4 f5 f6 f7 f8]   <- 9 values, rewritten/step
 *              stream: f_i comes from neighbour in direction -e_i
 *              collide: f_i <- f_i - omega (f_i - f_i^eq(rho,u))
 *
 *  ALGORITHM (per step, fused pull-stream + BGK collision, two lattices):
 *      1. For each cell, PULL the 9 incoming distributions from the upstream
 *         neighbours (periodic wrap).
 *      2. Macroscopic moments: density rho = sum_i f_i, velocity u = (sum_i f_i e_i)/rho.
 *      3. Equilibrium f_i^eq = w_i rho (1 + 3 e_i.u + 4.5 (e_i.u)^2 - 1.5 |u|^2).
 *      4. BGK COLLISION: f_i <- f_i - omega (f_i - f_i^eq), write into the second
 *         lattice, then swap lattices.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A structured grid, but NINE distribution arrays wide, fully rewritten every
 *      step, with a directional streaming shift between neighbour cells. This is a
 *      far wider, multi-stream grid write than the single-scalar stencils, which
 *      is what makes it a distinct D5 member. (Two lattices are used, so the
 *      allocated footprint is ~2x the live field.)
 *
 *  Real-world use: computational fluid dynamics, especially flow in complex or
 *  porous geometries (OpenLB, Palabos).
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Structured Grids. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (LBM steps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_lbm_v2";

/* D2Q9 lattice velocities and weights (index order matches the diagram above). */
static const int    EX[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
static const int    EY[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
static const double WT[9] = { 4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
                              1.0/36, 1.0/36, 1.0/36, 1.0/36 };

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Lattice-Boltzmann D2Q9 fluid; Structured-Grid kernel)\n"
"  --width W             Lattice width (default 256)\n"
"  --height H            Lattice height (default 256)\n"
"  --omega-milli OM      BGK relaxation rate x1000, in (0,2000) (default 1000 = 1.0)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on lattice bytes (default 8192)\n"
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
    long long W_ll       = p2_get_i64(argc, argv, "--width", 256);
    long long H_ll       = p2_get_i64(argc, argv, "--height", 256);
    long long omega_milli = p2_get_i64(argc, argv, "--omega-milli", 1000);   /* float as milli */
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (W_ll < 8 || W_ll > 8192 || H_ll < 8 || H_ll > 8192) { P2_LOG_ERR("W/H out of range (8..8192)"); return 2; }
    if (omega_milli <= 0 || omega_milli >= 2000) { P2_LOG_ERR("omega-milli %lld out of range (1..1999)", omega_milli); return 2; }
    int W = (int)W_ll, H = (int)H_ll;
    double omega = (double)omega_milli / 1000.0;
    size_t ncells = (size_t)W * (size_t)H;
    size_t bytes = 2 * ncells * 9 * sizeof(double);   /* two lattices, 9 dists/cell */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("lattice bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Structured Grids");
    p2_meta_kv_str(&m, "scheme", "Lattice-Boltzmann D2Q9 (fused pull-stream + BGK collision; 9 distributions/cell)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "width", W_ll);
    p2_meta_kv_i64(&m, "height", H_ll);
    p2_meta_kv_i64(&m, "omega_milli", omega_milli);
    p2_meta_kv_u64(&m, "lattice_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* Two lattices f and f2 (9 distributions per cell, cell-major AoS), one mmap'd
     * block. The step reads f (pull + collide) and writes f2, then swaps. */
    double *base = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(base, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(base, bytes);
    double *f = base, *f2 = base + ncells * 9;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* initialise at rest equilibrium with small per-cell density perturbations, so
     * pressure waves propagate and the field keeps evolving (mass is conserved). */
    for (size_t c = 0; c < ncells; c++) {
        double rho = 1.0 + 0.01 * (2.0 * rng_unit(&rng) - 1.0);
        for (int i = 0; i < 9; i++) f[c * 9 + i] = WT[i] * rho;   /* f_i^eq at u = 0 */
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t steps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                double fi[9];
                /* (1) STREAM (pull): gather f_i from the upstream neighbour x - e_i */
                for (int i = 0; i < 9; i++) {
                    int sx = (x - EX[i] + W) % W;
                    int sy = (y - EY[i] + H) % H;
                    fi[i] = f[((size_t)sy * W + sx) * 9 + i];
                }
                /* (2) macroscopic density and velocity */
                double rho = 0.0, ux = 0.0, uy = 0.0;
                for (int i = 0; i < 9; i++) { rho += fi[i]; ux += fi[i] * EX[i]; uy += fi[i] * EY[i]; }
                double inv = (rho > 1e-12) ? 1.0 / rho : 0.0;
                ux *= inv; uy *= inv;
                double usq = ux * ux + uy * uy;
                /* (3)+(4) equilibrium and BGK collision, written into the 2nd lattice */
                double *out = f2 + ((size_t)y * W + x) * 9;
                for (int i = 0; i < 9; i++) {
                    double eu = EX[i] * ux + EY[i] * uy;
                    double feq = WT[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq);
                    out[i] = fi[i] - omega * (fi[i] - feq);
                }
            }
        }
        double *tmp = f; f = f2; f2 = tmp;   /* swap lattices */
        steps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* report total mass (sum of all distributions) -- an LBM invariant */
    double mass = 0.0;
    for (size_t k = 0; k < ncells * 9; k++) mass += f[k];
    volatile double sink = f[0];

    munmap(base, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "steps", steps);
    p2_meta_kv_f64(&m, "total_mass", mass);
    p2_meta_kv_f64(&m, "f0_sample", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "nine-distribution streaming grid is the distinct write; periodic BCs, single-relaxation BGK");
    p2_meta_close(&m);
    return 0;
}
