/* kernel_fdtd_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  FDTD ELECTROMAGNETICS:  leapfrogged electric/magnetic fields on a Yee grid
 * ============================================================================
 *
 *  DWARF   : Structured Grids (Berkeley computational motif D5)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of a structured-grid method that advances TWO
 *            COUPLED fields in leapfrog, rather than relaxing one field like the
 *            jacobi/seidel/multigrid stencils. Finite-Difference Time-Domain steps
 *            Maxwell's equations: the magnetic field is updated from the curl of
 *            the electric field, then the electric field from the curl of the
 *            magnetic field, alternating. The dual-grid, E<->H coupled write is
 *            the distinctive pattern.
 *
 *  This is the 2D transverse-magnetic (TM) mode: one out-of-plane electric
 *  component Ez and two in-plane magnetic components Hx, Hy, on a periodic grid
 *  (normalised units, so the update constant is just the Courant number).
 *
 *  PICTURE (top view):  three co-located field grids, updated in two half-steps.
 *
 *          Ez grid          Hx, Hy grids
 *        +--+--+--+        +--+--+--+       step 1 (H from curl E):
 *        |Ez|Ez|Ez|        |Hx|Hx|..|         Hx -= C (Ez[y+1]-Ez[y])
 *        +--+--+--+   <-->  +--+--+--+         Hy += C (Ez[x+1]-Ez[x])
 *        |Ez|Ez|Ez|        |Hy|Hy|..|       step 2 (E from curl H):
 *        +--+--+--+        +--+--+--+         Ez += C((Hy[x]-Hy[x-1])
 *                                                     -(Hx[y]-Hx[y-1]))
 *
 *  ALGORITHM (per step, Yee leapfrog):
 *      1. Update H from the spatial curl of E:
 *           Hx[x,y] -= C (Ez[x,y+1] - Ez[x,y])
 *           Hy[x,y] += C (Ez[x+1,y] - Ez[x,y])
 *      2. Update E from the spatial curl of the just-updated H:
 *           Ez[x,y] += C ((Hy[x,y] - Hy[x-1,y]) - (Hx[x,y] - Hx[x,y-1]))
 *      C is the Courant number (<= 1/sqrt(2) in 2D for stability). The field is
 *      seeded with a Gaussian pulse in Ez and left to propagate and wrap around
 *      the periodic box; with no losses the discrete energy stays bounded.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Three field grids, two of them (H) then the third (E) fully rewritten every
 *      step, in a leapfrog where each depends on the other. This coupled dual-grid
 *      write is distinct from the single-field relaxation of jacobi/seidel and the
 *      multi-resolution pyramid of multigrid. Honest caveat: structurally it is
 *      closer to "two coupled Jacobi grids" than the nine-array Lattice-Boltzmann;
 *      the distinguishing feature is the E<->H coupling and the two-grid write.
 *
 *  Real-world use: antenna, radar-cross-section, and photonics/optics simulation
 *  (e.g. Meep).
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Structured Grids. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + pulse init) / measure (FDTD steps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_fdtd_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign 2D FDTD electromagnetics; Structured-Grid kernel)\n"
"  --width W             Grid width (default 512)\n"
"  --height H            Grid height (default 512)\n"
"  --courant-milli C     Courant number x1000, <= 707 for 2D stability (default 500 = 0.5)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (unused; deterministic pulse) (default 42)\n"
"  --max-mb N            Hard cap on field bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long W_ll        = p2_get_i64(argc, argv, "--width", 512);
    long long H_ll        = p2_get_i64(argc, argv, "--height", 512);
    long long cour_milli  = p2_get_i64(argc, argv, "--courant-milli", 500);   /* float as milli */
    long long duration_s  = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s    = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb      = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu         = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed        = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock    = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run     = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir    = p2_get_str(argc, argv, "--output-dir", NULL);

    if (W_ll < 8 || W_ll > 16384 || H_ll < 8 || H_ll > 16384) { P2_LOG_ERR("W/H out of range (8..16384)"); return 2; }
    if (cour_milli < 1 || cour_milli > 707) { P2_LOG_ERR("courant-milli %lld out of range (1..707)", cour_milli); return 2; }
    int W = (int)W_ll, H = (int)H_ll;
    double C = (double)cour_milli / 1000.0;
    size_t ncells = (size_t)W * (size_t)H;
    size_t bytes = 3 * ncells * sizeof(double);   /* Ez, Hx, Hy */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("field bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Structured Grids");
    p2_meta_kv_str(&m, "scheme", "2D TM FDTD (Yee leapfrog: H from curl E, then E from curl H; periodic)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "width", W_ll);
    p2_meta_kv_i64(&m, "height", H_ll);
    p2_meta_kv_i64(&m, "courant_milli", cour_milli);
    p2_meta_kv_u64(&m, "field_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* three field grids in one mmap'd block: Ez, Hx, Hy (row-major, idx = y*W + x) */
    double *base = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(base, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(base, bytes);
    double *Ez = base, *Hx = base + ncells, *Hy = base + 2 * ncells;
    (void)seed;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* seed a Gaussian pulse in Ez at the grid centre; H starts at zero. */
    for (size_t k = 0; k < 3 * ncells; k++) base[k] = 0.0;
    {
        double cx = W * 0.5, cy = H * 0.5, sig = (W < H ? W : H) * 0.05;
        double s2 = 2.0 * sig * sig;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                double dx = x - cx, dy = y - cy;
                Ez[(size_t)y * W + x] = exp(-(dx * dx + dy * dy) / s2);
            }
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t steps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* (1) update H from the curl of E (periodic neighbours in +x / +y) */
        for (int y = 0; y < H; y++) {
            int yp = (y + 1) % H;
            for (int x = 0; x < W; x++) {
                int xp = (x + 1) % W;
                size_t id = (size_t)y * W + x;
                Hx[id] -= C * (Ez[(size_t)yp * W + x] - Ez[id]);
                Hy[id] += C * (Ez[(size_t)y * W + xp] - Ez[id]);
            }
        }
        /* (2) update E from the curl of the just-updated H (neighbours in -x / -y) */
        for (int y = 0; y < H; y++) {
            int ym = (y - 1 + H) % H;
            for (int x = 0; x < W; x++) {
                int xm = (x - 1 + W) % W;
                size_t id = (size_t)y * W + x;
                Ez[id] += C * ((Hy[id] - Hy[(size_t)y * W + xm]) -
                               (Hx[id] - Hx[(size_t)ym * W + x]));
            }
        }
        steps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* discrete field energy (bounded for the lossless periodic scheme) */
    double energy = 0.0;
    for (size_t k = 0; k < ncells; k++)
        energy += Ez[k] * Ez[k] + Hx[k] * Hx[k] + Hy[k] * Hy[k];
    volatile double sink = Ez[(H / 2) * (size_t)W + (W / 2)];

    munmap(base, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "steps", steps);
    p2_meta_kv_f64(&m, "field_energy", energy);
    p2_meta_kv_f64(&m, "ez_center", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "coupled E/H dual-grid leapfrog is the distinct write; periodic BCs, no absorbing boundary");
    p2_meta_close(&m);
    return 0;
}
