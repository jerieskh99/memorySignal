/* kernel_nbody_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  BASELINE N-BODY:  2D particle simulation, K-sampled softened gravity
 * ============================================================================
 *
 *  DWARF   : N-Body Methods (D4)         (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the write behaviour of a genuinely time-evolving particle
 *            system. This is the BASELINE N-body: no tree, no cell list, no grid,
 *            just the raw particle update. The tree / list / grid variants layer
 *            extra structure on top of exactly this loop, so this file isolates
 *            the "particle arrays evolving smoothly" signature by itself.
 *
 *  PICTURE (top view):
 *
 *      periodic box [0,L) x [0,L)              state = four compact arrays:
 *      +----------------------------+
 *      |   .        \   .            |            px:  [ . . . . . . . . ]
 *      |     .   <--- * ---> .       |            py:  [ . . . . . . . . ]
 *      |       .    /|\    .         |            vx:  [ . . . . . . . . ]
 *      |    .      / | \      .      |            vy:  [ . . . . . . . . ]
 *      +----------------------------+
 *        each particle * is pulled by force arrows from a RANDOM sample of K
 *        others (not all N); every step nudges all four arrays by a small amount
 *
 *  ALGORITHM (per timestep, for every particle i):
 *      1. Sample K other particles at random and sum a softened gravitational
 *         pull from each. "Softened" means a small constant (SOFT) is added to
 *         the squared distance so two near-coincident particles never produce an
 *         infinite force -- this keeps the integration numerically stable.
 *      2. Integrate (semi-implicit Euler): update velocity from the force, apply
 *         a mild damping factor, then advance the position by the new velocity.
 *      3. Wrap the position back into the box with fmod (periodic boundary), so
 *         a particle leaving one edge re-enters from the opposite edge.
 *      Positions evolve SMOOTHLY across steps -- there is no re-seeding, so this
 *      is real time evolution, not a fresh random cloud each step.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Four compact 1D arrays (px, py, vx, vy) fully rewritten every step with
 *      small, low-magnitude increments. That smooth per-value drift is the
 *      distinctive tell: it looks nothing like the full recompute of a GEMM/FFT
 *      nor the 2D block structure of a stencil. The K-sample is a small indirect
 *      gather into the position arrays and, being read-only, is invisible to a
 *      write-only observer.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: N-Body Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (timesteps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_nbody_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign 2D N-body particle simulation; N-Body kernel)\n"
"  --particles N         Particle count (default 262144; uses 4 * N * 8 bytes)\n"
"  --neighbors K         Sampled interactions per particle per step (default 16)\n"
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
    long long parts      = p2_get_i64(argc, argv, "--particles", 262144);
    long long neigh      = p2_get_i64(argc, argv, "--neighbors", 16);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (parts < 256 || parts > 268435456LL) { P2_LOG_ERR("particles %lld out of range", parts); return 2; }
    if (neigh < 1 || neigh > 4096) { P2_LOG_ERR("neighbors %lld out of range (1..4096)", neigh); return 2; }
    size_t N = (size_t)parts, K = (size_t)neigh;
    size_t bytes = 4 * N * sizeof(double);    /* px, py, vx, vy */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "N-Body Methods");
    p2_meta_kv_str(&m, "scheme", "2D periodic box, K-sampled softened gravity");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "particles", parts);
    p2_meta_kv_i64(&m, "neighbors", neigh);
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
    /* One flat arena, sliced into four contiguous halves. Keeping px/py/vx/vy as
     * a structure-of-arrays (rather than an array of particle structs) makes each
     * per-step rewrite a set of long sequential streaming stores. */
    double *px = arena, *py = arena + N, *vx = arena + 2 * N, *vy = arena + 3 * N;

    /* Simulation constants. L: side of the periodic box. DT: integration
     * timestep. SOFT: softening added to squared distance (finite close-range
     * force). DAMP: per-step velocity damping (<1) that bleeds off energy so the
     * system stays bounded over a long measurement run. */
    const double L = 1000.0, DT = 0.01, SOFT = 1.0, DAMP = 0.999;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) {
        px[i] = rng_unit(&rng) * L; py[i] = rng_unit(&rng) * L;
        vx[i] = 0.0; vy[i] = 0.0;
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t steps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        for (size_t i = 0; i < N; i++) {
            /* Accumulate the net force on particle i from a random sample of K
             * other particles. Sampling K (rather than all N) keeps the per-step
             * cost O(N*K) instead of O(N^2) while preserving the smooth, diffuse
             * evolution characteristic of a gravitating cloud. */
            double fx = 0.0, fy = 0.0, xi = px[i], yi = py[i];
            for (size_t s = 0; s < K; s++) {
                size_t j = (size_t)(p2_rng_next(&rng) % (uint64_t)N);  /* random partner (may hit i; harmless) */
                double dx = px[j] - xi, dy = py[j] - yi;               /* displacement toward j */
                double d2 = dx * dx + dy * dy + SOFT;                  /* softened squared distance */
                double inv = 1.0 / sqrt(d2);
                double inv3 = inv * inv * inv;                         /* 1/d^3: the 1/r^2 law times the unit direction */
                fx += dx * inv3; fy += dy * inv3;
            }
            /* Semi-implicit Euler: update velocity first (with damping), then use
             * the NEW velocity to advance position -- more stable than plain Euler. */
            double nvx = (vx[i] + DT * fx) * DAMP;
            double nvy = (vy[i] + DT * fy) * DAMP;
            vx[i] = nvx; vy[i] = nvy;
            /* Advance and wrap into the periodic box. fmod can return a negative
             * value when the coordinate goes below 0, so fold it back up by L. */
            double nx = fmod(xi + DT * nvx, L); if (nx < 0) nx += L;
            double ny = fmod(yi + DT * nvy, L); if (ny < 0) ny += L;
            px[i] = nx; py[i] = ny;
        }
        steps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = px[N / 2] + py[N / 2];

    munmap(arena, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "steps", steps);
    p2_meta_kv_f64(&m, "sink", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: four compact particle arrays rewritten per step; smooth evolution");
    p2_meta_close(&m);
    return 0;
}
