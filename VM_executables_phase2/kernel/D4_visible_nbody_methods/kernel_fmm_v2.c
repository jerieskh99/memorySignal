/* kernel_fmm_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  FAST MULTIPOLE (single-level):  multipole-accelerated 2D N-body
 * ============================================================================
 *
 *  DWARF   : N-Body Methods (Berkeley computational motif D4)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of the N-body method that summarises clusters
 *            of particles by an EXPANSION. Every step it computes, per box of a
 *            grid, a set of multipole coefficients (a small complex vector) from
 *            that box's particles, then evaluates those expansions at far-away
 *            targets. The per-box expansion-coefficient arrays are the distinctive
 *            WRITE -- a structure that plain nbody, Barnes-Hut and cell-list MD do
 *            not produce.
 *
 *  This uses the classic 2D complex-analysis formulation. Writing a point as a
 *  complex number z = x + i y, the field of unit charges is  G(z) = sum_j q_j /
 *  (z - z_j), and the physical force on a charge is (Ex, Ey) = (Re G, -Im G). A
 *  cluster of charges around a box centre zc is summarised, for targets far from
 *  the box, by its multipole moments  M_k = sum_{j in box} q_j (z_j - zc)^k, via
 *      G(z) ~= sum_{k=0..P} M_k / (z - zc)^{k+1}.
 *
 *  PICTURE (top view):
 *      The plane is tiled into boxes. Each box condenses its particles into a
 *      short vector of multipole coefficients M_0..M_P (computed every step). A
 *      target particle sums the exact particle-particle force from its own box and
 *      the 8 neighbours (NEAR), and the cheap expansion of every other box (FAR).
 *
 *          +----+----+----+----+       box b -> M_0 M_1 M_2 ... M_P  (rebuilt/step)
 *          | Mb | Mb | Mb | Mb |
 *          +----+----+----+----+       target t:
 *          | Mb | N  N  N | Mb |         NEAR (N)  = direct sum over 3x3 boxes
 *          +----+ N [t] N +----+         FAR  (Mb) = evaluate each far box's M
 *          | Mb | N  N  N | Mb |
 *          +----+----+----+----+
 *
 *  ALGORITHM (per step):
 *      1. Rebuild the box grid over the particles' bounding box and bin the
 *         particles (a head/next linked list per box).
 *      2. P2M: for each box, accumulate its multipole moments M_0..M_P from its
 *         particles.  THIS IS THE SIGNATURE WRITE (a coefficient vector per box).
 *      3. For each particle: NEAR = exact (softened) force from the 3x3 block of
 *         boxes; FAR = sum over all other boxes of that box's multipole expansion
 *         evaluated at the particle.
 *      4. Push: v += force*dt ; x += v*dt.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Per step: the compact particle arrays, the box linked-list, and -- the
 *      distinctive part -- a rewritten array of complex multipole coefficients,
 *      one short vector per box. Honest caveats: (a) this is the SINGLE-LEVEL
 *      multipole method (P2M + far evaluation); the full O(N) FMM additionally
 *      writes M2M/M2L/L2L translation coefficients up and down a tree -- the same
 *      kind of expansion-array writes, just more of them; (b) the coefficient
 *      arrays are modest next to the particle arrays at these sizes.
 *
 *  Real-world use: electrostatics, acoustics, and other long-range-force problems
 *  (the full FMM is a landmark O(N) algorithm).
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: N-Body Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (steps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>
#include <complex.h>

static const char *TEST = "kernel_fmm_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign single-level fast-multipole N-body; N-Body kernel)\n"
"  --particles N         Number of charges (default 32768)\n"
"  --box-dim B           Boxes per dimension, grid is B x B (default 24)\n"
"  --terms P             Multipole terms per box (default 16)\n"
"  --soft-milli S        Near-field softening x1000 (default 20 = 0.02)\n"
"  --dt-milli DT         Timestep x1000 (default 5 = 0.005)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on multipole-array bytes (default 8192)\n"
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
    long long nparts     = p2_get_i64(argc, argv, "--particles", 32768);
    long long bdim       = p2_get_i64(argc, argv, "--box-dim", 24);
    long long terms      = p2_get_i64(argc, argv, "--terms", 16);
    long long soft_milli = p2_get_i64(argc, argv, "--soft-milli", 20);   /* float as integer-milli */
    long long dt_milli   = p2_get_i64(argc, argv, "--dt-milli", 5);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nparts < 64 || nparts > (1LL << 24)) { P2_LOG_ERR("particles %lld out of range (64..2^24)", nparts); return 2; }
    if (bdim < 4 || bdim > 1024) { P2_LOG_ERR("box-dim %lld out of range (4..1024)", bdim); return 2; }
    if (terms < 1 || terms > 64) { P2_LOG_ERR("terms %lld out of range (1..64)", terms); return 2; }
    size_t N = (size_t)nparts; int B = (int)bdim; int P = (int)terms;
    double soft = (double)soft_milli / 1000.0, soft2 = soft * soft;
    double dt = (double)dt_milli / 1000.0;
    size_t nbox = (size_t)B * (size_t)B;
    size_t ncoef = nbox * (size_t)(P + 1);           /* complex coefficients total */
    size_t mbytes = ncoef * sizeof(double complex);  /* the multipole-array footprint */
    if (mbytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("multipole bytes %zu exceed --max-mb %lld", mbytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "N-Body Methods");
    p2_meta_kv_str(&m, "scheme", "single-level fast multipole (P2M + far multipole eval, near direct); complex 2D expansions");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "particles", nparts);
    p2_meta_kv_i64(&m, "box_dim", bdim);
    p2_meta_kv_i64(&m, "terms", terms);
    p2_meta_kv_i64(&m, "soft_milli", soft_milli);
    p2_meta_kv_i64(&m, "dt_milli", dt_milli);
    p2_meta_kv_u64(&m, "multipole_bytes", mbytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The multipole coefficient block is the signature buffer -> mmap + mlock it.
     * Layout: box b owns M[b*(P+1) .. b*(P+1)+P]. */
    double complex *M = (double complex *)mmap(NULL, mbytes, PROT_READ | PROT_WRITE,
                                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (M == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", mbytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(M, mbytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(M, mbytes);

    /* particle state + per-box linked list + per-box centre */
    double *x  = (double *)malloc(N * sizeof(double));
    double *y  = (double *)malloc(N * sizeof(double));
    double *vx = (double *)malloc(N * sizeof(double));
    double *vy = (double *)malloc(N * sizeof(double));
    double *q  = (double *)malloc(N * sizeof(double));
    int    *head = (int *)malloc(nbox * sizeof(int));
    int    *next = (int *)malloc(N * sizeof(int));
    double complex *zc = (double complex *)malloc(nbox * sizeof(double complex));
    if (!x || !y || !vx || !vy || !q || !head || !next || !zc) {
        free(x); free(y); free(vx); free(vy); free(q); free(head); free(next); free(zc);
        munmap(M, mbytes); P2_LOG_ERR("malloc failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) {                 /* random charges in a unit box */
        x[i] = rng_unit(&rng); y[i] = rng_unit(&rng);
        vx[i] = 0.0; vy[i] = 0.0;
        q[i] = (i & 1) ? 1.0 : -1.0;                 /* alternating +/- : quasi-neutral, bounded */
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t steps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* (1) bounding box + bin particles into the B x B grid */
        double minx = x[0], maxx = x[0], miny = y[0], maxy = y[0];
        for (size_t i = 1; i < N; i++) {
            if (x[i] < minx) minx = x[i]; if (x[i] > maxx) maxx = x[i];
            if (y[i] < miny) miny = y[i]; if (y[i] > maxy) maxy = y[i];
        }
        double bw = (maxx - minx), bh = (maxy - miny);
        double hx = (bw > 0 ? bw : 1.0) / B, hy = (bh > 0 ? bh : 1.0) / B;   /* box sizes */
        for (size_t b = 0; b < nbox; b++) head[b] = -1;
        for (size_t i = 0; i < N; i++) {
            int bx = (int)((x[i] - minx) / hx); if (bx < 0) bx = 0; if (bx >= B) bx = B - 1;
            int by = (int)((y[i] - miny) / hy); if (by < 0) by = 0; if (by >= B) by = B - 1;
            int b = by * B + bx; next[i] = head[b]; head[b] = (int)i;
        }
        for (int by = 0; by < B; by++)                /* box centres */
            for (int bx = 0; bx < B; bx++)
                zc[by * B + bx] = (minx + (bx + 0.5) * hx) + (miny + (by + 0.5) * hy) * I;

        /* (2) P2M: multipole moments M_k = sum_j q_j (z_j - zc)^k per box. The
         * write of these coefficient vectors is the workload's signature. */
        for (size_t b = 0; b < nbox; b++)
            for (int k = 0; k <= P; k++) M[b * (P + 1) + k] = 0.0;
        for (int by = 0; by < B; by++) {
            for (int bx = 0; bx < B; bx++) {
                int b = by * B + bx;
                double complex *Mb = M + (size_t)b * (P + 1);
                double complex c = zc[b];
                for (int i = head[b]; i >= 0; i = next[i]) {
                    double complex dz = (x[i] + y[i] * I) - c;
                    double complex term = q[i];        /* q_j * dz^0 */
                    for (int k = 0; k <= P; k++) { Mb[k] += term; term *= dz; }
                }
            }
        }

        /* (3) forces: NEAR direct (softened) over the 3x3 boxes, FAR via each
         * other box's multipole expansion. Store the acceleration in (vx,vy)
         * increments computed on the fly (semi-implicit Euler). */
        for (int tby = 0; tby < B; tby++) {
            for (int tbx = 0; tbx < B; tbx++) {
                int tb = tby * B + tbx;
                for (int i = head[tb]; i >= 0; i = next[i]) {
                    double xi = x[i], yi = y[i];
                    double complex z = xi + yi * I;
                    double ex = 0.0, ey = 0.0;
                    /* NEAR: exact softened particle-particle over the 3x3 block */
                    for (int dby = -1; dby <= 1; dby++) {
                        int nby = tby + dby; if (nby < 0 || nby >= B) continue;
                        for (int dbx = -1; dbx <= 1; dbx++) {
                            int nbx = tbx + dbx; if (nbx < 0 || nbx >= B) continue;
                            for (int j = head[nby * B + nbx]; j >= 0; j = next[j]) {
                                if (j == i) continue;
                                double rx = xi - x[j], ry = yi - y[j];
                                double r2 = rx * rx + ry * ry + soft2;
                                double f = q[j] / r2;      /* softened 2D Coulomb field */
                                ex += f * rx; ey += f * ry;
                            }
                        }
                    }
                    /* FAR: evaluate every non-neighbour box's multipole expansion
                     * G(z) = sum_k M_k / (z - zc)^{k+1}; force = (Re G, -Im G). */
                    for (int fby = 0; fby < B; fby++) {
                        int ady = fby - tby; if (ady < 0) ady = -ady;
                        for (int fbx = 0; fbx < B; fbx++) {
                            int adx = fbx - tbx; if (adx < 0) adx = -adx;
                            if (adx <= 1 && ady <= 1) continue;   /* skip the near 3x3 */
                            int fb = fby * B + fbx;
                            const double complex *Mb = M + (size_t)fb * (P + 1);
                            double complex w = 1.0 / (z - zc[fb]);  /* 1/(z-zc) */
                            double complex powk = w, G = 0.0;
                            for (int k = 0; k <= P; k++) { G += Mb[k] * powk; powk *= w; }
                            ex += creal(G); ey += -cimag(G);
                        }
                    }
                    /* accelerate this charge (mass = 1); q_i sets the sign */
                    vx[i] += q[i] * ex * dt; vy[i] += q[i] * ey * dt;
                }
            }
        }

        for (size_t i = 0; i < N; i++) { x[i] += vx[i] * dt; y[i] += vy[i] * dt; }   /* (4) push */
        steps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = x[0];

    free(x); free(y); free(vx); free(vy); free(q); free(head); free(next); free(zc);
    munmap(M, mbytes);

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
                   "single-level multipole (P2M + far eval); full FMM adds M2M/M2L/L2L; coeff arrays modest vs particles");
    p2_meta_close(&m);
    return 0;
}
