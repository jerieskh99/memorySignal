/* kernel_sph_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  SMOOTHED-PARTICLE HYDRODYNAMICS:  a mesh-free fluid, two neighbour passes
 * ============================================================================
 *
 *  DWARF   : N-Body Methods (Berkeley computational motif D4)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of the N-body variant that carries FLUID
 *            state on every particle. Where plain nbody / MD write only position
 *            and velocity, SPH also writes a per-particle DENSITY and PRESSURE,
 *            computed by an extra neighbour pass. Those extra per-particle field
 *            arrays -- and the two-pass structure -- are the distinctive write.
 *
 *  SPH represents a fluid as particles; any field at a point is a smoothed sum of
 *  nearby particle values weighted by a smoothing kernel W of radius h. Each step
 *  needs TWO neighbour passes because pressure depends on density: first sum W to
 *  get each particle's density, convert to pressure, then sum the pressure-kernel
 *  gradient to get the force. Neighbours are found with a cell list (as in the MD
 *  kernel), cells sized >= h so only the 3x3 block around a particle is scanned.
 *
 *  PICTURE (top view):
 *      Particles carry fluid fields; each step makes two passes over the same 3x3
 *      cell neighbourhood.
 *
 *          neighbours within h              per particle i:
 *              .  o  .                        pass 1:  rho_i = sum_j m W(r_ij, h)
 *              o [i] o   <-- smoothing        pressure: p_i  = k (rho_i - rho0)
 *              .  o  .       radius h         pass 2:  f_i   = -sum_j (...) gradW
 *
 *      per step:  rebuild cell list  ->  DENSITY pass (write rho)  ->  pressure
 *                 ->  FORCE pass (write accel)  ->  integrate (+ reflective walls)
 *
 *  ALGORITHM (per step):
 *      1. Rebuild the linked cell list from the current positions.
 *      2. Density pass: rho_i = sum over 3x3-cell neighbours of m * W_poly6(r,h)
 *         (includes the self term at r = 0).                 -> WRITE rho[]
 *      3. Pressure (equation of state): p_i = k * (rho_i - rho0).  -> WRITE pres[]
 *      4. Force pass: pressure force from the spiky-kernel gradient over the same
 *         neighbours, divided by density, plus gravity.       -> WRITE ax[], ay[]
 *      5. Integrate (semi-implicit Euler) with light damping; reflect at the walls
 *         so the fluid stays in the box.
 *
 *  Kernels are the standard Mueller (2003) 2D poly6 (density) and spiky-gradient
 *  (pressure) smoothing kernels. rho0 (rest density) is measured from the initial
 *  lattice so the fluid starts near equilibrium, which keeps the run stable.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Per step: the compact particle position/velocity arrays, the cell-list
 *      buckets, AND -- the distinctive part -- the per-particle density and
 *      pressure arrays, written by an extra neighbour pass. Honest caveat: this is
 *      structurally the cell-list MD kernel plus two more per-particle field
 *      arrays and a second neighbour pass, so its signature is a close relative of
 *      md_lj rather than something wholly new.
 *
 *  Real-world use: fluid effects in film VFX (water, lava, smoke) and
 *  astrophysical gas dynamics.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: N-Body Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + lattice init + rest-density calibration) / measure / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_sph_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign smoothed-particle hydrodynamics; N-Body kernel)\n"
"  --particles N         Number of fluid particles (default 65536)\n"
"  --h-milli H           Smoothing length x1000 (default 2000 = 2.0 lattice units)\n"
"  --k-milli K           Pressure stiffness x1000 (default 1000 = 1.0)\n"
"  --gravity-milli G     Gravity x1000 (default 100 = 0.1)\n"
"  --dt-milli DT         Timestep x1000 (default 5 = 0.005)\n"
"  --damp-milli D        Velocity damping per step x1000 (default 10 = 0.01)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on particle-array bytes (default 8192)\n"
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

/* Fluid state. Particle columns are one mmap'd structure-of-arrays block; rho and
 * pres are the extra fields SPH writes that plain nbody/MD do not. The cell list
 * (head[cell], next[atom]) is the same linked-cell layout as the MD kernel. */
typedef struct {
    size_t N;
    double L;               /* box side */
    double h, h2;           /* smoothing length and its square */
    double m;               /* particle mass */
    double c6;              /* poly6 kernel normalisation (density)   */
    double csg;             /* spiky-gradient normalisation (force)   */
    int    nc; double cell; /* cells per dim, cell side (>= h)        */
    double *x, *y, *vx, *vy;
    double *rho, *pres;     /* the distinctive per-particle fields    */
    double *ax, *ay;
    int    *head, *next;
} SPH;

static inline int sph_cell_of(const SPH *s, double px, double py) {
    int cx = (int)(px / s->cell); if (cx < 0) cx = 0; if (cx >= s->nc) cx = s->nc - 1;
    int cy = (int)(py / s->cell); if (cy < 0) cy = 0; if (cy >= s->nc) cy = s->nc - 1;
    return cy * s->nc + cx;
}

/* (1) rebuild the linked cell list (same bucket write as md_lj). */
static void sph_build_cells(SPH *s) {
    int ncells = s->nc * s->nc;
    for (int c = 0; c < ncells; c++) s->head[c] = -1;
    for (size_t i = 0; i < s->N; i++) {
        int c = sph_cell_of(s, s->x[i], s->y[i]);
        s->next[i] = s->head[c]; s->head[c] = (int)i;
    }
}

/* (2) density pass: rho_i = sum_j m * W_poly6(r_ij, h), including the self term.
 * W_poly6(r,h) = c6 * (h^2 - r^2)^3 for r < h (c6 = 4 / (pi h^8) in 2D). This is
 * the first of the two neighbour passes and the primary extra WRITE. */
static void sph_density(SPH *s) {
    for (size_t i = 0; i < s->N; i++) s->rho[i] = 0.0;
    for (int cy = 0; cy < s->nc; cy++) {
        for (int cx = 0; cx < s->nc; cx++) {
            for (int i = s->head[cy * s->nc + cx]; i >= 0; i = s->next[i]) {
                double xi = s->x[i], yi = s->y[i], acc = 0.0;
                for (int dy = -1; dy <= 1; dy++) {
                    int ny = cy + dy; if (ny < 0 || ny >= s->nc) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = cx + dx; if (nx < 0 || nx >= s->nc) continue;
                        for (int j = s->head[ny * s->nc + nx]; j >= 0; j = s->next[j]) {
                            double rx = xi - s->x[j], ry = yi - s->y[j];
                            double r2 = rx * rx + ry * ry;
                            if (r2 < s->h2) {                 /* r < h (self: r2 = 0 counts) */
                                double d = s->h2 - r2;
                                acc += s->m * s->c6 * d * d * d;
                            }
                        }
                    }
                }
                s->rho[i] = acc;
            }
        }
    }
}

/* (4) force pass: pressure force from the spiky-kernel gradient over the same
 * neighbours. grad W_spiky(r,h) = csg * (h - r)^2 * r_vec / r, csg = -30/(pi h^5).
 * The symmetric SPH pressure force is  f_i = -sum_j m (p_i + p_j)/(2 rho_j) gradW.
 * Reads rho/pres (written in the density/pressure steps), writes the accelerations. */
static void sph_forces(SPH *s, double g) {
    for (int cy = 0; cy < s->nc; cy++) {
        for (int cx = 0; cx < s->nc; cx++) {
            for (int i = s->head[cy * s->nc + cx]; i >= 0; i = s->next[i]) {
                double xi = s->x[i], yi = s->y[i], pi = s->pres[i];
                double fx = 0.0, fy = 0.0;
                for (int dy = -1; dy <= 1; dy++) {
                    int ny = cy + dy; if (ny < 0 || ny >= s->nc) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = cx + dx; if (nx < 0 || nx >= s->nc) continue;
                        for (int j = s->head[ny * s->nc + nx]; j >= 0; j = s->next[j]) {
                            if (j == i) continue;
                            double rx = xi - s->x[j], ry = yi - s->y[j];
                            double r2 = rx * rx + ry * ry;
                            if (r2 <= 0.0 || r2 >= s->h2) continue;
                            double r = sqrt(r2);
                            double hr = s->h - r;
                            double gmag = s->csg * hr * hr;   /* |grad W| along r_hat */
                            double gx = gmag * rx / r, gy = gmag * ry / r;
                            double coeff = -s->m * (pi + s->pres[j]) / (2.0 * s->rho[j]);
                            fx += coeff * gx; fy += coeff * gy;
                        }
                    }
                }
                double invrho = (s->rho[i] > 1e-12) ? 1.0 / s->rho[i] : 0.0;
                s->ax[i] = fx * invrho;                 /* pressure acceleration */
                s->ay[i] = fy * invrho - g;             /* + downward gravity */
            }
        }
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nparts     = p2_get_i64(argc, argv, "--particles", 65536);
    long long h_milli    = p2_get_i64(argc, argv, "--h-milli", 2000);      /* floats as milli */
    long long k_milli    = p2_get_i64(argc, argv, "--k-milli", 1000);
    long long g_milli    = p2_get_i64(argc, argv, "--gravity-milli", 100);
    long long dt_milli   = p2_get_i64(argc, argv, "--dt-milli", 5);
    long long damp_milli = p2_get_i64(argc, argv, "--damp-milli", 10);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nparts < 64 || nparts > (1LL << 24)) { P2_LOG_ERR("particles %lld out of range (64..2^24)", nparts); return 2; }
    if (h_milli < 500) { P2_LOG_ERR("h-milli %lld too small (>=500)", h_milli); return 2; }
    size_t N = (size_t)nparts;
    double h = (double)h_milli / 1000.0;
    double k = (double)k_milli / 1000.0;
    double g = (double)g_milli / 1000.0;
    double dt = (double)dt_milli / 1000.0;
    double damp = (double)damp_milli / 1000.0;
    /* lattice: g_side x g_side points, unit spacing -> box side L = g_side */
    size_t gside = (size_t)ceil(sqrt((double)N));
    double L = (double)gside;
    int nc = (int)(L / h);
    if (nc < 3) { P2_LOG_ERR("box too small for cell list (nc=%d < 3); raise --particles or lower --h-milli", nc); return 2; }

    size_t pbytes = 8 * N * sizeof(double);   /* x,y,vx,vy,rho,pres,ax,ay */
    if (pbytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("particle bytes %zu exceed --max-mb %lld", pbytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "N-Body Methods");
    p2_meta_kv_str(&m, "scheme", "smoothed-particle hydrodynamics (poly6 density + spiky pressure force, cell-list, two neighbour passes)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "particles", nparts);
    p2_meta_kv_i64(&m, "h_milli", h_milli);
    p2_meta_kv_i64(&m, "k_milli", k_milli);
    p2_meta_kv_i64(&m, "gravity_milli", g_milli);
    p2_meta_kv_i64(&m, "dt_milli", dt_milli);
    p2_meta_kv_i64(&m, "cells_per_dim", nc);
    p2_meta_kv_u64(&m, "particle_bytes", pbytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *buf = (double *)mmap(NULL, pbytes, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", pbytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(buf, pbytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(buf, pbytes);

    int ncells = nc * nc;
    int *head = (int *)malloc((size_t)ncells * sizeof(int));
    int *next = (int *)malloc(N * sizeof(int));
    if (!head || !next) { free(head); free(next); munmap(buf, pbytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    SPH s;
    s.N = N; s.L = L; s.h = h; s.h2 = h * h; s.m = 1.0;
    s.c6  = 4.0 / (M_PI * pow(h, 8.0));      /* 2D poly6 normalisation */
    s.csg = -30.0 / (M_PI * pow(h, 5.0));    /* 2D spiky-gradient normalisation */
    s.nc = nc; s.cell = L / (double)nc;
    s.x = buf; s.y = buf + N; s.vx = buf + 2 * N; s.vy = buf + 3 * N;
    s.rho = buf + 4 * N; s.pres = buf + 5 * N; s.ax = buf + 6 * N; s.ay = buf + 7 * N;
    s.head = head; s.next = next;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) {                 /* lattice positions, tiny jitter */
        size_t gx = i % gside, gy = i / gside;
        s.x[i] = ((double)gx + 0.5) + 0.01 * (2.0 * rng_unit(&rng) - 1.0);
        s.y[i] = ((double)gy + 0.5) + 0.01 * (2.0 * rng_unit(&rng) - 1.0);
        s.vx[i] = 0.0; s.vy[i] = 0.0;
    }
    sph_build_cells(&s);
    sph_density(&s);
    double rho0 = 0.0; for (size_t i = 0; i < N; i++) rho0 += s.rho[i];
    rho0 /= (double)N;                               /* rest density = lattice mean */
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t steps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        sph_build_cells(&s);                                     /* (1) */
        sph_density(&s);                                         /* (2) write rho */
        for (size_t i = 0; i < N; i++) s.pres[i] = k * (s.rho[i] - rho0);   /* (3) write pres */
        sph_forces(&s, g);                                       /* (4) write ax,ay */
        for (size_t i = 0; i < N; i++) {                         /* (5) integrate + walls */
            s.vx[i] = (s.vx[i] + s.ax[i] * dt) * (1.0 - damp);
            s.vy[i] = (s.vy[i] + s.ay[i] * dt) * (1.0 - damp);
            s.x[i] += s.vx[i] * dt; s.y[i] += s.vy[i] * dt;
            if (s.x[i] < 0.0)     { s.x[i] = -s.x[i];         s.vx[i] = -0.5 * s.vx[i]; }
            if (s.x[i] > L)       { s.x[i] = 2.0 * L - s.x[i]; s.vx[i] = -0.5 * s.vx[i]; }
            if (s.y[i] < 0.0)     { s.y[i] = -s.y[i];         s.vy[i] = -0.5 * s.vy[i]; }
            if (s.y[i] > L)       { s.y[i] = 2.0 * L - s.y[i]; s.vy[i] = -0.5 * s.vy[i]; }
        }
        steps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = s.rho[0];                 /* a live density value */

    free(head); free(next);
    munmap(buf, pbytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "steps", steps);
    p2_meta_kv_f64(&m, "rest_density", rho0);
    p2_meta_kv_f64(&m, "particle0_rho", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "two-pass neighbour sum with extra rho/pres fields is the distinct write; close relative of md_lj");
    p2_meta_close(&m);
    return 0;
}
