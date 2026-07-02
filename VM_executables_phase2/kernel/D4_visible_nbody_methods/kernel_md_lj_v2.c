/* kernel_md_lj_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  MOLECULAR DYNAMICS:  Lennard-Jones particles with a cell-list neighbour search
 * ============================================================================
 *
 *  DWARF   : N-Body Methods (Berkeley computational motif D4)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of the N-body variant used most in practice.
 *            Like every N-body method it moves particle arrays each step, but the
 *            neighbour search adds a distinct extra WRITE: a cell list (which
 *            particle is in which spatial cell) that is REBUILT every step. That
 *            periodic bucket rewrite is what separates it, in the write-signal,
 *            from the plain gravity nbody and from Barnes-Hut's tree.
 *
 *  Lennard-Jones is the standard pairwise atomic potential; with a cutoff radius
 *  r_cut each atom only feels atoms within r_cut. To find those cheaply the box
 *  is chopped into cells of side >= r_cut, so an atom only checks the 3x3 block of
 *  cells around its own -- turning the O(N^2) all-pairs search into O(N).
 *
 *  PICTURE (top view):
 *      The box is periodic and tiled into cells of side >= r_cut. A head/next
 *      linked list records the atoms in each cell and is rebuilt every step.
 *
 *          +----+----+----+----+        cell list (rebuilt each step):
 *          |    |    |    |    |          head[c] -> p -> next[p] -> ... -> -1
 *          +----+----+----+----+
 *          |    | i  | .  |    |    atom i (in its cell) scans the 3x3
 *          +----+XXXX+----+----+    neighbourhood (X) and keeps only the
 *          |    | .  | .  |    |    atoms within r_cut of it
 *          +----+----+----+----+
 *
 *      per step:  half-kick v  ->  drift x (+ periodic wrap)  ->  REBUILD cell
 *                 list  ->  recompute LJ forces over the 3x3 cells  ->  half-kick v
 *
 *  ALGORITHM (velocity-Verlet integrator, per step):
 *      1. v += 0.5 * a * dt                 (first half kick, old accelerations)
 *      2. x += v * dt ; wrap into [0,L)     (drift, periodic box)
 *      3. Rebuild the cell list: clear head[], then for each atom p prepend it to
 *         its cell -> next[p] = head[c]; head[c] = p.   (the distinctive write)
 *      4. Recompute accelerations: for each atom, sum the Lennard-Jones force from
 *         the atoms in its own and the 8 neighbouring cells that lie within r_cut
 *         (minimum-image convention for the periodic box).
 *      5. v += 0.5 * a * dt                 (second half kick, new accelerations)
 *
 *  Atoms are initialised on a regular lattice (no overlaps) with small random
 *  velocities, which keeps the integration numerically stable.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Two kinds of writes every step: (a) the compact particle arrays
 *      (position/velocity/acceleration) updated smoothly, and (b) the cell-list
 *      buckets (head[] over the cells + next[] over the atoms) rewritten from
 *      scratch. (b) is the distinctive tell absent from the plain nbody. Honest
 *      caveat: the buckets are O(N) integers, modest next to the particle arrays,
 *      so the extra signal is a bounded fraction of the per-step write volume.
 *
 *  Real-world use: the inner loop of GROMACS / NAMD / LAMMPS / AMBER -- drug
 *  discovery, protein folding, and materials simulation.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: N-Body Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + lattice init + first force) / measure (steps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_md_lj_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Lennard-Jones molecular dynamics; N-Body kernel)\n"
"  --particles N         Number of atoms (default 65536)\n"
"  --density-milli D     Number density x1000 (default 600 = 0.6; sets box L=sqrt(N/D))\n"
"  --rc-milli R          Cutoff radius x1000 (default 2500 = 2.5 sigma)\n"
"  --dt-milli DT         Timestep x1000 (default 2 = 0.002)\n"
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

/* Simulation state. The particle arrays are one contiguous mmap'd block sliced
 * into six N-length columns (structure-of-arrays); the cell list is two integer
 * arrays: head[cell] (first atom in a cell) and next[atom] (next atom in the
 * same cell), the classic O(N) linked-cell layout. */
typedef struct {
    size_t N;               /* number of atoms                 */
    double L;               /* periodic box side               */
    double rc, rc2;         /* cutoff radius and its square     */
    int    nc;              /* cells per dimension              */
    double cell;            /* cell side (= L / nc, >= rc)      */
    double *x, *y;          /* positions                        */
    double *vx, *vy;        /* velocities                       */
    double *ax, *ay;        /* accelerations (forces / unit mass) */
    int    *head;           /* head[nc*nc]: first atom per cell */
    int    *next;           /* next[N]: linked-list successor   */
} MD;

/* Map a position to its cell index, clamped in case a wrapped coordinate lands
 * exactly on the L boundary due to rounding. */
static inline int md_cell_of(const MD *s, double px, double py) {
    int cx = (int)(px / s->cell); if (cx < 0) cx = 0; if (cx >= s->nc) cx = s->nc - 1;
    int cy = (int)(py / s->cell); if (cy < 0) cy = 0; if (cy >= s->nc) cy = s->nc - 1;
    return cy * s->nc + cx;
}

/* (3) Rebuild the linked-cell list from scratch: clear all heads, then prepend
 * each atom to the head of its current cell. This is the workload's signature
 * extra write -- head[] over the cells and next[] over the atoms are both
 * rewritten every step. */
static void md_build_cells(MD *s) {
    int ncells = s->nc * s->nc;
    for (int c = 0; c < ncells; c++) s->head[c] = -1;
    for (size_t i = 0; i < s->N; i++) {
        int c = md_cell_of(s, s->x[i], s->y[i]);
        s->next[i] = s->head[c];
        s->head[c] = (int)i;
    }
}

/* (4) Recompute Lennard-Jones accelerations. For each atom we visit the 3x3 block
 * of cells around it (wrapping at the box edges) and sum the force from every
 * neighbour within r_cut, using the minimum-image convention. The LJ force along
 * the separation r is  F = 24 * (2 (sigma/r)^12 - (sigma/r)^6) / r^2 * r_vec,
 * with sigma = epsilon = 1 in reduced units. Forces are summed independently per
 * atom (each pair is evaluated from both ends), which is simple and correct. */
static void md_forces(MD *s) {
    for (size_t i = 0; i < s->N; i++) { s->ax[i] = 0.0; s->ay[i] = 0.0; }
    for (int cy = 0; cy < s->nc; cy++) {
        for (int cx = 0; cx < s->nc; cx++) {
            for (int i = s->head[cy * s->nc + cx]; i >= 0; i = s->next[i]) {
                double xi = s->x[i], yi = s->y[i];
                double axi = 0.0, ayi = 0.0;
                /* scan this cell plus its 8 neighbours (periodic wrap) */
                for (int dy = -1; dy <= 1; dy++) {
                    int ny = (cy + dy + s->nc) % s->nc;
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = (cx + dx + s->nc) % s->nc;
                        for (int j = s->head[ny * s->nc + nx]; j >= 0; j = s->next[j]) {
                            if (j == i) continue;
                            /* minimum-image separation in the periodic box */
                            double rx = xi - s->x[j];
                            double ry = yi - s->y[j];
                            rx -= s->L * round(rx / s->L);
                            ry -= s->L * round(ry / s->L);
                            double r2 = rx * rx + ry * ry;
                            if (r2 >= s->rc2 || r2 == 0.0) continue;
                            double sr2 = 1.0 / r2;              /* (sigma/r)^2, sigma=1 */
                            double sr6 = sr2 * sr2 * sr2;
                            double sr12 = sr6 * sr6;
                            double f = 24.0 * (2.0 * sr12 - sr6) * sr2;   /* scalar / r */
                            axi += f * rx; ayi += f * ry;
                        }
                    }
                }
                s->ax[i] += axi; s->ay[i] += ayi;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nparts      = p2_get_i64(argc, argv, "--particles", 65536);
    /* floats passed as integer-milli (the phase2 arg helpers are integer-only) */
    long long dens_milli  = p2_get_i64(argc, argv, "--density-milli", 600);
    long long rc_milli    = p2_get_i64(argc, argv, "--rc-milli", 2500);
    long long dt_milli    = p2_get_i64(argc, argv, "--dt-milli", 2);
    long long duration_s  = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s    = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb      = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu         = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed        = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock    = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run     = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir    = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nparts < 64 || nparts > (1LL << 24)) { P2_LOG_ERR("particles %lld out of range (64..2^24)", nparts); return 2; }
    if (dens_milli < 10 || dens_milli > 2000) { P2_LOG_ERR("density-milli %lld out of range (10..2000)", dens_milli); return 2; }
    if (rc_milli < 500) { P2_LOG_ERR("rc-milli %lld too small (>=500)", rc_milli); return 2; }
    size_t N = (size_t)nparts;
    double density = (double)dens_milli / 1000.0;
    double rc = (double)rc_milli / 1000.0;
    double dt = (double)dt_milli / 1000.0;
    double L = sqrt((double)N / density);        /* box side for the target density */
    int nc = (int)(L / rc);                       /* cells per dimension */
    if (nc < 3) { P2_LOG_ERR("box too small for cell list (nc=%d < 3); raise --particles or lower --density-milli", nc); return 2; }
    double cell = L / (double)nc;

    size_t pbytes = 6 * N * sizeof(double);       /* x,y,vx,vy,ax,ay */
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
    p2_meta_kv_str(&m, "scheme", "Lennard-Jones molecular dynamics, velocity-Verlet + linked-cell list rebuilt each step");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "particles", nparts);
    p2_meta_kv_i64(&m, "density_milli", dens_milli);
    p2_meta_kv_i64(&m, "rc_milli", rc_milli);
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

    /* particle arrays: one mmap'd block, sliced structure-of-arrays */
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

    MD s = { N, L, rc, rc * rc, nc, cell,
             buf, buf + N, buf + 2 * N, buf + 3 * N, buf + 4 * N, buf + 5 * N,
             head, next };

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* lattice initialisation: place atoms on a regular grid (no overlaps) with
     * small random velocities -> a stable start for the integrator. */
    size_t g = (size_t)ceil(sqrt((double)N));     /* grid points per side */
    double spacing = L / (double)g;
    for (size_t i = 0; i < N; i++) {
        size_t gx = i % g, gy = i / g;
        s.x[i] = ((double)gx + 0.5) * spacing;
        s.y[i] = ((double)gy + 0.5) * spacing;
        s.vx[i] = 0.1 * (2.0 * rng_unit(&rng) - 1.0);
        s.vy[i] = 0.1 * (2.0 * rng_unit(&rng) - 1.0);
    }
    md_build_cells(&s);
    md_forces(&s);                                /* initial accelerations for step 1 */
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t steps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        for (size_t i = 0; i < N; i++) {                       /* (1) first half kick */
            s.vx[i] += 0.5 * s.ax[i] * dt;
            s.vy[i] += 0.5 * s.ay[i] * dt;
        }
        for (size_t i = 0; i < N; i++) {                       /* (2) drift + wrap */
            s.x[i] += s.vx[i] * dt;
            s.y[i] += s.vy[i] * dt;
            s.x[i] -= L * floor(s.x[i] / L);
            s.y[i] -= L * floor(s.y[i] / L);
        }
        md_build_cells(&s);                                    /* (3) rebuild cell list */
        md_forces(&s);                                         /* (4) new forces */
        for (size_t i = 0; i < N; i++) {                       /* (5) second half kick */
            s.vx[i] += 0.5 * s.ax[i] * dt;
            s.vy[i] += 0.5 * s.ay[i] * dt;
        }
        steps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = s.x[0];                             /* a live atom coordinate */

    free(head); free(next);
    munmap(buf, pbytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "steps", steps);
    p2_meta_kv_f64(&m, "atom0_x", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "cell list rebuilt every step (the distinct write); forces summed per-atom (each pair twice)");
    p2_meta_close(&m);
    return 0;
}
