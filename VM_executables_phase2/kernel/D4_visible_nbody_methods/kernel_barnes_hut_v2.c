/* kernel_barnes_hut_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * N-Body dwarf (Berkeley motif D4), the Barnes-Hut variant. A plain N-body
 * simulation moves N particles that all attract each other by gravity; the naive
 * force loop is O(N^2) because every particle looks at every other. Barnes-Hut
 * cuts that to O(N log N) with a spatial QUADTREE: a clump of far-away particles
 * is summarised by a single node (its total mass + centre of mass), so a distant
 * clump is read once instead of particle-by-particle.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs kernel_nbody_v2)
 * ----------------------------------------------------------------------------
 * Every N-body method WRITES the same thing each step: the particle position and
 * velocity arrays. The methods differ only in how they READ to get the force
 * (all-pairs reads everyone; Barnes-Hut reads a tree). Reads are invisible to a
 * host write-signal, so those variants would look identical -- EXCEPT that
 * Barnes-Hut also REBUILDS A TREE every step. That tree (a pool of nodes written
 * from scratch each timestep) is an extra, irregular write structure that the
 * plain particle-only nbody does not have. That is the distinct tell.
 *
 * ============================================================================
 * ALGORITHM (per timestep)
 * ============================================================================
 *   1. Bounding box: find the square that contains all particles.
 *   2. Build the quadtree: insert particles one by one; a cell holding two
 *      bodies subdivides into four child cells, recursively. Each node keeps its
 *      total mass and centre of mass (COM).
 *   3. Forces: for each particle, walk the tree from the root. If a node is far
 *      enough away that its size/distance < theta, use its COM as ONE mass;
 *      otherwise open it and recurse into its children. (theta = 0 opens every
 *      node down to single particles -> exact O(N^2); larger theta = faster,
 *      approximate. We exploit theta = 0 for the correctness test.)
 *   4. Integrate: v += a*dt; x += v*dt (semi-implicit Euler). Positions evolve
 *      SMOOTHLY (no re-seeding), like a real simulation.
 *
 * Softening (soft^2 added to the squared distance) keeps the force finite when
 * two particles are very close, so the simulation stays numerically stable.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: N-Body Methods. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (timesteps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_barnes_hut_v2";

/* Hard cap on tree depth: bounds recursion and the node pool for pathological
 * (highly clustered) inputs. At this depth we stop subdividing and let a cell
 * hold several bodies as one aggregate mass (a small, softened approximation). */
#define MAX_DEPTH 40

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Barnes-Hut N-body; N-Body kernel)\n"
"  --particles N         Number of bodies (default 16384)\n"
"  --theta-milli T       Opening angle x1000; 0 = exact O(n^2), 500-1000 = fast (default 700 = 0.7)\n"
"  --dt-milli DT         Timestep x1000 (default 10 = 0.01)\n"
"  --soft-milli S        Softening length x1000 (default 50 = 0.05)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on node-pool bytes (default 8192)\n"
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
 * Quadtree node. One cell of 2D space.
 *   cx, cy, half : cell centre and half-width (the cell spans [cx-half, cx+half]).
 *   comx, comy   : centre of mass of all bodies under this cell.
 *   mass         : total mass under this cell.
 *   child[4]     : indices of the four quadrant children (-1 = not created).
 *                  quadrant bit 0 = +x (right), bit 1 = +y (top).
 *   body         : particle index if this is a single-body leaf, else -1.
 *   count        : number of bodies under this cell.
 * Nodes live in one flat pool (an mmap'd array) that is reset and refilled each
 * timestep -- that rebuild is the workload's distinctive write.
 * --------------------------------------------------------------------------- */
typedef struct {
    double cx, cy, half;
    double comx, comy, mass;
    int child[4];
    int body;
    int count;
} BHNode;

/* Build context: the node pool plus a read-only view of the particle arrays. */
typedef struct {
    BHNode *pool;        /* flat array of nodes (mmap'd)   */
    int     npool;       /* nodes used so far this build   */
    int     cap;         /* pool capacity in nodes         */
    int     overflow;    /* set if we ran out of nodes     */
    const double *x, *y, *m;
} BH;

/* Allocate a fresh, empty cell from the pool. Returns its index, or -1 if the
 * pool is exhausted (then subdivision is skipped -- a safe approximation). */
static int bh_new_node(BH *b, double cx, double cy, double half) {
    if (b->npool >= b->cap) { b->overflow = 1; return -1; }
    int i = b->npool++;
    BHNode *n = &b->pool[i];
    n->cx = cx; n->cy = cy; n->half = half;
    n->comx = 0.0; n->comy = 0.0; n->mass = 0.0;
    n->child[0] = n->child[1] = n->child[2] = n->child[3] = -1;
    n->body = -1; n->count = 0;
    return i;
}

/* Which of the four child quadrants of cell nd does point (px,py) fall in? */
static inline int bh_quadrant(const BHNode *nd, double px, double py) {
    return (px >= nd->cx ? 1 : 0) | (py >= nd->cy ? 2 : 0);
}

/* Return the child node index for quadrant q of node ni, creating it on demand.
 * The pool is a fixed mmap'd array, so &b->pool[ni] stays valid across the
 * allocation (bh_new_node never moves existing nodes). */
static int bh_get_child(BH *b, int ni, int q) {
    if (b->pool[ni].child[q] < 0) {
        double h  = b->pool[ni].half * 0.5;
        double cx = b->pool[ni].cx + ((q & 1) ? h : -h);
        double cy = b->pool[ni].cy + ((q & 2) ? h : -h);
        b->pool[ni].child[q] = bh_new_node(b, cx, cy, h);   /* may be -1 on overflow */
    }
    return b->pool[ni].child[q];
}

/* Insert particle p into the subtree rooted at node ni (at the given depth). */
static void bh_insert(BH *b, int ni, int p, int depth) {
    BHNode *nd = &b->pool[ni];

    /* Case 1: empty cell -> becomes a single-body leaf holding p. */
    if (nd->count == 0) {
        nd->body = p;
        nd->comx = b->x[p]; nd->comy = b->y[p]; nd->mass = b->m[p];
        nd->count = 1;
        return;
    }

    /* Case 2: cell already holds exactly one body q -> it must subdivide.
     * Push q down into its quadrant child first; the cell becomes internal. */
    if (nd->body >= 0) {
        int q = nd->body;
        nd->body = -1;                         /* no longer a single-body leaf */
        if (depth < MAX_DEPTH) {
            int qc = bh_get_child(b, ni, bh_quadrant(&b->pool[ni], b->x[q], b->y[q]));
            if (qc >= 0) bh_insert(b, qc, q, depth + 1);
        }
        /* if we could not subdivide, q survives only in this cell's aggregate */
    }

    /* Now the cell is internal (or a depth-capped bucket): fold p into the
     * running centre of mass and mass, then recurse p into its quadrant. */
    nd = &b->pool[ni];                         /* re-fetch (pool base is fixed) */
    double mp = b->m[p];
    double mtot = nd->mass + mp;
    nd->comx = (nd->comx * nd->mass + b->x[p] * mp) / mtot;
    nd->comy = (nd->comy * nd->mass + b->y[p] * mp) / mtot;
    nd->mass = mtot;
    nd->count++;
    if (depth < MAX_DEPTH) {
        int pc = bh_get_child(b, ni, bh_quadrant(&b->pool[ni], b->x[p], b->y[p]));
        if (pc >= 0) bh_insert(b, pc, p, depth + 1);
    }
}

/* Accumulate the gravitational acceleration on particle p from the subtree ni.
 *   theta2 = theta*theta ; soft2 = soft*soft ; G = gravitational constant.
 * Opening criterion (compared squared to avoid a sqrt): use a node as one mass
 * when it is a leaf OR when (cell_width / distance)^2 < theta^2. */
static void bh_accel(const BH *b, int ni, int p,
                     double theta2, double soft2, double G,
                     double *ax, double *ay) {
    const BHNode *nd = &b->pool[ni];
    if (nd->count == 0) return;
    if (nd->body == p && nd->count == 1) return;    /* p's own leaf: skip self */

    double dx = nd->comx - b->x[p];
    double dy = nd->comy - b->y[p];
    double d2 = dx * dx + dy * dy + soft2;

    int is_leaf = (nd->child[0] < 0 && nd->child[1] < 0 &&
                   nd->child[2] < 0 && nd->child[3] < 0);
    double s = 2.0 * nd->half;                      /* cell width */

    if (is_leaf || s * s < theta2 * d2) {
        /* Treat the whole cell as one mass at its centre of mass. */
        double d = sqrt(d2);
        double a = G * nd->mass / (d2 * d);         /* |a| = G m / d^2, dir = (dx,dy)/d */
        *ax += a * dx; *ay += a * dy;
    } else {
        /* Too close/large: open the node and sum over its children. */
        for (int c = 0; c < 4; c++)
            if (nd->child[c] >= 0)
                bh_accel(b, nd->child[c], p, theta2, soft2, G, ax, ay);
    }
}

/* Rebuild the whole tree from the current particle positions. */
static void bh_build(BH *b, size_t N) {
    b->npool = 0; b->overflow = 0;
    /* bounding box of all particles */
    double minx = b->x[0], maxx = b->x[0], miny = b->y[0], maxy = b->y[0];
    for (size_t i = 1; i < N; i++) {
        if (b->x[i] < minx) minx = b->x[i];
        if (b->x[i] > maxx) maxx = b->x[i];
        if (b->y[i] < miny) miny = b->y[i];
        if (b->y[i] > maxy) maxy = b->y[i];
    }
    double cx = 0.5 * (minx + maxx), cy = 0.5 * (miny + maxy);
    double half = 0.5 * fmax(maxx - minx, maxy - miny);
    if (half <= 0.0) half = 1.0;
    half *= 1.0001;                                 /* pad so all bodies are inside */
    int root = bh_new_node(b, cx, cy, half);
    if (root < 0) return;
    for (size_t i = 0; i < N; i++) bh_insert(b, root, (int)i, 0);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nparts     = p2_get_i64(argc, argv, "--particles", 16384);
    /* Floating-point options are passed as integer-milli, because the phase2 arg
     * helpers are integer-only: --theta-milli 700 = 0.7, --dt-milli 10 = 0.01. */
    long long theta_milli = p2_get_i64(argc, argv, "--theta-milli", 700);
    long long dt_milli    = p2_get_i64(argc, argv, "--dt-milli", 10);
    long long soft_milli  = p2_get_i64(argc, argv, "--soft-milli", 50);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    double theta = (double)theta_milli / 1000.0;
    double dt    = (double)dt_milli / 1000.0;
    double soft  = (double)soft_milli / 1000.0;

    if (nparts < 16 || nparts > (1LL << 22)) { P2_LOG_ERR("particles %lld out of range (16..2^22)", nparts); return 2; }
    if (theta < 0.0 || theta > 4.0) { P2_LOG_ERR("theta %.3f out of range (0..4)", theta); return 2; }
    size_t N = (size_t)nparts;
    size_t cap = (size_t)N * 8 + 64;                /* node pool capacity */
    size_t bytes = cap * sizeof(BHNode);            /* the tree pool dominates the footprint */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("node-pool bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "N-Body Methods");
    p2_meta_kv_str(&m, "scheme", "Barnes-Hut quadtree N-body (tree rebuilt each step + particle integrate)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "particles", nparts);
    p2_meta_kv_i64(&m, "theta_milli", (long long)(theta * 1000.0 + 0.5));
    p2_meta_kv_i64(&m, "dt_milli", dt_milli);
    p2_meta_kv_i64(&m, "soft_milli", soft_milli);
    p2_meta_kv_u64(&m, "node_pool_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The node pool is the dominant buffer -> mmap + mlock it (it is rebuilt
     * every step, which is the workload's signature write). */
    BHNode *pool = (BHNode *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (pool == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(pool, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(pool, bytes);

    /* Particle state (compact arrays; these evolve smoothly across steps). */
    double *x  = (double *)malloc(N * sizeof(double));
    double *y  = (double *)malloc(N * sizeof(double));
    double *vx = (double *)malloc(N * sizeof(double));
    double *vy = (double *)malloc(N * sizeof(double));
    double *ms = (double *)malloc(N * sizeof(double));
    if (!x || !y || !vx || !vy || !ms) { free(x); free(y); free(vx); free(vy); free(ms);
        munmap(pool, bytes); P2_LOG_ERR("malloc failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    BH b = { pool, 0, (int)cap, 0, x, y, ms };
    const double G = 1.0, theta2 = theta * theta, soft2 = soft * soft;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) {                /* random cloud in [-1,1]^2 */
        x[i]  = 2.0 * rng_unit(&rng) - 1.0;
        y[i]  = 2.0 * rng_unit(&rng) - 1.0;
        vx[i] = 0.0; vy[i] = 0.0;
        ms[i] = 0.5 + rng_unit(&rng);               /* mass in [0.5, 1.5) */
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t steps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        bh_build(&b, N);                            /* (1)+(2) rebuild the tree */
        int root = 0;                               /* root is always node 0 */
        for (size_t i = 0; i < N; i++) {            /* (3) forces via tree walk */
            double ax = 0.0, ay = 0.0;
            bh_accel(&b, root, (int)i, theta2, soft2, G, &ax, &ay);
            vx[i] += ax * dt; vy[i] += ay * dt;     /* (4a) kick velocities */
        }
        for (size_t i = 0; i < N; i++) {            /* (4b) drift positions */
            x[i] += vx[i] * dt; y[i] += vy[i] * dt;
        }
        steps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = x[0];                    /* a live particle coordinate */

    free(x); free(y); free(vx); free(vy); free(ms);
    munmap(pool, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "steps", steps);
    p2_meta_kv_i64(&m, "tree_overflowed", b.overflow);
    p2_meta_kv_f64(&m, "particle0_x", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "the rebuilt tree is the distinct write vs plain nbody; theta trades accuracy for speed (0 = exact)");
    p2_meta_close(&m);
    return 0;
}
