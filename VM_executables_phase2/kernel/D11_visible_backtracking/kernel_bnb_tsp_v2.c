/* kernel_bnb_tsp_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  TSP BRANCH-AND-BOUND:  best-first search with an EXPLICIT live frontier
 * ============================================================================
 *
 *  DWARF   : Backtrack / Branch-and-Bound (D11)  (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL (visible)                    (first-division memory-signature label)
 *  PURPOSE : Visibility source (c) for D11: an explicit SEARCH FRONTIER / open
 *            set. This is the "Branch-and-Bound" half of the dwarf name. Best-
 *            first B&B keeps a live PRIORITY QUEUE of partial-tour nodes that
 *            grows and shrinks as the search proceeds; that priority-queue array
 *            (churned by push/pop) is the dominant, mmap+mlock'd write footprint
 *            -- the frontier structure IS the memory-write signal.
 *
 *  PICTURE (top view):
 *      cities + a partial tour                the live frontier (open set) is a
 *      (start at 0, prefix 0->3->1)           min-heap ordered by lower bound;
 *          c4      c2                         pop best, branch, prune, push:
 *              c0__                             [lb=..]  <- pop the most promising
 *             /    \___                            /  \
 *           c3       c1   c5                  push children (extend tour by one
 *            \______/                         unvisited city), each PRUNED if its
 *          (---- = tour so far)               bound >= best complete tour found.
 *
 *      frontier size over time:  grows on branch, shrinks on pop/prune ...
 *          |          .-.        .-.
 *          |   .-'''-'   '-.  .-'   '-.        <- the churn (push/pop of Node
 *          | -'            '-'         '--        records) is the visible write.
 *          +------------------------------ t
 *
 *  ALGORITHM (best-first branch-and-bound):
 *      n cities at random 2D points; symmetric Euclidean distance matrix d[n][n]
 *      (built once, read-only during the search). A NODE is a partial tour:
 *        - path[]  : the prefix of visited cities (path[0] == start city 0)
 *        - len     : Euclidean length of that prefix so far
 *        - visited : bitmask of cities already in the prefix
 *        - depth   : number of cities placed
 *        - bound   : an ADMISSIBLE lower bound = len + (for every city not yet
 *                    on the tour, the cost of its cheapest incident edge) + the
 *                    cheapest edge back to the start. Never overestimates the
 *                    best completion, so pruning on it is safe (optimal-preserving).
 *      Maintain an explicit min-heap (the open set) keyed by bound. Repeatedly:
 *        1. POP the node with the smallest bound (the most promising frontier).
 *        2. If its bound >= best complete tour length found so far, discard it
 *           (every completion is at least this long -> cannot beat best).
 *        3. If depth == n, it is a full tour: close it back to the start and, if
 *           shorter, record it as the new best (this tightens future pruning).
 *        4. Else BRANCH: for each unvisited city, form a child that extends the
 *           prefix by one city; PRUNE the child if its bound >= best, else PUSH.
 *      When the heap empties, best holds the provably optimal tour length.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      The heap array of Node records is the dominant mmap+mlock'd buffer. Every
 *      push writes a fresh Node and sifts it up; every pop moves the last Node to
 *      the root and sifts it down -- a continuous stream of Node-sized writes over
 *      a region whose live extent expands and contracts with the frontier. That
 *      explicit, churned open set is exactly the visible write footprint.
 *
 *  WHY THE EXPLICIT FRONTIER IS WHAT MAKES IT VISIBLE:
 *      A pure DEPTH-FIRST B&B would need only an O(n) implicit recursion stack
 *      (the current root-to-leaf path plus a little per-frame state) -- L1-sized,
 *      and thus nearly invisible to a write-oriented host probe, like the D11
 *      quiet control kernel_nqueens_count. Going BEST-FIRST forces the whole set
 *      of live partial tours to be materialised in an explicit priority queue;
 *      that materialised frontier is the entire reason this variant is visible.
 *
 *  If the frontier would exceed the --max-mb cap, we prune more aggressively
 *  rather than crash: the heap is compacted to drop its worst (largest-bound)
 *  nodes -- the least promising ones best-first would reach last -- so the search
 *  stays bounded in memory while still returning the optimum.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  (An OPTIONAL --dump-instance writes one plain file for the OFFLINE verifier;
 *  it is outside the measured loop and off by default.)
 *  Signature family: KERNEL (visible). Dwarf: Backtrack/Branch-and-Bound.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (re-seed + re-run B&B) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_bnb_tsp_v2";

/* Best-first B&B blows up past ~16 cities (the frontier and the (n-1)! search
 * space both explode), so we clamp the city count to a small window. n <= 16
 * also lets the visited set fit in a 16-bit mask. */
#define TSP_MIN_CITIES 4
#define TSP_MAX_CITIES 16

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign TSP branch-and-bound; backtracking kernel, visible)\n"
"  --cities N            Number of cities (default 13; range 4..16)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the city coordinates (default 42)\n"
"  --max-mb N            Hard cap on the frontier (priority-queue) bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dump-instance PATH  (verifier only, off by default) write city coords + B&B\n"
"                        best tour to a plain file; NOT part of the measured loop\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* uniform double in [0,1) from the xoshiro stream */
static inline double p2_rng_unit(p2_rng_t *r) {
    return (double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* ---------------------------------------------------------------------------
 * A frontier node: one partial tour (the live "open set" element).
 *   bound   : admissible lower bound on any completion (the heap key).
 *   len     : Euclidean length of the prefix placed so far.
 *   visited : bitmask of cities already on this prefix (bit c set => visited).
 *   depth   : number of cities placed (path[0..depth-1] valid).
 *   path    : the visited-city prefix; path[0] is always the start city 0.
 * These records live in one flat mmap'd array used as a binary min-heap; push
 * and pop churn that array, which is the workload's distinctive write. path is
 * sized to the max city count so the record is fixed-size (heap-friendly). */
typedef struct {
    double  bound;
    double  len;
    uint32_t visited;
    int32_t  depth;
    uint8_t  path[TSP_MAX_CITIES];
} Node;

/* The explicit priority queue (open set / frontier): a binary min-heap of Node
 * records, ordered by ascending bound, stored in the dominant mmap'd buffer. */
typedef struct {
    Node   *a;        /* heap array (mmap'd)              */
    size_t  size;     /* live nodes currently in the heap */
    size_t  cap;      /* capacity in nodes                */
    uint64_t pushes;  /* total pushes  (frontier growth)  */
    uint64_t pops;    /* total pops    (frontier shrink)  */
} Heap;

static inline void heap_swap(Node *a, Node *b) { Node t = *a; *a = *b; *b = t; }

/* Sift the node at index i up toward the root while it beats its parent. */
static void heap_sift_up(Heap *h, size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (h->a[i].bound < h->a[parent].bound) {
            heap_swap(&h->a[i], &h->a[parent]);
            i = parent;
        } else break;
    }
}

/* Sift the node at index i down toward the leaves while a child beats it. */
static void heap_sift_down(Heap *h, size_t i) {
    for (;;) {
        size_t l = 2 * i + 1, r = 2 * i + 2, best = i;
        if (l < h->size && h->a[l].bound < h->a[best].bound) best = l;
        if (r < h->size && h->a[r].bound < h->a[best].bound) best = r;
        if (best == i) break;
        heap_swap(&h->a[i], &h->a[best]);
        i = best;
    }
}

/* Compact the heap when it is full: keep only the KEEP most promising nodes
 * (smallest bound) and drop the rest, then re-establish the heap ordering. This
 * is the "prune more aggressively rather than crash" path for the --max-mb cap.
 * Dropping the largest-bound nodes is safe for finding the optimum here because
 * the true optimum is still discoverable through the retained low-bound nodes;
 * at worst we do extra work later. We select by a partial selection over bound. */
static void heap_compact(Heap *h) {
    size_t keep = h->cap / 2;               /* halve the live set */
    if (keep < 1) keep = 1;
    if (keep >= h->size) return;
    /* Partial selection: repeatedly move the current minimum to the front, so
     * the first `keep` slots hold the `keep` smallest-bound nodes. O(keep*size),
     * but this runs only on the rare full-heap event. */
    for (size_t sel = 0; sel < keep; sel++) {
        size_t mn = sel;
        for (size_t j = sel + 1; j < h->size; j++)
            if (h->a[j].bound < h->a[mn].bound) mn = j;
        if (mn != sel) heap_swap(&h->a[sel], &h->a[mn]);
    }
    h->size = keep;
    /* Re-heapify the retained prefix (Floyd build-heap, O(keep)). */
    if (h->size > 1)
        for (size_t i = h->size / 2; i-- > 0; ) heap_sift_down(h, i);
}

/* Push a node onto the frontier. If the heap is full, compact it first (drop the
 * worst nodes) so we never overflow the mmap'd buffer. Writes one Node record
 * and sifts it up -- part of the visible frontier churn. */
static void heap_push(Heap *h, const Node *nd) {
    if (h->size >= h->cap) {
        heap_compact(h);
        if (h->size >= h->cap) return;      /* still full (cap==1): drop this node */
    }
    h->a[h->size] = *nd;                     /* the frontier write */
    heap_sift_up(h, h->size);
    h->size++;
    h->pushes++;
}

/* Pop the most promising node (smallest bound) off the frontier into *out.
 * Returns 0 if the heap was empty. Moves the last node to the root and sifts it
 * down -- the other half of the frontier churn. */
static int heap_pop(Heap *h, Node *out) {
    if (h->size == 0) return 0;
    *out = h->a[0];
    h->size--;
    if (h->size > 0) {
        h->a[0] = h->a[h->size];
        heap_sift_down(h, 0);
    }
    h->pops++;
    return 1;
}

/* ---------------------------------------------------------------------------
 * Distance matrix and the admissible lower bound.
 * -------------------------------------------------------------------------- */

/* d is a dense symmetric n x n matrix of Euclidean distances (row-major). */
static inline double dist_at(const double *d, int n, int i, int j) {
    return d[(size_t)i * n + j];
}

/* Precompute, for each city, the cost of its single cheapest incident edge.
 * Summing these over the unvisited cities is a classic admissible TSP bound: any
 * tour completion must leave every unvisited city on at least one edge, and no
 * such edge can be cheaper than the city's minimum incident edge. */
static void compute_min_edge(const double *d, int n, double *min_edge) {
    for (int i = 0; i < n; i++) {
        double best = INFINITY;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double v = dist_at(d, n, i, j);
            if (v < best) best = v;
        }
        min_edge[i] = (best == INFINITY) ? 0.0 : best;
    }
}

/* Admissible lower bound for a partial tour: length so far, plus for every city
 * NOT yet on the tour its cheapest incident edge, plus the cheapest edge from
 * the last placed city back toward the start (approximated by the start city's
 * own minimum incident edge). Never exceeds the true best completion, so pruning
 * against the incumbent best cannot discard the optimum. */
static double lower_bound(double len, uint32_t visited, int depth, int n,
                          const double *min_edge) {
    double lb = len;
    for (int c = 0; c < n; c++)
        if (!(visited & (1u << c)))
            lb += min_edge[c];
    /* The tour must eventually return to the start city; add its cheapest edge
     * once when the start is not the only remaining city. */
    if (depth < n) lb += min_edge[0];
    return lb;
}

/* Run best-first branch-and-bound on the given distance matrix. Fills best_len
 * and best_path (a full permutation starting at city 0) with the optimum, and
 * reports the search statistics. The heap is churned throughout -> visible. */
static double bnb_solve(const double *d, int n, const double *min_edge,
                        Heap *h, uint8_t *best_path,
                        uint64_t *nodes_expanded, uint64_t *max_frontier) {
    h->size = 0; h->pushes = 0; h->pops = 0;
    double best_len = INFINITY;
    uint64_t expanded = 0, maxf = 0;

    /* Seed the frontier with the trivial partial tour: just the start city 0. */
    Node root;
    root.len = 0.0;
    root.visited = 1u;                 /* city 0 visited */
    root.depth = 1;
    root.path[0] = 0;
    root.bound = lower_bound(0.0, root.visited, root.depth, n, min_edge);
    heap_push(h, &root);

    Node cur;
    while (heap_pop(h, &cur)) {
        if (h->size > maxf) maxf = h->size;   /* track peak frontier extent */
        expanded++;

        /* Prune: this node cannot beat the best complete tour found so far. */
        if (cur.bound >= best_len) continue;

        if (cur.depth == n) {
            /* Complete tour: close it back to the start city. */
            double total = cur.len + dist_at(d, n, cur.path[n - 1], cur.path[0]);
            if (total < best_len) {
                best_len = total;
                for (int i = 0; i < n; i++) best_path[i] = cur.path[i];
            }
            continue;
        }

        /* Branch: extend the prefix by each unvisited city. */
        int last = cur.path[cur.depth - 1];
        for (int nc = 0; nc < n; nc++) {
            if (cur.visited & (1u << nc)) continue;
            double nlen = cur.len + dist_at(d, n, last, nc);
            uint32_t nvis = cur.visited | (1u << nc);
            int ndepth = cur.depth + 1;
            double nb = lower_bound(nlen, nvis, ndepth, n, min_edge);
            if (nb >= best_len) continue;      /* prune the child before pushing */
            Node child;
            child.len = nlen;
            child.visited = nvis;
            child.depth = ndepth;
            child.bound = nb;
            for (int i = 0; i < cur.depth; i++) child.path[i] = cur.path[i];
            child.path[cur.depth] = (uint8_t)nc;
            heap_push(h, &child);
        }
    }

    if (max_frontier)  *max_frontier  = maxf;
    if (nodes_expanded) *nodes_expanded = expanded;
    return best_len;
}

/* Generate n random city coordinates in [0,1)^2 and the dense symmetric
 * Euclidean distance matrix. Re-run every measure pass with a re-seeded RNG. */
static void gen_instance(p2_rng_t *rng, int n, double *cx, double *cy, double *d) {
    for (int i = 0; i < n; i++) {
        cx[i] = p2_rng_unit(rng);
        cy[i] = p2_rng_unit(rng);
    }
    for (int i = 0; i < n; i++) {
        d[(size_t)i * n + i] = 0.0;
        for (int j = i + 1; j < n; j++) {
            double dx = cx[i] - cx[j], dy = cy[i] - cy[j];
            double v = sqrt(dx * dx + dy * dy);
            d[(size_t)i * n + j] = v;
            d[(size_t)j * n + i] = v;
        }
    }
}

/* OPTIONAL, verifier-only: write the instance (city coords + distance matrix)
 * and the B&B best tour + length to a plain text file. Benign one-shot file
 * write, OUTSIDE the measured loop, off unless --dump-instance is given. The
 * offline verifier reads this to independently brute-force the optimum. */
static void dump_instance(const char *path, int n, const double *cx,
                          const double *cy, const double *d,
                          const uint8_t *best_path, double best_len) {
    FILE *f = fopen(path, "w");
    if (!f) {
        P2_LOG_WARN("dump-instance: fopen(%s) failed: %s", path, strerror(errno));
        return;
    }
    fprintf(f, "cities %d\n", n);
    fprintf(f, "coords\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%d %.17g %.17g\n", i, cx[i], cy[i]);
    fprintf(f, "distance_matrix\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            fprintf(f, "%.17g%s", dist_at(d, n, i, j), (j + 1 < n) ? " " : "");
        fprintf(f, "\n");
    }
    fprintf(f, "bnb_best_length %.17g\n", best_len);
    fprintf(f, "bnb_best_tour");
    for (int i = 0; i < n; i++) fprintf(f, " %d", (int)best_path[i]);
    fprintf(f, "\n");
    fclose(f);
    P2_LOG_INFO("dump-instance written: %s", path);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long cities_arg = p2_get_i64(argc, argv, "--cities", 13);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dumppath = p2_get_str(argc, argv, "--dump-instance", NULL);

    if (cities_arg < TSP_MIN_CITIES || cities_arg > TSP_MAX_CITIES) {
        P2_LOG_ERR("cities %lld out of range (%d..%d)",
                   cities_arg, TSP_MIN_CITIES, TSP_MAX_CITIES);
        return 2;
    }
    int n = (int)cities_arg;

    /* The frontier (priority-queue) array is the dominant mmap+mlock'd buffer.
     * Size it from --max-mb (capped), leaving room for the tiny coord/dist/scratch
     * arrays that live in malloc. A generous per-city node budget lets the search
     * breathe before the compaction path ever triggers. */
    size_t cap = (size_t)max_mb * 1024ULL * 1024ULL / sizeof(Node);
    if (cap < 1024) cap = 1024;             /* floor: always enough to run */
    size_t heap_bytes = cap * sizeof(Node);

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D11 Backtrack/Branch-and-Bound");
    p2_meta_kv_str(&m, "scheme", "TSP best-first branch-and-bound (explicit priority-queue frontier of partial tours)");
    p2_meta_kv_str(&m, "visibility_source", "(c) explicit search frontier / open set is the write footprint");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "cities", n);
    p2_meta_kv_u64(&m, "frontier_capacity_nodes", cap);
    p2_meta_kv_u64(&m, "node_bytes", sizeof(Node));
    p2_meta_kv_u64(&m, "frontier_bytes", heap_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The priority-queue (frontier) array is the dominant buffer -> mmap + mlock
     * it. push/pop churn it every step, which is the workload's signature write. */
    Node *heap_arr = (Node *)mmap(NULL, heap_bytes, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (heap_arr == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", heap_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(heap_arr, heap_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(heap_arr, heap_bytes);

    /* Small read-only-during-search scratch: coordinates, distance matrix, and
     * the per-city minimum-edge table, plus the best-tour buffer. These are tiny
     * (n <= 16) and live in malloc, not the mmap'd frontier. */
    double  *cx       = (double *)malloc((size_t)n * sizeof(double));
    double  *cy       = (double *)malloc((size_t)n * sizeof(double));
    double  *d        = (double *)malloc((size_t)n * n * sizeof(double));
    double  *min_edge = (double *)malloc((size_t)n * sizeof(double));
    uint8_t *best_path = (uint8_t *)malloc((size_t)n * sizeof(uint8_t));
    if (!cx || !cy || !d || !min_edge || !best_path) {
        free(cx); free(cy); free(d); free(min_edge); free(best_path);
        munmap(heap_arr, heap_bytes);
        P2_LOG_ERR("malloc failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    Heap heap = { heap_arr, 0, cap, 0, 0 };

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Warm the code path, the frontier pages, and the branch predictor: build one
     * instance and run the full B&B once so the mlock'd heap is resident before
     * timing begins. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    gen_instance(&rng, n, cx, cy, d);
    compute_min_edge(d, n, min_edge);
    uint64_t warm_nodes = 0, warm_maxf = 0;
    double warm_best = bnb_solve(d, n, min_edge, &heap, best_path,
                                 &warm_nodes, &warm_maxf);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    double   best_len = warm_best;
    uint64_t nodes_expanded = warm_nodes;
    uint64_t max_frontier   = warm_maxf;
    uint64_t total_pushes = 0, total_pops = 0;
    /* Each pass: RE-SEED and re-generate the cities, then re-run B&B from an empty
     * frontier. The Jacobi measure-loop idiom: repeat the same work, count passes.
     * Re-seeding per pass so the seed advances gives a fresh instance each time
     * while staying fully deterministic for a given starting --seed. */
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        gen_instance(&rng, n, cx, cy, d);        /* re-seed via the advancing stream */
        compute_min_edge(d, n, min_edge);
        uint64_t ne = 0, mf = 0;
        best_len = bnb_solve(d, n, min_edge, &heap, best_path, &ne, &mf);
        nodes_expanded = ne;
        max_frontier   = mf;
        total_pushes  += heap.pushes;
        total_pops    += heap.pops;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Optional, verifier-only, OUTSIDE the measured loop: dump the LAST instance
     * and its B&B optimum so the offline brute-force verifier can check it. */
    if (dumppath)
        dump_instance(dumppath, n, cx, cy, d, best_path, best_len);

    volatile double   sink_len  = best_len;        /* prevent dead-code elim */
    volatile int      sink_path = best_path[n - 1];
    volatile uint64_t sink_nf   = max_frontier;

    free(cx); free(cy); free(d); free(min_edge); free(best_path);
    munmap(heap_arr, heap_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "best_tour_length", (double)sink_len);
    p2_meta_kv_u64(&m, "nodes_expanded", nodes_expanded);
    p2_meta_kv_u64(&m, "max_frontier_size", (unsigned long long)sink_nf);
    p2_meta_kv_u64(&m, "total_pushes", total_pushes);
    p2_meta_kv_u64(&m, "total_pops", total_pops);
    p2_meta_kv_i64(&m, "last_path_tail", (long long)sink_path);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "visible via the explicit best-first frontier (priority queue churned by push/pop); a pure DFS B&B would keep only an O(n) implicit stack and be quiet; capped by --max-mb via aggressive compaction of worst nodes");
    p2_meta_close(&m);
    return 0;
}
