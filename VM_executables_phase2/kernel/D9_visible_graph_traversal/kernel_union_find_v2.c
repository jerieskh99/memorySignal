/* kernel_union_find_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Graph Traversal dwarf (Berkeley motif D9), the connected-components variant
 * built on a UNION-FIND (disjoint-set) forest. Given a stream of undirected
 * edges (u,v), union-find answers "are u and v in the same component?" by
 * maintaining, for every node, a PARENT POINTER into a forest of trees; all
 * nodes in one tree share one root, and two nodes are connected iff they share
 * a root. Processing an edge means union(u,v): find each endpoint's root and,
 * if they differ, link one tree under the other.
 *
 * WHY IT IS A DISTINCT (WRITE-VISIBLE) MEMORY SIGNATURE
 * ----------------------------------------------------------------------------
 * A plain graph traversal (e.g. BFS over a fixed adjacency, see kernel_bfs_v2)
 * READS a large structure and writes almost nothing back -- invisible to a host
 * memory WRITE-signal. Union-find is the opposite. The dominant buffer here is
 * parent[], and it is REWRITTEN continuously in two ways:
 *   (a) union() overwrites a root's parent to link two trees, and
 *   (b) find() PATH-COMPRESSES: after walking a node up to its root, it rewrites
 *       the parent of EVERY node on that path to point straight at the root, so
 *       the next query is shorter.
 * Path compression is a pointer-chasing READ (follow parent up) immediately
 * followed by a scattered WRITE-back over that same path. Those writes land at
 * data-dependent, non-sequential indices of parent[] -- an irregular write
 * stream that a write-signal can see, distinct from read-only traversal. That
 * is the tell this kernel is built to expose.
 *
 * PICTURE (top view):  find(a) walks to root r, then compresses the path.
 *
 *      before find(a):           parent chain a -> b -> c -> r (r is its own root)
 *
 *        parent : [ .. a:b .. b:c .. c:r .. r:r .. ]   (READS follow the chain up)
 *
 *      after find(a):            every node on the path now points DIRECTLY at r
 *
 *        parent : [ .. a:r .. b:r .. c:r .. r:r .. ]   <-- the scattered WRITE-back
 *                        ^^^      ^^^      ^^^
 *                        these overwrites are the write-visible signature
 *
 * ALGORITHM (per measured pass):
 *   1. Reset the forest: parent[i] = i and rank[i] = 0 for all i (N singletons).
 *      This full O(N) rewrite of parent[] is itself a large visible write.
 *   2. Process E random edges. For each, pick u,v uniformly in [0,N) and call
 *      union(u,v):
 *        - find(u) and find(v) locate the two roots, path-compressing en route;
 *        - if the roots differ, link by RANK (attach the shorter tree under the
 *          taller; equal ranks -> attach either way and bump the winner's rank).
 *          Union by rank keeps trees shallow so find() stays near-constant time.
 *   3. Count passes; at the end of the pass, count DISTINCT ROOTS (nodes with
 *      parent[i] == i) -- that is the number of connected components implied by
 *      the E edges seen this pass.
 *
 * Because parent[] is reset and then rewritten by unions + compression every
 * pass, the measured phase is a sustained, irregular write workload over the
 * dominant buffer -- the intended write-visible behaviour.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Graph Traversal. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + first forest reset) / measure (edge unions) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_union_find_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign union-find connected components; Graph-Traversal kernel)\n"
"  --nodes N             Number of set elements / forest nodes (default 2000000)\n"
"  --edges E             Random edges unioned per pass (default 4000000)\n"
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

/* ---------------------------------------------------------------------------
 * The disjoint-set forest lives in two parallel mmap'd arrays:
 *   parent[i] : index of i's parent in the forest; parent[i]==i marks a root.
 *               This is the DOMINANT buffer and the one path compression and
 *               union() rewrite -- the write-visible structure.
 *   rank[i]   : an upper bound on the height of the tree rooted at i, used only
 *               to decide which tree links under which (union by rank). Only a
 *               root's rank matters; it is bumped at most when two equal-rank
 *               trees merge.
 * --------------------------------------------------------------------------- */

/* find(x): walk parent pointers up to the root, then PATH-COMPRESS -- rewrite
 * the parent of every node on the path to point straight at the root. The
 * upward walk is a pointer-chasing read; the rewrite is the scattered write.
 * Done in two passes (find root, then relink) to keep it iterative -- no
 * recursion, so deep chains cannot overflow the stack. */
static inline int32_t uf_find(int32_t *parent, int32_t x) {
    int32_t r = x;
    while (parent[r] != r) r = parent[r];      /* pass 1: climb to the root (reads) */
    while (parent[x] != r) {                    /* pass 2: point the whole path at r */
        int32_t next = parent[x];
        parent[x] = r;                          /* the scattered write-back */
        x = next;
    }
    return r;
}

/* union(u,v): find both roots; if distinct, link the shorter tree under the
 * taller (union by rank). Equal ranks -> attach v-root under u-root and bump
 * u-root's rank. Each merge overwrites exactly one root's parent entry. */
static inline void uf_union(int32_t *parent, int32_t *rank, int32_t u, int32_t v) {
    int32_t ru = uf_find(parent, u);
    int32_t rv = uf_find(parent, v);
    if (ru == rv) return;                       /* already in the same component */
    if (rank[ru] < rank[rv]) {
        parent[ru] = rv;                        /* attach shorter (ru) under rv */
    } else if (rank[ru] > rank[rv]) {
        parent[rv] = ru;                        /* attach shorter (rv) under ru */
    } else {
        parent[rv] = ru;                        /* equal -> attach rv under ru */
        rank[ru]++;                             /* ...and the merged tree grew by one */
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nodes      = p2_get_i64(argc, argv, "--nodes", 2000000);
    long long edges      = p2_get_i64(argc, argv, "--edges", 4000000);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nodes < 1000 || nodes > 2000000000LL) { P2_LOG_ERR("nodes %lld out of range (1000..2e9)", nodes); return 2; }
    if (edges < 1 || edges > 20000000000LL) { P2_LOG_ERR("edges %lld out of range (1..2e10)", edges); return 2; }
    size_t N = (size_t)nodes, E = (size_t)edges;
    /* parent[N] + rank[N], both int32 -> 2*N*4 bytes; this is the whole footprint. */
    size_t bytes = 2 * N * sizeof(int32_t);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Graph Traversal");
    p2_meta_kv_str(&m, "scheme", "union-find connected components (path compression + union by rank rewrite parent[])");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "nodes", nodes);
    p2_meta_kv_i64(&m, "edges", edges);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* parent[] is the dominant buffer (rewritten by every union + compression)
     * -> mmap + advise + mlock it. rank[] rides alongside in its own mapping. */
    int32_t *parent = (int32_t *)mmap(NULL, N * sizeof(int32_t), PROT_READ | PROT_WRITE,
                                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (parent == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", N * sizeof(int32_t), strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    int32_t *rank = (int32_t *)mmap(NULL, N * sizeof(int32_t), PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (rank == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", N * sizeof(int32_t), strerror(errno));
        munmap(parent, N * sizeof(int32_t));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(parent, N * sizeof(int32_t), MADV_NOHUGEPAGE);
    p2_madvise(rank,   N * sizeof(int32_t), MADV_NOHUGEPAGE);
    if (!no_mlock) { p2_mlock_soft(parent, N * sizeof(int32_t)); p2_mlock_soft(rank, N * sizeof(int32_t)); }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* First forest reset during warmup: every element is its own singleton set.
     * This touches (and faults in) the whole parent[]/rank[] mapping before we
     * start timing unions. */
    for (size_t i = 0; i < N; i++) { parent[i] = (int32_t)i; rank[i] = 0; }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t last_components = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed the PRNG each pass so every pass unions the SAME edge stream
         * (deterministic work per pass), matching the SpMV/Barnes-Hut re-seed
         * pattern. The count-loop is the timed body. */
        p2_rng_seed(&rng, seed);
        /* (1) Reset the forest to N singletons -- a full O(N) rewrite of the
         * dominant buffer, itself a large sequential write each pass. */
        for (size_t i = 0; i < N; i++) { parent[i] = (int32_t)i; rank[i] = 0; }
        /* (2) Union E random edges. Each union() path-compresses inside find()
         * (scattered write-back over parent[]) and, on a real merge, overwrites
         * one root's parent -- the sustained irregular write signature. */
        for (size_t e = 0; e < E; e++) {
            int32_t u = (int32_t)(p2_rng_next(&rng) % (uint64_t)N);
            int32_t v = (int32_t)(p2_rng_next(&rng) % (uint64_t)N);
            uf_union(parent, rank, u, v);
        }
        /* (3) Count distinct roots = connected components implied this pass. */
        uint64_t components = 0;
        for (size_t i = 0; i < N; i++) if (parent[i] == (int32_t)i) components++;
        last_components = components;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile int32_t sink = parent[N / 2];              /* a live parent pointer */

    munmap(parent, N * sizeof(int32_t));
    munmap(rank,   N * sizeof(int32_t));

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "components_last_pass", last_components);
    p2_meta_kv_f64(&m, "sink", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "path compression + union rewrites of parent[] are the distinct write vs read-only traversal; edges are random self-loops/dups possible");
    p2_meta_close(&m);
    return 0;
}
