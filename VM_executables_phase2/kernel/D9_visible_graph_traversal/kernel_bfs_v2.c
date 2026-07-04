/* kernel_bfs_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  BREADTH-FIRST SEARCH on a STATIC graph  (QUIET control)
 * ============================================================================
 *
 *  DWARF   : Graph Traversal (D9) (Berkeley 13 computational motif)
 *  FAMILY  : IDLE                 (first-division, memory-signature label)
 *  PURPOSE : Serve as a deliberate QUIET / null control for graph work. The
 *            graph is a large read-only structure; a full BFS sweep touches all
 *            of it, but only by READING. The sole writes are a tiny per-node
 *            visited/distance array and a frontier queue. It tests the null
 *            hypothesis: is traversing a static graph invisible to the host
 *            memory WRITE-signal? (expected yes -- the "graph traversal is
 *            invisible" baseline.)
 *
 *  WHY IT IS THE QUIET CONTROL (vs a graph kernel that mutates)
 *  ----------------------------------------------------------------------------
 *  The graph is built ONCE, in warmup, and never written again. Each measured
 *  pass re-picks only the SOURCE node and re-runs BFS; the adjacency structure
 *  (col_idx, the dominant buffer) is pure input. Reads are invisible to a
 *  host write-signal, so all that traffic vanishes and only the small dist[]
 *  updates and the queue remain observable -- hence near-idle despite real work.
 *  HONEST CAVEAT: this "quiet" reading holds precisely because the graph is
 *  never regenerated or edited mid-run; a variant that rebuilt or mutated edges
 *  each pass would add generation writes and would no longer be a quiet control.
 *
 *  GRAPH LAYOUT (CSR, uniform out-degree).  Every node has exactly `degree`
 *  out-edges, so the usual CSR row_ptr is implicit -- node i owns the slice
 *  [i*degree, i*degree + degree) of col_idx -- and only col_idx is stored:
 *
 *      node i owns a length-degree slice (base = i*degree) of col_idx:
 *
 *        col_idx : [ .. | n0 n1 n2 .. n(degree-1) | .. ]   neighbour node ids
 *                          |  |            \
 *                          v  v             v   (each n is a destination node)
 *        (traversal reads these to enqueue unvisited neighbours) READ-ONLY
 *
 *  TRAVERSAL STATE (the ONLY things written during a pass):
 *
 *        dist  : [ d0 d1 d2 .. d(N-1) ]   int, -1 = unvisited, else BFS layer
 *        queue : [ .. FIFO of node ids .. ]   explicit ring-free array + head/tail
 *
 *  ALGORITHM (per measured pass):
 *      1. Reset dist[] to -1 (small O(N) write over one int array).
 *      2. Pick a NEW random source s (re-seed ONLY the source, NOT the graph).
 *      3. BFS: dist[s] = 0, push s; while the queue is non-empty pop u and, for
 *         each neighbour w in col_idx[u*degree .. +degree), if dist[w] == -1 set
 *         dist[w] = dist[u] + 1 and push w. Reads dominate (the adjacency slice
 *         per popped node); the only writes are dist[w] and the queue slots.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Almost nothing. The dominant memory traffic is READS -- the col_idx
 *      adjacency streamed as the frontier expands -- all invisible to a
 *      write-signal. The only writes are the small dist[] array (reset + one
 *      store per node visited) and the frontier queue, so the observable
 *      footprint is tiny and the workload looks near-idle.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: IDLE (quiet control). Dwarf: Graph Traversal.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (build graph) / measure (BFS traversals) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_bfs_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign static-graph BFS; QUIET control, Graph Traversal)\n"
"  --nodes N             Number of graph nodes (default 1000000)\n"
"  --degree D            Out-edges per node (default 16)\n"
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

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nodes      = p2_get_i64(argc, argv, "--nodes", 1000000);
    long long degree     = p2_get_i64(argc, argv, "--degree", 16);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nodes < 1000 || nodes > 200000000LL) { P2_LOG_ERR("nodes %lld out of range (1000..2e8)", nodes); return 2; }
    if (degree < 1 || degree > 1024) { P2_LOG_ERR("degree %lld out of range (1..1024)", degree); return 2; }
    size_t N = (size_t)nodes, D = (size_t)degree;
    size_t edges = N * D;
    /* col_idx (edges * int) is the dominant read-only buffer; dist + queue are
     * the small write buffers (one int per node each). */
    size_t bytes = edges * sizeof(int32_t) + 2 * N * sizeof(int32_t);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "IDLE");
    p2_meta_kv_str(&m, "dwarf", "Graph Traversal");
    p2_meta_kv_str(&m, "role", "quiet control: static read-only graph, only dist/queue written");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "nodes", nodes);
    p2_meta_kv_i64(&m, "degree", degree);
    p2_meta_kv_u64(&m, "edges", edges);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The adjacency (col_idx) is the dominant buffer -> mmap + advise + mlock it.
     * It is READ-ONLY during measurement (built once, below), which is exactly
     * what keeps the measured phase quiet on the write-signal. */
    int32_t *col = (int32_t *)mmap(NULL, edges * sizeof(int32_t), PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (col == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", edges * sizeof(int32_t), strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(col, edges * sizeof(int32_t), MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(col, edges * sizeof(int32_t));

    /* Traversal state: dist[] (visited/layer, -1 = unvisited) and an explicit
     * FIFO queue. These are the ONLY things written during a BFS pass. Sized to
     * N because a single BFS can enqueue every node at most once. */
    int32_t *dist  = (int32_t *)malloc(N * sizeof(int32_t));
    int32_t *queue = (int32_t *)malloc(N * sizeof(int32_t));
    if (!dist || !queue) {
        P2_LOG_ERR("malloc failed");
        free(dist); free(queue); munmap(col, edges * sizeof(int32_t));
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Build the ONE random directed graph once, during warmup. Every node gets
     * exactly `degree` out-edges to random destination nodes in [0, N). The
     * random destinations are what make the later traversal scattered and
     * cache-unfriendly. After this the adjacency is never written again -- that
     * read-only-ness is exactly what keeps the measured phase quiet. */
    for (size_t e = 0; e < edges; e++) {
        col[e] = (int32_t)(p2_rng_next(&rng) % (uint64_t)N);
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t visited_total = 0;   /* accumulated nodes reached (keeps work live) */
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* (1) Reset dist[] to -1 (unvisited). Small O(N) write over one int
         * array -- one of only two things this kernel ever writes -- so it stays
         * cheap and near-idle on the write-signal. */
        for (size_t i = 0; i < N; i++) dist[i] = -1;
        /* (2) Pick a NEW random source each pass. We re-seed ONLY the source
         * node, NOT the graph, so no generation writes are introduced. */
        int32_t s = (int32_t)(p2_rng_next(&rng) % (uint64_t)N);

        /* (3) BFS from s using the explicit FIFO queue. Reads dominate -- the
         * adjacency slice col[u*D .. +D) of each popped node is streamed -- and
         * the ONLY writes are dist[w] and the queue slots. Large read footprint,
         * tiny visible write footprint. */
        size_t head = 0, tail = 0;
        dist[s] = 0;
        queue[tail++] = s;                              /* push source */
        while (head < tail) {
            int32_t u = queue[head++];                  /* pop */
            size_t base = (size_t)u * D;                /* node u owns [base, base+D) */
            for (size_t k = 0; k < D; k++) {
                int32_t w = col[base + k];              /* READ a neighbour id */
                if (dist[w] == -1) {                    /* first time we reach w */
                    dist[w] = dist[u] + 1;              /* the lone dist write */
                    queue[tail++] = w;                  /* enqueue */
                }
            }
        }
        visited_total += (uint64_t)tail;                /* nodes reached this pass */
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile int sink = dist[N / 2];                    /* a live distance value */

    free(dist); free(queue);
    munmap(col, edges * sizeof(int32_t));

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "visited_total", visited_total);
    p2_meta_kv_i64(&m, "sink", (long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected NEAR-IDLE: static read-only graph invisible; only dist[]/queue (small) written");
    p2_meta_close(&m);
    return 0;
}
