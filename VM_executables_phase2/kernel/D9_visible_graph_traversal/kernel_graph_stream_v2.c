/* kernel_graph_stream_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Graph Traversal dwarf (Berkeley motif D9), the STREAMING / dynamic-graph
 * variant. A classical graph benchmark loads one fixed graph and then only READS
 * it -- BFS, PageRank, connected components all walk edges that never change.
 * This kernel does the opposite: it builds the graph by STREAMING edges into a
 * growing adjacency structure, so the graph STRUCTURE ITSELF is written. This is
 * the modern streaming-graph-analytics pattern (think a social/interaction graph
 * that keeps ingesting new edges) rather than a static, read-only traversal.
 *
 * WHY IT IS A DISTINCT (VISIBLE) MEMORY SIGNATURE
 * ----------------------------------------------------------------------------
 * A static traversal WRITES almost nothing -- it streams the adjacency as READS
 * and keeps only a tiny frontier/rank vector, so it is near-invisible to a host
 * memory WRITE-signal (that is exactly the "quiet control" regime). Streaming
 * ingestion inverts that: every inserted edge APPENDS into the adjacency array,
 * so the dominant memory buffer -- the adjacency itself -- is continuously
 * written as the structure grows. Those append writes ARE the signature. The
 * graph keeps CHANGING, and the change is what the write-signal sees.
 *
 * ============================================================================
 * DATA STRUCTURE (bucketed dynamic adjacency)
 * ============================================================================
 * A fixed-capacity bucketed adjacency: N nodes, each with room for C neighbours.
 *   adj    : flat array of N*C uint32 -- node u owns the slice [u*C, u*C + C).
 *            This is the dominant mmap'd buffer; its appends are the tell.
 *   degree : N uint32 -- degree[u] = how many neighbours node u currently holds,
 *            i.e. the next free slot within u's bucket.
 *
 * Insert edge (u, v):
 *   if degree[u] <  C  ->  adj[u*C + degree[u]] = v;  degree[u]++  (APPEND)
 *   else               ->  drop the edge, count it as an overflow  (bucket full)
 *
 * PICTURE (one bucket per node, capacity C = 4, degree marks the write cursor):
 *
 *      node 0  bucket: [ v v v _ ]   degree[0] = 3  -> next append lands here ^
 *      node 1  bucket: [ v _ _ _ ]   degree[1] = 1
 *      node 2  bucket: [ v v v v ]   degree[2] = 4  -> FULL, further edges drop
 *      ...
 *      adj (flat)   : [ v v v _ | v _ _ _ | v v v v | .. ]   <-- appends WRITE here
 *      degree       : [   3     |    1    |    4    | .. ]    <-- small cursor array
 *
 * ============================================================================
 * ALGORITHM (per measured pass)
 * ============================================================================
 *   1. Reset the structure: set every degree[u] = 0 (all buckets logically empty;
 *      the graph is rebuilt from scratch each pass, streaming-ingestion style).
 *   2. Stream E random edge insertions: draw u, v uniformly in [0, N) and append
 *      v into u's bucket (or drop on overflow). The appends sweep adj[] with the
 *      growing structure's writes -- the visible signature of this workload.
 * Count passes; carry the last pass's dropped/overflow tally into the metadata.
 *
 * MEMORY SIGNATURE (what the host write-signal actually sees):
 *   The adjacency array adj[] (the dominant buffer, potentially many hundreds of
 *   MB) is WRITTEN as edges append -- unlike a static traversal that would only
 *   READ it. degree[] is a small parallel write. Edge endpoints are generated,
 *   not read from a stored edge list, so reads stay minimal and the write side
 *   dominates. This is a VISIBLE / write-heavy workload by construction: the
 *   graph structure keeps changing, and that change is the tell.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Graph Traversal. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + one priming build) / measure (streaming passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_graph_stream_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign streaming/dynamic graph; VISIBLE, Graph-Traversal)\n"
"  --nodes N             Number of graph nodes (default 500000)\n"
"  --capacity C          Per-node bucket capacity (default 32)\n"
"  --edges E             Edge insertions streamed per pass (default 4000000)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on adjacency bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* Stream E random edge insertions into the bucketed adjacency, rebuilding from
 * empty. Returns the number of edges dropped because their source bucket was
 * already full. adj[]/degree[] are WRITTEN throughout -- the growing structure's
 * appends are the workload's visible signature.
 *   adj    : flat N*C neighbour array; node u owns [u*C, u*C + C).
 *   degree : per-node write cursor / current degree (also the free-slot index).
 * The RNG is passed in so each measured pass streams a fresh random edge set. */
static uint64_t stream_edges(uint32_t *adj, uint32_t *degree,
                             size_t N, size_t C, size_t E, p2_rng_t *rng) {
    for (size_t u = 0; u < N; u++) degree[u] = 0;   /* (1) reset: all buckets empty */
    uint64_t dropped = 0;
    for (size_t e = 0; e < E; e++) {                /* (2) stream E edge insertions */
        uint32_t u = (uint32_t)(p2_rng_next(rng) % (uint64_t)N);
        uint32_t v = (uint32_t)(p2_rng_next(rng) % (uint64_t)N);
        uint32_t d = degree[u];
        if (d < (uint32_t)C) {                       /* room in u's bucket -> APPEND */
            adj[(size_t)u * C + d] = v;              /* the growing-structure WRITE  */
            degree[u] = d + 1;                       /* advance u's write cursor     */
        } else {
            dropped++;                               /* bucket full -> drop the edge */
        }
    }
    return dropped;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nodes      = p2_get_i64(argc, argv, "--nodes", 500000);
    long long capacity   = p2_get_i64(argc, argv, "--capacity", 32);
    long long edges      = p2_get_i64(argc, argv, "--edges", 4000000);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nodes < 1000 || nodes > 200000000LL) { P2_LOG_ERR("nodes %lld out of range (1000..2e8)", nodes); return 2; }
    if (capacity < 1 || capacity > 4096) { P2_LOG_ERR("capacity %lld out of range (1..4096)", capacity); return 2; }
    if (edges < 1 || edges > 20000000000LL) { P2_LOG_ERR("edges %lld out of range (1..2e10)", edges); return 2; }
    size_t N = (size_t)nodes, C = (size_t)capacity, E = (size_t)edges;
    size_t nslot = N * C;                            /* total adjacency slots */
    size_t bytes = nslot * sizeof(uint32_t) + N * sizeof(uint32_t);  /* adj + degree */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("adjacency bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Graph Traversal");
    p2_meta_kv_str(&m, "scheme", "streaming/dynamic graph: edges appended into a growing bucketed adjacency (structure is written)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "nodes", nodes);
    p2_meta_kv_i64(&m, "capacity", capacity);
    p2_meta_kv_i64(&m, "edges_per_pass", edges);
    p2_meta_kv_u64(&m, "adjacency_slots", nslot);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The adjacency array is the dominant buffer -> mmap + mlock it (it is
     * appended into on every streamed edge, which is the workload's signature
     * write). degree[] is a small parallel array carried alongside it. */
    uint32_t *adj = (uint32_t *)mmap(NULL, nslot * sizeof(uint32_t), PROT_READ | PROT_WRITE,
                                     MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (adj == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", nslot * sizeof(uint32_t), strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(adj, nslot * sizeof(uint32_t), MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(adj, nslot * sizeof(uint32_t));

    uint32_t *degree = (uint32_t *)malloc(N * sizeof(uint32_t));
    if (!degree) {
        munmap(adj, nslot * sizeof(uint32_t)); P2_LOG_ERR("malloc failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    if (!no_mlock) p2_mlock_soft(degree, N * sizeof(uint32_t));

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Prime the structure with one full streaming build. This faults in the whole
     * adjacency array (first-touch) and exercises the exact append path the
     * measured phase repeats, so timing is not polluted by cold-page faults. */
    uint64_t dropped = stream_edges(adj, degree, N, C, E, &rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Each pass resets all buckets and STREAMS E fresh random edge insertions,
         * appending each into the adjacency. The RNG advances across passes so no
         * pass repeats the same edge set. The appends sweep adj[] with the growing
         * structure's writes -- the whole point: the graph keeps changing. */
        dropped = stream_edges(adj, degree, N, C, E, &rng);
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Touch both structures so the last streaming build cannot be optimised away:
     * read a neighbour slot for a node that actually received an edge. */
    size_t probe = N / 2;
    volatile uint32_t sink = (degree[probe] > 0) ? adj[probe * C] : degree[probe];

    free(degree);
    munmap(adj, nslot * sizeof(uint32_t));

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "last_pass_dropped", dropped);
    p2_meta_kv_f64(&m, "sink", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected VISIBLE: adjacency appends write the growing structure; overflow drops edges past per-node capacity C");
    p2_meta_close(&m);
    return 0;
}
