/* kernel_rmat_gen_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Graph Traversal dwarf (Berkeley motif D9), the GENERATION side of it. The
 * classic Graph500 benchmark has two halves: first BUILD a huge synthetic graph,
 * then TRAVERSE it (breadth-first search). This kernel is the build half -- R-MAT
 * (Recursive MATrix) graph generation, the exact edge generator Graph500 uses.
 *
 * A graph on N = 2^scale nodes with E = edge_factor * N edges is produced by
 * emitting E directed edges, each drawn from a 2x2 recursive probability matrix.
 * The whole edge list -- the large object -- is WRITTEN out, one endpoint pair at
 * a time, which is precisely what makes this workload visible.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (generation vs traversal)
 * ----------------------------------------------------------------------------
 * Traversing a graph (BFS, the other Graph500 half) is READ-dominated: you chase
 * an existing edge list and visit-array, writing almost nothing back, so a host
 * write-signal barely sees it (a QUIET tell). GENERATION is the opposite. Filling
 * src_arr[e] and dst_arr[e] for every one of the E edges sweeps the entire edge
 * list with fresh stores -- a large, sequential, fully VISIBLE write. Same dwarf,
 * same data structure, opposite read/write balance: that asymmetry is the tell.
 *
 * ============================================================================
 * ALGORITHM (R-MAT, one edge at a time, bit by bit)
 * ============================================================================
 * The N x N adjacency matrix is split into four equal quadrants with fixed
 * probabilities a, b, c, d (a+b+c+d = 1). Graph500 defaults: a=0.57, b=c=0.19,
 * d=0.05. To place ONE edge we recurse `scale` levels deep, at each level picking
 * one of the four quadrants and thereby fixing one bit of both the source and the
 * destination node id:
 *
 *      +---------------------+---------------------+
 *      |        a            |        b            |   top    -> src bit = 0
 *      |  (do nothing)       |  (set dst bit)      |
 *      +---------------------+---------------------+
 *      |        c            |        d            |   bottom -> src bit = 1
 *      |  (set src bit)      |  (set src+dst bit)  |
 *      +---------------------+---------------------+
 *        left -> dst bit = 0    right -> dst bit = 1
 *
 *   for level = 0 .. scale-1:
 *       draw a uniform double p in [0,1)
 *       bit = (scale-1-level)                    // most-significant bit first
 *       if      p <  a           : neither bit   (top-left quadrant)
 *       else if p <  a+b         : set dst bit   (top-right)
 *       else if p <  a+b+c       : set src bit   (bottom-left)
 *       else                     : set both bits (bottom-right)
 *
 * After `scale` levels src and dst are each in [0, N). The pair is stored into the
 * parallel int32 arrays src_arr[e], dst_arr[e]. Those two arrays are the dominant
 * mmap'd buffer and receive the bulk write.
 *
 * Each measured pass regenerates a FRESH edge list (all E edges rewritten) and
 * counts the generation, so the measured phase is a continuous stream of writes
 * over the large edge list -- the visible signature.
 *
 * Determinism note (also the basis of the correctness test): with a=1 every draw
 * lands top-left, so every edge is (0,0); with d=1 every draw lands bottom-right,
 * so every bit of both endpoints is set and every edge is (N-1, N-1). Those corner
 * cases pin the bit logic exactly.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Graph Traversal. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + first fill) / measure (regenerate edge list) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_rmat_gen_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign R-MAT graph generation; VISIBLE, Graph Traversal)\n"
"  --scale S             log2(nodes); N = 2^S (default 18 -> N=262144)\n"
"  --edge-factor EF      Edges per node; E = EF*N (default 16)\n"
"  --a-milli A           Quadrant a probability x1000 (default 570 = 0.57)\n"
"  --b-milli B           Quadrant b probability x1000 (default 190 = 0.19)\n"
"  --c-milli C           Quadrant c probability x1000 (default 190 = 0.19)\n"
"                        (d is computed as 1000-a-b-c over 1000)\n"
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

/* Generate the whole edge list once: E edges, each built bit-by-bit over `scale`
 * levels from the quadrant thresholds. This is the workload's signature write --
 * every src_arr[e]/dst_arr[e] slot is stored, sweeping the large edge buffer. */
static void rmat_generate(int32_t *src_arr, int32_t *dst_arr, size_t E, int scale,
                          double a, double ab, double abc, p2_rng_t *rng) {
    for (size_t e = 0; e < E; e++) {
        int32_t src = 0, dst = 0;
        for (int level = 0; level < scale; level++) {
            double p = rng_unit(rng);
            int bit = scale - 1 - level;            /* most-significant bit first */
            if (p < a) {
                /* top-left quadrant: neither bit set */
            } else if (p < ab) {
                dst |= (int32_t)1 << bit;            /* top-right: set dst bit */
            } else if (p < abc) {
                src |= (int32_t)1 << bit;            /* bottom-left: set src bit */
            } else {
                src |= (int32_t)1 << bit;            /* bottom-right: set both bits */
                dst |= (int32_t)1 << bit;
            }
        }
        src_arr[e] = src;                           /* the bulk visible writes */
        dst_arr[e] = dst;
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long scale      = p2_get_i64(argc, argv, "--scale", 18);
    long long edge_factor = p2_get_i64(argc, argv, "--edge-factor", 16);
    /* Quadrant probabilities are passed as integer-milli, because the phase2 arg
     * helpers are integer-only: --a-milli 570 = 0.57. d is the remainder. */
    long long a_milli    = p2_get_i64(argc, argv, "--a-milli", 570);
    long long b_milli    = p2_get_i64(argc, argv, "--b-milli", 190);
    long long c_milli    = p2_get_i64(argc, argv, "--c-milli", 190);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (scale < 4 || scale > 30) { P2_LOG_ERR("scale %lld out of range (4..30)", scale); return 2; }
    if (edge_factor < 1 || edge_factor > 256) { P2_LOG_ERR("edge-factor %lld out of range (1..256)", edge_factor); return 2; }
    long long d_milli = 1000 - a_milli - b_milli - c_milli;
    if (a_milli < 0 || b_milli < 0 || c_milli < 0 || d_milli < 0) {
        P2_LOG_ERR("a+b+c milli %lld exceeds 1000 (d would be negative)", a_milli + b_milli + c_milli); return 2;
    }
    /* Quadrant thresholds as running cumulative sums of the probabilities. */
    double a   = (double)a_milli / 1000.0;
    double ab  = (double)(a_milli + b_milli) / 1000.0;
    double abc = (double)(a_milli + b_milli + c_milli) / 1000.0;

    size_t N = (size_t)1 << (int)scale;
    size_t E = (size_t)edge_factor * N;
    size_t bytes = 2 * E * sizeof(int32_t);         /* src_arr + dst_arr dominate */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("edge-list bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Graph Traversal");
    p2_meta_kv_str(&m, "scheme", "R-MAT graph generation (Graph500 construction; edge list written each pass)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "scale", scale);
    p2_meta_kv_i64(&m, "edge_factor", edge_factor);
    p2_meta_kv_u64(&m, "nodes", N);
    p2_meta_kv_u64(&m, "edges", E);
    p2_meta_kv_i64(&m, "a_milli", a_milli);
    p2_meta_kv_i64(&m, "b_milli", b_milli);
    p2_meta_kv_i64(&m, "c_milli", c_milli);
    p2_meta_kv_i64(&m, "d_milli", d_milli);
    p2_meta_kv_u64(&m, "edge_list_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The edge list is the dominant buffer -> mmap + mlock it (it is fully
     * rewritten every pass, which is the workload's signature visible write). */
    int32_t *edges = (int32_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                     MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (edges == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(edges, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(edges, bytes);
    int32_t *src_arr = edges;                       /* first half: sources */
    int32_t *dst_arr = edges + E;                   /* second half: destinations */

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Generate the edge list once during warmup so the pages are faulted in and
     * the working set is resident before measurement begins. */
    rmat_generate(src_arr, dst_arr, E, (int)scale, a, ab, abc, &rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed so each generation draws a fresh edge list, then regenerate all
         * E edges. This rewrites the entire src_arr/dst_arr buffer every pass --
         * a large sequential VISIBLE write, the whole point of this kernel. */
        p2_rng_seed(&rng, seed + passes + 1);
        rmat_generate(src_arr, dst_arr, E, (int)scale, a, ab, abc, &rng);
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile int32_t sink = src_arr[E / 2] ^ dst_arr[E / 2];   /* keep the buffer live */

    munmap(edges, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_i64(&m, "sink", (long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "VISIBLE by design: generation rewrites the whole edge list each pass, unlike quiet BFS traversal");
    p2_meta_close(&m);
    return 0;
}
