/* kernel_label_prop_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Graph Traversal dwarf (Berkeley motif D9), the WRITE-VISIBLE variant: label
 * propagation for connected components. A connected component is a maximal set
 * of nodes all reachable from one another; labelling each node with the SMALLEST
 * node id in its component identifies the components. Label propagation computes
 * this iteratively: start every node with its own id as a label, then repeatedly
 * replace each node's label by the minimum of its own label and its neighbours'
 * labels. The minimum id in a component floods outward one hop per sweep, so
 * after enough sweeps every node in a component carries that component's minimum
 * id. This is a GRAPH STENCIL -- like a grid stencil, but the neighbourhood is an
 * irregular adjacency list instead of a fixed grid.
 *
 * WHY IT IS A DISTINCT (WRITE-VISIBLE) MEMORY SIGNATURE (vs kernel_bfs_v2)
 * ----------------------------------------------------------------------------
 * The BFS control in this same dwarf is QUIET: it traverses a static graph by
 * READING the adjacency and writes almost nothing (a small dist[] and a queue).
 * Reads are invisible to a host memory WRITE-signal, so BFS reads as near-idle.
 * Label propagation shares the same read-only adjacency, but it ALSO rewrites a
 * full node-label array (N ints) on every sweep, sweep after sweep. That
 * repeatedly-overwritten label array is the distinctive, VISIBLE write -- an
 * iterated node-label rewrite over an irregular graph. Same dwarf, opposite
 * write-signature: BFS is the quiet baseline, this is the loud one.
 *
 * ============================================================================
 * GRAPH LAYOUT (CSR, symmetric / undirected)
 * ============================================================================
 * The graph is stored in Compressed Sparse Row form with an explicit row_ptr,
 * because a symmetric random graph has a variable per-node degree (unlike the
 * uniform-degree BFS control, whose row_ptr is implicit):
 *
 *     row_ptr : [ r0 r1 r2 .. rN ]   length N+1; node i owns col_idx[r_i .. r_{i+1})
 *     col_idx : [ .............. ]   neighbour node ids, grouped by node
 *
 * It is built UNDIRECTED: whenever a random edge u--v is drawn, BOTH u->v and
 * v->u are stored, so the adjacency is symmetric and connected components are
 * well defined. The CSR is built ONCE, in warmup, and is READ-ONLY during
 * measurement -- only the label arrays are written.
 *
 * ============================================================================
 * THE VISIBLE WRITE BUFFERS
 * ============================================================================
 *     label     : [ l0 l1 .. l(N-1) ]   int, current label of each node
 *     label_new : [ .............. ]     int, next-sweep labels (double buffer)
 *
 * These two mmap'd int arrays are the dominant write traffic. One update sweep
 * computes, for every node i,
 *     label_new[i] = min( label[i], min over neighbours j of label[j] )
 * then swaps label and label_new. Iterating the sweep drives the minimum id of
 * each component outward until every node's label equals that minimum -- i.e.
 * convergence to connected components.
 *
 * ============================================================================
 * ALGORITHM (per measured pass)
 * ============================================================================
 *   1. Re-seed the labels only (NOT the graph): set label[i] = i for all i.
 *   2. Run --iters update sweeps of the min-label rule above, swapping the two
 *      label buffers after each sweep. (--iters is a fixed sweep budget, not a
 *      convergence test; with enough sweeps the labels reach the exact connected
 *      components, which the standalone verifier checks.)
 *   3. Count the pass.
 *
 * MEMORY SIGNATURE (what the host write-signal actually sees):
 *   A steady, iterated rewrite of the N-int label array -- one full pass over it
 *   per sweep, --iters sweeps per measured pass, pass after pass. The adjacency
 *   (row_ptr + col_idx, the larger buffers) is READ-ONLY and invisible; the
 *   visible footprint is the label double-buffer being overwritten again and
 *   again in an irregular, graph-driven order. This is the WRITE-VISIBLE
 *   counterpart to the quiet BFS control in the same dwarf.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Graph Traversal. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (build graph) / measure (label-propagation passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_label_prop_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign label-propagation connected components; Graph Traversal kernel)\n"
"  --nodes N             Number of graph nodes (default 500000)\n"
"  --degree D            Average out-edges per node (default 8)\n"
"  --iters K             Min-label sweeps per pass (default 30)\n"
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
    long long nodes      = p2_get_i64(argc, argv, "--nodes", 500000);
    long long degree     = p2_get_i64(argc, argv, "--degree", 8);
    long long iters      = p2_get_i64(argc, argv, "--iters", 30);
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
    if (iters < 1 || iters > 100000) { P2_LOG_ERR("iters %lld out of range (1..100000)", iters); return 2; }
    size_t N = (size_t)nodes, D = (size_t)degree;
    /* Undirected graph: each of the ~N*D drawn endpoints creates two directed
     * half-edges (u->v and v->u), so the CSR holds up to 2*N*D neighbour slots.
     * row_ptr is N+1 ints; col_idx is the (dominant, read-only) adjacency; label
     * and label_new (N ints each) are the VISIBLE double-buffered write. */
    size_t half_edges = 2 * N * D;
    size_t bytes = (N + 1) * sizeof(int32_t)      /* row_ptr                */
                 + half_edges * sizeof(int32_t)   /* col_idx (read-only)    */
                 + 2 * N * sizeof(int32_t);       /* label + label_new (WR) */
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
    p2_meta_kv_str(&m, "scheme", "min-label propagation for connected components (iterated node-label rewrite; graph stencil)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "nodes", nodes);
    p2_meta_kv_i64(&m, "degree", degree);
    p2_meta_kv_i64(&m, "iters", iters);
    p2_meta_kv_u64(&m, "half_edges_cap", half_edges);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The adjacency (row_ptr + col_idx) is the dominant read-only structure and
     * the label double-buffer is the dominant write structure -> mmap + advise +
     * mlock all of them. col_idx is READ-ONLY during measurement (built once,
     * below); label/label_new are rewritten every sweep, which is the workload's
     * signature write. */
    size_t rp_bytes  = (N + 1) * sizeof(int32_t);
    size_t col_bytes = half_edges * sizeof(int32_t);
    size_t lab_bytes = N * sizeof(int32_t);
    int32_t *row_ptr   = (int32_t *)mmap(NULL, rp_bytes,  PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    int32_t *col_idx   = (int32_t *)mmap(NULL, col_bytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    int32_t *label     = (int32_t *)mmap(NULL, lab_bytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    int32_t *label_new = (int32_t *)mmap(NULL, lab_bytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (row_ptr == MAP_FAILED || col_idx == MAP_FAILED ||
        label == MAP_FAILED || label_new == MAP_FAILED) {
        P2_LOG_ERR("mmap failed: %s", strerror(errno));
        if (row_ptr   != MAP_FAILED) munmap(row_ptr, rp_bytes);
        if (col_idx   != MAP_FAILED) munmap(col_idx, col_bytes);
        if (label     != MAP_FAILED) munmap(label, lab_bytes);
        if (label_new != MAP_FAILED) munmap(label_new, lab_bytes);
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(row_ptr,   rp_bytes,  MADV_NOHUGEPAGE);
    p2_madvise(col_idx,   col_bytes, MADV_NOHUGEPAGE);
    p2_madvise(label,     lab_bytes, MADV_NOHUGEPAGE);
    p2_madvise(label_new, lab_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) {
        p2_mlock_soft(row_ptr,   rp_bytes);
        p2_mlock_soft(col_idx,   col_bytes);
        p2_mlock_soft(label,     lab_bytes);
        p2_mlock_soft(label_new, lab_bytes);
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Build the ONE random UNDIRECTED graph once, during warmup, into symmetric
     * CSR. We draw N*D random edges u--v; each contributes a half-edge at BOTH u
     * and v, so the adjacency is symmetric and connected components are well
     * defined. Two-pass counting sort:
     *   (a) count each node's degree from the drawn edge list,
     *   (b) prefix-sum the degrees into row_ptr,
     *   (c) scatter both endpoints of every edge into col_idx at a per-node
     *       cursor.
     * The random endpoints are what make the later label sweep scattered and
     * cache-unfriendly. After this the adjacency is never written again -- that
     * read-only-ness is exactly what keeps the adjacency invisible while only the
     * labels stay visible. */
    size_t E = N * D;                                   /* number of undirected edges drawn */
    int32_t *esrc = (int32_t *)malloc(E * sizeof(int32_t));
    int32_t *edst = (int32_t *)malloc(E * sizeof(int32_t));
    if (!esrc || !edst) {
        P2_LOG_ERR("edge-list malloc failed");
        free(esrc); free(edst);
        munmap(row_ptr, rp_bytes); munmap(col_idx, col_bytes);
        munmap(label, lab_bytes); munmap(label_new, lab_bytes);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    /* (a) draw edges and count degrees. row_ptr[i+1] temporarily holds deg(i). */
    for (size_t i = 0; i <= N; i++) row_ptr[i] = 0;
    for (size_t e = 0; e < E; e++) {
        int32_t u = (int32_t)(p2_rng_next(&rng) % (uint64_t)N);
        int32_t v = (int32_t)(p2_rng_next(&rng) % (uint64_t)N);
        esrc[e] = u; edst[e] = v;
        row_ptr[u + 1]++;                               /* half-edge u->v */
        row_ptr[v + 1]++;                               /* half-edge v->u */
    }
    /* (b) prefix-sum degrees into CSR row offsets. */
    for (size_t i = 0; i < N; i++) row_ptr[i + 1] += row_ptr[i];
    size_t total_half = (size_t)row_ptr[N];             /* actual half-edge count */
    /* (c) scatter both half-edges of each undirected edge into col_idx. cursor[i]
     * walks node i's slice as its neighbours are placed. */
    int32_t *cursor = (int32_t *)malloc(N * sizeof(int32_t));
    if (!cursor) {
        P2_LOG_ERR("cursor malloc failed");
        free(esrc); free(edst);
        munmap(row_ptr, rp_bytes); munmap(col_idx, col_bytes);
        munmap(label, lab_bytes); munmap(label_new, lab_bytes);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    for (size_t i = 0; i < N; i++) cursor[i] = row_ptr[i];
    for (size_t e = 0; e < E; e++) {
        int32_t u = esrc[e], v = edst[e];
        col_idx[cursor[u]++] = v;
        col_idx[cursor[v]++] = u;
    }
    free(esrc); free(edst); free(cursor);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t sweeps_total = 0;      /* accumulated label sweeps (keeps work live) */
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* (1) Re-seed labels only: every node starts labelled with its own id.
         * This is an O(N) write over the label array; the graph is untouched, so
         * no adjacency-generation writes are introduced -- only labels move. */
        for (size_t i = 0; i < N; i++) label[i] = (int32_t)i;

        /* (2) Run K min-label sweeps. Each sweep is the graph stencil: for every
         * node i, take the minimum of its own label and all its neighbours'
         * labels, and write it into label_new[i]; then swap the buffers. The
         * adjacency (row_ptr/col_idx) is READ; label_new is fully WRITTEN each
         * sweep -- that repeated rewrite of the N-int label array is the visible
         * signature. Iterating floods each component's minimum id outward. */
        for (long long it = 0; it < iters; it++) {
            for (size_t i = 0; i < N; i++) {
                int32_t best = label[i];                 /* own label */
                size_t b = (size_t)row_ptr[i];           /* node i owns [b, en) */
                size_t en = (size_t)row_ptr[i + 1];
                for (size_t k = b; k < en; k++) {
                    int32_t lj = label[col_idx[k]];      /* READ a neighbour label */
                    if (lj < best) best = lj;            /* min over neighbours   */
                }
                label_new[i] = best;                     /* the lone visible write */
            }
            int32_t *tmp = label; label = label_new; label_new = tmp;  /* swap buffers */
        }
        sweeps_total += (uint64_t)iters;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile int sink = label[N / 2];                    /* a live component label */

    munmap(row_ptr, rp_bytes);
    munmap(col_idx, col_bytes);
    munmap(label, lab_bytes);
    munmap(label_new, lab_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "half_edges", total_half);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "sweeps_total", sweeps_total);
    p2_meta_kv_i64(&m, "sink", (long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "WRITE-VISIBLE: label double-buffer rewritten every sweep; adjacency read-only/invisible; --iters is a fixed sweep budget, not a convergence test");
    p2_meta_close(&m);
    return 0;
}
