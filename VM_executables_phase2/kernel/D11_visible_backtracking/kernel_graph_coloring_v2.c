/* kernel_graph_coloring_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  GRAPH m-COLOURING:  CSP backtracking with FORWARD CHECKING (domain prune/restore)
 * ============================================================================
 *
 *  DWARF   : Backtrack / Branch-and-Bound (D11)  (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL (write-visible)              (first-division memory-signature label)
 *  PURPOSE : Visibility source (b) LARGE WORKING STATE. Colour a big sparse
 *            graph by depth-first backtracking; the live state is a colour[]
 *            assignment array plus a per-vertex colour-DOMAIN table over a large
 *            vertex set. Forward checking CHURNS that state -- pruning a colour
 *            out of every neighbour's domain on assign, then RESTORING it on
 *            backtrack -- so the domain + colour arrays are written and rewritten
 *            all through the search. That churn over a large mmap'd working set
 *            is the visible write front.
 *
 *  PICTURE (top view):
 *      a sparse graph, vertices getting colours     forward-checking a neighbour
 *      (m = 3 palette: 1=R 2=G 3=B, 0=unassigned)    domain on assign / backtrack
 *
 *            (v0:R)---(v1:G)                          assign v0 <- R :
 *              |    \    |                              v1.dom &= ~{R}   (WRITE, trailed)
 *              |     \   |                              v2.dom &= ~{R}   (WRITE, trailed)
 *            (v2:B)---(v3:0)                          v3 still {R,G,B}
 *                       .                             ...deeper...
 *              domain[v3] : {R G B}  -- prune -->  {_ G B}   (push (v3,R) on trail)
 *                          on BACKTRACK, pop trail: {_ G B} -- restore -->  {R G B}
 *
 *      colour[] (assignment) and domain[] (m-bit mask per vertex) are the two
 *      dominant mmap'd buffers; the adjacency (CSR) is READ-ONLY once built.
 *
 *  ALGORITHM (CSP forward-checking backtracking):
 *      1. Build ONE large sparse UNDIRECTED graph in symmetric CSR: V vertices,
 *         ~V*d/2 random edges (average degree d). The palette is sized so the
 *         search GENUINELY backtracks: the default m (6)
 *         sits near the chromatic threshold of an average-degree-6 random graph,
 *         so greedy index-order assignment repeatedly wedges and must UNDO. Each
 *         backtrack RESTORES pruned domain bits (a WRITE), so the colour + domain
 *         arrays are churned by prune AND restore -- the restore-on-backtrack is
 *         the signature that distinguishes this from a monotone label relaxation
 *         (cf. D9 label_prop, which never undoes). A BACKTRACK BUDGET caps each
 *         pass so it always terminates, returning the best CONFLICT-FREE partial
 *         colouring (colored_all=0 when the budget is hit -- a normal solver-under-
 *         budget outcome, not an error; the verifier confirms zero conflicts). A
 *         generous m (> max degree, e.g. 16) recovers a near-instant greedy regime
 *         whose visibility is the large arrays alone, with ~0 backtracking.
 *      2. Order vertices by index. Maintain, for every vertex,
 *           colour[v] : 0 = unassigned, else a colour in 1..m
 *           domain[v] : an m-bit mask of colours still allowed at v
 *         Descend the vertices with an EXPLICIT stack (no unbounded C recursion,
 *         so V = 20000 is safe). At vertex v, for each colour still in domain[v]:
 *           - assign colour[v]; FORWARD CHECK: for each unassigned neighbour u,
 *             clear that colour bit from domain[u] (a WRITE) and push (u, bit) on
 *             a trail. If any neighbour's domain becomes empty, this colour is
 *             dead -> undo just these prunes and try the next colour.
 *           - otherwise recurse to v+1.
 *         On BACKTRACK, pop the trail back to v's mark and RESTORE every pruned
 *         bit (a WRITE), unassign colour[v], and try v's next colour.
 *      3. Success when all V vertices are assigned (a full proper colouring).
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A large, irregular write front over two arrays: colour[V] (assign/unassign)
 *      and domain[V] (bits pruned on the way down, restored on the way up), driven
 *      by the graph adjacency. The trail (a stack of undo records) is written and
 *      rewound in lockstep. The adjacency (row_ptr + col_idx, the larger buffers)
 *      is READ-ONLY during the search and thus invisible; the visible footprint is
 *      the colour + domain working state being churned pass after pass.
 *
 *  DISTINCTION FROM NEIGHBOURING KERNELS:
 *    - vs D9 label_prop (graph traversal, write-visible): that is MONOTONE
 *      relaxation -- a label array is overwritten with a running minimum and never
 *      undone. This is BACKTRACKING: domains are pruned AND restored (non-monotone,
 *      LIFO undo via a trail), which is a different, reversible write pattern.
 *    - vs the grid maze backtrack (the other large-working-state D11 instance):
 *      that churns a 2D GRID of cells. This churns a per-vertex DOMAIN table over
 *      a GRAPH adjacency list -- topologically distinct working state (irregular
 *      neighbours, not a fixed 4-/8-connected lattice).
 *    - vs the sibling nqueens_count (D11 IDLE control): that only COUNTS leaves
 *      into a scalar (invisible writes). This maintains and rewrites a large
 *      colour + domain working set (visible writes).
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  (An optional --dump-coloring writes the edge list + final colours ONCE for an
 *  external verifier; it is benign, off by default, and NOT in the measured loop.)
 *  Signature family: KERNEL (visible). Dwarf: Backtrack/Branch-and-Bound.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (build graph) / measure (repeated colouring passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_graph_coloring_v2";

/* Colour palette is held as an m-bit mask in a uint32_t, so m must fit in 32. */
#define GC_MAX_COLORS 32

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign graph m-colouring; backtracking + forward-checking kernel)\n"
"  --vertices V          Number of graph vertices (default 20000)\n"
"  --degree d            Average degree (default 6)\n"
"  --colors m            Number of colours in the palette (default 6; range 1..32). Small m (near the chromatic threshold) forces GENUINE backtracking; large m (> max degree) makes it greedy/backtrack-light.\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dump-coloring PATH  (verifier only) write edge list + final colours to PATH once, then continue\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* Colouring context. The two dominant buffers (colour + domain) are the visible
 * working state; the CSR adjacency is read-only; the trail is the undo stack for
 * forward checking; the explicit choice stack replaces C recursion. */
typedef struct {
    size_t   V;             /* vertex count                                     */
    int      m;             /* number of colours                                */
    uint32_t full;          /* low m bits set: the full colour palette as a mask */
    const int32_t *row_ptr; /* CSR row offsets (V+1), READ-ONLY during search    */
    const int32_t *col_idx; /* CSR neighbour ids,     READ-ONLY during search    */
    int32_t  *colour;       /* VISIBLE: colour[v] in 0(unassigned)..m            */
    uint32_t *domain;       /* VISIBLE: m-bit mask of colours still allowed at v */
    int32_t  *trail_v;      /* undo stack: vertex whose domain bit was cleared   */
    uint32_t *trail_bit;    /* undo stack: the colour bit that was cleared       */
    size_t    trail_top;    /* number of live entries on the trail               */
    size_t    trail_cap;    /* trail capacity (== total CSR half-edge slots)     */
    uint64_t  backtracks;   /* diagnostic: count of backtracks in the last pass  */
    uint64_t  bt_budget;    /* safety cap: abort a pass after this many backtracks */
    int       budget_hit;   /* set if the last pass aborted on the budget         */
} GC;

/* Forward-check the assignment colour c (bit cbit) at vertex v: remove c from the
 * domain of every still-unassigned neighbour, trailing each removal for later undo.
 * Returns 1 if all neighbour domains remain non-empty, 0 if some domain wiped out
 * (a dead end). Either way the caller undoes back to `mark` on failure/backtrack.
 * Only domain[] and the trail are written here; the adjacency is read. */
static int gc_forward_check(GC *g, int32_t v, uint32_t cbit) {
    size_t b  = (size_t)g->row_ptr[v];
    size_t en = (size_t)g->row_ptr[v + 1];
    for (size_t k = b; k < en; k++) {
        int32_t u = g->col_idx[k];
        if (g->colour[u] != 0) continue;          /* already assigned: not constrained here */
        if (g->domain[u] & cbit) {                /* c is still allowed at u -> prune it */
            g->domain[u] &= ~cbit;                /* VISIBLE domain write */
            g->trail_v[g->trail_top]   = u;       /* record for undo */
            g->trail_bit[g->trail_top] = cbit;
            g->trail_top++;
            if (g->domain[u] == 0u) return 0;     /* u has no colour left -> dead end */
        }
    }
    return 1;
}

/* Undo trail entries down to `mark`, restoring each pruned colour bit into the
 * neighbour domain it was removed from. This is the reversible (non-monotone)
 * write that distinguishes backtracking from the monotone label relaxation. */
static void gc_undo_to(GC *g, size_t mark) {
    while (g->trail_top > mark) {
        g->trail_top--;
        int32_t u = g->trail_v[g->trail_top];
        g->domain[u] |= g->trail_bit[g->trail_top];   /* VISIBLE domain restore */
    }
}

/* One colouring pass: reset the visible working state (colour[] + domain[]) and
 * run explicit-stack DFS over vertices in index order with forward checking.
 * Returns 1 if a full proper colouring was found, else 0. No C recursion: an
 * explicit per-depth "tried-colours" mask stack (tried[]) plus a per-depth trail
 * mark (mark[]) bound memory for large V. Each call fully rewrites colour[] +
 * domain[] and then churns them via prune-on-descend / restore-on-backtrack --
 * the large-working-state visible write. The adjacency is only READ. */
static int gc_color_pass(GC *g, uint32_t *tried, size_t *mark) {
    size_t V = g->V;
    g->trail_top = 0;
    g->backtracks = 0;
    g->budget_hit = 0;
    /* Per-pass reset: every vertex unassigned, every domain = full palette. An
     * O(V) rewrite of both visible arrays at the start of each pass. */
    for (size_t i = 0; i < V; i++) { g->colour[i] = 0; g->domain[i] = g->full; tried[i] = 0u; }

    /* Invariant maintained below: on every iteration, when we are about to CHOOSE
     * a colour for the vertex at `depth`, the trail sits exactly at mark[depth] --
     * i.e. NO domain prunes from any previously-tried colour at this depth, and
     * none from any deeper vertex, are still applied. mark[depth] is snapshotted
     * once, on first entry to that depth (tried[depth]==0), and the trail is undone
     * back to it before each retry (both when a colour fails its own forward check
     * AND when a deeper vertex exhausts and pops back up here). */
    size_t depth = 0;
    mark[0] = 0;                                             /* root enters with empty trail */
    while (1) {
        if (depth == V) return 1;                            /* all vertices assigned */
        int32_t v = (int32_t)depth;
        uint32_t avail = g->domain[v] & ~tried[depth];       /* untried, still in-domain */
        if (avail == 0u) {                                   /* no colour left at v */
            /* Exhausted this vertex: undo its own entry prunes (defensive; the
             * trail is already at mark[depth] by the invariant), unassign, reset
             * its tried set, and backtrack to the parent. */
            gc_undo_to(g, mark[depth]);
            g->colour[v] = 0;
            tried[depth] = 0u;
            if (depth == 0) return 0;                        /* exhausted the root */
            depth--;
            g->backtracks++;
            /* Retract the PARENT's currently-applied colour: undo its prunes back
             * to the parent's entry mark and unassign it, so on the next iteration
             * the parent chooses its next colour from a CLEAN domain state (this is
             * the fix for stale forward-check prunes across a retry). tried[parent]
             * still records which colours the parent has already tried. */
            gc_undo_to(g, mark[depth]);
            g->colour[depth] = 0;
            /* Safety net: with a well-sized palette (m > max degree) backtracking is
             * essentially never triggered, but a deliberately tight m on large V
             * could thrash chronologically. Cap the work so a pass ALWAYS terminates
             * and reports colored_all=0; the domain churn already happened, so the
             * write signal is preserved. */
            if (g->bt_budget && g->backtracks >= g->bt_budget) {
                g->budget_hit = 1;
                gc_undo_to(g, 0);                            /* fully rewind the trail */
                return 0;
            }
            continue;
        }
        uint32_t cbit = avail & (uint32_t)(-(int32_t)avail); /* lowest untried colour bit */
        tried[depth] |= cbit;                                /* mark it tried at this depth */
        g->colour[v] = (int32_t)(__builtin_ctz(cbit) + 1);   /* assign colour (1..m) */
        if (gc_forward_check(g, v, cbit)) {
            depth++;                                         /* consistent -> go deeper */
            tried[depth] = 0u;                               /* fresh vertex: nothing tried */
            mark[depth] = g->trail_top;                      /* snapshot its entry trail pos */
        } else {
            gc_undo_to(g, mark[depth]);                      /* wipe-out -> undo this colour */
            g->colour[v] = 0;
        }
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long vertices   = p2_get_i64(argc, argv, "--vertices", 20000);
    long long degree     = p2_get_i64(argc, argv, "--degree", 6);
    long long colors     = p2_get_i64(argc, argv, "--colors", 6);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dumppath = p2_get_str(argc, argv, "--dump-coloring", NULL);

    if (vertices < 4 || vertices > 100000000LL) { P2_LOG_ERR("vertices %lld out of range (4..1e8)", vertices); return 2; }
    if (degree < 1 || degree > 1024) { P2_LOG_ERR("degree %lld out of range (1..1024)", degree); return 2; }
    if (colors < 1 || colors > GC_MAX_COLORS) { P2_LOG_ERR("colors %lld out of range (1..%d)", colors, GC_MAX_COLORS); return 2; }

    size_t V = (size_t)vertices, D = (size_t)degree, M = (size_t)colors;
    /* Undirected graph: draw E = V*d/2 random edges so the AVERAGE degree is d
     * (each edge contributes a half-edge to both endpoints). The CSR holds up to
     * 2*E neighbour slots. The trail can hold at most one entry per half-edge
     * pruned along a live root-to-leaf path; 2*E slots is a safe upper bound. */
    size_t E = (V * D) / 2; if (E < 1) E = 1;
    size_t half_edges = 2 * E;

    size_t rp_bytes    = (V + 1) * sizeof(int32_t);   /* row_ptr                     */
    size_t col_bytes   = half_edges * sizeof(int32_t);/* col_idx (read-only)         */
    size_t colour_bytes= V * sizeof(int32_t);         /* colour[] (VISIBLE)          */
    size_t domain_bytes= V * sizeof(uint32_t);        /* domain[] (VISIBLE)          */
    size_t tv_bytes    = half_edges * sizeof(int32_t);/* trail_v (undo stack)        */
    size_t tb_bytes    = half_edges * sizeof(uint32_t);/* trail_bit (undo stack)     */
    size_t total_bytes = rp_bytes + col_bytes + colour_bytes + domain_bytes + tv_bytes + tb_bytes;
    if (total_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("total bytes %zu exceed --max-mb %lld", total_bytes, max_mb); return 2;
    }

    p2_meta_t meta;
    p2_meta_open(&meta, outdir, TEST);
    p2_meta_kv_str(&meta, "test_name", TEST);
    p2_meta_kv_str(&meta, "language", "C");
    p2_meta_kv_str(&meta, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&meta, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&meta, "dwarf", "D11 Backtrack/Branch-and-Bound");
    p2_meta_kv_str(&meta, "scheme", "graph m-colouring via backtracking + forward checking (colour + domain working state churned with prune/restore)");
    p2_meta_kv_str(&meta, "visibility_source", "(b) large working state: colour[] + per-vertex domain[] over a big vertex set");
    p2_meta_kv_str(&meta, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&meta, "vertices", vertices);
    p2_meta_kv_i64(&meta, "degree", degree);
    p2_meta_kv_i64(&meta, "colors", colors);
    p2_meta_kv_u64(&meta, "edges_target", E);
    p2_meta_kv_u64(&meta, "half_edges_cap", half_edges);
    p2_meta_kv_u64(&meta, "total_bytes", total_bytes);
    p2_meta_kv_i64(&meta, "duration_s", duration_s);
    p2_meta_kv_i64(&meta, "warmup_s", warmup_s);
    p2_meta_kv_u64(&meta, "seed", seed);
    p2_meta_kv_i64(&meta, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&meta, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&meta, "status", "dry_run"); p2_meta_close(&meta); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* mmap all six regions. colour[] + domain[] are the VISIBLE churned working
     * state; row_ptr + col_idx are the read-only adjacency; trail_v + trail_bit
     * are the LIFO undo stack written/rewound in lockstep with the search. */
    int32_t  *row_ptr   = (int32_t  *)mmap(NULL, rp_bytes,     PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    int32_t  *col_idx   = (int32_t  *)mmap(NULL, col_bytes,    PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    int32_t  *colour    = (int32_t  *)mmap(NULL, colour_bytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    uint32_t *domain    = (uint32_t *)mmap(NULL, domain_bytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    int32_t  *trail_v   = (int32_t  *)mmap(NULL, tv_bytes,     PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    uint32_t *trail_bit = (uint32_t *)mmap(NULL, tb_bytes,     PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (row_ptr == MAP_FAILED || col_idx == MAP_FAILED || colour == MAP_FAILED ||
        domain == MAP_FAILED || trail_v == MAP_FAILED || trail_bit == MAP_FAILED) {
        P2_LOG_ERR("mmap failed: %s", strerror(errno));
        if (row_ptr   != MAP_FAILED) munmap(row_ptr, rp_bytes);
        if (col_idx   != MAP_FAILED) munmap(col_idx, col_bytes);
        if (colour    != MAP_FAILED) munmap(colour, colour_bytes);
        if (domain    != MAP_FAILED) munmap(domain, domain_bytes);
        if (trail_v   != MAP_FAILED) munmap(trail_v, tv_bytes);
        if (trail_bit != MAP_FAILED) munmap(trail_bit, tb_bytes);
        p2_meta_kv_str(&meta, "status", "mmap_failed"); p2_meta_close(&meta); return 1;
    }
    p2_madvise(row_ptr,   rp_bytes,     MADV_NOHUGEPAGE);
    p2_madvise(col_idx,   col_bytes,    MADV_NOHUGEPAGE);
    p2_madvise(colour,    colour_bytes, MADV_NOHUGEPAGE);
    p2_madvise(domain,    domain_bytes, MADV_NOHUGEPAGE);
    p2_madvise(trail_v,   tv_bytes,     MADV_NOHUGEPAGE);
    p2_madvise(trail_bit, tb_bytes,     MADV_NOHUGEPAGE);
    if (!no_mlock) {
        p2_mlock_soft(row_ptr,   rp_bytes);
        p2_mlock_soft(col_idx,   col_bytes);
        p2_mlock_soft(colour,    colour_bytes);
        p2_mlock_soft(domain,    domain_bytes);
        p2_mlock_soft(trail_v,   tv_bytes);
        p2_mlock_soft(trail_bit, tb_bytes);
    }

    /* Explicit DFS scaffolding (heap; small relative to the mmap'd working set):
     *   tried[depth] : bitmask of colours already tried at the vertex at `depth`
     *   mark[depth]  : trail_top snapshot on entry to `depth` (undo target)     */
    uint32_t *tried = (uint32_t *)malloc(V * sizeof(uint32_t));
    size_t   *mark  = (size_t   *)malloc(V * sizeof(size_t));
    if (!tried || !mark) {
        P2_LOG_ERR("dfs scaffold malloc failed");
        free(tried); free(mark);
        munmap(row_ptr, rp_bytes); munmap(col_idx, col_bytes);
        munmap(colour, colour_bytes); munmap(domain, domain_bytes);
        munmap(trail_v, tv_bytes); munmap(trail_bit, tb_bytes);
        p2_meta_kv_str(&meta, "status", "alloc_failed"); p2_meta_close(&meta); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Build ONE random UNDIRECTED sparse graph in symmetric CSR, once, in warmup.
     * Draw E edges u--v (u != v); each contributes a half-edge at both endpoints.
     * Two-pass counting sort: (a) count degrees, (b) prefix-sum into row_ptr,
     * (c) scatter both half-edges into col_idx. Self-loops are skipped by redraw.
     * The adjacency is never written again after this -> read-only/invisible. */
    int32_t *esrc = (int32_t *)malloc(E * sizeof(int32_t));
    int32_t *edst = (int32_t *)malloc(E * sizeof(int32_t));
    if (!esrc || !edst) {
        P2_LOG_ERR("edge-list malloc failed");
        free(esrc); free(edst); free(tried); free(mark);
        munmap(row_ptr, rp_bytes); munmap(col_idx, col_bytes);
        munmap(colour, colour_bytes); munmap(domain, domain_bytes);
        munmap(trail_v, tv_bytes); munmap(trail_bit, tb_bytes);
        p2_meta_kv_str(&meta, "status", "alloc_failed"); p2_meta_close(&meta); return 1;
    }
    for (size_t i = 0; i <= V; i++) row_ptr[i] = 0;
    for (size_t e = 0; e < E; e++) {
        int32_t u = (int32_t)(p2_rng_next(&rng) % (uint64_t)V);
        int32_t v = (int32_t)(p2_rng_next(&rng) % (uint64_t)V);
        if (u == v) { v = (int32_t)((u + 1) % (int32_t)V); }   /* avoid self-loop */
        esrc[e] = u; edst[e] = v;
        row_ptr[u + 1]++;                                      /* half-edge u->v */
        row_ptr[v + 1]++;                                      /* half-edge v->u */
    }
    for (size_t i = 0; i < V; i++) row_ptr[i + 1] += row_ptr[i];   /* prefix sum */
    size_t total_half = (size_t)row_ptr[V];                        /* actual half-edges */
    int32_t *cursor = (int32_t *)malloc(V * sizeof(int32_t));
    if (!cursor) {
        P2_LOG_ERR("cursor malloc failed");
        free(esrc); free(edst); free(tried); free(mark);
        munmap(row_ptr, rp_bytes); munmap(col_idx, col_bytes);
        munmap(colour, colour_bytes); munmap(domain, domain_bytes);
        munmap(trail_v, tv_bytes); munmap(trail_bit, tb_bytes);
        p2_meta_kv_str(&meta, "status", "alloc_failed"); p2_meta_close(&meta); return 1;
    }
    for (size_t i = 0; i < V; i++) cursor[i] = row_ptr[i];
    for (size_t e = 0; e < E; e++) {
        int32_t u = esrc[e], v = edst[e];
        col_idx[cursor[u]++] = v;
        col_idx[cursor[v]++] = u;
    }
    free(esrc); free(edst); free(cursor);

    GC g;
    g.V = V; g.m = (int)M;
    g.full = (M >= 32) ? 0xFFFFFFFFu : ((1u << M) - 1u);
    g.row_ptr = row_ptr; g.col_idx = col_idx;
    g.colour = colour; g.domain = domain;
    g.trail_v = trail_v; g.trail_bit = trail_bit;
    g.trail_top = 0; g.trail_cap = half_edges;
    g.backtracks = 0;
    /* Backtrack budget: a SEARCH-EFFORT cap, not a completeness guarantee. At the
     * default m=6 (near the chromatic threshold) the search genuinely backtracks and
     * a large graph will often hit this cap; that is the intended heavy-churn regime.
     * The cap bounds each pass to ~O(budget) work (well under a second even for large
     * V at ~tens of millions of backtracks/sec) so a pass ALWAYS terminates instead
     * of thrashing unboundedly, and --duration is respected. When it fires, the pass
     * reports colored_all=0 and budget_hit=1 (an honest "search truncated" -- the
     * prune/restore domain churn, i.e. the write signal, already happened, and the
     * partial colour[] left behind is still conflict-free because forward checking
     * never assigns a colour that clashes with an assigned neighbour). Generous so
     * genuinely-colourable instances that need moderate backtracking still finish;
     * scales with V for proportional slack. A large m (> max degree) instead gives a
     * near-instant greedy regime with ~0 backtracks, where this never fires. */
    g.bt_budget = (uint64_t)V * 100 + 100000;
    g.budget_hit = 0;

    /* Warm the code path + fault in the visible pages: run one full colouring pass
     * so colour[], domain[], and the trail are resident before timing starts. */
    int warm_ok = gc_color_pass(&g, tried, mark);
    double t_warmup_end = p2_monotonic();

    /* Optional benign one-shot dump for the EXTERNAL verifier. Off by default,
     * happens once here (NOT in the measured loop): write the edge list and the
     * final colour[] so an independent checker can prove the colouring is proper.
     * This is plain compute-artifact output, not persistence of user data. */
    if (dumppath && dumppath[0]) {
        FILE *df = fopen(dumppath, "w");
        if (!df) {
            P2_LOG_WARN("dump-coloring open failed: %s (%s); continuing", dumppath, strerror(errno));
        } else {
            /* header: V m half_edges colored_all */
            fprintf(df, "%zu %d %zu %d\n", V, (int)M, total_half, warm_ok ? 1 : 0);
            /* colours: v colour[v]  (colour 0 = unassigned, else 1..m) */
            for (size_t i = 0; i < V; i++) fprintf(df, "c %zu %d\n", i, colour[i]);
            /* edges: each undirected edge once (u < v) from the CSR */
            for (size_t u = 0; u < V; u++) {
                size_t b = (size_t)row_ptr[u], en = (size_t)row_ptr[u + 1];
                for (size_t k = b; k < en; k++) {
                    int32_t w = col_idx[k];
                    if ((size_t)w > u) fprintf(df, "e %zu %d\n", u, w);
                }
            }
            fclose(df);
            P2_LOG_INFO("dump-coloring written: %s (colored_all=%d)", dumppath, warm_ok ? 1 : 0);
        }
    }

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t backtracks_last = 0;
    int colored_all = warm_ok;
    int any_budget_hit = g.budget_hit;      /* carry the warmup pass result */
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One colouring pass: reset the visible working state, then run the
         * explicit-stack forward-checking backtracking search. Each pass fully
         * rewrites colour[] + domain[] (the O(V) reset) and then CHURNS them via
         * prune-on-descend / restore-on-backtrack -- the large-working-state
         * write signature. The adjacency is only READ. */
        colored_all = gc_color_pass(&g, tried, mark);
        backtracks_last = g.backtracks;
        if (g.budget_hit) any_budget_hit = 1;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile int sink = colour[V / 2];              /* a live colour; prevent DCE */
    volatile uint32_t sink_dom = domain[V / 2];     /* keep the domain array live */

    free(tried); free(mark);
    munmap(row_ptr, rp_bytes);
    munmap(col_idx, col_bytes);
    munmap(colour, colour_bytes);
    munmap(domain, domain_bytes);
    munmap(trail_v, tv_bytes);
    munmap(trail_bit, tb_bytes);

    p2_meta_kv_f64(&meta, "warmup_t0_s", t0);
    p2_meta_kv_f64(&meta, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&meta, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&meta, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&meta, "edges", total_half / 2);
    p2_meta_kv_u64(&meta, "half_edges", total_half);
    p2_meta_kv_u64(&meta, "passes", passes);
    p2_meta_kv_i64(&meta, "colored_all", colored_all ? 1 : 0);
    p2_meta_kv_u64(&meta, "backtracks", backtracks_last);
    p2_meta_kv_u64(&meta, "bt_budget", g.bt_budget);
    p2_meta_kv_i64(&meta, "budget_hit", any_budget_hit ? 1 : 0);
    p2_meta_kv_i64(&meta, "sink", (long long)sink);
    p2_meta_kv_u64(&meta, "sink_domain", (unsigned long long)sink_dom);
    p2_meta_kv_str(&meta, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&meta, "end_time", tend);
    p2_meta_kv_str(&meta, "known_limitations",
                   "WRITE-VISIBLE large working state: colour[] + domain[] churned via forward-checking prune/restore (the restore-on-backtrack is the distinct write vs monotone label_prop); adjacency read-only/invisible. At the default m=6 (near the chromatic threshold) the search genuinely backtracks and a large graph typically hits bt_budget (then colored_all=0, budget_hit=1, but the partial colouring is still conflict-free); this is a solver-under-budget outcome, not an error. A large m (> max degree) instead gives a backtrack-light greedy regime (colored_all=1). Either way visibility is the working-set churn, not a stored solution.");
    p2_meta_close(&meta);
    return 0;
}
