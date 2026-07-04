/* kernel_bnb_knapsack_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  0/1-KNAPSACK BRANCH-AND-BOUND:  a best-first frontier of subset prefixes
 * ============================================================================
 *
 *  DWARF   : Backtrack / Branch-and-Bound (D11)  (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL (write-visible)              (first-division memory-signature label)
 *  PURPOSE : Visibility source (c) FRONTIER (the OPEN set of live search nodes).
 *            This is the SECOND instance of that mechanism (the sibling bnb_tsp is
 *            the first): deliberately the SAME write mechanism -- churn an explicit
 *            priority queue of live nodes by push/pop -- on a DIFFERENT search
 *            object. Here a frontier node is an include/exclude SUBSET prefix over
 *            the item order (a knapsack partial), whereas bnb_tsp's frontier nodes
 *            are PERMUTATION prefixes over the cities. Showing the tell survives the
 *            change of object is the point: what the host write-signal sees is the
 *            MECHANISM (a heap-shaped open set repeatedly rewritten as it grows and
 *            shrinks), not the particular combinatorial problem.
 *
 *  PICTURE (top view):  the include/exclude choice tree feeds a bound-ordered
 *                       priority-queue frontier (a binary max-heap).
 *
 *      items sorted by value/weight (ratio) descending, then branched in order:
 *
 *        level 0                (root)                 priority queue (open set),
 *                              /       \               a binary max-heap ordered
 *        level 1        take i0        skip i0         by fractional-LP bound:
 *                       /     \        /     \
 *        level 2   take i1  skip i1  take i1 skip i1     [ nodeA  bound=142 ]  <- pop best
 *                    :        :        :       :         [ nodeB  bound=138 ]
 *      prune if  weight>W  (infeasible include)          [ nodeC  bound=131 ]
 *      prune if  bound <= best_value_so_far              [ nodeD  bound=127 ]
 *      else PUSH the child onto the heap.                        ...  grows / shrinks
 *
 *      bound(node) = value_so_far + fractional relaxation of the remaining items:
 *      greedily add whole items by ratio, then a FRACTION of the first that
 *      overflows W. That LP over-estimate is admissible (never below the true best
 *      reachable value), so pruning a child whose bound <= best is always safe.
 *
 *  ALGORITHM (best-first branch-and-bound):
 *      1. Draw n items (weight,value) from the harness RNG; capacity W. Sort the
 *         items by value/weight ratio DESCENDING (needed for the fractional bound).
 *      2. Seed the heap with the root node (level -1, weight 0, value 0, bound =
 *         full LP relaxation). Then repeat:
 *           - POP the node with the greatest bound (the most promising subset
 *             prefix). If its bound <= best_value, stop: nothing on the frontier
 *             can beat what we already have (best-first optimality).
 *           - BRANCH on the next item (level+1) into two children:
 *               INCLUDE: add its weight/value. Prune if weight > W; else update
 *                 best_value and, if its bound > best_value, PUSH it.
 *               EXCLUDE: same weight/value as the parent, recomputed bound.
 *                 If its bound > best_value, PUSH it.
 *      3. best_value is the optimum. It MUST equal the classic bottom-up 0/1
 *         knapsack DP over capacity (see kernel_knapsack_v2, dwarf D10) -- SAME
 *         answer, DIFFERENT write pattern: a frontier heap churned by push/pop
 *         here, versus a dense capacity table filled in place there.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      The priority-queue array (the open set / frontier) is the dominant mmap'd +
 *      mlock'd buffer. Best-first search pushes and pops nodes constantly, and each
 *      heap sift-up/sift-down overwrites a logarithmic path of slots, so the live
 *      region is a heap-shaped area that swells and drains as the search runs --
 *      an irregular, repeatedly-rewritten frontier. That churn is the tell, and it
 *      is the SAME mechanism as bnb_tsp even though the node payload (a subset
 *      prefix vs a tour prefix) differs. The item arrays and DP cross-check table
 *      are tiny next to the heap; the visible mass is the frontier.
 *      Honest caveat: a single push/pop touches only a logarithmic slice and is far
 *      finer-grained than a 500 ms snapshot, so a snapshot resolves the aggregate
 *      hot heap region and its growth/shrink, not any individual sift.
 *
 *  SIZING / SAFETY:
 *      Heap capacity (node count) is derived from --max-mb. If a push would exceed
 *      capacity we do NOT crash: we prune more aggressively instead (raise the
 *      effective bound floor by dropping the current worst-bound node), so the
 *      search stays correct-if-slower under a tight memory cap. best_value is never
 *      affected because only nodes that cannot beat the incumbent are discarded.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  (The optional --dump-instance path is a verifier-only convenience OUTSIDE the
 *  measured loop; the measured workload writes only anonymous mmap memory.)
 *  Signature family: KERNEL. Dwarf: Backtrack / Branch-and-Bound. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (re-seed + re-run B&B each pass) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_bnb_knapsack_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign 0/1 knapsack branch-and-bound; backtracking kernel)\n"
"  --items N             Number of items (default 40; clamped to [1,40])\n"
"  --capacity W          Knapsack capacity (default 1000)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on priority-queue bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --dump-instance PATH  Verifier-only: after measuring, write items+capacity+best to PATH\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* An item: integer weight and value, plus its precomputed value/weight ratio
 * (the sort key for the fractional bound). Items are re-seeded each pass. */
typedef struct {
    long long w, v;
    double    ratio;      /* v / w, used only to order items for the LP bound */
} Item;

/* A live search node = one include/exclude SUBSET prefix over the item order.
 *   level  : index of the LAST item decided (-1 at the root; branch item = level+1).
 *   weight : total weight of the items INCLUDED along this prefix.
 *   value  : total value  of the items INCLUDED along this prefix.
 *   bound  : fractional-relaxation UPPER BOUND on any completion of this prefix.
 * These nodes live in one flat mmap'd array used as a binary max-heap keyed on
 * `bound`; push/pop churn is the workload's signature write. */
typedef struct {
    int       level;
    long long weight;
    long long value;
    double    bound;
} Node;

/* Sort items by value/weight ratio DESCENDING (qsort comparator). */
static int item_cmp_desc(const void *a, const void *b) {
    double ra = ((const Item *)a)->ratio, rb = ((const Item *)b)->ratio;
    if (ra < rb) return 1;
    if (ra > rb) return -1;
    return 0;
}

/* Fractional (LP-relaxation) upper bound for a node at `level`, having packed
 * `value` into `weight` of capacity W. Greedily add WHOLE remaining items in
 * ratio order, then a FRACTION of the first item that would overflow W. Items
 * are assumed pre-sorted by ratio descending. This is an admissible over-estimate:
 * it never under-states the best value reachable from the node, so pruning a child
 * whose bound <= incumbent is always safe. */
static double bnb_bound(const Item *items, int n, long long W,
                        int level, long long weight, long long value) {
    if (weight >= W) return (double)value;          /* already full: no room to add */
    double b = (double)value;
    long long room = W - weight;
    for (int i = level + 1; i < n; i++) {
        if (items[i].w <= room) {                   /* whole item fits */
            room -= items[i].w;
            b    += (double)items[i].v;
        } else {                                     /* take the fractional part */
            b += (double)items[i].v * ((double)room / (double)items[i].w);
            break;                                   /* capacity exhausted */
        }
    }
    return b;
}

/* ---- Binary max-heap of Nodes, ordered by `bound` (the open set / frontier) ----
 * The heap array is the dominant mmap'd buffer. sift_up/sift_down each overwrite a
 * logarithmic path of slots; push/pop churn is the visible frontier write. */
typedef struct {
    Node    *a;        /* flat node array (mmap'd)              */
    size_t   size;     /* live nodes currently on the frontier  */
    size_t   cap;      /* array capacity in nodes               */
    uint64_t max_size; /* high-water mark of `size` (metadata)  */
} Heap;

static inline void heap_swap(Node *a, size_t i, size_t j) {
    Node t = a[i]; a[i] = a[j]; a[j] = t;
}

/* Restore the max-heap order upward from index i (after a push at the tail). */
static void heap_sift_up(Heap *h, size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (h->a[parent].bound >= h->a[i].bound) break;
        heap_swap(h->a, parent, i);
        i = parent;
    }
}

/* Restore the max-heap order downward from index i (after a pop at the root). */
static void heap_sift_down(Heap *h, size_t i) {
    for (;;) {
        size_t l = 2 * i + 1, r = 2 * i + 2, best = i;
        if (l < h->size && h->a[l].bound > h->a[best].bound) best = l;
        if (r < h->size && h->a[r].bound > h->a[best].bound) best = r;
        if (best == i) break;
        heap_swap(h->a, i, best);
        i = best;
    }
}

/* Pop and return the max-bound node. Caller guarantees size > 0. */
static Node heap_pop(Heap *h) {
    Node top = h->a[0];
    h->size--;
    if (h->size > 0) {
        h->a[0] = h->a[h->size];
        heap_sift_down(h, 0);
    }
    return top;
}

/* Push a node. On a FULL heap we do not crash: we prune the current worst-bound
 * leaf instead. Any leaf is a safe candidate to drop (heap leaves occupy the
 * second half of the array); dropping a low-bound node only slows the search and
 * never changes best_value, because best_value is tracked independently. Returns
 * 1 if the node ended up on the frontier, 0 if it (or a worse leaf) was pruned. */
static int heap_push(Heap *h, Node nd) {
    if (h->size >= h->cap) {
        /* Frontier is full. Find the worst-bound leaf and evict it if the new
         * node is more promising; otherwise drop the new node. */
        size_t first_leaf = h->cap / 2;             /* indices [first_leaf, cap) are leaves */
        size_t worst = first_leaf;
        for (size_t i = first_leaf + 1; i < h->size; i++)
            if (h->a[i].bound < h->a[worst].bound) worst = i;
        if (nd.bound <= h->a[worst].bound) return 0; /* new node no better: drop it */
        /* Overwrite the worst leaf, then re-heapify from there (its bound rose). */
        h->a[worst] = nd;
        heap_sift_up(h, worst);
        return 1;
    }
    size_t i = h->size++;
    h->a[i] = nd;
    heap_sift_up(h, i);
    if ((uint64_t)h->size > h->max_size) h->max_size = (uint64_t)h->size;
    return 1;
}

/* Run one full best-first branch-and-bound over the (already ratio-sorted) items.
 * Returns the optimum value; reports nodes expanded and the frontier high-water
 * mark through out-params. The heap array is reused across passes (size reset). */
static long long bnb_solve(Heap *h, const Item *items, int n, long long W,
                           uint64_t *out_nodes_expanded) {
    h->size = 0;
    h->max_size = 0;
    long long best = 0;
    uint64_t expanded = 0;

    Node root = { -1, 0, 0, bnb_bound(items, n, W, -1, 0, 0) };
    heap_push(h, root);

    while (h->size > 0) {
        Node cur = heap_pop(h);
        expanded++;
        /* Best-first optimality: the popped node has the greatest bound on the
         * whole frontier. If even it cannot beat the incumbent, nothing can. */
        if (cur.bound <= (double)best) break;
        int next = cur.level + 1;
        if (next >= n) continue;                    /* no item left to branch on */

        /* Child A -- INCLUDE item `next`. */
        long long w_in = cur.weight + items[next].w;
        if (w_in <= W) {                            /* feasible: prune if overweight */
            long long v_in = cur.value + items[next].v;
            if (v_in > best) best = v_in;           /* a full prefix is a real solution */
            double b_in = bnb_bound(items, n, W, next, w_in, v_in);
            if (b_in > (double)best) {
                Node c = { next, w_in, v_in, b_in };
                heap_push(h, c);
            }
        }

        /* Child B -- EXCLUDE item `next` (weight/value unchanged, bound recomputed). */
        double b_ex = bnb_bound(items, n, W, next, cur.weight, cur.value);
        if (b_ex > (double)best) {
            Node c = { next, cur.weight, cur.value, b_ex };
            heap_push(h, c);
        }
    }

    if (out_nodes_expanded) *out_nodes_expanded = expanded;
    return best;
}

/* Draw one fresh instance: n items with integer weight/value from the RNG, and
 * their value/weight ratios. Weights in [1, W] so a single item can fit; values
 * in [1, 1000] (same value scale as the D10 DP knapsack). */
static void gen_instance(p2_rng_t *rng, Item *items, int n, long long W) {
    /* Strongly-correlated instance (Pisinger): weights in [1,100], value =
     * weight + 10. The near-tied value/weight ratios defeat the fractional-
     * relaxation upper bound, so best-first B&B cannot prune down to a handful
     * of nodes -- it must keep a LARGE live frontier (the whole point of this
     * visibility-source-(c) member). Uncorrelated random values would let the
     * bound collapse the frontier to ~O(n) and the kernel would read as quiet. */
    (void)W;
    for (int i = 0; i < n; i++) {
        long long w = 1 + (long long)(p2_rng_next(rng) % 100ULL);
        items[i].w = w;
        items[i].v = w + 10;
        items[i].ratio = (double)items[i].v / (double)items[i].w;
    }
    qsort(items, (size_t)n, sizeof(Item), item_cmp_desc);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long items_n    = p2_get_i64(argc, argv, "--items", 40);
    long long capacity   = p2_get_i64(argc, argv, "--capacity", 1000);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dump     = p2_get_str(argc, argv, "--dump-instance", NULL);

    /* Keep n small: the DP cross-check builds an n*W table and brute force is 2^n,
     * so both stay feasible for validation. */
    if (items_n < 1 || items_n > 40) {
        P2_LOG_ERR("items %lld out of range (1..40)", items_n); return 2;
    }
    if (capacity < 1 || capacity > (1LL << 30)) {
        P2_LOG_ERR("capacity %lld out of range (1..2^30)", capacity); return 2;
    }
    int n = (int)items_n;
    long long W = capacity;

    /* Frontier heap capacity (in nodes) from --max-mb. A best-first knapsack
     * frontier can, in the worst case, hold O(2^n) nodes; we cap it and prune
     * more aggressively rather than overflow. Give it generous room but never
     * exceed the byte cap. */
    size_t max_bytes = (size_t)max_mb * 1024ULL * 1024ULL;
    size_t node_sz   = sizeof(Node);
    size_t heap_cap  = max_bytes / node_sz;
    if (heap_cap < 1024) {
        P2_LOG_ERR("--max-mb %lld too small for a usable frontier (need >= %zu bytes)",
                   max_mb, (size_t)1024 * node_sz);
        return 2;
    }
    /* No need to reserve billions of slots for tiny n: a knapsack B&B never holds
     * more than ~2 nodes per tree level times the levels, and the tree has 2^n
     * leaves. Cap the reservation at a sane ceiling to avoid a huge idle mapping,
     * but stay within the byte cap. */
    size_t want = 1;
    for (int i = 0; i < n && want < heap_cap; i++) {
        if (want > heap_cap / 4) { want = heap_cap; break; }
        want *= 4;                                  /* ~4 live nodes per branched level */
    }
    if (want < 4096) want = 4096;                   /* floor so small n still churns a real heap */
    if (want > heap_cap) want = heap_cap;
    heap_cap = want;
    size_t heap_bytes = heap_cap * node_sz;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D11 Backtrack/Branch-and-Bound");
    p2_meta_kv_str(&m, "family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "scheme", "0/1 knapsack best-first branch-and-bound; priority-queue frontier (subset-prefix nodes) with fractional-LP upper bound");
    p2_meta_kv_str(&m, "visibility_source", "(c) frontier (open set); 2nd instance, subset nodes vs bnb_tsp permutation nodes");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "items", items_n);
    p2_meta_kv_i64(&m, "capacity", capacity);
    p2_meta_kv_u64(&m, "heap_capacity_nodes", (unsigned long long)heap_cap);
    p2_meta_kv_u64(&m, "total_bytes", (unsigned long long)heap_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The priority-queue array is the dominant buffer -> mmap + soft-mlock it. It
     * is churned by push/pop every pass, which is the workload's signature write. */
    Node *heap_a = (Node *)mmap(NULL, heap_bytes, PROT_READ | PROT_WRITE,
                                MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (heap_a == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", heap_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(heap_a, heap_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(heap_a, heap_bytes);

    /* Compact item array (re-seeded each pass; tiny next to the frontier heap). */
    Item *items = (Item *)malloc((size_t)n * sizeof(Item));
    if (!items) { munmap(heap_a, heap_bytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    Heap heap = { heap_a, 0, heap_cap, 0 };

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    gen_instance(&rng, items, n, W);                /* first instance (touch pages) */
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    long long best_value = 0;
    uint64_t  nodes_expanded = 0;
    uint64_t  max_frontier_size = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Fresh instance each pass: new weights/values (re-sorted by ratio) so the
         * search is a genuine recomputation, never a cache-warm rerun. */
        gen_instance(&rng, items, n, W);
        uint64_t expanded = 0;
        best_value = bnb_solve(&heap, items, n, W, &expanded);
        nodes_expanded = expanded;
        max_frontier_size = heap.max_size;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Verifier-only side channel (OUTSIDE the measured window): dump the LAST
     * pass's instance -- items (weight,value), capacity, and the B&B best_value --
     * to a plain file so an independent checker can run the DP and brute force and
     * assert equality. Benign local write; not part of the measured signature. */
    int dump_ok = -1;
    if (dump) {
        FILE *df = fopen(dump, "w");
        if (!df) {
            P2_LOG_WARN("dump-instance open failed: %s (%s)", dump, strerror(errno));
        } else {
            /* Line 1: n capacity best_value.  Then n lines: weight value. */
            fprintf(df, "%d %lld %lld\n", n, W, best_value);
            for (int i = 0; i < n; i++)
                fprintf(df, "%lld %lld\n", items[i].w, items[i].v);
            if (fclose(df) != 0) {
                P2_LOG_WARN("dump-instance write/close failed: %s", dump);
                dump_ok = 0;
            } else {
                dump_ok = 1;
                P2_LOG_INFO("dumped instance (%d items, W=%lld, best=%lld) to %s",
                            n, W, best_value, dump);
            }
        }
    }

    /* Prevent dead-code elimination of the frontier buffer: sample a live slot. */
    volatile double sink = heap.a[0].bound;
    (void)sink;

    free(items);
    munmap(heap_a, heap_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_i64(&m, "best_value", best_value);
    p2_meta_kv_u64(&m, "nodes_expanded", nodes_expanded);
    p2_meta_kv_u64(&m, "max_frontier_size", max_frontier_size);
    if (dump) p2_meta_kv_i64(&m, "dump_ok", dump_ok);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "optimum cross-checks against the D10 DP knapsack (same value); the distinct write is frontier heap churn (push/pop over subset-prefix nodes) vs the DP's dense table fill; a 500ms snapshot sees the aggregate hot heap, not a single sift");
    p2_meta_close(&m);
    return 0;
}
