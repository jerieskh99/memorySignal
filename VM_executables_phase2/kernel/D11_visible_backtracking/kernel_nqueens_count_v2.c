/* kernel_nqueens_count_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  N-QUEENS COUNTER:  bitmask depth-first backtracking that only COUNTS leaves
 * ============================================================================
 *
 *  DWARF   : Backtrack / Branch-and-Bound (D11)  (Berkeley 13 computational motif)
 *  FAMILY  : IDLE (quiet control)                (first-division memory-signature label)
 *  PURPOSE : The DELIBERATE QUIET CONTROL for D11. It demonstrates that a
 *            backtracking search which only COUNTS its solutions (a scalar
 *            reduction into one counter) is nearly invisible to a host
 *            memory-WRITE signal, even though it visits an EXPONENTIAL number
 *            of search-tree nodes. Exponential compute, near-zero writes --
 *            that contrast is the entire point.
 *
 *  PICTURE (top view):
 *      one n x n board, one solution           the DFS search tree it walks
 *      (n = 6 shown, a valid placement)         (branch = try a column; X = pruned
 *          . . . Q . .   row 0                   by the column/diagonal masks;
 *          Q . . . . .   row 1                   leaf @ depth n = one full board)
 *          . . . . Q .   row 2                            root
 *          . Q . . . .   row 3                          /  |  \  \
 *          . . . . . Q   row 4                        c0  c1  c2  ...   (row 0)
 *          . . Q . . .   row 5                        X  / \   X
 *      attacks masks (passed DOWN the                   c0  c3 ...      (row 1)
 *      recursion, never stored globally):               ...  \
 *        cols | diag-left | diag-right                       *leaf*     count += 1
 *
 *  ALGORITHM (bitmask backtracking, count at the leaves):
 *      Place one queen per row, top to bottom. Carry three integer bitmasks
 *      DOWN the recursion, one bit per board column that is currently attacked:
 *        cols : columns already occupied by a queen above
 *        dl   : squares hit by a "\" diagonal  (shift LEFT  one bit per row down)
 *        dr   : squares hit by a "/" diagonal  (shift RIGHT one bit per row down)
 *      The set of legal columns for this row is  free = full & ~(cols|dl|dr).
 *      For each set bit b in free, recurse with (cols|b, (dl|b)<<1, (dr|b)>>1).
 *      When depth == n every row is placed: that is one complete solution, so
 *      we INCREMENT A SINGLE 64-bit COUNTER. No board, no column list, no
 *      solution is ever stored. The live state is a handful of integers per
 *      stack frame plus the counter -- tiny and L1-resident by construction.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Almost nothing. The three masks live in registers / on the C stack and
 *      are consumed as they are produced; the only long-lived write target is
 *      the scalar solution counter. The exponential branching happens entirely
 *      in control flow and register traffic, which a WRITE-oriented host probe
 *      cannot see. This is the D11 analogue of the other IDLE quiet controls
 *      (spmv in D2, bfs in D9, mc_pi in D7): heavy inner work, invisible writes.
 *
 *  HONEST NOTE (why this is only the baseline):
 *      This kernel is the QUIET half of a matched pair. Its sibling
 *      nqueens_enum walks the IDENTICAL search tree with the IDENTICAL pruning,
 *      but at each leaf it STORES the solution (writes the n column indices into
 *      a growing results buffer) instead of merely counting it. That single
 *      change -- store vs count -- flips the same algorithm from IDLE (quiet)
 *      to VISIBLE, because the solution buffer becomes a large, streaming write
 *      front. Counting hides the work; storing reveals it.
 *
 *  To give the harness a (deliberately tiny) dominant buffer to mmap + mlock,
 *  we allocate a small per-row scratch array pos[n] (the column chosen at each
 *  depth). It is written during the search but is only n integers -- L1-sized
 *  by design, so it does not create a visible write front. It exists so the
 *  mmap/mlock plumbing has a real (if minute) region to hold.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: IDLE (quiet control). Dwarf: Backtrack/Branch-and-Bound.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (repeated full counts) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_nqueens_count_v2";

/* Hard clamp on board size. n > 16 explodes combinatorially (n=18 already visits
 * billions of nodes); we warn but still allow up to this ceiling. */
#define NQ_MIN 1
#define NQ_MAX 18
#define NQ_SLOW_WARN 16

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign N-Queens solution counter; backtracking kernel)\n"
"  --n N                 Board side length / number of queens (default 13; range 1..18)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (unused by the count; recorded for provenance) (default 42)\n"
"  --max-mb N            Hard cap on scratch bytes (default 8192; scratch is tiny by design)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* Backtracking context. The masks are passed by value down the recursion (they
 * are the search state); the only fields written through here are the scalar
 * solution counter and the tiny per-depth column scratch. */
typedef struct {
    uint64_t count;   /* THE scalar reduction target: number of full solutions   */
    int      n;       /* board side / queen count                                */
    uint32_t full;    /* low n bits set: the set of all columns                   */
    int     *pos;     /* tiny scratch: column chosen at each depth (n ints)       */
} NQ;

/* Depth-first count of non-attacking placements for rows [depth, n).
 *   cols : columns occupied by queens already placed (bit c set => column c used)
 *   dl   : "\"-diagonal attack mask, shifted LEFT one bit per row descended
 *   dr   : "/"-diagonal attack mask, shifted RIGHT one bit per row descended
 * Legal columns for this row are the clear bits of (cols|dl|dr) within full.
 * At depth == n the board is complete -> bump the single counter and return.
 * Nothing is stored: the recursion is pure control flow + register traffic. */
static void nq_count(NQ *q, int depth, uint32_t cols, uint32_t dl, uint32_t dr) {
    if (depth == q->n) {
        q->count++;                 /* the ONLY long-lived write: a scalar bump */
        return;
    }
    /* Columns that are free in this row: inside the board and unattacked. */
    uint32_t free = q->full & ~(cols | dl | dr);
    while (free) {
        uint32_t b = free & (uint32_t)(-(int32_t)free);   /* lowest set bit */
        free ^= b;                                        /* consume it */
        /* Record the column at this depth in the tiny scratch. This touches at
         * most n ints total on any root-to-leaf path -- an L1-resident write,
         * present only so the mmap/mlock buffer is genuinely used. It is NOT a
         * solution store: it is overwritten on every branch and never read out. */
        q->pos[depth] = __builtin_ctz(b);
        /* Descend: mark column b used, and spread the two diagonals by one row.
         * Masking with full keeps the shifted diagonals inside the board. */
        nq_count(q, depth + 1,
                 cols | b,
                 ((dl | b) << 1) & q->full,
                 ((dr | b) >> 1));
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long n_arg      = p2_get_i64(argc, argv, "--n", 13);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (n_arg < NQ_MIN || n_arg > NQ_MAX) {
        P2_LOG_ERR("n %lld out of range (%d..%d)", n_arg, NQ_MIN, NQ_MAX);
        return 2;
    }
    if (n_arg > NQ_SLOW_WARN) {
        P2_LOG_WARN("n=%lld is very slow (exponential node count); allowing but a single count may exceed --duration", n_arg);
    }
    int n = (int)n_arg;

    /* Scratch is deliberately tiny: one int per row (the column at each depth).
     * This is the "dominant" buffer only in the sense that it is the sole mmap'd
     * region -- by design it is a few dozen bytes, so the write footprint stays
     * L1-resident and the host write-signal sees essentially nothing. */
    size_t scratch_bytes = (size_t)n * sizeof(int);
    if (scratch_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("scratch bytes %zu exceed --max-mb %lld", scratch_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "IDLE (quiet control)");
    p2_meta_kv_str(&m, "dwarf", "D11 Backtrack/Branch-and-Bound");
    p2_meta_kv_str(&m, "scheme", "N-Queens count via bitmask DFS backtracking (scalar reduce at leaves; no solutions stored)");
    p2_meta_kv_str(&m, "quiet", "only a scalar counter is written (no solutions stored)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "n", n);
    p2_meta_kv_u64(&m, "scratch_bytes", scratch_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The tiny per-depth scratch is the sole mmap'd buffer -> mmap + mlock it so
     * the harness has a real (if minute) region to hold. It stays L1-sized. */
    int *pos = (int *)mmap(NULL, scratch_bytes, PROT_READ | PROT_WRITE,
                           MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (pos == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", scratch_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(pos, scratch_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(pos, scratch_bytes);

    NQ q = { 0, n, (n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1u)), pos };

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Warm the code path and the scratch page(s): run the full count once so the
     * recursion, the branch predictor, and the mlock'd scratch are all resident
     * before we start timing. (seed is unused by the deterministic count.) */
    (void)seed;
    for (int i = 0; i < n; i++) pos[i] = 0;
    q.count = 0;
    nq_count(&q, 0, 0u, 0u, 0u);
    uint64_t warm_count = q.count;          /* == the answer; sanity value */
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t solution_count = warm_count;   /* stable across passes (same n) */
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        q.count = 0;                        /* reset the scalar reduction */
        nq_count(&q, 0, 0u, 0u, 0u);        /* re-run the full backtracking count */
        solution_count = q.count;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile uint64_t sink = solution_count;   /* prevent dead-code elimination */
    volatile int sink_pos = pos[n - 1];        /* keep the scratch live too */

    munmap(pos, scratch_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "solution_count", (unsigned long long)sink);
    p2_meta_kv_i64(&m, "last_pos", (long long)sink_pos);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "quiet baseline: counting = scalar reduce, so writes are invisible despite exponential compute; sibling nqueens_enum flips to VISIBLE purely by STORING solutions instead of counting");
    p2_meta_close(&m);
    return 0;
}
