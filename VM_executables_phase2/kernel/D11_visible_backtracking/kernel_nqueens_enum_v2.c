/* kernel_nqueens_enum_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  N-QUEENS ENUMERATOR:  materialise every non-attacking placement to a buffer
 * ============================================================================
 *
 *  DWARF   : Backtrack / Branch-and-Bound (D11)   (Berkeley computational motif)
 *  FAMILY  : KERNEL (visible)                      (first-division memory label)
 *  PURPOSE : Visibility source (a) -- MATERIALISE THE OUTPUT. Run the exact same
 *            depth-first backtracking search as the QUIET control
 *            kernel_nqueens_count_v2, but instead of folding each solved board
 *            into a scalar counter, APPEND the full board (n column-indices) into
 *            a large mmap'd solutions buffer. The write footprint becomes
 *            O(#solutions x n) instead of O(1). This is the built proof that the
 *            RETURN TYPE alone -- count vs store -- flips a workload from quiet to
 *            write-visible, with the search itself held fixed.
 *
 *  PICTURE (top view):
 *      DFS explores a board...            ...every full board is appended:
 *          . Q . .                          solutions buffer (mmap + mlock)
 *          . . . Q                          +----+----+----+----+----+---
 *          Q . . .        depth == n        | b0 | b1 | b2 | b3 | b4 | ..   (bulk
 *          . . Q .    -- solution found --> +----+----+----+----+----+---   sequential
 *          pos[] = {1,3,0,2}                  each bi = n bytes, one uint8/row    append)
 *
 *  ALGORITHM (per enumerate pass):
 *      1. Three-bitmask DFS over rows. At row r, the free columns are the bits
 *         NOT set in (cols | diag1 | diag2): cols = columns already used,
 *         diag1 = "/" diagonals (indexed by row+col), diag2 = "\" diagonals
 *         (indexed by row-col). Pick a free column c, set pos[r]=c, recurse to
 *         row r+1 with the three masks updated (diag masks shifted by one row).
 *      2. At depth == n a full non-attacking board exists in pos[0..n-1]. APPEND
 *         its n column-indices (n bytes) to the tail of the solutions buffer:
 *         one bulk, sequential store of n bytes. Advance the write cursor.
 *      3. If appending the next board would pass the buffer capacity, STOP with a
 *         clear error asking for a larger --max-mb. We never silently truncate:
 *         validation requires #stored == count(n) exactly (OEIS A000170).
 *
 *  MEMORY SIGNATURE (what the host write-signal sees):
 *      A large, monotonically growing sequential write front into the solutions
 *      buffer -- the whole solution SET is laid down, n bytes at a time, in board
 *      order. The buffer is the dominant mmap+mlock region and is refilled front
 *      to back on every measure pass. The search's own scratch (pos[n] plus three
 *      scalar masks in registers/stack) is tiny; the buffer append is the tell.
 *
 *  CONTRAST WITH THE QUIET CONTROL (kernel_nqueens_count_v2):
 *      The count control walks the IDENTICAL three-bitmask DFS but at depth == n
 *      it does `count++` -- a single scalar, invisible to a write-signal (CPU/IDLE
 *      class). This enumerator changes exactly ONE thing: the leaf action becomes
 *      "store the board" instead of "increment a counter". Same branch factor,
 *      same pruning, same node count; only the RETURN TYPE (a u64 count vs an
 *      O(#solutions x n) byte array) differs. That single switch is what moves
 *      D11 from quiet to visible, and this file is the visible half of that pair.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  (The OPTIONAL --dump-solutions flag writes the raw buffer to one file for an
 *  offline verifier ONLY; it runs once after the measured loop and is never part
 *  of it.)
 *  Signature family: KERNEL (visible). Dwarf: Backtrack/Branch-and-Bound.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (enumerate passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_nqueens_enum_v2";

/* Board-size clamp. n in [1,16]: n=16 has 14772512 solutions (~236 MiB at 16
 * bytes/board), which is the practical ceiling for a mlock'd buffer here. */
#define NQ_MIN 1
#define NQ_MAX 16

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign N-queens enumerator; backtracking kernel, visible)\n"
"  --n N                 Board size / queen count (default 14; clamped to 1..16)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (accepted for interface parity; search is deterministic) (default 42)\n"
"  --max-mb N            Hard cap on solutions-buffer bytes; capacity = max_bytes / n (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dump-solutions PATH Write the raw solutions bytes to PATH once (offline verifier only; not measured)\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* ---------------------------------------------------------------------------
 * Enumerate context. The solutions buffer is a flat mmap'd byte array; every
 * complete board appends n bytes (one uint8 column-index per row) at `used`.
 *   buf   : solutions buffer base (mmap'd)         [dominant footprint]
 *   used  : bytes written so far this pass (append cursor)
 *   capB  : buffer capacity in bytes
 *   count : number of complete boards stored this pass
 *   n     : board size
 *   pos   : current partial placement, pos[r] = column chosen at row r
 *   overflow : set if a board could not fit (then we stop, never truncate)
 * --------------------------------------------------------------------------- */
typedef struct {
    uint8_t *buf;
    size_t   used;
    size_t   capB;
    uint64_t count;
    int      n;
    uint8_t  pos[NQ_MAX];
    int      overflow;
} NQ;

/* Depth-first placement over rows using three column/diagonal bitmasks.
 *   cols  : columns already occupied
 *   diag1 : "/" diagonals occupied, tracked in the column frame; shifted left
 *           by one when descending a row so it stays column-aligned
 *   diag2 : "\" diagonals occupied; shifted right by one per row descent
 * `free` is the set of columns not attacked by any earlier queen. We iterate the
 * set bits of `free` (isolate lowest set bit with x & -x) and recurse. At row n a
 * full board is in pos[] -> append it. Returns 0 on success, -1 if the buffer
 * overflowed (propagated up so enumeration stops cleanly without truncating). */
static int nq_solve(NQ *q, int row, unsigned cols, unsigned diag1, unsigned diag2) {
    const unsigned all = (q->n >= 32) ? ~0u : ((1u << q->n) - 1u);

    if (row == q->n) {
        /* Complete non-attacking board: bulk-append its n column indices. */
        if (q->used + (size_t)q->n > q->capB) { q->overflow = 1; return -1; }
        uint8_t *dst = q->buf + q->used;      /* sequential write front */
        for (int r = 0; r < q->n; r++) dst[r] = q->pos[r];
        q->used += (size_t)q->n;
        q->count++;
        return 0;
    }

    unsigned freecols = all & ~(cols | diag1 | diag2);
    while (freecols) {
        unsigned bit = freecols & (unsigned)(-(int)freecols);   /* lowest free column */
        freecols &= freecols - 1;                               /* clear it */
        /* column index of `bit` (0..n-1); used only to record the placement */
        int c = __builtin_ctz(bit);
        q->pos[row] = (uint8_t)c;
        /* descend: diag1 shifts left, diag2 shifts right so both stay aligned to
         * the next row's columns. Mask diag1 back to n bits to avoid stray highs. */
        if (nq_solve(q, row + 1,
                     cols | bit,
                     ((diag1 | bit) << 1) & all,
                     (diag2 | bit) >> 1) != 0) {
            return -1;   /* overflow: unwind without storing more */
        }
    }
    return 0;
}

/* Run one full enumeration pass: reset the append cursor and refill the buffer
 * front-to-back with every solution. Returns 0 on success, -1 on overflow. */
static int nq_enumerate(NQ *q) {
    q->used = 0;
    q->count = 0;
    q->overflow = 0;
    return nq_solve(q, 0, 0u, 0u, 0u);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long n_in       = p2_get_i64(argc, argv, "--n", 14);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dumppath = p2_get_str(argc, argv, "--dump-solutions", NULL);

    if (n_in < NQ_MIN || n_in > NQ_MAX) {
        P2_LOG_ERR("n %lld out of range (%d..%d)", n_in, NQ_MIN, NQ_MAX);
        return 2;
    }
    int n = (int)n_in;

    if (max_mb <= 0) {
        P2_LOG_ERR("max-mb %lld must be positive", max_mb);
        return 2;
    }
    /* Buffer capacity: cap boards = max_bytes / n; total bytes = cap * n. This is
     * the dominant mmap+mlock region. n=14 -> 365596 boards x 14 ~ 5 MiB fits the
     * 8192 MiB default with room to spare; the cap only bites for tiny --max-mb. */
    size_t max_bytes = (size_t)max_mb * 1024ULL * 1024ULL;
    size_t cap_boards = max_bytes / (size_t)n;
    if (cap_boards == 0) {
        P2_LOG_ERR("max-mb %lld too small to hold even one board of n=%d", max_mb, n);
        return 2;
    }
    size_t buf_bytes = cap_boards * (size_t)n;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D11 Backtrack/Branch-and-Bound");
    p2_meta_kv_str(&m, "scheme", "N-queens enumerator: three-bitmask DFS, stores every full board (vs quiet count control)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "n", n);
    p2_meta_kv_u64(&m, "buffer_capacity_boards", cap_boards);
    p2_meta_kv_u64(&m, "buffer_bytes", buf_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The solutions buffer is the dominant buffer -> mmap + mlock it. It is
     * refilled front-to-back every measure pass (the signature write). */
    uint8_t *buf = (uint8_t *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", buf_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(buf, buf_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(buf, buf_bytes);

    NQ q;
    q.buf = buf; q.used = 0; q.capB = buf_bytes; q.count = 0; q.n = n; q.overflow = 0;
    memset(q.pos, 0, sizeof(q.pos));
    (void)seed;   /* seed accepted for interface parity; the DFS is deterministic */

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Warm up by faulting/populating the buffer with one full enumeration. This
     * also lets us fail fast (before the timed loop) if --max-mb is too small. */
    int warm_rc = nq_enumerate(&q);
    if (warm_rc != 0) {
        P2_LOG_ERR("solutions buffer overflow at n=%d (stored %llu boards in %zu bytes); "
                   "raise --max-mb so capacity (%zu boards) >= count(n)",
                   n, (unsigned long long)q.count, q.capB, cap_boards);
        munmap(buf, buf_bytes);
        p2_meta_kv_u64(&m, "solutions_stored", q.count);
        p2_meta_kv_str(&m, "status", "buffer_overflow"); p2_meta_close(&m); return 1;
    }
    uint64_t solutions_stored = q.count;
    size_t   solutions_bytes  = q.used;
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    int meas_overflow = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One enumerate pass: reset the cursor and re-lay the whole solution set
         * front-to-back into the buffer (the dominant sequential write front). */
        if (nq_enumerate(&q) != 0) { meas_overflow = 1; break; }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Prevent dead-code elimination: touch one byte of the last stored board. */
    volatile uint8_t sink = (solutions_bytes > 0) ? buf[solutions_bytes - 1] : 0;

    /* OPTIONAL offline-verifier dump: a plain write of the raw solutions bytes to
     * one file, executed ONCE here, well outside the measured loop. Benign file
     * I/O for verification only; not part of the workload's behaviour signature. */
    if (dumppath && dumppath[0]) {
        FILE *df = fopen(dumppath, "wb");
        if (!df) {
            P2_LOG_WARN("dump-solutions open failed: %s (%s)", dumppath, strerror(errno));
        } else {
            size_t wrote = fwrite(buf, 1, solutions_bytes, df);
            if (wrote != solutions_bytes)
                P2_LOG_WARN("dump-solutions short write: %zu of %zu bytes", wrote, solutions_bytes);
            fclose(df);
            P2_LOG_INFO("dumped %zu solution bytes (%llu boards) to %s",
                        solutions_bytes, (unsigned long long)solutions_stored, dumppath);
        }
    }

    munmap(buf, buf_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "solutions_stored", solutions_stored);
    p2_meta_kv_u64(&m, "solutions_bytes", (unsigned long long)solutions_bytes);
    p2_meta_kv_i64(&m, "measure_overflowed", meas_overflow);
    p2_meta_kv_i64(&m, "last_byte", (int)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "visible via bulk append of the whole solution set (O(#solutions x n)); "
                   "the quiet control kernel_nqueens_count_v2 runs the same DFS but only count++ (scalar, invisible)");
    p2_meta_close(&m);
    return 0;
}
