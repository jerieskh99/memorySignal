/* kernel_smithwaterman_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Dynamic-Programming dwarf (Berkeley motif D10), the Smith-Waterman variant:
 * LOCAL sequence alignment of two strings a[] and b[] over a tiny DNA-like
 * alphabet. A score matrix H of shape (N+1) x (M+1) is filled by the same
 * strictly-ordered row-major wavefront as kernel_dp_v2 -- every cell depends on
 * its up, left, and diagonal neighbours, so the live write band is one row wide
 * and marches down the table. The recurrence differs from edit-distance in that
 * scores are clamped at zero (a local alignment may start fresh anywhere):
 *
 *       H[0][*] = H[*][0] = 0
 *       H[i][j] = max( 0,
 *                      H[i-1][j-1] + (a[i-1]==b[j-1] ? MATCH : MISMATCH),
 *                      H[i-1][j]   + GAP,
 *                      H[i][j-1]   + GAP )
 *
 * with MATCH=+2, MISMATCH=-1, GAP=-2. The single best-scoring cell is the tail
 * of the strongest local alignment.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs kernel_dp_v2)
 * ----------------------------------------------------------------------------
 * Honest disclosure: the FILL phase here is essentially the same write pattern
 * as kernel_dp_v2 -- a monotone, row-major sweep over a large int32 matrix. On
 * its own it would be indistinguishable to a host write-signal. The distinct
 * element is the TRACEBACK: after the fill, we start at the maximum cell and
 * walk BACKWARD along the neighbour that produced each score, stopping at the
 * first zero cell, and stamp the recovered path into H (we negate the visited
 * cells so the path is a real second write). This backward walk is a short,
 * data-dependent, ANTI-diagonal write that reverses the direction of the main
 * front -- a "fill down, then trace back up" two-phase shape the plain forward
 * DP does not have. See the MEMORY SIGNATURE caveat below for its limits.
 *
 * ============================================================================
 * ALGORITHM (per measure pass over the whole table)
 * ============================================================================
 *   1. Re-randomise both sequences a[] (length N) and b[] (length M) over the
 *      DNA-like alphabet, so every pass is a genuinely fresh alignment problem
 *      (never a cache-warm rerun of identical data).
 *   2. Zero the boundary: row 0 and column 0 of H are all zero (a local
 *      alignment may begin against any position, so empty prefixes score 0).
 *   3. FILL rows i = 1..N left-to-right. Each interior cell takes the best of
 *      three moves, then clamps at zero:
 *          diag = H[i-1][j-1] + (a[i-1]==b[j-1] ? MATCH : MISMATCH)
 *          up   = H[i-1][j]   + GAP
 *          left = H[i][j-1]   + GAP
 *          H[i][j] = max(0, diag, up, left)
 *      While filling, track the single maximum cell value and its (i,j).
 *   4. TRACEBACK from the max cell. At each step recompute which of the three
 *      predecessors produced the current score and move to it, stamping the
 *      path into H, until a zero cell is reached. The number of visited cells
 *      is the local-alignment path length.
 *
 * ============================================================================
 * MEMORY SIGNATURE (what the host write-signal actually sees)
 * ============================================================================
 *   Phase A (fill): a large buffer written by a MONOTONE row-major front, one
 *   thin live band migrating steadily from the top of the table to the bottom
 *   -- identical in shape to kernel_dp_v2.
 *   Phase B (traceback): a short BACKWARD walk from the bottom-right region up
 *   toward a zero cell, writing a sparse anti-diagonal trail against the fill
 *   direction. This reversal is the tell that separates Smith-Waterman from the
 *   forward-only edit-distance DP.
 *   Honest caveat: the fill dominates the byte count, so the traceback is a
 *   faint, brief write compared to the fill -- and both are only resolvable as
 *   separate phases when a SINGLE pass is slow enough to span several 500 ms
 *   snapshots. On a small/fast table a whole pass completes inside one snapshot
 *   interval, the two phases collapse together, and the signature is hard to
 *   tell from kernel_dp_v2 -- hence the large default dimensions.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Dynamic Programming. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (fill + traceback passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_smithwaterman_v2";

/* Smith-Waterman local-alignment scores. MATCH rewards an identical symbol,
 * MISMATCH lightly penalises a substitution, GAP penalises an insertion or
 * deletion. With these three values the optimal local alignment of two strings
 * that share one identical block of length L (surrounded by non-matching
 * padding) is exactly that block, scoring L*MATCH. */
#define SW_MATCH     2
#define SW_MISMATCH  (-1)
#define SW_GAP       (-2)

/* Size of the DNA-like alphabet the sequences are drawn from (A,C,G,T). A small
 * alphabet makes chance matches common, so the filled matrix is dense with
 * short positive-score runs rather than mostly zero. */
#define SW_ALPHABET  4

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Smith-Waterman local alignment; Dynamic-Programming kernel)\n"
"  --len-a N             First sequence length  (default 4096; matrix is (N+1)*(M+1)*4 bytes)\n"
"  --len-b M             Second sequence length (default 4096)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on matrix bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* Maximum of four ints -- the inner recurrence of the fill (the fourth argument
 * is the zero floor that makes the alignment LOCAL). Kept as a tiny branch-only
 * helper (no library call) so the innermost loop stays cheap. */
static inline int32_t max4(int32_t a, int32_t b, int32_t c, int32_t d) {
    int32_t m = a > b ? a : b;
    m = m > c ? m : c;
    return m > d ? m : d;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long len_a      = p2_get_i64(argc, argv, "--len-a", 4096);
    long long len_b      = p2_get_i64(argc, argv, "--len-b", 4096);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (len_a < 16 || len_a > 65535) { P2_LOG_ERR("len-a %lld out of range (16..65535)", len_a); return 2; }
    if (len_b < 16 || len_b > 65535) { P2_LOG_ERR("len-b %lld out of range (16..65535)", len_b); return 2; }
    size_t N = (size_t)len_a;            /* rows of a[]; matrix has N+1 rows    */
    size_t M = (size_t)len_b;            /* cols of b[]; matrix has M+1 columns */
    size_t rows = N + 1, cols = M + 1;   /* +1 for the zeroed boundary row/col  */
    size_t cells = rows * cols;
    size_t bytes = cells * sizeof(int32_t);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("matrix bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Dynamic Programming");
    p2_meta_kv_str(&m, "scheme", "Smith-Waterman local alignment (row-major wavefront fill + backward traceback)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "len_a", len_a);
    p2_meta_kv_i64(&m, "len_b", len_b);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The score matrix is the dominant buffer -> mmap + mlock it. It is refilled
     * every pass, which (with the traceback) is the workload's signature write. */
    int32_t *H = (int32_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (H == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(H, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(H, bytes);

    /* The two sequences (compact byte arrays over the small alphabet). */
    uint8_t *a = (uint8_t *)malloc(N), *b = (uint8_t *)malloc(M);
    if (!a || !b) { free(a); free(b); munmap(H, bytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) a[i] = (uint8_t)(p2_rng_next(&rng) % SW_ALPHABET);
    for (size_t j = 0; j < M; j++) b[j] = (uint8_t)(p2_rng_next(&rng) % SW_ALPHABET);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    int32_t  best_score = 0;    /* max cell value of the LAST completed pass  */
    size_t   path_len   = 0;    /* traceback path length of the LAST pass     */
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Fresh alignment problem each pass: re-seed both sequences so the fill
         * is a real recomputation, not a cache-warm rerun of identical data. */
        for (size_t i = 0; i < N; i++) a[i] = (uint8_t)(p2_rng_next(&rng) % SW_ALPHABET);
        for (size_t j = 0; j < M; j++) b[j] = (uint8_t)(p2_rng_next(&rng) % SW_ALPHABET);

        /* Boundary: a local alignment may start against any position, so row 0
         * and column 0 are all zero (the score of aligning against an empty
         * prefix is zero, never negative). */
        for (size_t j = 0; j < cols; j++) H[j] = 0;                 /* first row    */
        for (size_t i = 0; i < rows; i++) H[i * cols] = 0;          /* first column */

        /* ---- Phase A: FILL. Row-major wavefront, rows top-to-bottom and cells
         * left-to-right within a row. Every cell reads its up (prev[j]), left
         * (row[j-1]) and diagonal (prev[j-1]) neighbours, all already written,
         * which is exactly what forbids the front from advancing out of order.
         * We also track the single best-scoring cell for the traceback start. */
        int32_t max_val = 0; size_t max_i = 0, max_j = 0;
        for (size_t i = 1; i < rows; i++) {
            int32_t *row = H + i * cols; const int32_t *prev = H + (i - 1) * cols;
            uint8_t ai = a[i - 1];                                  /* symbol a[i-1] pairs with row i */
            for (size_t j = 1; j < cols; j++) {
                int32_t sub  = (ai == b[j - 1]) ? SW_MATCH : SW_MISMATCH;
                int32_t diag = prev[j - 1] + sub;                  /* align a[i-1] with b[j-1] */
                int32_t up   = prev[j]     + SW_GAP;               /* gap in b (delete)        */
                int32_t left = row[j - 1]  + SW_GAP;               /* gap in a (insert)        */
                int32_t v = max4(0, diag, up, left);               /* clamp at 0 -> LOCAL      */
                row[j] = v;
                if (v > max_val) { max_val = v; max_i = i; max_j = j; }
            }
        }

        /* ---- Phase B: TRACEBACK. Start at the best cell and walk backward to a
         * zero cell, following the neighbour that produced each score. This is
         * the distinct, direction-reversing second write phase. We stamp the
         * path by negating each visited cell (a real write) and count its length.
         * Negating cannot be mistaken for a fill value: fill scores are >= 0. */
        size_t ti = max_i, tj = max_j; size_t steps = 0;
        while (ti > 0 && tj > 0) {
            int32_t v = H[ti * cols + tj];
            if (v <= 0) break;                                     /* reached the local-alignment start */
            steps++;
            int32_t sub  = (a[ti - 1] == b[tj - 1]) ? SW_MATCH : SW_MISMATCH;
            /* Recompute which predecessor produced v and move to it. The order
             * (diagonal, then up, then left) is a fixed tie-break; any producer
             * of v is a valid step of an optimal local alignment. */
            if (H[(ti - 1) * cols + (tj - 1)] + sub == v) {
                H[ti * cols + tj] = -v; ti--; tj--;                /* came from the diagonal */
            } else if (H[(ti - 1) * cols + tj] + SW_GAP == v) {
                H[ti * cols + tj] = -v; ti--;                      /* came from above (gap in b) */
            } else {
                H[ti * cols + tj] = -v; tj--;                      /* came from the left (gap in a) */
            }
        }

        best_score = max_val;
        path_len   = steps;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile int32_t sink = best_score;   /* the best local-alignment score */

    free(a); free(b);
    munmap(H, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_i64(&m, "max_score", (long long)sink);
    p2_meta_kv_u64(&m, "traceback_path_len", (uint64_t)path_len);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "fill write matches kernel_dp_v2; the backward traceback is the distinct (but faint/brief) tell, resolvable only when one pass spans multiple 500ms snapshots");
    p2_meta_close(&m);
    return 0;
}
