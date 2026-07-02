/* kernel_matrixchain_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  MATRIX-CHAIN DP:  fill an N x N cost table along its ANTI-DIAGONALS.
 * ============================================================================
 *
 *  DWARF   : Dynamic Programming (D10)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the host write-signal produced by a DIAGONAL wavefront. The
 *            classic matrix-chain optimisation asks, for a product of N matrices
 *            A_i x ... x A_j, what parenthesisation minimises the number of
 *            scalar multiplications. Its DP table is filled by INCREASING chain
 *            length, which sweeps the table diagonal by diagonal rather than
 *            row by row -- a fundamentally different write order from the
 *            row-major edit-distance fill in kernel_dp_v2.
 *
 *  PICTURE (top view):
 *      The table m[i][j] holds the best cost for the sub-chain i..j, so only the
 *      UPPER TRIANGLE (i <= j) is meaningful. It is filled by chain length L: the
 *      main diagonal (L=1) is the seed m[i][i]=0, then each longer diagonal is
 *      written using cells on shorter diagonals below-left of it. The live write
 *      BAND is one anti-diagonal, and it marches from the main diagonal toward
 *      the top-right corner.
 *
 *            j ->  0    1    2    3    4   ...  N-1
 *          i  0 [  0 -> d1-> d2-> d3-> d4 ...  ANS ]   ANS = m[0][N-1] (goal)
 *             1 [  .    0 -> d1-> d2-> d3 ...   .  ]      each cell reads a
 *             2 [  .    .    0 -> d1-> d2 ...   .  ]      whole ROW segment to
 *             3 [  .    .    .    0 -> d1 ...   .  ]      its left and a whole
 *             4 [  .    .    .    .    0  ...   .  ]      COLUMN segment below it
 *           ... [                          0 ->  . ]      (the split loop over k)
 *          N-1 [  .    .    .    .    .   ...   0  ]   <- main diagonal (all zero)
 *
 *          Diagonals fill in order: L=1 (the 0s), then d1, d2, d3, ... The band
 *          is the set of cells with j - i = L-1; it moves up-and-right each pass.
 *
 *  ALGORITHM (per pass over the whole table):
 *      1. Re-randomise the dimension vector p[0..N] with values in [1,100]
 *         (a fresh chain each pass, so the table is genuinely recomputed).
 *      2. Seed the main diagonal: m[i][i] = 0 (a single matrix costs nothing).
 *      3. For chain length L = 2..N, for every start i with j = i+L-1 <= N-1,
 *         choose the split k that minimises the cost of the two halves plus the
 *         cost of the final multiply:
 *             m[i][j] = min over k=i..j-1 of
 *                       ( m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1] )
 *         The inner split loop is O(N); over all cells this is the classic
 *         O(N^3) matrix-chain recurrence.
 *      4. The top-right cell m[0][N-1] is the optimal scalar-multiplication cost
 *         for the whole chain.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A large N x N buffer whose writes advance as an ANTI-DIAGONAL front. Each
 *      new diagonal depends on cells already written on shorter diagonals (the
 *      left row segment m[i][k] and the lower column segment m[k+1][j]), so the
 *      front cannot skip ahead: it sweeps from the main diagonal up toward the
 *      top-right corner. Compared with kernel_dp_v2's row-major band, this front
 *      is diagonally oriented and each successive diagonal is SHORTER (the upper
 *      triangle narrows toward the corner), so the write rate tapers as the pass
 *      finishes -- a distinct diagonal-wavefront tell.
 *      Honest caveat: that migrating diagonal is only observable when a SINGLE
 *      fill is slow enough to span several 500 ms snapshots. On a small/fast
 *      table a whole fill completes inside one snapshot interval, so it collapses
 *      to a full-triangle writer and the diagonal is invisible -- hence the large
 *      default chain length. Note also that only the upper triangle is written;
 *      the lower triangle stays untouched after allocation.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Dynamic Programming. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (table fills) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_matrixchain_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign matrix-chain DP table fill; Dynamic-Programming kernel)\n"
"  --chain N             Number of matrices in the chain (default 1024; uses N*N * 8 bytes)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on table bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* Minimum of two int64 costs -- the reduction inside the split loop. Kept as a
 * tiny inline helper (no library call) so the innermost loop stays cheap. */
static inline int64_t min2_i64(int64_t a, int64_t b) { return a < b ? a : b; }

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long chain      = p2_get_i64(argc, argv, "--chain", 1024);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (chain < 8 || chain > 65536) { P2_LOG_ERR("chain %lld out of range (8..65536)", chain); return 2; }
    size_t N = (size_t)chain;
    size_t cells = N * N;                            /* full N x N table (upper triangle used) */
    size_t bytes = cells * sizeof(int64_t);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("table bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Dynamic Programming");
    p2_meta_kv_str(&m, "scheme", "matrix-chain multiplication order DP (anti-diagonal wavefront fill)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "chain", chain);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The cost table is the dominant buffer -> mmap + mlock it. It is refilled
     * every pass along anti-diagonals, which is the workload's signature write. */
    int64_t *tbl = (int64_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (tbl == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(tbl, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(tbl, bytes);

    /* Dimension vector: a chain of N matrices needs N+1 boundary dimensions, so
     * matrix i has shape p[i] x p[i+1]. Re-seeded each pass. */
    int32_t *p = (int32_t *)malloc((N + 1) * sizeof(int32_t));
    if (!p) { munmap(tbl, bytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i <= N; i++) p[i] = (int32_t)(p2_rng_next(&rng) % 100u) + 1;  /* [1,100] */
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t fills = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Fresh chain each pass: re-seed the dimensions so the fill is a real
         * recomputation (never a cache-warm rerun of identical data). */
        for (size_t i = 0; i <= N; i++) p[i] = (int32_t)(p2_rng_next(&rng) % 100u) + 1;
        /* Seed the main diagonal: a single matrix (chain of length 1) needs no
         * multiplication, so its cost is zero. */
        for (size_t i = 0; i < N; i++) tbl[i * N + i] = 0;
        /* Anti-diagonal wavefront: sweep by increasing chain length L. All cells
         * on diagonal L (those with j - i = L-1) are written here, and each reads
         * only cells on shorter diagonals -- the left row segment m[i][k] and the
         * lower column segment m[k+1][j], both already written. This dependency is
         * exactly what forbids the front from advancing out of order; the write
         * order the host signal sees is diagonal by diagonal. */
        for (size_t L = 2; L <= N; L++) {
            for (size_t i = 0; i + L <= N; i++) {
                size_t j = i + L - 1;               /* current sub-chain is i..j */
                int64_t best = INT64_MAX;
                int64_t pi = p[i], pj1 = p[j + 1];  /* outer boundary dims, fixed for this cell */
                const int64_t *rowi = tbl + i * N;  /* m[i][k] lives along row i          */
                /* Split loop: try every break point k, combining the best cost of
                 * the left half i..k with the right half k+1..j plus the cost of
                 * the one remaining multiply p[i] * p[k+1] * p[j+1]. */
                for (size_t k = i; k < j; k++) {
                    int64_t cost = rowi[k] + tbl[(k + 1) * N + j] + pi * (int64_t)p[k + 1] * pj1;
                    best = min2_i64(best, cost);
                }
                tbl[i * N + j] = best;
            }
        }
        fills++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile int64_t sink = tbl[0 * N + (N - 1)];   /* m[0][N-1]: the optimal cost */

    free(p);
    munmap(tbl, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "fills", fills);
    p2_meta_kv_i64(&m, "optimal_cost", (long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "diagonal wavefront visible only when one fill spans multiple 500ms snapshots (large chain); only the upper triangle is written");
    p2_meta_close(&m);
    return 0;
}
