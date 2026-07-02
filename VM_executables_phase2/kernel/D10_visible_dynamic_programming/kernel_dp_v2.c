/* kernel_dp_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  EDIT-DISTANCE DP:  fill an N x N alignment table by a monotone wavefront.
 * ============================================================================
 *
 *  DWARF   : Dynamic Programming (D10)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the host write-signal produced by a strictly ordered
 *            dependency sweep: every cell depends on already-written neighbours,
 *            so the write front cannot skip ahead -- it marches down the table.
 *
 *  PICTURE (top view):
 *      The table T is filled top-to-bottom, one row at a time. Each cell reads
 *      its up, left, and diagonal neighbours (all already written), so the live
 *      write BAND is one row wide and moves monotonically downward.
 *
 *            j ->  0  1  2  3  4  ...  N-1
 *          i  0 [  .  .  .  .  .  ...  .  ]  <- first row (seed: T[0][j] = j)
 *             1 [  .  .  .  .  .  ...  .  ]        up   diag
 *             2 [  .  .  .  .  .  ...  .  ]          \   |
 *      band ->3 [==================>....]   left --  \  v
 *             4 [  ?  ?  ?  ?  ?  ...  ?  ]  <- not yet   [i][j]
 *           ... [  ?  ?  ?  ?  ?  ...  ?  ]     written
 *
 *  ALGORITHM (per pass over the whole table):
 *      1. Re-randomise the two byte sequences a[] and b[] (fresh alignment
 *         problem each pass, so the table is genuinely recomputed, not reused).
 *      2. Seed the boundary: first row T[0][j] = j, first column T[i][0] = i
 *         (the cost of aligning against an empty prefix).
 *      3. Fill rows i = 1..N-1 left-to-right. Each interior cell takes the
 *         cheapest of three edits:
 *             T[i][j] = min( T[i-1][j] + 1,            (delete)
 *                            T[i][j-1] + 1,            (insert)
 *                            T[i-1][j-1] + (a[i]!=b[j]) )  (match / substitute)
 *      4. The bottom-right cell T[N-1][N-1] is the Levenshtein edit distance.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A large buffer written by a MONOTONE, row-major front. The data
 *      dependency (up/left/diagonal) forbids reordering, so the active write
 *      region is a thin band that migrates steadily from the top of the table
 *      to the bottom -- the wavefront tell that separates this from a GEMM-style
 *      writer that touches its whole output more uniformly.
 *      Honest caveat: that migrating band is only observable when a SINGLE fill
 *      is slow enough to span several 500 ms snapshots. On a small/fast table a
 *      whole fill completes inside one snapshot interval, so it collapses to a
 *      full-table writer and the band is invisible -- hence the large default dim.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Dynamic Programming. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (table fills) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_dp_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign edit-distance DP table fill; Dynamic-Programming kernel)\n"
"  --dim N               Table side length (default 8192; uses N*N * 4 bytes)\n"
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

/* Minimum of three costs -- the inner recurrence of the table fill. Kept as a
 * tiny branch-only helper (no library call) so the innermost loop stays cheap. */
static inline int32_t min3(int32_t a, int32_t b, int32_t c) {
    int32_t m = a < b ? a : b; return m < c ? m : c;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long dim        = p2_get_i64(argc, argv, "--dim", 8192);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (dim < 32 || dim > 65536) { P2_LOG_ERR("dim %lld out of range (32..65536)", dim); return 2; }
    size_t N = (size_t)dim;
    size_t cells = N * N;
    size_t bytes = cells * sizeof(int32_t);
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
    p2_meta_kv_str(&m, "scheme", "edit-distance / Needleman-Wunsch table fill (row-major wavefront)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "dim", dim);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    int32_t *T = (int32_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (T == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(T, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(T, bytes);
    uint8_t *a = (uint8_t *)malloc(N), *b = (uint8_t *)malloc(N);
    if (!a || !b) { free(a); free(b); munmap(T, bytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) { a[i] = (uint8_t)p2_rng_next(&rng); b[i] = (uint8_t)p2_rng_next(&rng); }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t fills = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Fresh alignment problem each pass: re-seed both sequences so the fill
         * is a real recomputation (never a cache-warm rerun of identical data). */
        for (size_t i = 0; i < N; i++) { a[i] = (uint8_t)p2_rng_next(&rng); b[i] = (uint8_t)p2_rng_next(&rng); }
        /* Boundary conditions: aligning either prefix against the empty string
         * costs one edit per character, so column 0 = i and row 0 = j. */
        for (size_t i = 0; i < N; i++) T[i * N] = (int32_t)i;   /* first column */
        for (size_t j = 0; j < N; j++) T[j] = (int32_t)j;       /* first row */
        /* Row-major wavefront: rows are filled top-to-bottom, and within a row
         * cells go left-to-right. This is the write order the host signal sees;
         * every cell needs its up (prev[j]), left (row[j-1]) and diagonal
         * (prev[j-1]) neighbours, all of which are already written, which is
         * exactly what forbids the front from advancing out of order. */
        for (size_t i = 1; i < N; i++) {
            int32_t *row = T + i * N; const int32_t *prev = T + (i - 1) * N; uint8_t ai = a[i];
            for (size_t j = 1; j < N; j++) {
                int32_t cost = (ai == b[j]) ? 0 : 1;                 /* 0 on match, 1 on substitution */
                row[j] = min3(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost);
            }
        }
        fills++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile int32_t sink = T[(N - 1) * N + (N - 1)];   /* the edit distance */

    free(a); free(b);
    munmap(T, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "fills", fills);
    p2_meta_kv_i64(&m, "edit_distance", (long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "wavefront visible only when one fill spans multiple 500ms snapshots (large table)");
    p2_meta_close(&m);
    return 0;
}
