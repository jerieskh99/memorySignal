/* kernel_floyd_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  FLOYD-WARSHALL:  all-pairs shortest paths by N full-matrix relaxations.
 * ============================================================================
 *
 *  DWARF   : Dynamic Programming (D10)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the host write-signal produced by a DP that revisits its
 *            ENTIRE state N times. Where the edit-distance kernel writes each
 *            cell once behind a monotone wavefront, Floyd-Warshall sweeps the
 *            whole N x N distance matrix once per pivot k, so the same cells are
 *            rewritten again and again -- a repeated full-matrix writer.
 *
 *  PICTURE (top view):
 *      The distance matrix D is relaxed against one pivot row/column k at a time.
 *      For pivot k, every cell D[i][j] is tested against the detour i -> k -> j;
 *      the read cross (row k, column k) is fixed, but the WRITE region is the
 *      whole matrix. After N pivots each cell has been visited N times.
 *
 *            j ->  0  1  2   k   ...  N-1
 *          i  0 [  .  .  .  Rk   ...  .  ]
 *             1 [  .  .  .  Rk   ...  .  ]      detour tested at cell [i][j]:
 *             2 [  .  .  .  Rk   ...  .  ]         D[i][j] = min( D[i][j],
 *          k -> [ Ck Ck Ck  X   Ck  Ck ]  <- pivot        D[i][k] + D[k][j] )
 *           ... [  .  .  .  Rk   ...  .  ]      Ck = pivot column (read)
 *          N-1 [  .  .  .  Rk   ...  .  ]      Rk = pivot row    (read)
 *                                             every other cell = written
 *
 *  ALGORITHM (per solve over the whole matrix):
 *      1. Re-seed a fresh random graph each pass (so the matrix is genuinely
 *         recomputed, never a cache-warm rerun of identical data):
 *             D[i][i] = 0 ; for i != j, with probability ~0.3 an edge of weight
 *             in [1,100), otherwise INF (a large sentinel meaning "no edge yet").
 *      2. For each pivot k = 0..N-1, relax every pair (i,j) through k:
 *             if D[i][k] + D[k][j] < D[i][j]:  D[i][j] = D[i][k] + D[k][j]
 *         Rows whose D[i][k] is INF are skipped, so an INF + INF detour can never
 *         overflow the sentinel and masquerade as a real (finite) path.
 *      3. After the N-th pivot, D[i][j] is the shortest-path cost from i to j
 *         (INF if j is unreachable from i).
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A large buffer touched by N back-to-back full-matrix sweeps. Unlike the
 *      one-shot wavefront fill of kernel_dp_v2 -- where the live write band
 *      migrates once from top to bottom and never returns -- Floyd-Warshall
 *      rewrites the same address range N times in a row, so the write-signal is
 *      a sustained, spatially uniform, repeatedly-revisited plateau rather than a
 *      travelling front. That "revisit the whole state N times" shape is the tell
 *      separating this DP from the single-pass wavefront DP.
 *      Honest caveat: within a single pivot sweep the writer is essentially flat
 *      row-major over the matrix, so the individual k-boundaries are not visible
 *      to a coarse 500 ms snapshot; what a snapshot resolves is the aggregate
 *      "one buffer, rewritten many times per solve" plateau. A small/fast matrix
 *      finishes an entire solve inside one snapshot, collapsing the N sweeps into
 *      a single full-matrix writer -- hence the large default dim.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Dynamic Programming. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (matrix solves) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_floyd_v2";

/* Sentinel for "no path known yet". Chosen large enough to dominate any real
 * path cost (max weight < 100, at most N-1 hops) yet small enough that a single
 * INF + finite addition stays finite -- and the INF-row skip in the inner loop
 * guarantees we never actually add INF + INF. */
static const double INF = 1e18;

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Floyd-Warshall all-pairs shortest paths; Dynamic-Programming kernel)\n"
"  --dim N               Matrix side length / node count (default 1024; uses N*N * 8 bytes)\n"
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

/* Draw a uniform double in [0,1). Uses the top 53 bits of the PRNG word so the
 * mantissa is filled evenly (same idiom as the N-body kernel). */
static inline double rng_unit(p2_rng_t *r) {
    return (double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Re-seed the distance matrix with a fresh random directed graph:
 * zero on the diagonal, and off-diagonal edges present with probability ~0.3
 * (weight in [1,100)) or INF otherwise. Called once per measure pass so every
 * solve recomputes a genuinely new problem. */
static void seed_graph(double *D, size_t N, p2_rng_t *rng) {
    for (size_t i = 0; i < N; i++) {
        double *row = D + i * N;
        for (size_t j = 0; j < N; j++) {
            if (i == j) { row[j] = 0.0; continue; }         /* distance to self is 0 */
            if (rng_unit(rng) < 0.3)                        /* ~30% edge density */
                row[j] = 1.0 + rng_unit(rng) * 99.0;        /* weight in [1, 100) */
            else
                row[j] = INF;                               /* no direct edge */
        }
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long dim        = p2_get_i64(argc, argv, "--dim", 1024);
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
    size_t bytes = cells * sizeof(double);
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
    p2_meta_kv_str(&m, "scheme", "Floyd-Warshall all-pairs shortest paths (N full-matrix relaxation sweeps per solve)");
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

    double *D = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (D == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(D, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(D, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    seed_graph(D, N, &rng);                     /* touch every page + first problem */
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t solves = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Fresh random graph each pass, so the matrix is really recomputed. */
        seed_graph(D, N, &rng);
        /* N full-matrix relaxation sweeps -- the repeated full-buffer write that
         * distinguishes this DP from the one-time wavefront fill. Pivot k fixes
         * the read cross (row k, column k); the whole matrix is the write set. */
        for (size_t k = 0; k < N; k++) {
            const double *rowk = D + k * N;         /* pivot row  D[k][*]  (read) */
            for (size_t i = 0; i < N; i++) {
                double *row = D + i * N;
                double dik = row[k];                /* pivot column D[i][k] (read) */
                if (dik >= INF) continue;           /* i cannot reach k: no detour, skip row */
                for (size_t j = 0; j < N; j++) {
                    double dkj = rowk[j];
                    double nd = dik + dkj;          /* cost of the detour i -> k -> j */
                    if (dkj < INF && nd < row[j]) row[j] = nd;   /* relax if shorter */
                }
            }
        }
        solves++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = D[(N - 1) * N + 0];  /* a live shortest-path cost (node N-1 -> 0) */

    munmap(D, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "solves", solves);
    p2_meta_kv_f64(&m, "sample_dist", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "repeated full-matrix rewrite (N sweeps/solve) is the distinct write vs the single wavefront of kernel_dp_v2; visible only when one solve spans multiple 500ms snapshots (large matrix)");
    p2_meta_close(&m);
    return 0;
}
