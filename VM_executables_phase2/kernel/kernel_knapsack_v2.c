/* kernel_knapsack_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  0/1 KNAPSACK DP:  a SINGLE capacity vector rewritten in reverse per item.
 * ============================================================================
 *
 *  DWARF   : Dynamic Programming (D10)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the host write-signal produced by an in-place, space-optimised
 *            DP that keeps ONLY the current capacity row. Each item sweeps the
 *            whole vector once, from high capacity down to its own weight, so the
 *            live write region is a full-width 1D bar that is repainted N times --
 *            deliberately DIFFERENT from the 2D tables of kernel_dp_v2.
 *
 *  PICTURE (top view):
 *      The state is one 1D array dp[0..C] (a single capacity row). For item i we
 *      sweep c from C down to w[i]; the '<' arrow shows the reverse write order.
 *      dp[c] reads dp[c - w[i]] (a LOWER, not-yet-touched index this pass), which
 *      is exactly why the reverse sweep keeps the 0/1 semantics correct -- an item
 *      can be taken at most once because its own update never feeds itself.
 *
 *            c ->  0    w[i]        c-w[i]        c            C
 *          dp   [  v .... v ......... r .......... W ......... . ]
 *                            \_______________________/
 *                             read dp[c-w[i]], write dp[c]
 *          item i sweep:   C  <==============================  w[i]   (downward)
 *          item i+1 :      C  <==============================  w[i+1] (again)
 *                          ^ same 1D bar, repainted once per item ^
 *
 *  ALGORITHM (per pass over all N items):
 *      1. Zero the capacity vector dp[0..C] (empty knapsack: value 0 everywhere).
 *      2. Re-seed the N items: weight w[i] in [1, C/8], value v[i] in [1, 1000]
 *         (a fresh instance each pass, so the DP is a real recomputation, never a
 *         cache-warm rerun of identical data).
 *      3. For each item i = 0..N-1, sweep capacity DOWNWARD:
 *             for c = C down to w[i]:
 *                 long long cand = dp[c - w[i]] + v[i];
 *                 if (cand > dp[c]) dp[c] = cand;
 *         The downward direction is load-bearing: sweeping upward would let one
 *         item be re-used (the unbounded-knapsack recurrence) and break 0/1.
 *      4. dp[C] is the best achievable value for capacity C.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      One large contiguous buffer (dp is the dominant, mmap'd allocation) that is
 *      swept end-to-end N times per pass. Unlike the 2D DP whose active band
 *      MIGRATES down a big table, here the SAME 1D vector is repainted in place,
 *      each item overwriting a high-to-low suffix of it. The tell is a full-width,
 *      stationary 1D footprint rewritten many times -- a 1D rolling update, not a
 *      2D table walk.
 *      Honest caveat: the reverse per-item sweep is far finer-grained than a 500 ms
 *      snapshot, so a snapshot cannot resolve the direction of any single sweep; it
 *      sees the aggregate as a hot, repeatedly-rewritten 1D region. The distinctive
 *      1D-vs-2D footprint therefore shows up in the SIZE and re-write density of
 *      the touched region, not in a visibly moving band.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Dynamic Programming. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (DP passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_knapsack_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign 0/1 knapsack DP; Dynamic-Programming kernel)\n"
"  --capacity C          Knapsack capacity (default 1048576; dp is (C+1)*8 bytes)\n"
"  --items N             Number of items (default 4096)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on dp bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long capacity   = p2_get_i64(argc, argv, "--capacity", 1048576);
    long long items      = p2_get_i64(argc, argv, "--items", 4096);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (capacity < 16 || capacity > (1LL << 30)) {
        P2_LOG_ERR("capacity %lld out of range (16..2^30)", capacity); return 2;
    }
    if (items < 1 || items > (1LL << 24)) {
        P2_LOG_ERR("items %lld out of range (1..2^24)", items); return 2;
    }
    size_t C = (size_t)capacity;
    size_t slots = C + 1;                               /* dp indices 0..C inclusive */
    size_t bytes = slots * sizeof(long long);           /* the 1D dp vector dominates */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("dp bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }
    size_t N = (size_t)items;
    /* Per-item weight is drawn from [1, C/8] so several items are needed to fill
     * the sack; guard the divisor for tiny capacities so the range stays >= 1. */
    long long wmax = (long long)(C / 8); if (wmax < 1) wmax = 1;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Dynamic Programming");
    p2_meta_kv_str(&m, "scheme", "0/1 knapsack; single 1D capacity vector, reverse in-place rolling update per item");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "capacity", capacity);
    p2_meta_kv_i64(&m, "items", items);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The dp vector is the dominant buffer -> mmap + soft-mlock it. It is zeroed
     * and swept N times every pass, which is the workload's signature write. */
    long long *dp = (long long *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (dp == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(dp, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(dp, bytes);

    /* Compact item arrays (re-seeded each pass; small next to the dp vector). */
    long long *w = (long long *)malloc(N * sizeof(long long));
    long long *v = (long long *)malloc(N * sizeof(long long));
    if (!w || !v) { free(w); free(v); munmap(dp, bytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) {                    /* first instance (touch pages) */
        w[i] = 1 + (long long)(p2_rng_next(&rng) % (uint64_t)wmax);
        v[i] = 1 + (long long)(p2_rng_next(&rng) % 1000ULL);
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Empty knapsack: value 0 at every capacity. This clears the whole 1D
         * bar before the item sweeps repaint it. */
        for (size_t c = 0; c < slots; c++) dp[c] = 0;
        /* Fresh instance each pass: new weights/values so the DP is a genuine
         * recomputation, not a cache-warm rerun of identical data. */
        for (size_t i = 0; i < N; i++) {
            w[i] = 1 + (long long)(p2_rng_next(&rng) % (uint64_t)wmax);
            v[i] = 1 + (long long)(p2_rng_next(&rng) % 1000ULL);
        }
        /* 1D rolling update: each item repaints a high-to-low suffix of the SAME
         * capacity vector. Sweeping DOWNWARD (c from C to w[i]) guarantees each
         * item is used at most once, because dp[c] reads dp[c - w[i]] -- a lower
         * index that this item has not yet overwritten this pass. */
        for (size_t i = 0; i < N; i++) {
            long long wi = w[i], vi = v[i];
            for (long long c = (long long)C; c >= wi; c--) {
                long long cand = dp[c - wi] + vi;       /* take item i at capacity c */
                if (cand > dp[c]) dp[c] = cand;         /* keep the better of skip/take */
            }
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile long long sink = dp[C];                    /* the best achievable value */

    free(w); free(v);
    munmap(dp, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_i64(&m, "best_value", (long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "distinct from 2D DP by a stationary full-width 1D footprint repainted N times; a 500ms snapshot cannot resolve a single reverse sweep's direction");
    p2_meta_close(&m);
    return 0;
}
