/* kernel_histogram_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * MapReduce / Monte Carlo dwarf (Berkeley motif D7), the histogram / binning
 * variant. This is the canonical MapReduce REDUCE with a LARGE reduction
 * target: instead of folding a stream of samples into one scalar, it folds
 * them into an array of B counters ("bins"). Each sample is MAPPED to a bin
 * index and REDUCED by a scatter-increment (bins[idx]++). The reduction target
 * is therefore a real memory footprint (B * 8 bytes), and the write pattern is
 * a RANDOM SCATTER across the whole bins array.
 *
 * WHY IT IS WRITE-VISIBLE (vs the quiet mc_pi scalar control)
 * ----------------------------------------------------------------------------
 * The Monte Carlo pi control (mc_pi) is the QUIET member of D7: it maps each
 * sample to a hit/miss test and reduces into ONE scalar counter that lives in
 * a register / one cache line, so a host write-signal sees essentially nothing.
 * This histogram kernel keeps the same map+reduce skeleton but swaps the scalar
 * reduction target for a large mlock'd array. The reduce now lands as writes
 * spread over the entire bins[] footprint, and (unlike the contiguous/strided
 * stencils) they arrive in a RANDOM, address-hopping scatter -- that scatter is
 * the distinct tell in the write-signal.
 *
 * ============================================================================
 * PICTURE (top view): samples stream in -> scatter into an array of bins
 * ============================================================================
 *
 *   N samples (from the RNG)                    bins[] : B int64 counters
 *   . . . . . . . . . . . .                     (the mlock'd reduction target)
 *      \    |    /   \   |                              __
 *       \   |   /     \  |                    __       |##|            __
 *   map each sample -> bin index -> bins[idx]++    __ |##|   __       |##|
 *        (uniform, or skew = heavier tail)   __   |##||##|  |##|  __  |##|  __
 *                                           |##|  |##||##|  |##| |##| |##| |##|
 *   +-----------------------------------------------------------------------+
 *   | bin 0                              bin idx                     bin B-1 |
 *   +-----------------------------------------------------------------------+
 *      random scatter: increments land all across the array, not in a stream.
 *
 * ============================================================================
 * ALGORITHM (per measure pass)
 * ============================================================================
 *   1. Zero the whole bins[] array (start each pass from a clean reduction).
 *   2. Re-seed the RNG (so every pass scatters the identical sample stream and
 *      the last-pass totals are reproducible / verifiable).
 *   3. For each of N samples:
 *        a. draw a unit sample u in [0,1) from the RNG,
 *        b. MAP it to a bin index in [0,B):
 *             uniform : idx = floor(u * B),
 *             skew    : idx = floor(u*u * B)  (square concentrates mass near 0,
 *                                              a heavier-tailed scatter),
 *        c. REDUCE: bins[idx]++  (the scatter-increment).
 *   4. Count the pass. The sum over all bins after a pass is exactly N
 *      (every sample increments exactly one bin) -- the conservation invariant.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL (visible). Dwarf: MapReduce / Monte Carlo.
 * See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (scatter passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_histogram_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign histogram/binning; MapReduce reduce kernel)\n"
"  --samples N           Number of samples to scatter per pass (default 50000000)\n"
"  --bins B              Number of histogram bins / reduction target (default 1048576)\n"
"  --dist uniform|skew   Index distribution: uniform, or skew (heavier tail) (default uniform)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the sample stream (default 42)\n"
"  --max-mb N            Hard cap on the bins-array bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* uniform double in [0,1) from the xoshiro stream */
static inline double p2_rng_unit(p2_rng_t *r) {
    return (double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long samples    = p2_get_i64(argc, argv, "--samples", 50000000);
    long long bins       = p2_get_i64(argc, argv, "--bins", 1048576);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dist     = p2_get_str(argc, argv, "--dist", "uniform");

    if (bins < 1) bins = 1;                          /* clamp bins >= 1 */
    /* Only two distributions are defined; anything else is a hard error so a
     * typo cannot silently fall through to the default. */
    int skew;
    if (strcmp(dist, "uniform") == 0)      skew = 0;
    else if (strcmp(dist, "skew") == 0)    skew = 1;
    else { P2_LOG_ERR("dist '%s' unknown (use uniform|skew)", dist); return 2; }

    if (samples < 0) { P2_LOG_ERR("samples %lld must be >= 0", samples); return 2; }
    if (bins > (1LL << 32)) { P2_LOG_ERR("bins %lld out of range (1..2^32)", bins); return 2; }
    size_t B = (size_t)bins;
    size_t bytes = B * sizeof(int64_t);              /* the bins array is the footprint */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("bins bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D7 MapReduce/MonteCarlo");
    p2_meta_kv_str(&m, "scheme", "histogram scatter-increment (MapReduce reduce into a large bins array)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "samples", samples);
    p2_meta_kv_i64(&m, "bins", bins);
    p2_meta_kv_str(&m, "dist", skew ? "skew" : "uniform");
    p2_meta_kv_u64(&m, "bins_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The bins array is the dominant buffer and the reduction target -> mmap +
     * mlock it. The scatter-increments land here in random order, so advise the
     * kernel that access is random (the defining contrast with the sequential
     * stencils). */
    int64_t *binv = (int64_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (binv == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(binv, bytes, MADV_NOHUGEPAGE);
    p2_madvise(binv, bytes, MADV_RANDOM);            /* scatter access, not sequential */
    if (!no_mlock) p2_mlock_soft(binv, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Touch the whole array once (fault it in) and start from a zeroed reduction. */
    memset(binv, 0, bytes);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t last_total = 0;                          /* total increments of the last pass */
    size_t N = (size_t)samples;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One scatter pass over N samples into the B bins.
         *   1. zero the bins (clean reduction target),
         *   2. re-seed so every pass scatters the identical stream,
         *   3. map each unit sample to a bin index and scatter-increment it.
         * Reading the increment count back out gives the conservation total. */
        memset(binv, 0, bytes);
        p2_rng_t rng; p2_rng_seed(&rng, seed);
        uint64_t total = 0;
        for (size_t s = 0; s < N; s++) {
            double u = p2_rng_unit(&rng);             /* map: unit sample in [0,1) */
            if (skew) u = u * u;                      /* skew: square -> heavier tail near 0 */
            size_t idx = (size_t)(u * (double)B);     /* -> bin index in [0,B) */
            if (idx >= B) idx = B - 1;                /* guard the u==... rounding edge */
            binv[idx]++;                              /* reduce: the scatter-increment */
            total++;
        }
        last_total = total;                           /* must equal N (conservation) */
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile int64_t sink = binv[B / 2];              /* a live bin: prevent dead-code elim */

    munmap(binv, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "last_pass_total", last_total);   /* conservation: == samples */
    p2_meta_kv_i64(&m, "center_bin", (long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: random scatter-increments across the bins array (write-visible reduce); "
                   "contrast the quiet mc_pi scalar reduction which writes essentially nothing");
    p2_meta_close(&m);
    return 0;
}
