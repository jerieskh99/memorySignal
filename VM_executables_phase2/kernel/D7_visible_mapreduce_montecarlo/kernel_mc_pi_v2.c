/* kernel_mc_pi_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  MONTE-CARLO PI:  scalar-reduction pi estimate by uniform dart-throwing
 * ============================================================================
 *
 *  DWARF   : MapReduce / Monte Carlo (D7)   (Berkeley 13 computational motif)
 *  FAMILY  : IDLE (quiet control)           (first-division, memory-signature label)
 *  PURPOSE : The DELIBERATE QUIET CONTROL for D7. It demonstrates that a
 *            Monte-Carlo estimator whose whole job is to accumulate a SCALAR
 *            (a count) is nearly invisible to a host memory-WRITE signal: the
 *            reduction never grows a buffer, so there is essentially nothing to
 *            see. It is the D7 analogue of spmv (D2) and bfs (D9).
 *
 *  PICTURE (top view):  N darts thrown into the unit square; the reduction
 *  counts how many land inside the quarter circle x^2 + y^2 < 1.
 *
 *      y=1 +--------------------+
 *          | .  *   . *  .    . |   *  = inside  (x^2 + y^2 < 1)
 *          |*.  ***. .  *  . *  |   .  = outside
 *          | .**###**.  . *  .* |   #  = quarter-circle arc (radius 1)
 *          |.***####*. *  .   . |
 *          |****#####*.  .  * . |   pi_est = 4 * inside / N
 *          |*****#####. .*  .   |          -> 3.14159265... as N -> infinity
 *      y=0 +--------------------+
 *          x=0                x=1
 *
 *  ALGORITHM:
 *      1. Allocate one small partials[] array of B int64 block-counts. This tiny
 *         array (a few KB) is the ONLY memory write of any size the kernel makes
 *         -- it exists purely so the harness has a dominant buffer to mmap+mlock.
 *      2. Split the N samples into B blocks. For each block, draw its share of
 *         (x,y) points uniformly in [0,1)^2 from the harness RNG, count how many
 *         satisfy x*x + y*y < 1.0, and write that single count into partials[b].
 *         The per-point work touches only CPU registers -- no array is grown.
 *      3. Reduce partials[] to a total inside-count and form pi_est = 4*inside/N.
 *      4. Each measure pass RE-SEEDS and recomputes every block (re-randomising),
 *         accumulating pi_est and counting passes for the timed duration.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Almost nothing. The sampling loop is pure register compute; the only
 *      writes are B int64 stores into partials[] (default B=4096 -> 32 KB) plus
 *      the scalar reduction. There is no dense write front, no growing working
 *      set, no periodic buffer rewrite -- so a write-only observer sees a flat,
 *      near-idle trace. That deliberate quietness IS the control: it is the
 *      baseline against which the "visible" D7 map/reduce writers are contrasted.
 *
 *  HONEST NOTE: this is the quiet baseline. Because the estimator only
 *  accumulates a scalar (into a handful of KB of partials), its host-visible
 *  write footprint is intentionally tiny -> writes are near-invisible. Do not
 *  expect a strong write signature here; that absence is the whole point.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: IDLE (quiet control). Dwarf: MapReduce / Monte Carlo.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (MC passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_mc_pi_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Monte-Carlo pi; MapReduce/Monte-Carlo quiet control)\n"
"  --samples N           Number of dart throws per pass (default 100000000)\n"
"  --blocks B            Reduction blocks -> partials[] size (default 4096)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on partials-array bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* uniform double in [0,1) from the xoshiro stream */
static inline double rng_unit(p2_rng_t *r) {
    return (double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long samples    = p2_get_i64(argc, argv, "--samples", 100000000);
    long long blocks     = p2_get_i64(argc, argv, "--blocks", 4096);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    /* Clamp: at least one block, and at least as many samples as blocks so every
     * block draws >= 1 point. No snapping otherwise. */
    if (blocks < 1) blocks = 1;
    if (samples < blocks) samples = blocks;

    size_t B = (size_t)blocks;
    size_t N = (size_t)samples;
    size_t buf_bytes = B * sizeof(int64_t);   /* the (tiny) partials array */
    if (buf_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("partials bytes %zu exceed --max-mb %lld", buf_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "IDLE (quiet control)");
    p2_meta_kv_str(&m, "dwarf", "D7 MapReduce/MonteCarlo");
    p2_meta_kv_str(&m, "family", "IDLE (quiet control)");
    p2_meta_kv_str(&m, "scheme", "Monte-Carlo pi via scalar reduction (count darts inside quarter circle)");
    p2_meta_kv_str(&m, "note", "quiet: only a scalar/partials accumulator is written");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "samples", samples);
    p2_meta_kv_i64(&m, "blocks", blocks);
    p2_meta_kv_u64(&m, "partials_bytes", buf_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The partials array is the dominant buffer -> mmap + mlock it. It is tiny
     * on purpose (a few KB): the scalar reduction is the workload's whole point,
     * so there is nearly nothing here for a host write-signal to observe. */
    int64_t *partials = (int64_t *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                                        MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (partials == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", buf_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(partials, buf_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(partials, buf_bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Warm-up: touch the partials array and run one throwaway sampling pass so
     * pages are faulted in and caches primed before timing begins. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t b = 0; b < B; b++) partials[b] = 0;
    {
        size_t warm = (N < 100000) ? N : 100000;   /* small priming pass */
        int64_t warm_inside = 0;
        for (size_t s = 0; s < warm; s++) {
            double x = rng_unit(&rng), y = rng_unit(&rng);
            if (x * x + y * y < 1.0) warm_inside++;
        }
        partials[0] = warm_inside;                  /* keep the work observable */
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    double pi_acc = 0.0;          /* running sum of per-pass pi estimates */
    double pi_est = 0.0;          /* most recent pass estimate            */
    uint64_t total_inside = 0;    /* inside-count of the most recent pass */
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One Monte-Carlo pass. Re-seed each pass (mixing in the pass index) so
         * every pass re-randomises the dart positions rather than repeating. */
        p2_rng_seed(&rng, seed + passes);

        /* MAP: split N samples across B blocks. Each block draws its share of
         * darts (pure register compute) and writes ONE int64 count. Distribute
         * the remainder so the block sample-counts sum to exactly N. */
        size_t base = N / B, rem = N % B;
        uint64_t inside = 0;
        for (size_t b = 0; b < B; b++) {
            size_t cnt = base + (b < rem ? 1 : 0);   /* darts for this block */
            int64_t bin = 0;
            for (size_t s = 0; s < cnt; s++) {
                double x = rng_unit(&rng), y = rng_unit(&rng);
                if (x * x + y * y < 1.0) bin++;       /* register accumulate */
            }
            partials[b] = bin;                        /* the only sized write */
        }
        /* REDUCE: sum the block counts into the total inside-count. */
        for (size_t b = 0; b < B; b++) inside += (uint64_t)partials[b];

        pi_est = 4.0 * (double)inside / (double)N;
        pi_acc += pi_est;
        total_inside = inside;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    double pi_mean = (passes > 0) ? (pi_acc / (double)passes) : pi_est;
    volatile double sink = pi_est;                   /* prevent dead-code elim */

    munmap(partials, buf_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "last_inside", total_inside);
    p2_meta_kv_f64(&m, "pi_est", (double)sink);
    p2_meta_kv_f64(&m, "pi_mean", pi_mean);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "quiet control: only a scalar/partials accumulator is written -> near-invisible host writes; pi_est converges as ~1/sqrt(N)");
    p2_meta_close(&m);
    return 0;
}
