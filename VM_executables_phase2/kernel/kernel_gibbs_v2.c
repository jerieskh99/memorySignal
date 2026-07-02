/* kernel_gibbs_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Graphical Models dwarf (Berkeley motif D12), the Gibbs-sampling variant. A
 * W x H grid of discrete spins, each in {0..K-1} (K=2 is the Ising model, K>2 is
 * the Potts model). The model energy favours agreement between 4-connected
 * neighbours: a cell's local conditional distribution, given its 4 neighbours, is
 *
 *       p(s) proportional to exp( beta * count_equal(s, neighbours) )
 *
 * where count_equal(s, .) is how many of the 4 neighbours already hold state s,
 * and beta (the inverse temperature) controls how strongly like sticks to like.
 * The grid is stored one byte per cell in an mmap'd buffer (the dominant buffer);
 * boundaries are periodic (the grid wraps into a torus).
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs kernel_hmm_v2)
 * ----------------------------------------------------------------------------
 * The HMM kernel fills a fresh trellis with a DETERMINISTIC, monotone column
 * front: every cell is a normalised probability computed by a dense matvec, and
 * the front never revisits a column. Gibbs is the opposite. Each measure pass is
 * one full SWEEP that RESAMPLES every cell in place from its local conditional,
 * drawing the new spin STOCHASTICALLY with p2_rng_next. There is no wavefront and
 * no reset: the same buffer is overwritten sweep after sweep as a Markov chain
 * that keeps evolving. So the distinct write here is a stochastic, whole-grid,
 * per-cell resample whose written CONTENT is random and driven by neighbour state
 * -- random-content writes, not the HMM's deterministic bounded probabilities.
 *
 * ============================================================================
 * ALGORITHM (per measure pass = one full Gibbs sweep)
 * ============================================================================
 *   Visit every cell (row-major). For the cell at (x,y):
 *     1. Read its 4 periodic neighbours (up, down, left, right).
 *     2. Form the K conditional weights w[s] = exp(beta * count_equal(s, nbrs)).
 *        count_equal(s, nbrs) is in {0..4}, so the weight is one of only five
 *        precomputed values exp(beta*0..4) -- no exp() call in the hot loop.
 *     3. Normalise implicitly by drawing u uniformly in [0, sum(w)) and walking
 *        the cumulative weights to pick the new state (inverse-CDF sampling).
 *     4. Write the sampled state back into the cell.
 *   The grid is NOT reset between sweeps: it is a Markov chain whose stationary
 *   distribution is the Potts/Ising model, so successive sweeps are correlated
 *   samples and the buffer content drifts continuously. Count sweeps.
 *
 * MEMORY SIGNATURE (what the host write-signal actually sees)
 * ----------------------------------------------------------------------------
 * A large byte grid overwritten in full, cell by cell, once per sweep. Every
 * write is a fresh random draw conditioned on the current neighbourhood, so the
 * write ADDRESSES march sequentially through the grid while the write CONTENT is
 * stochastic and evolving -- a dense, uniform-coverage, random-content write that
 * repeats every sweep. The read-only exp() table and the RNG state are tiny and
 * invisible to a write-only host signal; only the grid resample shows.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Graphical Models. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (Gibbs sweeps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_gibbs_v2";
#define MAX_STATES 256   /* spin stored in one byte, so K must fit in a uint8_t */

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Gibbs sampling on a Potts/Ising grid; graphical-models kernel)\n"
"  --width W             Grid width in cells (default 1024)\n"
"  --height H            Grid height in cells (default 1024; grid is W*H bytes)\n"
"  --states K            Discrete states per cell; 2 = Ising, >2 = Potts (default 2)\n"
"  --beta-milli B        Inverse temperature x1000 (default 400 = 0.4)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on grid bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* Uniform double in [0,1) from the PRNG (top 53 bits -> exact IEEE fraction). */
static inline double rng_unit(p2_rng_t *r) {
    return (double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Sample one new spin for a cell whose 4 neighbours hold states (n0,n1,n2,n3).
 * The local conditional is p(s) proportional to exp(beta * count_equal(s,nbrs)),
 * and expw[c] = exp(beta * c) is the precomputed weight for c neighbours equal
 * (c in {0..4}). We build the K unnormalised weights, then draw by inverse-CDF:
 * pick u in [0, total) and return the first state whose cumulative weight passes
 * u. This is exactly sampling from the normalised conditional without ever
 * forming the normalised probabilities explicitly. */
static inline uint8_t gibbs_sample(p2_rng_t *rng, size_t K, const double *expw,
                                   uint8_t n0, uint8_t n1, uint8_t n2, uint8_t n3) {
    /* Per-state count of equal neighbours (0..4), then weight = expw[count]. */
    double w[MAX_STATES];
    double total = 0.0;
    for (size_t s = 0; s < K; s++) {
        int c = (n0 == s) + (n1 == s) + (n2 == s) + (n3 == s);   /* in {0..4} */
        double ws = expw[c];
        w[s] = ws;
        total += ws;
    }
    /* Inverse-CDF draw: u in [0,total) walks the cumulative weights. */
    double u = rng_unit(rng) * total;
    double acc = 0.0;
    for (size_t s = 0; s < K; s++) {
        acc += w[s];
        if (u < acc) return (uint8_t)s;
    }
    return (uint8_t)(K - 1);   /* guard: only reachable via float rounding at the top end */
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long width      = p2_get_i64(argc, argv, "--width", 1024);
    long long height     = p2_get_i64(argc, argv, "--height", 1024);
    long long states     = p2_get_i64(argc, argv, "--states", 2);
    /* beta is passed as integer-milli because the phase2 arg helpers are
     * integer-only: --beta-milli 400 = 0.4 (the inverse temperature x1000). */
    long long beta_milli = p2_get_i64(argc, argv, "--beta-milli", 400);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    double beta = (double)beta_milli / 1000.0;

    if (width  < 8 || width  > (1LL << 16)) { P2_LOG_ERR("width %lld out of range (8..65536)", width); return 2; }
    if (height < 8 || height > (1LL << 16)) { P2_LOG_ERR("height %lld out of range (8..65536)", height); return 2; }
    if (states < 2 || states > MAX_STATES) { P2_LOG_ERR("states %lld out of range (2..%d)", states, MAX_STATES); return 2; }
    if (beta < 0.0 || beta > 16.0) { P2_LOG_ERR("beta %.3f out of range (0..16)", beta); return 2; }
    size_t W = (size_t)width, H = (size_t)height, K = (size_t)states;
    size_t cells = W * H;                 /* one spin per cell */
    size_t bytes = cells * sizeof(uint8_t);  /* the spin grid dominates the footprint */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("grid bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Graphical Models");
    p2_meta_kv_str(&m, "scheme", "Gibbs sampling on a 2D Potts/Ising grid (stochastic per-cell resample sweep, periodic)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "width", width);
    p2_meta_kv_i64(&m, "height", height);
    p2_meta_kv_i64(&m, "states", states);
    p2_meta_kv_i64(&m, "beta_milli", beta_milli);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The spin grid is the dominant buffer -> mmap + mlock it (it is resampled
     * in full every sweep, which is the workload's signature write). */
    uint8_t *grid = (uint8_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (grid == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(grid, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(grid, bytes);

    /* expw[c] = exp(beta * c) for c in {0..4}: the only distinct conditional
     * weights, since a cell has exactly 4 neighbours. Precomputing them keeps
     * exp() out of the per-cell hot loop. */
    double expw[5];
    for (int c = 0; c < 5; c++) expw[c] = exp(beta * (double)c);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Random initial spins (a high-temperature start): every cell independent
     * uniform in {0..K-1}. The chain then equilibrates towards the Potts model. */
    for (size_t i = 0; i < cells; i++) grid[i] = (uint8_t)(p2_rng_next(&rng) % K);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One full Gibbs sweep: resample every cell in place from its local
         * conditional. Neighbours wrap (periodic torus), so edge cells read the
         * opposite edge. The grid is never reset -- this is one step of a Markov
         * chain, and the buffer keeps evolving sweep after sweep. */
        for (size_t y = 0; y < H; y++) {
            size_t ym = ((y == 0) ? H - 1 : y - 1) * W;    /* row above (wraps) */
            size_t yp = ((y == H - 1) ? 0 : y + 1) * W;    /* row below (wraps) */
            size_t yc = y * W;                             /* current row base   */
            for (size_t x = 0; x < W; x++) {
                size_t xm = (x == 0) ? W - 1 : x - 1;      /* left  col (wraps)  */
                size_t xp = (x == W - 1) ? 0 : x + 1;      /* right col (wraps)  */
                uint8_t up    = grid[ym + x];
                uint8_t down  = grid[yp + x];
                uint8_t left  = grid[yc + xm];
                uint8_t right = grid[yc + xp];
                /* THE DISTINCT WRITE: a stochastic resample of this cell from
                 * p(s) proportional to exp(beta * count_equal(s, nbrs)). */
                grid[yc + x] = gibbs_sample(&rng, K, expw, up, down, left, right);
            }
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Final sink: the mean spin over the whole grid (a scalar summary of the
     * chain's current state, in [0, K-1]). Reading every cell also forces the
     * last sweep's writes to be observed, so the compiler cannot elide them. */
    double spin_sum = 0.0;
    for (size_t i = 0; i < cells; i++) spin_sum += (double)grid[i];
    volatile double sink = spin_sum / (double)cells;   /* mean spin in [0, K-1] */

    munmap(grid, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "mean_spin", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "stochastic per-cell resample sweep is the distinct write vs hmm's deterministic front; higher beta orders the grid (larger equal-neighbour clusters)");
    p2_meta_close(&m);
    return 0;
}
