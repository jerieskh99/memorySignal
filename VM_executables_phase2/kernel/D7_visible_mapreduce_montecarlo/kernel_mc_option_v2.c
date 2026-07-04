/* kernel_mc_option_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * MapReduce / Monte Carlo dwarf (Berkeley motif D7). A Monte-Carlo European-call
 * option pricer. It simulates many independent random price trajectories of one
 * underlying asset (geometric Brownian motion), STORES every full trajectory in a
 * big E x T array, then reduces that array to a single price: the discounted mean
 * of the terminal-column payoff. The "map" is the per-path GBM walk; the "reduce"
 * is the average payoff. This is a quantitative-finance BENCHMARK, not a trading
 * system: pure mmap + compute, no sockets, no orders, no persistence.
 *
 * WHY IT IS WRITE-VISIBLE (vs the histogram scatter and the scalar control)
 * ----------------------------------------------------------------------------
 * The distinctive write here is BULK PATH STORAGE. Every measure pass fills the
 * whole E x T path array front-to-back: E paths, each a dense contiguous run of T
 * doubles written left-to-right. That is a large, regular, sequential write front
 * a host write-signal can see -- distinct from a Monte-Carlo histogram, whose tell
 * is a small scattered bin array, and from a scalar running-sum control, which has
 * essentially no data-footprint write at all. Same dwarf, different write pattern.
 *
 * ============================================================================
 * PICTURE (top view): E GBM paths fanning out from spot over T steps
 * ============================================================================
 *   price
 *     ^                                             . S[e][T-1]  <- payoff read
 *     |                                        . '        here (terminal column)
 *     |                                   . '  . -----                max(S-K,0)
 *   K +------------------------------.-'------------------  strike line
 *     |                         . '   ` .
 *     | spot -> * === = = = . '           ` . _
 *     |             ` = = = . _                 ` .
 *     |                       ` . _                 ` .  (paths below K pay 0)
 *     +------------------------------------------------------> time step t
 *        t0     t1     t2     ...                     t=T-1
 *
 *   store S : E rows x T cols of doubles, filled row-by-row (the bulk write)
 *   reduce  : price = e^{-r*T_mat} * mean_e( max(S[e][T-1] - strike, 0) )
 *
 * ============================================================================
 * ALGORITHM (per measure pass)
 * ============================================================================
 *   1. GBM step: for each path e and step t, evolve the underlying by
 *        S[e][t] = S[e][t-1] * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
 *      with dt = maturity / T, S[e][0] = spot, and Z ~ N(0,1) a standard normal.
 *   2. Box-Muller: turn two harness uniforms u1,u2 in (0,1] into a normal draw
 *        Z = sqrt(-2 ln u1) * cos(2*pi*u2)
 *      (the paired sin() draw is used too, so no uniforms are wasted).
 *   3. Store paths: write the full E x T array S front-to-back (the dominant
 *      mmap+mlock buffer -- this is the bulk path-storage write pattern).
 *   4. Reduce: payoff(e) = max(S[e][T-1] - strike, 0); the option price is the
 *      discounted mean payoff  e^{-r*maturity} * (1/E) * sum_e payoff(e).
 *
 * Each measure pass re-seeds, regenerates all E paths into the array, recomputes
 * the price, and counts one pass -- the same footprint is rewritten every pass.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL (visible). Dwarf: MapReduce / Monte Carlo.
 * See docs/SAFETY_MODEL.md.
 *
 * CORRECTNESS: the Monte-Carlo price is checked against the Black-Scholes closed
 * form for a European call (an independent analytic formula), so verification is
 * non-circular. With enough paths the two agree to within Monte-Carlo error.
 *
 * Phases: warmup (alloc + init) / measure (regenerate paths + price) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_mc_option_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Monte-Carlo European-call pricer; MapReduce/MonteCarlo kernel)\n"
"  --paths E             Number of simulated GBM paths (default 65536)\n"
"  --steps T             Time steps per path (default 64; array is E*T*8 bytes)\n"
"  --spot-milli S        Spot price x1000 (default 100000 = 100.0)\n"
"  --strike-milli K      Strike price x1000 (default 100000 = 100.0)\n"
"  --rate-milli R        Risk-free rate x1000 (default 50 = 0.05)\n"
"  --vol-milli V         Volatility sigma x1000 (default 200 = 0.20)\n"
"  --maturity-milli M    Maturity in years x1000 (default 1000 = 1.0)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on path-array bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* uniform double in (0,1] from the xoshiro stream. Shifted off zero so that
 * log(u1) in Box-Muller never hits log(0). */
static inline double p2_rng_unit(p2_rng_t *r) {
    /* (v >> 11) is in [0, 2^53); +1 then /2^53 lands in (0, 1]. */
    return ((double)(p2_rng_next(r) >> 11) + 1.0) * (1.0 / 9007199254740992.0);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long paths      = p2_get_i64(argc, argv, "--paths", 65536);
    long long steps      = p2_get_i64(argc, argv, "--steps", 64);
    /* Floating-point options are passed as integer-milli, because the phase2 arg
     * helpers are integer-only: --spot-milli 100000 = 100.0, --rate-milli 50 = 0.05. */
    long long spot_milli     = p2_get_i64(argc, argv, "--spot-milli", 100000);
    long long strike_milli   = p2_get_i64(argc, argv, "--strike-milli", 100000);
    long long rate_milli     = p2_get_i64(argc, argv, "--rate-milli", 50);
    long long vol_milli      = p2_get_i64(argc, argv, "--vol-milli", 200);
    long long maturity_milli = p2_get_i64(argc, argv, "--maturity-milli", 1000);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    double spot     = (double)spot_milli / 1000.0;
    double strike   = (double)strike_milli / 1000.0;
    double rate     = (double)rate_milli / 1000.0;
    double vol      = (double)vol_milli / 1000.0;
    double maturity = (double)maturity_milli / 1000.0;

    if (paths < 16 || paths > (1LL << 26)) {
        P2_LOG_ERR("paths %lld out of range (16..2^26)", paths); return 2;
    }
    if (steps < 1 || steps > (1LL << 20)) {
        P2_LOG_ERR("steps %lld out of range (1..2^20)", steps); return 2;
    }
    if (spot <= 0.0 || strike <= 0.0) {
        P2_LOG_ERR("spot %.3f / strike %.3f must be positive", spot, strike); return 2;
    }
    if (vol <= 0.0 || vol > 10.0) {
        P2_LOG_ERR("vol %.3f out of range (0..10)", vol); return 2;
    }
    if (maturity <= 0.0 || maturity > 100.0) {
        P2_LOG_ERR("maturity %.3f out of range (0..100)", maturity); return 2;
    }
    size_t E = (size_t)paths;
    size_t T = (size_t)steps;
    /* Guard the E*T multiply against size_t overflow before sizing the buffer. */
    if (E > SIZE_MAX / T) {
        P2_LOG_ERR("paths*steps overflows size_t (E=%zu, T=%zu)", E, T); return 2;
    }
    size_t cells = E * T;
    if (cells > SIZE_MAX / sizeof(double)) {
        P2_LOG_ERR("path-array elements overflow size_t (cells=%zu)", cells); return 2;
    }
    size_t buf_bytes = cells * sizeof(double);      /* the E x T path array dominates */
    if (buf_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("path-array bytes %zu exceed --max-mb %lld", buf_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D7 MapReduce/MonteCarlo");
    p2_meta_kv_str(&m, "scheme", "Monte-Carlo European-call pricer (GBM paths stored in bulk, discounted mean payoff)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "paths", paths);
    p2_meta_kv_i64(&m, "steps", steps);
    p2_meta_kv_f64(&m, "spot", spot);
    p2_meta_kv_f64(&m, "strike", strike);
    p2_meta_kv_f64(&m, "rate", rate);
    p2_meta_kv_f64(&m, "vol", vol);
    p2_meta_kv_f64(&m, "maturity", maturity);
    p2_meta_kv_u64(&m, "path_array_bytes", buf_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The E x T path array is the dominant buffer -> mmap + mlock it. It is
     * rewritten front-to-back every measure pass (the signature bulk write). */
    double *S = (double *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (S == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", buf_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(S, buf_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(S, buf_bytes);

    /* GBM per-step coefficients (constant across paths and passes). */
    const double dt    = maturity / (double)T;
    const double drift = (rate - 0.5 * vol * vol) * dt;   /* deterministic log-drift */
    const double vsdt  = vol * sqrt(dt);                  /* diffusion scale per step */
    const double disc  = exp(-rate * maturity);           /* discount factor e^{-rT} */
    const double TWO_PI = 6.283185307179586476925286766559;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Touch the whole array once (first-write faulting) so the measured passes
     * time the compute + rewrite, not the initial page population. */
    for (size_t i = 0; i < cells; i++) S[i] = spot;
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    double price = 0.0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed each pass so every pass regenerates the identical path set: a
         * deterministic, repeatable bulk write of the whole E x T footprint. */
        p2_rng_t rng; p2_rng_seed(&rng, seed);
        double payoff_sum = 0.0;
        for (size_t e = 0; e < E; e++) {
            double *row = S + e * T;                 /* this path's contiguous row */
            double s = spot;
            size_t t = 0;
            /* Draw normals in Box-Muller pairs; consume both the cos and sin
             * halves so no uniform is wasted. */
            while (t < T) {
                double u1 = p2_rng_unit(&rng);
                double u2 = p2_rng_unit(&rng);
                double rmag = sqrt(-2.0 * log(u1));
                double z0 = rmag * cos(TWO_PI * u2);
                s *= exp(drift + vsdt * z0);         /* (1)+(2) GBM step */
                row[t++] = s;                        /* (3) store path point */
                if (t < T) {
                    double z1 = rmag * sin(TWO_PI * u2);
                    s *= exp(drift + vsdt * z1);
                    row[t++] = s;
                }
            }
            /* (4) reduce: terminal-column payoff of a European call. */
            double term = row[T - 1];
            double pay = term - strike;
            if (pay > 0.0) payoff_sum += pay;
        }
        price = disc * (payoff_sum / (double)E);     /* discounted mean payoff */
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = S[(E / 2) * T + (T - 1)];  /* a live terminal price */

    munmap(S, buf_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "mc_price", price);
    p2_meta_kv_f64(&m, "terminal_sample", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "MC price carries O(1/sqrt(paths)) statistical error vs Black-Scholes; bulk path storage is the distinct write vs histogram scatter");
    p2_meta_close(&m);
    return 0;
}
