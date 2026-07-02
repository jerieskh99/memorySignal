/* kernel_hmm_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  SCALED HMM FORWARD:  fill a T x S trellis column-by-column via dense matvec.
 * ============================================================================
 *
 *  DWARF   : Graphical Models (D12)      (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the host write-signal of a wavefront whose every step is a
 *            dense matrix-vector product: unlike the edit-distance DP (cheap
 *            per-cell, neighbour-local reads), each new trellis column reads the
 *            ENTIRE previous column through the S x S transition matrix, so the
 *            arithmetic-per-write is far higher for the same monotone front.
 *
 *  PICTURE (top view):
 *      The trellis alpha[t][s] holds, for each timestep t and hidden state s,
 *      the scaled probability of the observations up to t ending in state s.
 *      It is filled one COLUMN at a time, left to right; the live write band is
 *      one column (S cells) wide and marches along the time axis.
 *
 *              t=0   t=1   t=2  ...  t=T-1
 *        s=0 [  p    p    |==                ]
 *        s=1 [  p    p    |==   ?   ...   ?   ]      column t reads ALL of
 *        s=2 [  p    p    |==   ?   ...   ?   ]      column t-1, weighted by A:
 *        ... [  p    p    |==   ?   ...   ?   ]        cur[s] = (sum_s' prev[s']
 *      s=S-1 [  p    p    |==   ?   ...   ?   ]                    * A[s'][s])
 *                          ^^                                     * B[s][obs[t]]
 *                     write front (each column normalised to sum to 1)
 *
 *  ALGORITHM (per forward pass over the whole trellis):
 *      1. Re-randomise the observation sequence obs[] (fresh evidence each pass;
 *         the fixed model A, B, pi is reused, but the trellis is recomputed).
 *      2. Column 0: alpha[0][s] = pi[s] * B[s][obs[0]], then normalise so the
 *         column sums to 1.
 *      3. Columns t = 1..T-1: for each state s, form the dense matrix-vector
 *         product over the previous column, then apply the emission weight:
 *             alpha[t][s] = ( sum_s' alpha[t-1][s'] * A[s'][s] ) * B[s][obs[t]]
 *         Normalise the finished column to sum to 1. This per-column rescaling
 *         is the standard "scaling" trick that keeps the products from
 *         underflowing to zero over long sequences.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A large trellis written by a column-wise MONOTONE front (a wavefront,
 *      like the DP kernel). Two things distinguish it: the written values are
 *      NORMALISED probabilities (bounded, distinct content), and each column is
 *      backed by a dense read of the whole previous column through A -- a high
 *      compute-to-write ratio rather than the DP's cheap local recurrence.
 *      The transition matrix A and emission matrix B are read-only and invisible
 *      to a write-only host signal, so only the migrating column front shows.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Graphical Models. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (forward passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_hmm_v2";
#define OBS_ALPHABET 256

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign scaled HMM forward; graphical-models kernel)\n"
"  --states S            Hidden states (default 256; A is S*S)\n"
"  --steps T             Timesteps / trellis columns (default 65536; trellis T*S*8 bytes)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

static inline double rng_unit(p2_rng_t *r) {
    return (double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Fill a row-stochastic matrix M[rows][cols]: every row is a random probability
 * distribution (non-negative and summing to 1). Used to build the transition
 * matrix A, the emission matrix B, and the initial distribution pi so the model
 * is a valid HMM. The small 1e-6 floor keeps every entry strictly positive,
 * which guarantees the row sum s is non-zero before the normalising divide. */
static void fill_stochastic(double *M, size_t rows, size_t cols, p2_rng_t *rng) {
    for (size_t i = 0; i < rows; i++) {
        double s = 0.0; double *row = M + i * cols;
        for (size_t j = 0; j < cols; j++) { row[j] = rng_unit(rng) + 1e-6; s += row[j]; }  /* raw weights + running sum */
        for (size_t j = 0; j < cols; j++) row[j] /= s;                                     /* normalise the row to sum to 1 */
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long states     = p2_get_i64(argc, argv, "--states", 256);
    long long steps      = p2_get_i64(argc, argv, "--steps", 65536);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (states < 16 || states > 4096) { P2_LOG_ERR("states %lld out of range (16..4096)", states); return 2; }
    if (steps < 256 || steps > (1LL << 28)) { P2_LOG_ERR("steps %lld out of range", steps); return 2; }
    size_t S = (size_t)states, T = (size_t)steps, O = OBS_ALPHABET;
    size_t tre = T * S;                       /* trellis cells */
    size_t bytes = tre * sizeof(double);      /* the trellis dominates */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("trellis bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Graphical Models");
    p2_meta_kv_str(&m, "scheme", "scaled HMM forward (trellis fill, dense transition matvec/column)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "states", states);
    p2_meta_kv_i64(&m, "steps", steps);
    p2_meta_kv_u64(&m, "trellis_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *alpha = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (alpha == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(alpha, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(alpha, bytes);

    double *A = (double *)malloc(S * S * sizeof(double));
    double *B = (double *)malloc(S * O * sizeof(double));
    double *pi = (double *)malloc(S * sizeof(double));
    uint8_t *obs = (uint8_t *)malloc(T);
    if (!A || !B || !pi || !obs) { free(A); free(B); free(pi); free(obs); munmap(alpha, bytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    fill_stochastic(A, S, S, &rng);
    fill_stochastic(B, S, O, &rng);
    fill_stochastic(pi, 1, S, &rng);
    for (size_t t = 0; t < T; t++) obs[t] = (uint8_t)p2_rng_next(&rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        for (size_t t = 0; t < T; t++) obs[t] = (uint8_t)p2_rng_next(&rng);   /* fresh observations */
        /* Column 0 is the base case: prior pi weighted by the emission for the
         * first observation. Normalising it starts the scaled recurrence off
         * from a proper probability distribution. */
        double c0 = 0.0;
        for (size_t s = 0; s < S; s++) { double v = pi[s] * B[s * O + obs[0]]; alpha[s] = v; c0 += v; }
        if (c0 > 0) for (size_t s = 0; s < S; s++) alpha[s] /= c0;   /* guard: skip divide if the whole column is zero */
        /* Columns 1..T-1 are the wavefront: each column t is computed purely
         * from the (already normalised) column t-1, so the write front marches
         * strictly along the time axis and never revisits an earlier column. */
        for (size_t t = 1; t < T; t++) {
            const double *prev = alpha + (t - 1) * S;   /* read-only: the whole previous column */
            double *cur = alpha + t * S;                /* the column currently being written */
            uint8_t ot = obs[t];
            double csum = 0.0;
            for (size_t s = 0; s < S; s++) {
                /* Dense matrix-vector product: accumulate the previous column
                 * against column s of the transition matrix A. This inner loop
                 * over sp is where the O(T*S^2) work (and the high compute-per-
                 * write ratio that distinguishes this kernel) lives. */
                double acc = 0.0;
                for (size_t sp = 0; sp < S; sp++) acc += prev[sp] * A[sp * S + s];
                double v = acc * B[s * O + ot];         /* weight by the emission for obs[t] */
                cur[s] = v; csum += v;                  /* write the cell, accumulate the column sum */
            }
            /* Scaling step: rescale the finished column to sum to 1 so the
             * products cannot underflow to zero over a long observation
             * sequence (the standard HMM-forward numerical safeguard). */
            if (csum > 0) for (size_t s = 0; s < S; s++) cur[s] /= csum;
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = alpha[(T - 1) * S + S / 2];   /* a normalised probability in [0,1] */

    free(A); free(B); free(pi); free(obs);
    munmap(alpha, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "last_prob", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "trellis column-fill front + dense transition matvec; normalised-probability content");
    p2_meta_close(&m);
    return 0;
}
