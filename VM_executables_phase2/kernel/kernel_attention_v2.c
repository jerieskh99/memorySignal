/* kernel_attention_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  SCALED DOT-PRODUCT ATTENTION:  the transformer core, O = softmax(Q K^T / sqrt(d)) V
 * ============================================================================
 *
 *  DWARF   : Dense Linear Algebra (D1)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the write behaviour of the single most-executed kernel in
 *            modern AI. It is two dense matrix multiplies wrapped around an
 *            in-place row-wise softmax, so it stresses the host write-signal
 *            with a large, repeatedly overwritten intermediate matrix.
 *
 *  PICTURE (top view):
 *
 *      Q (L x d)   K (L x d)                        V (L x d)
 *         \           /                                |
 *          \  Q K^T  /   scale by 1/sqrt(d)            |
 *           v       v                                  |
 *          +-----------------+   row softmax  +-----------------+     O (L x d)
 *          |   S  (L x L)    | -------------> |   P  (L x L)    | --*--> +------+
 *          |  raw scores     |  (in place)    | each row sums=1 |  P V  | out  |
 *          +-----------------+                +-----------------+       +------+
 *              ^ the L x L score/probability matrix dominates the footprint
 *
 *  ALGORITHM (per measured pass):
 *      1. Re-seed Q, K, V with fresh random values (keeps the compute honest
 *         and stops the optimiser from hoisting the whole pass out of the loop).
 *      2. First matmul:  S = (Q K^T) * (1/sqrt(d))  -- fill the L x L score matrix.
 *      3. Row softmax IN PLACE: for each row of S subtract the row max (numerical
 *         stability), exponentiate, and normalise so the row sums to 1. S now
 *         holds the attention probabilities P.
 *      4. Second matmul:  O = P V  -- each output row is a probability-weighted
 *         average of the value rows.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      The L x L score matrix is written once by the QK^T matmul and then
 *      overwritten again in place by the softmax, so the dominant tell is a
 *      large dense buffer that is fully rewritten twice per pass. HONEST CAVEAT:
 *      to a write-only observer this is essentially a GEMM footprint plus one
 *      row-normalise pass -- it is deliberately close to plain matrix multiply.
 *      It is included anyway for real-world coverage, because attention is the
 *      most-run kernel in every transformer / large-language-model forward pass.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Dense Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (attention passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_attention_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign scaled dot-product attention; transformer Dense-LA kernel)\n"
"  --seq-len L           Sequence length / tokens (default 1024; scores are L*L*8 bytes)\n"
"  --d-model D           Head dimension (default 64)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on score-matrix bytes (default 8192)\n"
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

/* Overwrite an n-element matrix with fresh uniform random values in [-1,1).
 * Called once per pass on Q, K and V so every pass does real, non-repeating
 * work; without it the compiler could prove the result invariant and skip it. */
static void reseed(double *M, size_t n, p2_rng_t *rng) {
    for (size_t k = 0; k < n; k++) M[k] = 2.0 * rng_unit(rng) - 1.0;   /* [-1,1) */
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long seq        = p2_get_i64(argc, argv, "--seq-len", 1024);
    long long dmodel     = p2_get_i64(argc, argv, "--d-model", 64);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (seq < 16 || seq > 32768) { P2_LOG_ERR("seq-len %lld out of range (16..32768)", seq); return 2; }
    if (dmodel < 8 || dmodel > 1024) { P2_LOG_ERR("d-model %lld out of range (8..1024)", dmodel); return 2; }
    size_t L = (size_t)seq, D = (size_t)dmodel;
    size_t sbytes = L * L * sizeof(double);           /* score matrix dominates */
    if (sbytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("score bytes %zu exceed --max-mb %lld", sbytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Dense Linear Algebra");
    p2_meta_kv_str(&m, "scheme", "scaled dot-product attention (QK^T, row softmax, *V); transformer core");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "seq_len", seq);
    p2_meta_kv_i64(&m, "d_model", dmodel);
    p2_meta_kv_u64(&m, "score_bytes", sbytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *S = (double *)mmap(NULL, sbytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (S == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", sbytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(S, sbytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(S, sbytes);

    double *Q = (double *)malloc(L * D * sizeof(double));
    double *K = (double *)malloc(L * D * sizeof(double));
    double *V = (double *)malloc(L * D * sizeof(double));
    double *O = (double *)malloc(L * D * sizeof(double));
    if (!Q || !K || !V || !O) { free(Q); free(K); free(V); free(O); munmap(S, sbytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    const double scale = 1.0 / sqrt((double)D);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    reseed(Q, L * D, &rng); reseed(K, L * D, &rng); reseed(V, L * D, &rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        reseed(Q, L * D, &rng); reseed(K, L * D, &rng); reseed(V, L * D, &rng);

        /* (1) First matmul: S = (Q K^T) * scale. Score S[i][j] is the scaled dot
         * product of query row i with key row j, i.e. how much token i should
         * attend to token j. This fills the whole L x L score matrix -- the
         * dominant, most-written buffer of the workload. */
        for (size_t i = 0; i < L; i++) {
            const double *qi = Q + i * D;               /* query row i (length D) */
            double *si = S + i * L;                      /* score row i (length L) */
            for (size_t j = 0; j < L; j++) {
                const double *kj = K + j * D;            /* key row j (length D)   */
                double acc = 0.0;
                for (size_t t = 0; t < D; t++) acc += qi[t] * kj[t];   /* dot product */
                si[j] = acc * scale;                     /* scale by 1/sqrt(D)     */
            }
        }

        /* (2) Row-wise softmax, IN PLACE, turning raw scores into probabilities.
         * We subtract each row's maximum before exp() so the largest exponent is
         * exp(0) = 1: this is the standard shift that keeps exp() from overflowing
         * without changing the mathematical result (softmax is shift-invariant).
         * After this pass every row of S is non-negative and sums to 1. This is
         * the SECOND full rewrite of the L x L matrix in the same pass. */
        for (size_t i = 0; i < L; i++) {
            double *si = S + i * L;
            double mx = si[0];                           /* row max for stability  */
            for (size_t j = 1; j < L; j++) if (si[j] > mx) mx = si[j];
            double sum = 0.0;
            for (size_t j = 0; j < L; j++) { double e = exp(si[j] - mx); si[j] = e; sum += e; }
            double inv = sum > 0.0 ? 1.0 / sum : 0.0;    /* guard the empty/degenerate row */
            for (size_t j = 0; j < L; j++) si[j] *= inv; /* normalise so row sums to 1 */
        }

        /* (3) Second matmul: O = P V. Output row i is the softmax-weighted sum of
         * the value rows, so each token's output is a convex combination of all
         * value vectors weighted by its attention distribution. */
        for (size_t i = 0; i < L; i++) {
            const double *pi = S + i * L;                /* attention weights for token i */
            double *oi = O + i * D;                       /* output row i (length D)       */
            for (size_t t = 0; t < D; t++) oi[t] = 0.0;
            for (size_t j = 0; j < L; j++) {
                double w = pi[j];                         /* weight on value row j */
                const double *vj = V + j * D;
                for (size_t t = 0; t < D; t++) oi[t] += w * vj[t];
            }
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* a probability-weighted output value (finite, bounded by the V range) */
    volatile double sink = O[(L - 1) * D + (D / 2)];
    /* softmax sanity: last row should sum to ~1 */
    double rowsum = 0.0; for (size_t j = 0; j < L; j++) rowsum += S[(L - 1) * L + j];

    free(Q); free(K); free(V); free(O);
    munmap(S, sbytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "last_out", (double)sink);
    p2_meta_kv_f64(&m, "last_row_softmax_sum", rowsum);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "write signature ~ GEMM (two matmuls + a row softmax); included for real-world coverage");
    p2_meta_close(&m);
    return 0;
}
