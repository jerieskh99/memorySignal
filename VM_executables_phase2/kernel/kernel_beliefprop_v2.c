/* kernel_beliefprop_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  LOOPY BELIEF PROPAGATION:  sum-product message passing on a 2D grid MRF.
 * ============================================================================
 *
 *  DWARF   : Graphical Models (D12)      (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the host write-signal of an ITERATED message-passing solver.
 *            The HMM forward kernel fills one trellis ONCE with a monotone
 *            column front; loopy BP instead keeps a bank of message vectors on
 *            every edge and rewrites all of them, in place, on every sweep. The
 *            distinctive write is therefore not a migrating front but a large
 *            resident array that is entirely overwritten T times per pass.
 *
 *  PICTURE (top view of the W x H node grid; each node holds K states):
 *      Every node exchanges a K-vector message with each of its four grid
 *      neighbours. Store the four INCOMING directed messages per node, so the
 *      buffer is msg[node][dir][state] with dir in {UP,DOWN,LEFT,RIGHT}. A
 *      sweep recomputes every one of those vectors from the other three plus
 *      the local potential -- the whole grid of arrows is redrawn each sweep.
 *
 *              (i-1,j)                     dir encoding (incoming to a node):
 *                 |  msg UP                  UP    = from the neighbour above
 *                 v                          DOWN  = from the neighbour below
 *      (i,j-1)-->[ node (i,j) ]<--(i,j+1)    LEFT  = from the neighbour left
 *         msg LEFT   ^   msg RIGHT           RIGHT = from the neighbour right
 *                    |  msg DOWN
 *                 (i+1,j)
 *
 *      New message node->neighbour(dir) is built from unary(node) times the
 *      product of node's incoming messages from the OTHER THREE directions,
 *      pushed through the pairwise Potts factor and summed over the sender's
 *      states -- then normalised so the K-vector sums to 1.
 *
 *  ALGORITHM (per measure pass over the whole grid):
 *      1. Re-seed the per-node unary potentials u[node][state] (fresh evidence
 *         each pass; the fixed Potts pairwise factor is reused) and reset every
 *         message vector to the uniform 1.0 (a proper "no information" start).
 *      2. For T sweeps: for every node and every outgoing direction, form the
 *         belief that this node would send to that neighbour --
 *             m_out(t) = sum_s [ u(s) * (prod of the 3 OTHER incoming msgs)(s)
 *                                       * pairwise(s, t) ]
 *         and deposit it into the neighbour's matching incoming slot. Each
 *         out-message is normalised to sum to 1 so the products cannot drift
 *         or underflow across sweeps (the standard sum-product safeguard).
 *      3. Count the pass. The message bank is rewritten in full on every sweep,
 *         so over one pass it is overwritten T times.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A large, RESIDENT bank of K-vectors (4*K doubles per grid cell) that is
 *      overwritten every sweep -- iterated in-place message arrays, unlike the
 *      HMM's single write-once trellis fill. Two tells set it apart: the write
 *      set is the whole message buffer revisited T times per pass (not a
 *      one-shot monotone front), and the written values are NORMALISED
 *      probabilities (bounded, distinct content). The read-only unary and
 *      pairwise potentials are invisible to a write-only host signal, so only
 *      the churning message bank shows.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Graphical Models. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (BP passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_beliefprop_v2";

/* Four grid directions. The value stored at msg[node][DIR] is the message that
 * arrived at `node` FROM the neighbour lying in that direction. The opposite
 * direction is used when depositing an out-message into that neighbour's slot:
 * a message we send UP to the neighbour above lands in that neighbour's DOWN
 * incoming slot, and vice-versa; likewise LEFT<->RIGHT. */
enum { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, NDIR = 4 };
static inline int opposite(int d) {
    switch (d) { case UP: return DOWN; case DOWN: return UP;
                 case LEFT: return RIGHT; default: return LEFT; }
}

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign loopy belief propagation; graphical-models kernel)\n"
"  --width W             Grid width in nodes (default 256)\n"
"  --height H            Grid height in nodes (default 256)\n"
"  --states K            Discrete states per node (default 4; msg bank W*H*4*K*8 bytes)\n"
"  --iters T             Sum-product sweeps per pass (default 10)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on message-bank bytes (default 8192)\n"
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

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long width      = p2_get_i64(argc, argv, "--width", 256);
    long long height     = p2_get_i64(argc, argv, "--height", 256);
    long long states     = p2_get_i64(argc, argv, "--states", 4);
    long long iters      = p2_get_i64(argc, argv, "--iters", 10);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (width < 2 || width > 8192)   { P2_LOG_ERR("width %lld out of range (2..8192)", width); return 2; }
    if (height < 2 || height > 8192) { P2_LOG_ERR("height %lld out of range (2..8192)", height); return 2; }
    if (states < 2 || states > 256)  { P2_LOG_ERR("states %lld out of range (2..256)", states); return 2; }
    if (iters < 1 || iters > 100000) { P2_LOG_ERR("iters %lld out of range (1..100000)", iters); return 2; }

    size_t W = (size_t)width, H = (size_t)height, K = (size_t)states, T = (size_t)iters;
    size_t nodes = W * H;                          /* grid nodes                          */
    size_t vecs  = nodes * NDIR;                   /* one incoming K-vector per (node,dir) */
    size_t bytes = vecs * K * sizeof(double);      /* the message bank dominates           */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("message-bank bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Graphical Models");
    p2_meta_kv_str(&m, "scheme", "loopy belief propagation (sum-product, 2D grid MRF; message bank rewritten each sweep)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "width", width);
    p2_meta_kv_i64(&m, "height", height);
    p2_meta_kv_i64(&m, "states", states);
    p2_meta_kv_i64(&m, "iters", iters);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The message bank is the dominant buffer -> mmap + mlock it (it is
     * overwritten every sweep, which is the workload's signature write). */
    double *msg = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (msg == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(msg, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(msg, bytes);

    /* Per-node unary potentials u[node*K + s] (re-seeded each pass), the fixed
     * K x K Potts pairwise factor, and a small scratch K-vector for building an
     * out-message before it is normalised and deposited. These read-only /
     * scratch buffers are tiny next to the message bank. */
    double *unary   = (double *)malloc(nodes * K * sizeof(double));
    double *pair    = (double *)malloc(K * K * sizeof(double));
    double *incoming= (double *)malloc(K * sizeof(double));   /* product of the 3 other incoming msgs */
    double *out     = (double *)malloc(K * sizeof(double));   /* candidate out-message, pre-normalise  */
    if (!unary || !pair || !incoming || !out) {
        free(unary); free(pair); free(incoming); free(out); munmap(msg, bytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Fixed Potts pairwise factor: 1 on the diagonal (neighbours in the same
     * state), exp(-beta) < 1 off-diagonal (a smoothness prior that penalises
     * disagreeing neighbours). Built once and reused across every pass. */
    const double beta = 0.7, off = exp(-beta);
    for (size_t a = 0; a < K; a++)
        for (size_t b = 0; b < K; b++)
            pair[a * K + b] = (a == b) ? 1.0 : off;
    /* Seed the unary potentials for the first pass so warmup touches the state. */
    for (size_t i = 0; i < nodes * K; i++) unary[i] = rng_unit(&rng) + 1e-6;
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* (1) Fresh evidence: re-seed the unary potentials and reset every
         * message vector to the uniform 1.0 (proper "no information" start). */
        for (size_t i = 0; i < nodes * K; i++) unary[i] = rng_unit(&rng) + 1e-6;
        for (size_t i = 0; i < vecs * K; i++) msg[i] = 1.0;

        /* (2) T synchronous-ish sweeps. Messages are read and written in the
         * same bank, so each new out-message immediately overwrites its slot;
         * over one pass the entire bank is rewritten T times -- the tell. */
        for (size_t sweep = 0; sweep < T; sweep++) {
            for (size_t i = 0; i < H; i++) {
                for (size_t j = 0; j < W; j++) {
                    size_t node = i * W + j;
                    const double *u = unary + node * K;
                    double *mnode = msg + node * NDIR * K;   /* this node's 4 incoming vectors */

                    /* For each outgoing direction that has a neighbour, form the
                     * message this node sends there. */
                    for (int d = 0; d < NDIR; d++) {
                        /* Locate the neighbour in direction d; skip grid edges. */
                        size_t ni = i, nj = j;
                        if (d == UP)    { if (i == 0) continue;      ni = i - 1; }
                        else if (d == DOWN)  { if (i + 1 == H) continue; ni = i + 1; }
                        else if (d == LEFT)  { if (j == 0) continue;     nj = j - 1; }
                        else /* RIGHT */     { if (j + 1 == W) continue; nj = j + 1; }

                        /* Product of this node's incoming messages from the
                         * OTHER THREE directions (exclude d itself): this is the
                         * evidence the node forwards, minus what came from the
                         * target neighbour (the sum-product exclusion rule). */
                        for (size_t s = 0; s < K; s++) incoming[s] = u[s];
                        for (int e = 0; e < NDIR; e++) {
                            if (e == d) continue;
                            const double *me = mnode + (size_t)e * K;
                            for (size_t s = 0; s < K; s++) incoming[s] *= me[s];
                        }

                        /* Push through the pairwise Potts factor and sum over the
                         * sender's states s to get the out-message over the
                         * receiver's states t: out(t) = sum_s incoming(s)*pair(s,t). */
                        double osum = 0.0;
                        for (size_t t = 0; t < K; t++) {
                            double acc = 0.0;
                            for (size_t s = 0; s < K; s++) acc += incoming[s] * pair[s * K + t];
                            out[t] = acc; osum += acc;
                        }
                        /* Normalise the out-message to sum to 1 (sum-product
                         * safeguard against drift/underflow across sweeps). */
                        if (osum > 0.0) for (size_t t = 0; t < K; t++) out[t] /= osum;

                        /* Deposit into the neighbour's matching incoming slot:
                         * what we send in direction d lands in the neighbour's
                         * opposite-direction incoming vector. This write is the
                         * per-cell churn the host signal observes. */
                        size_t nnode = ni * W + nj;
                        double *dst = msg + nnode * NDIR * K + (size_t)opposite(d) * K;
                        for (size_t t = 0; t < K; t++) dst[t] = out[t];
                    }
                }
            }
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Final sink: the belief at the grid centre for state 0 -- the unary
     * potential times the product of all four incoming messages, normalised
     * over states. A bounded probability in [0,1] that also forces the last
     * pass's message writes to be observed. */
    volatile double sink = 0.0;
    {
        size_t cnode = (H / 2) * W + (W / 2);
        const double *u = unary + cnode * K;
        const double *mn = msg + cnode * NDIR * K;
        double bsum = 0.0, b0 = 0.0;
        for (size_t s = 0; s < K; s++) {
            double b = u[s];
            for (int e = 0; e < NDIR; e++) b *= mn[(size_t)e * K + s];
            if (s == 0) b0 = b;
            bsum += b;
        }
        sink = (bsum > 0.0) ? (b0 / bsum) : 0.0;
    }

    free(unary); free(pair); free(incoming); free(out);
    munmap(msg, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "center_belief0", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "iterated in-place message bank overwritten T times/pass (vs hmm's write-once trellis); "
                   "normalised-probability content; loopy grid BP is approximate (exact only on trees)");
    p2_meta_close(&m);
    return 0;
}
