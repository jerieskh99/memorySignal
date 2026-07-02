/* kernel_ldpc_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Graphical-Models dwarf (Berkeley motif D12), the LDPC belief-propagation
 * variant. A low-density parity-check (LDPC) code protects data with a sparse
 * set of parity constraints; decoding it is min-sum BELIEF PROPAGATION on a
 * bipartite TANNER GRAPH -- one side is the N variable nodes (the code bits),
 * the other is the M check nodes (the parity constraints), and an edge joins a
 * variable to every check it participates in. Decoding passes messages back and
 * forth along those edges until the bits settle. This is bipartite message
 * passing, a write-cousin of grid belief-propagation (kernel_hmm_v2's trellis
 * is a chain graph) but on an irregular Tanner graph instead of a lattice.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE
 * ----------------------------------------------------------------------------
 * The dominant, distinctive write is the pair of EDGE-MESSAGE arrays: one double
 * LLR per edge for the variable->check direction and one for the check->variable
 * direction (2 * E doubles, E = N*dv edges). Every min-sum iteration overwrites
 * both arrays in full -- first all check->variable messages, then all
 * variable->check messages -- so the host write-signal sees two large buffers
 * scanned and rewritten T times per decode, indexed through the sparse graph
 * rather than in dense row order. The read-only graph structure (which variable
 * touches which check) is invisible to a write-only host signal; only the
 * message traffic on the edges shows. Unlike the HMM trellis (a monotone
 * column front that never revisits a column), here BOTH message arrays are
 * revisited every iteration -- an iterated, whole-buffer rewrite, not a
 * migrating front.
 *
 * ============================================================================
 * ALGORITHM (per measure pass = one full decode)
 * ============================================================================
 * Setup (once, at startup): build a regular Tanner graph. Each variable sits in
 * exactly dv checks and each check covers exactly dc variables (a random-regular
 * construction via a shuffled check-socket list, so M = N*dv/dc). The graph is
 * stored as edge lists grouped both by check and by variable so each update can
 * walk the OTHER edges of a node cheaply.
 *
 * Each pass then decodes one received word:
 *   1. Transmit the ALL-ZERO codeword. The all-zero word satisfies every linear
 *      parity check, so it is a valid codeword of ANY parity-check matrix -- no
 *      generator matrix is needed to produce a legal transmit word.
 *   2. Binary symmetric channel: flip each bit independently with probability
 *      flip (default 0.02). The flipped positions are the errors the decoder
 *      must find. Form the channel LLR for each bit: +Lc if the received bit is
 *      0, -Lc if it is 1 (a positive LLR favours bit 0).
 *   3. Run T (default 20) min-sum iterations:
 *        a. Check-node update. For each edge of a check, the outgoing
 *           check->variable message is (product of the SIGNS of the incoming
 *           variable->check messages on the OTHER edges) times (the MINIMUM of
 *           their magnitudes). This is the min-sum approximation to the exact
 *           sum-product tanh rule.
 *        b. Variable-node update. For each edge of a variable, the outgoing
 *           variable->check message is the channel LLR plus the sum of the
 *           incoming check->variable messages on the OTHER edges (extrinsic
 *           information: exclude the edge we are about to send on).
 *   4. Hard decision. For each bit, form the total belief = channel LLR + sum of
 *      ALL incoming check->variable messages; decide bit 0 if that is >= 0, else
 *      bit 1. Then compute the SYNDROME weight: the number of checks whose
 *      covered decided bits XOR to 1 (unsatisfied parity). Zero means the
 *      decoder recovered a valid codeword.
 *
 * MEMORY SIGNATURE (what the host write-signal actually sees):
 *   Two large edge-message buffers (2 * E doubles) rewritten in full on every
 *   one of the T iterations, addressed through the sparse graph. The graph
 *   structure and the small per-bit LLR/decision arrays are dwarfed by this
 *   iterated message traffic.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Graphical Models. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + graph build + init) / measure (decode passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_ldpc_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign LDPC min-sum belief-propagation; graphical-models kernel)\n"
"  --bits N              Variable nodes / code bits (default 4096; M = N*dv/dc checks)\n"
"  --dv D                Variable degree: checks per bit (default 3)\n"
"  --dc D                Check degree: bits per check (default 6)\n"
"  --flip-milli F        Channel flip probability x1000 (default 20 = 0.02)\n"
"  --iters T             Min-sum iterations per decode (default 20)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on edge-message bytes (default 8192)\n"
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

/* Channel LLR magnitude. A fixed positive constant is enough for min-sum: the
 * decoder cares about the SIGN pattern and the relative magnitudes of the
 * messages, and min-sum is invariant to a global scaling of all LLRs, so the
 * exact value only sets the units of the beliefs. */
#define CHANNEL_LLR 2.0

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long bits       = p2_get_i64(argc, argv, "--bits", 4096);
    long long dv         = p2_get_i64(argc, argv, "--dv", 3);
    long long dc         = p2_get_i64(argc, argv, "--dc", 6);
    long long flip_milli = p2_get_i64(argc, argv, "--flip-milli", 20);
    long long iters      = p2_get_i64(argc, argv, "--iters", 20);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    double flip = (double)flip_milli / 1000.0;

    if (bits < 64 || bits > (1LL << 24)) { P2_LOG_ERR("bits %lld out of range (64..2^24)", bits); return 2; }
    if (dv < 2 || dv > 16) { P2_LOG_ERR("dv %lld out of range (2..16)", dv); return 2; }
    if (dc < 3 || dc > 64) { P2_LOG_ERR("dc %lld out of range (3..64)", dc); return 2; }
    /* Regularity constraint: every edge counted from the variable side (N*dv)
     * must partition evenly into checks of degree dc, i.e. M = N*dv/dc must be a
     * whole number. Otherwise no regular Tanner graph with these degrees exists. */
    if ((bits * dv) % dc != 0) {
        P2_LOG_ERR("bits*dv (%lld) not divisible by dc (%lld): no regular graph", bits * dv, dc); return 2;
    }
    if (flip < 0.0 || flip > 0.5) { P2_LOG_ERR("flip %.3f out of range (0..0.5)", flip); return 2; }
    if (iters < 1 || iters > 4096) { P2_LOG_ERR("iters %lld out of range (1..4096)", iters); return 2; }

    size_t N = (size_t)bits;                 /* variable nodes (code bits) */
    size_t DV = (size_t)dv, DC = (size_t)dc;
    size_t E = N * DV;                        /* edges = N*dv = M*dc */
    size_t M = E / DC;                        /* check nodes */
    size_t T = (size_t)iters;
    /* The two edge-message arrays (var->check and check->var) are the dominant,
     * signature buffer; everything else (graph indices, per-bit LLR/decision) is
     * comparatively tiny. Cap on this message footprint. */
    size_t msg_bytes = (size_t)2 * E * sizeof(double);
    if (msg_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("edge-message bytes %zu exceed --max-mb %lld", msg_bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Graphical Models");
    p2_meta_kv_str(&m, "scheme", "LDPC min-sum belief propagation (bipartite Tanner-graph message passing, iterated edge messages)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "bits", bits);
    p2_meta_kv_i64(&m, "dv", dv);
    p2_meta_kv_i64(&m, "dc", dc);
    p2_meta_kv_u64(&m, "checks", M);
    p2_meta_kv_u64(&m, "edges", E);
    p2_meta_kv_i64(&m, "flip_milli", flip_milli);
    p2_meta_kv_i64(&m, "iters", iters);
    p2_meta_kv_u64(&m, "total_bytes", msg_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* --- Dominant buffer: the two edge-message arrays, mmap'd together. ------
     * Layout is one contiguous mapping of 2*E doubles: [0, E) is the
     * variable->check direction (m_vc), [E, 2*E) is the check->variable
     * direction (m_cv). This is the buffer rewritten in full every iteration. */
    double *msg = (double *)mmap(NULL, msg_bytes, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (msg == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", msg_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(msg, msg_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(msg, msg_bytes);
    double *m_vc = msg;          /* variable -> check messages, one per edge */
    double *m_cv = msg + E;      /* check -> variable messages, one per edge */

    /* --- Graph structure (read-only after build) and per-bit working state. --
     * Edge e joins variable edge_var[e] to check edge_chk[e]. To update a node
     * over its OTHER edges we also keep CSR-style edge lists grouped by variable
     * (var_edges, dv per variable) and by check (chk_edges, dc per check). */
    int      *edge_var  = (int *)malloc(E * sizeof(int));
    int      *edge_chk  = (int *)malloc(E * sizeof(int));
    int      *var_edges = (int *)malloc(E * sizeof(int));   /* N * dv, grouped by variable */
    int      *chk_edges = (int *)malloc(E * sizeof(int));   /* M * dc, grouped by check   */
    double   *llr       = (double *)malloc(N * sizeof(double)); /* channel LLR per bit */
    uint8_t  *decision  = (uint8_t *)malloc(N);                 /* hard-decided bits   */
    /* Scratch for the check-socket construction: a shuffled list in which each
     * check index appears exactly dc times, consumed dv-per-variable. */
    int      *sockets   = (int *)malloc(E * sizeof(int));
    if (!edge_var || !edge_chk || !var_edges || !chk_edges || !llr || !decision || !sockets) {
        free(edge_var); free(edge_chk); free(var_edges); free(chk_edges);
        free(llr); free(decision); free(sockets); munmap(msg, msg_bytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);

    /* --- Build the regular Tanner graph via a shuffled check-socket list. -----
     * Fill sockets[] so each check index c appears exactly dc times, then
     * Fisher-Yates shuffle. Reading it in blocks of dv gives each variable dv
     * checks and leaves each check with exactly dc edges (perfect regularity).
     * We only reject a block that would put the SAME check on a variable twice
     * (parallel edges); a single local swap with a later socket fixes it, which
     * for dv much smaller than M essentially always succeeds. */
    for (size_t c = 0; c < M; c++)
        for (size_t j = 0; j < DC; j++) sockets[c * DC + j] = (int)c;
    for (size_t i = E; i > 1; i--) {                 /* Fisher-Yates, unbiased */
        size_t j = (size_t)(p2_rng_next(&rng) % i);
        int tmp = sockets[i - 1]; sockets[i - 1] = sockets[j]; sockets[j] = tmp;
    }
    for (size_t v = 0; v < N; v++) {
        size_t base = v * DV;
        for (size_t k = 0; k < DV; k++) {
            /* Ensure this variable's k-th check differs from its earlier ones by
             * swapping the offending socket forward until it is distinct (or we
             * run out of later sockets, which regularity makes vanishingly rare). */
            for (size_t attempt = base + k + 1; attempt < E; attempt++) {
                int dup = 0;
                for (size_t p = 0; p < k; p++)
                    if (sockets[base + p] == sockets[base + k]) { dup = 1; break; }
                if (!dup) break;
                int tmp = sockets[base + k]; sockets[base + k] = sockets[attempt]; sockets[attempt] = tmp;
            }
            int c = sockets[base + k];
            size_t e = base + k;
            edge_var[e] = (int)v;
            edge_chk[e] = c;
            var_edges[e] = (int)e;                    /* variable v owns edges [v*dv, v*dv+dv) */
        }
    }
    /* Group edges by check: bucket-fill chk_edges so check c owns the slice
     * [c*dc, c*dc+dc). We recount occupancy with a small per-check fill counter
     * reusing the decision[] region is unsafe (size N != M), so use a temp. */
    {
        int *fill = (int *)calloc(M, sizeof(int));
        if (!fill) {
            free(edge_var); free(edge_chk); free(var_edges); free(chk_edges);
            free(llr); free(decision); free(sockets); munmap(msg, msg_bytes);
            P2_LOG_ERR("calloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
        }
        for (size_t e = 0; e < E; e++) {
            int c = edge_chk[e];
            chk_edges[(size_t)c * DC + (size_t)fill[c]] = (int)e;
            fill[c]++;
        }
        free(fill);
    }
    free(sockets);                                    /* graph is built; sockets no longer needed */

    /* Initialise the message arrays once so the very first check-node update
     * reads defined variable->check messages. The all-zero received word gives
     * every bit the same positive channel LLR to start from. */
    for (size_t e = 0; e < E; e++) { m_vc[e] = CHANNEL_LLR; m_cv[e] = 0.0; }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t last_syndrome = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* (1)+(2) Transmit the all-zero codeword through a binary symmetric
         * channel and form channel LLRs. received bit = (0 XOR flip) so a flip
         * makes the bit 1; +Lc favours 0, -Lc favours 1. Re-seed the received
         * word every pass so each decode faces a fresh error pattern. */
        for (size_t v = 0; v < N; v++) {
            int rx = (rng_unit(&rng) < flip) ? 1 : 0;     /* channel output for the 0 that was sent */
            llr[v] = rx ? -CHANNEL_LLR : CHANNEL_LLR;
        }
        /* Prime the variable->check messages with the channel LLR (iteration 0's
         * variable belief before any check has spoken). */
        for (size_t v = 0; v < N; v++) {
            double lv = llr[v];
            const int *ve = var_edges + v * DV;
            for (size_t k = 0; k < DV; k++) m_vc[ve[k]] = lv;
        }

        /* (3) T min-sum iterations. Each iteration rewrites BOTH message arrays
         * in full -- first every check->variable message, then every
         * variable->check message -- which is the workload's signature write. */
        for (size_t it = 0; it < T; it++) {
            /* (3a) Check-node update. For each check, the outgoing message on
             * edge i excludes edge i's own incoming var->check message: its sign
             * is the product of the OTHER signs, its magnitude the MIN of the
             * other magnitudes. We get all dc outputs in O(dc) by tracking the
             * two smallest magnitudes and the total sign parity, then, per edge,
             * removing that edge's contribution. */
            for (size_t c = 0; c < M; c++) {
                const int *ce = chk_edges + c * DC;
                double min1 = INFINITY, min2 = INFINITY;  /* smallest, second-smallest |m_vc| */
                int argmin = 0;                            /* which edge holds min1 */
                int sign_prod = 1;                         /* product of all incoming signs (+/-1) */
                for (size_t j = 0; j < DC; j++) {
                    double val = m_vc[ce[j]];
                    if (val < 0.0) sign_prod = -sign_prod;
                    double a = fabs(val);
                    if (a < min1) { min2 = min1; min1 = a; argmin = (int)j; }
                    else if (a < min2) { min2 = a; }
                }
                for (size_t j = 0; j < DC; j++) {
                    double val = m_vc[ce[j]];
                    /* Exclude edge j: its sign leaves the product, and its
                     * magnitude is the min only if j is NOT the argmin (then the
                     * remaining min is min2). */
                    int s = sign_prod;
                    if (val < 0.0) s = -s;                 /* remove edge j's sign */
                    double mag = ((int)j == argmin) ? min2 : min1;
                    m_cv[ce[j]] = (double)s * mag;         /* signed min-sum output */
                }
            }
            /* (3b) Variable-node update. For each variable, the outgoing message
             * on edge k is the channel LLR plus the sum of the OTHER incoming
             * check->variable messages. Sum all dv incoming, then subtract edge
             * k's own contribution to get the extrinsic value in O(dv). */
            for (size_t v = 0; v < N; v++) {
                const int *ve = var_edges + v * DV;
                double total = llr[v];
                for (size_t k = 0; k < DV; k++) total += m_cv[ve[k]];
                for (size_t k = 0; k < DV; k++) m_vc[ve[k]] = total - m_cv[ve[k]];
            }
        }

        /* (4) Hard decision from the full belief (channel LLR + ALL incoming
         * check messages): >= 0 decodes bit 0, else bit 1. */
        for (size_t v = 0; v < N; v++) {
            const int *ve = var_edges + v * DV;
            double belief = llr[v];
            for (size_t k = 0; k < DV; k++) belief += m_cv[ve[k]];
            decision[v] = (belief >= 0.0) ? 0u : 1u;
        }
        /* Syndrome weight: count checks whose covered decided bits XOR to 1
         * (unsatisfied parity). Zero means a valid codeword was recovered. */
        uint64_t syndrome = 0;
        for (size_t c = 0; c < M; c++) {
            const int *ce = chk_edges + c * DC;
            int parity = 0;
            for (size_t j = 0; j < DC; j++) parity ^= decision[edge_var[ce[j]]];
            if (parity) syndrome++;
        }
        last_syndrome = syndrome;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = m_cv[E / 2];               /* a live check->variable LLR */

    free(edge_var); free(edge_chk); free(var_edges); free(chk_edges);
    free(llr); free(decision);
    munmap(msg, msg_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "final_syndrome_weight", last_syndrome);
    p2_meta_kv_f64(&m, "last_msg", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "iterated whole-buffer rewrite of both edge-message arrays vs the HMM's monotone front; "
                   "min-sum (not exact sum-product); random-regular Tanner graph may contain short cycles");
    p2_meta_close(&m);
    return 0;
}
