/* kernel_moe_dispatch_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Sparse-ROUTING dwarf, adjacent to the Sparse Linear Algebra motif (Berkeley
 * D2): the DISPATCH / COMBINE step of a Mixture-of-Experts (MoE) layer, the
 * routing machinery inside modern large language models. An MoE layer does not
 * send every token through every expert; a gating step assigns each token to
 * ONE expert (top-1), and the layer must physically REGROUP the tokens so that
 * every expert sees a contiguous batch of just its own tokens. That regroup is
 * a data-dependent SCATTER -- a token permutation into per-expert buffers --
 * which is exactly the sparse-routing pattern this kernel isolates.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs quiet SpMV)
 * ----------------------------------------------------------------------------
 * The quiet control in this dwarf (kernel_spmv_v2) is READ-dominated: it streams
 * a sparse structure and GATHERS a source vector, writing back only a tiny
 * result. Reads are invisible to a host memory WRITE-signal, so it reads as
 * near-idle. MoE routing is the opposite face of "sparse": the routing table
 * (which expert each token goes to) is small, but ACTING on it means SCATTERING
 * every token's whole D-dim feature vector into its expert's region of a full
 * N x D dispatch buffer, then GATHERING the processed vectors back to their
 * source rows in a full N x D output. Two complete N x D buffers are written
 * every pass in a permuted, data-dependent order. That permutation scatter is
 * the VISIBLE signature: the same "sparse" family, but WRITE-VISIBLE routing
 * rather than a quiet gather.
 *
 * PICTURE (top view):  a counting-sort permutation regroups tokens by expert.
 *
 *      X (N x D, dense)          expert[t] in [0,E)        BUF (N x D, dense)
 *      row t = token t's         (top-1 routing table)     tokens regrouped so
 *      feature vector            g[t] in (0,1] = gate      each expert is contiguous
 *
 *        X[t, :] = [ f .. f ] --- expert[t]=e --> BUF[ off[e]++ , :] = [ f .. f ]
 *                                 (scatter t into e's slice)          ^ src[slot]=t
 *
 *      DISPATCH: count tokens per expert, prefix-sum to per-expert offsets, then
 *                scatter each token's D-vector into BUF at off[expert[t]]++.
 *      EXPERT  : per-expert transform over each contiguous region of BUF.
 *      COMBINE : gather each processed BUF slot back to Y[src[slot], :].
 *
 *      BUF is the dominant buffer and the large visible scatter write; the
 *      routing table (expert[], g[]) is tiny and only READ during the scatter.
 *
 * ============================================================================
 * ALGORITHM (per measured pass)  --  MoE top-1 dispatch/combine
 * ============================================================================
 *   1. Re-seed X (N x D random features) and the routing: for each token draw a
 *      random expert in [0,E) and a random gate weight g in (0,1]. (A real gate
 *      takes an argmax over a learned score; a uniform random expert is a faithful
 *      stand-in for the memory pattern and keeps the benchmark deterministic.)
 *   2. DISPATCH (counting sort by expert):
 *        a. count[e] = number of tokens routed to expert e.
 *        b. prefix-sum counts -> start[e], the first BUF slot of expert e; copy
 *           start[] into a running off[] cursor.
 *        c. scatter: for each token t in order, slot = off[expert[t]]++; copy
 *           X[t, :] into BUF[slot, :] and record src[slot] = t. After this, BUF
 *           holds every expert's tokens in one contiguous run -- the signature
 *           permutation scatter.
 *   3. EXPERT: for each expert e, walk its contiguous slice [start[e], start[e]
 *      + count[e]) of BUF and apply a per-expert transform. To keep the pass
 *      INVERTIBLE for the correctness test the transform is a scale: multiply
 *      each slot's D-vector by its token's gate weight g and by a per-expert
 *      scalar escale[e] (escale defaults to all-1). (A production MoE would run
 *      a small feed-forward network here instead of a scalar.)
 *   4. COMBINE (gather): for each slot, divide out the gate weight and scatter
 *      the result back to Y[src[slot], :]. With escale = 1 the expert multiplied
 *      by g and combine divides by g, so identity expert + gate reproduces X
 *      exactly (X * g / g == X); the roundtrip test relies on this.
 *
 * Note: this is sparse token ROUTING -- a permutation into per-expert buffers,
 * plus the gather back -- adjacent to sparse linear algebra rather than a matmul.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Sparse Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + first seed) / measure (dispatch/expert/combine passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_moe_dispatch_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign MoE dispatch/combine; WRITE-VISIBLE sparse routing, Sparse-LA)\n"
"  --tokens N            Number of tokens / rows of X, BUF, Y (default 65536)\n"
"  --dim D               Feature dimension per token (default 128)\n"
"  --experts E           Number of experts to route across (default 8)\n"
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

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long tokens     = p2_get_i64(argc, argv, "--tokens", 65536);
    long long dim        = p2_get_i64(argc, argv, "--dim", 128);
    long long experts    = p2_get_i64(argc, argv, "--experts", 8);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (tokens  < 1  || tokens  > 200000000LL) { P2_LOG_ERR("tokens %lld out of range (1..2e8)", tokens); return 2; }
    if (dim     < 1  || dim     > 65536LL)     { P2_LOG_ERR("dim %lld out of range (1..65536)", dim); return 2; }
    if (experts < 1  || experts > tokens)      { P2_LOG_ERR("experts %lld out of range (1..tokens=%lld)", experts, tokens); return 2; }
    size_t N = (size_t)tokens, D = (size_t)dim, E = (size_t)experts;
    /* Buffers: dense X (N x D) and dense Y (N x D) are READ / written back per
     * row; BUF (N x D) is the dominant, data-dependent scatter target (the
     * visible write). Routing state is tiny: expert[N], gate g[N], src[N] slot
     * owners, plus per-expert count[E]/start[E]/off[E]. */
    size_t bytes_BUF   = N * D * sizeof(double);          /* the large visible scatter write */
    size_t bytes_X     = N * D * sizeof(double);
    size_t bytes_Y     = N * D * sizeof(double);
    size_t bytes_expert = N * sizeof(uint32_t);
    size_t bytes_gate  = N * sizeof(double);
    size_t bytes_src   = N * sizeof(uint32_t);
    size_t bytes_perE  = 3 * E * sizeof(size_t);          /* count + start + off */
    size_t bytes = bytes_BUF + bytes_X + bytes_Y + bytes_expert + bytes_gate + bytes_src + bytes_perE;
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("total bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Sparse Linear Algebra");
    p2_meta_kv_str(&m, "scheme", "MoE top-1 dispatch/combine: counting-sort token permutation scattered into per-expert buffers (visible), gathered back (routing adjacent to sparse-LA)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "tokens", tokens);
    p2_meta_kv_i64(&m, "dim", dim);
    p2_meta_kv_i64(&m, "experts", experts);
    p2_meta_kv_u64(&m, "dispatch_buf_bytes", bytes_BUF);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The dispatch buffer BUF is the dominant buffer -> mmap + madvise + mlock
     * it. Every pass scatters all N token vectors into it in a permuted,
     * data-dependent order, which is the workload's signature (large, visible)
     * write, driven by the top-1 routing table. */
    double *BUF = (double *)mmap(NULL, bytes_BUF, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (BUF == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes_BUF, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(BUF, bytes_BUF, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(BUF, bytes_BUF);

    /* X / Y (dense feature and output matrices) and the routing state live in
     * plain malloc'd arrays. X is READ during the scatter; Y is written back by
     * the gather. escale[] is the per-expert transform scalar (all 1 = identity
     * expert), kept constant across passes. */
    double   *X      = (double   *)malloc(N * D * sizeof(double));
    double   *Y      = (double   *)malloc(N * D * sizeof(double));
    uint32_t *expert = (uint32_t *)malloc(N * sizeof(uint32_t));
    double   *g      = (double   *)malloc(N * sizeof(double));
    uint32_t *src    = (uint32_t *)malloc(N * sizeof(uint32_t));
    size_t   *count  = (size_t   *)malloc(E * sizeof(size_t));
    size_t   *start  = (size_t   *)malloc(E * sizeof(size_t));
    size_t   *off    = (size_t   *)malloc(E * sizeof(size_t));
    double   *escale = (double   *)malloc(E * sizeof(double));
    if (!X || !Y || !expert || !g || !src || !count || !start || !off || !escale) {
        P2_LOG_ERR("malloc failed");
        free(X); free(Y); free(expert); free(g); free(src);
        free(count); free(start); free(off); free(escale);
        munmap(BUF, bytes_BUF);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    if (!no_mlock) { p2_mlock_soft(X, N * D * sizeof(double)); p2_mlock_soft(Y, N * D * sizeof(double)); }

    /* Per-expert transform scalars: default all 1 -> the expert is the identity
     * (so the correctness test's dispatch->expert->combine roundtrip reproduces
     * X exactly). A real MoE would replace this with a small FFN per expert. */
    for (size_t e = 0; e < E; e++) escale[e] = 1.0;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Seed X (features) and the routing table (random expert + gate per token)
     * once during warmup; the measured passes re-seed both. */
    for (size_t i = 0; i < N * D; i++) X[i] = rng_unit(&rng);
    for (size_t t = 0; t < N; t++) {
        expert[t] = (uint32_t)(p2_rng_next(&rng) % (uint64_t)E);
        g[t] = rng_unit(&rng) + (1.0 / 9007199254740992.0);   /* gate in (0,1] (strictly > 0) */
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed X and the routing each pass so the permutation and the output
         * are not trivially constant. X and the routing table are READS during
         * the scatter; only BUF (and the Y gather) are the sizeable writes. */
        for (size_t i = 0; i < N * D; i++) X[i] = rng_unit(&rng);
        for (size_t t = 0; t < N; t++) {
            expert[t] = (uint32_t)(p2_rng_next(&rng) % (uint64_t)E);
            g[t] = rng_unit(&rng) + (1.0 / 9007199254740992.0);   /* gate in (0,1] */
        }

        /* --- DISPATCH: counting sort by expert ------------------------------ */
        /* (a) histogram tokens per expert. */
        for (size_t e = 0; e < E; e++) count[e] = 0;
        for (size_t t = 0; t < N; t++) count[expert[t]]++;
        /* (b) prefix-sum counts into per-expert start offsets; off[] is the
         *     running write cursor, seeded from start[]. */
        size_t acc = 0;
        for (size_t e = 0; e < E; e++) { start[e] = acc; off[e] = acc; acc += count[e]; }
        /* (c) scatter each token's whole D-vector into its expert's slot. This
         *     permuted, data-dependent copy into BUF is the signature write. */
        for (size_t t = 0; t < N; t++) {
            size_t e    = expert[t];
            size_t slot = off[e]++;                          /* next free slot of expert e */
            src[slot]   = (uint32_t)t;                       /* remember source token */
            const double *xr = &X[t * D];                    /* token t's feature row */
            double       *br = &BUF[slot * D];               /* its assigned slot in BUF */
            for (size_t j = 0; j < D; j++) br[j] = xr[j];    /* scatter the D-vector */
        }

        /* --- EXPERT: per-expert transform over each contiguous region ------- */
        /* Walk expert e's slice [start[e], start[e]+count[e]) and scale each
         * slot by its token's gate weight and the per-expert scalar. Invertible
         * so the combine below can undo it exactly. */
        for (size_t e = 0; e < E; e++) {
            double se = escale[e];
            size_t s0 = start[e], s1 = start[e] + count[e];
            for (size_t slot = s0; slot < s1; slot++) {
                double w = g[src[slot]] * se;                /* gate weight x expert scalar */
                double *br = &BUF[slot * D];
                for (size_t j = 0; j < D; j++) br[j] *= w;
            }
        }

        /* --- COMBINE: gather each processed slot back to its source row ----- */
        /* Divide out the gate weight and scatter BUF[slot,:] back to Y[src,:].
         * With escale = 1 this reproduces X exactly (X*g/g == X). */
        for (size_t slot = 0; slot < N; slot++) {
            size_t t = src[slot];
            double inv = 1.0 / g[t];                         /* undo the gate weight */
            const double *br = &BUF[slot * D];
            double       *yr = &Y[t * D];                    /* source token's output row */
            for (size_t j = 0; j < D; j++) yr[j] = br[j] * inv;
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = Y[(N / 2) * D + (D / 2)];         /* one live output element */

    free(X); free(Y); free(expert); free(g); free(src);
    free(count); free(start); free(off); free(escale);
    munmap(BUF, bytes_BUF);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "sink", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "WRITE-VISIBLE: the token-permutation scatter into per-expert BUF (plus the gather back) is the tell vs quiet SpMV; X/routing reads invisible. Random top-1 routing stands in for a learned gate; identity scalar expert stands in for an FFN.");
    p2_meta_close(&m);
    return 0;
}
