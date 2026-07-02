/* kernel_conv_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  DIRECT 2D CONVOLUTION:  a CNN layer, sliding K x K filters over a feature map
 * ============================================================================
 *
 *  DWARF   : Dense Linear Algebra (D1)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, memory-signature label)
 *  PURPOSE : Probe the write behaviour of the classic vision workhorse. Unlike a
 *            plain matrix multiply, its reads march over the input in heavily
 *            overlapping windows while its writes lay down a large, structured
 *            output feature-map stack -- a stencil-like pattern with learned
 *            weights and many output channels.
 *
 *  PICTURE (top view):
 *
 *      one input channel (H x W)                 output feature maps (Cout x Ho x Wo)
 *      +--------------------------+                    +---+ +---+       +---+
 *      | . . . . . . . . . . . .  |   Cout filters     |   | |   |  ...  |   |
 *      | . +---+ . . . . . . . .  |   each Cin x K x K  +---+ +---+       +---+
 *      | . | K | ---> slide -->    |  =============>       one output pixel per
 *      | . | K |window . . . . .   |  multiply-accumulate  window position; every
 *      | . +---+ . . . . . . . .  |  over each K x K       output map fully rewritten
 *      | . . . . . . . . . . . .  |  window               each pass
 *      +--------------------------+
 *          a single K x K window produces exactly one output value
 *
 *  ALGORITHM (per measured pass):
 *      1. Re-seed the input feature map and all filter weights with fresh random
 *         values (keeps the compute honest; blocks the optimiser from hoisting).
 *      2. For each output channel oc (one filter), slide that filter over every
 *         (oy, ox) output position. At each position, multiply-accumulate the
 *         filter against the overlapping K x K window, summed over all input
 *         channels, and write the single result to OUT[oc][oy][ox].
 *      Output dimensions are Ho = H-K+1, Wo = W-K+1 ("valid" convolution: the
 *      window must fit entirely inside the input, so there is no padding).
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Reads sweep the input in overlapping windows (each input pixel is re-read
 *      by up to K x K neighbouring output positions), but those reads are
 *      invisible to a write-only observer. What IS visible is the full,
 *      structured rewrite of the (large) Cout x Ho x Wo output stack every pass,
 *      one output map at a time. HONEST CAVEAT: this is a direct (non-lowered)
 *      convolution -- real libraries often lower it to a GEMM via im2col, which
 *      would shift the footprint toward plain matrix multiply.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Dense Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (conv passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_conv_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign 2D convolution / CNN layer; Dense-LA kernel)\n"
"  --height H            Input height (default 256)\n"
"  --width W             Input width (default 256)\n"
"  --in-channels C       Input channels (default 3)\n"
"  --filters F           Output channels / filters (default 32)\n"
"  --ksize K             Square kernel size (default 3)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on output bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

static inline float rng_unitf(p2_rng_t *r) {
    return (float)((double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0));
}

/* Overwrite an n-element buffer with fresh uniform random values in [-1,1).
 * Called once per pass on the input map and the filter weights so every pass
 * does genuine work and the optimiser cannot treat the output as invariant. */
static void reseed(float *M, size_t n, p2_rng_t *rng) {
    for (size_t k = 0; k < n; k++) M[k] = 2.0f * rng_unitf(rng) - 1.0f;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long H  = p2_get_i64(argc, argv, "--height", 256);
    long long W  = p2_get_i64(argc, argv, "--width", 256);
    long long Cin = p2_get_i64(argc, argv, "--in-channels", 3);
    long long F  = p2_get_i64(argc, argv, "--filters", 32);
    long long K  = p2_get_i64(argc, argv, "--ksize", 3);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (H < 8 || H > 8192 || W < 8 || W > 8192) { P2_LOG_ERR("H/W out of range (8..8192)"); return 2; }
    if (Cin < 1 || Cin > 512) { P2_LOG_ERR("in-channels %lld out of range (1..512)", Cin); return 2; }
    if (F < 1 || F > 1024) { P2_LOG_ERR("filters %lld out of range (1..1024)", F); return 2; }
    if (K < 1 || K > 15 || K >= H || K >= W) { P2_LOG_ERR("ksize %lld out of range (1..15, < H,W)", K); return 2; }
    size_t h = (size_t)H, w = (size_t)W, cin = (size_t)Cin, f = (size_t)F, k = (size_t)K;
    size_t ho = h - k + 1, wo = w - k + 1;
    size_t obytes = f * ho * wo * sizeof(float);      /* output feature maps dominate */
    if (obytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("output bytes %zu exceed --max-mb %lld", obytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Dense Linear Algebra");
    p2_meta_kv_str(&m, "scheme", "2D convolution / CNN layer (sliding-window multiply-accumulate)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "height", H);
    p2_meta_kv_i64(&m, "width", W);
    p2_meta_kv_i64(&m, "in_channels", Cin);
    p2_meta_kv_i64(&m, "filters", F);
    p2_meta_kv_i64(&m, "ksize", K);
    p2_meta_kv_u64(&m, "output_bytes", obytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    float *OUT = (float *)mmap(NULL, obytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (OUT == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", obytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(OUT, obytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(OUT, obytes);

    float *IN = (float *)malloc(cin * h * w * sizeof(float));
    float *WT = (float *)malloc(f * cin * k * k * sizeof(float));
    if (!IN || !WT) { free(IN); free(WT); munmap(OUT, obytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    reseed(IN, cin * h * w, &rng); reseed(WT, f * cin * k * k, &rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        reseed(IN, cin * h * w, &rng); reseed(WT, f * cin * k * k, &rng);
        /* Output channels (filters): each iteration produces one full output map.
         * The maps are stored contiguously, so this is the outer stride of the
         * large output rewrite that dominates the write signature. */
        for (size_t oc = 0; oc < f; oc++) {
            float *omap = OUT + oc * ho * wo;            /* this filter's output map   */
            const float *wbase = WT + oc * cin * k * k;  /* this filter's weight block */
            /* Slide the filter over every valid output position (oy, ox). */
            for (size_t oy = 0; oy < ho; oy++) {
                for (size_t ox = 0; ox < wo; ox++) {
                    /* Accumulate one output pixel: sum over all input channels
                     * of the elementwise product of the filter with the K x K
                     * input window whose top-left corner is (oy, ox). */
                    float acc = 0.0f;
                    for (size_t ic = 0; ic < cin; ic++) {
                        const float *imap = IN + ic * h * w;     /* input channel ic   */
                        const float *wc = wbase + ic * k * k;    /* weights for that channel */
                        for (size_t ky = 0; ky < k; ky++) {
                            /* irow points at the start of window row ky inside the
                             * input; the overlap with neighbouring windows is why
                             * each input pixel is re-read many times. */
                            const float *irow = imap + (oy + ky) * w + ox;
                            const float *wrow = wc + ky * k;
                            for (size_t kx = 0; kx < k; kx++) acc += irow[kx] * wrow[kx];
                        }
                    }
                    omap[oy * wo + ox] = acc;            /* single write per window */
                }
            }
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile float sink = OUT[(f - 1) * ho * wo + (ho / 2) * wo + (wo / 2)];

    free(IN); free(WT);
    munmap(OUT, obytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "center_out", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "direct (non-lowered) convolution; visibility from the large output-feature-map rewrite");
    p2_meta_close(&m);
    return 0;
}
