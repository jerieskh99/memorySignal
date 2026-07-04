/* kernel_diffusion_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * MapReduce / Monte Carlo dwarf (Berkeley motif D7), the diffusion-sampler
 * variant -- a simplified stand-in for the iterative-denoising SAMPLING loop
 * that sits behind modern generative image models (Stable Diffusion, DALL-E).
 * Real diffusion sampling starts from pure noise and, over T steps, repeatedly
 * REWRITES the whole latent image toward a cleaner sample. This kernel keeps
 * that Monte-Carlo-style iterative-sampling skeleton: seed an image with random
 * noise, then take T denoising steps, each of which overwrites EVERY pixel.
 *
 * HONEST NOTE -- THE "DENOISER" IS NOT A NEURAL NETWORK
 * ----------------------------------------------------------------------------
 * A real diffusion model predicts the noise with a trained U-Net. That is not
 * verifiable in a benchmark, so the denoiser HERE is a fixed linear smoother:
 * a 3x3 periodic-boundary average (a mean filter with wraparound edges) plus a
 * decaying noise schedule. This is a mathematically well-defined stand-in that
 * reproduces the iterative-whole-image-rewrite MEMORY pattern of diffusion
 * sampling while staying checkable. It is a smoother, not a learned model, and
 * the header does not pretend otherwise.
 *
 * WHY IT IS WRITE-VISIBLE (and distinct from the other D7 members)
 * ----------------------------------------------------------------------------
 * Every denoising step recomputes the ENTIRE W x H image and lands it in the
 * scratch buffer, then swaps -- an "iterative whole-image rewrite". Over T
 * steps the write front sweeps the full image footprint T times, ping-ponging
 * between two fixed equal-size buffers with clean period-2 regularity. That is
 * a different tell from the other D7 kernels: the histogram's random scatter,
 * the Monte-Carlo option's bulk store, and the path-tracer's accumulate. Here
 * the reduce IS the image, rewritten in full on every step.
 *
 * ============================================================================
 * PICTURE (top view): one noisy image, rewritten whole, smoother each step
 * ============================================================================
 *
 *   step 0 (noise)        step t                    step T-1 (smooth)
 *   # . @ # . @ # .        . # . o . o . #           o o o . o o . o
 *   . @ # . @ # . @        o . # . o . # .           o . o o o . o o
 *   @ # . @ # . @ #  ==>   . o . # . # . o   ==> ... o o o . o o o .
 *   # . @ # . @ # .        # . o . # . o .           . o o o . o o o
 *      (random)              (blurring)                 (converged)
 *
 *   every step: blurred = Avg3x3_periodic(image)   // 3x3 mean, wraparound edges
 *               image   = (1-a)*image + a*blurred + noise(t)*eps
 *   the WHOLE image is overwritten each step (image <-> scratch double-buffer).
 *
 * ============================================================================
 * ALGORITHM (per measure pass)
 * ============================================================================
 *   1. Re-seed the RNG and fill the image with random noise in [0,1) (so every
 *      pass starts from the identical field and the last-pass stats reproduce).
 *   2. For step t = 0 .. T-1:
 *        a. blurred[p] = mean of the 3x3 periodic neighbourhood of image[p]
 *           (wraparound boundaries -> a doubly-stochastic averaging operator),
 *        b. image[p]   = (1-a)*image[p] + a*blurred[p] + noise(t)*eps
 *           where a in (0,1] is a fixed mix weight and noise(t) decays linearly
 *           from noise_start to 0 as t grows; eps is zero-mean per pixel.
 *        The write lands in the scratch buffer; then swap image <-> scratch so
 *        the just-written image is the input to the next step.
 *   3. Count the pass.
 *
 *   --no-noise VALIDATION MODE: force noise(t) = 0 for all steps, so each step
 *   is a PURE 3x3 periodic average. A periodic 3x3 mean is doubly stochastic:
 *   it CONSERVES the total sum exactly and MONOTONICALLY reduces the variance
 *   (contraction toward the mean). Those two properties are the non-circular
 *   checks the verifier asserts.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL (visible). Dwarf: MapReduce / Monte Carlo.
 * See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (denoising passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_diffusion_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign diffusion sampler; MapReduce/MonteCarlo kernel)\n"
"  --width W             Image width in pixels (default 512)\n"
"  --height H            Image height in pixels (default 512; uses 2 * W*H*8 bytes)\n"
"  --steps T             Denoising steps per pass (default 200)\n"
"  --mix-milli A         Mix weight a x1000 (default 1000 = 1.0 -> pure blur step)\n"
"  --noise-start-milli S Initial noise level x1000 (default 200 = 0.2; decays to 0)\n"
"  --no-noise            Validation mode: noise = 0 always (pure periodic average)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the initial noise field (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes, both buffers (default 8192)\n"
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

/* zero-mean noise sample in [-0.5, 0.5) from the xoshiro stream (the "eps") */
static inline double p2_rng_eps(p2_rng_t *r) {
    return p2_rng_unit(r) - 0.5;
}

/* Sum and variance of a W*H image (population variance about its own mean).
 * Used only at the boundaries of the last pass to expose the conservation +
 * variance-contraction invariants in the metadata. */
static void image_stats(const double *img, size_t n, double *out_sum, double *out_var) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += img[i];
    double mean = sum / (double)n;
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = img[i] - mean;
        acc += d * d;
    }
    *out_sum = sum;
    *out_var = acc / (double)n;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long width      = p2_get_i64(argc, argv, "--width", 512);
    long long height     = p2_get_i64(argc, argv, "--height", 512);
    long long steps      = p2_get_i64(argc, argv, "--steps", 200);
    /* Floating-point options are passed as integer-milli, because the phase2 arg
     * helpers are integer-only: --mix-milli 1000 = 1.0, --noise-start-milli 200 = 0.2. */
    long long mix_milli  = p2_get_i64(argc, argv, "--mix-milli", 1000);
    long long nstart_milli = p2_get_i64(argc, argv, "--noise-start-milli", 200);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       no_noise   = p2_flag_present(argc, argv, "--no-noise");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (width < 4 || width > 65536) { P2_LOG_ERR("width %lld out of range (4..65536)", width); return 2; }
    if (height < 4 || height > 65536) { P2_LOG_ERR("height %lld out of range (4..65536)", height); return 2; }
    if (steps < 0) { P2_LOG_ERR("steps %lld must be >= 0", steps); return 2; }
    double mix = (double)mix_milli / 1000.0;
    double noise_start = (double)nstart_milli / 1000.0;
    if (mix < 0.0 || mix > 1.0) { P2_LOG_ERR("mix %.3f out of range (0..1)", mix); return 2; }
    if (noise_start < 0.0) { P2_LOG_ERR("noise_start %.3f must be >= 0", noise_start); return 2; }

    size_t W = (size_t)width, H = (size_t)height;
    size_t npix = W * H;                             /* pixels in one image */
    size_t buf_bytes = npix * sizeof(double);        /* one image buffer */
    size_t total_bytes = 2 * buf_bytes;              /* image + scratch double-buffer */
    if (total_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("total bytes %zu (both buffers) exceed --max-mb %lld", total_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D7 MapReduce/MonteCarlo");
    p2_meta_kv_str(&m, "scheme", "diffusion sampler: iterative whole-image rewrite (3x3 periodic-average denoiser stand-in + decaying noise)");
    p2_meta_kv_str(&m, "denoiser_note", "linear 3x3 periodic-average smoother, NOT a neural net; chosen for verifiability");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "width", width);
    p2_meta_kv_i64(&m, "height", height);
    p2_meta_kv_i64(&m, "steps", steps);
    p2_meta_kv_i64(&m, "mix_milli", mix_milli);
    p2_meta_kv_i64(&m, "noise_start_milli", nstart_milli);
    p2_meta_kv_i64(&m, "no_noise", no_noise);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* Two equal-size image buffers: the current image and the scratch it is
     * rewritten into each step. The image is the dominant footprint and the
     * reduction target -> mmap + mlock both. The whole-image rewrite lands as a
     * dense sequential write front sweeping one buffer, then the roles swap. */
    double *image = (double *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    double *scratch = (double *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                                     MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (image == MAP_FAILED || scratch == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu x2) failed: %s", buf_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(image, buf_bytes, MADV_NOHUGEPAGE);
    p2_madvise(scratch, buf_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) { p2_mlock_soft(image, buf_bytes); p2_mlock_soft(scratch, buf_bytes); }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Seed the image with random noise in [0,1) and fault in the scratch buffer.
     * Re-seeding at the top of every pass reproduces this exact field. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < npix; i++) image[i] = p2_rng_unit(&rng);
    memset(scratch, 0, buf_bytes);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    /* Last-pass boundary stats: the verifier reads these to check that a pure
     * periodic average conserves the sum and contracts the variance. */
    double init_sum = 0.0, final_sum = 0.0, init_var = 0.0, final_var = 0.0;
    size_t T = (size_t)steps;
    /* Mix weights are constant across steps; hoist them out of the loops. */
    const double a = mix, one_minus_a = 1.0 - mix;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One denoising pass: re-seed to the identical noise field, then take T
         * whole-image-rewrite steps. cur points at the live image, nxt at the
         * scratch it is rewritten into; they swap after every step. */
        double *cur = image, *nxt = scratch;
        p2_rng_t prng; p2_rng_seed(&prng, seed);
        for (size_t i = 0; i < npix; i++) cur[i] = p2_rng_unit(&prng);

        image_stats(cur, npix, &init_sum, &init_var);   /* field going into the T steps */

        for (size_t t = 0; t < T; t++) {
            /* noise(t): decay linearly from noise_start to 0 over the T steps.
             * In --no-noise mode this is forced to 0 so each step is a pure
             * periodic 3x3 average (mass-conserving, variance-contracting). */
            double nlev = 0.0;
            if (!no_noise && T > 0)
                nlev = noise_start * (double)(T - t) / (double)T;

            /* Rewrite EVERY pixel of nxt from the 3x3 periodic neighbourhood of
             * cur. Row indices wrap around (periodic boundary), which makes the
             * 3x3 mean a doubly-stochastic operator: it conserves the total sum
             * exactly and cannot increase the variance. */
            for (size_t y = 0; y < H; y++) {
                size_t yup = (y == 0) ? H - 1 : y - 1;   /* wraparound row above */
                size_t ydn = (y == H - 1) ? 0 : y + 1;   /* wraparound row below */
                const double *r0 = cur + yup * W;
                const double *r1 = cur + y   * W;
                const double *r2 = cur + ydn * W;
                double *out = nxt + y * W;
                for (size_t x = 0; x < W; x++) {
                    size_t xl = (x == 0) ? W - 1 : x - 1; /* wraparound col left  */
                    size_t xr = (x == W - 1) ? 0 : x + 1; /* wraparound col right */
                    double blurred = (r0[xl] + r0[x] + r0[xr] +
                                      r1[xl] + r1[x] + r1[xr] +
                                      r2[xl] + r2[x] + r2[xr]) * (1.0 / 9.0);
                    /* image <- (1-a)*image + a*blurred + noise(t)*eps */
                    double v = one_minus_a * r1[x] + a * blurred;
                    if (nlev != 0.0) v += nlev * p2_rng_eps(&prng);
                    out[x] = v;                           /* whole-image rewrite */
                }
            }
            /* Swap: the image just written becomes the input to the next step.
             * A pointer swap (not a copy) makes the write target alternate
             * between the two fixed buffers with period-2 regularity. */
            double *tmp = cur; cur = nxt; nxt = tmp;
        }

        image_stats(cur, npix, &final_sum, &final_var);  /* field after the T steps */
        /* Leave the live image where cur points, so a later pass keeps working
         * on valid buffers regardless of the parity of T. */
        image = cur; scratch = nxt;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = image[(H / 2) * W + (W / 2)];  /* a live pixel: prevent dead-code elim */

    munmap(image, buf_bytes);
    munmap(scratch, buf_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    /* Last-pass invariants (for the non-circular verifier). With --no-noise a
     * pure periodic 3x3 average must satisfy final_sum == init_sum (mass
     * conservation) and final_var < init_var (variance contraction). */
    p2_meta_kv_f64(&m, "init_sum", init_sum);
    p2_meta_kv_f64(&m, "final_sum", final_sum);
    p2_meta_kv_f64(&m, "init_var", init_var);
    p2_meta_kv_f64(&m, "final_var", final_var);
    p2_meta_kv_f64(&m, "center_pixel", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "denoiser is a linear 3x3 periodic-average smoother, not a trained network; "
                   "expected signature: iterative whole-image rewrite, dense sequential write front "
                   "ping-ponging between two buffers with period-2 regularity; "
                   "--no-noise makes each step a mass-conserving variance-contracting average");
    p2_meta_close(&m);
    return 0;
}
