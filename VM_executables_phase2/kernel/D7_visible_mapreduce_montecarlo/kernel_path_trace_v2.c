/* kernel_path_trace_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  MONTE-CARLO PATH TRACE:  per-pixel radiance integration by ray accumulation
 * ============================================================================
 *
 *  DWARF   : MapReduce / Monte Carlo (D7)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL (write-visible)          (first-division, memory-signature label)
 *  PURPOSE : Probe the "image-buffer accumulation" write pattern of a Monte-Carlo
 *            path tracer: every pixel of one W x H image is hit by K random sample
 *            rays whose radiance is ADDED in, then the pixel is normalised. The
 *            host write-signal sees the whole image buffer repeatedly
 *            read-modify-written (accumulate) and finally swept once (the /K mean).
 *
 *  PICTURE (top view):  K random sample rays fan into a single pixel; their
 *  radiance L(sample) is accumulated, then the pixel is averaged (the MC mean).
 *
 *      image[] : one W x H grid of f64 pixels (the dominant mmap+mlock buffer)
 *
 *          +---+---+---+---+---+          per pixel p:
 *          |   |   |   |   |   |
 *          +---+---+---+---+---+            \  |  /   K random sample rays
 *          |   |   | p |   |   |    <----    \ | /    each returns L(sample)
 *          +---+---+---+---+---+              \|/     image[p] += L  (accumulate)
 *          |   |   |   |   |   |               *  <-- pixel p
 *          +---+---+---+---+---+          then image[p] /= K   (Monte-Carlo mean)
 *
 *      L(sample) = c + (u - 0.5) * noise_amp,   u ~ Uniform[0,1)   (zero-mean noise)
 *      E[L] = c  exactly, so every pixel converges to the constant radiance c.
 *
 *  ALGORITHM:
 *      1. Allocate one W x H f64 image (the only sized buffer; mmap + mlock it).
 *      2. Each measure pass RE-SEEDS the RNG and rewrites the whole image:
 *         for every pixel p, zero it, then shoot K random sample rays. Each ray
 *         draws a random hemisphere direction / sub-pixel sample coordinate from
 *         the harness RNG (pure register compute) and returns a radiance value
 *         L(sample) from a constant-radiance environment perturbed by zero-mean
 *         noise. Accumulate image[p] += L(sample) across the K samples.
 *      3. Normalise: image[p] /= K  -> the Monte-Carlo mean radiance of pixel p.
 *      4. Count passes for the timed duration; report the final image's mean
 *         pixel value (should approx c) and its max abs deviation from c.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A dense image buffer under sustained read-modify-write. Within a pixel the
 *      K accumulations hammer one f64 slot; across a pass the write front marches
 *      the full W x H grid, and each pass repeats it -- so the observer sees a
 *      large, periodically rewritten working set. This "image accumulate" tell is
 *      distinct from (a) the HISTOGRAM SCATTER pattern, whose writes land in
 *      data-dependent, scattered bins rather than sweeping a dense grid in order,
 *      and (b) the QUIET SCALAR CONTROL (sibling kernel_mc_pi_v2, IDLE family),
 *      whose Monte-Carlo estimator only accumulates a scalar into a few KB of
 *      partials and is therefore near-invisible to a write-only observer.
 *
 *  HONEST NOTE: this is NOT a full scene renderer. There is no geometry, no BVH,
 *  no materials, no light transport across surfaces. It is a per-pixel Monte-Carlo
 *  integrator that reproduces the WRITE PATTERN of a path tracer (accumulate K
 *  random samples into every pixel of an image buffer, then average). To keep it
 *  verifiable it integrates a KNOWN integrand: a constant-radiance environment
 *  perturbed by zero-mean noise, whose analytic per-pixel mean is exactly c
 *  regardless of the noise amplitude. So the truth is checkable in closed form:
 *  the image mean -> c and every pixel sits within Monte-Carlo error ~ noise/sqrt(K).
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL (visible). Dwarf: MapReduce / Monte Carlo.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init + priming pass) / measure (MC image passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_path_trace_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Monte-Carlo path trace; MapReduce/Monte-Carlo visible kernel)\n"
"  --width W             Image width in pixels (default 512)\n"
"  --height H            Image height in pixels (default 512; uses W*H*8 bytes)\n"
"  --samples K           Sample rays per pixel (default 64)\n"
"  --radiance-milli C    Constant radiance x1000 (default 500 = 0.5)\n"
"  --noise-milli A       Zero-mean noise amplitude x1000 (default 500 = 0.5)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on image-buffer bytes (default 8192)\n"
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

/* Radiance returned by one random sample ray for pixel p.
 * The random draws (a sub-pixel jitter and a cosine-weighted hemisphere
 * direction) are genuine per-sample register compute so the estimator is really
 * stochastic, but the integrand is a CONSTANT-radiance environment perturbed by
 * ZERO-MEAN noise:  L = c + (u - 0.5) * noise_amp,  u ~ Uniform[0,1).
 * Hence E[L] = c exactly and every pixel converges to c regardless of noise. */
static inline double sample_radiance(p2_rng_t *r, double c, double noise_amp) {
    /* Draw a jittered sub-pixel offset and a hemisphere direction. These feed no
     * geometry (there is none) -- they exist so each sample does the same random
     * work a real per-ray path sample would, keeping the RNG stream advancing. */
    double jx = rng_unit(r);            /* sub-pixel x jitter in [0,1)          */
    double jy = rng_unit(r);            /* sub-pixel y jitter in [0,1)          */
    double u1 = rng_unit(r);            /* hemisphere sample coord 1            */
    double u2 = rng_unit(r);            /* hemisphere sample coord 2            */
    /* Cosine-weighted hemisphere direction (Malley's method); computed and then
     * discarded -- it models the per-sample cost without steering the integrand. */
    double rr = sqrt(u1);
    double phi = 6.283185307179586 * u2;
    double dx = rr * cos(phi), dy = rr * sin(phi);
    /* Consume the jitter/direction so the compiler cannot elide the draws. */
    double u = rng_unit(r);             /* the noise draw (drives the estimator) */
    double keep = (jx + jy + dx + dy) * 0.0;   /* provably zero: no bias added   */
    return c + (u - 0.5) * noise_amp + keep;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long width          = p2_get_i64(argc, argv, "--width", 512);
    long long height         = p2_get_i64(argc, argv, "--height", 512);
    long long samples        = p2_get_i64(argc, argv, "--samples", 64);
    /* Floating-point options are passed as integer-milli, because the phase2 arg
     * helpers are integer-only: --radiance-milli 500 = 0.5, --noise-milli 500 = 0.5. */
    long long radiance_milli = p2_get_i64(argc, argv, "--radiance-milli", 500);
    long long noise_milli    = p2_get_i64(argc, argv, "--noise-milli", 500);
    long long duration_s     = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s       = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb         = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu            = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed           = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock       = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run        = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir       = p2_get_str(argc, argv, "--output-dir", NULL);
    double radiance = (double)radiance_milli / 1000.0;
    double noise    = (double)noise_milli / 1000.0;

    if (width < 8 || width > 65536) { P2_LOG_ERR("width %lld out of range (8..65536)", width); return 2; }
    if (height < 8 || height > 65536) { P2_LOG_ERR("height %lld out of range (8..65536)", height); return 2; }
    if (samples < 1 || samples > (1LL << 24)) { P2_LOG_ERR("samples %lld out of range (1..2^24)", samples); return 2; }
    if (noise < 0.0) { P2_LOG_ERR("noise %.3f must be non-negative", noise); return 2; }

    size_t W = (size_t)width, H = (size_t)height;
    size_t K = (size_t)samples;
    size_t npix = W * H;
    size_t buf_bytes = npix * sizeof(double);      /* the image buffer dominates the footprint */
    if (buf_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("image bytes %zu exceed --max-mb %lld", buf_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D7 MapReduce/MonteCarlo");
    p2_meta_kv_str(&m, "family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "scheme", "Monte-Carlo path trace: accumulate K random sample rays into every pixel, then average (image-buffer accumulation)");
    p2_meta_kv_str(&m, "note", "not a full scene renderer; integrates a known constant-radiance environment so per-pixel truth is verifiable");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "width", width);
    p2_meta_kv_i64(&m, "height", height);
    p2_meta_kv_i64(&m, "samples", samples);
    p2_meta_kv_f64(&m, "radiance", radiance);
    p2_meta_kv_f64(&m, "noise", noise);
    p2_meta_kv_u64(&m, "image_bytes", buf_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The image is the dominant buffer -> mmap + mlock it. Each pass rewrites the
     * whole grid (accumulate K samples per pixel, then normalise), which is the
     * workload's signature write: dense image-buffer accumulation. */
    double *image = (double *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (image == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", buf_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(image, buf_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(image, buf_bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Warm-up: fault the image pages in and run one throwaway accumulate pass so
     * pages are resident and caches primed before timing begins. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t p = 0; p < npix; p++) image[p] = 0.0;
    {
        size_t warm = (npix < 4096) ? npix : 4096;   /* prime a slice of pixels */
        for (size_t p = 0; p < warm; p++) {
            double acc = 0.0;
            for (size_t s = 0; s < K; s++) acc += sample_radiance(&rng, radiance, noise);
            image[p] = acc / (double)K;               /* keep the work observable */
        }
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One Monte-Carlo image pass. Re-seed each pass (mixing in the pass index)
         * so every pass re-randomises the sample rays rather than repeating. */
        p2_rng_seed(&rng, seed + passes);

        /* MAP + REDUCE per pixel: shoot K random sample rays and ACCUMULATE their
         * radiance into the pixel, then normalise by K (the Monte-Carlo mean).
         * The accumulation is the repeated read-modify-write of every image slot
         * -- the image-buffer accumulation write pattern. */
        for (size_t p = 0; p < npix; p++) {
            image[p] = 0.0;                           /* start this pixel's estimate */
            for (size_t s = 0; s < K; s++) {
                image[p] += sample_radiance(&rng, radiance, noise);  /* accumulate */
            }
            image[p] /= (double)K;                    /* Monte-Carlo mean radiance */
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Final-image statistics: mean pixel value (should approx radiance) and the
     * max abs deviation of any pixel from radiance (per-pixel Monte-Carlo error). */
    double sum = 0.0, max_dev = 0.0;
    for (size_t p = 0; p < npix; p++) {
        sum += image[p];
        double dev = fabs(image[p] - radiance);
        if (dev > max_dev) max_dev = dev;
    }
    double image_mean = (npix > 0) ? sum / (double)npix : 0.0;
    volatile double sink = image[(H / 2) * W + (W / 2)];   /* prevent dead-code elim */

    munmap(image, buf_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "image_mean", image_mean);
    p2_meta_kv_f64(&m, "max_abs_deviation", max_dev);
    p2_meta_kv_f64(&m, "center_value", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "not a scene renderer; integrates a known constant-radiance environment (zero-mean noise) so every pixel converges to radiance; image mean is very tight, per-pixel error ~ noise/sqrt(K)");
    p2_meta_close(&m);
    return 0;
}
