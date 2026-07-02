/* kernel_fft_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  1D FFT:  in-place radix-2 Cooley-Tukey Fast Fourier Transform of N points
 * ============================================================================
 *
 *  DWARF   : Spectral Methods (D3)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                  (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of a multi-stride, non-contiguous in-place
 *            rewrite. Unlike a stencil (fixed neighbour stride), the FFT touches
 *            the buffer at a stride that CHANGES on every stage, so the host sees
 *            a write pattern that walks across all scales of the array.
 *
 *  PICTURE (top view):
 *      The array is one contiguous complex vector, stored interleaved as
 *      double[2N] (re, im pairs). Two conceptually different write passes hit it:
 *
 *        (a) bit-reversal permutation -- a one-off scatter that swaps element i
 *            with element reverse-bits(i), e.g. for N = 8:
 *
 *                index :  0  1  2  3  4  5  6  7
 *                goes to:  0  4  2  6  1  5  3  7      (irregular, long-range)
 *
 *        (b) log2(N) butterfly stages -- each stage combines pairs a distance
 *            "len/2" apart, and that distance DOUBLES from stage to stage:
 *
 *                a --------o------- a' = a + w*b        (w = twiddle factor)
 *                           \   /
 *                            \ /
 *                             X
 *                            / \
 *                           /   \
 *                b --------o------- b' = a - w*b
 *
 *                stage 1: pairs 1 apart   |o o||o o||o o||o o|
 *                stage 2: pairs 2 apart   |o . o .||o . o .|
 *                stage 3: pairs 4 apart   |o . . . o . . .|      (stride grows)
 *
 *  ALGORITHM:
 *      1. Permute the array into bit-reversed order (the scatter above).
 *      2. For len = 2, 4, 8, ..., N (log2(N) stages): sweep the array in blocks
 *         of "len", and inside each block apply len/2 butterflies that fold the
 *         upper half into the lower half using twiddle factors w = exp(-2pi i/len)
 *         advanced incrementally (cos/sin computed once per stage).
 *      3. Re-randomise the real input and repeat, so the buffer keeps changing
 *         for the whole capture window (the transform alone is deterministic).
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A bit-reversal scatter followed by log2(N) in-place passes whose write
 *      stride doubles each pass (1, 2, 4, ..., N/2). Honest caveat: every pass
 *      still rewrites the entire buffer once, so the total write VOLUME per
 *      transform is uniform; the distinguishing feature is the varying stride /
 *      access geometry, not the amount written.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Spectral Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (transforms) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const char *TEST = "kernel_fft_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign in-place radix-2 FFT; spectral kernel)\n"
"  --n N                 FFT points (snapped to a power of 2; default 1048576)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the input (default 42)\n"
"  --max-mb N            Hard cap on buffer bytes (default 8192)\n"
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

/* In-place radix-2 forward FFT on the interleaved complex array a[2N] (re,im).
 * The transform runs in two passes over the same buffer: a bit-reversal
 * permutation that reorders the samples, then log2(N) butterfly stages that do
 * the actual arithmetic. Both passes write a[] in place; together they are the
 * distinctive multi-stride write the host signal is meant to observe. */
static void fft(double *a, size_t N) {
    /* Pass 1 -- bit-reversal permutation.
     * Cooley-Tukey needs the input in bit-reversed index order before the
     * butterflies. We walk i upward while maintaining j = reverse-bits(i) with an
     * incrementing "binary carry" trick (add one to the most-significant bit and
     * propagate downward). Each pair (i, j) is swapped exactly once by guarding
     * with (i < j), which is where the irregular long-range scatter happens. */
    for (size_t i = 1, j = 0; i < N; i++) {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;   /* clear the run of high 1-bits */
        j ^= bit;                              /* set the next lower 0-bit (carry) */
        if (i < j) {                           /* swap the interleaved re/im pair */
            double tr = a[2 * i], ti = a[2 * i + 1];
            a[2 * i] = a[2 * j]; a[2 * i + 1] = a[2 * j + 1];
            a[2 * j] = tr; a[2 * j + 1] = ti;
        }
    }
    /* Pass 2 -- butterfly stages.
     * "len" is the size of the sub-transform being merged and doubles each stage
     * (2, 4, 8, ..., N), so the distance between a butterfly's two partners,
     * len/2, also doubles -- this is the stride that grows across stages. */
    for (size_t len = 2; len <= N; len <<= 1) {
        /* wl = principal len-th root of unity exp(-2pi i / len). All twiddles in
         * this stage are integer powers of wl, generated incrementally below so
         * cos/sin are evaluated only once per stage rather than per butterfly. */
        double ang = -2.0 * M_PI / (double)len;
        double wl_re = cos(ang), wl_im = sin(ang);
        for (size_t i = 0; i < N; i += len) {       /* each length-"len" block */
            double w_re = 1.0, w_im = 0.0;          /* twiddle starts at w^0 = 1 */
            for (size_t k = 0; k < len / 2; k++) {
                /* One radix-2 butterfly on the pair (a0, a1) that are len/2 apart:
                 *   u = a0 ; v = w * a1  ; a0 = u + v ; a1 = u - v
                 * i.e. the classic (a, b) -> (a + w*b, a - w*b). */
                size_t a0 = 2 * (i + k), a1 = 2 * (i + k + len / 2);
                double u_re = a[a0], u_im = a[a0 + 1];
                double v_re = a[a1] * w_re - a[a1 + 1] * w_im;   /* complex w*a1 */
                double v_im = a[a1] * w_im + a[a1 + 1] * w_re;
                a[a0] = u_re + v_re; a[a0 + 1] = u_im + v_im;    /* lower output */
                a[a1] = u_re - v_re; a[a1 + 1] = u_im - v_im;    /* upper output */
                /* Advance the twiddle: w <- w * wl (one root-of-unity step). */
                double nw_re = w_re * wl_re - w_im * wl_im;
                double nw_im = w_re * wl_im + w_im * wl_re;
                w_re = nw_re; w_im = nw_im;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long n_in       = p2_get_i64(argc, argv, "--n", 1048576);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (n_in < 16 || n_in > (1LL << 30)) { P2_LOG_ERR("n %lld out of range", n_in); return 2; }
    /* snap N up to a power of two */
    size_t N = 1; while (N < (size_t)n_in) N <<= 1;

    size_t bytes = 2 * N * sizeof(double);    /* interleaved complex */
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Spectral Methods");
    p2_meta_kv_str(&m, "scheme", "in-place radix-2 Cooley-Tukey FFT");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_u64(&m, "n", N);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *a = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (a == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(a, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(a, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) { a[2 * i] = rng_unit(&rng); a[2 * i + 1] = 0.0; }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t transforms = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed with a fresh real-valued signal (real part random, imaginary
         * part zero) before every transform. The FFT itself is deterministic, so
         * without re-seeding the buffer contents would converge and the write
         * signal would go stale; re-randomising keeps it representative. */
        for (size_t i = 0; i < N; i++) { a[2 * i] = rng_unit(&rng); a[2 * i + 1] = 0.0; }
        fft(a, N);
        transforms++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = a[0] + a[1];

    munmap(a, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "transforms", transforms);
    p2_meta_kv_f64(&m, "sink", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: stage-varying-stride butterfly writes + bit-reversal scatter");
    p2_meta_close(&m);
    return 0;
}
