/* kernel_dwt_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  2D HAAR WAVELET TRANSFORM:  a multi-level Mallat pyramid over an image
 * ============================================================================
 *
 *  DWARF   : Spectral Methods (D3)      (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                     (first-division, memory-signature label)
 *  PURPOSE : Probe the write signature of a spectral transform whose active
 *            footprint SHRINKS geometrically across levels, as opposed to the
 *            fixed whole-array footprint of the FFT/NTT kernels.
 *
 *  A single level of the 2D discrete wavelet transform applies a filter-and-
 *  downsample step to every row and then to every column of the current image.
 *  With the Haar filter this is just averages and differences of adjacent pairs.
 *  The result splits the data into four quadrants: LL (coarse approximation),
 *  LH and HL (horizontal / vertical detail), and HH (diagonal detail). Because
 *  all the coarse energy is now packed into the top-left LL quadrant -- a quarter
 *  of the area -- the next level RECURSES on that quadrant alone. Each level
 *  therefore halves the side length (quarters the area) of the region it touches,
 *  producing the classic shrinking multi-resolution pyramid.
 *
 *  PICTURE (top view):  the quadrant split, and the pyramid recursing into LL.
 *
 *      after level 1            after level 2 (recurse on LL only)
 *      +--------+--------+      +----+----+--------+
 *      |        |        |      | LL | HL |        |
 *      |   LL   |   HL   |      +----+----+   HL   |   <- level-1 detail frozen
 *      |        |        |      | LH | HH |        |
 *      +--------+--------+      +----+----+--------+
 *      |        |        |      |         |        |
 *      |   LH   |   HH   |      |   LH    |   HH   |
 *      |        |        |      |         |        |
 *      +--------+--------+      +---------+--------+
 *        (LH/HL/HH are written once, then never revisited; work marches
 *         toward the top-left corner as the active region halves each level)
 *
 *  ALGORITHM (per measured pass):
 *      1. Refill the whole NxN image with fresh random samples.
 *      2. For each level, with the current side length m:
 *           a. transform all m rows      (Haar filter+downsample in place),
 *           b. transform all m columns (same, gathering each column first),
 *         which folds the coarse content into the top-left m/2 x m/2 quadrant.
 *      3. Set m = m/2 and repeat on that LL quadrant, until m < 2 or the
 *         requested level count is reached.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A write footprint that shrinks geometrically level by level -- the first
 *      level rewrites the entire array, the next a quarter of it, then a
 *      sixteenth, and so on -- a pyramid that migrates toward the top-left LL
 *      corner. This is distinct from the FFT/NTT (fixed whole-array butterfly)
 *      and the DCT (uniform small blocks). HONEST CAVEAT: the shrinking-pyramid
 *      tell is only visible if the level sweep is slow enough to span multiple
 *      write-signal snapshots; for a small image it can complete inside one
 *      snapshot and read as a single whole-array write. (The orthonormal Haar
 *      filter is used, so the transform is exactly invertible.)
 *
 *  Real-world use: JPEG2000 image compression, wavelet denoising, and multi-
 *  resolution analysis in audio and image processing.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Spectral Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (DWT passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_dwt_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign multi-level 2D Haar wavelet transform; Spectral kernel)\n"
"  --n N                 Square image side, snapped down to a power of 2 (default 1024)\n"
"  --levels L            Pyramid levels (default 0 = full, down to 2x2)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on image bytes (default 8192)\n"
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

/* One level of the 1D orthonormal Haar transform over a contiguous length-m
 * vector. Each adjacent pair (a, b) becomes a scaled average (a+b)/sqrt(2) and a
 * scaled difference (a-b)/sqrt(2); the M_SQRT1_2 factor (= 1/sqrt(2)) is what
 * makes the filter orthonormal and therefore exactly invertible. The m/2 averages
 * are packed into the FIRST half of the vector (the "low" band) and the m/2
 * differences into the SECOND half (the "high" band) -- this deinterleaving is
 * what later lets the 2D driver treat the top-left block as a self-contained
 * coarse image. A scratch buffer collects the result so input pairs are not
 * clobbered mid-pass, then it is copied back in place. */
static void haar1d(double *x, size_t m, double *tmp) {
    size_t h = m / 2;
    for (size_t i = 0; i < h; i++) {
        double a = x[2 * i], b = x[2 * i + 1];
        tmp[i]     = (a + b) * M_SQRT1_2;   /* low  band -> first half  */
        tmp[h + i] = (a - b) * M_SQRT1_2;   /* high band -> second half */
    }
    for (size_t i = 0; i < m; i++) x[i] = tmp[i];
}

/* Multi-level 2D Haar DWT on the NxN image (physical row stride is always N,
 * even as the active region shrinks), recursing on the LL quadrant. Each level
 * transforms the current m x m top-left sub-image in two separable passes -- all
 * rows, then all columns -- which sorts the coarse content into the top-left
 * m/2 x m/2 quadrant. Halving m each level is what makes the write footprint
 * contract toward the origin (the pyramid tell). Note the row pass is unit-stride
 * and cache-friendly, whereas the column pass strides by N and must gather each
 * column into a contiguous buffer before transforming and scattering it back. */
static void dwt2d(double *img, size_t N, int levels, double *rtmp, double *col, double *ctmp) {
    size_t m = N;
    for (int lev = 0; lev < levels && m >= 2; lev++) {
        for (size_t r = 0; r < m; r++) haar1d(img + r * N, m, rtmp);       /* transform rows */
        for (size_t c = 0; c < m; c++) {                                    /* transform columns */
            for (size_t r = 0; r < m; r++) col[r] = img[r * N + c];         /* gather column c */
            haar1d(col, m, ctmp);
            for (size_t r = 0; r < m; r++) img[r * N + c] = col[r];         /* scatter it back */
        }
        m /= 2;                                                             /* recurse on LL */
    }
}

static size_t snap_pow2(long long v) {
    size_t n = 1; while ((long long)(n << 1) <= v && (n << 1) != 0) n <<= 1; return n;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nreq       = p2_get_i64(argc, argv, "--n", 1024);
    long long levels_req = p2_get_i64(argc, argv, "--levels", 0);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nreq < 4 || nreq > 16384) { P2_LOG_ERR("n %lld out of range (4..16384)", nreq); return 2; }
    size_t N = snap_pow2(nreq);
    int max_levels = 0; for (size_t t = N; t >= 2; t >>= 1) max_levels++;
    int levels = (levels_req <= 0 || levels_req > max_levels) ? max_levels : (int)levels_req;
    size_t bytes = N * N * sizeof(double);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("image bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Spectral Methods");
    p2_meta_kv_str(&m, "scheme", "multi-level 2D Haar DWT (Mallat pyramid; recurses on LL, footprint halves per level)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_u64(&m, "n", N);
    p2_meta_kv_i64(&m, "levels", levels);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *img = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (img == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(img, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(img, bytes);

    double *rtmp = (double *)malloc(N * sizeof(double));
    double *col  = (double *)malloc(N * sizeof(double));
    double *ctmp = (double *)malloc(N * sizeof(double));
    if (!rtmp || !col || !ctmp) { free(rtmp); free(col); free(ctmp); munmap(img, bytes);
        P2_LOG_ERR("malloc failed"); p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t k = 0; k < N * N; k++) img[k] = rng_unit(&rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Repaint the whole image with fresh samples so each pass starts from a
         * full-array write, then run the pyramid. The refill dominates the write
         * volume; the transform contributes the shrinking-footprint structure. */
        for (size_t k = 0; k < N * N; k++) img[k] = rng_unit(&rng);   /* fresh image */
        dwt2d(img, N, levels, rtmp, col, ctmp);
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = img[0];   /* the coarsest LL average */

    free(rtmp); free(col); free(ctmp);
    munmap(img, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "coarsest_ll", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "Haar (shortest) wavelet; the shrinking-pyramid tell needs the level sweep to span multiple 500ms snapshots");
    p2_meta_close(&m);
    return 0;
}
