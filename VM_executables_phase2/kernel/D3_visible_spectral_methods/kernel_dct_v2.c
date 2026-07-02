/* kernel_dct_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  Blocked DCT:  8x8 2D Discrete Cosine Transform (DCT-II, the JPEG transform)
 * ============================================================================
 *
 *  DWARF   : Spectral Methods (D3)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                  (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of a BLOCKED spectral transform. Unlike the
 *            FFT, which rewrites one large array end to end, the DCT tiles the
 *            image into independent 8x8 blocks and transforms each in place, so
 *            the host sees a stream of many tiny, localised block rewrites.
 *
 *  PICTURE (top view):
 *      The image is tiled into a grid of 8x8 blocks. Each block is transformed
 *      on its own (Y = C X C^T) with no data shared between blocks -- the write
 *      front is a small 8x8 footprint sweeping across the image, tile by tile:
 *
 *          +----+----+----+----+ ... +
 *          | 8x8| 8x8| 8x8| 8x8|     |   each cell is an independent
 *          +----+----+----+----+ ... +   64-element block; block_dct()
 *          | 8x8| 8x8| 8x8| 8x8|     |   reads it, transforms it, and
 *          +----+----+----+----+ ... +   writes it back before moving on
 *          | 8x8| 8x8| 8x8| 8x8|     |
 *          +----+----+----+----+ ... +   (contrast: the FFT touches ONE array
 *          :    :    :    :    :     :    that spans the whole buffer at once)
 *
 *  ALGORITHM:
 *      1. Precompute the 8x8 orthonormal DCT-II matrix C once (cosine basis).
 *      2. For each 8x8 block at (r0, c0): copy it out to a local 8x8 tile X,
 *         apply the separable transform Y = C * X * C^T (a row DCT C*X followed
 *         by a column DCT (C*X)*C^T), and write Y back into the image in place.
 *      3. Re-seed the whole image and re-transform every block each pass, so the
 *         buffer keeps changing for the entire capture window.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A tiled sweep of many small, independent 8x8 rewrites across the image --
 *      a blocked footprint with real-valued content, quite unlike the FFT's
 *      single whole-array butterfly. Honest caveat: over one full pass the total
 *      write VOLUME is still the whole image; the distinguishing feature is the
 *      granularity (repeated 64-element blocks), not the amount written.
 *
 *  Real-world use: JPEG image and MPEG/H.264 video compression (the DCT is the
 *  energy-compaction step before quantisation).
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Spectral Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (block DCT passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_dct_v2";
#define BS 8   /* DCT block side (JPEG uses 8x8) */

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign blocked 8x8 2D DCT; Spectral kernel)\n"
"  --height H            Image height, snapped to a multiple of 8 (default 1024)\n"
"  --width W             Image width, snapped to a multiple of 8 (default 1024)\n"
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

/* Build the 8x8 orthonormal DCT-II basis matrix C once, up front.
 * Row k holds the k-th cosine basis vector; the k == 0 (DC) row uses the
 * normalising factor sqrt(1/BS) and all higher rows use sqrt(2/BS), which makes
 * C orthonormal (C C^T = I). Orthonormality is what lets the same matrix serve
 * both directions of the separable transform: the inverse DCT is simply C^T. */
static void build_dct_matrix(double C[BS][BS]) {
    for (size_t k = 0; k < BS; k++)
        for (size_t n = 0; n < BS; n++) {
            double a = (k == 0) ? sqrt(1.0 / BS) : sqrt(2.0 / BS);
            C[k][n] = a * cos(M_PI * (2.0 * n + 1.0) * k / (2.0 * BS));
        }
}

/* Transform one 8x8 image block in place as Y = C * X * C^T (the separable 2D
 * DCT-II). "img" is the full row-major image of width W; (r0, c0) is the block's
 * top-left corner. Work happens in small stack tiles (X, T, Y) so the image is
 * touched only twice per block -- one gather to read X and one scatter to write
 * Y -- which is exactly the localised 8x8 write the memory signature depends on. */
static void block_dct(double *img, size_t W, size_t r0, size_t c0, const double C[BS][BS]) {
    double X[BS][BS], T[BS][BS], Y[BS][BS];
    /* Gather the block out of the image into the local tile X. The row stride is
     * W, so consecutive rows of the block are W elements apart in memory. */
    for (size_t i = 0; i < BS; i++)
        for (size_t j = 0; j < BS; j++) X[i][j] = img[(r0 + i) * W + (c0 + j)];
    /* First (row) DCT: T = C * X. Each output column j is C applied to column j
     * of X, i.e. the cosine transform along the block's vertical axis. */
    for (size_t k = 0; k < BS; k++)
        for (size_t j = 0; j < BS; j++) {
            double s = 0.0; for (size_t n = 0; n < BS; n++) s += C[k][n] * X[n][j];
            T[k][j] = s;
        }
    /* Second (column) DCT: Y = T * C^T. Multiplying by C^T on the right applies
     * the cosine transform along the horizontal axis, completing the 2D DCT.
     * Note C[m][n] indexes the transpose (row m of C^T is column m of C). */
    for (size_t k = 0; k < BS; k++)
        for (size_t m = 0; m < BS; m++) {
            double s = 0.0; for (size_t n = 0; n < BS; n++) s += T[k][n] * C[m][n];
            Y[k][m] = s;
        }
    /* Scatter the finished coefficient block back into the image in place. */
    for (size_t i = 0; i < BS; i++)
        for (size_t j = 0; j < BS; j++) img[(r0 + i) * W + (c0 + j)] = Y[i][j];
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long Hin        = p2_get_i64(argc, argv, "--height", 1024);
    long long Win        = p2_get_i64(argc, argv, "--width", 1024);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (Hin < BS || Hin > 32768 || Win < BS || Win > 32768) { P2_LOG_ERR("H/W out of range (8..32768)"); return 2; }
    size_t H = (size_t)(Hin - (Hin % BS)), W = (size_t)(Win - (Win % BS));   /* snap to 8 */
    size_t bytes = H * W * sizeof(double);
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
    p2_meta_kv_str(&m, "scheme", "blocked 8x8 2D DCT-II (separable orthonormal; JPEG transform)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "height", (long long)H);
    p2_meta_kv_i64(&m, "width", (long long)W);
    p2_meta_kv_i64(&m, "block", BS);
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

    double C[BS][BS]; build_dct_matrix(C);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t k = 0; k < H * W; k++) img[k] = rng_unit(&rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed the whole image, then transform every 8x8 tile in row-major
         * block order. The nested r0/c0 loops step by BS so each iteration lands
         * on the top-left corner of one block; the "+ BS <= H/W" guards drop any
         * ragged edge that a non-multiple-of-8 dimension would leave (dimensions
         * are snapped to multiples of 8 above, so in practice nothing is lost). */
        for (size_t k = 0; k < H * W; k++) img[k] = rng_unit(&rng);   /* fresh image */
        for (size_t r0 = 0; r0 + BS <= H; r0 += BS)
            for (size_t c0 = 0; c0 + BS <= W; c0 += BS)
                block_dct(img, W, r0, c0, C);
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = img[0];   /* a DC coefficient */

    munmap(img, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "block0_dc", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "blocked-footprint tell (many small 8x8 rewrites) vs the FFT's whole-array butterfly");
    p2_meta_close(&m);
    return 0;
}
