/* kernel_fft2d_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  2D FFT:  separable row/column Fast Fourier Transform of an N x N field
 * ============================================================================
 *
 *  DWARF   : Spectral Methods (D3)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                  (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of a full-array TRANSPOSE. A 2D FFT is
 *            separable, so it is just 1D FFTs along both axes; the interesting
 *            memory event is the transpose that sits between the two directions,
 *            a long-range strided scatter that the 1D FFT never performs.
 *
 *  PICTURE (top view):
 *      The data is one N x N complex array in row-major order (row stride = N
 *      complex = 2N doubles). A full transform is four passes:
 *
 *        rows           transpose            "cols"          transpose back
 *      +--------+      (i,j)<->(j,i)       +--------+       (undo the swap)
 *      | ====== |     .----.               | ====== |      .----.
 *      | ====== | --> |    | --> FFT each   | ====== | -->  |    | --> natural
 *      | ====== |     '----' each row now   | ====== |      '----'    order
 *      | ====== |    the transpose is a     | ====== |
 *      +--------+    diagonal element swap  +--------+
 *
 *      FFT-every-row  ->  [ transpose ]  ->  FFT-every-row  ->  [ transpose ]
 *      (horizontal)       swaps D[i][j]      (was the columns)   restore layout
 *                         with D[j][i]
 *
 *  ALGORITHM:
 *      1. FFT every row in place (contiguous 1D transforms, sequential stride).
 *      2. Transpose the whole array: swap element (i,j) with (j,i) for j > i.
 *         Row i is read contiguously but scattered down column i, so this pass
 *         hits memory with a stride of one full row (N complex) -- the tell.
 *      3. FFT every row again. Because of step 2 these rows are the ORIGINAL
 *         columns, so this completes the transform along the second axis.
 *      4. Transpose back so the result is in natural (row-major) order.
 *      5. Re-randomise the field and repeat for the whole capture window.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Two directional sweeps of butterfly rewrites (steps 1 and 3), each
 *      separated by a full-array transpose (steps 2 and 4) whose writes jump a
 *      row-sized stride. This is distinct from the 1D FFT (single direction, no
 *      transpose) and from the DCT (small blocked footprint). Honest caveat: the
 *      transpose here is a naive in-place double loop, not a cache-blocked one,
 *      so its access pattern is the plain diagonal swap described above.
 *
 *  Real-world use: image and optical spectral filtering, medical imaging,
 *  crystallography, and the pressure solve in spectral turbulence (DNS) codes.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Spectral Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (2D FFT passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_fft2d_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign 2D FFT: row FFTs, transpose, column FFTs; Spectral kernel)\n"
"  --n N                 Square side, snapped down to a power of 2 (default 1024; uses 2*N*N * 8 bytes)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
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

/* In-place forward radix-2 Cooley-Tukey FFT of n complex points, interleaved as
 * [re, im, ...]. This is the standard one-dimensional building block that the 2D
 * driver applies to each row; the 2D character comes entirely from calling it
 * along both axes (with a transpose in between), not from this routine itself. */
static void fft1d(double *a, size_t n) {
    /* Bit-reversal permutation: reorder samples into bit-reversed index order,
     * maintaining j = reverse-bits(i) by the incrementing binary-carry trick. */
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;   /* clear the run of high 1-bits */
        j ^= bit;                              /* set the next lower 0-bit (carry) */
        if (i < j) {                           /* swap each pair once (i < j) */
            double tr = a[2 * i], ti = a[2 * i + 1];
            a[2 * i] = a[2 * j]; a[2 * i + 1] = a[2 * j + 1];
            a[2 * j] = tr; a[2 * j + 1] = ti;
        }
    }
    /* Butterfly stages: sub-transform size "len" doubles each stage, so partner
     * distance len/2 (the write stride) doubles too. wl is the len-th root of
     * unity; the per-stage twiddle w is advanced incrementally (see below). */
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        double wlr = cos(ang), wli = sin(ang);
        for (size_t i = 0; i < n; i += len) {       /* each length-"len" block */
            double wr = 1.0, wi = 0.0;              /* twiddle starts at w^0 = 1 */
            for (size_t k = 0; k < len / 2; k++) {
                /* Radix-2 butterfly (e = even/lower, o = odd/upper, len/2 apart):
                 *   t = w * a[o] ; a[e] = a[e] + t ; a[o] = a[e] - t. */
                size_t e = i + k, o = i + k + len / 2;
                double or_ = a[2 * o], oi = a[2 * o + 1];
                double tr = or_ * wr - oi * wi, ti = or_ * wi + oi * wr;  /* w*a[o] */
                double er = a[2 * e], ei = a[2 * e + 1];
                a[2 * e] = er + tr; a[2 * e + 1] = ei + ti;   /* lower output */
                a[2 * o] = er - tr; a[2 * o + 1] = ei - ti;   /* upper output */
                double nwr = wr * wlr - wi * wli; wi = wr * wli + wi * wlr; wr = nwr;
            }
        }
    }
}

/* In-place transpose of the N x N complex array (row stride = N complex points).
 * This is the pass that turns "FFT the rows" into "FFT the columns": after it,
 * what were columns occupy contiguous rows. Only the strict upper triangle is
 * visited (j > i) so every off-diagonal pair is swapped exactly once and the
 * diagonal is left untouched. Element (i,j) is read from a nearly contiguous row
 * but written down column i, a full row (2N doubles) away -- this long, regular
 * stride is the workload's signature access that no 1D transform produces. */
static void transpose(double *D, size_t N) {
    for (size_t i = 0; i < N; i++)
        for (size_t j = i + 1; j < N; j++) {
            /* a = linear index of (i,j), b = linear index of (j,i); *2 because
             * each complex element is a re/im pair of doubles. */
            size_t a = i * N + j, b = j * N + i;
            double tr = D[2 * a], ti = D[2 * a + 1];
            D[2 * a] = D[2 * b]; D[2 * a + 1] = D[2 * b + 1];
            D[2 * b] = tr; D[2 * b + 1] = ti;
        }
}

static size_t snap_pow2(long long v) {
    size_t n = 1; while ((long long)(n << 1) <= v && (n << 1) != 0) n <<= 1; return n;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nreq       = p2_get_i64(argc, argv, "--n", 1024);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nreq < 4 || nreq > 8192) { P2_LOG_ERR("n %lld out of range (4..8192)", nreq); return 2; }
    size_t N = snap_pow2(nreq);
    size_t bytes = 2 * N * N * sizeof(double);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("buffer bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Spectral Methods");
    p2_meta_kv_str(&m, "scheme", "2D FFT (row FFTs, transpose, column FFTs, transpose back); transpose = strided scatter");
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

    double *D = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (D == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(D, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(D, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t k = 0; k < 2 * N * N; k++) D[k] = rng_unit(&rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One full 2D FFT = row pass, transpose, "column" pass, transpose back.
         * The field is re-randomised first so the write signal stays fresh (the
         * transform is deterministic and would otherwise settle). Each fft1d call
         * is handed the base of row r: D + r*(2N) doubles into the buffer. */
        for (size_t k = 0; k < 2 * N * N; k++) D[k] = rng_unit(&rng);   /* fresh complex field */
        for (size_t r = 0; r < N; r++) fft1d(D + r * 2 * N, N);         /* row FFTs */
        transpose(D, N);
        for (size_t r = 0; r < N; r++) fft1d(D + r * 2 * N, N);         /* column FFTs */
        transpose(D, N);                                                /* back to natural order */
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = D[0];   /* DC (real part of the (0,0) coefficient) */

    munmap(D, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "dc_real", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "the transpose scatter is the distinct tell vs the 1D FFT; two directional passes per transform");
    p2_meta_close(&m);
    return 0;
}
