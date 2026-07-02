/* kernel_stencil_seidel_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  GAUSS-SEIDEL STENCIL:  2D 5-point red-black relaxation, updated in place
 * ============================================================================
 *
 *  DWARF   : Structured Grids (D5)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                  (first-division, memory-signature label)
 *  PURPOSE : Probe the in-place structured-grid write pattern: a SINGLE grid
 *            in which reads and writes alias the same pages, touched in a
 *            strided checkerboard order rather than one contiguous sweep.
 *
 *  PICTURE (top view):
 *      one buffer, checkerboard        two sub-passes per iteration
 *          R  B  R  B                  pass 1: write every R (reads its B nbrs)
 *          B  R  B  R                  pass 2: write every B (reads updated R)
 *          R  B  R  B                  Footprint ~1x: the same grid is both
 *          B  R  B  R                  the source and the destination.
 *      A cell's 4 neighbours are always the opposite colour, so within a
 *      sub-pass no cell depends on another cell written in that same sub-pass.
 *
 *  ALGORITHM:
 *      1. Allocate ONE N x N double grid. Seed the interior with random values
 *         and pin the boundary to a fixed constant; the boundary is never
 *         written by the sweeps, so it needs no second buffer to stay valid.
 *      2. Red sub-pass: overwrite every cell with (i+j) even by the mean of its
 *         four neighbours, read from the same buffer. Because neighbours are the
 *         opposite colour, these reads see only black cells and are consistent.
 *      3. Black sub-pass: overwrite every cell with (i+j) odd the same way; it
 *         now reads the RED cells that were just updated -- this is what makes it
 *         Gauss-Seidel (uses fresh values) rather than Jacobi (uses old values).
 *      4. Repeat both sub-passes for the timed duration.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A single fixed footprint (~1x, one grid) that is written and read on the
 *      SAME pages -- there is no separate output buffer. Each sub-pass writes
 *      only every other cell, so the writes land in a strided checkerboard with
 *      stride 2 rather than the dense contiguous front of the Jacobi variant.
 *      The read/write aliasing plus the interleaved-colour striping is the
 *      distinguishing tell versus the double-buffered Jacobi stencil.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Structured Grids. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (red-black sweeps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_stencil_seidel_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign 2D Gauss-Seidel red-black stencil; structured-grid kernel)\n"
"  --grid-n N            Grid side length (default 1024; uses N*N * 8 bytes, one buffer)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the initial field (default 42)\n"
"  --max-mb N            Hard cap on buffer bytes (default 8192)\n"
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

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long grid_n     = p2_get_i64(argc, argv, "--grid-n", 1024);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (grid_n < 16 || grid_n > 65536) {
        P2_LOG_ERR("grid-n %lld out of range (16..65536)", grid_n);
        return 2;
    }
    size_t N = (size_t)grid_n;
    size_t cells = N * N;
    size_t buf_bytes = cells * sizeof(double);   /* single buffer (in-place) */
    if (buf_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("buffer bytes %zu exceed --max-mb %lld", buf_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Structured Grids");
    p2_meta_kv_str(&m, "ordering", "red-black, in-place");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "grid_n", grid_n);
    p2_meta_kv_u64(&m, "total_bytes", buf_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *g = (double *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (g == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", buf_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(g, buf_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(g, buf_bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Interior random, boundary fixed to 1.0 (never written by the sweeps). */
    const double BOUND = 1.0;
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            int edge = (i == 0 || j == 0 || i == N - 1 || j == N - 1);
            g[i * N + j] = edge ? BOUND : p2_rng_unit(&rng);
        }
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t iters = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Two sub-passes: colour 0 (red, (i+j) even) then colour 1 (black).
         * The colours must be separated because Gauss-Seidel updates in place:
         * doing red first means the black pass reads already-updated red cells,
         * while within a single colour no cell reads another cell of the same
         * colour, so each sub-pass is internally order-independent. */
        for (int color = 0; color < 2; color++) {
            for (size_t i = 1; i < N - 1; i++) {
                double *row = g + i * N;              /* this row: written in place */
                const double *up = g + (i - 1) * N;   /* neighbour row above (N)    */
                const double *dn = g + (i + 1) * N;   /* neighbour row below (S)    */
                /* Index of the first interior cell of this colour in this row.
                 * Parity depends on the row, so this offset (0 or 1) alternates
                 * per row to keep the checkerboard aligned; from there we step by
                 * 2 to visit only the current colour. */
                size_t j0 = 1 + (((i + 1) + (size_t)color) & 1);
                for (size_t j = j0; j < N - 1; j += 2) {
                    /* In-place update: the same "row" pointer is both read (for
                     * the W/E neighbours) and written. The stride-2 j-loop is
                     * what produces the strided checkerboard write front. */
                    row[j] = 0.25 * (up[j] + dn[j] + row[j - 1] + row[j + 1]);
                }
            }
        }
        iters++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = g[(N / 2) * N + (N / 2)];

    munmap(g, buf_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "iterations", iters);
    p2_meta_kv_f64(&m, "center_value", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: in-place checkerboard writes; single footprint, periodic");
    p2_meta_close(&m);
    return 0;
}
