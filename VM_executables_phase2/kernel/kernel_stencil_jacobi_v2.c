/* kernel_stencil_jacobi_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  JACOBI STENCIL:  2D 5-point iterative grid relaxation with double buffering
 * ============================================================================
 *
 *  DWARF   : Structured Grids (D5)   (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                  (first-division, memory-signature label)
 *  PURPOSE : Probe the canonical iterative-grid write pattern: a full,
 *            contiguous rewrite of one N x N grid per sweep, with strong
 *            temporal periodicity because the same footprint is revisited
 *            every iteration.
 *
 *  PICTURE (top view):
 *      5-point stencil            two buffers, roles swap each sweep
 *          .  N  .                    read            write
 *          W  C  E                  +-------+       +-------+
 *          .  S  .                  |  A    | ----> |  B    |
 *                                   +-------+       +-------+
 *      next[C] = 0.25 * (N+S+W+E)   then swap: next sweep reads B, writes A.
 *                                   Footprint ~2x: two live grids at all times.
 *
 *  ALGORITHM:
 *      1. Allocate two N x N double grids ("cur" and "next"). Seed the interior
 *         of "cur" with random values and pin the boundary to a fixed constant
 *         in BOTH grids so the never-written edge stays valid across swaps.
 *      2. For every interior cell, write next[i][j] as the arithmetic mean of
 *         its four von-Neumann neighbours read from "cur". Reads and writes
 *         never alias: the input grid is untouched while the output is filled.
 *      3. Swap the "cur" and "next" pointers and repeat for the timed duration.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      One dense, sequential N x N write front per sweep, landing in whichever
 *      buffer is currently "next". Because the two buffers alternate roles, the
 *      write target ping-pongs between two fixed, equally sized regions with
 *      clean period-2 regularity. The read grid is quiescent for the whole
 *      sweep, so writes stay separated from reads -- the defining contrast with
 *      the in-place Gauss-Seidel variant, which writes and reads the same pages.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Structured Grids. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (Jacobi sweeps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_stencil_jacobi_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign 2D Jacobi stencil; structured-grid kernel)\n"
"  --grid-n N            Grid side length (default 1024; uses 2 * N*N * 8 bytes)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the initial field (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes (default 8192)\n"
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
    size_t buf_bytes = cells * sizeof(double);
    size_t total_bytes = 2 * buf_bytes;
    if (total_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("total bytes %zu exceed --max-mb %lld", total_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Structured Grids");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "grid_n", grid_n);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *cur = (double *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    double *next = (double *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (cur == MAP_FAILED || next == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu x2) failed: %s", buf_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(cur, buf_bytes, MADV_NOHUGEPAGE);
    p2_madvise(next, buf_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) { p2_mlock_soft(cur, buf_bytes); p2_mlock_soft(next, buf_bytes); }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Initial field: interior random, boundary fixed to 1.0 in BOTH buffers so
     * the never-written boundary stays valid across buffer swaps. */
    const double BOUND = 1.0;
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            int edge = (i == 0 || j == 0 || i == N - 1 || j == N - 1);
            cur[i * N + j]  = edge ? BOUND : p2_rng_unit(&rng);
            next[i * N + j] = edge ? BOUND : 0.0;
        }
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t iters = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One Jacobi sweep: fill every interior cell of "next" from "cur".
         * Row pointers are hoisted out of the inner loop so the neighbour reads
         * are three plain sequential streams (row above, this row, row below)
         * and the write is one dense sequential stream into "out" -- the tight,
         * cache-friendly access pattern that defines a structured-grid stencil. */
        for (size_t i = 1; i < N - 1; i++) {
            const double *up = cur + (i - 1) * N;     /* neighbour row above (N) */
            const double *dn = cur + (i + 1) * N;     /* neighbour row below (S) */
            const double *md = cur + i * N;           /* this row: supplies W and E */
            double *out = next + i * N;               /* destination row in "next" */
            for (size_t j = 1; j < N - 1; j++) {
                /* next[i][j] = mean of the 4 von-Neumann neighbours. Everything
                 * on the right comes from "cur", so this write cannot disturb
                 * any value still needed by the rest of the sweep. */
                out[j] = 0.25 * (up[j] + dn[j] + md[j - 1] + md[j + 1]);
            }
        }
        /* Swap the buffer roles: the grid just written becomes the read source
         * for the next sweep. This pointer swap (not a copy) is what makes the
         * write target alternate between two fixed regions period-by-period. */
        double *tmp = cur; cur = next; next = tmp;   /* swap buffers */
        iters++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = cur[(N / 2) * N + (N / 2)];   /* prevent dead-code elim */

    munmap(cur, buf_bytes);
    munmap(next, buf_bytes);

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
                   "expected signature: regular neighbour writes + strong temporal periodicity");
    p2_meta_close(&m);
    return 0;
}
