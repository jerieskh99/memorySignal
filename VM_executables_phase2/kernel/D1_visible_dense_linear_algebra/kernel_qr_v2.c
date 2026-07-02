/* kernel_qr_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  QR ORTHOGONALISATION:  modified Gram-Schmidt  A = Q * R  (column-major)
 * ============================================================================
 *
 *  DWARF   : Dense Linear Algebra (Berkeley computational motif D1)
 *  FAMILY  : KERNEL       (first-division, memory-signature label)
 *  PURPOSE : Probe the write-signal of a workload whose active region GROWS
 *            over time. Each column is orthogonalised against ALL columns to
 *            its left, so the work (and the span of memory read) per column
 *            rises as the sweep advances -- the exact mirror image of LU's
 *            shrinking front, and distinct from GEMM's static full rewrite.
 *
 *  PICTURE (top view):
 *      Columns are processed left to right. To finish column j the algorithm
 *      reads every already-orthonormal column q_0 .. q_{j-1} ("q" below) and
 *      rewrites the working column j ("|"); columns to the right ("." ) are
 *      untouched so far. The read/update span widens as j advances.
 *
 *          j = 1            j = 3            j = 5
 *      +--------------+  +--------------+  +--------------+
 *      | q | . . . .  |  | q q q | . .  |  | q q q q q |  |
 *      | q | . . . .  |  | q q q | . .  |  | q q q q q |  |
 *      | q | . . . .  |  | q q q | . .  |  | q q q q q |  |
 *      | q | . . . .  |  | q q q | . .  |  | q q q q q |  |
 *      +--------------+  +--------------+  +--------------+
 *        reads 1 col       reads 3 cols      reads 5 cols  ->  growing front
 *
 *  ALGORITHM (per orthogonalisation, for column j = 0 .. N-1):
 *      1. Project out every earlier orthonormal column: for each i < j compute
 *         the inner product r = <q_i, a_j> and subtract  a_j -= r * q_i.
 *         (This is the "modified" variant: each subtraction uses the partially
 *         updated a_j, which is numerically more stable than classical G-S.)
 *      2. Normalise the residual: q_j = a_j / ||a_j||, so column j becomes the
 *         next unit vector orthogonal to all the previous ones.
 *      Each measure pass first re-seeds A and re-orthogonalises from scratch.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      The matrix is rewritten one column at a time, and finishing each column
 *      involves an ever-longer series of read/update passes over the columns to
 *      its left -- a write front that expands as the sweep proceeds. Honest
 *      caveat: this growth is only resolvable if a single orthogonalisation
 *      spans several host snapshot intervals; at small N the whole sweep lands
 *      inside one snapshot and looks like a single full-matrix touch. Large N
 *      is required to expose the growing front.
 *
 *  Real-world use: least-squares regression, and the orthogonalisation inner
 *  loop of Krylov eigen/linear solvers (GMRES, Arnoldi).
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Dense Linear Algebra. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (orthogonalisations) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <math.h>

static const char *TEST = "kernel_qr_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Gram-Schmidt/QR orthogonalisation; Dense-LA kernel)\n"
"  --dim N               Matrix side length (default 1024; uses N*N * 8 bytes)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on matrix bytes (default 8192)\n"
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

/* Re-seed the whole matrix with fresh random entries in [-1,1). The layout is
 * column-major -- element (row, col) lives at A[col * N + row] -- so that each
 * logical column a_j is a single contiguous run A[j*N .. j*N + N-1]. That is the
 * unit the orthogonalisation reads and writes, so contiguity keeps every column
 * pass cache-friendly. Random (not diagonally biased) entries are fine here: QR
 * needs no pivoting, only linearly independent columns, which random data gives
 * with probability one. */
static void reseed(double *A, size_t N, p2_rng_t *rng) {
    size_t tot = N * N;
    for (size_t k = 0; k < tot; k++) A[k] = 2.0 * rng_unit(rng) - 1.0;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long dim        = p2_get_i64(argc, argv, "--dim", 1024);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (dim < 16 || dim > 16384) { P2_LOG_ERR("dim %lld out of range (16..16384)", dim); return 2; }
    size_t N = (size_t)dim;
    size_t bytes = N * N * sizeof(double);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("matrix bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Dense Linear Algebra");
    p2_meta_kv_str(&m, "scheme", "modified Gram-Schmidt / QR orthogonalisation (growing dependency front)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "dim", dim);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *A = (double *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (A == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(A, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(A, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    reseed(A, N, &rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t orths = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        reseed(A, N, &rng);   /* fresh random columns each pass */
        /* Modified Gram-Schmidt sweep. Column j is orthogonalised against the
         * j columns already made orthonormal to its left, so the inner i-loop
         * lengthens as j grows -- this is what makes the read/write front expand
         * across the sweep (the growing-front tell). */
        for (size_t j = 0; j < N; j++) {
            double *aj = A + j * N;                 /* working column j (contiguous) */
            for (size_t i = 0; i < j; i++) {
                const double *qi = A + i * N;       /* earlier orthonormal column q_i */
                /* Inner product r = <q_i, a_j>: how much of q_i is present in a_j. */
                double r = 0.0;
                for (size_t t = 0; t < N; t++) r += qi[t] * aj[t];
                /* Remove that component so a_j becomes orthogonal to q_i. Using
                 * the just-updated a_j (rather than the original column) is the
                 * "modified" step, which resists round-off growth. */
                for (size_t t = 0; t < N; t++) aj[t] -= r * qi[t];
            }
            /* Normalise the orthogonal residual to unit length -> the new q_j. */
            double nrm = 0.0;
            for (size_t t = 0; t < N; t++) nrm += aj[t] * aj[t];
            nrm = sqrt(nrm);
            /* Guard the reciprocal: a (near-)zero norm means the column was
             * linearly dependent, so scale by 0 instead of dividing by ~0. */
            double inv = nrm > 1e-300 ? 1.0 / nrm : 0.0;
            for (size_t t = 0; t < N; t++) aj[t] *= inv;
        }
        orths++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = A[(N - 1) * N + (N - 1)];   /* last normalised entry */

    munmap(A, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "orthogonalisations", orths);
    p2_meta_kv_f64(&m, "last_entry", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "growing-front tell needs one orthogonalisation to span multiple 500ms snapshots (large N)");
    p2_meta_close(&m);
    return 0;
}
