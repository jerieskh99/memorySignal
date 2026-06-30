/* cpu_matrix_mult_v2  --  benign system-behaviour benchmark
 *
 * Repeated naive square matrix multiply C = A * B. A and B are fixed after
 * warmup; C is rewritten every iteration, so this is compute-heavy with a
 * structured, reused write footprint (the C matrix). Tune --dim so the working
 * set straddles L2/L3/LLC.
 *
 * No network, no persistence, no filesystem writes, no sandbox. See
 * docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (init A,B) -> measure (matmul loop) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "cpu_matrix_mult_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign CPU+memory benchmark)\n"
"  --dim N               Square matrix dimension (default 512, max 4096)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long dim        = p2_get_i64(argc, argv, "--dim", 512);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (dim < 2 || dim > 4096) {
        P2_LOG_ERR("dim %lld out of range (2..4096)", dim);
        return 2;
    }
    size_t n = (size_t)dim;
    size_t elems = n * n;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "CPU");
    p2_meta_kv_i64(&m, "dim", dim);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/filesystem-writes");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    double *A = (double *)malloc(elems * sizeof(double));
    double *B = (double *)malloc(elems * sizeof(double));
    double *C = (double *)malloc(elems * sizeof(double));
    if (!A || !B || !C) {
        P2_LOG_ERR("malloc(3 x %zu doubles) failed", elems);
        free(A); free(B); free(C);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < elems; i++) {
        A[i] = (double)(p2_rng_next(&rng) & 0xFFFF) / 65535.0;
        B[i] = (double)(p2_rng_next(&rng) & 0xFFFF) / 65535.0;
    }
    memset(C, 0, elems * sizeof(double));
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    uint64_t passes = 0;
    volatile double sink = 0.0;
    while ((p2_monotonic() - t_warm) < (double)duration_s) {
        for (size_t i = 0; i < n; i++) {
            for (size_t k = 0; k < n; k++) {
                double a = A[i*n + k];
                const double *brow = &B[k*n];
                double *crow = &C[i*n];
                for (size_t j = 0; j < n; j++) crow[j] += a * brow[j];
            }
        }
        sink += C[(passes % n) * n + (passes % n)];
        memset(C, 0, elems * sizeof(double));   /* reset accumulator for next pass */
        passes++;
    }
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    free(A); free(B); free(C);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "matmul_passes", passes);
    p2_meta_kv_f64(&m, "sink", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "naive triple-loop matmul; cache behaviour depends on --dim vs LLC size");
    p2_meta_close(&m);
    return 0;
}
