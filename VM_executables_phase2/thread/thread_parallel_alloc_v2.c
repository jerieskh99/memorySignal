/* thread_parallel_alloc_v2  --  benign system-behaviour benchmark
 *
 * N threads each repeatedly malloc a random-sized block, touch it, and free it.
 * Drives allocator (slab/arena) churn and cross-thread interference in the heap.
 *
 * No network, no persistence, no filesystem writes, no sandbox. See
 * docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (spawn) -> measure (alloc/free churn) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <pthread.h>

static const char *TEST = "thread_parallel_alloc_v2";

static volatile int g_run = 1;

typedef struct { int id; uint64_t seed; long min_kb; long max_kb; uint64_t ops; } worker_t;

static void *worker(void *arg) {
    worker_t *w = (worker_t *)arg;
    p2_rng_t rng; p2_rng_seed(&rng, w->seed);
    uint64_t ops = 0;
    long span = w->max_kb - w->min_kb + 1;
    while (g_run) {
        long kb = w->min_kb + (long)(p2_rng_next(&rng) % (uint64_t)span);
        size_t n = (size_t)kb * 1024ULL;
        uint8_t *b = (uint8_t *)malloc(n);
        if (b) {
            /* touch one byte per page so the pages are actually committed */
            for (size_t off = 0; off < n; off += 4096) b[off] = (uint8_t)ops;
            free(b);
            ops++;
        }
    }
    w->ops = ops;
    return NULL;
}

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign parallel-allocation benchmark)\n"
"  --threads N           Worker threads (default = online CPUs, max 256)\n"
"  --min-kb N            Min allocation in KiB (default 1)\n"
"  --max-kb N            Max allocation in KiB (default 256)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu <= 0) ncpu = 4;
    long long threads    = p2_get_i64(argc, argv, "--threads", ncpu);
    long long min_kb     = p2_get_i64(argc, argv, "--min-kb", 1);
    long long max_kb     = p2_get_i64(argc, argv, "--max-kb", 256);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (threads < 1 || threads > 256) { P2_LOG_ERR("threads %lld out of range (1..256)", threads); return 2; }
    if (min_kb < 1 || max_kb < min_kb || max_kb > 1024*1024) {
        P2_LOG_ERR("invalid --min-kb/--max-kb (1 <= min <= max <= 1048576)");
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "THREAD");
    p2_meta_kv_i64(&m, "threads", threads);
    p2_meta_kv_i64(&m, "online_cpus", (long long)ncpu);
    p2_meta_kv_i64(&m, "min_kb", min_kb);
    p2_meta_kv_i64(&m, "max_kb", max_kb);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/filesystem-writes");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }

    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)threads);
    worker_t  *w   = (worker_t *)malloc(sizeof(worker_t) * (size_t)threads);
    if (!tid || !w) { free(tid); free(w); p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    for (long long i = 0; i < threads; i++) {
        w[i].id = (int)i; w[i].seed = seed ^ ((uint64_t)i * 0x9E3779B97F4A7C15ULL);
        w[i].min_kb = (long)min_kb; w[i].max_kb = (long)max_kb; w[i].ops = 0;
        if (pthread_create(&tid[i], NULL, worker, &w[i]) != 0) {
            P2_LOG_ERR("pthread_create failed at %lld", i);
            g_run = 0;
            for (long long j = 0; j < i; j++) pthread_join(tid[j], NULL);
            free(tid); free(w);
            p2_meta_kv_str(&m,"status","thread_create_failed"); p2_meta_close(&m); return 1;
        }
    }
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    while ((p2_monotonic() - t_warm) < (double)duration_s) p2_sleep_ns(50ULL*1000ULL*1000ULL);
    g_run = 0;
    for (long long i = 0; i < threads; i++) pthread_join(tid[i], NULL);
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    uint64_t total_ops = 0;
    for (long long i = 0; i < threads; i++) total_ops += w[i].ops;
    free(tid); free(w);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "alloc_free_ops", total_ops);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "allocator-metadata churn; footprint varies with size distribution");
    p2_meta_close(&m);
    return 0;
}
