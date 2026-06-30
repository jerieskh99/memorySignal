/* thread_lock_contention_v2  --  benign system-behaviour benchmark
 *
 * N threads (default = online CPU count) contend on a single mutex. Under the
 * lock each does a tiny update and writes one shared cache line, stressing the
 * futex path, the scheduler, and inter-CPU cache-line ping-pong.
 *
 * No network, no persistence, no filesystem writes, no sandbox. See
 * docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (spawn) -> measure (contended run) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <pthread.h>

static const char *TEST = "thread_lock_contention_v2";

static volatile int g_run = 1;
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static volatile uint64_t g_shared = 0;   /* the contended cache line */

typedef struct { int id; uint64_t ops; } worker_t;

static void *worker(void *arg) {
    worker_t *w = (worker_t *)arg;
    uint64_t ops = 0;
    while (g_run) {
        pthread_mutex_lock(&g_mtx);
        g_shared += 1;                    /* shared write under the lock */
        pthread_mutex_unlock(&g_mtx);
        ops++;
    }
    w->ops = ops;
    return NULL;
}

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign lock-contention benchmark)\n"
"  --threads N           Worker threads (default = online CPUs, max 256)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --seed N              PRNG seed (unused; recorded for provenance, default 42)\n"
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
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (threads < 1 || threads > 256) { P2_LOG_ERR("threads %lld out of range (1..256)", threads); return 2; }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "THREAD");
    p2_meta_kv_i64(&m, "threads", threads);
    p2_meta_kv_i64(&m, "online_cpus", (long long)ncpu);
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
        w[i].id = (int)i; w[i].ops = 0;
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
    p2_meta_kv_u64(&m, "lock_ops", total_ops);
    p2_meta_kv_u64(&m, "shared_counter", g_shared);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "throughput dominated by futex/scheduler; tiny dirtied footprint");
    p2_meta_close(&m);
    return 0;
}
