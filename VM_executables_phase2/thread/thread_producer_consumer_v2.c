/* thread_producer_consumer_v2  --  benign system-behaviour benchmark
 *
 * One producer and one consumer share a bounded ring buffer guarded by a mutex
 * and two condition variables (not-full / not-empty). Stresses queue
 * synchronisation and cross-CPU traffic between the two threads.
 *
 * No network, no persistence, no filesystem writes, no sandbox. See
 * docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (spawn) -> measure (run) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <pthread.h>

static const char *TEST = "thread_producer_consumer_v2";

static volatile int g_run = 1;
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_not_full  = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_not_empty = PTHREAD_COND_INITIALIZER;

static uint64_t *g_ring = NULL;
static long      g_ring_size = 0;
static long      g_head = 0, g_tail = 0, g_count = 0;
static uint64_t  g_produced = 0, g_consumed = 0;

static void *producer(void *arg) {
    (void)arg;
    uint64_t v = 0;
    while (g_run) {
        pthread_mutex_lock(&g_mtx);
        while (g_count == g_ring_size && g_run) pthread_cond_wait(&g_not_full, &g_mtx);
        if (!g_run) { pthread_mutex_unlock(&g_mtx); break; }
        g_ring[g_tail] = v++;
        g_tail = (g_tail + 1) % g_ring_size;
        g_count++; g_produced++;
        pthread_cond_signal(&g_not_empty);
        pthread_mutex_unlock(&g_mtx);
    }
    return NULL;
}

static void *consumer(void *arg) {
    (void)arg;
    volatile uint64_t sink = 0;
    while (1) {
        pthread_mutex_lock(&g_mtx);
        while (g_count == 0 && g_run) pthread_cond_wait(&g_not_empty, &g_mtx);
        if (g_count == 0 && !g_run) { pthread_mutex_unlock(&g_mtx); break; }
        sink += g_ring[g_head];
        g_head = (g_head + 1) % g_ring_size;
        g_count--; g_consumed++;
        pthread_cond_signal(&g_not_full);
        pthread_mutex_unlock(&g_mtx);
    }
    (void)sink;
    return NULL;
}

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign producer/consumer benchmark)\n"
"  --ring-size N         Ring-buffer slots (default 1024, max 1048576)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --seed N              PRNG seed (recorded for provenance, default 42)\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long ring       = p2_get_i64(argc, argv, "--ring-size", 1024);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (ring < 1 || ring > 1024*1024) { P2_LOG_ERR("ring-size %lld out of range (1..1048576)", ring); return 2; }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "THREAD");
    p2_meta_kv_i64(&m, "ring_size", ring);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/filesystem-writes");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }

    g_ring_size = (long)ring;
    g_ring = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)g_ring_size);
    if (!g_ring) { p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    pthread_t tp, tc;
    if (pthread_create(&tp, NULL, producer, NULL) != 0 ||
        pthread_create(&tc, NULL, consumer, NULL) != 0) {
        P2_LOG_ERR("pthread_create failed");
        g_run = 0;
        pthread_cond_broadcast(&g_not_full); pthread_cond_broadcast(&g_not_empty);
        free(g_ring);
        p2_meta_kv_str(&m,"status","thread_create_failed"); p2_meta_close(&m); return 1;
    }
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    while ((p2_monotonic() - t_warm) < (double)duration_s) p2_sleep_ns(50ULL*1000ULL*1000ULL);
    pthread_mutex_lock(&g_mtx);
    g_run = 0;
    pthread_cond_broadcast(&g_not_full);
    pthread_cond_broadcast(&g_not_empty);
    pthread_mutex_unlock(&g_mtx);
    pthread_join(tp, NULL);
    pthread_join(tc, NULL);
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    free(g_ring);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "produced", g_produced);
    p2_meta_kv_u64(&m, "consumed", g_consumed);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "condvar wakeups dominate; small dirtied footprint (the ring)");
    p2_meta_close(&m);
    return 0;
}
