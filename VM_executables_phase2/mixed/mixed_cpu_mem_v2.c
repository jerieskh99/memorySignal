/* mixed_cpu_mem_v2  --  benign system-behaviour benchmark
 *
 * Two concurrent threads: one runs a register-resident hash loop (compute), the
 * other writes an anonymous memory buffer at a fixed stride. Compute and memory
 * pressure run together.
 *
 * No network, no persistence, no filesystem writes, no sandbox. See
 * docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc) -> measure (concurrent run) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <pthread.h>

static const char *TEST = "mixed_cpu_mem_v2";

static volatile int g_run = 1;

typedef struct { const uint8_t *buf; size_t bytes; uint64_t h; uint64_t passes; } cpu_arg_t;
typedef struct { volatile uint8_t *buf; size_t bytes; long stride; uint64_t writes; } mem_arg_t;

static void *cpu_worker(void *a) {
    cpu_arg_t *w = (cpu_arg_t *)a;
    uint64_t h = 1469598103934665603ULL, passes = 0;
    while (g_run) {
        for (size_t i = 0; i < w->bytes; i++) { h ^= w->buf[i]; h *= 1099511628211ULL; }
        passes++;
    }
    w->h = h; w->passes = passes;
    return NULL;
}

static void *mem_worker(void *a) {
    mem_arg_t *w = (mem_arg_t *)a;
    uint64_t passes = 0;
    while (g_run) {
        for (size_t off = 0; off < w->bytes; off += (size_t)w->stride) {
            w->buf[off] = (uint8_t)(off ^ passes); w->writes++;
        }
        passes++;
    }
    return NULL;
}

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign concurrent CPU + memory benchmark)\n"
"  --working-set-mb N    Anonymous buffer in MiB (default 512)\n"
"  --cpu-block-kb N      Hash input buffer in KiB (default 4)\n"
"  --stride BYTES        Memory write stride (default 4096)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long ws_mb      = p2_get_i64(argc, argv, "--working-set-mb", 512);
    long long cpu_kb     = p2_get_i64(argc, argv, "--cpu-block-kb", 4);
    long long stride     = p2_get_i64(argc, argv, "--stride", 4096);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (ws_mb <= 0 || ws_mb > 16384) { P2_LOG_ERR("working-set-mb %lld invalid", ws_mb); return 2; }
    if (cpu_kb < 1 || cpu_kb > 65536) { P2_LOG_ERR("cpu-block-kb %lld invalid", cpu_kb); return 2; }
    if (stride < 1 || stride > 64*1024*1024) { P2_LOG_ERR("stride %lld invalid", stride); return 2; }
    size_t mbytes = (size_t)ws_mb * 1024ULL * 1024ULL;
    size_t cbytes = (size_t)cpu_kb * 1024ULL;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "MIXED");
    p2_meta_kv_i64(&m, "working_set_mb", ws_mb);
    p2_meta_kv_i64(&m, "cpu_block_kb", cpu_kb);
    p2_meta_kv_i64(&m, "stride", stride);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/filesystem-writes");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }

    void *buf = mmap(NULL, mbytes, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) { P2_LOG_ERR("mmap: %s", strerror(errno)); p2_meta_kv_str(&m,"status","mmap_failed"); p2_meta_close(&m); return 1; }
    p2_madvise(buf, mbytes, MADV_NOHUGEPAGE);
    uint8_t *cbuf = (uint8_t *)malloc(cbytes);
    if (!cbuf) { munmap(buf, mbytes); p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }
    p2_rng_t rng; p2_rng_seed(&rng, seed); p2_rng_fill(&rng, cbuf, cbytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    memset(buf, 0, mbytes);
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    cpu_arg_t ca = { cbuf, cbytes, 0, 0 };
    mem_arg_t ma = { (volatile uint8_t *)buf, mbytes, (long)stride, 0 };
    pthread_t tc, tmth;
    int okc = pthread_create(&tc, NULL, cpu_worker, &ca) == 0;
    int okm = pthread_create(&tmth, NULL, mem_worker, &ma) == 0;
    if (!okc || !okm) {
        P2_LOG_ERR("pthread_create failed");
        g_run = 0;
        if (okc) pthread_join(tc, NULL);
        if (okm) pthread_join(tmth, NULL);
        free(cbuf); munmap(buf, mbytes);
        p2_meta_kv_str(&m,"status","thread_create_failed"); p2_meta_close(&m); return 1;
    }
    while ((p2_monotonic() - t_warm) < (double)duration_s) p2_sleep_ns(50ULL*1000ULL*1000ULL);
    g_run = 0;
    pthread_join(tc, NULL);
    pthread_join(tmth, NULL);
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    free(cbuf);
    munmap(buf, mbytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "cpu_passes", ca.passes);
    p2_meta_kv_u64(&m, "hash_final", ca.h);
    p2_meta_kv_u64(&m, "mem_writes", ma.writes);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations", "concurrent compute + memory pressure");
    p2_meta_close(&m);
    return 0;
}
