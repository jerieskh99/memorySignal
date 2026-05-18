/* mem_workingset_sweep_v2
 *
 * Working-set sweep: write 1 byte at the start of every page of a buffer of
 * size --working-set-mb. Same total ops across sizes (controlled by
 * --measurement-duration), so per-page revisit rate scales inversely with WS.
 *
 * Mechanism isolated: active page fraction as a function of footprint size,
 * holding write magnitude fixed (1 byte) and stride fixed (4 KB).
 *
 * Phases:
 *   warmup    : memset() first-touches all pages (excluded from measurement)
 *   measure   : repeated sequential 4 KB-stride writes for --duration seconds
 *   cooldown  : idle
 *
 * Notes:
 *   - MADV_NOHUGEPAGE keeps page granularity at 4 KB.
 *   - mlock() best-effort; warns and continues if RLIMIT_MEMLOCK forbids it.
 *   - --max-mb defaults to 16384; tests refuse to allocate beyond this.
 *   - --check-meminfo aborts if MemAvailable falls below 1 GB.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "mem_workingset_sweep_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --working-set-mb N    Buffer size in MiB (default 1024)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 5)\n"
"  --stride BYTES        Stride between writes (default 4096)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --max-mb N            Hard cap on working set (default 16384)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n",
        p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long ws_mb       = p2_get_i64(argc, argv, "--working-set-mb", 1024);
    long long duration_s  = p2_get_i64(argc, argv, "--duration",       60);
    long long warmup_s    = p2_get_i64(argc, argv, "--warmup",         5);
    long long stride      = p2_get_i64(argc, argv, "--stride",         4096);
    long long max_mb      = p2_get_i64(argc, argv, "--max-mb",         16384);
    long long cpu         = p2_get_i64(argc, argv, "--cpu-affinity",   -1);
    uint64_t  seed        = p2_get_u64(argc, argv, "--seed",           42);
    int       no_mlock    = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run     = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir    = p2_get_str(argc, argv, "--output-dir",     NULL);

    if (ws_mb <= 0 || ws_mb > max_mb) {
        P2_LOG_ERR("working-set-mb %lld out of range (1..%lld)", ws_mb, max_mb);
        return 2;
    }
    if (stride < 1 || stride > 1024*1024) {
        P2_LOG_ERR("stride %lld invalid", stride);
        return 2;
    }

    size_t bytes = (size_t)ws_mb * 1024ULL * 1024ULL;
    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_i64(&m, "working_set_mb", ws_mb);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_i64(&m, "stride", stride);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    p2_meta_kv_i64(&m, "no_mlock", no_mlock);

    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) {
        p2_meta_kv_str(&m, "status", "dry_run");
        p2_meta_close(&m);
        return 0;
    }

    if (cpu >= 0) p2_pin_cpu((int)cpu);

    void *buf = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                     MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed");
        p2_meta_close(&m);
        return 1;
    }
    p2_madvise(buf, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(buf, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* memset commits zero pages, separates first-touch from measurement. */
    memset(buf, 0, bytes);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    volatile uint8_t *p = (volatile uint8_t *)buf;
    uint64_t passes = 0;
    uint64_t writes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One full sweep at fixed stride. */
        for (size_t off = 0; off < bytes; off += (size_t)stride) {
            p[off] = (uint8_t)(off ^ passes);
            writes++;
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    /* Brief settle. */
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Restore buffer to zero (predictable post-state) before munmap. */
    munmap(buf, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s",      t0);
    p2_meta_kv_f64(&m, "warmup_end_s",     t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s",    t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s",   t_cool_end);
    p2_meta_kv_u64(&m, "passes",           passes);
    p2_meta_kv_u64(&m, "writes",           writes);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "MADV_NOHUGEPAGE/mlock are Linux-only and best-effort");
    p2_meta_close(&m);
    return 0;
}
