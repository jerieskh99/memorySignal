/* cache_cold_scan_v2  --  benign system-behaviour benchmark
 *
 * Linear scan of a buffer larger than the last-level cache (default 2 GiB) so
 * every access misses cache and goes to DRAM. --mode rmw writes one byte per
 * cache line each pass (maximum dirty-page traffic, sequential); --mode read
 * only reads. The opposite extreme of cache_hot_loop_v2.
 *
 * No network, no persistence, no filesystem writes, no sandbox. See
 * docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (first-touch) -> measure (sequential scan) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "cache_cold_scan_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign large-footprint DRAM benchmark)\n"
"  --working-set-mb N    Buffer size in MiB (default 2048; should exceed LLC)\n"
"  --mode {read|rmw}     Read-only or read-modify-write (default rmw)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --max-mb N            Hard cap on working set (default 16384)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --seed N              PRNG seed (default 42)\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long ws_mb      = p2_get_i64(argc, argv, "--working-set-mb", 2048);
    const char *mode     = p2_get_str(argc, argv, "--mode", "rmw");
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 16384);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    int rmw;
    if      (!strcmp(mode, "read")) rmw = 0;
    else if (!strcmp(mode, "rmw"))  rmw = 1;
    else { P2_LOG_ERR("invalid --mode (read|rmw)"); return 2; }
    if (ws_mb <= 0 || ws_mb > max_mb) {
        P2_LOG_ERR("working-set-mb %lld out of range (1..%lld)", ws_mb, max_mb);
        return 2;
    }
    size_t bytes = (size_t)ws_mb * 1024ULL * 1024ULL;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "CACHE");
    p2_meta_kv_i64(&m, "working_set_mb", ws_mb);
    p2_meta_kv_str(&m, "mode", mode);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    p2_meta_kv_i64(&m, "no_mlock", no_mlock);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/filesystem-writes");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    void *buf = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(buf, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(buf, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    memset(buf, 0, bytes);
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    volatile uint8_t *p = (volatile uint8_t *)buf;
    volatile uint64_t sink = 0;
    uint64_t passes = 0;
    const size_t LINE = 64;
    while ((p2_monotonic() - t_warm) < (double)duration_s) {
        uint8_t k = (uint8_t)(passes & 0xFF);
        if (rmw) for (size_t off = 0; off < bytes; off += LINE) p[off] = (uint8_t)(p[off] ^ k);
        else     for (size_t off = 0; off < bytes; off += LINE) sink += p[off];
        passes++;
    }
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    munmap(buf, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "scan_passes", passes);
    p2_meta_kv_u64(&m, "read_sink", (unsigned long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected high, broad APF in rmw mode; read mode dirties almost nothing");
    p2_meta_close(&m);
    return 0;
}
