/* cache_hot_loop_v2  --  benign system-behaviour benchmark
 *
 * A small buffer (L1/L2-resident, default 32 KiB) is read-modified-written in a
 * tight loop. The data stays in cache and is rarely written back to DRAM, and
 * the footprint is only a handful of pages, so the workload is heavily active
 * yet (near-)invisible to the page-granular APF signal.
 *
 * Purpose: a CONTROL exposing the cache-vs-DRAM observability gap.
 *
 * No network, no persistence, no filesystem writes, no sandbox. See
 * docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (fill) -> measure (RMW loop) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "cache_hot_loop_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign cache-resident control workload)\n"
"  --buffer-kb N         Working buffer in KiB (default 32, cache-resident)\n"
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
    long long buffer_kb  = p2_get_i64(argc, argv, "--buffer-kb", 32);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (buffer_kb < 1 || buffer_kb > 65536) {
        P2_LOG_ERR("buffer-kb %lld out of range (1..65536)", buffer_kb);
        return 2;
    }
    size_t bytes = (size_t)buffer_kb * 1024ULL;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "CACHE");
    p2_meta_kv_i64(&m, "buffer_kb", buffer_kb);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/filesystem-writes");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    uint8_t *buf = (uint8_t *)malloc(bytes);
    if (!buf) {
        P2_LOG_ERR("malloc(%zu) failed", bytes);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    p2_rng_fill(&rng, buf, bytes);
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    volatile uint8_t *p = (volatile uint8_t *)buf;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_warm) < (double)duration_s) {
        uint8_t k = (uint8_t)(passes & 0xFF);
        for (size_t i = 0; i < bytes; i++) p[i] = (uint8_t)(p[i] ^ k);  /* RMW, cache-hot */
        passes++;
    }
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    free(buf);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "rmw_passes", passes);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "cache-resident; expected low APF despite heavy activity (small footprint)");
    p2_meta_close(&m);
    return 0;
}
