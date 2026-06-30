/* cpu_hash_loop_v2  --  benign system-behaviour benchmark
 *
 * Tight, register-resident FNV-1a hash over a small in-cache input buffer.
 * Compute-bound: the hash state lives in registers and the input buffer is
 * read-only after warmup, so the workload writes (almost) no memory pages.
 *
 * Purpose: a CONTROL. It tests whether pure computation is (near-)invisible to
 * the host APF signal -- expected memory pattern is close to idle, because no
 * pages change between snapshots.
 *
 * No network, no persistence, no filesystem writes, no sandbox. See
 * docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (fill input) -> measure (hash loop) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "cpu_hash_loop_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign CPU-bound control workload)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --block-kb N          In-cache input buffer in KiB (default 4, L1-resident)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long block_kb   = p2_get_i64(argc, argv, "--block-kb", 4);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (block_kb < 1 || block_kb > 1024*1024) {
        P2_LOG_ERR("block-kb %lld out of range (1..1048576)", block_kb);
        return 2;
    }
    size_t block = (size_t)block_kb * 1024ULL;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "CPU");
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "block_kb", block_kb);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/filesystem-writes");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    uint8_t *buf = (uint8_t *)malloc(block);
    if (!buf) {
        P2_LOG_ERR("malloc(%zu) failed", block);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    p2_rng_fill(&rng, buf, block);
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    /* FNV-1a 64-bit, register-resident; the input buffer is only read. */
    uint64_t h = 1469598103934665603ULL;
    uint64_t iters = 0;
    while ((p2_monotonic() - t_warm) < (double)duration_s) {
        for (size_t i = 0; i < block; i++) {
            h ^= buf[i];
            h *= 1099511628211ULL;
        }
        iters++;
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
    p2_meta_kv_u64(&m, "hash_passes", iters);
    p2_meta_kv_u64(&m, "hash_final", h);     /* keep the loop from being optimised away */
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "compute-bound control; expected near-idle APF (no page writes)");
    p2_meta_close(&m);
    return 0;
}
