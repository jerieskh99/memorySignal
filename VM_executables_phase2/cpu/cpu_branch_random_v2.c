/* cpu_branch_random_v2  --  benign system-behaviour benchmark
 *
 * Tight loop whose control flow is driven by a precomputed table of random
 * bits, stressing the branch predictor. The table is read-only after warmup and
 * the accumulator is register-resident, so memory writes are minimal.
 *
 * Purpose: a near-idle CONTROL with a distinct microarchitectural profile
 * (branch mispredictions), expected to look quiet on the APF signal.
 *
 * No network, no persistence, no filesystem writes, no sandbox. See
 * docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (fill table) -> measure (branchy loop) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "cpu_branch_random_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign CPU-bound control workload)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --table-kb N          Random branch table in KiB (default 64)\n"
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
    long long table_kb   = p2_get_i64(argc, argv, "--table-kb", 64);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (table_kb < 1 || table_kb > 1024*1024) {
        P2_LOG_ERR("table-kb %lld out of range (1..1048576)", table_kb);
        return 2;
    }
    size_t tsize = (size_t)table_kb * 1024ULL;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "CPU");
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "table_kb", table_kb);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/filesystem-writes");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    uint8_t *table = (uint8_t *)malloc(tsize);
    if (!table) {
        P2_LOG_ERR("malloc(%zu) failed", tsize);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    p2_rng_fill(&rng, table, tsize);   /* each byte's low bit drives a branch */
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    uint64_t taken = 0, nottaken = 0, iters = 0;
    while ((p2_monotonic() - t_warm) < (double)duration_s) {
        for (size_t i = 0; i < tsize; i++) {
            if (table[i] & 1u) taken++;     /* data-dependent, unpredictable branch */
            else               nottaken++;
        }
        iters++;
    }
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    free(table);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "table_passes", iters);
    p2_meta_kv_u64(&m, "branches_taken", taken);
    p2_meta_kv_u64(&m, "branches_nottaken", nottaken);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "branch-predictor stress; expected near-idle APF (table read-only)");
    p2_meta_close(&m);
    return 0;
}
