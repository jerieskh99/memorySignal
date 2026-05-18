/* mem_pagefault_density_v2
 *
 * Isolates the page-fault density signature from steady-state touch density.
 * Variants:
 *   --variant fault_only : allocate, no MAP_POPULATE, no warm-up; touch
 *                          every page exactly once during measurement.
 *                          Fault path dominates.
 *   --variant touch_only : pre-fault (MAP_POPULATE + memset), then steady
 *                          repeated writes for the duration. No new faults.
 *   --variant mixed      : partial pre-warm (50% of pages), then steady
 *                          writes that gradually fault-in remaining pages.
 *
 * Fixed total page operations across variants is impossible because variant 1
 * cannot revisit; instead we report active page count and revisit ratio in
 * metadata so the analyzer can normalize.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "mem_pagefault_density_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --variant {fault_only|touch_only|mixed}  default fault_only\n"
"  --working-set-mb N        Buffer (default 1024)\n"
"  --duration SEC            (default 60)\n"
"  --seed N                  (default 42)\n"
"  --output-dir PATH\n"
"  --cpu-affinity N\n"
"  --no-mlock / --dry-run / --help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    const char *variant = p2_get_str(argc, argv, "--variant", "fault_only");
    long long ws_mb     = p2_get_i64(argc, argv, "--working-set-mb", 1024);
    long long duration  = p2_get_i64(argc, argv, "--duration", 60);
    long long cpu       = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed      = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock  = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run   = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir  = p2_get_str(argc, argv, "--output-dir", NULL);

    int v;
    if      (!strcmp(variant,"fault_only")) v = 0;
    else if (!strcmp(variant,"touch_only")) v = 1;
    else if (!strcmp(variant,"mixed"))      v = 2;
    else { P2_LOG_ERR("invalid --variant"); return 2; }

    size_t bytes = (size_t)ws_mb * 1024ULL * 1024ULL;
    size_t pages = bytes / 4096ULL;

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"variant",variant);
    p2_meta_kv_i64(&m,"working_set_mb",ws_mb);
    p2_meta_kv_i64(&m,"duration_s",duration);
    p2_meta_kv_u64(&m,"seed",seed);
    p2_meta_kv_u64(&m,"pages",pages);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);

    if (dry_run) { p2_meta_kv_str(&m,"status","dry_run"); p2_meta_close(&m); return 0; }

    if (cpu >= 0) p2_pin_cpu((int)cpu);

    int mflags = MAP_ANONYMOUS|MAP_PRIVATE;
    if (v == 1) mflags |= MAP_POPULATE; /* touch_only pre-fault */
    void *buf = mmap(NULL, bytes, PROT_READ|PROT_WRITE, mflags, -1, 0);
    if (buf == MAP_FAILED) {
        P2_LOG_ERR("mmap: %s", strerror(errno));
        p2_meta_kv_str(&m,"status","mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(buf, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(buf, bytes);

    uint8_t *base = (uint8_t *)buf;

    /* Pre-warm phase (variant-dependent). Excluded from measurement. */
    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    if (v == 1) {
        for (size_t i = 0; i < bytes; i += 4096) base[i] = 0;
    } else if (v == 2) {
        for (size_t i = 0; i < bytes / 2; i += 4096) base[i] = 0;
    }
    double t_wend = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_mstart = t_wend;
    uint64_t writes = 0, new_faults = 0;
    size_t cursor = (v == 2) ? (bytes / 2) : 0;
    uint64_t passes = 0;

    while ((p2_monotonic() - t_mstart) < (double)duration) {
        if (v == 0) {
            /* Touch each page exactly once across the entire measurement window. */
            if (cursor >= bytes) break;
            base[cursor] = (uint8_t)(cursor & 0xFF);
            cursor += 4096;
            writes++; new_faults++;
        } else if (v == 1) {
            /* Steady-state revisits, no new faults. */
            for (size_t off = 0; off < bytes; off += 4096) {
                base[off] = (uint8_t)((off + passes) & 0xFF);
                writes++;
            }
            passes++;
        } else {
            /* Mixed: continue faulting new pages while revisiting prior ones. */
            if (cursor < bytes) {
                base[cursor] = (uint8_t)(cursor & 0xFF);
                cursor += 4096;
                new_faults++; writes++;
            }
            for (size_t off = 0; off < (cursor & ~((size_t)4095)); off += 4096*64) {
                base[off] = (uint8_t)((off + passes) & 0xFF);
                writes++;
            }
            passes++;
        }
    }
    double t_mend = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500000000ULL);
    double t_cend = p2_monotonic();
    munmap(buf, bytes);

    p2_meta_kv_f64(&m,"warmup_t0_s",t0);
    p2_meta_kv_f64(&m,"warmup_end_s",t_wend);
    p2_meta_kv_f64(&m,"measure_end_s",t_mend);
    p2_meta_kv_f64(&m,"cooldown_end_s",t_cend);
    p2_meta_kv_u64(&m,"writes",writes);
    p2_meta_kv_u64(&m,"new_faults_estimate",new_faults);
    p2_meta_kv_u64(&m,"passes",passes);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "fault_only ends when all pages touched; duration is upper bound only");
    p2_meta_close(&m);
    return 0;
}
