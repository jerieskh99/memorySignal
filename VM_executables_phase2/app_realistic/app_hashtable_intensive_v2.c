/* app_hashtable_intensive_v2
 *
 * Open-addressing hash table with linear probing. Phase 1 builds the table by
 * inserting --inserts entries; phase 2 issues --lookups probes. Two-phase
 * trajectory is the feature: build is write-dominant, probe is read-dominant.
 *
 * Capacity is a power of two; load factor < 0.5 at end of build. Hash is a
 * mixing function over the key (xorshift-based) — deterministic per --seed.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "app_hashtable_intensive_v2";

typedef struct {
    uint64_t key;
    uint64_t val;
} entry_t;

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --capacity-pow2 N    Table capacity = 2^N entries (default 24 = 16M entries = ~256 MiB)\n"
"  --inserts N          Number of inserts during build (default capacity*0.4)\n"
"  --lookups N          Number of lookups during probe (default 10000000)\n"
"  --duration SEC       Hard cap on each phase (default 120)\n"
"  --seed N             (default 42)\n"
"  --output-dir PATH\n"
"  --cpu-affinity N\n"
"  --no-mlock / --dry-run / --help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long cap_pow  = p2_get_i64(argc, argv, "--capacity-pow2", 24);
    long long inserts  = p2_get_i64(argc, argv, "--inserts", -1);
    long long lookups  = p2_get_i64(argc, argv, "--lookups", 10000000);
    long long duration = p2_get_i64(argc, argv, "--duration", 120);
    long long cpu      = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed     = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run  = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir = p2_get_str(argc, argv, "--output-dir", NULL);

    if (cap_pow < 10 || cap_pow > 30) { P2_LOG_ERR("capacity-pow2 range 10..30"); return 2; }
    uint64_t capacity = 1ULL << cap_pow;
    uint64_t mask = capacity - 1;
    if (inserts < 0) inserts = (long long)(capacity * 4 / 10);
    if ((uint64_t)inserts > capacity / 2) {
        P2_LOG_ERR("inserts exceed half capacity (load factor > 0.5)");
        return 2;
    }
    size_t bytes = capacity * sizeof(entry_t);

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"family","APP-REALISTIC");
    p2_meta_kv_i64(&m,"capacity_pow2",cap_pow);
    p2_meta_kv_u64(&m,"capacity",capacity);
    p2_meta_kv_i64(&m,"inserts",inserts);
    p2_meta_kv_i64(&m,"lookups",lookups);
    p2_meta_kv_i64(&m,"duration_cap_s",duration);
    p2_meta_kv_u64(&m,"seed",seed);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);
    if (dry_run) { p2_meta_kv_str(&m,"status","dry_run"); p2_meta_close(&m); return 0; }

    if (cpu >= 0) p2_pin_cpu((int)cpu);

    entry_t *tbl = (entry_t *)mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                                   MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
    if (tbl == MAP_FAILED) {
        P2_LOG_ERR("mmap failed: %s", strerror(errno));
        p2_meta_kv_str(&m,"status","mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(tbl, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(tbl, bytes);

    /* Initialize table to a sentinel (zero) — pages will fault as we touch them. */

    p2_phase(TEST, "build");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    uint64_t inserted = 0, build_probes = 0;
    while ((long long)inserted < inserts &&
           (p2_monotonic() - t0) < (double)duration) {
        uint64_t k = p2_rng_next(&rng);
        if (k == 0) k = 1; /* reserve zero as empty marker */
        uint64_t h = mix64(k) & mask;
        for (;;) {
            build_probes++;
            if (tbl[h].key == 0) {
                tbl[h].key = k;
                tbl[h].val = mix64(k ^ 0xDEADBEEFULL);
                inserted++;
                break;
            }
            if (tbl[h].key == k) break;
            h = (h + 1) & mask;
        }
    }
    double t_build_end = p2_monotonic();

    p2_phase(TEST, "probe");
    double t_p_start = t_build_end;
    p2_rng_t pr; p2_rng_seed(&pr, seed); /* same stream → mostly hits */
    uint64_t hits = 0, misses = 0, probes = 0;
    volatile uint64_t sink = 0;
    long long done = 0;
    while (done < lookups && (p2_monotonic() - t_p_start) < (double)duration) {
        uint64_t k = p2_rng_next(&pr);
        if (k == 0) k = 1;
        /* 20% miss probes: flip a bit so the key is not in the build stream */
        if ((p2_rng_next(&pr) & 0x7) == 0) k ^= 0x1ULL << 33;
        uint64_t h = mix64(k) & mask;
        for (;;) {
            probes++;
            if (tbl[h].key == 0) { misses++; break; }
            if (tbl[h].key == k) { sink ^= tbl[h].val; hits++; break; }
            h = (h + 1) & mask;
        }
        done++;
    }
    double t_p_end = p2_monotonic();

    munmap(tbl, bytes);
    p2_meta_kv_f64(&m,"build_start_s",t0);
    p2_meta_kv_f64(&m,"build_end_s",t_build_end);
    p2_meta_kv_f64(&m,"probe_end_s",t_p_end);
    p2_meta_kv_u64(&m,"inserted",inserted);
    p2_meta_kv_u64(&m,"build_probes",build_probes);
    p2_meta_kv_u64(&m,"lookups_done",done);
    p2_meta_kv_u64(&m,"hits",hits);
    p2_meta_kv_u64(&m,"misses",misses);
    p2_meta_kv_u64(&m,"probes",probes);
    p2_meta_kv_i64(&m,"sink",(long long)sink);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "Linear probing chooses cache locality; chaining variant would alter signal");
    p2_meta_close(&m);
    return 0;
}
