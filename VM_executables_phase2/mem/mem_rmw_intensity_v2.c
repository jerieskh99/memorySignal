/* mem_rmw_intensity_v2
 *
 * Isolates the read-modify-write signature from pure-read and pure-write.
 * --mode selects:
 *    write   : pure write, byte = (uint8_t)pass
 *    read    : pure read, accumulate into volatile sink
 *    rmw     : read-modify-write, byte = byte ^ src[pg]
 *
 * Same total ops per variant (fixed by --duration), same stride and footprint
 * so confusion / separability is between mechanisms, not work amount.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "mem_rmw_intensity_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --mode {write|read|rmw}   Variant (default rmw)\n"
"  --working-set-mb N        Buffer (default 1024)\n"
"  --stride BYTES            (default 4096)\n"
"  --duration SEC            (default 60)\n"
"  --warmup SEC              (default 5)\n"
"  --seed N                  (default 42)\n"
"  --output-dir PATH\n"
"  --cpu-affinity N\n"
"  --no-mlock / --dry-run / --help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    const char *mode    = p2_get_str(argc, argv, "--mode", "rmw");
    long long ws_mb     = p2_get_i64(argc, argv, "--working-set-mb", 1024);
    long long stride    = p2_get_i64(argc, argv, "--stride", 4096);
    long long duration  = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup    = p2_get_i64(argc, argv, "--warmup", 5);
    long long cpu       = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed      = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock  = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run   = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir  = p2_get_str(argc, argv, "--output-dir", NULL);

    int op;  /* 0=write, 1=read, 2=rmw */
    if      (!strcmp(mode,"write")) op = 0;
    else if (!strcmp(mode,"read"))  op = 1;
    else if (!strcmp(mode,"rmw"))   op = 2;
    else { P2_LOG_ERR("invalid --mode '%s'", mode); return 2; }

    size_t bytes = (size_t)ws_mb * 1024ULL * 1024ULL;

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"mode",mode);
    p2_meta_kv_i64(&m,"working_set_mb",ws_mb);
    p2_meta_kv_i64(&m,"stride",stride);
    p2_meta_kv_i64(&m,"duration_s",duration);
    p2_meta_kv_i64(&m,"warmup_s",warmup);
    p2_meta_kv_u64(&m,"seed",seed);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);

    if (dry_run) { p2_meta_kv_str(&m,"status","dry_run"); p2_meta_close(&m); return 0; }

    if (cpu >= 0) p2_pin_cpu((int)cpu);

    void *buf = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                     MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) { P2_LOG_ERR("mmap: %s", strerror(errno));
        p2_meta_kv_str(&m,"status","mmap_failed"); p2_meta_close(&m); return 1; }
    p2_madvise(buf, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(buf, bytes);

    /* Initialize buffer to a nonzero pattern (so reads return useful bytes). */
    p2_rng_t rng; p2_rng_seed(&rng, seed);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    {
        uint8_t *bp = (uint8_t *)buf;
        for (size_t i = 0; i < bytes; i += 4096) bp[i] = (uint8_t)(i & 0xFF);
    }
    double t_wend = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_mstart = t_wend;
    volatile uint8_t *p = (volatile uint8_t *)buf;
    volatile uint8_t sink = 0;
    uint64_t passes = 0, ops = 0;
    while ((p2_monotonic() - t_mstart) < (double)duration) {
        uint8_t k = (uint8_t)(passes & 0xFF);
        for (size_t off = 0; off < bytes; off += (size_t)stride) {
            if (op == 0) {
                p[off] = k;
            } else if (op == 1) {
                sink ^= p[off];
            } else { /* rmw */
                p[off] = (uint8_t)(p[off] ^ k);
            }
            ops++;
        }
        passes++;
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
    p2_meta_kv_u64(&m,"passes",passes);
    p2_meta_kv_u64(&m,"ops",ops);
    p2_meta_kv_i64(&m,"sink_val",(long long)sink);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "RMW may be indistinguishable from pure-write at signal level (pre-registered null)");
    p2_meta_close(&m);
    return 0;
}
