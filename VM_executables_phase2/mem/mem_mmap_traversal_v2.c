/* mem_mmap_traversal_v2
 *
 * Memory-mapped file traversal vs anonymous-mapping semantics.
 * Variants:
 *   --variant read   : MAP_SHARED file, read every page (no writes)
 *   --variant write  : MAP_SHARED file, write every page each pass
 *   --variant rmw    : MAP_SHARED file, read+xor+write each page
 *
 * --msync-interval-ms N triggers MS_ASYNC msync at the given cadence for
 * the write/rmw variants so writeback rhythm becomes a configurable feature.
 *
 * The backing file is created inside --backing-dir (default /tmp) and removed
 * unless --keep-backing is set. Sandbox validation applies.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "mem_mmap_traversal_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --variant {read|write|rmw}     (default write)\n"
"  --file-size-mb N               (default 1024)\n"
"  --duration SEC                 (default 60)\n"
"  --msync-interval-ms N          0 = disable (default 0)\n"
"  --backing-dir PATH             (default /tmp)\n"
"  --keep-backing                 keep backing file after exit\n"
"  --seed N                       (default 42)\n"
"  --output-dir PATH\n"
"  --cpu-affinity N\n"
"  --dry-run / --help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    const char *variant   = p2_get_str(argc, argv, "--variant", "write");
    long long size_mb     = p2_get_i64(argc, argv, "--file-size-mb", 1024);
    long long duration    = p2_get_i64(argc, argv, "--duration", 60);
    long long msync_ms    = p2_get_i64(argc, argv, "--msync-interval-ms", 0);
    long long cpu         = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed        = p2_get_u64(argc, argv, "--seed", 42);
    int       keep        = p2_flag_present(argc, argv, "--keep-backing");
    int       dry_run     = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir    = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *backdir   = p2_get_str(argc, argv, "--backing-dir", "/tmp");

    int v;
    if      (!strcmp(variant,"read"))  v = 0;
    else if (!strcmp(variant,"write")) v = 1;
    else if (!strcmp(variant,"rmw"))   v = 2;
    else { P2_LOG_ERR("invalid --variant"); return 2; }

    size_t bytes = (size_t)size_mb * 1024ULL * 1024ULL;
    char backing_path[PATH_MAX];
    snprintf(backing_path, sizeof(backing_path),
             "%s/phase2_mmap_%d_%llu.dat", backdir, (int)getpid(),
             (unsigned long long)seed);

    char backing_real[PATH_MAX];
    if (p2_sandbox_validate(backing_path, backdir, backing_real) != 0) {
        P2_LOG_ERR("backing path rejected by sandbox validator");
        return 3;
    }

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"variant",variant);
    p2_meta_kv_i64(&m,"file_size_mb",size_mb);
    p2_meta_kv_i64(&m,"duration_s",duration);
    p2_meta_kv_i64(&m,"msync_interval_ms",msync_ms);
    p2_meta_kv_str(&m,"backing_path",backing_real);
    p2_meta_kv_u64(&m,"seed",seed);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);

    if (dry_run) { p2_meta_kv_str(&m,"status","dry_run"); p2_meta_close(&m); return 0; }

    if (cpu >= 0) p2_pin_cpu((int)cpu);

    int fd = open(backing_real, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        P2_LOG_ERR("open(%s): %s", backing_real, strerror(errno));
        p2_meta_kv_str(&m,"status","open_failed"); p2_meta_close(&m); return 1;
    }
    if (ftruncate(fd, (off_t)bytes) != 0) {
        P2_LOG_ERR("ftruncate: %s", strerror(errno));
        close(fd); unlink(backing_real);
        p2_meta_kv_str(&m,"status","ftruncate_failed"); p2_meta_close(&m); return 1;
    }
    /* Pre-fill with deterministic bytes so reads return useful content. */
    {
        p2_rng_t rng; p2_rng_seed(&rng, seed);
        uint8_t chunk[65536];
        size_t remaining = bytes;
        while (remaining > 0) {
            size_t n = remaining > sizeof(chunk) ? sizeof(chunk) : remaining;
            p2_rng_fill(&rng, chunk, n);
            ssize_t w = write(fd, chunk, n);
            if (w <= 0) break;
            remaining -= (size_t)w;
        }
        fsync(fd);
    }

    void *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        P2_LOG_ERR("mmap MAP_SHARED: %s", strerror(errno));
        close(fd); if (!keep) unlink(backing_real);
        p2_meta_kv_str(&m,"status","mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(map, bytes, MADV_NOHUGEPAGE);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* No memset on warmup — we want first-touch behavior to be a measurable
       feature in this test (especially for write/rmw variants). */
    double t_wend = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_mstart = t_wend;
    double t_next_msync = (msync_ms > 0) ?
        (t_mstart + (double)msync_ms / 1000.0) : 0.0;
    volatile uint8_t *p = (volatile uint8_t *)map;
    volatile uint8_t sink = 0;
    uint64_t ops = 0, passes = 0, msync_calls = 0;
    while ((p2_monotonic() - t_mstart) < (double)duration) {
        uint8_t k = (uint8_t)(passes & 0xFF);
        for (size_t off = 0; off < bytes; off += 4096) {
            if (v == 0)      sink ^= p[off];
            else if (v == 1) p[off] = k;
            else             p[off] = (uint8_t)(p[off] ^ k);
            ops++;
        }
        passes++;
        if (msync_ms > 0 && v != 0 && p2_monotonic() >= t_next_msync) {
            msync((void *)map, bytes, MS_ASYNC);
            msync_calls++;
            t_next_msync += (double)msync_ms / 1000.0;
        }
    }
    double t_mend = p2_monotonic();

    p2_phase(TEST, "cooldown");
    if (v != 0) msync((void *)map, bytes, MS_SYNC);
    p2_sleep_ns(500000000ULL);
    double t_cend = p2_monotonic();

    munmap(map, bytes);
    close(fd);
    if (!keep) {
        if (unlink(backing_real) != 0) P2_LOG_WARN("unlink failed: %s", strerror(errno));
    }

    p2_meta_kv_f64(&m,"warmup_t0_s",t0);
    p2_meta_kv_f64(&m,"warmup_end_s",t_wend);
    p2_meta_kv_f64(&m,"measure_end_s",t_mend);
    p2_meta_kv_f64(&m,"cooldown_end_s",t_cend);
    p2_meta_kv_u64(&m,"passes",passes);
    p2_meta_kv_u64(&m,"ops",ops);
    p2_meta_kv_u64(&m,"msync_calls",msync_calls);
    p2_meta_kv_i64(&m,"sink_val",(long long)sink);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "writeback timing depends on dirty_ratio/writeback_centisecs in guest");
    p2_meta_close(&m);
    return 0;
}
