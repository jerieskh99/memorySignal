/* io_read_cache_hit_v2  --  benign system-behaviour benchmark
 *
 * Creates a backing file smaller than the page cache, pre-warms it (one full
 * read so it is cache-resident), then issues many random reads that all hit the
 * page cache. High syscall rate, but almost no pages are dirtied -- a "quiet
 * I/O" calibration probe expected to look near-idle on the APF signal.
 *
 * File I/O is confined to a sandbox-validated backing file under /tmp or
 * /var/tmp; no network, no persistence, no encryption. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (create + prewarm) -> measure (random cache-hot reads) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "io_read_cache_hit_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign cache-hot read benchmark)\n"
"  --file-size-mb N      Backing file size in MiB (default 256; keep < RAM)\n"
"  --block-bytes N       Read block size (default 4096)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --backing-dir PATH    Sandbox dir for the backing file (default /tmp)\n"
"  --safe-root PATH      Extra approved root for sandbox validation\n"
"  --keep-backing        Keep the backing file after exit (default: remove)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long size_mb    = p2_get_i64(argc, argv, "--file-size-mb", 256);
    long long block      = p2_get_i64(argc, argv, "--block-bytes", 4096);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       keep       = p2_flag_present(argc, argv, "--keep-backing");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *backdir  = p2_get_str(argc, argv, "--backing-dir", "/tmp");
    const char *saferoot = p2_get_str(argc, argv, "--safe-root", NULL);

    if (size_mb <= 0 || size_mb > 65536) { P2_LOG_ERR("file-size-mb %lld invalid", size_mb); return 2; }
    if (block < 64 || block > 16*1024*1024) { P2_LOG_ERR("block-bytes %lld invalid", block); return 2; }
    size_t bytes = (size_t)size_mb * 1024ULL * 1024ULL;

    char backing_path[PATH_MAX];
    snprintf(backing_path, sizeof(backing_path), "%s/phase2_io_%d_%llu.dat",
             backdir, (int)getpid(), (unsigned long long)seed);
    char backing_real[PATH_MAX];
    if (p2_sandbox_validate(backing_path, saferoot, backing_real) != 0) {
        P2_LOG_ERR("backing path rejected by sandbox validator");
        return 3;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "IO");
    p2_meta_kv_i64(&m, "file_size_mb", size_mb);
    p2_meta_kv_i64(&m, "block_bytes", block);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_str(&m, "backing_path", backing_real);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; sandbox-validated backing file; no network/persistence");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    int fd = open(backing_real, O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
    if (fd < 0) {
        P2_LOG_ERR("open(%s): %s", backing_real, strerror(errno));
        p2_meta_kv_str(&m, "status", "open_failed"); p2_meta_close(&m); return 1;
    }

    uint8_t *blk = (uint8_t *)malloc((size_t)block);
    if (!blk) { close(fd); unlink(backing_real); p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Create the file with deterministic content. */
    {
        p2_rng_t rng; p2_rng_seed(&rng, seed);
        size_t remaining = bytes;
        while (remaining > 0) {
            size_t n = remaining > (size_t)block ? (size_t)block : remaining;
            p2_rng_fill(&rng, blk, n);
            ssize_t w = write(fd, blk, n);
            if (w <= 0) { P2_LOG_ERR("write: %s", strerror(errno)); break; }
            remaining -= (size_t)w;
        }
        fsync(fd);
        /* Pre-warm: one full sequential read so the file is page-cache hot. */
        lseek(fd, 0, SEEK_SET);
        while (read(fd, blk, (size_t)block) > 0) { /* warm cache */ }
    }
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    p2_rng_t rng; p2_rng_seed(&rng, seed ^ 0x9E37);
    size_t nblocks = bytes / (size_t)block; if (nblocks == 0) nblocks = 1;
    uint64_t reads = 0; volatile uint64_t sink = 0;
    while ((p2_monotonic() - t_warm) < (double)duration_s) {
        for (int b = 0; b < 1024; b++) {
            off_t off = (off_t)((p2_rng_next(&rng) % nblocks) * (size_t)block);
            ssize_t r = pread(fd, blk, (size_t)block, off);
            if (r > 0) { sink += blk[0]; reads++; }
        }
    }
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    free(blk);
    close(fd);
    if (!keep && unlink(backing_real) != 0) P2_LOG_WARN("unlink failed: %s", strerror(errno));

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "reads", reads);
    p2_meta_kv_u64(&m, "read_sink", (unsigned long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "cache-hot reads dirty almost nothing; expected near-idle APF, high syscall rate");
    p2_meta_close(&m);
    return 0;
}
