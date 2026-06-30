/* mixed_cpu_io_v2  --  benign system-behaviour benchmark
 *
 * Two concurrent threads: one runs a register-resident hash loop (compute), the
 * other writes blocks to a sandbox-validated backing file. Stresses the
 * scheduler and I/O subsystem under CPU load.
 *
 * File I/O is confined to a sandbox-validated backing file under /tmp or
 * /var/tmp; no network, no persistence, no encryption. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + create) -> measure (concurrent run) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <pthread.h>

static const char *TEST = "mixed_cpu_io_v2";

static volatile int g_run = 1;

typedef struct { const uint8_t *buf; size_t bytes; uint64_t h; uint64_t passes; } cpu_arg_t;
typedef struct { int fd; size_t fsize; size_t block; uint8_t *blk; uint64_t writes; } io_arg_t;

static void *cpu_worker(void *a) {
    cpu_arg_t *w = (cpu_arg_t *)a;
    uint64_t h = 1469598103934665603ULL, passes = 0;
    while (g_run) {
        for (size_t i = 0; i < w->bytes; i++) { h ^= w->buf[i]; h *= 1099511628211ULL; }
        passes++;
    }
    w->h = h; w->passes = passes;
    return NULL;
}

static void *io_worker(void *a) {
    io_arg_t *w = (io_arg_t *)a;
    size_t nblocks = w->fsize / w->block; if (nblocks == 0) nblocks = 1;
    size_t cur = 0;
    while (g_run) {
        w->blk[0] = (uint8_t)w->writes;
        off_t off = (off_t)((cur++ % nblocks) * w->block);
        if (pwrite(w->fd, w->blk, w->block, off) > 0) w->writes++;
    }
    return NULL;
}

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign concurrent CPU + file-I/O benchmark)\n"
"  --file-size-mb N      Backing file size in MiB (default 256)\n"
"  --block-bytes N       I/O block size (default 65536)\n"
"  --cpu-block-kb N      Hash input buffer in KiB (default 4)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --backing-dir PATH    Sandbox dir for the backing file (default /tmp)\n"
"  --safe-root PATH      Extra approved root for sandbox validation\n"
"  --keep-backing        Keep the backing file after exit (default: remove)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long size_mb    = p2_get_i64(argc, argv, "--file-size-mb", 256);
    long long block      = p2_get_i64(argc, argv, "--block-bytes", 65536);
    long long cpu_kb     = p2_get_i64(argc, argv, "--cpu-block-kb", 4);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       keep       = p2_flag_present(argc, argv, "--keep-backing");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *backdir  = p2_get_str(argc, argv, "--backing-dir", "/tmp");
    const char *saferoot = p2_get_str(argc, argv, "--safe-root", NULL);

    if (size_mb <= 0 || size_mb > 65536) { P2_LOG_ERR("file-size-mb %lld invalid", size_mb); return 2; }
    if (block < 512 || block > 16*1024*1024) { P2_LOG_ERR("block-bytes %lld invalid", block); return 2; }
    if (cpu_kb < 1 || cpu_kb > 65536) { P2_LOG_ERR("cpu-block-kb %lld invalid", cpu_kb); return 2; }
    size_t fbytes = (size_t)size_mb * 1024ULL * 1024ULL;
    size_t cbytes = (size_t)cpu_kb * 1024ULL;

    char backing_path[PATH_MAX];
    snprintf(backing_path, sizeof(backing_path), "%s/phase2_mixedio_%d_%llu.dat",
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
    p2_meta_kv_str(&m, "behavior_family", "MIXED");
    p2_meta_kv_i64(&m, "file_size_mb", size_mb);
    p2_meta_kv_i64(&m, "block_bytes", block);
    p2_meta_kv_i64(&m, "cpu_block_kb", cpu_kb);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_str(&m, "backing_path", backing_real);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; sandbox-validated backing file; no network/persistence");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }

    int fd = open(backing_real, O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
    if (fd < 0) { P2_LOG_ERR("open(%s): %s", backing_real, strerror(errno)); p2_meta_kv_str(&m,"status","open_failed"); p2_meta_close(&m); return 1; }
    if (ftruncate(fd, (off_t)fbytes) != 0) { P2_LOG_ERR("ftruncate: %s", strerror(errno)); close(fd); unlink(backing_real); p2_meta_kv_str(&m,"status","ftruncate_failed"); p2_meta_close(&m); return 1; }

    uint8_t *cbuf = (uint8_t *)malloc(cbytes);
    uint8_t *ioblk = (uint8_t *)malloc((size_t)block);
    if (!cbuf || !ioblk) { free(cbuf); free(ioblk); close(fd); unlink(backing_real); p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    p2_rng_fill(&rng, cbuf, cbytes);
    p2_rng_fill(&rng, ioblk, (size_t)block);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    cpu_arg_t ca = { cbuf, cbytes, 0, 0 };
    io_arg_t  ia = { fd, fbytes, (size_t)block, ioblk, 0 };
    pthread_t tc, ti;
    int okc = pthread_create(&tc, NULL, cpu_worker, &ca) == 0;
    int oki = pthread_create(&ti, NULL, io_worker, &ia) == 0;
    if (!okc || !oki) {
        P2_LOG_ERR("pthread_create failed");
        g_run = 0;
        if (okc) pthread_join(tc, NULL);
        if (oki) pthread_join(ti, NULL);
        free(cbuf); free(ioblk); close(fd); if (!keep) unlink(backing_real);
        p2_meta_kv_str(&m,"status","thread_create_failed"); p2_meta_close(&m); return 1;
    }
    while ((p2_monotonic() - t_warm) < (double)duration_s) p2_sleep_ns(50ULL*1000ULL*1000ULL);
    g_run = 0;
    pthread_join(tc, NULL);
    pthread_join(ti, NULL);
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    fsync(fd);
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    free(cbuf); free(ioblk);
    close(fd);
    if (!keep && unlink(backing_real) != 0) P2_LOG_WARN("unlink failed: %s", strerror(errno));

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "cpu_passes", ca.passes);
    p2_meta_kv_u64(&m, "hash_final", ca.h);
    p2_meta_kv_u64(&m, "io_writes", ia.writes);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations", "concurrent compute + file I/O under scheduler pressure");
    p2_meta_close(&m);
    return 0;
}
