/* mixed_mem_io_v2  --  benign system-behaviour benchmark
 *
 * Two concurrent threads: one writes an anonymous memory buffer at a fixed
 * stride, the other writes blocks to a sandbox-validated backing file. Puts
 * simultaneous pressure on the memory and the I/O subsystems.
 *
 * File I/O is confined to a sandbox-validated backing file under /tmp or
 * /var/tmp; no network, no persistence, no encryption. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + create) -> measure (concurrent run) -> cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"
#include <pthread.h>

static const char *TEST = "mixed_mem_io_v2";

static volatile int g_run = 1;

typedef struct { volatile uint8_t *buf; size_t bytes; long stride; uint64_t writes; } mem_arg_t;
typedef struct { int fd; size_t fsize; size_t block; uint8_t *blk; uint64_t writes; } io_arg_t;

static void *mem_worker(void *a) {
    mem_arg_t *w = (mem_arg_t *)a;
    uint64_t passes = 0;
    while (g_run) {
        for (size_t off = 0; off < w->bytes; off += (size_t)w->stride) {
            w->buf[off] = (uint8_t)(off ^ passes);
            w->writes++;
        }
        passes++;
    }
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
"Usage: %s [options]   (benign concurrent memory + file-I/O benchmark)\n"
"  --working-set-mb N    Anonymous buffer in MiB (default 512)\n"
"  --file-size-mb N      Backing file size in MiB (default 256)\n"
"  --block-bytes N       I/O block size (default 65536)\n"
"  --stride BYTES        Memory write stride (default 4096)\n"
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
    long long ws_mb      = p2_get_i64(argc, argv, "--working-set-mb", 512);
    long long size_mb    = p2_get_i64(argc, argv, "--file-size-mb", 256);
    long long block      = p2_get_i64(argc, argv, "--block-bytes", 65536);
    long long stride     = p2_get_i64(argc, argv, "--stride", 4096);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       keep       = p2_flag_present(argc, argv, "--keep-backing");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *backdir  = p2_get_str(argc, argv, "--backing-dir", "/tmp");
    const char *saferoot = p2_get_str(argc, argv, "--safe-root", NULL);

    if (ws_mb <= 0 || ws_mb > 16384) { P2_LOG_ERR("working-set-mb %lld invalid", ws_mb); return 2; }
    if (size_mb <= 0 || size_mb > 65536) { P2_LOG_ERR("file-size-mb %lld invalid", size_mb); return 2; }
    if (block < 512 || block > 16*1024*1024) { P2_LOG_ERR("block-bytes %lld invalid", block); return 2; }
    if (stride < 1 || stride > 64*1024*1024) { P2_LOG_ERR("stride %lld invalid", stride); return 2; }
    size_t mbytes = (size_t)ws_mb * 1024ULL * 1024ULL;
    size_t fbytes = (size_t)size_mb * 1024ULL * 1024ULL;

    char backing_path[PATH_MAX];
    snprintf(backing_path, sizeof(backing_path), "%s/phase2_mixed_%d_%llu.dat",
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
    p2_meta_kv_i64(&m, "working_set_mb", ws_mb);
    p2_meta_kv_i64(&m, "file_size_mb", size_mb);
    p2_meta_kv_i64(&m, "block_bytes", block);
    p2_meta_kv_i64(&m, "stride", stride);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_str(&m, "backing_path", backing_real);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_str(&m, "safety", "benign-benchmark; sandbox-validated backing file; no network/persistence");
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }

    void *buf = mmap(NULL, mbytes, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) { P2_LOG_ERR("mmap: %s", strerror(errno)); p2_meta_kv_str(&m,"status","mmap_failed"); p2_meta_close(&m); return 1; }
    p2_madvise(buf, mbytes, MADV_NOHUGEPAGE);

    int fd = open(backing_real, O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
    if (fd < 0) { P2_LOG_ERR("open(%s): %s", backing_real, strerror(errno)); munmap(buf, mbytes); p2_meta_kv_str(&m,"status","open_failed"); p2_meta_close(&m); return 1; }
    if (ftruncate(fd, (off_t)fbytes) != 0) { P2_LOG_ERR("ftruncate: %s", strerror(errno)); close(fd); unlink(backing_real); munmap(buf, mbytes); p2_meta_kv_str(&m,"status","ftruncate_failed"); p2_meta_close(&m); return 1; }

    uint8_t *ioblk = (uint8_t *)malloc((size_t)block);
    if (!ioblk) { close(fd); unlink(backing_real); munmap(buf, mbytes); p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }
    p2_rng_t rng; p2_rng_seed(&rng, seed); p2_rng_fill(&rng, ioblk, (size_t)block);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    memset(buf, 0, mbytes);
    double t_warm = p2_monotonic();

    p2_phase(TEST, "measure");
    mem_arg_t ma = { (volatile uint8_t *)buf, mbytes, (long)stride, 0 };
    io_arg_t  ia = { fd, fbytes, (size_t)block, ioblk, 0 };
    pthread_t tm, ti;
    int okm = pthread_create(&tm, NULL, mem_worker, &ma) == 0;
    int oki = pthread_create(&ti, NULL, io_worker, &ia) == 0;
    if (!okm || !oki) {
        P2_LOG_ERR("pthread_create failed");
        g_run = 0;
        if (okm) pthread_join(tm, NULL);
        if (oki) pthread_join(ti, NULL);
        free(ioblk); close(fd); if (!keep) unlink(backing_real); munmap(buf, mbytes);
        p2_meta_kv_str(&m,"status","thread_create_failed"); p2_meta_close(&m); return 1;
    }
    while ((p2_monotonic() - t_warm) < (double)duration_s) p2_sleep_ns(50ULL*1000ULL*1000ULL);
    g_run = 0;
    pthread_join(tm, NULL);
    pthread_join(ti, NULL);
    double t_meas = p2_monotonic();

    p2_phase(TEST, "cooldown");
    fsync(fd);
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool = p2_monotonic();

    free(ioblk);
    close(fd);
    if (!keep && unlink(backing_real) != 0) P2_LOG_WARN("unlink failed: %s", strerror(errno));
    munmap(buf, mbytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warm);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool);
    p2_meta_kv_u64(&m, "mem_writes", ma.writes);
    p2_meta_kv_u64(&m, "io_writes", ia.writes);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations", "concurrent memory + file I/O pressure");
    p2_meta_close(&m);
    return 0;
}
