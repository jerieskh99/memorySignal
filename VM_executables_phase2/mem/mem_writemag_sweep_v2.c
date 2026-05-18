/* mem_writemag_sweep_v2
 *
 * Per-page write magnitude sweep over a fixed 1 GiB buffer.
 * Variants control how many bytes are written per page each pass:
 *   --bytes-per-page 1   (A: minimal magnitude, parity with mem_stream_v2)
 *   --bytes-per-page 64  (B: cache-line)
 *   --bytes-per-page 1024 (C)
 *   --bytes-per-page 4096 (D: full page; saturates Hamming delta)
 *
 * Source bytes are PRNG-generated (seeded) so writes change page content
 * (no all-zero writes that the snapshot pipeline could collapse).
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "mem_writemag_sweep_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --working-set-mb N     Buffer size (default 1024)\n"
"  --bytes-per-page N     Bytes written per page per pass (default 64)\n"
"  --duration SEC         Measurement duration (default 60)\n"
"  --warmup SEC           Warm-up (default 5)\n"
"  --seed N               PRNG seed (default 42)\n"
"  --output-dir PATH      Metadata dir\n"
"  --cpu-affinity N       Pin CPU\n"
"  --no-mlock             Skip mlock\n"
"  --phase-markers        Emit phase markers\n"
"  --dry-run / --help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long ws_mb     = p2_get_i64(argc, argv, "--working-set-mb", 1024);
    long long bpp       = p2_get_i64(argc, argv, "--bytes-per-page", 64);
    long long duration  = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup    = p2_get_i64(argc, argv, "--warmup", 5);
    long long cpu       = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed      = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock  = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run   = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir  = p2_get_str(argc, argv, "--output-dir", NULL);

    if (bpp < 1 || bpp > 4096) { P2_LOG_ERR("bytes-per-page must be 1..4096"); return 2; }

    size_t bytes = (size_t)ws_mb * 1024ULL * 1024ULL;
    size_t pages = bytes / 4096ULL;

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_i64(&m, "working_set_mb", ws_mb);
    p2_meta_kv_i64(&m, "bytes_per_page", bpp);
    p2_meta_kv_i64(&m, "duration_s", duration);
    p2_meta_kv_i64(&m, "warmup_s", warmup);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_u64(&m, "pages", pages);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status","dry_run"); p2_meta_close(&m); return 0; }

    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* Source bytes (one page worth) regenerated each pass via PRNG. */
    uint8_t *src = (uint8_t *)aligned_alloc(64, 4096);
    if (!src) { P2_LOG_ERR("aligned_alloc source failed"); p2_meta_close(&m); return 1; }
    p2_rng_t rng; p2_rng_seed(&rng, seed);

    void *buf = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                     MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        P2_LOG_ERR("mmap failed: %s", strerror(errno));
        free(src); p2_meta_kv_str(&m,"status","mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(buf, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(buf, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    memset(buf, 0, bytes);
    double t_wend = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_mstart = t_wend;
    uint8_t *base = (uint8_t *)buf;
    uint64_t passes = 0, writes = 0, written_bytes = 0;
    while ((p2_monotonic() - t_mstart) < (double)duration) {
        p2_rng_fill(&rng, src, 4096);   /* fresh source for this pass */
        for (size_t pg = 0; pg < pages; pg++) {
            memcpy(base + pg * 4096ULL, src, (size_t)bpp);
            writes++;
            written_bytes += (uint64_t)bpp;
        }
        passes++;
    }
    double t_mend = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500000000ULL);
    double t_cend = p2_monotonic();

    munmap(buf, bytes);
    free(src);

    p2_meta_kv_f64(&m,"warmup_t0_s",t0);
    p2_meta_kv_f64(&m,"warmup_end_s",t_wend);
    p2_meta_kv_f64(&m,"measure_end_s",t_mend);
    p2_meta_kv_f64(&m,"cooldown_end_s",t_cend);
    p2_meta_kv_u64(&m,"passes",passes);
    p2_meta_kv_u64(&m,"writes",writes);
    p2_meta_kv_u64(&m,"bytes_written",written_bytes);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "Hamming-delta saturates above ~2KB/page; cosine ~0 at <64B/page");
    p2_meta_close(&m);
    return 0;
}
