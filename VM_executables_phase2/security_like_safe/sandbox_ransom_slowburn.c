/* sandbox_ransom_slowburn
 *
 * Paced sequential variant: one file every --interval-s seconds (default 5).
 * Mimics the *cadence* pattern of slow-burn behavioral patterns. Safe-sandbox
 * rules apply (XOR transform, no network, no persistence).
 */
#include "../common/phase2_common.h"
#include "../common/phase2_sandbox.h"

static const char *TEST = "sandbox_ransom_slowburn";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --files N             (default 100)\n"
"  --file-size-bytes N   (default 1048576)\n"
"  --interval-s N        Sleep between files (default 5)\n"
"  --duration SEC        Hard time cap (default 600)\n"
"  --sandbox-dir PATH    (default /tmp)\n"
"  --safe-root PATH\n"
"  --seed N              (default 42)\n"
"  --output-dir PATH\n"
"  --no-cleanup\n"
"  --dry-run / --phase-markers / --help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }

    long long n_files   = p2_get_i64(argc, argv, "--files", 100);
    long long fsize     = p2_get_i64(argc, argv, "--file-size-bytes", 1024*1024);
    long long interval  = p2_get_i64(argc, argv, "--interval-s", 5);
    long long duration  = p2_get_i64(argc, argv, "--duration", 600);
    uint64_t  seed      = p2_get_u64(argc, argv, "--seed", 42);
    const char *sandbox = p2_get_str(argc, argv, "--sandbox-dir", "/tmp");
    const char *safe_root = p2_get_str(argc, argv, "--safe-root", NULL);
    const char *outdir  = p2_get_str(argc, argv, "--output-dir", NULL);
    int       cleanup   = !p2_flag_present(argc, argv, "--no-cleanup");
    int       dry_run   = p2_flag_present(argc, argv, "--dry-run");
    int       markers   = p2_flag_present(argc, argv, "--phase-markers");

    if (interval < 0 || interval > 600) { P2_LOG_ERR("interval out of range"); return 2; }
    if (duration > P2_SANDBOX_MAX_DURATION_S) { P2_LOG_ERR("duration cap"); return 2; }
    if (p2_sandbox_check_root(sandbox, safe_root) != 0) return 3;
    if (p2_sandbox_check_caps((int)n_files, (size_t)fsize, (int)duration) != 0) return 3;

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"family","SECURITY-LIKE");
    p2_meta_kv_i64(&m,"files",n_files);
    p2_meta_kv_i64(&m,"file_size_bytes",fsize);
    p2_meta_kv_i64(&m,"interval_s",interval);
    p2_meta_kv_i64(&m,"duration_cap_s",duration);
    p2_meta_kv_u64(&m,"seed",seed);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);

    if (dry_run) {
        P2_LOG_INFO("dry-run: %lld files at %llds interval in %s", n_files, interval, sandbox);
        p2_meta_kv_str(&m,"status","dry_run"); p2_meta_close(&m); return 0;
    }

    p2_sandbox_t s;
    if (p2_sandbox_init(&s, sandbox, safe_root, (int)n_files, (size_t)fsize, seed, 0) != 0) {
        p2_meta_kv_str(&m,"status","init_failed"); p2_meta_close(&m); return 1;
    }
    p2_meta_kv_str(&m,"sandbox_path",s.root);

    if (markers) p2_phase(TEST, "generate");
    if (p2_sandbox_generate_files(&s, "dat", 0) != 0) {
        p2_sandbox_finalize(&s, 1);
        p2_meta_kv_str(&m,"status","gen_failed"); p2_meta_close(&m); return 1;
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) { p2_sandbox_finalize(&s, 1);
                p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }

    double t_run_start = p2_monotonic();
    int processed = 0;
    for (int i = 0; i < n_files; i++) {
        if ((p2_monotonic() - t_run_start) >= (double)duration) break;
        char path[PATH_MAX]; snprintf(path,sizeof(path),"%s/file_%05d.dat",s.root,i);
        if (markers) p2_phase(TEST, "file_begin");
        struct stat st; lstat(path, &st);
        if (p2_sandbox_phase_read(&s, path, buf, (size_t)fsize) != 0) break;
        p2_sandbox_xor(buf, (size_t)fsize, s.key);
        if (p2_sandbox_phase_write(&s, path, buf, (size_t)fsize) != 0) break;
        p2_sandbox_phase_rename(&s, path, "simx");
        processed++;
        if (markers) p2_phase(TEST, "file_done");
        if (interval > 0 && i + 1 < n_files) {
            p2_sleep_ns((uint64_t)interval * 1000000000ULL);
        }
    }
    double t_run_end = p2_monotonic();
    free(buf);
    int cleanup_rc = p2_sandbox_finalize(&s, cleanup);

    p2_meta_kv_f64(&m,"run_start_s",t_run_start);
    p2_meta_kv_f64(&m,"run_end_s",t_run_end);
    p2_meta_kv_i64(&m,"files_processed",processed);
    p2_meta_kv_u64(&m,"bytes_read",s.bytes_read);
    p2_meta_kv_u64(&m,"bytes_written",s.bytes_written);
    p2_meta_kv_i64(&m,"cleanup_rc",cleanup_rc);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "Per-file phases are short relative to sleep; cadence dominates signal");
    p2_meta_close(&m);
    return 0;
}
