/* sandbox_scanner_metadata
 *
 * Pure metadata scan: stat every file in a sandbox tree. No reads, no writes,
 * no transforms. Designed to isolate dentry/inode cache signature from broad
 * write footprint.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_sandbox.h"

static const char *TEST = "sandbox_scanner_metadata";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --files N             Total files (default 5000)\n"
"  --subdirs N           Number of subdirectories (default 50)\n"
"  --file-size-bytes N   Per-file size (default 4096; small to focus on metadata)\n"
"  --passes N            Repeat the scan N times (default 5)\n"
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

    long long n_files   = p2_get_i64(argc, argv, "--files", 5000);
    long long subdirs   = p2_get_i64(argc, argv, "--subdirs", 50);
    long long fsize     = p2_get_i64(argc, argv, "--file-size-bytes", 4096);
    long long passes    = p2_get_i64(argc, argv, "--passes", 5);
    long long duration  = p2_get_i64(argc, argv, "--duration", 600);
    uint64_t  seed      = p2_get_u64(argc, argv, "--seed", 42);
    const char *sandbox = p2_get_str(argc, argv, "--sandbox-dir", "/tmp");
    const char *safe_root = p2_get_str(argc, argv, "--safe-root", NULL);
    const char *outdir  = p2_get_str(argc, argv, "--output-dir", NULL);
    int       cleanup   = !p2_flag_present(argc, argv, "--no-cleanup");
    int       dry_run   = p2_flag_present(argc, argv, "--dry-run");
    int       markers   = p2_flag_present(argc, argv, "--phase-markers");

    if (n_files > P2_SANDBOX_MAX_FILES) { P2_LOG_ERR("file cap"); return 2; }
    if (subdirs < 1 || subdirs > 1000) { P2_LOG_ERR("subdirs range"); return 2; }
    if (p2_sandbox_check_root(sandbox, safe_root) != 0) return 3;
    if (p2_sandbox_check_caps((int)n_files, (size_t)fsize, (int)duration) != 0) return 3;

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"family","SECURITY-LIKE");
    p2_meta_kv_i64(&m,"files",n_files);
    p2_meta_kv_i64(&m,"subdirs",subdirs);
    p2_meta_kv_i64(&m,"file_size_bytes",fsize);
    p2_meta_kv_i64(&m,"passes",passes);
    p2_meta_kv_i64(&m,"duration_cap_s",duration);
    p2_meta_kv_u64(&m,"seed",seed);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);

    if (dry_run) {
        P2_LOG_INFO("dry-run: would scan %lld files in %lld subdirs (sandbox %s)",
                    n_files, subdirs, sandbox);
        p2_meta_kv_str(&m,"status","dry_run"); p2_meta_close(&m); return 0;
    }

    p2_sandbox_t s;
    if (p2_sandbox_init(&s, sandbox, safe_root, (int)n_files, (size_t)fsize, seed, 0) != 0) {
        p2_meta_kv_str(&m,"status","init_failed"); p2_meta_close(&m); return 1;
    }
    p2_meta_kv_str(&m,"sandbox_path",s.root);

    if (markers) p2_phase(TEST, "generate");
    if (p2_sandbox_generate_files(&s, "dat", (int)subdirs) != 0) {
        p2_sandbox_finalize(&s, 1);
        p2_meta_kv_str(&m,"status","gen_failed"); p2_meta_close(&m); return 1;
    }

    /* Pure scan: walk the tree, stat every file. */
    if (markers) p2_phase(TEST, "scan");
    double t_run_start = p2_monotonic();
    uint64_t stats_done = 0;
    int aborted = 0;
    for (long long pass = 0; pass < passes; pass++) {
        for (long long d = 0; d < subdirs; d++) {
            char dpath[PATH_MAX];
            snprintf(dpath, sizeof(dpath), "%s/d_%03lld", s.root, d);
            DIR *dh = opendir(dpath);
            if (!dh) { P2_LOG_WARN("opendir %s: %s", dpath, strerror(errno)); continue; }
            struct dirent *de;
            while ((de = readdir(dh)) != NULL) {
                if (de->d_name[0] == '.') continue;
                char path[PATH_MAX];
                snprintf(path, sizeof(path), "%s/%s", dpath, de->d_name);
                struct stat st; lstat(path, &st);
                stats_done++;
                if ((p2_monotonic() - t_run_start) >= (double)duration) {
                    aborted = 1; closedir(dh); goto done;
                }
            }
            closedir(dh);
        }
    }
done:;
    double t_run_end = p2_monotonic();
    int cleanup_rc = p2_sandbox_finalize(&s, cleanup);

    p2_meta_kv_f64(&m,"run_start_s",t_run_start);
    p2_meta_kv_f64(&m,"run_end_s",t_run_end);
    p2_meta_kv_u64(&m,"stats_done",stats_done);
    p2_meta_kv_i64(&m,"aborted_by_cap",aborted);
    p2_meta_kv_u64(&m,"bytes_written",s.bytes_written);
    p2_meta_kv_i64(&m,"cleanup_rc",cleanup_rc);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "Dentry/inode cache state at start depends on prior workload; cold-start variant requires drop_caches");
    p2_meta_close(&m);
    return 0;
}
