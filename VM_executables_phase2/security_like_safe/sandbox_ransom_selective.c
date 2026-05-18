/* sandbox_ransom_selective
 *
 * Discovery is extension-filtered: sandbox holds 2N generated files of two
 * extensions ("dat" and "bin"); only ".dat" files are processed. Adds a
 * discovery-phase cost that the analyzer can detect.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_sandbox.h"

static const char *TEST = "sandbox_ransom_selective";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --files N             Files of EACH extension (.dat + .bin = 2N total) (default 500)\n"
"  --file-size-bytes N   (default 524288 = 512 KiB)\n"
"  --duration SEC        (default 600)\n"
"  --sandbox-dir PATH    (default /tmp)\n"
"  --safe-root PATH\n"
"  --seed N              (default 42)\n"
"  --output-dir PATH\n"
"  --no-cleanup\n"
"  --dry-run / --phase-markers / --help\n", p);
}

static int has_ext(const char *name, const char *ext) {
    size_t ln = strlen(name), le = strlen(ext);
    return ln > le + 1 && name[ln - le - 1] == '.' &&
           strcmp(name + ln - le, ext) == 0;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }

    long long n_each    = p2_get_i64(argc, argv, "--files", 500);
    long long fsize     = p2_get_i64(argc, argv, "--file-size-bytes", 512*1024);
    long long duration  = p2_get_i64(argc, argv, "--duration", 600);
    uint64_t  seed      = p2_get_u64(argc, argv, "--seed", 42);
    const char *sandbox = p2_get_str(argc, argv, "--sandbox-dir", "/tmp");
    const char *safe_root = p2_get_str(argc, argv, "--safe-root", NULL);
    const char *outdir  = p2_get_str(argc, argv, "--output-dir", NULL);
    int       cleanup   = !p2_flag_present(argc, argv, "--no-cleanup");
    int       dry_run   = p2_flag_present(argc, argv, "--dry-run");
    int       markers   = p2_flag_present(argc, argv, "--phase-markers");

    long long total_files = n_each * 2;
    if (total_files > P2_SANDBOX_MAX_FILES) { P2_LOG_ERR("total files cap"); return 2; }
    if (duration > P2_SANDBOX_MAX_DURATION_S) { P2_LOG_ERR("duration cap"); return 2; }
    if (p2_sandbox_check_root(sandbox, safe_root) != 0) return 3;
    if (p2_sandbox_check_caps((int)total_files, (size_t)fsize, (int)duration) != 0) return 3;

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"family","SECURITY-LIKE");
    p2_meta_kv_i64(&m,"files_each_ext",n_each);
    p2_meta_kv_i64(&m,"files_total",total_files);
    p2_meta_kv_i64(&m,"file_size_bytes",fsize);
    p2_meta_kv_i64(&m,"duration_cap_s",duration);
    p2_meta_kv_u64(&m,"seed",seed);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);

    if (dry_run) {
        P2_LOG_INFO("dry-run: %lld .dat + %lld .bin files in %s",
                    n_each, n_each, sandbox);
        p2_meta_kv_str(&m,"status","dry_run"); p2_meta_close(&m); return 0;
    }

    /* Use sandbox helper with N=n_each then create n_each additional .bin
       files in the same directory. */
    p2_sandbox_t s;
    if (p2_sandbox_init(&s, sandbox, safe_root, (int)n_each, (size_t)fsize, seed, 0) != 0) {
        p2_meta_kv_str(&m,"status","init_failed"); p2_meta_close(&m); return 1;
    }
    p2_meta_kv_str(&m,"sandbox_path",s.root);

    if (markers) p2_phase(TEST, "generate_dat");
    if (p2_sandbox_generate_files(&s, "dat", 0) != 0) {
        p2_sandbox_finalize(&s, 1);
        p2_meta_kv_str(&m,"status","gen_failed"); p2_meta_close(&m); return 1;
    }

    /* Generate .bin counterparts by writing additional files (same root). */
    if (markers) p2_phase(TEST, "generate_bin");
    {
        p2_rng_t rng; p2_rng_seed(&rng, seed ^ 0xBEEFDEAD);
        uint8_t chunk[65536];
        for (int i = 0; i < n_each; i++) {
            char path[PATH_MAX];
            snprintf(path,sizeof(path),"%s/file_%05d.bin",s.root,i);
            int fd = open(path, O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW, 0600);
            if (fd < 0) { P2_LOG_ERR("open bin: %s", strerror(errno)); break; }
            size_t remaining = (size_t)fsize;
            while (remaining > 0) {
                size_t n = remaining > sizeof(chunk) ? sizeof(chunk) : remaining;
                p2_rng_fill(&rng, chunk, n);
                ssize_t w = write(fd, chunk, n);
                if (w <= 0) break;
                remaining -= (size_t)w;
                s.bytes_written += (uint64_t)w;
            }
            close(fd);
            s.files_created++;
        }
    }

    /* Discovery phase: enumerate, filter to .dat. */
    if (markers) p2_phase(TEST, "discover");
    double t_disc0 = p2_monotonic();
    DIR *d = opendir(s.root);
    if (!d) { P2_LOG_ERR("opendir: %s", strerror(errno));
              p2_sandbox_finalize(&s, 1);
              p2_meta_kv_str(&m,"status","opendir_failed"); p2_meta_close(&m); return 1; }

    char (*matched)[PATH_MAX] = (char(*)[PATH_MAX])malloc(sizeof(*matched) * (size_t)n_each);
    if (!matched) { closedir(d); p2_sandbox_finalize(&s, 1);
                    p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }
    int matched_n = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL && matched_n < n_each) {
        if (de->d_name[0] == '.') continue;
        if (!has_ext(de->d_name, "dat")) continue;
        snprintf(matched[matched_n++], PATH_MAX, "%s/%s", s.root, de->d_name);
    }
    closedir(d);
    double t_disc_end = p2_monotonic();

    /* Process matched files sequentially with full 5-phase pipeline. */
    if (markers) p2_phase(TEST, "process");
    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) { free(matched); p2_sandbox_finalize(&s, 1);
                p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }

    double t_run_start = p2_monotonic();
    int processed = 0;
    for (int i = 0; i < matched_n; i++) {
        if ((p2_monotonic() - t_run_start) >= (double)duration) break;
        struct stat st; lstat(matched[i], &st);
        if (p2_sandbox_phase_read(&s, matched[i], buf, (size_t)fsize) != 0) break;
        p2_sandbox_xor(buf, (size_t)fsize, s.key);
        if (p2_sandbox_phase_write(&s, matched[i], buf, (size_t)fsize) != 0) break;
        p2_sandbox_phase_rename(&s, matched[i], "simx");
        processed++;
    }
    double t_run_end = p2_monotonic();
    free(buf); free(matched);
    int cleanup_rc = p2_sandbox_finalize(&s, cleanup);

    p2_meta_kv_f64(&m,"discover_start_s",t_disc0);
    p2_meta_kv_f64(&m,"discover_end_s",t_disc_end);
    p2_meta_kv_f64(&m,"run_start_s",t_run_start);
    p2_meta_kv_f64(&m,"run_end_s",t_run_end);
    p2_meta_kv_i64(&m,"matched_n",matched_n);
    p2_meta_kv_i64(&m,"files_processed",processed);
    p2_meta_kv_u64(&m,"bytes_read",s.bytes_read);
    p2_meta_kv_u64(&m,"bytes_written",s.bytes_written);
    p2_meta_kv_i64(&m,"cleanup_rc",cleanup_rc);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "Filter is exact-extension match only; richer rules not implemented");
    p2_meta_close(&m);
    return 0;
}
