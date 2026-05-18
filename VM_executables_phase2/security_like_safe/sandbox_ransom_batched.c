/* sandbox_ransom_batched
 *
 * Same five behavioral phases as sandbox_ransom_seq but batched:
 *   phase 1: stat ALL files
 *   phase 2: read ALL files into memory buffers
 *   phase 3: XOR-transform ALL buffers in memory
 *   phase 4: write ALL buffers back to their files
 *   phase 5: rename ALL files
 *
 * Designed to produce mechanism-aligned segment boundaries for the segment-
 * level analyzer to recover.
 *
 * Memory cap: N * file_size must fit in --mem-cap-mb (default 1024); the test
 * refuses to start if the requested in-memory footprint would exceed it.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_sandbox.h"

static const char *TEST = "sandbox_ransom_batched";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --files N             (default 1000)\n"
"  --file-size-bytes N   (default 1048576)\n"
"  --duration SEC        (default 600)\n"
"  --mem-cap-mb N        Max in-memory buffer total (default 1024)\n"
"  --sandbox-dir PATH    (default /tmp)\n"
"  --safe-root PATH\n"
"  --seed N              (default 42)\n"
"  --output-dir PATH\n"
"  --no-cleanup          (default cleanup on)\n"
"  --dry-run / --phase-markers / --help\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }

    long long n_files   = p2_get_i64(argc, argv, "--files", 1000);
    long long fsize     = p2_get_i64(argc, argv, "--file-size-bytes", 1024*1024);
    long long duration  = p2_get_i64(argc, argv, "--duration", 600);
    long long mem_cap   = p2_get_i64(argc, argv, "--mem-cap-mb", 1024);
    uint64_t  seed      = p2_get_u64(argc, argv, "--seed", 42);
    const char *sandbox = p2_get_str(argc, argv, "--sandbox-dir", "/tmp");
    const char *safe_root = p2_get_str(argc, argv, "--safe-root", NULL);
    const char *outdir  = p2_get_str(argc, argv, "--output-dir", NULL);
    int       cleanup   = !p2_flag_present(argc, argv, "--no-cleanup");
    int       dry_run   = p2_flag_present(argc, argv, "--dry-run");
    int       markers   = p2_flag_present(argc, argv, "--phase-markers");
    if (p2_sandbox_check_root(sandbox, safe_root) != 0) return 3;
    if (p2_sandbox_check_caps((int)n_files, (size_t)fsize, (int)duration) != 0) return 3;

    uint64_t footprint = (uint64_t)n_files * (uint64_t)fsize;
    if (footprint > (uint64_t)mem_cap * 1024ULL * 1024ULL) {
        P2_LOG_ERR("in-memory footprint %llu exceeds --mem-cap-mb=%lld",
                   (unsigned long long)footprint, mem_cap);
        return 2;
    }
    if (duration > P2_SANDBOX_MAX_DURATION_S) {
        P2_LOG_ERR("duration cap exceeded"); return 2;
    }

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"family","SECURITY-LIKE");
    p2_meta_kv_i64(&m,"files",n_files);
    p2_meta_kv_i64(&m,"file_size_bytes",fsize);
    p2_meta_kv_i64(&m,"duration_cap_s",duration);
    p2_meta_kv_i64(&m,"mem_cap_mb",mem_cap);
    p2_meta_kv_u64(&m,"seed",seed);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);
    p2_meta_kv_str(&m,"safety_model","sandbox-only, reversible XOR, in-memory cap enforced");

    if (dry_run) {
        P2_LOG_INFO("dry-run: would batch-process %lld files (%llu bytes total) in %s",
                    n_files, (unsigned long long)footprint, sandbox);
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

    uint8_t **bufs = (uint8_t **)calloc((size_t)n_files, sizeof(uint8_t *));
    if (!bufs) {
        p2_sandbox_finalize(&s, 1);
        p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1;
    }
    for (int i = 0; i < n_files; i++) {
        bufs[i] = (uint8_t *)malloc((size_t)fsize);
        if (!bufs[i]) {
            for (int j = 0; j < i; j++) free(bufs[j]);
            free(bufs);
            p2_sandbox_finalize(&s, 1);
            p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1;
        }
    }

    double t_run_start = p2_monotonic();

    /* Phase 1: stat all */
    if (markers) p2_phase(TEST, "stat_all");
    double tp1 = p2_monotonic();
    for (int i = 0; i < n_files; i++) {
        char path[PATH_MAX]; snprintf(path,sizeof(path),"%s/file_%05d.dat",s.root,i);
        struct stat st; lstat(path, &st);
    }
    /* Phase 2: read all */
    if (markers) p2_phase(TEST, "read_all");
    double tp2 = p2_monotonic();
    for (int i = 0; i < n_files; i++) {
        char path[PATH_MAX]; snprintf(path,sizeof(path),"%s/file_%05d.dat",s.root,i);
        if (p2_sandbox_phase_read(&s, path, bufs[i], (size_t)fsize) != 0) break;
    }
    /* Phase 3: transform all */
    if (markers) p2_phase(TEST, "transform_all");
    double tp3 = p2_monotonic();
    for (int i = 0; i < n_files; i++) p2_sandbox_xor(bufs[i], (size_t)fsize, s.key);

    /* Phase 4: write all */
    if (markers) p2_phase(TEST, "write_all");
    double tp4 = p2_monotonic();
    for (int i = 0; i < n_files; i++) {
        char path[PATH_MAX]; snprintf(path,sizeof(path),"%s/file_%05d.dat",s.root,i);
        if (p2_sandbox_phase_write(&s, path, bufs[i], (size_t)fsize) != 0) break;
    }
    /* Phase 5: rename all */
    if (markers) p2_phase(TEST, "rename_all");
    double tp5 = p2_monotonic();
    for (int i = 0; i < n_files; i++) {
        char path[PATH_MAX]; snprintf(path,sizeof(path),"%s/file_%05d.dat",s.root,i);
        p2_sandbox_phase_rename(&s, path, "simx");
    }
    double t_run_end = p2_monotonic();

    for (int i = 0; i < n_files; i++) free(bufs[i]);
    free(bufs);
    int cleanup_rc = p2_sandbox_finalize(&s, cleanup);

    p2_meta_kv_f64(&m,"run_start_s",t_run_start);
    p2_meta_kv_f64(&m,"phase_stat_start_s",tp1);
    p2_meta_kv_f64(&m,"phase_read_start_s",tp2);
    p2_meta_kv_f64(&m,"phase_transform_start_s",tp3);
    p2_meta_kv_f64(&m,"phase_write_start_s",tp4);
    p2_meta_kv_f64(&m,"phase_rename_start_s",tp5);
    p2_meta_kv_f64(&m,"run_end_s",t_run_end);
    p2_meta_kv_u64(&m,"bytes_read",s.bytes_read);
    p2_meta_kv_u64(&m,"bytes_written",s.bytes_written);
    p2_meta_kv_u64(&m,"files_renamed",s.files_renamed);
    p2_meta_kv_i64(&m,"cleanup_rc",cleanup_rc);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "Holds all buffers in RAM; not representative of stream-mode tools");
    p2_meta_close(&m);
    return 0;
}
