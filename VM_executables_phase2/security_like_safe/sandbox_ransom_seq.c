/* sandbox_ransom_seq
 *
 * Safe synthetic workload that mimics the high-level behavioral *pattern*
 * of per-file sequential processing (stat → read → reversible-XOR →
 * write → rename). This is NOT ransomware: it operates only on disposable
 * generated files inside a validated sandbox, uses a reversible XOR with a
 * fixed key, never persists, never communicates over a network, never touches
 * anything outside its sandbox dir.
 *
 * Phases per file:
 *   1. stat       (metadata probe)
 *   2. read       (whole-file)
 *   3. transform  (XOR with fixed 32-byte key in memory)
 *   4. write      (overwrite original; same size)
 *   5. rename     (append .simx suffix)
 */
#include "../common/phase2_common.h"
#include "../common/phase2_sandbox.h"

static const char *TEST = "sandbox_ransom_seq";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]\n"
"  --files N             Number of files to process (default 1000, max 5000)\n"
"  --file-size-bytes N   Bytes per file (default 1048576 = 1 MiB)\n"
"  --duration SEC        Hard time cap (default 600)\n"
"  --sandbox-dir PATH    Parent of sandbox (default /tmp)\n"
"  --safe-root PATH      Additional approved root\n"
"  --seed N              PRNG seed (default 42)\n"
"  --output-dir PATH     Metadata dir\n"
"  --no-cleanup          Keep sandbox for inspection (default cleanup on)\n"
"  --dry-run             Validate args, no file I/O\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --help\n"
"\n"
"Safety: only operates inside <sandbox-dir>/phase2_sandbox_<pid>_<seed>/.\n"
"        XOR transform is reversible. No network. No persistence.\n", p);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }

    long long n_files   = p2_get_i64(argc, argv, "--files", 1000);
    long long fsize     = p2_get_i64(argc, argv, "--file-size-bytes", 1024*1024);
    long long duration  = p2_get_i64(argc, argv, "--duration", 600);
    uint64_t  seed      = p2_get_u64(argc, argv, "--seed", 42);
    const char *sandbox = p2_get_str(argc, argv, "--sandbox-dir", "/tmp");
    const char *safe_root = p2_get_str(argc, argv, "--safe-root", NULL);
    const char *outdir  = p2_get_str(argc, argv, "--output-dir", NULL);
    int       cleanup   = !p2_flag_present(argc, argv, "--no-cleanup");
    int       dry_run   = p2_flag_present(argc, argv, "--dry-run");

    if (duration > P2_SANDBOX_MAX_DURATION_S) {
        P2_LOG_ERR("duration %lld exceeds cap %d", duration, P2_SANDBOX_MAX_DURATION_S);
        return 2;
    }
    /* Pre-flight sandbox path check — runs in dry-run too so the user
     * cannot accidentally believe an invalid path was accepted. */
    if (p2_sandbox_check_root(sandbox, safe_root) != 0) return 3;
    if (p2_sandbox_check_caps((int)n_files, (size_t)fsize, (int)duration) != 0) return 3;

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"family","SECURITY-LIKE");
    p2_meta_kv_i64(&m,"files",n_files);
    p2_meta_kv_i64(&m,"file_size_bytes",fsize);
    p2_meta_kv_i64(&m,"duration_cap_s",duration);
    p2_meta_kv_u64(&m,"seed",seed);
    p2_meta_kv_i64(&m,"cleanup",cleanup);
    p2_meta_kv_i64(&m,"dry_run",dry_run);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);
    p2_meta_kv_str(&m,"safety_model","sandbox-only, reversible XOR, no network, no persistence");

    if (dry_run) {
        P2_LOG_INFO("dry-run: would process %lld files of %lld bytes inside %s",
                    n_files, fsize, sandbox);
        p2_meta_kv_str(&m,"status","dry_run");
        p2_meta_close(&m);
        return 0;
    }

    p2_sandbox_t s;
    if (p2_sandbox_init(&s, sandbox, safe_root, (int)n_files, (size_t)fsize, seed, 0) != 0) {
        p2_meta_kv_str(&m,"status","init_failed"); p2_meta_close(&m); return 1;
    }
    p2_meta_kv_str(&m,"sandbox_path",s.root);

    p2_phase(TEST, "generate");
    double tg0 = p2_monotonic();
    if (p2_sandbox_generate_files(&s, "dat", 0) != 0) {
        p2_sandbox_finalize(&s, 1);
        p2_meta_kv_str(&m,"status","gen_failed"); p2_meta_close(&m); return 1;
    }
    double tg_end = p2_monotonic();

    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) {
        p2_sandbox_finalize(&s, 1);
        p2_meta_kv_str(&m,"status","malloc_failed"); p2_meta_close(&m); return 1;
    }

    double t_run_start = p2_monotonic();
    int processed = 0;
    int aborted = 0;
    for (int i = 0; i < n_files; i++) {
        if ((p2_monotonic() - t_run_start) >= (double)duration) {
            P2_LOG_WARN("duration cap reached after %d files", processed);
            aborted = 1; break;
        }
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/file_%05d.dat", s.root, i);

        struct stat st;
        if (p2_flag_present(argc, argv, "--phase-markers")) p2_phase(TEST, "stat");
        if (lstat(path, &st) != 0) { P2_LOG_ERR("lstat %s: %s", path, strerror(errno)); break; }

        if (p2_flag_present(argc, argv, "--phase-markers")) p2_phase(TEST, "read");
        if (p2_sandbox_phase_read(&s, path, buf, (size_t)fsize) != 0) break;

        if (p2_flag_present(argc, argv, "--phase-markers")) p2_phase(TEST, "transform");
        p2_sandbox_xor(buf, (size_t)fsize, s.key);

        if (p2_flag_present(argc, argv, "--phase-markers")) p2_phase(TEST, "write");
        if (p2_sandbox_phase_write(&s, path, buf, (size_t)fsize) != 0) break;

        if (p2_flag_present(argc, argv, "--phase-markers")) p2_phase(TEST, "rename");
        if (p2_sandbox_phase_rename(&s, path, "simx") != 0) break;
        processed++;
    }
    double t_run_end = p2_monotonic();
    free(buf);

    int cleanup_rc = p2_sandbox_finalize(&s, cleanup);

    p2_meta_kv_f64(&m,"generate_t0_s",tg0);
    p2_meta_kv_f64(&m,"generate_end_s",tg_end);
    p2_meta_kv_f64(&m,"run_start_s",t_run_start);
    p2_meta_kv_f64(&m,"run_end_s",t_run_end);
    p2_meta_kv_i64(&m,"files_processed",processed);
    p2_meta_kv_i64(&m,"aborted_by_cap",aborted);
    p2_meta_kv_u64(&m,"bytes_read",s.bytes_read);
    p2_meta_kv_u64(&m,"bytes_written",s.bytes_written);
    p2_meta_kv_u64(&m,"files_renamed",s.files_renamed);
    p2_meta_kv_i64(&m,"cleanup_rc",cleanup_rc);
    p2_meta_kv_str(&m,"status","ok");
    char tend[32]; p2_iso_timestamp(tend,sizeof(tend));
    p2_meta_kv_str(&m,"end_time",tend);
    p2_meta_kv_str(&m,"known_limitations",
                   "Behavioral-pattern mimicry only; no encryption, persistence, or evasion");
    p2_meta_close(&m);
    return 0;
}
