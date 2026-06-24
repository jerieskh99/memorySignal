/* sandbox_stealth_scattered  --  SAFE SYNTHETIC TEST WORKLOAD (NOT malware)
 *
 * Purpose. A spatially-diffuse variant of the stealth pattern: single-page rewrites issued in
 * a RANDOMISED order across a pool of files, so consecutive operations touch distant locations
 * and any one snapshot window catches only a page or two, each fully rewritten (reversible
 * XOR). Tests whether the detector's per-page magnitude signal survives when the changed pages
 * are scattered rather than sequential. A defensive test case to harden the detector; it does
 * not evade anything.
 *
 * Safety. Identical guarantees to the other security_like_safe simulators: disposable
 * sandbox files only; reversible XOR (NOT encryption); no network, persistence, privilege use,
 * or anti-analysis; resource- and time-capped; cleanup confined to the sandbox. See
 * ../docs/SAFETY_MODEL.md and ../../RESEARCH_SAFETY_NOTICE.md.
 *
 * Pattern. Generate single-page files, shuffle their order deterministically (seeded), then
 * process one every --interval-ms: read -> XOR(whole page) -> write -> rename.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_sandbox.h"

static const char *TEST = "sandbox_stealth_scattered";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (SAFE synthetic test workload; low-APF / high-Hamming, scattered)\n"
"  --files N             (default 512)\n"
"  --file-size-bytes N   (default 4096 = one page)\n"
"  --interval-ms N       Sleep between pages (default 200)\n"
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

    long long n_files     = p2_get_i64(argc, argv, "--files", 512);
    long long fsize       = p2_get_i64(argc, argv, "--file-size-bytes", 4096);
    long long interval_ms = p2_get_i64(argc, argv, "--interval-ms", 200);
    long long duration    = p2_get_i64(argc, argv, "--duration", 600);
    uint64_t  seed        = p2_get_u64(argc, argv, "--seed", 42);
    const char *sandbox   = p2_get_str(argc, argv, "--sandbox-dir", "/tmp");
    const char *safe_root = p2_get_str(argc, argv, "--safe-root", NULL);
    const char *outdir    = p2_get_str(argc, argv, "--output-dir", NULL);
    int       cleanup     = !p2_flag_present(argc, argv, "--no-cleanup");
    int       dry_run     = p2_flag_present(argc, argv, "--dry-run");
    int       markers     = p2_flag_present(argc, argv, "--phase-markers");

    if (interval_ms < 0 || interval_ms > 600000) { P2_LOG_ERR("interval-ms out of range"); return 2; }
    if (duration > P2_SANDBOX_MAX_DURATION_S) { P2_LOG_ERR("duration cap"); return 2; }
    if (p2_sandbox_check_root(sandbox, safe_root) != 0) return 3;
    if (p2_sandbox_check_caps((int)n_files, (size_t)fsize, (int)duration) != 0) return 3;

    p2_meta_t m; p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m,"test_name",TEST);
    p2_meta_kv_str(&m,"language","C");
    p2_meta_kv_str(&m,"family","SECURITY-LIKE");
    p2_meta_kv_str(&m,"subfamily","STEALTH");
    p2_meta_kv_str(&m,"safety","safe-synthetic; reversible XOR; sandbox-only; no network/persistence/evasion");
    p2_meta_kv_i64(&m,"files",n_files);
    p2_meta_kv_i64(&m,"file_size_bytes",fsize);
    p2_meta_kv_i64(&m,"interval_ms",interval_ms);
    p2_meta_kv_i64(&m,"duration_cap_s",duration);
    p2_meta_kv_u64(&m,"seed",seed);
    char tstart[32]; p2_iso_timestamp(tstart,sizeof(tstart));
    p2_meta_kv_str(&m,"start_time",tstart);

    if (dry_run) {
        P2_LOG_INFO("dry-run: %lld scattered one-page rewrites at %lldms in %s", n_files, interval_ms, sandbox);
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

    /* Deterministic Fisher-Yates shuffle of the processing order (spatial diffusion). */
    int *order = (int *)malloc((size_t)n_files * sizeof(int));
    if (!order) { p2_sandbox_finalize(&s, 1);
                  p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }
    for (int i = 0; i < n_files; i++) order[i] = i;
    p2_rng_t rng; p2_rng_seed(&rng, seed ^ 0x5CA77E5EDULL);
    for (int i = (int)n_files - 1; i > 0; i--) {
        uint64_t r; p2_rng_fill(&rng, &r, sizeof(r));
        int j = (int)(r % (uint64_t)(i + 1));
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) { free(order); p2_sandbox_finalize(&s, 1);
                p2_meta_kv_str(&m,"status","alloc_failed"); p2_meta_close(&m); return 1; }

    double t_run_start = p2_monotonic();
    int processed = 0;
    for (int k = 0; k < n_files; k++) {
        if ((p2_monotonic() - t_run_start) >= (double)duration) break;
        char path[PATH_MAX]; snprintf(path,sizeof(path),"%s/file_%05d.dat",s.root,order[k]);
        if (markers) p2_phase(TEST, "page_begin");
        if (p2_sandbox_phase_read(&s, path, buf, (size_t)fsize) != 0) break;
        p2_sandbox_xor(buf, (size_t)fsize, s.key);          /* full-page flip -> high Hamming */
        if (p2_sandbox_phase_write(&s, path, buf, (size_t)fsize) != 0) break;
        p2_sandbox_phase_rename(&s, path, "simx");
        processed++;
        if (markers) p2_phase(TEST, "page_done");
        if (interval_ms > 0 && k + 1 < n_files)
            p2_sleep_ns((uint64_t)interval_ms * 1000000ULL);
    }
    double t_run_end = p2_monotonic();
    free(buf); free(order);
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
                   "Scattered single-page rewrites: low APF, high Hamming-per-page, spatially diffuse");
    p2_meta_close(&m);
    return 0;
}
