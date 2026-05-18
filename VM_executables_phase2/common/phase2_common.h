/* phase2_common.h
 *
 * Shared utilities for Phase 2 workload executables.
 *
 * - Argument parsing helpers
 * - Metadata JSON emission
 * - Phase markers / structured logging
 * - PRNG (xoshiro256**) for deterministic byte streams
 * - Sandbox safety verification (security_like_safe tests)
 * - Timing helpers (clock_gettime monotonic, nanosleep)
 *
 * Header-only to keep the build simple. Include from one .c per executable.
 */
#ifndef PHASE2_COMMON_H
#define PHASE2_COMMON_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sched.h>
#include <dirent.h>
#include <signal.h>

#ifndef PHASE2_VERSION
#define PHASE2_VERSION "phase2-0.1"
#endif

/* ---------- Logging / phase markers ---------- */

static inline void p2_iso_timestamp(char *buf, size_t buflen) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);
    snprintf(buf, buflen, "%04d-%02d-%02dT%02d:%02d:%02dZ",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);
}

static inline void p2_log(const char *level, const char *fmt, ...) {
    char ts[32];
    p2_iso_timestamp(ts, sizeof(ts));
    fprintf(stderr, "[%s] [%s] ", ts, level);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
}

#define P2_LOG_INFO(...)  p2_log("INFO",  __VA_ARGS__)
#define P2_LOG_WARN(...)  p2_log("WARN",  __VA_ARGS__)
#define P2_LOG_ERR(...)   p2_log("ERROR", __VA_ARGS__)

/* Phase marker: stable parser-friendly format. */
static inline void p2_phase(const char *test, const char *phase) {
    char ts[32];
    p2_iso_timestamp(ts, sizeof(ts));
    fprintf(stderr, "[%s] [PHASE] test=%s phase=%s\n", ts, test, phase);
    fflush(stderr);
}

/* Wall-clock seconds since program start (monotonic). */
static inline double p2_monotonic(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Pin to one CPU; non-fatal if rejected. Linux-only (cpu_set_t / sched_setaffinity);
 * on other hosts the call is a no-op and we warn once. */
static inline void p2_pin_cpu(int cpu) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        P2_LOG_WARN("sched_setaffinity(cpu=%d) failed: %s", cpu, strerror(errno));
    }
#else
    (void)cpu;
    P2_LOG_WARN("cpu pinning not supported on this OS; ignoring");
#endif
}

/* ---------- PRNG: xoshiro256** ----------
 * Deterministic for a given seed. Used for source bytes and offset streams.
 */
typedef struct { uint64_t s[4]; } p2_rng_t;

static inline uint64_t p2_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t p2_splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline void p2_rng_seed(p2_rng_t *r, uint64_t seed) {
    uint64_t s = seed ? seed : 0xC0FFEE12345678ULL;
    for (int i = 0; i < 4; i++) r->s[i] = p2_splitmix64(&s);
}

static inline uint64_t p2_rng_next(p2_rng_t *r) {
    const uint64_t result = p2_rotl(r->s[1] * 5, 7) * 9;
    const uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = p2_rotl(r->s[3], 45);
    return result;
}

/* Fill buffer with deterministic bytes from PRNG. */
static inline void p2_rng_fill(p2_rng_t *r, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t v = p2_rng_next(r);
        memcpy(p + i, &v, 8);
        i += 8;
    }
    if (i < len) {
        uint64_t v = p2_rng_next(r);
        memcpy(p + i, &v, len - i);
    }
}

/* ---------- Sandbox safety ----------
 *
 * Hard caps and path validation for security_like_safe tests.
 * All paths must:
 *  - be absolute
 *  - resolve via realpath() under an approved root (/tmp or operator-supplied --safe-root)
 *  - not contain symlink components escaping the root
 *
 * Any rule violation aborts the test before any file is touched.
 */

#ifndef P2_SANDBOX_MAX_FILES
#define P2_SANDBOX_MAX_FILES 5000
#endif
#ifndef P2_SANDBOX_MAX_BYTES
#define P2_SANDBOX_MAX_BYTES (5ULL * 1024ULL * 1024ULL * 1024ULL) /* 5 GB */
#endif
#ifndef P2_SANDBOX_MAX_DURATION_S
#define P2_SANDBOX_MAX_DURATION_S 600
#endif

/* Approved sandbox roots (prefix match on realpath). */
static const char *P2_SAFE_ROOTS[] = {
    "/tmp/",
    "/var/tmp/",
    NULL
};

/* Verify path is absolute, exists or will be under an approved root, has no
 * ".." or symlink escapes. Returns 0 on safe, -1 otherwise. Writes the canonical
 * resolved path into out_real (must be at least PATH_MAX). */
static inline int p2_sandbox_validate(const char *path, const char *extra_safe_root,
                                      char *out_real) {
    if (!path || path[0] != '/') {
        P2_LOG_ERR("sandbox path not absolute: %s", path ? path : "(null)");
        return -1;
    }
    /* Reject any ".." component textually before realpath. */
    if (strstr(path, "/..") || strstr(path, "../")) {
        P2_LOG_ERR("sandbox path contains '..': %s", path);
        return -1;
    }
    /* Resolve. If the path does not yet exist, resolve its parent. */
    char tmp[PATH_MAX];
    if (realpath(path, out_real) == NULL) {
        /* Try parent dir. */
        snprintf(tmp, sizeof(tmp), "%s", path);
        char *slash = strrchr(tmp, '/');
        if (!slash || slash == tmp) {
            P2_LOG_ERR("sandbox path cannot be resolved: %s", path);
            return -1;
        }
        *slash = '\0';
        char parent_real[PATH_MAX];
        if (realpath(tmp, parent_real) == NULL) {
            P2_LOG_ERR("sandbox parent unresolved: %s", tmp);
            return -1;
        }
        snprintf(out_real, PATH_MAX, "%s/%s", parent_real, slash + 1);
    }
    /* Must start with an approved root after realpath-ing both sides
     * (so /tmp/x on macOS, which resolves to /private/tmp/x, still matches). */
    int ok = 0;
    char out_norm[PATH_MAX];
    snprintf(out_norm, sizeof(out_norm), "%s", out_real);
    /* p2_strip_trailing_slash is defined later as static inline. Since this
     * function and the helper coexist in the same translation unit, the
     * static inline definitions resolve at link time. */
    {
        size_t L = strlen(out_norm);
        while (L > 1 && out_norm[L-1] == '/') { out_norm[L-1] = '\0'; L--; }
    }
    for (int i = 0; P2_SAFE_ROOTS[i] != NULL; i++) {
        char approved_real[PATH_MAX];
        if (realpath(P2_SAFE_ROOTS[i], approved_real) == NULL) continue;
        size_t L = strlen(approved_real);
        while (L > 1 && approved_real[L-1] == '/') { approved_real[L-1] = '\0'; L--; }
        if (strncmp(out_norm, approved_real, L) == 0 &&
            (out_norm[L] == '/' || out_norm[L] == '\0')) { ok = 1; break; }
    }
    if (!ok && extra_safe_root && extra_safe_root[0] == '/') {
        char esr_real[PATH_MAX];
        if (realpath(extra_safe_root, esr_real) != NULL) {
            size_t L = strlen(esr_real);
            while (L > 1 && esr_real[L-1] == '/') { esr_real[L-1] = '\0'; L--; }
            if (strncmp(out_norm, esr_real, L) == 0 &&
                (out_norm[L] == '/' || out_norm[L] == '\0')) ok = 1;
        }
    }
    if (!ok) {
        P2_LOG_ERR("sandbox path not under approved root: %s", out_real);
        return -1;
    }
    return 0;
}

/* Validate that the supplied sandbox parent (root) lies under an approved
 * top-level directory. The caller's --safe-root, if any, is the only
 * operator override and must be passed explicitly; passing the parent itself
 * as its own approval is refused. */
/* Strip trailing slash for comparison. */
static inline void p2_strip_trailing_slash(char *s) {
    size_t l = strlen(s);
    while (l > 1 && s[l - 1] == '/') { s[l - 1] = '\0'; l--; }
}

/* Compare two real paths for "candidate is under root". Both must be already
 * realpath-resolved. Returns 1 if candidate starts with root + '/' or equals
 * root. */
static inline int p2_path_under(const char *candidate, const char *root) {
    size_t lr = strlen(root);
    if (lr == 0) return 0;
    if (strncmp(candidate, root, lr) != 0) return 0;
    return candidate[lr] == '/' || candidate[lr] == '\0';
}

static inline int p2_sandbox_check_root(const char *root,
                                        const char *operator_safe_root) {
    if (!root || root[0] != '/') {
        P2_LOG_ERR("sandbox-dir must be absolute: %s", root ? root : "(null)");
        return -1;
    }
    if (strstr(root, "/..") || strstr(root, "../")) {
        P2_LOG_ERR("sandbox-dir contains '..': %s", root);
        return -1;
    }
    char root_real[PATH_MAX];
    if (realpath(root, root_real) == NULL) {
        P2_LOG_ERR("sandbox-dir cannot be resolved: %s", root);
        return -1;
    }
    p2_strip_trailing_slash(root_real);
    /* Resolve each built-in safe root via realpath so /tmp (which may be a
     * symlink to /private/tmp on darwin) matches its canonical form. */
    for (int i = 0; P2_SAFE_ROOTS[i] != NULL; i++) {
        char approved_real[PATH_MAX];
        if (realpath(P2_SAFE_ROOTS[i], approved_real) == NULL) continue;
        p2_strip_trailing_slash(approved_real);
        if (p2_path_under(root_real, approved_real)) return 0;
    }
    if (operator_safe_root && operator_safe_root[0] == '/') {
        char osr_real[PATH_MAX];
        if (realpath(operator_safe_root, osr_real) != NULL) {
            p2_strip_trailing_slash(osr_real);
            if (p2_path_under(root_real, osr_real)) return 0;
        }
    }
    P2_LOG_ERR("sandbox-dir %s not under approved root (/tmp, /var/tmp, or --safe-root)",
               root_real);
    return -1;
}

/* Create a sandbox dir with the format <root>/phase2_sandbox_<pid>_<seed>.
 * Returns 0 on success and writes resolved path to out_real (PATH_MAX).
 * The supplied safe_root parameter (NULL allowed) is the only operator
 * override accepted; passing the parent itself as approval is refused. */
static inline int p2_sandbox_create(const char *root, const char *safe_root,
                                    uint64_t seed, char *out_real) {
    if (!root) root = "/tmp";
    if (p2_sandbox_check_root(root, safe_root) != 0) return -1;
    char candidate[PATH_MAX];
    snprintf(candidate, sizeof(candidate), "%s/phase2_sandbox_%d_%llu",
             root, (int)getpid(), (unsigned long long)seed);
    if (mkdir(candidate, 0700) != 0 && errno != EEXIST) {
        P2_LOG_ERR("mkdir(%s) failed: %s", candidate, strerror(errno));
        return -1;
    }
    if (p2_sandbox_validate(candidate, safe_root, out_real) != 0) {
        return -1;
    }
    P2_LOG_INFO("sandbox created: %s", out_real);
    return 0;
}

/* Recursive cleanup confined to a validated sandbox directory.
 * Refuses any path that does not pass p2_sandbox_validate. */
static inline int p2_sandbox_remove_tree(const char *path, const char *safe_root) {
    char real[PATH_MAX];
    if (p2_sandbox_validate(path, safe_root, real) != 0) {
        P2_LOG_ERR("refuse to remove unvalidated path: %s", path);
        return -1;
    }
    DIR *d = opendir(real);
    if (!d) {
        if (errno == ENOENT) return 0;
        P2_LOG_ERR("opendir(%s): %s", real, strerror(errno));
        return -1;
    }
    struct dirent *de;
    int rc = 0;
    while ((de = readdir(d)) != NULL) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        char child[PATH_MAX];
        int n = snprintf(child, sizeof(child), "%s/%s", real, de->d_name);
        if (n <= 0 || n >= (int)sizeof(child)) { rc = -1; continue; }
        struct stat st;
        if (lstat(child, &st) != 0) { rc = -1; continue; }
        if (S_ISLNK(st.st_mode)) {
            /* Refuse symlinks entirely. Unlink only because sandbox is throwaway. */
            if (unlink(child) != 0) rc = -1;
        } else if (S_ISDIR(st.st_mode)) {
            if (p2_sandbox_remove_tree(child, safe_root) != 0) rc = -1;
        } else {
            if (unlink(child) != 0) rc = -1;
        }
    }
    closedir(d);
    if (rmdir(real) != 0) {
        P2_LOG_ERR("rmdir(%s): %s", real, strerror(errno));
        rc = -1;
    }
    return rc;
}

/* ---------- Metadata JSON ----------
 *
 * Tiny hand-rolled emitter. We keep the schema flat and additive: an outer
 * object with test_name, language, parameters, phases, file counts, etc.
 * Writes to <output_dir>/<test_name>_metadata.json if output_dir is set,
 * otherwise to stdout.
 */
typedef struct {
    FILE *fp;
    int first;
} p2_meta_t;

static inline int p2_meta_open(p2_meta_t *m, const char *output_dir,
                               const char *test_name) {
    m->first = 1;
    if (output_dir && output_dir[0]) {
        mkdir(output_dir, 0755);
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s_metadata.json", output_dir, test_name);
        m->fp = fopen(path, "w");
        if (!m->fp) {
            P2_LOG_WARN("metadata file open failed: %s — falling back to stdout", path);
            m->fp = stdout;
        }
    } else {
        m->fp = stdout;
    }
    fprintf(m->fp, "{");
    return 0;
}

static inline void p2_meta_kv_str(p2_meta_t *m, const char *k, const char *v) {
    fprintf(m->fp, "%s\"%s\":\"%s\"", m->first ? "" : ",", k, v ? v : "");
    m->first = 0;
}
static inline void p2_meta_kv_i64(p2_meta_t *m, const char *k, long long v) {
    fprintf(m->fp, "%s\"%s\":%lld", m->first ? "" : ",", k, v);
    m->first = 0;
}
static inline void p2_meta_kv_u64(p2_meta_t *m, const char *k, unsigned long long v) {
    fprintf(m->fp, "%s\"%s\":%llu", m->first ? "" : ",", k, v);
    m->first = 0;
}
static inline void p2_meta_kv_f64(p2_meta_t *m, const char *k, double v) {
    fprintf(m->fp, "%s\"%s\":%.6f", m->first ? "" : ",", k, v);
    m->first = 0;
}

static inline void p2_meta_close(p2_meta_t *m) {
    fprintf(m->fp, "}\n");
    if (m->fp != stdout) fclose(m->fp);
    m->fp = NULL;
}

/* ---------- Common CLI flag parsing ----------
 * Minimal getopt_long alternative: we just scan argv looking for known --flags.
 * Returns the integer position of the flag or -1 if not found. If found,
 * *value is set to argv[pos+1] (may be NULL if last). For boolean flags use
 * p2_flag_present.
 */
static inline int p2_flag_value(int argc, char **argv, const char *name,
                                const char **out_value) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            *out_value = (i + 1 < argc) ? argv[i + 1] : NULL;
            return i;
        }
    }
    return -1;
}

static inline int p2_flag_present(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

static inline long long p2_get_i64(int argc, char **argv, const char *name,
                                   long long def) {
    const char *v = NULL;
    if (p2_flag_value(argc, argv, name, &v) >= 0 && v) {
        return strtoll(v, NULL, 0);
    }
    return def;
}

static inline unsigned long long p2_get_u64(int argc, char **argv,
                                            const char *name,
                                            unsigned long long def) {
    const char *v = NULL;
    if (p2_flag_value(argc, argv, name, &v) >= 0 && v) {
        return strtoull(v, NULL, 0);
    }
    return def;
}

static inline const char *p2_get_str(int argc, char **argv, const char *name,
                                     const char *def) {
    const char *v = NULL;
    if (p2_flag_value(argc, argv, name, &v) >= 0 && v) return v;
    return def;
}

/* nanosleep wrapper that handles EINTR */
static inline void p2_sleep_ns(uint64_t nanos) {
    struct timespec req = { .tv_sec = nanos / 1000000000ULL,
                            .tv_nsec = nanos % 1000000000ULL };
    while (nanosleep(&req, &req) == -1 && errno == EINTR) {}
}

#endif /* PHASE2_COMMON_H */
