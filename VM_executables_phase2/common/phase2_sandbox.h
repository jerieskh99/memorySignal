/* phase2_sandbox.h
 *
 * Shared safe-sandbox helpers for security_like_safe tests.
 *
 *   - Generate N disposable files inside a validated sandbox dir.
 *   - Reversible XOR transform with a fixed 32-byte key (no real encryption).
 *   - Hard caps enforced before any file is touched.
 *   - Cleanup confined to the validated sandbox dir.
 *
 * Forbids:
 *   - Touching anything outside the sandbox.
 *   - Following symlinks.
 *   - File names containing '/' or '..'.
 *   - Real cryptographic operations, persistence, network, exec().
 */
#ifndef PHASE2_SANDBOX_H
#define PHASE2_SANDBOX_H

#include "phase2_common.h"
#include "phase2_portable.h"

#define P2_XOR_KEY_LEN 32

typedef struct {
    char  root[PATH_MAX];      /* validated sandbox real path */
    int   num_files;           /* generated file count */
    size_t file_size;          /* generated file size in bytes */
    uint8_t key[P2_XOR_KEY_LEN];
    uint64_t seed;
    uint64_t bytes_written;
    uint64_t bytes_read;
    uint64_t files_created;
    uint64_t files_renamed;
    int   max_files_cap;       /* hard cap snapshot */
    uint64_t max_bytes_cap;
    int   duration_cap_s;
} p2_sandbox_t;

/* Pre-flight cap check usable from dry-run paths too. Returns 0 if all
 * caps are satisfied, -1 otherwise (with a logged error). */
static inline int p2_sandbox_check_caps(int num_files, size_t file_size, int duration_s) {
    if (num_files <= 0 || num_files > P2_SANDBOX_MAX_FILES) {
        P2_LOG_ERR("num_files=%d outside [1..%d]", num_files, P2_SANDBOX_MAX_FILES);
        return -1;
    }
    uint64_t total = (uint64_t)num_files * (uint64_t)file_size;
    if (total > P2_SANDBOX_MAX_BYTES) {
        P2_LOG_ERR("total bytes %llu exceeds cap %llu",
                   (unsigned long long)total, (unsigned long long)P2_SANDBOX_MAX_BYTES);
        return -1;
    }
    if (duration_s > P2_SANDBOX_MAX_DURATION_S) {
        P2_LOG_ERR("duration %d exceeds cap %d", duration_s, P2_SANDBOX_MAX_DURATION_S);
        return -1;
    }
    return 0;
}

/* Derive a fixed 32-byte XOR key from the seed (deterministic). */
static inline void p2_sandbox_derive_key(uint64_t seed, uint8_t key[P2_XOR_KEY_LEN]) {
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    p2_rng_fill(&rng, key, P2_XOR_KEY_LEN);
}

/* Validate caps and create files; if dry_run only logs intended actions.
 * safe_root (NULL allowed) is the operator-supplied --safe-root extension to
 * the approved root list (/tmp, /var/tmp). Passing sandbox_root as its own
 * safe_root is refused inside p2_sandbox_check_root. */
static inline int p2_sandbox_init(p2_sandbox_t *s, const char *sandbox_root,
                                  const char *safe_root,
                                  int num_files, size_t file_size,
                                  uint64_t seed, int dry_run) {
    memset(s, 0, sizeof(*s));
    s->seed = seed;
    s->num_files = num_files;
    s->file_size = file_size;
    s->max_files_cap = P2_SANDBOX_MAX_FILES;
    s->max_bytes_cap = P2_SANDBOX_MAX_BYTES;
    s->duration_cap_s = P2_SANDBOX_MAX_DURATION_S;
    if (num_files <= 0 || num_files > P2_SANDBOX_MAX_FILES) {
        P2_LOG_ERR("num_files=%d outside [1..%d]", num_files, P2_SANDBOX_MAX_FILES);
        return -1;
    }
    uint64_t total = (uint64_t)num_files * (uint64_t)file_size;
    if (total > P2_SANDBOX_MAX_BYTES) {
        P2_LOG_ERR("total bytes %llu exceeds cap %llu",
                   (unsigned long long)total,
                   (unsigned long long)P2_SANDBOX_MAX_BYTES);
        return -1;
    }
    if (p2_sandbox_create(sandbox_root ? sandbox_root : "/tmp",
                          safe_root, seed, s->root) != 0) return -1;
    p2_sandbox_derive_key(seed, s->key);

    if (dry_run) {
        P2_LOG_INFO("dry-run: would create %d files of %zu bytes in %s",
                    num_files, file_size, s->root);
        return 0;
    }
    return 0;
}

/* Generate file <root>/file_<idx>.dat with deterministic content. */
static inline int p2_sandbox_generate_files(p2_sandbox_t *s, const char *suffix_ext,
                                            int subdirs) {
    if (subdirs < 0) subdirs = 0;
    if (subdirs > 0) {
        for (int d = 0; d < subdirs; d++) {
            char dpath[PATH_MAX];
            snprintf(dpath, sizeof(dpath), "%s/d_%03d", s->root, d);
            if (mkdir(dpath, 0700) != 0 && errno != EEXIST) {
                P2_LOG_ERR("mkdir(%s): %s", dpath, strerror(errno));
                return -1;
            }
        }
    }
    p2_rng_t rng; p2_rng_seed(&rng, s->seed ^ 0xA5A5A5A5);
    uint8_t *chunk = (uint8_t *)malloc(65536);
    if (!chunk) return -1;
    for (int i = 0; i < s->num_files; i++) {
        char path[PATH_MAX];
        if (subdirs > 0) {
            snprintf(path, sizeof(path), "%s/d_%03d/file_%05d.%s",
                     s->root, i % subdirs, i, suffix_ext ? suffix_ext : "dat");
        } else {
            snprintf(path, sizeof(path), "%s/file_%05d.%s",
                     s->root, i, suffix_ext ? suffix_ext : "dat");
        }
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
        if (fd < 0) {
            P2_LOG_ERR("open(%s): %s", path, strerror(errno));
            free(chunk); return -1;
        }
        size_t remaining = s->file_size;
        while (remaining > 0) {
            size_t n = remaining > 65536 ? 65536 : remaining;
            p2_rng_fill(&rng, chunk, n);
            ssize_t w = write(fd, chunk, n);
            if (w <= 0) {
                P2_LOG_ERR("write %s: %s", path, strerror(errno));
                close(fd); free(chunk); return -1;
            }
            remaining -= (size_t)w;
            s->bytes_written += (uint64_t)w;
        }
        close(fd);
        s->files_created++;
    }
    free(chunk);
    return 0;
}

/* Reversible XOR transform on a buffer with key (cycled). */
static inline void p2_sandbox_xor(uint8_t *buf, size_t n, const uint8_t *key) {
    for (size_t i = 0; i < n; i++) buf[i] ^= key[i % P2_XOR_KEY_LEN];
}

/* Phase helpers used by sandbox_ransom_* tests. Operate on one file. */
static inline int p2_sandbox_phase_read(p2_sandbox_t *s, const char *path,
                                        uint8_t *buf, size_t bufsz) {
    int fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) { P2_LOG_ERR("open(%s): %s", path, strerror(errno)); return -1; }
    size_t total = 0;
    while (total < bufsz) {
        ssize_t r = read(fd, buf + total, bufsz - total);
        if (r <= 0) break;
        total += (size_t)r;
        s->bytes_read += (uint64_t)r;
    }
    close(fd);
    return 0;
}

static inline int p2_sandbox_phase_write(p2_sandbox_t *s, const char *path,
                                         const uint8_t *buf, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
    if (fd < 0) { P2_LOG_ERR("open(%s): %s", path, strerror(errno)); return -1; }
    size_t written = 0;
    while (written < n) {
        ssize_t w = write(fd, buf + written, n - written);
        if (w <= 0) { close(fd); return -1; }
        written += (size_t)w;
        s->bytes_written += (uint64_t)w;
    }
    close(fd);
    return 0;
}

static inline int p2_sandbox_phase_rename(p2_sandbox_t *s, const char *src,
                                          const char *suffix) {
    char dst[PATH_MAX];
    snprintf(dst, sizeof(dst), "%s.%s", src, suffix ? suffix : "simx");
    /* Refuse outside sandbox. */
    char dst_real[PATH_MAX];
    if (p2_sandbox_validate(dst, s->root, dst_real) != 0) return -1;
    if (rename(src, dst) != 0) {
        P2_LOG_ERR("rename(%s -> %s): %s", src, dst, strerror(errno));
        return -1;
    }
    s->files_renamed++;
    return 0;
}

static inline int p2_sandbox_finalize(p2_sandbox_t *s, int do_cleanup) {
    if (do_cleanup) {
        int rc = p2_sandbox_remove_tree(s->root, s->root);
        if (rc != 0) P2_LOG_WARN("sandbox cleanup encountered errors");
        return rc;
    }
    return 0;
}

#endif
