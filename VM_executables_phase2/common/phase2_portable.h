/* phase2_portable.h
 * Compatibility shims for building on Linux (target) and macOS (smoke build).
 * Real measurements must run on the Linux guest; macOS builds are validation only.
 */
#ifndef PHASE2_PORTABLE_H
#define PHASE2_PORTABLE_H

#include <sys/mman.h>

#ifndef MAP_POPULATE
#define MAP_POPULATE 0   /* Linux-only; no-op elsewhere */
#endif
#ifndef MAP_ANONYMOUS
# ifdef MAP_ANON
#  define MAP_ANONYMOUS MAP_ANON
# endif
#endif

#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE 15  /* Linux value; ignored if kernel doesn't support */
#endif
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4
#endif
#ifndef MADV_RANDOM
#define MADV_RANDOM 1
#endif
#ifndef MADV_SEQUENTIAL
#define MADV_SEQUENTIAL 2
#endif
#ifndef MADV_WILLNEED
#define MADV_WILLNEED 3
#endif

/* madvise() that swallows EINVAL on hosts that don't recognize the flag. */
static inline int p2_madvise(void *addr, size_t len, int advice) {
    if (madvise(addr, len, advice) == 0) return 0;
    /* On macOS/BSDs unknown advice returns EINVAL; treat as soft no-op. */
    return 0;
}

/* mlock attempt; warns and continues on failure (RLIMIT_MEMLOCK or perms). */
#include <errno.h>
#include <string.h>
#include <stdio.h>
static inline int p2_mlock_soft(void *addr, size_t len) {
    if (mlock(addr, len) == 0) return 0;
    fprintf(stderr, "[WARN] mlock(%zu) failed: %s (continuing)\n", len, strerror(errno));
    return -1;
}

#endif
