/* kernel_brackets_enum_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  BALANCED-PARENTHESES ENUMERATOR:  emit every well-formed string of n pairs
 * ============================================================================
 *
 *  DWARF   : Backtrack / Branch-and-Bound (D11)  (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL (write-visible)              (first-division memory-signature label)
 *  PURPOSE : Visibility source (a) MATERIALIZE THE OUTPUT. This is the SECOND
 *            instance of that mechanism (the sibling nqueens_enum is the first):
 *            deliberately the SAME write mechanism -- append one full solution
 *            per search leaf into a big mmap'd buffer -- on a DIFFERENT
 *            combinatorial object (Catalan strings here vs permutation boards
 *            there). Showing the tell survives the change of object is the point:
 *            it is the MECHANISM (bulk sequential append of the whole solution
 *            set) that is visible, not the particular puzzle.
 *
 *  PICTURE (top view):  the '(' / ')' choice tree, one leaf -> one appended row.
 *
 *        depth 0            "("                 append buffer (grows down):
 *                          /   \                 +--------------------------+
 *        depth 1        "(("   "()"              | ( ( ( ) ) )   <- leaf 1  |
 *                       / \      \               | ( ( ) ( ) )   <- leaf 2  |
 *      ...  place '(' while open<n  ...          | ( ( ) ) ( )   <- leaf 3  |
 *           place ')' while close<open           | ( ) ( ( ) )   <- ...     |
 *                        \                        | ( ) ( ) ( )              |
 *      len==2n  =>  a COMPLETE balanced string    +--------------------------+
 *                   is APPENDED (2n bytes)         one dense write front,
 *                                                  strictly sequential.
 *
 *  ALGORITHM (depth-first backtracking, no recursion of the buffer):
 *      1. Build a length-2n scratch string by DFS. At each step:
 *           - place '(' if open  < n            (still have opens to spend);
 *           - place ')' if close < open         (keep every prefix balanced).
 *         These two pruning rules are the branch-and-bound: a partial string is
 *         only ever extended into arrangements that can still complete, so the
 *         tree has exactly Catalan(n) leaves and no dead ends are enumerated.
 *      2. When length reaches 2n the scratch holds one complete well-formed
 *         sequence -> APPEND its 2n bytes to the mmap'd output buffer (bulk
 *         sequential append, one row per solution). Never mutate an already
 *         written row; the buffer only grows.
 *      3. The number of leaves (= strings stored) is the Catalan number
 *              count(n) = Catalan(n) = binomial(2n, n) / (n + 1).
 *         n=12 -> 208012 strings;  n=13 -> 742900 strings.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      One large, append-only region filled strictly front-to-back: solution k
 *      lands at byte offset k*(2n) and is never revisited. That is a single dense
 *      sequential write stream whose total size equals the full solution set --
 *      the same materialize-the-output tell as nqueens_enum, reproduced on a
 *      Catalan object. The DFS scratch and counters are tiny; the visible mass
 *      is the bulk append of every string.
 *
 *  SIZING / SAFETY:
 *      The output buffer capacity is derived from --max-mb (cap = max_bytes /
 *      (2n) strings). If enumeration would exceed that capacity we STOP with a
 *      clear error asking for a larger --max-mb -- we never silently truncate,
 *      because strings_stored must equal Catalan(n) for the run to be valid.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  (The optional --dump-strings path is a verifier-only convenience OUTSIDE the
 *  measured loop; the measured workload writes only anonymous mmap memory.)
 *  Signature family: KERNEL. Dwarf: Backtrack / Branch-and-Bound. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (re-enumerate, append every string) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_brackets_enum_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign balanced-parentheses enumerator; backtracking kernel)\n"
"  --n N                 Pairs of parentheses (default 13; clamped to [1,15])\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (unused by the search; recorded for parity) (default 42)\n"
"  --max-mb N            Hard cap on output-buffer bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dump-strings PATH   Verifier-only: after measuring, write the raw buffer to PATH\n"
"                        (benign plain file, NOT part of the measured loop)\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* ---------------------------------------------------------------------------
 * The append-only output buffer. One flat mmap'd byte array; solutions are
 * appended strictly front-to-back (solution k at offset k*slen). `used` is the
 * next free byte; `overflow` trips if the search would exceed capacity.
 * --------------------------------------------------------------------------- */
typedef struct {
    char  *buf;        /* flat output byte array (mmap'd)     */
    size_t used;       /* bytes written so far this pass      */
    size_t cap_bytes;  /* buffer capacity in bytes            */
    size_t slen;       /* length of one solution = 2n bytes   */
    uint64_t stored;   /* number of complete strings appended */
    int    overflow;   /* set if capacity was exceeded        */
} Sink;

/* Append one complete solution (the scratch string, slen bytes). Bulk
 * sequential write: memcpy the whole row at the current tail, advance. */
static inline void sink_append(Sink *s, const char *sol) {
    if (s->used + s->slen > s->cap_bytes) { s->overflow = 1; return; }
    memcpy(s->buf + s->used, sol, s->slen);
    s->used += s->slen;
    s->stored++;
}

/* Depth-first backtracking enumeration.
 *   sol   : length-2n scratch string being built.
 *   pos   : current length of the partial string (index of next char).
 *   open  : number of '(' placed so far  (0..n).
 *   close : number of ')' placed so far  (0..open).
 * Pruning: place '(' only while open < n; place ')' only while close < open.
 * At pos == 2n the string is a complete balanced sequence -> append it. */
static void enumerate(Sink *s, char *sol, int pos, int open, int close, int n) {
    if (s->overflow) return;                 /* stop early once capacity is gone */
    if (pos == 2 * n) {                      /* leaf: a well-formed string       */
        sink_append(s, sol);
        return;
    }
    if (open < n) {                          /* branch 1: open a new pair        */
        sol[pos] = '(';
        enumerate(s, sol, pos + 1, open + 1, close, n);
    }
    if (close < open) {                      /* branch 2: close an open pair     */
        sol[pos] = ')';
        enumerate(s, sol, pos + 1, open, close + 1, n);
    }
}

/* Run one full enumeration pass: reset the sink and re-fill the buffer with
 * every well-formed string. Returns the number of strings stored this pass. */
static uint64_t enumerate_all(Sink *s, char *sol, int n) {
    s->used = 0; s->stored = 0; s->overflow = 0;
    enumerate(s, sol, 0, 0, 0, n);
    return s->stored;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long n_pairs    = p2_get_i64(argc, argv, "--n", 13);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dump     = p2_get_str(argc, argv, "--dump-strings", NULL);

    /* Clamp n to [1, 15] (per spec). 15 pairs -> Catalan(15) = 9694845 strings
     * of 30 bytes ~ 277 MiB, comfortably inside the default --max-mb. */
    if (n_pairs < 1)  { P2_LOG_WARN("n %lld below 1; clamping to 1", n_pairs);  n_pairs = 1; }
    if (n_pairs > 15) { P2_LOG_WARN("n %lld above 15; clamping to 15", n_pairs); n_pairs = 15; }
    int n = (int)n_pairs;
    size_t slen = (size_t)2 * (size_t)n;            /* bytes per stored string */

    /* Capacity from --max-mb: how many whole strings fit. cap must be >= 1. */
    size_t max_bytes = (size_t)max_mb * 1024ULL * 1024ULL;
    size_t cap_strings = max_bytes / slen;          /* whole strings that fit  */
    if (cap_strings == 0) {
        P2_LOG_ERR("--max-mb %lld too small to hold even one %zu-byte string; raise --max-mb",
                   max_mb, slen);
        return 2;
    }
    size_t cap_bytes = cap_strings * slen;          /* usable buffer bytes     */

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D11 Backtrack/Branch-and-Bound");
    p2_meta_kv_str(&m, "family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "scheme", "balanced-parentheses enumerator; DFS backtracking, append one full string per leaf");
    p2_meta_kv_str(&m, "visibility_source", "(a) materialize the output (2nd instance; same mechanism as nqueens_enum, Catalan object)");
    p2_meta_kv_str(&m, "count_formula", "strings_stored == Catalan(n) == binomial(2n,n)/(n+1)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "n", n);
    p2_meta_kv_u64(&m, "string_len_bytes", (unsigned long long)slen);
    p2_meta_kv_u64(&m, "buffer_capacity_bytes", (unsigned long long)cap_bytes);
    p2_meta_kv_u64(&m, "buffer_capacity_strings", (unsigned long long)cap_strings);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The output buffer is the dominant, visible region -> mmap + mlock it. It
     * is refilled front-to-back every measure pass (the workload's signature
     * write: a bulk sequential append of the whole solution set). */
    char *buf = (char *)mmap(NULL, cap_bytes, PROT_READ | PROT_WRITE,
                             MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", cap_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(buf, cap_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(buf, cap_bytes);

    /* Tiny DFS scratch string (slen bytes); this is not the visible mass. */
    char *sol = (char *)malloc(slen);
    if (!sol) {
        munmap(buf, cap_bytes);
        P2_LOG_ERR("malloc(%zu) failed", slen);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    Sink s = { buf, 0, cap_bytes, slen, 0, 0 };
    (void)seed;                                     /* search is deterministic  */

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* One warm-up enumeration to fault the buffer in and confirm it fits. If the
     * full solution set does not fit, STOP now with a clear error (never
     * truncate): strings_stored must equal Catalan(n) for a valid run. */
    uint64_t stored = enumerate_all(&s, sol, n);
    if (s.overflow) {
        free(sol); munmap(buf, cap_bytes);
        P2_LOG_ERR("output buffer too small for n=%d (capacity %zu strings, "
                   "%llu appended before overflow); Catalan(%d) exceeds it -- "
                   "raise --max-mb (current %lld)",
                   n, cap_strings, (unsigned long long)stored, n, max_mb);
        p2_meta_kv_str(&m, "status", "buffer_overflow"); p2_meta_close(&m); return 2;
    }
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One measure pass: re-run the full enumeration, re-appending every
         * well-formed string front-to-back into the buffer (bulk sequential
         * append of the whole solution set). Same footprint every pass -> strong
         * temporal periodicity, like the Jacobi sweep revisiting its grid. */
        stored = enumerate_all(&s, sol, n);
        if (s.overflow) {                           /* cannot happen after warmup */
            free(sol); munmap(buf, cap_bytes);
            P2_LOG_ERR("unexpected overflow during measure at n=%d -- raise --max-mb", n);
            p2_meta_kv_str(&m, "status", "buffer_overflow"); p2_meta_close(&m); return 2;
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Verifier-only side channel (OUTSIDE the measured window): dump the raw
     * buffer -- exactly `used` bytes = stored * slen -- to a plain file so an
     * independent checker can validate every string. Benign local write; not
     * part of the workload's measured memory signature. */
    int dump_ok = -1;
    if (dump) {
        FILE *df = fopen(dump, "wb");
        if (!df) {
            P2_LOG_WARN("dump-strings open failed: %s (%s)", dump, strerror(errno));
        } else {
            size_t wrote = fwrite(s.buf, 1, s.used, df);
            if (fclose(df) != 0 || wrote != s.used) {
                P2_LOG_WARN("dump-strings write short/failed: %s", dump);
                dump_ok = 0;
            } else {
                dump_ok = 1;
                P2_LOG_INFO("dumped %zu strings (%zu bytes) to %s",
                            (size_t)s.stored, s.used, dump);
            }
        }
    }

    /* Prevent dead-code elimination of the buffer: sample a live byte. For n>=1
     * the first stored byte is always '(' (offset 0). */
    volatile char sink = s.buf[0];

    uint64_t strings_stored = s.stored;
    size_t   strings_bytes  = s.used;

    free(sol);
    munmap(buf, cap_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "strings_stored", strings_stored);
    p2_meta_kv_u64(&m, "strings_bytes", (unsigned long long)strings_bytes);
    p2_meta_kv_i64(&m, "first_byte", (int)sink);
    if (dump) p2_meta_kv_i64(&m, "dump_ok", dump_ok);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: bulk sequential append of the whole solution set "
                   "(materialize-the-output, 2nd instance vs nqueens_enum); strings_stored must equal Catalan(n)");
    p2_meta_close(&m);
    return 0;
}
