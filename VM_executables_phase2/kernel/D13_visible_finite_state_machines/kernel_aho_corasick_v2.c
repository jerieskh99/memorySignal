/* kernel_aho_corasick_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Aho-Corasick multi-pattern string search. Dwarf D13 (Finite State Machines),
 * of the Berkeley 13-motif corpus. Aho-Corasick locates ALL occurrences of a
 * whole DICTIONARY of P patterns in one text in a single left-to-right pass. It
 * does this by compiling the patterns into one finite-state automaton: a trie of
 * the patterns (the goto function) augmented with FAILURE links (where to jump
 * when the next byte does not extend the current match) and OUTPUT sets (which
 * patterns end at each state). This is a multi-pattern search BENCHMARK of the
 * kind used in log processing, not a security tool.
 *
 * PICTURE (top view):
 *
 *   TRIE of patterns {he, she, his, hers}          FAIL links (dashed)
 *   over a small alphabet, plus fail links:
 *                                                   scan text  "ushers"
 *        (root:0)                                     u s h e r s
 *        h/         \s                                      ^ ^ ^ ^
 *      (1)           (3)                              state walks goto; on a
 *    e/  \i        h |                                mismatch it hops a fail
 *   (2)   (6)       (4)  ...- fail ->(1) 'h'          link, never rescanning.
 *  r|      s|      e |
 *  (7)     (8)     (5:"she","he")   <- OUTPUT set     MATCH LIST (append-only):
 *  s|                                                  +--------------------+
 *  (9:"hers")   OUTPUT sets chain via a               | (pos=3, pid=she)   |
 *               dictionary-suffix link.               | (pos=3, pid=he)    |
 *                                                      | (pos=... , ...)    |  ...grows
 *                                                      +--------------------+
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *   BUILD (visible write #1 -- the automaton, a growing structure):
 *     1. Trie / goto: insert each of the P patterns byte by byte, allocating a
 *        fresh state from a flat node pool whenever an edge is missing. Each
 *        state owns a goto row of ALPHA child slots. The state that a pattern
 *        ends on records that pattern id in its output.
 *     2. Fail links by BFS: process states in breadth-first order from the root.
 *        The root's depth-1 children fail to the root; every deeper state fails
 *        to goto(fail(parent), edge). A state also inherits the output of the
 *        state its fail link points to, chained by a dictionary-suffix link so
 *        every pattern ending in the current suffix is enumerable in O(matches).
 *   SCAN (visible write #2 -- the match list, append-only):
 *     3. Walk the text one byte at a time holding the current state s. On byte c:
 *        while goto(s,c) is missing, follow s = fail(s); then s = goto(s,c).
 *        Follow the dictionary-suffix chain from s and APPEND every
 *        (text_position, pattern_id) record it emits into a large match-list
 *        buffer. Re-scan the whole text every pass (re-fill the match list) and
 *        count passes; the match list is the dominant visible-write buffer.
 *
 * ============================================================================
 * MEMORY SIGNATURE (what the host write-signal actually sees) & WHY DISTINCT
 * ============================================================================
 * Two visible write structures, both absent from a plain scalar recogniser:
 *   (1) the BUILD writes an irregular, growing automaton (goto rows + fail + out
 *       + dict-link arrays) into the node pool during warmup; and
 *   (2) the SCAN sequentially APPENDS match records into a large match list on
 *       every measured pass -- a dense, monotonic write front whose length is
 *       data-dependent (it tracks how many matches the text yields).
 * CONTRAST WITH THE QUIET dfa_match CONTROL: a single-pattern DFA recogniser in
 * this same dwarf reads the text and the transition table but writes only ONE
 * scalar (the current state, plus an accept flag) -- it is near-idle on the
 * write-signal. Aho-Corasick shares the finite-state-machine READ character but
 * adds these two large visible WRITES (build the automaton, append the matches),
 * which is exactly the tell that separates the two D13 workloads.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * (The optional --dump-* flags are a one-shot verifier aid OUTSIDE the measured
 * loop; the measured loop never touches a file.)
 * Signature family: KERNEL (write-visible). Dwarf: Finite State Machines.
 * See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + generate text/patterns + BUILD automaton) /
 *         measure (SCAN passes, append matches) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_aho_corasick_v2";

/* One appended match record: where in the text, and which pattern. Packs to
 * 16 bytes; the match list is a flat mmap'd array of these -- the append target
 * that dominates the visible write footprint. */
typedef struct {
    uint64_t pos;    /* text index where the matched pattern ENDS (0-based)   */
    uint32_t pid;    /* pattern id (0..P-1)                                    */
    uint32_t _pad;   /* explicit pad -> deterministic 16-byte record          */
} Match;

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Aho-Corasick multi-pattern search; FSM kernel)\n"
"  --patterns P          Number of dictionary patterns (default 2000)\n"
"  --pattern-len L       Pattern length; upper end of a [max(1,L-2), L] range (default 6)\n"
"  --text-mb M           Text size in MiB, generated once (default 16)\n"
"  --alphabet A          Alphabet size 2..26 (default 4; small = many matches)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes; also caps the match list (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dump-text PATH      (verifier only) write the generated text once, then continue\n"
"  --dump-patterns PATH  (verifier only) write the patterns once (one per line: id byte-values)\n"
"  --dump-matches PATH   (verifier only) write the last pass's matches once (pos pid per line)\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* ---------------------------------------------------------------------------
 * The automaton. All state arrays live in ONE flat mmap'd pool (the BUILD's
 * visible write). For a state s and symbol c in [0, alpha):
 *   gotot[(size_t)s * alpha + c] : child state, or -1 if the edge is absent.
 *   fail[s]  : failure link (where to resume on a mismatch); root's is 0.
 *   out[s]   : the SINGLE pattern id that ends exactly at s, or -1 if none.
 *   dict[s]  : dictionary-suffix link -- the nearest fail-ancestor that is
 *              itself the end of some pattern, or -1. Walking dict from s emits
 *              every pattern that ends at the current text position.
 * State 0 is always the root.
 * --------------------------------------------------------------------------- */
typedef struct {
    int32_t *gotot;   /* nodes_cap * alpha goto rows */
    int32_t *fail;    /* nodes_cap                   */
    int32_t *out;     /* nodes_cap                   */
    int32_t *dict;    /* nodes_cap                   */
    int32_t  nodes;   /* states used so far          */
    int32_t  cap;     /* state capacity              */
    int      alpha;   /* alphabet size               */
} AC;

/* Allocate a fresh empty state (root or a new trie edge target). All ALPHA goto
 * slots start absent (-1); the pool is a fixed mmap'd region so earlier states
 * never move. Returns the new state id, or -1 if the pool is exhausted. */
static int32_t ac_new_state(AC *a) {
    if (a->nodes >= a->cap) return -1;
    int32_t s = a->nodes++;
    int32_t *row = a->gotot + (size_t)s * a->alpha;
    for (int c = 0; c < a->alpha; c++) row[c] = -1;
    a->fail[s] = 0;
    a->out[s]  = -1;
    a->dict[s] = -1;
    return s;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long npat       = p2_get_i64(argc, argv, "--patterns", 2000);
    long long plen       = p2_get_i64(argc, argv, "--pattern-len", 6);
    long long text_mb    = p2_get_i64(argc, argv, "--text-mb", 16);
    long long alpha_ll   = p2_get_i64(argc, argv, "--alphabet", 4);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dump_text     = p2_get_str(argc, argv, "--dump-text", NULL);
    const char *dump_patterns = p2_get_str(argc, argv, "--dump-patterns", NULL);
    const char *dump_matches  = p2_get_str(argc, argv, "--dump-matches", NULL);

    if (npat < 1 || npat > (1LL << 24)) { P2_LOG_ERR("patterns %lld out of range (1..2^24)", npat); return 2; }
    if (plen < 1 || plen > 256) { P2_LOG_ERR("pattern-len %lld out of range (1..256)", plen); return 2; }
    if (alpha_ll < 2 || alpha_ll > 26) { P2_LOG_ERR("alphabet %lld out of range (2..26)", alpha_ll); return 2; }
    if (text_mb < 1 || text_mb > 65536) { P2_LOG_ERR("text-mb %lld out of range (1..65536)", text_mb); return 2; }
    int alpha = (int)alpha_ll;
    size_t P = (size_t)npat;
    /* Pattern lengths span [lmin, lmax] so the trie is bushy, not a single stalk. */
    int lmax = (int)plen;
    int lmin = lmax > 2 ? lmax - 2 : 1;

    /* Text buffer. */
    size_t text_len = (size_t)text_mb * 1024ULL * 1024ULL;

    /* Automaton capacity: at most (sum of pattern lengths + 1) states. We do not
     * know the true count until we build (shared prefixes collapse), so bound it
     * by the worst case P*lmax + 1 and clamp to --max-mb below. */
    size_t states_cap = P * (size_t)lmax + 1;
    /* Per-state cost in the pool: one goto row of `alpha` int32 + fail+out+dict. */
    size_t per_state = ((size_t)alpha + 3) * sizeof(int32_t);
    size_t auto_bytes = states_cap * per_state;

    /* Pattern storage: P patterns, each up to lmax bytes, plus a length per id. */
    size_t pat_store_bytes = P * (size_t)lmax;             /* packed pattern bytes  */
    size_t pat_len_bytes   = P * sizeof(int32_t);          /* per-pattern length    */

    /* Match list gets whatever of --max-mb remains after the fixed structures,
     * so the total footprint never exceeds the cap. */
    size_t cap_bytes = (size_t)max_mb * 1024ULL * 1024ULL;
    size_t fixed_bytes = text_len + auto_bytes + pat_store_bytes + pat_len_bytes;
    if (fixed_bytes + (64ULL * 1024ULL) > cap_bytes) {
        P2_LOG_ERR("text+automaton+patterns %zu B exceed --max-mb %lld (raise --max-mb or shrink inputs)",
                   fixed_bytes, max_mb);
        return 2;
    }
    size_t match_bytes = cap_bytes - fixed_bytes;
    size_t match_cap = match_bytes / sizeof(Match);       /* records that fit      */
    if (match_cap < 1024) { P2_LOG_ERR("match-list capacity %zu too small; raise --max-mb", match_cap); return 2; }
    match_bytes = match_cap * sizeof(Match);
    size_t total_bytes = fixed_bytes + match_bytes;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D13 Finite State Machines");
    p2_meta_kv_str(&m, "scheme", "Aho-Corasick multi-pattern search (build trie+fail-link automaton, scan text, append matches)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "patterns", npat);
    p2_meta_kv_i64(&m, "pattern_len", plen);
    p2_meta_kv_i64(&m, "text_mb", text_mb);
    p2_meta_kv_i64(&m, "alphabet", alpha);
    p2_meta_kv_u64(&m, "text_bytes", text_len);
    p2_meta_kv_u64(&m, "match_list_bytes", match_bytes);
    p2_meta_kv_u64(&m, "match_list_capacity", match_cap);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* ---- mmap the visible-write buffers: the automaton pool and the match list.
     * These two are the workload's signature writes (BUILD + APPEND), so they are
     * the mmap+mlock'd regions. The text and the small pattern tables are read
     * inputs generated once. ---- */
    uint8_t *text   = (uint8_t *)mmap(NULL, text_len, PROT_READ | PROT_WRITE,
                                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    uint8_t *apool  = (uint8_t *)mmap(NULL, auto_bytes, PROT_READ | PROT_WRITE,
                                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    Match  *matches = (Match *)mmap(NULL, match_bytes, PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (text == MAP_FAILED || apool == MAP_FAILED || matches == MAP_FAILED) {
        P2_LOG_ERR("mmap failed (text=%zu automaton=%zu matches=%zu): %s",
                   text_len, auto_bytes, match_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(text, text_len, MADV_SEQUENTIAL);       /* text is streamed L->R */
    p2_madvise(apool, auto_bytes, MADV_NOHUGEPAGE);
    p2_madvise(matches, match_bytes, MADV_SEQUENTIAL); /* match list is appended */
    if (!no_mlock) { p2_mlock_soft(apool, auto_bytes); p2_mlock_soft(matches, match_bytes); }

    /* Pattern tables (plain heap: read-only inputs after generation). */
    uint8_t *pat_bytes = (uint8_t *)malloc(pat_store_bytes ? pat_store_bytes : 1);
    int32_t *pat_len   = (int32_t *)malloc(pat_len_bytes);
    if (!pat_bytes || !pat_len) {
        P2_LOG_ERR("malloc failed");
        free(pat_bytes); free(pat_len);
        munmap(text, text_len); munmap(apool, auto_bytes); munmap(matches, match_bytes);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    /* Carve the automaton pool into its typed sub-arrays (all within apool). */
    AC ac;
    ac.alpha = alpha;
    ac.cap   = (int32_t)states_cap;
    ac.gotot = (int32_t *)apool;
    ac.fail  = ac.gotot + states_cap * (size_t)alpha;
    ac.out   = ac.fail  + states_cap;
    ac.dict  = ac.out   + states_cap;
    ac.nodes = 0;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);

    /* (a) Generate the ONE text over a small alphabet [0, alpha). A small
     * alphabet makes short patterns collide often, so matches are plentiful --
     * the point being a long, data-dependent append stream in the scan. */
    for (size_t i = 0; i < text_len; i++) {
        text[i] = (uint8_t)(p2_rng_next(&rng) % (uint64_t)alpha);
    }

    /* (b) Generate P random patterns over the same alphabet, each length in
     * [lmin, lmax], and pack them. */
    for (size_t p = 0; p < P; p++) {
        int span = lmax - lmin + 1;
        int L = lmin + (int)(p2_rng_next(&rng) % (uint64_t)span);
        pat_len[p] = L;
        uint8_t *dst = pat_bytes + p * (size_t)lmax;
        for (int k = 0; k < L; k++) dst[k] = (uint8_t)(p2_rng_next(&rng) % (uint64_t)alpha);
    }

    /* (c) BUILD #1 -- the trie / goto function. Insert each pattern; allocate a
     * state whenever an edge is missing. The final state of pattern p records p
     * in its output slot. If two patterns share the exact same string, the later
     * one overwrites out[]; that is fine -- the verifier counts each distinct
     * pattern id, and a duplicate string is a distinct id only if the last one
     * to land wins, which we accept (documented). To keep every pattern id
     * findable we instead keep the FIRST writer, matching brute force below. */
    int32_t root = ac_new_state(&ac);                  /* state 0 */
    if (root < 0) {
        P2_LOG_ERR("automaton pool exhausted at root (raise --max-mb)");
        free(pat_bytes); free(pat_len);
        munmap(text, text_len); munmap(apool, auto_bytes); munmap(matches, match_bytes);
        p2_meta_kv_str(&m, "status", "automaton_overflow"); p2_meta_close(&m); return 1;
    }
    int build_overflow = 0;
    for (size_t p = 0; p < P && !build_overflow; p++) {
        int32_t s = root;
        const uint8_t *ps = pat_bytes + p * (size_t)lmax;
        int L = pat_len[p];
        for (int k = 0; k < L; k++) {
            int c = ps[k];
            int32_t *row = ac.gotot + (size_t)s * alpha;
            if (row[c] < 0) {
                int32_t ns = ac_new_state(&ac);
                if (ns < 0) { build_overflow = 1; break; }
                /* re-fetch row: ac_new_state only appended, never moved apool */
                ac.gotot[(size_t)s * alpha + c] = ns;
            }
            s = ac.gotot[(size_t)s * alpha + c];
        }
        if (!build_overflow && ac.out[s] < 0) ac.out[s] = (int32_t)p;  /* first writer wins */
    }
    if (build_overflow) {
        P2_LOG_ERR("automaton pool exhausted during trie build (raise --max-mb)");
        free(pat_bytes); free(pat_len);
        munmap(text, text_len); munmap(apool, auto_bytes); munmap(matches, match_bytes);
        p2_meta_kv_str(&m, "status", "automaton_overflow"); p2_meta_close(&m); return 1;
    }

    /* (d) BUILD #1 (cont.) -- FAIL and DICT links by BFS over the trie.
     * A scratch queue of state ids drives the breadth-first order. */
    int32_t *bq = (int32_t *)malloc((size_t)ac.nodes * sizeof(int32_t));
    if (!bq) {
        P2_LOG_ERR("malloc(bfs queue) failed");
        free(pat_bytes); free(pat_len);
        munmap(text, text_len); munmap(apool, auto_bytes); munmap(matches, match_bytes);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    size_t qh = 0, qt = 0;
    /* Depth-1 states fail to the root; seed the queue with them. Also, an absent
     * root edge is redirected to the root so goto(root,c) is always defined. */
    {
        int32_t *rrow = ac.gotot + (size_t)root * alpha;
        for (int c = 0; c < alpha; c++) {
            int32_t v = rrow[c];
            if (v < 0) { rrow[c] = root; }             /* self-loop at root on miss */
            else { ac.fail[v] = root; bq[qt++] = v; }
        }
    }
    while (qh < qt) {
        int32_t u = bq[qh++];
        int32_t *urow = ac.gotot + (size_t)u * alpha;
        /* dict link of u: the nearest fail-ancestor that is a pattern end. */
        int32_t f = ac.fail[u];
        ac.dict[u] = (ac.out[f] >= 0) ? f : ac.dict[f];
        for (int c = 0; c < alpha; c++) {
            int32_t v = urow[c];
            if (v < 0) {
                /* Precompute the goto automaton: a missing edge points where the
                 * fail link would send us, so the scan needs no while-loop. */
                urow[c] = ac.gotot[(size_t)ac.fail[u] * alpha + c];
            } else {
                ac.fail[v] = ac.gotot[(size_t)ac.fail[u] * alpha + c];
                bq[qt++] = v;
            }
        }
    }
    free(bq);
    int32_t automaton_nodes = ac.nodes;
    double t_warmup_end = p2_monotonic();

    /* ---- MEASURE: SCAN the text; APPEND every match. Re-scan each pass, count
     * passes. The append into `matches` is the dominant visible write. ---- */
    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t matches_found = 0;     /* count from the LAST completed pass         */
    int overflow = 0;               /* set if a pass would exceed match_cap       */
    while (!overflow && (p2_monotonic() - t_meas_start) < (double)duration_s) {
        size_t nm = 0;                                 /* append cursor, reset each pass */
        int32_t s = root;
        for (size_t i = 0; i < text_len; i++) {
            /* Precomputed goto: exactly one table lookup per byte, no fail loop. */
            s = ac.gotot[(size_t)s * alpha + text[i]];
            /* Emit the pattern (if any) ending exactly at s, then walk the
             * dictionary-suffix chain for every shorter pattern also ending here. */
            if (ac.out[s] >= 0) {
                int32_t e = s;
                do {
                    if (ac.out[e] >= 0) {
                        if (nm >= match_cap) { overflow = 1; break; }
                        matches[nm].pos  = (uint64_t)i;         /* END index of the match */
                        matches[nm].pid  = (uint32_t)ac.out[e];
                        matches[nm]._pad = 0;
                        nm++;
                    }
                    e = ac.dict[e];
                } while (e >= 0);
            } else {
                /* No pattern ends at s itself, but a suffix state might. */
                int32_t e = ac.dict[s];
                while (e >= 0) {
                    if (ac.out[e] >= 0) {
                        if (nm >= match_cap) { overflow = 1; break; }
                        matches[nm].pos  = (uint64_t)i;
                        matches[nm].pid  = (uint32_t)ac.out[e];
                        matches[nm]._pad = 0;
                        nm++;
                    }
                    e = ac.dict[e];
                }
            }
            if (overflow) break;
        }
        if (overflow) break;
        matches_found = (uint64_t)nm;                  /* full pass completed */
        passes++;
    }
    double t_meas_end = p2_monotonic();

    if (overflow) {
        /* Do not silently truncate: tell the operator to raise --max-mb. */
        P2_LOG_ERR("match list full (capacity %zu records); raise --max-mb to fit all matches",
                   match_cap);
        munmap(text, text_len); munmap(apool, auto_bytes); munmap(matches, match_bytes);
        free(pat_bytes); free(pat_len);
        p2_meta_kv_u64(&m, "match_list_capacity", match_cap);
        p2_meta_kv_str(&m, "status", "match_list_overflow"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Live sink so the last pass's matches are not dead code. */
    volatile uint64_t sink = matches_found ? matches[matches_found - 1].pos : 0;

    /* ---- OPTIONAL one-shot verifier dumps (NOT in the measured loop). Benign
     * file writes so an independent brute-force checker can compare match sets.
     * Only reached after measurement; refuse nothing, but each is guarded. ---- */
    if (dump_text) {
        FILE *f = fopen(dump_text, "wb");
        if (!f) { P2_LOG_WARN("dump-text open failed: %s", strerror(errno)); }
        else { fwrite(text, 1, text_len, f); fclose(f);
               P2_LOG_INFO("dumped text (%zu bytes) to %s", text_len, dump_text); }
    }
    if (dump_patterns) {
        FILE *f = fopen(dump_patterns, "w");
        if (!f) { P2_LOG_WARN("dump-patterns open failed: %s", strerror(errno)); }
        else {
            /* One pattern per line: "<id> <len> b0 b1 ... b{len-1}" (byte values). */
            for (size_t p = 0; p < P; p++) {
                int L = pat_len[p];
                const uint8_t *ps = pat_bytes + p * (size_t)lmax;
                fprintf(f, "%zu %d", p, L);
                for (int k = 0; k < L; k++) fprintf(f, " %u", (unsigned)ps[k]);
                fputc('\n', f);
            }
            fclose(f);
            P2_LOG_INFO("dumped %zu patterns to %s", P, dump_patterns);
        }
    }
    if (dump_matches) {
        FILE *f = fopen(dump_matches, "w");
        if (!f) { P2_LOG_WARN("dump-matches open failed: %s", strerror(errno)); }
        else {
            /* "<pos> <pid>" per line, where pos is the END index of the match. */
            for (uint64_t i = 0; i < matches_found; i++)
                fprintf(f, "%llu %u\n", (unsigned long long)matches[i].pos, matches[i].pid);
            fclose(f);
            P2_LOG_INFO("dumped %llu matches to %s",
                        (unsigned long long)matches_found, dump_matches);
        }
    }

    munmap(text, text_len);
    munmap(apool, auto_bytes);
    munmap(matches, match_bytes);
    free(pat_bytes); free(pat_len);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_i64(&m, "automaton_nodes", automaton_nodes);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "matches_found", matches_found);
    p2_meta_kv_u64(&m, "sink_last_match_pos", (uint64_t)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "two visible writes: BUILD the automaton (growing pool) + SCAN append match list; "
                   "contrast the quiet dfa_match control that writes only a scalar state");
    p2_meta_close(&m);
    return 0;
}
