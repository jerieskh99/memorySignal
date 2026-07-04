/* kernel_dfa_build_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Finite State Machines dwarf (Berkeley motif D13), the DFA-CONSTRUCTION member.
 * A finite automaton can be run over a stream (cheap, quiet: a few registers
 * walking an existing table) or it can be BUILT. This kernel BUILDS one: it
 * turns a nondeterministic automaton (NFA) into an equivalent deterministic one
 * (DFA) by the classic subset construction, and in doing so WRITES a large,
 * dense transition table. The table construction -- not any recognition run --
 * is the workload. This is the exact inverse of a dfa_match kernel: match RUNS
 * the machine (invisible reads over a fixed table); build MAKES the machine
 * (a big dense write front).
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE
 * ----------------------------------------------------------------------------
 * Running a DFA over text touches only tiny state and reads a table that was
 * written once -- to a host write-signal it is almost silent. Subset
 * construction is the opposite: every reachable subset of NFA states becomes a
 * new DFA row, and each row's |alphabet| cells are written densely as they are
 * discovered. The parametric NFA below ("accept iff the k-th symbol from the end
 * is the marker") is textbook-famous for a determinised blow-up of ~2^k states,
 * so the written table grows large and dense on demand -- that construction is
 * the tell.
 *
 * ============================================================================
 * PICTURE (top view):  NFA  ->  subset construction  ->  dense DFA table
 * ============================================================================
 *
 *   NFA (bitset state sets)          intern subsets              DFA table
 *   overlapping possibilities        (hash map:                 trans[state][sym]
 *                                      K-bit mask -> id)         written dense
 *      (q0)--a-->(q1)                 {q0}        = S0           sym:  a  b  c  d
 *        |  \  a,b  \                  {q0,q1}     = S1          S0 [  .  .  .  . ]
 *        a   `-a-->(q2)               {q0,q1,q2}  = S2          S1 [  .  .  .  . ]
 *        v          |                  {q0,q2,q3}  = S3          S2 [  .  .  .  . ]
 *      (q0)         c                  ...          ...          S3 [  .  .  .  . ]
 *   (a self-loop keeps                              |              ^  each cell:
 *    all runs alive)               each new subset  |              |  next subset,
 *                                  = one new DFA row +------------->+  interned & stored
 *
 *   move(S, sym) = union over q in S of NFA delta(q, sym), as a K-bit mask;
 *   intern that mask (new id if unseen), then WRITE trans[S][sym] = its id.
 *
 * ============================================================================
 * ALGORITHM  (subset construction, NFA -> DFA)
 * ============================================================================
 *   State sets are K-bit masks stored as arrays of 64-bit words (W words each,
 *   W = ceil(K/64)). The NFA is given by, for every (state, symbol), a mask of
 *   destination states -- delta[q * A + sym]. There are no epsilon moves in this
 *   construction, so the start set is simply {q0} and no epsilon-closure step is
 *   needed (the closure is the identity).
 *
 *     1. Start set = {q0}. Intern it as DFA state 0 and push it on a worklist.
 *     2. While the worklist is non-empty, pop an unmarked DFA state S (a mask):
 *          for each alphabet symbol sym in [0, A):
 *            - next = OR of delta[q][sym] over every NFA state q set in S
 *                     (bitset union: word-wise OR across the W words).
 *            - id   = intern(next): look the mask up in a hash map; if unseen,
 *                     copy it into the subset pool, give it the next DFA id, and
 *                     push it on the worklist.
 *            - WRITE trans[S_id * A + sym] = id.   <-- the dense table write
 *     3. Stop when no unmarked DFA states remain (or the state cap is hit).
 *
 *   Accept: a DFA state accepts iff its subset contains any NFA accepting state
 *   (mask AND accept-mask is non-zero). Computed after construction by scanning
 *   the interned subsets; never touched in the measured build loop.
 *
 *   Bound: subset construction can blow up exponentially, so the number of DFA
 *   states is capped by --max-states and by the table's --max-mb budget. Hitting
 *   the cap is a clean, logged error, not a crash.
 *
 * Each measure pass rebuilds the DFA from scratch: reset the interning map and
 * the subset pool, re-run subset construction, and re-write the whole table.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * (The optional --dump-nfa / --dump-dfa one-shot writes are for the external
 * verifier only and never run inside the measured loop.)
 * Signature family: KERNEL (write-visible). Dwarf: Finite State Machines.
 * See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init NFA) / measure (repeated subset construction) /
 * cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_dfa_build_v2";

/* Upper bound on the NFA state count K. The subset pool and the interning map
 * both hold W-word masks; a modest K already determinises to a large table
 * (the "k-th from last" NFA blows up to ~2^(K-1) DFA states), so K need not be
 * large to fill the buffer. This cap keeps a single mask small and bounds the
 * per-mask work. */
#define MAX_NFA_STATES 512

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign NFA->DFA subset construction; FSM kernel)\n"
"  --nfa-states K        NFA state count (default 16, max 512). Determinises to ~2^(K-1) DFA states\n"
"  --alphabet A          Alphabet symbol count (default 4, range 2..256)\n"
"  --max-states S        Hard cap on DFA states built (default 200000)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on transition-table bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dump-nfa PATH       (verifier only) dump NFA to PATH and exit; NOT in measure loop\n"
"  --dump-dfa PATH       (verifier only) dump built DFA to PATH and exit; NOT in measure loop\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* ---------------------------------------------------------------------------
 * Bitset helpers. A state set over K NFA states is W = ceil(K/64) words.
 * All masks (start set, NFA delta rows, accept mask, interned subsets) share
 * this fixed W-word layout so unions are a straight word-wise OR.
 * --------------------------------------------------------------------------- */
static inline size_t bs_words(int K) { return (size_t)((K + 63) / 64); }

static inline void bs_set(uint64_t *m, int bit) {
    m[bit >> 6] |= (uint64_t)1 << (bit & 63);
}
static inline int bs_test(const uint64_t *m, int bit) {
    return (m[bit >> 6] >> (bit & 63)) & 1;
}
static inline void bs_zero(uint64_t *m, size_t W) {
    for (size_t i = 0; i < W; i++) m[i] = 0;
}
/* dst |= src, word-wise (the subset-union core of move()). */
static inline void bs_or(uint64_t *dst, const uint64_t *src, size_t W) {
    for (size_t i = 0; i < W; i++) dst[i] |= src[i];
}
static inline int bs_equal(const uint64_t *a, const uint64_t *b, size_t W) {
    for (size_t i = 0; i < W; i++) if (a[i] != b[i]) return 0;
    return 1;
}
/* FNV-1a over the W words -> hash-map bucket for a subset mask. */
static inline uint64_t bs_hash(const uint64_t *m, size_t W) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < W; i++) {
        h ^= m[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* ---------------------------------------------------------------------------
 * The NFA. delta[q * A + sym] is the destination-state MASK for taking symbol
 * sym from state q (a W-word bitset). accept is the W-word mask of accepting
 * NFA states. Built once in warmup, read-only during construction.
 * --------------------------------------------------------------------------- */
typedef struct {
    int       K;          /* number of NFA states                       */
    int       A;          /* alphabet size                              */
    size_t    W;          /* words per mask = ceil(K/64)                */
    uint64_t *delta;      /* K*A masks, each W words (row-major q,sym)   */
    uint64_t *accept;     /* W-word accepting-state mask                 */
    int       start;      /* start state id (always 0 here)             */
} NFA;

/* Build the parametric "k-th symbol from the end is the marker" NFA over K
 * states and alphabet A. This is the standard construction whose determinised
 * form has ~2^(K-1) states -- a reliable, tunable table-size blow-up.
 *
 *   states  0..K-1, start = 0, accept = {K-1}
 *   symbol 0 is the "marker".
 *   delta(0, any)      = {0}            (self-loop keeps the machine alive)
 *   delta(0, marker=0) = {0, 1}         (nondeterministically guess "K-1 symbols
 *                                         from here to the end starts now")
 *   delta(i, any)      = {i+1}          for 1 <= i <= K-2  (count down the tail)
 *   delta(K-1, any)    = {}             (accepting sink; the guess must land here)
 *
 * A random-pattern-union NFA is an alternative visibility source, but the
 * k-from-last family gives a cleaner, monotone size knob, so we use it. */
static int nfa_build(NFA *nfa, int K, int A, uint64_t seed) {
    (void)seed;                       /* deterministic structure; seed reserved */
    nfa->K = K; nfa->A = A;
    nfa->W = bs_words(K);
    nfa->start = 0;
    size_t nmask = (size_t)K * (size_t)A;
    nfa->delta  = (uint64_t *)calloc(nmask * nfa->W, sizeof(uint64_t));
    nfa->accept = (uint64_t *)calloc(nfa->W, sizeof(uint64_t));
    if (!nfa->delta || !nfa->accept) { free(nfa->delta); free(nfa->accept); return -1; }

    for (int sym = 0; sym < A; sym++) {
        uint64_t *row0 = nfa->delta + ((size_t)0 * A + sym) * nfa->W;
        bs_set(row0, 0);                          /* delta(0, any) contains 0  */
        if (sym == 0) bs_set(row0, (K > 1) ? 1 : 0); /* marker also guesses ->1 */
    }
    for (int i = 1; i <= K - 2; i++) {
        for (int sym = 0; sym < A; sym++) {
            uint64_t *row = nfa->delta + ((size_t)i * A + sym) * nfa->W;
            bs_set(row, i + 1);                   /* delta(i, any) = {i+1}     */
        }
    }
    /* state K-1: no outgoing edges (rows already zeroed) -> accepting sink. */
    bs_set(nfa->accept, K - 1);
    return 0;
}

static void nfa_free(NFA *nfa) {
    if (!nfa) return;
    free(nfa->delta); free(nfa->accept);
    nfa->delta = NULL; nfa->accept = NULL;
}

/* ---------------------------------------------------------------------------
 * Subset-construction workspace. The transition table `trans` is the dominant,
 * mmap+mlock'd buffer and is WRITTEN dense during construction. The subset pool
 * stores one W-word mask per DFA state; the open-addressed hash map interns
 * masks -> DFA ids so identical subsets collapse to one row.
 * --------------------------------------------------------------------------- */
typedef struct {
    size_t    W;          /* words per mask                                    */
    int       A;          /* alphabet                                          */
    long long cap_states; /* max DFA states (from --max-states and table cap)  */

    int32_t  *trans;      /* [cap_states * A] DFA transition table (mmap'd)     */
    uint64_t *pool;       /* [cap_states * W] one subset mask per DFA state     */
    long long nstates;    /* DFA states created so far this build              */

    int32_t  *map;        /* open-addressing hash map: bucket -> dfa id (-1)    */
    size_t    map_cap;    /* map bucket count (power of two)                    */
    size_t    map_mask;   /* map_cap - 1                                        */

    int32_t  *work;       /* worklist stack of unmarked DFA ids                */
    long long work_top;   /* worklist stack pointer                            */
    int       overflow;   /* set if the state cap was hit                      */
} Subset;

/* Reset the interning map and counters for a fresh build (the pool/table are
 * simply overwritten as new states are created). map is cleared to -1. */
static void subset_reset(Subset *s) {
    s->nstates  = 0;
    s->work_top = 0;
    s->overflow = 0;
    for (size_t i = 0; i < s->map_cap; i++) s->map[i] = -1;
}

/* Intern a subset mask: return its DFA id, creating a new DFA state (and
 * pushing it on the worklist) if the mask is unseen. Returns -1 on state-cap
 * overflow. Open addressing with linear probing over map_cap buckets. */
static long long subset_intern(Subset *s, const uint64_t *mask) {
    uint64_t h = bs_hash(mask, s->W);
    size_t b = (size_t)h & s->map_mask;
    for (;;) {
        int32_t id = s->map[b];
        if (id < 0) {
            /* Empty bucket -> this is a new subset. Bound the state count. */
            if (s->nstates >= s->cap_states) { s->overflow = 1; return -1; }
            long long nid = s->nstates++;
            uint64_t *slot = s->pool + (size_t)nid * s->W;
            for (size_t i = 0; i < s->W; i++) slot[i] = mask[i];   /* copy mask */
            s->map[b] = (int32_t)nid;
            s->work[s->work_top++] = (int32_t)nid;                 /* mark unmarked */
            return nid;
        }
        const uint64_t *cand = s->pool + (size_t)id * s->W;
        if (bs_equal(cand, mask, s->W)) return id;                 /* already interned */
        b = (b + 1) & s->map_mask;                                 /* linear probe */
    }
}

/* Run subset construction for `nfa` into workspace `s`, writing the dense
 * transition table. `scratch` is one reusable W-word mask for building moves.
 * Returns 0 on success, -1 if the DFA-state cap was hit. */
static int subset_build(Subset *s, const NFA *nfa, uint64_t *scratch) {
    subset_reset(s);
    const size_t W = s->W;
    const int    A = s->A;

    /* Start set = {start}. (No epsilon moves, so closure is the identity.) */
    bs_zero(scratch, W);
    bs_set(scratch, nfa->start);
    if (subset_intern(s, scratch) < 0) return -1;

    /* Process unmarked DFA states from the worklist. */
    while (s->work_top > 0) {
        long long sid = s->work[--s->work_top];
        /* Snapshot this state's mask: interning below can reallocate nothing
         * (fixed pool) but may append rows, so read our source words up front
         * via the stable pool pointer. */
        const uint64_t *src = s->pool + (size_t)sid * W;

        for (int sym = 0; sym < A; sym++) {
            /* move(src, sym) = union of delta[q][sym] for every q set in src. */
            bs_zero(scratch, W);
            for (size_t wi = 0; wi < W; wi++) {
                uint64_t bits = src[wi];
                while (bits) {
                    int q = (int)(wi * 64) + __builtin_ctzll(bits);
                    bits &= bits - 1;                              /* clear low bit */
                    const uint64_t *drow = nfa->delta + ((size_t)q * A + sym) * W;
                    bs_or(scratch, drow, W);
                }
            }
            long long nid = subset_intern(s, scratch);
            if (nid < 0) return -1;                                /* state cap hit */
            /* re-fetch src: subset_intern may have advanced the pool cursor, but
             * the pool base is fixed (mmap'd), so &pool[sid*W] is still valid. */
            src = s->pool + (size_t)sid * W;
            s->trans[(size_t)sid * A + sym] = (int32_t)nid;        /* DENSE WRITE  */
        }
    }
    return s->overflow ? -1 : 0;
}

/* Count accepting DFA states of the last build: a DFA state accepts iff its
 * subset mask intersects the NFA accept mask. Read-only over the pool; run
 * once after a build for metadata, never inside the timed loop. */
static long long subset_count_accepting(const Subset *s, const NFA *nfa) {
    long long acc = 0;
    for (long long i = 0; i < s->nstates; i++) {
        const uint64_t *mask = s->pool + (size_t)i * s->W;
        int hit = 0;
        for (size_t w = 0; w < s->W; w++) {
            if (mask[w] & nfa->accept[w]) { hit = 1; break; }
        }
        if (hit) acc++;
    }
    return acc;
}

/* Round up to the next power of two (>= 1), for the hash-map bucket count. */
static size_t next_pow2(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

/* ---- optional verifier dumps (benign one-shot writes, NOT in measure loop) ---- */

/* Dump the NFA: dimensions, start, accept mask, and every delta row as a
 * space-separated list of destination state ids. Text format the standalone
 * verifier parses. Returns 0 on success. */
static int dump_nfa(const NFA *nfa, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { P2_LOG_ERR("dump-nfa fopen(%s): %s", path, strerror(errno)); return -1; }
    fprintf(f, "NFA\n");
    fprintf(f, "K %d\n", nfa->K);
    fprintf(f, "A %d\n", nfa->A);
    fprintf(f, "start %d\n", nfa->start);
    fprintf(f, "accept");
    for (int q = 0; q < nfa->K; q++) if (bs_test(nfa->accept, q)) fprintf(f, " %d", q);
    fprintf(f, "\n");
    for (int q = 0; q < nfa->K; q++) {
        for (int sym = 0; sym < nfa->A; sym++) {
            const uint64_t *row = nfa->delta + ((size_t)q * nfa->A + sym) * nfa->W;
            fprintf(f, "delta %d %d", q, sym);
            for (int t = 0; t < nfa->K; t++) if (bs_test(row, t)) fprintf(f, " %d", t);
            fprintf(f, "\n");
        }
    }
    fclose(f);
    return 0;
}

/* Dump the built DFA: dimensions, start (always 0), per-state accept flags, and
 * the full transition table (state sym -> next). Text format for the verifier.
 * Returns 0 on success. */
static int dump_dfa(const Subset *s, const NFA *nfa, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { P2_LOG_ERR("dump-dfa fopen(%s): %s", path, strerror(errno)); return -1; }
    fprintf(f, "DFA\n");
    fprintf(f, "states %lld\n", s->nstates);
    fprintf(f, "A %d\n", s->A);
    fprintf(f, "start 0\n");
    fprintf(f, "accept");
    for (long long i = 0; i < s->nstates; i++) {
        const uint64_t *mask = s->pool + (size_t)i * s->W;
        int hit = 0;
        for (size_t w = 0; w < s->W; w++) if (mask[w] & nfa->accept[w]) { hit = 1; break; }
        if (hit) fprintf(f, " %lld", i);
    }
    fprintf(f, "\n");
    for (long long i = 0; i < s->nstates; i++) {
        fprintf(f, "trans %lld", i);
        for (int sym = 0; sym < s->A; sym++)
            fprintf(f, " %d", s->trans[(size_t)i * s->A + sym]);
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nfa_states = p2_get_i64(argc, argv, "--nfa-states", 16);
    long long alphabet   = p2_get_i64(argc, argv, "--alphabet", 4);
    long long max_states = p2_get_i64(argc, argv, "--max-states", 200000);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dump_nfa_path = p2_get_str(argc, argv, "--dump-nfa", NULL);
    const char *dump_dfa_path = p2_get_str(argc, argv, "--dump-dfa", NULL);

    /* ---- argument validation ---- */
    if (nfa_states < 2 || nfa_states > MAX_NFA_STATES) {
        P2_LOG_ERR("nfa-states %lld out of range (2..%d)", nfa_states, MAX_NFA_STATES);
        return 2;
    }
    if (alphabet < 2 || alphabet > 256) {
        P2_LOG_ERR("alphabet %lld out of range (2..256)", alphabet);
        return 2;
    }
    if (max_states < 1 || max_states > (1LL << 31) - 1) {
        P2_LOG_ERR("max-states %lld out of range (1..2^31-1)", max_states);
        return 2;
    }
    int K = (int)nfa_states;
    int A = (int)alphabet;

    /* Table budget: trans is max_states * A int32 cells. Clamp the state cap so
     * the table fits under --max-mb (subset construction is otherwise
     * unbounded). */
    size_t cell = sizeof(int32_t);
    size_t max_bytes = (size_t)max_mb * 1024ULL * 1024ULL;
    long long cap_by_mb = (long long)(max_bytes / ((size_t)A * cell));
    if (cap_by_mb < 1) {
        P2_LOG_ERR("--max-mb %lld too small for even one DFA row (%d cells)", max_mb, A);
        return 2;
    }
    long long cap_states = max_states < cap_by_mb ? max_states : cap_by_mb;
    size_t W = bs_words(K);
    size_t table_bytes = (size_t)cap_states * (size_t)A * cell;
    size_t pool_bytes  = (size_t)cap_states * W * sizeof(uint64_t);

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D13 Finite State Machines");
    p2_meta_kv_str(&m, "family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "scheme", "NFA->DFA subset construction (dense transition-table write; inverse of dfa_match)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "nfa_states", nfa_states);
    p2_meta_kv_i64(&m, "alphabet", alphabet);
    p2_meta_kv_i64(&m, "max_states", max_states);
    p2_meta_kv_i64(&m, "cap_states", cap_states);
    p2_meta_kv_u64(&m, "table_bytes", table_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* ---- allocate the NFA (small; plain malloc via nfa_build) ---- */
    NFA nfa;
    if (nfa_build(&nfa, K, A, seed) != 0) {
        P2_LOG_ERR("nfa alloc failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    /* ---- the transition table is the dominant buffer -> mmap + mlock it ----
     * It is written dense every build pass; that write front is the signature. */
    int32_t *trans = (int32_t *)mmap(NULL, table_bytes, PROT_READ | PROT_WRITE,
                                     MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (trans == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) table failed: %s", table_bytes, strerror(errno));
        nfa_free(&nfa);
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(trans, table_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(trans, table_bytes);

    /* Subset pool (one mask per DFA state) also mmap'd -- it too is rewritten
     * each build as subsets are interned. */
    uint64_t *pool = (uint64_t *)mmap(NULL, pool_bytes, PROT_READ | PROT_WRITE,
                                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (pool == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) pool failed: %s", pool_bytes, strerror(errno));
        munmap(trans, table_bytes); nfa_free(&nfa);
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(pool, pool_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(pool, pool_bytes);

    /* Interning hash map (load factor <= 0.5) and worklist stack. */
    size_t map_cap = next_pow2((size_t)cap_states * 2 + 16);
    int32_t *map = (int32_t *)malloc(map_cap * sizeof(int32_t));
    int32_t *work = (int32_t *)malloc((size_t)cap_states * sizeof(int32_t));
    uint64_t *scratch = (uint64_t *)malloc(W * sizeof(uint64_t));
    if (!map || !work || !scratch) {
        free(map); free(work); free(scratch);
        munmap(trans, table_bytes); munmap(pool, pool_bytes); nfa_free(&nfa);
        P2_LOG_ERR("subset workspace malloc failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    Subset s;
    s.W = W; s.A = A; s.cap_states = cap_states;
    s.trans = trans; s.pool = pool; s.nstates = 0;
    s.map = map; s.map_cap = map_cap; s.map_mask = map_cap - 1;
    s.work = work; s.work_top = 0; s.overflow = 0;

    /* ---- verifier one-shot dump paths: build once, dump, exit (NOT timed) ---- */
    if (dump_nfa_path || dump_dfa_path) {
        int rc = 0;
        if (subset_build(&s, &nfa, scratch) != 0) {
            P2_LOG_ERR("subset construction hit state cap (%lld) during dump; raise --max-states/--max-mb or lower --nfa-states",
                       cap_states);
            rc = 1;
        }
        if (rc == 0 && dump_nfa_path && dump_nfa(&nfa, dump_nfa_path) != 0) rc = 1;
        if (rc == 0 && dump_dfa_path && dump_dfa(&s, &nfa, dump_dfa_path) != 0) rc = 1;
        p2_meta_kv_i64(&m, "dfa_states", s.nstates);
        p2_meta_kv_str(&m, "status", rc ? "dump_failed" : "dumped");
        p2_meta_close(&m);
        free(map); free(work); free(scratch);
        munmap(trans, table_bytes); munmap(pool, pool_bytes); nfa_free(&nfa);
        return rc;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Warm the table and pool pages by building once (also validates the cap
     * before timing). Not counted as a measured pass. */
    int warm_overflow = 0;
    if (subset_build(&s, &nfa, scratch) != 0) warm_overflow = 1;
    long long dfa_states_last = s.nstates;
    double t_warmup_end = p2_monotonic();

    if (warm_overflow) {
        P2_LOG_ERR("subset construction hit DFA-state cap (%lld); raise --max-states/--max-mb or lower --nfa-states (K=%d determinises to ~2^(K-1))",
                   cap_states, K);
        p2_meta_kv_i64(&m, "dfa_states", dfa_states_last);
        p2_meta_kv_i64(&m, "state_cap_hit", 1);
        p2_meta_kv_str(&m, "status", "state_cap_exceeded");
        p2_meta_close(&m);
        free(map); free(work); free(scratch);
        munmap(trans, table_bytes); munmap(pool, pool_bytes); nfa_free(&nfa);
        return 2;
    }

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    int measure_overflow = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Rebuild the DFA from scratch: reset interning, re-run subset
         * construction, re-write the whole dense transition table. */
        if (subset_build(&s, &nfa, scratch) != 0) { measure_overflow = 1; break; }
        dfa_states_last = s.nstates;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    long long accepting = subset_count_accepting(&s, &nfa);
    /* Live reads of the built table so the construction cannot be elided. */
    volatile int32_t sink = s.trans[0];
    if (s.nstates > 1) sink ^= s.trans[(size_t)(s.nstates - 1) * A];

    free(map); free(work); free(scratch);
    munmap(trans, table_bytes);
    munmap(pool, pool_bytes);
    nfa_free(&nfa);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_i64(&m, "dfa_states", dfa_states_last);
    p2_meta_kv_i64(&m, "accepting_dfa_states", accepting);
    p2_meta_kv_i64(&m, "state_cap_hit", measure_overflow);
    p2_meta_kv_i64(&m, "table_sink", (long long)sink);
    p2_meta_kv_str(&m, "status", measure_overflow ? "state_cap_exceeded" : "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "the dense transition-table construction is the write signal (inverse of dfa_match, which only reads a fixed table); subset construction is exponential and bounded by --max-states/--max-mb");
    p2_meta_close(&m);
    return 0;
}
