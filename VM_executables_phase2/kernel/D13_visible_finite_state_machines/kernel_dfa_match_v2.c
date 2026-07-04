/* kernel_dfa_match_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  DFA STRING RECOGNIZER over a byte stream  (QUIET control)
 * ============================================================================
 *
 *  DWARF   : Finite State Machines (D13) (Berkeley 13 computational motif)
 *  FAMILY  : IDLE                        (first-division, memory-signature label)
 *  PURPOSE : Serve as the deliberate QUIET / null control for FSM work. A finite
 *            automaton reads a big input stream and advances one tiny CURRENT-STATE
 *            word per byte, bumping a match COUNTER when it lands on an accept
 *            state. The stream is huge and read-only; the machine's own state is a
 *            few bytes. It tests the null hypothesis: is running a state machine
 *            over a stream invisible to the host memory WRITE-signal? (expected
 *            yes -- the "FSM traversal is invisible" baseline.)
 *
 *  PICTURE (top view):  a byte stream flows through ONE state bubble; the only
 *  things that change are the state word and the match counter.
 *
 *        input[] (READ-ONLY, huge)
 *        +---+---+---+---+---+---+---+ ... +---+
 *        | a | c | g | t | a | c | g | ... | t |   scanned left -> right
 *        +---+---+---+---+---+---+---+ ... +---+
 *              \   each byte b indexes the transition table
 *               v
 *            state <- trans[state][b]        (ONE word, updated in place)
 *               |
 *               +--> if state is ACCEPT:  matches++   (ONE counter)
 *
 *        trans[num_states][256]  (READ-ONLY after build; the DFA program)
 *
 *  ALGORITHM:
 *      1. Build a small fixed DFA that recognizes every occurrence of a fixed
 *         ASCII pattern P (default "acgt"). We use the Knuth-Morris-Pratt
 *         failure function to fill a dense transition table trans[s][c] for
 *         states s = 0..|P| and all 256 byte values c: from state s on byte c,
 *         go to the longest prefix of P that is still a suffix of what we have
 *         matched. State |P| is the unique ACCEPT state (a full match ended
 *         here). This table is the machine's "program": built once, then only
 *         READ. num_states = |P| + 1.
 *      2. Generate a large random input byte stream ONCE, during warmup (like a
 *         static graph or corpus). Pattern bytes are also sprinkled in at random
 *         positions so there is a real, countable number of matches. After this
 *         the input is never written again.
 *      3. MEASURE loop: SCAN the whole input read-only, one pass after another:
 *             state = trans[state][input[i]];
 *             if (state == accept) matches++;
 *         Each pass re-scans the SAME bytes and counts the SAME matches; we just
 *         count how many passes fit in the measurement window. The input is NOT
 *         regenerated inside the loop -- doing so would be a big write and would
 *         destroy the control. Only `state` and `matches` are written.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      Almost nothing. The dominant memory traffic is READS -- the input stream
 *      streamed byte by byte, plus scattered reads of the tiny transition table
 *      -- all invisible to a write-signal. The ONLY writes are a single
 *      current-state word and a match counter, so the observable footprint is
 *      microscopic and the workload looks near-idle despite chewing through
 *      gigabytes of input per second. This is the D13 analogue of spmv / bfs /
 *      mc_pi / nqueens_count.
 *
 *  HONEST NOTE (why the VISIBLE D13 siblings differ):
 *      The visible D13 kernels (lexer / dfa_build / aho_corasick / fsm_transduce)
 *      are NOT visible because they "run a machine harder" -- running the machine
 *      is always this quiet. They become visible only by producing a LARGE
 *      OUTPUT: a lexer emits a token array, a transducer writes a transformed
 *      stream, dfa_build materializes a big transition table, Aho-Corasick fills
 *      a match/offset list. It is the OUTPUT write, not the state stepping, that
 *      the host signal detects. Stepping states + counting is invisible; writing
 *      a result stream is not.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  (An OPTIONAL --dump-input writes the generated stream ONCE, before the timed
 *  loop, for out-of-band verification only; it is off by default and never runs
 *  inside the measure loop.)
 *  Signature family: IDLE (quiet control). Dwarf: Finite State Machines.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (build DFA + generate input) / measure (DFA scans) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_dfa_match_v2";

/* Longest pattern we will build a DFA for. Bounds the transition table
 * (num_states = plen + 1 rows of 256 columns) and keeps the "program" tiny. */
#define MAX_PATTERN 64

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign DFA string recognizer; QUIET control, Finite State Machines)\n"
"  --input-mb M          Read-only input stream size in MiB (default 64)\n"
"  --pattern STR         Fixed ASCII pattern the DFA recognizes (default \"acgt\")\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the input stream (default 42)\n"
"  --max-mb N            Hard cap on input bytes; clamps --input-mb (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --dump-input PATH     Verifier only: dump generated input ONCE before the timed\n"
"                        loop (benign one-shot write; NOT in the measure loop)\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* -------------------------------------------------------------------------
 * Build a dense KMP transition table for pattern P (length plen).
 *   trans has (plen + 1) rows and 256 columns. trans[s][c] is the next state
 *   after being in state s (meaning: the first s bytes of P are currently
 *   matched) and reading byte c. State plen is the ACCEPT state.
 *
 * Standard construction from the KMP failure function:
 *   - trans[0][c] = (c == P[0]) ? 1 : 0
 *   - for s >= 1: trans[s][c] = (c == P[s]) ? s+1 : trans[fail[s]][c]
 * where fail[s] is the length of the longest proper prefix of P[0..s-1] that is
 * also a suffix of it. Row `plen` reuses fail[plen] so matching continues past
 * an accept (this is what makes overlapping occurrences count correctly).
 * ------------------------------------------------------------------------- */
static void build_dfa(const unsigned char *P, size_t plen, int32_t *trans) {
    /* KMP failure function over states 0..plen. fail[0] is unused (set 0). */
    size_t fail[MAX_PATTERN + 1];
    fail[0] = 0;
    if (plen >= 1) fail[1] = 0;
    for (size_t s = 1; s < plen; s++) {
        size_t k = fail[s];
        while (k > 0 && P[s] != P[k]) k = fail[k];
        if (P[s] == P[k]) k++;
        fail[s + 1] = k;
    }

    /* Row 0: from the empty match, only P[0] advances. */
    for (int c = 0; c < 256; c++)
        trans[0 * 256 + c] = ((unsigned char)c == P[0]) ? 1 : 0;

    /* Rows 1..plen: advance on P[s], else fall back through the failure link.
     * For the accept row s == plen there is no P[plen], so every byte falls back
     * via trans[fail[plen]][c]; this keeps scanning for the next occurrence. */
    for (size_t s = 1; s <= plen; s++) {
        for (int c = 0; c < 256; c++) {
            if (s < plen && (unsigned char)c == P[s])
                trans[s * 256 + c] = (int32_t)(s + 1);
            else
                trans[s * 256 + c] = trans[fail[s] * 256 + c];
        }
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long input_mb   = p2_get_i64(argc, argv, "--input-mb", 64);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *pattern  = p2_get_str(argc, argv, "--pattern", "acgt");
    const char *dump_in  = p2_get_str(argc, argv, "--dump-input", NULL);

    if (input_mb < 1) { P2_LOG_ERR("input-mb %lld out of range (>=1)", input_mb); return 2; }
    if (max_mb < 1)   { P2_LOG_ERR("max-mb %lld out of range (>=1)", max_mb); return 2; }
    /* Clamp the input to --max-mb (the hard cap). */
    if (input_mb > max_mb) {
        P2_LOG_WARN("input-mb %lld exceeds --max-mb %lld; clamping to %lld",
                    input_mb, max_mb, max_mb);
        input_mb = max_mb;
    }

    size_t plen = strlen(pattern);
    if (plen < 1 || plen > MAX_PATTERN) {
        P2_LOG_ERR("pattern length %zu out of range (1..%d)", plen, MAX_PATTERN);
        return 2;
    }
    const unsigned char *P = (const unsigned char *)pattern;
    size_t num_states = plen + 1;                 /* states 0..plen; plen = accept */

    size_t input_bytes = (size_t)input_mb * 1024ULL * 1024ULL;
    /* trans is (plen+1)*256 int32; tiny next to the input, but count it in. */
    size_t trans_bytes = num_states * 256 * sizeof(int32_t);

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "IDLE (quiet control)");
    p2_meta_kv_str(&m, "dwarf", "D13 Finite State Machines");
    p2_meta_kv_str(&m, "family", "IDLE (quiet control)");
    p2_meta_kv_str(&m, "role", "quiet control: input is read-only in the measure loop; only a state word + match counter are written");
    p2_meta_kv_str(&m, "scheme", "KMP transition-table DFA scanning a fixed input stream, counting pattern occurrences");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_str(&m, "pattern", pattern);
    p2_meta_kv_i64(&m, "input_mb", input_mb);
    p2_meta_kv_u64(&m, "input_bytes", input_bytes);
    p2_meta_kv_u64(&m, "num_states", num_states);
    p2_meta_kv_u64(&m, "trans_bytes", trans_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The input stream is the dominant buffer -> mmap + advise + mlock it. It is
     * READ-ONLY during measurement (generated once, below), which is exactly what
     * keeps the measured phase quiet on the write-signal. Advise SEQUENTIAL: the
     * scan is a pure forward stream. */
    unsigned char *input = (unsigned char *)mmap(NULL, input_bytes, PROT_READ | PROT_WRITE,
                                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (input == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", input_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(input, input_bytes, MADV_SEQUENTIAL);
    if (!no_mlock) p2_mlock_soft(input, input_bytes);

    /* The DFA transition table (the machine's "program"). Small heap array;
     * built once, then READ-ONLY -- a scattered read per byte, invisible to the
     * write-signal. */
    int32_t *trans = (int32_t *)malloc(trans_bytes);
    if (!trans) {
        P2_LOG_ERR("malloc(trans %zu) failed", trans_bytes);
        munmap(input, input_bytes);
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();

    /* (a) Build the fixed DFA once. */
    build_dfa(P, plen, trans);
    const int32_t accept = (int32_t)plen;         /* the unique accept state */

    /* (b) Generate the ONE input stream once, during warmup. Bytes are random
     * over a small ASCII alphabet drawn from the pattern's own letters, so the
     * pattern arises naturally and often; we also splice the exact pattern in at
     * random positions to guarantee a healthy, countable match count. After this
     * the input is never written again -- that read-only-ness is what keeps the
     * measured phase quiet. Generation is deterministic for a given seed, which
     * is what makes the out-of-band verification reproducible. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);

    /* Alphabet = the distinct bytes of the pattern (so matches occur frequently
     * even before we splice). If the pattern were a single repeated byte this is
     * just that byte; fall back to a fixed 4-letter alphabet if somehow empty. */
    unsigned char alpha[MAX_PATTERN];
    size_t nalpha = 0;
    for (size_t i = 0; i < plen; i++) {
        int seen = 0;
        for (size_t j = 0; j < nalpha; j++) if (alpha[j] == P[i]) { seen = 1; break; }
        if (!seen) alpha[nalpha++] = P[i];
    }
    if (nalpha == 0) { alpha[0] = 'a'; alpha[1] = 'c'; alpha[2] = 'g'; alpha[3] = 't'; nalpha = 4; }

    for (size_t i = 0; i < input_bytes; i++)
        input[i] = alpha[p2_rng_next(&rng) % nalpha];

    /* Splice the exact pattern in at ~ one per 4 KiB, at random aligned-ish
     * offsets, to guarantee occurrences regardless of alphabet luck. Writes stay
     * inside [0, input_bytes) and are part of the ONE-TIME warmup generation. */
    uint64_t splices = 0;
    if (input_bytes >= plen) {
        size_t target = input_bytes / 4096;       /* rough number of splices */
        for (size_t k = 0; k < target; k++) {
            size_t pos = (size_t)(p2_rng_next(&rng) % (uint64_t)(input_bytes - plen + 1));
            memcpy(input + pos, P, plen);
            splices++;
        }
    }

    /* (c) OPTIONAL out-of-band dump for the verifier. One-shot benign write of
     * the generated bytes to a file, BEFORE the timed loop. Off by default and
     * never touched inside the measure loop, so it cannot affect the signal. */
    if (dump_in && dump_in[0]) {
        FILE *df = fopen(dump_in, "wb");
        if (!df) {
            P2_LOG_WARN("--dump-input open failed: %s: %s", dump_in, strerror(errno));
        } else {
            size_t wr = fwrite(input, 1, input_bytes, df);
            if (wr != input_bytes)
                P2_LOG_WARN("--dump-input short write: %zu/%zu", wr, input_bytes);
            fclose(df);
            P2_LOG_INFO("dumped %zu input bytes to %s", input_bytes, dump_in);
        }
    }

    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t matches = 0;                          /* matches on the LAST pass */
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One full read-only scan of the input through the DFA. The ONLY writes
         * in this entire loop body are `state` (one word) and `pass_matches`
         * (one counter) -- everything on the right is a READ of the input stream
         * or the transition table. This is the whole point of the control. */
        int32_t state = 0;
        uint64_t pass_matches = 0;
        for (size_t i = 0; i < input_bytes; i++) {
            state = trans[state * 256 + input[i]]; /* READ input + table -> tiny state write */
            if (state == accept) pass_matches++;   /* the lone counter write */
        }
        matches = pass_matches;                    /* same every pass (deterministic) */
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile uint64_t sink = matches;              /* prevent dead-code elimination */

    free(trans);
    munmap(input, input_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "splices", splices);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "matches", matches);        /* u64, last pass */
    p2_meta_kv_u64(&m, "sink", (unsigned long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected NEAR-IDLE: input stream read-only + invisible; only a state word + match counter written per pass");
    p2_meta_kv_str(&m, "quiet",
                   "input is read-only in the measure loop; only a state word + match counter are written");
    p2_meta_close(&m);
    return 0;
}
