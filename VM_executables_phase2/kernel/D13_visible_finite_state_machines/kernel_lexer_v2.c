/* kernel_lexer_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Lexer / tokenizer: a character-class Finite State Machine (Berkeley motif
 * D13) that scans a source-like byte stream and SEGMENTS it into lexemes. For
 * every complete lexeme it EMITS one token record into a large output array.
 * The scan is a classic FSM: the current state is a lexeme kind (IDENT, NUMBER,
 * OPERATOR, STRING, WHITESPACE, PUNCT); each input byte's character class either
 * extends the current lexeme or closes it and opens a new one.
 *
 * PICTURE (top view):
 *      character stream                 FSM over char classes            token array (grows)
 *      +---------------------+          (state = lexeme kind)            +----------------------+
 *      | foo = "hi" + 42 ;   |  ---->   START                    ---->  | {IDENT , off, len}   |
 *      +---------------------+           |  alpha -> IDENT               | {WS    , off, len}   |
 *        alpha digit op '"' ws            |  digit -> NUMBER             | {OP    , off, len}   |
 *        punct ...                        |  '"'   -> STRING (to '"')    | {WS    , off, len}   |
 *                                         |  op    -> OPERATOR           | {STRING, off, len}   |
 *      each closed lexeme                 |  ws    -> WHITESPACE         | {WS    , off, len}   |
 *      APPENDS one record  ------------>  |  else  -> PUNCT (1 byte)     | {OP    , off, len}   |
 *                                         v                             | {WS    , off, len}   |
 *      token = { u8 type; u32 offset; u32 length }                      | {NUMBER, off, len}   |
 *      offset+length index back into the input                          |  ...  (front->back)  |
 *                                                                        +----------------------+
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *   Warmup (ONCE, not measured):
 *     Generate a realistic source-like input stream from the harness RNG: a
 *     weighted mix of identifiers ([A-Za-z_][A-Za-z0-9_]*), numbers ([0-9]+),
 *     operators (+ - * / = < > ! & | %), double-quoted string literals, runs of
 *     whitespace, and single punctuation ( ( ) { } [ ] ; , . : ). The buffer is
 *     filled front-to-back until nearly full, then padded with a newline. This
 *     buffer is READ-ONLY for the whole measure loop.
 *
 *   Measure (repeated each pass):
 *     Run the lexer FSM left-to-right over the input. Starting at each lexeme
 *     boundary, classify the first byte to pick a state, consume the maximal run
 *     of bytes that belongs to that state (the "maximal munch" rule), then
 *     APPEND a token record {type, offset, length} at the next free slot of the
 *     token array and advance. Re-lex every pass (refill the token array
 *     front-to-back from slot 0), and count passes.
 *
 *   The token array is the dominant buffer AND the visible write: a dense,
 *   sequential, append-only write front of small fixed-size records that is
 *   rewritten from the front on every pass -- strongly periodic in footprint.
 *
 * ============================================================================
 * WHY VISIBLE (family = KERNEL visible) + CONTRAST WITH THE QUIET CONTROL
 * ============================================================================
 * This kernel MATERIALISES an output: a token stream. A host write-signal sees
 * the growing token array being (re)filled each pass. Contrast the quiet
 * dfa_match control (also a D13 FSM): a matcher only RECOGNISES its input and
 * writes a single scalar (a count / accept flag), so it leaves almost no write
 * footprint. Same motif, opposite visibility: recognise = one scalar; lex = emit
 * a whole token array. The token-array append is the distinct tell here.
 *
 * Token-array capacity is derived from --max-mb (shared with the input buffer);
 * if the input produces more tokens than fit, the run stops with a clear error
 * telling the operator to raise --max-mb -- it never silently truncates.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL (visible). Dwarf: Finite State Machines (D13).
 * See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + generate input) / measure (re-lex passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_lexer_v2";

/* Token kinds. The FSM state while scanning IS the kind of the lexeme in flight;
 * TOK_END is a sentinel used only by the verifier dump, never emitted. */
enum {
    TOK_IDENT = 0,   /* [A-Za-z_][A-Za-z0-9_]*                     */
    TOK_NUMBER,      /* [0-9]+                                     */
    TOK_OPERATOR,    /* run of + - * / = < > ! & | %               */
    TOK_STRING,      /* "..." including both quotes (no escapes)   */
    TOK_WS,          /* run of spaces / tabs / newlines            */
    TOK_PUNCT,       /* a single ( ) { } [ ] ; , . : byte          */
    TOK_KIND_COUNT
};

/* One emitted token record. Packed so the on-disk layout the verifier reads via
 * --dump-tokens matches the in-memory layout byte-for-byte (u8 tag + two u32).
 * This is the small fixed-size record that tiles the token array. */
#pragma pack(push, 1)
typedef struct {
    uint8_t  type;      /* one of the TOK_* kinds        */
    uint32_t offset;    /* start byte in the input       */
    uint32_t length;    /* lexeme length in bytes (>= 1) */
} Token;
#pragma pack(pop)

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign lexer/tokenizer FSM; finite-state-machine kernel, visible)\n"
"  --input-mb M          Source-like input stream size in MiB (default 32)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the source stream (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes (input + tokens) (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* uniform double in [0,1) from the xoshiro stream */
static inline double p2_rng_unit(p2_rng_t *r) {
    return (double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Bounded random integer in [0, n) from the xoshiro stream. */
static inline uint32_t rng_below(p2_rng_t *r, uint32_t n) {
    return (uint32_t)(p2_rng_unit(r) * (double)n);
}

/* ---------- Character-class tables (drive the FSM transitions) ----------
 * Built once at startup. Classifying a byte is a single array lookup, so the
 * scanner's inner loop is branch-light. */
static uint8_t IS_IDENT_START[256];   /* letter or underscore              */
static uint8_t IS_IDENT_CONT[256];    /* letter, digit, or underscore      */
static uint8_t IS_DIGIT[256];         /* 0-9                               */
static uint8_t IS_OP[256];            /* operator bytes                    */
static uint8_t IS_WS[256];            /* space, tab, newline, CR           */

static void build_class_tables(void) {
    for (int c = 'a'; c <= 'z'; c++) { IS_IDENT_START[c] = 1; IS_IDENT_CONT[c] = 1; }
    for (int c = 'A'; c <= 'Z'; c++) { IS_IDENT_START[c] = 1; IS_IDENT_CONT[c] = 1; }
    IS_IDENT_START['_'] = 1; IS_IDENT_CONT['_'] = 1;
    for (int c = '0'; c <= '9'; c++) { IS_DIGIT[c] = 1; IS_IDENT_CONT[c] = 1; }
    const char *ops = "+-*/=<>!&|%";
    for (const char *o = ops; *o; o++) IS_OP[(unsigned char)*o] = 1;
    IS_WS[' '] = 1; IS_WS['\t'] = 1; IS_WS['\n'] = 1; IS_WS['\r'] = 1;
}

/* ---------- Source-stream generator (warmup only) ----------
 * Emit a weighted mix of lexemes into buf[0..cap), returning the number of bytes
 * written. Each lexeme is followed by a run of whitespace often enough that the
 * stream reads like source. We stop when the next lexeme might not fit, then the
 * caller pads the tail so the buffer is fully defined. */
static const char PUNCT_CHARS[] = "(){}[];,.:";
static const char OP_CHARS[]    = "+-*/=<>!&|%";

static size_t gen_source(uint8_t *buf, size_t cap, p2_rng_t *rng) {
    size_t n = 0;
    /* Leave headroom for the longest single lexeme we might emit plus a
     * trailing whitespace byte, so we never write past cap. */
    const size_t MAX_LEX = 40;
    while (n + MAX_LEX + 2 < cap) {
        uint32_t roll = rng_below(rng, 100);
        if (roll < 40) {                         /* identifier (40%) */
            buf[n++] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
                       [rng_below(rng, 53)];
            uint32_t len = 1 + rng_below(rng, 11);          /* 1..11 chars total */
            for (uint32_t k = 1; k < len; k++)
                buf[n++] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                           [rng_below(rng, 63)];
        } else if (roll < 62) {                  /* number (22%) */
            uint32_t len = 1 + rng_below(rng, 8);           /* 1..8 digits */
            for (uint32_t k = 0; k < len; k++)
                buf[n++] = (uint8_t)('0' + rng_below(rng, 10));
        } else if (roll < 80) {                  /* operator run (18%) */
            uint32_t len = 1 + rng_below(rng, 2);           /* 1..2 op bytes */
            for (uint32_t k = 0; k < len; k++)
                buf[n++] = (uint8_t)OP_CHARS[rng_below(rng, (uint32_t)sizeof(OP_CHARS) - 1)];
        } else if (roll < 88) {                  /* string literal (8%) */
            buf[n++] = '"';
            uint32_t len = rng_below(rng, 12);              /* 0..11 inner chars */
            for (uint32_t k = 0; k < len; k++) {
                /* printable, non-quote, non-backslash so there are no escapes */
                uint8_t ch = (uint8_t)(0x20 + rng_below(rng, 0x5f)); /* 0x20..0x7e */
                if (ch == '"' || ch == '\\') ch = 'x';
                buf[n++] = ch;
            }
            buf[n++] = '"';
        } else {                                 /* punctuation (12%) */
            buf[n++] = (uint8_t)PUNCT_CHARS[rng_below(rng, (uint32_t)sizeof(PUNCT_CHARS) - 1)];
        }
        /* whitespace separator: 1..3 bytes, biased to a single space */
        uint32_t ws = 1 + rng_below(rng, 3);
        for (uint32_t k = 0; k < ws; k++) {
            uint32_t w = rng_below(rng, 10);
            buf[n++] = (uint8_t)(w < 7 ? ' ' : (w < 9 ? '\n' : '\t'));
        }
    }
    /* Pad the remainder with newlines so every input byte is defined and the
     * final lexeme (a whitespace run) closes cleanly at end-of-input. */
    while (n < cap) buf[n++] = '\n';
    return n;
}

/* ---------- The lexer FSM (the measured hot loop) ----------
 * Scan src[0..len) left to right. For each lexeme, classify the leading byte to
 * choose a state, consume the maximal run for that state (maximal munch), then
 * APPEND a {type, offset, length} record to tok[]. Returns the number of tokens
 * emitted, or SIZE_MAX if the token array (capacity tok_cap) overflowed -- the
 * caller then stops and asks the operator to raise --max-mb. Because lexemes
 * partition the stream, the emitted tokens exactly tile [0, len). */
static size_t lex_stream(const uint8_t *src, size_t len,
                         Token *tok, size_t tok_cap,
                         uint64_t *class_counts) {
    size_t i = 0, nt = 0;
    while (i < len) {
        if (nt >= tok_cap) return (size_t)-1;    /* overflow: signal caller */
        size_t start = i;
        uint8_t c = src[i];
        uint8_t type;

        if (IS_IDENT_START[c]) {                 /* IDENT: start + cont* */
            type = TOK_IDENT;
            i++;
            while (i < len && IS_IDENT_CONT[src[i]]) i++;
        } else if (IS_DIGIT[c]) {                /* NUMBER: digit+ */
            type = TOK_NUMBER;
            i++;
            while (i < len && IS_DIGIT[src[i]]) i++;
        } else if (c == '"') {                   /* STRING: quote .. quote */
            type = TOK_STRING;
            i++;
            while (i < len && src[i] != '"') i++;
            if (i < len) i++;                    /* consume the closing quote */
        } else if (IS_OP[c]) {                   /* OPERATOR: op-byte+ */
            type = TOK_OPERATOR;
            i++;
            while (i < len && IS_OP[src[i]]) i++;
        } else if (IS_WS[c]) {                   /* WHITESPACE: ws-byte+ */
            type = TOK_WS;
            i++;
            while (i < len && IS_WS[src[i]]) i++;
        } else {                                 /* PUNCT: exactly one byte */
            type = TOK_PUNCT;
            i++;
        }

        /* APPEND the token record: the visible, append-only write front. */
        tok[nt].type   = type;
        tok[nt].offset = (uint32_t)start;
        tok[nt].length = (uint32_t)(i - start);
        nt++;
        class_counts[type]++;
    }
    return nt;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long input_mb   = p2_get_i64(argc, argv, "--input-mb", 32);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (input_mb < 1 || input_mb > 65536) {
        P2_LOG_ERR("input-mb %lld out of range (1..65536)", input_mb);
        return 2;
    }
    size_t input_bytes = (size_t)input_mb * 1024ULL * 1024ULL;
    size_t cap_bytes   = (size_t)max_mb * 1024ULL * 1024ULL;

    /* Clamp the input buffer to --max-mb, reserving room for the token array.
     * The offset/length fields are u32, so a single input buffer must stay under
     * 4 GiB for its byte indices to be representable. */
    if (input_bytes >= (size_t)UINT32_MAX) {
        P2_LOG_ERR("input-mb %lld too large: input bytes must be < 4 GiB (u32 offsets)", input_mb);
        return 2;
    }
    if (input_bytes >= cap_bytes) {
        P2_LOG_ERR("input bytes %zu leave no room for tokens under --max-mb %lld; lower --input-mb or raise --max-mb",
                   input_bytes, max_mb);
        return 2;
    }

    /* Give the remaining budget to the token array. Worst case (all single-byte
     * punctuation separated by single-byte whitespace) is ~1 token per 2 input
     * bytes; our generated mix is far coarser, but we size generously from the
     * leftover budget and still hard-stop if a pass overflows. */
    size_t tok_budget = cap_bytes - input_bytes;
    size_t tok_cap    = tok_budget / sizeof(Token);
    if (tok_cap == 0) {
        P2_LOG_ERR("no room for token records under --max-mb %lld; raise --max-mb", max_mb);
        return 2;
    }
    size_t tok_bytes  = tok_cap * sizeof(Token);
    size_t total_bytes = input_bytes + tok_bytes;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D13 Finite State Machines");
    p2_meta_kv_str(&m, "family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "scheme", "character-class lexer FSM; appends one token record per lexeme into a large token array");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "input_mb", input_mb);
    p2_meta_kv_u64(&m, "input_bytes", input_bytes);
    p2_meta_kv_u64(&m, "token_capacity", tok_cap);
    p2_meta_kv_u64(&m, "token_array_bytes", tok_bytes);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The token array is the dominant buffer and the visible write -> mmap +
     * mlock it. The read-only input stream gets its own mapping. */
    Token *tok = (Token *)mmap(NULL, tok_bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    uint8_t *src = (uint8_t *)mmap(NULL, input_bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (tok == MAP_FAILED || src == MAP_FAILED) {
        P2_LOG_ERR("mmap failed (tok=%zu, src=%zu): %s", tok_bytes, input_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(tok, tok_bytes, MADV_NOHUGEPAGE);
    p2_madvise(src, input_bytes, MADV_SEQUENTIAL);
    if (!no_mlock) { p2_mlock_soft(tok, tok_bytes); p2_mlock_soft(src, input_bytes); }

    build_class_tables();

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Generate the source-like stream ONCE. Read-only for the whole measure. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    size_t src_len = gen_source(src, input_bytes, &rng);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t tokens_last = 0;
    uint64_t class_counts[TOK_KIND_COUNT] = {0};
    int overflowed = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        uint64_t counts[TOK_KIND_COUNT] = {0};
        size_t nt = lex_stream(src, src_len, tok, tok_cap, counts);
        if (nt == (size_t)-1) {
            /* The token array cannot hold one full pass: stop cleanly and tell
             * the operator to raise --max-mb. Never silently truncate. */
            P2_LOG_ERR("token array overflow: %zu-token capacity too small for input; raise --max-mb (currently %lld)",
                       tok_cap, max_mb);
            overflowed = 1;
            break;
        }
        tokens_last = (uint64_t)nt;
        for (int k = 0; k < TOK_KIND_COUNT; k++) class_counts[k] = counts[k];
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Prevent dead-code elimination: touch the last emitted record. */
    volatile uint32_t sink = (tokens_last > 0) ? tok[tokens_last - 1].offset : 0u;

    munmap(tok, tok_bytes);
    munmap(src, input_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "source_len_bytes", src_len);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "tokens_emitted", tokens_last);
    p2_meta_kv_u64(&m, "identifiers", class_counts[TOK_IDENT]);
    p2_meta_kv_u64(&m, "numbers", class_counts[TOK_NUMBER]);
    p2_meta_kv_u64(&m, "operators", class_counts[TOK_OPERATOR]);
    p2_meta_kv_u64(&m, "strings", class_counts[TOK_STRING]);
    p2_meta_kv_u64(&m, "whitespace", class_counts[TOK_WS]);
    p2_meta_kv_u64(&m, "punctuation", class_counts[TOK_PUNCT]);
    p2_meta_kv_i64(&m, "token_overflow", overflowed);
    p2_meta_kv_u64(&m, "last_token_offset", (uint64_t)sink);
    p2_meta_kv_str(&m, "status", overflowed ? "token_overflow" : "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "visible write is the append-only token array refilled per pass; contrast quiet dfa_match (recognise -> scalar). tokens exactly tile the input");
    p2_meta_close(&m);
    return overflowed ? 1 : 0;
}
