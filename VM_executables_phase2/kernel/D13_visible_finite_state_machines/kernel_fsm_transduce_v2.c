/* kernel_fsm_transduce_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  MEALY-MACHINE STREAM TRANSDUCER: reversible byte escape/framing codec
 * ============================================================================
 *
 *  DWARF   : Finite State Machines (D13) (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                       (first-division, write-visible label)
 *  PURPOSE : Probe the "produce an output stream" write pattern of an FSM. A
 *            Mealy machine reads an input byte stream and, for EACH input byte,
 *            EMITS one or two output bytes into a second, dominant buffer. The
 *            output stream is ~ the size of the input (larger when escapes fire),
 *            so the dominant memory traffic is a dense sequential WRITE front --
 *            the exact opposite of the quiet dfa_match control, which steps a
 *            state word and writes nothing else.
 *
 *  PICTURE (top view):  each input byte drives a Mealy transition that emits
 *  1 or 2 output bytes; the concatenation is a full output stream ~ input size.
 *
 *        input[] (READ-ONLY, huge)
 *        +---+---+---+---+---+---+---+ ... +---+
 *        | h | i | E |DEL| o | E | . | ... | z |   scanned left -> right
 *        +---+---+---+---+---+---+---+ ... +---+
 *          |   |   \       \    |   \    |
 *          v   v    v       v   v    v   v      Mealy: (state,in) -> (state,out*)
 *        [ h ][ i ][E ][Ee][E ][Ed][ o ][E ][Ee] ...        1 or 2 bytes each
 *        +------------------------------------------ ... --+
 *        output[] (WRITTEN every pass, the dominant visible write)
 *
 *      where  E   = the ESC byte,   DEL = the DELIMITER byte,
 *             Ee  = ESC's mapped byte,  Ed = DELIM's mapped byte.
 *      An ordinary byte copies straight through (1 out); ESC or DELIM expands to
 *      a 2-byte escape (ESC followed by a mapped byte). Worst case output = 2x
 *      input (every byte special); typical output is a bit over 1x.
 *
 *  ALGORITHM:
 *    ENCODER FSM (Mealy, two states):
 *        state NORMAL: read byte b.
 *          - b == ESC   -> emit {ESC, EMAP_ESC},  stay NORMAL   (2 out)
 *          - b == DELIM -> emit {ESC, EMAP_DELIM}, stay NORMAL   (2 out)
 *          - otherwise  -> emit {b},               stay NORMAL   (1 out)
 *        (The 2-byte emissions momentarily pass through an internal ESC-PENDING
 *        step -- state tracks whether the previous emitted byte began an escape --
 *        which is what makes this a Mealy machine rather than a plain map.)
 *    DECODER FSM (the exact inverse, used ONLY by the standalone verifier):
 *        state COPY: read byte e.
 *          - e == ESC -> go to state AFTER_ESC
 *          - else     -> output e, stay COPY
 *        state AFTER_ESC: read byte n.
 *          - n == EMAP_ESC   -> output ESC,   go to COPY
 *          - n == EMAP_DELIM -> output DELIM, go to COPY
 *          - else            -> MALFORMED (rejected)
 *      decode(encode(x)) == x for every byte string x: the codec is a pure,
 *      REVERSIBLE framing/escape transform (the kind every serialiser or protocol
 *      framer uses to make a delimiter safe inside a payload). It is NOT
 *      encryption and hides nothing -- correctness is proven by a round trip.
 *
 *    Byte choices (fixed, distinct, so the round trip is unambiguous):
 *        ESC = 0x5C ('\\'), DELIM = 0x00,
 *        EMAP_ESC = 0x5C ('\\'  -> "\\\\"),  EMAP_DELIM = 0x30 ('0' -> "\\0").
 *      EMAP_ESC != EMAP_DELIM, so on ESC the decoder always knows which literal
 *      byte was meant. (This is the classic backslash-escape convention.)
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      A single dense, sequential WRITE front sweeping the output buffer, one
 *      pass after another (the buffer is re-filled from scratch each pass and the
 *      pass count is reported). The input stream is quiescent (read-only) for the
 *      whole loop. So writes dominate and stay separated from reads -- unlike
 *      dfa_match, where the same input is scanned but nothing large is written.
 *
 *  HONEST NOTE (recognise vs transduce):
 *      dfa_match (the QUIET D13 control) and this kernel run the SAME kind of
 *      byte-stream FSM. The difference is not that transduction "runs the machine
 *      harder" -- stepping states is always quiet. It is that a RECOGNIZER writes
 *      one scalar (a match counter), while a TRANSDUCER writes a full OUTPUT
 *      STREAM ~ the input size. It is the output write, not the state stepping,
 *      that the host signal detects. This is the "produce an output stream"
 *      visibility source, distinct from the lexer's token array, dfa_build's
 *      transition table, and Aho-Corasick's match/offset list.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  (OPTIONAL --dump-input / --dump-output write the input and last-pass output
 *  ONCE, before/after the timed loop, for out-of-band verification only; both are
 *  off by default and never run inside the measure loop.)
 *  Signature family: KERNEL (write-visible). Dwarf: Finite State Machines.
 *  See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (generate input) / measure (encode passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_fsm_transduce_v2";

/* Fixed codec alphabet. ESC introduces a 2-byte escape; DELIM is the byte a
 * framer would want to keep unambiguous inside a payload. The two mapped bytes
 * must differ from each other so the decoder can tell which literal was meant. */
#define TR_ESC        ((unsigned char)0x5C)   /* '\\' : the escape byte          */
#define TR_DELIM      ((unsigned char)0x00)   /* NUL  : the delimiter byte       */
#define TR_EMAP_ESC   ((unsigned char)0x5C)   /* '\\' -> "\\\\" (ESC escapes self) */
#define TR_EMAP_DELIM ((unsigned char)0x30)   /* '0'  -> "\\0"  (DELIM mapped byte) */

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign reversible escape transducer; KERNEL visible, Finite State Machines)\n"
"  --input-mb M          Read-only input stream size in MiB (default 32)\n"
"  --esc-density D        1 special byte (ESC/DELIM) per D input bytes; larger = fewer escapes (default 16)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the input stream (default 42)\n"
"  --max-mb N            Hard cap on total (input + output) bytes; clamps buffers (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --dump-input PATH     Verifier only: dump generated input ONCE before the timed\n"
"                        loop (benign one-shot write; NOT in the measure loop)\n"
"  --dump-output PATH    Verifier only: dump last-pass encoded output ONCE after the\n"
"                        timed loop (benign one-shot write; NOT in the measure loop)\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* Encode one input buffer into out[] via the Mealy escape FSM. Returns the
 * number of output bytes produced, or 0 with *overflow set if out_cap would be
 * exceeded (caller treats that as a hard error). Ordinary bytes copy straight
 * through (1 out); ESC and DELIM each expand to a 2-byte escape. The `pending`
 * flag is the Mealy state bit: it marks that the byte just emitted began an
 * escape, so the mapped byte is emitted on the next step. */
static size_t tr_encode(const unsigned char *in, size_t in_len,
                        unsigned char *out, size_t out_cap, int *overflow) {
    size_t o = 0;
    int pending = 0;                 /* Mealy state: 0 = NORMAL, 1 = ESC-PENDING */
    for (size_t i = 0; i < in_len; i++) {
        unsigned char b = in[i];
        if (b == TR_ESC || b == TR_DELIM) {
            /* (state,in) -> emit ESC, momentarily enter ESC-PENDING, then emit
             * the mapped byte and fall back to NORMAL. Two output bytes. */
            if (o + 2 > out_cap) { *overflow = 1; return 0; }
            out[o++] = TR_ESC;
            pending = 1;
            out[o++] = (b == TR_ESC) ? TR_EMAP_ESC : TR_EMAP_DELIM;
            pending = 0;
        } else {
            /* NORMAL -> NORMAL, copy the byte straight through. One output byte. */
            if (o + 1 > out_cap) { *overflow = 1; return 0; }
            out[o++] = b;
        }
    }
    (void)pending;
    return o;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long input_mb    = p2_get_i64(argc, argv, "--input-mb", 32);
    long long esc_density = p2_get_i64(argc, argv, "--esc-density", 16);
    long long duration_s  = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s    = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb      = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu         = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed        = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock    = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run     = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir    = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dump_in   = p2_get_str(argc, argv, "--dump-input", NULL);
    const char *dump_out  = p2_get_str(argc, argv, "--dump-output", NULL);

    if (input_mb < 1)     { P2_LOG_ERR("input-mb %lld out of range (>=1)", input_mb); return 2; }
    if (max_mb < 1)       { P2_LOG_ERR("max-mb %lld out of range (>=1)", max_mb); return 2; }
    if (esc_density < 2)  { P2_LOG_ERR("esc-density %lld out of range (>=2)", esc_density); return 2; }

    /* Output worst case is 2x the input (every byte an escape). Budget input +
     * output against --max-mb; if 3 * input_mb would exceed the cap, clamp the
     * input so input + 2*input fits. This keeps both buffers inside the cap. */
    long long budget_mb = 3 * input_mb;              /* input (1x) + output (<=2x) */
    if (budget_mb > max_mb) {
        long long clamped = max_mb / 3;
        if (clamped < 1) {
            P2_LOG_ERR("max-mb %lld too small for any input (need >=3 MiB)", max_mb);
            return 2;
        }
        P2_LOG_WARN("input-mb %lld needs %lld MiB with 2x output headroom; "
                    "clamping input to %lld MiB to fit --max-mb %lld",
                    input_mb, budget_mb, clamped, max_mb);
        input_mb = clamped;
    }

    size_t input_bytes = (size_t)input_mb * 1024ULL * 1024ULL;
    size_t out_cap     = input_bytes * 2;            /* worst-case escape expansion */
    size_t total_bytes = input_bytes + out_cap;

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D13 Finite State Machines");
    p2_meta_kv_str(&m, "family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "role", "write-visible: a Mealy FSM emits an output stream ~ input size (1-2 out bytes per input byte)");
    p2_meta_kv_str(&m, "scheme", "reversible escape/framing transducer (ESC + mapped byte); NOT encryption; verified by round-trip");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "input_mb", input_mb);
    p2_meta_kv_u64(&m, "input_bytes", input_bytes);
    p2_meta_kv_u64(&m, "output_cap_bytes", out_cap);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "esc_density", esc_density);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* Two mmap'd buffers. The INPUT is read-only during measurement (generated
     * once, below); advise SEQUENTIAL since encoding is a pure forward scan. The
     * OUTPUT is the dominant buffer -- rewritten from scratch every pass, which is
     * the workload's signature write; advise SEQUENTIAL for the forward fill. */
    unsigned char *input = (unsigned char *)mmap(NULL, input_bytes, PROT_READ | PROT_WRITE,
                                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (input == MAP_FAILED) {
        P2_LOG_ERR("mmap(input %zu) failed: %s", input_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    unsigned char *output = (unsigned char *)mmap(NULL, out_cap, PROT_READ | PROT_WRITE,
                                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (output == MAP_FAILED) {
        P2_LOG_ERR("mmap(output %zu) failed: %s", out_cap, strerror(errno));
        munmap(input, input_bytes);
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(input, input_bytes, MADV_SEQUENTIAL);
    p2_madvise(output, out_cap, MADV_SEQUENTIAL);
    if (!no_mlock) { p2_mlock_soft(input, input_bytes); p2_mlock_soft(output, out_cap); }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();

    /* Generate the ONE input stream once, during warmup. Most bytes are ordinary
     * printable ASCII (which never triggers an escape); roughly one byte in
     * `esc_density` is forced to a SPECIAL byte (ESC or DELIM) so the encoder
     * expands it to two bytes and the output ends up meaningfully larger than the
     * input. Deterministic for a given seed, which is what makes out-of-band
     * verification reproducible. After this the input is never written again --
     * that read-only-ness is what keeps the input side of the loop quiet. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    uint64_t special_in = 0;                         /* count of special input bytes */
    for (size_t i = 0; i < input_bytes; i++) {
        uint64_t r = p2_rng_next(&rng);
        if ((r % (uint64_t)esc_density) == 0) {
            /* Force a special byte: alternate ESC / DELIM by the next bit. */
            input[i] = (r & 0x100ULL) ? TR_ESC : TR_DELIM;
            special_in++;
        } else {
            /* Ordinary byte: printable ASCII 0x20..0x7E, and never ESC ('\\').
             * Mapping into a 94-wide window then skipping the ESC code keeps the
             * generated ordinary bytes strictly non-special. */
            unsigned char c = (unsigned char)(0x20 + (r >> 16) % 95);   /* 0x20..0x7E */
            if (c == TR_ESC) c = (unsigned char)0x7E;                    /* dodge ESC  */
            input[i] = c;
        }
    }

    /* OPTIONAL out-of-band dump of the input for the verifier. One-shot benign
     * write of the generated bytes to a file, BEFORE the timed loop. Off by
     * default; never touched inside the measure loop, so it cannot affect the
     * signal. */
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
    size_t   out_len = 0;                             /* output size (last pass)    */
    int      overflow = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One full encode pass: run the Mealy FSM over the whole input and fill
         * the output buffer from scratch. This dense sequential WRITE of ~ the
         * input size is the dominant, host-visible memory traffic -- the whole
         * point of the transducer (contrast dfa_match, which writes one counter).
         * out_cap is sized to the 2x worst case, so overflow cannot occur for a
         * correctly generated input; we still guard and stop on it. */
        out_len = tr_encode(input, input_bytes, output, out_cap, &overflow);
        if (overflow) {
            P2_LOG_ERR("encoder output overflow: input %zu would exceed output cap %zu",
                       input_bytes, out_cap);
            break;
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile unsigned char sink = output[out_len ? out_len - 1 : 0];  /* no dead-code elim */

    double expansion = input_bytes ? (double)out_len / (double)input_bytes : 0.0;

    /* OPTIONAL out-of-band dump of the last-pass encoded output for the verifier.
     * One-shot benign write AFTER the timed loop; off by default and never inside
     * the measure loop. The verifier independently decodes this and asserts
     * decode(output) == input byte-for-byte. */
    if (!overflow && dump_out && dump_out[0]) {
        FILE *df = fopen(dump_out, "wb");
        if (!df) {
            P2_LOG_WARN("--dump-output open failed: %s: %s", dump_out, strerror(errno));
        } else {
            size_t wr = fwrite(output, 1, out_len, df);
            if (wr != out_len)
                P2_LOG_WARN("--dump-output short write: %zu/%zu", wr, out_len);
            fclose(df);
            P2_LOG_INFO("dumped %zu output bytes to %s", out_len, dump_out);
        }
    }

    if (overflow) { munmap(input, input_bytes); munmap(output, out_cap);
        p2_meta_kv_str(&m, "status", "output_overflow"); p2_meta_close(&m); return 1; }

    munmap(input, input_bytes);
    munmap(output, out_cap);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "special_input_bytes", special_in);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "output_bytes", (unsigned long long)out_len);
    p2_meta_kv_f64(&m, "expansion_ratio", expansion);
    p2_meta_kv_u64(&m, "sink", (unsigned long long)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected VISIBLE: a dense sequential output-stream write ~ input size per pass; recognise=scalar, transduce=full stream");
    p2_meta_kv_str(&m, "verify",
                   "round-trip identity: decode(encode(input)) == input; output well-formed (every ESC followed by a valid mapped byte)");
    p2_meta_close(&m);
    return 0;
}
