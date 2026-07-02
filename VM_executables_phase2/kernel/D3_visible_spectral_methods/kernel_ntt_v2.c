/* kernel_ntt_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 *  NUMBER-THEORETIC TRANSFORM:  an FFT-style butterfly over modular integers
 * ============================================================================
 *
 *  DWARF   : Spectral Methods (D3)      (Berkeley 13 computational motif)
 *  FAMILY  : KERNEL                     (first-division, memory-signature label)
 *  PURPOSE : Probe the write signature of a wide, multi-stream in-place spectral
 *            transform whose content is integer/modular rather than complex float,
 *            so it is distinguishable from the floating-point FFT kernel.
 *
 *  The NTT is the exact same radix-2 Cooley-Tukey dataflow as the FFT, but every
 *  complex twiddle factor is replaced by a power of a primitive root of unity in
 *  the finite field Z/qZ (q a prime), and every add/multiply is done modulo q.
 *  This is the compute core of lattice cryptography, CKKS homomorphic encryption,
 *  and big-integer / polynomial multiplication. To mirror the Residue-Number-
 *  System (RNS) layout those schemes use, we run L independent coefficient
 *  vectors ("limbs") of length N side by side, making the transform a WIDE,
 *  multi-stream rewrite rather than a single vector.
 *
 *  IMPORTANT: this is a pure polynomial-arithmetic BENCHMARK. There is NO
 *  encryption, NO key material, and NO secret data of any kind -- the coefficients
 *  are just uniform random residues. Only the memory-access pattern of the
 *  transform is under study. (For simplicity all limbs share one NTT prime; a real
 *  RNS would use several distinct primes. The write signature is unchanged.)
 *
 *  PICTURE (top view):  L limbs stacked as rows; each is a length-N vector that is
 *  first bit-reversal permuted, then rewritten in place by log2(N) butterfly
 *  stages. All cells hold integers mod q, not complex floats.
 *
 *      limb 0 : [ c0 c1 c2 c3 c4 c5 c6 c7 ]   bit-reverse -> butterflies (mod q)
 *      limb 1 : [ c0 c1 c2 c3 c4 c5 c6 c7 ]   bit-reverse -> butterflies (mod q)
 *      limb 2 : [ c0 c1 c2 c3 c4 c5 c6 c7 ]   bit-reverse -> butterflies (mod q)
 *        ...        (L independent streams, each transformed the same way)
 *      limb L-1:[ c0 c1 c2 c3 c4 c5 c6 c7 ]   bit-reverse -> butterflies (mod q)
 *
 *      one butterfly stage (stride "len/2", doubling each stage):
 *          u = a[i+j]                     a[i+j]        <- (u + v) mod q
 *          v = a[i+j+len/2]*w mod q  ==>  a[i+j+len/2]  <- (u - v) mod q
 *
 *  ALGORITHM (per measured transform):
 *      1. Refill every coefficient with a fresh uniform residue in [0, q).
 *      2. For each limb, permute its vector into bit-reversed order in place.
 *      3. For each limb, sweep log2(N) butterfly stages with the stride doubling
 *         (2, 4, 8, ... N); each stage rewrites all N coefficients using powers
 *         of the stage root computed by modular exponentiation.
 *
 *  MEMORY SIGNATURE (what the host write-signal actually sees):
 *      The classic FFT footprint -- an initial bit-reversal scatter followed by
 *      log2(N) full-array butterfly passes with doubling stride -- but replicated
 *      across L limbs, so the same pattern repeats L times back to back per
 *      transform. Every stage touches (writes) all N cells, so the write volume is
 *      dense and regular. The distinguishing feature versus kernel_fft_v2 is the
 *      integer/modular content (uniform mod q) and the multi-stream repetition,
 *      not the shape of the traversal, which is identical.
 *
 *  Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 *  Signature family: KERNEL. Dwarf: Spectral Methods. See docs/SAFETY_MODEL.md.
 *
 *  Phases: warmup (alloc + init) / measure (forward NTTs) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_ntt_v2";

/* NTT-friendly prime q = 119 * 2^23 + 1, and a primitive root g = 3 of the
 * multiplicative group modulo q. Because (q - 1) is divisible by 2^23, the field
 * contains a 2^k-th root of unity for every k up to 23, which is exactly what a
 * radix-2 transform of length up to 2^23 needs to close on itself. */
#define NTT_Q   998244353ULL
#define NTT_G   3ULL

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign number-theoretic transform; Spectral kernel, no crypto)\n"
"  --n N                 Transform length, power of 2 (default 65536; snapped; N <= 2^23)\n"
"  --limbs L             Independent RNS-style coefficient vectors (default 8)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* Modular exponentiation b^e mod q by square-and-multiply. Used to derive each
 * stage's root of unity from the primitive root; kept separate so the hot
 * butterfly loop only ever does one multiply and one modulo per coefficient. */
static inline uint64_t modpow(uint64_t b, uint64_t e, uint64_t q) {
    uint64_t r = 1; b %= q;
    while (e) { if (e & 1) r = (r * b) % q; b = (b * b) % q; e >>= 1; }
    return r;
}

/* Forward iterative Cooley-Tukey NTT, in place, on a single length-n vector
 * (n a power of 2). Structurally identical to a radix-2 FFT, but the arithmetic
 * is in the field Z/qZ, so twiddle factors are powers of the root of unity and
 * every operation carries a "% NTT_Q". */
static void ntt_forward(uint64_t *a, size_t n) {
    /* Stage 0 -- bit-reversal permutation. The iterative butterflies below
     * consume their inputs in bit-reversed index order, so we first scatter the
     * vector into that order. jrev is maintained as the bit-reversal of i by
     * propagating a carry from the top bit downward; each element is swapped at
     * most once (guarded by i < jrev). This is the one irregular, scattered
     * write of the transform before the dense sequential stages begin. */
    for (size_t i = 1, jrev = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; jrev & bit; bit >>= 1) jrev ^= bit;
        jrev ^= bit;
        if (i < jrev) { uint64_t t = a[i]; a[i] = a[jrev]; a[jrev] = t; }
    }
    /* Stages 1..log2(n) -- butterfly passes with the block length "len" doubling
     * each stage (2, 4, 8, ... n). Every stage rewrites all n coefficients, which
     * is why the write footprint stays dense and whole-array across the sweep. */
    for (size_t len = 2; len <= n; len <<= 1) {
        /* Principal len-th root of unity for this stage: g^((q-1)/len) mod q. */
        uint64_t wlen = modpow(NTT_G, (NTT_Q - 1) / len, NTT_Q);
        for (size_t i = 0; i < n; i += len) {           /* independent blocks */
            uint64_t w = 1;                             /* running twiddle w^j */
            for (size_t j = 0; j < len / 2; j++) {
                /* Butterfly on the pair (i+j, i+j+len/2): combine the "even" half
                 * u with the twiddled "odd" half v into their sum and difference,
                 * both taken mod q. The "+ NTT_Q -" form does the subtraction
                 * without ever going negative in unsigned arithmetic. */
                uint64_t u = a[i + j];
                uint64_t v = (a[i + j + len / 2] * w) % NTT_Q;
                a[i + j]           = (u + v) % NTT_Q;
                a[i + j + len / 2] = (u + NTT_Q - v) % NTT_Q;
                w = (w * wlen) % NTT_Q;                  /* advance to w^(j+1) */
            }
        }
    }
}

static size_t snap_pow2(long long v) {
    size_t n = 1; while ((long long)(n << 1) <= v && (n << 1) != 0) n <<= 1; return n;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nreq       = p2_get_i64(argc, argv, "--n", 65536);
    long long limbs      = p2_get_i64(argc, argv, "--limbs", 8);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nreq < 16 || nreq > (1LL << 23)) { P2_LOG_ERR("n %lld out of range (16..2^23)", nreq); return 2; }
    if (limbs < 1 || limbs > 256) { P2_LOG_ERR("limbs %lld out of range (1..256)", limbs); return 2; }
    size_t N = snap_pow2(nreq);
    size_t L = (size_t)limbs;
    size_t bytes = L * N * sizeof(uint64_t);
    if (bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("buffer bytes %zu exceed --max-mb %lld", bytes, max_mb); return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Spectral Methods");
    p2_meta_kv_str(&m, "scheme", "number-theoretic transform (modular radix-2 butterfly, L RNS-style limbs); CKKS/lattice core, no crypto");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no encryption/keys/network/persistence/sandbox; compute-only");
    p2_meta_kv_u64(&m, "n", N);
    p2_meta_kv_i64(&m, "limbs", limbs);
    p2_meta_kv_u64(&m, "prime_q", NTT_Q);
    p2_meta_kv_u64(&m, "total_bytes", bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    uint64_t *A = (uint64_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (A == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(A, bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(A, bytes);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t k = 0; k < L * N; k++) A[k] = p2_rng_next(&rng) % NTT_Q;
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t transforms = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Refill all L*N coefficients so each iteration transforms genuinely new
         * data (prevents the values from collapsing and keeps the write volume
         * representative). Then transform each limb's slice independently and in
         * place, which produces the back-to-back multi-stream write pattern. */
        for (size_t k = 0; k < L * N; k++) A[k] = p2_rng_next(&rng) % NTT_Q;   /* fresh coefficients */
        for (size_t l = 0; l < L; l++) ntt_forward(A + l * N, N);
        transforms += L;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile uint64_t sink = A[(L - 1) * N + (N / 2)];   /* a transform coefficient mod q */

    munmap(A, bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "transforms", transforms);
    p2_meta_kv_u64(&m, "last_coeff", (uint64_t)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "forward transform only; single NTT prime across limbs (memory pattern faithful, not a real RNS)");
    p2_meta_close(&m);
    return 0;
}
