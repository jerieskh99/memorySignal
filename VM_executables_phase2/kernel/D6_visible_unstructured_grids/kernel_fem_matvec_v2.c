/* kernel_fem_matvec_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Unstructured Grids dwarf (Berkeley motif D6), the MATRIX-FREE finite-element
 * matvec variant. A finite-element operator K acts on a node vector: y = K x.
 * The textbook route ASSEMBLES the global N x N matrix K first, then multiplies.
 * The matrix-free route never forms K at all: it walks the mesh element by
 * element, applies each element's small local matrix ke to that element's few
 * nodes, and SCATTER-ADDS the partial results into the global result vector y.
 * Summed over all elements this reproduces K x exactly -- the same answer, with
 * no global matrix in memory.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (and where it sits within D6)
 * ----------------------------------------------------------------------------
 * The mesh is UNSTRUCTURED: element e connects NPE nodes chosen by an arbitrary
 * connectivity table, not by a fixed grid stride. So the gather (read xe from x)
 * and, crucially, the scatter-add (write into y) hit node slots at data-driven,
 * non-contiguous offsets -- indirect indexing through elem[e][*], the hallmark
 * of unstructured-grid access. That irregular, index-driven write front is the
 * distinct tell versus a structured stencil, whose writes march in fixed strides.
 *
 * Honesty note: this is the QUIETER member of the D6 family. Its dominant
 * written buffer is the result VECTOR y (N doubles) -- the matrix-free analog of
 * SpMV, whose output is likewise a vector. It is deliberately NOT the heavyweight
 * variant that assembles the large global sparse/dense matrix (that build writes
 * an O(nnz)- or O(N^2)-sized structure and is the loud D6 member). We state this
 * plainly: the signature here is scattered vector writes, not bulk matrix fill.
 *
 * ============================================================================
 * ALGORITHM (matrix-free FEM matvec y = K x, element-by-element)
 * ============================================================================
 *   Data:  N node values in x (length N), N node values in y (length N),
 *          E elements, each naming NPE nodes via elem[e][0..NPE-1], and one
 *          fixed symmetric local element matrix ke (NPE x NPE) shared by all
 *          elements (ke[i][j] = (i==j ? NPE-1 : -1); a graph-Laplacian stencil).
 *
 *   Each measure pass:
 *     1. RE-SEED x with fresh random node values and regenerate the random
 *        element connectivity, then ZERO the whole result vector y.
 *     2. For each element e:
 *          a. GATHER    xe[i] = x[elem[e][i]]                 (i = 0..NPE-1)
 *          b. LOCAL MV  ye[i] = sum_j ke[i][j] * xe[j]        (NPE x NPE apply)
 *          c. SCATTER   y[elem[e][i]] += ye[i]                (indirect add)
 *     3. Summed over every element, y now equals K x -- the global matrix K was
 *        never materialised. Count the pass.
 *
 * The local matrix ke is symmetric, so the global operator it induces is
 * symmetric too; that is what lets an independent test ASSEMBLE the dense K by
 * scatter-adding ke and check this matrix-free result against a plain dense K x.
 *
 * MEMORY SIGNATURE (what the host write-signal actually sees):
 *   The dominant written buffer is y (mmap'd, N doubles, mlock'd). Per pass it is
 *   first zeroed (one dense sequential sweep) and then peppered with scatter-adds
 *   at data-driven node offsets -- NPE indirect read-modify-writes per element,
 *   landing wherever the connectivity points. So the write pattern is: one clean
 *   linear clear, followed by an irregular, index-scattered update front over the
 *   same fixed-size region, repeated every pass. Reads of x and of the tiny ke
 *   are invisible to a write-signal; the scatter into y is the observable event.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Unstructured Grids. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (matrix-free matvec passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_fem_matvec_v2";

/* Hard cap on nodes-per-element: bounds the local matrix (NPE x NPE) and the
 * per-element gather/scatter buffers. NPE = 4 models a tetrahedral (or quad)
 * element; the algorithm is correct for any NPE in this range. */
#define MAX_NPE 32

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign matrix-free FEM matvec; unstructured-grid kernel)\n"
"  --nodes N             Number of mesh nodes; result vector y is N * 8 bytes (default 1000000)\n"
"  --elements E          Number of elements (random connectivity) (default 2000000)\n"
"  --npe NPE             Nodes per element / local matrix side (default 4; 2..32)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for x and connectivity (default 42)\n"
"  --max-mb N            Hard cap on buffer bytes (y + x + connectivity) (default 8192)\n"
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

/* Fill the fixed symmetric local element matrix ke (NPE x NPE), stored row-major
 * in the caller's flat buffer. ke[i][j] = (i==j ? NPE-1 : -1): every off-diagonal
 * is -1 and the diagonal balances the row to zero -- a graph-Laplacian element
 * stencil. Symmetric by construction, which the assembled-matrix cross-check
 * relies on. */
static void build_ke(double *ke, size_t npe) {
    for (size_t i = 0; i < npe; i++)
        for (size_t j = 0; j < npe; j++)
            ke[i * npe + j] = (i == j) ? (double)(npe - 1) : -1.0;
}

/* (Re)generate the random element connectivity: each element names NPE node
 * indices in [0, N). This is the unstructured mesh -- offsets are data-driven,
 * not a fixed grid stride, so the later gather/scatter are indirect. */
static void seed_connectivity(int *elem, size_t E, size_t npe, size_t N,
                              p2_rng_t *rng) {
    size_t total = E * npe;
    for (size_t k = 0; k < total; k++)
        elem[k] = (int)(p2_rng_next(rng) % (uint64_t)N);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nodes      = p2_get_i64(argc, argv, "--nodes", 1000000);
    long long elements   = p2_get_i64(argc, argv, "--elements", 2000000);
    long long npe_arg    = p2_get_i64(argc, argv, "--npe", 4);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nodes < 16 || nodes > (1LL << 30)) {
        P2_LOG_ERR("nodes %lld out of range (16..2^30)", nodes);
        return 2;
    }
    if (elements < 1 || elements > (1LL << 31)) {
        P2_LOG_ERR("elements %lld out of range (1..2^31)", elements);
        return 2;
    }
    if (npe_arg < 2 || npe_arg > MAX_NPE) {
        P2_LOG_ERR("npe %lld out of range (2..%d)", npe_arg, MAX_NPE);
        return 2;
    }
    size_t N   = (size_t)nodes;
    size_t E   = (size_t)elements;
    size_t NPE = (size_t)npe_arg;

    /* y is the dominant written buffer (the scatter-add target). x is the read
     * source; the connectivity table is the mesh. All three are sized here and
     * jointly capped by --max-mb. */
    size_t y_bytes    = N * sizeof(double);
    size_t x_bytes    = N * sizeof(double);
    size_t conn_bytes = E * NPE * sizeof(int);
    size_t total_bytes = y_bytes + x_bytes + conn_bytes;
    if (total_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("total bytes %zu exceed --max-mb %lld", total_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Unstructured Grids");
    p2_meta_kv_str(&m, "scheme",
                   "matrix-free FEM matvec (element gather -> local matvec -> scatter-add into result vector y; K never assembled)");
    p2_meta_kv_str(&m, "d6_variant",
                   "quieter member: output is a vector (matrix-free analog of SpMV), not the large global-matrix assembly");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "nodes", nodes);
    p2_meta_kv_i64(&m, "elements", elements);
    p2_meta_kv_i64(&m, "npe", npe_arg);
    p2_meta_kv_u64(&m, "y_bytes", y_bytes);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* y is the dominant buffer -> mmap + mlock it (it is zeroed and then
     * scatter-added into every pass, which is the workload's signature write). */
    double *y = (double *)mmap(NULL, y_bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    double *x = (double *)mmap(NULL, x_bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    int *elem = (int *)mmap(NULL, conn_bytes, PROT_READ | PROT_WRITE,
                            MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (y == MAP_FAILED || x == MAP_FAILED || elem == MAP_FAILED) {
        P2_LOG_ERR("mmap(y %zu / x %zu / conn %zu) failed: %s",
                   y_bytes, x_bytes, conn_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(y, y_bytes, MADV_NOHUGEPAGE);
    p2_madvise(x, x_bytes, MADV_NOHUGEPAGE);
    p2_madvise(elem, conn_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) {
        p2_mlock_soft(y, y_bytes);
        p2_mlock_soft(x, x_bytes);
        p2_mlock_soft(elem, conn_bytes);
    }

    /* Fixed local element matrix ke and per-element scratch (gathered node values
     * xe and their local matvec result ye). These are tiny (NPE-sized) and stay
     * in cache; they are not part of the observable write front. */
    double ke[MAX_NPE * MAX_NPE];
    double xe[MAX_NPE];
    double ye[MAX_NPE];
    build_ke(ke, NPE);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Initial x (random node values) and initial random mesh connectivity. Both
     * are re-seeded again inside the measure loop; this warmup fill just touches
     * every page so the timed passes are not paying first-touch faults. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) x[i] = p2_rng_unit(&rng);
    seed_connectivity(elem, E, NPE, N, &rng);
    for (size_t i = 0; i < N; i++) y[i] = 0.0;
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed the input node vector x and the mesh connectivity so each pass
         * writes fresh values through fresh (data-driven) offsets, then clear the
         * whole result vector -- one dense sequential write front over y. */
        p2_rng_seed(&rng, seed + passes);
        for (size_t i = 0; i < N; i++) x[i] = p2_rng_unit(&rng);
        seed_connectivity(elem, E, NPE, N, &rng);
        for (size_t i = 0; i < N; i++) y[i] = 0.0;

        /* Matrix-free matvec: for each element, gather its node values, apply the
         * small local matrix ke, and scatter-add the result back into y. Summed
         * over all elements this equals K x, with K never assembled. The scatter
         * (indirect read-modify-write of y at elem[e][*]) is the signature write. */
        for (size_t e = 0; e < E; e++) {
            const int *idx = elem + e * NPE;          /* this element's node list */
            for (size_t i = 0; i < NPE; i++)
                xe[i] = x[idx[i]];                     /* (a) gather node values   */
            for (size_t i = 0; i < NPE; i++) {         /* (b) local NPE x NPE apply */
                const double *ki = ke + i * NPE;
                double acc = 0.0;
                for (size_t j = 0; j < NPE; j++)
                    acc += ki[j] * xe[j];
                ye[i] = acc;
            }
            for (size_t i = 0; i < NPE; i++)
                y[idx[i]] += ye[i];                    /* (c) scatter-add into y    */
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = y[N / 2];                   /* prevent dead-code elim */

    munmap(y, y_bytes);
    munmap(x, x_bytes);
    munmap(elem, conn_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "y_mid_value", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "quieter D6 member: output is a vector (matrix-free analog of SpMV); scatter-add into y is the observable write, not a bulk matrix assembly");
    p2_meta_close(&m);
    return 0;
}
