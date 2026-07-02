/* kernel_fem_assembly_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Unstructured Grids dwarf (Berkeley motif D6): a PDE computation on an
 * IRREGULAR mesh. On a structured grid (D5) a cell's neighbours are its
 * index +-1 / +-N -- known by arithmetic, so the write is dense and sequential.
 * An unstructured mesh has no such rule: which nodes a mesh element touches is
 * stored explicitly in a CONNECTIVITY LIST, and the computation reaches its data
 * through that list (indirect gather/scatter). This kernel runs the archetypal
 * unstructured-grid operation -- finite-element stiffness ASSEMBLY -- which
 * scatter-ADDS each element's small local matrix into a single LARGE global
 * matrix. That assembled global matrix is the big output, and filling it is the
 * signature write.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs the structured-grid stencils)
 * ----------------------------------------------------------------------------
 * A structured-grid stencil (D5) writes one dense, contiguous grid front per
 * sweep: fully sequential, perfectly predictable addresses. Assembly writes the
 * same total region (the global matrix K) but reaches it by INDIRECTION: the
 * destination of every update, K[elem[e][i]*N + elem[e][j]], comes out of the
 * connectivity list, so the write is an indexed SCATTER-ADD (read-modify-write
 * at data-dependent, irregular offsets) rather than a linear stream. Same-sized
 * footprint, but scattered accumulate instead of sequential fill -- that is the
 * tell that separates unstructured assembly from structured stencils.
 *
 * ============================================================================
 * ALGORITHM (finite-element stiffness assembly)
 * ============================================================================
 *   N nodes, E elements, NPE nodes per element (default 4). The mesh is given by
 *   a connectivity list elem[e][0..NPE-1] = the NPE node indices that element e
 *   touches (here random indices in [0,N) -- a synthetic irregular mesh). Every
 *   element contributes the SAME fixed, symmetric NPE x NPE local element matrix
 *   ke, a local graph Laplacian: ke[i][j] = (i==j ? NPE-1 : -1).
 *
 *   Global stiffness matrix K is stored DENSE, N x N, row-major, in one mmap'd
 *   buffer (the dominant, visible write). Each measure pass:
 *     1. Zero K (clear the previous assembly).
 *     2. RE-SEED the element connectivity (fresh random mesh -> the scatter
 *        addresses change every pass, keeping the write pattern live).
 *     3. ASSEMBLE: for each element e, for i in 0..NPE-1, for j in 0..NPE-1:
 *            K[ elem[e][i]*N + elem[e][j] ] += ke[i][j];
 *        i.e. scatter-add element e's local matrix into the global rows/columns
 *        named by its connectivity. This indexed accumulate IS the workload.
 *   Passes are counted for the whole timed duration.
 *
 * Storing K dense (rather than sparse) is a deliberate choice: it makes the
 * assembled global matrix a single large, contiguous target so the host
 * write-signal sees one big region filled by scattered accumulates.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Unstructured Grids. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (assembly passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_fem_assembly_v2";

/* Upper bound on nodes-per-element: keeps the fixed-size element scratch (elem
 * row + local matrix ke) small and bounds the inner assembly loops. Real element
 * types (tri=3, tet/quad=4, hex=8) sit well under this. */
#define MAX_NPE 64

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign FEM stiffness assembly; unstructured-grid kernel)\n"
"  --nodes N             Number of mesh nodes (default 2048; K uses N*N*8 bytes)\n"
"  --elements E          Number of mesh elements (default 8192)\n"
"  --npe NPE             Nodes per element (default 4; range 2..64)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the connectivity (default 42)\n"
"  --max-mb N            Hard cap on global-matrix bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* Draw a node index in [0,N): the connectivity list is filled from this, so the
 * assembly targets scattered, data-dependent offsets into the global matrix. */
static inline size_t rng_index(p2_rng_t *r, size_t N) {
    return (size_t)(p2_rng_next(r) % (uint64_t)N);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nodes      = p2_get_i64(argc, argv, "--nodes", 2048);
    long long elements   = p2_get_i64(argc, argv, "--elements", 8192);
    long long npe        = p2_get_i64(argc, argv, "--npe", 4);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    if (nodes < 16 || nodes > 65536) {
        P2_LOG_ERR("nodes %lld out of range (16..65536)", nodes);
        return 2;
    }
    if (elements < 1 || elements > (1LL << 26)) {
        P2_LOG_ERR("elements %lld out of range (1..2^26)", elements);
        return 2;
    }
    if (npe < 2 || npe > MAX_NPE) {
        P2_LOG_ERR("npe %lld out of range (2..%d)", npe, MAX_NPE);
        return 2;
    }
    size_t N   = (size_t)nodes;
    size_t E   = (size_t)elements;
    size_t NPE = (size_t)npe;
    size_t cells = N * N;                            /* dense global matrix entries */
    size_t buf_bytes = cells * sizeof(double);       /* K = N*N*8 bytes */
    if (buf_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("global-matrix bytes %zu exceed --max-mb %lld", buf_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Unstructured Grids");
    p2_meta_kv_str(&m, "scheme", "FEM stiffness assembly (element matrices scatter-added into a dense global matrix)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "nodes", nodes);
    p2_meta_kv_i64(&m, "elements", elements);
    p2_meta_kv_i64(&m, "npe", npe);
    p2_meta_kv_u64(&m, "global_matrix_bytes", buf_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The dense global matrix K is the dominant buffer -> mmap + mlock it. It is
     * zeroed and refilled by scatter-add every pass, which is the workload's
     * signature write (a large region filled at irregular, indexed offsets). */
    double *K = (double *)mmap(NULL, buf_bytes, PROT_READ | PROT_WRITE,
                               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (K == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu) failed: %s", buf_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(K, buf_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) p2_mlock_soft(K, buf_bytes);

    /* Connectivity list elem[E][NPE] (flat) and the fixed local element matrix
     * ke[NPE][NPE]. The connectivity is the indirection table that turns the
     * sequential element loop into scattered writes into K. */
    size_t *elem = (size_t *)malloc(E * NPE * sizeof(size_t));
    double *ke   = (double *)malloc(NPE * NPE * sizeof(double));
    if (!elem || !ke) {
        free(elem); free(ke);
        munmap(K, buf_bytes);
        P2_LOG_ERR("malloc failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    /* Fixed symmetric local element matrix: a local graph Laplacian.
     * ke[i][j] = (i==j ? NPE-1 : -1). Each element contributes this same block;
     * assembling it over the mesh builds the global graph Laplacian of the mesh. */
    for (size_t i = 0; i < NPE; i++)
        for (size_t j = 0; j < NPE; j++)
            ke[i * NPE + j] = (i == j) ? (double)(NPE - 1) : -1.0;

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Seed an initial random mesh so the first measured pass has connectivity to
     * scatter through; every pass re-seeds it below to keep the pattern live. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t e = 0; e < E; e++)
        for (size_t i = 0; i < NPE; i++)
            elem[e * NPE + i] = rng_index(&rng, N);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* (1) Zero the global matrix: clear the previous assembly so this pass
         * starts from a blank K (one dense sequential clear of the whole region). */
        for (size_t c = 0; c < cells; c++) K[c] = 0.0;

        /* (2) Re-seed the connectivity list: a fresh random mesh means the
         * scatter addresses differ from the last pass, so the assembly keeps
         * hitting new, data-dependent offsets rather than a fixed pattern. */
        for (size_t e = 0; e < E; e++)
            for (size_t i = 0; i < NPE; i++)
                elem[e * NPE + i] = rng_index(&rng, N);

        /* (3) Assemble: scatter-add every element's local matrix into K at the
         * global rows/columns named by its connectivity. The destination offset
         * elem[e][i]*N + elem[e][j] comes out of the connectivity list, so this
         * is an indexed read-modify-write at irregular addresses -- the defining
         * unstructured-grid write, as opposed to a structured stencil's dense
         * sequential fill. */
        for (size_t e = 0; e < E; e++) {
            const size_t *conn = elem + e * NPE;    /* this element's node indices */
            for (size_t i = 0; i < NPE; i++) {
                size_t ri = conn[i] * N;            /* global row base for local i */
                const double *kerow = ke + i * NPE;
                for (size_t j = 0; j < NPE; j++) {
                    /* scatter-add: accumulate local entry (i,j) into the global
                     * matrix at the (conn[i], conn[j]) position. */
                    K[ri + conn[j]] += kerow[j];
                }
            }
        }
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = K[0];                    /* prevent dead-code elim */

    free(elem); free(ke);
    munmap(K, buf_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_f64(&m, "k00_value", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: indexed scatter-add into a large global matrix (irregular offsets) vs structured stencils' sequential fill");
    p2_meta_close(&m);
    return 0;
}
