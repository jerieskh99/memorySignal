/* kernel_dg_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Unstructured Grids dwarf (Berkeley motif D6), a Discontinuous Galerkin (DG)
 * time-step. A DG solver splits the domain into E small ELEMENTS. Each element
 * carries its own little block of P degrees-of-freedom (DOFs) -- think of it as
 * the local polynomial coefficients of the solution inside that element. One
 * time-step advances every element's block using two contributions:
 *
 *   VOLUME term : an element-LOCAL dense operator M (a fixed P x P matrix, the
 *                 differentiation/mass operator) applied to that element's own
 *                 DOF vector. This is a small dense matvec per element.
 *   FLUX  term  : a conservative exchange with the element's face-neighbours.
 *                 Neighbouring elements do not share DOFs (that is what makes
 *                 the mesh "discontinuous"); they only talk through fluxes
 *                 across the shared faces. We use a simple centred/penalty flux
 *                 alpha*(U[n] - U[e]) summed over neighbours -- antisymmetric in
 *                 (e,n), so on a symmetric neighbour graph the total is conserved.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs a structured-grid stencil, D5)
 * ----------------------------------------------------------------------------
 * The structured-grid stencil (kernel_stencil_jacobi_v2) writes ONE scalar per
 * grid point from a fixed 5-point neighbourhood: a single dense sweep with a
 * clean, regular footprint. This DG step instead rewrites, per element, a whole
 * small DENSE BLOCK of P DOFs (the P x P volume matvec touches all P outputs from
 * all P inputs), and the neighbour coupling is a SCATTER through an irregular,
 * data-dependent index list nbr[e][*] rather than fixed +/-1 offsets. The
 * host-visible write is the same shape every step -- the full E x P solution
 * array rewritten -- but it is produced by many small dense blocks plus an
 * irregular gather from neighbour blocks, not one flat scalar stencil. That
 * block-structured-plus-irregular-gather write is the distinct tell of an
 * unstructured-grid method.
 *
 * ============================================================================
 * ALGORITHM (one DG time-step, double-buffered)
 * ============================================================================
 *   For every element e (row e of the E x P solution array U):
 *     1. VOLUME: vol = M * U[e, :]                 (P x P dense matvec, local)
 *     2. FLUX  : flux = sum over neighbours n of nbr[e]:  alpha * (U[n,:] - U[e,:])
 *     3. WRITE : Unew[e, :] = U[e, :] + vol + flux (the full block is rewritten)
 *   Then swap U and Unew, so the block just written becomes next step's input.
 *   Each pass re-seeds U (and rebuilds the neighbour graph) so the timed loop
 *   keeps issuing the signature write; passes are counted.
 *
 * The neighbour list nbr[e][*] holds random element indices (self-references and
 * duplicates are harmless: alpha*(U[e]-U[e]) = 0). A conservative flux only truly
 * conserves the global total when the graph is symmetric; the standalone verifier
 * (/tmp/vdg.c) builds a symmetric graph to check that property directly.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Unstructured Grids. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (DG steps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_dg_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Discontinuous Galerkin step; unstructured-grid kernel)\n"
"  --elements E          Number of mesh elements (default 65536)\n"
"  --dofs P              Degrees-of-freedom per element block (default 16; 2..64)\n"
"  --neighbors F         Face-neighbours per element (default 3; 1..32)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the field and neighbour graph (default 42)\n"
"  --max-mb N            Hard cap on solution-buffer bytes (default 8192)\n"
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

/* The fixed element-local volume operator, entry (a,b) of the P x P matrix M.
 * Diagonal-dominant so the local matvec stays well-conditioned; the exact form
 * does not matter for the memory signature, only that it couples every output
 * DOF to every input DOF (that is what makes the per-element write a dense block
 * rather than a single scalar). The verifier reproduces this same formula with an
 * independent plain matvec, so the definition here is the single source of truth. */
static inline double dg_M(int a, int b) {
    int d = a - b; if (d < 0) d = -d;                  /* |a - b| */
    return (a == b) ? 0.5 : (0.1 / (1.0 + (double)d));
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long elements   = p2_get_i64(argc, argv, "--elements", 65536);
    long long dofs       = p2_get_i64(argc, argv, "--dofs", 16);
    long long neighbors  = p2_get_i64(argc, argv, "--neighbors", 3);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);

    /* alpha: the flux penalty coefficient (kept small so the coupling is a gentle,
     * conservative exchange rather than a dominant term). Fixed, not a flag. */
    const double alpha = 0.05;

    if (elements < 16 || elements > (1LL << 24)) {
        P2_LOG_ERR("elements %lld out of range (16..2^24)", elements); return 2;
    }
    if (dofs < 2 || dofs > 64) {
        P2_LOG_ERR("dofs %lld out of range (2..64)", dofs); return 2;
    }
    if (neighbors < 1 || neighbors > 32) {
        P2_LOG_ERR("neighbors %lld out of range (1..32)", neighbors); return 2;
    }
    size_t E = (size_t)elements;
    size_t P = (size_t)dofs;
    size_t F = (size_t)neighbors;
    size_t sol_cells  = E * P;                          /* entries in one solution array */
    size_t sol_bytes  = sol_cells * sizeof(double);     /* bytes of one E x P array */
    size_t total_bytes = 2 * sol_bytes;                 /* double-buffered: U + Unew */
    if (total_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("solution bytes %zu exceed --max-mb %lld", total_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL");
    p2_meta_kv_str(&m, "dwarf", "Unstructured Grids");
    p2_meta_kv_str(&m, "scheme", "Discontinuous Galerkin step (per-element dense volume matvec + conservative neighbour flux)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "elements", elements);
    p2_meta_kv_i64(&m, "dofs", dofs);
    p2_meta_kv_i64(&m, "neighbors", neighbors);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The two E x P solution arrays are the dominant, host-visible buffers: mmap
     * them and pin them. Unew is the write target each step; the pointer swap
     * makes that target alternate between the two regions, period-2, exactly like
     * a double-buffered field update. */
    double *U    = (double *)mmap(NULL, sol_bytes, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    double *Unew = (double *)mmap(NULL, sol_bytes, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (U == MAP_FAILED || Unew == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu x2) failed: %s", sol_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(U, sol_bytes, MADV_NOHUGEPAGE);
    p2_madvise(Unew, sol_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) { p2_mlock_soft(U, sol_bytes); p2_mlock_soft(Unew, sol_bytes); }

    /* Neighbour list: F element indices per element (E x F, row-major). This is
     * the irregular, data-dependent index structure that the flux gathers
     * through -- the counterpart of the fixed +/-1 offsets in a structured
     * stencil. It is rebuilt each pass alongside the field re-seed. */
    int *nbr = (int *)malloc(E * F * sizeof(int));
    /* Materialise the fixed P x P operator once into a small dense buffer so the
     * inner matvec reads it as a contiguous row-major array (cache-friendly, and
     * independent of the dg_M() call form). */
    double *Mop = (double *)malloc(P * P * sizeof(double));
    if (!nbr || !Mop) {
        free(nbr); free(Mop);
        munmap(U, sol_bytes); munmap(Unew, sol_bytes);
        P2_LOG_ERR("malloc failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    for (size_t a = 0; a < P; a++)
        for (size_t b = 0; b < P; b++)
            Mop[a * P + b] = dg_M((int)a, (int)b);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Initial field: every DOF random. Neighbour graph: random element indices in
     * [0, E). Both are regenerated each pass in the measure loop below; this first
     * fill just makes the buffers resident before timing starts. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < sol_cells; i++) U[i] = p2_rng_unit(&rng);
    for (size_t i = 0; i < E * F; i++) nbr[i] = (int)(p2_rng_next(&rng) % (uint64_t)E);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t iters = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed the field and rebuild the neighbour graph so each pass performs
         * the full DG write on fresh data (matching the re-seed+count structure of
         * the sibling kernels). U is the read source; Unew is the write target. */
        p2_rng_seed(&rng, seed + iters);
        for (size_t i = 0; i < sol_cells; i++) U[i] = p2_rng_unit(&rng);
        for (size_t i = 0; i < E * F; i++) nbr[i] = (int)(p2_rng_next(&rng) % (uint64_t)E);

        /* One DG step. Each element rewrites its whole P-DOF block of Unew. */
        for (size_t e = 0; e < E; e++) {
            const double *ue = U + e * P;              /* this element's DOF block */
            const int    *ne = nbr + e * F;            /* this element's neighbours */
            double *out = Unew + e * P;                /* destination block in Unew */

            /* VOLUME term: out = M * ue  (dense P x P matvec into the block). Every
             * output DOF a sums over every input DOF b -- the dense coupling that
             * distinguishes this from a single-scalar stencil. Seed with the
             * element's own value so out already holds U[e] + volume. */
            for (size_t a = 0; a < P; a++) {
                const double *Mrow = Mop + a * P;
                double acc = 0.0;
                for (size_t b = 0; b < P; b++) acc += Mrow[b] * ue[b];
                out[a] = ue[a] + acc;                  /* U[e,a] + volume_a */
            }

            /* FLUX term: for each neighbour n, add alpha*(U[n,:] - U[e,:]) to the
             * block. This is the conservative face exchange; the gather U[n,:]
             * follows the irregular nbr index, the antisymmetric form is what
             * conserves the global total on a symmetric graph. */
            for (size_t f = 0; f < F; f++) {
                const double *un = U + (size_t)ne[f] * P;   /* neighbour DOF block */
                for (size_t a = 0; a < P; a++)
                    out[a] += alpha * (un[a] - ue[a]);
            }
        }
        /* Swap the buffer roles: the solution just written becomes the read source
         * for the next step (a pointer swap, not a copy). */
        double *tmp = U; U = Unew; Unew = tmp;
        iters++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = U[(sol_cells) / 2];         /* prevent dead-code elim */

    free(nbr); free(Mop);
    munmap(U, sol_bytes);
    munmap(Unew, sol_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "iterations", iters);
    p2_meta_kv_f64(&m, "mid_value", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: per-element dense block writes (volume) + irregular neighbour-gather flux, whole solution rewritten each step; conservation holds on a symmetric neighbour graph");
    p2_meta_close(&m);
    return 0;
}
