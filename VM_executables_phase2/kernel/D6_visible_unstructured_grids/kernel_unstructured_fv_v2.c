/* kernel_unstructured_fv_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Unstructured Grids dwarf (Berkeley motif D6), the finite-volume variant. A
 * finite-volume (FV) solver stores a conserved scalar u on N mesh CELLS and
 * updates it by exchanging FLUXES across FACES: every face joins exactly two
 * cells and carries some quantity from one into the other. The mesh is
 * UNSTRUCTURED, so a cell's neighbours are not its grid-index neighbours --
 * they are named by an explicit face list (cL[f], cR[f]), an arbitrary pair of
 * cell indices per face. That indirection is the whole point of the dwarf.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs structured stencils, D5)
 * ----------------------------------------------------------------------------
 * A structured-grid stencil (D5) writes its output as one dense, sequential
 * front: cell (i,j) is written right after (i,j-1), so the write stream is
 * perfectly contiguous and predictable. This FV update does the opposite. It
 * walks the FACE list -- not the cell array -- and for each face SCATTER-ADDS a
 * flux into TWO cell slots chosen by cL[f] and cR[f]. Because the faces connect
 * arbitrary cell pairs, those two destination slots jump all over the du array:
 * the write stream is scattered and data-dependent, addressed indirectly
 * through the face endpoints. That indirect, scattered scatter-add into the
 * cell array is the tell that separates an unstructured mesh from a structured
 * one, even though both ultimately rewrite an array of cells.
 *
 * WHY THE SCATTER IS EQUAL-AND-OPPOSITE (conservation, the defining property)
 * ----------------------------------------------------------------------------
 * A face flux is a transfer: whatever leaves cell cL across face f must arrive
 * in cell cR. So for a flux F we do  du[cL] -= F  and  du[cR] += F  -- the same
 * magnitude, opposite sign, into the two endpoints. Summed over the whole mesh
 * every face contributes +F and -F, which cancel, so the TOTAL of du is exactly
 * zero and sum(u) is unchanged by the update. That exact conservation is what
 * makes the scheme "finite-volume", and it is the non-circular property the
 * verifier checks (a plain scatter that only added F would not conserve).
 *
 * ============================================================================
 * ALGORITHM (one explicit FV step, timestep dt)
 * ============================================================================
 *   1. Zero the accumulator: du[i] = 0 for every cell i.
 *   2. Face loop (the signature write): for each face f, read the two endpoint
 *      cell values u[cL[f]] and u[cR[f]], form the central/diffusive flux
 *          F = a[f] * (u[cL[f]] - u[cR[f]]),
 *      where a[f] is a fixed face coefficient (a random velocity x area proxy),
 *      then SCATTER-ADD it conservatively:  du[cL[f]] -= F;  du[cR[f]] += F.
 *   3. Explicit update: u[i] += dt * du[i] for every cell i.
 *
 * A face's flux is proportional to the value DIFFERENCE across it, so the update
 * diffuses u toward its neighbours -- a stable, physical central scheme. Faces
 * are built once as random cell pairs with cL != cR (no self-loops), the way an
 * arbitrary unstructured connectivity would look to the memory subsystem.
 *
 * The u array (N cells) is the mmap'd DOMINANT buffer -- the thing every step
 * ultimately rewrites and the surface the host write-signal observes. du (N)
 * and the face arrays (cL, cR, a) support the update.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Unstructured Grids. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (FV steps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_unstructured_fv_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign unstructured finite-volume; Unstructured-Grids kernel)\n"
"  --cells N            Number of mesh cells (default 1000000; u+du use 2 * N*8 bytes)\n"
"  --faces-per-cell FPC Faces per cell x1; NF = N*FPC/2 (default 3)\n"
"  --dt-milli DT        Timestep x1000 (default 100 = 0.1)\n"
"  --duration SEC       Measurement duration (default 60)\n"
"  --warmup SEC         Warm-up duration (default 2)\n"
"  --seed N             PRNG seed for field + face connectivity (default 42)\n"
"  --max-mb N           Hard cap on total buffer bytes (default 8192)\n"
"  --no-mlock           Skip mlock() entirely\n"
"  --output-dir PATH    Where to write metadata JSON\n"
"  --cpu-affinity N     Pin to CPU N\n"
"  --phase-markers      Emit phase markers to stderr\n"
"  --dry-run            Validate args and exit\n"
"  --help               Show this help\n", p);
}

/* uniform double in [0,1) from the xoshiro stream */
static inline double p2_rng_unit(p2_rng_t *r) {
    return (double)(p2_rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* Draw a cell index in [0, N) from the PRNG stream. */
static inline size_t rng_cell(p2_rng_t *r, size_t N) {
    return (size_t)(p2_rng_next(r) % (uint64_t)N);
}

/* Seed the mesh: fill the cell field u and build the random face connectivity.
 * Each face joins two DISTINCT cells (cL != cR, no self-loops) and carries a
 * fixed positive coefficient a in [0.5, 1.5). Called once per measure pass so
 * every timed step starts from the same field and the same mesh -- the field is
 * what the FV step then rewrites. */
static void seed_mesh(p2_rng_t *rng, uint64_t seed, size_t N, size_t NF,
                      double *u, size_t *cL, size_t *cR, double *a) {
    p2_rng_seed(rng, seed);
    for (size_t i = 0; i < N; i++)
        u[i] = p2_rng_unit(rng);                 /* conserved scalar, cell-centred */
    for (size_t f = 0; f < NF; f++) {
        size_t l = rng_cell(rng, N);
        size_t r = rng_cell(rng, N);
        while (r == l) r = rng_cell(rng, N);     /* forbid a face onto the same cell */
        cL[f] = l;
        cR[f] = r;
        a[f]  = 0.5 + p2_rng_unit(rng);          /* face coefficient (velocity x area) */
    }
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long ncells     = p2_get_i64(argc, argv, "--cells", 1000000);
    long long fpc        = p2_get_i64(argc, argv, "--faces-per-cell", 3);
    /* dt is passed as integer-milli because the phase2 arg helpers are
     * integer-only: --dt-milli 100 = 0.1. */
    long long dt_milli   = p2_get_i64(argc, argv, "--dt-milli", 100);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    double dt = (double)dt_milli / 1000.0;

    if (ncells < 16 || ncells > (1LL << 28)) {
        P2_LOG_ERR("cells %lld out of range (16..2^28)", ncells);
        return 2;
    }
    if (fpc < 1 || fpc > 64) {
        P2_LOG_ERR("faces-per-cell %lld out of range (1..64)", fpc);
        return 2;
    }
    size_t N = (size_t)ncells;
    size_t NF = (N * (size_t)fpc) / 2;               /* each face is shared by 2 cells */
    if (NF < 1) NF = 1;
    /* Buffers: u (N doubles) + du (N doubles) + faces (cL,cR each NF size_t; a
     * NF doubles). u is the dominant, host-visible cell array. */
    size_t cell_bytes = (size_t)N * sizeof(double);
    size_t face_bytes = (size_t)NF * (2 * sizeof(size_t) + sizeof(double));
    size_t total_bytes = 2 * cell_bytes + face_bytes;
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
    p2_meta_kv_str(&m, "scheme", "unstructured finite-volume; conservative face-flux scatter-add into cells");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "cells", ncells);
    p2_meta_kv_i64(&m, "faces_per_cell", fpc);
    p2_meta_kv_u64(&m, "faces", NF);
    p2_meta_kv_i64(&m, "dt_milli", dt_milli);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The cell field u is the dominant, host-visible buffer -> mmap + mlock it.
     * du is the same size (the per-step scatter target); keep it mmap'd too so
     * the scatter-add writes land in a pinned, huge-page-free region. */
    double *u  = (double *)mmap(NULL, cell_bytes, PROT_READ | PROT_WRITE,
                                MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    double *du = (double *)mmap(NULL, cell_bytes, PROT_READ | PROT_WRITE,
                                MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (u == MAP_FAILED || du == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu x2) failed: %s", cell_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(u, cell_bytes, MADV_NOHUGEPAGE);
    p2_madvise(du, cell_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) { p2_mlock_soft(u, cell_bytes); p2_mlock_soft(du, cell_bytes); }

    /* Face connectivity (built once, re-seeded each pass): cL, cR name the two
     * cells each face couples; a is its coefficient. These drive the indirect,
     * scattered addressing of the scatter-add. */
    size_t *cL = (size_t *)malloc((size_t)NF * sizeof(size_t));
    size_t *cR = (size_t *)malloc((size_t)NF * sizeof(size_t));
    double *a  = (double *)malloc((size_t)NF * sizeof(double));
    if (!cL || !cR || !a) {
        free(cL); free(cR); free(a);
        munmap(u, cell_bytes); munmap(du, cell_bytes);
        P2_LOG_ERR("malloc(faces) failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng;
    seed_mesh(&rng, seed, N, NF, u, cL, cR, a);      /* initial field + mesh */
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t steps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* Re-seed the field and mesh so every timed step starts identically,
         * then run ONE explicit finite-volume step and count it. */
        seed_mesh(&rng, seed, N, NF, u, cL, cR, a);

        /* (1) Zero the flux accumulator: one dense sequential write over du. */
        for (size_t i = 0; i < N; i++) du[i] = 0.0;

        /* (2) Face loop -- the signature write. Walk the FACE list, not the cell
         * array. For each face read its two endpoint cell values, form the
         * central/diffusive flux, and SCATTER-ADD it equal-and-opposite into the
         * two endpoint slots of du. The destinations cL[f], cR[f] are arbitrary
         * cell indices, so these are scattered, indirectly addressed, data-
         * dependent writes -- the unstructured-mesh tell. The +F/-F pairing is
         * what conserves the total (every face cancels itself in the sum). */
        for (size_t f = 0; f < NF; f++) {
            size_t l = cL[f], r = cR[f];
            double F = a[f] * (u[l] - u[r]);         /* flux from cL toward cR */
            du[l] -= F;                              /* leaves cell cL           */
            du[r] += F;                              /* arrives in cell cR       */
        }

        /* (3) Explicit update of the cell field: one dense sequential write over
         * u, the dominant host-visible buffer. */
        for (size_t i = 0; i < N; i++) u[i] += dt * du[i];

        steps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = u[N / 2];                 /* prevent dead-code elim */

    free(cL); free(cR); free(a);
    munmap(u, cell_bytes);
    munmap(du, cell_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "steps", steps);
    p2_meta_kv_f64(&m, "center_value", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: scattered conservative scatter-add into the cell array via the face list, distinct from dense structured-stencil writes");
    p2_meta_close(&m);
    return 0;
}
