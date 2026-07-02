/* kernel_mesh_smooth_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Unstructured Grids dwarf (Berkeley motif D6): a computation on an IRREGULAR
 * mesh. On a structured grid (D5) a node's neighbours are its index +-1 / +-N --
 * known by arithmetic, so the stencil write is dense and sequential. An
 * unstructured mesh has no such rule: which nodes are adjacent is stored
 * explicitly in an ADJACENCY LIST, and the computation reaches its neighbours
 * through that list (indirect gather). This kernel runs the archetypal
 * unstructured-grid operation -- LAPLACIAN MESH SMOOTHING -- where each node is
 * pulled toward the average of its irregular neighbours, rewriting the whole node
 * value array each sweep. That full node-array rewrite is the signature write.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (vs the structured-grid stencils)
 * ----------------------------------------------------------------------------
 * The write itself is an unstructured stencil: for every node we gather its
 * neighbour values through the adjacency list, then write one dense node-value
 * array (val_new), swap buffers, repeat. HONEST CAVEAT: the WRITE footprint here
 * closely RESEMBLES a structured-grid stencil (D5) -- both do a full, contiguous
 * rewrite of one value array per sweep into a double buffer, so a host
 * write-signal sees nearly the same dense sequential write front and the same
 * period-2 ping-pong between two fixed regions. The distinction from D5 is on the
 * READ side: the neighbour gather here is INDIRECT (val[adj_idx[k]] at
 * data-dependent, scattered offsets) instead of the +-1 / +-N arithmetic of a
 * structured grid. Reads are largely invisible to a write-only host signal, so
 * the irregular access that makes this "unstructured" is mostly a read-side tell;
 * the emitted write pattern is close to D5's by construction.
 *
 * ============================================================================
 * ALGORITHM (unstructured Laplacian smoothing)
 * ============================================================================
 *   N nodes, each holding one scalar value. Irregular adjacency in CSR form:
 *     adj_ptr[N+1]         -- adj_ptr[i]..adj_ptr[i+1] is node i's neighbour slice
 *     adj_idx[total_edges] -- the neighbour node indices, concatenated
 *   Here each node is given ~degree random neighbours (a synthetic irregular
 *   mesh), so deg_i = adj_ptr[i+1] - adj_ptr[i]. Value arrays val[N] and
 *   val_new[N] form a double buffer, mmap'd as the dominant write buffers.
 *
 *   Smoothing update with weight w in [0,1]:
 *     val_new[i] = (1-w)*val[i] + w * (1/deg_i) * sum over neighbours j of val[j]
 *   If deg_i == 0 the node has no neighbours to average, so val_new[i] = val[i].
 *   Then swap val/val_new so the next sweep reads the array just written.
 *
 *   Each measure pass:
 *     1. RE-SEED the adjacency (fresh random neighbour lists -> the gather offsets
 *        change every pass, keeping the irregular read pattern live).
 *     2. RE-SEED val (fresh random node field) so the value array keeps evolving
 *        rather than converging to a flat fixed point and going quiescent.
 *     3. Run ONE smoothing sweep (indirect gather + full val_new rewrite + swap).
 *   Sweeps are counted for the whole timed duration.
 *
 * Storing the field as two plain N-length arrays (rather than in-place) is a
 * deliberate double-buffer choice: it makes each sweep one large, contiguous
 * write target so the host write-signal sees one dense region filled per sweep.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * Signature family: KERNEL. Dwarf: Unstructured Grids. See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (smoothing sweeps) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_mesh_smooth_v2";

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign Laplacian mesh smoothing; unstructured-grid kernel)\n"
"  --nodes N             Number of mesh nodes (default 1000000; val+val_new use 2*N*8 bytes)\n"
"  --degree D            Average neighbours per node (default 6; adjacency = N*D*4 bytes)\n"
"  --weight-milli W      Smoothing weight x1000 (default 500 = 0.5; range 0..1000)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for the field and adjacency (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes (default 8192)\n"
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

/* Draw a node index in [0,N): neighbour lists are filled from this, so the
 * smoothing gather reaches scattered, data-dependent offsets into the value
 * array (the irregular access that makes this mesh "unstructured"). */
static inline uint32_t rng_index(p2_rng_t *r, size_t N) {
    return (uint32_t)(p2_rng_next(r) % (uint64_t)N);
}

/* Fill the adjacency slice of every node with `degree` random neighbour indices.
 * The CSR row pointers are fixed (each node gets exactly `degree` slots), so only
 * the neighbour indices are re-seeded; that is enough to move the gather offsets
 * around every pass while keeping the buffer sizes constant. */
static void seed_adjacency(p2_rng_t *rng, uint32_t *adj_idx, size_t total_edges,
                           size_t N) {
    for (size_t k = 0; k < total_edges; k++)
        adj_idx[k] = rng_index(rng, N);
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long nodes      = p2_get_i64(argc, argv, "--nodes", 1000000);
    long long degree     = p2_get_i64(argc, argv, "--degree", 6);
    /* The smoothing weight is a fraction in [0,1]; the phase2 arg helpers are
     * integer-only, so it is passed as integer-milli: --weight-milli 500 = 0.5. */
    long long weight_milli = p2_get_i64(argc, argv, "--weight-milli", 500);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    double w = (double)weight_milli / 1000.0;

    if (nodes < 16 || nodes > (1LL << 26)) {
        P2_LOG_ERR("nodes %lld out of range (16..2^26)", nodes);
        return 2;
    }
    if (degree < 0 || degree > 1024) {
        P2_LOG_ERR("degree %lld out of range (0..1024)", degree);
        return 2;
    }
    if (w < 0.0 || w > 1.0) {
        P2_LOG_ERR("weight %.3f out of range (0..1)", w);
        return 2;
    }
    size_t N   = (size_t)nodes;
    size_t D   = (size_t)degree;
    size_t total_edges = N * D;                      /* fixed degree per node */
    size_t val_bytes   = N * sizeof(double);         /* one value array */
    size_t buf_bytes   = 2 * val_bytes;              /* val + val_new (the dominant write) */
    size_t ptr_bytes   = (N + 1) * sizeof(uint32_t); /* CSR row pointers */
    size_t idx_bytes   = total_edges * sizeof(uint32_t); /* CSR neighbour indices */
    size_t total_bytes = buf_bytes + ptr_bytes + idx_bytes;
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
    p2_meta_kv_str(&m, "scheme", "Laplacian mesh smoothing (indirect neighbour gather then full node-array rewrite)");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "nodes", nodes);
    p2_meta_kv_i64(&m, "degree", degree);
    p2_meta_kv_i64(&m, "weight_milli", weight_milli);
    p2_meta_kv_u64(&m, "value_buffers_bytes", buf_bytes);
    p2_meta_kv_u64(&m, "adjacency_bytes", ptr_bytes + idx_bytes);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The two value arrays are the dominant buffers -> mmap + mlock them. One of
     * them (val_new) is fully rewritten every sweep, which is the workload's
     * signature write (a large, dense node-array rewrite that ping-pongs between
     * the two regions as the buffers swap). */
    double *val = (double *)mmap(NULL, val_bytes, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    double *val_new = (double *)mmap(NULL, val_bytes, PROT_READ | PROT_WRITE,
                                     MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (val == MAP_FAILED || val_new == MAP_FAILED) {
        P2_LOG_ERR("mmap(%zu x2) failed: %s", val_bytes, strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(val, val_bytes, MADV_NOHUGEPAGE);
    p2_madvise(val_new, val_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) { p2_mlock_soft(val, val_bytes); p2_mlock_soft(val_new, val_bytes); }

    /* CSR adjacency: fixed row pointers (each node has `degree` slots) plus the
     * neighbour-index array that the smoothing gather reads through. This is the
     * indirection table that turns the sequential node loop into scattered reads
     * of the value array. */
    uint32_t *adj_ptr = (uint32_t *)malloc((N + 1) * sizeof(uint32_t));
    uint32_t *adj_idx = (uint32_t *)malloc(total_edges ? total_edges * sizeof(uint32_t) : 1);
    if (!adj_ptr || !adj_idx) {
        free(adj_ptr); free(adj_idx);
        munmap(val, val_bytes); munmap(val_new, val_bytes);
        P2_LOG_ERR("malloc failed");
        p2_meta_kv_str(&m, "status", "alloc_failed"); p2_meta_close(&m); return 1;
    }
    /* Fixed CSR row pointers: node i owns slots [i*D, (i+1)*D). Only the neighbour
     * indices inside those slots are re-seeded each pass. */
    for (size_t i = 0; i <= N; i++) adj_ptr[i] = (uint32_t)(i * D);

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    /* Seed an initial random field and mesh so the first measured sweep has data
     * to smooth and an adjacency to gather through; every pass re-seeds both. */
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    for (size_t i = 0; i < N; i++) val[i] = p2_rng_unit(&rng);
    seed_adjacency(&rng, adj_idx, total_edges, N);
    double t_warmup_end = p2_monotonic();

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t sweeps = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* (1) Re-seed the adjacency: a fresh random mesh means the gather offsets
         * differ from the last sweep, so the irregular read keeps hitting new,
         * data-dependent positions rather than a fixed pattern. */
        seed_adjacency(&rng, adj_idx, total_edges, N);

        /* (2) Re-seed the value field: keeps the node array evolving (re-randomised
         * each pass) instead of converging to a flat field and going quiescent. */
        for (size_t i = 0; i < N; i++) val[i] = p2_rng_unit(&rng);

        /* (3) One smoothing sweep: for every node, gather its neighbour values
         * through the adjacency list and write the smoothed result into val_new.
         *   val_new[i] = (1-w)*val[i] + w * mean(neighbours of i)
         * The gather (val[adj_idx[k]]) is the irregular, indexed read; the store
         * into val_new is one dense sequential write front -- the unstructured
         * stencil. Nodes with no neighbours copy their own value unchanged. */
        for (size_t i = 0; i < N; i++) {
            uint32_t beg = adj_ptr[i];
            uint32_t end = adj_ptr[i + 1];
            uint32_t deg = end - beg;
            if (deg == 0) {
                val_new[i] = val[i];                 /* no neighbours: fixed point */
                continue;
            }
            double sum = 0.0;
            for (uint32_t k = beg; k < end; k++)
                sum += val[adj_idx[k]];              /* indirect neighbour gather */
            double mean = sum / (double)deg;
            val_new[i] = (1.0 - w) * val[i] + w * mean;
        }

        /* Swap the buffer roles: the array just written becomes the read source
         * for the next sweep. This pointer swap (not a copy) is what makes the
         * write target alternate between two fixed regions period-by-period. */
        double *tmp = val; val = val_new; val_new = tmp;   /* swap buffers */
        sweeps++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    volatile double sink = val[N / 2];               /* prevent dead-code elim */

    free(adj_ptr); free(adj_idx);
    munmap(val, val_bytes);
    munmap(val_new, val_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "sweeps", sweeps);
    p2_meta_kv_f64(&m, "mid_value", (double)sink);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "expected signature: indirect neighbour gather then a full dense node-array rewrite; WRITE footprint resembles a structured-grid stencil (D5) -- the irregular access is the read-side distinction, mostly invisible to a write-only signal");
    p2_meta_close(&m);
    return 0;
}
