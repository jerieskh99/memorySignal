/* kernel_maze_backtrack_v2  --  benign system-behaviour benchmark (NOT malware).
 *
 * ============================================================================
 * WHAT THIS IS
 * ============================================================================
 * Backtracking / Branch-and-Bound dwarf (Berkeley motif D11). A maze solver
 * that finds a path from the entrance to the exit of a LARGE grid maze by
 * depth-first search with explicit backtracking. The search marks a visited
 * grid and a parent (came-from) grid AS IT EXPLORES; on a dead end it pops the
 * stack and backs up. When the exit is reached, the single path is rebuilt by
 * walking the parent pointers back to the start.
 *
 * WHY IT IS A DISTINCT MEMORY SIGNATURE (visibility source (b): working state)
 * ----------------------------------------------------------------------------
 * The enumerate-style kernels are visible via their OUTPUT buffer. This one is
 * visible via a LARGE WORKING STATE mutated during the search. The solver
 * returns just ONE path, but to find it, it writes across a big visited[W*H]
 * and parent[W*H] region as the frontier snakes through the maze and backtracks
 * out of dead ends. Those two grids are the dominant buffers and the distinct
 * write tell -- the host write-signal sees the search itself, not just a result.
 *
 * CRITICAL -- the grid must be LARGE. A small, hot maze mutated at high
 * frequency stays CACHE-RESIDENT and is QUIET: the write-signal never leaves the
 * cache to reach the host. The signal only appears when the mutated state spans
 * MANY PAGES. So the default grid is 512 x 512 = 262144 cells and the maze is
 * carved to FORCE broad exploration, so a large fraction of those pages are
 * touched every solve.
 *
 * ============================================================================
 * PICTURE (top view -- one solve in progress, then the reconstructed path)
 * ============================================================================
 *   maze (# wall, . open)     DFS frontier snaking + backtracks
 *   S . . # . . . . .         S * * # . . . . .      * = on current DFS stack
 *   # # . # . # # # .         # # * # . # # # .      x = visited dead end
 *   . . . . . # . . .         . x x * * # . . .      (popped by backtracking)
 *   . # # # # # . # .         . # # # # # * # .
 *   . . . . . . . . E         . x x x x x * * E      frontier reaches exit E
 *                             reconstruct via parent[] pointers:
 *   final path (o):           S o o # . . . . .
 *                             # # o # . # # # .
 *                             . . o o o # . . .      one simple walk S -> E,
 *                             . # # # # # o # .      built by following came-from
 *                             . . . . . . o o E      links back from E to S.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *   1. CARVE a maze on the W x H grid with a randomized-DFS ("recursive
 *      backtracker"), driven by the harness RNG (p2_rng). This yields a PERFECT
 *      maze: exactly one simple path between any two open cells, with long
 *      winding corridors and many dead ends -- so solving must explore much of
 *      the grid. Carving itself uses an explicit stack (no C recursion).
 *   2. SOLVE by DFS backtracking from the entrance (0,0) toward the exit
 *      (W-1,H-1). Maintain visited[W*H] and parent[W*H]. On visiting a cell:
 *      mark visited, record which neighbour we came from in parent[]. Push open
 *      unvisited neighbours; on a dead end the explicit stack pops (backtrack).
 *   3. RECONSTRUCT: once the exit is popped/visited, walk parent[] from the exit
 *      back to the entrance to recover the one path; count its length.
 *
 * Each measure pass RE-SEEDS, RE-CARVES, and RE-SOLVES -- re-marking the whole
 * large grid every pass is the repeated visible write (the Jacobi measure-loop
 * idiom: the same big footprint revisited every iteration).
 *
 * NOTE: a 512x512 DFS would blow a recursive call stack, so BOTH the carve and
 * the solve use an EXPLICIT array stack, never deep C recursion.
 *
 * Pure memory + compute: no file I/O, no network, no persistence, no sandbox.
 * (The optional --dump-path writes ONE plain file for an external verifier and
 * runs OUTSIDE the measured loop; it is off by default.)
 * Signature family: KERNEL (visible). Dwarf: Backtrack / Branch-and-Bound.
 * See docs/SAFETY_MODEL.md.
 *
 * Phases: warmup (alloc + init) / measure (carve+solve passes) / cooldown.
 */
#include "../common/phase2_common.h"
#include "../common/phase2_portable.h"

static const char *TEST = "kernel_maze_backtrack_v2";

/* One grid cell. Kept to a small fixed size so 2 * W*H * sizeof(cell) is a
 * predictable footprint. "walls" is a 4-bit mask: which of the cell's four
 * sides are walls (bit N/E/S/W). "visited" and "parent" are the search state
 * written across the explored region -- the large-working-state write signal.
 *   walls  : bit0=N open? no -> encoded as wall-present bits (see DIR table)
 *   visited: 0 unseen, 1 seen during this solve
 *   parent : direction index (0..3) we entered this cell from, or PARENT_NONE
 */
typedef struct {
    uint8_t walls;    /* wall bitmask: bit d set => wall on side d (blocked)   */
    uint8_t visited;  /* marked as the DFS frontier passes through             */
    uint8_t parent;   /* came-from direction (0..3), or PARENT_NONE at start   */
    uint8_t _pad;     /* keep sizeof(Cell)==4 (stable, page-predictable)       */
} Cell;

#define PARENT_NONE 0xFF

/* Four cardinal directions. dir d: (drow, dcol); OPP[d] is the reverse dir.
 * Walls are shared between adjacent cells, so carving a passage clears the wall
 * bit on BOTH the current cell (side d) and the neighbour (side OPP[d]). */
static const int DR[4]  = { -1,  0,  1,  0 };   /* N, E, S, W (row delta) */
static const int DC[4]  = {  0,  1,  0, -1 };   /* N, E, S, W (col delta) */
static const int OPP[4] = {  2,  3,  0,  1 };   /* opposite direction index */

static void usage(const char *p) {
    fprintf(stderr,
"Usage: %s [options]   (benign DFS-backtracking maze solver; backtrack kernel)\n"
"  --width W             Maze width in cells (default 512)\n"
"  --height H            Maze height in cells (default 512)\n"
"  --duration SEC        Measurement duration (default 60)\n"
"  --warmup SEC          Warm-up duration (default 2)\n"
"  --seed N              PRNG seed for maze carving (default 42)\n"
"  --max-mb N            Hard cap on total buffer bytes (default 8192)\n"
"  --no-mlock            Skip mlock() entirely\n"
"  --output-dir PATH     Where to write metadata JSON\n"
"  --cpu-affinity N      Pin to CPU N\n"
"  --phase-markers       Emit phase markers to stderr\n"
"  --dump-path PATH      (verifier only, OUTSIDE the measured loop) write the\n"
"                        maze walls + reconstructed path to a plain file, then\n"
"                        exit. For the external non-circular path verifier.\n"
"  --dry-run             Validate args and exit\n"
"  --help                Show this help\n", p);
}

/* uniform integer in [0, n) from the xoshiro stream (rejection-free enough for
 * neighbour shuffling; slight bias is irrelevant to the write pattern). */
static inline uint64_t p2_rng_below(p2_rng_t *r, uint64_t n) {
    return p2_rng_next(r) % n;
}

/* ---------------------------------------------------------------------------
 * CARVE: randomized-DFS ("recursive backtracker") using an EXPLICIT stack.
 * Start with every wall present, then walk: from the current cell pick a random
 * unvisited neighbour, knock down the wall between them, push and advance; on no
 * unvisited neighbour, pop (backtrack). Produces a perfect maze (one simple path
 * between any two cells). The carve writes every cell's walls exactly once.
 * --------------------------------------------------------------------------- */
static void carve_maze(Cell *grid, uint8_t *cstack_seen, int *stack,
                       size_t W, size_t H, p2_rng_t *rng) {
    size_t cells = W * H;
    /* Reset: all walls present, nothing carve-visited, no parent, not solved. */
    for (size_t i = 0; i < cells; i++) {
        grid[i].walls   = 0x0F;          /* all four sides walled (bits 0..3) */
        grid[i].visited = 0;
        grid[i].parent  = PARENT_NONE;
        grid[i]._pad    = 0;
        cstack_seen[i]  = 0;
    }

    int top = 0;
    stack[top++] = 0;                    /* start carving at cell (0,0) = idx 0 */
    cstack_seen[0] = 1;

    while (top > 0) {
        int cur = stack[top - 1];
        int cr = cur / (int)W;
        int cc = cur % (int)W;

        /* Collect carve-unvisited neighbours. */
        int cand[4];
        int ncand = 0;
        for (int d = 0; d < 4; d++) {
            int nr = cr + DR[d];
            int nc = cc + DC[d];
            if (nr < 0 || nr >= (int)H || nc < 0 || nc >= (int)W) continue;
            int ni = nr * (int)W + nc;
            if (!cstack_seen[ni]) cand[ncand++] = d;
        }

        if (ncand == 0) { top--; continue; }             /* dead end: backtrack */

        int d  = cand[p2_rng_below(rng, (uint64_t)ncand)];
        int nr = cr + DR[d];
        int nc = cc + DC[d];
        int ni = nr * (int)W + nc;

        /* Knock down the shared wall between cur and ni (both sides). */
        grid[cur].walls = (uint8_t)(grid[cur].walls & ~(1u << d));
        grid[ni].walls  = (uint8_t)(grid[ni].walls  & ~(1u << OPP[d]));

        cstack_seen[ni] = 1;
        stack[top++] = ni;                               /* advance into neighbour */
    }
}

/* ---------------------------------------------------------------------------
 * SOLVE: DFS backtracking from start (idx 0) to exit (idx W*H-1) using the same
 * EXPLICIT stack. Marks grid[].visited and records grid[].parent (the direction
 * we entered each cell from). Returns cells_visited via *out_visited, and 1 if
 * the exit was reached, else 0. These visited/parent writes are the visible
 * large-working-state signal.
 * --------------------------------------------------------------------------- */
static int solve_maze(Cell *grid, int *stack, size_t W, size_t H,
                      uint64_t *out_visited) {
    int start = 0;
    int exit  = (int)(W * H) - 1;
    uint64_t visited_count = 0;

    int top = 0;
    stack[top++] = start;
    grid[start].visited = 1;
    grid[start].parent  = PARENT_NONE;
    visited_count++;

    int found = 0;
    while (top > 0) {
        int cur = stack[top - 1];
        if (cur == exit) { found = 1; break; }
        int cr = cur / (int)W;
        int cc = cur % (int)W;

        /* Find the first open, unvisited neighbour to descend into. */
        int advanced = 0;
        for (int d = 0; d < 4; d++) {
            if (grid[cur].walls & (1u << d)) continue;    /* wall on this side */
            int nr = cr + DR[d];
            int nc = cc + DC[d];
            if (nr < 0 || nr >= (int)H || nc < 0 || nc >= (int)W) continue;
            int ni = nr * (int)W + nc;
            if (grid[ni].visited) continue;

            grid[ni].visited = 1;
            grid[ni].parent  = (uint8_t)OPP[d];           /* came from side OPP[d] */
            stack[top++] = ni;
            visited_count++;
            advanced = 1;
            break;
        }
        if (!advanced) top--;                             /* dead end: backtrack */
    }

    *out_visited = visited_count;
    return found;
}

/* Reconstruct the path by walking parent[] from the exit back to the start.
 * Returns the path length (number of cells, start..exit inclusive), or 0 if the
 * exit was never reached / the chain is broken. Read-only over the grid. */
static uint64_t reconstruct_len(const Cell *grid, size_t W, size_t H, int found) {
    if (!found) return 0;
    int start = 0;
    int exit  = (int)(W * H) - 1;
    uint64_t len = 0;
    size_t guard = W * H + 1;                             /* hard bound vs cycles */
    int cur = exit;
    while (guard--) {
        len++;
        if (cur == start) return len;
        uint8_t pd = grid[cur].parent;
        if (pd == PARENT_NONE) return 0;                  /* broken chain */
        int cr = cur / (int)W;
        int cc = cur % (int)W;
        int pr = cr + DR[pd];
        int pc = cc + DC[pd];
        if (pr < 0 || pr >= (int)H || pc < 0 || pc >= (int)W) return 0;
        cur = pr * (int)W + pc;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * --dump-path writer (VERIFIER-ONLY, OUTSIDE the measured loop, off by default).
 * Writes a plain text file the external verifier reads: the maze walls (so it
 * can check open vs wall) plus the reconstructed path as (row,col) cells. This
 * is a benign one-shot file write for non-circular verification, NOT part of
 * the measured workload. Returns 0 on success, -1 on failure.
 * --------------------------------------------------------------------------- */
static int dump_path(const char *path, const Cell *grid, size_t W, size_t H,
                     int found) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        P2_LOG_ERR("dump-path fopen(%s) failed: %s", path, strerror(errno));
        return -1;
    }
    /* Header: dimensions and whether a path was found. */
    fprintf(fp, "# maze dump for path verifier\n");
    fprintf(fp, "width %zu\n", W);
    fprintf(fp, "height %zu\n", H);
    fprintf(fp, "path_found %d\n", found ? 1 : 0);

    /* Walls grid: one hex nibble per cell (bit d set => wall on side d). Row per
     * line. Lets the verifier reconstruct open/blocked and adjacency itself. */
    fprintf(fp, "walls\n");
    for (size_t r = 0; r < H; r++) {
        for (size_t c = 0; c < W; c++) {
            fprintf(fp, "%x", grid[r * W + c].walls & 0x0F);
        }
        fputc('\n', fp);
    }

    /* Reconstructed path: entrance..exit as (row,col), one per line, in order.
     * We walk parent[] from the exit then reverse via a temporary buffer. */
    fprintf(fp, "path\n");
    if (found) {
        size_t cap = W * H;
        int *chain = (int *)malloc(cap * sizeof(int));
        if (!chain) { fclose(fp); P2_LOG_ERR("dump-path malloc failed"); return -1; }
        size_t n = 0;
        int start = 0;
        int exit  = (int)(W * H) - 1;
        int cur = exit;
        size_t guard = cap + 1;
        int ok = 0;
        while (guard-- && n < cap) {
            chain[n++] = cur;
            if (cur == start) { ok = 1; break; }
            uint8_t pd = grid[cur].parent;
            if (pd == PARENT_NONE) break;
            int cr = cur / (int)W;
            int cc = cur % (int)W;
            cur = (cr + DR[pd]) * (int)W + (cc + DC[pd]);
        }
        if (ok) {
            for (size_t i = n; i-- > 0; ) {              /* reverse: start -> exit */
                fprintf(fp, "%d %d\n", chain[i] / (int)W, chain[i] % (int)W);
            }
        }
        free(chain);
    }
    if (fclose(fp) != 0) {
        P2_LOG_ERR("dump-path fclose(%s) failed: %s", path, strerror(errno));
        return -1;
    }
    P2_LOG_INFO("dump-path written: %s", path);
    return 0;
}

int main(int argc, char **argv) {
    if (p2_flag_present(argc, argv, "--help")) { usage(argv[0]); return 0; }
    long long width      = p2_get_i64(argc, argv, "--width", 512);
    long long height     = p2_get_i64(argc, argv, "--height", 512);
    long long duration_s = p2_get_i64(argc, argv, "--duration", 60);
    long long warmup_s   = p2_get_i64(argc, argv, "--warmup", 2);
    long long max_mb     = p2_get_i64(argc, argv, "--max-mb", 8192);
    long long cpu        = p2_get_i64(argc, argv, "--cpu-affinity", -1);
    uint64_t  seed       = p2_get_u64(argc, argv, "--seed", 42);
    int       no_mlock   = p2_flag_present(argc, argv, "--no-mlock");
    int       dry_run    = p2_flag_present(argc, argv, "--dry-run");
    const char *outdir   = p2_get_str(argc, argv, "--output-dir", NULL);
    const char *dumppath = p2_get_str(argc, argv, "--dump-path", NULL);

    if (width < 4 || width > 65536) { P2_LOG_ERR("width %lld out of range (4..65536)", width); return 2; }
    if (height < 4 || height > 65536) { P2_LOG_ERR("height %lld out of range (4..65536)", height); return 2; }
    size_t W = (size_t)width;
    size_t H = (size_t)height;
    size_t cells = W * H;

    /* Dominant buffers: the grid (visited+parent+walls) plus the DFS explicit
     * stack and the carve seen-array. The grid is 2x visible state (visited and
     * parent) folded into one Cell array; clamp so it respects --max-mb. */
    size_t grid_bytes  = cells * sizeof(Cell);
    size_t stack_bytes = cells * sizeof(int);
    size_t seen_bytes  = cells * sizeof(uint8_t);
    size_t total_bytes = grid_bytes + stack_bytes + seen_bytes;
    if (total_bytes > (size_t)max_mb * 1024ULL * 1024ULL) {
        P2_LOG_ERR("total bytes %zu exceed --max-mb %lld", total_bytes, max_mb);
        return 2;
    }

    p2_meta_t m;
    p2_meta_open(&m, outdir, TEST);
    p2_meta_kv_str(&m, "test_name", TEST);
    p2_meta_kv_str(&m, "language", "C");
    p2_meta_kv_str(&m, "phase2_version", PHASE2_VERSION);
    p2_meta_kv_str(&m, "behavior_family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "dwarf", "D11 Backtrack/Branch-and-Bound");
    p2_meta_kv_str(&m, "family", "KERNEL (visible)");
    p2_meta_kv_str(&m, "scheme", "randomized-DFS carved maze, solved by DFS backtracking with visited+parent, path reconstructed");
    p2_meta_kv_str(&m, "safety", "benign-benchmark; no network/persistence/sandbox; compute-only");
    p2_meta_kv_i64(&m, "width", width);
    p2_meta_kv_i64(&m, "height", height);
    p2_meta_kv_u64(&m, "cells", cells);
    p2_meta_kv_u64(&m, "total_bytes", total_bytes);
    p2_meta_kv_i64(&m, "duration_s", duration_s);
    p2_meta_kv_i64(&m, "warmup_s", warmup_s);
    p2_meta_kv_u64(&m, "seed", seed);
    p2_meta_kv_i64(&m, "cpu_pin", cpu);
    char tstart[32]; p2_iso_timestamp(tstart, sizeof(tstart));
    p2_meta_kv_str(&m, "start_time", tstart);

    if (dry_run) { p2_meta_kv_str(&m, "status", "dry_run"); p2_meta_close(&m); return 0; }
    if (cpu >= 0) p2_pin_cpu((int)cpu);

    /* The grid is the dominant buffer (visited+parent marked across the search)
     * -> mmap + mlock it. The explicit DFS stack and carve seen-array are also
     * mmap'd so the whole working state is pinned and page-predictable. */
    Cell *grid = (Cell *)mmap(NULL, grid_bytes, PROT_READ | PROT_WRITE,
                              MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    int *stack = (int *)mmap(NULL, stack_bytes, PROT_READ | PROT_WRITE,
                             MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    uint8_t *seen = (uint8_t *)mmap(NULL, seen_bytes, PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (grid == MAP_FAILED || stack == MAP_FAILED || seen == MAP_FAILED) {
        P2_LOG_ERR("mmap failed: %s", strerror(errno));
        p2_meta_kv_str(&m, "status", "mmap_failed"); p2_meta_close(&m); return 1;
    }
    p2_madvise(grid, grid_bytes, MADV_NOHUGEPAGE);
    p2_madvise(stack, stack_bytes, MADV_NOHUGEPAGE);
    p2_madvise(seen, seen_bytes, MADV_NOHUGEPAGE);
    if (!no_mlock) {
        p2_mlock_soft(grid, grid_bytes);
        p2_mlock_soft(stack, stack_bytes);
        p2_mlock_soft(seen, seen_bytes);
    }

    p2_phase(TEST, "warmup");
    double t0 = p2_monotonic();
    p2_rng_t rng; p2_rng_seed(&rng, seed);
    /* Prime the buffers with one carve so pages are resident before timing. */
    carve_maze(grid, seen, stack, W, H, &rng);
    double t_warmup_end = p2_monotonic();

    /* --dump-path branch: one carve+solve, write the file, exit. This runs
     * BEFORE (never inside) the measured loop -- verifier support only. */
    if (dumppath) {
        p2_rng_t drng; p2_rng_seed(&drng, seed);
        carve_maze(grid, seen, stack, W, H, &drng);
        uint64_t vc = 0;
        int fnd = solve_maze(grid, stack, W, H, &vc);
        uint64_t plen = reconstruct_len(grid, W, H, fnd);
        int rc = dump_path(dumppath, grid, W, H, fnd);
        p2_meta_kv_u64(&m, "cells_visited", vc);
        p2_meta_kv_u64(&m, "path_length", plen);
        p2_meta_kv_i64(&m, "path_found", fnd ? 1 : 0);
        p2_meta_kv_str(&m, "status", rc == 0 ? "dump_ok" : "dump_failed");
        p2_meta_close(&m);
        munmap(grid, grid_bytes); munmap(stack, stack_bytes); munmap(seen, seen_bytes);
        return rc == 0 ? 0 : 1;
    }

    p2_phase(TEST, "measure");
    double t_meas_start = t_warmup_end;
    uint64_t passes = 0;
    uint64_t last_visited = 0, last_pathlen = 0;
    int last_found = 0;
    while ((p2_monotonic() - t_meas_start) < (double)duration_s) {
        /* One pass: re-seed + re-carve (rewrites walls) + re-solve (re-marks the
         * visited+parent grid across the explored region). Re-marking the large
         * grid every pass is the repeated visible large-working-state write. */
        carve_maze(grid, seen, stack, W, H, &rng);
        uint64_t vc = 0;
        int fnd = solve_maze(grid, stack, W, H, &vc);
        uint64_t plen = reconstruct_len(grid, W, H, fnd);
        last_visited = vc; last_pathlen = plen; last_found = fnd;
        passes++;
    }
    double t_meas_end = p2_monotonic();

    p2_phase(TEST, "cooldown");
    p2_sleep_ns(500ULL * 1000ULL * 1000ULL);
    double t_cool_end = p2_monotonic();

    /* Prevent dead-code elimination: touch a live cell of the last solve. */
    volatile uint8_t sink = grid[cells - 1].visited;
    (void)sink;

    munmap(grid, grid_bytes);
    munmap(stack, stack_bytes);
    munmap(seen, seen_bytes);

    p2_meta_kv_f64(&m, "warmup_t0_s", t0);
    p2_meta_kv_f64(&m, "warmup_end_s", t_warmup_end);
    p2_meta_kv_f64(&m, "measure_end_s", t_meas_end);
    p2_meta_kv_f64(&m, "cooldown_end_s", t_cool_end);
    p2_meta_kv_u64(&m, "passes", passes);
    p2_meta_kv_u64(&m, "cells_visited", last_visited);
    p2_meta_kv_u64(&m, "path_length", last_pathlen);
    p2_meta_kv_i64(&m, "path_found", last_found ? 1 : 0);
    p2_meta_kv_str(&m, "status", "ok");
    char tend[32]; p2_iso_timestamp(tend, sizeof(tend));
    p2_meta_kv_str(&m, "end_time", tend);
    p2_meta_kv_str(&m, "known_limitations",
                   "visibility comes from the LARGE visited+parent grid marked across the search; a small hot grid would stay cache-resident and quiet");
    p2_meta_close(&m);
    return 0;
}
