#!/usr/bin/env python3
"""Phase 2 data: plain-language rewrite of each workload's algorithm.

Consumed by make_workload_algorithms.py, which renders the rewrite ABOVE the
verbatim source (in a collapsible). This file is the ONLY authored prose; the
verbatim extraction stays the source of truth, so the two can be compared.

Each entry:
  plain  -- what the algorithm does, in plain language (professor voice).
  signal -- one line: what its memory WRITE-signal looks like and why it is in
            the campaign (the thesis-relevant point).

Inline [[term]] marks a glossary word (defined in GLOSSARY); [[term|inline def]]
defines one on the spot. The generator turns both into hover definitions.
"""

GLOSSARY = {
    "mmap": "A request to the operating system for a block of memory, or to map a file into memory. This is the moment a program says 'give me N bytes.'",
    "working set": "The region of memory a workload actively touches. Larger working set = more distinct pages changing = a louder memory signal.",
    "stride": "The gap between successive memory addresses a loop touches. A 4096-byte stride touches one byte per page, hitting many pages with little data.",
    "page": "The 4096-byte unit the memory system tracks. The signal is measured per page: did anything in this 4 KB change since the last snapshot?",
    "dirty page": "A page that has been written to since the last snapshot. Counting dirty pages is essentially what the memory signal measures.",
    "anonymous memory": "Memory backed only by RAM, not by any file. It cannot be reclaimed by writing it elsewhere, so it is the kind that can exhaust the guest.",
    "in-place": "Overwriting a buffer's own cells rather than writing to a fresh one. Keeps the memory footprint small but still produces writes.",
    "double-buffer": "Reading from one array while writing the next state into a second, then swapping. Doubles the footprint but avoids overwriting data still in use.",
    "wavefront": "A diagonal band of cells being filled in a table, which sweeps across as the computation proceeds -- a moving front of writes.",
    "scatter": "Writing to many non-adjacent memory locations chosen by data, rather than in order. Produces a diffuse, spread-out write pattern.",
    "gather": "Reading from many non-adjacent locations chosen by data. Gathers are reads, so they are invisible to a write-based memory signal.",
    "sparse matrix": "A matrix that is mostly zeros, stored compactly by listing only the non-zero entries. Reading it is a gather; the signal depends on whether the OUTPUT is large.",
    "CSR": "Compressed Sparse Row -- the standard compact layout for a sparse matrix: three arrays holding values, their columns, and where each row starts.",
    "butterfly": "The cross-combining write pattern of an FFT: each stage pairs up elements across a stride that halves (or doubles) each pass.",
    "bit-reversal": "The scrambled output order of an in-place FFT, where element i ends up at the position given by reversing i's bits -- a scattered write.",
    "trellis": "A grid of probabilities filled column by column in sequence models (HMMs), each column depending on the previous -- a left-to-right write front.",
    "frontier": "The set of partial solutions a search is still considering, held in a queue or heap and churned constantly as the search pushes and pops.",
    "backtracking": "Trying a choice, and if it fails, undoing it and trying another. The undo (restore) writes are a distinctive part of the signal.",
    "branch-and-bound": "A search that prunes whole regions once it proves they cannot beat the best solution found so far. Its frontier heap is the visible write.",
    "stencil": "Updating each grid cell from its immediate neighbors, repeatedly. The canonical regular, periodic write pattern of physical simulations.",
    "PRNG": "Pseudo-random number generator -- a seeded formula producing repeatable 'random' bytes, used so a workload's behavior is deterministic across runs.",
    "entropy": "How unpredictable data is. High-entropy (random-looking) bytes do not compress, which matters for how large the stored snapshots become.",
    "tiling": "Splitting a big matrix operation into small blocks that fit in cache, processing one block at a time. Also called blocking.",
    "DFA": "Deterministic Finite Automaton -- a state machine with a lookup table: current state + next character picks the next state. Matching is just table reads.",
    "cell list": "A spatial grid that buckets particles by location so each particle only checks nearby ones. Rebuilt each step as particles move.",
    "quadtree": "A tree that recursively splits 2D space into quarters, letting a simulation approximate far-away groups of particles as one -- O(n log n) instead of O(n^2).",
    "leapfrog": "A time-stepping scheme where two coupled fields are updated in alternation, each half a step ahead of the other.",
    "fill-in": "New non-zero entries that appear in a sparse matrix during factorization, in positions that were zero before -- extra writes beyond the original structure.",
    "reduction": "Combining many values into one (a sum, a max, a count). The result is tiny, so a pure reduction is nearly invisible to the write-signal.",
}

REWRITES = {

# ---------------- CPU ----------------
"cpu_hash_loop_v2": {
 "plain": "Repeatedly hashes a small block of data in a tight loop. All the work happens in CPU registers and a tiny buffer that stays in cache -- almost nothing is written back to main memory.",
 "signal": "The quiet baseline: heavy computation, near-zero write-signal. Establishes what 'no real memory activity' looks like."},
"cpu_matrix_mult_v2": {
 "plain": "A plain triple-loop multiply of two square matrices, writing the result matrix C. Small-scale, naive version of [[GEMM|the standard dense matrix-multiply operation]] -- no [[tiling]].",
 "signal": "A small, regular output rewrite. A gentle control point between the silent CPU workloads and the loud memory ones."},
"cpu_branch_random_v2": {
 "plain": "Hammers the CPU's branch predictor by following a data-dependent path through a small read-only table. The table is only read; the loop writes essentially nothing.",
 "signal": "Near-idle by design -- stresses control flow, not memory, so it stays in the quiet baseline group."},

# ---------------- CACHE ----------------
"cache_hot_loop_v2": {
 "plain": "Loops over a buffer small enough to live entirely in CPU cache, touching it repeatedly. Because the data never leaves cache, main memory barely sees the activity.",
 "signal": "Loud CPU work, quiet memory signal -- tests that heavy activity in a small [[working set]] stays nearly invisible."},
"cache_cold_scan_v2": {
 "plain": "Sweeps a buffer far larger than cache, in read-modify-write mode so every page is both read and rewritten. Each pass evicts the last, so the CPU is constantly going to main memory.",
 "signal": "The loud counterpart to the hot loop: a broad, high write-signal across the whole large [[working set]]."},
"cache_stride_sweep_v2": {
 "plain": "Walks a large buffer jumping by a fixed [[stride]], writing one spot per jump. By tuning the stride against the [[page]] size, it controls how many distinct pages get touched for a given amount of data written.",
 "signal": "How broad the dirtied-page footprint is depends on stride versus page size -- it maps that relationship directly."},

# ---------------- MEM ----------------
"mem_workingset_sweep_v2": {
 "plain": "Sweeps sequentially across a buffer of a chosen size, writing one byte per [[page]] each pass. Varying the buffer size varies how much memory is 'active' while everything else stays fixed.",
 "signal": "The active-page fraction should grow directly with [[working set]] size -- the most direct test of the core thesis metric."},
"mem_writemag_sweep_v2": {
 "plain": "Fixes the buffer size but varies how many bytes it writes into each [[page]] (1, 64, 1024, or 4096). Uses fresh random bytes each pass so the writes genuinely change content.",
 "signal": "Tests how the per-page change metric responds to write magnitude -- it saturates once enough of a page changes, and this maps that curve."},
"mem_rmw_intensity_v2": {
 "plain": "Read-modify-writes a large buffer at a fixed [[stride]]: read each spot, tweak it, write it back. A deliberate contrast to pure writing, to see if reading-then-writing looks different.",
 "signal": "A pre-registered null test -- read-modify-write may be indistinguishable from pure writes at the signal level, and this checks that."},
"mem_pagefault_density_v2": {
 "plain": "Touches fresh pages to force the operating system to allocate them ([[page|first-touch]] faults), controlling how densely faults occur over time.",
 "signal": "Isolates the page-fault contribution to the signal. Ends as soon as all pages are touched, so it can finish well before its time limit."},
"mem_mmap_traversal_v2": {
 "plain": "Maps a file into memory and writes through it, so the operating system streams the changes back to the file on disk in the background.",
 "signal": "The one memory workload whose pages are [[file-backed|backed by a file, so they can be dropped and reloaded]] rather than pure RAM -- it tests writeback behavior, and its footprint is safely evictable."},
"mem_random_write_pages_v2": {
 "plain": "Writes a few bytes into pages chosen at random across a large buffer, spreading the activity unpredictably instead of sweeping in order.",
 "signal": "A broad, diffuse, high write-signal -- the 'scattered writes everywhere' shape, contrasted with the orderly sweeps."},
"mem_stride_sweep_large_v2": {
 "plain": "Like the stride sweep, but over a much larger buffer, deliberately stressing the CPU's address-translation cache and hardware prefetcher.",
 "signal": "Companion to cache_stride_sweep at a bigger scale -- separates translation/prefetch pressure from raw footprint."},

# ---------------- THREAD ----------------
"thread_lock_contention_v2": {
 "plain": "Several threads fight over a single shared lock, incrementing one shared counter. Almost all the cost is the operating system arbitrating the contention, not memory traffic.",
 "signal": "Tiny dirtied footprint (just the counter) despite heavy CPU/scheduler activity -- concurrency overhead is nearly invisible to the memory signal."},
"thread_producer_consumer_v2": {
 "plain": "One set of threads pushes items into a small ring buffer while another set pops them, coordinated by condition-variable wakeups.",
 "signal": "The wakeups dominate the cost; only the small ring changes -- another near-silent concurrency shape."},
"thread_parallel_alloc_v2": {
 "plain": "Many threads allocate and free blocks of varying sizes at once, churning the memory allocator's internal bookkeeping.",
 "signal": "The visible writes are allocator metadata, not application data -- footprint varies with the size mix."},

# ---------------- IO ----------------
"io_read_cache_hit_v2": {
 "plain": "Reads the same file repeatedly after it is already cached in memory, so every read is served from cache without touching the disk.",
 "signal": "Cache-hot reads dirty almost nothing -- expected near-idle. (This is exactly the workload the tmpfs bug corrupted: on a RAM disk its 'reads' looked like memory writes.)"},
"io_direct_write_like_v2": {
 "plain": "Writes to a file in a way that bypasses the normal page cache, pushing data toward the disk directly in fixed-size blocks.",
 "signal": "Tests the disk-write shape of the signal, deliberately different from an in-RAM memory write -- the IO-vs-MEM distinction the campaign exists to measure."},

# ---------------- MIXED ----------------
"mixed_mem_io_v2": {
 "plain": "Runs memory writes and file I/O at the same time, under scheduler pressure, to see whether the combination looks like the sum of its parts.",
 "signal": "A blended signal -- tests whether concurrent memory + I/O separate cleanly or interfere."},
"mixed_cpu_mem_v2": {
 "plain": "Interleaves pure computation with memory writes, mixing a quiet behavior with a loud one in a single workload.",
 "signal": "Tests whether a CPU+memory blend is distinguishable from either alone."},
"mixed_cpu_io_v2": {
 "plain": "Interleaves computation with file I/O, combining a silent behavior with a disk-facing one.",
 "signal": "Tests a CPU+I/O blend -- computation should stay quiet while the I/O carries the signal."},

# ---------------- APP ----------------
"app_compress_gzip_v2": {
 "plain": "Generates a deterministic random input file, then gzip-compresses it. Random bytes are [[entropy|incompressible]] by design, so the compression work stays steady and CPU-bound with I/O.",
 "signal": "A realistic sustained CPU+I/O mix with no strong rhythm -- application-shaped rather than synthetic."},
"app_decompress_gzip_v2": {
 "plain": "Decompresses a gzip stream, producing a large output. The CPU-versus-I/O balance shifts with the compression level.",
 "signal": "The decompression counterpart -- writes a big output, so it is heavier than the compress side."},
"app_json_parse_v2": {
 "plain": "Parses a large JSON document with Python's C-backed parser, building the in-memory object tree.",
 "signal": "Allocation-heavy in a Python-specific rhythm -- a realistic parsing workload rather than a microbenchmark."},
"app_sqlite_analytical_v2": {
 "plain": "Runs analytical SQL queries (aggregations, top-N, range scans) over a generated SQLite database, materializing temporary tables as it goes.",
 "signal": "Read-heavy with small writes from the temp tables -- an OLAP-style database shape."},
"app_sqlite_oltp_v2": {
 "plain": "Runs many small transactional inserts/updates/reads against SQLite, in a realistic transaction mix, with the write-ahead log checkpointing periodically.",
 "signal": "Frequent small writes plus periodic checkpoint bursts -- a transactional (OLTP) database shape."},
"app_hashtable_intensive_v2": {
 "plain": "Hammers a large in-memory hash table with a chosen number of slots, using linear probing so lookups stay cache-local.",
 "signal": "The probing pattern shapes the signal; a large table means real memory activity across it."},

# ---------------- SANDBOX ----------------
"sandbox_ransom_seq": {
 "plain": "Simulates ransomware behavior: walks a directory and rewrites each file in sequence with high-[[entropy|random, encryption-like]] content. Behavioral mimicry only -- no real encryption, persistence, or evasion.",
 "signal": "A steady, high-entropy full-rewrite sweep across files -- the canonical 'bulk encryptor' shape for the security-detection angle."},
"sandbox_ransom_batched": {
 "plain": "The same file-rewriting behavior, but buffers many files in memory and flushes them in batches rather than one at a time.",
 "signal": "Bursty rather than steady -- tests whether batching changes the encryptor signature. (Its all-in-RAM buffering is why file count had to be capped.)"},
"sandbox_ransom_slowburn": {
 "plain": "Rewrites files slowly, one every few seconds, imitating a low-and-slow attacker trying not to spike.",
 "signal": "The cadence dominates the signal -- tests detection of a deliberately paced, stretched-out rewrite."},
"sandbox_ransom_selective": {
 "plain": "Rewrites only files matching certain extensions, imitating an attacker that targets specific document types.",
 "signal": "A sparser, filtered version of the sweep -- fewer files touched, more selective footprint."},
"sandbox_scanner_metadata": {
 "plain": "Walks a large tree of many small files reading their metadata (stat-style), touching the directory structure heavily but writing little.",
 "signal": "Stat-heavy, write-light -- the 'enumeration/reconnaissance' shape, mostly reads."},
"sandbox_stealth_microwrite": {
 "plain": "Writes tiny amounts to files at a slow, fixed interval, trying to stay under a detection threshold by keeping each write small.",
 "signal": "Low-rate, high-intensity trickle -- tests whether small paced writes can hide in the noise floor."},
"sandbox_stealth_paced": {
 "plain": "Similar trickle behavior at a slightly different pacing, spacing writes out over time to blend into normal activity.",
 "signal": "A stealth-cadence variant -- probes the timing dimension of hiding."},
"sandbox_stealth_scattered": {
 "plain": "Spreads small writes across many files at intervals, scattering the activity spatially as well as pacing it in time.",
 "signal": "Scattered + paced -- tests hiding by spatial spread on top of slow timing."},

# ---------------- KERNEL: D1 dense linear algebra ----------------
"kernel_gemm_v2": {
 "plain": "Multiplies two dense square matrices into an output C, in cache-friendly [[tiling|small blocks]]. Every pass re-seeds the inputs and rewrites all of C.",
 "signal": "A static, full-footprint rewrite -- the whole output changes every pass. The reference against which LU's and QR's moving fronts are compared."},
"kernel_lu_v2": {
 "plain": "Factors a matrix into lower/upper triangular pieces [[in-place]], eliminating one column at a time so the still-active region shrinks as it goes.",
 "signal": "A shrinking trailing-submatrix front -- writes concentrate in an ever-smaller corner. The classic direct-solve pattern (LAPACK's core)."},
"kernel_qr_v2": {
 "plain": "Orthogonalizes a matrix's columns one by one (Gram-Schmidt/QR), so the finished region grows as each new column is processed.",
 "signal": "A growing orthogonalized-column front -- the mirror image of LU's shrinking one. Used in least-squares and eigensolvers."},
"kernel_attention_v2": {
 "plain": "Computes transformer attention: multiply queries by keys, softmax each row, multiply by values. Two matrix multiplies with a row-normalization between.",
 "signal": "Write-signature close to [[GEMM|dense matmul]] plus a row softmax -- the per-token core of every large language model."},
"kernel_conv_v2": {
 "plain": "Slides a small filter window across an image and multiply-accumulates at each position, writing a large output feature map (a CNN layer).",
 "signal": "An overlapping-window rewrite of the output map -- the memory shape of convolutional vision models."},

# ---------------- KERNEL: D2 sparse linear algebra ----------------
"kernel_spmv_v2": {
 "plain": "Multiplies a big [[sparse matrix]] (stored [[CSR|compactly]]) by a vector. The matrix is read by [[gather|scattered reads]]; the only output is a small vector.",
 "signal": "The quiet control: enormous read work, tiny write -- the classic 'important but invisible' case. The near-idle baseline for sparse work."},
"kernel_spmm_v2": {
 "plain": "Multiplies a sparse matrix by a DENSE matrix, producing a large dense output that is rewritten each pass -- the aggregation step of graph neural networks.",
 "signal": "Visible: the big dense output carries the write-signal, unlike quiet SpMV which writes only a vector."},
"kernel_sparse_cholesky_v2": {
 "plain": "Factors a banded [[sparse matrix]] symmetrically, progressively writing [[fill-in|new non-zeros]] within the band.",
 "signal": "Visible: the factor fills in progressively across the band -- direct solvers for finite-element and circuit problems."},
"kernel_spgemm_v2": {
 "plain": "Multiplies two sparse matrices to produce a NEW sparse matrix, whose structure (with [[fill-in]]) is not known in advance and must be built up.",
 "signal": "Visible: writes an entirely new sparse matrix -- the setup step of algebraic multigrid and triangle counting."},
"kernel_sddmm_v2": {
 "plain": "Multiplies two dense matrices but only computes the entries at positions marked by a sparse mask, [[scatter|scattering]] results into those spots.",
 "signal": "Visible: scattered writes at the mask positions only -- used in graph-attention networks and recommenders."},
"kernel_moe_dispatch_v2": {
 "plain": "Routes each input token to a chosen expert by [[scatter|permuting tokens into per-expert buffers]], then gathers the results back -- the Mixture-of-Experts dispatch used in modern LLMs.",
 "signal": "Visible: a token-permutation scatter into expert buffers -- the routing write of >60% of recent large models."},

# ---------------- KERNEL: D3 spectral ----------------
"kernel_fft_v2": {
 "plain": "Transforms a signal to the frequency domain [[in-place]] using the radix-2 FFT: repeated [[butterfly]] passes, ending with a [[bit-reversal|scrambled-order]] output.",
 "signal": "Visible++: a 1D butterfly write pattern plus a bit-reversal scatter -- the core of all digital signal processing."},
"kernel_ntt_v2": {
 "plain": "An FFT done in modular integer arithmetic across several number bases (RNS limbs) -- the transform at the heart of lattice cryptography and homomorphic encryption.",
 "signal": "Visible++: multi-stream modular [[butterfly]] with integer content -- the CKKS/lattice-crypto core."},
"kernel_dct_v2": {
 "plain": "Applies the discrete cosine transform in small 8x8 blocks -- the transform behind JPEG and MPEG image/video compression.",
 "signal": "Visible: many small blocks rewritten with real content -- distinct from the FFT's whole-array [[butterfly]]."},
"kernel_dwt_v2": {
 "plain": "Applies a wavelet transform as a filter-and-downsample [[pyramid|cascade of ever-smaller passes]], halving the data at each level.",
 "signal": "Visible: a shrinking multi-resolution pyramid -- JPEG2000 and wavelet denoising."},
"kernel_fft2d_v2": {
 "plain": "A 2D FFT: FFT every row, transpose the whole array, FFT every row again. The transpose is a large [[scatter]].",
 "signal": "Visible++: the transpose scatter plus two directional passes -- image spectral filtering, turbulence simulation."},

# ---------------- KERNEL: D4 n-body ----------------
"kernel_nbody_v2": {
 "plain": "Simulates particles under gravity by summing pairwise forces, then updating positions and velocities each step -- a few compact per-particle arrays.",
 "signal": "Visible: four small particle arrays rewritten every step, evolving smoothly."},
"kernel_barnes_hut_v2": {
 "plain": "Speeds up n-body gravity by building a [[quadtree]] each step so distant particle groups are approximated as one -- O(n log n) instead of O(n^2).",
 "signal": "Visible/irregular: the rebuilt tree is the distinctive write versus plain n-body's flat arrays."},
"kernel_md_lj_v2": {
 "plain": "Molecular dynamics with a Lennard-Jones potential, using a [[cell list]] rebuilt each step so each atom only checks nearby atoms.",
 "signal": "Visible: the periodic cell-list rebuild plus position/velocity rewrite is the distinctive part."},
"kernel_pic_v2": {
 "plain": "Particle-in-cell plasma simulation: [[scatter]] particle charge onto a grid, solve the field on the grid, then [[gather]] the field back to move particles.",
 "signal": "Visible: the particle-to-grid scatter-deposit plus grid rewrite -- the distinctive coupled write."},
"kernel_fmm_v2": {
 "plain": "The fast multipole method: represent clusters of particles by expansion coefficients on a tree, so far-field effects are cheap.",
 "signal": "Visible: modest expansion-coefficient arrays plus the particle rewrite."},
"kernel_sph_v2": {
 "plain": "Smoothed-particle hydrodynamics -- a fluid modeled as particles, each carrying density and pressure fields summed from its neighbors.",
 "signal": "Visible: extra per-particle fields on top of the particle rewrite; a close relative of molecular dynamics."},

# ---------------- KERNEL: D5 structured grids ----------------
"kernel_stencil_jacobi_v2": {
 "plain": "A 5-point [[stencil]] on a 2D grid, updating each cell from its four neighbors into a fresh grid ([[double-buffer]]), then swapping.",
 "signal": "Visible++: a periodic full-grid rewrite at roughly double footprint -- the textbook regular-simulation pattern."},
"kernel_stencil_seidel_v2": {
 "plain": "The same neighbor-update, but [[in-place]] in a red-black checkerboard order so no separate second grid is needed.",
 "signal": "Visible++: in-place checkerboard writes at single footprint -- contrasts with Jacobi's double-buffered version."},
"kernel_multigrid_v2": {
 "plain": "Solves a grid problem by cycling through coarser and finer grids (a V-cycle), so the active [[working set]] changes scale over time.",
 "signal": "Visible++: a multi-scale, time-varying footprint as it moves between grid levels."},
"kernel_lbm_v2": {
 "plain": "Lattice-Boltzmann fluid simulation: nine distribution arrays are streamed to neighbors and collided locally each step.",
 "signal": "Visible++: nine distribution grids streamed every step -- a heavy, structured write."},
"kernel_fdtd_v2": {
 "plain": "Simulates electromagnetic fields by [[leapfrog|alternately updating]] coupled electric and magnetic grids in time.",
 "signal": "Visible++: two coupled field grids in an E-then-H leapfrog -- the distinctive dual-grid write."},

# ---------------- KERNEL: D6 unstructured grids ----------------
"kernel_fem_assembly_v2": {
 "plain": "Assembles a global finite-element matrix by [[scatter|scatter-adding]] each small element's contribution into shared, irregularly-located slots.",
 "signal": "Visible: indexed scatter-accumulate into a large matrix -- irregular, unlike a regular [[stencil]]."},
"kernel_fem_matvec_v2": {
 "plain": "The matrix-free finite-element version: for each element, gather inputs, apply locally, [[scatter]]-add into a result vector -- never forming the matrix.",
 "signal": "Quieter: the output is a vector, so it writes less than assembly -- the unstructured analog of quiet SpMV."},
"kernel_dg_v2": {
 "plain": "A discontinuous-Galerkin step: dense per-element updates plus flux exchange across element faces.",
 "signal": "Visible: per-element dense blocks rewritten, coupled by face fluxes."},
"kernel_mesh_smooth_v2": {
 "plain": "Smooths an unstructured mesh by moving each node toward the average of its neighbors (Laplacian smoothing), then rewriting the whole node array.",
 "signal": "Visible: a full node-array rewrite -- similar write volume to a [[stencil]] but with irregular neighbor access."},
"kernel_unstructured_fv_v2": {
 "plain": "A finite-volume step: [[gather]] fluxes along a face list and conservatively [[scatter]]-add them into cell values.",
 "signal": "Visible: face-list gather plus conservative cell scatter-add -- the unstructured conservation pattern."},

# ---------------- KERNEL: D7 mapreduce / monte carlo ----------------
"kernel_mc_pi_v2": {
 "plain": "Estimates pi by sampling random points and counting how many land in a circle -- a pure [[reduction]] into a single running total.",
 "signal": "The quiet control: exponential sampling work, but only a scalar accumulator is written -- near-invisible."},
"kernel_histogram_v2": {
 "plain": "Bins many random samples by [[scatter|scatter-incrementing]] counts across a large bins array.",
 "signal": "Visible: random scatter across the whole bins array -- the write-visible counterpart to quiet MC-pi."},
"kernel_mc_option_v2": {
 "plain": "Prices a financial option by simulating many random price paths, storing them all, then averaging the payoff.",
 "signal": "Visible: bulk path-array storage rewritten each pass -- the stored paths are the write."},
"kernel_path_trace_v2": {
 "plain": "A Monte-Carlo renderer that accumulates many random light rays into an image buffer, re-sweeping the whole image each pass.",
 "signal": "Visible: whole-image accumulation, reswept every pass."},
"kernel_diffusion_v2": {
 "plain": "A diffusion-model style sampler that iteratively denoises a whole image/latent, rewriting all of it every step.",
 "signal": "Visible: the entire image rewritten each step -- steady, full-footprint."},

# ---------------- KERNEL: D9 graph traversal ----------------
"kernel_bfs_v2": {
 "plain": "Breadth-first search over a graph built once and then only read. It touches the whole graph, but only by [[gather|reading]]; the sole writes are a small visited/distance array and a queue.",
 "signal": "The quiet control: traversing a static graph is nearly invisible to a write-signal -- the 'graph traversal is invisible' baseline."},
"kernel_rmat_gen_v2": {
 "plain": "Generates a graph by writing out an edge list with the R-MAT recursive random pattern -- the graph is the output, not the input.",
 "signal": "Visible: a bulk edge-list write -- the opposite of quiet BFS, since here the graph is being written."},
"kernel_graph_stream_v2": {
 "plain": "Inserts edges one at a time into a growing adjacency structure, building the graph incrementally.",
 "signal": "Visible: the graph structure itself is written and grown as edges stream in."},
"kernel_label_prop_v2": {
 "plain": "Finds connected components by repeatedly setting each node's label to the smallest label among its neighbors until nothing changes.",
 "signal": "Visible: an iterated node-label array rewrite -- a stencil over a graph."},
"kernel_union_find_v2": {
 "plain": "Merges sets with union-find, flattening the parent-pointer tree as it goes (path compression).",
 "signal": "Visible: the parent-pointer array is rewritten by unions and compression -- distinct from read-only traversal."},

# ---------------- KERNEL: D10 dynamic programming ----------------
"kernel_dp_v2": {
 "plain": "Fills an edit-distance table row by row, each cell depending on its neighbors -- a [[wavefront]] sweeping across a large table.",
 "signal": "Visible: a row-major wavefront, a migrating band of writes across a big table."},
"kernel_floyd_v2": {
 "plain": "Floyd-Warshall all-pairs shortest paths: rewrites the entire n-by-n distance matrix, n times, once per intermediate node.",
 "signal": "Visible: the whole matrix rewritten n times per solve -- very write-heavy."},
"kernel_matrixchain_v2": {
 "plain": "Finds the optimal way to parenthesize a chain of matrix multiplies by filling a table along its anti-diagonals.",
 "signal": "Visible: an anti-diagonal [[wavefront]] fill with an O(n^3) inner loop."},
"kernel_knapsack_v2": {
 "plain": "Solves 0/1 knapsack with a single rolling capacity array, repainted in reverse for each item to save space.",
 "signal": "Visible: one capacity vector repainted per item -- a stationary footprint rewritten many times."},
"kernel_smithwaterman_v2": {
 "plain": "Local sequence alignment: fill a scoring table as a [[wavefront]], then trace the best path backward through it.",
 "signal": "Visible: a row-major fill wavefront plus a fainter backward traceback path."},

# ---------------- KERNEL: D11 backtracking / branch-and-bound ----------------
"kernel_nqueens_count_v2": {
 "plain": "Counts N-queens solutions with fast bitmask [[backtracking]], combining everything into a single counter -- it never stores a solution.",
 "signal": "The quiet control: exponential search, but only a counter is written -- invisible despite the work."},
"kernel_nqueens_enum_v2": {
 "plain": "The exact same search as the counter, but it STORES every solution it finds into a growing array.",
 "signal": "Visible: the whole solution set is appended -- storing the output is what flips this from quiet to loud."},
"kernel_brackets_enum_v2": {
 "plain": "Enumerates every balanced-parenthesis string of a given size and appends each one to an output buffer.",
 "signal": "Visible: Catalan-number-many strings materialized -- a bulk sequential append."},
"kernel_maze_backtrack_v2": {
 "plain": "Solves a maze by depth-first [[backtracking]], marking a large visited/parent grid as it explores and unmarks.",
 "signal": "Visible: a large grid marked and unmarked across the search -- the working state is the write."},
"kernel_graph_coloring_v2": {
 "plain": "Colors a graph as a constraint problem, using forward-checking that prunes and then RESTORES a color/domain array on [[backtracking]].",
 "signal": "Visible: the prune-and-restore churn -- the restore writes are the distinctive backtracking signature."},
"kernel_bnb_tsp_v2": {
 "plain": "Solves traveling-salesman by best-first [[branch-and-bound]], keeping an explicit priority queue of partial tours.",
 "signal": "Visible: the best-first [[frontier]] churned by millions of push/pop operations."},
"kernel_bnb_knapsack_v2": {
 "plain": "Best-first [[branch-and-bound]] over include/exclude choices on a hard, strongly-correlated knapsack instance.",
 "signal": "Visible: the subset-node [[frontier]] heap -- its optimum cross-checks the DP knapsack for correctness."},

# ---------------- KERNEL: D12 graphical models ----------------
"kernel_hmm_v2": {
 "plain": "The scaled forward algorithm for a hidden Markov model: fill a [[trellis]] column by column, each from the previous via a dense matrix-vector step.",
 "signal": "Visible: a column-fill front plus dense matvec, with normalized-probability content."},
"kernel_beliefprop_v2": {
 "plain": "Loopy belief propagation on a grid: iteratively pass and update messages between neighboring cells until they settle.",
 "signal": "Visible: message arrays per grid cell, overwritten each iteration."},
"kernel_kalman_v2": {
 "plain": "Runs an ensemble of Kalman filters, each maintaining and updating its own small dense covariance matrix every step.",
 "signal": "Visible: many small covariance matrices rewritten per step."},
"kernel_gibbs_v2": {
 "plain": "Gibbs sampling on a Potts/Ising grid: sweep the grid resampling each cell from its neighbors, stochastically.",
 "signal": "Visible: a stochastic per-cell resample sweep of the whole grid."},
"kernel_ldpc_v2": {
 "plain": "Decodes an error-correcting LDPC code by passing min-sum messages back and forth across a bipartite (Tanner) graph.",
 "signal": "Visible: the bipartite edge-message arrays iterated until convergence."},

# ---------------- KERNEL: D13 finite state machines ----------------
"kernel_dfa_match_v2": {
 "plain": "Runs a table-driven [[DFA]] over a read-only text stream, writing only the current state and a match counter.",
 "signal": "The quiet control: streaming reads plus a scalar -- matching a pattern is nearly invisible."},
"kernel_lexer_v2": {
 "plain": "A lexer: a character-class state machine that emits one token record per lexeme into a large token array.",
 "signal": "Visible: a token-array append whose size tiles the input -- emitting output is the write."},
"kernel_dfa_build_v2": {
 "plain": "Builds a [[DFA]] from a nondeterministic one by subset construction, writing out a dense transition table.",
 "signal": "Visible: a dense transition-table construction -- the inverse of quietly RUNNING a DFA."},
"kernel_aho_corasick_v2": {
 "plain": "Builds a multi-pattern matching automaton, then scans text appending every match found.",
 "signal": "Visible: the automaton build plus a match-list append."},
"kernel_fsm_transduce_v2": {
 "plain": "A Mealy-machine transducer that reads an input stream and emits a reframed/escaped output stream of similar size (benign, reversible -- not encryption).",
 "signal": "Visible: an output stream roughly the size of the input -- a translating write."},

# ---------------- METHODOLOGY ----------------
"mp_phase_boundary_inference": {
 "plain": "Not a workload itself: it runs another workload as a child and tries to detect the boundaries between its phases from the signal. (The detector here is a placeholder stub.)",
 "signal": "Analysis step -- its memory activity comes from the child workload it launches, not from itself."},
"mp_workingset_metric_linearity": {
 "plain": "Not a workload: it reads the results of an earlier step (the working-set sweep) and checks whether a chosen metric scales linearly with working-set size.",
 "signal": "Pure analysis -- it consumes another step's output, so it must run after that step and cannot be tested in isolation."},
}
