#!/usr/bin/env python3
"""Test/Workload families spec (the DWARFS family) -> docs/test_families_spec.md + .pdf.

One content model, two renderers (Markdown + PDF), same style as
make_feature_substrate_spec.py. Organises the workload corpus by the Berkeley 13
dwarfs (Colella's 7 + Berkeley's 6 -- "A View from Berkeley", 2006). Each dwarf is
a compute/communication motif; we filter by WRITE-visibility (what the host memory
signal can see).

Rules encoded here:
  * A workload that already exists in another family is POINTED to, not duplicated.
  * Each dwarf targets 4-5 distinct workloads; v1 pre-fills only the existing
    pointers and marks the gaps. The new ones are chosen together, dwarf by dwarf,
    in iterations -- edit the WORKLOADS lists below and re-run.

Run: python3 plan07_campaign/make_test_families_spec.py
"""
import html as _html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, KeepTogether)

DOCS = Path(__file__).resolve().parent.parent / "docs"
DATE = "June 2026"
INK = "#1A1A1A"; MUTED = "#5B5B5B"; LINE = "#D8D6CE"; PANEL = "#F5F3EE"

# ---------------------------------------------------------------- content model
# Each dwarf: id, name, origin, vis (Visible|Irregular|Quiet), maps, what, example,
# and workloads: list of (name, status, mech_or_pointer, signature).
#   status: "exists" (already built; mech_or_pointer names its family) or "planned".

# Each dwarf workload row: (name, status, family-pointer, write-signature, used-in real world).
# status "candidate" = a real, heavily-used algorithm in this dwarf's domain that is listed for
# coverage but is NOT (yet) implemented as a test. It still gets a glossary hover + a "Used in" cell.
DWARFS = [
 {"id": "D1", "name": "Dense Linear Algebra", "origin": "Colella-7", "vis": "Visible", "maps": "KERNEL",
  "what": "O(n^3) compute on O(n^2) data; regular, blocked access; rewrites the output / factorised matrix. "
          "Anchored on the kernels that dominate production: GEMM (every neural-net layer), LU/QR (linear "
          "solve, least-squares), plus attention + convolution (the transformer and CNN cores). Three distinct "
          "write fronts -- static full rewrite (GEMM), shrinking trailing front (LU), growing orthogonalised "
          "front (QR) -- with attention/conv added for real-world coverage (their write signature is ~GEMM).",
  "example": "GEMM, LU, QR, attention (transformers), convolution (CNNs), BLAS-3",
  "workloads": [
    ("kernel_gemm_v2", "under-testing", "KERNEL family: blocked dense matmul (-> kernel/D1_visible_dense_linear_algebra/kernel_gemm_v2.c)", "Visible (static full output-C rewrite)", "Every neural-net layer; BLAS-3 (cuBLAS/MKL); scientific computing"),
    ("cpu_matrix_mult_v2", "exists", "CPU family: naive square matmul, writes output C", "Visible (regular; small-scale GEMM)", "Small-scale GEMM control"),
    ("kernel_lu_v2", "planned", "KERNEL family: in-place LU factorisation (-> kernel/D1_visible_dense_linear_algebra/kernel_lu_v2.c)", "Visible (shrinking trailing-submatrix front)", "Linear solve: SPICE circuit sim, finite-element, optimisation (LAPACK dgetrf)"),
    ("kernel_qr_v2", "planned", "KERNEL family: Gram-Schmidt / QR orthogonalisation (-> kernel/D1_visible_dense_linear_algebra/kernel_qr_v2.c)", "Visible (growing orthogonalised-column front)", "Least-squares regression; GMRES/Arnoldi eigensolvers"),
    ("kernel_attention_v2", "planned", "KERNEL family: scaled dot-product attention QK^T/softmax/*V (-> kernel/D1_visible_dense_linear_algebra/kernel_attention_v2.c)", "Visible (transformer core; ~GEMM + row-softmax)", "Every transformer / LLM (the per-token core)"),
    ("kernel_conv_v2", "planned", "KERNEL family: 2D convolution / CNN layer, sliding-window MAC (-> kernel/D1_visible_dense_linear_algebra/kernel_conv_v2.c)", "Visible (overlapping-window feature-map rewrite)", "CNNs: image & video vision models"),
    ("Cholesky factorisation", "candidate", "KERNEL (candidate): in-place symmetric factorisation, triangular shrinking front", "Visible (LU-like triangular front)", "Kalman filters, finance covariance, normal-equation least-squares"),
    ("Triangular solve (TRSM)", "candidate", "KERNEL (candidate): forward/back substitution, column wavefront", "Visible (column-wise solve front)", "The solve step inside LU/Cholesky; preconditioners"),
  ]},
 {"id": "D2", "name": "Sparse Linear Algebra", "origin": "Colella-7", "vis": "Visible", "maps": "split: IDLE (spmv control) + KERNEL (writers)",
  "what": "Sparse linear algebra is QUIET only when the OUTPUT is small: SpMV reads a big sparse matrix by "
          "indirect gather but writes just a vector, so it is near-idle -- the classic 'important but "
          "invisible' case (kept as kernel_spmv_v2, the control). But sparse work that produces a LARGE "
          "output IS write-visible, and that is most of modern practice: SpMM (a dense output, the graph-"
          "neural-network aggregation kernel), sparse Cholesky (fill-in factors), SpGEMM (a new sparse "
          "matrix, algebraic multigrid), SDDMM (a sampled sparse output, graph attention), and MoE dispatch "
          "(a token-permutation scatter, modern LLMs). So D2 spans a quiet half and a visible half.",
  "example": "SpMV (quiet control), SpMM, sparse Cholesky, SpGEMM, SDDMM, MoE dispatch",
  "workloads": [
    ("kernel_spmv_v2", "under-testing", "IDLE family (QUIET control): CSR SpMV, gather-dominated, tiny vector write (-> kernel/D2_visible_sparse_linear_algebra/kernel_spmv_v2.c)", "Quiet / near-idle (the invisible baseline)", "Inner loop of CG/GMRES solvers; recommenders; graph-as-matrix"),
    ("kernel_spmm_v2", "under-testing", "KERNEL family (VISIBLE): sparse x dense -> dense output (-> kernel/D2_visible_sparse_linear_algebra/kernel_spmm_v2.c)", "Visible (large dense output rewritten each pass)", "Graph neural networks (the aggregation kernel; DGL/PyG)"),
    ("kernel_sparse_cholesky_v2", "under-testing", "KERNEL family (VISIBLE): banded sparse Cholesky, fill-in (-> kernel/D2_visible_sparse_linear_algebra/kernel_sparse_cholesky_v2.c)", "Visible (factor fills in within the band; progressive write)", "Direct solvers (CHOLMOD/MUMPS): FEM, circuits, optimisation"),
    ("kernel_spgemm_v2", "under-testing", "KERNEL family (VISIBLE): sparse x sparse -> new sparse matrix (-> kernel/D2_visible_sparse_linear_algebra/kernel_spgemm_v2.c)", "Visible (writes a new sparse matrix with fill-in)", "Algebraic multigrid setup; triangle counting; graph contraction"),
    ("kernel_sddmm_v2", "under-testing", "KERNEL family (VISIBLE): sampled dense-dense -> sparse output (-> kernel/D2_visible_sparse_linear_algebra/kernel_sddmm_v2.c)", "Visible (scattered writes at the sparse mask positions)", "Graph-attention networks; recommender systems"),
    ("kernel_moe_dispatch_v2", "under-testing", "KERNEL family (VISIBLE): MoE token dispatch/combine scatter-gather (-> kernel/D2_visible_sparse_linear_algebra/kernel_moe_dispatch_v2.c)", "Visible (token-permutation scatter into expert buffers)", "Mixture-of-Experts LLMs (Mixtral, DeepSeek); >60% of 2025-26 releases"),
    ("PageRank", "covered", "covered by kernel_spmv_v2 (repeated SpMV; same quiet gather + small vector write)", "same quiet gather", "Google web ranking; centrality; link analysis"),
    ("Conjugate Gradient (CG)", "covered", "covered by kernel_spmv_v2 (SpMV-bound iteration, small vector writes)", "same quiet gather", "FEM/CFD SPD iterative solver"),
    ("Sparse triangular solve (SpTRSV)", "covered", "covered by kernel_spmv_v2 (dependency-ordered sparse solve, tiny writes)", "same quiet gather", "ILU preconditioners"),
  ]},
 {"id": "D3", "name": "Spectral Methods", "origin": "Colella-7", "vis": "Visible++", "maps": "KERNEL",
  "what": "Butterfly / bit-reversed, strided, multi-pass; in-place rewrite of the whole array. FIVE distinct "
          "write-patterns cover the dwarf: 1D butterfly (FFT), multi-stream butterfly (NTT), blocked (DCT), "
          "halving pyramid (DWT), transpose + two-direction (2D-FFT). Famous transforms that reduce to these "
          "-- DTFT, cepstrum, MFCC, STFT, wavelet scattering, FFT-convolution -- are marked 'covered' and "
          "point to the built test whose signature already captures them (building them would be pseudoreplication).",
  "example": "FFT, NTT, DCT, DWT, 2D-FFT (covers DTFT, cepstrum, MFCC, STFT, wavelet scattering)",
  "workloads": [
    ("kernel_fft_v2", "under-testing", "KERNEL family: in-place radix-2 FFT (-> kernel/D3_visible_spectral_methods/kernel_fft_v2.c)", "Visible++ (1D butterfly; bit-reversal scatter)", "All DSP: audio, radio/5G, MRI, image filtering"),
    ("kernel_ntt_v2", "under-testing", "KERNEL family: RNS multi-limb number-theoretic transform, CKKS/lattice core (-> kernel/D3_visible_spectral_methods/kernel_ntt_v2.c)", "Visible++ (multi-stream modular butterfly; integer content)", "Lattice crypto / CKKS homomorphic encryption; big-integer multiply"),
    ("kernel_dct_v2", "planned", "KERNEL family: blocked 8x8 discrete cosine transform (-> kernel/D3_visible_spectral_methods/kernel_dct_v2.c)", "Visible (many small blocks; real content)", "JPEG / MPEG / H.264 image & video compression"),
    ("kernel_dwt_v2", "planned", "KERNEL family: discrete wavelet transform, filter+downsample pyramid (-> kernel/D3_visible_spectral_methods/kernel_dwt_v2.c)", "Visible (halving multi-resolution pyramid)", "JPEG2000, denoising, audio/image compression"),
    ("kernel_fft2d_v2", "planned", "KERNEL family: 2D FFT (row FFTs, transpose, column FFTs) (-> kernel/D3_visible_spectral_methods/kernel_fft2d_v2.c)", "Visible++ (transpose scatter + two-direction passes)", "Image/optics spectral filtering, turbulence DNS, crystallography"),
    ("DTFT / direct DFT", "covered", "covered by kernel_fft_v2 (computable DTFT = DFT = FFT; naive DFT = dense matvec)", "same 1D-butterfly signature", "Frequency analysis (textbook form of the FFT)"),
    ("Cepstrum", "covered", "covered by kernel_fft_v2 (FFT -> log|.| -> inverse FFT)", "two butterfly passes + pointwise", "Pitch detection, echo / speaker analysis"),
    ("MFCC", "covered", "covered by kernel_fft_v2 + kernel_dct_v2 (FFT -> mel filterbank -> log -> DCT)", "butterfly + blocked cosine", "The classic speech-recognition audio feature"),
    ("STFT / spectrogram", "covered", "covered by kernel_fft_v2 (sliding windowed FFTs filling a spectrogram)", "repeated 1D butterfly (sliding window)", "Every audio ML front-end; speech, music"),
    ("Wavelet scattering (WST)", "covered", "covered by kernel_dwt_v2 (cascade of wavelet transforms + modulus)", "repeated pyramid", "Audio / image classification features"),
    ("FFT convolution", "covered", "covered by kernel_fft_v2 (forward FFT, pointwise multiply, inverse FFT)", "two butterfly passes + pointwise", "Large-kernel convolution; polynomial / big-integer multiply"),
  ]},
 {"id": "D4", "name": "N-Body Methods", "origin": "Colella-7", "vis": "Visible", "maps": "KERNEL",
  "what": "N particles interacting pairwise; rewrites compact particle arrays (position/velocity) smoothly "
          "each timestep. Variants differ mostly in how they READ to compute forces (all-pairs, tree, "
          "neighbour lists) -- invisible to the write-signal -- so they are distinguished by the EXTRA "
          "structure they WRITE: Barnes-Hut a tree, FMM expansion arrays, MD cell/neighbour lists, PIC a "
          "grid (scatter-deposit). All-pairs adds no new write, so it is 'covered' by the baseline nbody.",
  "example": "gravity nbody, Barnes-Hut, molecular dynamics, particle-in-cell, FMM, SPH",
  "workloads": [
    ("kernel_nbody_v2", "under-testing", "KERNEL family: 2D K-sampled gravity (-> kernel/D4_visible_nbody_methods/kernel_nbody_v2.c)", "Visible (four compact arrays rewritten per step, smooth)", "2D gravity model; astrophysics, particle effects"),
    ("kernel_barnes_hut_v2", "planned", "KERNEL family: quadtree build + traversal, O(n log n) (-> kernel/D4_visible_nbody_methods/kernel_barnes_hut_v2.c)", "Visible/irregular (tree nodes rebuilt each step + particle rewrite)", "Cosmology (GADGET), galaxy formation"),
    ("kernel_md_lj_v2", "planned", "KERNEL family: Lennard-Jones MD with cell lists (-> kernel/D4_visible_nbody_methods/kernel_md_lj_v2.c)", "Visible (periodic cell-list rebuild + position/velocity rewrite)", "GROMACS / NAMD / AMBER: drug discovery, protein folding, materials"),
    ("kernel_pic_v2", "planned", "KERNEL family: particle-in-cell (scatter to grid, field solve, gather) (-> kernel/D4_visible_nbody_methods/kernel_pic_v2.c)", "Visible (particle->grid scatter-deposit + grid rewrite)", "Plasma physics, accelerators, semiconductor device sim"),
    ("kernel_fmm_v2", "planned", "KERNEL family: fast multipole (multipole/local expansion arrays on a tree) (-> kernel/D4_visible_nbody_methods/kernel_fmm_v2.c)", "Visible (expansion-coefficient arrays + particle rewrite)", "Electrostatics, acoustics, fast O(n) far-field"),
    ("kernel_sph_v2", "planned", "KERNEL family: smoothed-particle hydrodynamics (density/pressure fields) (-> kernel/D4_visible_nbody_methods/kernel_sph_v2.c)", "Visible (extra per-particle fields + particle rewrite)", "Fluid sim, film VFX (water/lava), astrophysics"),
    ("All-pairs direct N-body", "covered", "covered by kernel_nbody_v2 (same particle-array writes; differs only in the force READS)", "same smooth particle rewrite", "Exact small-N molecular dynamics; reference force"),
  ]},
 {"id": "D5", "name": "Structured Grids", "origin": "Colella-7", "vis": "Visible++", "maps": "KERNEL",
  "what": "Regular neighbour sweep over a grid, iterative; rewrites the grid each iteration. Covered by "
          "3 genuinely-distinct members (double-buffer / in-place / multigrid); further stencil variants "
          "(9-point, 3D, separable) differ only in reads/content -> signal-redundant. LBM/FDTD listed as "
          "distinct candidates (multiple field arrays, leapfrog).",
  "example": "stencil, Jacobi/Gauss-Seidel, multigrid, Lattice-Boltzmann, FDTD",
  "workloads": [
    ("kernel_stencil_jacobi_v2", "under-testing", "KERNEL family: 2D 5-point Jacobi, double-buffer (-> kernel/D5_visible_structured_grids/kernel_stencil_jacobi_v2.c)", "Visible++ (periodic full rewrite, ~2x footprint)", "Finite-difference PDE: heat / Poisson / diffusion solvers"),
    ("kernel_stencil_seidel_v2", "under-testing", "KERNEL family: Gauss-Seidel red-black, in-place (-> kernel/D5_visible_structured_grids/kernel_stencil_seidel_v2.c)", "Visible++ (checkerboard in-place, ~1x footprint)", "PDE relaxation; smoother inside multigrid"),
    ("kernel_multigrid_v2", "under-testing", "KERNEL family: multigrid V-cycle (-> kernel/D5_visible_structured_grids/kernel_multigrid_v2.c)", "Visible++ (multi-scale, time-varying footprint)", "The optimal PDE solver; CFD, electrostatics"),
    ("kernel_lbm_v2", "under-testing", "KERNEL family: Lattice-Boltzmann D2Q9, stream + collide over 9 distribution arrays (-> kernel/D5_visible_structured_grids/kernel_lbm_v2.c)", "Visible++ (nine distribution arrays streamed each step)", "CFD: porous media, aerodynamics (OpenLB / Palabos)"),
    ("kernel_fdtd_v2", "under-testing", "KERNEL family: 2D TM FDTD, leapfrog of coupled E/H field grids (-> kernel/D5_visible_structured_grids/kernel_fdtd_v2.c)", "Visible++ (two coupled field grids, E<->H leapfrog)", "Antenna / radar / photonics simulation (Meep)"),
  ]},
 {"id": "D6", "name": "Unstructured Grids", "origin": "Colella-7", "vis": "Visible", "maps": "KERNEL (irregular access)",
  "what": "The irregular cousin of D5: PDE computation on unstructured meshes, reaching neighbours through "
          "explicit connectivity / adjacency lists (indirect gather). It writes real mesh/matrix arrays, so it "
          "is visible -- but the honest nuance is that the irregular ACCESS is mostly reads (invisible), so a "
          "plain unstructured relaxation writes the same footprint as a structured stencil. The genuinely "
          "distinct write is the SCATTER-ACCUMULATE that structured grids do not do: assembling a global "
          "matrix, or scatter-adding fluxes into cells. FEM matvec is the quieter member (a vector output, "
          "the matrix-free analog of SpMV).",
  "example": "FEM assembly, matrix-free FEM matvec, discontinuous Galerkin, mesh smoothing, unstructured finite-volume",
  "workloads": [
    ("kernel_fem_assembly_v2", "under-testing", "KERNEL family: scatter-add element matrices into a global matrix (-> kernel/D6_visible_unstructured_grids/kernel_fem_assembly_v2.c)", "Visible (indexed scatter-accumulate into a large matrix)", "FEM assembly: ANSYS / Abaqus structural & crash; aerospace"),
    ("kernel_fem_matvec_v2", "under-testing", "KERNEL family: matrix-free FEM matvec, element gather-apply-scatter (-> kernel/D6_visible_unstructured_grids/kernel_fem_matvec_v2.c)", "Quieter (scatter-add into a result VECTOR; the matrix-free SpMV analog)", "Large FEM/CFD iterative solvers (matrix-free)"),
    ("kernel_dg_v2", "under-testing", "KERNEL family: discontinuous Galerkin step, per-element dense + face flux (-> kernel/D6_visible_unstructured_grids/kernel_dg_v2.c)", "Visible (per-element dense blocks rewritten + flux coupling)", "High-order CFD & seismic wave propagation"),
    ("kernel_mesh_smooth_v2", "under-testing", "KERNEL family: unstructured Laplacian mesh smoothing (-> kernel/D6_visible_unstructured_grids/kernel_mesh_smooth_v2.c)", "Visible (node-array rewrite; write ~ D5 stencil, distinct in access)", "Graphics mesh processing; remeshing"),
    ("kernel_unstructured_fv_v2", "under-testing", "KERNEL family: finite-volume, conservative face-flux scatter-add into cells (-> kernel/D6_visible_unstructured_grids/kernel_unstructured_fv_v2.c)", "Visible (face-list gather + conservative cell scatter-add)", "OpenFOAM CFD; aerodynamics"),
    ("Mesh partitioning (METIS)", "candidate", "KERNEL-irregular (candidate): graph coarsen / partition / refine", "Irregular (graph rewrite; closer to D9)", "Parallel FEM domain decomposition"),
  ]},
 {"id": "D7", "name": "MapReduce / Monte Carlo", "origin": "Colella-7", "vis": "Partial", "maps": "split",
  "what": "Embarrassingly parallel map (writes intermediates) + reduce (small); or RNG accumulate. "
          "sqlite_analytical is a loose proxy; the real members are Monte-Carlo (quiet RNG-accumulate) and "
          "histogram/word-count (visible scatter-write map).",
  "example": "Monte-Carlo integration, option pricing, MCMC, word-count, path tracing",
  "workloads": [
    ("app_sqlite_analytical_v2", "exists", "APP family (loose): read-heavy aggregation/scan over a DB", "Quiet-ish (read-dominated)", "Analytical SQL scan / aggregate (OLAP)"),
    ("Monte-Carlo integration", "candidate", "IDLE (candidate): RNG sample + running accumulate", "Quiet (register-resident accumulate)", "Physics, finance, Bayesian; high-dimensional integrals"),
    ("Monte-Carlo option pricing", "candidate", "IDLE (candidate): simulate many price paths, average payoff", "Quiet (RNG + small accumulators)", "Quant finance: derivatives, VaR, risk"),
    ("MCMC (Metropolis-Hastings)", "candidate", "IDLE (candidate): proposal + accept/reject random walk", "Quiet (small state rewrite)", "Bayesian inference (Stan / PyMC), statistical physics"),
    ("Histogram / word-count", "candidate", "MEM (candidate): scatter increments into bins / a hash map", "Visible-ish (scattered bin writes)", "MapReduce / Spark ETL; analytics"),
    ("Path tracing", "candidate", "MEM (candidate): trace random rays, accumulate into an image buffer", "Visible (image-buffer accumulation)", "Film & game rendering (RenderMan, Blender Cycles)"),
  ]},
 {"id": "D8", "name": "Combinational Logic", "origin": "Berkeley+6", "vis": "Quiet / Visible", "maps": "control OR threat-labeled",
  "what": "Simple bit-level ops over large data. CRC/hash = quiet; ENCRYPTION = visible high-entropy rewrite.",
  "example": "compression, SHA-256, AES, CRC32, hashing",
  "workloads": [
    ("app_compress_gzip_v2", "exists", "APP family: LZ77+Huffman stream compress, writes output", "Visible (entropy-changing)", "Web (HTTP gzip), storage, backups (DEFLATE)"),
    ("app_decompress_gzip_v2", "exists", "APP family: inverse, small read -> large write", "Visible", "Web / storage decompression"),
    ("cpu_hash_loop_v2", "exists", "CPU family: FNV hash over a stream", "Quiet (register-resident)", "Hash tables, checksums"),
    ("sandbox_ransom_* (4 variants)", "exists", "SECURITY family (THREAT-labeled): discover->read->XOR->write->rename", "Visible++ (high-entropy rewrite)", "Benign XOR; encryption-shaped rewrite control"),
    ("SHA-256", "candidate", "CPU/IDLE (candidate): streaming compression function, tiny digest", "Quiet (stream read, 32-byte write)", "git, blockchain, TLS certificates, dedup, integrity"),
    ("AES block cipher", "covered", "covered by sandbox_ransom_* (same high-entropy full-rewrite signature)", "Visible (high-entropy output rewrite)", "HTTPS/TLS, disk encryption (BitLocker / FileVault), VPN"),
    ("CRC32", "candidate", "CPU/IDLE (candidate): table-driven rolling checksum", "Quiet (register-resident)", "Ethernet, ZIP, storage error detection"),
  ]},
 {"id": "D9", "name": "Graph Traversal", "origin": "Berkeley+6", "vis": "Irregular", "maps": "KERNEL-irregular / scanner",
  "what": "Visit objects via indirect lookups, little compute; writes visited/frontier/distance arrays. "
          "scanner_metadata is a directory-walk proxy; the real member is BFS (Graph500). Heavily used, "
          "but quiet (gather-dominated, small frontier/visited writes).",
  "example": "BFS, DFS, Dijkstra/A*, connected components",
  "workloads": [
    ("sandbox_scanner_metadata", "exists", "SECURITY family (loose): directory enumeration via stat", "Quiet / metadata", "Directory-walk proxy"),
    ("Breadth-first search (BFS)", "candidate", "KERNEL-irregular (candidate): frontier expand + visited-bitmap churn", "Quiet/irregular (frontier + visited writes)", "Graph500; shortest unweighted path; GC mark; social graph"),
    ("Depth-first search (DFS)", "candidate", "KERNEL-irregular (candidate): explicit stack, visited marks", "Quiet (stack + visited writes)", "Topological sort, cycle detection, package resolvers (npm/cargo)"),
    ("Dijkstra / A* shortest path", "candidate", "KERNEL-irregular (candidate): priority-queue relax, distance array", "Quiet (heap + distance writes)", "GPS routing, network routing, game pathfinding"),
    ("Connected components", "candidate", "KERNEL-irregular (candidate): union-find or label propagation", "Quiet (label array writes)", "Clustering, image segmentation, fraud-ring detection"),
  ]},
 {"id": "D10", "name": "Dynamic Programming", "origin": "Berkeley+6", "vis": "Visible", "maps": "KERNEL",
  "what": "Fill a 1D/2D table, each cell from neighbours; regular monotone fill front (wavefront).",
  "example": "edit distance, Smith-Waterman, Viterbi, Floyd-Warshall, knapsack",
  "workloads": [
    ("kernel_dp_v2", "under-testing", "KERNEL family: edit-distance table fill (-> kernel/D10_visible_dynamic_programming/kernel_dp_v2.c)", "Visible (row-major wavefront; migrating band on a large table)", "git diff, spell-check, fuzzy matching"),
    ("kernel_floyd_v2", "under-testing", "KERNEL family: Floyd-Warshall all-pairs shortest paths (-> kernel/D10_visible_dynamic_programming/kernel_floyd_v2.c)", "Visible (whole n*n matrix rewritten n times per solve)", "All-pairs shortest path; routing; transitive closure"),
    ("kernel_matrixchain_v2", "under-testing", "KERNEL family: matrix-chain optimal parenthesisation (-> kernel/D10_visible_dynamic_programming/kernel_matrixchain_v2.c)", "Visible (anti-diagonal fill by chain length, O(n^3) inner)", "Compiler / query-plan optimisation, NLP parsing"),
    ("kernel_knapsack_v2", "under-testing", "KERNEL family: 0/1 knapsack, space-optimised 1D rolling array (-> kernel/D10_visible_dynamic_programming/kernel_knapsack_v2.c)", "Visible (single capacity vector repainted in reverse per item)", "Resource allocation, scheduling, finance"),
    ("kernel_smithwaterman_v2", "under-testing", "KERNEL family: Smith-Waterman local alignment, fill + traceback (-> kernel/D10_visible_dynamic_programming/kernel_smithwaterman_v2.c)", "Visible (row-major wavefront + backward traceback path)", "Genomics local alignment (BLAST family)"),
    ("Needleman-Wunsch", "covered", "covered by kernel_dp_v2 (identical global-alignment row-major wavefront)", "same row-major wavefront", "Global DNA / protein sequence alignment"),
    ("Viterbi decoding", "covered", "covered by kernel_hmm_v2 (same trellis column-fill, max-product instead of sum)", "same column-fill front", "Speech recognition, error-correction decode, POS tagging"),
  ]},
 {"id": "D11", "name": "Backtrack / Branch-and-Bound", "origin": "Berkeley+6", "vis": "Quiet", "maps": "CPU/IDLE control",
  "what": "Explore + prune a search tree; writes a small search stack / current solution. Real and heavily "
          "used (SAT, MILP) but quiet -- deep recursion over a tiny working set. No test built yet.",
  "example": "N-queens, Sudoku, DPLL/CDCL SAT, branch-and-bound MILP, TSP",
  "workloads": [
    ("N-queens", "candidate", "CPU/IDLE (candidate): place + backtrack over a board", "Quiet (tiny board state)", "Classic constraint-satisfaction benchmark"),
    ("Sudoku solver", "candidate", "CPU/IDLE (candidate): constraint propagation + backtracking", "Quiet (81-cell grid)", "Constraint-propagation teaching / puzzle solvers"),
    ("DPLL / CDCL SAT", "candidate", "CPU/IDLE (candidate): unit-propagate, decide, learn clauses, backtrack", "Quiet (clause DB reads, small writes)", "Hardware/chip verification, planning (MiniSat / Z3)"),
    ("Branch-and-bound MILP", "candidate", "CPU/IDLE (candidate): LP-relaxation bound, branch, prune", "Quiet (search tree, small writes)", "Logistics, scheduling, optimisation (Gurobi / CPLEX)"),
    ("TSP branch-and-bound", "candidate", "CPU/IDLE (candidate): tour bound + prune search", "Quiet (path/stack writes)", "Routing, VLSI, operations research"),
  ]},
 {"id": "D12", "name": "Graphical Models", "origin": "Berkeley+6", "vis": "Visible", "maps": "KERNEL",
  "what": "Probability ops over a graph; writes belief/message tables (matrix-like).",
  "example": "HMM, Kalman filter, LDPC belief propagation, loopy BP, Gibbs sampling",
  "workloads": [
    ("kernel_hmm_v2", "under-testing", "KERNEL family: scaled HMM forward, trellis fill (-> kernel/D12_visible_graphical_models/kernel_hmm_v2.c)", "Visible (column-fill front + dense matvec; normalised-probability content)", "Speech recognition, gene finding, time-series"),
    ("kernel_beliefprop_v2", "under-testing", "KERNEL family: loopy sum-product belief propagation on a grid MRF (-> kernel/D12_visible_graphical_models/kernel_beliefprop_v2.c)", "Visible (iterated message arrays per grid cell)", "Stereo vision, image denoising, MRF / CRF"),
    ("kernel_kalman_v2", "under-testing", "KERNEL family: ensemble of Kalman filters, dense covariance updates (-> kernel/D12_visible_graphical_models/kernel_kalman_v2.c)", "Visible (many small d*d covariance matrices rewritten per step)", "GPS/INS navigation, object tracking, sensor fusion"),
    ("kernel_gibbs_v2", "under-testing", "KERNEL family: Gibbs sampling on a Potts/Ising grid (-> kernel/D12_visible_graphical_models/kernel_gibbs_v2.c)", "Visible (stochastic per-cell resample sweep of the grid)", "Bayesian inference, topic models (LDA)"),
    ("kernel_ldpc_v2", "under-testing", "KERNEL family: LDPC min-sum decoder, message passing on a Tanner graph (-> kernel/D12_visible_graphical_models/kernel_ldpc_v2.c)", "Visible (bipartite edge-message arrays iterated)", "5G / WiFi / SSD / satellite error correction"),
  ]},
 {"id": "D13", "name": "Finite State Machines", "origin": "Berkeley+6", "vis": "Quiet", "maps": "CPU/IDLE control / parser",
  "what": "Read a stream, transition through states via table lookup; tiny memory write. Ubiquitous in "
          "systems software (parsing, regex, protocol decode) but quiet -- stream read + branch, tiny state.",
  "example": "JSON parse, regex/DFA, Aho-Corasick, HTTP parser, lexer",
  "workloads": [
    ("app_json_parse_v2", "exists", "APP family: streaming JSON parse (the canonical FSM)", "Quiet (read + branch)", "Every REST / JSON API; config parsing"),
    ("cpu_branch_random_v2", "exists", "CPU family (loose): data-dependent random branches", "Quiet", "Branch-predictor control"),
    ("Regex / DFA matcher", "candidate", "CPU/IDLE (candidate): table-driven state transitions over a stream", "Quiet (state-table lookups, tiny writes)", "grep, input validation, log processing"),
    ("Aho-Corasick", "candidate", "CPU/IDLE (candidate): multi-pattern automaton over a stream", "Quiet (goto/fail table reads)", "Antivirus / IDS multi-pattern scan (ClamAV / Snort)"),
    ("HTTP / protocol parser", "candidate", "CPU/IDLE (candidate): byte-by-byte protocol state machine", "Quiet (small parse-state writes)", "Web servers (nginx), TCP/IP stacks, deep-packet inspection"),
    ("Lexer / tokenizer", "candidate", "CPU/IDLE (candidate): character-class FSM emitting tokens", "Quiet (token-buffer writes)", "Every compiler / interpreter front-end"),
  ]},
]

TARGET = "4-5"

# ---- FIRST DIVISION: the behaviour families by memory signature (all 38 built workloads) ----
# Each workload: (name, dwarf cross-ref or "--", mechanism / note).
FAMILIES = [
 {"id": "S1", "name": "IDLE (+ CPU boundary)", "sig": "near-zero writes",
  "intro": "The no-write floor. CPU-bound workloads sit here as the warm/active boundary (pure "
           "compute is near-invisible to a write signal); matrix_mult drifts toward MEM via its "
           "output writes.",
  "workloads": [
    ("run_idle", "--", "sleep baseline"),
    ("idle_long_baseline", "--", "long uncontaminated idle (optional cache-drop)"),
    ("idle_post_workload_recovery", "--", "idle after a prior workload (writeback decay)"),
    ("cpu_hash_loop_v2", "Combinational Logic", "register-resident hash; near-idle (CPU boundary)"),
    ("cpu_branch_random_v2", "Finite State Machines", "random branches; near-idle (CPU boundary)"),
    ("cpu_matrix_mult_v2", "Dense Linear Algebra", "matmul; writes output C -> drifts to MEM"),
    ("kernel_spmv_v2", "Sparse Linear Algebra", "SpMV quiet control: gather-dominated, read-only structure -> near-idle (kernel/D2_quiet_sparse_linear_algebra/kernel_spmv_v2.c)"),
  ]},
 {"id": "S2", "name": "MEM (+ CACHE sub-family)", "sig": "working-set writes",
  "intro": "Anonymous working-set writes. CACHE workloads are a footprint/locality sub-family "
           "(small = quiet, large = loud). These are access primitives, not computational motifs "
           "(no dwarf).",
  "workloads": [
    ("mem_workingset_sweep_v2", "--", "page-strided writes, footprint sweep"),
    ("mem_writemag_sweep_v2", "--", "per-page write-magnitude sweep"),
    ("mem_rmw_intensity_v2", "--", "read-modify-write intensity"),
    ("mem_pagefault_density_v2", "--", "fault vs steady-state touch"),
    ("mem_mmap_traversal_v2", "--", "mmap file traversal"),
    ("mem_random_write_pages_v2", "--", "random page writes"),
    ("mem_stride_sweep_large_v2", "--", "large-buffer stride sweep"),
    ("cache_hot_loop_v2", "--", "L1-resident RMW (CACHE sub; quiet)"),
    ("cache_cold_scan_v2", "--", "> LLC linear scan (CACHE sub)"),
    ("cache_stride_sweep_v2", "--", "> LLC stride (CACHE sub)"),
  ]},
 {"id": "S3", "name": "IO", "sig": "page-cache + metadata writes",
  "intro": "Writes through the page cache + file metadata. Cold reads (cache fills) count here; "
           "io_read_cache_hit is the weak (near-idle) member.",
  "workloads": [
    ("io_read_cache_hit_v2", "--", "cache-hot reads; near-idle (weak member)"),
    ("io_direct_write_like_v2", "--", "O_DIRECT writes, bypass page cache"),
  ]},
 {"id": "S4", "name": "THREAD", "sig": "shared-line + allocator writes",
  "intro": "Concurrency write patterns: shared cache lines, futex/scheduler, allocator churn. "
           "Contention types are quiet in APF (visible via temporal/revisit features).",
  "workloads": [
    ("thread_lock_contention_v2", "--", "mutex + shared cacheline"),
    ("thread_producer_consumer_v2", "--", "ring buffer + condvar"),
    ("thread_parallel_alloc_v2", "--", "concurrent malloc/free churn"),
  ]},
 {"id": "S5", "name": "BULK-REWRITE / encryptor", "sig": "high-entropy full rewrites",
  "intro": "The threat-shaped cluster: discover -> read -> transform -> write -> rename over many "
           "files, high-entropy output. Dwarf: Combinational Logic (encryption).",
  "workloads": [
    ("sandbox_ransom_seq", "Combinational Logic", "sequential 5-phase per file"),
    ("sandbox_ransom_batched", "Combinational Logic", "batched phases (all-at-once)"),
    ("sandbox_ransom_slowburn", "Combinational Logic", "paced: one file / interval"),
    ("sandbox_ransom_selective", "Combinational Logic", "selective subset (.dat only)"),
  ]},
 {"id": "S6", "name": "ENUMERATION / metadata", "sig": "stat-heavy, few writes",
  "intro": "Directory / metadata enumeration (stat, readdir), little content write. "
           "Dwarf: Graph Traversal.",
  "workloads": [
    ("sandbox_scanner_metadata", "Graph Traversal", "directory enumeration via stat"),
  ]},
 {"id": "S7", "name": "STEALTH / trickle", "sig": "low-rate, high-intensity writes",
  "intro": "Low page-count, high per-page intensity -- the low-APF / high-Hamming probe. A "
           "modulation of a write motif, not a dwarf of its own.",
  "workloads": [
    ("sandbox_stealth_microwrite", "--", "single-page micro-rewrites"),
    ("sandbox_stealth_scattered", "--", "scattered page rewrites"),
    ("sandbox_stealth_paced", "--", "time-jittered single-page writes"),
  ]},
 {"id": "S8", "name": "APP", "sig": "real-application write mixes",
  "intro": "Whole real applications (not primitives). Each also carries a dwarf label: "
           "compression = Combinational Logic, parsing = FSM, DB = MapReduce.",
  "workloads": [
    ("app_hashtable_intensive_v2", "Combinational Logic", "open-addressing hash build/probe"),
    ("app_sqlite_oltp_v2", "MapReduce / Monte Carlo", "OLTP WAL append + checkpoint"),
    ("app_sqlite_analytical_v2", "MapReduce / Monte Carlo", "analytical aggregation / scan"),
    ("app_compress_gzip_v2", "Combinational Logic", "gzip compress (LZ77 + Huffman)"),
    ("app_decompress_gzip_v2", "Combinational Logic", "gzip decompress"),
    ("app_json_parse_v2", "Finite State Machines", "streaming JSON parse"),
  ]},
 {"id": "S9", "name": "MIXED", "sig": "blended write patterns",
  "intro": "Deliberate superpositions of other families; visible because their mem/io parts write. "
           "A blend by construction.",
  "workloads": [
    ("mixed_mem_io_v2", "--", "concurrent mem writes + file IO"),
    ("mixed_cpu_mem_v2", "--", "compute + mem pressure"),
    ("mixed_cpu_io_v2", "--", "cpu loop + file IO"),
  ]},
 {"id": "S10", "name": "KERNEL (compute motifs)", "sig": "structured-compute writes",
  "intro": "Structured-compute writers -- the Berkeley dwarf kernels. Regular / periodic write "
           "patterns, distinct from MEM's amorphous sweeps. Each carries its dwarf label on axis 2. "
           "Built one at a time (see Status); the first is the D5 stencil.",
  "workloads": [
    ("kernel_stencil_jacobi_v2", "Structured Grids", "2D 5-point Jacobi; full-grid rewrite, double-buffer (kernel/D5_visible_structured_grids/kernel_stencil_jacobi_v2.c)"),
    ("kernel_stencil_seidel_v2", "Structured Grids", "Gauss-Seidel red-black, in-place; checkerboard writes (kernel/D5_visible_structured_grids/kernel_stencil_seidel_v2.c)"),
    ("kernel_multigrid_v2", "Structured Grids", "geometric multigrid V-cycle; multi-scale, time-varying footprint (kernel/D5_visible_structured_grids/kernel_multigrid_v2.c)"),
    ("kernel_fft_v2", "Spectral Methods", "in-place radix-2 FFT; stage-varying-stride butterfly + bit-reversal scatter (kernel/D3_visible_spectral_methods/kernel_fft_v2.c)"),
    ("kernel_gemm_v2", "Dense Linear Algebra", "blocked dense matmul C=A*B; large output-C rewrite (kernel/D1_visible_dense_linear_algebra/kernel_gemm_v2.c)"),
    ("kernel_nbody_v2", "N-Body Methods", "2D particle sim; four compact arrays rewritten per step, smooth evolution (kernel/D4_visible_nbody_methods/kernel_nbody_v2.c)"),
    ("kernel_dp_v2", "Dynamic Programming", "edit-distance DP table fill; row-major wavefront (kernel/D10_visible_dynamic_programming/kernel_dp_v2.c)"),
    ("kernel_hmm_v2", "Graphical Models", "scaled HMM forward; probability-trellis column fill + dense transition matvec (kernel/D12_visible_graphical_models/kernel_hmm_v2.c)"),
    ("kernel_lu_v2", "Dense Linear Algebra", "in-place LU factorisation; shrinking trailing-submatrix front (kernel/D1_visible_dense_linear_algebra/kernel_lu_v2.c)"),
    ("kernel_qr_v2", "Dense Linear Algebra", "modified Gram-Schmidt / QR; growing orthogonalised-column front (kernel/D1_visible_dense_linear_algebra/kernel_qr_v2.c)"),
    ("kernel_attention_v2", "Dense Linear Algebra", "scaled dot-product attention (QK^T, row softmax, *V); transformer core (kernel/D1_visible_dense_linear_algebra/kernel_attention_v2.c)"),
    ("kernel_conv_v2", "Dense Linear Algebra", "2D convolution / CNN layer; overlapping-window MAC, feature-map rewrite (kernel/D1_visible_dense_linear_algebra/kernel_conv_v2.c)"),
    ("kernel_ntt_v2", "Spectral Methods", "multi-limb number-theoretic transform (modular butterfly); CKKS/lattice core, no crypto (kernel/D3_visible_spectral_methods/kernel_ntt_v2.c)"),
    ("kernel_dct_v2", "Spectral Methods", "blocked 8x8 2D DCT-II (JPEG transform); many small block rewrites (kernel/D3_visible_spectral_methods/kernel_dct_v2.c)"),
    ("kernel_dwt_v2", "Spectral Methods", "multi-level 2D Haar wavelet; shrinking multi-resolution pyramid (kernel/D3_visible_spectral_methods/kernel_dwt_v2.c)"),
    ("kernel_fft2d_v2", "Spectral Methods", "2D FFT (row FFTs, transpose, column FFTs); transpose scatter (kernel/D3_visible_spectral_methods/kernel_fft2d_v2.c)"),
    ("kernel_barnes_hut_v2", "N-Body Methods", "Barnes-Hut quadtree N-body; tree rebuilt each step + particle integrate (kernel/D4_visible_nbody_methods/kernel_barnes_hut_v2.c)"),
    ("kernel_md_lj_v2", "N-Body Methods", "Lennard-Jones molecular dynamics; cell list rebuilt each step + velocity-Verlet integrate (kernel/D4_visible_nbody_methods/kernel_md_lj_v2.c)"),
    ("kernel_pic_v2", "N-Body Methods", "electrostatic particle-in-cell; CIC scatter/gather + Jacobi Poisson solve on a grid (kernel/D4_visible_nbody_methods/kernel_pic_v2.c)"),
    ("kernel_fmm_v2", "N-Body Methods", "single-level fast multipole; per-box complex expansion coefficients + far eval (kernel/D4_visible_nbody_methods/kernel_fmm_v2.c)"),
    ("kernel_sph_v2", "N-Body Methods", "smoothed-particle hydrodynamics; two-pass neighbour sum with per-particle density/pressure fields (kernel/D4_visible_nbody_methods/kernel_sph_v2.c)"),
    ("kernel_lbm_v2", "Structured Grids", "Lattice-Boltzmann D2Q9; 9 distribution arrays streamed + BGK collide each step (kernel/D5_visible_structured_grids/kernel_lbm_v2.c)"),
    ("kernel_fdtd_v2", "Structured Grids", "2D FDTD electromagnetics; coupled E/H field grids in Yee leapfrog (kernel/D5_visible_structured_grids/kernel_fdtd_v2.c)"),
    ("kernel_floyd_v2", "Dynamic Programming", "Floyd-Warshall all-pairs shortest paths; whole matrix relaxed n times (kernel/D10_visible_dynamic_programming/kernel_floyd_v2.c)"),
    ("kernel_matrixchain_v2", "Dynamic Programming", "matrix-chain optimal parenthesisation; anti-diagonal fill, O(n^3) (kernel/D10_visible_dynamic_programming/kernel_matrixchain_v2.c)"),
    ("kernel_knapsack_v2", "Dynamic Programming", "0/1 knapsack, space-optimised 1D rolling capacity array (kernel/D10_visible_dynamic_programming/kernel_knapsack_v2.c)"),
    ("kernel_smithwaterman_v2", "Dynamic Programming", "Smith-Waterman local alignment; wavefront fill + traceback (kernel/D10_visible_dynamic_programming/kernel_smithwaterman_v2.c)"),
    ("kernel_beliefprop_v2", "Graphical Models", "loopy sum-product belief propagation on a grid MRF; iterated message arrays (kernel/D12_visible_graphical_models/kernel_beliefprop_v2.c)"),
    ("kernel_kalman_v2", "Graphical Models", "ensemble of Kalman filters; small dense covariance matrices updated per step (kernel/D12_visible_graphical_models/kernel_kalman_v2.c)"),
    ("kernel_gibbs_v2", "Graphical Models", "Gibbs sampling on a Potts/Ising grid; stochastic per-cell resample sweep (kernel/D12_visible_graphical_models/kernel_gibbs_v2.c)"),
    ("kernel_ldpc_v2", "Graphical Models", "LDPC min-sum decoder; bipartite Tanner-graph message passing (kernel/D12_visible_graphical_models/kernel_ldpc_v2.c)"),
    ("kernel_spmm_v2", "Sparse Linear Algebra", "SpMM: sparse x dense -> dense output; the GNN aggregation kernel (kernel/D2_visible_sparse_linear_algebra/kernel_spmm_v2.c)"),
    ("kernel_sparse_cholesky_v2", "Sparse Linear Algebra", "banded sparse Cholesky; factor fills in within the band (kernel/D2_visible_sparse_linear_algebra/kernel_sparse_cholesky_v2.c)"),
    ("kernel_spgemm_v2", "Sparse Linear Algebra", "SpGEMM: sparse x sparse -> new sparse matrix, fill-in (kernel/D2_visible_sparse_linear_algebra/kernel_spgemm_v2.c)"),
    ("kernel_sddmm_v2", "Sparse Linear Algebra", "SDDMM: sampled dense-dense -> sparse output at mask positions (kernel/D2_visible_sparse_linear_algebra/kernel_sddmm_v2.c)"),
    ("kernel_moe_dispatch_v2", "Sparse Linear Algebra", "MoE dispatch: token-permutation scatter into expert buffers + combine (kernel/D2_visible_sparse_linear_algebra/kernel_moe_dispatch_v2.c)"),
    ("kernel_fem_assembly_v2", "Unstructured Grids", "FEM stiffness assembly: scatter-add element matrices into a global matrix (kernel/D6_visible_unstructured_grids/kernel_fem_assembly_v2.c)"),
    ("kernel_fem_matvec_v2", "Unstructured Grids", "matrix-free FEM matvec: element gather-apply-scatter into a result vector (kernel/D6_visible_unstructured_grids/kernel_fem_matvec_v2.c)"),
    ("kernel_dg_v2", "Unstructured Grids", "discontinuous Galerkin step: per-element dense volume + face-flux coupling (kernel/D6_visible_unstructured_grids/kernel_dg_v2.c)"),
    ("kernel_mesh_smooth_v2", "Unstructured Grids", "unstructured Laplacian mesh smoothing over an adjacency list (kernel/D6_visible_unstructured_grids/kernel_mesh_smooth_v2.c)"),
    ("kernel_unstructured_fv_v2", "Unstructured Grids", "finite-volume: conservative face-flux scatter-add into cells (kernel/D6_visible_unstructured_grids/kernel_unstructured_fv_v2.c)"),
  ]},
]
N_WORKLOADS = sum(len(f["workloads"]) for f in FAMILIES)

INTRO = [
 "The corpus has TWO orthogonal divisions of the SAME workloads. (1) The behaviour FAMILIES, "
 "organised by MEMORY SIGNATURE (what the write-signal actually clusters), kept as finalised: "
 "IDLE -- near-zero writes (CPU is its warm/active boundary); MEM -- working-set writes (CACHE is a "
 "footprint/locality sub-family); IO -- page-cache + metadata writes (cold reads count here); "
 "THREAD -- shared-line + allocator writes; BULK-REWRITE / encryptor -- high-entropy full rewrites "
 "(the ransomware cluster); ENUMERATION / metadata -- scanner-like; STEALTH / trickle -- low-rate, "
 "high-intensity; APP; and MIXED. This is the 'which behaviour' division -- designed by signature, "
 "validated by cohesion. (2) This document is a SECOND, CROSS-CUTTING division by the Berkeley 13 "
 "dwarfs (Colella's seven, 2004, + Berkeley's six, A View from Berkeley, 2006) -- the 'which "
 "computation motif' division. Every workload keeps its family label AND gets a dwarf label where one "
 "applies; the two taxonomies coexist, they do not replace each other. A dwarf is an algorithmic "
 "method that captures a pattern of computation and communication -- largely a MEMORY-ACCESS pattern, "
 "which is what the host memory signal sees.",
 "We filter every dwarf by WRITE-visibility, because the signal only sees pages that are written. "
 "Visible / Visible++ = write-heavy, structured (a real signal). Irregular = visible writes with "
 "irregular access. Partial = part visible. Quiet = read/compute-bound -> near-idle (a CONTROL, the "
 "'is this motif invisible to host introspection?' null). Threats are labelled motifs, not a "
 "separate family: ransomware = Combinational Logic (encryption), scanner = Graph Traversal / FSM.",
 "Rules: a workload that already exists in another family is POINTED to (status 'exists', with its "
 "family), never duplicated. Each dwarf targets " + TARGET + " distinct workloads. v1 pre-fills only "
 "the existing pointers and leaves the gaps; the new workloads are chosen together, dwarf by dwarf, "
 "in iterations (edit the WORKLOADS lists in the generator and re-run).",
]

SOURCES = [
 ("A View from Berkeley (tech report)", "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf"),
 ("The 13 Motifs of Parallel Programming", "https://www.adrian.idv.hk/2010-10-01-13motifs/"),
 ("Reprising the 13 Dwarfs of OpenCL (HPCwire)", "https://www.hpcwire.com/2013/10/14/reprising-13-dwarfs-opencl/"),
]


# Per-workload implementation status. Anything not listed here is "exists" (built).
# States: planned -> under-development -> under-testing -> exists.
STATUS = {
    "kernel_stencil_jacobi_v2": "under-testing",
    "kernel_stencil_seidel_v2": "under-testing",
    "kernel_multigrid_v2": "under-testing",
    "kernel_fft_v2": "under-testing",
    "kernel_gemm_v2": "under-testing",
    "kernel_spmv_v2": "under-testing",
    "kernel_nbody_v2": "under-testing",
    "kernel_dp_v2": "under-testing",
    "kernel_hmm_v2": "under-testing",
    "kernel_lu_v2": "under-testing",
    "kernel_qr_v2": "under-testing",
    "kernel_attention_v2": "under-testing",
    "kernel_conv_v2": "under-testing",
    "kernel_ntt_v2": "under-testing",
    "kernel_dct_v2": "under-testing",
    "kernel_dwt_v2": "under-testing",
    "kernel_fft2d_v2": "under-testing",
    "kernel_barnes_hut_v2": "under-testing",
    "kernel_md_lj_v2": "under-testing",
    "kernel_pic_v2": "under-testing",
    "kernel_fmm_v2": "under-testing",
    "kernel_sph_v2": "under-testing",
    "kernel_lbm_v2": "under-testing",
    "kernel_fdtd_v2": "under-testing",
    "kernel_floyd_v2": "under-testing",
    "kernel_matrixchain_v2": "under-testing",
    "kernel_knapsack_v2": "under-testing",
    "kernel_smithwaterman_v2": "under-testing",
    "kernel_beliefprop_v2": "under-testing",
    "kernel_kalman_v2": "under-testing",
    "kernel_gibbs_v2": "under-testing",
    "kernel_ldpc_v2": "under-testing",
    "kernel_spmm_v2": "under-testing",
    "kernel_sparse_cholesky_v2": "under-testing",
    "kernel_spgemm_v2": "under-testing",
    "kernel_sddmm_v2": "under-testing",
    "kernel_moe_dispatch_v2": "under-testing",
    "kernel_fem_assembly_v2": "under-testing",
    "kernel_fem_matvec_v2": "under-testing",
    "kernel_dg_v2": "under-testing",
    "kernel_mesh_smooth_v2": "under-testing",
    "kernel_unstructured_fv_v2": "under-testing",
}
STATUS_ORDER = ["candidate", "covered", "planned", "under-development", "under-testing", "exists"]
STATUS_COLOR = {"candidate": "#8A6D9E", "covered": "#2E7D7D", "planned": "#8A8A8A",
                "under-development": "#3F6CA8", "under-testing": "#B9822A", "exists": "#2E7D52"}


def wstatus(name, fallback="exists"):
    return STATUS.get(name, fallback)


def have(d):
    # "have" = built (exists or under-testing), for the dwarf target tracking.
    return sum(1 for w in d["workloads"] if wstatus(w[0], w[1]) in ("exists", "under-testing"))


def status_cell(w):
    return wstatus(w[0], w[1])


def _row(w):
    # Normalise a dwarf workload row to (name, status, family-pointer, signature, used-in).
    # Older 4-tuples (no used-in) default the last cell to "".
    name, status, ptr, sig = w[0], w[1], w[2], w[3]
    used = w[4] if len(w) > 4 else ""
    return name, status, ptr, sig, used


# ---------------------------------------------------------------- markdown renderer
def _mdc(s):
    return s.replace("|", r"\|")


def render_md():
    out = ["# Workload Corpus -- two orthogonal divisions (behaviour families + the 13 dwarfs)", "",
           f"*First division: {N_WORKLOADS} workloads by memory-signature family. "
           f"Second division: the Berkeley 13 dwarfs. {DATE}.*", ""]
    for p in INTRO:
        out += [p, ""]
    # PART 1 -- first division: behaviour families (all built workloads)
    out += [f"## Part 1 -- First division: behaviour families (by signature) -- {N_WORKLOADS} workloads", "",
            "Every workload (built and planned), grouped by its memory-signature family. The Status "
            "column tracks implementation (planned -> under-development -> under-testing -> exists); the "
            "Dwarf column cross-references Part 2 (`--` = an access/IO/concurrency primitive, no motif).", ""]
    for f in FAMILIES:
        out += [f"### {f['id']} -- {f['name']}  ({f['sig']})", "", f["intro"], "",
                "| Workload | Status | Dwarf (Part 2) | Mechanism / note |", "|---|---|---|---|"]
        for name, dwarf, note in f["workloads"]:
            out.append(f"| {_mdc(name)} | {wstatus(name)} | {_mdc(dwarf)} | {_mdc(note)} |")
        out += [f"", f"*{f['name']}: {len(f['workloads'])} workloads.*", ""]
    allwl = [w[0] for f in FAMILIES for w in f["workloads"]]
    brk = {s: sum(1 for n in allwl if wstatus(n) == s) for s in STATUS_ORDER}
    out += [f"**First division total: {N_WORKLOADS} workloads across {len(FAMILIES)} signature families** "
            f"-- exists {brk['exists']}, under-testing {brk['under-testing']}, "
            f"under-development {brk['under-development']}, planned {brk['planned']}.", "",
            "*Status legend: candidate (violet, a real domain algorithm catalogued but not built) / "
            "planned (grey) -> under-development (blue) -> under-testing (gold) -> exists (green).*", ""]
    # PART 2 -- second division: the dwarfs
    out += ["## Part 2 -- Second division: the Berkeley 13 dwarfs", ""]
    out += ["## Coverage summary", "",
            "| Dwarf | Origin | Visibility | Maps to | Have | Target |",
            "|---|---|---|---|---|---|"]
    for d in DWARFS:
        out.append(f"| {d['id']} {_mdc(d['name'])} | {d['origin']} | {d['vis']} | {_mdc(d['maps'])} "
                   f"| {have(d)} | {TARGET} |")
    out.append("")
    n_cov = sum(1 for d in DWARFS if have(d) > 0)
    n_wl = sum(have(d) for d in DWARFS)
    out += [f"Covered (>=1 workload): **{n_cov}/13** dwarfs. Existing workloads pointed in: **{n_wl}**. "
            f"Empty dwarfs to fill: **{13 - n_cov}**.", ""]
    # per-dwarf
    for d in DWARFS:
        out.append(f"## {d['id']} -- {d['name']}  ({d['vis']})")
        out.append("")
        out.append(f"*{d['origin']}. Maps to: {d['maps']}. Example: {d['example']}.*")
        out.append("")
        out.append(d["what"]); out.append("")
        out.append(f"**Target {TARGET} workloads -- have {have(d)}.**")
        out.append("")
        out.append("| Workload / Algorithm | Status | Mechanism / points-to | Memory signature | Used in (real world) |")
        out.append("|---|---|---|---|---|")
        if d["workloads"]:
            for w in d["workloads"]:
                name, status, ptr, sig, used = _row(w)
                out.append(f"| {_mdc(name)} | {status_cell((name,status,ptr,sig))} | {_mdc(ptr)} "
                           f"| {_mdc(sig)} | {_mdc(used) if used else '--'} |")
        else:
            out.append("| _(to define together -- iteration pending)_ | planned | -- | -- | -- |")
        out.append("")
    out += ["## Sources", ""]
    for name, url in SOURCES:
        out.append(f"- [{name}]({url})")
    out.append("")
    (DOCS / "test_families_spec.md").write_text("\n".join(out))
    print(f"md  -> {DOCS / 'test_families_spec.md'}")


# ---------------------------------------------------------------- pdf renderer
def _esc(s):
    return _html.escape(s, quote=False)


VIS_COLOR = {"Visible": "#2E7D52", "Visible++": "#2E7D52", "Irregular": "#B9822A",
             "Partial": "#B9822A", "Quiet": "#8A8A8A", "Quiet / Visible": "#B9822A"}


def render_pdf():
    ss = getSampleStyleSheet()
    ink = colors.HexColor(INK); muted = colors.HexColor(MUTED)
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Times-Bold", fontSize=20,
                        textColor=ink, spaceAfter=4, leading=24)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=13.5,
                        textColor=ink, spaceBefore=14, spaceAfter=4, leading=16)
    LEAD = ParagraphStyle("LEAD", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5,
                          textColor=muted, spaceAfter=3)
    BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5,
                          leading=13.5, spaceAfter=6, textColor=ink)
    CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
    CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
    TH = ParagraphStyle("TH", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)

    story = [Paragraph("Memory-Signal Behaviour Detection &middot; Workload corpus", LEAD),
             Paragraph("Workload Corpus &mdash; Families + the 13 Dwarfs", H1),
             Paragraph(f"Two orthogonal divisions: {N_WORKLOADS} workloads by signature family, and the "
                       f"Berkeley 13 dwarfs &middot; {DATE}", LEAD),
             Spacer(1, 6)]
    for p in INTRO:
        story.append(Paragraph(_esc(p), BODY))

    avail = 7.0 * inch
    # PART 1 -- first division: behaviour families
    story.append(Paragraph(f"Part 1 &mdash; First division: behaviour families ({N_WORKLOADS} workloads)", H2))
    story.append(Paragraph("Every built workload by memory-signature family. The Dwarf column "
                           "cross-references Part 2 (-- = access / IO / concurrency primitive, no motif).", BODY))
    fcols = ["Workload", "Status", "Dwarf (Part 2)", "Mechanism / note"]
    fwidths = [1.7 * inch, 0.95 * inch, 1.5 * inch, 2.85 * inch]
    for f in FAMILIES:
        fhead = [Paragraph(f'{f["id"]} &mdash; {_esc(f["name"])} '
                           f'<font color="{MUTED}">({_esc(f["sig"])})</font>', H2),
                 Paragraph(_esc(f["intro"]), BODY)]
        fdata = [[Paragraph(_esc(c), TH) for c in fcols]]
        for name, dwarf, note in f["workloads"]:
            stt = wstatus(name)
            fdata.append([Paragraph(_esc(name), CELLB),
                          Paragraph(f'<font color="{STATUS_COLOR[stt]}">{_esc(stt)}</font>', CELL),
                          Paragraph(_esc(dwarf), CELL), Paragraph(_esc(note), CELL)])
        ftb = Table(fdata, colWidths=fwidths, repeatRows=1)
        ftb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ink),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF9F5")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
        story.append(KeepTogether(fhead)); story += [Spacer(1, 2), ftb, Spacer(1, 4)]
    story.append(Paragraph(f"<b>First division total: {N_WORKLOADS} workloads across {len(FAMILIES)} "
                           f"signature families.</b>", BODY))
    # PART 2 -- second division: the dwarfs
    story.append(Paragraph("Part 2 &mdash; Second division: the Berkeley 13 dwarfs", H2))
    # summary table
    story.append(Paragraph("Coverage summary", H2))
    cols = ["Dwarf", "Origin", "Visibility", "Maps to", "Have", "Target"]
    widths = [1.85 * inch, 0.95 * inch, 1.0 * inch, 1.6 * inch, 0.5 * inch, 0.6 * inch]
    data = [[Paragraph(_esc(c), TH) for c in cols]]
    for d in DWARFS:
        data.append([Paragraph(f'<b>{d["id"]}</b> {_esc(d["name"])}', CELL), Paragraph(_esc(d["origin"]), CELL),
                     Paragraph(_esc(d["vis"]), CELL), Paragraph(_esc(d["maps"]), CELL),
                     Paragraph(str(have(d)), CELLB), Paragraph(TARGET, CELL)])
    tb = Table(data, colWidths=widths, repeatRows=1)
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ink),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF9F5")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    story += [Spacer(1, 3), tb, Spacer(1, 6)]

    for d in DWARFS:
        head = [Paragraph(f'{d["id"]} &mdash; {_esc(d["name"])} '
                          f'<font color="{VIS_COLOR.get(d["vis"],"#8A8A8A")}">[{_esc(d["vis"])}]</font>', H2),
                Paragraph(f'<i>{_esc(d["origin"])}. Maps to: {_esc(d["maps"])}. Example: {_esc(d["example"])}.</i>', LEAD),
                Paragraph(_esc(d["what"]), BODY),
                Paragraph(f'<b>Target {TARGET} workloads &mdash; have {have(d)}.</b>', BODY)]
        cols = ["Workload / Algorithm", "Status", "Mechanism / points-to", "Memory signature", "Used in (real world)"]
        widths = [1.45 * inch, 0.6 * inch, 2.05 * inch, 1.35 * inch, 1.55 * inch]
        wdata = [[Paragraph(_esc(c), TH) for c in cols]]
        if d["workloads"]:
            for w in d["workloads"]:
                name, status, ptr, sig, used = _row(w)
                stt = status_cell((name, status, ptr, sig))
                wdata.append([Paragraph(_esc(name), CELLB),
                              Paragraph(f'<font color="{STATUS_COLOR[stt]}">{_esc(stt)}</font>', CELL),
                              Paragraph(_esc(ptr), CELL), Paragraph(_esc(sig), CELL),
                              Paragraph(_esc(used) if used else "--", CELL)])
        else:
            wdata.append([Paragraph("<i>(to define together &mdash; iteration pending)</i>", CELL),
                          Paragraph("planned", CELL), Paragraph("--", CELL), Paragraph("--", CELL),
                          Paragraph("--", CELL)])
        wt = Table(wdata, colWidths=widths, repeatRows=1)
        wt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ink),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF9F5")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
        story.append(KeepTogether(head))
        story += [Spacer(1, 2), wt, Spacer(1, 4)]

    story.append(Paragraph("Sources", H2))
    for name, url in SOURCES:
        story.append(Paragraph(f'<link href="{url}" color="#3F6CA8">{_esc(name)}</link>', BODY))

    def _footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(muted)
        canvas.drawString(0.75 * inch, 0.5 * inch, "Workload families -- the 13 dwarfs")
        canvas.drawRightString(7.75 * inch, 0.5 * inch, f"p. {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(DOCS / "test_families_spec.pdf"), pagesize=LETTER,
                      leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                      topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                      title="Workload Families -- the 13 Dwarfs").build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"pdf -> {DOCS / 'test_families_spec.pdf'}")


# ---------------------------------------------------------------- html renderer
# Plain-language, one-to-two-sentence description per workload (hover tooltip).
GLOSSARY = {
 "run_idle": "A do-nothing baseline: the guest simply sleeps, so the only memory activity is background operating-system noise. It defines the floor that active workloads are measured against.",
 "idle_long_baseline": "A long, uncontaminated idle window (optionally after dropping the page cache) used as the stable reference for a machine doing nothing.",
 "idle_post_workload_recovery": "An idle period that follows a real workload, capturing how the residual writeback and caches decay back to rest.",
 "cpu_hash_loop_v2": "A tight, register-resident hashing loop that is almost pure computation and barely touches memory; the control for whether pure CPU work is invisible to the memory signal.",
 "cpu_branch_random_v2": "A loop dominated by unpredictable, data-dependent branches that stresses the CPU branch predictor while writing almost nothing to memory.",
 "cpu_matrix_mult_v2": "A small dense matrix multiply: compute-bound, writing only a modest output, sitting at the quiet boundary between pure compute and memory work.",
 "kernel_spmv_v2": "Sparse matrix-vector multiply (y = A x). Its bulk is a large, read-only sparse matrix reached by indirect gather lookups, while only a tiny vector changes; a control for whether sparse/gather access is invisible to the write-signal.",
 "kernel_spmm_v2": "Sparse matrix times dense matrix (C = A B) producing a dense output. Unlike SpMV's tiny vector, the large dense output is a real write, so it is the visible face of sparse algebra. It is the aggregation kernel of graph neural networks (DGL, PyG).",
 "kernel_sparse_cholesky_v2": "A banded sparse Cholesky factorisation (A = L L^T): the factor gains new non-zeros (fill-in) inside the band, so the band buffer is progressively rewritten. The write-heavy heart of sparse direct solvers used in finite-element analysis, circuit simulation and optimisation.",
 "kernel_spgemm_v2": "Sparse matrix times sparse matrix producing a new sparse matrix, computed row by row with a dense accumulator. Writing the new (filled-in) sparse result is the signature write. It drives algebraic-multigrid setup, triangle counting and graph contraction.",
 "kernel_sddmm_v2": "Sampled dense-dense matrix multiplication: it computes full dense dot products but only at the non-zero positions of a sparse mask, writing a scattered sparse output. Used in graph-attention networks and recommender systems.",
 "kernel_moe_dispatch_v2": "The Mixture-of-Experts dispatch/combine step: each token's feature vector is scattered into its assigned expert's buffer (a counting-sort permutation) and later gathered back. The token-permutation scatter is the signature write. Used in modern MoE large language models.",
 "kernel_fem_assembly_v2": "Finite-element stiffness assembly: it computes each element's small local matrix and scatter-adds it into a large global matrix through the mesh connectivity. That indexed scatter-accumulate into a big matrix is the distinctive write. The first heavy step of structural and crash simulation (ANSYS/Abaqus).",
 "kernel_fem_matvec_v2": "A matrix-free finite-element matrix-vector product: instead of assembling the global matrix it visits each element, gathers the local solution, applies the element operator, and scatter-adds into a result vector. The quieter member of the dwarf (a vector output, the matrix-free analog of sparse matrix-vector multiply).",
 "kernel_dg_v2": "A discontinuous Galerkin time step: every element carries a small dense block of unknowns updated by an element-local dense operator (the volume term) plus a conservative flux exchange with neighbouring elements. It rewrites the whole per-element solution array each step. Used for high-order CFD and seismic simulation.",
 "kernel_mesh_smooth_v2": "Unstructured Laplacian mesh smoothing: each node is moved toward the average of its neighbours, reached through an irregular adjacency list, rewriting the node value array. The unstructured cousin of a grid stencil; used in mesh processing and remeshing.",
 "kernel_unstructured_fv_v2": "An unstructured finite-volume update: conserved quantities on cells are changed by fluxes across faces, each flux scatter-added equal-and-opposite into the two cells it separates (so the total is exactly conserved). The face-based conservative scatter is the distinctive write. The basis of OpenFOAM and industrial CFD.",
 "kernel_lu_v2": "In-place LU factorisation: it decomposes a matrix into lower/upper triangular factors to solve linear systems, overwriting the matrix as it goes so the active region shrinks toward the bottom-right. The workhorse behind circuit simulation, finite-element solvers and optimisation.",
 "kernel_qr_v2": "Gram-Schmidt / QR orthogonalisation: it rewrites each column to be orthogonal to all the previous ones, so the dependency front grows as it advances. Used in least-squares regression and the inner loop of iterative eigen/linear solvers such as GMRES and Arnoldi.",
 "kernel_attention_v2": "Scaled dot-product attention, the core of every transformer: it scores each token against every other (Q times K-transpose), normalises the scores with a row-wise softmax, then mixes the values. The single most-run compute pattern in modern AI.",
 "kernel_conv_v2": "A 2D convolution layer, the core of image and vision models: it slides a small learned filter across the input, multiplying and accumulating over each overlapping window to write an output feature map.",
 "kernel_ntt_v2": "The number-theoretic transform: an FFT-style butterfly over modular integers instead of complex numbers, run across several residue channels at once. It is the compute core of lattice cryptography and CKKS homomorphic encryption; here it is a pure polynomial-arithmetic benchmark with no keys or encryption.",
 "mem_workingset_sweep_v2": "Writes one byte to the start of every page across a buffer of a chosen size, isolating how the signal scales with the working-set footprint at a fixed write granularity.",
 "mem_writemag_sweep_v2": "Varies how many bytes are written per page, isolating the per-page change magnitude from the number of pages touched.",
 "mem_rmw_intensity_v2": "Compares pure-read, pure-write and read-modify-write access at a matched footprint and stride, isolating read-modify-write traffic.",
 "mem_pagefault_density_v2": "Separates the one-off cost of first-touching (faulting in) pages from steady-state re-touching, via fault-only, touch-only and mixed variants.",
 "mem_mmap_traversal_v2": "Traverses a memory-mapped file (read, write, or read-modify-write), exercising page-cache and writeback behaviour rather than anonymous memory.",
 "mem_random_write_pages_v2": "Writes a few bytes to randomly-chosen pages across a large buffer, producing a high-rate, broadband, low-locality write pattern.",
 "mem_stride_sweep_large_v2": "Sweeps a large buffer at a configurable stride (page, TLB, or irregular), exposing how access stride and footprint shape the signal.",
 "cache_hot_loop_v2": "Repeatedly read-modify-writes a tiny buffer that fits in the L1 cache, so the work stays in cache and rarely reaches main memory; a control for the cache-versus-memory observability gap.",
 "cache_cold_scan_v2": "Linearly scans and rewrites a buffer larger than the last-level cache, generating maximum, sequential main-memory traffic.",
 "cache_stride_sweep_v2": "Walks a buffer larger than the cache at a chosen stride, probing how stride interacts with cache and TLB behaviour.",
 "io_read_cache_hit_v2": "Pre-warms a file into the page cache then reads it back at random offsets, so almost nothing is written; a quiet-I/O control with a high syscall rate.",
 "io_direct_write_like_v2": "Writes to a file with O_DIRECT, bypassing the page cache so activity is dominated by block-layer and writeback work rather than dirtied cache pages.",
 "thread_lock_contention_v2": "Many threads contend for a single shared mutex, writing one shared cache line and stressing the kernel futex and scheduler with inter-core ping-pong.",
 "thread_producer_consumer_v2": "One producer and one consumer exchange items through a bounded ring buffer and condition variable, stressing queue synchronisation and cross-core traffic.",
 "thread_parallel_alloc_v2": "Multiple threads each churn through their own allocations and frees, stressing the memory allocator and creating cross-thread interference.",
 "sandbox_ransom_seq": "A SAFE simulation of a file-processing pipeline that, per file, discovers, reads, applies a reversible XOR, writes and renames, one file at a time. No real encryption; disposable sandbox files only.",
 "sandbox_ransom_batched": "The same SAFE five-stage file-processing pattern, but batched: all files discovered, then all read, then all transformed, and so on.",
 "sandbox_ransom_slowburn": "The same SAFE pattern run slowly, one file every few seconds, to imitate a low-and-slow processing cadence.",
 "sandbox_ransom_selective": "The same SAFE pattern applied to only a selected subset of files (e.g. one extension), isolating the cost of the discovery/filter stage.",
 "sandbox_ransom_* (4 variants)": "The four SAFE file-processing-pipeline simulations (sequential, batched, slow-burn, selective): reversible XOR on disposable sandbox files, no real encryption.",
 "sandbox_scanner_metadata": "A SAFE simulation that only enumerates files and reads their metadata (stat/readdir) without reading or modifying contents; a pure directory-walk pattern.",
 "sandbox_stealth_microwrite": "A SAFE test workload that makes tiny (a few bytes) writes per file at a high syscall rate but very low total volume; a low-footprint, high-frequency pattern.",
 "sandbox_stealth_scattered": "A SAFE test workload that writes to scattered random offsets within files, an anti-sequential, spatially-diffuse pattern.",
 "sandbox_stealth_paced": "A SAFE test workload that paces its small writes with irregular, jittered timing between files.",
 "app_hashtable_intensive_v2": "Builds and probes a large open-addressing hash table: a write-heavy build phase followed by a read-heavy lookup phase, a realistic data-structure workload.",
 "app_sqlite_oltp_v2": "Runs an online-transaction-processing rhythm against SQLite (inserts/updates with write-ahead-log appends and checkpoints), a realistic database write pattern.",
 "app_sqlite_analytical_v2": "Runs read-heavy analytical queries (counts, sums, aggregates) against SQLite, with occasional temporary-table write bursts.",
 "app_compress_gzip_v2": "Continuously compresses near-incompressible data with gzip, a steady mix of CPU and I/O whose output is high-entropy.",
 "app_decompress_gzip_v2": "The inverse of compression: a small compressed input expands into large output, inverting the read/write balance.",
 "app_json_parse_v2": "Streams and parses JSON line by line, an allocation-heavy, parser-style workload.",
 "mixed_mem_io_v2": "Runs memory writes and file I/O at the same time, putting concurrent pressure on both subsystems.",
 "mixed_cpu_mem_v2": "Interleaves compute-heavy and memory-heavy work, mixing the two pressures.",
 "mixed_cpu_io_v2": "Runs a CPU loop and file I/O concurrently, stressing the scheduler and I/O path under compute load.",
 "kernel_stencil_jacobi_v2": "A 2-D Jacobi stencil: every grid cell is repeatedly replaced by the average of its four neighbours using two buffers swapped each pass; the classic iterative-grid (PDE / image-smoothing) pattern.",
 "kernel_stencil_seidel_v2": "A 2-D Gauss-Seidel stencil with red-black ordering, updating the grid in place in two checkerboard sweeps; a faster-converging, single-buffer variant of Jacobi.",
 "kernel_multigrid_v2": "A multigrid solver that smooths a grid at several resolutions (fine to coarse to fine, a V-cycle), so the active region changes size and migrates across grid levels over time.",
 "kernel_fft_v2": "An in-place Fast Fourier Transform: it reorders the data (bit-reversal) then applies log-N butterfly stages whose stride doubles each stage; the core of signal and spectral processing.",
 "kernel_gemm_v2": "A blocked dense matrix multiply (C = A B), the fundamental linear-algebra and machine-learning kernel, writing a large output matrix each pass.",
 "kernel_nbody_v2": "A particle simulation in which each particle is pushed by the gravity of a sample of others and then moved, with the particle arrays rewritten smoothly every timestep.",
 "kernel_dp_v2": "A dynamic-programming edit-distance computation that fills a large table row by row, each cell derived from its neighbours; a monotone wavefront sweeping across the table.",
 "kernel_floyd_v2": "Floyd-Warshall all-pairs shortest paths: for each intermediate node k it relaxes every pair (i,j), rewriting the whole n-by-n distance matrix n times. The repeated full-matrix relaxation is a distinct write pattern from a one-time table fill. Used for routing tables and transitive closure.",
 "kernel_matrixchain_v2": "Matrix-chain multiplication: it finds the cheapest way to parenthesise a chain of matrix products by filling a cost table along its anti-diagonals (by increasing sub-chain length), with an O(n) split search per cell. Used in query-plan and compiler optimisation.",
 "kernel_knapsack_v2": "The 0/1 knapsack dynamic program in its space-optimised form: a single capacity vector is repainted from high capacity down to low once per item, so the whole solution lives in one rolling 1D array. Used in resource allocation, scheduling and budgeting.",
 "kernel_smithwaterman_v2": "Smith-Waterman local sequence alignment: it fills a scoring matrix as a wavefront (clamping negatives to zero), tracks the best cell, then traces the optimal local alignment backward to a zero. The gold-standard local-alignment method in genomics.",
 "kernel_hmm_v2": "The forward algorithm over a Hidden Markov Model: it fills a probability trellis column by column, each column a dense matrix-vector step, rescaled to stay a valid probability distribution.",
 "kernel_beliefprop_v2": "Loopy belief propagation on a grid Markov random field: every cell keeps a small message vector to each neighbour and these messages are recomputed and passed around, iteration after iteration, until the beliefs settle. Used for stereo vision, image denoising and other MRF/CRF inference.",
 "kernel_kalman_v2": "An ensemble of Kalman filters: each tracks a moving system by alternating a prediction step and a measurement-correction step over a small state vector and covariance matrix, and many run side by side. The covariance matrices are rewritten every timestep. The backbone of GPS/INS navigation and object tracking.",
 "kernel_gibbs_v2": "Gibbs sampling on a Potts/Ising grid: it sweeps the grid and resamples each cell's state from the local distribution implied by its neighbours, so the whole grid is stochastically repainted each pass. A staple of Bayesian inference and statistical physics.",
 "kernel_ldpc_v2": "An LDPC error-correcting decoder: it passes belief messages back and forth across a bipartite Tanner graph of bits and parity checks until the received word is corrected to a valid codeword. Used in 5G, WiFi, satellite links and SSD storage.",
 # --- D1 Dense LA candidates ---
 "Cholesky factorisation": "Factors a symmetric positive-definite matrix into a lower triangle times its transpose, used to solve linear systems and sample correlated variables. The active region shrinks toward the bottom-right as it proceeds, like LU but triangular.",
 "Triangular solve (TRSM)": "Solves a triangular system (L x = b) by forward or back substitution, computing one unknown at a time so a wavefront sweeps along the vector. It is the solve step that follows an LU or Cholesky factorisation.",
 # --- D2 Sparse LA candidates ---
 "PageRank": "Google's original web-ranking algorithm: it repeatedly spreads each page's score along its outgoing links until the scores settle, which is a sparse matrix-vector multiply applied over and over. Heavily used, but gather-dominated so nearly invisible to a write-signal.",
 "Conjugate Gradient (CG)": "The standard iterative solver for large symmetric-positive-definite linear systems (the backbone of finite-element and CFD solvers). Each iteration is one sparse matrix-vector product plus a few vector updates.",
 "Sparse triangular solve (SpTRSV)": "Solves a sparse triangular system where each unknown depends on earlier ones, forcing a partly serial dependency order. It is the bottleneck inside incomplete-LU preconditioners and sparse direct solvers.",
 "Sparse matrix-matrix (SpGEMM)": "Multiplies two sparse matrices, first working out where the non-zeros land and then computing them. It powers algebraic multigrid, graph contraction and triangle counting.",
 # --- D3 Spectral candidates ---
 "Discrete Cosine Transform (DCT)": "A real-valued cousin of the FFT that concentrates a signal's energy into a few coefficients, applied in small blocks. It is the heart of JPEG image and MPEG/H.264 video compression.",
 "kernel_dct_v2": "A blocked discrete cosine transform: it splits data into small 8x8 blocks and expresses each as a sum of cosine patterns, packing most of the energy into a few coefficients. The core of JPEG image and MPEG/H.264 video compression.",
 "kernel_dwt_v2": "A discrete wavelet transform: it repeatedly splits a signal into coarse and detail halves with a small filter and a downsample, forming a multi-resolution pyramid whose active footprint halves at each level. Used in JPEG2000, denoising and compression.",
 "kernel_fft2d_v2": "A two-dimensional FFT: it transforms every row, transposes the array, then transforms every column. The transpose (a large strided scatter) is its distinctive memory access. Used in image and optical filtering, crystallography and turbulence simulation.",
 "kernel_barnes_hut_v2": "A Barnes-Hut N-body simulation: it builds a spatial quadtree each step so a whole cluster of distant particles can be treated as one, cutting the cost from O(n^2) to O(n log n). Beyond moving the particles it rewrites the tree every step -- the distinctive extra write. The workhorse of cosmological galaxy simulation.",
 "kernel_md_lj_v2": "A molecular-dynamics simulation of atoms under the Lennard-Jones potential, using cell lists so each atom only checks nearby cells. It periodically rebuilds those lists (the distinctive extra write) on top of moving the atoms. The core loop of GROMACS/NAMD/AMBER for drug discovery and materials.",
 "kernel_pic_v2": "A particle-in-cell plasma simulation: it scatter-deposits particle charge onto a grid, solves the field on that grid, then gathers the force back to the particles. The particle-to-grid scatter and grid rewrite are its distinctive access. Used in plasma physics, accelerators and semiconductor device simulation.",
 "kernel_fmm_v2": "A fast multipole N-body method: it summarises clusters of particles by compact multipole and local expansions on a tree, reaching O(n) cost. It writes those expansion-coefficient arrays each step (the distinctive extra write) as well as moving the particles. Used for electrostatics and acoustics.",
 "kernel_sph_v2": "Smoothed-particle hydrodynamics, a mesh-free fluid simulation where each particle carries fluid properties updated from a smoothing kernel over its neighbours. It writes extra per-particle fields (density, pressure) beyond position and velocity. Used for liquid effects in film VFX and for astrophysical gas.",
 "DTFT / direct DFT": "The discrete-time Fourier transform, in its computable form, is the DFT -- evaluated fast as an FFT, or naively as a dense matrix-vector multiply. It carries no memory pattern beyond the FFT, so the FFT test covers it.",
 "Cepstrum": "The inverse FFT of the log-magnitude spectrum of a signal, used to separate pitch from timbre. Computationally it is two FFT passes with a pointwise log between them, so the FFT test covers its pattern.",
 "MFCC": "Mel-frequency cepstral coefficients, the classic speech feature: an FFT, a mel filterbank, a log, and a DCT. Its memory pattern is exactly an FFT followed by a blocked cosine transform, both of which are built.",
 "STFT / spectrogram": "The short-time Fourier transform slides a window along a signal and takes an FFT of each frame, filling a time-frequency spectrogram column by column. The front-end of nearly all audio machine learning; its pattern is the FFT, repeated.",
 "Wavelet scattering (WST)": "A wavelet scattering transform cascades wavelet transforms with a modulus and averaging between them to build stable, translation-invariant features. Its memory pattern is the wavelet pyramid applied repeatedly, so the DWT test covers it.",
 "FFT convolution": "Convolves two large signals quickly by transforming both with an FFT, multiplying point by point, then transforming back. Used for large-kernel filtering and for multiplying very large polynomials or integers.",
 "2D FFT": "A two-dimensional Fourier transform done as a pass of row transforms followed by a pass of column transforms (with a transpose between). Used in image and optical filtering, crystallography and medical imaging.",
 # --- D4 N-Body candidates ---
 "All-pairs direct N-body": "Computes the exact force between every pair of particles each timestep (order n-squared work) and then moves them. The accurate but expensive reference method for small particle counts.",
 "Barnes-Hut tree": "Speeds up an N-body simulation by grouping distant particles in a spatial tree and treating each group as one mass, cutting the cost to order n-log-n. The workhorse of cosmological galaxy simulations.",
 "Fast Multipole Method (FMM)": "An order-n N-body method that represents clusters of particles by compact multipole and local expansions, achieving linear scaling. Used for electrostatics, acoustics and other long-range force problems.",
 "Molecular dynamics (Lennard-Jones)": "Simulates atoms interacting through a short-range potential, using neighbour or cell lists so each atom only checks nearby ones, then integrating their motion. The core of GROMACS/NAMD/AMBER for drug discovery and materials.",
 "Smoothed-particle hydrodynamics (SPH)": "A mesh-free fluid simulation where each particle carries fluid properties and is updated from a smoothing kernel over its neighbours. Widely used for liquid effects in film VFX and for astrophysical gas.",
 # --- D5 Structured Grid candidates ---
 "Lattice-Boltzmann (LBM)": "Models fluid flow by streaming and colliding particle-distribution values on a regular lattice, rewriting several distribution arrays each step. Popular for flow in complex or porous geometries.",
 "FDTD electromagnetics": "Finite-Difference Time-Domain: it leapfrogs the electric and magnetic fields on interleaved grids, updating each from the curl of the other. The standard method for antenna, radar and photonics design.",
 "kernel_lbm_v2": "A Lattice-Boltzmann fluid solver (D2Q9): every cell keeps nine particle-distribution values that are streamed to neighbour cells and then relaxed toward equilibrium each step. The wide, nine-array streaming grid write sets it apart from the single-field stencils. Used for computational fluid dynamics in complex geometries.",
 "kernel_fdtd_v2": "A 2D Finite-Difference Time-Domain electromagnetics solver: it advances the electric field Ez and the magnetic fields Hx, Hy in leapfrog, each computed from the curl of the other. The coupled two-grid, E-then-H write distinguishes it from the single-field relaxation stencils. Used for antenna, radar and photonics simulation.",
 # --- D6 Unstructured Grid candidates ---
 "FEM stiffness assembly": "Builds the global system for a finite-element model by computing each element's small matrix and scattering it into a large sparse matrix via the mesh connectivity. The first heavy step in structural and crash simulation (ANSYS/Abaqus).",
 "Unstructured finite-volume": "Solves conservation laws on an irregular mesh by gathering fluxes across each cell's faces (reached through face lists) and updating the cell value. The basis of OpenFOAM and most industrial CFD.",
 "Mesh Laplacian smoothing": "Improves a mesh by moving each vertex toward the average of its connected neighbours, reached through an irregular adjacency list. A common geometry-processing and remeshing operation.",
 "Discontinuous Galerkin (DG)": "A high-order method that keeps a small independent polynomial solution inside each element and couples elements only through their shared faces. Used for high-accuracy CFD and seismic wave propagation.",
 "Mesh partitioning (METIS)": "Splits a large mesh or graph into balanced pieces with few connections cut, so the work can be divided across processors. A standard preprocessing step for parallel finite-element runs.",
 # --- D7 MapReduce / Monte Carlo candidates ---
 "Monte-Carlo integration": "Estimates an integral or expected value by drawing many random samples and averaging them, with error shrinking as more samples are added. Ubiquitous in physics, finance and Bayesian statistics.",
 "Monte-Carlo option pricing": "Prices a financial derivative by simulating many random price paths for the underlying asset and averaging the discounted payoff. A core quantitative-finance method for derivatives, value-at-risk and risk.",
 "MCMC (Metropolis-Hastings)": "Samples from a complicated probability distribution by taking a guided random walk that accepts or rejects proposed moves. The engine behind Bayesian inference tools such as Stan and PyMC.",
 "Histogram / word-count": "The canonical map-reduce job: scan a large stream and tally counts into bins or a hash map. The textbook example of large-scale data processing on Spark/Hadoop.",
 "Path tracing": "Renders a photorealistic image by shooting many random light rays per pixel and accumulating their contributions into an image buffer. The basis of modern film and game rendering (RenderMan, Blender Cycles).",
 # --- D8 Combinational candidates ---
 "SHA-256": "A cryptographic hash that turns any input into a fixed 256-bit fingerprint by streaming it through a fixed compression function. Used in git, blockchains, TLS certificates, deduplication and integrity checks.",
 "AES block cipher": "The standard symmetric cipher: it transforms data in fixed blocks through several substitution-and-permutation rounds, producing high-entropy output. Used in HTTPS/TLS, disk encryption and VPNs.",
 "CRC32": "A fast 32-bit checksum that detects accidental data corruption using a table-driven rolling computation. Built into Ethernet, ZIP files and storage error detection.",
 # --- D9 Graph Traversal candidates ---
 "Breadth-first search (BFS)": "Explores a graph level by level from a start node, expanding a frontier and marking visited nodes. The canonical graph workload (the Graph500 benchmark) and the basis of shortest-unweighted-path and garbage-collection marking.",
 "Depth-first search (DFS)": "Explores a graph by going as deep as possible before backtracking, using an explicit or recursive stack. Underlies topological sorting, cycle detection and dependency resolution in package managers.",
 "Dijkstra / A* shortest path": "Finds the cheapest route through a weighted graph by repeatedly expanding the nearest unvisited node from a priority queue (A* adds a goal-directed heuristic). The core of GPS navigation, network routing and game pathfinding.",
 "Connected components": "Groups a graph's nodes into clusters that are reachable from one another, using union-find or label propagation. Used for clustering, image segmentation and fraud-ring detection.",
 # --- D10 Dynamic Programming candidates ---
 "Smith-Waterman": "Finds the best-matching local region between two biological sequences by filling a scoring matrix along its anti-diagonals and tracing back. The gold-standard local-alignment method in genomics.",
 "Needleman-Wunsch": "Aligns two sequences end to end by filling a scoring table where each cell comes from its top, left and diagonal neighbours. The classic global DNA and protein alignment algorithm.",
 "Viterbi decoding": "Finds the single most likely sequence of hidden states by filling a trellis column by column, keeping the best path into each state. Used in speech recognition, error-correction decoding and part-of-speech tagging.",
 "Floyd-Warshall": "Computes shortest paths between every pair of nodes by repeatedly allowing one more intermediate node, rewriting an n-by-n distance matrix in each pass. Used for routing tables and transitive closure.",
 "Knapsack / subset-sum": "Chooses items to maximise value under a capacity limit by filling a table indexed by remaining capacity. A textbook dynamic program behind resource-allocation and scheduling problems.",
 # --- D11 Backtrack / B&B candidates ---
 "N-queens": "Places queens on a chessboard so none attack each other by trying positions and backtracking on conflict. A classic constraint-satisfaction benchmark with a tiny working set.",
 "Sudoku solver": "Fills a Sudoku grid by propagating constraints and backtracking when a cell has no legal value. A compact illustration of constraint propagation plus search.",
 "DPLL / CDCL SAT": "Decides whether a Boolean formula can be satisfied by assigning variables, propagating forced consequences, learning from conflicts and backtracking. The engine inside hardware verification and modern solvers such as MiniSat and Z3.",
 "Branch-and-bound MILP": "Solves integer optimisation problems by exploring a tree of sub-problems and pruning branches that cannot beat the best solution so far, using a relaxed bound. The basis of commercial optimisers (Gurobi, CPLEX) for logistics and scheduling.",
 "TSP branch-and-bound": "Searches for the shortest tour visiting all cities by branching on choices and pruning with a lower-bound estimate. A canonical operations-research and routing problem.",
 # --- D12 Graphical Models candidates ---
 "Kalman filter": "Tracks a moving system by alternating a prediction step and a measurement-correction step over small state and covariance matrices. It runs in essentially every GPS, drone, robot and sensor-fusion system.",
 "LDPC belief propagation": "Decodes error-correcting codes by passing probability messages back and forth on a bipartite graph until they agree. Used in 5G, WiFi, satellite links and SSD storage.",
 "Loopy belief propagation": "Approximate inference on a graph with cycles by iterating local probability messages over a grid until they stabilise. Used for stereo vision, image denoising and Markov-random-field models.",
 "Gibbs sampling": "Draws samples from a complex joint distribution by repeatedly resampling one variable at a time given the others. A staple of Bayesian inference and topic models such as LDA.",
 # --- D13 FSM candidates ---
 "Regex / DFA matcher": "Matches text against a pattern by walking a finite-state machine one character at a time through a transition table. Underlies grep, input validation and log processing.",
 "Aho-Corasick": "Searches a stream for many patterns at once using a single automaton with success and failure links. The matching core of antivirus and intrusion-detection scanners (ClamAV, Snort).",
 "HTTP / protocol parser": "Decodes a byte stream into structured messages by stepping a small state machine through the protocol's grammar. Found in web servers, TCP/IP stacks and deep-packet inspection.",
 "Lexer / tokenizer": "Turns raw source characters into tokens by recognising character classes with a finite-state machine. The first stage of every compiler and interpreter.",
}


def _wl(name):
    g = GLOSSARY.get(name)
    if g:
        return f"<span class='gl' data-def=\"{_html.escape(g, quote=True)}\">{_esc(name)}</span>"
    return _esc(name)


HTML_CSS = """
body{margin:0;background:#E8E6DF;color:#1A1A1A;line-height:1.55;
 font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-size:15px}
.page{max-width:1000px;margin:24px auto;background:#fff;padding:38px 50px 60px;border-radius:5px;
 box-shadow:0 2px 18px rgba(0,0,0,.12)}
h1{font-size:25px;margin:0 0 4px} .sub{color:#5B5B5B;margin:0 0 14px;font-size:14px}
h2{font-size:20px;border-bottom:2px solid #D8D6CE;padding-bottom:5px;margin:30px 0 8px}
h3{font-size:15px;margin:18px 0 4px} .sig,.vis{color:#5B5B5B;font-weight:400;font-size:13px}
p{max-width:84ch} .intro{color:#5B5B5B;font-size:13px;margin:3px 0 8px}
table{border-collapse:collapse;width:100%;margin:6px 0 14px;font-size:13px}
th,td{border:1px solid #E2DFD6;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#F5F3EE} tr:nth-child(even) td{background:#FAF9F5}
td.wl{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:nowrap}
.badge{display:inline-block;color:#fff;font-size:11px;font-weight:700;padding:1px 8px;border-radius:20px}
.legend .badge{margin-right:5px} .total{margin:8px 0} a{color:#3F6CA8}
.gl{cursor:help;border-bottom:1px dotted #B9822A;position:relative}
.gl:hover::after{content:attr(data-def);position:absolute;left:0;top:140%;z-index:60;width:max-content;
 max-width:360px;background:#0d0f12;color:#fff;border:1px solid #333;border-radius:6px;padding:9px 12px;
 font-size:12px;font-weight:400;line-height:1.5;white-space:normal;
 font-family:system-ui,-apple-system,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.5)}
.gl:hover{color:#000}
"""


def _badge(s):
    return f"<span class='badge' style='background:{STATUS_COLOR.get(s,'#8A8A8A')}'>{s}</span>"


def render_html():
    o = ["<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width, initial-scale=1'>",
         "<title>Workload Corpus -- families + dwarfs</title>",
         f"<style>{HTML_CSS}</style></head><body><div class='page'>",
         "<h1>Workload Corpus &mdash; Families + the 13 Dwarfs</h1>",
         f"<p class='sub'>First division: {N_WORKLOADS} workloads by signature family. "
         f"Second division: the Berkeley 13 dwarfs. {DATE}.</p>"]
    for p in INTRO:
        o.append(f"<p>{_esc(p)}</p>")
    o.append(f"<h2>Part 1 &mdash; First division: behaviour families ({N_WORKLOADS} workloads)</h2>")
    o.append("<p class='legend'>Status: " + " ".join(_badge(s) for s in STATUS_ORDER) +
             " &nbsp;&middot;&nbsp; <span style='color:#5B5B5B;font-size:13px'>hover any workload "
             "name for a plain-language description</span></p>")
    for f in FAMILIES:
        o.append(f"<h3>{f['id']} &mdash; {_esc(f['name'])} <span class='sig'>({_esc(f['sig'])})</span></h3>")
        o.append(f"<p class='intro'>{_esc(f['intro'])}</p>")
        o.append("<table><thead><tr><th>Workload</th><th>Status</th><th>Dwarf</th>"
                 "<th>Mechanism / note</th></tr></thead><tbody>")
        for name, dwarf, note in f["workloads"]:
            o.append(f"<tr><td class='wl'>{_wl(name)}</td><td>{_badge(wstatus(name))}</td>"
                     f"<td>{_esc(dwarf)}</td><td>{_esc(note)}</td></tr>")
        o.append("</tbody></table>")
    allwl = [w[0] for f in FAMILIES for w in f["workloads"]]
    brk = {s: sum(1 for n in allwl if wstatus(n) == s) for s in STATUS_ORDER}
    o.append(f"<p class='total'><b>First division total: {N_WORKLOADS} workloads across "
             f"{len(FAMILIES)} signature families</b> &mdash; exists {brk['exists']}, under-testing "
             f"{brk['under-testing']}, under-development {brk['under-development']}, planned {brk['planned']}.</p>")
    o.append("<h2>Part 2 &mdash; Second division: the Berkeley 13 dwarfs</h2>")
    o.append("<h3>Coverage summary</h3>")
    o.append("<table><thead><tr><th>Dwarf</th><th>Origin</th><th>Visibility</th><th>Maps to</th>"
             "<th>Have</th><th>Target</th></tr></thead><tbody>")
    for d in DWARFS:
        o.append(f"<tr><td><b>{d['id']}</b> {_esc(d['name'])}</td><td>{_esc(d['origin'])}</td>"
                 f"<td>{_esc(d['vis'])}</td><td>{_esc(d['maps'])}</td><td>{have(d)}</td><td>{TARGET}</td></tr>")
    o.append("</tbody></table>")
    for d in DWARFS:
        o.append(f"<h3>{d['id']} &mdash; {_esc(d['name'])} <span class='vis'>[{_esc(d['vis'])}]</span></h3>")
        o.append(f"<p class='intro'><i>{_esc(d['origin'])}. Maps to: {_esc(d['maps'])}. "
                 f"Example: {_esc(d['example'])}.</i></p>")
        o.append(f"<p>{_esc(d['what'])}</p>")
        o.append(f"<p class='total'><b>Target {TARGET} workloads &mdash; have {have(d)}.</b></p>")
        o.append("<table><thead><tr><th>Workload / Algorithm</th><th>Status</th><th>Mechanism / points-to</th>"
                 "<th>Memory signature</th><th>Used in (real world)</th></tr></thead><tbody>")
        if d["workloads"]:
            for w in d["workloads"]:
                name, status, ptr, sig, used = _row(w)
                o.append(f"<tr><td class='wl'>{_wl(name)}</td><td>{_badge(status_cell((name,status,ptr,sig)))}</td>"
                         f"<td>{_esc(ptr)}</td><td>{_esc(sig)}</td><td class='use'>{_esc(used) if used else '--'}</td></tr>")
        else:
            o.append("<tr><td colspan='5'><i>(to define together -- iteration pending)</i></td></tr>")
        o.append("</tbody></table>")
    o.append("<h2>Sources</h2><ul>")
    for name, url in SOURCES:
        o.append(f"<li><a href='{url}'>{_esc(name)}</a></li>")
    o.append("</ul></div></body></html>")
    (DOCS / "test_families_spec.html").write_text("\n".join(o))
    print(f"html -> {DOCS / 'test_families_spec.html'}")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    render_md()
    render_html()
    render_pdf()
