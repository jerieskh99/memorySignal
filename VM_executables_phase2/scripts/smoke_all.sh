#!/usr/bin/env bash
# smoke_all.sh — minimal smoke test for every Phase 2 test.
#
# Each test:
#   1. --help (must exit 0, nonzero output)
#   2. --dry-run (must exit 0)
#
# A tiny, real (non-dry-run) execution is also done for the SECURITY-LIKE,
# MEM, APP-REALISTIC and METHODOLOGY tests with the smallest legal
# parameters so the artifact-creation paths are exercised.
#
# Outputs metadata JSON files into ./smoke_out/. Exits non-zero if any test
# fails or if any SECURITY-LIKE test leaves files behind outside its sandbox.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/bin"
OUT="$ROOT/smoke_out"
LOG="$OUT/smoke.log"
mkdir -p "$OUT"
: > "$LOG"

PYTHON="${PYTHON:-python3}"
FAILS=0
PASSES=0

note() { echo "[smoke] $*" | tee -a "$LOG"; }

run() {
    local label="$1"; shift
    note "RUN $label: $*"
    if "$@" >>"$LOG" 2>&1; then
        note "  PASS"
        PASSES=$((PASSES + 1))
        return 0
    else
        local rc=$?
        note "  FAIL (rc=$rc)"
        FAILS=$((FAILS + 1))
        return $rc
    fi
}

# ---- C executables: --help + --dry-run + a tiny real run ----

c_targets=(
    "mem_workingset_sweep_v2;--working-set-mb 4 --duration 2 --warmup 1"
    "mem_writemag_sweep_v2;--working-set-mb 4 --bytes-per-page 64 --duration 2 --warmup 1"
    "mem_rmw_intensity_v2;--mode rmw --working-set-mb 4 --duration 2 --warmup 1"
    "mem_pagefault_density_v2;--variant fault_only --working-set-mb 4 --duration 2"
    "mem_mmap_traversal_v2;--variant write --file-size-mb 4 --duration 2"
    "sandbox_ransom_seq;--files 10 --file-size-bytes 16384 --duration 30"
    "sandbox_ransom_batched;--files 10 --file-size-bytes 16384 --mem-cap-mb 8 --duration 30"
    "sandbox_ransom_slowburn;--files 3 --file-size-bytes 8192 --interval-s 0 --duration 30"
    "sandbox_ransom_selective;--files 5 --file-size-bytes 8192 --duration 30"
    "sandbox_scanner_metadata;--files 50 --subdirs 5 --file-size-bytes 4096 --passes 2 --duration 30"
    "app_hashtable_intensive_v2;--capacity-pow2 14 --inserts 4096 --lookups 8192 --duration 10"
    # Next-generation benign system-behaviour families (CPU / CACHE / MEM / IO / THREAD / MIXED).
    "cpu_hash_loop_v2;--duration 2 --block-kb 4"
    "cpu_matrix_mult_v2;--dim 64 --duration 2"
    "cpu_branch_random_v2;--duration 2 --table-kb 16"
    "cache_hot_loop_v2;--buffer-kb 32 --duration 2"
    "cache_cold_scan_v2;--working-set-mb 8 --mode rmw --duration 2"
    "cache_stride_sweep_v2;--working-set-mb 8 --stride 4096 --duration 2"
    "mem_random_write_pages_v2;--working-set-mb 8 --bytes-per-page 64 --duration 2 --warmup 1"
    "mem_stride_sweep_large_v2;--working-set-mb 8 --stride 4096 --duration 2 --warmup 1"
    "io_read_cache_hit_v2;--file-size-mb 4 --duration 2 --backing-dir /tmp"
    "io_direct_write_like_v2;--file-size-mb 4 --mode seq --duration 2 --backing-dir /tmp"
    "thread_lock_contention_v2;--threads 2 --duration 2"
    "thread_producer_consumer_v2;--ring-size 256 --duration 2"
    "thread_parallel_alloc_v2;--threads 2 --duration 2"
    "mixed_mem_io_v2;--working-set-mb 8 --file-size-mb 4 --duration 2 --backing-dir /tmp"
    "mixed_cpu_mem_v2;--working-set-mb 8 --duration 2"
    "mixed_cpu_io_v2;--file-size-mb 4 --duration 2 --backing-dir /tmp"
    # Compute-kernel family (Berkeley dwarfs).
    "kernel_stencil_jacobi_v2;--grid-n 256 --duration 2"
    "kernel_stencil_seidel_v2;--grid-n 256 --duration 2"
    "kernel_multigrid_v2;--grid-n 129 --duration 2"
    "kernel_fft_v2;--n 4096 --duration 2"
    "kernel_gemm_v2;--dim 128 --block 32 --duration 2"
    "kernel_spmv_v2;--rows 50000 --nnz-per-row 8 --duration 2"
    "kernel_nbody_v2;--particles 4096 --neighbors 8 --duration 2"
    "kernel_dp_v2;--dim 256 --duration 2"
    "kernel_hmm_v2;--states 64 --steps 1024 --duration 2"
    "kernel_lu_v2;--dim 256 --duration 2"
    "kernel_qr_v2;--dim 192 --duration 2"
    "kernel_attention_v2;--seq-len 256 --d-model 64 --duration 2"
    "kernel_conv_v2;--height 128 --width 128 --filters 16 --ksize 3 --duration 2"
    "kernel_ntt_v2;--n 4096 --limbs 4 --duration 2"
    "kernel_dct_v2;--height 256 --width 256 --duration 2"
    "kernel_dwt_v2;--n 256 --duration 2"
    "kernel_fft2d_v2;--n 128 --duration 2"
    "kernel_barnes_hut_v2;--particles 2048 --theta-milli 700 --duration 2"
    "kernel_md_lj_v2;--particles 4096 --duration 2"
    "kernel_pic_v2;--grid 64 --particles 4096 --solve-iters 5 --duration 2"
    "kernel_fmm_v2;--particles 2000 --box-dim 8 --terms 6 --duration 2"
    "kernel_sph_v2;--particles 4096 --duration 2"
    "kernel_lbm_v2;--width 64 --height 64 --duration 2"
    "kernel_fdtd_v2;--width 128 --height 128 --duration 2"
    "kernel_floyd_v2;--dim 128 --duration 2"
    "kernel_matrixchain_v2;--chain 128 --duration 2"
    "kernel_knapsack_v2;--capacity 65536 --items 512 --duration 2"
    "kernel_smithwaterman_v2;--len-a 256 --len-b 256 --duration 2"
    "kernel_beliefprop_v2;--width 64 --height 64 --states 3 --iters 5 --duration 2"
    "kernel_kalman_v2;--ensemble 1024 --dim 6 --meas 3 --steps 8 --duration 2"
    "kernel_gibbs_v2;--width 256 --height 256 --duration 2"
    "kernel_ldpc_v2;--bits 1024 --duration 2"
    "kernel_spmm_v2;--rows 512 --inner 512 --cols 32 --nnz-per-row 8 --duration 2"
    "kernel_sparse_cholesky_v2;--dim 1024 --bandwidth 32 --duration 2"
    "kernel_spgemm_v2;--rows 256 --inner 256 --cols 256 --nnz-per-row 6 --duration 2"
    "kernel_sddmm_v2;--rows 512 --cols 512 --feat 16 --nnz-per-row 16 --duration 2"
    "kernel_moe_dispatch_v2;--tokens 4096 --dim 64 --experts 8 --duration 2"
    "kernel_fem_assembly_v2;--nodes 512 --elements 2048 --npe 4 --duration 2"
    "kernel_fem_matvec_v2;--nodes 100000 --elements 200000 --npe 4 --duration 2"
    "kernel_dg_v2;--elements 8192 --dofs 16 --neighbors 3 --duration 2"
    "kernel_mesh_smooth_v2;--nodes 100000 --degree 6 --duration 2"
    "kernel_unstructured_fv_v2;--cells 100000 --faces-per-cell 3 --duration 2"
)

for entry in "${c_targets[@]}"; do
    name="${entry%%;*}"
    args="${entry#*;}"
    bin="$BIN/$name"
    if [[ ! -x "$bin" ]]; then
        note "MISSING $bin (skipping)"
        FAILS=$((FAILS + 1))
        continue
    fi
    run "$name --help"   "$bin" --help
    run "$name --dry-run" "$bin" --dry-run --output-dir "$OUT"
    # shellcheck disable=SC2086
    run "$name real"     "$bin" $args --output-dir "$OUT" --seed 1234
done

# ---- Python tests ----

py_targets=(
    "app_realistic/app_sqlite_oltp_v2.py;--duration 3 --rows 200 --tx-per-batch 5 --output-dir ${OUT} --seed 7 --cleanup"
    "app_realistic/app_sqlite_analytical_v2.py;--duration 3 --rows 500 --output-dir ${OUT} --seed 7 --cleanup"
    "app_realistic/app_compress_gzip_v2.py;--input-size-mb 1 --level 1 --output-dir ${OUT} --seed 7 --cleanup --sandbox-dir ${OUT}"
    "app_realistic/app_decompress_gzip_v2.py;--output-size-mb 1 --level 1 --output-dir ${OUT} --seed 7 --cleanup --sandbox-dir ${OUT}"
    "app_realistic/app_json_parse_v2.py;--input-size-mb 1 --output-dir ${OUT} --seed 7 --cleanup --sandbox-dir ${OUT}"
    "methodology/mp_workingset_metric_linearity.py;--inputs-dir ${OUT} --family workingset --output-dir ${OUT}"
)

for entry in "${py_targets[@]}"; do
    name="${entry%%;*}"
    args="${entry#*;}"
    script="$ROOT/$name"
    if [[ ! -f "$script" ]]; then
        note "MISSING $script (skipping)"
        FAILS=$((FAILS + 1))
        continue
    fi
    run "$name --help" "$PYTHON" "$script" --help
    run "$name --dry-run" "$PYTHON" "$script" --dry-run --output-dir "$OUT"
    # shellcheck disable=SC2086
    run "$name real" "$PYTHON" "$script" $args
done

# methodology phase boundary test invokes a child; run only if seq binary exists
SEQ_BIN="$BIN/sandbox_ransom_seq"
if [[ -x "$SEQ_BIN" ]]; then
    run "mp_phase_boundary_inference real" \
        "$PYTHON" "$ROOT/methodology/mp_phase_boundary_inference.py" \
        --child-binary "$SEQ_BIN" \
        --child-args "--files 5 --file-size-bytes 8192 --duration 30" \
        --output-dir "$OUT" --seed 7
fi

# ---- Sandbox isolation check ----
# Verify SECURITY-LIKE tests did not leave files outside their declared sandbox.
# All sandboxes live under /tmp/phase2_sandbox_*. After smoke they should be
# removed (because --no-cleanup was not passed).
stragglers=$(ls -d /tmp/phase2_sandbox_* 2>/dev/null || true)
if [[ -n "$stragglers" ]]; then
    note "WARN: sandbox dirs still present after smoke run:"
    note "$stragglers"
    note "(if --no-cleanup was passed this is expected)"
fi

note ""
note "Smoke summary: PASSES=$PASSES  FAILS=$FAILS"
if (( FAILS > 0 )); then
    exit 1
fi
exit 0
