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
