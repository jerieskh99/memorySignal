#!/usr/bin/env bash
# run_phase2_min.sh — minimal viable Phase 2 batch.
#
# Runs the 7-test minimum viable batch from the Phase 2 plan with
# conservative parameters suitable for a local validation run (not a real
# capture cycle). For a real capture cycle, drive these executables via the
# QEMU sampler harness rather than this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/bin"
PYTHON="${PYTHON:-python3}"
OUT="${1:-$ROOT/min_batch_out}"
mkdir -p "$OUT"

"$BIN/sandbox_ransom_seq"          --files 100 --file-size-bytes 65536  --duration 60  --output-dir "$OUT" --seed 1
"$BIN/sandbox_ransom_batched"      --files 100 --file-size-bytes 65536  --mem-cap-mb 64 --duration 60 --output-dir "$OUT" --seed 1
"$BIN/sandbox_scanner_metadata"    --files 500 --subdirs 10 --file-size-bytes 4096 --passes 3 --duration 60 --output-dir "$OUT" --seed 1
"$BIN/mem_workingset_sweep_v2"     --working-set-mb 64  --duration 30 --output-dir "$OUT" --seed 1
"$PYTHON" "$ROOT/app_realistic/app_sqlite_oltp_v2.py"     --duration 30 --rows 20000 --output-dir "$OUT" --sandbox-dir "$OUT" --seed 1 --cleanup
"$PYTHON" "$ROOT/app_realistic/app_compress_gzip_v2.py"   --input-size-mb 32  --output-dir "$OUT" --sandbox-dir "$OUT" --seed 1 --cleanup
"$PYTHON" "$ROOT/methodology/mp_phase_boundary_inference.py" \
        --child-binary "$BIN/sandbox_ransom_seq" \
        --child-args "--files 10 --file-size-bytes 16384 --duration 30" \
        --output-dir "$OUT" --seed 1
echo "Phase 2 minimal batch complete. Metadata in $OUT"
