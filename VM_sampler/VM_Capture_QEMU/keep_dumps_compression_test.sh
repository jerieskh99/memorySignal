#!/usr/bin/env bash
#
# keep_dumps_compression_test.sh
# ─────────────────────────────────────────────────────────────────────────────
# Compression benchmark for raw VM memory dumps.
#
# Drives Section 4 of docs/tuning_plans/keep_dumps_audit.md. The team gated
# the implementation of any retention mode (mode 1 / mode 2) on the operator
# producing real zstd-3 ratio numbers from this host. This script produces
# those numbers reproducibly.
#
# What it does
# ─────────────
#   1. Locates a representative memory_dump-*.raw under the libvirt dump dir.
#      If none exists, prints a one-cell capture recipe and exits.
#   2. Benchmarks zstd-3, zstd-9, zstd-19, gzip-3, and lz4 against that dump.
#      Records compression time, decompression time, output size, and ratio.
#   3. Verifies decompression round-trips byte-for-byte against the original.
#   4. Writes a summary JSON + a human-readable table to the current directory.
#   5. Cleans up temp files.
#
# What it does NOT do
# ────────────────────
#   - Does not modify the producer.
#   - Does not modify the orchestrator.
#   - Does not enable --keep-dumps anywhere.
#   - Does not run any VM operation.
#   - The dump file under test is read-only; never deleted.
#
# Outputs
# ────────
#   ./keep_dumps_compression_report.json   (machine readable · drop this back)
#   ./keep_dumps_compression_report.txt    (human readable · paste in chat)
#
# Usage
# ──────
#   chmod +x keep_dumps_compression_test.sh   # one time
#   ./keep_dumps_compression_test.sh          # default · uses newest dump
#   ./keep_dumps_compression_test.sh /path/to/dump.raw   # specific file
#
# Estimated runtime
# ──────────────────
#   ~3-8 minutes for one 1 GiB dump (zstd-19 dominates · slow but necessary
#   for a complete picture). zstd-3 finishes in ~3-5 s.
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_DIR_DEFAULT="/var/lib/libvirt/qemu/dump"
WORK_DIR="${TMPDIR:-/tmp}/keep_dumps_test_$$"
REPORT_JSON="${PWD}/keep_dumps_compression_report.json"
REPORT_TXT="${PWD}/keep_dumps_compression_report.txt"

DUMP="${1:-}"

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

log() { printf "%s\n" "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

cleanup() {
    rm -rf "$WORK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wall-clock seconds elapsed by running CMD. Prints ELAPSED on stdout.
# Suppresses CMD output.
time_seconds() {
    local start end
    start=$(date +%s.%N)
    "$@" >/dev/null 2>&1
    end=$(date +%s.%N)
    python3 -c "print(round($end - $start, 3))"
}

bytes_of() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

# ─────────────────────────────────────────────────────────────────────────────
# preflight · pick the dump
# ─────────────────────────────────────────────────────────────────────────────

if [ -n "$DUMP" ]; then
    if [ ! -f "$DUMP" ]; then
        log "ERROR: file not found: $DUMP"
        exit 2
    fi
else
    DUMP=$(ls -t "$IMAGE_DIR_DEFAULT"/memory_dump-*.raw 2>/dev/null | head -1 || true)
    if [ -z "$DUMP" ]; then
        cat >&2 <<'EOF'
No memory_dump-*.raw found in /var/lib/libvirt/qemu/dump/.

Quickest way to generate one:

  cd /project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU
  export SSH_KEY=$HOME/.ssh/id_rsa
  rm -rf /tmp/onecell && mkdir -p /tmp/onecell/cells
  python3 plan02_manifest.py build \
      --workloads sandbox_ransom_batched \
      --intervals-ms 500 --durations-s 60 --replicates 1 \
      --block-size 4 --no-warmup \
      --output /tmp/onecell/manifest.csv \
      --cell-output-dir /tmp/onecell/cells \
      --workload-command "sandbox_ransom_batched=/home/kali/memorySignal/VM_executables_phase2/bin/sandbox_ransom_batched --rounds 1" \
      --ssh-target kali@192.168.222.63 \
      --keep-dumps
  python3 plan02_run.py --manifest /tmp/onecell/manifest.csv \
      --output-dir /tmp/onecell/cells \
      --purge-stale-dumps --min-dumps-headroom 3

Then re-run this script.
EOF
        exit 3
    fi
fi

RAW_SIZE=$(bytes_of "$DUMP")
log ""
log "============================================================"
log "  keep-dumps compression benchmark"
log "============================================================"
log "  dump file: $DUMP"
log "  raw size : $RAW_SIZE bytes ($(python3 -c "print(round($RAW_SIZE/1e9,2))") GB)"
log ""

mkdir -p "$WORK_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# preflight · tooling
# ─────────────────────────────────────────────────────────────────────────────

MISSING=()
for tool in zstd gzip lz4 python3; do
    have "$tool" || MISSING+=("$tool")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    log "WARN: missing tools: ${MISSING[*]}"
    log "      Install with: sudo apt install -y ${MISSING[*]}"
    log "      Skipping any benchmark that requires a missing tool."
fi

# ─────────────────────────────────────────────────────────────────────────────
# benchmarks
# ─────────────────────────────────────────────────────────────────────────────

# Each row: name, compressor cmd-prefix template, output-extension
# Compressor cmd takes 2 args: input, output
declare -a NAMES=()
declare -a SIZES=()
declare -a COMPRESS_TIMES=()
declare -a DECOMPRESS_TIMES=()
declare -a RATIOS=()
declare -a OK_FLAGS=()
declare -a NOTES=()

run_bench() {
    local name="$1" out_path="$2"
    shift 2
    NAMES+=("$name")

    local ctime dtime size_bytes ratio decomp_path checksum_match
    ctime=$(time_seconds "$@" "$DUMP" || echo "NaN")
    if [ ! -f "$out_path" ] || [ "$(bytes_of "$out_path")" = "0" ]; then
        log "  $name: compression FAILED"
        SIZES+=("NaN")
        COMPRESS_TIMES+=("NaN")
        DECOMPRESS_TIMES+=("NaN")
        RATIOS+=("NaN")
        OK_FLAGS+=("false")
        NOTES+=("compression command failed")
        return
    fi
    size_bytes=$(bytes_of "$out_path")
    ratio=$(python3 -c "print(round($size_bytes / $RAW_SIZE, 4))")

    # Decompression test (only for the zstd family · others are slow)
    decomp_path="$WORK_DIR/decompressed_$name"
    checksum_match="not_tested"
    dtime="NaN"
    if [[ "$name" == zstd* ]]; then
        dtime=$(time_seconds zstd -d -f -o "$decomp_path" "$out_path" || echo "NaN")
        if [ -f "$decomp_path" ] && cmp -s "$DUMP" "$decomp_path"; then
            checksum_match="ok"
        else
            checksum_match="MISMATCH"
        fi
        rm -f "$decomp_path"
    fi

    SIZES+=("$size_bytes")
    COMPRESS_TIMES+=("$ctime")
    DECOMPRESS_TIMES+=("$dtime")
    RATIOS+=("$ratio")
    OK_FLAGS+=("true")
    NOTES+=("round-trip=$checksum_match")

    log "  $name: ratio=$(python3 -c "print(f'{$ratio:.2%}')") compress=${ctime}s decompress=${dtime}s round-trip=$checksum_match"
}

log "running benchmarks..."

# Raw baseline (no compression · for reference)
NAMES+=("raw")
SIZES+=("$RAW_SIZE")
COMPRESS_TIMES+=("0.000")
DECOMPRESS_TIMES+=("0.000")
RATIOS+=("1.0000")
OK_FLAGS+=("true")
NOTES+=("identity")

if have zstd; then
    run_bench "zstd-3"  "$WORK_DIR/dump.zst3"   zstd -3  -q -f -o "$WORK_DIR/dump.zst3"
    run_bench "zstd-9"  "$WORK_DIR/dump.zst9"   zstd -9  -q -f -o "$WORK_DIR/dump.zst9"
    run_bench "zstd-19" "$WORK_DIR/dump.zst19"  zstd -19 -q -f -o "$WORK_DIR/dump.zst19"
fi
if have gzip; then
    # gzip writes to stdout with -c · use bash -c to capture
    OUT="$WORK_DIR/dump.gz3"
    ctime=$(time_seconds bash -c "gzip -3 -c '$DUMP' > '$OUT'")
    if [ -f "$OUT" ] && [ "$(bytes_of "$OUT")" != "0" ]; then
        size_bytes=$(bytes_of "$OUT")
        ratio=$(python3 -c "print(round($size_bytes / $RAW_SIZE, 4))")
        NAMES+=("gzip-3"); SIZES+=("$size_bytes")
        COMPRESS_TIMES+=("$ctime"); DECOMPRESS_TIMES+=("NaN")
        RATIOS+=("$ratio"); OK_FLAGS+=("true")
        NOTES+=("not round-trip tested")
        log "  gzip-3: ratio=$(python3 -c "print(f'{$ratio:.2%}')") compress=${ctime}s"
    fi
fi

if have lz4; then
    OUT="$WORK_DIR/dump.lz4"
    ctime=$(time_seconds lz4 -3 -q -f "$DUMP" "$OUT")
    if [ -f "$OUT" ] && [ "$(bytes_of "$OUT")" != "0" ]; then
        size_bytes=$(bytes_of "$OUT")
        ratio=$(python3 -c "print(round($size_bytes / $RAW_SIZE, 4))")
        NAMES+=("lz4"); SIZES+=("$size_bytes")
        COMPRESS_TIMES+=("$ctime"); DECOMPRESS_TIMES+=("NaN")
        RATIOS+=("$ratio"); OK_FLAGS+=("true")
        NOTES+=("not round-trip tested")
        log "  lz4: ratio=$(python3 -c "print(f'{$ratio:.2%}')") compress=${ctime}s"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# write reports
# ─────────────────────────────────────────────────────────────────────────────

log ""
log "============================================================"
log "  summary"
log "============================================================"

{
    printf "%-10s %14s %10s %10s %10s %s\n" \
        "method" "bytes" "ratio" "comp(s)" "decomp(s)" "notes"
    printf "%-10s %14s %10s %10s %10s %s\n" \
        "------" "-----" "-----" "-------" "---------" "-----"
    for i in "${!NAMES[@]}"; do
        printf "%-10s %14s %10s %10s %10s %s\n" \
            "${NAMES[$i]}" "${SIZES[$i]}" "${RATIOS[$i]}" \
            "${COMPRESS_TIMES[$i]}" "${DECOMPRESS_TIMES[$i]}" "${NOTES[$i]}"
    done
} | tee "$REPORT_TXT"

# JSON · pass arrays via a TSV staging file so quoting can't break the python
STAGE="$WORK_DIR/stage.tsv"
: > "$STAGE"
for i in "${!NAMES[@]}"; do
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${NAMES[$i]}" "${SIZES[$i]}" "${RATIOS[$i]}" \
        "${COMPRESS_TIMES[$i]}" "${DECOMPRESS_TIMES[$i]}" \
        "${OK_FLAGS[$i]}" "${NOTES[$i]}" >> "$STAGE"
done

DUMP_PATH="$DUMP" RAW_SIZE_BYTES="$RAW_SIZE" STAGE_PATH="$STAGE" \
    python3 - > "$REPORT_JSON" <<'PYEOF'
import json, math, os
def maybe_float(s):
    if s in ("", "NaN", "nan", "None"): return None
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except (ValueError, TypeError): return None
def maybe_int(s):
    f = maybe_float(s)
    return int(f) if f is not None else None
results = []
with open(os.environ["STAGE_PATH"]) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 7: continue
        name, sz, ratio, ctime, dtime, ok, notes = parts[:7]
        results.append({
            "method": name,
            "bytes": maybe_int(sz),
            "ratio": maybe_float(ratio),
            "compress_seconds": maybe_float(ctime),
            "decompress_seconds": maybe_float(dtime),
            "ok": ok == "true",
            "notes": notes,
        })
raw = int(os.environ["RAW_SIZE_BYTES"])
report = {
    "host": os.uname().nodename,
    "dump_path": os.environ["DUMP_PATH"],
    "raw_size_bytes": raw,
    "raw_size_gb": round(raw / 1e9, 3),
    "results": results,
}
print(json.dumps(report, indent=2))
PYEOF

log ""
log "wrote: $REPORT_JSON"
log "wrote: $REPORT_TXT"
log ""
log "Drop $REPORT_TXT back in chat. Team will pick mode 1 / mode 2 / neither."

exit 0
