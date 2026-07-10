#!/usr/bin/env bash
# Health-check a zstd delta-chain capture (produced by capture with ZSTD=1).
#
# Verifies, for every per-workload chain folder under ZSTD_DIR:
#   - a full base (000000.zst) is present,
#   - the deltas are gap-free and sequential (000001, 000002, ...),
#   - no file is empty,
# reports snapshot count + sizes + compression, optionally checks the local dump
# directory is not accumulating, and can spot-check reconstruction (bounded).
#
# Usage:
#   check_zstd_capture.sh <ZSTD_DIR> [--imagedir <dump_dir>] [--reconstruct <chain_subdir>] [--depth N]
#
# Examples:
#   check_zstd_capture.sh /mnt/nfs/jeries/zstd_test
#   check_zstd_capture.sh /mnt/nfs/jeries/zstd_test --imagedir /var/lib/libvirt/qemu/dump \
#       --reconstruct test1_mem_workingset_sweep_v2__20260710123456 --depth 3

set -uo pipefail

fsize() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null; }

ZDIR="${1:?usage: check_zstd_capture.sh <ZSTD_DIR> [--imagedir DIR] [--reconstruct SUBDIR] [--depth N]}"
shift
IMAGEDIR=""; RECON=""; DEPTH=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --imagedir) IMAGEDIR="$2"; shift 2;;
    --reconstruct) RECON="$2"; shift 2;;
    --depth) DEPTH="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0

echo "== zstd capture check: $ZDIR =="
[[ -d "$ZDIR" ]] || { echo "FAIL: ZSTD_DIR not found: $ZDIR"; exit 1; }

nchains=0
for chain in "$ZDIR"/*/; do
  [[ -d "$chain" ]] || continue
  nchains=$((nchains + 1))
  name="$(basename "$chain")"

  cnt=$(ls "$chain"/*.zst 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$cnt" -eq 0 ]]; then echo "  [FAIL] $name: no .zst files"; fail=1; continue; fi
  [[ -f "$chain/000000.zst" ]] || { echo "  [FAIL] $name: missing base 000000.zst"; fail=1; }

  exp=0; gap=""; empty=""
  while IFS= read -r f; do
    num=$((10#$(basename "$f" .zst)))
    [[ "$num" -eq "$exp" ]] || { gap="expected $(printf %06d "$exp"), found $(printf %06d "$num")"; break; }
    [[ "$(fsize "$f")" -gt 0 ]] || empty="$(basename "$f")"
    exp=$((exp + 1))
  done < <(ls "$chain"/*.zst 2>/dev/null | sort)

  base_kb=$(du -k "$chain/000000.zst" 2>/dev/null | awk '{print $1}')
  tot_kb=$(du -sk "$chain" | awk '{print $1}')
  ndelta=$((cnt - 1)); avg=0
  [[ "$ndelta" -gt 0 ]] && avg=$(( (tot_kb - base_kb) / ndelta ))

  if [[ -n "$gap" ]]; then
    echo "  [FAIL] $name: chain GAP ($gap); $cnt files, ${tot_kb}K"; fail=1
  elif [[ -n "$empty" ]]; then
    echo "  [FAIL] $name: empty file $empty; $cnt files"; fail=1
  else
    echo "  [ok]   $name: $cnt snapshots (1 base + $ndelta deltas), total ${tot_kb}K, base ${base_kb}K, avg delta ${avg}K"
  fi
done

echo "chains found: $nchains"
[[ "$nchains" -eq 0 ]] && { echo "FAIL: no chains under $ZDIR"; exit 1; }

if [[ -n "$IMAGEDIR" ]]; then
  dn=$(find "$IMAGEDIR" -maxdepth 1 -name 'memory_dump-*.raw' 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$dn" -le 3 ]]; then
    echo "  [ok]   local dumps in $IMAGEDIR: $dn (not accumulating)"
  else
    echo "  [WARN] local dumps in $IMAGEDIR: $dn (expected ~0 -- deletion may be failing)"; fail=1
  fi
fi

if [[ -n "$RECON" ]]; then
  src="$ZDIR/$RECON"
  [[ -d "$src" ]] || { echo "FAIL: --reconstruct target not found: $src"; exit 1; }
  tmp="$(mktemp -d)"; mkdir -p "$tmp/chain"
  i=0
  for f in $(ls "$src"/*.zst 2>/dev/null | sort); do
    cp "$f" "$tmp/chain/"; i=$((i + 1)); [[ "$i" -ge "$DEPTH" ]] && break
  done
  echo "== reconstruct spot-check: first $i file(s) of $RECON =="
  if bash "$here/reconstruct_zstd_chain.sh" "$tmp/chain" "$tmp/out" >/dev/null 2>&1; then
    sizes=$(for r in "$tmp"/out/snap_*.raw; do fsize "$r"; done | sort -u)
    nsz=$(printf '%s\n' "$sizes" | grep -c .)
    first=$(printf '%s\n' "$sizes" | head -1)
    if [[ "$nsz" -eq 1 && "${first:-0}" -gt 0 ]]; then
      echo "  [ok]   reconstructed $i dump(s), all $first bytes (full-size, non-empty)"
    else
      echo "  [FAIL] reconstructed dumps have inconsistent/zero sizes: $(echo $sizes)"; fail=1
    fi
  else
    echo "  [FAIL] reconstruction failed"; fail=1
  fi
  rm -rf "$tmp"
fi

echo ""
if [[ "$fail" -eq 0 ]]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; exit 1; fi
