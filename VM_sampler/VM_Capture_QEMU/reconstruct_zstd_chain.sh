#!/usr/bin/env bash
# Rebuild raw memory snapshots from a zstd delta chain produced by capture with ZSTD=1.
#
# A chain folder holds:
#   000000.zst  = the full base snapshot (s0)
#   000001.zst  = a `zstd --patch-from` delta that reconstructs s1 from s0
#   000002.zst  = delta that reconstructs s2 from s1   ...
# This walks them in order: decompress the base, then apply each delta against the
# previously reconstructed snapshot, writing snap_NNNNNN.raw into <out_dir>.
#
# Usage: reconstruct_zstd_chain.sh <chain_dir> <out_dir>

set -euo pipefail

dir="${1:?usage: reconstruct_zstd_chain.sh <chain_dir> <out_dir>}"
out="${2:?usage: reconstruct_zstd_chain.sh <chain_dir> <out_dir>}"
mkdir -p "$out"

prev=""
expected=0
count=0
for f in $(ls "$dir"/*.zst 2>/dev/null | sort); do
  n=$(basename "$f" .zst)
  num=$((10#$n))  # base-10 (avoid octal from leading zeros)
  if [[ "$num" -ne "$expected" ]]; then
    printf 'chain gap: expected %06d.zst but found %s.zst; stopping.\n' "$expected" "$n" >&2
    echo "(the raw snapshot for the missing number was kept locally during capture)" >&2
    exit 1
  fi
  tgt="$out/snap_${n}.raw"
  if [[ -z "$prev" ]]; then
    zstd -d -q "$f" -o "$tgt"                                   # 000000 = full base
  else
    zstd -d -q --long=31 --patch-from="$prev" "$f" -o "$tgt"    # delta vs previous
  fi
  prev="$tgt"
  expected=$((expected + 1))
  count=$((count + 1))
done

if [[ "$count" -eq 0 ]]; then
  echo "no .zst files found in $dir" >&2
  exit 1
fi
echo "reconstructed $count snapshots into $out"
