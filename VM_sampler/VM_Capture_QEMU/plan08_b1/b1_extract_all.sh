#!/usr/bin/env bash
# b1_extract_all.sh -- batch B1 field/trajectory extraction over a list of
# substrate CSVs. Resumable: skips any cell whose trajectory already carries the
# final sentinel, so a re-run only does what is missing.
#
# Usage: b1_extract_all.sh <sources_list> <out_dir> [n_pages]
#   sources_list : file with one substrate_trajectory.csv[.zst|.gz] path per line
#   out_dir      : output root; one <cksum-of-path>/ dir per source
#   n_pages      : total pages (default 262144 = 1 GiB / 4 KiB)
#
# Run under screen/tmux for a long batch. Reads b1_extract_hamming.py from its
# own directory, so this stays in sync with the tracked extractor.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_LIST="${1:?usage: b1_extract_all.sh <sources_list> <out_dir> [n_pages]}"
OUT="${2:?usage: b1_extract_all.sh <sources_list> <out_dir> [n_pages]}"
NP="${3:-262144}"
mkdir -p "$OUT"

n_done=0 n_skip=0 n_fail=0
while IFS= read -r SRC; do
  [ -n "$SRC" ] || continue
  cid=$(printf '%s' "$SRC" | cksum | cut -d' ' -f1)
  od="$OUT/$cid"
  if [ -f "$od/b1_trajectory.jsonl" ] && grep -q '"final"' "$od/b1_trajectory.jsonl"; then
    echo "skip (done): $SRC"; n_skip=$((n_skip + 1)); continue
  fi
  b=$(basename "$SRC"); b=${b%.zst}; b=${b%.gz}; b=${b%.csv}
  wl=$(printf '%s' "$b" | sed -E 's/^run_matrix_test[0-9]+_//; s/(\.npy)?\.substrate_trajectory$//')
  echo "=== $wl ==="
  if python3 "$HERE/b1_extract_hamming.py" "$SRC" "$od" \
        --n-pages "$NP" --cell-id "$cid" --workload "$wl"; then
    n_done=$((n_done + 1))
  else
    echo "FAIL: $SRC"; n_fail=$((n_fail + 1))
  fi
done < "$SRC_LIST"

echo "done=$n_done skipped=$n_skip failed=$n_fail"
du -ch "$OUT"/*/hc_field.csv.zst 2>/dev/null | tail -1
