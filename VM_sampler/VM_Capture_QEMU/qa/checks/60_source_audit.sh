#!/usr/bin/env bash
# Static audit of workload SOURCE for properties that affect data integrity.
#
# Deliberately narrow. Estimating a workload's footprint by parsing C size
# expressions is brittle -- 40_smoke measures peak RSS instead, which is ground
# truth. What IS reliably static is whether a workload handles failure and
# cleans up after itself, and both have already cost this project real data:
#
#   * all 8 sandbox_* workloads have no cleanup path at all, so every run
#     leaves its payload (up to 1 GB) on the guest
#   * an allocation whose result is not checked segfaults or, worse, writes
#     through a null/short buffer and produces plausible but wrong data
#
# Runs against the local checkout, so it needs no guest and costs nothing.

section "Workload source audit"

SRC_ROOT="${QA_WORKLOAD_SRC:-$CAPTURE_ROOT/../../VM_executables_phase2}"
if [[ ! -d "$SRC_ROOT" ]]; then
  warn "workload sources not found at $SRC_ROOT (set QA_WORKLOAD_SRC); skipped"
  return 0 2>/dev/null || exit 0
fi

# Only audit sources the campaign actually invokes.
# (while-read, not mapfile: mapfile is bash 4+, absent on macOS's stock bash 3.2.)
used=()
while IFS= read -r w; do [[ -n "$w" ]] && used+=("$w"); done < <(
  grep -vE '^\s*#|^\s*$' "$STEPS_FILE" 2>/dev/null \
  | grep -oE '/(bin|app_realistic|methodology)/[A-Za-z0-9_.-]+' \
  | xargs -n1 basename | sed 's/\.py$//' | sort -u
)
(( ${#used[@]} > 0 )) || { warn "no workloads resolved from $STEPS_FILE"; return 0 2>/dev/null || exit 0; }

# Workloads the campaign gives a real payload directory (--sandbox-dir /
# --backing-dir). Only these are expected to clean up after themselves;
# --output-dir alone is metadata and a few KB.
payload_workloads=()
while IFS= read -r w; do [[ -n "$w" ]] && payload_workloads+=("$w"); done < <(
  grep -vE '^\s*#|^\s*$' "$STEPS_FILE" 2>/dev/null \
  | grep -E '\-\-(sandbox-dir|backing-dir)' \
  | grep -oE '/(bin|app_realistic|methodology)/[A-Za-z0-9_.-]+' \
  | xargs -n1 basename | sed 's/\.py$//' | sort -u
)
(( ${#payload_workloads[@]} > 0 )) || payload_workloads=("__none__")

unchecked=(); nocleanup=(); audited=0

for w in "${used[@]}"; do
  src=$(find "$SRC_ROOT" -name "${w}.c" -o -name "${w}.py" 2>/dev/null | head -1)
  [[ -n "$src" ]] || continue
  audited=$((audited+1))

  case "$src" in
    *.c)
      # Allocation sites vs failure checks. These sources use the idiom
      # `P2_LOG_ERR("... failed")` / `MAP_FAILED` right after allocating, so a
      # large gap between the two counts means some site is unguarded.
      allocs=$(grep -cE '\b(malloc|calloc|realloc)\s*\(|\bmmap\s*\(' "$src" 2>/dev/null)
      guards=$(grep -cE 'MAP_FAILED|alloc_failed|malloc failed|== *NULL|!= *NULL|mmap\([0-9%]' "$src" 2>/dev/null)
      if (( allocs > 0 && guards == 0 )); then
        unchecked+=("$w ($allocs allocation site(s), no failure check)")
      fi

      # Does it remove what it writes? Only asked of workloads the campaign
      # actually hands a payload directory -- a kernel that fopen()s a metadata
      # JSON is not a file-writing workload in the sense that matters here.
      if printf '%s\n' "${payload_workloads[@]}" | grep -qx "$w"; then
        cleanups=$(grep -cE '\bunlink\s*\(|\bremove\s*\(|\brmdir\s*\(|nftw' "$src" 2>/dev/null)
        (( cleanups == 0 )) && nocleanup+=("$w")
      fi
      ;;
    *.py)
      if printf '%s\n' "${payload_workloads[@]}" | grep -qx "$w"; then
        grep -qE 'cleanup|unlink|rmtree|remove\(' "$src" 2>/dev/null || nocleanup+=("$w")
      fi
      ;;
  esac
done

info "audited $audited of ${#used[@]} referenced workloads"

if (( ${#unchecked[@]} == 0 )); then
  pass "every audited workload checks its allocations"
else
  for u in "${unchecked[@]}"; do
    fail "unchecked allocation: $u -- a failed alloc may corrupt output rather than exit"
  done
fi

if (( ${#nocleanup[@]} == 0 )); then
  pass "every file-writing workload has a cleanup path"
else
  warn "${#nocleanup[@]} file-writing workload(s) never delete their payload: ${nocleanup[*]}"
  info "the orchestrator's post-cell reclaim covers this, and covers the crash case they cannot"
fi
