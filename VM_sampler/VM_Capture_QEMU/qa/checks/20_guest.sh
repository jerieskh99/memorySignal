#!/usr/bin/env bash
# Guest reachability, capacity, scratch-filesystem type, and built binaries.
#
# The scratch-type probe is the important one: the guest mounts /tmp as a
# 483 MB tmpfs (Linux defaults tmpfs to 50% of RAM). Workloads pointed there
# both run out of space AND have their file writes counted as memory-signal
# page changes, which silently invalidates every IO-family measurement.

section "Guest environment"

if guest_wait "$QA_GUEST_BOOT_BUDGET"; then
  pass "guest reachable at $SSH_TARGET"
else
  fail "guest unreachable at $SSH_TARGET after ${QA_GUEST_BOOT_BUDGET}s -- remaining guest checks skipped"
  return 0 2>/dev/null || exit 0
fi

if probe_guest_facts; then
  load_facts
  pass "guest RAM ${GUEST_RAM_MB} MB (available ${GUEST_AVAIL_MB} MB, swap ${GUEST_SWAP_MB} MB), kernel ${GUEST_KERNEL}"
  (( GUEST_AVAIL_MB < 200 )) && warn "only ${GUEST_AVAIL_MB} MB available; workloads will swap"
else
  fail "could not probe guest memory facts"
  return 0 2>/dev/null || exit 0
fi

# Scratch roots referenced by the steps file, probed for type and capacity.
# (while-read, not mapfile: mapfile is bash 4+, absent on macOS's stock bash 3.2.)
scratch_dirs=()
while IFS= read -r d; do [[ -n "$d" ]] && scratch_dirs+=("$d"); done < <(
  grep -vE '^\s*#|^\s*$' "$STEPS_FILE" 2>/dev/null \
  | grep -oE '\-\-(sandbox-dir|backing-dir|output-dir|inputs-dir) [^ ]+' \
  | awk '{print $2}' | xargs -n1 dirname 2>/dev/null | sort -u
)
QA_SCRATCH_ARGS=()
if (( ${#scratch_dirs[@]} == 0 )); then
  warn "no scratch dirs found in $STEPS_FILE"
else
  for d in "${scratch_dirs[@]}"; do
    line="$(probe_guest_dir "$d")"
    [[ -n "$line" ]] || { warn "could not probe guest dir $d"; continue; }
    # shellcheck disable=SC2206
    kv=($line)
    typ="${kv[0]#type=}"; avail="${kv[1]#avail_mb=}"; src="${kv[2]#source=}"; w="${kv[3]#writable=}"
    QA_SCRATCH_ARGS+=(--scratch "$d:type=$typ,avail_mb=$avail,writable=$w")
    if [[ "$typ" == "tmpfs" ]]; then
      fail "scratch $d is tmpfs (${avail} MB, RAM-backed) -- file writes consume guest RAM and contaminate the memory signal"
    elif [[ "$w" != "yes" ]]; then
      fail "scratch $d not writable by the guest user"
    else
      pass "scratch $d on $src ($typ, ${avail} MB free, writable)"
    fi
  done
fi
printf '%s\n' "${QA_SCRATCH_ARGS[@]}" > "$QA_SCRATCH_FILE" 2>/dev/null || true

# Every binary/script the steps file invokes must exist and be executable --
# a missing one costs a full step (exit 127) mid-campaign.
missing=0; checked=0
while read -r prog; do
  [[ -n "$prog" ]] || continue
  checked=$((checked+1))
  gssh "test -x '$prog' -o -r '$prog'" >/dev/null 2>&1 || { fail "guest program missing or not executable: $prog"; missing=$((missing+1)); }
done < <(
  grep -vE '^\s*#|^\s*$' "$STEPS_FILE" 2>/dev/null \
  | grep -oE '/[^ ]*/(bin|app_realistic|methodology)/[A-Za-z0-9_.-]+' | sort -u
)
(( missing == 0 )) && pass "all $checked referenced guest programs present and executable"
