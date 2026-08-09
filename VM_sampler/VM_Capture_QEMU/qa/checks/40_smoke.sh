#!/usr/bin/env bash
# Empirical per-workload smoke: run every step briefly on the guest and check
# it exits 0. Static analysis reasons about declared sizes; this catches what
# it cannot -- a workload that segfaults, needs a missing runtime, mishandles a
# flag, or fails only once actually executed.
#
# Opt-in (--smoke) because it costs ~QA_SMOKE_SECONDS per step plus overhead.
# For 101 steps at 3s that is roughly 10-15 minutes -- cheap against a 9-hour
# campaign, and it is the only check that would have caught cache_cold_scan_v2
# failing with mmap(2147483648) before the campaign consumed a night.

section "Workload smoke (${QA_SMOKE_SECONDS}s each)"

if [[ "${QA_SMOKE:-0}" != "1" ]]; then
  info "skipped (pass --smoke to enable)"
  return 0 2>/dev/null || exit 0
fi

if ! guest_reachable; then
  fail "guest unreachable; cannot smoke workloads"
  return 0 2>/dev/null || exit 0
fi

# Isolated scratch so a smoke run cannot pollute campaign data, and so a
# workload that leaves files behind does not skew the real run's disk state.
SMOKE_ROOT="/var/tmp/qa_smoke_$$"
gssh "mkdir -p '$SMOKE_ROOT'" >/dev/null 2>&1

ok=0; bad=0; n=0
total=$(grep -vcE '^\s*#|^\s*$' "$STEPS_FILE")

while IFS= read -r line; do
  [[ -n "$line" ]] || continue
  n=$((n+1))
  name=$(printf '%s' "$line" | grep -oE '/(bin|app_realistic|methodology)/[A-Za-z0-9_.-]+' | head -1 | xargs -r basename)
  name="${name:-step$n}"

  # Shorten duration and redirect every scratch path into the smoke sandbox so
  # the probe is fast and self-contained.
  cmd=$(printf '%s' "$line" \
        | sed -E "s/--duration [0-9]+/--duration $QA_SMOKE_SECONDS/g" \
        | sed -E "s#(--(sandbox|backing|output|inputs)-dir )[^ ]+#\1$SMOKE_ROOT/$name#g")

  if out=$(gssh "timeout $((QA_SMOKE_SECONDS + QA_SMOKE_GRACE)) sh -c $(printf '%q' "$cmd")" 2>&1); then
    ok=$((ok+1))
    printf '  %s[ok]%s   %-38s %s\n' "$_c_grn" "$_c_off" "$name" "(${n}/${total})"
  else
    rc=$?
    bad=$((bad+1))
    reason=$(printf '%s' "$out" | grep -iE 'error|failed|cannot|no space|denied|not found|Traceback|MemoryError' | head -1 | cut -c1-100)
    [[ $rc -eq 124 ]] && reason="${reason:-exceeded ${QA_SMOKE_SECONDS}s+${QA_SMOKE_GRACE}s budget (may be normal for long-setup workloads)}"
    fail "$name (step $n) exit=$rc ${reason:+-- $reason}"
  fi
done < <(grep -vE '^\s*#|^\s*$' "$STEPS_FILE")

gssh "rm -rf '$SMOKE_ROOT'" >/dev/null 2>&1
(( bad == 0 )) && pass "all $ok workloads exited 0"
info "smoke sandbox removed: $SMOKE_ROOT"
