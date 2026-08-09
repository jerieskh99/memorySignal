#!/usr/bin/env bash
# Empirical per-workload probe: run every step briefly on the guest and measure
# what it ACTUALLY consumes -- exit code, peak RSS, and bytes left in scratch.
#
# Measuring beats inferring here. Most workloads declare size in domain units,
# not megabytes: kernel_dp_v2 --dim 8192 allocates dim^2 * 4 = 268 MB, and
# kernel_mesh_smooth_v2 allocates its value array TWICE (val + val_new). No
# flag reveals either. Static analysis of --working-set-mb style flags is blind
# to all 64 kernel workloads; peak RSS is not, and it is ground truth rather
# than a parse of someone's C expression.
#
# Allocations happen at startup, so a 3s run captures the real peak.
#
# Opt-in (--smoke): ~QA_SMOKE_SECONDS + overhead per step.

section "Workload probe (${QA_SMOKE_SECONDS}s each: exit code, peak RSS, scratch)"

if [[ "${QA_SMOKE:-0}" != "1" ]]; then
  info "skipped (pass --smoke to enable)"
  return 0 2>/dev/null || exit 0
fi
if ! guest_reachable; then
  fail "guest unreachable; cannot probe workloads"
  return 0 2>/dev/null || exit 0
fi

load_facts
RAM_BUDGET_MB=$(( ${GUEST_RAM_MB:-0} - ${QA_GUEST_OS_RESERVE_MB:-300} ))
(( RAM_BUDGET_MB < 64 )) && RAM_BUDGET_MB=64

# Peak RSS comes from /proc/PID/status VmHWM rather than GNU time, which is not
# installed on this guest. VmHWM is the kernel's own monotonic high-water mark,
# so any sample taken after the peak reports it correctly -- these workloads
# allocate at startup and hold, so a coarse poll is sufficient and needs no
# extra package on the guest.
info "peak RSS via /proc VmHWM (no guest package required)"

SMOKE_ROOT="/var/tmp/qa_smoke_$$"
gssh "mkdir -p '$SMOKE_ROOT'" >/dev/null 2>&1

REPORT="$QA_TMP/smoke_report.tsv"
printf 'workload\texit\tpeak_rss_mb\tscratch_mb\n' > "$REPORT"

ok=0; bad=0; n=0
total=$(grep -vcE '^\s*#|^\s*$' "$STEPS_FILE")

while IFS= read -r line; do
  [[ -n "$line" ]] || continue
  n=$((n+1))
  name=$(printf '%s' "$line" | grep -oE '/(bin|app_realistic|methodology)/[A-Za-z0-9_.-]+' | head -1 | xargs -r basename)
  name="${name:-step$n}"
  sbox="$SMOKE_ROOT/$name"

  cmd=$(printf '%s' "$line" \
        | sed -E "s/--duration [0-9]+/--duration $QA_SMOKE_SECONDS/g" \
        | sed -E "s#(--(sandbox|backing|output|inputs)-dir )[^ ]+#\1$sbox#g")

  # Launch the workload, poll its VmHWM while it lives, report the max. `sh -c`
  # execs a single binary, so the PID we watch is the workload itself.
  wrapped="mkdir -p '$sbox'
    timeout $((QA_SMOKE_SECONDS + QA_SMOKE_GRACE)) sh -c $(printf '%q' "$cmd") >/dev/null 2>'$sbox/.err' &
    wpid=\$!
    hwm=0
    while kill -0 \$wpid 2>/dev/null; do
      v=\$(awk '/VmHWM/{print \$2}' /proc/\$wpid/status 2>/dev/null)
      [ -n \"\$v\" ] && [ \"\$v\" -gt \"\$hwm\" ] 2>/dev/null && hwm=\$v
      sleep 0.2
    done
    wait \$wpid; rc=\$?
    echo \"RC=\$rc RSS=\$hwm SCRATCH=\$(du -sm '$sbox' 2>/dev/null | awk '{print \$1}')\"
    tail -3 '$sbox/.err' 2>/dev/null"

  out="$(gssh "$wrapped" 2>&1)"
  rc=$(printf '%s' "$out"  | grep -oE 'RC=[0-9]+'      | head -1 | cut -d= -f2)
  rss=$(printf '%s' "$out" | grep -oE 'RSS=[0-9]*'     | head -1 | cut -d= -f2)
  scr=$(printf '%s' "$out" | grep -oE 'SCRATCH=[0-9]*' | head -1 | cut -d= -f2)
  rss_mb=$(( ${rss:-0} / 1024 ))
  printf '%s\t%s\t%s\t%s\n' "$name" "${rc:-?}" "$rss_mb" "${scr:-0}" >> "$REPORT"

  if [[ "${rc:-1}" != "0" ]]; then
    bad=$((bad+1))
    reason=$(printf '%s' "$out" | grep -iE 'error|failed|cannot|no space|denied|not found|Traceback|MemoryError|Killed' | head -1 | cut -c1-90)
    [[ "${rc:-}" == "124" ]] && reason="${reason:-hit the ${QA_SMOKE_SECONDS}s+${QA_SMOKE_GRACE}s budget}"
    fail "$name (step $n) exit=${rc:-?} ${reason:+-- $reason}"
  elif (( rss_mb > RAM_BUDGET_MB )); then
    bad=$((bad+1))
    fail "$name (step $n) peak RSS ${rss_mb} MB exceeds the ~${RAM_BUDGET_MB} MB guest budget -- will swap or be OOM-killed at full size"
  elif (( rss_mb > RAM_BUDGET_MB * 60 / 100 )); then
    ok=$((ok+1))
    warn "$name peak RSS ${rss_mb} MB is over 60% of the ~${RAM_BUDGET_MB} MB budget (scratch ${scr:-0} MB)"
  else
    ok=$((ok+1))
    printf '  %s[ok]%s   %-34s rss %4s MB  scratch %4s MB  (%s/%s)\n' \
      "$_c_grn" "$_c_off" "$name" "$rss_mb" "${scr:-0}" "$n" "$total"
  fi
done < <(grep -vE '^\s*#|^\s*$' "$STEPS_FILE")

gssh "rm -rf '$SMOKE_ROOT'" >/dev/null 2>&1
(( bad == 0 )) && pass "all $ok workloads exited 0 within the ~${RAM_BUDGET_MB} MB budget"

# Peak footprints are the input to capacity planning; keep them.
if [[ -n "${QA_KEEP_REPORT:-}" ]]; then
  cp "$REPORT" "$QA_KEEP_REPORT" && info "footprint report: $QA_KEEP_REPORT"
else
  info "top footprints:"
  tail -n +2 "$REPORT" | sort -t$'\t' -k3 -rn | head -6 \
    | awk -F'\t' '{printf "    %-34s rss %5s MB  scratch %5s MB\n",$1,$3,$4}'
fi
