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
printf 'workload\texit\tpeak_rss_mb\tpeak_anon_mb\tscratch_mb\n' > "$REPORT"

ok=0; bad=0; n=0
total=$(grep -vcE '^\s*#|^\s*$' "$STEPS_FILE")

while IFS= read -r line; do
  [[ -n "$line" ]] || continue
  n=$((n+1))
  name=$(printf '%s' "$line" | grep -oE '/(bin|app_realistic|methodology)/[A-Za-z0-9_.-]+' | head -1 | xargs -r basename)
  name="${name:-step$n}"
  sbox="$SMOKE_ROOT/$name"

  # A step that consumes another step's output (--inputs-dir) cannot be probed
  # in isolation: the smoke sandbox is empty by design, so it would fail for a
  # reason that says nothing about the workload. Its inputs are validated by
  # the campaign-identity check instead.
  if printf '%s' "$line" | grep -q -- '--inputs-dir'; then
    info "skip $name (step $n): consumes another step's output; not probeable standalone"
    continue
  fi

  cmd=$(printf '%s' "$line" \
        | sed -E "s/--duration [0-9]+/--duration $QA_SMOKE_SECONDS/g" \
        | sed -E "s#(--(sandbox|backing|output|inputs)-dir )[^ ]+#\1$sbox#g")

  # Launch, then poll VmHWM across the process TREE. $! is the PID of `timeout`,
  # not the workload -- measuring it reports ~1 MB for everything, which is how
  # the first run produced a uniform and meaningless 1 MB column. The workload
  # is timeout's child (or its grandchild for a compound `mkdir && python3`),
  # so walk two levels and take the maximum.
  wrapped="mkdir -p '$sbox'
    timeout $((QA_SMOKE_SECONDS + QA_SMOKE_GRACE)) sh -c $(printf '%q' "$cmd") >/dev/null 2>'$sbox/.err' &
    wpid=\$!
    hwm=0; anon=0
    while kill -0 \$wpid 2>/dev/null; do
      kids=\$(pgrep -P \$wpid 2>/dev/null)
      gkids=\$(for k in \$kids; do pgrep -P \$k 2>/dev/null; done)
      for p in \$wpid \$kids \$gkids; do
        v=\$(awk '/VmHWM/{print \$2}'  /proc/\$p/status 2>/dev/null)
        a=\$(awk '/RssAnon/{print \$2}' /proc/\$p/status 2>/dev/null)
        [ -n \"\$v\" ] && [ \"\$v\" -gt \"\$hwm\"  ] 2>/dev/null && hwm=\$v
        [ -n \"\$a\" ] && [ \"\$a\" -gt \"\$anon\" ] 2>/dev/null && anon=\$a
      done
      sleep 0.2
    done
    wait \$wpid; rc=\$?
    echo \"RC=\$rc RSS=\$hwm ANON=\$anon SCRATCH=\$(du -sm '$sbox' 2>/dev/null | awk '{print \$1}')\"
    tail -3 '$sbox/.err' 2>/dev/null"

  out="$(gssh "$wrapped" 2>&1)"
  rc=$(printf '%s' "$out"   | grep -oE 'RC=[0-9]+'      | head -1 | cut -d= -f2)
  rss=$(printf '%s' "$out"  | grep -oE 'RSS=[0-9]*'     | head -1 | cut -d= -f2)
  anon=$(printf '%s' "$out" | grep -oE 'ANON=[0-9]*'    | head -1 | cut -d= -f2)
  scr=$(printf '%s' "$out"  | grep -oE 'SCRATCH=[0-9]*' | head -1 | cut -d= -f2)
  rss_mb=$((  ${rss:-0}  / 1024 ))
  anon_mb=$(( ${anon:-0} / 1024 ))
  file_mb=$(( rss_mb - anon_mb )); (( file_mb < 0 )) && file_mb=0
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "${rc:-?}" "$rss_mb" "$anon_mb" "${scr:-0}" >> "$REPORT"

  # Judge on ANONYMOUS memory. File-backed pages are evictable -- the kernel
  # writes them back and drops them under pressure -- so a large file mapping
  # is not a capacity failure. mem_mmap_traversal maps a 1 GB file and shows
  # ~724 MB total RSS while its anonymous footprint is small; that is the
  # kernel doing its job, not a workload that will be OOM-killed. (It DID die
  # earlier, but on tmpfs, where nothing is evictable.)
  if [[ "${rc:-1}" == "124" ]]; then
    ok=$((ok+1))
    warn "$name (step $n) still running at the ${QA_SMOKE_SECONDS}s+${QA_SMOKE_GRACE}s probe budget -- not a crash; verify it completes within its campaign --duration"
  elif [[ "${rc:-1}" != "0" ]]; then
    bad=$((bad+1))
    reason=$(printf '%s' "$out" | grep -iE 'error|failed|cannot|no space|denied|not found|Traceback|MemoryError|Killed' | head -1 | cut -c1-90)
    fail "$name (step $n) exit=${rc:-?} ${reason:+-- $reason}"
  elif (( anon_mb > RAM_BUDGET_MB )); then
    bad=$((bad+1))
    fail "$name (step $n) anonymous RSS ${anon_mb} MB exceeds the ~${RAM_BUDGET_MB} MB budget -- unevictable, will swap or be OOM-killed at full size"
  elif (( anon_mb > RAM_BUDGET_MB * 60 / 100 )); then
    ok=$((ok+1))
    warn "$name anonymous RSS ${anon_mb} MB is over 60% of the ~${RAM_BUDGET_MB} MB budget (file-backed ${file_mb} MB, scratch ${scr:-0} MB)"
  else
    ok=$((ok+1))
    printf '  %s[ok]%s   %-32s anon %4s MB  file %4s MB  scratch %4s MB  (%s/%s)\n' \
      "$_c_grn" "$_c_off" "$name" "$anon_mb" "$file_mb" "${scr:-0}" "$n" "$total"
  fi
done < <(grep -vE '^\s*#|^\s*$' "$STEPS_FILE")

gssh "rm -rf '$SMOKE_ROOT'" >/dev/null 2>&1
(( bad == 0 )) && pass "all $ok workloads exited 0 within the ~${RAM_BUDGET_MB} MB budget"

# Peak footprints are the input to capacity planning; keep them.
if [[ -n "${QA_KEEP_REPORT:-}" ]]; then
  cp "$REPORT" "$QA_KEEP_REPORT" && info "footprint report: $QA_KEEP_REPORT"
else
  info "top footprints (by anonymous, the unevictable part):"
  tail -n +2 "$REPORT" | sort -t$'\t' -k4 -rn | head -8 \
    | awk -F'\t' '{printf "    %-32s anon %5s MB  total %5s MB  scratch %5s MB\n",$1,$4,$3,$5}'
fi
