#!/usr/bin/env bash
# Contamination sources that are NOT the workload itself: state left behind by
# a previous run, on either side of the boundary.
#
# Each of these has bitten this project: orphaned producers suspending the VM
# during the next step's SSH wait, leftover guest scratch, a queue still holding
# jobs from an aborted run, and chain folders from a different configuration
# sitting alongside current ones under the same tree.

section "Cleanliness (prior-run residue)"

# 1. Orphaned capture processes. A producer left running suspends the VM every
#    few seconds, which makes the next step's SSH probe fail intermittently.
for p in capture_producer_qemu_pmemsave.sh capture_consumer_qemu.sh; do
  n=$(pgrep -fc "$p" 2>/dev/null || echo 0)
  if (( n == 0 )); then pass "no stray $p"
  else fail "$n stray $p process(es) running -- will fight the next run for the VM"; fi
done

# 2. Domain state. A paused domain answers neither SSH nor shutdown.
state="$(virsh -c "$VIRSH_URI" domstate "$VM_DOMAIN" 2>/dev/null | tr -d '\n')"
case "$state" in
  *paused*) fail "domain is paused (a producer was likely killed mid-snapshot); resume or destroy before running" ;;
  *running*|*"shut off"*) pass "domain state: $state" ;;
  *) warn "domain state: ${state:-unknown}" ;;
esac

# 3. Job queue. Anything in pending/processing belongs to an aborted run and
#    would be consumed by the next one, mixing steps.
qd="$(cfg queueDir)"
if [[ -n "$qd" && -d "$qd" ]]; then
  for sub in pending processing failed; do
    c=$(ls "$qd/$sub" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$sub" == "failed" ]]; then
      (( c == 0 )) && pass "queue/$sub empty" || warn "queue/$sub holds $c job(s) from a previous run"
    else
      (( c == 0 )) && pass "queue/$sub empty" || fail "queue/$sub holds $c job(s) -- clear before running"
    fi
  done
else
  warn "queueDir not resolved from config; skipped queue check"
fi

# 4. Guest scratch. Payload left from a prior run both consumes space and, if
#    a later workload reads it, can alter behaviour.
if guest_reachable && [[ -s "$QA_SCRATCH_FILE" ]]; then
  while read -r a; do
    [[ "$a" == "--scratch" ]] && continue
    d="${a%%:*}"; [[ -n "$d" ]] || continue
    used=$(gssh "du -sm '$d' 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
    if [[ -z "$used" || "$used" -le 1 ]]; then
      pass "guest scratch $d is clean (${used:-0} MB)"
    else
      warn "guest scratch $d holds ${used} MB from a previous run -- reclaim before capture"
    fi
  done < <(tr ' ' '\n' < "$QA_SCRATCH_FILE" | grep -v '^--scratch$' | grep -v '^$')
fi

# 5. Existing chains. Multiple run_ids under one tree is fine (they are
#    separable by folder), but pooling them in analysis is not -- flag the mix.
if [[ -n "${ZSTD_DIR:-}" && -d "$ZSTD_DIR" ]]; then
  ids=$(find "$ZSTD_DIR" -type d -name 'rep*' 2>/dev/null | sed 's|.*__||' | sort -u | wc -l | tr -d ' ')
  chains=$(find "$ZSTD_DIR" -type d -name 'rep*' 2>/dev/null | wc -l | tr -d ' ')
  if (( chains == 0 )); then
    pass "retention tree empty -- clean slate"
  elif (( ids <= 1 )); then
    pass "$chains existing chain(s) from a single run_id"
  else
    warn "$chains chains spanning $ids run_ids -- separable on disk, but do not pool them in analysis"
  fi
fi
