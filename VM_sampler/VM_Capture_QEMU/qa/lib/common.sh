#!/usr/bin/env bash
# Shared helpers for the capture preflight checks.
#
# Every check sources this, then uses pass/fail/warn/info to report. The runner
# (preflight.sh) aggregates results; a check never exits the process itself, so
# one failure does not hide the remaining findings.

: "${QA_ROOT:?common.sh must be sourced by preflight.sh}"

# --- result accounting -------------------------------------------------------
QA_PASS=0; QA_WARN=0; QA_FAIL=0
QA_FAILURES=()

_c_red=$'\033[31m'; _c_yel=$'\033[33m'; _c_grn=$'\033[32m'; _c_dim=$'\033[2m'; _c_off=$'\033[0m'
[[ -t 1 ]] || { _c_red=""; _c_yel=""; _c_grn=""; _c_dim=""; _c_off=""; }

pass() { QA_PASS=$((QA_PASS+1)); printf '  %s[ok]%s   %s\n' "$_c_grn" "$_c_off" "$*"; }
warn() { QA_WARN=$((QA_WARN+1)); printf '  %s[warn]%s %s\n' "$_c_yel" "$_c_off" "$*"; }
fail() { QA_FAIL=$((QA_FAIL+1)); QA_FAILURES+=("$*"); printf '  %s[FAIL]%s %s\n' "$_c_red" "$_c_off" "$*"; }
info() { printf '  %s%s%s\n' "$_c_dim" "$*" "$_c_off"; }
section() { printf '\n== %s ==\n' "$*"; }

# --- config ------------------------------------------------------------------
# Read a key from the capture config JSON without requiring jq.
cfg() {
  local key="$1"
  python3 - "$CAPTURE_CONFIG" "$key" <<'PY' 2>/dev/null
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
key = sys.argv[2]
def find(d):
    if isinstance(d, dict):
        if key in d: return d[key]
        for v in d.values():
            r = find(v)
            if r is not None: return r
    return None
v = find(cfg)
if v is not None: print(v)
PY
}

# --- guest access ------------------------------------------------------------
# All guest commands go through here so a single place controls timeouts and
# the key/target. Returns the command's own exit code; prints its stdout.
gssh() {
  ssh -o ConnectTimeout="${QA_SSH_TIMEOUT:-8}" \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR \
      ${SSH_KEY:+-i "$SSH_KEY"} "$SSH_TARGET" "$@"
}

guest_reachable() { gssh true >/dev/null 2>&1; }

# Wait for the guest to answer, starting the VM if needed. Bounded, so a dead
# guest fails the preflight in minutes rather than hanging (the orchestrator's
# own SSH_WAIT_TIMEOUT defaults to ~14 days, which is how a stalled campaign
# looked frozen instead of failed).
guest_wait() {
  local budget="${1:-180}" waited=0
  guest_reachable && return 0
  local state; state="$(virsh -c "$VIRSH_URI" domstate "$VM_DOMAIN" 2>/dev/null | tr -d '\n')"
  if [[ "$state" != *running* ]]; then
    info "guest not running (state=${state:-unknown}); starting"
    virsh -c "$VIRSH_URI" start "$VM_DOMAIN" >/dev/null 2>&1
  fi
  while (( waited < budget )); do
    guest_reachable && return 0
    sleep 5; waited=$((waited+5))
  done
  return 1
}

# --- guest facts -------------------------------------------------------------
# Probed once, cached in $QA_FACTS as KEY=VALUE, sourced by later checks. Keeps
# each check cheap and lets them reason about capacity without re-querying.
probe_guest_facts() {
  local out
  out="$(gssh 'set -e
    ram=$(awk "/MemTotal/{print int(\$2/1024)}" /proc/meminfo)
    avail=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
    swap=$(awk "/SwapTotal/{print int(\$2/1024)}" /proc/meminfo)
    echo "GUEST_RAM_MB=$ram"
    echo "GUEST_AVAIL_MB=$avail"
    echo "GUEST_SWAP_MB=$swap"
    echo "GUEST_KERNEL=$(uname -r)"
  ' 2>/dev/null)" || return 1
  printf '%s\n' "$out" > "$QA_FACTS"
  [[ -s "$QA_FACTS" ]]
}

# Facts about one guest directory: filesystem type, free MB, writability.
# Answers the question that cost this project a full campaign -- is the
# workload scratch a real disk, or a RAM-backed tmpfs?
probe_guest_dir() {
  local d="$1"
  gssh "d='$d'
    mkdir -p \"\$d\" 2>/dev/null
    t=\$(stat -f -c %T \"\$d\" 2>/dev/null || echo unknown)
    a=\$(df -Pm \"\$d\" 2>/dev/null | awk 'NR==2{print \$4}')
    s=\$(df -P  \"\$d\" 2>/dev/null | awk 'NR==2{print \$1}')
    w=no; touch \"\$d/.qa_probe\" 2>/dev/null && { w=yes; rm -f \"\$d/.qa_probe\"; }
    echo \"type=\$t avail_mb=\${a:-0} source=\$s writable=\$w\"
  " 2>/dev/null
}

load_facts() { [[ -r "$QA_FACTS" ]] && . "$QA_FACTS"; }
