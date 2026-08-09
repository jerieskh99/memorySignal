#!/usr/bin/env bash
# Preflight QA for a capture campaign.
#
# Runs every check in qa/checks/ (numeric order) and reports one verdict. The
# point is to catch, in minutes, the failures that otherwise surface hours into
# a run -- undersized guest RAM, scratch on tmpfs, unbuilt binaries, residue
# from a previous run, a domain stuck paused.
#
# Adding a check: drop an executable-or-not .sh into qa/checks/ named NN_name.sh.
# It is sourced with common.sh already loaded, so it can call pass/warn/fail,
# gssh, cfg, and read the cached guest facts. No registration needed.
#
# Usage:
#   ./qa/preflight.sh [--smoke] [--steps FILE] [--json REPORT]
#
# Env (same names the campaign uses, so one export set serves both):
#   SSH_TARGET SSH_KEY VM_DOMAIN VIRSH_URI CAPTURE_CONFIG ZSTD_DIR STEPS_FILE
#
# Exit: 0 clean (warnings allowed), 1 one or more FAIL findings, 2 usage error.

set -uo pipefail

QA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPTURE_ROOT="$(cd "$QA_ROOT/.." && pwd)"

# --- defaults ----------------------------------------------------------------
VIRSH_URI="${VIRSH_URI:-qemu:///system}"
VM_DOMAIN="${VM_DOMAIN:-Kali Jeries}"
SSH_TARGET="${SSH_TARGET:-}"
SSH_KEY="${SSH_KEY:-}"
CAPTURE_CONFIG="${CAPTURE_CONFIG:-$CAPTURE_ROOT/config_qemu_upc.json}"
STEPS_FILE="${STEPS_FILE:-$CAPTURE_ROOT/plan07_campaign/full_campaign_steps.txt}"
ZSTD_DIR="${ZSTD_DIR:-}"

QA_SMOKE="${QA_SMOKE:-0}"
QA_SMOKE_SECONDS="${QA_SMOKE_SECONDS:-3}"
QA_SMOKE_GRACE="${QA_SMOKE_GRACE:-25}"
QA_GUEST_BOOT_BUDGET="${QA_GUEST_BOOT_BUDGET:-180}"
QA_MIN_HOST_FREE_GB="${QA_MIN_HOST_FREE_GB:-40}"
QA_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)   QA_SMOKE=1; shift ;;
    --steps)   STEPS_FILE="$2"; shift 2 ;;
    --json)    QA_JSON="$2"; shift 2 ;;
    --seconds) QA_SMOKE_SECONDS="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$SSH_TARGET" ]] || { echo "ERROR: SSH_TARGET is required (e.g. SSH_TARGET=kali@192.168.222.63)" >&2; exit 2; }

QA_TMP="$(mktemp -d)"; trap 'rm -rf "$QA_TMP"' EXIT
QA_FACTS="$QA_TMP/guest_facts"
QA_SCRATCH_FILE="$QA_TMP/scratch_args"
: > "$QA_FACTS"; : > "$QA_SCRATCH_FILE"

export QA_ROOT CAPTURE_ROOT VIRSH_URI VM_DOMAIN SSH_TARGET SSH_KEY \
       CAPTURE_CONFIG STEPS_FILE ZSTD_DIR QA_FACTS QA_SCRATCH_FILE \
       QA_SMOKE QA_SMOKE_SECONDS QA_SMOKE_GRACE QA_GUEST_BOOT_BUDGET QA_MIN_HOST_FREE_GB

# shellcheck source=lib/common.sh
. "$QA_ROOT/lib/common.sh"

printf 'Capture preflight\n'
printf '  domain      : %s (%s)\n' "$VM_DOMAIN" "$VIRSH_URI"
printf '  guest       : %s\n' "$SSH_TARGET"
printf '  steps       : %s (%s steps)\n' "$STEPS_FILE" "$(grep -vcE '^\s*#|^\s*$' "$STEPS_FILE" 2>/dev/null || echo '?')"
printf '  retention   : %s\n' "${ZSTD_DIR:-<unset>}"
printf '  smoke       : %s\n' "$([[ "$QA_SMOKE" == "1" ]] && echo "yes (${QA_SMOKE_SECONDS}s/workload)" || echo "no (--smoke to enable)")"

shopt -s nullglob
for chk in "$QA_ROOT"/checks/[0-9][0-9]_*.sh; do
  # shellcheck disable=SC1090
  . "$chk"
done
shopt -u nullglob

printf '\n== Verdict ==\n'
printf '  passed %d   warnings %d   failures %d\n' "$QA_PASS" "$QA_WARN" "$QA_FAIL"

if [[ -n "$QA_JSON" ]]; then
  { printf '{\n  "passed": %d,\n  "warnings": %d,\n  "failures": %d,\n  "findings": [' "$QA_PASS" "$QA_WARN" "$QA_FAIL"
    for i in "${!QA_FAILURES[@]}"; do
      [[ $i -gt 0 ]] && printf ','
      printf '\n    %s' "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${QA_FAILURES[$i]}")"
    done
    printf '\n  ]\n}\n'
  } > "$QA_JSON"
  printf '  report: %s\n' "$QA_JSON"
fi

if (( QA_FAIL > 0 )); then
  printf '\n  %sNOT READY%s -- resolve these before starting a campaign:\n' "$_c_red" "$_c_off"
  for f in "${QA_FAILURES[@]}"; do printf '    - %s\n' "$f"; done
  exit 1
fi

printf '\n  %sREADY%s%s\n' "$_c_grn" "$_c_off" \
  "$( (( QA_WARN > 0 )) && printf ' (with %d warning(s) -- review above)' "$QA_WARN")"
exit 0
