#!/usr/bin/env bash
# Host-side prerequisites: tools, config, libvirt, disk headroom.
# These are cheap and fail fast -- no point probing the guest if the host
# cannot compress, cannot reach libvirt, or has no room for the output.

section "Host environment"

for t in zstd virsh python3 rsync; do
  if command -v "$t" >/dev/null 2>&1; then
    pass "$t present ($("$t" --version 2>&1 | head -1 | cut -c1-40))"
  else
    fail "$t not found on the host"
  fi
done

if [[ -r "$CAPTURE_CONFIG" ]] && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$CAPTURE_CONFIG" 2>/dev/null; then
  pass "capture config parses: $CAPTURE_CONFIG"
else
  fail "capture config missing or not valid JSON: $CAPTURE_CONFIG"
fi

if virsh -c "$VIRSH_URI" domstate "$VM_DOMAIN" >/dev/null 2>&1; then
  pass "libvirt reachable; domain '$VM_DOMAIN' state=$(virsh -c "$VIRSH_URI" domstate "$VM_DOMAIN" 2>/dev/null | tr -d '\n')"
else
  fail "cannot query domain '$VM_DOMAIN' via $VIRSH_URI"
fi

# Free space where snapshots land and where chains accumulate. The raw dumps
# transit imageDir at 1 GiB each; the chains are the campaign's real output.
for pair in "imageDir:$(cfg imageDir)" "ZSTD_DIR:${ZSTD_DIR:-}"; do
  label="${pair%%:*}"; d="${pair#*:}"
  [[ -n "$d" ]] || { warn "$label not set; skipping space check"; continue; }
  mkdir -p "$d" 2>/dev/null
  if [[ -d "$d" ]]; then
    avail_gb=$(df -PBG "$d" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
    if [[ -n "$avail_gb" ]] && (( avail_gb < QA_MIN_HOST_FREE_GB )); then
      fail "$label $d has ${avail_gb} GiB free (< ${QA_MIN_HOST_FREE_GB} GiB needed)"
    else
      pass "$label $d has ${avail_gb:-?} GiB free"
    fi
  else
    fail "$label $d does not exist and could not be created"
  fi
done

# Leftover raw dumps mean a previous run did not finish cleanly. At 1 GiB each
# they fill a shared disk quickly, and they indicate archival may have failed.
img="$(cfg imageDir)"
if [[ -n "$img" ]]; then
  n=$(find -L "$img" -maxdepth 1 -name 'memory_dump-*.raw' 2>/dev/null | wc -l | tr -d ' ')
  if (( n == 0 )); then
    pass "no leftover raw dumps in $img"
  else
    kb=$(find -L "$img" -maxdepth 1 -name 'memory_dump-*.raw' -exec du -k {} + 2>/dev/null | awk '{s+=$1} END{print s+0}')
    warn "$n leftover raw dump(s) in $img (~$((kb/1024)) MiB) -- clean before a long run"
  fi
fi
