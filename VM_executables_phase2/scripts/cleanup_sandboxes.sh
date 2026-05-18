#!/usr/bin/env bash
# cleanup_sandboxes.sh — remove leftover Phase 2 sandbox directories.
#
# Only removes paths matching /tmp/phase2_sandbox_* or /var/tmp/phase2_sandbox_*.
# Refuses any path outside of these explicit roots.
set -euo pipefail

removed=0
for root in /tmp /var/tmp; do
    for d in "$root"/phase2_sandbox_*; do
        [[ -e "$d" ]] || continue
        if [[ -d "$d" ]]; then
            real="$(cd "$d" && pwd -P)"
            case "$real" in
                /tmp/phase2_sandbox_*|/var/tmp/phase2_sandbox_*)
                    rm -rf -- "$real"
                    echo "removed $real"
                    removed=$((removed+1))
                    ;;
                *)
                    echo "refuse: $d resolves to $real (not under approved root)" >&2
                    ;;
            esac
        fi
    done
done
echo "cleanup_sandboxes: removed=$removed"
