#!/usr/bin/env bash
# Server helper: normalize shell scripts to Unix LF and ensure executable bits.
# Use this on the Linux server after pull if you hit:
#   env: 'bash\r': No such file or directory

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

targets=(
  "run_qemu_capture.sh"
  "capture_producer_qemu_pmemsave.sh"
  "capture_consumer_qemu.sh"
  "cleanup_qemu_capture.sh"
  "capture_producer_qemu.sh"
  "capture_producer_qemu_user_raw.sh"
)

echo "[FIX-EOL] Root: $ROOT"
for rel in "${targets[@]}"; do
  path="$ROOT/$rel"
  if [[ -f "$path" ]]; then
    sed -i 's/\r$//' "$path"
    chmod +x "$path"
    echo "[FIX-EOL] Normalized +x: $rel"
  fi
done

echo "[FIX-EOL] Done."

