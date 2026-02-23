#!/usr/bin/env bash
# Helper script for BORG snapshot management - retrieve consecutive snapshots for delta calculation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/test_borg_snapshot.py"

usage() {
  cat <<EOF
Usage: $0 <command> [options]

Commands:
  create <monitor-socket> <borg-repo>
    Create a new snapshot from QEMU VM

  list <borg-repo>
    List all snapshots in repository

  metrics <borg-repo>
    Show repository metrics (size, count, etc.)

  get-consecutive <borg-repo> <index> [output-dir]
    Get two consecutive snapshots (index and index+1) for delta calculation
    Example: get-consecutive /path/to/repo 0 /tmp/extracted
             Gets first two snapshots (0 and 1)

  get-pair <borg-repo> <archive1> <archive2> [output-dir]
    Get specific two snapshots by archive name
    Example: get-pair /path/to/repo snapshot-20250116120000 snapshot-20250116120030 /tmp/extracted

  get-latest <borg-repo> [output-dir]
    Get the two most recent snapshots

Examples:
  # Create a snapshot
  $0 create /tmp/qemu-monitor.sock /tmp/borg_repo

  # List all snapshots
  $0 list /tmp/borg_repo

  # Get metrics
  $0 metrics /tmp/borg_repo

  # Get first two snapshots (for delta calc)
  $0 get-consecutive /tmp/borg_repo 0 /tmp/extracted

  # Get latest two snapshots
  $0 get-latest /tmp/borg_repo /tmp/extracted
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

command="$1"
shift

case "$command" in
  create)
    if [[ $# -lt 2 ]]; then
      echo "Usage: $0 create <monitor-socket> <borg-repo>"
      exit 1
    fi
    python3 "$PYTHON_SCRIPT" --monitor-socket "$1" --borg-repo "$2" --create-snapshot
    ;;
  
  list)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 list <borg-repo>"
      exit 1
    fi
    python3 "$PYTHON_SCRIPT" --borg-repo "$1" --list-snapshots
    ;;
  
  metrics)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 metrics <borg-repo>"
      exit 1
    fi
    python3 "$PYTHON_SCRIPT" --borg-repo "$1" --metrics
    ;;
  
  get-consecutive)
    if [[ $# -lt 2 ]]; then
      echo "Usage: $0 get-consecutive <borg-repo> <index> [output-dir]"
      exit 1
    fi
    repo="$1"
    index="$2"
    output_dir="${3:-/tmp/borg_extracted_$(date +%s)}"
    
    # Get sorted list of archives (parse from Python output)
    archives_json=$(python3 "$PYTHON_SCRIPT" --borg-repo "$repo" --list-snapshots 2>/dev/null | grep -E "^\s+\[[0-9]+\]" | awk '{print $2}')
    readarray -t archives <<< "$archives_json"
    
    if [[ ${#archives[@]} -lt $((index + 2)) ]]; then
      echo "[ERROR] Not enough snapshots. Need at least $((index + 2)), have ${#archives[@]}"
      exit 1
    fi
    
    arch1="${archives[$index]}"
    arch2="${archives[$((index + 1))]}"
    
    echo "[INFO] Extracting consecutive snapshots:"
    echo "  prev: $arch1"
    echo "  curr: $arch2"
    echo "  output: $output_dir"
    
    python3 "$PYTHON_SCRIPT" --borg-repo "$repo" --get-snapshots "$arch1" "$arch2" --output-dir "$output_dir"
    
    # Find .elf files and print paths
    echo ""
    echo "[SUCCESS] Snapshots extracted. Use these paths for delta calculation:"
    find "$output_dir" -name "*.elf" -type f | sort | while read -r elf; do
      echo "  $elf"
    done
    ;;
  
  get-pair)
    if [[ $# -lt 3 ]]; then
      echo "Usage: $0 get-pair <borg-repo> <archive1> <archive2> [output-dir]"
      exit 1
    fi
    repo="$1"
    arch1="$2"
    arch2="$3"
    output_dir="${4:-/tmp/borg_extracted_$(date +%s)}"
    
    echo "[INFO] Extracting snapshot pair:"
    echo "  prev: $arch1"
    echo "  curr: $arch2"
    echo "  output: $output_dir"
    
    python3 "$PYTHON_SCRIPT" --borg-repo "$repo" --get-snapshots "$arch1" "$arch2" --output-dir "$output_dir"
    
    echo ""
    echo "[SUCCESS] Snapshots extracted. Use these paths for delta calculation:"
    find "$output_dir" -name "*.elf" -type f | sort | while read -r elf; do
      echo "  $elf"
    done
    ;;
  
  get-latest)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 get-latest <borg-repo> [output-dir]"
      exit 1
    fi
    repo="$1"
    output_dir="${2:-/tmp/borg_extracted_$(date +%s)}"
    
    # Get sorted list (newest last)
    archives_json=$(python3 "$PYTHON_SCRIPT" --borg-repo "$repo" --list-snapshots 2>/dev/null | grep -E "^\s+\[[0-9]+\]" | awk '{print $2}')
    readarray -t archives <<< "$archives_json"
    
    if [[ ${#archives[@]} -lt 2 ]]; then
      echo "[ERROR] Need at least 2 snapshots, have ${#archives[@]}"
      exit 1
    fi
    
    # Get last two
    arch1="${archives[-2]}"
    arch2="${archives[-1]}"
    
    echo "[INFO] Extracting latest two snapshots:"
    echo "  prev: $arch1"
    echo "  curr: $arch2"
    echo "  output: $output_dir"
    
    python3 "$PYTHON_SCRIPT" --borg-repo "$repo" --get-snapshots "$arch1" "$arch2" --output-dir "$output_dir"
    
    echo ""
    echo "[SUCCESS] Snapshots extracted. Use these paths for delta calculation:"
    find "$output_dir" -name "*.elf" -type f | sort | while read -r elf; do
      echo "  $elf"
    done
    ;;
  
  *)
    echo "Unknown command: $command"
    usage
    exit 1
    ;;
esac
