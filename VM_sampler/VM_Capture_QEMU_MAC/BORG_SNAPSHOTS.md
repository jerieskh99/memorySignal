# BORG-backed VM Memory Snapshot Management

This directory contains tools for storing VM memory snapshots in a **BORG backup repository** for efficient compression and deduplication. This enables long-term storage of raw memory dumps while allowing retrieval of consecutive snapshots for delta calculation.

## Overview

**Problem**: Raw memory dumps are large (often GBs) and storing many snapshots consumes significant disk space.

**Solution**: Use BORG backup to:
- **Compress** snapshots (typically 2-5x reduction)
- **Deduplicate** across snapshots (consecutive dumps share most memory pages)
- **Store** efficiently with versioning
- **Retrieve** any two snapshots (especially consecutive) for delta calculation

## Requirements

- **BORG Backup**: `brew install borgbackup`
- **QEMU** running with monitor socket (same as main capture pipeline)
- **Python 3** with standard library

## Quick Start

### 1. Initialize BORG Repository

```bash
# Create repository (one-time setup)
borg init --encryption=none /path/to/borg_repo
```

### 2. Create a Snapshot

```bash
# Using Python script directly
python3 test_borg_snapshot.py \
  --monitor-socket /tmp/qemu-monitor.sock \
  --borg-repo /path/to/borg_repo \
  --create-snapshot

# Or using helper script
./borg_helper.sh create /tmp/qemu-monitor.sock /path/to/borg_repo
```

### 3. List Snapshots

```bash
# List all snapshots
./borg_helper.sh list /path/to/borg_repo

# Or directly
python3 test_borg_snapshot.py --borg-repo /path/to/borg_repo --list-snapshots
```

### 4. Get Metrics

```bash
./borg_helper.sh metrics /path/to/borg_repo
```

Shows:
- Total snapshot count
- Repository size (compressed)
- Cache size
- Unique chunks (deduplication stats)
- Total uncompressed size

### 5. Retrieve Consecutive Snapshots for Delta Calculation

```bash
# Get first two snapshots (index 0 and 1)
./borg_helper.sh get-consecutive /path/to/borg_repo 0 /tmp/extracted

# Get latest two snapshots
./borg_helper.sh get-latest /path/to/borg_repo /tmp/extracted

# Get specific pair by archive name
./borg_helper.sh get-pair /path/to/borg_repo snapshot-20250116120000 snapshot-20250116120030 /tmp/extracted
```

After extraction, you'll get paths to `.elf` files that can be used with `live_delta_calc`:

```bash
# Example output:
[SUCCESS] Snapshots extracted. Use these paths for delta calculation:
  /tmp/extracted/snapshot-20250116120000/memory_dump-20250116120000.elf
  /tmp/extracted/snapshot-20250116120030/memory_dump-20250116120030.elf

# Run delta calculation
live_delta_calc \
  /tmp/extracted/snapshot-20250116120000/memory_dump-20250116120000.elf \
  /tmp/extracted/snapshot-20250116120030/memory_dump-20250116120030.elf \
  /path/to/output
```

## Tools

### `test_borg_snapshot.py`

Main Python script with full functionality:

```bash
# Create snapshot
python3 test_borg_snapshot.py --monitor-socket <socket> --borg-repo <repo> --create-snapshot

# List snapshots
python3 test_borg_snapshot.py --borg-repo <repo> --list-snapshots

# Get metrics
python3 test_borg_snapshot.py --borg-repo <repo> --metrics

# Extract snapshots
python3 test_borg_snapshot.py --borg-repo <repo> --get-snapshots <arch1> <arch2> --output-dir <dir>
```

### `borg_helper.sh`

Convenience wrapper for common operations:

```bash
./borg_helper.sh <command> [options]

Commands:
  create <monitor-socket> <borg-repo>
  list <borg-repo>
  metrics <borg-repo>
  get-consecutive <borg-repo> <index> [output-dir]
  get-pair <borg-repo> <archive1> <archive2> [output-dir]
  get-latest <borg-repo> [output-dir]
```

## Integration with Capture Pipeline

You can integrate BORG storage into the capture pipeline:

1. **Option A**: Modify `capture_producer_qemu_mac.sh` to store snapshots in BORG instead of (or in addition to) local disk
2. **Option B**: Run a separate process that periodically moves snapshots from `imageDir` to BORG
3. **Option C**: Use BORG as the primary storage and extract on-demand for delta calculation

Example integration (Option B - background archiver):

```bash
# In a separate terminal/process
while true; do
  # Wait for new snapshots
  for dump in /path/to/imageDir/memory_dump-*.elf; do
    if [[ -f "$dump" ]]; then
      # Create BORG archive
      python3 test_borg_snapshot.py \
        --borg-repo /path/to/borg_repo \
        --create-snapshot \
        --source-file "$dump"
      
      # Optionally remove original after archiving
      # rm "$dump"
    fi
  done
  sleep 60
done
```

## BORG Benefits

- **Compression**: Typically 2-5x size reduction for memory dumps
- **Deduplication**: Consecutive snapshots share most pages → huge space savings
- **Versioning**: All snapshots preserved with timestamps
- **Integrity**: Checksums and error detection
- **Efficiency**: Only changed chunks stored

## Example Workflow

```bash
# 1. Initialize repository
borg init --encryption=none /tmp/vm_snapshots

# 2. Create several snapshots (e.g., every 30 seconds)
for i in {1..10}; do
  ./borg_helper.sh create /tmp/qemu-monitor.sock /tmp/vm_snapshots
  sleep 30
done

# 3. Check metrics
./borg_helper.sh metrics /tmp/vm_snapshots
# Output:
#   Total archives: 10
#   Cache size: 2.5 GB
#   Unique chunks: 15000
#   Total snapshot size (uncompressed): 40.0 GB
#   (Actual repo size might be only 5-10 GB due to compression + dedup)

# 4. List snapshots
./borg_helper.sh list /tmp/vm_snapshots
# Output:
#   [0] snapshot-20250116120000 - 2025-01-16T12:00:00 - 4.0 GB
#   [1] snapshot-20250116120030 - 2025-01-16T12:00:30 - 4.0 GB
#   ...

# 5. Extract consecutive pair for delta calculation
./borg_helper.sh get-consecutive /tmp/vm_snapshots 0 /tmp/extracted
# Output:
#   [SUCCESS] Snapshots extracted. Use these paths for delta calculation:
#     /tmp/extracted/snapshot-20250116120000/memory_dump-20250116120000.elf
#     /tmp/extracted/snapshot-20250116120030/memory_dump-20250116120030.elf

# 6. Run delta calculation
live_delta_calc \
  /tmp/extracted/snapshot-20250116120000/memory_dump-20250116120000.elf \
  /tmp/extracted/snapshot-20250116120030/memory_dump-20250116120030.elf \
  /tmp/delta_output
```

## Notes

- **Encryption**: Repository initialized with `--encryption=none` for speed. Add encryption if needed: `borg init --encryption=repokey /path/to/repo`
- **Cleanup**: Extracted snapshots are temporary; delete `/tmp/extracted` after delta calculation
- **Performance**: BORG extraction is fast (only decompresses needed chunks)
- **Storage**: BORG repos can be moved/copied; use `borg check` to verify integrity
