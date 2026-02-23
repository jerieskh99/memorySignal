#!/usr/bin/env python3
"""
Sanity test: Dump VM memory snapshots to BORG repository and enable retrieval.

This script:
1. Connects to QEMU monitor socket
2. Pauses VM, dumps memory snapshot, resumes VM
3. Stores snapshot in BORG repository with timestamped archive
4. Provides metrics (repository size, snapshot count, etc.)
5. Allows retrieval of snapshots (especially consecutive pairs for delta calc)

Usage:
    python3 test_borg_snapshot.py --monitor-socket /tmp/qemu-monitor.sock --borg-repo /path/to/repo
    python3 test_borg_snapshot.py --list-snapshots --borg-repo /path/to/repo
    python3 test_borg_snapshot.py --get-snapshots <archive1> <archive2> --borg-repo /path/to/repo --output-dir /tmp/extracted
"""

import argparse
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from datetime import datetime
import json
import shutil

try:
    import borgmatic.borg.extract as borg_extract
except ImportError:
    pass


def check_borg():
    """Check if borg is installed."""
    try:
        result = subprocess.run(['borg', '--version'], capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, None


def qemu_monitor_cmd(socket_path: str, command: str) -> tuple[str, bool]:
    """Send command to QEMU monitor socket and return response."""
    # Try socat first, then nc
    for cmd_tool in ['socat', 'nc']:
        try:
            proc = subprocess.run(
                [cmd_tool, '-', f'UNIX-CONNECT:{socket_path}'],
                input=command + '\n',
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                return proc.stdout, True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "", False


def get_vm_status(socket_path: str) -> str:
    """Get VM status from QEMU monitor."""
    output, success = qemu_monitor_cmd(socket_path, "info status")
    if not success:
        return "unknown"
    # Parse: "VM status: paused" or "VM status: running"
    for line in output.split('\n'):
        if 'VM status:' in line:
            status = line.split('VM status:')[1].strip().lower()
            if 'paused' in status:
                return "paused"
            elif 'running' in status:
                return "running"
    return "unknown"


def wait_state(socket_path: str, desired_state: str, timeout_sec: int = 30) -> bool:
    """Wait for VM to reach desired state."""
    import time
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        state = get_vm_status(socket_path)
        if state == desired_state:
            return True
        time.sleep(0.2)
    return False


def dump_memory(socket_path: str, output_path: str) -> bool:
    """Dump VM memory to file via QEMU monitor."""
    # Pause VM
    qemu_monitor_cmd(socket_path, "stop")
    if not wait_state(socket_path, "paused", timeout_sec=30):
        print(f"[ERROR] Failed to pause VM")
        qemu_monitor_cmd(socket_path, "cont")
        return False

    # Dump memory
    dump_cmd = f'dump-guest-memory elf "{output_path}"'
    output, success = qemu_monitor_cmd(socket_path, dump_cmd)
    if not success:
        print(f"[ERROR] dump-guest-memory failed: {output}")
        qemu_monitor_cmd(socket_path, "cont")
        return False

    # Wait a moment for dump to complete
    import time
    time.sleep(0.5)

    if not os.path.exists(output_path):
        print(f"[ERROR] Dump file not created: {output_path}")
        qemu_monitor_cmd(socket_path, "cont")
        return False

    # Resume VM
    qemu_monitor_cmd(socket_path, "cont")
    wait_state(socket_path, "running", timeout_sec=30)
    return True


def borg_init_repo(repo_path: str) -> bool:
    """Initialize BORG repository if it doesn't exist."""
    if os.path.exists(repo_path):
        return True
    try:
        subprocess.run(['borg', 'init', '--encryption=none', repo_path], check=True, capture_output=True)
        print(f"[INFO] Initialized BORG repository: {repo_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to init BORG repo: {e.stderr.decode()}")
        return False


def borg_create_archive(repo_path: str, archive_name: str, source_path: str) -> bool:
    """Create BORG archive from source path."""
    try:
        cmd = ['borg', 'create', '--stats', f'{repo_path}::{archive_name}', source_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[INFO] Created archive: {archive_name}")
        # Print stats
        for line in result.stdout.split('\n'):
            if 'Original size' in line or 'Compressed size' in line or 'Deduplicated size' in line:
                print(f"  {line.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create archive: {e.stderr}")
        return False


def borg_list_archives(repo_path: str) -> list[dict]:
    """List all archives in BORG repository."""
    try:
        # Try JSON first, fallback to text parsing
        cmd = ['borg', 'list', '--json', repo_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        try:
            data = json.loads(result.stdout)
            archives = []
            for arch in data.get('archives', []):
                archives.append({
                    'name': arch['name'],
                    'time': arch.get('time', 0),
                    'size': arch.get('size', 0)
                })
            return sorted(archives, key=lambda x: x['time'])
        except json.JSONDecodeError:
            # Fallback: parse text output
            cmd = ['borg', 'list', repo_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            archives = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        arch_name = parts[0]
                        # Try to parse date/time from line
                        time_str = ' '.join(parts[1:4]) if len(parts) >= 4 else ''
                        archives.append({
                            'name': arch_name,
                            'time': 0,  # Can't parse from text easily
                            'size': 0
                        })
            return archives
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to list archives: {e.stderr.decode() if e.stderr else str(e)}")
        return []


def borg_info(repo_path: str) -> dict:
    """Get BORG repository info."""
    try:
        cmd = ['borg', 'info', '--json', repo_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to get repo info: {e}")
        return {}


def borg_extract_archives(repo_path: str, archive_names: list[str], output_dir: str) -> bool:
    """Extract specific archives to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    success = True
    for arch_name in archive_names:
        try:
            # Extract to subdirectory named after archive
            arch_output = os.path.join(output_dir, arch_name)
            os.makedirs(arch_output, exist_ok=True)
            # Extract archive contents (without --strip-components to preserve structure)
            cmd = ['borg', 'extract', f'{repo_path}::{arch_name}']
            result = subprocess.run(cmd, cwd=arch_output, check=True, capture_output=True, text=True)
            print(f"[INFO] Extracted {arch_name} -> {arch_output}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            print(f"[ERROR] Failed to extract {arch_name}: {error_msg}")
            success = False
    return success


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description='BORG-backed VM memory snapshot management')
    parser.add_argument('--monitor-socket', type=str, help='QEMU monitor socket path')
    parser.add_argument('--borg-repo', type=str, required=True, help='BORG repository path')
    parser.add_argument('--list-snapshots', action='store_true', help='List all snapshots')
    parser.add_argument('--get-snapshots', nargs='+', metavar='ARCHIVE', help='Extract specific snapshots')
    parser.add_argument('--output-dir', type=str, help='Output directory for extraction')
    parser.add_argument('--create-snapshot', action='store_true', help='Create a new snapshot')
    parser.add_argument('--metrics', action='store_true', help='Show repository metrics')
    args = parser.parse_args()

    # Check BORG
    borg_ok, borg_version = check_borg()
    if not borg_ok:
        print("[ERROR] BORG not found. Install with: brew install borgbackup")
        sys.exit(1)
    print(f"[INFO] Using BORG: {borg_version}")

    # Initialize repo if needed
    if not borg_init_repo(args.borg_repo):
        sys.exit(1)

    # List snapshots
    if args.list_snapshots:
        archives = borg_list_archives(args.borg_repo)
        if not archives:
            print("[INFO] No snapshots found")
        else:
            print(f"\n[INFO] Found {len(archives)} snapshots:")
            for i, arch in enumerate(archives):
                time_str = datetime.fromtimestamp(arch['time']).isoformat() if arch['time'] > 0 else 'N/A'
                size_str = format_size(arch.get('size', 0)) if arch.get('size', 0) > 0 else 'N/A'
                print(f"  [{i}] {arch['name']} - {time_str} - {size_str}")
        return

    # Get metrics
    if args.metrics:
        info = borg_info(args.borg_repo)
        archives = borg_list_archives(args.borg_repo)
        print(f"\n[BORG Repository Metrics]")
        print(f"  Repository: {args.borg_repo}")
        print(f"  Total archives: {len(archives)}")
        if info:
            cache = info.get('cache', {})
            print(f"  Cache size: {format_size(cache.get('size', 0))}")
            print(f"  Unique chunks: {cache.get('stats', {}).get('total_unique_chunks', 0)}")
        if archives:
            total_size = sum(a.get('size', 0) for a in archives)
            print(f"  Total snapshot size (uncompressed): {format_size(total_size)}")
        return

    # Extract snapshots
    if args.get_snapshots:
        if not args.output_dir:
            print("[ERROR] --output-dir required for extraction")
            sys.exit(1)
        if len(args.get_snapshots) < 1:
            print("[ERROR] Specify at least one archive to extract")
            sys.exit(1)
        success = borg_extract_archives(args.borg_repo, args.get_snapshots, args.output_dir)
        if success:
            print(f"\n[SUCCESS] Extracted {len(args.get_snapshots)} snapshot(s) to {args.output_dir}")
            print(f"  Use these paths for delta calculation:")
            for arch_name in args.get_snapshots:
                arch_path = os.path.join(args.output_dir, arch_name)
                # Find the .elf file inside
                for root, dirs, files in os.walk(arch_path):
                    for f in files:
                        if f.endswith('.elf'):
                            print(f"    {os.path.join(root, f)}")
        else:
            sys.exit(1)
        return

    # Create snapshot
    if args.create_snapshot:
        if not args.monitor_socket:
            print("[ERROR] --monitor-socket required for creating snapshot")
            sys.exit(1)
        if not os.path.exists(args.monitor_socket):
            print(f"[ERROR] Monitor socket not found: {args.monitor_socket}")
            sys.exit(1)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_name = f"snapshot-{timestamp}"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = os.path.join(tmpdir, f"memory_dump-{timestamp}.elf")
            print(f"[INFO] Dumping memory to {dump_path}...")
            
            if not dump_memory(args.monitor_socket, dump_path):
                print("[ERROR] Failed to dump memory")
                sys.exit(1)
            
            dump_size = os.path.getsize(dump_path)
            print(f"[INFO] Dump size: {format_size(dump_size)}")
            
            print(f"[INFO] Creating BORG archive: {archive_name}...")
            if borg_create_archive(args.borg_repo, archive_name, dump_path):
                print(f"[SUCCESS] Snapshot saved: {archive_name}")
            else:
                sys.exit(1)
        return

    # Default: show help or create snapshot if monitor socket provided
    if args.monitor_socket:
        # Create snapshot by default if monitor socket is provided
        args.create_snapshot = True
        main()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
