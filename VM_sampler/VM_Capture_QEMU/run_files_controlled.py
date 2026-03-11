#!/usr/bin/env python3
"""
Controlled VM workload runner (host-side orchestration) using os.system.

Flow per step:
  1) Ensure VM is running
  2) Wait for SSH reachability
  3) Run one command on VM over SSH and wait until it finishes
  4) Stop VM
  5) Move to next command
"""

import os
import shlex
import time
import json
from pathlib import Path
import re


VIRSH_URI = os.environ.get("VIRSH_URI", "qemu:///system")
VM_DOMAIN = os.environ.get("VM_DOMAIN", "Kali Jeries")
SSH_TARGET = os.environ.get("SSH_TARGET", "")  # required: user@host
SSH_KEY = os.environ.get("SSH_KEY", "")
SSH_OPTS = os.environ.get("SSH_OPTS", "")
SSH_WAIT_TIMEOUT = int(os.environ.get("SSH_WAIT_TIMEOUT", "1200"))
STOP_TIMEOUT = int(os.environ.get("STOP_TIMEOUT", "60"))
FORCE_DESTROY = os.environ.get("FORCE_DESTROY", "1").lower() in {"1", "true", "yes"}
STEPS_FILE = os.environ.get("STEPS_FILE", "")

# Optional capture mode: start/stop capture around each workload step.
CAPTURE_MODE = os.environ.get("CAPTURE_MODE", "0").lower() in {"1", "true", "yes"}
CAPTURE_ROOT = os.environ.get(
    "CAPTURE_ROOT",
    os.path.dirname(os.path.abspath(__file__)),
)
CAPTURE_CONFIG = os.environ.get("CAPTURE_CONFIG", os.path.join(CAPTURE_ROOT, "config_qemu_upc.json"))
CAPTURE_PRODUCER_SCRIPT = os.environ.get(
    "CAPTURE_PRODUCER_SCRIPT",
    os.path.join(CAPTURE_ROOT, "capture_producer_qemu_pmemsave.sh"),
)
CAPTURE_WARMUP_SECONDS = int(os.environ.get("CAPTURE_WARMUP_SECONDS", "2"))


def run(cmd: str) -> int:
    return os.system(cmd)


def virsh_state() -> str:
    out_path = "/tmp/vm_state.txt"
    cmd = (
        f"virsh -c {shlex.quote(VIRSH_URI)} domstate {shlex.quote(VM_DOMAIN)} "
        f"> {shlex.quote(out_path)} 2>/dev/null"
    )
    rc = run(cmd)
    if rc != 0:
        return ""
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            return f.read().strip().replace("\r", "")
    except Exception:
        return ""


def ensure_vm_running() -> None:
    state = virsh_state().lower()
    if "running" in state:
        return
    if "paused" in state:
        print(f"[CONTROL] VM paused -> resuming: {VM_DOMAIN}")
        run(f"virsh -c {shlex.quote(VIRSH_URI)} resume {shlex.quote(VM_DOMAIN)} >/dev/null")
        return
    print(f"[CONTROL] VM not running -> starting: {VM_DOMAIN}")
    run(f"virsh -c {shlex.quote(VIRSH_URI)} start {shlex.quote(VM_DOMAIN)} >/dev/null")


def ssh_base() -> str:
    parts = ["ssh", "-o", "BatchMode=no", "-o", "ConnectTimeout=5"]
    if SSH_KEY:
        parts += ["-i", SSH_KEY]
    if SSH_OPTS:
        parts += SSH_OPTS.split()
    parts.append(SSH_TARGET)
    return " ".join(shlex.quote(p) for p in parts)


def wait_for_ssh() -> bool:
    base = ssh_base()
    deadline = time.time() + SSH_WAIT_TIMEOUT
    while time.time() < deadline:
        rc = run(f"{base} 'echo ok' >/dev/null 2>&1")
        if rc == 0:
            return True
        time.sleep(2)
    return False


def stop_vm() -> None:
    print(f"[CONTROL] Stopping VM: {VM_DOMAIN}")
    run(f"virsh -c {shlex.quote(VIRSH_URI)} shutdown {shlex.quote(VM_DOMAIN)} >/dev/null 2>&1")
    deadline = time.time() + STOP_TIMEOUT
    while time.time() < deadline:
        state = virsh_state().lower()
        if "shut off" in state or "shut" in state or "off" in state:
            print("[CONTROL] VM stopped.")
            return
        time.sleep(2)
    if FORCE_DESTROY:
        print("[CONTROL] Graceful stop timed out -> force destroy.")
        run(f"virsh -c {shlex.quote(VIRSH_URI)} destroy {shlex.quote(VM_DOMAIN)} >/dev/null 2>&1")
    else:
        print("[CONTROL] WARNING: VM may still be running (graceful stop timed out).")


def start_capture() -> int:
    root_q = shlex.quote(CAPTURE_ROOT)
    cfg_q = shlex.quote(CAPTURE_CONFIG)
    producer_q = shlex.quote(CAPTURE_PRODUCER_SCRIPT)
    borg_mode = os.environ.get("BORG", "")
    borg_repo = os.environ.get("BORG_REPO", "")
    borg_pass = os.environ.get("BORG_PASSPHRASE", "")
    env_prefix = ""
    if borg_mode:
        env_prefix += f"BORG={shlex.quote(borg_mode)} "
    if borg_repo:
        env_prefix += f"BORG_REPO={shlex.quote(borg_repo)} "
    if borg_pass:
        env_prefix += f"BORG_PASSPHRASE={shlex.quote(borg_pass)} "
    cmd = (
        f"cd {root_q} && "
        f"{env_prefix}CONFIG={cfg_q} PRODUCER_SCRIPT={producer_q} BACKGROUND=1 ./run_qemu_capture.sh"
    )
    print(f"[CONTROL] Starting capture (root={CAPTURE_ROOT})")
    return run(cmd) >> 8


def stop_capture() -> None:
    # Keep this narrow to this capture pipeline scripts.
    print("[CONTROL] Stopping capture producer/consumer...")
    run("pkill -f capture_producer_qemu_pmemsave.sh >/dev/null 2>&1")
    run("pkill -f capture_consumer_qemu.sh >/dev/null 2>&1")


def _capture_process_pids() -> list[int]:
    pids: set[int] = set()
    for pat in ("capture_producer_qemu_pmemsave.sh", "capture_consumer_qemu.sh"):
        # pgrep returns one pid per line; ignore non-zero exits when not found.
        out = os.popen(f"pgrep -f {shlex.quote(pat)} 2>/dev/null").read().strip()
        if not out:
            continue
        for line in out.splitlines():
            try:
                pids.add(int(line.strip()))
            except Exception:
                pass
    return sorted(pids)


def pause_capture_processes() -> list[int]:
    pids = _capture_process_pids()
    if pids:
        run("kill -STOP " + " ".join(str(p) for p in pids) + " >/dev/null 2>&1")
        print(f"[CONTROL] Paused capture processes: {pids}")
    return pids


def resume_capture_processes(pids: list[int]) -> None:
    if pids:
        run("kill -CONT " + " ".join(str(p) for p in pids) + " >/dev/null 2>&1")
        print(f"[CONTROL] Resumed capture processes: {pids}")


def capture_output_dir() -> Path | None:
    try:
        with open(CAPTURE_CONFIG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        out = cfg.get("outputDir", "")
        if not out:
            return None
        return Path(out)
    except Exception:
        return None


def rotate_delta_files(test_name: str) -> None:
    out_dir = capture_output_dir()
    if out_dir is None:
        print("[CONTROL] WARNING: could not resolve outputDir from CAPTURE_CONFIG; skipping rotation.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep producer output layout intact: Rust expects output_dir/{hamming,cosine}/ as directories.
    # We rotate only files *inside* these directories into:
    #   output_dir/rotated/<test_name>/{hamming,cosine}/
    rotated_root = out_dir / "rotated" / test_name
    for metric in ("hamming", "cosine"):
        src_dir = out_dir / metric
        dst_dir = rotated_root / metric
        src_dir.mkdir(parents=True, exist_ok=True)
        dst_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        for p in src_dir.glob("*.txt"):
            target = dst_dir / p.name
            if target.exists():
                target = dst_dir / f"{p.stem}_{int(time.time())}{p.suffix}"
            p.rename(target)
            moved += 1
        print(f"[CONTROL] Rotated {moved} {metric} file(s) -> {dst_dir}")


def step_name_from_command(remote_cmd: str) -> str:
    # Best effort extraction of script/workload name from command.
    # Examples:
    #   python3 ~/VM_executables/mem_stream.py ... -> mem_stream
    #   bash ~/VM_executables/run_idle.sh --time 30 -> run_idle
    # Falls back to generic "step".
    try:
        tokens = shlex.split(remote_cmd)
    except Exception:
        tokens = remote_cmd.split()

    candidates = [t for t in tokens if t.endswith(".py") or t.endswith(".sh")]
    if candidates:
        name = Path(candidates[0]).stem
    else:
        # fallback: use first token (e.g., custom binary)
        name = Path(tokens[0]).name if tokens else "step"

    # Keep names filesystem-safe and compact
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return safe or "step"


def load_steps() -> list[str]:
    if STEPS_FILE:
        if not os.path.isfile(STEPS_FILE):
            raise FileNotFoundError(f"STEPS_FILE not found: {STEPS_FILE}")
        steps: list[str] = []
        with open(STEPS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                steps.append(s)
        return steps

    # Default sequence (same intent as run_files.sh style workloads).
    return [
        "bash ~/memorySignal/VM_executables/run_idle.sh --time 30",
        "python3 ~/memorySignal/VM_executables/mem_stream.py --mb 128 --seconds 300",
        "python3 ~/memorySignal/VM_executables/mem_pointer_chase.py --mb 1024 --seconds 300 --seed 123",
        "python3 ~/memorySignal/VM_executables/mem_alloc_touch_pages.py --objects 2000 --object-kb 256 --sleep-ms 20 --seconds 300",
        "python3 ~/memorySignal/VM_executables/io_seq_fsync.py --seconds 300 --kb 4096 --fsync-wait 1 --path io_seq.bin",
        "python3 ~/memorySignal/VM_executables/io_rand_rw.py --seconds 300 --file-mb 2048 --block-kb 64 --write-ratio 0.5 --path io_rand.bin --seed 123",
        "python3 ~/memorySignal/VM_executables/io_many_files.py --seconds 300 --files-per-batch 500 --payload-bytes 1024 --seed 123",
    ]


def main() -> int:
    if not SSH_TARGET:
        print("ERROR: SSH_TARGET is required (e.g. SSH_TARGET=user@vm-ip).")
        return 1

    steps = load_steps()
    if not steps:
        print("ERROR: no steps to run.")
        return 1

    print(f"[CONTROL] VM_DOMAIN={VM_DOMAIN}  SSH_TARGET={SSH_TARGET}  STEPS={len(steps)}")
    base = ssh_base()

    for i, remote_cmd in enumerate(steps, start=1):
        test_label = step_name_from_command(remote_cmd)
        test_name = f"test{i}_{test_label}"
        print(f"\n[CONTROL] ===== Step {i}/{len(steps)} =====")
        print(f"[CONTROL] Command: {remote_cmd}")

        ensure_vm_running()
        if not wait_for_ssh():
            print(f"[CONTROL] ERROR: SSH did not become reachable within {SSH_WAIT_TIMEOUT}s.")
            return 1

        if CAPTURE_MODE:
            cap_rc = start_capture()
            if cap_rc != 0:
                print(f"[CONTROL] ERROR: failed to start capture (exit={cap_rc}).")
                return cap_rc
            if CAPTURE_WARMUP_SECONDS > 0:
                print(f"[CONTROL] Capture warmup: sleeping {CAPTURE_WARMUP_SECONDS}s")
                time.sleep(CAPTURE_WARMUP_SECONDS)

        print("[CONTROL] Running command over SSH...")
        rc = run(f"{base} {shlex.quote(remote_cmd)}")
        rc = rc >> 8  # os.system stores wait status
        print(f"[CONTROL] SSH command exit code: {rc}")

        if CAPTURE_MODE:
            paused = pause_capture_processes()
            rotate_delta_files(test_name)
            resume_capture_processes(paused)
            stop_capture()

        stop_vm()

        if rc != 0:
            print(f"[CONTROL] ERROR: Step {i} failed (exit={rc}). Stopping sequence.")
            return rc

    print("\n[CONTROL] All steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

