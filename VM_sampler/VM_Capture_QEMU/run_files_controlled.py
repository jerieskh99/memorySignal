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
import sys
import threading
import time
import json
from pathlib import Path
import re


VIRSH_URI = os.environ.get("VIRSH_URI", "qemu:///system")
VM_DOMAIN = os.environ.get("VM_DOMAIN", "Kali Jeries")
SSH_TARGET = os.environ.get("SSH_TARGET", "")  # required: user@host
SSH_KEY = os.environ.get("SSH_KEY", "")
SSH_OPTS = os.environ.get("SSH_OPTS", "")
# SSH_PASS: plaintext password passed to sshpass so no interactive prompt appears.
# Example: SSH_PASS=kali python3 run_files_controlled.py ...
# Requires sshpass to be installed: apt install sshpass
SSH_PASS = os.environ.get("SSH_PASS", "")
SSH_WAIT_TIMEOUT = int(os.environ.get("SSH_WAIT_TIMEOUT", "1200000"))
STOP_TIMEOUT = int(os.environ.get("STOP_TIMEOUT", "60"))
FORCE_DESTROY = os.environ.get("FORCE_DESTROY", "1").lower() in {"1", "true", "yes"}
STEPS_FILE = os.environ.get("STEPS_FILE", "")
TEST_EXEC_SECONDS = int(os.environ.get("TEST_EXEC_SECONDS", "300"))

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
CAPTURE_WARMUP_SECONDS = int(os.environ.get("CAPTURE_WARMUP_SECONDS", "0"))

# ---------------------------------------------------------------------------
# Step-gated offline metrics
# ---------------------------------------------------------------------------
# OFFLINE_METRICS_MODE=1 enables running offline_step_metrics.py after each step.
OFFLINE_METRICS_MODE = os.environ.get("OFFLINE_METRICS_MODE", "0").lower() in {"1", "true", "yes"}
OFFLINE_METRICS_SCRIPT = os.environ.get(
    "OFFLINE_METRICS_SCRIPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_step_metrics.py"),
)
# Path to the memorySignal repo root so offline_step_metrics.py can import
# coherence_temp_spec_stability modules.  Falls back to streaming.projectRoot
# from the capture config if not set explicitly.
OFFLINE_PROJECT_ROOT = os.environ.get("OFFLINE_PROJECT_ROOT", "")
# Directory where baseline_plv.npy is persisted across steps.
# Defaults to <outputDir>/offline/baseline derived at runtime.
OFFLINE_BASELINE_DIR = os.environ.get("OFFLINE_BASELINE_DIR", "")
OFFLINE_WINDOW_SIZE = int(os.environ.get("OFFLINE_WINDOW_SIZE", "128"))
OFFLINE_STEP_SIZE = int(os.environ.get("OFFLINE_STEP_SIZE", "64"))
# If set, overrides the offline output root (default: same as capture outputDir).
OFFLINE_OUTPUT_ROOT = os.environ.get("OFFLINE_OUTPUT_ROOT", "")
# Which step number (1-based) is treated as the clean idle baseline.
BASELINE_STEP_NUMBER = int(os.environ.get("BASELINE_STEP_NUMBER", "1"))
# How long to wait for the consumer queue to drain before timing out (minutes).
QUEUE_DRAIN_TIMEOUT_MINUTES = int(os.environ.get("QUEUE_DRAIN_TIMEOUT_MINUTES", "30"))
PROGRESS_BAR = os.environ.get("PROGRESS_BAR", "1").lower() in {"1", "true", "yes"}


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:d}m{s:02d}s"


def _read_vm_state_file() -> str:
    """Read vm_state.txt written by the producer — 'paused' or 'running'.

    Zero virsh calls; never blocks the main thread. Returns empty string if
    the file does not exist yet (treated as running by the spinner).
    """
    try:
        q = capture_queue_dir()
        if q is None:
            return ""
        p = q / "vm_state.txt"
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8").strip().lower()
    except Exception:
        return ""


class _WorkloadSpinner:
    """Spinner that tracks VM-active vs paused time via producer state file.

    Reads queueDir/vm_state.txt written by the producer on every suspend/resume.
    No virsh calls from the spinner thread — zero race with SSH or capture.
    """

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last = 0.0
        self.active_secs = 0.0
        self.paused_secs = 0.0

    def _loop(self) -> None:
        glyphs = ("|", "/", "-", "\\")
        i = 0
        while not self._stop.is_set():
            now = time.time()
            dt = max(0.0, now - self._last)
            self._last = now
            state = _read_vm_state_file()
            if state == "paused":
                self.paused_secs += dt
            else:
                self.active_secs += dt
            wall = self.active_secs + self.paused_secs
            try:
                sys.stderr.write(
                    f"\r[PROGRESS] {self.label} {glyphs[i % len(glyphs)]} "
                    f"active={_fmt_hms(self.active_secs)} "
                    f"paused={_fmt_hms(self.paused_secs)} "
                    f"wall={_fmt_hms(wall)}   "
                )
                sys.stderr.flush()
            except Exception:
                return
            i += 1
            if self._stop.wait(1.0):
                break

    def __enter__(self) -> "_WorkloadSpinner":
        self._last = time.time()
        if PROGRESS_BAR and sys.stderr.isatty():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        elif PROGRESS_BAR:
            print(f"[PROGRESS] {self.label}: running")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        wall = self.active_secs + self.paused_secs
        if PROGRESS_BAR and sys.stderr.isatty():
            try:
                sys.stderr.write("\r" + " " * 140 + "\r")
                sys.stderr.flush()
            except Exception:
                pass
        if PROGRESS_BAR:
            print(
                f"[PROGRESS] {self.label}: finished in {_fmt_hms(wall)}"
                f"  (active={_fmt_hms(self.active_secs)} paused={_fmt_hms(self.paused_secs)})"
            )


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
    parts: list[str] = []
    if SSH_PASS:
        # Prepend sshpass so SSH never shows an interactive password prompt.
        parts += ["sshpass", "-p", SSH_PASS]
    parts += ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"]
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


def start_capture(run_matrix_path: str = "") -> tuple[int, list[int]]:
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
    # Per-step matrix path: consumer will append frames here instead of the
    # shared default run_matrix.npy.
    if run_matrix_path:
        # Ensure a clean matrix/stream for this step.
        for p in (
            run_matrix_path,
            f"{run_matrix_path}.frames.bin",
            f"{run_matrix_path}.frames.meta",
        ):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
        env_prefix += f"RUN_MATRIX={shlex.quote(run_matrix_path)} "
    # In step-gated offline mode disable live streaming inside the consumer;
    # offline metrics are computed by offline_step_metrics.py after each step.
    if OFFLINE_METRICS_MODE:
        env_prefix += "OFFLINE_MODE=1 "
    cmd = (
        f"cd {root_q} && "
        f"{env_prefix}CONFIG={cfg_q} PRODUCER_SCRIPT={producer_q} BACKGROUND=1 ./run_qemu_capture.sh"
    )
    print(f"[CONTROL] Starting capture (root={CAPTURE_ROOT})")
    rc = run(cmd) >> 8
    pids: list[int] = []
    pid_file = Path(CAPTURE_ROOT) / "capture_pids.txt"
    time.sleep(0.5)
    if pid_file.exists():
        try:
            lines = pid_file.read_text().strip().splitlines()
            for line in lines[:2]:
                pids.append(int(line.strip()))
        except Exception:
            pass
    return rc, pids


def stop_producer() -> None:
    """Stop only the producer so the consumer can keep draining the queue."""
    print("[CONTROL] Stopping capture producer...")
    run("pkill -f capture_producer_qemu_pmemsave.sh >/dev/null 2>&1")


def stop_consumer() -> None:
    """Stop only the consumer (call after queue is fully drained)."""
    print("[CONTROL] Stopping capture consumer...")
    run("pkill -f capture_consumer_qemu.sh >/dev/null 2>&1")


def stop_capture() -> None:
    stop_producer()
    stop_consumer()


def capture_queue_dir() -> Path | None:
    """Return the queueDir path from the active capture config."""
    try:
        with open(CAPTURE_CONFIG, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        q = cfg.get("queueDir", "")
        return Path(q) if q else None
    except Exception:
        return None


def step_run_matrix_path(test_name: str) -> str:
    """Return a per-step run_matrix path so each test has its own isolated matrix."""
    q = capture_queue_dir()
    if q is None:
        return ""
    return str(q / f"run_matrix_{test_name}.npy")


def wait_for_queue_drain(timeout_minutes: int = 30) -> bool:
    """Poll pending+processing until both reach zero or timeout expires.

    Returns True when the queue is empty, False on timeout.
    """
    q = capture_queue_dir()
    if q is None:
        print("[CONTROL] WARNING: cannot read queueDir from config; skipping drain wait.")
        return True
    pending = q / "pending"
    processing = q / "processing"
    deadline = time.time() + timeout_minutes * 60
    poll_interval = 5
    last_report = 0.0
    while time.time() < deadline:
        n_pending = len(list(pending.glob("*.json"))) if pending.is_dir() else 0
        n_processing = len(list(processing.glob("*.json"))) if processing.is_dir() else 0
        if n_pending + n_processing == 0:
            print("[CONTROL] Queue fully drained (pending=0, processing=0).")
            return True
        now = time.time()
        if now - last_report >= 15:
            remaining = int(deadline - now)
            print(
                f"[CONTROL] Waiting for queue drain: pending={n_pending},"
                f" processing={n_processing}, timeout_remaining={remaining}s"
            )
            last_report = now
        time.sleep(poll_interval)
    n_pending = len(list(pending.glob("*.json"))) if pending.is_dir() else 0
    n_processing = len(list(processing.glob("*.json"))) if processing.is_dir() else 0
    print(
        f"[CONTROL] TIMEOUT: queue not fully drained after {timeout_minutes} min."
        f" pending={n_pending}, processing={n_processing}"
    )
    return False


def _resolve_offline_project_root() -> str:
    """Return project root for offline metrics, from env or streaming config."""
    if OFFLINE_PROJECT_ROOT:
        return OFFLINE_PROJECT_ROOT
    try:
        with open(CAPTURE_CONFIG, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg.get("streaming", {}).get("projectRoot", "")
    except Exception:
        return ""


def run_offline_step_metrics(step_name: str, matrix_path: str, is_baseline: bool) -> int:
    """Invoke offline_step_metrics.py for one completed test step.

    Returns the exit code (0 = success; non-zero logged as warning but not fatal
    unless you want to be strict).
    """
    if not os.path.isfile(OFFLINE_METRICS_SCRIPT):
        print(
            f"[CONTROL] WARNING: offline_step_metrics.py not found at"
            f" {OFFLINE_METRICS_SCRIPT}; skipping offline metrics."
        )
        return 0
    if not os.path.isfile(matrix_path):
        print(
            f"[CONTROL] WARNING: step matrix not found: {matrix_path};"
            f" skipping offline metrics for {step_name}."
        )
        return 0

    project_root = _resolve_offline_project_root()
    if not project_root:
        print("[CONTROL] WARNING: OFFLINE_PROJECT_ROOT not set; skipping offline metrics.")
        return 0

    out_dir = capture_output_dir()
    output_root = OFFLINE_OUTPUT_ROOT or (str(out_dir) if out_dir else "")
    if not output_root:
        print("[CONTROL] WARNING: cannot resolve output root for offline metrics; skipping.")
        return 0

    baseline_dir = OFFLINE_BASELINE_DIR or (
        os.path.join(output_root, "offline", "baseline")
    )

    cmd_parts = [
        "python3", OFFLINE_METRICS_SCRIPT,
        "--matrix", matrix_path,
        "--step-name", step_name,
        "--output-root", output_root,
        "--project-root", project_root,
        "--baseline-dir", baseline_dir,
        "--window-size", str(OFFLINE_WINDOW_SIZE),
        "--step-size", str(OFFLINE_STEP_SIZE),
    ]
    if is_baseline:
        cmd_parts.append("--is-baseline")
    cmd = " ".join(shlex.quote(p) for p in cmd_parts)
    print(f"[CONTROL] Offline step metrics: step={step_name} is_baseline={is_baseline}")
    rc = run(cmd) >> 8
    if rc != 0:
        print(
            f"[CONTROL] WARNING: offline_step_metrics.py exited rc={rc}"
            f" for step {step_name} (non-fatal)."
        )
    return rc


def finalize_run_matrix_from_stream(matrix_path: str) -> None:
    """Build run_matrix.npy once from append-only frame stream.

    Stream layout: concatenated float64 frame vectors [num_pages] in append order.
    Meta file stores num_pages.
    """
    stream_path = f"{matrix_path}.frames.bin"
    meta_path = f"{matrix_path}.frames.meta"
    if not os.path.isfile(stream_path):
        return

    cmd_parts = [
        "python3",
        "-c",
        (
            "import os,sys,numpy as np;"
            "stream,meta,out=sys.argv[1],sys.argv[2],sys.argv[3];"
            "data=np.fromfile(stream,dtype=np.float64);"
            "n=int(open(meta,'r',encoding='utf-8').read().strip()) if os.path.isfile(meta) else 0;"
            "assert n>0, f'missing/invalid meta pages: {meta}';"
            "assert data.size % n == 0, f'stream size {data.size} not divisible by pages {n}';"
            "t=data.size//n;"
            "mat=data.reshape(t,n).T;"
            "np.save(out,mat);"
            "print(f'[CONTROL] Finalized matrix from stream: T={t} N={n} -> {out}')"
        ),
        stream_path,
        meta_path,
        matrix_path,
    ]
    run(" ".join(shlex.quote(p) for p in cmd_parts))


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


def pause_capture_processes(capture_pids: list[int] | None = None) -> list[int]:
    if capture_pids:
        # Only pause PIDs we started; filter to those still running
        alive = [p for p in capture_pids if _pid_alive(p)]
        pids = alive
    else:
        pids = _capture_process_pids()
    if pids:
        run("kill -STOP " + " ".join(str(p) for p in pids) + " >/dev/null 2>&1")
        print(f"[CONTROL] Paused capture processes: {pids}")
    return pids


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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
        f"python3 ~/memorySignal/VM_executables/mem_stream.py --mb 128 --seconds {TEST_EXEC_SECONDS}",
        f"python3 ~/memorySignal/VM_executables/mem_pointer_chase.py --mb 128 --seconds {TEST_EXEC_SECONDS} --seed 123",
        f"python3 ~/memorySignal/VM_executables/mem_alloc_touch_pages.py --objects 256 --object-kb 256 --sleep-ms 20 --seconds {TEST_EXEC_SECONDS}",
        f"python3 ~/memorySignal/VM_executables/io_seq_fsync.py --seconds {TEST_EXEC_SECONDS} --kb 4096 --fsync-wait 1 --path io_seq.bin",
        f"python3 ~/memorySignal/VM_executables/io_rand_rw.py --seconds {TEST_EXEC_SECONDS} --file-mb 2048 --block-kb 64 --write-ratio 0.5 --path io_rand.bin --seed 123",
        f"python3 ~/memorySignal/VM_executables/io_many_files.py --seconds {TEST_EXEC_SECONDS} --files-per-batch 500 --payload-bytes 1024 --seed 123",
    ]


def main() -> int:
    if not SSH_TARGET:
        print("ERROR: SSH_TARGET is required (e.g. SSH_TARGET=user@vm-ip).")
        return 1

    steps = load_steps()
    if not steps:
        print("ERROR: no steps to run.")
        return 1

    print(
        f"[CONTROL] VM_DOMAIN={VM_DOMAIN}  SSH_TARGET={SSH_TARGET}"
        f"  STEPS={len(steps)}  CAPTURE_MODE={CAPTURE_MODE}"
        f"  OFFLINE_METRICS_MODE={OFFLINE_METRICS_MODE}"
    )
    base = ssh_base()

    for i, remote_cmd in enumerate(steps, start=1):
        test_label = step_name_from_command(remote_cmd)
        test_name = f"test{i}_{test_label}"
        is_baseline_step = (i == BASELINE_STEP_NUMBER)
        print(f"\n[CONTROL] ===== Step {i}/{len(steps)} : {test_name} =====")
        print(f"[CONTROL] Command : {remote_cmd}")
        if OFFLINE_METRICS_MODE and is_baseline_step:
            print("[CONTROL] (this step will produce the shared PLV baseline)")

        ensure_vm_running()

        # Derive a step-specific matrix path so each test's frames are isolated.
        step_matrix = ""
        if CAPTURE_MODE:
            step_matrix = step_run_matrix_path(test_name)
            cap_rc, _ = start_capture(run_matrix_path=step_matrix)
            if cap_rc != 0:
                print(f"[CONTROL] ERROR: failed to start capture (exit={cap_rc}).")
                return cap_rc
            if CAPTURE_WARMUP_SECONDS > 0:
                print(f"[CONTROL] Capture warmup: sleeping {CAPTURE_WARMUP_SECONDS}s")
                time.sleep(CAPTURE_WARMUP_SECONDS)

        if not wait_for_ssh():
            print(f"[CONTROL] ERROR: SSH did not become reachable within {SSH_WAIT_TIMEOUT}s.")
            return 1

        print("[CONTROL] Running command over SSH...")
        with _WorkloadSpinner(f"step {i}/{len(steps)} {test_name}"):
            rc = run(f"{base} {shlex.quote(remote_cmd)}")
        rc = rc >> 8  # os.system stores wait status
        print(f"[CONTROL] SSH command exit code: {rc}")

        if CAPTURE_MODE:
            # 1. Stop the producer immediately — no more new dumps for this step.
            stop_producer()

            # 2. Wait for the consumer to finish processing all queued jobs.
            #    The consumer keeps running and draining; we just poll until done.
            drained = wait_for_queue_drain(timeout_minutes=QUEUE_DRAIN_TIMEOUT_MINUTES)
            if not drained:
                print(
                    f"[CONTROL] ERROR: Queue drain timeout at step {i} ({test_name})."
                    " Stopping run to avoid contaminating subsequent steps."
                )
                stop_consumer()
                stop_vm()
                return 1

            # 3. Now that the queue is empty it is safe to stop the consumer.
            stop_consumer()

            # 3.5 Offline-mode append-only stream -> finalize matrix once.
            if OFFLINE_METRICS_MODE and step_matrix:
                finalize_run_matrix_from_stream(step_matrix)

            # 4. Run offline metrics for this isolated, fully-drained step.
            if OFFLINE_METRICS_MODE and step_matrix:
                run_offline_step_metrics(
                    step_name=test_name,
                    matrix_path=step_matrix,
                    is_baseline=is_baseline_step,
                )

            # 5. Rotate cosine/hamming output files under a per-test subfolder.
            rotate_delta_files(test_name)

        stop_vm()

        if rc != 0:
            print(f"[CONTROL] ERROR: Step {i} failed (exit={rc}). Stopping sequence.")
            return rc

    print("\n[CONTROL] All steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

