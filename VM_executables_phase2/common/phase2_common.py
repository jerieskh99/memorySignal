"""phase2_common.py — shared helpers for Python Phase 2 executables.

- Deterministic PRNG byte generation (uses random.Random with fixed seed).
- Sandbox path validation under /tmp or operator-supplied --safe-root.
- Metadata JSON emission.
- Phase markers / structured logging compatible with the C-side format.
- Hard caps mirroring phase2_common.h for security_like_safe tests.

Standard library only; no third-party deps required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import errno
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


VERSION = "phase2-0.1"

SANDBOX_MAX_FILES = 5000
SANDBOX_MAX_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
SANDBOX_MAX_DURATION_S = 600

SAFE_ROOTS = ("/tmp/", "/var/tmp/")


# ---------- Logging ----------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_info(msg: str) -> None:
    print(f"[{_ts()}] [INFO] {msg}", file=sys.stderr, flush=True)


def log_warn(msg: str) -> None:
    print(f"[{_ts()}] [WARN] {msg}", file=sys.stderr, flush=True)


def log_err(msg: str) -> None:
    print(f"[{_ts()}] [ERROR] {msg}", file=sys.stderr, flush=True)


def phase(test_name: str, phase_name: str) -> None:
    print(
        f"[{_ts()}] [PHASE] test={test_name} phase={phase_name}",
        file=sys.stderr,
        flush=True,
    )


def monotonic() -> float:
    return time.monotonic()


# ---------- PRNG ----------

def deterministic_rng(seed: int) -> random.Random:
    """Stable PRNG. Same seed → same byte stream across Python versions
    (random.Random.getrandbits is stable for a given Python release; we use it
    only for byte generation, not for cryptographic purposes)."""
    return random.Random(seed)


def deterministic_bytes(rng: random.Random, length: int) -> bytes:
    """Generate `length` pseudo-random bytes from the supplied rng."""
    return rng.getrandbits(8 * length).to_bytes(length, "little") if length > 0 else b""


# ---------- Sandbox validation ----------

class SandboxError(RuntimeError):
    pass


def validate_sandbox(path: str | os.PathLike, extra_safe_root: str | None = None) -> str:
    """Return realpath if `path` is safely under /tmp, /var/tmp, or
    `extra_safe_root`. Otherwise raise SandboxError. Refuses '..',
    non-absolute paths, and symlink escapes."""
    if not path:
        raise SandboxError("empty sandbox path")
    p = os.fspath(path)
    if not p.startswith("/"):
        raise SandboxError(f"sandbox path not absolute: {p}")
    if "/.." in p or p.startswith("../") or "/.." in p:
        raise SandboxError(f"sandbox path contains '..': {p}")
    parent = os.path.dirname(p)
    if not os.path.isdir(parent):
        raise SandboxError(f"sandbox parent does not exist: {parent}")
    real_parent = os.path.realpath(parent)
    real = os.path.join(real_parent, os.path.basename(p))
    if os.path.lexists(real) and os.path.islink(real):
        raise SandboxError(f"refuse to operate on symlink: {real}")

    # Resolve each safe root via realpath so /tmp on macOS (a symlink to
    # /private/tmp) matches its canonical form.
    resolved_roots = []
    for r in SAFE_ROOTS:
        try:
            rr = os.path.realpath(r)
        except OSError:
            continue
        rr = rr.rstrip("/") + "/"
        resolved_roots.append(rr)
    if extra_safe_root:
        extra = os.path.realpath(extra_safe_root)
        extra = extra.rstrip("/") + "/"
        resolved_roots.append(extra)
    real_with_slash = real if real.endswith("/") else real + "/"
    if not any(real_with_slash.startswith(r) for r in resolved_roots):
        raise SandboxError(
            f"sandbox path not under approved root: {real} (approved: {resolved_roots})"
        )
    return real


def create_sandbox(root: str = "/tmp", seed: int = 0) -> str:
    """Create and return a resolved sandbox dir path."""
    candidate = os.path.join(root, f"phase2_sandbox_{os.getpid()}_{seed}")
    os.makedirs(candidate, mode=0o700, exist_ok=True)
    real = validate_sandbox(candidate, extra_safe_root=root)
    log_info(f"sandbox created: {real}")
    return real


def remove_sandbox(path: str, safe_root: str | None = None) -> bool:
    """Recursively remove the sandbox tree, refusing if validation fails or
    if the tree contains symlinks that point outside itself."""
    try:
        real = validate_sandbox(path, extra_safe_root=safe_root)
    except SandboxError as e:
        log_err(f"refuse to remove unvalidated path: {e}")
        return False
    if not os.path.isdir(real):
        return True
    # Walk and refuse symlinks escaping the sandbox.
    for dirpath, dirs, files in os.walk(real, followlinks=False):
        for name in files + dirs:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                target = os.readlink(full)
                resolved = os.path.realpath(full)
                if not resolved.startswith(real):
                    log_err(f"sandbox contains escaping symlink, refusing cleanup: {full} -> {target}")
                    return False
    shutil.rmtree(real)
    return True


# ---------- Metadata JSON ----------

class Metadata:
    def __init__(self, test_name: str, language: str, output_dir: str | None):
        self.test_name = test_name
        self.language = language
        self.output_dir = output_dir
        self.data: dict = {
            "test_name": test_name,
            "language": language,
            "phase2_version": VERSION,
            "parameters": {},
            "phases": [],
            "files_created": 0,
            "files_touched": 0,
            "bytes_written": 0,
            "bytes_read": 0,
            "start_time": _ts(),
            "end_time": None,
            "sandbox_path": None,
            "known_limitations": [],
        }
        self._t0 = time.monotonic()

    def set_param(self, **kwargs) -> None:
        self.data["parameters"].update(kwargs)

    def add_phase(self, name: str, t_start: float, t_end: float) -> None:
        self.data["phases"].append(
            {"name": name, "t_start_s": round(t_start, 6), "t_end_s": round(t_end, 6)}
        )

    def set(self, key: str, value) -> None:
        self.data[key] = value

    def add_limitation(self, note: str) -> None:
        self.data["known_limitations"].append(note)

    def write(self) -> str | None:
        self.data["end_time"] = _ts()
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, f"{self.test_name}_metadata.json")
            with open(path, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
            return path
        json.dump(self.data, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return None


# ---------- Argument parsing helper ----------

def base_argparser(prog: str, description: str) -> argparse.ArgumentParser:
    """Return a parser with the standard Phase 2 flags. Tests should add
    their test-specific flags on top."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("--duration", type=float, default=None,
                   help="Measurement duration in seconds")
    p.add_argument("--seed", type=int, default=42,
                   help="PRNG seed for reproducibility")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to write metadata + logs")
    p.add_argument("--sandbox-dir", type=str, default=None,
                   help="Sandbox path for tests that touch files")
    p.add_argument("--safe-root", type=str, default=None,
                   help="Additional approved root (must be absolute path)")
    p.add_argument("--phase-markers", action="store_true",
                   help="Emit phase markers to stderr")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate parameters and exit without running")
    p.add_argument("--cleanup", action="store_true",
                   help="Remove sandbox/output artifacts on exit")
    p.add_argument("--cpu-affinity", type=int, default=None,
                   help="Pin process to this CPU (best effort)")
    p.add_argument("--verbose", action="store_true")
    return p


def pin_cpu(cpu: int | None) -> None:
    if cpu is None:
        return
    try:
        os.sched_setaffinity(0, {cpu})
        log_info(f"pinned to CPU {cpu}")
    except (AttributeError, OSError) as e:
        log_warn(f"cpu pinning failed: {e}")


def enforce_sandbox_caps(num_files: int, total_bytes: int, duration_s: float) -> None:
    if num_files > SANDBOX_MAX_FILES:
        raise SandboxError(
            f"file count {num_files} exceeds hard cap {SANDBOX_MAX_FILES}"
        )
    if total_bytes > SANDBOX_MAX_BYTES:
        raise SandboxError(
            f"total bytes {total_bytes} exceeds hard cap {SANDBOX_MAX_BYTES}"
        )
    if duration_s > SANDBOX_MAX_DURATION_S:
        raise SandboxError(
            f"duration {duration_s} exceeds hard cap {SANDBOX_MAX_DURATION_S} s"
        )
