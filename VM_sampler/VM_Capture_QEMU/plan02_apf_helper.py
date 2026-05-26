#!/usr/bin/env python3
"""plan02_apf_helper.py -- B+3.1 streaming APF helper.

Owner: ML + EN + SA.
Reference: docs/d51_bplus3_improved_proposal.html · Δ-1 / Δ-2 / Δ-3.

Invoked by the producer bash script after each pmemsave. Runs in the
background. Computes active-page-fraction (APF) between two dumps,
appends one line to a shared trajectory file, writes a per-pair ack
file, deletes the previous dump.

CLI
---
    plan02_apf_helper.py \\
        --prev /path/to/memory_dump-N.raw \\
        --curr /path/to/memory_dump-N+1.raw \\
        --apf-jsonl /workdir/apf_trajectory.jsonl \\
        --ack-dir   /workdir/apf_acks \\
        --seq 12 \\
        --page-size 4096   (optional · default 4096)

Exit codes
----------
    0   ok · APF computed · JSONL line appended · prev deleted · ack written
    1   transient io error · helper logs and exits; producer logs but
        continues to next snap. Pair is recorded as a gap.
    2   numpy import failure · helper cannot compute · producer should
        abort cell after N consecutive code-2 helpers.
    3   disk full while writing JSONL or ack · operator must free space.
    4   curr dump corrupt or zero-byte · pair recorded as gap.

Ack file
--------
Atomic write via tempfile + rename. Path:
    <ack-dir>/seq_<NNNNNNN>.apf_done
Content:
    {"seq":N, "prev":"...", "curr":"...", "exit_code":C,
     "helper_duration_ms":M, "apf":F (if exit_code==0)}

Race-safety
-----------
Each helper owns exactly one prev dump file. Multiple helpers can run
concurrently but never on the same file. JSONL appends use Linux
O_APPEND which is atomic for writes <= PIPE_BUF (4096 bytes). Each
helper writes one ~150-byte line.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PAGE_SIZE_DEFAULT = 4096


# ---------------------------------------------------------------------------
# Exit codes (mirrored in producer + validator)
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_TRANSIENT_IO = 1
EXIT_NUMPY_MISSING = 2
EXIT_DISK_FULL = 3
EXIT_CORRUPT_DUMP = 4


@dataclass
class AckRecord:
    seq: int
    prev: str
    curr: str
    exit_code: int
    helper_duration_ms: int
    apf: float | None = None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_active_page_fraction(prev_path: Path, curr_path: Path,
                                  page_size: int) -> float:
    """Identical math to plan02_metrics_per_cell.active_page_fraction.
    Kept inline here so the helper has no dependency on plan02_metrics_per_cell
    (which may not be on $PATH when producer launches us)."""
    import numpy as np  # type: ignore[import]
    a = np.memmap(prev_path, dtype=np.uint8, mode="r")
    b = np.memmap(curr_path, dtype=np.uint8, mode="r")
    if a.shape != b.shape or a.size == 0:
        return 0.0
    n_pages = a.size // page_size
    if n_pages == 0:
        return 0.0
    a = a[:n_pages * page_size].reshape(n_pages, page_size)
    b = b[:n_pages * page_size].reshape(n_pages, page_size)
    differ = (a != b).any(axis=1)
    return float(differ.sum()) / float(n_pages)


# ---------------------------------------------------------------------------
# Ack file (atomic write via temp + rename)
# ---------------------------------------------------------------------------

def _write_ack(ack_dir: Path, record: AckRecord) -> None:
    ack_dir.mkdir(parents=True, exist_ok=True)
    target = ack_dir / f"seq_{record.seq:07d}.apf_done"
    payload = json.dumps(asdict(record), separators=(",", ":"))
    # Write to a tempfile in the same directory, then rename. rename(2)
    # is atomic on the same filesystem.
    fd, tmp_path = tempfile.mkstemp(prefix=".ack-", suffix=".tmp",
                                    dir=str(ack_dir))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.write("\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _append_apf_line(apf_jsonl: Path, payload: dict) -> None:
    """O_APPEND atomic append. JSON line is < 200 bytes, well under PIPE_BUF."""
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    fd = os.open(str(apf_jsonl), flags, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plan02_apf_helper.py")
    p.add_argument("--prev", required=True)
    p.add_argument("--curr", required=True)
    p.add_argument("--apf-jsonl", required=True)
    p.add_argument("--ack-dir", required=True)
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--page-size", type=int, default=PAGE_SIZE_DEFAULT)
    args = p.parse_args(argv)

    started = time.monotonic()
    prev_path = Path(args.prev)
    curr_path = Path(args.curr)
    apf_jsonl = Path(args.apf_jsonl)
    ack_dir = Path(args.ack_dir)

    record = AckRecord(
        seq=args.seq,
        prev=str(prev_path),
        curr=str(curr_path),
        exit_code=EXIT_OK,
        helper_duration_ms=0,
        apf=None,
    )

    # Step 0: numpy availability
    try:
        import numpy  # noqa: F401
    except ImportError:
        record.exit_code = EXIT_NUMPY_MISSING
        record.helper_duration_ms = int((time.monotonic() - started) * 1000)
        try:
            _write_ack(ack_dir, record)
        except OSError:
            pass
        return EXIT_NUMPY_MISSING

    # Step 1: validate inputs (curr must be non-empty)
    try:
        curr_size = curr_path.stat().st_size
    except OSError:
        record.exit_code = EXIT_CORRUPT_DUMP
        record.helper_duration_ms = int((time.monotonic() - started) * 1000)
        try:
            _write_ack(ack_dir, record)
        except OSError:
            pass
        return EXIT_CORRUPT_DUMP
    if curr_size == 0:
        record.exit_code = EXIT_CORRUPT_DUMP
        record.helper_duration_ms = int((time.monotonic() - started) * 1000)
        _write_ack(ack_dir, record)
        return EXIT_CORRUPT_DUMP

    # Step 2: compute APF
    try:
        apf = _compute_active_page_fraction(prev_path, curr_path, args.page_size)
    except (OSError, ValueError) as e:
        # OSError covers ENOSPC during memmap? unlikely; map to transient
        if isinstance(e, OSError) and e.errno == errno.ENOSPC:
            record.exit_code = EXIT_DISK_FULL
        else:
            record.exit_code = EXIT_TRANSIENT_IO
        record.helper_duration_ms = int((time.monotonic() - started) * 1000)
        try:
            _write_ack(ack_dir, record)
        except OSError:
            pass
        return record.exit_code

    record.apf = apf

    # Step 3: append APF line to shared JSONL
    apf_line = {
        "seq": args.seq,
        "t_emit_epoch": time.time(),
        "prev": str(prev_path),
        "curr": str(curr_path),
        "apf": apf,
    }
    try:
        _append_apf_line(apf_jsonl, apf_line)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            record.exit_code = EXIT_DISK_FULL
        else:
            record.exit_code = EXIT_TRANSIENT_IO
        record.helper_duration_ms = int((time.monotonic() - started) * 1000)
        try:
            _write_ack(ack_dir, record)
        except OSError:
            pass
        return record.exit_code

    # Step 4: delete prev (no longer needed)
    try:
        prev_path.unlink()
    except OSError:
        # Non-fatal: prev may have already been removed by a manual cleanup.
        # The APF is already recorded; downstream doesn't care.
        pass

    # Step 5: write success ack
    record.helper_duration_ms = int((time.monotonic() - started) * 1000)
    try:
        _write_ack(ack_dir, record)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            return EXIT_DISK_FULL
        return EXIT_TRANSIENT_IO

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
