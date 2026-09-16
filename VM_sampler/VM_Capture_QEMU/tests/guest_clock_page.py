#!/usr/bin/env python3
"""Guest-side clock page writer (runs INSIDE the guest).

Writes the guest's own clock readings into one locked page of memory about once per
millisecond, so that every memory dump the host takes contains the guest's clock at the
instant of the dump. The host-side reader (tests/read_clock_page.py) finds the page in each
dump by its marker and reads the record back.

Record layout, little-endian, 64 bytes at the start of the page:
    16s  marker
    Q    counter (head)
    Q    CLOCK_MONOTONIC      ns
    Q    CLOCK_MONOTONIC_RAW  ns   (not subject to NTP frequency slew)
    Q    CLOCK_REALTIME       ns
    Q    CLOCK_BOOTTIME       ns
    Q    counter (tail)       equal to the head unless the dump caught the write mid-way

The marker is assembled at run time from two halves so the literal never appears in this
file and therefore never appears in the page cache as a false hit. The record is packed
straight into the page (struct.pack_into) so no complete copy of it lingers on the Python
heap; the reader still validates head == tail and picks the freshest valid record among
any hits.

Proposal: al-Kindi, time_axes_paper/council/01_al_kindi_contribution_or_metaphor.md,
section 6. This version adds an absolute-deadline loop (so the write rate does not drift),
CLOCK_MONOTONIC_RAW, and a heartbeat file.

Usage in the guest:
    python3 guest_clock_page.py --seconds 90 --hz 1000 --heartbeat ~/clock_page_heartbeat.jsonl
Nothing is written anywhere except the heartbeat file if one is named.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import struct
import sys
import time

MARKER = b"CLKPAGE-" + b"MEMSIG02"      # 16 bytes; assembled so the literal is not in this file
FMT = "<16sQQQQQQ"
RECORD_SIZE = struct.calcsize(FMT)      # 64
PAGE_SIZE = 4096


def _clock(name: str, fallback: int) -> int:
    """Return the clock id, or the fallback where a platform lacks it (only for smoke tests
    off-guest; the guest is Linux and has all four)."""
    return getattr(time, name, fallback)


CLK_MONO = time.CLOCK_MONOTONIC
CLK_MONO_RAW = _clock("CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
CLK_REAL = time.CLOCK_REALTIME
CLK_BOOT = _clock("CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)


def pack_record(page, n: int) -> None:
    """One record, packed in place. Field order matters for torn-write detection: the head
    counter is written first and the tail counter last."""
    struct.pack_into(
        FMT, page, 0,
        MARKER, n,
        time.clock_gettime_ns(CLK_MONO),
        time.clock_gettime_ns(CLK_MONO_RAW),
        time.clock_gettime_ns(CLK_REAL),
        time.clock_gettime_ns(CLK_BOOT),
        n,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=120.0, help="how long to keep writing")
    ap.add_argument("--hz", type=float, default=1000.0, help="target write rate")
    ap.add_argument("--heartbeat", default="", help="optional JSONL file for a heartbeat line every --hz writes")
    ap.add_argument("--quiet", action="store_true", help="no heartbeat on stdout")
    a = ap.parse_args()

    page = mmap.mmap(-1, PAGE_SIZE)
    buf = (ctypes.c_char * PAGE_SIZE).from_buffer(page)
    addr = ctypes.addressof(buf)
    locked = False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(PAGE_SIZE)) == 0:
            locked = True
        else:
            print(f"[clock-page] mlock failed (errno {ctypes.get_errno()}); continuing unlocked", file=sys.stderr)
    except Exception as e:  # pragma: no cover
        print(f"[clock-page] mlock unavailable ({e}); continuing unlocked", file=sys.stderr)

    hb = open(a.heartbeat, "a") if a.heartbeat else None
    period = 1.0 / a.hz
    t_start = time.monotonic()
    t_end = t_start + a.seconds
    n = 0
    late = 0
    start_line = {
        "event": "start", "pid": os.getpid(), "locked": locked, "page_size": PAGE_SIZE,
        "record_size": RECORD_SIZE, "hz": a.hz, "seconds": a.seconds,
        "mono_ns": time.clock_gettime_ns(CLK_MONO), "real_ns": time.clock_gettime_ns(CLK_REAL),
        "boot_ns": time.clock_gettime_ns(CLK_BOOT), "mono_raw_ns": time.clock_gettime_ns(CLK_MONO_RAW),
    }
    if not a.quiet:
        print("[clock-page] " + json.dumps(start_line), flush=True)
    if hb:
        hb.write(json.dumps(start_line) + "\n"); hb.flush()

    every = max(1, int(a.hz))
    while True:
        now = time.monotonic()
        if now >= t_end:
            break
        n += 1
        pack_record(page, n)
        if n % every == 0:
            line = {"event": "hb", "n": n, "mono_ns": time.clock_gettime_ns(CLK_MONO),
                    "real_ns": time.clock_gettime_ns(CLK_REAL), "late_writes": late}
            if not a.quiet:
                print("[clock-page] " + json.dumps(line), flush=True)
            if hb:
                hb.write(json.dumps(line) + "\n"); hb.flush()
        # absolute deadline for the next write, so the rate does not drift with the work
        nxt = t_start + n * period
        rem = nxt - time.monotonic()
        if rem > 0:
            time.sleep(rem)
        else:
            late += 1

    end_line = {"event": "end", "n": n, "late_writes": late,
                "mono_ns": time.clock_gettime_ns(CLK_MONO), "real_ns": time.clock_gettime_ns(CLK_REAL)}
    if not a.quiet:
        print("[clock-page] " + json.dumps(end_line), flush=True)
    if hb:
        hb.write(json.dumps(end_line) + "\n"); hb.close()
    # keep the last record in place until exit; the page dies with the process
    return 0


if __name__ == "__main__":
    sys.exit(main())
