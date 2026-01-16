import argparse, time, gc, os
import numpy as np


def get_page_size():
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except:
        return 4096


def touch_page(buf: bytearray, page_size: int, val: int):
    n = len(buf)
    for off in range(0, n, page_size):
        buf[off] = (buf[off] + val) & 0xFF


def run_allocator(obj_bytes: int, T: int, n_obj: int, sleep_ms: int, page_size: int):
    batches = 0
    t_end = time.time() + T
    checksum = 0
    v = 1

    while time.time() < t_end:
        buffers = []
        for _ in range(n_obj):
            curr_obj = bytearray(obj_bytes)
            touch_page(curr_obj, page_size, v)
            checksum ^= curr_obj[0]
            buffers.append(curr_obj)
        
        buffers.clear()
        del buffers

        batches += 1
        v = (v+1) & 0xFF

        if batches % 10 == 0:
            print(f"batches={batches} checksum={checksum}")

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120, help="Run duration")
    parser.add_argument("--objects", type=int, default=2000, help="Objects per batch")
    parser.add_argument("--object-kb", type=int, default=256, help="Object size in KB")
    parser.add_argument("--sleep-ms", type=int, default=50, help="sleep duration between batches in mSec")
    parser.add_argument("--gc", action ="store_true", help="Enable Python GC (default false)")
    parser.add_argument("--page-size", type=int, default=None, help="Overrides the OS defined page size")
    args = parser.parse_args()

    if args.gc:
        gc.enable()
    else:
        gc.disable()

    page_size = args.page_size or get_page_size()

    run_allocator(1024 * args.object_kb, args.seconds, args.objects, args.sleep_ms, page_size)




if __name__ == "__main__":
    main()