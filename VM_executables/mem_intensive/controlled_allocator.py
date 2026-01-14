import argparse, time, gc, os
import numpy as np


def get_page_size() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except:
        return 4096


def touch_page(buf: bytearray, page_size: int):
    n = len(buf)
    for b in range(0, n, page_size):
        buf[b] = (buf[b] + 1) & 0xFF


def run_allocator(s_obj: int, T: int, n_obj: int, tau: int, page_size: int):
    batches = 0
    t_end = time.time() + T

    while time.time() < t_end:
        buffer = []
        for _ in range(n_obj):
            curr_obj = bytearray(s_obj)
            touch_page(curr_obj, page_size)
            checksum ^= curr_obj[0]
            buffer.append(curr_obj)
        
        buffer = None
        batches += 0

        if batches % 10 == 0:
            print (f"Batches = {batches}")

        if tau > 0:
            time.sleep(tau / 1000.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120, help="Run duration")
    parser.add_argument("--objects", type=int, default=2000, help="Objects per batch")
    parser.add_argument("--object-kb", type=int, default=256, help="Object size in KB")
    parser.add_argument("--sleep-ms", type=int, default=50, help="sleep duration between batches in mSec")
    parser.add_argument("--gc", action ="store_true", help="Enable Python GC (default false)")
    parser.add_argument("--page-size", type=int, default=None, help="Overrides the OS defined page size")
    parser.add_help("Example Command: python3 controlled_allocator.py --seconds 120 --objs 3000 --obj_kb 256 --sleep_ms 20")
    args = parser.parse_args()

    if args.gc:
        gc.enable()
    else:
        gc.disable

    obj_size = 1024 * args.object_kb
    page_size = args.page_size

    if not page_size:
        page_size = get_page_size()

    run_allocator(obj_size, args.seconds, args.objects, args.sleep_ms, page_size)




if __name__ == "__main__":
    main()