import argparse, time
import numpy as np


def swap_arrays(a, b, x, T):
    t_end = time.time() + T

    heartBeat = 0
    while time.time() < t_end:
        x[:] = a
        a[:] = b
        b[:] = x

        heartBeat += 1
        if heartBeat % 10 == 0:
            print(f"iter={heartBeat}")


def sweep(buf: np.ndarray, T: int, page_stride: int = 4096):
    t_end = time.time() + T
    heartbeat = 0
    v = np.uint8(0)
    n = buf.size

    while time.time() < t_end:
        # Touch one byte in each page of the buffer:
        for i in range(0, n, page_stride):
            buf[i] = v
        v = np.uint8(v+1)

        heartbeat += 1
        if heartbeat % 32 == 0:
            print(f"iter={heartbeat}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=int, default=512, help="Working set size in MB")
    parser.add_argument("--seconds", type=int, default=120, help="Run duration")

    args = parser.parse_args()

    nMB = 1024 * 1024 * args.mb

    buf = np.zeros(nMB, dtype=np.uint8)
    # arr2 = np.ones(nMB, dtype=np.uint8)
    # tmp  = np.zeros(nMB, dtype=np.uint8)

    sweep(buf, args.seconds)
    #swap_arrays(arr1, arr2, tmp, args.seconds)


if __name__ == "__main__":
    main()