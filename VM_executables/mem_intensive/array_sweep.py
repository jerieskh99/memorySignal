import argparse, time
import numpy as np


def swap_arrays(a, b, x, T):
    t_end = time.time() + T

    heartBeat = 0
    while time.time() < t_end:
        x[:] = a
        a[:] = b
        b[:] = x

        if heartBeat % 10 == 0:
            print(f"iter={heartBeat}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=int, default=2048, help="Working set size in MB")
    parser.add_argument("--seconds", type=int, default=120, help="Run duration")
    parser.add_help("Example Command: python3 array_sweep.py --mb 4096 --seconds 120")

    args = parser.parse_args()

    nMB = 2048 * args.mb * args.mb

    arr1 = np.zeros(nMB, dtype=np.uint8)
    arr2 = np.ones(nMB, dtype=np.uint8)
    tmp  = np.zeros(nMB, dtype=np.uint8)

    swap_arrays(arr1, arr2, tmp, args.seconds)


if __name__ == "__main__":
    main()