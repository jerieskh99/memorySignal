import argparse, time
import numpy as np


# Use LCG Linear Congruential Generator : find ext pseudo-random generated numbers
def traverse_array(arr: np.ndarray, T: int, seed: int, stride: int = 4096):
    t_end = time.time() + T
    n =arr.size
    # Work on "page slot" size
    m = max(1, n // stride)
 
    # LCG params:
    a = 1664525
    c = 1013904223
    x = seed % m

    acc = 0
    steps = 0

    while time.time() < t_end:
        i = (a * x + c) % m
        idx = x * stride

        acc ^= int(arr[idx])
        steps += 1
        if steps % 1048576 == 0: # Size of MB
            print(f"steps={steps} acc={acc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=int, default=512, help="Working set size in MB")
    parser.add_argument("--seconds", type=int, default=120, help="Run duration")
    parser.add_argument("--seed", type=int, default=120, help="RNG seed")
    parser.add_argument("--stride", type=int, default=4096)
    
    args = parser.parse_args()

    # rng = np.random.default_rng(args.seed)

    nbytes = 1024 * 1024 * args.mb
    # n_elements = bytes # uint8 => 1 byte per element

    # Allocate the random array and create a permutation of the indices.
    # perm = rng.permutation(n_elements).astype(np.int64, copy=False)
    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, 256, size=nbytes, dtype=np.uint8)

    traverse_array(data, args.seconds, args.seed, args.stride)

if __name__ == "__main__":
    main()