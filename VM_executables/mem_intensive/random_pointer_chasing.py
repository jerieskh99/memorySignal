import argparse, time
import numpy as np


def traverse_array(i_perm, arr, T):
    t_end = time.time() + T
    i = np.int32(0)
    acc = 0
    steps = 0

    while time.time() < t_end:
        i = i_perm[i]
        d = arr[i]
        acc ^= d
        steps += 1
        if steps % 1048576: # Size of MB
            print(f"Steps {steps}, acc {acc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=int, default=2048, help="Working set size in MB")
    parser.add_argument("--seconds", type=int, default=120, help="Run duration")
    parser.add_argument("--seed", type=int, default=120, help="RNG seed")
    parser.add_help("Example Command: python3 random_pointer_chasing.py --mb 2048 --seconds 120")
    
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    bytes = 1024 * 1024 * args.mb
    n_elements = bytes // 4

    # Allocate the random array and create a permutation of the indices.
    idx = np.random.permutation(n_elements)
    data = rng.integers(0, 256, size=n_elements, dtype=np.uint8)

    traverse_array(idx, data, args.seconds)

if __name__ == "__main__":
    main()