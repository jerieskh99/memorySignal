import argparse, time, os

def disk_force_allocate(D, K, p, T):
    chunks = 0
    t_end = time.time() + T

    with open(p, "wb", buffering=0) as f:
        while time.time() < t_end:
            f.write(D)
            chunks += 1

            if chunks % K == 0:
                os.fsync(f.fileno())
            
            if chunks % 64 == 0:
                print(f"Chunks={chunks}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120, help="Run duration")
    parser.add_argument("--kb", type=int, default=4096, help="Size of chunk in KB")
    parser.add_argument("--fsyncWait", type=int, default=1, help="Call fsync after k chunks")
    parser.add_argument("--path", type=str, default="io_seq.bin", help="Output file")
    parser.add_help("Example Command: python3 sequential_write.py --seconds 120 --chunk_kb 4096 --fsync_every 1")
    
    args = parser.parse_args()

    chunk_size = args.kb * 1024
    data = os.urandom(chunk_size)

    disk_force_allocate(data, args.fsyncWait, args.path, args.seconds)



if __name__ == "__main__":
    main()