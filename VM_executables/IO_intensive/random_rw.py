import argparse, time, os, random

def preallocate_file(path: str, size_mb: int):
    size = 1024 * 1024 * size_mb
    with open(path, "wb") as f:
        f.truncate(size)

def read_write_random(T: int, path: str, wRatio: float, blockSizeKB: int, fileSizeMB: int):
    file_size = 1024 * 1024 * fileSizeMB
    block_size = 1024 * blockSizeKB
    max_offset = file_size - block_size
    data = os.urandom(block_size)
    heartbeat = 0
    t_end = time.time() + T
    
    with open(path, "r+b", buffering=0) as f:
        while time.time() < t_end:
            off = random.randrange(0, max_offset, block_size)
            f.seek(off, whence=os.SEEK_SET)

            if random.random() < wRatio:
                f.write(data)
            else:
                _ = f.read(block_size)
            
            heartbeat += 1
            if heartbeat % 5000 == 0:
                print(f"heartbeat={heartbeat}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120, help="Run duration")
    parser.add_argument("--fileMB", type=int, default=4096, help="Size of file to allocate before random rw in MB")
    parser.add_argument("--blockKB", type=int, default=1, help="I/O block size in KB")
    parser.add_argument("--writeRatio", type=float, default=0.5, help="0..1 probability of write")  
    parser.add_argument("--path", type=str, default="io_rand.bin")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_help("Example Command: python3 sequential_write.py --seconds 120 --chunk_kb 4096 --fsync_every 1")
    args = parser.parse_args()

    random.seed(args.seed)
    preallocate_file(args.path, args.fileMB)

    read_write_random(args.seconds, args.path, args.writeRatio, args.blockKB, args.fileMB)



if __name__ == "__main__":
    main()