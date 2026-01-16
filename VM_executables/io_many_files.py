import argparse, os, time, tempfile, shutil, random

def small_file_metadata_storm(base: str, payload: bytes, T: int, files_per_batch: int, keep: bool):
    print(f"dir={base}")
    batch = 0
    t_end = time.time() + T

    try:
        while time.time() < t_end:
            batch_files = []
            for i in range(files_per_batch):
                suffix = random.getrandbits(30)
                f_path = os.path.join(base, f"f_{batch}_{i}_{suffix}.bin")
                with open(f_path, "wb") as f:
                    f.write(payload)
                batch_files.append(f_path)
            
            if not keep:
                for fp in batch_files:
                    os.unlink(fp)
            
            batch += 1
            if batch % 5 == 0:
                print(f"batches={batch}")
    finally:
        if not keep:
            shutil.rmtree(base, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120, help="Run duration")
    parser.add_argument("--files-per-batch", type=int, default=500)
    parser.add_argument("--payload-bytes", type=int, default=1024)
    parser.add_argument("--keep", action="store_true", help="Keep files (default deletes)")
    parser.add_argument("--seed", type=int, default=123)
    
    args = parser.parse_args()

    random.seed(args.seed)
    payload = os.urandom(args.payload_bytes)
    base = tempfile.mkdtemp(prefix="io_many_")



    small_file_metadata_storm(base, payload, args.seconds, args.files_per_batch, args.keep)




if __name__ == "__main__":
    main()