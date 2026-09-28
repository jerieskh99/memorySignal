"""Recover each kernel cell's true seed from its retention folder name.

The campaign names a cell folder by the retention signature of its command
(plan07_campaign/ui/place_csv.py `signature`, the same algorithm as
run_files_controlled.py): the command's tokens without the binary path and
--output-dir, joined with "_", cut to 60 characters, then "_" and the first 8
hex characters of the sha1 of the uncut string. For long parameter lists the cut
falls before the seed (gibbs, histogram, rmat_gen, spmm) or inside it
(fem_assembly), so the seed cannot be read from the name.

This script rebuilds the uncut string for every candidate seed and keeps the one
whose signature equals the folder name exactly. A folder that matches no seed, or
more than one, stops the script. Base settings are the kernels' corpus settings
(plan07_campaign/full_campaign_steps.txt, with --duration 600 as the corpus ran).

usage: make_seed_map.py (--root ROOT | --listing FILE) --out CSV
  --root     the retention root holding kernel/<test_label>/<sig>/rep*__*/
  --listing  a text file of cell paths relative to the root, one per line
"""
import argparse, csv, hashlib, re, sys
from pathlib import Path

BASE = {
    "kernel_gemm_v2": "--dim 1024 --block 64",
    "kernel_floyd_v2": "--dim 1024",
    "kernel_gibbs_v2": "--width 1024 --height 1024 --states 2 --beta-milli 400",
    "kernel_nbody_v2": "--particles 262144 --neighbors 16",
    "kernel_spmm_v2": "--rows 4096 --inner 4096 --cols 64 --nnz-per-row 16",
    "kernel_stencil_jacobi_v2": "--grid-n 1024",
    "kernel_fft_v2": "--n 1048576",
    "kernel_histogram_v2": "--samples 50000000 --bins 1048576 --dist uniform",
    "kernel_fem_assembly_v2": "--nodes 2048 --elements 8192 --npe 4",
    "kernel_lexer_v2": "--input-mb 32",
    "kernel_rmat_gen_v2": "--scale 18 --edge-factor 16 --a-milli 570 --b-milli 190 --c-milli 190",
    "kernel_bnb_tsp_v2": "--cities 13",
}
TAIL = "--duration 600 --seed {seed} --phase-markers --max-mb 512"
SEEDS = range(100000)


def signature(args: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(args.split())).strip("._-")
    return s if len(s) <= 60 else s[:60] + "_" + hashlib.sha1(s.encode()).hexdigest()[:8]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--root")
    g.add_argument("--listing")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.root:
        root = Path(a.root)
        cells = sorted(str(p.relative_to(root)) for p in root.glob("kernel/*/*/rep*__*") if p.is_dir())
    else:
        cells = sorted(l.strip() for l in open(a.listing) if l.strip())
    table: dict[str, dict[str, list[int]]] = {k: {} for k in BASE}
    for k, b in BASE.items():
        for s in SEEDS:
            table[k].setdefault(signature(f"{b} {TAIL.format(seed=s)}"), []).append(s)
    rows, bad = [], []
    for c in cells:
        parts = Path(c).parts
        if len(parts) != 4 or parts[0] != "kernel" or parts[1] not in BASE:
            continue
        hits = table[parts[1]].get(parts[2], [])
        if len(hits) != 1:
            bad.append((c, hits))
            continue
        rows.append({"path": c, "seed": hits[0],
                     "source": "retention signature reproduced exactly (place_csv.py signature, sha1 fingerprint)"})
    if bad:
        for c, h in bad:
            print(f"no unique seed for {c}: {h}", file=sys.stderr)
        return 2
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "seed", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cells -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
