#!/usr/bin/env python3
"""extract_cli.py -- extract one recording into an L1 store. The remote half of a remote run.

Runs where the recordings are. A remote run invokes this over ssh in the repo on the server,
then pulls back only the npz it names: the chain stays where it is, and what crosses the
network is the extracted channels, not the dumps.

It is an ordinary CLI, so the same command can be run by hand on the server to see what a
remote run would do:

    python3 plan10_analysis/runner/extract_cli.py --root /project/.../zstd_local \\
        --rec-id mem/mem_x_v2/args/rep001__run --speed 2 --columns hamming,cosine [--max-pairs 40]

Prints one JSON object on stdout: {"npz", "meta", "n_pairs", "n_pages", "reused"}. Progress
goes to stderr, so a caller can read stdout as data. Exit 0 on success, 2 on a bad request,
1 on a failure during extraction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent.parent
if str(QEMU_DIR) not in sys.path:
    sys.path.insert(0, str(QEMU_DIR))

# NOT imported at module level: runner.differ and runner.extract need numpy, and --probe
# exists precisely to report a server that lacks it. Importing here would make the probe
# die of the condition it is meant to detect, which is how this first behaved.


def _probe(root: str) -> dict:
    import shutil
    out = {"python": sys.version.split()[0], "executable": sys.executable,
           "zstd": shutil.which("zstd") is not None, "root_exists": Path(root).is_dir(),
           "cwd": str(Path.cwd()), "numpy": None, "differ": None}
    try:
        import numpy
        out["numpy"] = numpy.__version__
    except ImportError as e:
        out["differ"] = {"error": f"not checked: numpy is missing ({e})"}
        return out
    from plan10_analysis.runner import differ as _differ
    try:
        out["differ"] = _differ.differ_version()
    except _differ.DifferError as e:
        out["differ"] = {"error": str(e)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="trace root holding the recording")
    ap.add_argument("--rec-id", required=True, help="recording id, relative to the root")
    ap.add_argument("--speed", type=int, required=True)
    ap.add_argument("--columns", required=True, help="comma-separated differ columns")
    ap.add_argument("--store", default=None, help="L1 store directory (default ~/.cache/plan10/l1)")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--probe", action="store_true", help="report the environment and exit without extracting")
    a = ap.parse_args()

    if a.probe:
        print(json.dumps(_probe(a.root)))
        return 0

    try:
        from plan10_analysis.runner import chain, differ, extract
    except ImportError as e:
        print(json.dumps({"error": f"the server cannot import the runner: {e}. Run with --probe to see what it has"}),
              file=sys.stderr)
        return 1

    rec_dir = Path(a.root) / a.rec_id
    if not rec_dir.is_dir():
        print(json.dumps({"error": f"recording not found: {rec_dir}"}), file=sys.stderr)
        return 2
    cols = [c for c in a.columns.split(",") if c]
    if not cols:
        print(json.dumps({"error": "no columns"}), file=sys.stderr)
        return 2
    store = extract.store_dir(a.store)
    before = extract.existing(store, a.rec_id, a.speed, cols, a.max_pairs)
    try:
        npz = extract.extract(a.rec_id, rec_dir, a.speed, cols, store, max_pairs=a.max_pairs,
                              progress=lambda d, n: print(f"pair {d}/{n}", file=sys.stderr, flush=True))
    except (chain.ChainError, differ.DifferError, ValueError, OSError) as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), file=sys.stderr)
        return 1
    meta = npz.with_name(npz.name.replace(".npz", ".meta.json"))
    d = extract.load(npz)
    print(json.dumps({"npz": str(npz), "meta": str(meta), "n_pairs": int(d["n_pairs"]),
                      "n_pages": int(d["n_pages"]), "reused": before is not None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
