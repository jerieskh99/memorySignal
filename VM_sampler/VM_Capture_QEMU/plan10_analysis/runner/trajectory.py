"""trajectory.py -- read the substrate trajectory a capture already wrote.

A capture run with CAPTURE_METRIC=substrate ran the differ on every pair and wrote the
per-page vectors to `run_matrix_test<N>_<workload>.npy.substrate_trajectory.csv[.zst]`.
That file holds exactly what runner/extract.py otherwise spends hours recomputing: one row
per changed page per pair, with the differ's columns.

Two format differences from the differ's own per-pair output, both load-bearing:

  * the per-pair CSV starts at `page_index`; the trajectory starts at `seq, page_index`.
    differ.parse_sparse_csv reads row[0] as the page index, so it CANNOT read this file --
    it would silently return sequence numbers as page indices.
  * the trajectory numbers pairs from 0; chain.walk_chain yields 1..n-1. This module adds
    one, so a seq here means the same pair it means everywhere else in the runner.

What it cannot tell you is the differ speed the capture used: that is unrecorded per
recording (config `substrateSpeed` is the only record, which is why the console labels the
value "assumed"). A caller reusing this file inherits that assumption; extract records it.
"""
from __future__ import annotations

import csv
import gzip
import io
import subprocess
from array import array
from pathlib import Path

import numpy as np

SUFFIXES = (".csv", ".csv.zst", ".csv.gz")
MARKER = "substrate_trajectory"


class TrajectoryError(RuntimeError):
    pass


def find(rec_dir: Path) -> Path | None:
    """The trajectory file sitting in a recording's own directory, if the capture left one."""
    rec_dir = Path(rec_dir)
    if not rec_dir.is_dir():
        return None
    hits = sorted(p for p in rec_dir.iterdir()
                  if p.is_file() and MARKER in p.name and p.name.endswith(SUFFIXES))
    return hits[0] if hits else None


def _open_text(path: Path):
    name = path.name
    if name.endswith(".zst"):
        p = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.stdout is None:
            raise TrajectoryError(f"could not decompress {path}")
        return io.TextIOWrapper(p.stdout, newline=""), p
    if name.endswith(".gz"):
        return gzip.open(path, "rt", newline=""), None
    return open(path, newline=""), None


def _close(fh, proc, early: bool) -> None:
    """Close the stream; when we stopped reading on purpose (header only, or max_pairs) the
    decompressor's broken-pipe complaint is expected and swallowed. A failure on a full read
    is a real one and is raised with zstd's own words."""
    try:
        fh.close()
    except OSError:
        pass
    if proc is None:
        return
    if early:
        proc.terminate()
    err = b""
    try:
        err = proc.stderr.read() if proc.stderr else b""
    except OSError:
        pass
    rc = proc.wait()
    if not early and rc != 0:
        raise TrajectoryError(f"zstd exited {rc}: {err.decode(errors='replace').strip()[:200]}")


def columns(path: Path) -> list[str]:
    """The header, which is the honest list of what this recording can offer."""
    fh, proc = _open_text(Path(path))
    try:
        head = next(csv.reader(fh), [])
    finally:
        _close(fh, proc, early=True)
    if len(head) < 3 or head[0] != "seq" or head[1] != "page_index":
        raise TrajectoryError(f"{path}: not a substrate trajectory (header {head[:3]})")
    return [c for c in head[2:] if c]


def read(path: Path, want: list[str], max_pairs: int | None = None,
         progress=None) -> dict[str, np.ndarray]:
    """Stream the file, keeping `want`. Returns seq (1-based), page_index and one array each.

    Rows are read into array.array accumulators rather than Python lists: a full recording is
    millions of rows and the compact form keeps a 4-million-row file inside a few tens of MB.
    """
    path = Path(path)
    fh, proc = _open_text(path)
    ok, stop = False, False
    try:
        rd = csv.reader(fh)
        head = next(rd, [])
        idx = {name: i for i, name in enumerate(head)}
        missing = [c for c in want if c not in idx]
        if missing:
            raise TrajectoryError(f"{path}: columns not in this trajectory: {missing}; it has {head[2:]}")
        take = [idx[c] for c in want]
        seqs, pages = array("i"), array("i")
        vals = [array("f") for _ in want]
        last_seq, n_pairs = -1, 0
        for row in rd:
            if not row:
                continue
            s = int(row[0])
            if s != last_seq:
                last_seq = s
                n_pairs += 1
                if max_pairs is not None and n_pairs > max_pairs:
                    stop = True
                if progress and not stop:
                    progress(n_pairs, max_pairs)
            if stop:
                break
            seqs.append(s + 1)                 # trajectory counts pairs from 0, walk_chain from 1
            pages.append(int(row[1]))
            for j, i in enumerate(take):
                vals[j].append(float(row[i]))
        ok = True
    finally:
        _close(fh, proc, early=(stop or not ok))
    out = {"seq": np.frombuffer(seqs, dtype=np.int32).copy(),
           "page_index": np.frombuffer(pages, dtype=np.int32).copy(),
           "n_pairs": min(n_pairs, max_pairs) if max_pairs is not None else n_pairs}
    for c, v in zip(want, vals):
        out[c] = np.frombuffer(v, dtype=np.float32).copy()
    return out
