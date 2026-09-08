#!/usr/bin/env python3
"""chain.py -- walk one zstd patch chain, one snapshot pair at a time.

A recording is `000000.zst` (the full base dump) followed by `000001.zst ...`, each a
`zstd --patch-from` delta against the previous reconstructed snapshot
(reconstruct_zstd_chain.sh). Reconstruction is therefore sequential, and holding every
snapshot would cost n GB. This walker keeps a two-file rolling window: reconstruct
snapshot k, yield the pair (k-1, k), delete k-1.

    for seq, prev_raw, curr_raw in walk_chain(rec_dir, work_dir):
        ...            # seq == k, the index of curr; the pair is (k-1, k)

`start_seq` lets a resumed run skip pairs it already has: the walker still has to
reconstruct every snapshot up to start_seq (the chain is sequential) but does not yield
them. `max_pairs` bounds the walk and is the caller's responsibility to record.

Uses the `zstd` CLI exactly as the reconstruct script does. Pure stdlib.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

_RE_SNAP = re.compile(r"^(\d{6})\.zst$")


class ChainError(RuntimeError):
    pass


def chain_files(rec_dir: Path) -> list[Path]:
    """The .zst files of a recording in chain order, refusing gaps like the script does."""
    files = sorted((p for p in Path(rec_dir).iterdir() if _RE_SNAP.match(p.name)), key=lambda p: int(p.stem))
    for expected, p in enumerate(files):
        if int(p.stem) != expected:
            raise ChainError(f"chain gap in {rec_dir}: expected {expected:06d}.zst, found {p.name}")
    return files


def zstd_available() -> bool:
    return shutil.which("zstd") is not None


def _decompress(src: Path, dst: Path, patch_from: Path | None) -> None:
    cmd = ["zstd", "-d", "-q", "-f"]
    if patch_from is not None:
        cmd += ["--long=31", f"--patch-from={patch_from}"]
    cmd += [str(src), "-o", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ChainError(f"zstd failed on {src.name} (exit {r.returncode}): {r.stderr.strip()[:300]}")


def walk_chain(rec_dir: Path, work_dir: Path, start_seq: int = 1, max_pairs: int | None = None) -> Iterator[tuple[int, Path, Path]]:
    """Yield (seq, prev_raw, curr_raw) for seq = 1..n-1, reconstructing as it goes."""
    if not zstd_available():
        raise ChainError("zstd CLI not on PATH")
    files = chain_files(rec_dir)
    if len(files) < 2:
        raise ChainError(f"chain has {len(files)} snapshot(s); need at least 2")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    prev = work_dir / "snap_prev.raw"
    curr = work_dir / "snap_curr.raw"
    _decompress(files[0], prev, None)
    yielded = 0
    try:
        for k, f in enumerate(files[1:], start=1):
            _decompress(f, curr, prev)
            if k >= start_seq:
                yield k, prev, curr
                yielded += 1
                if max_pairs is not None and yielded >= max_pairs:
                    return
            # roll the window: curr becomes prev
            prev.unlink(missing_ok=True)
            curr.rename(prev)
    finally:
        prev.unlink(missing_ok=True)
        curr.unlink(missing_ok=True)


def reconstruct_base(rec_dir: Path, dst: Path) -> Path:
    """Only the base snapshot; used to learn a recording's byte size (page count)."""
    files = chain_files(rec_dir)
    if not files:
        raise ChainError(f"no snapshots in {rec_dir}")
    _decompress(files[0], dst, None)
    return dst
