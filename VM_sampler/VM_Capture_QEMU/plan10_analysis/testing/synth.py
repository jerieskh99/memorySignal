#!/usr/bin/env python3
"""synth.py -- a small synthetic corpus in the real chain format, with known answers.

Dumps are 4 MiB (1024 pages), the smallest size the differ indexes correctly
(runner/differ.py). Each snapshot k rewrites a known set of pages relative to k-1:

    full pages  : pages {base_full + k*3 + i for i in range(n_full)}  fully re-randomised
    touched     : page  (200 + k)                                     one byte flipped
    steady      : pages {900, 901}                                    re-randomised every k

so the changed-page count per pair is n_full + 1 + 2 and APF = that / 1024. Chains are
written with the same zstd invocations the capture consumer uses (base: plain zstd; then
`zstd --long=31 --patch-from=<prev>`), so chain.py and reconstruct_zstd_chain.sh read them.

    root = make_corpus(tmpdir, n_snapshots=6)
    -> <root>/<family>/<workload>/<variant>/rep001__synth/000000.zst ...
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

PAGE = 4096
N_PAGES = 1024                      # 4 MiB


def base_dump(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=N_PAGES * PAGE, dtype=np.uint8)


def next_snapshot(prev: np.ndarray, k: int, seed: int, n_full: int = 4) -> tuple[np.ndarray, set[int]]:
    rng = np.random.default_rng(seed * 1000 + k)
    cur = prev.copy()
    changed: set[int] = set()
    for i in range(n_full):
        p = (10 + k * 3 + i) % 800
        cur[p * PAGE:(p + 1) * PAGE] = rng.integers(0, 256, size=PAGE, dtype=np.uint8)
        changed.add(p)
    t = 200 + (k % 100)
    cur[t * PAGE] ^= 0xFF
    changed.add(t)
    for p in (900, 901):
        cur[p * PAGE:(p + 1) * PAGE] = rng.integers(0, 256, size=PAGE, dtype=np.uint8)
        changed.add(p)
    return cur, changed


def write_chain(rec_dir: Path, snapshots: list[np.ndarray]) -> None:
    rec_dir.mkdir(parents=True, exist_ok=True)
    prev_raw = None
    for k, snap in enumerate(snapshots):
        raw = rec_dir / f"snap_{k:06d}.raw"
        raw.write_bytes(snap.tobytes())
        out = rec_dir / f"{k:06d}.zst"
        if k == 0:
            subprocess.run(["zstd", "-q", "-3", "-f", str(raw), "-o", str(out)], check=True)
        else:
            subprocess.run(["zstd", "-q", "-3", "-f", "--long=31", f"--patch-from={prev_raw}", str(raw), "-o", str(out)], check=True)
        if prev_raw is not None:
            prev_raw.unlink()
        prev_raw = raw
    if prev_raw is not None:
        prev_raw.unlink()


def make_recording(rec_dir: Path, n_snapshots: int, seed: int, n_full: int = 4) -> list[set[int]]:
    """Write one chain; return the changed-page set for each pair (index 1..n-1)."""
    snaps = [base_dump(seed)]
    changes: list[set[int]] = [set()]
    for k in range(1, n_snapshots):
        cur, ch = next_snapshot(snaps[-1], k, seed, n_full)
        snaps.append(cur)
        changes.append(ch)
    write_chain(rec_dir, snaps)
    return changes


def make_corpus(root: Path, n_snapshots: int = 6, workloads: tuple = (("mem", "mem_synth_a_v2", 2), ("cpu", "cpu_synth_b_v2", 1))) -> dict:
    """family/workload/variant/rep dirs with one chain each. Returns {rec_id: changes}."""
    out: dict[str, list[set[int]]] = {}
    seed = 1
    for fam, wl, reps in workloads:
        for r in range(1, reps + 1):
            rec = Path(root) / fam / wl / f"pages_1024_--duration_{n_snapshots}_--seed_{seed}" / f"rep{r:03d}__synth"
            out[str(rec.relative_to(root))] = make_recording(rec, n_snapshots, seed)
            seed += 1
    return out
