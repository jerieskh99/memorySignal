#!/usr/bin/env python3
"""extract.py -- the L1 store: per recording, the requested differ columns for every pair.

One extraction per (recording, speed, channel set), written once and reused by every
scheme that asks for a subset of those channels. The file is an npz:

    seq         int32[n_rows]   the pair index (1..n_pairs) of each row
    page_index  int32[n_rows]   the page the row describes
    <channel>   float32[n_rows] one array per requested column
    n_pairs     int             pairs actually walked
    n_pages     int             pages per dump (dump size // 4096)

plus <name>.meta.json with the recording id, speed, columns, max_pairs, differ version,
chain file count, and `complete`. A store whose meta says complete for a superset of the
requested columns at the same speed is reused; anything else is extracted again.

Resume granularity is the recording: a run interrupted mid-recording redoes that one.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from plan10_analysis.runner import chain, differ

DEFAULT_STORE = "~/.cache/plan10/l1"


def store_dir(base: str | Path | None = None) -> Path:
    p = Path(os.path.expanduser(str(base or DEFAULT_STORE)))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _key(rec_id: str, speed: int, columns: list[str]) -> str:
    h = hashlib.sha256((rec_id + "|" + ",".join(sorted(columns))).encode()).hexdigest()[:12]
    return f"{h}_s{speed}"


def existing(store: Path, rec_id: str, speed: int, columns: list[str], max_pairs: int | None) -> Path | None:
    """A complete store for this recording at this speed covering these columns, if any."""
    for meta in store.glob(f"*_s{speed}.meta.json"):
        m = json.loads(meta.read_text())
        if m.get("rec_id") != rec_id or not m.get("complete"):
            continue
        if not set(columns) <= set(m.get("columns", [])):
            continue
        if (m.get("max_pairs") or None) != (max_pairs or None):
            continue
        npz = meta.with_name(meta.name.replace(".meta.json", ".npz"))
        if npz.exists():
            return npz
    return None


def extract(rec_id: str, rec_dir: Path, speed: int, columns: list[str], store: Path,
            max_pairs: int | None = None, progress: Callable[[int, int], None] | None = None,
            work_dir: Path | None = None) -> Path:
    """Walk the chain, diff every pair, keep the requested columns. Returns the npz path."""
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    hit = existing(store, rec_id, speed, columns, max_pairs)
    if hit:
        return hit
    cols = sorted(set(columns) | {"hamming"})     # hamming defines "changed"; always kept
    key = _key(rec_id, speed, cols)
    npz_path = store / f"{key}.npz"
    meta_path = store / f"{key}.meta.json"
    files = chain.chain_files(rec_dir)
    n_total = len(files) - 1 if not max_pairs else min(len(files) - 1, max_pairs)
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="plan10_"))
    work.mkdir(parents=True, exist_ok=True)
    binary = differ.find_differ()
    seqs, pages = [], []
    colvals: dict[str, list[np.ndarray]] = {c: [] for c in cols}
    n_pages = None
    done = 0
    meta = {"rec_id": rec_id, "speed": speed, "columns": cols, "max_pairs": max_pairs, "n_chain_files": len(files),
            "differ": differ.differ_version(binary), "complete": False, "n_pairs": 0}
    meta_path.write_text(json.dumps(meta, indent=1))
    for seq, prev, curr in chain.walk_chain(rec_dir, work, start_seq=1, max_pairs=max_pairs):
        if n_pages is None:
            n_pages = differ.check_dump_size(prev)
        rows = differ.diff_pair(prev, curr, speed, cols, work, binary)
        n = rows["page_index"].shape[0]
        seqs.append(np.full(n, seq, dtype=np.int32))
        pages.append(rows["page_index"])
        for c in cols:
            colvals[c].append(rows[c])
        done += 1
        if progress:
            progress(done, n_total)
    arrays = {"seq": np.concatenate(seqs) if seqs else np.zeros(0, np.int32),
              "page_index": np.concatenate(pages) if pages else np.zeros(0, np.int32),
              "n_pairs": np.int64(done), "n_pages": np.int64(n_pages or 0)}
    for c in cols:
        arrays[c] = np.concatenate(colvals[c]) if colvals[c] else np.zeros(0, np.float32)
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")     # numpy appends .npz to any other name
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz_path)
    meta.update(complete=True, n_pairs=done, n_pages=int(n_pages or 0))
    meta_path.write_text(json.dumps(meta, indent=1))
    return npz_path


def extract_from_trajectory(rec_id: str, csv_path: Path, speed: int, columns: list[str], store: Path,
                            max_pairs: int | None = None, n_pages: int | None = None,
                            progress: Callable[[int, int | None], None] | None = None) -> Path:
    """The capture's own per-page rows, read instead of recomputed. Same store layout as extract().

    A capture with CAPTURE_METRIC=substrate already ran the differ on every pair and kept the
    result beside the chain; walking the chain again reproduces it at hours per recording. The
    store this writes is interchangeable with extract()'s -- same key, same arrays -- with two
    facts the sidecar must carry: the differ speed is the config assumption (no recording records
    its own), and n_pages is the config default unless the data addresses a page beyond it.
    """
    from plan10_analysis.runner import trajectory
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    hit = existing(store, rec_id, speed, columns, max_pairs)
    if hit:
        return hit
    cols = sorted(set(columns) | {"hamming"})
    key = _key(rec_id, speed, cols)
    npz_path = store / f"{key}.npz"
    meta_path = store / f"{key}.meta.json"
    meta = {"rec_id": rec_id, "speed": speed, "speed_assumed": True, "columns": cols, "max_pairs": max_pairs,
            "source": "substrate_csv", "trajectory": Path(csv_path).name,
            "differ": "capture-time (version unrecorded)", "complete": False, "n_pairs": 0}
    meta_path.write_text(json.dumps(meta, indent=1))
    rows = trajectory.read(csv_path, cols, max_pairs=max_pairs, progress=progress)
    npg = int(n_pages or 0)
    src_npg = "config default"
    if rows["page_index"].size and int(rows["page_index"].max()) >= npg:
        npg = int(rows["page_index"].max()) + 1          # never fewer pages than the data addresses
        src_npg = "max page_index + 1"
    arrays = {"seq": rows["seq"], "page_index": rows["page_index"],
              "n_pairs": np.int64(rows["n_pairs"]), "n_pages": np.int64(npg)}
    for c in cols:
        arrays[c] = rows[c]
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz_path)
    meta.update(complete=True, n_pairs=int(rows["n_pairs"]), n_pages=npg, n_pages_source=src_npg)
    meta_path.write_text(json.dumps(meta, indent=1))
    return npz_path


def load(npz_path: Path) -> dict:
    z = np.load(npz_path)
    out = {k: z[k] for k in z.files}
    out["n_pairs"] = int(out["n_pairs"])
    out["n_pages"] = int(out["n_pages"])
    return out
