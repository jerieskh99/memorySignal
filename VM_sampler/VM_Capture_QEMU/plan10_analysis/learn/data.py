#!/usr/bin/env python3
"""data.py -- what a run wrote, as a Dataset the pipeline reads.

  rows    features.npz: X (n, f), names = the metrics
  path    tiles.npz with shape path: X (n, W, k), k = blocks x channels; a series tile (n, W)
          is read as a path with k = 1
  image   tiles.npz with shape image: X (n, W, P), pages (P,)

Keys are the ones every run writes (recording, workload, family, block, t_index, seq_start)
plus campaign, derived from the recording id. Complex tiles become two real channels along the
last axis, magnitude then angle, and the dataset says so. Labels come from the keys: family,
workload or recording.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from plan10_analysis.learn.splits import campaign_of

KEYS = ("recording", "workload", "family", "block", "t_index", "seq_start")
_KERNEL_WL = re.compile(r"^kernel_(.+)_v2$")


def archetype_table() -> dict[str, str]:
    """kernel -> predicted archetype, imported from plan11's schema (never copied), so this
    preview and the toolkit can never disagree."""
    try:
        from plan11_encoding_ladder.schema import ARCHETYPE_OF
    except ImportError as e:
        raise DataError(f"the archetype table lives in plan11_encoding_ladder/schema.py and could not be imported: {e}")
    return dict(ARCHETYPE_OF)


def archetype_of_workload(workload: str, table: dict[str, str] | None = None) -> str:
    """kernel_<k>_v2 -> ARCHETYPE_OF[k]; "" for a workload that is not one of the twelve kernels."""
    table = table if table is not None else archetype_table()
    m = _KERNEL_WL.match(str(workload))
    return table.get(m.group(1), "") if m else ""


def kernel_of_workload(workload: str) -> str:
    m = _KERNEL_WL.match(str(workload))
    return m.group(1) if m else ""


class DataError(ValueError):
    pass


class Dataset:
    def __init__(self, run: str, source: str, shape: str, X: np.ndarray, keys: dict, names: list[str], meta: dict):
        self.run, self.source, self.shape, self.X, self.keys, self.names, self.meta = run, source, shape, X, keys, names, meta
        self.n = int(X.shape[0])

    def labels(self, target: str = "family") -> np.ndarray:
        if target == "archetype":
            table = archetype_table()
            return np.asarray([archetype_of_workload(w, table) for w in self.keys["workload"]], dtype=str)
        if target not in ("family", "workload", "recording"):
            raise DataError(f"target must be family, workload, recording or archetype, not {target!r}")
        return np.asarray(self.keys[target]).astype(str)

    def archetype_rows(self) -> np.ndarray:
        """Mask of the rows that have an archetype: workload kernel_<k>_v2 with k in the table.
        The idle cells and anything else are False; they are dropped for target archetype."""
        return self.labels("archetype") != ""

    def subset(self, idx: np.ndarray, why: str | None = None) -> "Dataset":
        idx = np.asarray(idx)
        keys = {k: np.asarray(v)[idx] for k, v in self.keys.items()}
        meta = dict(self.meta, rows_dropped=int(self.n - int(idx.sum() if idx.dtype == bool else idx.size)), rows_dropped_why=why)
        return Dataset(self.run, self.source, self.shape, self.X[idx], keys, self.names, meta)

    def describe(self) -> dict:
        return {"run": self.run, "source": self.source, "shape": self.shape, "n": self.n, "tile_shape": list(self.X.shape[1:]),
                "names": self.names if self.shape == "rows" else None,
                "n_families": len(set(self.keys["family"].tolist())), "n_workloads": len(set(self.keys["workload"].tolist())),
                "n_recordings": len(set(self.keys["recording"].tolist())), "meta": self.meta}


def _keys_of(tk: np.ndarray) -> dict:
    keys = {k: np.asarray(tk[k]) for k in KEYS if tk.dtype.names and k in tk.dtype.names}
    keys["campaign"] = np.asarray([campaign_of(r) for r in keys["recording"]])
    return keys


def load_rows(run_dir: Path) -> Dataset:
    f = Path(run_dir) / "features.npz"
    if not f.exists():
        raise DataError(f"{Path(run_dir).name}: no features.npz (add a Write module to the scheme, or read tiles)")
    z = np.load(f, allow_pickle=False)
    X = np.asarray(z["X"], dtype=np.float32)
    names = [str(n) for n in z["feature_names"]]
    if X.ndim != 2 or X.shape[1] != len(names):
        raise DataError(f"{Path(run_dir).name}: features.npz X is {X.shape} for {len(names)} names")
    return Dataset(Path(run_dir).name, "features", "rows", X, _keys_of(z["tile_keys"]), names, {"n_features": len(names)})


def load_tiles(run_dir: Path) -> Dataset:
    f = Path(run_dir) / "tiles.npz"
    if not f.exists():
        raise DataError(f"{Path(run_dir).name}: no tiles.npz (add a Write tiles module to the scheme)")
    z = np.load(f, allow_pickle=False)
    shape = str(z["shape"])
    X = np.asarray(z["X"])
    meta = {"written_shape": shape, "w": int(z["w"]), "h": int(z["h"]), "channels": [str(c) for c in z["channels"]],
            "n_blocks": int(z["n_blocks"]), "complex": bool(z["complex"])}
    if np.iscomplexobj(X):
        X = np.concatenate([np.abs(X), np.angle(X)], axis=-1).astype(np.float32)
        meta["complex_as"] = "magnitude then angle, along the last axis"
    else:
        X = X.astype(np.float32)
    if shape == "series":
        if X.ndim == 2:
            X = X[:, :, None]
        shape = "path"
        meta["n_blocks"] = meta["n_blocks"] or 1
    if shape == "image":
        meta["pages"] = np.asarray(z["pages"]).astype(int).tolist() if "pages" in z else None
    if shape not in ("path", "image") or X.ndim != 3:
        raise DataError(f"{Path(run_dir).name}: tiles.npz holds {shape} of {X.shape}, which no model here reads")
    names = [f"f{i}" for i in range(X.shape[-1])]
    return Dataset(Path(run_dir).name, "tiles", shape, X, _keys_of(z["tile_keys"]), names, meta)


def load(run_dir: Path, source: str = "features") -> Dataset:
    if source == "features":
        return load_rows(run_dir)
    if source == "tiles":
        return load_tiles(run_dir)
    raise DataError(f"source must be features or tiles, not {source!r}")


def keys_only(run_dir: Path, source: str = "features") -> dict:
    """Just the keys of a run's rows or tiles, without the arrays."""
    f = Path(run_dir) / ("features.npz" if source == "features" else "tiles.npz")
    if not f.exists():
        raise DataError(f"{Path(run_dir).name}: no {f.name}")
    z = np.load(f, allow_pickle=False)
    return _keys_of(z["tile_keys"])


def available(run_dir: Path) -> dict:
    """What a run directory offers the Learn view, read from its files and sidecar."""
    d = Path(run_dir)
    out = {"label": d.name, "features": (d / "features.npz").exists(), "tiles": (d / "tiles.npz").exists(), "tiles_shape": None,
           "n_rows": None, "n_features": None, "n_tiles": None, "tile_shape": None, "written_at": None}
    side = d / "sidecar.json"
    if side.exists():
        try:
            s = json.loads(side.read_text())
            out.update(n_rows=s.get("n_rows"), n_features=s.get("n_features"), written_at=s.get("written_at"))
            t = s.get("tiles") or {}
            if t:
                out.update(tiles_shape=("path" if t.get("shape") == "series" else t.get("shape")), n_tiles=t.get("n_tiles"), tile_shape=t.get("tile_shape"))
        except (OSError, json.JSONDecodeError):
            pass
    return out
