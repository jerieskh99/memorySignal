#!/usr/bin/env python3
"""results_view.py -- what a run produced, summarised and aggregated for the console's Explore view.

Pure functions over a run directory's features.npz (X, feature_names, tile_keys) and its
sidecar.json; no HTTP here. analysis_bridge.py exposes them as /results/summary and
/results/agg, and the Explore view draws from the numbers these return, never from rows it
aggregated itself in the browser.

Every statistic is computed by numpy in float64 from the float32 the run wrote, NaN-aware,
and rounded to 6 significant digits on the way out (the precision features.csv carries).
Percentiles use numpy's default linear interpolation; std is the sample standard deviation
(ddof=1), absent for n < 2.

Several runs aggregate as one frame: feature columns are aligned by name (the intersection,
in the first run's order), rows are stacked, and a `run` key is added. The same five views
then group by run, or by run and family, with every run read the same way.

Views (`aggregate`):
  distribution  per group: n, n_nan, min, p5, p25, median, p75, p95, max, mean, std, and a
                histogram over edges shared by every group (log-spaced when scale=log)
  time          per group, per x (t_index or seq_start): the chosen stat, p25, p75, n
  matrix        stat of one feature over (rows key x cols key), with n per cell
  scatter       two features per group, capped at MAX_POINTS by a fixed stride, with
                Pearson r and Spearman rho per group and overall
  table         stat of every feature per group
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np

KEYS = ("recording", "workload", "family", "block", "t_index", "seq_start")
NUMERIC_KEYS = frozenset({"block", "t_index", "seq_start"})
VIEWS = ("distribution", "time", "matrix", "scatter", "table")
STATS = ("median", "mean", "min", "max", "std", "p5", "p25", "p75", "p95", "count")
MAX_CELLS = 20000
MAX_POINTS = 5000
MAX_BINS = 200
_SEED = re.compile(r"seed_(\d+)")


class ViewError(ValueError):
    """The request names something the runs do not have, or a view or stat that is not one."""


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

class Run:
    def __init__(self, label: str, run_dir: Path, X: np.ndarray, names: list[str], keys: dict, sidecar: dict):
        self.label, self.dir, self.X, self.names, self.keys, self.sidecar = label, run_dir, X, names, keys, sidecar
        self.n_rows = int(X.shape[0])


def load_run(run_dir: Path) -> Run:
    f = run_dir / "features.npz"
    if not f.exists():
        raise ViewError(f"{run_dir.name}: no features.npz (the run has not written features, or wrote another format)")
    z = np.load(f, allow_pickle=False)
    for k in ("X", "feature_names", "tile_keys"):
        if k not in z:
            raise ViewError(f"{run_dir.name}: features.npz lacks {k}")
    X = np.asarray(z["X"], dtype=np.float64)
    names = [str(n) for n in z["feature_names"]]
    tk = z["tile_keys"]
    if X.ndim != 2 or X.shape[1] != len(names) or X.shape[0] != tk.shape[0]:
        raise ViewError(f"{run_dir.name}: X is {X.shape}, {len(names)} names, {tk.shape[0]} keys")
    keys = {k: np.asarray(tk[k]) for k in KEYS if tk.dtype.names and k in tk.dtype.names}
    side_f = run_dir / "sidecar.json"
    try:
        side = json.loads(side_f.read_text()) if side_f.exists() else {}
    except (json.JSONDecodeError, OSError):
        side = {}
    return Run(run_dir.name, run_dir, X, names, keys, side)


_CACHE: dict[str, tuple[tuple, Run]] = {}


def _signature(run_dir: Path) -> tuple:
    sig = []
    for name in ("features.npz", "sidecar.json"):
        try:
            st = os.stat(run_dir / name)
            sig.append((st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append(None)
    return tuple(sig)


def get_run(run_dir: Path) -> Run:
    """load_run, cached per directory until features.npz or sidecar.json changes on disk."""
    key, sig = str(run_dir), _signature(run_dir)
    hit = _CACHE.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    run = load_run(run_dir)
    _CACHE[key] = (sig, run)
    return run


def short_recording(rid: str) -> str:
    """`family/workload/variant/rep001__campaign` -> `seed_NNNN campaign`, the way the Monitor names traces."""
    parts = str(rid).split("/")
    m = _SEED.search(rid)
    tag = ("seed_" + m.group(1)) if m else (parts[2][-8:] if len(parts) >= 3 else rid)
    rep = parts[3] if len(parts) >= 4 else ""
    rep = rep.split("__", 1)[1] if "__" in rep else rep
    return f"{tag} {rep}".strip()


def short_names(ids) -> dict[str, str]:
    """Short label per recording id, made unique by appending the variant hash where two collide."""
    ids = [str(i) for i in ids]
    out = {i: short_recording(i) for i in ids}
    seen: dict[str, list[str]] = {}
    for i, s in out.items():
        seen.setdefault(s, []).append(i)
    for s, dup in seen.items():
        if len(dup) > 1:
            for i in dup:
                parts = i.split("/")
                out[i] = f"{s} {parts[2][-8:] if len(parts) >= 3 else i}"
    return out


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------

def _r(x):
    """6 significant digits, the precision features.csv carries; NaN and inf become null."""
    if x is None:
        return None
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.6g}")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (float, np.floating, np.integer)):
        return _r(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    return obj


def _finite(v: np.ndarray) -> np.ndarray:
    return v[np.isfinite(v)]


def _stats(v: np.ndarray) -> dict:
    fin = _finite(v)
    n = int(fin.size)
    d = {"n": n, "n_nan": int(v.size - n)}
    if n == 0:
        d.update({k: None for k in ("min", "p5", "p25", "median", "p75", "p95", "max", "mean", "std")})
        return d
    q = np.percentile(fin, [0, 5, 25, 50, 75, 95, 100])
    d.update(min=q[0], p5=q[1], p25=q[2], median=q[3], p75=q[4], p95=q[5], max=q[6],
             mean=float(fin.mean()), std=(float(fin.std(ddof=1)) if n > 1 else None))
    return d


def _stat(fin: np.ndarray, name: str):
    """One statistic of an already-finite array; None where it is undefined."""
    if name == "count":
        return int(fin.size)
    if fin.size == 0:
        return None
    if name == "median":
        return float(np.median(fin))
    if name == "mean":
        return float(fin.mean())
    if name == "min":
        return float(fin.min())
    if name == "max":
        return float(fin.max())
    if name == "std":
        return float(fin.std(ddof=1)) if fin.size > 1 else None
    if name in ("p5", "p25", "p75", "p95"):
        return float(np.percentile(fin, int(name[1:])))
    raise ViewError(f"unknown stat {name!r}; one of {', '.join(STATS)}")


def _rankdata(v: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties sharing the mean of the positions they occupy."""
    u, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    return ((starts + ends) / 2.0)[inv]


def _pearson(a: np.ndarray, b: np.ndarray):
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray):
    if a.size < 2:
        return None
    return _pearson(_rankdata(a), _rankdata(b))


def _split_by(codes: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Row indices per code 0..n_groups-1, each in ascending row order."""
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_groups + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n_groups)]


# ---------------------------------------------------------------------------
# frames: one run or several, aligned by feature name
# ---------------------------------------------------------------------------

class Frame:
    def __init__(self, X, names, keys, runs, excluded):
        self.X, self.names, self.keys, self.runs, self.excluded = X, names, keys, runs, excluded
        self.n = int(X.shape[0])
        self.shorts = short_names(np.unique(keys["recording"])) if "recording" in keys else {}

    def feature(self, name: str) -> int:
        if name not in self.names:
            raise ViewError(f"no metric {name!r} in {'these runs' if len(self.runs) > 1 else 'this run'}; there are: {', '.join(self.names)}")
        return self.names.index(name)

    def display(self, key: str, value) -> str:
        if key == "recording":
            return self.shorts.get(str(value), str(value))
        return str(value)


def make_frame(runs: list[Run], need: list[str] | None = None) -> Frame:
    if not runs:
        raise ViewError("no runs given")
    need = [n for n in (need or []) if n]
    used, excluded = [], []
    for r in runs:
        missing = [n for n in need if n not in r.names]
        if missing:
            excluded.append({"label": r.label, "missing": missing})
        else:
            used.append(r)
    if not used:
        raise ViewError("no run has " + ", ".join(need))
    names = [n for n in used[0].names if all(n in r.names for r in used)]
    if not names:
        raise ViewError("the runs share no metric name")
    X = np.vstack([r.X[:, [r.names.index(n) for n in names]] for r in used])
    keys = {}
    for k in KEYS:
        if all(k in r.keys for r in used):
            keys[k] = np.concatenate([r.keys[k] for r in used])
    keys["run"] = np.concatenate([np.full(r.n_rows, r.label, dtype=f"U{max(1, len(r.label))}") for r in used]).astype(str)
    return Frame(X, names, keys, [r.label for r in used], excluded)


def _groups(fr: Frame, group: list[str]):
    """(labels, key dicts, row-index arrays) for the combination of the given keys, in natural
    order: numeric keys ascending, runs in the order given, everything else lexical."""
    group = [g for g in group if g]
    if not group:
        return ["all"], [{}], [np.arange(fr.n)]
    uniq, codes = [], []
    for g in group:
        if g not in fr.keys:
            raise ViewError(f"unknown key {g!r}; keys here: {', '.join(sorted(fr.keys))}")
        u, inv = np.unique(fr.keys[g], return_inverse=True)
        inv = inv.reshape(-1)
        if g == "run":
            order = sorted(range(len(u)), key=lambda i: fr.runs.index(str(u[i])))
            remap = np.empty(len(order), dtype=np.int64)
            remap[order] = np.arange(len(order))
            u, inv = u[order], remap[inv]
        uniq.append(u)
        codes.append(inv.astype(np.int64))
    comb = np.zeros(fr.n, dtype=np.int64)
    for u, inv in zip(uniq, codes):
        comb = comb * len(u) + inv
    ucomb, inv2 = np.unique(comb, return_inverse=True)
    idx = _split_by(inv2.reshape(-1), len(ucomb))
    labels, kds = [], []
    for code in ucomb.tolist():
        vals = []
        for u in reversed(uniq):
            code, i = divmod(code, len(u))
            vals.append(u[i])
        vals.reverse()
        kd = {g: (v.item() if hasattr(v, "item") else v) for g, v in zip(group, vals)}
        labels.append(" / ".join(fr.display(g, v) for g, v in zip(group, vals)))
        kds.append(kd)
    return labels, kds, idx


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------

def distribution(fr: Frame, y: str, group: list[str], bins: int = 30, scale: str = "linear") -> dict:
    col = fr.X[:, fr.feature(y)]
    fin = np.isfinite(col)
    if scale == "log":
        ok = fin & (col > 0)
        n_excl = int(fin.sum() - ok.sum())
    else:
        ok, n_excl = fin, 0
    bins = max(2, min(int(bins), MAX_BINS))
    base = col[ok]
    edges = np.zeros(0)
    if base.size:
        lo, hi = float(base.min()), float(base.max())
        if hi <= lo:
            pad = abs(lo) * 1e-6 or 1e-9
            lo, hi = lo - pad, hi + pad
        if scale == "log":
            lo = max(lo, np.finfo(float).tiny)
            edges = np.geomspace(lo, hi, bins + 1)
        else:
            edges = np.linspace(lo, hi, bins + 1)
    labels, kds, idx = _groups(fr, group)
    out = []
    for lab, kd, ix in zip(labels, kds, idx):
        v = col[ix]
        d = {"label": lab, "key": kd, **_stats(v)}
        vv = v[ok[ix]]
        d["counts"] = np.histogram(vv, bins=edges)[0].tolist() if edges.size else []
        out.append(d)
    return {"view": "distribution", "y": y, "group": group, "scale": scale, "bins": bins, "edges": edges,
            "n_excluded_nonpositive": n_excl, "groups": out}


def time_series(fr: Frame, y: str, x: str = "t_index", group: list[str] = ("recording",), stat: str = "median") -> dict:
    if x not in NUMERIC_KEYS:
        raise ViewError(f"the x axis must be one of {', '.join(sorted(NUMERIC_KEYS))}, not {x!r}")
    if x not in fr.keys:
        raise ViewError(f"no key {x!r} in these runs")
    if stat not in STATS:
        raise ViewError(f"unknown stat {stat!r}; one of {', '.join(STATS)}")
    col = fr.X[:, fr.feature(y)]
    xs = fr.keys[x]
    labels, kds, idx = _groups(fr, list(group))
    series = []
    for lab, kd, ix in zip(labels, kds, idx):
        ux, inv = np.unique(xs[ix], return_inverse=True)
        pieces = [col[ix][p] for p in _split_by(inv.reshape(-1), len(ux))]
        ys, lo, hi, n = [], [], [], []
        for p in pieces:
            f = _finite(p)
            n.append(int(f.size))
            if f.size == 0:
                ys.append(0 if stat == "count" else None), lo.append(None), hi.append(None)
                continue
            ys.append(_stat(f, stat))
            if f.size > 1:
                q = np.percentile(f, [25, 75])
                lo.append(q[0]), hi.append(q[1])
            else:
                lo.append(None), hi.append(None)
        series.append({"label": lab, "key": kd, "x": ux.tolist(), "y": ys, "lo": lo, "hi": hi, "n": n, "n_rows": int(ix.size)})
    return {"view": "time", "y": y, "x": x, "group": list(group), "stat": stat, "band": "p25-p75 where n > 1", "series": series}


def matrix(fr: Frame, y: str, rows: str, cols: str, stat: str = "median") -> dict:
    if stat not in STATS:
        raise ViewError(f"unknown stat {stat!r}; one of {', '.join(STATS)}")
    rl, _rk, ridx = _groups(fr, [rows])
    cl, _ck, cidx = _groups(fr, [cols])
    if len(rl) * len(cl) > MAX_CELLS:
        raise ViewError(f"{len(rl)} x {len(cl)} = {len(rl) * len(cl)} cells; the cap is {MAX_CELLS}. Pick coarser keys.")
    col = fr.X[:, fr.feature(y)]
    rcode = np.empty(fr.n, dtype=np.int64)
    ccode = np.empty(fr.n, dtype=np.int64)
    for i, ix in enumerate(ridx):
        rcode[ix] = i
    for i, ix in enumerate(cidx):
        ccode[ix] = i
    comb = rcode * len(cl) + ccode
    values = [[0 if stat == "count" else None] * len(cl) for _ in rl]
    ns = [[0] * len(cl) for _ in rl]
    u, inv = np.unique(comb, return_inverse=True)
    for code, ix in zip(u.tolist(), _split_by(inv.reshape(-1), len(u))):
        r, c = divmod(int(code), len(cl))
        f = _finite(col[ix])
        ns[r][c] = int(f.size)
        values[r][c] = _stat(f, stat) if (f.size or stat == "count") else None
    return {"view": "matrix", "y": y, "rows": rows, "cols": cols, "stat": stat,
            "row_labels": rl, "col_labels": cl, "values": values, "n": ns,
            "n_empty": sum(1 for row in ns for v in row if v == 0)}


def scatter(fr: Frame, x: str, y: str, group: list[str], max_points: int = MAX_POINTS) -> dict:
    a, b = fr.X[:, fr.feature(x)], fr.X[:, fr.feature(y)]
    ok = np.isfinite(a) & np.isfinite(b)
    total = int(ok.sum())
    stride = max(1, math.ceil(total / max(1, int(max_points))))
    labels, kds, idx = _groups(fr, group)
    groups = []
    for lab, kd, ix in zip(labels, kds, idx):
        sel = ix[ok[ix]]
        sub = sel[::stride]
        groups.append({"label": lab, "key": kd, "n": int(sel.size), "n_shown": int(sub.size),
                       "r_pearson": _pearson(a[sel], b[sel]), "rho_spearman": _spearman(a[sel], b[sel]),
                       "points": np.column_stack([a[sub], b[sub]]).tolist() if sub.size else []})
    return {"view": "scatter", "x": x, "y": y, "group": group, "n": total, "stride": stride,
            "r_pearson": _pearson(a[ok], b[ok]), "rho_spearman": _spearman(a[ok], b[ok]), "groups": groups}


def table(fr: Frame, group: list[str], stat: str = "median") -> dict:
    if stat not in STATS:
        raise ViewError(f"unknown stat {stat!r}; one of {', '.join(STATS)}")
    labels, kds, idx = _groups(fr, group)
    rows = []
    for lab, kd, ix in zip(labels, kds, idx):
        sub = fr.X[ix]
        rows.append({"label": lab, "key": kd, "n": int(ix.size),
                     "values": [_stat(_finite(sub[:, j]), stat) for j in range(len(fr.names))]})
    return {"view": "table", "group": group, "stat": stat, "features": fr.names, "rows": rows}


def aggregate(runs: list[Run], view: str, y: str | None = None, x: str | None = None, group=(), stat: str = "median",
              scale: str = "linear", bins: int = 30, rows: str | None = None, cols: str | None = None) -> dict:
    """One view over one run or several; the resolved choices are echoed back so the page can
    show what it drew when it left a choice to the default."""
    if view not in VIEWS:
        raise ViewError(f"unknown view {view!r}; one of {', '.join(VIEWS)}")
    if stat not in STATS:
        raise ViewError(f"unknown stat {stat!r}; one of {', '.join(STATS)}")
    if scale not in ("linear", "log"):
        raise ViewError(f"scale is linear or log, not {scale!r}")
    group = [g for g in (group or []) if g]
    fr = make_frame(runs, [n for n in ((x if view == "scatter" else None), y) if n])
    if view != "table":
        y = y or (fr.names[1] if view == "scatter" and len(fr.names) > 1 else fr.names[0])
    if view == "scatter":
        x = x or fr.names[0]
    if view == "distribution":
        out = distribution(fr, y, group, bins, scale)
    elif view == "time":
        out = time_series(fr, y, x or "t_index", group or ["recording"], stat)
    elif view == "matrix":
        out = matrix(fr, y, rows or ("run" if len(fr.runs) > 1 else "workload"), cols or "t_index", stat)
    elif view == "scatter":
        out = scatter(fr, x, y, group)
    else:
        out = table(fr, group, stat)
    out["runs"] = fr.runs
    out["excluded"] = fr.excluded
    out["features"] = fr.names
    out["keys"] = sorted(fr.keys)
    out["n_rows"] = fr.n
    return _clean(out)


# ---------------------------------------------------------------------------
# summary: what one run holds
# ---------------------------------------------------------------------------

def summary(run: Run) -> dict:
    feats = [{"name": n, **_stats(run.X[:, j])} for j, n in enumerate(run.names)]
    keys = {}
    shorts = short_names(np.unique(run.keys["recording"])) if "recording" in run.keys else {}
    for k in KEYS:
        if k not in run.keys:
            continue
        u, c = np.unique(run.keys[k], return_counts=True)
        if k in NUMERIC_KEYS:
            keys[k] = {"n_unique": int(len(u)), "min": u.min(), "max": u.max()}
        else:
            vals = [{"value": str(v), "n": int(n)} for v, n in zip(u, c)]
            if k == "recording":
                for d in vals:
                    d["short"] = shorts.get(d["value"], d["value"])
            keys[k] = {"n_unique": int(len(u)), "values": vals}
    side = run.sidecar or {}
    src = side.get("source") or {}
    nodes = (side.get("scheme") or {}).get("nodes") or []
    win = next((n.get("params") for n in nodes if n.get("module") == "window"), None)
    facts = {
        "written_at": side.get("written_at"),
        "speed": side.get("speed"), "speed_source": side.get("speed_source"),
        "max_pairs": side.get("max_pairs"),
        "extraction_ran": side.get("extraction_ran") or src.get("mode") or ("local" if src.get("kind") == "local" else None),
        "source": {k: src.get(k) for k in ("kind", "host", "user", "root", "remote_root", "mode") if src.get(k) is not None},
        "acknowledged": [a.get("id") if isinstance(a, dict) else a for a in (side.get("acknowledged") or [])],
        "modules": [n.get("module") for n in nodes],
        "window": {k: win.get(k) for k in ("w", "h", "edge") if k in win} if isinstance(win, dict) else None,
        "n_recordings": len(side.get("recordings") or []),
        "differ": side.get("differ"), "numpy": side.get("numpy"), "python": side.get("python"),
        "format": side.get("format"),
        "files": {n: str(run.dir / n) for n in ("features.npz", "features.csv", "sidecar.json") if (run.dir / n).exists()},
    }
    return _clean({"label": run.label, "n_rows": run.n_rows, "n_features": len(run.names),
                   "features": feats, "keys": keys, "facts": facts})
