#!/usr/bin/env python3
"""series.py -- per-rung per-snapshot series from an extract, head drop, level normalization,
windowing, the eight shape features, feature matrices (SPEC section 3.1), and the shared file
helpers every builder 2 module uses (cells.csv, extract.csv, sidecar.json, result files with
their ``schema`` / ``params`` / ``citation`` block, input hashes).

Citation: SPEC.md section 3.1; P2_STRUCTURE.md section IV (the rungs) and section V G-L (i)
(the normalization rule per rung); CR 2.2 item 24 (level normalization); CR 2.1 item 19 (per-cell
statistics, (W, H) per encoding never per kernel).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import schema  # builder 1's; the epoch-1 fallback _schema_compat.py is deleted (SPEC_epoch2 B21)
from plan11_encoding_ladder import verdicts as V

RUNGS = ("apf", "wapf", "persist", "content", "combined")
AXIS_OF_RUNG = {"apf": "breadth", "wapf": "breadth x amount folded", "persist": "identity over time",
                "content": "amount", "combined": "all three"}
PAIR_RUNGS = ("persist", "content", "combined", "cmp_dhodapkar", "cmp_law")   # epoch 2: the two comparators that read pair adjacency
TEMPORAL_CHANNEL = "r_l0_q50_per"                        # SPEC 3.5 head; section 8 item 9
WAPF_NORM_DEFAULT = "median_K"                           # section 8 item 7
DUTY_RULE = "0.1 * max"                                  # section 8 item 8
FEAT = ("mean", "std", "cov", "median", "max", "p95", "peak2med", "duty")
WHOLE_GRID_ID = "Wall_Hall"
GF_DEFAULT_GRID = "W8_H4"                                # section 8 item 38
CONTENT_PRIMARY = ("r_l0_q50_per", "r_l1l0_q50_per", "r_haml0_q50_per")
CONTENT_CHANNELS = tuple(list(CONTENT_PRIMARY) + [c for c in schema.EXTRACT_COLUMNS
                                                  if c.startswith("r_") and c not in CONTENT_PRIMARY])
assert len(CONTENT_CHANNELS) == 15

N = schema.N_PAGES
BITS = schema.BITS_PER_PAGE


# --------------------------------------------------------------------------- file helpers

def sha256_file(path) -> str:
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def inputs_sha256(paths, out: Path | None = None) -> dict:
    """{relative path: sha256} of every input file read (SPEC_review_al_farabi.md item 2.8)."""
    res = {}
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        key = str(p.relative_to(out)) if out and str(p).startswith(str(out)) else str(p)
        res[key] = sha256_file(p)
    return res


def fmt_num(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return ""
        return format(float(x), ".10g")
    return str(x)


def write_csv(path: Path, columns, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(columns))
        for r in rows:
            if isinstance(r, dict):
                w.writerow([fmt_num(r.get(c)) for c in columns])
            else:
                w.writerow([fmt_num(v) for v in r])
    tmp.replace(path)
    return path


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def write_json(path: Path, schema_name: str, params: dict, citation: str, payload: dict) -> Path:
    """Every JSON result: ``schema``, ``params``, ``citation``, then the payload (SPEC 1)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": schema_name, "params": params, "citation": citation}
    doc.update(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=_json_default))
    tmp.replace(path)
    return path


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_params(path: Path, schema_name: str, params: dict, citation: str) -> Path:
    """A CSV-only result gets ``<name>.params.json`` beside it with the same three keys."""
    return write_json(Path(str(path) + ".params.json").with_name(Path(path).stem + ".params.json"),
                      schema_name, params, citation, {})


def to_float(s) -> float:
    if s is None or s == "" or s == "None":
        return float("nan")
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- cells, sidecars, extracts

def load_cells(cells_csv: Path, *, only_ok: bool = True) -> list[dict]:
    rows = read_csv(cells_csv)
    for r in rows:
        r["rep"] = int(float(r["rep"])) if r.get("rep") not in (None, "") else 0
        r["seed"] = int(float(r["seed"])) if r.get("seed") not in (None, "") else None
    if only_ok:
        rows = [r for r in rows if r.get("status", "ok") == "ok"]
    return rows


def cells_csv_path(out: Path, cells_csv: Path | None = None) -> Path:
    return Path(cells_csv) if cells_csv else Path(out) / "cells.csv"


def sidecar_path(out: Path, cell_id: str) -> Path:
    return Path(out) / "extract" / cell_id / "sidecar.json"


def extract_path(out: Path, cell_id: str) -> Path:
    return Path(out) / "extract" / cell_id / "extract.csv"


def load_sidecar(out: Path, cell_id: str) -> dict:
    return read_json(sidecar_path(out, cell_id))


def load_extract(out: Path, cell_id: str) -> dict:
    """The per-cell extract as {column: float64 array} (blank -> NaN; ``seq`` int64)."""
    p = extract_path(out, cell_id)
    with p.open(newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        data = list(rd)
    cols = {}
    arr = np.array(data, dtype=object) if data else np.zeros((0, len(header)), dtype=object)
    for j, name in enumerate(header):
        col = arr[:, j] if len(data) else np.zeros(0, dtype=object)
        vals = np.array([to_float(v) for v in col], dtype=np.float64)
        cols[name] = vals
    cols["seq"] = cols["seq"].astype(np.int64)
    cols["_n_rows"] = len(data)
    cols["_path"] = p
    return cols


_EXTRACT_CACHE: dict = {}


def load_extract_cached(out: Path, cell_id: str) -> dict:
    key = (str(out), cell_id)
    if key not in _EXTRACT_CACHE:
        _EXTRACT_CACHE[key] = load_extract(out, cell_id)
    return _EXTRACT_CACHE[key]


# --------------------------------------------------------------------------- head drop (3.1.2)

HEAD_DROP_COLUMNS = ("kernel", "head_drop_pairs", "reason")


def write_head_drop_template(path: Path) -> Path:
    """inputs/head_drop.csv, default 0 for every kernel (SPEC 3.1.2; section 8 item 6)."""
    rows = [{"kernel": k, "head_drop_pairs": 0,
             "reason": "default 0; the lexer's first pass is the author's number (P2 Sec. VI)"}
            for k, _ in schema.KERNELS]
    rows.append({"kernel": "idle", "head_drop_pairs": 0, "reason": "control"})
    return write_csv(path, HEAD_DROP_COLUMNS, rows)


def load_head_drop(path: Path | None) -> dict:
    """{kernel: pairs to drop}; missing file or kernel -> 0 (SPEC 3.1.2)."""
    if path is None or not Path(path).is_file():
        return {}
    out = {}
    for r in read_csv(path):
        try:
            out[r["kernel"]] = int(float(r.get("head_drop_pairs") or 0))
        except ValueError:
            out[r["kernel"]] = 0
    return out


def head_drop_for(head_drop: dict | int | None, kernel: str, role: str | None = None) -> int:
    """The pairs to drop for one cell (SPEC 3.1.2; ``inputs/head_drop.csv`` keyed by kernel with the
    row ``idle`` for the control cells, ``series.write_head_drop_template``). ``role == "idle"``
    selects the ``idle`` row whatever ``kernel`` says (the index names an idle cell's kernel after its
    test label, ``sleep`` on the synthetic corpus; CHECK_3 M9, SPEC_epoch2 B9). The two-argument form
    keeps working: with ``role`` omitted the key is ``kernel``. A missing file or key gives 0."""
    if head_drop is None:
        return 0
    if isinstance(head_drop, int):
        return head_drop
    key = "idle" if role == "idle" else kernel
    return int(head_drop.get(key, 0))


# --------------------------------------------------------------------------- rung series (3.1.1)

def k_median_cell(ex: dict, head_drop: int = 0) -> float:
    """Median of K over the cell's rows after head drop (all remaining rows, the last seq included);
    P2 Sec. V 'level normalization from per-cell statistics'; CR 2.1 item 19."""
    K = ex["K"][head_drop:]
    return float(np.median(K)) if len(K) else float("nan")


def rung_channels(rung: str, normalized: bool) -> tuple[str, ...]:
    if rung == "apf":
        return ("k_over_med",) if normalized else ("k_over_n",)
    if rung == "wapf":
        return ("wapf_norm",) if normalized else ("wapf",)
    if rung == "persist":
        return ("j_excess",) if normalized else ("j",)
    if rung == "content":
        return CONTENT_CHANNELS
    if rung == "combined":
        if not normalized:
            raise ValueError("the raw variant of combined is not defined (SPEC 3.1.1)")
        return ("k_over_med", "wapf_norm", "j_excess") + CONTENT_CHANNELS
    raise ValueError(rung)


def rung_series(ex: dict, rung: str, *, normalized: bool, head_drop: int = 0,
                wapf_norm: str = WAPF_NORM_DEFAULT) -> np.ndarray:
    """The per-snapshot channel matrix S[rung], shape [n_series, d], n_series = n_pairs - 1 -
    head_drop (SPEC 3.1.1 table). apf: K/N raw, K/K_median_cell normalized (P2 Sec. V G-L (i);
    CR 2.2 item 24). wapf: ham_sum_all/(N*32768) raw; normalized by K_median_cell*32768
    (wapf_norm='median_K', the literal count-rung rule) or by the cell's own median wapf
    ('median_self'); section 8 item 7. persist: J raw, J - J_null normalized (J against its
    independence null). content: the 15 r_*_per columns, primary three first, level-free so raw ==
    normalized; K is not a feature of this rung (P2 Sec. IV rung 2). combined: the four normalized
    matrices concatenated (P2 Sec. IV rung 3). Blank J or r_* -> NaN (imputed per fold, 4.2)."""
    n_rows = int(ex["_n_rows"])
    lo, hi = head_drop, n_rows - 1
    if hi <= lo:
        return np.zeros((0, len(rung_channels(rung, normalized))))
    kmed = k_median_cell(ex, head_drop)
    if rung == "apf":
        K = ex["K"][lo:hi]
        x = K / kmed if normalized else K / N
        return x[:, None].astype(np.float64)
    if rung == "wapf":
        h = ex["ham_sum_all"][lo:hi]
        if not normalized:
            x = h / (N * BITS)
        elif wapf_norm == "median_K":
            x = h / (kmed * BITS)
        elif wapf_norm == "median_self":
            w = h / (N * BITS)
            m = float(np.median(w))
            x = w / m if m else w
        else:
            raise ValueError(wapf_norm)
        return x[:, None].astype(np.float64)
    if rung == "persist":
        J = ex["J"][lo:hi]
        x = J - ex["J_null"][lo:hi] if normalized else J
        return x[:, None].astype(np.float64)
    if rung == "content":
        return np.stack([ex[c][lo:hi] for c in CONTENT_CHANNELS], axis=1).astype(np.float64)
    if rung == "combined":
        if not normalized:
            raise ValueError("the raw variant of combined is not defined (SPEC 3.1.1)")
        parts = [rung_series(ex, r, normalized=True, head_drop=head_drop, wapf_norm=wapf_norm)
                 for r in ("apf", "wapf", "persist", "content")]
        return np.concatenate(parts, axis=1)
    raise ValueError(rung)


def temporal_series(ex: dict, rung: str, head_drop: int = 0, wapf_norm: str = WAPF_NORM_DEFAULT) -> np.ndarray:
    """The level-normalized single-channel series the temporal gates run on: the rung's one
    channel, or ``r_l0_q50_per`` for content and combined (SPEC 3.5 head; section 8 item 9)."""
    S = rung_series(ex, rung, normalized=True, head_drop=head_drop, wapf_norm=wapf_norm)
    if rung in ("content", "combined"):
        names = rung_channels(rung, True)
        return S[:, names.index(TEMPORAL_CHANNEL)]
    return S[:, 0]


def cell_headline(ex: dict, rung: str, head_drop: int = 0) -> float:
    """Per-cell headline reading (SPEC 3.1.5): apf median K/N; wapf median wapf; persist median J;
    content median r_l0_q50_per; combined not defined (G-F (ii) runs on the four rungs)."""
    if rung == "combined":
        raise ValueError("combined has no headline reading (SPEC 3.1.5)")
    x = rung_series(ex, rung, normalized=False, head_drop=head_drop)[:, 0]
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float("nan")


# --------------------------------------------------------------------------- windows (3.1.3)

def n_windows(n: int, W: int, H: int) -> int:
    """Plan 02 ``compute_n_windows``: ``(n - W) // H + 1`` if ``n >= W`` else 0 (SPEC 3.1.3)."""
    if n < W:
        return 0
    return max(0, (n - W) // H + 1)


def grid_points_ids() -> list[tuple]:
    """The 13 declared points as ``(grid_id, W, H)``; the whole cell is ``("Wall_Hall", None, None)``
    and resolves per cell to ``W = H = n_series`` (SPEC 3.1.3; K2 Sec. 2 rung 0 (b))."""
    pts = []
    for W in schema.GRID_WINDOWS:
        if W == "whole":
            pts.append((WHOLE_GRID_ID, None, None))
            continue
        for r in schema.GRID_HOP_RATIOS:
            H = max(1, round(W * r))
            pts.append((schema.grid_id(W, H), W, H))
    assert len(pts) == 13
    return pts


def parse_grid_id(grid_id: str) -> tuple:
    """``"W8_H4" -> (8, 4)``; ``"Wall_Hall" -> (None, None)``."""
    if grid_id == WHOLE_GRID_ID:
        return None, None
    w, h = grid_id.split("_")
    return int(w[1:]), int(h[1:])


def resolve_wh(W, H, n_series: int) -> tuple[int, int]:
    if W is None:
        return n_series, n_series
    return int(W), int(H)


def hop_ratio(W, H) -> float:
    if W is None or not W:
        return 1.0
    return H / W


# --------------------------------------------------------------------------- shape features (3.1.4)

def _pct(xs_sorted, q):
    # copied from plan08_b1/b1_features.py:_pct, 2026-09-16
    if not xs_sorted:
        return 0.0
    return xs_sorted[min(len(xs_sorted) - 1, int(q * len(xs_sorted)))]


def features(xs) -> dict:
    # copied from plan08_b1/b1_features.py:features, 2026-09-16 (verbatim; duty = 0.1 * max)
    if not xs:
        return {k: 0.0 for k in FEAT}
    s = sorted(xs)
    mean = st.fmean(xs)
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    med = st.median(xs)
    mx = s[-1]
    thr = 0.1 * mx
    return {
        "mean": mean, "std": sd, "cov": (sd / mean if mean else 0.0),
        "median": med, "max": mx, "p95": _pct(s, 0.95),
        "peak2med": (mx / med if med else 0.0),
        "duty": sum(1 for x in xs if x > thr) / len(xs),
    }


def shape_features(x) -> np.ndarray:
    """The eight B1 shape features of one window, ``(mean, std_population, cov, median, max, p95,
    peak2med, duty)`` with the zero guards of ``b1_features.features`` (SPEC 3.1.4). Any NaN in the
    window -> eight NaNs (the imputer handles them, SPEC 3.1.5)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(8)
    if np.any(np.isnan(x)):
        return np.full(8, np.nan)
    f = features(x.tolist())
    return np.array([f[k] for k in FEAT], dtype=np.float64)


def shape_features_windows(x: np.ndarray, W: int, H: int) -> np.ndarray:
    """Vectorized ``shape_features`` over every window of ``x`` (rows ``[i*H, i*H+W)``);
    ``[n_windows, 8]``. Equal to the per-window copy within floating rounding (tested)."""
    x = np.asarray(x, dtype=np.float64)
    nw = n_windows(len(x), W, H)
    if nw == 0:
        return np.zeros((0, 8))
    idx = np.arange(nw)[:, None] * H + np.arange(W)[None, :]
    blk = x[idx]                                        # [nw, W]
    out = np.full((nw, 8), np.nan)
    ok = ~np.any(np.isnan(blk), axis=1)
    if not np.any(ok):
        return out
    b = blk[ok]
    mean = b.mean(axis=1)
    sd = b.std(axis=1) if W > 1 else np.zeros(len(b))
    s = np.sort(b, axis=1)
    med = np.median(b, axis=1)
    mx = s[:, -1]
    p95 = s[:, min(W - 1, int(0.95 * W))]
    thr = 0.1 * mx
    duty = (b > thr[:, None]).sum(axis=1) / W
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.where(mean != 0, sd / mean, 0.0)
        p2m = np.where(med != 0, mx / med, 0.0)
    out[ok] = np.stack([mean, sd, cov, med, mx, p95, p2m, duty], axis=1)
    return out


def channel_plan(rung: str) -> list[tuple]:
    """Per rung the (channel index, mode, channel name, name prefix) list of SPEC 3.1.5:
    apf/wapf/persist the 8 shape features of the one channel; content the 8 shape features of the
    three primary channels plus the window mean of the 12 secondary quantile channels;
    combined = apf + wapf + persist + content (60), names carry the source rung."""
    if rung in ("apf", "wapf", "persist"):
        return [(0, "shape", rung_channels(rung, True)[0], rung)]
    if rung == "content":
        plan = [(i, "shape", c, "content") for i, c in enumerate(CONTENT_PRIMARY)]
        plan += [(i + 3, "wmean", c, "content") for i, c in enumerate(CONTENT_CHANNELS[3:])]
        return plan
    if rung == "combined":
        plan = [(0, "shape", "k_over_med", "apf"), (1, "shape", "wapf_norm", "wapf"),
                (2, "shape", "j_excess", "persist")]
        plan += [(i + 3, "shape", c, "content") for i, c in enumerate(CONTENT_PRIMARY)]
        plan += [(i + 6, "wmean", c, "content") for i, c in enumerate(CONTENT_CHANNELS[3:])]
        return plan
    raise ValueError(rung)


def feature_names(rung: str, normalized: bool = True) -> list[str]:
    names = []
    for _, mode, ch, prefix in channel_plan(rung):
        if not normalized and prefix in ("apf", "wapf", "persist"):
            ch = rung_channels(prefix, False)[0]
        if mode == "shape":
            names += [f"{prefix}.{ch}.{f}" for f in FEAT]
        else:
            names.append(f"{prefix}.{ch}.wmean")
    return names


def window_features(S: np.ndarray, rung: str, W: int, H: int) -> tuple:
    """Window feature matrix ``[n_windows_kept, d]``, window starts, and the count of windows
    dropped because every channel value was NaN (SPEC 3.1.5)."""
    n = S.shape[0]
    nw = n_windows(n, W, H)
    if nw == 0:
        return np.zeros((0, len(feature_names(rung)))), np.zeros(0, dtype=np.int64), 0
    plan = channel_plan(rung)
    parts = []
    for ci, mode, _, _ in plan:
        x = S[:, ci]
        if mode == "shape":
            parts.append(shape_features_windows(x, W, H))
        else:
            idx = np.arange(nw)[:, None] * H + np.arange(W)[None, :]
            parts.append(x[idx].mean(axis=1)[:, None])
    F = np.concatenate(parts, axis=1)
    starts = np.arange(nw) * H
    idx = np.arange(nw)[:, None] * H + np.arange(W)[None, :]
    all_nan = np.all(np.isnan(S[idx]), axis=(1, 2))
    keep = ~all_nan
    return F[keep], starts[keep], int((~keep).sum())


# --------------------------------------------------------------------------- admissibility

def preconditions_map(out: Path) -> dict:
    """{cell_id: row} of gates/preconditions.csv when it exists, else {}."""
    p = Path(out) / "gates" / "preconditions.csv"
    if not p.is_file():
        return {}
    return {r["cell_id"]: r for r in read_csv(p)}


def admissible_cells(out: Path, cells: list[dict], rung: str | None = None) -> tuple:
    """Cells that enter a stage: status ok in cells.csv, ``all_hard_pass`` in preconditions.csv
    when it exists (SPEC 3.3.1), and for the pair rungs (persist, content, combined) a
    ``failed_verdict`` that is not a refusal (SPEC_review_al_farabi.md item 2.4). Returns
    (kept, excluded_hard, excluded_pair_rungs, preconditions_present)."""
    pm = preconditions_map(out)
    kept, ex_hard, ex_pair = [], [], []
    for c in cells:
        if c.get("status", "ok") != "ok":
            ex_hard.append(c["cell_id"])
            continue
        row = pm.get(c["cell_id"])
        if row is not None and str(row.get("all_hard_pass", "true")).lower() != "true":
            ex_hard.append(c["cell_id"])
            continue
        if rung in PAIR_RUNGS and row is not None and V.is_refusal(row.get("failed_verdict", V.PASS)):
            ex_pair.append(c["cell_id"])
            continue
        kept.append(c)
    return kept, ex_hard, ex_pair, bool(pm)


def gk0_relabel(out: Path) -> dict:
    """{kernel: archetype_measured} for kernels relabelled IDLE by G-K0 (gates/gk0.csv), else {}.
    Applied at read time, never stored in a feature file (SPEC_review_al_farabi.md item 2.3)."""
    p = Path(out) / "gates" / "gk0.csv"
    if not p.is_file():
        return {}
    res = {}
    for r in read_csv(p):
        if r.get("verdict") == V.GK0_IDLE_MEASURED:
            res[r["kernel"]] = r.get("archetype_measured") or "IDLE"
    return res


def gc_verdict(out: Path, rung: str) -> str | None:
    """The rung's ``rep = all`` G-C verdict from gates/gc.csv, or None when G-C has not run
    (SPEC_review_al_farabi.md item 2.5: written into every later result file's params)."""
    p = Path(out) / "gates" / "gc.csv"
    if not p.is_file():
        return None
    for r in read_csv(p):
        if r.get("rung") == rung and r.get("rep") == "all":
            return r.get("verdict")
    return None


def selected_grid_id(out: Path, rung: str, default: str | None = GF_DEFAULT_GRID) -> tuple:
    """(grid_id, source): the rung's selected point from gates/selection.json when it exists, else
    ``default`` (SPEC 7.1 '--grid-id defaults'); (None, 'no selection') when default is None."""
    p = Path(out) / "gates" / "selection.json"
    if p.is_file():
        sel = read_json(p)
        entry = sel.get(rung) or (sel.get("selection") or {}).get(rung)
        if entry and entry.get("grid_id"):
            return entry["grid_id"], "selection.json"
    if default is None:
        return None, "no selection"
    return default, "default"


# --------------------------------------------------------------------------- feature files (3.1.5)

def features_path(out: Path, rung: str, grid_id: str, normalized: bool) -> Path:
    return Path(out) / "features" / rung / f"{grid_id}_{'norm' if normalized else 'raw'}.npz"


def build_features(out: Path, cells_csv: Path | None, rung: str, W, H, normalized: bool,
                   head_drop=None, *, wapf_norm: str = WAPF_NORM_DEFAULT, cells=None) -> Path:
    """Write ``features/<rung>/<grid_id>_{raw|norm}.npz`` (SPEC 3.1.5): ``X`` [n_rows, d],
    ``feature_names``, per-row ``cell_id``, ``kernel``, ``archetype`` (the PREDICTED archetype from
    cells.csv, never G-K0's measured one: SPEC_review_al_farabi.md item 2.3), ``campaign``,
    ``role``, ``rep``, ``win_start``, ``n_series_cell``; scalars ``W``, ``H``, ``grid_id``,
    ``normalized``, ``head_drop_json``, ``n_windows_dropped``. Rows: cells in cells.csv order,
    windows chronological. Idle cells (``role == "idle"``) are written with ``archetype = "IDLE"``
    and ``kernel = "idle"`` whatever ``cells.csv`` carries for them (``control`` and the
    label-derived name, which ``cells.csv`` and the sidecars keep unchanged): SPEC 3.1.5's own
    text (CERT 6.11; SPEC_epoch2 B24). Their head drop is the ``idle`` row of ``head_drop.csv``
    (B9). Every cell with status ok is written (admissibility is applied by the reader, so the
    artifact does not depend on a gate's verdict). W = None means the whole-cell point."""
    out = Path(out)
    if cells is None:
        cells = load_cells(cells_csv_path(out, cells_csv))
    if rung == "combined" and not normalized:
        raise ValueError("the raw variant of combined is not written (SPEC 3.1.5)")
    gid = WHOLE_GRID_ID if W is None else schema.grid_id(W, H)
    names = feature_names(rung, normalized)
    Xs, meta = [], {k: [] for k in ("cell_id", "kernel", "archetype", "campaign", "role", "rep",
                                     "win_start", "n_series_cell")}
    n_dropped = 0
    for c in cells:
        ex = load_extract_cached(out, c["cell_id"])
        idle = c.get("role") == "idle"
        hd = head_drop_for(head_drop, c["kernel"], c.get("role"))
        S = rung_series(ex, rung, normalized=normalized, head_drop=hd, wapf_norm=wapf_norm)
        ns = S.shape[0]
        w, h = resolve_wh(W, H, ns)
        if ns == 0:
            continue
        F, starts, nd = window_features(S, rung, w, h)
        n_dropped += nd
        Xs.append(F)
        k = len(F)
        meta["cell_id"] += [c["cell_id"]] * k
        meta["kernel"] += ["idle" if idle else c["kernel"]] * k
        meta["archetype"] += ["IDLE" if idle else c["archetype_predicted"]] * k
        meta["campaign"] += [c["campaign"]] * k
        meta["role"] += [c["role"]] * k
        meta["rep"] += [int(c["rep"])] * k
        meta["win_start"] += starts.tolist()
        meta["n_series_cell"] += [ns] * k
    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, len(names)))
    p = features_path(out, rung, gid, normalized)
    p.parent.mkdir(parents=True, exist_ok=True)
    hd_json = json.dumps(head_drop if isinstance(head_drop, dict) else {"all": head_drop or 0}, sort_keys=True)
    np.savez(p, X=X, feature_names=np.array(names), cell_id=np.array(meta["cell_id"]),
             kernel=np.array(meta["kernel"]), archetype=np.array(meta["archetype"]),
             campaign=np.array(meta["campaign"]), role=np.array(meta["role"]),
             rep=np.array(meta["rep"], dtype=np.int64), win_start=np.array(meta["win_start"], dtype=np.int64),
             n_series_cell=np.array(meta["n_series_cell"], dtype=np.int64),
             W=np.array(-1 if W is None else W), H=np.array(-1 if H is None else H), grid_id=np.array(gid),
             normalized=np.array(bool(normalized)), head_drop_json=np.array(hd_json),
             n_windows_dropped=np.array(n_dropped), wapf_norm=np.array(wapf_norm))
    return p


def load_features(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    res = {k: d[k] for k in d.files}
    for k in ("cell_id", "kernel", "archetype", "campaign", "role", "feature_names"):
        res[k] = res[k].astype(str)
    res["grid_id"] = str(res["grid_id"])
    res["normalized"] = bool(res["normalized"])
    return res


def build_all_grid(out: Path, cells_csv: Path | None, rung: str, head_drop=None, *,
                   variants=("raw", "norm"), wapf_norm: str = WAPF_NORM_DEFAULT) -> list[Path]:
    """Every grid point's feature file (SPEC_review_al_farabi.md item 2.2: selection is a choice
    among artifacts already on disk). ``combined`` has no raw variant."""
    out = Path(out)
    cells = load_cells(cells_csv_path(out, cells_csv))
    paths = []
    for gid, W, H in grid_points_ids():
        for var in variants:
            norm = var == "norm"
            if rung == "combined" and not norm:
                continue
            paths.append(build_features(out, None, rung, W, H, norm, head_drop, wapf_norm=wapf_norm, cells=cells))
    return paths


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="series.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("head-drop-template")
    a.add_argument("--out", required=True)
    a.add_argument("--force", action="store_true", help="overwrite an existing author input (SPEC_epoch2 B20; default: kept)")
    b = sub.add_parser("features")
    b.add_argument("--out", required=True)
    b.add_argument("--cells-csv", default=None)
    b.add_argument("--rung", required=True, choices=RUNGS)
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--grid-id", default=None)
    g.add_argument("--all-grid", action="store_true")
    v = b.add_mutually_exclusive_group()
    v.add_argument("--raw", action="store_true")
    v.add_argument("--norm", action="store_true")
    v.add_argument("--both", action="store_true")
    b.add_argument("--wapf-norm", default=WAPF_NORM_DEFAULT, choices=("median_K", "median_self"))
    b.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.cmd == "head-drop-template":
        p = out / "inputs" / "head_drop.csv"
        if p.is_file() and not args.force:                     # SPEC_epoch2 B20 (CERT 6.12, 7.7): the author's input is never overwritten
            print(f"kept: author input exists: {p}")
            return 0
        print(write_head_drop_template(p))
        return 0
    cells_csv = cells_csv_path(out, args.cells_csv)
    if not cells_csv.is_file():
        print(f"missing input: {cells_csv}", file=sys.stderr)
        return 2
    hd = load_head_drop(out / "inputs" / "head_drop.csv")
    variants = ("raw", "norm") if (args.both or not (args.raw or args.norm)) else (("raw",) if args.raw else ("norm",))
    try:
        if args.all_grid:
            for p in build_all_grid(out, cells_csv, args.rung, hd, variants=variants, wapf_norm=args.wapf_norm):
                print(p)
        else:
            W, H = parse_grid_id(args.grid_id)
            for var in variants:
                if args.rung == "combined" and var == "raw":
                    continue
                print(build_features(out, cells_csv, args.rung, W, H, var == "norm", hd, wapf_norm=args.wapf_norm))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
