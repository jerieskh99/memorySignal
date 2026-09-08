#!/usr/bin/env python3
"""stages.py -- one pure function per module kind.

Signals, as plain dicts so every stage is separable and testable on its own:

  Field     {"seq": int32[], "page_index": int32[], "cols": {name: float32[]} | None,
             "z": complex64[] | None, "channels": [...], "n_pairs", "n_pages", "block": int32[] | None}
            one row per changed page per pair; complex fields carry z instead of cols
  Series    {"values": float32[T] | complex64[T] | float32[T, k], "channels": [...],
             "complex": bool, "block": int | None, "n_pages": int}
  Tiles     {"X": (n, W) or (n, W, k), "keys": [(block, t_index, seq_start)], "w", "h",
             "taper", "channels", "complex", "series_mean", "series_std"}
  Features  {"names": [...], "rows": float32[n, f], "keys": [...]}
  Reference {"kind": "plv", "baseline_plv": float[], "recording": id}

Where the project already has the code, it is reused rather than rewritten:
  simple statistics  plan08_b1/b1_features.py features()
  cepstrum           coherence_temp_spec_stability/cepstrum_stability.py CepstrumStability
  PLV                coherence_temp_spec_stability/plv_calcolator.py PLVStability
  CUSUM              plan04_cusum.py detect_boundaries_cusum / stationarity_score

Not implemented in this pass, each refusing with a message rather than pretending:
wavelet (pywt), scattering (kymatio/torch), MSC, full-page-resolution tiles, block hop
smaller than block width, the benign-envelope baseline.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent.parent
REPO = QEMU_DIR.parent.parent
for p in (str(QEMU_DIR), str(QEMU_DIR / "plan08_b1"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from b1_features import features as b1_features                          # noqa: E402  (stdlib module)
from coherence_temp_spec_stability.cepstrum_stability import CepstrumStability  # noqa: E402
from coherence_temp_spec_stability.plv_calcolator import PLVStability           # noqa: E402
import plan04_cusum                                                           # noqa: E402


class NotImplementedStage(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# field tier
# ---------------------------------------------------------------------------

def field_from_store(store: dict, channels: list[str]) -> dict:
    """Field over the L1 store, keeping only the requested columns."""
    missing = [c for c in channels if c not in store]
    if missing:
        raise KeyError(f"store lacks columns {missing}")
    return {"seq": store["seq"], "page_index": store["page_index"], "cols": {c: store[c] for c in channels},
            "z": None, "channels": list(channels), "n_pairs": int(store["n_pairs"]), "n_pages": int(store["n_pages"]), "block": None}


def single(field: dict) -> dict:
    if len(field["channels"]) != 1:
        raise ValueError(f"Single needs exactly one channel; got {field['channels']}")
    return field


def vectorize(field: dict) -> dict:
    return field


def complex_field(mag: dict, dirn: dict, phase: str) -> dict:
    """z = magnitude * exp(i * angle(direction)); rows must be the same changed pages."""
    if len(mag["channels"]) != 1 or len(dirn["channels"]) != 1:
        raise ValueError("Complex needs one magnitude and one direction channel")
    if mag["seq"].shape != dirn["seq"].shape or not (np.array_equal(mag["seq"], dirn["seq"]) and np.array_equal(mag["page_index"], dirn["page_index"])):
        raise ValueError("magnitude and direction rows differ; both must come from the same differ pass")
    m = mag["cols"][mag["channels"][0]].astype(np.float64)
    d = dirn["cols"][dirn["channels"][0]].astype(np.float64)
    if phase == "2pi":
        ang = 2.0 * np.pi * d
    elif phase == "pi":
        ang = np.pi * d
    elif phase == "arccos":
        ang = np.arccos(np.clip(1.0 - d, -1.0, 1.0))     # cosine distance -> the geometric angle
    else:
        raise ValueError(f"unknown phase convention {phase!r}")
    z = (m * np.exp(1j * ang)).astype(np.complex64)
    return {"seq": mag["seq"], "page_index": mag["page_index"], "cols": None, "z": z,
            "channels": [mag["channels"][0], dirn["channels"][0]], "n_pairs": mag["n_pairs"], "n_pages": mag["n_pages"], "block": None}


def block(field: dict, wp: int, hp: int) -> dict:
    if hp != wp:
        raise NotImplementedStage("block hop different from block width (overlapping blocks) is not implemented")
    if wp < 1:
        raise ValueError("block width must be at least one page")
    out = dict(field)
    out["block"] = (field["page_index"] // wp).astype(np.int32)
    out["block_w"] = wp
    return out


def collapse(field: dict, unchanged: str = "zero", reduce: str = "mean") -> dict | list[dict]:
    """Reduce the page axis per pair. Returns one Series, or one per block for a blocked field.

    reduce = "mean"              the mean of the channel values over the pages (unchanged pages
                                 count as zero, or are excluded)
    reduce = "changed_fraction"  the fraction of pages that changed at all: K/N, which is APF
                                 (plan02_apf_helper) when N is the page count
    """
    T = field["n_pairs"]
    n_pages = field["n_pages"]
    seq = field["seq"]
    blocks = field.get("block")

    def reduce_rows(mask, denom_pages):
        if reduce == "changed_fraction":
            cnt = np.bincount(seq[mask] - 1, minlength=T).astype(np.float32)
            return cnt / float(denom_pages)
        if reduce != "mean":
            raise ValueError(f"unknown reduce {reduce!r}")
        if field["z"] is not None:
            vals = field["z"]
            out = np.zeros(T, dtype=np.complex64)
        else:
            k = len(field["channels"])
            vals = np.stack([field["cols"][c] for c in field["channels"]], axis=1) if k > 1 else field["cols"][field["channels"][0]]
            out = np.zeros((T, k) if k > 1 else T, dtype=np.float32)
        s = seq[mask] - 1
        v = vals[mask]
        sums = np.zeros_like(out)
        np.add.at(sums, s, v)
        if unchanged == "zero":
            out = sums / float(denom_pages)
        elif unchanged == "excluded":
            cnt = np.bincount(s, minlength=T).astype(np.float64)
            cnt[cnt == 0] = np.nan
            out = (sums.T / cnt).T if sums.ndim == 2 else sums / cnt
            out = np.nan_to_num(out, nan=0.0)
        else:
            raise ValueError(f"unknown unchanged mode {unchanged!r}")
        return out.astype(np.complex64 if field["z"] is not None else np.float32)

    base = {"channels": field["channels"] if reduce == "mean" else ["changed_fraction"],
            "complex": field["z"] is not None and reduce == "mean", "n_pages": n_pages}
    if blocks is None:
        return dict(base, values=reduce_rows(np.ones(len(seq), dtype=bool), n_pages), block=None)
    bw = field["block_w"]
    n_blocks = max(1, math.ceil(n_pages / bw))
    out = []
    for b in range(n_blocks):
        pages_in_block = min(bw, n_pages - b * bw)
        out.append(dict(base, values=reduce_rows(blocks == b, pages_in_block), block=b))
    return out


# ---------------------------------------------------------------------------
# divide tier
# ---------------------------------------------------------------------------

def _taper(w: int, kind: str) -> np.ndarray:
    if kind == "rectangular":
        return np.ones(w)
    if kind == "hann":
        return np.hanning(w)
    if kind == "hamming":
        return np.hamming(w)
    if kind == "blackman":
        return np.blackman(w)
    raise ValueError(f"unknown taper {kind!r}")


def window(series: dict, w: int, h: int, edge: str = "drop", taper: str = "rectangular") -> dict:
    v = series["values"]
    T = v.shape[0]
    if w < 1 or h < 1:
        raise ValueError("W and H must be at least 1")
    if edge == "drop":
        starts = list(range(0, T - w + 1, h))
    else:
        starts = list(range(0, T, h))
        last_end = starts[-1] + w if starts else 0
        pad = max(0, last_end - T)
        if pad:
            mode = {"zero": "constant", "reflect": "reflect", "replicate": "edge"}[edge]
            widths = [(0, pad)] + [(0, 0)] * (v.ndim - 1)
            v = np.pad(v, widths, mode=mode)
    X = np.stack([v[s:s + w] for s in starts]) if starts else np.zeros((0, w) + v.shape[1:], dtype=v.dtype)
    keys = [(series.get("block"), i, s + 1) for i, s in enumerate(starts)]
    real = np.abs(series["values"]) if series["complex"] else series["values"]
    return {"X": X, "keys": keys, "w": w, "h": h, "taper": taper, "channels": series["channels"], "complex": series["complex"],
            "series_mean": float(np.mean(real)) if T else 0.0, "series_std": float(np.std(real)) if T else 0.0}


# ---------------------------------------------------------------------------
# lens tier: each returns Features with one row per tile
# ---------------------------------------------------------------------------

def _per_channel(tiles: dict):
    """Yield (suffix, X2d) so lenses can run per channel on vector tiles."""
    X = tiles["X"]
    if X.ndim == 2:
        yield "", X
    else:
        for j, c in enumerate(tiles["channels"]):
            yield ":" + c, X[:, :, j]


def _real(X: np.ndarray) -> np.ndarray:
    return np.abs(X) if np.iscomplexobj(X) else X


def stats(tiles: dict, feats: list[str]) -> dict:
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        R = _real(X)
        rows = []
        for t in range(R.shape[0]):
            f = b1_features([float(x) for x in R[t]])
            f["duty_gt05"] = float(np.mean(R[t] > 0.05))
            rows.append([f[k] for k in feats])
        names += [k + suffix for k in feats]
        cols.append(np.asarray(rows, dtype=np.float32).reshape(R.shape[0], len(feats)))
    return {"names": names, "rows": np.concatenate(cols, axis=1) if cols else np.zeros((0, 0), np.float32), "keys": tiles["keys"]}


def _tau(x: np.ndarray) -> float:
    """Decorrelation time: first zero crossing of the autocorrelation, in frames."""
    x = x - x.mean()
    if np.allclose(x, 0):
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    below = np.where(ac <= 0)[0]
    return float(below[0]) if len(below) else float(len(x))


def _entropy(x: np.ndarray, bins: int = 16) -> float:
    h, _ = np.histogram(x, bins=bins)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def _ceps_peak(x: np.ndarray) -> tuple[float, float]:
    cs = CepstrumStability()
    ceps = cs.compute_cepstrum(np.asarray(x, dtype=float).reshape(-1, 1))
    if not np.all(np.isfinite(ceps)):
        return float("nan"), float("nan")
    min_quef = max(1, len(x) // 8)
    Q = ceps.shape[0]
    if min_quef >= Q:
        return float("nan"), float("nan")
    idx = int(cs.compute_cepstral_peak(ceps, min_quef_idx=min_quef)[0])
    mag = np.abs(ceps[:, 0])
    tail = mag[min_quef:]
    med = float(np.median(tail)) if tail.size else 0.0
    pk = float(mag[idx])
    snr = 10.0 * math.log10(pk / med) if med > 0 and pk > 0 else float("nan")
    return float(idx), snr


def deep(tiles: dict, feats: list[str]) -> dict:
    gm, gs = tiles["series_mean"], tiles["series_std"]
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        R = _real(X).astype(np.float64)
        rows = []
        for t in range(R.shape[0]):
            x = R[t]
            f = {}
            f["stat_pass_frac"] = 1.0 if gs == 0 else float(abs(x.mean() - gm) / gs < 1.0)
            if "n_boundaries" in feats:
                f["n_boundaries"] = float(len(plan04_cusum.detect_boundaries_cusum(x)))
            if "cepstral_peak_idx" in feats or "ceps_peak_snr_db" in feats:
                f["cepstral_peak_idx"], f["ceps_peak_snr_db"] = _ceps_peak(x)
            f["tau"] = _tau(x)
            sd = x.std()
            f["skew"] = float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
            f["kurtosis"] = float(((x - x.mean()) ** 4).mean() / sd ** 4 - 3.0) if sd > 0 else 0.0
            f["entropy"] = _entropy(x)
            f["cv_workingset"] = float(sd / x.mean()) if x.mean() else 0.0
            f["f1_phase"] = float("nan")          # structurally unavailable (D-86); flagged in the registry
            f["coverage_ratio"] = float("nan")    # a per-family constant, not a tile property; flagged
            rows.append([f[k] for k in feats])
        names += [k + suffix for k in feats]
        cols.append(np.asarray(rows, dtype=np.float32).reshape(R.shape[0], len(feats)))
    return {"names": names, "rows": np.concatenate(cols, axis=1) if cols else np.zeros((0, 0), np.float32), "keys": tiles["keys"]}


def fft(tiles: dict, out: str = "bands", n_bands: int = 4) -> dict:
    tp = _taper(tiles["w"], tiles["taper"])
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        Y = np.fft.rfft(_real(X) * tp, axis=1)
        P = np.abs(Y) ** 2
        if out == "spectrum":
            names += [f"fft_bin{i}{suffix}" for i in range(P.shape[1])]
            cols.append(P)
        elif out == "peak":
            idx = np.argmax(P[:, 1:], axis=1) + 1 if P.shape[1] > 1 else np.zeros(P.shape[0])
            names += [f"fft_peak_bin{suffix}", f"fft_peak_power{suffix}"]
            cols.append(np.stack([idx, P[np.arange(P.shape[0]), idx.astype(int)]], axis=1))
        elif out == "bands":
            edges = np.linspace(0, P.shape[1], n_bands + 1).astype(int)
            bands = np.stack([P[:, edges[i]:edges[i + 1]].sum(axis=1) for i in range(n_bands)], axis=1)
            names += [f"fft_band{i}{suffix}" for i in range(n_bands)]
            cols.append(bands)
        else:
            raise ValueError(f"unknown fft output {out!r}")
    return {"names": names, "rows": np.concatenate(cols, axis=1).astype(np.float32), "keys": tiles["keys"]}


def cepstrum(tiles: dict) -> dict:
    tp = _taper(tiles["w"], tiles["taper"])
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        R = _real(X) * tp
        rows = [list(_ceps_peak(R[t])) for t in range(R.shape[0])]
        names += ["cepstral_peak_idx" + suffix, "ceps_peak_snr_db" + suffix]
        cols.append(np.asarray(rows, dtype=np.float32).reshape(R.shape[0], 2))
    return {"names": names, "rows": np.concatenate(cols, axis=1), "keys": tiles["keys"]}


def cusum(tiles: dict, k: float = 2.0, h: float = 4.0) -> dict:
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        R = _real(X).astype(np.float64)
        rows = [[float(len(plan04_cusum.detect_boundaries_cusum(R[t], k=k, h=h))), plan04_cusum.stationarity_score(list(R[t]))] for t in range(R.shape[0])]
        names += ["n_boundaries" + suffix, "stationarity_score" + suffix]
        cols.append(np.asarray(rows, dtype=np.float32).reshape(R.shape[0], 2))
    return {"names": names, "rows": np.concatenate(cols, axis=1), "keys": tiles["keys"]}


def wavelet(tiles: dict, fam: str, levels: int) -> dict:
    try:
        import pywt  # type: ignore
    except ImportError as e:
        raise NotImplementedStage("wavelet needs pywt in the analysis environment") from e
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        R = _real(X)
        rows = []
        for t in range(R.shape[0]):
            coeffs = pywt.wavedec(R[t], fam, level=levels)
            rows.append([float(np.sum(np.square(c))) for c in coeffs])
        names += [f"wav_l{i}_energy{suffix}" for i in range(levels + 1)]
        cols.append(np.asarray(rows, dtype=np.float32))
    return {"names": names, "rows": np.concatenate(cols, axis=1), "keys": tiles["keys"]}


def scattering(tiles: dict, J: int, Q: int) -> dict:
    raise NotImplementedStage("scattering needs kymatio and torch; not in the analysis environment")


def msc(tiles: dict, iw: int, ih: int) -> dict:
    raise NotImplementedStage("MSC is not implemented in this pass")


# ---------------------------------------------------------------------------
# reference tier
# ---------------------------------------------------------------------------

def baseline(tiles: dict, mode: str, recording: str) -> dict:
    if mode != "cell":
        raise NotImplementedStage("the benign-envelope baseline is not implemented in this pass")
    if not tiles["complex"]:
        raise ValueError("PLV baseline needs complex tiles")
    X = tiles["X"]
    flat = X.reshape(-1, 1) if X.ndim == 2 else X.reshape(-1, X.shape[-1])
    plv = PLVStability()
    return {"kind": "plv", "baseline_plv": plv.fit_baseline(flat), "recording": recording, "n_samples": int(flat.shape[0])}


def plv(tiles: dict, ref: dict, drop: float, normal: float) -> dict:
    if not tiles["complex"]:
        raise ValueError("PLV reads the phase of a complex signal; upstream tiles are real")
    if ref.get("kind") != "plv":
        raise ValueError("PLV needs a PLV baseline reference")
    p = PLVStability(baseline_plv=np.asarray(ref["baseline_plv"]))
    rows, names = [], None
    for t in range(tiles["X"].shape[0]):
        X = tiles["X"][t]
        Xt = X.reshape(-1, 1) if X.ndim == 1 else X
        r = p.evaluate_run(Xt, drop_threshold=drop, normal_threshold=normal)
        keys = sorted(k for k, v in r.items() if isinstance(v, (int, float)))
        names = names or ["plv_" + k for k in keys]
        rows.append([float(r[k]) for k in keys])
    return {"names": names or [], "rows": np.asarray(rows, dtype=np.float32).reshape(len(rows), len(names or [])), "keys": tiles["keys"]}


# ---------------------------------------------------------------------------
# output tier
# ---------------------------------------------------------------------------

def concat(blocks: list[dict]) -> dict:
    if not blocks:
        return {"names": [], "rows": np.zeros((0, 0), np.float32), "keys": []}
    keys = blocks[0]["keys"]
    for b in blocks[1:]:
        if b["keys"] != keys:
            raise ValueError("concat: feature blocks do not share tile keys")
    return {"names": [n for b in blocks for n in b["names"]], "rows": np.concatenate([b["rows"] for b in blocks], axis=1), "keys": keys}
