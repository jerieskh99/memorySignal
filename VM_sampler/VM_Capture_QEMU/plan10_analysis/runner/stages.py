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


def n_blocks(n_pages: int, wp: int, hp: int) -> int:
    """How many whole blocks of `wp` pages, stepped by `hp`, fit in `n_pages`.

    Whole blocks only, the same convention as Window's edge="drop" on the time axis: a
    trailing run of pages too short to fill a block is dropped rather than averaged over a
    smaller denominator. This is the single definition; scheme.py's tile estimate and the
    console's readout compute it the same way, so what the page counts is what the runner
    produces.
    """
    if wp < 1 or hp < 1:
        raise ValueError("block width and hop must be at least one page")
    return 0 if n_pages < wp else (n_pages - wp) // hp + 1


def block(field: dict, wp: int, hp: int) -> dict:
    """Cut the address axis into blocks of `wp` pages stepped by `hp`.

    With hp == wp the blocks tile the axis and every changed page belongs to exactly one.
    With hp < wp they overlap, a page belongs to several, and its row is REPLICATED once
    per block it falls in: the field grows by about wp/hp. That replication is what lets a
    later Collapse reduce each block independently, and it is why a small hop is expensive.

    With hp > wp the blocks are spaced apart and pages in the gaps belong to none; those
    rows are dropped, which is a legitimate sampling of the address axis but silently
    discards data, so scheme.py warns.
    """
    nb = n_blocks(field["n_pages"], wp, hp)
    if nb < 1:
        raise ValueError(f"a block of {wp} pages does not fit in {field['n_pages']} pages")
    p = field["page_index"].astype(np.int64)
    # blocks containing page p: those b with b*hp <= p < b*hp + wp
    b_lo = np.maximum(0, (p - wp) // hp + 1)
    b_hi = np.minimum(p // hp, nb - 1)
    counts = np.maximum(0, b_hi - b_lo + 1)
    total = int(counts.sum())
    idx = np.repeat(np.arange(p.shape[0], dtype=np.int64), counts)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])) if counts.size else np.zeros(0, np.int64)
    ordinal = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    blk = (np.repeat(b_lo, counts) + ordinal).astype(np.int32)

    out = dict(field)
    out["seq"] = field["seq"][idx]
    out["page_index"] = field["page_index"][idx]
    if field.get("z") is not None:
        out["z"] = field["z"][idx]
    if field.get("cols"):
        out["cols"] = {c: v[idx] for c, v in field["cols"].items()}
    out["block"] = blk
    out["block_w"] = int(wp)
    out["block_h"] = int(hp)
    out["n_blocks"] = nb
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
    bw, bh = field["block_w"], field.get("block_h", field["block_w"])
    nb = field.get("n_blocks") or n_blocks(n_pages, bw, bh)
    out = []
    for b in range(nb):
        # whole blocks only, so every block spans exactly bw pages; that is the denominator
        out.append(dict(base, values=reduce_rows(blocks == b, bw), block=b))
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


def dense_page_matrix(field: dict, page_mode: str = "active", max_bytes: int = 256 * 1024 * 1024) -> tuple:
    """The sparse field as the dense (T, P) matrix the methodology calls the page-by-time image.

    page_mode="active" keeps only the pages that change at least once in this recording,
    which is the whole point of the differ's sparse output: at a few percent activity the
    dense form over every page is mostly zeros and tens of times larger. The page indices
    kept are returned, so a row still names the page it came from.
    page_mode="all" keeps every page of the dump, which is what a fixed-address reading
    needs and what the memory guard usually refuses.

    A single channel or a complex field only: several real channels would make the tile
    four-dimensional, which no lens here reads.
    """
    if page_mode not in ("active", "all"):
        raise ValueError(f"unknown page mode {page_mode!r}")
    T, n_pages = field["n_pairs"], field["n_pages"]
    if field.get("z") is None and len(field["channels"]) != 1:
        raise ValueError(f"a page-resolution tile carries one channel or a complex field; got {len(field['channels'])} channels. "
                         "Use Single, or Collapse the page axis")
    pages = (np.unique(field["page_index"]) if page_mode == "active"
             else np.arange(n_pages, dtype=field["page_index"].dtype))
    dtype = np.complex64 if field.get("z") is not None else np.float32
    need = int(T) * int(pages.shape[0]) * np.dtype(dtype).itemsize
    if need > max_bytes:
        raise ValueError(
            f"a dense {T} x {pages.shape[0]} page matrix needs {need / 1e6:.0f} MB, over the {max_bytes / 1e6:.0f} MB budget. "
            f"Use page_mode=active ({np.unique(field['page_index']).shape[0]} pages change here, of {n_pages}), "
            "raise the budget, or Block/Collapse the page axis")
    col = np.searchsorted(pages, field["page_index"])
    keep = (col < pages.shape[0]) & (pages[np.minimum(col, pages.shape[0] - 1)] == field["page_index"])
    M = np.zeros((T, pages.shape[0]), dtype=dtype)
    vals = field["z"] if field.get("z") is not None else field["cols"][field["channels"][0]]
    M[field["seq"][keep] - 1, col[keep]] = vals[keep]
    return M, pages


def run_lens(tiles: dict, fn):
    """Apply a lens, reducing a page axis by the median across pages.

    A page-resolution tile is (n_tiles, W, P). Every lens here is written for one series per
    tile, so it runs per page and the results are reduced by the median across pages: the
    convention StabilityValidator already uses, which reports msc_peak_snr_db_median and
    cepstral_peak_idx_median beside their per-page arrays. PLV and Baseline are not routed
    through here, because their code takes [T, N] and aggregates the page axis itself.
    """
    if not tiles.get("page_axis"):
        return fn(tiles)
    X = tiles["X"]
    outs = []
    for j in range(X.shape[2]):
        sub = dict(tiles)
        sub["X"] = X[:, :, j]
        sub["page_axis"] = False
        sub["channels"] = ["page"]
        outs.append(fn(sub))
    if not outs:
        return {"names": [], "rows": np.zeros((X.shape[0], 0), np.float32), "keys": tiles["keys"]}
    rows = np.median(np.stack([o["rows"] for o in outs], axis=2), axis=2)
    return {"names": [n + "_median" for n in outs[0]["names"]], "rows": rows.astype(np.float32), "keys": tiles["keys"]}


def window(series: dict, w: int, h: int, edge: str = "drop", taper: str = "rectangular",
           page_mode: str = "active", max_bytes: int = 256 * 1024 * 1024) -> dict:
    """Slide a window over snapshots. A field with a page axis becomes page-resolution tiles."""
    if "page_index" in series:
        M, pages = dense_page_matrix(series, page_mode, max_bytes)
        ser = {"values": M, "channels": series["channels"], "complex": series.get("z") is not None,
               "block": None, "n_pages": series["n_pages"]}
        t = window(ser, w, h, edge, taper)
        need = t["X"].size * t["X"].dtype.itemsize
        if need > max_bytes:
            raise ValueError(f"the tiles need {need / 1e6:.0f} MB, over the {max_bytes / 1e6:.0f} MB budget; "
                             "widen the hop, narrow the window, or reduce the page axis")
        t["page_axis"] = True
        t["pages"] = pages
        return t
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
            "series_mean": float(np.mean(real)) if T else 0.0, "series_std": float(np.std(real)) if T else 0.0,
            "page_axis": False, "pages": None}


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


def fft(tiles: dict, out: str = "bands", n_bands: int = 4, detrend: str = "mean") -> dict:
    """Power spectrum of each tile.

    `detrend="mean"` (the default) subtracts each tile's own mean before the transform.
    Without it the DC term dominates: a taper spreads DC into the low bins, so the peak
    lands on bin 1 whatever the signal does. Measured on a period-8 series at W=32: the
    true peak is bin 4, and the untreated path reports bin 1 under a Hann taper (bin 4
    under a rectangular one). The tile's mean is not lost, it is what `stats.mean` reports;
    these features describe the fluctuation around it. `detrend="none"` keeps the old
    behaviour and is recorded in the scheme like any other choice.
    """
    tp = _taper(tiles["w"], tiles["taper"])
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        R = _real(X).astype(np.float64)
        if detrend == "mean":
            R = R - R.mean(axis=1, keepdims=True)
        elif detrend != "none":
            raise ValueError(f"unknown detrend {detrend!r}")
        Y = np.fft.rfft(R * tp, axis=1)
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
    """Quefrency peak and its SNR, on the RAW tile.

    The window's taper is deliberately not applied here. The cepstrum takes the log of the
    magnitude spectrum, so a taper's own envelope enters the log and dominates the
    quefrency: measured on a period-8 series at W=32, the tapered path peaks at quefrency
    31 while the raw path peaks at 8, the true period. plan03_metric_kernel feeds its
    cepstrum the raw trajectory for the same reason. The mean is left in for the same
    reason (removing it moved the peak to 12).
    """
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        R = _real(X)
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


def wavelet_max_level(fam: str, w: int) -> int:
    """Levels this family can take on a window of w samples, or -1 if pywt is absent.

    The limit is filter-length aware, not log2(w): at w=32, haar allows 5 levels, db4 two,
    sym5 one and coif3 none, because each level halves the signal and the filter needs room.
    """
    try:
        import pywt  # type: ignore
    except ImportError:
        return -1
    return int(pywt.dwt_max_level(w, pywt.Wavelet(fam)))


def wavelet(tiles: dict, fam: str, levels: int, mode: str = "periodization") -> dict:
    """Per-level coefficient energy, one feature per level plus the approximation.

    `mode="periodization"` (the default) is the only extension that keeps the transform
    energy-preserving: on a 32-sample tile db4 then emits exactly 32 coefficients whose
    energy equals the signal's, where the pywt default ("symmetric") emits 45 and inflates
    the energy by padding. The mode is recorded in the scheme like any other choice.

    Refuses a level count the family cannot take at this window size, naming the limit,
    rather than letting pywt warn that every coefficient is a boundary effect.
    """
    try:
        import pywt  # type: ignore
    except ImportError as e:
        raise NotImplementedStage("wavelet needs pywt in the analysis environment") from e
    if fam not in pywt.wavelist(kind="discrete"):
        raise NotImplementedStage(
            f"{fam!r} is not a discrete wavelet; wavedec needs one of pywt.wavelist(kind='discrete'). "
            "Continuous families (morl, mexh, gaus*, cmor, ...) would need a CWT lens, which is not built")
    lim = wavelet_max_level(fam, tiles["w"])
    if levels < 1 or levels > lim:
        raise ValueError(f"{fam} on a {tiles['w']}-sample window allows 1 to {lim} level(s); got {levels}")
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        R = _real(X)
        rows = []
        for t in range(R.shape[0]):
            coeffs = pywt.wavedec(R[t], fam, level=levels, mode=mode)
            rows.append([float(np.sum(np.square(c))) for c in coeffs])
        names += [f"wav_l{i}_energy{suffix}" for i in range(levels + 1)]
        cols.append(np.asarray(rows, dtype=np.float32))
    return {"names": names, "rows": np.concatenate(cols, axis=1), "keys": tiles["keys"]}


_SCAT_CACHE: dict = {}


def scattering_max_J(w: int, Q: int) -> int:
    """Largest J whose filters fit in a w-sample window without border effects, or -1 without kymatio.

    kymatio decides this numerically, not by a formula: it warns "Signal support is too
    small to avoid border effects" when a filter's L1 tail does not decay inside the
    window. So this measures it, by building the transform and watching for that warning,
    rather than guessing a rule. Returns 0 when even J=1 borders.
    """
    try:
        import warnings

        from kymatio.numpy import Scattering1D  # type: ignore
    except ImportError:
        return -1
    key = ("maxJ", int(w), int(Q))
    if key in _SCAT_CACHE:
        return _SCAT_CACHE[key]
    best = 0
    for J in range(int(w).bit_length(), 0, -1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                Scattering1D(J=J, shape=(int(w),), Q=int(Q))
            best = J
            break
        except Exception:
            continue
    _SCAT_CACHE[key] = best
    return best


def _scat_op(w: int, J: int, Q: int):
    key = ("op", int(w), int(J), int(Q))
    if key not in _SCAT_CACHE:
        from kymatio.numpy import Scattering1D  # type: ignore
        _SCAT_CACHE[key] = Scattering1D(J=int(J), shape=(int(w),), Q=int(Q))
    return _SCAT_CACHE[key]


def scattering(tiles: dict, J: int, Q: int) -> dict:
    """Time-averaged wavelet scattering coefficients, one feature per path.

    Scattering is in this pipeline for the reason Entry 1 gives: where a change happened is
    noise, what kind it is is signal. The transform is translation invariant by
    construction. Measured on a 64-sample window, moving an 8-sample burst from position 8
    to position 40 changes the coefficients by 0.008 relative, against 0.058 for FFT band
    energies, while flat and periodic signals still differ by 0.157.

    It is not the only shift-tolerant lens here, and the docstring said so wrongly at first:
    per-level wavelet energy scored 0.007 on the same test, because summing squared
    coefficients over a level discards position too. What scattering adds is second-order
    paths, which keep the amplitude modulation that a single energy per level averages away.

    kymatio's numpy frontend is used, so this needs neither torch nor a GPU (the
    first-generation code in VMsig_featureExctraction reached for the torch backend).
    S(x) returns (n_paths, w / 2^J); each path is averaged over its time bins, which is the
    standard scattering feature vector. Refuses a J the window cannot support, naming the
    measured limit, rather than returning border artefacts.
    """
    try:
        from kymatio.numpy import Scattering1D  # noqa: F401
    except ImportError as e:
        raise NotImplementedStage(
            "scattering needs kymatio in the analysis environment (pip install kymatio; "
            "the numpy frontend needs no torch)") from e
    w = int(tiles["w"])
    lim = scattering_max_J(w, Q)
    if J < 1 or J > lim:
        raise ValueError(
            f"scattering J={J} on a {w}-sample window at Q={Q} allows 1 to {lim}; "
            "beyond that kymatio's filters do not fit and every coefficient is a border effect")
    op = _scat_op(w, J, Q)
    order = np.asarray(op.meta()["order"]).astype(int)
    names, cols = [], []
    for suffix, X in _per_channel(tiles):
        R = _real(X).astype(np.float32)
        rows = np.stack([op(R[t]).mean(axis=1) for t in range(R.shape[0])]) if R.shape[0] else np.zeros((0, len(order)), np.float32)
        seen: dict[int, int] = {}
        for o in order:
            seen[o] = seen.get(o, 0) + 1
            names.append(f"scat_o{o}_{seen[o] - 1}{suffix}")
        cols.append(rows.astype(np.float32))
    return {"names": names, "rows": np.concatenate(cols, axis=1) if cols else np.zeros((0, 0), np.float32), "keys": tiles["keys"]}


def msc_min_window(iw: int, ih: int, method: str = "welch") -> int:
    """Shortest tile each method needs.

    welch  three segments, so two pairs to average over: iw + 2*ih. With one pair the ratio
           is identically 1 whatever the data (see `msc`), so two is the real minimum.
    legacy two segments: iw + ih.
    """
    return int(iw) + (2 if method == "welch" else 1) * int(ih)


def msc(tiles: dict, iw: int, ih: int, method: str = "welch", detrend: str = "mean") -> dict:
    """Magnitude-squared coherence between adjacent internal windows of the same signal.

    What it measures: how much a signal's spectrum repeats itself over time. It is NOT
    coherence between two channels, and an earlier constraint in this project said it was
    and demanded two of them; one channel is enough and several are computed independently.

    Two methods, because the project's own implementation is degenerate.

    `method="welch"` (the default) accumulates the cross-spectrum and both auto-spectra
    over every adjacent segment pair and takes the ratio ONCE, at the end:
    |<Pxy>|^2 / (<Pxx><Pyy>). This is the definition; the averaging is what makes the
    quantity mean anything.

    `method="legacy_adjacent"` reproduces coherence_temp_spec_stability/
    magnitude_squared_coherence.py, which takes the ratio per pair and averages the
    ratios. That is identically 1 wherever both windows hold power, because for a single
    FFT pair |X conj(Y)|^2 and |X|^2 |Y|^2 are the same quantity: measured, two INDEPENDENT
    random windows score 1.0000000000 in every bin, and its `msc_mean` equals the fraction
    of bins holding power exactly (0.0606 for a sine, 1.0000 for white noise). It is a
    spectral-occupancy measure, not a coherence. Kept because earlier numbers were produced
    with it; flagged in the known-issue registry as `msc_single_segment`.

    `detrend="mean"` removes each segment's mean before its FFT, for the same reason the
    FFT lens does: these trajectories carry a large offset, DC then holds nearly all the
    power, and DC is trivially coherent between any two windows. Measured at W=512,
    iw=ih=64, power-weighted coherence without detrending was 1.00 / 0.97 / 0.93 for a
    stationary sine, a chirp and white noise, which says nothing; with it, 1.00 / 0.18 /
    0.17, against the 1/7 = 0.14 that independent segments should give at seven pairs.

    Four features per channel. `msc_weighted` is the one to read: coherence averaged over
    bins weighted by their power, which answers "does the energy repeat". `msc_mean` is the
    unweighted average over all bins, so it is dragged down by the empty ones and behaves
    partly as occupancy (0.03 for a sine whose weighted coherence is 1.00); it is kept
    because the legacy path reports it.
    """
    w = int(tiles["w"])
    if method not in ("welch", "legacy_adjacent"):
        raise ValueError(f"unknown MSC method {method!r}")
    need = msc_min_window(iw, ih, method)
    if w < need:
        raise ValueError(
            f"MSC ({method}) at {iw}/{ih} needs a tile of at least {need} samples; this one is {w}. "
            "Lower the internal window, or widen the Window module")
    X = tiles["X"]
    chans = tiles["channels"] if X.ndim == 3 else [None]
    eps = 1e-10

    if detrend not in ("mean", "none"):
        raise ValueError(f"unknown detrend {detrend!r}")

    def spectra(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(coherence (F, n_channels), power (F, n_channels)) for one tile."""
        starts = list(range(0, arr.shape[0] - int(iw) + 1, int(ih)))
        pxy = pxx = pyy = 0.0
        for a, b in zip(starts, starts[1:]):
            xa, xb = arr[a:a + int(iw)], arr[b:b + int(iw)]
            if detrend == "mean":
                xa, xb = xa - xa.mean(axis=0, keepdims=True), xb - xb.mean(axis=0, keepdims=True)
            Xf, Yf = np.fft.rfft(xa, axis=0), np.fft.rfft(xb, axis=0)
            pxy = pxy + Xf * np.conj(Yf)
            pxx = pxx + np.abs(Xf) ** 2
            pyy = pyy + np.abs(Yf) ** 2
        if method == "legacy_adjacent":
            from coherence_temp_spec_stability.magnitude_squared_coherence import MagnitudeSquaredCoherence
            return MagnitudeSquaredCoherence(window_size=int(iw), window_step=int(ih)).compute_msc(arr), pxx
        return np.abs(pxy) ** 2 / (pxx * pyy + eps), pxx

    rows = []
    for t in range(X.shape[0]):
        arr = _real(X[t]).astype(np.float64)
        arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        spec, power = spectra(arr)
        peak, noise = spec.max(axis=0), spec.mean(axis=0)
        snr = 10.0 * np.log10((peak + eps) / (noise + eps))
        weighted = (spec * power).sum(axis=0) / (power.sum(axis=0) + eps)
        rows.append([v for j in range(arr.shape[1])
                     for v in (float(snr[j]), float(noise[j]), float(peak[j]), float(weighted[j]))])
    names = [f"{k}{'' if c is None else ':' + c}" for c in chans
             for k in ("msc_peak_snr_db", "msc_mean", "msc_max", "msc_weighted")]
    return {"names": names,
            "rows": np.asarray(rows, dtype=np.float32).reshape(X.shape[0], len(names)),
            "keys": tiles["keys"]}


# ---------------------------------------------------------------------------
# reference tier
# ---------------------------------------------------------------------------

def baseline_envelope(blocks: list, q_lo: float = 5.0, q_hi: float = 95.0) -> dict:
    """The benign operating region: a per-feature p5-p95 band over the benign rows.

    The project's own definition, from plan05_campaign/normal_profile.py: "per-feature
    benign p5-p95 band = the normal operating region, plus a simple INTERPRETABLE deviation
    detector (count of features a cell falls outside the band)". That detector is the
    `deviation` stage below.

    `blocks` is a list of (recording_id, features). Every block must carry the same feature
    names, since the band is per feature.
    """
    if not blocks:
        raise ValueError("the benign envelope needs at least one recording's features")
    names = list(blocks[0][1]["names"])
    for rid, b in blocks:
        if list(b["names"]) != names:
            raise ValueError(f"{rid} carries different feature names; the envelope is per feature")
        if np.asarray(b["rows"]).shape[1] != len(names):
            raise ValueError(f"{rid} has {np.asarray(b['rows']).shape[1]} columns for {len(names)} feature names")
    X = np.concatenate([np.asarray(b["rows"], dtype=np.float64) for _, b in blocks], axis=0)
    if X.shape[0] < 2:
        raise ValueError(f"the benign envelope needs at least 2 rows to have a band; got {X.shape[0]}")
    allnan = np.all(np.isnan(X), axis=0)
    lo = np.where(allnan, np.nan, np.nanpercentile(np.where(np.isnan(X), np.nan, X), q_lo, axis=0))
    hi = np.where(allnan, np.nan, np.nanpercentile(np.where(np.isnan(X), np.nan, X), q_hi, axis=0))
    med = np.where(allnan, np.nan, np.nanmedian(X, axis=0))
    return {"kind": "envelope", "names": names, "lo": lo, "hi": hi, "median": med,
            "width": hi - lo, "n_rows": int(X.shape[0]),
            "recordings": [rid for rid, _ in blocks], "q": [q_lo, q_hi]}


def deviation(features: dict, ref: dict) -> dict:
    """How far each tile falls outside the benign envelope.

    `dev_n_outside` is normal_profile.py's detector verbatim: the count of features whose
    value sits outside the benign band. A feature that is NaN counts as inside, as it does
    there.

    The excess features normalise the distance past the edge so features on different scales
    compare. The scale is the band's own width; where the benign set pins a feature to one
    value the width is zero, and the fallback is |median|, then 1.0. Without that fallback a
    feature that is constant across the benign set could be violated by any margin and still
    report an excess of zero, which is how this first behaved: a recording writing ten times
    more scored 2 features outside and an excess of 0.00.
    """
    if ref.get("kind") != "envelope":
        raise ValueError(f"deviation needs an envelope reference; got {ref.get('kind')!r}")
    if list(features["names"]) != list(ref["names"]):
        raise ValueError("the features and the envelope carry different feature names")
    X = np.asarray(features["rows"], dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(ref["names"]):
        raise ValueError(f"the features have {X.shape[1] if X.ndim == 2 else '?'} columns "
                         f"for {len(ref['names'])} feature names in the envelope")
    lo, hi, width = ref["lo"], ref["hi"], ref["width"]
    excess = np.maximum(0.0, np.maximum(lo - X, X - hi))
    known = ~np.isnan(X) & ~np.isnan(lo) & ~np.isnan(hi)
    outside = (excess > 0) & known
    med = np.abs(np.asarray(ref["median"], dtype=np.float64))
    scale = np.where((width > 0) & ~np.isnan(width), width,
                     np.where((med > 0) & ~np.isnan(med), med, 1.0))
    norm = np.where(known, excess / scale, 0.0)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0)
    rows = np.stack([outside.sum(axis=1), outside.mean(axis=1), norm.max(axis=1), norm.sum(axis=1)], axis=1)
    return {"names": ["dev_n_outside", "dev_frac_outside", "dev_max_excess", "dev_total_excess"],
            "rows": rows.astype(np.float32), "keys": features["keys"]}


def baseline(tiles: dict, mode: str, recording: str) -> dict:
    if mode != "cell":
        raise ValueError(f"baseline mode {mode!r} is not fitted from tiles; the benign envelope reads features")
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
