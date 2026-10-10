#!/usr/bin/env python3
"""page_fourier.py -- the per-page Fourier test beside the run of plan12_grounding (2026-10-10; move 13 in the
console's Grounding paper tab, after the idle class check (11) and the new-block test (12)).

  python3 -m plan12_grounding.page_fourier run --run <main run out> --step 1|2 [--move 13|14|15|16] [--out <dir>]
        [--cuts declared,measured] [--only cellA,cellB] [--chunk-pages 2000] [--n-shuffles 1000] [--seed 20260930]
        [--image-cell floyd__rep00__01c1] [--check-only] [--force] [--dry-run]
  python3 -m plan12_grounding.page_fourier summary --run <main run out> [--out <dir>] [--dry-run]
  (run from VM_sampler/VM_Capture_QEMU/)

THE GRID (the author's, 2026-10-10; moves 14 to 16 added to this module with defaults that reproduce move 13 exactly,
checked byte for byte on two recordings). Two switches, every combination, each with the two steps:
  move 13  each page counts once (its spectrum normalised to sum 1)   level removed (the mean removed per segment, the f = 0 bin dropped in the between-recordings test)
  move 14  raw spectra (the plain mean of the pages' power spectra, so pages with more bits weigh more)   level kept (no mean removed, the f = 0 bin kept, in the spectra and in the between-recordings test)
  move 15  each page counts once                                        level kept
  move 16  raw spectra                                                  level removed
Everything else is the same (the data, the cuts, the identity check, the activity groups, the asymmetry, move 5's test
and its null, the figures, the outputs), so the eight variants compare directly. Move 13 writes <out>/<step>/ as before;
moves 14 to 16 write <out>/move<N>/<step>/ with the record keys move<N>_<step>. `summary` collects the eight variants
(both steps, both cuts) into <out>/grid_summary.csv, .svg and .json: the gap, its null p95 and p, and the mean
asymmetry; a variant not yet run reads "missing".

WHY. The SPL paper (spl_paper/main_v2.tex, Section IV, eq. DeltaEntry and DeltaVector) defines the signal as a complex
matrix Delta in C^{N x (T-1)}: page n at pair t gives Delta(t,n) = h * e^{j phi}, zero when the page did not change; each
row is one page's signal over time, and the rows together are one big signal. Moves 3 to 12 summed the page axis away
first (N_t, H_t, A_t per pair). This move analyses the rows. The author's phase is the DOUBLED angle, phi = 2 theta,
theta = arccos(clip(1 - d, 0, 1)) as extract.py computes it, so "kept its pattern" (theta = 0) and "nothing in common"
(theta = 90 degrees) point opposite ways (0 and 180 degrees). Step 1 uses phi = 2 theta (the headline); step 2 the same
with phi = theta (the control), so the author sees what the doubling changes.

THE DATA (read only). The per-page L1 stores move 1 recorded in <run>/moves/01_extract/extract.json, field
recordings.<cell>.store: arrays seq (the pair index, 1-based), page_index, hamming (h) and cosine (the distance d);
all admissible recordings with a done series (the real run: 103), both cuts of params.json. The same pairs as the
run's series: the keep-first rule of extract.build_series (rows with seq <= seq_first + N - 1 for the three double
recordings, from the record's keep_first_pairs), the pair axis = every pair of the kept range, then the cut of
stats.cut_series (the first C pairs and the last pair dropped). Rows with h > 0 only, as build_series counts them.
IDENTITY CHECK, before any result: summing each pair's page arrows with the SINGLE angle (N = the count, H = sum h,
C = sum h cos theta, S = sum h sin theta) must reproduce the series' N, H, C and S in <run>/series/<cell>.npz after
the cut (relative tolerance 1e-6, the stores hold float32); a failure stops the step (exit 1), nothing scored.

THE TEST
- Per page that changes at least once after the cut: its complex row over the cut pairs; the page's mean arrow (its
  level: the mean of the row over the cut pairs, and over the pairs it changed in) kept separately; Welch two-sided on
  the complex row (128-pair Hann segments, half overlap, the mean removed per segment, as similarity.log_spectrum does
  for one real series; scipy's periodic Hann), the spectrum over f = -0.5 .. 0.5 cycles per pair, normalised to sum 1,
  so every page counts once. A page whose row has no power (a constant row under level removed; under level kept a
  row whose only changes fall where the periodic Hann window is zero) is counted as "constant" and left out of the averages.
- Per recording: the page-averaged spectrum (the spectrum of the whole signal), also for three activity groups by the
  share of cut pairs the page changed in: busy (>= 95%), rare (<= 5%), the rest; and the asymmetry, the share of power
  at positive minus negative frequencies (the f = 0 bin in neither; 0 for a real signal), per page (its distribution)
  and of the page-averaged spectrum.
- Between recordings, as move 5's spectral test (similarity.spectral_similarity, the same code copied here with the
  spectrum given): the log of the page-averaged spectrum with the f = 0 bin dropped, correlations within a kernel
  against between kernels, the gap, a label null of --n-shuffles shuffles (the run's seed), idle as its own group as
  move 5 has it; move 5's angle result (moves/05_similarity/cut<C>/spectral_similarity.csv, the N, H and A rows) is
  written beside it.

THE OUTPUTS, under <out>/<step>/cut<C>/ (<step> = step1_doubled or step2_single; <out> defaults to <run>_pagefourier)
  recordings.csv        per recording: group, the pairs after the cut, nperseg, the pages, the three activity groups'
                        counts, the constant pages, the asymmetry (the pages' mean and median, the averaged spectrum's),
                        the identity check's verdict and largest deviation, the time taken
  spectra/<cell>.csv    per recording: f, the page-averaged spectrum (all pages, busy, rare, rest) and the page counts
  pages/<cell>.npz      per recording and page (compressed; np.load reads it): page_index, n_changes, activity, group, the
                        mean arrow (|mean| and its angle, over the cut pairs and over the changed pairs), asymmetry, the peak
                        frequency, constant
  spectra_by_group.csv  the mean over each group's recordings of the page-averaged spectrum (the figure's data)
  between.csv           this step's within, between, gap, null and p (page-averaged spectrum), with move 5's rows beside it
  per_group.csv         within-group and to-other-groups correlations per group
  asymmetry.csv         per recording and per group
  identity_check.csv    per recording: N, H, C, S reproduced after the cut, the largest relative deviation
  spectrum_by_group.svg, within_between.svg, asymmetry_by_kernel.svg (both steps when both exist), pages_image_<cell>.svg
                        (the many signals at once: pages sorted by activity x frequency; the PNG it embeds, the matrix as
                        .npy, the page order as CSV), each with its CSV
  summary.json          the cut's record
  <out>/<step>/step.json, <out>/record.json (one entry per run: command, inputs with sha256, params, the code
  fingerprint of this module's closure, status; a finished step is skipped unless --force or an input, a parameter or
  the code changed: idle_class's rule)
--check-only runs the identity check alone on every selected recording (identity_check.csv per cut) and stops.

THE RULES THAT PROTECT THE LIVE RUN (as idle_class.py and new_blocks.py)
- ONE new file; nothing existing in plan12_grounding/ is edited; nothing imports this module, so no finished move turns
  stale (run_moves.module_closure). The run folder and the L1 store are read only here; --out inside the run is refused
  (exit 2). The step refuses to start (exit 3) while <run>/.driver.lock belongs to a live process; --dry-run skips that
  refusal and writes nothing. One writer per output folder: <out>/.page_fourier.lock; run_moves.install_sigterm.
- Memory: the rows are built in chunks of --chunk-pages pages (a recording's store is read once; idle touches about
  30,000 pages, a kernel about 10,000).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import base64  # noqa: E402
import csv  # noqa: E402
import html  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import get_window  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding import idle_class as I  # noqa: E402  (the lock state, the output-folder rule, the record pattern; imported, never edited)
from plan12_grounding.run_moves import _pid_alive, code_fingerprint, install_sigterm, now_iso, read_json, sha256_file, write_json  # noqa: E402
from plan12_grounding.stats import ORDER, cut_series, cuts_of, load_runs  # noqa: E402
from plan12_grounding.figures import PALETTE, svg_open, write_csv  # noqa: E402
from plan10_analysis.runner import extract as L1  # noqa: E402  (the console's L1 store; read only)

CITATION = ("spl_paper/main_v2.tex Section IV (Delta(t,n) = Delta_mag e^{j Delta_ori}, Delta in C^{N x (T-1)}); the author's phase phi = 2 theta "
            "(2026-10-10); plan12_grounding/SPEC.md section 3 (theta = arccos(clip(1 - d, 0, 1))), move 5 (the spectral test); plan12_grounding/extract.py "
            "build_series (the keep-first rule, the pair axis, rows with h > 0), stats.py cut_series (the cut), similarity.py log_spectrum / spectral_similarity "
            "(Welch, 128-pair Hann, half overlap, the mean removed; within against between, the label null), copied here with the spectrum given")
STEPS = {1: ("step1_doubled", 2.0, "phi = 2 theta (the headline: 0 and 180 degrees opposite)"), 2: ("step2_single", 1.0, "phi = theta (the control)")}
# the grid: move -> (weighting, level); 13 is the original and the default
VARIANTS = {13: ("once", "removed"), 14: ("raw", "kept"), 15: ("once", "kept"), 16: ("raw", "removed")}
WEIGHTING_TEXT = {"once": "each page counts once (its spectrum normalised to sum 1 before the average)",
                  "raw": "raw spectra (the plain mean of the pages' power spectra; pages with more bits weigh more)"}
LEVEL_TEXT = {"removed": "level removed (the mean removed per segment; the f = 0 bin dropped in the between-recordings test)",
              "kept": "level kept (no mean removed; the f = 0 bin kept in the spectra and in the between-recordings test)"}
OUT_SUFFIX = "_pagefourier"
LOCK_NAME = ".page_fourier.lock"
RECORD_NAME = "record.json"
NPERSEG = 128
BUSY_MIN, RARE_MAX = 0.95, 0.05
RTOL = 1e-6
DEFAULT_CHUNK, DEFAULT_SHUFFLES, DEFAULT_SEED = 2000, 1000, 20260930
DEFAULT_IMAGE_CELL = "floyd__rep00__01c1"
EXIT_OK, EXIT_ERROR, EXIT_MISSING, EXIT_REFUSED = I.EXIT_OK, I.EXIT_ERROR, I.EXIT_MISSING, I.EXIT_REFUSED
GROUPS = ("busy", "rare", "rest")
ASYM_RULE = "the share of the page's normalised power at f > 0 minus the share at f < 0 (the f = 0 bin in neither; 0 for a real signal)"
Stop = I.Stop


def fmt(x) -> str:
    return I.fmt(x)


# ---------------------------------------------------------------------------------------------
# the rows of one recording: the store after the keep-first rule and the cut, the identity check
# ---------------------------------------------------------------------------------------------
def load_store_rows(rec: dict, cut: int) -> dict:
    """The store's rows on the run's pair axis after the keep-first rule and the cut: (seq, page, h, theta) with h > 0,
    the cut pair axis, and the per-pair sums with the single angle for the identity check."""
    z = L1.load(Path(rec["store"]))
    seq = np.asarray(z["seq"]).astype(np.int64)
    h = np.asarray(z["hamming"]).astype(np.float64)
    d = np.asarray(z["cosine"]).astype(np.float64)
    page = np.asarray(z["page_index"]).astype(np.int64)
    keep = np.ones(seq.size, dtype=bool)
    kf = rec.get("keep_first_pairs")
    kf = int(kf) if kf not in (None, "", "None") else None
    if kf and seq.size:
        keep &= seq <= int(seq.min()) + kf - 1
    pairs = np.unique(seq[keep])                       # the pair axis as build_series makes it
    n = int(pairs.size)
    k = min(max(int(cut), 0), n)
    hi = max(k, n - 1)
    pairs_cut = pairs[k:hi]                           # stats.cut_series: the first `cut` pairs and the last pair dropped
    T = int(pairs_cut.size)
    m = keep & (h > 0)
    if T:
        m &= (seq >= pairs_cut[0]) & (seq <= pairs_cut[-1])
    else:
        m &= False
    seq_k, h_k, d_k, page_k = seq[m], h[m], d[m], page[m]
    theta = np.arccos(np.clip(1.0 - d_k, 0.0, 1.0))
    col = np.searchsorted(pairs_cut, seq_k) if T else np.zeros(0, dtype=np.int64)
    sums = {"N": np.bincount(col, minlength=T).astype(np.int64), "H": np.bincount(col, weights=h_k, minlength=T),
            "C": np.bincount(col, weights=h_k * np.cos(theta), minlength=T), "S": np.bincount(col, weights=h_k * np.sin(theta), minlength=T)}
    return {"pairs_cut": pairs_cut, "T": T, "col": col, "page": page_k, "h": h_k, "theta": theta, "sums": sums,
            "n_rows_file": int(seq.size), "n_rows_cut": int(m.sum()), "n_pages_file": int(z["n_pages"]), "keep_first_pairs": kf}


def identity_check(run_rec: dict, rows: dict, cut: int) -> dict:
    """The series' N, H, C, S after the cut (stats.cut_series on the run's npz) against the sums of the page arrows."""
    a = cut_series(run_rec, cut)
    res = {"cell_id": run_rec["cell_id"], "cut": int(cut), "n_pairs_series": int(a["pair"].size), "n_pairs_store": rows["T"], "ok": True, "max_rel_dev": 0.0, "first_difference": None}
    if int(a["pair"].size) != rows["T"] or (rows["T"] and not np.array_equal(a["pair"], rows["pairs_cut"])):
        res.update(ok=False, first_difference=f"the pair axis differs: series {int(a['pair'].size)} pairs, store {rows['T']}")
        return res
    for key in ("N", "H", "C", "S"):
        x, y = np.asarray(a[key], dtype=np.float64), rows["sums"][key].astype(np.float64)
        dev = np.abs(x - y) / np.maximum(1.0, np.abs(x))
        res["max_rel_dev"] = max(res["max_rel_dev"], float(dev.max()) if dev.size else 0.0)
        if dev.size and dev.max() > RTOL:
            j = int(np.argmax(dev))
            res.update(ok=False, first_difference=f"{key} at pair {int(a['pair'][j])}: series {x[j]!r}, page arrows {y[j]!r}")
            return res
    return res


# ---------------------------------------------------------------------------------------------
# the spectra: Welch two-sided on complex rows, in chunks of pages
# ---------------------------------------------------------------------------------------------
def welch_two_sided(rows: np.ndarray, nperseg: int, remove_mean: bool = True) -> np.ndarray:
    """Welch on complex rows (P x T): nperseg-sample segments with half overlap, the mean removed per segment (level
    removed) or kept, scipy's periodic Hann, |FFT|^2 averaged over the segments, fftshift so that bin 0 is f = -0.5;
    (P x nperseg)."""
    hop = nperseg // 2
    T = rows.shape[1]
    n_seg = (T - nperseg) // hop + 1
    idx = np.arange(nperseg)[None, :] + hop * np.arange(n_seg)[:, None]
    seg = rows[:, idx]
    if remove_mean:
        seg = seg - seg.mean(axis=2, keepdims=True)
    w = get_window("hann", nperseg).astype(np.float64)
    X = np.fft.fft(seg * w, axis=2)
    P = (np.abs(X) ** 2).mean(axis=1)
    return np.fft.fftshift(P, axes=1)


def freqs(nperseg: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(nperseg))


def recording_spectra(rows: dict, phase_factor: float, nperseg: int, chunk: int, want_image: bool, weighting: str = "once", level: str = "removed") -> dict:
    """Every changed page's two-sided spectrum (normalised to sum 1 under `once`, the plain power under `raw`), averaged
    over all pages and over the activity groups; the asymmetry per page (always a share of the page's own power); the
    pages' table; optionally the full page x frequency matrix for the image. `level` removed takes the mean out per
    segment; kept leaves it in (the f = 0 bin then holds the level)."""
    T = rows["T"]
    pages, inv = np.unique(rows["page"], return_inverse=True)
    n_pages = int(pages.size)
    f = freqs(nperseg)
    pos, neg = f > 0, f < 0
    dc = int(np.flatnonzero(np.isclose(f, 0.0))[0])
    n_changes = np.bincount(inv, minlength=n_pages)
    activity = n_changes / float(T)
    group = np.where(activity >= BUSY_MIN, "busy", np.where(activity <= RARE_MAX, "rare", "rest"))
    phi = phase_factor * rows["theta"]
    arrow = rows["h"] * np.exp(1j * phi)
    mean_all = np.zeros(n_pages, dtype=np.complex128)
    np.add.at(mean_all, inv, arrow)
    mean_changed = mean_all / np.maximum(n_changes, 1)
    mean_all = mean_all / float(T)
    sum_all = np.zeros(nperseg); sum_group = {g: np.zeros(nperseg) for g in GROUPS}
    n_used = 0; n_used_group = {g: 0 for g in GROUPS}
    asym = np.full(n_pages, np.nan); peak = np.full(n_pages, -1, dtype=np.int64); constant = np.zeros(n_pages, dtype=bool)
    image = np.zeros((n_pages, nperseg), dtype=np.float32) if want_image else None
    order = np.argsort(inv, kind="stable")             # the rows grouped by page
    starts = np.searchsorted(inv[order], np.arange(n_pages + 1))
    for c0 in range(0, n_pages, chunk):
        c1 = min(n_pages, c0 + chunk)
        sel = order[starts[c0]:starts[c1]]
        M = np.zeros((c1 - c0, T), dtype=np.complex128)
        M[inv[sel] - c0, rows["col"][sel]] = arrow[sel]
        P = welch_two_sided(M, nperseg, remove_mean=(level == "removed"))
        tot = P.sum(axis=1)
        ok = tot > 0
        share = np.where(ok[:, None], P / np.where(ok, tot, 1.0)[:, None], 0.0)      # the page's own power as shares, for its asymmetry and its peak
        P = share if weighting == "once" else np.where(ok[:, None], P, 0.0)
        constant[c0:c1] = ~ok
        asym[c0:c1] = np.where(ok, share[:, pos].sum(axis=1) - share[:, neg].sum(axis=1), np.nan)
        Pp = share.copy(); Pp[:, dc] = -1.0
        peak[c0:c1] = np.where(ok, np.argmax(Pp, axis=1), -1)
        sum_all += P[ok].sum(axis=0); n_used += int(ok.sum())
        for g in GROUPS:
            mg = ok & (group[c0:c1] == g)
            sum_group[g] += P[mg].sum(axis=0); n_used_group[g] += int(mg.sum())
        if image is not None:
            image[c0:c1] = P.astype(np.float32)
    mean_all_spec = sum_all / max(n_used, 1)
    mean_group_spec = {g: (sum_group[g] / n_used_group[g] if n_used_group[g] else np.full(nperseg, np.nan)) for g in GROUPS}
    if weighting == "once":                              # the pages' spectra each sum to 1, so their mean does too: the difference is already a share (move 13's expression, kept as written)
        asym_spec = float(mean_all_spec[pos].sum() - mean_all_spec[neg].sum()) if n_used else float("nan")
    else:                                                # raw power: the difference as a share of the averaged spectrum's power
        tot_mean = float(mean_all_spec.sum())
        asym_spec = float((mean_all_spec[pos].sum() - mean_all_spec[neg].sum()) / tot_mean) if (n_used and tot_mean > 0) else float("nan")
    return {"f": f, "pages": pages, "n_pages": n_pages, "n_changes": n_changes, "activity": activity, "group": group,
            "mean_all": mean_all, "mean_changed": mean_changed, "asym_pages": asym, "peak": peak, "constant": constant,
            "spectrum": mean_all_spec, "spectrum_group": mean_group_spec, "n_used": n_used, "n_used_group": n_used_group,
            "asym_spectrum": asym_spec, "image": image, "dc": dc, "weighting": weighting, "level": level}


# ---------------------------------------------------------------------------------------------
# between recordings: move 5's spectral test with the spectrum given (copied from similarity.spectral_similarity)
# ---------------------------------------------------------------------------------------------
def within_between(L: np.ndarray, labels: np.ndarray, rng: np.random.Generator, n_shuffles: int) -> dict:
    # copied from plan12_grounding/similarity.py spectral_similarity, 2026-10-10, with the log spectra given
    C = np.corrcoef(L) if L.shape[0] > 1 else np.ones((1, 1))
    C = np.nan_to_num(C, nan=0.0)
    off = ~np.eye(L.shape[0], dtype=bool)

    def wb(lab):
        same = lab[:, None] == lab[None, :]
        w = C[same & off]; b = C[~same]
        return (float(w.mean()) if w.size else float("nan")), (float(b.mean()) if b.size else float("nan"))
    w, b = wb(labels)
    obs = w - b
    null = np.array([np.subtract(*wb(labels[rng.permutation(labels.size)])) for _ in range(n_shuffles)])
    null = null[np.isfinite(null)]
    per_group = []
    for g in dict.fromkeys(labels.tolist()):
        m = labels == g
        ww = C[np.ix_(m, m)][off[np.ix_(m, m)]]
        bb = C[np.ix_(m, ~m)]
        per_group.append({"group": g, "n_runs": int(m.sum()), "within_corr": float(ww.mean()) if ww.size else float("nan"),
                          "to_other_groups_corr": float(bb.mean()) if bb.size else float("nan")})
    return {"n_bins": int(L.shape[1]), "within_corr": w, "between_corr": b, "within_minus_between": obs,
            "null_mean": float(null.mean()) if null.size else float("nan"), "null_p95": float(np.quantile(null, 0.95)) if null.size else float("nan"),
            "p_value": float((1 + (null >= obs).sum()) / (1 + null.size)) if null.size else float("nan"), "n_shuffles": int(null.size), "per_group": per_group}


def move5_rows(run: Path, cut: int) -> list[dict]:
    p = run / "moves" / "05_similarity" / f"cut{cut}" / "spectral_similarity.csv"
    if not p.is_file():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
def _colour(g: str) -> str:
    return "#9a9a9a" if g == "idle" else PALETTE[ORDER.index(g) % len(PALETTE)] if g in ORDER else "#333333"


def spectrum_by_group_svg(by_group: dict, f: np.ndarray, step_text: str, cut: int) -> str:
    groups = [g for g in ORDER if g in by_group] + [g for g in by_group if g not in ORDER]
    cols, pw, ph = 4, 215, 110
    rows_ = (len(groups) + cols - 1) // cols
    W_, H_ = 40 + cols * (pw + 15), 60 + rows_ * (ph + 40)
    out = svg_open(W_, H_, f"The page-averaged spectrum per kernel and idle, cut of {cut} pairs, {step_text}",
                   "each panel: the mean over the group's recordings of its page-averaged two-sided spectrum (log10 of the normalised power) against f in cycles per pair, -0.5 to 0.5")
    allv = np.concatenate([np.log10(np.asarray(v) + 1e-12) for v in by_group.values() if np.isfinite(v).any()])
    ylo, yhi = float(np.nanmin(allv)), float(np.nanmax(allv))
    if yhi <= ylo:
        yhi = ylo + 1.0
    for gi, g in enumerate(groups):
        px, py = 40 + (gi % cols) * (pw + 15), 50 + (gi // cols) * (ph + 40)
        out.append(f'<text x="{px}" y="{py - 5}" font-weight="bold" fill="{_colour(g)}">{html.escape(g)}</text>')
        out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="none" stroke="#ccc"/>')
        xm = px + pw / 2
        out.append(f'<line x1="{xm:.1f}" y1="{py}" x2="{xm:.1f}" y2="{py + ph}" stroke="#eee"/>')
        v = np.log10(np.asarray(by_group[g]) + 1e-12)
        pts = " ".join(f"{px + pw * (fi + 0.5):.1f},{py + ph - ph * (vi - ylo) / (yhi - ylo):.1f}" for fi, vi in zip(f, v) if np.isfinite(vi))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{_colour(g)}" stroke-width="1.6"/>')
        for t in (-0.5, 0.0, 0.5):
            out.append(f'<text x="{px + pw * (t + 0.5):.1f}" y="{py + ph + 11}" text-anchor="middle" font-size="8">{t:g}</text>')
        out.append(f'<text x="{px - 3}" y="{py + 8}" text-anchor="end" font-size="7">{yhi:.1f}</text><text x="{px - 3}" y="{py + ph}" text-anchor="end" font-size="7">{ylo:.1f}</text>')
    out.append(f'<text x="40" y="{H_ - 8}" fill="#555">f &lt; 0 and f &gt; 0 differ only for a complex signal; a real signal is symmetric</text></svg>')
    return "\n".join(out)


def within_between_svg(res: dict, m5: list[dict], step_text: str, cut: int) -> str:
    bars = [("this step: pages", res)] + [(f"move 5: {r['series']} (real, one-sided)", {k: float(r[k]) for k in ("within_corr", "between_corr", "within_minus_between", "null_p95", "p_value")}) for r in m5]
    W_, H_ = 120 + len(bars) * 150, 300
    out = svg_open(W_, H_, f"Within a kernel against between kernels: the log spectra's correlations, cut of {cut} pairs, {step_text}",
                   "per test: within (dark), between (light), the gap (within minus between) as the number, the null's 95th percentile of the gap as the red tick")
    x0, y0, w, h = 60, 60, len(bars) * 150, 190
    lo = min(-0.1, min(min(b["within_corr"], b["between_corr"], b["null_p95"]) for _, b in bars if np.isfinite(b["within_corr"])))
    hi = max(1.0, max(max(b["within_corr"], b["between_corr"]) for _, b in bars if np.isfinite(b["within_corr"])))
    Y = lambda v: y0 + h - h * (v - lo) / (hi - lo)         # noqa: E731
    out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
    for t in (0.0, 0.5, 1.0):
        out.append(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}" stroke="#eee"/><text x="{x0 - 4}" y="{Y(t) + 3:.1f}" text-anchor="end">{t:.1f}</text>')
    for i, (name, b) in enumerate(bars):
        bx = x0 + i * 150 + 20
        out.append(f'<rect x="{bx}" y="{Y(max(b["within_corr"], 0)):.1f}" width="40" height="{abs(Y(0) - Y(b["within_corr"])):.1f}" fill="#2b5d8a"/>')
        out.append(f'<rect x="{bx + 50}" y="{Y(max(b["between_corr"], 0)):.1f}" width="40" height="{abs(Y(0) - Y(b["between_corr"])):.1f}" fill="#9fbad6"/>')
        out.append(f'<line x1="{bx - 5}" y1="{Y(b["null_p95"]):.1f}" x2="{bx + 95}" y2="{Y(b["null_p95"]):.1f}" stroke="#b03a2e" stroke-width="1.5"/>')
        out.append(f'<text x="{bx + 45}" y="{y0 + h + 12}" text-anchor="middle" font-size="9">{html.escape(name)}</text>')
        out.append(f'<text x="{bx + 45}" y="{y0 - 6}" text-anchor="middle" font-size="9">gap {b["within_minus_between"]:.3f}, p {b["p_value"]:.3g}</text>')
    out.append("</svg>")
    return "\n".join(out)


def asymmetry_svg(rows_by_step: dict, cut: int) -> str:
    """Per group, the mean asymmetry of the page-averaged spectrum over the group's recordings, one bar per step present."""
    steps = list(rows_by_step)
    groups = [g for g in ORDER if any(g in rows_by_step[s] for s in steps)] + sorted({g for s in steps for g in rows_by_step[s] if g not in ORDER})
    n = len(groups)
    gw = 14 * len(steps) + 10
    W_, ph = 70 + n * gw + 30, 150
    H_ = 60 + ph + 60
    out = svg_open(W_, H_, f"The asymmetry per kernel and idle, cut of {cut} pairs",
                   "bars: the mean over the group's recordings of the page-averaged spectrum's asymmetry (power at f > 0 minus f < 0), " + ", ".join(f"{s}" for s in steps) + " left to right; 0 for a real signal")
    vals = [rows_by_step[s][g]["mean"] for s in steps for g in groups if g in rows_by_step[s]]
    lim = max(0.05, max(abs(v) for v in vals if np.isfinite(v)) if vals else 0.05)
    px, py = 60, 50
    Y = lambda v: py + ph / 2 - (ph / 2) * v / lim         # noqa: E731
    out.append(f'<rect x="{px}" y="{py}" width="{n * gw}" height="{ph}" fill="none" stroke="#ccc"/>')
    out.append(f'<line x1="{px}" y1="{Y(0):.1f}" x2="{px + n * gw}" y2="{Y(0):.1f}" stroke="#888"/>')
    for t in (-lim, 0.0, lim):
        out.append(f'<text x="{px - 4}" y="{Y(t) + 3:.1f}" text-anchor="end" font-size="8">{t:+.2f}</text>')
    shades = ["#2b5d8a", "#d9822b", "#3a9d5d"]
    for gi, g in enumerate(groups):
        gx = px + gi * gw
        for si, s in enumerate(steps):
            v = rows_by_step[s].get(g, {}).get("mean")
            if v is None or not np.isfinite(v):
                continue
            out.append(f'<rect x="{gx + 4 + si * 14}" y="{min(Y(v), Y(0)):.1f}" width="12" height="{abs(Y(v) - Y(0)):.1f}" fill="{shades[si % 3]}"/>')
        out.append(f'<text x="{gx + gw / 2:.1f}" y="{py + ph + 10}" text-anchor="end" font-size="8" transform="rotate(-45 {gx + gw / 2:.1f},{py + ph + 10})">{html.escape(g)}</text>')
    legend = "; ".join(f'<tspan fill="{shades[i % 3]}">{html.escape(s)}</tspan>' for i, s in enumerate(steps))
    out.append(f'<text x="{px}" y="{H_ - 8}" fill="#555">{legend}</text></svg>')
    return "\n".join(out)


def pages_image(spec: dict, cell_id: str, step_text: str, cut: int, d: Path) -> dict:
    """One recording as an image: its pages sorted by activity (busiest at the top) against frequency, log10 of each
    page's normalised power; a PNG (matplotlib), the matrix as .npy, the page order as CSV, and the SVG that embeds the PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(-spec["activity"], kind="stable")
    M = spec["image"][order]
    f = spec["f"]
    np.save(d / f"pages_image_{cell_id}.npy", M)
    write_csv(d / f"pages_image_{cell_id}.csv", ["row", "page_index", "activity", "n_changes", "group", "constant"],
              [[i, int(spec["pages"][j]), float(spec["activity"][j]), int(spec["n_changes"][j]), str(spec["group"][j]), bool(spec["constant"][j])] for i, j in enumerate(order)])
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=110)
    with np.errstate(divide="ignore"):
        im = ax.imshow(np.log10(M + 1e-12), aspect="auto", cmap="viridis", extent=[f[0], f[-1], M.shape[0], 0], interpolation="nearest")
    ax.set_xlabel("frequency, cycles per pair (two-sided)")
    ax.set_ylabel("pages, sorted by activity (busiest at the top)")
    ax.set_title(f"{cell_id}: every changed page's spectrum, cut {cut}, {step_text}", fontsize=9)
    fig.colorbar(im, ax=ax, label="log10 normalised power")
    buf = io.BytesIO()
    fig.savefig(buf, format="png"); fig.savefig(d / f"pages_image_{cell_id}.png", format="png")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    out = svg_open(820, 700, f"{cell_id}: the many signals at once, cut of {cut} pairs, {step_text}",
                   f"{M.shape[0]} pages (rows, sorted by activity) x {M.shape[1]} frequency bins; the PNG beside this file is the image, the .npy the matrix, the .csv the page order")
    out.append(f'<image x="10" y="40" width="800" height="650" href="data:image/png;base64,{b64}"/></svg>')
    (d / f"pages_image_{cell_id}.svg").write_text("\n".join(out))
    return {"svg": str(d / f"pages_image_{cell_id}.svg"), "png": str(d / f"pages_image_{cell_id}.png"), "n_pages": int(M.shape[0])}


# ---------------------------------------------------------------------------------------------
# one cut of one step
# ---------------------------------------------------------------------------------------------
def one_cut(step_dir: Path, cut_name: str, cut: int, runs: list[dict], ex_recs: dict, run: Path, phase_factor: float, step_text: str, *,
            chunk: int, n_shuffles: int, seed: int, image_cell: str | None, check_only: bool, other_step_dir: Path | None,
            weighting: str = "once", level: str = "removed") -> dict:
    d = step_dir / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "spectra").mkdir(exist_ok=True); (d / "pages").mkdir(exist_ok=True)
    rec_rows, id_rows, asym_rows, per_rec = [], [], [], {}
    nperseg = None
    labels = []
    spectra_log, f_axis = [], None
    t_cut = time.time()
    for r in runs:
        t0 = time.time()
        rec = ex_recs[r["cell_id"]]
        rows = load_store_rows(rec, cut)
        ic = identity_check(r, rows, cut)
        id_rows.append([r["cell_id"], r["group"], cut, ic["n_pairs_series"], ic["n_pairs_store"], ic["ok"], ic["max_rel_dev"], ic["first_difference"] or ""])
        if not ic["ok"]:
            write_csv(d / "identity_check.csv", ["cell_id", "group", "cut", "n_pairs_series", "n_pairs_store", "ok", "max_rel_dev", "first_difference"], id_rows)
            raise Stop(EXIT_ERROR, f"stopped: the identity check failed for {r['cell_id']} at cut {cut}: {ic['first_difference']} (largest relative deviation {ic['max_rel_dev']:.3g}); nothing scored")
        if check_only:
            print(f"[pagefourier] cut {cut} {r['cell_id']:<32} identity ok (max rel dev {ic['max_rel_dev']:.2e}; {rows['n_rows_cut']} rows, {rows['T']} pairs) {time.time() - t0:.1f} s", flush=True)
            continue
        if rows["T"] < 4:
            rec_rows.append([r["cell_id"], r["group"], r["kernel"], r["rep"], rows["T"], None, 0, 0, 0, 0, 0, None, None, None, True, ic["max_rel_dev"], round(time.time() - t0, 1),
                             f"not run: {rows['T']} pairs after the cut"])
            print(f"[pagefourier] cut {cut} {r['cell_id']:<32} skipped: {rows['T']} pairs after the cut", flush=True)
            continue
        if nperseg is None:                             # one frequency grid for every recording: the shortest series after the cut bounds the segment
            nperseg = max(4, min(NPERSEG, min(int(cut_series(x, cut)["pair"].size) for x in runs)))
            f_axis = freqs(nperseg)
        want_image = (image_cell is not None and r["cell_id"] == image_cell)
        spec = recording_spectra(rows, phase_factor, nperseg, chunk, want_image, weighting, level)
        per_rec[r["cell_id"]] = spec
        labels.append(r["group"])
        L = spec["spectrum"].copy()
        if level == "removed":
            L = np.delete(L, spec["dc"])                   # the f = 0 bin holds nothing once the mean is removed; kept under `kept`
        spectra_log.append(np.log(L / (L.sum() + 1e-300) + 1e-300))
        write_csv(d / "spectra" / f"{r['cell_id']}.csv", ["f", "all_pages", "busy", "rare", "rest", "n_pages_all", "n_busy", "n_rare", "n_rest"],
                  [[float(f_axis[i]), float(spec["spectrum"][i])] + [float(spec["spectrum_group"][g][i]) for g in GROUPS] + [spec["n_used"]] + [spec["n_used_group"][g] for g in GROUPS]
                   for i in range(nperseg)])
        # the pages' table, compressed (a real recording has 10,000 to 30,000 pages; a CSV per page would run to gigabytes over the steps and cuts)
        np.savez_compressed(d / "pages" / f"{r['cell_id']}.npz", page_index=spec["pages"].astype(np.int64), n_changes=spec["n_changes"].astype(np.int64),
                            activity=spec["activity"].astype(np.float32), group=spec["group"].astype(str), mean_abs_all_pairs=np.abs(spec["mean_all"]).astype(np.float32),
                            mean_angle_all_pairs=np.angle(spec["mean_all"]).astype(np.float32), mean_abs_changed_pairs=np.abs(spec["mean_changed"]).astype(np.float32),
                            mean_angle_changed_pairs=np.angle(spec["mean_changed"]).astype(np.float32), asymmetry=spec["asym_pages"].astype(np.float32),
                            peak_f=np.where(spec["peak"] >= 0, f_axis[np.maximum(spec["peak"], 0)], np.nan).astype(np.float32), constant=spec["constant"],
                            columns=np.array(["page_index", "n_changes", "activity", "group", "mean_abs_all_pairs", "mean_angle_all_pairs", "mean_abs_changed_pairs",
                                              "mean_angle_changed_pairs", "asymmetry", "peak_f", "constant"]), cell_id=np.array(r["cell_id"]), cut=np.int64(cut))
        ap = spec["asym_pages"][np.isfinite(spec["asym_pages"])]
        rec_rows.append([r["cell_id"], r["group"], r["kernel"], r["rep"], rows["T"], nperseg, spec["n_pages"], spec["n_used_group"]["busy"], spec["n_used_group"]["rare"], spec["n_used_group"]["rest"],
                         int(spec["constant"].sum()), float(ap.mean()) if ap.size else None, float(np.median(ap)) if ap.size else None, spec["asym_spectrum"], True, ic["max_rel_dev"],
                         round(time.time() - t0, 1), "ok"])
        asym_rows.append([r["cell_id"], r["group"], float(ap.mean()) if ap.size else None, float(np.median(ap)) if ap.size else None,
                          float(np.quantile(ap, 0.05)) if ap.size else None, float(np.quantile(ap, 0.95)) if ap.size else None, spec["asym_spectrum"], spec["n_used"]])
        if want_image:
            pages_image(spec, r["cell_id"], step_text, cut, d)
            spec["image"] = None
        print(f"[pagefourier] cut {cut} {r['cell_id']:<32} identity ok; {spec['n_pages']} pages (busy {spec['n_used_group']['busy']}, rare {spec['n_used_group']['rare']}, rest "
              f"{spec['n_used_group']['rest']}, constant {int(spec['constant'].sum())}); asymmetry of the averaged spectrum {spec['asym_spectrum']:+.4f}, pages' mean "
              f"{(float(ap.mean()) if ap.size else float('nan')):+.4f}; {time.time() - t0:.1f} s", flush=True)
    write_csv(d / "identity_check.csv", ["cell_id", "group", "cut", "n_pairs_series", "n_pairs_store", "ok", "max_rel_dev", "first_difference"], id_rows)
    if check_only:
        return {"cut": int(cut), "cut_name": cut_name, "check_only": True, "n_recordings": len(id_rows), "identity_all_ok": all(r[5] for r in id_rows),
                "max_rel_dev": max((r[6] for r in id_rows), default=0.0), "elapsed_s": round(time.time() - t_cut, 1)}
    write_csv(d / "recordings.csv", ["cell_id", "group", "kernel", "rep", "n_pairs_cut", "nperseg", "n_pages", "n_busy", "n_rare", "n_rest", "n_constant", "asym_pages_mean",
                                     "asym_pages_median", "asym_spectrum", "identity_ok", "identity_max_rel_dev", "time_s", "status"], rec_rows)
    write_csv(d / "asymmetry.csv", ["cell_id", "group", "asym_pages_mean", "asym_pages_median", "asym_pages_p05", "asym_pages_p95", "asym_spectrum", "n_pages_used"], asym_rows)
    summary = {"cut": int(cut), "cut_name": cut_name, "step": step_text, "phase_factor": phase_factor, "nperseg": nperseg, "n_recordings": len(rec_rows), "n_scored": len(per_rec),
               "weighting": weighting, "weighting_text": WEIGHTING_TEXT[weighting], "level": level, "level_text": LEVEL_TEXT[level],
               "asymmetry_rule": ASYM_RULE, "activity_groups": {"busy": f">= {BUSY_MIN:.0%} of the cut pairs", "rare": f"<= {RARE_MAX:.0%}", "rest": "the others"},
               "identity_all_ok": all(r[5] for r in id_rows), "identity_max_rel_dev": max((r[6] for r in id_rows), default=0.0),
               "time_per_recording_s": {"mean": float(np.mean([r[16] for r in rec_rows])) if rec_rows else None, "max": max((r[16] for r in rec_rows), default=None)}}
    if not per_rec:
        summary["status"] = "not run: no recording with enough pairs after the cut"
        write_json(d / "summary.json", {"schema": "plan12.page_fourier_cut.v1", "citation": CITATION, **summary})
        return summary
    # the group means of the page-averaged spectrum, the figure
    by_group = {}
    for g in dict.fromkeys(labels):
        S = np.array([per_rec[c]["spectrum"] for c, lab in zip(per_rec, labels) if lab == g])
        by_group[g] = S.mean(axis=0)
    write_csv(d / "spectra_by_group.csv", ["group", "f", "mean_power", "n_recordings"],
              [[g, float(f_axis[i]), float(by_group[g][i]), int(labels.count(g))] for g in by_group for i in range(nperseg)])
    (d / "spectrum_by_group.svg").write_text(spectrum_by_group_svg(by_group, f_axis, step_text, cut))
    write_csv(d / "spectrum_by_group.csv", ["group", "f", "log10_mean_power"], [[g, float(f_axis[i]), float(np.log10(by_group[g][i] + 1e-12))] for g in by_group for i in range(nperseg)])
    # between recordings: move 5's test with this step's page-averaged spectra, move 5's own rows beside it
    rng = np.random.default_rng(seed)
    labs = np.array(labels)
    wb = within_between(np.array(spectra_log), labs, rng, n_shuffles)
    m5 = move5_rows(run, cut)
    write_csv(d / "between.csv", ["source", "series", "nperseg", "n_bins", "within_corr", "between_corr", "within_minus_between", "null_mean", "null_p95", "p_value", "n_shuffles"],
              [[("this step (pages, two-sided, f = 0 dropped)" if (weighting, level) == ("once", "removed") else f"this step (pages, two-sided, {weighting}, f = 0 {'dropped' if level == 'removed' else 'kept'})"),
                f"pages, {step_text}", nperseg, wb["n_bins"], wb["within_corr"], wb["between_corr"], wb["within_minus_between"], wb["null_mean"], wb["null_p95"],
                wb["p_value"], wb["n_shuffles"]]] + [[f"move 5 (moves/05_similarity/cut{cut}/spectral_similarity.csv)", r["series"], r["nperseg"], r["n_bins"], r["within_corr"], r["between_corr"],
                                                     r["within_minus_between"], r["null_mean"], r["null_p95"], r["p_value"], ""] for r in m5])
    write_csv(d / "per_group.csv", ["group", "n_runs", "within_corr", "to_other_groups_corr"], [[g["group"], g["n_runs"], g["within_corr"], g["to_other_groups_corr"]] for g in wb["per_group"]])
    (d / "within_between.svg").write_text(within_between_svg(wb, m5, step_text, cut))
    write_csv(d / "within_between.csv", ["source", "within_corr", "between_corr", "within_minus_between", "null_p95", "p_value"],
              [["this step", wb["within_corr"], wb["between_corr"], wb["within_minus_between"], wb["null_p95"], wb["p_value"]]] + [[f"move 5 {r['series']}", r["within_corr"], r["between_corr"], r["within_minus_between"], r["null_p95"], r["p_value"]] for r in m5])
    # the asymmetry per group, this step and the other step when it exists
    rows_by_step = {step_dir.name: {}}
    for g in dict.fromkeys(labels):
        v = np.array([x[6] for x in asym_rows if x[1] == g and x[6] is not None], dtype=float)
        rows_by_step[step_dir.name][g] = {"mean": float(v.mean()) if v.size else float("nan"), "n": int(v.size)}
    if other_step_dir is not None and (other_step_dir / f"cut{cut}" / "asymmetry.csv").is_file():
        with open(other_step_dir / f"cut{cut}" / "asymmetry.csv", newline="") as fh:
            orows = list(csv.DictReader(fh))
        rows_by_step[other_step_dir.name] = {}
        for g in dict.fromkeys(x["group"] for x in orows):
            v = np.array([float(x["asym_spectrum"]) for x in orows if x["group"] == g and x["asym_spectrum"] not in ("", None)], dtype=float)
            rows_by_step[other_step_dir.name][g] = {"mean": float(v.mean()) if v.size else float("nan"), "n": int(v.size)}
        rows_by_step = dict(sorted(rows_by_step.items()))
    (d / "asymmetry_by_kernel.svg").write_text(asymmetry_svg(rows_by_step, cut))
    write_csv(d / "asymmetry_by_kernel.csv", ["step", "group", "mean_asym_spectrum", "n_recordings"], [[s, g, v["mean"], v["n"]] for s, gs in rows_by_step.items() for g, v in gs.items()])
    summary.update({"status": "ok", "between": {k: v for k, v in wb.items() if k != "per_group"}, "per_group": wb["per_group"], "move5_rows": m5,
                    "asymmetry_by_group": rows_by_step, "image_cell": image_cell if image_cell in per_rec else None,
                    "files": sorted(p.name for p in d.iterdir() if p.is_file()), "elapsed_s": round(time.time() - t_cut, 1)})
    write_json(d / "summary.json", {"schema": "plan12.page_fourier_cut.v1", "citation": CITATION, **summary})
    print(f"[pagefourier] cut {cut} ({cut_name}): {len(per_rec)} recordings scored, nperseg {nperseg}; within {wb['within_corr']:.3f}, between {wb['between_corr']:.3f}, gap "
          f"{wb['within_minus_between']:.3f} (null p95 {wb['null_p95']:.3f}, p {wb['p_value']:.3g})" + "".join(f"; move 5 {r['series']}: gap {float(r['within_minus_between']):.3f}" for r in m5)
          + f"; {summary['elapsed_s']} s", flush=True)
    return summary


# ---------------------------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------------------------
def select_runs(run: Path, only: list[str] | None) -> list[dict]:
    runs = load_runs(run)
    if only:
        want = set(only)
        missing = want - {r["cell_id"] for r in runs}
        if missing:
            raise Stop(EXIT_MISSING, f"--only names cells without a complete series in this run: {sorted(missing)}")
        runs = [r for r in runs if r["cell_id"] in want]
    return runs


def default_out(run: Path) -> Path:
    return run.parent / (run.name + OUT_SUFFIX)


def resolve_out(run: Path, out_arg: str | None) -> Path:
    out = Path(os.path.expanduser(out_arg)).resolve() if out_arg else default_out(run)
    if out == run or run in out.parents:
        raise Stop(EXIT_MISSING, f"refused: --out {out} lies inside the run folder {run}, which is read only here; the default is {default_out(run)}")
    return out


def acquire_out_lock(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    lock = out / LOCK_NAME
    if lock.exists():
        try:
            old = read_json(lock)
        except Exception:                                  # noqa: BLE001
            old = {}
        pid = old.get("pid")
        if pid and _pid_alive(pid) and int(pid) != os.getpid():
            raise Stop(EXIT_REFUSED, f"refused: another page_fourier step is writing {out} ({lock}: pid {pid}, started {old.get('started_at')}); one writer per output folder")
        print(f"[pagefourier] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
    write_json(lock, {"pid": os.getpid(), "started_at": now_iso(), "argv": sys.argv})
    return lock


def release_out_lock(lock: Path) -> None:
    try:
        if lock.exists() and read_json(lock).get("pid") == os.getpid():
            lock.unlink()
    except Exception:                                      # noqa: BLE001
        pass


def load_record(out: Path) -> dict:
    p = out / RECORD_NAME
    if p.is_file():
        try:
            return read_json(p)
        except Exception:                                  # noqa: BLE001
            p.rename(p.with_suffix(".json.unreadable_" + now_iso().replace(":", "")))
    return {"schema": "plan12.page_fourier.record.v1", "citation": CITATION, "package_version": __version__, "entries": []}


def save_record(out: Path, rec: dict) -> None:
    write_json(out / RECORD_NAME, rec)


def last_done(rec: dict, key: str) -> dict | None:
    for e in reversed(rec.get("entries") or []):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def code_identity() -> dict:
    fp = code_fingerprint("page_fourier")
    files = toolkit_fingerprint()["files"]
    return {"fingerprint": fp["sha256"], "modules": fp["modules"], "sha256": {f"{m}.py": files.get(f"{m}.py") for m in fp["modules"]}}


def inputs_identity(run: Path, runs: list[dict], ex_recs: dict, cuts: dict) -> dict:
    rels = ["cells.csv", "params.json", "moves/01_extract/extract.json"] + [f"moves/05_similarity/cut{c}/spectral_similarity.csv" for c in cuts.values()]
    files = {rel: (sha256_file(run / rel) if (run / rel).is_file() else "absent") for rel in rels}
    stores = {}
    for r in runs:
        rec = ex_recs[r["cell_id"]]
        p = Path(rec["store"])
        stores[str(p)] = rec.get("store_sha256") or (sha256_file(p) if p.is_file() else "absent")   # the sha256 move 1 recorded for the store
        s = Path(rec.get("series") or (run / "series" / f"{r['cell_id']}.npz"))
        key = str(s.relative_to(run)) if str(s).startswith(str(run)) else str(s)
        files[key] = sha256_file(s) if s.is_file() else "absent"
    return {"run": str(run), "run_files": files, "stores": stores}


def run_step(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0 and 1 done)")
    step = int(o.step)
    step_name, phase_factor, step_text = STEPS[step]
    other_name = STEPS[2 if step == 1 else 1][0]
    move = int(getattr(o, "move", 13) or 13)
    if move not in VARIANTS:
        raise Stop(EXIT_MISSING, f"--move {move}: the grid is {sorted(VARIANTS)}")
    weighting, level = VARIANTS[move]
    variant_text = f"move {move}: {WEIGHTING_TEXT[weighting]}; {LEVEL_TEXT[level]}"
    sub = Path("") if move == 13 else Path(f"move{move}")       # move 13 keeps its folders and keys; the grid's moves sit beside them
    key_prefix = "" if move == 13 else f"move{move}_"
    out = resolve_out(run, o.out)
    lock = I.driver_lock_state(run)
    if lock["alive"] and not o.dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}, started {lock['started_at']}, {lock['argv_tail']}); "
                                 "this test never competes with a running move. Run again after it ends; --dry-run shows the plan meanwhile.")
    cuts = I.parse_cuts(o.cuts, cuts_of(run))
    exj = read_json(run / "moves" / "01_extract" / "extract.json")
    ex_recs = exj.get("recordings") or {}
    only = [s.strip() for s in str(o.only).split(",") if s.strip()] if o.only else None
    runs = select_runs(run, only)
    missing_store = [r["cell_id"] for r in runs if not Path((ex_recs.get(r["cell_id"]) or {}).get("store") or "").is_file()]
    if missing_store:
        raise Stop(EXIT_MISSING, f"missing input: the L1 store of {len(missing_store)} recording(s) is not on this machine (first: {missing_store[0]}; extract.json names it)")
    image_cell = o.image_cell if (o.image_cell and any(r["cell_id"] == o.image_cell for r in runs)) else None
    n_idle = sum(1 for r in runs if r["group"] == "idle")
    print(f"[pagefourier] {variant_text}")
    print(f"[pagefourier] step {step}: {step_text}; the per-page Fourier test on {len(runs)} recordings ({len(runs) - n_idle} kernel + {n_idle} idle), cuts "
          f"{', '.join(f'{k} = {v}' for k, v in cuts.items())}; Welch {NPERSEG}-pair Hann segments, half overlap, two-sided; chunks of {o.chunk_pages} pages; "
          f"{o.n_shuffles} label shuffles, seed {o.seed}; image of {image_cell or 'no recording (the named cell is not in the selection)'}")
    print(f"[pagefourier] run {run} (read only); stores {Path(ex_recs[runs[0]['cell_id']]['store']).parent} (read only); out {out}{' (exists)' if out.exists() else ' (to be created)'}")
    print(f"[pagefourier] driver lock: " + (f"held by pid {lock['pid']} ({'alive' if lock['alive'] else 'dead'}), started {lock['started_at']}: {lock['argv_tail']}" if lock["exists"] else "absent")
          + ("; the refusal is skipped by --dry-run" if (lock["alive"] and o.dry_run) else ""))
    print(f"[pagefourier] identity check first, every recording: N, H, C, S of the page arrows (single angle) against series/<cell>.npz after the cut, rtol {RTOL:g}"
          + ("; --check-only: nothing else" if o.check_only else ""))
    step_dir = out / sub / step_name
    if o.dry_run:
        print(f"[pagefourier] dry run: would write {step_dir}/cut<C>/ (recordings.csv, spectra/<cell>.csv, pages/<cell>.npz, spectra_by_group.csv, between.csv, per_group.csv, "
              f"asymmetry.csv, identity_check.csv, the figures and summary.json), {step_dir / 'step.json'} and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    code = code_identity()
    key = key_prefix + step_name + ("_check" if o.check_only else "")
    entry = {"key": key, "step": step, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out), "cuts": cuts, "move": move, "weighting": weighting, "level": level, "phase_factor": phase_factor, "phase": step_text,
                        "nperseg": NPERSEG, "chunk_pages": int(o.chunk_pages),
                        "n_shuffles": int(o.n_shuffles), "seed": int(o.seed), "image_cell": image_cell, "only": only, "check_only": bool(o.check_only), "rtol": RTOL,
                        "n_recordings": len(runs), "force": bool(o.force)},
             "inputs_sha256": inputs_identity(run, runs, ex_recs, cuts),
             "code": {"page_fourier.py": code["sha256"].get("page_fourier.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
             "outputs": []}
    sig = {k: v for k, v in entry["params"].items() if k not in ("force", "chunk_pages")}
    rec = load_record(out)
    prev = last_done(rec, key)
    outputs_exist = (step_dir / "step.json").is_file() and all((step_dir / f"cut{c}" / ("identity_check.csv" if o.check_only else "summary.json")).is_file() for c in cuts.values())
    try:
        if prev is not None and not o.force and outputs_exist:
            why = None
            if prev.get("inputs_sha256") != entry["inputs_sha256"]:
                why = "an input changed"
            elif {k: v for k, v in (prev.get("params") or {}).items() if k not in ("force", "chunk_pages")} != sig:
                why = "the parameters changed"
            elif (prev.get("code") or {}).get("fingerprint") != code["fingerprint"]:
                why = "the code changed"
            if why is None:
                entry.update(status="skipped: outputs exist and inputs, parameters and code unchanged", exit_code=0, finished_at=now_iso(),
                             made_by={"finished_at": prev.get("finished_at"), "code_fingerprint": (prev.get("code") or {}).get("fingerprint")}, outputs=prev.get("outputs") or [])
                rec["entries"].append(entry)
                save_record(out, rec)
                print(f"[pagefourier] skipped: {step_dir} stands as made on {prev.get('finished_at')} (inputs, parameters and code unchanged; --force re-runs)")
                return EXIT_OK
            print(f"[pagefourier] re-running: {why} since {prev.get('finished_at')}")
        rec["entries"].append(entry)
        save_record(out, rec)
        step_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        results = [one_cut(step_dir, name, cut, runs, ex_recs, run, phase_factor, step_text, chunk=int(o.chunk_pages), n_shuffles=int(o.n_shuffles), seed=int(o.seed),
                           image_cell=image_cell, check_only=bool(o.check_only), other_step_dir=(out / sub / other_name), weighting=weighting, level=level) for name, cut in cuts.items()]
        step_rec = {"schema": "plan12.page_fourier_step.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv, "step": step,
                    "move": move, "variant": variant_text, "weighting": weighting, "level": level,
                    "phase": step_text, "phase_factor": phase_factor, "why": __doc__.split("WHY.", 1)[1].split("THE DATA", 1)[0].strip(), "params": entry["params"],
                    "recordings": [r["cell_id"] for r in runs], "code": entry["code"], "elapsed_s": round(time.time() - t0, 1), "results": results}
        if not o.check_only:
            write_json(step_dir / "step.json", step_rec)
        else:
            write_json(step_dir / "identity_check.json", step_rec)
        entry["outputs"] = sorted(str(p.relative_to(out)) for p in step_dir.rglob("*") if p.is_file())
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=step_rec["elapsed_s"])
        for s in results:
            if s.get("check_only"):
                print(f"[pagefourier] cut {s['cut']}: identity check {'passed' if s['identity_all_ok'] else 'FAILED'} on {s['n_recordings']} recordings, largest relative deviation {s['max_rel_dev']:.3g}, {s['elapsed_s']} s")
        print(f"[pagefourier] move {move} step {step} done in {step_rec['elapsed_s']} s: {step_dir}")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        code_ = getattr(exc, "code", None)
        entry.update(status=f"failed: {type(exc).__name__}" + (f" (exit {code_})" if isinstance(code_, int) else ""), exit_code=code_ if isinstance(code_, int) else EXIT_ERROR,
                     finished_at=now_iso(), error=str(exc)[:500])
        raise
    finally:
        if entry["status"] == "running":
            entry.update(status="failed: stopped before the end", finished_at=now_iso())
        rec = load_record(out)
        rec["entries"] = [e for e in rec["entries"] if not (e.get("key") == entry["key"] and e.get("started_at") == entry["started_at"])] + [entry]
        save_record(out, rec)
        release_out_lock(out_lock)


# ---------------------------------------------------------------------------------------------
# the grid's summary: the eight variants (moves 13 to 16, two steps) at both cuts
# ---------------------------------------------------------------------------------------------
def variant_dir(out: Path, move: int, step_name: str) -> Path:
    return out / step_name if move == 13 else out / f"move{move}" / step_name


def grid_summary(run: Path, out: Path, dry_run: bool = False) -> dict:
    cuts = cuts_of(run)
    rows, missing = [], []
    for move in sorted(VARIANTS):
        w, lv = VARIANTS[move]
        for step in sorted(STEPS):
            step_name, _, step_text = STEPS[step]
            for cut_name, cut in cuts.items():
                d = variant_dir(out, move, step_name) / f"cut{cut}"
                row = {"move": move, "weighting": w, "level": lv, "step": step, "phase": step_text, "cut": cut, "cut_name": cut_name, "status": "missing (not run)",
                       "n_recordings": None, "within_corr": None, "between_corr": None, "gap": None, "null_p95": None, "p_value": None, "mean_asym_spectrum": None, "folder": str(d)}
                sj = d / "summary.json"
                if sj.is_file():
                    try:
                        j = read_json(sj)
                        b = j.get("between") or {}
                        row.update(status=j.get("status") or "ok", n_recordings=j.get("n_scored"), within_corr=b.get("within_corr"), between_corr=b.get("between_corr"),
                                   gap=b.get("within_minus_between"), null_p95=b.get("null_p95"), p_value=b.get("p_value"))
                        ac = d / "asymmetry.csv"
                        if ac.is_file():
                            with open(ac, newline="") as fh:
                                vals = [float(r["asym_spectrum"]) for r in csv.DictReader(fh) if r.get("asym_spectrum") not in ("", None)]
                            row["mean_asym_spectrum"] = float(np.mean(vals)) if vals else None
                    except (OSError, ValueError, json.JSONDecodeError) as e:
                        row["status"] = f"unreadable: {e}"
                else:
                    missing.append(f"move {move} step {step} cut {cut}")
                rows.append(row)
    if dry_run:
        print(f"[pagefourier] summary dry run: {len(rows)} variant rows, {len(missing)} missing ({', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}); would write {out / 'grid_summary.csv'}, .svg, .json; nothing written")
        return {"rows": rows, "missing": missing}
    out.mkdir(parents=True, exist_ok=True)
    cols = ["move", "weighting", "level", "step", "phase", "cut", "cut_name", "status", "n_recordings", "within_corr", "between_corr", "gap", "null_p95", "p_value", "mean_asym_spectrum", "folder"]
    write_csv(out / "grid_summary.csv", cols, [[r[c] for c in cols] for r in rows])
    (out / "grid_summary.svg").write_text(grid_summary_svg(rows, cuts))
    write_json(out / "grid_summary.json", {"schema": "plan12.page_fourier_grid.v1", "citation": CITATION, "written_at": now_iso(), "cuts": cuts,
                                           "variants": {m: {"weighting": v[0], "level": v[1]} for m, v in VARIANTS.items()}, "rows": rows, "missing": missing})
    print(f"[pagefourier] grid summary: {len(rows) - len(missing)} of {len(rows)} variant rows present ({len(missing)} missing); {out / 'grid_summary.csv'}, .svg, .json")
    return {"rows": rows, "missing": missing}


def grid_summary_svg(rows: list[dict], cuts: dict) -> str:
    """Per cut a panel: the eight variants (moves 13 to 16 x the two steps) as bars of the within-minus-between gap, the
    null's 95th percentile as a red tick, p and the mean asymmetry printed; a missing variant reads "missing"."""
    variants = [(m, st) for m in sorted(VARIANTS) for st in sorted(STEPS)]
    pw, ph = 60 + len(variants) * 52, 170
    W_, H_ = 40 + pw, 70 + len(cuts) * (ph + 80)
    out = svg_open(W_, H_, "The grid: the per-page Fourier test's eight variants side by side",
                   "bars: the within-minus-between gap of the log page-averaged spectra; red tick: the null's 95th percentile; under each bar p and the mean asymmetry of the averaged spectrum")
    vals = [r["gap"] for r in rows if r["gap"] is not None] + [r["null_p95"] for r in rows if r["null_p95"] is not None]
    lo, hi = (min(-0.05, min(vals)), max(0.2, max(vals))) if vals else (-0.05, 0.2)
    for ci, (cut_name, cut) in enumerate(cuts.items()):
        px, py = 60, 60 + ci * (ph + 80)
        Y = lambda v: py + ph - ph * (v - lo) / (hi - lo)          # noqa: E731
        out.append(f'<text x="{px}" y="{py - 6}" font-weight="bold">cut of {cut} pairs ({cut_name})</text>')
        out.append(f'<rect x="{px}" y="{py}" width="{len(variants) * 52}" height="{ph}" fill="none" stroke="#ccc"/>')
        out.append(f'<line x1="{px}" y1="{Y(0):.1f}" x2="{px + len(variants) * 52}" y2="{Y(0):.1f}" stroke="#888"/>')
        for t in (lo, 0.0, hi):
            out.append(f'<text x="{px - 4}" y="{Y(t) + 3:.1f}" text-anchor="end" font-size="8">{t:.2f}</text>')
        for vi, (m, st) in enumerate(variants):
            r = next((x for x in rows if x["move"] == m and x["step"] == st and x["cut"] == cut), None)
            bx = px + vi * 52 + 8
            label = f"m{m} s{st}"
            out.append(f'<text x="{bx + 18}" y="{py + ph + 12}" text-anchor="middle" font-size="8">{label}</text>')
            out.append(f'<text x="{bx + 18}" y="{py + ph + 22}" text-anchor="middle" font-size="7" fill="#555">{VARIANTS[m][0]}, {VARIANTS[m][1]}</text>')
            if r is None or r["gap"] is None:
                out.append(f'<text x="{bx + 18}" y="{Y(0) - 6:.1f}" text-anchor="middle" font-size="8" fill="#999">missing</text>')
                continue
            colour = "#2b5d8a" if st == 1 else "#d9822b"
            out.append(f'<rect x="{bx}" y="{min(Y(r["gap"]), Y(0)):.1f}" width="36" height="{abs(Y(r["gap"]) - Y(0)):.1f}" fill="{colour}" fill-opacity="0.85"/>')
            if r["null_p95"] is not None:
                out.append(f'<line x1="{bx - 3}" y1="{Y(r["null_p95"]):.1f}" x2="{bx + 39}" y2="{Y(r["null_p95"]):.1f}" stroke="#b03a2e" stroke-width="1.5"/>')
            out.append(f'<text x="{bx + 18}" y="{py + ph + 32}" text-anchor="middle" font-size="7">p {r["p_value"]:.3g}</text>')
            if r["mean_asym_spectrum"] is not None:
                out.append(f'<text x="{bx + 18}" y="{py + ph + 41}" text-anchor="middle" font-size="7">asym {r["mean_asym_spectrum"]:+.4f}</text>')
    out.append(f'<text x="60" y="{H_ - 8}" fill="#555">step 1 (blue): phi = 2 theta; step 2 (orange): phi = theta; m13 once/removed, m14 raw/kept, m15 once/kept, m16 raw/removed</text></svg>')
    return "\n".join(out)


def run_summary(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "params.json").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'params.json'}")
    out = resolve_out(run, o.out)
    lock = I.driver_lock_state(run)
    if lock["alive"] and not o.dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}); run again after it ends")
    if o.dry_run:
        grid_summary(run, out, dry_run=True)
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    entry = {"key": "grid_summary", "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out)}, "code": code_identity(), "outputs": ["grid_summary.csv", "grid_summary.svg", "grid_summary.json"]}
    rec = load_record(out)
    rec["entries"].append(entry); save_record(out, rec)
    try:
        res = grid_summary(run, out)
        entry.update(status="done", exit_code=0, finished_at=now_iso(), n_missing=len(res["missing"]))
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        entry.update(status=f"failed: {type(exc).__name__}", exit_code=EXIT_ERROR, finished_at=now_iso(), error=str(exc)[:500])
        raise
    finally:
        rec = load_record(out)
        rec["entries"] = [e for e in rec["entries"] if not (e.get("key") == entry["key"] and e.get("started_at") == entry["started_at"])] + [entry]
        save_record(out, rec)
        release_out_lock(out_lock)


def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--move", type=int, default=13, choices=sorted(VARIANTS), help="the grid's variant: 13 once/removed (the original, default), 14 raw/kept, 15 once/kept, 16 raw/removed")
    p.add_argument("--run", required=True, help="the main run's output folder (read only here)")
    p.add_argument("--step", required=True, type=int, choices=sorted(STEPS), help="1 = phi = 2 theta (the headline); 2 = phi = theta (the control)")
    p.add_argument("--out", default=None, help="the output folder (default: a sibling of the run, <run>_pagefourier; never inside the run)")
    p.add_argument("--cuts", default=None, help="which cuts, by name: declared,measured (default both; the values come from params.json)")
    p.add_argument("--only", default=None, help="a comma-separated list of cell ids (a test on a few recordings)")
    p.add_argument("--chunk-pages", type=int, default=DEFAULT_CHUNK, help="pages per chunk when the rows are built (memory)")
    p.add_argument("--n-shuffles", type=int, default=DEFAULT_SHUFFLES, help="label shuffles of the between-recording null (move 5: 1000)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="the null's seed (the run's: 20260930)")
    p.add_argument("--image-cell", default=DEFAULT_IMAGE_CELL, help="the recording drawn as an image (default floyd rep00)")
    p.add_argument("--check-only", action="store_true", help="the identity check alone, on every selected recording; nothing else")
    p.add_argument("--force", action="store_true", help="re-run a finished step")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write nothing; skip the live-lock refusal")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.page_fourier", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="step 1 (phi = 2 theta) or step 2 (phi = theta): the per-page Fourier test, both cuts; --move picks the grid's variant"))
    sm = sub.add_parser("summary", help="the grid's summary: moves 13 to 16, two steps, both cuts, side by side")
    sm.add_argument("--run", required=True)
    sm.add_argument("--out", default=None, help="the sibling folder (default <run>_pagefourier)")
    sm.add_argument("--dry-run", action="store_true")
    o = ap.parse_args(argv)
    try:
        return run_summary(o) if o.cmd == "summary" else run_step(o)
    except I.Stop as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return EXIT_MISSING
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(f"stopped: exit {exc.code}", file=sys.stderr)
            return int(exc.code) if isinstance(exc.code, int) else EXIT_ERROR
        return EXIT_OK
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
