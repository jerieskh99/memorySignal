#!/usr/bin/env python3
"""time_shuffle.py -- the time-shuffled control of the per-page Fourier test, beside the run of plan12_grounding
(2026-10-10; move 17 in the console's Grounding paper tab, after the grid of moves 13 to 16).

  python3 -m plan12_grounding.time_shuffle run --run <main run out> --step 1|2 [--out <dir>] [--cuts declared,measured]
        [--only cellA,cellB] [--chunk-pages 2000] [--n-time-shuffles 10] [--block-pairs 64] [--n-shuffles 1000]
        [--seed 20260930] [--pagefourier-out <dir>] [--check-unshuffled] [--force] [--dry-run]
  python3 -m plan12_grounding.time_shuffle summary --run <main run out> [--out <dir>] [--pagefourier-out <dir>] [--dry-run]
  (run from VM_sampler/VM_Capture_QEMU/)

THE QUESTION. The per-page Fourier test (page_fourier.py, moves 13 to 16) finds that the log page-averaged spectra of
one kernel's recordings agree more with each other than with other kernels' (the within-minus-between gap, far above
its label null). Is that gap carried by the ORDER of the pairs in time, or would any arrangement of the same pairs
give it? This control keeps every page's set of arrows and destroys the long-range order: the cut pairs are cut into
blocks of 64 pairs in time order, the order of the blocks is shuffled, and the SAME new order is applied to every
page of the recording (the pages stay aligned with each other, so the shuffle is a shuffle of time, not of pages).
The short tail block (the pairs after the last full block) stays last. Ten shuffles per recording; everything
page_fourier computes is recomputed on the shuffled rows, for the eight variants of the grid.

THE STEPS, as page_fourier's: step 1 is the author's phase phi = 2 theta, step 2 is phi = theta (the control's
control). Each step computes the four (weighting, level) variants of the grid at its phase: move 13 (each page
counts once, level removed), 14 (raw, kept), 15 (once, kept), 16 (raw, removed). Over the two steps: the eight
variants. One shuffle is one block order per recording, the same for both steps and for the four variants.

THE COMPUTATION, per recording, cut and shuffle, with page_fourier's own functions (imported, never edited):
page_fourier.load_store_rows reads the store on the run's pair axis after the keep-first rule and the cut;
page_fourier.identity_check checks the page arrows reproduce the series' N, H, C, S (a failure stops the step).
The shuffle changes only the column (pair) index of every row; then one Welch pass per level (page_fourier.
welch_two_sided, the mean removed per segment or kept) over the pages in chunks, from which BOTH weightings are
accumulated at once: the normalised spectrum (each page's spectrum divided by its sum, "each page counts once")
and the raw power, with the same set of pages with power; divided by that count they are exactly page_fourier.
recording_spectra's page-averaged spectra (checked on the first scored recording of every cut against
recording_spectra itself under the unshuffled order, to 1e-9; a mismatch stops the step). Then page_fourier's
between-recordings test (within_between: the log of the page-averaged spectrum, the f = 0 bin dropped under level
removed and kept under level kept, within a kernel against between kernels, the label null of --n-shuffles
shuffles with the run's seed, one fresh generator per call as page_fourier makes one per cut). Welch and the
between test are two passes per shuffle instead of four: the two levels need their own Welch, the two weightings do not.

SEEDS. The block order of shuffle k for recording c at cut C comes from numpy's default_rng seeded with
[--seed, C, k, sha256(c)[:4]], so a recording's orders do not depend on which recordings are selected (--only)
or on the step. The label null's generator is default_rng(--seed), fresh before every within_between call, so every
variant and shuffle draws the same label permutations, as page_fourier's single call per cut does.

THE OUTPUTS, under <out>/<step>/cut<C>/ (<step> = step1_doubled or step2_single; <out> defaults to <run>_timeshuffle)
  recordings.csv     per recording: group, the pairs after the cut, nperseg, the full blocks and the tail, the pages,
                     the pages with power per level (shuffle 0, and the range over the shuffles: under level removed
                     the count does not depend on the order; under level kept it can, because a page whose only change
                     sits at the first pair of a block has no power when that block comes first, the periodic Hann
                     window being zero at a segment's first sample, page_fourier's rule), the identity check, the time taken
  block_orders.csv   per recording and shuffle: the block order applied (old block indices in their new order)
  spectra/<cell>.npz per recording: the page-averaged spectrum of each variant and shuffle (variant x shuffle x f),
                     the frequency axis, the variants, the orders
  gaps.csv           per variant and shuffle: within, between, the gap, the label null's mean and 95th percentile, p
  summary.csv        per variant: the unshuffled gap, null p95 and p read from <run>_pagefourier (the variant's
                     summary.json; "missing" when that variant has not run), and the shuffled gap as mean, spread
                     (std), min and max over the shuffles, the shuffled nulls' p95 mean, and how many shuffles reach
                     the unshuffled gap
  gaps_unshuffled_vs_shuffled.svg (+ .csv)  the figure: per variant the unshuffled gap as a bar, the shuffled gaps as
                     a dot (mean) with whiskers (min to max) and a thick bar (mean +- std), the unshuffled null's
                     p95 as a red tick; this step's four variants, and the other step's four when it has run
  identity_check.csv per recording; summary.json the cut's record (the accumulator check, the timing, the variants)
  <out>/<step>/gaps_both_cuts.svg (+ .csv)  both cuts in one figure; <out>/<step>/step.json the step's record
  <out>/record.json  one entry per run (command, inputs with sha256, params, the code fingerprint of this module's
                     closure, status); a finished step is skipped unless --force or an input, a parameter or the
                     code changed. The page_fourier results are read, not hashed: `summary` re-reads them.
`summary` rewrites summary.csv, summary.json's variant table and the figures of every step and cut present from the
saved gaps.csv and the page_fourier results as they stand now (for a step run before moves 13 to 16 had all finished),
without recomputing; its record key is "summary" and its done file <out>/summary_refresh.json.
--check-unshuffled also recomputes the unshuffled order through this module's pass for every recording and runs the
between test on it; its gap is written beside page_fourier's (summary.csv recomputed_* columns): they must agree.

THE RULES THAT PROTECT THE LIVE RUN (as page_fourier.py)
- ONE new file; nothing existing in plan12_grounding/ is edited; page_fourier.py and new_blocks.py are imported as
  they are (their recorded code fingerprints stay); nothing imports this module. The run folder, <run>_pagefourier
  and the L1 stores are read only here; --out inside the run is refused (exit 2). The step refuses to start (exit 3)
  while <run>/.driver.lock belongs to a live process; --dry-run skips that refusal and writes nothing. One writer
  per output folder: <out>/.time_shuffle.lock; run_moves.install_sigterm.
- Memory: as page_fourier, the rows are built in chunks of --chunk-pages pages; the per-recording results kept in
  memory are the variants' spectra (4 x 10 x 128 numbers per recording).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding import idle_class as I  # noqa: E402  (the lock state, the record pattern; imported, never edited)
from plan12_grounding import page_fourier as PF  # noqa: E402  (the test itself: load_store_rows, identity_check, welch_two_sided, freqs, recording_spectra, within_between)
from plan12_grounding.run_moves import _pid_alive, code_fingerprint, install_sigterm, now_iso, read_json, write_json  # noqa: E402
from plan12_grounding.stats import cut_series, cuts_of  # noqa: E402
from plan12_grounding.figures import svg_open, write_csv  # noqa: E402

CITATION = ("plan12_grounding/page_fourier.py (moves 13 to 16: the per-page Fourier test, its grid, its between-recordings test and label null; "
            "its functions are called here on the shuffled rows); the author's design of the time-shuffled control (2026-10-10): 64-pair blocks "
            "in time order, the block order shuffled, the same order for every page, the tail block last, ten shuffles per recording")
STEPS = PF.STEPS                                           # 1: ("step1_doubled", 2.0, phi = 2 theta); 2: ("step2_single", 1.0, phi = theta)
VARIANTS = PF.VARIANTS                                     # move -> (weighting, level): 13 once/removed, 14 raw/kept, 15 once/kept, 16 raw/removed
VARIANT_LIST = [(m, w, lv) for m, (w, lv) in sorted(VARIANTS.items())]
LEVELS = ("removed", "kept")
OUT_SUFFIX = "_timeshuffle"
LOCK_NAME = ".time_shuffle.lock"
RECORD_NAME = "record.json"
DEFAULT_TIME_SHUFFLES, DEFAULT_BLOCK_PAIRS = 10, 64
DEFAULT_CHUNK, DEFAULT_SHUFFLES, DEFAULT_SEED = PF.DEFAULT_CHUNK, PF.DEFAULT_SHUFFLES, PF.DEFAULT_SEED
ACC_TOL = 1e-9                                             # the accumulator against recording_spectra (the same arithmetic in a different order of summation)
EXIT_OK, EXIT_ERROR, EXIT_MISSING, EXIT_REFUSED = I.EXIT_OK, I.EXIT_ERROR, I.EXIT_MISSING, I.EXIT_REFUSED
SHUFFLE_RULE = ("the cut pairs in time order cut into blocks of --block-pairs pairs; the order of the full blocks drawn from numpy default_rng([seed, cut, shuffle, "
                "sha256(cell_id)[:4]]).permutation; the tail block (the pairs after the last full block) stays last; the same order applied to every page of the "
                "recording (only the pair index of each row changes); a recording with fewer than two full blocks keeps its order (noted in recordings.csv)")
Stop = I.Stop


def fmt(x) -> str:
    return I.fmt(x)


def cell_seed(cell_id: str) -> int:
    return int.from_bytes(hashlib.sha256(cell_id.encode("utf-8")).digest()[:4], "big")


# ---------------------------------------------------------------------------------------------
# the shuffle: a block order per recording, the pair index remapped
# ---------------------------------------------------------------------------------------------
def block_order(T: int, B: int, rng: np.random.Generator | None) -> dict:
    """The full blocks of a series of T pairs (B pairs each), their order drawn from `rng` (None or fewer than two
    full blocks: the unshuffled order), and new_col[old pair index] = the pair index after the shuffle (the tail last)."""
    n_full = int(T // B)
    tail = int(T - n_full * B)
    perm = rng.permutation(n_full) if (rng is not None and n_full >= 2) else np.arange(n_full)
    new_col = np.arange(T, dtype=np.int64)
    for j, b in enumerate(perm.tolist()):
        new_col[b * B:(b + 1) * B] = np.arange(j * B, (j + 1) * B)
    return {"perm": perm.astype(np.int64), "new_col": new_col, "n_full": n_full, "tail": tail, "shuffled": bool(rng is not None and n_full >= 2)}


def prepare(rows: dict, phase_factor: float) -> dict:
    """What page_fourier.recording_spectra computes before the Welch pass and that no shuffle changes: the pages, each
    row's page, the arrows h e^{j phi}, the rows grouped by page."""
    pages, inv = np.unique(rows["page"], return_inverse=True)
    arrow = rows["h"] * np.exp(1j * (phase_factor * rows["theta"]))
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(int(pages.size) + 1))
    return {"T": rows["T"], "pages": pages, "n_pages": int(pages.size), "inv": inv, "arrow": arrow, "order": order, "starts": starts}


def spectra_pass(prep: dict, col: np.ndarray, nperseg: int, chunk: int) -> dict:
    """{level: {"once": spectrum, "raw": spectrum, "n_used": n}}: page_fourier.recording_spectra's page-averaged spectrum
    under both weightings from ONE Welch pass per level. `col` is the pair index of every row (shuffled or not). The
    arithmetic is recording_spectra's: per page P = welch_two_sided(row); tot = P.sum(); a page counts when tot > 0;
    "once" sums P / tot over those pages, "raw" sums P; both are divided by the count of pages with power."""
    T, n_pages = prep["T"], prep["n_pages"]
    acc = {lv: {"once": np.zeros(nperseg), "raw": np.zeros(nperseg), "n_used": 0} for lv in LEVELS}
    for c0 in range(0, n_pages, chunk):
        c1 = min(n_pages, c0 + chunk)
        sel = prep["order"][prep["starts"][c0]:prep["starts"][c1]]
        M = np.zeros((c1 - c0, T), dtype=np.complex128)
        M[prep["inv"][sel] - c0, col[sel]] = prep["arrow"][sel]
        for lv in LEVELS:
            P = PF.welch_two_sided(M, nperseg, remove_mean=(lv == "removed"))
            tot = P.sum(axis=1)
            ok = tot > 0
            share = np.where(ok[:, None], P / np.where(ok, tot, 1.0)[:, None], 0.0)
            acc[lv]["once"] += share[ok].sum(axis=0)
            acc[lv]["raw"] += P[ok].sum(axis=0)
            acc[lv]["n_used"] += int(ok.sum())
    for lv in LEVELS:
        n = max(acc[lv]["n_used"], 1)
        acc[lv]["once"] = acc[lv]["once"] / n
        acc[lv]["raw"] = acc[lv]["raw"] / n
    return acc


def log_spectrum(spectrum: np.ndarray, level: str, dc: int) -> np.ndarray:
    """page_fourier.one_cut's log spectrum for the between-recordings test: the f = 0 bin dropped under level removed."""
    L = spectrum.copy()
    if level == "removed":
        L = np.delete(L, dc)
    return np.log(L / (L.sum() + 1e-300) + 1e-300)


def accumulator_check(rows: dict, phase_factor: float, nperseg: int, chunk: int, acc0: dict) -> dict:
    """The pass above under the unshuffled order against page_fourier.recording_spectra itself, the four variants."""
    worst, per = 0.0, {}
    for m, w, lv in VARIANT_LIST:
        ref = PF.recording_spectra(rows, phase_factor, nperseg, chunk, False, w, lv)
        dev = float(np.max(np.abs(ref["spectrum"] - acc0[lv][w]))) if ref["n_used"] else float(acc0[lv]["n_used"] != 0)
        per[f"move{m}"] = {"max_abs_dev": dev, "n_used_recording_spectra": int(ref["n_used"]), "n_used_pass": int(acc0[lv]["n_used"])}
        worst = max(worst, dev, float(abs(ref["n_used"] - acc0[lv]["n_used"])))
    return {"max_abs_dev": worst, "ok": bool(worst <= ACC_TOL), "tolerance": ACC_TOL, "per_variant": per}


# ---------------------------------------------------------------------------------------------
# the unshuffled results of page_fourier, read as they stand
# ---------------------------------------------------------------------------------------------
def pagefourier_result(pf_out: Path, move: int, step_name: str, cut: int) -> dict:
    d = PF.variant_dir(pf_out, move, step_name) / f"cut{cut}"
    sj = d / "summary.json"
    res = {"source": str(sj), "status": "missing (this variant of page_fourier has not run)", "gap": None, "null_p95": None, "null_mean": None, "p_value": None,
           "within_corr": None, "between_corr": None, "n_recordings": None}
    if not sj.is_file():
        return res
    try:
        j = read_json(sj)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        res["status"] = f"unreadable: {e}"
        return res
    b = j.get("between") or {}
    if not b:
        res["status"] = f"{j.get('status') or 'no between-recordings result'}"
        return res
    res.update(status="ok", gap=b.get("within_minus_between"), null_p95=b.get("null_p95"), null_mean=b.get("null_mean"), p_value=b.get("p_value"),
               within_corr=b.get("within_corr"), between_corr=b.get("between_corr"), n_recordings=j.get("n_scored"))
    return res


def read_gaps(step_dir: Path, cut: int) -> list[dict]:
    p = step_dir / f"cut{cut}" / "gaps.csv"
    if not p.is_file():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------------------------
# the variant table and the figure (from the saved gaps and page_fourier's results)
# ---------------------------------------------------------------------------------------------
def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def variant_rows(step: int, gaps: list[dict], pf_out: Path, cut: int, recomputed: dict | None = None) -> list[dict]:
    """One row per variant of `step` from its gaps (per shuffle) and the unshuffled result of page_fourier."""
    step_name, _, phase = STEPS[step]
    rows = []
    for m, w, lv in VARIANT_LIST:
        g = [r for r in gaps if int(r["move"]) == m and int(r["step"]) == step and int(r["cut"]) == cut and r.get("shuffle", "") != "unshuffled"]
        vals = np.array([_f(r["gap"]) for r in g if _f(r["gap"]) is not None], dtype=float)
        p95 = np.array([_f(r["null_p95"]) for r in g if _f(r["null_p95"]) is not None], dtype=float)
        pv = np.array([_f(r["p_value"]) for r in g if _f(r["p_value"]) is not None], dtype=float)
        un = pagefourier_result(pf_out, m, step_name, cut)
        n_rec = int(g[0]["n_recordings"]) if g else None
        comparable = (un["gap"] is not None and n_rec is not None and un["n_recordings"] == n_rec)      # page_fourier's gap is over its own recording set (a --only run is not comparable)
        n_at_or_above = int(np.sum(vals >= un["gap"])) if (comparable and vals.size) else None
        status = un["status"] if (un["gap"] is None or comparable) else f"ok, not comparable: page_fourier's result is over {un['n_recordings']} recordings, this run over {n_rec}"
        rc = (recomputed or {}).get(f"move{m}") or {}
        if not rc:                                         # a refresh: the recomputed unshuffled gap stands in gaps.csv (shuffle "unshuffled") when --check-unshuffled ran
            gu = [r for r in gaps if int(r["move"]) == m and int(r["step"]) == step and int(r["cut"]) == cut and r.get("shuffle") == "unshuffled"]
            if gu and _f(gu[0]["gap"]) is not None:
                rg = _f(gu[0]["gap"])
                rc = {"gap": rg, "matches": (abs(rg - float(un["gap"])) <= 1e-9) if comparable else None}
        rows.append({"move": m, "weighting": w, "level": lv, "step": step, "phase": phase, "cut": cut, "label": f"m{m} s{step}",
                     "n_recordings": n_rec, "n_time_shuffles": int(vals.size),
                     "unshuffled_status": status, "unshuffled_gap": un["gap"], "unshuffled_null_p95": un["null_p95"], "unshuffled_p": un["p_value"],
                     "shuffled_gap_mean": float(vals.mean()) if vals.size else None, "shuffled_gap_std": float(vals.std(ddof=1)) if vals.size > 1 else (0.0 if vals.size else None),
                     "shuffled_gap_min": float(vals.min()) if vals.size else None, "shuffled_gap_max": float(vals.max()) if vals.size else None,
                     "shuffled_null_p95_mean": float(p95.mean()) if p95.size else None, "shuffled_p_min": float(pv.min()) if pv.size else None,
                     "shuffled_p_max": float(pv.max()) if pv.size else None, "n_shuffles_at_or_above_unshuffled": n_at_or_above,
                     "shuffled_gaps": vals.tolist(), "unshuffled_source": un["source"],
                     "recomputed_gap": rc.get("gap"), "recomputed_matches_pagefourier": rc.get("matches")})
    return rows


SUMMARY_COLS = ["move", "weighting", "level", "step", "phase", "cut", "label", "n_recordings", "n_time_shuffles", "unshuffled_status", "unshuffled_gap", "unshuffled_null_p95",
                "unshuffled_p", "shuffled_gap_mean", "shuffled_gap_std", "shuffled_gap_min", "shuffled_gap_max", "shuffled_null_p95_mean", "shuffled_p_min", "shuffled_p_max",
                "n_shuffles_at_or_above_unshuffled", "recomputed_gap", "recomputed_matches_pagefourier", "unshuffled_source"]


def gaps_svg(panels: list[tuple[str, int, list[dict]]], title: str) -> str:
    """Per panel (a cut): the variants side by side; per variant the unshuffled gap as a bar (blue step 1, orange step 2),
    the shuffled gaps as a black dot (the mean) with whiskers (min to max) and a thick grey bar (mean +- std), the
    unshuffled label null's 95th percentile as a red tick; a variant page_fourier has not run reads "missing"."""
    n_var = max((len(rows) for _, _, rows in panels), default=1)
    SLOT = 88                                              # per variant: the bar (24 wide), the dot and whiskers beside it, the labels under them
    pw, ph = 60 + n_var * SLOT, 170
    W_, H_ = 40 + pw, 70 + len(panels) * (ph + 90)
    out = svg_open(W_, H_, title, "bar: the unshuffled within-minus-between gap (page_fourier); dot and whiskers: the time-shuffled gaps (mean, min to max; thick: mean +- std); "
                               "red tick: the unshuffled label null's 95th percentile")
    vals = []
    for _, _, rows in panels:
        for r in rows:
            vals += [v for v in (r["unshuffled_gap"], r["unshuffled_null_p95"], r["shuffled_gap_min"], r["shuffled_gap_max"]) if v is not None]
    lo, hi = (min(-0.05, min(vals)), max(0.2, max(vals))) if vals else (-0.05, 0.2)
    for ci, (cut_name, cut, rows) in enumerate(panels):
        px, py = 60, 60 + ci * (ph + 90)
        Y = lambda v: py + ph - ph * (v - lo) / (hi - lo)          # noqa: E731
        out.append(f'<text x="{px}" y="{py - 6}" font-weight="bold">cut of {cut} pairs ({cut_name})</text>')
        out.append(f'<rect x="{px}" y="{py}" width="{n_var * SLOT}" height="{ph}" fill="none" stroke="#ccc"/>')
        out.append(f'<line x1="{px}" y1="{Y(0):.1f}" x2="{px + n_var * SLOT}" y2="{Y(0):.1f}" stroke="#888"/>')
        for t in (lo, 0.0, hi):
            out.append(f'<text x="{px - 4}" y="{Y(t) + 3:.1f}" text-anchor="end" font-size="8">{t:.2f}</text>')
        for vi, r in enumerate(rows):
            bx = px + vi * SLOT + 14                       # the bar's left edge; the slot's centre is bx + 30
            mid = bx + 30
            out.append(f'<text x="{mid}" y="{py + ph + 12}" text-anchor="middle" font-size="8">{r["label"]}</text>')
            out.append(f'<text x="{mid}" y="{py + ph + 22}" text-anchor="middle" font-size="7" fill="#555">{r["weighting"]}, {r["level"]}</text>')
            colour = "#2b5d8a" if int(r["step"]) == 1 else "#d9822b"
            if r["unshuffled_gap"] is None:
                out.append(f'<text x="{bx + 12}" y="{Y(0) - 6:.1f}" text-anchor="middle" font-size="7" fill="#999">missing</text>')
            else:
                g = float(r["unshuffled_gap"])
                out.append(f'<rect x="{bx}" y="{min(Y(g), Y(0)):.1f}" width="24" height="{abs(Y(g) - Y(0)):.1f}" fill="{colour}" fill-opacity="0.85"/>')
                if r["unshuffled_null_p95"] is not None:
                    out.append(f'<line x1="{bx - 3}" y1="{Y(float(r["unshuffled_null_p95"])):.1f}" x2="{bx + 27}" y2="{Y(float(r["unshuffled_null_p95"])):.1f}" stroke="#b03a2e" stroke-width="1.5"/>')
            if r["shuffled_gap_mean"] is not None:
                cx = bx + 44
                mn, mx, mu, sd = float(r["shuffled_gap_min"]), float(r["shuffled_gap_max"]), float(r["shuffled_gap_mean"]), float(r["shuffled_gap_std"] or 0.0)
                out.append(f'<line x1="{cx}" y1="{Y(mn):.1f}" x2="{cx}" y2="{Y(mx):.1f}" stroke="#333" stroke-width="1"/>')
                out.append(f'<line x1="{cx}" y1="{Y(mu - sd):.1f}" x2="{cx}" y2="{Y(mu + sd):.1f}" stroke="#999" stroke-width="4"/>')
                out.append(f'<circle cx="{cx}" cy="{Y(mu):.1f}" r="2.5" fill="#111"/>')
                out.append(f'<text x="{mid}" y="{py + ph + 32}" text-anchor="middle" font-size="7">shuffled {mu:.3f} +- {sd:.3f}</text>')
            if r["unshuffled_gap"] is not None:
                out.append(f'<text x="{mid}" y="{py + ph + 41}" text-anchor="middle" font-size="7">unshuffled {float(r["unshuffled_gap"]):.3f}</text>')
            if r["n_shuffles_at_or_above_unshuffled"] is not None:
                out.append(f'<text x="{mid}" y="{py + ph + 50}" text-anchor="middle" font-size="7" fill="#555">{r["n_shuffles_at_or_above_unshuffled"]} of {r["n_time_shuffles"]} reach it</text>')
    out.append(f'<text x="60" y="{H_ - 8}" fill="#555">s1 (blue): phi = 2 theta; s2 (orange): phi = theta; m13 once/removed, m14 raw/kept, m15 once/kept, m16 raw/removed; '
               f'{SHUFFLE_RULE[:110]}...</text></svg>')
    return "\n".join(out)


def collect(out: Path, step: int, cut_name: str, cut: int, pf_out: Path, recomputed: dict | None = None, quiet: bool = False) -> list[dict]:
    """summary.csv, the variant table in summary.json and the cut's figure, from the saved gaps.csv of this step (and
    of the other step, for the eight-variant figure) and page_fourier's results as they stand."""
    step_name = STEPS[step][0]
    other = 2 if step == 1 else 1
    d = out / step_name / f"cut{cut}"
    mine = variant_rows(step, read_gaps(out / step_name, cut), pf_out, cut, recomputed)
    write_csv(d / "summary.csv", SUMMARY_COLS, [[r[c] for c in SUMMARY_COLS] for r in mine])
    rows_fig = list(mine)
    og = read_gaps(out / STEPS[other][0], cut)
    if og:
        rows_fig += variant_rows(other, og, pf_out, cut)
        rows_fig.sort(key=lambda r: (r["move"], r["step"]))
    (d / "gaps_unshuffled_vs_shuffled.svg").write_text(gaps_svg([(cut_name, cut, rows_fig)], f"The time-shuffled control: the gap unshuffled against shuffled, cut of {cut} pairs"))
    write_csv(d / "gaps_unshuffled_vs_shuffled.csv", SUMMARY_COLS, [[r[c] for c in SUMMARY_COLS] for r in rows_fig])
    sj = d / "summary.json"
    if sj.is_file():
        try:
            j = read_json(sj)
        except (OSError, ValueError, json.JSONDecodeError):
            j = {}
        j["variants"] = mine
        j["variants_collected_at"] = now_iso()
        j["pagefourier_out"] = str(pf_out)
        write_json(sj, j)
    if not quiet:
        for r in mine:
            print(f"[timeshuffle] cut {cut} {r['label']} ({r['weighting']}, {r['level']}): unshuffled gap {fmt(r['unshuffled_gap'])} (null p95 {fmt(r['unshuffled_null_p95'])}; "
                  f"{r['unshuffled_status'] if r['unshuffled_status'] != 'ok' else 'page_fourier'}); shuffled {fmt(r['shuffled_gap_mean'])} +- {fmt(r['shuffled_gap_std'])} "
                  f"[{fmt(r['shuffled_gap_min'])}, {fmt(r['shuffled_gap_max'])}] over {r['n_time_shuffles']} shuffles"
                  + (f"; {r['n_shuffles_at_or_above_unshuffled']} reach the unshuffled gap" if r["n_shuffles_at_or_above_unshuffled"] is not None else "")
                  + (f"; recomputed unshuffled {fmt(r['recomputed_gap'])} ({'matches page_fourier' if r['recomputed_matches_pagefourier'] else ('DIFFERS from page_fourier' if r['recomputed_matches_pagefourier'] is False else 'no comparison: a different recording set')})"
                     if r.get("recomputed_gap") is not None else ""),
                  flush=True)
    return rows_fig


def both_cuts_figure(out: Path, step: int, cuts: dict, pf_out: Path) -> None:
    step_name = STEPS[step][0]
    other = 2 if step == 1 else 1
    panels = []
    for cut_name, cut in cuts.items():
        rows = variant_rows(step, read_gaps(out / step_name, cut), pf_out, cut)
        og = read_gaps(out / STEPS[other][0], cut)
        if og:
            rows += variant_rows(other, og, pf_out, cut)
            rows.sort(key=lambda r: (r["move"], r["step"]))
        panels.append((cut_name, cut, rows))
    (out / step_name / "gaps_both_cuts.svg").write_text(gaps_svg(panels, "The time-shuffled control: the gap unshuffled against shuffled, both cuts"))
    write_csv(out / step_name / "gaps_both_cuts.csv", SUMMARY_COLS, [[r[c] for c in SUMMARY_COLS] for _, _, rows in panels for r in rows])


# ---------------------------------------------------------------------------------------------
# one cut of one step
# ---------------------------------------------------------------------------------------------
def one_cut(step_dir: Path, cut_name: str, cut: int, step: int, runs: list[dict], ex_recs: dict, run: Path, pf_out: Path, *, chunk: int, n_time: int, B: int,
            n_shuffles: int, seed: int, check_unshuffled: bool) -> dict:
    step_name, phase_factor, phase = STEPS[step]
    d = step_dir / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "spectra").mkdir(exist_ok=True)
    t_cut = time.time()
    nperseg, f_axis, dc = None, None, None
    rec_rows, id_rows, order_rows = [], [], []
    labels, scored = [], []
    logs = {(m, k): [] for m, _, _ in VARIANT_LIST for k in range(n_time)}      # (move, shuffle) -> [log spectrum per scored recording]
    logs_un = {m: [] for m, _, _ in VARIANT_LIST}
    acc_check = None
    for r in runs:
        t0 = time.time()
        cid = r["cell_id"]
        rec = ex_recs[cid]
        rows = PF.load_store_rows(rec, cut)
        ic = PF.identity_check(r, rows, cut)
        id_rows.append([cid, r["group"], cut, ic["n_pairs_series"], ic["n_pairs_store"], ic["ok"], ic["max_rel_dev"], ic["first_difference"] or ""])
        if not ic["ok"]:
            write_csv(d / "identity_check.csv", ["cell_id", "group", "cut", "n_pairs_series", "n_pairs_store", "ok", "max_rel_dev", "first_difference"], id_rows)
            raise Stop(EXIT_ERROR, f"stopped: the identity check failed for {cid} at cut {cut}: {ic['first_difference']} (largest relative deviation {ic['max_rel_dev']:.3g}); nothing scored")
        if rows["T"] < 4:
            rec_rows.append([cid, r["group"], r["kernel"], r["rep"], rows["T"], None, 0, 0, False, 0, None, None, None, None, None, True, ic["max_rel_dev"], round(time.time() - t0, 1),
                             f"not run: {rows['T']} pairs after the cut"])
            print(f"[timeshuffle] cut {cut} {cid:<32} skipped: {rows['T']} pairs after the cut", flush=True)
            continue
        if nperseg is None:                                # page_fourier's rule: one frequency grid, the shortest series after the cut bounds the segment
            nperseg = max(4, min(PF.NPERSEG, min(int(cut_series(x, cut)["pair"].size) for x in runs)))
            f_axis = PF.freqs(nperseg)
            dc = int(np.flatnonzero(np.isclose(f_axis, 0.0))[0])
        prep = prepare(rows, phase_factor)
        spec = np.full((len(VARIANT_LIST), n_time, nperseg), np.nan)
        n_used = {lv: [] for lv in LEVELS}
        perms = []
        bo = None
        for k in range(n_time):
            rng = np.random.default_rng([int(seed), int(cut), int(k), cell_seed(cid)])
            bo = block_order(rows["T"], B, rng)
            acc = spectra_pass(prep, bo["new_col"][rows["col"]], nperseg, chunk)
            for vi, (m, w, lv) in enumerate(VARIANT_LIST):
                spec[vi, k] = acc[lv][w]
                logs[(m, k)].append(log_spectrum(acc[lv][w], lv, dc))
            for lv in LEVELS:
                n_used[lv].append(acc[lv]["n_used"])
            perms.append(bo["perm"])
            order_rows.append([cid, k, bo["n_full"], bo["tail"], bo["shuffled"], " ".join(str(int(b)) for b in bo["perm"].tolist())])
        spec_un = None
        if check_unshuffled or acc_check is None:
            acc0 = spectra_pass(prep, rows["col"], nperseg, chunk)
            if acc_check is None:                          # the first scored recording: the pass against recording_spectra itself
                acc_check = {"cell_id": cid, **accumulator_check(rows, phase_factor, nperseg, chunk, acc0)}
                if not acc_check["ok"]:
                    raise Stop(EXIT_ERROR, f"stopped: this module's spectrum pass does not reproduce page_fourier.recording_spectra on {cid} at cut {cut} "
                                           f"(largest deviation {acc_check['max_abs_dev']:.3g}, tolerance {ACC_TOL:g}); nothing scored")
                print(f"[timeshuffle] cut {cut} accumulator check on {cid}: the pass reproduces recording_spectra's spectra (largest deviation {acc_check['max_abs_dev']:.2e})", flush=True)
            if check_unshuffled:
                spec_un = np.array([acc0[lv][w] for _, w, lv in VARIANT_LIST])
                for m, w, lv in VARIANT_LIST:
                    logs_un[m].append(log_spectrum(acc0[lv][w], lv, dc))
        rng_used = {lv: f"{min(n_used[lv])}..{max(n_used[lv])}" for lv in LEVELS}          # per level, over the shuffles (see the header: level kept can vary with the order)
        np.savez_compressed(d / "spectra" / f"{cid}.npz", spectrum=spec, f=f_axis, variants=np.array([f"move{m}" for m, _, _ in VARIANT_LIST]),
                            weighting=np.array([w for _, w, _ in VARIANT_LIST]), level=np.array([lv for _, _, lv in VARIANT_LIST]),
                            n_pages_with_power=np.array([[n_used[lv][k] for k in range(n_time)] for lv in LEVELS], dtype=np.int64), levels=np.array(LEVELS),
                            block_orders=np.array(perms, dtype=np.int64), n_full_blocks=np.int64(bo["n_full"]), tail_pairs=np.int64(bo["tail"]), block_pairs=np.int64(B),
                            unshuffled_spectrum=(spec_un if spec_un is not None else np.zeros(0)), cell_id=np.array(cid), cut=np.int64(cut), phase_factor=np.float64(phase_factor))
        labels.append(r["group"]); scored.append(cid)
        rec_rows.append([cid, r["group"], r["kernel"], r["rep"], rows["T"], nperseg, bo["n_full"], bo["tail"], bo["shuffled"], prep["n_pages"], n_used["removed"][0], n_used["kept"][0],
                         rng_used["removed"], rng_used["kept"], n_time, True, ic["max_rel_dev"], round(time.time() - t0, 1), "ok" if bo["shuffled"] else "ok (fewer than two full blocks: the order kept)"])
        print(f"[timeshuffle] cut {cut} {cid:<32} identity ok; {prep['n_pages']} pages, {bo['n_full']} blocks of {B} + tail {bo['tail']}; pages with power over the shuffles "
              f"{rng_used['removed']} (removed) / {rng_used['kept']} (kept); {n_time} shuffles x 2 Welch passes; {time.time() - t0:.1f} s", flush=True)
    write_csv(d / "identity_check.csv", ["cell_id", "group", "cut", "n_pairs_series", "n_pairs_store", "ok", "max_rel_dev", "first_difference"], id_rows)
    write_csv(d / "recordings.csv", ["cell_id", "group", "kernel", "rep", "n_pairs_cut", "nperseg", "n_full_blocks", "tail_pairs", "shuffled", "n_pages", "n_pages_with_power_removed_shuffle0",
                                     "n_pages_with_power_kept_shuffle0", "n_pages_with_power_removed_over_shuffles", "n_pages_with_power_kept_over_shuffles", "n_time_shuffles", "identity_ok",
                                     "identity_max_rel_dev", "time_s", "status"], rec_rows)
    write_csv(d / "block_orders.csv", ["cell_id", "shuffle", "n_full_blocks", "tail_pairs", "shuffled", "order_old_blocks_in_new_order"], order_rows)
    times = [r[17] for r in rec_rows if r[18].startswith("ok")]
    summary = {"schema": "plan12.time_shuffle_cut.v1", "citation": CITATION, "cut": int(cut), "cut_name": cut_name, "step": step, "phase": phase, "phase_factor": phase_factor,
               "nperseg": nperseg, "n_recordings": len(rec_rows), "n_scored": len(scored), "block_pairs": B, "n_time_shuffles": n_time, "shuffle_rule": SHUFFLE_RULE,
               "label_null": {"n_shuffles": n_shuffles, "seed": seed, "rule": "page_fourier.within_between with a fresh default_rng(seed) before every call"},
               "variants_of_this_step": [{"move": m, "weighting": w, "level": lv} for m, w, lv in VARIANT_LIST],
               "identity_all_ok": all(r[5] for r in id_rows), "identity_max_rel_dev": max((r[6] for r in id_rows), default=0.0), "accumulator_check": acc_check,
               "time_per_recording_s": {"mean": float(np.mean(times)) if times else None, "max": max(times, default=None), "sum": float(np.sum(times)) if times else None},
               "check_unshuffled": bool(check_unshuffled)}
    if not scored:
        summary["status"] = "not run: no recording with enough pairs after the cut"
        write_json(d / "summary.json", summary)
        return summary
    # the between-recordings test per variant and shuffle, page_fourier's own function and null
    labs = np.array(labels)
    gap_rows = []
    recomputed = {}
    t_b = time.time()
    for m, w, lv in VARIANT_LIST:
        for k in range(n_time):
            wb = PF.within_between(np.array(logs[(m, k)]), labs, np.random.default_rng(int(seed)), n_shuffles)
            gap_rows.append([m, w, lv, step, phase, cut, k, wb["within_corr"], wb["between_corr"], wb["within_minus_between"], wb["null_mean"], wb["null_p95"], wb["p_value"],
                             wb["n_shuffles"], len(scored)])
        if check_unshuffled:
            wb = PF.within_between(np.array(logs_un[m]), labs, np.random.default_rng(int(seed)), n_shuffles)
            un = pagefourier_result(pf_out, m, step_name, cut)
            same_set = (un["n_recordings"] == len(scored))         # page_fourier's gap is over its own recording set; a --only selection is not comparable
            match = (abs(float(un["gap"]) - wb["within_minus_between"]) <= 1e-9) if (un["gap"] is not None and same_set) else None
            recomputed[f"move{m}"] = {"gap": wb["within_minus_between"], "null_p95": wb["null_p95"], "p_value": wb["p_value"], "pagefourier_gap": un["gap"],
                                      "pagefourier_n_recordings": un["n_recordings"], "matches": match,
                                      "note": None if same_set else "no comparison: page_fourier's result is over a different recording set"}
            gap_rows.append([m, w, lv, step, phase, cut, "unshuffled", wb["within_corr"], wb["between_corr"], wb["within_minus_between"], wb["null_mean"], wb["null_p95"],
                             wb["p_value"], wb["n_shuffles"], len(scored)])
    write_csv(d / "gaps.csv", ["move", "weighting", "level", "step", "phase", "cut", "shuffle", "within_corr", "between_corr", "gap", "null_mean", "null_p95", "p_value",
                               "n_label_shuffles", "n_recordings"], gap_rows)
    summary.update({"status": "ok", "between_test_time_s": round(time.time() - t_b, 1), "recomputed_unshuffled": recomputed or None, "elapsed_s": round(time.time() - t_cut, 1),
                    "files": sorted(p.name for p in d.iterdir() if p.is_file())})
    write_json(d / "summary.json", summary)
    summary["variants"] = [{k: v for k, v in r.items() if k != "shuffled_gaps"} for r in collect(step_dir.parent, step, cut_name, cut, pf_out, recomputed)
                           if int(r["step"]) == step]
    print(f"[timeshuffle] cut {cut} ({cut_name}): {len(scored)} recordings, {n_time} shuffles, {len(VARIANT_LIST)} variants; {summary['elapsed_s']} s "
          f"(recordings {summary['time_per_recording_s']['sum']:.0f} s, the between tests {summary['between_test_time_s']} s)", flush=True)
    return summary


# ---------------------------------------------------------------------------------------------
# the step: the output folder, the locks, the record (page_fourier's pattern)
# ---------------------------------------------------------------------------------------------
def default_out(run: Path) -> Path:
    return run.parent / (run.name + OUT_SUFFIX)


def resolve_out(run: Path, out_arg: str | None) -> Path:
    out = Path(os.path.expanduser(out_arg)).resolve() if out_arg else default_out(run)
    if out == run or run in out.parents:
        raise Stop(EXIT_MISSING, f"refused: --out {out} lies inside the run folder {run}, which is read only here; the default is {default_out(run)}")
    return out


def resolve_pf_out(run: Path, arg: str | None) -> Path:
    return Path(os.path.expanduser(arg)).resolve() if arg else PF.default_out(run)


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
            raise Stop(EXIT_REFUSED, f"refused: another time_shuffle run is writing {out} ({lock}: pid {pid}, started {old.get('started_at')}); one writer per output folder")
        print(f"[timeshuffle] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
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
    return {"schema": "plan12.time_shuffle.record.v1", "citation": CITATION, "package_version": __version__, "entries": []}


def save_record(out: Path, rec: dict) -> None:
    write_json(out / RECORD_NAME, rec)


def last_done(rec: dict, key: str) -> dict | None:
    for e in reversed(rec.get("entries") or []):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def code_identity() -> dict:
    fp = code_fingerprint("time_shuffle")
    files = toolkit_fingerprint()["files"]
    return {"fingerprint": fp["sha256"], "modules": fp["modules"], "sha256": {f"{m}.py": files.get(f"{m}.py") for m in fp["modules"]}}


def plan_blocks(runs: list[dict], cut: int, B: int) -> dict:
    """The dry run's counts from the series alone: full blocks and tail per recording."""
    n_full = [int(cut_series(r, cut)["pair"].size) // B for r in runs]
    return {"recordings": len(runs), "full_blocks_min": min(n_full, default=0), "full_blocks_max": max(n_full, default=0),
            "recordings_with_fewer_than_two_blocks": sum(1 for n in n_full if n < 2)}


def _finish(out: Path, entry: dict, out_lock: Path) -> None:
    if entry["status"] == "running":
        entry.update(status="failed: stopped before the end", finished_at=now_iso())
    rec = load_record(out)
    rec["entries"] = [e for e in rec["entries"] if not (e.get("key") == entry["key"] and e.get("started_at") == entry["started_at"])] + [entry]
    save_record(out, rec)
    release_out_lock(out_lock)


def run_step(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0 and 1 done)")
    step = int(o.step)
    step_name, phase_factor, phase = STEPS[step]
    n_time, B = int(o.n_time_shuffles), int(o.block_pairs)
    if n_time < 1 or B < 2:
        raise Stop(EXIT_MISSING, f"--n-time-shuffles must be 1 or more (given {n_time}) and --block-pairs 2 or more (given {B})")
    out = resolve_out(run, o.out)
    pf_out = resolve_pf_out(run, o.pagefourier_out)
    lock = I.driver_lock_state(run)
    if lock["alive"] and not o.dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}, started {lock['started_at']}, {lock['argv_tail']}); "
                                 "this control never competes with a running move. Run again after it ends; --dry-run shows the plan meanwhile.")
    cuts = I.parse_cuts(o.cuts, cuts_of(run))
    exj = read_json(run / "moves" / "01_extract" / "extract.json")
    ex_recs = exj.get("recordings") or {}
    only = [s.strip() for s in str(o.only).split(",") if s.strip()] if o.only else None
    runs = PF.select_runs(run, only)
    missing_store = [r["cell_id"] for r in runs if not Path((ex_recs.get(r["cell_id"]) or {}).get("store") or "").is_file()]
    if missing_store:
        raise Stop(EXIT_MISSING, f"missing input: the L1 store of {len(missing_store)} recording(s) is not on this machine (first: {missing_store[0]}; extract.json names it)")
    n_idle = sum(1 for r in runs if r["group"] == "idle")
    present = {f"move{m} {STEPS[s][0]} cut{c}": pagefourier_result(pf_out, m, STEPS[s][0], c)["status"] == "ok" for m in sorted(VARIANTS) for s in sorted(STEPS) for c in cuts.values()}
    print(f"[timeshuffle] step {step}: {phase}; the time-shuffled control of the per-page Fourier test on {len(runs)} recordings ({len(runs) - n_idle} kernel + {n_idle} idle), "
          f"cuts {', '.join(f'{k} = {v}' for k, v in cuts.items())}; blocks of {B} pairs, {n_time} shuffles per recording (seed {o.seed}); the four variants of this step "
          f"({', '.join(f'move {m} {w}/{lv}' for m, w, lv in VARIANT_LIST)}); {o.n_shuffles} label shuffles per between test; chunks of {o.chunk_pages} pages"
          + ("; --check-unshuffled: the unshuffled order recomputed too" if o.check_unshuffled else ""))
    print(f"[timeshuffle] run {run} (read only); stores {Path(ex_recs[runs[0]['cell_id']]['store']).parent} (read only); page_fourier results {pf_out} (read only; "
          f"{sum(present.values())} of {len(present)} variant results present{': missing ' + ', '.join(k for k, v in present.items() if not v) if not all(present.values()) else ''}); "
          f"out {out}{' (exists)' if out.exists() else ' (to be created)'}")
    print(f"[timeshuffle] driver lock: " + (f"held by pid {lock['pid']} ({'alive' if lock['alive'] else 'dead'}), started {lock['started_at']}: {lock['argv_tail']}" if lock["exists"] else "absent")
          + ("; the refusal is skipped by --dry-run" if (lock["alive"] and o.dry_run) else ""))
    for name, cut in cuts.items():
        pb = plan_blocks(runs, cut, B)
        print(f"[timeshuffle] cut {cut} ({name}): {pb['recordings']} recordings, {pb['full_blocks_min']} to {pb['full_blocks_max']} full blocks of {B} pairs each"
              + (f"; {pb['recordings_with_fewer_than_two_blocks']} with fewer than two blocks (order kept)" if pb["recordings_with_fewer_than_two_blocks"] else ""))
    step_dir = out / step_name
    if o.dry_run:
        print(f"[timeshuffle] dry run: would write {step_dir}/cut<C>/ (recordings.csv, block_orders.csv, spectra/<cell>.npz, gaps.csv, summary.csv, identity_check.csv, "
              f"gaps_unshuffled_vs_shuffled.svg/.csv, summary.json), {step_dir / 'gaps_both_cuts.svg'}, {step_dir / 'step.json'} and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    code = code_identity()
    entry = {"key": step_name, "step": step, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out), "pagefourier_out": str(pf_out), "cuts": cuts, "phase_factor": phase_factor, "phase": phase, "nperseg": PF.NPERSEG,
                        "block_pairs": B, "n_time_shuffles": n_time, "n_shuffles": int(o.n_shuffles), "seed": int(o.seed), "chunk_pages": int(o.chunk_pages), "only": only,
                        "check_unshuffled": bool(o.check_unshuffled), "rtol": PF.RTOL, "n_recordings": len(runs), "force": bool(o.force)},
             "inputs_sha256": PF.inputs_identity(run, runs, ex_recs, cuts),
             "code": {"time_shuffle.py": code["sha256"].get("time_shuffle.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
             "outputs": []}
    sig = {k: v for k, v in entry["params"].items() if k not in ("force", "chunk_pages", "pagefourier_out")}
    rec = load_record(out)
    prev = last_done(rec, step_name)
    outputs_exist = (step_dir / "step.json").is_file() and all((step_dir / f"cut{c}" / "summary.json").is_file() for c in cuts.values())
    try:
        if prev is not None and not o.force and outputs_exist:
            why = None
            if prev.get("inputs_sha256") != entry["inputs_sha256"]:
                why = "an input changed"
            elif {k: v for k, v in (prev.get("params") or {}).items() if k not in ("force", "chunk_pages", "pagefourier_out")} != sig:
                why = "the parameters changed"
            elif (prev.get("code") or {}).get("fingerprint") != code["fingerprint"]:
                why = "the code changed"
            if why is None:
                entry.update(status="skipped: outputs exist and inputs, parameters and code unchanged", exit_code=0, finished_at=now_iso(),
                             made_by={"finished_at": prev.get("finished_at"), "code_fingerprint": (prev.get("code") or {}).get("fingerprint")}, outputs=prev.get("outputs") or [])
                rec["entries"].append(entry)
                save_record(out, rec)
                print(f"[timeshuffle] skipped: {step_dir} stands as made on {prev.get('finished_at')} (inputs, parameters and code unchanged; --force re-runs; "
                      f"`summary` refreshes the unshuffled columns)")
                return EXIT_OK
            print(f"[timeshuffle] re-running: {why} since {prev.get('finished_at')}")
        rec["entries"].append(entry)
        save_record(out, rec)
        step_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        results = [one_cut(step_dir, name, cut, step, runs, ex_recs, run, pf_out, chunk=int(o.chunk_pages), n_time=n_time, B=B, n_shuffles=int(o.n_shuffles), seed=int(o.seed),
                           check_unshuffled=bool(o.check_unshuffled)) for name, cut in cuts.items()]
        both_cuts_figure(out, step, cuts, pf_out)
        step_rec = {"schema": "plan12.time_shuffle_step.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv, "step": step,
                    "phase": phase, "phase_factor": phase_factor, "question": __doc__.split("THE QUESTION.", 1)[1].split("THE STEPS", 1)[0].strip(), "shuffle_rule": SHUFFLE_RULE,
                    "params": entry["params"], "recordings": [r["cell_id"] for r in runs], "code": entry["code"], "elapsed_s": round(time.time() - t0, 1), "results": results}
        write_json(step_dir / "step.json", step_rec)
        entry["outputs"] = sorted(str(p.relative_to(out)) for p in step_dir.rglob("*") if p.is_file())
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=step_rec["elapsed_s"])
        print(f"[timeshuffle] step {step} done in {step_rec['elapsed_s']} s: {step_dir}")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        code_ = getattr(exc, "code", None)
        entry.update(status=f"failed: {type(exc).__name__}" + (f" (exit {code_})" if isinstance(code_, int) else ""), exit_code=code_ if isinstance(code_, int) else EXIT_ERROR,
                     finished_at=now_iso(), error=str(exc)[:500])
        raise
    finally:
        _finish(out, entry, out_lock)


def run_summary(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "params.json").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'params.json'}")
    out = resolve_out(run, o.out)
    pf_out = resolve_pf_out(run, o.pagefourier_out)
    lock = I.driver_lock_state(run)
    if lock["alive"] and not o.dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}); run again after it ends")
    cuts = cuts_of(run)
    todo = [(s, name, cut) for s in sorted(STEPS) for name, cut in cuts.items() if (out / STEPS[s][0] / f"cut{cut}" / "gaps.csv").is_file()]
    if o.dry_run:
        print(f"[timeshuffle] summary dry run: {len(todo)} step-and-cut results present in {out} ({', '.join(f'{STEPS[s][0]} cut{c}' for s, _, c in todo) or 'none'}); "
              f"would rewrite their summary.csv, summary.json variant tables and figures from {pf_out}; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    entry = {"key": "summary", "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out), "pagefourier_out": str(pf_out)}, "code": code_identity(), "outputs": []}
    rec = load_record(out)
    rec["entries"].append(entry); save_record(out, rec)
    try:
        if not todo:
            raise Stop(EXIT_MISSING, f"nothing to summarise: no step of the control has run in {out}")
        for s, name, cut in todo:
            collect(out, s, name, cut, pf_out)
        for s in sorted({s for s, _, _ in todo}):
            both_cuts_figure(out, s, cuts, pf_out)
        write_json(out / "summary_refresh.json", {"schema": "plan12.time_shuffle_summary.v1", "citation": CITATION, "written_at": now_iso(), "pagefourier_out": str(pf_out),
                                                  "refreshed": [f"{STEPS[s][0]}/cut{c}" for s, _, c in todo]})
        entry["outputs"] = ["summary_refresh.json"] + [f"{STEPS[s][0]}/cut{c}/summary.csv" for s, _, c in todo]
        entry.update(status="done", exit_code=0, finished_at=now_iso())
        print(f"[timeshuffle] summary refreshed for {len(todo)} step-and-cut results from {pf_out}")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        code_ = getattr(exc, "code", None)
        entry.update(status=f"failed: {type(exc).__name__}" + (f" (exit {code_})" if isinstance(code_, int) else ""), exit_code=code_ if isinstance(code_, int) else EXIT_ERROR,
                     finished_at=now_iso(), error=str(exc)[:500])
        raise
    finally:
        _finish(out, entry, out_lock)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True, help="the main run's output folder (read only here)")
    p.add_argument("--step", required=True, type=int, choices=sorted(STEPS), help="1 = phi = 2 theta (page_fourier's headline); 2 = phi = theta (its control)")
    p.add_argument("--out", default=None, help="the output folder (default: a sibling of the run, <run>_timeshuffle; never inside the run)")
    p.add_argument("--pagefourier-out", default=None, help="where page_fourier wrote its results (default <run>_pagefourier; read only)")
    p.add_argument("--cuts", default=None, help="which cuts, by name: declared,measured (default both; the values come from params.json)")
    p.add_argument("--only", default=None, help="a comma-separated list of cell ids (a test on a few recordings)")
    p.add_argument("--chunk-pages", type=int, default=DEFAULT_CHUNK, help="pages per chunk when the rows are built (memory)")
    p.add_argument("--n-time-shuffles", type=int, default=DEFAULT_TIME_SHUFFLES, help="block orders drawn per recording (default 10)")
    p.add_argument("--block-pairs", type=int, default=DEFAULT_BLOCK_PAIRS, help="pairs per block (default 64, half a Welch segment)")
    p.add_argument("--n-shuffles", type=int, default=DEFAULT_SHUFFLES, help="label shuffles of the between-recordings null (page_fourier: 1000)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="the seed of the block orders and of the label null (the run's: 20260930)")
    p.add_argument("--check-unshuffled", action="store_true", help="also recompute the unshuffled order through this module's pass and its between test, beside page_fourier's")
    p.add_argument("--force", action="store_true", help="re-run a finished step")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write nothing; skip the live-lock refusal")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.time_shuffle", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="step 1 (phi = 2 theta) or step 2 (phi = theta): the time-shuffled control, the four variants of the step, both cuts"))
    sm = sub.add_parser("summary", help="rewrite the unshuffled columns and the figures of every finished step from the page_fourier results as they stand")
    sm.add_argument("--run", required=True)
    sm.add_argument("--out", default=None, help="the sibling folder (default <run>_timeshuffle)")
    sm.add_argument("--pagefourier-out", default=None, help="where page_fourier wrote its results (default <run>_pagefourier)")
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
