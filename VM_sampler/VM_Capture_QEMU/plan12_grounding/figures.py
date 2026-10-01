#!/usr/bin/env python3
"""figures.py -- moves 3 and 4 of plan12_grounding (SPEC moves 3, 4; section 6 figures 1 and 2).

  python3 -m plan12_grounding.figures every-run --out O     (move 3)
  python3 -m plan12_grounding.figures portraits --out O     (move 4)

Move 3, every run: for each kernel and idle and each series (N_t, H_t, A_t), the runs' traces over
the pair index, overlaid, with the kernel's average (thick line); one SVG per kernel and series,
its data as CSV beside it. The gallery `index.html` follows apf_paper/previews/make_shape_gallery.py:
section 1 the averages per kernel, section 2 the whole runs (the overlays), section 3 every
minute of every run (one SVG per run, one row per minute of 93 pairs, N and H as a share of the
run's maximum on the left scale, A in radians on the right scale). Move 4, kernel portraits: every
kernel's average shape side by side with idle, per series, on one shared scale, with the data as
CSV. Both moves at both cuts (`cut16/`, `cut112/` from params.json), hand-written SVG with a
viewBox as the gallery does, and `figures.json` recording every file and the generating command.
Nothing here computes a statistic: the figures draw the series as move 1 wrote them.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import csv  # noqa: E402
import html  # noqa: E402
import os  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import now_iso, write_json  # noqa: E402
from plan12_grounding.stats import ORDER, SERIES, cut_series, cuts_of, load_runs  # noqa: E402

CITATION = "plan12_grounding/SPEC.md moves 3 and 4, section 6; the style of apf_paper/previews/make_shape_gallery.py"
MINUTE = 93                                   # pairs per minute, about 60 s of guest time at 0.645 s per pair (the gallery)
SERIES_LABEL = {"N": "N_t, changed pages per pair", "H": "H_t, bits flipped per pair", "A": "A_t, bit-weighted mean angle (rad)"}
PALETTE = ["#2b5d8a", "#d9822b", "#3a9d5d", "#b03a2e", "#7b52ab", "#8a6d3b", "#c2378d", "#4b8b9b"]
C_MEAN, C_IDLE, C_N, C_H, C_A = "#111111", "#9a9a9a", "#2b5d8a", "#8a8a8a", "#d9822b"
FONT = 'font-family="Helvetica,Arial,sans-serif" font-size="10"'


# ---------------------------------------------------------------------------------------------
# SVG helpers (the gallery's shapes: a white sheet, a framed panel, polylines, scale labels)
# ---------------------------------------------------------------------------------------------
def svg_open(width: int, height: int, title: str, sub: str = "") -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" {FONT}>',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="10" y="16" font-size="12" font-weight="bold">{html.escape(title)}</text>'
            + (f'<text x="10" y="30" fill="#555">{html.escape(sub)}</text>' if sub else "")]


def _pts(xs, ys, X, Y) -> str:
    return " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, ys) if np.isfinite(v))


def panel(out: list[str], x0: int, y0: int, w: int, h: int, xlo: float, xhi: float, ylo: float, yhi: float,
          traces: list[tuple], title: str = "", sub: str = "", xlabel: str = "") -> None:
    """A framed panel with `traces` = [(xs, ys, colour, width, opacity), ...] on the scales given."""
    if title or sub:
        out.append(f'<text x="{x0}" y="{y0 - 4}" font-weight="bold">{html.escape(title)}<tspan font-weight="normal" fill="#666"> {html.escape(sub)}</tspan></text>')
    out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
    if xhi <= xlo:
        xhi = xlo + 1.0
    if yhi <= ylo:
        yhi = ylo + 1.0
    X = lambda x: x0 + w * (x - xlo) / (xhi - xlo)            # noqa: E731
    Y = lambda v: y0 + h - h * (v - ylo) / (yhi - ylo)        # noqa: E731
    for xs, ys, col, wd, op in traces:
        pts = _pts(xs, ys, X, Y)
        if pts:
            out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wd}" stroke-opacity="{op}"/>')
    out.append(f'<text x="{x0 - 3}" y="{y0 + 8}" text-anchor="end">{_fmt(yhi)}</text>')
    out.append(f'<text x="{x0 - 3}" y="{y0 + h}" text-anchor="end">{_fmt(ylo)}</text>')
    out.append(f'<text x="{x0}" y="{y0 + h + 12}">{_fmt(xlo)}</text>')
    out.append(f'<text x="{x0 + w}" y="{y0 + h + 12}" text-anchor="end">{_fmt(xhi)}</text>')
    if xlabel:
        out.append(f'<text x="{x0 + w / 2:.0f}" y="{y0 + h + 12}" text-anchor="middle" fill="#666">{html.escape(xlabel)}</text>')


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "?"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:.0f}k"
    if abs(v) >= 100 or v == int(v):
        return f"{v:.0f}"
    return f"{v:.3g}"


def _range(vals: list[np.ndarray]) -> tuple[float, float]:
    allv = np.concatenate([np.asarray(v, dtype=np.float64) for v in vals if np.asarray(v).size]) if vals else np.zeros(1)
    allv = allv[np.isfinite(allv)]
    if not allv.size:
        return 0.0, 1.0
    lo, hi = float(allv.min()), float(allv.max())
    lo = min(0.0, lo)
    return lo, (hi * 1.06 if hi > lo else lo + 1.0)


def write_csv(path: Path, columns: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow(["" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v for v in r])


# ---------------------------------------------------------------------------------------------
# the aligned traces of a group: runs by pair index, the average over the runs present at each pair
# ---------------------------------------------------------------------------------------------
def group_traces(runs: list[dict], cut: int, series: str) -> dict:
    """{pairs, per_run: [(cell_id, xs, ys)], mean (over the runs present at each pair), n_min (the
    shortest run's last pair)}. Runs are aligned by pair number and the average uses every run
    present at a pair (the gallery cuts to the shortest run; here the CSV keeps every pair and says
    how many runs each average point has)."""
    per = []
    for r in runs:
        a = cut_series(r, cut)
        per.append((r["cell_id"], a["pair"].astype(np.int64), a[series].astype(np.float64)))
    if not per:
        return {"pairs": np.zeros(0, np.int64), "per_run": [], "mean": np.zeros(0), "count": np.zeros(0, np.int64), "n_min": 0}
    lo = min(int(p[1].min()) for p in per if p[1].size)
    hi = max(int(p[1].max()) for p in per if p[1].size)
    pairs = np.arange(lo, hi + 1)
    tot = np.zeros(pairs.size)
    cnt = np.zeros(pairs.size, dtype=np.int64)
    for _, xs, ys in per:
        idx = xs - lo
        ok = np.isfinite(ys)
        np.add.at(tot, idx[ok], ys[ok])
        np.add.at(cnt, idx[ok], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return {"pairs": pairs, "per_run": per, "mean": mean, "count": cnt, "n_min": min(int(p[1].max()) for p in per if p[1].size)}


# ---------------------------------------------------------------------------------------------
# move 3
# ---------------------------------------------------------------------------------------------
def overlay_svg(group: str, series: str, tr: dict, cut: int) -> str:
    pw, ph, ml, mt = 820, 150, 60, 44
    lo, hi = _range([ys for _, _, ys in tr["per_run"]])
    xlo, xhi = (float(tr["pairs"][0]), float(tr["pairs"][-1])) if tr["pairs"].size else (0.0, 1.0)
    out = svg_open(pw + 100, ph + 90, f"{group}: {SERIES_LABEL[series]}, every run overlaid, with the average",
                   f"{len(tr['per_run'])} runs, pairs {int(xlo)} to {int(xhi)} after a cut of {cut} pairs; thin lines the runs, the thick line the mean over the runs present at each pair")
    traces = [(xs, ys, PALETTE[i % len(PALETTE)], 0.8, 0.75) for i, (_, xs, ys) in enumerate(tr["per_run"])]
    traces.append((tr["pairs"], tr["mean"], C_MEAN, 2.0, 1.0))
    panel(out, ml, mt, pw, ph, xlo, xhi, lo, hi, traces, xlabel="pair index")
    y = mt + ph + 30
    for i, (cid, _, _) in enumerate(tr["per_run"]):
        out.append(f'<text x="{ml + (i % 4) * 205}" y="{y + 14 * (i // 4)}"><tspan fill="{PALETTE[i % len(PALETTE)]}" font-weight="bold">&#9644;</tspan> {html.escape(cid)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def average_svg(group: str, trs: dict, cut: int) -> str:
    pw, ph, ml, mt, gap = 820, 90, 60, 44, 34
    out = svg_open(pw + 100, mt + 3 * (ph + gap) + 10, f"{group}: the average of its runs, pair by pair, after a cut of {cut} pairs",
                   "each point is the mean over the runs present at that pair (the CSV beside this figure says how many)")
    for i, s in enumerate(SERIES):
        tr = trs[s]
        lo, hi = _range([tr["mean"]])
        xlo, xhi = (float(tr["pairs"][0]), float(tr["pairs"][-1])) if tr["pairs"].size else (0.0, 1.0)
        panel(out, ml, mt + i * (ph + gap), pw, ph, xlo, xhi, lo, hi, [(tr["pairs"], tr["mean"], {"N": C_N, "H": C_H, "A": C_A}[s], 1.6, 1.0)],
              title=SERIES_LABEL[s], xlabel="pair index" if i == 2 else "")
    out.append("</svg>")
    return "\n".join(out)


def minutes_svg(run: dict, cut: int) -> str:
    a = cut_series(run, cut)
    pairs, N, H, A = a["pair"], a["N"].astype(float), a["H"].astype(float), a["A"].astype(float)
    n = int(pairs.size)
    nmax, hmax = (float(np.nanmax(N)) if n else 1.0) or 1.0, (float(np.nanmax(H)) if n else 1.0) or 1.0
    rows = [(k, min(k + MINUTE, n)) for k in range(0, n, MINUTE)] or [(0, 0)]
    pw, ph, ml, mt = 820, 64, 60, 46
    label = f"run {run['rep'] + 1}" if run["group"] == "idle" else f"seed {run['seed']}"
    out = svg_open(pw + 100, mt + len(rows) * (ph + 22) + 20,
                   f"{run['group']}, {label}: {n} pairs after a cut of {cut}, one row per minute ({MINUTE} pairs)",
                   "N (blue) and H (grey) as a share of this run's maximum, left scale 0 to 1; A in radians (orange), right scale 0 to pi/2")
    for i, (k0, k1) in enumerate(rows):
        y0 = mt + i * (ph + 22)
        xs = pairs[k0:k1]
        sub = f"minute {i + 1}: pairs {int(pairs[k0]) if k1 > k0 else '-'} to {int(pairs[k1 - 1]) if k1 > k0 else '-'}" + (" (partial)" if k1 - k0 < MINUTE else "")
        xlo = float(xs[0]) if xs.size else 0.0
        xhi = xlo + MINUTE - 1
        panel(out, ml, y0, pw, ph, xlo, xhi, 0.0, 1.0,
              [(xs, H[k0:k1] / hmax, C_H, 1.0, 1.0), (xs, N[k0:k1] / nmax, C_N, 1.2, 1.0), (xs, A[k0:k1] / (np.pi / 2), C_A, 1.1, 1.0)], sub=sub)
        out.append(f'<text x="{ml + pw + 3}" y="{y0 + 8}" fill="{C_A}">1.571</text><text x="{ml + pw + 3}" y="{y0 + ph}" fill="{C_A}">0</text>')
    out.append("</svg>")
    return "\n".join(out)


def every_run(out: Path, cut_name: str, cut: int, runs: list[dict], argv: list[str], moves_dir: Path | None = None) -> dict:
    d = (moves_dir or out / "moves") / "03_every_run" / f"cut{cut}"
    for sub in ("overlay", "average", "runs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    groups = [g for g in ORDER if any(r["group"] == g for r in runs)]
    files = []
    page = ["<!doctype html><meta charset='utf-8'><title>Every run (plan12 move 3)</title>",
            "<style>body{font-family:Helvetica,Arial,sans-serif;margin:16px;background:#fff;color:#111}"
            "img{width:100%;max-width:930px;display:block;margin:4px 0 18px}h2{margin-top:28px}p.n{color:#555;font-size:13px}</style>",
            f"<h1>Every run, per kernel and idle: N_t, H_t, A_t over the pair index (cut of {cut} pairs, {cut_name})</h1>",
            "<p class='n'>Move 3 of plan12_grounding. Each figure draws the series as move 1 wrote them; its data is the CSV of the same name. "
            "Pairs are aligned by index (time since the run started). The other cut: <a href='../index.html'>index</a>.</p>",
            "<p><a href='#avg'>1. Average of the runs, per kernel</a> &middot; <a href='#whole'>2. The whole run, every run overlaid</a> &middot; <a href='#min'>3. Every minute of every run</a></p>",
            "<h1 id='avg'>1. Average of the runs, per kernel</h1><p class='n'>Runs aligned by pair number; the average at each pair is over the runs present there.</p>"]
    for g in groups:
        rs = [r for r in runs if r["group"] == g]
        trs = {s: group_traces(rs, cut, s) for s in SERIES}
        (d / "average" / f"{g}.svg").write_text(average_svg(g, trs, cut))
        cols = ["pair"] + [f"{s}_mean" for s in SERIES] + ["n_runs"]
        rows = [[int(p)] + [float(trs[s]["mean"][i]) for s in SERIES] + [int(trs["N"]["count"][i])] for i, p in enumerate(trs["N"]["pairs"])]
        write_csv(d / "average" / f"{g}.csv", cols, rows)
        files.append({"figure": f"cut{cut}/average/{g}.svg", "data": f"cut{cut}/average/{g}.csv", "group": g, "kind": "average"})
        page.append(f"<p class='n'><b>{html.escape(g)}</b>: {len(rs)} runs</p><img src='average/{g}.svg' alt='{html.escape(g)} average'>")
        for s in SERIES:
            tr = trs[s]
            (d / "overlay" / f"{g}__{s}.svg").write_text(overlay_svg(g, s, tr, cut))
            cols = ["pair"] + [cid for cid, _, _ in tr["per_run"]] + ["mean", "n_runs"]
            lo = int(tr["pairs"][0]) if tr["pairs"].size else 0
            table = {cid: dict(zip(xs.tolist(), ys.tolist())) for cid, xs, ys in tr["per_run"]}
            rows = [[int(p)] + [table[cid].get(int(p)) for cid, _, _ in tr["per_run"]] + [float(tr["mean"][i]), int(tr["count"][i])] for i, p in enumerate(tr["pairs"])]
            write_csv(d / "overlay" / f"{g}__{s}.csv", cols, rows)
            files.append({"figure": f"cut{cut}/overlay/{g}__{s}.svg", "data": f"cut{cut}/overlay/{g}__{s}.csv", "group": g, "series": s, "kind": "overlay"})
    page.append("<h1 id='whole'>2. The whole run, every run overlaid</h1><p class='n'>Thin lines the runs, the thick line the kernel's average; one panel per series.</p>")
    for g in groups:
        page.append(f"<h2>{html.escape(g)}</h2>" + "".join(f"<img src='overlay/{g}__{s}.svg' alt='{html.escape(g)} {s}'>" for s in SERIES))
    page.append("<h1 id='min'>3. Every minute of every run</h1>")
    for g in groups:
        rs = [r for r in runs if r["group"] == g]
        page.append(f"<h2>{html.escape(g)} ({len(rs)} runs)</h2>")
        for r in rs:
            (d / "runs" / f"{r['cell_id']}.svg").write_text(minutes_svg(r, cut))
            a = cut_series(r, cut)
            write_csv(d / "runs" / f"{r['cell_id']}.csv", ["pair", "N", "H", "A"], zip(a["pair"].tolist(), a["N"].tolist(), a["H"].tolist(), a["A"].tolist()))
            files.append({"figure": f"cut{cut}/runs/{r['cell_id']}.svg", "data": f"cut{cut}/runs/{r['cell_id']}.csv", "group": g, "cell_id": r["cell_id"], "kind": "minutes"})
            label = f"run {r['rep'] + 1}" if g == "idle" else f"seed {r['seed']}"
            page.append(f"<p class='n'>{html.escape(label)}: {int(a['pair'].size)} pairs after the cut</p><img src='runs/{r['cell_id']}.svg' alt='{html.escape(r['cell_id'])}'>")
    (d / "index.html").write_text("\n".join(page))
    return {"cut": cut, "cut_name": cut_name, "n_runs": len(runs), "groups": groups, "files": files, "index": f"cut{cut}/index.html"}


# ---------------------------------------------------------------------------------------------
# move 4
# ---------------------------------------------------------------------------------------------
def portrait_svg(series: str, per_group: dict, idle: dict | None, cut: int) -> str:
    groups = [g for g in ORDER if g in per_group]
    cols, pw, ph, ml, mt = 4, 250, 130, 46, 48
    nrow = (len(groups) + cols - 1) // cols
    lo, hi = _range([per_group[g]["mean"] for g in groups] + ([idle["mean"]] if idle else []))
    xhi = max(float(per_group[g]["pairs"][-1]) for g in groups if per_group[g]["pairs"].size)
    xlo = min(float(per_group[g]["pairs"][0]) for g in groups if per_group[g]["pairs"].size)
    out = svg_open(cols * pw + 30, mt + nrow * ph + 40, f"Kernel portraits: {SERIES_LABEL[series]}, each kernel's average shape beside idle's, one scale, cut of {cut} pairs",
                   "each panel: the kernel's average over its runs (colour) with idle's average underneath (grey); pair index on the horizontal, the same vertical scale everywhere")
    for i, g in enumerate(groups):
        cx, cy = 10 + (i % cols) * pw, mt + (i // cols) * ph
        tr = per_group[g]
        traces = []
        if idle and g != "idle":
            traces.append((idle["pairs"], idle["mean"], C_IDLE, 1.2, 1.0))
        traces.append((tr["pairs"], tr["mean"], {"N": C_N, "H": C_H if g != "idle" else C_IDLE, "A": C_A}[series] if g != "idle" else C_IDLE, 1.5, 1.0))
        panel(out, cx + ml, cy + 14, pw - ml - 20, ph - 40, xlo, xhi, lo, hi, traces, title=g)
    out.append(f'<text x="10" y="{mt + nrow * ph + 26}">Horizontal: pair index {int(xlo)} to {int(xhi)}; vertical: {_fmt(lo)} to {_fmt(hi)} on every panel.</text></svg>')
    return "\n".join(out)


def portraits(out: Path, cut_name: str, cut: int, runs: list[dict], argv: list[str], moves_dir: Path | None = None) -> dict:
    d = (moves_dir or out / "moves") / "04_portraits" / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    groups = [g for g in ORDER if any(r["group"] == g for r in runs)]
    files = []
    for s in SERIES:
        per = {g: group_traces([r for r in runs if r["group"] == g], cut, s) for g in groups}
        idle = per.get("idle")
        (d / f"portrait_{s}.svg").write_text(portrait_svg(s, per, idle, cut))
        lo = min(int(per[g]["pairs"][0]) for g in groups if per[g]["pairs"].size)
        hi = max(int(per[g]["pairs"][-1]) for g in groups if per[g]["pairs"].size)
        pairs = np.arange(lo, hi + 1)
        cols = ["pair"] + [f"{g}_mean" for g in groups]
        table = {g: dict(zip(per[g]["pairs"].tolist(), per[g]["mean"].tolist())) for g in groups}
        write_csv(d / f"portrait_{s}.csv", cols, [[int(p)] + [table[g].get(int(p)) for g in groups] for p in pairs])
        files.append({"figure": f"cut{cut}/portrait_{s}.svg", "data": f"cut{cut}/portrait_{s}.csv", "series": s})
    page = ["<!doctype html><meta charset='utf-8'><title>Kernel portraits (plan12 move 4)</title>",
            "<style>body{font-family:Helvetica,Arial,sans-serif;margin:16px}img{width:100%;max-width:1040px;display:block;margin:4px 0 18px}p.n{color:#555;font-size:13px}</style>",
            f"<h1>Kernel portraits, cut of {cut} pairs ({cut_name})</h1><p class='n'>Move 4 of plan12_grounding: every kernel's average shape beside idle's, per series, one scale. The other cut: <a href='../index.html'>index</a>.</p>"]
    page += [f"<img src='portrait_{s}.svg' alt='portrait {s}'>" for s in SERIES]
    (d / "index.html").write_text("\n".join(page))
    return {"cut": cut, "cut_name": cut_name, "n_runs": len(runs), "groups": groups, "files": files, "index": f"cut{cut}/index.html"}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _top_index(d: Path, title: str, results: list[dict]) -> None:
    page = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>",
            "<style>body{font-family:Helvetica,Arial,sans-serif;margin:16px}p.n{color:#555;font-size:13px}</style>",
            f"<h1>{html.escape(title)}</h1><p class='n'>Every number of the paper is computed at both start-up cuts (SPEC section 2); pick one:</p><ul>"]
    for r in results:
        page.append(f"<li><a href='{r['index']}'>cut of {r['cut']} pairs ({r['cut_name']})</a>: {r['n_runs']} runs, {len(r['files'])} figures</li>")
    page.append("</ul>")
    (d / "index.html").write_text("\n".join(page))


def run_figures(out: Path, moves_dir: Path, runs: list[dict], which: str, argv: list[str]) -> list[dict]:
    """Move 3 or 4 at both cuts under `moves_dir` (the driver's `moves/`, or move 9's `moves/09_removed/`)."""
    cuts = cuts_of(out)
    fn = every_run if which == "every-run" else portraits
    mdir = moves_dir / ("03_every_run" if which == "every-run" else "04_portraits")
    results = [fn(out, name, cut, runs, argv, moves_dir) for name, cut in cuts.items()]
    _top_index(mdir, "Every run (plan12 move 3)" if which == "every-run" else "Kernel portraits (plan12 move 4)", results)
    write_json(mdir / "figures.json", {"schema": f"plan12.{'every_run' if which == 'every-run' else 'portraits'}.v1", "citation": CITATION,
                                       "package_version": __version__, "toolkit_fingerprint": toolkit_fingerprint()["sha256"],
                                       "command": argv, "written_at": now_iso(), "cuts": cuts, "n_runs": len(runs),
                                       "results": results})
    for r in results:
        print(f"[figures] {which} at cut {r['cut']} ({r['cut_name']}): {len(r['files'])} figures for {len(r['groups'])} groups -> {mdir / r['index']}")
    return results


def run_move(o: argparse.Namespace, which: str) -> int:
    out = Path(os.path.expanduser(o.out))
    cuts = cuts_of(out)
    if o.dry_run:
        print(f"[figures] dry run: would draw {which} at cuts {cuts} under {out / 'moves'}")
        return 0
    runs = load_runs(out)
    if not runs:
        print("no runs with a complete series (run move 1 first)", file=sys.stderr)
        return 2
    run_figures(out, out / "moves", runs, which, sys.argv)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plan12_grounding.figures", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, h in (("every-run", "move 3: every run per kernel and idle, the gallery"), ("portraits", "move 4: the kernel portraits")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--out", required=True)
        p.add_argument("--dry-run", action="store_true")
    o = ap.parse_args(argv)
    try:
        return run_move(o, o.cmd)
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
