#!/usr/bin/env python3
"""floor.py -- move 7 of plan12_grounding (SPEC move 7): the noise floor. Each kernel against the 8
idle runs, per series (N, H, A), at both cuts.

  python3 -m plan12_grounding.floor floor --out O [--dry-run]

The council's test (grounding_paper/council/04_dsp_engineer_how_to_prove.md P3; 04_dsp_artifacts/
ensemble.py part (b)): per frequency bin, the kernel's 8 log spectra (Welch, 128-pair Hann segments,
half overlap, constant detrend; f = 0 dropped; the absolute power, not standardized) against idle's 8,
with Welch's t-test (unequal variances) and Bonferroni over the bins (0.05 / n_bins); a bin is above
the floor when it passes with the kernel above idle, below when with the kernel below. Done on the
raw series and on the despiked series: a spike pair is one whose residual from a 9-pair running
median exceeds 6 x 1.4826 x MAD of the residuals (head.py's rule); spikes are located on N_t and
replaced, in every series, by that series' own 9-pair running median (ensemble.py's rule for H). The
segment is 128 pairs, or the shortest series after the cut when that is shorter (recorded). The run
means: the 8 kernel run means against the 8 idle run means, Welch's t-test, Bonferroni over the 12
kernels, with the ratio of medians.

Writes `moves/07_floor/cut<H>/` (floor_bins.csv, floor_means.csv, spectra_<S>_<mode>.csv,
floor_<S>_<mode>.svg, summary.json) and `moves/07_floor/floor.json`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import html  # noqa: E402
import os  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal, stats as sps  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import install_sigterm, now_iso, write_json  # noqa: E402
from plan12_grounding.stats import ORDER, SERIES, cut_series, cuts_of, load_runs  # noqa: E402
from plan12_grounding.figures import SERIES_LABEL, svg_open, write_csv  # noqa: E402

CITATION = ("plan12_grounding/SPEC.md move 7; grounding_paper/council/04_dsp_engineer_how_to_prove.md P3 and "
            "04_dsp_artifacts/ensemble.py part (b) (Welch 128-pair segments, per-bin Welch t-test, Bonferroni 0.05/n_bins, raw and despiked), "
            "04_dsp_artifacts/head.py despike (9-pair running median, 6 x 1.4826 x MAD)")
NPERSEG = 128
ALPHA = 0.05
Z_SPIKE = 6.0
MED_WINDOW = 9
MODES = ("raw", "despiked")
KERNEL_COLOUR = {"stencil_jacobi": "#1f7a8c", "gemm": "#e07b00", "gibbs": "#5c9e31", "spmm": "#a23b72", "fem_assembly": "#8c564b",
                 "lexer": "#17becf", "rmat_gen": "#bcbd22", "bnb_tsp": "#7f7f7f", "floyd": "#1f77b4", "histogram": "#d62728",
                 "nbody": "#9467bd", "fft": "#ff7f0e"}


# ---------------------------------------------------------------------------------------------
# the despike rule (head.py) and the spectrum (ensemble.py)
# ---------------------------------------------------------------------------------------------
def spike_mask(x: np.ndarray, z: float = Z_SPIKE, window: int = MED_WINDOW) -> np.ndarray:
    """head.py: med = medfilt(x, 9); r = x - med; spike where r > z * 1.4826 * MAD(r)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return np.zeros(x.size, dtype=bool)
    k = min(window, x.size if x.size % 2 == 1 else x.size - 1)
    med = signal.medfilt(x, k)
    r = x - med
    mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    return r > z * mad


def running_median(x: np.ndarray, window: int = MED_WINDOW) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return x.copy()
    k = min(window, x.size if x.size % 2 == 1 else x.size - 1)
    return signal.medfilt(x, k)


def despike_all(arrays: dict) -> tuple[dict, np.ndarray]:
    """Spikes located on N; every series has its spike pairs replaced by its own running median."""
    m = spike_mask(arrays["N"])
    out = {}
    for s in SERIES:
        x = np.asarray(arrays[s], dtype=np.float64)
        med = running_median(x)
        out[s] = np.where(m, med, x)
    return out, m


def log_power(x: np.ndarray, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch (Hann, half overlap, constant detrend) of the series as it is (absolute power), f = 0
    dropped, log."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, p = signal.welch(x, nperseg=nperseg, noverlap=nperseg // 2, window="hann", detrend="constant")
    return f[1:], np.log(p[1:] + 1e-300)


# ---------------------------------------------------------------------------------------------
# one cut
# ---------------------------------------------------------------------------------------------
def prepared(runs: list[dict], cut: int) -> list[dict]:
    """Per run: the cut arrays, raw and despiked, and the spike count."""
    out = []
    for r in runs:
        a = cut_series(r, cut)
        if int(a["pair"].size) < 3:
            continue
        raw = {s: np.asarray(a[s], dtype=np.float64) for s in SERIES}
        des, m = despike_all(raw)
        out.append({**{k: r[k] for k in ("cell_id", "kernel", "group", "role", "seed", "rep")}, "n": int(a["pair"].size),
                    "raw": raw, "despiked": des, "n_spikes": int(m.sum())})
    return out


def floor_svg(series: str, mode: str, cut: int, freqs: np.ndarray, idle: dict, per_kernel: list[dict], nperseg: int) -> str:
    cols, rows = 4, 3
    pw, ph = 200, 120
    W_ = 60 + cols * (pw + 30)
    H_ = 60 + rows * (ph + 50)
    out = svg_open(W_, H_, f"Noise floor, {SERIES_LABEL[series]}, {mode} series, cut of {cut} pairs",
                   f"grey band: idle's 8 log spectra (min to max), grey line: their mean; colour: the kernel's mean log spectrum; shaded bins: above the floor "
                   f"(Welch t, Bonferroni {ALPHA}/{len(freqs)}); {nperseg}-pair segments")
    allv = [idle["mean"], idle["lo"], idle["hi"]] + [k["mean"] for k in per_kernel]
    v = np.concatenate([np.asarray(a) for a in allv if np.asarray(a).size])
    v = v[np.isfinite(v) & (v > np.log(1e-200))]          # a bin without power (a constant series) is clipped at the panel's floor, not scaled to
    ylo, yhi = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
    if yhi <= ylo:
        yhi = ylo + 1.0
    xlo, xhi = float(freqs.min()), float(freqs.max())
    if xhi <= xlo:
        xhi = xlo + 1.0
    X = lambda f: 0 + pw * (f - xlo) / (xhi - xlo)          # noqa: E731
    Y = lambda y: ph - ph * (min(max(y, ylo), yhi) - ylo) / (yhi - ylo)          # noqa: E731  clipped to the panel
    for i, k in enumerate(per_kernel):
        cx, cy = 60 + (i % cols) * (pw + 30), 50 + (i // cols) * (ph + 50)
        out.append(f'<g transform="translate({cx},{cy})">')
        ratio_txt = f'{k["median_ratio"]:.3g}' if k["median_ratio"] < 1e6 else "no idle power"
        out.append(f'<text x="0" y="-5" font-weight="bold">{html.escape(k["kernel"])}<tspan font-weight="normal" fill="#666" font-size="8"> {k["n_above"]} above, {k["n_below"]} below of {len(freqs)}; ratio {ratio_txt}</tspan></text>')
        out.append(f'<rect x="0" y="0" width="{pw}" height="{ph}" fill="none" stroke="#ccc"/>')
        bw = pw / max(1, len(freqs))
        for j in np.flatnonzero(k["above"]):
            out.append(f'<rect x="{X(freqs[j]) - bw / 2:.1f}" y="0" width="{bw:.1f}" height="{ph}" fill="#ffe9c9"/>')
        band = " ".join(f"{X(f):.1f},{Y(y):.1f}" for f, y in zip(freqs, idle["hi"])) + " " + " ".join(f"{X(f):.1f},{Y(y):.1f}" for f, y in zip(freqs[::-1], idle["lo"][::-1]))
        out.append(f'<polygon points="{band}" fill="#dddddd" fill-opacity="0.8"/>')
        out.append(f'<polyline points="{" ".join(f"{X(f):.1f},{Y(y):.1f}" for f, y in zip(freqs, idle["mean"]))}" fill="none" stroke="#777" stroke-width="1"/>')
        out.append(f'<polyline points="{" ".join(f"{X(f):.1f},{Y(y):.1f}" for f, y in zip(freqs, k["mean"]))}" fill="none" stroke="{KERNEL_COLOUR.get(k["kernel"], "#2b5d8a")}" stroke-width="1.6"/>')
        out.append(f'<text x="-3" y="8" text-anchor="end" font-size="8">{yhi:.1f}</text><text x="-3" y="{ph}" text-anchor="end" font-size="8">{ylo:.1f}</text>')
        out.append(f'<text x="0" y="{ph + 11}" font-size="8">{xlo:.3f}</text><text x="{pw}" y="{ph + 11}" text-anchor="end" font-size="8">{xhi:.3f} cycles per pair</text>')
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


def one_cut(out: Path, moves_dir: Path, cut_name: str, cut: int, runs: list[dict]) -> dict:
    d = moves_dir / "07_floor" / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    pre = prepared(runs, cut)
    idle = [p for p in pre if p["group"] == "idle"]
    kernels = [g for g in ORDER if g != "idle" and any(p["group"] == g for p in pre)]
    res = {"cut": cut, "cut_name": cut_name, "n_runs": len(pre), "n_idle": len(idle), "kernels": kernels}
    if len(idle) < 2:
        res["status"] = f"not applicable: {len(idle)} idle runs (the floor needs the idle runs)"
        write_json(d / "summary.json", {"schema": "plan12.floor_cut.v1", "citation": CITATION, **res})
        return res
    shortest = min(p["n"] for p in pre)
    nperseg = min(NPERSEG, shortest)
    if nperseg % 2 == 1 and nperseg > 2:
        nperseg -= 1
    n_bins = nperseg // 2
    bonf = ALPHA / max(1, n_bins)
    res.update(nperseg=nperseg, n_bins=n_bins, shortest_series=shortest, bonferroni_alpha=bonf, status="ok")
    bin_rows, mean_rows, per_series = [], [], {}
    for s in SERIES:
        per_series[s] = {}
        for mode in MODES:
            spectra = {}
            freqs = None
            for p in pre:
                f, lp = log_power(p[mode][s], nperseg)
                freqs = f
                spectra[p["cell_id"]] = lp
            I = np.array([spectra[p["cell_id"]] for p in idle])
            idle_band = {"mean": I.mean(0), "lo": I.min(0), "hi": I.max(0), "sd": I.std(0, ddof=1) if len(idle) > 1 else np.zeros(I.shape[1])}
            per_kernel = []
            imeans = np.array([float(np.mean(p[mode][s])) for p in idle])
            for k in kernels:
                kr = [p for p in pre if p["group"] == k]
                Xk = np.array([spectra[p["cell_id"]] for p in kr])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t, pv = sps.ttest_ind(Xk, I, axis=0, equal_var=False)
                t = np.where(np.isfinite(t), t, 0.0)
                pv = np.where(np.isfinite(pv), pv, 1.0)
                above = (pv < bonf) & (t > 0)
                below = (pv < bonf) & (t < 0)
                ratio = float(np.exp(np.median(Xk.mean(0) - I.mean(0))))
                n_idle_zero = int(np.sum(I.mean(0) < np.log(1e-200)))       # a constant idle series has no power: the ratio is then meaningless
                kmeans = np.array([float(np.mean(p[mode][s])) for p in kr])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tm, pm = sps.ttest_ind(kmeans, imeans, equal_var=False)
                tm = float(tm) if np.isfinite(tm) else 0.0
                pm = float(pm) if np.isfinite(pm) else 1.0
                med_ratio = float(np.median(kmeans) / np.median(imeans)) if np.median(imeans) != 0 else float("nan")
                entry = {"kernel": k, "n_runs": len(kr), "n_above": int(above.sum()), "n_below": int(below.sum()), "median_ratio": ratio, "n_idle_zero_bins": n_idle_zero,
                         "above": above, "below": below, "mean": Xk.mean(0), "t": t, "p": pv,
                         "mean_kernel_median": float(np.median(kmeans)), "mean_idle_median": float(np.median(imeans)), "mean_ratio": med_ratio,
                         "mean_t": tm, "mean_p": pm, "mean_above": bool(pm < ALPHA / max(1, len(kernels)) and tm > 0),
                         "mean_below": bool(pm < ALPHA / max(1, len(kernels)) and tm < 0)}
                per_kernel.append(entry)
                bin_rows.append([s, mode, k, len(kr), n_bins, entry["n_above"], entry["n_below"], ratio, n_idle_zero, nperseg])
                mean_rows.append([s, mode, k, len(kr), entry["mean_kernel_median"], entry["mean_idle_median"], med_ratio, tm, pm, entry["mean_above"], entry["mean_below"]])
            cols = ["freq_cycles_per_pair", "idle_mean", "idle_min", "idle_max", "idle_sd"] + [f"{e['kernel']}_{c}" for e in per_kernel for c in ("mean", "above", "below")]
            table = []
            for j in range(len(freqs)):
                row = [float(freqs[j]), float(idle_band["mean"][j]), float(idle_band["lo"][j]), float(idle_band["hi"][j]), float(idle_band["sd"][j])]
                for e in per_kernel:
                    row += [float(e["mean"][j]), int(e["above"][j]), int(e["below"][j])]
                table.append(row)
            write_csv(d / f"spectra_{s}_{mode}.csv", cols, table)
            (d / f"floor_{s}_{mode}.svg").write_text(floor_svg(s, mode, cut, freqs, idle_band, per_kernel, nperseg))
            per_series[s][mode] = {e["kernel"]: {"n_above": e["n_above"], "n_below": e["n_below"], "median_ratio": e["median_ratio"], "n_idle_zero_bins": e["n_idle_zero_bins"],
                                                 "mean_ratio": e["mean_ratio"], "mean_p": e["mean_p"], "mean_above": e["mean_above"]} for e in per_kernel}
    write_csv(d / "floor_bins.csv", ["series", "mode", "kernel", "n_runs", "n_bins", "n_above", "n_below", "median_power_ratio", "n_idle_zero_power_bins", "nperseg"], bin_rows)
    write_csv(d / "floor_means.csv", ["series", "mode", "kernel", "n_runs", "kernel_median_of_run_means", "idle_median_of_run_means", "ratio_of_medians",
                                      "t_welch", "p", "above_bonferroni_kernels", "below_bonferroni_kernels"], mean_rows)
    write_csv(d / "spikes.csv", ["cell_id", "group", "n_pairs", "n_spikes_N"], [[p["cell_id"], p["group"], p["n"], p["n_spikes"]] for p in pre])
    res.update(per_series=per_series, files=sorted(p.name for p in d.iterdir()),
               counts={s: {mode: {"kernels_above_half_bins": sum(1 for k in kernels if per_series[s][mode][k]["n_above"] > n_bins / 2),
                                  "kernels_mean_above": sum(1 for k in kernels if per_series[s][mode][k]["mean_above"])} for mode in MODES} for s in SERIES})
    write_json(d / "summary.json", {"schema": "plan12.floor_cut.v1", "citation": CITATION, **res})
    return res


def run_floor(out: Path, moves_dir: Path, runs: list[dict], argv: list[str]) -> list[dict]:
    cuts = cuts_of(out)
    results = [one_cut(out, moves_dir, name, cut, runs) for name, cut in cuts.items()]
    mdir = moves_dir / "07_floor"
    write_json(mdir / "floor.json", {"schema": "plan12.floor.v1", "citation": CITATION, "package_version": __version__,
                                     "toolkit_fingerprint": toolkit_fingerprint()["sha256"], "command": argv, "written_at": now_iso(),
                                     "params": {"nperseg": NPERSEG, "alpha": ALPHA, "z_spike": Z_SPIKE, "median_window": MED_WINDOW, "cuts": cuts,
                                                "despike_rule": "spikes located on N_t (residual from the 9-pair running median above 6 x 1.4826 x MAD); "
                                                                "each series' spike pairs replaced by its own 9-pair running median",
                                                "spectrum": "Welch, Hann, half overlap, constant detrend, absolute power, f = 0 dropped, log",
                                                "bins_test": "per bin Welch t-test kernel (8) against idle (8), Bonferroni 0.05 / n_bins, above = pass with t > 0",
                                                "means_test": "run means, kernel (8) against idle (8), Welch t-test, Bonferroni 0.05 / n_kernels"},
                                     "results": results})
    for r in results:
        if r.get("status") != "ok":
            print(f"[floor] cut {r['cut']} ({r['cut_name']}): {r.get('status')}")
            continue
        parts = []
        for s in SERIES:
            c = r["counts"][s]
            parts.append(f"{s}: {c['raw']['kernels_above_half_bins']} raw / {c['despiked']['kernels_above_half_bins']} despiked kernels above the floor in more than half the bins, "
                         f"{c['raw']['kernels_mean_above']} / {c['despiked']['kernels_mean_above']} with a run mean above idle's")
        print(f"[floor] cut {r['cut']} ({r['cut_name']}): {r['n_runs']} runs, {r['n_idle']} idle, {r['nperseg']}-pair segments ({r['n_bins']} bins); " + "; ".join(parts))
    return results


def run(o: argparse.Namespace) -> int:
    out = Path(os.path.expanduser(o.out))
    cuts = cuts_of(out)
    if o.dry_run:
        print(f"[floor] dry run: would test each kernel's spectra and run means against idle's at cuts {cuts}, raw and despiked, under {out / 'moves' / '07_floor'}")
        return 0
    runs = load_runs(out)
    if not runs:
        print("no runs with a complete series (run move 1 first)", file=sys.stderr)
        return 2
    run_floor(out, out / "moves", runs, sys.argv)
    return 0


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.floor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("floor", help="move 7: the noise floor, each kernel against the idle runs")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    o = ap.parse_args(argv)
    try:
        return run(o)
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
