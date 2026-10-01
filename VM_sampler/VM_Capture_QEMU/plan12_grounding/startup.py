#!/usr/bin/env python3
"""startup.py -- move 8 of plan12_grounding (SPEC move 8): the start-up. The spike rate against the
pair index, pooled over recordings, with the two cuts marked, and stencil seed 1298 on its own.

  python3 -m plan12_grounding.startup startup --out O [--dry-run]

Per recording and series (N, H, A), from the first pair (no cut: the start-up is what is shown), a
pair is a spike when its residual from the series' own 9-pair running median exceeds 6 x 1.4826 x MAD
of the residuals (head.py's rule, here on each series itself). The pooled rate at pair index p is
the share of the pooled recordings that spike at p; stencil seed 1298 (two regimes) is excluded from
the pool and drawn separately. As in head.py, the rate is also given per 16-pair bin against the
steady rate (pairs after 272 when the common length allows 64 such pairs, else the last half of the
common length; recorded) with a one-sided binomial test.

Writes `moves/08_startup/` (spike_rate.csv, spike_rate_by_group.csv, bins.csv, spikes_per_run.csv,
stencil_1298.csv when present, startup.svg, startup.json).
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

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import install_sigterm, now_iso, write_json  # noqa: E402
from plan12_grounding.stats import ORDER, SERIES, cuts_of, load_runs  # noqa: E402
from plan12_grounding.figures import C_A, C_H, C_N, panel, svg_open, write_csv  # noqa: E402
from plan12_grounding.floor import MED_WINDOW, Z_SPIKE, spike_mask  # noqa: E402

CITATION = ("plan12_grounding/SPEC.md move 8; grounding_paper/council/04_dsp_engineer_how_to_prove.md P5 and 04_dsp_artifacts/head.py "
            "(the ensemble spike rate by pair position, 16-pair bins against the steady rate, binomial test)")
BIN = 16
STEADY_AFTER = 272          # head.py: pairs 256 onward of a series that begins at raw pair 17
STENCIL = ("stencil_jacobi", 1298)
SERIES_COLOUR = {"N": C_N, "H": C_H, "A": C_A}


def is_stencil_1298(run: dict) -> bool:
    return run["kernel"] == STENCIL[0] and run.get("seed") == STENCIL[1]


def run_startup(out: Path, moves_dir: Path, runs: list[dict], argv: list[str]) -> dict:
    d = moves_dir / "08_startup"
    d.mkdir(parents=True, exist_ok=True)
    cuts = cuts_of(out)
    pooled = [r for r in runs if not is_stencil_1298(r)]
    control = [r for r in runs if is_stencil_1298(r)]
    masks = {}
    per_run_rows = []
    for r in runs:
        a = r["arrays"]
        m = {s: spike_mask(np.asarray(a[s], dtype=np.float64)) for s in SERIES}
        masks[r["cell_id"]] = (np.asarray(a["pair"]).astype(np.int64), m)
        per_run_rows.append([r["cell_id"], r["group"], r["seed"], int(a["pair"].size)] + [int(m[s].sum()) for s in SERIES] + [int(is_stencil_1298(r))])
    write_csv(d / "spikes_per_run.csv", ["cell_id", "group", "seed", "n_pairs"] + [f"n_spikes_{s}" for s in SERIES] + ["stencil_1298"], per_run_rows)
    if not pooled:
        rec = {"status": "not applicable: no recording to pool"}
        write_json(d / "startup.json", {"schema": "plan12.startup.v1", "citation": CITATION, **rec})
        return rec
    lo = min(int(masks[r["cell_id"]][0].min()) for r in pooled)
    hi = max(int(masks[r["cell_id"]][0].max()) for r in pooled)
    common = min(int(masks[r["cell_id"]][0].max()) for r in pooled)
    pairs = np.arange(lo, hi + 1)
    tot = {s: np.zeros(pairs.size) for s in SERIES}
    cnt = np.zeros(pairs.size, dtype=np.int64)
    by_group: dict = {}
    for r in pooled:
        p, m = masks[r["cell_id"]]
        idx = p - lo
        np.add.at(cnt, idx, 1)
        g = by_group.setdefault(r["group"], {"cnt": np.zeros(pairs.size, dtype=np.int64), **{s: np.zeros(pairs.size) for s in SERIES}})
        np.add.at(g["cnt"], idx, 1)
        for s in SERIES:
            np.add.at(tot[s], idx, m[s].astype(np.float64))
            np.add.at(g[s], idx, m[s].astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = {s: np.where(cnt > 0, tot[s] / np.maximum(cnt, 1), np.nan) for s in SERIES}
    write_csv(d / "spike_rate.csv", ["pair", "n_recordings"] + [f"rate_{s}" for s in SERIES] + [f"n_spikes_{s}" for s in SERIES],
              [[int(pairs[i]), int(cnt[i])] + [rate[s][i] for s in SERIES] + [int(tot[s][i]) for s in SERIES] for i in range(pairs.size)])
    grp_rows = []
    for g in [g for g in ORDER if g in by_group]:
        gg = by_group[g]
        with np.errstate(invalid="ignore", divide="ignore"):
            grp_rows += [[g, int(pairs[i]), int(gg["cnt"][i])] + [(gg[s][i] / gg["cnt"][i]) if gg["cnt"][i] else None for s in SERIES] for i in range(pairs.size) if gg["cnt"][i]]
    write_csv(d / "spike_rate_by_group.csv", ["group", "pair", "n_recordings"] + [f"rate_{s}" for s in SERIES], grp_rows)
    # the bins against the steady rate (head.py), on the common length only
    if common - STEADY_AFTER >= 64:
        steady_lo, steady_rule = STEADY_AFTER + 1, f"pairs after {STEADY_AFTER} (head.py: pairs 256 onward of the cut series)"
    else:
        steady_lo, steady_rule = lo + (common - lo + 1) // 2, "the last half of the common length (shorter than head.py's steady region)"
    bin_rows, steady = [], {}
    for s in SERIES:
        sel = (pairs >= steady_lo) & (pairs <= common)
        k_st, n_st = float(tot[s][sel].sum()), float(cnt[sel].sum())
        steady[s] = (k_st / n_st) if n_st else float("nan")
        b0 = lo
        while b0 + BIN - 1 <= common:
            sel_b = (pairs >= b0) & (pairs <= b0 + BIN - 1)
            k, n = int(tot[s][sel_b].sum()), int(cnt[sel_b].sum())
            r_b = k / n if n else float("nan")
            p = float(sps.binomtest(k, n, min(max(steady[s], 1e-12), 1 - 1e-12), alternative="greater").pvalue) if (n and np.isfinite(steady[s])) else float("nan")
            bin_rows.append([s, b0, b0 + BIN - 1, n, k, r_b, (r_b / steady[s]) if steady[s] else float("nan"), p])
            b0 += BIN
    write_csv(d / "bins.csv", ["series", "pair_lo", "pair_hi", "n_obs", "n_spikes", "rate", "ratio_to_steady", "p_binomial_greater"], bin_rows)
    stencil_rows = []
    for r in control:
        p, m = masks[r["cell_id"]]
        a = r["arrays"]
        stencil_rows += [[r["cell_id"], int(p[i])] + [float(a[s][i]) for s in SERIES] + [int(m[s][i]) for s in SERIES] for i in range(p.size)]
    if stencil_rows:
        write_csv(d / "stencil_1298.csv", ["cell_id", "pair"] + list(SERIES) + [f"spike_{s}" for s in SERIES], stencil_rows)
    (d / "startup.svg").write_text(startup_svg(pairs, cnt, rate, steady, cuts, control, masks, steady_lo, common))
    rec = {"schema": "plan12.startup.v1", "citation": CITATION, "package_version": __version__, "toolkit_fingerprint": toolkit_fingerprint()["sha256"],
           "command": argv, "written_at": now_iso(), "status": "ok",
           "params": {"z_spike": Z_SPIKE, "median_window": MED_WINDOW, "spike_rule": "each series against its own 9-pair running median (head.py's rule per series)",
                      "bin_pairs": BIN, "steady_from_pair": int(steady_lo), "steady_rule": steady_rule, "common_length_pairs": int(common), "marks": cuts},
           "n_pooled": len(pooled), "n_control": len(control), "stencil_1298": ([r["cell_id"] for r in control] or "absent from this corpus"),
           "steady_rate": steady,
           "first_bins": {s: [{"pairs": f"{r[1]}-{r[2]}", "rate": r[5], "ratio": r[6], "p": r[7]} for r in bin_rows if r[0] == s][:8] for s in SERIES},
           "files": sorted(p.name for p in d.iterdir())}
    write_json(d / "startup.json", rec)
    print(f"[startup] {len(pooled)} recordings pooled ({len(control)} stencil 1298 apart); steady rate N {steady['N']:.4f} from pair {steady_lo}; "
          + "; ".join(f"pairs {b['pairs']} ratio {b['ratio']:.1f} (p {b['p']:.1e})" for b in rec["first_bins"]["N"][:3]))
    return rec


def startup_svg(pairs, cnt, rate, steady, cuts, control, masks, steady_lo, common) -> str:
    W_, H_ = 940, 320 + (160 if control else 0)
    out = svg_open(W_, H_, "The start-up: spike rate against the pair index, pooled over recordings",
                   f"a spike: residual above 6 x 1.4826 x MAD from the series' own 9-pair running median; marks at the two cuts ({cuts['declared']}, {cuts['measured']}); "
                   f"dotted: the steady rate from pair {steady_lo}; grey: recordings present")
    x0, y0, w, h = 60, 50, 840, 200
    xs = pairs.astype(np.float64)
    traces = [(xs, rate[s], SERIES_COLOUR[s], 1.4, 0.95) for s in SERIES]
    yhi = float(np.nanmax(np.concatenate([rate[s] for s in SERIES]))) if pairs.size else 1.0
    yhi = max(0.05, min(1.0, yhi * 1.1))
    panel(out, x0, y0, w, h, float(pairs[0]), float(pairs[-1]), 0.0, yhi, traces, "pooled spike rate", f"N blue, H grey, A orange; {int(cnt.max())} recordings at most", "pair index")
    X = lambda x: x0 + w * (x - pairs[0]) / max(1, pairs[-1] - pairs[0])   # noqa: E731
    for name, c in cuts.items():
        out.append(f'<line x1="{X(c):.1f}" y1="{y0}" x2="{X(c):.1f}" y2="{y0 + h}" stroke="#b03a2e" stroke-dasharray="4,3"/>'
                   f'<text x="{X(c) + 3:.1f}" y="{y0 + 12}" fill="#b03a2e">{name} cut, {c}</text>')
    for s in SERIES:
        if np.isfinite(steady[s]):
            yy = y0 + h - h * steady[s] / yhi
            out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" stroke="{SERIES_COLOUR[s]}" stroke-dasharray="1,3"/>')
    cmax = float(cnt.max()) if cnt.size else 1.0
    out.append(f'<polyline points="{" ".join(f"{X(p):.1f},{y0 + h + 30 - 20 * c / cmax:.1f}" for p, c in zip(pairs, cnt))}" fill="none" stroke="#bbb"/>'
               f'<text x="{x0 + w + 4}" y="{y0 + h + 30}" fill="#888" font-size="8">n</text>')
    if control:
        r = control[0]
        p, m = masks[r["cell_id"]]
        a = r["arrays"]
        y1 = y0 + h + 60
        N = np.asarray(a["N"], dtype=np.float64)
        panel(out, x0, y1, w, 120, float(p[0]), float(p[-1]), 0.0, float(np.nanmax(N)) * 1.06 if N.size else 1.0,
              [(p.astype(np.float64), N, C_N, 1.0, 0.9)], f"stencil seed 1298 ({html.escape(r['cell_id'])}), N_t, shown apart", f"{int(m['N'].sum())} spike pairs marked", "pair index")
        for i in np.flatnonzero(m["N"]):
            out.append(f'<line x1="{x0 + w * (p[i] - p[0]) / max(1, p[-1] - p[0]):.1f}" y1="{y1 + 120}" x2="{x0 + w * (p[i] - p[0]) / max(1, p[-1] - p[0]):.1f}" y2="{y1 + 112}" stroke="#b03a2e"/>')
    out.append("</svg>")
    return "\n".join(out)


def run(o: argparse.Namespace) -> int:
    out = Path(os.path.expanduser(o.out))
    if o.dry_run:
        print(f"[startup] dry run: would pool the spike rate by pair index over every recording (stencil 1298 apart) under {out / 'moves' / '08_startup'}")
        return 0
    runs = load_runs(out)
    if not runs:
        print("no runs with a complete series (run move 1 first)", file=sys.stderr)
        return 2
    run_startup(out, out / "moves", runs, sys.argv)
    return 0


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.startup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("startup", help="move 8: the spike rate against the pair index")
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
