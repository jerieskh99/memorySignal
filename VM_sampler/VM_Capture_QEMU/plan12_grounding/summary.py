#!/usr/bin/env python3
"""summary.py -- move 10 of plan12_grounding (SPEC move 10): one table, a row per kernel, at both
cuts; the paper's figure set under report/figures/; a manifest of every output with sha256.

  python3 -m plan12_grounding.summary summary --out O [--dry-run]

Per kernel and cut: above the floor or not, per series (move 7: the despiked bins above the floor
of the bins tested, and whether the run mean is above idle's); how alike its runs are (move 5: the
leave-one-seed-out rate of the kernel, and the median ICC over statistics, which is one number for
the corpus and is repeated on every row); how well it is recognized (move 6: the kernel's unit
recall under LORO and LOKO, E0 and E2); what the cosine adds (move 6: E2 minus E1, the kernel's
recall under LORO and LOKO, and the corpus accuracy margins). A missing move leaves its columns
blank with the reason in summary.json.

Writes `moves/10_summary/summary.json`, `report/table_per_kernel.csv` (+ `.html`),
`report/table_overall.csv`, `report/figures/` (copies of the figure set, SVG with CSV, with
`report/figures/index.csv` naming each source) and `report/manifest.json`.
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
import shutil  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import LEDGER, LOCK, install_sigterm, now_iso, read_json, sha256_file, write_json  # noqa: E402
from plan12_grounding.stats import ORDER, SERIES, cuts_of  # noqa: E402
from plan12_grounding.figures import write_csv  # noqa: E402
from plan11_encoding_ladder import schema  # noqa: E402

CITATION = "plan12_grounding/SPEC.md move 10 and section 6 (the figure set) and section 7 (report/)"
KERNELS = [k for k in ORDER if k != "idle"]
COLUMNS = ["cut", "kernel", "archetype", "n_runs",
           "floor_N", "floor_H", "floor_A", "floor_mean_N", "floor_mean_H", "floor_mean_A",
           "loso_rate", "icc_median_corpus",
           "loro_recall_E0", "loro_recall_E2", "loko_recall_E0", "loko_recall_E2",
           "cosine_adds_loro_E2_minus_E1", "cosine_adds_loko_E2_minus_E1"]


def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def per_kernel_table(out: Path, cut: int) -> tuple[list[dict], dict]:
    notes = {}
    m = out / "moves"
    fl = {(r["series"], r["mode"], r["kernel"]): r for r in _csv_rows(m / "07_floor" / f"cut{cut}" / "floor_bins.csv")}
    fm = {(r["series"], r["mode"], r["kernel"]): r for r in _csv_rows(m / "07_floor" / f"cut{cut}" / "floor_means.csv")}
    if not fl:
        notes["floor"] = "move 7 not found for this cut"
    loso = {r["group"]: r for r in _csv_rows(m / "05_similarity" / f"cut{cut}" / "loso_summary.csv")}
    sim = m / "05_similarity" / f"cut{cut}" / "summary.json"
    icc_med = read_json(sim).get("icc", {}).get("median") if sim.is_file() else None
    if not loso:
        notes["similarity"] = "move 5 not found for this cut"
    rk = {(r["split"], r["encoding"], r["kernel"]): _f(r["recall"]) for r in _csv_rows(m / "06_classify" / f"cut{cut}" / "recall_per_kernel.csv")}
    if not rk:
        notes["classify"] = "move 6 not found for this cut, or no recall per kernel (its splits not applicable)"
    n_runs = {}
    for r in _csv_rows(out / "cells.csv"):
        if r.get("admissible", "").lower() == "true" and r.get("role") == "kernel":
            n_runs[r["kernel"]] = n_runs.get(r["kernel"], 0) + 1
    rows = []
    for k in KERNELS:
        if k not in n_runs:
            continue
        row = {"cut": cut, "kernel": k, "archetype": schema.ARCHETYPE_OF.get(k, ""), "n_runs": n_runs[k]}
        for s in SERIES:
            b = fl.get((s, "despiked", k))
            row[f"floor_{s}"] = (f"{'yes' if int(b['n_above']) > int(b['n_bins']) / 2 else 'no'} ({b['n_above']} of {b['n_bins']} bins above)") if b else ""
            mm = fm.get((s, "despiked", k))
            row[f"floor_mean_{s}"] = (f"{'yes' if mm['above_bonferroni_kernels'] == 'True' else 'no'} (x{_f(mm['ratio_of_medians']):.2f})" if (mm and _f(mm["ratio_of_medians"]) is not None) else "")
        row["loso_rate"] = _f(loso[k]["rate_all"]) if k in loso else None
        row["icc_median_corpus"] = icc_med
        for sp in ("loro", "loko"):
            for e in ("E0", "E2"):
                row[f"{sp}_recall_{e}"] = rk.get((sp, e, k))
            e2, e1 = rk.get((sp, "E2", k)), rk.get((sp, "E1", k))
            row[f"cosine_adds_{sp}_E2_minus_E1"] = (e2 - e1) if (e2 is not None and e1 is not None) else None
        rows.append(row)
    return rows, notes


def overall_rows(out: Path, cut: int) -> list[dict]:
    rows = []
    for r in _csv_rows(out / "moves" / "06_classify" / f"cut{cut}" / "scores.csv"):
        rows.append({"cut": cut, "split": r["split"], "encoding": r["encoding"], "status": r["status"], "accuracy": _f(r["accuracy"]),
                     "macro_recall": _f(r["macro_recall"]), "majority": _f(r["majority"]), "null_p95": _f(r["null_p95"]), "n_units": r["n_units"]})
    for r in _csv_rows(out / "moves" / "06_classify" / f"cut{cut}" / "margins.csv"):
        rows.append({"cut": cut, "split": r["split"], "encoding": r["comparison"], "status": r["status"], "accuracy": _f(r["delta_accuracy"]),
                     "macro_recall": _f(r["delta_macro_recall"]), "majority": None, "null_p95": _f(r["null_delta_p95"]), "n_units": ""})
    return rows


def table_html(rows: list[dict], overall: list[dict], cuts: dict) -> str:
    h = ["<!doctype html><meta charset='utf-8'><title>The grounding paper: per-kernel table (plan12 move 10)</title>",
         "<style>body{font-family:Helvetica,Arial,sans-serif;margin:16px;color:#111}table{border-collapse:collapse;font-size:12px}"
         "th,td{border:1px solid #ddd;padding:3px 6px;text-align:right}th{background:#f3f3f3}td:nth-child(2),td:nth-child(3){text-align:left}"
         "td.f{text-align:left}p.n{color:#555;font-size:13px}</style>",
         "<h1>One table, a row per kernel, at both cuts</h1>",
         "<p class='n'>floor_*: above the noise floor (move 7, despiked): the bins above idle of the bins tested, and the run mean against idle's with the ratio of medians. "
         "loso_rate: leave-one-seed-out hit rate of the kernel's runs (move 5); icc_median_corpus: the median ICC(1) over statistics, one number for the corpus. "
         "loro/loko recall: the kernel's unit recall (move 6). cosine adds: E2 minus E1 recall.</p>"]
    for cut in cuts.values():
        h.append(f"<h2>Cut of {cut} pairs</h2><table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in COLUMNS[1:]) + "</tr>")
        for r in rows:
            if r["cut"] != cut:
                continue
            h.append("<tr>" + "".join(f"<td class='{'f' if str(c).startswith('floor') else ''}'>{html.escape(_fmt(r.get(c)))}</td>" for c in COLUMNS[1:]) + "</tr>")
        h.append("</table>")
        h.append("<h3>Corpus accuracy per split and encoding, and the margins</h3><table><tr><th>split</th><th>encoding</th><th>status</th><th>accuracy or delta</th><th>macro recall or delta</th><th>majority</th><th>null p95</th><th>units</th></tr>")
        for r in overall:
            if r["cut"] != cut:
                continue
            h.append("<tr>" + "".join(f"<td>{html.escape(_fmt(r.get(c)))}</td>" for c in ("split", "encoding", "status", "accuracy", "macro_recall", "majority", "null_p95", "n_units")) + "</tr>")
        h.append("</table>")
    return "\n".join(h)


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


FIGURE_SET = [  # (report name pattern, the SVG under <out>/moves, its data CSV under <out>/moves, per cut?)
    ("fig1_every_run_{S}_{G}_cut{cut}", "03_every_run/cut{cut}/overlay/{G}__{S}.svg", "03_every_run/cut{cut}/overlay/{G}__{S}.csv", True),
    ("fig2_portrait_{S}_cut{cut}", "04_portraits/cut{cut}/portrait_{S}.svg", "04_portraits/cut{cut}/portrait_{S}.csv", True),
    ("fig3a_pca_map_cut{cut}", "05_similarity/cut{cut}/pca_map.svg", "05_similarity/cut{cut}/pca_map.csv", True),
    ("fig3b_icc_bars_cut{cut}", "05_similarity/cut{cut}/icc_bars.svg", "05_similarity/cut{cut}/icc.csv", True),
    ("fig4_classification_bars_cut{cut}", "06_classify/cut{cut}/bars.svg", "06_classify/cut{cut}/bars.csv", True),
    ("fig4b_confusion_loko_cut{cut}", "06_classify/cut{cut}/confusion_loko.svg", "06_classify/cut{cut}/confusion_loko_E2.csv", True),
    ("fig4c_confusion_loro_cut{cut}", "06_classify/cut{cut}/confusion_loro.svg", "06_classify/cut{cut}/confusion_loro_E2.csv", True),
    ("fig5_floor_{S}_despiked_cut{cut}", "07_floor/cut{cut}/floor_{S}_despiked.svg", "07_floor/cut{cut}/spectra_{S}_despiked.csv", True),
    ("fig6_startup", "08_startup/startup.svg", "08_startup/spike_rate.csv", False),
    ("fig7_before_after", "09_removed/before_after.svg", "09_removed/before_after.csv", False),
]


def copy_figures(out: Path, cuts: dict) -> list[dict]:
    """The paper's figure set (SPEC section 6) copied under report/figures/ with stable names, each SVG
    with its data CSV, and an index naming every source."""
    fdir = out / "report" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    groups = sorted({r["kernel"] if r.get("role") == "kernel" else "idle" for r in _csv_rows(out / "cells.csv") if r.get("admissible", "").lower() == "true"},
                    key=lambda g: ORDER.index(g) if g in ORDER else 99)
    index = []
    for pattern, svg, csv_src, per_cut in FIGURE_SET:
        for cut in (cuts.values() if per_cut else [None]):
            for S in (SERIES if "{S}" in pattern else [None]):
                for G in (groups if "{G}" in pattern else [None]):
                    name = pattern.format(S=S, G=G, cut=cut)
                    for src, ext in ((svg, ".svg"), (csv_src, ".csv")):
                        sp = out / "moves" / src.format(S=S, G=G, cut=cut)
                        if sp.is_file():
                            shutil.copyfile(sp, fdir / (name + ext))
                            index.append({"figure": name + ext, "source": str(sp.relative_to(out)), "cut": cut, "sha256": sha256_file(sp)})
    write_csv(fdir / "index.csv", ["figure", "source", "cut", "sha256"], [[i["figure"], i["source"], i["cut"], i["sha256"]] for i in index])
    return index


def manifest(out: Path) -> dict:
    """Every output of the engine under <out> with its sha256. Left out: the record book and its lock,
    the manifest itself, the console's own folder (`.console/`, which the engine neither reads nor
    hashes) and any path with a component starting with a dot."""
    skip = {LEDGER, LOCK}
    files = {}
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.name in skip:
            continue
        rel = p.relative_to(out)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.as_posix() == "report/manifest.json":
            continue
        files[rel.as_posix()] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return files


def run_summary(out: Path, argv: list[str]) -> dict:
    cuts = cuts_of(out)
    rdir = out / "report"
    rdir.mkdir(parents=True, exist_ok=True)
    mdir = out / "moves" / "10_summary"
    mdir.mkdir(parents=True, exist_ok=True)
    rows, notes, overall = [], {}, []
    for name, cut in cuts.items():
        r, n = per_kernel_table(out, cut)
        rows += r
        notes[str(cut)] = n
        overall += overall_rows(out, cut)
    write_csv(rdir / "table_per_kernel.csv", COLUMNS, [[r.get(c) for c in COLUMNS] for r in rows])
    write_csv(rdir / "table_overall.csv", ["cut", "split", "encoding", "status", "accuracy_or_delta", "macro_recall_or_delta", "majority", "null_p95", "n_units"],
              [[r[c] for c in ("cut", "split", "encoding", "status", "accuracy", "macro_recall", "majority", "null_p95", "n_units")] for r in overall])
    (rdir / "table_per_kernel.html").write_text(table_html(rows, overall, cuts))
    figs = copy_figures(out, cuts)
    files = manifest(out)
    write_json(rdir / "manifest.json", {"schema": "plan12.manifest.v1", "citation": CITATION, "package_version": __version__,
                                        "toolkit_fingerprint": toolkit_fingerprint()["sha256"], "command": argv, "written_at": now_iso(),
                                        "n_files": len(files), "files": files})
    rec = {"schema": "plan12.summary.v1", "citation": CITATION, "package_version": __version__, "toolkit_fingerprint": toolkit_fingerprint()["sha256"],
           "command": argv, "written_at": now_iso(), "cuts": cuts, "n_kernel_rows": len(rows), "notes": notes, "table": rows, "overall": overall,
           "n_figures_copied": len(figs), "n_files_in_manifest": len(files),
           "files": {"table_per_kernel": str(rdir / "table_per_kernel.csv"), "table_overall": str(rdir / "table_overall.csv"),
                     "html": str(rdir / "table_per_kernel.html"), "figures": str(rdir / "figures"), "manifest": str(rdir / "manifest.json")}}
    write_json(mdir / "summary.json", rec)
    print(f"[summary] {len(rows)} kernel rows over cuts {list(cuts.values())}; {len(figs)} figure files copied to report/figures; manifest of {len(files)} files; "
          + ("notes: " + "; ".join(f"cut {c}: {', '.join(n.values())}" for c, n in notes.items() if n) if any(notes.values()) else "every move found"))
    return rec


def run(o: argparse.Namespace) -> int:
    out = Path(os.path.expanduser(o.out))
    if o.dry_run:
        print(f"[summary] dry run: would build the per-kernel table at cuts {cuts_of(out)}, copy the figure set and write the manifest under {out / 'report'}")
        return 0
    run_summary(out, sys.argv)
    return 0


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.summary", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("summary", help="move 10: the per-kernel table, the figure set, the manifest")
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
