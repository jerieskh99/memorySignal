#!/usr/bin/env python3
"""removal.py -- move 9 of plan12_grounding (SPEC move 9; decision D1): the noise-floor removal, a
switch, OFF by default (`--room-removal` on the driver). Rule A only.

  python3 -m plan12_grounding.removal removal --out O [--encoding-out E] [source flags]
        [--n-shuffles 1000] [--seed 20260930] [--null-perm 500] [--null-splits loko,within_trace]
        [--n-jobs 1] [--n-estimators 300] [--seed-offset 0] [--dry-run]

Rule A: from every recording, remove the pages that change (hamming > 0) in at least 95 percent of
the pairs in every one of the idle runs (the council found 26 such pages). The sets come from the
console's per-page store that move 1 wrote (the kept pairs of each run); the removal set R is their
intersection over the idle runs. Every series (N_t, H_t, A_t, ...) is rebuilt without those pages
(`extract.build_series` with `exclude_pages`) into `moves/09_removed/series/`, recorded in
`moves/09_removed/extract.json`; moves 3 to 7 are then recomputed on the rebuilt series into
`moves/09_removed/` (the same modules, another output folder) as a before-and-after. Per recording
the report gives its own always-changing set, how many of R it holds, and a flag when R is not
contained in it (the council: 6 of 96 runs). Move 6 under removal keeps E0 as it is (the encoding
run's own extract is the declared baseline and is not recomputed); E1, E2 and E_new change with
the series. The E0 identity check is reused from move 6 when it passed.

Writes `moves/09_removed/` (removal.json, removal_set.csv, pages_per_recording.csv, before_after.csv,
before_after.svg, series/, extract.json, 03_every_run/, 04_portraits/, 05_similarity/, 06_classify/,
07_floor/).
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
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import now_iso, read_json, sha256_file, write_json  # noqa: E402
from plan12_grounding.stats import SERIES, cuts_of, load_runs  # noqa: E402
from plan12_grounding.extract import build_series, save_series  # noqa: E402
from plan12_grounding.figures import svg_open, write_csv, run_figures  # noqa: E402
from plan12_grounding.similarity import run_similarity  # noqa: E402
from plan12_grounding.classify import add_classify_args, run_classify  # noqa: E402
from plan12_grounding.floor import run_floor  # noqa: E402
from plan10_analysis.runner import extract as L1  # noqa: E402

CITATION = ("plan12_grounding/SPEC.md move 9 (rule A) and decision D1; grounding_paper/council/05_al_kindi_idea.md section 3.1 "
            "(the idle core: pages changing in at least 95 percent of pairs; 26 shared by the 8 idle runs; 6 of 96 kernel runs without them)")
THRESHOLD = 0.95
SUBDIR = "09_removed"


def always_changing(npz_path: Path, meta: dict) -> tuple[set, int]:
    """The pages that change in at least THRESHOLD of the run's kept pairs (from the store's rows
    with hamming > 0, seq within the kept range). Returns (pages, n_pairs_counted)."""
    z = L1.load(npz_path)
    seq = np.asarray(z["seq"]).astype(np.int64)
    page = np.asarray(z["page_index"]).astype(np.int64)
    h = np.asarray(z["hamming"]).astype(np.float64)
    keep = h > 0
    bound = meta.get("keep_first_bound_seq")
    if bound is not None:
        keep &= seq <= int(bound)
    seq_k, page_k = seq[keep], page[keep]
    n_pairs = int(np.unique(seq_k).size)
    if n_pairs == 0:
        return set(), 0
    pairs_seen = np.unique(np.stack([seq_k, page_k], axis=1), axis=0)          # one row per (pair, page)
    pages, counts = np.unique(pairs_seen[:, 1], return_counts=True)
    return set(int(p) for p in pages[counts >= THRESHOLD * n_pairs]), n_pairs


def before_after_svg(rows: list[dict]) -> str:
    W_ = 900
    H_ = 60 + 18 * len(rows) + 20
    out = svg_open(W_, H_, "Noise-floor removal (rule A): before and after", "the same numbers computed on the series as move 1 wrote them (before) and without the removed pages (after)")
    out.append('<text x="10" y="50" font-weight="bold">quantity</text><text x="520" y="50" font-weight="bold" text-anchor="end">before</text>'
               '<text x="640" y="50" font-weight="bold" text-anchor="end">after</text><text x="760" y="50" font-weight="bold" text-anchor="end">change</text>')
    for i, r in enumerate(rows):
        y = 68 + 18 * i
        b, a = r["before"], r["after"]
        fmt = lambda v: ("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))   # noqa: E731
        ch = (a - b) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None
        out.append(f'<text x="10" y="{y}">{html.escape(r["quantity"])}</text><text x="520" y="{y}" text-anchor="end">{fmt(b)}</text>'
                   f'<text x="640" y="{y}" text-anchor="end">{fmt(a)}</text><text x="760" y="{y}" text-anchor="end" fill="{"#b03a2e" if (ch is not None and ch < 0) else "#3a9d5d"}">{fmt(ch) if ch is not None else ""}</text>')
    out.append("</svg>")
    return "\n".join(out)


def key_numbers(out: Path, moves_dir: Path, cut: int) -> dict:
    """The quantities compared before and after, from the move records of one folder."""
    q = {}
    sim = moves_dir / "05_similarity" / f"cut{cut}" / "summary.json"
    if sim.is_file():
        s = read_json(sim)
        q["ICC median over statistics"] = s.get("icc", {}).get("median")
        q["statistics above the ICC null p95"] = s.get("icc", {}).get("n_above_null_p95")
        q["leave-one-seed-out hit rate"] = (s.get("loso_rate") or {}).get("all")
    cl = moves_dir / "06_classify" / f"cut{cut}" / "summary.json"
    if cl.is_file():
        s = read_json(cl)
        for e in ("E0", "E1", "E2", "E_new"):
            for sp in ("loro", "loko", "within_trace"):
                q[f"unit accuracy {e} {sp}"] = ((s.get("scores") or {}).get(e) or {}).get(sp, {}).get("accuracy")
    fl = moves_dir / "07_floor" / f"cut{cut}" / "summary.json"
    if fl.is_file():
        s = read_json(fl)
        for ser in SERIES:
            c = (s.get("counts") or {}).get(ser) or {}
            q[f"kernels above the floor in more than half the bins, {ser} despiked"] = (c.get("despiked") or {}).get("kernels_above_half_bins")
            q[f"kernels with a run mean above idle's, {ser} despiked"] = (c.get("despiked") or {}).get("kernels_mean_above")
    return q


def run_removal(out: Path, o: argparse.Namespace, argv: list[str]) -> dict:
    runs = load_runs(out)
    if not runs:
        raise SystemExit("no runs with a complete series (run move 1 first)")
    mdir = out / "moves" / SUBDIR
    cuts = cuts_of(out)
    idle = [r for r in runs if r["group"] == "idle"]
    if o.dry_run:
        print(f"[removal] dry run: rule A over {len(idle)} idle runs of {len(runs)} recordings; would rebuild every series without the shared always-changing pages "
              f"into {mdir / 'series'} and recompute moves 3 to 7 into {mdir}")
        return {"dry_run": True}
    if not idle:
        raise SystemExit("rule A needs the idle runs: none in the corpus")
    mdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    own, n_pairs_of = {}, {}
    for r in runs:
        own[r["cell_id"]], n_pairs_of[r["cell_id"]] = always_changing(Path(r["meta"]["store"]["npz"]), r["meta"])
    R = set.intersection(*[own[r["cell_id"]] for r in idle]) if idle else set()
    write_csv(mdir / "removal_set.csv", ["page_index"], [[p] for p in sorted(R)])
    per_rows, flagged = [], []
    for r in runs:
        s = own[r["cell_id"]]
        missing = R - s
        if missing and r["group"] != "idle":
            flagged.append(r["cell_id"])
        per_rows.append([r["cell_id"], r["group"], n_pairs_of[r["cell_id"]], len(s), len(R & s), len(missing), int(bool(missing))])
    write_csv(mdir / "pages_per_recording.csv", ["cell_id", "group", "n_pairs_counted", "n_always_changing", "n_removed_from_always_changing", "n_of_removal_set_absent", "flag_removal_set_not_contained"], per_rows)
    print(f"[removal] rule A: {len(R)} pages change in at least {int(THRESHOLD * 100)} percent of pairs in every idle run ({len(idle)} idle runs); "
          f"{len(flagged)} kernel runs lack some of them", flush=True)
    # the rebuilt series
    recs = {}
    sdir = mdir / "series"
    for r in runs:
        rec = {"cell_id": r["cell_id"], "rec_rel": r["rec_rel"], "kernel": r["kernel"], "role": r["role"], "seed": r["seed"], "rep": r["rep"], "campaign": r["campaign"],
               "cut_declared": cuts["declared"], "cut_measured": cuts["measured"], "head_drop_pairs": r["meta"].get("head_drop_pairs_declared")}
        arrays, meta = build_series(Path(r["meta"]["store"]["npz"]), rec, r["meta"].get("keep_first_pairs"), exclude_pages=R)
        meta["removal"] = {"rule": "A", "threshold": THRESHOLD, "n_pages_removed": len(R), "source_series": str(out / "series" / f"{r['cell_id']}.npz")}
        sp = sdir / f"{r['cell_id']}.npz"
        save_series(sp, arrays, meta)
        recs[r["cell_id"]] = {"cell_id": r["cell_id"], "rec_rel": r["rec_rel"], "status": "done", "series": str(sp), "series_sha256": sha256_file(sp),
                              "n_pairs": meta["n_pairs_series"], "n_rows": meta["n_rows_kept"], "n_rows_removed": meta["excluded_pages"]["n_rows_removed"] if meta.get("excluded_pages") else 0}
    write_json(mdir / "extract.json", {"schema": "plan12.extract.v1", "citation": CITATION, "note": "the series of move 1 rebuilt without the removal set (move 9, rule A)",
                                       "written_at": now_iso(), "recordings": recs})
    print(f"[removal] {len(recs)} series rebuilt without {len(R)} pages ({round(time.time() - t0, 1)} s)", flush=True)
    # moves 3 to 7 again, on the rebuilt series, into moves/09_removed/
    runs2 = load_runs(out, series_index=mdir / "extract.json")
    run_figures(out, mdir, runs2, "every-run", argv)
    run_figures(out, mdir, runs2, "portraits", argv)
    run_similarity(out, mdir, runs2, int(o.n_shuffles), int(o.seed), argv)
    ident_path = out / "moves" / "06_classify" / "e0_identity.json"
    identity = read_json(ident_path) if ident_path.is_file() else None
    if identity is not None and not identity.get("passed"):
        identity = None
    run_classify(out, mdir, runs2, o, argv, identity=({**identity, "reused_from": str(ident_path)} if identity else None))
    run_floor(out, mdir, runs2, argv)
    # before and after
    ba_rows, ba_csv = [], []
    for name, cut in cuts.items():
        before, after = key_numbers(out, out / "moves", cut), key_numbers(out, mdir, cut)
        for k in before:
            ba_rows.append({"quantity": f"cut {cut}: {k}", "before": before[k], "after": after.get(k)})
            ba_csv.append([cut, name, k, before[k], after.get(k)])
    write_csv(mdir / "before_after.csv", ["cut", "cut_name", "quantity", "before", "after"], ba_csv)
    (mdir / "before_after.svg").write_text(before_after_svg(ba_rows))
    rec = {"schema": "plan12.removal.v1", "citation": CITATION, "package_version": __version__, "toolkit_fingerprint": toolkit_fingerprint()["sha256"],
           "command": argv, "written_at": now_iso(), "status": "ok",
           "params": {"rule": "A", "threshold": THRESHOLD, "n_idle_runs": len(idle), "cuts": cuts, "n_shuffles": int(o.n_shuffles), "seed": int(o.seed),
                      "null_perm": int(o.null_perm), "null_splits": o.null_splits, "n_estimators": int(o.n_estimators),
                      "e0_note": "E0 is the encoding run's own extract and is not recomputed under removal; E1, E2, E_new use the rebuilt series"},
           "removal_set": {"n_pages": len(R), "pages": sorted(R)},
           "recordings": {"n": len(runs), "n_flagged_kernel_runs_missing_pages": len(flagged), "flagged": flagged},
           "before_after": ba_rows, "elapsed_s": round(time.time() - t0, 1)}
    write_json(mdir / "removal.json", rec)
    print(f"[removal] done in {rec['elapsed_s']} s; before-and-after in {mdir / 'before_after.csv'}")
    return rec


def run(o: argparse.Namespace) -> int:
    out = Path(os.path.expanduser(o.out))
    run_removal(out, o, sys.argv)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plan12_grounding.removal", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("removal", help="move 9: rule A, then moves 3 to 7 again on the rebuilt series")
    add_classify_args(p)
    p.add_argument("--n-shuffles", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260930)
    o = ap.parse_args(argv)
    try:
        return run(o)
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(str(exc.code), file=sys.stderr)
            return 1
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
