#!/usr/bin/env python3
"""classify.py -- move 6 of plan12_grounding (SPEC move 6; decisions D2, D3): what the cosine adds
to the encoding paper's readings, under the encoding paper's own classification settings.

  python3 -m plan12_grounding.classify classify --out O [--encoding-out E]
        [--null-perm 500] [--null-splits loko,within_trace] [--n-jobs 1] [--n-estimators 300]
        [--seed-offset 0] [--dry-run]

E0 (decision D2) is the encoding run's own move-1 extract: its `extract/<cell>/extract.csv` and
`cells.csv`, read in place, never copied as logic. Every encoding-run file this move reads is declared
with its sha256 in `e0_declared.json` (cells.csv, the extracts and sidecars used, gates/selection.json,
gates/gk0.csv, gates/preconditions.csv). E0 is used only when move 1's identity check passed
(`moves/01_extract/e0_identity.json`: one recording recomputed with `extract_cell` and found
byte-identical); this move fetches nothing. E0's features are `series.window_features` of the
combined rung (60 features) at the window of decision D3: the encoding run's selected grid point
for its combined rung (`series.selected_grid_id`), else W=8, H=4.

The other encodings add the per-window statistics of slice 2 (`stats.window_stats`, the three
statistic sets per series): E1 = E0 + the statistics of N_t and H_t; E2 = E1 + those of A_t;
E_new = the statistics of N_t, H_t, A_t alone. A window of E1, E2 covers the same pairs as the E0
window it extends (pair = extract seq + 1, checked per window).

E0 reproduces the encoding paper's Table 2: the cells are taken in the encoding run's own cells.csv
order, its pair-rung exclusion is applied (`series.admissible_cells` with the combined rung), and
B1-G3's quarantine runs as there (`models.quarantine_l1`, the re-run without the quarantined
features, `models.effective_scores`). Splits, forest, unit scoring, majority baseline and
label-shuffle nulls are the encoding toolkit's (`splits.folds_for`, `models.fit_predict_units`,
`models.score_units`, `models.majority_baseline`, `models.headline_classes_of`,
`nulls.shuffle_labels_units`, `models.b1_g1_verdict`) with Table 2's settings: 300 trees, unit =
cell (majority vote over its windows), 500 permutations of the per-cell labels, LOKO on archetype
labels scored over the headline classes, kernel cells only. One list of permutations serves every
encoding at a cut, so the margins of E1 and E2 over E0 are paired, per cell and per permutation.
A null that did not run, or ran with fewer than 500 permutations, reads plan11's own strings
("not run: N permutations < 500"; "not run: null not requested (--null-splits)"). The check
`table2_check.json` says, for E0 at the declared cut, whether each Table 2 row of the encoding run
(its combined-rung split records at the D3 grid point) is reproduced exactly, naming the first
difference otherwise. Both cuts.

Writes `moves/06_classify/cut<H>/` (scores.csv, margins.csv, gap.csv, recall_per_kernel.csv,
predictions_*.csv, confusion_*.csv, null_*.csv, features.npz, excluded_cells.csv, bars.svg + bars.csv,
confusion_loko.svg, confusion_loro.svg, summary.json), `e0_declared.json`, `table2_check.json` and
`classify.json` (the D2 folder and the D3 grid point used; the console reads these).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import install_sigterm, now_iso, read_json, sha256_file, write_json  # noqa: E402
from plan12_grounding.stats import cut_series, cuts_of, load_runs, stat_names, window_stats  # noqa: E402
from plan12_grounding.figures import svg_open, write_csv  # noqa: E402
from plan11_encoding_ladder import series as S11  # noqa: E402
from plan11_encoding_ladder import splits as SP  # noqa: E402
from plan11_encoding_ladder import models as M  # noqa: E402
from plan11_encoding_ladder import nulls as NL  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST, SEED_LABEL_NULL  # noqa: E402

CITATION = ("plan12_grounding/SPEC.md move 6, decisions D2 and D3; plan11_encoding_ladder splits.py folds_for, models.py "
            "fit_predict_units / score_units / majority_baseline / headline_classes_of / quarantine_l1 / effective_scores / b1_g1_verdict, "
            "nulls.py shuffle_labels_units / null_summary (the encoding paper's Table 2 settings: 300 trees, unit = cell, 500 permutations, "
            "LOKO on archetypes, B1-G3's quarantine, the encoding run's cell order and pair-rung exclusion)")
ENCODINGS = ("E0", "E1", "E2", "E_new")
ENCODING_TEXT = {"E0": "the encoding paper's readings (combined rung, window features)",
                 "E1": "E0 + statistics of N_t and H_t", "E2": "E1 + statistics of A_t",
                 "E_new": "statistics of N_t, H_t, A_t alone"}
SPLITS = ("within_trace", "loro", "loko")
LABELSPACE = {"within_trace": "kernel", "loro": "kernel", "loko": "archetype"}
FALLBACK_GRID = "W8_H4"
E0_RUNG = "combined"
MARGIN_PAIRS = (("E1", "E0"), ("E2", "E0"), ("E2", "E1"), ("E_new", "E0"))
NULL_NOT_REQUESTED = V.not_run("null not requested (--null-splits)")
IDENTITY_RECORD = "moves/01_extract/e0_identity.json"
SCORE_KEYS = ("accuracy", "macro_recall", "majority", "null_p95", "b1_g1", "n_units")


# ---------------------------------------------------------------------------------------------
# D2: the encoding run's files, declared; D3: the window
# ---------------------------------------------------------------------------------------------
def encoding_cells(enc_out: Path) -> list[dict]:
    """The encoding run's cells.csv rows (every status), in the file's own order."""
    p = enc_out / "cells.csv"
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return S11.load_cells(p, only_ok=False)


def sidecar_of(enc_out: Path, cell_id: str) -> dict | None:
    p = enc_out / "extract" / cell_id / "sidecar.json"
    return read_json(p) if p.is_file() else None


def declare_e0(enc_out: Path, cell_ids: list[str]) -> dict:
    """sha256 of every encoding-run file this move reads."""
    files = {}
    for rel in ("cells.csv", "gates/selection.json", "gates/gk0.csv", "gates/preconditions.csv", "extract/extract_all.json"):
        p = enc_out / rel
        files[rel] = sha256_file(p) if p.is_file() else "absent"
    for c in cell_ids:
        for name in ("extract.csv", "sidecar.json"):
            p = enc_out / "extract" / c / name
            files[f"extract/{c}/{name}"] = sha256_file(p) if p.is_file() else "absent"
    return {"encoding_out": str(enc_out), "n_files": len(files), "files": files, "declared_at": now_iso()}


def resolve_window(enc_out: Path | None) -> dict:
    """Decision D3: the encoding run's selected grid point for its combined rung, else W=8, H=4."""
    if enc_out is not None:
        gid, src = S11.selected_grid_id(enc_out, E0_RUNG, default=None)
        if gid:
            W, H = S11.parse_grid_id(gid)
            return {"grid_id": gid, "W": W, "H": H, "source": f"{enc_out / 'gates' / 'selection.json'} ({src})", "whole_cell": W is None}
    W, H = S11.parse_grid_id(FALLBACK_GRID)
    return {"grid_id": FALLBACK_GRID, "W": W, "H": H, "whole_cell": False,
            "source": "fallback W=8, H=4 (SPEC D3): " + ("no encoding run named" if enc_out is None else "no selection for the combined rung in the encoding run")}


def identity_record(out: Path) -> dict:
    p = out / IDENTITY_RECORD
    if not p.is_file():
        return {"passed": False, "status": "not run: move 1 wrote no identity record"}
    return read_json(p)


# ---------------------------------------------------------------------------------------------
# the four encodings, per cut: aligned windows of E0 (the extract) and of the series (N, H, A)
# ---------------------------------------------------------------------------------------------
def cell_windows(run: dict, ex: dict | None, cut: int, win: dict) -> tuple:
    """(E0 rows [nw, 60] or None, statistic rows [nw, 76], window starts as row offsets, the pairs
    each window begins at, reason). The rows are the series after the cut (the first `cut` pairs and
    the last pair dropped, as the encoding toolkit's rung_series drops them); pair identity
    (pair == seq + 1) is required on every common row."""
    a = cut_series(run, cut)
    n12 = int(a["pair"].size)
    n = n12
    S = None
    if ex is not None:
        S = S11.rung_series(ex, E0_RUNG, normalized=True, head_drop=cut)
        n0 = int(S.shape[0])
        seq = np.asarray(ex["seq"][cut:cut + n0]).astype(np.int64)
        n = min(n12, n0)
        if n12 != n0:
            return None, None, None, None, f"row count differs after the cut: series {n12}, extract {n0}"
        if n and not np.array_equal(a["pair"][:n], seq[:n] + 1):
            j = int(np.flatnonzero(a["pair"][:n] != seq[:n] + 1)[0])
            return None, None, None, None, f"pair identity failed at common row {j}: series pair {int(a['pair'][j])}, extract seq {int(seq[j])} (pair = seq + 1 expected)"
    w, h = (n, n) if win["whole_cell"] else (int(win["W"]), int(win["H"]))
    if n == 0 or n < w:
        return None, None, None, None, f"no window: {n} rows after the cut, W = {w}"
    nw = S11.n_windows(n, w, h)
    all_starts = np.arange(nw) * h
    arrays = {k: v[:n] for k, v in a.items()}
    ws = window_stats(arrays, w, h)
    if ws["rows"].shape[0] != nw:
        return None, None, None, None, f"window count mismatch: stats {ws['rows'].shape[0]}, grid {nw}"
    if S is None:
        return None, ws["rows"], all_starts, a["pair"][all_starts], None
    F, starts, _ = S11.window_features(S[:n], E0_RUNG, w, h)
    keep = np.isin(all_starts, starts)
    return F, ws["rows"][keep], all_starts[keep], a["pair"][all_starts[keep]], None


def ordered_runs(runs: list[dict], enc_rows: list[dict] | None) -> list[dict]:
    """The kernel runs in the encoding run's cells.csv order (its feature rows and null permutations
    follow that order), else in cells.csv order of this run."""
    kernel = [r for r in runs if r["role"] == "kernel"]
    if not enc_rows:
        return kernel
    pos = {r["cell_id"]: i for i, r in enumerate(enc_rows)}
    return sorted(kernel, key=lambda r: (pos.get(r["cell_id"], len(pos)), r["cell_id"]))


def build_encodings(runs: list[dict], enc_out: Path | None, enc_rows: list[dict] | None, cut: int, win: dict, relabel: dict,
                    pair_rung_excluded: set) -> dict:
    """The feature block of every kernel run at one cut, with the column sets of the four encodings
    and the label dict the fold functions read."""
    e0_names = S11.feature_names(E0_RUNG)
    blocks, meta, excluded = [], {k: [] for k in ("cell_id", "kernel", "archetype", "campaign", "rep", "win_start", "pair_start")}, []
    st_names = stat_names()
    n_e0_windows = 0
    enc_by_id = {r["cell_id"]: r for r in (enc_rows or [])}
    for r in ordered_runs(runs, enc_rows):
        ex = None
        if enc_out is not None:
            ec = enc_by_id.get(r["cell_id"])
            if ec is None or ec.get("status") != "ok":
                excluded.append({"cell_id": r["cell_id"], "reason": "not an ok cell of the encoding run" if ec is None else f"encoding run status: {ec.get('status')}"})
                continue
            if r["cell_id"] in pair_rung_excluded:
                excluded.append({"cell_id": r["cell_id"], "reason": "the encoding run's pair-rung exclusion (failed_verdict refused; series.admissible_cells)"})
                continue
            sc = sidecar_of(enc_out, r["cell_id"])
            if sc is None or sc.get("status") != "ok" or not S11.extract_path(enc_out, r["cell_id"]).is_file():
                excluded.append({"cell_id": r["cell_id"], "reason": "no ok extract in the encoding run"})
                continue
            ex = S11.load_extract(enc_out, r["cell_id"])
        F, Xs, starts, pairs, why = cell_windows(r, ex, cut, win)
        if why is not None:
            excluded.append({"cell_id": r["cell_id"], "reason": why})
            continue
        k = Xs.shape[0]
        if F is None:
            F = np.full((k, len(e0_names)), np.nan)
        else:
            n_e0_windows += k
        blocks.append(np.concatenate([F, Xs], axis=1))
        meta["cell_id"] += [r["cell_id"]] * k
        meta["kernel"] += [r["kernel"]] * k
        meta["archetype"] += [relabel.get(r["kernel"], r["archetype"])] * k
        meta["campaign"] += [str(r["campaign"])] * k
        meta["rep"] += [int(r["rep"]) if r["rep"] is not None else 0] * k
        meta["win_start"] += [int(s) for s in starts]
        meta["pair_start"] += [int(p) for p in pairs]
    names = e0_names + st_names
    X = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, len(names)))
    cols = {"E0": [j for j, n in enumerate(names) if j < len(e0_names)],
            "E1": [j for j, n in enumerate(names) if j < len(e0_names) or n.startswith(("N.", "H."))],
            "E2": list(range(len(names))),
            "E_new": [j for j, n in enumerate(names) if j >= len(e0_names)]}
    lab = {"n": len(meta["cell_id"]), "_rows": np.arange(len(meta["cell_id"])),
           "cell_id": np.array(meta["cell_id"], dtype=str), "kernel": np.array(meta["kernel"], dtype=str),
           "archetype": np.array(meta["archetype"], dtype=str), "rep": np.array(meta["rep"], dtype=np.int64),
           "win_start": np.array(meta["win_start"], dtype=np.int64), "campaign": np.array(meta["campaign"], dtype=str)}
    return {"X": X, "names": names, "cols": cols, "lab": lab, "meta": meta, "excluded": excluded,
            "e0_available": enc_out is not None and n_e0_windows > 0, "n_e0_windows": n_e0_windows}


# ---------------------------------------------------------------------------------------------
# one split of one encoding: the point, the majority baseline, the paired null, B1-G3's quarantine
# ---------------------------------------------------------------------------------------------
def unit_maps(lab: dict) -> tuple:
    cells = list(dict.fromkeys(lab["cell_id"].tolist()))
    first = {c: int(np.flatnonzero(lab["cell_id"] == c)[0]) for c in cells}
    kernel_of = {c: str(lab["kernel"][first[c]]) for c in cells}
    arche_of = {c: str(lab["archetype"][first[c]]) for c in cells}
    rep_of = {c: int(lab["rep"][first[c]]) for c in cells}
    return cells, kernel_of, arche_of, rep_of


def permutations_for(lab: dict, split: str, labelspace: str, n_perm: int, seed_null: int) -> list[np.ndarray]:
    """The null's permutations, drawn as plan11's _null_scores draws them (the same seed, the cells in
    first-seen order), once per split and cut; every encoding uses the same list."""
    cells, kernel_of, arche_of, _ = unit_maps(lab)
    kern = np.array([kernel_of[c] for c in cells])
    labs = np.array([arche_of[c] if labelspace == "archetype" else kernel_of[c] for c in cells])
    rng = np.random.default_rng(seed_null)
    return [NL.shuffle_labels_units(np.array(cells), kern, labs, split, labelspace, rng) for _ in range(int(n_perm))]


def _fit_score(X, lab, y_unit, folds, kernel_of, headline, seed, n_jobs, n_est) -> tuple:
    y_win = np.array([y_unit[c] for c in lab["cell_id"]])
    preds = M.fit_predict_units(X, y_win, folds, lab["cell_id"], seed=seed, n_jobs=n_jobs, n_estimators=n_est)
    y_pred = {c: v["y_pred"] for c, v in preds.items()}
    return M.score_units(y_unit, y_pred, kernel_of, headline), preds


def _point(X, lab, y_unit, folds, kernel_of, arche_of, headline, split, labelspace, perms, run_null, seed, n_jobs, n_est) -> dict:
    """One model on one feature set: the point scores, the majority baseline, the null and its
    verdict in plan11's words (b1_g1_verdict), the per-fold width."""
    sc, preds = _fit_score(X, lab, y_unit, folds, kernel_of, headline, seed, n_jobs, n_est)
    maj = M.majority_baseline(y_unit, kernel_of, arche_of, split, labelspace, folds, lab)
    per_fold = {}
    for v in preds.values():
        per_fold.setdefault(str(v["fold"]), int(v["d_used"]))
    lo, hi = (min(per_fold.values()), max(per_fold.values())) if per_fold else (None, None)
    cells = list(y_unit)
    null = np.zeros(0)
    if run_null and perms:
        def one(pl):
            yu = {c: str(pl[i]) for i, c in enumerate(cells)}
            s, _ = _fit_score(X, lab, yu, folds, kernel_of, None, seed, 1, n_est)
            return s["accuracy"] if s["accuracy"] is not None else np.nan
        if n_jobs and n_jobs > 1 and len(perms) > 1:
            from joblib import Parallel, delayed
            vals = Parallel(n_jobs=n_jobs)(delayed(one)(pl) for pl in perms)
        else:
            vals = [one(pl) for pl in perms]
        null = np.asarray(vals, dtype=np.float64)
        verdict, summ = M.b1_g1_verdict(sc["accuracy"], null)
    else:
        verdict, summ = NULL_NOT_REQUESTED, NL.null_summary(sc["accuracy"], null)
    if str(verdict).startswith("not run"):
        summ = {**summ, "exceeds": None}            # no verdict exists: never a boolean then
    return {**sc, "majority": maj, "b1_g1": verdict, "b1_g1_rank": summ.get("rank"), "null_p95": summ.get("p95"), "null_summary": summ,
            "null": null.tolist(), "feature_count": int(X.shape[1]), "feature_count_used": (lo if lo == hi else f"{lo}-{hi}"),
            "feature_count_used_per_fold": per_fold, "preds": preds}


def run_split(X: np.ndarray, names: list[str], lab: dict, split: str, *, perms: list, run_null: bool, seed: int, n_jobs: int, n_est: int) -> dict:
    labelspace = LABELSPACE[split]
    cells, kernel_of, arche_of, rep_of = unit_maps(lab)
    y_unit = dict(arche_of) if labelspace == "archetype" else dict(kernel_of)
    res = {"split": split, "labelspace": labelspace, "n_units": len(cells), "n_windows": int(lab["n"]), "feature_count": int(X.shape[1])}
    if lab["n"] == 0:
        res.update(status=V.not_applicable("no window"), accuracy=None)
        return res
    one_per_cell = all(int(np.sum(lab["cell_id"] == c)) == 1 for c in cells)
    if split == "within_trace" and one_per_cell:
        res.update(status=V.not_applicable("one window per cell"), accuracy=None)
        return res
    if len(set(y_unit.values())) < 2:
        res.update(status=V.not_applicable("one class"), accuracy=None)
        return res
    folds = SP.folds_for(split, lab)
    headline = M.headline_classes_of({kernel_of[c]: arche_of[c] for c in cells}) if labelspace == "archetype" else None
    args = (lab, y_unit, folds, kernel_of, arche_of, headline, split, labelspace, perms, run_null, seed, n_jobs, n_est)
    full = _point(X, *args)
    # B1-G3 as plan11 runs it: a feature whose one-feature threshold model reproduces the full model's units is quarantined,
    # the split is re-run without it, and the re-run is the rung's score (models.effective_scores)
    quar = M.quarantine_l1(X, names, lab, y_unit, folds, full["preds"], seed=seed)
    with_q = None
    if quar:
        qnames = {q["feature"] for q in quar}
        keep = [j for j, n in enumerate(names) if n not in qnames]
        if keep:
            with_q = {**_point(X[:, keep], *args), "quarantined_features": sorted(qnames)}
        else:
            with_q = {"status": V.not_run("every feature quarantined"), "quarantined_features": sorted(qnames)}
    doc = {k: v for k, v in full.items() if k != "preds"}
    doc["with_quarantine"] = ({k: v for k, v in with_q.items() if k != "preds"} if with_q else None)
    eff = M.effective_scores(doc)
    use = with_q if (with_q and "preds" in with_q) else full
    preds = use["preds"]
    res.update(status="ok" if run_null else "ok; null not run (--null-splits)",
               **{k: eff.get(k) for k in ("accuracy", "macro_recall", "recall_per_class", "recall_per_kernel", "majority", "b1_g1", "b1_g1_rank", "null_p95",
                                          "null_summary", "feature_count", "feature_count_used", "feature_count_used_per_fold")},
               null=use.get("null", []), score_source=eff.get("score_source", M.SCORE_SOURCE_FULL), quarantined_features=eff.get("quarantined_features", []),
               full_model={k: full.get(k) for k in ("accuracy", "macro_recall", "majority", "b1_g1", "null_p95", "feature_count")},
               quarantine=quar, n_folds=len(folds), headline_classes=headline,
               predictions={c: {"y_true": y_unit[c], "y_pred": preds[c]["y_pred"] if c in preds else "", "fold": preds[c]["fold"] if c in preds else "",
                                "vote_fraction": preds[c]["vote_fraction"] if c in preds else None, "n_windows": int(np.sum(lab["cell_id"] == c)),
                                "kernel": kernel_of[c], "archetype": arche_of[c], "rep": rep_of[c]} for c in cells},
               predictions_full=({c: {"y_true": y_unit[c], "y_pred": full["preds"][c]["y_pred"] if c in full["preds"] else ""} for c in cells}
                                 if use is not full else None))
    return res


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
ENC_COLOUR = {"E0": "#8a8a8a", "E1": "#2b5d8a", "E2": "#d9822b", "E_new": "#3a9d5d"}


def bars_svg(results: dict, cut: int) -> str:
    """Per split, the four encodings' unit accuracy as bars, the null's p05 to p95 band behind each
    bar, the majority baseline as a dashed line, the numbers as text."""
    W_, H_ = 900, 300
    out = svg_open(W_, H_, f"Classification at the unit (cell), cut of {cut} pairs",
                   "bars: unit accuracy per encoding; grey band: the label-shuffle null's 5th to 95th percentile; dashed: the majority baseline")
    x0, y0, w, h = 60, 60, 260, 180
    for si, split in enumerate(SPLITS):
        px = x0 + si * (w + 30)
        out.append(f'<text x="{px}" y="{y0 - 6}" font-weight="bold">{split}</text>')
        out.append(f'<rect x="{px}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
        for k, t in enumerate((0.0, 0.5, 1.0)):
            yy = y0 + h - h * t
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + w}" y2="{yy:.1f}" stroke="#eee"/><text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.1f}</text>')
        bw = w / len(ENCODINGS)
        for ei, enc in enumerate(ENCODINGS):
            r = results.get(enc, {}).get(split, {})
            bx = px + ei * bw + 8
            acc = r.get("accuracy")
            ns = r.get("null_summary") or {}
            if ns.get("p95") is not None:
                lo, hi = float(ns["p05"]), float(ns["p95"])
                out.append(f'<rect x="{bx - 3:.1f}" y="{y0 + h - h * hi:.1f}" width="{bw - 10:.1f}" height="{max(1.0, h * (hi - lo)):.1f}" fill="#dddddd"/>')
            if acc is not None:
                out.append(f'<rect x="{bx:.1f}" y="{y0 + h - h * acc:.1f}" width="{bw - 16:.1f}" height="{h * acc:.1f}" fill="{ENC_COLOUR[enc]}" fill-opacity="0.85"/>')
                out.append(f'<text x="{bx + (bw - 16) / 2:.1f}" y="{y0 + h - h * acc - 3:.1f}" text-anchor="middle">{acc:.2f}</text>')
            else:
                out.append(f'<text x="{bx + (bw - 16) / 2:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" fill="#999" font-size="9">{html.escape(str(r.get("status", "not run"))[:24])}</text>')
            if r.get("majority") is not None:
                my = y0 + h - h * float(r["majority"])
                out.append(f'<line x1="{bx - 3:.1f}" y1="{my:.1f}" x2="{bx + bw - 13:.1f}" y2="{my:.1f}" stroke="#b03a2e" stroke-dasharray="3,2"/>')
            out.append(f'<text x="{bx + (bw - 16) / 2:.1f}" y="{y0 + h + 12}" text-anchor="middle">{enc}</text>')
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555">{"; ".join(f"{e}: {ENCODING_TEXT[e]}" for e in ENCODINGS)}</text></svg>')
    return "\n".join(out)


def confusion_table(pred: dict) -> tuple:
    labels = sorted(set(v["y_true"] for v in pred.values()) | set(v["y_pred"] for v in pred.values() if v["y_pred"]))
    idx = {l: i for i, l in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for v in pred.values():
        if v["y_pred"]:
            C[idx[v["y_true"]], idx[v["y_pred"]]] += 1
    return labels, C


def confusion_svg(results: dict, split: str, cut: int) -> str:
    panels = [(enc, results.get(enc, {}).get(split, {})) for enc in ENCODINGS]
    labels = None
    for _, r in panels:
        if r.get("predictions"):
            labels, _ = confusion_table(r["predictions"])
            break
    n = len(labels or [])
    cell = 16 if n > 6 else 24
    pw = 90 + n * cell + 20
    W_ = 20 + len(panels) * pw
    H_ = 90 + n * cell + 40
    out = svg_open(W_, H_, f"Confusion at the unit, {split} ({LABELSPACE[split]} labels), cut of {cut} pairs",
                   "rows: the true label; columns: the predicted label; the count of cells; shade: share of the row")
    for pi, (enc, r) in enumerate(panels):
        px = 20 + pi * pw + 80
        py = 70
        out.append(f'<text x="{px}" y="{py - 30}" font-weight="bold">{enc}</text>')
        if not r.get("predictions") or not labels:
            out.append(f'<text x="{px}" y="{py}" fill="#999">{html.escape(str(r.get("status", "not run"))[:40])}</text>')
            continue
        labs, C = confusion_table(r["predictions"])
        if labs != labels:
            labs = sorted(set(labs) | set(labels))
            labels = labs
        for j, l in enumerate(labs):
            out.append(f'<text x="{px + j * cell + cell / 2:.1f}" y="{py - 4}" text-anchor="end" font-size="8" transform="rotate(-60 {px + j * cell + cell / 2:.1f},{py - 4})">{html.escape(l[:14])}</text>')
        for i, l in enumerate(labs):
            out.append(f'<text x="{px - 4}" y="{py + i * cell + cell * 0.7:.1f}" text-anchor="end" font-size="8">{html.escape(l[:14])}</text>')
            row = C[i].sum() if i < C.shape[0] else 0
            for j in range(len(labs)):
                v = int(C[i, j]) if (i < C.shape[0] and j < C.shape[1]) else 0
                share = v / row if row else 0.0
                fill = f"rgb({int(255 - 180 * share)},{int(255 - 120 * share)},{int(255 - 60 * share)})" if v else "white"
                out.append(f'<rect x="{px + j * cell}" y="{py + i * cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ddd"/>')
                if v:
                    out.append(f'<text x="{px + j * cell + cell / 2:.1f}" y="{py + i * cell + cell * 0.7:.1f}" text-anchor="middle" font-size="8">{v}</text>')
        acc = r.get("accuracy")
        out.append(f'<text x="{px}" y="{py + n * cell + 14}" fill="#555" font-size="9">accuracy {acc:.3f}, {r.get("n_units")} cells</text>' if acc is not None else "")
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# one cut
# ---------------------------------------------------------------------------------------------
def one_cut(out: Path, moves_dir: Path, cut_name: str, cut: int, runs: list[dict], enc_out: Path | None, enc_rows: list[dict] | None,
            win: dict, relabel: dict, pair_rung_excluded: set, e0_status: str | None, *, n_perm: int, null_splits: set, seed_offset: int,
            n_jobs: int, n_est: int) -> dict:
    d = moves_dir / "06_classify" / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    enc = build_encodings(runs, enc_out, enc_rows, cut, win, relabel, pair_rung_excluded)
    X, names, lab = enc["X"], enc["names"], enc["lab"]
    seed = SEED_FOREST + seed_offset
    seed_null = SEED_LABEL_NULL + seed_offset
    np.savez_compressed(d / "features.npz", X=X, feature_names=np.array(names, dtype=str), cell_id=lab["cell_id"], kernel=lab["kernel"],
                        archetype=lab["archetype"], rep=lab["rep"], win_start=lab["win_start"], pair_start=np.array(enc["meta"]["pair_start"], dtype=np.int64),
                        campaign=lab["campaign"], cols_json=np.array(json.dumps(enc["cols"])), W=np.int64(win["W"] or 0), H=np.int64(win["H"] or 0),
                        grid_id=np.array(win["grid_id"]), cut=np.int64(cut))
    write_csv(d / "excluded_cells.csv", ["cell_id", "reason"], [[e["cell_id"], e["reason"]] for e in enc["excluded"]])
    results: dict = {e: {} for e in ENCODINGS}
    perms_by_split = {s: permutations_for(lab, s, LABELSPACE[s], n_perm, seed_null) if lab["n"] else [] for s in SPLITS}
    if e0_status is None and not enc["e0_available"]:
        e0_status = V.not_run("no E0 window (no ok extract, or the rows after the cut are shorter than W)")
    for e in ENCODINGS:
        for s in SPLITS:
            if e != "E_new" and e0_status is not None:
                results[e][s] = {"split": s, "labelspace": LABELSPACE[s], "status": e0_status, "accuracy": None, "n_units": 0}
                continue
            cols = enc["cols"][e]
            t0 = time.time()
            results[e][s] = run_split(X[:, cols], [names[j] for j in cols], lab, s, perms=perms_by_split[s], run_null=(s in null_splits),
                                      seed=seed, n_jobs=n_jobs, n_est=n_est)
            results[e][s]["elapsed_s"] = round(time.time() - t0, 1)
            r = results[e][s]
            print(f"[classify] cut {cut} {e:<5} {s:<13} {str(r.get('status'))[:40]:<40} acc {r.get('accuracy')} null {str(r.get('b1_g1'))[:36]}"
                  f"{' quarantined ' + str(len(r['quarantined_features'])) if r.get('quarantined_features') else ''} ({r['elapsed_s']} s)", flush=True)
    # tables
    score_rows = []
    for e in ENCODINGS:
        for s in SPLITS:
            r = results[e][s]
            ns = r.get("null_summary") or {}
            score_rows.append([s, LABELSPACE[s], e, r.get("status"), r.get("n_units"), r.get("n_windows"), r.get("accuracy"), r.get("macro_recall"), r.get("majority"),
                               ns.get("n"), ns.get("p95"), ns.get("p05"), ns.get("mean"), ns.get("rank"), r.get("b1_g1"),
                               r.get("score_source"), ";".join(r.get("quarantined_features") or []),
                               r.get("feature_count"), r.get("feature_count_used"), r.get("n_folds")])
    write_csv(d / "scores.csv", ["split", "labelspace", "encoding", "status", "n_units", "n_windows", "accuracy", "macro_recall", "majority",
                                 "null_n", "null_p95", "null_p05", "null_mean", "null_rank", "null_verdict", "score_source", "quarantined_features",
                                 "feature_count", "feature_count_used", "n_folds"], score_rows)
    margin_rows = []
    for s in SPLITS:
        for b, a in MARGIN_PAIRS:
            rb, ra = results[b][s], results[a][s]
            if ra.get("accuracy") is None or rb.get("accuracy") is None:
                margin_rows.append([s, f"{b}-{a}", None, None, None, None, None, None, None, None, "not run: " + str(ra.get("status") if ra.get("accuracy") is None else rb.get("status"))])
                continue
            pa, pb = ra["predictions"], rb["predictions"]
            gained = sum(1 for c in pb if pb[c]["y_pred"] == pb[c]["y_true"] and pa[c]["y_pred"] != pa[c]["y_true"])
            lost = sum(1 for c in pb if pb[c]["y_pred"] != pb[c]["y_true"] and pa[c]["y_pred"] == pa[c]["y_true"])
            na, nb = np.asarray(ra.get("null") or []), np.asarray(rb.get("null") or [])
            dn = nb - na if (na.size and na.size == nb.size) else np.zeros(0)
            delta = rb["accuracy"] - ra["accuracy"]
            dmac = (rb["macro_recall"] - ra["macro_recall"]) if (ra.get("macro_recall") is not None and rb.get("macro_recall") is not None) else None
            margin_rows.append([s, f"{b}-{a}", delta, dmac, gained, lost,
                                float(np.quantile(dn, 0.95)) if dn.size else None, float(np.quantile(dn, 0.05)) if dn.size else None,
                                float(dn.mean()) if dn.size else None, (float((1 + np.sum(dn >= delta)) / (1 + dn.size)) if dn.size else None),
                                "ok" if dn.size else "ok; no paired null (the null did not run on this split)"])
    write_csv(d / "margins.csv", ["split", "comparison", "delta_accuracy", "delta_macro_recall", "n_cells_gained", "n_cells_lost",
                                  "null_delta_p95", "null_delta_p05", "null_delta_mean", "p_paired_null", "status"], margin_rows)
    gap_rows = []
    for e in ENCODINGS:
        wt, lo = results[e]["within_trace"].get("accuracy"), results[e]["loro"].get("accuracy")
        gap_rows.append([e, wt, lo, (wt - lo) if (wt is not None and lo is not None) else None])
    write_csv(d / "gap.csv", ["encoding", "within_trace_accuracy", "loro_accuracy", "gap_within_minus_loro"], gap_rows)
    rk_rows = []
    for e in ENCODINGS:
        for s in SPLITS:
            for k, v in sorted((results[e][s].get("recall_per_kernel") or {}).items()):
                rk_rows.append([s, e, k, v])
    write_csv(d / "recall_per_kernel.csv", ["split", "encoding", "kernel", "recall"], rk_rows)
    for e in ENCODINGS:
        for s in SPLITS:
            r = results[e][s]
            if not r.get("predictions"):
                continue
            write_csv(d / f"predictions_{s}_{e}.csv", ["cell_id", "kernel", "archetype", "rep", "fold", "y_true", "y_pred", "n_windows", "vote_fraction"],
                      [[c, v["kernel"], v["archetype"], v["rep"], v["fold"], v["y_true"], v["y_pred"], v["n_windows"], v["vote_fraction"]] for c, v in r["predictions"].items()])
            if r.get("predictions_full"):
                write_csv(d / f"predictions_full_model_{s}_{e}.csv", ["cell_id", "y_true", "y_pred"], [[c, v["y_true"], v["y_pred"]] for c, v in r["predictions_full"].items()])
            if s in ("loko", "loro"):
                labs, C = confusion_table(r["predictions"])
                write_csv(d / f"confusion_{s}_{e}.csv", ["true\\predicted"] + labs, [[l] + C[i].tolist() for i, l in enumerate(labs)])
            if r.get("null"):
                write_csv(d / f"null_{s}_{e}.csv", ["permutation", "accuracy"], [[i, v] for i, v in enumerate(r["null"])])
    write_csv(d / "bars.csv", ["split", "encoding", "accuracy", "null_p05", "null_p95", "majority", "null_verdict"],
              [[s, e, results[e][s].get("accuracy"), (results[e][s].get("null_summary") or {}).get("p05"), (results[e][s].get("null_summary") or {}).get("p95"),
                results[e][s].get("majority"), results[e][s].get("b1_g1")] for s in SPLITS for e in ENCODINGS])
    (d / "bars.svg").write_text(bars_svg(results, cut))
    for s in ("loko", "loro"):
        (d / f"confusion_{s}.svg").write_text(confusion_svg(results, s, cut))
    summary = {"cut": cut, "cut_name": cut_name, "n_cells": int(len(set(lab["cell_id"].tolist()))), "n_windows": int(lab["n"]),
               "n_excluded": len(enc["excluded"]), "e0_available": enc["e0_available"], "e0_status": e0_status,
               "cell_order": "the encoding run's cells.csv order" if enc_rows else "this run's cells.csv order",
               "feature_counts": {e: len(enc["cols"][e]) for e in ENCODINGS},
               "scores": {e: {s: {k: results[e][s].get(k) for k in ("status", "accuracy", "macro_recall", "majority", "n_units", "feature_count_used",
                                                                       "b1_g1", "null_p95", "score_source", "quarantined_features")}
                              | {"null_n": (results[e][s].get("null_summary") or {}).get("n")} for s in SPLITS} for e in ENCODINGS},
               "margins": [dict(zip(("split", "comparison", "delta_accuracy", "delta_macro_recall", "n_cells_gained", "n_cells_lost", "null_delta_p95", "null_delta_p05", "null_delta_mean", "p_paired_null", "status"), r)) for r in margin_rows],
               "gap": [dict(zip(("encoding", "within_trace_accuracy", "loro_accuracy", "gap_within_minus_loro"), r)) for r in gap_rows],
               "files": sorted(p.name for p in d.iterdir())}
    write_json(d / "summary.json", {"schema": "plan12.classify_cut.v1", "citation": CITATION, **summary})
    return summary


# ---------------------------------------------------------------------------------------------
# E0 against the encoding run's own Table 2 rows (its combined-rung split records at the D3 point)
# ---------------------------------------------------------------------------------------------
def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(abs(float(a) - float(b)) <= 1e-12)
    return str(a) == str(b)


def table2_check(enc_out: Path | None, win: dict, cut_results: list[dict], cut_declared: int) -> dict:
    """For E0 at the declared cut, each Table 2 row of the encoding run (gates/splits/combined/<grid>/
    <split>__<labelspace>/scores.json, read through models.effective_scores) against this move's
    effective scores: reproduced exactly, or the first difference named."""
    rows = []
    mine = next((r for r in cut_results if r["cut"] == cut_declared), None)
    for split in SPLITS:
        lab = LABELSPACE[split]
        row = {"split": split, "labelspace": lab, "reproduced": None}
        if enc_out is None:
            row["note"] = "no encoding run named (decision D2)"
            rows.append(row); continue
        p = enc_out / "gates" / "splits" / E0_RUNG / win["grid_id"] / f"{split}__{lab}" / "scores.json"
        row["encoding_run_record"] = str(p.relative_to(enc_out))
        if not p.is_file():
            row["note"] = "the encoding run has no split record for this row (its moves 12 did not run, or another grid point)"
            rows.append(row); continue
        theirs = M.effective_scores(read_json(p))
        ours = (mine or {}).get("scores", {}).get("E0", {}).get(split) if mine else None
        if not ours or ours.get("accuracy") is None:
            row["note"] = f"E0 not computed here: {(ours or {}).get('status')}"
            row["theirs"] = {k: theirs.get(k) for k in SCORE_KEYS}
            rows.append(row); continue
        row["theirs"] = {k: theirs.get(k) for k in SCORE_KEYS}
        row["ours"] = {k: ours.get(k) for k in SCORE_KEYS}
        diffs = []
        for k in ("n_units", "accuracy", "macro_recall", "majority"):
            if not _close(theirs.get(k), ours.get(k)):
                diffs.append({"key": k, "encoding_run": theirs.get(k), "here": ours.get(k)})
        tv, ov = str(theirs.get("b1_g1")), str(ours.get("b1_g1"))
        both_ran = not tv.startswith("not run") and not ov.startswith("not run")
        if both_ran:
            for k in ("null_p95", "b1_g1"):
                if not _close(theirs.get(k), ours.get(k)):
                    diffs.append({"key": k, "encoding_run": theirs.get(k), "here": ours.get(k)})
        else:
            row["null_note"] = f"null verdicts not compared: encoding run '{tv}', here '{ov}'"
        row["reproduced"] = not diffs
        if diffs:
            row["first_difference"] = diffs[0]
            row["differences"] = diffs
        rows.append(row)
    n_checked = sum(1 for r in rows if r["reproduced"] is not None)
    return {"cut": cut_declared, "grid_id": win["grid_id"], "rows": rows, "n_rows_checked": n_checked,
            "n_reproduced": sum(1 for r in rows if r["reproduced"]), "all_reproduced": (n_checked > 0 and all(r["reproduced"] for r in rows if r["reproduced"] is not None))}


# ---------------------------------------------------------------------------------------------
# the move
# ---------------------------------------------------------------------------------------------
def run_classify(out: Path, moves_dir: Path, runs: list[dict], o: argparse.Namespace, argv: list[str]) -> dict:
    """The whole move at both cuts (called by the driver's move 6 and by move 9 with another moves_dir).
    Nothing is fetched: E0 stands on move 1's identity record."""
    cuts = cuts_of(out)
    params = read_json(out / "params.json")
    enc_out = Path(os.path.expanduser(o.encoding_out)) if getattr(o, "encoding_out", None) else (
        Path(params.get("D2_baseline", {}).get("encoding_out")) if params.get("D2_baseline", {}).get("encoding_out") else None)
    win = resolve_window(enc_out)
    null_splits = {s.strip() for s in str(o.null_splits).split(",") if s.strip()}
    mdir = moves_dir / "06_classify"
    if o.dry_run:
        print(f"[classify] dry run: encoding run {enc_out}; window {win}; splits {SPLITS}; null on {sorted(null_splits)} with {o.null_perm} permutations; "
              f"{o.n_estimators} trees; cuts {cuts}; would write under {mdir}; nothing written")
        return {"dry_run": True}
    mdir.mkdir(parents=True, exist_ok=True)
    enc_rows, declared, relabel, pair_excluded, e0_status = None, None, {}, set(), None
    identity = identity_record(out)
    if enc_out is not None:
        if not (enc_out / "cells.csv").is_file():
            raise FileNotFoundError(f"{enc_out / 'cells.csv'} (decision D2 names an encoding run without a cells.csv)")
        enc_rows = encoding_cells(enc_out)
        kernel_ids = [r["cell_id"] for r in runs if r["role"] == "kernel"]
        declared = declare_e0(enc_out, kernel_ids)
        write_json(mdir / "e0_declared.json", {"schema": "plan12.e0_declared.v1", "citation": CITATION, **declared})
        relabel = S11.gk0_relabel(enc_out)
        _, _, ex_pair, _ = S11.admissible_cells(enc_out, [r for r in enc_rows if r.get("status", "ok") == "ok"], E0_RUNG)
        pair_excluded = set(ex_pair)
        if not identity.get("passed"):
            e0_status = V.not_run(f"E0 identity check not passed in move 1 ({identity.get('status')}{': ' + str(identity.get('reason')) if identity.get('reason') else ''})")
            print(f"[classify] E0 not used: {e0_status}", flush=True)
        else:
            ref = (identity.get("reference") or {}).get("sha256")
            now = declared["files"].get(f"extract/{identity.get('cell_id')}/extract.csv")
            if ref and now and ref != now:
                e0_status = V.not_run(f"the encoding run's extract of {identity.get('cell_id')} changed since move 1's identity check (run move 1 again)")
                print(f"[classify] E0 not used: {e0_status}", flush=True)
            else:
                print(f"[classify] E0 identity check (move 1) on {identity.get('cell_id')}: passed", flush=True)
    else:
        e0_status = V.not_run("no encoding run named (decision D2)")
    results = [one_cut(out, moves_dir, name, cut, runs, enc_out, enc_rows, win, relabel, pair_excluded, e0_status, n_perm=int(o.null_perm),
                       null_splits=null_splits, seed_offset=int(o.seed_offset), n_jobs=int(o.n_jobs), n_est=int(o.n_estimators)) for name, cut in cuts.items()]
    t2 = table2_check(enc_out, win, results, cuts["declared"])
    write_json(mdir / "table2_check.json", {"schema": "plan12.table2_check.v1", "citation": CITATION, "written_at": now_iso(), **t2})
    rec = {"schema": "plan12.classify.v1", "citation": CITATION, "package_version": __version__, "toolkit_fingerprint": toolkit_fingerprint()["sha256"],
           "command": argv, "written_at": now_iso(),
           "params": {"D2_encoding_out": str(enc_out) if enc_out else None, "D2_cells_csv_sha256": (declared or {}).get("files", {}).get("cells.csv"),
                      "D2_declared_files": str(mdir / "e0_declared.json") if declared else None, "D3_window": win, "D3_grid_id": win["grid_id"],
                      "e0_status": e0_status or "used", "e0_identity": {k: identity.get(k) for k in ("cell_id", "passed", "status", "reason", "checked_at")},
                      "n_perm": int(o.null_perm), "null_splits": sorted(null_splits), "n_jobs": int(o.n_jobs), "n_estimators": int(o.n_estimators),
                      "seed_forest": SEED_FOREST + int(o.seed_offset), "seed_label_null": SEED_LABEL_NULL + int(o.seed_offset), "seed_offset": int(o.seed_offset),
                      "unit": "cell (majority vote over its windows; models.aggregate_units cell_majority)", "include_idle": False,
                      "test_frac_within_trace": SP.TEST_FRAC, "loro_mode": SP.LORO_MODE, "gk0_relabelled_kernels": sorted(relabel),
                      "pair_rung_excluded_cells": sorted(pair_excluded), "quarantine": "B1-G3 as plan11 (models.quarantine_l1, the re-run, effective_scores)",
                      "dimension_rule": "models.fit_predict_units auto_reduce: a fold whose feature count exceeds its training cell count is reduced to that count by training-fold importance (the encoding paper's rule); feature_count_used records it",
                      "cuts": cuts, "cut_convention": params.get("cuts", {}).get("convention"), "encodings": ENCODING_TEXT},
           "table2_check": t2, "results": results}
    write_json(mdir / "classify.json", rec)
    for r in results:
        line = "; ".join(f"{e} {s} {r['scores'][e][s]['accuracy']}" for e in ENCODINGS for s in SPLITS if r["scores"][e][s]["accuracy"] is not None)
        print(f"[classify] cut {r['cut']} ({r['cut_name']}): {r['n_cells']} cells, {r['n_windows']} windows, E0 {'used' if r['e0_available'] else 'absent'}; {line}")
    print(f"[classify] E0 against the encoding run's Table 2 rows at cut {t2['cut']}, grid {t2['grid_id']}: {t2['n_reproduced']} of {t2['n_rows_checked']} checked rows reproduced"
          + "".join(f"; {r['split']}: {r.get('note') or ('reproduced' if r['reproduced'] else 'differs at ' + str(r.get('first_difference')))}" for r in t2["rows"]))
    return rec


def run(o: argparse.Namespace) -> int:
    out = Path(os.path.expanduser(o.out))
    runs = load_runs(out) if not o.dry_run else []
    if not runs and not o.dry_run:
        print("no runs with a complete series (run move 1 first)", file=sys.stderr)
        return 2
    run_classify(out, out / "moves", runs, o, sys.argv)
    return 0


def add_classify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True)
    p.add_argument("--encoding-out", default=None, help="decision D2: the encoding run (default: params.json D2_baseline.encoding_out)")
    p.add_argument("--null-perm", type=int, default=M.B1G1_MIN_PERM, help="label permutations of the null (Table 2: 500)")
    p.add_argument("--null-splits", default="loko,within_trace", help="splits whose null runs (the encoding run's paper preset; add loro to pay its 96-fold cost)")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--n-estimators", type=int, default=M.N_ESTIMATORS)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.classify", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("classify", help="move 6: the four encodings under the three splits, both cuts")
    add_classify_args(p)
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
