#!/usr/bin/env python3
"""gates_comparison.py -- B1-G1, B1-G3, B1-G6 (re-exported from models.py where the split stage
applies them), G-L, G-N, G-X, G-DIM, G-M (SPEC section 3.7).

Citation: P2_STRUCTURE.md section V 5.1 Plan 08 (B1-G1, G3, G6 restated) and 5.2 'Comparisons'
(G-L, G-N, G-X, G-DIM, G-M); CR 2.1 items 9, 10, 11; CR 2.2 items 24, 25, 26, 32, 33;
K2 Sec. 4 item 3 (G-X as a measured leak test); SPEC_review_al_farabi.md item 2.9 (b) (no
selection -> 'not run: no selection for <rung>').
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import series as S
from plan11_encoding_ladder import models as M
from plan11_encoding_ladder import nulls as NL
from plan11_encoding_ladder import verdicts as V
from plan11_encoding_ladder.series import schema
from plan11_encoding_ladder.models import (b1_g1_verdict, quarantine_l1, majority_baseline, headline_classes_of,  # noqa: F401
                                           B1G1_MIN_PERM, B1G3_MAX_DISAGREE, DIM_MATCH_METHOD)

GL2_LEVEL = "kernel"                       # section 8 item 26
GL2_R2_MAX = 0.5
GL_FEATURE_DROP = ("cov", "std", "peak2med")
GN_HEADLINE_MIN = 3                        # CR 2.2 item 25
GX_N_PERM = 500                            # section 8 item 27
GM_N_SEEDS = 5                             # section 8 item 29
GM_SIGN_RULE = ((6, 0), (7, 1))            # (improving >= 6 and worsening == 0) or (improving >= 7 and worsening <= 1)
CIT_GL = "P2 Sec. V 5.2 G-L; CR 2.2 item 24"
CIT_GN = "P2 Sec. V 5.2 G-N; CR 2.2 item 25"
CIT_GX = "P2 Sec. V 5.2 G-X; CR 2.2 item 26; K2 Sec. 4 item 3"
CIT_GDIM = "P2 Sec. V 5.2 G-DIM; CR 2.2 item 32"
CIT_GM = "P2 Sec. V 5.2 G-M; CR 2.2 item 33"


SCORE_SOURCE = ("scores.json read through models.effective_scores: with_quarantine when a feature is "
                "quarantined (SPEC 3.7.2), else the full model; the same reading as the tables (CHECK_2.md B1)")


def _scores(out: Path, rung: str, gid: str, split: str, labelspace: str, base: str = "splits") -> dict | None:
    """A split's scores as the rung's score: the re-run without the quarantined features when B1-G3
    quarantined one, else the full model (SPEC 3.7.2 'Table rows use the re-run'; CR 2.1 item 10;
    models.effective_scores, the one reading shared with the tables; CHECK_2.md B1)."""
    p = M.split_dir(out, rung, gid, split, labelspace, base) / "scores.json"
    return M.effective_scores(S.read_json(p)) if p.is_file() else None


def _no_selection_rows(rung: str, extra: dict) -> dict:
    return {"rung": rung, "verdict": V.not_run(f"no selection for {rung}"), **extra}


# --------------------------------------------------------------------------- G-L (3.7.4)

GL_COLUMNS = ("rung", "part", "grid_id", "score_norm", "null_p95", "r2", "slope", "n_points", "verdict")


def gl_part2_regression(cv_per_kernel: dict, floor_per_kernel: dict, *, r2_max: float = GL2_R2_MAX) -> dict:
    """G-L (ii), the shot-noise route (DSP; CR 2.2 item 24): ordinary least squares of the per-kernel
    mean within-window CV on the per-kernel mean of 1 / sqrt(K_median_cell); r2 > 0.5 -> GL_SHOT_NOISE."""
    ks = [k for k in cv_per_kernel if k in floor_per_kernel and cv_per_kernel[k] is not None and floor_per_kernel[k] is not None]
    x = np.array([floor_per_kernel[k] for k in ks]); y = np.array([cv_per_kernel[k] for k in ks])
    n = len(ks)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"r2": None, "slope": None, "n_points": n, "verdict": V.not_run("fewer than three points or no spread")}
    A = np.stack([x, np.ones(n)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = float(np.sum((y - A @ coef) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"r2": r2, "slope": float(coef[0]), "n_points": n, "verdict": V.GL_SHOT_NOISE if r2 > r2_max else V.PASS}


def gate_gl(out: Path, *, gl2_level: str = GL2_LEVEL, r2_max: float = GL2_R2_MAX) -> Path:
    """G-L, level blindness (P2 Sec. V 5.2 G-L; CR 2.2 item 24). Part (i): per rung, the normalized
    LOKO/archetype score at the rung's selected point against its B1-G1 null p95 (scores.json);
    ``pass`` or GL_LEVEL_ONLY (the rung cannot be cited in the recovery clause; the raw score, where
    it exists, is the level-inclusive ceiling). Part (ii): at the selected APF point, per cell the
    mean over windows of the within-window CV of the raw APF series (the ``apf.k_over_n.cov`` feature);
    per kernel the mean of that and of 1 / sqrt(K_median_cell); OLS across the kernels
    (``gl2_level = "kernel"``; ``"cell"`` uses every cell); r2 > 0.5 -> GL_SHOT_NOISE (the driver then
    re-runs the split stage with feature_drop = ("cov", "std", "peak2med") and reports both); else pass.
    A rung without a selection writes ``not run: no selection for <rung>``."""
    out = Path(out)
    rows = []
    for rung in S.RUNGS:
        gid, _ = S.selected_grid_id(out, rung, None)
        if gid is None:
            rows.append(_no_selection_rows(rung, {"part": "i"})); continue
        sc = _scores(out, rung, gid, "loko", "archetype")
        if sc is None:
            rows.append({"rung": rung, "part": "i", "grid_id": gid, "verdict": V.not_run("LOKO/archetype scores.json missing")}); continue
        if str(sc.get("b1_g1", "")).startswith("not run"):
            v = V.not_run(str(sc["b1_g1"]).split(": ", 1)[1])
        elif sc.get("accuracy") is None:
            na = str(sc.get("b1_g1") or "")               # a `not applicable:` split carries its own string
            rows.append({"rung": rung, "part": "i", "grid_id": gid, "verdict": na if na.startswith("not applicable") else V.not_run("LOKO/archetype score missing")}); continue
        else:
            v = V.PASS if (sc.get("null_p95") is not None and sc["accuracy"] > sc["null_p95"]) else V.GL_LEVEL_ONLY
        rows.append({"rung": rung, "part": "i", "grid_id": gid, "score_norm": sc.get("accuracy"), "null_p95": sc.get("null_p95"), "verdict": v})
    gid, _ = S.selected_grid_id(out, "apf", None)
    if gid is None:
        rows.append(_no_selection_rows("apf", {"part": "ii"}))
    else:
        p = S.features_path(out, "apf", gid, False)
        if not p.is_file():
            rows.append({"rung": "apf", "part": "ii", "grid_id": gid, "verdict": V.not_run("raw APF feature file missing")})
        else:
            feat = S.load_features(p)
            j = list(feat["feature_names"]).index("apf.k_over_n.cov")
            cells = S.load_cells(out / "cells.csv")
            cells, _, _, _ = S.admissible_cells(out, cells, "apf")
            hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
            cv_cell, fl_cell, k_of = {}, {}, {}
            for c in cells:
                if c["role"] != "kernel":
                    continue
                m = feat["cell_id"] == c["cell_id"]
                if not m.any():
                    continue
                cv_cell[c["cell_id"]] = float(np.nanmean(feat["X"][m, j]))
                kmed = S.k_median_cell(S.load_extract_cached(out, c["cell_id"]), S.head_drop_for(hd, c["kernel"], c["role"]))
                fl_cell[c["cell_id"]] = 1.0 / np.sqrt(kmed) if kmed > 0 else None
                k_of[c["cell_id"]] = c["kernel"]
            if gl2_level == "kernel":
                cvk, flk = {}, {}
                for k in set(k_of.values()):
                    cs = [c for c in cv_cell if k_of[c] == k and fl_cell[c] is not None]
                    cvk[k] = float(np.mean([cv_cell[c] for c in cs])) if cs else None
                    flk[k] = float(np.mean([fl_cell[c] for c in cs])) if cs else None
                res = gl_part2_regression(cvk, flk, r2_max=r2_max)
            else:
                res = gl_part2_regression(cv_cell, fl_cell, r2_max=r2_max)
            rows.append({"rung": "apf", "part": "ii", "grid_id": gid, **res})
    p = S.write_csv(out / "gates" / "gl.csv", GL_COLUMNS, rows)
    S.write_params(p, "plan11.gl.v1", {"gl2_level": gl2_level, "r2_max": r2_max, "feature_drop_when_refused": list(GL_FEATURE_DROP),
                                       "score_source": SCORE_SOURCE,
                                       "cv_feature": "apf.k_over_n.cov (raw APF, mean over windows)",
                                       "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "gates" / "selection.json"], out)}, CIT_GL)
    return p


# --------------------------------------------------------------------------- G-N (3.7.5)

def gn_status(n_kernels: int) -> str:
    """n >= 3 -> GN_HEADLINE; 2 -> GN_ONE_TRAIN; 1 -> GN_NOVELTY; 0 -> GN_NO_ROW (CR 2.2 item 25)."""
    if n_kernels >= GN_HEADLINE_MIN:
        return V.GN_HEADLINE
    if n_kernels == 2:
        return V.GN_ONE_TRAIN
    if n_kernels == 1:
        return V.GN_NOVELTY
    return V.GN_NO_ROW


def gate_gn(out: Path, cells: list[dict] | None = None) -> Path:
    """G-N, class support (P2 Sec. V 5.2 G-N; CR 2.2 item 25). Per archetype row after G-K0's
    relabelling: the kernel count and its status; macro recall is taken over the headline rows only
    (models.score_units); a per-row recall with n for the others; IDLE has no kernel row unless
    G-K0 relabelled one. Output gates/gn.csv: archetype, n_kernels, kernels, status."""
    out = Path(out)
    if cells is None:
        cells = S.load_cells(out / "cells.csv")
    cells, _, _, _ = S.admissible_cells(out, cells, None)
    relabel = S.gk0_relabel(out)
    kern = {}
    for c in cells:
        if c["role"] == "kernel":
            kern[c["kernel"]] = relabel.get(c["kernel"], c["archetype_predicted"])
    rows = []
    for a in schema.ARCHETYPES:
        ks = sorted(k for k, aa in kern.items() if aa == a)
        rows.append({"archetype": a, "n_kernels": len(ks), "kernels": " ".join(ks), "status": gn_status(len(ks))})
    for a in sorted(set(kern.values()) - set(schema.ARCHETYPES)):
        ks = sorted(k for k, aa in kern.items() if aa == a)
        rows.append({"archetype": a, "n_kernels": len(ks), "kernels": " ".join(ks), "status": gn_status(len(ks))})
    p = S.write_csv(out / "gates" / "gn.csv", ("archetype", "n_kernels", "kernels", "status"), rows)
    S.write_params(p, "plan11.gn.v1", {"headline_min_kernels": GN_HEADLINE_MIN, "gk0_applied": bool(relabel), "relabelled_kernels": sorted(relabel),
                                       "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "gates" / "gk0.csv"], out)}, CIT_GN)
    return p


# --------------------------------------------------------------------------- G-X (3.7.6)

def gx_confound(kernel_campaigns: dict, kernel_arche: dict) -> tuple:
    """The confound record (ML's rule, CR 2.2 item 26): for every archetype with at least two kernels,
    the set of campaigns its kernels sit in; GX_CONFOUND_TOTAL if some archetype's kernels all sit in
    one campaign while another archetype's all sit in a different one; GX_CONFOUND_PARTIAL if any
    archetype's kernels sit in exactly one campaign; else GX_CONFOUND_NONE."""
    sets = {}
    for k, a in kernel_arche.items():
        sets.setdefault(a, set()).update(kernel_campaigns.get(k, set()))
    counts = {}
    for k, a in kernel_arche.items():
        counts[a] = counts.get(a, 0) + 1
    multi = {a: s for a, s in sets.items() if counts.get(a, 0) >= 2}
    single = {a: next(iter(s)) for a, s in multi.items() if len(s) == 1}
    verdict = V.GX_CONFOUND_NONE
    if single:
        verdict = V.GX_CONFOUND_PARTIAL
        if len(set(single.values())) >= 2:
            verdict = V.GX_CONFOUND_TOTAL
    return verdict, {a: sorted(s) for a, s in sets.items()}


GX_PERM_FLOOR_CLI = B1G1_MIN_PERM          # SPEC_epoch2 B3: the `gx` CLI default (500); the function default is 0


def gate_gx(out: Path, rung: str = "apf", *, n_perm: int = GX_N_PERM, n_jobs: int = 1, n_estimators: int = M.N_ESTIMATORS,
            seed_offset: int = 0, grid_id: str | None = None, perm_floor: int = 0) -> Path:
    """G-X, the blind campaign-leak test (P2 Sec. V 5.2 G-X; CR 2.2 item 26; K2 Sec. 4 item 3). The
    same forest, the same normalized features at the rung's selected point, label = campaign, split
    LOKO, unit accuracy, null = n_perm campaign-label shuffles across cells; GX_LEAK if the score
    strictly exceeds p95 else GX_POOLING_STANDS. Then the confound record regardless (gx_confound),
    and under LORO the held-out cell's campaign is in predictions.csv (models.run_split_stage). When
    GX_LEAK and GX_CONFOUND_TOTAL both hold the LOKO archetype headline is marked ``refused: campaign
    leak with total confound`` (gx.csv column ``headline_mark``). Output gates/gx.csv (rows of this rung
    replaced) and gates/gx.json (per-archetype campaign sets; inputs/cell_order.csv recorded when
    present). A campaign with a single label writes ``not applicable: one campaign label``.
    ``perm_floor`` (SPEC_epoch2 B3; CHECK_3 M3; P2 Sec. V 5.2 G-X reads "the unit-level null", whose
    floor Plan 08 states as at least 500 permutations): when ``0 < n < perm_floor`` (``n`` the null
    permutations scored) ``leak_verdict`` reads ``not run: <n> permutations < <perm_floor>`` exactly as
    ``models.b1_g1_verdict`` writes it, the score, null p95 and rank stay in their columns and no
    headline mark is set. Two defaults on purpose: 0 at the function (the epoch-1 contract of the
    direct calls) and ``GX_PERM_FLOOR_CLI = 500`` on the ``gx`` CLI, which the driver runs."""
    out = Path(out)
    gid = grid_id or S.selected_grid_id(out, rung, None)[0]
    cols = ("rung", "grid_id", "score", "null_p95", "rank", "leak_verdict", "confound_verdict", "headline_mark", "n_labels")
    p = out / "gates" / "gx.csv"
    old = [r for r in S.read_csv(p) if r["rung"] != rung] if p.is_file() else []
    cells = S.load_cells(out / "cells.csv")
    cells, _, _, _ = S.admissible_cells(out, cells, rung)
    relabel = S.gk0_relabel(out)
    kc, ka = {}, {}
    for c in cells:
        if c["role"] == "kernel":
            kc.setdefault(c["kernel"], set()).add(c["campaign"]); ka[c["kernel"]] = relabel.get(c["kernel"], c["archetype_predicted"])
    conf, sets = gx_confound(kc, ka)
    labels = sorted({c["campaign"] for c in cells if c["role"] == "kernel"})
    order_p = out / "inputs" / "cell_order.csv"
    order = S.read_csv(order_p) if order_p.is_file() else None
    jp = out / "gates" / "gx.json"
    jdoc = S.read_json(jp) if jp.is_file() else {"per_rung": {}}
    if gid is None:
        row = {"rung": rung, "leak_verdict": V.not_run(f"no selection for {rung}"), "confound_verdict": conf, "n_labels": len(labels)}
    elif len(labels) < 2:
        row = {"rung": rung, "grid_id": gid, "leak_verdict": V.not_applicable("one campaign label"), "confound_verdict": V.not_applicable("one campaign label"), "n_labels": len(labels)}
    else:
        d = M.run_split_stage(out, rung, gid, "loko", "campaign", n_perm=n_perm, n_jobs=n_jobs, n_estimators=n_estimators,
                              seed_offset=seed_offset, base_dir="gx_runs", quarantine=False,
                              label_override={c["cell_id"]: c["campaign"] for c in cells if c["role"] == "kernel"})
        sc = S.read_json(d / "scores.json")
        summ = sc.get("null_summary") or {}
        leak = V.GX_LEAK if summ.get("exceeds") else V.GX_POOLING_STANDS
        if sc.get("n_perm", 0) == 0:
            leak = V.not_run("no null permutation")
        elif perm_floor and 0 < int(sc.get("n_perm", 0)) < perm_floor:
            leak = V.not_run(f"{int(sc['n_perm'])} permutations < {perm_floor}")     # SPEC_epoch2 B3
        mark = V.refused("campaign leak with total confound") if (leak == V.GX_LEAK and conf == V.GX_CONFOUND_TOTAL) else ""
        row = {"rung": rung, "grid_id": gid, "score": sc.get("accuracy"), "null_p95": sc.get("null_p95"), "rank": sc.get("b1_g1_rank"),
               "leak_verdict": leak, "confound_verdict": conf, "headline_mark": mark, "n_labels": len(labels)}
    S.write_csv(p, cols, old + [row])
    jdoc["per_rung"][rung] = row
    S.write_json(jp, "plan11.gx.v1", {"n_perm": n_perm, "perm_floor": perm_floor, "labels": labels, "gk0_applied": bool(relabel), "seed_offset": seed_offset,
                                      "inputs_sha256": S.inputs_sha256([out / "cells.csv", order_p, out / "gates" / "selection.json"], out)},
                 CIT_GX, {"per_rung": jdoc["per_rung"], "archetype_campaign_sets": sets, "kernel_campaigns": {k: sorted(v) for k, v in kc.items()},
                          "cell_order": order})
    return p


# --------------------------------------------------------------------------- G-DIM (3.7.7)

MATCHED_COMBOS = (("loko", "archetype"), ("loro", "kernel"), ("loro", "archetype"),
                  ("within_trace", "kernel"), ("within_trace", "archetype"))   # SPEC_epoch2 3.5.1: Table 7's five (split, label space)
GDIM_MATCHED_SPLITS = "all"          # SPEC_epoch2 3.5.1 (Part 4 item 16): "all" | "loko" (the epoch-1 single LOKO run)
GDIM_NULL_SPLITS = "loko,loro,within_trace"
GDIM_COLUMNS = ("rung", "grid_id", "d", "d_matched", "method", "status", "loko_score", "matched_to",
                "split", "labelspace")    # epoch 2: the two appended columns (blank on rung rows; SPEC_epoch2 3.5.1)


def gate_gdim(out: Path, *, method: str = DIM_MATCH_METHOD, n_perm: int = B1G1_MIN_PERM, n_jobs: int = 1,
              n_estimators: int = M.N_ESTIMATORS, seed_offset: int = 0,
              matched_splits: str = GDIM_MATCHED_SPLITS, null_splits: str | set | tuple = GDIM_NULL_SPLITS) -> Path:
    """G-DIM, dimension parity (P2 Sec. V 5.2 G-DIM; CR 2.2 item 32). Every Table 7 row carries its
    feature count (scores.json feature_count). The combined rung is accompanied by a feature-count-
    matched comparison: d* = the feature count of the strongest single rung by LOKO score; the
    combined vector is reduced to d* per fold by ``dim_match_method`` ('train_importance': the forest's
    impurity importance fitted on the training fold only, top d*; 'pca': PCA fitted on the training
    fold) and written under gates/splits_matched/combined/<grid_id>/<split>__<labelspace>/. A rung whose d
    exceeds the training cell count in a fold ran with the reduction to n_train_cells (GDIM_REDUCED,
    from scores.json dim_status) else GDIM_FULL. Output gates/gdim.csv: rung, d, d_matched, method, status.

    Build epoch 2 (SPEC_epoch2 3.5.1; E1 sec. 4 M4; SPEC 6.3 plans a ``combined (matched)`` row for
    every split): the matched run is made for every (split, label space) of ``MATCHED_COMBOS`` under
    ``matched_splits = "all"`` (``"loko"`` keeps the epoch-1 single run), with the one d* chosen by
    LOKO for all five, each null governed by ``null_splits`` (``run_null = split in null_splits``, as the
    driver's ``--null-splits`` governs the rungs' own nulls). ``gdim.csv`` gains ``split`` and
    ``labelspace`` (appended after ``matched_to``; blank on the rung rows); the ``combined (matched)``
    rows are written LOKO first, so a reader that takes the first match (``tables._gdim_text``) reads
    the LOKO row as before. ``loko_score`` is filled on the LOKO row only; each matched split's own
    score lives in its ``scores.json``, which Table 7 reads. Within-trace at the whole-cell point stays
    ``not applicable: one window per cell`` (the split stage's own rule)."""
    out = Path(out)
    if isinstance(null_splits, str):
        null_splits = {x.strip() for x in null_splits.split(",") if x.strip()}
    null_splits = set(null_splits)
    combos = MATCHED_COMBOS if matched_splits == "all" else (("loko", "archetype"),)
    rows, best, best_score = [], None, -1.0
    for rung in S.RUNGS:
        gid, _ = S.selected_grid_id(out, rung, None)
        if gid is None:
            rows.append({"rung": rung, "status": V.not_run(f"no selection for {rung}")}); continue
        sc = _scores(out, rung, gid, "loko", "archetype")
        if sc is None:
            rows.append({"rung": rung, "grid_id": gid, "status": V.not_run("LOKO/archetype scores.json missing")}); continue
        rows.append({"rung": rung, "grid_id": gid, "d": sc.get("feature_count"), "d_matched": sc.get("feature_count"), "method": "",
                     "status": sc.get("dim_status") or V.GDIM_FULL, "loko_score": sc.get("accuracy")})
        if rung != "combined" and sc.get("accuracy") is not None and sc["accuracy"] > best_score:
            best, best_score = rung, sc["accuracy"]
    cg, _ = S.selected_grid_id(out, "combined", None)
    comb = next((r for r in rows if r["rung"] == "combined"), {})
    matched_dirs = {}
    if cg is not None and best is not None and comb.get("d") is not None:
        d_star = next(r["d"] for r in rows if r["rung"] == best)
        for split, ls in combos:
            try:
                d = M.run_split_stage(out, "combined", cg, split, ls, n_perm=n_perm, n_jobs=n_jobs, n_estimators=n_estimators,
                                      seed_offset=seed_offset, reduce_to=int(d_star), reduce_method=method, base_dir="splits_matched",
                                      run_null=split in null_splits)
                sc = S.read_json(d / "scores.json")
                matched_dirs[f"{split}__{ls}"] = str(d.relative_to(out))
                rows.append({"rung": "combined (matched)", "grid_id": cg, "d": comb.get("d"),
                             "d_matched": int(d_star), "method": method, "status": V.GDIM_REDUCED,
                             "loko_score": sc.get("accuracy") if (split, ls) == ("loko", "archetype") else None,
                             "matched_to": best, "split": split, "labelspace": ls})
            except FileNotFoundError as e:
                rows.append({"rung": "combined (matched)", "grid_id": cg, "status": V.not_run(f"missing input {e}"), "split": split, "labelspace": ls})
    elif cg is not None:
        rows.append({"rung": "combined (matched)", "grid_id": cg, "status": V.not_run("combined or the strongest single rung has no LOKO score")})
    p = S.write_csv(out / "gates" / "gdim.csv", GDIM_COLUMNS, rows)
    S.write_params(p, "plan11.gdim.v1", {"dim_match_method": method, "matched_to": best, "n_perm": n_perm, "reduction_target_over_dimension": "n_train_cells",
                                         "score_source": SCORE_SOURCE,
                                         "matched_splits": matched_splits, "matched_combos": [list(c) for c in combos],
                                         "null_splits": sorted(null_splits), "matched_dirs": matched_dirs, "epoch": 2,
                                         "inputs_sha256": S.inputs_sha256([out / "gates" / "selection.json"], out)}, CIT_GDIM)
    return p


# --------------------------------------------------------------------------- G-M (3.7.8)

def gm_compare(scores_a: dict, scores_b: dict, spread: float) -> dict:
    """G-M's rule for one ordered pair (A, B) on one split (P2 Sec. V 5.2 G-M; CR 2.2 item 33): diff =
    score_A - score_B; the sign test on the per-kernel recalls (A minus B; a tie is neither);
    GM_BEATS when diff > spread and (improving >= 6 and worsening == 0, or improving >= 7 and
    worsening <= 1); else GM_DIFFERENCE. ``spread`` is the measured max - min of the APF LOKO score
    over the G-M seeds."""
    ra, rb = scores_a.get("recall_per_kernel") or {}, scores_b.get("recall_per_kernel") or {}
    ks = sorted(set(ra) & set(rb))
    imp = sum(1 for k in ks if ra[k] > rb[k]); wor = sum(1 for k in ks if ra[k] < rb[k]); ties = len(ks) - imp - wor
    sa, sb = scores_a.get("accuracy"), scores_b.get("accuracy")
    if sa is None or sb is None or spread is None:
        return {"diff": None, "improving": imp, "worsening": wor, "ties": ties, "verdict": V.not_run("a score or the spread is missing")}
    diff = sa - sb
    sign_ok = any(imp >= i and wor <= w for i, w in GM_SIGN_RULE)
    return {"diff": diff, "improving": imp, "worsening": wor, "ties": ties,
            "verdict": V.GM_BEATS if (diff > spread and sign_ok) else V.GM_DIFFERENCE}


def gate_gm(out: Path, *, n_seeds: int = GM_N_SEEDS, n_estimators: int = M.N_ESTIMATORS, n_jobs: int = 1, seed_offset: int = 0) -> Path:
    """G-M, paired margin (P2 Sec. V 5.2 G-M; CR 2.2 item 33). Margin: the APF rung's LOKO score
    re-run with ``gm_n_seeds`` forest seeds (SEED_FOREST + i, written under gates/gm_runs/seed<i>/);
    spread = max - min of the scores (fold assignment is fixed by the split definitions, so the seed
    is the only source of spread; section 8 item 29). Then gm_compare for every ordered pair of rungs
    on every split (archetype space for LOKO, kernel space otherwise) from scores.json. Output
    gates/gm.csv: split, rung_a, rung_b, score_a, score_b, diff, spread, improving, worsening, ties, verdict."""
    out = Path(out)
    gid, _ = S.selected_grid_id(out, "apf", None)
    cols = ("split", "rung_a", "rung_b", "score_a", "score_b", "diff", "spread", "improving", "worsening", "ties", "verdict")
    if gid is None:
        p = S.write_csv(out / "gates" / "gm.csv", cols, [{"split": "loko", "rung_a": "apf", "verdict": V.not_run("no selection for apf")}])
        S.write_params(p, "plan11.gm.v1", {"n_seeds": n_seeds}, CIT_GM)
        return p
    seeds_scores = []
    for i in range(n_seeds):
        d = M.run_split_stage(out, "apf", gid, "loko", "archetype", n_perm=0, run_null=False, n_estimators=n_estimators, n_jobs=n_jobs,
                              seed=NL.SEED_FOREST + i, seed_offset=seed_offset, base_dir=f"gm_runs/seed{i}", quarantine=False)
        seeds_scores.append(S.read_json(d / "scores.json").get("accuracy"))
    valid = [s for s in seeds_scores if s is not None]
    spread = (max(valid) - min(valid)) if valid else None
    rows = []
    for split, ls in (("loko", "archetype"), ("loro", "kernel"), ("within_trace", "kernel")):
        sc = {}
        for rung in S.RUNGS:
            g, _ = S.selected_grid_id(out, rung, None)
            if g is None:
                continue
            s_ = _scores(out, rung, g, split, ls)
            if s_ is not None and s_.get("accuracy") is not None:
                sc[rung] = s_
        for a, b in itertools.permutations(sorted(sc, key=S.RUNGS.index), 2):
            res = gm_compare(sc[a], sc[b], spread)
            rows.append({"split": split, "rung_a": a, "rung_b": b, "score_a": sc[a]["accuracy"], "score_b": sc[b]["accuracy"], "spread": spread, **res})
    p = S.write_csv(out / "gates" / "gm.csv", cols, rows)
    S.write_params(p, "plan11.gm.v1", {"n_seeds": n_seeds, "seeds": [NL.SEED_FOREST + i + seed_offset for i in range(n_seeds)], "seed_scores": seeds_scores,
                                       "spread": spread, "sign_rule": [list(x) for x in GM_SIGN_RULE], "grid_id_apf": gid, "score_source": SCORE_SOURCE,
                                       "inputs_sha256": S.inputs_sha256([out / "gates" / "selection.json"], out)}, CIT_GM)
    return p


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gates_comparison.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("gl"); l.add_argument("--out", required=True); l.add_argument("--gl2-level", default=GL2_LEVEL, choices=("kernel", "cell"))
    n = sub.add_parser("gn"); n.add_argument("--out", required=True)
    x = sub.add_parser("gx"); x.add_argument("--out", required=True); x.add_argument("--rung", required=True, choices=S.RUNGS)
    x.add_argument("--null-perm", type=int, default=GX_N_PERM); x.add_argument("--n-jobs", type=int, default=1); x.add_argument("--n-estimators", type=int, default=M.N_ESTIMATORS)
    x.add_argument("--perm-floor", type=int, default=GX_PERM_FLOOR_CLI,
                   help="leak_verdict reads `not run: N permutations < floor` when 0 < null-perm < floor (SPEC_epoch2 B3)")
    d = sub.add_parser("gdim"); d.add_argument("--out", required=True); d.add_argument("--method", default=DIM_MATCH_METHOD, choices=("train_importance", "pca"))
    d.add_argument("--null-perm", type=int, default=B1G1_MIN_PERM); d.add_argument("--n-jobs", type=int, default=1); d.add_argument("--n-estimators", type=int, default=M.N_ESTIMATORS)
    d.add_argument("--matched-splits", default=GDIM_MATCHED_SPLITS, choices=("all", "loko"),
                   help="the combined (matched) run for every Table 7 (split, label space) or for LOKO only (SPEC_epoch2 3.5.1)")
    d.add_argument("--null-splits", default=GDIM_NULL_SPLITS, help="the matched splits whose null runs (the driver passes its own)")
    m = sub.add_parser("gm"); m.add_argument("--out", required=True); m.add_argument("--n-seeds", type=int, default=GM_N_SEEDS)
    m.add_argument("--n-jobs", type=int, default=1); m.add_argument("--n-estimators", type=int, default=M.N_ESTIMATORS)
    for sp in (l, n, x, d, m):
        sp.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr); return 2
    try:
        if args.cmd == "gl":
            print(gate_gl(out, gl2_level=args.gl2_level))
        elif args.cmd == "gn":
            print(gate_gn(out))
        elif args.cmd == "gx":
            print(gate_gx(out, args.rung, n_perm=args.null_perm, n_jobs=args.n_jobs, n_estimators=args.n_estimators, seed_offset=args.seed_offset,
                          perm_floor=args.perm_floor))
        elif args.cmd == "gdim":
            print(gate_gdim(out, method=args.method, n_perm=args.null_perm, n_jobs=args.n_jobs, n_estimators=args.n_estimators, seed_offset=args.seed_offset,
                            matched_splits=args.matched_splits, null_splits=args.null_splits))
        elif args.cmd == "gm":
            print(gate_gm(out, n_seeds=args.n_seeds, n_estimators=args.n_estimators, n_jobs=args.n_jobs, seed_offset=args.seed_offset))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
