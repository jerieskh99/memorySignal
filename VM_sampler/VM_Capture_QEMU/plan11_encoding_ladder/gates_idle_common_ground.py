#!/usr/bin/env python3
"""gates_idle_common_ground.py -- the idle common-ground test (move 16; added 2026-10-05, after the
run of 2026-09-29; P2_AUTHOR_ANSWERS.md A21, A22; SPEC_epoch2.md Part 4 item 30).

  python3 -m plan11_encoding_ladder.gates_idle_common_ground run --out O [--rungs apf,wapf,persist,content,combined]
        [--null-perm 500] [--n-jobs 1] [--n-estimators 300] [--seed-offset 0] [--perm-floor 500]

For each rung at its selected point (gates/selection.json): leave-one-run-out on the level-normalized
features with the 8 idle runs as a 13th class beside the 12 kernels (`models.prepare_split_data` with
`include_idle=True`, the `kernel` label space, in which an idle cell's label is `idle`), the forest of
SPEC 4.2 (`models.fit_predict_units`, unit = cell, majority vote over its windows, the dimension rule
as the splits apply it), scored at the unit (`models.score_units`): the recall of every class (each
kernel and idle), the accuracy over the 13 classes, the majority baseline. The null is the same
run-level label shuffle the LORO null uses (`nulls.shuffle_labels_units` with split `loro` and label
space `kernel`, seed SEED_LABEL_NULL): each permutation gives the accuracy and the idle recall, so
both carry a null. The verdicts are `models.b1_g1_verdict`'s: `pass` when the observed value strictly
exceeds the null's 95th percentile, `near_unfalsifiable` otherwise, and `not run: N permutations <
500` below the floor (the numbers are kept). B1-G3's quarantine is not applied here: this is a
recognition test per class, not a Table 2 score.

The second Table 2 (tables_eusipco.table2_corrected) reads the idle verdict: a reading is void only
when a held-out idle run is not recognised as idle above chance (idle recall not above the null's
95th percentile), in place of the floor check's part (i).

Outputs, never over an existing gate file: gates/added/idle_common_ground/<rung>/{scores.json,
predictions.csv, recall_per_class.csv, null.json} and gates/added/idle_common_ground.csv (one row
per rung) with its params.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import splits as SP  # noqa: E402
from plan11_encoding_ladder import models as M  # noqa: E402
from plan11_encoding_ladder import nulls as NL  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402

ADDED_2026_10_05 = "added 2026-10-05, after the run of 2026-09-29"
CITATION = ("P2_AUTHOR_ANSWERS.md A21 (the author's second proposal: a held-out idle run recognised as idle), A22 (move 16, item 2); "
            "SPEC_epoch2.md Part 4 item 30; P2 Sec. V 'The splits' (LORO, unit = cell); models.run_split_stage's include_idle, unused by moves 0 to 15")
IDLE_LABEL = "idle"
SPLIT, LABELSPACE = "loro", "kernel"
N_PERM_DEFAULT = M.B1G1_MIN_PERM
PERM_FLOOR_DEFAULT = M.B1G1_MIN_PERM
SUMMARY_COLUMNS = ("rung", "grid_id", "grid_source", "n_units", "n_idle", "n_kernel_cells", "feature_count", "feature_count_used",
                   "accuracy", "macro_recall", "majority", "idle_recall", "idle_null_p95", "idle_verdict",
                   "accuracy_null_p95", "accuracy_verdict", "n_perm", "note")


def _one_rung(out: Path, rung: str, *, n_perm: int, n_estimators: int, n_jobs: int, seed_offset: int, perm_floor: int) -> dict:
    gid, gsrc = S.selected_grid_id(out, rung, None)
    row = {"rung": rung, "grid_id": gid, "grid_source": gsrc, "note": ADDED_2026_10_05}
    if gid is None:
        row.update(idle_verdict=V.not_run(f"no selection for {rung}"), accuracy_verdict=V.not_run(f"no selection for {rung}"))
        return row
    norm = True
    p = S.features_path(out, rung, gid, norm)
    if not p.is_file():
        W, H = S.parse_grid_id(gid)
        hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
        S.build_features(out, None, rung, W, H, norm, hd)
    data = M.prepare_split_data(out, rung, gid, normalized=norm, include_idle=True)
    X, names, lab = data["X"], data["names"], data["lab"]
    cells = list(dict.fromkeys(lab["cell_id"].tolist()))
    first = {c: int(np.flatnonzero(lab["cell_id"] == c)[0]) for c in cells}
    kernel_of = {c: str(lab["kernel"][first[c]]) for c in cells}
    arche_of = {c: str(lab["archetype"][first[c]]) for c in cells}
    y_unit = dict(kernel_of)
    n_idle = sum(1 for c in cells if y_unit[c] == IDLE_LABEL)
    row.update(n_units=len(cells), n_idle=n_idle, n_kernel_cells=len(cells) - n_idle, feature_count=int(X.shape[1]))
    if lab["n"] == 0 or n_idle == 0 or len(set(y_unit.values())) < 2:
        v = V.not_run("no idle cell with windows at this point" if n_idle == 0 else "no window")
        row.update(idle_verdict=v, accuracy_verdict=v)
        return row
    folds = SP.folds_for(SPLIT, lab)
    seed = NL.SEED_FOREST + seed_offset
    y_win = np.array([y_unit[c] for c in lab["cell_id"]])
    preds = M.fit_predict_units(X, y_win, folds, lab["cell_id"], seed=seed, n_jobs=n_jobs, n_estimators=n_estimators)
    sc = M.score_units(y_unit, {c: v["y_pred"] for c, v in preds.items()}, kernel_of, None)
    maj = M.majority_baseline(y_unit, kernel_of, arche_of, SPLIT, LABELSPACE, folds, lab)
    idle_recall = (sc.get("recall_per_class") or {}).get(IDLE_LABEL)
    per_fold = {}
    for v in preds.values():
        per_fold.setdefault(str(v["fold"]), int(v["d_used"]))
    lo, hi = (min(per_fold.values()), max(per_fold.values())) if per_fold else (None, None)
    # the null: the LORO null's run-level label shuffles, scored on the accuracy and on the idle recall
    kern = np.array([kernel_of[c] for c in cells])
    rng = np.random.default_rng(NL.SEED_LABEL_NULL + seed_offset)
    perms = [NL.shuffle_labels_units(np.array(cells), kern, kern.copy(), SPLIT, LABELSPACE, rng) for _ in range(int(n_perm))]

    def one(pl):
        yu = {c: str(pl[i]) for i, c in enumerate(cells)}
        yw = np.array([yu[c] for c in lab["cell_id"]])
        pr = M.fit_predict_units(X, yw, folds, lab["cell_id"], seed=seed, n_jobs=1, n_estimators=n_estimators)
        s = M.score_units(yu, {c: v["y_pred"] for c, v in pr.items()}, kernel_of, None)
        return (s["accuracy"] if s["accuracy"] is not None else np.nan, (s.get("recall_per_class") or {}).get(IDLE_LABEL, np.nan))
    if n_jobs and n_jobs > 1 and len(perms) > 1:
        from joblib import Parallel, delayed
        vals = Parallel(n_jobs=n_jobs)(delayed(one)(pl) for pl in perms)
    else:
        vals = [one(pl) for pl in perms]
    null_acc = np.array([v[0] for v in vals], dtype=np.float64)
    null_idle = np.array([v[1] for v in vals], dtype=np.float64)
    v_acc, s_acc = M.b1_g1_verdict(sc["accuracy"], null_acc, perm_floor)
    v_idle, s_idle = M.b1_g1_verdict(idle_recall, null_idle, perm_floor)
    d = out / "gates" / "added" / "idle_common_ground" / rung
    d.mkdir(parents=True, exist_ok=True)
    params = {"added": ADDED_2026_10_05, "rung": rung, "grid_id": gid, "grid_source": gsrc, "split": SPLIT, "labelspace": LABELSPACE, "include_idle": True,
              "n_perm": int(n_perm), "perm_floor": int(perm_floor), "n_estimators": int(n_estimators), "seed_forest": seed, "seed_label_null": NL.SEED_LABEL_NULL + seed_offset,
              "seed_offset": int(seed_offset), "n_jobs": int(n_jobs), "normalized": norm, "unit": M.UNIT_AGGREGATION, "quarantine": "not applied (a recognition test per class)",
              "dimension_rule": "models.fit_predict_units auto_reduce, as the splits", "null": "nulls.shuffle_labels_units(split=loro, labelspace=kernel): the LORO null's run-level shuffles, scored on the accuracy and on the idle recall",
              "excluded_cells_hard": data["excluded_hard"], "excluded_cells_pair_rungs": data["excluded_pair_rungs"], "gk0_applied": data["gk0_applied"],
              "inputs_sha256": S.inputs_sha256([out / "cells.csv", p, out / "gates" / "preconditions.csv", out / "gates" / "gk0.csv", out / "gates" / "selection.json"], out)}
    S.write_json(d / "scores.json", "plan11.idle_common_ground.v1", params, CITATION, {
        "status": "ok", "accuracy": sc["accuracy"], "macro_recall": sc["macro_recall"], "recall_per_class": sc["recall_per_class"],
        "recall_per_kernel": sc["recall_per_kernel"], "majority": maj, "n_units": len(cells), "n_idle": n_idle,
        "idle_recall": idle_recall, "idle_verdict": v_idle, "idle_null": s_idle, "accuracy_verdict": v_acc, "accuracy_null": s_acc,
        "feature_count": int(X.shape[1]), "feature_count_used": (lo if lo == hi else f"{lo}-{hi}"), "feature_count_used_per_fold": per_fold,
        "n_folds": len(folds), "n_perm": int(len(null_acc))})
    S.write_csv(d / "predictions.csv", ["cell_id", "kernel", "archetype", "y_true", "y_pred", "fold", "vote_fraction", "n_windows"],
                [{"cell_id": c, "kernel": kernel_of[c], "archetype": arche_of[c], "y_true": y_unit[c], "y_pred": preds[c]["y_pred"] if c in preds else "",
                  "fold": preds[c]["fold"] if c in preds else "", "vote_fraction": preds[c]["vote_fraction"] if c in preds else "",
                  "n_windows": int(np.sum(lab["cell_id"] == c))} for c in cells])
    S.write_csv(d / "recall_per_class.csv", ["class", "recall", "n_cells"],
                [{"class": k, "recall": v, "n_cells": sum(1 for c in cells if y_unit[c] == k)} for k, v in sorted(sc["recall_per_class"].items())])
    S.write_json(d / "null.json", "plan11.idle_common_ground_null.v1", {"n_perm": int(len(null_acc)), "seed_label_null": NL.SEED_LABEL_NULL + seed_offset}, CITATION,
                 {"accuracy": null_acc.tolist(), "idle_recall": null_idle.tolist(), "summary_accuracy": s_acc, "summary_idle_recall": s_idle})
    row.update(accuracy=sc["accuracy"], macro_recall=sc["macro_recall"], majority=maj, idle_recall=idle_recall, idle_null_p95=s_idle.get("p95"),
               idle_verdict=v_idle, accuracy_null_p95=s_acc.get("p95"), accuracy_verdict=v_acc, n_perm=int(len(null_acc)),
               feature_count_used=(lo if lo == hi else f"{lo}-{hi}"))
    return row


def run_idle_common_ground(out: Path, *, rungs=S.RUNGS, n_perm: int = N_PERM_DEFAULT, n_estimators: int = M.N_ESTIMATORS, n_jobs: int = 1,
                           seed_offset: int = 0, perm_floor: int = PERM_FLOOR_DEFAULT) -> Path:
    out = Path(out)
    rows = []
    for rung in rungs:
        r = _one_rung(out, rung, n_perm=n_perm, n_estimators=n_estimators, n_jobs=n_jobs, seed_offset=seed_offset, perm_floor=perm_floor)
        rows.append({k: r.get(k, "") for k in SUMMARY_COLUMNS})
        print(f"[idle common ground] {rung}: {r.get('grid_id')} units {r.get('n_units')} accuracy {r.get('accuracy')} idle recall {r.get('idle_recall')} "
              f"null p95 {r.get('idle_null_p95')} -> {r.get('idle_verdict')}", flush=True)
    d = out / "gates" / "added"
    d.mkdir(parents=True, exist_ok=True)
    p = S.write_csv(d / "idle_common_ground.csv", list(SUMMARY_COLUMNS), rows)
    S.write_json(d / "idle_common_ground.params.json", "plan11.idle_common_ground_summary.v1",
                 {"added": ADDED_2026_10_05, "rungs": list(rungs), "n_perm": int(n_perm), "perm_floor": int(perm_floor), "n_estimators": int(n_estimators),
                  "seed_offset": int(seed_offset), "n_jobs": int(n_jobs), "void_rule": "a reading is void only when a held-out idle run is not recognised as idle above chance "
                  "(idle recall not above the null's 95th percentile): the second Table 2 applies it in place of G-F part (i)",
                  "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "gates" / "selection.json", out / "gates" / "preconditions.csv", out / "gates" / "gk0.csv"], out)},
                 CITATION, {"rows": rows})
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gates_idle_common_ground.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="move 16 (added 2026-10-05): the idle common-ground test per rung at its selected point")
    r.add_argument("--out", required=True)
    r.add_argument("--rungs", default=",".join(S.RUNGS))
    r.add_argument("--null-perm", type=int, default=N_PERM_DEFAULT)
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--n-estimators", type=int, default=M.N_ESTIMATORS)
    r.add_argument("--seed-offset", type=int, default=0)
    r.add_argument("--perm-floor", type=int, default=PERM_FLOOR_DEFAULT, help="below this many permutations the verdicts read `not run: N permutations < floor`, numbers kept")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr)
        return 2
    rungs = tuple(x.strip() for x in a.rungs.split(",") if x.strip())
    bad = [x for x in rungs if x not in S.RUNGS]
    if bad:
        print(f"unknown rung(s): {bad}", file=sys.stderr)
        return 2
    try:
        print(run_idle_common_ground(out, rungs=rungs, n_perm=a.null_perm, n_estimators=a.n_estimators, n_jobs=a.n_jobs,
                                     seed_offset=a.seed_offset, perm_floor=a.perm_floor))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
