#!/usr/bin/env python3
"""detection_levels.py -- level 2 (which sub-family, one member held out) and level 3 (the exact
member under leave-one-rep-out, the signature ceiling) of the three-level question
(SPEC_DETECTION.md section 3.4, builder A; P3 0a).

Corrections from the reviews of SPEC_DETECTION.md applied here (each marked "must change"):
  al-Kindi 2.3   the level-2 null statistic is per letter, never a macro average (P3 0a: "a level-2
                 headline is reported per sub-family, never averaged"); confusion.csv carries
                 null_p95, rank, verdict per row from its own letter's array; macro_recall stays in
                 scores.json as a number with no verdict;
  al-Kindi 2.4   at-floor cells leave the level-2 and level-3 denominators (K3 F3; ML 3.6); both
                 files carry n_at_floor; predictions.csv marks in_denominator; level 3 records
                 null_unit = "cell" with its reason;
  al-Farabi M2   TRAIN_ON_AT_FLOOR (detection_metrics) is honoured: at-floor cells leave every
                 training set unless the author sets it, and the value is in params;
  ML 2.1         the row unit is the cell; the majority vote over windows becomes the row's argmax.

Citation: P3 0a (levels 2 and 3; sub-family B "one member, no held-out test"; level 3 labelled the
signature ceiling); CR3 2.5 (G-N's counts); SPEC 4.2 (the forest); ML 1.4 (the null's unit).
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import models as M  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder import classes as C  # noqa: E402
from plan11_encoding_ladder import detection_splits as DS  # noqa: E402
from plan11_encoding_ladder import detection_metrics as DM  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST, SEED_LABEL_NULL  # noqa: E402

LEVEL2_MIN_MEMBERS_HEADLINE = 3      # G-N's headline count (CR3 2.5); P3 0a (A and C scored, B not)
LEVEL2_MIN_MEMBERS_TEST = 2          # 2 members: the row carries L2_ONE_TRAIN
LEVEL3_SPLIT = DS.LEVEL3_SPLIT       # section 7 item 17
LEVEL3_NULL_UNIT = "cell"
LEVEL3_NULL_UNIT_REASON = ("the label is the workload itself, so a workload-level permutation is a relabel and leaves accuracy "
                           "unchanged; the per-cell member-label vector is permuted across the sandbox cells (the weak null of ML 1.4; "
                           "the level is the signature ceiling anyway)")
CITATION_L2 = "P3 0a (level 2: which sub-family, one member held out, reported per sub-family, never averaged); CR3 2.5 (G-N); K3 F3 (the denominator)"
CITATION_L3 = "P3 0a (level 3: the exact member under leave-one-rep-out, labelled the signature ceiling); K3 F3 (the denominator); ML 1.4 (the null's unit)"
PRED_COLUMNS = ("cell_id", "member_index", "subfamily_letter", "rep", "fold", "y_true", "y_pred", "vote_fraction", "floor_verdict", "in_denominator")


def level_dir(out: Path, rung: str, grid_id: str, level: str) -> Path:
    return C.det_dir(out) / "splits" / rung / grid_id / level


def _multiset_permutations(letters: list[str], cap: int, rng, n_draw: int) -> tuple[list, bool, int]:
    """The distinct permutations of the letter multiset (exhaustive when their number is below ``cap``,
    else ``n_draw`` random permutations). Returns (list of tuples, exhaustive, n_distinct_total)."""
    counts = {}
    for l in letters:
        counts[l] = counts.get(l, 0) + 1
    total = math.factorial(len(letters))
    for c in counts.values():
        total //= math.factorial(c)
    if total < cap and total <= int(n_draw):
        seen = set()
        for p in itertools.permutations(letters):
            if p not in seen:
                seen.add(p)
        return sorted(seen), True, total
    arr = np.array(letters)
    return [tuple(arr[rng.permutation(len(arr))].tolist()) for _ in range(int(n_draw))], False, total


def _fit_predict(X, y_rows, folds, cell_ids, *, seed, n_jobs, n_estimators, rule="cell_majority") -> dict:
    """The multiclass forest of SPEC 4.2 (models.make_forest, oob off) per fold; the cell's prediction
    is models.aggregate_units over its rows (the row's argmax under cell rows). Returns {cell: {y_pred,
    vote_fraction, fold}}."""
    res = {}
    for f in folds:
        tr, te = np.asarray(f["train"], dtype=np.int64), np.asarray(f["test"], dtype=np.int64)
        if len(tr) == 0 or len(te) == 0 or len(set(y_rows[tr].tolist())) < 2:
            for c in dict.fromkeys(np.asarray(cell_ids)[te].tolist()):
                res[c] = {"y_pred": "", "vote_fraction": None, "fold": f["name"], "status": V.not_applicable("one class in training")}
            continue
        clf = M.make_forest(seed, n_jobs, n_estimators).fit(X[tr], y_rows[tr])
        proba = clf.predict_proba(X[te]); classes = list(clf.classes_)
        pred = np.array(classes)[np.argmax(proba, axis=1)]
        for c, (p, vf) in M.aggregate_units(np.asarray(cell_ids)[te], pred, proba, classes, rule).items():
            res[c] = {"y_pred": p, "vote_fraction": vf, "fold": f["name"], "status": "ok"}
    return res


def _drop_train_at_floor(folds: list[dict], lab: dict, train_on_at_floor: bool) -> list[dict]:
    if train_on_at_floor:
        return folds
    out = []
    for f in folds:
        tr = np.asarray(f["train"], dtype=np.int64)
        keep = ~np.isin(lab["floor_verdict"][tr], DM.FLOOR_VERDICTS_OUT)
        g = dict(f); g["train"] = tr[keep]; g["n_train_dropped_at_floor"] = int(len(tr) - keep.sum())
        out.append(g)
    return out


def _load(out: Path, rung: str, gid: str, normalized: bool) -> dict:
    data = DM.load_detection_data(out, rung, gid, normalized=normalized, mask_classes=("sandbox",))
    return data


def run_level2(out: Path, rung: str, grid_id: str | None = None, *, normalized: bool = True, n_perm: int = DM.N_PERM, seed: int = SEED_FOREST,
               seed_offset: int = 0, n_jobs: int = 1, n_estimators: int = DM.N_ESTIMATORS, min_members_headline: int = LEVEL2_MIN_MEMBERS_HEADLINE,
               min_members_test: int = LEVEL2_MIN_MEMBERS_TEST, train_on_at_floor: bool = DM.TRAIN_ON_AT_FLOOR, n_perm_required: int = DM.N_PERM) -> Path:
    """Level 2, which sub-family (P3 0a; G-N, CR3 2.5). Sandbox rows only (at-floor cells kept, marked,
    out of the denominator and, unless train_on_at_floor, out of training). The forest of SPEC 4.2
    (models.make_forest, oob off) as a multiclass model over the sub-family letters; folds =
    fold_level2; the cell's prediction is the argmax of its row (the majority vote over its windows
    under window rows). Writes splits/<rung>/<grid_id>/level2/: confusion.csv (rung, true_subfamily,
    n_members, n_cells, n_at_floor, n_in_denominator, pred_<letter> for every letter present, recall,
    null_p95, rank, verdict, status) where status is GN_HEADLINE when n_members >=
    min_members_headline, L2_ONE_TRAIN when n_members == min_members_test, level2_no_heldout(n) when
    n_members < min_members_test (the row's counts empty); members.csv (rung, member_index,
    subfamily_letter, hits, denominator, at_floor, eighths); predictions.csv; scores.json
    (macro_recall over the headline rows as a number with no verdict, n_folds, null); null.json.
    Null: the letters permuted across members with the counts kept (the multiset permutations;
    exhaustive when their number is below NULL_EXHAUSTIVE_BELOW; 8!/(4!1!3!) = 280 on stage 1), per
    permutation the recall of every headline letter separately (al-Kindi review 2.3), null_verdict
    per letter as 3.3.4 (NULL_NOT_ESTIMABLE below 20 distinct assignments). Citation: CITATION_L2."""
    out = Path(out)
    seed = int(seed) + int(seed_offset); seed_null = SEED_LABEL_NULL + int(seed_offset)
    gid, gsrc = DM._grid(out, rung, grid_id)
    params = {"rung": rung, "grid_id": gid, "grid_source": gsrc, "level": 2, "normalized": bool(normalized), "n_perm": int(n_perm), "seed": seed, "seed_null": seed_null,
              "seed_offset": int(seed_offset), "n_estimators": int(n_estimators), "min_members_headline": int(min_members_headline), "min_members_test": int(min_members_test),
              "train_on_at_floor": bool(train_on_at_floor), "row_unit": DS.ROW_UNIT, "null_statistic": "recall per letter (never a macro average)", "n_perm_required": int(n_perm_required),
              "excluded_by_declaration": list(DM.EXCLUDED_BY_DECLARATION)}
    d = level_dir(out, rung, gid or "no_selection", "level2"); d.mkdir(parents=True, exist_ok=True)
    conf_cols = ["rung", "true_subfamily", "n_members", "n_cells", "n_at_floor", "n_in_denominator"]
    if gid is None:
        S.write_json(d / "scores.json", "plan11.detection.level2.v1", params, CITATION_L2, {"status": V.not_run(f"no selection for {rung} (run classes inherit-selection)")})
        S.write_csv(d / "confusion.csv", conf_cols + ["recall", "null_p95", "rank", "verdict", "status"], [{"rung": rung, "status": V.not_run(f"no selection for {rung}")}])
        return d
    try:
        data = _load(out, rung, gid, normalized)
    except FileNotFoundError as e:
        S.write_json(d / "scores.json", "plan11.detection.level2.v1", params, CITATION_L2, {"status": V.not_run(f"input missing: {Path(str(e)).name}")})
        return d
    lab, X = data["lab"], data["X"]
    params["inputs_sha256"] = S.inputs_sha256(data["inputs"], out)
    letters = lab["subfamily_letter"].astype(str); members = lab["member_index"]
    member_letter = {int(m): str(letters[members == m][0]) for m in dict.fromkeys(members.tolist())}
    present = sorted(set(member_letter.values()))
    n_members_of = {l: sum(1 for m, ll in member_letter.items() if ll == l) for l in present}
    headline = [l for l in present if n_members_of[l] >= int(min_members_headline)]
    cols = conf_cols + [f"pred_{l}" for l in present] + ["recall", "null_p95", "rank", "verdict", "status"]
    in_den = ~np.isin(lab["floor_verdict"], DM.FLOOR_VERDICTS_OUT)

    def score_letters(letter_of_member: dict) -> tuple[dict, dict]:
        """Per letter the recall over in-denominator cells under the folds of that assignment; returns (recalls, predictions)."""
        lab2 = dict(lab); lab2["subfamily_letter"] = np.array([letter_of_member[int(m)] for m in members], dtype=str)
        folds = _drop_train_at_floor(DS.fold_level2(lab2, min_members_test=min_members_test), lab2, train_on_at_floor)
        preds = _fit_predict(X, lab2["subfamily_letter"], folds, lab2["cell_id"], seed=seed, n_jobs=n_jobs, n_estimators=n_estimators)
        rec = {}
        for l in present:
            idx = [i for i in range(lab["n"]) if lab2["subfamily_letter"][i] == l and in_den[i] and str(lab["cell_id"][i]) in preds and preds[str(lab["cell_id"][i])]["status"] == "ok"]
            rec[l] = (float(np.mean([preds[str(lab["cell_id"][i])]["y_pred"] == l for i in idx])) if idx else None)
        return rec, preds
    obs, preds = score_letters(member_letter)
    # the null: letters permuted across members with the counts kept
    rng = np.random.default_rng(seed_null)
    ms = sorted(member_letter)
    perms, exhaustive, n_assign = _multiset_permutations([member_letter[m] for m in ms], DM.NULL_EXHAUSTIVE_BELOW, rng, int(n_perm)) if int(n_perm) > 0 else ([], False, 0)
    if int(n_perm) == 0:
        _, _, n_assign = _multiset_permutations([member_letter[m] for m in ms], 1, rng, 0)
    null_arrays = {l: [] for l in present}
    for pm in perms:
        rec, _ = score_letters({m: pm[i] for i, m in enumerate(ms)})
        for l in present:
            null_arrays[l].append(rec[l] if rec[l] is not None else np.nan)
    rows, mrows = [], []
    for l in present:
        n_m = n_members_of[l]
        cells_l = [i for i in range(lab["n"]) if letters[i] == l]
        row = {"rung": rung, "true_subfamily": l, "n_members": n_m, "n_cells": len(cells_l), "n_at_floor": int(sum(1 for i in cells_l if not in_den[i])),
               "n_in_denominator": int(sum(1 for i in cells_l if in_den[i]))}
        if n_m < int(min_members_test):
            row.update({"recall": "", "null_p95": "", "rank": "", "verdict": V.level2_no_heldout(n_m), "status": V.level2_no_heldout(n_m)})
            for l2 in present:
                row[f"pred_{l2}"] = ""
        else:
            for l2 in present:
                row[f"pred_{l2}"] = int(sum(1 for i in cells_l if in_den[i] and preds.get(str(lab["cell_id"][i]), {}).get("y_pred") == l2))
            row["recall"] = obs[l]
            row["status"] = V.GN_HEADLINE if n_m >= int(min_members_headline) else V.L2_ONE_TRAIN
            if perms:
                verdict, summ = DM.null_verdict(obs[l], np.asarray(null_arrays[l], dtype=np.float64), n_assign, exhaustive=exhaustive, n_perm_required=n_perm_required)
                row.update({"null_p95": summ.get("p95"), "rank": summ.get("rank_text", ""), "verdict": verdict})
            else:
                row.update({"null_p95": "", "rank": "", "verdict": V.not_run("null not requested")})
        rows.append(row)
    for m in ms:
        idx = [i for i in range(lab["n"]) if int(members[i]) == m]
        den = [i for i in idx if in_den[i] and preds.get(str(lab["cell_id"][i]), {}).get("status") == "ok"]
        hits = sum(1 for i in den if preds[str(lab["cell_id"][i])]["y_pred"] == member_letter[m])
        mrows.append({"rung": rung, "member_index": m, "subfamily_letter": member_letter[m], "hits": hits, "denominator": len(den), "at_floor": int(sum(1 for i in idx if not in_den[i])),
                      "eighths": f"{hits}/{len(den)}"})
    prows = []
    for i in range(lab["n"]):
        c = str(lab["cell_id"][i]); pr = preds.get(c, {})
        prows.append({"cell_id": c, "member_index": int(members[i]), "subfamily_letter": letters[i], "rep": int(lab["rep"][i]), "fold": pr.get("fold", ""), "y_true": letters[i],
                      "y_pred": pr.get("y_pred", pr.get("status", "")), "vote_fraction": pr.get("vote_fraction"), "floor_verdict": str(lab["floor_verdict"][i]), "in_denominator": bool(in_den[i])})
    S.write_csv(d / "confusion.csv", cols, rows)
    S.write_csv(d / "members.csv", ("rung", "member_index", "subfamily_letter", "hits", "denominator", "at_floor", "eighths"), mrows)
    S.write_csv(d / "predictions.csv", PRED_COLUMNS, prows)
    hl = [obs[l] for l in headline if obs[l] is not None]
    payload = {"status": "ok" if headline else V.not_applicable("no headline sub-family"), "macro_recall": (float(np.mean(hl)) if hl else V.not_applicable("no headline sub-family")),
               "macro_recall_note": "a number with no verdict; the headline is per sub-family (P3 0a)", "headline_letters": headline, "recall_per_letter": obs,
               "n_folds": len(DS.fold_level2(lab, min_members_test=min_members_test)), "members": member_letter, "n_members_per_letter": n_members_of,
               "null": {"n_assignments": n_assign, "exhaustive": exhaustive, "n_perm": len(perms), "per_letter": {l: {k: v for k, v in DM.null_verdict(obs[l], np.asarray(null_arrays[l], dtype=np.float64), n_assign, exhaustive=exhaustive, n_perm_required=n_perm_required)[1].items() if k in ("p95", "p05", "rank", "rank_text", "n", "verdict")} for l in present if perms and n_members_of[l] >= int(min_members_test)}},
               "n_at_floor": int((~in_den).sum()), "grid_source": gsrc, "seed": seed, "feature_count": int(X.shape[1]), "row_unit": DS.ROW_UNIT}
    S.write_json(d / "scores.json", "plan11.detection.level2.v1", params, CITATION_L2, payload)
    S.write_json(d / "null.json", "plan11.detection.level2_null.v1", params, CITATION_L2, {"n_assignments": n_assign, "exhaustive": exhaustive, "n_perm": len(perms),
                                                                                           "arrays": {l: [None if (isinstance(v, float) and np.isnan(v)) else v for v in null_arrays[l]] for l in present}})
    return d


def run_level3(out: Path, rung: str, grid_id: str | None = None, *, normalized: bool = True, split: str = LEVEL3_SPLIT, n_perm: int = DM.N_PERM, seed: int = SEED_FOREST,
               seed_offset: int = 0, n_jobs: int = 1, n_estimators: int = DM.N_ESTIMATORS, train_on_at_floor: bool = DM.TRAIN_ON_AT_FLOOR, n_perm_required: int = DM.N_PERM) -> Path:
    """Level 3, the exact member under leave-one-rep-out (P3 0a), labelled SIGNATURE_CEILING on every
    row. The same multiclass forest over member indices; folds = fold_level3(split); confusion.csv
    (rung, true_member, n_cells, n_at_floor, n_in_denominator, pred_<m> for every member, recall,
    label = SIGNATURE_CEILING); members.csv as level 2; predictions.csv; scores.json (accuracy over
    the in-denominator cells, per-member recall, null); null = the per-cell member-label vector
    permuted across sandbox cells (null_unit = "cell", LEVEL3_NULL_UNIT_REASON), n_perm permutations,
    the statistic = accuracy. With one member the file reads not_applicable('one member'). Citation: CITATION_L3."""
    out = Path(out)
    seed = int(seed) + int(seed_offset); seed_null = SEED_LABEL_NULL + int(seed_offset)
    gid, gsrc = DM._grid(out, rung, grid_id)
    params = {"rung": rung, "grid_id": gid, "grid_source": gsrc, "level": 3, "split": split, "normalized": bool(normalized), "n_perm": int(n_perm), "seed": seed, "seed_null": seed_null,
              "seed_offset": int(seed_offset), "n_estimators": int(n_estimators), "train_on_at_floor": bool(train_on_at_floor), "row_unit": DS.ROW_UNIT, "null_unit": LEVEL3_NULL_UNIT,
              "null_unit_reason": LEVEL3_NULL_UNIT_REASON, "label": V.SIGNATURE_CEILING, "n_perm_required": int(n_perm_required), "excluded_by_declaration": list(DM.EXCLUDED_BY_DECLARATION)}
    d = level_dir(out, rung, gid or "no_selection", "level3"); d.mkdir(parents=True, exist_ok=True)
    base_cols = ["rung", "true_member", "n_cells", "n_at_floor", "n_in_denominator"]
    if gid is None:
        S.write_json(d / "scores.json", "plan11.detection.level3.v1", params, CITATION_L3, {"status": V.not_run(f"no selection for {rung} (run classes inherit-selection)")})
        S.write_csv(d / "confusion.csv", base_cols + ["recall", "label"], [{"rung": rung, "label": V.not_run(f"no selection for {rung}")}])
        return d
    try:
        data = _load(out, rung, gid, normalized)
    except FileNotFoundError as e:
        S.write_json(d / "scores.json", "plan11.detection.level3.v1", params, CITATION_L3, {"status": V.not_run(f"input missing: {Path(str(e)).name}")})
        return d
    lab, X = data["lab"], data["X"]
    params["inputs_sha256"] = S.inputs_sha256(data["inputs"], out)
    members = lab["member_index"]; ms = sorted(set(int(m) for m in members.tolist()))
    cols = base_cols + [f"pred_{m}" for m in ms] + ["recall", "label"]
    if len(ms) < 2:
        S.write_json(d / "scores.json", "plan11.detection.level3.v1", params, CITATION_L3, {"status": V.not_applicable("one member"), "n_members": len(ms)})
        S.write_csv(d / "confusion.csv", cols, [{"rung": rung, "true_member": m, "label": V.not_applicable("one member")} for m in ms])
        return d
    y = members.astype(str)
    in_den = ~np.isin(lab["floor_verdict"], DM.FLOOR_VERDICTS_OUT)
    folds = _drop_train_at_floor(DS.fold_level3(lab, split), lab, train_on_at_floor)

    def accuracy_of(y_rows):
        preds = _fit_predict(X, y_rows, folds, lab["cell_id"], seed=seed, n_jobs=n_jobs, n_estimators=n_estimators)
        idx = [i for i in range(lab["n"]) if in_den[i] and preds.get(str(lab["cell_id"][i]), {}).get("status") == "ok"]
        acc = float(np.mean([preds[str(lab["cell_id"][i])]["y_pred"] == y_rows[i] for i in idx])) if idx else None
        return acc, preds
    acc, preds = accuracy_of(y)
    rng = np.random.default_rng(seed_null)
    null = []
    for _ in range(int(n_perm)):
        a, _ = accuracy_of(y[rng.permutation(lab["n"])])
        null.append(a if a is not None else np.nan)
    n_assign = math.factorial(lab["n"])
    for m in ms:
        n_assign //= math.factorial(int(np.sum(members == m)))
    verdict, summ = DM.null_verdict(acc, np.asarray(null, dtype=np.float64), min(n_assign, 10 ** 9), exhaustive=False, n_perm_required=n_perm_required) if n_perm > 0 else (V.not_run("null not requested"), {})
    rows, mrows = [], []
    for m in ms:
        idx = [i for i in range(lab["n"]) if int(members[i]) == m]
        den = [i for i in idx if in_den[i] and preds.get(str(lab["cell_id"][i]), {}).get("status") == "ok"]
        row = {"rung": rung, "true_member": m, "n_cells": len(idx), "n_at_floor": int(sum(1 for i in idx if not in_den[i])), "n_in_denominator": len(den), "label": V.SIGNATURE_CEILING}
        for m2 in ms:
            row[f"pred_{m2}"] = int(sum(1 for i in den if preds[str(lab["cell_id"][i])]["y_pred"] == str(m2)))
        hits = row[f"pred_{m}"]
        row["recall"] = (hits / len(den)) if den else None
        rows.append(row)
        mrows.append({"rung": rung, "member_index": m, "subfamily_letter": str(lab["subfamily_letter"][idx[0]]), "hits": hits, "denominator": len(den),
                      "at_floor": row["n_at_floor"], "eighths": f"{hits}/{len(den)}"})
    prows = []
    for i in range(lab["n"]):
        c = str(lab["cell_id"][i]); pr = preds.get(c, {})
        prows.append({"cell_id": c, "member_index": int(members[i]), "subfamily_letter": str(lab["subfamily_letter"][i]), "rep": int(lab["rep"][i]), "fold": pr.get("fold", ""), "y_true": y[i],
                      "y_pred": pr.get("y_pred", pr.get("status", "")), "vote_fraction": pr.get("vote_fraction"), "floor_verdict": str(lab["floor_verdict"][i]), "in_denominator": bool(in_den[i])})
    S.write_csv(d / "confusion.csv", cols, rows)
    S.write_csv(d / "members.csv", ("rung", "member_index", "subfamily_letter", "hits", "denominator", "at_floor", "eighths"), mrows)
    S.write_csv(d / "predictions.csv", PRED_COLUMNS, prows)
    payload = {"status": "ok", "accuracy": acc, "label": V.SIGNATURE_CEILING, "per_member_recall": {str(r["true_member"]): r["recall"] for r in rows}, "n_folds": len(folds),
               "n_members": len(ms), "n_at_floor": int((~in_den).sum()), "null": {"statistic": "accuracy", "null_unit": LEVEL3_NULL_UNIT, "null_unit_reason": LEVEL3_NULL_UNIT_REASON,
                                                                                   "n_perm": len(null), "n_assignments": n_assign, "verdict": verdict,
                                                                                   **{k: summ.get(k) for k in ("p95", "p05", "rank", "rank_text", "n", "exceeds")}},
               "grid_source": gsrc, "seed": seed, "feature_count": int(X.shape[1]), "row_unit": DS.ROW_UNIT}
    S.write_json(d / "scores.json", "plan11.detection.level3.v1", params, CITATION_L3, payload)
    S.write_json(d / "null.json", "plan11.detection.level3_null.v1", params, CITATION_L3, {"n_perm": len(null), "n_assignments": n_assign, "arrays": {"accuracy": [None if (isinstance(v, float) and np.isnan(v)) else v for v in null]}})
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="detection_levels.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("level2", "level3"):
        a = sub.add_parser(name)
        a.add_argument("--out", required=True); a.add_argument("--rung", required=True, choices=S.RUNGS); a.add_argument("--grid-id", default=None)
        a.add_argument("--null-perm", type=int, default=DM.N_PERM); a.add_argument("--n-jobs", type=int, default=1); a.add_argument("--seed-offset", type=int, default=0)
        a.add_argument("--n-estimators", type=int, default=DM.N_ESTIMATORS); a.add_argument("--train-on-at-floor", default="true" if DM.TRAIN_ON_AT_FLOOR else "false", choices=("true", "false"))
        a.add_argument("--n-perm-required", type=int, default=DM.N_PERM)
        if name == "level3":
            a.add_argument("--split", default=LEVEL3_SPLIT, choices=("rep_index", "cell"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr); return 2
    try:
        gid = args.grid_id or S.selected_grid_id(out, args.rung, None)[0]
        if args.cmd == "level2":
            print(run_level2(out, args.rung, gid, n_perm=args.null_perm, n_jobs=args.n_jobs, seed_offset=args.seed_offset, n_estimators=args.n_estimators,
                             train_on_at_floor=(args.train_on_at_floor == "true"), n_perm_required=args.n_perm_required))
        else:
            print(run_level3(out, args.rung, gid, split=args.split, n_perm=args.null_perm, n_jobs=args.n_jobs, seed_offset=args.seed_offset, n_estimators=args.n_estimators,
                             train_on_at_floor=(args.train_on_at_floor == "true"), n_perm_required=args.n_perm_required))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
