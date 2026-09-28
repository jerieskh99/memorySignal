#!/usr/bin/env python3
"""detection_splits.py -- the detection label dict and the folds of the detection layer
(SPEC_DETECTION.md section 3.2, builder A): LOWO over both classes, LOCO, LOFO, the one-class
folds, the level-2 and level-3 folds, the order-test folds and the idle anchor folds. Every fold
list is passed through ``splits._assert_grouped`` on ``cell_id`` and, where the fold groups by
workload, on ``workload_key``.

Unit = cell (CR3 1.5; ML 1.1): whole cells are held out; no cell's windows straddle train and
test. The row unit is the cell (``ROW_UNIT = "cell"``, SPEC_DETECTION_review_ml.md 2.1): the loaded
feature file (one row per window) is collapsed to one row per admissible classed cell by
``models.cell_vectors``' rule (the nanmean over the cell's windows), so the learner sees one row
per cell per rung (ML 1.1; K3 2.2 "windows never become rows"). ``ROW_UNIT = "window"`` is the
declared alternative for the author (section 7).

Citation: ML 1.1, 1.3; CR3 1.5; K3 2.2; P3 D5, P3 0a; SPEC 4.1 (``splits.py``).
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import splits as SP  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder.splits import _assert_grouped, _groups  # noqa: E402

ROW_UNIT = "cell"                 # SPEC_DETECTION_review_ml.md 2.1; alternative "window"
CELL_VECTOR = "nanmean_over_windows"   # models.cell_vectors' rule (ML review, for the author item 2)
LOCO_MODE = "cell"                # section 7 item 11; alternative "rep_index"
LEVEL3_SPLIT = "rep_index"        # section 7 item 17
ORDER_HALF_RULE = "within_workload"   # al-Kindi 2.11; the ML-literal alternative "within_class"
PENDING_GK0 = V.pending("gk0 not run")
CITATION = "ML 1.1, 1.3; CR3 1.5; K3 2.2; P3 D5, P3 0a; SPEC_DETECTION.md 3.2"

_STR_KEYS = ("cell_id", "y", "cls", "workload_key", "family", "subfamily_letter", "campaign", "split_role",
             "floor_verdict", "kernel", "archetype")
_INT_KEYS = ("member_index", "rep", "order_index", "win_start", "n_windows")


def make_detection_labels(feat: dict, join: list[dict], admissible, *, mask_classes=None, row_unit: str = ROW_UNIT,
                          floor_by_cell: dict | None = None) -> dict:
    """The label dict (SPEC_DETECTION 3.2.1) from a loaded feature file (``series.load_features``), the
    join rows (``classes.load_join``) and the admissible cell ids: rows restricted to admissible,
    classed cells (and to ``mask_classes`` when given). Under ``row_unit = "cell"`` the rows are
    collapsed to one per cell (the nanmean of the window features; ``n_windows`` counts them,
    ``win_start`` is -1). Keys per row: ``X`` (the feature rows), n, _rows, cell_id, y (sandbox |
    benign), cls, workload_key, family, member_index (0 for a non-member), subfamily_letter (- for a
    non-member), rep, campaign, order_index (-1 when missing), split_role, win_start, n_windows,
    floor_verdict (from ``floor_by_cell`` when given, else ``pending: gk0 not run``), plus the
    aliases ``kernel`` (= workload_key) and ``archetype`` (= cls) so ``splits.fold_loro``,
    ``fold_loro_rep_index`` and ``_assert_grouped`` run unchanged. ``_cells_without_rows`` lists the
    admissible classed cells absent from the feature file (al-Farabi M4: they are scored as
    ``not applicable: no finite window`` and never counted as a miss). Citation: ML 1.1; CR3 1.5."""
    if row_unit not in ("cell", "window"):
        raise ValueError(row_unit)
    adm = set(admissible) if admissible is not None else None
    jmap = {j["cell_id"]: j for j in join if j.get("class") and j["class"] != "unassigned"}
    if mask_classes is not None:
        keep_cls = set(mask_classes)
        jmap = {k: j for k, j in jmap.items() if j["class"] in keep_cls}
    ids = np.asarray(feat["cell_id"]).astype(str)
    X = np.asarray(feat["X"], dtype=np.float64)
    keep = np.array([(c in jmap) and (adm is None or c in adm) for c in ids], dtype=bool)
    idx = np.flatnonzero(keep)
    fb = floor_by_cell or {}

    def meta_of(c: str, k: int, ws: int) -> dict:
        j = jmap[c]
        oi = j.get("order_index", -1)
        oi = int(oi) if oi not in ("", None) else -1
        return {"cell_id": c, "y": j["y"], "cls": j["class"], "workload_key": j["workload_key"], "family": j["family"],
                "member_index": int(j.get("member_index") or 0), "subfamily_letter": j.get("subfamily_letter") or "-",
                "rep": int(j.get("rep") or 0), "campaign": str(j.get("campaign") or ""), "order_index": oi,
                "split_role": j.get("split_role") or "train_test", "win_start": ws, "n_windows": k,
                "floor_verdict": fb.get(c, PENDING_GK0), "kernel": j["workload_key"], "archetype": j["class"]}

    rows_meta: list[dict] = []
    if row_unit == "cell":
        Xs = []
        for c, rows in _groups(ids[idx]).items():
            r = idx[rows]
            with np.errstate(all="ignore"):
                Xs.append(np.nanmean(X[r], axis=0))
            rows_meta.append(meta_of(c, len(r), -1))
        Xc = np.stack(Xs) if Xs else np.zeros((0, X.shape[1]))
        lab = {"X": Xc, "n": len(rows_meta), "_rows": np.arange(len(rows_meta)), "row_unit": "cell"}
    else:
        ws = np.asarray(feat["win_start"]).astype(int)
        counts = {c: len(rows) for c, rows in _groups(ids[idx]).items()}
        for i in idx:
            rows_meta.append(meta_of(ids[i], counts[ids[i]], int(ws[i])))
        lab = {"X": X[idx], "n": len(rows_meta), "_rows": idx, "row_unit": "window"}
    for k in _STR_KEYS:
        lab[k] = np.array([m[k] for m in rows_meta], dtype=str) if rows_meta else np.zeros(0, dtype=str)
    for k in _INT_KEYS:
        lab[k] = np.array([m[k] for m in rows_meta], dtype=np.int64) if rows_meta else np.zeros(0, dtype=np.int64)
    present = set(ids[idx].tolist())
    lab["_cells_without_rows"] = sorted(c for c in jmap if (adm is None or c in adm) and c not in present)
    lab["cell_vector"] = CELL_VECTOR
    return lab


def cells_of(lab: dict) -> list[str]:
    return list(dict.fromkeys(lab["cell_id"].tolist()))


def _rows_where(lab: dict, mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def _train_test_mask(lab: dict) -> np.ndarray:
    return lab["split_role"] == "train_test"


def _final_fold(lab: dict, name: str, train_mask: np.ndarray) -> dict | None:
    test = _rows_where(lab, lab["split_role"] == "test_only")
    if len(test) == 0:
        return None
    return {"name": name, "held_out": "external", "y_held": "sandbox", "family": "external", "member_index": 0,
            "train": _rows_where(lab, train_mask), "test": test}


def fold_lowo(lab: dict) -> list[dict]:
    """Leave-one-workload-out over all workloads of both classes (ML 1.3; CR3 1.5; K3 2.2; P3 D5).
    One fold per workload_key among rows with split_role == "train_test"; the fold holds out every
    cell of that workload (all reps) and trains on every other train_test cell of both classes.
    Sandbox folds give the true-positive rate, benign folds the false-positive rate. Fold dict:
    name "lowo/<workload_key>", held_out, y_held (sandbox | benign), family, member_index, train,
    test. If any test_only cells exist, one extra fold {name: "lowo/final", held_out: "external",
    train: every train_test row, test: every test_only row} is appended (P3 0a stage 3). Asserted
    grouped on workload_key and cell_id."""
    tt = _train_test_mask(lab)
    folds = []
    for wk, idx in _groups(lab["workload_key"]).items():
        idx = idx[tt[idx]]
        if len(idx) == 0:
            continue
        test = idx
        train = _rows_where(lab, tt & (lab["workload_key"] != wk))
        folds.append({"name": f"lowo/{wk}", "held_out": wk, "y_held": str(lab["y"][idx[0]]), "family": str(lab["family"][idx[0]]),
                      "member_index": int(lab["member_index"][idx[0]]), "train": train, "test": test})
    fin = _final_fold(lab, "lowo/final", tt)
    if fin is not None:
        folds.append(fin)
    _assert_grouped(folds, lab, "workload_key")
    _assert_grouped(folds, lab, "cell_id")
    return folds


def fold_loco(lab: dict, mode: str = LOCO_MODE) -> list[dict]:
    """Leave-one-cell-out, the signature ceiling (ML 1.3; CR3 1.5; K3 F2): mode "cell" is
    splits.fold_loro on the detection labels (one fold per cell; the model trains on the sibling
    reps); mode "rep_index" is splits.fold_loro_rep_index (one fold per rep index; section 7 item
    11). test_only rows are never in train and are appended as the "loco/final" fold as in
    fold_lowo. Asserted grouped on cell_id."""
    tt = _train_test_mask(lab)
    sub_idx = _rows_where(lab, tt)
    sub = {k: (lab[k][sub_idx] if isinstance(lab[k], np.ndarray) and lab[k].shape[:1] == (lab["n"],) else lab[k]) for k in lab}
    sub["n"] = len(sub_idx)
    raw = SP.fold_loro(sub) if mode == "cell" else SP.fold_loro_rep_index(sub)
    folds = []
    for f in raw:
        te = sub_idx[f["test"]]
        folds.append({"name": f["name"].replace("loro/", "loco/"), "held_out": f["held_out"],
                      "y_held": str(lab["y"][te[0]]), "family": str(lab["family"][te[0]]),
                      "member_index": int(lab["member_index"][te[0]]), "train": sub_idx[f["train"]], "test": te})
    fin = _final_fold(lab, "loco/final", tt)
    if fin is not None:
        folds.append(fin)
    _assert_grouped(folds, lab, "cell_id")
    return folds


def fold_lofo(lab: dict) -> list[dict]:
    """Leave-one-benign-family-out (K3 2.2, LOFO; CR3 1.5; K3 F8): one fold per benign family (the
    family key of the benign rows); the fold holds out every cell of that family and trains on
    every other train_test cell of both classes; sandbox cells are in every training set, so LOFO
    yields a false-positive rate per unseen family and no true-positive rate. Returns [] when no
    benign family exists. Asserted grouped on family and cell_id."""
    tt = _train_test_mask(lab)
    folds = []
    for fam, idx in _groups(lab["family"]).items():
        idx = idx[tt[idx] & (lab["y"][idx] == "benign")]
        if len(idx) == 0:
            continue
        train = _rows_where(lab, tt & (lab["family"] != fam))
        folds.append({"name": f"lofo/{fam}", "held_out": fam, "y_held": "benign", "family": fam, "member_index": 0,
                      "train": train, "test": idx})
    _assert_grouped(folds, lab, "family")
    _assert_grouped(folds, lab, "cell_id")
    return folds


def fold_one_class(lab: dict) -> list[dict]:
    """The one-class reading (ML 1.3; K3 2.2; CR3 2.19): benign-only training. One fold per benign
    workload_key (train = every other benign train_test cell; test = that workload's cells) for the
    false-positive rate, then the fold "one_class/final" (train = every benign train_test cell;
    test = every sandbox cell and every test_only cell), so every sandbox cell is scored by a model
    that never saw a sandbox cell. Asserted grouped on workload_key and cell_id."""
    tt = _train_test_mask(lab)
    ben = tt & (lab["y"] == "benign")
    folds = []
    for wk, idx in _groups(lab["workload_key"]).items():
        idx = idx[ben[idx]]
        if len(idx) == 0:
            continue
        train = _rows_where(lab, ben & (lab["workload_key"] != wk))
        folds.append({"name": f"one_class/{wk}", "held_out": wk, "y_held": "benign", "family": str(lab["family"][idx[0]]),
                      "member_index": 0, "train": train, "test": idx})
    pos = _rows_where(lab, lab["y"] == "sandbox")
    if len(pos):
        folds.append({"name": "one_class/final", "held_out": "sandbox", "y_held": "sandbox", "family": "sandbox", "member_index": 0,
                      "train": _rows_where(lab, ben), "test": pos})
    _assert_grouped(folds, lab, "workload_key")
    _assert_grouped(folds, lab, "cell_id")
    return folds


def fold_level2(lab: dict, *, min_members_test: int = 2) -> list[dict]:
    """Level 2 (P3 0a; G-N): rows restricted to class sandbox. One fold per member whose sub-family
    has at least min_members_test members (test = that member's cells; train = every other sandbox
    cell of every sub-family). A sub-family with fewer members has no fold and is reported by
    detection_levels as level2_no_heldout(n). Asserted grouped on workload_key."""
    sb = lab["cls"] == "sandbox"
    members_of: dict = {}
    for m, idx in _groups(lab["member_index"]).items():
        idx = idx[sb[idx]]
        if len(idx):
            members_of.setdefault(str(lab["subfamily_letter"][idx[0]]), []).append(int(m))
    folds = []
    for m, idx in _groups(lab["member_index"]).items():
        idx = idx[sb[idx]]
        if len(idx) == 0:
            continue
        letter = str(lab["subfamily_letter"][idx[0]])
        if len(members_of[letter]) < int(min_members_test):
            continue
        train = _rows_where(lab, sb & (lab["member_index"] != m))
        folds.append({"name": f"level2/member_{int(m)}", "held_out": f"sandbox_member_{int(m)}", "y_held": letter,
                      "family": "sandbox", "member_index": int(m), "subfamily_letter": letter, "train": train, "test": idx})
    _assert_grouped(folds, lab, "workload_key")
    _assert_grouped(folds, lab, "cell_id")
    return folds


def fold_level3(lab: dict, mode: str = LEVEL3_SPLIT) -> list[dict]:
    """Level 3 (P3 0a): rows restricted to class sandbox; mode "rep_index": one fold per rep index,
    holding out that rep of every member (the leave-one-rep-out reading); mode "cell": one fold per
    cell. Labelled SIGNATURE_CEILING by the caller. Asserted grouped on cell_id."""
    sb = lab["cls"] == "sandbox"
    folds = []
    if mode == "rep_index":
        for r, idx in _groups(lab["rep"]).items():
            idx = idx[sb[idx]]
            if len(idx) == 0:
                continue
            folds.append({"name": f"level3/rep{int(r):02d}", "held_out": f"rep{int(r):02d}", "y_held": "", "family": "sandbox",
                          "member_index": 0, "train": _rows_where(lab, sb & (lab["rep"] != r)), "test": idx})
    elif mode == "cell":
        for c, idx in _groups(lab["cell_id"]).items():
            idx = idx[sb[idx]]
            if len(idx) == 0:
                continue
            folds.append({"name": f"level3/{c}", "held_out": c, "y_held": "", "family": "sandbox",
                          "member_index": int(lab["member_index"][idx[0]]), "train": _rows_where(lab, sb & (lab["cell_id"] != c)), "test": idx})
    else:
        raise ValueError(mode)
    _assert_grouped(folds, lab, "cell_id")
    return folds


def order_half_labels(lab: dict, cls: str, half_rule: str = ORDER_HALF_RULE) -> np.ndarray:
    """The half label of the order test (ML 3.2; CR3 2.20; al-Kindi 2.11): an array over the rows,
    "first" | "second" for rows of class ``cls`` with ``order_index >= 1`` and "" elsewhere.
    ``within_class``: the rank of order_index within the class, the first floor(n / 2) cells are
    "first" (the ML-literal rule). ``within_workload``: the first floor(k / 2) cells of each
    workload by realized order are "first" (the default when a class is blocked by workload:
    position once identity is stripped)."""
    half = np.array([""] * lab["n"], dtype=object)
    in_cls = np.ones(lab["n"], dtype=bool) if cls == "all" else (lab["cls"] == cls)
    m = in_cls & (lab["order_index"] >= 1)
    idx = _rows_where(lab, m)
    if len(idx) == 0:
        return half.astype(str)
    if half_rule == "within_class":
        order = idx[np.argsort(lab["order_index"][idx], kind="stable")]
        k = len(order) // 2
        half[order[:k]] = "first"
        half[order[k:]] = "second"
    elif half_rule == "within_workload":
        for wk, rows in _groups(lab["workload_key"][idx]).items():
            r = idx[rows]
            order = r[np.argsort(lab["order_index"][r], kind="stable")]
            k = len(order) // 2
            half[order[:k]] = "first"
            half[order[k:]] = "second"
    else:
        raise ValueError(half_rule)
    return half.astype(str)


def fold_order(lab: dict, cls: str, half_rule: str = ORDER_HALF_RULE) -> list[dict]:
    """The order test (ML 3.2; CR3 2.20; P3 0a): rows restricted to class cls (``"all"``: every classed
    row, the campaign-wide scope of al-Farabi M12) with order_index >= 1;
    LOWO folds within the class (one per workload_key), the label is the half (order_half_labels).
    For a class with one workload (idle) the folds are leave-one-cell-out instead and the fold dict
    carries unit = "cell". Under half_rule "within_workload" a workload with fewer than two cells has
    no second half and is skipped. Asserted grouped on cell_id (and workload_key when LOWO)."""
    half = order_half_labels(lab, cls, half_rule)
    in_cls = np.ones(lab["n"], dtype=bool) if cls == "all" else (lab["cls"] == cls)
    m = in_cls & (lab["order_index"] >= 1) & (half != "")
    idx = _rows_where(lab, m)
    folds = []
    if len(idx) == 0:
        return folds
    wks = list(dict.fromkeys(lab["workload_key"][idx].tolist()))
    if len(wks) >= 2:
        for wk in wks:
            test = idx[lab["workload_key"][idx] == wk]
            train = idx[lab["workload_key"][idx] != wk]
            folds.append({"name": f"order_{cls}/{wk}", "held_out": wk, "unit": "workload", "train": train, "test": test})
        _assert_grouped(folds, lab, "workload_key")
    else:
        for c in dict.fromkeys(lab["cell_id"][idx].tolist()):
            test = idx[lab["cell_id"][idx] == c]
            train = idx[lab["cell_id"][idx] != c]
            folds.append({"name": f"order_{cls}/{c}", "held_out": c, "unit": "cell", "train": train, "test": test})
    _assert_grouped(folds, lab, "cell_id")
    return folds


def fold_anchor_idle(lab: dict, cls: str = "idle") -> list[dict]:
    """G-ANCHOR part (ii) (ML 3.1; CR3 2.14): rows of class idle (harness_idle apart, by ``cls``),
    leave-one-cell-out, label = campaign. Asserted grouped on cell_id."""
    idx = _rows_where(lab, lab["cls"] == cls)
    folds = []
    for c in dict.fromkeys(lab["cell_id"][idx].tolist()):
        test = idx[lab["cell_id"][idx] == c]
        train = idx[lab["cell_id"][idx] != c]
        folds.append({"name": f"anchor_{cls}/{c}", "held_out": c, "unit": "cell", "train": train, "test": test})
    _assert_grouped(folds, lab, "cell_id")
    return folds


def folds_for(split: str, lab: dict, *, loco_mode: str = LOCO_MODE, level3_split: str = LEVEL3_SPLIT,
              min_members_test: int = 2, order_class: str | None = None, half_rule: str = ORDER_HALF_RULE) -> list[dict]:
    """The fold list of one split name (lowo | loco | lofo | one_class | level2 | level3 | order | anchor_idle)."""
    if split == "lowo":
        return fold_lowo(lab)
    if split == "loco":
        return fold_loco(lab, loco_mode)
    if split == "lofo":
        return fold_lofo(lab)
    if split == "one_class":
        return fold_one_class(lab)
    if split == "level2":
        return fold_level2(lab, min_members_test=min_members_test)
    if split == "level3":
        return fold_level3(lab, level3_split)
    if split == "order":
        return fold_order(lab, order_class or "sandbox", half_rule)
    if split == "anchor_idle":
        return fold_anchor_idle(lab)
    raise ValueError(split)


def sublab(lab: dict, mask: np.ndarray) -> dict:
    """The label dict restricted to ``mask`` rows (arrays sliced, ``X`` included, ``n`` updated)."""
    idx = _rows_where(lab, mask)
    out = {}
    for k, v in lab.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == lab["n"]:
            out[k] = v[idx]
        else:
            out[k] = v
    out["n"] = len(idx)
    out["_rows"] = np.asarray(lab["_rows"])[idx] if len(idx) else np.zeros(0, dtype=np.int64)
    return out
