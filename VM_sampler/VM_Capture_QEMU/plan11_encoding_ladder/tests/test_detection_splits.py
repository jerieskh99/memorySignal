"""detection_splits.py: the label dict, the folds and the grouping assertion, on hand-built label
dicts (no corpus) and on the main corpus for the fold counts of SPEC_DETECTION 4.5."""
from __future__ import annotations

import numpy as np
import pytest

from _det_common import corpus, C, V, DM, N_EST
from plan11_encoding_ladder import detection_splits as DS
from plan11_encoding_ladder import series as S


def _join_rows(members=(1, 2, 5, 6), letters=None, kernels=("gemm", "gibbs"), reps=2, idle=2, external=()):
    letters = letters or {1: "A", 2: "A", 3: "A", 4: "A", 5: "B", 6: "C", 7: "C", 8: "C"}
    join, order = [], 0
    for k in kernels:
        for r in range(reps):
            order += 1
            join.append({"cell_id": f"{k}__rep{r:02d}__synth", "class": "benign_kernel", "y": "benign", "workload_key": k, "family": "kernels", "member_index": 0,
                         "subfamily_letter": "-", "rep": r, "campaign": "c1" if r % 2 == 0 else "c2", "order_index": order, "order_token": f"Br{r}", "split_role": "train_test"})
    for m in members:
        for r in range(reps):
            order += 1
            join.append({"cell_id": f"sandbox_member_{m}__rep{r:02d}__stage1", "class": "sandbox", "y": "sandbox", "workload_key": f"sandbox_member_{m}", "family": "sandbox",
                         "member_index": m, "subfamily_letter": letters[m], "rep": r, "campaign": "stage1", "order_index": order, "order_token": f"S{m}r{r}", "split_role": "train_test"})
    for r in range(idle):
        order += 1
        join.append({"cell_id": f"idle__rep{r:02d}__idle", "class": "idle", "y": "benign", "workload_key": "idle", "family": "idle", "member_index": 0, "subfamily_letter": "-",
                     "rep": r, "campaign": "c1" if r % 2 == 0 else "c2", "order_index": order, "order_token": f"Ir{r}", "split_role": "train_test"})
    for m in external:
        order += 1
        join.append({"cell_id": f"external_member_{m}__rep00__stage1", "class": "external", "y": "sandbox", "workload_key": f"external_member_{m}", "family": "external",
                     "member_index": m, "subfamily_letter": "-", "rep": 0, "campaign": "stage1", "order_index": order, "order_token": f"X{m}r0", "split_role": "test_only"})
    return join


def _feat(join, n_windows=3, d=4, seed=0):
    rng = np.random.default_rng(seed)
    ids, X, ws = [], [], []
    for j in join:
        for w in range(n_windows):
            ids.append(j["cell_id"]); ws.append(w * 4)
            X.append(rng.normal(size=d) + (5.0 if j["y"] == "sandbox" else 0.0))
    return {"X": np.array(X), "cell_id": np.array(ids), "win_start": np.array(ws), "feature_names": np.array([f"f{i}" for i in range(d)])}


def _lab(join=None, feat=None, **kw):
    join = join or _join_rows()
    feat = feat or _feat(join)
    fb = {j["cell_id"]: (V.GK0_AT_FLOOR if j["member_index"] == 6 else ("control" if j["class"] == "idle" else V.GK0_ABOVE_FLOOR)) for j in join}
    return DS.make_detection_labels(feat, join, {j["cell_id"] for j in join}, floor_by_cell=fb, **kw)


def test_make_labels_cell_rows_collapse_windows():
    join = _join_rows(); feat = _feat(join)
    lab = _lab(join, feat)
    assert lab["row_unit"] == "cell" and lab["n"] == len(join) and lab["X"].shape == (len(join), 4)
    i = list(lab["cell_id"]).index("gemm__rep00__synth")
    assert np.allclose(lab["X"][i], feat["X"][feat["cell_id"] == "gemm__rep00__synth"].mean(axis=0))
    assert lab["n_windows"][i] == 3 and lab["win_start"][i] == -1
    assert lab["kernel"][i] == "gemm" and lab["archetype"][i] == "benign_kernel" and lab["y"][i] == "benign"
    j = list(lab["cell_id"]).index("sandbox_member_6__rep00__stage1")
    assert lab["floor_verdict"][j] == V.GK0_AT_FLOOR and lab["member_index"][j] == 6 and lab["subfamily_letter"][j] == "C"
    assert lab["_cells_without_rows"] == []


def test_make_labels_window_rows_and_missing_cells():
    join = _join_rows(); feat = _feat(join)
    lab = DS.make_detection_labels(feat, join, {j["cell_id"] for j in join}, row_unit="window")
    assert lab["row_unit"] == "window" and lab["n"] == 3 * len(join) and (lab["floor_verdict"] == DS.PENDING_GK0).all()
    adm = {j["cell_id"] for j in join} | {"ghost__rep00__synth"}
    join2 = join + [{**join[0], "cell_id": "ghost__rep00__synth", "workload_key": "ghost"}]
    lab2 = DS.make_detection_labels(feat, join2, adm)
    assert lab2["_cells_without_rows"] == ["ghost__rep00__synth"]
    lab3 = DS.make_detection_labels(feat, join, {j["cell_id"] for j in join}, mask_classes=("sandbox",))
    assert set(lab3["cls"]) == {"sandbox"}


def test_fold_lowo_counts_and_grouping():
    lab = _lab()
    folds = DS.fold_lowo(lab)
    assert [f["held_out"] for f in folds] == ["gemm", "gibbs", "sandbox_member_1", "sandbox_member_2", "sandbox_member_5", "sandbox_member_6", "idle"]
    assert {f["y_held"] for f in folds} == {"benign", "sandbox"}
    for f in folds:
        assert len(f["train"]) + len(f["test"]) == lab["n"] and not set(f["train"]) & set(f["test"])
    lab_x = _lab(_join_rows(external=(1,)))
    folds = DS.fold_lowo(lab_x)
    assert folds[-1]["name"] == "lowo/final" and folds[-1]["held_out"] == "external"
    ext = np.flatnonzero(lab_x["split_role"] == "test_only")
    assert all(not set(f["train"]) & set(ext) for f in folds) and set(folds[-1]["test"]) == set(ext)


def test_assert_grouped_refuses_a_leaking_fold():
    lab = _lab()
    i = np.flatnonzero(lab["workload_key"] == "gemm")
    bad = [{"name": "leak", "train": np.array([i[0]] + [k for k in range(lab["n"]) if k not in i]), "test": np.array([i[1]])}]
    with pytest.raises(AssertionError):
        DS._assert_grouped(bad, lab, "workload_key")


def test_fold_loco_modes_and_final():
    lab = _lab(_join_rows(external=(2,)))
    cell = DS.fold_loco(lab, "cell")
    n_tt = int((lab["split_role"] == "train_test").sum())
    assert len(cell) == n_tt + 1 and cell[-1]["name"] == "loco/final"
    rep = DS.fold_loco(lab, "rep_index")
    assert len(rep) == 2 + 1 and all(f["name"].startswith("loco/rep") for f in rep[:-1])
    ext = set(np.flatnonzero(lab["split_role"] == "test_only"))
    assert all(not set(f["train"]) & ext for f in cell + rep)


def test_fold_lofo_and_one_class():
    lab = _lab()
    lofo = DS.fold_lofo(lab)
    assert [f["held_out"] for f in lofo] == ["kernels", "idle"]
    sb = set(np.flatnonzero(lab["y"] == "sandbox"))
    assert all(sb <= set(f["train"]) for f in lofo)
    assert DS.fold_lofo(_lab(_join_rows(kernels=(), idle=0))) == []
    oc = DS.fold_one_class(lab)
    assert [f["held_out"] for f in oc] == ["gemm", "gibbs", "idle", "sandbox"]
    assert all(not set(f["train"]) & sb for f in oc) and set(oc[-1]["test"]) == sb


def test_fold_level2_and_level3():
    lab = _lab(_join_rows(members=(1, 2, 3, 4, 5, 6, 7, 8)))
    l2 = DS.fold_level2(lab)
    assert len(l2) == 7 and {f["subfamily_letter"] for f in l2} == {"A", "C"} and all(f["member_index"] != 5 for f in l2)
    assert DS.fold_level2(lab, min_members_test=5) == []
    l3 = DS.fold_level3(lab, "rep_index")
    assert len(l3) == 2 and all(set(lab["cls"][f["train"]]) == {"sandbox"} for f in l3)
    assert len(DS.fold_level3(lab, "cell")) == 16


def test_order_half_labels_and_folds():
    lab = _lab()
    h = DS.order_half_labels(lab, "sandbox", "within_workload")
    for wk in ("sandbox_member_1", "sandbox_member_5"):
        idx = np.flatnonzero(lab["workload_key"] == wk)
        assert sorted(h[idx]) == ["first", "second"]
    assert (h[lab["cls"] != "sandbox"] == "").all()
    hc = DS.order_half_labels(lab, "sandbox", "within_class")
    assert int((hc == "first").sum()) == 4 and int((hc == "second").sum()) == 4
    folds = DS.fold_order(lab, "sandbox", "within_workload")
    assert len(folds) == 4 and all(f["unit"] == "workload" for f in folds)
    idle = DS.fold_order(lab, "idle", "within_class")
    assert len(idle) == 2 and all(f["unit"] == "cell" for f in idle)
    anchor = DS.fold_anchor_idle(lab)
    assert len(anchor) == 2 and all(len(f["test"]) == 1 for f in anchor)
    lab2 = _lab(); lab2["order_index"][:] = -1
    assert DS.fold_order(lab2, "sandbox") == []


def test_folds_on_the_main_corpus():
    out = corpus("main")
    data = DM.load_detection_data(out, "content", "W8_H4")
    lab = data["lab"]
    assert lab["n"] == 46 and lab["row_unit"] == "cell"
    folds = DS.fold_lowo(lab)
    assert len(folds) == 6 + 1 + 8           # six kernels (the lexer kept under the report rule), idle, eight members
    assert len(DS.fold_lofo(lab)) == 2 and len(DS.fold_one_class(lab)) == 7 + 1
    assert len(DS.fold_level2(lab)) == 7 and len(DS.fold_level3(lab)) == 3
    assert len(DS.fold_loco(lab, "rep_index")) == 4 and len(DS.fold_loco(lab, "cell")) == 46     # rep 3 exists on the four idle cells
    names = list(S.load_features(S.features_path(out, "content", "W8_H4", True))["feature_names"])
    assert names == S.feature_names("content", True) and not set(DM.EXCLUDED_BY_DECLARATION) & set(names)
    for rung in S.RUNGS:
        f = S.load_features(S.features_path(out, rung, "W8_H4", True))
        assert list(f["feature_names"]) == S.feature_names(rung, True)
        assert not any(n.split(".")[-1] in DM.EXCLUDED_BY_DECLARATION for n in f["feature_names"])
