"""detection_metrics.py: the in-fold threshold, the operating point, the workload-level null, the
one-feature baseline and B1-G3, the split stage, the one-class run and the ladder
(SPEC_DETECTION 4.5, with the corrections of the three reviews)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from _det_common import corpus, corpus_root, split, scores, rows, jload, run_cli, tmp_out, C, S, V, DM, DS, N_EST
from test_detection_splits import _join_rows, _feat, _lab


# --------------------------------------------------------------------------- table tests

def test_in_fold_threshold_linear_and_strict_flag():
    sc = {f"c{i}": float(i) for i in range(100)}
    assert DM.in_fold_threshold(sc, 0.05) == pytest.approx(94.05)
    assert DM.in_fold_threshold(sc, 0.01) == pytest.approx(98.01)
    assert DM.in_fold_threshold(sc, 0.05, method="higher") == 95.0
    assert math.isnan(DM.in_fold_threshold({}, 0.05))
    assert DM.cell_scores(["a", "a", "b"], [0.2, 0.4, float("nan")]) == {"a": pytest.approx(0.3), "b": pytest.approx(float("nan"), nan_ok=True)}
    assert DM.cell_scores(["a", "a"], [0.2, 0.9], "vote_fraction") == {"a": 0.5}


def test_count_assignments_and_permutations():
    assert DM.count_assignments(8, 13) == 203490 and DM.count_assignments(3, 3) == 20
    rng = np.random.default_rng(0)
    subs, ex = DM.workload_label_permutations([f"w{i}" for i in range(6)], 2, 500, rng)
    assert ex and len(subs) == 15 and all(len(s) == 2 for s in subs)
    subs, ex = DM.workload_label_permutations([f"w{i}" for i in range(6)], 2, 5, rng)
    assert not ex and len(subs) == 5        # a smoke run below the count is not exhaustive
    subs, ex = DM.workload_label_permutations([f"w{i}" for i in range(21)], 8, 30, rng)
    assert not ex and len(subs) == 30


def test_null_verdict_rules():
    null = np.linspace(0, 1, 500)
    v, s = DM.null_verdict(0.5, null, 19, exhaustive=True)
    assert v == V.NULL_NOT_ESTIMABLE and V.is_refusal(v)
    v, s = DM.null_verdict(0.999, null, 203490, exhaustive=False)
    assert v == V.PASS and s["rank_text"] == "rank 499 of 500"
    v, s = DM.null_verdict(float(np.quantile(null, 0.95)), null, 203490, exhaustive=False)
    assert v == V.NULL_INSIDE                                   # a tie fails
    v, s = DM.null_verdict(0.999, null[:20], 203490, exhaustive=False)
    assert v == V.not_run("20 permutations < 500")
    v, s = DM.null_verdict(0.999, null[:20], 203490, exhaustive=True)
    assert v == V.PASS
    v, s = DM.null_verdict(None, null, 203490, exhaustive=False)
    assert v.startswith("not run:")


def test_exact_binary_label_permutations():
    lab = _lab()
    y = DS.order_half_labels(lab, "sandbox", "within_workload")
    perms, unit, reason, n = DM.binary_label_permutations(lab, y, "order", "within_workload", 5, np.random.default_rng(1))
    assert unit == "cell within workload" and n == 2 ** 4 and len(perms) == 5
    for yp in perms:
        for wk in set(lab["workload_key"][lab["cls"] == "sandbox"]):
            idx = lab["workload_key"] == wk
            assert sorted(yp[idx]) == sorted(y[idx])
    yc = DS.order_half_labels(lab, "sandbox", "within_class")
    perms, unit, reason, n = DM.binary_label_permutations(lab, yc, "order", "within_class", 5, np.random.default_rng(1))
    assert unit == "workload" and n == math.comb(4, 2)
    perms, unit, _, n = DM.binary_label_permutations(lab, lab["campaign"].astype(str), "anchor_idle", "", 3, np.random.default_rng(1))
    assert unit == "cell"


# --------------------------------------------------------------------------- the operating point on hand-built data

def _separable(n_workloads=6, reps=3, d=5, gap=6.0, seed=0):
    rng = np.random.default_rng(seed)
    join = []
    for w in range(n_workloads):
        pos = w < n_workloads // 2
        for r in range(reps):
            join.append({"cell_id": f"w{w}__rep{r:02d}__x", "class": "sandbox" if pos else "benign_kernel", "y": "sandbox" if pos else "benign",
                         "workload_key": f"w{w}", "family": "sandbox" if pos else "kernels", "member_index": w + 1 if pos else 0, "subfamily_letter": "A" if pos else "-",
                         "rep": r, "campaign": "c", "order_index": -1, "order_token": "", "split_role": "train_test"})
    X = np.array([rng.normal(size=d) + (gap if j["y"] == "sandbox" else 0.0) for j in join])
    feat = {"X": X, "cell_id": np.array([j["cell_id"] for j in join]), "win_start": np.zeros(len(join), dtype=int), "feature_names": np.array([f"f{i}" for i in range(d)])}
    fb = {j["cell_id"]: V.GK0_ABOVE_FLOOR for j in join}
    lab = DS.make_detection_labels(feat, join, {j["cell_id"] for j in join}, floor_by_cell=fb)
    return lab, fb


def test_run_operating_point_separable_and_chance():
    lab, fb = _separable()
    folds = DS.fold_lowo(lab)
    res = DM.run_operating_point(lab["X"], lab, folds, n_estimators=N_EST)
    summ = DM.summarize_operating_point(res["records"], res["folds"], fb)
    assert summ["tpr_05"] == 1.0 and summ["fpr_05_realized"] == 0.0 and summ["auc"] == 1.0
    assert all(f["setters_05"] for f in res["folds"]) and all(f["threshold_05"] is not None for f in res["folds"])
    assert summ["per_member"]["1"]["eighths"] == "3/3" and summ["majority_accuracy"] == pytest.approx(0.5)
    assert summ["gop"]["n_realized_fp_cells"] == 0 and summ["gcal"]["per_fold_tpr05"] == 1.0
    lab0, fb0 = _separable(gap=0.0)
    res0 = DM.run_operating_point(lab0["X"], lab0, DS.fold_lowo(lab0), n_estimators=N_EST)
    s0 = DM.summarize_operating_point(res0["records"], res0["folds"], fb0)
    assert 0.2 <= s0["auc"] <= 0.8


def test_threshold_sources_and_window_rows():
    lab, fb = _separable()
    folds = DS.fold_lowo(lab)
    for src in DM.THRESHOLD_SOURCES:
        res = DM.run_operating_point(lab["X"], lab, folds, n_estimators=N_EST, threshold_source=src)
        assert all(f["threshold_05"] is not None for f in res["folds"]), src
    res = DM.run_operating_point(lab["X"], lab, folds, n_estimators=N_EST, threshold_source="inner_group_kfold", inner_k=2)
    assert all(len(f["setters_05"]) >= 1 for f in res["folds"])


def test_denominator_rules_control_pending_and_no_score():
    lab, fb = _separable()
    folds = DS.fold_lowo(lab)
    res = DM.run_operating_point(lab["X"], lab, folds, n_estimators=N_EST)
    # a positive cell whose verdict is control or at floor leaves the denominator
    fb2 = dict(fb); fb2["w0__rep00__x"] = "control"; fb2["w0__rep01__x"] = V.GK0_AT_FLOOR
    summ = DM.summarize_operating_point(res["records"], res["folds"], fb2)
    assert summ["n_positive_at_floor"] == 2 and summ["n_positive_in_denominator"] == 7 and summ["per_member"]["1"]["eighths"] == "1/1"
    # a pending G-K0 makes every true-positive field 'not run: gk0 not run'
    fb3 = dict(fb); fb3["w1__rep00__x"] = DS.PENDING_GK0
    summ = DM.summarize_operating_point(res["records"], res["folds"], fb3)
    assert summ["tpr_05"] == V.not_run("gk0 not run") and summ["tpr_01"] == V.not_run("gk0 not run") and summ["gk0_pending"]
    assert summ["per_member"]["1"]["eighths"] == V.not_run("gk0 not run") and isinstance(summ["fpr_05_realized"], float)
    # a cell without a finite score is excluded and never counted as a miss
    recs = [dict(r) for r in res["records"]]
    r0 = next(r for r in recs if r["cell_id"] == "w0__rep00__x")
    r0.update({"score": None, "flag_05": None, "flag_01": None, "score_status": DM.NO_SCORE})
    summ = DM.summarize_operating_point(recs, res["folds"], fb)
    assert summ["excluded_no_score"] == ["w0__rep00__x"] and summ["n_without_score"] == 1 and summ["tpr_05"] == 1.0 and summ["n_positive_in_denominator"] == 8


def test_at_floor_positives_leave_training():
    lab, fb = _separable()
    fb2 = dict(fb); fb2["w0__rep00__x"] = V.GK0_AT_FLOOR
    lab["floor_verdict"] = np.array([fb2[c] for c in lab["cell_id"]], dtype=str)
    folds = DS.fold_lowo(lab)
    res = DM.run_operating_point(lab["X"], lab, folds, n_estimators=N_EST, train_on_at_floor=False)
    f = next(f for f in res["folds"] if f["held_out"] == "w1")
    assert f["n_train_dropped_at_floor"] == 1 and f["n_train_cells"] == lab["n"] - 3 - 1
    res2 = DM.run_operating_point(lab["X"], lab, folds, n_estimators=N_EST, train_on_at_floor=True)
    f2 = next(f for f in res2["folds"] if f["held_out"] == "w1")
    assert f2["n_train_dropped_at_floor"] == 0 and f2["n_train_cells"] == lab["n"] - 3


def test_l1_row_and_quarantine_on_hand_built_data():
    lab, fb = _separable(d=3)
    lab["X"][:, 1] = np.random.default_rng(3).normal(size=lab["n"])      # feature 1 carries nothing
    folds = DS.fold_lowo(lab)
    l1 = DM.l1_row(lab["X"], lab, folds, names=["f0", "noise", "f2"], floor_by_cell=fb, per_feature=True)
    assert l1["tpr_05"] == 1.0 and l1["auc"] == 1.0 and set(l1["features_chosen"]) <= {"f0", "f2"}
    assert l1["per_feature"]["noise"]["auc"] < 0.8
    res = DM.run_operating_point(lab["X"], lab, folds, n_estimators=N_EST)
    q = DM.quarantine_l1_two_class(l1["runs"], ["f0", "noise", "f2"], res["records"])
    assert all(x["n_disagree_workloads"] <= DM.B1G3_MAX_DISAGREE_WORKLOADS for x in q) and "noise" not in {x["feature"] for x in q}
    # the one-feature models of f0 and f2 reproduce the forest's decisions on every positive cell; their benign
    # false positives fall on other cells than the forest's (both thresholds are set on few benign workloads here),
    # so the count of disagreeing held-out workloads is small but not always within one
    q2 = DM.quarantine_l1_two_class(l1["runs"], ["f0", "noise", "f2"], res["records"], max_disagree_workloads=2)
    assert {x["feature"] for x in q2} >= {"f0", "f2"} and "noise" not in {x["feature"] for x in q2}


# --------------------------------------------------------------------------- the split stage on the main corpus

def test_lowo_content_known_answers():
    d = split("main", "content", "lowo")
    sc = scores(d)
    assert sc["status"] == "ok" and sc["n_folds"] == 15 and sc["row_unit"] == "cell" and sc["score_source"] == DM.SCORE_SOURCE_FULL
    pm = sc["per_member"]
    assert pm["6"]["at_floor"] == 3 and pm["6"]["eighths"] == "0/0" and sc["n_positive_at_floor"] == 3 and sc["n_positive_in_denominator"] == 21
    assert sum(pm[m]["hits"] for m in ("1", "2", "3", "4")) >= 6              # the amount-axis members are heard
    assert pm["5"]["eighths"] == "0/3" and pm["7"]["eighths"] == "0/3" and pm["8"]["eighths"] == "0/3"   # nothing on the content axis
    assert sc["auc"] > 0.7 and sc["fpr_05_realized"] <= 0.25 and sc["per_family_fpr"]["idle"]["fpr"] == 0.0
    assert sc["dim_status"] == V.GDIM_FULL and sc["feature_count"] == 36 and sc["pooled_label"] == V.POST_HOC_LABEL
    assert sc["null"]["status"] == "ok" and sc["null"]["S"] == 8 and sc["null"]["B"] == 7 and sc["null"]["n_assignments"] == 6435
    assert sc["null"]["tpr05"]["verdict"] in (V.PASS, V.NULL_INSIDE) and "verdict" in sc["null"]["auc"] and sc["null"]["l1"]
    assert sc["params"]["threshold_source"] == "inner_lowo" and sc["params"]["train_on_at_floor"] is False and sc["params"]["null_denominator_rule"] == DM.NULL_DENOMINATOR_RULE
    assert sc["params"]["score_aggregation"].startswith("not applicable") and sc["params"]["excluded_by_declaration"] == list(DM.EXCLUDED_BY_DECLARATION)
    assert "head_drop_json" in sc["params"] and sc["params"]["grid_source"].startswith("default: W8_H4")
    assert sc["l1"]["features_chosen"] and isinstance(sc["l1"]["tpr_05"], float)
    assert sc["gcal"]["verdict"] in (V.GCAL_AGREE, V.GCAL_PERFOLD)
    pr = rows(d / "predictions.csv")
    assert list(pr[0].keys()) == list(DM.PRED_COLUMNS) and len(pr) == 46
    m6 = [r for r in pr if r["member_index"] == "6"]
    assert all(r["in_denominator"] == "false" and r["floor_verdict"] == V.GK0_AT_FLOOR for r in m6)
    assert (d / "roc.csv").is_file() and (d / "folds.json").is_file() and (d / "l1_quarantine.json").is_file() and (d / "null.json").is_file()
    fj = jload(d / "folds.json")["folds"]
    assert all(f["n_train_dropped_at_floor"] >= 0 and f["threshold_05"] is not None for f in fj)
    assert all("sandbox_member_6" not in c for f in fj for c in f["setters_05"])
    # the default corpus quarantines a content feature and keeps both readings (ML review 2.4)
    assert sc["quarantine"]["quarantined_features"] and sc["with_quarantine"] and "tpr_05" in sc["with_quarantine"]
    assert (d / "predictions_with_quarantine.csv").is_file()


def test_lowo_smoke_null_verb():
    out = corpus("main")
    d = DM.run_detection_split(out, "content", "W8_H4", "lowo", n_perm=4, n_estimators=N_EST, quarantine=False, dir_name="lowo_smoke")
    sc = scores(d)
    assert sc["null"]["tpr05"]["verdict"] == V.not_run("4 permutations < 500") and isinstance(sc["null"]["tpr05"]["p95"], float)
    assert sc["gcal"]["verdict"] in (V.GCAL_AGREE, V.GCAL_PERFOLD)


def test_loco_and_lofo():
    sc = scores(split("main", "content", "loco", loco_mode="rep_index", quarantine=False))
    assert sc["status"] == "ok" and sc["n_folds"] == 4 and sc["null"]["status"] == "ok"      # rep indices 0 to 2, and the idle cells' rep 3
    sc = scores(split("main", "content", "lofo", quarantine=False))
    assert sc["status"] == "ok" and sc["n_folds"] == 2
    assert sc["tpr_05"] == V.not_applicable("sandbox never held out under LOFO") and sc["per_member"] == {}
    assert set(sc["per_family_fpr"]) == {"kernels", "idle"}
    lab = DM.load_detection_data(corpus("main"), "content", "W8_H4")["lab"]
    lab2 = DS.sublab(lab, lab["y"] == "sandbox")
    assert DS.fold_lofo(lab2) == []


def test_split_written_refusals():
    out = corpus("main")
    d = DM.run_detection_split(out, "combined", "W8_H4", "lowo", normalized=False, n_perm=0)
    assert scores(d)["status"].startswith("not applicable: raw variant of combined")
    d = DM.run_detection_split(out, "content", None if False else "W8_H4", "lowo", n_perm=0, row_unit="window", threshold_source="oob", dir_name="lowo_window_oob")
    assert scores(d)["status"] == V.not_applicable("out-of-bag threshold under window rows (sibling windows in bag)")
    o2 = tmp_out()
    S.write_csv(o2 / "cells.csv", ("cell_id",), [])
    d = DM.run_detection_split(o2, "content", None, "lowo", n_perm=0)
    assert scores(d)["status"] == V.not_run("no selection for content (run classes inherit-selection)")
    # gk0_cells.csv missing is a written refusal (al-Farabi M3)
    gk = C.det_dir(out) / "gk0_cells.csv"
    bak = gk.read_bytes(); gk.unlink()
    try:
        d = DM.run_detection_split(out, "content", "W8_H4", "lowo", n_perm=0, dir_name="lowo_nogk0")
        assert scores(d)["status"] == V.not_run("gates/detection/gk0_cells.csv missing (run gk0-cells)")
    finally:
        gk.write_bytes(bak)


def test_subset_and_exclude_families():
    out = corpus("main")
    d = DM.run_detection_split(out, "content", "W8_H4", "lowo", n_perm=0, n_estimators=N_EST, quarantine=False,
                               subset_workloads=("sandbox_member_1", "sandbox_member_2", "gemm", "gibbs"), dir_name="glm_test")
    sc = scores(d)
    assert sc["n_folds"] == 4 and set(sc["per_member"]) == {"1", "2"} and sc["params"]["subset_workloads"] == ["sandbox_member_1", "sandbox_member_2", "gemm", "gibbs"]
    d = DM.run_detection_split(out, "content", "W8_H4", "lowo", n_perm=0, n_estimators=N_EST, quarantine=False, exclude_families=("idle",), dir_name="lowo__without_idle")
    sc = scores(d)
    assert "idle" not in sc["per_family_fpr"] and sc["n_folds"] == 14


def test_reduce_to_strongest_cli_and_matched_row():
    out = corpus("main")
    split("main", "content", "lowo")
    for r in ("apf", "wapf", "persist"):
        DM.run_detection_split(out, r, "W8_H4", "lowo", n_perm=0, n_estimators=N_EST, quarantine=False)
    rung, d = DM.strongest_single_rung(out, {r: "W8_H4" for r in ("apf", "wapf", "persist", "content")})
    assert rung in ("apf", "wapf", "persist", "content") and d >= 8
    res = run_cli("detection_metrics", "splits", "--out", out, "--rung", "combined", "--split", "lowo", "--norm", "--null-perm", "0", "--n-estimators", N_EST, "--reduce-to-strongest", check=False)
    assert res.returncode == 0, res.stderr
    sc = scores(DM.split_dir(out, "combined", "W8_H4", "lowo_matched", True))
    assert sc["status"] == "ok" and sc["feature_count_used"] == d and sc["dim_status"] == V.GDIM_REDUCED and sc["params"]["matched_to_rung"] == rung


# --------------------------------------------------------------------------- the one-class run

def test_one_class_primary_null_and_usage_refusal():
    out = corpus("main")
    d = DM.run_one_class(out, "combined", "W8_H4", n_perm=4, n_perm_required=4, n_estimators=N_EST)
    sc = scores(d)
    assert sc["status"] == "ok" and sc["model"] == "isolation_forest" and sc["primary"] and sc["g1c_label"] == V.G1C_PRIMARY
    assert sc["n_folds"] == 8 and isinstance(sc["threshold_05"], float) and sc["threshold_source"] == "inner_lowo"
    assert set(sc["per_member"]) == {str(m) for m in range(1, 9)} and sc["per_member"]["6"]["at_floor"] == 3
    assert sc["null"]["status"] == "ok" and sc["null"]["tpr05"]["verdict"] in (V.PASS, V.NULL_INSIDE) and sc["null"]["n_perm"] == 4
    pr = rows(d / "predictions.csv")
    assert all(r["fold"] == "one_class/final" for r in pr if r["y"] == "sandbox") and all(r["fold"].startswith("one_class/") for r in pr)
    # no sandbox cell in any training fold
    fj = jload(d / "folds.json")["folds"]
    lab = DM.load_detection_data(out, "combined", "W8_H4")["lab"]
    for f in fj:
        assert set(lab["y"][np.asarray(f["train"], dtype=int)]) == {"benign"}
    r = run_cli("detection_metrics", "one-class", "--out", out, "--rung", "combined", "--model", "gmm", check=False)
    assert r.returncode == 3 and "usage: a second one-class model needs --secondary" in r.stderr
    assert not DM.split_dir(out, "combined", "W8_H4", "one_class__gmm", True).exists()
    d2 = DM.run_one_class(out, "combined", "W8_H4", model="gmm", primary=False, n_perm=0, n_estimators=N_EST)
    assert d2.name == "one_class__gmm__norm" and scores(d2)["g1c_label"] == V.G1C_SECONDARY


# --------------------------------------------------------------------------- the ladder

def test_prefix_rows_and_slice_extract():
    assert [DM.prefix_rows(932, p, 0.644) for p in DM.LADDER_PREFIXES_S] == [47, 93, 186, 466, 932]
    assert [DM.prefix_rows(900, p, 0.5) for p in DM.LADDER_PREFIXES_S] == [60, 120, 240, 600, 900]
    assert [DM.prefix_rows(240, p, "per_cell") for p in DM.LADDER_PREFIXES_S] == [12, 24, 48, 120, 240]
    assert DM.prefix_rows(120, 30, "per_cell") == 6
    ex = {"seq": np.arange(1, 11), "K": np.arange(10) * 10.0, "_n_rows": 10, "_path": "x"}
    s = DM.slice_extract(ex, 2, 5)
    assert s["_n_rows"] == 5 and list(s["seq"]) == [3, 4, 5, 6, 7] and s["_path"] == "x"
    assert DM.slice_extract(ex, 8, 5)["_n_rows"] == 2
    assert DM.boundary_start("c", {"c": [30, 60]}, 1) == 29 and DM.boundary_start("d", {"c": [30]}, 1) == 0


def test_ladder_prefixes_and_whole_cell_equals_headline():
    out = corpus("main")
    p = DM.run_ladder(out, "content", "W8_H4", prefixes_s=(30, 600), dt=0.644, n_estimators=N_EST, quarantine=False)
    lad = [r for r in rows(p) if r["rung"] == "content"]
    fp1 = {r["prefix_s"]: r for r in lad if r["reading"] == "from_pair1"}
    assert fp1["30"]["n_rows_median"] == "47" and fp1["600"]["n_rows_median"] == "120" and fp1["30"]["n_cells_with_window"] == "46"
    assert float(fp1["30"]["tpr_05"]) >= 0.0 and fp1["30"]["null_verdict"] == V.not_run("ladder null not requested")
    head = scores(split("main", "content", "lowo"))
    assert float(fp1["600"]["tpr_05"]) == pytest.approx(head["tpr_05"])
    a = S.load_features(S.features_path(out, "content", "W8_H4", True))
    b = S.load_features(DM.prefix_features_path(out, "content", "W8_H4", True, 600, "from_pair1"))
    for c in set(a["cell_id"]):
        assert np.allclose(a["X"][a["cell_id"] == c], b["X"][b["cell_id"] == c], equal_nan=True)
    fb = {r["prefix_s"]: r for r in lad if r["reading"] == "from_boundary"}
    assert fb["30"]["tpr_05"] != "" and fb["30"]["note"] == ""          # the boundaries file names gemm and floyd
    assert (C.det_dir(out) / "ladder" / "content" / "W8_H4" / "from_pair1" / "prefix30s" / "scores.json").is_file()
    lj = jload(C.det_dir(out) / "ladder.json")
    assert lj["params"]["ladder_head_drop_rule"] == DM.LADDER_HEAD_DROP_RULE and lj["per_rung"]["content"]["have_boundary"]
    # without a boundaries file the from_boundary rows read 'from pair 1 only'
    bp = out / "inputs" / "iteration_boundaries.csv"
    bak = bp.read_bytes(); bp.unlink()
    try:
        p = DM.run_ladder(out, "apf", "W8_H4", prefixes_s=(600,), dt=0.644, n_estimators=N_EST, quarantine=False)
        r = [x for x in rows(p) if x["rung"] == "apf" and x["reading"] == "from_boundary"][0]
        assert r["note"] == V.LADDER_FROM_PAIR1_ONLY and r["tpr_05"] == ""
    finally:
        bp.write_bytes(bak)
    # a prefix shorter than one window is a written refusal
    p = DM.run_ladder(out, "content", "W8_H4", prefixes_s=(30,), dt="per_cell", readings=("from_pair1",), n_estimators=N_EST)
    r = [x for x in rows(p) if x["rung"] == "content" and x["prefix_s"] == "30"][0]
    assert r["tpr_05"] == V.not_applicable("prefix shorter than one window (n = 6 < W = 8)")
