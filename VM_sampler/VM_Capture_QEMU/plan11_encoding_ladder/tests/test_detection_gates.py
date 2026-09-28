"""gates_detection.py: for every gate one case that must pass and one that must refuse (or the
named outcome), on the synthetic corpora of _det_common and on hand-written fixtures in the
schemas of SPEC_DETECTION section 3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _det_common import (corpus, corpus_root, split, scores, rows, row_where, jload, run_cli, tmp_out, write_fixture_scores, fixture_selection,
                         restrict_admissible, swap_file, C, S, V, GD, DM, DS, GP, N_EST)
from plan11_encoding_ladder import gates_comparison as GX


def det(out: Path) -> Path:
    return C.det_dir(out)


# --------------------------------------------------------------------------- admissibility (3.5.1)

def test_admissibility_report_and_exclude():
    out = corpus("main")
    adm = {r["cell_id"]: r for r in rows(det(out) / "admissibility.csv")}
    assert list(rows(det(out) / "admissibility.csv")[0].keys()) == list(GD.ADM_COLUMNS)
    m6 = [r for r in adm.values() if r["cell_id"].startswith("sandbox_member_6_")]
    assert len(m6) == 3 and all(r["admissible"] == "true" and r["C1"] == V.FAIL and r["reason"].startswith("C1 fail: reported") for r in m6)
    lex = [r for r in adm.values() if r["cell_id"].startswith("lexer_")]
    assert all(r["admissible"] == "true" and r["reason"].startswith("C1 fail: reported") for r in lex)      # the same rule for every class
    assert all(r["admissible"] == "true" and r["reason"] == "ok" for r in adm.values() if r["cell_id"].startswith("gemm_"))
    aj = jload(det(out) / "admissibility.json")
    assert aj["counts_per_class"]["sandbox"]["n_c1_fail_reported"] == 3 and aj["counts_per_class"]["benign_kernel"]["n_c1_fail_reported"] == 3
    assert jload(det(out) / "cell_classes.json")["S"] == 8 and jload(det(out) / "cell_classes.json")["params"]["S_source"] == "admissibility.csv"
    bak = (det(out) / "admissibility.csv").read_bytes()
    try:
        GD.admissibility(out, det_c1_rule="exclude")
        adm = {r["cell_id"]: r for r in rows(det(out) / "admissibility.csv")}
        assert all(adm[c]["admissible"] == "false" for c in adm if c.startswith("sandbox_member_6_") or c.startswith("lexer_"))
        assert adm["sandbox_member_6__rep00__stage1"]["reason"] == "C1 fail under det_c1_rule = exclude"
    finally:
        GD.admissibility(out, det_c1_rule="report")
        assert (det(out) / "admissibility.csv").read_bytes() == bak
    with pytest.raises(ValueError):
        GD.admissibility(out, det_c1_rule="maybe")


def test_admissibility_without_preconditions_and_unassigned():
    o = tmp_out()
    S.write_csv(o / "cells.csv", ("cell_id", "kernel", "role", "archetype_predicted", "seed", "rep", "rep_dir", "label", "campaign", "path", "traj_file", "status"),
                [{"cell_id": "gemm__rep00__synth", "kernel": "gemm", "role": "kernel", "rep": 0, "path": "/r/kernel/kernel_gemm_v2/--seed_42_--duration_40/rep001__synth", "status": "ok"},
                 {"cell_id": "odd__rep00__synth", "kernel": "odd", "role": "unknown", "rep": 0, "path": "/r/f/odd/--seed_42_--duration_40/rep001__synth", "status": "ok"}])
    (o / "inputs").mkdir()
    (o / "inputs" / "classes.csv").write_text("path_prefix,class,member_index,subfamily_letter,rep,order_index,family,workload_key\nkernel,benign_kernel,,,,,,\n")
    C.build_join(o)
    GD.admissibility(o)
    adm = {r["cell_id"]: r for r in rows(det(o) / "admissibility.csv")}
    assert adm["gemm__rep00__synth"]["admissible"] == "false" and adm["gemm__rep00__synth"]["reason"] == V.not_run("gates/preconditions.csv missing")
    assert adm["odd__rep00__synth"]["class"] == C.UNASSIGNED and adm["odd__rep00__synth"]["reason"] == "unassigned: no class row in inputs/classes.csv"
    assert jload(det(o) / "admissibility.json")["unassigned_cells"] == ["odd__rep00__synth"]


# --------------------------------------------------------------------------- G-K0 two-class (3.5.2)

def test_gk0_cells_floor_verdicts_and_edge():
    out = corpus("main")
    cells = {r["cell_id"]: r for r in rows(det(out) / "gk0_cells.csv")}
    assert list(rows(det(out) / "gk0_cells.csv")[0].keys()) == list(GD.GK0_CELL_COLUMNS)
    assert all(cells[c]["verdict"] == V.GK0_AT_FLOOR for c in cells if c.startswith("sandbox_member_6_"))
    assert all(cells[c]["verdict"] == V.GK0_ABOVE_FLOOR for c in cells if c.startswith("sandbox_member_1_"))
    assert all(cells[c]["verdict"] == "control" for c in cells if c.startswith("idle_"))
    assert all(cells[c]["verdict"] == V.GK0_AT_FLOOR for c in cells if c.startswith("lexer_"))
    edge = float(cells["gemm__rep00__synth"]["idle_band_edge"])
    assert edge == float(row_where(out / "gates" / "gk0.csv", kernel="gemm")["idle_band_edge"])
    m6 = cells["sandbox_member_6__rep00__stage1"]
    assert float(m6["frac_above_band"]) == 0.0 and m6["l0_med_above"] == "" and m6["J_consec_above"] == ""
    g1 = cells["gemm__rep00__synth"]
    assert float(g1["frac_above_band"]) > 0.9 and g1["l0_med_above"] != "" and g1["J_consec_above"] != ""
    mem = {r["member_index"]: r for r in rows(det(out) / "gk0_members.csv")}
    assert mem["6"]["verdict"] == V.GK0_AT_FLOOR and mem["6"]["n_at_floor"] == "3" and mem["1"]["verdict"] == V.GK0_ABOVE_FLOOR and mem["1"]["source_statement"] == "unstated"
    gk = jload(det(out) / "gk0.json")
    assert gk["refusal"] is None and gk["idle_band_edge"] == edge and gk["params"]["verdict_quantities"] == list(GD.GK0_VERDICT_QUANTITIES)
    assert (out / "inputs" / "gk0_source_sandbox.csv").is_file() and len(rows(out / "inputs" / "gk0_source_sandbox.csv")) == 8


def test_gk0_cells_refusals():
    out = corpus("main")
    p = det(out) / "gk0_cells.csv"
    bak = {q: (det(out) / q).read_bytes() for q in ("gk0_cells.csv", "gk0_members.csv", "gk0.json")}
    try:
        with restrict_admissible(out, lambda r: r["class"] == "idle"):
            GD.gk0_cells(out)
            assert all(r["verdict"] == V.not_run("no admissible idle cell") for r in rows(p))
        gj = out / "gates" / "gk0.params.json"
        doc = jload(gj); doc["params"]["idle_band_edge"] = 999.0
        with swap_file(gj, json.dumps(doc)):
            GD.gk0_cells(out)
            vs = {r["verdict"] for r in rows(p)}
            assert len(vs) == 1 and next(iter(vs)).startswith("refused: idle band edge differs from gates/gk0.csv (")
            assert jload(det(out) / "gk0.json")["refusal"].startswith("refused: idle band edge differs")
    finally:
        for q, b in bak.items():
            (det(out) / q).write_bytes(b)


# --------------------------------------------------------------------------- G-N (3.5.3)

def _fixture_out_for_gn(members: int) -> Path:
    o = tmp_out()
    join, adm, gk = [], [], []
    for m in range(1, members + 1):
        for r in range(2):
            c = f"sandbox_member_{m}__rep{r:02d}__stage1"
            join.append({"cell_id": c, "class": "sandbox", "y": "sandbox", "workload_key": f"sandbox_member_{m}", "family": "sandbox", "member_index": m, "subfamily_letter": "A",
                         "rep": r, "campaign": "stage1", "order_index": "", "order_token": f"S{m}r{r}", "split_role": "train_test"})
            adm.append({"cell_id": c, "class": "sandbox", "admissible": True, "admissible_pair_rungs": True})
            gk.append({"cell_id": c, "verdict": V.GK0_ABOVE_FLOOR, "K_med": 100 * m})
    for k in ("gemm", "gibbs"):
        for r in range(2):
            c = f"{k}__rep{r:02d}__synth"
            join.append({"cell_id": c, "class": "benign_kernel", "y": "benign", "workload_key": k, "family": "kernels", "member_index": 0, "subfamily_letter": "-", "rep": r,
                         "campaign": "synth", "order_index": "", "order_token": f"Br{r}", "split_role": "train_test"})
            adm.append({"cell_id": c, "class": "benign_kernel", "admissible": True, "admissible_pair_rungs": True})
            gk.append({"cell_id": c, "verdict": V.GK0_ABOVE_FLOOR, "K_med": 2000})
    c = "idle__rep00__idle"
    join.append({"cell_id": c, "class": "idle", "y": "benign", "workload_key": "idle", "family": "idle", "member_index": 0, "subfamily_letter": "-", "rep": 0, "campaign": "idle",
                 "order_index": "", "order_token": "Ir0", "split_role": "train_test"})
    adm.append({"cell_id": c, "class": "idle", "admissible": True, "admissible_pair_rungs": True}); gk.append({"cell_id": c, "verdict": "control", "K_med": 150})
    S.write_csv(C.join_path(o), C.JOIN_COLUMNS, join)
    S.write_csv(det(o) / "admissibility.csv", GD.ADM_COLUMNS, adm)
    S.write_csv(det(o) / "gk0_cells.csv", GD.GK0_CELL_COLUMNS, gk)
    S.write_csv(o / "cells.csv", ("cell_id",), [])
    return o


def test_gn_two_class_counts():
    out = corpus("main")
    gn = {r["row"]: r for r in rows(det(out) / "gn.csv")}
    assert gn["sandbox"]["status"] == V.GN_HEADLINE and gn["sandbox"]["n_workloads"] == "7" and "sandbox_member_6" not in gn["sandbox"]["workloads"]
    assert gn["idle"]["status"] == V.GN_SINGLE_WORKLOAD and gn["kernels"]["status"] == V.GN_HEADLINE and gn["kernels"]["n_workloads"] == "6"
    o = _fixture_out_for_gn(2)
    GD.gn_two_class(o)
    assert row_where(det(o) / "gn.csv", row="sandbox")["status"] == V.GN_ONE_TRAIN_WORKLOAD
    o = _fixture_out_for_gn(1)
    GD.gn_two_class(o)
    assert row_where(det(o) / "gn.csv", row="sandbox")["status"] == V.GN_NO_SUPERVISED
    assert row_where(det(o) / "gn.csv", row="kernels")["status"] == V.GN_HEADLINE


# --------------------------------------------------------------------------- G-L (i) (3.5.4)

def test_gl_two_class_pass_and_level_only():
    out = corpus("main")
    split("main", "content", "lowo")
    GD.gl_two_class(out, "content")
    r = row_where(det(out) / "gl.csv", rung="content")
    assert r["verdict"] in (V.PASS, V.GL_LEVEL_ONLY) and r["tpr05_norm"] != "" and r["rank_norm"].startswith("rank") and r["tpr05_raw"] == ""
    o = tmp_out(); fixture_selection(o)
    write_fixture_scores(o, "apf", "lowo", {"null": {"status": "ok", "tpr05": {"verdict": V.NULL_INSIDE, "p95": 0.6, "rank_text": "rank 10 of 500"}, "auc": {"verdict": V.PASS}}})
    write_fixture_scores(o, "apf", "lowo", {"tpr_05": 0.9}, normalized=False)
    GD.gl_two_class(o, "apf")
    r = row_where(det(o) / "gl.csv", rung="apf")
    assert r["verdict"] == V.GL_LEVEL_ONLY and V.is_refusal(r["verdict"]) and r["tpr05_raw"] == "0.9"
    write_fixture_scores(o, "wapf", "lowo", {"null": {"status": "ok", "tpr05": {"verdict": V.NULL_NOT_ESTIMABLE}, "auc": {}}})
    GD.gl_two_class(o, "wapf")
    assert row_where(det(o) / "gl.csv", rung="wapf")["verdict"] == V.NULL_NOT_ESTIMABLE
    GD.gl_two_class(o, "persist")
    assert row_where(det(o) / "gl.csv", rung="persist")["verdict"] == V.not_run("lowo norm split not run")


# --------------------------------------------------------------------------- G-OP (3.5.5)

def test_gop_supported_and_set_by_few():
    assert GD.gop_verdict(5, 3) == V.GOP_SUPPORTED and GD.gop_verdict(4, 3) == V.GOP_SET_BY_FEW and GD.gop_verdict(8, 1) == V.GOP_SET_BY_FEW
    o = tmp_out(); fixture_selection(o)
    per_fold = [{"fold": f"lowo/w{i}", "n_setter_cells": 6, "n_setter_workloads": 3, "setter_workloads": ["a", "b", "c"]} for i in range(4)]
    gop = {"per_fold": per_fold, "worst_fold_n_setter_cells": 6, "worst_fold_n_setter_workloads": 3, "n_union_setter_cells": 9, "n_union_setter_workloads": 4,
           "setter_families": ["kernels"], "n_realized_fp_cells": 2, "n_realized_fp_workloads": 2, "fp_families": ["kernels"]}
    write_fixture_scores(o, "combined", "lowo", {"gop": gop})
    GD.gop(o, "combined")
    r = row_where(det(o) / "gop.csv", rung="combined", split="lowo")
    assert r["verdict"] == V.GOP_SUPPORTED and r["n_folds_unsupported"] == "0" and r["threshold_in_fold"] == "true"
    # the union clears the counts but one fold's setters come from one workload: set by few (ML review 2.3)
    per_fold[2] = {"fold": "lowo/w2", "n_setter_cells": 8, "n_setter_workloads": 1, "setter_workloads": ["idle"]}
    gop["worst_fold_n_setter_workloads"] = 1; gop["setter_families"] = ["idle", "kernels"]
    write_fixture_scores(o, "combined", "lowo", {"gop": gop})
    GD.gop(o, "combined")
    r = row_where(det(o) / "gop.csv", rung="combined", split="lowo")
    assert r["verdict"] == V.GOP_SET_BY_FEW and V.is_refusal(r["verdict"]) and r["n_folds_unsupported"] == "1" and "idle" in r["setter_families"]
    GD.gop(o, "combined", cells_rule="realized_fps")
    assert row_where(det(o) / "gop.csv", rung="combined", split="lowo")["verdict"] == V.GOP_SET_BY_FEW
    gop["n_realized_fp_cells"] = 6; gop["n_realized_fp_workloads"] = 3
    write_fixture_scores(o, "combined", "lowo", {"gop": gop})
    GD.gop(o, "combined", cells_rule="realized_fps")
    assert row_where(det(o) / "gop.csv", rung="combined", split="lowo")["verdict"] == V.GOP_SUPPORTED
    out = corpus("main"); split("main", "content", "lowo")
    GD.gop(out, "content")
    r = row_where(det(out) / "gop.csv", rung="content", split="lowo")
    assert r["verdict"] in (V.GOP_SUPPORTED, V.GOP_SET_BY_FEW) and r["n_folds"] == "15"


# --------------------------------------------------------------------------- G-LM (3.5.6)

@pytest.mark.parametrize("args,expected", [
    ((0.9, 0.05, 0.85, 0.06, 40, 5), V.GLM_SURVIVES),
    ((0.9, 0.05, 0.05, 0.06, 40, 5), V.GLM_LEVEL_ONLY),
    ((0.0, 0.05, 0.0, 0.0, 40, 5), V.GLM_NOT_DETECTED),
    ((0.9, 0.05, 0.9, 0.0, 8, 0), V.GLM_EMPTY_BAND),
    ((0.9, 0.05, 0.9, 0.0, 8, 1), V.GOP_SET_BY_FEW + " (level band)"),
])
def test_glm_verdict_table(args, expected):
    assert GD.glm_verdict(*args) == expected


def test_glm_below_half_rule():
    assert GD.glm_verdict(0.9, 0.05, 0.4, 0.3, 40, 5, rule="below_half_unrestricted") == V.GLM_LEVEL_ONLY
    assert GD.glm_verdict(0.9, 0.05, 0.5, 0.3, 40, 5, rule="below_half_unrestricted") == V.GLM_SURVIVES


def test_glm_on_the_corpus():
    out = corpus("main"); split("main", "content", "lowo")
    GD.glm(out, "content", n_estimators=N_EST)
    g = {r["member_index"]: r for r in rows(det(out) / "glm.csv") if r["rung"] == "content"}
    assert all(g[m]["level_label"] == V.GLM_LABEL_MEDIAN_K for m in g)
    assert g["6"]["verdict"] == V.not_applicable("at floor (G-K0)") and g["8"]["verdict"] == V.GLM_EMPTY_BAND and g["8"]["n_band_workloads"] == "0"
    assert g["7"]["verdict"] == V.GLM_NOT_DETECTED
    surv = [m for m in ("1", "2", "3", "4") if g[m]["verdict"] == V.GLM_SURVIVES]
    assert len(surv) >= 2 and all(g[m]["verdict"] in (V.GLM_SURVIVES, V.GLM_NOT_DETECTED, V.GLM_LEVEL_ONLY, V.GOP_SET_BY_FEW + " (level band)") for m in ("1", "2", "3", "4"))
    assert float(g["1"]["band_lo"]) < float(g["1"]["level"]) < float(g["1"]["band_hi"]) and g["1"]["recall_lm"] != ""
    assert DM.split_dir(out, "content", "W8_H4", "glm_1", True).is_dir()
    GD.glm(out, "content", model="headline_oof")
    g2 = {r["member_index"]: r for r in rows(det(out) / "glm.csv") if r["rung"] == "content"}
    assert V.POST_HOC_LABEL in g2["1"]["level_label"] and g2["8"]["verdict"] == V.GLM_EMPTY_BAND
    lvl, per_cell = GD.level_of_workload(out, C.load_join(out), quantity="per_iteration_K_sum")
    assert "gemm" in lvl and all(str(v).startswith("not run: no iteration boundary") for c, v in per_cell.items() if c.startswith("gibbs"))


# --------------------------------------------------------------------------- G-ANCHOR, the order test, drift (3.5.7)

def test_anchor_main_not_applicable_and_gx_missing():
    out = corpus("main")
    with swap_file(out / "gates" / "gx.csv", None):
        GD.anchor(out, rung="content", n_perm=4, n_estimators=N_EST, n_perm_required=4)
    a = {r["part"]: r for r in rows(det(out) / "ganchor.csv") if r["rung"] == "content"}
    assert a["kernels"]["verdict"] == V.not_run("gates/gx.csv missing (run gates_comparison gx)")
    assert a["idle_sets"]["verdict"] == V.not_applicable("one idle campaign (n = 4 cells)")
    assert a["idle_early_late"]["verdict"] in (V.NULL_NOT_ESTIMABLE, V.ORDER_AUDIBLE, V.ORDER_NOT_AUDIBLE)     # four idle cells: six assignments


def test_anchor_on_corpus_audible():
    out = corpus("on")
    GX.gate_gx(out, "combined", n_perm=10, n_estimators=10, grid_id="W8_H4")
    GD.anchor(out, rung="combined", n_perm=20, n_estimators=N_EST, n_perm_required=20)
    a = {r["part"]: r for r in rows(det(out) / "ganchor.csv") if r["rung"] == "combined"}
    assert a["kernels"]["verdict"] in (V.GANCHOR_AUDIBLE, V.GANCHOR_NOT_AUDIBLE) and a["kernels"]["n_labels"] == "3"
    # the two idle sets differ by their floor churn (0.02 against 0.10; the floor level alone is normalized away): campaign audible
    assert a["idle_sets"]["verdict"] == V.GANCHOR_AUDIBLE and a["idle_sets"]["n_labels"] == "2" and float(a["idle_sets"]["score"]) == 1.0
    d = DM.split_dir(out, "combined", "W8_H4", "anchor_idle", True)
    sc = scores(d)
    assert sc["auc"] == 1.0 and sc["null"]["null_unit"] == "cell" and sc["null"]["n_assignments"] == 70
    assert a["idle_early_late"]["verdict"] in (V.ORDER_AUDIBLE, V.ORDER_NOT_AUDIBLE)


def test_order_test_not_audible_audible_void_and_missing():
    out = corpus("main")
    GD.order_test(out, rung="content", n_perm=6, n_estimators=N_EST, n_perm_required=6)
    o = {(r["class"], r["half_rule"]): r for r in rows(det(out) / "order.csv") if r["rung"] == "content"}
    assert o[("sandbox", "within_workload")]["verdict"] == V.ORDER_NOT_AUDIBLE and o[("benign_kernel", "within_workload")]["verdict"] == V.ORDER_NOT_AUDIBLE
    assert o[("sandbox", "within_class")]["verdict"] == V.ORDER_NOT_AUDIBLE and o[("sandbox", "within_class")]["order_null_unit"] == "cell"
    assert o[("sandbox", "within_workload")]["order_null_unit"] == "cell within workload" and o[("sandbox", "within_workload")]["consequence"] == "size"
    assert o[("idle", "within_class")]["verdict"] in (V.NULL_NOT_ESTIMABLE, V.ORDER_NOT_AUDIBLE)
    assert jload(det(out) / "order.params.json")["params"]["scope"] == "within_class"
    on = corpus("on")
    GD.order_test(on, rung="combined", n_perm=20, n_estimators=N_EST, n_perm_required=20)
    o = {(r["class"], r["half_rule"]): r for r in rows(det(on) / "order.csv") if r["rung"] == "combined"}
    assert o[("sandbox", "within_class")]["verdict"] == V.ORDER_AUDIBLE and o[("sandbox", "within_class")]["order_null_unit"] == "workload"
    assert o[("sandbox", "within_class")]["n_assignments"] == "70"
    assert o[("benign_kernel", "within_class")]["verdict"] in (V.ORDER_AUDIBLE, V.ORDER_NOT_AUDIBLE, V.NULL_NOT_ESTIMABLE)
    GD.order_test(on, rung="combined", n_perm=20, n_estimators=N_EST, n_perm_required=20, consequence="void", half_rules=("within_class",))
    o = {(r["class"], r["half_rule"]): r for r in rows(det(on) / "order.csv") if r["rung"] == "combined"}
    assert o[("sandbox", "within_class")]["verdict"] == V.ORDER_VOID and V.is_refusal(V.ORDER_VOID)
    tiny = corpus("tiny")
    GD.order_test(tiny, rung="content", n_perm=2, n_estimators=N_EST)
    assert all(r["verdict"].startswith("not run: order_index missing for") for r in rows(det(tiny) / "order.csv"))
    GD.order_test(on, rung="apf", n_perm=4, n_estimators=N_EST, scope="campaign", half_rules=("within_class",))
    assert row_where(det(on) / "order.csv", rung="apf", **{"class": "all"})["scope"] == "campaign"


def test_drift_none_and_slope():
    out = corpus("main")
    GD.drift_regression(out, n_perm=40)
    d = {(r["class"], r["unit"]): r for r in rows(det(out) / "drift.csv")}
    assert d[("sandbox", "within_workload")]["verdict"] == V.DRIFT_NONE and d[("benign_kernel", "within_workload")]["verdict"] == V.DRIFT_NONE
    assert d[("sandbox", "plain")]["label"] == "confounded with workload order (blocked)" and d[("sandbox", "within_workload")]["label"] == "disclosure"
    on = corpus("on")
    GD.drift_regression(on, n_perm=100)
    d = {(r["class"], r["unit"]): r for r in rows(det(on) / "drift.csv")}
    assert d[("sandbox", "within_workload")]["verdict"] == V.DRIFT_SLOPE and float(d[("sandbox", "within_workload")]["slope"]) > 0
    assert d[("benign_kernel", "within_workload")]["verdict"] == V.DRIFT_SLOPE
    tiny = corpus("tiny")
    GD.drift_regression(tiny, n_perm=5)
    assert all(r["verdict"].startswith("not run: order_index missing") for r in rows(det(tiny) / "drift.csv"))


# --------------------------------------------------------------------------- G-SIG (3.5.8)

def test_gsig_reported_identity_and_not_run():
    o = tmp_out(); fixture_selection(o)
    write_fixture_scores(o, "content", "lowo", {"tpr_05": 0.5, "auc": 0.8})
    write_fixture_scores(o, "content", "loco", {"tpr_05": 0.9, "auc": 0.95})
    GD.gsig(o, "content")
    r = row_where(det(o) / "gsig.csv", rung="content")
    assert r["verdict"] == V.GSIG_REPORTED and float(r["gap_tpr05"]) == pytest.approx(0.4) and float(r["gap_auc"]) == pytest.approx(0.15)
    write_fixture_scores(o, "content", "lowo", {"tpr_05": 0.1, "null": {"status": "ok", "tpr05": {"verdict": V.NULL_INSIDE}, "auc": {"verdict": V.NULL_INSIDE}}})
    GD.gsig(o, "content")
    r = row_where(det(o) / "gsig.csv", rung="content")
    assert r["verdict"] == V.GSIG_IDENTITY and V.is_refusal(r["verdict"]) and float(r["gap_tpr05"]) == pytest.approx(0.8)
    write_fixture_scores(o, "content", "loco", {"tpr_05": 0.9, "null": {"status": V.not_run("null not requested")}})
    GD.gsig(o, "content")
    r = row_where(det(o) / "gsig.csv", rung="content")
    assert r["verdict"] == V.not_run("loco null not run") and float(r["gap_tpr05"]) == pytest.approx(0.8)
    out = corpus("main"); split("main", "content", "lowo"); split("main", "content", "loco", loco_mode="rep_index", quarantine=False)
    GD.gsig(out, "content")
    assert row_where(det(out) / "gsig.csv", rung="content")["verdict"] in (V.GSIG_REPORTED, V.GSIG_IDENTITY)


# --------------------------------------------------------------------------- G-FP (3.5.9)

def test_gfp_attributed_and_inseparable():
    out = corpus("main"); split("main", "content", "lowo")
    GD.gfp(out, "content", n_estimators=N_EST)
    g = {r["family"]: r for r in rows(det(out) / "gfp.csv") if r["rung"] == "content"}
    assert set(g) == {"kernels", "idle"} and g["idle"]["verdict"] == V.GFP_ATTRIBUTED and g["idle"]["tpr05_without"] == ""
    preds = rows(DM.split_dir(out, "content", "W8_H4", "lowo", True) / "predictions.csv")
    fake = [dict(r) for r in preds]
    for r in fake:
        if r["family"] == "idle":
            r["flag_05"] = "true"
    GD.gfp(out, "content", n_estimators=N_EST, predictions=fake)
    g = {r["family"]: r for r in rows(det(out) / "gfp.csv") if r["rung"] == "content"}
    assert g["idle"]["verdict"] == V.GFP_INSEPARABLE and V.is_refusal(g["idle"]["verdict"]) and g["idle"]["fraction"] == "1"
    assert g["idle"]["tpr05_without"] != "" and DM.split_dir(out, "content", "W8_H4", "lowo__without_idle", True).is_dir()
    assert "idle" not in scores(DM.split_dir(out, "content", "W8_H4", "lowo__without_idle", True))["per_family_fpr"]


# --------------------------------------------------------------------------- G-1C (3.5.10)

def test_g1c_primary_and_search():
    out = corpus("main")
    DM.run_one_class(out, "content", "W8_H4", n_perm=0, n_estimators=N_EST)
    GD.g1c(out, "content")
    r = row_where(det(out) / "g1c.csv", rung="content", directory="one_class__norm")
    assert r["label"] == V.G1C_PRIMARY and r["primary"] == "true" and r["verdict"] == V.PASS and r["model"] == "isolation_forest"
    o = tmp_out(); fixture_selection(o)
    for name in ("one_class__gmm", "one_class__ocsvm"):
        write_fixture_scores(o, "apf", name, {"model": name.split("__")[1], "primary": False, "threshold_source": "inner_lowo"})
    GD.g1c(o, "apf")
    rs = [r for r in rows(det(o) / "g1c.csv") if r["rung"] == "apf"]
    assert len(rs) == 2 and all(r["verdict"] == V.G1C_SEARCH and r["label"] == V.G1C_SECONDARY for r in rs) and V.is_refusal(V.G1C_SEARCH)
    GD.g1c(o, "wapf")
    assert row_where(det(o) / "g1c.csv", rung="wapf")["verdict"] == V.not_run("one-class split not run")


# --------------------------------------------------------------------------- the harness clause (3.5.11)

def test_harness_stage2_absent_and_present():
    out = corpus("main")
    GD.harness(out, "content")
    r = row_where(det(out) / "harness.csv", rung="content")
    assert r["verdict"] == V.HARNESS_STAGE2_ABSENT and r["margin_sb"] == ""
    tiny = corpus("tiny")
    j = rows(C.join_path(tiny))
    assert {x["class"] for x in j} >= {"benign_relaunched", "harness_idle", "sandbox", "benign_kernel", "idle"}
    assert all(x["workload_key"] == "gemm" for x in j if x["class"] == "benign_relaunched")
    split("tiny", "content", "lowo", n_perm=0, quarantine=False)
    GD.harness(tiny, "content")
    hs = [r for r in rows(det(tiny) / "harness.csv") if r["rung"] == "content"]
    feats = [r for r in hs if r["block"] == "features"]
    assert len(feats) == 36 and all(r["margin_sb"] != "" and r["margin_hi"] != "" and r["margin_rp"] != "" for r in feats)
    assert all(r["verdict"] in (V.HARNESS_COMPARABLE, V.HARNESS_CLASS_EXCEEDS) for r in feats) and any(r["verdict"] == V.HARNESS_COMPARABLE for r in feats)
    rel = [r for r in hs if r["block"] == "relaunch"][0]
    assert rel["verdict"] in (V.HARNESS_RELAUNCH_NOT_CLASS, V.PASS) and rel["relaunched_flagged_fraction"] != ""
    gk = {r["cell_id"]: r for r in rows(det(tiny) / "gk0_cells.csv")}
    assert all(gk[c]["verdict"] == "control" for c in gk if c.startswith("harness_idle_")) and all(gk[c]["harness_env_K_med"] != "" for c in gk)


# --------------------------------------------------------------------------- G-CAL (3.5.12)

def test_gcal_agree_and_perfold():
    out = corpus("main"); split("main", "content", "lowo")
    GD.gcal(out, "content")
    r = row_where(det(out) / "gcal.csv", rung="content")
    assert r["verdict"] in (V.GCAL_AGREE, V.GCAL_PERFOLD) and r["per_fold_tpr05"] != "" and r["pooled_tpr_at_fpr05"] != ""
    o = tmp_out(); fixture_selection(o)
    write_fixture_scores(o, "apf", "lowo", {"gcal": {"per_fold_tpr05": 0.5, "pooled_tpr_at_fpr05": 0.52, "difference": -0.02, "null_spread": 0.1, "verdict": V.GCAL_AGREE}})
    GD.gcal(o, "apf")
    assert row_where(det(o) / "gcal.csv", rung="apf")["verdict"] == V.GCAL_AGREE
    write_fixture_scores(o, "apf", "lowo", {"gcal": {"per_fold_tpr05": 0.5, "pooled_tpr_at_fpr05": 0.9, "difference": -0.4, "null_spread": 0.1, "verdict": V.GCAL_PERFOLD}})
    GD.gcal(o, "apf")
    assert row_where(det(o) / "gcal.csv", rung="apf")["verdict"] == V.GCAL_PERFOLD
    write_fixture_scores(o, "wapf", "lowo", {"gcal": {"per_fold_tpr05": 0.5, "pooled_tpr_at_fpr05": 0.9, "difference": -0.4, "null_spread": None, "verdict": V.not_run("null spread unavailable")}})
    GD.gcal(o, "wapf")
    assert row_where(det(o) / "gcal.csv", rung="wapf")["verdict"] == V.not_run("null spread unavailable")


# --------------------------------------------------------------------------- G-M and G-DIM (3.5.13)

def test_exact_sign_test_and_gm():
    assert GD.exact_sign_test(5, 0) == pytest.approx(0.03125) and GD.exact_sign_test(7, 1) == pytest.approx(0.03515625) and GD.exact_sign_test(3, 0) == 0.125
    assert GD.exact_sign_test(8, 1) == pytest.approx(0.01953125) and GD.exact_sign_test(0, 0) == 1.0
    o = tmp_out(); fixture_selection(o)
    out_a = {f"w{i}": 1.0 for i in range(6)}; out_b = {f"w{i}": 0.0 for i in range(6)}
    write_fixture_scores(o, "content", "lowo", {"tpr_05": 1.0, "per_workload_outcome": out_a})
    write_fixture_scores(o, "persist", "lowo", {"tpr_05": 0.0, "per_workload_outcome": out_b})
    GD.gm(o, n_seeds=0)
    r = row_where(det(o) / "gm.csv", row_a="content", row_b="persist")
    assert r["improving"] == "6" and r["worsening"] == "0" and float(r["p_exact"]) == pytest.approx(1 / 64) and r["verdict"] == V.not_run("seed spread unavailable")
    out = corpus("main"); split("main", "content", "lowo")
    DM.run_detection_split(out, "apf", "W8_H4", "lowo", n_perm=0, n_estimators=N_EST, quarantine=False)
    GD.gm(out, n_seeds=2, n_estimators=N_EST)
    rs = rows(det(out) / "gm.csv")
    r = row_where(det(out) / "gm.csv", row_a="content", row_b="apf")
    assert r["verdict"] in (V.GM_BEATS, V.GM_DIFFERENCE) and r["spread"] != ""
    assert int(r["improving"]) + int(r["worsening"]) + int(r["ties"]) == 14          # member 6 is at floor: no outcome, not a unit
    assert DM.split_dir(out, "apf", "W8_H4", "lowo_seed1", True).is_dir()
    GD.gdim(out)
    g = {r["row"]: r for r in rows(det(out) / "gdim.csv")}
    assert g["content"]["status"] == V.GDIM_FULL and g["content"]["d"] == "36"
    assert g["combined (matched)"]["status"].startswith("not run") or g["combined (matched)"]["status"] == V.GDIM_REDUCED


# --------------------------------------------------------------------------- the alias falsifier and the leak probes (3.5.14)

def _set_dt(out: Path, fn) -> dict:
    """Rewrite dt_est_s in every sidecar by fn(cell_id, old); returns the originals."""
    olds = {}
    for c in [j["cell_id"] for j in C.load_join(out)]:
        p = S.sidecar_path(out, c); sc = S.read_json(p); olds[c] = sc["dt_est_s"]
        sc["dt_est_s"] = fn(c, sc["dt_est_s"]); p.write_text(json.dumps(sc))
    return olds


def test_alias_stays_moves_and_cadence_row():
    out = corpus("main"); split("main", "content", "lowo")
    rng = np.random.default_rng(7)
    olds = _set_dt(out, lambda c, old: float(old) * (1.0 + 0.05 * rng.normal()))
    try:
        GD.alias_detection(out, "content", n_perm=6, n_perm_required=6)
        rs = [r for r in rows(det(out) / "alias.csv") if r["rung"] == "content"]
        feat = [r for r in rs if r["regressor"] == "dt_est_s" and r["feature"] != "cadence_as_class"]
        assert len(feat) == GD.ALIAS_TOP_K and all(r["verdict"] == V.ALIAS_STAYS and r["unit"] == "within_workload" for r in feat)
        assert row_where(det(out) / "alias.csv", rung="content", regressor="iteration_count")["verdict"] == V.not_run("no iteration count (stage 1)")
        cad = row_where(det(out) / "alias.csv", rung="content", feature="cadence_as_class")
        assert cad["verdict"] in (V.CADENCE_AUDIBLE, V.LEAK_NOT_AUDIBLE) and cad["slope"] != ""
        # dt proportional to the strongest feature: the feature moves with the interval
        d = DM.load_detection_data(out, "content", "W8_H4"); lab = d["lab"]
        imp = scores(split("main", "content", "lowo"))["importance_mean"]
        top = max(imp, key=imp.get); j = d["names"].index(top)
        vals = {c: float(lab["X"][i, j]) for i, c in enumerate(lab["cell_id"])}
        _set_dt(out, lambda c, old: 0.5 + 0.1 * vals[c])
        GD.alias_detection(out, "content", n_perm=0)
        assert row_where(det(out) / "alias.csv", rung="content", feature=top, regressor="dt_est_s")["verdict"] == V.ALIAS_MOVES
    finally:
        _set_dt(out, lambda c, old: olds[c])


def test_leak_probe_not_audible_and_audible():
    out = corpus("main")
    GD.leak_probe(out, n_perm=6, n_perm_required=6)
    lp = {r["quantity"]: r for r in rows(det(out) / "leak_probe.csv")}
    assert set(lp) == set(GD.LEAK_QUANTITIES) and lp["n_pairs"]["verdict"] == V.LEAK_NOT_AUDIBLE and lp["dt_est_s"]["verdict"] == V.LEAK_NOT_AUDIBLE
    assert lp["K_med"]["note"].startswith("level, not a leak") and lp["n_pairs"]["n_assignments"] == "6435"
    tiny = corpus("tiny")
    GD.leak_probe(tiny, n_perm=6, n_perm_required=6)
    lp = {r["quantity"]: r for r in rows(det(tiny) / "leak_probe.csv")}
    assert float(lp["n_pairs"]["auc"]) == 1.0 and lp["n_pairs"]["verdict"] in (V.LEAK_AUDIBLE, V.NULL_NOT_ESTIMABLE)
    prm = jload(det(tiny) / "leak_probe.params.json")["params"]["per_class"]
    assert prm["n_pairs"]["median_sandbox"] == 32.0 and prm["n_pairs"]["median_benign"] == 40.0


# --------------------------------------------------------------------------- G-V two-class (3.5.15)

def test_gv_two_class_report():
    out = corpus("main")
    GD.gv_two_class(out, "content")
    rs = [r for r in rows(det(out) / "gv_two_class.csv") if r["rung"] == "content"]
    assert len(rs) == 36 and list(rs[0].keys()) == list(GD.GV_COLUMNS) and all(r["L0"] != "" and r["L3"] != "" and r["L3_families"] != "" for r in rs)
    s = row_where(det(out) / "gv_two_class_summary.csv", rung="content")
    assert s["n_features"] == "36" and s["note"] in ("", "a class whose L3 is inside the benign families' mutual spread has no more form than any two families have between them")
    m = {r["member_index"]: r for r in rows(det(out) / "gv_two_class_members.csv") if r["rung"] == "content"}
    assert set(m) == {str(i) for i in range(1, 9)} and all(r["L0_ratio"] != "" for r in m.values())
    assert all(r["note"] in ("", V.REPS_NEAR_IDENTICAL) for r in m.values())
    assert jload(det(out) / "gv_two_class_members.params.json")["params"]["reps_identical_ratio"] == GD.REPS_IDENTICAL_RATIO


# --------------------------------------------------------------------------- the miss table (3.5.16)

def test_miss_table_and_fp_table():
    out = corpus("main"); split("main", "content", "lowo")
    GD.miss_table(out, "content")
    ms = [r for r in rows(det(out) / "miss_table.csv") if r["rung"] == "content"]
    assert list(ms[0].keys()) == list(GD.MISS_COLUMNS)
    m6 = [r for r in ms if r["member_index"] == "6"]
    assert len(m6) == 3 and all(r["status"] == V.AT_FLOOR_NOT_A_MISS and r["nearest_workload"] == "" for r in m6)
    misses = [r for r in ms if r["status"] == "miss"]
    assert misses and all(r["nearest_family"] == "kernels" and r["axis"] in ("amount", "identity") and r["axis"] != r["axis_of_largest"] for r in misses)
    assert all(r["d_amount"] != "" and r["d_identity"] != "" and float(r["distance"]) >= 0 for r in misses)
    m5 = [r for r in misses if r["member_index"] == "5"]
    assert m5 and all(r["axis"] == "amount" and r["axis_of_largest"] == "identity" for r in m5)       # double content, churn 0.60: identity is where it differs
    m7 = [r for r in misses if r["member_index"] == "7"]
    assert m7 and all(float(r["distance"]) < 1.0 for r in m7)
    assert (det(out) / "fp_table.csv").is_file() and list(rows(det(out) / "fp_table.csv")[0].keys()) == list(GD.FP_COLUMNS)
    prm = jload(det(out) / "miss_table.params.json")["params"]
    assert prm["identity"] == "excess" and prm["plane_mask"].startswith("unmasked") and prm["distance"] == GD.MISS_DISTANCE
    # with gates/gj.json present the mask is one per-pair rule for every class
    gj = out / "gates" / "gj.json"
    with swap_file(gj, json.dumps({"schema": "x", "params": {"k_factor": 1.5}, "citation": "c", "floor_median_K": 150.0})):
        GD.miss_table(out, "content", identity="raw", distance="nearest_cell")
        prm = jload(det(out) / "miss_table.params.json")["params"]
        assert prm["plane_mask"] == "gj_mask_K, every class" and prm["identity"] == "raw" and prm["k_factor"] == 1.5
        ms = [r for r in rows(det(out) / "miss_table.csv") if r["rung"] == "content" and r["status"] == "miss"]
        assert ms
    GD.miss_table(out, "content")


# --------------------------------------------------------------------------- the head-drop template and the CLI

def test_head_drop_template_and_cli_exit_codes():
    out = corpus("main")
    p = GD.head_drop_template(out)
    hd = S.load_head_drop(p)
    assert set(hd) >= {f"sandbox_member_{m}" for m in range(1, 9)} | {"gemm", "idle"} and all(v == 0 for v in hd.values())
    r = run_cli("gates_detection", "gn", "--out", "/nonexistent/out", check=False)
    assert r.returncode == 2 and "missing input" in r.stderr
    r = run_cli("gates_detection", "gl", "--out", out, "--rung", "content", check=False)
    assert r.returncode == 0
    o = tmp_out(); S.write_csv(o / "cells.csv", ("cell_id",), []); (o / "inputs").mkdir()
    GD.gl_two_class(o, "content")
    assert row_where(det(o) / "gl.csv", rung="content")["verdict"] == V.not_run("no selection for content (run classes inherit-selection)")
