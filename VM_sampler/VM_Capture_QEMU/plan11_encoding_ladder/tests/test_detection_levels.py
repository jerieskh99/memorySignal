"""detection_levels.py: level 2 (the sub-family rows, B's string, the per-letter null) and level 3
(the signature ceiling, the cell-level null, the one-member refusal)."""
from __future__ import annotations

from _det_common import corpus, rows, jload, run_cli, restrict_admissible, C, V, DL, DM, N_EST


def test_level2_rows_per_letter_null_and_denominator():
    out = corpus("main")
    d = DL.run_level2(out, "content", "W8_H4", n_perm=300, n_estimators=10, n_perm_required=300)
    conf = {r["true_subfamily"]: r for r in rows(d / "confusion.csv")}
    assert set(conf) == {"A", "B", "C"}
    a = conf["A"]
    assert a["n_members"] == "4" and a["status"] == V.GN_HEADLINE and float(a["recall"]) >= 0.75 and a["verdict"] in (V.PASS, V.NULL_INSIDE)
    assert a["pred_A"] and a["null_p95"] != "" and a["rank"].startswith("rank ")
    b = conf["B"]
    assert b["status"] == V.level2_no_heldout(1) == "one member, no held-out test" and b["recall"] == "" and b["pred_A"] == ""
    c = conf["C"]
    assert c["n_members"] == "3" and c["status"] == V.GN_HEADLINE and c["n_at_floor"] == "3" and c["n_in_denominator"] == "6"
    sc = jload(d / "scores.json")
    assert sc["status"] == "ok" and sc["null"]["exhaustive"] and sc["null"]["n_assignments"] == 280 and sc["null"]["n_perm"] == 280
    assert set(sc["null"]["per_letter"]) == {"A", "C"} and isinstance(sc["macro_recall"], float) and "verdict" not in sc
    assert sc["headline_letters"] == ["A", "C"] and sc["n_at_floor"] == 3 and sc["params"]["train_on_at_floor"] is False
    mem = {r["member_index"]: r for r in rows(d / "members.csv")}
    assert mem["6"]["denominator"] == "0" and mem["6"]["at_floor"] == "3" and mem["1"]["eighths"].endswith("/3")
    pr = rows(d / "predictions.csv")
    assert all(r["in_denominator"] == "false" for r in pr if r["member_index"] == "6") and len(pr) == 24
    nj = jload(d / "null.json")
    assert set(nj["arrays"]) == {"A", "B", "C"} and len(nj["arrays"]["A"]) == 280


def test_level2_smoke_null_and_no_headline():
    out = corpus("main")
    d = DL.run_level2(out, "content", "W8_H4", n_perm=5, n_estimators=10)
    conf = {r["true_subfamily"]: r for r in rows(d / "confusion.csv")}
    assert conf["A"]["verdict"] == V.not_run("5 permutations < 500")
    sc = jload(d / "scores.json")
    assert not sc["null"]["exhaustive"] and sc["null"]["n_perm"] == 5
    d = DL.run_level2(out, "content", "W8_H4", n_perm=0, n_estimators=10, min_members_headline=5)
    sc = jload(d / "scores.json")
    assert sc["status"] == V.not_applicable("no headline sub-family") and sc["macro_recall"] == V.not_applicable("no headline sub-family")
    conf = {r["true_subfamily"]: r for r in rows(d / "confusion.csv")}
    assert conf["A"]["status"] == V.GN_HEADLINE or conf["A"]["status"] == V.L2_ONE_TRAIN or conf["A"]["status"].endswith("no held-out test")
    d = DL.run_level2(out, "content", "W8_H4", n_perm=0, n_estimators=10, min_members_test=5)
    conf = {r["true_subfamily"]: r for r in rows(d / "confusion.csv")}
    assert conf["A"]["status"] == V.level2_no_heldout(4) == "4 members, no held-out test"


def test_level3_signature_ceiling_and_one_member():
    out = corpus("main")
    d = DL.run_level3(out, "combined", "W8_H4", n_perm=4, n_estimators=10, n_perm_required=4)
    conf = rows(d / "confusion.csv")
    assert len(conf) == 8 and all(r["label"] == V.SIGNATURE_CEILING for r in conf)
    assert all(f"pred_{m}" in conf[0] for m in range(1, 9)) and conf[5]["true_member"] == "6" and conf[5]["n_in_denominator"] == "0" and conf[5]["recall"] == ""
    sc = jload(d / "scores.json")
    assert sc["status"] == "ok" and isinstance(sc["accuracy"], float) and sc["label"] == V.SIGNATURE_CEILING and sc["n_folds"] == 3
    assert sc["null"]["null_unit"] == "cell" and sc["null"]["verdict"] in (V.PASS, V.NULL_INSIDE) and sc["params"]["null_unit_reason"]
    assert (d / "members.csv").is_file() and (d / "predictions.csv").is_file() and (d / "null.json").is_file()
    d = DL.run_level3(out, "combined", "W8_H4", split="cell", n_perm=0, n_estimators=10)
    assert jload(d / "scores.json")["n_folds"] == 24
    with restrict_admissible(out, lambda r: r["class"] == "sandbox" and not r["cell_id"].startswith("sandbox_member_1_")):
        d = DL.run_level3(out, "combined", "W8_H4", n_perm=0, n_estimators=10)
        assert jload(d / "scores.json")["status"] == V.not_applicable("one member")
        assert rows(d / "confusion.csv")[0]["label"] == V.not_applicable("one member")


def test_level_cli_exit_codes():
    r = run_cli("detection_levels", "level2", "--out", "/nonexistent/out", "--rung", "apf", check=False)
    assert r.returncode == 2
