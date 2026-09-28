"""synth_detection.py: the two-class corpus (member presets, the realized order, the class file,
the boundaries file), the truth against the extract, the refusals."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from _det_common import corpus, corpus_root, rows, jload, run_cli, tmp_out, C, S, SD
from plan11_encoding_ladder import schema, synth


def test_member_presets_and_letters():
    assert SD.MEMBER_LETTERS == {1: "A", 2: "A", 3: "A", 4: "A", 5: "B", 6: "C", 7: "C", 8: "C"}
    p1, p2, p3, p4 = (SD.member_preset(m) for m in (1, 2, 3, 4))
    assert p1["content"] == "spin" and p1["spin_bytes"] == (48, 96) and [p["K0"] for p in (p1, p2, p3, p4)] == [2048, 4096, 1024, 3072]
    assert SD.member_preset(5) == dict(K0=2048, content="double", churn=0.60, k_noise=0.02)
    assert SD.member_preset(6)["K0"] == 0 and SD.member_preset(6)["content"] == "idle"
    assert SD.member_preset(7) == synth.PRESETS["spmm"]
    assert SD.member_preset(8)["K0"] == SD.MEMBER8_K0 == 40000 and SD.member_preset(8, member8_k0=10000)["K0"] == 10000
    assert SD.member_preset(1, conjunction=True)["spin_bytes"] == (200, 300)
    with pytest.raises(ValueError):
        SD.member_preset(9)
    with pytest.raises(ValueError):
        SD.corpus_specs(members="9")


def test_corpus_specs_layout_off_and_on():
    recs, layout = SD.corpus_specs(n_pairs=20, reps=4, idle=4, kernels="gemm,gibbs", members="1,2")
    assert len(recs) == 8 + 8 + 4 and layout["order_confound"] == "off"
    order = sorted(recs, key=lambda r: r["order_index"])
    assert [r["class"] for r in order[:8]] == ["benign_kernel"] * 8 and [r["class"] for r in order[8:16]] == ["sandbox"] * 8 and [r["class"] for r in order[16:]] == ["idle"] * 4
    # off: every workload has cells in both halves of its class block
    for wk in ("gemm", "gibbs", "sandbox_member_1", "sandbox_member_2"):
        pos = sorted(r["order_index"] for r in recs if r["workload_key"] == wk)
        block = [r["order_index"] for r in recs if r["class"] == (recs[[x["workload_key"] for x in recs].index(wk)]["class"])]
        mid = min(block) + len(block) // 2
        assert any(p < mid for p in pos) and any(p >= mid for p in pos), wk
    assert {r["spec"].k_noise for r in recs if r["class"] == "sandbox"} == {0.02}
    assert all(r["spec"].family == "synthfam" and r["spec"].test_label == f"synthfam_member_{r['member_index']}" for r in recs if r["class"] == "sandbox")
    recs, layout = SD.corpus_specs(n_pairs=20, reps=2, idle=2, kernels="gemm,gibbs", members="1,2,3", order_confound="on")
    m = {r["member_index"]: r["spec"].k_noise for r in recs if r["class"] == "sandbox"}
    assert m == {1: pytest.approx(0.02), 2: pytest.approx(0.05), 3: pytest.approx(0.08)}
    pos = {wk: sorted(r["order_index"] for r in recs if r["workload_key"] == wk) for wk in ("sandbox_member_1", "sandbox_member_2", "sandbox_member_3")}
    assert pos["sandbox_member_1"] == [5, 6] and pos["sandbox_member_2"] == [7, 8] and pos["sandbox_member_3"] == [9, 10]    # blocks by member
    k = {r["workload_key"]: r["spec"].k_noise for r in recs if r["class"] == "benign_kernel"}
    assert k["gemm"] == pytest.approx(0.02) and k["gibbs"] == pytest.approx(0.03)
    recs, _ = SD.corpus_specs(n_pairs=20, reps=2, idle=2, kernels="gemm", members="1", drift_level=True)
    k0 = sorted((r["order_index"], r["spec"].K0) for r in recs if r["workload_key"] == "gemm")
    assert k0[1][1] > k0[0][1]
    recs, _ = SD.corpus_specs(n_pairs=20, reps=3, idle=4, kernels="gemm,fft", members="1", campaign_labels="round_robin", idle_campaigns=2)
    labels = {r["spec"].label for r in recs if r["class"] == "benign_kernel"}
    assert labels == set(SD.CAMPAIGN_LABELS_RR)
    idle = [r["spec"] for r in recs if r["class"] == "idle"]
    assert {s.label for s in idle} == set(SD.IDLE_CAMPAIGN_LABELS) and {s.floor_F for s in idle} == {150, 190} and {s.floor_churn for s in idle} == {0.02, 0.10}
    recs, _ = SD.corpus_specs(n_pairs=20, reps=2, idle=2, kernels="gemm", members="1", drift_level=True, drift_rate=0.5)
    k0 = sorted((r["order_index"], r["spec"].K0) for r in recs if r["workload_key"] == "gemm")
    assert k0[1][1] == round(4096 * 1.5)
    recs, _ = SD.corpus_specs(n_pairs=20, reps=1, idle=1, kernels="gemm,fft,rmat_gen", members="1", campaign_labels="confounded")
    lk = {r["workload_key"]: (r["spec"].label, r["spec"].k_noise) for r in recs if r["class"] == "benign_kernel"}
    assert lk["gemm"][0] == "sandbox_deepdive_01c" and lk["fft"][0] == "sandbox_deepdive_01c1" and lk["rmat_gen"][0] == "dwarfs1_synth" and lk["rmat_gen"][1] == 0.08
    recs, _ = SD.corpus_specs(n_pairs=20, reps=1, idle=1, kernels="gemm", members="1", stage2_fixture=True, sandbox_n_pairs=12)
    assert sum(1 for r in recs if r["class"] == "benign_relaunched") == 4 and sum(1 for r in recs if r["class"] == "harness_idle") == 4
    assert next(r["spec"].n_pairs for r in recs if r["class"] == "sandbox") == 12
    assert next(r["spec"].test_label for r in recs if r["class"] == "harness_idle") == "harness_floor"


def test_written_corpus_classes_and_boundaries():
    root = corpus_root("main")
    cls = rows(root / "classes.csv")
    assert list(cls[0].keys()) == list(SD.CLASSES_COLUMNS)
    wl = [r for r in cls if not r["order_index"]]
    assert len(wl) == 8 + 2 and {r["class"] for r in wl} == {"sandbox", "benign_kernel", "idle"}
    assert all(r["path_prefix"] == f"synthfam/synthfam_member_{r['member_index']}" for r in wl if r["class"] == "sandbox")
    per_cell = [r for r in cls if r["order_index"]]
    assert len(per_cell) == 46 and len({r["order_index"] for r in per_cell}) == 46
    assert len(rows(root / "classes.no_order.csv")) == 10
    b = rows(root / "iteration_boundaries.csv")
    assert b and all(r["cell_id"].startswith(("gemm_", "floyd_")) for r in b) and all(";" in r["boundary_seqs"] or r["boundary_seqs"].isdigit() for r in b)
    man = jload(root / "corpus_detection.json")
    assert man["n_cells"] == 46 and man["params"]["member8_k0"] == 10000 and "expected_cell_id" in man["cells"][0]
    # no workload name of the family anywhere: the member label is an index
    assert all("synthfam_member_" in c["path"] for c in man["cells"] if c["class"] == "sandbox")


def test_truth_reproduces_the_extract_on_a_member_cell():
    root, out = corpus_root("main"), corpus("main")
    man = jload(root / "corpus_detection.json")
    rec = next(c for c in man["cells"] if c["class"] == "sandbox" and c["member_index"] == 1 and c["rep"] == 0)
    truth = jload(Path(rec["path"]) / "truth.json")
    cid = rec["expected_cell_id"]
    assert cid == "sandbox_member_1__rep00__stage1" and (out / "extract" / cid / "extract.csv").is_file()
    with (out / "extract" / cid / "extract.csv").open(newline="") as f:
        ex = list(csv.DictReader(f))
    per = truth["per_seq_channels"]
    assert list(ex[0].keys()) == list(schema.EXTRACT_COLUMNS) and len(ex) == len(per["seq"])
    for i, row in enumerate(ex):
        for col in schema.EXTRACT_COLUMNS:
            tv = per[col][i]; ev = row[col]
            if tv is None:
                assert ev == ""
            elif col in schema.EXTRACT_INT_COLUMNS:
                assert int(ev) == int(tv)
            else:
                assert abs(float(ev) - float(tv)) <= 1e-9 * max(1.0, abs(float(ev)), abs(float(tv))), (i, col)


def test_cli_refuses_member_9():
    r = run_cli("synth_detection", "corpus", "--root", tmp_out() / "r", "--members", "9", "--n-pairs", "8", "--reps", "1", "--idle", "0", "--kernels", "gemm", check=False)
    assert r.returncode == 3 and "usage:" in r.stderr
