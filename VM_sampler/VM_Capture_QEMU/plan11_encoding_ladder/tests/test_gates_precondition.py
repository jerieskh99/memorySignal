"""gates_precondition.py: C1-C8, the failed count, G-K0, G-F (SPEC 3.3, 5.3)."""
import json

from _b2_common import S, V, SY, corpus, tmp_out, rows, row_where, jload, admissibility
from plan11_encoding_ladder import gates_precondition as GP


def test_c1_pass_refuse_and_idle_not_applicable():
    out = tmp_out()
    SY.write_corpus(out, [SY.SynthSpec("gemm", 42, K0=6000, content="double"), SY.SynthSpec("gibbs", 42, K0=100, content="spin"),
                          SY.SynthSpec("sleep", 1, role="idle", **SY.IDLE_PRESET)])
    GP.gate_preconditions(out)
    p = out / "gates" / "preconditions.csv"
    assert row_where(p, cell_id="gemm__rep00__01c")["C1"] == V.PASS
    assert row_where(p, cell_id="gibbs__rep00__01c")["C1"] == V.FAIL
    idle = row_where(p, cell_id="idle__rep00__01c")
    assert idle["C1"] == GP.C1_IDLE_STRING and idle["all_hard_pass"] == "true"
    assert jload(out / "gates" / "preconditions.json")["excluded_cells"] == ["gibbs__rep00__01c"]
    assert row_where(p, cell_id="gemm__rep00__01c")["C7"] == GP.C7_PENDING
    assert row_where(p, cell_id="gemm__rep00__01c")["C4"] == GP.C4_STRING and row_where(p, cell_id="gemm__rep00__01c")["C8"] == GP.C8_STRING


def test_c2_c3_c6_pass_and_refuse():
    out = tmp_out()
    SY.write_corpus(out, [SY.SynthSpec("gemm", 42, K0=6000, n_pairs=120), SY.SynthSpec("fft", 42, K0=6000, n_pairs=4),
                          SY.SynthSpec("floyd", 42, K0=6000, n_pairs=120, status="refused: seq not monotone at row 77"),
                          SY.SynthSpec("gibbs", 42, K0=6000, n_pairs=120, gap_seqs=(5, 9))])
    GP.gate_preconditions(out)
    p = out / "gates" / "preconditions.csv"
    g = row_where(p, cell_id="gemm__rep00__01c")
    assert g["C2"] == V.PASS and g["C3"] == V.PASS and g["C6"] == V.PASS and g["C6_reason"] == "n_seq_gaps 0"
    f = row_where(p, cell_id="fft__rep00__01c")
    assert f["C2"] == V.FAIL and f["C3"] == V.FAIL and f["all_hard_pass"] == "false"
    fl = row_where(p, cell_id="floyd__rep00__01c")
    assert fl["C6"] == V.FAIL and "seq not monotone" in fl["C6_reason"] and fl["all_hard_pass"] == "false"
    gb = row_where(p, cell_id="gibbs__rep00__01c")
    assert gb["C6"] == V.PASS and gb["C6_reason"] == "n_seq_gaps 2"       # a gap never fails C6


def test_c6_header_reference_is_modal_and_ties_refuse():
    out = tmp_out()
    SY.write_corpus(out, [SY.SynthSpec("gemm", 42, K0=6000), SY.SynthSpec("fft", 42, K0=6000), SY.SynthSpec("floyd", 42, K0=6000)])
    sc = out / "extract" / "floyd__rep00__01c" / "sidecar.json"
    d = json.loads(sc.read_text()); d["header_sha256"] = "other"; sc.write_text(json.dumps(d))
    GP.gate_preconditions(out)
    p = out / "gates" / "preconditions.csv"
    assert row_where(p, cell_id="floyd__rep00__01c")["C6"] == V.FAIL and row_where(p, cell_id="gemm__rep00__01c")["C6"] == V.PASS
    sc2 = out / "extract" / "fft__rep00__01c" / "sidecar.json"
    d = json.loads(sc2.read_text()); d["header_sha256"] = "other"; sc2.write_text(json.dumps(d))
    out2 = tmp_out(); SY.write_corpus(out2, [SY.SynthSpec("gemm", 42, K0=6000), SY.SynthSpec("fft", 42, K0=6000)])
    sc3 = out2 / "extract" / "fft__rep00__01c" / "sidecar.json"
    d = json.loads(sc3.read_text()); d["header_sha256"] = "other"; sc3.write_text(json.dumps(d))
    GP.gate_preconditions(out2)
    assert all(r["C6"] == V.refused("header mismatch, no majority") for r in rows(out2 / "gates" / "preconditions.csv"))


def test_failed_count_verdicts():
    assert GP.failed_verdict(0) == (V.PASS, "recorded")
    assert GP.failed_verdict(2) == (V.refused("failed count 2 > 0, seq axis uncorrected"), "recorded")
    assert GP.failed_verdict(None) == (V.refused("failed count not recorded"), "not recorded")
    v, src = GP.failed_verdict(None, assume_failed_zero=True, assume_reason="AA A5: any failed job re-runs the whole cell")
    assert v == V.PASS and src == "declared zero: AA A5: any failed job re-runs the whole cell"
    out = tmp_out()
    SY.write_corpus(out, [SY.SynthSpec("gemm", 42, K0=6000, failed_count=None), SY.SynthSpec("fft", 42, K0=6000, failed_count=2),
                          SY.SynthSpec("floyd", 42, K0=6000, failed_count=0)])
    GP.gate_preconditions(out)
    p = out / "gates" / "preconditions.csv"
    assert row_where(p, cell_id="gemm__rep00__01c")["failed_verdict"] == V.refused("failed count not recorded")
    assert row_where(p, cell_id="fft__rep00__01c")["failed_verdict"].startswith("refused: failed count 2")
    assert row_where(p, cell_id="floyd__rep00__01c")["failed_verdict"] == V.PASS
    assert jload(out / "gates" / "preconditions.json")["excluded_cells_pair_rungs"] == ["gemm__rep00__01c", "fft__rep00__01c"]
    GP.gate_preconditions(out, assume_failed_zero=True, assume_reason="AA A5")
    g = row_where(p, cell_id="gemm__rep00__01c")
    assert g["failed_verdict"] == V.PASS and g["failed_source"] == "declared zero: AA A5"
    # inputs/failed_counts.csv overrides the sidecar
    S.write_csv(out / "inputs" / "failed_counts.csv", ("cell_id", "failed_count", "source"), [{"cell_id": "floyd__rep00__01c", "failed_count": 1, "source": "test"}])
    GP.gate_preconditions(out)
    assert row_where(p, cell_id="floyd__rep00__01c")["failed_verdict"].startswith("refused: failed count 1")


def test_gk0_relabels_the_lexer_and_not_histogram():
    out = corpus(reps=2, idle=3, kernels=["histogram", "lexer"])
    GP.write_gk0_template(out / "inputs" / "gk0_source.csv")
    GP.gate_gk0(out)
    p = out / "gates" / "gk0.csv"
    assert row_where(p, kernel="histogram")["verdict"] == V.GK0_ABOVE_FLOOR
    lx = row_where(p, kernel="lexer")
    assert lx["verdict"] == V.GK0_IDLE_MEASURED and lx["archetype_measured"] == "IDLE" and lx["source_statement"] == "no (after the first pass)"
    assert row_where(p, kernel="idle")["verdict"] == "control"
    assert S.gk0_relabel(out) == {"lexer": "IDLE"}
    out2 = corpus(reps=2, idle=0, kernels=["histogram", "lexer"])
    GP.gate_gk0(out2)
    assert row_where(out2 / "gates" / "gk0.csv", kernel="lexer")["verdict"] == V.not_run("no admissible idle cell")
    GP.gate_gk0(out, idle_pool="cell_medians")
    assert row_where(p, kernel="lexer")["verdict"] == V.GK0_IDLE_MEASURED


def test_gf_part1_inseparable_vs_void_and_part2_pass_vs_at_floor():
    out = corpus(reps=2, idle=6, kernels=["gemm", "lexer"], n_pairs=90)
    admissibility(out)
    GP.gate_gf(out, rung="apf", n_perm=30, n_estimators=10)
    p = out / "gates" / "gf.csv"
    assert row_where(p, part="i", kernel="idle")["verdict"] == V.GF_INSEPARABLE
    assert row_where(p, part="ii", kernel="gemm")["verdict"] == V.PASS
    assert row_where(p, part="ii", kernel="lexer")["verdict"] == V.GF_AT_FLOOR
    fl = jload(out / "gates" / "gf_floors.json")
    assert len(fl["K"]) == 5 and len(fl["J"]) == 5 and fl["n_idle_cells"] == 6
    # must refuse part (i): idle reps whose floor jitters differently per rep (visible after level normalization)
    out2 = corpus(reps=2, idle=6, kernels=["gemm"], n_pairs=90, idle_overrides=lambda r: dict(floor_noise=0.05 * (r + 1)))
    admissibility(out2)
    GP.gate_gf(out2, rung="apf", n_perm=30, n_estimators=10)
    assert row_where(out2 / "gates" / "gf.csv", part="i", kernel="idle")["verdict"] == V.GF_VOID
    GP.gate_gf(out2, rung="apf", n_perm=30, n_estimators=10, part1_consequence="report")
    assert row_where(out2 / "gates" / "gf.csv", part="i", kernel="idle")["verdict"] == GP.GF_SEPARABLE_REPORTED
    # the SPEC 5.3 level step (floor_F = 150 + 40 * rep) is removed by level normalization: inseparable under
    # the normalized rung, void under the raw features (part1_features = "raw")
    out3 = corpus(reps=2, idle=6, kernels=["gemm"], n_pairs=90, idle_overrides=lambda r: dict(floor_F=150 + 40 * r))
    admissibility(out3)
    GP.gate_gf(out3, rung="apf", n_perm=30, n_estimators=10)
    assert row_where(out3 / "gates" / "gf.csv", part="i", kernel="idle")["verdict"] == V.GF_INSEPARABLE
    GP.gate_gf(out3, rung="apf", n_perm=30, n_estimators=10, part1_features="raw")
    assert row_where(out3 / "gates" / "gf.csv", part="i", kernel="idle")["verdict"] == V.GF_VOID
    # the admissibility record missing -> not run on every row; no idle cell -> not run
    out5 = corpus(reps=2, idle=2, kernels=["gemm"], n_pairs=90)
    GP.gate_gf(out5, rung="apf", n_perm=5, n_estimators=10)
    assert all(r["verdict"] == V.not_run("no admissible idle cell; admissibility record missing") for r in rows(out5 / "gates" / "gf.csv"))
    out4 = corpus(reps=2, idle=0, kernels=["gemm"], n_pairs=90); admissibility(out4)
    GP.gate_gf(out4, rung="apf", n_perm=5, n_estimators=10)
    assert all(r["verdict"] == V.not_run("no admissible idle cell") for r in rows(out4 / "gates" / "gf.csv"))


def test_templates_and_cli_exit_codes():
    out = tmp_out()
    assert GP.main(["gk0-template", "--out", str(out)]) == 0 and (out / "inputs" / "gk0_source.csv").is_file()
    assert GP.main(["idle-admissibility-template", "--out", str(out)]) == 0
    assert set(json.loads((out / "inputs" / "idle_admissibility.json").read_text())) >= set(GP.ADMISSIBILITY_KEYS)
    assert GP.main(["preconditions", "--out", str(out)]) == 2          # cells.csv missing
    assert GP.main(["preconditions", "--out", str(out), "--assume-failed-zero"]) == 2


# ---------------------------------------------------------------------------------------------
# build epoch 2, builder 3 (SPEC_epoch2 section 4 and 6.3 items 1, 2): the C1 re-map of AD 2026-09-17
# ---------------------------------------------------------------------------------------------

def _c1_two_kernels(out, **extra):
    """Two kernel cells with no floor set: K_max about 300 (above 262) and about 100 (below)."""
    specs = [SY.SynthSpec("gemm", 42, K0=300, floor_F=0, content="double"),
             SY.SynthSpec("gibbs", 42, K0=100, floor_F=0, content="spin")]
    specs += extra.get("more", [])
    SY.write_corpus(out, specs)
    return out


def test_c1_remap_absolute_default_without_idle():
    # SPEC_epoch2 6.3 item 1: no idle cell -> the CLI default (auto) applies the declared absolute of 0.001 N = 262 pages
    out = _c1_two_kernels(tmp_out())
    assert GP.C1_ABS_PAGES == 262 and GP.C1_ABS_FRACTION == 0.001
    assert GP.main(["preconditions", "--out", str(out)]) == 0            # the CLI default is --c1-rule auto
    p = out / "gates" / "preconditions.csv"
    hi, lo = row_where(p, cell_id="gemm__rep00__01c"), row_where(p, cell_id="gibbs__rep00__01c")
    assert int(hi["C1_K_max"]) >= 262 > int(lo["C1_K_max"])
    assert hi["C1"] == V.PASS and lo["C1"] == V.FAIL
    assert hi["C1_rule"] == lo["C1_rule"] == "absolute_0.001"
    assert hi["C1_threshold_pages"] == lo["C1_threshold_pages"] == "262"
    prm = jload(out / "gates" / "preconditions.json")["params"]
    assert prm["C1_rule_requested"] == "auto" and prm["C1_rule_applied"] == "absolute" and prm["C1_rule_in_force"] == "absolute_0.001"
    assert prm["C1_idle_band_edge"] is None and prm["C1_idle_cells_in_floor"] == [] and prm["C1_abs_pages"] == 262
    assert prm["C1_threshold_pages"] == 262 and prm["C1_operand"].startswith("K_max")
    assert "AD 2026-09-17" in prm["C1_change_record"] and "legacy_apf_max" in prm["C1_change_record"]
    assert jload(out / "gates" / "preconditions.json")["excluded_cells"] == ["gibbs__rep00__01c"]
    # the documented alternative: the inherited apf_max >= 0.02 (5,243 pages) refuses both
    assert GP.main(["preconditions", "--out", str(out), "--c1-rule", "legacy_apf_max"]) == 0
    assert row_where(p, cell_id="gemm__rep00__01c")["C1"] == V.FAIL and row_where(p, cell_id="gibbs__rep00__01c")["C1"] == V.FAIL
    assert row_where(p, cell_id="gemm__rep00__01c")["C1_rule"] == "legacy_apf_max_0.02"
    assert row_where(p, cell_id="gemm__rep00__01c")["C1_threshold_pages"] == "5243"
    prm = jload(out / "gates" / "preconditions.json")["params"]
    assert prm["C1_rule_applied"] == "legacy_apf_max" and prm["C1_legacy_apf_max"] == 0.02 and prm["C1_operand"].startswith("apf_max")
    # the function default is the inherited rule (SPEC_epoch2_review_al_farabi.md 5.1): a direct call with no
    # argument keeps epoch 1's contract, and c1_activity_min=0.0 admits every kernel cell as before
    GP.gate_preconditions(out)
    assert row_where(p, cell_id="gemm__rep00__01c")["C1"] == V.FAIL and row_where(p, cell_id="gemm__rep00__01c")["C1_rule"] == "legacy_apf_max_0.02"
    GP.gate_preconditions(out, c1_activity_min=0.0)
    assert row_where(p, cell_id="gibbs__rep00__01c")["C1"] == V.PASS
    assert jload(out / "gates" / "preconditions.json")["params"]["C1_rule_default_function"] == "legacy_apf_max"
    assert jload(out / "gates" / "preconditions.json")["params"]["C1_rule_default_cli"] == "auto"
    # a declared fraction other than the default is recorded in the rule label (the record follows the value)
    GP.gate_preconditions(out, c1_rule="absolute", c1_abs_fraction=0.0005)
    assert row_where(p, cell_id="gibbs__rep00__01c")["C1_rule"] == "absolute_0.0005"
    assert row_where(p, cell_id="gibbs__rep00__01c")["C1_threshold_pages"] == "131"
    # `idle_floor` demanded with no idle cell: a refusal on every kernel row, never a silent fallback
    GP.gate_preconditions(out, c1_rule="idle_floor")
    rs = [r for r in rows(p) if r["role"] == "kernel"]
    assert rs and all(r["C1"] == GP.C1_FLOOR_REFUSAL and r["all_hard_pass"] == "false" for r in rs)
    assert jload(out / "gates" / "preconditions.json")["params"]["C1_floor_refused"] is True
    # an unknown rule is a ValueError, not a silent default
    import pytest
    with pytest.raises(ValueError):
        GP.gate_preconditions(out, c1_rule="something_else")


def test_c1_remap_idle_floor_when_idle_cells_exist():
    # SPEC_epoch2 6.3 item 2: three idle cells (K = 150 every row) -> auto takes the floor: K_max > idle p95 of K
    out = tmp_out()
    more = [SY.SynthSpec("fft", 7, K0=0, floor_F=100, floor_churn=0.0, content="double")]     # K_max = 100 <= edge
    more += [SY.SynthSpec("sleep", 100 + i, role="idle", rep=i, **SY.IDLE_PRESET) for i in range(3)]
    _c1_two_kernels(out, more=more)
    assert GP.main(["preconditions", "--out", str(out)]) == 0
    p = out / "gates" / "preconditions.csv"
    prm = jload(out / "gates" / "preconditions.json")["params"]
    assert prm["C1_rule_applied"] == "idle_floor" and prm["C1_rule_in_force"] == "idle_floor_p95"
    edge = prm["C1_idle_band_edge"]
    assert edge is not None and sorted(prm["C1_idle_cells_in_floor"]) == [f"idle__rep0{i}__01c" for i in range(3)]
    for cid in ("gemm__rep00__01c", "gibbs__rep00__01c", "fft__rep00__01c"):
        r = row_where(p, cell_id=cid)
        assert r["C1_rule"] == "idle_floor_p95" and abs(float(r["C1_threshold_pages"]) - edge) < 1e-9
    assert row_where(p, cell_id="gemm__rep00__01c")["C1"] == V.PASS          # about 300 > 150
    assert row_where(p, cell_id="fft__rep00__01c")["C1"] == V.FAIL           # 100 <= 150 (strict "exceeds")
    assert int(row_where(p, cell_id="fft__rep00__01c")["C1_K_max"]) == 100
    for i in range(3):
        idle = row_where(p, cell_id=f"idle__rep0{i}__01c")
        assert idle["C1"] == GP.C1_IDLE_STRING and idle["all_hard_pass"] == "true" and idle["C1_rule"] == ""
    # one number, two files: the same edge G-K0 writes as idle_band_edge on the same directory
    GP.write_gk0_template(out / "inputs" / "gk0_source.csv")
    GP.gate_gk0(out)
    gk = row_where(out / "gates" / "gk0.csv", kernel="gemm")
    assert abs(float(gk["idle_band_edge"]) - edge) < 1e-9
    assert abs(GP.idle_band_edge(out, [{"cell_id": f"idle__rep0{i}__01c"} for i in range(3)]) - edge) < 1e-9
    assert GP.idle_band_edge(out, []) is None
    # the idle cells' extract.csv are in the hashed inputs (SPEC_epoch2_review_al_farabi.md 5.7)
    assert any(k.endswith("idle__rep00__01c/extract.csv") for k in prm["inputs_sha256"])
    # --c1-rule absolute on the same directory applies 262 and still records the measured edge
    assert GP.main(["preconditions", "--out", str(out), "--c1-rule", "absolute"]) == 0
    prm = jload(out / "gates" / "preconditions.json")["params"]
    assert prm["C1_rule_applied"] == "absolute" and prm["C1_idle_band_edge"] is not None and abs(prm["C1_idle_band_edge"] - edge) < 1e-9
    assert row_where(p, cell_id="gemm__rep00__01c")["C1_threshold_pages"] == "262"
    assert row_where(p, cell_id="gibbs__rep00__01c")["C1"] == V.FAIL and row_where(p, cell_id="gemm__rep00__01c")["C1"] == V.PASS
    # a higher min-idle-cells than present: auto falls back to the absolute and says so
    GP.gate_preconditions(out, c1_rule="auto", c1_min_idle_cells=4)
    prm = jload(out / "gates" / "preconditions.json")["params"]
    assert prm["C1_rule_applied"] == "absolute" and prm["C1_min_idle_cells"] == 4
    # the percentile travels into the label and the threshold
    GP.gate_preconditions(out, c1_rule="idle_floor", c1_idle_percentile=50.0)
    assert row_where(p, cell_id="gemm__rep00__01c")["C1_rule"] == "idle_floor_p50"
    # an idle cell that fails its own C2 does not enter the floor
    out2 = tmp_out()
    _c1_two_kernels(out2, more=[SY.SynthSpec("sleep", 5, role="idle", n_pairs=4, **SY.IDLE_PRESET)])
    GP.gate_preconditions(out2, c1_rule="auto")
    prm2 = jload(out2 / "gates" / "preconditions.json")["params"]
    assert prm2["C1_idle_cells_in_floor"] == [] and prm2["C1_rule_applied"] == "absolute"
