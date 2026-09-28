"""gates_calibration.py: the pass table, G-P, G-C, the alias falsifier (SPEC 3.4, 5.3;
SPEC_review_al_kindi.md items 1, 2, 5)."""
import numpy as np

from _b2_common import S, V, SY, corpus, tmp_out, rows, row_where, jload, pass_table_with
from plan11_encoding_ladder import gates_calibration as GC

TRIPLE = ("gemm", "gibbs", "histogram")


def test_pass_table_template_and_loading():
    out = tmp_out()
    GC.write_pass_table_template(out / "inputs" / "pass_table.csv")
    pt = GC.load_pass_table(out / "inputs" / "pass_table.csv")
    assert pt["nbody"].passes == 6147 and pt["nbody"].source_kind == "declared"
    assert all(pt[k].source_kind == "undeclared" for k in pt if k != "nbody")
    assert GC.load_pass_table(None)["gemm"].source_kind == "undeclared"


def test_gp_cell_verdicts():
    e = lambda p, k="declared": GC.PassEntry(p, k)
    assert GC.gp_cell(120, e(10))["verdict_pairs"] == V.GP_RESOLVABLE
    assert GC.gp_cell(120, e(40))["verdict_pairs"] == V.GP_MARGINAL
    assert GC.gp_cell(120, e(100))["verdict_pairs"] == V.GP_ALIASED
    assert GC.gp_cell(120, e(None, "undeclared"))["verdict_pairs"] == V.GP_UNDECLARED
    r = GC.gp_cell(120, e(3))
    assert r["rhythm_verdict"] == V.GP_RHYTHM_UNDERSAMPLED and r["within_pass_verdict"] == V.GP_ADMITTED
    assert GC.gp_cell(120, e(100))["within_pass_verdict"] == V.GP_PASS_ALIASED
    assert GC.gp_cell(120, e(10, "inferred"))["verdict_pairs"] == V.GP_RESOLVABLE + " (INFERRED)"
    n = GC.gp_cell(931, e(6147))
    assert n["verdict_pairs"] == V.GP_ALIASED and abs(n["T_pairs"] - 0.1515) < 1e-3 and n["verdict_dt_0500"] == V.GP_ALIASED
    assert GC.gp_verdict_T(2.5, 0.5) == V.GP_RESOLVABLE and GC.gp_verdict_T(1.2, 0.5) == V.GP_MARGINAL and GC.gp_verdict_T(0.9, 0.5) == V.GP_ALIASED


def test_gp_file_and_kernel_rollup():
    out = corpus(reps=2, idle=0, kernels=["gemm", "gibbs"])
    pass_table_with(out, {"gemm": (10, "declared: synth")})
    GC.gate_gp(out)
    p = out / "gates" / "gp.csv"
    assert row_where(p, kernel="gemm", cell_id="all")["verdict_pairs"] == V.GP_RESOLVABLE
    assert row_where(p, kernel="gibbs", cell_id="all")["verdict_pairs"] == V.GP_UNDECLARED
    assert GC.gp_kernel_verdicts(out)["gemm"]["rhythm_verdict"] == V.GP_ADMITTED


def test_gc_apf_persist_wapf_pass_on_the_corpus_and_stat_a_below_two():
    out = corpus(reps=3, idle=0, kernels=TRIPLE)
    for rung in ("apf", "persist", "wapf", "content", "combined"):
        GC.gate_gc(out, rung=rung)
    p = out / "gates" / "gc.csv"
    for rung in ("apf", "persist", "wapf", "content", "combined"):
        assert row_where(p, rung=rung, rep="all")["verdict"] == V.PASS, rung
    r0 = row_where(p, rung="apf", rep="0")
    assert 1.5 <= float(r0["stat_a"]) < 2.0 and int(r0["n_events"]) >= 1           # al-Kindi item 1: (2K + F)/(K + F) < 2
    assert float(row_where(p, rung="persist", rep="0")["j_at_event"]) <= GC.GC_J_DIP_MAX
    params = jload(out / "gates" / "gc.apf.params.json")["params"]
    assert params["jump_predicted"] == 2.0 and params["jump_detect_ratio"] == 1.5


def test_gc_refuses_disconnected_lead_and_names_the_aliased_regime():
    # no pulse, gemm below the full-footprint band -> disconnected lead
    out = corpus(reps=3, idle=0, kernels=TRIPLE, break_pulse=True, overrides={"gemm": dict(K0=1024)})
    GC.gate_gc(out, rung="apf"); GC.gate_gc(out, rung="persist")
    p = out / "gates" / "gc.csv"
    assert row_where(p, rung="apf", rep="all")["verdict"] == V.GC_DISCONNECTED
    assert row_where(p, rung="persist", rep="all")["verdict"] == V.GC_DISCONNECTED
    GC.gate_gc(out, rung="wapf"); GC.gate_gc(out, rung="content"); GC.gate_gc(out, rung="combined")
    assert row_where(p, rung="combined", rep="all")["verdict"] == V.GC_DISCONNECTED
    assert S.gc_verdict(out, "apf") == V.GC_DISCONNECTED
    # no pulse, gemm at its full footprint with J near one -> the second cause (al-Kindi item 2), neither passed nor voided
    out2 = corpus(reps=3, idle=0, kernels=TRIPLE, break_pulse=True)
    GC.gate_gc(out2, rung="apf")
    assert row_where(out2 / "gates" / "gc.csv", rung="apf", rep="all")["verdict"] == V.GC_ALIASED_BY_DESIGN
    # mixed reps (one rep pulses, two do not) -> disconnected lead
    specs = SY.corpus_specs(reps=3, idle=0, kernels=TRIPLE)
    for s in specs:
        if s.name == "gemm" and s.rep > 0:
            s.pulse_period = None; s.pulse_extra = 0
    out3 = tmp_out(); SY.write_corpus(out3, specs)
    GC.gate_gc(out3, rung="apf")
    assert row_where(out3 / "gates" / "gc.csv", rung="apf", rep="all")["verdict"] == V.GC_DISCONNECTED


def test_gc_content_ordering_pass_and_break_order():
    out = corpus(reps=3, idle=0, kernels=TRIPLE)
    GC.gate_gc(out, rung="content")
    assert row_where(out / "gates" / "gc.csv", rung="content", rep="all")["verdict"] == V.PASS
    GC.gate_gc(out, rung="content", pairing="envelope")
    assert row_where(out / "gates" / "gc.csv", rung="content", rep="all")["verdict"] == V.PASS
    out2 = corpus(reps=3, idle=0, kernels=TRIPLE, break_order=True)
    GC.gate_gc(out2, rung="content")
    assert row_where(out2 / "gates" / "gc.csv", rung="content", rep="all")["verdict"] == V.GC_DISCONNECTED
    GC.gate_gc(out2, rung="content", content_page_set="all")
    assert row_where(out2 / "gates" / "gc.csv", rung="content", rep="all")["verdict"] == V.GC_DISCONNECTED


def test_alias_falsifier_moves_and_stays():
    dt = {f"c{i}": 0.635 + 0.005 * i for i in range(8)}
    moves = {c: 3.0 * v + 0.01 for c, v in dt.items()}
    stays = {c: 0.5 + 1e-3 * ((i * 7) % 5) for i, c in enumerate(dt)}
    assert GC.alias_falsifier(moves, dt)["verdict"] == V.ALIAS_MOVES
    r = GC.alias_falsifier(stays, dt)
    assert r["verdict"] == V.ALIAS_STAYS and r["n"] == 8 and abs(r["dt_min"] - 0.635) < 1e-9
    assert GC.alias_falsifier({"a": 1.0}, {"a": 0.6})["verdict"].startswith("not run")


def test_run_alias_writes_g3_and_table6_rows():
    out = corpus(reps=3, idle=0, kernels=["floyd", "histogram", "nbody"], n_pairs=90)
    S.write_csv(out / "gates" / "g3_flags.csv", ("rung", "kernel", "cell_id", "ceps_peak_freq_cyc_per_pair"),
                [{"rung": "apf", "kernel": "floyd", "cell_id": f"floyd__rep{r:02d}__01c", "ceps_peak_freq_cyc_per_pair": 0.1 + 0.001 * r} for r in range(3)])
    from plan11_encoding_ladder import gates_temporal as GT
    GT.gate_grid(out, "apf", n_surrogates=2); GT.select(out, "apf")
    GC.run_alias(out)
    rs = rows(out / "gates" / "alias.csv")
    assert any(r["kind"] == "g3_peak" and r["kernel"] == "floyd" for r in rs)
    assert all(r["verdict"] for r in rs)
