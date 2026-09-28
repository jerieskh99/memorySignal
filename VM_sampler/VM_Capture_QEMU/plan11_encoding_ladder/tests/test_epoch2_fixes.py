"""Build epoch 2, builder B (the fix pass): the tests SPEC_epoch2.md Part 2 names per item, on
synthetic data only (the in-test generator `_synth_b2` where no trajectory is needed, `synth.py`
where one is). Items whose test the spec places in an existing gate test file are collected here
instead (a documented deviation, BUILD_epoch2_fixes.md): another builder appends to the same
files concurrently in this epoch, and a second appender on one file is a collision waiting to
happen.

Items covered: B1 (the C1 rule and its flags), B2 (the keyed alias inputs), B3 (the permutation
floor), B4 (the matched comparison's splits), B7 (G1 with no applicable kernel), B8 (Table 6 and
the wAPF table over admissible cells), B9 (the idle head drop), B10 (the G-L (ii) re-run step),
B11 (`--pass-frac`), B15/B16/B17 (the selection record), B18 (the splits CLI fallback), B19
(`wapf_norm`), B20 (the template guard), B24 (the idle rows of a feature file), B25/B26 (the
driver's forest and G-ORD flags), B30 (every runbook and SPEC section 7 command parses).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from _b2_common import S, V, SY, corpus, tmp_out, rows, row_where, jload, admissibility, pass_table_with, PKG
from plan11_encoding_ladder import gates_precondition as GP
from plan11_encoding_ladder import gates_calibration as GC
from plan11_encoding_ladder import gates_temporal as GT
from plan11_encoding_ladder import gates_comparison as GX
from plan11_encoding_ladder import gates_readings as GR
from plan11_encoding_ladder import models as M
from plan11_encoding_ladder import run_moves
from plan11_encoding_ladder import tables as T
from report_fixtures import make_out

N_EST = 10


def _ns(**kw) -> argparse.Namespace:
    """The driver Namespace of tests/test_driver.py::_ns (the epoch-1 keys only; every epoch-2 key is
    read with getattr in build_plan)."""
    base = dict(out="/x/out", root="/x/root", cells_csv=None, moves="0-13", assume_failed_zero=False, assume_reason=None,
                n_jobs=1, null_perm=500, null_splits="loko,loro,within_trace", seed_offset=0, force=False, dry_run=False,
                skip_missing_modules=False, only_modules=None, persist_side=None, failed_counts=None, table8_rung="combined",
                piano_cell=None, piano_stride=16, standalone_tex=None, cmd="plan")
    base.update(kw)
    return argparse.Namespace(**base)


def _set_sidecar(out: Path, cell_id: str, **kv) -> None:
    p = S.sidecar_path(out, cell_id)
    d = json.loads(p.read_text()); d.update(kv); p.write_text(json.dumps(d))


# ------------------------------------------------------------------------------------------ B1
def test_b1_c1_floor_reads_the_gk0_edge_and_the_interim_and_the_inherited_rule():
    """B1 (AA T1; CHECK_3 M1; CERT 6.5; E1 6.7): (a) with admissible idle cells the floor is the idle
    p95 of K, one number with gk0.csv's idle_band_edge; a kernel cell above it passes, one below
    fails; (b) without idle cells the interim page count (AA T1: 200) applies, K_max 199 fails and
    200 passes; (c) idle cells present but the floor switched off: the interim; (d) the function
    default is the inherited fraction rule (SPEC_epoch2_review_al_farabi.md 5.1: every existing
    direct call keeps its contract)."""
    # (a) the floor, with idle cells: gemm at K0 = 6000 passes, a 100-page floor-only kernel fails
    out = tmp_out()
    SY.write_corpus(out, [SY.SynthSpec("gemm", 42, K0=6000, content="double"),
                          SY.SynthSpec("gibbs", 42, K0=0, floor_F=100, floor_churn=0.0, content="spin")]
                    + [SY.SynthSpec("sleep", 1 + r, rep=r, role="idle", **SY.IDLE_PRESET) for r in range(3)])
    GP.gate_preconditions(out, c1_rule="auto")
    p = out / "gates" / "preconditions.csv"
    g, b = row_where(p, cell_id="gemm__rep00__01c"), row_where(p, cell_id="gibbs__rep00__01c")
    assert g["C1"] == V.PASS and b["C1"] == V.FAIL
    assert g["C1_rule"].startswith("idle_floor") and b["C1_rule"] == g["C1_rule"]
    assert int(b["C1_K_max"]) == 100
    idle = row_where(p, cell_id="idle__rep00__01c")
    assert idle["C1"] == GP.C1_IDLE_STRING and idle["all_hard_pass"] == "true"
    GP.gate_gk0(out)
    edge = float(row_where(out / "gates" / "gk0.csv", kernel="gemm")["idle_band_edge"])
    assert abs(float(g["C1_threshold_pages"]) - edge) < 1e-9            # one number, two files
    prm = jload(out / "gates" / "preconditions.json")["params"]
    assert prm["C1_rule_applied"] == "idle_floor" and abs(prm["C1_idle_band_edge"] - edge) < 1e-9
    assert set(prm["C1_idle_cells_in_floor"]) == {f"idle__rep0{r}__01c" for r in range(3)}
    assert prm["C1_rule_in_force"] == g["C1_rule"]
    # the idle extracts the edge was computed from are among the hashed inputs (al-Farabi review 5.7)
    assert any(k.endswith("idle__rep00__01c/extract.csv") for k in prm["inputs_sha256"])
    # (c) the floor switched off with idle cells present: the interim page count
    GP.gate_preconditions(out, c1_rule="absolute", c1_activity_min_pages=200)
    g = row_where(p, cell_id="gemm__rep00__01c")
    assert g["C1_rule"] == "absolute_200pages" and g["C1_threshold_pages"] == "200" and g["C1"] == V.PASS
    assert jload(out / "gates" / "preconditions.json")["params"]["C1_activity_min_pages"] == 200
    # (b) no idle cell: the interim; 199 fails, 200 passes
    out2 = tmp_out()
    SY.write_corpus(out2, [SY.SynthSpec("gemm", 42, K0=6000, content="double")])
    _set_sidecar(out2, "gemm__rep00__01c", K_max=199, apf_max=199 / 262144)
    GP.gate_preconditions(out2, c1_rule="auto", c1_activity_min_pages=200)
    r = row_where(out2 / "gates" / "preconditions.csv", cell_id="gemm__rep00__01c")
    assert r["C1"] == V.FAIL and r["C1_rule"].startswith("absolute") and r["C1_threshold_pages"] == "200"
    assert jload(out2 / "gates" / "preconditions.json")["params"]["C1_rule_applied"] == "absolute"
    _set_sidecar(out2, "gemm__rep00__01c", K_max=200, apf_max=200 / 262144)
    GP.gate_preconditions(out2, c1_rule="auto", c1_activity_min_pages=200)
    assert row_where(out2 / "gates" / "preconditions.csv", cell_id="gemm__rep00__01c")["C1"] == V.PASS
    # the AD bullet's 262 without the page count (the on-disk default of the absolute rule)
    GP.gate_preconditions(out2, c1_rule="auto")
    r = row_where(out2 / "gates" / "preconditions.csv", cell_id="gemm__rep00__01c")
    assert r["C1_threshold_pages"] == "262" and r["C1"] == V.FAIL
    # (d) the function default: the inherited fraction rule, unchanged for every direct call
    GP.gate_preconditions(out2)
    r = row_where(out2 / "gates" / "preconditions.csv", cell_id="gemm__rep00__01c")
    assert r["C1_rule"].startswith("legacy_apf_max") and r["C1"] == V.FAIL          # apf_max = 200 / N < 0.02
    GP.gate_preconditions(out2, c1_activity_min=0.0)
    assert row_where(out2 / "gates" / "preconditions.csv", cell_id="gemm__rep00__01c")["C1"] == V.PASS
    # the CLI carries the page count
    assert GP.main(["preconditions", "--out", str(out2), "--c1-activity-min-pages", "200"]) == 0
    assert jload(out2 / "gates" / "preconditions.json")["params"]["C1_activity_min_pages"] == 200


def test_b1_driver_flags_and_table4_c1_rule():
    """B1 (e): build_plan carries the C1 flags to `preconditions` when set and omits them when unset;
    (f) Table 4's Plan 02 row names the rule in force from preconditions.json."""
    P = run_moves.build_plan(_ns())
    pre = [c for c in P if c["name"] == "preconditions"][0]["args"]
    for flag in ("--c1-activity-min", "--c1-activity-min-pages", "--c1-rule"):
        assert flag not in pre
    P = run_moves.build_plan(_ns(c1_activity_min=0.0, c1_activity_min_pages=200, c1_rule="auto"))
    pre = [c for c in P if c["name"] == "preconditions"][0]["args"]
    assert pre[pre.index("--c1-activity-min-pages") + 1] == "200" and pre[pre.index("--c1-activity-min") + 1] == "0.0"
    assert pre[pre.index("--c1-rule") + 1] == "auto"
    rc, out, _ = _run_driver(["plan", "--out", "/x", "--root", "/r", "--moves", "2", "--c1-activity-min-pages", "200"])
    assert rc == 0 and "--c1-activity-min-pages 200" in out
    # (f) the table
    tmp = Path(tempfile.mkdtemp(prefix="plan11_e2_"))
    try:
        out = make_out(tmp / "out", n_pairs=40)
        pj = out / "gates" / "preconditions.json"
        d = json.loads(pj.read_text()); d["params"]["C1_rule_in_force"] = "idle_floor_p95"; pj.write_text(json.dumps(d))
        T.table4_status(out)
        p02 = [r for r in rows(out / "report" / "tables" / "table4_status.csv") if r["plan"] == "02"][0]["APF at 500 ms"]
        assert "C1 rule: idle_floor_p95" in p02 and p02.startswith("104 of 104 cells all_hard_pass")
        pj.unlink()
        T.table4_status(out)
        p02 = [r for r in rows(out / "report" / "tables" / "table4_status.csv") if r["plan"] == "02"][0]["APF at 500 ms"]
        assert "C1 rule: not run: gates/preconditions.json missing" in p02
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_driver(argv):
    import io
    from contextlib import redirect_stderr, redirect_stdout
    so, se = io.StringIO(), io.StringIO()
    with redirect_stdout(so), redirect_stderr(se):
        rc = run_moves.main(argv)
    return rc, so.getvalue(), se.getvalue()


# ------------------------------------------------------------------------------------------ B2
def test_b2_alias_steps_read_the_apf_rows_of_g3_flags():
    """B2 (CHECK_3 M2; CERT 1(c), 7.3): the two alias steps declare `csv:gates/g3_flags.csv:rung=apf`,
    whose digest moves when an apf row changes and not when another rung's rows are appended."""
    P = run_moves.build_plan(_ns())
    for name in ("alias", "alias (again, table6 features)"):
        c = [c for c in P if c["name"] == name][0]
        assert "csv:gates/g3_flags.csv:rung=apf" in c["inputs"] and "gates/g3_flags.csv" not in c["inputs"]
    out = tmp_out(); (out / "gates").mkdir(parents=True)
    p = out / "gates" / "g3_flags.csv"
    S.write_csv(p, ("rung", "kernel", "cell_id", "flag_cell"), [{"rung": "apf", "kernel": "gemm", "cell_id": "a", "flag_cell": V.G3_PRESENT}])
    h0 = run_moves._input_hash(out, "csv:gates/g3_flags.csv:rung=apf")
    S.write_csv(p, ("rung", "kernel", "cell_id", "flag_cell"), [{"rung": "apf", "kernel": "gemm", "cell_id": "a", "flag_cell": V.G3_PRESENT},
                                                              {"rung": "persist", "kernel": "gemm", "cell_id": "a", "flag_cell": V.G3_ABSENT}])
    assert run_moves._input_hash(out, "csv:gates/g3_flags.csv:rung=apf") == h0
    S.write_csv(p, ("rung", "kernel", "cell_id", "flag_cell"), [{"rung": "apf", "kernel": "gemm", "cell_id": "a", "flag_cell": V.G3_ABSENT}])
    assert run_moves._input_hash(out, "csv:gates/g3_flags.csv:rung=apf") != h0
    assert run_moves._input_hash(out, "csv:gates/nothing.csv:rung=apf") == "absent"


# ------------------------------------------------------------------------------------------ B3
def test_b3_perm_floor_on_gf_gx_and_the_clustering():
    """B3 (CHECK_3 M3; CERT 3, 6.7; E1 6.36): the CLIs refuse an under-powered null in B1-G1's own
    form; the direct calls keep the epoch-1 contract (floor 0) unless `perm_floor` is given."""
    # G-F part (i)
    out = corpus(reps=2, idle=6, kernels=["gemm", "lexer"], n_pairs=90, idle_overrides=lambda r: dict(floor_noise=0.05 * (r + 1)))
    admissibility(out)
    GP.gate_gf(out, rung="apf", n_perm=30, n_estimators=N_EST)
    p = out / "gates" / "gf.csv"
    assert row_where(p, part="i", kernel="idle")["verdict"] == V.GF_VOID              # the existing call form, unchanged
    GP.gate_gf(out, rung="apf", n_perm=30, n_estimators=N_EST, perm_floor=500)
    r = row_where(p, part="i", kernel="idle")
    assert r["verdict"] == V.not_run("30 permutations < 500") and r["score"] != "" and r["null_p95"] != ""
    assert GP.main(["gf", "--out", str(out), "--rung", "apf", "--n-perm", "5", "--n-estimators", str(N_EST)]) == 0
    assert row_where(p, part="i", kernel="idle")["verdict"] == V.not_run("5 permutations < 500")
    assert jload(out / "gates" / "gf.apf.W8_H4.params.json")["params"]["perm_floor"] == 500
    # G-X and the clustering, on a corpus with three campaigns and a selection
    out2 = corpus(reps=3, idle=0, n_pairs=60, campaign_of=lambda k, r: ["01c", "01c1", "dwarfs1"][r % 3])
    GT.gate_grid(out2, "apf", n_surrogates=2); GT.select(out2, "apf")
    GX.gate_gx(out2, "apf", n_perm=5, n_estimators=N_EST)
    assert row_where(out2 / "gates" / "gx.csv", rung="apf")["leak_verdict"] in (V.GX_LEAK, V.GX_POOLING_STANDS)   # floor 0 at the function
    assert GX.main(["gx", "--out", str(out2), "--rung", "apf", "--null-perm", "5", "--n-estimators", str(N_EST)]) == 0
    r = row_where(out2 / "gates" / "gx.csv", rung="apf")
    assert r["leak_verdict"] == V.not_run("5 permutations < 500") and r["score"] != "" and r["headline_mark"] == ""
    assert jload(out2 / "gates" / "gx.json")["params"]["perm_floor"] == 500
    M.run_clustering(out2, "apf", n_perm=5)
    assert row_where(out2 / "gates" / "clustering.csv", algo="kmeans")["exceeds_ari"] in ("true", "false")
    assert M.main(["cluster", "--out", str(out2), "--rung", "apf", "--null-perm", "5"]) == 0
    r = row_where(out2 / "gates" / "clustering.csv", algo="kmeans")
    assert r["exceeds_ari"] == V.not_run("5 permutations < 500") and r["exceeds_nmi"] == r["exceeds_ari"] and r["ari"] != ""
    assert jload(out2 / "gates" / "clustering.json")["params"]["perm_floor"] == 500


# ------------------------------------------------------------------------------------------ B4
def test_b4_matched_comparison_for_every_table7_split():
    """B4 (CHECK_3 M4; E1 6.48; Part 4 item 16): after gate_gdim every one of the five
    `gates/splits_matched/combined/<gid>/<split>__<ls>/scores.json` exists and Table 7's
    `combined (matched)` LORO row prints a number."""
    out = corpus(reps=2, idle=0, n_pairs=60)
    for rung in ("apf", "combined"):
        GT.gate_grid(out, rung, n_surrogates=2); GT.select(out, rung)
        gid = S.selected_grid_id(out, rung)[0]
        M.run_split_stage(out, rung, gid, "loko", "archetype", n_perm=0, run_null=False, n_estimators=N_EST, quarantine=False)
    GX.gate_gdim(out, n_perm=0, n_estimators=N_EST, null_splits="")
    cg = S.selected_grid_id(out, "combined")[0]
    for split, ls in (("loko", "archetype"), ("loro", "kernel"), ("within_trace", "kernel"), ("loro", "archetype"), ("within_trace", "archetype")):
        assert (out / "gates" / "splits_matched" / "combined" / cg / f"{split}__{ls}" / "scores.json").is_file(), (split, ls)
    matched = [r for r in rows(out / "gates" / "gdim.csv") if r["rung"] == "combined (matched)"]
    assert len(matched) == 5 and matched[0]["split"] == "loko" and {r["labelspace"] for r in matched} == {"archetype", "kernel"}
    T.table7(out)
    t7 = [r for r in rows(out / "report" / "tables" / "table7.csv") if r["rung"] == "combined (matched)" and r["split"] == "LORO" and r["label space"] == "kernel"][0]
    assert re.match(r"^[01](\.\d+)?$", t7["accuracy"]), t7["accuracy"]


# ------------------------------------------------------------------------------------------ B7, B15, B16, B17
def _grid_csvs(out: Path, rung: str, kernels: list[str], *, g1: str, missing: tuple = ()) -> None:
    for gid, W, H in S.grid_points_ids():
        if gid in missing:
            continue
        d = out / "gates" / "grid" / rung / gid
        d.mkdir(parents=True, exist_ok=True)
        rws = []
        for k in kernels:
            w = W if W is not None else 59
            rws.append({"rung": rung, "grid_id": gid, "W": w, "H": H if H is not None else 59, "hop_ratio": S.hop_ratio(W, H) if W else 1.0,
                        "kernel": k, "n_cells": 2, "n_windows_median": 10, "n_windows_min": 10, "n_windows_nonoverlap_median": 5,
                        "stat_pass_frac_median": 0.9, "g1_surrogate_p05": 0.5, "g1_trend_cells": 2, "G1": g1,
                        "coverage_0500": None, "coverage_0644": None, "coverage_pairs": None, "G2_0500": V.GP_UNDECLARED,
                        "G2_0644": V.GP_UNDECLARED, "G2_pairs": V.GP_UNDECLARED, "G2": V.GP_UNDECLARED,
                        "G4": V.PASS if (W is not None and GT.g4_pass(W, H)) else V.FAIL, "G5": V.PASS,
                        "gord_score_ordered": None, "gord_score_shuffled_mean": None, "gord_null_spread": None,
                        "GORD": V.GORD_ORDER_BLIND_BY_CONSTRUCTION if W is None else V.GORD_ORDER_BLIND})
        S.write_csv(d / "temporal_per_kernel.csv", GT.PER_KERNEL_COLUMNS, rws)


def test_b7_g1_with_no_applicable_kernel_drop_and_refuse():
    """B7 (CHECK_3 M7; CERT 3, 6.6, 7.4; Part 4 item 17): thirteen hand-written grid CSVs with every
    kernel TREND_PRESENT and G2 undeclared: under `drop` every integer-W point with G4 = pass reads
    `1 of 1` and the selection passes acceptance at W8_H4; under `refuse` the entry is refused."""
    out = tmp_out()
    _grid_csvs(out, "apf", ["gemm", "gibbs"], g1=V.TREND_PRESENT)
    GT.select(out, "apf")
    grid = {r["grid_id"]: r for r in rows(out / "gates" / "table5_grid.csv")}
    for gid, W, H in S.grid_points_ids():
        if W is not None and GT.g4_pass(W, H):
            assert grid[gid]["gates_passed"] == "1 of 1", gid
        assert grid[gid]["G1"].startswith("not applicable")
    assert "gf_part1" not in grid["W8_H4"]                                        # B15
    sel = jload(out / "gates" / "selection.json")["apf"]
    assert sel["grid_id"] == "W8_H4" and sel["passes_acceptance"] is True and sel["refusal"] == ""
    assert sel["params"]["g1_none_applicable"] == "drop"
    GT.select(out, "apf", g1_none_applicable="refuse")
    sel = jload(out / "gates" / "selection.json")["apf"]
    assert sel["refusal"] == V.not_run("no applicable kernel for G1") and sel["passes_acceptance"] is False
    assert sel["selected_by"] == "no applicable kernel" and sel["grid_id"] is not None
    assert "gates_passed" in rows(out / "gates" / "table5_grid.csv")[0]           # the grid rows keep their columns
    assert GT.main(["select", "--out", str(out), "--rung", "apf", "--g1-none-applicable", "refuse"]) == 2   # no cells.csv here: exit 2


def test_b16_b17_grid_incomplete_and_acceptance_failed_refusals():
    """B16 (CERT 1(d), 7.4): a missing grid CSV refuses the selection and marks selected_by; B17 (CERT
    3): a best-feasible selection names the gates that failed acceptance."""
    out = corpus(reps=2, idle=1, kernels=["gemm", "gibbs", "nbody"], n_pairs=120)
    pass_table_with(out, {"gemm": (100, "declared: synth"), "nbody": (6147, "declared: synth")})
    GT.gate_grid(out, "apf", n_surrogates=5)
    GT.select(out, "apf", kernel_refusals="blocks")                                # the `blocks` case: best-feasible
    sel = jload(out / "gates" / "selection.json")["apf"]
    assert sel["selected_by"] == "best-feasible" and sel["passes_acceptance"] is False
    assert sel["refusal"].startswith("acceptance failed: G2:"), sel["refusal"]
    assert V.is_refusal(sel["refusal"]) is False                                 # the fourth form, SPEC_epoch2_review_al_farabi.md 6.7
    (out / "gates" / "grid" / "apf" / "W16_H8" / "temporal_per_kernel.csv").unlink()
    GT.select(out, "apf")
    sel = jload(out / "gates" / "selection.json")["apf"]
    assert sel["refusal"] == V.not_run("grid incomplete (1 points missing)") and sel["passes_acceptance"] is False
    assert sel["selected_by"].endswith(" (grid incomplete)") and sel["grid_id"] is not None
    assert sel["params"]["n_grid_points_missing"] == 1
    assert jload(out / "gates" / "grid_complete.json")["apf"]["complete"] is False
    assert "gf_part1" not in rows(out / "gates" / "table5_grid.csv")[0]           # B15


# ------------------------------------------------------------------------------------------ B8
def test_b8_table6_and_wapf_table_over_admissible_cells():
    """B8 (CHECK_3 M8; E1 6.42): a kernel whose every cell is excluded prints the refusal in its score
    cells and leaves `n`; the fixture without the key is unchanged."""
    tmp = Path(tempfile.mkdtemp(prefix="plan11_e2_"))
    try:
        out = make_out(tmp / "out", n_pairs=40)
        pj = out / "gates" / "preconditions.json"
        d = json.loads(pj.read_text())
        floyd = [c["cell_id"] for c in csv.DictReader(open(out / "cells.csv")) if c["kernel"] == "floyd"]
        assert len(floyd) == 8
        d["excluded_cells"] = floyd; pj.write_text(json.dumps(d))
        T.table6(out)
        t6 = {r["row"]: r for r in rows(out / "report" / "tables" / "table6.csv")}
        assert t6["floyd"]["LOKO norm"] == T.EXCLUDED_KERNEL_TEXT and t6["floyd"]["within-trace raw"] == T.EXCLUDED_KERNEL_TEXT
        assert t6["floyd"]["n"] == "0" and t6["all"]["n"] == "88" and t6["gemm"]["n"] == "8"
        assert re.match(r"^[01](\.\d+)?$", t6["gemm"]["LOKO norm"])
        T.wapf_over_apf(out)
        w = {r["kernel"]: r for r in rows(out / "report" / "tables" / "table_wapf_over_apf.csv")}
        assert w["floyd"]["mean APF"] == T.EXCLUDED_KERNEL_TEXT and w["floyd"]["n cells"] == "0" and w["gemm"]["n cells"] == "8"
        # a partial exclusion counts the admissible cells
        d["excluded_cells"] = floyd[:3]; pj.write_text(json.dumps(d))
        T.table6(out); T.wapf_over_apf(out)
        t6 = {r["row"]: r for r in rows(out / "report" / "tables" / "table6.csv")}
        assert t6["floyd"]["n"] == "5" and t6["all"]["n"] == "93"
        assert {r["kernel"]: r for r in rows(out / "report" / "tables" / "table_wapf_over_apf.csv")}["floyd"]["n cells"] == "5"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------ B9, B24
def test_b9_b24_idle_head_drop_and_idle_rows_of_the_feature_file():
    """B9 (CHECK_3 M9; E1 6.10): `inputs/head_drop.csv` with `idle,5` drops five pairs of every idle
    cell whatever its kernel name; B24 (CERT 6.11; SPEC 3.1.5): the feature file's idle rows read
    `IDLE` / `idle` while cells.csv keeps `control` and the label-derived name."""
    assert S.head_drop_for({"gemm": 3, "idle": 5}, "gemm") == 3 and S.head_drop_for({"gemm": 3, "idle": 5}, "sleep", "idle") == 5
    assert S.head_drop_for({"idle": 5}, "gemm", "kernel") == 0 and S.head_drop_for(None, "x", "idle") == 0 and S.head_drop_for(2, "x", "idle") == 2
    out = corpus(reps=1, idle=2, kernels=["gemm"], n_pairs=60)
    S.write_csv(out / "inputs" / "head_drop.csv", S.HEAD_DROP_COLUMNS, [{"kernel": "gemm", "head_drop_pairs": 0, "reason": "t"},
                                                                       {"kernel": "idle", "head_drop_pairs": 5, "reason": "t"}])
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    feat = S.load_features(S.build_features(out, None, "apf", None, None, True, hd))
    idle = feat["role"] == "idle"
    assert set(feat["n_series_cell"][idle].tolist()) == {60 - 1 - 5} and set(feat["n_series_cell"][~idle].tolist()) == {59}
    assert set(feat["archetype"][idle].tolist()) == {"IDLE"} and set(feat["kernel"][idle].tolist()) == {"idle"}
    assert set(feat["archetype"][~idle].tolist()) == {"WORKING-SET"} and set(feat["kernel"][~idle].tolist()) == {"gemm"}
    cells = S.load_cells(out / "cells.csv")
    assert all(c["kernel"] == "idle" for c in cells if c["role"] == "idle")     # cells.csv itself is not touched by build_features
    admissibility(out)
    GP.gate_gf(out, rung="apf", n_perm=5, n_estimators=N_EST)
    prm = jload(out / "gates" / "gf.apf.W8_H4.params.json")["params"]
    assert prm["head_drop_idle"] == 5
    GT.gate_grid(out, "apf", n_surrogates=2)
    assert jload(out / "gates" / "grid" / "apf" / "W8_H4" / "temporal.params.json")["params"]["head_drop_idle"] == 5


# ------------------------------------------------------------------------------------------ B10
def test_b10_gl2_rerun_step():
    """B10 (CHECK_3 M10; CERT 6.9; E1 6.38; SPEC 3.7.4): with part (ii) at GL_SHOT_NOISE the driver's
    step runs the feature-drop re-run into splits_gl2drop (mocked here); with `pass` nothing runs;
    under `manual` the step records the author's."""
    out = tmp_out(); (out / "gates").mkdir(parents=True)
    def gl_csv(part2):
        S.write_csv(out / "gates" / "gl.csv", GX.GL_COLUMNS, [{"rung": "apf", "part": "i", "grid_id": "W8_H4", "verdict": V.PASS},
                                                             {"rung": "apf", "part": "ii", "grid_id": "W8_H4", "r2": 0.7, "verdict": part2}])
    gl_csv(V.GL_SHOT_NOISE)
    o = _ns(out=str(out), null_perm=7, null_splits="loko", n_estimators=10, n_jobs=2, gl2_rerun="auto")
    rec = {}
    with mock.patch.object(run_moves.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="", stderr="")) as sp:
        status, detail = run_moves.gl2_rerun(out, o, rec)
    assert status == "done" and sp.call_count == 1
    argv = rec["argv_nested"]
    assert argv[argv.index("--feature-drop") + 1] == "cov,std,peak2med" and argv[argv.index("--base-dir") + 1] == "splits_gl2drop"
    assert "--raw-and-norm" in argv and argv[argv.index("--null-perm") + 1] == "7" and argv[argv.index("--n-estimators") + 1] == "10"
    j = jload(out / "gates" / "gl2_rerun.json")
    assert j["status"] == "done" and j["params"]["base_dir"] == "splits_gl2drop" and j["params"]["feature_drop"] == ["cov", "std", "peak2med"]
    with mock.patch.object(run_moves.subprocess, "run") as sp:
        status, _ = run_moves.gl2_rerun(out, _ns(out=str(out), gl2_rerun="manual"))
    assert status == "not run: manual (--gl2-rerun manual)" and sp.call_count == 0
    gl_csv(V.PASS)
    with mock.patch.object(run_moves.subprocess, "run") as sp:
        status, _ = run_moves.gl2_rerun(out, o)
    assert status == "not run: G-L (ii) passed" and sp.call_count == 0
    # the step sits right after `gl` at move 7 and the driver accepts the flag
    P = run_moves.build_plan(_ns())
    m7 = [c["name"] for c in P if c["move"] == 7]
    assert m7.index("gl2 rerun (feature drop after G-L (ii))") == m7.index("gl") + 1
    assert [c for c in P if c["sub"] == "gl2-rerun"][0]["internal"] is True
    rc, out_txt, _ = _run_driver(["plan", "--out", "/x", "--root", "/r", "--moves", "7", "--gl2-rerun", "manual"])
    assert rc == 0 and "gl2-rerun" in out_txt


# ------------------------------------------------------------------------------------------ B11
def test_b11_pass_frac_flag_on_gdec():
    """B11 (CHECK_3 M11; E1 6.46): the gdec CLI accepts --pass-frac and records it."""
    out = corpus(reps=2, idle=0, kernels=["floyd", "gibbs"], n_pairs=60)
    GC.gate_gp(out)
    assert GR.main(["gdec", "--out", str(out), "--pass-frac", "0.7", "--n-surrogates", "5"]) == 0
    assert jload(out / "gates" / "gdec.params.json")["params"]["pass_frac"] == 0.7


# ------------------------------------------------------------------------------------------ B18
def test_b18_splits_cli_uses_the_selection_and_refuses_without_one():
    """B18 (CERT 3 third marker, 6.13, 7.5; E1 6.56): without --grid-id the splits CLI runs at the
    selected point (`params.grid_source = "selection.json"`) and exits 2 when there is none."""
    out = corpus(reps=1, idle=0, kernels=["gemm", "gibbs"], n_pairs=60)
    S.build_features(out, None, "apf", 8, 4, True)
    import io
    from contextlib import redirect_stderr
    se = io.StringIO()
    with redirect_stderr(se):
        rc = M.main(["splits", "--out", str(out), "--rung", "apf", "--split", "loro", "--labelspace", "kernel", "--null-perm", "0",
                     "--null-splits", "none", "--n-estimators", str(N_EST)])
    assert rc == 2 and "missing input: gates/selection.json has no entry for apf (run gates_temporal select first)" in se.getvalue()
    GT.gate_grid(out, "apf", n_surrogates=2); GT.select(out, "apf")
    gid = S.selected_grid_id(out, "apf")[0]
    assert M.main(["splits", "--out", str(out), "--rung", "apf", "--split", "loro", "--labelspace", "kernel", "--null-perm", "0",
                   "--null-splits", "none", "--n-estimators", str(N_EST)]) == 0
    sc = jload(M.split_dir(out, "apf", gid, "loro", "kernel") / "scores.json")
    assert sc["params"]["grid_source"] == "selection.json" and sc["params"]["grid_id"] == gid
    assert M.main(["splits", "--out", str(out), "--rung", "apf", "--grid-id", "W8_H4", "--split", "loro", "--labelspace", "kernel",
                   "--null-perm", "0", "--null-splits", "none", "--n-estimators", str(N_EST), "--base-dir", "splits_gl2drop"]) == 0
    sc = jload(M.split_dir(out, "apf", "W8_H4", "loro", "kernel", "splits_gl2drop") / "scores.json")
    assert sc["params"]["grid_source"] == "argument"                               # B10's --base-dir lands the run there


# ------------------------------------------------------------------------------------------ B19
def test_b19_wapf_norm_on_grid_and_the_driver():
    """B19 (CERT 5, 6.13, 7.6; E1 6.31): `gate_grid(..., wapf_norm="median_self")` leaves the wAPF
    feature files with that scalar and records it; the default path is unchanged; the driver passes
    the flag to `features` and `grid` when it departs from median_K."""
    out = corpus(reps=1, idle=0, kernels=["gemm", "gibbs"], n_pairs=60)
    GT.gate_grid(out, "wapf", n_surrogates=2, wapf_norm="median_self")
    f = S.load_features(out / "features" / "wapf" / "W8_H4_norm.npz")
    assert str(f["wapf_norm"]) == "median_self"
    assert jload(out / "gates" / "grid" / "wapf" / "W8_H4" / "temporal.params.json")["params"]["wapf_norm"] == "median_self"
    GT.gate_grid(out, "wapf", n_surrogates=2)
    assert str(S.load_features(out / "features" / "wapf" / "W8_H4_norm.npz")["wapf_norm"]) == "median_K"
    P = run_moves.build_plan(_ns(wapf_norm="median_self"))
    for name in ("features wapf all grid", "grid wapf", "features apf all grid", "grid apf"):
        a = [c for c in P if c["name"] == name][0]["args"]
        assert a[a.index("--wapf-norm") + 1] == "median_self", name
    P = run_moves.build_plan(_ns())
    assert "--wapf-norm" not in [c for c in P if c["name"] == "grid wapf"][0]["args"]


# ------------------------------------------------------------------------------------------ B20
def test_b20_template_clis_keep_an_existing_author_input():
    """B20 (CERT 6.12, 7.7; E1 6.13): each template CLI refuses to overwrite unless --force."""
    out = tmp_out()
    cmds = [(GC.main, ["pass-table", "--out", str(out)], out / "inputs" / "pass_table.csv"),
            (GP.main, ["gk0-template", "--out", str(out)], out / "inputs" / "gk0_source.csv"),
            (GP.main, ["idle-admissibility-template", "--out", str(out)], out / "inputs" / "idle_admissibility.json"),
            (S.main, ["head-drop-template", "--out", str(out)], out / "inputs" / "head_drop.csv")]
    import io
    from contextlib import redirect_stdout
    for fn, argv, path in cmds:
        assert fn(argv) == 0 and path.is_file()
        path.write_text(path.read_text() + "\n# edited by the author\n")
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        so = io.StringIO()
        with redirect_stdout(so):
            assert fn(argv) == 0
        assert so.getvalue().strip() == f"kept: author input exists: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == h
        assert fn(argv + ["--force"]) == 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() != h


# ------------------------------------------------------------------------------------------ B25, B26
def test_b25_b26_driver_forest_and_gord_flags():
    """B25 (E1 6.61): --n-estimators reaches gord, splits, gf, gx, gdim, gm and nothing else; B26:
    the G-ORD cost flags reach gord as --n-order-perm / --null-perm."""
    P = run_moves.build_plan(_ns(n_estimators=10, gord_n_order_perm=2, gord_null_perm=5))
    # builder A's comparator commands carry --n-estimators on their own (SPEC_epoch2 Part 1.7); they are not this rule's
    with_flag = {c["name"] for c in P if "--n-estimators" in c["args"] and c["module"] != "comparators"}
    expect = {c["name"] for c in P if (c["module"], c["sub"]) in run_moves.NEST_COMMANDS}
    assert with_flag == expect and {"gord apf", "splits apf", "gx apf", "gdim", "gm"} <= with_flag
    assert "gf all rungs at " + run_moves.GF_DEFAULT_GRID in with_flag
    for c in P:
        if "--n-estimators" in c["args"] and c["module"] != "comparators":
            assert c["args"][c["args"].index("--n-estimators") + 1] == "10"
    g = [c for c in P if c["name"] == "gord apf"][0]["args"]
    assert g[g.index("--n-order-perm") + 1] == "2" and g[g.index("--null-perm") + 1] == "5"
    P = run_moves.build_plan(_ns())
    assert not any("--n-estimators" in c["args"] for c in P if c["module"] != "comparators")
    assert "--n-order-perm" not in [c for c in P if c["name"] == "gord apf"][0]["args"]
    # --duration-s reaches extract all when it departs from 600 (B12)
    P = run_moves.build_plan(_ns(duration_s=38.64))
    a = [c for c in P if c["name"] == "extract all"][0]["args"]
    assert a[a.index("--duration-s") + 1] == "38.64"
    assert "--duration-s" not in [c for c in run_moves.build_plan(_ns()) if c["name"] == "extract all"][0]["args"]


# ------------------------------------------------------------------------------------------ B30
class _Parsed(BaseException):
    """Raised from the patched parse_args so that a module's main() stops before running anything
    (BaseException: some mains catch Exception)."""
    def __init__(self, extras):
        self.extras = extras


_PLACEHOLDERS = {"<out>": "/tmp/p11x/out", "<root>": "/tmp/p11x/root", "<retention root>": "/tmp/p11x/root", "<tmp>": "/tmp/p11x",
                 "<csv>": "/tmp/p11x/failed.csv", "<cell_id>": "gemm__rep00__01c", "$r": "apf", "<rung>": "apf", "<path>": "/tmp/p11x/x.tex",
                 "<NAME,...>": "table6", "<n>": "5", "<value>": "0.04", "<ID>": "gemm__rep00__01c", "<regex>": "gemm"}


def _parse_command(line: str) -> tuple[str, list[str]]:
    """`python3 -m plan11_encoding_ladder.<module> <args>` or `<module>.py <args>` -> (module, argv), placeholders
    substituted; returns ("", []) for a line that is not a toolkit command."""
    txt = line.strip()
    for k, v in _PLACEHOLDERS.items():
        txt = txt.replace(k, v)
    txt = re.sub(r"<[^>]*>", "x", txt)
    txt = txt.replace("[", "").replace("]", "")                 # a bracketed optional flag is checked as a given one
    try:
        toks = shlex.split(txt.split(" #", 1)[0]) if txt else []
    except ValueError:                                            # prose inside a fence (the "where things are" listing)
        return "", []
    if len(toks) >= 3 and toks[0] == "python3" and toks[1] == "-m" and toks[2].startswith("plan11_encoding_ladder."):
        return toks[2].split(".", 1)[1], toks[3:]
    if toks and toks[0].endswith(".py") and (PKG / toks[0]).is_file():
        return toks[0][:-3], toks[1:]
    return "", []


def _check_command(module: str, argv: list[str]) -> list[str]:
    """The flags argparse does not know, for one command line (parse_known_args on the module's own
    parser, reached through main()); [] when every flag is known."""
    mod = importlib.import_module(f"plan11_encoding_ladder.{module}")

    def fake_parse_args(self, args=None, namespace=None):
        ns, extras = self.parse_known_args(args, namespace)
        raise _Parsed(extras)
    with mock.patch.object(argparse.ArgumentParser, "parse_args", fake_parse_args):
        try:
            mod.main(argv)
        except _Parsed as e:
            return list(e.extras)
        except SystemExit as e:                                   # a required flag missing or an invalid choice
            return [f"argparse error (exit {e.code})"]
    return ["main() returned without parsing"]


def _runbook_commands() -> list[tuple[str, str]]:
    lines, fenced, buf = [], False, ""
    for raw in (PKG / "RUNBOOK.md").read_text().splitlines():
        if raw.strip().startswith("```"):
            fenced = not fenced; buf = ""; continue
        if not fenced:
            continue
        s = raw.rstrip()
        if s.endswith("\\"):
            buf += s[:-1] + " "; continue
        line = (buf + s).strip(); buf = ""
        if line.startswith("for ") or line == "done":
            continue
        lines.append(line)
    return [(l, "RUNBOOK.md") for l in lines]


def _spec_section7_commands() -> list[tuple[str, str]]:
    text = (PKG / "SPEC.md").read_text()
    sec = text[text.index("## 7. The driver and the runbook"):text.index("### 7.1 The CLI contract")]
    cmds = []
    for row in sec.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", row):
            continue
        cell = row.split("|")[2]
        for m in re.finditer(r"`([^`]*)`", cell):
            c = m.group(1).strip()
            if re.match(r"^\w+\.py\s", c):
                cmds.append((c, "SPEC.md section 7"))
    return cmds


def test_b30_runbook_and_spec_commands_parse():
    """B30 (with B23, B13, B14, B27): every fenced toolkit command of RUNBOOK.md and every command cell
    of SPEC.md section 7's move table parses against the module's own argparse parser with no unknown
    flag (a shorthand like `grid/g3/gord/select` is expanded; `driver.py` is run_moves' alias)."""
    checked, bad = 0, []
    for line, where in _runbook_commands() + _spec_section7_commands():
        module, argv = _parse_command(line)
        if not module:
            continue
        if module == "driver":
            module = "run_moves"
        variants = [argv]
        if argv and "/" in argv[0] and not argv[0].startswith("-"):
            variants = [[sub] + argv[1:] for sub in argv[0].split("/")]
        for a in variants:
            if "--out" not in a and module != "synth":             # SPEC 7's shorthand rows omit --out <out>; every command but synth takes it
                a = a + ["--out", "/tmp/p11x/out"]
            extras = _check_command(module, a)
            checked += 1
            if extras:
                bad.append((where, line, extras))
    assert checked >= 40, checked
    assert not bad, "\n".join(f"{w}: {l} -> {e}" for w, l, e in bad)


# ------------------------------------------------------------------------------------------ B22
def test_b22_unittest_runner_stops_on_the_guard():
    """B22 (E1 4 "Documentation known to be stale"): `python3 -m unittest tests.test_runner_guard` exits
    non-zero with the instruction to run pytest; under pytest the guard passes (it is in this suite)."""
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    proc = subprocess.run([sys.executable, "-m", "unittest", "tests.test_runner_guard"], cwd=str(PKG), capture_output=True, text=True, env=env)
    assert proc.returncode != 0
    assert "the gate tests are pytest functions that unittest does not discover; run: python3 -m pytest -q tests" in proc.stdout + proc.stderr
    assert os.environ.get("PYTEST_CURRENT_TEST")
