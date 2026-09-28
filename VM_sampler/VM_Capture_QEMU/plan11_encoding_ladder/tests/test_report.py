#!/usr/bin/env python3
"""Builder 3's tests for tables.py, figures.py and latex_skeleton.py (SPEC section 1: run on
the synthetic corpus end to end; here the synthetic `<out>` tree of report_fixtures.py, since
the other builders' modules were written in parallel). Runs under `python3 -m pytest -q` and
`python3 -m unittest`."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

from plan11_encoding_ladder import figures, latex_skeleton, tables  # noqa: E402
from plan11_encoding_ladder._report_common import (  # noqa: E402
    GC_DISCONNECTED, GF_VOID, GRID_IDS, KERNEL_NAMES, NEAR_UNFALSIFIABLE, RUNGS, refused,
)
from report_fixtures import make_out  # noqa: E402


def _rows(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plan11_report_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def out(self, name="out", **kw) -> Path:
        return make_out(self.tmp / name, n_pairs=40, **kw)


class TestTable5(_Base):
    def test_shape_selection_and_review_columns(self):
        out = self.out(best_feasible_rung="wapf")
        tables.table5(out)
        rows = _rows(out / "report" / "tables" / "table5.csv")
        self.assertEqual(len(rows), 5 * 13)  # every grid point kept (P2 Sec. V; SPEC 6.1)
        self.assertIn("G2 (pairs)", rows[0])  # al-Kindi review 7
        sel = [r for r in rows if r["selected"]]
        self.assertEqual(len(sel), 5)
        by_enc = {r["encoding"]: r for r in sel}
        self.assertEqual(by_enc["APF"]["selected"], "selected")
        self.assertEqual(by_enc["wAPF"]["selected"], "selected: best-feasible")  # al-Farabi 2.11(c)
        self.assertIn("G-F (i): inseparable at floor", by_enc["APF"]["refusal"])  # al-Farabi 2.6
        self.assertEqual(by_enc["APF"]["grid_point (W x H)"], "16 x 8")
        self.assertTrue(any(r["grid_point (W x H)"] == "whole cell" for r in rows))
        self.assertTrue((out / "report" / "tables" / "table5.tex").exists())
        self.assertTrue((out / "report" / "tables" / "table5.md").exists())

    def test_gc_disconnected_carried_on_every_row_of_the_rung(self):
        out = self.out(gc_disconnected_rung="persist")
        tables.table5(out)
        rows = _rows(out / "report" / "tables" / "table5.csv")
        pers = [r for r in rows if r["encoding"] == "persistence"]
        self.assertEqual(len(pers), 13)
        self.assertTrue(all(f"G-C: {GC_DISCONNECTED}" in r["refusal"] for r in pers))  # al-Farabi 2.5
        apf = [r for r in rows if r["encoding"] == "APF"]
        self.assertTrue(all("G-C" not in r["refusal"] for r in apf))

    def test_missing_grid_file_writes_not_run_naming_it(self):
        out = self.out()
        (out / "gates" / "table5_grid.csv").unlink()
        tables.table5(out)
        rows = _rows(out / "report" / "tables" / "table5.csv")
        self.assertEqual(len(rows), 65)
        self.assertTrue(all(r["G1"].startswith("not run: gates/table5_grid.csv") for r in rows))

    def test_table5_g3_companion(self):
        out = self.out()
        tables.table5_g3(out)
        rows = _rows(out / "report" / "tables" / "table5_g3.csv")
        gemm = [r for r in rows if r["kernel"] == "gemm" and r["encoding"] == "APF"][0]
        self.assertEqual(gemm["flag"], "rhythm flag: present")
        self.assertEqual(gemm["cells present"], "8 of 8")
        self.assertEqual(gemm["cepstral SNR (dB)"], "9")


class TestTable6(_Base):
    def test_rows_scores_and_markers(self):
        out = self.out()
        tables.table6(out)
        rows = _rows(out / "report" / "tables" / "table6.csv")
        names = [r["row"] for r in rows]
        for k in KERNEL_NAMES:
            self.assertIn(k, names)
        self.assertEqual(names[-1], "all")
        self.assertIn("WORKING-SET", names)
        self.assertIn("IDLE", names)  # the lexer relabelled by G-K0 in the fixture
        byrow = {r["row"]: r for r in rows}
        self.assertEqual(byrow["floyd"]["level set"], "A")
        self.assertEqual(byrow["gemm"]["level set"], "B")
        self.assertEqual(byrow["gibbs"]["level set"], "")
        self.assertEqual(byrow["all"]["n"], "96")
        self.assertRegex(byrow["gemm"]["LOKO norm"], r"^[01](\.\d+)?$")
        self.assertRegex(byrow["gemm"]["LOKO raw"], r"^[01](\.\d+)?$")
        self.assertEqual(byrow["gemm"]["rank (LOKO norm)"], "rank 500 of 500")
        self.assertEqual(byrow["all"]["G-X"], "pooling stands; confound: partial")
        self.assertEqual(byrow["gemm"]["G-X"], "")
        self.assertEqual(byrow["WORKING-SET"]["G-N status"], "headline")
        self.assertEqual(byrow["IDLE"]["LOKO norm"], "--")  # undefined number prints --

    def test_near_unfalsifiable_prints_in_every_score_cell_of_that_split(self):
        out = self.out(unfalsifiable_split=("apf", "loko", "archetype"))
        tables.table6(out)
        rows = _rows(out / "report" / "tables" / "table6.csv")
        self.assertTrue(all(r["LOKO norm"] == NEAR_UNFALSIFIABLE for r in rows))  # al-Farabi 2.7
        self.assertTrue(all(r["LORO norm"] != NEAR_UNFALSIFIABLE for r in rows))
        txt = (out / "report" / "tables" / "table6.tex").read_text()
        self.assertIn("near\\_unfalsifiable", txt)

    def test_smoke_run_marks_null_and_rank_cells(self):
        out = self.out(smoke_perm=20)
        tables.table6(out)
        rows = _rows(out / "report" / "tables" / "table6.csv")
        self.assertTrue(all(r["rank (LOKO norm)"] == "not run: 20 permutations < 500" for r in rows))  # SPEC 3.7.1
        self.assertTrue(all(r["null p95 (LOKO norm)"] == "not run: 20 permutations < 500" for r in rows))
        self.assertRegex(rows[0]["LOKO norm"], r"^[01](\.\d+)?$")  # the score row is printed

    def test_no_selection_and_total_leak(self):
        out = self.out(no_selection_rung="apf")
        tables.table6(out)
        rows = _rows(out / "report" / "tables" / "table6.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["LOKO norm"], "not run: no selection for apf")  # al-Farabi 2.9(b)
        out2 = self.out("out2", gx_total_leak=True)
        tables.table6(out2)
        rows = _rows(out2 / "report" / "tables" / "table6.csv")
        self.assertIn(refused("campaign leak with total confound"), rows[-1]["G-X"])


class TestTable7(_Base):
    def test_rows_columns_and_gm_text(self):
        out = self.out()
        tables.table7(out)
        rows = _rows(out / "report" / "tables" / "table7.csv")
        rungs = [r["rung"] for r in rows]
        self.assertEqual(len(rows), 6 * 3 + 6 * 2)  # primary rows + appended archetype rows
        self.assertIn("combined (matched)", rungs)
        for col in ("G-C", "G-F (i)", "G-L", "G-DIM", "G-M vs APF", "G-X", "feature count"):
            self.assertIn(col, rows[0])
        loko = {r["rung"]: r for r in rows if r["split"] == "LOKO"}
        self.assertEqual(loko["apf"]["G-M vs APF"], "--")
        self.assertEqual(loko["content"]["G-M vs APF"], "beats (diff 0.200 > spread 0.040; 8 up, 0 down)")
        self.assertEqual(loko["wapf"]["G-L"], "(i) level only")
        self.assertEqual(loko["apf"]["G-L"], "(i) pass; (ii) pass")
        self.assertEqual(loko["combined (matched)"]["G-DIM"], "declared reduction (d = 60, matched 36, train_importance)")
        self.assertEqual(loko["apf"]["resolution (W x H)"], "16 x 8")
        self.assertEqual(loko["apf"]["feature count"], "8")
        self.assertEqual(loko["combined"]["feature count"], "60")
        self.assertRegex(loko["combined (matched)"]["accuracy"], r"^[01](\.\d+)?$")
        appended = [r for r in rows if r["split"] == "LORO" and r["label space"] == "archetype"]
        self.assertEqual(len(appended), 6)

    def test_gf_void_and_gc_disconnected_replace_score_cells(self):
        out = self.out(gf_void_rung="wapf", gc_disconnected_rung="content")
        tables.table7(out)
        rows = _rows(out / "report" / "tables" / "table7.csv")
        for r in rows:
            if r["rung"] == "wapf":
                self.assertEqual(r["accuracy"], GF_VOID)  # al-Farabi 2.6
                self.assertEqual(r["G-F (i)"], GF_VOID)
            elif r["rung"] == "content":
                self.assertEqual(r["accuracy"], refused(GC_DISCONNECTED))  # al-Farabi 2.5
                self.assertEqual(r["G-C"], GC_DISCONNECTED)
            elif r["rung"] == "apf":
                self.assertRegex(r["accuracy"], r"^[01](\.\d+)?$|^not applicable")

    def test_no_selection_and_best_feasible(self):
        out = self.out(no_selection_rung="persist", best_feasible_rung="content")
        tables.table7(out)
        rows = _rows(out / "report" / "tables" / "table7.csv")
        pers = [r for r in rows if r["rung"] == "persist"]
        self.assertTrue(all(r["accuracy"] == "not run: no selection for persist" for r in pers))
        self.assertTrue(all(r["resolution (W x H)"] == "not run: no selection for persist" for r in pers))
        cont = [r for r in rows if r["rung"] == "content"][0]
        self.assertEqual(cont["resolution (W x H)"], "8 x 4 (selected: best-feasible)")  # al-Farabi 2.11(c)


class TestTable7MatchedRowsEpoch2(_Base):
    """Build epoch 2, builder 3 (SPEC_epoch2 3.5.1, 3.5.2, 6.3 item 6): the `combined (matched)` row is
    filled for every split and label space from gates/splits_matched/, and its `feature count` prints
    the matched dimension (`feature_count_used`) while the `combined` rows keep the full width."""

    def test_table7_matched_rows_all_splits(self):
        out = self.out()
        tables.table7(out)
        rows = _rows(out / "report" / "tables" / "table7.csv")
        self.assertEqual(len(rows), 30)                       # no row added or removed
        matched = [r for r in rows if r["rung"] == "combined (matched)"]
        self.assertEqual(len(matched), 5)
        combos = {(r["split"], r["label space"]) for r in matched}
        self.assertEqual(combos, {("LOKO", "archetype"), ("LORO", "kernel"), ("LORO", "archetype"),
                                  ("within-trace", "kernel"), ("within-trace", "archetype")})
        for r in matched:
            self.assertRegex(r["accuracy"], r"^[01](\.\d+)?$", (r["split"], r["label space"], r["accuracy"]))
            self.assertEqual(r["feature count"], "36", (r["split"], r["label space"]))
            self.assertEqual(r["G-DIM"], "declared reduction (d = 60, matched 36, train_importance)")
        for r in rows:
            if r["rung"] == "combined":
                self.assertEqual(r["feature count"], "60")
        # without the epoch-2 files the row falls back to the epoch-1 path (the fixture's combined_matched, d = 8)
        out1 = self.out("out1", matched_all_splits=False)
        tables.table7(out1)
        rows1 = _rows(out1 / "report" / "tables" / "table7.csv")
        m1 = [r for r in rows1 if r["rung"] == "combined (matched)"]
        self.assertEqual(len(m1), 5)
        self.assertTrue(all(r["feature count"] == "8" for r in m1))
        # and with neither path the cell names the epoch-2 file
        shutil.rmtree(out1 / "gates" / "splits" / "combined_matched")
        tables.table7(out1)
        rows1 = _rows(out1 / "report" / "tables" / "table7.csv")
        m1 = [r for r in rows1 if r["rung"] == "combined (matched)" and r["split"] == "LORO"][0]
        self.assertTrue(m1["accuracy"].startswith("not run: gates/splits_matched/combined/"), m1["accuracy"])
        self.assertIn("loro__kernel/scores.json missing", m1["accuracy"])
        # a matched scores.json whose folds used different widths prints the range, not a single number
        gid = json.loads((out / "gates" / "selection.json").read_text())["combined"]["grid_id"]
        sp = out / "gates" / "splits_matched" / "combined" / gid / "loro__kernel" / "scores.json"
        sc = json.loads(sp.read_text()); sc["feature_count_used"] = "30-36"; sp.write_text(json.dumps(sc))
        tables.table7(out)
        rows = _rows(out / "report" / "tables" / "table7.csv")
        r = [r for r in rows if r["rung"] == "combined (matched)" and r["split"] == "LORO" and r["label space"] == "kernel"][0]
        self.assertEqual(r["feature count"], "30-36")
        # the gf check of move 13 still passes on the table (every row carries its G-F (i) cell)
        from plan11_encoding_ladder import run_moves
        verdict, _ = run_moves.gf_check(out)
        self.assertEqual(verdict, "pass")


class TestTable4C1RuleEpoch2(_Base):
    """Build epoch 2, builder 3 (SPEC_epoch2 section 4; AA T1 "the runbook records which rule was in force
    for each run"): Table 4's Plan 02 cell names the C1 rule in force from gates/preconditions.json."""

    def test_plan02_row_names_the_c1_rule(self):
        out = self.out()
        pj = out / "gates" / "preconditions.json"
        tables.table4_status(out)
        t4 = _rows(out / "report" / "tables" / "table4_status.csv")
        p02 = [r for r in t4 if r["plan"] == "02"][0]["APF at 500 ms"]
        self.assertTrue(p02.startswith("104 of 104 cells all_hard_pass"))          # the epoch-1 assertion still holds
        self.assertIn("; C1 rule: not run: gates/preconditions.json has no C1_rule_in_force", p02)   # a pre-epoch-2 file
        doc = json.loads(pj.read_text()); doc["params"]["C1_rule_in_force"] = "idle_floor_p95"; pj.write_text(json.dumps(doc))
        tables.table4_status(out)
        t4 = _rows(out / "report" / "tables" / "table4_status.csv")
        self.assertTrue([r for r in t4 if r["plan"] == "02"][0]["APF at 500 ms"].endswith("; C1 rule: idle_floor_p95"))
        pj.unlink()
        tables.table4_status(out)
        t4 = _rows(out / "report" / "tables" / "table4_status.csv")
        self.assertIn("; C1 rule: not run: gates/preconditions.json missing", [r for r in t4 if r["plan"] == "02"][0]["APF at 500 ms"])


class TestTable8AndGV(_Base):
    def test_table8_assignments(self):
        out = self.out()
        tables.table8(out)
        rows = _rows(out / "report" / "tables" / "table8.csv")
        self.assertEqual(len(rows), 5)
        byrow = {r["predicted \\ measured"].split(" (")[0]: r for r in rows}
        self.assertIn("lexer", byrow["SEQUENTIAL-GROW"]["IDLE (measured)"])  # G-K0 relabelling
        self.assertTrue(byrow["WORKING-SET"]["WORKING-SET"].startswith("6 ("))
        self.assertEqual(byrow["IDLE"]["predicted \\ measured"], "IDLE (0 predicted)")
        self.assertRegex(byrow["WORKING-SET"]["clusters (k = 4)"], r"^c\d+:\d+")
        self.assertEqual(byrow["WORKING-SET"]["physical reason"], "")
        txt = (out / "report" / "tables" / "table8.tex").read_text()
        self.assertIn("never a confusion matrix", txt)

    def test_table8_reads_builder_2_clustering_layout(self):
        # CHECK_1.md B5: `per_algo[kmeans].labels` aligned with `cells`; the counts per predicted archetype
        out = self.out()
        doc = json.loads((out / "gates" / "clustering.json").read_text())
        self.assertIn("per_algo", doc)
        self.assertIsInstance(doc["cells"], list)
        counts, k, err = tables._cluster_counts(out, "combined", tables.load_cells(out))
        self.assertEqual(err, "")
        self.assertEqual(k, 4)
        self.assertEqual(sum(sum(c.values()) for c in counts.values()), len(doc["cells"]))
        tables.table8(out)
        rows = _rows(out / "report" / "tables" / "table8.csv")
        byrow = {r["predicted \\ measured"].split(" (")[0]: r for r in rows}
        for a in ("WORKING-SET", "SCATTER", "SEQUENTIAL-GROW", "FRONTIER-CHURN"):
            self.assertRegex(byrow[a]["clusters (k = 4)"], r"^c\d+:\d+", a)
        # the count-matrix route when the labels are absent
        for a in doc["per_algo"]:
            doc["per_algo"][a].pop("labels")
        (out / "gates" / "clustering.json").write_text(json.dumps(doc))
        counts2, k2, err2 = tables._cluster_counts(out, "combined", tables.load_cells(out))
        self.assertEqual(err2, "")
        self.assertEqual({a: dict(c) for a, c in counts2.items()}, {a: dict(c) for a, c in counts.items()})
        # builder 2's no-selection file names its refusal
        (out / "gates" / "clustering.json").write_text(json.dumps({"schema": "plan11.clustering.v1", "params": {}, "status": "not run: no selection for combined"}))
        _, _, err3 = tables._cluster_counts(out, "combined", tables.load_cells(out))
        self.assertEqual(err3, "not run: no selection for combined")

    def test_table8_missing_predictions(self):
        out = self.out()
        shutil.rmtree(out / "gates" / "splits" / "combined")
        tables.table8(out)
        rows = _rows(out / "report" / "tables" / "table8.csv")
        self.assertTrue(rows[1]["WORKING-SET"].startswith("not run: gates/splits/combined/"))

    def test_gv_sorted_with_summary(self):
        out = self.out()
        tables.tablegv(out)
        rows = _rows(out / "report" / "tables" / "tablegv.csv")
        for rung in RUNGS:
            rr = [r for r in rows if r["rung"] == rung]
            self.assertTrue(rr[-1]["feature"].startswith("summary"))
            self.assertIn(rr[-1]["verdict"], ("estimable", "LOKO not estimable"))
            vals = [float(r["L0/L3"]) for r in rr[:-1]]
            self.assertEqual(vals, sorted(vals, reverse=True))
        self.assertEqual([r for r in rows if r["rung"] == "wapf"][-1]["verdict"], "LOKO not estimable")
        self.assertEqual([r for r in rows if r["rung"] == "apf"][-1]["verdict"], "estimable")


class TestOtherTablesAndForms(_Base):
    def test_status_wapf_preconditions_manifest(self):
        out = self.out()
        tables.run(out)
        t4 = _rows(out / "report" / "tables" / "table4_status.csv")
        self.assertTrue(any(r["plan"] == "02" and r["APF at 500 ms"].startswith("104 of 104 cells all_hard_pass") for r in t4))
        self.assertTrue(any(r["gates"] == "G-K0" and "lexer" in r["APF at 500 ms"] for r in t4))
        w = _rows(out / "report" / "tables" / "table_wapf_over_apf.csv")
        self.assertEqual(len(w), 13)
        gemm = [r for r in w if r["kernel"] == "gemm"][0]
        self.assertEqual(gemm["n cells"], "8")
        self.assertGreater(float(gemm["mean flipped bits per changed page"]), 0)
        pre = _rows(out / "report" / "tables" / "preconditions.csv")
        self.assertEqual(len(pre), 104)
        m = json.loads((out / "report" / "manifest.json").read_text())
        self.assertEqual(m["schema"], "plan11.manifest.v1")
        self.assertIn("report/tables/table5.csv", m["files"])
        self.assertEqual(len(m["files"]["report/tables/table5.csv"]["sha256"]), 64)
        self.assertIn("report/tables/table5.json", m["params_blocks"])

    def test_latex_and_markdown_forms(self):
        out = self.out()
        tables.run(out)
        tex = (out / "report" / "tables" / "table7.tex").read_text()
        for tok in ("\\toprule", "\\midrule", "\\bottomrule", "\\begin{tabular}", "% columns:", "\\caption{}", "\\label{tab:table7}"):
            self.assertIn(tok, tex)
        self.assertIn("stencil\\_jacobi", (out / "report" / "tables" / "table6.tex").read_text())
        self.assertNotIn("nan", tex.lower().replace("stencil", ""))
        md = (out / "report" / "tables" / "table6.md").read_text().splitlines()
        self.assertTrue(md[0].startswith("| row | level set |"))
        self.assertTrue(md[1].startswith("|---|"))
        for name in ("table5", "table5_g3", "table6", "table7", "table8", "tablegv", "table4_status", "preconditions", "table_wapf_over_apf"):
            for ext in (".csv", ".md", ".tex"):
                self.assertTrue((out / "report" / "tables" / f"{name}{ext}").exists(), name + ext)
        j = json.loads((out / "report" / "tables" / "table6.json").read_text())
        self.assertEqual(set(j) >= {"schema", "params", "citation"}, True)

    def test_without_any_gate_file_every_cell_names_what_is_missing(self):
        out = self.out(with_gates=False)
        rc = tables.main(["--out", str(out)])
        self.assertEqual(rc, 0)
        t6 = _rows(out / "report" / "tables" / "table6.csv")
        self.assertEqual(t6[0]["LOKO norm"], "not run: no selection for apf")
        t5 = _rows(out / "report" / "tables" / "table5.csv")
        self.assertTrue(t5[0]["G1"].startswith("not run: gates/table5_grid.csv missing"))
        gv = _rows(out / "report" / "tables" / "tablegv.csv")
        self.assertEqual(gv[0]["verdict"], "not run: gates/gv.csv missing")

    def test_cli_exit_codes(self):
        self.assertEqual(tables.main(["--out", str(self.tmp / "nowhere")]), 2)
        out = self.out()
        self.assertEqual(tables.main(["--out", str(out), "--only", "table5,nonsense"]), 1)
        self.assertEqual(tables.main(["--out", str(out), "--only", "table5"]), 0)


class TestFigures(_Base):
    def test_all_figures_written(self):
        out = self.out()
        tables.table5(out)
        w = figures.run(out)
        for n in figures.FIGURE_NAMES:
            self.assertTrue((out / "report" / "figures" / f"fig_{n}.pdf").exists(), n)
            self.assertTrue((out / "report" / "figures" / f"fig_{n}.png").exists(), n)
        self.assertGreater(w["piano_roll"]["n_rows_streamed"], 0)  # the one figure that re-streams
        self.assertEqual(w["piano_roll"]["cell_id"], "gemm__rep00__dwarfs1")
        self.assertGreater(w["fused_plane"]["n_masked_points"], 0)  # G-J mask applied
        self.assertEqual(w["j_hist"]["floor_quantiles"], [0.6, 0.8, 0.9, 0.95, 0.99])
        self.assertGreater(w["floyd_decay"]["n_passes_drawn"], 0)
        j = json.loads((out / "report" / "figures" / "figures.json").read_text())
        self.assertEqual(j["status"], "ok")
        self.assertEqual(j["params"]["fused_plane_mask"], "K")

    def test_j_hist_reads_builder_2_floor_quantiles(self):
        # CHECK_1.md B4: builder 2's gj.json shape (gates_readings.py gate_gj) is read first
        out = self.out()
        gj = out / "gates" / "gj.json"
        doc = json.loads(gj.read_text())
        self.assertIn("idle_J", doc)
        self.assertEqual(doc["idle_J"]["quantiles"], [0.05, 0.25, 0.5, 0.75, 0.95])
        q, reason = figures._floor_j_quantiles(out)
        self.assertEqual(q, [0.6, 0.8, 0.9, 0.95, 0.99])
        self.assertEqual(reason, "")
        w = figures.run(out, ["j_hist"])
        self.assertEqual(w["j_hist"]["floor_quantiles"], [0.6, 0.8, 0.9, 0.95, 0.99])
        self.assertEqual(w["j_hist"]["floor_quantiles_absent_reason"], "")
        # no idle cell: idle_J = null, the reason names it (not "gj.json missing")
        doc["idle_J"] = None
        gj.write_text(json.dumps(doc))
        q, reason = figures._floor_j_quantiles(out)
        self.assertIsNone(q)
        self.assertIn("no idle cell", reason)
        w = figures.run(out, ["j_hist"])
        self.assertIsNone(w["j_hist"]["floor_quantiles"])
        self.assertIn("no idle cell", w["j_hist"]["floor_quantiles_absent_reason"])
        # the file itself absent
        gj.unlink()
        q, reason = figures._floor_j_quantiles(out)
        self.assertIsNone(q)
        self.assertIn("gj.json missing", reason)
        # the fallback walker still accepts a `*quant*J*` key
        gj.write_text(json.dumps({"schema": "x", "floor_J_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5]}))
        self.assertEqual(figures._floor_j_quantiles(out), ([0.1, 0.2, 0.3, 0.4, 0.5], ""))

    def test_floyd_placeholder_and_persist_mask(self):
        out = self.out(gdec_verdict="no decay beyond breadth")
        w = figures.run(out, ["floyd_decay", "fused_plane"], fused_plane_mask="persist")
        self.assertTrue((out / "report" / "figures" / "fig_floyd_decay.pdf").exists())
        self.assertNotIn("n_passes_drawn", w["floyd_decay"])  # a placeholder with the verdict, not a decay drawing
        self.assertGreater(w["fused_plane"]["n_masked_points"], 0)

    def test_skipped_without_matplotlib(self):
        out = self.out(with_gates=False)
        env = dict(os.environ, PLAN11_NO_MPL="1")
        rc = subprocess.run([sys.executable, "-m", "plan11_encoding_ladder.figures", "--out", str(out)],
                            cwd=str(_PKG.parent), env=env, capture_output=True, text=True).returncode
        self.assertEqual(rc, 0)
        txt = (out / "report" / "figures" / "SKIPPED.txt").read_text()
        self.assertIn("matplotlib", txt)

    def test_missing_piano_cell_writes_placeholder(self):
        out = self.out()
        w = figures.run(out, ["piano_roll"], piano_cell="nope__rep00__01c")
        self.assertTrue(Path(w["piano_roll"]["pdf"]).exists())
        self.assertNotIn("n_rows_streamed", w["piano_roll"])


class TestSkeleton(_Base):
    def test_targets_exist_and_no_prose(self):
        out = self.out()
        tables.run(out)
        figures.run(out)
        w = latex_skeleton.write_skeleton(out, standalone=self.tmp / "p2_skeleton.tex")
        tex = w["report"].read_text()
        self.assertTrue(tex.startswith("\\documentclass[conference]{IEEEtran}"))
        for t in latex_skeleton.targets(tex):
            self.assertTrue((out / "report" / t).exists(), t)
        self.assertEqual(latex_skeleton.prose_lines(tex), [])
        for sec in ("Introduction", "Background and prior work", "Apparatus", "Encodings of the delta: the ladder",
                    "The gate chain: how a realization's parameters are fixed from the form", "Experimental design",
                    "Results", "Discussion", "Limitations", "Conclusions", "Artifact and reproducibility"):
            self.assertIn(sec, tex)
        self.assertEqual(tex.count("\\begin{document}"), 1)
        self.assertEqual(tex.count("\\end{document}"), 1)
        for env in ("table", "table*", "tabular", "figure", "figure*", "abstract"):
            self.assertEqual(tex.count(f"\\begin{{{env}}}"), tex.count(f"\\end{{{env}}}"), env)
        self.assertIn("6147 per 600 s", tex)  # the pass table's declared column merged into Table 3
        self.assertIn("\\usepackage{booktabs}", tex)
        self.assertTrue((self.tmp / "p2_skeleton.tex").exists())
        self.assertEqual((self.tmp / "p2_skeleton.tex").read_text(), tex)
        art = latex_skeleton.build_skeleton(documentclass="article")
        self.assertTrue(art.startswith("\\documentclass[10pt,twocolumn]{article}"))
        self.assertEqual(latex_skeleton.prose_lines(art), [])
        # every non-comment line is a command, a tabular row or environment syntax
        body = [ln for ln in tex.splitlines() if ln.strip() and not ln.lstrip().startswith("%")]
        self.assertTrue(all(ln.lstrip().startswith(("\\", "}", "{")) or ln.rstrip().endswith("\\\\") or "&" in ln for ln in body))

    def test_cli(self):
        out = self.out(with_gates=False)
        rc = latex_skeleton.main(["--out", str(out), "--documentclass", "article"])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "report" / "paper2_skeleton.tex").exists())
        j = json.loads((out / "report" / "paper2_skeleton.json").read_text())
        self.assertEqual(j["n_prose_lines"], 0)
        self.assertGreaterEqual(len(j["targets"]), 17)



class TestStandaloneSkeleton(unittest.TestCase):
    """apf_paper/p2_skeleton.tex is HAND-MAINTAINED; check its bones, not its bytes.

    latex_skeleton.build_skeleton() emits a scaffold with zero prose_lines(). The live paper
    carries author-approved content the scaffold does not: the separation paragraph (A-B1),
    Table 2's "Claim expected" column and caption (A-I12, A-I13), Sec. IV renamed with
    \\label{sec:readings} (A-I11), the live bibliography and the EUSIPCO citation. An equality
    assertion against the builder would be a demand to delete all of it, so this checks the
    invariants instead. write_skeleton() refuses to overwrite it; this catches it if that guard
    is ever bypassed.
    """

    PAPER = Path("/Users/jeries/Desktop/projects/thesis/memorySignal/apf_paper/p2_skeleton.tex")
    BIB = PAPER.parent / "p2.bib"

    def setUp(self):
        if not self.PAPER.exists():
            self.skipTest(f"{self.PAPER} not present")
        self.tex = self.PAPER.read_text(encoding="utf-8")

    def test_not_regenerated_over(self):
        self.assertGreater(len(latex_skeleton.prose_lines(self.tex)), 0,
                           "p2_skeleton.tex has lost its prose: was it regenerated over?")
        self.assertIn("HAND-MAINTAINED", self.tex.splitlines()[1],
                      "the do-not-regenerate header is missing")

    def test_structural_anchors(self):
        labels = set(re.findall(r"\\label\{([^}]*)\}", self.tex))
        for lab in ("sec:readings", "tab:table2", "tab:table5", "tab:table7", "tab:table8"):
            self.assertIn(lab, labels, lab)
        refs = set(re.findall(r"\\ref\{([^}]*)\}", self.tex))
        self.assertEqual(refs - labels, set(), "dangling \\ref")
        self.assertEqual(self.tex.count("{"), self.tex.count("}"), "unbalanced braces")

    def test_vocabulary_and_citations(self):
        # A-I11 (2026-09-27): readings and reductions, never ladder or rung. The two literal
        # module paths plan11_encoding_ladder/... are the only permitted occurrences.
        hits = [ln for ln in self.tex.splitlines()
                if re.search(r"(?i)\brung\b|\bladder\b", ln.replace("plan11_encoding_ladder", ""))]
        self.assertEqual(hits, [], "ladder/rung vocabulary returned")
        # A-I12/A-I13: Table 2's column states an expectation, not a result.
        self.assertIn("Claim expected", self.tex)
        self.assertNotIn("Claim carried", self.tex)
        # The bibliography is live and every cited key exists.
        self.assertRegex(self.tex, r"(?m)^\\bibliography\{p2\}")
        keys = set(re.findall(r"^@\w+\{([^,]+),", self.BIB.read_text(encoding="utf-8"), re.M))
        cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", self.tex) for k in grp.split(",")}
        self.assertEqual(cited - keys, set(), "cited but not in p2.bib")

if __name__ == "__main__":
    unittest.main()
