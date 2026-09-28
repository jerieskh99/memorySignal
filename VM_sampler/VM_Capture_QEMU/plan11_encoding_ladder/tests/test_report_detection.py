#!/usr/bin/env python3
"""Builder B's tests for tables_detection.py, figures_detection.py and latex_skeleton_p3.py
(SPEC_DETECTION 4.5, the report tests): every table and figure is written on the synthetic
`<out>` tree of detection_fixtures.py and checked (columns exact, every verdict cell a
vocabulary string or `--`, no number in a verdict cell, no blank cell), every "refusal printed as
a string" rule of SPEC_DETECTION section 5 is exercised through one fixture knob, and the
skeleton is checked structurally (brace balance, environments balanced, every `\\input` and
`\\includegraphics` target present after a full run, zero prose lines). Synthetic data only.
Runs under `python3 -m pytest -q`."""
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
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from detection_fixtures import (  # noqa: E402
    AT_FLOOR_NOT_A_MISS, GC_DISCONNECTED, GF_VOID, GSIG_IDENTITY, HARNESS_STAGE2_ABSENT, NULL_NOT_ESTIMABLE, ORDER_VOID,
    make_out,
)
from plan11_encoding_ladder import figures_detection as F  # noqa: E402
from plan11_encoding_ladder import latex_skeleton_p3 as S  # noqa: E402
from plan11_encoding_ladder import tables_detection as T  # noqa: E402
from plan11_encoding_ladder._report_common import RUNGS, to_float  # noqa: E402

VERDICT_COLUMNS = {"verdict", "verdict (roll-up)", "null verdict", "G-CAL", "G-OP", "G-L (i)", "G-C", "G-F (i)", "G-ANCHOR (ii)", "early-late idle",
                   "G-DIM", "G-SIG", "G-1C", "status", "label", "G-K0 verdict"}


def _rows(out: Path, name: str) -> tuple[list[str], list[dict]]:
    p = out / "report" / "detection" / "tables" / f"{name}.csv"
    with open(p, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames or []), list(rd)


class TestTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="plan11_det_tables_"))
        cls.out = make_out(cls.tmp / "out", with_external=True, n_unassigned=1, no_score_cell="sandbox_member_1__rep01__stage1",
                           order_void_rung="wapf", null_inside_rung="persist", gop_set_by_few_rung="apf", one_class_inside_null=True,
                           quarantine_rung="content", gf_void_rung="content", gc_disconnected_rung="combined")
        cls.written = T.run(cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_table_written_with_exact_columns_and_no_blank_cell(self):
        expected = {"table4_tiers": T.TABLE4_COLUMNS, "table5_gates": T.TABLE5_COLUMNS, "table6_validity": T.TABLE6_COLUMNS,
                    "table7_detection": T.TABLE7_COLUMNS, "table9_pitfalls": T.TABLE9_COLUMNS, "table11_splits": T.TABLE11_COLUMNS,
                    "table_ladder": T.LADDER_COLUMNS, "table10_misses": T.MISS_COLUMNS + [T.REASON_COLUMN],
                    "table10_false_positives": T.FP_COLUMNS + [T.REASON_COLUMN]}
        for name, cols in expected.items():
            hdr, rows = _rows(self.out, name)
            self.assertEqual(hdr, list(cols), name)
            self.assertTrue(rows, name)
            for r in rows:
                for c, v in r.items():
                    if c == T.REASON_COLUMN:
                        continue  # the author's empty text column (SPEC_DETECTION 5.2.6)
                    self.assertNotEqual(str(v).strip(), "", f"{name}: blank cell in column {c!r} of row {r.get('rung', r.get('gate', ''))!r}")
        for name in T.TABLE_NAMES:
            if name == "manifest":
                self.assertTrue((self.out / "report" / "detection" / "manifest.json").exists())
                continue
            d = self.out / "report" / "detection" / "tables"
            for ext in ("csv", "md", "tex", "json"):
                self.assertTrue((d / f"{name}.{ext}").exists(), f"{name}.{ext}")
        for rung in RUNGS:
            for stem in ("table10_misses", "table10_false_positives", "table_level2", "table_level3"):
                self.assertTrue((self.out / "report" / "detection" / "tables" / f"{stem}_{rung}.tex").exists(), f"{stem}_{rung}")

    def test_verdict_cells_hold_vocabulary_strings_never_numbers(self):
        for name in ("table5_gates", "table6_validity", "table7_detection", "table9_pitfalls", "table11_splits", "table_level2", "table_level3", "table_ladder"):
            hdr, rows = _rows(self.out, name)
            for r in rows:
                for c in hdr:
                    if c in VERDICT_COLUMNS:
                        v = str(r[c])
                        self.assertIsNone(to_float(v), f"{name}: number {v!r} in verdict column {c!r}")
                        self.assertTrue(v == "--" or v, f"{name}: blank verdict in {c!r}")

    def test_table7_rows_and_refusal_strings(self):
        hdr, rows = _rows(self.out, "table7_detection")
        by = {r["rung"]: r for r in rows}
        for disp in ("apf raw", "apf", "wapf", "persist", "content", "combined", "combined (matched)", "content channel 2'", "comparator"):
            self.assertIn(disp, by)
        # ORDER_VOID (consequence void) in every score cell of the rung; the verdict column keeps its verdict (SPEC_DETECTION 5)
        for c in T.SCORE_COLUMNS_7:
            self.assertEqual(by["wapf"][c], ORDER_VOID, c)
        self.assertEqual(by["content"]["G-F (i)"], GF_VOID)
        for c in T.SCORE_COLUMNS_7:
            self.assertEqual(by["content"][c], GF_VOID, c)
        self.assertEqual(by["combined"]["G-C"], GC_DISCONNECTED)
        self.assertTrue(by["combined"]["ROC area"].startswith("refused: disconnected lead"))
        # the placeholders of the preamble in every number cell
        self.assertEqual(by["content channel 2'"]["ROC area"], T.RUNG2P_NOT_BUILT)
        self.assertEqual(by["comparator"]["TPR at 5% (in-fold)"], T.COMPARATOR_ELSEWHERE)
        # G-L (i): the NULL_INSIDE rung reads level only; G-OP set by few carries the family
        self.assertEqual(by["persist"]["G-L (i)"], "level only")
        self.assertTrue(by["apf"]["G-OP"].startswith(T.GOP_SET_BY_FEW))
        self.assertEqual(by["apf raw"]["G-L (i)"], "level-inclusive ceiling (raw)")
        # al-Farabi M4: n without score beside n at floor
        self.assertEqual(by["apf"]["n without score"], "1")
        self.assertEqual(by["apf"]["n at floor"], "4")
        # ML 2.4: the quarantined feature named in the B1-G3 column, the re-run's TPR beside it, the l1 row per rung
        self.assertEqual(by["content"]["B1-G3 quarantine"], "content.r_l0_q50_per.mean")
        self.assertEqual(by["apf"]["B1-G3 quarantine"], "none")
        self.assertTrue(by["apf"]["TPR at 5% without quarantined"].startswith("not applicable"))
        l1 = [r for r in rows if "l1 (best single feature)" in r["rung"]]
        self.assertEqual(len(l1), 6)  # apf raw, apf, wapf, persist, content, combined
        self.assertTrue(l1[1]["B1-G3 quarantine"].startswith("feature: "))
        # the external block (P3 0a stage 3)
        ext = [r for r in rows if r["rung"].endswith("(external)")]
        self.assertEqual(len(ext), 6)
        self.assertTrue(ext[1]["null verdict"].startswith("not applicable: external cells are test only"))
        # the random scorer and the pooled label; the note line carries the majority baseline and G-N
        self.assertEqual(by["apf"]["random scorer"], "AUC 0.5, TPR = FPR")
        tex = (self.out / "report" / "detection" / "tables" / "table7_detection.tex").read_text()
        self.assertIn("majority (always benign)", tex)
        self.assertIn("G-N sandbox:", tex)
        self.assertIn(T.POST_HOC_LABEL, tex)
        self.assertIn("threshold_source = inner_lowo", tex)

    def test_table8_eighths_letters_ratio_and_external(self):
        hdr, rows = _rows(self.out, "table8_member_recall")
        self.assertEqual(hdr[:2], ["rung", "split"])
        self.assertIn("member 1", hdr)
        self.assertIn("external 1", hdr)
        self.assertNotIn("mean", [h.lower() for h in hdr])
        self.assertEqual(rows[0]["rung"], "sub-family")
        self.assertEqual(rows[0]["member 1"], "A")
        self.assertEqual(rows[0]["member 3"], "B")
        apf = {r["split"]: r for r in rows if r["rung"] == "apf"}
        self.assertEqual(apf["lowo"]["member 3"], "at floor (4)")
        self.assertRegex(apf["lowo"]["member 2"], r"^\d+/\d+$")
        self.assertEqual(apf["lowo"]["n without score"], "1")
        self.assertIn("loco: L0 ratio (reps identical?)", apf)
        self.assertIn(T.REPS_IDENTICAL_NOTE, apf["loco: L0 ratio (reps identical?)"]["note"])
        wapf = {r["split"]: r for r in rows if r["rung"] == "wapf"}
        self.assertEqual(wapf["lowo"]["member 1"], ORDER_VOID)
        raw = {r["split"]: r for r in rows if r["rung"] == "apf raw"}
        self.assertTrue(raw["one_class"]["member 1"].startswith("not applicable: the one-class run is on the normalized features"))

    def test_table11_gsig_identity_and_f9_note(self):
        hdr, rows = _rows(self.out, "table11_splits")
        by = {r["rung"]: r for r in rows}
        self.assertEqual(by["persist"]["G-SIG"], GSIG_IDENTITY)
        self.assertIn(T.F9_SENTENCE, by["apf"]["note"])
        self.assertIn("signature ceiling", hdr[3])
        self.assertEqual(by["wapf"]["LOWO TPR 5% (FPR)"], ORDER_VOID)
        self.assertRegex(by["apf"]["LOFO FPR per family"], r"idle [0-9.]+; kernels [0-9.]+")
        for c in ("G-F (i)", "G-ANCHOR (ii)", "early-late idle"):
            self.assertIn(c, hdr)
            self.assertNotEqual(by["apf"][c], "--")

    def test_table4_counts_unassigned_and_stage2_rows(self):
        hdr, rows = _rows(self.out, "table4_tiers")
        by = {r["tier (class)"]: r for r in rows}
        self.assertEqual(by["sandbox"]["n cells"], "12")
        self.assertEqual(by["sandbox"]["n at floor"], "4")
        self.assertEqual(by["sandbox"]["n C1 fail (reported)"], "4")
        self.assertEqual(by["unassigned"]["n cells"], "1")
        self.assertEqual(by["benign_relaunched"]["n admissible"], HARNESS_STAGE2_ABSENT)
        self.assertEqual(by["external"]["letter"], "X")
        self.assertEqual(by["total"]["n cells"], "29")

    def test_table5_rollup_and_table6_rows(self):
        hdr, rows = _rows(self.out, "table5_gates")
        by = {r["gate"]: r for r in rows}
        self.assertIn("(", by["G-OP"]["verdict (roll-up)"])
        self.assertTrue(by["G-OP"]["verdict (roll-up)"].startswith(T.GOP_SET_BY_FEW))
        self.assertTrue(by["harness clause"]["refusal string"].startswith(HARNESS_STAGE2_ABSENT))
        self.assertEqual(by["classes validate"]["verdict (roll-up)"], "ok")
        self.assertTrue(by["G-C"]["verdict (roll-up)"].startswith(GC_DISCONNECTED))
        self.assertTrue(by["tripwire check (D15)"]["verdict (roll-up)"].startswith("not run:"))
        for g in ("C1", "C8", "failed/ count", "G-K0 kernels", "G-F", "G-J", "G-X", "G-P", "leak probe", "B1-G1 two-class", "B1-G3 two-class"):
            self.assertIn(g, by)
        hdr6, rows6 = _rows(self.out, "table6_validity")
        names = {r["row"] for r in rows6}
        for n in ("G-C (calibrated core)", "idle floor", "harness floor", "idle set against idle set (G-ANCHOR ii)", "early against late idle",
                  "drift regression", "state-change yield", "G-K0 counts per tier", "G-F (i)"):
            self.assertIn(n, names)
        yield_row = [r for r in rows6 if r["row"] == "state-change yield"][0]
        self.assertEqual(yield_row["verdict"], T.YIELD_NOT_RECORDED)
        gk = [r for r in rows6 if r["row"] == "G-K0 counts per tier" and r["quantity"].startswith("sandbox")][0]
        self.assertEqual(gk["value"], "4 / 0 / 8")
        drift = [r for r in rows6 if r["row"] == "drift regression"]
        self.assertEqual(len(drift), 6)  # three classes x two drift units (al-Kindi review 10)

    def test_table9_pitfalls_sizes(self):
        hdr, rows = _rows(self.out, "table9_pitfalls")
        pit = {(r["pitfall"], r["instrument"]) for r in rows}
        for k in (("level", "G-L (i)"), ("level", "G-LM"), ("campaign", "G-ANCHOR (i)"), ("campaign", "G-ANCHOR (ii)"), ("order", "order test"),
                  ("order", "drift regression"), ("harness", "harness clause"), ("campaign", "cross-campaign row"), ("cadence", "leak probe"),
                  ("active fraction", "leak probe")):
            self.assertIn(k, pit)
        cc = [r for r in rows if r["instrument"] == "cross-campaign row"][0]
        self.assertEqual(cc["size"], T.CROSS_CAMPAIGN_STAGE1)
        order = [r for r in rows if r["instrument"] == "order test" and r["rung"] == "wapf" and r["class or member"] == "sandbox"]
        self.assertEqual({r["verdict"] for r in order}, {ORDER_VOID})
        self.assertEqual(len(order), 2)  # within_workload and within_class half rules (al-Kindi review 11)
        for r in rows:
            self.assertNotEqual(r["size"], "")
            if r["verdict"] == "pass":
                self.assertIsNotNone(to_float(r["size"]) if r["instrument"] != "G-L (i)" or r["rung"] == "apf" else 0)
        glm = [r for r in rows if r["instrument"] == "G-LM" and "member 3" in r["class or member"]][0]
        self.assertEqual(glm["verdict"], "not applicable: at floor (G-K0)")
        self.assertIn(T.GLM_LABEL_MEDIAN_K, glm["note"])

    def test_table10_level_tables_ladder_gv_cells(self):
        hdr, rows = _rows(self.out, "table10_misses")
        self.assertEqual(hdr[-1], T.REASON_COLUMN)
        self.assertIn("axis_of_largest", hdr)
        self.assertTrue(all(r["rung"] == "combined" for r in rows))
        self.assertIn(AT_FLOOR_NOT_A_MISS, {r["status"] for r in rows})
        _, rows_fp = _rows(self.out, "table10_false_positives")
        self.assertTrue(rows_fp)
        hdr2c, rows2c = _rows(self.out, "table_level2")   # the chosen rung (combined) is the G-C disconnected rung: the override prints
        self.assertEqual(rows2c[0]["recall"], "refused: disconnected lead")
        hdr2, rows2 = _rows(self.out, "table_level2_apf")
        self.assertIn("predicted A", hdr2)
        self.assertIn("n at floor", hdr2)
        B = [r for r in rows2 if r["true sub-family"] == "B"][0]
        self.assertEqual(B["status"], "1 member, no held-out test")
        self.assertEqual(B["recall"], "--")
        self.assertEqual(B["predicted A"], "--")
        A = [r for r in rows2 if r["true sub-family"] == "A"][0]
        self.assertNotEqual(A["null p95"], "--")
        self.assertNotEqual(A["verdict"], "--")
        hdr3, rows3 = _rows(self.out, "table_level3_apf")
        self.assertEqual({r["label"] for r in rows3}, {T.SIGNATURE_CEILING})
        self.assertEqual(rows3[-1]["true member"], "accuracy")
        self.assertEqual(rows3[2]["recall (eighths)"], "at floor (4)")
        hdrl, rowsl = _rows(self.out, "table_ladder")
        fb = [r for r in rowsl if r["reading"] == "from_boundary"]
        self.assertTrue(fb)
        self.assertEqual({r["TPR at 5%"] for r in fb}, {T.LADDER_FROM_PAIR1_ONLY})
        fp = [r for r in rowsl if r["reading"] == "from_pair1" and r["rung"] == "apf"]
        self.assertEqual([r["pairs at 0.644 s"] for r in fp], ["47", "93", "186", "466", "932"])
        self.assertEqual({r["TPR at 5%"] for r in rowsl if r["rung"] == "wapf" and r["reading"] == "from_pair1"}, {ORDER_VOID})
        hdrg, rowsg = _rows(self.out, "table_gv_two_class")
        self.assertTrue(any(r["feature"].startswith("summary") for r in rowsg))
        hdrc, rowsc = _rows(self.out, "table_cells_detection")
        for banned in ("path", "label", "test_label", "traj_file", "order_index"):
            self.assertNotIn(banned, hdrc)
        self.assertIn("order token", hdrc)
        toks = {r["order token"] for r in rowsc}
        self.assertIn("S1r0", toks)
        noscore = [r for r in rowsc if r["cell_id"] == "sandbox_member_1__rep01__stage1"][0]
        self.assertTrue(noscore["LOWO score apf"].startswith("not applicable: no finite window"))
        un = [r for r in rowsc if r["class"] == "unassigned"]
        self.assertEqual(len(un), 1)
        man = json.loads((self.out / "report" / "detection" / "manifest.json").read_text())
        self.assertEqual(man["schema"], "plan11.detection.manifest.v1")
        self.assertGreater(man["n_files"], 40)
        self.assertEqual(man["S"], 3)
        self.assertTrue(any(k.endswith("table7_detection.json") for k in man["params_blocks"]))


class TestTableRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plan11_det_ref_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_smoke_run_prints_the_permutation_refusal(self):
        out = make_out(self.tmp / "out", smoke_perm=20)
        T.table7_detection(out)
        _, rows = _rows(out, "table7_detection")
        by = {r["rung"]: r for r in rows}
        self.assertEqual(by["apf"]["null verdict"], "not run: 20 permutations < 500")
        self.assertIsNotNone(to_float(by["apf"]["TPR at 5% (in-fold)"]))  # the numbers stay

    def test_null_not_estimable_carries_no_verb(self):
        out = make_out(self.tmp / "out", null_not_estimable_rung="content")
        T.table7_detection(out)
        _, rows = _rows(out, "table7_detection")
        by = {r["rung"]: r for r in rows}
        self.assertEqual(by["content"]["null verdict"], NULL_NOT_ESTIMABLE)
        self.assertEqual(by["content"]["TPR null p95"], NULL_NOT_ESTIMABLE)
        self.assertEqual(by["content"]["TPR rank"], NULL_NOT_ESTIMABLE)

    def test_missing_split_and_missing_gate_file_print_the_file(self):
        out = make_out(self.tmp / "out", missing_split=("persist", "loco"))
        (out / "gates" / "detection" / "gsig.csv").unlink()
        T.table11_splits(out)
        _, rows = _rows(out, "table11_splits")
        by = {r["rung"]: r for r in rows}
        self.assertTrue(by["persist"]["LOCO TPR 5% (FPR) [signature ceiling]"].startswith("not run: gates/detection/splits/persist/W8_H4/loco__norm/scores.json missing"))
        self.assertEqual(by["apf"]["G-SIG"], "not run: gates/detection/gsig.csv missing")
        T.table5_gates(out)
        _, r5 = _rows(out, "table5_gates")
        g = [r for r in r5 if r["gate"] == "G-SIG"][0]
        self.assertEqual(g["verdict (roll-up)"], "not run: gates/detection/gsig.csv missing")

    def test_no_selection_rung(self):
        out = make_out(self.tmp / "out", no_selection_rung="wapf")
        T.table7_detection(out)
        _, rows = _rows(out, "table7_detection")
        by = {r["rung"]: r for r in rows}
        self.assertEqual(by["wapf"]["resolution (W x H)"], "not run: no selection for wapf")
        self.assertTrue(by["wapf"]["ROC area"].startswith("not run: no selection for wapf"))
        T.table_level2(out)
        _, rows2 = _rows(out, "table_level2_wapf")
        self.assertTrue(rows2[0]["status"].startswith("not run: no selection for wapf"))

    def test_absent_join_and_absent_gates(self):
        out = make_out(self.tmp / "out", with_gates=False)
        T.run(out)
        _, rows = _rows(out, "table4_tiers")
        self.assertEqual(rows[0]["n cells"], "8")
        _, rows7 = _rows(out, "table7_detection")
        by = {r["rung"]: r for r in rows7}
        self.assertTrue(by["apf"]["ROC area"].startswith("not run: gates/selection.json") or by["apf"]["ROC area"].startswith("not run: no selection"))
        self.assertEqual(by["apf"]["G-C"], "not run: gates/gc.csv missing")
        _, rows6 = _rows(out, "table6_validity")
        floor = [r for r in rows6 if r["row"] == "idle floor"][0]
        self.assertEqual(floor["value"], "not run: gates/detection/gk0.json missing")

    def test_stage2_two_campaigns_and_no_order_index(self):
        out = make_out(self.tmp / "out", stage2=True, two_idle_campaigns=True, no_order_index=True)
        T.table9_pitfalls(out)
        _, rows = _rows(out, "table9_pitfalls")
        h = [r for r in rows if r["instrument"] == "harness clause" and r["rung"] == "apf"]
        self.assertTrue(all(r["verdict"] != HARNESS_STAGE2_ABSENT for r in h))
        a2 = [r for r in rows if r["instrument"] == "G-ANCHOR (ii)" and r["rung"] == "apf"][0]
        self.assertEqual(a2["verdict"], "campaign audible")
        od = [r for r in rows if r["instrument"] == "order test"]
        self.assertEqual({r["verdict"] for r in od}, {"not run: order_index missing"})
        self.assertEqual({r["size"] for r in od}, {"not run: order_index missing"})
        T.table_cells_detection(out)
        _, rc = _rows(out, "table_cells_detection")
        self.assertTrue(rc[0]["order token"].startswith("not run: order_index missing"))

    def test_cli_exit_codes(self):
        so, se = io.StringIO(), io.StringIO()
        with redirect_stdout(so), redirect_stderr(se):
            rc = T.main(["--out", str(self.tmp / "absent")])
        self.assertEqual(rc, 2)
        self.assertIn("cells.csv", se.getvalue())
        out = make_out(self.tmp / "out")
        with redirect_stdout(so), redirect_stderr(se):
            rc = T.main(["--out", str(out), "--only", "nonesuch"])
        self.assertEqual(rc, 1)
        with redirect_stdout(so), redirect_stderr(se):
            rc = T.main(["--out", str(out), "--only", "table4_tiers,manifest", "--table10-rung", "content"])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "report" / "detection" / "manifest.json").exists())


class TestFigures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plan11_det_figs_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_every_figure_written_and_params_recorded(self):
        out = make_out(self.tmp / "out", with_external=True, two_idle_campaigns=True)
        w = F.run(out, plane_per_letter=True)
        fd = out / "report" / "detection" / "figures"
        for n in F.FIGURE_NAMES:
            self.assertIn(n, w)
            self.assertTrue((fd / f"{n}.pdf").exists(), n)
            self.assertTrue((fd / f"{n}.png").exists(), n)
        j = json.loads((fd / "figures.json").read_text())
        self.assertEqual(j["status"], "ok")
        self.assertEqual(j["params"]["identity"], "excess")
        self.assertEqual(j["params"]["plane_mask"], F.PLANE_MASK_LABEL)
        self.assertEqual(w["fig4_fused_plane_tiers"]["plane_mask"], F.PLANE_MASK_LABEL)
        self.assertGreater(w["fig4_fused_plane_tiers"]["n_masked_points"], 0)
        self.assertEqual(w["fig5_roc_lowo"]["missing"], [])

    def test_placeholders_and_raw_identity(self):
        out = make_out(self.tmp / "out", with_gates=False)
        w = F.run(out, identity="raw")
        fd = out / "report" / "detection" / "figures"
        for n in F.FIGURE_NAMES:
            self.assertTrue((fd / f"{n}.pdf").exists(), n)
        j = json.loads((fd / "figures.json").read_text())
        self.assertEqual(j["status"], "ok")  # placeholders carry the not-run strings, no error
        self.assertEqual(w["fig4_fused_plane_tiers"]["plane_mask"], "unmasked: gates/gj.json absent")

    def test_matplotlib_absent_writes_skipped(self):
        out = make_out(self.tmp / "out", with_gates=False, with_extracts=False)
        os.environ["PLAN11_NO_MPL"] = "1"
        try:
            w = F.run(out)
        finally:
            del os.environ["PLAN11_NO_MPL"]
        self.assertIn("skipped", w)
        self.assertTrue((out / "report" / "detection" / "figures" / "SKIPPED.txt").exists())

    def test_cli(self):
        out = make_out(self.tmp / "out")
        so, se = io.StringIO(), io.StringIO()
        with redirect_stdout(so), redirect_stderr(se):
            rc = F.main(["--out", str(out), "--only", "fig6_ladder,fig_level_map"])
        self.assertEqual(rc, 0)
        with redirect_stdout(so), redirect_stderr(se):
            rc = F.main(["--out", str(self.tmp / "absent")])
        self.assertEqual(rc, 2)


class TestSkeleton(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plan11_det_tex_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_structure_no_prose_and_targets_exist_after_a_full_run(self):
        out = make_out(self.tmp / "out")
        T.run(out)
        F.run(out)
        w = S.write_skeleton(out, standalone=self.tmp / "standalone" / "p3_skeleton.tex")
        tex = w["report"].read_text()
        self.assertEqual(tex, w["standalone"].read_text())
        self.assertTrue(tex.startswith("\\documentclass[runningheads]{llncs}"))
        self.assertEqual(S.prose_lines(tex), [])
        self.assertEqual(S.brace_balance(tex), 0)
        begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
        ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
        self.assertEqual(sorted(begins), sorted(ends))
        for sec in ("Introduction", "Background and prior work", "The channel and the instrument", "Dataset and capture design", "Evaluation protocol",
                    "Validity of the campaign", "Results", "Discussion and limitations", "Conclusion"):
            self.assertIn(f"\\section{{{sec}}}", tex)
        for rq in ("RQ1", "RQ2", "RQ3", "RQ4", "RQ5"):
            self.assertIn(f"\\subsection{{{rq}", tex)
        self.assertIn("% \\subsection{RQ6", tex)
        self.assertIn("% box if it holds:", tex)
        self.assertIn("% box if it fails:", tex)
        self.assertIn("commit & fcc184e", tex)
        self.assertIn("\\cite{clark2005livemigration}", tex)
        self.assertIn("\\bibliography{p3}", tex)
        self.assertIn("\\bibliographystyle{splncs04}", tex)
        rep = out / "report" / "detection"
        for t in S.targets(tex):
            self.assertTrue((rep / t).exists(), t)
        j = json.loads((rep / "paper3_skeleton.json").read_text())
        self.assertEqual(j["n_prose_lines"], 0)
        self.assertEqual(j["brace_balance"], 0)
        # no sentence outside a comment: no line outside comments ends in a period
        for raw in tex.splitlines():
            line = raw.strip()
            if line.startswith("%") or not line:
                continue
            self.assertFalse(line.endswith(".") and not line.startswith("\\"), raw)

    def test_article_fallback_and_cli(self):
        tex = S.build_skeleton(documentclass="article")
        self.assertTrue(tex.startswith("\\documentclass[10pt]{article}"))
        self.assertNotIn("\\institute{}", tex)
        self.assertEqual(S.prose_lines(tex), [])
        so, se = io.StringIO(), io.StringIO()
        with redirect_stdout(so), redirect_stderr(se):
            rc = S.main(["--out", str(self.tmp / "out"), "--documentclass", "article"])
        self.assertEqual(rc, 0)
        self.assertTrue((self.tmp / "out" / "report" / "detection" / "paper3_skeleton.tex").exists())

    def test_prose_detector_catches_a_sentence(self):
        tex = S.build_skeleton() + "\nThe detector reads behaviour and not identity.\n"
        self.assertEqual(len(S.prose_lines(tex)), 1)


if __name__ == "__main__":
    unittest.main()
