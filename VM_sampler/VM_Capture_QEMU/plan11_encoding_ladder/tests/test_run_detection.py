#!/usr/bin/env python3
"""Builder B's tests for run_detection.py (the detection driver, SPEC_DETECTION section 6). The
synthetic `<out>` tree of detection_fixtures.py stands in for builder A; the driver's own steps
(the tables, the figures, the skeleton, the two internal steps, the ledger, the skip and
staleness rules, the class-file placement) run for real, builder A's commands are filtered with
`--only-modules` and recorded (SPEC_DETECTION 4.5). Synthetic data only. Runs under
`python3 -m pytest -q`."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import argparse
import csv
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from detection_fixtures import make_out  # noqa: E402
from plan11_encoding_ladder import run_detection as D  # noqa: E402
from plan11_encoding_ladder import run_moves  # noqa: E402
from plan11_encoding_ladder._report_common import RUNGS  # noqa: E402

MINE = "tables_detection,figures_detection,latex_skeleton_p3,driver"


def _run(argv):
    so, se = io.StringIO(), io.StringIO()
    with redirect_stdout(so), redirect_stderr(se):
        rc = D.main(argv)
    return rc, so.getvalue(), se.getvalue()


def _ledger(out: Path) -> dict:
    return json.loads((out / D.LEDGER).read_text())


def _ns(**kw) -> argparse.Namespace:
    base = dict(out="/x/out", root="/x/root", classes=None, selection_from=None, grid_default="W8_H4", moves="0-15", assume_failed_zero=False,
                assume_reason=None, c1_activity_min=None, c1_rule="report", campaign_label=None, kernel_family_rule="tier", relaunched_grouping="parent",
                n_jobs=1, null_perm=500, null_splits=D.NULL_SPLITS_DEFAULT, null_rungs=",".join(RUNGS), ladder_null_perm=0, loco_mode="cell",
                one_class_model="isolation_forest", threshold_source=D.THRESHOLD_SOURCE_DEFAULT, row_unit=None, train_on_at_floor=None,
                n_estimators=None, level_quantity="median_K", order_consequence="size", order_scope=None, gop_cells_rule=None, table10_rung="combined",
                identity="excess", plane_per_letter=False, documentclass="llncs", seed_offset=0, force=False, dry_run=False, skip_missing_modules=False,
                only_modules=None, persist_side=None, failed_counts=None, standalone_tex=None, cmd="plan")
    base.update(kw)
    return argparse.Namespace(**base)


class TestPlan(unittest.TestCase):
    def test_parse_moves_bounded_by_15_through_the_additive_extension(self):
        self.assertEqual(run_moves.parse_moves("0-15", 15), list(range(16)))
        with self.assertRaises(ValueError):
            run_moves.parse_moves("16", 15)
        # the epoch-1 default is unchanged in shape: a bound below 15 still refuses 15
        with self.assertRaises(ValueError):
            run_moves.parse_moves("15", 13)

    def test_move_table_carries_the_review_corrections(self):
        P = D.build_plan(_ns(seed_offset=3, n_jobs=4, campaign_label="stage1"))
        names = [(c["move"], c["name"]) for c in P]
        self.assertEqual([n for m, n in names if m == 0], ["extract index", "classes validate", "classes apply", "classes inherit-selection", "classes letter-sequence"])
        # al-Farabi M5: the head-drop template covers every workload key
        self.assertIn((2, "head-drop template (every workload key)"), names)
        hd = [c for c in P if c["name"] == "head-drop template (every workload key)"][0]
        self.assertEqual((hd["module"], hd["sub"]), ("gates_detection", "head-drop-template"))
        self.assertTrue(hd["template"])
        self.assertEqual(len([c for c in P if c["template"]]), 5)
        # al-Kindi 1: inner_lowo passed to every split and to the ladder
        for c in P:
            if (c["module"], c["sub"]) in (("detection_metrics", "splits"), ("detection_metrics", "ladder")):
                self.assertIn("inner_lowo", c["args"], c["name"])
        # al-Kindi 2 / ML 2.7: the one-class run carries --null-perm and --n-jobs; one_class in --null-splits
        oc = [c for c in P if c["name"] == "one-class apf"][0]
        self.assertIn("--null-perm", oc["args"])
        self.assertIn("500", oc["args"])
        self.assertIn("--n-jobs", oc["args"])
        sp = [c for c in P if c["name"] == "splits apf"][0]
        self.assertIn("lowo,loco,one_class", sp["args"])
        self.assertIn("--raw-and-norm", sp["args"])
        self.assertIn("--seed-offset", sp["args"])
        # ML 2.3: G-OP on lowo, loco, lofo per rung
        for rung in RUNGS:
            for split in ("lowo", "loco", "lofo"):
                self.assertIn((D.MOVE_OF_RUNG[rung], f"gop {rung} {split}"), names)
        # ML 2.6: the leak probe at D7
        self.assertIn((7, "leak probe"), names)
        # al-Kindi 5: --c1-rule passed through
        adm = [c for c in P if c["name"] == "admissibility"][0]
        self.assertEqual(adm["args"][-2:], ["--c1-rule", "report"])
        # al-Kindi 12 / 7: figure flags
        f4 = [c for c in P if c["name"] == "figures fig4_fused_plane_tiers"][0]
        self.assertIn("--identity", f4["args"])
        # the campaign label reaches classes apply
        ap = [c for c in P if c["name"] == "classes apply"][0]
        self.assertIn("--campaign-label", ap["args"])
        # the order in al-Kindi's moves: D7 apf, D9 persist, D10 content, D11 wapf, D12 combined; D15 last
        self.assertEqual([c["name"] for c in P if c["move"] == 15], ["tripwire check"])
        m12 = [c["name"] for c in P if c["move"] == 12]
        self.assertLess(m12.index("splits combined"), m12.index("splits combined (matched)"))
        self.assertLess(m12.index("gm"), m12.index("tables table7,table8,table11,levels"))
        m14 = [c["name"] for c in P if c["move"] == 14]
        self.assertLess(m14.index("tables (all)"), m14.index("tables manifest"))
        self.assertLess(m14.index("latex skeleton p3"), m14.index("tables manifest"))
        # the declared inputs (SPEC_DETECTION 6.1): the class file, the join, admissibility, the selection and the floor
        for c in P:
            if c["module"] in ("detection_metrics", "detection_levels") and not c["internal"]:
                for i in ("inputs/classes.csv", "cells.csv", "gates/detection/cell_classes.csv", "gates/detection/admissibility.csv", "gates/detection/gk0_cells.csv"):
                    self.assertIn(i, c["inputs"], c["name"])
                self.assertTrue(any(i.startswith("json:gates/selection.json:") for i in c["inputs"]), c["name"])
        # the null is not requested for a rung outside --null-rungs
        P2 = D.build_plan(_ns(null_rungs="apf,combined"))
        sp = [c for c in P2 if c["name"] == "splits persist"][0]
        i = sp["args"].index("--null-splits")
        self.assertEqual(sp["args"][i + 1], "")
        oc = [c for c in P2 if c["name"] == "one-class persist"][0]
        self.assertEqual(oc["args"][oc["args"].index("--null-perm") + 1], "0")
        # optional pass-throughs appear only when given
        P3 = D.build_plan(_ns(train_on_at_floor="false", row_unit="cell", order_scope="within_class", gop_cells_rule="realized_fps", selection_from="/enc/gates/selection.json"))
        sp = [c for c in P3 if c["name"] == "splits apf"][0]
        self.assertIn("--train-on-at-floor", sp["args"])
        self.assertIn("--row-unit", sp["args"])
        self.assertIn("--scope", [c for c in P3 if c["name"] == "order test"][0]["args"])
        self.assertIn("--cells-rule", [c for c in P3 if c["name"] == "gop apf lowo"][0]["args"])
        sel = [c for c in P3 if c["name"] == "classes inherit-selection"][0]
        self.assertEqual(sel["args"][-2:], ["--from", "/enc/gates/selection.json"])
        self.assertNotIn("--train-on-at-floor", [c for c in P if c["name"] == "splits apf"][0]["args"])

    def test_cli_guards(self):
        rc, _, err = _run(["run", "--out", "/x", "--assume-failed-zero"])
        self.assertEqual(rc, 2)
        self.assertIn("--assume-reason", err)
        rc, _, err = _run(["run", "--out", "/x", "--moves", "0"])
        self.assertEqual(rc, 2)
        self.assertIn("--root", err)
        rc, _, err = _run(["run", "--out", "/x", "--root", "/r", "--moves", "0"])
        self.assertEqual(rc, 2)
        self.assertIn("--selection-from", err)
        rc, _, err = _run(["run", "--out", "/x", "--moves", "16"])
        self.assertEqual(rc, 2)
        rc, out, _ = _run(["plan", "--out", "/x", "--root", "/r", "--grid-default", "W8_H4", "--moves", "7"])
        self.assertEqual(rc, 0)
        self.assertIn("gates_detection leak-probe", out)
        self.assertIn("--threshold-source inner_lowo", out)


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plan11_det_driver_"))
        self.out = make_out(self.tmp / "out", n_pairs=40)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_records_every_command(self):
        cls = self.tmp / "classes.csv"
        shutil.copyfile(self.out / "inputs" / "classes.csv", cls)
        rc, _, _ = _run(["run", "--out", str(self.out), "--root", str(self.out / "synth_root"), "--grid-default", "W8_H4", "--classes", str(cls), "--dry-run"])
        self.assertEqual(rc, 0)
        L = _ledger(self.out)
        self.assertEqual(L["schema"], "plan11.driver_state.v1")
        st = [c["status"] for c in L["commands"]]
        self.assertTrue(all(s in ("dry-run", "done") or s.startswith("kept:") for s in st))
        self.assertEqual(L["commands"][0]["name"], "classes copy")
        self.assertEqual(L["commands"][0]["status"], "done")
        self.assertEqual(sum(1 for c in L["commands"] if c["move"] == 15), 1)
        self.assertTrue(any(c["status"] == "kept: author input exists" and c["name"] == "head-drop template (every workload key)" for c in L["commands"]))
        self.assertTrue(all("inputs_sha256" in c for c in L["commands"]))
        self.assertFalse((self.out / "driver_state.json").exists())  # its own ledger, never the encoding paper's

    def test_classes_copy_refusal_is_recorded(self):
        other = self.tmp / "other_classes.csv"
        other.write_text("path_prefix,class,member_index,subfamily_letter,rep,order_index,family,workload_key\nkernel,benign_kernel,,,,,,\n")
        rc, _, err = _run(["run", "--out", str(self.out), "--root", str(self.out / "synth_root"), "--grid-default", "W8_H4", "--classes", str(other), "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn("refused: inputs/classes.csv exists; edit it or pass --force", err)
        L = _ledger(self.out)
        self.assertEqual(L["commands"][-1]["status"], "refused: inputs/classes.csv exists; edit it or pass --force")
        before = (self.out / "inputs" / "classes.csv").read_text()
        rc, _, _ = _run(["run", "--out", str(self.out), "--root", str(self.out / "synth_root"), "--grid-default", "W8_H4", "--classes", str(other), "--dry-run", "--force"])
        self.assertEqual(rc, 0)
        self.assertNotEqual((self.out / "inputs" / "classes.csv").read_text(), before)
        # absent inputs/classes.csv and no --classes: refused (the file is the author's)
        (self.out / "inputs" / "classes.csv").unlink()
        rc, _, err = _run(["run", "--out", str(self.out), "--root", str(self.out / "synth_root"), "--grid-default", "W8_H4", "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn("refused: no --classes given", err)

    def test_real_run_skip_stale_force_and_status(self):
        argv = ["run", "--out", str(self.out), "--moves", "4-15", "--only-modules", MINE, "--skip-missing-modules", "--null-perm", "20"]
        rc, _, err = _run(argv)
        self.assertEqual(rc, 0, err)
        L = _ledger(self.out)
        done = {c["name"] for c in L["commands"] if c["status"] == "done"}
        for n in ("features at selection apf", "features at selection combined", "figures fig2_three_floors,fig_level_map", "figures fig_apf_per_tier",
                  "tables table9_pitfalls", "figures fig4_fused_plane_tiers", "tables table7,table8,table11,levels", "tables table_ladder", "figures fig6_ladder",
                  "tables (all)", "figures (all)", "latex skeleton p3", "tables manifest", "tripwire check"):
            self.assertIn(n, done)
        # the internal feature step wrote the feature files at the inherited grid point
        fa = json.loads((self.out / "gates" / "detection" / "features_at_selection.json").read_text())
        self.assertEqual(fa["rungs"]["apf"]["verdict"], "done")
        self.assertEqual(fa["rungs"]["apf"]["grid_source"], "default: W8_H4 (no inherited selection)")
        for v in ("raw", "norm"):
            self.assertTrue((self.out / "features" / "apf" / f"W8_H4_{v}.npz").exists())
        self.assertFalse((self.out / "features" / "combined" / "W8_H4_raw.npz").exists())
        self.assertTrue((self.out / "features" / "combined" / "W8_H4_norm.npz").exists())
        z = np.load(self.out / "features" / "content" / "W8_H4_norm.npz")
        self.assertEqual(len(z["feature_names"]), 36)
        self.assertIn("sandbox_member_1__rep00__stage1", set(z["cell_id"].astype(str)))
        # the report exists and the tripwire passed
        self.assertTrue((self.out / "report" / "detection" / "tables" / "table7_detection.csv").exists())
        self.assertTrue((self.out / "report" / "detection" / "paper3_skeleton.tex").exists())
        self.assertTrue((self.out / "report" / "detection" / "manifest.json").exists())
        tw = json.loads((self.out / "gates" / "detection" / "tripwire_check.json").read_text())
        self.assertEqual(tw["verdict"], "pass")
        man = json.loads((self.out / "report" / "detection" / "manifest.json").read_text())
        self.assertIsNotNone(man["driver_detection_state"])
        filtered = [c for c in L["commands"] if c["status"].startswith("not run: module")]
        self.assertTrue(filtered)
        # second run: skipped
        rc, _, _ = _run(["run", "--out", str(self.out), "--moves", "14,15", "--only-modules", MINE, "--skip-missing-modules"])
        self.assertEqual(rc, 0)
        L = _ledger(self.out)
        last = [c for c in L["commands"] if c["move"] == 14 and c["name"] == "tables (all)"][-1]
        self.assertTrue(last["status"].startswith("skipped"))
        # a changed class file makes the tables stale (SPEC_DETECTION 6.1)
        with open(self.out / "inputs" / "classes.csv", "a") as fh:
            fh.write("x,idle,,,,,,\n")
        rc, _, _ = _run(["run", "--out", str(self.out), "--moves", "14", "--only-modules", "tables_detection", "--skip-missing-modules"])
        self.assertEqual(rc, 0)
        L = _ledger(self.out)
        last = [c for c in L["commands"] if c["name"] == "tables (all)"][-1]
        self.assertEqual(last["status"], "done")
        self.assertTrue(last["stale"].startswith("stale: classes.csv changed since move 14"))
        # --force: unconditional re-run
        rc, _, _ = _run(["run", "--out", str(self.out), "--moves", "15", "--only-modules", MINE, "--force"])
        L = _ledger(self.out)
        self.assertEqual([c for c in L["commands"] if c["move"] == 15][-1]["status"], "done")
        rc, out, _ = _run(["status", "--out", str(self.out)])
        self.assertEqual(rc, 0)
        self.assertIn("tripwire check", out)

    def test_features_at_selection_without_a_selection(self):
        (self.out / "gates" / "selection.json").unlink()
        verdict, detail = D.features_at_selection(self.out, ["apf"])
        self.assertTrue(verdict.startswith("not run: no selection for apf"))
        fa = json.loads((self.out / "gates" / "detection" / "features_at_selection.json").read_text())
        self.assertIn("apf", fa["rungs"])

    def test_tripwire_refuses_when_a_row_lacks_the_verdict(self):
        from plan11_encoding_ladder import tables_detection as T
        T.table7_detection(self.out)
        T.table11_splits(self.out)
        verdict, detail = D.tripwire_check(self.out, [])
        self.assertEqual(verdict, "pass")
        p = self.out / "report" / "detection" / "tables" / "table7_detection.csv"
        with open(p, newline="") as fh:
            rr = list(csv.DictReader(fh))
            hdr = rr and list(rr[0].keys())
        rr[0]["G-F (i)"] = ""
        rr[1]["G-ANCHOR (ii)"] = "--"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr)
            w.writeheader()
            w.writerows(rr)
        verdict, detail = D.tripwire_check(self.out, [])
        self.assertEqual(verdict, "refused: 2 rows without a tripwire verdict")
        self.assertEqual(len(detail["rows_without"]), 2)
        p.unlink()
        verdict, _ = D.tripwire_check(self.out, [])
        self.assertTrue(verdict.startswith("not run:"))
        tw = json.loads((self.out / "gates" / "detection" / "tripwire_check.json").read_text())
        self.assertEqual(tw["verdict"], verdict)


if __name__ == "__main__":
    unittest.main()
