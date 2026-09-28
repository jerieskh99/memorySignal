#!/usr/bin/env python3
"""Builder 3's tests for run_moves.py (the driver; `driver.py` is its alias). The synthetic
`<out>` tree of report_fixtures.py stands in for builders 1 and 2; the driver's own steps
(tables, figures, skeleton, the grid and G-F checks, the ledger, the skip and staleness rules)
run for real, the other builders' commands are filtered with `--only-modules` and recorded.
Runs under `python3 -m pytest -q` and `python3 -m unittest`."""
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
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from plan11_encoding_ladder import run_moves  # noqa: E402
from plan11_encoding_ladder._report_common import GRID_IDS, RUNGS  # noqa: E402
from report_fixtures import make_out  # noqa: E402

MINE = "tables,figures,latex_skeleton,driver"


def _run(argv):
    so, se = io.StringIO(), io.StringIO()
    with redirect_stdout(so), redirect_stderr(se):
        rc = run_moves.main(argv)
    return rc, so.getvalue(), se.getvalue()


def _ledger(out: Path) -> dict:
    return json.loads((out / "driver_state.json").read_text())


def _ns(**kw) -> argparse.Namespace:
    base = dict(out="/x/out", root="/x/root", cells_csv=None, moves="0-13", assume_failed_zero=False, assume_reason=None,
                n_jobs=1, null_perm=500, null_splits="loko,loro,within_trace", seed_offset=0, force=False, dry_run=False,
                skip_missing_modules=False, only_modules=None, persist_side=None, failed_counts=None, table8_rung="combined",
                piano_cell=None, piano_stride=16, standalone_tex=None, cmd="plan")
    base.update(kw)
    return argparse.Namespace(**base)


class TestPlan(unittest.TestCase):
    def test_parse_moves(self):
        self.assertEqual(run_moves.parse_moves("0-13"), list(range(14)))
        self.assertEqual(run_moves.parse_moves("6,7"), [6, 7])
        self.assertEqual(run_moves.parse_moves("2-4,12"), [2, 3, 4, 12])
        with self.assertRaises(ValueError):
            run_moves.parse_moves("15")

    def test_move_table_carries_the_review_corrections(self):
        P = run_moves.build_plan(_ns(seed_offset=3, n_jobs=4))
        names = [(c["move"], c["name"]) for c in P]
        self.assertIn((2, "gp"), names)                      # al-Kindi 6
        self.assertNotIn((6, "gp"), names)
        self.assertIn((6, "alias"), names)                   # al-Kindi 5, twice
        self.assertIn((7, "alias (again, table6 features)"), names)
        for m, r in ((7, "apf"), (9, "persist"), (10, "content"), (11, "wapf"), (12, "combined")):
            self.assertIn((m, f"gx {r}"), names)             # al-Kindi 8
            self.assertIn((m, f"grid complete {r}"), names)  # al-Farabi 2.2
        for m, r in ((6, "apf"), (9, "persist"), (10, "content"), (11, "wapf"), (12, "combined")):
            first = [c for c in P if c["move"] == m][0]
            self.assertEqual(first["name"], f"features {r} all grid")  # al-Farabi 2.2, first in the stage
            self.assertIn("--all-grid", first["args"])
        m12 = [c["name"] for c in P if c["move"] == 12]
        self.assertLess(m12.index("gf all rungs at the selected points"), m12.index("tables (all)"))  # al-Farabi 2.6
        self.assertEqual([c["name"] for c in P if c["move"] == 13], ["gf check on Table 7"])
        comb = [c for c in P if c["name"] == "features combined all grid"][0]
        self.assertIn("--norm", comb["args"])
        self.assertNotIn("--both", comb["args"])
        grid = [c for c in P if c["name"] == "grid apf"][0]
        self.assertIn("--seed-offset", grid["args"])          # a random command
        self.assertIn("--n-jobs", grid["args"])
        gn = [c for c in P if c["name"] == "gn"][0]
        self.assertNotIn("--seed-offset", gn["args"])         # not a random command
        tmpl = [c for c in P if c["template"]]
        self.assertEqual(len(tmpl), 4)
        # CHECK_1.md B3: G-C runs for the combined rung, after the four it reads
        m3 = [c["name"] for c in P if c["move"] == 3]
        self.assertEqual(m3, ["gc apf", "gc persist", "gc content", "gc wapf", "gc combined"])
        # CHECK_1.md B2: G-L runs again at move 12, after the combined selection and before gf --all-rungs
        self.assertIn("gl (all rungs)", m12)
        self.assertLess(m12.index("gx combined"), m12.index("gl (all rungs)"))
        self.assertLess(m12.index("gl (all rungs)"), m12.index("gf all rungs at the selected points"))
        # CHECK_1.md M6: gdim's matched split runs at the driver's --null-perm
        gdim = [c for c in P if c["name"] == "gdim"][0]
        self.assertIn("--null-perm", gdim["args"])
        # al-Farabi certification 7.1: a rebuilt grid makes the selection stale
        sel = [c for c in P if c["name"] == "select apf"][0]
        for g in GRID_IDS:
            self.assertIn(f"gates/grid/apf/{g}/temporal_per_kernel.csv", sel["inputs"])
        self.assertIn("gates/gk0.csv", sel["inputs"])
        self.assertIn("gates/gc.csv", sel["inputs"])
        # al-Farabi certification cycle 2, 7.1: the admissibility record is an input of every step from move 3 on
        for c in P:
            if c["move"] >= 3 and not c["internal"] and c["module"] != "latex_skeleton" and not c["name"].startswith("alias"):
                self.assertIn("gates/preconditions.json", c["inputs"], c["name"])
        # 7.2: G-ORD's inputs are a superset of the grid's triggers, so a rebuilt grid re-runs G-ORD before select
        for rung in RUNGS:
            g = [c for c in P if c["name"] == f"grid {rung}"][0]
            o_ = [c for c in P if c["name"] == f"gord {rung}"][0]
            self.assertTrue(set(g["inputs"]) <= set(o_["inputs"]), (g["inputs"], o_["inputs"]))
        # CHECK_2.md M9: gdim takes the seed offset; gdim and gm take --n-jobs
        P3 = run_moves.build_plan(_ns(seed_offset=7, n_jobs=3))
        for n in ("gdim", "gm"):
            self.assertIn("--n-jobs", [c for c in P3 if c["name"] == n][0]["args"])
        self.assertIn("--seed-offset", [c for c in P3 if c["name"] == "gdim"][0]["args"])
        # al-Kindi review 1's detection ratio and the runbook's AA A5 text are passed through only when asked
        P2 = run_moves.build_plan(_ns(assume_failed_zero=True, assume_reason="AA A5: any failed job re-runs the whole cell"))
        pre = [c for c in P2 if c["name"] == "preconditions"][0]
        self.assertIn("--assume-failed-zero", pre["args"])

    def test_cli_guards(self):
        rc, _, err = _run(["run", "--out", "/x", "--assume-failed-zero"])
        self.assertEqual(rc, 2)
        self.assertIn("--assume-reason", err)
        rc, _, err = _run(["run", "--out", "/x", "--moves", "0"])
        self.assertEqual(rc, 2)
        self.assertIn("--root", err)
        rc, _, err = _run(["run", "--out", "/x", "--moves", "15"])
        self.assertEqual(rc, 2)
        rc, out, _ = _run(["plan", "--out", "/x", "--root", "/r", "--moves", "2"])
        self.assertEqual(rc, 0)
        self.assertIn("gates_calibration gp", out)


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plan11_driver_"))
        self.out = make_out(self.tmp / "out", n_pairs=40)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_records_every_command(self):
        rc, _, _ = _run(["run", "--out", str(self.out), "--root", str(self.out / "synth_root"), "--dry-run"])
        self.assertEqual(rc, 0)
        L = _ledger(self.out)
        self.assertEqual(L["schema"], "plan11.driver_state.v1")
        st = [c["status"] for c in L["commands"]]
        self.assertTrue(all(s == "dry-run" or s.startswith("kept:") for s in st))
        self.assertEqual(sum(1 for c in L["commands"] if c["move"] == 13), 1)
        self.assertTrue(any(c["status"] == "kept: author input exists" and c["name"] == "pass-table template" for c in L["commands"]))
        self.assertTrue(all("inputs_sha256" in c for c in L["commands"]))

    def test_real_run_skip_stale_and_force(self):
        argv = ["run", "--out", str(self.out), "--moves", "5,7-13", "--only-modules", MINE, "--piano-stride", "4"]
        rc, _, err = _run(argv)
        self.assertEqual(rc, 0, err)
        L = _ledger(self.out)
        done = {c["name"] for c in L["commands"] if c["status"] == "done"}
        for n in ("figures apf_per_kernel,level_matched", "tables table6", "figures fused_plane", "tables (all)",
                  "latex skeleton", "figures (all)", "tables manifest", "gf check on Table 7"):
            self.assertIn(n, done)
        self.assertTrue((self.out / "report" / "tables" / "table7.csv").exists())
        self.assertTrue((self.out / "report" / "paper2_skeleton.tex").exists())
        gf = json.loads((self.out / "gates" / "gf_check.json").read_text())
        self.assertEqual(gf["verdict"], "pass")
        gc = json.loads((self.out / "gates" / "grid_complete.json").read_text())
        self.assertTrue(gc["grid_complete"]["apf"]["verdict"].startswith("refused: grid incomplete"))
        filtered = [c for c in L["commands"] if c["status"].startswith("not run: module")]
        self.assertTrue(filtered)
        # second run: skipped
        rc, _, _ = _run(["run", "--out", str(self.out), "--moves", "12,13", "--only-modules", MINE])
        self.assertEqual(rc, 0)
        L = _ledger(self.out)
        last = [c for c in L["commands"] if c["move"] == 12 and c["name"] == "tables (all)"][-1]
        self.assertTrue(last["status"].startswith("skipped"))
        # an edited author input: stale, re-run (al-Farabi 2.8)
        with open(self.out / "inputs" / "pass_table.csv", "a") as fh:
            fh.write("x,,,\n")
        rc, _, _ = _run(["run", "--out", str(self.out), "--moves", "12", "--only-modules", "latex_skeleton"])
        self.assertEqual(rc, 0)
        L = _ledger(self.out)
        last = [c for c in L["commands"] if c["name"] == "latex skeleton"][-1]
        self.assertEqual(last["status"], "done")
        self.assertTrue(last["stale"].startswith("stale: pass_table.csv changed since move 12"))
        # --force: unconditional re-run
        rc, _, _ = _run(["run", "--out", str(self.out), "--moves", "13", "--only-modules", MINE, "--force"])
        L = _ledger(self.out)
        self.assertEqual([c for c in L["commands"] if c["move"] == 13][-1]["status"], "done")
        rc, out, _ = _run(["status", "--out", str(self.out)])
        self.assertEqual(rc, 0)
        self.assertIn("gf check on Table 7", out)

    def test_missing_module_stops_unless_skip_flag(self):
        with mock.patch.object(run_moves, "_module_path", lambda m: self.tmp / f"absent_{m}.py"):
            rc, _, err = _run(["run", "--out", str(self.out), "--moves", "2"])
            self.assertEqual(rc, 2)
            self.assertIn("missing module", err)
            L = _ledger(self.out)
            self.assertTrue(L["commands"][-1]["status"].startswith("failed: module"))
            rc, _, _ = _run(["run", "--out", str(self.out), "--moves", "2", "--skip-missing-modules"])
            self.assertEqual(rc, 0)
            L = _ledger(self.out)
            self.assertTrue(any(c["status"].startswith("not run: module") for c in L["commands"]))

    def test_grid_check_refuses_the_split_stage(self):
        # when the split module will run, an incomplete grid stops the driver (SPEC 3.5.7; al-Farabi 2.2)
        with mock.patch.object(run_moves, "_module_path", lambda m: _PKG / "tables.py"):
            rc, _, err = _run(["run", "--out", str(self.out), "--moves", "7", "--only-modules", "driver,models"])
        self.assertEqual(rc, 1)
        self.assertIn("grid incomplete for apf", err)
        gc = json.loads((self.out / "gates" / "grid_complete.json").read_text())
        self.assertEqual(len(gc["grid_complete"]["apf"]["missing"]), 39)
        # a complete grid passes the check
        for gid in run_moves.GRID_IDS:
            (self.out / "gates" / "grid" / "apf" / gid).mkdir(parents=True, exist_ok=True)
            (self.out / "gates" / "grid" / "apf" / gid / "temporal_per_kernel.csv").write_text("rung\n")
            (self.out / "features" / "apf").mkdir(parents=True, exist_ok=True)
            for v in ("raw", "norm"):
                (self.out / "features" / "apf" / f"{gid}_{v}.npz").write_bytes(b"")
        verdict, detail = run_moves.grid_check(self.out, "apf")
        self.assertEqual(verdict, "complete")

    def test_gf_check_refuses_when_a_row_lacks_the_verdict(self):
        from plan11_encoding_ladder import tables
        tables.table7(self.out)
        p = self.out / "report" / "tables" / "table7.csv"
        rows = p.read_text().splitlines()
        hdr = rows[0].split(",")
        i = hdr.index("G-F (i)")
        import csv
        with open(p, newline="") as fh:
            rr = list(csv.DictReader(fh))
        rr[0]["G-F (i)"] = ""
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr)
            w.writeheader()
            w.writerows(rr)
        verdict, detail = run_moves.gf_check(self.out)
        self.assertTrue(verdict.startswith("refused: 1 Table 7 rows without a G-F (i) verdict"))
        self.assertEqual(len(detail["rows_without_gf"]), 1)
        p.unlink()
        verdict, _ = run_moves.gf_check(self.out)
        self.assertTrue(verdict.startswith("not run:"))


class TestEpoch2Builder3(unittest.TestCase):
    """Build epoch 2, builder 3 (SPEC_epoch2 sections 4, 5.1 and 6.3 items 4, 5): the C1 flags, the
    keyed staleness of the split stages and G-X, `gdim --null-splits`."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plan11_driver_e2_"))
        self.out = make_out(self.tmp / "out", n_pairs=40)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_move_table_epoch2(self):
        P = run_moves.build_plan(_ns())
        pre = [c for c in P if c["name"] == "preconditions"][0]
        for flag in ("--c1-rule", "--c1-abs-fraction", "--c1-idle-percentile", "--c1-activity-min"):
            self.assertNotIn(flag, pre["args"])                 # the default adds nothing: the command's own default is auto
        P2 = run_moves.build_plan(_ns(c1_rule="absolute", c1_abs_fraction=0.002, c1_idle_percentile=90.0, c1_activity_min=0.0))
        pre = [c for c in P2 if c["name"] == "preconditions"][0]
        a = pre["args"]
        self.assertEqual(a[a.index("--c1-rule") + 1], "absolute")
        self.assertEqual(a[a.index("--c1-abs-fraction") + 1], "0.002")
        self.assertEqual(a[a.index("--c1-idle-percentile") + 1], "90.0")
        self.assertEqual(a[a.index("--c1-activity-min") + 1], "0.0")
        # the split stages and G-X declare the rung's own selection entry (SPEC_epoch2 5.1)
        for m, r in ((7, "apf"), (9, "persist"), (10, "content"), (11, "wapf"), (12, "combined")):
            sp = [c for c in P if c["move"] == m and c["name"] == f"splits {r}"][0]
            gx = [c for c in P if c["move"] == m and c["name"] == f"gx {r}"][0]
            self.assertIn(f"json:gates/selection.json:{r}", sp["inputs"])
            self.assertIn(f"json:gates/selection.json:{r}", gx["inputs"])
            self.assertNotIn("gates/selection.json", sp["inputs"])
            self.assertNotIn("gates/selection.json", gx["inputs"])
            self.assertIn("gates/preconditions.json", sp["inputs"])
        # every other consumer of the selection keeps the whole file
        for n in ("gl", "gl (all rungs)", "gdim", "gm", "variance", "cluster", "tables (all)", "tables table6", "gf all rungs at the selected points"):
            c = [c for c in P if c["name"] == n][0]
            self.assertIn("gates/selection.json", c["inputs"], n)
        # gdim carries the driver's --null-splits (SPEC_epoch2 3.5.1)
        gdim = [c for c in run_moves.build_plan(_ns(null_splits="loko")) if c["name"] == "gdim"][0]
        self.assertEqual(gdim["args"][gdim["args"].index("--null-splits") + 1], "loko")
        # the CLI accepts the flags
        rc, out, _ = _run(["plan", "--out", "/x", "--root", "/r", "--moves", "2", "--c1-rule", "idle_floor"])
        self.assertEqual(rc, 0)
        self.assertIn("--c1-rule idle_floor", out)

    def test_input_hash_forms(self):
        out = self.out
        sel_p = out / "gates" / "selection.json"
        full = json.loads(sel_p.read_text())
        h_apf = run_moves._input_hash(out, "json:gates/selection.json:apf")
        h_per = run_moves._input_hash(out, "json:gates/selection.json:persist")
        self.assertNotEqual(h_apf, h_per)
        self.assertEqual(h_apf, run_moves._input_hash(out, "json:gates/selection.json:apf"))
        # another rung's entry added or changed: apf's hash does not move; apf's own entry changed: it does
        doc = json.loads(json.dumps(full)); doc["persist"]["grid_id"] = "W64_H64"; doc["new_rung"] = {"grid_id": "W8_H4"}
        sel_p.write_text(json.dumps(doc))
        self.assertEqual(h_apf, run_moves._input_hash(out, "json:gates/selection.json:apf"))
        self.assertNotEqual(h_per, run_moves._input_hash(out, "json:gates/selection.json:persist"))
        doc["apf"]["grid_id"] = "W64_H64"; sel_p.write_text(json.dumps(doc))
        self.assertNotEqual(h_apf, run_moves._input_hash(out, "json:gates/selection.json:apf"))
        # the per-rung params and inputs_sha256_<rung> are part of the key
        doc["apf"]["grid_id"] = full["apf"]["grid_id"]; doc.setdefault("params", {})["inputs_sha256_apf"] = {"x": "y"}
        sel_p.write_text(json.dumps(doc))
        self.assertNotEqual(h_apf, run_moves._input_hash(out, "json:gates/selection.json:apf"))
        sel_p.write_text(json.dumps(full))
        self.assertEqual(h_apf, run_moves._input_hash(out, "json:gates/selection.json:apf"))
        # absent file and missing key
        self.assertEqual(run_moves._input_hash(out, "json:gates/nothing.json:apf"), "absent")
        self.assertEqual(run_moves._input_hash(out, "csv:gates/nothing.csv:rung=apf"), "absent")
        self.assertEqual(run_moves._input_hash(out, "inputs/not_there.csv"), "absent")
        # the csv form: an apf row changed moves the hash, a persist row appended does not
        gx_p = out / "gates" / "gx.csv"
        text = gx_p.read_text()
        h_gx = run_moves._input_hash(out, "csv:gates/gx.csv:rung=apf")
        with open(gx_p, "a") as fh:
            fh.write("persist,0.1,0.2,3,pooling stands,confound: partial\n")
        self.assertEqual(h_gx, run_moves._input_hash(out, "csv:gates/gx.csv:rung=apf"))
        rows = text.splitlines()
        rows = [rows[0]] + [("apf,0.999" + r[len("apf,0.3"):] if r.startswith("apf,") else r) for r in rows[1:]]
        gx_p.write_text("\n".join(rows) + "\n")
        self.assertNotEqual(h_gx, run_moves._input_hash(out, "csv:gates/gx.csv:rung=apf"))
        gx_p.write_text(text)
        # the ledger keys are the spec strings
        d = run_moves._inputs_sha256(out, ["cells.csv", "json:gates/selection.json:apf"])
        self.assertEqual(set(d), {"cells.csv", "json:gates/selection.json:apf"})
        self.assertEqual(run_moves._stale_part("json:gates/selection.json:apf"), "selection.json[apf]")
        self.assertEqual(run_moves._stale_part("csv:gates/g3_flags.csv:rung=apf"), "g3_flags.csv[rung=apf]")
        self.assertEqual(run_moves._stale_part("inputs/pass_table.csv"), "pass_table.csv")

    def test_unchanged_resume_runs_zero_split_stages(self):
        out = self.out
        sel_p = out / "gates" / "selection.json"
        full = json.loads(sel_p.read_text())
        # the state after move 7 of a fresh run: selection.json holds apf only
        only_apf = {k: v for k, v in full.items() if k in ("schema", "params", "citation", "apf")}
        sel_p.write_text(json.dumps(only_apf))
        plan = run_moves.build_plan(_ns(out=str(out), root=str(out / "synth_root")))
        by_name = {c["name"]: c for c in plan}
        ledger = run_moves.load_ledger(out)

        def seed(name):
            c = by_name[name]
            ledger["commands"].append({"key": run_moves._cmd_key(c), "move": c["move"], "name": name, "module": c["module"], "sub": c["sub"],
                                       "status": "done", "exit_code": 0, "started_at": "t0", "finished_at": "t1",
                                       "argv": [sys.executable, "-m", f"plan11_encoding_ladder.{c['module']}"] + ([c["sub"]] if c["sub"] else []) + c["args"],
                                       "inputs_sha256": run_moves._inputs_sha256(out, c["inputs"])})
        seed("splits apf"); seed("gx apf")
        # the four other rungs' selections arrive (moves 9 to 12); persist's split stage ran once its selection existed
        sel_p.write_text(json.dumps(full))
        seed("splits persist")
        run_moves.save_ledger(out, ledger)
        # a resume of move 7 runs no split stage
        rc, _, err = _run(["run", "--out", str(out), "--moves", "7", "--only-modules", "driver"])
        self.assertEqual(rc, 0, err)
        L = _ledger(out)
        for name in ("splits apf", "gx apf"):
            last = [c for c in L["commands"] if c["name"] == name][-1]
            self.assertEqual(last["status"], "skipped: outputs exist and inputs unchanged", name)
        # a --dry-run of moves 7 to 13 after the same state records no `stale` on the split stages of apf
        rc, _, _ = _run(["run", "--out", str(out), "--moves", "7", "--dry-run"])
        L = _ledger(out)
        self.assertEqual([c for c in L["commands"] if c["name"] == "splits apf"][-1]["status"], "skipped: outputs exist and inputs unchanged")
        # a changed cost flag (--n-jobs) is not a changed argument; a changed --null-perm is (al-Farabi review 6.3)
        rc, _, _ = _run(["run", "--out", str(out), "--moves", "7", "--only-modules", "driver", "--n-jobs", "3"])
        L = _ledger(out)
        self.assertEqual([c for c in L["commands"] if c["name"] == "splits apf"][-1]["status"], "skipped: outputs exist and inputs unchanged")
        rc, _, _ = _run(["run", "--out", str(out), "--moves", "7", "--only-modules", "driver", "--null-perm", "7"])
        L = _ledger(out)
        last = [c for c in L["commands"] if c["name"] == "splits apf"][-1]
        self.assertTrue(last.get("stale", "").startswith("stale: arguments changed since move 7"), last.get("stale"))
        self.assertEqual(run_moves._argv_signature(["py", "-m", "x", "--n-jobs", "4", "--null-perm", "5", "--jobs", "2"]), ["-m", "x", "--null-perm", "5"])
        # apf's own entry changes: splits apf is stale and says which part; the seeded persist record stays skipped
        doc = json.loads(sel_p.read_text()); doc["apf"]["grid_id"] = "W64_H64"; sel_p.write_text(json.dumps(doc))
        rc, _, err = _run(["run", "--out", str(out), "--moves", "7,9", "--only-modules", "driver"])
        self.assertEqual(rc, 0, err)
        L = _ledger(out)
        last = [c for c in L["commands"] if c["name"] == "splits apf"][-1]
        self.assertTrue(last.get("stale", "").startswith("stale: selection.json[apf] changed since move 7"), last.get("stale"))
        last_p = [c for c in L["commands"] if c["name"] == "splits persist"][-1]
        self.assertEqual(last_p["status"], "skipped: outputs exist and inputs unchanged")
        sel_p.write_text(json.dumps(full))


if __name__ == "__main__":
    unittest.main()
