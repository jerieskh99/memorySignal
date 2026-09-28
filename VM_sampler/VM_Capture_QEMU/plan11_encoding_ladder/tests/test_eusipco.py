#!/usr/bin/env python3
"""Builder 2 (eusipco), build epoch 2: tests for tables_eusipco.py, latex_skeleton_eusipco.py,
the p2.bib block and the runbook section. Fixed fixture files live under tests/fixtures_eusipco/
(both comparator-row layouts of Table 7, the expected tables, gp.csv, alias.csv, selection.json
and the fixed arrays the feature files and LORO split files are written from). Synthetic data
only; no server. Runs under `python3 -m pytest -q tests`."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

import csv
import hashlib
import json
import re
import shlex
import shutil
import tempfile
import unittest

import numpy as np

from plan11_encoding_ladder import latex_skeleton, latex_skeleton_eusipco as LSE, tables_eusipco as TE  # noqa: E402
from plan11_encoding_ladder._report_common import write_csv, write_json  # noqa: E402

FIX = _HERE / "fixtures_eusipco"
BIB = _PKG.parent.parent.parent.parent / "apf_paper" / "p2.bib"   # mem_sig/../apf_paper
RUNBOOK = _PKG / "RUNBOOK.md"

# p2.bib before the epoch-2 block was appended (2026-09-17): its length and sha256, recorded at
# the time of appending. The test asserts the prefix is byte-identical.
BIB_PREFIX_LEN = 41179
BIB_PREFIX_SHA256 = "b3060ccefd626f3c07202be9e92cc67e3394eab9dbe04dadffc06ab0bc4b4756"
BIB_AUTHOR_BLOCK_MARKER = "% I. Author's own papers and released record"
BIB_AUTHOR_KEYS = ("khoury2027eusipco", "khoury2028encoding", "khoury2027dataset")
BIB_NEW_KEYS = ("dhodapkar2003comparing", "dhodapkar2002managing", "akoush2010predicting", "qemu_calc_dirty_rate",
                "bitchebe2020pml", "ferreira2011libhashckpt", "gioiosa2005tick", "svard2011delta",
                "sancho2004incremental", "ibrahim2011precopy", "nathan2015model")
BIB_ALLOWED_FIELDS = {"author", "title", "booktitle", "editor", "series", "pages", "publisher", "address", "year", "doi",
                      "note", "howpublished", "url", "eprint", "archivePrefix", "primaryClass"}


def _rows(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read(p: Path) -> str:
    return Path(p).read_text(encoding="utf-8")


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plan11_eusipco_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def out_with_table7(self, layout: str = "layout_a", name: str = "out") -> Path:
        out = self.tmp / name
        (out / "report" / "tables").mkdir(parents=True, exist_ok=True)
        for f in (FIX / layout).iterdir():
            shutil.copy(f, out / "report" / "tables" / f.name)
        (out / "cells.csv").write_text("cell_id,kernel,role,status\n", encoding="utf-8")
        return out

    def out_for_table3(self, name: str = "out3", *, with_persist_selection: bool = False,
                       ds_id: str = "dhodapkar_smith", with_files: bool = True) -> Path:
        """The Table 3 fixture: selection.json, gp.csv, alias.csv copied; the feature files and the
        LORO/kernel split stages written from the fixed arrays of features_and_splits.json."""
        out = self.tmp / name
        (out / "gates").mkdir(parents=True, exist_ok=True)
        (out / "report" / "tables").mkdir(parents=True, exist_ok=True)
        sel = json.loads(_read(FIX / "selection.json"))
        if with_persist_selection:
            sel["persist"] = dict(sel["apf"])
        write_json(out / "gates" / "selection.json", sel)
        shutil.copy(FIX / "gp.csv", out / "gates" / "gp.csv")
        shutil.copy(FIX / "alias.csv", out / "gates" / "alias.csv")
        if not with_files:
            return out
        data = json.loads(_read(FIX / "features_and_splits.json"))
        arch = data["archetype"]
        for R, fd in data["features"].items():
            rid = ds_id if R == "dhodapkar_smith" else R
            names = fd["feature_names"]
            X, meta = [], {k: [] for k in ("cell_id", "kernel", "archetype", "campaign", "role", "rep", "win_start", "n_series_cell")}
            for k, rows in fd["rows"].items():
                for r, vec in enumerate(rows):
                    X.append(vec)
                    meta["cell_id"].append(f"{k}__rep{r:02d}__01c")
                    meta["kernel"].append(k)
                    meta["archetype"].append(arch[k])
                    meta["campaign"].append("01c")
                    meta["role"].append("kernel")
                    meta["rep"].append(r)
                    meta["win_start"].append(0)
                    meta["n_series_cell"].append(59)
            p = out / "features" / rid / f"{fd['grid_id']}_norm.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            if R == "dhodapkar_smith":
                names = [n.replace("dhodapkar_smith", rid) for n in names]
            W = -1 if fd["grid_id"] == "Wall_Hall" else int(fd["grid_id"][1:].split("_")[0])
            H = -1 if fd["grid_id"] == "Wall_Hall" else int(fd["grid_id"].split("_H")[1])
            np.savez(p, X=np.array(X, dtype=np.float64), feature_names=np.array(names), cell_id=np.array(meta["cell_id"]),
                     kernel=np.array(meta["kernel"]), archetype=np.array(meta["archetype"]), campaign=np.array(meta["campaign"]),
                     role=np.array(meta["role"]), rep=np.array(meta["rep"], dtype=np.int64),
                     win_start=np.array(meta["win_start"], dtype=np.int64), n_series_cell=np.array(meta["n_series_cell"], dtype=np.int64),
                     W=np.array(W), H=np.array(H), grid_id=np.array(fd["grid_id"]), normalized=np.array(True),
                     head_drop_json=np.array(json.dumps({"all": 0})), n_windows_dropped=np.array(0), wapf_norm=np.array(""))
            sp = data["splits"][R]
            d = out / "gates" / "splits" / rid / fd["grid_id"] / "loro__kernel"
            d.mkdir(parents=True, exist_ok=True)
            pred_cols = ["cell_id", "kernel", "archetype", "campaign", "rep", "fold", "y_true", "y_pred", "n_windows", "vote_fraction", "held_out_campaign"]

            def preds(y_pred_map):
                rows = []
                for k, yps in y_pred_map.items():
                    for r, yp in enumerate(yps):
                        cid = f"{k}__rep{r:02d}__01c"
                        rows.append({"cell_id": cid, "kernel": k, "archetype": arch[k], "campaign": "01c", "rep": r, "fold": cid,
                                     "y_true": k, "y_pred": yp if yp is not None else "not run: no windows",
                                     "n_windows": 0 if yp is None else 1, "vote_fraction": "" if yp is None else 1.0, "held_out_campaign": "01c"})
                return rows
            sc = {"schema": "plan11.scores.v1", "params": {"split": "loro", "labelspace": "kernel", "n_perm": 500, "normalized": True,
                                                            "grid_id": fd["grid_id"]},
                  "citation": "SPEC 4.5", "status": "ok", "accuracy": 0.5, "macro_recall": 0.5, "recall_per_class": {},
                  "majority": 0.2, "b1_g1": "pass", "b1_g1_rank": 500, "null_p95": 0.3, "feature_count": len(names),
                  "dim_status": "full vector", "n_perm": 500, "predictions_file": "predictions.csv"}
            if "quarantine" in sp:
                sc["recall_per_kernel"] = sp["recall_per_kernel_full"]
                sc["with_quarantine"] = {"quarantined_features": [n.replace("dhodapkar_smith", rid) for n in sp["quarantine"]],
                                         "accuracy": 0.5, "recall_per_kernel": sp["recall_per_kernel"],
                                         "predictions_file": "predictions_with_quarantine.csv", "feature_count": len(names) - 1}
                write_csv(d / "predictions.csv", preds({k: [k, k, k] for k in sp["y_pred"]}), pred_cols)   # the full model: all correct
                write_csv(d / "predictions_with_quarantine.csv", preds(sp["y_pred"]), pred_cols)
            else:
                sc["recall_per_kernel"] = sp["recall_per_kernel"]
                write_csv(d / "predictions.csv", preds(sp["y_pred"]), pred_cols)
            write_json(d / "scores.json", sc)
        return out


class TestTable2(_Base):
    def test_table2_from_fixture_layout_a(self):
        """Layout (A): the comparator rows inside table7.csv (`comparator: <display> [<key>]`)."""
        out = self.out_with_table7("layout_a")
        paths = TE.table2(out)
        rows = _rows(paths["csv"])
        exp = _rows(FIX / "expected_table2.csv")
        self.assertEqual(len(rows), 7)
        self.assertEqual(list(rows[0].keys()), TE.EUSIPCO_TABLE2_COLUMNS)
        self.assertEqual(rows, exp)
        tex = _read(paths["tex"])
        self.assertIn("\\cite{savoldi2010uncertainty}", tex)
        self.assertIn("Dhodapkar-Smith 2003~\\cite{dhodapkar2003comparing}", tex)
        self.assertIn("\\caption{}", tex)
        self.assertIn("\\label{tab:p2e_table2}", tex)
        self.assertIn("\\begin{table*}", tex)
        self.assertEqual(latex_skeleton.prose_lines(tex), [])
        md = _read(paths["md"])
        self.assertIn("| Savoldi 2010 | 2 | 0.38 |", md)
        pj = json.loads(_read(out / "report" / "tables" / "eusipco_table2.params.json"))
        self.assertEqual(pj["schema"], "plan11.eusipco_table2.v1")
        self.assertTrue(pj["params"]["comparator_sources"]["savoldi2010uncertainty"].startswith("table7.csv (comparator: Savoldi 2010"))
        self.assertEqual(pj["params"]["epoch"], 2)
        # --include-matched appends the eighth row from table7.csv's `combined (matched)` rows
        rows8 = _rows(TE.table2(out, include_matched=True)["csv"])
        self.assertEqual(len(rows8), 8)
        self.assertEqual(rows8[-1]["reduction"], "combined (matched)")
        self.assertEqual(rows8[-1]["feature count"], "36")
        self.assertEqual(rows8[-1]["LORO accuracy"], "0.91")
        # Law is the IFIP row: added through --comparators
        rows_law = _rows(TE.table2(out, comparators=TE.COMPARATORS_DEFAULT + ("law2010volatile",))["csv"])
        self.assertEqual(rows_law[-1]["reduction"], "Law 2010")
        self.assertEqual(rows_law[-1]["LOKO accuracy"], "0.64")

    def test_table2_missing_comparator_rows_refuse_by_name(self):
        out = self.out_with_table7("layout_a")
        t7 = out / "report" / "tables" / "table7.csv"
        kept = [r for r in _rows(t7) if "dhodapkar2003comparing" not in r["rung"]]
        write_csv(t7, kept, list(kept[0].keys()))
        rows = _rows(TE.table2(out)["csv"])
        ds = [r for r in rows if r["reduction"] == "Dhodapkar-Smith 2003"][0]
        for c in TE.EUSIPCO_TABLE2_COLUMNS[1:]:
            self.assertTrue(ds[c].startswith("not run: table7.csv has no comparator row for dhodapkar2003comparing"), ds[c])
        self.assertEqual([r for r in rows if r["reduction"] == "Savoldi 2010"][0]["LOKO accuracy"], "0.38")
        # a rung missing one split names the split and the label space
        kept2 = [r for r in kept if not (r["rung"] == "wapf" and r["split"] == "LORO" and r["label space"] == "kernel")]
        write_csv(t7, kept2, list(kept2[0].keys()))
        rows = _rows(TE.table2(out)["csv"])
        w = [r for r in rows if r["reduction"] == "wAPF"][0]
        self.assertEqual(w["LORO accuracy"], "not run: table7.csv has no LORO/kernel row for wAPF")
        self.assertEqual(w["LOKO accuracy"], "0.45")

    def test_table2_layout_b_and_variant(self):
        """Layout (B): the comparator rows in table7_comparators.csv, the raw row by default and
        the level-normalized row on request; the source label is recorded."""
        out = self.out_with_table7("layout_b")
        rows = _rows(TE.table2(out)["csv"])
        self.assertEqual(rows, _rows(FIX / "expected_table2.csv"))
        pj = json.loads(_read(out / "report" / "tables" / "eusipco_table2.params.json"))
        self.assertEqual(pj["params"]["comparator_variant"], "as published")
        self.assertEqual(pj["params"]["comparator_sources"]["dhodapkar2003comparing"],
                         "table7_comparators.csv (Dhodapkar-Smith 2003 (as published; delta_th = 0.04, declared default))")
        rows_n = _rows(TE.table2(out, comparator_variant="level-normalized")["csv"])
        sav = [r for r in rows_n if r["reduction"] == "Savoldi 2010"][0]
        self.assertEqual(sav["LOKO accuracy"], "0.31")
        self.assertEqual(sav["LORO accuracy"], "0.41")
        self.assertEqual([r for r in rows_n if r["reduction"] == "APF"][0]["LOKO accuracy"], "0.42")
        # a variant that is not on disk refuses by name; an unknown variant is a ValueError
        (out / "report" / "tables" / "table7_comparators.csv").unlink()
        rows_x = _rows(TE.table2(out)["csv"])
        self.assertTrue(rows_x[5]["LOKO accuracy"].startswith("not run: table7.csv has no comparator row for savoldi2010uncertainty"))
        with self.assertRaises(ValueError):
            TE.comparator_rows(out, "savoldi2010uncertainty", variant="raw")

    def test_cli_exit_codes_and_dispatch(self):
        out = self.tmp / "empty"
        out.mkdir()
        self.assertEqual(TE.main(["--out", str(out)]), 2)                       # table7.csv missing
        self.assertEqual(TE.main(["--out", str(out), "--only", "table3"]), 0)   # Table 3 alone needs no Table 7
        self.assertTrue((out / "report" / "tables" / "eusipco_table3.csv").exists())
        out2 = self.out_with_table7("layout_a", "out2")
        self.assertEqual(TE.main(["--out", str(out2), "--only", "table2", "--include-matched",
                                  "--comparators", "savoldi2010uncertainty"]), 0)
        rows = _rows(out2 / "report" / "tables" / "eusipco_table2.csv")
        self.assertEqual([r["reduction"] for r in rows], ["APF", "wAPF", "content-change", "persistence", "combined",
                                                          "Savoldi 2010", "combined (matched)"])
        with self.assertRaises(ValueError):
            TE.run(out2, ["table9"])


class TestTable3(_Base):
    def test_table3_from_fixture(self):
        out = self.out_for_table3()
        paths = TE.table3(out)
        rows = _rows(paths["csv"])
        exp = _rows(FIX / "expected_table3.csv")
        self.assertEqual(list(rows[0].keys()), TE.EUSIPCO_TABLE3_CSV_COLUMNS)
        self.assertEqual(rows, exp)
        md = _read(paths["md"])
        self.assertIn("| A | floyd, histogram, nbody | none; LORO 0.33; conf 0.67 | 1/2 sep; LORO 0.72; conf 0.25 | "
                      "not run: no selection for persist | 1/3 sep; LORO 0.70; conf 0.33 | "
                      "does not move with the interval; moves with the interval | -- |", md)
        self.assertIn("| B | fft, gemm | none; LORO 0.50; conf 0.50 | 1/2 sep; LORO 0.85; conf 0.17 | "
                      "not run: no selection for persist | none; LORO 0.30; conf 0.67 | -- | "
                      "not resolved; within pass: no |", md)
        tex = _read(paths["tex"])
        self.assertIn("\\label{tab:p2e_table3}", tex)
        self.assertIn("\\begin{table*}", tex)
        self.assertEqual(latex_skeleton.prose_lines(tex), [])
        self.assertEqual(tex.count("&"), 3 * (len(TE.EUSIPCO_TABLE3_COLUMNS) - 1))   # header + two rows
        pj = json.loads(_read(out / "report" / "tables" / "eusipco_table3.params.json"))
        self.assertEqual(pj["schema"], "plan11.eusipco_table3.v1")
        self.assertEqual(pj["params"]["ds_id"], "dhodapkar_smith")
        self.assertEqual(pj["params"]["readings"]["dhodapkar_smith"]["grid_id"], "Wall_Hall")
        self.assertEqual(pj["params"]["readings"]["persist"]["grid_id"], None)
        self.assertEqual(pj["params"]["n_kernels_used"]["A:content"], 3)
        self.assertEqual(pj["params"]["ds_threshold_record"]["source"], "absent")
        self.assertTrue(any(k.endswith("loro__kernel/predictions_with_quarantine.csv") for k in pj["params"]["inputs_sha256"]))
        # with persistence selected, its files are read (set B separates on both features)
        out2 = self.out_for_table3("out3b", with_persist_selection=True)
        rows2 = _rows(TE.table3(out2)["csv"])
        self.assertEqual(rows2[0]["under persistence: separating features"], "0")
        self.assertEqual(rows2[1]["under persistence: separating features"], "2")
        self.assertEqual(rows2[1]["under persistence: LORO kernel recall (set mean)"], "1")
        self.assertEqual(rows2[1]["under persistence: within-set confusion"], "0")
        self.assertEqual(rows2[0]["under persistence: within-set confusion"], "0.667")
        md2 = _read(out2 / "report" / "tables" / "eusipco_table3.md")
        self.assertIn("| 2/2 sep; LORO 1.00; conf 0.00 |", md2)

    def test_table3_two_builder_ids_and_threshold_record(self):
        """The Dhodapkar-Smith reading resolves `cmp_dhodapkar` (layout B) from disk and copies the
        comparator module's threshold record verbatim."""
        out = self.out_for_table3("out_b", ds_id="cmp_dhodapkar")
        write_json(out / "gates" / "comparators" / "dhodapkar.params.json",
                   {"schema": "plan11.comparators.dhodapkar.v1", "params": {"grid": [0.04, 0.1], "default": 0.04,
                                                                             "default_source": "AA 2026-09-17: 0.04 marked as the default"},
                    "citation": "C14 cand. 3"})
        rows = _rows(TE.table3(out)["csv"])
        self.assertEqual(rows[0]["under Dhodapkar-Smith 2003: separating features"], "1")
        self.assertEqual(rows[0]["under Dhodapkar-Smith 2003: within-set confusion"], "0.333")
        pj = json.loads(_read(out / "report" / "tables" / "eusipco_table3.params.json"))
        self.assertEqual(pj["params"]["ds_id"], "cmp_dhodapkar")
        self.assertEqual(pj["params"]["ds_threshold_record"]["default"], 0.04)
        self.assertEqual(pj["params"]["ds_threshold_record"]["source"], "gates/comparators/dhodapkar.params.json")
        # an explicit --ds-id that is absent from disk names its files
        rows3 = _rows(TE.table3(out, ds_id="dhodapkar_smith")["csv"])
        self.assertEqual(rows3[0]["under Dhodapkar-Smith 2003: separating features"], "not run: features/dhodapkar_smith/Wall_Hall_norm.npz missing")
        self.assertEqual(rows3[0]["under Dhodapkar-Smith 2003: LORO kernel recall (set mean)"],
                         "not run: gates/splits/dhodapkar_smith/Wall_Hall/loro__kernel/scores.json missing")

    def test_table3_without_inputs_names_every_missing_file(self):
        out = self.out_for_table3("out_none", with_files=False)
        (out / "gates" / "gp.csv").unlink()
        (out / "gates" / "alias.csv").unlink()
        rows = _rows(TE.table3(out)["csv"])
        self.assertEqual(rows[0]["under APF: separating features"], "not run: features/apf/W16_H8_norm.npz missing")
        self.assertEqual(rows[0]["under APF: LORO kernel recall (set mean)"], "not run: gates/splits/apf/W16_H8/loro__kernel/scores.json missing")
        self.assertEqual(rows[0]["under content-change: feature count"], "not run: features/content/W8_H4_norm.npz missing")
        self.assertEqual(rows[0]["alias check (APF)"], "not run: gates/alias.csv missing")
        self.assertEqual(rows[1]["gemm pass period"], "not run: gates/gp.csv missing")
        self.assertEqual(rows[0]["gemm pass period"], "--")
        md = _read(out / "report" / "tables" / "eusipco_table3.md")
        self.assertIn("not run: features/apf/W16_H8_norm.npz missing; not run: gates/splits/apf/W16_H8/loro__kernel/scores.json missing", md)
        # no selection.json at all: every rung reading says so, the comparator reading names its file
        (out / "gates" / "selection.json").unlink()
        rows = _rows(TE.table3(out)["csv"])
        self.assertEqual(rows[0]["under APF: separating features"], "not run: no selection for apf")
        self.assertEqual(rows[0]["under Dhodapkar-Smith 2003: separating features"], "not run: features/dhodapkar_smith/Wall_Hall_norm.npz missing")

    def test_compact_cell_forms(self):
        self.assertEqual(TE.compact_cell(0, 2, 0.333, 0.6667), "none; LORO 0.33; conf 0.67")
        self.assertEqual(TE.compact_cell(3, 8, 1.0, 0.0), "3/8 sep; LORO 1.00; conf 0.00")
        self.assertEqual(TE.compact_cell(1, 2, None, None), "1/2 sep; LORO --; conf --")
        m = "not run: no selection for persist"
        self.assertEqual(TE.compact_cell(m, m, m, m), m)
        self.assertEqual(TE.compact_cell(2, 3, m, m), f"2/3 sep; {m}")


class TestSkeleton(_Base):
    def test_p2e_skeleton_structure(self):
        tex = LSE.build_p2e_skeleton()
        self.assertTrue(tex.startswith("\\documentclass[conference]{IEEEtran}"))
        self.assertTrue(LSE.build_p2e_skeleton(documentclass="article").startswith("\\documentclass[10pt,twocolumn]{article}"))
        secs = re.findall(r"^\\section\{([^}]+)\}", tex, re.M)
        self.assertEqual(secs, list(LSE.SECTION_TITLES))
        self.assertEqual(len(secs), 5)
        self.assertEqual(tex.count("\\begin{equation}"), 4)
        for lab in ("eq:set", "eq:breadth", "eq:content", "eq:persistence"):
            self.assertEqual(tex.count(f"\\label{{{lab}}}"), 1, lab)
        self.assertIn("\\IfFileExists{tables/eusipco_table2.tex}", tex)
        self.assertIn("\\IfFileExists{tables/eusipco_table3.tex}", tex)
        self.assertIn("\\IfFileExists{figures/fig_fused_plane.pdf}", tex)
        self.assertIn("\\label{fig:p2e_plane}", tex)
        self.assertIn("\\label{fig:p2e_assign}", tex)
        self.assertIn("\\label{tab:p2e_table1}", tex)
        self.assertIn("\\label{tab:p2e_table2}", tex)
        self.assertIn("\\label{tab:p2e_table3}", tex)
        for col in TE.EUSIPCO_TABLE2_COLUMNS + TE.EUSIPCO_TABLE3_COLUMNS:
            self.assertIn(latex_skeleton.latex_escape(col), tex)
        self.assertIn("dataset DOI &  \\\\", tex)      # Table 1's DOI cell blank until the release
        self.assertEqual(latex_skeleton.prose_lines(tex), [])
        self.assertEqual(tex.count("\\begin{document}"), 1)
        self.assertEqual(tex.count("\\end{document}"), 1)
        for env in ("table", "table*", "tabular", "figure", "figure*", "abstract", "equation"):
            self.assertEqual(tex.count(f"\\begin{{{env}}}"), tex.count(f"\\end{{{env}}}"), env)
        self.assertEqual(tex.count("{"), tex.count("}"))
        # the \cite placeholders: 19 to 23, every key in p2.bib, the twelve keys of the plan present
        keys = LSE.cite_keys(tex)
        self.assertGreaterEqual(len(keys), 19)
        self.assertLessEqual(len(keys), 23)
        self.assertEqual(len(keys), len(set(keys)))
        bib_keys = set(re.findall(r"^@\w+\{([^,]+),", _read(BIB), re.M))
        for k in keys:
            self.assertIn(k, bib_keys, k)
        for k in ("law2010volatile", "savoldi2010uncertainty", "oliveri2025inconsistencies", "hirano2022ransomware",
                  "asanovic2006landscape", "purnaye2022bishm", "khoury2026architecture", "vanderkouwe2019sok",
                  "kalibera2013rigorous", "clark2005livemigration", "dhodapkar2003comparing", "dhodapkar2002managing"):
            self.assertIn(k, keys, k)
        # every non-comment line is a command, a tabular row or environment syntax (no body prose)
        body = [ln for ln in tex.splitlines() if ln.strip() and not ln.lstrip().startswith("%")]
        self.assertTrue(all(ln.lstrip().startswith(("\\", "}", "{")) or ln.rstrip().endswith("\\\\") or "&" in ln for ln in body))
        # the substance is in comments: the four gates, the compression map, the venue facts, the comparator picks
        self.assertIn("% the four gates of P2E sec. 0", tex)
        self.assertIn("Table 2 (from Table 7 plus the comparator)", tex)
        self.assertIn("Savoldi 2010 [savoldi2010uncertainty], Dhodapkar-Smith 2003 [dhodapkar2003comparing]", tex)
        self.assertIn("% \\bibliography{p2}", tex)

    def test_write_targets_exist_and_standalone_identical(self):
        out = self.out_with_table7("layout_a")
        TE.run(out)
        w = LSE.write_p2e_skeleton(out, standalone=self.tmp / "p2e.tex")
        tex = _read(w["report"])
        self.assertEqual(_read(self.tmp / "p2e.tex"), tex)
        for t in latex_skeleton.targets(tex):
            if t.startswith("tables/"):
                self.assertTrue((out / "report" / t).exists(), t)
        j = json.loads(_read(out / "report" / "p2e_skeleton.json"))
        self.assertEqual(j["schema"], "plan11.latex_skeleton_eusipco.v1")
        self.assertEqual(j["n_prose_lines"], 0)
        self.assertEqual(j["n_cite"], len(LSE.CITE_PLAN))
        self.assertEqual(j["params"]["epoch"], 2)
        self.assertEqual(LSE.main(["--out", str(out), "--documentclass", "article"]), 0)
        self.assertTrue(_read(out / "report" / "p2e_skeleton.tex").startswith("\\documentclass[10pt,twocolumn]{article}"))

    def test_standalone_copy_in_apf_paper_is_sound(self):
        """apf_paper/p2e_skeleton.tex is HAND-MAINTAINED; check its bones, not its bytes.

        It forked from this builder on 2026-09-17 (its own header says "Originally generated
        by ...") and now carries the author-approved prose, including the EUSIPCO/encoding
        separation. The builder emits a scaffold with zero prose_lines(), so an equality
        assertion against it would demand the prose be deleted. What must stay true is the
        structure the builder established, plus the invariants the prose must not break.
        """
        p = BIB.parent / "p2e_skeleton.tex"
        self.assertTrue(p.exists(), p)
        tex = _read(p)

        # 1. It is the living document, not a regenerated scaffold.
        self.assertGreater(len(LSE.prose_lines(tex)), 20,
                           "p2e_skeleton.tex has lost its prose: was it regenerated over?")

        # 2. Every structural anchor the builder established is still present.
        labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
        for lab in ("sec:repr", "sec:data", "sec:results",
                    "tab:p2e_table1", "tab:p2e_table2", "tab:p2e_table3",
                    "eq:set", "eq:breadth", "eq:content", "eq:persistence"):
            self.assertIn(lab, labels, lab)

        # 3. No dangling cross-reference.
        refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
        self.assertEqual(refs - labels, set(), "dangling \\ref")

        # 4. The note macros are defined (\slot, \pred, \verifyp, \nn).
        macros = set(re.findall(r"\\newcommand\{\\(\w+)\}", tex))
        self.assertTrue({"slot", "pred", "verifyp", "nn"} <= macros, macros)

        # 5. Balanced braces.
        self.assertEqual(tex.count("{"), tex.count("}"), "unbalanced braces")

        # 6. Every cited key exists in p2.bib.
        keys = set(re.findall(r"^@\w+\{([^,]+),", _read(BIB), re.M))
        cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", tex) for k in grp.split(",")}
        self.assertEqual(cited - keys, set(), "cited but not in p2.bib")

        # 7. A-B3 (2026-09-27): no gate label in body text. Comments are exempt by the author's
        #    decision of that date; line 20's build note keeps G-V on purpose.
        body_hits = [ln for ln in tex.splitlines()
                     if not ln.lstrip().startswith("%")
                     and re.search(r"G-[A-Z]|\bG[1-5]\b|\bC[1-8]\b|B1-G", ln)]
        self.assertEqual(body_hits, [], "gate label in body text")


class TestBib(unittest.TestCase):
    def test_bib_block_appended_and_prefix_untouched(self):
        raw = BIB.read_bytes()
        self.assertGreater(len(raw), BIB_PREFIX_LEN)
        self.assertEqual(hashlib.sha256(raw[:BIB_PREFIX_LEN]).hexdigest(), BIB_PREFIX_SHA256)
        txt = raw.decode("utf-8")
        keys = re.findall(r"^@\w+\{([^,]+),", txt, re.M)
        self.assertEqual(len(keys), len(set(keys)), "duplicate bib keys")
        for k in ("law2010volatile", "savoldi2010uncertainty", "clark2005livemigration", "vomel2013evaluation", "khoury2026architecture"):
            self.assertEqual(keys.count(k), 1, k)
        for k in BIB_NEW_KEYS:
            self.assertEqual(keys.count(k), 1, k)
        # every entry of the appended block: balanced braces, a tier comment above it, only the allowed fields
        # The author's own unpublished papers and the released record are a separate, later block
        # (session A3, 2026-09-24 to 27). They carry no tier line and no URL, because none exists;
        # they are checked by their own rules below, not by the epoch 2 literature rules.
        after_prefix = txt[BIB_PREFIX_LEN:]
        split = after_prefix.find(BIB_AUTHOR_BLOCK_MARKER)
        self.assertNotEqual(split, -1, "the author's own block marker is missing from p2.bib")
        block, author_block = after_prefix[:split], after_prefix[split:]

        author_entries = re.findall(r"(?ms)^@(\w+)\{([^,]+),\n(.*?)\n\}$", author_block)
        self.assertEqual(sorted(e[1] for e in author_entries), sorted(BIB_AUTHOR_KEYS))
        for _typ, key, body in author_entries:
            self.assertEqual(body.count("{"), body.count("}"), key)
            fields = re.findall(r"^\s*(\w+)\s*=", body, re.M)
            self.assertIn("title", fields, key)
            self.assertIn("year", fields, key)

        self.assertIn("% H. Build epoch 2 additions (council/14_hunayn_exact_input_comparators.md", block)
        entries = re.findall(r"(?ms)^@(\w+)\{([^,]+),\n(.*?)\n\}$", block)
        self.assertEqual(sorted(e[1] for e in entries), sorted(BIB_NEW_KEYS))
        for _typ, key, body in entries:
            self.assertEqual(body.count("{"), body.count("}"), key)
            fields = re.findall(r"^\s*(\w+)\s*=", body, re.M)
            self.assertTrue(set(fields) <= BIB_ALLOWED_FIELDS, (key, set(fields) - BIB_ALLOWED_FIELDS))
            self.assertIn("title", fields, key)
            self.assertIn("year", fields, key)
            head = block[:block.find(f"{{{key},")]
            tier_lines = [ln for ln in head.rstrip().splitlines()[-12:] if ln.startswith("%")]
            self.assertTrue(any(("tier" in ln or "LEAD" in ln) for ln in tier_lines), key)
            self.assertTrue(any("http" in ln for ln in tier_lines), key)


class TestRunbook(unittest.TestCase):
    def test_runbook_section_and_commands_parse(self):
        txt = _read(RUNBOOK)
        self.assertIn("### The EUSIPCO outputs", txt)
        self.assertIn("tables_eusipco", txt)
        self.assertIn("latex_skeleton_eusipco", txt)
        cmds = [ln.strip() for ln in txt.splitlines()
                if ln.strip().startswith("python3 -m plan11_encoding_ladder.tables_eusipco")
                or ln.strip().startswith("python3 -m plan11_encoding_ladder.latex_skeleton_eusipco")]
        self.assertGreaterEqual(len(cmds), 2)
        for c in cmds:
            argv = shlex.split(c.split("#", 1)[0])[3:]
            mod = TE if "tables_eusipco" in c else LSE
            argv = [a.replace("<out>", "/x") for a in argv]
            ap = mod.build_parser()
            ns, unknown = ap.parse_known_args(argv)
            self.assertEqual(unknown, [], c)


if __name__ == "__main__":
    unittest.main()
