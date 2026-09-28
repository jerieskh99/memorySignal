#!/usr/bin/env python3
"""tests/test_extract.py -- builder 1's tests of extract.py against synth.py's known answers
(SPEC 2.1 to 2.7, 5.1 truth.json): every extract column on every row (J, J_null, the
persistent-page sums, quantiles and ratios, gaps, the last row's blanks), both persist sides,
the sidecar's counters and refusals, the two-snapshot memory bound (instrumented), the
reader's fallbacks, the cell index, the batch mode and the CLI exit codes.

Run: python3 -m pytest -q plan11_encoding_ladder/tests
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from plan11_encoding_ladder import extract, schema, synth  # noqa: E402

QEMU_DIR = Path(__file__).resolve().parents[2]
PKG_DIR = QEMU_DIR / "plan11_encoding_ladder"


def read_extract(out: Path, cell_id: str) -> list[dict]:
    with open(out / "extract" / cell_id / "extract.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def read_sidecar(out: Path, cell_id: str) -> dict:
    with open(out / "extract" / cell_id / "sidecar.json") as fh:
        return json.load(fh)


def truth_of(cell: Path) -> dict:
    with open(cell / "truth.json") as fh:
        return json.load(fh)


def assert_rows_match_truth(tc: unittest.TestCase, rows: list[dict], per_seq: dict, *, rel: float = 1e-9) -> None:
    """Every column, every row, to 1e-9 relative (the CSV carries 10 significant digits)."""
    tc.assertEqual(list(rows[0].keys()), list(schema.EXTRACT_COLUMNS))
    tc.assertEqual(len(rows), len(per_seq["seq"]))
    for i, row in enumerate(rows):
        for col in schema.EXTRACT_COLUMNS:
            tv = per_seq[col][i]
            ev = row[col]
            if tv is None:
                tc.assertEqual(ev, "", f"row {i} {col}: expected blank, got {ev!r}")
                continue
            tc.assertNotEqual(ev, "", f"row {i} {col}: expected {tv}, got blank")
            if col in schema.EXTRACT_INT_COLUMNS:
                tc.assertEqual(int(ev), int(tv), f"row {i} {col}")
            else:
                a, b = float(ev), float(tv)
                tc.assertLessEqual(abs(a - b), rel * max(1.0, abs(a), abs(b)), f"row {i} {col}: {ev} vs {tv}")


class TestExtractAgainstTruth(unittest.TestCase):
    """SPEC 5.1: `extract.csv` equals `truth.per_seq_channels` on every column and row."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="p11_ext_"))
        cls.root = cls.tmp / "root"
        cls.out = cls.tmp / "out"
        # the re-seed pulse + gaps + floor cell (gemm-like), compressed
        cls.pulse_spec = synth.SynthSpec(name="gemm", seed=42, n_pairs=40, K0=600, k_noise=0.02, churn=0.02,
                                         pulse_period=12, pulse_extra=600, content="double", floor_F=50,
                                         gap_seqs=(5, 6, 20), label="synth")
        cls.pulse_cell = synth.write_cell(cls.pulse_spec, cls.root)
        # a persistent set (churn 0), a floor-like idle cell, a level-matched pair
        cls.persist_cell = synth.write_cell(synth.SynthSpec(name="nbody", seed=42, n_pairs=15, K0=300, churn=0.0,
                                                            floor_churn=0.0, floor_F=20, k_noise=0.0), cls.root)
        cls.idle_cell = synth.write_cell(synth.SynthSpec(name="sleep", seed=42, n_pairs=20, K0=0, content="idle",
                                                         floor_F=150, floor_churn=0.02, label="idle"), cls.root)
        cls.lm_a = synth.write_cell(synth.SynthSpec(name="floyd", seed=42, n_pairs=20, K0=400, content="double",
                                                    churn=0.01, floor_F=30), cls.root)
        cls.lm_b = synth.write_cell(synth.SynthSpec(name="histogram", seed=42, n_pairs=20, K0=400, content="counter",
                                                    churn=0.01, floor_F=30), cls.root)
        cls.shuffled_cell = synth.write_cell(synth.SynthSpec(name="spmm", seed=42, n_pairs=10, K0=150, floor_F=10,
                                                             row_order="random", seq_first=0), cls.root, compress=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, cell: Path, **kw) -> tuple[dict, list[dict], dict]:
        sc = extract.extract_cell(cell, self.out, **kw)
        self.assertEqual(sc["status"], "ok", sc)
        return sc, read_extract(self.out, sc["cell_id"]), truth_of(cell)

    def test_pulse_gaps_floor_cell(self):
        sc, rows, truth = self._run(self.pulse_cell)
        assert_rows_match_truth(self, rows, truth["per_seq_channels"])
        self.assertEqual(sc["cell_id"], "gemm__rep00__synth")
        self.assertEqual(sc["n_pairs"], 40)
        self.assertEqual(sc["n_seq_gaps"], 3)
        self.assertEqual(sc["gap_seqs"], [5, 6, 20])
        self.assertEqual(sc["n_seq_present"], 37)
        # the J series: the dip at the pass boundary, the gap semantics, the last row blank
        J = [r["J"] for r in rows]
        self.assertEqual(J[-1], "")
        self.assertEqual(J[3], "0")            # S_4 against the empty S_5
        self.assertEqual(J[4], "")             # S_5 and S_6 both empty
        self.assertEqual(J[5], "0")            # empty S_6 against S_7
        for b in truth["boundaries"]:
            self.assertLess(float(J[b - 2]), 0.62)
            self.assertGreater(float(J[b - 2]), 0.38)
        # gap rows: K = 0, all-zero sums, blank quantiles, blank per columns, J_null 0
        g = rows[4]
        self.assertEqual(g["K"], "0")
        self.assertEqual((g["ham_sum_all"], g["l0_sum_all"], g["l1_sum_all"]), ("0", "0", "0"))
        self.assertEqual(g["ham_q50_all"], "")
        self.assertEqual(g["ham_sum_per"], "")
        self.assertEqual(g["n_persist"], "0")
        self.assertEqual(g["J_null_inter"], "0")
        self.assertEqual(g["J_null"], "0")
        # the last row keeps its _all block and blanks every pair column
        last = rows[-1]
        self.assertNotEqual(last["K"], "0")
        self.assertNotEqual(last["ham_q50_all"], "")
        for col in ("n_persist", "n_union", "J_null_inter", "J_null", "ham_sum_per", "r_l0_q50_per", "r_haml0_q95_per"):
            self.assertEqual(last[col], "")
        # the independence null is small and J - J_null is near J
        self.assertLess(float(rows[0]["J_null"]), 0.01)
        self.assertGreater(float(rows[0]["J"]), 0.85)

    def test_persist_side_t1(self):
        sc = extract.extract_cell(self.pulse_cell, self.out, persist_side="t+1", cell_id="pulse_t1")
        self.assertEqual(sc["status"], "ok")
        self.assertEqual(sc["persist_side"], "t+1")
        self.assertEqual(sc["params"]["persist_side"], "t+1")
        rows = read_extract(self.out, "pulse_t1")
        truth = truth_of(self.pulse_cell)
        assert_rows_match_truth(self, rows, truth["per_seq_channels_t1"])
        # the two sides agree on the pair columns and differ on the per-channel values
        extract.extract_cell(self.pulse_cell, self.out, persist_side="t", cell_id="pulse_t")
        rows_t = read_extract(self.out, "pulse_t")
        for a, b in zip(rows_t, rows):
            self.assertEqual((a["J"], a["n_persist"], a["n_union"]), (b["J"], b["n_persist"], b["n_union"]))
        self.assertTrue(any(a["l1_sum_per"] != b["l1_sum_per"] for a, b in zip(rows_t, rows) if a["l1_sum_per"]))
        with self.assertRaises(ValueError):
            extract.extract_cell(self.pulse_cell, self.out, persist_side="both")

    def test_persistent_set_cell(self):
        sc, rows, truth = self._run(self.persist_cell)
        assert_rows_match_truth(self, rows, truth["per_seq_channels"])
        for r in rows[:-1]:
            self.assertEqual(r["J"], "1")
            self.assertEqual(r["n_persist"], r["K"])
            self.assertEqual(r["n_union"], r["K"])
            self.assertEqual(r["ham_sum_per"], r["ham_sum_all"])
            self.assertEqual(r["l0_q50_per"], r["l0_q50_all"])
        self.assertEqual(rows[0]["K"], "320")

    def test_idle_floor_cell(self):
        sc, rows, truth = self._run(self.idle_cell)
        assert_rows_match_truth(self, rows, truth["per_seq_channels"])
        self.assertEqual(sc["role"], "idle")
        self.assertEqual(sc["archetype_predicted"], "control")
        self.assertEqual(sc["cell_id"], "idle__rep00__idle")
        self.assertEqual(sc["kernel"], "sleep")
        self.assertTrue(all(abs(int(r["K"]) - 150) <= 3 for r in rows))
        self.assertLess(sc["apf_max"], 0.02)      # C1's re-map threshold would have refused this cell
        for r in rows[:-1]:
            self.assertGreater(float(r["J"]), 0.9)
            self.assertLessEqual(float(r["r_l0_q95_per"]), 4 / 4096)

    def test_level_matched_pair(self):
        sa, ra, ta = self._run(self.lm_a)
        sb, rb, tb = self._run(self.lm_b)
        assert_rows_match_truth(self, ra, ta["per_seq_channels"])
        assert_rows_match_truth(self, rb, tb["per_seq_channels"])
        self.assertEqual(sa["cell_id"], "floyd__rep00__synth")
        self.assertEqual(sb["cell_id"], "histogram__rep00__synth")
        self.assertLess(abs(sa["K_median"] - sb["K_median"]), 25)   # the same level
        r_a = np.median([float(r["r_l1l0_q50_per"]) for r in ra[:-1]])
        r_b = np.median([float(r["r_l1l0_q50_per"]) for r in rb[:-1]])
        self.assertGreater(r_a, 70)         # doubles
        self.assertLess(r_b, 60)            # counters
        l0_a = np.median([float(r["r_l0_q50_per"]) for r in ra[:-1]])
        l0_b = np.median([float(r["r_l0_q50_per"]) for r in rb[:-1]])
        self.assertGreater(l0_a, 100 / 4096)
        self.assertLess(l0_b, 3 / 4096)

    def test_random_row_order_and_seq_first_zero(self):
        sc, rows, truth = self._run(self.shuffled_cell)
        assert_rows_match_truth(self, rows, truth["per_seq_channels"])
        self.assertEqual(sc["seq_first"], 0)
        self.assertEqual(sc["seq_last"], 9)
        self.assertEqual(sc["n_pairs"], 10)
        self.assertEqual(rows[0]["seq"], "0")
        self.assertTrue(sc["traj_file"].endswith(".csv"))

    def test_sidecar_fields(self):
        sc = read_sidecar(self.out, "gemm__rep00__synth") if (self.out / "extract" / "gemm__rep00__synth" / "sidecar.json").exists() \
            else extract.extract_cell(self.pulse_cell, self.out)
        truth = truth_of(self.pulse_cell)
        self.assertEqual(sc["schema"], "plan11.extract.v1")
        self.assertEqual(sc["extractor_version"], "0.1.0")
        for key in ("params", "citation", "source_sha256", "started_at", "finished_at", "elapsed_s"):
            self.assertIn(key, sc)
        self.assertEqual(sc["N"], 262144)
        self.assertEqual(sc["page_size"], 4096)
        self.assertEqual(sc["bits_per_page"], 32768)
        self.assertEqual(sc["duration_s_declared"], 600)
        self.assertEqual(sc["quantiles"], [0.05, 0.25, 0.5, 0.75, 0.95])
        self.assertEqual(sc["persist_side"], "t")
        self.assertEqual(sc["header_ncols"], 66)
        self.assertEqual(sc["header_sha256"], hashlib.sha256(schema.TRAJ_HEADER_LINE.encode()).hexdigest())
        self.assertEqual(sc["columns_used"], {"seq": 0, "page_index": 1, "hamming": 2, "l0": 4, "l1": 5})
        self.assertEqual(sc["n_rows_in"], truth["n_rows"])
        self.assertEqual(sc["n_rows_skipped"], 0)
        self.assertEqual(sc["n_rows_dup_page"], 0)
        self.assertEqual(sc["n_rows_zero_hamming"], 0)
        self.assertEqual(sc["n_rows_zero_l0"], 0)
        self.assertEqual((sc["seq_first"], sc["seq_last"], sc["n_pairs"]), (1, 40, 40))
        self.assertAlmostEqual(sc["dt_est_s"], 600 / 40)
        self.assertEqual(sc["dt_bracket_s"], [0.5, 0.644])
        self.assertEqual(sc["K_max"], max(truth["K"]))
        self.assertAlmostEqual(sc["K_median"], float(np.median(truth["K"])))
        self.assertAlmostEqual(sc["apf_max"], max(truth["K"]) / 262144)
        self.assertIsNone(sc["failed_count"])
        self.assertEqual(sc["failed_count_source"], "not recorded")
        self.assertEqual(sc["status"], "ok")
        self.assertEqual(sc["seed"], 42)
        self.assertEqual(sc["rep"], 0)
        self.assertEqual(sc["rep_dir"], 1)
        self.assertEqual(sc["rep_source"], "rep_dir - 1 (no cells.csv rep given)")
        self.assertEqual(sc["label"], "synth")
        self.assertEqual(sc["campaign"], "synth")
        self.assertEqual(sc["archetype_predicted"], "WORKING-SET")
        traj = self.pulse_cell / sc["traj_file"]
        self.assertEqual(sc["source_bytes"], traj.stat().st_size)
        self.assertEqual(sc["source_sha256"], hashlib.sha256(traj.read_bytes()).hexdigest())
        self.assertEqual(sc["params"]["inputs_sha256"][sc["traj_file"]], sc["source_sha256"])
        self.assertIn("K2 Sec. 5 move 1", sc["citation"])

    def test_failed_count_inputs(self):
        sc = extract.extract_cell(self.persist_cell, self.out, failed_count=2, cell_id="fc2")
        self.assertEqual(sc["failed_count"], 2)
        self.assertEqual(sc["failed_count_source"], "--failed-count")
        fd = self.tmp / "failed"
        fd.mkdir(exist_ok=True)
        for i in range(3):
            (fd / f"job{i}.json").write_text("{}")
        sc = extract.extract_cell(self.persist_cell, self.out, failed_dir=fd, cell_id="fc3")
        self.assertEqual(sc["failed_count"], 3)
        self.assertTrue(sc["failed_count_source"].startswith("--failed-dir"))
        sc = extract.extract_cell(self.persist_cell, self.out, failed_count=0, failed_count_source="inputs/failed_counts.csv", cell_id="fc0")
        self.assertEqual((sc["failed_count"], sc["failed_count_source"]), (0, "inputs/failed_counts.csv"))
        with self.assertRaises(FileNotFoundError):
            extract.extract_cell(self.persist_cell, self.out, failed_dir=self.tmp / "nope", cell_id="fcx")

    def test_role_and_id_overrides(self):
        sc = extract.extract_cell(self.lm_a, self.out, role="idle", cell_id="forced_idle", rep=5)
        self.assertEqual((sc["role"], sc["archetype_predicted"], sc["cell_id"], sc["rep"], sc["rep_source"]),
                         ("idle", "control", "forced_idle", 5, "cells.csv"))
        self.assertTrue((self.out / "extract" / "forced_idle" / "extract.csv").is_file())
        sc = extract.extract_cell(self.lm_a, self.out, archetype_predicted="SCATTER", cell_id="forced_arch")
        self.assertEqual(sc["archetype_predicted"], "SCATTER")

    def test_trajectory_file_path_accepted(self):
        traj = next(p for p in self.persist_cell.glob(schema.TRAJ_GLOB))
        sc = extract.extract_cell(traj, self.out, cell_id="by_file")
        self.assertEqual(sc["status"], "ok")
        self.assertEqual(sc["kernel"], "nbody")


class TestMemoryBound(unittest.TestCase):
    """SPEC 2.4: at most two snapshots of page indices are alive at any moment (instrumented)."""

    def test_never_more_than_two_snapshots(self):
        tmp = Path(tempfile.mkdtemp(prefix="p11_mem_"))
        try:
            spec = synth.SynthSpec(name="gemm", seed=1, n_pairs=25, K0=200, floor_F=10, pulse_period=8,
                                   pulse_extra=100, gap_seqs=(3, 4, 9))
            cell = synth.write_cell(spec, tmp / "root", keep_sets=False)
            seen: list[int] = []
            pairs: list[tuple[int, object]] = []

            def hook(prev, cur):
                seen.append(len(extract._LIVE_SNAPSHOTS))
                pairs.append((prev.seq, None if cur is None else cur.seq))
                # prev and cur are the only live snapshots, and they are the adjacent seqs
                live = sorted(s.seq for s in extract._LIVE_SNAPSHOTS)
                self.assertLessEqual(len(live), 2)
                self.assertIn(prev.seq, live)
                if cur is not None:
                    self.assertEqual(live, [prev.seq, cur.seq])

            extract.EMIT_HOOK = hook
            try:
                sc = extract.extract_cell(cell, tmp / "out")
            finally:
                extract.EMIT_HOOK = None
            self.assertEqual(sc["status"], "ok")
            self.assertEqual(len(seen), 25)                 # one emit per seq, gaps included
            self.assertEqual(max(seen), 2)                  # the bound is reached (a real count)
            self.assertEqual(seen[-1], 1)                   # the last row is emitted against None
            self.assertEqual([p for p, _ in pairs], list(range(1, 26)))
            self.assertEqual([c for _, c in pairs][:-1], list(range(2, 26)))
            self.assertIsNone(pairs[-1][1])
            self.assertEqual(len(extract._LIVE_SNAPSHOTS), 0)   # nothing survives the pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _write_plain_cell(root: Path, name: str, lines: list[str], header: str = schema.TRAJ_HEADER_LINE,
                      sig: str = "--seed_42_--duration_5", label: str = "synth") -> Path:
    cell = root / "kernel" / f"kernel_{name}_v2" / sig / f"rep001__{label}"
    cell.mkdir(parents=True, exist_ok=True)
    body = "\n".join([header] + lines) + ("\n" if lines else "\n")
    (cell / f"run_matrix_test1_kernel_{name}_v2.npy.substrate_trajectory.csv").write_text(body)
    return cell


def _row(seq: int, page: int, ham: int, l0: int, l1: int) -> str:
    return f"{seq},{page},{ham},0,{l0},{l1},0,0,0" + ",0" * 57


class TestRefusalsAndCounters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="p11_ref_"))
        cls.root = cls.tmp / "root"
        cls.out = cls.tmp / "out"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_seq_not_monotone_refused(self):
        spec = synth.SynthSpec(name="gemm", seed=3, n_pairs=8, K0=40, floor_F=0, corrupt="seq_reverse")
        cell = synth.write_cell(spec, self.root / "corrupt", keep_sets=False)
        sc = extract.extract_cell(cell, self.out / "corrupt")
        self.assertTrue(sc["status"].startswith("refused: seq not monotone at row "), sc["status"])
        n = int(sc["status"].rsplit(" ", 1)[1])
        self.assertGreater(n, 1)
        cdir = self.out / "corrupt" / "extract" / sc["cell_id"]
        self.assertTrue((cdir / "sidecar.json").is_file())
        self.assertFalse((cdir / "extract.csv").exists())
        self.assertFalse((cdir / "extract.csv.tmp").exists())
        self.assertIsNone(sc["n_pairs"])
        # the CLI treats a written refusal as success (exit 0)
        rc = extract.main(["cell", "--cell-dir", str(cell), "--out", str(self.out / "corrupt_cli")])
        self.assertEqual(rc, 0)

    def test_trajectory_count_refused(self):
        cell = _write_plain_cell(self.root / "two", "gemm", [_row(1, 5, 3, 1, 4)])
        (cell / "run_matrix_test2_kernel_gemm_v2.npy.substrate_trajectory.csv").write_text(schema.TRAJ_HEADER_LINE + "\n")
        sc = extract.extract_cell(cell, self.out / "two")
        self.assertEqual(sc["status"], "refused: trajectory file count != 1")
        self.assertIsNone(sc["traj_file"])
        empty = self.root / "none" / "kernel" / "kernel_gemm_v2" / "--seed_42" / "rep001__x"
        empty.mkdir(parents=True)
        sc = extract.extract_cell(empty, self.out / "none")
        self.assertEqual(sc["status"], "refused: trajectory file count != 1")
        self.assertTrue((self.out / "none" / "extract" / sc["cell_id"] / "sidecar.json").is_file())

    def test_header_and_empty_refusals(self):
        bad_header = "seq,page_index,hamming,cosine,l1"
        cell = _write_plain_cell(self.root / "hdr", "gemm", ["1,5,3,0,4"], header=bad_header)
        sc = extract.extract_cell(cell, self.out / "hdr")
        self.assertEqual(sc["status"], "refused: header lacks column l0")
        cell = _write_plain_cell(self.root / "empty", "gemm", [])
        sc = extract.extract_cell(cell, self.out / "empty")
        self.assertEqual(sc["status"], "refused: no data rows")
        cell = _write_plain_cell(self.root / "nohdr", "gemm", [], header="")
        (cell / "run_matrix_test1_kernel_gemm_v2.npy.substrate_trajectory.csv").write_text("")
        sc = extract.extract_cell(cell, self.out / "nohdr")
        self.assertEqual(sc["status"], "refused: empty trajectory file")

    def test_duplicate_zero_and_skipped_rows(self):
        lines = [
            _row(1, 10, 3, 1, 4), _row(1, 20, 5, 2, 9), _row(1, 20, 99, 99, 99),     # duplicate page 20: first kept
            _row(1, 30, 0, 0, 0),                                                     # hamming 0 and l0 0: kept in S_t, counted
            "1,abc,1,0,1,1,0,0,0" + ",0" * 57,                                        # parse failure: skipped
            "1,40,1,0,1",                                                             # short row: skipped
            _row(2, 10, 3, 1, 4), _row(2, 20, 4, 2, 6), _row(2, 30, 2, 1, 1),
        ]
        cell = _write_plain_cell(self.root / "dups", "gemm", lines)
        sc = extract.extract_cell(cell, self.out / "dups")
        self.assertEqual(sc["status"], "ok")
        self.assertEqual(sc["n_rows_in"], 9)
        self.assertEqual(sc["n_rows_skipped"], 2)
        self.assertEqual(sc["n_rows_dup_page"], 1)
        self.assertEqual(sc["n_rows_zero_hamming"], 1)
        self.assertEqual(sc["n_rows_zero_l0"], 1)
        rows = read_extract(self.out / "dups", sc["cell_id"])
        r1, r2 = rows
        self.assertEqual(r1["K"], "3")                       # pages 10, 20, 30 (the zero-hamming row counts)
        self.assertEqual(r1["ham_sum_all"], "8")             # 3 + 5 + 0 (the duplicate's 99 dropped)
        self.assertEqual(r1["l1_sum_all"], "13")
        self.assertEqual(r1["n_persist"], "3")
        self.assertEqual(r1["J"], "1")
        self.assertEqual(r1["ham_sum_per"], "8")
        # the ratio quantiles exclude the l0 = 0 row (two rows remain: l1/l0 = 4 and 4.5)
        self.assertEqual(r1["r_l1l0_q05_per"], format(4 + 0.05 * 0.5, ".10g"))
        self.assertEqual(r1["r_l1l0_q95_per"], format(4 + 0.95 * 0.5, ".10g"))
        self.assertEqual(r1["r_haml0_q50_per"], format((3 + 2.5) / 2, ".10g"))
        # r_l0 keeps all three persistent rows (0, 1, 2 bytes)
        self.assertEqual(r1["r_l0_q50_per"], format(1 / 4096, ".10g"))
        self.assertEqual(r2["K"], "3")
        self.assertEqual(r2["J"], "")

    def test_all_persistent_rows_have_zero_l0(self):
        lines = [_row(1, 10, 0, 0, 0), _row(2, 10, 0, 0, 0), _row(2, 11, 4, 2, 5)]
        cell = _write_plain_cell(self.root / "zero", "gemm", lines)
        sc = extract.extract_cell(cell, self.out / "zero")
        rows = read_extract(self.out / "zero", sc["cell_id"])
        self.assertEqual(rows[0]["n_persist"], "1")
        self.assertEqual(rows[0]["r_l0_q50_per"], "0")
        self.assertEqual(rows[0]["r_l1l0_q50_per"], "")
        self.assertEqual(rows[0]["r_haml0_q50_per"], "")
        self.assertEqual(sc["n_rows_zero_l0"], 2)

    def test_gap_at_file_start_is_not_a_gap(self):
        # seq_first is whatever the file starts with; only interior gaps count
        lines = [_row(3, 1, 1, 1, 1), _row(5, 1, 1, 1, 1)]
        cell = _write_plain_cell(self.root / "start", "gemm", lines)
        sc = extract.extract_cell(cell, self.out / "start")
        self.assertEqual((sc["seq_first"], sc["seq_last"], sc["n_pairs"], sc["n_seq_gaps"], sc["gap_seqs"]), (3, 5, 3, 1, [4]))
        rows = read_extract(self.out / "start", sc["cell_id"])
        self.assertEqual([r["seq"] for r in rows], ["3", "4", "5"])
        self.assertEqual(rows[0]["J"], "0")
        self.assertEqual(rows[1]["J"], "0")
        self.assertEqual(rows[1]["K"], "0")


class TestReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="p11_rd_"))
        cls.lines = [schema.TRAJ_HEADER_LINE] + [_row(1, i, 1, 1, 1) for i in range(5)] + [_row(2, i, 2, 2, 2) for i in range(3)]
        cls.text = "\n".join(cls.lines) + "\n"
        (cls.tmp / "a.csv").write_text(cls.text)
        with gzip.open(cls.tmp / "a.csv.gz", "wt") as fh:
            fh.write(cls.text)
        cls.has_zstd = shutil.which("zstd") is not None
        if cls.has_zstd:
            subprocess.run(["zstd", "-q", "-f", str(cls.tmp / "a.csv"), "-o", str(cls.tmp / "a.csv.zst")], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _orig_open_text(self):
        src = QEMU_DIR / "plan08_b1" / "b1_extract_hamming.py"
        if not src.is_file():
            return None
        spec = importlib.util.spec_from_file_location("b1_extract_hamming_orig", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.open_text

    def test_open_text_matches_original(self):
        orig = self._orig_open_text()
        if orig is None:
            self.skipTest("plan08_b1/b1_extract_hamming.py not importable")
        for name in ("a.csv", "a.csv.gz") + (("a.csv.zst",) if self.has_zstd else ()):
            with extract.open_text(str(self.tmp / name)) as fh:
                mine = fh.read()
            with orig(str(self.tmp / name)) as fh:
                theirs = fh.read()
            self.assertEqual(mine, theirs, name)
            self.assertEqual(mine, self.text, name)

    def test_gz_trajectory_extracts(self):
        cell = self.tmp / "root" / "kernel" / "kernel_gemm_v2" / "--seed_42" / "rep001__x"
        cell.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.tmp / "a.csv.gz", cell / "run_matrix_test1_kernel_gemm_v2.npy.substrate_trajectory.csv.gz")
        sc = extract.extract_cell(cell, self.tmp / "out_gz")
        self.assertEqual(sc["status"], "ok")
        self.assertEqual(sc["n_rows_in"], 8)
        rows = read_extract(self.tmp / "out_gz", sc["cell_id"])
        self.assertEqual([r["K"] for r in rows], ["5", "3"])
        self.assertEqual(rows[0]["n_persist"], "3")
        self.assertEqual(rows[0]["J"], "0.6")

    def test_zst_fallback_message_when_nothing_available(self):
        if not self.has_zstd:
            self.skipTest("no zstd binary to build the fixture")
        real_popen = extract.subprocess.Popen

        def no_binary(*a, **k):
            raise FileNotFoundError("zstd")

        extract.subprocess.Popen = no_binary
        saved = sys.modules.get("zstandard", "<absent>")
        sys.modules["zstandard"] = None   # makes `import zstandard` raise ImportError
        try:
            with self.assertRaises(RuntimeError) as cm:
                with extract.open_text(str(self.tmp / "a.csv.zst")) as fh:
                    fh.read()
            self.assertIn("zstd", str(cm.exception))
            self.assertIn("zstandard", str(cm.exception))
        finally:
            extract.subprocess.Popen = real_popen
            if saved == "<absent>":
                del sys.modules["zstandard"]
            else:
                sys.modules["zstandard"] = saved

    def test_zstandard_module_path(self):
        try:
            import zstandard  # noqa: F401
        except ImportError:
            self.skipTest("zstandard module not installed")
        if not self.has_zstd:
            self.skipTest("no zstd binary to build the fixture")
        real_popen = extract.subprocess.Popen

        def no_binary(*a, **k):
            raise FileNotFoundError("zstd")

        extract.subprocess.Popen = no_binary
        try:
            with extract.open_text(str(self.tmp / "a.csv.zst")) as fh:
                self.assertEqual(fh.read(), self.text)
        finally:
            extract.subprocess.Popen = real_popen

    def test_refusal_mid_file_does_not_mask_the_reason(self):
        if not self.has_zstd:
            self.skipTest("no zstd binary")
        # a long compressed file so zstd is still writing when the reader stops
        big = self.tmp / "big.csv"
        with open(big, "w") as fh:
            fh.write(schema.TRAJ_HEADER_LINE + "\n")
            for s in range(1, 50):
                for p in range(400):
                    fh.write(_row(s, p, 1, 1, 1) + "\n")
            fh.write(_row(2, 1, 1, 1, 1) + "\n")       # seq decreases here
            for s in range(50, 400):
                for p in range(400):
                    fh.write(_row(s, p, 1, 1, 1) + "\n")
        subprocess.run(["zstd", "-q", "-f", str(big), "-o", str(big) + ".zst"], check=True)
        cell = self.tmp / "root2" / "kernel" / "kernel_gemm_v2" / "--seed_42" / "rep001__x"
        cell.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(big) + ".zst", cell / "run_matrix_test1_kernel_gemm_v2.npy.substrate_trajectory.csv.zst")
        sc = extract.extract_cell(cell, self.tmp / "out_big")
        self.assertEqual(sc["status"], "refused: seq not monotone at row 19601")


class TestIndexAndBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="p11_idx_"))
        cls.root = cls.tmp / "root"
        cls.out = cls.tmp / "out"
        cls.manifest = synth.write_corpus(cls.root, n_pairs=6, reps=3, idle=2, keep_sets=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_build_index_on_corpus(self):
        rows = extract.build_index(self.root, self.out / "cells.csv")
        self.assertEqual(len(rows), 12 * 3 + 2)
        self.assertTrue(all(r["status"] == "ok" for r in rows))
        self.assertTrue((self.out / "cells.csv").is_file())
        self.assertTrue((self.out / "cells.index.json").is_file())
        meta = json.load(open(self.out / "cells.index.json"))
        self.assertEqual(meta["schema"], "plan11.cells.v1")
        self.assertEqual(meta["n_ok"], 38)
        self.assertIn("params", meta)
        by_id = {r["cell_id"]: r for r in rows}
        self.assertEqual(len(by_id), 38)
        for ki, (k, arch) in enumerate(schema.KERNELS):
            self.assertEqual(by_id[f"{k}__rep00__synth"]["seed"], 42)
            self.assertEqual(by_id[f"{k}__rep01__synth"]["seed"], 1000 + ki)
            self.assertEqual(by_id[f"{k}__rep02__synth"]["seed"], 2000 + ki)
            self.assertEqual(by_id[f"{k}__rep00__synth"]["archetype_predicted"], arch)
            self.assertEqual(by_id[f"{k}__rep00__synth"]["role"], "kernel")
        self.assertEqual(by_id["idle__rep00__idle"]["role"], "idle")
        self.assertEqual(by_id["idle__rep00__idle"]["archetype_predicted"], "control")
        self.assertEqual(by_id["idle__rep01__idle"]["seed"], 1012)
        self.assertEqual(by_id["idle__rep00__idle"]["kernel"], "sleep")
        with open(self.out / "cells.csv", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        self.assertEqual(list(csv_rows[0].keys()), list(schema.CELLS_COLUMNS))
        self.assertEqual(len(csv_rows), 38)
        self.assertTrue(all(r["traj_file"].startswith("run_matrix_test1_") for r in csv_rows))

    def test_index_refusals_and_overrides(self):
        root = self.tmp / "root_ref"
        synth.write_cell(synth.SynthSpec(name="gemm", seed=42, n_pairs=3, K0=10, floor_F=0), root, keep_sets=False)
        synth.write_cell(synth.SynthSpec(name="gemm", seed=1000, n_pairs=3, K0=10, floor_F=0), root, keep_sets=False)
        synth.write_cell(synth.SynthSpec(name="gemm", seed=1000, n_pairs=3, K0=10, floor_F=0, rep_dir=2), root, keep_sets=False)
        synth.write_cell(synth.SynthSpec(name="foo", seed=42, n_pairs=3, K0=10, floor_F=0), root, keep_sets=False)
        empty = root / "kernel" / "kernel_fft_v2" / "--seed_42_--duration_3" / "rep001__synth"
        empty.mkdir(parents=True)
        (root / "kernel" / "kernel_fft_v2" / "--seed_42_--duration_3" / "notacell").mkdir()
        rows = extract.build_index(root, self.tmp / "out_ref" / "cells.csv")
        by_path = {Path(r["path"]).relative_to(root).as_posix(): r for r in rows}
        self.assertEqual(len(rows), 5)
        self.assertEqual(by_path["kernel/kernel_gemm_v2/--seed_42_--duration_3/rep001__synth"]["status"], "ok")
        self.assertEqual(by_path["kernel/kernel_gemm_v2/--seed_42_--duration_3/rep001__synth"]["rep"], 0)
        dup1 = by_path["kernel/kernel_gemm_v2/--seed_1000_--duration_3/rep001__synth"]
        dup2 = by_path["kernel/kernel_gemm_v2/--seed_1000_--duration_3/rep002__synth"]
        self.assertEqual((dup1["status"], dup2["status"]), ("refused: duplicate seed", "refused: duplicate seed"))
        self.assertEqual({dup1["rep"], dup2["rep"]}, {1, 2})
        self.assertNotEqual(dup1["cell_id"], dup2["cell_id"])
        self.assertEqual(by_path["kernel/kernel_foo_v2/--seed_42_--duration_3/rep001__synth"]["status"], "refused: unknown kernel")
        self.assertEqual(by_path["kernel/kernel_fft_v2/--seed_42_--duration_3/rep001__synth"]["status"], "refused: trajectory file count != 1")
        # role override turns the unknown kernel into an idle control
        ov = self.tmp / "overrides.csv"
        ov.write_text("path,role\nkernel/kernel_foo_v2/--seed_42_--duration_3/rep001__synth,idle\n")
        rows = extract.build_index(root, self.tmp / "out_ref2" / "cells.csv", role_overrides=ov)
        foo = [r for r in rows if r["kernel"] == "foo"][0]
        self.assertEqual((foo["role"], foo["archetype_predicted"], foo["status"], foo["cell_id"]),
                         ("idle", "control", "ok", "idle__rep00__synth"))
        # a custom idle marker
        rows = extract.build_index(root, self.tmp / "out_ref3" / "cells.csv", idle_markers=("foo",))
        self.assertEqual([r for r in rows if r["kernel"] == "foo"][0]["role"], "idle")

    def test_extract_all_sequential_and_parallel(self):
        extract.build_index(self.root, self.out / "cells.csv")
        fc = self.tmp / "failed_counts.csv"
        fc.write_text("cell_id,failed_count,source\ngibbs__rep00__synth,2,test input\n")
        summary = extract.extract_all(self.out / "cells.csv", self.out, jobs=1, only=r"^gibbs|^idle", failed_counts=fc)
        self.assertEqual(summary["n_run"], 5)
        self.assertEqual(summary["status_counts"], {"ok": 5})
        self.assertTrue((self.out / "extract" / "extract_all.json").is_file())
        sc = read_sidecar(self.out, "gibbs__rep00__synth")
        self.assertEqual((sc["failed_count"], sc["failed_count_source"]), (2, "test input"))
        self.assertEqual(sc["rep_source"], "cells.csv")
        self.assertIn("cells.csv", sc["params"]["inputs_sha256"])
        self.assertIn("failed_counts.csv", sc["params"]["inputs_sha256"])
        sc1 = read_sidecar(self.out, "gibbs__rep01__synth")
        self.assertIsNone(sc1["failed_count"])
        self.assertEqual(sc1["rep"], 1)
        # every extract matches its truth
        for cid in ("gibbs__rep00__synth", "idle__rep01__idle"):
            sc = read_sidecar(self.out, cid)
            assert_rows_match_truth(self, read_extract(self.out, cid), truth_of(Path(sc["path"]))["per_seq_channels"])
        # a second run skips the finished cells; --force redoes them
        summary = extract.extract_all(self.out / "cells.csv", self.out, jobs=1, only=r"^gibbs|^idle")
        self.assertEqual(summary["n_run"], 0)
        self.assertEqual(summary["n_skipped"], 5)
        summary = extract.extract_all(self.out / "cells.csv", self.out, jobs=2, only=r"^gibbs|^idle|^lexer", force=True)
        self.assertEqual(summary["n_run"], 8)
        self.assertEqual(summary["status_counts"], {"ok": 8})
        self.assertEqual(summary["params"]["jobs"], 2)
        # cells refused in cells.csv are skipped with the reason recorded
        rows = extract.read_cells_csv(self.out / "cells.csv")
        rows[0]["status"] = "refused: duplicate seed"
        with open(self.out / "cells2.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(schema.CELLS_COLUMNS))
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in schema.CELLS_COLUMNS})
        summary = extract.extract_all(self.out / "cells2.csv", self.out, only=rows[0]["cell_id"], force=True)
        self.assertEqual(summary["n_run"], 0)
        self.assertEqual(summary["skipped"][0]["reason"], "cells.csv status: refused: duplicate seed")

    def test_cli_exit_codes_and_entry_points(self):
        self.assertEqual(extract.main(["cell", "--cell-dir", str(self.tmp / "nope"), "--out", str(self.out)]), 2)
        self.assertEqual(extract.main(["all", "--cells-csv", str(self.tmp / "nope.csv"), "--out", str(self.out)]), 2)
        self.assertEqual(extract.main(["index", "--root", str(self.tmp / "nope"), "--out", str(self.out)]), 2)
        self.assertEqual(extract.main(["all", "--cells-csv", str(self.out / "cells.csv"), "--out", str(self.out),
                                       "--failed-counts", str(self.tmp / "nope.csv")]), 2)
        self.assertEqual(extract.main(["index", "--root", str(self.root), "--out", str(self.tmp / "out_cli")]), 0)
        self.assertTrue((self.tmp / "out_cli" / "cells.csv").is_file())
        cell = Path(self.manifest["cells"][0]["path"])
        self.assertEqual(extract.main(["cell", "--cell-dir", str(cell), "--out", str(self.tmp / "out_cli"), "--failed-count", "0"]), 0)
        cid = extract.derive_cell_id(cell)
        self.assertTrue((self.tmp / "out_cli" / "extract" / cid / "sidecar.json").is_file())
        before = (self.tmp / "out_cli" / "extract" / cid / "sidecar.json").stat().st_mtime_ns
        self.assertEqual(extract.main(["cell", "--cell-dir", str(cell), "--out", str(self.tmp / "out_cli")]), 0)
        self.assertEqual((self.tmp / "out_cli" / "extract" / cid / "sidecar.json").stat().st_mtime_ns, before)  # skipped
        self.assertEqual(extract.main(["cell", "--cell-dir", str(cell), "--out", str(self.tmp / "out_cli"), "--force"]), 0)
        self.assertGreater((self.tmp / "out_cli" / "extract" / cid / "sidecar.json").stat().st_mtime_ns, before)  # redone
        r = subprocess.run([sys.executable, "-m", "plan11_encoding_ladder.extract", "all", "--cells-csv",
                            str(self.tmp / "out_cli" / "cells.csv"), "--out", str(self.tmp / "out_cli"), "--jobs", "2",
                            "--only", "^fft"], cwd=QEMU_DIR, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ran 3 cells", r.stdout)
        r = subprocess.run([sys.executable, str(PKG_DIR / "extract.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([sys.executable, str(PKG_DIR / "extract.py"), "cell", "--cell-dir", str(cell), "--out",
                            str(self.tmp / "out_path"), "--persist-side", "t+1"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(": ok", r.stdout)


if __name__ == "__main__":
    unittest.main()


class TestEpoch2B12Duration(unittest.TestCase):
    def test_extract_at_the_truth_duration_gives_the_median_spacing(self):
        """SPEC_epoch2 B12 (ii): extracting with duration_s = truth["duration_s"] gives dt_est_s = 0.644 inside
        DT_BRACKET_S, the sidecar records the float (77.28, not 77), and the default stays 600."""
        tmp = Path(tempfile.mkdtemp(prefix="p11_extract_e2_"))
        try:
            cell = synth.write_cell(synth.SynthSpec(name="gemm", seed=42, n_pairs=120, K0=500), tmp / "root", compress=False)
            truth = truth_of(cell)
            sc = extract.extract_cell(cell, tmp / "out", duration_s=truth["duration_s"])
            self.assertEqual(sc["duration_s_declared"], 77.28)
            self.assertAlmostEqual(sc["dt_est_s"], 0.644)
            self.assertTrue(schema.DT_BRACKET_S[0] <= sc["dt_est_s"] <= schema.DT_BRACKET_S[1] + 1e-9)
            sc2 = extract.extract_cell(cell, tmp / "out2")
            self.assertEqual(sc2["duration_s_declared"], 600)
            rc = extract.main(["cell", "--cell-dir", str(cell), "--out", str(tmp / "out4"), "--duration-s", "77.28"])
            self.assertEqual(rc, 0)
            self.assertEqual(read_sidecar(tmp / "out4", sc["cell_id"])["duration_s_declared"], 77.28)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
