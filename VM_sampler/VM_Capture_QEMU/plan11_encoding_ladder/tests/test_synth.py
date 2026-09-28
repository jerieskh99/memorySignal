#!/usr/bin/env python3
"""tests/test_synth.py -- builder 1's tests of synth.py (SPEC section 5): the header, the
differ's identities on every row, the set dynamics (pulse, gaps, persistence, trend, step,
decay), the content models' expected ratios, the corpus presets and their case switches, and
the CLI.

Run: python3 -m pytest -q plan11_encoding_ladder/tests
"""
from __future__ import annotations

import csv
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from plan11_encoding_ladder import schema, synth  # noqa: E402
from plan11_encoding_ladder.extract import open_text  # noqa: E402

QEMU_DIR = Path(__file__).resolve().parents[2]


def read_rows(cell: Path) -> tuple[list[str], list[list[str]]]:
    traj = next(p for p in cell.iterdir() if "substrate_trajectory" in p.name)
    with open_text(str(traj)) as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = [row for row in r]
    return header, rows


def truth_of(cell: Path) -> dict:
    with open(cell / "truth.json") as fh:
        return json.load(fh)


class TestOneCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="p11_synth_"))
        cls.spec = synth.SynthSpec(name="gemm", seed=42, n_pairs=30, K0=500, k_noise=0.02, churn=0.02,
                                   pulse_period=10, pulse_extra=500, content="double", floor_F=40,
                                   gap_seqs=(4, 5, 17), label="synth")
        cls.cell = synth.write_cell(cls.spec, cls.tmp / "root")
        cls.header, cls.rows = read_rows(cls.cell)
        cls.truth = truth_of(cls.cell)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_layout_and_header(self):
        self.assertEqual(self.cell, self.tmp / "root" / "kernel" / "kernel_gemm_v2" / "--seed_42_--duration_30" / "rep001__synth")
        self.assertEqual(self.header, list(schema.TRAJ_HEADER))
        self.assertEqual(len(self.header), 66)
        files = [p.name for p in self.cell.iterdir()]
        self.assertIn("truth.json", files)
        traj = [f for f in files if "substrate_trajectory" in f]
        self.assertEqual(len(traj), 1)
        self.assertTrue(traj[0].startswith("run_matrix_test1_kernel_gemm_v2.npy.substrate_trajectory.csv"))
        if synth.zstd_available():
            self.assertTrue(traj[0].endswith(".csv.zst"))
        self.assertEqual(self.truth["schema"], "plan11.synth_truth.v1")
        self.assertEqual(self.truth["header"], schema.TRAJ_HEADER_LINE)
        self.assertEqual(self.truth["n_rows"], len(self.rows))
        self.assertIn("citation", self.truth)

    def test_row_identities(self):
        for row in self.rows:
            self.assertEqual(len(row), 66)
            ham, cos, l0, l1, l2, linf, mean_abs = (int(row[2]), float(row[3]), int(row[4]), int(row[5]),
                                                     float(row[6]), int(row[7]), float(row[8]))
            self.assertTrue(1 <= l0 <= 4096, row[:9])
            self.assertGreaterEqual(l1, l0)
            self.assertGreaterEqual(ham, 1)
            self.assertLessEqual(ham, 8 * l0)
            self.assertLessEqual(linf, 255)
            self.assertGreaterEqual(linf, 1)
            self.assertLessEqual(l2, l1 + 1e-3)
            self.assertAlmostEqual(mean_abs, l1 / 4096.0, places=4)
            self.assertEqual(cos, 0.0)
            self.assertTrue(all(v == "0" for v in row[9:]), row[9:12])
            self.assertTrue(0 <= int(row[1]) < schema.N_PAGES)

    def test_rows_grouped_by_seq_and_gaps(self):
        seqs = [int(r[0]) for r in self.rows]
        self.assertEqual(seqs, sorted(seqs))
        present = sorted(set(seqs))
        self.assertEqual(present, [s for s in self.truth["seq"] if s not in (4, 5, 17)])
        self.assertEqual(self.truth["seq"], list(range(1, 31)))
        for s in (4, 5, 17):
            self.assertNotIn(s, present)
        # pages unique within a seq and ascending (the differ's order)
        by_seq: dict[int, list[int]] = {}
        for r in self.rows:
            by_seq.setdefault(int(r[0]), []).append(int(r[1]))
        for s, pages in by_seq.items():
            self.assertEqual(pages, sorted(pages))
            self.assertEqual(len(pages), len(set(pages)))
            self.assertEqual(len(pages), self.truth["K"][s - 1])

    def test_pulse_and_J(self):
        t = self.truth
        self.assertEqual(t["boundaries"], [11, 21])
        K = t["K"]
        for b in t["boundaries"]:
            self.assertGreater(K[b - 1], 500 + 500 - 60)       # K0 + pulse_extra (+ floor, jitter)
            self.assertLess(K[b - 2], 500 + 60 + 40)            # the pair before: no pulse
            j_before = t["J"][b - 2]                              # J(b-1): S_{b-1} against S_b
            self.assertLess(j_before, 0.6)
            self.assertGreater(j_before, 0.4)
        self.assertAlmostEqual(t["expected"]["J_at_boundary"], 0.5)
        self.assertAlmostEqual(t["expected"]["J_steady"], (1 - 0.02) / (1 + 0.02))
        self.assertAlmostEqual(t["expected"]["K_jump_ratio"], (500 + 500 + 40) / (500 + 40))
        self.assertIsNone(t["J"][-1])
        self.assertEqual(t["J"][3 - 1], 0.0)      # S_3 against the empty S_4
        self.assertIsNone(t["J"][4 - 1])          # S_4 and S_5 both empty
        self.assertEqual(t["J"][5 - 1], 0.0)      # empty S_5 against S_6
        self.assertEqual(t["K"][3], 0)
        self.assertEqual(t["K"][4], 0)
        self.assertEqual(t["K"][16], 0)
        steady = [j for i, j in enumerate(t["J"][:-1]) if j is not None and (i + 1) not in (3, 4, 5, 10, 11, 16, 17, 20, 21)]
        self.assertGreater(min(steady), 0.85)
        self.assertLess(max(steady), 1.0)

    def test_truth_columns_complete_and_consistent(self):
        per = self.truth["per_seq_channels"]
        self.assertEqual(set(per.keys()), set(schema.EXTRACT_COLUMNS))
        for col, vals in per.items():
            self.assertEqual(len(vals), 30, col)
        self.assertEqual(set(self.truth["per_seq_channels_t1"].keys()), set(schema.EXTRACT_COLUMNS))
        # the l1 sum over all rows of a seq equals the file's rows
        sums: dict[int, int] = {}
        for r in self.rows:
            sums[int(r[0])] = sums.get(int(r[0]), 0) + int(r[5])
        for i, s in enumerate(self.truth["seq"]):
            self.assertEqual(per["l1_sum_all"][i], sums.get(s, 0))
        self.assertEqual(per["K"], self.truth["K"])
        self.assertEqual(per["J"], self.truth["J"])
        # sets kept (n_pairs * K0 = 15,000 <= 2e6) and equal to the rows' pages
        self.assertIsNotNone(self.truth["sets"])
        by_seq: dict[int, list[int]] = {}
        for r in self.rows:
            by_seq.setdefault(int(r[0]), []).append(int(r[1]))
        for i, s in enumerate(self.truth["seq"]):
            self.assertEqual(self.truth["sets"][i], by_seq.get(s, []))
        # last row blank pair columns; a gap row has K 0 and blank quantiles
        self.assertIsNone(per["n_persist"][-1])
        self.assertIsNone(per["r_l0_q50_per"][-1])
        self.assertEqual(per["ham_sum_all"][3], 0)
        self.assertIsNone(per["ham_q50_all"][3])

    def test_deterministic(self):
        cell2 = synth.write_cell(self.spec, self.tmp / "root2")
        h2, rows2 = read_rows(cell2)
        self.assertEqual(rows2, self.rows)
        t2 = truth_of(cell2)
        self.assertEqual(t2["per_seq_channels"], self.truth["per_seq_channels"])


class TestContentModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="p11_synth_c_"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _ratios(self, spec):
        cell = synth.write_cell(spec, self.tmp / spec.name, keep_sets=False)
        _, rows = read_rows(cell)
        l0 = np.array([int(r[4]) for r in rows]); l1 = np.array([int(r[5]) for r in rows]); ham = np.array([int(r[2]) for r in rows])
        return cell, l0, l1, ham

    def test_spin(self):
        cell, l0, l1, ham = self._ratios(synth.SynthSpec(name="gibbs", seed=3, n_pairs=6, K0=200, content="spin", floor_F=0))
        self.assertTrue(np.all(l1 == l0))
        self.assertTrue(np.all((l0 >= 2) & (l0 <= 8)))
        self.assertGreater((ham / l0).mean(), 1.5)
        self.assertLess((ham / l0).mean(), 2.6)

    def test_counter(self):
        cell, l0, l1, ham = self._ratios(synth.SynthSpec(name="histogram", seed=4, n_pairs=6, K0=800, content="counter", floor_F=0))
        self.assertTrue(np.all((l0 >= 1) & (l0 <= 2)))
        r = (l1 / l0).mean()
        self.assertGreater(r, 20)
        self.assertLess(r, 60)
        self.assertGreater((ham / l0).mean(), 1.5)
        self.assertLess((ham / l0).mean(), 4.0)

    def test_double(self):
        cell, l0, l1, ham = self._ratios(synth.SynthSpec(name="gemm", seed=5, n_pairs=6, K0=300, content="double", floor_F=0))
        self.assertTrue(np.all(l0 % 8 == 0))
        self.assertGreater((l1 / l0).mean(), 80)
        self.assertLess((l1 / l0).mean(), 92)
        self.assertGreater((ham / l0).mean(), 3.6)
        self.assertLess((ham / l0).mean(), 4.4)
        self.assertGreater(np.median(l0), 400)
        self.assertLess(np.median(l0), 640)

    def test_idle(self):
        cell, l0, l1, ham = self._ratios(synth.SynthSpec(name="sleep", seed=6, n_pairs=6, K0=0, content="idle", floor_F=150, label="idle"))
        self.assertTrue(np.all((l0 >= 1) & (l0 <= 4)))
        t = truth_of(cell)
        self.assertTrue(all(abs(k - 150) <= 2 for k in t["K"]))
        self.assertEqual(schema.parse_cell_path(cell)["role"], "idle")

    def test_decay_resets_at_boundary(self):
        spec = synth.SynthSpec(name="floyd", seed=7, n_pairs=30, K0=400, content="decay", decay_factor=0.7,
                               pulse_period=10, pulse_extra=400, floor_F=0, k_noise=0.0)
        cell = synth.write_cell(spec, self.tmp / "floyd", keep_sets=False)
        t = truth_of(cell)
        med = t["per_seq_channels"]["l0_q50_all"]
        # inside a pass the per-snapshot median l0 falls; it resets upward at each boundary
        for start in (0, 10, 20):
            seg = med[start:start + 8]
            self.assertTrue(all(seg[i] > seg[i + 1] for i in range(len(seg) - 1)), seg)
        self.assertGreater(med[10], med[9])
        self.assertGreater(med[20], med[19])
        self.assertEqual(t["boundaries"], [11, 21])

    def test_trend_step_kdecay_l0trend(self):
        t = truth_of(synth.write_cell(synth.SynthSpec(name="rmat_gen", seed=8, n_pairs=20, K0=400, trend=1.0, floor_F=0, k_noise=0.0, churn=0.0),
                                      self.tmp / "trend", keep_sets=False))
        self.assertLess(t["K"][0], 430)
        self.assertGreater(t["K"][-1], 740)
        t = truth_of(synth.write_cell(synth.SynthSpec(name="gemm", seed=9, n_pairs=20, K0=300, step_factor=3.0, floor_F=0, k_noise=0.0),
                                      self.tmp / "step", keep_sets=False))
        self.assertEqual(t["K"][:10], [300] * 10)
        self.assertEqual(t["K"][10:], [900] * 10)
        t = truth_of(synth.write_cell(synth.SynthSpec(name="floyd", seed=10, n_pairs=20, K0=400, pulse_period=5, pulse_extra=100,
                                                      k_decay_factor=0.5, floor_F=0, k_noise=0.0, content="decay"),
                                      self.tmp / "kdecay", keep_sets=False))
        self.assertEqual(t["K"][0:5], [400, 200, 100, 50, 25])
        self.assertEqual(t["K"][5], 500)     # boundary: K0 + pulse_extra
        t = truth_of(synth.write_cell(synth.SynthSpec(name="sleep", seed=11, n_pairs=40, K0=0, content="idle", floor_F=200, l0_trend=-0.75),
                                      self.tmp / "l0trend", keep_sets=False))
        med = t["per_seq_channels"]["l0_q50_all"]
        self.assertGreater(statistics.mean(med[:10]), statistics.mean(med[-10:]))

    def test_persistent_set_churn_zero(self):
        spec = synth.SynthSpec(name="nbody", seed=12, n_pairs=12, K0=300, churn=0.0, floor_F=0, k_noise=0.0)
        t = truth_of(synth.write_cell(spec, self.tmp / "persist", keep_sets=False))
        self.assertEqual(t["J"][:-1], [1.0] * 11)
        self.assertEqual(t["n_persist"][:-1], [300] * 11)
        spec = synth.SynthSpec(name="nbody", seed=12, n_pairs=40, K0=2000, churn=0.10, floor_F=0, k_noise=0.0)
        t = truth_of(synth.write_cell(spec, self.tmp / "persist2", keep_sets=False))
        js = t["J"][:-1]
        self.assertAlmostEqual(statistics.mean(js), (1 - 0.10) / (1 + 0.10), delta=0.02)

    def test_random_row_order_and_seq_first(self):
        spec = synth.SynthSpec(name="spmm", seed=13, n_pairs=5, K0=100, floor_F=0, row_order="random", seq_first=0)
        cell = synth.write_cell(spec, self.tmp / "rand", keep_sets=True)
        _, rows = read_rows(cell)
        seqs = [int(r[0]) for r in rows]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(seqs[0], 0)
        pages0 = [int(r[1]) for r in rows if int(r[0]) == 0]
        self.assertNotEqual(pages0, sorted(pages0))
        t = truth_of(cell)
        self.assertEqual(t["seq"], [0, 1, 2, 3, 4])
        self.assertEqual(t["sets"][0], sorted(pages0))

    def test_corrupt_seq_reverse(self):
        spec = synth.SynthSpec(name="gemm", seed=14, n_pairs=8, K0=50, floor_F=0, corrupt="seq_reverse")
        cell = synth.write_cell(spec, self.tmp / "corrupt", keep_sets=False)
        _, rows = read_rows(cell)
        seqs = [int(r[0]) for r in rows]
        self.assertNotEqual(seqs, sorted(seqs))
        self.assertTrue(any(seqs[i + 1] < seqs[i] for i in range(len(seqs) - 1)))

    def test_no_compress_writes_plain_csv(self):
        spec = synth.SynthSpec(name="fft", seed=15, n_pairs=3, K0=20, floor_F=0)
        cell = synth.write_cell(spec, self.tmp / "plain", compress=False)
        names = [p.name for p in cell.iterdir()]
        self.assertIn("run_matrix_test1_kernel_fft_v2.npy.substrate_trajectory.csv", names)
        self.assertEqual(truth_of(cell)["compressed"], None)
        self.assertEqual(truth_of(synth.write_cell(spec, self.tmp / "plain2", keep_sets=False))["sets"], None)

    def test_pulse_snapshots(self):
        s = synth.SynthSpec(name="a", seed=1, n_pairs=60, pulse_period=12)
        self.assertEqual(synth.pulse_snapshots(s), {12, 24, 36, 48})
        b = synth.SynthSpec(name="a", seed=1, n_pairs=60, pulse_period=12, pulse_burst=True)
        self.assertEqual(synth.pulse_snapshots(b), {24, 25, 48, 49})
        self.assertEqual(synth.pulse_snapshots(synth.SynthSpec(name="a", seed=1)), set())

    def test_validate(self):
        with self.assertRaises(ValueError):
            synth.SynthSpec(name="a", seed=1, content="nope").validate()
        with self.assertRaises(ValueError):
            synth.SynthSpec(name="a", seed=1, corrupt="bad").validate()
        with self.assertRaises(ValueError):
            synth.SynthSpec(name="a", seed=1, K0=262144, floor_F=1).validate()


class TestCorpus(unittest.TestCase):
    def test_corpus_specs_default(self):
        specs = synth.corpus_specs(n_pairs=120, reps=8, idle=8)
        self.assertEqual(len(specs), 12 * 8 + 8)
        by_name: dict[str, list] = {}
        for s in specs:
            by_name.setdefault(s.name, []).append(s)
        self.assertEqual(sorted(by_name), sorted(list(schema.KERNEL_NAMES) + ["sleep"]))
        for ki, (k, _) in enumerate(schema.KERNELS):
            seeds = [s.seed for s in by_name[k]]
            self.assertEqual(seeds, [42] + [1000 * r + ki for r in range(1, 8)])
            self.assertTrue(all(s.label == "synth" for s in by_name[k]))
        idle = by_name["sleep"]
        self.assertTrue(all(s.label == "idle" and s.K0 == 0 and s.content == "idle" and s.floor_F == 150 for s in idle))
        self.assertEqual([s.seed for s in idle], [42] + [1000 * r + 12 for r in range(1, 8)])
        g = by_name["gemm"][0]
        self.assertEqual((g.K0, g.pulse_period, g.pulse_extra, g.churn, g.content), (4096, 24, 4096, 0.02, "double"))
        f = by_name["floyd"][0]
        self.assertEqual((f.K0, f.pulse_period, f.pulse_extra, f.content, f.decay_factor), (2048, 10, 2048, "decay", 0.7))
        self.assertEqual((by_name["gibbs"][0].content, by_name["histogram"][0].content), ("spin", "counter"))
        self.assertEqual(by_name["lexer"][0].K0, 0)
        self.assertEqual(by_name["rmat_gen"][0].trend, 0.5)
        self.assertEqual(by_name["bnb_tsp"][0].k_noise, 0.4)
        # cv-case random: a per-kernel draw in [0.01, 0.10] for every kernel without an explicit k_noise
        for k in ("gemm", "floyd", "fft", "lexer"):
            self.assertTrue(0.01 <= by_name[k][0].k_noise <= 0.10)
            self.assertEqual(len({s.k_noise for s in by_name[k]}), 1)
        self.assertNotEqual(by_name["gemm"][0].k_noise, by_name["fft"][0].k_noise)

    def test_corpus_cases(self):
        bp = {s.name: s for s in synth.corpus_specs(reps=1, idle=0, break_pulse=True)}
        self.assertIsNone(bp["gemm"].pulse_period)
        self.assertEqual(bp["gemm"].pulse_extra, 0)
        bo = {s.name: s for s in synth.corpus_specs(reps=1, idle=0, break_order=True)}
        self.assertEqual((bo["gibbs"].content, bo["gemm"].content), ("double", "spin"))
        self.assertEqual((bo["gibbs"].K0, bo["gemm"].K0), (256, 4096))
        sh = {s.name: s for s in synth.corpus_specs(reps=1, idle=0, cv_case="shot")}
        self.assertAlmostEqual(sh["gemm"].k_noise, 2 / 64)
        self.assertAlmostEqual(sh["gibbs"].k_noise, 2 / 16)
        pr = {s.name: s for s in synth.corpus_specs(reps=1, idle=0, cv_case="preset")}
        self.assertEqual(pr["gemm"].k_noise, 0.02)
        lo = {s.name: s for s in synth.corpus_specs(reps=1, idle=0, level_only=True)}
        self.assertEqual((lo["gemm"].K0, lo["fft"].K0, lo["lexer"].K0, lo["bnb_tsp"].K0), (2048, 4096, 8192, 16384))
        self.assertTrue(all(s.pulse_period is None and s.content == "double" for s in lo.values()))
        op = synth.corpus_specs(reps=1, idle=0, one_preset=True)
        self.assertEqual(len({(s.K0, s.content, s.churn, s.pulse_period, round(s.k_noise, 6)) for s in op}), 1)
        oc = {s.name: s for s in synth.corpus_specs(reps=1, idle=0, ord_case=True)}
        self.assertEqual((oc["gemm"].pulse_period, oc["fft"].pulse_period, oc["lexer"].pulse_period, oc["bnb_tsp"].pulse_period), (6, 12, 6, 12))
        self.assertEqual(len({s.K0 for s in oc.values()}), 1)
        self.assertTrue(all(s.pulse_extra == 2048 and not s.pulse_burst for s in oc.values()))
        ob = {s.name: s for s in synth.corpus_specs(reps=1, idle=0, ord_case=True, ord_mode="burst")}
        self.assertEqual((ob["gemm"].pulse_period, ob["fft"].pulse_period), (12, 12))
        self.assertEqual((ob["gemm"].pulse_burst, ob["fft"].pulse_burst, ob["lexer"].pulse_burst, ob["bnb_tsp"].pulse_burst),
                         (False, True, False, True))
        with self.assertRaises(ValueError):
            synth.corpus_specs(cv_case="x")

    def test_write_small_corpus(self):
        tmp = Path(tempfile.mkdtemp(prefix="p11_corpus_"))
        try:
            m = synth.write_corpus(tmp / "root", n_pairs=4, reps=2, idle=1, keep_sets=False)
            self.assertEqual(m["n_cells"], 25)
            self.assertEqual(m["schema"], "plan11.synth_corpus.v1")
            self.assertIn("params", m)
            self.assertIn("citation", m)
            self.assertTrue((tmp / "root" / "corpus.json").is_file())
            dirs = sorted(p for p in (tmp / "root").rglob("rep*__*") if p.is_dir())
            self.assertEqual(len(dirs), 25)
            roles = {}
            for d in dirs:
                ident = schema.parse_cell_path(d)
                roles[ident["role"]] = roles.get(ident["role"], 0) + 1
                self.assertIsNotNone(ident["seed"])
                self.assertEqual(len([p for p in d.glob(schema.TRAJ_GLOB)]), 1)
                self.assertTrue((d / "truth.json").is_file())
            self.assertEqual(roles, {"kernel": 24, "idle": 1})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestCLI(unittest.TestCase):
    def test_cell_cli_flags(self):
        tmp = Path(tempfile.mkdtemp(prefix="p11_cli_"))
        try:
            rc = synth.main(["cell", "--root", str(tmp / "r"), "--name", "gemm", "--seed", "7", "--n-pairs", "12",
                             "--k0", "80", "--floor-f", "10", "--pulse-period", "4", "--pulse-extra", "40",
                             "--gap-seqs", "3,7", "--step", "--no-compress", "--spin-bytes", "2,8"])
            self.assertEqual(rc, 0)
            cell = tmp / "r" / "kernel" / "kernel_gemm_v2" / "--seed_7_--duration_12" / "rep001__synth"
            self.assertTrue(cell.is_dir())
            t = truth_of(cell)
            self.assertEqual(t["spec"]["step_factor"], 3.0)
            self.assertEqual(t["spec"]["gap_seqs"], [3, 7])
            self.assertEqual(t["K"][2], 0)
            rc = synth.main(["cell", "--root", str(tmp / "r2"), "--name", "floyd", "--seed", "1", "--n-pairs", "8",
                             "--k0", "40", "--content", "decay", "--pulse-period", "4", "--k-decay", "--corrupt", "seq_reverse",
                             "--no-truth-sets"])
            self.assertEqual(rc, 0)
            cell2 = next(p for p in (tmp / "r2").rglob("rep001__synth"))
            t2 = truth_of(cell2)
            self.assertEqual(t2["spec"]["k_decay_factor"], 0.5)
            self.assertEqual(t2["spec"]["corrupt"], "seq_reverse")
            self.assertIsNone(t2["sets"])
            rc = synth.main(["corpus", "--root", str(tmp / "c"), "--n-pairs", "3", "--reps", "1", "--idle", "1", "--no-compress"])
            self.assertEqual(rc, 0)
            self.assertEqual(json.load(open(tmp / "c" / "corpus.json"))["n_cells"], 13)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_runs_as_module_and_by_path(self):
        r = subprocess.run([sys.executable, "-m", "plan11_encoding_ladder.synth", "--help"], cwd=QEMU_DIR,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([sys.executable, str(QEMU_DIR / "plan11_encoding_ladder" / "synth.py"), "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()


class TestEpoch2B12Duration(unittest.TestCase):
    def test_truth_records_the_duration(self):
        """SPEC_epoch2 B12 (i): truth.json carries duration_s (n_pairs x 0.644 by default, the declared value when set)."""
        tmp = Path(tempfile.mkdtemp(prefix="p11_synth_e2_"))
        try:
            cell = synth.write_cell(synth.SynthSpec(name="gemm", seed=42, n_pairs=30, K0=500), tmp / "root", compress=False)
            truth = truth_of(cell)
            self.assertAlmostEqual(truth["duration_s"], 30 * 0.644)
            self.assertAlmostEqual(truth["spec"]["duration_s"], 30 * 0.644)
            cell2 = synth.write_cell(synth.SynthSpec(name="gemm", seed=43, n_pairs=30, K0=500, duration_s=600.0), tmp / "root")
            self.assertEqual(truth_of(cell2)["duration_s"], 600.0)
            self.assertEqual(truth_of(cell2)["duration_s_source"], "declared")
            specs = synth.corpus_specs(n_pairs=20, reps=1, idle=1, duration_s=12.88)
            self.assertTrue(all(s.duration_s == 12.88 for s in specs))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
