#!/usr/bin/env python3
"""tests/test_schema.py -- builder 1's tests of schema.py (SPEC 1, 2.2, 2.6, 2.7, 3.1.3).

Run: python3 -m pytest -q plan11_encoding_ladder/tests   (or python3 -m unittest discover -s plan11_encoding_ladder/tests)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plan11_encoding_ladder import __version__, schema  # noqa: E402


class TestConstants(unittest.TestCase):
    def test_instrument_constants(self):
        self.assertEqual(__version__, "0.1.0")
        self.assertEqual(schema.N_PAGES, 262144)
        self.assertEqual(schema.PAGE_SIZE, 4096)
        self.assertEqual(schema.BITS_PER_PAGE, 32768)
        self.assertEqual(schema.BITS_PER_PAGE, schema.PAGE_SIZE * 8)
        self.assertEqual(schema.DURATION_S, 600)
        self.assertEqual(schema.DT_BRACKET_S, (0.500, 0.644))
        self.assertEqual(schema.QUANTILES, (0.05, 0.25, 0.50, 0.75, 0.95))
        self.assertEqual(schema.SIDECAR_SCHEMA, "plan11.extract.v1")

    def test_trajectory_header(self):
        self.assertEqual(len(schema.TRAJ_HEADER), 66)
        self.assertEqual(schema.HEADER_NCOLS_EXPECTED, 66)
        self.assertEqual(schema.TRAJ_HEADER[:9], ("seq", "page_index", "hamming", "cosine", "l0", "l1", "l2", "linf", "mean_abs"))
        self.assertEqual(schema.TRAJ_HEADER.index("hamming"), 2)
        self.assertEqual(schema.TRAJ_HEADER.index("l0"), 4)
        self.assertEqual(schema.TRAJ_HEADER.index("l1"), 5)
        self.assertEqual(schema.TRAJ_HEADER[-1], "max_run_len")
        self.assertTrue(schema.TRAJ_HEADER_LINE.startswith("seq,page_index,hamming,cosine,l0,l1,l2,linf,mean_abs,"))
        self.assertEqual(len(set(schema.TRAJ_HEADER)), 66)


class TestExtractColumns(unittest.TestCase):
    def test_count_and_order(self):
        cols = schema.EXTRACT_COLUMNS
        self.assertEqual(len(cols), 58)
        self.assertEqual(len(set(cols)), 58)
        self.assertEqual(cols[:7], ("seq", "K", "n_persist", "n_union", "J", "J_null_inter", "J_null"))
        self.assertEqual(cols[7:13], ("ham_sum_all", "ham_q05_all", "ham_q25_all", "ham_q50_all", "ham_q75_all", "ham_q95_all"))
        self.assertEqual(cols[13], "l0_sum_all")
        self.assertEqual(cols[19], "l1_sum_all")
        self.assertEqual(cols[25], "ham_sum_per")
        self.assertEqual(cols[31], "l0_sum_per")
        self.assertEqual(cols[37], "l1_sum_per")
        self.assertEqual(cols[43:48], ("r_l0_q05_per", "r_l0_q25_per", "r_l0_q50_per", "r_l0_q75_per", "r_l0_q95_per"))
        self.assertEqual(cols[48], "r_l1l0_q05_per")
        self.assertEqual(cols[53], "r_haml0_q05_per")
        self.assertEqual(cols[-1], "r_haml0_q95_per")
        self.assertEqual(len(schema.RATIO_COLUMNS), 15)

    def test_int_columns(self):
        self.assertIn("seq", schema.EXTRACT_INT_COLUMNS)
        self.assertIn("K", schema.EXTRACT_INT_COLUMNS)
        self.assertIn("ham_sum_per", schema.EXTRACT_INT_COLUMNS)
        self.assertNotIn("J", schema.EXTRACT_INT_COLUMNS)
        self.assertNotIn("l0_q50_all", schema.EXTRACT_INT_COLUMNS)

    def test_format_value(self):
        self.assertEqual(schema.format_value("K", 12), "12")
        self.assertEqual(schema.format_value("K", 12.0), "12")
        self.assertEqual(schema.format_value("J", 0.5), "0.5")
        self.assertEqual(schema.format_value("J", 1 / 3), "0.3333333333")
        self.assertEqual(schema.format_value("l0_q50_all", 2048.0), "2048")
        self.assertEqual(schema.format_value("J", None), "")
        self.assertEqual(schema.format_value("K", None), "")


class TestKernels(unittest.TestCase):
    def test_kernels_and_archetypes(self):
        self.assertEqual(len(schema.KERNELS), 12)
        names = [k for k, _ in schema.KERNELS]
        self.assertEqual(sorted(names), sorted(["bnb_tsp", "fem_assembly", "fft", "floyd", "gemm", "gibbs",
                                                "histogram", "lexer", "nbody", "rmat_gen", "spmm", "stencil_jacobi"]))
        counts = {}
        for _, a in schema.KERNELS:
            counts[a] = counts.get(a, 0) + 1
        self.assertEqual(counts, {"WORKING-SET": 6, "SCATTER": 3, "SEQUENTIAL-GROW": 2, "FRONTIER-CHURN": 1})
        self.assertEqual(schema.ARCHETYPES, ("IDLE", "WORKING-SET", "SCATTER", "SEQUENTIAL-GROW", "FRONTIER-CHURN"))
        self.assertEqual(schema.LEVEL_MATCHED_SETS, (("floyd", "histogram", "nbody"), ("fft", "gemm"), ("fft", "stencil_jacobi")))
        self.assertEqual(schema.ARCHETYPE_OF["gemm"], "WORKING-SET")
        self.assertEqual(schema.ARCHETYPE_OF["fft"], "SCATTER")
        self.assertEqual(schema.ARCHETYPE_OF["lexer"], "SEQUENTIAL-GROW")
        self.assertEqual(schema.ARCHETYPE_OF["bnb_tsp"], "FRONTIER-CHURN")
        self.assertEqual(schema.REP0_SEED, 42)


class TestGrid(unittest.TestCase):
    def test_grid_points_and_ids(self):
        pts = schema.grid_points(119)
        self.assertEqual(len(pts), 13)
        ids = [schema.grid_id(W, H) for W, H in pts]
        self.assertEqual(ids, ["W8_H2", "W8_H4", "W8_H8", "W16_H4", "W16_H8", "W16_H16",
                               "W32_H8", "W32_H16", "W32_H32", "W64_H16", "W64_H32", "W64_H64", "Wall_Hall"])
        self.assertEqual(pts[-1], (119, 119))
        self.assertEqual(schema.grid_points(900)[-1], (900, 900))
        self.assertEqual(schema.hop_of(8, 0.25), 2)
        self.assertEqual(schema.hop_of(64, 1.0), 64)
        self.assertEqual(schema.grid_id("whole", "whole"), "Wall_Hall")
        self.assertEqual(schema.grid_id("whole", 5), "Wall_Hall")
        with self.assertRaises(ValueError):
            schema.grid_id(8, 3)
        with self.assertRaises(ValueError):
            schema.grid_id(119, 60)
        with self.assertRaises(ValueError):
            schema.grid_id(12, 6)
        self.assertEqual(schema.GRID_WINDOWS, (8, 16, 32, 64, "whole"))
        self.assertEqual(schema.GRID_HOP_RATIOS, (0.25, 0.50, 1.00))
        self.assertAlmostEqual(schema.hop_ratio_of(8, 4), 0.5)


class TestCellIdentity(unittest.TestCase):
    def test_campaign_of(self):
        self.assertEqual(schema.campaign_of("dwarfs1"), "dwarfs1")
        self.assertEqual(schema.campaign_of("dwarfs1_resume"), "dwarfs1")
        self.assertEqual(schema.campaign_of("dwarfs1_resume3"), "dwarfs1")
        self.assertEqual(schema.campaign_of("sandbox_deepdive_01c"), "01c")
        self.assertEqual(schema.campaign_of("sandbox_deepdive_01c1"), "01c1")
        self.assertEqual(schema.campaign_of("synth"), "synth")
        self.assertEqual(schema.campaign_of(""), "")

    def test_kernel_of_test_label(self):
        self.assertEqual(schema.kernel_of_test_label("kernel_gemm_v2"), "gemm")
        self.assertEqual(schema.kernel_of_test_label("kernel_stencil_jacobi_v2"), "stencil_jacobi")
        self.assertEqual(schema.kernel_of_test_label("kernel_fem_assembly_v12"), "fem_assembly")
        self.assertEqual(schema.kernel_of_test_label("kernel_bnb_tsp"), "bnb_tsp")

    def test_parse_kernel_cell(self):
        p = "/some/root/kernel/kernel_gemm_v2/--dim_1024_--block_64_--seed_42_--duration_600/rep001__sandbox_deepdive_01c1"
        d = schema.parse_cell_path(p)
        self.assertEqual(d["family"], "kernel")
        self.assertEqual(d["test_label"], "kernel_gemm_v2")
        self.assertEqual(d["kernel"], "gemm")
        self.assertEqual(d["seed"], 42)
        self.assertEqual(d["rep_dir"], 1)
        self.assertEqual(d["label"], "sandbox_deepdive_01c1")
        self.assertEqual(d["campaign"], "01c1")
        self.assertEqual(d["role"], "kernel")
        self.assertEqual(d["archetype_predicted"], "WORKING-SET")

    def test_parse_stencil_dwarfs1_and_hash_suffix(self):
        p = ("kernel/kernel_stencil_jacobi_v2/--grid-n_1024_--seed_1298_--duration_600_--sustain_loop_1_xx_"
             "1a2b3c4d/rep001__dwarfs1_resume")
        d = schema.parse_cell_path(p)
        self.assertEqual(d["kernel"], "stencil_jacobi")
        self.assertEqual(d["seed"], 1298)
        self.assertEqual(d["campaign"], "dwarfs1")
        self.assertEqual(d["archetype_predicted"], "WORKING-SET")

    def test_parse_idle_and_unknown(self):
        d = schema.parse_cell_path(Path("root/idle/idle_sleep/sleep_600/rep003__idle_01c"))
        self.assertEqual(d["role"], "idle")
        self.assertEqual(d["archetype_predicted"], "control")
        self.assertIsNone(d["seed"])
        self.assertEqual(d["rep_dir"], 3)
        self.assertEqual(d["label"], "idle_01c")
        d2 = schema.parse_cell_path("root/kernel/kernel_sleep_v2/--seed_42_--duration_120/rep001__idle")
        self.assertEqual(d2["role"], "idle")
        d3 = schema.parse_cell_path("root/kernel/kernel_foo_v2/--seed_42/rep001__x")
        self.assertEqual(d3["role"], "unknown")
        self.assertEqual(d3["archetype_predicted"], "unknown")
        d4 = schema.parse_cell_path("root/kernel/kernel_gemm_v2/--seed_42/rep002", idle_markers=("gemm",))
        self.assertEqual(d4["role"], "idle")
        self.assertEqual(d4["label"], "")
        with self.assertRaises(ValueError):
            schema.parse_cell_path("a/b/c")

    def test_cell_id_of(self):
        self.assertEqual(schema.cell_id_of("gemm", "kernel", 0, "01c1"), "gemm__rep00__01c1")
        self.assertEqual(schema.cell_id_of("sleep", "idle", 7, "idle"), "idle__rep07__idle")
        self.assertEqual(schema.CELLS_COLUMNS[0], "cell_id")
        self.assertEqual(schema.CELLS_COLUMNS[-1], "status")


class TestSchemaCompatDeleted(unittest.TestCase):
    """Build epoch 2 (SPEC_epoch2 section 1.3 and B21; E1 sec. 4 "Named but not implemented": the inert
    fallback `_schema_compat.py` "can be deleted"): the file is gone and `series` binds builder 1's
    `schema` with no fallback."""

    def test_schema_compat_deleted_and_series_binds_schema(self):
        pkg = Path(__file__).resolve().parents[1]
        self.assertFalse((pkg / "_schema_compat.py").exists())
        from plan11_encoding_ladder import series
        self.assertIs(series.schema, schema)
        src = (pkg / "series.py").read_text()
        self.assertNotIn("import _schema_compat", src)
        self.assertNotIn("except ImportError", src.split("from plan11_encoding_ladder import schema", 1)[1][:200])


if __name__ == "__main__":
    unittest.main()


class TestEpoch2B21(unittest.TestCase):
    def test_schema_compat_deleted_and_series_imports_schema(self):
        """SPEC_epoch2 B21 (E1 4 "can be deleted"): the inert fallback is gone and series.schema is schema."""
        from plan11_encoding_ladder import series
        self.assertFalse((Path(__file__).resolve().parents[1] / "_schema_compat.py").exists())
        self.assertIs(series.schema, schema)
