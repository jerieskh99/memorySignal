#!/usr/bin/env python3
"""tests/test_plan05_fidelity.py -- Plan 05 APF-fidelity gate (G-T2) tests.

No VM required. Validates:
  - tost_mean(): equivalence pass/fail, zero-SE constant arms, too-few NA
  - bootstrap_std_equiv(): same/different spread, determinism under fixed seed
  - ks_not_rejected(): identical-distribution pass, disjoint reject
  - fidelity_gate(): composite pass, shifted-mean fail, short-arm NA

Run with:
    python3 -m unittest tests.test_plan05_fidelity
    python3 tests/test_plan05_fidelity.py
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import plan05_fidelity as F  # noqa: E402


class TestTostMean(unittest.TestCase):
    def test_identical_pass(self):
        a = list(np.linspace(0.2, 0.4, 60))
        r = F.tost_mean(a, a)
        self.assertTrue(r.applicable)
        self.assertTrue(r.passed)

    def test_shift_fail(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.3, 0.05, 200).tolist()
        b = (np.asarray(a) + 0.1).tolist()
        r = F.tost_mean(a, b)
        self.assertTrue(r.applicable)
        self.assertFalse(r.passed)

    def test_constant_equal_zero_se(self):
        r = F.tost_mean([0.3] * 10, [0.3] * 10)
        self.assertTrue(r.detail.get("degenerate_se"))
        self.assertTrue(r.passed)

    def test_constant_diff_zero_se_fail(self):
        r = F.tost_mean([0.3] * 10, [0.5] * 10)
        self.assertTrue(r.detail.get("degenerate_se"))
        self.assertFalse(r.passed)

    def test_too_few_not_applicable(self):
        r = F.tost_mean([0.3], [0.3, 0.4])
        self.assertFalse(r.applicable)


class TestBootstrapStd(unittest.TestCase):
    def test_same_std_pass(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.3, 0.05, 200).tolist()
        b = rng.normal(0.3, 0.05, 200).tolist()
        self.assertTrue(F.bootstrap_std_equiv(a, b).passed)

    def test_diff_std_fail(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0.3, 0.03, 200).tolist()
        b = rng.normal(0.3, 0.15, 200).tolist()
        self.assertFalse(F.bootstrap_std_equiv(a, b).passed)

    def test_deterministic(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0.3, 0.05, 80).tolist()
        b = rng.normal(0.3, 0.05, 80).tolist()
        r1 = F.bootstrap_std_equiv(a, b)
        r2 = F.bootstrap_std_equiv(a, b)
        self.assertEqual(r1.detail["ci_low"], r2.detail["ci_low"])
        self.assertEqual(r1.detail["ci_high"], r2.detail["ci_high"])


class TestKs(unittest.TestCase):
    def test_same_not_rejected(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0.3, 0.05, 200).tolist()
        b = rng.normal(0.3, 0.05, 200).tolist()
        self.assertTrue(F.ks_not_rejected(a, b).passed)

    def test_disjoint_rejected(self):
        self.assertFalse(F.ks_not_rejected([0.1] * 30, [0.9] * 30).passed)


class TestFidelityGate(unittest.TestCase):
    def test_same_pass(self):
        rng = np.random.default_rng(6)
        a = rng.normal(0.3, 0.05, 200).tolist()
        b = rng.normal(0.3, 0.05, 200).tolist()
        r = F.fidelity_gate(a, b)
        self.assertTrue(r.applicable)
        self.assertTrue(r.passed)

    def test_shift_fail(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0.3, 0.05, 200).tolist()
        b = (np.asarray(a) + 0.1).tolist()
        r = F.fidelity_gate(a, b)
        self.assertFalse(r.passed)
        names = {c["name"]: c["passed"] for c in r.checks}
        self.assertFalse(names["tost_mean"])

    def test_short_not_applicable(self):
        r = F.fidelity_gate([0.3] * 200, [0.3] * 5)
        self.assertFalse(r.applicable)
        self.assertFalse(r.passed)


if __name__ == "__main__":
    unittest.main()
