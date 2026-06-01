#!/usr/bin/env python3
"""tests/test_plan05_aggregate.py -- Plan 05 throughput-pilot gate tests.

No VM required. Validates:
  - aggregate(): G-T1/G-T2/G-T3 verdicts, the three distinct failure modes,
    recommendation picks the best passing arm, missing-baseline group skip
  - _median_or_none()
  - load_records() from a results JSON 'runs' list
  - _arm_apf() reading a real apf_trajectory.jsonl (load_streaming_trajectory)

Run with:
    python3 -m unittest tests.test_plan05_aggregate
    python3 tests/test_plan05_aggregate.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import plan05_aggregate as A  # noqa: E402

GiB = 1024 ** 3


def _apf(seed: int, n: int = 40) -> list[float]:
    return np.random.default_rng(seed).normal(0.30, 0.05, n).clip(0, 1).tolist()


def _runs(pmems, arm, wl="w", dur=120, snaps=12, peak_gib=2.0, seed0=10):
    return [dict(workload=wl, duration_s=dur, arm=arm, replicate=i,
                 mean_pmemsave_sec=pm, snapshots_completed=snaps,
                 peak_dump_bytes=int(peak_gib * GiB), apf=_apf(seed0 + i))
            for i, pm in enumerate(pmems)]


class TestAggregate(unittest.TestCase):
    def _records(self):
        r = []
        r += _runs([7.1, 7.0, 6.9], "ssd_keep", seed0=10)                 # baseline
        r += _runs([3.0, 3.1, 2.9], "tmpfs_keep", snaps=28, peak_gib=2.2, seed0=20)  # pass
        r += _runs([5.0, 5.1, 4.9], "weak", snaps=18, peak_gib=2.1, seed0=30)        # G-T1 fail
        r += _runs([3.0, 3.0, 3.0], "overdisk", snaps=28, peak_gib=4.0, seed0=40)    # G-T3 fail
        return r

    def test_pass_and_fail_modes(self):
        s = A.aggregate(self._records())
        by = {c["arm"]: c for c in s["comparisons"]}
        self.assertTrue(by["tmpfs_keep"]["passed"])
        self.assertFalse(by["weak"]["passed"])
        self.assertFalse(by["weak"]["g_t1_throughput"]["passed"])
        self.assertTrue(by["weak"]["g_t2_fidelity"]["passed"])
        self.assertFalse(by["overdisk"]["passed"])
        self.assertFalse(by["overdisk"]["g_t3_disk"]["passed"])

    def test_speedup_value(self):
        s = A.aggregate(self._records())
        by = {c["arm"]: c for c in s["comparisons"]}
        self.assertAlmostEqual(by["tmpfs_keep"]["g_t1_throughput"]["speedup"],
                               7.0 / 3.0, places=4)

    def test_reco_picks_passing(self):
        s = A.aggregate(self._records())
        reco = s["recommendations"][0]
        self.assertEqual(reco["recommended_arm"], "tmpfs_keep")
        self.assertTrue(reco["passes_all_gates"])

    def test_fidelity_fail_mode(self):
        r = _runs([7.0, 7.0, 7.0], "ssd_keep", seed0=10)
        lev = _runs([3.0, 3.0, 3.0], "tmpfs_keep", snaps=28, peak_gib=2.0, seed0=20)
        for rec in lev:
            rec["apf"] = [x + 0.1 for x in rec["apf"]]  # shift out of equivalence
        s = A.aggregate(r + lev)
        c = next(c for c in s["comparisons"] if c["arm"] == "tmpfs_keep")
        self.assertFalse(c["passed"])
        self.assertFalse(c["g_t2_fidelity"]["passed"])

    def test_missing_baseline_skips(self):
        s = A.aggregate(_runs([3.0], "tmpfs_keep"))  # no ssd_keep baseline
        self.assertEqual(s["comparisons"], [])
        self.assertTrue(any("no baseline" in n for n in s["notes"]))

    def test_median_or_none(self):
        self.assertEqual(A._median_or_none([1, 3, 2]), 2.0)
        self.assertIsNone(A._median_or_none([None, "x"]))


class TestLoadRecords(unittest.TestCase):
    def test_results_json_runs_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "res.json"
            p.write_text(json.dumps({"runs": [
                {"workload": "w", "duration_s": 120, "arm": "ssd_keep",
                 "mean_pmemsave_sec": 7.0}]}))
            self.assertEqual(len(A.load_records(p, None)), 1)

    def test_apf_jsonl_path(self):
        with tempfile.TemporaryDirectory() as d:
            jp = pathlib.Path(d) / "apf_trajectory.jsonl"
            lines = [json.dumps({"seq": i, "apf": 0.3}) for i in range(10)]
            lines.append(json.dumps({"final": True}))
            jp.write_text("\n".join(lines))
            pooled = A._arm_apf([{"apf_jsonl": str(jp)}])
            self.assertEqual(len(pooled), 10)


if __name__ == "__main__":
    unittest.main()
