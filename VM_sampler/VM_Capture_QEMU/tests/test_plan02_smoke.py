#!/usr/bin/env python3
"""tests/test_plan02_smoke.py -- end-to-end smoke tests for Plan 02 tooling.

No VM required. Validates:
  - plan02_schema.py: write/validate/roundtrip
  - plan02_manifest.py: build/save/load/transitions
  - migrate_schema_v1_to_v2.py: v1 -> v2 roundtrip with backup
  - plan02_analysis.py: synthetic null trace runs ANOVA, recommends an iv

Run with:
    python3 -m unittest VM_sampler/VM_Capture_QEMU/tests/test_plan02_smoke.py

Or:
    python3 VM_sampler/VM_Capture_QEMU/tests/test_plan02_smoke.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make the parent dir importable when running this file directly.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import plan02_schema as sc       # noqa: E402
import plan02_manifest as mf     # noqa: E402
import plan02_analysis as an     # noqa: E402
import migrate_schema_v1_to_v2 as mig  # noqa: E402


class SchemaTests(unittest.TestCase):

    def test_cell_id_is_deterministic(self):
        a = sc.cell_id("ransom", 100, 60, 0)
        b = sc.cell_id("ransom", 100, 60, 0)
        self.assertEqual(a, b)
        c = sc.cell_id("ransom", 100, 60, 1)
        self.assertNotEqual(a, c)

    def test_validate_v2_ok(self):
        rec = _minimal_v2_record()
        ok, errors = sc.validate_v2(rec.to_dict())
        self.assertTrue(ok, errors)

    def test_validate_v2_rejects_v1(self):
        v1 = {"experiment": "old", "config": {"interval_ms": 100}}
        ok, errors = sc.validate_v2(v1)
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_atomic_write_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.json"
            rec = _minimal_v2_record()
            sc.write_json_atomic(path, rec)
            with path.open() as f:
                got = json.load(f)
            self.assertEqual(got["schema_version"], 2)
            self.assertEqual(got["run_meta"]["cell_id"], "test_cell_id_1")


class ManifestTests(unittest.TestCase):

    def test_build_manifest_has_correct_count(self):
        with tempfile.TemporaryDirectory() as td:
            rows = mf.build_manifest(
                workloads=["w1", "w2"],
                intervals_ms=[100, 500, 1000],
                durations_s=[60, 120],
                replicates=3,
                output_dir=Path(td),
                seed=42,
                block_size=8,
                add_warmup_per_block=False,
            )
        self.assertEqual(len(rows), 2 * 3 * 2 * 3)

    def test_warmup_cells_added_per_block(self):
        with tempfile.TemporaryDirectory() as td:
            rows = mf.build_manifest(
                workloads=["w1"],
                intervals_ms=[100, 250],
                durations_s=[60],
                replicates=2,
                output_dir=Path(td),
                seed=0,
                block_size=2,
                add_warmup_per_block=True,
            )
        warmups = [r for r in rows if r.is_warmup]
        # 4 cells / block_size=2 => 2 blocks => 2 warmups
        self.assertEqual(len(warmups), 2)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.csv"
            rows = mf.build_manifest(
                workloads=["w1"],
                intervals_ms=[100],
                durations_s=[60],
                replicates=2,
                output_dir=Path(td),
                seed=0,
                block_size=4,
                add_warmup_per_block=False,
            )
            mf.save(path, rows)
            loaded = mf.load(path)
            self.assertEqual(len(loaded), len(rows))
            self.assertEqual(loaded[0].cell_id, rows[0].cell_id)
            self.assertEqual(loaded[0].is_warmup, rows[0].is_warmup)

    def test_status_transitions(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.csv"
            rows = mf.build_manifest(
                workloads=["w1"], intervals_ms=[100], durations_s=[60],
                replicates=1, output_dir=Path(td), seed=0, block_size=4,
                add_warmup_per_block=False,
            )
            mf.save(path, rows)
            cid = rows[0].cell_id
            mf.set_status_on_disk(path, cid, "running")
            mf.set_status_on_disk(path, cid, "ok", notes_append="all good")
            again = mf.load(path)
            self.assertEqual(again[0].status, "ok")
            self.assertIn("all good", again[0].notes)

    def test_crash_recovery_marks_running_as_failed(self):
        rows = mf.build_manifest(
            workloads=["w1"], intervals_ms=[100], durations_s=[60],
            replicates=2, output_dir=Path("/tmp"), seed=0, block_size=4,
            add_warmup_per_block=False,
        )
        rows[0].status = "running"
        fixed = mf.crashed_running_to_failed(rows)
        self.assertEqual(fixed, 1)
        self.assertEqual(rows[0].status, "failed")
        self.assertIn("crashed", rows[0].notes)

    def test_next_pending_returns_first_pending(self):
        rows = mf.build_manifest(
            workloads=["w1"], intervals_ms=[100, 200], durations_s=[60],
            replicates=1, output_dir=Path("/tmp"), seed=0, block_size=4,
            add_warmup_per_block=False,
        )
        rows[0].status = "ok"
        nxt = mf.next_pending(rows)
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.status, "pending")


class MigratorTests(unittest.TestCase):

    def test_v1_to_v2_roundtrip(self):
        v1 = {
            "experiment": "sandbox_ransom_batched",
            "timestamp": "2026-05-20T00:00:00Z",
            "config": {"interval_ms": 250, "duration_sec": 60,
                       "ram_size_mb": 1024, "vm_domain": "Kali Jeries",
                       "self_clean": True, "queue_files_drained": 5},
            "summary": {"snapshots_attempted": 30, "snapshots_completed": 30,
                        "mean_pmemsave_sec": 0.77,
                        "mean_host_snapshot_cycle_sec": 1.62,
                        "backpressure_events": 0, "queue_max_depth": 0,
                        "estimated_vm_pause_fraction": 0.92},
            "notes": ["bc fix applied"],
        }
        upgraded = mig.migrate_one(v1, workload_override=None, replicate=0)
        ok, errors = sc.validate_v2(upgraded)
        self.assertTrue(ok, errors)
        self.assertEqual(upgraded["run_meta"]["workload"], "sandbox_ransom_batched")
        self.assertEqual(upgraded["run_meta"]["interval_ms"], 250)
        self.assertEqual(upgraded["producer_stats"]["snapshots_completed"], 30)
        self.assertEqual(upgraded["schema_version"], 2)


class AnalysisTests(unittest.TestCase):

    def test_synthetic_null_has_expected_moments(self):
        f1s = an.synthetic_null_f1(seed=0, n_trials=200)
        import statistics as st
        self.assertAlmostEqual(st.fmean(f1s), 0.52, delta=0.03)
        self.assertAlmostEqual(st.stdev(f1s), 0.06, delta=0.02)

    def test_one_way_anova_detects_signal(self):
        # Three groups, clearly different means.
        groups = {
            100: [1.0, 1.1, 0.9, 1.05],
            500: [2.0, 2.1, 1.9, 2.05],
            1000: [3.0, 3.1, 2.9, 3.05],
        }
        result = an.one_way_anova(groups)
        self.assertGreater(result["f_stat"], 100,
                           f"F-stat should be large for clear groups; got {result}")
        self.assertGreater(result["eta_squared"], 0.9)

    def test_one_way_anova_null_signal(self):
        # Same distribution in all groups
        import random
        rng = random.Random(0)
        groups = {
            100: [rng.gauss(0, 1) for _ in range(20)],
            500: [rng.gauss(0, 1) for _ in range(20)],
            1000: [rng.gauss(0, 1) for _ in range(20)],
        }
        result = an.one_way_anova(groups)
        # F-stat should be small (typically < 5) for no real effect
        self.assertLess(result["f_stat"], 5.0,
                        f"F-stat unexpectedly large for null: {result}")

    def test_recommend_iv_picks_slowest_passing(self):
        # Build synthetic cells: iv=100/250/500 all pass; iv=1000 fails on
        # n_windows. Recommendation should be 500.
        cells = []
        for iv in (100, 250, 500, 1000):
            for rep in range(3):
                rm = sc.RunMeta(
                    cell_id=sc.cell_id("ransom", iv, 300, rep),
                    manifest_id="t",
                    block_id=0,
                    workload="sandbox_ransom_batched",
                    interval_ms=iv,
                    duration_s=300,
                    replicate=rep,
                    git_sha="t",
                    host_uname="t",
                    host_kernel="t",
                    qemu_version="t",
                    vm_image_sha256="t",
                    run_started_at="t",
                    run_ended_at="t",
                    exit_status="ok",
                )
                ps = sc.ProducerStats(
                    snapshots_attempted=100,
                    snapshots_completed=100,
                    mean_guest_run_interval_sec=iv / 1000.0 + 0.025,
                    std_guest_run_interval_sec=(iv / 1000.0 + 0.025) * 0.02,
                    backpressure_events=0,
                )
                n_win_value = 200 if iv < 1000 else 10
                ao = sc.AnalyzerOutputs(f1_phase=0.85, cv_workingset=None,
                                        n_windows=n_win_value,
                                        n_snapshots=100)
                rec = sc.PerCellRecord(
                    schema_version=sc.SCHEMA_VERSION,
                    run_meta=rm, producer_stats=ps,
                    analyzer_outputs=ao, notes=[],
                )
                cells.append(rec.to_dict())
        recs = an.recommend_iv_per_workload(cells, an.AcceptanceThresholds())
        wl_rec = recs.get("sandbox_ransom_batched")
        self.assertIsNotNone(wl_rec)
        self.assertEqual(wl_rec["recommended_iv"], 500,
                         f"expected slowest passing iv=500, got {wl_rec}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_v2_record() -> sc.PerCellRecord:
    rm = sc.RunMeta(
        cell_id="test_cell_id_1",
        manifest_id="test_manifest",
        block_id=0,
        workload="sandbox_ransom_batched",
        interval_ms=250,
        duration_s=60,
        replicate=0,
        git_sha="abc123",
        host_uname="Linux 5.0 x86_64",
        host_kernel="5.0",
        qemu_version="7.0",
        vm_image_sha256="deadbeef",
        run_started_at="2026-05-22T00:00:00Z",
        run_ended_at="2026-05-22T00:01:00Z",
        exit_status="ok",
    )
    return sc.PerCellRecord(
        schema_version=sc.SCHEMA_VERSION,
        run_meta=rm,
        producer_stats=sc.ProducerStats(snapshots_completed=30),
        analyzer_outputs=sc.AnalyzerOutputs(),
        notes=[],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
