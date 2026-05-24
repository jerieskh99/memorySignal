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
import plan02_backfill_nwindows as bf  # noqa: E402
import plan02_run as pr  # noqa: E402


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


class BackfillTests(unittest.TestCase):

    def test_compute_n_windows_below_threshold(self):
        # 100 snaps, window=128, hop=64 -> 0 (no complete window)
        self.assertEqual(bf.compute_n_windows(100, 128, 64), 0)

    def test_compute_n_windows_exact_threshold(self):
        # 128 snaps, window=128, hop=64 -> 1 window
        self.assertEqual(bf.compute_n_windows(128, 128, 64), 1)
        # 192 snaps -> floor((192-128)/64)+1 = 2
        self.assertEqual(bf.compute_n_windows(192, 128, 64), 2)
        # 256 -> 3
        self.assertEqual(bf.compute_n_windows(256, 128, 64), 3)

    def test_backfill_one_populates_n_windows(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            rec = _minimal_v2_record()
            # 256 snaps -> 3 windows at window=128, hop=64.
            # That's below min_windows=50 -> skipped=True.
            rec.producer_stats.snapshots_completed = 256
            cell_path = tdpath / "cell_x.json"
            sc.write_json_atomic(cell_path, rec)
            result = bf.backfill_one(cell_path, 128, 64, 50, dry_run=False)
            self.assertEqual(result["n_windows"], 3)
            self.assertTrue(result["skipped"])  # 3 < min_windows=50
            with cell_path.open() as f:
                got = json.load(f)
            self.assertEqual(got["analyzer_outputs"]["n_windows"], 3)
            self.assertTrue(any("step-1.5a" in n for n in got["notes"]))

    def test_backfill_one_above_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            rec = _minimal_v2_record()
            # To pass 50 windows: (n - 128) // 64 + 1 >= 50
            # -> n >= 50 * 64 + 128 = 3328
            rec.producer_stats.snapshots_completed = 3500
            cell_path = tdpath / "cell_z.json"
            sc.write_json_atomic(cell_path, rec)
            result = bf.backfill_one(cell_path, 128, 64, 50, dry_run=False)
            self.assertGreaterEqual(result["n_windows"], 50)
            self.assertFalse(result["skipped"])

    def test_backfill_marks_below_threshold_as_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            rec = _minimal_v2_record()
            rec.producer_stats.snapshots_completed = 50  # < window=128
            cell_path = tdpath / "cell_y.json"
            sc.write_json_atomic(cell_path, rec)
            result = bf.backfill_one(cell_path, 128, 64, 50, dry_run=False)
            self.assertEqual(result["n_windows"], 0)
            self.assertTrue(result["skipped"])

    def test_apply_skip_to_manifest_flips_ok_to_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.csv"
            rows = mf.build_manifest(
                workloads=["w1"], intervals_ms=[100], durations_s=[60],
                replicates=2, output_dir=Path(td), seed=0, block_size=4,
                add_warmup_per_block=False,
            )
            for r in rows:
                r.status = "ok"
            mf.save(path, rows)
            target = rows[0].cell_id
            flipped = bf.apply_skip_to_manifest(path, {target}, dry_run=False)
            self.assertEqual(flipped, 1)
            again = mf.load(path)
            self.assertEqual(again[0].status, "skipped")
            self.assertEqual(again[1].status, "ok")


class ManifestWorkloadColumnTests(unittest.TestCase):

    def test_build_manifest_populates_workload_command(self):
        with tempfile.TemporaryDirectory() as td:
            rows = mf.build_manifest(
                workloads=["sandbox_ransom_batched"],
                intervals_ms=[100], durations_s=[60], replicates=1,
                output_dir=Path(td), seed=0, block_size=4,
                add_warmup_per_block=False,
                workload_commands={"sandbox_ransom_batched": "/bin/foo --rounds 5"},
                ssh_target="kali@1.2.3.4",
                keep_dumps=True,
            )
            self.assertEqual(rows[0].workload_command, "/bin/foo --rounds 5")
            self.assertEqual(rows[0].ssh_target, "kali@1.2.3.4")
            self.assertTrue(rows[0].keep_dumps)

    def test_csv_roundtrip_preserves_workload_columns(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.csv"
            rows = mf.build_manifest(
                workloads=["w1"], intervals_ms=[100], durations_s=[60],
                replicates=1, output_dir=Path(td), seed=0, block_size=4,
                add_warmup_per_block=False,
                workload_commands={"w1": "echo hi"},
                ssh_target="user@host",
                keep_dumps=True,
            )
            mf.save(path, rows)
            loaded = mf.load(path)
            self.assertEqual(loaded[0].workload_command, "echo hi")
            self.assertEqual(loaded[0].ssh_target, "user@host")
            self.assertTrue(loaded[0].keep_dumps)


class Day8QualityChecksTests(unittest.TestCase):
    """Tests for Day-8 quantitative ok and pause-fraction estimator."""

    def test_pause_fraction_at_known_iv(self):
        self.assertAlmostEqual(pr.estimated_pause_fraction(100), 0.924)
        self.assertAlmostEqual(pr.estimated_pause_fraction(2000), 0.424)

    def test_pause_fraction_interpolates(self):
        # iv=750 is between 500 (0.741) and 1000 (0.593). Linear interp.
        pf = pr.estimated_pause_fraction(750)
        self.assertTrue(0.593 < pf < 0.741, f"got {pf}")

    def test_pause_fraction_clamps(self):
        self.assertAlmostEqual(pr.estimated_pause_fraction(10), 0.924)
        self.assertAlmostEqual(pr.estimated_pause_fraction(99999), 0.424)

    def test_expected_snapshots_reasonable(self):
        # iv=500 ms, d=300 s, pause=0.741 -> guest=78 s, snaps ~ 78/0.525 = 148
        n = pr.expected_snapshots(300, 500)
        self.assertTrue(140 <= n <= 160, f"got {n}")

    def test_expected_snapshots_iv_zero_safe(self):
        self.assertEqual(pr.expected_snapshots(60, 0), 0)


class Day9DiskGuardrailTests(unittest.TestCase):

    def test_disk_free_gib_returns_positive(self):
        with tempfile.TemporaryDirectory() as td:
            free = pr._disk_free_gib(Path(td))
            self.assertGreater(free, 0.0)

    def test_count_stale_dumps_zero_in_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(pr._count_stale_dumps(Path(td)), 0)

    def test_count_stale_dumps_counts_files(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "memory_dump-20260524000001.raw").write_text("x")
            (tdp / "memory_dump-20260524000002.raw").write_text("x")
            (tdp / "other.txt").write_text("x")  # not counted
            self.assertEqual(pr._count_stale_dumps(tdp), 2)

    def test_pre_cell_disk_check_passes_on_empty_tmp(self):
        with tempfile.TemporaryDirectory() as td:
            ok, info = pr.pre_cell_disk_check(Path(td), ram_mb=1, min_headroom_dumps=1)
            self.assertTrue(ok, info)

    def test_pre_cell_disk_check_unrealistic_ram_fails(self):
        with tempfile.TemporaryDirectory() as td:
            # Pretend we need 10 PiB of headroom (1024 * 10^10 MiB).
            ok, info = pr.pre_cell_disk_check(Path(td), ram_mb=10**10, min_headroom_dumps=1)
            self.assertFalse(ok, info)

    def test_scan_producer_log_extracts_errors(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "producer.log"
            log_path.write_text(
                "[2026] starting cycle\n"
                "[2026] error: pmemsave failed: No space left on device\n"
                "[2026] cycle done\n"
                "[2026] error: virsh resume denied\n"
            )
            lines = pr.scan_producer_log(log_path)
            self.assertEqual(len(lines), 2)
            self.assertTrue(any("No space" in l for l in lines))

    def test_scan_producer_log_missing_file_safe(self):
        self.assertEqual(pr.scan_producer_log(Path("/nope/does/not/exist")), [])


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
