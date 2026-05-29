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
import time
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
import plan02_validate_session as pv  # noqa: E402
import plan02_metrics_per_cell as pmc  # noqa: E402
import plan02_apf_helper as apf  # noqa: E402


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


class Day10ValidatorTests(unittest.TestCase):

    def test_phase_marker_count_zero_when_no_file(self):
        self.assertEqual(pv._phase_marker_count(Path("/no/such/file")), 0)

    def test_phase_marker_count_extracts(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "workload_stderr.log"
            log.write_text(
                "starting\n"
                "[2026-05-24T00:00:01Z] [PHASE] test=ransom phase=scan\n"
                "[2026-05-24T00:00:02Z] [PHASE] test=ransom phase=encrypt\n"
                "done\n"
            )
            self.assertEqual(pv._phase_marker_count(log), 2)

    def test_compute_n_windows(self):
        self.assertEqual(pv._compute_n_windows(100, 128, 64), 0)
        self.assertEqual(pv._compute_n_windows(192, 128, 64), 2)
        self.assertEqual(pv._compute_n_windows(3328, 128, 64), 51)

    def test_ratio_from_notes(self):
        cell = {"notes": ["foo", "snap completion: actual=148 expected=148 ratio=1.00"]}
        self.assertAlmostEqual(pv._ratio_from_notes(cell), 1.00)

    def test_settle_retries_from_notes(self):
        cell = {"notes": ["vm settle: state='running' lock_retries=3 other_errors=0"]}
        self.assertEqual(pv._settle_retries_from_notes(cell), 3)

    def test_producer_errors_from_notes(self):
        cell = {"notes": ["foo", "producer.log errors (2): err1 | err2"]}
        self.assertEqual(pv._producer_errors_from_notes(cell), 2)
        cell2 = {"notes": ["no errors"]}
        self.assertEqual(pv._producer_errors_from_notes(cell2), 0)

    def test_evaluate_cell_warmup_passes_c1(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            cells_dir = tdpath / "cells"
            workdir = cells_dir / "work"
            workdir.mkdir(parents=True)
            rec = _minimal_v2_record()
            rec.notes = [
                "WARMUP CELL -- discarded by analysis",
                "snap completion: actual=85 expected=85 ratio=1.00",
                "vm settle: state='running' lock_retries=0 other_errors=0",
            ]
            rec.producer_stats.snapshots_completed = 3500  # > 50 windows
            cell_path = cells_dir / "warmup_block0.json"
            cells_dir.mkdir(exist_ok=True)
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir, 0.85, 50, 128, 64)
            self.assertTrue(r["claims"]["C1_workload_ran"]["pass"])
            self.assertTrue(r["claims"]["C2_ratio_healthy"]["pass"])
            self.assertTrue(r["claims"]["C3_enough_windows"]["pass"])
            self.assertTrue(r["claims"]["C4_no_settle_retries"]["pass"])
            self.assertTrue(r["claims"]["C5_producer_log_clean"]["pass"])
            self.assertTrue(r["ok"])

    def test_evaluate_cell_real_workload_needs_phase_markers(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            cells_dir = tdpath / "cells"
            workdir = cells_dir / "work"
            cells_dir.mkdir()
            cid = "abc123"
            (workdir / cid).mkdir(parents=True)
            # No stderr file -> C1 should fail for non-warmup
            rec = _minimal_v2_record()
            rec.run_meta.cell_id = cid
            rec.notes = [
                "snap completion: actual=148 expected=148 ratio=1.00",
                "vm settle: state='running' lock_retries=0 other_errors=0",
            ]
            rec.producer_stats.snapshots_completed = 3500
            cell_path = cells_dir / f"cell_{cid}.json"
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir, 0.85, 50, 128, 64)
            self.assertFalse(r["claims"]["C1_workload_ran"]["pass"])
            self.assertFalse(r["ok"])

    def test_evaluate_cell_low_windows_still_operational(self):
        """C3 failure (low n_windows) must NOT flip ok=False (D-25)."""
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            cells_dir = tdpath / "cells"
            workdir = cells_dir / "work"
            cells_dir.mkdir()
            cid = "low_win_cell"
            (workdir / cid).mkdir(parents=True)
            # add PHASE markers so C1 passes
            (workdir / cid / "workload_stderr.log").write_text(
                "[2026-05-24T00:00:00Z] [PHASE] test=ransom phase=scan\n"
            )
            rec = _minimal_v2_record()
            rec.run_meta.cell_id = cid
            rec.notes = [
                "snap completion: actual=148 expected=148 ratio=1.00",
                "vm settle: state='running' lock_retries=0 other_errors=0",
            ]
            rec.producer_stats.snapshots_completed = 148  # < window=128 + 50*64
            cell_path = cells_dir / f"cell_{cid}.json"
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir, 0.85, 50, 128, 64)
            self.assertTrue(r["claims"]["C1_workload_ran"]["pass"])
            self.assertTrue(r["claims"]["C2_ratio_healthy"]["pass"])
            self.assertFalse(r["claims"]["C3_enough_windows"]["pass"])
            self.assertTrue(r["claims"]["C4_no_settle_retries"]["pass"])
            self.assertTrue(r["claims"]["C5_producer_log_clean"]["pass"])
            self.assertTrue(r["ok"], "operational ok must not depend on C3")
            # Plan 03 contract: C3 is informational only; analysis_ready
            # gates on C7 (window/hop recommendation). With no
            # plan03_recommendation.json present, C7 is NA (pass), so
            # analysis_ready == ok == True. Low n_windows is documented
            # via C3 but no longer blocks analysis_ready.
            self.assertTrue(r["analysis_ready"],
                            "Plan 03: analysis_ready gates on C7 not C3")


class Day12MetricsTests(unittest.TestCase):
    """D-51 · plan02_metrics_per_cell tests."""

    def test_phase_marker_regex_extracts(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "ws.log"
            log.write_text(
                "[2026-05-23T12:11:23Z] [INFO] starting\n"
                "[2026-05-23T12:11:24Z] [PHASE] test=ransom phase=generate\n"
                "[2026-05-23T12:11:30Z] [PHASE] test=ransom phase=encrypt\n"
                "[2026-05-23T12:11:35Z] [PHASE] test=ransom phase=cleanup\n"
            )
            markers = pmc.parse_phase_markers(log)
            self.assertEqual(len(markers), 3)
            self.assertEqual(markers[0][1], "generate")
            self.assertEqual(markers[-1][1], "cleanup")

    def test_n_windows_math(self):
        self.assertEqual(pmc.compute_n_windows(100, 128, 64), 0)
        self.assertEqual(pmc.compute_n_windows(128, 128, 64), 1)
        self.assertEqual(pmc.compute_n_windows(3328, 128, 64), 51)

    def test_cv_workingset_simple(self):
        v = pmc.cv_workingset([0.10, 0.11, 0.09, 0.10])
        self.assertIsNotNone(v)
        self.assertLess(v, 0.15)

    def test_cv_workingset_zero_mean_returns_none(self):
        self.assertIsNone(pmc.cv_workingset([0.0, 0.0, 0.0]))

    def test_cv_workingset_singleton_returns_none(self):
        self.assertIsNone(pmc.cv_workingset([0.5]))

    def test_detect_boundaries_diff_finds_spikes(self):
        # Flat then spike then flat
        traj = [0.10] * 5 + [0.95] + [0.10] * 5
        bnds = pmc.detect_boundaries_diff(traj)
        # Spike at index 5 (the high value); detector should find boundary
        self.assertTrue(any(abs(b - 5) <= 1 for b in bnds),
                        f"expected boundary near 5, got {bnds}")

    def test_f1_score_perfect_match(self):
        r = pmc.f1_score([3, 7, 12], [3, 7, 12], tolerance=0)
        self.assertEqual(r["f1"], 1.0)
        self.assertEqual(r["tp"], 3)

    def test_f1_score_no_overlap(self):
        r = pmc.f1_score([1, 2, 3], [10, 20, 30], tolerance=1)
        self.assertEqual(r["f1"], 0.0)

    def test_f1_score_tolerance_window(self):
        r = pmc.f1_score([5], [6], tolerance=1)
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["f1"], 1.0)

    def test_active_page_fraction_identical_dumps(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.raw"
            b = Path(td) / "b.raw"
            data = bytes(range(256)) * 32  # 8 KiB
            a.write_bytes(data)
            b.write_bytes(data)
            apf = pmc.active_page_fraction(a, b, page_size=4096)
            self.assertEqual(apf, 0.0)

    def test_active_page_fraction_one_page_differs(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.raw"
            b = Path(td) / "b.raw"
            page = bytes(4096)
            # 2 pages: identical first, differ second
            a.write_bytes(page + page)
            b.write_bytes(page + bytes(b"\xff" * 4096))
            apf = pmc.active_page_fraction(a, b, page_size=4096)
            self.assertAlmostEqual(apf, 0.5)

    def test_compute_metrics_for_cell_no_dumps(self):
        """Cell with no dumps in image_dir gracefully returns empty metrics."""
        with tempfile.TemporaryDirectory() as td:
            image_dir = Path(td) / "image_dir"
            image_dir.mkdir()
            jsonl = Path(td) / "snap.jsonl"
            jsonl.write_text("")
            m = pmc.compute_metrics_for_cell(
                cell_id="empty",
                image_dir=image_dir,
                run_start_epoch=0.0,
                jsonl_path=jsonl,
                workload_stderr_path=None,
                workload_type="phasic",
            )
            self.assertEqual(m.n_dumps_examined, 0)
            self.assertIsNone(m.f1_phase)
            self.assertIsNone(m.cv_workingset)
            self.assertTrue(any("no dumps" in n for n in m.notes))


class Day13ApfHelperTests(unittest.TestCase):
    """B+3.1 · plan02_apf_helper end-to-end tests."""

    def _make_dump(self, path: Path, n_pages: int, marker: int) -> None:
        # n_pages × 4096 bytes, all bytes set to `marker`
        path.write_bytes(bytes([marker]) * (n_pages * 4096))

    def test_helper_success_writes_ack_jsonl_and_deletes_prev(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            prev = tdp / "memory_dump-prev.raw"
            curr = tdp / "memory_dump-curr.raw"
            self._make_dump(prev, n_pages=4, marker=0x00)
            self._make_dump(curr, n_pages=4, marker=0xFF)
            apf_jsonl = tdp / "apf_trajectory.jsonl"
            ack_dir = tdp / "acks"
            rc = apf.main([
                "--prev", str(prev), "--curr", str(curr),
                "--apf-jsonl", str(apf_jsonl), "--ack-dir", str(ack_dir),
                "--seq", "0",
            ])
            self.assertEqual(rc, apf.EXIT_OK)
            self.assertFalse(prev.exists(), "prev should be deleted")
            self.assertTrue(curr.exists(), "curr must remain")
            self.assertTrue(apf_jsonl.exists())
            ack = ack_dir / "seq_0000000.apf_done"
            self.assertTrue(ack.exists())
            ack_data = json.loads(ack.read_text())
            self.assertEqual(ack_data["exit_code"], apf.EXIT_OK)
            self.assertAlmostEqual(ack_data["apf"], 1.0)

    def test_helper_corrupt_curr_exits_code_4(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            prev = tdp / "p.raw"
            curr = tdp / "c.raw"
            self._make_dump(prev, n_pages=4, marker=0x00)
            curr.write_bytes(b"")  # zero-byte = corrupt
            rc = apf.main([
                "--prev", str(prev), "--curr", str(curr),
                "--apf-jsonl", str(tdp / "j.jsonl"),
                "--ack-dir", str(tdp / "acks"),
                "--seq", "0",
            ])
            self.assertEqual(rc, apf.EXIT_CORRUPT_DUMP)
            # ack file should still exist with the exit code captured
            ack = tdp / "acks" / "seq_0000000.apf_done"
            self.assertTrue(ack.exists())
            self.assertEqual(json.loads(ack.read_text())["exit_code"],
                             apf.EXIT_CORRUPT_DUMP)

    def test_helper_concurrent_append_no_interleave(self):
        """Multiple helpers writing to the same JSONL must not corrupt lines.

        Production note: producer spawns helpers iv-paced (one per snap
        interval, typically 100-2000 ms apart), so each helper's prev has
        time to be memmapped before the next helper might delete what
        was THIS helper's curr (and that helper's prev). We model the
        same with a small spawn stagger.
        """
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            apf_jsonl = tdp / "t.jsonl"
            ack_dir = tdp / "acks"
            ack_dir.mkdir()
            # Make 6 dump files, then spawn 5 helpers in parallel for pairs.
            for i in range(6):
                p = tdp / f"d{i}.raw"
                self._make_dump(p, n_pages=2, marker=i)
            procs = []
            for seq in range(5):
                prev = tdp / f"d{seq}.raw"
                curr = tdp / f"d{seq+1}.raw"
                procs.append(subprocess.Popen([
                    sys.executable,
                    str(Path(__file__).resolve().parent.parent / "plan02_apf_helper.py"),
                    "--prev", str(prev), "--curr", str(curr),
                    "--apf-jsonl", str(apf_jsonl), "--ack-dir", str(ack_dir),
                    "--seq", str(seq),
                ]))
                # Mimic production: helpers staggered by the iv-pacing,
                # not all fired at once. 100 ms is enough for each helper
                # to open its memmaps before the next one launches.
                time.sleep(0.1)
            for p in procs:
                rc = p.wait(timeout=30)
                self.assertEqual(rc, 0)
            # All 5 lines must parse cleanly
            lines = apf_jsonl.read_text().splitlines()
            self.assertEqual(len(lines), 5)
            seqs = sorted(json.loads(l)["seq"] for l in lines)
            self.assertEqual(seqs, [0, 1, 2, 3, 4])

    def test_load_streaming_trajectory_with_sentinel(self):
        with tempfile.TemporaryDirectory() as td:
            apf_jsonl = Path(td) / "t.jsonl"
            apf_jsonl.write_text(
                json.dumps({"seq": 2, "apf": 0.30}) + "\n" +
                json.dumps({"seq": 0, "apf": 0.10}) + "\n" +
                json.dumps({"seq": 1, "apf": 0.20}) + "\n" +
                json.dumps({"final": True, "n_pairs_expected": 3,
                            "n_ok": 3, "n_failed": 0, "gap_seqs": []}) + "\n"
            )
            traj, sentinel = pmc.load_streaming_trajectory(apf_jsonl)
            self.assertEqual(traj, [0.10, 0.20, 0.30])
            self.assertIsNotNone(sentinel)
            self.assertEqual(sentinel["n_ok"], 3)

    def test_compute_metrics_for_cell_streaming_path(self):
        """compute_metrics_for_cell prefers streaming trajectory when present."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            image_dir = tdp / "img"
            image_dir.mkdir()  # empty · would have caused dump-scan fail
            jsonl = tdp / "snap.jsonl"
            jsonl.write_text(
                json.dumps({"t0_before_suspend": 100.0}) + "\n" +
                json.dumps({"t0_before_suspend": 101.0}) + "\n" +
                json.dumps({"t0_before_suspend": 102.0}) + "\n"
            )
            apf_jsonl = tdp / "apf.jsonl"
            apf_jsonl.write_text(
                json.dumps({"seq": 0, "apf": 0.10}) + "\n" +
                json.dumps({"seq": 1, "apf": 0.12}) + "\n" +
                json.dumps({"final": True, "n_pairs_expected": 2,
                            "n_ok": 2, "n_failed": 0, "gap_seqs": []}) + "\n"
            )
            m = pmc.compute_metrics_for_cell(
                cell_id="stream-1", image_dir=image_dir,
                run_start_epoch=0.0, jsonl_path=jsonl,
                workload_stderr_path=None, workload_type="steady",
                streaming_apf_jsonl=apf_jsonl,
            )
            self.assertEqual(m.n_pairs_examined, 2)
            self.assertAlmostEqual(m.apf_mean, 0.11)
            self.assertIsNotNone(m.cv_workingset)


class Day13ValidatorC6Tests(unittest.TestCase):

    def _build_cell_with_trajectory(self, td: Path, with_sentinel: bool,
                                     n_ok: int = 3, n_expected: int = 3) -> Path:
        """Set up a cell directory with workdir/apf_trajectory.jsonl.
        Returns the cell JSON path that plan02_validate_session expects."""
        cells_dir = td / "cells"
        workdir_root = cells_dir / "work"
        cid = "test_cell_cid"
        cell_workdir = workdir_root / cid
        cell_workdir.mkdir(parents=True)
        apf_jsonl = cell_workdir / "apf_trajectory.jsonl"
        lines = []
        for i in range(n_ok):
            lines.append(json.dumps({"seq": i, "apf": 0.1 * (i + 1)}))
        if with_sentinel:
            lines.append(json.dumps({
                "final": True,
                "n_pairs_expected": n_expected,
                "n_ok": n_ok,
                "n_failed": n_expected - n_ok,
                "gap_seqs": [],
            }))
        apf_jsonl.write_text("\n".join(lines) + "\n")

        # Build a passing cell JSON
        rec = _minimal_v2_record()
        rec.run_meta.cell_id = cid
        rec.notes = [
            "snap completion: actual=148 expected=148 ratio=1.00",
            "vm settle: state='running' lock_retries=0 other_errors=0",
        ]
        rec.producer_stats.snapshots_completed = 4000  # > 50 windows
        (workdir_root / cid / "workload_stderr.log").write_text(
            "[2026-05-24T00:00:00Z] [PHASE] test=ransom phase=scan\n"
        )
        cell_path = cells_dir / f"cell_{cid}.json"
        sc.write_json_atomic(cell_path, rec)
        return cell_path, workdir_root

    def test_c6_passes_when_complete_trajectory(self):
        with tempfile.TemporaryDirectory() as td:
            cell_path, workdir = self._build_cell_with_trajectory(
                Path(td), with_sentinel=True, n_ok=3, n_expected=3
            )
            r = pv.evaluate_cell(cell_path, workdir, 0.85, 50, 128, 64)
            self.assertTrue(r["claims"]["C6_apf_complete"]["pass"])
            self.assertTrue(r["ok"])

    def test_c6_fails_when_no_sentinel(self):
        with tempfile.TemporaryDirectory() as td:
            cell_path, workdir = self._build_cell_with_trajectory(
                Path(td), with_sentinel=False
            )
            r = pv.evaluate_cell(cell_path, workdir, 0.85, 50, 128, 64)
            self.assertFalse(r["claims"]["C6_apf_complete"]["pass"])
            self.assertFalse(r["ok"], "operational ok must reflect C6 fail")

    def test_c6_fails_when_too_many_gaps(self):
        with tempfile.TemporaryDirectory() as td:
            # Only 2 of 10 expected pairs OK
            cell_path, workdir = self._build_cell_with_trajectory(
                Path(td), with_sentinel=True, n_ok=2, n_expected=10
            )
            r = pv.evaluate_cell(cell_path, workdir, 0.85, 50, 128, 64)
            self.assertFalse(r["claims"]["C6_apf_complete"]["pass"])
            self.assertFalse(r["ok"])

    def test_c2_lower_threshold_for_keep_dumps_cell(self):
        """Day-14 D-71 / v3 D-80: B+3.1 cells (keep_dumps=True) accept a
        low ratio (floor now 0.08) while v1 cells require >= 0.85."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            workdir_root = cells_dir / "work"
            cid = "bplus31_low_ratio"
            (workdir_root / cid).mkdir(parents=True)
            (workdir_root / cid / "workload_stderr.log").write_text(
                "[2026-05-24T00:00:00Z] [PHASE] test=ransom phase=scan\n"
            )
            # Build apf trajectory file so C6 applies + passes
            (workdir_root / cid / "apf_trajectory.jsonl").write_text(
                json.dumps({"seq": 0, "apf": 0.1}) + "\n" +
                json.dumps({"final": True, "n_pairs_expected": 1,
                            "n_ok": 1, "n_failed": 0, "gap_seqs": []}) + "\n"
            )
            rec = _minimal_v2_record()
            rec.run_meta.cell_id = cid
            rec.run_meta.keep_dumps = True
            rec.notes = [
                # Ratio 0.24 · would FAIL v1 (need >= 0.85)
                # BUT must PASS B+3.1 (need >= 0.15)
                "snap completion: actual=8 expected=34 ratio=0.24 threshold=0.15 (mode=B+3.1)",
                "vm settle: state='running' lock_retries=0 other_errors=0",
            ]
            rec.producer_stats.snapshots_completed = 4000
            cell_path = cells_dir / f"cell_{cid}.json"
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir_root, 0.85, 50, 128, 64)
            self.assertTrue(r["claims"]["C2_ratio_healthy"]["pass"],
                            f"C2 should pass for B+3.1 at ratio=0.24, got {r}")
            self.assertIn("B+3.1", r["claims"]["C2_ratio_healthy"]["why"])

    def test_c2_strict_threshold_for_v1_cell(self):
        """v1 cells (keep_dumps=False) still require ratio >= 0.85."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            workdir_root = cells_dir / "work"
            cid = "v1_low_ratio"
            (workdir_root / cid).mkdir(parents=True)
            (workdir_root / cid / "workload_stderr.log").write_text(
                "[2026-05-24T00:00:00Z] [PHASE] test=ransom phase=scan\n"
            )
            rec = _minimal_v2_record()
            rec.run_meta.cell_id = cid
            rec.run_meta.keep_dumps = False
            rec.notes = [
                "snap completion: actual=20 expected=100 ratio=0.20 threshold=0.30 (mode=v1)",
                "vm settle: state='running' lock_retries=0 other_errors=0",
            ]
            rec.producer_stats.snapshots_completed = 4000
            cell_path = cells_dir / f"cell_{cid}.json"
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir_root, 0.85, 50, 128, 64)
            self.assertFalse(r["claims"]["C2_ratio_healthy"]["pass"])
            self.assertIn("v1", r["claims"]["C2_ratio_healthy"]["why"])

    def test_c2_v3_sustained_ratio_passes_at_new_floor(self):
        """v3 D-80: a B+3.1 sustained-workload cell at ratio=0.12 (which
        false-failed under the old 0.15 floor) must now PASS at 0.08."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            workdir_root = cells_dir / "work"
            cid = "v3_sustained"
            (workdir_root / cid).mkdir(parents=True)
            (workdir_root / cid / "workload_stderr.log").write_text(
                "[2026-05-28T00:00:00Z] [PHASE] test=rmw phase=measure\n"
            )
            (workdir_root / cid / "apf_trajectory.jsonl").write_text(
                json.dumps({"seq": 0, "apf": 0.3}) + "\n" +
                json.dumps({"final": True, "n_pairs_expected": 1,
                            "n_ok": 1, "n_failed": 0, "gap_seqs": []}) + "\n"
            )
            rec = _minimal_v2_record()
            rec.run_meta.cell_id = cid
            rec.run_meta.keep_dumps = True
            rec.notes = [
                # ratio 0.12 · failed at old 0.15 floor · passes at 0.08
                "snap completion: actual=18 expected=148 ratio=0.12 threshold=0.08 (mode=B+3.1)",
                "vm settle: state='running' lock_retries=0 other_errors=0",
            ]
            rec.producer_stats.snapshots_completed = 18
            cell_path = cells_dir / f"cell_{cid}.json"
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir_root, 0.85, 50, 128, 64)
            self.assertTrue(r["claims"]["C2_ratio_healthy"]["pass"],
                            f"C2 should pass for B+3.1 at ratio=0.12, got {r}")

    def test_c2_true_stall_still_fails_under_new_floor(self):
        """v3 D-80: the lowered 0.08 floor must still catch a true capture
        stall (VM absent / workload missing -> ratio ~0.03)."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            workdir_root = cells_dir / "work"
            cid = "v3_stall"
            (workdir_root / cid).mkdir(parents=True)
            (workdir_root / cid / "workload_stderr.log").write_text(
                "[2026-05-28T00:00:00Z] [PHASE] test=rmw phase=measure\n"
            )
            (workdir_root / cid / "apf_trajectory.jsonl").write_text(
                json.dumps({"seq": 0, "apf": 0.3}) + "\n" +
                json.dumps({"final": True, "n_pairs_expected": 1,
                            "n_ok": 1, "n_failed": 0, "gap_seqs": []}) + "\n"
            )
            rec = _minimal_v2_record()
            rec.run_meta.cell_id = cid
            rec.run_meta.keep_dumps = True
            rec.notes = [
                # ratio 0.03 · a genuine stall · must still FAIL at 0.08
                "snap completion: actual=5 expected=148 ratio=0.03 threshold=0.08 (mode=B+3.1)",
                "vm settle: state='running' lock_retries=0 other_errors=0",
            ]
            rec.producer_stats.snapshots_completed = 5
            cell_path = cells_dir / f"cell_{cid}.json"
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir_root, 0.85, 50, 128, 64)
            self.assertFalse(r["claims"]["C2_ratio_healthy"]["pass"],
                             f"C2 must still fail a true stall at ratio=0.03, got {r}")

    def test_c4_na_when_no_settle_note(self):
        """D-76: cells without a `vm settle: ... lock_retries=N` note must
        NOT fail C4. Backward compat with v1-era + crash recovery."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            workdir_root = cells_dir / "work"
            cid = "no_settle_line"
            (workdir_root / cid).mkdir(parents=True)
            (workdir_root / cid / "workload_stderr.log").write_text(
                "[2026-05-24T00:00:00Z] [PHASE] test=ransom phase=scan\n"
            )
            rec = _minimal_v2_record()
            rec.run_meta.cell_id = cid
            rec.run_meta.keep_dumps = False
            rec.notes = [
                "snap completion: actual=148 expected=148 ratio=1.00",
                # NO 'vm settle:' line · only a warning (matches the D-76 bug)
                "vm settle warning: Command timed out after 5 seconds",
            ]
            rec.producer_stats.snapshots_completed = 4000
            cell_path = cells_dir / f"cell_{cid}.json"
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir_root, 0.85, 50, 128, 64)
            self.assertTrue(r["claims"]["C4_no_settle_retries"]["pass"],
                            "C4 must pass when no settle note exists (NA)")
            self.assertIn("NA", r["claims"]["C4_no_settle_retries"]["why"])

    def test_c4_fails_when_lock_retries_positive(self):
        """C4 still flags genuine contention (lock_retries > 0)."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            workdir_root = cells_dir / "work"
            cid = "had_lock_retries"
            (workdir_root / cid).mkdir(parents=True)
            (workdir_root / cid / "workload_stderr.log").write_text(
                "[2026-05-24T00:00:00Z] [PHASE] test=ransom phase=scan\n"
            )
            rec = _minimal_v2_record()
            rec.run_meta.cell_id = cid
            rec.run_meta.keep_dumps = False
            rec.notes = [
                "snap completion: actual=148 expected=148 ratio=1.00",
                "vm settle: state='running' lock_retries=3 timeout_retries=0 other_errors=0",
            ]
            rec.producer_stats.snapshots_completed = 4000
            cell_path = cells_dir / f"cell_{cid}.json"
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir_root, 0.85, 50, 128, 64)
            self.assertFalse(r["claims"]["C4_no_settle_retries"]["pass"])
            self.assertIn("lock_retries=3", r["claims"]["C4_no_settle_retries"]["why"])

    def test_c6_na_when_no_trajectory_file(self):
        """v1-era cells without keep_dumps must not be penalized."""
        with tempfile.TemporaryDirectory() as td:
            cells_dir = Path(td) / "cells"
            workdir_root = cells_dir / "work"
            workdir_root.mkdir(parents=True)
            cid = "v1cell"
            (workdir_root / cid).mkdir()
            (workdir_root / cid / "workload_stderr.log").write_text(
                "[2026-05-24T00:00:00Z] [PHASE] test=ransom phase=scan\n"
            )
            rec = _minimal_v2_record()
            rec.run_meta.cell_id = cid
            rec.notes = [
                "snap completion: actual=148 expected=148 ratio=1.00",
                "vm settle: state='running' lock_retries=0 other_errors=0",
            ]
            rec.producer_stats.snapshots_completed = 4000
            cell_path = cells_dir / f"cell_{cid}.json"
            sc.write_json_atomic(cell_path, rec)
            r = pv.evaluate_cell(cell_path, workdir_root, 0.85, 50, 128, 64)
            self.assertTrue(r["claims"]["C6_apf_complete"]["pass"],
                            "NA C6 must pass trivially for v1 cells")
            self.assertFalse(r["claims"]["C6_apf_complete"]["operational"])
            self.assertTrue(r["ok"])


class Day13BarrierTests(unittest.TestCase):
    """B+3.1 cell-end barrier in plan02_run."""

    def test_barrier_zero_pairs_returns_clean(self):
        with tempfile.TemporaryDirectory() as td:
            info = pr.wait_for_apf_helpers(Path(td), n_pairs_expected=0,
                                            timeout_s=1.0)
            self.assertEqual(info["n_pairs_expected"], 0)
            self.assertEqual(info["n_ok"], 0)

    def test_barrier_collects_existing_acks(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            for s in range(3):
                ack = tdp / f"seq_{s:07d}.apf_done"
                ack.write_text(json.dumps({"seq": s, "exit_code": 0}))
            info = pr.wait_for_apf_helpers(tdp, n_pairs_expected=3,
                                            timeout_s=1.0)
            self.assertEqual(info["n_ok"], 3)
            self.assertEqual(info["gap_seqs"], [])
            self.assertFalse(info["timed_out"])

    def test_barrier_detects_gap(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            for s in (0, 1, 3, 4):  # seq=2 missing
                ack = tdp / f"seq_{s:07d}.apf_done"
                ack.write_text(json.dumps({"seq": s, "exit_code": 0}))
            info = pr.wait_for_apf_helpers(tdp, n_pairs_expected=5,
                                            timeout_s=0.5)
            self.assertEqual(info["gap_seqs"], [2])
            self.assertTrue(info["timed_out"])

    def test_sentinel_writer(self):
        with tempfile.TemporaryDirectory() as td:
            apf_jsonl = Path(td) / "t.jsonl"
            apf_jsonl.write_text(json.dumps({"seq": 0, "apf": 0.1}) + "\n")
            pr.write_apf_final_sentinel(apf_jsonl, {
                "n_pairs_expected": 5, "n_ok": 4, "n_failed": 1,
                "gap_seqs": [3], "timed_out": False,
            })
            lines = apf_jsonl.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            sentinel = json.loads(lines[-1])
            self.assertTrue(sentinel["final"])
            self.assertEqual(sentinel["n_ok"], 4)

    def test_barrier_waits_through_progressive_acks_bug_o(self):
        """Bug O: slow helpers must not trip the barrier when they are
        still making progress."""
        import threading
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            def drip_acks():
                for s in range(5):
                    time.sleep(0.25)
                    ack = tdp / f"seq_{s:07d}.apf_done"
                    ack.write_text(json.dumps({"seq": s, "exit_code": 0}))

            t = threading.Thread(target=drip_acks, daemon=True)
            t.start()
            # idle_timeout 1 s is shorter than the 1.25 s total run, but
            # each ack restarts the idle clock, so the barrier must wait
            # for all 5.
            info = pr.wait_for_apf_helpers(
                tdp, n_pairs_expected=5,
                timeout_s=10.0, idle_timeout_s=1.0,
                poll_interval_s=0.05,
            )
            t.join(timeout=2.0)
            self.assertEqual(info["n_ok"], 5)
            self.assertEqual(info["gap_seqs"], [])
            self.assertFalse(info["timed_out"])

    def test_barrier_idle_quits_when_helpers_stall_bug_o(self):
        """Bug O: when helpers stop making progress, barrier must quit
        on idle timeout rather than burn the full hard cap."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # Only 3 of 5 expected acks present; nothing else will appear.
            for s in range(3):
                ack = tdp / f"seq_{s:07d}.apf_done"
                ack.write_text(json.dumps({"seq": s, "exit_code": 0}))
            t0 = time.monotonic()
            info = pr.wait_for_apf_helpers(
                tdp, n_pairs_expected=5,
                timeout_s=30.0, idle_timeout_s=0.5,
                poll_interval_s=0.05,
            )
            elapsed = time.monotonic() - t0
            # Should quit ~0.5 s after the 3 acks were observed, well
            # before the 30 s hard cap.
            self.assertLess(elapsed, 2.0)
            self.assertTrue(info["timed_out"])
            self.assertEqual(info["n_ok"], 3)
            self.assertEqual(info["gap_seqs"], [3, 4])

    def test_barrier_backward_compat_single_timeout_arg(self):
        """Legacy callers passing only timeout_s must still see the
        original semantics (idle = hard cap)."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            t0 = time.monotonic()
            info = pr.wait_for_apf_helpers(
                tdp, n_pairs_expected=2, timeout_s=0.3,
            )
            elapsed = time.monotonic() - t0
            self.assertGreaterEqual(elapsed, 0.25)  # full timeout consumed
            self.assertTrue(info["timed_out"])
            self.assertEqual(info["n_ok"], 0)


class DurationInjectionTests(unittest.TestCase):
    """v3: orchestrator auto-injects --duration + --phase-markers into the
    workload command (plan02_run._augment_workload_command)."""

    def test_duration_appended_when_absent(self):
        out = pr._augment_workload_command("/bin/mem_x", 300)
        self.assertIn("--duration 300", out)
        self.assertIn("--phase-markers", out)

    def test_duration_not_doubled_when_present(self):
        out = pr._augment_workload_command("/bin/mem_x --duration 600", 300)
        self.assertEqual(out.count("--duration"), 1)
        self.assertIn("--duration 600", out)
        self.assertNotIn("--duration 300", out)

    def test_phase_markers_appended_when_absent(self):
        out = pr._augment_workload_command("/bin/mem_x", 120)
        self.assertEqual(out.count("--phase-markers"), 1)

    def test_phase_markers_not_doubled(self):
        out = pr._augment_workload_command("/bin/mem_x --phase-markers", 120)
        self.assertEqual(out.count("--phase-markers"), 1)

    def test_both_present_passthrough(self):
        base = "/bin/mem_x --phase-markers --duration 600"
        out = pr._augment_workload_command(base, 300)
        self.assertEqual(out.count("--phase-markers"), 1)
        self.assertEqual(out.count("--duration"), 1)
        self.assertIn("--duration 600", out)

    def test_duration_value_is_exact_cell_duration(self):
        out = pr._augment_workload_command("/bin/mem_x", 120)
        self.assertIn("--duration 120", out)

    def test_binary_path_token_unchanged(self):
        # The orchestrator's binary-existence probe uses wcmd.split()[0];
        # appended flags must never change token 0.
        out = pr._augment_workload_command("/bin/sandbox_ransom_seq --files 4000", 600)
        self.assertEqual(out.split()[0], "/bin/sandbox_ransom_seq")


class WorkloadClassifierTests(unittest.TestCase):
    """v3: lock the workload->family mapping for all Phase-2 workloads so a
    rename cannot silently reclassify (plan02_run._classify_workload)."""

    PHASIC = [
        "sandbox_ransom_batched", "sandbox_ransom_seq",
        "sandbox_ransom_slowburn", "sandbox_ransom_selective",
        "sandbox_scanner_metadata",
    ]
    STEADY = [
        "mem_workingset_sweep_v2", "mem_mmap_traversal_v2",
        "mem_pagefault_density_v2", "mem_rmw_intensity_v2",
        "mem_writemag_sweep_v2", "app_hashtable_intensive_v2",
    ]

    def test_classify_all_phasic(self):
        for name in self.PHASIC:
            with self.subTest(name=name):
                self.assertEqual(pr._classify_workload(name), "phasic")

    def test_classify_all_steady(self):
        for name in self.STEADY:
            with self.subTest(name=name):
                self.assertEqual(pr._classify_workload(name), "steady")

    def test_classify_bimodal_hashtable_is_steady(self):
        # hashtable is build->probe bimodal; labeled steady (probe dominates).
        self.assertEqual(
            pr._classify_workload("app_hashtable_intensive_v2"), "steady")

    def test_classify_scanner_is_phasic(self):
        self.assertEqual(
            pr._classify_workload("sandbox_scanner_metadata"), "phasic")

    def test_classify_unknown_fallthrough(self):
        self.assertEqual(pr._classify_workload("totally_made_up"), "unknown")

    def test_classify_case_insensitive(self):
        self.assertEqual(
            pr._classify_workload("MEM_WORKINGSET_SWEEP_V2"), "steady")


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
