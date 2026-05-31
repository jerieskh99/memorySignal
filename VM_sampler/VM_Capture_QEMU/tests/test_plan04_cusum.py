#!/usr/bin/env python3
"""tests/test_plan04_cusum.py -- Plan 04 detector/driver/validator tests.

No VM required. Validates:
  - plan04_cusum: rolling mean, two-sided CUSUM, boundary detection,
    snap-index mapping, full pipeline, stationarity score, NaN/MAD edges.
  - plan04_run: CSV columns, JSON schema, per-workload aggregation,
    legacy-detector path, missing-recommendation skip, --force gating,
    per-workload marker tolerance.
  - plan02_validate_session._evaluate_plan04_segmenter + evaluate_cell
    C8 wiring: NA when artifact absent / workload not evaluated, phasic
    pass/fail, steady pass/fail, schema rejection, analysis_ready wiring.

Run with:
    python3 -m unittest tests.test_plan04_cusum
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Make the parent dir importable when running this file directly.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import plan02_manifest as mf  # noqa: E402
import plan02_schema as sc  # noqa: E402
import plan02_validate_session as pv  # noqa: E402
import plan04_cusum as cusum  # noqa: E402
import plan04_run as p4run  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers (mirror style of test_plan03_sweep)
# ---------------------------------------------------------------------------

def _write_apf_jsonl(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, v in enumerate(values):
        lines.append(json.dumps({"seq": i, "apf": v}))
    lines.append(json.dumps({
        "final": True,
        "n_pairs_expected": len(values),
        "n_ok": len(values),
        "n_failed": 0,
        "gap_seqs": [],
    }))
    path.write_text("\n".join(lines) + "\n")


def _make_manifest_with(cells: list[tuple[str, str, int, int, str]],
                        td: Path) -> Path:
    """cells is a list of (cell_id, workload, iv_ms, duration_s, status)."""
    rows: list[mf.ManifestRow] = []
    for cid, workload, iv, dur, status in cells:
        rows.append(mf.ManifestRow(
            cell_id=cid, manifest_id="test-mf", block_id=0,
            workload=workload, interval_ms=iv, duration_s=dur,
            replicate=0, is_warmup=False, status=status,
            expected_path=str(td / f"cell_{cid}.json"),
        ))
    path = td / "manifest.csv"
    mf.save(path, rows)
    return path


def _write_recommendation(path: Path,
                          per_workload: list[tuple[str, int, int]]) -> None:
    """Write plan03_recommendation.json with the given (workload, W, H) tuples."""
    payload = {
        "schema": "plan03.window_hop_recommendations.v1",
        "recommendations": [
            {"workload": wl, "recommended_window": w, "recommended_hop": h,
             "passes_acceptance": True, "degrades_v2_baseline": False}
            for wl, w, h in per_workload
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_plan04_results(path: Path,
                          per_workload_entries: list[dict]) -> None:
    """Write a plan04_segmenter_results.json fixture."""
    payload = {
        "schema": p4run.RESULTS_SCHEMA,
        "captured_at": "2026-06-01T00:00:00Z",
        "input": {},
        "detector": {"name": "cusum", "k": 2.0, "h": 4.0,
                     "min_separation": 2, "marker_tolerance": "auto"},
        "per_workload": per_workload_entries,
        "gates": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _minimal_cell_record(cell_id: str, workload: str,
                         snapshots_completed: int = 100) -> sc.PerCellRecord:
    rm = sc.RunMeta(
        cell_id=cell_id,
        manifest_id="test_manifest",
        block_id=0,
        workload=workload,
        interval_ms=100,
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
        producer_stats=sc.ProducerStats(
            snapshots_completed=snapshots_completed),
        analyzer_outputs=sc.AnalyzerOutputs(),
        notes=[
            "snap completion: actual=100 expected=100 ratio=1.00",
            "vm settle: state='running' lock_retries=0 other_errors=0",
        ],
    )


def _setup_validator_cell(cells_dir: Path, cell_id: str,
                          workload: str) -> Path:
    """Build a per-cell JSON + workdir + stderr that pass C1..C5/C6."""
    cells_dir.mkdir(parents=True, exist_ok=True)
    workdir = cells_dir / "work"
    (workdir / cell_id).mkdir(parents=True, exist_ok=True)
    # PHASE marker → C1 pass.
    (workdir / cell_id / "workload_stderr.log").write_text(
        "[2026-05-24T00:00:00Z] [PHASE] test=demo phase=run\n"
    )
    # Complete trajectory → C6 pass.
    _write_apf_jsonl(workdir / cell_id / "apf_trajectory.jsonl",
                     [0.5] * 32)
    cell_path = cells_dir / f"cell_{cell_id}.json"
    sc.write_json_atomic(cell_path,
                          _minimal_cell_record(cell_id, workload,
                                               snapshots_completed=3500))
    return cell_path


# ===========================================================================
# Detector tests
# ===========================================================================

class Plan04CusumDetectorTests(unittest.TestCase):

    def test_rolling_mean_constant(self):
        # Constant trajectory → all window means equal.
        out = cusum.rolling_mean_apf([0.42] * 40, window=8, hop=4)
        self.assertGreater(out.shape[0], 0)
        self.assertTrue(np.allclose(out, 0.42))

    def test_rolling_mean_window_count(self):
        # For T=40, W=8, H=4 → floor((40-8)/4)+1 = 9.
        T, W, H = 40, 8, 4
        out = cusum.rolling_mean_apf(list(range(T)), window=W, hop=H)
        self.assertEqual(out.shape[0], (T - W) // H + 1)

    def test_rolling_mean_empty(self):
        self.assertEqual(cusum.rolling_mean_apf([], 8, 4).shape[0], 0)

    def test_rolling_mean_short(self):
        # T < W → empty.
        self.assertEqual(cusum.rolling_mean_apf([0.1] * 5, 8, 4).shape[0], 0)

    def test_windowed_cusum_zero_on_constant(self):
        rolling = cusum.rolling_mean_apf([0.5] * 40, 8, 4)
        s_pos, s_neg = cusum.windowed_cusum(rolling)
        self.assertEqual(s_pos.shape, s_neg.shape)
        self.assertEqual(s_pos.shape[0], rolling.shape[0])
        self.assertTrue(np.allclose(s_pos, 0.0))
        self.assertTrue(np.allclose(s_neg, 0.0))

    def test_windowed_cusum_step_triggers_pos(self):
        # Synthetic step (low → high). After standardisation, the high
        # half pushes z above 0. With k below the post-step z magnitude
        # (~0.67 on a clean 0/1 step under MAD scaling), S_pos must
        # rise away from 0.
        traj = [0.0] * 30 + [1.0] * 30
        rolling = cusum.rolling_mean_apf(traj, 8, 4)
        s_pos, _s_neg = cusum.windowed_cusum(rolling, k=0.1)
        self.assertGreater(float(s_pos.max()), 0.0,
                            "S_pos must rise above zero on an up-step")

    def test_detect_boundaries_constant_no_boundaries(self):
        # Constant input must never emit a false positive.
        rolling = cusum.rolling_mean_apf([0.5] * 40, 8, 4)
        self.assertEqual(cusum.detect_boundaries_cusum(rolling), [])

    def test_detect_boundaries_step_finds_one(self):
        # 30 zeros + 30 ones at W=8, H=4. Default k=2.0 gates against
        # ±0.67-sigma steps (MAD scaling); for a clean ±1 step we relax
        # k/h to verify the change-point lands near the truth.
        traj = [0.0] * 30 + [1.0] * 30
        rolling = cusum.rolling_mean_apf(traj, 8, 4)
        bnd = cusum.detect_boundaries_cusum(rolling, k=0.1, h=0.5)
        self.assertGreaterEqual(len(bnd), 1,
                                 f"expected ≥1 boundary, got {bnd}")
        truth_window_idx = (30 - 8 // 2) // 4  # ≈ 6
        self.assertTrue(any(abs(b - truth_window_idx) <= 4 for b in bnd),
                         f"no boundary near truth ({truth_window_idx}): {bnd}")

    def test_detect_boundaries_two_steps(self):
        # 20 zeros + 20 ones + 20 zeros → two truth steps.
        # Relax k/h as in the single-step case; min_separation kept low
        # so both boundaries survive.
        traj = [0.0] * 20 + [1.0] * 20 + [0.0] * 20
        rolling = cusum.rolling_mean_apf(traj, 8, 4)
        bnd = cusum.detect_boundaries_cusum(
            rolling, k=0.1, h=0.5, min_separation=2)
        truth_window_indices = [(20 - 4) // 4, (40 - 4) // 4]  # ≈ [4, 9]
        self.assertGreaterEqual(len(bnd), 1,
                                 f"expected boundaries, got {bnd}")
        # Each truth step must have at least one boundary within ±4.
        for t in truth_window_indices:
            self.assertTrue(any(abs(b - t) <= 4 for b in bnd),
                             f"no boundary near truth {t}: {bnd}")

    def test_min_separation_merges_nearby(self):
        # Two boundaries one window apart must collapse to one when
        # min_separation=2. Construct a clean synthetic by forcing CUSUM
        # to emit twice in close succession via an up-step followed by
        # an immediate down-step.
        rolling = np.array([0.0, 0.0, 0.0, 5.0, 5.0, -5.0, 0.0, 0.0],
                            dtype=np.float64)
        b_strict = cusum.detect_boundaries_cusum(
            rolling, k=0.5, h=1.0, min_separation=1)
        b_merged = cusum.detect_boundaries_cusum(
            rolling, k=0.5, h=1.0, min_separation=10)
        self.assertGreaterEqual(len(b_strict), 1)
        self.assertLessEqual(len(b_merged), len(b_strict),
                              "min_separation must not produce more boundaries")
        if len(b_strict) >= 2:
            self.assertLess(len(b_merged), len(b_strict),
                             "with adjacent boundaries, merging must reduce count")

    def test_boundaries_to_snap_centered(self):
        # window_idx=0, W=8, H=4 → snap_idx = 0*4 + 8//2 = 4.
        out = cusum.boundaries_to_snap_indices([0], 8, 4)
        self.assertEqual(out, [4])
        # window_idx=3, W=8, H=4 → 3*4 + 4 = 16.
        out2 = cusum.boundaries_to_snap_indices([3], 8, 4)
        self.assertEqual(out2, [16])

    def test_detect_boundaries_full_smoke(self):
        # 50-sample burst trajectory: low / high / low.
        traj = [0.0] * 17 + [1.0] * 17 + [0.0] * 16
        out = cusum.detect_boundaries_full(traj, 8, 4)
        self.assertEqual(out, sorted(set(out)),
                          "snap indices must be sorted and unique")
        # All snap indices are within trajectory bounds (plus W//2 head).
        for s in out:
            self.assertGreaterEqual(s, 0)
            self.assertLess(s, len(traj))

    def test_stationarity_score_constant(self):
        # Constant trajectory → no boundaries → score = 1.0.
        self.assertEqual(cusum.stationarity_score([0.5] * 40), 1.0)

    def test_stationarity_score_with_boundaries(self):
        # A trajectory with several alternating bursts → expect ≥3
        # boundaries → score saturates at 0.0.
        traj: list[float] = []
        for _ in range(6):
            traj += [0.0] * 6 + [1.0] * 6
        score = cusum.stationarity_score(traj)
        rolling = cusum.rolling_mean_apf(traj, 8, 4)
        n_bnd = len(cusum.detect_boundaries_cusum(rolling))
        if n_bnd >= 3:
            self.assertEqual(score, 0.0,
                             f"3+ boundaries should saturate to 0 "
                             f"(n_bnd={n_bnd}, score={score})")
        else:
            self.assertAlmostEqual(score, 1.0 - n_bnd / 3.0, places=6)

    def test_mad_zero_fallback(self):
        # MAD = 0 but std > 0: e.g. mostly identical values with two
        # outliers (median == 0 stays unaffected; MAD-fallback to std).
        # We construct a rolling mean that the detector consumes directly.
        rolling = np.array([0.0] * 10 + [1.0] + [0.0] * 10,
                            dtype=np.float64)
        # MAD here is 0 (median = 0; ≥half the entries are 0). std > 0.
        med = float(np.median(rolling))
        mad = float(np.median(np.abs(rolling - med)))
        self.assertEqual(mad, 0.0, "fixture must exercise the MAD=0 path")
        self.assertGreater(float(np.std(rolling)), 0.0)
        # Must not raise; output is a (possibly empty) list of ints.
        out = cusum.detect_boundaries_cusum(rolling)
        self.assertIsInstance(out, list)
        for v in out:
            self.assertIsInstance(v, int)

    def test_nan_input_no_crash(self):
        # NaN entries propagate but must not produce a Python-level crash.
        traj = [0.1, 0.2, math.nan, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        try:
            out = cusum.detect_boundaries_full(traj, 8, 4)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"detect_boundaries_full crashed on NaN input: {exc!r}")
        # Output must be a list (possibly empty). NaN ordering inside
        # the detector is implementation-defined; we only require sanity.
        self.assertIsInstance(out, list)


# ===========================================================================
# Driver tests
# ===========================================================================

class Plan04DriverTests(unittest.TestCase):

    def _build_minimal_session(self, td: Path,
                                cells: list[tuple[str, str]],
                                traj_by_cell: dict[str, list[float]] | None = None,
                                recommendations: list[tuple[str, int, int]] | None = None,
                                ) -> tuple[Path, Path, Path]:
        """Build cells_dir + manifest + recommendation under td. Returns
        (cells_dir, manifest, rec_path)."""
        cells_dir = td / "cells"
        cells_dir.mkdir(parents=True, exist_ok=True)
        manifest_cells = [(cid, wl, 100, 60, "ok") for cid, wl in cells]
        manifest = _make_manifest_with(manifest_cells, td)
        # APF trajectory per cell.
        traj_by_cell = traj_by_cell or {}
        for cid, _wl in cells:
            traj = traj_by_cell.get(cid, [0.3 + 0.01 * i for i in range(40)])
            _write_apf_jsonl(cells_dir / "work" / cid / "apf_trajectory.jsonl",
                              traj)
        # Recommendation (default: one (8,4) per workload).
        if recommendations is None:
            distinct = sorted({wl for _cid, wl in cells})
            recommendations = [(wl, 8, 4) for wl in distinct]
        rec_path = cells_dir / "plan03_recommendation.json"
        _write_recommendation(rec_path, recommendations)
        return cells_dir, manifest, rec_path

    def test_driver_writes_csv_with_expected_columns(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir, manifest, rec_path = self._build_minimal_session(
                tdp, [
                    ("c_ph", "sandbox_ransom_batched"),
                    ("c_st", "app_workingset"),
                ],
            )
            out_csv = tdp / "p4.csv"
            csv_rows, recs = p4run.sweep(
                cells_dir=cells_dir, manifest_path=manifest,
                recommendation_path=rec_path, output_csv=out_csv,
                detector="cusum",
            )
            self.assertTrue(out_csv.is_file())
            with out_csv.open() as f:
                reader = csv.DictReader(f)
                self.assertEqual(list(reader.fieldnames), p4run.CSV_FIELDS)
                rows = list(reader)
            self.assertEqual(len(rows), 2,
                              f"expected 2 rows, got {len(rows)}: {rows}")
            self.assertEqual(len(csv_rows), 2)

    def test_driver_emits_json_with_v1_schema(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir, manifest, rec_path = self._build_minimal_session(
                tdp, [("c0", "sandbox_ransom_batched")])
            out_csv = tdp / "p4.csv"
            out_json = tdp / "p4.json"
            rc = p4run._main([
                "--cells-dir", str(cells_dir),
                "--manifest", str(manifest),
                "--recommendation", str(rec_path),
                "--output-csv", str(out_csv),
                "--output-json", str(out_json),
            ])
            self.assertEqual(rc, 0,
                              f"plan04_run._main exited {rc}, expected 0")
            self.assertTrue(out_json.is_file())
            with out_json.open() as f:
                payload = json.load(f)
            self.assertEqual(payload.get("schema"),
                              "plan04.segmenter_results.v1")
            for k in ("captured_at", "input", "detector",
                       "per_workload", "gates"):
                self.assertIn(k, payload, f"JSON missing key {k}")

    def test_driver_per_workload_aggregation(self):
        # 4 ransom cells (phasic, marker-less since we don't write
        # workload_stderr.log) → single per_workload entry with G3
        # plausibility metric.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells = [(f"c{i}", "sandbox_ransom_batched") for i in range(4)]
            cells_dir, manifest, rec_path = self._build_minimal_session(
                tdp, cells)
            out_csv = tdp / "p4.csv"
            out_json = tdp / "p4.json"
            rc = p4run._main([
                "--cells-dir", str(cells_dir),
                "--manifest", str(manifest),
                "--recommendation", str(rec_path),
                "--output-csv", str(out_csv),
                "--output-json", str(out_json),
            ])
            self.assertEqual(rc, 0)
            with out_json.open() as f:
                payload = json.load(f)
            pw = payload["per_workload"]
            self.assertEqual(len(pw), 1,
                              f"expected 1 workload entry, got {len(pw)}: {pw}")
            entry = pw[0]
            self.assertEqual(entry["workload"], "sandbox_ransom_batched")
            self.assertEqual(entry["n_cells"], 4)
            # Either marker-rich median or marker-less plausibility was
            # produced; check that at least one of the medians is present
            # (None is allowed for the other one).
            self.assertIn("median_f1_phase_cusum", entry)
            self.assertIn("plausibility_pass_fraction", entry)

    def test_driver_legacy_detector_path(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir, manifest, rec_path = self._build_minimal_session(
                tdp, [("c0", "sandbox_ransom_batched")])
            out_csv = tdp / "p4.csv"
            csv_rows, _recs = p4run.sweep(
                cells_dir=cells_dir, manifest_path=manifest,
                recommendation_path=rec_path, output_csv=out_csv,
                detector="legacy",
            )
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["detector"], "legacy")

    def test_driver_skips_workload_without_recommendation(self):
        # rec lists only ransom; manifest contains a cell whose workload
        # (mem_workingset) has no entry → driver must skip it.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir, manifest, _rec_path = self._build_minimal_session(
                tdp,
                [("c_yes", "sandbox_ransom_batched"),
                 ("c_no",  "app_workingset")],
                recommendations=[("sandbox_ransom_batched", 8, 4)],
            )
            # Recommendation only covers ransom.
            rec_path = cells_dir / "plan03_recommendation.json"
            out_csv = tdp / "p4.csv"
            csv_rows, _recs = p4run.sweep(
                cells_dir=cells_dir, manifest_path=manifest,
                recommendation_path=rec_path, output_csv=out_csv,
            )
            cell_ids = {r["cell_id"] for r in csv_rows}
            self.assertIn("c_yes", cell_ids)
            self.assertNotIn("c_no", cell_ids,
                              "cell without recommendation must be skipped")
            self.assertEqual(len(csv_rows), 1)

    def test_driver_force_flag_required_when_csv_exists(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir, manifest, rec_path = self._build_minimal_session(
                tdp, [("c0", "sandbox_ransom_batched")])
            out_csv = tdp / "p4.csv"
            out_json = tdp / "p4.json"
            out_csv.write_text("pre-existing\n")  # block overwrite
            rc = p4run._main([
                "--cells-dir", str(cells_dir),
                "--manifest", str(manifest),
                "--recommendation", str(rec_path),
                "--output-csv", str(out_csv),
                "--output-json", str(out_json),
            ])
            self.assertEqual(rc, 2,
                              f"expected exit 2 without --force, got {rc}")
            # With --force, the run must succeed and rewrite the file.
            rc2 = p4run._main([
                "--cells-dir", str(cells_dir),
                "--manifest", str(manifest),
                "--recommendation", str(rec_path),
                "--output-csv", str(out_csv),
                "--output-json", str(out_json),
                "--force",
            ])
            self.assertEqual(rc2, 0,
                              f"--force run failed with rc={rc2}")

    def test_driver_marker_tolerance_per_workload(self):
        # batched → tol=1, seq → tol=2 (per PER_WORKLOAD_TOLERANCE).
        self.assertEqual(p4run.PER_WORKLOAD_TOLERANCE["sandbox_ransom_batched"], 1)
        self.assertEqual(p4run.PER_WORKLOAD_TOLERANCE["sandbox_ransom_seq"], 2)
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir, manifest, rec_path = self._build_minimal_session(
                tdp,
                [("c_batched", "sandbox_ransom_batched"),
                 ("c_seq",     "sandbox_ransom_seq")],
                recommendations=[("sandbox_ransom_batched", 8, 4),
                                 ("sandbox_ransom_seq",     8, 4)],
            )
            out_csv = tdp / "p4.csv"
            csv_rows, _recs = p4run.sweep(
                cells_dir=cells_dir, manifest_path=manifest,
                recommendation_path=rec_path, output_csv=out_csv,
            )
            tol_by_cell = {r["cell_id"]: r["tolerance"] for r in csv_rows}
            self.assertEqual(tol_by_cell["c_batched"], 1,
                              f"expected tol=1 for ransom_batched, got "
                              f"{tol_by_cell['c_batched']}")
            self.assertEqual(tol_by_cell["c_seq"], 2,
                              f"expected tol=2 for ransom_seq, got "
                              f"{tol_by_cell['c_seq']}")


# ===========================================================================
# Validator C8 wiring tests
# ===========================================================================

class Plan04ValidatorC8Tests(unittest.TestCase):

    def test_c8_na_when_no_results_file(self):
        # No plan04_segmenter_results.json anywhere → NA (operational=False,
        # but per evaluate_cell: c8=True trivially).
        with tempfile.TemporaryDirectory() as td:
            cells_dir = Path(td) / "cells"
            cells_dir.mkdir()
            c8 = pv._evaluate_plan04_segmenter(cells_dir)
            self.assertFalse(c8["present"])
            self.assertIn("NA", c8["why"])
            # Verify evaluate_cell wiring: when c8 is absent, claim passes
            # with operational=False.
            cell_path = _setup_validator_cell(cells_dir, "cnone",
                                                "sandbox_ransom_batched")
            r = pv.evaluate_cell(cell_path, cells_dir / "work",
                                  0.85, 50, 8, 4,
                                  c7_result=None, c8_result=c8)
            self.assertTrue(r["claims"]["C8_segmenter_quality"]["pass"])
            self.assertFalse(r["claims"]["C8_segmenter_quality"]["operational"])

    def test_c8_na_when_workload_not_evaluated(self):
        # File exists but per_workload is empty / does not list this
        # cell's workload → NA per-cell.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            _write_plan04_results(
                cells_dir / "plan04_segmenter_results.json",
                [{"workload": "other_workload", "family": "steady",
                  "gates": {"gate_pass_steady_stationarity": True}}],
            )
            c8 = pv._evaluate_plan04_segmenter(cells_dir)
            self.assertTrue(c8["present"])
            cell_path = _setup_validator_cell(cells_dir, "cmiss",
                                                "sandbox_ransom_batched")
            r = pv.evaluate_cell(cell_path, cells_dir / "work",
                                  0.85, 50, 8, 4,
                                  c7_result=None, c8_result=c8)
            self.assertTrue(r["claims"]["C8_segmenter_quality"]["pass"],
                            "missing workload entry must yield NA pass")
            self.assertFalse(r["claims"]["C8_segmenter_quality"]["operational"])

    def test_c8_phasic_pass(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            _write_plan04_results(
                cells_dir / "plan04_segmenter_results.json",
                [{"workload": "sandbox_ransom_batched", "family": "phasic",
                  "gates": {"gate_pass_phasic_f1": True}}],
            )
            c8 = pv._evaluate_plan04_segmenter(cells_dir)
            cell_path = _setup_validator_cell(cells_dir, "cph",
                                                "sandbox_ransom_batched")
            r = pv.evaluate_cell(cell_path, cells_dir / "work",
                                  0.85, 50, 8, 4,
                                  c7_result=None, c8_result=c8)
            claim = r["claims"]["C8_segmenter_quality"]
            self.assertTrue(claim["pass"], f"got {claim}")
            self.assertTrue(claim["operational"])

    def test_c8_phasic_fail(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            _write_plan04_results(
                cells_dir / "plan04_segmenter_results.json",
                [{"workload": "sandbox_ransom_batched", "family": "phasic",
                  "gates": {"gate_pass_phasic_f1": False}}],
            )
            c8 = pv._evaluate_plan04_segmenter(cells_dir)
            cell_path = _setup_validator_cell(cells_dir, "cph_fail",
                                                "sandbox_ransom_batched")
            r = pv.evaluate_cell(cell_path, cells_dir / "work",
                                  0.85, 50, 8, 4,
                                  c7_result=None, c8_result=c8)
            claim = r["claims"]["C8_segmenter_quality"]
            self.assertFalse(claim["pass"], f"got {claim}")
            self.assertTrue(claim["operational"])
            self.assertFalse(r["ok"])

    def test_c8_steady_pass(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            _write_plan04_results(
                cells_dir / "plan04_segmenter_results.json",
                [{"workload": "app_workingset", "family": "steady",
                  "gates": {"gate_pass_steady_stationarity": True}}],
            )
            c8 = pv._evaluate_plan04_segmenter(cells_dir)
            cell_path = _setup_validator_cell(cells_dir, "cs",
                                                "app_workingset")
            r = pv.evaluate_cell(cell_path, cells_dir / "work",
                                  0.85, 50, 8, 4,
                                  c7_result=None, c8_result=c8)
            claim = r["claims"]["C8_segmenter_quality"]
            self.assertTrue(claim["pass"], f"got {claim}")
            self.assertTrue(claim["operational"])

    def test_c8_steady_fail(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            _write_plan04_results(
                cells_dir / "plan04_segmenter_results.json",
                [{"workload": "app_workingset", "family": "steady",
                  "gates": {"gate_pass_steady_stationarity": False}}],
            )
            c8 = pv._evaluate_plan04_segmenter(cells_dir)
            cell_path = _setup_validator_cell(cells_dir, "cs_fail",
                                                "app_workingset")
            r = pv.evaluate_cell(cell_path, cells_dir / "work",
                                  0.85, 50, 8, 4,
                                  c7_result=None, c8_result=c8)
            claim = r["claims"]["C8_segmenter_quality"]
            self.assertFalse(claim["pass"], f"got {claim}")
            self.assertTrue(claim["operational"])

    def test_c8_schema_invalid_rejects(self):
        # Missing top-level schema key.
        with tempfile.TemporaryDirectory() as td:
            cells_dir = Path(td) / "cells"
            cells_dir.mkdir()
            (cells_dir / "plan04_segmenter_results.json").write_text(
                json.dumps({"per_workload": []}))
            c8 = pv._evaluate_plan04_segmenter(cells_dir)
            self.assertTrue(c8["present"])
            self.assertTrue(c8.get("load_error"))
        # Missing per_workload key.
        with tempfile.TemporaryDirectory() as td:
            cells_dir = Path(td) / "cells"
            cells_dir.mkdir()
            (cells_dir / "plan04_segmenter_results.json").write_text(
                json.dumps({"schema": p4run.RESULTS_SCHEMA}))
            c8 = pv._evaluate_plan04_segmenter(cells_dir)
            self.assertTrue(c8["present"])
            self.assertTrue(c8.get("load_error"))
            # evaluate_cell with load_error → C8 fail + operational.
            cell_path = _setup_validator_cell(cells_dir, "csch",
                                                "sandbox_ransom_batched")
            r = pv.evaluate_cell(cell_path, cells_dir / "work",
                                  0.85, 50, 8, 4,
                                  c7_result=None, c8_result=c8)
            claim = r["claims"]["C8_segmenter_quality"]
            self.assertFalse(claim["pass"])
            self.assertTrue(claim["operational"])

    def test_analysis_ready_requires_both_c7_c8(self):
        # C7 pass + C8 fail → analysis_ready = False.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            _write_plan04_results(
                cells_dir / "plan04_segmenter_results.json",
                [{"workload": "sandbox_ransom_batched", "family": "phasic",
                  "gates": {"gate_pass_phasic_f1": False}}],
            )
            c8 = pv._evaluate_plan04_segmenter(cells_dir)
            # C7 explicitly pass (operational).
            c7 = {"pass": True, "operational": True,
                   "why": "fixture pass", "artifact": None,
                   "recommendations": []}
            cell_path = _setup_validator_cell(cells_dir, "cboth",
                                                "sandbox_ransom_batched")
            r = pv.evaluate_cell(cell_path, cells_dir / "work",
                                  0.85, 50, 8, 4,
                                  c7_result=c7, c8_result=c8)
            self.assertTrue(r["claims"]["C7_window_hop_recommended"]["pass"])
            self.assertFalse(r["claims"]["C8_segmenter_quality"]["pass"])
            self.assertFalse(r["analysis_ready"],
                              "analysis_ready must require both C7 and C8")

    def test_analysis_ready_na_c8_when_no_file(self):
        # Pre-Plan-04 session (no plan04 file) + Plan 03 ok → analysis_ready=True.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            c8 = pv._evaluate_plan04_segmenter(cells_dir)  # NA
            c7 = {"pass": True, "operational": True,
                   "why": "fixture pass", "artifact": None,
                   "recommendations": []}
            cell_path = _setup_validator_cell(cells_dir, "cprep04",
                                                "sandbox_ransom_batched")
            r = pv.evaluate_cell(cell_path, cells_dir / "work",
                                  0.85, 50, 8, 4,
                                  c7_result=c7, c8_result=c8)
            self.assertTrue(r["claims"]["C8_segmenter_quality"]["pass"])
            self.assertFalse(r["claims"]["C8_segmenter_quality"]["operational"])
            self.assertTrue(r["analysis_ready"],
                             "C8 NA must not block analysis_ready")


if __name__ == "__main__":
    unittest.main(verbosity=2)
