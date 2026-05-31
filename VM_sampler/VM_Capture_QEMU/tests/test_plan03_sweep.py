#!/usr/bin/env python3
"""tests/test_plan03_sweep.py -- Plan 03 kernel/sweep/aggregator/validator tests.

No VM required. Validates:
  - plan03_metric_kernel.score(): status branches, determinism, F1, CV, stationarity
  - plan03_sweep.sweep(): CSV columns, status/workload filters, traj_sha256,
    deterministic output, --max-cells cap
  - plan03_aggregate.aggregate(): gates G1-G5, winner picker, regression guard,
    null-winner emission, summary schema
  - plan02_validate_session._evaluate_plan03_recommendation: C7 NA/pass/fail
    states + integration with sweep+aggregate

Run with:
    python3 -m unittest tests.test_plan03_sweep
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

# Make the parent dir importable when running this file directly.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import plan02_manifest as mf  # noqa: E402
import plan02_schema as sc  # noqa: E402
import plan02_validate_session as pv  # noqa: E402
import plan03_aggregate as agg  # noqa: E402
import plan03_metric_kernel as kernel  # noqa: E402
import plan03_sweep as sw  # noqa: E402


# Fixed canonical column list. Mirrors plan03_sweep.CSV_FIELDS.
SWEEP_FIELDS = [
    "cell_id", "workload", "iv_ms", "duration_s", "replicate",
    "n_pairs", "window", "hop", "hop_ratio", "n_windows",
    "apf_mean", "apf_std", "cv_workingset", "f1_phase",
    "cepstral_peak_idx", "ceps_peak_snr_db", "stat_pass_frac",
    "coverage_ratio", "traj_sha256", "status",
]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_apf_jsonl(path: Path, values: list[float],
                     n_pairs_expected: int | None = None,
                     n_ok: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, v in enumerate(values):
        lines.append(json.dumps({"seq": i, "apf": v}))
    lines.append(json.dumps({
        "final": True,
        "n_pairs_expected": (len(values) if n_pairs_expected is None
                              else n_pairs_expected),
        "n_ok": (len(values) if n_ok is None else n_ok),
        "n_failed": 0,
        "gap_seqs": [],
    }))
    path.write_text("\n".join(lines) + "\n")


def _make_manifest_with(cells: list[tuple[str, str, int, int, str]],
                        td: Path) -> Path:
    """cells is a list of (cell_id, workload, iv_ms, duration_s, status).
    Returns the manifest CSV path."""
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


def _write_sweep_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        w.writeheader()
        for r in rows:
            full = {k: r.get(k, "") for k in SWEEP_FIELDS}
            w.writerow(full)


def _write_cell_json_for_baseline(cells_dir: Path, cell_id: str,
                                  f1_phase: float | None = None,
                                  cv_workingset: float | None = None) -> None:
    """Emit a minimal cell_<id>.json carrying the analyzer_outputs block
    that the Delta-5 regression guard reads."""
    cells_dir.mkdir(parents=True, exist_ok=True)
    ao: dict = {}
    if f1_phase is not None:
        ao["f1_phase"] = f1_phase
    if cv_workingset is not None:
        ao["cv_workingset"] = cv_workingset
    (cells_dir / f"cell_{cell_id}.json").write_text(json.dumps({
        "schema_version": 2,
        "analyzer_outputs": ao,
    }))


def _make_workingset_row(*, cell_id: str, window: int, hop: int,
                         cv: float, stat_pass_frac: float = 0.85,
                         n_windows: int = 10, duration_s: int = 60,
                         coverage_ratio: float = 2.5,
                         n_pairs: int = 100,
                         status: str = "ok") -> dict:
    return {
        "cell_id": cell_id, "workload": "app_workingset",
        "iv_ms": 100, "duration_s": duration_s, "replicate": 0,
        "n_pairs": n_pairs, "window": window, "hop": hop,
        "hop_ratio": hop / window, "n_windows": n_windows,
        "apf_mean": 0.5, "apf_std": 0.1, "cv_workingset": cv,
        "f1_phase": "", "cepstral_peak_idx": 5, "ceps_peak_snr_db": 10.0,
        "stat_pass_frac": stat_pass_frac, "coverage_ratio": coverage_ratio,
        "traj_sha256": "deadbeefcafebabe", "status": status,
    }


def _make_ransom_row(*, cell_id: str, window: int, hop: int,
                     f1: float = 1.0, stat_pass_frac: float = 0.85,
                     n_windows: int = 10, coverage_ratio: float = 2.5,
                     ceps_peak_snr_db: float = 10.0,
                     n_pairs: int = 100,
                     status: str = "ok") -> dict:
    return {
        "cell_id": cell_id, "workload": "sandbox_ransom_batched",
        "iv_ms": 100, "duration_s": 60, "replicate": 0,
        "n_pairs": n_pairs, "window": window, "hop": hop,
        "hop_ratio": hop / window, "n_windows": n_windows,
        "apf_mean": 0.5, "apf_std": 0.1, "cv_workingset": "",
        "f1_phase": f1, "cepstral_peak_idx": 5,
        "ceps_peak_snr_db": ceps_peak_snr_db,
        "stat_pass_frac": stat_pass_frac, "coverage_ratio": coverage_ratio,
        "traj_sha256": "deadbeefcafebabe", "status": status,
    }


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------

class Plan03MetricKernelTests(unittest.TestCase):

    def test_score_short_traj_returns_skip_short(self):
        r = kernel.score([0.1] * 5, window=8, hop=4)
        self.assertEqual(r["status"], "skip:short")
        for k in ("apf_mean", "apf_std", "cv_workingset", "f1_phase",
                  "cepstral_peak_idx", "ceps_peak_snr_db", "stat_pass_frac"):
            self.assertIsNone(r[k], f"{k} must be None on skip:short")

    def test_score_steady_ramp_deterministic(self):
        traj = [i / 100.0 for i in range(50)]
        r1 = kernel.score(traj, window=8, hop=4, workload_type="steady")
        r2 = kernel.score(traj, window=8, hop=4, workload_type="steady")
        self.assertEqual(r1["status"], "ok")
        self.assertGreaterEqual(r1["n_windows"], 5)
        self.assertIsNotNone(r1["cv_workingset"])
        self.assertTrue(math.isfinite(r1["cv_workingset"]))
        self.assertGreater(r1["cv_workingset"], 0.0)
        self.assertEqual(r1, r2, "kernel must be deterministic for identical inputs")

    def test_score_phasic_with_markers_computes_f1(self):
        traj = [0.0] * 30 + [1.0] * 10 + [0.0] * 30
        r = kernel.score(traj, window=8, hop=4,
                          phase_marker_indices=[30, 40],
                          workload_type="phasic")
        self.assertEqual(r["status"], "ok")
        self.assertIsNotNone(r["f1_phase"])
        self.assertGreater(r["f1_phase"], 0.0)

    def test_score_steady_workload_skips_f1(self):
        traj = [0.0] * 30 + [1.0] * 10 + [0.0] * 30
        r = kernel.score(traj, window=8, hop=4,
                          phase_marker_indices=[30, 40],
                          workload_type="steady")
        self.assertIsNone(r["f1_phase"],
                         "F1 must not be computed for steady workloads")

    def test_score_phasic_no_markers_skips_f1(self):
        traj = [0.0] * 30 + [1.0] * 10 + [0.0] * 30
        r = kernel.score(traj, window=8, hop=4,
                          phase_marker_indices=None,
                          workload_type="phasic")
        self.assertIsNone(r["f1_phase"],
                         "F1 must not be computed when markers are absent")

    def test_score_constant_traj_stationarity_all_pass(self):
        r = kernel.score([0.5] * 40, window=8, hop=4)
        # global_std == 0 -> stationarity_per_window returns 1.0 trivially.
        self.assertEqual(r["stat_pass_frac"], 1.0)

    def test_score_window_equals_n_pairs_yields_one_window(self):
        traj = [i / 20.0 for i in range(20)]
        r = kernel.score(traj, window=20, hop=10)
        self.assertEqual(r["n_windows"], 1)

    def test_score_handles_nan_input(self):
        traj = [0.1, 0.2, math.nan, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        r = kernel.score(traj, window=8, hop=4)
        # NaN entries propagate through the FFT and are caught either
        # by the finite-check (skip:nan) or by the broad except (error:*).
        self.assertTrue(r["status"].startswith("skip:nan")
                        or r["status"].startswith("error:"),
                        f"got status={r['status']!r}")


# ---------------------------------------------------------------------------
# Sweep tests
# ---------------------------------------------------------------------------

class Plan03SweepTests(unittest.TestCase):

    def _build_cell(self, cells_dir: Path, cell_id: str, workload: str,
                    traj: list[float]) -> None:
        wd = cells_dir / "work" / cell_id
        _write_apf_jsonl(wd / "apf_trajectory.jsonl", traj)

    def test_sweep_writes_csv_with_expected_columns(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            manifest = _make_manifest_with([
                ("cell_a", "sandbox_ransom_batched", 100, 60, "ok"),
                ("cell_b", "app_workingset",         100, 60, "ok"),
            ], tdp)
            traj = [0.1 + 0.02 * i for i in range(15)]
            self._build_cell(cells_dir, "cell_a", "sandbox_ransom_batched", traj)
            self._build_cell(cells_dir, "cell_b", "app_workingset", traj)

            out_csv = tdp / "sweep.csv"
            n = sw.sweep(
                cells_dir=cells_dir, manifest_path=manifest,
                output_csv=out_csv, windows=[8], hop_ratios=[0.5],
            )
            self.assertEqual(n, 2)
            with out_csv.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, SWEEP_FIELDS)
            self.assertEqual(len(rows), 2,
                             f"expected 2 (cells) x 1 (combo) = 2 rows, got {len(rows)}")

    def test_sweep_status_filter_excludes_failed_cells(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            manifest = _make_manifest_with([
                ("ok_cell",   "sandbox_ransom_batched", 100, 60, "ok"),
                ("fail_cell", "sandbox_ransom_batched", 100, 60, "failed"),
            ], tdp)
            traj = [0.1 + 0.02 * i for i in range(15)]
            self._build_cell(cells_dir, "ok_cell", "sandbox_ransom_batched", traj)
            self._build_cell(cells_dir, "fail_cell", "sandbox_ransom_batched", traj)

            out_csv = tdp / "sweep.csv"
            n = sw.sweep(
                cells_dir=cells_dir, manifest_path=manifest,
                output_csv=out_csv, windows=[8], hop_ratios=[0.5],
                status_filter={"ok"},
            )
            self.assertEqual(n, 1, "only the ok cell should be processed")
            with out_csv.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cell_id"], "ok_cell")

    def test_sweep_workload_filter_regex_applied(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            manifest = _make_manifest_with([
                ("ransom_c",  "sandbox_ransom_batched", 100, 60, "ok"),
                ("ws_c",      "app_workingset",         100, 60, "ok"),
            ], tdp)
            traj = [0.1 + 0.02 * i for i in range(15)]
            self._build_cell(cells_dir, "ransom_c", "sandbox_ransom_batched", traj)
            self._build_cell(cells_dir, "ws_c", "app_workingset", traj)

            out_csv = tdp / "sweep.csv"
            n = sw.sweep(
                cells_dir=cells_dir, manifest_path=manifest,
                output_csv=out_csv, windows=[8], hop_ratios=[0.5],
                workload_filter="ransom",
            )
            self.assertEqual(n, 1)
            with out_csv.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cell_id"], "ransom_c")

    def test_sweep_traj_sha256_stable(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            manifest = _make_manifest_with([
                ("cell_x", "app_workingset", 100, 60, "ok"),
            ], tdp)
            traj = [0.05 * i for i in range(20)]
            self._build_cell(cells_dir, "cell_x", "app_workingset", traj)

            out_csv = tdp / "sweep.csv"
            sw.sweep(cells_dir=cells_dir, manifest_path=manifest,
                     output_csv=out_csv, windows=[8], hop_ratios=[0.5])
            with out_csv.open() as f:
                sha_first = next(csv.DictReader(f))["traj_sha256"]

            out_csv2 = tdp / "sweep2.csv"
            sw.sweep(cells_dir=cells_dir, manifest_path=manifest,
                     output_csv=out_csv2, windows=[8], hop_ratios=[0.5])
            with out_csv2.open() as f:
                sha_second = next(csv.DictReader(f))["traj_sha256"]

            self.assertNotEqual(sha_first, "")
            self.assertEqual(sha_first, sha_second,
                             "traj_sha256 must be stable across runs")

    def test_sweep_deterministic_output(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            manifest = _make_manifest_with([
                ("d_a", "sandbox_ransom_batched", 100, 60, "ok"),
                ("d_b", "app_workingset",         100, 60, "ok"),
            ], tdp)
            traj_a = [0.1 + 0.02 * i for i in range(15)]
            traj_b = [0.2 + 0.01 * i for i in range(15)]
            self._build_cell(cells_dir, "d_a", "sandbox_ransom_batched", traj_a)
            self._build_cell(cells_dir, "d_b", "app_workingset", traj_b)

            out1 = tdp / "out1.csv"
            out2 = tdp / "out2.csv"
            sw.sweep(cells_dir=cells_dir, manifest_path=manifest,
                     output_csv=out1, windows=[8], hop_ratios=[0.5])
            sw.sweep(cells_dir=cells_dir, manifest_path=manifest,
                     output_csv=out2, windows=[8], hop_ratios=[0.5])
            self.assertEqual(out1.read_bytes(), out2.read_bytes(),
                             "sweep output must be byte-identical for identical input")

    def test_sweep_max_cells_caps_iteration(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            cells_spec = [(f"c{i}", "app_workingset", 100, 60, "ok")
                          for i in range(5)]
            manifest = _make_manifest_with(cells_spec, tdp)
            traj = [0.1 + 0.02 * i for i in range(15)]
            for cid, *_ in cells_spec:
                self._build_cell(cells_dir, cid, "app_workingset", traj)

            out_csv = tdp / "sweep.csv"
            n = sw.sweep(cells_dir=cells_dir, manifest_path=manifest,
                         output_csv=out_csv, windows=[8], hop_ratios=[0.5],
                         max_cells=3)
            self.assertEqual(n, 3)
            with out_csv.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------

class Plan03AggregateTests(unittest.TestCase):

    def test_aggregate_gates_g1_stationarity_boundary(self):
        # Median stat_pass_frac == 0.80 -> G1 pass (>=). 0.79 -> fail.
        for spf, expected in [(0.80, True), (0.79, False)]:
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                csv_path = tdp / "sweep.csv"
                rows = [
                    _make_workingset_row(cell_id=f"c{i}", window=8, hop=4,
                                         cv=0.04, stat_pass_frac=spf,
                                         n_windows=10)
                    for i in range(3)
                ]
                _write_sweep_csv(csv_path, rows)
                summary = agg.aggregate(csv_path, tdp / "summary.json")
                grp = summary["by_group"][0]
                self.assertEqual(grp["G1_pass"], expected,
                                 f"G1 at spf={spf}: expected {expected}, got {grp}")

    def test_aggregate_gates_g3_ransom_snr_boundary(self):
        # G3 ransom: median ceps_peak_snr_db >= 4.5 dB (v3 D-83
        # recalibration; was 5.0). Test the boundary (4.5 -> pass;
        # 4.49 -> fail). F1 is intentionally window-independent and is
        # NOT part of the gate any more.
        for snr, expected in [(4.5, True), (4.49, False), (10.0, True)]:
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                csv_path = tdp / "sweep.csv"
                rows = [
                    _make_ransom_row(cell_id=f"c{i}", window=8, hop=4,
                                     ceps_peak_snr_db=snr,
                                     stat_pass_frac=0.95, n_windows=10)
                    for i in range(3)
                ]
                _write_sweep_csv(csv_path, rows)
                summary = agg.aggregate(csv_path, tdp / "summary.json")
                grp = summary["by_group"][0]
                self.assertEqual(grp["G3_pass"], expected,
                                 f"G3 ransom at snr={snr}: expected "
                                 f"{expected}, got {grp}")

    def test_aggregate_g3_ransom_ignores_low_n_windows_cells(self):
        # SNR=12.0 for the 2 cells with n_windows>=5, SNR=0.0 for a cell
        # with n_windows=4 (excluded). Median over the eligible 2 is
        # 12.0 -> pass. Including the n_windows=4 cell would drag the
        # median below threshold; the gate must skip it.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "sweep.csv"
            rows = [
                _make_ransom_row(cell_id="c0", window=8, hop=4,
                                 ceps_peak_snr_db=12.0, n_windows=10,
                                 stat_pass_frac=0.95, n_pairs=60),
                _make_ransom_row(cell_id="c1", window=8, hop=4,
                                 ceps_peak_snr_db=12.0, n_windows=10,
                                 stat_pass_frac=0.95, n_pairs=60),
                _make_ransom_row(cell_id="c2", window=8, hop=4,
                                 ceps_peak_snr_db=0.0, n_windows=4,
                                 stat_pass_frac=0.95, n_pairs=60),
            ]
            _write_sweep_csv(csv_path, rows)
            summary = agg.aggregate(csv_path, tdp / "summary.json")
            grp = summary["by_group"][0]
            self.assertTrue(grp["G3_pass"],
                            f"G3 ransom must ignore low-n_windows cells: {grp}")

    def test_aggregate_gates_g3_workingset_short_d(self):
        # short_duration cell (duration_s <= 120). v3 D-83 ceiling
        # raised 0.05 -> 0.30: cv=0.30 -> pass; cv=0.31 -> fail.
        for cv, expected in [(0.30, True), (0.31, False)]:
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                csv_path = tdp / "sweep.csv"
                rows = [
                    _make_workingset_row(cell_id=f"c{i}", window=8, hop=4,
                                         cv=cv, duration_s=60)
                    for i in range(3)
                ]
                _write_sweep_csv(csv_path, rows)
                summary = agg.aggregate(csv_path, tdp / "summary.json")
                grp = summary["by_group"][0]
                self.assertEqual(grp["G3_pass"], expected,
                                 f"G3 at cv={cv}: expected {expected}, got {grp}")

    def test_g5_excludes_ineligible_cells_from_denominator(self):
        # New G5 denominator: cells with n_pairs >= W + 4*H. A cell
        # whose trajectory is too short for this combo (n_pairs<min)
        # MUST NOT be counted against the gate. Otherwise short-d cells
        # would make G5 unreachable for larger windows on v2 data.
        #
        # 3 eligible cells (n_pairs=60, all delivering n_windows>=5) +
        # 2 ineligible cells (n_pairs=10 < 8+4*4=24). Eligible fraction
        # = 3/3 = 1.0 (>=0.80) -> pass. Naive denominator (5) would
        # yield 3/5 = 0.6 -> fail.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "sweep.csv"
            rows: list[dict] = []
            for i in range(3):
                rows.append(_make_workingset_row(
                    cell_id=f"long_{i}", window=8, hop=4,
                    cv=0.04, stat_pass_frac=0.95, n_windows=10,
                    n_pairs=60))
            for i in range(2):
                rows.append(_make_workingset_row(
                    cell_id=f"short_{i}", window=8, hop=4,
                    cv=0.04, stat_pass_frac=0.95, n_windows=0,
                    n_pairs=10, status="skip:short"))
            _write_sweep_csv(csv_path, rows)
            summary = agg.aggregate(csv_path, tdp / "summary.json")
            grp = summary["by_group"][0]
            self.assertTrue(grp["G5_pass"],
                            f"G5 must use eligible denominator: {grp}")
            self.assertEqual(grp["n_cells_eligible_for_5w"], 3,
                             f"only the 3 long cells are eligible: {grp}")
            self.assertEqual(grp["n_cells_delivered_5w"], 3)
            self.assertAlmostEqual(grp["g5_eligible_fraction"], 1.0)

    def test_g5_fails_when_eligible_cells_underdeliver(self):
        # 5 eligible cells; only 3 delivered (the 0.80 floor demands 4+).
        # The 0.80 catches numerical anomalies among physically-eligible
        # cells (NaN cepstrum, skip:nan).
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "sweep.csv"
            rows: list[dict] = []
            for i in range(3):
                rows.append(_make_workingset_row(
                    cell_id=f"ok_{i}", window=8, hop=4,
                    cv=0.04, stat_pass_frac=0.95, n_windows=10,
                    n_pairs=60))
            for i in range(2):
                rows.append(_make_workingset_row(
                    cell_id=f"nan_{i}", window=8, hop=4,
                    cv=0.04, stat_pass_frac=0.95, n_windows=0,
                    n_pairs=60, status="skip:nan"))
            _write_sweep_csv(csv_path, rows)
            summary = agg.aggregate(csv_path, tdp / "summary.json")
            grp = summary["by_group"][0]
            self.assertFalse(grp["G5_pass"],
                             f"G5 must fail at 3/5 = 0.60: {grp}")
            self.assertEqual(grp["n_cells_eligible_for_5w"], 5)
            self.assertEqual(grp["n_cells_delivered_5w"], 3)

    def test_aggregate_gates_g4_hop_ratio(self):
        # hop=5, window=8 -> hop > window/2 -> G4 fail regardless of metrics.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "sweep.csv"
            rows = [
                _make_workingset_row(cell_id=f"c{i}", window=8, hop=5,
                                     cv=0.04, stat_pass_frac=0.95)
                for i in range(3)
            ]
            _write_sweep_csv(csv_path, rows)
            summary = agg.aggregate(csv_path, tdp / "summary.json")
            grp = summary["by_group"][0]
            self.assertFalse(grp["G4_pass"], "hop > window/2 must fail G4")
            self.assertFalse(grp["all_gates_pass"])

    def test_aggregate_winner_smallest_window(self):
        # Two passing combos: W=8 and W=16. Winner must be W=8.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "sweep.csv"
            rows: list[dict] = []
            for W in (8, 16):
                for i in range(3):
                    rows.append(_make_workingset_row(
                        cell_id=f"c{i}_W{W}", window=W, hop=W // 2,
                        cv=0.04, stat_pass_frac=0.85, n_windows=10,
                    ))
            _write_sweep_csv(csv_path, rows)
            summary = agg.aggregate(csv_path, tdp / "summary.json")
            recs = summary["recommendations"]
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["recommended_window"], 8,
                             f"smallest W must win: {recs[0]}")
            self.assertEqual(recs[0]["recommended_hop"], 4)

    def test_aggregate_winner_v2_regression_guard(self):
        # ransom cell baseline F1 = 0.90 (< 0.95 baseline) read from
        # cell_<id>.json::analyzer_outputs.f1_phase. Sweep rows pass
        # G1-G5 (high SNR, high stat_pass_frac, n_windows=10). Guard
        # should trip and clear passes_acceptance.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "sweep.csv"
            cells_dir = tdp / "cells"
            rows = [
                _make_ransom_row(cell_id=f"c{i}", window=8, hop=4,
                                 ceps_peak_snr_db=12.0,
                                 coverage_ratio=2.5)
                for i in range(3)
            ]
            for i in range(3):
                _write_cell_json_for_baseline(cells_dir, f"c{i}",
                                              f1_phase=0.90)
            _write_sweep_csv(csv_path, rows)
            summary = agg.aggregate(csv_path, tdp / "summary.json",
                                    cells_dir=cells_dir)
            recs = summary["recommendations"]
            self.assertEqual(len(recs), 1)
            self.assertTrue(recs[0]["degrades_v2_baseline"])
            self.assertFalse(recs[0]["passes_acceptance"],
                             "passes_acceptance must reflect baseline regression")

    def test_aggregate_winner_regression_guard_passes_when_baseline_high(self):
        # Same fixture but f1_phase baseline = 0.98 -> guard does not trip.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "sweep.csv"
            cells_dir = tdp / "cells"
            rows = [
                _make_ransom_row(cell_id=f"c{i}", window=8, hop=4,
                                 ceps_peak_snr_db=12.0,
                                 coverage_ratio=2.5)
                for i in range(3)
            ]
            for i in range(3):
                _write_cell_json_for_baseline(cells_dir, f"c{i}",
                                              f1_phase=0.98)
            _write_sweep_csv(csv_path, rows)
            summary = agg.aggregate(csv_path, tdp / "summary.json",
                                    cells_dir=cells_dir)
            recs = summary["recommendations"]
            self.assertEqual(len(recs), 1)
            self.assertFalse(recs[0]["degrades_v2_baseline"])
            self.assertTrue(recs[0]["passes_acceptance"])

    def test_aggregate_no_combo_passes_writes_best_feasible_winner(self):
        # G4 fail forces no combo to clear all 5 gates. The aggregator
        # falls back to a best-feasible pick that records which gate(s)
        # were relaxed; passes_acceptance stays False. This replaces
        # the prior null-winner contract so Plan 04 / C7 can still see
        # a defensible default on partially-passing v2 data.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "sweep.csv"
            rows = [
                _make_workingset_row(cell_id=f"c{i}", window=8, hop=5,
                                     cv=0.04, stat_pass_frac=0.95)
                for i in range(3)
            ]
            _write_sweep_csv(csv_path, rows)
            summary = agg.aggregate(csv_path, tdp / "summary.json")
            recs = summary["recommendations"]
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["recommended_window"], 8,
                             "best-feasible must surface the single combo")
            self.assertEqual(recs[0]["recommended_hop"], 5)
            self.assertFalse(recs[0]["passes_acceptance"],
                             "passes_acceptance must stay False on fallback")
            self.assertIn("best-feasible", recs[0]["rationale"].lower())
            self.assertIn("g4", recs[0]["rationale"].lower(),
                          "rationale must name the relaxed gate (G4)")

    def test_aggregate_no_combos_at_all_yields_null(self):
        # If the workload bucket is empty (no rows of that workload),
        # winner stays None. Recommendations list still emits an entry.
        # Test by writing only ransom rows but asking for a workingset
        # bucket via a parallel path -- we synthesize the edge case by
        # writing zero rows of a given workload and inspecting an
        # in-process call to _pick_winner directly.
        winner, reason = agg._pick_winner([], "mem_workingset_sweep_v2")
        self.assertIsNone(winner)
        self.assertIn("empty bucket", reason)

    def test_aggregate_csv_to_summary_json(self):
        # 6-row sweep.csv -> summary.json with expected schema fields.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "sweep.csv"
            rows: list[dict] = []
            for W in (8, 16):
                for i in range(3):
                    rows.append(_make_workingset_row(
                        cell_id=f"c{i}_W{W}", window=W, hop=W // 2,
                        cv=0.04, stat_pass_frac=0.85, n_windows=10,
                    ))
            _write_sweep_csv(csv_path, rows)
            summary_json = tdp / "summary.json"
            agg.aggregate(csv_path, summary_json)
            with summary_json.open() as f:
                payload = json.load(f)
            for k in ("schema", "generated_at", "sweep_csv_sha256",
                      "n_cells_used", "n_combos", "by_group", "recommendations"):
                self.assertIn(k, payload, f"summary missing key {k}")
            self.assertEqual(payload["schema"], agg.SUMMARY_SCHEMA)
            self.assertEqual(payload["n_combos"], 2,
                             "two distinct (workload, W, H) combos expected")


# ---------------------------------------------------------------------------
# Validator C7 tests
# ---------------------------------------------------------------------------

class Plan03ValidatorC7Tests(unittest.TestCase):

    def test_c7_na_when_no_recommendation_file(self):
        with tempfile.TemporaryDirectory() as td:
            cells_dir = Path(td) / "cells"
            cells_dir.mkdir()
            r = pv._evaluate_plan03_recommendation(cells_dir)
            self.assertTrue(r["pass"])
            self.assertFalse(r["operational"])
            self.assertIn("NA", r["why"])

    def test_c7_fails_when_passes_acceptance_false(self):
        with tempfile.TemporaryDirectory() as td:
            cells_dir = Path(td) / "cells"
            cells_dir.mkdir()
            payload = {
                "schema": "plan03.window_hop_recommendations.v1",
                "recommendations": [
                    {"workload": "a", "recommended_window": 8,
                     "recommended_hop": 4, "passes_acceptance": True,
                     "degrades_v2_baseline": False},
                    {"workload": "b", "recommended_window": 16,
                     "recommended_hop": 8, "passes_acceptance": False,
                     "degrades_v2_baseline": False},
                ],
            }
            (cells_dir / "plan03_recommendation.json").write_text(
                json.dumps(payload))
            r = pv._evaluate_plan03_recommendation(cells_dir)
            self.assertFalse(r["pass"])
            self.assertTrue(r["operational"])
            self.assertIn("b", r["why"])  # mentions the failing workload

    def test_c7_passes_when_all_entries_pass(self):
        with tempfile.TemporaryDirectory() as td:
            cells_dir = Path(td) / "cells"
            cells_dir.mkdir()
            payload = {
                "schema": "plan03.window_hop_recommendations.v1",
                "recommendations": [
                    {"workload": "a", "recommended_window": 8,
                     "recommended_hop": 4, "passes_acceptance": True,
                     "degrades_v2_baseline": False},
                ],
            }
            (cells_dir / "plan03_recommendation.json").write_text(
                json.dumps(payload))
            r = pv._evaluate_plan03_recommendation(cells_dir)
            self.assertTrue(r["pass"])
            self.assertTrue(r["operational"])

    def test_c7_schema_check_rejects_malformed_json(self):
        # File present but the recommendations key is missing -> fail with
        # operational=True and a why explaining the schema/empty problem.
        with tempfile.TemporaryDirectory() as td:
            cells_dir = Path(td) / "cells"
            cells_dir.mkdir()
            (cells_dir / "plan03_recommendation.json").write_text(json.dumps({
                "schema": "plan03.window_hop_recommendations.v1",
                # no "recommendations" key
            }))
            r = pv._evaluate_plan03_recommendation(cells_dir)
            self.assertFalse(r["pass"])
            self.assertTrue(r["operational"])
            self.assertTrue(r["why"], "why must be non-empty on schema error")

        # Wrong schema string also rejected.
        with tempfile.TemporaryDirectory() as td:
            cells_dir = Path(td) / "cells"
            cells_dir.mkdir()
            (cells_dir / "plan03_recommendation.json").write_text(json.dumps({
                "schema": "bogus.schema.v0",
                "recommendations": [],
            }))
            r = pv._evaluate_plan03_recommendation(cells_dir)
            self.assertFalse(r["pass"])
            self.assertIn("schema", r["why"].lower())


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class Plan03IntegrationTests(unittest.TestCase):

    def test_full_sweep_then_aggregate_then_validate(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cells_dir = tdp / "cells"
            cells_dir.mkdir()
            # 1 ransom + 1 workingset cell, both passing manifest status='ok'.
            manifest = _make_manifest_with([
                ("ic_ransom", "sandbox_ransom_batched", 100, 60, "ok"),
                ("ic_ws",     "app_workingset",         100, 60, "ok"),
            ], tdp)
            traj = [0.1 + 0.02 * i for i in range(15)]
            _write_apf_jsonl(cells_dir / "work" / "ic_ransom" /
                             "apf_trajectory.jsonl", traj)
            _write_apf_jsonl(cells_dir / "work" / "ic_ws" /
                             "apf_trajectory.jsonl", traj)

            # Sweep -> aggregate -> recommendation
            sweep_csv = tdp / "sweep.csv"
            sw.sweep(cells_dir=cells_dir, manifest_path=manifest,
                     output_csv=sweep_csv, windows=[8], hop_ratios=[0.5])
            summary_json = tdp / "summary.json"
            rec_json = cells_dir / "plan03_recommendation.json"
            agg.aggregate(sweep_csv, summary_json, recommendation_json=rec_json)
            self.assertTrue(rec_json.is_file())

            with rec_json.open() as f:
                payload = json.load(f)
            self.assertEqual(payload["schema"],
                             "plan03.window_hop_recommendations.v1")
            self.assertGreaterEqual(len(payload["recommendations"]), 1)

            # Validator should now find the recommendation artifact and
            # produce a deterministic C7 result whose pass field equals
            # all(passes_acceptance) across the workloads.
            c7 = pv._evaluate_plan03_recommendation(cells_dir)
            self.assertTrue(c7["operational"])
            expected = all(r.get("passes_acceptance")
                           for r in payload["recommendations"])
            self.assertEqual(c7["pass"], expected,
                             f"C7 pass={c7['pass']} but all-passes={expected} "
                             f"(recs={payload['recommendations']})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
