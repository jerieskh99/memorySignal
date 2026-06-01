#!/usr/bin/env python3
"""tests/test_plan05_run.py -- Plan 05 lean-runner pure helpers.

No VM, no subprocess. Validates the matrix math and argv construction:
  - build_cells(): count = W*D*A*R, unique cell_ids
  - parse_workload_commands(): valid, missing '=', placeholder reject
  - inject_duration(): idempotent
  - cell_to_e1_argv(): tmpfs adds --image-dir, keep adds --keep-dumps,
    self-clean adds neither; every cell streams APF + instruments peak disk;
    a command injects --test-command with --duration
  - record_from_run_json(): maps summary -> aggregator record (apf_trajectory)

Run with:
    python3 -m unittest tests.test_plan05_run
    python3 tests/test_plan05_run.py
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import plan05_run as R  # noqa: E402


class TestBuildCells(unittest.TestCase):
    def test_count_and_uniqueness(self):
        cells = R.build_cells(["w1", "w2", "w3"], [120, 600],
                              ["ssd_keep", "tmpfs_keep"], 3)
        self.assertEqual(len(cells), 3 * 2 * 2 * 3)
        self.assertEqual(len({c.cell_id for c in cells}), len(cells))

    def test_default_matrix_is_72(self):
        cells = R.build_cells(R.DEFAULT_WORKLOADS, R.DEFAULT_DURATIONS,
                              R.DEFAULT_ARMS, R.DEFAULT_REPLICATES)
        self.assertEqual(len(cells), 3 * 2 * 4 * 3)


class TestParseWorkloadCommands(unittest.TestCase):
    def test_valid(self):
        d = R.parse_workload_commands(["a=/bin/a --x", "b=/bin/b"])
        self.assertEqual(d, {"a": "/bin/a --x", "b": "/bin/b"})

    def test_missing_equals(self):
        with self.assertRaises(ValueError):
            R.parse_workload_commands(["noequals"])

    def test_placeholder_rejected(self):
        with self.assertRaises(ValueError):
            R.parse_workload_commands(["a=/bin/<RANSOM_PATH>"])

    def test_empty_ok(self):
        self.assertEqual(R.parse_workload_commands([]), {})


class TestInjectDuration(unittest.TestCase):
    def test_appends_both(self):
        out = R.inject_duration("/bin/x", 120)
        self.assertIn("--phase-markers", out)
        self.assertIn("--duration 120", out)

    def test_idempotent(self):
        once = R.inject_duration("/bin/x --duration 99 --phase-markers", 120)
        self.assertEqual(once, "/bin/x --duration 99 --phase-markers")  # untouched


class TestCellToArgv(unittest.TestCase):
    def _argv(self, arm, command=None):
        c = R.Cell("w", 120, arm, 0)
        return R.cell_to_e1_argv(
            c, config="/cfg.json", tmpfs_dir="/dev/shm/d",
            cell_dir=pathlib.Path("/wd/cell"), out_json=pathlib.Path("/wd/cell/r.json"),
            command=command, ssh_target="u@h", ssh_key="", ssh_opts="",
            ram_mb=None, virsh_uri="qemu:///system", producer_script=None,
            no_vm_start=True, skip_ram_check=True, grace_stop_seconds=10)

    def test_every_arm_streams_and_instruments(self):
        for arm in R.ARM_SPECS:
            a = self._argv(arm)
            self.assertIn("--stream-apf", a)
            self.assertIn("--instrument-peak-disk", a)

    def test_tmpfs_adds_image_dir(self):
        a = self._argv("tmpfs_keep")
        self.assertIn("--image-dir", a)
        self.assertEqual(a[a.index("--image-dir") + 1], "/dev/shm/d")

    def test_ssd_omits_image_dir(self):
        self.assertNotIn("--image-dir", self._argv("ssd_keep"))

    def test_keep_adds_keep_dumps(self):
        self.assertIn("--keep-dumps", self._argv("tmpfs_keep"))

    def test_selfclean_omits_keep_dumps(self):
        a = self._argv("ssd_selfclean")
        self.assertNotIn("--keep-dumps", a)
        self.assertNotIn("--image-dir", a)

    def test_command_injects_test_command_with_duration(self):
        a = self._argv("ssd_keep", command="/bin/work")
        self.assertIn("--test-command", a)
        tc = a[a.index("--test-command") + 1]
        self.assertIn("/bin/work", tc)
        self.assertIn("--duration 120", tc)
        self.assertIn("--ssh-target", a)


class TestRecordFromRunJson(unittest.TestCase):
    def test_projection(self):
        c = R.Cell("w", 600, "tmpfs_keep", 2)
        rj = {"summary": {"mean_pmemsave_sec": 3.1, "snapshots_completed": 30,
                          "peak_dump_bytes": 2_000_000, "apf_trajectory": "/p/apf.jsonl"}}
        rec = R.record_from_run_json(c, rj)
        self.assertEqual(rec["workload"], "w")
        self.assertEqual(rec["duration_s"], 600)
        self.assertEqual(rec["arm"], "tmpfs_keep")
        self.assertEqual(rec["replicate"], 2)
        self.assertEqual(rec["mean_pmemsave_sec"], 3.1)
        self.assertEqual(rec["apf_jsonl"], "/p/apf.jsonl")

    def test_missing_summary_is_none_fields(self):
        rec = R.record_from_run_json(R.Cell("w", 120, "ssd_keep", 0), {})
        self.assertIsNone(rec["mean_pmemsave_sec"])
        self.assertIsNone(rec["apf_jsonl"])


if __name__ == "__main__":
    unittest.main()
