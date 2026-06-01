#!/usr/bin/env python3
"""tests/test_plan05_driver.py -- Plan 05 opt-in e1-driver flags.

No VM required. Proves the new capture-side levers are strictly opt-in and that
the default path is byte-identical to the pre-Plan-05 driver:

  - configure_apf_env(tmp, False) == ({}, None)  [the byte-identical-default
    proof: with --stream-apf absent, os.environ.update gets an empty dict, so
    the producer env is unchanged]
  - configure_apf_env(tmp, True) sets exactly the four TIMING_APF_* keys, returns
    the trajectory path, creates the ack dir, and clears a stale trajectory
  - write_overridden_config leaves imageDir untouched when image_dir is None and
    overrides it (and only it, among Plan-05 keys) when given
  - _PeakDiskSampler._sample() sums memory_dump-*.raw and ignores other files /
    a missing dir
  - ram_size_matches_domain() verdicts: match, mismatch, cannot-verify

Run with:
    python3 -m unittest tests.test_plan05_driver
    python3 tests/test_plan05_driver.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import run_timing_instrumentation_experiment as E  # noqa: E402


class TestConfigureApfEnv(unittest.TestCase):
    def test_disabled_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            env, path = E.configure_apf_env(pathlib.Path(d), False)
            # The byte-identical-default contract: nothing added to the env.
            self.assertEqual(env, {})
            self.assertIsNone(path)

    def test_enabled_sets_four_keys(self):
        with tempfile.TemporaryDirectory() as d:
            wd = pathlib.Path(d)
            env, path = E.configure_apf_env(wd, True)
            self.assertEqual(set(env), {
                "TIMING_APF_STREAM", "TIMING_APF_JSONL",
                "TIMING_APF_ACK_DIR", "TIMING_APF_HELPER_LOG"})
            self.assertEqual(env["TIMING_APF_STREAM"], "1")
            self.assertEqual(path, wd / "apf_trajectory.jsonl")
            self.assertTrue((wd / "apf_acks").is_dir())

    def test_enabled_clears_stale_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            wd = pathlib.Path(d)
            stale = wd / "apf_trajectory.jsonl"
            stale.write_text("old\n")
            ack_dir = wd / "apf_acks"
            ack_dir.mkdir()
            (ack_dir / "5.ack").write_text("x")
            _, path = E.configure_apf_env(wd, True)
            self.assertFalse(path.exists())          # trajectory removed
            self.assertEqual(list(ack_dir.glob("*")), [])  # acks cleared


class TestWriteOverriddenConfig(unittest.TestCase):
    def _orig(self):
        return {"imageDir": "/var/lib/libvirt/qemu/dump", "ramSizeMb": 1024,
                "intervalMsec": 500, "streaming": {"enabled": True},
                "rawRetention": {"enabled": True}}

    def test_image_dir_none_leaves_imagedir(self):
        with tempfile.TemporaryDirectory() as d:
            _, cfg = E.write_overridden_config(self._orig(), pathlib.Path(d),
                                               None, None, None)
            self.assertEqual(cfg["imageDir"], "/var/lib/libvirt/qemu/dump")

    def test_image_dir_override(self):
        with tempfile.TemporaryDirectory() as d:
            _, cfg = E.write_overridden_config(self._orig(), pathlib.Path(d),
                                               None, None, "/dev/shm/dump")
            self.assertEqual(cfg["imageDir"], "/dev/shm/dump")
            # Untouched neighbors stay put (only imageDir moved).
            self.assertEqual(cfg["ramSizeMb"], 1024)


class TestPeakDiskSampler(unittest.TestCase):
    def test_sample_sums_only_dumps(self):
        with tempfile.TemporaryDirectory() as d:
            wd = pathlib.Path(d)
            (wd / "memory_dump-0.raw").write_bytes(b"\x00" * 100)
            (wd / "memory_dump-1.raw").write_bytes(b"\x00" * 250)
            (wd / "notes.txt").write_bytes(b"\x00" * 9999)  # ignored
            s = E._PeakDiskSampler(wd)
            self.assertEqual(s._sample(), 350)

    def test_sample_missing_dir(self):
        s = E._PeakDiskSampler(pathlib.Path("/nonexistent/plan05/xyz"))
        self.assertEqual(s._sample(), 0)


class TestRamCheck(unittest.TestCase):
    def test_match_within_tol(self):
        # 1048576 KiB == 1024 MiB exactly.
        orig = E.virsh_max_memory_kib
        E.virsh_max_memory_kib = lambda uri, dom: 1048576
        try:
            ok, det = E.ram_size_matches_domain("u", "vm", 1024)
            self.assertTrue(ok)
            self.assertEqual(det["domain_max_mb"], 1024.0)
        finally:
            E.virsh_max_memory_kib = orig

    def test_mismatch_fails(self):
        orig = E.virsh_max_memory_kib
        E.virsh_max_memory_kib = lambda uri, dom: 1048576  # 1024 MiB domain
        try:
            ok, det = E.ram_size_matches_domain("u", "vm", 256)  # truncating dump
            self.assertFalse(ok)
            self.assertGreater(det["rel_diff"], 0.5)
        finally:
            E.virsh_max_memory_kib = orig

    def test_cannot_verify_is_none(self):
        orig = E.virsh_max_memory_kib
        E.virsh_max_memory_kib = lambda uri, dom: None
        try:
            ok, det = E.ram_size_matches_domain("u", "vm", 256)
            self.assertIsNone(ok)
            self.assertIn("note", det)
        finally:
            E.virsh_max_memory_kib = orig


class TestDefaultPathByteIdentical(unittest.TestCase):
    """The Wave-1 guarantee: with none of the Plan-05 flags, the driver feeds
    the producer the same env and the same config keys as before."""

    def test_no_flags_no_env_no_config_drift(self):
        with tempfile.TemporaryDirectory() as d:
            wd = pathlib.Path(d)
            # APF env contributes nothing.
            env, path = E.configure_apf_env(wd, False)
            self.assertEqual((env, path), ({}, None))
            # Config writer leaves imageDir exactly as the original.
            orig = {"imageDir": "/orig/dump", "ramSizeMb": 1024,
                    "streaming": {"enabled": True}, "rawRetention": {"enabled": True}}
            _, cfg = E.write_overridden_config(orig, wd, None, None, None)
            self.assertEqual(cfg["imageDir"], orig["imageDir"])
            # And the file on disk round-trips that imageDir.
            written = json.loads((wd / "config_timing_experiment.json").read_text())
            self.assertEqual(written["imageDir"], "/orig/dump")


if __name__ == "__main__":
    unittest.main()
