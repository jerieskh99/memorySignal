#!/usr/bin/env python3
"""tests/test_run_files_capture_metric.py -- the CAPTURE_METRIC=apf seam.

No VM. Proves the opt-in APF capture metric is strictly additive: with the
default ('delta' or unset) the env prefix is EMPTY, so the launched capture
command is byte-identical to the pre-APF pipeline; with 'apf' it carries exactly
the TIMING_APF_* + TIMING_SUDO_DELETE keys that make run_qemu_capture.sh skip the
delta consumer and the producer stream APF.

Run:
    python3 -m unittest tests.test_run_files_capture_metric
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import run_files_controlled as R  # noqa: E402


class TestApfEnvPrefix(unittest.TestCase):
    def test_delta_and_unset_are_empty(self):
        # The byte-identical-default contract: nothing added to the command.
        self.assertEqual(R._apf_env_prefix("delta", "/x.jsonl", "/acks", "/h.log"), "")
        self.assertEqual(R._apf_env_prefix("", "/x.jsonl", "/acks", "/h.log"), "")
        self.assertEqual(R._apf_env_prefix("DELTA", "/x.jsonl", "/acks", "/h.log"), "")

    def test_apf_sets_exactly_the_expected_keys(self):
        p = R._apf_env_prefix("apf", "/p/apf.jsonl", "/p/acks", "/p/h.log")
        for key in ("CAPTURE_METRIC=apf", "TIMING_APF_STREAM=1",
                    "TIMING_APF_JSONL=", "TIMING_APF_ACK_DIR=",
                    "TIMING_APF_HELPER_LOG=", "TIMING_SUDO_DELETE=1"):
            self.assertIn(key, p)
        # paths are present (shell-quoted)
        self.assertIn("/p/apf.jsonl", p)
        self.assertIn("/p/acks", p)
        # delta keys (Rust consumer) are NOT referenced
        self.assertNotIn("OFFLINE_MODE", p)

    def test_apf_queue_enqueues_no_stream(self):
        # B path: producer ENQUEUES (no TIMING_APF_STREAM) and the apf_calc
        # consumer computes APF, appending to TIMING_APF_JSONL.
        p = R._apf_env_prefix("apf_queue", "/p/apf.jsonl", "/p/acks", "/p/h.log")
        self.assertIn("CAPTURE_METRIC=apf_queue", p)
        self.assertIn("TIMING_APF_JSONL=", p)
        self.assertIn("/p/apf.jsonl", p)
        # the producer must NOT stream in queue mode (the consumer computes APF)
        self.assertNotIn("TIMING_APF_STREAM", p)
        # no inline-helper ack/log keys
        self.assertNotIn("TIMING_APF_ACK_DIR", p)
        self.assertNotIn("TIMING_APF_HELPER_LOG", p)


class TestSustainWrap(unittest.TestCase):
    def test_off_is_passthrough(self):
        orig = R.SUSTAIN_LOOP
        R.SUSTAIN_LOOP = False
        try:
            cmd = "/bin/work --duration 120 --x"
            self.assertEqual(R._sustain_wrap(cmd), cmd)
        finally:
            R.SUSTAIN_LOOP = orig

    def test_on_wraps_with_timeout_loop(self):
        orig = R.SUSTAIN_LOOP
        R.SUSTAIN_LOOP = True
        try:
            w = R._sustain_wrap("/bin/work --capacity 9 --duration 600 --x")
            self.assertIn("timeout 600", w)
            self.assertIn("while :; do", w)
            self.assertIn("/bin/work --capacity 9 --duration 600 --x", w)
            # `timeout` exits 124 at the end of the window; swallow it so the
            # orchestrator does not read the normal end as a step failure.
            self.assertTrue(w.rstrip().endswith("|| true"))
        finally:
            R.SUSTAIN_LOOP = orig

    def test_on_without_duration_is_passthrough(self):
        orig = R.SUSTAIN_LOOP
        R.SUSTAIN_LOOP = True
        try:
            cmd = "/bin/work --x"
            self.assertEqual(R._sustain_wrap(cmd), cmd)
        finally:
            R.SUSTAIN_LOOP = orig


if __name__ == "__main__":
    unittest.main()
