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


if __name__ == "__main__":
    unittest.main()
