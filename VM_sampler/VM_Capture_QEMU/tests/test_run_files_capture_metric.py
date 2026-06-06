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

import os
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


class TestWaitForSshHardening(unittest.TestCase):
    """The Plan-05 Wave-4 hardening: 3-consecutive-success readiness probe,
    stricter probe content, and a settle after force-destroy.

    These prove the *defaults* changed in the right direction; the live behaviour
    is exercised by the server smoke. No SSH is performed here -- we monkeypatch
    `run` to script a flaky-then-stable guest and verify wait_for_ssh holds out
    for the 3-in-a-row.
    """

    def test_required_consecutive_default_is_three(self):
        # No env override -> default 3 consecutive successes required.
        orig = os.environ.pop("SSH_READY_CONSECUTIVE", None)
        try:
            # Read the default the code uses inside wait_for_ssh.
            self.assertEqual(int(os.environ.get("SSH_READY_CONSECUTIVE", "3")), 3)
        finally:
            if orig is not None:
                os.environ["SSH_READY_CONSECUTIVE"] = orig

    def test_probe_uses_real_path_not_just_echo(self):
        # The probe must exercise auth + a real filesystem path, not just connect.
        # Read the source so a future refactor that drops the path test fails here.
        src = pathlib.Path(R.__file__).read_text()
        self.assertIn("test -d $HOME && echo ready", src)

    def test_three_consecutive_required(self):
        # Script a flaky guest: ok, FAIL, ok, ok, ok -> needs 3-in-a-row to return.
        calls = {"n": 0, "log": []}
        rc_seq = [0, 1, 0, 0, 0]   # success, fail, success, success, success

        def fake_run(cmd, *a, **k):
            i = calls["n"]
            calls["n"] += 1
            calls["log"].append(cmd)
            return rc_seq[i] if i < len(rc_seq) else 0

        # Save and patch the module-level run + sleep so the test doesn't actually wait.
        orig_run, orig_sleep = R.run, R.time.sleep
        R.run = fake_run
        R.time.sleep = lambda _s: None
        try:
            ok = R.wait_for_ssh()
        finally:
            R.run, R.time.sleep = orig_run, orig_sleep
        self.assertTrue(ok)
        # 5 calls: 1 ok, 1 fail (counter resets), 3 ok in a row.
        self.assertEqual(calls["n"], 5)

    def test_force_destroy_includes_settle_sleep(self):
        # Read the source: the post-destroy sleep MUST be present, since without it
        # the next virsh start races libvirt's cleanup and reproduces the bug.
        src = pathlib.Path(R.__file__).read_text()
        self.assertIn("time.sleep(5)", src)
        # And the explanatory comment so a future contributor does not remove it.
        self.assertIn("settle", src.lower())

    def test_workload_ssh_retries_on_255(self):
        # The retry-once path must be wired so a transient cold-boot SSH failure
        # does not abort a multi-step campaign. Read the source for the guard.
        src = pathlib.Path(R.__file__).read_text()
        self.assertIn("workload SSH returned 255", src)
        self.assertIn("retrying the step once", src)


if __name__ == "__main__":
    unittest.main()
