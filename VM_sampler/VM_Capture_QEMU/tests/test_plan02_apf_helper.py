#!/usr/bin/env python3
"""tests/test_plan02_apf_helper.py -- APF helper prev-dump delete sudo gate.

Plan 05 needs the streaming APF helper to keep working even when the dump dir
is owned by the libvirt-qemu user (the default /var/lib/libvirt/qemu/dump),
where the unprivileged runner cannot unlink. The fix is an opt-in privileged
fallback. This proves the gate both ways:

  - default (TIMING_SUDO_DELETE unset): a failed unlink stays a silent no-op and
    NO sudo subprocess is spawned -- byte-identical to the pre-Plan-05 helper
  - opt-in (TIMING_SUDO_DELETE=1): a failed unlink falls back to
    ``sudo -n rm -f <prev>`` (non-interactive, mirrors the producer self-clean)

Both still return EXIT_OK: the APF line is already appended, so prev deletion is
best-effort and never fails the pair.

No VM. numpy required (already a project dep). The unlink is forced to fail by
monkeypatching Path.unlink; subprocess.run is stubbed so no real sudo is spawned.

Run with:
    python3 -m unittest tests.test_plan02_apf_helper
    python3 tests/test_plan02_apf_helper.py
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import plan02_apf_helper as H  # noqa: E402


class TestPrevDeleteSudoGate(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.wd = pathlib.Path(self._td.name)
        self._env0 = os.environ.get("TIMING_SUDO_DELETE")
        self._unlink0 = pathlib.Path.unlink
        self._run0 = subprocess.run
        self.calls: list[list] = []

        def fail_unlink(_self, *a, **k):
            raise OSError("EACCES (simulated libvirt-owned dump dir)")

        def rec_run(cmd, *a, **k):
            self.calls.append(list(cmd))

            class _R:  # minimal CompletedProcess stand-in (helper ignores it)
                returncode = 0

            return _R()

        pathlib.Path.unlink = fail_unlink   # only Step 4 uses Path.unlink
        subprocess.run = rec_run            # local `import subprocess` sees this

    def tearDown(self):
        pathlib.Path.unlink = self._unlink0
        subprocess.run = self._run0
        if self._env0 is None:
            os.environ.pop("TIMING_SUDO_DELETE", None)
        else:
            os.environ["TIMING_SUDO_DELETE"] = self._env0
        self._td.cleanup()

    def _invoke(self) -> int:
        prev = self.wd / "memory_dump-0.raw"
        curr = self.wd / "memory_dump-1.raw"
        prev.write_bytes(b"\x00" * 4096)
        curr.write_bytes(b"\x01" * 4096)
        # apf-jsonl + ack-dir live in the same writable temp dir so Steps 1-3
        # succeed; only the prev unlink (Step 4) is forced to fail.
        return H.main([
            "--prev", str(prev), "--curr", str(curr),
            "--apf-jsonl", str(self.wd / "apf.jsonl"),
            "--ack-dir", str(self.wd / "acks"), "--seq", "1",
        ])

    def test_default_no_sudo(self):
        os.environ.pop("TIMING_SUDO_DELETE", None)
        rc = self._invoke()
        self.assertEqual(rc, H.EXIT_OK)
        self.assertEqual(self.calls, [])   # no sudo spawned on the default path

    def test_optin_calls_sudo_rm(self):
        os.environ["TIMING_SUDO_DELETE"] = "1"
        rc = self._invoke()
        self.assertEqual(rc, H.EXIT_OK)
        self.assertEqual(len(self.calls), 1)
        cmd = self.calls[0]
        self.assertEqual(cmd[:4], ["sudo", "-n", "rm", "-f"])
        self.assertTrue(str(cmd[-1]).endswith("memory_dump-0.raw"))


if __name__ == "__main__":
    unittest.main()
