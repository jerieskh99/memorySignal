#!/usr/bin/env python3
"""tests/test_runner_guard.py -- the one test that `python3 -m unittest` does discover, so that
the unsupported runner stops with the instruction to use pytest (SPEC_epoch2.md Part 2 item B22;
EPOCH1_BUILD_REPORT.md section 4 "Documentation known to be stale": unittest discovers only the
unittest-style files and reports OK without the pytest-style gate tests).

Under pytest the environment variable ``PYTEST_CURRENT_TEST`` is set during every test, so the
guard passes; under ``python3 -m unittest`` it is absent and the guard fails with the message
below. The supported runner is ``python3 -m pytest -q tests`` from ``plan11_encoding_ladder/``.
"""
from __future__ import annotations

import os
import unittest

RUNNER_MESSAGE = ("the gate tests are pytest functions that unittest does not discover; run: "
                  "python3 -m pytest -q tests")


class TestRunnerGuard(unittest.TestCase):
    def test_running_under_pytest(self):
        self.assertTrue(os.environ.get("PYTEST_CURRENT_TEST"), RUNNER_MESSAGE)


if __name__ == "__main__":
    unittest.main()
