#!/usr/bin/env python3
"""build_analysis_console.py: the console holds no copy of its own to drift from.

Checks the no-drift contract (the template names no channel and no workload),
that a build over a synthetic manifest injects every data block, that the
output carries no network code, that the known-issue registry is recomputed
from the artifacts in the tree, and that a missing or empty corpus is refused
rather than replaced with sample data.

Run:  python3 tests/test_plan10_build.py
      pytest tests/test_plan10_build.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import channel_roster, corpus_manifest, known_issues  # noqa: E402

UI = QEMU_DIR / "plan10_analysis" / "ui"
BUILD = UI / "build_analysis_console.py"
TEMPLATE = UI / "analysis_console.template.html"


def _synthetic_manifest(td: Path) -> Path:
    root = td / "zstd_local"
    for wl, n in (("mem_alpha_v2", 700), ("cpu_beta_v2", 690)):
        d = root / wl.split("_", 1)[0] / wl / "args_--duration_450_--seed_1" / "rep001__t"
        d.mkdir(parents=True)
        for j in range(n + 1):
            (d / f"{j:06d}.zst").write_bytes(b"x")
    m = corpus_manifest.scan(root)
    p = td / "m.json"
    p.write_text(json.dumps(m))
    return p


def test_template_names_no_channel_and_no_workload():
    text = TEMPLATE.read_text()
    assert "/*@@GENERATED_DATA@@*/" in text
    for c in channel_roster.csv_header_names():
        assert not re.search(r"\b" + re.escape(c) + r"\b", text), c


def test_build_injects_real_data_and_no_network_code():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mp = _synthetic_manifest(td)
        out = td / "console.html"
        r = subprocess.run([sys.executable, str(BUILD), "--manifest", str(mp), "--out", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        html = out.read_text()
        for block in ("ROSTER", "MANIFEST", "ISSUES", "MODULES", "CONFIG", "EXAMPLES", "BUILD"):
            assert f"const {block} = " in html, block
        assert "/*@@GENERATED_DATA@@*/" not in html
        assert not re.search(r"\b(fetch\s*\(|XMLHttpRequest|WebSocket|EventSource)\b", html)
        # the injected roster is the derived one
        m = re.search(r"const ROSTER = (\{.*?\});\nconst MANIFEST", html, re.S)
        roster = json.loads(m.group(1))
        assert roster["n_total"] == 64 and [l["computed"] for l in roster["levels"]] == [64, 63, 51, 50, 48]
        # the manifest is the one given, not a sample
        m = re.search(r"const MANIFEST = (\{.*?\});\nconst ISSUES", html, re.S)
        assert json.loads(m.group(1))["n_recordings"] == 2
        # the external references are reported, not hidden
        assert "fonts.googleapis.com" in r.stdout


def test_registry_is_recomputed_from_artifacts():
    reg = {e["id"]: e for e in known_issues.build_registry()}
    assert reg["f1_phase"]["verified"] and reg["f1_phase"]["numbers"]["n_filled"] == 0
    assert reg["coverage_ratio"]["verified"] and reg["coverage_ratio"]["numbers"]["acc_alone"] == 1.0
    assert reg["ceps_snr"]["verified"] and 0.5 < reg["ceps_snr"]["numbers"]["acc_alone"] < 0.6
    assert reg["gate2"]["verified"] and reg["gate2"]["numbers"]["frac_max"] < 0.2
    assert reg["phase_2pi"]["verified"] and reg["phase_2pi"]["numbers"]["n_sites"] > 1
    # the one whose artifact is not on this machine is emitted as unverified, not dropped
    assert not reg["cusum_band"]["verified"] and "NOT recomputed" in reg["cusum_band"]["verification"]
    assert "(unverified)" in reg["cusum_band"]["detail"]


def test_missing_corpus_is_refused():
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "empty"
        empty.mkdir()
        out = Path(td) / "console.html"
        r = subprocess.run([sys.executable, str(BUILD), "--root", str(empty), "--out", str(out)], capture_output=True, text=True)
        assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)
        assert not out.exists()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
