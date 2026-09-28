"""SPEC_epoch2.md B28 (E1 4 "Not reached by any automated test"; CHECK_1 M10; CHECK_2 M17): the driver
end to end on a small synthetic corpus across every module, moves 0 to 13, then a `--dry-run` resume.

The corpus is `synth corpus --reps 2 --idle 2 --n-pairs 60` (26 cells, no server, no real data). The
run is `run_moves run --moves 0-13 --null-perm 5 --n-estimators 10 --n-jobs 2 --gord-n-order-perm 2
--gord-null-perm 5 --duration-s 38.64 --assume-failed-zero --assume-reason "end-to-end test"`. With
`--null-perm 5` every null-judging gate reads `not run: 5 permutations < 500` by design (B3, B27) and
the run is not admissible for the paper; what the test checks is that every module runs on every
other module's real output, every table and figure exists, no ledger record fails, and a resume with
nothing changed re-runs no split stage.

Measured cost at build time (2026-09-17, the eight-core reference machine while four other test
suites and a second driver ran beside it): 403 s of CPU time, 25 min of wall time; on an idle
machine expect five to eight minutes. SPEC_epoch2 B28 asks for under five minutes and does not
mark the test slow; it is not marked slow here either, and the number is the honest one.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from _b2_common import PKG, tmp_out
from plan11_encoding_ladder import figures, run_moves, tables


def _run(argv):
    so, se = io.StringIO(), io.StringIO()
    with redirect_stdout(so), redirect_stderr(se):
        rc = run_moves.main(argv)
    return rc, so.getvalue(), se.getvalue()


def _table_csv(out: Path, name: str) -> Path | None:
    """The CSV a table name writes (tables.run), or None for the manifest."""
    if name == "manifest":
        return None
    stem = "table_wapf_over_apf" if name == "wapf_over_apf" else name
    return out / "report" / "tables" / f"{stem}.csv"


def test_driver_end_to_end_then_dry_run_resume():
    root = tmp_out(); out = root / "out"
    subprocess.run([sys.executable, "-m", "plan11_encoding_ladder.synth", "corpus", "--root", str(root / "root"), "--reps", "2",
                    "--idle", "2", "--n-pairs", "60", "--no-compress", "--jobs", "2"], check=True, cwd=str(PKG.parent))
    flags = ["--out", str(out), "--root", str(root / "root"), "--moves", "0-13", "--null-perm", "5", "--n-estimators", "10",
             "--n-jobs", "2", "--gord-n-order-perm", "2", "--gord-null-perm", "5", "--duration-s", "38.64",
             "--assume-failed-zero", "--assume-reason", "end-to-end test"]
    rc, so, se = _run(["run", *flags])
    L = json.loads((out / "driver_state.json").read_text())
    failed = [(c["name"], c["status"], (c.get("stderr_tail") or "")[-800:]) for c in L["commands"] if str(c.get("status", "")).startswith("failed")]
    assert rc == 0 and not failed, (rc, se[-2000:], failed)
    # every table, every figure, the skeleton and the manifest
    for name in tables.TABLE_NAMES:
        p = _table_csv(out, name)
        if p is not None:
            assert p.is_file(), name
    skipped = (out / "report" / "figures" / "SKIPPED.txt").exists()
    for name in figures.FIGURE_NAMES:
        pdf, png = out / "report" / "figures" / f"fig_{name}.pdf", out / "report" / "figures" / f"fig_{name}.png"
        assert skipped or (pdf.is_file() and png.is_file()), name
    assert (out / "report" / "paper2_skeleton.tex").is_file() and (out / "report" / "manifest.json").is_file()
    # the per-move record: every planned command of moves 0 to 13 has a done, kept or skipped status
    statuses = {c["name"]: c["status"] for c in L["commands"]}
    for name in ("extract index", "extract all", "preconditions", "gp", "gc combined", "gk0", "select apf", "splits apf", "gx apf", "gl",
                 "gl2 rerun (feature drop after G-L (ii))", "tables table6", "gj", "gdec", "select combined", "splits combined",
                 "gdim", "gm", "variance", "cluster", "tables (all)", "latex skeleton", "figures (all)", "tables manifest", "gf check on Table 7"):
        assert statuses[name] == "done" or statuses[name].startswith(("kept:", "skipped:")), (name, statuses[name])
    # the epoch-2 records are in the files the run wrote
    pre = json.loads((out / "gates" / "preconditions.json").read_text())["params"]
    assert pre["C1_rule_applied"] in ("idle_floor", "absolute") and "C1_rule_in_force" in pre                   # B1
    gl2 = json.loads((out / "gates" / "gl2_rerun.json").read_text())
    assert gl2["status"].startswith(("done", "not run:"))                                                         # B10
    sc = json.loads((out / "extract" / "gemm__rep00__synth" / "sidecar.json").read_text())
    assert sc["duration_s_declared"] == 38.64                                                                     # B12
    gf = (out / "gates" / "gf.csv").read_text()
    assert "5 permutations < 500" in gf or "no admissible idle cell" in gf                                        # B3 (the gf CLI floor)
    # a dry-run resume with the same flags: nothing is stale but the two move-7 steps that declare the whole
    # selection.json (SPEC_epoch2 B2 keeps the whole file for `gl`; the other builder's test_driver asserts it for
    # `tables table6` too), which grew at moves 9 to 12
    rc, so, se = _run(["run", *flags, "--dry-run"])
    assert rc == 0, se[-2000:]
    L = json.loads((out / "driver_state.json").read_text())
    last_run_start = L["runs"][-1]["started_at"]
    recent = [c for c in L["commands"] if c.get("started_at", "") >= last_run_start]
    assert recent, "the dry run recorded nothing"
    stale = {c["name"]: c["stale"] for c in recent if c.get("stale")}
    assert set(stale) <= {"gl", "tables table6"}, stale
    for c in recent:
        if c["name"].startswith(("splits ", "gx ", "select ", "grid ", "gord ", "g3 ", "features ")):
            assert c["status"].startswith("skipped:"), (c["name"], c["status"], c.get("stale"))
