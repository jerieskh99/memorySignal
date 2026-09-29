#!/usr/bin/env python3
"""encoding_panel.py: the console's window onto plan11_encoding_ladder launches the toolkit's own
driver one move at a time, in order, and shows the toolkit's files as written.

Uses the toolkit's own synth.py for a tiny corpus (12 kernels x 1 rep + 1 idle at 40 pairs) plus
two idle cells laid out like the real capture (sleep/sleep/sleep_600/rep00N__idle_01c) and one
directory of another family, so the cells table's rule (kernel and idle rows listed, anything
else counted) is exercised without naming anything. Nothing here touches a server or writes under
plan11_encoding_ladder/.

Run:  python3 tests/test_plan10_encoding_panel.py
      pytest tests/test_plan10_encoding_panel.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import encoding_panel as EP   # noqa: E402

TOOLKIT_OK = EP.toolkit_present()


def make_corpus(td: Path) -> Path:
    root = td / "corpus"
    r = subprocess.run([sys.executable, "-m", "plan11_encoding_ladder.synth", "corpus", "--root", str(root), "--reps", "1", "--idle", "1",
                        "--n-pairs", "40", "--jobs", "2"], cwd=str(QEMU_DIR), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    src = next((root / "kernel").glob("kernel_sleep_v2/*/rep001__idle/*.csv.zst"))
    for n in (1, 2):
        d = root / "sleep" / "sleep" / "sleep_600" / f"rep00{n}__idle_01c"
        d.mkdir(parents=True)
        shutil.copy(src, d / "run_matrix_test1_sleep_600.npy.substrate_trajectory.csv.zst")
    other = root / "other" / "other_thing_v2" / "args_--seed_42" / "rep001__x"
    other.mkdir(parents=True)
    shutil.copy(src, other / "run_matrix_test1_other_thing_v2.npy.substrate_trajectory.csv.zst")
    return root


def wait_done(P: EP.Panel, timeout: float = 300) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        b = P.board()
        if not b["running"]:
            return b
        time.sleep(0.5)
    raise AssertionError("the launch did not finish in time")


def test_flags_plan_and_identity():
    if not TOOLKIT_OK:
        print("skip: the toolkit is not on disk")
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        P = EP.Panel(td / "cfg.json")
        c = P.set_config({"out": str(td / "out"), "root": str(td / "corpus"), "preset": "smoke"})
        assert c["preset"] == "smoke" and c["flags"]["null_perm"] == 20 and c["flags"]["duration_s"] == 77.28
        names = {f["dest"]: f for f in c["driver_flags"]}
        assert {"null_perm", "n_jobs", "null_splits", "assume_failed_zero", "assume_reason", "c1_rule", "seed_offset"} <= set(names)
        assert names["assume_failed_zero"]["kind"] == "bool" and names["null_perm"]["default"] == 500 and "auto" in names["c1_rule"]["choices"]
        assert "--null-perm 20" in c["run_line"] and "--assume-reason 'smoke run'" in c["run_line"] and c["run_line"].startswith("python3 -m plan11_encoding_ladder.run_moves run --out ")
        tk = c["toolkit"]
        assert tk["present"] and tk["n_py_files"] > 20 and len(tk["content_fingerprint"]) == 16 and tk["package_version"]
        assert isinstance(tk["committed"], bool) and (tk["repo_head"] is None or len(tk["repo_head"]) == 40)
        # tokens are stable and every set flag travels, default-valued or not
        toks = EP.flag_tokens({"null_perm": 500, "assume_failed_zero": True, "assume_reason": "x", "n_jobs": 4})
        assert toks == ["--null-perm", "500", "--assume-failed-zero", "--assume-reason", "x", "--n-jobs", "4"]
        assert EP.flag_tokens({"assume_failed_zero": False, "assume_reason": ""}) == []
        # a flag that is not the driver's is refused; a preset edit becomes custom
        try:
            P.set_config({"flags": {"nope": 1}}); assert False
        except EP.PanelError:
            pass
        c = P.set_config({"flags": {"n_jobs": 1}})
        assert c["preset"] == "custom" and c["flags"]["n_jobs"] == 1
        # the plan is the driver's own: 15 moves, the runbook's titles, move 0 first
        plan = P.plan()
        assert len(plan) > 80 and plan[0]["move"] == 0 and plan[0]["line"].startswith("python3 -m plan11_encoding_ladder.extract index --root ")
        text = EP.plan_text(str(td / "out"), str(td / "corpus"), P.cfg["flags"], 0)
        assert "extract index" in text and plan[0]["line"].split(" ", 4)[4] in text
        sec = EP.runbook_sections()
        assert sec[0]["title"] == "the cell index" and sec[14]["title"].startswith("the comparators") and EP.EUSIPCO_KEY in sec
        assert sec[6]["look_at"].startswith("Look at:")
        # the board before anything ran: every move not run, only move 0 runnable, dependencies linear
        b = P.board()
        states = {m["move"]: m for m in b["moves"]}
        assert all(m["state"] == "not run" for m in b["moves"]) and states[0]["runnable"] and not states[1]["runnable"]
        assert states[7]["requires"] == [6] and 6 in states[7]["reads_from"] and states[EP.EUSIPCO_KEY]["requires"] == [14]
        assert states[12]["output_dirs"][0].startswith("features/combined")
        try:
            P.launch(1); assert False
        except EP.PanelError as e:
            assert "waits for move" in str(e)
        # the file readers refuse anything outside <out>
        for bad in ("../x", "/etc/passwd", "a/../../b"):
            try:
                EP.safe_path(td / "out", bad); assert False, bad
            except EP.PanelError:
                pass
        assert EP.text_file(td / "out", "cells.csv")["exists"] is False
        assert EP.cells(td / "out")["exists"] is False


def test_moves_run_in_order_and_files_are_shown_as_written():
    if not TOOLKIT_OK:
        print("skip: the toolkit is not on disk")
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = make_corpus(td)
        out = td / "out"
        P = EP.Panel(td / "cfg.json")
        P.set_config({"out": str(out), "root": str(root), "preset": "smoke", "flags": {"duration_s": 25.76, "n_jobs": 2}})
        # a refusal the driver prints is a launch with a non-zero exit and no ledger change
        P.set_config({"flags": {"assume_failed_zero": True, "assume_reason": ""}})
        rec = P.launch(0)
        assert rec["move"] == 0 and rec["shell"][0].startswith("python3 -m plan11_encoding_ladder.run_moves run") and rec["toolkit"]["content_fingerprint"]
        b = wait_done(P)
        lr = P.launch_record(rec["id"])
        assert lr["exit_code"] == 2 and lr["finished_at"] and lr["driver_params"] is None
        tail = P.log_tail(rec["id"])["lines"]
        assert any("--assume-reason" in ln for ln in tail), tail
        assert all(m["state"] == "not run" for m in b["moves"])
        P.set_config({"flags": {"assume_reason": "smoke run"}})
        # moves 0 to 5 in order, each its own launch of the driver
        for mv in range(6):
            rec = P.launch(mv)
            assert rec["config"]["flags"]["assume_reason"] == "smoke run" and (Path(rec["log"])).exists()
            b = wait_done(P)
            st = {m["move"]: m for m in b["moves"]}
            lr = P.launch_record(rec["id"])
            assert lr["exit_code"] == 0, (mv, P.log_tail(rec["id"])["lines"][-8:])
            assert st[mv]["state"] == "done", (mv, st[mv])
            assert st[mv + 1]["runnable"] and not st[mv + 2]["runnable"]
            assert lr["driver_params"]["null_perm"] == 20 and lr["driver_run"]["moves"] == [mv] and lr["driver_run"]["exit_code"] == 0
        # the launch records are under <out>/.console, which the toolkit neither reads nor hashes
        assert (out / EP.CONSOLE_DIR / "launches").is_dir() and len(P.launches()) == 7
        # the ledger's own words per command, and a move's commands are the driver's plan
        st = {m["move"]: m for m in b["moves"]}
        c2 = st[2]["commands"]
        assert [c["name"] for c in c2][:2] == ["preconditions", "pass-table template"] and all(c["status"] in ("done",) for c in c2)
        assert st[2]["output_dirs"] == ["gates", "inputs"]
        # the cells table: kernel and idle rows joined with their sidecars and preconditions; the other family counted, not listed
        cells = EP.cells(out)
        assert cells["exists"] and cells["n_kernel"] == 12 and cells["n_idle"] == 3 and cells["n_other_dirs"] == 1
        assert list(cells["other_dirs_status"]) == ["refused: unknown kernel"]
        assert not any("other_thing" in json.dumps(r) for r in cells["rows"])
        idle = [r for r in cells["rows"] if r["role"] == "idle" and r["label"] == "idle_01c"]
        assert len(idle) == 2 and {r["rep"] for r in idle} == {"0", "1"} and {r["rep_dir"] for r in idle} == {"1", "2"} and all(r["kernel"] == "sleep" for r in idle)
        assert all(r["n_pairs"] == 40 and r["extract_status"] == "ok" for r in cells["rows"])
        assert all(r["all_hard_pass"] in ("true", "false") for r in cells["rows"]) and cells["preconditions"]["exists"]
        lex = [r for r in cells["rows"] if r["kernel"] == "lexer"]
        assert lex and lex[0]["C1_rule"] and cells["index_params"]["idle_markers"] == ["sleep", "idle"]
        # files as written: a CSV keeps its strings, a JSON its object, a PNG is binary with a PDF beside it
        gc = EP.text_file(out, "gates/gc.csv")
        assert gc["kind"] == "csv" and "verdict" in gc["header"] and gc.get("params_file") is None      # gc has gc.<rung>.params.json, one per rung
        gf = EP.text_file(out, "gates/gf.csv")
        iv = gf["header"].index("verdict")
        assert all(r[iv].startswith("not run: no admissible idle cell") for r in gf["rows"]), gf["rows"][:2]
        pj = EP.text_file(out, "gates/preconditions.json")
        assert pj["kind"] == "json" and "excluded_cells" in pj["json"] and "C1_rule_in_force" in pj["json"]["params"]
        fig = EP.text_file(out, "report/figures/fig_apf_per_kernel.png")
        assert fig["kind"] == "binary" and fig["content_type"] == "image/png"
        vs = {v["id"]: v for v in EP.views(out, [])}
        assert vs["count"]["any"] and vs["count"]["files"][0]["pdf"] == "report/figures/fig_apf_per_kernel.pdf"
        assert not vs["table6"]["any"] and all(not f["exists"] for f in vs["table6"]["files"])
        assert EP.text_file(out, "report/tables/table6.csv") == {"path": "report/tables/table6.csv", "exists": False, "kind": None}
        ls = EP.listing(out, "")
        assert {e["name"] for e in ls["entries"]} >= {"cells.csv", "extract", "gates", "inputs", "report", "driver_state.json"} and EP.CONSOLE_DIR not in {e["name"] for e in ls["entries"]}
        # the params blocks, read-only: the driver's, the inputs, every result's
        pb = EP.params_blocks(out)
        assert pb["driver_params"]["assume_reason"] == "smoke run" and pb["inputs"]["pass_table.csv"]["exists"] and pb["inputs"]["failed_counts.csv"]["exists"] is False
        paths = {b["path"] for b in pb["blocks"]}
        assert {"gates/gk0.params.json", "gates/gp.params.json", "gates/preconditions.json", "cells.index.json", "report/figures/figures.json"} <= paths
        assert pb["n_sidecars_not_listed"] == 15 and not any(b["path"].startswith(".console") for b in pb["blocks"])
        # re-run with --force is a runbook command too; and a second writer is refused while one runs
        rec = P.launch(0, force=True)
        try:
            P.launch(1); assert False
        except EP.PanelError as e:
            assert "still running" in str(e)
        wait_done(P)
        assert P.launch_record(rec["id"])["force"] is True and "--force" in P.launch_record(rec["id"])["shell"][0]
        # stop: a launched driver can be terminated; afterwards nothing runs
        rec = P.launch(1, force=True)
        r = P.stop()
        assert r["stopped"] == rec["id"] and P.board()["running"] is None
        assert P.launch_record(rec["id"])["exit_code"] is not None
        # the EUSIPCO row waits for move 14 and names its two commands
        st = {m["move"]: m for m in P.board()["moves"]}
        assert st[EP.EUSIPCO_KEY]["state"] == "not run" and not st[EP.EUSIPCO_KEY]["runnable"]
        assert EP.eusipco_argv("/o")[1][-2:] == ["--out", "/o"] and EP.eusipco_argv("/o", "/tex.tex")[1][-2:] == ["--standalone", "/tex.tex"]


def test_a_stopped_move_reads_not_run():
    """A driver stopped mid-move leaves an open `runs` entry in its ledger; the move then reads
    not run whatever finished before the stop, the next move stays disabled, and a later
    attempt that records every command clears it. Built on a ledger written to a temp dir by
    the test, in the driver's own shape."""
    if not TOOLKIT_OK:
        print("skip: the toolkit is not on disk")
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / "out"
        (out / "gates").mkdir(parents=True)
        P = EP.Panel(td / "cfg.json")
        P.set_config({"out": str(out), "root": str(td / "corpus"), "preset": "smoke"})
        plan = P.plan()
        m3 = [c for c in plan if c["move"] == 3]
        assert len(m3) == 5
        rec = lambda c, t, status="done": {"key": c["key"], "move": c["move"], "name": c["name"], "started_at": t, "finished_at": t, "status": status, "exit_code": 0}
        # every move before 3 done in finished runs; move 3 cut after its second command; its outputs exist
        cmds, runs = [], []
        for mv in (0, 1, 2):
            t = f"2026-09-17T10:0{mv}:00+00:00"
            runs.append({"started_at": t, "finished_at": t, "moves": [mv], "exit_code": 0})
            cmds += [rec(c, t) for c in plan if c["move"] == mv]
        (out / "cells.csv").write_text("cell_id\n")
        runs.append({"started_at": "2026-09-17T10:10:00+00:00", "finished_at": None, "moves": [3]})
        cmds += [rec(c, "2026-09-17T10:10:01+00:00") for c in m3[:2]]
        (out / "gates" / "gc.csv").write_text("rung,verdict\napf,pass\n")
        (out / "driver_state.json").write_text(json.dumps({"runs": runs, "commands": cmds, "params": {}}))
        st = {m["move"]: m for m in P.board()["moves"]}
        assert st[2]["state"] == "done" and st[3]["state"] == "not run" and st[3]["interrupted_at"] == "2026-09-17T10:10:00+00:00"
        assert st[3]["runnable"] and not st[4]["runnable"]
        assert [c["state"] for c in st[3]["commands"]] == ["done", "done", "not run", "not run", "not run"]
        # a later attempt that records every command of the move (skipped ones count) clears the cut
        runs.append({"started_at": "2026-09-17T10:20:00+00:00", "finished_at": "2026-09-17T10:20:09+00:00", "moves": [3], "exit_code": 0})
        cmds += [rec(c, "2026-09-17T10:20:01+00:00", "skipped: outputs exist and inputs unchanged") for c in m3[:2]] + [rec(c, "2026-09-17T10:20:05+00:00") for c in m3[2:]]
        (out / "driver_state.json").write_text(json.dumps({"runs": runs, "commands": cmds, "params": {}}))
        st = {m["move"]: m for m in P.board()["moves"]}
        assert st[3]["state"] == "done" and st[3]["interrupted_at"] is None and st[4]["runnable"]
        # a cut single-command move that had run before reads not run, not done
        runs.append({"started_at": "2026-09-17T10:30:00+00:00", "finished_at": None, "moves": [1]})
        (out / "driver_state.json").write_text(json.dumps({"runs": runs, "commands": cmds, "params": {}}))
        st = {m["move"]: m for m in P.board()["moves"]}
        assert st[1]["state"] == "not run" and st[1]["interrupted_at"] and not st[2]["runnable"]


def test_classify_and_external_detection():
    assert EP.classify(None) == "not run" and EP.classify("done") == "done" and EP.classify("skipped: outputs exist and inputs unchanged") == "done"
    assert EP.classify("kept: author input exists") == "done" and EP.classify("failed: exit 2") == "failed" and EP.classify("refused: grid incomplete") == "refused"
    assert EP.classify("not run: module absent") == "not run" and EP.classify("dry-run") == "not run"
    with tempfile.TemporaryDirectory() as td:
        assert EP.external_drivers(Path(td)) == []
        assert EP._first_move("6-14") == 6 and EP._first_move("2,4") == 2 and EP._first_move(None) == 0 and EP._moves_of("2-4,12") == [2, 3, 4, 12]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
