"""QA suite for subset_run.py -- config-to-steps-file generator.

Unit tests hit the pure functions directly; integration tests drive main() via
subprocess with real temp configs (main() uses argparse + sys.exit, so a
subprocess is the clean way to exercise it end to end).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "subset_run.py"
sys.path.insert(0, str(HERE))

import subset_run as S  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def base_cfg(**over):
    cfg = {
        "label": "qa_test",
        "duration_s": [60],
        "scales": [1.0],
        "retention": "metrics",
        "capture_metric": "delta",
        "workload_family": {"sandbox": {"asc": 2}},
    }
    cfg.update(over)
    return cfg


def run_main(tmp_path, cfg, *extra, out="steps.txt"):
    """Run subset_run.py as a subprocess. Returns CompletedProcess."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    cmd = [sys.executable, str(SCRIPT),
           "--config", str(cfg_path),
           "--out", str(tmp_path / out),
           "--runs-dir", str(tmp_path / "runs")]
    cmd += list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)


def load_cfg(tmp_path, cfg):
    """Call load_config directly; returns cfg or raises SystemExit."""
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return S.load_config(str(p))


# --------------------------------------------------------------------------
# load_config -- validation
# --------------------------------------------------------------------------
class TestLoadConfig:
    def test_valid(self, tmp_path):
        assert load_cfg(tmp_path, base_cfg())["label"] == "qa_test"

    @pytest.mark.parametrize("missing",
        ["label", "duration_s", "scales", "retention", "capture_metric", "workload_family"])
    def test_missing_required_key(self, tmp_path, missing):
        cfg = base_cfg()
        del cfg[missing]
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, cfg)

    def test_bad_retention(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(retention="raw_plus"))

    def test_bad_capture_metric(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(capture_metric="cosine"))

    def test_empty_duration(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(duration_s=[]))

    def test_negative_scale(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(scales=[1.0, -2.0]))

    def test_bool_scale_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(scales=[True]))

    def test_float_duration_rejected(self, tmp_path):
        # would emit `--duration 300.0`, unparseable by guest + orchestrator regex
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(duration_s=[300.0]))

    def test_int_duration_ok(self, tmp_path):
        assert load_cfg(tmp_path, base_cfg(duration_s=[60, 300]))["duration_s"] == [60, 300]

    def test_float_scale_still_allowed(self, tmp_path):
        # scales are legitimately fractional -- must NOT be rejected
        assert load_cfg(tmp_path, base_cfg(scales=[0.5, 2.0]))["scales"] == [0.5, 2.0]

    def test_label_trailing_newline_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(label="run01\n"))

    def test_empty_named_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(workload_family={"sandbox": {"named": []}}))

    def test_reps_absent_defaults_to_one(self, tmp_path):
        assert load_cfg(tmp_path, base_cfg())["reps"] == 1

    def test_reps_valid(self, tmp_path):
        assert load_cfg(tmp_path, base_cfg(reps=3))["reps"] == 3

    def test_reps_zero_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(reps=0))

    def test_reps_float_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(reps=1.5))

    def test_reps_bool_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(reps=True))

    def test_label_with_space_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(label="has space"))

    def test_label_with_slash_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(label="a/../b"))

    def test_label_non_string(self, tmp_path):
        # a JSON numeric label -- should exit cleanly, not crash
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(label=123))

    def test_workload_family_flat_string_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(workload_family={"sandbox": "asc"}))

    def test_workload_family_multikey_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(workload_family={"sandbox": {"asc": 1, "desc": 2}}))

    def test_bad_order_type(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(workload_family={"sandbox": {"bogus": 1}}))

    def test_all_as_order_type_rejected(self, tmp_path):
        # the docstring example {"sandbox": {"all": null}} -- is it actually valid?
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(workload_family={"sandbox": {"all": None}}))

    def test_named_non_list_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(workload_family={"io": {"named": "io_read_cache_hit_v2"}}))

    def test_feature_group_unknown(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(capture_metric_group={"bogus": "all"}))

    def test_feature_column_unknown(self, tmp_path):
        with pytest.raises(SystemExit):
            load_cfg(tmp_path, base_cfg(capture_metric_group={"amount": ["not_real"]}))

    def test_capture_metric_group_optional(self, tmp_path):
        # absent is fine
        assert "capture_metric_group" not in load_cfg(tmp_path, base_cfg())


# --------------------------------------------------------------------------
# resolve_features / metric_group_env
# --------------------------------------------------------------------------
class TestResolveFeatures:
    def test_worked_example(self):
        feats = {"amount": ["hamming", "lz_change"], "direction": "none",
                 "content": "all", "texture": ["edge_energy"]}
        assert S.resolve_features(feats) == "positional,informational,content,texture"

    def test_all_none_empty(self):
        assert S.resolve_features(
            {"amount": "none", "direction": "none", "content": "none", "texture": "none"}) == ""

    def test_all_equals_explicit_full_list(self):
        full = list(S.FEATURES_BY_GROUP["amount"])
        assert S.resolve_features({"amount": "all"}) == S.resolve_features({"amount": full})

    def test_every_column_covered_by_a_submodule(self):
        covered = set()
        for cols in S.SUBMODULE_COLUMNS.values():
            covered |= set(cols)
        for grp, cols in S.FEATURES_BY_GROUP.items():
            assert set(cols) <= covered, f"{grp} has columns no submodule covers"

    def test_env_empty_for_no_feats(self):
        assert S.metric_group_env(None) == {}
        assert S.metric_group_env({}) == {}


# --------------------------------------------------------------------------
# env-var producers
# --------------------------------------------------------------------------
class TestEnvProducers:
    @pytest.mark.parametrize("ret,expect", [
        ("raw", {"ZSTD": "1"}), ("combined", {"ZSTD": "1"}), ("metrics", {})])
    def test_retention_env(self, ret, expect):
        assert S.retention_env(ret) == expect

    def test_capture_metric_env_normalises(self):
        assert S.capture_metric_env("  SUBSTRATE ") == {"CAPTURE_METRIC": "substrate"}

    def test_capture_metric_env_rejects(self):
        with pytest.raises(SystemExit):
            S.capture_metric_env("bogus")

    def test_add_env_all_survive(self):
        out = S.add_env_variables("./x", {"a": "1", "b": "2", "c": "3"})
        assert out == "A=1 B=2 C=3 ./x"

    def test_add_env_quotes_spaces(self):
        out = S.add_env_variables("./x", {"zstd_dir": "/a b/c"})
        assert "ZSTD_DIR='/a b/c'" in out

    def test_host_env_partial(self):
        assert S.host_env("u@h", None, None) == {"SSH_TARGET": "u@h"}
        assert S.host_env(None, None, None) == {}

    def test_constants_all_present(self):
        out = S.add_constant_envvars("./x")
        for k in ("CAPTURE_MODE", "SSH_WAIT_TIMEOUT", "CONTINUE_ON_FAILURE",
                  "MIN_FREE_DISK_GB", "PYTHONUNBUFFERED"):
            assert k in out


# --------------------------------------------------------------------------
# rep_seed
# --------------------------------------------------------------------------
class TestRepSeed:
    def test_rep0_is_baseline(self):
        assert S.rep_seed(42, 0, "kernel_gemm_v2") == 42

    def test_later_reps_differ(self):
        seeds = [S.rep_seed(42, r, "kernel_gemm_v2") for r in range(4)]
        assert len(set(seeds)) == 4  # all distinct

    def test_stable_across_calls(self):
        # crc32-based, not salted hash() -> same answer every call/process
        assert S.rep_seed(42, 2, "sandbox_ransom_seq") == S.rep_seed(42, 2, "sandbox_ransom_seq")

    def test_decorrelated_by_workload(self):
        assert S.rep_seed(42, 1, "kernel_gemm_v2") != S.rep_seed(42, 1, "sandbox_ransom_seq")


# --------------------------------------------------------------------------
# integration: main() end to end
# --------------------------------------------------------------------------
class TestMainIntegration:
    def test_metrics_run_writes_steps_and_meta(self, tmp_path):
        r = run_main(tmp_path, base_cfg(), "--ssh-target", "u@h", "--ssh-key", "/k")
        assert r.returncode == 0, r.stderr
        assert (tmp_path / "steps.txt").exists()
        assert (tmp_path / "runs" / "qa_test.json").exists()

    def test_steps_have_two_sandbox_cells(self, tmp_path):
        run_main(tmp_path, base_cfg(), "--ssh-target", "u@h", "--ssh-key", "/k")
        lines = (tmp_path / "steps.txt").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_steps_file_absolute_in_launch_line(self, tmp_path):
        r = run_main(tmp_path, base_cfg(), "--ssh-target", "u@h", "--ssh-key", "/k")
        # the STEPS_FILE= on the launch line must be an absolute path, since the
        # launch runs ./run_files_controlled.py from a different cwd
        launch = [ln for ln in r.stdout.splitlines() if "run_files_controlled.py" in ln][0]
        steps_tok = [t for t in launch.split() if t.startswith("STEPS_FILE=")][0]
        assert steps_tok.split("=", 1)[1].startswith("/"), f"STEPS_FILE not absolute: {steps_tok}"

    def test_raw_without_zstd_dir_refused(self, tmp_path):
        r = run_main(tmp_path, base_cfg(retention="raw"), "--ssh-target", "u@h", "--ssh-key", "/k")
        assert r.returncode != 0
        assert "zstd-dir" in (r.stdout + r.stderr)

    def test_raw_with_zstd_dir_ok(self, tmp_path):
        r = run_main(tmp_path, base_cfg(retention="raw"),
                     "--ssh-target", "u@h", "--ssh-key", "/k", "--zstd-dir", str(tmp_path / "z"))
        assert r.returncode == 0
        assert "ZSTD=1" in r.stdout

    def test_label_collision_refused(self, tmp_path):
        run_main(tmp_path, base_cfg(), "--ssh-target", "u@h", "--ssh-key", "/k")
        r2 = run_main(tmp_path, base_cfg(), "--ssh-target", "u@h", "--ssh-key", "/k", out="steps2.txt")
        assert r2.returncode != 0
        assert "already exists" in (r2.stdout + r2.stderr)

    def test_rand_reproducible_same_seed(self, tmp_path):
        cfg = base_cfg(workload_family={"kernel": {"rand": 5}})
        run_main(tmp_path, cfg, "--ssh-target", "u@h", "--ssh-key", "/k", "--seed", "7", out="a.txt")
        # second run needs a fresh label + runs dir to avoid the collision guard
        cfg2 = base_cfg(label="qa_test2", workload_family={"kernel": {"rand": 5}})
        run_main(tmp_path, cfg2, "--ssh-target", "u@h", "--ssh-key", "/k", "--seed", "7", out="b.txt")
        assert (tmp_path / "a.txt").read_text() == (tmp_path / "b.txt").read_text()

    def test_reps_multiplies_cells(self, tmp_path):
        run_main(tmp_path, base_cfg(), "--ssh-target", "u@h", "--ssh-key", "/k", out="one.txt")
        cfg3 = base_cfg(label="qa3", reps=3)
        run_main(tmp_path, cfg3, "--ssh-target", "u@h", "--ssh-key", "/k", out="three.txt")
        n1 = len((tmp_path / "one.txt").read_text().strip().splitlines())
        n3 = len((tmp_path / "three.txt").read_text().strip().splitlines())
        assert n3 == 3 * n1  # 2 sandbox workloads x 3 reps

    def test_reps_all_distinct_no_collapse(self, tmp_path):
        run_main(tmp_path, base_cfg(reps=3), "--ssh-target", "u@h", "--ssh-key", "/k")
        lines = (tmp_path / "steps.txt").read_text().strip().splitlines()
        assert len(lines) == len(set(lines))  # no duplicates dropped

    def test_reps_rep0_keeps_baseline_seed(self, tmp_path):
        run_main(tmp_path, base_cfg(reps=3, workload_family={"sandbox": {"named": ["sandbox_ransom_seq"]}}),
                 "--ssh-target", "u@h", "--ssh-key", "/k", "--seed", "42")
        lines = (tmp_path / "steps.txt").read_text().strip().splitlines()
        seeds = sorted(int(l.split("--seed")[1].split()[0]) for l in lines)
        assert 42 in seeds  # rep 0 preserved the baseline seed
        assert len(seeds) == 3

    def test_reps_absent_matches_reps_one(self, tmp_path):
        run_main(tmp_path, base_cfg(), "--ssh-target", "u@h", "--ssh-key", "/k", out="absent.txt")
        run_main(tmp_path, base_cfg(label="qa1", reps=1), "--ssh-target", "u@h", "--ssh-key", "/k", out="one.txt")
        assert (tmp_path / "absent.txt").read_text() == (tmp_path / "one.txt").read_text()

    def test_no_cells_selected_exits(self, tmp_path):
        r = run_main(tmp_path, base_cfg(workload_family={"sandbox": {"none": None}}),
                     "--ssh-target", "u@h", "--ssh-key", "/k")
        assert r.returncode != 0
        assert "no cells" in (r.stdout + r.stderr)

    def test_named_cross_family_rejected(self, tmp_path):
        # a KERNEL workload under the SANDBOX key -- strict: reject with guidance
        cfg = base_cfg(workload_family={"sandbox": {"named": ["kernel_gemm_v2"]}})
        r = run_main(tmp_path, cfg, "--ssh-target", "u@h", "--ssh-key", "/k")
        assert r.returncode != 0
        msg = r.stdout + r.stderr
        assert "kernel_gemm_v2" in msg
        assert "'kernel'" in msg and "'sandbox'" in msg  # names both the real + given family
        assert not (tmp_path / "steps.txt").exists()  # strict = pre-selection, no partial file

    def test_named_same_family_ok(self, tmp_path):
        cfg = base_cfg(workload_family={"sandbox": {"named": ["sandbox_ransom_seq"]}})
        r = run_main(tmp_path, cfg, "--ssh-target", "u@h", "--ssh-key", "/k")
        assert r.returncode == 0
        assert "sandbox_ransom_seq" in (tmp_path / "steps.txt").read_text()

    def test_named_unknown_workload_rejected(self, tmp_path):
        cfg = base_cfg(workload_family={"sandbox": {"named": ["does_not_exist"]}})
        r = run_main(tmp_path, cfg, "--ssh-target", "u@h", "--ssh-key", "/k")
        assert r.returncode != 0
        assert "unknown workload" in (r.stdout + r.stderr)

    def test_no_partial_output_on_guard_failure(self, tmp_path):
        # raw without zstd-dir must be refused pre-flight -- no steps file written
        r = run_main(tmp_path, base_cfg(retention="raw"), "--ssh-target", "u@h", "--ssh-key", "/k")
        assert r.returncode != 0
        assert not (tmp_path / "steps.txt").exists(), "steps file written despite refused run"

    def test_ssh_target_required(self, tmp_path):
        # no --ssh-target -> refuse at generation (fail fast, not on the server)
        r = run_main(tmp_path, base_cfg())  # note: no --ssh-target
        assert r.returncode != 0
        assert "ssh-target" in (r.stdout + r.stderr)

    def test_over_ceiling_cell_skipped_not_aborted(self, tmp_path):
        # app family at scale 2.0 has cells that exceed the ceiling; scale 1.0 is
        # safe. Skip the over-ceiling ones, keep the baseline, do NOT abort.
        cfg = base_cfg(workload_family={"app": {"asc": "all"}}, scales=[1.0, 2.0])
        r = run_main(tmp_path, cfg, "--ssh-target", "u@h", "--ssh-key", "/k")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "skipped" in r.stdout
        lines = (tmp_path / "steps.txt").read_text().strip().splitlines()
        assert len(lines) > 0  # baseline cells survived
        meta = json.loads((tmp_path / "runs" / (cfg["label"] + ".json")).read_text())
        assert meta["skipped_over_ceiling"], "expected some skipped cells recorded"

    def test_all_cells_skipped_exits(self, tmp_path):
        # a single workload with only an over-ceiling scale -> nothing survives
        cfg = base_cfg(workload_family={"mem": {"named": ["mem_mmap_traversal_v2"]}},
                       scales=[4.0])  # --file-size-mb 1024 * 4 way over ceiling
        r = run_main(tmp_path, cfg, "--ssh-target", "u@h", "--ssh-key", "/k")
        assert r.returncode != 0
        assert "no cells" in (r.stdout + r.stderr)

    def test_no_partial_output_on_label_collision(self, tmp_path):
        run_main(tmp_path, base_cfg(), "--ssh-target", "u@h", "--ssh-key", "/k")
        # second run, same label, different out file -- must abort before writing it
        r2 = run_main(tmp_path, base_cfg(), "--ssh-target", "u@h", "--ssh-key", "/k", out="steps2.txt")
        assert r2.returncode != 0
        assert not (tmp_path / "steps2.txt").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
