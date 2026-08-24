from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict as ddict
import sys
import random
import re
import shlex
import subprocess
import zlib
from datetime import datetime, timezone

# label becomes ZSTD_RUN_ID, which becomes part of a real retention folder name
# (rep001__<runid> -- see DATABASE_DESIGN.md), so it must be filesystem-safe.
# NOTE: no ^...$ anchors -- in Python `$` also matches just before a trailing
# newline, so "run\n" would pass ^...$ and smuggle a newline into a path. Use
# fullmatch, which requires the WHOLE string to match end to end.
LABEL_RE = re.compile(r"[A-Za-z0-9_-]+")

FEATURES_BY_GROUP = {
    "amount": [  # family A -- 20
        "hamming","l0","l1","l2","linf","mean_abs","gradient_mag",
        "tv","chi2","hellinger","kl","js","wasserstein","bhattacharyya","hist_inter_dist",
        "ent_delta","csize_delta","ncd","struct_ent_change","lz_change"],
    "direction": [  # family B -- 19
        "cosine","pearson","ssim_struct","spearman","kendall",
        "mean_shift","ssim_lum","median_shift",
        "var_ratio","std_delta","ssim_contrast","range_delta","iqr_delta",
        "polarity","sign_delta_ent","zero_mass_delta",
        "cross_corr_lag","phase_corr","byte_rotation"],
    "content": [  # family C -- 13
        "ent_q","struct_ent_q","distinct_bytes","zero_frac","fill_frac","printable_frac",
        "mean_q","var_q","skew_q","kurt_q","chi2_uniform","bigram_ent","autocorr_peak"],
    "texture": [  # family D -- 12
        "changed_runs","change_span","change_centroid","longest_changed_run","change_density",
        "edge_energy","glcm_contrast","glcm_homogeneity","glcm_energy","glcm_correlation",
        "high_freq_frac","max_run_len"],
}

# One level finer than FEATURES_BY_GROUP -- this is what the Rust differ actually
# gates on. Mirrors the 12 structs in live_delta_calc_modular/src/metrics/family_*/mod.rs
# (positional/distributional/informational under family_a, structure/level/spread/
# polarity/dist_direction/spatial under family_b, family_c is flat with no submodules,
# change_location/texture under family_d). Nothing enforces agreement with that
# source -- if a submodule there changes, update this by hand.
SUBMODULE_COLUMNS = {
    "positional":      ["hamming","l0","l1","l2","linf","mean_abs","gradient_mag"],
    "distributional":  ["tv","chi2","hellinger","kl","js","wasserstein","bhattacharyya","hist_inter_dist"],
    "informational":   ["ent_delta","csize_delta","ncd","struct_ent_change","lz_change"],
    "structure":       ["cosine","pearson","ssim_struct","spearman","kendall"],
    "level":           ["mean_shift","ssim_lum","median_shift"],
    "spread":          ["var_ratio","std_delta","ssim_contrast","range_delta","iqr_delta"],
    "polarity":        ["polarity","sign_delta_ent"],
    "dist_direction":  ["zero_mass_delta"],
    "spatial":         ["cross_corr_lag","phase_corr","byte_rotation"],
    "content":         FEATURES_BY_GROUP["content"],  # family_c has no submodule split
    "change_location": ["changed_runs","change_span","change_centroid","longest_changed_run","change_density"],
    "texture":         ["edge_energy","glcm_contrast","glcm_homogeneity","glcm_energy","glcm_correlation",
                         "high_freq_frac","max_run_len"],
}

from generate_database_steps import (
    scale_command,
    family_of,
    binary_of,
)

HERE = Path(__file__).resolve().parent
BASE_STEPS = HERE / "full_campaign_steps.txt"

def _positive_number_list(cfg: dict, key: str, ints_only: bool = False) -> list:
    values = cfg[key]
    if not isinstance(values, list) or not values:
        sys.exit(f"invalid {key}: {values}, must be a non-empty list of positive numbers")
    # duration_s must be int: it lands verbatim as `--duration N` in each command
    # (scale_command does str(duration)), and both the guest workloads and the
    # orchestrator's SUSTAIN_LOOP regex `--duration\s+(\d+)` expect an integer.
    # A float writes `--duration 300.0`, which the guest can't parse and the
    # regex silently truncates to 300 -- host and guest then disagree.
    ok_type = (int,) if ints_only else (int, float)
    if not all(isinstance(v, ok_type) and not isinstance(v, bool) and v > 0 for v in values):
        kind = "positive integers" if ints_only else "positive numbers"
        sys.exit(f"invalid {key}: {values}, must be {kind}")
    return values


def load_config(json_steps: str):
    cfg = json.loads(Path(json_steps).read_text())
    for req in ("label", "duration_s", "scales", "retention", "capture_metric", "workload_family"):
        if req not in cfg:
            sys.exit(f"config missing from {json_steps}: {req}")

    if not isinstance(cfg["label"], str) or not LABEL_RE.fullmatch(cfg["label"]):
        sys.exit(f"invalid label: {cfg['label']!r}, must be a string matching "
                 f"{LABEL_RE.pattern} end to end (it becomes ZSTD_RUN_ID, part "
                 "of a retention folder name)")

    if cfg["retention"] not in ("raw", "metrics", "combined"):
        sys.exit(f"invalid retention: {cfg['retention']}, must be one of raw, metrics, combined")

    if cfg["capture_metric"] not in ("delta", "apf", "apf_queue", "substrate"):
        sys.exit(f"invalid capture_metric: {cfg['capture_metric']}, "
                 "must be one of delta, apf, apf_queue, substrate")

    _positive_number_list(cfg, "duration_s", ints_only=True)
    _positive_number_list(cfg, "scales")  # scales are legitimately fractional (0.5, 0.25)

    # reps: optional, default 1. Capture each (workload, duration, scale) N times
    # with a distinct seed per rep (see rep_seed). Absent -> 1 (today's behaviour).
    reps = cfg.get("reps", 1)
    if not (isinstance(reps, int) and not isinstance(reps, bool) and reps >= 1):
        sys.exit(f"invalid reps: {reps}, must be a positive integer")
    cfg["reps"] = reps  # normalise so downstream always sees an int

    # capture_metric_group is a SEPARATE, optional key from capture_metric (the
    # mode above) -- the per-submodule feature selection, with no backend yet
    # (see metric_group_env). Absent from the config -> no group validated, no
    # features requested. It is NOT "all" by default.
    feats = cfg.get("capture_metric_group")
    if feats is not None:
        for group, selected in feats.items():
            if group not in FEATURES_BY_GROUP:
                sys.exit(f"invalid feature group: {group}, must be one of {list(FEATURES_BY_GROUP.keys())}")
            if selected not in ("all", "none") and not isinstance(selected, list):
                sys.exit(f"invalid feature selection: {selected}, must be 'all', 'none', or a list of columns")
            if isinstance(selected, list):
                wrong = set(selected) - set(FEATURES_BY_GROUP[group])
                if wrong:
                    sys.exit(f"invalid features for {group}: {wrong}, must be one of {FEATURES_BY_GROUP[group]}")

    # workload_family: {family_name: {order_type: order_value}}, e.g.
    # {"sandbox": {"all": null}, "kernel": {"asc": 8}, "io": {"named": ["io_read_cache_hit_v2"]}}
    wrkld_family_dict = cfg["workload_family"]
    if not isinstance(wrkld_family_dict, dict) or not wrkld_family_dict:
        sys.exit(f"invalid workload_family: {wrkld_family_dict}, must be a non-empty dict of "
                 "family -> {order_type: order_value}")
    for fam, order in wrkld_family_dict.items():
        if not isinstance(order, dict) or len(order) != 1:
            sys.exit(f"invalid workload_family[{fam}]: {order}, must be a single-key dict, e.g. {{'asc': 8}}")
        order_type, order_value = next(iter(order.items()))
        if order_type not in ("asc", "desc", "rand", "named", "none"):
            sys.exit(f"invalid order type for {fam}: {order_type}, must be one of asc, desc, rand, named, none")
        if order_type in ("asc", "desc", "rand"):
            valid = order_value == "all" or (isinstance(order_value, int) and not isinstance(order_value, bool))
            if not valid:
                sys.exit(f"invalid order value for {fam}.{order_type}: {order_value}, must be 'all' or an integer")
        elif order_type == "named":
            if not (isinstance(order_value, list) and order_value
                    and all(isinstance(x, str) for x in order_value)):
                sys.exit(f"invalid named list for {fam}: {order_value}, must be a "
                         "non-empty list of workload names (use {'none': null} to "
                         "select nothing)")
        # order_type == "none" takes no order_value shape check -- any value is ignored

    return cfg


def add_env_variables(cmd: str, env_vars: dict[str, str]) -> str:
    """Prefix `cmd` with `NAME=value ` for every entry in env_vars.

    `cmd` must be the ORCHESTRATOR launch line (./run_files_controlled.py), not a
    guest workload line -- these vars are read by os.environ.get on the host.
    Names are upper-cased because that lookup is case-sensitive; values are
    shlex-quoted so a path with spaces can't break the launch line.
    """
    prefix = "".join(
        f"{var.upper()}={shlex.quote(str(value))} " for var, value in env_vars.items()
    )
    return f"{prefix}{cmd}"


def add_steps_file_envvar(cmd: str, steps_file: str) -> str:
    if not Path(steps_file).exists():
        sys.exit(f"STEPS_FILE {steps_file} does not exist")
    return add_env_variables(cmd, {"STEPS_FILE": str(steps_file)})


def add_constant_envvars(cmd: str, capture_mode=1, ssh_wait_timeout=300,
                         continue_on_failure=1, min_free_disk_gb=40,
                         pythonunbuffered=1) -> str:
    env_vars = {
        "CAPTURE_MODE": str(capture_mode),
        "SSH_WAIT_TIMEOUT": str(ssh_wait_timeout),
        "CONTINUE_ON_FAILURE": str(continue_on_failure),
        "MIN_FREE_DISK_GB": str(min_free_disk_gb),
        "PYTHONUNBUFFERED": str(pythonunbuffered),
    }
    return add_env_variables(cmd, env_vars)



def retention_env(retention: str) -> dict[str, str]:
    """Map the config's retention choice to the ZSTD switch the host reads.

    Retention controls ONE bit: keep the raw page-dump chains, or drop them. The
    live metric trajectory (CAPTURE_METRIC) is written regardless, so `raw` and
    `combined` are the same at the env level (both keep the chains).
        raw, combined -> ZSTD=1   (chains kept, zstd-compressed, in ZSTD_DIR)
        metrics       -> {}       (chains dropped after diffing; trajectory only)
    ZSTD_DIR (the host path) is required when ZSTD=1 but is host-specific, so the
    caller adds it from a CLI flag -- it is not derived here.
    """
    if retention in ("raw", "combined"):
        return {"ZSTD": "1"}
    if retention == "metrics":
        return {}
    sys.exit(f"invalid retention: {retention}, must be one of raw, metrics, combined")


def capture_metric_env(capture_metric: str) -> dict[str, str]:
    """Map the config's capture_metric (the capture MODE) to CAPTURE_METRIC.

    One of delta / apf / apf_queue / substrate -- selects which differ/producer
    the host runs. run_files_controlled.py reads CAPTURE_METRIC with .strip().lower()
    and dispatches on exact match, so we normalise the same way. The per-submodule
    feature selection (capture_metric_group) is a separate axis with no backend
    yet, so it is NOT emitted here.
    """
    mode = str(capture_metric).strip().lower()
    if mode not in ("delta", "apf", "apf_queue", "substrate"):
        sys.exit(f"invalid capture_metric: {capture_metric}, "
                 "must be one of delta, apf, apf_queue, substrate")
    return {"CAPTURE_METRIC": mode}


def resolve_features(feats: dict) -> str:
    """Config's capture_metric_group -> the submodule list SUBSTRATE_FEATURES
    would carry. Two steps, group first:

      1. per GROUP (amount/direction/content/texture), expand its selection
         ("all" | "none" | [column, ...]) into the actual requested columns.
      2. per SUBMODULE (finer than group, SUBMODULE_COLUMNS), enable it if ANY
         of its columns was requested. Gating is submodule-level and all-or-
         nothing -- e.g. requesting texture's "edge_energy" alone still enables
         the whole texture submodule (confirmed: texture::compute in
         family_d/mod.rs is one call producing all 7 columns, GLCM included).

    A group absent from `feats` contributes no columns -- it is not "all" by
    default. feats={} or all-"none" groups -> "" (SUBSTRATE_FEATURES unset).
    """
    wanted: set[str] = set()
    for group, selected in feats.items():
        if selected == "all":
            wanted |= set(FEATURES_BY_GROUP[group])
        elif selected == "none":
            pass
        else:
            wanted |= set(selected)

    enabled = [sm for sm, cols in SUBMODULE_COLUMNS.items() if wanted & set(cols)]
    return ",".join(enabled)


def metric_group_env(feats: dict | None) -> dict[str, str]:
    """SUBSTRATE_FEATURES from the config's capture_metric_group.

    NOT yet consumed by anything: live_delta_calc_modular/src/main.rs has no
    --features flag, and run_files_controlled.py never reads SUBSTRATE_FEATURES.
    Emitting this is a no-op on today's orchestrator -- kept so the resolved
    value is correct and ready ahead of that Rust/orchestrator wiring.
    """
    if not feats:
        return {}
    spec = resolve_features(feats)
    return {"SUBSTRATE_FEATURES": spec} if spec else {}


def rep_seed(base_seed: int, rep: int, workload: str) -> int:
    """Per-rep seed for the `reps` feature. Rep 0 reuses base_seed unchanged so
    the first capture of every cell reproduces the proven campaign command (and a
    reps>1 config never mutates the cell a reps=1/absent config would produce).
    Reps 1..N-1 vary, decorrelated per workload via a STABLE hash -- zlib.crc32,
    not Python's per-process-salted hash(), so the output is reproducible across
    runs and machines.
    """
    if rep == 0:
        return base_seed
    return base_seed + rep * 1000 + zlib.crc32(workload.encode()) % 997


def _git_sha() -> str | None:
    """Best-effort short git SHA of the generating code, for provenance. None if
    not in a git checkout (e.g. copied standalone to the server)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def zstd_run_id_env(label: str) -> dict[str, str]:
    """ZSTD_RUN_ID from the config's label -- ties every chain this subset
    writes back to runs/INDEX.tsv (DATABASE_DESIGN.md's run registration).
    load_config already enforces LABEL_RE, so the value is known path-safe.
    """
    return {"ZSTD_RUN_ID": label}


def host_env(ssh_target: str | None, ssh_key: str | None, zstd_dir: str | None) -> dict[str, str]:
    """Host-specific identity: SSH target/key and the ZSTD retention path.

    Deliberately NOT config keys -- the config describes the experiment and
    should stay portable across machines ("move to cloud later is just rsync",
    DATABASE_DESIGN.md); these three are specific to whichever host is running
    it right now, so they come from CLI flags instead.

    Only includes what was actually given. Whether ZSTD_DIR is REQUIRED depends
    on retention (only matters when retention_env produced ZSTD=1) -- that
    check belongs where the two are combined, not here.
    """
    env: dict[str, str] = {}
    if ssh_target:
        env["SSH_TARGET"] = ssh_target
    if ssh_key:
        env["SSH_KEY"] = ssh_key
    if zstd_dir:
        env["ZSTD_DIR"] = zstd_dir
    return env


def build_cells(cfg: dict, base_seed: int) -> tuple[list[dict], int, list[str]]:
    """Selection + crossing -> the exact cells this config produces.

    Returns (cells, dropped, skipped). Each cell is a dict
    {family, workload, duration, scale, rep, cmd}. Self-contained and
    deterministic: seeds Python's RNG from base_seed so 'rand' orderings
    reproduce. This is the ONE generator -- main() writes its cmds to the steps
    file, and the console bridge reads it via --print-cells, so the UI preview
    and the launched steps share identical logic.

    Crossing is workload x duration x scale x rep. rep varies the --seed only
    (rep_seed); dedup drops scale-collapsed duplicates; over-ceiling (scale-up
    beyond the guest-safe max, per scale_command strict) SKIPS the whole cell and
    logs it -- baseline scale=1.0 can never be skipped (value == base <= ceiling).
    """
    random.seed(base_seed)
    base_cmd_lst = [l.strip() for l in BASE_STEPS.read_text().splitlines()
                    if l.strip() and not l.strip().startswith("#")]
    cmd_family_dict: dict[str, list[str]] = ddict(list)
    cmd_binary_dict: dict[str, list[str]] = ddict(list)
    for cmd in base_cmd_lst:
        cmd_binary_dict[binary_of(cmd)].append(cmd)
        cmd_family_dict[family_of(binary_of(cmd))].append(cmd)

    wrkld_family_choices_dct: dict[str, list[str]] = {}
    for fam, order in cfg["workload_family"].items():
        if fam not in cmd_family_dict:
            sys.exit(f"invalid workload family: {fam}, must be one of {list(cmd_family_dict.keys())}")

        order_type, order_value = next(iter(order.items()))
        pool = cmd_family_dict[fam]
        # Sort/select by WORKLOAD NAME, not the raw command string -- sorting
        # command text mixes interpreter prefixes (python3 vs /bin/...) into the
        # ordering (confirmed wrong for the 'app' family).
        if order_type in ("asc", "desc"):
            ordered = sorted(pool, key=binary_of, reverse=(order_type == "desc"))
            if order_value == "all":
                wrkld_family_choices_dct[fam] = ordered
            else:
                if not (0 < order_value <= len(pool)):
                    sys.exit(f"invalid order value for {fam}.{order_type}: {order_value}, "
                             f"must be between 1 and {len(pool)}")
                wrkld_family_choices_dct[fam] = ordered[:order_value]

        elif order_type == "rand":
            pool_copy = list(pool)
            random.shuffle(pool_copy)  # shuffles in place; returns None
            if order_value == "all":
                wrkld_family_choices_dct[fam] = pool_copy
            else:
                if not (0 < order_value <= len(pool)):
                    sys.exit(f"invalid order value for {fam}.rand: {order_value}, "
                             f"must be between 1 and {len(pool)}")
                wrkld_family_choices_dct[fam] = pool_copy[:order_value]

        elif order_type == "named":
            family_names = {binary_of(c) for c in pool}
            for name in order_value:
                if name in family_names:
                    continue
                if name in cmd_binary_dict:
                    real_fam = family_of(name)
                    sys.exit(
                        f"workload_family[{fam!r}] names {name!r}, which is a "
                        f"{real_fam!r} workload, not {fam!r}. Either move it under a "
                        f"{real_fam!r} key, or pick a {fam!r} workload: "
                        f"{sorted(family_names)}")
                sys.exit(f"workload_family[{fam!r}] names unknown workload {name!r}. "
                         f"{fam!r} workloads are: {sorted(family_names)}")
            wrkld_family_choices_dct[fam] = [
                c for name in order_value for c in cmd_binary_dict[name]
            ]

        elif order_type == "none":
            wrkld_family_choices_dct[fam] = []

    cells: list[dict] = []
    seen: set[str] = set()
    dropped = 0
    skipped: list[str] = []
    for fam, choices in wrkld_family_choices_dct.items():
        for cmd in choices:
            workload = binary_of(cmd)
            for duration in cfg["duration_s"]:
                for scale_factor in cfg["scales"]:
                    for rep in range(cfg["reps"]):
                        seed = rep_seed(base_seed, rep, workload)
                        try:
                            scaled_cmd = scale_command(cmd, scale_factor, seed, duration, strict=True)
                        except ValueError as e:
                            # ceiling depends on scale, not seed/rep -- identical
                            # for every rep, so log once and skip the whole cell.
                            skipped.append(f"{fam}/{workload} @ scale={scale_factor}: {e}")
                            break
                        if scaled_cmd in seen:
                            dropped += 1
                            continue
                        seen.add(scaled_cmd)
                        cells.append({"family": fam, "workload": workload,
                                      "duration": duration, "scale": scale_factor,
                                      "rep": rep, "cmd": scaled_cmd})
    return cells, dropped, skipped


def apply_curation(cells: list[dict], curation: list[dict]) -> list[dict]:
    """Filter + reorder `cells` to exactly the (workload,duration,scale,rep) keys
    listed in `curation`, in that order. This is how the UI's per-cell curation
    (uncheck / reorder) reaches the steps file WITHOUT the UI ever sending a raw
    command -- keys select from the trusted, re-derived cells. A key not present
    in the plan is an error (the config or seed must have changed under the UI).
    Note: (1, 1.0) hash/compare equal in Python, so int/float scale mismatches in
    the JSON round-trip still match.
    """
    by_key = {(c["workload"], c["duration"], c["scale"], c["rep"]): c for c in cells}
    out: list[dict] = []
    for k in curation:
        key = (k["workload"], k["duration"], k["scale"], k["rep"])
        if key not in by_key:
            sys.exit(f"curation references a cell not in the generated plan: {key} "
                     "-- the config or seed changed; regenerate the plan")
        out.append(by_key[key])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="JSON file containing steps")
    ap.add_argument("--out", type=str, default="subset_run.txt", help="write subset steps here")
    ap.add_argument("--seed", type=int, default=42, help="shuffle/seed")
    ap.add_argument("--print-cells", action="store_true",
                    help="emit the generated cells as JSON and exit (no files written, "
                         "no ssh-target/zstd-dir required) -- used by the console bridge")
    ap.add_argument("--curation", type=str, default=None,
                    help="JSON file: list of {workload,duration,scale,rep} cells to keep, "
                         "in order (the UI's per-cell curation). Absent = keep all, generation order")
    ap.add_argument("--ssh-target", type=str, default=None,
                    help="user@host for the guest VM (host-specific, not in config)")
    ap.add_argument("--ssh-key", type=str, default=None,
                    help="path to the SSH private key (host-specific, not in config)")
    ap.add_argument("--zstd-dir", type=str, default=None,
                    help="host path for ZSTD raw-chain retention; required when "
                         "retention is raw/combined")
    ap.add_argument("--runs-dir", type=str, default="runs",
                    help="directory for the per-subset <label>.json metadata record")
    a = ap.parse_args()

    cfg = load_config(a.config)

    # Generate the cells (selection + crossing) -- side-effect-free. Same logic
    # main() and the console bridge share, so the UI preview and the launched
    # steps are identical.
    cells, dropped, skipped = build_cells(cfg, a.seed)

    # --print-cells: emit the plan as JSON and exit BEFORE the write-path guards,
    # so a preview needs neither --ssh-target nor --zstd-dir and writes nothing.
    if a.print_cells:
        print(json.dumps({"cells": cells, "dropped": dropped, "skipped": skipped}))
        return 0

    # Pre-flight guards for the WRITE/LAUNCH path -- fail before writing any file.
    # SSH_TARGET has no orchestrator default and is required there
    # (run_files_controlled.py:1130). --ssh-key is NOT forced: SSH can auth via
    # agent or SSH_PASS.
    if not a.ssh_target:
        sys.exit("--ssh-target is required (user@host) -- the orchestrator cannot "
                 "reach the guest without it")

    meta_path = Path(a.runs_dir) / f"{cfg['label']}.json"
    if meta_path.exists():
        # label == ZSTD_RUN_ID: reusing it commingles this run's chains with an
        # earlier one under the same runid. Refuse rather than overwrite.
        sys.exit(f"run metadata {meta_path} already exists -- label {cfg['label']!r} "
                 "was used before; pick a unique label so runs stay separable")
    if retention_env(cfg["retention"]).get("ZSTD") == "1" and not a.zstd_dir:
        # raw/combined retention sets ZSTD=1, which REQUIRES ZSTD_DIR or the
        # orchestrator warns and silently skips all delta retention (:1166).
        sys.exit(f"retention={cfg['retention']} needs raw-chain storage, but "
                 "--zstd-dir was not given (host-specific path, required for "
                 "raw/combined retention)")

    # Curation (the UI's per-cell keep/reorder): filter+reorder the trusted cells
    # to exactly the keys the UI chose. Absent -> keep all, generation order.
    if a.curation:
        cells = apply_curation(cells, json.loads(Path(a.curation).read_text()))

    if dropped:
        print(f"[subset_run] dropped {dropped} duplicate cell(s) -- "
              "different scale collapsed to the same command after clamping")
    if skipped:
        print(f"[subset_run] skipped {len(skipped)} cell(s) exceeding the guest-safe "
              "ceiling (baseline scale=1.0 is never affected):")
        for s in skipped:
            print(f"  - {s}")

    steps = [c["cmd"] for c in cells]
    if not steps:
        sys.exit("no cells selected -- every cell was skipped, curated out, or the "
                 "workload_family selection was empty; check the config")

    out_path = Path(a.out).resolve()  # absolute: the launch line runs
    # ./run_files_controlled.py from VM_Capture_QEMU/, a different cwd than where
    # --out may point, so a relative STEPS_FILE would not resolve there.
    out_path.write_text("\n".join(steps) + "\n")
    print(f"[subset_run] wrote {len(steps)} cells -> {out_path}")

    # Assemble the orchestrator launch line from every env source. Order is
    # cosmetic (env assignments are unordered to the shell); grouped by origin
    # for readability. All of these are read by os.environ.get on the HOST --
    # none reach the guest (that's what the steps file above is for).
    run_env: dict[str, str] = {}
    run_env.update(retention_env(cfg["retention"]))
    run_env.update(capture_metric_env(cfg["capture_metric"]))
    run_env.update(metric_group_env(cfg.get("capture_metric_group")))
    run_env.update(zstd_run_id_env(cfg["label"]))
    run_env.update(host_env(a.ssh_target, a.ssh_key, a.zstd_dir))
    # (the ZSTD-needs-ZSTD_DIR guard already ran pre-flight, above)

    launch = add_constant_envvars("./run_files_controlled.py")
    launch = add_steps_file_envvar(launch, str(out_path))
    launch = add_env_variables(launch, run_env)

    # Minimal run registration: record what produced this subset so traces from
    # many runs stay comparable (DATABASE_DESIGN.md's <runid>.json). The INDEX.tsv
    # + symlink view are reconstructable later from these files -- every chain the
    # orchestrator writes carries __<runid> (== label) in its path, so a post-run
    # tool can glob them; nothing here is a source of truth that gets lost.
    # meta_path + collision check already done pre-flight (above); just create
    # the dir and write. A concurrent run could still race in between, but this
    # is a single-user workflow -- the pre-flight check covers the real case
    # (re-running the same config) without a partial steps file on abort.
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "label": cfg["label"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "cells": len(steps),
        "reps": cfg["reps"],
        "dropped_duplicates": dropped,
        "skipped_over_ceiling": skipped,
        "curated": a.curation is not None,
        "seed": a.seed,
        "steps_file": str(out_path.resolve()),
        "config": cfg,
        "launch_line": launch,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[subset_run] wrote run metadata -> {meta_path}")

    print("[subset_run] launch the capture with:\n")
    print(f"  {launch}\n")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())