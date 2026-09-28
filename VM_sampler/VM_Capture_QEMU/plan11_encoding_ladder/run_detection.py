#!/usr/bin/env python3
"""run_detection.py -- the driver for the detection moves D0 to D15 in al-Kindi's order (K3 Sec.
4; SPEC_DETECTION section 6), reusing run_moves.py's ledger, skip and staleness machinery through
the additive extensions of SPEC_DETECTION 1.3 (`run_plan(..., max_move=15,
ledger_name="driver_detection_state.json", internal_steps=...)`).

Builder B (report), 2026-09-17. Every command is a subprocess `python3 -m
plan11_encoding_ladder.<module> <sub> ...` from the package's parent directory, recorded in
`<out>/driver_detection_state.json` with its command line, start, end, exit code and the keyed
hashes of its declared inputs; the driver skips a command whose outputs exist and whose inputs
still hash as recorded, re-runs it when an input changed (the ledger names the part), never
overwrites an author input (`kept: author input exists`), and stops at the first non-zero exit.
The declared inputs of every step include `inputs/classes.csv`, `cells.csv`,
`gates/detection/cell_classes.csv`, `gates/detection/admissibility.csv`, `gates/selection.json`
(per rung) and `gates/detection/gk0_cells.csv` where the step reads them, so a change to the
class file, the selection or the floor makes every later move stale on its own (SPEC_DETECTION
6.1). Two internal steps: `features-at-selection <rung>` (the feature files at the inherited
grid point, raw and norm, norm only for combined) and `tripwire-check` (D15).

Corrections from the three SPEC_DETECTION reviews carried by the driver:
  - al-Kindi 1: `--threshold-source inner_lowo` is the default passed to every two-class split
    and to the ladder (`oob` stays as the labelled alternative).
  - al-Kindi 2 / ML 2.7: the one-class run takes `--null-perm` and `--n-jobs`; `--null-splits`
    accepts `one_class` and defaults to `lowo,loco,one_class`.
  - al-Kindi 5: `--c1-rule report` applies to every class alike (the flag is passed through).
  - al-Kindi 6 / al-Farabi M5: `gates_detection head-drop-template` writes one head-drop row per
    workload key of the join at D2 (template; the author's file is never overwritten).
  - al-Kindi 12 / 7: `figures_detection` takes `--identity` and `--plane-per-letter`.
  - ML 2.3: `gates_detection gop` runs for `lowo`, `loco` and `lofo`.
  - ML 2.6: `gates_detection leak-probe` runs at D7 (the cadence and active-fraction probes).
  - ML 2.5 / al-Farabi M2: `--train-on-at-floor true|false` is passed through when given.
  - ML 2.1: `--row-unit cell|window` is passed through when given (the constant otherwise).
  - al-Farabi M10: a differing `--classes` against an existing `inputs/classes.csv` is refused
    and the refusal is recorded as the D0 command's status in the ledger.
  - al-Farabi M12: `--order-scope within_class|campaign` is passed through when given.

CLI (SPEC_DETECTION 6.1):
  run_detection.py run --out O --root R --classes CSV (--selection-from PATH | --grid-default W8_H4)
        [--moves 0-15] [--assume-failed-zero --assume-reason TEXT] [--c1-activity-min F] [--c1-rule report|exclude]
        [--campaign-label TEXT] [--n-jobs 1] [--null-perm 500] [--null-splits lowo,loco,one_class]
        [--null-rungs apf,wapf,persist,content,combined] [--ladder-null-perm 0] [--loco-mode cell|rep_index]
        [--one-class-model isolation_forest] [--level-quantity median_K|per_iteration_K_sum]
        [--order-consequence size|void] [--table10-rung combined] [--seed-offset 0] [--force] [--dry-run]
        [--skip-missing-modules] [--only-modules M,...] [--standalone-tex PATH] ...
  run_detection.py status --out O
  run_detection.py plan --out O ... [--moves 0-15]
Exit 0 on success (a written refusal is a success), 2 when a required input is missing (its path
on stderr), 1 when a move fails (the failing command and its exit code are in the ledger).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse
import shutil
import traceback

from plan11_encoding_ladder import run_moves  # noqa: E402  (the additive extensions of SPEC_DETECTION 1.3)
from plan11_encoding_ladder._report_common import (  # noqa: E402
    PACKAGE_NAME, RUNGS, not_run, now_iso, read_csv, read_json, refused, result_json, sha256_file, write_json,
)
from plan11_encoding_ladder.run_moves import _cmd, load_ledger, print_status, run_plan, save_ledger  # noqa: E402

CITATION = "K3 Sec. 4 (the blind moves); SPEC_DETECTION section 6; SPEC_DETECTION reviews al-Kindi 1, 2, 5, 6, 7, 12; ML 2.1, 2.3, 2.5, 2.6, 2.7; al-Farabi M5, M10, M12"
LEDGER = "driver_detection_state.json"
MAX_MOVE = 15
GRID_DEFAULT = "W8_H4"
THRESHOLD_SOURCE_DEFAULT = "inner_lowo"          # al-Kindi review 1 (the SPEC's `oob` is the labelled alternative)
NULL_SPLITS_DEFAULT = "lowo,loco,one_class"       # ML review 2.7; al-Kindi review 2
RUNG_ORDER = ("apf", "persist", "content", "wapf", "combined")   # D7, D9, D10, D11, D12 (K3 moves 10, 12, 13, 14, 16)
MOVE_OF_RUNG = {"apf": 7, "persist": 9, "content": 10, "wapf": 11, "combined": 12}
CLASSES = "inputs/classes.csv"
JOIN = "gates/detection/cell_classes.csv"
ADM = "gates/detection/admissibility.csv"
SEL = "gates/selection.json"
FLOOR = "gates/detection/gk0_cells.csv"
PRE = "gates/preconditions.json"
# the commands that take --seed-offset and --n-jobs (checked against builder A's argparse definitions on 2026-09-17:
# leak-probe, drift and alias take --seed-offset but not --n-jobs)
RANDOM_COMMANDS = {("detection_metrics", "splits"), ("detection_metrics", "one-class"), ("detection_metrics", "ladder"),
                   ("detection_levels", "level2"), ("detection_levels", "level3"), ("gates_detection", "glm"), ("gates_detection", "anchor"),
                   ("gates_detection", "order"), ("gates_detection", "drift"), ("gates_detection", "gfp"), ("gates_detection", "gm"),
                   ("gates_detection", "leak-probe"), ("gates_detection", "alias"), ("gates_precondition", "gf"), ("gates_comparison", "gx")}
NJOBS_COMMANDS = {("detection_metrics", "splits"), ("detection_metrics", "one-class"), ("detection_metrics", "ladder"),
                  ("detection_levels", "level2"), ("detection_levels", "level3"), ("gates_detection", "glm"), ("gates_detection", "anchor"),
                  ("gates_detection", "order"), ("gates_detection", "gfp"), ("gates_detection", "gm"),
                  ("gates_precondition", "gf"), ("gates_comparison", "gx")}


def _sel_input(rung: str) -> str:
    return f"json:{SEL}:{rung}"


def build_plan(o: argparse.Namespace) -> list[dict]:
    """The move table of SPEC_DETECTION 6.2 with the review corrections in the module docstring."""
    out = str(Path(o.out))
    O = ["--out", out]
    P = []
    null_rungs = set(x.strip() for x in (o.null_rungs or "").split(",") if x.strip())
    common_inputs = [CLASSES, "cells.csv", JOIN, ADM]
    # ---- D0: identity (K3 move 1)
    P.append(_cmd(0, "extract index", "extract", "index", ["--root", o.root or "<root required>", *O], outputs=["cells.pre_classes.csv|cells.csv"]))
    P.append(_cmd(0, "classes validate", "classes", "validate", [*O, "--classes", f"{out}/inputs/classes.csv", "--relaunched-grouping", o.relaunched_grouping],
                  outputs=["gates/detection/classes_validation.json"], inputs=[CLASSES, "cells.csv"]))
    a = [*O, "--kernel-family-rule", o.kernel_family_rule, "--relaunched-grouping", o.relaunched_grouping]
    if o.campaign_label:
        a += ["--campaign-label", o.campaign_label]
    P.append(_cmd(0, "classes apply", "classes", "apply", a, outputs=[JOIN, "gates/detection/cell_classes.json"], inputs=[CLASSES, "cells.csv"]))
    if o.selection_from:
        a = [*O, "--from", o.selection_from]
    else:
        a = [*O, "--default", o.grid_default or GRID_DEFAULT]
    if o.force:
        a += ["--force"]
    P.append(_cmd(0, "classes inherit-selection", "classes", "inherit-selection", a, outputs=[SEL, "gates/detection/inherit_selection.json"],
                  inputs=([o.selection_from] if o.selection_from else [])))
    P.append(_cmd(0, "classes letter-sequence", "classes", "letter-sequence", [*O], outputs=["gates/detection/letter_sequence.csv"], inputs=[CLASSES, JOIN]))
    # ---- D1: the extracts
    a = ["--cells-csv", f"{out}/cells.csv", *O, "--jobs", o.n_jobs]
    if o.persist_side:
        a += ["--persist-side", o.persist_side]
    if o.failed_counts:
        a += ["--failed-counts", o.failed_counts]
    P.append(_cmd(1, "extract all", "extract", "all", a, outputs=["extract"], inputs=["cells.csv", "inputs/failed_counts.csv"]))
    # ---- D2: preconditions, the templates, G-P, admissibility (K3 move 2)
    a = [*O]
    if o.assume_failed_zero:
        a += ["--assume-failed-zero", "--assume-reason", o.assume_reason]
    if o.failed_counts:
        a += ["--failed-counts", o.failed_counts]
    if o.c1_activity_min is not None:
        a += ["--c1-activity-min", o.c1_activity_min]
    P.append(_cmd(2, "preconditions", "gates_precondition", "preconditions", a, outputs=["gates/preconditions.csv", PRE], inputs=["cells.csv", "inputs/failed_counts.csv"]))
    P.append(_cmd(2, "pass-table template", "gates_calibration", "pass-table", [*O], outputs=["inputs/pass_table.csv"], template=True))
    P.append(_cmd(2, "gk0 template", "gates_precondition", "gk0-template", [*O], outputs=["inputs/gk0_source.csv"], template=True))
    P.append(_cmd(2, "idle admissibility template", "gates_precondition", "idle-admissibility-template", [*O], outputs=["inputs/idle_admissibility.json"], template=True))
    # al-Farabi M5 / al-Kindi 6: one head-drop row per workload key of the join (every class), default 0, declared at D2
    P.append(_cmd(2, "head-drop template (every workload key)", "gates_detection", "head-drop-template", [*O], outputs=["inputs/head_drop.csv"], template=True))
    P.append(_cmd(2, "gk0 sandbox source template", "gates_detection", "gk0-sandbox-template", [*O], outputs=["inputs/gk0_source_sandbox.csv"], template=True))
    P.append(_cmd(2, "gp", "gates_calibration", "gp", [*O], outputs=["gates/gp.csv"], inputs=["cells.csv", "inputs/pass_table.csv"]))
    P.append(_cmd(2, "admissibility", "gates_detection", "admissibility", [*O, "--c1-rule", o.c1_rule], outputs=[ADM, "gates/detection/admissibility.json"],
                  inputs=[CLASSES, "cells.csv", JOIN, "gates/preconditions.csv", PRE]))
    # ---- D3: G-C per rung, combined last (K3 move 3)
    for rung in ("apf", "persist", "content", "wapf", "combined"):
        P.append(_cmd(3, f"gc {rung}", "gates_calibration", "gc", [*O, "--rung", rung], outputs=["gates/gc.csv"], inputs=["cells.csv", "inputs/head_drop.csv", PRE]))
    # ---- D4: floors, the level map, the features at the inherited grid (K3 moves 4, 5)
    P.append(_cmd(4, "gk0", "gates_precondition", "gk0", [*O], outputs=["gates/gk0.csv"], inputs=["cells.csv", "inputs/gk0_source.csv", "inputs/head_drop.csv", PRE]))
    P.append(_cmd(4, "gk0-cells (two-class)", "gates_detection", "gk0-cells", [*O], outputs=[FLOOR, "gates/detection/gk0_members.csv", "gates/detection/gk0.json"],
                  inputs=common_inputs + ["inputs/gk0_source_sandbox.csv", "gates/gk0.csv", "inputs/head_drop.csv"]))
    P.append(_cmd(4, "gn (two-class)", "gates_detection", "gn", [*O], outputs=["gates/detection/gn.csv"], inputs=common_inputs + [FLOOR]))
    for rung in RUNGS:
        variants = "norm" if rung == "combined" else "raw,norm"
        P.append(_cmd(4, f"features at selection {rung}", "driver", "features-at-selection", [rung], internal=True,
                      outputs=[f"json:gates/detection/features_at_selection.json:{rung}"],
                      inputs=["cells.csv", "inputs/head_drop.csv", _sel_input(rung), ADM]))
    P.append(_cmd(4, "gf all rungs at the selected points", "gates_precondition", "gf", [*O, "--all-rungs", "--n-perm", o.null_perm], outputs=["gates/gf.csv"],
                  inputs=["cells.csv", "inputs/idle_admissibility.json", "inputs/head_drop.csv", SEL, PRE]))
    P.append(_cmd(4, "anchor idle_sets", "gates_detection", "anchor", [*O, "--part", "idle_sets", "--null-perm", o.null_perm], outputs=["csv:gates/detection/ganchor.csv:part=idle_sets"],
                  inputs=common_inputs + [SEL, FLOOR]))
    P.append(_cmd(4, "anchor idle_early_late", "gates_detection", "anchor", [*O, "--part", "idle_early_late", "--null-perm", o.null_perm],
                  outputs=["csv:gates/detection/ganchor.csv:part=idle_early_late"], inputs=common_inputs + [SEL, FLOOR]))
    P.append(_cmd(4, "drift", "gates_detection", "drift", [*O, "--null-perm", o.null_perm], outputs=["gates/detection/drift.csv"], inputs=common_inputs + [FLOOR]))
    P.append(_cmd(4, "figures fig2_three_floors,fig_level_map", "figures_detection", None, [*O, "--only", "fig2_three_floors,fig_level_map"],
                  outputs=["report/detection/figures/fig2_three_floors.pdf|report/detection/figures/SKIPPED.txt"], inputs=common_inputs + [FLOOR]))
    # ---- D5: APF(t) per tier (K3 move 6)
    P.append(_cmd(5, "figures fig_apf_per_tier", "figures_detection", None, [*O, "--only", "fig_apf_per_tier"],
                  outputs=["report/detection/figures/fig_apf_per_tier.pdf|report/detection/figures/SKIPPED.txt"], inputs=common_inputs))
    # ---- D6: the harness clause (K3 moves 7, 8)
    for rung in RUNGS:
        P.append(_cmd(6, f"harness {rung}", "gates_detection", "harness", [*O, "--rung", rung], outputs=["csv:gates/detection/harness.csv:rung=" + rung],
                      inputs=common_inputs + [_sel_input(rung), FLOOR]))

    def rung_stage(move: int, rung: str):
        variant = "--raw-and-norm" if rung == "apf" else "--norm"
        ns = o.null_splits if rung in null_rungs else ""
        a = [*O, "--rung", rung, variant, "--all-splits", "--null-perm", o.null_perm, "--null-splits", ns,
             "--threshold-source", o.threshold_source, "--loco-mode", o.loco_mode]
        if o.n_estimators is not None:
            a += ["--n-estimators", o.n_estimators]
        if o.train_on_at_floor is not None:
            a += ["--train-on-at-floor", o.train_on_at_floor]
        if o.row_unit is not None:
            a += ["--row-unit", o.row_unit]
        P.append(_cmd(move, f"splits {rung}", "detection_metrics", "splits", a, outputs=[f"gates/detection/splits/{rung}"],
                      inputs=common_inputs + [_sel_input(rung), FLOOR, "inputs/head_drop.csv"]))
        a = [*O, "--rung", rung, "--model", o.one_class_model, "--null-perm", (o.null_perm if ("one_class" in ns.split(",")) else 0)]
        if o.train_on_at_floor is not None:
            a += ["--train-on-at-floor", o.train_on_at_floor]
        if o.row_unit is not None:
            a += ["--row-unit", o.row_unit]
        P.append(_cmd(move, f"one-class {rung}", "detection_metrics", "one-class", a, outputs=[f"gates/detection/splits/{rung}"],
                      inputs=common_inputs + [_sel_input(rung), FLOOR]))
        P.append(_cmd(move, f"gx {rung}", "gates_comparison", "gx", [*O, "--rung", rung, "--null-perm", o.null_perm], outputs=[f"csv:gates/gx.csv:rung={rung}"],
                      inputs=["cells.csv", _sel_input(rung), "inputs/cell_order.csv", PRE]))
        for split in ("lowo", "loco", "lofo"):   # ML review 2.3: G-OP on every split
            a = [*O, "--rung", rung, "--split", split]
            if o.gop_cells_rule:
                a += ["--cells-rule", o.gop_cells_rule]
            P.append(_cmd(move, f"gop {rung} {split}", "gates_detection", "gop", a, outputs=[f"csv:gates/detection/gop.csv:rung={rung}"],
                          inputs=common_inputs + [_sel_input(rung), f"gates/detection/splits/{rung}"]))
        a = [*O, "--rung", rung, "--level-quantity", o.level_quantity]
        P.append(_cmd(move, f"glm {rung}", "gates_detection", "glm", a, outputs=[f"csv:gates/detection/glm.csv:rung={rung}"],
                      inputs=common_inputs + [_sel_input(rung), FLOOR, "inputs/head_drop.csv", "inputs/iteration_boundaries.csv"]))
        P.append(_cmd(move, f"gl {rung}", "gates_detection", "gl", [*O, "--rung", rung], outputs=[f"csv:gates/detection/gl.csv:rung={rung}"],
                      inputs=common_inputs + [_sel_input(rung), f"gates/detection/splits/{rung}"]))

    def levels(move: int, rung: str):
        for level in ("level2", "level3"):
            a = [*O, "--rung", rung, "--null-perm", o.null_perm]
            if o.train_on_at_floor is not None:
                a += ["--train-on-at-floor", o.train_on_at_floor]
            if o.row_unit is not None:
                a += ["--row-unit", o.row_unit]
            P.append(_cmd(move, f"{level} {rung}", "detection_levels", level, a, outputs=[f"gates/detection/splits/{rung}"],
                          inputs=common_inputs + [_sel_input(rung), FLOOR]))

    # ---- D7: apf, breadth alone (K3 move 10)
    rung_stage(7, "apf")
    P.append(_cmd(7, "anchor kernels", "gates_detection", "anchor", [*O, "--part", "kernels", "--null-perm", o.null_perm], outputs=["csv:gates/detection/ganchor.csv:part=kernels"],
                  inputs=common_inputs + [SEL, "gates/gx.csv"]))
    a = [*O, "--consequence", o.order_consequence, "--null-perm", o.null_perm]
    if o.order_scope:
        a += ["--scope", o.order_scope]
    P.append(_cmd(7, "order test", "gates_detection", "order", a, outputs=["gates/detection/order.csv"], inputs=common_inputs + [SEL, FLOOR]))
    P.append(_cmd(7, "leak probe", "gates_detection", "leak-probe", [*O, "--null-perm", o.null_perm], outputs=["gates/detection/leak_probe.csv"],
                  inputs=common_inputs + [FLOOR]))
    P.append(_cmd(7, "tables table9_pitfalls", "tables_detection", None, [*O, "--only", "table9_pitfalls"], outputs=["report/detection/tables/table9_pitfalls.csv"],
                  inputs=common_inputs + [SEL]))
    # ---- D8: the fused plane (K3 move 11)
    P.append(_cmd(8, "gj", "gates_readings", "gj", [*O], outputs=["gates/gj.csv", "gates/gj.json"], inputs=["cells.csv", "inputs/head_drop.csv", PRE]))
    fa = [*O, "--only", "fig4_fused_plane_tiers", "--identity", o.identity]
    if o.plane_per_letter:
        fa += ["--plane-per-letter"]
    P.append(_cmd(8, "figures fig4_fused_plane_tiers", "figures_detection", None, fa,
                  outputs=["report/detection/figures/fig4_fused_plane_tiers.pdf|report/detection/figures/SKIPPED.txt"], inputs=common_inputs + ["gates/gj.json"]))
    # ---- D9, D10, D11: persist, content, wapf (K3 moves 12, 13, 14)
    for rung in ("persist", "content", "wapf"):
        rung_stage(MOVE_OF_RUNG[rung], rung)
        if rung == "persist":
            levels(9, "persist")
    # ---- D12: combined, the matched row, the levels of the other rungs, the per-rung gates, the detection tables (K3 move 16)
    rung_stage(12, "combined")
    a = [*O, "--rung", "combined", "--norm", "--split", "lowo", "--reduce-to-strongest", "--null-perm", o.null_perm,
         "--null-splits", (o.null_splits if "combined" in null_rungs else ""), "--threshold-source", o.threshold_source]
    P.append(_cmd(12, "splits combined (matched)", "detection_metrics", "splits", a, outputs=["gates/detection/splits/combined"],
                  inputs=common_inputs + [_sel_input("combined"), FLOOR] + [f"gates/detection/splits/{r}" for r in ("apf", "wapf", "persist", "content")]))
    for rung in ("apf", "wapf", "content", "combined"):
        levels(12, rung)
    for rung in RUNGS:
        for g in ("gsig", "gfp", "g1c", "gcal"):
            P.append(_cmd(12, f"{g} {rung}", "gates_detection", g, [*O, "--rung", rung], outputs=[f"csv:gates/detection/{g}.csv:rung={rung}"],
                          inputs=common_inputs + [_sel_input(rung), f"gates/detection/splits/{rung}"]))
    P.append(_cmd(12, "gm", "gates_detection", "gm", [*O], outputs=["gates/detection/gm.csv"], inputs=common_inputs + [SEL] + [f"gates/detection/splits/{r}" for r in RUNGS]))
    P.append(_cmd(12, "gdim", "gates_detection", "gdim", [*O], outputs=["gates/detection/gdim.csv"], inputs=common_inputs + [SEL] + [f"gates/detection/splits/{r}" for r in RUNGS]))
    P.append(_cmd(12, "tables table7,table8,table11,levels", "tables_detection", None,
                  [*O, "--only", "table7_detection,table8_member_recall,table11_splits,table_level2,table_level3", "--table10-rung", o.table10_rung],
                  outputs=["report/detection/tables/table7_detection.csv"], inputs=common_inputs + [SEL, FLOOR]))
    # ---- D13: the time-to-hear ladder (K3 move 17)
    for rung in RUNGS:
        a = [*O, "--rung", rung, "--null-perm", o.ladder_null_perm, "--threshold-source", o.threshold_source]
        P.append(_cmd(13, f"ladder {rung}", "detection_metrics", "ladder", a, outputs=[f"csv:gates/detection/ladder.csv:rung={rung}"],
                      inputs=common_inputs + [_sel_input(rung), FLOOR, "inputs/head_drop.csv", "inputs/iteration_boundaries.csv"]))
    P.append(_cmd(13, "tables table_ladder", "tables_detection", None, [*O, "--only", "table_ladder"], outputs=["report/detection/tables/table_ladder.csv"],
                  inputs=common_inputs + [SEL, "gates/detection/ladder.csv"]))
    P.append(_cmd(13, "figures fig6_ladder", "figures_detection", None, [*O, "--only", "fig6_ladder"],
                  outputs=["report/detection/figures/fig6_ladder.pdf|report/detection/figures/SKIPPED.txt"], inputs=common_inputs + ["gates/detection/ladder.csv"]))
    # ---- D14: the miss table, the alias falsifier, G-V, the report (K3 moves 18, 19, 20, 21)
    for rung in RUNGS:
        P.append(_cmd(14, f"miss-table {rung}", "gates_detection", "miss-table", [*O, "--rung", rung], outputs=[f"csv:gates/detection/miss_table.csv:rung={rung}"],
                      inputs=common_inputs + [_sel_input(rung), FLOOR, f"gates/detection/splits/{rung}", "gates/gj.json"]))
        P.append(_cmd(14, f"alias {rung}", "gates_detection", "alias", [*O, "--rung", rung], outputs=[f"csv:gates/detection/alias.csv:rung={rung}"],
                      inputs=common_inputs + [_sel_input(rung), f"gates/detection/splits/{rung}", "inputs/iteration_counts.csv"]))
        P.append(_cmd(14, f"gv {rung}", "gates_detection", "gv", [*O, "--rung", rung], outputs=[f"csv:gates/detection/gv_two_class.csv:rung={rung}"],
                      inputs=common_inputs + [_sel_input(rung)]))
    P.append(_cmd(14, "tables (all)", "tables_detection", None, [*O, "--table10-rung", o.table10_rung],
                  outputs=["report/detection/tables/table4_tiers.csv", "report/detection/tables/table5_gates.csv", "report/detection/tables/table10_misses.csv"],
                  inputs=common_inputs + [SEL, FLOOR]))
    fa = [*O, "--table10-rung", o.table10_rung, "--identity", o.identity]
    if o.plane_per_letter:
        fa += ["--plane-per-letter"]
    P.append(_cmd(14, "figures (all)", "figures_detection", None, fa,
                  outputs=["report/detection/figures/fig5_roc_lowo.pdf|report/detection/figures/SKIPPED.txt"], inputs=common_inputs + [FLOOR]))
    sk = [*O, "--documentclass", o.documentclass]
    if o.standalone_tex:
        sk += ["--standalone", o.standalone_tex]
    P.append(_cmd(14, "latex skeleton p3", "latex_skeleton_p3", None, sk, outputs=["report/detection/paper3_skeleton.tex"], inputs=[]))
    P.append(_cmd(14, "tables manifest", "tables_detection", None, [*O, "--only", "manifest"], outputs=["report/detection/manifest.json"], inputs=common_inputs))
    # ---- D15: the tripwire check (K3 move 23)
    P.append(_cmd(15, "tripwire check", "driver", "tripwire-check", [], internal=True, outputs=["gates/detection/tripwire_check.json"]))
    # seed offsets and n-jobs on the commands that take them
    for c in P:
        key = (c["module"], c["sub"])
        if key in RANDOM_COMMANDS and int(o.seed_offset or 0) != 0:
            c["args"] += ["--seed-offset", str(o.seed_offset)]
        if key in NJOBS_COMMANDS and int(o.n_jobs) != 1:
            c["args"] += ["--n-jobs", str(o.n_jobs)]
    return P


# ----------------------------------------------------------------------------------------------
# internal steps
# ----------------------------------------------------------------------------------------------
def features_at_selection(out: Path, args: list[str]) -> tuple[str, dict]:
    """D4's internal step: the feature files at the rung's inherited grid point (raw and norm; norm
    only for combined) through `series.build_features`, reading `gates/selection.json` at run
    time and recording `grid_source` (SPEC_DETECTION 6.2, D4; 3.1.1). Written to
    `gates/detection/features_at_selection.json` under the rung key."""
    from plan11_encoding_ladder import series  # the epoch-1 module, imported, never changed
    out = Path(out)
    rung = args[0]
    p = out / "gates" / "detection" / "features_at_selection.json"
    j = read_json(p) if p.exists() else result_json("detection.features_at_selection", {}, "SPEC_DETECTION 3.1.1, 6.2 (D4)", {"rungs": {}})
    sel_p = out / "gates" / "selection.json"
    if not sel_p.exists():
        verdict = not_run(f"no selection for {rung} (run classes inherit-selection)")
        j["rungs"][rung] = {"verdict": verdict, "checked_at": now_iso()}
        j[rung] = j["rungs"][rung]
        write_json(p, j)
        return verdict, {}
    sel = read_json(sel_p)
    entry = sel.get(rung) or (sel.get("selection") or {}).get(rung)
    grid_source = (sel.get("params") or {}).get("grid_source") or "selection.json"
    if not entry or not entry.get("grid_id"):
        verdict = not_run(f"no selection for {rung} (run classes inherit-selection)")
        j["rungs"][rung] = {"verdict": verdict, "grid_source": grid_source, "checked_at": now_iso()}
        j[rung] = j["rungs"][rung]
        write_json(p, j)
        return verdict, {"grid_source": grid_source}
    gid = str(entry["grid_id"])
    W, H = series.parse_grid_id(gid) if gid != series.WHOLE_GRID_ID else (None, None)
    head_drop = series.load_head_drop(out / "inputs" / "head_drop.csv")
    written = []
    try:
        for normalized in ((True,) if rung == "combined" else (False, True)):
            written.append(str(series.build_features(out, None, rung, W, H, normalized, head_drop)))
        verdict = "done"
    except FileNotFoundError as e:
        verdict = not_run(f"{e}")
    except Exception as e:
        verdict = refused(f"features {rung} at {gid}: {type(e).__name__}: {e}")
    detail = {"grid_id": gid, "grid_source": grid_source, "written": written, "head_drop": head_drop,
              "head_drop_sha256": sha256_file(out / "inputs" / "head_drop.csv") if (out / "inputs" / "head_drop.csv").exists() else "absent"}
    j["rungs"][rung] = {"verdict": verdict, **detail, "checked_at": now_iso()}
    j[rung] = j["rungs"][rung]
    write_json(p, j)
    return verdict, detail


TRIPWIRE_COLUMNS = ("G-F (i)", "G-ANCHOR (ii)", "early-late idle")


def tripwire_check(out: Path, args: list[str]) -> tuple[str, dict]:
    """D15 (K3 move 23; CR3 2.12; SPEC_DETECTION 6.1): every Table 7 and Table 11 row carries a G-F
    (i) verdict and the drift-clause verdicts (G-ANCHOR (ii), early-against-late idle) of its rung;
    a row without them is listed and the verdict is `refused: <n> rows without a tripwire verdict`.
    Written to `gates/detection/tripwire_check.json`."""
    out = Path(out)
    detail = {"tables": {}, "rows_without": []}
    n_rows = 0
    missing_tables = []
    for name in ("table7_detection", "table11_splits"):
        p = out / "report" / "detection" / "tables" / f"{name}.csv"
        if not p.exists():
            missing_tables.append(str(p.relative_to(out)))
            continue
        rows = read_csv(p)
        n_rows += len(rows)
        per_col = {}
        for r in rows:
            for c in TRIPWIRE_COLUMNS:
                v = str(r.get(c, "")).strip()
                per_col.setdefault(c, set()).add(v)
                if not v or v == "--":
                    detail["rows_without"].append(f"{name}/{r.get('rung')}/{c}")
        detail["tables"][name] = {"n_rows": len(rows), "verdicts": {c: sorted(v) for c, v in per_col.items()}}
    if missing_tables:
        verdict = not_run(f"{', '.join(missing_tables)} missing")
    elif n_rows == 0:
        verdict = refused("Table 7 and Table 11 have no rows")
    elif detail["rows_without"]:
        verdict = refused(f"{len(detail['rows_without'])} rows without a tripwire verdict")
    else:
        verdict = "pass"
    write_json(out / "gates" / "detection" / "tripwire_check.json",
               result_json("detection.tripwire_check", {"columns": list(TRIPWIRE_COLUMNS)}, "K3 move 23; CR3 2.12; SPEC_DETECTION 6.1 (D15)",
                           {"verdict": verdict, **detail}))
    return verdict, detail


INTERNAL_STEPS = {"features-at-selection": features_at_selection, "tripwire-check": tripwire_check}


# ----------------------------------------------------------------------------------------------
# the class file at D0 (al-Farabi M10)
# ----------------------------------------------------------------------------------------------
def place_classes(out: Path, classes: str | None, *, force: bool = False) -> tuple[str, dict]:
    """Copy `--classes` to `<out>/inputs/classes.csv` when absent there; a differing file already
    present is `refused: inputs/classes.csv exists; edit it or pass --force` (SPEC_DETECTION 6.1;
    al-Farabi M10 records the refusal as the D0 command's status in the ledger)."""
    out = Path(out)
    dst = out / "inputs" / "classes.csv"
    if not classes:
        return ("done" if dst.exists() else refused("no --classes given and inputs/classes.csv absent")), {"dst": str(dst)}
    src = Path(classes)
    if not src.exists():
        return not_run(f"{src} missing"), {"src": str(src)}
    if dst.exists():
        if sha256_file(src) == sha256_file(dst):
            return "done", {"status": "identical file already present", "dst": str(dst)}
        if not force:
            return refused("inputs/classes.csv exists; edit it or pass --force"), {"dst": str(dst), "src": str(src)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return "done", {"status": "copied", "src": str(src), "dst": str(dst), "sha256": sha256_file(dst)}


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def print_plan(plan: list[dict], moves: str) -> None:
    """The command list of the selected moves (run_moves.print_plan bounded by MAX_MOVE = 15)."""
    sel = set(run_moves.parse_moves(moves, MAX_MOVE))
    for c in plan:
        if c["move"] in sel:
            argv = ("(internal) " + c["sub"] + " " + " ".join(c["args"])) if c["internal"] else \
                f"python3 -m {PACKAGE_NAME}.{c['module']} " + (c["sub"] + " " if c["sub"] else "") + " ".join(c["args"])
            print(f"[move {c['move']:>2}] {c['name']:<44} {argv}")


def _add_run_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="the retention root you pass (required when move 0 runs)")
    ap.add_argument("--classes", default=None, help="the author's mapping file; copied to <out>/inputs/classes.csv at D0")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--selection-from", default=None, help="the encoding run's gates/selection.json (the inherited (W, H) per rung)")
    g.add_argument("--grid-default", default=None, help=f"a default grid id when no selection is inherited (e.g. {GRID_DEFAULT})")
    ap.add_argument("--moves", default=f"0-{MAX_MOVE}")
    ap.add_argument("--assume-failed-zero", action="store_true")
    ap.add_argument("--assume-reason", default=None)
    ap.add_argument("--c1-activity-min", default=None)
    ap.add_argument("--c1-rule", default="report", choices=["report", "exclude"], help="C1 fail reported for every class (al-Kindi review 5) or excluded")
    ap.add_argument("--campaign-label", default=None, help="the campaign token of every sandbox and external cell_id (default stage1, al-Farabi M1)")
    ap.add_argument("--kernel-family-rule", default="tier", choices=["tier", "archetype"])
    ap.add_argument("--relaunched-grouping", default="parent", choices=["parent", "own"])
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--null-perm", type=int, default=500)
    ap.add_argument("--null-splits", default=NULL_SPLITS_DEFAULT, help="splits that carry the workload-level null (lowo, loco, one_class)")
    ap.add_argument("--null-rungs", default=",".join(RUNGS))
    ap.add_argument("--ladder-null-perm", type=int, default=0)
    ap.add_argument("--loco-mode", default="cell", choices=["cell", "rep_index"])
    ap.add_argument("--one-class-model", default="isolation_forest", choices=["isolation_forest", "gmm", "ocsvm"])
    ap.add_argument("--threshold-source", default=THRESHOLD_SOURCE_DEFAULT, choices=["inner_lowo", "inner_group_kfold", "oob"],
                    help="the in-fold threshold rule (al-Kindi review 1: oob is optimistic under window rows)")
    ap.add_argument("--row-unit", default=None, choices=[None, "cell", "window"], help="passed through when given (ML review 2.1; the module constant otherwise)")
    ap.add_argument("--train-on-at-floor", default=None, choices=[None, "true", "false"], help="passed through when given (ML review 2.5; al-Farabi M2)")
    ap.add_argument("--n-estimators", default=None)
    ap.add_argument("--level-quantity", default="median_K", choices=["median_K", "per_iteration_K_sum"])
    ap.add_argument("--order-consequence", default="size", choices=["size", "void"])
    ap.add_argument("--order-scope", default=None, choices=[None, "within_class", "campaign"], help="passed through when given (al-Farabi M12)")
    ap.add_argument("--gop-cells-rule", default=None, choices=[None, "threshold_setters", "realized_fps"])
    ap.add_argument("--table10-rung", default="combined", choices=list(RUNGS))
    ap.add_argument("--identity", default="excess", choices=["excess", "raw"], help="the fused plane's identity coordinate (al-Kindi review 7)")
    ap.add_argument("--plane-per-letter", action="store_true")
    ap.add_argument("--documentclass", default="llncs", choices=["llncs", "article"])
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="record every command in the ledger without running it")
    ap.add_argument("--skip-missing-modules", action="store_true", help="record `not run: module absent` and continue when a module is not on disk (smoke tests only)")
    ap.add_argument("--only-modules", default=None, help="comma-separated module names to run; the others are recorded as not run")
    ap.add_argument("--persist-side", default=None, choices=[None, "t", "t+1"])
    ap.add_argument("--failed-counts", default=None)
    ap.add_argument("--standalone-tex", default=None, help="also write the LaTeX skeleton to this path (apf_paper/p3_skeleton.tex)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan11 detection driver: al-Kindi's moves D0 to D15 in order (builder B)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); _add_run_args(r)
    s = sub.add_parser("status"); s.add_argument("--out", required=True)
    pl = sub.add_parser("plan"); _add_run_args(pl)
    o = ap.parse_args(argv)
    if o.cmd == "status":
        return print_status(Path(o.out), LEDGER)
    try:
        moves = run_moves.parse_moves(o.moves, MAX_MOVE)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    if o.assume_failed_zero and not o.assume_reason:
        print("--assume-failed-zero requires --assume-reason TEXT (SPEC 3.3.2)", file=sys.stderr)
        return 2
    if 0 in moves and not o.root and o.cmd == "run":
        print("missing input: --root is required when move 0 (extract index) runs", file=sys.stderr)
        return 2
    if 0 in moves and not (o.selection_from or o.grid_default):
        print("missing input: one of --selection-from PATH or --grid-default GRID_ID is required when move 0 runs (SPEC_DETECTION 2.7)", file=sys.stderr)
        return 2
    plan = build_plan(o)
    if o.cmd == "plan":
        print_plan(plan, o.moves)
        return 0
    out = Path(o.out)
    if 0 in moves:
        out.mkdir(parents=True, exist_ok=True)
        verdict, detail = place_classes(out, o.classes, force=o.force)
        ledger = load_ledger(out, LEDGER)
        ledger["commands"].append({"key": "0:classes copy", "move": 0, "name": "classes copy", "module": "driver", "sub": "classes-copy",
                                   "started_at": now_iso(), "finished_at": now_iso(), "status": verdict if verdict != "done" else "done",
                                   "verdict": verdict, "detail": detail, "exit_code": 0 if verdict == "done" else 2, "inputs_sha256": {}})
        save_ledger(out, ledger, LEDGER)
        print(f"[move  0] {'classes copy':<44} {verdict}")
        if verdict != "done":
            print(verdict, file=sys.stderr)
            return 2
    if 1 in moves and 0 not in moves and not (out / "cells.csv").exists():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr)
        return 2
    try:
        return run_plan(o, plan, max_move=MAX_MOVE, ledger_name=LEDGER, internal_steps=INTERNAL_STEPS)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
