#!/usr/bin/env python3
"""tables_detection.py -- the tables of the detection paper (SPEC_DETECTION section 5): Table 4
(tiers with counts), 5 (the gates in one row each), 6 (validity), 7 (detection under LOWO),
8 (per-member recall in eighths), 9 (the three pitfalls as sizes), 10 (the miss table and the
false-positive table), 11 (the four splits side by side), the level-2 and level-3 tables, the
ladder table, the G-V two-class table, the per-cell appendix table and the manifest, each as
CSV, Markdown and LaTeX through `_report_common.write_table`.

Builder B (report), 2026-09-17. Nothing here computes a verdict: every verdict string is copied
from builder A's result files under `<out>/gates/detection/` (and the carried-over plan11 gates
under `<out>/gates/`) by their documented column names (SPEC_DETECTION section 1, the
interface rule). A verdict cell prints the verdict string; an undefined number prints `--`; a
missing input file prints `not run: <file> missing`; a `NULL_INSIDE`, `NULL_NOT_ESTIMABLE`,
`GL_LEVEL_ONLY`, `ORDER_VOID`, `GF_VOID` or `GC_DISCONNECTED` verdict is printed in the row's
verdict column and, for `ORDER_VOID` (consequence void), `GF_VOID` and `GC_DISCONNECTED`, in
every score cell of the affected rung as well (the numbers stay in `scores.json`). A refusal is
never a number and never a blank. No prose anywhere; the note lines are labels (SPEC_DETECTION
section 5, preamble).

Corrections from the three SPEC_DETECTION reviews that this module implements (each marked
"must change before build"):
  - al-Kindi 3: the level-2 table prints `null p95, rank, verdict` per sub-family row from
    `confusion.csv`; `macro_recall` is never printed.
  - al-Kindi 4: the level tables carry `n at floor`.
  - al-Kindi 5: Table 4 carries `n C1 fail (reported)` per class.
  - al-Kindi 10, 11: Table 9 prints one drift row per (class, drift unit) and one order row
    per (rung, class, half rule).
  - ML 2.3: Table 5 rolls G-OP up over the `lowo`, `loco` and `lofo` rows of `gop.csv`.
  - ML 2.4: Table 7 prints the full-feature forest as the headline (`score_source = full`),
    the quarantined feature in the B1-G3 column, the re-run's TPR beside it, and one
    `l1 (best single feature)` row per rung from `scores.json["l1"]`.
  - ML 2.6: Table 9 carries the `cadence` and `active fraction` rows from `leak_probe.csv`.
  - ML 2.7 / al-Kindi 2: Table 11's note reads the one-class null verdict from the one-class
    `scores.json["null"]`.
  - ML 2.8: Table 8 prints, under each member, the `L0_ratio` of `gv_two_class_members.csv`
    beside the LOCO row and the `reps near identical` note.
  - al-Farabi M4: `n without score` beside `n at floor` in Tables 7 and 8.
  - al-Farabi M9: Table 4 carries an `unassigned` row.
  - al-Farabi M11: the per-cell table never prints `path`, `label`, `test_label`,
    `traj_file` or `order_index`.
  - The tripwire check (SPEC_DETECTION 6.1, move D15) reads a G-F (i) verdict and the
    drift-clause verdicts (G-ANCHOR (ii), early-against-late idle) off every Table 7 and
    Table 11 row, so both tables carry those three columns.

CLI (SPEC 7.1): tables_detection.py --out O [--only NAME,...] [--table10-rung combined]
Names: table4_tiers, table5_gates, table6_validity, table7_detection, table8_member_recall,
table9_pitfalls, table10_misses, table10_false_positives, table11_splits, table_level2,
table_level3, table_ladder, table_gv_two_class, table_cells_detection, manifest.
Exit 0 on success (a written refusal is a success), 2 when `<out>/cells.csv` is missing, 1 on
an internal error.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse
import math
import statistics
import traceback
from collections import Counter, OrderedDict

from plan11_encoding_ladder._report_common import (  # noqa: E402
    AXIS_OF_RUNG, FAIL, GC_DISCONNECTED, GDIM_FULL, GF_VOID, GK0_ABOVE_FLOOR, GN_HEADLINE,
    GX_POOLING_STANDS, PACKAGE_VERSION, PASS, RUNGS, SCHEMA_SOURCE, VERDICTS_SOURCE, fmt_num,
    gc_rung_verdict, gf_part1_verdict, grid_label, inputs_sha256, is_refusal, load_cells, not_applicable,
    not_run, now_iso, read_csv, read_json, refused, result_json, sha256_file, to_float, write_json,
    write_table,
)

try:  # builder A's verdicts.py may carry the detection vocabulary when it is present
    from plan11_encoding_ladder import verdicts as V  # type: ignore
except Exception:  # pragma: no cover
    V = None


def _vattr(name, default):
    return getattr(V, name, default) if V is not None else default


# ----------------------------------------------------------------------------------------------
# the detection vocabulary (SPEC_DETECTION 3.0; builder B falls back to the literals)
# ----------------------------------------------------------------------------------------------
NULL_NOT_ESTIMABLE = _vattr("NULL_NOT_ESTIMABLE", "null_not_estimable")
NULL_INSIDE = _vattr("NULL_INSIDE", "inside the workload-level null")
GOP_SUPPORTED = _vattr("GOP_SUPPORTED", "operating point supported")
GOP_SET_BY_FEW = _vattr("GOP_SET_BY_FEW", "set by one workload, family named")
GLM_SURVIVES = _vattr("GLM_SURVIVES", "detection survives level matching")
GLM_LEVEL_ONLY = _vattr("GLM_LEVEL_ONLY", "detected by level in this lead")
GLM_LABEL_MEDIAN_K = _vattr("GLM_LABEL_MEDIAN_K", "median K, stage 1")
GANCHOR_AUDIBLE = _vattr("GANCHOR_AUDIBLE", "campaign audible")
GANCHOR_NOT_AUDIBLE = _vattr("GANCHOR_NOT_AUDIBLE", "campaign not audible above the null")
ORDER_AUDIBLE = _vattr("ORDER_AUDIBLE", "position audible")
ORDER_NOT_AUDIBLE = _vattr("ORDER_NOT_AUDIBLE", "position not audible above the null")
ORDER_VOID = _vattr("ORDER_VOID", "void: position audible inside the interleaved campaign")
DRIFT_SLOPE = _vattr("DRIFT_SLOPE", "drift: slope above the shuffle null")
DRIFT_NONE = _vattr("DRIFT_NONE", "no drift above the shuffle null")
GSIG_REPORTED = _vattr("GSIG_REPORTED", "gap reported")
GSIG_IDENTITY = _vattr("GSIG_IDENTITY", "recognizes workload identity, not family behaviour")
GFP_ATTRIBUTED = _vattr("GFP_ATTRIBUTED", "attributed")
GFP_INSEPARABLE = _vattr("GFP_INSEPARABLE", "inseparable from sandbox under this rung")
G1C_PRIMARY = _vattr("G1C_PRIMARY", "primary")
G1C_SECONDARY = _vattr("G1C_SECONDARY", "secondary, not citable")
G1C_SEARCH = _vattr("G1C_SEARCH", "refused: a second one-class model without a declared primary reads as a search")
HARNESS_STAGE2_ABSENT = _vattr("HARNESS_STAGE2_ABSENT", "not run: stage 2 absent")
GCAL_AGREE = _vattr("GCAL_AGREE", "per-fold and pooled agree within the null spread")
GCAL_PERFOLD = _vattr("GCAL_PERFOLD", "per-fold reported; pooled curve threshold set post hoc on the held-out scores")
POST_HOC_LABEL = _vattr("POST_HOC_LABEL", "threshold set post hoc on the held-out scores")
GK0_AT_FLOOR = _vattr("GK0_AT_FLOOR", "at floor")
GK0_AT_HARNESS_FLOOR = _vattr("GK0_AT_HARNESS_FLOOR", "at harness floor")
GK0_MIXED = _vattr("GK0_MIXED", "cells in more than one verdict")
GN_ONE_TRAIN_WORKLOAD = _vattr("GN_ONE_TRAIN_WORKLOAD", "one training workload per fold")
GN_NO_SUPERVISED = _vattr("GN_NO_SUPERVISED", "no supervised headline")
GN_SINGLE_WORKLOAD = _vattr("GN_SINGLE_WORKLOAD", "single workload")
L2_ONE_TRAIN = _vattr("L2_ONE_TRAIN", "one training member per fold")
SIGNATURE_CEILING = _vattr("SIGNATURE_CEILING", "signature ceiling")
LADDER_FROM_PAIR1_ONLY = _vattr("LADDER_FROM_PAIR1_ONLY", "from pair 1 only")
AT_FLOOR_NOT_A_MISS = _vattr("AT_FLOOR_NOT_A_MISS", "at floor, not a miss")
CROSS_CAMPAIGN_STAGE1 = _vattr("CROSS_CAMPAIGN_STAGE1", "not applicable: stage 1 (the 01c cells are the benign class)")
RUNG2P_NOT_BUILT = _vattr("RUNG2P_NOT_BUILT", "not run: content family columns not in the plan11 extract (rung 2' needs an extract extension)")
COMPARATOR_ELSEWHERE = _vattr("COMPARATOR_ELSEWHERE", "not run: comparator row from another session (RQ5)")
YIELD_NOT_RECORDED = _vattr("YIELD_NOT_RECORDED", "not run: no vmstat record (stage 1)")
LEAK_AUDIBLE = _vattr("LEAK_AUDIBLE", "leak audible")
LEAK_NOT_AUDIBLE = _vattr("LEAK_NOT_AUDIBLE", "leak not audible above the null")
F9_SENTENCE = "separable when heard, not flagged when not"          # a label of Table 11 (K3 F9), not a verdict
ONE_CLASS_NORM_ONLY = "not applicable: the one-class run is on the normalized features (SPEC_DETECTION 3.3.6)"
REPS_IDENTICAL_NOTE = "reps near identical: LOCO reads as within-trace"  # ML review 2.8
REPS_IDENTICAL_RATIO = 0.1                                            # ML review 2.8 (proposed default; section 7)

DET_CLASSES = ("benign_kernel", "benign_relaunched", "benign_breadth", "idle", "harness_idle", "sandbox", "external")
STAGE2_CLASSES = ("benign_relaunched", "benign_breadth", "harness_idle")
CLASS_LETTER = {"sandbox": "S", "benign_kernel": "B", "benign_breadth": "B", "idle": "I", "harness_idle": "H",
                "benign_relaunched": "R", "external": "X", "unassigned": "-"}
DET_SPLITS = ("lowo", "loco", "lofo", "one_class")
# the rung display names of SPEC_DETECTION section 5 with their (rung, variant, directory stem)
RUNG_ROWS = OrderedDict([
    ("apf raw", ("apf", "raw", "lowo")), ("apf", ("apf", "norm", "lowo")), ("wapf", ("wapf", "norm", "lowo")),
    ("persist", ("persist", "norm", "lowo")), ("content", ("content", "norm", "lowo")),
    ("combined", ("combined", "norm", "lowo")), ("combined (matched)", ("combined", "norm", "lowo_matched")),
    ("content channel 2'", None), ("comparator", ("comparator", "norm", "lowo")),
])
PLACEHOLDER_ROWS = {"content channel 2'": RUNG2P_NOT_BUILT, "comparator": COMPARATOR_ELSEWHERE}
TABLE_NAMES = ("table4_tiers", "table5_gates", "table6_validity", "table7_detection", "table8_member_recall",
               "table9_pitfalls", "table10_misses", "table10_false_positives", "table11_splits", "table_level2",
               "table_level3", "table_ladder", "table_gv_two_class", "table_cells_detection", "manifest")
CITATIONS = {
    "table4_tiers": "P3 Sec. 2 (section 4, the six tiers with counts); N3 Sec. 2; SPEC_DETECTION 5.1; al-Kindi review 5; al-Farabi review M9",
    "table5_gates": "P3 Sec. 2 (section 5, the gates in one row each); P3 Sec. 3; CR3 2; SPEC_DETECTION 5.2.1; ML review 2.3",
    "table6_validity": "N3 Sec. 1 (the validity block); P3 D6; CR3 2.12; SPEC_DETECTION 5.2.2",
    "table7_detection": "P3 D5; N3 Sec. 1 RQ1 (Table 7); CR3 1.5, 2.1, 2.2, 2.3, 2.13, 2.18; SPEC_DETECTION 5.2.3; ML review 2.4; al-Farabi review M4",
    "table8_member_recall": "P3 D5 (per-member recall in eighths, never a mean); ML 2.4; N3 Sec. 1 RQ1 (Table 8); SPEC_DETECTION 5.2.4; ML review 2.8; al-Farabi review M4",
    "table9_pitfalls": "N3 Sec. 1 RQ3 (Table 9); CR3 2.4, 2.14, 2.17, 2.20, 2.21; K3 F1, F4, F5; SPEC_DETECTION 5.2.5; al-Kindi review 10, 11; ML review 2.6",
    "table10_misses": "K3 move 18; N3 Sec. 1 RQ2 (Table 10); SPEC_DETECTION 5.2.6; al-Kindi review 8",
    "table10_false_positives": "K3 move 18; N3 Sec. 1 RQ2 (Table 10); SPEC_DETECTION 5.2.6; al-Kindi review 8",
    "table11_splits": "N3 Sec. 1 RQ4 (Table 11); CR3 1.5, 2.15, 2.16, 2.19; K3 F8, F9; SPEC_DETECTION 5.2.7; ML review 2.7",
    "table_level2": "P3 0a (level 2, per sub-family, never averaged); CR3 2.5; SPEC_DETECTION 5.2.8; al-Kindi review 3, 4",
    "table_level3": "P3 0a (level 3, the signature ceiling); SPEC_DETECTION 5.2.9; al-Kindi review 4",
    "table_ladder": "K3 move 17; CR3 2.29; N3 Sec. 1 RQ1 (Figure 6); SPEC_DETECTION 5.2.10; al-Farabi review M6",
    "table_gv_two_class": "CR3 2.11; K3 move 20; SPEC_DETECTION 5.2.11",
    "table_cells_detection": "P3 D8 (per-cell metadata with the order index); SPEC_DETECTION 5.2.12; al-Farabi review M11",
    "manifest": "SPEC_DETECTION 5.2.13 (report/detection/manifest.json)",
}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _tables_dir(out: Path) -> Path:
    return Path(out) / "report" / "detection" / "tables"


def _det(out: Path) -> Path:
    return Path(out) / "gates" / "detection"


def _rel(out: Path, p: Path) -> str:
    try:
        return str(Path(p).relative_to(Path(out)))
    except ValueError:
        return str(p)


def _missing(out: Path, p: Path) -> str:
    return not_run(f"{_rel(out, p)} missing")


def _cell(x, digits: int = 3) -> str:
    """A table cell: a verdict string as itself, a number through fmt_num, an undefined number
    (None, '', NaN) as `--`; never a blank (SPEC_DETECTION section 5 preamble)."""
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return "--"
        f = to_float(s)
        return s if f is None else fmt_num(f, digits)
    if isinstance(x, (list, tuple)):
        return "; ".join(_cell(v, digits) for v in x) if x else "--"
    return fmt_num(x, digits)


def _num(x):
    """A float or None from a scores.json field that may hold a number or a refusal string."""
    if isinstance(x, str):
        return None
    return to_float(x)


def _str_or_num(x, digits: int = 3) -> str:
    """The refusal string itself, else the number, else `--`."""
    if isinstance(x, str) and to_float(x) is None:
        return x if x.strip() else "--"
    return _cell(x, digits)


def _params_base(out: Path, extra: dict | None = None) -> dict:
    p = {"out": str(out), "package_version": PACKAGE_VERSION, "schema_source": SCHEMA_SOURCE,
         "verdicts_source": VERDICTS_SOURCE, "written_at": now_iso(), "grid_source": grid_source(out)}
    if extra:
        p.update(extra)
    return p


def _write_params(out: Path, name: str, params: dict, payload: dict | None = None) -> Path:
    d = result_json(f"detection.tables.{name}", params, CITATIONS[name], payload or {})
    return write_json(_tables_dir(out) / f"{name}.json", d)


def _read_csv_opt(out: Path, rel: str) -> list[dict] | None:
    p = Path(out) / rel
    return read_csv(p) if p.exists() else None


def _read_json_opt(out: Path, rel: str) -> dict | None:
    p = Path(out) / rel
    if not p.exists():
        return None
    try:
        return read_json(p)
    except Exception:
        return None


def grid_source(out: Path) -> str:
    """`params.grid_source` of `gates/selection.json` (SPEC_DETECTION 1.4, 2.7), or the not-run
    string when the file or the key is absent."""
    sel = _read_json_opt(out, "gates/selection.json")
    if sel is None:
        return not_run("gates/selection.json missing (run classes inherit-selection)")
    src = (sel.get("params") or {}).get("grid_source")
    return str(src) if src else "selection.json"


def det_grid(out: Path, rung: str) -> str | None:
    """The rung's grid id from `gates/selection.json` (top-level rung keys or under `selection`);
    None when the rung has no selection (the tables then print `not run: no selection for <rung>`)."""
    sel = _read_json_opt(out, "gates/selection.json")
    if not sel:
        return None
    entry = sel.get(rung) or (sel.get("selection") or {}).get(rung)
    if isinstance(entry, dict) and entry.get("grid_id"):
        return str(entry["grid_id"])
    return None


def split_dir(out: Path, rung: str, split_stem: str, variant: str = "norm", gid: str | None = None) -> Path | None:
    """`gates/detection/splits/<rung>/<gid>/<stem>__<variant>/` when its scores.json exists."""
    gid = gid or det_grid(out, rung)
    if gid is None:
        return None
    d = _det(out) / "splits" / rung / gid / f"{split_stem}__{variant}"
    return d if (d / "scores.json").exists() else None


def load_scores(out: Path, rung: str, split_stem: str, variant: str = "norm") -> dict | None:
    d = split_dir(out, rung, split_stem, variant)
    if d is None:
        return None
    try:
        return read_json(d / "scores.json")
    except Exception:
        return None


def scores_path_text(out: Path, rung: str, split_stem: str, variant: str = "norm") -> str:
    gid = det_grid(out, rung)
    if gid is None:
        return not_run(f"no selection for {rung} (run classes inherit-selection)")
    return _missing(out, _det(out) / "splits" / rung / gid / f"{split_stem}__{variant}" / "scores.json")


def null_field(sc: dict | None, statistic: str, key: str):
    """`scores.json["null"][statistic][key]` or None."""
    if not sc or not isinstance(sc.get("null"), dict):
        return None
    st = sc["null"].get(statistic)
    if not isinstance(st, dict):
        return None
    return st.get(key)


def null_verdict_text(sc: dict | None, statistic: str = "tpr05") -> str:
    """The null verdict of a split for one statistic; `not run: null not run` when the split ran
    without a null (SPEC_DETECTION 3.3.4, 6.1)."""
    if sc is None:
        return not_run("scores.json missing")
    if sc.get("status") and str(sc["status"]) != "ok":
        return str(sc["status"])
    if not isinstance(sc.get("null"), dict):
        return not_run("null not run")
    st = sc["null"].get("status")
    if st is not None and str(st) != "ok":
        return str(st)                       # builder A: `not run: null not requested`, `not applicable: no workload to permute`
    v = null_field(sc, statistic, "verdict")
    return str(v) if v is not None else not_run("null not run")


def order_void_rungs(out: Path) -> set:
    """Rungs whose order test carries consequence `void` with ORDER_VOID (K3 F5): every score
    cell of that rung prints the verdict string (SPEC_DETECTION 3.5.7, 5)."""
    rows = _read_csv_opt(out, "gates/detection/order.csv") or []
    return {r.get("rung") for r in rows if str(r.get("verdict", "")) == ORDER_VOID and str(r.get("consequence", "")) == "void"}


def rung_override(out: Path, rung: str) -> str | None:
    """The string that replaces every score cell of a rung: G-C's `refused: disconnected lead`,
    G-F (i)'s void string, or ORDER_VOID under consequence void; None when none applies."""
    gc = gc_rung_verdict(out, rung)
    if gc is not None and gc == GC_DISCONNECTED:
        return refused(GC_DISCONNECTED)
    gf = gf_part1_verdict(out, rung, det_grid(out, rung))
    if gf is not None and gf == GF_VOID:
        return GF_VOID
    if rung in order_void_rungs(out):
        return ORDER_VOID
    return None


def gf_text(out: Path, rung: str) -> str:
    v = gf_part1_verdict(out, rung, det_grid(out, rung))
    return v if v else _missing(out, Path(out) / "gates" / "gf.csv")


def gc_text(out: Path, rung: str) -> str:
    v = gc_rung_verdict(out, rung)
    return v if v else _missing(out, Path(out) / "gates" / "gc.csv")


def anchor_rows(out: Path) -> list[dict] | None:
    return _read_csv_opt(out, "gates/detection/ganchor.csv")


def anchor_text(out: Path, rung: str, part: str) -> str:
    rows = anchor_rows(out)
    if rows is None:
        return _missing(out, _det(out) / "ganchor.csv")
    hits = [r for r in rows if r.get("rung") == rung and r.get("part") == part]
    if not hits:
        return not_run(f"ganchor.csv has no {part} row for {rung}")
    return str(hits[-1].get("verdict") or "--")


def severity(v: str) -> int:
    """Roll-up order (SPEC_DETECTION 5.2.1: a refusal beats a pass): 3 `refused:` or fail, 2 any
    other refusal (not run, not applicable, a named refusal), 1 any other non-pass string, 0 pass."""
    s = str(v or "").strip()
    head = s.split(" (", 1)[0]
    if s == "" or s == "--":
        return 2
    if s.startswith("refused:") or s == FAIL or head == FAIL:
        return 3
    if is_refusal(s):
        return 2
    if head in (PASS, "ok", "complete", "done", GOP_SUPPORTED, GLM_SURVIVES, GANCHOR_NOT_AUDIBLE, ORDER_NOT_AUDIBLE,
                DRIFT_NONE, GSIG_REPORTED, GFP_ATTRIBUTED, G1C_PRIMARY, GCAL_AGREE, GK0_ABOVE_FLOOR, GN_HEADLINE,
                GX_POOLING_STANDS, GDIM_FULL, LEAK_NOT_AUDIBLE, "inseparable at floor", "control", "true", "estimable",
                "does not move with the interval", "attributed"):
        return 0
    return 1


def rollup(verdicts: list[str]) -> tuple[str, str]:
    """(roll-up text, refusal string): the worst verdict across the rows with the count of rows
    carrying it in parentheses as text; the refusal string is that verdict when it is a refusal,
    else `--`."""
    vs = [str(v) if v is not None else "" for v in verdicts]
    if not vs:
        return "--", "--"
    worst = max(vs, key=severity)
    sev = severity(worst)
    n = sum(1 for v in vs if severity(v) == sev)
    text = f"{worst if worst else '--'} ({n} of {len(vs)} rows)"
    return text, (worst if (sev >= 2 and worst) else "--")


def class_counts(out: Path) -> dict:
    rows = _read_csv_opt(out, "gates/detection/cell_classes.csv") or []
    return dict(Counter(r.get("class", "") for r in rows))


def note_line(out: Path, extra: list[str] | None = None) -> str:
    """The `% note:` line every table carries: grid_source, the class counts, the standing labels."""
    cc = class_counts(out)
    parts = [f"note: grid_source = {grid_source(out)}",
             "class counts = " + ("; ".join(f"{k} {v}" for k, v in sorted(cc.items())) if cc else "not run: gates/detection/cell_classes.csv missing")]
    if extra:
        parts += [str(e) for e in extra]
    return "\n".join(parts)


def _gn_sandbox_status(out: Path) -> str:
    rows = _read_csv_opt(out, "gates/detection/gn.csv")
    if rows is None:
        return _missing(out, _det(out) / "gn.csv")
    for r in rows:
        if r.get("row") == "sandbox":
            return str(r.get("status") or "--")
    return not_run("gn.csv has no sandbox row")


def _members(out: Path) -> list[tuple[int, str]]:
    """(member_index, letter) of every sandbox member in the join, by index."""
    rows = _read_csv_opt(out, "gates/detection/cell_classes.csv") or []
    seen = {}
    for r in rows:
        if r.get("class") == "sandbox" and str(r.get("member_index", "")).strip():
            try:
                seen[int(float(r["member_index"]))] = r.get("subfamily_letter") or "-"
            except ValueError:
                continue
    return sorted(seen.items())


def _external_members(out: Path) -> list[int]:
    rows = _read_csv_opt(out, "gates/detection/cell_classes.csv") or []
    seen = set()
    for r in rows:
        if r.get("class") == "external" and str(r.get("member_index", "")).strip():
            try:
                seen.add(int(float(r["member_index"])))
            except ValueError:
                continue
    return sorted(seen)


def _params_val(sc: dict | None, key: str, default="--"):
    if not sc:
        return default
    v = (sc.get("params") or {}).get(key, sc.get(key, None))
    return default if v is None else v


# ==============================================================================================
# Table 4, the tiers with counts (SPEC_DETECTION 5.1)
# ==============================================================================================
TABLE4_COLUMNS = ["tier (class)", "n workloads", "n cells", "n admissible", "n C1 fail (reported)", "n at floor",
                  "n at harness floor", "n unassigned", "family key rule", "letter"]


def table4_tiers(out: Path) -> dict:
    """One row per class present, in CLASSES order, an `unassigned` row (al-Farabi M9) and a
    `total` row; absent stage-2 classes print 0 cells and `not run: stage 2 absent` in the
    admissible column (SPEC_DETECTION 5.1; al-Kindi review 5 adds `n C1 fail (reported)`)."""
    out = Path(out)
    join = _read_csv_opt(out, "gates/detection/cell_classes.csv")
    adm = _read_csv_opt(out, "gates/detection/admissibility.csv")
    gk0 = _read_csv_opt(out, "gates/detection/gk0_cells.csv")
    cj = _read_json_opt(out, "gates/detection/cell_classes.json") or {}
    rule = str((cj.get("params") or {}).get("kernel_family_rule") or "--")
    rows = []
    miss_join = _missing(out, _det(out) / "cell_classes.csv")
    miss_adm = _missing(out, _det(out) / "admissibility.csv")
    miss_gk0 = _missing(out, _det(out) / "gk0_cells.csv")
    adm_by = {r["cell_id"]: r for r in adm} if adm else {}
    gk0_by = {r["cell_id"]: r for r in gk0} if gk0 else {}
    tot = Counter()
    for cls in list(DET_CLASSES) + ["unassigned"]:
        cells = [r for r in (join or []) if r.get("class") == cls]
        if join is None:
            rows.append({"tier (class)": cls, "n workloads": miss_join, "n cells": miss_join, "n admissible": miss_join,
                         "n C1 fail (reported)": miss_join, "n at floor": miss_join, "n at harness floor": miss_join,
                         "n unassigned": miss_join, "family key rule": rule, "letter": CLASS_LETTER.get(cls, "-")})
            continue
        if not cells:
            if cls in STAGE2_CLASSES:
                rows.append({"tier (class)": cls, "n workloads": 0, "n cells": 0, "n admissible": HARNESS_STAGE2_ABSENT,
                             "n C1 fail (reported)": HARNESS_STAGE2_ABSENT, "n at floor": HARNESS_STAGE2_ABSENT,
                             "n at harness floor": HARNESS_STAGE2_ABSENT, "n unassigned": 0, "family key rule": rule,
                             "letter": CLASS_LETTER.get(cls, "-")})
            elif cls == "unassigned":
                rows.append({"tier (class)": cls, "n workloads": 0, "n cells": 0, "n admissible": 0, "n C1 fail (reported)": 0,
                             "n at floor": 0, "n at harness floor": 0, "n unassigned": 0, "family key rule": rule, "letter": "-"})
            continue
        ids = [r["cell_id"] for r in cells]
        n_adm = sum(1 for i in ids if str(adm_by.get(i, {}).get("admissible", "")).lower() == "true") if adm else miss_adm
        n_c1 = (sum(1 for i in ids if str(adm_by.get(i, {}).get("C1", "")) == FAIL and str(adm_by.get(i, {}).get("admissible", "")).lower() == "true")
                if adm else miss_adm)
        n_floor = sum(1 for i in ids if str(gk0_by.get(i, {}).get("verdict", "")) == GK0_AT_FLOOR) if gk0 else miss_gk0
        n_hfloor = sum(1 for i in ids if str(gk0_by.get(i, {}).get("verdict", "")) == GK0_AT_HARNESS_FLOOR) if gk0 else miss_gk0
        n_wk = len({r.get("workload_key") for r in cells if r.get("workload_key")})
        rows.append({"tier (class)": cls, "n workloads": n_wk, "n cells": len(cells), "n admissible": n_adm,
                     "n C1 fail (reported)": n_c1, "n at floor": n_floor, "n at harness floor": n_hfloor,
                     "n unassigned": len(cells) if cls == "unassigned" else 0, "family key rule": rule,
                     "letter": CLASS_LETTER.get(cls, "-")})
        tot["cells"] += len(cells)
        for k, v in (("adm", n_adm), ("c1", n_c1), ("floor", n_floor), ("hfloor", n_hfloor)):
            if isinstance(v, int):
                tot[k] += v
        tot["wk"] += n_wk
        if cls == "unassigned":
            tot["un"] += len(cells)
    if join is not None:
        rows.append({"tier (class)": "total", "n workloads": tot["wk"], "n cells": tot["cells"], "n admissible": tot["adm"],
                     "n C1 fail (reported)": tot["c1"], "n at floor": tot["floor"], "n at harness floor": tot["hfloor"],
                     "n unassigned": tot["un"], "family key rule": rule, "letter": "--"})
    S, B = cj.get("S", "--"), cj.get("B", "--")
    note = note_line(out, [f"S = {S}; B = {B}; n_assignments = {cj.get('n_assignments', '--')}",
                           "n C1 fail (reported): cells admitted through det_c1_rule = report (al-Kindi review 5; K3 F3)"])
    paths = write_table(_tables_dir(out), "table4_tiers", TABLE4_COLUMNS, [{k: _cell(v) for k, v in r.items()} for r in rows],
                        note_comment=note, wide=False)
    _write_params(out, "table4_tiers", _params_base(out, {"n_rows": len(rows), "kernel_family_rule": rule,
                                                           "inputs_sha256": inputs_sha256([_det(out) / "cell_classes.csv", _det(out) / "admissibility.csv",
                                                                                           _det(out) / "gk0_cells.csv"])}))
    return paths


# ==============================================================================================
# Table 5, the gates in one row each (SPEC_DETECTION 5.2.1)
# ==============================================================================================
TABLE5_COLUMNS = ["gate", "what it checks", "threshold", "verdict (roll-up)", "refusal string", "file"]


def _verdicts_of(out: Path, rel: str, col: str = "verdict", where=None) -> list[str] | None:
    rows = _read_csv_opt(out, rel)
    if rows is None:
        return None
    if where:
        rows = [r for r in rows if where(r)]
    return [str(r.get(col, "")) for r in rows]


def _gate_registry(out: Path) -> list[dict]:
    """(gate, what it checks, threshold, file, verdicts) for every gate of the layer and every
    carried-over plan11 gate that ran inside <out>; the strings are labels from the docstrings of
    SPEC_DETECTION 3.5 and CR3 section 2, never prose."""
    D = "gates/detection/"
    G = "gates/"
    pre = _read_csv_opt(out, G + "preconditions.csv")
    reg = [
        ("admissibility", "status ok, C2, C6 pass; C1 fail reported for every class (det_c1_rule)", "det_c1_rule = report", D + "admissibility.csv",
         _verdicts_of(out, D + "admissibility.csv", "admissible")),
        ("G-K0 two-class", "three quantities against the idle and harness envelopes; three verdicts", "envelope percentile 95; quantities K_med, K_q90, frac_above_band",
         D + "gk0_members.csv", _verdicts_of(out, D + "gk0_members.csv")),
        ("G-N", "members above floor >= 3 for the headline; families >= 2 workloads", "3 members; 2 workloads", D + "gn.csv", _verdicts_of(out, D + "gn.csv", "status")),
        ("G-L (i)", "normalized LOWO tpr05 above the workload-level null, else level only", "strict exceedance of the null p95", D + "gl.csv", _verdicts_of(out, D + "gl.csv")),
        ("G-OP", "cells behind the declared rate: >= 5 cells from >= 3 benign workloads, per fold (worst fold)", "5 cells; 3 workloads; lowo, loco, lofo",
         D + "gop.csv", _verdicts_of(out, D + "gop.csv")),
        ("G-LM", "operating point recomputed against level-matched benign within a factor-two band", "band factor 2.0; vanish rule tpr_le_fpr; level = median K, stage 1",
         D + "glm.csv", _verdicts_of(out, D + "glm.csv")),
        ("G-ANCHOR (i)", "campaign predictability on the kernels from level-normalized features (G-X)", "500 label shuffles; read as a size", D + "ganchor.csv",
         _verdicts_of(out, D + "ganchor.csv", where=lambda r: r.get("part") == "kernels")),
        ("G-ANCHOR (ii)", "idle set against idle set, leave-one-cell-out, label = campaign", "500 cell-level shuffles", D + "ganchor.csv",
         _verdicts_of(out, D + "ganchor.csv", where=lambda r: r.get("part") == "idle_sets")),
        ("early-against-late idle", "first half against second half of the idle cells by order_index", "500 cell-level shuffles", D + "ganchor.csv",
         _verdicts_of(out, D + "ganchor.csv", where=lambda r: r.get("part") == "idle_early_late")),
        ("order test", "half label within class under LOWO; within_workload and within_class half rules", "500 permutations; consequence size (stage 1)",
         D + "order.csv", _verdicts_of(out, D + "order.csv")),
        ("drift regression", "per-cell floor statistic on order_index; within workload and blocked", "95th percentile of |slope| under 500 shuffles",
         D + "drift.csv", _verdicts_of(out, D + "drift.csv")),
        ("leak probe", "one-feature model on n_pairs, dt_est_s, frac_above_band, K_med under LOWO", "workload-level null", D + "leak_probe.csv",
         _verdicts_of(out, D + "leak_probe.csv")),
        ("G-SIG", "LOCO minus LOWO; identity when LOCO above the null and LOWO inside it", "the two null verdicts", D + "gsig.csv", _verdicts_of(out, D + "gsig.csv")),
        ("G-FP", "false positives attributed per benign family; inseparable at or above half", "flag fraction 0.5", D + "gfp.csv", _verdicts_of(out, D + "gfp.csv")),
        ("G-1C", "one primary one-class model declared before the data", "primary = isolation_forest", D + "g1c.csv", _verdicts_of(out, D + "g1c.csv")),
        ("harness clause", "three margins per feature; re-launched control flagged at the members' rate", "comparable tol 0.10", D + "harness.csv",
         _verdicts_of(out, D + "harness.csv")),
        ("G-CAL", "per-fold tpr05 against the pooled reading within the null spread", "spread = p95 - p05 of the tpr05 null", D + "gcal.csv", _verdicts_of(out, D + "gcal.csv")),
        ("G-M", "paired per workload; exact one-sided binomial over non-tied units and the seed spread", "alpha 0.05; 5 seeds", D + "gm.csv", _verdicts_of(out, D + "gm.csv")),
        ("G-DIM", "feature count per rung; the matched row reduced to the strongest rung's d", "train_importance", D + "gdim.csv", _verdicts_of(out, D + "gdim.csv", "status")),
        ("B1-G1 two-class", "workload-level label-shuffle null per split", "500 permutations; null_not_estimable below 20 assignments", D + "splits/",
         [null_verdict_text(load_scores(out, r, "lowo")) for r in RUNGS]),
        ("B1-G3 two-class", "no single feature reproduces the forest's flag_05 on all but one held-out workload", "max disagree workloads 1", D + "splits/",
         [("quarantined: " + "; ".join((sc.get("quarantine") or {}).get("quarantined_features") or [])) if (sc and (sc.get("quarantine") or {}).get("quarantined_features")) else ("none" if sc else scores_path_text(out, r, "lowo"))
          for r in RUNGS for sc in [load_scores(out, r, "lowo")]]),
        ("alias falsifier", "per-cell feature on dt_est_s and the iteration count within workload", "r2 > 0.5; top 10 features", D + "alias.csv", _verdicts_of(out, D + "alias.csv")),
        ("G-V two-class", "L0, L2, L3 against the benign families' mutual spread", "note at half the features", D + "gv_two_class_summary.csv",
         [("note" if r.get("note") else "report") for r in (_read_csv_opt(out, D + "gv_two_class_summary.csv") or [])] if _read_csv_opt(out, D + "gv_two_class_summary.csv") is not None else None),
    ]
    # carried-over plan11 gates run unchanged inside <out>
    for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
        reg.append((c, "precondition per cell (P2 Plan 02, C1 re-mapped)", "SPEC 3.3", G + "preconditions.csv",
                    [str(r.get(c, "")) for r in pre] if pre is not None else None))
    reg.append(("failed/ count", "failed jobs per cell (declared zero or recorded)", "0", G + "preconditions.csv",
                [str(r.get("failed_verdict", "")) for r in pre] if pre is not None else None))
    reg.append(("G-C", "calibration pulse on the kernels; lead connected on the benign side", "SPEC 3.4.3", G + "gc.csv",
                _verdicts_of(out, G + "gc.csv", where=lambda r: str(r.get("rep", "")) == "all")))
    reg.append(("G-P", "pass period declared or undeclared per kernel", "SPEC 3.4.1", G + "gp.csv", _verdicts_of(out, G + "gp.csv", "rhythm_verdict")))
    reg.append(("G-K0 kernels", "state-change disclosure per kernel against the idle band", "idle percentile 95", G + "gk0.csv", _verdicts_of(out, G + "gk0.csv")))
    reg.append(("G-F", "idle reps mutually inseparable under every rung (part i); kernels above the idle envelope (part ii)", "500 shuffles", G + "gf.csv",
                _verdicts_of(out, G + "gf.csv")))
    reg.append(("G-J", "J interpretable above k_factor times the floor's median K", "k_factor 3.0", G + "gj.csv", _verdicts_of(out, G + "gj.csv", "mask_verdict")))
    reg.append(("G-X", "campaign predictability on the kernels (LOKO, label = campaign)", "500 shuffles", G + "gx.csv", _verdicts_of(out, G + "gx.csv", "leak_verdict")))
    return [{"gate": g, "what": w, "threshold": t, "file": f, "verdicts": v} for g, w, t, f, v in reg]


def table5_gates(out: Path) -> dict:
    """One row per gate of this layer and per carried-over gate, the roll-up being the worst
    verdict across rungs (a refusal beats a pass) with the count in parentheses as text
    (SPEC_DETECTION 5.2.1; ML review 2.3 for G-OP over three splits)."""
    out = Path(out)
    rows = []
    for g in _gate_registry(out):
        if g["verdicts"] is None:
            miss = _missing(out, Path(out) / g["file"])
            rows.append({"gate": g["gate"], "what it checks": g["what"], "threshold": g["threshold"], "verdict (roll-up)": miss,
                         "refusal string": miss, "file": g["file"]})
            continue
        roll, ref = rollup(g["verdicts"])
        rows.append({"gate": g["gate"], "what it checks": g["what"], "threshold": g["threshold"], "verdict (roll-up)": roll,
                     "refusal string": ref, "file": g["file"]})
    val = _read_json_opt(out, "gates/detection/classes_validation.json")
    status = str(val.get("status", "--")) if val else _missing(out, _det(out) / "classes_validation.json")
    rows.insert(0, {"gate": "classes validate", "what it checks": "the author's mapping file against cells.csv (18 refusals)",
                    "threshold": "status ok", "verdict (roll-up)": status, "refusal string": status if is_refusal(status) or status == "refused" else "--",
                    "file": "gates/detection/classes_validation.json"})
    tw = _read_json_opt(out, "gates/detection/tripwire_check.json")
    tv = str(tw.get("verdict", "--")) if tw else _missing(out, _det(out) / "tripwire_check.json")
    rows.append({"gate": "tripwire check (D15)", "what it checks": "every Table 7 and Table 11 row carries G-F (i), G-ANCHOR (ii) and early-late idle",
                 "threshold": "0 rows without", "verdict (roll-up)": tv, "refusal string": tv if is_refusal(tv) else "--", "file": "gates/detection/tripwire_check.json"})
    note = note_line(out, ["roll-up = the worst verdict across rows (a refusal beats a pass); count in parentheses",
                           "classes_validation.json: status printed, never its refusals list (SPEC_DETECTION 2.2)"])
    paths = write_table(_tables_dir(out), "table5_gates", TABLE5_COLUMNS, [{k: _cell(v) for k, v in r.items()} for r in rows], note_comment=note, wide=False)
    _write_params(out, "table5_gates", _params_base(out, {"n_rows": len(rows)}))
    return paths


# ==============================================================================================
# Table 6, validity (SPEC_DETECTION 5.2.2)
# ==============================================================================================
TABLE6_COLUMNS = ["row", "rung", "quantity", "value", "null p95", "rank", "verdict", "note"]


def table6_validity(out: Path) -> dict:
    """Rows G-C, idle floor, harness floor, idle set against idle set, early against late idle,
    drift regression, state-change yield, G-K0 counts per tier, G-F (i) per rung (N3 Sec. 1
    validity block; SPEC_DETECTION 5.2.2; al-Kindi review 10 for the two drift units)."""
    out = Path(out)
    rows = []
    for rung in RUNGS:
        rows.append({"row": "G-C (calibrated core)", "rung": rung, "quantity": "pulse on the kernels", "value": "--", "null p95": "--", "rank": "--",
                     "verdict": gc_text(out, rung), "note": "lead connected, calibrated on the benign side (CR3 2.9)"})
    gk = _read_json_opt(out, "gates/detection/gk0.json")
    if gk is None:
        m = _missing(out, _det(out) / "gk0.json")
        rows.append({"row": "idle floor", "rung": "--", "quantity": "idle band edge (K)", "value": m, "null p95": "--", "rank": "--", "verdict": m, "note": "--"})
    else:
        env = gk.get("idle_envelope") or {}
        rows.append({"row": "idle floor", "rung": "--", "quantity": "idle band edge (K); envelope K_med, K_q90, frac", "value": _cell(gk.get("idle_band_edge")),
                     "null p95": "--", "rank": "--", "verdict": str(gk.get("status") or "ok"),
                     "note": f"envelope: K_med {_cell(env.get('K_med'))}; K_q90 {_cell(env.get('K_q90'))}; frac {_cell(env.get('frac_above_band'))}"})
        henv = gk.get("harness_envelope")
        rows.append({"row": "harness floor", "rung": "--", "quantity": "harness-idle envelope", "value": HARNESS_STAGE2_ABSENT if not henv else _cell(henv.get("K_med")),
                     "null p95": "--", "rank": "--", "verdict": HARNESS_STAGE2_ABSENT if not henv else "present", "note": "--"})
    ga = anchor_rows(out)
    for rung in RUNGS:
        for part, label in (("idle_sets", "idle set against idle set (G-ANCHOR ii)"), ("idle_early_late", "early against late idle")):
            if ga is None:
                m = _missing(out, _det(out) / "ganchor.csv")
                rows.append({"row": label, "rung": rung, "quantity": "AUC", "value": m, "null p95": "--", "rank": "--", "verdict": m, "note": "--"})
                continue
            hits = [r for r in ga if r.get("rung") == rung and r.get("part") == part]
            if not hits:
                rows.append({"row": label, "rung": rung, "quantity": "AUC", "value": "--", "null p95": "--", "rank": "--",
                             "verdict": not_run(f"ganchor.csv has no {part} row for {rung}"), "note": "--"})
                continue
            r = hits[-1]
            rows.append({"row": label, "rung": rung, "quantity": f"AUC ({r.get('label_space', 'campaign')}; n = {_cell(r.get('n_cells'))} cells)",
                         "value": _str_or_num(r.get("score")), "null p95": _cell(r.get("null_p95")), "rank": _cell(r.get("rank")),
                         "verdict": str(r.get("verdict") or "--"),
                         "note": "stage 1: one idle set captured after the sandbox run (P3 0a); the kernel period has no idle floor of its own" if part == "idle_sets" else "drift clause of the tripwire (CR3 2.12)"})
    dr = _read_csv_opt(out, "gates/detection/drift.csv")
    if dr is None:
        m = _missing(out, _det(out) / "drift.csv")
        rows.append({"row": "drift regression", "rung": "--", "quantity": "slope of K_med on order_index", "value": m, "null p95": "--", "rank": "--", "verdict": m, "note": "--"})
    else:
        for r in dr:
            rows.append({"row": "drift regression", "rung": "--", "quantity": f"{r.get('class', '--')}: slope of {r.get('statistic', 'K_med')} on order_index ({r.get('unit') or r.get('drift_unit') or '--'})",
                         "value": _str_or_num(r.get("slope")), "null p95": _cell(r.get("null_p95_abs_slope", r.get("null_p95"))), "rank": _cell(r.get("rank")),
                         "verdict": str(r.get("verdict") or "--"), "note": f"r2 {_cell(r.get('r2'))}; n {_cell(r.get('n'))}" + (f"; {r['label']}" if r.get("label") else "")})
    rows.append({"row": "state-change yield", "rung": "--", "quantity": "differ K against /proc/vmstat deltas (F11)", "value": YIELD_NOT_RECORDED, "null p95": "--",
                 "rank": "--", "verdict": YIELD_NOT_RECORDED, "note": "CR3 2.27"})
    gk0c = _read_csv_opt(out, "gates/detection/gk0_cells.csv")
    join = _read_csv_opt(out, "gates/detection/cell_classes.csv") or []
    classes_present = [c for c in DET_CLASSES if any(r.get("class") == c for r in join)] or list(DET_CLASSES[:1])
    for cls in classes_present:
        if gk0c is None:
            m = _missing(out, _det(out) / "gk0_cells.csv")
            rows.append({"row": "G-K0 counts per tier", "rung": "--", "quantity": cls, "value": m, "null p95": "--", "rank": "--", "verdict": m, "note": "--"})
            continue
        cc = [r for r in gk0c if r.get("class") == cls]
        nf = sum(1 for r in cc if r.get("verdict") == GK0_AT_FLOOR)
        nh = sum(1 for r in cc if r.get("verdict") == GK0_AT_HARNESS_FLOOR)
        na = sum(1 for r in cc if r.get("verdict") == GK0_ABOVE_FLOOR)
        nc = sum(1 for r in cc if r.get("verdict") == "control")
        rows.append({"row": "G-K0 counts per tier", "rung": "--", "quantity": f"{cls}: n at floor / n at harness floor / n above floor",
                     "value": f"{nf} / {nh} / {na}" + (f" (control {nc})" if nc else ""), "null p95": "--", "rank": "--",
                     "verdict": rollup([r.get("verdict", "") for r in cc])[0] if cc else not_run(f"no {cls} row in gk0_cells.csv"), "note": "at floor leaves the denominator (K3 F3)"})
    for rung in RUNGS:
        rows.append({"row": "G-F (i)", "rung": rung, "quantity": "idle reps mutually inseparable", "value": "--", "null p95": "--", "rank": "--",
                     "verdict": gf_text(out, rung), "note": "void voids every negative of the rung (F7)"})
    note = note_line(out, ["harness floor and state-change yield read their stage-1 strings",
                           "G-ANCHOR (ii) at stage 1: not applicable, one idle campaign (al-Kindi review, for the author 5)"])
    paths = write_table(_tables_dir(out), "table6_validity", TABLE6_COLUMNS, [{k: _cell(v) for k, v in r.items()} for r in rows], note_comment=note, wide=False)
    _write_params(out, "table6_validity", _params_base(out, {"n_rows": len(rows)}))
    return paths


# ==============================================================================================
# Table 7, detection under LOWO (SPEC_DETECTION 5.2.3)
# ==============================================================================================
TABLE7_COLUMNS = ["rung", "axis", "resolution (W x H)", "grid source", "feature count", "G-DIM", "ROC area", "AUC null p95", "AUC rank",
                  "TPR at 5% (in-fold)", "realized FPR", "TPR at 5% (pooled, post hoc)", "G-CAL", "TPR at 1% (resolution limit)",
                  "realized FPR at 1%", "TPR null p95", "TPR rank", "null verdict", "n assignments", "random scorer", "G-OP", "G-L (i)",
                  "G-C", "G-F (i)", "G-ANCHOR (ii)", "early-late idle", "B1-G3 quarantine", "TPR at 5% without quarantined",
                  "n sandbox cells scored", "n at floor", "n without score"]
SCORE_COLUMNS_7 = ["ROC area", "AUC null p95", "AUC rank", "TPR at 5% (in-fold)", "realized FPR", "TPR at 5% (pooled, post hoc)",
                   "TPR at 1% (resolution limit)", "realized FPR at 1%", "TPR null p95", "TPR rank", "TPR at 5% without quarantined"]


def _gl_text(out: Path, rung: str) -> str:
    rows = _read_csv_opt(out, "gates/detection/gl.csv")
    if rows is None:
        return _missing(out, _det(out) / "gl.csv")
    hits = [r for r in rows if r.get("rung") == rung]
    return str(hits[-1].get("verdict") or "--") if hits else not_run(f"gl.csv has no row for {rung}")


def _gop_text(out: Path, rung: str, split: str = "lowo", variant: str = "norm") -> str:
    rows = _read_csv_opt(out, "gates/detection/gop.csv")
    if rows is None:
        return _missing(out, _det(out) / "gop.csv")
    hits = [r for r in rows if r.get("rung") == rung and r.get("split", "lowo") == split and r.get("variant", "norm") == variant]
    if not hits:
        return not_run(f"gop.csv has no {split} row for {rung}")
    r = hits[-1]
    v = str(r.get("verdict") or "--")
    if v == GOP_SET_BY_FEW and r.get("setter_families"):
        v = f"{v} ({r['setter_families']})"
    return v


def _gdim_text(out: Path, display: str, rung: str, sc: dict | None) -> str:
    rows = _read_csv_opt(out, "gates/detection/gdim.csv")
    if rows is not None:
        key = "combined (matched)" if display == "combined (matched)" else rung
        hits = [r for r in rows if r.get("row") == display] or [r for r in rows if r.get("rung") == key and (r.get("row") in (None, "", rung, display))]
        if hits:
            return str(hits[-1].get("status") or "--")
    if sc and sc.get("dim_status"):
        return str(sc["dim_status"])
    return _missing(out, _det(out) / "gdim.csv")


def _gcal_text(sc: dict | None) -> str:
    if not sc or not isinstance(sc.get("gcal"), dict):
        return "--" if sc else not_run("scores.json missing")
    return str(sc["gcal"].get("verdict") or "--")


def _rank_text(sc: dict | None, statistic: str) -> str:
    v = null_field(sc, statistic, "rank_text")
    if v:
        return str(v)
    r = null_field(sc, statistic, "rank")
    n = null_field(sc, statistic, "n")
    if r is None:
        return "--"
    return f"rank {int(r)} of {int(n)}" if n else f"rank {int(r)}"


def _n_assignments(sc: dict | None, default):
    nl = sc.get("null") if (sc and isinstance(sc.get("null"), dict)) else None
    if nl and nl.get("n_assignments") is not None:
        return nl["n_assignments"]
    v = (sc.get("params") or {}).get("n_assignments") if sc else None
    return v if v is not None else default


def _external_from_predictions(out: Path, rung: str, stem: str, variant: str) -> dict | None:
    """The external block computed from predictions.csv rows of the `<split>/final` fold when
    scores.json carries no `external` block (builder A pools test-only cells into the positive
    denominators by member index; this fallback keeps P3 0a stage 3's own block and says so)."""
    d = split_dir(out, rung, stem, variant)
    if d is None or not (d / "predictions.csv").exists():
        return None
    rows = [r for r in read_csv(d / "predictions.csv") if r.get("class") == "external" or str(r.get("fold", "")).endswith("/final")]
    rows = [r for r in rows if r.get("class") == "external"]
    if not rows:
        return None
    scored = [r for r in rows if not r.get("score_status") and str(r.get("score", "")).strip()]
    pm = {}
    for r in rows:
        m = str(r.get("member_index", ""))
        d_ = pm.setdefault(m, {"hits": 0, "denominator": 0, "at_floor": 0, "n_cells": 0})
        d_["n_cells"] += 1
        if r.get("floor_verdict") in (GK0_AT_FLOOR, GK0_AT_HARNESS_FLOOR):
            d_["at_floor"] += 1
        elif r in scored:
            d_["denominator"] += 1
            d_["hits"] += int(str(r.get("flag_05", "")).lower() == "true")
    for m, d_ in pm.items():
        d_["eighths"] = f"{d_['hits']}/{d_['denominator']}" if d_["denominator"] else f"at floor ({d_['n_cells']})"
    den = [r for r in scored if r.get("floor_verdict") not in (GK0_AT_FLOOR, GK0_AT_HARNESS_FLOOR)]
    return {"n_scored": len(scored), "n_at_floor": sum(1 for r in rows if r.get("floor_verdict") in (GK0_AT_FLOOR, GK0_AT_HARNESS_FLOOR)),
            "n_without_score": len(rows) - len(scored),
            "tpr_05": (sum(str(r.get("flag_05", "")).lower() == "true" for r in den) / len(den)) if den else None,
            "tpr_01": (sum(str(r.get("flag_01", "")).lower() == "true" for r in den) / len(den)) if den else None,
            "per_member": pm, "fold": rows[0].get("fold", "lowo/final"), "source": "predictions.csv (scores.json has no external block)"}


def _row7(out: Path, display: str, rung: str, variant: str, stem: str, sc: dict | None, missing: str, override: str | None,
          n_assign_default) -> dict:
    row = {c: "--" for c in TABLE7_COLUMNS}
    gid = det_grid(out, rung)
    row.update({"rung": display, "axis": AXIS_OF_RUNG.get(rung, "--"), "resolution (W x H)": grid_label(gid) if gid else not_run(f"no selection for {rung}"),
                "grid source": grid_source(out), "G-L (i)": _gl_text(out, rung) if variant == "norm" else "level-inclusive ceiling (raw)",
                "G-C": gc_text(out, rung), "G-F (i)": gf_text(out, rung), "G-ANCHOR (ii)": anchor_text(out, rung, "idle_sets"),
                "early-late idle": anchor_text(out, rung, "idle_early_late"), "G-DIM": _gdim_text(out, display, rung, sc),
                "G-OP": _gop_text(out, rung, "lowo", variant), "random scorer": "AUC 0.5, TPR = FPR", "G-CAL": _gcal_text(sc)})
    if sc is None:
        for c in SCORE_COLUMNS_7 + ["feature count", "null verdict", "n assignments", "B1-G3 quarantine", "n sandbox cells scored", "n at floor", "n without score"]:
            row[c] = missing
        return row
    status = str(sc.get("status") or "ok")
    q = (sc.get("quarantine") or {}).get("quarantined_features") or []
    wq = sc.get("with_quarantine") if isinstance(sc.get("with_quarantine"), dict) else None
    row.update({"feature count": _cell(sc.get("feature_count_used") if sc.get("feature_count_used") not in (None, "") else sc.get("feature_count")),
                "null verdict": null_verdict_text(sc), "n assignments": _cell(_n_assignments(sc, n_assign_default)),
                "B1-G3 quarantine": ("; ".join(q) if q else "none"),
                "TPR at 5% without quarantined": (_str_or_num(wq.get("tpr_05")) if (q and wq) else ("not applicable: no feature quarantined" if not q else not_run("re-run without the quarantined feature missing"))),
                "n sandbox cells scored": _cell(sc.get("n_positive_scored")), "n at floor": _cell(sc.get("n_positive_at_floor")),
                "n without score": _cell(sc.get("n_without_score", 0))})
    if status != "ok":
        for c in SCORE_COLUMNS_7:
            row[c] = status
        return row
    if override:
        for c in SCORE_COLUMNS_7:
            row[c] = override
        return row
    row.update({"ROC area": _str_or_num(sc.get("auc")), "AUC null p95": _cell(null_field(sc, "auc", "p95")), "AUC rank": _rank_text(sc, "auc"),
                "TPR at 5% (in-fold)": _str_or_num(sc.get("tpr_05")), "realized FPR": _str_or_num(sc.get("fpr_05_realized")),
                "TPR at 5% (pooled, post hoc)": _str_or_num(sc.get("tpr_at_fpr05_pooled")),
                "TPR at 1% (resolution limit)": _str_or_num(sc.get("tpr_01")), "realized FPR at 1%": _str_or_num(sc.get("fpr_01_realized")),
                "TPR null p95": _cell(null_field(sc, "tpr05", "p95")), "TPR rank": _rank_text(sc, "tpr05")})
    nv = null_verdict_text(sc)
    if nv in (NULL_NOT_ESTIMABLE,):
        row["TPR null p95"] = nv
        row["TPR rank"] = nv
        row["AUC null p95"] = nv
        row["AUC rank"] = nv
    return row


def _l1_row(out: Path, display: str, rung: str, sc: dict | None, override: str | None) -> dict | None:
    """The `l1 (best single feature)` row of a rung (ML review 2.4): `scores.json["l1"]`."""
    if not sc:
        return None
    l1 = sc.get("l1")
    row = {c: "--" for c in TABLE7_COLUMNS}
    row.update({"rung": f"{display}: l1 (best single feature)", "axis": AXIS_OF_RUNG.get(rung, "--"), "resolution (W x H)": grid_label(det_grid(out, rung)),
                "grid source": grid_source(out), "feature count": 1, "G-DIM": "one feature", "random scorer": "AUC 0.5, TPR = FPR",
                "G-C": gc_text(out, rung), "G-F (i)": gf_text(out, rung), "G-ANCHOR (ii)": anchor_text(out, rung, "idle_sets"),
                "early-late idle": anchor_text(out, rung, "idle_early_late"), "G-L (i)": "--", "G-OP": "--", "G-CAL": "--",
                "n sandbox cells scored": _cell(sc.get("n_positive_scored")), "n at floor": _cell(sc.get("n_positive_at_floor")),
                "n without score": _cell(sc.get("n_without_score", 0)), "TPR at 5% without quarantined": "not applicable: one feature"})
    if not isinstance(l1, dict):
        row["B1-G3 quarantine"] = not_run("scores.json has no l1 block (ML review 2.4)")
        for c in SCORE_COLUMNS_7[:-1]:
            row[c] = not_run("l1 block missing")
        return row
    feat = l1.get("feature") or l1.get("features_chosen") or l1.get("best_feature_by_fold")
    if isinstance(feat, dict):
        feat = sorted(set(str(v) for v in feat.values()))
    if isinstance(feat, (list, tuple)):
        feat = "; ".join(str(f) for f in feat)
    row["B1-G3 quarantine"] = f"feature: {feat or '--'}"
    if override:
        for c in SCORE_COLUMNS_7:
            row[c] = override
        return row
    nl = l1.get("null") if isinstance(l1.get("null"), dict) else None
    tp = (nl or {}).get("tpr05") if nl else None
    row.update({"ROC area": _str_or_num(l1.get("auc")), "TPR at 5% (in-fold)": _str_or_num(l1.get("tpr_05")), "realized FPR": _str_or_num(l1.get("fpr_05_realized")),
                "TPR null p95": _cell((tp or {}).get("p95")) if tp else not_run("l1 null not run"),
                "TPR rank": str((tp or {}).get("rank_text") or "--") if tp else not_run("l1 null not run"),
                "null verdict": str((tp or {}).get("verdict") or "--") if tp else not_run("l1 null not run")})
    return row


def table7_detection(out: Path) -> dict:
    """LOWO, one row per rung display name (`apf raw`, `apf`, `wapf`, `persist`, `content`,
    `combined`, `combined (matched)`, `content channel 2'`, `comparator`), the external block
    when `external` cells exist, and one `l1 (best single feature)` row per rung (ML review 2.4).
    The full-feature forest is the headline (`score_source = full`); the majority baseline, G-N's
    sandbox status, row_unit, threshold_source and train_on_at_floor print in the note line
    (SPEC_DETECTION 5.2.3; ML review 2.1, 2.5; al-Farabi review M2, M4)."""
    out = Path(out)
    rows, l1_rows, ext_rows = [], [], []
    cj = _read_json_opt(out, "gates/detection/cell_classes.json") or {}
    n_assign_default = cj.get("n_assignments", "--")
    majority = "--"
    params_seen = {}
    for display, spec in RUNG_ROWS.items():
        if spec is None:
            row = {c: PLACEHOLDER_ROWS[display] for c in TABLE7_COLUMNS}
            row.update({"rung": display, "axis": "content of what changed (F10)", "grid source": grid_source(out), "random scorer": "AUC 0.5, TPR = FPR"})
            rows.append(row)
            continue
        rung, variant, stem = spec
        if display == "comparator":
            sc = load_scores(out, "comparator", "lowo", "norm")
            if sc is None:
                row = {c: COMPARATOR_ELSEWHERE for c in TABLE7_COLUMNS}
                row.update({"rung": display, "axis": "labelled analogue (RQ5)", "grid source": grid_source(out), "random scorer": "AUC 0.5, TPR = FPR"})
                rows.append(row)
                continue
        else:
            sc = load_scores(out, rung, stem, variant)
        override = rung_override(out, rung) if rung in RUNGS else None
        missing = scores_path_text(out, rung, stem, variant)
        rows.append(_row7(out, display, rung, variant, stem, sc, missing, override, n_assign_default))
        if sc and display not in ("combined (matched)", "comparator"):
            l1 = _l1_row(out, display, rung, sc, override)
            if l1:
                l1_rows.append(l1)
        ex = sc.get("external") if (sc and isinstance(sc.get("external"), dict)) else (_external_from_predictions(out, rung, stem, variant) if sc else None)
        if ex:
            row = {c: "--" for c in TABLE7_COLUMNS}
            row.update({"rung": f"{display} (external)", "axis": AXIS_OF_RUNG.get(rung, "--"), "resolution (W x H)": grid_label(det_grid(out, rung)),
                        "grid source": grid_source(out), "feature count": _cell(sc.get("feature_count_used") or sc.get("feature_count")),
                        "G-DIM": _gdim_text(out, display, rung, sc), "TPR at 5% (in-fold)": override or _str_or_num(ex.get("tpr_05")),
                        "TPR at 1% (resolution limit)": override or _str_or_num(ex.get("tpr_01")), "null verdict": not_applicable("external cells are test only (P3 0a stage 3)"),
                        "random scorer": "AUC 0.5, TPR = FPR", "G-C": gc_text(out, rung), "G-F (i)": gf_text(out, rung),
                        "G-ANCHOR (ii)": anchor_text(out, rung, "idle_sets"), "early-late idle": anchor_text(out, rung, "idle_early_late"),
                        "n sandbox cells scored": _cell(ex.get("n_scored")), "n at floor": _cell(ex.get("n_at_floor", 0)), "n without score": _cell(ex.get("n_without_score", 0)),
                        "B1-G3 quarantine": f"fold {ex.get('fold', 'lowo/final')}" + (f"; {ex['source']}" if ex.get("source") else "")})
            ext_rows.append(row)
        if sc and majority == "--" and sc.get("majority_accuracy") is not None and variant == "norm":
            majority = f"{_cell(sc.get('majority_accuracy'))} (n_b = {_cell(sc.get('n_benign_scored'))}, n_s = {_cell(sc.get('n_positive_in_denominator'))})"
        if sc and not params_seen:
            for k in ("row_unit", "threshold_source", "train_on_at_floor", "null_denominator_rule", "score_aggregation", "det_c1_rule", "score_source"):
                params_seen[k] = _params_val(sc, k)
    all_rows = rows + ext_rows + l1_rows
    note = note_line(out, [f"majority (always benign): accuracy a = n_b / (n_b + n_s) = {majority} (CR3 2.3)",
                           f"G-N sandbox: {_gn_sandbox_status(out)}",
                           f"TPR at 5% (pooled, post hoc): {POST_HOC_LABEL} (ML 1.6 item 2)",
                           "headline = the full-feature forest (score_source full); the re-run without the quarantined feature beside it (ML review 2.4)",
                           "; ".join(f"{k} = {_cell(v)}" for k, v in params_seen.items()) if params_seen else "params: not run: no scores.json read"])
    paths = write_table(_tables_dir(out), "table7_detection", TABLE7_COLUMNS, [{k: _cell(v) for k, v in r.items()} for r in all_rows], note_comment=note, wide=False)
    _write_params(out, "table7_detection", _params_base(out, {"n_rows": len(all_rows), "majority": majority, "split_params": params_seen,
                                                               "columns_with_post_hoc_label": ["TPR at 5% (pooled, post hoc)"]}))
    return paths


# ==============================================================================================
# Table 8, per-member recall in eighths (SPEC_DETECTION 5.2.4)
# ==============================================================================================
def _eighths(pm: dict | None, m: int) -> str:
    if not isinstance(pm, dict):
        return "--"
    e = pm.get(str(m)) or pm.get(m)
    if not isinstance(e, dict):
        return "--"
    return str(e.get("eighths") or "--")


def _recall_of(pm: dict | None, m: int):
    e = (pm or {}).get(str(m)) if isinstance(pm, dict) else None
    if not isinstance(e, dict) or not e.get("denominator"):
        return None
    try:
        return float(e["hits"]) / float(e["denominator"])
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def table8_member_recall(out: Path) -> dict:
    """One row per (rung, split) for lowo, loco, one_class; the member cells as `k/n` from
    `per_member.eighths` (n reduced by the member's at-floor cells; `at floor (n)` when every cell
    is at floor); min, median, max and never a mean (ML 2.4; P3 D5); the sub-family letters as
    the first data row; the LOCO row followed by the `L0 ratio` row of ML review 2.8; `n at floor`
    and `n without score` (al-Farabi M4); external members as `external m`."""
    out = Path(out)
    members = _members(out)
    ext = _external_members(out)
    mcols = [f"member {m}" for m, _ in members] + [f"external {m}" for m in ext]
    columns = ["rung", "split"] + mcols + ["min", "median", "max", "n at floor", "n without score", "note"]
    rows = [{"rung": "sub-family", "split": "--", **{f"member {m}": L for m, L in members}, **{f"external {m}": "X" for m in ext},
             "min": "--", "median": "--", "max": "--", "n at floor": "--", "n without score": "--", "note": "letters (P3 0a)"}]
    gvm = _read_csv_opt(out, "gates/detection/gv_two_class_members.csv")
    for display, spec in RUNG_ROWS.items():
        if spec is None or display in ("comparator",):
            continue
        rung, variant, stem = spec
        if stem != "lowo":
            continue
        for split in ("lowo", "loco", "one_class"):
            sc = load_scores(out, rung, split, variant)
            override = rung_override(out, rung)
            row = {"rung": display, "split": split + (" (signature ceiling)" if split == "loco" else ""), "note": "--"}
            if sc is None:
                m = (ONE_CLASS_NORM_ONLY if (split == "one_class" and variant == "raw") else scores_path_text(out, rung, split, variant))
                for c in mcols + ["min", "median", "max", "n at floor", "n without score"]:
                    row[c] = m
                rows.append(row)
                continue
            status = str(sc.get("status") or "ok")
            pm = sc.get("per_member")
            vals = []
            for m, _ in members:
                if override:
                    row[f"member {m}"] = override
                elif status != "ok":
                    row[f"member {m}"] = status
                else:
                    row[f"member {m}"] = _eighths(pm, m)
                    r = _recall_of(pm, m)
                    if r is not None:
                        vals.append(r)
            exb = sc.get("external") if isinstance(sc.get("external"), dict) else (_external_from_predictions(out, rung, split, variant) if ext else None)
            epm = (exb or {}).get("per_member")
            for m in ext:
                row[f"external {m}"] = override or (_eighths(epm, m) if epm else not_applicable("external cells not scored under this split"))
            row["min"] = _cell(min(vals)) if vals else "--"
            row["median"] = _cell(statistics.median(vals)) if vals else "--"
            row["max"] = _cell(max(vals)) if vals else "--"
            row["n at floor"] = _cell(sc.get("n_positive_at_floor"))
            row["n without score"] = _cell(sc.get("n_without_score", 0))
            if split == "one_class":
                row["note"] = f"model {sc.get('model', '--')}; {sc.get('g1c_label', '--')}"
            rows.append(row)
            if split == "loco":
                r2 = {"rung": display, "split": "loco: L0 ratio (reps identical?)", "min": "--", "median": "--", "max": "--",
                      "n at floor": "--", "n without score": "--", "note": f"ratio below {REPS_IDENTICAL_RATIO}: {REPS_IDENTICAL_NOTE}"}
                notes = []
                for m, _ in members:
                    if gvm is None:
                        r2[f"member {m}"] = _missing(out, _det(out) / "gv_two_class_members.csv")
                        continue
                    hits = [g for g in gvm if g.get("rung") == rung and str(g.get("member_index")) == str(m)]
                    if not hits:
                        r2[f"member {m}"] = not_run(f"no member {m} row for {rung}")
                        continue
                    r2[f"member {m}"] = _cell(hits[-1].get("L0_ratio"))
                    if hits[-1].get("note"):
                        notes.append(f"member {m}: {hits[-1]['note']}")
                for m in ext:
                    r2[f"external {m}"] = "--"
                if notes:
                    r2["note"] = "; ".join(notes)
                rows.append(r2)
    note = note_line(out, ["k/n = hits over the member's cells above floor; at floor (n) when every cell is at floor; no mean column (ML 2.4; P3 D5)",
                           "L0 ratio = L0_member / median benign L0_b (gv_two_class_members.csv; ML review 2.8)"])
    paths = write_table(_tables_dir(out), "table8_member_recall", columns, [{k: _cell(v) for k, v in r.items()} for r in rows], note_comment=note, wide=False)
    _write_params(out, "table8_member_recall", _params_base(out, {"n_rows": len(rows), "members": [m for m, _ in members], "external_members": ext,
                                                                   "reps_identical_ratio": REPS_IDENTICAL_RATIO}))
    return paths


# ==============================================================================================
# Table 9, the three pitfalls as sizes (SPEC_DETECTION 5.2.5)
# ==============================================================================================
TABLE9_COLUMNS = ["pitfall", "instrument", "rung", "class or member", "quantity", "size", "null p95", "rank", "verdict", "note"]


def table9_pitfalls(out: Path) -> dict:
    """Rows level x G-L (i), level x G-LM, campaign x G-ANCHOR (i), campaign x G-ANCHOR (ii), order
    x order test (per rung, class, half rule), order x drift regression (per class, unit), harness
    x harness clause, campaign x cross-campaign row, cadence and active fraction x leak probe
    (SPEC_DETECTION 5.2.5; al-Kindi review 10, 11; ML review 2.6). Every size is a number as text
    or a refusal string; no pass mark without its size."""
    out = Path(out)
    rows = []
    gl = _read_csv_opt(out, "gates/detection/gl.csv")
    for rung in RUNGS:
        if gl is None:
            m = _missing(out, _det(out) / "gl.csv")
            rows.append({"pitfall": "level", "instrument": "G-L (i)", "rung": rung, "class or member": "sandbox", "quantity": "TPR at 5% raw minus normalized (LOWO; the raw variant exists for apf)",
                         "size": m, "null p95": "--", "rank": "--", "verdict": m, "note": "--"})
            continue
        hits = [r for r in gl if r.get("rung") == rung]
        if not hits:
            rows.append({"pitfall": "level", "instrument": "G-L (i)", "rung": rung, "class or member": "sandbox", "quantity": "TPR at 5% raw minus normalized (LOWO; the raw variant exists for apf)",
                         "size": not_run(f"gl.csv has no row for {rung}"), "null p95": "--", "rank": "--", "verdict": not_run(f"gl.csv has no row for {rung}"), "note": "--"})
            continue
        r = hits[-1]
        raw, nrm = to_float(r.get("tpr05_raw")), to_float(r.get("tpr05_norm"))
        size = _cell(raw - nrm) if (raw is not None and nrm is not None) else (not_applicable("no raw variant for this rung") if rung != "apf" else "--")
        rows.append({"pitfall": "level", "instrument": "G-L (i)", "rung": rung, "class or member": "sandbox", "quantity": "TPR at 5% raw minus normalized (LOWO; the raw variant exists for apf)",
                     "size": size, "null p95": _cell(r.get("null_p95_norm")), "rank": _cell(r.get("rank_norm")), "verdict": str(r.get("verdict") or "--"),
                     "note": f"raw {_cell(r.get('tpr05_raw'))}; normalized {_cell(r.get('tpr05_norm'))}"})
    glm = _read_csv_opt(out, "gates/detection/glm.csv")
    if glm is None:
        m = _missing(out, _det(out) / "glm.csv")
        rows.append({"pitfall": "level", "instrument": "G-LM", "rung": "--", "class or member": "--", "quantity": "recall_lm", "size": m, "null p95": "--", "rank": "--",
                     "verdict": m, "note": "--"})
    else:
        for r in glm:
            rows.append({"pitfall": "level", "instrument": "G-LM", "rung": (r.get("rung", "--") + (f" ({r['variant']})" if r.get("variant") else "")),
                         "class or member": f"member {r.get('member_index', '--')} ({r.get('subfamily_letter', '-')})",
                         "quantity": f"recall_lm at level {_cell(r.get('level'))} in band [{_cell(r.get('band_lo'))}, {_cell(r.get('band_hi'))}] ({_cell(r.get('n_band_workloads'))} workloads)",
                         "size": _str_or_num(r.get("recall_lm")) if str(r.get("recall_lm", "")).strip() else str(r.get("verdict") or "--"),
                         "null p95": "--", "rank": "--", "verdict": str(r.get("verdict") or "--"),
                         "note": f"{r.get('level_label', GLM_LABEL_MEDIAN_K)}; recall_unrestricted {_cell(r.get('recall_unrestricted'))}; fpr_lm {_cell(r.get('fpr_lm'))}"})
    ga = anchor_rows(out)
    for rung in RUNGS:
        for part, inst in (("kernels", "G-ANCHOR (i)"), ("idle_sets", "G-ANCHOR (ii)")):
            if ga is None:
                m = _missing(out, _det(out) / "ganchor.csv")
                rows.append({"pitfall": "campaign", "instrument": inst, "rung": rung, "class or member": "kernels" if part == "kernels" else "idle", "quantity": "AUC",
                             "size": m, "null p95": "--", "rank": "--", "verdict": m, "note": "read as a size (K3 section 5 point 3)"})
                continue
            hits = [r for r in ga if r.get("rung") == rung and r.get("part") == part]
            if not hits:
                v = not_run(f"ganchor.csv has no {part} row for {rung}")
                rows.append({"pitfall": "campaign", "instrument": inst, "rung": rung, "class or member": "kernels" if part == "kernels" else "idle", "quantity": "AUC",
                             "size": v, "null p95": "--", "rank": "--", "verdict": v, "note": "--"})
                continue
            r = hits[-1]
            v = str(r.get("verdict") or "--")
            rows.append({"pitfall": "campaign", "instrument": inst, "rung": rung, "class or member": "kernels" if part == "kernels" else "idle",
                         "quantity": f"campaign predictability ({r.get('label_space', 'campaign')}; {_cell(r.get('n_labels'))} labels)",
                         "size": _str_or_num(r.get("score")) if str(r.get("score", "")).strip() else v, "null p95": _cell(r.get("null_p95")), "rank": _cell(r.get("rank")),
                         "verdict": v, "note": "read as a size, never as a switch between two headlines (K3 section 5 point 3)"})
    od = _read_csv_opt(out, "gates/detection/order.csv")
    if od is None:
        m = _missing(out, _det(out) / "order.csv")
        rows.append({"pitfall": "order", "instrument": "order test", "rung": "--", "class or member": "--", "quantity": "AUC of the half label", "size": m,
                     "null p95": "--", "rank": "--", "verdict": m, "note": "--"})
    else:
        for r in od:
            v = str(r.get("verdict") or "--")
            rows.append({"pitfall": "order", "instrument": "order test", "rung": r.get("rung", "--"), "class or member": r.get("class", "--"),
                         "quantity": f"AUC of the half label (half_rule {r.get('half_rule', '--')}; null unit {r.get('order_null_unit', '--')})",
                         "size": _str_or_num(r.get("score")) if str(r.get("score", "")).strip() else v, "null p95": _cell(r.get("null_p95")), "rank": _cell(r.get("rank")),
                         "verdict": v, "note": f"consequence {r.get('consequence', '--')}; n_assignments {_cell(r.get('n_assignments'))}"})
    dr = _read_csv_opt(out, "gates/detection/drift.csv")
    if dr is None:
        m = _missing(out, _det(out) / "drift.csv")
        rows.append({"pitfall": "order", "instrument": "drift regression", "rung": "--", "class or member": "--", "quantity": "slope of K_med on order_index", "size": m,
                     "null p95": "--", "rank": "--", "verdict": m, "note": "--"})
    else:
        for r in dr:
            v = str(r.get("verdict") or "--")
            rows.append({"pitfall": "order", "instrument": "drift regression", "rung": "--", "class or member": r.get("class", "--"),
                         "quantity": f"slope of {r.get('statistic', 'K_med')} on order_index ({r.get('unit') or r.get('drift_unit') or '--'})",
                         "size": _str_or_num(r.get("slope")) if str(r.get("slope", "")).strip() else v, "null p95": _cell(r.get("null_p95_abs_slope", r.get("null_p95"))),
                         "rank": _cell(r.get("rank")), "verdict": v, "note": f"r2 {_cell(r.get('r2'))}; n {_cell(r.get('n'))}" + (f"; {r['label']}" if r.get("label") else "")})
    hs = _read_csv_opt(out, "gates/detection/harness.csv")
    for rung in RUNGS:
        if hs is None:
            m = _missing(out, _det(out) / "harness.csv")
            rows.append({"pitfall": "harness", "instrument": "harness clause", "rung": rung, "class or member": "--", "quantity": "margins", "size": m, "null p95": "--",
                         "rank": "--", "verdict": m, "note": "--"})
            continue
        hits = [r for r in hs if r.get("rung") == rung]
        if not hits:
            v = not_run(f"harness.csv has no row for {rung}")
            rows.append({"pitfall": "harness", "instrument": "harness clause", "rung": rung, "class or member": "--", "quantity": "margins", "size": v, "null p95": "--",
                         "rank": "--", "verdict": v, "note": "--"})
            continue
        rel = [r for r in hits if r.get("block") == "relaunch"]
        feats = [r for r in hits if r.get("block") == "features"]
        if rel:
            r = rel[-1]
            rows.append({"pitfall": "harness", "instrument": "harness clause", "rung": rung, "class or member": "benign_relaunched",
                         "quantity": "re-launched control flagged fraction against the median member recall",
                         "size": _str_or_num(r.get("relaunched_flagged_fraction")), "null p95": "--", "rank": "--", "verdict": str(r.get("verdict") or "--"),
                         "note": f"median member recall {_cell(r.get('median_member_recall'))}"})
        vs = [str(r.get("verdict") or "--") for r in feats]
        roll, _ = rollup(vs) if vs else ("--", "--")
        comparable = sum(1 for v in vs if v.startswith("harness's"))
        rows.append({"pitfall": "harness", "instrument": "harness clause", "rung": rung, "class or member": "features",
                     "quantity": "features whose three margins are comparable (harness's)",
                     "size": (str(comparable) if vs and not all(is_refusal(v) for v in vs) else (vs[0] if vs else "--")), "null p95": "--", "rank": "--",
                     "verdict": roll, "note": f"{len(vs)} feature rows" if vs else "--"})
    rows.append({"pitfall": "campaign", "instrument": "cross-campaign row", "rung": "--", "class or member": "kernels (01c)", "quantity": "FPR of the benign model on the 01c cells",
                 "size": CROSS_CAMPAIGN_STAGE1, "null p95": "--", "rank": "--", "verdict": CROSS_CAMPAIGN_STAGE1, "note": "K3 move 21"})
    lp = _read_csv_opt(out, "gates/detection/leak_probe.csv")
    probes = (("n_pairs", "cadence"), ("dt_est_s", "cadence"), ("frac_above_band", "active fraction"), ("K_med", "level (beside G-L)"))
    for q, pit in probes:
        if lp is None:
            m = _missing(out, _det(out) / "leak_probe.csv")
            rows.append({"pitfall": pit, "instrument": "leak probe", "rung": "rung-free", "class or member": "sandbox", "quantity": f"one-feature model on {q} under LOWO",
                         "size": m, "null p95": "--", "rank": "--", "verdict": m, "note": "ML 3.4, 3.5; ML review 2.6"})
            continue
        hits = [r for r in lp if r.get("quantity") == q]
        if not hits:
            v = not_run(f"leak_probe.csv has no {q} row")
            rows.append({"pitfall": pit, "instrument": "leak probe", "rung": "rung-free", "class or member": "sandbox", "quantity": f"one-feature model on {q} under LOWO",
                         "size": v, "null p95": "--", "rank": "--", "verdict": v, "note": "ML 3.4, 3.5; ML review 2.6"})
            continue
        r = hits[-1]
        v = str(r.get("verdict") or "--")
        rows.append({"pitfall": pit, "instrument": "leak probe", "rung": "rung-free", "class or member": "sandbox", "quantity": f"one-feature model on {q} under LOWO (AUC)",
                     "size": _str_or_num(r.get("auc")) if str(r.get("auc", "")).strip() else v, "null p95": _cell(r.get("null_p95")), "rank": _cell(r.get("rank")),
                     "verdict": v, "note": "K_med audible is not a leak (ML review 2.6)" if q == "K_med" else "a log, never a feature (ML 3.4)"})
    note = note_line(out, ["every size is a number as text or a refusal string; the table never prints a pass mark without its size",
                           f"G-LM level label: {GLM_LABEL_MEDIAN_K}"])
    paths = write_table(_tables_dir(out), "table9_pitfalls", TABLE9_COLUMNS, [{k: _cell(v) for k, v in r.items()} for r in rows], note_comment=note, wide=False)
    _write_params(out, "table9_pitfalls", _params_base(out, {"n_rows": len(rows)}))
    return paths


# ==============================================================================================
# Table 10, the miss table and the false-positive table (SPEC_DETECTION 5.2.6)
# ==============================================================================================
MISS_COLUMNS = ["cell_id", "member_index", "subfamily_letter", "rung", "score", "threshold_05", "nearest_workload", "nearest_family", "distance", "axis",
                "axis_of_largest", "d_amount", "d_identity", "amount_cell", "identity_cell", "amount_centroid", "identity_centroid", "status"]
FP_COLUMNS = ["cell_id", "family", "workload_key", "rung", "score", "threshold_05", "nearest_member_index", "nearest_subfamily_letter", "distance", "axis",
              "axis_of_largest", "d_amount", "d_identity", "amount_cell", "identity_cell"]
REASON_COLUMN = "physical reason (M1 to M6)"


def _table10(out: Path, name: str, src: str, default_cols: list[str], table10_rung: str) -> dict:
    out = Path(out)
    rows = _read_csv_opt(out, f"gates/detection/{src}")
    if rows is None:
        cols = default_cols + [REASON_COLUMN]
        m = _missing(out, _det(out) / src)
        r = {c: m for c in default_cols}
        r[REASON_COLUMN] = ""
        r["rung"] = table10_rung
        note = note_line(out, ["assignments with counts, never a confusion matrix", "physical reason (M1 to M6): the author's column"])
        paths = {"all": write_table(_tables_dir(out), name, cols, [r], note_comment=note, wide=False)}
        for rung in RUNGS:
            paths[rung] = write_table(_tables_dir(out), f"{name}_{rung}", cols, [dict(r, rung=rung)], note_comment=note, wide=False)
        _write_params(out, name, _params_base(out, {"table10_rung": table10_rung, "n_rows": 0, "source": src}))
        return paths
    cols = (list(rows[0].keys()) if rows else default_cols)
    for c in default_cols:
        if c not in cols:
            cols.append(c)
    cols = cols + [REASON_COLUMN]
    note = note_line(out, ["assignments with counts, never a confusion matrix", "physical reason (M1 to M6): the author's column, left empty",
                           "axis = the resemblance axis (smaller standardized difference); axis_of_largest = where the residual lies (al-Kindi review 8)"])

    def fmt_rows(rs):
        outr = []
        for r in rs:
            d = {c: _cell(r.get(c)) for c in cols if c != REASON_COLUMN}
            d[REASON_COLUMN] = ""
            outr.append(d)
        return outr

    paths = {}
    for rung in RUNGS:
        sub = [r for r in rows if r.get("rung") == rung]
        if not sub:
            sub_rows = [{**{c: not_run(f"no {rung} row in {src}") for c in cols if c != REASON_COLUMN}, REASON_COLUMN: ""}]
            sub_rows[0]["rung"] = rung
        else:
            sub_rows = fmt_rows(sub)
        paths[rung] = write_table(_tables_dir(out), f"{name}_{rung}", cols, sub_rows, note_comment=note, wide=False)
    chosen = [r for r in rows if r.get("rung") == table10_rung]
    if chosen:
        main_rows = fmt_rows(chosen)
    else:
        main_rows = [{**{c: not_run(f"no {table10_rung} row in {src}") for c in cols if c != REASON_COLUMN}, REASON_COLUMN: ""}]
        main_rows[0]["rung"] = table10_rung
    paths["all"] = write_table(_tables_dir(out), name, cols, main_rows, note_comment=note, wide=False)
    _write_params(out, name, _params_base(out, {"table10_rung": table10_rung, "n_rows": len(rows), "source": src,
                                                 "n_at_floor_not_a_miss": sum(1 for r in rows if str(r.get("status", "")) == AT_FLOOR_NOT_A_MISS)}))
    return paths


def table10_misses(out: Path, *, table10_rung: str = "combined") -> dict:
    """The columns of `miss_table.csv` in that order with `physical reason (M1 to M6)` appended
    empty for the author; every rung's file as `table10_misses_<rung>`, the unsuffixed one for
    `--table10-rung` (SPEC_DETECTION 5.2.6; K3 move 18; al-Kindi review 8)."""
    return _table10(out, "table10_misses", "miss_table.csv", MISS_COLUMNS, table10_rung)


def table10_false_positives(out: Path, *, table10_rung: str = "combined") -> dict:
    """The columns of `fp_table.csv` likewise (SPEC_DETECTION 5.2.6)."""
    return _table10(out, "table10_false_positives", "fp_table.csv", FP_COLUMNS, table10_rung)


# ==============================================================================================
# Table 11, the four splits side by side (SPEC_DETECTION 5.2.7)
# ==============================================================================================
TABLE11_COLUMNS = ["rung", "LOWO TPR 5% (FPR)", "LOWO AUC", "LOCO TPR 5% (FPR) [signature ceiling]", "LOCO AUC", "G-SIG gap (TPR)", "G-SIG gap (AUC)", "G-SIG",
                   "LOFO FPR per family", "one-class TPR 5% (FPR)", "one-class AUC", "one-class model", "G-1C", "G-FP (families flagged at or above half)",
                   "benign recall per family (LOWO)", "G-F (i)", "G-ANCHOR (ii)", "early-late idle", "note"]


def _tpr_fpr(sc: dict | None, override: str | None, missing: str) -> str:
    if sc is None:
        return missing
    if override:
        return override
    st = str(sc.get("status") or "ok")
    if st != "ok":
        return st
    t = sc.get("tpr_05")
    if isinstance(t, str):
        return t
    return f"{_cell(t)} ({_cell(sc.get('fpr_05_realized'))})"


def _auc_text(sc, override, missing) -> str:
    if sc is None:
        return missing
    if override:
        return override
    st = str(sc.get("status") or "ok")
    return st if st != "ok" else _str_or_num(sc.get("auc"))


def _family_list(d) -> str:
    if not isinstance(d, dict) or not d:
        return "--"
    return "; ".join(f"{k} {_cell(v.get('fpr') if isinstance(v, dict) else v)}" for k, v in sorted(d.items()))


def table11_splits(out: Path) -> dict:
    """One row per rung: LOWO, LOCO (signature ceiling), LOFO (FPR per family), one-class, with
    G-SIG, G-1C, G-FP and the benign recall per family; the note prints the F9 sentence when the
    one-class tpr05 null is inside while LOWO's passes (SPEC_DETECTION 5.2.7; ML review 2.7;
    al-Kindi review 2); the three tripwire columns (SPEC_DETECTION 6.1)."""
    out = Path(out)
    rows = []
    gsig = _read_csv_opt(out, "gates/detection/gsig.csv")
    gfp = _read_csv_opt(out, "gates/detection/gfp.csv")
    g1c = _read_csv_opt(out, "gates/detection/g1c.csv")
    for display, spec in RUNG_ROWS.items():
        if spec is None or display == "comparator":
            row = {c: PLACEHOLDER_ROWS.get(display, COMPARATOR_ELSEWHERE) for c in TABLE11_COLUMNS}
            row["rung"] = display
            rows.append(row)
            continue
        rung, variant, stem = spec
        if stem != "lowo":
            continue
        override = rung_override(out, rung)
        lw = load_scores(out, rung, "lowo", variant)
        lo = load_scores(out, rung, "loco", variant)
        lf = load_scores(out, rung, "lofo", variant)
        oc = load_scores(out, rung, "one_class", variant)
        row = {"rung": display,
               "LOWO TPR 5% (FPR)": _tpr_fpr(lw, override, scores_path_text(out, rung, "lowo", variant)), "LOWO AUC": _auc_text(lw, override, scores_path_text(out, rung, "lowo", variant)),
               "LOCO TPR 5% (FPR) [signature ceiling]": _tpr_fpr(lo, override, scores_path_text(out, rung, "loco", variant)),
               "LOCO AUC": _auc_text(lo, override, scores_path_text(out, rung, "loco", variant)),
               "LOFO FPR per family": (override or (_family_list(lf.get("per_family_fpr")) if lf else scores_path_text(out, rung, "lofo", variant))),
               "one-class TPR 5% (FPR)": _tpr_fpr(oc, override, ONE_CLASS_NORM_ONLY if variant == "raw" else scores_path_text(out, rung, "one_class", variant)),
               "one-class AUC": _auc_text(oc, override, ONE_CLASS_NORM_ONLY if variant == "raw" else scores_path_text(out, rung, "one_class", variant)),
               "one-class model": (str(oc.get("model") or "--") if oc else "--"),
               "benign recall per family (LOWO)": (override or (("; ".join(f"{k} {_cell(1 - float(v['fpr']))}" for k, v in sorted(lw['per_family_fpr'].items()) if isinstance(v, dict) and to_float(v.get('fpr')) is not None) or "--") if lw and isinstance(lw.get("per_family_fpr"), dict) else (scores_path_text(out, rung, "lowo", variant) if lw is None else "--"))),
               "G-F (i)": gf_text(out, rung), "G-ANCHOR (ii)": anchor_text(out, rung, "idle_sets"), "early-late idle": anchor_text(out, rung, "idle_early_late")}
        if variant == "raw":
            row["LOFO FPR per family"] = override or row["LOFO FPR per family"]
        if gsig is None:
            m = _missing(out, _det(out) / "gsig.csv")
            row.update({"G-SIG gap (TPR)": m, "G-SIG gap (AUC)": m, "G-SIG": m})
        else:
            hits = [r for r in gsig if r.get("rung") == rung]
            if hits:
                r = hits[-1]
                row.update({"G-SIG gap (TPR)": override or _cell(r.get("gap_tpr05")), "G-SIG gap (AUC)": override or _cell(r.get("gap_auc")), "G-SIG": str(r.get("verdict") or "--")})
            else:
                v = not_run(f"gsig.csv has no row for {rung}")
                row.update({"G-SIG gap (TPR)": v, "G-SIG gap (AUC)": v, "G-SIG": v})
        if g1c is None:
            row["G-1C"] = _missing(out, _det(out) / "g1c.csv")
        else:
            hits = [r for r in g1c if r.get("rung") == rung]
            row["G-1C"] = (rollup([str(r.get("verdict") or r.get("label") or "--") for r in hits])[0] if hits else not_run(f"g1c.csv has no row for {rung}"))
        if gfp is None:
            row["G-FP (families flagged at or above half)"] = _missing(out, _det(out) / "gfp.csv")
        else:
            hits = [r for r in gfp if r.get("rung") == rung]
            insep = [r.get("family", "--") for r in hits if str(r.get("verdict", "")) == GFP_INSEPARABLE]
            row["G-FP (families flagged at or above half)"] = ("; ".join(insep) if insep else ("none" if hits else not_run(f"gfp.csv has no row for {rung}")))
        notes = []
        if variant == "raw":
            notes.append("level-inclusive ceiling")
        if oc is None:
            pass
        elif not isinstance(oc.get("null"), dict):
            notes.append("one-class null not run")
        else:
            ocv = null_verdict_text(oc)
            lwv = null_verdict_text(lw) if lw else "--"
            if ocv == NULL_INSIDE and lwv == PASS:
                notes.append(F9_SENTENCE)
            notes.append(f"one-class null: {ocv}")
        row["note"] = "; ".join(notes) if notes else "--"
        rows.append(row)
    note = note_line(out, ["LOCO column header carries `signature ceiling` (CR3 1.5)", f"F9 note when the one-class tpr05 null is inside while LOWO's passes: {F9_SENTENCE}",
                           "LOFO under kernel_family_rule = tier has two folds on stage 1 (section 7 item 3); read as a size"])
    paths = write_table(_tables_dir(out), "table11_splits", TABLE11_COLUMNS, [{k: _cell(v) for k, v in r.items()} for r in rows], note_comment=note, wide=False)
    _write_params(out, "table11_splits", _params_base(out, {"n_rows": len(rows)}))
    return paths


# ==============================================================================================
# level 2 and level 3 (SPEC_DETECTION 5.2.8, 5.2.9)
# ==============================================================================================
def _level_dir(out: Path, rung: str, level: str) -> Path | None:
    gid = det_grid(out, rung)
    if gid is None:
        return None
    return _det(out) / "splits" / rung / gid / level


def table_level2(out: Path, *, table10_rung: str = "combined") -> dict:
    """Per rung `table_level2_<rung>` plus `table_level2` for the chosen rung: rows per true
    sub-family with the predicted counts, recall, status, and the per-row null (`null_p95, rank,
    verdict` from `confusion.csv`, al-Kindi review 3); the B row's counts print `--` and its
    status the `no held-out test` string; `n at floor` (al-Kindi review 4). `macro_recall` is
    never printed (P3 0a)."""
    out = Path(out)
    paths = {}
    for rung in list(RUNGS):
        d = _level_dir(out, rung, "level2")
        letters = []
        rows = []
        if d is None or not (d / "confusion.csv").exists():
            m = not_run(f"no selection for {rung}") if d is None else _missing(out, d / "confusion.csv")
            cols = ["true sub-family", "n members", "n cells", "n at floor", "recall", "status", "null p95", "rank", "verdict"]
            rows.append({c: m for c in cols})
        else:
            conf = read_csv(d / "confusion.csv")
            letters = sorted({k[len("pred_"):] for r in conf for k in r.keys() if k.startswith("pred_")})
            cols = ["true sub-family", "n members", "n cells", "n at floor"] + [f"predicted {L}" for L in letters] + ["recall", "status", "null p95", "rank", "verdict"]
            override = rung_override(out, rung)
            for r in conf:
                row = {"true sub-family": r.get("true_subfamily", "--"), "n members": _cell(r.get("n_members")), "n cells": _cell(r.get("n_cells")),
                       "n at floor": _cell(r.get("n_at_floor")), "status": str(r.get("status") or "--")}
                held = str(r.get("status", "")).endswith("no held-out test")
                for L in letters:
                    row[f"predicted {L}"] = "--" if held else (override or _cell(r.get(f"pred_{L}")))
                row["recall"] = "--" if held else (override or _str_or_num(r.get("recall")))
                row["null p95"] = "--" if held else _cell(r.get("null_p95"))
                row["rank"] = "--" if held else _cell(r.get("rank"))
                row["verdict"] = str(r.get("verdict") or ("--" if held else not_run("per-row null verdict missing (al-Kindi review 3)")))
                rows.append(row)
        note = note_line(out, [f"rung {rung}; per sub-family, never averaged (P3 0a); status headline at >= 3 members, {L2_ONE_TRAIN} at 2",
                               "at-floor cells leave the denominator (al-Kindi review 4)"])
        paths[rung] = write_table(_tables_dir(out), f"table_level2_{rung}", cols, rows, note_comment=note, wide=False)
        if rung == table10_rung:
            paths["all"] = write_table(_tables_dir(out), "table_level2", cols, rows, note_comment=note, wide=False)
    if "all" not in paths:
        paths["all"] = paths[RUNGS[-1]]
    _write_params(out, "table_level2", _params_base(out, {"table10_rung": table10_rung}))
    return paths


def table_level3(out: Path, *, table10_rung: str = "combined") -> dict:
    """Per rung `table_level3_<rung>` plus `table_level3` for the chosen rung: rows per true
    member with the predicted members, `recall (eighths)` and `label` (SIGNATURE_CEILING), then
    the `accuracy` summary row with its null (SPEC_DETECTION 5.2.9; al-Kindi review 4)."""
    out = Path(out)
    paths = {}
    for rung in list(RUNGS):
        d = _level_dir(out, rung, "level3")
        rows = []
        if d is None or not (d / "confusion.csv").exists():
            m = not_run(f"no selection for {rung}") if d is None else _missing(out, d / "confusion.csv")
            cols = ["true member", "n at floor", "recall (eighths)", "label", "null p95", "rank", "verdict"]
            rows.append({c: m for c in cols})
        else:
            conf = read_csv(d / "confusion.csv")
            mem = {str(r.get("member_index")): r for r in read_csv(d / "members.csv")} if (d / "members.csv").exists() else {}
            sc = read_json(d / "scores.json") if (d / "scores.json").exists() else None
            members = sorted({k[len("pred_"):] for r in conf for k in r.keys() if k.startswith("pred_")}, key=lambda s: (len(s), s))
            cols = ["true member", "n at floor"] + [f"predicted {m}" for m in members] + ["recall (eighths)", "label", "null p95", "rank", "verdict"]
            override = rung_override(out, rung)
            for r in conf:
                m = str(r.get("true_member"))
                row = {"true member": m, "n at floor": _cell(r.get("n_at_floor")), "label": str(r.get("label") or SIGNATURE_CEILING), "null p95": "--", "rank": "--", "verdict": "--"}
                for m2 in members:
                    row[f"predicted {m2}"] = override or _cell(r.get(f"pred_{m2}"))
                row["recall (eighths)"] = override or (str(mem[m].get("eighths") or "--") if m in mem else _str_or_num(r.get("recall")))
                rows.append(row)
            acc = {"true member": "accuracy", "n at floor": _cell(sum(to_float(r.get("n_at_floor")) or 0 for r in conf)), "label": SIGNATURE_CEILING}
            for m2 in members:
                acc[f"predicted {m2}"] = "--"
            if sc is None:
                acc.update({"recall (eighths)": _missing(out, d / "scores.json"), "null p95": "--", "rank": "--", "verdict": _missing(out, d / "scores.json")})
            else:
                acc["recall (eighths)"] = override or _str_or_num(sc.get("accuracy"))
                nl = sc.get("null") if isinstance(sc.get("null"), dict) else None
                nb = (nl.get("accuracy") if isinstance(nl.get("accuracy"), dict) else (nl if "p95" in nl or "verdict" in nl else None)) if nl else None
                acc["null p95"] = _cell((nb or {}).get("p95")) if nb else not_run("null not run")
                acc["rank"] = str((nb or {}).get("rank_text") or "--") if nb else not_run("null not run")
                acc["verdict"] = str((nb or {}).get("verdict") or "--") if nb else not_run("null not run")
            rows.append(acc)
        note = note_line(out, [f"rung {rung}; {SIGNATURE_CEILING} on every row (P3 0a); null unit = cell (al-Kindi review 4)"])
        paths[rung] = write_table(_tables_dir(out), f"table_level3_{rung}", cols, rows, note_comment=note, wide=False)
        if rung == table10_rung:
            paths["all"] = write_table(_tables_dir(out), "table_level3", cols, rows, note_comment=note, wide=False)
    if "all" not in paths:
        paths["all"] = paths[RUNGS[-1]]
    _write_params(out, "table_level3", _params_base(out, {"table10_rung": table10_rung}))
    return paths


# ==============================================================================================
# the ladder (SPEC_DETECTION 5.2.10)
# ==============================================================================================
LADDER_COLUMNS = ["rung", "reading", "prefix (s)", "dt rule", "pairs (median over cells)", "pairs at 0.644 s", "pairs at 0.500 s", "n windows (median)",
                  "n cells with a window", "TPR at 5%", "realized FPR", "ROC area", "null verdict", "note"]


def table_ladder(out: Path) -> dict:
    """One row per (rung, reading, prefix) of `ladder.csv`; `pairs at 0.644 s` and `pairs at
    0.500 s` are the declared fixed-spacing counts round(prefix_s / dt) of K3 move 17 (uncapped);
    `pairs (median over cells)` is the file's `n_rows_median` (SPEC_DETECTION 5.2.10, 3.3.9)."""
    out = Path(out)
    lad = _read_csv_opt(out, "gates/detection/ladder.csv")
    rows = []
    if lad is None:
        m = _missing(out, _det(out) / "ladder.csv")
        for rung in RUNGS:
            rows.append({**{c: m for c in LADDER_COLUMNS}, "rung": rung})
    else:
        for r in lad:
            T = to_float(r.get("prefix_s"))
            rung = r.get("rung", "--")
            override = rung_override(out, rung) if rung in RUNGS else None
            note = str(r.get("note") or "")
            tpr = r.get("tpr_05")
            rows.append({"rung": rung, "reading": r.get("reading", "--"), "prefix (s)": _cell(T), "dt rule": _cell(r.get("dt")),
                         "pairs (median over cells)": _cell(r.get("n_rows_median")),
                         "pairs at 0.644 s": _cell(int(round(T / 0.644))) if T is not None else "--",
                         "pairs at 0.500 s": _cell(int(round(T / 0.500))) if T is not None else "--",
                         "n windows (median)": _cell(r.get("n_windows_median")), "n cells with a window": _cell(r.get("n_cells_with_window")),
                         "TPR at 5%": (note if note == LADDER_FROM_PAIR1_ONLY else (override or _str_or_num(tpr))),
                         "realized FPR": (note if note == LADDER_FROM_PAIR1_ONLY else (override or _str_or_num(r.get("fpr_05_realized")))),
                         "ROC area": (note if note == LADDER_FROM_PAIR1_ONLY else (override or _str_or_num(r.get("auc")))),
                         "null verdict": (note if note == LADDER_FROM_PAIR1_ONLY else str(r.get("null_verdict") or not_run("ladder null not run"))),
                         "note": note or "--"})
    lj = _read_json_opt(out, "gates/detection/ladder.json") or {}
    lp = lj.get("params") or {}
    note = note_line(out, [f"ladder_norm = {lp.get('norm', '--')}; ladder_head_drop_rule = {lp.get('ladder_head_drop_rule', '--')}; dt = {lp.get('dt', '--')} (K3 move 17; al-Farabi review M6)",
                           "pairs at 0.644 s and 0.500 s: the declared fixed-spacing counts, uncapped; the capped count is pairs (median over cells)"])
    paths = write_table(_tables_dir(out), "table_ladder", LADDER_COLUMNS, [{k: _cell(v) for k, v in r.items()} for r in rows], note_comment=note, wide=False)
    _write_params(out, "table_ladder", _params_base(out, {"n_rows": len(rows), "ladder_params": lp}))
    return paths


# ==============================================================================================
# G-V two-class (SPEC_DETECTION 5.2.11)
# ==============================================================================================
def table_gv_two_class(out: Path) -> dict:
    """The columns of `gv_two_class.csv` sorted by rung then `L3_over_L3_families` descending,
    with the summary row per rung from `gv_two_class_summary.csv` (SPEC_DETECTION 5.2.11)."""
    out = Path(out)
    gv = _read_csv_opt(out, "gates/detection/gv_two_class.csv")
    gs = _read_csv_opt(out, "gates/detection/gv_two_class_summary.csv") or []
    cols = ["rung", "feature", "L0", "L2", "L3", "L0_b", "L2_b", "L3_families", "L3_over_L3_families", "note"]
    rows = []
    if gv is None:
        m = _missing(out, _det(out) / "gv_two_class.csv")
        for rung in RUNGS:
            rows.append({**{c: m for c in cols}, "rung": rung})
    else:
        cols = [c for c in gv[0].keys()] + ["note"] if gv else cols
        for rung in RUNGS:
            sub = sorted([r for r in gv if r.get("rung") == rung], key=lambda r: -(to_float(r.get("L3_over_L3_families")) or -math.inf))
            for r in sub:
                rows.append({**{c: _cell(r.get(c)) for c in cols if c != "note"}, "note": "--"})
            srow = [s for s in gs if s.get("rung") == rung]
            if srow:
                s = srow[-1]
                rows.append({**{c: "--" for c in cols}, "rung": rung, "feature": f"summary: {_cell(s.get('n_features_L3_le_L3_families'))} of {_cell(s.get('n_features'))} features with L3 <= L3_families",
                             "note": s.get("note") or "--"})
            else:
                rows.append({**{c: "--" for c in cols}, "rung": rung, "feature": "summary", "note": _missing(out, _det(out) / "gv_two_class_summary.csv")})
    note = note_line(out, ["L0 within member, L2 across members, L3 across the two classes; the benign side likewise (CR3 2.11)"])
    paths = write_table(_tables_dir(out), "table_gv_two_class", cols, rows, note_comment=note, wide=False)
    _write_params(out, "table_gv_two_class", _params_base(out, {"n_rows": len(rows)}))
    return paths


# ==============================================================================================
# the per-cell appendix (SPEC_DETECTION 5.2.12)
# ==============================================================================================
def table_cells_detection(out: Path) -> dict:
    """`cell_id, class, member, sub-family, rep, order token, campaign, n pairs, dt (s), K median,
    G-K0 verdict, admissible, LOWO score per rung, flagged at 5% per rung`. Never prints `path`,
    `label`, `test_label`, `traj_file` or `order_index` (SPEC_DETECTION 5.2.12; al-Farabi M11)."""
    out = Path(out)
    cells = load_cells(out)
    join = {r["cell_id"]: r for r in (_read_csv_opt(out, "gates/detection/cell_classes.csv") or [])}
    adm = {r["cell_id"]: r for r in (_read_csv_opt(out, "gates/detection/admissibility.csv") or [])}
    gk0 = {r["cell_id"]: r for r in (_read_csv_opt(out, "gates/detection/gk0_cells.csv") or [])}
    ls = _read_csv_opt(out, "gates/detection/letter_sequence.csv")
    ls_line = None
    p_ls = _det(out) / "letter_sequence.csv"
    if ls is None and p_ls.exists():
        ls_line = p_ls.read_text(encoding="utf-8").strip()
    elif ls is not None and ls and "token" not in ls[0]:
        ls_line = p_ls.read_text(encoding="utf-8").strip().splitlines()[0]
    preds = {}
    for rung in RUNGS:
        d = split_dir(out, rung, "lowo", "norm")
        if d is not None and (d / "predictions.csv").exists():
            preds[rung] = {r["cell_id"]: r for r in read_csv(d / "predictions.csv")}
    cols = ["cell_id", "class", "member", "sub-family", "rep", "order token", "campaign", "n pairs", "dt (s)", "K median", "G-K0 verdict", "admissible"] + \
           [f"LOWO score {r}" for r in RUNGS] + [f"flagged at 5% {r}" for r in RUNGS]
    rows = []
    for c in cells:
        cid = c["cell_id"]
        j = join.get(cid, {})
        sc_p = Path(out) / "extract" / cid / "sidecar.json"
        sc = None
        if sc_p.exists():
            try:
                sc = read_json(sc_p)
            except Exception:
                sc = None
        token = j.get("order_token") or (ls_line if ls_line else (not_run("order_index missing") if j else "--"))
        row = {"cell_id": cid, "class": j.get("class") or ("unassigned" if not j else "--"), "member": _cell(j.get("member_index")) if j.get("class") in ("sandbox", "external") else "--",
               "sub-family": j.get("subfamily_letter") or "--", "rep": _cell(c.get("rep")), "order token": token, "campaign": j.get("campaign") or c.get("campaign") or "--",
               "n pairs": _cell(sc.get("n_pairs")) if sc else _missing(out, sc_p), "dt (s)": _cell(sc.get("dt_est_s")) if sc else "--",
               "K median": _cell(gk0[cid].get("K_med")) if cid in gk0 else (_cell(sc.get("K_median")) if sc else "--"),
               "G-K0 verdict": gk0[cid].get("verdict") if cid in gk0 else _missing(out, _det(out) / "gk0_cells.csv"),
               "admissible": adm[cid].get("admissible") if cid in adm else _missing(out, _det(out) / "admissibility.csv")}
        for rung in RUNGS:
            p = preds.get(rung, {}).get(cid)
            if rung not in preds:
                row[f"LOWO score {rung}"] = scores_path_text(out, rung, "lowo", "norm").replace("scores.json", "predictions.csv")
                row[f"flagged at 5% {rung}"] = row[f"LOWO score {rung}"]
            elif p is None:
                row[f"LOWO score {rung}"] = not_applicable("not scored under LOWO")
                row[f"flagged at 5% {rung}"] = not_applicable("not scored under LOWO")
            else:
                st = p.get("score_status") or ""
                row[f"LOWO score {rung}"] = st if st else _cell(p.get("score"))
                row[f"flagged at 5% {rung}"] = st if st else _cell(p.get("flag_05"))
        rows.append(row)
    note = note_line(out, ["order token from letter_sequence.csv; path, label, test_label, traj_file and order_index are never printed (al-Farabi M11)"])
    paths = write_table(_tables_dir(out), "table_cells_detection", cols, [{k: _cell(v) for k, v in r.items()} for r in rows], note_comment=note, wide=False)
    _write_params(out, "table_cells_detection", _params_base(out, {"n_rows": len(rows)}))
    return paths


# ==============================================================================================
# the manifest (SPEC_DETECTION 5.2.13)
# ==============================================================================================
def manifest(out: Path) -> Path:
    """`report/detection/manifest.json`: every file under `report/detection/` and `gates/detection/`
    with its sha256 and size, the package version, the detection ledger, every `params` block
    collected, the class counts, S, B, n_assignments, grid_source."""
    out = Path(out)
    files, params_blocks = {}, {}
    for sub in ("report/detection", "gates/detection"):
        d = out / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.name == "manifest.json":
                continue
            rel = str(p.relative_to(out))
            files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            if p.suffix == ".json":
                try:
                    j = read_json(p)
                    if isinstance(j, dict) and "params" in j:
                        params_blocks[rel] = j["params"]
                except Exception:
                    params_blocks[rel] = "unreadable"
    ledger = None
    lp = out / "driver_detection_state.json"
    if lp.exists():
        try:
            ledger = read_json(lp)
        except Exception:
            ledger = "unreadable"
    cj = _read_json_opt(out, "gates/detection/cell_classes.json") or {}
    d = result_json("detection.manifest", _params_base(out), CITATIONS["manifest"],
                    {"package_version": PACKAGE_VERSION, "n_files": len(files), "files": files, "params_blocks": params_blocks,
                     "driver_detection_state": ledger, "class_counts": class_counts(out), "S": cj.get("S"), "B": cj.get("B"),
                     "n_assignments": cj.get("n_assignments"), "grid_source": grid_source(out)})
    return write_json(out / "report" / "detection" / "manifest.json", d)


# ==============================================================================================
# CLI
# ==============================================================================================
def run(out: Path, only: list[str] | None = None, table10_rung: str = "combined") -> dict:
    names = list(only) if only else list(TABLE_NAMES)
    unknown = [n for n in names if n not in TABLE_NAMES]
    if unknown:
        raise ValueError(f"unknown table name(s): {unknown}; known: {TABLE_NAMES}")
    written = {}
    fns = {"table4_tiers": table4_tiers, "table5_gates": table5_gates, "table6_validity": table6_validity,
           "table7_detection": table7_detection, "table8_member_recall": table8_member_recall, "table9_pitfalls": table9_pitfalls,
           "table11_splits": table11_splits, "table_ladder": table_ladder, "table_gv_two_class": table_gv_two_class,
           "table_cells_detection": table_cells_detection}
    for n in names:
        if n in fns:
            written[n] = fns[n](out)
        elif n == "table10_misses":
            written[n] = table10_misses(out, table10_rung=table10_rung)["all"]
        elif n == "table10_false_positives":
            written[n] = table10_false_positives(out, table10_rung=table10_rung)["all"]
        elif n == "table_level2":
            written[n] = table_level2(out, table10_rung=table10_rung)["all"]
        elif n == "table_level3":
            written[n] = table_level3(out, table10_rung=table10_rung)["all"]
    if "manifest" in names:
        written["manifest"] = manifest(out)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan11 detection tables (builder B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated: " + ",".join(TABLE_NAMES))
    ap.add_argument("--table10-rung", default="combined", choices=list(RUNGS))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not (out / "cells.csv").exists():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr)
        return 2
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None
    try:
        written = run(out, only, a.table10_rung)
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    for n, v in written.items():
        print(f"{n}: {v['csv'] if isinstance(v, dict) else v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
