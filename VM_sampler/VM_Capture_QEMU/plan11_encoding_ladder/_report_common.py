#!/usr/bin/env python3
"""_report_common.py -- helpers shared by builder 3's modules (tables, figures, skeleton, driver).

Written 2026-09-16 (builder 3, report). SPEC.md section 1 assigns builder 3 `tables.py`,
`figures.py`, `latex_skeleton.py`, the driver and the runbook. This module exists because the
three builders work in parallel: at build time `schema.py` (builder 1) and `verdicts.py`
(builder 2) were not yet on disk, so every constant builder 3 needs is imported from them when
they exist and otherwise taken from the values SPEC.md declares (sections 2.6, 3.0, 3.1.3).
Nothing here computes a verdict; builder 3 reads builder 2's result files by their documented
column names and never recomputes one (SPEC section 1, "Interfaces between builders").

Citation for the fallbacks: SPEC 2.6 (`KERNELS`, `ARCHETYPES`, `LEVEL_MATCHED_SETS`,
`campaign_of`), SPEC 3.0 (the verdict vocabulary), SPEC 3.1.3 (the grid), P2 Table 2 (the
axis of each rung), P2 Table 3 (the kernels and their predicted archetypes).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))

import contextlib
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
from datetime import datetime, timezone

PACKAGE_NAME = "plan11_encoding_ladder"

# ----------------------------------------------------------------------------------------------
# schema.py (builder 1): import when present, else SPEC 2.6 / 3.1.3 values
# ----------------------------------------------------------------------------------------------
try:  # pragma: no cover - depends on builder 1's file being present
    from plan11_encoding_ladder import schema as _schema  # type: ignore
    SCHEMA_SOURCE = "plan11_encoding_ladder.schema"
except Exception:  # ImportError or a namespace package without schema.py
    _schema = None
    SCHEMA_SOURCE = "fallback: SPEC 2.6 / 3.1.3 values (schema.py absent at build time)"

try:  # pragma: no cover
    from plan11_encoding_ladder import __version__ as PACKAGE_VERSION  # type: ignore
except Exception:
    PACKAGE_VERSION = "0.1.0"


def _sattr(name, default):
    return getattr(_schema, name, default) if _schema is not None else default


N_PAGES = _sattr("N_PAGES", 262144)
PAGE_SIZE = _sattr("PAGE_SIZE", 4096)
BITS_PER_PAGE = _sattr("BITS_PER_PAGE", 32768)
DURATION_S = _sattr("DURATION_S", 600)
DT_BRACKET_S = _sattr("DT_BRACKET_S", (0.500, 0.644))
QUANTILES = _sattr("QUANTILES", (0.05, 0.25, 0.50, 0.75, 0.95))
GRID_WINDOWS = _sattr("GRID_WINDOWS", (8, 16, 32, 64, "whole"))
GRID_HOP_RATIOS = _sattr("GRID_HOP_RATIOS", (0.25, 0.50, 1.00))

KERNELS = _sattr("KERNELS", (
    ("gemm", "WORKING-SET"), ("floyd", "WORKING-SET"), ("gibbs", "WORKING-SET"),
    ("nbody", "WORKING-SET"), ("spmm", "WORKING-SET"), ("stencil_jacobi", "WORKING-SET"),
    ("fft", "SCATTER"), ("histogram", "SCATTER"), ("fem_assembly", "SCATTER"),
    ("lexer", "SEQUENTIAL-GROW"), ("rmat_gen", "SEQUENTIAL-GROW"),
    ("bnb_tsp", "FRONTIER-CHURN"),
))
ARCHETYPES = _sattr("ARCHETYPES", ("IDLE", "WORKING-SET", "SCATTER", "SEQUENTIAL-GROW", "FRONTIER-CHURN"))
LEVEL_MATCHED_SETS = _sattr("LEVEL_MATCHED_SETS", (("floyd", "histogram", "nbody"), ("fft", "gemm"), ("fft", "stencil_jacobi")))
LEVEL_MATCHED_ADDED = _sattr("LEVEL_MATCHED_ADDED", {2: "2026-09-28"})     # AA A12: set index -> the date it was added
LEVEL_MATCHED_LETTERS = tuple(chr(ord("A") + i) for i in range(len(LEVEL_MATCHED_SETS)))   # A, B, C


def level_matched_status(i: int) -> str:
    """"declared", or "added <date>" for a set listed in LEVEL_MATCHED_ADDED (AA A12); as schema."""
    return f"added {LEVEL_MATCHED_ADDED[i]}" if i in LEVEL_MATCHED_ADDED else "declared"


KERNEL_NAMES = tuple(k for k, _ in KERNELS)
ARCHETYPE_OF = dict(KERNELS)
LEVEL_SET_OF = {}        # kernel -> its set letters joined by ", " (fft is in B and C since AA A12)
for _i, _set in enumerate(LEVEL_MATCHED_SETS):
    for _k in _set:
        LEVEL_SET_OF[_k] = f"{LEVEL_SET_OF[_k]}, {LEVEL_MATCHED_LETTERS[_i]}" if _k in LEVEL_SET_OF else LEVEL_MATCHED_LETTERS[_i]

EXTRACT_COLUMNS = _sattr("EXTRACT_COLUMNS", tuple(
    ["seq", "K", "n_persist", "n_union", "J", "J_null_inter", "J_null"]
    + [f"{ch}_{s}_all" for ch in ("ham", "l0", "l1") for s in ("sum", "q05", "q25", "q50", "q75", "q95")]
    + [f"{ch}_{s}_per" for ch in ("ham", "l0", "l1") for s in ("sum", "q05", "q25", "q50", "q75", "q95")]
    + [f"r_l0_{q}_per" for q in ("q05", "q25", "q50", "q75", "q95")]
    + [f"r_l1l0_{q}_per" for q in ("q05", "q25", "q50", "q75", "q95")]
    + [f"r_haml0_{q}_per" for q in ("q05", "q25", "q50", "q75", "q95")]
))


def grid_id(W, H) -> str:
    """SPEC 3.1.3: `W{W}_H{H}` for integer points, `Wall_Hall` for the whole cell."""
    if _schema is not None and hasattr(_schema, "grid_id"):
        return _schema.grid_id(W, H)
    if W == "whole" or H == "whole" or W is None:
        return "Wall_Hall"
    return f"W{int(W)}_H{int(H)}"


def grid_points(n_series: int | None = None) -> list:
    """SPEC 3.1.3: the 13 (W, H) pairs in grid order; the whole cell is (n_series, n_series)."""
    if _schema is not None and hasattr(_schema, "grid_points") and n_series is not None:
        return list(_schema.grid_points(n_series))
    pts = []
    for W in GRID_WINDOWS:
        if W == "whole":
            continue
        for r in GRID_HOP_RATIOS:
            pts.append((int(W), max(1, int(round(W * r)))))
    pts.append(("whole", "whole"))
    return pts


GRID_IDS = tuple(grid_id(W, H) for W, H in grid_points())


def campaign_of(label: str) -> str:
    """SPEC 2.6 campaign normalization."""
    if _schema is not None and hasattr(_schema, "campaign_of"):
        return _schema.campaign_of(label)
    if label.startswith("dwarfs1"):
        return "dwarfs1"
    if label == "sandbox_deepdive_01c":
        return "01c"
    if label == "sandbox_deepdive_01c1":
        return "01c1"
    return label


# ----------------------------------------------------------------------------------------------
# verdicts.py (builder 2): import when present, else SPEC 3.0 vocabulary
# ----------------------------------------------------------------------------------------------
try:  # pragma: no cover
    from plan11_encoding_ladder import verdicts as V  # type: ignore
    VERDICTS_SOURCE = "plan11_encoding_ladder.verdicts"
except Exception:
    V = None
    VERDICTS_SOURCE = "fallback: SPEC 3.0 vocabulary (verdicts.py absent at build time)"


def _vattr(name, default):
    return getattr(V, name, default) if V is not None else default


PASS = _vattr("PASS", "pass")
FAIL = _vattr("FAIL", "fail")
TREND_PRESENT = _vattr("TREND_PRESENT", "trend present")
G2_ABOVE_NYQUIST = _vattr("G2_ABOVE_NYQUIST", "not applicable, rhythm above Nyquist")
G2_UNDETERMINED = _vattr("G2_UNDETERMINED", "undetermined by the interval calibration")
GP_UNDECLARED = _vattr("GP_UNDECLARED", "undeclared")
G3_PRESENT = _vattr("G3_PRESENT", "rhythm flag: present")
G3_ABSENT = _vattr("G3_ABSENT", "rhythm flag: absent")
GORD_ORDER_BLIND = _vattr("GORD_ORDER_BLIND", "order-blind")
GORD_RESOLUTION = _vattr("GORD_RESOLUTION", "resolution")
GK0_IDLE_MEASURED = _vattr("GK0_IDLE_MEASURED", "IDLE, measured")
GK0_ABOVE_FLOOR = _vattr("GK0_ABOVE_FLOOR", "above floor")
GF_INSEPARABLE = _vattr("GF_INSEPARABLE", "inseparable at floor")
GF_VOID = _vattr("GF_VOID", "void: idle reps separable under this rung")
GF_AT_FLOOR = _vattr("GF_AT_FLOOR", "at floor in this lead")
GC_DISCONNECTED = _vattr("GC_DISCONNECTED", "disconnected lead")
GDEC_DECAY = _vattr("GDEC_DECAY", "decay")
GDEC_NO_DECAY = _vattr("GDEC_NO_DECAY", "no decay")
GDEC_NO_BEYOND_BREADTH = _vattr("GDEC_NO_BEYOND_BREADTH", "no decay beyond breadth")
GDEC_NO_BEYOND_FLOOR = _vattr("GDEC_NO_BEYOND_FLOOR", "no decay beyond floor or host")
GDEC_NOT_RESOLVED = _vattr("GDEC_NOT_RESOLVED", "decay not resolved")
NEAR_UNFALSIFIABLE = _vattr("NEAR_UNFALSIFIABLE", "near_unfalsifiable")
GL_LEVEL_ONLY = _vattr("GL_LEVEL_ONLY", "level only")
GL_SHOT_NOISE = _vattr("GL_SHOT_NOISE", "refused: shot noise explains CV")
GN_HEADLINE = _vattr("GN_HEADLINE", "headline")
GN_ONE_TRAIN = _vattr("GN_ONE_TRAIN", "one training kernel per fold")
GN_NOVELTY = _vattr("GN_NOVELTY", "structural novelty")
GN_NO_ROW = _vattr("GN_NO_ROW", "no kernel row")
GX_POOLING_STANDS = _vattr("GX_POOLING_STANDS", "pooling stands")
GX_LEAK = _vattr("GX_LEAK", "campaign predictable")
GX_CONFOUND_TOTAL = _vattr("GX_CONFOUND_TOTAL", "confound: total")
GDIM_FULL = _vattr("GDIM_FULL", "full vector")
GDIM_REDUCED = _vattr("GDIM_REDUCED", "declared reduction")
GM_BEATS = _vattr("GM_BEATS", "beats")
GM_DIFFERENCE = _vattr("GM_DIFFERENCE", "difference with margin")
GV_NOT_ESTIMABLE = _vattr("GV_NOT_ESTIMABLE", "LOKO not estimable")
GV_ESTIMABLE = _vattr("GV_ESTIMABLE", "estimable")

_NAMED_REFUSALS = {
    TREND_PRESENT, G2_ABOVE_NYQUIST, G2_UNDETERMINED, GF_VOID, GF_AT_FLOOR, GC_DISCONNECTED,
    GDEC_NO_BEYOND_BREADTH, GDEC_NO_BEYOND_FLOOR, GDEC_NOT_RESOLVED, NEAR_UNFALSIFIABLE,
    GL_LEVEL_ONLY, GL_SHOT_NOISE, GV_NOT_ESTIMABLE,
}


def not_run(reason: str) -> str:
    return V.not_run(reason) if V is not None and hasattr(V, "not_run") else f"not run: {reason}"


def not_applicable(reason: str) -> str:
    return (V.not_applicable(reason) if V is not None and hasattr(V, "not_applicable")
            else f"not applicable: {reason}")


def refused(reason: str) -> str:
    return V.refused(reason) if V is not None and hasattr(V, "refused") else f"refused: {reason}"


def pending(reason: str) -> str:
    return V.pending(reason) if V is not None and hasattr(V, "pending") else f"pending: {reason}"


def is_refusal(s) -> bool:
    """SPEC 3.0: FAIL, any `refused:` / `not run:` / `not applicable:` string, or a named refusal."""
    if V is not None and hasattr(V, "is_refusal"):
        return bool(V.is_refusal(s))
    if s is None:
        return False
    s = str(s)
    if s == FAIL or s in _NAMED_REFUSALS:
        return True
    return s.startswith("refused:") or s.startswith("not run:") or s.startswith("not applicable:")


# ----------------------------------------------------------------------------------------------
# rungs, splits, axes (SPEC 3.1.1, 4.1; P2 Table 2)
# ----------------------------------------------------------------------------------------------
RUNGS = ("apf", "wapf", "persist", "content", "combined")
RUNG_DISPLAY = {"apf": "APF", "wapf": "wAPF", "persist": "persistence",
                "content": "content-change", "combined": "combined"}
AXIS_OF_RUNG = {  # P2 Table 2, "Axis kept"
    "apf": "breadth", "wapf": "breadth x amount folded", "persist": "identity over time",
    "content": "amount", "combined": "all three",
}
SPLITS = ("within_trace", "loro", "loko")
SPLIT_DISPLAY = {"within_trace": "within-trace", "loro": "LORO", "loko": "LOKO"}
LABELSPACES = ("kernel", "archetype")
# Table 7's primary label space per split (SPEC 6.3); the other is appended below, marked.
PRIMARY_LABELSPACE = {"loko": "archetype", "loro": "kernel", "within_trace": "kernel"}
GF_DEFAULT_GRID = "W8_H4"  # SPEC section 8 item 38


def grid_label(gid: str) -> str:
    """`W8_H4` -> `8 x 4`; `Wall_Hall` -> `whole cell` (Table 5's `grid_point (W x H)`)."""
    if gid is None or gid == "":
        return "--"
    if gid == "Wall_Hall":
        return "whole cell"
    try:
        w, h = gid.split("_")
        return f"{int(w[1:])} x {int(h[1:])}"
    except Exception:
        return str(gid)


# ----------------------------------------------------------------------------------------------
# small I/O helpers
# ----------------------------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path) -> list[dict]:
    """Read a CSV with a header row into a list of dicts (strings; blanks kept as '')."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows: list[dict], columns: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})
    tmp.replace(path)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default)
    tmp.replace(path)
    return path


def _json_default(o):
    try:
        import numpy as np  # noqa
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def result_json(name: str, params: dict, citation: str, payload: dict) -> dict:
    """SPEC section 1: every JSON result file has `schema`, `params`, `citation`, then payload."""
    d = {"schema": f"plan11.{name}.v1", "params": dict(params), "citation": citation}
    d.update(payload)
    return d


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inputs_sha256(paths) -> dict:
    """al-Farabi review 2.8: `inputs_sha256`, a map from each input file read to its sha256
    (absent files map to 'absent')."""
    out = {}
    for p in paths:
        p = Path(p)
        out[str(p)] = sha256_file(p) if p.exists() and p.is_file() else "absent"
    return out


def to_float(x):
    """'' / None / non-numeric -> None; else float."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return None if (isinstance(x, float) and math.isnan(x)) else float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fmt_num(x, digits: int = 3) -> str:
    """A number as text; an undefined number prints `--` (SPEC section 6). Strings that are
    not numbers (verdicts) pass through unchanged."""
    if x is None:
        return "--"
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return "--"
        f = to_float(s)
        if f is None:
            return s
        x = f
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "--"
        if x == int(x) and abs(x) < 1e12:
            return str(int(x))
        return f"{x:.{digits}f}"
    return str(x)


def cell_text(x) -> str:
    """Verdict strings print as themselves; numbers through fmt_num; blanks as ''."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return fmt_num(x)


# ----------------------------------------------------------------------------------------------
# LaTeX and Markdown table fragments (SPEC section 6)
# ----------------------------------------------------------------------------------------------
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}",
}


def latex_escape(s) -> str:
    s = "" if s is None else str(s)
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in s)


def tex_table(columns: list[str], rows: list[dict], *, label: str, columns_comment: str = "",
              note_comment: str = "", wide: bool | None = None, size: str = "footnotesize") -> str:
    """A booktabs `tabular` inside a `table` environment with `\\caption{}` and `\\label{}` left
    empty of prose and a `% columns:` comment line (SPEC section 6). `wide` selects `table*`
    (two-column span) and defaults to true when there are more than seven columns."""
    if wide is None:
        wide = len(columns) > 7
    env = "table*" if wide else "table"
    colspec = "l" * len(columns)
    lines = [f"% columns: {columns_comment or ', '.join(columns)}"]
    if note_comment:
        for ln in note_comment.splitlines():
            lines.append(f"% {ln}")
    lines.append(f"\\begin{{{env}}}[t]")
    lines.append("\\centering")
    lines.append(f"\\{size}")
    lines.append("\\caption{}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(latex_escape(c) for c in columns) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(latex_escape(cell_text(r.get(c))) for c in columns) + " \\\\")
    if not rows:
        lines.append(" & ".join([""] * len(columns)) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def md_table(columns: list[str], rows: list[dict]) -> str:
    def esc(s):
        return str(s).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(esc(c) for c in columns) + " |",
           "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(cell_text(r.get(c))) for c in columns) + " |")
    return "\n".join(out) + "\n"


def write_table(out_dir: Path, name: str, columns: list[str], rows: list[dict], *,
                label: str | None = None, columns_comment: str = "", note_comment: str = "",
                wide: bool | None = None) -> dict:
    """Write `<out_dir>/<name>.csv`, `.md` and `.tex`; return the three paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = write_csv(out_dir / f"{name}.csv", rows, columns)
    md_p = out_dir / f"{name}.md"
    md_p.write_text(md_table(columns, rows), encoding="utf-8")
    tex_p = out_dir / f"{name}.tex"
    tex_p.write_text(tex_table(columns, rows, label=label or f"tab:{name}",
                               columns_comment=columns_comment, note_comment=note_comment,
                               wide=wide), encoding="utf-8")
    return {"csv": csv_p, "md": md_p, "tex": tex_p}


# ----------------------------------------------------------------------------------------------
# reading the other builders' artifacts
# ----------------------------------------------------------------------------------------------
def load_cells(out: Path) -> list[dict]:
    """`<out>/cells.csv` (SPEC 2.7). Raises FileNotFoundError when absent (exit 2 upstream)."""
    p = Path(out) / "cells.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return read_csv(p)


def ok_cells(cells: list[dict]) -> list[dict]:
    return [c for c in cells if (c.get("status") or "ok") == "ok"]


def load_selection(out: Path) -> dict:
    """`gates/selection.json` (SPEC 3.5.7): `{rung: {grid_id, W, H, passes_acceptance,
    selected_by, gates_passed, refusal}}` beside the `schema/params/citation` keys (SPEC
    section 1). Both layouts (rung keys at the top level, or under `selection` / `rungs`) are
    read. Returns {} when the file is absent."""
    p = Path(out) / "gates" / "selection.json"
    if not p.exists():
        return {}
    d = read_json(p)
    for k in ("selection", "rungs"):
        if isinstance(d.get(k), dict):
            d = d[k]
            break
    return {r: v for r, v in d.items() if r in RUNGS and isinstance(v, dict)}


def selected_grid(out: Path, rung: str) -> str | None:
    """The rung's selected grid id, or None when the rung has no selection (al-Farabi review
    2.9(b): tables then write `not run: no selection for <rung>`)."""
    sel = load_selection(out)
    if rung in sel and sel[rung].get("grid_id"):
        return str(sel[rung]["grid_id"])
    return None


def resolution_text(out: Path, rung: str) -> str:
    """Table 7's `resolution (W x H)` cell. al-Farabi review 2.11(c): when `selected_by` is
    `best-feasible` the cell reads `selected: best-feasible`, never as a gated resolution."""
    sel = load_selection(out).get(rung)
    if not sel or not sel.get("grid_id"):
        return not_run(f"no selection for {rung}")
    gid = str(sel["grid_id"])
    txt = grid_label(gid)
    if str(sel.get("selected_by", "")) == "best-feasible" or sel.get("passes_acceptance") in (False, "false", "False", 0):
        txt = f"{txt} (selected: best-feasible)"
    return txt


def gc_rung_verdict(out: Path, rung: str) -> str | None:
    """The `rep = "all"` row of `gates/gc.csv` for the rung (SPEC 3.4.3; al-Farabi review 2.5).
    None when the file or the row is absent."""
    p = Path(out) / "gates" / "gc.csv"
    if not p.exists():
        return None
    for r in read_csv(p):
        if r.get("rung") == rung and str(r.get("rep", "")).strip() == "all":
            return r.get("verdict", "")
    return None


def gf_part1_verdict(out: Path, rung: str, grid: str | None = None) -> str | None:
    """G-F part (i) verdict for the rung (`gates/gf.csv`, part `i`, kernel `idle`; SPEC 3.3.4).
    With `grid` given, the row at that grid id is preferred; otherwise the last row written for
    the rung (move 12's re-run at the selected point comes after move 4's default-grid run).
    None when absent."""
    p = Path(out) / "gates" / "gf.csv"
    if not p.exists():
        return None
    rows = [r for r in read_csv(p) if r.get("rung") == rung
            and str(r.get("part", "")).strip().lower() in ("i", "1", "(i)", "part1", "part (i)")]
    if not rows:
        return None
    if grid:
        for r in reversed(rows):
            if r.get("grid_id") == grid:
                return r.get("verdict", "")
    return rows[-1].get("verdict", "")


def rung_override(out: Path, rung: str, grid: str | None = None) -> str | None:
    """The string that replaces every score cell of a rung: `refused: disconnected lead` when
    G-C refused the rung (al-Farabi review 2.5) or G-F (i)'s void string when the idle reps
    separated (al-Farabi review 2.6). None when neither applies."""
    gc = gc_rung_verdict(out, rung)
    if gc is not None and gc == GC_DISCONNECTED:
        return refused(GC_DISCONNECTED)
    gf = gf_part1_verdict(out, rung, grid)
    if gf is not None and gf == GF_VOID:
        return GF_VOID
    return None


# ---- move 16 (added 2026-10-05, after the run of 2026-09-29; A21, A22; SPEC_epoch2 Part 4 item 30): the twin readers
IDLE_CG_VOID = "void: a held-out idle run is not recognised as idle above chance"
IDLE_CG_FILE = "gates/added/idle_common_ground.csv"
GC_CORRECTED_FILE = "gates/added/gc_corrected.csv"


def gc_corrected_verdict(out: Path, rung: str) -> str | None:
    """The corrected instrument check's `rep = "all"` verdict for `content` and `combined` (move 16,
    gates/added/gc_corrected.csv, `verdict_corrected`); the original gc.csv verdict for the other
    rungs, whose check the correction does not touch. For `content` and `combined` a missing file or
    row reads `not run: ...` naming the corrected file (never gates/gc.csv, which the twin table does
    not read for them); None only when the original reader returns None for the other rungs."""
    if rung not in ("content", "combined"):
        return gc_rung_verdict(out, rung)
    p = Path(out) / GC_CORRECTED_FILE
    if p.exists():
        for r in read_csv(p):
            if r.get("rung") == rung and str(r.get("rep", "")).strip() == "all":
                return r.get("verdict_corrected", "")
    return not_run(f"{GC_CORRECTED_FILE} missing or has no rep = all row for {rung} (move 16)")


def idle_common_ground_row(out: Path, rung: str) -> dict | None:
    """The idle common-ground test's row for the rung (move 16, gates/added/idle_common_ground.csv);
    None when the file or the row is absent."""
    p = Path(out) / IDLE_CG_FILE
    if not p.exists():
        return None
    for r in read_csv(p):
        if r.get("rung") == rung:
            return r
    return None


def idle_common_ground_text(out: Path, rung: str, grid: str | None = None) -> str | None:
    """What the second Table 2 prints in the floor column for the rung: the idle verdict with the idle
    recall and its null p95, or `not run: ...` naming the idle test's file when it has no row for the rung."""
    r = idle_common_ground_row(out, rung)
    if r is None:
        return not_run(f"{IDLE_CG_FILE} missing or has no row for {rung} (move 16)")
    v = str(r.get("idle_verdict", ""))
    if r.get("idle_recall") not in (None, ""):
        return f"{v} (idle recall {fmt_num(to_float(r.get('idle_recall')))}, null p95 {fmt_num(to_float(r.get('idle_null_p95')))})"
    return v


def rung_override_corrected(out: Path, rung: str, grid: str | None = None) -> str | None:
    """The twin of `rung_override` for the second Table 2 (A22; move 16): `refused: disconnected lead`
    when the CORRECTED instrument check refused the rung (content per changed byte; combined inherits),
    and, in place of G-F part (i), the idle common-ground rule: the rung is void only when a held-out
    idle run is not recognised as idle above chance (the idle test's verdict `near_unfalsifiable`:
    idle recall not above the null's 95th percentile). A test whose null stayed below the permutation
    floor voids nothing (its verdict reads `not run: N permutations < 500`, as G-F's does); a rung the
    test has not run for prints `not run: ...` in every score cell, so nothing passes by absence.
    None when neither applies."""
    gc = gc_corrected_verdict(out, rung)
    if gc is not None and gc == GC_DISCONNECTED:
        return refused(GC_DISCONNECTED)
    r = idle_common_ground_row(out, rung)
    if r is None:
        return not_run(f"idle common-ground test not run for {rung} (move 16)")
    v = str(r.get("idle_verdict", ""))
    if v == NEAR_UNFALSIFIABLE:
        return IDLE_CG_VOID
    return None


def split_dir(out: Path, rung: str, gid: str, split: str, labelspace: str, raw: bool = False) -> Path | None:
    """The split stage directory (SPEC 4.5). The raw variant's location is not fixed by SPEC
    4.5 (the path has no raw/norm segment); builder 3 accepts `<split>__<labelspace>__raw/`,
    then `<split>__<labelspace>/raw/`, then `gates/splits_raw/<rung>/<gid>/<split>__<labelspace>/`,
    and finally the shared directory when its `scores.json` `params.normalized` is false
    (listed under 'For the author' as an interface gap for builder 2). Returns None when absent."""
    base = Path(out) / "gates" / "splits" / rung / gid
    if raw:
        cands = (base / f"{split}__{labelspace}__raw", base / f"{split}__{labelspace}" / "raw",
                 Path(out) / "gates" / "splits_raw" / rung / gid / f"{split}__{labelspace}")
        for cand in cands:
            if (cand / "scores.json").exists():
                return cand
        # last resort: the shared directory when its scores.json declares normalized = false
        cand = base / f"{split}__{labelspace}"
        if (cand / "scores.json").exists():
            try:
                if read_json(cand / "scores.json").get("params", {}).get("normalized") is False:
                    return cand
            except Exception:
                pass
        return None
    cand = base / f"{split}__{labelspace}"
    if (cand / "scores.json").exists():
        try:
            if read_json(cand / "scores.json").get("params", {}).get("normalized") is False:
                return None  # the directory holds the raw run, not the norm run
        except Exception:
            pass
        return cand
    return None


def load_scores(d: Path | None) -> dict | None:
    if d is None:
        return None
    p = Path(d) / "scores.json"
    return read_json(p) if p.exists() else None


def effective_scores(sc: dict | None) -> dict | None:
    """SPEC 3.7.2: table rows use the re-run without the quarantined features when a quarantine
    exists (`scores.json["with_quarantine"]` with a non-empty `quarantined_features`). The
    reading is `models.effective_scores` (builder 2, the writer of `scores.json`), the same
    function the comparison gates G-L, G-DIM and G-M read through (CHECK_2.md B1); the copy
    below is the fallback for a process without builder 2's module (it must stay identical)."""
    try:  # pragma: no cover - builder 2's module is present on every real run
        from plan11_encoding_ladder.models import effective_scores as _es  # type: ignore
        return _es(sc)
    except Exception:
        pass
    if sc is None:
        return None
    wq = sc.get("with_quarantine")
    if isinstance(wq, dict) and wq.get("quarantined_features"):
        merged = dict(sc)
        if "accuracy" not in wq:                      # every feature quarantined: no re-run exists
            merged.update({"accuracy": None, "macro_recall": None, "recall_per_class": {}, "recall_per_kernel": {},
                           "majority": None, "b1_g1_rank": None, "null_p95": None,
                           "b1_g1": wq.get("status") or not_run("every feature quarantined")})
        for k, v in wq.items():
            if k != "quarantined_features":
                merged[k] = v
        merged["quarantined_features"] = list(wq["quarantined_features"])
        merged["score_source"] = "with_quarantine (the re-run without the quarantined features; SPEC 3.7.2)"
        return merged
    return sc


def b1g1_block(sc: dict | None) -> str | None:
    """The string that replaces a split's score cells under B1-G1: `near_unfalsifiable` prints
    in every score cell of that split (SPEC 6.2 with al-Farabi review 2.7); a `not run: N
    permutations < 500` verdict prints in the null and rank cells only (SPEC 3.7.1) and is not
    returned here. None when the split passed."""
    if sc is None:
        return None
    v = sc.get("b1_g1")
    if v is None:
        return None
    v = str(v)
    if v == NEAR_UNFALSIFIABLE:
        return NEAR_UNFALSIFIABLE
    return None


def rank_text(sc: dict | None) -> str:
    """SPEC 3.7.1: the rank is reported as `rank r of N`; a smoke run prints its `not run:`."""
    if sc is None:
        return not_run("scores.json missing")
    v = str(sc.get("b1_g1", ""))
    if v.startswith("not run:"):
        return v
    r = sc.get("b1_g1_rank")
    if r is None:
        return "--"
    if isinstance(r, str):
        return r
    n = sc.get("n_perm")
    return f"rank {int(r)} of {int(n)}" if n is not None else f"rank {int(r)}"


def null_text(sc: dict | None) -> str:
    if sc is None:
        return not_run("scores.json missing")
    v = str(sc.get("b1_g1", ""))
    if v.startswith("not run:"):
        return v
    return fmt_num(sc.get("null_p95"))


# ----------------------------------------------------------------------------------------------
# the streaming text reader (piano roll only)
# ----------------------------------------------------------------------------------------------
@contextlib.contextmanager
def open_text(path: str):
    """Yield a text line stream for a plain / .gz / .zst file. Used only by the piano roll
    (SPEC 6.7) when `extract.open_text` (builder 1) is not importable.
    # copied from plan08_b1/b1_extract_hamming.py:open_text, 2026-09-16
    # extended per SPEC (binding rule): zstd binary -> zstandard module -> gzip -> plain.
    """
    path = str(path)
    if path.endswith(".zst"):
        try:
            proc = subprocess.Popen(["zstd", "-dc", "-q", path], stdout=subprocess.PIPE)
        except FileNotFoundError:
            proc = None
        if proc is not None:
            try:
                yield io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
            finally:
                if proc.stdout:
                    proc.stdout.close()
                if proc.wait() != 0:
                    raise RuntimeError(f"zstd -dc failed on {path} (rc={proc.returncode})")
            return
        import zstandard  # type: ignore  # second choice per SPEC
        with open(path, "rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            yield fh
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield fh


def get_open_text():
    """`extract.open_text` (builder 1) when importable, else the copy above."""
    try:  # pragma: no cover
        from plan11_encoding_ladder.extract import open_text as ot  # type: ignore
        return ot
    except Exception:
        return open_text
