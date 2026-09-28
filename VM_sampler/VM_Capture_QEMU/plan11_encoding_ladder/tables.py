#!/usr/bin/env python3
"""tables.py -- Tables 5, 6, 7, 8 and G-V (plus the Table 4 status column, the preconditions
copy, the wAPF-over-APF table and the report manifest) as CSV, Markdown and LaTeX (booktabs).

Builder 3 (report), 2026-09-16. Implements SPEC.md section 6 (6.1 to 6.6) with the corrections
the two SPEC reviews mark "must change before build":
  - al-Farabi 2.5: Table 7 gains a `G-C` column; Table 5 carries G-C's verdict in `refusal` on
    every row of a disconnected rung; every score cell of that rung prints
    `refused: disconnected lead` (the number stays in `scores.json`).
  - al-Farabi 2.6: Table 7 gains a `G-F (i)` column; Table 5's selected row carries it in
    `refusal`; `GF_VOID` prints in the rung's score cells.
  - al-Farabi 2.7: a `near_unfalsifiable` split prints the refusal string in every score cell
    of that split; the score is never printed.
  - al-Farabi 2.9(b): a rung without a selection writes `not run: no selection for <rung>`.
  - al-Farabi 2.11(c): a `best-feasible` selection prints `selected: best-feasible`, never as a
    gated resolution.
  - al-Kindi 7: Table 5 carries `G2 (pairs)` (coverage in pair units) beside the two dt columns.
  - al-Kindi 8: Table 6's G-X cell is APF's; Table 7 carries each rung's own G-X leak verdict.
Every table cell that reads `not run:` names the file that is missing (SPEC section 7).
Nothing here computes a verdict: every verdict string is copied from builder 2's files.

CLI (SPEC 7.1): tables.py --out O [--only NAME,...] [--table8-rung combined]
Names: table5, table5_g3, table6, table7, table8, tablegv, table4_status, preconditions,
wapf_over_apf, manifest. Exit 0 on success (a written refusal is a success), 2 when
`<out>/cells.csv` is missing, 1 on an internal error.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse
import json
import statistics
import traceback
from collections import Counter, defaultdict

from plan11_encoding_ladder._report_common import (  # noqa: E402
    ARCHETYPES, ARCHETYPE_OF, AXIS_OF_RUNG, BITS_PER_PAGE, GC_DISCONNECTED, GF_DEFAULT_GRID,
    GK0_IDLE_MEASURED, GRID_IDS, GX_CONFOUND_TOTAL, GX_LEAK, KERNEL_NAMES, LEVEL_SET_OF, N_PAGES,
    PACKAGE_VERSION, PRIMARY_LABELSPACE, RUNGS, RUNG_DISPLAY, SCHEMA_SOURCE, SPLITS, SPLIT_DISPLAY,
    VERDICTS_SOURCE, b1g1_block, cell_text, effective_scores, fmt_num, gc_rung_verdict,
    gf_part1_verdict, grid_label, inputs_sha256, is_refusal, load_cells, load_scores,
    load_selection, not_run, now_iso, null_text, ok_cells, rank_text, read_csv, read_json,
    refused, resolution_text, result_json, rung_override, selected_grid, sha256_file, split_dir,
    to_float, write_json, write_table,
)

TABLE_NAMES = ("table5", "table5_g3", "table6", "table7", "table8", "tablegv", "table4_status",
               "preconditions", "wapf_over_apf", "table7_comparators", "table_comparators", "manifest")
# ---- build epoch 2 (builder A): the comparator rows of Table 7 and the methods' own per-kernel table.
# The names and declared defaults come from schema.py's COMPARATORS block (SPEC_epoch2.md Part 1.0);
# the mirrored tuple below is the fallback of a process without builder 1's module, as _report_common does.
try:
    from plan11_encoding_ladder import schema as _schema  # noqa: E402
    COMPARATORS = tuple(_schema.COMPARATORS)
    COMPARATOR_DECLARED_DEFAULTS = dict(_schema.COMPARATOR_DECLARED_DEFAULTS)
    COMPARATOR_GRID_ID = _schema.COMPARATOR_GRID_ID
except Exception:  # pragma: no cover - schema.py is present on every real run
    COMPARATORS = (("cmp_savoldi", "Savoldi 2010"), ("cmp_dhodapkar", "Dhodapkar-Smith 2003"), ("cmp_law", "Law 2010"))
    COMPARATOR_DECLARED_DEFAULTS = {"cmp_dhodapkar": ("delta_th", 0.04), "cmp_law": ("X", 4)}
    COMPARATOR_GRID_ID = "Wall_Hall"
TABLE7_VARIANT = "both"        # SPEC_epoch2 Part 4 item 11: "both" (the raw row, as published, then the level-normalized row) | "raw" | "norm"
CITATIONS = {
    "table5": "P2 Sec. 4 Table 5; SPEC 6.1; al-Farabi review 2.5, 2.6, 2.11(c); al-Kindi review 7",
    "table5_g3": "P2 Sec. V 5.1 G3 as a per-kernel flag (option (a)); SPEC 6.1 companion; CR 2.1 item 5",
    "table6": "P2 Sec. 4 Table 6, Sec. VII; SPEC 6.2; al-Farabi review 2.5, 2.6, 2.7; al-Kindi review 8",
    "table7": "P2 Sec. 4 Table 7; SPEC 6.3; al-Farabi review 2.5, 2.6, 2.9(b), 2.11(c); al-Kindi review 8",
    "table8": "P2 Sec. 4 Table 8; SPEC 6.4 (assignments with counts, never a confusion matrix)",
    "tablegv": "P2 Sec. V 5.2 G-V; CR 2.3 item 35; SPEC 6.5",
    "table4_status": "P2 Sec. 4 Table 4; SPEC 6.6",
    "preconditions": "P2 Sec. V 5.1 Plan 02 (C1-C8); SPEC 3.3, 6.6",
    "wapf_over_apf": "P2 Sec. IV rung 0'; K2 move 11; SPEC 6.6",
    "table7_comparators": ("C14 cand. 1: Savoldi, Gubian, Echizen 2010, 'Uncertainty in Live Forensics' (U = mu_dmp +/- sigma_dmp of the "
                           "per-pair changed-page count); C14 cand. 3: Dhodapkar and Smith 2003, 'Comparing Program Phase Detection "
                           "Techniques' (delta = one minus Jaccard, threshold phase detection, stability, mean phase length); C14 cand. 2: "
                           "Law et al. 2010, 'Identifying Volatile Data from Multiple Memory Dumps in Live Forensics' (pages dynamic and "
                           "static for X consecutive dumps); P2 Sec. 4 Table 7; P2E Sec. IV Table 2 (the external comparator row); "
                           "SPEC_epoch2 Part 1.5 (which gates apply) and 1.8 (the rows and cells)"),
    "table_comparators": ("C14 cand. 1 (Savoldi 2010, the per-run form U = mean +/- SD); C14 cand. 3 (Dhodapkar-Smith 2003: boundaries, "
                          "stability, mean phase length at the declared threshold); C14 cand. 2 (Law 2010: pages dynamic and static for X); "
                          "P2 Sec. 4 Table 7; P2E Sec. IV Table 2 (the external comparator row); SPEC_epoch2 Part 1.8 (the per-kernel table)"),
    "manifest": "SPEC 6.8 (report/manifest.json)",
}


def _tables_dir(out: Path) -> Path:
    return Path(out) / "report" / "tables"


def _params_base(out: Path, extra: dict | None = None) -> dict:
    p = {"out": str(out), "package_version": PACKAGE_VERSION, "schema_source": SCHEMA_SOURCE,
         "verdicts_source": VERDICTS_SOURCE, "written_at": now_iso()}
    if extra:
        p.update(extra)
    return p


def _write_params(out: Path, name: str, params: dict, payload: dict | None = None) -> Path:
    """Every table also writes `<name>.json` with `schema`, `params`, `citation` (SPEC section 1)."""
    d = result_json(f"tables.{name}", params, CITATIONS[name], payload or {})
    return write_json(_tables_dir(out) / f"{name}.json", d)


def _kernels_present(cells: list[dict]) -> list[str]:
    present = {c["kernel"] for c in ok_cells(cells) if c.get("role") == "kernel"}
    ordered = [k for k in KERNEL_NAMES if k in present]
    ordered += sorted(k for k in present if k not in KERNEL_NAMES)
    return ordered


def _archetype_measured(out: Path, cells: list[dict]) -> dict:
    """Per kernel: `archetype_measured` from `gates/gk0.csv` when it exists (a relabelled kernel
    reads IDLE), else `archetype_predicted` from `cells.csv` (SPEC 3.3.3, 4.1)."""
    pred = {}
    for c in ok_cells(cells):
        if c.get("role") == "kernel":
            pred.setdefault(c["kernel"], c.get("archetype_predicted") or ARCHETYPE_OF.get(c["kernel"], "unknown"))
    meas = dict(pred)
    p = Path(out) / "gates" / "gk0.csv"
    if p.exists():
        for r in read_csv(p):
            k = r.get("kernel")
            if k in meas and r.get("archetype_measured"):
                meas[k] = r["archetype_measured"]
            elif k in meas and r.get("verdict") == GK0_IDLE_MEASURED:
                meas[k] = "IDLE"
    return meas


# ==============================================================================================
# Table 5
# ==============================================================================================
TABLE5_COLUMNS = ["encoding", "axis", "grid_point (W x H)", "n_windows (median)", "G1",
                  "G2 (0.500 s)", "G2 (0.644 s)", "G2 (pairs)", "G4",
                  "G5 (n non-overlapping windows)", "G-ORD", "selected", "refusal"]


def table5(out: Path) -> dict:
    """Table 5, gate verdicts per rung at each grid point (P2 Sec. 4 Table 5; SPEC 6.1).
    From `gates/table5_grid.csv` (one row per (rung, grid_id), SPEC 3.5.7) and
    `gates/selection.json`. 65 rows: five rungs by thirteen declared grid points, every point
    kept (P2 Sec. V, al-Farabi's condition). The selected point's row carries `selected`; a
    `best-feasible` selection reads `selected: best-feasible` (al-Farabi review 2.11(c)).
    `refusal` carries the row's own refusal, G-C's verdict on every row of a disconnected rung
    (al-Farabi review 2.5), and G-F (i)'s verdict on the selected row (al-Farabi review 2.6).
    `G2 (pairs)` is al-Kindi review item 7 (coverage in pair units, `G2_pairs`, `coverage_pairs`).
    A grid point missing from builder 2's file prints `not run: gates/table5_grid.csv has no
    row for <rung>/<grid_id>` in its gate cells."""
    src = Path(out) / "gates" / "table5_grid.csv"
    sel = load_selection(out)
    rows_by = {}
    if src.exists():
        for r in read_csv(src):
            rows_by[(r.get("rung"), r.get("grid_id"))] = r
    rows = []
    for rung in RUNGS:
        gc = gc_rung_verdict(out, rung)
        sel_gid = sel.get(rung, {}).get("grid_id")
        sel_by = str(sel.get(rung, {}).get("selected_by", ""))
        gf = gf_part1_verdict(out, rung, sel_gid) if sel_gid else None
        for gid in GRID_IDS:
            r = rows_by.get((rung, gid))
            row = {"encoding": RUNG_DISPLAY[rung], "axis": AXIS_OF_RUNG[rung],
                   "grid_point (W x H)": grid_label(gid)}
            if r is None:
                miss = not_run(f"gates/table5_grid.csv has no row for {rung}/{gid}"
                               if src.exists() else "gates/table5_grid.csv missing")
                for c in ("n_windows (median)", "G1", "G2 (0.500 s)", "G2 (0.644 s)", "G2 (pairs)",
                          "G4", "G5 (n non-overlapping windows)", "G-ORD"):
                    row[c] = miss if c not in ("n_windows (median)",) else "--"
                row["selected"] = ""
                refusal_parts = [miss]
            else:
                if r.get("axis"):
                    row["axis"] = r["axis"]
                row["n_windows (median)"] = fmt_num(r.get("n_windows_median"))
                row["G1"] = r.get("G1", "")
                row["G2 (0.500 s)"] = r.get("G2_0500", "")
                row["G2 (0.644 s)"] = r.get("G2_0644", "")
                g2p = r.get("G2_pairs", "")
                cov = r.get("coverage_pairs", "")
                if g2p == "" and cov == "":
                    row["G2 (pairs)"] = not_run("G2_pairs not in gates/table5_grid.csv (al-Kindi review 7)")
                else:
                    row["G2 (pairs)"] = f"{g2p} ({fmt_num(cov)})" if cov != "" else g2p
                row["G4"] = r.get("G4", "")
                g5 = r.get("G5", "")
                nno = r.get("n_windows_nonoverlap_median", "")
                row["G5 (n non-overlapping windows)"] = f"{g5} ({fmt_num(nno)})" if nno != "" else g5
                row["G-ORD"] = r.get("GORD", "")
                is_sel = (str(r.get("selected", "")).strip().lower() in ("selected", "true", "1")) or (gid == sel_gid)
                if is_sel:
                    by = str(r.get("selected_by", "")) or sel_by
                    row["selected"] = "selected: best-feasible" if by == "best-feasible" else "selected"
                else:
                    row["selected"] = ""
                refusal_parts = [r.get("refusal", "")] if r.get("refusal") else []
                if is_sel and gf is not None:
                    refusal_parts.append(f"G-F (i): {gf}")
            if gc is not None and gc == GC_DISCONNECTED and gc not in refusal_parts:   # CHECK_2 M12: the grid row already carries it
                refusal_parts.append(f"G-C: {gc}")
            row["refusal"] = "; ".join(p for p in refusal_parts if p)
            rows.append(row)
    paths = write_table(_tables_dir(out), "table5", TABLE5_COLUMNS, rows, label="tab:table5",
                        note_comment="one row per (encoding, grid point); every grid point kept; "
                                     "refusals written, never filled")
    _write_params(out, "table5", _params_base(out, {
        "source": str(src), "n_rows": len(rows), "grid_ids": list(GRID_IDS),
        "inputs_sha256": inputs_sha256([src, Path(out) / "gates" / "selection.json",
                                        Path(out) / "gates" / "gc.csv", Path(out) / "gates" / "gf.csv"])}))
    return paths


TABLE5_G3_COLUMNS = ["encoding", "kernel", "cepstral SNR (dB)", "surrogate p95", "flag", "CV",
                     "CV / shot floor", "cells present"]


def table5_g3(out: Path) -> dict:
    """The Table 5 companion, G3 as a per-kernel signal flag (SPEC 6.1; P2 Sec. V 5.1, Sec. 6
    item 7 option (a); CR 2.1 item 5). From `gates/g3_flags.csv` (one row per cell): per
    (rung, kernel) the median over cells of `ceps_snr_db`, `snr_surrogate_p95`, `cv`, `cv_ratio`
    (SPEC 3.5.3: CV carries no verdict, it is reported beside the shot floor), the kernel flag
    `flag_kernel`, and the count of cells whose `flag_cell` is present over the cell count."""
    src = Path(out) / "gates" / "g3_flags.csv"
    rows = []
    if not src.exists():
        for rung in RUNGS:
            rows.append({"encoding": RUNG_DISPLAY[rung], "kernel": "--",
                         "cepstral SNR (dB)": "--", "surrogate p95": "--",
                         "flag": not_run("gates/g3_flags.csv missing"), "CV": "--",
                         "CV / shot floor": "--", "cells present": "--"})
    else:
        by = defaultdict(list)
        for r in read_csv(src):
            by[(r.get("rung"), r.get("kernel"))].append(r)
        order = [(rg, k) for rg in RUNGS for k in list(KERNEL_NAMES) + ["idle"]]
        keys = [k for k in order if k in by] + sorted(k for k in by if k not in order)
        for rung, kernel in keys:
            rs = by[(rung, kernel)]

            def med(col):
                vals = [to_float(r.get(col)) for r in rs]
                vals = [v for v in vals if v is not None]
                return statistics.median(vals) if vals else None
            flags = [r.get("flag_kernel", "") for r in rs if r.get("flag_kernel")]
            n_present = sum(1 for r in rs if str(r.get("flag_cell", "")).endswith("present"))
            rows.append({"encoding": RUNG_DISPLAY.get(rung, rung), "kernel": kernel,
                         "cepstral SNR (dB)": fmt_num(med("ceps_snr_db")),
                         "surrogate p95": fmt_num(med("snr_surrogate_p95")),
                         "flag": flags[0] if flags else not_run("flag_kernel blank in gates/g3_flags.csv"),
                         "CV": fmt_num(med("cv")), "CV / shot floor": fmt_num(med("cv_ratio")),
                         "cells present": f"{n_present} of {len(rs)}"})
    paths = write_table(_tables_dir(out), "table5_g3", TABLE5_G3_COLUMNS, rows, label="tab:table5_g3",
                        note_comment="G3 is a per-kernel flag off the (W, H) decision; CV has no verdict")
    _write_params(out, "table5_g3", _params_base(out, {"source": str(src), "n_rows": len(rows),
                                                       "inputs_sha256": inputs_sha256([src])}))
    return paths


# ==============================================================================================
# Table 6
# ==============================================================================================
TABLE6_COLUMNS = ["row", "level set", "n", "within-trace raw", "within-trace norm", "LORO raw",
                  "LORO norm", "LOKO raw", "LOKO norm", "null p95 (LOKO norm)", "majority (LOKO)",
                  "rank (LOKO norm)", "G-N status", "G-X"]


def _score_cell(sc: dict | None, override: str | None, missing: str, getter):
    """One score cell: the rung override (G-C / G-F void) first, then the split's B1-G1 block,
    then the number (or `--` when undefined)."""
    if override:
        return override
    if sc is None:
        return missing
    blk = b1g1_block(sc)
    if blk:
        return blk
    return fmt_num(getter(sc))


def _per_kernel(sc: dict, kernel: str):
    d = sc.get("recall_per_kernel") or {}
    return d.get(kernel)


def _per_class(sc: dict, cls: str):
    d = sc.get("recall_per_class") or {}
    return d.get(cls)


def _mean_members(sc: dict, members: list[str]):
    vals = [to_float(_per_kernel(sc, k)) for k in members]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _gn_status(out: Path) -> dict:
    p = Path(out) / "gates" / "gn.csv"
    if not p.exists():
        return {}
    return {r.get("archetype"): r.get("status", "") for r in read_csv(p)}


def _gx_row(out: Path, rung: str) -> dict | None:
    p = Path(out) / "gates" / "gx.csv"
    if not p.exists():
        return None
    for r in read_csv(p):
        if r.get("rung") == rung:
            return r
    return None


def _gx_text(out: Path, rung: str) -> str:
    r = _gx_row(out, rung)
    if r is None:
        return not_run("gates/gx.csv missing or has no row for " + rung)
    txt = r.get("leak_verdict", "")
    if r.get("confound_verdict"):
        txt += f"; {r['confound_verdict']}"
    return txt


def _gx_total_leak(out: Path, rung: str) -> bool:
    r = _gx_row(out, rung)
    return bool(r) and r.get("leak_verdict") == GX_LEAK and r.get("confound_verdict") == GX_CONFOUND_TOTAL


EXCLUDED_KERNEL_TEXT = not_run("excluded by C1-C8 (all_hard_pass false)")


def _excluded_cells(out: Path) -> set:
    """The cells `gates/preconditions.json` lists under `excluded_cells` (SPEC 3.3.1: the cells whose
    `all_hard_pass` is false and which no split stage reads; SPEC_epoch2 B8, CHECK_3 M8). A file
    without the key, or no file at all, excludes nothing."""
    p = Path(out) / "gates" / "preconditions.json"
    if not p.exists():
        return set()
    try:
        ex = read_json(p).get("excluded_cells")
    except Exception:
        return set()
    return set(ex) if isinstance(ex, list) else set()


def table6(out: Path) -> dict:
    """Table 6, APF alone under the three splits (P2 Sec. 4 Table 6, Sec. VII; SPEC 6.2).
    Rows: one per kernel, then one per archetype (after G-K0's relabelling, `gates/gk0.csv`),
    then `all`. From `gates/splits/apf/<selected grid>/<split>__<labelspace>[__raw]/scores.json`
    (SPEC 4.5): kernel rows print `recall_per_kernel` in kernel space (within-trace, LORO) and
    under LOKO the fraction of the kernel's cells assigned to its archetype (`recall_per_kernel`
    of `loko__archetype`); archetype rows print `recall_per_class` under LOKO and the mean of the
    member kernels' recalls under the other two; `all` prints `accuracy`. `null p95`, `majority`
    and `rank` are LOKO's on every row (P2 Sec. VII: null and majority on every row). `G-N
    status` from `gates/gn.csv`; `G-X` is APF's leak verdict on the `all` row (al-Kindi review
    8). `level set` marks the level-matched sets (`A` floyd/histogram/nbody, `B` fft/gemm; P2
    Sec. 2). A `near_unfalsifiable` split prints the string in every score cell of that split
    (al-Farabi review 2.7); a disconnected or void rung prints its string in every score cell
    (al-Farabi review 2.5, 2.6); a smoke run's `not run: N permutations < 500` prints in the
    null and rank cells (SPEC 3.7.1). Table rows use the B1-G3 re-run when a quarantine exists
    (SPEC 3.7.2). Admissibility (SPEC_epoch2 B8; CHECK_3 M8; SPEC 3.3.1): `n` counts the admissible
    cells (those not in `gates/preconditions.json` `excluded_cells`); a kernel whose every cell is
    excluded prints `not run: excluded by C1-C8 (all_hard_pass false)` in every score cell, is not
    counted in `n` and is not a member of its archetype row; a `preconditions.json` without the key,
    or no file, excludes nothing."""
    cells = load_cells(out)
    kernels = _kernels_present(cells)
    excluded = _excluded_cells(out)
    n_cells = Counter(c["kernel"] for c in ok_cells(cells) if c.get("role") == "kernel" and c["cell_id"] not in excluded)
    fully_excluded = [k for k in kernels if n_cells.get(k, 0) == 0]
    meas = _archetype_measured(out, cells)
    gn = _gn_status(out)
    gid = selected_grid(out, "apf")
    rows = []
    if gid is None:
        rows.append({"row": "all", "level set": "", "n": sum(n_cells.values()),
                     **{c: not_run("no selection for apf") for c in TABLE6_COLUMNS[3:]}})
        paths = write_table(_tables_dir(out), "table6", TABLE6_COLUMNS, rows, label="tab:table6")
        _write_params(out, "table6", _params_base(out, {"grid_id": None, "n_rows": 1}))
        return paths
    override = rung_override(out, "apf", gid)

    def sc_of(split, ls, raw):
        return effective_scores(load_scores(split_dir(out, "apf", gid, split, ls, raw=raw)))

    def miss(split, ls, raw):
        suf = "__raw" if raw else ""
        return not_run(f"gates/splits/apf/{gid}/{split}__{ls}{suf}/scores.json missing")
    S = {
        "within-trace raw": (sc_of("within_trace", "kernel", True), miss("within_trace", "kernel", True), "within_trace"),
        "within-trace norm": (sc_of("within_trace", "kernel", False), miss("within_trace", "kernel", False), "within_trace"),
        "LORO raw": (sc_of("loro", "kernel", True), miss("loro", "kernel", True), "loro"),
        "LORO norm": (sc_of("loro", "kernel", False), miss("loro", "kernel", False), "loro"),
        "LOKO raw": (sc_of("loko", "archetype", True), miss("loko", "archetype", True), "loko"),
        "LOKO norm": (sc_of("loko", "archetype", False), miss("loko", "archetype", False), "loko"),
    }
    loko_norm = S["LOKO norm"][0]
    null_cell = override or null_text(loko_norm) if loko_norm or override else S["LOKO norm"][1]
    rank_cell = override or rank_text(loko_norm) if loko_norm or override else S["LOKO norm"][1]
    maj_cell = (override or (fmt_num(loko_norm.get("majority")) if loko_norm else S["LOKO norm"][1]))
    gx_all = _gx_text(out, "apf")
    if _gx_total_leak(out, "apf"):
        gx_all += "; LOKO archetype headline " + refused("campaign leak with total confound")
    # kernel rows
    for k in kernels:
        row = {"row": k, "level set": LEVEL_SET_OF.get(k, ""), "n": n_cells.get(k, 0)}
        for col, (sc, m, split) in S.items():
            row[col] = EXCLUDED_KERNEL_TEXT if k in fully_excluded else _score_cell(sc, override, m, lambda s, k=k: _per_kernel(s, k))
        row["null p95 (LOKO norm)"] = null_cell
        row["majority (LOKO)"] = maj_cell
        row["rank (LOKO norm)"] = rank_cell
        row["G-N status"] = gn.get(meas.get(k), "" if gn else not_run("gates/gn.csv missing"))
        row["G-X"] = ""
        rows.append(row)
    # archetype rows
    members = defaultdict(list)
    for k in kernels:
        if k in fully_excluded:
            continue                                    # SPEC_epoch2 B8: no admissible cell, no membership
        members[meas.get(k, ARCHETYPE_OF.get(k, "unknown"))].append(k)
    for a in list(ARCHETYPES) + sorted(x for x in members if x not in ARCHETYPES):
        if a not in members:
            continue
        row = {"row": a, "level set": "", "n": len(members[a])}
        for col, (sc, m, split) in S.items():
            if split == "loko":
                row[col] = _score_cell(sc, override, m, lambda s, a=a: _per_class(s, a))
            else:
                row[col] = _score_cell(sc, override, m, lambda s, mem=members[a]: _mean_members(s, mem))
        row["null p95 (LOKO norm)"] = null_cell
        row["majority (LOKO)"] = maj_cell
        row["rank (LOKO norm)"] = rank_cell
        row["G-N status"] = gn.get(a, "" if gn else not_run("gates/gn.csv missing"))
        row["G-X"] = ""
        rows.append(row)
    row = {"row": "all", "level set": "", "n": sum(n_cells.values())}
    for col, (sc, m, split) in S.items():
        row[col] = _score_cell(sc, override, m, lambda s: s.get("accuracy"))
    row["null p95 (LOKO norm)"] = null_cell
    row["majority (LOKO)"] = maj_cell
    row["rank (LOKO norm)"] = rank_cell
    row["G-N status"] = ""
    row["G-X"] = gx_all
    rows.append(row)
    paths = write_table(_tables_dir(out), "table6", TABLE6_COLUMNS, rows, label="tab:table6",
                        note_comment=f"APF at grid point {gid} ({resolution_text(out, 'apf')}); "
                                     "kernel rows: per-kernel recall; archetype rows: per-class recall under LOKO; "
                                     "all: accuracy")
    _write_params(out, "table6", _params_base(out, {
        "grid_id": gid, "n_rows": len(rows), "rung_override": override,
        "excluded_cells": sorted(excluded), "fully_excluded_kernels": fully_excluded,
        "relabelled_kernels": [k for k in kernels if meas.get(k) == "IDLE"],
        "inputs_sha256": inputs_sha256([Path(out) / "cells.csv", Path(out) / "gates" / "gk0.csv",
                                        Path(out) / "gates" / "gn.csv", Path(out) / "gates" / "gx.csv"])}))
    return paths


# ==============================================================================================
# Table 7
# ==============================================================================================
TABLE7_COLUMNS = ["rung", "resolution (W x H)", "feature count", "split", "label space", "accuracy",
                  "macro recall (headline rows)", "null p95", "rank", "majority", "G-C", "G-F (i)",
                  "G-L", "G-DIM", "G-M vs APF", "G-X"]


def _gate_csv(out: Path, name: str, rung: str) -> tuple[Path, str]:
    """(path, its name relative to `<out>`): a gate file of the comparator rows lives under
    `gates/comparators/<name>` when `rung` starts with `cmp_` (SPEC_epoch2 Part 1.6), else under
    `gates/<name>` as before; the three helpers below read through this and behave exactly as before
    for the rungs (SPEC_epoch2 Part 1.8)."""
    rel = f"gates/{'comparators/' if str(rung).startswith('cmp_') else ''}{name}"
    return Path(out) / rel, rel


def _gl_text(out: Path, rung: str) -> str:
    p, rel = _gate_csv(out, "gl.csv", rung)
    if not p.exists():
        return not_run(f"{rel} missing")
    parts = []
    for r in read_csv(p):
        if r.get("rung") != rung and not (rung == "apf" and r.get("part", "").strip().lower() in ("ii", "2", "(ii)")):
            continue
        part = str(r.get("part", "")).strip()
        parts.append(f"({part}) {r.get('verdict', '')}" if part else r.get("verdict", ""))
    return "; ".join(parts) if parts else not_run(f"{rel} has no row for {rung}")


def _gdim_text(out: Path, rung: str) -> str:
    p, rel = _gate_csv(out, "gdim.csv", rung)
    if not p.exists():
        return not_run(f"{rel} missing")
    aliases = {rung, rung.replace(" (matched)", "_matched"), rung.replace(" ", "_")}
    for r in read_csv(p):
        if r.get("rung") in aliases:
            txt = r.get("status", "")
            d, dm = r.get("d", ""), r.get("d_matched", "")
            if d != "":
                txt += f" (d = {d}"
                if dm not in ("", None) and str(dm) != str(d):
                    txt += f", matched {dm}, {r.get('method', '')}"
                txt += ")"
            return txt
    return not_run(f"{rel} has no row for {rung}")


def _gm_text(out: Path, rung: str, split: str, rung_b: str = "apf") -> str:
    """`"beats (diff 0.17 > spread 0.04; 8 up, 0 down)"` (SPEC 6.3) from `gates/gm.csv` (or
    `gates/comparators/gm.csv` for a comparator row, whose raw variant is matched against
    `rung_b = "apf__raw"`, SPEC_epoch2 Part 1.8); a comparator row whose verdict is a `not run:` /
    `not applicable:` string prints that string alone (there is no margin to print)."""
    if rung == "apf":
        return "--"
    p, rel = _gate_csv(out, "gm.csv", rung)
    if not p.exists():
        return not_run(f"{rel} missing")
    aliases = {rung, rung.replace(" (matched)", "_matched"), rung.replace(" ", "_")}
    for r in read_csv(p):
        if r.get("split") == split and r.get("rung_a") in aliases and r.get("rung_b") == rung_b:
            verdict = r.get("verdict", "")
            if str(rung).startswith("cmp_") and str(verdict).startswith("not "):
                return verdict
            diff, spread = to_float(r.get("diff")), to_float(r.get("spread"))
            rel_ = "--"
            if diff is not None and spread is not None:
                rel_ = ">" if diff > spread else "<="
            return (f"{verdict} (diff {fmt_num(diff)} {rel_} spread {fmt_num(spread)}; "
                    f"{r.get('improving', '--')} up, {r.get('worsening', '--')} down)")
    return not_run(f"{rel} has no ({split}, {rung}, {rung_b}) row")


def _matched_scores(out: Path, gid: str | None, split: str, ls: str) -> tuple[dict | None, str]:
    """The `combined (matched)` row's scores (SPEC 3.7.7; SPEC_epoch2 3.5.1; E1 sec. 4 M4). Since
    build epoch 2 `gates_comparison.gate_gdim` writes the matched run for every Table 7 (split,
    label space) under `gates/splits_matched/combined/<gid>/<split>__<ls>/`, so that path is read
    first; then the epoch-1 alternatives (`gates/splits/combined_matched/<gid>/...`, then
    `gates/gdim.json` `matched[<split>__<ls>]`); otherwise the row prints `not run:` naming the
    epoch-2 path. Read through `effective_scores`, so the re-run after a B1-G3 quarantine is the
    row's score as for every rung."""
    if gid is None:
        return None, not_run("no selection for combined")
    d = None
    primary = Path(out) / "gates" / "splits_matched" / "combined" / gid / f"{split}__{ls}"
    if (primary / "scores.json").exists():
        d = primary
    if d is None:
        d = split_dir(out, "combined_matched", gid, split, ls)
    if d is not None:
        return effective_scores(load_scores(d)), ""
    p = Path(out) / "gates" / "gdim.json"
    if p.exists():
        j = read_json(p)
        m = (j.get("matched") or {}).get(f"{split}__{ls}")
        if isinstance(m, dict):
            return m, ""
    return None, not_run(f"gates/splits_matched/combined/{gid}/{split}__{ls}/scores.json missing (gdim, move 12)")


def _feature_count_text(sc: dict | None):
    """CHECK_3 M5 (SPEC_epoch2 Part 1.8 and 2 B5; al-Farabi review 5.6): the `feature count` cell
    prints `scores.json["feature_count_used"]` (the count the fold actually trained on after a
    declared reduction, written by builder B's item B5) when that key is present and non-null, else
    `feature_count`; applies to the `not applicable` branch and the normal branch alike."""
    if sc is None:
        return None
    v = sc.get("feature_count_used")
    return v if v is not None else sc.get("feature_count")


def table7(out: Path) -> dict:
    """Table 7, rungs compared at gated resolution (P2 Sec. 4 Table 7; SPEC 6.3). Rows `apf,
    wapf, persist, content, combined, combined (matched)`; one row per (rung, split) in the
    split's primary label space (archetype for LOKO, kernel for LORO and within-trace), then the
    archetype-space rows of LORO and within-trace appended below, marked in `label space`.
    Columns: resolution from `gates/selection.json` (al-Farabi review 2.11(c) for
    best-feasible), `feature_count` from `scores.json`, accuracy, `macro_recall` (over G-N
    headline rows in archetype space, SPEC 4.2), null p95, rank, majority (all from
    `scores.json`, SPEC 4.5), `G-C` (`gates/gc.csv` rep = all; al-Farabi review 2.5), `G-F (i)`
    (`gates/gf.csv`; al-Farabi review 2.6), `G-L` (`gates/gl.csv`, SPEC 3.7.4), `G-DIM`
    (`gates/gdim.csv`, SPEC 3.7.7), `G-M vs APF` (`gates/gm.csv`, SPEC 3.7.8, as text with the
    margin), `G-X` (the rung's own leak verdict; al-Kindi review 8; a LOKO headline under a
    total-confound leak reads `refused: campaign leak with total confound`, SPEC 3.7.6)."""
    rows = []
    plan = [(r, s, PRIMARY_LABELSPACE[s]) for r in list(RUNGS) + ["combined (matched)"] for s in ("loko", "loro", "within_trace")]
    plan += [(r, s, "archetype") for r in list(RUNGS) + ["combined (matched)"] for s in ("loro", "within_trace")]
    sel_cache = {}
    for rung, split, ls in plan:
        base_rung = "combined" if rung == "combined (matched)" else rung
        gid = sel_cache.setdefault(base_rung, selected_grid(out, base_rung))
        row = {"rung": rung, "split": SPLIT_DISPLAY[split], "label space": ls,
               "resolution (W x H)": resolution_text(out, base_rung)}
        if gid is None:
            m = not_run(f"no selection for {base_rung}")
            for c in ("feature count", "accuracy", "macro recall (headline rows)", "null p95", "rank", "majority"):
                row[c] = m
            row["G-C"] = gc_rung_verdict(out, base_rung) or not_run("gates/gc.csv missing or has no row for " + base_rung)
            row["G-F (i)"] = gf_part1_verdict(out, base_rung) or not_run("gates/gf.csv missing or has no part (i) row for " + base_rung)
            row["G-L"] = _gl_text(out, base_rung)
            row["G-DIM"] = _gdim_text(out, rung)
            row["G-M vs APF"] = _gm_text(out, rung, split)
            row["G-X"] = _gx_text(out, base_rung)
            rows.append(row)
            continue
        override = rung_override(out, base_rung, gid)
        if rung == "combined (matched)":
            sc, missing = _matched_scores(out, gid, split, ls)
        else:
            d = split_dir(out, rung, gid, split, ls)
            sc = effective_scores(load_scores(d))
            missing = not_run(f"gates/splits/{rung}/{gid}/{split}__{ls}/scores.json missing")
        if sc is not None and str(sc.get("accuracy", "")).startswith("not applicable"):
            # e.g. within-trace at the whole-cell point: `not applicable: one window per cell`
            na = str(sc.get("accuracy"))
            for c in ("accuracy", "macro recall (headline rows)", "null p95", "rank", "majority"):
                row[c] = na
            row["feature count"] = fmt_num(_feature_count_text(sc))
        else:
            row["feature count"] = fmt_num(_feature_count_text(sc)) if sc else missing
            row["accuracy"] = _score_cell(sc, override, missing, lambda s: s.get("accuracy"))
            row["macro recall (headline rows)"] = _score_cell(sc, override, missing, lambda s: s.get("macro_recall"))
            row["null p95"] = override or (null_text(sc) if sc else missing)
            row["rank"] = override or (rank_text(sc) if sc else missing)
            row["majority"] = override or (fmt_num(sc.get("majority")) if sc else missing)
        if split == "loko" and ls == "archetype" and _gx_total_leak(out, base_rung):
            row["accuracy"] = refused("campaign leak with total confound")
            row["macro recall (headline rows)"] = refused("campaign leak with total confound")
        row["G-C"] = gc_rung_verdict(out, base_rung) or not_run("gates/gc.csv missing or has no row for " + base_rung)
        row["G-F (i)"] = gf_part1_verdict(out, base_rung, gid) or not_run("gates/gf.csv missing or has no part (i) row for " + base_rung)
        row["G-L"] = _gl_text(out, base_rung)
        row["G-DIM"] = _gdim_text(out, rung)
        row["G-M vs APF"] = _gm_text(out, rung, split)
        row["G-X"] = _gx_text(out, base_rung)
        rows.append(row)
    paths = write_table(_tables_dir(out), "table7", TABLE7_COLUMNS, rows, label="tab:table7",
                        note_comment="one row per (rung, split); LOKO in archetype space, LORO and within-trace "
                                     "in kernel space; archetype-space rows for LORO and within-trace appended below")
    _write_params(out, "table7", _params_base(out, {
        "grid_ids": sel_cache, "n_rows": len(rows),
        "inputs_sha256": inputs_sha256([Path(out) / "gates" / f for f in
                                        ("selection.json", "gc.csv", "gf.csv", "gl.csv", "gdim.csv", "gm.csv", "gx.csv")])}))
    return paths


# ==============================================================================================
# Table 7, the comparator rows, and the comparators' own per-kernel table (build epoch 2, builder A)
# ==============================================================================================
CMP_RESOLUTION_TEXT = "whole cell (per cell, by definition)"
CMP_GC_TEXT = "not applicable: comparator, not a lead of the ladder (G-C calibrates the ladder's leads against the gemm pulse)"
CMP_GF1_TEXT = "not applicable: one row per cell (G-F (i) is the within-trace window design)"
CMP_GL_RAW_TEXT = "level-inclusive (as published)"
CMP_PARAMS_FILE = {"cmp_savoldi": "savoldi.params.json", "cmp_dhodapkar": "dhodapkar.params.json", "cmp_law": "law.params.json"}
CMP_PER_KERNEL_FILE = {"cmp_savoldi": "savoldi_per_kernel.csv", "cmp_dhodapkar": "dhodapkar_per_kernel.csv", "cmp_law": "law_per_kernel.csv"}


def _num_text(v) -> str:
    """A declared default as text: `0.04`, `4` (ten significant digits, never a fixed decimal count)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else format(f, ".10g")


def _cmp_params(out: Path, name: str) -> dict | None:
    p = Path(out) / "gates" / "comparators" / CMP_PARAMS_FILE[name]
    if not p.exists():
        return None
    try:
        return read_json(p).get("params") or {}
    except Exception:
        return None


def _cmp_default_text(out: Path, name: str) -> tuple[str, str]:
    """(the parameter's value as text, its suffix): `declared default` when the value that ran equals
    the declared one of `schema.COMPARATOR_DECLARED_DEFAULTS` (AA 2026-09-17 for delta_th; SPEC_epoch2
    Part 4 item 6 for X), else `default set by --delta-th-default` / `--x-default` (al-Farabi review
    section 2 (a): the record follows the value); the declared value when the params file is absent
    (the comparator has not run: every score cell of the row says so)."""
    pname, declared = COMPARATOR_DECLARED_DEFAULTS[name]
    params = _cmp_params(out, name)
    key = "default" if name == "cmp_dhodapkar" else "x_default"
    flag = "--delta-th-default" if name == "cmp_dhodapkar" else "--x-default"
    if params is None or params.get(key) is None:
        return _num_text(declared), "declared default"
    v = params.get(key)
    try:
        same = abs(float(v) - float(declared)) <= 1e-9
    except (TypeError, ValueError):
        same = False
    return _num_text(v), ("declared default" if same else f"default set by {flag}")


def _cmp_rung_text(out: Path, name: str, normalized: bool) -> str:
    """The `rung` cell: the display name, the variant and the declared default (SPEC_epoch2 Part 1.8)."""
    disp = dict(COMPARATORS)[name]
    var = "level-normalized" if normalized else "as published"
    if name == "cmp_savoldi":
        return f"{disp} ({var}; U = mean +/- SD of K)"
    v, suffix = _cmp_default_text(out, name)
    pname = COMPARATOR_DECLARED_DEFAULTS[name][0]
    return f"{disp} ({var}; {pname} = {v}, {suffix})"


def _cmp_variants() -> list[bool]:
    if TABLE7_VARIANT == "both":
        return [False, True]
    if TABLE7_VARIANT == "raw":
        return [False]
    if TABLE7_VARIANT == "norm":
        return [True]
    raise ValueError(f"TABLE7_VARIANT must be both, raw or norm, got {TABLE7_VARIANT!r}")


def table7_comparators(out: Path) -> dict:
    """The comparator rows of Table 7 (`report/tables/table7_comparators.{csv,md,tex}`; the same
    columns as Table 7; SPEC_epoch2 Part 1.8): for each comparator in `schema.COMPARATORS` order
    (C14 cand. 1 Savoldi 2010, cand. 3 Dhodapkar-Smith 2003, cand. 2 Law 2010; P2 Sec. 0 Baseline;
    P2E Sec. 7), for each variant of `TABLE7_VARIANT` (raw = as published, then level-normalized),
    the three primary rows (LOKO archetype, LORO kernel, within-trace kernel) then the appended
    archetype-space rows of LORO and within-trace: 3 x 2 x 5 = 30 rows under `both`.
    Cells: `resolution` the fixed whole-cell string; feature count (`feature_count_used` when
    present, CHECK_3 M5), accuracy, macro recall, null p95, rank, majority from
    `gates/splits/cmp_<name>/Wall_Hall/<split>__<ls>[__raw]/scores.json` through `effective_scores`
    with the `b1g1_block`, `rank_text` and `null_text` rules unchanged (a missing file prints
    `not run: <path> missing (move 14)`; a `not applicable:` accuracy prints in every score cell);
    `G-C` and `G-F (i)` the fixed strings of SPEC_epoch2 Part 1.5 (a comparator is a published
    reduction, not a lead of the ladder; one row per cell); `G-L` from `gates/comparators/gl.csv`
    for the norm row and `level-inclusive (as published)` for the raw row; `G-DIM` from
    `gates/comparators/gdim.csv` by `cmp_<name>` / `cmp_<name>__raw`; `G-M vs APF` from
    `gates/comparators/gm.csv` with `rung_b` `apf` (norm) or `apf__raw` (raw); `G-X` the comparator's
    row of `gates/gx.csv` on both rows (G-X runs on the norm features); the total-confound refusal of
    SPEC 3.7.6 on the LOKO row as in `table7`. Nothing here computes a verdict."""
    rows = []
    gid = COMPARATOR_GRID_ID
    for name, _disp in COMPARATORS:
        for normalized in _cmp_variants():
            plan = [(s, PRIMARY_LABELSPACE[s]) for s in ("loko", "loro", "within_trace")] + [("loro", "archetype"), ("within_trace", "archetype")]
            for split, ls in plan:
                rung_key = name if normalized else f"{name}__raw"
                row = {"rung": _cmp_rung_text(out, name, normalized), "split": SPLIT_DISPLAY[split], "label space": ls,
                       "resolution (W x H)": CMP_RESOLUTION_TEXT}
                d = split_dir(out, name, gid, split, ls, raw=not normalized)
                sc = effective_scores(load_scores(d))
                missing = not_run(f"gates/splits/{name}/{gid}/{split}__{ls}{'' if normalized else '__raw'}/scores.json missing (move 14)")
                if sc is not None and str(sc.get("accuracy", "")).startswith("not applicable"):
                    na = str(sc.get("accuracy"))
                    for c in ("accuracy", "macro recall (headline rows)", "null p95", "rank", "majority"):
                        row[c] = na
                    row["feature count"] = fmt_num(_feature_count_text(sc))
                else:
                    row["feature count"] = fmt_num(_feature_count_text(sc)) if sc else missing
                    row["accuracy"] = _score_cell(sc, None, missing, lambda s: s.get("accuracy"))
                    row["macro recall (headline rows)"] = _score_cell(sc, None, missing, lambda s: s.get("macro_recall"))
                    row["null p95"] = null_text(sc) if sc else missing
                    row["rank"] = rank_text(sc) if sc else missing
                    row["majority"] = fmt_num(sc.get("majority")) if sc else missing
                if split == "loko" and ls == "archetype" and _gx_total_leak(out, name):
                    row["accuracy"] = refused("campaign leak with total confound")
                    row["macro recall (headline rows)"] = refused("campaign leak with total confound")
                row["G-C"] = CMP_GC_TEXT
                row["G-F (i)"] = CMP_GF1_TEXT
                row["G-L"] = _gl_text(out, name) if normalized else CMP_GL_RAW_TEXT
                row["G-DIM"] = _gdim_text(out, rung_key)
                row["G-M vs APF"] = _gm_text(out, rung_key, split, rung_b="apf" if normalized else "apf__raw")
                row["G-X"] = _gx_text(out, name)
                rows.append(row)
    paths = write_table(_tables_dir(out), "table7_comparators", TABLE7_COLUMNS, rows, label="tab:table7_comparators",
                        note_comment="comparator rows of Table 7; append below tables/table7.tex; one row per (comparator, variant, "
                                     "split) in the split's primary label space, then the archetype-space rows of LORO and within-trace")
    _write_params(out, "table7_comparators", _params_base(out, {
        "grid_id": gid, "table7_variant": TABLE7_VARIANT, "n_rows": len(rows),
        "declared_defaults": {k: list(v) for k, v in COMPARATOR_DECLARED_DEFAULTS.items()},
        "not_applied": {"G-C": CMP_GC_TEXT, "G-F (i)": CMP_GF1_TEXT},
        "inputs_sha256": inputs_sha256([Path(out) / "gates" / "comparators" / f for f in ("gl.csv", "gdim.csv", "gm.csv", "verdicts.csv")]
                                       + [Path(out) / "gates" / "gx.csv"])}))
    return paths


def table_comparators_columns(delta_default: str = "0.04", x_default: str = "4") -> list[str]:
    """The columns of the per-kernel comparator table with the declared defaults in the headers."""
    return ["kernel", "n cells", "Savoldi U (median cell)", "Savoldi mean K (median)", "Savoldi SD K (median)",
            f"D-S boundaries (median, delta_th = {delta_default})", "D-S stability (median)", "D-S mean phase length (median)",
            f"Law dynamic-for-X (median, X = {x_default})", "Law static-for-X fraction (median)", "Law dynamic ever (median)"]


TABLE_COMPARATORS_COLUMNS = table_comparators_columns(_num_text(COMPARATOR_DECLARED_DEFAULTS["cmp_dhodapkar"][1]),
                                                      _num_text(COMPARATOR_DECLARED_DEFAULTS["cmp_law"][1]))


def table_comparators(out: Path) -> dict:
    """The methods' own per-kernel numbers, the "per-run form" of C14 (`report/tables/table_comparators.*`;
    SPEC_epoch2 Part 1.8): one row per kernel in `schema.KERNELS` order then `idle`, from
    `gates/comparators/{savoldi,dhodapkar,law}_per_kernel.csv` (medians over admissible cells;
    Savoldi's `U_text_median` is the median cell's U, C14 cand. 1's "U = 65.5% +/- 0.15%" form;
    Dhodapkar-Smith at the declared threshold, C14 cand. 3; Law at the declared X, C14 cand. 2).
    A missing file prints `not run: gates/comparators/<file> missing (move 14)` in its cells."""
    d_default, _ = _cmp_default_text(out, "cmp_dhodapkar")
    x_default, _ = _cmp_default_text(out, "cmp_law")
    columns = table_comparators_columns(d_default, x_default)
    src = {}
    for name, fname in CMP_PER_KERNEL_FILE.items():
        p = Path(out) / "gates" / "comparators" / fname
        src[name] = ({r.get("kernel"): r for r in read_csv(p)} if p.exists() else None,
                     not_run(f"gates/comparators/{fname} missing (move 14)"))
    kernels = list(KERNEL_NAMES)
    extra = set()
    for tbl, _ in src.values():
        if tbl:
            extra |= set(tbl) - set(kernels) - {"idle"}
    kernels += sorted(extra) + ["idle"]
    cells = load_cells(out)
    n_by = Counter(("idle" if c.get("role") == "idle" else c.get("kernel")) for c in ok_cells(cells))
    rows = []

    def cell(name: str, k: str, col: str) -> str:
        tbl, miss = src[name]
        if tbl is None:
            return miss
        r = tbl.get(k)
        if r is None:
            return not_run(f"gates/comparators/{CMP_PER_KERNEL_FILE[name]} has no row for {k}")
        return cell_text(r.get(col, ""))

    for k in kernels:
        if k == "idle" and n_by.get("idle", 0) == 0 and not any(tbl and "idle" in tbl for tbl, _ in src.values()):
            continue
        n_cells = next((tbl[k].get("n_cells") for tbl, _ in src.values() if tbl and k in tbl and tbl[k].get("n_cells")), None)
        rows.append({
            "kernel": k, "n cells": n_cells if n_cells is not None else fmt_num(n_by.get(k, 0)),
            columns[2]: cell("cmp_savoldi", k, "U_text_median"), columns[3]: cell("cmp_savoldi", k, "K_mean_median"),
            columns[4]: cell("cmp_savoldi", k, "K_sd_median"), columns[5]: cell("cmp_dhodapkar", k, "n_boundaries_median"),
            columns[6]: cell("cmp_dhodapkar", k, "stability_median"), columns[7]: cell("cmp_dhodapkar", k, "mean_phase_length_median"),
            columns[8]: cell("cmp_law", k, "dyn_mean_median"), columns[9]: cell("cmp_law", k, "sta_frac_mean_median"),
            columns[10]: cell("cmp_law", k, "dyn_ever_median")})
    paths = write_table(_tables_dir(out), "table_comparators", columns, rows, label="tab:table_comparators",
                        note_comment=f"the comparators' own per-kernel numbers (medians over admissible cells); the two declared defaults "
                                     f"are delta_th = {d_default} (Dhodapkar-Smith) and X = {x_default} (Law); every grid point is on disk in "
                                     f"gates/comparators/dhodapkar_sweep.csv and law_sweep.csv")
    _write_params(out, "table_comparators", _params_base(out, {
        "n_rows": len(rows), "delta_th_default": d_default, "x_default": x_default,
        "inputs_sha256": inputs_sha256([Path(out) / "gates" / "comparators" / f for f in CMP_PER_KERNEL_FILE.values()])}))
    return paths


# ==============================================================================================
# Table 8
# ==============================================================================================
def _loko_assignments(out: Path, rung: str, gid: str) -> tuple[dict, str]:
    """Per kernel: the majority archetype over the kernel's cells in
    `gates/splits/<rung>/<gid>/loko__archetype/predictions.csv` (SPEC 6.4); when B1-G3
    quarantined a feature the re-run's `predictions_with_quarantine.csv` beside it is read
    instead (SPEC 3.7.2 'Table rows use the re-run'; CHECK_2.md B1). Returns
    ({kernel: archetype}, error) with error '' on success."""
    d = split_dir(out, rung, gid, "loko", "archetype")
    if d is None or not (d / "predictions.csv").exists():
        return {}, not_run(f"gates/splits/{rung}/{gid}/loko__archetype/predictions.csv missing")
    pred_file = d / "predictions_with_quarantine.csv"
    if not pred_file.exists():
        pred_file = d / "predictions.csv"
    votes = defaultdict(Counter)
    for r in read_csv(pred_file):
        k, yp = r.get("kernel"), r.get("y_pred", "")
        if not k or not yp or yp.startswith("not run"):
            continue
        votes[k][yp] += 1
    out_map = {}
    for k, c in votes.items():
        top = max(c.items(), key=lambda kv: (kv[1], kv[0]))  # ties by count then class name
        out_map[k] = top[0]
    return out_map, ""


def _cluster_counts(out: Path, rung: str, cells: list[dict], algo: str = "kmeans") -> tuple[dict, int | None, str]:
    """From `gates/clustering.json` (SPEC 4.4): per predicted archetype the count of its cells
    per cluster label. Builder 2 writes `{"k": k, "cells": [cell_id, ...], "per_algo": {algo:
    {"labels": [...], "ari": ..., "nmi": ..., "cluster_by_predicted_archetype": {archetype:
    {"c0": n, ...}}}}}` (models.py run_clustering; CHECK_1.md B5) and that layout is read first:
    `per_algo[algo].labels` aligned with `cells`, else `cluster_by_predicted_archetype` as the
    count matrix. Fallback: the keys `labels` ({cell_id: label} or a list aligned with
    `cell_id`), nested by rung and/or algorithm when the file is so laid out; else the
    `count_matrix` key as given."""
    p = Path(out) / "gates" / "clustering.json"
    if not p.exists():
        return {}, None, not_run("gates/clustering.json missing")
    j = read_json(p)
    if isinstance(j.get("status"), str) and j["status"].startswith(("not run", "not applicable", "refused")):
        return {}, None, j["status"]        # builder 2's no-selection file carries its refusal
    node = j
    pa = j.get("per_algo", {}).get(algo) if isinstance(j.get("per_algo"), dict) else None
    if isinstance(pa, dict):
        node = {**pa, "cell_id": j.get("cells"), "k": j.get("k")}
        if "count_matrix" not in node:
            node["count_matrix"] = pa.get("cluster_by_predicted_archetype")
    else:
        for key in (rung, algo):
            if isinstance(node, dict) and isinstance(node.get(key), dict):
                node = node[key]
    labels = node.get("labels") if isinstance(node, dict) else None
    k = node.get("k") if isinstance(node, dict) else None
    if k is None:
        kp = Path(out) / "gates" / "clustering.csv"
        if kp.exists():
            for r in read_csv(kp):
                if r.get("rung") == rung and r.get("algo") == algo:
                    k = int(float(r["k"])) if r.get("k") else None
                    break
    pred = {c["cell_id"]: c.get("archetype_predicted", "") for c in ok_cells(cells) if c.get("role") == "kernel"}
    counts = defaultdict(Counter)
    if isinstance(labels, dict):
        for cid, lab in labels.items():
            if cid in pred:
                counts[pred[cid]][int(lab)] += 1
    elif isinstance(labels, list) and isinstance(node.get("cell_id"), list):
        for cid, lab in zip(node["cell_id"], labels):
            if cid in pred:
                counts[pred[cid]][int(lab)] += 1
    elif isinstance(node.get("count_matrix"), dict):
        for a, row in node["count_matrix"].items():
            for lab, n in row.items():
                counts[a][int(str(lab).lstrip("c"))] += int(n)
    else:
        return {}, k, not_run("gates/clustering.json has no labels or count_matrix")
    return counts, k, ""


def table8(out: Path, *, table8_rung: str = "combined") -> dict:
    """Table 8, store-predicted against state-measured (P2 Sec. 4 Table 8; SPEC 6.4). Rows: the
    predicted archetypes with their kernel counts (`IDLE (0 predicted)`, `WORKING-SET (6)`, ...
    from `cells.csv`); columns: `IDLE (measured)` (the kernels relabelled by G-K0,
    `gates/gk0.csv`), then the four archetypes as the LOKO assignment column (the majority
    archetype over the kernel's cells in the `table8_rung`'s `predictions.csv` at its selected
    point; SPEC section 8 item 32, default `combined`), each cell `n (kernel, kernel)`; then
    `clusters (k = ...)` as `c0:n c1:n ...` from `gates/clustering.json` (the primary
    algorithm, KMeans; SPEC 4.4); then `physical reason`, an empty text column for the author.
    Read as assignments with counts, never as a confusion matrix in the statistical sense."""
    cells = load_cells(out)
    kernels = _kernels_present(cells)
    pred = {}
    for c in ok_cells(cells):
        if c.get("role") == "kernel":
            pred.setdefault(c["kernel"], c.get("archetype_predicted") or ARCHETYPE_OF.get(c["kernel"], "unknown"))
    meas = _archetype_measured(out, cells)
    relabelled = [k for k in kernels if meas.get(k) == "IDLE"]
    gid = selected_grid(out, table8_rung)
    err = ""
    assign = {}
    if gid is None:
        err = not_run(f"no selection for {table8_rung}")
    else:
        assign, err = _loko_assignments(out, table8_rung, gid)
    override = rung_override(out, table8_rung, gid) if gid else None
    counts, k, cerr = _cluster_counts(out, table8_rung, cells)
    measured_cols = ["IDLE (measured)"] + [a for a in ARCHETYPES if a != "IDLE"]
    clusters_col = f"clusters (k = {k if k is not None else '--'})"
    columns = ["predicted \\ measured"] + measured_cols + [clusters_col, "physical reason"]
    rows = []
    n_pred = Counter(pred[k] for k in kernels)
    for a in ARCHETYPES:
        row = {"predicted \\ measured": f"{a} ({n_pred.get(a, 0)} predicted)" if a == "IDLE" else f"{a} ({n_pred.get(a, 0)})"}
        members = [k for k in kernels if pred[k] == a]
        for col in measured_cols:
            target = "IDLE" if col == "IDLE (measured)" else col
            if col == "IDLE (measured)":
                hits = [k for k in members if k in relabelled]
            elif err or override:
                row[col] = override or err
                continue
            else:
                hits = [k for k in members if k not in relabelled and assign.get(k) == target]
            row[col] = f"{len(hits)} ({', '.join(hits)})" if hits else "0"
        if cerr:
            row[clusters_col] = cerr
        else:
            c = counts.get(a, Counter())
            row[clusters_col] = " ".join(f"c{lab}:{n}" for lab, n in sorted(c.items())) if c else "0"
        row["physical reason"] = ""
        rows.append(row)
    paths = write_table(_tables_dir(out), "table8", columns, rows, label="tab:table8",
                        note_comment="assignments with counts (LOKO majority per kernel from the "
                                     f"{table8_rung} rung at {gid}); never a confusion matrix in the statistical sense",
                        wide=True)
    pred_dir = split_dir(out, table8_rung, gid, "loko", "archetype") if gid else None
    _write_params(out, "table8", _params_base(out, {
        "table8_rung": table8_rung, "grid_id": gid, "relabelled_kernels": relabelled,
        "assignments": assign, "cluster_k": k, "rung_override": override,
        "predictions_file": ("predictions_with_quarantine.csv" if pred_dir is not None and (pred_dir / "predictions_with_quarantine.csv").exists()
                             else "predictions.csv") + " (the re-run's file first when B1-G3 quarantined a feature; SPEC 3.7.2)",
        "inputs_sha256": inputs_sha256([Path(out) / "cells.csv", Path(out) / "gates" / "gk0.csv",
                                        Path(out) / "gates" / "clustering.json"])}))
    return paths


# ==============================================================================================
# G-V table
# ==============================================================================================
TABLEGV_COLUMNS = ["rung", "feature", "L0 (within-kernel)", "L2 (within-archetype)",
                   "L3 (between-archetype)", "L0/L3", "verdict"]


def tablegv(out: Path) -> dict:
    """The G-V variance table (P2 Sec. V 5.2 G-V; CR 2.3 item 35; SPEC 6.5). From
    `gates/gv.csv` (`rung, feature, L0, L2, L3, L0_over_L3`) and `gates/gv_summary.csv`
    (`rung, n_features, n_features_L0_gt_L3, verdict`), sorted by rung (ladder order) then
    `L0/L3` descending, with one summary row per rung carrying `estimable` or `LOKO not
    estimable` (the only verdict G-V has)."""
    src, ssrc = Path(out) / "gates" / "gv.csv", Path(out) / "gates" / "gv_summary.csv"
    rows = []
    if not src.exists():
        for rung in RUNGS:
            rows.append({"rung": rung, "feature": "summary", "L0 (within-kernel)": "--",
                         "L2 (within-archetype)": "--", "L3 (between-archetype)": "--", "L0/L3": "--",
                         "verdict": not_run("gates/gv.csv missing")})
    else:
        by = defaultdict(list)
        for r in read_csv(src):
            by[r.get("rung")].append(r)
        summ = {}
        if ssrc.exists():
            summ = {r.get("rung"): r for r in read_csv(ssrc)}
        order = [r for r in RUNGS if r in by] + sorted(r for r in by if r not in RUNGS)
        for rung in order:
            rs = sorted(by[rung], key=lambda r: -(to_float(r.get("L0_over_L3")) if to_float(r.get("L0_over_L3")) is not None else -1e300))
            for r in rs:
                rows.append({"rung": rung, "feature": r.get("feature", ""),
                             "L0 (within-kernel)": fmt_num(r.get("L0"), 4),
                             "L2 (within-archetype)": fmt_num(r.get("L2"), 4),
                             "L3 (between-archetype)": fmt_num(r.get("L3"), 4),
                             "L0/L3": fmt_num(r.get("L0_over_L3"), 3), "verdict": ""})
            s = summ.get(rung)
            rows.append({"rung": rung,
                         "feature": (f"summary ({s.get('n_features', '--')} features, "
                                     f"{s.get('n_features_L0_gt_L3', '--')} with L0 > L3)") if s else "summary",
                         "L0 (within-kernel)": "--", "L2 (within-archetype)": "--",
                         "L3 (between-archetype)": "--", "L0/L3": "--",
                         "verdict": s.get("verdict", "") if s else not_run("gates/gv_summary.csv missing or has no row for " + rung)})
    paths = write_table(_tables_dir(out), "tablegv", TABLEGV_COLUMNS, rows, label="tab:tablegv",
                        note_comment="population variances on level-normalized features at the selected point; "
                                     "L2 over archetypes with at least two kernels")
    _write_params(out, "tablegv", _params_base(out, {"source": str(src), "n_rows": len(rows),
                                                     "inputs_sha256": inputs_sha256([src, ssrc])}))
    return paths


# ==============================================================================================
# Table 4 status, preconditions copy, wAPF over APF
# ==============================================================================================
TABLE4_COLUMNS = ["plan", "fixes", "gates", "scope", "APF at 500 ms", "wAPF", "persistence",
                  "content-change", "combined"]


def _sel_text(out: Path, rung: str) -> str:
    sel = load_selection(out).get(rung)
    if not sel or not sel.get("grid_id"):
        return not_run(f"no selection for {rung}")
    acc = sel.get("passes_acceptance")
    by = sel.get("selected_by", "")
    return f"{sel['grid_id']} ({'passes acceptance' if acc in (True, 'true', 'True', 1) else 'best-feasible'}; {by})"


def _b1g1_text(out: Path, rung: str) -> str:
    gid = selected_grid(out, rung)
    if gid is None:
        return not_run(f"no selection for {rung}")
    sc = load_scores(split_dir(out, rung, gid, "loko", "archetype"))
    if sc is None:
        return not_run(f"gates/splits/{rung}/{gid}/loko__archetype/scores.json missing")
    return f"B1-G1 (LOKO norm): {sc.get('b1_g1', '--')}"


def _per_rung_from_csv(out: Path, fname: str, rung: str, key: str = "rung", col: str = "verdict",
                       where=None) -> str:
    p = Path(out) / "gates" / fname
    if not p.exists():
        return not_run(f"gates/{fname} missing")
    for r in read_csv(p):
        if r.get(key) == rung and (where is None or where(r)):
            return r.get(col, "")
    return not_run(f"gates/{fname} has no row for {rung}")


def table4_status(out: Path) -> dict:
    """The Table 4 status column (P2 Sec. 4 Table 4; SPEC 6.6): one row per plan with the
    verdict strings the toolkit produced. Plan 02: the count of cells passing `all_hard_pass`
    and C7's status (`gates/preconditions.csv`), and since build epoch 2 the C1 rule in force
    (`gates/preconditions.json` `params.C1_rule_in_force`; SPEC_epoch2 section 4, AD 2026-09-17,
    AA T1); Plan 03: the selection per rung
    (`gates/selection.json`); Plan 04: not in this paper (P2 Sec. 6 item 8); Plan 05: the fixed
    not-applicable strings (P2 Sec. V 5.1); Plan 07 and 09: cited; Plan 08: B1-G1's LOKO
    verdict per rung; the council gates: one row each with the roll-up verdict per rung."""
    pre = Path(out) / "gates" / "preconditions.csv"
    if pre.exists():
        prs = read_csv(pre)
        n_ok = sum(1 for r in prs if str(r.get("all_hard_pass", "")).lower() in ("true", "1", "pass"))
        c7 = Counter(r.get("C7", "") for r in prs)
        p02 = f"{n_ok} of {len(prs)} cells all_hard_pass; C7: " + ", ".join(f"{k} ({v})" for k, v in c7.items())
        # epoch 2 (SPEC_epoch2 section 4; AD 2026-09-17; AA T1 "the runbook records which rule was in force"):
        # the C1 rule in force for the run, from gates/preconditions.json params
        pj = Path(out) / "gates" / "preconditions.json"
        if not pj.exists():
            p02 += "; C1 rule: " + not_run("gates/preconditions.json missing")
        else:
            try:
                rule = (read_json(pj).get("params") or {}).get("C1_rule_in_force")
            except Exception:
                rule = None
            p02 += "; C1 rule: " + (str(rule) if rule else not_run("gates/preconditions.json has no C1_rule_in_force (written before epoch 2)"))
    else:
        p02 = not_run("gates/preconditions.csv missing")
    rows = [
        {"plan": "02", "fixes": "interval; session validity", "gates": "C1-C8 (C1 re-mapped)", "scope": "per dataset",
         "APF at 500 ms": p02, "wAPF": "cited", "persistence": "cited", "content-change": "cited", "combined": "cited"},
        {"plan": "03", "fixes": "window, hop", "gates": "G1-G5 as amended, G-ORD", "scope": "per encoding",
         **{c: _sel_text(out, r) for c, r in zip(TABLE4_COLUMNS[4:], RUNGS)}},
        {"plan": "04", "fixes": "segment count k", "gates": "segmenter gates", "scope": "per encoding, if kept",
         **{c: "not applicable: change-point view not in this paper (decided 2026-09-16)" for c in TABLE4_COLUMNS[4:]}},
        {"plan": "05", "fixes": "throughput levers", "gates": "G-T1, G-T2, G-T3", "scope": "per dataset / G-T2 per encoding",
         "APF at 500 ms": "G-T1, G-T3: not applicable (no lever); G-T2: not applicable, inherited by A1 and A3",
         **{c: "not applicable, inherited by A1 and A3" for c in TABLE4_COLUMNS[5:]}},
        {"plan": "07", "fixes": "scale-up", "gates": "stage 13", "scope": "per dataset",
         **{c: "cited" for c in TABLE4_COLUMNS[4:]}},
        {"plan": "08", "fixes": "analysis validity", "gates": "B1-G1 to G6, restated", "scope": "per encoding",
         **{c: _b1g1_text(out, r) for c, r in zip(TABLE4_COLUMNS[4:], RUNGS)}},
        {"plan": "09", "fixes": "campaign worth", "gates": "Gates 0-3", "scope": "per dataset",
         "APF at 500 ms": "passed (numbers with the author)", **{c: "cited" for c in TABLE4_COLUMNS[5:]}},
    ]
    # council gates, one row each
    gk0 = Path(out) / "gates" / "gk0.csv"
    if gk0.exists():
        rel = [r["kernel"] for r in read_csv(gk0) if r.get("verdict") == GK0_IDLE_MEASURED]
        gk0_txt = f"{len(rel)} relabelled IDLE, measured" + (f" ({', '.join(rel)})" if rel else "")
    else:
        gk0_txt = not_run("gates/gk0.csv missing")
    council = [
        ("G-K0", "per dataset", {c: gk0_txt for c in TABLE4_COLUMNS[4:]}),
        ("G-F (i)", "per rung", {c: (gf_part1_verdict(out, r) or not_run("gates/gf.csv missing or has no part (i) row for " + r)) for c, r in zip(TABLE4_COLUMNS[4:], RUNGS)}),
        ("G-C", "per rung", {c: (gc_rung_verdict(out, r) or not_run("gates/gc.csv missing or has no row for " + r)) for c, r in zip(TABLE4_COLUMNS[4:], RUNGS)}),
        ("G-P", "per kernel", {c: _gp_summary(out) for c in TABLE4_COLUMNS[4:]}),
        ("G-J", "persistence", {**{c: "--" for c in TABLE4_COLUMNS[4:]}, "persistence": _gj_summary(out)}),
        ("G-DEC", "content-change", {**{c: "--" for c in TABLE4_COLUMNS[4:]}, "content-change": _gdec_summary(out)}),
        ("G-L", "per rung", {c: _gl_text(out, r) for c, r in zip(TABLE4_COLUMNS[4:], RUNGS)}),
        ("G-N", "per dataset", {c: _gn_summary(out) for c in TABLE4_COLUMNS[4:]}),
        ("G-X", "per rung", {c: _gx_text(out, r) for c, r in zip(TABLE4_COLUMNS[4:], RUNGS)}),
        ("G-DIM", "per rung", {c: _gdim_text(out, r) for c, r in zip(TABLE4_COLUMNS[4:], RUNGS)}),
        ("G-M", "per rung vs APF (LOKO)", {c: _gm_text(out, r, "loko") for c, r in zip(TABLE4_COLUMNS[4:], RUNGS)}),
        ("G-V", "per rung", {c: _per_rung_from_csv(out, "gv_summary.csv", r) for c, r in zip(TABLE4_COLUMNS[4:], RUNGS)}),
    ]
    for name, scope, cols in council:
        rows.append({"plan": "council", "fixes": "preconditions, calibration, readings, comparisons",
                     "gates": name, "scope": scope, **cols})
    paths = write_table(_tables_dir(out), "table4_status", TABLE4_COLUMNS, rows, label="tab:table4_status")
    _write_params(out, "table4_status", _params_base(out, {"n_rows": len(rows)}))
    return paths


def _gp_summary(out: Path) -> str:
    p = Path(out) / "gates" / "gp.csv"
    if not p.exists():
        return not_run("gates/gp.csv missing")
    per_kernel = {}
    for r in read_csv(p):
        per_kernel.setdefault(r.get("kernel"), Counter())[r.get("verdict_pairs", "")] += 1
    parts = []
    for k, c in per_kernel.items():
        top = c.most_common(1)[0][0]
        parts.append(f"{k}: {top}" + ("" if len(c) == 1 else f" ({dict(c)})"))
    return "; ".join(parts)


def _gj_summary(out: Path) -> str:
    p = Path(out) / "gates" / "gj.csv"
    if not p.exists():
        return not_run("gates/gj.csv missing")
    c = Counter(r.get("mask_verdict", "") for r in read_csv(p))
    return ", ".join(f"{k} ({v} cells)" for k, v in c.items())


def _gdec_summary(out: Path) -> str:
    p = Path(out) / "gates" / "gdec.csv"
    if not p.exists():
        return not_run("gates/gdec.csv missing")
    parts = [f"{r.get('kernel')} ({r.get('role_in_test', '')}): {r.get('verdict', '')}"
             for r in read_csv(p) if str(r.get("cell_id", "")) == "all"]
    return "; ".join(parts) if parts else not_run("gates/gdec.csv has no cell_id = all row")


def _gn_summary(out: Path) -> str:
    p = Path(out) / "gates" / "gn.csv"
    if not p.exists():
        return not_run("gates/gn.csv missing")
    return "; ".join(f"{r.get('archetype')} ({r.get('n_kernels')}): {r.get('status', '')}" for r in read_csv(p))


def preconditions(out: Path) -> dict:
    """`report/tables/preconditions.csv/.md/.tex`, a copy of `gates/preconditions.csv` (SPEC 6.6)."""
    src = Path(out) / "gates" / "preconditions.csv"
    if not src.exists():
        columns = ["cell_id", "status"]
        rows = [{"cell_id": "--", "status": not_run("gates/preconditions.csv missing")}]
    else:
        rows = read_csv(src)
        columns = list(rows[0].keys()) if rows else ["cell_id"]
    paths = write_table(_tables_dir(out), "preconditions", columns, rows, label="tab:preconditions", wide=True)
    _write_params(out, "preconditions", _params_base(out, {"source": str(src), "n_rows": len(rows),
                                                           "inputs_sha256": inputs_sha256([src])}))
    return paths


WAPF_COLUMNS = ["kernel", "n cells", "mean APF", "mean wAPF", "wAPF / APF",
                "mean flipped bits per changed page"]


def wapf_over_apf(out: Path) -> dict:
    """`table_wapf_over_apf` (K2 move 11; P2 Sec. IV rung 0'; SPEC 6.6): per kernel the means
    of the per-cell means of APF (`K / N`) and wAPF (`ham_sum_all / (N x 32768)`) from the
    extracts, their ratio, and the mean flipped bits per changed page (= wAPF / APF x 32768).
    Idle cells are one row `idle`. A kernel whose extracts are missing prints `not run:`. Only
    admissible cells enter the means (SPEC_epoch2 B8; CHECK_3 M8): a cell listed in
    `gates/preconditions.json` `excluded_cells` is left out, and a kernel whose every cell is
    excluded prints `not run: excluded by C1-C8 (all_hard_pass false)` in its four number cells."""
    cells = load_cells(out)
    per_cell = defaultdict(list)
    missing = defaultdict(int)
    excluded = _excluded_cells(out)
    n_excluded = defaultdict(int)
    for c in ok_cells(cells):
        key = "idle" if c.get("role") == "idle" else c["kernel"]
        if c["cell_id"] in excluded:
            n_excluded[key] += 1
            continue
        ex = Path(out) / "extract" / c["cell_id"] / "extract.csv"
        if not ex.exists():
            missing[key] += 1
            continue
        apf, wapf = [], []
        for r in read_csv(ex):
            K = to_float(r.get("K"))
            hs = to_float(r.get("ham_sum_all"))
            if K is None or hs is None:
                continue
            apf.append(K / N_PAGES)
            wapf.append(hs / (N_PAGES * BITS_PER_PAGE))
        if apf:
            per_cell[key].append((statistics.fmean(apf), statistics.fmean(wapf)))
    rows = []
    for k in list(KERNEL_NAMES) + ["idle"]:
        if k not in per_cell and k not in missing and k not in n_excluded:
            continue
        vals = per_cell.get(k, [])
        if not vals and k in n_excluded and k not in missing:
            rows.append({"kernel": k, "n cells": 0, "mean APF": EXCLUDED_KERNEL_TEXT, "mean wAPF": EXCLUDED_KERNEL_TEXT,
                         "wAPF / APF": EXCLUDED_KERNEL_TEXT, "mean flipped bits per changed page": EXCLUDED_KERNEL_TEXT})
            continue
        if not vals:
            rows.append({"kernel": k, "n cells": 0, "mean APF": not_run(f"extract/<{k} cells>/extract.csv missing"),
                         "mean wAPF": "--", "wAPF / APF": "--", "mean flipped bits per changed page": "--"})
            continue
        ma = statistics.fmean(v[0] for v in vals)
        mw = statistics.fmean(v[1] for v in vals)
        ratio = mw / ma if ma else None
        rows.append({"kernel": k, "n cells": len(vals), "mean APF": f"{ma:.6g}", "mean wAPF": f"{mw:.6g}",
                     "wAPF / APF": fmt_num(ratio, 4),
                     "mean flipped bits per changed page": fmt_num(ratio * BITS_PER_PAGE if ratio is not None else None, 1)})
    paths = write_table(_tables_dir(out), "table_wapf_over_apf", WAPF_COLUMNS, rows, label="tab:wapf_over_apf",
                        note_comment="per-kernel means of the per-cell means, from the extracts")
    _write_params(out, "wapf_over_apf", _params_base(out, {"n_rows": len(rows), "N": N_PAGES, "bits_per_page": BITS_PER_PAGE,
                                                            "excluded_cells": sorted(excluded), "n_excluded_per_kernel": dict(n_excluded)}))
    return paths


# ==============================================================================================
# manifest
# ==============================================================================================
def manifest(out: Path) -> Path:
    """`report/manifest.json` (SPEC 6.8): every file under `report/` and `gates/` with its sha256
    and size, the package version, the `driver_state.json` ledger and every `params` block
    collected from the JSON result files."""
    out = Path(out)
    files = {}
    params_blocks = {}
    for sub in ("report", "gates"):
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
    lp = out / "driver_state.json"
    if lp.exists():
        try:
            ledger = read_json(lp)
        except Exception:
            ledger = "unreadable"
    d = result_json("manifest", _params_base(out), CITATIONS["manifest"],
                    {"package_version": PACKAGE_VERSION, "n_files": len(files), "files": files,
                     "params_blocks": params_blocks, "driver_state": ledger})
    return write_json(out / "report" / "manifest.json", d)


# ==============================================================================================
# CLI
# ==============================================================================================
def run(out: Path, only: list[str] | None = None, table8_rung: str = "combined") -> dict:
    names = list(only) if only else list(TABLE_NAMES)
    unknown = [n for n in names if n not in TABLE_NAMES]
    if unknown:
        raise ValueError(f"unknown table name(s): {unknown}; known: {TABLE_NAMES}")
    written = {}
    for n in names:
        if n == "table5":
            written[n] = table5(out)
        elif n == "table5_g3":
            written[n] = table5_g3(out)
        elif n == "table6":
            written[n] = table6(out)
        elif n == "table7":
            written[n] = table7(out)
        elif n == "table8":
            written[n] = table8(out, table8_rung=table8_rung)
        elif n == "tablegv":
            written[n] = tablegv(out)
        elif n == "table4_status":
            written[n] = table4_status(out)
        elif n == "preconditions":
            written[n] = preconditions(out)
        elif n == "wapf_over_apf":
            written[n] = wapf_over_apf(out)
        elif n == "table7_comparators":
            written[n] = table7_comparators(out)
        elif n == "table_comparators":
            written[n] = table_comparators(out)
    if "manifest" in names:
        written["manifest"] = manifest(out)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan11 tables (builder 3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated: " + ",".join(TABLE_NAMES))
    ap.add_argument("--table8-rung", default="combined", choices=list(RUNGS))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not (out / "cells.csv").exists():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr)
        return 2
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None
    try:
        written = run(out, only, a.table8_rung)
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    for n, v in written.items():
        if isinstance(v, dict):
            print(f"{n}: {v['csv']}")
        else:
            print(f"{n}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
