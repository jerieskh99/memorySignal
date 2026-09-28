#!/usr/bin/env python3
"""tables_eusipco.py -- the EUSIPCO paper's two result tables, read from Table 7's rows and the
toolkit's gate files; CSV, Markdown and LaTeX (booktabs). No verdict is computed here: every
cell is copied from a file another module wrote, or is a `not run:` string naming the file
that is missing (SPEC.md section 7).

Builder 2 (eusipco) of build epoch 2, 2026-09-17. Definitions implemented:

  Table 2, reductions compared: `apf_paper/P2E_STRUCTURE.md` section 3.IV ("Table 2: reductions
  compared. Rows: APF, wAPF, content-change, persistence, combined, external comparator.
  Columns: feature count, LOKO accuracy and macro recall over qualifying archetypes, LORO
  accuracy, the null's 95th percentile, the majority baseline, the paired margin against APF.
  Refusals printed as words.") and section 5 ("Table 2 (from Table 7 plus the comparator)");
  the comparator rows are the author's picks in `apf_paper/P2_AUTHOR_ANSWERS.md`, "Decisions of
  2026-09-17" (AA 2026-09-17: Savoldi 2010 and Dhodapkar-Smith 2003 in the EUSIPCO table, Law
  2010 held for the IFIP version), whose definitions are `council/14_hunayn_exact_input_comparators.md`
  candidates 1, 3 and 2. Every number is Table 7's (`report/tables/table7.csv`, `tables.table7`,
  P2 Sec. 4 Table 7); the LOKO/archetype row gives the feature count, accuracy, macro recall,
  null p95, majority and the G-M margin, the LORO/kernel row gives the LORO accuracy.

  Table 3, the level-matched test: P2E section 3.IV ("Table 3: the level-matched test. Rows: the
  2,048-page set (floyd, histogram, nbody) and the 4,096-page pair (fft, gemm). Columns:
  separable under APF (no), under content-change, under persistence, with the pass-period status
  of gemm stated."); P2 Sec. 2 falsifier (2) and Sec. VI (`schema.LEVEL_MATCHED_SETS`); the
  separation count is the envelope rule of `SPEC_review_al_kindi.md` item 5 through
  `gates_calibration.separating_features` (disjoint ranges of the per-cell means of the
  normalized features; no threshold), the LORO kernel recall and the within-set confusion are
  read from the LORO/kernel split stage (`models.run_split_stage`, SPEC 4.5, through
  `models.effective_scores`, SPEC 3.7.2), the alias check from `gates/alias.csv`
  (`gates_calibration.run_alias`, SPEC 3.4.4) and gemm's pass period from `gates/gp.csv`
  (`gates_calibration.gate_gp`, SPEC 3.4.2). Readings, no verdict: "separable: no" is the
  author's sentence, written when the count is zero.

The comparator rows' interface (two layouts, both read; `params["comparator_sources"]` says which):
  (A) `SPEC_epoch2.three_builders_0407.md` section 3.5.3: Table 7 itself carries the comparator
      rows, `rung = "comparator: <display> [<bib key>]"` (regex `COMPARATOR_ROW_RE`).
  (B) `SPEC_epoch2.md` (the two-builder addendum, the file on disk as `SPEC_EPOCH2.md`) Part 1.8:
      the comparator rows live in `report/tables/table7_comparators.csv` with the same sixteen
      columns, `rung = "<display> (as published; ...)"` and `"<display> (level-normalized; ...)"`;
      `--comparator-variant` chooses the row (default `as published`, the method as published;
      Part 4 item 11 of that file prints both in Table 7).
  When neither file has a row for a comparator key, its cells read `not run:` naming both files.

Where a definition leaves a choice, the choice is a parameter with its default here and listed
under "For the author" in BUILD2_eusipco.md: `COMPARATORS_DEFAULT`, `COMPARATOR_VARIANT_DEFAULT`,
`INCLUDE_MATCHED_DEFAULT`, `DS_IDS` (the Dhodapkar-Smith reading's internal id, resolved from what
exists on disk), `TABLE3_FEATURE_VARIANT` (Table 3 reads the `_norm.npz` file, al-Kindi item 5's
"normalized feature"), `TABLE_SIZE`.

CLI (SPEC 7.1 form):
  tables_eusipco.py --out O [--only table2,table3]
                    [--comparators savoldi2010uncertainty,dhodapkar2003comparing] [--include-matched]
                    [--comparator-variant "as published"|level-normalized] [--ds-id dhodapkar_smith|cmp_dhodapkar]
Exit 0 on success (a written `not run:` cell is a success), 2 when Table 2 is requested and
`<out>/report/tables/table7.csv` is missing (its path on stderr), 1 on an internal error.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse
import re
import statistics
import traceback
from collections import Counter

from plan11_encoding_ladder import series as S  # noqa: E402  (builder 2's readers; called, never edited)
from plan11_encoding_ladder._report_common import (  # noqa: E402
    LEVEL_MATCHED_SETS, PACKAGE_VERSION, SPLIT_DISPLAY, cell_text, effective_scores, fmt_num,
    inputs_sha256, latex_escape, load_scores, md_table, not_run, now_iso, read_csv, read_json,
    selected_grid, split_dir, to_float, write_csv, write_json,
)
from plan11_encoding_ladder.gates_calibration import separating_features  # noqa: E402  (al-Kindi item 5)

EPOCH = 2
CITATION_TABLE2 = ("P2E sec. 3.IV Table 2; P2E sec. 5 (Table 2 from Table 7 plus the comparator); "
                   "AA 2026-09-17 (Savoldi 2010 and Dhodapkar-Smith 2003 in the EUSIPCO table, Law 2010 held for IFIP); "
                   "C14 cand. 1, 3, 2 (the comparators' definitions); P2 Sec. 4 Table 7 (every number)")
CITATION_TABLE3 = ("P2E sec. 3.IV Table 3; P2 Sec. 2 falsifier (2), Sec. VI (the level-matched sets); "
                   "SPEC_review_al_kindi.md item 5 (the envelope rule, gates_calibration.separating_features); "
                   "SPEC 4.5 and 3.7.2 (the LORO/kernel split read through effective_scores); "
                   "SPEC 3.4.4 (gates/alias.csv); SPEC 3.4.2 (gates/gp.csv)")

# ----------------------------------------------------------------------------------------------
# Table 2 constants (P2E sec. 3.IV; AA 2026-09-17)
# ----------------------------------------------------------------------------------------------
EUSIPCO_TABLE2_COLUMNS = ["reduction", "feature count", "LOKO accuracy", "LOKO macro recall (headline archetypes)",
                          "LOKO null p95", "LOKO majority", "LORO accuracy", "margin vs APF (LOKO)"]
TABLE2_RUNG_ROWS = (("apf", "APF"), ("wapf", "wAPF"), ("content", "content-change"),
                    ("persist", "persistence"), ("combined", "combined"))   # P2E sec. 3.IV row order
TABLE2_MATCHED_ROW = ("combined (matched)", "combined (matched)")
# The three exact-input comparators (C14 candidates 1, 3, 2), keyed by their p2.bib key. The bib
# key is the row's identity in both layouts; the display is the Table 7 text; the internal ids are
# the feature-file / split-stage directory names of the two addenda ((A) first, (B) second).
COMPARATOR_KEYS = ("savoldi2010uncertainty", "dhodapkar2003comparing", "law2010volatile")
COMPARATOR_DISPLAY = {"savoldi2010uncertainty": "Savoldi 2010", "dhodapkar2003comparing": "Dhodapkar-Smith 2003",
                      "law2010volatile": "Law 2010"}
COMPARATOR_IDS = {"savoldi2010uncertainty": ("savoldi", "cmp_savoldi"),
                  "dhodapkar2003comparing": ("dhodapkar_smith", "cmp_dhodapkar"),
                  "law2010volatile": ("law", "cmp_law")}
COMPARATORS_DEFAULT = ("savoldi2010uncertainty", "dhodapkar2003comparing")   # AA 2026-09-17; Law is the IFIP row
INCLUDE_MATCHED_DEFAULT = False                                              # P2E sec. 3.IV lists six rows
COMPARATOR_ROW_RE = re.compile(r"^comparator: (?P<display>.+) \[(?P<key>[^\]]+)\]$")   # layout (A)
TABLE7_FILE = "table7.csv"
TABLE7_COMPARATORS_FILE = "table7_comparators.csv"                          # layout (B)
COMPARATOR_VARIANTS = ("as published", "level-normalized")
COMPARATOR_VARIANT_DEFAULT = "as published"     # the method as published (SPEC_epoch2.md 1.8: the raw row)
TABLE_SIZE = "scriptsize"                       # eight columns; the skeleton's shells use the same
LOKO, LORO = SPLIT_DISPLAY["loko"], SPLIT_DISPLAY["loro"]

# ----------------------------------------------------------------------------------------------
# Table 3 constants (P2E sec. 3.IV; P2 Sec. 2, Sec. VI)
# ----------------------------------------------------------------------------------------------
TABLE3_READINGS = (("apf", "under APF"), ("content", "under content-change"), ("persist", "under persistence"),
                   ("dhodapkar_smith", "under Dhodapkar-Smith 2003"))
TABLE3_SUBCOLS = ("separating features", "feature count", "LORO kernel recall (set mean)", "within-set confusion")
EUSIPCO_TABLE3_COLUMNS = ["set", "kernels"] + [disp for _, disp in TABLE3_READINGS] + ["alias check (APF)", "gemm pass period"]
EUSIPCO_TABLE3_CSV_COLUMNS = (["set", "kernels"] + [f"{disp}: {sub}" for _, disp in TABLE3_READINGS for sub in TABLE3_SUBCOLS]
                              + ["alias check (APF)", "gemm pass period"])
DS_IDS = COMPARATOR_IDS["dhodapkar2003comparing"]   # ("dhodapkar_smith", "cmp_dhodapkar"): resolved from disk, (A) first
DS_GRID_ID = "Wall_Hall"                            # one vector per cell, by definition (both addenda)
TABLE3_FEATURE_VARIANT = "norm"                     # al-Kindi item 5: the per-cell means of the normalized features
SET_LETTERS = tuple("AB"[i] if i < 2 else str(i) for i in range(len(LEVEL_MATCHED_SETS)))


def _tables_dir(out: Path) -> Path:
    return Path(out) / "report" / "tables"


def _params(out: Path, extra: dict) -> dict:
    p = {"out": str(out), "epoch": EPOCH, "package_version": PACKAGE_VERSION, "written_at": now_iso()}
    p.update(extra)
    return p


# ----------------------------------------------------------------------------------------------
# the cite-aware LaTeX writer
# ----------------------------------------------------------------------------------------------
def tex_table_cite(columns: list[str], rows: list[dict], *, label: str, cite_col: str | None = None,
                   cite_keys: dict | None = None, columns_comment: str = "", note_comment: str = "",
                   wide: bool | None = None, size: str = "footnotesize") -> str:
    """A booktabs `tabular` in a `table` / `table*` environment with `\\caption{}` and `\\label{}`
    empty of prose and a `% columns:` line: exactly the shape of `_report_common.tex_table`
    (modelled on it, 2026-09-17), with one difference: the `cite_col` cell of a row whose text is
    a key of `cite_keys` is emitted as `<escaped display>~\\cite{<bib key>}` (P2E sec. 3.IV: the
    external comparator is a named method in the same table). Every other cell passes through
    `latex_escape`."""
    if wide is None:
        wide = len(columns) > 7
    cite_keys = cite_keys or {}
    env = "table*" if wide else "table"
    lines = [f"% columns: {columns_comment or ', '.join(columns)}"]
    if note_comment:
        for ln in note_comment.splitlines():
            lines.append(f"% {ln}")
    lines += [f"\\begin{{{env}}}[t]", "\\centering", f"\\{size}", "\\caption{}", f"\\label{{{label}}}",
              f"\\begin{{tabular}}{{{'l' * len(columns)}}}", "\\toprule",
              " & ".join(latex_escape(c) for c in columns) + " \\\\", "\\midrule"]
    for r in rows:
        cells = []
        for c in columns:
            txt = cell_text(r.get(c))
            if c == cite_col and txt in cite_keys:
                cells.append(f"{latex_escape(txt)}~\\cite{{{cite_keys[txt]}}}")
            else:
                cells.append(latex_escape(txt))
        lines.append(" & ".join(cells) + " \\\\")
    if not rows:
        lines.append(" & ".join([""] * len(columns)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def _write_three(out: Path, name: str, columns: list[str], rows: list[dict], *, label: str, tex_columns=None,
                 tex_rows=None, cite_col=None, cite_keys=None, note_comment: str = "", wide: bool = True) -> dict:
    """`<out>/report/tables/<name>.csv` (the full columns), `.md` and `.tex` (the display columns
    when they differ, as Table 3's compact cells do)."""
    d = _tables_dir(out)
    d.mkdir(parents=True, exist_ok=True)
    tex_columns = tex_columns or columns
    tex_rows = tex_rows if tex_rows is not None else rows
    csv_p = write_csv(d / f"{name}.csv", rows, columns)
    md_p = d / f"{name}.md"
    md_p.write_text(md_table(tex_columns, tex_rows), encoding="utf-8")
    tex_p = d / f"{name}.tex"
    tex_p.write_text(tex_table_cite(tex_columns, tex_rows, label=label, cite_col=cite_col, cite_keys=cite_keys,
                                    note_comment=note_comment, wide=wide, size=TABLE_SIZE), encoding="utf-8")
    return {"csv": csv_p, "md": md_p, "tex": tex_p}


# ----------------------------------------------------------------------------------------------
# Table 2
# ----------------------------------------------------------------------------------------------
def _pick(rows: list[dict], split: str, labelspace: str) -> dict | None:
    for r in rows:
        if r.get("split") == split and r.get("label space") == labelspace:
            return r
    return None


def comparator_rows(out: Path, key: str, *, variant: str = COMPARATOR_VARIANT_DEFAULT,
                    table7_rows: list[dict] | None = None) -> tuple[list[dict], str, str]:
    """The Table 7 rows of one comparator, by its p2.bib key, from whichever layout exists.
    Layout (A) (`SPEC_epoch2.three_builders_0407.md` 3.5.3): rows of `report/tables/table7.csv`
    whose `rung` matches `comparator: <display> [<key>]`. Layout (B) (`SPEC_epoch2.md` Part 1.8):
    rows of `report/tables/table7_comparators.csv` whose `rung` starts with `<display> (<variant>`
    (`as published` is the raw row, `level-normalized` the norm row). Returns `(rows, source,
    label)`: `source` names the file and the matched `rung` text (recorded in `params`), and is
    `""` with no rows when neither file has the comparator. (A) is searched first."""
    if variant not in COMPARATOR_VARIANTS:
        raise ValueError(f"comparator variant must be one of {COMPARATOR_VARIANTS}, got {variant!r}")
    display = COMPARATOR_DISPLAY.get(key, key)
    if table7_rows is None:
        p = _tables_dir(out) / TABLE7_FILE
        table7_rows = read_csv(p) if p.exists() else []
    found, label = [], ""
    for r in table7_rows:
        m = COMPARATOR_ROW_RE.match(str(r.get("rung", "")))
        if m and m.group("key") == key:
            found.append(r)
            label = str(r.get("rung"))
    if found:
        return found, f"{TABLE7_FILE} ({label})", label
    p = _tables_dir(out) / TABLE7_COMPARATORS_FILE
    if p.exists():
        prefix = f"{display} ({variant}"
        for r in read_csv(p):
            if str(r.get("rung", "")).startswith(prefix):
                found.append(r)
                label = str(r.get("rung"))
        if found:
            return found, f"{TABLE7_COMPARATORS_FILE} ({label})", label
    return [], "", ""


def _table2_row(label: str, rows: list[dict], source_file: str, missing_all: str | None = None) -> dict:
    """One Table 2 row from a label's Table 7 rows: LOKO/archetype gives six cells, LORO/kernel one.
    Every cell is copied verbatim (refusals printed as words; `--` stays `--`)."""
    row = {"reduction": label}
    if missing_all:
        for c in EUSIPCO_TABLE2_COLUMNS[1:]:
            row[c] = missing_all
        return row
    loko = _pick(rows, LOKO, "archetype")
    loro = _pick(rows, LORO, "kernel")
    m_loko = not_run(f"{source_file} has no {LOKO}/archetype row for {label}")
    m_loro = not_run(f"{source_file} has no {LORO}/kernel row for {label}")
    for col, src in (("feature count", "feature count"), ("LOKO accuracy", "accuracy"),
                     ("LOKO macro recall (headline archetypes)", "macro recall (headline rows)"),
                     ("LOKO null p95", "null p95"), ("LOKO majority", "majority"), ("margin vs APF (LOKO)", "G-M vs APF")):
        row[col] = str(loko.get(src, "")) if loko is not None else m_loko
    row["LORO accuracy"] = str(loro.get("accuracy", "")) if loro is not None else m_loro
    return row


def table2(out: Path, *, comparators=COMPARATORS_DEFAULT, include_matched: bool = INCLUDE_MATCHED_DEFAULT,
           comparator_variant: str = COMPARATOR_VARIANT_DEFAULT) -> dict:
    """EUSIPCO Table 2, reductions compared (P2E sec. 3.IV; P2E sec. 5; AA 2026-09-17). Rows in
    order: APF, wAPF, content-change, persistence, combined (Table 7's `rung` values), one row per
    comparator key in `comparators` (default AA 2026-09-17's two; `law2010volatile` is the IFIP
    row), then `combined (matched)` when `include_matched`. Columns `EUSIPCO_TABLE2_COLUMNS`: from
    the label's Table 7 row with `split == LOKO` and `label space == archetype` the feature count,
    accuracy, macro recall (headline rows), null p95, majority and `G-M vs APF`; from the row with
    `split == LORO` and `label space == kernel` the accuracy. Cells are copied verbatim, so a
    refusal prints as words and `--` stays `--`; a missing row prints `not run: <file> has no
    <split>/<label space> row for <label>`. Comparator rows come through `comparator_rows` (either
    layout); the `.tex` prints their `reduction` cell as `<display>~\\cite{<key>}`. Raises
    FileNotFoundError naming `report/tables/table7.csv` when it is missing (the CLI exits 2)."""
    out = Path(out)
    t7p = _tables_dir(out) / TABLE7_FILE
    if not t7p.exists():
        raise FileNotFoundError(str(t7p))
    t7 = read_csv(t7p)
    comparators = tuple(comparators)
    rows, sources, cite_keys = [], {}, {}
    for rung, label in TABLE2_RUNG_ROWS:
        rows.append(_table2_row(label, [r for r in t7 if r.get("rung") == rung], TABLE7_FILE))
    for key in comparators:
        display = COMPARATOR_DISPLAY.get(key, key)
        crow, source, label = comparator_rows(out, key, variant=comparator_variant, table7_rows=t7)
        sources[key] = source or "absent"
        cite_keys[display] = key
        if not crow:
            miss = not_run(f"{TABLE7_FILE} has no comparator row for {key} ({display}; nor "
                           f"{TABLE7_COMPARATORS_FILE}, variant '{comparator_variant}')")
            rows.append(_table2_row(display, [], TABLE7_FILE, missing_all=miss))
        else:
            rows.append(_table2_row(display, crow, source.split(" (", 1)[0]))
    if include_matched:
        rung, label = TABLE2_MATCHED_ROW
        rows.append(_table2_row(label, [r for r in t7 if r.get("rung") == rung], TABLE7_FILE))
    note = ("EUSIPCO Table 2 (P2E sec. 3.IV): rows APF, wAPF, content-change, persistence, combined, then the "
            "comparators of AA 2026-09-17; every cell copied from Table 7 (LOKO/archetype row: feature count, "
            "accuracy, macro recall, null p95, majority, G-M margin; LORO/kernel row: accuracy)")
    for key in comparators:
        note += f"\ncomparator {COMPARATOR_DISPLAY.get(key, key)} [{key}] <- {sources[key]}"
    paths = _write_three(out, "eusipco_table2", EUSIPCO_TABLE2_COLUMNS, rows, label="tab:p2e_table2",
                         cite_col="reduction", cite_keys=cite_keys, note_comment=note, wide=True)
    write_json(_tables_dir(out) / "eusipco_table2.params.json", {
        "schema": "plan11.eusipco_table2.v1",
        "params": _params(out, {"comparators": list(comparators), "comparator_display": {k: COMPARATOR_DISPLAY.get(k, k) for k in comparators},
                                "comparator_variant": comparator_variant, "comparator_sources": sources,
                                "include_matched": bool(include_matched), "columns": EUSIPCO_TABLE2_COLUMNS,
                                "inputs_sha256": inputs_sha256([t7p, _tables_dir(out) / TABLE7_COMPARATORS_FILE])}),
        "citation": CITATION_TABLE2, "n_rows": len(rows), "rows": [r["reduction"] for r in rows]})
    return paths


# ----------------------------------------------------------------------------------------------
# Table 3
# ----------------------------------------------------------------------------------------------
def _resolve_ds_id(out: Path, ds_id: str | None) -> str:
    """The Dhodapkar-Smith reading's directory name: `--ds-id` when given, else the first of
    `DS_IDS` with a whole-cell feature file or split directory on disk, else `DS_IDS[0]`."""
    if ds_id:
        return ds_id
    for cand in DS_IDS:
        if (Path(out) / "features" / cand).exists() or (Path(out) / "gates" / "splits" / cand).exists():
            return cand
    return DS_IDS[0]


def _separation(out: Path, rid: str, gid: str, letter: str, inputs: list) -> tuple:
    """`(k, d)`: the number of distinct features that separate at least one pair of the set under
    the envelope rule (`gates_calibration.separating_features`, al-Kindi item 5) and the feature
    file's width. `not run:` strings name the feature file when it is missing."""
    fp = S.features_path(out, rid, gid, TABLE3_FEATURE_VARIANT == "norm")
    inputs.append(fp)
    if not fp.is_file():
        m = not_run(f"features/{rid}/{gid}_{TABLE3_FEATURE_VARIANT}.npz missing")
        return m, m
    feat = S.load_features(fp)
    d = int(len(feat["feature_names"]))
    seps = separating_features(out, rid, gid)
    k = len({s["feature"] for s in seps if s.get("set") == letter})
    return k, d


def _loro_reading(out: Path, rid: str, gid: str, kernels: tuple, inputs: list) -> tuple:
    """`(r, c, n_used)`: the set mean of `recall_per_kernel` from the LORO/kernel `scores.json`
    (through `effective_scores`, SPEC 3.7.2) and, from the predictions file that `scores.json`
    names (`predictions_with_quarantine.csv` after a B1-G3 quarantine, else `predictions.csv`),
    the fraction of the set's cells with a numeric-space prediction whose `y_pred` is another
    member of the same set. `not run:` strings name the missing file."""
    d = split_dir(out, rid, gid, "loro", "kernel")
    rel = f"gates/splits/{rid}/{gid}/loro__kernel"
    if d is None:
        inputs.append(Path(out) / rel / "scores.json")
        m = not_run(f"{rel}/scores.json missing")
        return m, m, 0
    inputs.append(d / "scores.json")
    sc = effective_scores(load_scores(d))
    acc = sc.get("accuracy") if sc else None
    if isinstance(acc, str) and acc.startswith(("not applicable", "not run", "refused")):
        return acc, acc, 0
    rpk = sc.get("recall_per_kernel") or {}
    vals = [to_float(rpk.get(k)) for k in kernels if k in rpk]
    vals = [v for v in vals if v is not None]
    r = statistics.fmean(vals) if vals else None
    pf = str(sc.get("predictions_file") or "predictions.csv")
    pp = d / pf
    inputs.append(pp)
    if not pp.is_file():
        return r, not_run(f"{rel}/{pf} missing"), len(vals)
    n, wrong = 0, 0
    for p in read_csv(pp):
        if p.get("kernel") not in kernels:
            continue
        yp = str(p.get("y_pred", ""))
        if yp.startswith("not run:") or yp == "":
            continue
        n += 1
        if yp in kernels and yp != str(p.get("y_true", "")):
            wrong += 1
    c = wrong / n if n else None
    return r, c, len(vals)


def _alias_check(out: Path, letter: str, inputs: list) -> str:
    """The distinct `verdict` values of `gates/alias.csv` rows with `kind == table6_feature` whose
    `pair_or_set` starts with the set letter, in order of first appearance, joined by `; `; `--`
    when none (SPEC 3.4.4, `gates_calibration.run_alias`)."""
    p = Path(out) / "gates" / "alias.csv"
    inputs.append(p)
    if not p.exists():
        return not_run("gates/alias.csv missing")
    seen = []
    for r in read_csv(p):
        if r.get("kind") == "table6_feature" and str(r.get("pair_or_set", "")).startswith(letter):
            v = r.get("verdict", "")
            if v not in seen:
                seen.append(v)
    return "; ".join(seen) if seen else "--"


def _majority(values: list[str]) -> str:
    cnt = Counter(values)
    best = max(cnt.values())
    for v in values:            # ties resolve to the first appearance
        if cnt[v] == best:
            return v
    return ""


def _gemm_pass_period(out: Path, inputs: list) -> str:
    """gemm's rows of `gates/gp.csv` (SPEC 3.4.2): the majority `verdict_pairs` and
    `within_pass_verdict`, printed `<verdict_pairs>; within pass: <within_pass_verdict>`."""
    p = Path(out) / "gates" / "gp.csv"
    inputs.append(p)
    if not p.exists():
        return not_run("gates/gp.csv missing")
    rows = [r for r in read_csv(p) if r.get("kernel") == "gemm"]
    if not rows:
        return not_run("gates/gp.csv has no gemm row")
    vp = _majority([str(r.get("verdict_pairs", "")) for r in rows])
    wp = _majority([str(r.get("within_pass_verdict", "")) for r in rows])
    return f"{vp}; within pass: {wp}"


def _is_refusal_text(x) -> bool:
    return isinstance(x, str) and x.startswith(("not run:", "not applicable:", "refused:"))


def compact_cell(k, d, r, c) -> str:
    """The `.md` / `.tex` cell of one reading: `<k>/<d> sep; LORO <r>; conf <c>` with `none` in
    place of `k/d` when k = 0 and a `not run:` string verbatim where an input is missing (repeated
    strings printed once)."""
    parts = []
    if _is_refusal_text(k):
        parts.append(k)
    elif k == 0:
        parts.append("none")
    else:
        parts.append(f"{k}/{d} sep")
    parts.append(r if _is_refusal_text(r) else ("LORO --" if r is None else f"LORO {float(r):.2f}"))
    parts.append(c if _is_refusal_text(c) else ("conf --" if c is None else f"conf {float(c):.2f}"))
    dedup = []
    for p in parts:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    return "; ".join(dedup)


def _num_or_text(x) -> str:
    return x if isinstance(x, str) else fmt_num(x)


def table3(out: Path, *, ds_id: str | None = None) -> dict:
    """EUSIPCO Table 3, the level-matched test (P2E sec. 3.IV; P2 Sec. 2 falsifier (2), Sec. VI).
    One row per level-matched set (`schema.LEVEL_MATCHED_SETS`: A = floyd, histogram, nbody; B =
    fft, gemm). Under each of the four readings `TABLE3_READINGS` (APF, content-change,
    persistence at their selected grid point from `gates/selection.json`; Dhodapkar-Smith 2003 at
    `Wall_Hall`): `k` the separating-feature count (al-Kindi item 5 through
    `gates_calibration.separating_features`), `d` the feature file's width, `r` the set-mean LORO
    kernel recall and `c` the within-set confusion (both from the LORO/kernel split stage through
    `effective_scores`). Then `alias check (APF)` (`gates/alias.csv`, SPEC 3.4.4) and `gemm pass
    period (G-P)` (`gates/gp.csv`, SPEC 3.4.2; filled on the set that holds gemm, `--` on the
    other). The CSV carries the four decomposed columns per reading; the `.md` and `.tex` print
    `compact_cell`. A rung without a selection prints `not run: no selection for <rung>` in its four
    cells; every other missing input prints `not run:` naming the file. Readings, no verdict."""
    out = Path(out)
    rid_ds = _resolve_ds_id(out, ds_id)
    inputs: list = [Path(out) / "gates" / "selection.json"]
    csv_rows, disp_rows, readings, n_used = [], [], {}, {}
    for letter, kernels in zip(SET_LETTERS, LEVEL_MATCHED_SETS):
        kernels = tuple(kernels)
        crow = {"set": letter, "kernels": ", ".join(kernels)}
        drow = {"set": letter, "kernels": ", ".join(kernels)}
        for R, disp in TABLE3_READINGS:
            rid = rid_ds if R == "dhodapkar_smith" else R
            gid = DS_GRID_ID if R == "dhodapkar_smith" else selected_grid(out, R)
            if gid is None:
                m = not_run(f"no selection for {R}")
                k = d = r = c = m
                readings[R] = {"id": rid, "grid_id": None}
            else:
                k, d = _separation(out, rid, gid, letter, inputs)
                r, c, nk = _loro_reading(out, rid, gid, kernels, inputs)
                n_used[f"{letter}:{R}"] = nk
                readings[R] = {"id": rid, "grid_id": gid}
            for sub, val in zip(TABLE3_SUBCOLS, (k, d, r, c)):
                crow[f"{disp}: {sub}"] = _num_or_text(val)
            drow[disp] = compact_cell(k, d, r, c)
        crow["alias check (APF)"] = drow["alias check (APF)"] = _alias_check(out, letter, inputs)
        crow["gemm pass period"] = drow["gemm pass period"] = (_gemm_pass_period(out, inputs)
                                                                           if "gemm" in kernels else "--")
        csv_rows.append(crow)
        disp_rows.append(drow)
    note = ("EUSIPCO Table 3 (P2E sec. 3.IV), the level-matched test: per reading `k/d sep` (features separating a pair "
            "of the set under the envelope rule, al-Kindi item 5; `none` when k = 0), `LORO r` (set-mean LORO kernel "
            "recall), `conf c` (within-set confusion); readings, no verdict; the CSV carries the four columns per reading")
    paths = _write_three(out, "eusipco_table3", EUSIPCO_TABLE3_CSV_COLUMNS, csv_rows, label="tab:p2e_table3",
                         tex_columns=EUSIPCO_TABLE3_COLUMNS, tex_rows=disp_rows, note_comment=note, wide=True)
    ds_record = _ds_threshold_record(out, rid_ds, inputs)
    seen, uniq = set(), []
    for p in inputs:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    write_json(_tables_dir(out) / "eusipco_table3.params.json", {
        "schema": "plan11.eusipco_table3.v1",
        "params": _params(out, {"readings": readings, "ds_id": rid_ds, "ds_id_candidates": list(DS_IDS),
                                "ds_threshold_record": ds_record, "feature_variant": TABLE3_FEATURE_VARIANT,
                                "feature_variant_reason": "SPEC_review_al_kindi.md item 5: the per-cell means of the normalized features",
                                "sets": {l: list(k) for l, k in zip(SET_LETTERS, LEVEL_MATCHED_SETS)},
                                "n_kernels_used": n_used, "csv_columns": EUSIPCO_TABLE3_CSV_COLUMNS,
                                "display_columns": EUSIPCO_TABLE3_COLUMNS, "inputs_sha256": inputs_sha256(uniq)}),
        "citation": CITATION_TABLE3, "n_rows": len(csv_rows)})
    return paths


def _ds_threshold_record(out: Path, rid: str, inputs: list) -> dict:
    """The Dhodapkar-Smith threshold as the comparator module recorded it, copied verbatim so the
    record follows the value that ran (SPEC_epoch2_review_al_farabi.md section 2 (a)): the
    `params` of `gates/comparators/dhodapkar.params.json` (layout (B)) or the comparator's entry
    in `gates/comparators/registry.json` (layout (A)); `{"source": "absent"}` when neither exists."""
    pb = Path(out) / "gates" / "comparators" / "dhodapkar.params.json"
    pa = Path(out) / "gates" / "comparators" / "registry.json"
    inputs.extend([pb, pa])
    try:
        if pb.exists():
            j = read_json(pb)
            pr = j.get("params", j) if isinstance(j, dict) else {}
            return {"source": "gates/comparators/dhodapkar.params.json",
                    "default": pr.get("default"), "grid": pr.get("grid"), "default_source": pr.get("default_source"),
                    "grid_source": pr.get("grid_source")}
        if pa.exists():
            j = read_json(pa)
            ent = (j.get("comparators") or {}).get(rid) or {}
            pr = ent.get("params") or {}
            return {"source": "gates/comparators/registry.json", "delta_th": pr.get("delta_th"),
                    "delta_th_sweep": pr.get("delta_th_sweep"), "delta_th_source": pr.get("delta_th_source")}
    except Exception as e:  # a malformed record is reported, never guessed
        return {"source": "unreadable", "error": str(e)}
    return {"source": "absent"}


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
TABLE_NAMES = ("table2", "table3")


def run(out: Path, only: list[str] | None = None, *, comparators=COMPARATORS_DEFAULT,
        include_matched: bool = INCLUDE_MATCHED_DEFAULT, comparator_variant: str = COMPARATOR_VARIANT_DEFAULT,
        ds_id: str | None = None) -> dict:
    """Write the named tables (default both) with the given choices; returns `{name: paths}`
    as `tables.run` does (SPEC 6; the driver's `tables --only` form)."""
    names = list(only) if only else list(TABLE_NAMES)
    unknown = [n for n in names if n not in TABLE_NAMES]
    if unknown:
        raise ValueError(f"unknown table name(s): {unknown}; known: {TABLE_NAMES}")
    written = {}
    for n in names:
        if n == "table2":
            written[n] = table2(out, comparators=comparators, include_matched=include_matched,
                                comparator_variant=comparator_variant)
        elif n == "table3":
            written[n] = table3(out, ds_id=ds_id)
    return written


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (exposed so the runbook check can parse the documented commands)."""
    ap = argparse.ArgumentParser(description="plan11 EUSIPCO tables (builder 2, epoch 2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated: " + ",".join(TABLE_NAMES))
    ap.add_argument("--comparators", default=",".join(COMPARATORS_DEFAULT),
                    help="comma-separated p2.bib keys of the comparator rows (AA 2026-09-17 default; add law2010volatile for IFIP)")
    ap.add_argument("--include-matched", action="store_true", default=INCLUDE_MATCHED_DEFAULT,
                    help="append the `combined (matched)` row")
    ap.add_argument("--comparator-variant", default=COMPARATOR_VARIANT_DEFAULT, choices=list(COMPARATOR_VARIANTS),
                    help="which comparator row of table7_comparators.csv (layout B) enters Table 2")
    ap.add_argument("--ds-id", default=None, help="directory name of the Dhodapkar-Smith reading (default: resolved from disk)")
    return ap


def main(argv: list[str] | None = None) -> int:
    """The CLI of the module docstring: exit 0 on success, 2 on a missing `table7.csv` (its path on
    stderr) when Table 2 is requested, 1 on an internal error (SPEC 7.1's exit-code contract)."""
    a = build_parser().parse_args(argv)
    out = Path(a.out)
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None
    comps = tuple(s.strip() for s in a.comparators.split(",") if s.strip())
    if (only is None or "table2" in only) and not (_tables_dir(out) / TABLE7_FILE).exists():
        print(f"missing input: {_tables_dir(out) / TABLE7_FILE}", file=sys.stderr)
        return 2
    try:
        written = run(out, only, comparators=comps, include_matched=a.include_matched,
                      comparator_variant=a.comparator_variant, ds_id=a.ds_id)
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    for n, v in written.items():
        print(f"{n}: {v['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
